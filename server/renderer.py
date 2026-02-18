"""
Headless OpenGL renderer using EGL for offscreen rendering.

Creates an EGL context with a pbuffer surface, compiles GLSL shaders,
renders to an FBO, and reads back pixels as raw RGB bytes.
"""

import ctypes
import logging
import math
from typing import Optional

import numpy as np
from OpenGL import EGL
from OpenGL.GL import *

from .isco import isco
from .shader import VERTEX_SHADER_SRC, build_frag_src

logger = logging.getLogger(__name__)

# Cache key type: tuple of (method, steps, obsDist, starLayers, stepSize, bgMode)
_CacheKey = tuple


class Renderer:
    """Headless EGL/OpenGL renderer for black hole ray tracing."""

    def __init__(self):
        self._dpy = None
        self._ctx = None
        self._surface = None
        self._fbo: Optional[int] = None
        self._rbo_color: Optional[int] = None
        self._vao: Optional[int] = None
        self._vbo: Optional[int] = None
        self._program_cache: dict[_CacheKey, int] = {}
        self._current_fbo_size: tuple[int, int] = (0, 0)
        self._gpu_info: str = "unknown"
        self._initialized = False

    def initialize(self) -> None:
        """Initialize EGL display, context, and surface."""
        if self._initialized:
            return

        # Get default display
        dpy = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        if dpy == EGL.EGL_NO_DISPLAY:
            raise RuntimeError("Failed to get EGL display")

        # Initialize EGL
        major = ctypes.c_long()
        minor = ctypes.c_long()
        if not EGL.eglInitialize(dpy, major, minor):
            raise RuntimeError("Failed to initialize EGL")
        logger.info("EGL initialized: %d.%d", major.value, minor.value)

        # Configure for offscreen rendering
        config_attribs = (EGL.EGLint * 13)(
            EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
            EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
            EGL.EGL_RED_SIZE, 8,
            EGL.EGL_GREEN_SIZE, 8,
            EGL.EGL_BLUE_SIZE, 8,
            EGL.EGL_NONE,
        )

        config = EGL.EGLConfig()
        num_configs = ctypes.c_long()
        if not EGL.eglChooseConfig(dpy, config_attribs, config, 1, num_configs):
            raise RuntimeError("Failed to choose EGL config")
        if num_configs.value == 0:
            raise RuntimeError("No suitable EGL config found")

        # Bind OpenGL API (not OpenGL ES)
        if not EGL.eglBindAPI(EGL.EGL_OPENGL_API):
            raise RuntimeError("Failed to bind OpenGL API")

        # Create context
        ctx = EGL.eglCreateContext(dpy, config, EGL.EGL_NO_CONTEXT, None)
        if ctx == EGL.EGL_NO_CONTEXT:
            raise RuntimeError("Failed to create EGL context")

        # Try surfaceless context first (works on NVIDIA EGL in containers),
        # fall back to pbuffer surface if surfaceless fails.
        surface = EGL.EGL_NO_SURFACE
        if EGL.eglMakeCurrent(dpy, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, ctx):
            logger.info("Using surfaceless EGL context (rendering to FBO)")
        else:
            logger.info("Surfaceless context not supported, trying pbuffer surface")
            pbuffer_attribs = (EGL.EGLint * 5)(
                EGL.EGL_WIDTH, 1,
                EGL.EGL_HEIGHT, 1,
                EGL.EGL_NONE,
            )
            surface = EGL.eglCreatePbufferSurface(dpy, config, pbuffer_attribs)
            if surface == EGL.EGL_NO_SURFACE:
                raise RuntimeError("Failed to create EGL pbuffer surface")
            if not EGL.eglMakeCurrent(dpy, surface, surface, ctx):
                raise RuntimeError("Failed to make EGL context current")

        self._dpy = dpy
        self._ctx = ctx
        self._surface = surface

        # Query GPU info
        vendor = glGetString(GL_VENDOR)
        renderer = glGetString(GL_RENDERER)
        version = glGetString(GL_VERSION)
        glsl_version = glGetString(GL_SHADING_LANGUAGE_VERSION)
        if vendor and renderer:
            self._gpu_info = f"{vendor.decode()} {renderer.decode()} ({version.decode() if version else 'unknown'})"
        else:
            self._gpu_info = "EGL headless (info unavailable)"
        logger.info("GPU: %s", self._gpu_info)
        logger.info("GLSL version: %s", glsl_version.decode() if glsl_version else "unknown")
        logger.info("OpenGL version: %s", version.decode() if version else "unknown")

        # Set up fullscreen quad VBO
        self._setup_quad()

        # Create initial FBO
        self._fbo = glGenFramebuffers(1)
        self._rbo_color = glGenRenderbuffers(1)

        self._initialized = True
        logger.info("Renderer initialized successfully")

    @property
    def gpu_info(self) -> str:
        """Return GPU information string."""
        return self._gpu_info

    def _make_current(self) -> None:
        """Make the EGL context current on the calling thread.

        OpenGL contexts are thread-local. Since render_frame() runs in a
        thread pool via asyncio.to_thread(), we must call eglMakeCurrent()
        on each render thread before any GL calls.
        """
        if not EGL.eglMakeCurrent(self._dpy, self._surface, self._surface, self._ctx):
            raise RuntimeError("Failed to make EGL context current on render thread")

    def _setup_quad(self) -> None:
        """Set up a fullscreen quad (2 triangles covering clip space [-1,1])."""
        # Vertex data: 2 triangles forming a quad
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
            -1.0,  1.0,
             1.0, -1.0,
             1.0,  1.0,
        ], dtype=np.float32)

        self._vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _ensure_fbo_size(self, width: int, height: int) -> None:
        """Resize the FBO renderbuffer if needed."""
        if self._current_fbo_size == (width, height):
            return

        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glBindRenderbuffer(GL_RENDERBUFFER, self._rbo_color)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB8, width, height)
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_RENDERBUFFER, self._rbo_color
        )

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer incomplete: {status}")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self._current_fbo_size = (width, height)
        logger.debug("FBO resized to %dx%d", width, height)

    def _compile_shader(self, source: str, shader_type: int) -> int:
        """Compile a single shader and return its handle."""
        shader_type_name = "vertex" if shader_type == GL_VERTEX_SHADER else "fragment"
        shader = glCreateShader(shader_type)
        if shader == 0:
            raise RuntimeError(f"glCreateShader({shader_type_name}) returned 0")

        glShaderSource(shader, source)
        glCompileShader(shader)

        status = glGetShaderiv(shader, GL_COMPILE_STATUS)
        info_log = glGetShaderInfoLog(shader)
        info_str = info_log.decode() if isinstance(info_log, bytes) else info_log

        if info_str:
            logger.info("%s shader info log: %s", shader_type_name, info_str.strip())

        if not status:
            # Dump full source for debugging
            logger.error("Full %s shader source:\n%s", shader_type_name, source)
            glDeleteShader(shader)
            raise RuntimeError(f"{shader_type_name} shader compilation failed:\n{info_str}")

        logger.info("%s shader compiled successfully (handle=%d)", shader_type_name, shader)
        return shader

    def _get_program(self, opts: dict) -> int:
        """Get or compile+cache a shader program for the given options."""
        cache_key = (
            opts.get("method", "separated"),
            opts.get("steps", 200),
            opts.get("obsDist", 40),
            opts.get("starLayers", 3),
            opts.get("stepSize", 0.30),
            opts.get("bgMode", 1),
        )

        if cache_key in self._program_cache:
            return self._program_cache[cache_key]

        logger.info("Compiling shader program for key: %s", cache_key)

        # Clear any pending GL errors before shader compilation
        while glGetError() != GL_NO_ERROR:
            pass

        frag_src = build_frag_src(opts)

        # Log shader source length for diagnostics
        logger.info("Fragment shader source length: %d chars", len(frag_src))

        vert_shader = self._compile_shader(VERTEX_SHADER_SRC, GL_VERTEX_SHADER)
        frag_shader = self._compile_shader(frag_src, GL_FRAGMENT_SHADER)

        program = glCreateProgram()
        if program == 0:
            raise RuntimeError("glCreateProgram() returned 0 — GL context may be lost")

        glAttachShader(program, vert_shader)
        err_attach1 = glGetError()
        glAttachShader(program, frag_shader)
        err_attach2 = glGetError()
        if err_attach1 != GL_NO_ERROR or err_attach2 != GL_NO_ERROR:
            logger.error("glAttachShader errors: vert=%s, frag=%s", err_attach1, err_attach2)

        glLinkProgram(program)

        gl_err = glGetError()
        link_status = glGetProgramiv(program, GL_LINK_STATUS)
        if not link_status:
            log = glGetProgramInfoLog(program)
            log_str = log.decode() if isinstance(log, bytes) else log
            # Also capture individual shader info logs for diagnostics
            vert_log = glGetShaderInfoLog(vert_shader)
            frag_log = glGetShaderInfoLog(frag_shader)
            vert_log_str = vert_log.decode() if isinstance(vert_log, bytes) else vert_log
            frag_log_str = frag_log.decode() if isinstance(frag_log, bytes) else frag_log
            logger.error("Shader link failed. Link status: %s, glGetError: %s", link_status, gl_err)
            logger.error("Link log: [%s]", log_str)
            logger.error("Vertex shader info log: [%s]", vert_log_str)
            logger.error("Fragment shader info log: [%s]", frag_log_str)
            logger.error("Vertex shader source:\n%s", VERTEX_SHADER_SRC)
            logger.error("Fragment shader source (first 2000 chars):\n%s", frag_src[:2000])
            # Also try to validate shaders individually
            vert_status = glGetShaderiv(vert_shader, GL_COMPILE_STATUS)
            frag_status = glGetShaderiv(frag_shader, GL_COMPILE_STATUS)
            logger.error("Vert compile status: %s, Frag compile status: %s", vert_status, frag_status)
            glDeleteProgram(program)
            glDeleteShader(vert_shader)
            glDeleteShader(frag_shader)
            raise RuntimeError(
                f"Shader program linking failed:\n"
                f"Link: {log_str}\nVert: {vert_log_str}\nFrag: {frag_log_str}\n"
                f"glError: {gl_err}, vert_compiled: {vert_status}, frag_compiled: {frag_status}"
            )

        # Shaders can be detached and deleted after linking
        glDetachShader(program, vert_shader)
        glDetachShader(program, frag_shader)
        glDeleteShader(vert_shader)
        glDeleteShader(frag_shader)

        self._program_cache[cache_key] = program
        return program

    def render_frame(self, params: dict) -> bytes:
        """Render a single frame and return raw RGB pixel data.

        Args:
            params: dict with all render parameters including:
                spin, charge, inclination (degrees), fov, width, height,
                method, steps, step_size, obs_dist, bg_mode, show_disk,
                show_grid, disk_temp, star_layers, phi0

        Returns:
            Raw RGB bytes (width * height * 3), top-to-bottom row order.
        """
        if not self._initialized:
            raise RuntimeError("Renderer not initialized. Call initialize() first.")

        width = params.get("width", 1280)
        height = params.get("height", 720)

        # Build shader options (these affect #defines, requiring recompilation)
        shader_opts = {
            "method": params.get("method", "separated"),
            "steps": params.get("steps", 200),
            "obsDist": params.get("obs_dist", 40),
            "starLayers": params.get("star_layers", 3),
            "stepSize": params.get("step_size", 0.30),
            "bgMode": params.get("bg_mode", 1),
        }

        program = self._get_program(shader_opts)

        # Ensure FBO is the right size
        self._ensure_fbo_size(width, height)

        # Bind FBO and set viewport
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glViewport(0, 0, width, height)

        # Use shader program
        glUseProgram(program)

        # Compute ISCO
        spin = params.get("spin", 0.6)
        charge = params.get("charge", 0.0)
        isco_radius = isco(spin, charge)

        # Set uniforms
        incl_rad = math.radians(params.get("inclination", 80.0))

        loc = lambda name: glGetUniformLocation(program, name)
        glUniform2f(loc("u_res"), float(width), float(height))
        glUniform1f(loc("u_a"), float(spin))
        glUniform1f(loc("u_incl"), float(incl_rad))
        glUniform1f(loc("u_fov"), float(params.get("fov", 8.0)))
        glUniform1f(loc("u_disk"), 1.0 if params.get("show_disk", True) else 0.0)
        glUniform1f(loc("u_grid"), 1.0 if params.get("show_grid", True) else 0.0)
        glUniform1f(loc("u_temp"), float(params.get("disk_temp", 1.0)))
        glUniform1f(loc("u_phi0"), float(params.get("phi0", 0.0)))
        glUniform1f(loc("u_Q"), float(charge))
        glUniform1f(loc("u_isco"), float(isco_radius))

        # Bind VBO and set vertex attribute
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        a_pos_loc = glGetAttribLocation(program, "a_pos")
        glEnableVertexAttribArray(a_pos_loc)
        glVertexAttribPointer(a_pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)

        # Clear and draw
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        # Read pixels
        glFinish()
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        pixel_array = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)

        # Flip vertically (OpenGL origin is bottom-left)
        pixel_array = np.flipud(pixel_array)

        # Clean up state
        glDisableVertexAttribArray(a_pos_loc)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        return pixel_array.tobytes()

    def shutdown(self) -> None:
        """Clean up OpenGL and EGL resources."""
        if not self._initialized:
            return

        # Delete cached programs
        for program in self._program_cache.values():
            glDeleteProgram(program)
        self._program_cache.clear()

        # Delete FBO and RBO
        if self._fbo is not None:
            glDeleteFramebuffers(1, [self._fbo])
        if self._rbo_color is not None:
            glDeleteRenderbuffers(1, [self._rbo_color])
        if self._vbo is not None:
            glDeleteBuffers(1, [self._vbo])

        # Tear down EGL
        if self._dpy is not None:
            EGL.eglMakeCurrent(
                self._dpy, EGL.EGL_NO_SURFACE,
                EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT
            )
            if self._surface is not None and self._surface != EGL.EGL_NO_SURFACE:
                EGL.eglDestroySurface(self._dpy, self._surface)
            if self._ctx is not None:
                EGL.eglDestroyContext(self._dpy, self._ctx)
            EGL.eglTerminate(self._dpy)

        self._initialized = False
        logger.info("Renderer shut down")
