FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install Python and OpenGL libs (OpenGL kept temporarily for A/B validation)
# libopengl0   — provides libOpenGL.so.0 (GLVND OpenGL dispatch, needed by PyOpenGL)
# libglu1-mesa — provides libGLU.so.0
# libegl1      — provides libEGL.so.1 (GLVND EGL dispatch)
# libgl1       — provides libGL.so.1 (legacy GL support)
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libegl1 libgl1 libopengl0 libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Register NVIDIA as an EGL vendor so eglInitialize uses the GPU
# (the NVIDIA runtime injects libEGL_nvidia.so.0 but doesn't always
# provide the ICD JSON inside the container)
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN mkdir -p server
COPY . server/

# Tell PyOpenGL to use EGL for context management (not GLX/GLUT).
# Without this, PyOpenGL's contextdata module can't find the EGL context
# and raises "Attempt to retrieve context when no valid context".
ENV PYOPENGL_PLATFORM=egl

# Request all NVIDIA driver capabilities so the container runtime injects
# EGL/OpenGL libraries (libEGL_nvidia.so.0, libGLX_nvidia.so.0, etc.)
# in addition to the default compute+utility libs.
# Without this, the CUDA base image only gets compute,utility and EGL fails.
ENV NVIDIA_DRIVER_CAPABILITIES=all

EXPOSE 8420
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8420", "--workers", "1"]
