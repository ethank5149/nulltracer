// ============================================================
//  WEBGL RENDERER
//  WebGL initialization, shader compilation, program building,
//  resize handling, and render loop.
// ============================================================

import { buildFragSrc } from './shader-generator.js';

// ---- Module state ----
let gl = null;
let canvas = null;
let program = null;
let uniforms = {};
let aPos = -1;
let buf = null;
let fc = 0;
let lastFT = performance.now();
let needsRender = true;
let rafId = null;
let vertSrc = '';

// These are set by main.js via setters
let stateRef = null;
let onResizeCallback = null;

export function initWebGL(canvasEl, loadMsgEl, errMsgEl) {
    canvas = canvasEl;
    gl = canvas.getContext('webgl', {antialias:false, preserveDrawingBuffer:false, powerPreference:'high-performance'});
    if (!gl) {
        loadMsgEl.textContent = 'WebGL not available';
        errMsgEl.style.display = 'block';
        errMsgEl.textContent = 'Try Chrome/Firefox on desktop.';
        return null;
    }

    buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW);

    vertSrc = document.getElementById('vs').textContent;

    return gl;
}

export function getGL() { return gl; }
export function getCanvas() { return canvas; }

export function setStateRef(ref) { stateRef = ref; }

export function markDirty() {
    needsRender = true;
    if (!rafId) rafId = requestAnimationFrame(render);
}

function compile(src, type) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) return {s:null, e:gl.getShaderInfoLog(s)};
    return {s, e:null};
}

export function buildProgram() {
    if (program) gl.deleteProgram(program);
    const fragSrc = buildFragSrc({
        method: stateRef.qMethod,
        steps: stateRef.qSteps,
        obsDist: stateRef.qObsDist,
        starLayers: stateRef.qStarLayers,
        stepSize: stateRef.qStepSize,
        bgMode: stateRef.bgMode
    });
    const vr = compile(vertSrc, gl.VERTEX_SHADER);
    if (!vr.s) return {ok:false, error:'Vertex: '+vr.e};
    const fr = compile(fragSrc, gl.FRAGMENT_SHADER);
    if (!fr.s) return {ok:false, error:'Fragment: '+fr.e};
    const p = gl.createProgram();
    gl.attachShader(p, vr.s); gl.attachShader(p, fr.s); gl.linkProgram(p);
    gl.deleteShader(vr.s); gl.deleteShader(fr.s);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) { gl.deleteProgram(p); return {ok:false, error:'Link: '+gl.getProgramInfoLog(p)}; }
    program = p; gl.useProgram(program);
    aPos = gl.getAttribLocation(program, 'a_pos');
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    uniforms = {};
    ['u_res','u_a','u_incl','u_fov','u_disk','u_grid','u_temp','u_phi0','u_Q','u_isco'].forEach(n => { uniforms[n]=gl.getUniformLocation(program,n); });
    document.getElementById('integ-info').textContent = {yoshida4:'Yoshida4 symplectic',rk4:'RK4',yoshida6:'Yoshida6 symplectic',yoshida8:'Yoshida8 symplectic',rkdp8:'Dormand-Prince RK8'}[stateRef.qMethod] || stateRef.qMethod;
    return {ok:true};
}

export function resize() {
    const w = Math.floor(innerWidth*stateRef.qResScale), h = Math.floor(innerHeight*stateRef.qResScale);
    canvas.width=w; canvas.height=h; gl.viewport(0,0,w,h);
    document.getElementById('res').textContent = w+'×'+h;
    markDirty();
}

function render() {
    rafId = null;
    if (stateRef.autoRotate) {
        stateRef.rotAngle += 0.008;
        needsRender = true;
    }

    // Skip expensive GPU work in server-only mode (server provides the frames)
    if (needsRender && stateRef.renderMode !== 'server') {
        gl.uniform2f(uniforms.u_res, canvas.width, canvas.height);
        gl.uniform1f(uniforms.u_a, stateRef.spin);
        gl.uniform1f(uniforms.u_incl, stateRef.incl*Math.PI/180);
        gl.uniform1f(uniforms.u_fov, stateRef.fov);
        gl.uniform1f(uniforms.u_disk, stateRef.showDisk);
        gl.uniform1f(uniforms.u_grid, stateRef.showGrid);
        gl.uniform1f(uniforms.u_temp, stateRef.diskTemp);
        gl.uniform1f(uniforms.u_phi0, stateRef.rotAngle);
        gl.uniform1f(uniforms.u_Q, stateRef.charge);
        gl.uniform1f(uniforms.u_isco, stateRef.getIsco(stateRef.spin, stateRef.charge));
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        needsRender = false;
    }

    fc++; const now=performance.now();
    if (now-lastFT > 800) { document.getElementById('fps').textContent=Math.round(fc/((now-lastFT)*0.001)); fc=0; lastFT=now; }

    // Keep the loop alive only when animating
    if (stateRef.autoRotate) rafId = requestAnimationFrame(render);
}
