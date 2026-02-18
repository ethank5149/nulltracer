// ============================================================
//  SERVER CLIENT
//  Hybrid server rendering: fetch logic, mode switching,
//  mobile detection, and auto-configuration.
// ============================================================

import { markDirty } from './webgl-renderer.js';

// ---- Module state ----
let serverUrl = '';
let renderMode = 'local';  // 'local' | 'hybrid' | 'server'
let serverQuality = 720;
let serverAbort = null;
let serverDebounce = null;
let serverFailCount = 0;
const MAX_SERVER_FAILS = 3;

let serverFrame = null;
let serverDot = null;
let serverStatusEl = null;
let canvas = null;
let stateRef = null;

export function initServerClient(opts) {
    serverFrame = opts.serverFrame;
    serverDot = opts.serverDot;
    serverStatusEl = opts.serverStatusEl;
    canvas = opts.canvas;
    stateRef = opts.stateRef;
}

export function getServerUrl() { return serverUrl; }
export function setServerUrl(url) { serverUrl = url; }
export function getRenderMode() { return renderMode; }
export function setRenderMode(mode) { renderMode = mode; }
export function getServerQuality() { return serverQuality; }
export function setServerQuality(q) { serverQuality = q; }
export function resetServerFailCount() { serverFailCount = 0; }

// Detect if we're on a mobile/low-power device
export function detectMobile() {
    const isMobile = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
    const canvas2 = document.createElement('canvas');
    const gl2 = canvas2.getContext('webgl');
    let isLowGPU = false;
    if (gl2) {
        const dbg = gl2.getExtension('WEBGL_debug_renderer_info');
        if (dbg) {
            const renderer = gl2.getParameter(dbg.UNMASKED_RENDERER_WEBGL).toLowerCase();
            isLowGPU = /mali|adreno|powervr|apple gpu|intel.*hd|intel.*uhd/i.test(renderer);
        }
    }
    return isMobile || isLowGPU;
}

export function setServerStatus(status, text) {
    serverDot.className = 'server-dot ' + status;
    serverStatusEl.textContent = text || status;
}

export async function checkServerHealth() {
    if (!serverUrl) { setServerStatus('disconnected', 'server off'); return false; }
    try {
        const resp = await fetch(serverUrl + '/health', { signal: AbortSignal.timeout(3000) });
        if (resp.ok) {
            const data = await resp.json();
            setServerStatus('connected', 'server ok');
            serverFailCount = 0;
            return true;
        }
    } catch (e) {}
    setServerStatus('disconnected', 'server unreachable');
    return false;
}

function buildServerParams() {
    const aspect = innerWidth / innerHeight;
    let w, h;
    if (aspect >= 1) {
        h = serverQuality;
        w = Math.round(h * aspect);
    } else {
        w = serverQuality;
        h = Math.round(w / aspect);
    }
    return {
        spin: stateRef.spin,
        charge: stateRef.charge,
        inclination: stateRef.incl,
        fov: stateRef.fov,
        width: w,
        height: h,
        method: stateRef.qMethod,
        steps: stateRef.qSteps,
        step_size: stateRef.qStepSize,
        obs_dist: stateRef.qObsDist,
        bg_mode: stateRef.bgMode,
        show_disk: stateRef.showDisk > 0.5,
        show_grid: stateRef.showGrid > 0.5,
        disk_temp: stateRef.diskTemp,
        star_layers: stateRef.qStarLayers,
        phi0: stateRef.rotAngle,
        format: 'jpeg',
        quality: 85
    };
}

async function requestServerFrame() {
    if (!serverUrl || renderMode === 'local') return;
    if (serverFailCount >= MAX_SERVER_FAILS) {
        setServerStatus('disconnected', 'server failed');
        return;
    }

    // Cancel any in-flight request
    if (serverAbort) { serverAbort.abort(); }
    serverAbort = new AbortController();

    setServerStatus('loading', 'rendering…');

    try {
        const params = buildServerParams();
        const resp = await fetch(serverUrl + '/render', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
            signal: serverAbort.signal
        });

        if (!resp.ok) throw new Error('Server returned ' + resp.status);

        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);

        // Crossfade the server frame in
        serverFrame.onload = function() {
            serverFrame.classList.add('visible');
            const cache = resp.headers.get('X-Cache') || '?';
            const ms = resp.headers.get('X-Render-Time-Ms') || '?';
            setServerStatus('connected', cache + ' ' + ms + 'ms');
            URL.revokeObjectURL(url);
        };
        serverFrame.src = url;
        serverFailCount = 0;

    } catch (e) {
        if (e.name === 'AbortError') return;  // Expected on cancel
        serverFailCount++;
        console.warn('Server render failed:', e.message, '(' + serverFailCount + '/' + MAX_SERVER_FAILS + ')');
        if (serverFailCount >= MAX_SERVER_FAILS) {
            setServerStatus('disconnected', 'server failed — using local');
            serverFrame.classList.remove('visible');
        } else {
            setServerStatus('disconnected', 'retry…');
        }
    }
}

export function scheduleServerRender() {
    if (renderMode === 'local' || !serverUrl) return;
    // Keep the old server frame visible until the new one arrives (no grey gap)
    clearTimeout(serverDebounce);
    serverDebounce = setTimeout(requestServerFrame, 200);
}

// In server-only mode, hide the canvas and only show server frames
export function updateRenderMode(mode) {
    renderMode = mode;
    document.getElementById('btn-local-only').classList.toggle('active', mode === 'local');
    document.getElementById('btn-hybrid').classList.toggle('active', mode === 'hybrid');
    document.getElementById('btn-server-only').classList.toggle('active', mode === 'server');

    // Update stateRef so renderer knows the mode
    stateRef.renderMode = mode;

    if (mode === 'local') {
        serverFrame.classList.remove('visible');
        canvas.style.opacity = '1';
        markDirty();
    } else if (mode === 'hybrid') {
        canvas.style.opacity = '1';
        markDirty();
        scheduleServerRender();
    } else if (mode === 'server') {
        canvas.style.opacity = '0.15';  // Dim local, rely on server
        scheduleServerRender();
    }
}

export function autoDetectServer() {
    // Auto-detect same-origin server (when served via Caddy/reverse proxy)
    if (location.protocol !== 'file:') {
        fetch('/health', { signal: AbortSignal.timeout(2000) })
            .then(r => r.ok ? r.json() : Promise.reject())
            .then(data => {
                serverUrl = location.origin;
                document.getElementById('server-url').value = location.origin;
                setServerStatus('connected', 'server ok');
                serverFailCount = 0;
                // Default to server-only mode when server is available
                updateRenderMode('server');
            })
            .catch(() => {});  // Server not available at same origin, that's fine
    }
}
