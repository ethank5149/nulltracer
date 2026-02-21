// ============================================================
//  SERVER CLIENT
//  Server-only rendering: fetch logic, auto-configuration,
//  mobile detection, and health checks.
//  No local/hybrid modes — server is the sole renderer.
// ============================================================

// ---- Module state ----
let serverUrl = '';
let serverQuality = 720;
let serverAbort = null;
let serverDebounce = null;
let serverFailCount = 0;
const MAX_SERVER_FAILS = 3;

let serverFrame = null;
let serverDot = null;
let serverStatusEl = null;
let stateRef = null;
let onFirstFrameCb = null;
let firstFrameReceived = false;

export function initServerClient(opts) {
    serverFrame = opts.serverFrame;
    serverDot = opts.serverDot;
    serverStatusEl = opts.serverStatusEl;
    stateRef = opts.stateRef;
    onFirstFrameCb = opts.onFirstFrame || null;
}

export function getServerUrl() { return serverUrl; }
export function setServerUrl(url) { serverUrl = url; }
export function getServerQuality() { return serverQuality; }
export function setServerQuality(q) { serverQuality = q; }
export function resetServerFailCount() { serverFailCount = 0; }

// Detect if we're on a mobile/low-power device
export function detectMobile() {
    return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
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
        srgb_output: !!stateRef.srgbOutput,
        disk_alpha: stateRef.diskAlpha,
        disk_max_crossings: stateRef.diskMaxCrossings,
        bloom_enabled: !!stateRef.bloomEnabled,
        bloom_radius: stateRef.bloomRadius,
        format: 'jpeg',
        quality: 85
    };
}

async function requestServerFrame() {
    if (!serverUrl) return;
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

        // Show the server frame
        serverFrame.onload = function() {
            serverFrame.classList.add('visible');
            const cache = resp.headers.get('X-Cache') || '?';
            const ms = resp.headers.get('X-Render-Time-Ms') || '?';
            setServerStatus('connected', cache + ' ' + ms + 'ms');
            URL.revokeObjectURL(url);

            // Fire first-frame callback once
            if (!firstFrameReceived && onFirstFrameCb) {
                firstFrameReceived = true;
                onFirstFrameCb();
            }
        };
        serverFrame.src = url;
        serverFailCount = 0;

    } catch (e) {
        if (e.name === 'AbortError') return;  // Expected on cancel
        serverFailCount++;
        console.warn('Server render failed:', e.message, '(' + serverFailCount + '/' + MAX_SERVER_FAILS + ')');
        if (serverFailCount >= MAX_SERVER_FAILS) {
            setServerStatus('disconnected', 'server failed');
        } else {
            setServerStatus('disconnected', 'retry…');
        }
    }
}

export function scheduleServerRender() {
    if (!serverUrl) return;
    // Keep the old server frame visible until the new one arrives (no grey gap)
    clearTimeout(serverDebounce);
    serverDebounce = setTimeout(requestServerFrame, 200);
}

export function autoDetectServer() {
    // Auto-detect same-origin server (when served via Caddy/reverse proxy)
    if (location.protocol !== 'file:') {
        fetch('/health', { signal: AbortSignal.timeout(2000) })
            .then(r => r.ok ? r.json() : Promise.reject())
            .then(data => {
                serverUrl = location.origin;
                const urlInput = document.getElementById('server-url');
                if (urlInput) urlInput.value = location.origin;
                setServerStatus('connected', 'server ok');
                serverFailCount = 0;
                // Immediately request first frame
                scheduleServerRender();
            })
            .catch(() => {
                setServerStatus('disconnected', 'no server found');
            });
    }
}

// ── Scene API helpers ───────────────────────────────────────

export async function fetchScenes() {
    const resp = await fetch(`${serverUrl}/scenes`);
    if (!resp.ok) throw new Error('Failed to fetch scenes');
    return resp.json();
}

export async function fetchScene(name) {
    const resp = await fetch(`${serverUrl}/scenes/${encodeURIComponent(name)}`);
    if (!resp.ok) throw new Error(`Scene '${name}' not found`);
    return resp.json();
}

export async function saveSceneAPI(name, params) {
    const resp = await fetch(`${serverUrl}/scenes/${encodeURIComponent(name)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
    });
    if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data.detail || 'Failed to save scene');
    }
    return resp.json();
}

export async function deleteSceneAPI(name) {
    const resp = await fetch(`${serverUrl}/scenes/${encodeURIComponent(name)}`, {
        method: 'DELETE',
    });
    if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data.detail || 'Failed to delete scene');
    }
    return resp.json();
}
