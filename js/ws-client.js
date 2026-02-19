// ============================================================
//  WEBSOCKET STREAMING CLIENT
//  Replaces HTTP-based server-client.js as the primary
//  transport for interactive rendering via ws /stream.
//  Binary protocol: 8-byte header + JPEG/WebP image bytes.
// ============================================================

// ---- Module state ----
let ws = null;                    // WebSocket instance
let wsUrl = null;                 // WebSocket URL (derived from server URL)
let serverHttpUrl = null;         // Original HTTP server URL (for reconnection)
let stateRef = null;              // Shared state object reference
let serverFrame = null;           // <img> element for frame display
let serverDot = null;             // Status dot element
let serverStatusEl = null;        // Status text element
let onFirstFrameCb = null;        // Callback for first frame received
let firstFrameReceived = false;   // Track first frame
let reconnectTimer = null;        // Reconnection timer
let reconnectAttempts = 0;        // Current reconnect attempt count
let reconnectDelay = 500;         // Starting delay (ms)
const MAX_RECONNECT_DELAY = 10000;  // Max delay (ms)
const MAX_RECONNECT_ATTEMPTS = 20;  // Give up after this many
let pendingParams = null;         // Latest params waiting to be sent
let renderInFlight = false;       // Whether we're waiting for a frame response
let prevBlobUrl = null;           // Previous blob URL to revoke
let fallbackRenderFn = null;      // HTTP fallback render function

// ---- Exported: initialization ----

/**
 * Initialize the WebSocket client module.
 * @param {Object} opts
 * @param {HTMLImageElement} opts.serverFrame  - <img> element for frame display
 * @param {HTMLElement}      opts.serverDot    - Status dot element
 * @param {HTMLElement}      opts.serverStatusEl - Status text element
 * @param {Object}           opts.stateRef     - Shared state object reference
 * @param {Function}         [opts.onFirstFrame] - Callback for first frame received
 */
export function initWsClient({ serverFrame: frame, serverDot: dot, serverStatusEl: statusEl, stateRef: state, onFirstFrame }) {
    serverFrame = frame;
    serverDot = dot;
    serverStatusEl = statusEl;
    stateRef = state;
    onFirstFrameCb = onFirstFrame || null;
}

// ---- Exported: set HTTP fallback ----

/**
 * Register a fallback render function (e.g. scheduleServerRender from server-client.js).
 * Called when WebSocket is not connected and params need to be rendered.
 * @param {Function} fn - Fallback render callback
 */
export function setFallbackRender(fn) {
    fallbackRenderFn = fn;
}

// ---- Exported: connection management ----

/**
 * Establish a WebSocket connection to the server's /stream endpoint.
 * Derives the WS URL from the given HTTP server URL.
 * @param {string} httpUrl - HTTP server URL (e.g. "http://localhost:8000")
 * @returns {WebSocket} The WebSocket instance
 */
export function connectWebSocket(httpUrl) {
    // Store HTTP URL for reconnection
    serverHttpUrl = httpUrl;

    // Derive WebSocket URL from HTTP URL
    wsUrl = httpUrl
        .replace(/^https:\/\//, 'wss://')
        .replace(/^http:\/\//, 'ws://')
        .replace(/\/$/, '') + '/stream';

    // Close existing connection if any
    if (ws) {
        try { ws.close(); } catch (_) { /* ignore */ }
        ws = null;
    }

    console.log('[WS] Connecting to', wsUrl);
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = function () {
        console.log('[WS] Connected');
        reconnectAttempts = 0;
        setWsStatus('connected', 'ws connected');

        // Send any params that were queued while disconnected
        if (pendingParams) {
            const p = pendingParams;
            pendingParams = null;
            ws.send(JSON.stringify(p));
            renderInFlight = true;
            setWsStatus('loading', 'rendering…');
        }

        // If no frame has been received yet, request one
        if (!firstFrameReceived && stateRef) {
            const params = buildParams();
            if (params) {
                ws.send(JSON.stringify(params));
                renderInFlight = true;
                setWsStatus('loading', 'rendering…');
            }
        }
    };

    ws.onclose = function (event) {
        renderInFlight = false;
        console.log('[WS] Closed (code=' + event.code + ', reason=' + (event.reason || 'none') + ')');
        setWsStatus('disconnected', 'ws disconnected');
        ws = null;
        scheduleReconnect();
    };

    ws.onerror = function (event) {
        console.error('[WS] Error:', event);
    };

    ws.onmessage = handleMessage;

    return ws;
}

/**
 * Cleanly disconnect the WebSocket.
 */
export function disconnectWebSocket() {
    // Cancel any pending reconnection
    if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }
    reconnectAttempts = 0;

    // Close the socket
    if (ws) {
        try { ws.close(); } catch (_) { /* ignore */ }
        ws = null;
    }
}

// ---- Exported: send parameters ----

/**
 * Build current render parameters from stateRef and send them over WebSocket.
 * Falls back to HTTP rendering if WebSocket is not connected.
 */
export function sendParams() {
    const params = buildParams();

    if (!ws || ws.readyState !== WebSocket.OPEN) {
        // Store for when connection opens
        pendingParams = params;

        // Fall back to HTTP render
        if (fallbackRenderFn) {
            fallbackRenderFn();
        }
        return;
    }

    // WebSocket is open — send immediately
    if (renderInFlight) {
        // A render is already in progress; queue the latest params
        // The message handler will send them when the current frame arrives
        pendingParams = params;
        return;
    }

    setWsStatus('loading', 'rendering…');
    ws.send(JSON.stringify(params));
    renderInFlight = true;
}

// ---- Exported: status query ----

/**
 * Returns true if the WebSocket is open and ready.
 * @returns {boolean}
 */
export function isWsConnected() {
    return ws !== null && ws.readyState === WebSocket.OPEN;
}

// ---- Exported: status display ----

/**
 * Update the status dot and text elements.
 * @param {'connected'|'disconnected'|'loading'} status - Status class name
 * @param {string} [text] - Status text to display
 */
export function setWsStatus(status, text) {
    if (serverDot) {
        serverDot.className = 'server-dot ' + status;
    }
    if (serverStatusEl) {
        serverStatusEl.textContent = text || status;
    }
}

// ---- Private: build render parameters ----

function buildParams() {
    const el = (id) => document.getElementById(id);
    const aspect = window.innerWidth / window.innerHeight;
    const quality = parseInt(el('srv-quality')?.value || '480', 10);
    const height = quality;
    const width = Math.round(height * aspect);

    return {
        spin:        stateRef.spin,
        charge:      stateRef.charge,
        inclination: stateRef.incl,
        fov:         stateRef.fov,
        width,
        height,
        method:      stateRef.qMethod,
        steps:       stateRef.qSteps,
        step_size:   stateRef.qStepSize,
        obs_dist:    stateRef.qObsDist,
        bg_mode:     stateRef.bgMode,
        show_disk:   !!stateRef.showDisk,
        show_grid:   !!stateRef.showGrid,
        disk_temp:   stateRef.diskTemp,
        star_layers: stateRef.qStarLayers,
        phi0:        stateRef.rotAngle || 0,
        format:      'jpeg',
        quality:     80
    };
}

// ---- Private: message handler ----

function handleMessage(event) {
    if (typeof event.data === 'string') {
        // JSON text message (error from server)
        try {
            const msg = JSON.parse(event.data);
            if (msg.error) {
                console.error('[WS] Server error:', msg.error);
                setWsStatus('disconnected', 'error: ' + msg.error);
            }
        } catch (e) {
            console.error('[WS] Unparseable text message:', event.data);
        }
        renderInFlight = false;
        return;
    }

    // Binary frame
    const buffer = event.data; // ArrayBuffer
    if (buffer.byteLength < 8) {
        console.error('[WS] Frame too small:', buffer.byteLength);
        renderInFlight = false;
        return;
    }

    // Parse 8-byte header: [width:u16le][height:u16le][format:u8][reserved:3×0x00]
    const header = new DataView(buffer, 0, 8);
    const width  = header.getUint16(0, true);  // little-endian
    const height = header.getUint16(2, true);
    const format = header.getUint8(4);          // 0 = JPEG, 1 = WebP

    // Extract image bytes (everything after the 8-byte header)
    const imageBytes = new Uint8Array(buffer, 8);
    const mimeType = format === 1 ? 'image/webp' : 'image/jpeg';
    const blob = new Blob([imageBytes], { type: mimeType });

    // Revoke previous blob URL to prevent memory leak
    if (prevBlobUrl) {
        URL.revokeObjectURL(prevBlobUrl);
    }

    const url = URL.createObjectURL(blob);
    prevBlobUrl = url;
    serverFrame.src = url;
    serverFrame.classList.add('visible');

    // Update status
    setWsStatus('connected', 'ws frame ' + width + '\u00d7' + height);
    renderInFlight = false;

    // Fire first-frame callback once
    if (!firstFrameReceived) {
        firstFrameReceived = true;
        if (onFirstFrameCb) onFirstFrameCb();
    }

    // If params changed while we were rendering, send the latest
    if (pendingParams) {
        const p = pendingParams;
        pendingParams = null;
        ws.send(JSON.stringify(p));
        renderInFlight = true;
        setWsStatus('loading', 'rendering…');
    }
}

// ---- Private: reconnection with exponential backoff ----

function scheduleReconnect() {
    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        setWsStatus('disconnected', 'connection lost');
        console.warn('[WS] Max reconnect attempts (' + MAX_RECONNECT_ATTEMPTS + ') reached, giving up');
        return;
    }

    const delay = Math.min(
        reconnectDelay * Math.pow(1.5, reconnectAttempts),
        MAX_RECONNECT_DELAY
    );
    reconnectAttempts++;

    setWsStatus('disconnected', 'reconnecting in ' + Math.round(delay / 1000) + 's\u2026');
    console.log('[WS] Reconnect attempt ' + reconnectAttempts + ' in ' + Math.round(delay) + 'ms');

    reconnectTimer = setTimeout(() => {
        if (serverHttpUrl) {
            connectWebSocket(serverHttpUrl);
        }
    }, delay);
}
