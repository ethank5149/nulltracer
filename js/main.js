// ============================================================
//  MAIN ENTRY POINT
//  Imports server-client and UI modules, wires them together.
//  No local WebGL rendering — server is the sole renderer.
// ============================================================

import { initUI, refreshLabels } from './ui-controller.js';
import { initServerClient, autoDetectServer, scheduleServerRender } from './server-client.js';
import { initWsClient, setFallbackRender, connectWebSocket } from './ws-client.js';

// ============================================================
//  SHARED APPLICATION STATE
//  All modules reference this object for shared mutable state.
// ============================================================
const state = {
    spin: 0,
    charge: 0,
    incl: 89,
    diskTemp: 1,
    showDisk: 1,
    showGrid: 1,
    autoRotate: false,
    rotAngle: 0,
    fov: 8,
    bgMode: 1,
    qMethod: 'rkdp8',
    qSteps: 200,
    qResScale: 1.0,
    qStepSize: 0.3,
    qObsDist: 40,
    qStarLayers: 3,
    renderMode: 'server',
};

// ============================================================
//  INITIALIZATION
// ============================================================
const loadEl = document.getElementById('loading');
const loadMsg = document.getElementById('loading-msg');

loadMsg.textContent = 'Connecting to render server…';

// Initialize server client
initServerClient({
    serverFrame: document.getElementById('server-frame'),
    serverDot: document.getElementById('server-dot'),
    serverStatusEl: document.getElementById('server-status'),
    stateRef: state,
    onFirstFrame() {
        // Hide loading overlay once the first server frame arrives
        loadEl.classList.add('hidden');
    }
});

// Initialize WebSocket client with same DOM refs and state
initWsClient({
    serverFrame: document.getElementById('server-frame'),
    serverDot: document.getElementById('server-dot'),
    serverStatusEl: document.getElementById('server-status'),
    stateRef: state,
    onFirstFrame: () => {
        document.getElementById('loading')?.classList.add('hidden');
    }
});

// Set HTTP fallback for when WebSocket is not connected
setFallbackRender(scheduleServerRender);

// Initialize UI (event handlers, sliders, etc.)
initUI(state);

// Populate initial label values
refreshLabels();

// Auto-detect same-origin server and begin rendering
autoDetectServer();
connectWebSocket(location.origin);
