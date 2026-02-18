// ============================================================
//  MAIN ENTRY POINT
//  Imports all modules and wires them together.
// ============================================================

import { iscoJS } from './isco-calculator.js';
import { initWebGL, setStateRef, markDirty, buildProgram, resize } from './webgl-renderer.js';
import { initUI, updInfo, refreshLabels, recompile, applyMobileDefaults } from './ui-controller.js';
import { initServerClient, detectMobile, autoDetectServer } from './server-client.js';

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
    qMethod: 'yoshida4',
    qSteps: 200,
    qResScale: 1.0,
    qStepSize: 0.3,
    qObsDist: 40,
    qStarLayers: 3,
    renderMode: 'local',

    // Cached ISCO value (expensive to compute with charge > 0)
    _cachedIsco: null,
    _cachedIscoSpin: null,
    _cachedIscoCharge: null,
    getIsco(a, Q) {
        if (a === this._cachedIscoSpin && Q === this._cachedIscoCharge && this._cachedIsco !== null) return this._cachedIsco;
        this._cachedIsco = iscoJS(a, Q);
        this._cachedIscoSpin = a;
        this._cachedIscoCharge = Q;
        return this._cachedIsco;
    }
};

// ============================================================
//  INITIALIZATION
// ============================================================
const canvas = document.getElementById('canvas');
const loadEl = document.getElementById('loading');
const loadMsg = document.getElementById('loading-msg');
const errMsg = document.getElementById('error-msg');

const gl = initWebGL(canvas, loadMsg, errMsg);
if (!gl) {
    // WebGL not available — initWebGL already showed the error
} else {
    // Wire up shared state to renderer
    setStateRef(state);

    // Initialize server client
    initServerClient({
        serverFrame: document.getElementById('server-frame'),
        serverDot: document.getElementById('server-dot'),
        serverStatusEl: document.getElementById('server-status'),
        canvas: canvas,
        stateRef: state
    });

    // Initialize UI (event handlers, sliders, etc.)
    initUI(state);

    // Auto-detect mobile and suggest hybrid mode
    if (detectMobile()) {
        applyMobileDefaults();
        // Auto-show settings panel so user can configure server
        document.getElementById('settings-panel').classList.remove('hidden');
        document.getElementById('btn-settings').classList.add('active');
    }

    // Run initial setup
    updInfo();
    refreshLabels();
    loadMsg.textContent = 'Compiling shaders…';
    resize();
    const initResult = buildProgram();
    if (!initResult.ok) {
        loadMsg.textContent = 'Shader compile failed';
        errMsg.style.display = 'block';
        errMsg.textContent = initResult.error;
    } else {
        loadMsg.textContent = 'Tracing first frame…';
        requestAnimationFrame(() => {
            markDirty();
            setTimeout(() => loadEl.classList.add('hidden'), 500);
        });
    }

    // Auto-detect same-origin server
    autoDetectServer();
}
