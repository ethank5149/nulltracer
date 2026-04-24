// ============================================================
//  UI CONTROLLER
//  All DOM event handlers, slider logic, button toggles,
//  panel show/hide, presets, and label updates.
//  No local WebGL rendering — all changes trigger server render.
// ============================================================

import {
    scheduleServerRender, checkServerHealth,
    setServerUrl, setServerQuality, resetServerFailCount,
    setServerStatus,
    fetchScenes, fetchScene, saveSceneAPI, deleteSceneAPI
} from './server-client.js';
import { sendParams, isWsConnected, connectWebSocket, disconnectWebSocket } from './ws-client.js';

let stateRef = null;

/**
 * Unified render trigger — prefers WebSocket, falls back to HTTP.
 */
function requestRender() {
    if (isWsConnected()) {
        sendParams();
    } else {
        scheduleServerRender();
    }
}

const presets = {
    low:    {method:'rk4',      steps:80,  stepSize:0.5,  obsDist:30, starLayers:1},
    medium: {method:'rkdp8',    steps:200, stepSize:0.3,  obsDist:40, starLayers:3},
    high:   {method:'rkdp8',    steps:180, stepSize:0.40, obsDist:50, starLayers:4},
    ultra:  {method:'rkdp8',    steps:200, stepSize:0.45, obsDist:60, starLayers:4},
};

export function refreshLabels() {
    const integLabels = {rk4:'Runge-Kutta 4th', rkdp8:'Dormand-Prince RK8', tao_yoshida4:'Tao-Yoshida 4th', tao_yoshida6:'Tao-Yoshida 6th', tao_kahan_li8:'Tao-Kahan-Li 8th'};
    document.getElementById('integ-label').textContent = integLabels[stateRef.qMethod] || stateRef.qMethod;
    document.getElementById('steps-val').textContent = stateRef.qSteps;
    document.getElementById('stepsize-val').textContent = stateRef.qStepSize.toFixed(2);
    document.getElementById('obs-val').textContent = stateRef.qObsDist+' M';
    document.getElementById('starlayers-val').textContent = stateRef.qStarLayers;
    document.getElementById('integrator-warn').style.display = (stateRef.qMethod!=='rkdp8') ? 'inline-block' : 'none';
}

function applyPreset(name) {
    const p=presets[name]; if(!p) return;
    stateRef.qMethod=p.method; stateRef.qSteps=p.steps; stateRef.qStepSize=p.stepSize; stateRef.qObsDist=p.obsDist; stateRef.qStarLayers=p.starLayers;
    document.getElementById('sel-integrator').value=p.method;
    document.getElementById('steps').value=p.steps;
    document.getElementById('stepsize').value=p.stepSize;
    document.getElementById('obs-dist').value=p.obsDist;
    document.getElementById('star-layers').value=p.starLayers;
    refreshLabels();
    document.querySelectorAll('.preset-btn').forEach(b=>b.classList.toggle('active',b.dataset.preset===name));
    requestRender();
}

function clearPH() { document.querySelectorAll('.preset-btn').forEach(b=>b.classList.remove('active')); }

export function initUI(state) {
    stateRef = state;

    // Charge slider + constraint: a^2 + Q^2 <= 1
    document.getElementById('charge').addEventListener('input', function(){
        stateRef.charge=+this.value;
        // Enforce a^2 + Q^2 <= 1
        const maxA = Math.sqrt(Math.max(1 - stateRef.charge*stateRef.charge, 0));
        if (stateRef.spin > maxA) { stateRef.spin = Math.floor(maxA*500)/500; document.getElementById('spin').value = stateRef.spin; document.getElementById('spin-val').textContent='a = '+stateRef.spin.toFixed(3); }
        document.getElementById('charge-val').textContent='Q = '+stateRef.charge.toFixed(3);
        requestRender();
    });
    // Also enforce constraint when spin changes
    document.getElementById('spin').addEventListener('input', function(){
        stateRef.spin=+this.value;
        const maxQ = Math.sqrt(Math.max(1 - stateRef.spin*stateRef.spin, 0));
        if (stateRef.charge > maxQ) { stateRef.charge = Math.floor(maxQ*500)/500; document.getElementById('charge').value = stateRef.charge; document.getElementById('charge-val').textContent='Q = '+stateRef.charge.toFixed(3); }
        document.getElementById('spin-val').textContent='a = '+stateRef.spin.toFixed(3);
        requestRender();
    });
    document.getElementById('incl').addEventListener('input', function(){ stateRef.incl=+this.value; document.getElementById('incl-val').innerHTML='&theta; = '+stateRef.incl.toFixed(1)+'&deg;'; requestRender(); });
    document.getElementById('temp').addEventListener('input', function(){ stateRef.diskTemp=+this.value; document.getElementById('temp-val').textContent=stateRef.diskTemp.toFixed(2); requestRender(); });
    document.getElementById('qed-coupling').addEventListener('input', function(){ stateRef.qedCoupling=+this.value; document.getElementById('qed-val').textContent=stateRef.qedCoupling.toFixed(2); requestRender(); });
    document.getElementById('hawking-boost').addEventListener('input', function(){ stateRef.hawkingBoost=+this.value; document.getElementById('hawking-val').textContent=stateRef.hawkingBoost; requestRender(); });
    document.getElementById('disk-alpha').addEventListener('input', function(){ stateRef.diskAlpha=+this.value; document.getElementById('disk-alpha-val').textContent=stateRef.diskAlpha.toFixed(2); requestRender(); });
    document.getElementById('disk-max-crossings').addEventListener('input', function(){ stateRef.diskMaxCrossings=+this.value; document.getElementById('disk-max-crossings-val').textContent=stateRef.diskMaxCrossings; requestRender(); });
    document.getElementById('btn-disk').addEventListener('click', function(){ stateRef.showDisk=stateRef.showDisk>0.5?0:1; this.classList.toggle('active',stateRef.showDisk>0.5); requestRender(); });
    document.getElementById('btn-grid').addEventListener('click', function(){ stateRef.showGrid=stateRef.showGrid>0.5?0:1; this.classList.toggle('active',stateRef.showGrid>0.5); requestRender(); });
    document.getElementById('btn-srgb').addEventListener('click', function(){ stateRef.srgbOutput=!stateRef.srgbOutput; this.classList.toggle('active',stateRef.srgbOutput); requestRender(); });
    document.getElementById('btn-bloom').addEventListener('click', function(){
        stateRef.bloomEnabled=!stateRef.bloomEnabled;
        this.classList.toggle('active',stateRef.bloomEnabled);
        document.getElementById('bloom-radius-row').style.display=stateRef.bloomEnabled?'':'none';
        requestRender();
    });
    document.getElementById('bloom-radius').addEventListener('input', function(){ stateRef.bloomRadius=+this.value; document.getElementById('bloom-radius-val').textContent=stateRef.bloomRadius.toFixed(1); requestRender(); });
    document.getElementById('btn-rotate').addEventListener('click', function(){ stateRef.autoRotate=!stateRef.autoRotate; this.classList.toggle('active',stateRef.autoRotate); if(stateRef.autoRotate) startAutoRotate(); });
    document.getElementById('btn-settings').addEventListener('click', function(){
        const p=document.getElementById('settings-panel'); p.classList.toggle('hidden'); this.classList.toggle('active',!p.classList.contains('hidden'));
    });
    document.getElementById('btn-legend').addEventListener('click', function(){
        const p=document.getElementById('legend-panel'); p.classList.toggle('hidden'); this.classList.toggle('active',!p.classList.contains('hidden'));
    });
    document.getElementById('btn-legend-close').addEventListener('click', function(){
        document.getElementById('legend-panel').classList.add('hidden'); document.getElementById('btn-legend').classList.remove('active');
    });
    document.getElementById('btn-ann-toggle').addEventListener('click', function(){
        const on=!document.getElementById('ann-container').classList.contains('annotations-hidden');
        document.getElementById('ann-container').classList.toggle('annotations-hidden',on);
        this.classList.toggle('active',!on);
    });
    document.querySelectorAll('[data-bg]').forEach(btn => {
        btn.addEventListener('click', function(){
            stateRef.bgMode=parseInt(this.dataset.bg);
            document.querySelectorAll('[data-bg]').forEach(b=>b.classList.toggle('active',parseInt(b.dataset.bg)===stateRef.bgMode));
            requestRender();
        });
    });
    document.querySelectorAll('.preset-btn').forEach(btn => btn.addEventListener('click', ()=>applyPreset(btn.dataset.preset)));
    document.getElementById('sel-integrator').addEventListener('change', function(){ stateRef.qMethod=this.value; refreshLabels(); clearPH(); requestRender(); });
    document.getElementById('steps').addEventListener('input', function(){ stateRef.qSteps=+this.value; document.getElementById('steps-val').textContent=stateRef.qSteps; clearPH(); requestRender(); });
    document.getElementById('stepsize').addEventListener('input', function(){ stateRef.qStepSize=+this.value; document.getElementById('stepsize-val').textContent=stateRef.qStepSize.toFixed(2); clearPH(); requestRender(); });
    document.getElementById('obs-dist').addEventListener('input', function(){ stateRef.qObsDist=+this.value; document.getElementById('obs-val').textContent=stateRef.qObsDist+' M'; clearPH(); requestRender(); });
    document.getElementById('star-layers').addEventListener('input', function(){ stateRef.qStarLayers=+this.value; document.getElementById('starlayers-val').textContent=stateRef.qStarLayers; clearPH(); requestRender(); });

    // FOV via scroll wheel on the server frame / viewport
    const frameEl = document.getElementById('server-frame');
    document.addEventListener('wheel', function(e){ e.preventDefault(); stateRef.fov*=(1+e.deltaY*0.001); stateRef.fov=Math.max(2,Math.min(25,stateRef.fov)); requestRender(); }, {passive:false});
    let ltd=0;
    document.addEventListener('touchstart', function(e){ if(e.touches.length===2){ const dx=e.touches[0].clientX-e.touches[1].clientX,dy=e.touches[0].clientY-e.touches[1].clientY; ltd=Math.sqrt(dx*dx+dy*dy); }});
    document.addEventListener('touchmove', function(e){ if(e.touches.length===2){ e.preventDefault(); const dx=e.touches[0].clientX-e.touches[1].clientX,dy=e.touches[0].clientY-e.touches[1].clientY,d=Math.sqrt(dx*dx+dy*dy); if(ltd>0){stateRef.fov*=ltd/d;stateRef.fov=Math.max(2,Math.min(25,stateRef.fov));}ltd=d; requestRender(); }}, {passive:false});

    // ---- Server UI Event Listeners ----
    document.getElementById('server-url').addEventListener('change', function() {
        const url = this.value.replace(/\/+$/, '');  // trim trailing slashes
        setServerUrl(url);
        resetServerFailCount();
        if (url) {
            checkServerHealth().then(ok => {
                if (ok) scheduleServerRender();
            });
        } else {
            setServerStatus('disconnected', 'server off');
            document.getElementById('server-frame').classList.remove('visible');
        }

        // Reconnect WebSocket to new server
        disconnectWebSocket();
        if (url) {
            connectWebSocket(url);
        }
    });

    document.getElementById('srv-quality').addEventListener('change', function() {
        setServerQuality(parseInt(this.value));
        document.getElementById('srv-quality-val').textContent = this.value + 'p';
        scheduleServerRender();
    });

    // ---- Scene management button handlers ----
    document.getElementById('btn-load-scene').addEventListener('click', function() {
        const name = document.getElementById('scene-select').value;
        if (name) loadScene(name);
    });

    document.getElementById('btn-save-scene').addEventListener('click', function() {
        const name = document.getElementById('scene-name-input').value.trim();
        saveScene(name);
    });

    document.getElementById('btn-delete-scene').addEventListener('click', function() {
        const name = document.getElementById('scene-select').value;
        if (name) deleteScene(name);
    });
}

// ---- Auto-rotate support ----
let rotateInterval = null;
function startAutoRotate() {
    if (rotateInterval) return;
    rotateInterval = setInterval(() => {
        if (!stateRef.autoRotate) { clearInterval(rotateInterval); rotateInterval = null; return; }
        stateRef.rotAngle += 0.02;
        requestRender();
    }, 100);
}

// ── Scene management ────────────────────────────────────────

/**
 * Populate the scene dropdown from the server.
 */
export async function loadSceneList() {
    const select = document.getElementById('scene-select');
    if (!select) return;
    try {
        const data = await fetchScenes();
        const scenes = data.scenes || [];
        // Preserve current selection if possible
        const prev = select.value;
        select.innerHTML = '<option value="">— Select Scene —</option>';
        for (const s of scenes) {
            const opt = document.createElement('option');
            opt.value = s.name;
            opt.textContent = s.name + (s.builtin ? ' ★' : '');
            if (s.description) opt.title = s.description;
            select.appendChild(opt);
        }
        // Restore previous selection
        if (prev && [...select.options].some(o => o.value === prev)) {
            select.value = prev;
        }
    } catch (e) {
        console.warn('Failed to load scene list:', e.message);
    }
}

/**
 * Map from scene JSON keys to state object keys and UI elements.
 * This is the central mapping that loadScene uses to update everything.
 */
const SCENE_PARAM_MAP = {
    spin:               { state: 'spin',             slider: 'spin',              label: 'spin-val',              fmt: v => 'a = ' + v.toFixed(3) },
    charge:             { state: 'charge',           slider: 'charge',            label: 'charge-val',            fmt: v => 'Q = ' + v.toFixed(3) },
    inclination:        { state: 'incl',             slider: 'incl',              label: 'incl-val',              fmt: v => 'θ = ' + v.toFixed(1) + '°' },
    fov:                { state: 'fov' },
    phi0:               { state: 'rotAngle' },
    steps:              { state: 'qSteps',           slider: 'steps',             label: 'steps-val',             fmt: v => '' + v },
    step_size:          { state: 'qStepSize',        slider: 'stepsize',          label: 'stepsize-val',          fmt: v => v.toFixed(2) },
    obs_dist:           { state: 'qObsDist',         slider: 'obs-dist',          label: 'obs-val',               fmt: v => v + ' M' },
    method:             { state: 'qMethod',          select: 'sel-integrator' },
    bg_mode:            { state: 'bgMode' },
    star_layers:        { state: 'qStarLayers',      slider: 'star-layers',       label: 'starlayers-val',        fmt: v => '' + v },
    show_disk:          { state: 'showDisk',         toggle: 'btn-disk' },
    show_grid:          { state: 'showGrid',         toggle: 'btn-grid' },
    disk_temp:          { state: 'diskTemp',         slider: 'temp',              label: 'temp-val',              fmt: v => v.toFixed(2) },
    disk_alpha:         { state: 'diskAlpha',        slider: 'disk-alpha',        label: 'disk-alpha-val',        fmt: v => v.toFixed(2) },
    disk_max_crossings: { state: 'diskMaxCrossings', slider: 'disk-max-crossings',label: 'disk-max-crossings-val',fmt: v => '' + v },
    qed_coupling:       { state: 'qedCoupling',      slider: 'qed-coupling',      label: 'qed-val',               fmt: v => v.toFixed(2) },
    hawking_boost:      { state: 'hawkingBoost',     slider: 'hawking-boost',     label: 'hawking-val',           fmt: v => '' + v },
    srgb_output:        { state: 'srgbOutput',       toggle: 'btn-srgb' },
    bloom_enabled:      { state: 'bloomEnabled',     toggle: 'btn-bloom' },
    bloom_radius:       { state: 'bloomRadius',      slider: 'bloom-radius',      label: 'bloom-radius-val',      fmt: v => v.toFixed(1) },
};

/**
 * Load a scene by name: fetch params from server, update state + UI, re-render.
 */
async function loadScene(name) {
    if (!name) return;
    try {
        const params = await fetchScene(name);

        // Apply each parameter to state and UI
        for (const [key, val] of Object.entries(params)) {
            const mapping = SCENE_PARAM_MAP[key];
            if (!mapping) continue;

            // Update state
            if (mapping.state) stateRef[mapping.state] = val;

            // Update slider
            if (mapping.slider) {
                const el = document.getElementById(mapping.slider);
                if (el) el.value = val;
            }

            // Update label
            if (mapping.label && mapping.fmt) {
                const el = document.getElementById(mapping.label);
                if (el) el.textContent = mapping.fmt(val);
            }

            // Update select
            if (mapping.select) {
                const el = document.getElementById(mapping.select);
                if (el) el.value = val;
            }

            // Update toggle button
            if (mapping.toggle) {
                const el = document.getElementById(mapping.toggle);
                if (el) {
                    const boolVal = typeof val === 'number' ? val > 0.5 : !!val;
                    el.classList.toggle('active', boolVal);
                }
            }
        }

        // Special handling: show_disk and show_grid are stored as booleans in scene
        // but as 0/1 in state
        if ('show_disk' in params) stateRef.showDisk = params.show_disk ? 1 : 0;
        if ('show_grid' in params) stateRef.showGrid = params.show_grid ? 1 : 0;

        // Update background mode buttons
        if ('bg_mode' in params) {
            document.querySelectorAll('[data-bg]').forEach(b =>
                b.classList.toggle('active', parseInt(b.dataset.bg) === stateRef.bgMode)
            );
        }

        // Update bloom radius row visibility
        document.getElementById('bloom-radius-row').style.display = stateRef.bloomEnabled ? '' : 'none';

        // Refresh integrator labels
        refreshLabels();

        // Clear preset highlights (scene may not match any preset)
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));

        // Trigger re-render
        requestRender();

        console.log('Loaded scene:', name);
    } catch (e) {
        console.error('Failed to load scene:', e.message);
        alert('Failed to load scene: ' + e.message);
    }
}

/**
 * Save the current state as a named scene.
 */
async function saveScene(name) {
    if (!name) { alert('Please enter a scene name.'); return; }

    // Collect current params from state
    const params = {
        spin: stateRef.spin,
        charge: stateRef.charge,
        inclination: stateRef.incl,
        fov: stateRef.fov,
        phi0: stateRef.rotAngle,
        steps: stateRef.qSteps,
        step_size: stateRef.qStepSize,
        obs_dist: stateRef.qObsDist,
        method: stateRef.qMethod,
        bg_mode: stateRef.bgMode,
        star_layers: stateRef.qStarLayers,
        show_disk: stateRef.showDisk > 0.5,
        show_grid: stateRef.showGrid > 0.5,
        disk_temp: stateRef.diskTemp,
        srgb_output: !!stateRef.srgbOutput,
        disk_alpha: stateRef.diskAlpha,
        disk_max_crossings: stateRef.diskMaxCrossings,
        qed_coupling: stateRef.qedCoupling,
        hawking_boost: stateRef.hawkingBoost,
        bloom_enabled: !!stateRef.bloomEnabled,
        bloom_radius: stateRef.bloomRadius,
    };

    try {
        await saveSceneAPI(name, params);
        console.log('Saved scene:', name);
        await loadSceneList();
        // Select the newly saved scene
        const select = document.getElementById('scene-select');
        if (select) select.value = name;
    } catch (e) {
        console.error('Failed to save scene:', e.message);
        alert('Failed to save scene: ' + e.message);
    }
}

/**
 * Delete a scene by name (with confirmation).
 */
async function deleteScene(name) {
    if (!name) return;
    if (!confirm(`Delete scene "${name}"?`)) return;

    try {
        await deleteSceneAPI(name);
        console.log('Deleted scene:', name);
        await loadSceneList();
    } catch (e) {
        console.error('Failed to delete scene:', e.message);
        alert('Failed to delete scene: ' + e.message);
    }
}
