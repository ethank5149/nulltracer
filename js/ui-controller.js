// ============================================================
//  UI CONTROLLER
//  All DOM event handlers, slider logic, button toggles,
//  panel show/hide, presets, and label updates.
// ============================================================

import { markDirty, buildProgram, resize } from './webgl-renderer.js';
import { rPlus, iscoJS } from './isco-calculator.js';
import {
    scheduleServerRender, updateRenderMode, checkServerHealth,
    setServerUrl, setServerQuality, resetServerFailCount,
    setServerStatus, detectMobile, getServerUrl, getRenderMode
} from './server-client.js';

let stateRef = null;

const presets = {
    low:    {method:'yoshida4', steps:80,  resScale:0.5,  stepSize:0.5,  obsDist:30, starLayers:1},
    medium: {method:'yoshida4', steps:200, resScale:1,    stepSize:0.3,  obsDist:40, starLayers:3},
    high:   {method:'yoshida6',  steps:200, resScale:1.25, stepSize:0.35, obsDist:50, starLayers:4},
    ultra:  {method:'rkdp8',     steps:180, resScale:1.25, stepSize:0.45, obsDist:60, starLayers:4},
};

export function updInfo() {
    document.getElementById('rp').textContent = rPlus(stateRef.spin, stateRef.charge).toFixed(3);
    document.getElementById('risco').textContent = iscoJS(stateRef.spin, stateRef.charge).toFixed(3);
    document.getElementById('Qval').textContent = stateRef.charge.toFixed(3);
}

export function refreshLabels() {
    const integLabels = {yoshida4:'Yoshida 4th symplectic', rk4:'Runge-Kutta 4th', yoshida6:'Yoshida 6th symplectic', yoshida8:'Yoshida 8th symplectic', rkdp8:'Dormand-Prince RK8'};
    document.getElementById('integ-label').textContent = integLabels[stateRef.qMethod] || stateRef.qMethod;
    document.getElementById('steps-val').textContent = stateRef.qSteps;
    document.getElementById('res-scale-val').textContent = stateRef.qResScale.toFixed(2)+'×';
    document.getElementById('stepsize-val').textContent = stateRef.qStepSize.toFixed(2);
    document.getElementById('obs-val').textContent = stateRef.qObsDist+' M';
    document.getElementById('starlayers-val').textContent = stateRef.qStarLayers;
    document.getElementById('integrator-warn').style.display = (stateRef.qMethod!=='yoshida4') ? 'inline-block' : 'none';
}

let rTimer = null;
export function recompile() {
    clearTimeout(rTimer);
    rTimer = setTimeout(() => {
        resize();
        const r = buildProgram();
        if (!r.ok) {
            console.error('Shader fail:', r.error);
            if (stateRef.qMethod !== 'yoshida4') {
                const failedMethod = stateRef.qMethod;
                stateRef.qMethod='yoshida4'; document.getElementById('sel-integrator').value='yoshida4'; refreshLabels();
                if (buildProgram().ok) alert(failedMethod + ' integrator failed on your GPU. Fell back to Yoshida 4th-order.');
            }
        }
        markDirty();
    }, 120);
}

function applyPreset(name) {
    const p=presets[name]; if(!p) return;
    stateRef.qMethod=p.method; stateRef.qSteps=p.steps; stateRef.qResScale=p.resScale; stateRef.qStepSize=p.stepSize; stateRef.qObsDist=p.obsDist; stateRef.qStarLayers=p.starLayers;
    document.getElementById('sel-integrator').value=p.method;
    document.getElementById('steps').value=p.steps;
    document.getElementById('res-scale').value=p.resScale;
    document.getElementById('stepsize').value=p.stepSize;
    document.getElementById('obs-dist').value=p.obsDist;
    document.getElementById('star-layers').value=p.starLayers;
    refreshLabels();
    document.querySelectorAll('.preset-btn').forEach(b=>b.classList.toggle('active',b.dataset.preset===name));
    recompile();
}

function clearPH() { document.querySelectorAll('.preset-btn').forEach(b=>b.classList.remove('active')); }

// Apply mobile-optimized settings
export function applyMobileDefaults() {
    stateRef.qMethod = 'yoshida4'; stateRef.qSteps = 80; stateRef.qResScale = 0.25; stateRef.qStepSize = 0.5; stateRef.qObsDist = 30; stateRef.qStarLayers = 1;
    document.getElementById('sel-integrator').value = 'yoshida4';
    document.getElementById('steps').value = 80;
    document.getElementById('res-scale').value = 0.25;
    document.getElementById('stepsize').value = 0.5;
    document.getElementById('obs-dist').value = 30;
    document.getElementById('star-layers').value = 1;
    refreshLabels();
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    recompile();
}

export function initUI(state) {
    stateRef = state;

    // Charge slider + constraint: a^2 + Q^2 <= 1
    document.getElementById('charge').addEventListener('input', function(){
        stateRef.charge=+this.value;
        // Enforce a^2 + Q^2 <= 1
        const maxA = Math.sqrt(Math.max(1 - stateRef.charge*stateRef.charge, 0));
        if (stateRef.spin > maxA) { stateRef.spin = Math.floor(maxA*500)/500; document.getElementById('spin').value = stateRef.spin; document.getElementById('spin-val').textContent='a = '+stateRef.spin.toFixed(3); }
        document.getElementById('charge-val').textContent='Q = '+stateRef.charge.toFixed(3);
        updInfo(); markDirty();
        scheduleServerRender();
    });
    // Also enforce constraint when spin changes
    document.getElementById('spin').addEventListener('input', function(){
        stateRef.spin=+this.value;
        const maxQ = Math.sqrt(Math.max(1 - stateRef.spin*stateRef.spin, 0));
        if (stateRef.charge > maxQ) { stateRef.charge = Math.floor(maxQ*500)/500; document.getElementById('charge').value = stateRef.charge; document.getElementById('charge-val').textContent='Q = '+stateRef.charge.toFixed(3); }
        document.getElementById('spin-val').textContent='a = '+stateRef.spin.toFixed(3);
        updInfo(); markDirty();
        scheduleServerRender();
    });
    document.getElementById('incl').addEventListener('input', function(){ stateRef.incl=+this.value; document.getElementById('incl-val').innerHTML='&theta; = '+stateRef.incl.toFixed(1)+'&deg;'; markDirty(); scheduleServerRender(); });
    document.getElementById('temp').addEventListener('input', function(){ stateRef.diskTemp=+this.value; document.getElementById('temp-val').textContent=stateRef.diskTemp.toFixed(2); markDirty(); scheduleServerRender(); });
    document.getElementById('btn-disk').addEventListener('click', function(){ stateRef.showDisk=stateRef.showDisk>0.5?0:1; this.classList.toggle('active',stateRef.showDisk>0.5); markDirty(); scheduleServerRender(); });
    document.getElementById('btn-grid').addEventListener('click', function(){ stateRef.showGrid=stateRef.showGrid>0.5?0:1; this.classList.toggle('active',stateRef.showGrid>0.5); markDirty(); scheduleServerRender(); });
    document.getElementById('btn-rotate').addEventListener('click', function(){ stateRef.autoRotate=!stateRef.autoRotate; this.classList.toggle('active',stateRef.autoRotate); if(!stateRef.autoRotate) scheduleServerRender(); else { markDirty(); var sf=document.getElementById('server-frame'); if(sf) sf.classList.remove('visible'); } });
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
            recompile();
            scheduleServerRender();
        });
    });
    document.querySelectorAll('.preset-btn').forEach(btn => btn.addEventListener('click', ()=>applyPreset(btn.dataset.preset)));
    document.getElementById('sel-integrator').addEventListener('change', function(){ stateRef.qMethod=this.value; refreshLabels(); clearPH(); recompile(); scheduleServerRender(); });
    document.getElementById('steps').addEventListener('input', function(){ stateRef.qSteps=+this.value; document.getElementById('steps-val').textContent=stateRef.qSteps; clearPH(); recompile(); scheduleServerRender(); });
    document.getElementById('res-scale').addEventListener('input', function(){ stateRef.qResScale=+this.value; document.getElementById('res-scale-val').textContent=stateRef.qResScale.toFixed(2)+'×'; clearPH(); resize(); scheduleServerRender(); });
    document.getElementById('stepsize').addEventListener('input', function(){ stateRef.qStepSize=+this.value; document.getElementById('stepsize-val').textContent=stateRef.qStepSize.toFixed(2); clearPH(); recompile(); scheduleServerRender(); });
    document.getElementById('obs-dist').addEventListener('input', function(){ stateRef.qObsDist=+this.value; document.getElementById('obs-val').textContent=stateRef.qObsDist+' M'; clearPH(); recompile(); scheduleServerRender(); });
    document.getElementById('star-layers').addEventListener('input', function(){ stateRef.qStarLayers=+this.value; document.getElementById('starlayers-val').textContent=stateRef.qStarLayers; clearPH(); recompile(); scheduleServerRender(); });

    const canvasEl = document.getElementById('canvas');
    canvasEl.addEventListener('wheel', function(e){ e.preventDefault(); stateRef.fov*=(1+e.deltaY*0.001); stateRef.fov=Math.max(2,Math.min(25,stateRef.fov)); markDirty(); scheduleServerRender(); }, {passive:false});
    let ltd=0;
    canvasEl.addEventListener('touchstart', function(e){ if(e.touches.length===2){ const dx=e.touches[0].clientX-e.touches[1].clientX,dy=e.touches[0].clientY-e.touches[1].clientY; ltd=Math.sqrt(dx*dx+dy*dy); }});
    canvasEl.addEventListener('touchmove', function(e){ if(e.touches.length===2){ e.preventDefault(); const dx=e.touches[0].clientX-e.touches[1].clientX,dy=e.touches[0].clientY-e.touches[1].clientY,d=Math.sqrt(dx*dx+dy*dy); if(ltd>0){stateRef.fov*=ltd/d;stateRef.fov=Math.max(2,Math.min(25,stateRef.fov));}ltd=d; markDirty(); scheduleServerRender(); }}, {passive:false});

    // ---- Server UI Event Listeners ----
    document.getElementById('server-url').addEventListener('change', function() {
        const url = this.value.replace(/\/+$/, '');  // trim trailing slashes
        setServerUrl(url);
        resetServerFailCount();
        if (url) {
            checkServerHealth().then(ok => {
                if (ok && getRenderMode() !== 'local') scheduleServerRender();
            });
        } else {
            setServerStatus('disconnected', 'server off');
            document.getElementById('server-frame').classList.remove('visible');
        }
    });

    document.getElementById('btn-local-only').addEventListener('click', function() { updateRenderMode('local'); });
    document.getElementById('btn-hybrid').addEventListener('click', function() { updateRenderMode('hybrid'); });
    document.getElementById('btn-server-only').addEventListener('click', function() { updateRenderMode('server'); });

    document.getElementById('srv-quality').addEventListener('change', function() {
        setServerQuality(parseInt(this.value));
        document.getElementById('srv-quality-val').textContent = this.value + 'p';
        scheduleServerRender();
    });

    addEventListener('resize', resize);
}
