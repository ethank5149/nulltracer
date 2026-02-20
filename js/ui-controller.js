// ============================================================
//  UI CONTROLLER
//  All DOM event handlers, slider logic, button toggles,
//  panel show/hide, presets, and label updates.
//  No local WebGL rendering — all changes trigger server render.
// ============================================================

import {
    scheduleServerRender, checkServerHealth,
    setServerUrl, setServerQuality, resetServerFailCount,
    setServerStatus, getServerUrl
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
    document.getElementById('btn-disk').addEventListener('click', function(){ stateRef.showDisk=stateRef.showDisk>0.5?0:1; this.classList.toggle('active',stateRef.showDisk>0.5); requestRender(); });
    document.getElementById('btn-grid').addEventListener('click', function(){ stateRef.showGrid=stateRef.showGrid>0.5?0:1; this.classList.toggle('active',stateRef.showGrid>0.5); requestRender(); });
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
