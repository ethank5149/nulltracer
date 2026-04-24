import re

with open("web/index.html", 'r') as f:
    content = f.read()

new_controls = """  <div class="control-row">
    <div class="control-label" title="QED vacuum polarization strength"><span>QED Coupling</span><span class="control-value" id="qed-val">0.0</span></div>
    <input type="range" id="qed-coupling" min="0" max="1" step="0.01" value="0.0">
  </div>
  <div class="control-row">
    <div class="control-label" title="Hawking radiation glow amplification"><span>Hawking Boost</span><span class="control-value" id="hawking-val">0</span></div>
    <input type="range" id="hawking-boost" min="0" max="1000000" step="1000" value="0">
  </div>
"""

content = content.replace('  <div class="control-row">\n    <div class="control-label"><span>Disk Temperature</span><span class="control-value" id="temp-val">1.00</span></div>', new_controls + '  <div class="control-row">\n    <div class="control-label"><span>Disk Temperature</span><span class="control-value" id="temp-val">1.00</span></div>')

with open("web/index.html", 'w') as f:
    f.write(content)

with open("web/js/ui-controller.js", 'r') as f:
    js_content = f.read()
    
js_events = """    document.getElementById('temp').addEventListener('input', function(){ stateRef.diskTemp=+this.value; document.getElementById('temp-val').textContent=stateRef.diskTemp.toFixed(2); requestRender(); });
    document.getElementById('qed-coupling').addEventListener('input', function(){ stateRef.qedCoupling=+this.value; document.getElementById('qed-val').textContent=stateRef.qedCoupling.toFixed(2); requestRender(); });
    document.getElementById('hawking-boost').addEventListener('input', function(){ stateRef.hawkingBoost=+this.value; document.getElementById('hawking-val').textContent=stateRef.hawkingBoost; requestRender(); });"""

js_content = js_content.replace("    document.getElementById('temp').addEventListener('input', function(){ stateRef.diskTemp=+this.value; document.getElementById('temp-val').textContent=stateRef.diskTemp.toFixed(2); requestRender(); });", js_events)

map_add = """    disk_max_crossings: { state: 'diskMaxCrossings', slider: 'disk-max-crossings',label: 'disk-max-crossings-val',fmt: v => '' + v },
    qed_coupling:       { state: 'qedCoupling',      slider: 'qed-coupling',      label: 'qed-val',               fmt: v => v.toFixed(2) },
    hawking_boost:      { state: 'hawkingBoost',     slider: 'hawking-boost',     label: 'hawking-val',           fmt: v => '' + v },"""

js_content = js_content.replace("    disk_max_crossings: { state: 'diskMaxCrossings', slider: 'disk-max-crossings',label: 'disk-max-crossings-val',fmt: v => '' + v },", map_add)

save_scene = """        disk_max_crossings: stateRef.diskMaxCrossings,
        qed_coupling: stateRef.qedCoupling,
        hawking_boost: stateRef.hawkingBoost,"""

js_content = js_content.replace("        disk_max_crossings: stateRef.diskMaxCrossings,", save_scene)

with open("web/js/ui-controller.js", 'w') as f:
    f.write(js_content)
    
with open("web/js/state.js", 'r') as f:
    state_content = f.read()

state_content = state_content.replace("    bloomRadius: 1.0,", "    bloomRadius: 1.0,\n    qedCoupling: 0.0,\n    hawkingBoost: 0,")

with open("web/js/state.js", 'w') as f:
    f.write(state_content)

with open("web/js/server-client.js", 'r') as f:
    srv_content = f.read()

srv_content = srv_content.replace("        bloom_radius: state.bloomRadius,", "        bloom_radius: state.bloomRadius,\n        qed_coupling: state.qedCoupling,\n        hawking_boost: state.hawkingBoost,")
with open("web/js/server-client.js", 'w') as f:
    f.write(srv_content)

with open("web/js/ws-client.js", 'r') as f:
    ws_content = f.read()

ws_content = ws_content.replace("        bloom_radius: state.bloomRadius,", "        bloom_radius: state.bloomRadius,\n        qed_coupling: state.qedCoupling,\n        hawking_boost: state.hawkingBoost,")
with open("web/js/ws-client.js", 'w') as f:
    f.write(ws_content)

print("HTML and JS updated")
