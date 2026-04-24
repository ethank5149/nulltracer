with open("web/js/main.js", 'r') as f:
    state_content = f.read()

state_content = state_content.replace("    bloomRadius: 1.0,", "    bloomRadius: 1.0,\n    qedCoupling: 0.0,\n    hawkingBoost: 0.0,")

with open("web/js/main.js", 'w') as f:
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

print("Done")
