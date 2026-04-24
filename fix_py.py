import re

with open("nulltracer/_params.py", 'r') as f:
    content = f.read()

content = content.replace('("sky_height", ctypes.c_double),', '("sky_height", ctypes.c_double),\n    ("qed_coupling", ctypes.c_double),\n    ("hawking_boost", ctypes.c_double),')

with open("nulltracer/_params.py", 'w') as f:
    f.write(content)

with open("nulltracer/renderer.py", 'r') as f:
    content = f.read()
    
content = content.replace('sky_height=0.0,\n        )', 'sky_height=0.0,\n            qed_coupling=float(params.get("qed_coupling", 0.0)),\n            hawking_boost=float(params.get("hawking_boost", 0.0)),\n        )')
content = content.replace('steps = _resolve_steps(\n            params, spin=spin, charge=charge, method=method,\n            obs_dist=obs_dist, step_size=step_size,\n        )', 'steps = int(params.get("steps", 2000))')
content = content.replace('step_size=step_size,', 'step_size=float(params.get("step_size", 0.08)),')

with open("nulltracer/renderer.py", 'w') as f:
    f.write(content)

with open("nulltracer/server.py", 'r') as f:
    content = f.read()
    
content = content.replace('bloom_radius: float = 1.0\n    format: str = "jpeg"', 'bloom_radius: float = 1.0\n    qed_coupling: float = 0.0\n    hawking_boost: float = 0.0\n    format: str = "jpeg"')
content = content.replace('steps: Optional[int] = None\n    step_size: float = 0.30', 'steps: Optional[int] = 2000\n    step_size: float = 0.08')
content = content.replace('disk_max_crossings: int = 5', 'disk_max_crossings: int = 8')

with open("nulltracer/server.py", 'w') as f:
    f.write(content)
print("Done")
