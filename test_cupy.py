import cupy as cp
try:
    props = cp.cuda.runtime.getDeviceProperties(0)
    print("Props:", props["name"])
except Exception as e:
    print("Props failed:", e)
try:
    x = cp.array([1, 2, 3])
    print("Array created:", x)
except Exception as e:
    print("Array creation failed:", e)
