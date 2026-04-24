import pytest
import numpy as np

def dump_h():
    from nulltracer.renderer import CudaRenderer
    renderer = CudaRenderer()
    renderer.initialize()

    from nulltracer.ray import trace_ray
    res = trace_ray(
        method="rk4",
        steps=1000,
        step_size=0.1,
        alpha=0.0,
        beta=5.0,
    )
    print("rk4 init state:", res["initial_state"])
    print("rk4 final state:", res["final_state"])

dump_h()
