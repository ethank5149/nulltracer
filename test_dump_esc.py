import pytest
import numpy as np

def dump_esc():
    from nulltracer.renderer import CudaRenderer
    renderer = CudaRenderer()
    renderer.initialize()

    from nulltracer.ray import trace_ray
    res = trace_ray(
        method="rk4",
        steps=1000,
        step_size=0.1,
        alpha=10.0,
        beta=10.0,
    )
    print("rk4 esc init:", res["initial_state"])
    print("rk4 esc final:", res["final_state"])
    print("rk4 esc term:", res["termination"])

dump_esc()
