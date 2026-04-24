import pytest
import numpy as np

def dump_h():
    from nulltracer.renderer import CudaRenderer
    renderer = CudaRenderer()
    renderer.initialize()

    from nulltracer.ray import trace_ray
    res = trace_ray(
        method="symplectic8",
        steps=2000,
        step_size=0.1,
        alpha=0.0,
        beta=5.0,
    )
    print("tao init state:", res["initial_state"])
    print("tao final state:", res["final_state"])
    print("tao term:", res["termination"])

dump_h()
