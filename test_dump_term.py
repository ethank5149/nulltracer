import pytest
import numpy as np

def dump_term():
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
    print("term reason:", res["termination"])

dump_term()
