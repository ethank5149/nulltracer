import pytest
import numpy as np

def dump_h():
    from nulltracer.renderer import CudaRenderer
    renderer = CudaRenderer()
    renderer.initialize()

    from nulltracer.ray import trace_ray
    max_diff = 0
    for i in range(100):
        alpha = np.random.uniform(-2.0, 2.0)
        beta = np.random.uniform(3.0, 8.0)
        res = trace_ray(
            method="tao_kahan_li8",
            steps=1000,
            step_size=0.1,
            mode="impact_parameter",
            alpha=alpha,
            beta=beta,
        )
        if res["termination"]["reason"] not in ["horizon", "nan"]:
            diff = abs(res["final_state"]["H"] - res["initial_state"]["H"])
            if diff > max_diff:
                max_diff = diff
                print(f"New max diff: {max_diff} | Reason: {res['termination']['reason']}")
                print(f"alpha={alpha}, beta={beta}")
                print(f"init={res['initial_state']['H']}, final={res['final_state']['H']}")

dump_h()
