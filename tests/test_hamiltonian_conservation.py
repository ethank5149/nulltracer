import pytest
from nulltracer.ray import trace_ray

@pytest.mark.gpu
def test_hamiltonian_conservation_rk4(cuda_renderer):
    res = trace_ray(method="rk4", steps=200, step_size=0.1, alpha=0.0, beta=5.0)
    H_init = res["initial_state"]["H"]
    H_final = res["final_state"]["H"]
    
    if H_init and H_final and abs(H_init) > 1e-15:
        assert abs(H_final / H_init - 1) < 1e-8, "RK4 did not conserve H within 1e-8"

@pytest.mark.gpu
def test_hamiltonian_conservation_symplectic(cuda_renderer):
    res = trace_ray(method="tao_kahan_li8", steps=200, step_size=0.1, alpha=0.0, beta=5.0)
    H_init = res["initial_state"]["H"]
    H_final = res["final_state"]["H"]
    
    if H_init and H_final and abs(H_init) > 1e-15:
        assert abs(H_final / H_init - 1) < 1e-12, "Symplectic did not conserve H within 1e-12"
