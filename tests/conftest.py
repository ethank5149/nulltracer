"""Shared test configuration."""
import pytest
import sys

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: marks tests that require a GPU")

@pytest.fixture(scope="session")
def cuda_renderer():
    try:
        import cupy as cp
        cp.cuda.Device(0).use()
        from nulltracer.renderer import CudaRenderer
        renderer = CudaRenderer()
        renderer.initialize()
        return renderer
    except Exception as e:
        pytest.skip(f"GPU not available or cupy not installed: {e}")

