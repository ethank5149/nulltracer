"""Shared test configuration.

GPU tests (marked with @pytest.mark.gpu) require a CUDA GPU and cupy.
When no GPU is available, these tests are skipped with a visible warning.

To run all tests including GPU:
    pytest tests/ -v

To run only CPU tests:
    pytest tests/ -m "not gpu" -v

To run only GPU tests:
    pytest tests/ -m gpu -v
"""
import pytest
import sys
import warnings


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: marks tests that require a GPU")


def pytest_collection_modifyitems(config, items):
    """Report how many GPU tests are in the collection."""
    gpu_tests = [item for item in items if "gpu" in item.keywords]
    non_gpu_tests = [item for item in items if "gpu" not in item.keywords]

    # Check if GPU tests are being skipped by marker selection
    marker_expr = config.getoption("-m", default="")
    if "not gpu" in str(marker_expr):
        if gpu_tests:
            warnings.warn(
                f"\n⚠ Skipping {len(gpu_tests)} GPU tests "
                f"(running {len(non_gpu_tests)} CPU-only tests). "
                f"Run without '-m \"not gpu\"' to include physics validation tests.",
                stacklevel=1,
            )


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
