#!/usr/bin/env bash
# Run the full nulltracer test suite.
#
# GPU tests (@pytest.mark.gpu) include critical physics validation:
#   - Hamiltonian conservation across all integrators
#   - Carter constant and angular momentum conservation
#   - Shadow boundary accuracy
#   - Doppler asymmetry
#
# These tests MUST pass before any release.

set -e

echo "=== Running CPU-only tests ==="
pytest tests/ -m "not gpu" -v

echo ""
echo "=== Checking GPU availability ==="
if python -c "import cupy; cupy.cuda.Device(0).use()" 2>/dev/null; then
    echo "GPU available — running GPU physics tests"
    pytest tests/ -m "gpu" -v
    echo ""
    echo "=== All tests (CPU + GPU) passed ==="
else
    echo "⚠  GPU not available or cupy not configured."
    echo "⚠  SKIPPING GPU physics tests (Hamiltonian conservation, shadow boundary, etc.)"
    echo "⚠  These tests validate core physics — run on a GPU machine before releasing."
    echo ""
    # Count how many tests were skipped
    GPU_COUNT=$(pytest tests/ -m "gpu" --collect-only -q 2>/dev/null | tail -1 | grep -oP '\d+' | head -1 || echo "?")
    echo "⚠  $GPU_COUNT GPU tests were not run."
fi
