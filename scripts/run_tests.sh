#!/usr/bin/env bash

set -e

echo "Running CPU-only tests..."
pytest tests/ -m "not gpu" -v

echo ""
echo "Attempting to run GPU tests..."
if python -c "import cupy; cupy.cuda.Device(0).use()" 2>/dev/null; then
    pytest tests/ -m "gpu" -v
else
    echo "GPU not available or cupy not configured properly. Skipping GPU tests."
fi
