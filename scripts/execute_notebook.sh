#!/usr/bin/env bash
# Execute the hero notebook on a CUDA-capable host and commit the result.
#
# This is the reproducibility capstone (P9) from the publication-readiness
# patch series. A portfolio notebook that claims to validate against the
# EHT measurements is not credible if no cell has been executed; running
# this script produces an in-place executed copy with all outputs, figures,
# and timing information baked in, and commits it.
#
# Prerequisites:
#   - NVIDIA GPU with CUDA toolkit available
#   - `pip install -e ".[analysis,notebook]"` already run
#   - nbconvert and a Jupyter kernel named "python3" installed
#
# Usage:
#   ./scripts/execute_notebook.sh               # execute + commit
#   ./scripts/execute_notebook.sh --no-commit   # execute only
#   ./scripts/execute_notebook.sh --timeout 3600 # custom cell timeout
#
# Safety:
#   - Refuses to run if the working tree is dirty (to avoid clobbering uncommitted changes).
#   - Never force-pushes. Produces a local commit only.
#   - Bails on any cell error (--ExecutePreprocessor.allow_errors=False is the default).
set -euo pipefail

TIMEOUT=1800          # 30-minute per-cell timeout — spin sweeps can be slow
COMMIT=1
NOTEBOOK="notebooks/nulltracer.ipynb"

while [ $# -gt 0 ]; do
    case "$1" in
        --no-commit) COMMIT=0; shift ;;
        --timeout)   TIMEOUT="$2"; shift 2 ;;
        --notebook)  NOTEBOOK="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -e/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# ── Preflight ─────────────────────────────────────────────────────────
if [ ! -f "$NOTEBOOK" ]; then
    echo "error: $NOTEBOOK not found. Run from repository root." >&2
    exit 1
fi

if ! command -v jupyter >/dev/null 2>&1; then
    echo "error: jupyter not on PATH. Install with: pip install -e \".[notebook]\"" >&2
    exit 1
fi

if ! python3 -c "import cupy; cupy.cuda.runtime.getDeviceCount()" 2>/dev/null; then
    echo "error: CuPy cannot see a CUDA device. Execution would fail." >&2
    echo "       Ensure CUDA toolkit is installed and nvidia-smi works." >&2
    exit 1
fi

if [ "$COMMIT" = "1" ] && [ -n "$(git status --porcelain)" ]; then
    echo "error: working tree is dirty. Commit or stash changes first," >&2
    echo "       or run with --no-commit to skip the commit step." >&2
    git status --short
    exit 1
fi

# ── Execute ───────────────────────────────────────────────────────────
echo "=== Executing $NOTEBOOK (timeout ${TIMEOUT}s per cell) ==="
python3 -c "import cupy; p = cupy.cuda.runtime.getDeviceProperties(0); \
    print('GPU:', p['name'].decode() if isinstance(p['name'], bytes) else p['name'])"

jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout="$TIMEOUT" \
    --ExecutePreprocessor.kernel_name=python3 \
    "$NOTEBOOK"

echo "=== Executed successfully ==="

# Copy the figures that the notebook wrote to the working directory
# into a docs/images/ folder so README can reference them.
mkdir -p docs/images
for f in m87_spin_sweep.png sgra_spin_sweep.png kerr_radii.png; do
    if [ -f "$f" ]; then
        mv -v "$f" "docs/images/$f"
    fi
done

# ── Commit ────────────────────────────────────────────────────────────
if [ "$COMMIT" = "1" ]; then
    git add -A "$NOTEBOOK" docs/images/ 2>/dev/null || true
    if git diff --cached --quiet; then
        echo "=== No changes to commit ==="
    else
        GPU_NAME=$(python3 -c "import cupy; p = cupy.cuda.runtime.getDeviceProperties(0); \
            print(p['name'].decode() if isinstance(p['name'], bytes) else p['name'])")
        git commit -m "notebooks: execute hero notebook on ${GPU_NAME}

Reproducibility capstone (P9). Executed $NOTEBOOK top-to-bottom with
all outputs, figures, and timing data baked in. Figures exported to
docs/images/ for README reference."
        echo "=== Committed ==="
        git log -1 --oneline
    fi
else
    echo "=== --no-commit: leaving executed notebook in working tree ==="
    git status --short
fi
