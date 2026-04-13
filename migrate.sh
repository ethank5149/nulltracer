#!/usr/bin/env bash
# migrate.sh — Restructure nulltracer from server architecture to Python package
#
# Run from the repo root:
#   chmod +x migrate.sh && ./migrate.sh
#
# This script:
#   1. Moves CUDA kernels from server/kernels/ → nulltracer/kernels/
#   2. Creates the nulltracer/ Python package from the refactored files
#   3. Moves large data files to data/
#   4. Archives the old server/js/Docker infrastructure to legacy/
#   5. Moves notebooks to notebooks/
#
# After running, do:
#   pip install -e .
#   jupyter lab notebooks/nulltracer.ipynb

set -euo pipefail

echo "=== Nulltracer Migration: Server → Python Package ==="
echo ""

# ── Sanity check ──────────────────────────────────────────────
if [ ! -d "server/kernels" ]; then
    echo "ERROR: Must be run from the nulltracer repo root (server/kernels/ not found)"
    exit 1
fi

if [ -d "nulltracer/kernels" ]; then
    echo "ERROR: nulltracer/kernels/ already exists. Migration may have already run."
    exit 1
fi

# ── 1. Create package directory structure ─────────────────────
echo "[1/6] Creating nulltracer/ package structure..."
mkdir -p nulltracer/kernels/integrators
mkdir -p data
mkdir -p legacy
mkdir -p notebooks

# ── 2. Move CUDA kernels (the physics — untouched) ───────────
echo "[2/6] Moving CUDA kernels to nulltracer/kernels/..."
git mv server/kernels/geodesic_base.cu  nulltracer/kernels/
git mv server/kernels/backgrounds.cu    nulltracer/kernels/
git mv server/kernels/disk.cu           nulltracer/kernels/
git mv server/kernels/ray_trace.cu      nulltracer/kernels/

for f in server/kernels/integrators/*.cu; do
    git mv "$f" nulltracer/kernels/integrators/
done

# ── 3. Move data files ───────────────────────────────────────
echo "[3/6] Moving data files to data/..."
for f in starmap_*.jpg starmap_*.png starmap_*.exr; do
    [ -f "$f" ] && git mv "$f" "data/$f"
done
# Reference images
for f in *.png *.jpg; do
    # Skip the icon
    [[ "$f" == nulltracer-icon* ]] && continue
    [ -f "$f" ] && git mv "$f" "data/$f"
done

# ── 4. Archive legacy server/client infrastructure ────────────
echo "[4/6] Archiving server + client to legacy/..."
git mv server legacy/server
git mv js legacy/js
git mv index.html legacy/
git mv styles.css legacy/
git mv bench.html legacy/
[ -f docker-compose.yml ] && git mv docker-compose.yml legacy/
[ -f deploy.sh ] && git mv deploy.sh legacy/
[ -f nulltracer-renderer.xml ] && git mv nulltracer-renderer.xml legacy/
[ -f Caddyfile ] && git mv Caddyfile legacy/ 2>/dev/null || true
[ -f Caddyfile.current ] && git mv Caddyfile.current legacy/ 2>/dev/null || true
[ -d .forgejo ] && git mv .forgejo legacy/ 2>/dev/null || true

# ── 5. Move notebooks ────────────────────────────────────────
echo "[5/6] Moving notebooks..."
[ -f nulltracer.ipynb ] && git mv nulltracer.ipynb notebooks/
[ -f nulltracer.copy.ipynb ] && git mv nulltracer.copy.ipynb notebooks/
[ -f eht_comparison_final.ipynb ] && git mv eht_comparison_final.ipynb notebooks/
[ -f eht_comparison_gpu.ipynb ] && git mv eht_comparison_gpu.ipynb notebooks/

# ── 6. Remove superseded files ────────────────────────────────
echo "[6/6] Cleaning up superseded files..."
# The old standalone kernel module is replaced by the package
[ -f nulltracer_kernels.py ] && git rm nulltracer_kernels.py

# Planning docs that are now stale
[ -f ARCHITECTURE.md ] && git mv ARCHITECTURE.md legacy/
[ -f FEATURE-PLAN.md ] && git mv FEATURE-PLAN.md legacy/
[ -f EHT_Portfolio_Project_Plan.md ] && git mv EHT_Portfolio_Project_Plan.md legacy/

echo ""
echo "=== Migration complete ==="
echo ""
echo "Next steps:"
echo "  1. Copy the new Python files into nulltracer/:"
echo "     (they are provided separately as the refactored package)"
echo ""
echo "       nulltracer/__init__.py"
echo "       nulltracer/_kernel_utils.py"
echo "       nulltracer/_params.py"
echo "       nulltracer/isco.py"
echo "       nulltracer/bloom.py"
echo "       nulltracer/render.py"
echo "       nulltracer/ray.py"
echo "       nulltracer/skymap.py"
echo "       nulltracer/compare.py"
echo ""
echo "  2. Copy pyproject.toml to repo root"
echo "  3. Install in dev mode:  pip install -e ."
echo "  4. Open notebooks/nulltracer.ipynb"
echo "  5. Replace 'from nulltracer_kernels import ...' with 'import nulltracer as nt'"
echo "  6. git add -A && git commit -m 'refactor: server → Python package'"
