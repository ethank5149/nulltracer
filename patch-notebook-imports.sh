#!/usr/bin/env bash
# patch-notebook-imports.sh — Update notebook imports after migration
#
# Run from the repo root after migrate.sh:
#   bash patch-notebook-imports.sh
#
# This patches notebooks/nulltracer.ipynb to use the new package API.
# It's conservative — only changes import lines and known function calls.
# Review the diff before committing.

set -euo pipefail

NB="notebooks/nulltracer.ipynb"

if [ ! -f "$NB" ]; then
    echo "ERROR: $NB not found. Run migrate.sh first."
    exit 1
fi

echo "Patching $NB..."

# Back up
cp "$NB" "${NB}.bak"

# ── Import replacements ──────────────────────────────────────
# Old: from nulltracer_kernels import compile_all, render_kerr, classify_kerr
# Old: from nulltracer_kernels import compare_integrators, METHODS
# Old: from nulltracer_kernels import load_skymap, isco_kerr
# New: import nulltracer as nt

python3 << 'PYEOF'
import json, re, sys

nb_path = "notebooks/nulltracer.ipynb"
with open(nb_path) as f:
    nb = json.load(f)

# Replacement map for function calls
replacements = {
    # imports → single import
    r'from nulltracer_kernels import .*':
        'import nulltracer as nt',
    # function renames
    r'\bcompile_all\b':         'nt.compile_all',
    r'\brender_kerr\b':         'nt.render_frame',
    r'\bclassify_kerr\b':       'nt.classify_shadow',
    r'\bcompare_integrators\b': 'nt.compare_integrators',
    r'\bload_skymap\b':         'nt.load_skymap',
    r'\bisco_kerr\b':           'nt.isco_kerr',
    r'\bshadow_boundary\b':     'nt.shadow_boundary',
    r'\bfit_ellipse_to_shadow\b': 'nt.fit_ellipse_to_shadow',
    r'\bauto_steps\b':          'nt.auto_steps',
    # dict key renames (RenderInfo is now a dataclass)
    # info['render_ms'] → info.render_ms
    # info['max_steps'] → info.max_steps
    r"info\['render_ms'\]":     'info.render_ms',
    r"info\['max_steps'\]":     'info.max_steps',
    r"info\['obs_dist'\]":      'info.obs_dist',
    r"info\['method'\]":        'info.method',
    # METHODS dict → nt._kernel_utils.KernelCache.METHOD_LABELS
    # (or just inline the labels)
    r'\bMETHODS\b':             'nt._kernel_utils.KernelCache.METHOD_LABELS',
}

changed = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    new_source = []
    for line in cell['source']:
        original = line
        for pattern, replacement in replacements.items():
            line = re.sub(pattern, replacement, line)
        if line != original:
            changed += 1
        new_source.append(line)
    cell['source'] = new_source

# Remove duplicate nt imports (multiple old import lines → one)
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    lines = cell['source']
    seen_import = False
    filtered = []
    for line in lines:
        if line.strip() == 'import nulltracer as nt':
            if seen_import:
                continue  # skip duplicate
            seen_import = True
        filtered.append(line)
    cell['source'] = filtered

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Patched {changed} lines in {nb_path}")
print("Review with: git diff notebooks/nulltracer.ipynb")
PYEOF

echo "Done. Original backed up to ${NB}.bak"
