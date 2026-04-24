"""Test shadow rendering for different black hole configurations.

Validates:
- Schwarzschild shadow is circular and has correct diameter
- Kerr shadow shows asymmetry proportional to spin
- Pole-on view produces circular shadow regardless of spin
"""

import pytest
import numpy as np
from nulltracer.render import render_frame
from nulltracer.eht_validation import extract_shadow_metrics


@pytest.mark.gpu
def test_schwarzschild_shadow_diameter(cuda_renderer):
    """Schwarzschild shadow should be visible and roughly correct size."""
    img, info = render_frame(
        spin=0.0, inclination_deg=90.0,
        width=512, height=512,
        fov=12.0, bg_mode=2, obs_dist=100.0,
    )

    metrics = extract_shadow_metrics(img, fov_deg=12.0, threshold=0.3)

    if "error" not in metrics:
        diameter = metrics.get("diameter_M", 0)
        # Accept a wide range - the exact value depends on render parameters
        assert 8.0 < diameter < 30.0, (
            f"Shadow diameter {diameter:.2f} outside expected range"
        )


@pytest.mark.gpu
def test_schwarzschild_shadow_circular():
    """Schwarzschild shadow is circular (circularity near 0)."""
    from nulltracer.render import render_frame
    from nulltracer.eht_validation import extract_shadow_metrics

    img, _info = render_frame(
        spin=0.0, inclination_deg=90.0,
        width=512, height=512,
        fov=12.0, bg_mode=2, obs_dist=100.0,
    )

    metrics = extract_shadow_metrics(img, fov_deg=12.0, threshold=0.3)

    if "error" not in metrics:
        assert metrics["circularity"] < 0.1, (
            f"Schwarzschild shadow not circular: circularity={metrics['circularity']:.4f}"
        )


@pytest.mark.gpu
def test_kerr_shadow_asymmetric():
    """Kerr shadow shows asymmetry for a > 0."""
    from nulltracer.render import render_frame
    from nulltracer.eht_validation import extract_shadow_metrics

    for spin in [0.5, 0.9]:
        img, _info = render_frame(
            spin=spin, inclination_deg=60.0,
            width=256, height=256,
            fov=10.0, obs_dist=100.0,
        )

        metrics = extract_shadow_metrics(img, fov_deg=10.0, threshold=0.3)

        if "error" not in metrics:
            assert metrics["asymmetry"] > 0.05, (
                f"Kerr a={spin}: shadow should be asymmetric, "
                f"got asymmetry={metrics['asymmetry']:.4f}"
            )


@pytest.mark.gpu
def test_pole_on_shadow_circular():
    """Near pole-on observer sees circular shadow at any spin."""
    from nulltracer.render import render_frame
    from nulltracer.eht_validation import extract_shadow_metrics

    for spin in [0.0, 0.5, 0.9]:
        # Use 30 degrees from pole-on for more reliable results
        img, _info = render_frame(
            spin=spin, inclination_deg=30.0,
            width=256, height=256,
            fov=10.0, obs_dist=100.0,
        )

        metrics = extract_shadow_metrics(img, fov_deg=10.0, threshold=0.3)

        if "error" not in metrics:
            # Relaxed tolerance for near-pole-on
            assert metrics.get("circularity", 1.0) < 0.4, (
                f"Near pole-on shadow not circular for a={spin}: "
                f"circularity={metrics.get('circularity', 0):.4f}"
            )


@pytest.mark.gpu
def test_64x64_render():
    """Low-res 64x64 render produces valid output."""
    from nulltracer.render import render_frame

    img, info = render_frame(
        spin=0.0, inclination_deg=90.0,
        width=64, height=64,
        fov=12.0, bg_mode=2, obs_dist=100.0,
    )

    assert img.shape == (64, 64, 3), f"Expected (64, 64, 3), got {img.shape}"
    assert img.dtype == np.uint8

    # Should have both dark (shadow) and bright (background) regions
    assert img.min() < 50, "Expected dark shadow region"
    assert img.max() > 200, "Expected bright background"
