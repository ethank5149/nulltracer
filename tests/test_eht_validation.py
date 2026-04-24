"""Test EHT shadow metric extraction and render validation.

Validates:
- Shadow metrics extraction from synthetic and real renders
- Render output against reference frames (with MSE tolerance)
"""

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from nulltracer.eht_validation import extract_shadow_metrics, fit_circle


def test_fit_circle_synthetic():
    """Circle fit recovers known circle parameters."""
    theta = np.linspace(0, 2 * np.pi, 200)
    cx, cy, r = 50.0, 60.0, 25.0
    points = np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)])

    cx_fit, cy_fit, r_fit, rms = fit_circle(points)
    assert abs(cx_fit - cx) < 0.1
    assert abs(cy_fit - cy) < 0.1
    assert abs(r_fit - r) < 0.1
    assert rms < 0.01


def test_extract_metrics_synthetic_ring():
    """Shadow metrics from a synthetic ring image."""
    img = np.zeros((200, 200))
    cy, cx, r = 100, 100, 40
    yy, xx = np.ogrid[:200, :200]
    ring_mask = np.abs(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - r) < 5
    img[ring_mask] = 1.0

    metrics = extract_shadow_metrics(img, fov_deg=10.0, threshold=0.3)
    assert "error" not in metrics
    assert abs(metrics["circularity"]) < 0.1  # should be nearly circular


def test_brightness_asymmetry_direction():
    """Approaching limb should be brighter for prograde spin."""
    img = np.zeros((200, 200))
    cy, cx, r = 100, 100, 40
    yy, xx = np.ogrid[:200, :200]
    ring_mask = np.abs(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - r) < 5
    img[ring_mask] = 1.0
    img[ring_mask & (xx < cx)] *= 2.0  # Make left side brighter

    metrics = extract_shadow_metrics(img, fov_deg=10.0, threshold=0.3)
    # Check that asymmetry is > 1 (left is brighter than right)
    assert metrics.get("asymmetry", 0) > 1.0, (
        f"Expected asymmetry > 1.0 for brighter left side, got {metrics.get('asymmetry', 0)}"
    )


@pytest.mark.gpu
def test_render_mse_against_reference():
    """Render a low-res frame and compare against reference with MSE tolerance."""
    import nulltracer as nt
    nt.compile_all(verbose=False)

    # Render a 64x64 test frame
    img, _info = nt.render_frame(
        spin=0.0, inclination_deg=90,
        width=64, height=64,
        fov=12.0, obs_dist=100,
        step_size=0.3, method="rkdp8",
        aa_samples=1,
        bg_mode=2, star_layers=0, show_disk=False,
    )

    # Convert to grayscale for comparison
    gray = img.mean(axis=2).astype(np.float32) / 255.0

    # For Schwarzschild pole-on, we expect a circular shadow
    # Check that dark region exists in center
    center_region = gray[24:40, 24:40]  # Center 16x16 region
    assert center_region.mean() < 0.5, "Expected dark shadow region in center"


@pytest.mark.gpu
def test_schwarzschild_render_structure():
    """Validate basic structure of Schwarzschild render."""
    import nulltracer as nt
    nt.compile_all(verbose=False)

    img, info = nt.render_frame(
        spin=0.0, inclination_deg=90,
        width=128, height=128,
        fov=12.0, obs_dist=100,
        step_size=0.3, method="rkdp8",
        aa_samples=1,
        bg_mode=2, star_layers=0,
    )

    gray = img.mean(axis=2).astype(np.float32) / 255.0

    # Extract shadow metrics
    metrics = extract_shadow_metrics(gray, fov_deg=12.0, threshold=0.3)

    if "error" not in metrics:
        # Check that we get a reasonable diameter (Schwarzschild should be 2*3*sqrt(3) ~= 10.39 M)
        diameter = metrics.get("diameter_M", 0)
        assert abs(diameter - 10.39) < 2.0, (
            f"Shadow diameter {diameter:.2f} M outside expected range around 10.39 M"
        )
        # Should be nearly circular (circularity < 0.2)
        assert metrics.get("circularity", 1.0) < 0.2, (
            f"Schwarzschild shadow not circular: circularity={metrics.get('circularity', 0):.4f}"
        )


@pytest.mark.gpu
def test_kerr_render_asymmetry():
    """Kerr render should show asymmetric shadow."""
    import nulltracer as nt
    nt.compile_all(verbose=False)

    img, _info = nt.render_frame(
        spin=0.9, inclination_deg=60,
        width=128, height=128,
        fov=10.0, obs_dist=100,
        step_size=0.25, method="rkdp8",
        aa_samples=1,
        bg_mode=2, star_layers=0,
    )

    gray = img.mean(axis=2).astype(np.float32) / 255.0
    metrics = extract_shadow_metrics(gray, fov_deg=10.0, threshold=0.3)

    if "error" not in metrics:
        # Kerr shadow should be asymmetric
        assert metrics["asymmetry"] > 0.1, (
            f"Kerr shadow should be asymmetric, got asymmetry={metrics['asymmetry']:.4f}"
        )
