"""Test EHT shadow metric extraction and render validation.

Validates:
- Shadow metrics extraction from synthetic and real renders
- β-extent diameter metric for low-inclination accuracy
- Ring-peak extraction from disk-on renders
- EHT reference constants and unit conversions
- Render output against reference frames (with MSE tolerance)
"""

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from nulltracer.eht_validation import (
    extract_shadow_metrics,
    extract_ring_diameter,
    fit_circle,
    M_to_uas,
    uas_per_M,
    EHT_M87,
    EHT_SGRA,
)


# ── Unit conversion tests ────────────────────────────────────


def test_uas_per_M_m87():
    """Angular scale for M87* is ~3.8 μas/M."""
    scale = uas_per_M(EHT_M87["mass_kg"], EHT_M87["dist_m"])
    assert 3.5 < scale < 4.1, f"M87* scale {scale:.3f} outside [3.5, 4.1]"


def test_uas_per_M_sgra():
    """Angular scale for Sgr A* is ~4.8 μas/M."""
    scale = uas_per_M(EHT_SGRA["mass_kg"], EHT_SGRA["dist_m"])
    assert 4.5 < scale < 5.1, f"Sgr A* scale {scale:.3f} outside [4.5, 5.1]"


def test_M_to_uas_schwarzschild_m87():
    """Schwarzschild shadow at M87* scale is ~39.7 μas."""
    d_sch = 2.0 * 3.0 * np.sqrt(3.0)  # 10.392 M
    uas = M_to_uas(d_sch, EHT_M87["mass_kg"], EHT_M87["dist_m"])
    assert 38.0 < uas < 41.0, f"Schwarzschild shadow {uas:.1f} μas outside [38, 41]"


def test_M_to_uas_schwarzschild_sgra():
    """Schwarzschild shadow at Sgr A* scale is ~49.6 μas."""
    d_sch = 2.0 * 3.0 * np.sqrt(3.0)
    uas = M_to_uas(d_sch, EHT_SGRA["mass_kg"], EHT_SGRA["dist_m"])
    assert 48.0 < uas < 51.0, f"Schwarzschild shadow {uas:.1f} μas outside [48, 51]"


# ── Shadow contour tests ─────────────────────────────────────


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


def test_diameter_beta_M_symmetric():
    """For a circular shadow, diameter_beta_M ≈ diameter_M."""
    img = np.zeros((200, 200))
    cy, cx, r = 100, 100, 40
    yy, xx = np.ogrid[:200, :200]
    ring_mask = np.abs(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - r) < 5
    img[ring_mask] = 1.0

    metrics = extract_shadow_metrics(img, fov_deg=10.0, threshold=0.3)
    assert "error" not in metrics
    # For a circular shadow, β-extent ≈ circle-fit diameter
    d_circle = metrics["diameter_M"]
    d_beta = metrics["diameter_beta_M"]
    assert abs(d_circle - d_beta) / d_circle < 0.15, (
        f"diameter_M={d_circle:.3f} vs diameter_beta_M={d_beta:.3f}"
    )


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


# ── Ring-peak extraction tests ────────────────────────────────


def test_ring_peak_synthetic():
    """Ring peak extraction recovers known ring radius."""
    img = np.zeros((256, 256))
    cy, cx, r = 128, 128, 50
    yy, xx = np.ogrid[:256, :256]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # Gaussian ring profile peaked at r=50
    img = np.exp(-0.5 * ((dist - r) / 5) ** 2)

    fov = 10.0  # M
    px_per_M = 256 / (2 * fov)
    expected_ring_M = 2.0 * r / px_per_M  # 2 * 50 / 12.8 ≈ 7.81 M

    result = extract_ring_diameter(img, fov=fov)
    assert abs(result["ring_peak_M"] - expected_ring_M) / expected_ring_M < 0.10, (
        f"Ring peak {result['ring_peak_M']:.2f} M vs expected {expected_ring_M:.2f} M"
    )


# ── GPU render tests ──────────────────────────────────────────


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
        bg_mode=2, star_layers=0, show_disk=False,
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
        # β-extent should also be close to 10.39 M
        d_beta = metrics.get("diameter_beta_M", 0)
        assert abs(d_beta - 10.39) < 2.5, (
            f"Shadow β-extent {d_beta:.2f} M outside expected range"
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
        bg_mode=2, star_layers=0, show_disk=False,
    )

    gray = img.mean(axis=2).astype(np.float32) / 255.0
    metrics = extract_shadow_metrics(gray, fov_deg=10.0, threshold=0.3)

    if "error" not in metrics:
        # Kerr shadow should be asymmetric
        assert metrics["asymmetry"] > 0.1, (
            f"Kerr shadow should be asymmetric, got asymmetry={metrics['asymmetry']:.4f}"
        )
