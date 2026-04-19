"""Test EHT shadow metric extraction."""

import numpy as np
import pytest

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
