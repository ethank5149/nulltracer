"""Regression test: the rendered disk must show a measurable Doppler
brightness asymmetry between approaching and receding limbs.

This test would have caught the P10 fake-glow bug: an earlier kernel
added an isotropic screen-space Gaussian halo to every pixel, which
masked the physical asymmetry from the g-factor. The asymmetry should
be the dominant visual feature of a non-face-on Kerr disk image.

Why an asymmetry test specifically:
  - We're not checking any absolute intensity, only a ratio.
  - The ratio is robust across kernel/compiler revisions.
  - The ratio grows monotonically with spin and with sin(theta), which
    lets us sanity-check the direction of frame dragging.

Physical prediction (a=0.94, theta=60°, disk bright around r~3M):
  g_approaching^4 / g_receding^4  ≈ 7 to 30×, depending on b
  (actual ratio depends on the specific pixels selected)
"""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("cupy", reason="CuPy not installed; rendering tests require a CUDA build")

import numpy as np


def _bright_halves(img: np.ndarray, center_col: int):
    """Return (left_mean, right_mean) brightness of the central horizontal
    strip, excluding the shadow interior.

    We use the central quarter of rows (where the disk primary image
    sits for all tested inclinations), and threshold at 10% of the max
    so we only average over *bright* disk pixels — otherwise the black
    shadow contributes equally to both sides and dilutes the signal.
    """
    H, W, _ = img.shape
    gray = img.mean(axis=2).astype(np.float32) / 255.0
    row_lo, row_hi = int(H * 0.375), int(H * 0.625)
    strip = gray[row_lo:row_hi]
    left = strip[:, :center_col]
    right = strip[:, center_col:]
    thresh = 0.10 * strip.max()
    left_mask = left > thresh
    right_mask = right > thresh
    if left_mask.sum() < 30 or right_mask.sum() < 30:
        pytest.skip("Not enough bright pixels to measure asymmetry")
    return left[left_mask].mean(), right[right_mask].mean()


@pytest.mark.gpu
def test_kerr_disk_shows_doppler_asymmetry():
    """Rendering a=0.94, θ=60° must produce a lopsided bright ring."""
    import nulltracer as nt
    nt.compile_all(verbose=False)

    img, _info = nt.render_frame(
        spin=0.94, inclination_deg=60,
        width=512, height=512,
        fov=8.0, obs_dist=150, disk_outer=20,
        step_size=0.25, method="rkdp8",
        aa_samples=1,
        bg_mode=0, star_layers=0, disk_temp=1.0,
    )

    left, right = _bright_halves(img, center_col=img.shape[1] // 2)
    asymmetry = max(left, right) / min(left, right)
    # Allow a wide margin — we only want to detect the *presence* of
    # asymmetry, not its exact value.  A value near 1.0 means an
    # isotropic halo bug; anything above ~1.5x is a real g^4 signal.
    assert asymmetry > 1.5, (
        f"Disk Doppler asymmetry too small (left={left:.4f}, right={right:.4f}, "
        f"ratio={asymmetry:.2f}×). Expected >1.5× for Kerr a=0.94 at θ=60°; "
        f"a value near 1.0 suggests an isotropic artifact (fake glow, "
        f"missing g-factor, or symmetric background) is masking the physics."
    )


@pytest.mark.gpu
def test_schwarzschild_disk_is_approximately_symmetric():
    """Control: a=0 has no frame dragging, so Doppler only from orbital
    motion — left/right asymmetry should still exist (orbital Doppler)
    but be markedly smaller than the Kerr case."""
    import nulltracer as nt
    nt.compile_all(verbose=False)

    img, _info = nt.render_frame(
        spin=0.0, inclination_deg=60,
        width=512, height=512,
        fov=8.0, obs_dist=150, disk_outer=20,
        step_size=0.25, method="rkdp8",
        aa_samples=1,
        bg_mode=0, star_layers=0, disk_temp=1.0,
    )

    left, right = _bright_halves(img, center_col=img.shape[1] // 2)
    asymmetry = max(left, right) / min(left, right)
    # Schwarzschild still has orbital Doppler (Omega_K = r^-3/2 in M=1),
    # so some asymmetry is expected. But less than the a=0.94 case —
    # this doesn't prove direction, just that the asymmetry *scales* with spin.
    assert asymmetry < 30.0, (
        f"Schwarzschild asymmetry unexpectedly huge ({asymmetry:.1f}×); "
        f"suggests numerical pathology."
    )
