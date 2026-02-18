#!/usr/bin/env python3
"""
Kerr-Newman black hole icon generator.

Produces a physically accurate rendering of a Kerr-Newman black hole with
accretion disk using full Hamiltonian ray tracing (RK4 integration in
Boyer-Lindquist coordinates). The output is a PNG image suitable for use
as an application icon.

This is a standalone CPU ray tracer — no GPU required. It uses the same
physics as the server's GLSL shader but runs in pure Python/NumPy.

Usage:
    python3 generate_icon.py [--size 512] [--output icon.png] [--svg icon.svg]

Parameters are chosen to showcase the most striking Kerr-Newman features:
  - High spin (a=0.95): dramatic frame dragging, asymmetric shadow
  - Moderate charge (Q=0.3): visible effect on ISCO and shadow size
  - 80° inclination: shows disk structure clearly with visible warping
  - Observer at r=30: close enough for strong lensing effects
"""

import argparse
import base64
import io
import math
import sys

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


# ── Physics parameters ──────────────────────────────────────────────────

# Spin parameter (0=Schwarzschild, 1=extremal Kerr)
# a=0.95 gives dramatic frame dragging: the shadow becomes D-shaped,
# the prograde ISCO shrinks to ~1.94M, and the approaching side of the
# disk is strongly Doppler-boosted.
SPIN = 0.95

# Charge parameter (0=Kerr, sqrt(1-a²)=extremal Kerr-Newman)
# Q=0.3 is moderate — enough to visibly shrink the shadow and shift
# the ISCO compared to pure Kerr, but not so large as to eliminate
# the ergosphere. The maximum for a=0.95 is Q≈0.312.
CHARGE = 0.3

# Observer inclination from spin axis (degrees)
# 75° gives a good balance: the disk is clearly visible with strong
# Doppler asymmetry, the shadow is prominent, and the lensed back-image
# wraps visibly above the shadow. Not so edge-on that the shadow is hidden.
INCLINATION_DEG = 75.0

# Observer distance (in units of M)
# r=40M gives a wider view with the full disk and shadow visible,
# with moderate gravitational lensing effects.
OBS_DIST = 40.0

# Field of view (impact parameter scale, NOT an angle in radians).
# This maps screen coordinates [-1,1] to impact parameters in units of M.
# With FOV=10, the screen edge maps to b ≈ 10*sin(incl) ≈ 9.7M,
# which is well outside the critical impact parameter (~5M for Kerr a=0.95).
# The disk at r=14M is visible at impact parameter ~14M, so FOV=12
# frames the inner disk and shadow nicely with the disk filling ~85%.
FOV = 12.0

# Accretion disk outer radius
DISK_OUTER = 14.0

# Disk temperature parameter (controls overall brightness)
DISK_TEMP = 1.0

# Integration parameters
STEPS = 800       # More steps needed for adaptive stepping near horizon
STEP_SIZE = 0.30  # Base step size (adaptive near horizon)

# Regularization for sin²θ (prevents pole divergence)
S2_EPS = 0.0004

# Star field parameters
STAR_LAYERS = 3


# ── Derived quantities ──────────────────────────────────────────────────

A2 = SPIN * SPIN
Q2 = CHARGE * CHARGE
INCL = INCLINATION_DEG * math.pi / 180.0
SIN_I = math.sin(INCL)
COS_I = math.cos(INCL)

# Event horizon (outer)
R_HORIZON = 1.0 + math.sqrt(max(1.0 - A2 - Q2, 0.0))

# Escape radius (must be > OBS_DIST for rays to "escape")
R_ESCAPE = OBS_DIST + 15.0


def compute_isco(a: float, Q: float) -> float:
    """Compute prograde ISCO radius for Kerr-Newman via bisection on dE/dr=0.

    Uses the same method as server/isco.py: computes the energy of circular
    equatorial orbits and finds where dE/dr changes sign (the ISCO).
    """
    a2 = a * a
    Q2 = Q * Q
    rh = 1.0 + math.sqrt(max(1.0 - a2 - Q2, 1e-12))

    def energy(r: float) -> float:
        """Energy of a circular orbit at radius r (equatorial plane)."""
        r2 = r * r
        delta = r2 - 2.0 * r + a2 + Q2
        if delta <= 0:
            return float("nan")

        gtt = -(1.0 - (2.0 * r - Q2) / r2)
        gtph = -a * (2.0 * r - Q2) / r2
        gphph = (r2 * r2 + a2 * r2 + a2 * (2.0 * r - Q2)) / r2

        dgtt = 2.0 * (Q2 - r) / (r2 * r)
        dgtph = 2.0 * a * (r - Q2) / (r * r2)
        dgphph = 2.0 * r + a2 * (-2.0 / r2 + 2.0 * Q2 / (r2 * r))

        disc = dgtph * dgtph - dgtt * dgphph
        if disc < 0:
            return float("nan")
        Om = (-dgtph + math.sqrt(disc)) / dgphph  # prograde

        denom = -(gtt + 2.0 * gtph * Om + gphph * Om * Om)
        if denom <= 0:
            return float("nan")
        ut = 1.0 / math.sqrt(denom)
        return -(gtt + gtph * Om) * ut

    dr = 1e-5

    def dEdr(r: float) -> float:
        Ep = energy(r + dr)
        Em = energy(r - dr)
        if math.isnan(Ep) or math.isnan(Em):
            return float("nan")
        return (Ep - Em) / (2.0 * dr)

    lo = rh + 0.01
    hi = 9.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        d = dEdr(mid)
        if math.isnan(d) or d < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


R_ISCO = compute_isco(SPIN, CHARGE)


# ── Ray tracing ─────────────────────────────────────────────────────────

def geodesic_rhs(r, th, pr, pth, b):
    """Compute RHS of Kerr-Newman geodesic equations (Hamiltonian form)."""
    sth = math.sin(th)
    cth = math.cos(th)
    s2 = sth * sth + S2_EPS
    c2 = cth * cth
    r2 = r * r

    sig = r2 + A2 * c2
    delta = r2 - 2.0 * r + A2 + Q2
    sdel = max(delta, 1e-6)
    rpa2 = r2 + A2
    w = 2.0 * r - Q2  # KN cross-term: w = 2Mr - Q² (M=1)
    A_ = rpa2 * rpa2 - sdel * A2 * s2

    isig = 1.0 / sig
    SD = sig * sdel
    iSD = 1.0 / SD
    is2 = 1.0 / s2

    grr = sdel * isig
    gthth = isig
    gff = (sig - w) * iSD * is2
    gtf = -SPIN * w * iSD

    dr = grr * pr
    dth = gthth * pth
    dphi = gff * b - gtf

    # ∂H/∂r
    dsig_r = 2.0 * r
    ddel_r = 2.0 * r - 2.0
    dA_r = 4.0 * r * rpa2 - ddel_r * A2 * s2
    dSD_r = dsig_r * sdel + sig * ddel_r
    dgtt_r = -(dA_r * SD - A_ * dSD_r) / (SD * SD)
    dgtf_r = -SPIN * (2.0 * SD - w * dSD_r) / (SD * SD)
    dgrr_r = (ddel_r * sig - sdel * dsig_r) / (sig * sig)
    dgthth_r = -dsig_r * isig * isig
    num_ff = sig - w
    den_ff = SD * s2
    dgff_r = ((dsig_r - 2.0) * den_ff - num_ff * dSD_r * s2) / (den_ff * den_ff)
    dpr = -0.5 * (dgtt_r - 2.0 * b * dgtf_r + dgrr_r * pr * pr
                   + dgthth_r * pth * pth + dgff_r * b * b)

    # ∂H/∂θ
    dsig_th = -2.0 * A2 * sth * cth
    ds2_th = 2.0 * sth * cth
    dA_th = -sdel * A2 * ds2_th
    dSD_th = dsig_th * sdel
    dgtt_th = -(dA_th * SD - A_ * dSD_th) / (SD * SD)
    dgtf_th = SPIN * w * dSD_th / (SD * SD)
    dgrr_th = -sdel * dsig_th / (sig * sig)
    dgthth_th = -dsig_th * isig * isig
    dgff_th = (dsig_th * den_ff - num_ff * (dsig_th * sdel * s2 + SD * ds2_th)) / (den_ff * den_ff)
    dpth = -0.5 * (dgtt_th - 2.0 * b * dgtf_th + dgrr_th * pr * pr
                    + dgthth_th * pth * pth + dgff_th * b * b)

    return dr, dth, dphi, dpr, dpth


def hash2d(px: float, py: float) -> float:
    """Simple hash function for procedural star field."""
    px = (px * 443.8) % 1.0
    py = (py * 441.4) % 1.0
    px += px * py + 19.19
    py += px * py + 19.19
    return (px * py) % 1.0


def background_stars(th: float, phi: float) -> tuple:
    """Generate a starfield background from ray direction."""
    sth = math.sin(th)
    dx = sth * math.cos(phi)
    dy = sth * math.sin(phi)
    dz = math.cos(th)

    # Milky Way band (bright near equator)
    mw = math.exp(-8.0 * dz * dz)

    # Base deep space color (very dark blue-black)
    cr = 0.005 + 0.012 * mw
    cg = 0.006 + 0.010 * mw
    cb = 0.014 + 0.022 * mw

    # Cube-map projection for stars
    ax, ay, az = abs(dx), abs(dy), abs(dz)
    if az >= ax and az >= ay:
        face = 0.0 if dz > 0 else 1.0
        u, v = dx / max(az, 1e-10), dy / max(az, 1e-10)
    elif ax >= ay:
        face = 2.0 if dx > 0 else 3.0
        u, v = dy / max(ax, 1e-10), dz / max(ax, 1e-10)
    else:
        face = 4.0 if dy > 0 else 5.0
        u, v = dx / max(ay, 1e-10), dz / max(ay, 1e-10)

    fu = u * 0.5 + 0.5
    fv = v * 0.5 + 0.5

    # Stars — multiple layers for depth
    for L in range(STAR_LAYERS):
        sc = 12.0 + L * 16.0
        cx = math.floor(fu * sc)
        cy = math.floor(fv * sc)
        seed_x = (cx + face * 100 + L * 47) / 1000.0
        seed_y = (cy + face * 100 + L * 47) / 1000.0
        h = hash2d(seed_x, seed_y)
        if h > 0.85:
            sp_x = (cx + 0.3 + 0.4 * hash2d(seed_x + 0.5, seed_y)) / sc
            sp_y = (cy + 0.3 + 0.4 * hash2d(seed_x, seed_y + 1.5)) / sc
            dist = math.sqrt((fu - sp_x) ** 2 + (fv - sp_y) ** 2) * sc
            s = math.exp(-dist * dist * 6.0)
            t = hash2d(seed_x + 77, seed_y)
            if t < 0.2:
                sc2 = (1, 0.7, 0.4)      # warm red/orange star
            elif t < 0.55:
                sc2 = (1, 0.95, 0.85)     # white star
            elif t < 0.8:
                sc2 = (0.8, 0.9, 1)       # blue-white star
            else:
                sc2 = (0.6, 0.75, 1)      # blue star
            brightness = s * (0.5 + 2.5 * hash2d(seed_x + 33, seed_y))
            cr += sc2[0] * brightness
            cg += sc2[1] * brightness
            cb += sc2[2] * brightness

    return cr, cg, cb


def disk_color(r: float, phi: float) -> tuple:
    """Compute accretion disk emission at (r, phi) in equatorial plane."""
    ri = R_ISCO
    if r < ri * 0.85 or r > DISK_OUTER:
        return 0.0, 0.0, 0.0

    x = r / ri
    tp = x ** (-0.75) * DISK_TEMP
    I = tp ** 4 / (r * 0.3)

    # Smooth edges
    I *= min(1.0, max(0.0, (r - ri * 0.85) / (ri * 0.45)))
    I *= min(1.0, max(0.0, (DISK_OUTER - r) / (DISK_OUTER * 0.45)))

    # Doppler boosting
    vo = 1.0 / math.sqrt(r)
    dop = 1.0 + 0.65 * vo * math.sin(phi)
    boost = max(dop, 0.1) ** 3
    I *= boost

    # Temperature-based color (blackbody approximation)
    t = min(3.5, tp * boost * 0.45)
    if t < 0.4:
        f = t * 2.5
        col = (0.25 * (1 - f) + 0.85 * f,
               0.03 * (1 - f) + 0.15 * f,
               0.0 * (1 - f) + 0.01 * f)
    elif t < 0.9:
        f = (t - 0.4) * 2.0
        col = (0.85 * (1 - f) + 1.0 * f,
               0.15 * (1 - f) + 0.55 * f,
               0.01 * (1 - f) + 0.08 * f)
    elif t < 1.7:
        f = (t - 0.9) / 0.8
        col = (1.0,
               0.55 * (1 - f) + 0.92 * f,
               0.08 * (1 - f) + 0.6 * f)
    elif t < 2.5:
        f = (t - 1.7) / 0.8
        col = (1.0,
               0.92 * (1 - f) + 1.0 * f,
               0.6 * (1 - f) + 0.95 * f)
    else:
        col = (1.0, 1.0, 1.0)

    # Turbulence texture (subtle — avoid graininess at icon size)
    tu = 0.80 + 0.20 * hash2d(r * 5.0, phi * 3.0)
    tu2 = 0.90 + 0.10 * hash2d(r * 18.0, phi * 9.0)

    I *= 3.2 * tu * tu2
    return col[0] * I, col[1] * I, col[2] * I


def trace_ray(alpha: float, beta: float) -> tuple:
    """Trace a single ray from the observer screen to the scene.

    Args:
        alpha: horizontal screen coordinate (radians)
        beta: vertical screen coordinate (radians)

    Returns:
        (r, g, b) color tuple with values in [0, 1+]
    """
    b = -alpha * SIN_I  # angular momentum parameter

    # Initial conditions
    r = OBS_DIST
    th = INCL
    phi = 0.0

    sth = math.sin(th)
    cth = math.cos(th)
    s2 = sth * sth + S2_EPS
    c2 = cth * cth

    sig = r * r + A2 * c2
    delta = r * r - 2.0 * r + A2 + Q2
    sdel = max(delta, 1e-6)
    rpa2 = r * r + A2
    A_ = rpa2 * rpa2 - sdel * A2 * s2

    iSD = 1.0 / (sig * sdel)
    is2 = 1.0 / s2
    grr = sdel / sig
    gthi = 1.0 / sig

    pth = beta
    w_init = 2.0 * r - Q2
    rest = (-A_ * iSD + 2.0 * SPIN * b * w_init * iSD
            + gthi * pth * pth + (sig - w_init) * iSD * is2 * b * b)
    pr2 = -rest / grr
    pr = -math.sqrt(max(pr2, 0.0))

    color = [0.0, 0.0, 0.0]
    prev_th = th

    for _ in range(STEPS):
        # Adaptive step size — much smaller near horizon to prevent blowup
        dist_to_horizon = r - R_HORIZON
        if dist_to_horizon < 0.1:
            he = STEP_SIZE * 0.005
        elif dist_to_horizon < 0.5:
            he = STEP_SIZE * 0.02
        elif dist_to_horizon < 2.0:
            he = STEP_SIZE * dist_to_horizon * 0.15
        else:
            he = STEP_SIZE * min(1.0, dist_to_horizon * 0.3)

        # RK4 integration
        k1 = geodesic_rhs(r, th, pr, pth, b)
        r2 = r + he / 2 * k1[0]
        th2 = th + he / 2 * k1[1]
        pr2_ = pr + he / 2 * k1[3]
        pth2 = pth + he / 2 * k1[4]

        k2 = geodesic_rhs(r2, th2, pr2_, pth2, b)
        r3 = r + he / 2 * k2[0]
        th3 = th + he / 2 * k2[1]
        pr3 = pr + he / 2 * k2[3]
        pth3 = pth + he / 2 * k2[4]

        k3 = geodesic_rhs(r3, th3, pr3, pth3, b)
        r4 = r + he * k3[0]
        th4 = th + he * k3[1]
        pr4 = pr + he * k3[3]
        pth4 = pth + he * k3[4]

        k4 = geodesic_rhs(r4, th4, pr4, pth4, b)

        prev_th = th
        r += he / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        th += he / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        phi += he / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        pr += he / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        pth += he / 6 * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])

        # Disk crossing check (θ crosses π/2)
        if r > R_ISCO * 0.85 and r < DISK_OUTER:
            if (prev_th - math.pi / 2) * (th - math.pi / 2) < 0:
                dc = disk_color(r, phi)
                color[0] += dc[0]
                color[1] += dc[1]
                color[2] += dc[2]

        # Escape
        if r > R_ESCAPE:
            bg = background_stars(th, phi)
            color[0] += bg[0]
            color[1] += bg[1]
            color[2] += bg[2]
            break

        # Capture (or numerical blowup)
        if r < R_HORIZON * 1.01 or r != r:  # r != r catches NaN
            break

    return tuple(color)


def render(size: int) -> Image.Image:
    """Render the full icon image.

    Args:
        size: Output image dimensions (square).

    Returns:
        PIL Image with RGBA channels (rounded corners).
    """
    img = np.zeros((size, size, 3), dtype=np.float32)
    aspect = 1.0  # square

    for py in tqdm(range(size), desc="  Rendering", unit="row"):
        for px in range(size):
            # Screen coordinates: [-1, 1]
            x = (px - size / 2 + 0.5) / (size / 2)
            y = (size / 2 - py - 0.5) / (size / 2)

            alpha = x * FOV * aspect
            beta = y * FOV

            color = trace_ray(alpha, beta)
            img[py, px] = color

    # Tone mapping (simple Reinhard)
    img = img / (1.0 + img)

    # Gamma correction
    img = np.power(np.clip(img, 0, 1), 1.0 / 2.2)

    # Convert to 8-bit
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    # Create PIL image with rounded corners
    pil_img = Image.fromarray(img_u8, "RGB").convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    radius = max(1, size // 8)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    pil_img.putalpha(mask)

    return pil_img


def save_svg(pil_img: Image.Image, path: str) -> None:
    """Save the rendered image as an SVG with embedded PNG data URI."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    png_b64 = base64.b64encode(buf.getvalue()).decode()

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        'viewBox="0 0 128 128" width="128" height="128">\n'
        f'  <image width="128" height="128" '
        f'xlink:href="data:image/png;base64,{png_b64}"/>\n'
        "</svg>\n"
    )

    with open(path, "w") as f:
        f.write(svg)
    print(f"  SVG saved: {path} ({len(svg):,} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Kerr-Newman black hole icon via CPU ray tracing."
    )
    parser.add_argument(
        "--size", type=int, default=512,
        help="Output image size in pixels (square). Default: 512"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output PNG file path. Default: nulltracer-icon.png"
    )
    parser.add_argument(
        "--svg", type=str, default=None,
        help="Output SVG file path (embeds PNG as data URI). Default: nulltracer-icon.svg"
    )
    args = parser.parse_args()

    print(f"Kerr-Newman Black Hole Icon Generator")
    print(f"  Spin a = {SPIN}")
    print(f"  Charge Q = {CHARGE}")
    print(f"  Inclination = {INCLINATION_DEG}°")
    print(f"  Observer distance = {OBS_DIST}M")
    print(f"  ISCO = {R_ISCO:.3f}M")
    print(f"  Horizon = {R_HORIZON:.3f}M")
    print(f"  Image size = {args.size}×{args.size}")
    print(f"  Integration: {STEPS} steps (max), h={STEP_SIZE} (adaptive)")
    print()

    pil_img = render(args.size)

    # Save PNG
    png_path = args.output or "nulltracer-icon.png"
    pil_img.save(png_path, format="PNG", optimize=True)
    print(f"  PNG saved: {png_path}")

    # Save SVG
    svg_path = args.svg or "nulltracer-icon.svg"
    save_svg(pil_img, svg_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
