"""
nulltracer_kernels.py — Production-grade rendering and shadow classification
=============================================================================

Loads CUDA kernels directly from server/kernels/*.cu files.
No CUDA code is embedded in Python strings (except the small
classification kernel which is ~80 lines).

Two rendering paths:
  render_kerr()   — Production visual pipeline (disk, stars, ACES, sRGB)
                    Uses the exact same .cu files as the production server.
  classify_kerr() — Shadow measurement (outputs class/r_disk/g_factor)
                    Uses production geoRHS but no visual pipeline.

Usage:
    from nulltracer_kernels import compile_all, render_kerr, classify_kerr
    from nulltracer_kernels import compare_integrators, METHODS
"""

import ctypes
import math
import re
import time as _time
from pathlib import Path

import cupy as cp
import numpy as np


_MODULE_DIR = Path(__file__).resolve().parent
_KERNEL_DIR = _MODULE_DIR / "server" / "kernels"
_INTEGRATOR_DIR = _KERNEL_DIR / "integrators"


# ── Include resolver (from production renderer.py) ────────────

def _resolve_includes(source, base_dir):
    pattern = re.compile(r'#include\s+"([^"]+)"')
    resolved = set()

    def replace(match, bdir):
        rel = match.group(1)
        absp = (bdir / rel).resolve()
        if absp in resolved:
            return f"/* Already included: {rel} */"
        resolved.add(absp)
        if not absp.exists():
            raise FileNotFoundError(f"Include not found: {rel} ({absp})")
        text = absp.read_text()
        return pattern.sub(lambda m: replace(m, absp.parent), text)

    return pattern.sub(lambda m: replace(m, base_dir), source)


# ── RenderParams struct (must match geodesic_base.cu exactly) ──

class RenderParams(ctypes.Structure):
    _fields_ = [
        ("width",              ctypes.c_double),
        ("height",             ctypes.c_double),
        ("spin",               ctypes.c_double),
        ("charge",             ctypes.c_double),
        ("incl",               ctypes.c_double),
        ("fov",                ctypes.c_double),
        ("phi0",               ctypes.c_double),
        ("isco",               ctypes.c_double),
        ("steps",              ctypes.c_double),
        ("obs_dist",           ctypes.c_double),
        ("esc_radius",         ctypes.c_double),
        ("disk_outer",         ctypes.c_double),
        ("step_size",          ctypes.c_double),
        ("bg_mode",            ctypes.c_double),
        ("star_layers",        ctypes.c_double),
        ("show_disk",          ctypes.c_double),
        ("show_grid",          ctypes.c_double),
        ("disk_temp",          ctypes.c_double),
        ("doppler_boost",      ctypes.c_double),
        ("srgb_output",        ctypes.c_double),
        ("disk_alpha",         ctypes.c_double),
        ("disk_max_crossings", ctypes.c_double),
        ("bloom_enabled",      ctypes.c_double),
    ]


# ── Kernel registry ───────────────────────────────────────────

_RENDER_REGISTRY = {
    "rk4":           ("rk4.cu",           "trace_rk4"),
    "rkdp8":         ("rkdp8.cu",         "trace_rkdp8"),
    "tao_yoshida4":  ("tao_yoshida4.cu",  "trace_tao_yoshida4"),
    "tao_kahan_li8": ("tao_kahan_li8.cu", "trace_tao_kahan_li8"),
}

METHODS = {
    "rk4":           "Runge-Kutta 4th-order",
    "rkdp8":         "Dormand-Prince 8(7) adaptive",
    "tao_yoshida4":  "Tao + Yoshida 4th symplectic",
    "tao_kahan_li8": "Tao + Kahan-Li 8th symplectic",
}


# ── Classification kernel body (the ONLY inline CUDA) ─────────

_CLASSIFY_BODY = r"""
__device__ double adaptive_step_classify(double r, double rp, double h_base) {
    double x = fmax((r - rp) / rp, 0.0);
    double f = x / (1.0 + x);
    double h = h_base * fmax(f * (1.0 + 0.5 * x), 0.02);
    return fmin(h, r / 6.0);
}

extern "C" __global__ void classify_kerr(
    double* out_class, double* out_rdisk, double* out_g, double* out_b,
    int W, int H, double a, double thobs, double fov, double robs,
    double risco, int maxsteps, double hbase
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= W || iy >= H) return;
    int idx = iy * W + ix;
    double a2 = a * a, Q2 = 0.0;
    double asp = (double)W / (double)H;
    double ux = (2.0 * (ix + 0.5) / (double)W - 1.0);
    double uy = (2.0 * (iy + 0.5) / (double)H - 1.0);
    double alpha = ux * fov * asp;
    double beta  = uy * fov;
    double sO = sin(thobs), cO = cos(thobs);
    double b = -alpha * sO;
    double r = robs, th = thobs, phi = 0.0;
    double s2o = sO * sO + S2_EPS, c2o = cO * cO;
    double r02 = robs * robs;
    double sig0 = r02 + a2 * c2o;
    double del0 = r02 - 2.0 * robs + a2 + Q2;
    double sdel0 = fmax(del0, 1e-14);
    double rpa2_0 = r02 + a2;
    double w0 = 2.0 * robs - Q2;
    double A_0 = rpa2_0 * rpa2_0 - sdel0 * a2 * s2o;
    double iSD0 = 1.0 / (sig0 * sdel0);
    double is2_0 = 1.0 / s2o;
    double grr0 = sdel0 / sig0;
    double gthi0 = 1.0 / sig0;
    double pth = -beta;
    double rest = -A_0 * iSD0 + 2.0 * a * b * w0 * iSD0
                  + gthi0 * beta * beta + (sig0 - w0) * iSD0 * is2_0 * b * b;
    double pr2 = -rest / grr0;
    double pr = (pr2 > 0.0) ? -sqrt(pr2) : 0.0;
    double rp = 1.0 + sqrt(fmax(1.0 - a2 - Q2, 0.0));
    double resc = robs + 12.0;
    int term = 0; double rdh = 0.0, gh = 1.0, oldTh = th;
    for (int i = 0; i < maxsteps; i++) {
        double he = adaptive_step_classify(r, rp, hbase);
        double oldR = r; oldTh = th;
        double r0=r,th0=th,phi0=phi,pr0=pr,pth0=pth;
        double k1r,k1t,k1p,k1pr,k1pt,k2r,k2t,k2p,k2pr,k2pt;
        double k3r,k3t,k3p,k3pr,k3pt,k4r,k4t,k4p,k4pr,k4pt;
        geoRHS(r0,th0,pr0,pth0,a,b,Q2,&k1r,&k1t,&k1p,&k1pr,&k1pt);
        geoRHS(r0+.5*he*k1r,th0+.5*he*k1t,pr0+.5*he*k1pr,pth0+.5*he*k1pt,a,b,Q2,&k2r,&k2t,&k2p,&k2pr,&k2pt);
        geoRHS(r0+.5*he*k2r,th0+.5*he*k2t,pr0+.5*he*k2pr,pth0+.5*he*k2pt,a,b,Q2,&k3r,&k3t,&k3p,&k3pr,&k3pt);
        geoRHS(r0+he*k3r,th0+he*k3t,pr0+he*k3pr,pth0+he*k3pt,a,b,Q2,&k4r,&k4t,&k4p,&k4pr,&k4pt);
        r  =r0 +he/6.*(k1r +2.*k2r +2.*k3r +k4r);
        th =th0+he/6.*(k1t +2.*k2t +2.*k3t +k4t);
        phi=phi0+he/6.*(k1p+2.*k2p +2.*k3p +k4p);
        pr =pr0+he/6.*(k1pr+2.*k2pr+2.*k3pr+k4pr);
        pth=pth0+he/6.*(k1pt+2.*k2pt+2.*k3pt+k4pt);
        if(th<0.005){th=0.005;pth=fabs(pth);}
        if(th>PI-0.005){th=PI-0.005;pth=-fabs(pth);}
        if(r<=rp*1.01){term=1;break;}
        if(i>2&&rdh==0.0){
            double c=(oldTh-PI*0.5)*(th-PI*0.5);
            if(c<0.0&&r>risco&&r<20.0){
                double f=fabs(oldTh-PI*0.5)/fmax(fabs(th-oldTh),1e-14);
                f=fmin(fmax(f,0.),1.);rdh=oldR+f*(r-oldR);
                double rh=rdh,sq=sqrt(rh),Om=1./(rh*sq+a);
                gh=sqrt(fmax(1.-3./rh+2.*a/(rh*sq),1e-12))/(fabs(1.-Om*b)+1e-30);
            }
        }
        if(r>resc){term=2;break;}
        if(r<0.5||r!=r||th!=th){term=1;break;}
    }
    if(term==0) term=1;
    out_class[idx]=(double)term; out_rdisk[idx]=rdh;
    out_g[idx]=gh; out_b[idx]=b;
}
"""


# ── Kernel cache ──────────────────────────────────────────────

_render_cache = {}
_classify_cache = None


def _compile_render(method):
    if method in _render_cache:
        return _render_cache[method]
    if method not in _RENDER_REGISTRY:
        raise ValueError(f"Unknown method '{method}'. Available: {list(_RENDER_REGISTRY.keys())}")
    filename, entry = _RENDER_REGISTRY[method]
    path = _INTEGRATOR_DIR / filename
    source = _resolve_includes(path.read_text(), path.parent)
    _render_cache[method] = cp.RawKernel(source, entry)
    return _render_cache[method]


def _compile_classify():
    global _classify_cache
    if _classify_cache is not None:
        return _classify_cache
    geo = (_KERNEL_DIR / "geodesic_base.cu").read_text()
    _classify_cache = cp.RawKernel(geo + "\n" + _CLASSIFY_BODY, 'classify_kerr')
    return _classify_cache


def compile_all():
    for m in _RENDER_REGISTRY:
        _compile_render(m)
        print(f"  Compiled render:   {METHODS[m]} ({m})")
    _compile_classify()
    print(f"  Compiled classify: RK4 + production geoRHS")
    print(f"All {len(_RENDER_REGISTRY)} render + 1 classify kernels ready.")


# ── ISCO ──────────────────────────────────────────────────────

def isco_kerr(a):
    z1 = 1.0 + (1.0 - a**2)**(1/3) * ((1.0 + a)**(1/3) + max(1.0 - a, 0.0)**(1/3))
    z2 = np.sqrt(3.0 * a**2 + z1**2)
    return 3.0 + z2 - np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


# ── Auto step budget ──────────────────────────────────────────

def auto_steps(obs_dist, h_base=0.3, rp=2.0, safety=3.0, method='rk4'):
    is_symp = method.startswith('tao')
    if method == '_classify':
        N_near = 20.0 / h_base
        N_far = (2 * rp / h_base) * math.log(max(obs_dist / rp, 2.0))
        return max(int((N_near + N_far) * safety), 400)
    elif is_symp:
        return max(int((obs_dist + 200.0 / h_base) * safety), 400)
    else:
        h_scaled = h_base * (obs_dist / 30.0) * 1.7
        h_max = 1.4 if method == 'rk4' else 3.0
        N = obs_dist / min(h_scaled, h_max) + 60.0 / h_base
        return max(int(N * safety), 200)


# ── Render (production visuals) ───────────────────────────────

def render_kerr(spin, inclination_deg, width=512, height=512, fov=7.0,
                obs_dist=40.0, max_steps=None, step_size=0.3,
                method='rk4', show_disk=True, bg_mode=0, star_layers=3,
                disk_temp=1.0, phi0=0.0, **kwargs):
    """Render with production visual pipeline. Returns (img, info)."""
    kernel = _compile_render(method)
    rp_est = 1.0 + math.sqrt(max(1.0 - spin**2, 0.0))
    if max_steps is None:
        max_steps = auto_steps(obs_dist, h_base=step_size, rp=rp_est, method=method)
    rp = RenderParams(
        width=float(width), height=float(height),
        spin=float(spin), charge=0.0,
        incl=math.radians(inclination_deg), fov=float(fov),
        phi0=float(phi0), isco=float(isco_kerr(spin)),
        steps=float(max_steps), obs_dist=float(obs_dist),
        esc_radius=float(obs_dist) + 12.0, disk_outer=14.0,
        step_size=float(step_size), bg_mode=float(bg_mode),
        star_layers=float(star_layers),
        show_disk=1.0 if show_disk else 0.0, show_grid=0.0,
        disk_temp=float(disk_temp), doppler_boost=2.0,
        srgb_output=1.0, disk_alpha=0.95,
        disk_max_crossings=5.0, bloom_enabled=0.0,
    )
    h_params = np.frombuffer(bytes(rp), dtype=np.uint8)
    d_params = cp.asarray(h_params)
    d_output = cp.zeros(width * height * 3, dtype=cp.uint8)
    block = (16, 16, 1)
    grid = ((width + 15) // 16, (height + 15) // 16)
    t0 = _time.time()
    kernel(grid, block, (d_params, d_output))
    cp.cuda.Device(0).synchronize()
    ms = (_time.time() - t0) * 1000
    img = np.flipud(d_output.get().reshape(height, width, 3))
    return img, {'render_ms': ms, 'max_steps': max_steps,
                 'obs_dist': obs_dist, 'method': method}


# ── Classify (shadow measurement) ─────────────────────────────

def classify_kerr(spin, inclination_deg, width=512, height=512, fov=7.0,
                  obs_dist=500.0, max_steps=None, step_size=0.15):
    """Classify pixels for shadow measurement. Returns (shadow_mask, info)."""
    kernel = _compile_classify()
    rp_est = 1.0 + math.sqrt(max(1.0 - spin**2, 0.0))
    if max_steps is None:
        max_steps = auto_steps(obs_dist, h_base=step_size, rp=rp_est, method='_classify')
    n = width * height
    d_class = cp.zeros(n, dtype=cp.float64)
    d_rdisk = cp.zeros(n, dtype=cp.float64)
    d_g     = cp.zeros(n, dtype=cp.float64)
    d_b     = cp.zeros(n, dtype=cp.float64)
    block = (16, 16, 1)
    grid = ((width + 15) // 16, (height + 15) // 16)
    t0 = _time.time()
    kernel(grid, block, (
        d_class, d_rdisk, d_g, d_b,
        np.int32(width), np.int32(height),
        np.float64(spin), np.float64(math.radians(inclination_deg)),
        np.float64(fov), np.float64(obs_dist),
        np.float64(float(isco_kerr(spin))),
        np.int32(max_steps), np.float64(step_size)
    ))
    cp.cuda.Device(0).synchronize()
    ms = (_time.time() - t0) * 1000
    h_class = np.flipud(d_class.get().reshape(height, width))
    h_rdisk = np.flipud(d_rdisk.get().reshape(height, width))
    h_g     = np.flipud(d_g.get().reshape(height, width))
    return (h_class == 1.0), {'render_ms': ms, 'max_steps': max_steps,
            'obs_dist': obs_dist, 'r_disk': h_rdisk, 'g_factor': h_g}


# ── Integrator comparison ─────────────────────────────────────

def compare_integrators(spin=0.6, inclination=80, obs_dist=40, step_size=0.3,
                        width=512, height=512, fov=7.0,
                        methods=None, fit_fn=None):
    import matplotlib.pyplot as plt
    if methods is None:
        methods = list(_RENDER_REGISTRY.keys())
    compile_all()
    results = []
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5))
    if len(methods) == 1: axes = [axes]
    for ax, m in zip(axes, methods):
        img, info = render_kerr(spin, inclination, width, height, fov=fov,
                                obs_dist=obs_dist, step_size=step_size, method=m)
        ax.imshow(img); ax.axis('off')
        ax.set_title(f"{METHODS[m]}\n{info['render_ms']:.0f} ms, "
                     f"{info['max_steps']} steps", fontsize=10)
        row = {'method': m, 'label': METHODS[m],
               'render_ms': info['render_ms'], 'max_steps': info['max_steps']}
        if fit_fn is not None:
            shadow, _ = classify_kerr(spin, inclination, width, height,
                                       fov=fov, obs_dist=obs_dist,
                                       step_size=min(step_size, 0.15))
            obs = fit_fn(shadow, fov=fov, img_size=width)
            if obs:
                row['diameter_M'] = obs['diameter_M']
                row['delta_C'] = obs['delta_C']
                row['circularity'] = obs['circularity']
        results.append(row)
    fig.suptitle(f'Integrator Comparison — $a={spin}$, '
                 rf'$\theta={inclination}°$, '
                 rf'$r_{{\rm obs}}={obs_dist}\,M$', fontsize=13, y=1.02)
    plt.tight_layout()
    hdr = f"\n{'Method':<35} {'Time':>8} {'Steps':>7}"
    if fit_fn: hdr += f" {'Diameter':>10} {'ΔC':>8}"
    print(hdr); print("─" * 75)
    for r in results:
        line = f"{r['label']:<35} {r['render_ms']:>7.0f}ms {r['max_steps']:>7d}"
        if 'diameter_M' in r: line += f" {r['diameter_M']:>9.4f}M {r['delta_C']:>7.4f}"
        print(line)
    return results, fig
