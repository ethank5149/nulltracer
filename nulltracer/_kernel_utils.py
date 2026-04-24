"""
Internal kernel compilation utilities.

Handles #include resolution, compilation caching, and CuPy RawKernel management.
"""

import re
from pathlib import Path

import cupy as cp

__all__ = ["resolve_includes", "KernelCache"]

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
INTEGRATOR_DIR = KERNEL_DIR / "integrators"

_INCLUDE_RE = re.compile(r'#include\s+"([^"]+)"')


def resolve_includes(source: str, base_dir: Path) -> str:
    """Inline all ``#include "..."`` directives recursively.

    CuPy's RawKernel does not support the preprocessor ``#include``
    directive, so we manually concatenate referenced files.

    Parameters
    ----------
    source : str
        CUDA source text.
    base_dir : Path
        Directory to resolve relative include paths against.

    Returns
    -------
    str
        Source with all includes inlined.
    """
    resolved: set[Path] = set()

    def _replace(match: re.Match, bdir: Path) -> str:
        rel = match.group(1)
        absp = (bdir / rel).resolve()
        if absp in resolved:
            return f"/* already included: {rel} */"
        resolved.add(absp)
        if not absp.exists():
            raise FileNotFoundError(
                f"Include not found: {rel} (resolved to {absp})"
            )
        text = absp.read_text(encoding='utf-8')
        return _INCLUDE_RE.sub(lambda m: _replace(m, absp.parent), text)

    return _INCLUDE_RE.sub(lambda m: _replace(m, base_dir), source)


class KernelCache:
    """Thread-safe compilation cache for CuPy RawKernels.

    Kernels are compiled lazily on first use and cached for the
    lifetime of the process.  Call :meth:`purge` after editing
    ``.cu`` files **and** restarting the Jupyter kernel (CuPy also
    caches internally).
    """

    def __init__(self) -> None:
        self._cache: dict[str, cp.RawKernel] = {}

    # ?????? render kernels (full-frame, one thread per pixel) ??????????????????

    #: Maps method name ??? (filename, entry_point) for full-frame rendering.
    RENDER_REGISTRY: dict[str, tuple[str, str]] = {
        "rk4":           ("rk4.cu",           "trace_rk4"),
        "rkdp8":         ("rkdp8.cu",         "trace_rkdp8"),
        "symplectic8":   ("symplectic8.cu",    "trace_symplectic8"),
    }

    #: Maps method name ??? entry_point for single-ray tracing.
    RAY_TRACE_REGISTRY: dict[str, str] = {
        "rk4":           "ray_trace_rk4",
        "rkdp8":         "ray_trace_rkdp8",
        "symplectic8":   "ray_trace_symplectic8",
    }

    #: Human-readable labels.
    METHOD_LABELS: dict[str, str] = {
        "rk4":           "Runge–Kutta 4th order",
        "rkdp8":         "Dormand–Prince 8(7) adaptive",
        "symplectic8":   "Tao–Kahan-Li 8th–10th symplectic (ASΦ + Wisdom corrector)",
    }

    _COMPILE_OPTS = ("--std=c++14", "-use_fast_math")

    # ?????? public helpers ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

    @property
    def available_methods(self) -> list[str]:
        """All integration methods with full-frame render kernels."""
        return list(self.RENDER_REGISTRY)

    @property
    def ray_trace_methods(self) -> list[str]:
        """Methods with single-ray trace kernels."""
        return list(self.RAY_TRACE_REGISTRY)

    def purge(self) -> int:
        """Drop all cached kernels.  Returns count purged."""
        n = len(self._cache)
        self._cache.clear()
        return n

    # ?????? compilation ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

    def get_render_kernel(self, method: str) -> cp.RawKernel:
        """Return (compiling if needed) the full-frame render kernel."""
        key = f"render:{method}"
        if key not in self._cache:
            if method not in self.RENDER_REGISTRY:
                raise ValueError(
                    f"Unknown method {method!r}. "
                    f"Available: {self.available_methods}"
                )
            filename, entry = self.RENDER_REGISTRY[method]
            path = INTEGRATOR_DIR / filename
            src = resolve_includes(path.read_text(), path.parent)
            self._cache[key] = cp.RawKernel(src, entry, options=self._COMPILE_OPTS)
        return self._cache[key]

    def get_ray_trace_kernel(self, method: str) -> cp.RawKernel:
        """Return (compiling if needed) the single-ray trace kernel."""
        key = f"ray:{method}"
        if key not in self._cache:
            if method not in self.RAY_TRACE_REGISTRY:
                raise ValueError(
                    f"No ray-trace kernel for {method!r}. "
                    f"Available: {self.ray_trace_methods}"
                )
            entry = self.RAY_TRACE_REGISTRY[method]
            path = KERNEL_DIR / "ray_trace.cu"
            src = resolve_includes(path.read_text(), path.parent)
            self._cache[key] = cp.RawKernel(src, entry, options=self._COMPILE_OPTS)
        return self._cache[key]

    def get_classify_kernel(self) -> cp.RawKernel:
        """Return (compiling if needed) the shadow-classification kernel."""
        key = "classify"
        if key not in self._cache:
            geo = (KERNEL_DIR / "geodesic_base.cu").read_text()
            src = geo + "\n" + _CLASSIFY_KERNEL_BODY
            self._cache[key] = cp.RawKernel(src, "classify_kerr")
        return self._cache[key]

    def compile_all(self, *, verbose: bool = True) -> None:
        """Pre-compile every kernel (render + classify)."""
        for m in self.RENDER_REGISTRY:
            self.get_render_kernel(m)
            if verbose:
                print(f"  ??? render   {self.METHOD_LABELS.get(m, m)}")
        self.get_classify_kernel()
        if verbose:
            print(f"  ??? classify RK4 + production geoRHS")
            print(
                f"All {len(self.RENDER_REGISTRY)} render "
                f"+ 1 classify kernels ready."
            )


# ?????? Inline classify kernel (the only CUDA not in .cu files) ?????????

_CLASSIFY_KERNEL_BODY = r"""
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
