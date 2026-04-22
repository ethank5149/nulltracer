"""Regression tests for renderer parameter resolution helpers.

These pin down the behaviour introduced in P1 of the publication-readiness
patch series: the dict-based `CudaRenderer.render_frame` API must accept
both the canonical `inclination` key and the shorthand `incl` alias used
throughout the notebook and web client, and must auto-size the integration
step budget when none is provided.

A silent prior version of the code defaulted to `inclination=80.0` whenever
the caller only supplied `incl=...`, causing every dict-based render to run
at θ=80° regardless of input — including every EHT-validation figure in the
hero notebook.
"""

import pytest

# These helpers are pure — no GPU required — so import directly.
pytest.importorskip("numpy")
# But importing nulltracer.renderer pulls in CuPy. Skip the whole module
# gracefully on CPU-only runners so the CI runs cleanly without a GPU.
pytest.importorskip("cupy", reason="CuPy not installed; renderer tests require a CUDA build")


def test_resolve_inclination_accepts_canonical_key():
    from nulltracer.renderer import _resolve_inclination
    assert _resolve_inclination({"inclination": 17.0}) == 17.0
    assert _resolve_inclination({"inclination": 50}) == 50.0


def test_resolve_inclination_accepts_incl_alias():
    from nulltracer.renderer import _resolve_inclination
    assert _resolve_inclination({"incl": 17.0}) == 17.0
    assert _resolve_inclination({"incl": 90}) == 90.0


def test_resolve_inclination_canonical_wins_over_alias():
    from nulltracer.renderer import _resolve_inclination
    # If both are supplied, the canonical key takes precedence.
    got = _resolve_inclination({"inclination": 30.0, "incl": 80.0})
    assert got == 30.0


def test_resolve_inclination_default():
    from nulltracer.renderer import _resolve_inclination
    assert _resolve_inclination({}) == 80.0


def test_resolve_steps_respects_explicit_steps():
    from nulltracer.renderer import _resolve_steps
    got = _resolve_steps(
        {"steps": 321}, spin=0.9, charge=0.0, method="rkdp8",
        obs_dist=40.0, step_size=0.3,
    )
    assert got == 321


def test_resolve_steps_respects_explicit_max_steps():
    from nulltracer.renderer import _resolve_steps
    got = _resolve_steps(
        {"max_steps": 555}, spin=0.0, charge=0.0, method="rk4",
        obs_dist=40.0, step_size=0.3,
    )
    assert got == 555


def test_resolve_steps_falls_back_to_auto():
    """When neither steps nor max_steps is supplied, call auto_steps()."""
    from nulltracer.renderer import _resolve_steps
    got = _resolve_steps(
        {}, spin=0.0, charge=0.0, method="rk4",
        obs_dist=500.0, step_size=0.15,
    )
    # auto_steps() must return a positive integer; crucially, it must scale
    # with obs_dist — the old hard-coded 200 gave 30 M of affine length at
    # step_size=0.15, which never reached the black hole at obs_dist=500.
    assert isinstance(got, int)
    assert got > 200, f"auto_steps returned {got}; should exceed old default of 200"
