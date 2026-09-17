"""JAX-differentiable twin of `pic_toolkit.models.waveguide.waveguide`.

Architectural rule: this file NEVER imports meep. It reads the same cached
`fitted_model` (n_eff) that `models/waveguide.py` reads, via the same design
point, but reimplements the S-parameter formula with `jax.numpy` instead of
plain `numpy` so `length_um` can be a JAX tracer -- this is what lets
`jax.grad` differentiate the MUX2 objective with respect to `L_upper`
(the reference arm's length). `models/waveguide.py` itself is not modified;
that function is only ever called with a fixed (non-traced) `length_um`, and
mixing plain numpy into a jax.grad trace raises a concretization error.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from .. import design_points

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_POINT_PATH = _REPO_ROOT / "data" / "design_points" / "waveguide.yaml"
_PORT_NAMES = ("o1", "o2")


def _load_n_eff() -> float:
    design_point = design_points.load_design_point(_DESIGN_POINT_PATH)
    return float(design_point["fitted_model"]["n_eff"])


_N_EFF = _load_n_eff()


def waveguide_diff(wl, length_um) -> dict:
    """SAX model fn: wl (array-like, usually NOT traced), length_um (scalar,
    MAY be a JAX tracer -- this carries L_upper during optimization) -> SDict
    over (o1, o2). Exact formula mirror of `models.waveguide.waveguide`:
    S21=S12=exp(2j*pi*n_eff*length_um/wl), S11=S22=0 (lossless, negligible
    reflection -- same justification as the original)."""
    wl = jnp.asarray(wl, dtype=jnp.float32)
    phase = 2.0 * jnp.pi * _N_EFF * length_um / wl
    s21 = jnp.exp(1j * phase)
    zero = jnp.zeros_like(wl, dtype=jnp.complex64)
    o1, o2 = _PORT_NAMES
    return {(o1, o1): zero, (o1, o2): s21, (o2, o1): s21, (o2, o2): zero}
