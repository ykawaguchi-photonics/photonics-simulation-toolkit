"""SAX-compatible component model for the add-drop ring resonator.

Architectural rule: this file NEVER imports meep, and NEVER launches an FDTD
simulation (same rule as every other `models/*.py`). It evaluates the
standard TWO-COUPLER coupled-mode-theory transfer function for an add-drop
ring (Bogaerts et al., "Silicon microring resonators," Laser & Photonics
Rev. 6(1), 47-73 (2012)) -- the direct extension of `models/racetrack.py`'s
single-coupler all-pass formula to two independent coupling coefficients,
`kappa1` (input-bus coupler) and `kappa2` (add/drop-bus coupler), fit by a
multi-started nonlinear `scipy.optimize.curve_fit` to a single, finely
resolved resonance (NOT a closed-form solve, and NOT the full baseline
band -- see `10_add_drop_ring_resonator.ipynb`'s Section 12 and
`docs/simulation_settings_record.md` for why a global full-band fit was
tried and abandoned).

`t1 = sqrt(1-kappa1**2)`, `t2 = sqrt(1-kappa2**2)` (each coupler's own
lossless self-coupling amplitude), `a = exp(-alpha*L/2)` (round-trip
amplitude transmission, `alpha` = the ring waveguide's material power
attenuation per um), `phi = 2*pi*n_eff*L/wl` (round-trip phase),
`L = 2*pi*radius_um + 2*coupling_length_um` (same formula as racetrack.py --
loss integrates over the SAME total ring length regardless of how many
couplers tap it). `S_through` and `S_add_drop` swap which of `t1`/`t2`
plays the "near" (currently-excited) coupler's role -- they are NOT the
same function, since `kappa1 != kappa2` in general (this toolkit's own fit
found kappa1=0.274, kappa2=0.229 -- close but not equal, a genuine
difference between the two propagation directions through the ring, not
fitting noise):

    S_through  = (t2 - t1*a*exp(1j*phi)) / (1 - t1*t2*a*exp(1j*phi))   # input -> through
    S_add_drop = (t1 - t2*a*exp(1j*phi)) / (1 - t1*t2*a*exp(1j*phi))   # add -> drop
    S_drop     = -kappa1*kappa2*sqrt(a)*exp(1j*phi/2) / (1 - t1*t2*a*exp(1j*phi))  # input->drop == add->through (symmetric in kappa1,kappa2)

Passivity is guaranteed by construction whenever t1, t2, a are each in
[0,1] (same guarantee racetrack.py's single-coupler formula relies on).

Like `models/racetrack.py`, this model does NOT generalize to arbitrary
`gap_um`/`coupling_length_um` -- it only evaluates at the one design point
the notebook selected. Reflection terms (`S_input_refl`, `S_add_refl`) and
the direct bus-to-bus leakage terms (`S_input_add`, `S_add_input`) are
modeled as exactly zero -- the same explicit simplification
`models/racetrack.py`/`models/waveguide.py` make for their own small,
unfit residual terms.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .. import design_points

assert "meep" not in sys.modules, "pic_toolkit.models.add_drop_ring must never coexist with a meep import"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_POINT_PATH = _REPO_ROOT / "data" / "design_points" / "add_drop_ring.yaml"

_PORT_NAMES = ("input", "through", "add", "drop")


def add_drop_ring(wl=1.35):
    """SAX model function: wavelength (um, scalar or array) -> SDict.

    Evaluates the two-coupler add-drop CMT transfer function (see module
    docstring) at `data/design_points/add_drop_ring.yaml`'s `fitted_model`
    (`kappa1`, `kappa2`, `alpha`, `n_eff`, fit by a multi-started nonlinear
    optimization to a single narrowband resonance -- see the design
    point's own `fitted_model.fit_method` for the full description). Never
    runs a new simulation.

    Limitations (explicit, not silent):
    (1) Only valid at the one `gap_um`/`coupling_length_um` design point
        baked into `fitted_model` -- does not generalize across geometry,
        same limitation as `models/racetrack.py`.
    (2) `S_input_refl`/`S_add_refl` (reflection) and `S_input_add`/
        `S_add_input` (direct bus-to-bus leakage) are modeled as exactly
        zero -- measured small but nonzero by FDTD, not currently fit.
    """
    design_point = design_points.load_design_point(_DESIGN_POINT_PATH)
    fitted = design_point["fitted_model"]

    wl = np.asarray(wl, dtype=float)
    freqs = 1.0 / wl
    L = 2 * np.pi * fitted["radius_um"] + 2 * fitted["coupling_length_um"]
    t1 = np.sqrt(1 - fitted["kappa1"] ** 2)
    t2 = np.sqrt(1 - fitted["kappa2"] ** 2)
    a = np.exp(-fitted["alpha"] * L / 2)
    phi = 2 * np.pi * fitted["n_eff"] * freqs * L

    denom = 1 - t1 * t2 * a * np.exp(1j * phi)
    s_through = (t2 - t1 * a * np.exp(1j * phi)) / denom
    s_add_drop = (t1 - t2 * a * np.exp(1j * phi)) / denom  # NOT the same as s_through when kappa1 != kappa2
    s_drop = -fitted["kappa1"] * fitted["kappa2"] * np.sqrt(a) * np.exp(1j * phi / 2) / denom
    zero = np.zeros(wl.shape, dtype=complex)

    input_, through, add, drop = _PORT_NAMES
    return {
        (input_, input_): zero, (add, add): zero,               # reflection, not fit
        (input_, add): zero, (add, input_): zero,                # direct bus leakage, not fit
        (input_, through): s_through, (through, input_): s_through,
        (add, drop): s_add_drop, (drop, add): s_add_drop,
        (input_, drop): s_drop, (drop, input_): s_drop,
        (add, through): s_drop, (through, add): s_drop,
    }
