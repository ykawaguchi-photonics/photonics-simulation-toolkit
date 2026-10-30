"""SAX-compatible component model for the add-drop ring resonator.

Architectural rule: this file NEVER imports meep, and NEVER launches an FDTD
simulation (same rule as every other `models/*.py`). It evaluates the
two-coupler coupling-matrix model of an add-drop ring (Yariv, Electron. Lett.
36, 321 (2000); Bogaerts et al., Laser Photonics Rev. 6, 47 (2012)) with the
parameters `10_add_drop_ring_resonator.ipynb` fitted to one finely resolved
FDTD resonance (`pic_toolkit.ring_fit.fit_add_drop`).

`t1 = sqrt(1-kappa1**2)` is the INPUT-bus coupler's self-coupling, `t2` the
DROP-bus coupler's, `a = exp(-alpha*L/2)` the single-pass field transmission
(`alpha` = power attenuation per um), `L = 2*pi*radius_um + 2*coupling_length_um`.
The round-trip phase is linearized around the fitted resonance with the group
index measured from the free spectral range,
`phi = 2*pi*n_g*L*(1/wl - 1/lambda_res_um)` (a single resonance does not
determine n_eff, so n_eff is not a parameter). With D = 1 - t1*t2*a*exp(i phi):

    input -> through = (t1 - t2*a*exp(i phi)) / D
    add   -> drop    = (t2 - t1*a*exp(i phi)) / D
    input -> drop = add -> through = -kappa1*kappa2*sqrt(a)*exp(i phi/2) / D

(light launched into a bus meets that bus's own coupler first). Passivity holds
by construction for t1, t2, a in [0, 1].

Like `models/racetrack.py`, this model only evaluates at the one
`gap_um`/`coupling_length_um` design point the notebook selected, and is
accurate near the fitted resonance (the linearized phase reproduces the
neighbouring resonances only to the accuracy of n_g). Reflection terms
(`S_input_refl`, `S_add_refl`) and direct bus-to-bus leakage (`S_input_add`,
`S_add_input`) are modeled as exactly zero -- small but nonzero in FDTD, not fit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .. import design_points, ring_fit

assert "meep" not in sys.modules, "pic_toolkit.models.add_drop_ring must never coexist with a meep import"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_POINT_PATH = _REPO_ROOT / "si" / "data" / "design_points" / "add_drop_ring.yaml"

_PORT_NAMES = ("input", "through", "add", "drop")


def add_drop_ring(wl=1.35):
    """SAX model function: wavelength (um, scalar or array) -> SDict.

    Evaluates the two-coupler add-drop model (see module docstring) at
    `si/data/design_points/add_drop_ring.yaml`'s `fitted_model` (`t1`, `t2`, `a`,
    `n_g`, `lambda_res_um`; `fitted_model.fit_method` describes the fit). Never
    runs a new simulation.
    """
    design_point = design_points.load_design_point(_DESIGN_POINT_PATH)
    fitted = design_point["fitted_model"]

    wl = np.asarray(wl, dtype=float)
    L = 2 * np.pi * fitted["radius_um"] + 2 * fitted["coupling_length_um"]
    s = ring_fit.add_drop(1.0 / wl, fitted["t1"], fitted["t2"], fitted["a"],
                          1.0 / fitted["lambda_res_um"], 1.0 / (fitted["n_g"] * L))
    zero = np.zeros(wl.shape, dtype=complex)

    input_, through, add, drop = _PORT_NAMES
    return {
        (input_, input_): zero, (add, add): zero,               # reflection, not fit
        (input_, add): zero, (add, input_): zero,                # direct bus leakage, not fit
        (input_, through): s["through"], (through, input_): s["through"],
        (add, drop): s["add_drop"], (drop, add): s["add_drop"],
        (input_, drop): s["drop"], (drop, input_): s["drop"],
        (add, through): s["add_through"], (through, add): s["add_through"],
    }
