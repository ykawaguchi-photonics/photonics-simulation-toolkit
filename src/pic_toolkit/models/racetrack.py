"""SAX-compatible component model for the all-pass racetrack resonator.

Architectural rule: this file NEVER imports meep, and NEVER launches an FDTD
simulation. It evaluates the all-pass coupling-matrix model of a ring coupled to
one bus (Yariv, Electron. Lett. 36, 321 (2000); Bogaerts et al., Laser Photonics
Rev. 6, 47 (2012)) with the parameters `05_racetrack_resonator.ipynb` fitted to
one finely resolved FDTD resonance (`pic_toolkit.ring_fit.fit_all_pass` /
`choose_branch`) at the one `coupling_length_um` the notebook selected. Unlike
`models/waveguide.py`/`models/bend.py`, it does NOT generalize to other
`coupling_length_um` values.

Independent of the since-removed models/ring.py -- this is a different
component (`racetrack`, not `ring`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .. import design_points, ring_fit, sparams

assert "meep" not in sys.modules, "pic_toolkit.models.racetrack must never coexist with a meep import"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_POINT_PATH = _REPO_ROOT / "si" / "data" / "design_points" / "racetrack.yaml"


def racetrack(wl=1.35):
    """SAX model function: wavelength (um, scalar or array) -> SDict.

    S21 = S12 = (t - a*exp(i phi)) / (1 - t*a*exp(i phi)) with
    t = bus-ring self-coupling, a = exp(-alpha*L/2) (alpha = power attenuation per
    um), L = 2*pi*radius_um + 2*coupling_length_um, and the round-trip phase
    linearized around the fitted resonance with the group index measured from the
    free spectral range, phi = 2*pi*n_g*L*(1/wl - 1/lambda_res_um). All values come
    from si/data/design_points/racetrack.yaml's `fitted_model` (which also records the
    coupling regime and why that t<->a branch was chosen). Never runs a new
    simulation.

    Limitations (explicit, not silent):
    (1) Only valid at the one coupling_length_um baked into `fitted_model`.
    (2) Accurate near the fitted resonance; the linearized phase places the
        neighbouring resonances only to the accuracy of n_g (about +-2 %, set by
        the FDTD wavelength grid).
    (3) S11/S22 are modeled as exactly zero (the measured reflection is small but
        nonzero, not fit). Passivity holds by construction for t, a in [0, 1].
    """
    design_point = design_points.load_design_point(_DESIGN_POINT_PATH)
    fitted = design_point["fitted_model"]

    wl = np.asarray(wl, dtype=float)
    L = 2 * np.pi * fitted["radius_um"] + 2 * fitted["coupling_length_um"]
    s21 = ring_fit.all_pass(1.0 / wl, fitted["t"], fitted["a"], 1.0 / fitted["lambda_res_um"],
                            1.0 / (fitted["n_g"] * L))
    zero = np.zeros(wl.shape, dtype=complex)

    return sparams.to_sdict({"11": zero, "12": s21, "21": s21, "22": zero}, ("o1", "o2"))
