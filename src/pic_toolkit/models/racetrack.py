"""SAX-compatible component model for the all-pass racetrack resonator.

Architectural rule: this file NEVER imports meep, and NEVER launches an FDTD
simulation. It evaluates the standard all-pass-ring coupled-mode-theory
transfer function (Bogaerts et al., "Silicon microring resonators," Laser &
Photonics Rev. 6(1), 47-73 (2012)), parameterized by the cross-coupling
coefficient `kappa` (`fitted_model` in the design point -- `kappa`, `alpha`,
`n_eff` extracted in closed form from the resonance lineshape in
`05_racetrack_resonator.ipynb`'s Section 12) at the one `coupling_length_um`
the notebook selected. Unlike `models/waveguide.py`/`models/bend.py`, this
model does NOT generalize to arbitrary `coupling_length_um` values -- an
earlier automated fit-vs-coupling_length_um generalization was tried and
abandoned (see the notebook's Section 12 markdown for why) in favor of a
single, honestly-validated design point.

Independent of the since-removed models/ring.py -- this is a different
component (`racetrack`, not `ring`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .. import design_points, sparams

assert "meep" not in sys.modules, "pic_toolkit.models.racetrack must never coexist with a meep import"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_POINT_PATH = _REPO_ROOT / "data" / "design_points" / "racetrack.yaml"


def racetrack(wl=1.35):
    """SAX model function: wavelength (um, scalar or array) -> SDict.

    Evaluates the all-pass-ring coupled-mode-theory transfer function
    (data/design_points/racetrack.yaml's `fitted_model`: `kappa`, `alpha`,
    `n_eff` extracted in closed form from the resonance lineshape in
    05_racetrack_resonator.ipynb's Section 12, per Bogaerts et al. 2012)
    at the single `coupling_length_um` the notebook selected. Never runs a
    new simulation.

    t = sqrt(1 - kappa**2) (lossless-coupler assumption, Bogaerts' r^2+k^2=1).
    S21 = S12 = (t - a*exp(i*phi)) / (1 - t*a*exp(i*phi)), the standard
    all-pass ring transfer function, phi = 2*pi*n_eff*L/wl, L = 2*pi*radius_um
    + 2*coupling_length_um, a = exp(-alpha*L/2) (alpha = power attenuation
    per um). S11/S22 are modeled as exactly zero, the same explicit
    simplification models/waveguide.py makes -- this device's actual
    reflection is small (a few percent) but nonzero; not currently fit.

    Limitations (explicit, not silent):
    (1) Unlike models/waveguide.py/models/bend.py, this model does NOT
        generalize to arbitrary parameter values -- it only evaluates at the
        one coupling_length_um baked into `fitted_model`. The earlier
        automated fit-vs-coupling_length_um generalization was tried and
        abandoned (see the notebook's Section 12 markdown).
    (2) `kappa`/`alpha`/`n_eff` are extracted from one FDTD resonance's
        measured depth, linewidth, and location (not a blind least-squares
        fit over the raw spectrum) -- an explicit, disclosed approximation
        (e.g. group index approximated by `n_eff`), not a high-precision
        device model. Passivity (|S21|<=1) IS guaranteed regardless, since
        `t=sqrt(1-kappa**2)` and `a=exp(-alpha*L/2)` are each constrained to
        [0,1].
    """
    design_point = design_points.load_design_point(_DESIGN_POINT_PATH)
    fitted = design_point["fitted_model"]

    wl = np.asarray(wl, dtype=float)
    freqs = 1.0 / wl
    L = 2 * np.pi * fitted["radius_um"] + 2 * fitted["coupling_length_um"]
    t = np.sqrt(1 - fitted["kappa"] ** 2)
    a = np.exp(-fitted["alpha"] * L / 2)
    phi = 2 * np.pi * fitted["n_eff"] * freqs * L
    s21 = (t - a * np.exp(1j * phi)) / (1 - t * a * np.exp(1j * phi))
    zero = np.zeros(wl.shape, dtype=complex)

    return sparams.to_sdict({"11": zero, "12": s21, "21": s21, "22": zero}, ("o1", "o2"))
