"""SAX-compatible component model for the straight waveguide.

Architectural rule: this file NEVER imports meep, and NEVER launches an FDTD
simulation. It only reads the cached, validated `fitted_model` coefficients
(`n_eff` from the length-sweep cutback method, `n_g` from an MPB dispersion
solve -- see `notebooks/01_waveguide_baseline.ipynb` Sections 5/5.2) via the
design point that stores them. If that design point doesn't exist yet,
loading it below raises a clear error telling the user to run the baseline
notebook first.

Evaluating an analytic model (rather than interpolating the raw FDTD S21
lookup table the earlier version of this file used) is what makes `length_um`
a free argument here: a fixed-length lookup table can't answer "what if the
waveguide is a different length", which every real circuit composition (e.g.
an MZI's two arms) needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import design_points, sparams

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_POINT_PATH = _REPO_ROOT / "data" / "design_points" / "waveguide.yaml"

# gdsfactory's gf.components.straight() always names its two ports this way
# (see meep_sim/waveguide.py's build_gf_component) -- fixed here rather than
# read from the design point, since it's a property of the model, not data.
_PORT_NAMES = ("o1", "o2")


def waveguide(wl=1.35, length_um: float | None = None):
    """SAX model function: wavelength (um, scalar or array), waveguide length
    (um, defaults to the characterized length_um the fit was performed at) ->
    SDict.

    S21/S12 = exp(2j*pi*n_eff(wl)*length_um/wl) -- lossless, but WITH a
    dispersion correction: `n_eff(wl) = n_eff0 + (n_eff0-n_g0)/wl0*(wl-wl0)`,
    the same linear-in-wavelength formula `circuits/mzi_lattice.py::
    ideal_arm_model` already validates. A bare constant `n_eff` (this
    function's own earlier version) is fine for the short lengths this
    component's own notebook characterizes, but `circuits/wdm_sax.ipynb`/
    `circuits/wdm_mux4_sax.ipynb` reuse this model for a much longer
    (26um/16um) reference arm, where the interference phase is sensitive
    enough to dispersion that the group index genuinely matters -- see
    `notebooks/01_waveguide_baseline.ipynb` Section 5.2 for how `n_g` was
    measured (MPB dispersion solve).

    The cutback sweep's amplitude channel measures FDTD's own noise floor
    here, not a real propagation loss (this cross-section has no absorption
    mechanism); a meaningful loss number needs a lossy material model
    instead (see `racetrack.py`'s conductivity trick). S11/S22 are modeled as exactly zero
    (a well-designed straight waveguide's reflection is negligible,
    confirmed by notebooks/01_waveguide_baseline.ipynb's passivity/energy-
    conservation checks on the raw FDTD data).
    """
    design_point = design_points.load_design_point(_DESIGN_POINT_PATH)
    fitted = design_point["fitted_model"]
    n_eff0 = fitted["n_eff"]
    n_g0 = fitted["n_g"]
    wl0 = fitted["lambda_um"]

    if length_um is None:
        length_um = design_point["parameters"]["length_um"]

    wl = np.asarray(wl, dtype=float)
    dn_dwl = (n_eff0 - n_g0) / wl0
    n_eff = n_eff0 + dn_dwl * (wl - wl0)
    phase = 2 * np.pi * n_eff * length_um / wl
    s21 = np.exp(1j * phase)
    zero = np.zeros_like(wl, dtype=complex)

    return sparams.to_sdict({"11": zero, "12": s21, "21": s21, "22": zero}, _PORT_NAMES)
