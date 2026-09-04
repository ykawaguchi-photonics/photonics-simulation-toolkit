"""SAX-compatible component model for the 90-degree bent waveguide.

Architectural rule: this file NEVER imports meep, and NEVER launches an FDTD
simulation. It only reads the cached, validated `fitted_model` coefficients
that the Meep baseline notebook already fit per bend_type via a radius sweep
(loss_db as an exponential decay to a floor, phase linearly), via the design
point that stores them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import design_points, sparams

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_POINT_PATH = _REPO_ROOT / "data" / "design_points" / "bend.yaml"

# gdsfactory's own bend components (see meep_sim/bend.py's build_gf_component)
# always name their two ports this way -- fixed here rather than read from
# the design point, since it's a property of the model, not data.
_PORT_NAMES = ("o1", "o2")


def bend(wl=1.35, radius_um: float | None = None, bend_type: str = "circular"):
    """SAX model function: wavelength (um, scalar or array -- see Limitations
    below), bend radius (um, defaults to the design point's baseline
    radius_um), bend_type ("circular" or "euler") -> SDict.

    S21/S12 = 10**(-loss_dB(radius_um)/20) * exp(1j*phase(radius_um)), where
    loss_dB(r) = floor + amplitude * exp(-r / decay_um) (an exponential decay
    to a noise floor -- bend loss falls off steeply at small radius and
    flattens out at large radius, which a straight line cannot represent) and
    phase is a straight-line fit against radius_um (phase scales with arc
    length, which is already linear in radius). Both are fit independently
    per bend_type against a 9-point radius sweep -- see notebooks/
    02_bent_waveguide.ipynb Section 5. S11/S22 are modeled as exactly
    zero, the same simplification models/waveguide.py makes.

    Limitations (explicit, not silent):
    (1) both fits were performed at ONE representative wavelength (this
        design's O-band center) -- this model does NOT capture wavelength
        dispersion; `wl` only sets the output array's shape, every requested
        wavelength gets the same S21.
    (2) 9 sweep points (radius_um=1.0-3.0, 0.25um steps) back each fit; the
        exponential-plus-floor form matches the qualitative shape of the
        measured data far better than a straight line, but is still a
        4-parameter phenomenological fit, not a first-principles bend-loss
        model -- treat extrapolation well beyond 1.0-3.0um with the same
        caution any curve fit deserves.
    (3) bend_type="euler" is WORSE than circular below radius_um~1.5-1.75,
        not better: gdsfactory's bend_euler(..., with_arc_floorplan=True) --
        the default, which build_gf_component uses -- keeps the same
        footprint as a circular bend of that nominal radius, which forces
        the Euler curve's peak curvature TIGHTER than radius_um at small
        radii (see notebooks/02_bent_waveguide.ipynb Section 4.2b's
        markdown for the full investigation, including a direct gdsfactory
        bbox comparison). The crossover was pinned down with the 9-point
        sweep to between radius_um=1.5 (euler still worse, 0.497dB vs
        circular's 0.307dB) and radius_um=1.75 (euler clearly better,
        0.227dB vs 0.292dB) -- do not use this Euler fit below
        radius_um~1.75 as if Euler were the better choice there; it isn't.
    """
    design_point = design_points.load_design_point(_DESIGN_POINT_PATH)
    fitted = design_point["fitted_model"][bend_type]

    if radius_um is None:
        radius_um = design_point["parameters"]["radius_um"]

    loss_db = fitted["loss_db_floor"] + fitted["loss_db_amplitude"] * np.exp(
        -radius_um / fitted["loss_db_decay_um"]
    )
    phase = fitted["phase_slope"] * radius_um + fitted["phase_intercept"]
    amplitude = 10 ** (-loss_db / 20.0)
    s21_value = amplitude * np.exp(1j * phase)

    wl = np.asarray(wl, dtype=float)
    s21 = np.full(wl.shape, s21_value, dtype=complex)
    zero = np.zeros(wl.shape, dtype=complex)

    return sparams.to_sdict({"11": zero, "12": s21, "21": s21, "22": zero}, _PORT_NAMES)
