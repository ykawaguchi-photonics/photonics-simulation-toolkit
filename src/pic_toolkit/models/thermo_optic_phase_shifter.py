"""SAX-compatible model of a thermo-optic phase shifter (si_active track).

Architectural rule: this file NEVER imports meep. It reads the design point written by
si_active/notebooks/01_thermo_optic_phase_shifter.ipynb, which holds, per heater variant,
the mode's n_eff and group index and the heater's P_pi (from the 2D heat solve and the
MPB mode-overlap Delta n_eff).

Model (2-port, reflectionless, DC / quasi-static -- the thermal frequency response is
in the design point as tau and f_3dB, not in the S-matrix):

    S21 = 10^(-IL/20) * exp(+j (phi0 + dphi)),
    phi0 = 2 pi L [n_eff / wl0 + n_g (1/wl - 1/wl0)]     (first-order dispersion)
    dphi = pi * (P / P_pi) * (wl0 / wl)                   (fixed Delta n_eff, phase ~ 1/wl)

P_pi does not depend on L in a cross-section that is uniform along the heater (the
temperature rise scales with P / L), so `length_um` changes phi0 and the loss only.
The ~1 %/50 nm wavelength dependence of Si's dn/dT (Frey 2006) is neglected. Sign
convention exp(+j phi), as in models/tfln_mzm.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .. import design_points

assert "meep" not in sys.modules, "pic_toolkit.models.thermo_optic_phase_shifter must never coexist with a meep import"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_POINT_PATH = _REPO_ROOT / "si_active" / "data" / "design_points" / "thermo_optic_phase_shifter.yaml"


def load(variant: str = "tin") -> dict:
    dp = design_points.load_design_point(_DESIGN_POINT_PATH)
    if variant not in dp["variants"]:
        raise KeyError(f"variant {variant!r} not in {list(dp['variants'])}")
    return {"wl0_um": dp["wl0_um"], **dp["variants"][variant]}


def thermo_optic_phase_shifter(wl=1.31, power_mW=0.0, length_um=100.0, variant: str = "tin",
                               loss_db_per_cm: float | None = None):
    """SAX model: wavelength (um), heater power (mW) -> SDict over (o1, o2).
    `loss_db_per_cm` defaults to the variant's modeled heater-induced loss (0 for TiN;
    the free-carrier estimate for the N++ heater), on top of which the caller may add
    the waveguide's own propagation loss."""
    v = load(variant)
    wl = np.asarray(wl, dtype=float)
    wl0 = v["wl0_um"]
    n_eff, n_g = v["optical"]["n_eff"], v["optical"]["n_g_dispersive"]
    il = v["loss_db_per_cm"] if loss_db_per_cm is None else loss_db_per_cm
    phi0 = 2 * np.pi * length_um * (n_eff / wl0 + n_g * (1 / wl - 1 / wl0))
    dphi = np.pi * power_mW / v["p_pi_mW"] * (wl0 / wl)
    s = 10 ** (-il * length_um * 1e-4 / 20) * np.exp(1j * (phi0 + dphi))
    return {("o1", "o2"): s, ("o2", "o1"): s}
