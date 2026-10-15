"""SAX-compatible, VOLTAGE-dependent model of the x-cut TFLN electro-optic MZM.

Architectural rule: this file NEVER imports meep (same rule as every module in
pic_toolkit.models). It composes, by hand, quantities notebooks/11_tfln_mzm.ipynb
already measured and saved:

- the TFLN 50:50 directional coupler's complex S-parameters (FDTD, cached
  artifact named in data/design_points/tfln_mzm.yaml -> coupler.source_artifact),
  used for BOTH the splitter and the combiner;
- the push-pull phase from the Pockels effect,
      Delta phi(V, lambda) = pi * V / V_pi(lambda),  V_pi(lambda) = (V_pi L)(lambda0) / L * lambda / lambda0,
  where V_pi L comes from the cross-section overlap analysis (section 6) and the
  lambda/lambda0 scaling holds while n_e, r33 and the overlap factor are taken
  constant across the band (the toolkit's non-dispersive convention);
- the common arm phase 2 pi L [n_eff/lambda0 + n_g (1/lambda - 1/lambda0)]
  (first-order dispersion via the dispersive group index of section 9).

Unlike models/mzm.py (fixed at its V_pi operating point), this model takes the
drive voltage `V` as an argument -- a DC transfer function; the RF frequency
response is NOT part of a SAX S-matrix (see notebook 11 section 9 for it).
Propagation loss defaults to 0 dB/cm and can be set with `loss_db_per_cm`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .. import design_points

assert "meep" not in sys.modules, "pic_toolkit.models.tfln_mzm must never coexist with a meep import"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_POINT_PATH = _REPO_ROOT / "data" / "design_points" / "tfln_mzm.yaml"


def _load_coupler(stem: Path) -> dict:
    """Meep-free reader for meep_sim.coupler.save_artifact's .npz/.json pair."""
    stem = stem if stem.is_absolute() else _REPO_ROOT / stem
    npz = stem.parent / (stem.name + ".npz")
    if not npz.exists():
        raise FileNotFoundError(f"TFLN coupler artifact not found at {npz}. Run notebooks/11_tfln_mzm.ipynb first.")
    d = np.load(npz)
    meta = json.loads((stem.parent / (stem.name + ".json")).read_text())
    return {k: d[k] for k in ("wavelengths_um", "S_through", "S_cross", "S_through_bot", "S_cross_bot")} | {
        "port_names": tuple(meta["port_names"])}


def _interp_complex(wl, wl_grid, values):
    """Magnitude + unwrapped-phase interpolation (same as models/coupler.py). The
    TFLN coupler's S-parameter phase winds by ~1.7 rad between adjacent artifact
    wavelength samples; interpolating real/imaginary parts instead cuts the chord of
    that arc and collapses |S| between samples (|S_through| 0.63 -> 0.43 at 1.40 um)."""
    wl = np.asarray(wl, dtype=float)
    order = np.argsort(wl_grid)   # artifact wavelengths are descending (1/freqs)
    x, v = wl_grid[order], values[order]
    mag = np.interp(wl, x, np.abs(v))
    phase = np.interp(wl, x, np.unwrap(np.angle(v)))
    return mag * np.exp(1j * phase)


def tfln_mzm(wl=1.35, V=0.0, loss_db_per_cm=0.0):
    """SAX model: wavelength (um) and drive voltage V (volts, push-pull) -> SDict over
    (in_top, in_bot, out_top, out_bot). Reflections are not modeled (the cached
    coupler's own reflection is < 1e-3 in power)."""
    dp = design_points.load_design_point(_DESIGN_POINT_PATH)
    c = _load_coupler(Path(dp["coupler"]["source_artifact"]))
    wl = np.asarray(wl, dtype=float)
    wl0 = dp["wl0_um"]
    L_um = dp["design"]["arm_length_mm"] * 1e3
    vpi = dp["eo"]["vpi_l_V_cm"] / (L_um * 1e-4) * wl / wl0
    n_eff, n_g = dp["optical"]["n_eff"], dp["optical"]["n_g_dispersive"]

    t1 = _interp_complex(wl, c["wavelengths_um"], c["S_through"])
    x1 = _interp_complex(wl, c["wavelengths_um"], c["S_cross"])
    t2 = _interp_complex(wl, c["wavelengths_um"], c["S_through_bot"])
    x2 = _interp_complex(wl, c["wavelengths_um"], c["S_cross_bot"])
    # coupler transfer: [out_top, out_bot] = [[t1, x2], [x1, t2]] [in_top, in_bot]
    phi0 = 2 * np.pi * L_um * (n_eff / wl0 + n_g * (1 / wl - 1 / wl0))
    dphi = np.pi * V / vpi
    amp = 10 ** (-loss_db_per_cm * L_um * 1e-4 / 20)
    a_top = amp * np.exp(1j * (phi0 + dphi / 2))
    a_bot = amp * np.exp(1j * (phi0 - dphi / 2))
    # out = C * diag(a_top, a_bot) * C
    s_tt = t1 * a_top * t1 + x2 * a_bot * x1      # in_top -> out_top
    s_tb = x1 * a_top * t1 + t2 * a_bot * x1      # in_top -> out_bot
    s_bt = t1 * a_top * x2 + x2 * a_bot * t2      # in_bot -> out_top
    s_bb = x1 * a_top * x2 + t2 * a_bot * t2      # in_bot -> out_bot
    return {
        ("in_top", "out_top"): s_tt, ("out_top", "in_top"): s_tt,
        ("in_top", "out_bot"): s_tb, ("out_bot", "in_top"): s_tb,
        ("in_bot", "out_top"): s_bt, ("out_top", "in_bot"): s_bt,
        ("in_bot", "out_bot"): s_bb, ("out_bot", "in_bot"): s_bb,
    }
