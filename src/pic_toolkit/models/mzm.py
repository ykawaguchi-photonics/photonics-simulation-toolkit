"""SAX-compatible component model for the Mach-Zehnder Modulator.

Architectural rule: this file NEVER imports meep, and NEVER launches an FDTD
simulation -- same rule as models/mzi.py, models/coupler.py, etc. It reads
the cached, validated artifact that notebooks/08_mzm.ipynb already produced
at the Vpi push-pull-ON operating point, via the design point that names it.

Unlike models/coupler.py (sqrt(power)-only, zero-phase S-parameters), this
model returns genuine COMPLEX S-parameters directly, since
meep_sim.mzm.simulate_baseline records real phase, not just power -- same
convention models/mzi.py already established.

This model exposes ONE fixed operating point (the cached Vpi-ON baseline);
it does not interpolate across Delta_n the way a real voltage-driven device
would need to. A Delta_n-dependent circuit model is future work -- see
notebooks/08_mzm.ipynb Section 10's Delta_n sweep for the underlying
Meep-measured data this would be built from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .. import design_points

assert "meep" not in sys.modules, "pic_toolkit.models.mzm must never coexist with a meep import"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_POINT_PATH = _REPO_ROOT / "data" / "design_points" / "mzm.yaml"


def _resolve_artifact_stem(design_point: dict) -> Path:
    source = Path(design_point["source_artifact"])
    return source if source.is_absolute() else _REPO_ROOT / source


def _load_mzm_artifact(path_stem: Path) -> dict:
    """Meep-free re-implementation of meep_sim.mzm.load_artifact's
    .npz/.json format -- see that module's save_artifact for the writer."""
    path_stem = Path(path_stem)
    npz_path = path_stem.parent / (path_stem.name + ".npz")
    json_path = path_stem.parent / (path_stem.name + ".json")
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"MZM artifact not found at {path_stem}(.npz/.json). "
            "Run notebooks/08_mzm.ipynb first to generate it."
        )
    data = np.load(npz_path)
    metadata = json.loads(json_path.read_text())
    return {
        "wavelengths_um": data["wavelengths_um"],
        "S11": data["S11"], "S21": data["S21"], "S31": data["S31"], "S41": data["S41"],
        "S22": data["S22"], "S12": data["S12"], "S32": data["S32"], "S42": data["S42"],
        "port_names": tuple(metadata["port_names"]),
    }


def _interp_complex(wl, wl_grid, values):
    # wl_grid is wavelengths_um = 1/freqs from the artifact, i.e. DESCENDING
    # (freqs are saved increasing) -- np.interp silently requires its xp
    # argument ascending, so sort first, same fix models/mzi.py's
    # _interp_complex already applies.
    wl = np.asarray(wl, dtype=float)
    order = np.argsort(wl_grid)
    wl_sorted, values_sorted = wl_grid[order], values[order]
    re = np.interp(wl, wl_sorted, values_sorted.real)
    im = np.interp(wl, wl_sorted, values_sorted.imag)
    return re + 1j * im


def mzm(wl=1.35):
    """SAX model function: wavelength (um, scalar or array) -> SDict over the
    4 ports (in_top, in_bot, out_top, out_bot), at the cached Vpi push-pull-ON
    operating point, with genuine complex (magnitude and phase) S-parameters.

    Loads the cached artifact referenced by data/design_points/mzm.yaml.
    """
    design_point = design_points.load_design_point(_DESIGN_POINT_PATH)
    artifact_stem = _resolve_artifact_stem(design_point)
    artifact = _load_mzm_artifact(artifact_stem)

    wl_grid = artifact["wavelengths_um"]
    in_top, in_bot, out_top, out_bot = artifact["port_names"]

    def _s(key):
        return _interp_complex(wl, wl_grid, artifact[key])

    s11, s31, s41 = _s("S11"), _s("S31"), _s("S41")
    s22, s32, s42 = _s("S22"), _s("S32"), _s("S42")

    return {
        (in_top, in_top): s11,
        (in_top, out_top): s31, (out_top, in_top): s31,
        (in_top, out_bot): s41, (out_bot, in_top): s41,
        (in_bot, in_bot): s22,
        (in_bot, out_top): s32, (out_top, in_bot): s32,
        (in_bot, out_bot): s42, (out_bot, in_bot): s42,
    }
