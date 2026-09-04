"""SAX-compatible component model for the directional coupler / 50:50 splitter.

Architectural rule: this file NEVER imports meep, and NEVER launches an FDTD
simulation -- same rule as models/racetrack.py, models/waveguide.py, etc. It
reads the cached, validated artifact that notebooks/06_directional_coupler.ipynb
already produced and selected, via the design point that names it.

Unlike models/racetrack.py, this does NOT import pic_toolkit.sparams
(hardcoded to a 2x2 s_matrix, can't represent this device's 4-port shape) or
pic_toolkit.meep_sim.coupler (imports meep at module level -- importing it
here would break the assert below even though none of ITS meep usage would
ever run). Instead this re-implements its own small, meep-free artifact
loader for meep_sim.coupler.save_artifact's .npz/.json format -- same
"duplication over a shared helper" convention racetrack.py itself already
established relative to the since-removed ring.py.

Returns genuine COMPLEX (magnitude and phase) S-parameters, from the
artifact's S_through/S_cross/S_reflect (+_bot) fields -- meep_sim.coupler.py
was extended to capture and persist these (previously it only kept
np.abs(...)**2, discarding phase before it ever reached the artifact; see
CLAUDE.md Sec 7's "persist complex S-parameters" note). This closes the gap
this module's own docstring used to flag: a power-only coupler model cannot
drive a genuinely interferometric MZI circuit, which depends on the
coupler's own through/cross phase relationship, not just its power split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .. import design_points

assert "meep" not in sys.modules, "pic_toolkit.models.coupler must never coexist with a meep import"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_POINT_PATH = _REPO_ROOT / "data" / "design_points" / "coupler.yaml"


def _resolve_artifact_stem(design_point: dict) -> Path:
    source = Path(design_point["source_artifact"])
    return source if source.is_absolute() else _REPO_ROOT / source


def _load_coupler_artifact(path_stem: Path) -> dict:
    """Meep-free re-implementation of meep_sim.coupler.load_artifact's
    .npz/.json format -- see that module's save_artifact for the writer.
    """
    path_stem = Path(path_stem)
    npz_path = path_stem.parent / (path_stem.name + ".npz")
    json_path = path_stem.parent / (path_stem.name + ".json")
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"Coupler artifact not found at {path_stem}(.npz/.json). "
            "Run notebooks/06_directional_coupler.ipynb first to generate it."
        )
    data = np.load(npz_path)
    metadata = json.loads(json_path.read_text())
    return {
        "wavelengths_um": data["wavelengths_um"],
        "S_through": data["S_through"], "S_cross": data["S_cross"], "S_reflect": data["S_reflect"],
        "S_through_bot": data["S_through_bot"], "S_cross_bot": data["S_cross_bot"],
        "S_reflect_bot": data["S_reflect_bot"],
        "port_names": tuple(metadata["port_names"]),
    }


def _interp_complex(wl, wl_grid, values):
    # wl_grid is wavelengths_um = 1/freqs from the artifact, i.e. DESCENDING
    # (freqs are saved increasing) -- np.interp silently requires its xp
    # argument ascending, so sort first rather than pass it through as-is
    # (same fix models/mzi.py's/models/bend_topopt.py's own _interp_complex
    # already applies).
    #
    # Interpolates MAGNITUDE and UNWRAPPED PHASE separately, not real/
    # imaginary parts independently -- see models/mzi_arm.py's own
    # _interp_complex docstring for why: linearly interpolating real/imag
    # parts across a step where phase winds by a large fraction of a full
    # turn cuts the chord across the true circular arc, underestimating
    # magnitude between grid points. This coupler's own path is short enough
    # that the old real/imag approach was not observed to cause a visible
    # problem, but there is no downside to the more robust method here too.
    wl = np.asarray(wl, dtype=float)
    order = np.argsort(wl_grid)
    wl_sorted, values_sorted = wl_grid[order], values[order]
    magnitude = np.interp(wl, wl_sorted, np.abs(values_sorted))
    phase = np.interp(wl, wl_sorted, np.unwrap(np.angle(values_sorted)))
    return magnitude * np.exp(1j * phase)


def _sdict_from_artifact(artifact: dict, wl) -> dict:
    wl_grid = artifact["wavelengths_um"]
    in_top, in_bot, out_top, out_bot = artifact["port_names"]

    def _s(key):
        return _interp_complex(wl, wl_grid, artifact[key])

    s_through, s_cross, s_r = _s("S_through"), _s("S_cross"), _s("S_reflect")
    s_through_bot, s_cross_bot, s_r_bot = _s("S_through_bot"), _s("S_cross_bot"), _s("S_reflect_bot")

    return {
        (in_top, out_top): s_through, (out_top, in_top): s_through,
        (in_top, out_bot): s_cross, (out_bot, in_top): s_cross,
        (in_top, in_top): s_r,
        (in_bot, out_bot): s_through_bot, (out_bot, in_bot): s_through_bot,
        (in_bot, out_top): s_cross_bot, (out_top, in_bot): s_cross_bot,
        (in_bot, in_bot): s_r_bot,
    }


def coupler(wl=1.35):
    """SAX model function: wavelength (um, scalar or array) -> SDict over the
    4 ports (in_top, in_bot, out_top, out_bot), with genuine complex
    (magnitude and phase) S-parameters.

    Loads the cached artifact referenced by data/design_points/coupler.yaml
    (the selected 50:50 design, coupling_length_um~=13.848). `simulate_
    baseline()` only excites in_top/in_bot (out_top/out_bot are output-only
    in this component's 50:50-splitter use case), so the reverse direction of
    each through/cross pair reuses the same measured complex value (assumed
    reciprocal, consistent with this device's own passivity/reciprocity
    validation) rather than being independently measured.
    """
    design_point = design_points.load_design_point(_DESIGN_POINT_PATH)
    artifact_stem = _resolve_artifact_stem(design_point)
    artifact = _load_coupler_artifact(artifact_stem)
    return _sdict_from_artifact(artifact, wl)


def coupler_at_sweep_point(coupling_length_um: float, wl=1.35) -> dict:
    """SAX model function for a coupler at a DIFFERENT coupling_length_um than
    the single selected 50:50 design point -- reads directly from
    `notebooks/06_directional_coupler.ipynb`'s own Section 10 sweep artifacts
    (`data/sparams/coupler/sweep/coupler_couplinglength<Lc>.npz`), which
    `meep_sim.coupler.py`'s own sweep loop calls `simulate_baseline()` for
    (same complex S_* fields as the selected point, not a separate/lesser
    measurement). No design point YAML exists for these (they were never
    individually re-validated/selected the way coupler.yaml's own point was),
    so this is for exploratory/comparison use (e.g. building a lattice filter
    stage at a non-50:50 ratio from already-measured real data) rather than a
    "trusted design" the way `coupler()` is -- callers should treat the
    resulting kappa as whatever this sweep point actually measured, not
    assume it hits a specific target exactly.

    `coupling_length_um` must match one of Section 10's swept values exactly
    (raises FileNotFoundError with a clear message otherwise, listing what
    exists, rather than silently interpolating between physically distinct
    geometries)."""
    stem = _REPO_ROOT / "data" / "sparams" / "coupler" / "sweep" / f"coupler_couplinglength{coupling_length_um}"
    npz_path = stem.parent / (stem.name + ".npz")
    if not npz_path.exists():
        available = sorted(p.stem.replace("coupler_couplinglength", "")
                            for p in stem.parent.glob("coupler_couplinglength*.npz"))
        raise FileNotFoundError(
            f"No sweep artifact at coupling_length_um={coupling_length_um} ({npz_path}). "
            f"Available swept values: {available}"
        )
    artifact = _load_coupler_artifact(stem)
    return _sdict_from_artifact(artifact, wl)
