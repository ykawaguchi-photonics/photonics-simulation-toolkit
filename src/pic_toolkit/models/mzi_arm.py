"""SAX-compatible component model for the 4-Euler-bend "jog" MZI delay arm.

Architectural rule: this file NEVER imports meep, and NEVER launches an FDTD
simulation. It only reads the cached, validated S-parameter artifact that
`notebooks/03_mzi_arm.ipynb` already produced (the representative baseline at
`delta_L_um=15.0`), via the design point that names it.

Same shape as `models/bend_topopt.py` -- the artifact was saved via the shared
`pic_toolkit.sparams.save_artifact` (a plain 2-port device, unlike
`coupler.py`'s own parallel 4-port artifact system), so this file reuses
`sparams.load_artifact` directly rather than re-implementing a loader.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .. import design_points, sparams

assert "meep" not in sys.modules, "pic_toolkit.models.mzi_arm must never coexist with a meep import"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_POINT_PATH = _REPO_ROOT / "data" / "design_points" / "mzi_arm.yaml"


def _resolve_artifact_stem(design_point: dict) -> Path:
    source = Path(design_point["source_artifact"])
    return source if source.is_absolute() else _REPO_ROOT / source


def _interp_complex(wl, wl_grid, s_grid):
    # wl_grid is wavelengths_um = 1/freqs from the artifact, i.e. DESCENDING
    # (freqs are saved increasing) -- np.interp silently requires its xp
    # argument ascending, so sort first, same fix models/bend_topopt.py's own
    # _interp_complex applies.
    #
    # Interpolates MAGNITUDE and UNWRAPPED PHASE separately, NOT real/imaginary
    # parts (the pattern models/mzi.py/models/bend_topopt.py/models/coupler.py
    # use) -- this component's physical path is long enough (delta_L_um=15
    # on top of the ~26um lead/bend span) that S21's phase winds by ~114deg
    # between this artifact's own n_freq=21 wavelength samples (confirmed
    # directly: np.diff(np.unwrap(np.angle(s21))) ~ -114deg/step). Linearly
    # interpolating real/imaginary parts independently across a step that
    # large cuts the chord across the true circular arc, UNDERESTIMATING
    # magnitude between grid points -- confirmed empirically: composing this
    # component into circuits/mzi_real_fdtd_sax.ipynb's SAX circuit with the
    # old real/imag interpolation produced a spurious total-power deficit up
    # to ~40%, oscillating with exactly this artifact's own 5nm grid spacing;
    # switching to magnitude+phase interpolation removed it entirely (see
    # docs/simulation_settings_record.md's mzi_arm section for the isolation
    # steps). Magnitude and phase both vary smoothly and slowly by comparison
    # (no wrapping issue for magnitude; phase is unwrapped before
    # interpolating), so this is safe even at large per-step phase changes.
    wl = np.asarray(wl, dtype=float)
    order = np.argsort(wl_grid)
    wl_sorted, s_sorted = wl_grid[order], s_grid[order]
    magnitude = np.interp(wl, wl_sorted, np.abs(s_sorted))
    phase = np.interp(wl, wl_sorted, np.unwrap(np.angle(s_sorted)))
    return magnitude * np.exp(1j * phase)


def mzi_arm(wl=1.35):
    """SAX model function: wavelength (um, scalar or array) -> SDict over the
    2 ports (o1, o2), genuine complex (magnitude and phase) S-parameters.

    Loads the cached artifact referenced by data/design_points/mzi_arm.yaml --
    the representative baseline geometry at delta_L_um=15.0 (matching the
    SAX-designed lattice-filter notebook). Never runs a new simulation, and
    does not interpolate across delta_L_um -- unlike wavelength, delta_L_um
    is a discrete design choice fixed at the artifact's own baseline, same
    convention as models/mzi.py/models/racetrack.py (one selected design
    point, not a runtime-adjustable geometry parameter).
    """
    design_point = design_points.load_design_point(_DESIGN_POINT_PATH)
    artifact_stem = _resolve_artifact_stem(design_point)
    artifact = sparams.load_artifact(artifact_stem)

    wl_grid = artifact.wavelengths_um
    o1, o2 = artifact.port_names

    return {
        (o1, o1): _interp_complex(wl, wl_grid, artifact.s_matrix["11"]),
        (o1, o2): _interp_complex(wl, wl_grid, artifact.s_matrix["12"]),
        (o2, o1): _interp_complex(wl, wl_grid, artifact.s_matrix["21"]),
        (o2, o2): _interp_complex(wl, wl_grid, artifact.s_matrix["22"]),
    }


# radius_um -> the sweep subdirectory holding that radius's own extra_straight_um-tagged
# artifacts. `03_mzi_arm.ipynb` Section 10's own sweep only ever varied extra_straight_um at
# a single fixed radius_um=5.0 (data/sparams/mzi_arm/sweep/) -- its floor delta_L (13.272um)
# cannot reach the Mux4 tree's own stage2 targets (7.5/7.64um), so a second radius (2.5um,
# floor 6.632um) was swept separately into its own directory to avoid any filename ambiguity
# with the existing radius=5.0 points (grid_sweep's own tag encodes only the SWEPT param,
# extra_straight_um, not radius_um).
_SWEEP_DIRS_BY_RADIUS = {
    5.0: _REPO_ROOT / "data" / "sparams" / "mzi_arm" / "sweep",
    2.5: _REPO_ROOT / "data" / "sparams" / "mzi_arm" / "sweep_r2.5",
}


def mzi_arm_at_sweep_point(extra_straight_um: float, radius_um: float = 5.0, wl=1.35) -> dict:
    """SAX model function for an mzi_arm at a specific (radius_um, extra_straight_um)
    OTHER than the single selected design point (radius_um=5.0, extra_straight_um=0.864,
    delta_L_um=15.0) -- generalizes this module the same way `models.mzi.
    mzi_at_sweep_point`/`models.coupler.coupler_at_sweep_point` already do for their own
    components. Reads directly from the relevant sweep directory's own
    `mzi_arm_extra_straight<X>.npz` artifact (`03_mzi_arm.ipynb` Section 10, or the
    Mux4-tree-specific radius_um=2.5 sweep -- see `_SWEEP_DIRS_BY_RADIUS` above). No
    interpolation across extra_straight_um/radius_um (a discrete geometry choice, not a
    runtime-adjustable parameter) -- raises FileNotFoundError listing available values on
    a miss, exactly like `mzi_at_sweep_point`.

    Uses the same magnitude+unwrapped-phase wavelength interpolation as `mzi_arm()` above
    (`_interp_complex`) -- this component's own long physical path already established
    that convention (see that function's docstring)."""
    if radius_um not in _SWEEP_DIRS_BY_RADIUS:
        raise ValueError(
            f"No sweep data for radius_um={radius_um}. Available radii: "
            f"{sorted(_SWEEP_DIRS_BY_RADIUS)}"
        )
    sweep_dir = _SWEEP_DIRS_BY_RADIUS[radius_um]
    stem = sweep_dir / f"mzi_arm_extra_straight{extra_straight_um}"
    npz_path = stem.parent / (stem.name + ".npz")
    if not npz_path.exists():
        available = sorted(
            p.stem.replace("mzi_arm_extra_straight", "") for p in sweep_dir.glob("mzi_arm_extra_straight*.npz")
        )
        raise FileNotFoundError(
            f"No sweep artifact at radius_um={radius_um}, extra_straight_um={extra_straight_um} "
            f"({npz_path}). Available extra_straight_um values at this radius: {available}"
        )
    artifact = sparams.load_artifact(stem)
    wl_grid = artifact.wavelengths_um
    o1, o2 = artifact.port_names

    return {
        (o1, o1): _interp_complex(wl, wl_grid, artifact.s_matrix["11"]),
        (o1, o2): _interp_complex(wl, wl_grid, artifact.s_matrix["12"]),
        (o2, o1): _interp_complex(wl, wl_grid, artifact.s_matrix["21"]),
        (o2, o2): _interp_complex(wl, wl_grid, artifact.s_matrix["22"]),
    }
