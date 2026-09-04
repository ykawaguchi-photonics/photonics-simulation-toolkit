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
