"""SAX-compatible component model for the fabrication-bias-robust
adjoint-topology-optimized bend.

Architectural rule: this file NEVER imports meep, and NEVER launches an FDTD
simulation. It only reads the cached, validated S-parameter artifact that
`04b_bend_topopt_robust.ipynb` already produced (from the honest, plain
forward two-port `simulate_baseline()` validation of the design's NOMINAL
sub-design -- not the optimizer's own per-iteration objective, and not the
eroded/dilated sub-designs, which are QA/provenance data recorded in the
design point's `robustness` block, not exposed here -- a circuit composition
wants one typical S-matrix per component, same as every other model in this
toolkit).

Same shape as models/bend_topopt.py -- see that file for the general pattern
this mirrors. This is a genuinely separate component from `bend_topopt`
(notebook 04's own design point/model are untouched by this one).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .. import design_points, sparams

assert "meep" not in sys.modules, "pic_toolkit.models.bend_topopt_robust must never coexist with a meep import"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_POINT_PATH = _REPO_ROOT / "data" / "design_points" / "bend_topopt_robust.yaml"


def _resolve_artifact_stem(design_point: dict) -> Path:
    source = Path(design_point["source_artifact"])
    return source if source.is_absolute() else _REPO_ROOT / source


def _interp_complex(wl, wl_grid, s_grid):
    # wl_grid is wavelengths_um = 1/freqs from the artifact, i.e. DESCENDING
    # (freqs are saved increasing) -- np.interp silently requires its xp
    # argument ascending, so sort first rather than pass it through as-is.
    wl = np.asarray(wl, dtype=float)
    order = np.argsort(wl_grid)
    wl_sorted = wl_grid[order]
    real = np.interp(wl, wl_sorted, s_grid.real[order])
    imag = np.interp(wl, wl_sorted, s_grid.imag[order])
    return real + 1j * imag


def bend_topopt_robust(wl=1.35):
    """SAX model function: wavelength (um, scalar or array) -> SDict.

    Loads the cached S-parameter artifact referenced by
    data/design_points/bend_topopt_robust.yaml (the NOMINAL sub-design's
    honest measurement). Never runs a new simulation.
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
