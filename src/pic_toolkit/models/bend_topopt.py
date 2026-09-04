"""SAX-compatible component model for the adjoint-topology-optimized bend.

Architectural rule: this file NEVER imports meep, and NEVER launches an FDTD
simulation. It only reads the cached, validated S-parameter artifact that the
`04_bend_topology_optimization.ipynb` notebook already produced (from the
honest, plain forward two-port `simulate_baseline()` validation, not from the
optimizer's own per-iteration objective), via the design point that names it.

Same shape as the since-removed models/ring.py -- see that file for the general pattern this
mirrors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .. import design_points, sparams

assert "meep" not in sys.modules, "pic_toolkit.models.bend_topopt must never coexist with a meep import"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_POINT_PATH = _REPO_ROOT / "data" / "design_points" / "bend_topopt.yaml"


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


def bend_topopt(wl=1.35):
    """SAX model function: wavelength (um, scalar or array) -> SDict.

    Loads the cached S-parameter artifact referenced by
    data/design_points/bend_topopt.yaml. Never runs a new simulation.
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
