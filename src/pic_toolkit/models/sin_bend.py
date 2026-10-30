"""SAX-compatible model for the SiN-track 90-degree bend.

Same model form as models/bend.py (loss_dB = floor + amplitude*exp(-R/decay),
phase linear in R, one fit per bend_type at the band center), read from the SiN
design point that sin/notebooks/02_sin_bent_waveguide.ipynb saves. Never
imports meep. The fit covers the SiN radius sweep stored in the design point;
do not extrapolate far outside it.
"""

from __future__ import annotations

from pathlib import Path

from .. import design_points
from .bend import bend_from_record

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_POINT_PATH = _REPO_ROOT / "sin" / "data" / "design_points" / "sin_bend.yaml"


def sin_bend(wl=1.35, radius_um: float | None = None, bend_type: str = "circular"):
    """SAX model function: wavelength (um), bend radius (um, defaults to the
    design radius), bend_type ("circular" or "euler") -> SDict. See
    models.bend.bend."""
    design_point = design_points.load_design_point(_DESIGN_POINT_PATH)
    return bend_from_record(design_point, wl, radius_um, bend_type)
