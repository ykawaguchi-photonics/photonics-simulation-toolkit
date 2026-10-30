"""SAX-compatible model for the SiN-track straight waveguide.

Same model form as models/waveguide.py (dispersive phase from n_eff and n_g,
lossless, no reflection), read from the SiN design point that
sin/notebooks/01_sin_waveguide_baseline.ipynb saves. Never imports meep.
"""

from __future__ import annotations

from pathlib import Path

from .. import design_points
from .waveguide import waveguide_from_record

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_POINT_PATH = _REPO_ROOT / "sin" / "data" / "design_points" / "sin_waveguide.yaml"


def sin_waveguide(wl=1.35, length_um: float | None = None):
    """SAX model function: wavelength (um), length (um, defaults to the
    characterized length) -> SDict. See models.waveguide.waveguide."""
    design_point = design_points.load_design_point(_DESIGN_POINT_PATH)
    return waveguide_from_record(design_point, wl, length_um)
