"""Shared color palette so every notebook draws from the same, deliberately
chosen colors instead of matplotlib's default `tab10` cycle.

Built from Paul Tol's colorblind-safe "muted" qualitative palette (SRON --
the standard reference for categorical colors in physics/astronomy
publishing), re-ordered so the backbone reads cool (blue -> teal -> green)
with the palette's own rose reserved as a sparing pink accent.

Usage convention:
- `COLOR_STEEL` -- "this is measured/baseline data" (raw FDTD points).
- `COLOR_SEAGREEN` (+ `CMAP_PERMITTIVITY_OPTIMIZED`) -- "this is the final/
  optimized result." Nothing else uses green, so it stays a clear signal.
- `COLOR_REFERENCE` (gray) -- fit/theory overlay lines, PML shading,
  residuals.
- `COLOR_CYCLE` -- any plot with 3+ data series (a sweep, a multi-type
  comparison), so multi-curve plots never fall back to matplotlib defaults.
- `COLOR_ROSE` -- sparing one-off emphasis (e.g. the single chosen design
  point picked out of a sweep). Never a full data series.
"""

from __future__ import annotations

# Cool backbone -- multi-series sweep plots (radius/length sweeps, bend-type
# comparisons, wavelength curves): use in this order so adjacent series are
# always visually distinct.
COLOR_INDIGO = "#332288"
COLOR_STEEL = "#2E6E9E"
COLOR_CYAN = "#88CCEE"
COLOR_TEAL = "#44AA99"
COLOR_GREEN = "#117733"
COLOR_SEAGREEN = "#2F8F5B"
COLOR_OLIVE = "#999933"
COLOR_PURPLE = "#AA4499"

COLOR_CYCLE = [COLOR_INDIGO, COLOR_CYAN, COLOR_TEAL, COLOR_GREEN, COLOR_OLIVE, COLOR_PURPLE]

# Neutral + accents
COLOR_REFERENCE = "#6C757D"
COLOR_ROSE = "#CC6677"
COLOR_ANNOTATION = "crimson"

CMAP_PERMITTIVITY_BASELINE = "Blues"
CMAP_PERMITTIVITY_OPTIMIZED = "Greens"
CMAP_FIELD = "RdBu"
CMAP_DENSITY = "viridis"


def apply_style() -> None:
    """Call once near the top of a notebook to make every subsequent plot
    draw from this palette and a consistent, uncluttered figure style."""
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update({
        "axes.prop_cycle": cycler(color=COLOR_CYCLE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 11,
    })
