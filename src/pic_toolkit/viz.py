"""Shared plotting helpers so every component's baseline notebook produces
visually consistent figures. No meep import -- these work on plain arrays.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import style


def _shade_pml(ax, extent_um: tuple, pml_um: float | None):
    """Shade the PML region -- a `pml_um`-wide strip inset from each of the
    4 edges of `extent_um` -- as semi-transparent gray. Draws 4 non-
    overlapping rectangles (left/right span the full height, top/bottom span
    only the strip between them) so no corner is double-shaded. Makes no
    assumption that `extent_um` is symmetric about the origin.
    """
    if not pml_um:
        return
    xmin, xmax, ymin, ymax = extent_um
    kwargs = dict(facecolor=style.COLOR_REFERENCE, edgecolor="none", alpha=0.35, zorder=2.5)
    ax.add_patch(Rectangle((xmin, ymin), pml_um, ymax - ymin, **kwargs))
    ax.add_patch(Rectangle((xmax - pml_um, ymin), pml_um, ymax - ymin, **kwargs))
    ax.add_patch(Rectangle((xmin + pml_um, ymin), (xmax - xmin) - 2 * pml_um, pml_um, **kwargs))
    ax.add_patch(Rectangle((xmin + pml_um, ymax - pml_um), (xmax - xmin) - 2 * pml_um, pml_um, **kwargs))


def plot_permittivity(eps: np.ndarray, extent_um: tuple, ax=None, ports=None, pml_um=None,
                       cmap: str = style.CMAP_PERMITTIVITY_BASELINE):
    """`ports`, if given, is a list of (x_um, y_um, label, orientation) tuples,
    `orientation` being "v" (vertical dashed line -- an x-normal port, e.g.
    propagation along x) or "h" (horizontal dashed line -- a y-normal port).
    Drawn as a full-span reference-plane line rather than a single point
    marker, since a port is a monitor PLANE, not a location.

    `pml_um`, if given, shades the PML region (see `_shade_pml`). `cmap`
    defaults to the baseline-geometry blue; pass `style.CMAP_PERMITTIVITY_
    OPTIMIZED` ("Greens") for a final/optimized design instead.
    """
    ax = ax or plt.gca()
    im = ax.imshow(
        eps.T, origin="lower", extent=extent_um, cmap=cmap, aspect="equal"
    )
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Permittivity")
    plt.colorbar(im, ax=ax, label="epsilon")
    _shade_pml(ax, extent_um, pml_um)
    for x, y, label, orientation in ports or []:
        if orientation == "v":
            ax.axvline(x, color=style.COLOR_ANNOTATION, linestyle="--", linewidth=1)
            ax.annotate(label, (x, extent_um[3]), color=style.COLOR_ANNOTATION, ha="center",
                        va="bottom", fontsize=9, annotation_clip=False)
        else:
            ax.axhline(y, color=style.COLOR_ANNOTATION, linestyle="--", linewidth=1)
            ax.annotate(label, (extent_um[1], y), color=style.COLOR_ANNOTATION, ha="left",
                        va="center", fontsize=9, annotation_clip=False)
    return ax


def plot_field(field: np.ndarray, eps: np.ndarray, extent_um: tuple, component: str, ax=None, pml_um=None):
    """Signed Re(field), diverging colormap (`style.CMAP_FIELD`) -- the de
    facto convention for EM field plots in the Meep/photonics community, kept
    deliberately rather than switched to an intensity map. The structure
    boundary is drawn as a thin contour line (not a filled, alpha-blended
    raster) so it doesn't mute the field colors.

    `pml_um`, if given, shades the PML region (see `_shade_pml`).
    """
    ax = ax or plt.gca()
    vmax = np.max(np.abs(field.real))
    im = ax.imshow(
        field.real.T,
        origin="lower",
        extent=extent_um,
        cmap=style.CMAP_FIELD,
        aspect="equal",
        vmin=-vmax,
        vmax=vmax,
    )
    eps_mid = (eps.min() + eps.max()) / 2
    if eps.max() > eps.min():
        ax.contour(eps.T, levels=[eps_mid], extent=extent_um, colors=style.COLOR_REFERENCE,
                   linewidths=0.8, alpha=0.8)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(f"Re({component})")
    plt.colorbar(im, ax=ax, label=f"Re({component})")
    _shade_pml(ax, extent_um, pml_um)
    return ax


def plot_sparams(wavelengths_um: np.ndarray, s_matrix: dict, pairs=("11", "21"), show_phase: bool = True):
    """`show_phase=False` drops the phase panel, leaving magnitude only -- for a
    resonant device with several resonances packed into the analyzed band, a
    correctly-unwrapped phase can still look like discrete jumps if each
    resonance's fast ~2pi excursion is undersampled by the wavelength grid
    (narrow dip, coarse grid) -- a sampling-density limitation, not a wrapping
    bug. See `05_racetrack_resonator.ipynb` Section 6 for that case."""
    if show_phase:
        fig, (ax_mag, ax_phase) = plt.subplots(1, 2, figsize=(11, 4))
    else:
        fig, ax_mag = plt.subplots(figsize=(6, 4))
    for key, color in zip(pairs, style.COLOR_CYCLE):
        s = s_matrix[key]
        ax_mag.plot(wavelengths_um, np.abs(s) ** 2, label=f"|S{key}|^2", color=color)
        if show_phase:
            ax_phase.plot(wavelengths_um, np.unwrap(np.angle(s)), label=f"phase(S{key})", color=color)
    ax_mag.set_xlabel("wavelength (um)")
    ax_mag.set_ylabel("power fraction")
    ax_mag.set_title("S-parameter magnitude")
    ax_mag.legend()
    if show_phase:
        ax_phase.set_xlabel("wavelength (um)")
        ax_phase.set_ylabel("phase (rad)")
        ax_phase.set_title("S-parameter phase")
        ax_phase.legend()
    fig.tight_layout()
    return fig
