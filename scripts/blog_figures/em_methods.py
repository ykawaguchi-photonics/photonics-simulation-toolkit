"""Figures and numbers for the blog article "FDTD, FEM and RCWA".

Fig. 1 is a schematic of how each method discretizes a problem (Yee grid,
triangular mesh, layered Fourier expansion). Fig. 2 re-plots the saved
mesh-convergence study of the SiN grating coupler (meta/data/step1,
one-atom starting design): coupling efficiency and wall time versus Meep
resolution for the two geometry representations (subpixel-averaged blocks
and a per-cell fill fraction). No FDTD is run and meep is never imported.

Usage (from the repo root):
    conda run -n mp python scripts/blog_figures/em_methods.py <out_dir>
"""

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.spatial import Delaunay

from pic_toolkit import style

REPO = Path(__file__).resolve().parents[2]
STEP1 = REPO / "meta" / "data" / "step1"

style.apply_style()
assert "meep" not in sys.modules


def arrow(ax, p0, p1, color="k", **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, color=color, lw=0.9, **kw))


def yee_panel(ax):
    """2D TE Yee grid: Hz at cell centres, Ex on horizontal edges, Ey on vertical edges."""
    n = 5
    ax.add_patch(Rectangle((0, 2), n, 1, fc=style.COLOR_CYAN, ec="none", alpha=0.45, zorder=0))
    for i in range(n + 1):
        ax.plot([i, i], [0, n], color=style.COLOR_REFERENCE, lw=0.6, zorder=1)
        ax.plot([0, n], [i, i], color=style.COLOR_REFERENCE, lw=0.6, zorder=1)
    for i in range(n):
        for j in range(n):
            ax.plot(i + 0.5, j + 0.5, "o", ms=4, color=style.COLOR_INDIGO, zorder=3)
    for i in range(n):
        for j in range(n + 1):
            ax.plot(i + 0.5, j, ">", ms=4, color=style.COLOR_TEAL, zorder=3)
    for i in range(n + 1):
        for j in range(n):
            ax.plot(i, j + 0.5, "^", ms=4, color=style.COLOR_OLIVE, zorder=3)
    ax.set_xlim(-0.3, n + 0.3)
    ax.set_ylim(-1.6, n + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("FDTD: Yee grid, stepped in time", fontsize=10)
    ax.plot([], [], "o", color=style.COLOR_INDIGO, ms=4, label="$H_z$ (cell centre)")
    ax.plot([], [], ">", color=style.COLOR_TEAL, ms=4, label="$E_x$ (edge)")
    ax.plot([], [], "^", color=style.COLOR_OLIVE, ms=4, label="$E_y$ (edge)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, frameon=False,
              handletextpad=0.2, columnspacing=0.8)


def fem_panel(ax):
    """Triangular mesh around a rectangular waveguide core, refined near the core."""
    rng = np.random.default_rng(1)
    w, h = 1.6, 0.8                       # core (drawn units)
    pts = [(x, y) for x in np.linspace(-2.5, 2.5, 9) for y in np.linspace(-1.8, 1.8, 7)]
    pts += [(x, y) for x in np.linspace(-w / 2, w / 2, 7) for y in (-h / 2, h / 2)]
    pts += [(x, y) for x in (-w / 2, w / 2) for y in np.linspace(-h / 2, h / 2, 4)]
    near = rng.uniform([-1.4, -0.9], [1.4, 0.9], size=(60, 2))
    near = near[(np.abs(near[:, 0]) > w / 2 + 0.08) | (np.abs(near[:, 1]) > h / 2 + 0.08)]
    inner = rng.uniform([-w / 2 + 0.1, -h / 2 + 0.1], [w / 2 - 0.1, h / 2 - 0.1], size=(10, 2))
    P = np.vstack([np.array(pts), near, inner])
    tri = Delaunay(P)
    cen = P[tri.simplices].mean(axis=1)
    incore = (np.abs(cen[:, 0]) < w / 2) & (np.abs(cen[:, 1]) < h / 2)
    ax.tripcolor(P[:, 0], P[:, 1], tri.simplices, facecolors=incore.astype(float),
                 cmap=matplotlib.colors.ListedColormap(["white", style.COLOR_CYAN]), alpha=0.6)
    ax.triplot(P[:, 0], P[:, 1], tri.simplices, color=style.COLOR_REFERENCE, lw=0.5)
    ax.add_patch(Rectangle((-w / 2, -h / 2), w, h, fill=False, ec=style.COLOR_INDIGO, lw=1.2))
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.8, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("FEM: mesh of the cross-section", fontsize=10)
    ax.text(0, -2.4, "small elements follow the core;\nsolved per frequency (here: modes)",
            ha="center", va="center", fontsize=7.5)


def rcwa_panel(ax):
    """One period of a layered grating: uniform layers + a patterned layer, Fourier orders out."""
    L = 4.0
    ax.add_patch(Rectangle((0, 0), L, 1.0, fc="0.85", ec="none"))                      # substrate
    for x0 in (0.0, 2.0):
        ax.add_patch(Rectangle((x0, 1.0), 1.2, 0.8, fc=style.COLOR_CYAN, ec="none", alpha=0.8))  # teeth
    ax.add_patch(Rectangle((0, 1.8), L, 0.5, fc=style.COLOR_TEAL, ec="none", alpha=0.35))      # film
    for y in (1.0, 1.8, 2.3):
        ax.plot([0, L], [y, y], color=style.COLOR_REFERENCE, lw=0.8, ls="--")
    for x in (0, 2.0, L):
        ax.plot([x, x], [0, 2.3], color=style.COLOR_INDIGO, lw=0.6, ls=":")
    arrow(ax, (1.2, 4.0), (2.0, 2.45))
    for ang, lab in ((-40, "−1"), (0, "0"), (40, "+1")):
        a = np.deg2rad(ang)
        arrow(ax, (2.9, 2.45), (2.9 + 1.0 * np.sin(a), 2.45 + 1.0 * np.cos(a)), color=style.COLOR_INDIGO)
        ax.text(2.9 + 1.25 * np.sin(a), 2.45 + 1.25 * np.cos(a), lab, ha="center", va="center", fontsize=7.5)
    ax.annotate("", xy=(0, -0.35), xytext=(2.0, -0.35),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="k"))
    ax.text(1.0, -0.65, "period Λ", ha="center", va="center", fontsize=7.5)
    ax.text(L + 0.15, 2.05, "layer 1", ha="left", va="center", fontsize=7)
    ax.text(L + 0.15, 1.4, "layer 2", ha="left", va="center", fontsize=7)
    ax.text(L + 0.15, 0.5, "substrate", ha="left", va="center", fontsize=7)
    ax.set_xlim(-0.3, L + 1.5)
    ax.set_ylim(-1.6, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("RCWA: layers × Fourier orders", fontsize=10)


def load_mesh(kind):
    rows = []
    for f in STEP1.glob(f"mesh_1atom_{kind}_res*.pkl"):
        res = int(re.findall(r"res(\d+)", f.name)[0])
        d = pickle.load(open(f, "rb"))
        rows.append((res, d["eta"], d["wall_s"]))
    return np.array(sorted(rows))


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numbers = {}

    # ---- Fig. 1: discretization schematics
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    yee_panel(axes[0])
    fem_panel(axes[1])
    rcwa_panel(axes[2])
    fig.tight_layout()
    fig.savefig(out / "fig1_discretization.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 2: FDTD resolution convergence and cost (saved SiN grating study)
    blk, grd = load_mesh("block"), load_mesh("grid")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.9))
    a1.plot(blk[:, 0], 10 * np.log10(blk[:, 1]), "o-", color=style.COLOR_INDIGO, ms=4,
            label="subpixel-averaged blocks")
    a1.plot(grd[:, 0], 10 * np.log10(grd[:, 1]), "s-", color=style.COLOR_TEAL, ms=4,
            label="per-cell fill fraction")
    a1.set_xlabel("resolution (pixels/µm)")
    a1.set_ylabel("coupling efficiency (dB)")
    a1.set_xscale("log")
    a1.set_xticks([20, 40, 80, 160])
    a1.set_xticklabels(["20", "40", "80", "160"])
    a1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    a2.loglog(grd[:, 0], grd[:, 2], "s", color=style.COLOR_TEAL, ms=4, label="FDTD runs (2D)")
    r = np.array([20, 160])
    c3 = np.exp(np.mean(np.log(grd[:, 2]) - 3 * np.log(grd[:, 0])))   # slope-3 line through the data
    a2.loglog(r, c3 * r ** 3.0, color=style.COLOR_REFERENCE, lw=1, ls="--",
              label="∝ resolution³")
    a2.set_xlabel("resolution (pixels/µm)")
    a2.set_ylabel("wall time per run (s)")
    a2.set_xticks([20, 40, 80, 160])
    a2.set_xticklabels(["20", "40", "80", "160"])
    a2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig2_convergence.png", dpi=200)
    plt.close(fig)

    slope = np.polyfit(np.log(grd[:, 0]), np.log(grd[:, 2]), 1)[0]
    numbers["convergence"] = dict(
        block_db=dict(zip(map(int, blk[:, 0]), map(float, 10 * np.log10(blk[:, 1])))),
        grid_db=dict(zip(map(int, grd[:, 0]), map(float, 10 * np.log10(grd[:, 1])))),
        wall_s=dict(zip(map(int, grd[:, 0]), map(float, grd[:, 2]))),
        walltime_slope=float(slope),
        block_spread_80_160_db=float(np.ptp(10 * np.log10(blk[blk[:, 0] >= 80, 1]))),
        grid_spread_80_160_db=float(np.ptp(10 * np.log10(grd[grd[:, 0] >= 80, 1]))),
    )
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "blog_em_methods")
