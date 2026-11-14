"""Figures for the blog article "Topological Photonic Crystals".

Reads only the caches of topological/notebooks/01-03 (topological/data/bands, walls,
coupling/gm) -- no MPB or FDTD is run. pic_toolkit.meep_sim.topological_pc is imported
only for the crystal geometry (unit-cell permittivity on a grid).

Usage (from the repo root):
    conda run -n mp python scripts/blog_figures/topological_photonics.py <out_dir>
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from pic_toolkit import style
from pic_toolkit.meep_sim import topological_pc as tp

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "topological" / "data"
style.apply_style()

# Shared bulk gaps of notebook 01 (MPB, resolution 48; printed in its gap table).
GAP_NB01 = {"spin_te": (0.4014, 0.4142), "valley_te": (0.2305, 0.2466)}


def fig_unit_cells(out):
    """Fig. 1: TE (air-hole) unit cells, tiled: honeycomb, shrunken, expanded, valley-Hall."""
    cells = [("spin_te", "honeycomb", "honeycomb\n(a0/R = 3, gapless)", "a0"),
             ("spin_te", "trivial", "shrunken clusters\n(a0/R = 3.125, trivial)", "a0"),
             ("spin_te", "topological", "expanded clusters\n(a0/R = 2.9, topological)", "a0"),
             ("valley_te", "vpc1", "valley-Hall\n(unequal triangles)", "a")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0), layout="constrained")
    for ax, (kind, var, title, unit) in zip(axes, cells):
        e = tp.cell_epsilon(kind, var, resolution=48)
        for m in range(-3, 4):
            for n in range(-3, 4):
                X, Y = tp.cell_corner_grid(e.shape[0], m, n)
                ax.pcolormesh(X, Y, e, cmap=style.CMAP_PERMITTIVITY_BASELINE, vmin=1, vmax=tp.EPS_D,
                              shading="flat", rasterized=True)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"x / {unit}")
    axes[0].set_ylabel("y / lattice constant")
    sm = plt.cm.ScalarMappable(cmap=style.CMAP_PERMITTIVITY_BASELINE, norm=plt.Normalize(1, tp.EPS_D))
    fig.colorbar(sm, ax=axes, shrink=0.75, label="ε (air 1, silicon 11.7)")
    fig.savefig(out / "fig1_unit_cells.png", dpi=200)
    plt.close(fig)


def fig_poincare(out):
    """Static version of the Poincaré-sphere sketch (the post uses the interactive PoincareSphere
    component instead; kept for reference, not called)."""
    fig = plt.figure(figsize=(6.2, 5.6))
    ax = fig.add_subplot(projection="3d")
    u, v = np.linspace(0, 2 * np.pi, 60), np.linspace(0, np.pi, 30)
    ax.plot_wireframe(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                      np.outer(np.ones_like(u), np.cos(v)), color=style.COLOR_REFERENCE, lw=0.3, alpha=0.5)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), 0 * t, color=style.COLOR_REFERENCE, lw=0.8)  # equator
    # loop in the S1-S3 plane: H -> north -> V -> south -> H; it encloses a hemisphere
    loop = np.array([np.cos(t), 0 * t, np.sin(t)])
    ax.plot(*loop, color=style.COLOR_STEEL, lw=2.4)
    for k in (25, 75, 125, 175):
        a, b = loop[:, k], loop[:, k + 4]
        ax.quiver(*a, *(b - a), color=style.COLOR_STEEL, arrow_length_ratio=2.5, lw=1.5)
    pts = {"linear (S1 = +1)": (1, 0, 0), "linear (S1 = −1)": (-1, 0, 0),
           "circular (S3 = +1)": (0, 0, 1), "opposite circular (S3 = −1)": (0, 0, -1)}
    offs = {"linear (S1 = +1)": (1.95, 0, 0), "linear (S1 = −1)": (-2.0, 0, 0),
            "circular (S3 = +1)": (0, 0, 1.3), "opposite circular (S3 = −1)": (0, 0, -1.35)}
    for lab, p in pts.items():
        ax.scatter(*p, color=style.COLOR_ROSE if "circular" in lab else style.COLOR_STEEL, s=30, depthshade=False)
        ax.text(*offs[lab], lab, ha="center", va="center", fontsize=9)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.view_init(elev=18, azim=-60)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_zlim(-1.6, 1.6)
    fig.savefig(out / "fig2_poincare.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _bulk(ax, kind, var, ylim, gap):
    b = dict(np.load(DATA / "bands" / f"bulk_{kind}_{var}.npz"))
    f, d, t = b["freqs"], b["dist"], b["ticks"]
    ax.axhspan(*gap, color=style.COLOR_REFERENCE, alpha=0.25, lw=0)
    if "weights" in b:
        w = np.nan_to_num(b["weights"])
        pd = w[..., 1] + w[..., 2]
        pfrac = np.divide(w[..., 1], pd, out=np.zeros_like(pd), where=pd > 0)
        for n in range(f.shape[1]):
            dom = pd[:, n] > 0.5
            ax.scatter(d[dom], f[dom, n], c=pfrac[dom, n], cmap=style.CMAP_DENSITY, vmin=0, vmax=1, s=6)
            ax.scatter(d[~dom], f[~dom, n], c="k", s=6)
    else:
        for n in range(f.shape[1]):
            ax.scatter(d, f[:, n], color=style.COLOR_STEEL, s=6)
    ax.set_xticks(t, ["Γ", "M", "K", "Γ"]); ax.set_xlim(d[0], d[-1]); ax.set_ylim(*ylim)
    for x in t[1:-1]:
        ax.axvline(x, color=style.COLOR_REFERENCE, lw=0.5)


def fig_bulk_bands(out):
    """Fig. 3: TE bulk bands, spin-Hall shrunken / expanded (p/d colour) and valley-Hall."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    _bulk(axes[0], "spin_te", "trivial", (0.30, 0.50), GAP_NB01["spin_te"])
    _bulk(axes[1], "spin_te", "topological", (0.30, 0.50), GAP_NB01["spin_te"])
    _bulk(axes[2], "valley_te", "vpc1", (0.10, 0.35), GAP_NB01["valley_te"])
    axes[0].set_title("spin-Hall, shrunken (trivial)", fontsize=10)
    axes[1].set_title("spin-Hall, expanded (topological)", fontsize=10)
    axes[2].set_title("valley-Hall (gap at K)", fontsize=10)
    axes[0].set_ylabel("frequency (c/a0)"); axes[1].set_ylabel("frequency (c/a0)")
    axes[2].set_ylabel("frequency (c/a)")
    sm = plt.cm.ScalarMappable(cmap=style.CMAP_DENSITY); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[:2], shrink=0.7, ticks=[0, 1], location="bottom", aspect=40)
    cb.ax.set_xticklabels(["d-like", "p-like"])
    fig.savefig(out / "fig3_bulk_bands.png", dpi=200)
    plt.close(fig)


def fig_edge_bands(out):
    """Fig. 4: ribbon bands; edge states (coloured) cross the shared bulk gap (gray)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    yr = {"spin_te": (0.38, 0.44), "valley_te": (0.20, 0.27)}
    unit = {"spin_te": "a0", "valley_te": "a"}
    title = {"spin_te": "spin-Hall wall (edge bands cross at Γ)", "valley_te": "valley-Hall wall (edge band near K)"}
    for ax, kind in zip(axes, ("spin_te", "valley_te")):
        rb = dict(np.load(DATA / "bands" / f"ribbon_{kind}.npz"))
        kx, f = rb["kx"], rb["freqs"]
        # All states in one style: inside the shared bulk gap every state is a wall state;
        # outside it, wall and bulk states are not distinguished.
        ax.axhspan(*GAP_NB01[kind], color=style.COLOR_REFERENCE, alpha=0.25, lw=0)
        for sgn in (1, -1):
            for n in range(f.shape[1]):
                ax.scatter(sgn * kx, f[:, n], color=style.COLOR_STEEL, s=5)
        if kind == "valley_te":
            for kk in (-1 / 3, 1 / 3):
                ax.axvline(kk, color=style.COLOR_REFERENCE, ls=":", lw=0.8)
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(*yr[kind])
        ax.set_xlabel(f"kx (2π/{unit[kind]})"); ax.set_ylabel(f"frequency (c/{unit[kind]})")
        ax.set_title(title[kind], fontsize=10)
    fig.savefig(out / "fig4_edge_bands.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_one_way(out):
    """Fig. 5: valley-Hall hole wall driven by S+ and S- dipoles: |E|^2 goes one way."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), layout="constrained")
    for ax, (s, lab) in zip(axes, (("p", "S+ source"), ("m", "S− source"))):
        o = dict(np.load(DATA / "walls" / f"valley_te_f0.2368_s{s}.npz"))
        x, y = o["x"], o["y"]
        ix = np.abs(x) <= o["n_x"] / 2
        iy = np.abs(y) <= 6.0
        X, Y = np.meshgrid(x[ix], y[iy], indexing="ij")
        E2 = o["E2"][np.ix_(ix, iy)]
        far = np.hypot(X - o["src_xy"][0], Y - o["src_xy"][1]) > 1.5
        im = ax.pcolormesh(x[ix], y[iy], (E2 / E2[far].max()).T, cmap="inferno", norm=LogNorm(1e-3, 1),
                           shading="auto", rasterized=True)
        ax.contour(x[ix], y[iy], o["eps"][np.ix_(ix, iy)].T, levels=[6], colors="w", linewidths=0.15)
        ax.plot(*o["src_xy"], "*", color=style.COLOR_ROSE, ms=11, mec="k")
        tot = o["P_left"] + o["P_right"] + o["P_top"] + o["P_bottom"]
        pl, pr = o["P_left"] / tot, o["P_right"] / tot
        eta = (pl - pr) / (pl + pr)
        ax.set_title(f"{lab}: {pl:.3f} of the power goes left, {pr:.3f} right (η = {eta:+.3f})",
                     fontsize=10)
        ax.set_aspect("equal"); ax.set_ylabel("y / a")
    axes[1].set_xlabel("x / a  (domain wall along y = 0)")
    fig.colorbar(im, ax=axes, shrink=0.7, label="|E|² (normalized, log)")
    fig.savefig(out / "fig5_one_way.png", dpi=200)
    plt.close(fig)


def fig_coupling(out):
    """Fig. 6: strip -> wall forward power (full height), butt joint vs topology-optimized."""
    runs = {"spin": ("gm_spin_{}_r32n12", tp.A0_NM, (0.4015, 0.4065), "spin-Hall wall"),
            "valley": ("gm_valley_{}_r36n20", tp.A_NM, (0.2355, 0.2385), "valley-Hall wall")}
    gaps = {"spin": "gm_ribbon_spin_te_r32", "valley": "gm_ribbon_valley_te_r36"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True, layout="constrained")
    for ax, (sc, (tag, a_nm, band, title)) in zip(axes, runs.items()):
        g = np.load(DATA / "bands" / f"{gaps[sc]}.npz")["gap"]
        ax.axvspan(a_nm / g[1], a_nm / g[0], color=style.COLOR_REFERENCE, alpha=0.2, lw=0)
        ax.axvspan(a_nm / band[1], a_nm / band[0], color=style.COLOR_CYAN, alpha=0.3, lw=0)
        for case, col, lab in (("butt", style.COLOR_STEEL, "butt joint"),
                               ("topopt2", style.COLOR_SEAGREEN, "topology-optimized transition")):
            o = np.load(DATA / "coupling" / "gm" / f"{tag.format(case)}.npz")
            ax.plot(a_nm / o["freqs"], o["T_all"], color=col, lw=2, label=lab)
        ax.set_xlim(a_nm / (g[1] + 0.006 * (2 if sc == "spin" else 1)), a_nm / (g[0] - 0.006 * (2 if sc == "spin" else 1)))
        ax.set_ylim(0, 1); ax.set_xlabel("wavelength (nm)"); ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("forward power / incident strip power")
    from matplotlib.patches import Patch
    h, l = axes[0].get_legend_handles_labels()
    h += [Patch(color=style.COLOR_REFERENCE, alpha=0.2, lw=0), Patch(color=style.COLOR_CYAN, alpha=0.3, lw=0)]
    l += ["shared bulk gap", "operating band"]
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=9)
    fig.savefig(out / "fig6_coupling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    fig_unit_cells(out)
    fig_bulk_bands(out)
    fig_edge_bands(out)
    fig_one_way(out)
    tp.set_gap_matched()  # notebook 03 crystals (only the scale constants are used below)
    fig_coupling(out)
    print("done", sorted(p.name for p in out.glob("*.png")))
