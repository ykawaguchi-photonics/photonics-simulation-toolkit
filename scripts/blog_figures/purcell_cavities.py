"""Figures for the blog article "Purcell Effect and Photonic-Crystal Nanocavities".

Reads only the caches of sin_cavity/notebooks/01-03 (sin_cavity/data/bands, l3, nanobeam)
-- no MPB or FDTD is run and meep is never imported.

Usage (from the repo root):
    conda run -n mp python scripts/blog_figures/purcell_cavities.py <out_dir>
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from pic_toolkit import style

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "sin_cavity" / "data"
style.apply_style()
assert "meep" not in sys.modules


def load(rel):
    return dict(np.load(DATA / rel))


def fig_geometry(out):
    """Fig. 1: top views of the final L3 (shifted holes coloured) and the own N = 15 nanobeam."""
    l3, nb = load("l3/l3_final_nb.npz"), load("nanobeam/own_N15.npz")
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6.2), layout="constrained",
                                   gridspec_kw=dict(height_ratios=[2.2, 1]))
    ax0.add_patch(Rectangle((-2.2, -1.2), 4.4, 2.4, color=style.COLOR_STEEL, alpha=0.35, lw=0))
    row0 = np.abs(l3["holes"][:, 1]) < 1e-6
    shifted = np.unique(np.round(np.abs(l3["holes"][row0, 0]), 6))[:3]
    cols = dict(zip(shifted, (style.COLOR_CYCLE[0], style.COLOR_CYCLE[2], style.COLOR_CYCLE[4])))
    for (hx, hy, hr), on in zip(l3["holes"], row0):
        if abs(hx) < 2.3 and abs(hy) < 1.3:
            fc = cols.get(round(abs(hx), 6), "white") if on else "white"
            ax0.add_patch(Circle((hx, hy), hr, fc=fc, ec="k", lw=0.4))
    for k, c in enumerate(cols.values()):
        ax0.plot([], [], "o", ms=8, mfc=c, mec="k", ls="none", label=f"end hole shifted by S{k + 1}")
    ax0.set_xlim(-2.2, 2.2); ax0.set_ylim(-1.2, 1.2); ax0.set_aspect("equal")
    ax0.set_title(f"L3 cavity: three missing holes, a = {l3['a'] * 1e3:.1f} nm, r = 0.30 a", fontsize=10)
    ax0.set_ylabel("y (µm)")
    ax0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
    w = float(nb["w"])
    ax1.add_patch(Rectangle((-4.2, -w / 2), 8.4, w, color=style.COLOR_STEEL, alpha=0.35, lw=0))
    for hx, hr in nb["holes"]:
        for s in (1, -1):
            if abs(hx) < 4.2:
                ax1.add_patch(Circle((s * hx, 0), hr, fc="white", ec="k", lw=0.4))
    ax1.set_xlim(-4.2, 4.2); ax1.set_ylim(-0.45, 0.45); ax1.set_aspect("equal")
    ax1.set_title("nanobeam cavity: 300 nm wide beam, hole radius tapered toward the centre", fontsize=10)
    ax1.set_xlabel("x (µm)"); ax1.set_ylabel("y (µm)")
    fig.savefig(out / "fig1_geometries.png", dpi=200)
    plt.close(fig)


def fig_slab_gap(out):
    """Fig. 2: guided TE-like gap of the triangular hole lattice in a 200 nm SiN membrane."""
    B = load("bands/slab_gap.npz")
    gap = 200 * (B["gap_hi"] - B["gap_lo"]) / (B["gap_hi"] + B["gap_lo"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    for j, a in enumerate(B["a_um"]):
        ax.plot(B["r_a"], gap[:, j], "o-", ms=4, color=style.COLOR_CYCLE[j], label=f"a = {a * 1e3:.0f} nm")
    ax.axvline(0.30, color=style.COLOR_REFERENCE, ls="--", lw=1)
    ax.set_xlabel("hole radius r / a"); ax.set_ylabel("guided TE-like gap (%)"); ax.set_ylim(0, 20)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=8)
    fig.savefig(out / "fig2_slab_gap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _field(ax, o, xlim, ylim, title):
    bx, by = o["box"][:2]
    x = np.linspace(-bx / 2, bx / 2, o["e2_mid"].shape[0])
    y = np.linspace(-by / 2, by / 2, o["e2_mid"].shape[1])
    im = ax.pcolormesh(x, y, (o["e2_mid"] / o["e2_mid"].max()).T, cmap="inferno", vmin=0, vmax=1,
                       shading="auto", rasterized=True)
    ax.contour(x, y, o["eps_mid"].T, levels=[2.5], colors="w", linewidths=0.4)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_title(title, fontsize=10); ax.set_ylabel("y (µm)")
    return im


def fig_fields(out):
    """Fig. 3: |E|^2 in the mid-plane, final L3 (top) and own N = 15 nanobeam (bottom)."""
    l3, nb = load("l3/l3_final_nb.npz"), load("nanobeam/own_N15.npz")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.0), layout="constrained",
                             gridspec_kw=dict(height_ratios=[2.2, 1]))
    _field(axes[0], l3, (-2.2, 2.2), (-1.1, 1.1),
           f"L3: λ = {1e3 / l3['f']:.0f} nm, Q = {l3['Q']:.1e}, V = {l3['V']:.2f} (λ/n)³")
    im = _field(axes[1], nb, (-4.2, 4.2), (-0.45, 0.45),
                f"nanobeam: λ = {1e3 / nb['f']:.0f} nm, Q ~ 10⁶–10⁷, V = {nb['V']:.2f} (λ/n)³")
    axes[1].set_xlabel("x (µm)")
    fig.colorbar(im, ax=axes, shrink=0.7, label="|E|² (normalized)")
    fig.savefig(out / "fig3_fields.png", dpi=200)
    plt.close(fig)


def fig_reproduction(out):
    """Fig. 4: Q and V vs cavity gap s, this model vs McCutcheon & Loncar (2008, read from their Fig. 4)."""
    s_ours = [80, 85, 90, 95, 100, 105, 110]
    runs = {s: load(f"nanobeam/{'mcc_edge_res64' if s == 95 else f'mcc_edge_s{s}'}.npz") for s in s_ours}
    paper_s = [80, 90, 93, 95, 97, 100, 110]
    paper_q = [1.2e4, 7.7e4, 1.6e5, 2.3e5, 1.9e5, 1.1e5, 2.1e4]
    paper_v = {80: 0.46, 90: 0.52, 95: 0.55, 100: 0.58, 110: 0.64}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    ax0.semilogy(s_ours, [runs[s]["Q"] for s in s_ours], "o-", color=style.COLOR_STEEL, label="this model (Meep 3D)")
    ax0.semilogy(paper_s, paper_q, "s--", color=style.COLOR_REFERENCE, label="McCutcheon & Lončar 2008")
    ax0.set_xlabel("cavity gap s (nm)"); ax0.set_ylabel("Q"); ax0.set_ylim(3e3, 5e5)
    ax1.plot(s_ours, [runs[s]["V"] for s in s_ours], "o-", color=style.COLOR_STEEL)
    ax1.plot(list(paper_v), list(paper_v.values()), "s--", color=style.COLOR_REFERENCE)
    ax1.set_xlabel("cavity gap s (nm)"); ax1.set_ylabel("V ((λ/n)³)"); ax1.set_ylim(0.4, 0.7)
    fig.legend(*ax0.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9)
    fig.savefig(out / "fig4_reproduction.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_qv_map(out):
    """Fig. 5: Q-V map with lines of constant Purcell factor F = 3/(4 pi^2) Q/V."""
    cases = {"L3, final": "l3/l3_final_nb.npz", "nanobeam, McCutcheon (s = 105 nm)": "nanobeam/mcc_edge_s105.npz",
             "nanobeam, own N = 15": "nanobeam/own_N15.npz", "nanobeam, own N = 15, at 637 nm": "nanobeam/own_N15s.npz"}
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    V = np.logspace(np.log10(0.3), np.log10(3), 50)
    for e in (2, 3, 4, 5):
        ax.loglog(V, 10 ** e * 4 * np.pi ** 2 / 3 * V, color=style.COLOR_REFERENCE, lw=0.7, ls=":")
        ax.text(2.9, 10 ** e * 4 * np.pi ** 2 / 3 * 2.9 * 1.3, f"F = 10$^{e}$", fontsize=8,
                color=style.COLOR_REFERENCE, ha="right", va="bottom")
    for (name, f), c, m in zip(cases.items(), style.COLOR_CYCLE[::2] + [style.COLOR_PURPLE], "osdv"):
        o = load(f)
        ax.loglog(o["V"], o["Q"], m, ms=9, color=c, mec="k", ls="none", label=name)
    # Measured SiN nanobeam (Khan et al. 2011): measured Q, and the V that paper gives for its
    # design (simulated). The measured SiN slab cavities state no V and are left out.
    ax.loglog(0.55, 5.5e4, "*", ms=13, mfc="none", mec=style.COLOR_ROSE, mew=1.5, ls="none",
              label="measured SiN nanobeam (Khan 2011; V from design)")
    ax.set_xlim(0.3, 3); ax.set_ylim(1e2, 1e8)
    ax.set_xticks([0.3, 0.5, 1, 2, 3], ["0.3", "0.5", "1", "2", "3"]); ax.minorticks_off()
    ax.set_xlabel("mode volume V ((λ/n)³)"); ax.set_ylabel("quality factor Q")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)
    fig.savefig(out / "fig5_qv_map.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    fig_geometry(out)
    fig_slab_gap(out)
    fig_fields(out)
    fig_reproduction(out)
    fig_qv_map(out)
    print("done", sorted(p.name for p in out.glob("*.png")))
