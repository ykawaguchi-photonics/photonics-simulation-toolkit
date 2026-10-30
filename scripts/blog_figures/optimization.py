"""Figures and numbers for the blog article "Topology Optimization and the
Adjoint Method".

Fig. 1 is a schematic of the optimization loop. Fig. 2 illustrates the
density filter and tanh projection on a 1D toy density (closed form, no
simulation). Fig. 3 re-plots the saved adjoint topology optimization of the
90-degree bend (si/data/sparams/bend_topopt*, notebooks 04/04b): objective
history and the final single-objective and robust designs. Fig. 4 compares
the optimized bend with an Euler bend at matched footprints and shows the
fabrication-bias test. Fig. 5 re-plots the Adam histories of the two-parameter
SiN grating optimization (meta/data/step2). No FDTD is run and meep is never
imported.

Usage (from the repo root):
    conda run -n mp python scripts/blog_figures/optimization.py <out_dir>
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from pic_toolkit import style

REPO = Path(__file__).resolve().parents[2]
SP = REPO / "si" / "data" / "sparams"
DP = REPO / "si" / "data" / "design_points"
STEP2 = REPO / "meta" / "data" / "step2"

# Optimized-bend transmissions at matched footprints, resolution 40, binarized
# designs: measured in si/notebooks/04_bend_topology_optimization.ipynb, Section 3
# (the Euler-bend column is also saved in conventional_reference.json).
TOPOPT_FOOTPRINT = {2.0: 0.6870, 2.5: 0.8163, 3.0: 0.8678, 3.5: 0.8897, 4.05: 0.9412, 4.5: 0.9638}
EULER_3P5 = 0.8699  # 3.5 um row of the same table (not in conventional_reference.json)

style.apply_style()
assert "meep" not in sys.modules


def box(ax, x, y, w, h, text, color, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="none", alpha=0.92))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="white")


def arrow(ax, p0, p1, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, color="k", lw=0.9, **kw))


def conic_filter_1d(rho, r):
    k = np.maximum(0.0, 1.0 - np.abs(np.arange(-r, r + 1)) / (r + 1))
    k /= k.sum()
    pad = np.pad(rho, r, mode="edge")
    return np.convolve(pad, k, mode="valid")


def tanh_projection(x, beta, eta):
    return (np.tanh(beta * eta) + np.tanh(beta * (x - eta))) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numbers = {}

    # ---- Fig. 1: the optimization loop (schematic)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    I, T, O = style.COLOR_INDIGO, style.COLOR_TEAL, style.COLOR_OLIVE
    box(ax, 0.1, 2.0, 1.7, 0.9, "design variables\n(pixels $\\rho$ or\nparameters $p$)", I)
    box(ax, 2.2, 2.0, 1.7, 0.9, "filter + projection\n$\\rho \\to \\tilde\\rho \\to \\bar\\rho \\to \\varepsilon$", I)
    box(ax, 4.3, 2.0, 1.7, 0.9, "forward simulation\nobjective $F$", T)
    box(ax, 6.4, 2.0, 1.7, 0.9, "adjoint simulation\n$\\partial F/\\partial\\varepsilon$", T)
    box(ax, 8.2, 0.7, 1.7, 0.9, "chain rule\n$dF/d\\rho$", T)
    box(ax, 4.3, 0.7, 2.9, 0.9, "gradient-based update\n(MMA, Adam, L-BFGS)", O)
    for x0, x1 in ((1.8, 2.2), (3.9, 4.3), (6.0, 6.4)):
        arrow(ax, (x0, 2.45), (x1, 2.45))
    arrow(ax, (8.1, 2.2), (8.8, 1.6))
    arrow(ax, (8.2, 1.15), (7.2, 1.15))
    arrow(ax, (4.3, 1.15), (0.95, 1.15), connectionstyle="arc3,rad=0")
    arrow(ax, (0.95, 1.15), (0.95, 2.0))
    ax.text(2.0, 3.25, "problem formulation\n(topology optimization)", ha="center", va="center", fontsize=9, color=I)
    ax.text(6.2, 3.25, "gradient computation\n(adjoint method: 2 simulations)", ha="center", va="center", fontsize=9, color=T)
    ax.text(5.75, 0.25, "update algorithm (gradient-based optimizer)", ha="center", va="center", fontsize=9, color=O)
    fig.tight_layout()
    fig.savefig(out / "fig1_loop.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 2: filter and projection on a 1D toy density
    rng = np.random.default_rng(3)
    x = np.arange(120)
    rho = np.clip(0.5 + 0.5 * np.sin(x / 9.0) + 0.35 * rng.standard_normal(x.size), 0, 1)
    rt = conic_filter_1d(rho, 6)
    keep = x <= 104                      # drop the padded right edge
    x, rho, rt = x[keep], rho[keep], rt[keep]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    a1.plot(x, rho, color=style.COLOR_REFERENCE, lw=0.8, label=r"raw density $\rho$")
    a1.plot(x, rt, color=style.COLOR_INDIGO, lw=1.6, label=r"filtered $\tilde\rho$")
    a1.set_xlabel("position (pixels)")
    a1.set_ylabel("density")
    a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    for beta, c in ((4, style.COLOR_CYAN), (32, style.COLOR_SEAGREEN)):
        a2.plot(x, tanh_projection(rt, beta, 0.5), color=c, lw=1.6, label=rf"projected, $\beta={beta}$")
    a2.plot(x, tanh_projection(rt, 32, 0.556), color=style.COLOR_OLIVE, lw=1.0, ls="--",
            label=r"$\beta=32$, $\eta=0.556$ (eroded)")
    a2.set_xlabel("position (pixels)")
    a2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    for a in (a1, a2):
        a.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out / "fig2_filter_projection.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 3: bend topology optimization: history and final designs
    single = np.load(SP / "bend_topopt_robust" / "final" / "baseline_single_objective.npz")
    robust = np.load(SP / "bend_topopt_robust" / "final" / "robust_3way.npz")
    base = yaml.safe_load(open(DP / "bend_topopt.yaml"))
    betas = base["parameters"]["beta_schedule"]
    fig = plt.figure(figsize=(10.5, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1])
    a1 = fig.add_subplot(gs[0])
    it = np.arange(1, 81)
    a1.plot(it, single["evaluation_history"], color=style.COLOR_SEAGREEN, lw=1.6, label="single objective")
    a1.plot(it, robust["evaluation_history_avg"], color=style.COLOR_INDIGO, lw=1.2, ls="--",
            label="robust (3-way average)")
    for k in (20, 40, 60):
        a1.axvline(k + 0.5, color=style.COLOR_REFERENCE, lw=0.6, ls=":")
    for k, b in enumerate(betas):
        a1.text(10.5 + 20 * k, 1.02, rf"$\beta$={b:g}", ha="center", va="bottom", fontsize=7.5,
                transform=a1.get_xaxis_transform())
    a1.set_xlabel("iteration")
    a1.set_ylabel(r"objective $|S_{21}|^2$ (training run)")
    a1.set_ylim(0, 1)
    a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    ext = [0, 2.5, 0, 2.5]
    for k, (w, title) in enumerate(((single["final_weights_binarized"], "single objective"),
                                    (robust["final_weights_binarized_nominal"], "robust"))):
        a = fig.add_subplot(gs[k + 1])
        a.imshow(w.T, origin="lower", extent=ext, cmap=style.CMAP_PERMITTIVITY_OPTIMIZED, vmin=0, vmax=1.3)
        a.set_title(f"final design: {title}", fontsize=9)
        a.set_xlabel("x (µm)")
        a.set_ylabel("y (µm)")
        a.grid(False)
    fig.tight_layout()
    fig.savefig(out / "fig3_bend_history.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 4: matched-footprint comparison and fabrication-bias test
    conv = json.load(open(SP / "bend_topopt_robust" / "conventional_reference" / "conventional_reference.json"))
    euler = {round(c["region_um"], 2): c["T21"] for c in conv}
    euler[3.5] = EULER_3P5
    regions = sorted(TOPOPT_FOOTPRINT)
    rb = yaml.safe_load(open(DP / "bend_topopt_robust.yaml"))["robustness"]["fabrication_transmission_comparison"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9))
    a1.plot(regions, [euler[r] for r in regions], "o-", color=style.COLOR_STEEL, ms=4, label="Euler bend")
    a1.plot(regions, [TOPOPT_FOOTPRINT[r] for r in regions], "s-", color=style.COLOR_SEAGREEN, ms=4,
            label="topology-optimized")
    a1.set_xlabel("design-region size (µm)")
    a1.set_ylabel(r"transmission $|S_{21}|^2$ at 1.35 µm")
    a1.set_ylim(0.4, 1.0)
    a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    labels = ["eroded\n(−10 nm)", "nominal", "dilated\n(+10 nm)"]
    s = rb["notebook_04_single_objective"]
    r = rb["this_robust_design"]
    vs = [s["transmission_minus_bias_eroded"], s["transmission_nominal"], s["transmission_plus_bias_dilated"]]
    vr = [r["transmission_minus_bias_eroded"], r["transmission_nominal"], r["transmission_plus_bias_dilated"]]
    xx = np.arange(3)
    a2.bar(xx - 0.18, vs, 0.34, color=style.COLOR_SEAGREEN, label="single objective")
    a2.bar(xx + 0.18, vr, 0.34, color=style.COLOR_INDIGO, label="robust")
    a2.set_xticks(xx)
    a2.set_xticklabels(labels, fontsize=8)
    a2.set_ylabel(r"transmission $|S_{21}|^2$")
    a2.set_ylim(0.7, 0.83)
    a2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig4_footprint_robustness.png", dpi=200)
    plt.close(fig)
    numbers["bend"] = dict(euler=euler, topopt=TOPOPT_FOOTPRINT, robustness=rb,
                           final_objective=base["optimization"] if "optimization" in base else None)

    # ---- Fig. 5: Adam on the two-parameter SiN grating
    summ = json.load(open(STEP2 / "summary.json"))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    runs = (("adam_duty_0.5", "start: 50 % duty, 200 nm etch"),
            ("adam_narrow_tooth", "start: narrow tooth, deep etch"),
            ("adam_wide_tooth", "start: wide tooth, shallow etch"))
    hist = {}
    for (name, lab), c in zip(runs, (style.COLOR_INDIGO, style.COLOR_TEAL, style.COLOR_OLIVE)):
        h = pickle.load(open(STEP2 / f"{name}.pkl", "rb"))["history"]
        it = [e["iter"] for e in h]
        eta = [e["eta"] for e in h]
        ax.plot(it, eta, "o-", ms=2.5, lw=1.1, color=c, label=lab)
        hist[name] = dict(eta0=eta[0], eta_end=eta[-1], theta_end=list(map(float, h[-1]["theta"])))
    ax.axhline(summ["landscape_max"], color=style.COLOR_REFERENCE, lw=1, ls="--", label="best point of the mapped (w, e) grid")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel(r"coupling efficiency $\eta$ (res 60)")
    ax.set_ylim(0, 0.42)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig5_adam_grating.png", dpi=200)
    plt.close(fig)
    numbers["adam"] = dict(runs=hist, landscape_max=summ["landscape_max"], res160_grid=summ["res160"]["grid"]["eta_db"])
    print(json.dumps(numbers, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "blog_optimization")
