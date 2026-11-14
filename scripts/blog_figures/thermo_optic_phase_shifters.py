"""Figures and numbers for the blog article "Thermo-Optic Phase Shifters".

Reads only si_active/data/blog_figures/thermo_optic_phase_shifters.npz (written by
thermo_optic_phase_shifters_data.py, which re-runs the solves of
si_active/notebooks/01_thermo_optic_phase_shifter.ipynb). meep is never imported.

Usage (from the repo root):
    conda run -n mp python scripts/blog_figures/thermo_optic_phase_shifters.py <out_dir>
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from pic_toolkit import style

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "si_active" / "data" / "blog_figures" / "thermo_optic_phase_shifters.npz"
METAL = "#C9A227"          # TiN heater in schematics only
OXIDE = "#DCE9F5"
SUBSTRATE = "#9AA5B1"
SI = style.COLOR_STEEL
NPP = style.COLOR_PURPLE
D_SIO2 = 1.38 / (2203 * 709)   # thermal diffusivity of SiO2 (Jacques 2019 Table 3), m^2/s
L_DEMO_UM = 100.0              # heater length used to turn W/m into a temperature

style.apply_style()
assert "meep" not in sys.modules

VARIANTS = [("tin", "(a) TiN heater"), ("tin_trench", "(b) TiN + deep trenches"), ("n_doped", "(c) N++ slab heater")]
SERIES = {"tin": style.COLOR_CYCLE[0], "tin_trench": style.COLOR_CYCLE[2], "n_doped": style.COLOR_PURPLE}
LABEL = {"tin": "TiN", "tin_trench": "TiN + trenches", "n_doped": "N++ slab"}


def draw_structure(ax, d, key):
    s_top, box = float(d["stack_clad_top_um"]), float(d["stack_box_um"])
    ax.add_patch(Rectangle((-20, -box - 6), 40, 6, fc=SUBSTRATE, ec="none"))
    ax.add_patch(Rectangle((-20, -box), 40, box + s_top, fc=OXIDE, ec="none"))
    for x0, x1, y0, y1, code in d[f"{key}_rects"]:
        if code in (1, 2):
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=SI, ec="none"))
    for x0, x1, y0, y1, code in d[f"{key}_rects"]:
        if code == 3:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=METAL if key != "n_doped" else NPP, ec="none"))
        if code == 4:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="white", ec=style.COLOR_REFERENCE, lw=0.6))


def fig_structures(d, out):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (key, title) in zip(axes, VARIANTS):
        draw_structure(ax, d, key)
        ax.set_xlim(-6, 6); ax.set_ylim(-3.5, 3.2); ax.set_aspect("equal")
        ax.set_title(title, fontsize=10); ax.set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")
    handles = [Patch(fc=SI, label="Si (core, slab)"), Patch(fc=METAL, label="TiN heater"), Patch(fc=NPP, label="N++ Si heater"),
               Patch(fc=OXIDE, label="SiO$_2$ (BOX, cladding)"), Patch(fc=SUBSTRATE, label="Si substrate (heat sink)"),
               Patch(fc="white", ec=style.COLOR_REFERENCE, label="air trench / air above")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.subplots_adjust(bottom=0.33, wspace=0.08)
    fig.savefig(out / "fig1_cross_sections.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_temperature(d, out):
    fields = {}
    for key, _ in VARIANTS:
        p_line = float(d[f"{key}_p_pi_mW"]) * 1e-3 / (L_DEMO_UM * 1e-6)    # W/m at P = P_pi for 100 um
        fields[key] = d[f"{key}_T"] * p_line
    vmax = max(f.max() for f in fields.values())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0), sharey=True)
    for ax, (key, title) in zip(axes, VARIANTS):
        im = ax.pcolormesh(d[f"{key}_x"], d[f"{key}_y"], fields[key], cmap="inferno", shading="gouraud", vmin=0, vmax=vmax)
        for x0, x1, y0, y1, code in d[f"{key}_rects"]:
            if code in (1, 2, 3, 4):
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=0.6))
        m = "rib" if key == "n_doped" else "strip"
        ax.contour(d[f"{m}_x"], d[f"{m}_y"], d[f"{m}_E2"], levels=[np.exp(-2)], colors=[style.COLOR_CYAN], linewidths=0.9)
        ax.axhline(-float(d["stack_box_um"]), color="white", lw=0.5, ls=":")
        ax.set_xlim(-6, 6); ax.set_ylim(-3, float(d["stack_clad_top_um"])); ax.set_aspect("equal")
        ax.set_title(f"{title}\nP = P$_\\pi$ = {float(d[key + '_p_pi_mW']):.1f} mW", fontsize=10)
        ax.set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")
    fig.subplots_adjust(wspace=0.08, right=0.88)
    cax = fig.add_axes([0.90, 0.2, 0.012, 0.6])
    fig.colorbar(im, cax=cax, label="temperature rise (K)")
    fig.savefig(out / "fig2_temperature.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_dynamics(d, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for key, _ in VARIANTS:
        c, tau = SERIES[key], float(d[f"{key}_tau_us"])
        a1.semilogx(d[f"{key}_t"] * 1e6, d[f"{key}_step"], color=c, label=f"{LABEL[key]}: τ = {tau:.1f} µs")
        a1.plot([tau], [1 - np.exp(-1)], "o", color=c, ms=5)
        H = np.abs(d[f"{key}_H"])
        a2.semilogx(d[f"{key}_f"], 20 * np.log10(H), color=c,
                    label=f"{LABEL[key]}: f$_{{3dB}}$ = {float(d[key + '_f3db_kHz']):.0f} kHz")
    a1.axhline(1 - np.exp(-1), color=style.COLOR_REFERENCE, lw=0.8, ls="--")
    a1.set_xlim(1e-2, 1e3); a1.set_ylim(0, 1.05)
    a1.set_xlabel("time after power step (µs)"); a1.set_ylabel("phase / final phase"); a1.set_title("(a) step response", fontsize=10)
    a2.axhline(-3, color=style.COLOR_REFERENCE, lw=0.8, ls="--")
    a2.set_xlim(1e2, 1e7); a2.set_ylim(-40, 2)
    a2.set_xlabel("modulation frequency of the heater power (Hz)"); a2.set_ylabel("phase response (dB)")
    a2.set_title("(b) small-signal frequency response", fontsize=10)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=9)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=9)
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(out / "fig3_dynamics.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_trench(d, out):
    dd, pp, tt = d["trench_d_um"], d["trench_p_pi_mW"], d["trench_tau_us"]
    p0, t0 = float(d["tin_p_pi_mW"]), float(d["tin_tau_us"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    a1.plot(dd, pp, "o-", color=style.COLOR_STEEL, label="with trenches")
    a1.axhline(p0, color=style.COLOR_REFERENCE, ls="--", lw=1, label="no trench")
    a1.set_xlabel("trench distance d (µm)"); a1.set_ylabel("P$_\\pi$ (mW)")
    a1.set_title("(a) power for a π phase shift", fontsize=10); a1.set_ylim(0, 19)
    a2.plot(dd, tt, "s-", color=style.COLOR_STEEL, label="with trenches")
    a2.axhline(t0, color=style.COLOR_REFERENCE, ls="--", lw=1, label="no trench")
    a2.set_xlabel("trench distance d (µm)"); a2.set_ylabel("τ (µs)")
    a2.set_title("(b) thermal time constant", fontsize=10); a2.set_ylim(0, 12)
    for a in (a1, a2):
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(out / "fig4_trench_tradeoff.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_npp(d, out):
    b, p, loss = d["buffer_um"], d["buffer_p_pi_mW"], d["buffer_loss_db_per_cm"]
    fig, a1 = plt.subplots(figsize=(6.2, 3.6))
    l1, = a1.plot(b, p, "o-", color=style.COLOR_STEEL, label="P$_\\pi$ (left axis)")
    a1.set_xlabel("distance of the N++ strips from the rib edge, b (µm)"); a1.set_ylabel("P$_\\pi$ (mW)")
    a1.set_ylim(0, 25)
    a2 = a1.twinx()
    l2, = a2.semilogy(b, loss, "s--", color=style.COLOR_PURPLE, label="heater-induced loss, estimate (right axis)")
    a2.set_ylabel("optical loss (dB/cm)")
    a1.legend(handles=[l1, l2], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False, fontsize=9)
    fig.savefig(out / "fig5_npp_tradeoff.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def slab_length_um(box_um, clad_top_um):
    return 2 * (box_um + clad_top_um) / np.pi


def fig_crosstalk(d, out):
    ell = slab_length_um(float(d["stack_box_um"]), float(d["stack_clad_top_um"]))
    x, T = d["profile_x_um"], d["profile_T"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.semilogy(x, T / T[0], color=style.COLOR_STEEL, label="2D heat solve (no trench)")
    sel = x >= 4
    a1.semilogy(x[sel], T[sel][0] / T[0] * np.exp(-(x[sel] - x[sel][0]) / ell), "--", color=style.COLOR_REFERENCE,
                label=f"slab model, exp(−x/ℓ), ℓ = 2H/π = {ell:.2f} µm")
    a1.set_xlim(0, 30); a1.set_ylim(1e-4, 1.5)
    a1.set_xlabel("lateral distance from the heater, x (µm)"); a1.set_ylabel("ΔT(x) / ΔT(0) at the Si layer")
    a1.set_title("(a) lateral temperature profile", fontsize=10)
    a2.semilogy(d["xt_pitch_um"], d["xt_none"], "o-", color=style.COLOR_STEEL, label="no trench")
    a2.semilogy(d["xt_trench_pitch_um"], d["xt_trench"], "s-", color=style.COLOR_CYCLE[2], label="trenches at d = 3 µm")
    a2.set_xlim(0, 21); a2.set_ylim(1e-4, 1.5)
    a2.set_xlabel("waveguide pitch, p (µm)"); a2.set_ylabel("neighbour phase / own phase")
    a2.set_title("(b) thermal crosstalk", fontsize=10)
    for a in (a1, a2):
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=9)
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(out / "fig6_crosstalk.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_map(d, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    tt = np.linspace(2, 13, 100)
    for fom in (50, 100, 150):
        ax.plot(tt, fom / tt, color=style.COLOR_REFERENCE, lw=0.7, ls=":")
        y_lab = min(28.3, fom / 2.05)                       # where the curve leaves the top / left edge
        ax.text(fom / y_lab + 0.12, y_lab, f"{fom} mW·µs", fontsize=8, color=style.COLOR_REFERENCE, ha="left", va="center")
    ax.plot(d["trench_tau_us"], d["trench_p_pi_mW"], "o-", color=style.COLOR_CYCLE[2], label="TiN + trenches, d = 2–8 µm")
    ax.plot(d["box_tau_us"], d["box_p_pi_mW"], "^-", color=style.COLOR_CYCLE[0], label="TiN, BOX = 1, 2, 3 µm")
    ax.plot(d["buffer_tau_us"], d["buffer_p_pi_mW"], "s-", color=style.COLOR_PURPLE, label="N++, b = 0.4–1.2 µm")
    ax.plot([float(d["tin_tau_us"])], [float(d["tin_p_pi_mW"])], "o", ms=9, mfc="none", mec=style.COLOR_ROSE, mew=1.5,
            label="TiN, standard stack (BOX 2 µm)")
    ax.set_xlim(2, 13); ax.set_ylim(0, 30)
    ax.set_xlabel("thermal time constant τ (µs)"); ax.set_ylabel("P$_\\pi$ (mW)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=8.5)
    fig.savefig(out / "fig7_tradeoff_map.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def numbers(d):
    print("== variants ==")
    for key, _ in VARIANTS:
        tau = float(d[f"{key}_tau_us"])
        print(f"{key:10s} P_pi {float(d[key + '_p_pi_mW']):6.2f} mW (point {float(d[key + '_p_pi_point_mW']):.2f}), "
              f"tau {tau:5.2f} us, f3dB {float(d[key + '_f3db_kHz']):5.1f} kHz, 1/(2 pi tau) {1e3 / (2 * np.pi * tau):5.1f} kHz, "
              f"f3dB*tau {float(d[key + '_f3db_kHz']) * 1e3 * tau * 1e-6:.3f}, P_pi*tau {float(d[key + '_p_pi_mW']) * tau:.0f}")
    dneff_dT = None
    print("== slab model ==")
    for b, p, t, x5, x10 in zip(d["box_um"], d["box_p_pi_mW"], d["box_tau_us"], d["box_xt5"], d["box_xt10"]):
        ell = slab_length_um(b, float(d["stack_clad_top_um"]))
        ell_fit = 5 / np.log(x5 / x10)
        tau_slab = (ell * 1e-6) ** 2 / D_SIO2 * 1e6
        print(f"BOX {b:.0f}: P_pi {p:5.2f} tau {t:5.2f} xt5 {100 * x5:5.2f}% xt10 {100 * x10:5.2f}%  "
              f"ell model {ell:.2f} fit {ell_fit:.2f} um, ell^2/D {tau_slab:.1f} us")
    xp, xt = d["xt_pitch_um"], d["xt_none"]
    for i in range(len(xp) - 1):
        print(f"  pitch {xp[i]:4.1f}->{xp[i + 1]:4.1f}: ell = {(xp[i + 1] - xp[i]) / np.log(xt[i] / xt[i + 1]):.2f} um")
    print("trench:", [f"d {a}: {p:.2f} mW {t:.2f} us {p * t:.0f}" for a, p, t in zip(d["trench_d_um"], d["trench_p_pi_mW"], d["trench_tau_us"])])
    print("buffer:", [f"b {a}: {p:.2f} mW {t:.2f} us {l:.3g} dB/cm" for a, p, t, l in
                      zip(d["buffer_um"], d["buffer_p_pi_mW"], d["buffer_tau_us"], d["buffer_loss_db_per_cm"])])
    print("xt none :", dict(zip(d["xt_pitch_um"], np.round(100 * d["xt_none"], 2))))
    print("xt trench:", dict(zip(d["xt_trench_pitch_um"], np.round(100 * d["xt_trench"], 2))))
    dn = float(d["tin_dneff_per_W_per_m"])
    print(f"dneff per W/m {dn:.3e}; strip n_eff {float(d['strip_n_eff']):.4f} n_g {float(d['strip_n_g']):.4f}")
    for key, _ in VARIANTS:
        p_line = float(d[f"{key}_p_pi_mW"]) * 1e-3 / (L_DEMO_UM * 1e-6)
        print(f"{key}: peak dT at P_pi (100 um) = {d[key + '_T'].max() * p_line:.1f} K")


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    d = dict(np.load(DATA))
    numbers(d)
    fig_structures(d, out); fig_temperature(d, out); fig_dynamics(d, out)
    fig_trench(d, out); fig_npp(d, out); fig_crosstalk(d, out); fig_map(d, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
