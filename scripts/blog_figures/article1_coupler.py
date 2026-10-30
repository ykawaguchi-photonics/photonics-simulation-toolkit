"""Figures and numbers for the blog article "Directional Couplers in Silicon
Photonics: From Coupled-Mode Theory to FDTD Simulation".

Reads only already-saved artifacts (si/data/design_points/coupler.yaml and
si/data/sparams/coupler/{sweep,sweep_res40,sweep_2d_res40,selected}) -- no FDTD is
run and meep is never imported. The only new computation is a 1D
finite-difference solve of the slab (supermode) cross-section, which takes
seconds.

Usage (from the repo root):
    conda run -n mp python scripts/blog_figures/article1_coupler.py <out_dir>
"""

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from scipy.optimize import curve_fit

from pic_toolkit import style
from pic_toolkit.slab_modes import slab_modes as _slab_modes

REPO = Path(__file__).resolve().parents[2]
SPARAMS = REPO / "si" / "data" / "sparams" / "coupler"
WL0 = 1.35  # design wavelength used by notebooks 06/06b (um)
N_CLAD = 1.44

style.apply_style()
assert "meep" not in sys.modules


def load_artifact(npz_path):
    """S-parameter arrays from a coupler .npz, sorted by wavelength."""
    d = np.load(npz_path)
    wl = d["wavelengths_um"]
    order = np.argsort(wl)
    return {k: d[k][order] for k in d.files if d[k].shape == wl.shape}


def at_wl(data, key, wl=WL0):
    return data[key][np.argmin(np.abs(data["wavelengths_um"] - wl))]


def cmt_power(L, kappa, L0, eta):
    """Cross-port power of a lossless, symmetric coupler (CMT), with an
    effective extra length L0 contributed by the S-bends and a near-unity
    transmission factor eta."""
    return eta * np.sin(kappa * (L + L0)) ** 2


def slab_modes(gap, wl=WL0, **kw):
    """1D slab (supermode) solver -- see pic_toolkit.slab_modes; defaults to this article's design wavelength."""
    return _slab_modes(gap, wl=wl, **kw)


def length_sweep():
    rows = []
    for p in glob.glob(str(SPARAMS / "sweep" / "*.npz")):
        L = float(re.search(r"length([\d.]+)\.npz", p).group(1))
        d = load_artifact(p)
        rows.append((L, abs(at_wl(d, "S_through")) ** 2, abs(at_wl(d, "S_cross")) ** 2,
                     abs(at_wl(d, "S_reflect")) ** 2))
    return np.array(sorted(rows)).T


def gap_grid():
    grid = {}
    for p in glob.glob(str(SPARAMS / "sweep_2d_res40" / "*.npz")) + glob.glob(str(SPARAMS / "sweep_res40" / "*.npz")):
        m = re.search(r"(?:gap([\d.]+)_)?couplinglength([\d.]+)\.npz", p)
        grid[(float(m.group(1) or 0.2), float(m.group(2)))] = load_artifact(p)
    return grid


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dp = yaml.safe_load((REPO / "si" / "data" / "design_points" / "coupler.yaml").read_text())["parameters"]
    numbers = {}

    # ---- Fig. 1: top view of the saved design point (outline only, from parameters)
    Lc, gap, w = dp["coupling_length_um"], dp["gap_um"], dp["wg_width_um"]
    sb, wide = dp["sbend_len_um"], dp["wide_sep_um"]
    x = np.linspace(-(Lc / 2 + sb + 2), Lc / 2 + sb + 2, 2000)
    ch, wh = (gap + w) / 2, wide / 2
    t = np.clip((np.abs(x) - Lc / 2) / sb, 0, 1)
    yc = ch + (wh - ch) * 0.5 * (1 - np.cos(np.pi * t))
    fig, ax = plt.subplots(figsize=(8, 2.6))
    for sgn in (1, -1):
        ax.fill_between(x, sgn * yc - w / 2, sgn * yc + w / 2, color=style.COLOR_STEEL, lw=0)
    ax.fill_between([-Lc / 2, Lc / 2], -ch - w / 2 - 0.15, ch + w / 2 + 0.15,
                    color=style.COLOR_REFERENCE, alpha=0.12, lw=0)
    ax.annotate("", xy=(-Lc / 2, -1.6), xytext=(Lc / 2, -1.6),
                arrowprops=dict(arrowstyle="<->", color="k", lw=0.8))
    ax.text(0, -1.75, f"coupling length $L_c$ = {Lc:.2f} µm", ha="center", va="top", fontsize=9)
    # Leader runs along y = 0, through the open space between the two arms, to a
    # label placed beyond the waveguide ends so no text overlaps the structure.
    ax.annotate(f"gap $g$ = {gap*1000:.0f} nm", xy=(Lc / 2 - 0.5, 0), xytext=(x[-1] + 0.6, 0),
                fontsize=8, va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color="k", lw=0.6))
    ax.text(x[0] + 0.2, wh + 0.45, "in 1", fontsize=8)
    ax.text(x[0] + 0.2, -wh - 0.75, "in 2", fontsize=8)
    ax.text(x[-1] - 0.2, wh + 0.45, "out 1 (through)", ha="right", fontsize=8)
    ax.text(x[-1] - 0.2, -wh - 0.75, "out 2 (cross)", ha="right", fontsize=8)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_ylim(-2.5, 1.9)
    ax.set_xlim(x[0] - 0.5, x[-1] + 6.0)
    ax.set_yticks([-2, -1, 0, 1])
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / "fig1_geometry.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 2: isolated mode and even/odd supermodes (gap 0.2 um)
    n1, y1, H1, _ = slab_modes(None, n_modes=1)
    (ne, no), y, H, e = slab_modes(0.2)
    gamma = 2 * np.pi / WL0 * np.sqrt(n1[0] ** 2 - N_CLAD**2)
    numbers.update(n_eff_single=float(n1[0]), gamma_per_um=float(gamma), n_even=float(ne), n_odd=float(no),
                   kappa_supermode_gap0p2=float(np.pi * (ne - no) / WL0))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for a in (a1, a2):
        for c in ((0.2 + 0.5) / 2, -(0.2 + 0.5) / 2):
            a.axvspan(c - 0.25, c + 0.25, color=style.COLOR_REFERENCE, alpha=0.15, lw=0)
        a.axhline(0, color=style.COLOR_REFERENCE, lw=0.6)
        a.set_xlim(-1.5, 1.5)
        a.set_xlabel("y (µm)")
    H1s = H1[:, 0] * np.sign(H1[np.argmin(np.abs(y1)), 0])
    a1.plot(y1 + 0.35, H1s, color=style.COLOR_STEEL, label="waveguide 1 alone")
    a1.plot(y1 - 0.35, H1s, color=style.COLOR_TEAL, ls="--", label="waveguide 2 alone")
    a1.set_ylabel("$H_z$ (normalized)")
    a1.set_title("Isolated modes: evanescent tails overlap", fontsize=10)
    # Legends sit below each panel so no text lands on the shaded cores.
    a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    he = H[:, 0] * np.sign(H[np.argmin(np.abs(y - 0.35)), 0])
    ho = H[:, 1] * np.sign(H[np.argmin(np.abs(y - 0.35)), 1])
    a2.plot(y, he, color=style.COLOR_INDIGO, label=f"even, $n_e$ = {ne:.4f}")
    a2.plot(y, ho, color=style.COLOR_PURPLE, label=f"odd, $n_o$ = {no:.4f}")
    a2.set_title("Coupled pair: even and odd supermodes", fontsize=10)
    a2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    a1.set_ylim(-1.15, 1.15)
    fig.tight_layout()
    fig.savefig(out / "fig2_supermodes.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 4: beat curve vs coupling length (gap 0.2 um) + CMT fit
    L, Tt, Tc, R = length_sweep()
    (kap, L0, eta), cov = curve_fit(cmt_power, L, Tc, p0=[0.05, 2.5, 1.0])
    err = np.sqrt(np.diag(cov))
    numbers.update(kappa_fit=float(kap), kappa_fit_err=float(err[0]), L0_fit=float(L0), L0_fit_err=float(err[1]),
                   eta_fit=float(eta), L_pi=float(np.pi / (2 * kap)), L50_fit=float(np.pi / (4 * kap) - L0),
                   fit_rms=float(np.sqrt(np.mean((cmt_power(L, kap, L0, eta) - Tc) ** 2))),
                   sum_TXR_min=float((Tt + Tc + R).min()), sum_TXR_max=float((Tt + Tc + R).max()),
                   n_length_points=int(len(L)))
    Lf = np.linspace(-L0, 34, 400)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(Lf, cmt_power(Lf, kap, L0, eta), color=style.COLOR_REFERENCE, lw=1.2,
            label=r"CMT fit  $\eta\,\sin^2[\kappa(L_c+L_0)]$")
    ax.plot(Lf, eta - cmt_power(Lf, kap, L0, eta), color=style.COLOR_REFERENCE, lw=1.2, ls="--",
            label=r"CMT fit  $\eta\,\cos^2[\kappa(L_c+L_0)]$")
    ax.plot(L, Tc, "o", color=style.COLOR_TEAL, label="FDTD, cross port")
    ax.plot(L, Tt, "s", color=style.COLOR_INDIGO, label="FDTD, through port")
    ax.plot([dp["coupling_length_um"]], [0.5], "D", color=style.COLOR_ROSE, ms=8, zorder=5,
            label=f"50:50 design, $L_c$ = {dp['coupling_length_um']:.2f} µm")
    ax.axvline(-L0, color=style.COLOR_ANNOTATION, lw=0.8, ls=":")
    ax.text(-L0 + 0.3, 0.30, f"$L_c=-L_0$\n({-L0:.2f} µm)", color=style.COLOR_ANNOTATION, fontsize=8)
    ax.set_xlabel("coupling length $L_c$ (µm)")
    ax.set_ylabel(r"power fraction at $\lambda$ = 1.35 µm")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "fig4_beat_curve.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 5: coupling coefficient vs gap: FDTD (CMT fit per gap) vs supermode theory
    grid = gap_grid()
    gaps = sorted({g for g, _ in grid})
    rows = []
    for g in gaps:
        Ls = np.array(sorted(l for gg, l in grid if gg == g))
        Ts = np.array([abs(at_wl(grid[(g, l)], "S_cross")) ** 2 for l in Ls])
        (kg, L0g, etag), _ = curve_fit(cmt_power, Ls, Ts, p0=[0.05 * np.exp(-18 * (g - 0.2)), 2.5, 1.0],
                                       bounds=([1e-4, -10, 0.8], [1, 20, 1.05]))
        (nge, ngo), *_ = slab_modes(g)
        rows.append(dict(gap_um=g, kappa_fdtd=float(kg), L0_um=float(L0g), eta=float(etag),
                         kappa_supermode=float(np.pi * (nge - ngo) / WL0), n_even=float(nge), n_odd=float(ngo)))
    gg = np.array([r["gap_um"] for r in rows])
    kf = np.array([r["kappa_fdtd"] for r in rows])
    ks = np.array([r["kappa_supermode"] for r in rows])
    slope, icpt = np.polyfit(gg, np.log(kf), 1)
    numbers.update(gap_table=rows, ln_kappa_slope_per_um=float(slope))
    fig, ax = plt.subplots(figsize=(6, 4.6))
    gl = np.linspace(0.13, 0.37, 50)
    ax.semilogy(gl, kf[1] * np.exp(-gamma * (gl - 0.2)), color=style.COLOR_REFERENCE, ls="--", lw=1,
                label=rf"$\propto e^{{-\gamma g}}$, $\gamma$ = {gamma:.2f} µm$^{{-1}}$ (isolated-mode tail)")
    ax.semilogy(gg, ks, "-", color=style.COLOR_REFERENCE, lw=1.2, label=r"supermodes  $\pi(n_e-n_o)/\lambda$")
    ax.semilogy(gg, kf, "o", color=style.COLOR_STEEL, label="FDTD (CMT fit per gap)")
    ax.set_xlabel("gap $g$ (µm)")
    ax.set_ylabel(r"coupling coefficient $\kappa$ (µm$^{-1}$)")
    ax2 = ax.secondary_yaxis("right", functions=(lambda k: np.pi / 2 / np.maximum(k, 1e-9),
                                                  lambda Lp: np.pi / 2 / np.maximum(Lp, 1e-9)))
    ax2.set_ylabel(r"$L_\pi = \pi/2\kappa$ (µm)")
    ax2.set_yticks([20, 30, 50, 100, 150])
    ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig5_kappa_vs_gap.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 6: spectrum of the saved 50:50 design
    s = load_artifact(SPARAMS / "selected" / "coupler_selected.npz")
    wl = s["wavelengths_um"]
    Tt_s, Tc_s, R_s = abs(s["S_through"]) ** 2, abs(s["S_cross"]) ** 2, abs(s["S_reflect"]) ** 2
    rel = np.degrees(np.angle(s["S_cross"] / s["S_through"]))
    numbers.update(selected_T_through_wl0=float(at_wl(s, "S_through").__abs__() ** 2),
                   selected_T_cross_wl0=float(abs(at_wl(s, "S_cross")) ** 2),
                   selected_Tcross_1p30=float(abs(at_wl(s, "S_cross", 1.30)) ** 2),
                   selected_Tcross_1p40=float(abs(at_wl(s, "S_cross", 1.40)) ** 2),
                   selected_rel_phase_deg_min=float(rel.min()), selected_rel_phase_deg_max=float(rel.max()),
                   selected_max_R=float(R_s.max()), selected_sum_min=float((Tt_s + Tc_s + R_s).min()))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(wl * 1000, Tc_s, color=style.COLOR_TEAL, label="cross port")
    ax.plot(wl * 1000, Tt_s, color=style.COLOR_INDIGO, label="through port")
    ax.plot(wl * 1000, Tt_s + Tc_s + R_s, color=style.COLOR_REFERENCE, ls="--", label="sum (incl. reflection)")
    ax.axvline(WL0 * 1000, color=style.COLOR_ANNOTATION, lw=0.8, ls=":")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("power fraction")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower center", ncol=3)
    fig.tight_layout()
    fig.savefig(out / "fig6_spectrum.png", dpi=200)
    plt.close(fig)

    print(json.dumps(numbers, indent=2))
    return numbers


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "blog_figures_out")
