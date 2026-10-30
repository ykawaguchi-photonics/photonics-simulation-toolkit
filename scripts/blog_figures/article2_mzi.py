"""Figures and numbers for the blog article "Mach-Zehnder Interferometers in
Silicon Photonics: Phase, Interference, and Optical Switching".

Reads only saved artifacts (si/data/design_points/{mzi,coupler,mzm}.yaml and
si/data/sparams/{mzi,coupler,mzm}/...) -- no FDTD is run and meep is never imported.
New computations are cheap: a circuit model built from the saved coupler
S-parameters, the 1D slab mode solver of article1_coupler.py for the arm's
n_eff(lambda), and fits to the saved spectra.

Usage (from the repo root):
    conda run -n mp python scripts/blog_figures/article2_mzi.py <out_dir>
"""

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from scipy.optimize import least_squares
from scipy.signal import argrelmax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pic_toolkit.slab_modes import slab_modes  # noqa: E402  (1D slab solver, meep-free)
from pic_toolkit import style  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SP = REPO / "si" / "data" / "sparams"
WL0 = 1.35

style.apply_style()
assert "meep" not in sys.modules


def load(path):
    d = np.load(path)
    wl = d["wavelengths_um"]
    o = np.argsort(wl)
    return {k: d[k][o] for k in d.files if d[k].shape == wl.shape}


def ideal_mzi(dphi, K):
    """Bar/cross power of an MZI made of two identical lossless couplers with
    cross-power fraction K: C = [[t, -jk], [-jk, t]], arm phase difference dphi."""
    t2, k2 = 1 - K, K
    bar = t2**2 + k2**2 - 2 * t2 * k2 * np.cos(dphi)
    cross = 2 * t2 * k2 * (1 + np.cos(dphi))
    return bar, cross


def raised_cosine_amplitude(dL, span):
    """Bump amplitude whose raised-cosine centerline is dL longer than span (bisection)."""
    x = np.linspace(0, span, 4001)

    def length(a):
        y = a * 0.5 * (1 - np.cos(2 * np.pi * x / span))
        return np.sum(np.hypot(np.diff(x), np.diff(y)))

    lo, hi = 0.0, 200.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if length(mid) < span + dL else (lo, mid)
    return hi


def refined_peaks(wl, T, edge_um=0.005):
    """Fringe maxima of T(wl): prominent (> 0.5 above the spectrum's minimum),
    clear of the band edges, refined by a parabola through the three samples."""
    out = []
    for i in argrelmax(T, order=3)[0]:
        prominent = T[i] - T.min() > 0.5
        inside = wl[0] + edge_um < wl[i] < wl[-1] - edge_um
        if 0 < i < len(T) - 1 and prominent and inside:
            y0, y1, y2 = T[i - 1], T[i], T[i + 1]
            d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            out.append(wl[i] + d * (wl[i + 1] - wl[i - 1]) / 2)
    return out


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numbers = {}
    dp = yaml.safe_load((REPO / "si" / "data" / "design_points" / "mzi.yaml").read_text())["parameters"]

    # ---- Fig. 1: layout of the saved MZI design point (outline from parameters)
    Lc, gap, w, sb, wide = (dp[k] for k in ("coupling_length_um", "gap_um", "wg_width_um", "sbend_len_um",
                                             "wide_sep_um"))
    span, dL = dp["delay_region_len_um"], dp["delta_L_um"]
    amp = raised_cosine_amplitude(dL, span)
    ch, wh = (gap + w) / 2, wide / 2
    cpl = Lc + 2 * sb
    x1 = np.linspace(0, cpl, 400)
    t = np.clip((np.abs(x1 - cpl / 2) - Lc / 2) / sb, 0, 1)
    y_c = ch + (wh - ch) * 0.5 * (1 - np.cos(np.pi * t))
    xa = np.linspace(cpl, cpl + span, 800)
    y_bump = -wh - amp * 0.5 * (1 - np.cos(2 * np.pi * (xa - cpl) / span))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for x0 in (0, cpl + span):
        for sgn in (1, -1):
            ax.fill_between(x0 + x1, sgn * y_c - w / 2, sgn * y_c + w / 2, color=style.COLOR_STEEL, lw=0)
    ax.fill_between(xa, wh - w / 2, wh + w / 2, color=style.COLOR_STEEL, lw=0)
    ax.fill_between(xa, y_bump - w / 2, y_bump + w / 2, color=style.COLOR_STEEL, lw=0)
    x_end = 2 * cpl + span
    # labels kept outside the waveguides
    ax.text(cpl / 2, wh + 1.4, "coupler 1\n(50:50 splitter)", ha="center", va="bottom", fontsize=8)
    ax.text(cpl + span + cpl / 2, wh + 1.4, "coupler 2\n(combiner)", ha="center", va="bottom", fontsize=8)
    ax.text(cpl + span / 2, wh + 1.0, "reference arm", ha="center", va="bottom", fontsize=8)
    ax.text(cpl + span / 2, -wh - amp - 1.2, f"delay arm: +ΔL = {dL:.1f} µm of path", ha="center", va="top",
            fontsize=8)
    for yy, lab in ((wh, "1"), (-wh, "2")):
        ax.text(-1.5, yy, f"in {lab}", ha="right", va="center", fontsize=8)
    ax.text(x_end + 1.5, wh, "out 3 (bar)", ha="left", va="center", fontsize=8)
    ax.text(x_end + 1.5, -wh, "out 4 (cross)", ha="left", va="center", fontsize=8)
    ax.set_xlim(-9, x_end + 14)
    ax.set_ylim(-wh - amp - 4.5, wh + 5.5)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / "fig1_layout.png", dpi=200)
    plt.close(fig)
    numbers["layout"] = dict(bump_amplitude_um=amp, delta_L_um=dL, coupler_Lc_um=Lc)

    # ---- Fig. 2: ideal transfer functions vs arm phase difference, for K = 0.5 and K = 0.4
    dphi = np.linspace(0, 4 * np.pi, 600)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, K in zip(axes, (0.5, 0.4)):
        bar, cross = ideal_mzi(dphi, K)
        ax.plot(dphi / np.pi, bar, color=style.COLOR_INDIGO, label="bar (out 3)")
        ax.plot(dphi / np.pi, cross, color=style.COLOR_TEAL, label="cross (out 4)")
        ax.set_xlabel(r"arm phase difference $\Delta\phi/\pi$")
        ax.set_title(f"couplers with cross power K = {K}", fontsize=10)
        ax.set_xlim(0, 4)
        ax.set_xticks([0, 1, 2, 3, 4])
    axes[0].set_ylabel("output power fraction")
    # the floor is labelled through the shared legend, so no text sits on the curves
    axes[1].axhline((1 - 2 * 0.4) ** 2, color=style.COLOR_REFERENCE, ls=":", lw=1,
                    label=r"bar-port floor $(1-2K)^2$ = 0.04 (K = 0.4)")
    handles, labels = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(handles + h1[2:], labels + l1[2:], loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.5)
    fig.savefig(out / "fig2_transfer.png", dpi=200)
    plt.close(fig)

    # ---- data: saved MZI (dL = 12.5 um) and coupler (Lc = 13.71 um)
    m = load(SP / "mzi" / "selected" / "mzi_selected.npz")
    c = load(SP / "coupler" / "selected" / "coupler_selected.npz")
    wl = m["wavelengths_um"]
    assert np.allclose(wl, c["wavelengths_um"])
    T3, T4, R1 = abs(m["S31"]) ** 2, abs(m["S41"]) ** 2, abs(m["S11"]) ** 2
    K = abs(c["S_cross"]) ** 2
    Tt = abs(c["S_through"]) ** 2
    minima = {}
    for name, T in (("bar_S31", T3), ("cross_S41", T4)):
        idx = [i for i in argrelmax(-T, order=3)[0] if T[i] < 0.1]
        minima[name] = [dict(wl_um=float(wl[i]), T=float(T[i]), K=float(K[i]),
                             bar_floor_pred=float((Tt[i] - K[i]) ** 2)) for i in idx]
    numbers["selected"] = dict(ER_bar_dB=float(10 * np.log10(T3.max() / T3.min())),
                               ER_cross_dB=float(10 * np.log10(T4.max() / T4.min())),
                               sum_min=float((T3 + T4 + R1).min()), sum_max=float((T3 + T4 + R1).max()),
                               minima=minima)

    # circuit model: measured coupler S-matrix, arm phase from slab n_eff(lambda)
    neff = np.array([slab_modes(None, wl=x, n_modes=1)[0][0] for x in wl])
    i0 = int(np.argmin(np.abs(wl - WL0)))
    ng = neff - wl * np.gradient(neff, wl)

    def circuit(phi0, dL_eff=dL):
        ph = 2 * np.pi * neff * dL_eff / wl + phi0
        St, Sx = c["S_through"], c["S_cross"]
        a_top, a_bot = St, Sx * np.exp(-1j * ph)
        return abs(St * a_top + Sx * a_bot) ** 2, abs(Sx * a_top + St * a_bot) ** 2

    def resid(q):
        a, b = circuit(q[0])
        return np.concatenate([a - T3, b - T4])

    fit = min((least_squares(resid, [p0]) for p0 in np.linspace(-np.pi, np.pi, 13)), key=lambda r: r.cost)
    Tm3, Tm4 = circuit(fit.x[0])
    numbers["circuit_model"] = dict(phi0_rad=float(fit.x[0]), rms=float(np.sqrt(np.mean(resid(fit.x) ** 2))),
                                    n_eff_arm=float(neff[i0]), n_g_arm=float(ng[i0]))

    # ---- Fig. 3: FDTD spectra (dB) with the circuit model overlaid
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(wl * 1000, 10 * np.log10(T3), color=style.COLOR_INDIGO, label="FDTD, bar (out 3)")
    ax.plot(wl * 1000, 10 * np.log10(T4), color=style.COLOR_TEAL, label="FDTD, cross (out 4)")
    ax.plot(wl * 1000, 10 * np.log10(Tm3), color=style.COLOR_REFERENCE, ls="--", lw=1,
            label="circuit model (measured couplers + arm phase)")
    ax.plot(wl * 1000, 10 * np.log10(Tm4), color=style.COLOR_REFERENCE, ls="--", lw=1)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("transmission (dB)")
    ax.set_ylim(-45, 3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig3_spectrum.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 4: cross-port spectra for the dL sweep + group index from fringe spacing
    sweep = []
    for p in glob.glob(str(SP / "mzi" / "sweep" / "mzi_deltaL*.npz")):
        val = float(re.search(r"deltaL([\d.]+)\.npz", p).group(1))
        if val in (2.5, 5.0, 7.5, 10.0, 12.5, 15.0):  # the six points of notebook 07, Section 10
            sweep.append((val, load(p)))
    sweep.sort(key=lambda s: s[0])
    ng_pairs = []
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for k, (val, d) in enumerate(sweep):
        ax.plot(d["wavelengths_um"] * 1000, abs(d["S41"]) ** 2 + 1.1 * k, color=style.COLOR_CYCLE[k % 6],
                label=f"ΔL = {val:g} µm")
        if val < 10.0:  # FSR (~lambda^2/(n_g dL)) exceeds the 100 nm band: no full fringe to measure
            continue
        for port in ("S31", "S41"):
            pk = refined_peaks(d["wavelengths_um"], abs(d[port]) ** 2)
            for a, b in zip(pk[:-1], pk[1:]):
                ng_pairs.append(dict(delta_L_um=val, port=port, wl1=a, wl2=b, fsr_nm=(b - a) * 1000,
                                     n_g=a * b / ((b - a) * val)))
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("cross-port power (offset by 1.1 per curve)")
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig4_dL_sweep.png", dpi=200)
    plt.close(fig)
    ngs = np.array([p["n_g"] for p in ng_pairs])
    numbers["group_index"] = dict(pairs=ng_pairs, mean=float(ngs.mean()), std=float(ngs.std()),
                                  slab_n_g=float(ng[i0]),
                                  notebook_fit_if_lambda_fixed_1p35=[float(WL0**2 / (p["fsr_nm"] / 1000 * p["delta_L_um"]))
                                                                      for p in ng_pairs])

    # ---- Fig. 5: switching with a push-pull index change (MZM, notebook 08)
    rows = []
    for p in sorted(glob.glob(str(SP / "mzm" / "sweep" / "mzm_dn*.npz"))):
        dn = float(re.search(r"dn(\d)p(\d+)", p).group(0)[2:].replace("p", "."))
        d = load(p)
        i = int(np.argmin(np.abs(d["wavelengths_um"] - WL0)))
        S31, S41 = d["S31"][i], d["S41"][i]
        rows.append((dn, abs(S31) ** 2, abs(S41) ** 2, S31 / S41))
    dn = np.array([r[0] for r in rows])
    Tb, Tx = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    ratio = np.array([r[3] for r in rows])
    # for identical 50:50 couplers S31/S41 = tan(dphi/2) (real): its sign fixes the branch
    ref = np.angle(ratio[1])
    sign = np.where(np.cos(np.angle(ratio) - ref) > 0, 1.0, -1.0)
    dphi_m = np.mod(2 * np.arctan(sign * np.abs(ratio)), 2 * np.pi)
    use = dn > 0  # at dn = 0 the bar amplitude is ~0 and its phase is undefined
    slope, icpt = np.polyfit(dn[use], dphi_m[use], 1)
    mz = yaml.safe_load((REPO / "si" / "data" / "design_points" / "mzm.yaml").read_text())
    Lp = mz["plateau_length_um"]
    gamma_meas = slope / (2 * np.pi / WL0 * Lp)
    n0 = slab_modes(None, n_modes=1)[0][0]
    n1 = slab_modes(None, n_core=2.7 + 1e-3, n_modes=1)[0][0]
    numbers["mzm"] = dict(dn=dn.tolist(), T_bar=Tb.tolist(), T_cross=Tx.tolist(),
                          dphi_over_pi=(dphi_m / np.pi).tolist(), slope_rad_per_dn=float(slope),
                          intercept_rad=float(icpt), dphi_at_vpi_over_pi=float((slope * 0.002 + icpt) / np.pi),
                          gamma_eff_measured=float(gamma_meas), gamma_slab_perturbation=float((n1 - n0) / 1e-3),
                          plateau_length_um=Lp, notebook_dphi_at_vpi_over_pi=mz["dphi_meep_at_vpi_pi"],
                          notebook_gamma=mz["phase_factor_measured"])
    dnf = np.linspace(0, 0.0022, 200)
    ph = slope * dnf + icpt
    loss = (Tb + Tx).mean()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(dnf * 1e3, loss * np.sin(ph / 2) ** 2, color=style.COLOR_REFERENCE, lw=1,
            label=r"$\sin^2(\Delta\phi/2)$, $\cos^2(\Delta\phi/2)$ with linear $\Delta\phi(\Delta n)$")
    ax.plot(dnf * 1e3, loss * np.cos(ph / 2) ** 2, color=style.COLOR_REFERENCE, lw=1)
    ax.plot(dn * 1e3, Tb, "s", color=style.COLOR_INDIGO, label="FDTD, bar (out 3)")
    ax.plot(dn * 1e3, Tx, "o", color=style.COLOR_TEAL, label="FDTD, cross (out 4)")
    ax.set_xlabel(r"push-pull index change $\Delta n$ ($10^{-3}$)")
    ax.set_ylabel(r"output power at $\lambda$ = 1.35 µm")
    ax.set_ylim(-0.03, 1.05)
    top = ax.secondary_xaxis("top", functions=(lambda x: (slope * x * 1e-3 + icpt) / np.pi,
                                               lambda p: (p * np.pi - icpt) / slope * 1e3))
    top.set_xlabel(r"arm phase difference $\Delta\phi/\pi$")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig5_switching.png", dpi=200)
    plt.close(fig)

    print(json.dumps(numbers, indent=1, default=float))
    return numbers


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "blog_figures_out")
