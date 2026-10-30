"""Figures and numbers for the blog article "WDM Demultiplexers".

Circuit models of cascaded-MZI lattice filters and a 4-channel binary-tree
demultiplexer, composed with 2x2 transfer matrices (forward direction only,
reflections neglected). Two kinds of building blocks:
- ideal: lossless couplers with a constant power coupling K and the
  maximally flat (halfband) synthesis of circuits/mzi_lattice.py;
- realistic: the FDTD-measured complex S-parameters of the directional
  coupler (si/data/sparams/coupler/selected and /sweep, si/notebooks/06), whose
  split ratio and phase vary with wavelength.
In both, the delay arms carry the phase 2 pi n_eff(wl) dL / wl of the 2D
model's mode (pic_toolkit.slab_modes), and each stage gets one constant phase
offset that places its channels on the grid (in a chip: a heater setting).
The realistic stage is checked against the full-device FDTD MZI of
si/notebooks/07_mzi.ipynb (si/data/sparams/mzi/sweep/mzi_deltaL15.0).
meep is never imported.

Conventions (as in the toolkit): exp(-i w t); a coupler's cross port carries
sign * i * sqrt(K); an arm of length L multiplies by exp(+i 2 pi n_eff L / wl).

Usage (from the repo root):
    conda run -n mp python scripts/blog_figures/wdm_demultiplexers.py <out_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pic_toolkit import style
from pic_toolkit.circuits import mzi_lattice as ml
from pic_toolkit.models import coupler as mc
from pic_toolkit.models import mzi as mm
from pic_toolkit.slab_modes import slab_modes

style.apply_style()
assert "meep" not in sys.modules

WL = np.linspace(1.30, 1.40, 2001)
WL0 = 1.35
SPACING = 0.020                                   # 20 nm channel grid (CWDM-like)
CH = np.array([1.32, 1.34, 1.36, 1.38])           # inside the simulated 1.30-1.40 um band
WIN = 0.003                                       # +-3 nm passband window for the metrics


# ---------------------------------------------------------------- 2D-model mode index
def neff_model():
    wls = np.linspace(1.29, 1.41, 25)
    n = np.array([slab_modes(None, wl=x, n_modes=1)[0][0] for x in wls])
    c = np.polyfit(wls, n, 3)
    ng = np.polyval(c, WL0) - WL0 * np.polyval(np.polyder(c), WL0)
    return (lambda wl: np.polyval(c, wl)), float(np.polyval(c, WL0)), float(ng)


NEFF, NEFF0, NG0 = neff_model()
DL1 = WL0**2 / (NG0 * 2 * SPACING)                 # stage 1: FSR = 2 x spacing
DL2 = DL1 / 2                                      # stage 2: FSR = 4 x spacing


# ---------------------------------------------------------------- building blocks
def ideal_coupler(K, sign=1):
    t, c = np.sqrt(1 - K), sign * 1j * np.sqrt(K)
    M = np.array([[t, c], [c, t]], dtype=complex)
    return np.broadcast_to(M, (len(WL), 2, 2))


def real_coupler(Lc=None):
    """FDTD coupler as a 2x2 forward matrix [[top<-top, top<-bot], [bot<-top, bot<-bot]]."""
    s = mc.coupler(WL) if Lc is None else mc.coupler_at_sweep_point(Lc, WL)
    return np.array([[s[("in_top", "out_top")], s[("in_bot", "out_top")]],
                     [s[("in_top", "out_bot")], s[("in_bot", "out_bot")]]]).transpose(2, 0, 1)


def delay(dL, extra=0.0):
    """Arm pair: top = reference, bottom = longer by dL (um) plus a constant phase `extra`."""
    ph = 2 * np.pi * NEFF(WL) * dL / WL + extra
    M = np.zeros((len(WL), 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = np.exp(1j * ph)
    return M


def stage(couplers, delays):
    """couplers: list of (n_wl,2,2); delays: list of (dL, extra) between them."""
    M = couplers[0]
    for C, (dL, ex) in zip(couplers[1:], delays):
        M = C @ delay(dL, ex) @ M
    return M  # column 0 = input at the top port


# Maximally flat (halfband) synthesis for N = 4 couplers (circuits/mzi_lattice.py)
K4, S4 = ml.synthesize_maximally_flat_kappas(4)


def lattice(n_couplers, dL, offset, real=False, sign_fix=True):
    """One lattice stage. N = 2: a plain MZI. N = 4: synthesized; its second coupler
    has K = 0 and is omitted, so the two delays around it merge: couplers K1, K3, K4
    with delays 2 dL and dL.
    The synthesis needs sign -1 on the last coupler. A real coupler always has the
    + sign; flipping it is the same as adding pi to the arm before it (sign_fix)."""
    if n_couplers == 2:
        C = [real_coupler(), real_coupler()] if real else [ideal_coupler(0.5), ideal_coupler(0.5)]
        return stage(C, [(dL, offset)])
    Ks = [K4[0], K4[2], K4[3]]
    signs = [S4[0], S4[2], S4[3]]
    assert abs(K4[1]) < 1e-6   # synthesis returns ~4e-7, i.e. zero
    if real:
        C = [real_coupler(24.0), real_coupler(20.0), real_coupler(14.0)]
        extra_last = np.pi if (sign_fix and signs[2] < 0) else 0.0
        return stage(C, [(2 * dL, 2 * offset), (dL, offset + extra_last)])
    C = [ideal_coupler(k, s) for k, s in zip(Ks, signs)]
    return stage(C, [(2 * dL, 2 * offset), (dL, offset)])


def best_offset(fn, targets):
    """Constant phase offset that maximizes the mean power into the target
    (output index, wavelength) pairs: the heater setting of the stage."""
    th = np.linspace(0, 2 * np.pi, 721)[:-1]
    idx = [(o, int(np.argmin(np.abs(WL - w)))) for o, w in targets]
    score = []
    for t in th:
        M = fn(t)
        score.append(np.mean([abs(M[i, o, 0]) ** 2 for o, i in idx]))
    return th[int(np.argmax(score))]


def tree(n_couplers, real=False):
    """4-channel binary tree. Stage 1 (dL1) sends ch1, ch3 to its top output and
    ch2, ch4 to its bottom output; stage 2 (dL2) splits each pair. Channel powers
    multiply along each path (no interference between stages)."""
    f1 = lambda t: lattice(n_couplers, DL1, t, real)
    f2 = lambda t: lattice(n_couplers, DL2, t, real)
    th1 = best_offset(f1, [(0, CH[0]), (0, CH[2]), (1, CH[1]), (1, CH[3])])
    th_up = best_offset(f2, [(0, CH[0]), (1, CH[2])])
    th_dn = best_offset(f2, [(0, CH[1]), (1, CH[3])])
    M1, Mu, Md = f1(th1), f2(th_up), f2(th_dn)
    p1t, p1b = abs(M1[:, 0, 0]) ** 2, abs(M1[:, 1, 0]) ** 2
    chans = np.array([p1t * abs(Mu[:, 0, 0]) ** 2, p1b * abs(Md[:, 0, 0]) ** 2,
                      p1t * abs(Mu[:, 1, 0]) ** 2, p1b * abs(Md[:, 1, 0]) ** 2])  # ch1..ch4
    return chans, {"theta1": th1, "theta_up": th_up, "theta_dn": th_dn,
                   "up_minus_dn_deg": float(np.degrees((th_up - th_dn + np.pi) % (2 * np.pi) - np.pi))}


def metrics(chans, detune=0.0):
    """Per channel, within +-WIN of (center + detune): worst insertion loss (dB)
    and worst crosstalk = strongest other channel relative to this one (dB)."""
    out = []
    for k, c in enumerate(CH):
        m = np.abs(WL - (c + detune)) <= WIN
        own = chans[k][m]
        other = np.max(np.delete(chans, k, axis=0)[:, m], axis=0)
        out.append((-10 * np.log10(own.min()), 10 * np.log10(np.max(other / own))))
    return np.array(out)


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-7))


# ---------------------------------------------------------------- figures
def fig_schematic(out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 2.96), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    # one N = 4 lattice stage (three couplers, delays dL and 2 dL)
    for x0, lab in ((0.0, "K₁"), (4.0, "K₃"), (7.0, "K₄")):
        a1.add_patch(Rectangle((x0, -0.55), 1.0, 1.1, fc="0.85", ec="none"))
        a1.text(x0 + 0.5, -0.85, lab, ha="center", va="top", fontsize=9)
    for y in (0.35, -0.35):
        a1.plot([-0.8, 0.0], [y, y], color=style.COLOR_STEEL, lw=1.6)
        a1.plot([8.0, 8.8], [y, y], color=style.COLOR_STEEL, lw=1.6)
    for xa, xb, amp in ((1.0, 4.0, 1.5), (5.0, 7.0, 0.9)):
        xs = np.linspace(xa, xb, 80)
        a1.plot(xs, 0.35 + 0 * xs, color=style.COLOR_STEEL, lw=1.6)
        a1.plot(xs, -0.35 - amp * np.sin(np.pi * (xs - xa) / (xb - xa)) ** 2, color=style.COLOR_STEEL, lw=1.6)
    for x0, x1 in ((0.0, 1.0), (4.0, 5.0), (7.0, 8.0)):
        a1.plot([x0, x1], [0.35, 0.35], color=style.COLOR_STEEL, lw=1.6)
        a1.plot([x0, x1], [-0.35, -0.35], color=style.COLOR_STEEL, lw=1.6)
    a1.text(2.5, -2.15, "+2ΔL", ha="center", va="top", fontsize=9)
    a1.text(6.0, -1.55, "+ΔL", ha="center", va="top", fontsize=9)
    a1.text(-0.9, 0.35, "in", ha="right", va="center", fontsize=9)
    a1.text(8.9, 0.35, "bar", ha="left", va="center", fontsize=9)
    a1.text(8.9, -0.35, "cross", ha="left", va="center", fontsize=9)
    a1.set_xlim(-1.8, 10.2)
    a1.set_ylim(-2.6, 1.2)
    a1.set_aspect("equal")
    a1.axis("off")
    a1.set_title("one flat-top lattice stage (N = 4 design)", fontsize=9)

    # binary tree
    def box(ax, x, y, text, w=2.6, h=0.9):
        ax.add_patch(Rectangle((x, y - h / 2), w, h, fc="0.9", ec=style.COLOR_STEEL, lw=1.0))
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=9)

    box(a2, 0.6, 0.0, "stage 1\nFSR = 2Δλ")
    box(a2, 5.0, 1.4, "stage 2 (upper)\nFSR = 4Δλ", w=3.0, h=1.25)
    box(a2, 5.0, -1.4, "stage 2 (lower)\nFSR = 4Δλ,\nshifted by Δλ", w=3.0, h=1.25)
    a2.plot([-0.4, 0.6], [0, 0], color=style.COLOR_STEEL, lw=1.6)
    a2.text(-0.5, 0, "λ₁ λ₂ λ₃ λ₄", ha="right", va="center", fontsize=9)
    for y in (1.4, -1.4):
        a2.plot([3.2, 4.1, 4.1, 5.0], [0.2 * np.sign(y), 0.2 * np.sign(y), y, y], color=style.COLOR_STEEL, lw=1.6)
    a2.text(4.2, 0.75, "λ₁ λ₃", ha="left", va="center", fontsize=9)
    a2.text(4.2, -0.75, "λ₂ λ₄", ha="left", va="center", fontsize=9)
    for y, labs in ((1.4, ("λ₁", "λ₃")), (-1.4, ("λ₂", "λ₄"))):
        for dy, lab in ((0.2, labs[0]), (-0.2, labs[1])):
            a2.plot([8.0, 8.8], [y + dy, y + dy], color=style.COLOR_STEEL, lw=1.6)
            a2.text(8.9, y + dy, lab, ha="left", va="center", fontsize=9)
    a2.set_xlim(-3.0, 9.8)
    a2.set_ylim(-2.3, 2.3)
    a2.set_aspect("equal")
    a2.axis("off")
    a2.set_title("4-channel binary tree", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig1_layout.png", dpi=200)
    plt.close(fig)


def fig_stage(out, numbers):
    """Ideal single stage, N = 2 vs N = 4, versus the stage phase over one period."""
    phi = np.linspace(0, 2 * np.pi, 2001)

    def resp(n):
        Ms = []
        for p in phi:
            if n == 2:
                C = np.array([[np.sqrt(.5), 1j * np.sqrt(.5)], [1j * np.sqrt(.5), np.sqrt(.5)]])
                A = np.diag([1, np.exp(1j * p)])
                Ms.append(abs((C @ A @ C)[0, 0]) ** 2)
            else:
                Cs = [np.array([[np.sqrt(1 - k), s * 1j * np.sqrt(k)], [s * 1j * np.sqrt(k), np.sqrt(1 - k)]])
                      for k, s in zip(K4, S4)]
                A = np.diag([1, np.exp(1j * p)])
                M = Cs[0]
                for C in Cs[1:]:
                    M = C @ A @ M
                Ms.append(abs(M[0, 0]) ** 2)
        return np.array(Ms)

    r2, r4 = resp(2), resp(4)
    # the K = 0 coupler can be dropped and the sign moved into the arm: same response
    Cs = [np.array([[np.sqrt(1 - k), 1j * np.sqrt(k)], [1j * np.sqrt(k), np.sqrt(1 - k)]]) for k in (K4[0], K4[2], K4[3])]
    alt = []
    for p in phi:
        M = Cs[1] @ np.diag([1, np.exp(2j * p)]) @ Cs[0]
        M = Cs[2] @ np.diag([1, np.exp(1j * (p + np.pi))]) @ M
        alt.append(abs(M[0, 0]) ** 2)
    assert np.max(np.abs(np.array(alt) - r4)) < 1e-3, "sign flip != pi in the preceding arm"

    def width(r, level_db):
        ok = db(r) >= -level_db
        return np.mean(ok)  # fraction of the period inside the band (one passband per period)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.57))
    x = phi / np.pi
    for r, lab, col, ls in ((r2, "N = 2 (one MZI)", style.COLOR_REFERENCE, "--"),
                            (r4, "N = 4 (maximally flat)", style.COLOR_INDIGO, "-")):
        a1.plot(x, r, color=col, ls=ls, lw=1.6, label=lab)
        a2.plot(x, db(r), color=col, ls=ls, lw=1.6)
    a1.set_ylabel("bar-port power")
    a2.set_ylabel("bar-port power (dB)")
    a2.set_ylim(-40, 2)
    for a in (a1, a2):
        a.set_xlabel("arm phase φ (units of π)")
        a.set_xlim(0, 2)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig2_stage.png", dpi=200)
    plt.close(fig)
    numbers["stage_ideal"] = {"K4": [float(k) for k in K4], "S4": [int(s) for s in S4],
                              "w1dB_frac_N2": width(r2, 1), "w1dB_frac_N4": width(r4, 1),
                              "w3dB_frac_N2": width(r2, 3), "w3dB_frac_N4": width(r4, 3),
                              "w20dB_stop_frac_N2": float(np.mean(db(r2) <= -20)),
                              "w20dB_stop_frac_N4": float(np.mean(db(r4) <= -20))}


def fig_tree(out, numbers):
    res = {}
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.6), sharex=True, sharey=True)
    cols = [style.COLOR_INDIGO, style.COLOR_TEAL, style.COLOR_OLIVE, style.COLOR_PURPLE]
    for j, n in enumerate((2, 4)):
        for i, real in enumerate((False, True)):
            chans, info = tree(n, real)
            key = f"N{n}_{'real' if real else 'ideal'}"
            met = metrics(chans)
            res[key] = (chans, info, met)
            ax = axes[i, j]
            for k in range(4):
                ax.plot(WL * 1e3, db(chans[k]), color=cols[k], lw=1.3, label=f"ch{k + 1}")
                ax.axvspan((CH[k] - WIN) * 1e3, (CH[k] + WIN) * 1e3, color="0.92", zorder=0, lw=0)
            ax.set_ylim(-45, 2)
            ax.set_title(f"N = {n}, {'FDTD couplers' if real else 'ideal couplers'}", fontsize=10)
            if i == 1:
                ax.set_xlabel("wavelength (nm)")
            if j == 0:
                ax.set_ylabel("transmission (dB)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "fig4_tree.png", dpi=200)
    plt.close(fig)
    numbers["tree"] = {k: {"IL_dB": v[2][:, 0].tolist(), "XT_dB": v[2][:, 1].tolist(),
                           "offset_up_minus_dn_deg": v[1]["up_minus_dn_deg"]} for k, v in res.items()}
    return res


def fig_detune(out, numbers, res):
    d = np.linspace(0, 0.008, 33)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.57))
    numbers["detune"] = {}
    for key, col, ls, lab in (("N2_ideal", style.COLOR_STEEL, "--", "N = 2, ideal"),
                              ("N4_ideal", style.COLOR_INDIGO, "--", "N = 4, ideal"),
                              ("N2_real", style.COLOR_STEEL, "-", "N = 2, FDTD couplers"),
                              ("N4_real", style.COLOR_INDIGO, "-", "N = 4, FDTD couplers")):
        chans = res[key][0]
        # worst channel at offsets +dd and -dd from every channel centre (both directions,
        # like the +-3 nm window of the table)
        il, xt = [], []
        for dd in d:
            ils, xts = [], []
            for sgn in (1, -1):
                idx = [int(np.argmin(np.abs(WL - (c + sgn * dd)))) for c in CH]
                own = np.array([chans[k][i] for k, i in enumerate(idx)])
                other = np.array([np.max(np.delete(chans[:, i], k)) for k, i in enumerate(idx)])
                ils.append(np.max(-10 * np.log10(own)))
                xts.append(np.max(10 * np.log10(other / own)))
            il.append(max(ils))
            xt.append(max(xts))
        a1.plot(d * 1e3, il, color=col, ls=ls, lw=1.6, label=lab)
        a2.plot(d * 1e3, xt, color=col, ls=ls, lw=1.6)
        numbers["detune"][key] = {"d_nm": (d * 1e3).tolist(), "worst_IL_dB": il, "worst_XT_dB": xt}
    a1.set_ylabel("worst-channel insertion loss (dB)")
    a2.set_ylabel("worst-channel crosstalk (dB)")
    a2.set_ylim(-60, 3)
    for a in (a1, a2):
        a.set_xlabel("laser offset ±d from channel centre (nm)")
        a.set_xlim(0, 8)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out / "fig5_detuning.png", dpi=200)
    plt.close(fig)


def fig_validation(out, numbers):
    """Realistic N = 2 stage (FDTD coupler + mode-solver arm phase + one fitted
    constant offset) versus the full-device FDTD MZI with dL = 15 um."""
    s = mm.mzi_at_sweep_point(15.0, WL)
    bar_f, cross_f = abs(s[("in_top", "out_top")]) ** 2, abs(s[("in_top", "out_bot")]) ** 2
    th = np.linspace(0, 2 * np.pi, 1441)[:-1]
    errs = []
    for t in th:
        M = lattice(2, 15.0, t, real=True)
        errs.append(np.sqrt(np.mean((abs(M[:, 0, 0]) ** 2 - bar_f) ** 2 + (abs(M[:, 1, 0]) ** 2 - cross_f) ** 2) / 2))
    t0 = th[int(np.argmin(errs))]
    M = lattice(2, 15.0, t0, real=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(WL * 1e3, db(bar_f), color=style.COLOR_INDIGO, lw=1.6, label="full-device FDTD, bar")
    ax.plot(WL * 1e3, db(cross_f), color=style.COLOR_TEAL, lw=1.6, label="full-device FDTD, cross")
    ax.plot(WL * 1e3, db(abs(M[:, 0, 0]) ** 2), color=style.COLOR_REFERENCE, lw=1.0, ls="--", label="circuit model")
    ax.plot(WL * 1e3, db(abs(M[:, 1, 0]) ** 2), color=style.COLOR_REFERENCE, lw=1.0, ls="--")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("transmission (dB)")
    ax.set_ylim(-45, 2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig3_validation.png", dpi=200)
    plt.close(fig)
    numbers["validation"] = {"rms_power": float(np.min(errs)), "offset_rad": float(t0)}


def coupler_numbers(numbers):
    out = {}
    for name, Lc in (("50:50 design", None), ("Lc 24", 24.0), ("Lc 20", 20.0), ("Lc 14", 14.0)):
        C = real_coupler(Lc)
        K = abs(C[:, 1, 0]) ** 2
        out[name] = {"K_1300": float(K[0]), "K_1350": float(np.interp(1.35, WL, K)), "K_1400": float(K[-1])}
    numbers["couplers"] = out


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numbers = {"neff0": NEFF0, "ng0": NG0, "dL1_um": DL1, "dL2_um": DL2,
               "fsr1_nm": WL0**2 / (NG0 * DL1) * 1e3, "quarter_wave_nm": WL0 / (4 * NEFF0) * 1e3,
               "half_wave_nm": WL0 / (2 * NEFF0) * 1e3}
    coupler_numbers(numbers)
    fig_schematic(out)
    fig_stage(out, numbers)
    res = fig_tree(out, numbers)
    fig_detune(out, numbers, res)
    fig_validation(out, numbers)
    # the same tree with the N = 4 sign left uncorrected (all couplers with their natural sign)
    nofix = lambda t, dL: lattice(4, dL, t, real=True, sign_fix=False)
    chans = None
    f1 = lambda t: nofix(t, DL1)
    f2 = lambda t: nofix(t, DL2)
    th1 = best_offset(f1, [(0, CH[0]), (0, CH[2]), (1, CH[1]), (1, CH[3])])
    tu = best_offset(f2, [(0, CH[0]), (1, CH[2])])
    td = best_offset(f2, [(0, CH[1]), (1, CH[3])])
    M1, Mu, Md = f1(th1), f2(tu), f2(td)
    p1t, p1b = abs(M1[:, 0, 0]) ** 2, abs(M1[:, 1, 0]) ** 2
    chans = np.array([p1t * abs(Mu[:, 0, 0]) ** 2, p1b * abs(Md[:, 0, 0]) ** 2,
                      p1t * abs(Mu[:, 1, 0]) ** 2, p1b * abs(Md[:, 1, 0]) ** 2])
    m = metrics(chans)
    numbers["tree"]["N4_real_no_sign_fix"] = {"IL_dB": m[:, 0].tolist(), "XT_dB": m[:, 1].tolist()}
    # diagnostic: the same N = 4 tree with each FDTD coupler frozen at its 1.35 um split
    # ratio (no wavelength dependence) -- separates coupler dispersion from K mismatch
    K1350 = [float(np.interp(1.35, WL, abs(real_coupler(L)[:, 1, 0]) ** 2)) for L in (24.0, 20.0, 14.0)]

    def frozen(t, dL):
        C = [ideal_coupler(K1350[0]), ideal_coupler(K1350[1]), ideal_coupler(K1350[2], -1)]
        return stage(C, [(2 * dL, 2 * t), (dL, t)])

    f1 = lambda t: frozen(t, DL1)
    f2 = lambda t: frozen(t, DL2)
    th1 = best_offset(f1, [(0, CH[0]), (0, CH[2]), (1, CH[1]), (1, CH[3])])
    tu = best_offset(f2, [(0, CH[0]), (1, CH[2])])
    td = best_offset(f2, [(0, CH[1]), (1, CH[3])])
    M1, Mu, Md = f1(th1), f2(tu), f2(td)
    p1t, p1b = abs(M1[:, 0, 0]) ** 2, abs(M1[:, 1, 0]) ** 2
    chans = np.array([p1t * abs(Mu[:, 0, 0]) ** 2, p1b * abs(Md[:, 0, 0]) ** 2,
                      p1t * abs(Mu[:, 1, 0]) ** 2, p1b * abs(Md[:, 1, 0]) ** 2])
    m = metrics(chans)
    numbers["tree"]["N4_K1350_frozen"] = {"K": K1350, "IL_dB": m[:, 0].tolist(), "XT_dB": m[:, 1].tolist()}
    txt = json.dumps(numbers, indent=1, default=float)
    print(txt)
    (out / "numbers.json").write_text(txt)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "blog_figures_out/wdm_demultiplexers")
