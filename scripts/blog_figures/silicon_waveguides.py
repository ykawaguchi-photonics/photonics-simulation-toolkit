"""Figures and numbers for the blog article "Silicon Waveguides".

Reads only saved data:
- si/data/blog_figures/silicon_waveguides_mpb.npz (MPB cross-section solves of the
  real 500 x 220 nm strip, written by silicon_waveguides_data.py);
- si/data/sparams/waveguide/sweep/*.npz and si/data/design_points/waveguide.yaml
  (2D FDTD length sweep of si/notebooks/01_waveguide_baseline.ipynb);
- 3d/data/design_points/waveguide_3d.yaml (3D FDTD of 3d/notebooks/01b).
meep is never imported. New computations are cheap and analytic: the
symmetric-slab dispersion relation, the effective-index method, and the 1D
slab solver pic_toolkit.slab_modes for the toolkit's 2D model.

Usage (from the repo root):
    conda run -n mp python scripts/blog_figures/silicon_waveguides.py <out_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml
from scipy.optimize import brentq

from pic_toolkit import style
from pic_toolkit.slab_modes import slab_modes

REPO = Path(__file__).resolve().parents[2]
MPB = REPO / "si" / "data" / "blog_figures" / "silicon_waveguides_mpb.npz"
SWEEP = REPO / "si" / "data" / "sparams" / "waveguide" / "sweep"
WL0 = 1.35
N_SI, N_OX = 3.48, 1.44

style.apply_style()
assert "meep" not in sys.modules


# ---------------------------------------------------------------- analytic slab
def slab_neff(n1, n2, d, wl, m=0, pol="TE"):
    """Symmetric slab, mode order m: k_x d = 2 arctan(r gamma / k_x) + m pi,
    r = 1 (TE, E parallel to the interfaces) or (n1/n2)^2 (TM, E normal to them)."""
    k0 = 2 * np.pi / wl
    r = 1.0 if pol == "TE" else (n1 / n2) ** 2

    def f(n):
        kx = k0 * np.sqrt(n1**2 - n**2)
        g = k0 * np.sqrt(n**2 - n2**2)
        return kx * d - 2 * np.arctan(r * g / kx) - m * np.pi

    lo, hi = n2 + 1e-9, n1 - 1e-9
    if f(lo) * f(hi) > 0:
        return np.nan
    return brentq(f, lo, hi)


def cutoff_thickness(n1, n2, wl, m):
    """Symmetric slab: mode m appears at d = m wl / (2 sqrt(n1^2 - n2^2)) (TE and TM alike)."""
    return m * wl / (2 * np.sqrt(n1**2 - n2**2))


def eim_strip(w, h=0.22, wl=WL0):
    """Effective-index method for the TE-like strip mode: vertical TE slab, then a
    lateral slab whose field is normal to the sidewalls (TM-type boundary)."""
    ns = slab_neff(N_SI, N_OX, h, wl, 0, "TE")
    return slab_neff(ns, N_OX, w, wl, 0, "TM"), ns


# ---------------------------------------------------------------- data helpers
def classify(rows):
    """MPB width-sweep rows (w, family, band, n_eff, n_g, te_share) -> named modes.
    family 0 = MPB ODD_Y (TE0, TM1, TE2), family 1 = EVEN_Y (TM0, TE1)."""
    modes = {"TE0": [], "TE1": [], "TM0": [], "TM1": []}
    for w in np.unique(rows[:, 0]):
        r = rows[rows[:, 0] == w]
        f0 = r[r[:, 1] == 0]
        f1 = r[r[:, 1] == 1]
        te0 = f0[f0[:, 5] > 0.5]
        tm1 = f0[f0[:, 5] < 0.5]
        tm0 = f1[f1[:, 5] < 0.5]
        te1 = f1[f1[:, 5] > 0.5]
        for name, sel in (("TE0", te0[:1]), ("TM1", tm1[:1]), ("TM0", tm0[:1]), ("TE1", te1[:1])):
            if len(sel):
                modes[name].append((w, sel[0, 3], sel[0, 4]))
    return {k: np.array(v) for k, v in modes.items()}


def cutoff_width(curve, n_clad=N_OX):
    """Linear interpolation of where a mode's n_eff first exceeds the cladding index."""
    w, n = curve[:, 0], curve[:, 1]
    i = 0
    return w[i] - (n[i] - n_clad) / ((n[i + 1] - n[i]) / (w[i + 1] - w[i]))


def load_sweep():
    """The length sweep listed in the notebook's manifest (10.0-11.0 um, 0.1 um steps)."""
    import csv
    rows = []
    for rec in csv.DictReader(open(SWEEP / "manifest.csv")):
        L = float(rec["length_um"])
        d = np.load(SWEEP / f"waveguide_length{L:.1f}.npz")
        o = np.argsort(d["wavelengths_um"])
        rows.append((L, d["wavelengths_um"][o], d["s21"][o]))
    rows.sort(key=lambda r: r[0])
    L = np.array([r[0] for r in rows])
    wl = rows[0][1]
    s21 = np.array([r[2] for r in rows])  # (n_L, n_wl)
    return L, wl, s21


# ---------------------------------------------------------------- figures
def fig_cross_section(d, out):
    E_te, E_tm = d["E_te0"], d["E_tm0"]
    sv, sl = d["cell_um"]
    nv, nl = E_te.shape[:2]
    v = (np.arange(nv) + 0.5) * sv / nv - sv / 2
    lat = (np.arange(nl) + 0.5) * sl / nl - sl / 2

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.3), gridspec_kw=dict(width_ratios=[1.55, 1, 1]))
    ax = axes[0]
    fs = 10
    ax.add_patch(Rectangle((-1.6, -1.0), 3.2, 1.0, fc="0.88", ec="none"))          # BOX
    ax.add_patch(Rectangle((-1.6, -1.35), 3.2, 0.35, fc="0.65", ec="none"))       # Si handle
    ax.add_patch(Rectangle((-1.6, 0.0), 3.2, 0.45, fc="0.94", ec="none"))         # top cladding
    ax.add_patch(Rectangle((-0.25, 0.0), 0.5, 0.22, fc=style.COLOR_STEEL, ec="none"))
    # labels in one column to the right of the stack, each level with its layer
    ax.annotate("Si core\n500 × 220 nm", xy=(0.22, 0.2), xytext=(1.8, 0.95), fontsize=fs, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="k"))
    ax.text(1.8, 0.22, "SiO₂ cladding", ha="left", va="center", fontsize=fs)
    ax.text(1.8, -0.5, "buried oxide\n(BOX, 2–3 µm)", ha="left", va="center", fontsize=fs)
    ax.text(1.8, -1.18, "Si substrate", ha="left", va="center", fontsize=fs)
    ax.set_xlim(-1.7, 4.2)
    ax.set_ylim(-1.45, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("cross-section (schematic)", fontsize=10.5)

    for ax, E, title in ((axes[1], E_te, "TE-like mode: |E|²"), (axes[2], E_tm, "TM-like mode: |E|²")):
        I = np.sum(np.abs(E) ** 2, axis=2)
        I /= I.max()
        ax.pcolormesh(lat, v, I, cmap=style.CMAP_DENSITY, shading="auto", vmin=0, vmax=1)
        ax.add_patch(Rectangle((-0.25, -0.11), 0.5, 0.22, fill=False, ec="w", lw=0.8))
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect("equal")
        ax.set_xlabel("lateral position (µm)")
        ax.set_title(title, fontsize=10.5)
    axes[1].set_ylabel("vertical position (µm)")
    fig.tight_layout()
    fig.savefig(out / "fig1_cross_section.png", dpi=200)
    plt.close(fig)


def fig_slab(out, numbers):
    d = np.linspace(0.02, 0.6, 400)
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    styles = (("TE", 0, style.COLOR_INDIGO, "-"), ("TE", 1, style.COLOR_INDIGO, "--"),
              ("TM", 0, style.COLOR_TEAL, "-"), ("TM", 1, style.COLOR_TEAL, "--"))
    for pol, m, c, ls in styles:
        n = [slab_neff(N_SI, N_OX, x, WL0, m, pol) for x in d]
        ax.plot(d * 1e3, n, color=c, ls=ls, lw=1.6, label=f"{pol}{m}")
    ax.axhline(N_OX, color=style.COLOR_REFERENCE, lw=0.8, ls=":")
    ax.axvline(220, color=style.COLOR_ROSE, lw=0.9)
    dc = cutoff_thickness(N_SI, N_OX, WL0, 1)
    ax.set_xlabel("film thickness d (nm)")
    ax.set_ylabel("effective index $n_\\mathrm{eff}$")
    ax.set_xlim(0, 600)
    ax.set_ylim(1.3, 3.5)
    ax.text(580, N_OX - 0.04, "cladding index 1.44", ha="right", va="top", fontsize=8, color=style.COLOR_REFERENCE)
    ax.text(228, 3.42, "220 nm", ha="left", va="top", fontsize=8, color=style.COLOR_ROSE)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig2_slab_dispersion.png", dpi=200)
    plt.close(fig)
    numbers["slab"] = {
        "te0_220_135": slab_neff(N_SI, N_OX, 0.22, 1.35, 0, "TE"),
        "tm0_220_135": slab_neff(N_SI, N_OX, 0.22, 1.35, 0, "TM"),
        "te0_220_155": slab_neff(N_SI, N_OX, 0.22, 1.55, 0, "TE"),
        "cutoff_m1_nm_131": cutoff_thickness(N_SI, N_OX, 1.31, 1) * 1e3,
        "cutoff_m1_nm_135": dc * 1e3,
        "cutoff_m1_nm_155": cutoff_thickness(N_SI, N_OX, 1.55, 1) * 1e3,
        "thickness_for_2p7_nm": brentq(lambda x: slab_neff(N_SI, N_OX, x, 1.35) - 2.7, 0.05, 0.22) * 1e3,
        "na_si_sio2": float(np.sqrt(N_SI**2 - N_OX**2)),
        "critical_angle_deg": float(np.degrees(np.arcsin(N_OX / N_SI))),
    }


def fig_width(d, out, numbers):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.76), sharey=True)
    numbers["width"] = {}
    for ax, wl, key in ((axes[0], 1.35, "wsweep_135"), (axes[1], 1.55, "wsweep_155")):
        m = classify(d[key])
        for name, c, ls in (("TE0", style.COLOR_INDIGO, "-"), ("TE1", style.COLOR_INDIGO, "--"),
                            ("TM0", style.COLOR_TEAL, "-"), ("TM1", style.COLOR_TEAL, "--")):
            if len(m[name]):
                ax.plot(m[name][:, 0] * 1e3, m[name][:, 1], ls, color=c, lw=1.6, marker="o", ms=2.5,
                        label=f"{name} (3D mode solver)")
        ws = np.linspace(0.2, 0.8, 61)
        ax.plot(ws * 1e3, [eim_strip(w, wl=wl)[0] for w in ws], color=style.COLOR_REFERENCE, lw=1.0,
                ls="-.", label="TE0, effective-index method")
        ax.axhline(N_OX, color=style.COLOR_REFERENCE, lw=0.8, ls=":")
        wc = cutoff_width(m["TE1"])
        ax.axvline(wc * 1e3, color=style.COLOR_ROSE, lw=0.9)
        ax.text(wc * 1e3 - 8, 2.75, f"TE1 appears\n≈ {wc * 1e3:.0f} nm", ha="right", va="center",
                fontsize=8, color=style.COLOR_ROSE)
        ax.set_title(f"λ = {wl:.2f} µm, h = 220 nm", fontsize=9)
        ax.set_xlabel("core width w (nm)")
        ax.set_xlim(200, 800)
        ax.set_ylim(1.35, 3.0)
        te0_500 = m["TE0"][np.isclose(m["TE0"][:, 0], 0.5), 1][0]
        numbers["width"][str(wl)] = {"te1_cutoff_nm": wc * 1e3, "te0_500": te0_500,
                                     "eim_500": eim_strip(0.5, wl=wl)[0],
                                     "eim_slab_index": eim_strip(0.5, wl=wl)[1],
                                     "tm0_500": m["TM0"][np.isclose(m["TM0"][:, 0], 0.5), 1][0]}
        if len(m["TE1"]) and np.any(np.isclose(m["TE1"][:, 0], 0.5)):
            numbers["width"][str(wl)]["te1_500"] = m["TE1"][np.isclose(m["TE1"][:, 0], 0.5), 1][0]
    axes[0].set_ylabel("effective index $n_\\mathrm{eff}$")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out / "fig3_width_modes.png", dpi=200)
    plt.close(fig)


def fig_dispersion(d, out, numbers):
    wl = d["wl_um"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.67))
    a1.plot(wl, d["neff_nondisp"], color=style.COLOR_INDIGO, lw=1.6, label="constant material indices")
    a1.plot(wl, d["neff_disp"], color=style.COLOR_TEAL, lw=1.6, ls="--", label="with Si and SiO₂ dispersion")
    a2.plot(wl, d["ng_nondisp"], color=style.COLOR_INDIGO, lw=1.6)
    a2.plot(wl, d["ng_disp"], color=style.COLOR_TEAL, lw=1.6, ls="--")
    a1.set_ylabel("effective index $n_\\mathrm{eff}$")
    a2.set_ylabel("group index $n_g$")
    a2.set_ylim(3.6, 4.4)
    for a in (a1, a2):
        a.set_xlabel("wavelength (µm)")
        a.set_xlim(1.26, 1.62)
    h, lab = a1.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig4_dispersion.png", dpi=200)
    plt.close(fig)
    i135 = int(np.argmin(np.abs(wl - 1.35)))  # 1.34/1.36 grid: interpolate instead
    numbers["dispersion"] = {
        "neff_nd_135": float(np.interp(1.35, wl, d["neff_nondisp"])),
        "ng_nd_135": float(np.interp(1.35, wl, d["ng_nondisp"])),
        "neff_d_135": float(np.interp(1.35, wl, d["neff_disp"])),
        "ng_d_135": float(np.interp(1.35, wl, d["ng_disp"])),
        "neff_nd_155": float(np.interp(1.55, wl, d["neff_nondisp"])),
        "ng_nd_155": float(np.interp(1.55, wl, d["ng_nondisp"])),
        "neff_d_155": float(np.interp(1.55, wl, d["neff_disp"])),
        "ng_d_155": float(np.interp(1.55, wl, d["ng_disp"])),
        "dneff_dwl_d_135": float(np.gradient(d["neff_disp"], wl)[i135]),
        "fine_te0_135": float(d["neff_te0"]), "fine_ng_te0_135": float(d["ng_te0"]),
        "fine_tm0_135": float(d["neff_tm0"]),
    }


def fig_cutback(out, numbers):
    L, wl, s21 = load_sweep()
    k0 = 2 * np.pi / wl
    phase = np.unwrap(np.angle(s21), axis=0)                 # unwrap along length
    slope = np.polyfit(L, phase, 1)[0]                        # rad/um, one per wavelength
    neff_fdtd = np.abs(slope) / k0
    neff_solver = np.array([slab_modes(None, wl=x, n_modes=1)[0][0] for x in wl])
    i0 = int(np.argmin(np.abs(wl - WL0)))
    p0 = np.polyfit(L, phase[:, i0], 1)
    resid = phase[:, i0] - np.polyval(p0, L)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.67))
    sgn = np.sign(p0[0])    # accumulated phase relative to L = 10 um, positive (sign convention-free)
    a1.plot(L, sgn * (phase[:, i0] - phase[0, i0]), "o", color=style.COLOR_STEEL, ms=5, label="FDTD")
    a1.plot(L, sgn * (np.polyval(p0, L) - phase[0, i0]), color=style.COLOR_REFERENCE, lw=1, ls="--", label="linear fit")
    a1.set_xlabel("waveguide length L (µm)")
    a1.set_ylabel("accumulated phase $\\Delta\\phi$ (rad)")
    a1.set_title(f"λ = {wl[i0]:.2f} µm", fontsize=10)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=9)
    a2.plot(wl, neff_fdtd, "o", color=style.COLOR_STEEL, ms=4, label="FDTD cutback (20 px/µm)")
    a2.plot(wl, neff_solver, color=style.COLOR_REFERENCE, lw=1.4, label="mode solver, same 2D model")
    a2.set_xlabel("wavelength (µm)")
    a2.set_ylabel("effective index $n_\\mathrm{eff}$")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig5_cutback.png", dpi=200)
    plt.close(fig)

    # group index from the FDTD n_eff(wl) and from the solver, same formula
    def ng_of(n):
        c = np.polyfit(wl, n, 2)
        return float(np.polyval(c, WL0) - WL0 * np.polyval(np.polyder(c), WL0))

    numbers["cutback"] = {
        "wl_point": float(wl[i0]), "neff_fdtd": float(neff_fdtd[i0]), "neff_solver": float(neff_solver[i0]),
        "excess_pct": float(100 * (neff_fdtd[i0] / neff_solver[i0] - 1)),
        "excess_pct_range": [float(100 * np.min(neff_fdtd / neff_solver - 1)),
                             float(100 * np.max(neff_fdtd / neff_solver - 1))],
        "max_resid_rad": float(np.max(np.abs(resid))),
        "ng_fdtd": ng_of(neff_fdtd), "ng_solver": ng_of(neff_solver),
        "phase_per_um": float(abs(p0[0])), "n_lengths": int(len(L)), "L_range": [float(L[0]), float(L[-1])],
        "guided_wavelength_um": float(wl[i0] / neff_solver[i0]),
        "cells_per_guided_wavelength_res20": float(20 * wl[i0] / neff_solver[i0]),
        "design_point_neff": yaml.safe_load(open(REPO / "si/data/design_points/waveguide.yaml"))["fitted_model"]["n_eff"],
    }
    numbers["model_2d"] = {"neff_135": float(slab_modes(None, wl=1.35, n_modes=1)[0][0])}


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(MPB)
    numbers = {}
    fig_cross_section(d, out)
    fig_slab(out, numbers)
    fig_width(d, out, numbers)
    fig_dispersion(d, out, numbers)
    fig_cutback(out, numbers)
    numbers["fdtd_3d"] = yaml.safe_load(open(REPO / "3d/data/design_points/waveguide_3d.yaml"))
    print(json.dumps(numbers, indent=1, default=float))
    (out / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "blog_figures_out/silicon_waveguides")
