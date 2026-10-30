"""Figures and numbers for the blog article "From Waveguide Coupling to
Interference and Resonance: A Physics-Based Introduction to PIC Circuits".

Fig. 1 is a schematic; Fig. 2 re-plots saved FDTD spectra of the MZI
(si/data/sparams/mzi/selected) and the racetrack ring (si/data/sparams/racetrack/sweep,
Lc = 3 um) on a common axis normalized to each device's free spectral range;
Fig. 3 is the closed-form maximally flat (halfband) lattice response that
circuits/wdm_sax.ipynb synthesizes. No FDTD is run and meep is never imported.

Usage (from the repo root):
    conda run -n mp python scripts/blog_figures/article4_synthesis.py <out_dir>
"""

import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.optimize import least_squares

from pic_toolkit import style

REPO = Path(__file__).resolve().parents[2]
SP = REPO / "si" / "data" / "sparams"
NG_MZI = 2.908   # group index from the MZI fringe spacing (article 2)
NG_RING = 2.978  # group index from the racetrack FSR (article 3)

style.apply_style()
assert "meep" not in sys.modules


def load(path):
    d = np.load(path)
    wl = d["wavelengths_um"]
    o = np.argsort(wl)
    return {k: d[k][o] for k in d.files if d[k].shape == wl.shape}


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="none", alpha=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color="white")


def arrow(ax, p0, p1, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, color="k", lw=0.9, **kw))


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numbers = {}

    # ---- Fig. 1: the three building blocks as signal-flow diagrams (schematic)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for ax in axes:
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 3)
        ax.axis("off")
    a1, a2, a3 = axes
    box(a1, 1.3, 1.0, 1.4, 1.0, "C", style.COLOR_STEEL)
    for y in (1.25, 1.75):
        arrow(a1, (0.3, y), (1.3, y))
        arrow(a1, (2.7, y), (3.7, y))
    a1.set_title("Coupling\n(directional coupler)", fontsize=10)
    a1.text(2.0, 0.45, "2 supermodes beat\nalong the coupler", ha="center", va="center", fontsize=8)
    box(a2, 0.2, 1.0, 0.8, 1.0, "C", style.COLOR_STEEL)
    box(a2, 3.0, 1.0, 0.8, 1.0, "C", style.COLOR_STEEL)
    box(a2, 1.45, 1.75, 1.1, 0.45, r"$e^{-j\phi_1}$", style.COLOR_TEAL)
    box(a2, 1.45, 0.8, 1.1, 0.45, r"$e^{-j\phi_2}$", style.COLOR_TEAL)
    arrow(a2, (1.0, 1.75), (1.45, 1.95))
    arrow(a2, (2.55, 1.95), (3.0, 1.75))
    arrow(a2, (1.0, 1.25), (1.45, 1.05))
    arrow(a2, (2.55, 1.05), (3.0, 1.25))
    a2.set_title("Interference\n(Mach–Zehnder)", fontsize=10)
    a2.text(2.0, 0.3, "2 paths, one pass", ha="center", va="center", fontsize=8)
    box(a3, 1.3, 1.6, 1.4, 0.8, "C", style.COLOR_STEEL)
    box(a3, 1.45, 0.45, 1.1, 0.45, r"$a\,e^{j\phi}$", style.COLOR_TEAL)
    arrow(a3, (0.3, 2.2), (1.3, 2.2))
    arrow(a3, (2.7, 2.2), (3.7, 2.2))
    arrow(a3, (2.7, 1.8), (3.1, 0.7), connectionstyle="arc3,rad=-0.4")
    arrow(a3, (3.1, 0.67), (2.55, 0.67))
    arrow(a3, (1.45, 0.67), (0.9, 0.67))
    arrow(a3, (0.9, 0.7), (1.3, 1.8), connectionstyle="arc3,rad=-0.4")
    a3.set_title("Resonance\n(ring resonator)", fontsize=10)
    a3.text(2.0, 0.08, "feedback: infinitely many round trips", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_building_blocks.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 2: interfere once vs. many times -- FDTD spectra on an FSR-normalized axis
    m = load(SP / "mzi" / "selected" / "mzi_selected.npz")
    nu_m, Tm = 1 / m["wavelengths_um"], abs(m["S41"]) ** 2
    i_m = int(np.argmin(np.where(np.abs(m["wavelengths_um"] - 1.337) < 0.01, Tm, 9)))
    fsr_m = 1 / (NG_MZI * 12.5)
    x_m = (nu_m - nu_m[i_m]) / fsr_m
    r = load(SP / "racetrack" / "sweep" / "racetrack_coupling_length3.0.npz")
    nu_r, Tr = 1 / r["wavelengths_um"], abs(r["s21"]) ** 2
    i_r = int(np.argmin(Tr))
    L = 2 * np.pi * 5 + 2 * 3.0
    fsr_r = 1 / (NG_RING * L)
    x_r = (nu_r - nu_r[i_r]) / fsr_r
    # finesse = FSR / FWHM (FWHM at half depth)
    def fwhm_x(x, T):
        half = (1 + T.min()) / 2
        idx = np.where(T < half)[0]
        return x[idx[-1]] - x[idx[0]]
    F_ring = 1 / abs(fwhm_x(x_r, Tr))
    t, a = 0.9814, 0.9177  # under-coupled fit for Lc = 3 um (article 3)
    F_ring_theory = np.pi * np.sqrt(a * t) / (1 - a * t)
    numbers["finesse"] = dict(mzi_theory=2.0, ring_measured=float(F_ring), ring_theory=float(F_ring_theory),
                              ring_fsr_nm=float(r["wavelengths_um"][i_r] ** 2 / (NG_RING * L) * 1e3),
                              mzi_fsr_nm=float(m["wavelengths_um"][i_m] ** 2 / (NG_MZI * 12.5) * 1e3))
    xx = np.linspace(-0.5, 0.5, 600)
    e = np.exp(1j * 2 * np.pi * xx)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(xx, np.sin(np.pi * xx) ** 2, color=style.COLOR_REFERENCE, lw=1)
    ax.plot(xx, abs((t - a * e) / (1 - t * a * e)) ** 2, color=style.COLOR_REFERENCE, lw=1, ls="--")
    sel = np.abs(x_m) <= 0.5
    ax.plot(x_m[sel], Tm[sel], "o", ms=3, color=style.COLOR_INDIGO,
            label="MZI cross port, FDTD (ΔL = 12.5 µm; gray: sin²)")
    sel = np.abs(x_r) <= 0.5
    ax.plot(x_r[sel], Tr[sel], ".", ms=2.5, color=style.COLOR_TEAL,
            label="ring through port, FDTD ($L_c$ = 3 µm; gray dashed: all-pass)")
    ax.set_xlabel(r"detuning from the transmission minimum, $(\nu-\nu_0)/\mathrm{FSR}$")
    ax.set_ylabel("transmitted power")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig2_once_vs_many.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 3: maximally flat (halfband) lattice responses, closed form
    phi = np.linspace(0, 2 * np.pi, 800)
    xs = np.sin(phi / 2) ** 2
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    widths = {}
    for k, K in enumerate((1, 2, 3)):
        P = sum(comb(2 * K - 1, j) * xs**j * (1 - xs) ** (2 * K - 1 - j) for j in range(K, 2 * K))
        ax.plot(phi / np.pi, P, color=style.COLOR_CYCLE[[0, 2, 5][k]],
                label=f"N = {2 * K} couplers" + (" (plain MZI)" if K == 1 else ""))
        above = phi[P >= 10 ** -0.1]
        widths[2 * K] = float((above.max() - above.min()) / (2 * np.pi))
    ax.set_xlabel(r"delay-line phase $\phi/\pi$ (one FSR = 2)")
    ax.set_ylabel("bar-port power (ideal, lossless)")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig3_lattice.png", dpi=200)
    plt.close(fig)
    numbers["lattice_1dB_width_fraction_of_FSR"] = widths

    print(json.dumps(numbers, indent=1))
    return numbers


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "blog_figures_out")
