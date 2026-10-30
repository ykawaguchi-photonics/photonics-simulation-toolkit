"""Figures and numbers for the blog article "Mach-Zehnder Modulators".

Reads only saved data:
- si/data/blog_figures/mach_zehnder_modulators.npz (cross-section, electrode, gap,
  RF and BOX solves, written by mach_zehnder_modulators_data.py);
- tfln/data/design_points/tfln_mzm.yaml and tfln/data/sparams/tfln_mzm/{baseline,sweep}
  (2D FDTD of the full interferometer, tfln/notebooks/01_tfln_mzm.ipynb section 8).
meep is never imported. The traveling-wave EO response is the closed-form model
of meep_sim/tfln_mzm.eo_response, copied here so that this script stays meep-free.

Usage (from the repo root):
    conda run -n mp python scripts/blog_figures/mach_zehnder_modulators.py <out_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import yaml

from pic_toolkit import style

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "si" / "data" / "blog_figures" / "mach_zehnder_modulators.npz"
DP = REPO / "tfln" / "data" / "design_points" / "tfln_mzm.yaml"
SP = REPO / "tfln" / "data" / "sparams" / "tfln_mzm"
C0 = 299792458.0
METAL = "#C9A227"   # gold electrodes in schematics only

style.apply_style()
assert "meep" not in sys.modules


def eo_response(f_hz, length_m, n_rf, n_g, alpha_np_per_m, z0_ohm, zs_ohm=50.0, zl_ohm=50.0):
    """Small-signal EO response m(f) = |Phi(f)| / |Phi(0)| of a traveling-wave electrode:
    forward wave plus load reflection, V(z) = V+ (e^{-gz} + G_L e^{-2gL} e^{gz}),
    g = alpha + j w n_rf / c, integrated along the photon's path z -> t0 + z n_g / c.
    Same model as pic_toolkit.meep_sim.tfln_mzm.eo_response."""
    f = np.asarray(f_hz, dtype=float)
    w = 2 * np.pi * f
    a = np.broadcast_to(np.asarray(alpha_np_per_m, dtype=float), f.shape)
    L = length_m
    gam = a + 1j * w * n_rf / C0
    bo = w * n_g / C0
    GL = (zl_ohm - z0_ohm) / (zl_ohm + z0_ohm)
    eL = np.exp(-2 * gam * L)
    v_plus = 1.0 / ((1 + GL * eL) + zs_ohm / z0_ohm * (1 - GL * eL))

    def integ(k):
        k = np.asarray(k, dtype=complex)
        small = np.abs(k * L) < 1e-9
        out = np.empty_like(k)
        out[~small] = (np.exp(k[~small] * L) - 1) / k[~small]
        out[small] = L
        return out

    phi = v_plus * (integ(-gam + 1j * bo) + GL * eL * integ(gam + 1j * bo))
    return np.abs(phi) / abs(zl_ohm / (zl_ohm + zs_ohm) * L)


def f3db(f_hz, m, level_db=-3.0):
    db = 20 * np.log10(np.maximum(m, 1e-12))
    below = np.nonzero(db < level_db)[0]
    if len(below) == 0:
        return float("nan")
    i = below[0]
    return float(np.interp(level_db, [db[i], db[i - 1]], [f_hz[i], f_hz[i - 1]]))


# ---------------------------------------------------------------- Fig. 1
def fig_schematic(d, out):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.4, 8.0), gridspec_kw=dict(height_ratios=[1.0, 1.0]))
    fs = 10
    wgc = style.COLOR_STEEL

    # ---------------- top view
    wg = dict(color=wgc, lw=2.2)
    xs = np.linspace(0, 1, 50)
    a1.plot([-1.2, 0.0], [0, 0], **wg)
    for sg in (1, -1):
        a1.plot(xs, sg * 0.9 * (3 * xs**2 - 2 * xs**3), **wg)
        a1.plot([1.0, 7.0], [sg * 0.9, sg * 0.9], **wg)
        a1.plot(7.0 + xs, sg * 0.9 * (1 - (3 * xs**2 - 2 * xs**3)), **wg)
    a1.plot([8.0, 9.2], [0, 0], **wg)
    a1.add_patch(Rectangle((1.5, -0.5), 5.0, 1.0, fc=METAL, ec="none"))      # signal
    a1.add_patch(Rectangle((1.5, 1.3), 5.0, 0.9, fc=METAL, ec="none"))       # ground
    a1.add_patch(Rectangle((1.5, -2.2), 5.0, 0.9, fc=METAL, ec="none"))      # ground
    a1.text(1.4, 1.75, "G", ha="right", va="center", fontsize=fs, fontweight="bold")
    a1.text(1.4, 0.0, "S", ha="right", va="center", fontsize=fs, fontweight="bold")
    a1.text(1.4, -1.75, "G", ha="right", va="center", fontsize=fs, fontweight="bold")
    a1.text(4.0, 2.45, "light and RF wave travel together along the electrode (length L) →",
            ha="center", va="bottom", fontsize=9, color=style.COLOR_REFERENCE)
    a1.text(-1.3, 0.0, "light in", ha="right", va="center", fontsize=fs)
    a1.text(9.3, 0.0, "modulated\nlight out", ha="left", va="center", fontsize=fs)
    a1.annotate("RF drive V(t) applied\nbetween S and G at this end", xy=(1.42, -1.05), xytext=(-0.8, -2.75), fontsize=9,
                ha="center", va="top", arrowprops=dict(arrowstyle="->", lw=0.8))
    a1.annotate("50 Ω termination\nbetween S and G at this end", xy=(6.58, -1.05), xytext=(8.8, -2.75), fontsize=9,
                ha="center", va="top", arrowprops=dict(arrowstyle="->", lw=0.8))
    a1.annotate("arm 1", xy=(6.8, 0.9), xytext=(7.6, 1.75), fontsize=9, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6))
    a1.annotate("arm 2", xy=(6.8, -0.9), xytext=(7.6, -1.75), fontsize=9, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6))
    handles = [Patch(fc=METAL, label="gold electrodes: S = signal, G = ground"),
               Line2D([], [], color=wgc, lw=2.2, label="LN rib waveguides (splitter, two arms, combiner)")]
    a1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=9)
    a1.set_xlim(-2.6, 11.0)
    a1.set_ylim(-4.3, 3.0)
    a1.set_aspect("equal")
    a1.axis("off")
    a1.set_title("top view (schematic)", fontsize=10.5)

    # ---------------- cross-section with the drive circuit (schematic, not to scale)
    slab, rib_top, m_top = 0.45, 0.9, 1.7
    X = 11.0
    a2.add_patch(Rectangle((-X, -2.6), 2 * X, 0.6, fc="0.65", ec="none"))           # Si handle
    a2.add_patch(Rectangle((-X, -2.0), 2 * X, 2.0, fc="0.88", ec="none"))           # buried oxide
    a2.add_patch(Rectangle((-X, 0.0), 2 * X, slab, fc=wgc, ec="none", alpha=0.45))  # LN slab
    for c in (-4.0, 4.0):
        a2.add_patch(Rectangle((c - 0.6, 0.0), 1.2, rib_top, fc=wgc, ec="none"))   # LN ribs
    for x0, x1 in ((-X, -6.0), (-2.0, 2.0), (6.0, X)):
        a2.add_patch(Rectangle((x0, slab), x1 - x0, m_top - slab, fc=METAL, ec="none"))
    # drive circuit: source between S and both grounds
    ys = 4.6
    a2.add_patch(Circle((0, ys), 0.45, fill=False, ec="k", lw=1.0))
    a2.text(0, ys, "~", ha="center", va="center", fontsize=13)
    a2.text(0.65, ys + 0.35, "V(t)", ha="left", va="bottom", fontsize=fs)
    a2.plot([0, 0], [ys - 0.45, m_top], color="k", lw=1.0)                          # to S
    a2.plot([-0.45, -9.0, -9.0], [ys, ys, m_top], color="k", lw=1.0)                # to left G
    a2.plot([0.45, 9.0, 9.0], [ys, ys, m_top], color="k", lw=1.0)                   # to right G
    a2.text(0.25, 2.3, "S: +V", ha="left", va="center", fontsize=fs)
    a2.text(-8.75, 2.3, "G: 0 V", ha="left", va="center", fontsize=fs)
    a2.text(9.25, 2.3, "G: 0 V", ha="left", va="center", fontsize=fs)
    # field in the two gaps: from S towards G, opposite directions -> push-pull
    for xa, xb in ((-2.3, -5.7), (2.3, 5.7)):
        a2.annotate("", xy=(xb, 1.3), xytext=(xa, 1.3), arrowprops=dict(arrowstyle="->", lw=1.6, color=style.COLOR_ROSE))
    a2.text(-4.0, 3.4, "gap over arm 1:\nfield E points left", ha="center", va="center", fontsize=9,
            color=style.COLOR_ROSE)
    a2.text(4.0, 3.4, "gap over arm 2:\nfield E points right", ha="center", va="center", fontsize=9,
            color=style.COLOR_ROSE)
    a2.text(X + 0.4, 0.2, "LN slab", ha="left", va="center", fontsize=fs)
    a2.text(X + 0.4, -1.1, "buried oxide", ha="left", va="center", fontsize=fs)
    a2.text(X + 0.4, -2.3, "Si handle", ha="left", va="center", fontsize=fs)
    a2.set_xlim(-X - 0.3, X + 6.5)
    a2.set_ylim(-2.8, 5.5)
    a2.axis("off")
    handles2 = [Patch(fc=METAL, label="gold electrodes (S, G)"),
                Patch(fc=wgc, label="LN rib waveguides (arm 1 left, arm 2 right)"),
                Line2D([], [], color=style.COLOR_ROSE, lw=1.6, marker=">", label="electric field E in the gaps")]
    a2.legend(handles=handles2, loc="upper center", bbox_to_anchor=(0.42, 0.0), ncol=2, frameon=False, fontsize=9)
    a2.set_title("cross-section with the drive circuit (schematic, not to scale)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out / "fig1_layout.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 2
def fig_transfer(out):
    v = np.linspace(-1.5, 1.5, 601)                # V / V_pi
    T = np.cos((np.pi / 2 + np.pi * v) / 2) ** 2  # biased at quadrature
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.plot(v, T, color=style.COLOR_STEEL, lw=1.8, label="output power, biased at quadrature")
    vs = np.linspace(-0.2, 0.2, 2)
    ax.plot(vs, 0.5 - np.pi / 2 * vs, color=style.COLOR_REFERENCE, lw=1.0, ls="--",
            label="slope at quadrature, $\\pi/(2V_\\pi)$")
    ax.plot([0], [0.5], "o", color=style.COLOR_ROSE, ms=6, zorder=5, label="quadrature bias point")
    for lv in (-0.15, -0.05, 0.05, 0.15):
        ax.plot([lv], [np.cos((np.pi / 2 + np.pi * lv) / 2) ** 2], "s", color=style.COLOR_INDIGO, ms=4, zorder=6)
    ax.plot([], [], "s", color=style.COLOR_INDIGO, ms=4, label="four PAM4 drive levels ($\\pm0.05$, $\\pm0.15\\,V_\\pi$)")
    ax.set_xlabel("drive voltage $V/V_\\pi$")
    ax.set_ylabel("output power (normalized)")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig2_transfer.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 3
def fig_overlap(d, out):
    x, y, Ex = d["field_x_um"], d["field_y_um"], d["field_Ex"] * 1e-6  # V/um for 1 V
    xc = float(d["wg_center_um"])
    gap, ws, slab, film, rib, tm_ = (d["gap_um"], d["signal_width_um"], d["slab_t_um"], d["film_um"],
                                     d["rib_width_um"], d["metal_um"])
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    im = ax.pcolormesh(x, y, np.abs(Ex), cmap=style.CMAP_DENSITY, vmin=0, vmax=0.25, shading="auto")
    mx, my, I = d["mode_x_um"] + xc, d["mode_y_um"], d["mode_I"] / d["mode_I"].max()
    ax.contour(mx, my, I, levels=[0.1, 0.4, 0.8], colors="w", linewidths=1.0)
    ax.add_patch(Rectangle((xc - rib / 2, 0), rib, film, fill=False, ec="w", lw=1.2, ls="--"))
    for x0, x1 in ((xc - gap / 2 - 3, xc - gap / 2), (xc + gap / 2, xc + gap / 2 + 3)):
        ax.add_patch(Rectangle((x0, slab), x1 - x0, tm_, fc=METAL, ec="none"))
    ax.plot([x.min(), x.max()], [0, 0], color="0.85", lw=0.6)
    ax.plot([x.min(), x.max()], [slab, slab], color="0.85", lw=0.6)
    x0, x1 = xc - gap / 2 - 1.2, xc + gap / 2 + 1.2
    ax.set_xlim(x0, x1)
    ax.set_ylim(-1.0, 2.0)
    # labels outside the plotted area (above the axes), with leaders down to the structures
    lab = dict(fontsize=9, ha="center", va="bottom", annotation_clip=False,
               arrowprops=dict(arrowstyle="-", lw=0.7, color="0.3"))
    xg, xs_ = (x0 + xc - gap / 2) / 2, (x1 + xc + gap / 2) / 2
    ax.annotate("ground electrode\n(G, 0 V)", xy=(xg, 1.3), xytext=(xg, 2.75), **lab)
    ax.annotate("signal electrode\n(S, +V)", xy=(xs_, 1.3), xytext=(xs_, 2.75), **lab)
    ax.annotate("LN rib waveguide", xy=(xc, film), xytext=(xc, 2.75), **lab)
    ax.annotate("", xy=(xc - gap / 2 + 0.2, 2.35), xytext=(xc - 0.25, 2.35), annotation_clip=False,
                arrowprops=dict(arrowstyle="->", lw=1.4, color=style.COLOR_ROSE))
    ax.annotate("", xy=(xc + 0.25, 2.35), xytext=(xc + gap / 2 - 0.2, 2.35), annotation_clip=False,
                arrowprops=dict(arrowstyle="-", lw=1.4, color=style.COLOR_ROSE))
    ax.text(xc + gap / 4, 2.42, "field E: from S towards G", ha="center", va="bottom", fontsize=8.5,
            color=style.COLOR_ROSE)
    ax.set_aspect("equal")
    ax.set_xlabel("lateral position (µm)")
    ax.set_ylabel("height (µm)")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("lateral RF field $|E_x|$ (V/µm per volt)")
    fig.tight_layout()
    fig.savefig(out / "fig3_overlap.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 4
def fig_gap(d, out):
    g = d["gap_sweep"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.38))
    a1.plot(g[:, 0], g[:, 1], "o-", color=style.COLOR_STEEL, ms=5)
    a1.set_xlabel("electrode gap g (µm)")
    a1.set_ylabel("$V_\\pi L$, push-pull (V·cm)")
    a2.semilogy(g[:, 0], g[:, 3], "o-", color=style.COLOR_STEEL, ms=5)
    a2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.set_xlabel("electrode gap g (µm)")
    a2.set_ylabel("mode intensity at electrode corner\n(relative to peak)")
    for a in (a1, a2):
        a.axvline(5.0, color=style.COLOR_ROSE, lw=0.9)
    a1.text(5.12, 3.55, "design\ng = 5 µm", ha="left", va="top", fontsize=8, color=style.COLOR_ROSE)
    fig.tight_layout()
    fig.savefig(out / "fig4_gap_tradeoff.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 6 (bandwidth)
def fig_bandwidth(d, out, numbers):
    # extend the saved conductor loss (computed to 200 GHz) to 400 GHz: the skin-effect
    # model is exactly alpha ~ sqrt(f) (R_s = sqrt(pi f mu0 / sigma)), so scale its 100 GHz value
    a100 = float(np.interp(100e9, d["f_hz"], d["alpha_np_per_m"]))
    F = np.linspace(0.5e9, 400e9, 800)
    a = a100 * np.sqrt(F / 100e9)
    assert np.allclose(np.interp(d["f_hz"], F, a), d["alpha_np_per_m"], rtol=2e-3)
    nrf, ng, z0 = float(d["n_rf"]), float(d["ng_dispersive"]), float(d["z0_ohm"])
    vpi_l = float(d["vpi_l_V_cm"])
    cases = [("velocity mismatch only", dict(n_rf=nrf, alpha=0 * a, z0=50.0)),
             ("conductor loss only", dict(n_rf=ng, alpha=a, z0=50.0)),
             ("impedance mismatch only", dict(n_rf=ng, alpha=0 * a, z0=z0)),
             ("all three", dict(n_rf=nrf, alpha=a, z0=z0))]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.6))
    numbers["bw_1cm"] = {}
    for (name, c), col in zip(cases, [style.COLOR_INDIGO, style.COLOR_CYAN, style.COLOR_OLIVE, style.COLOR_STEEL]):
        m = eo_response(F, 0.01, c["n_rf"], ng, c["alpha"], c["z0"])
        f3 = f3db(F, m)
        numbers["bw_1cm"][name] = f3 / 1e9
        lab = f"{name}: {f3 / 1e9:.0f} GHz" if np.isfinite(f3) else f"{name}: stays above −3 dB"
        a1.plot(F / 1e9, 20 * np.log10(m), color=col, lw=2.0 if name == "all three" else 1.2, label=lab)
    a1.axhline(-3, color=style.COLOR_REFERENCE, lw=0.8, ls=":")
    a1.set_ylim(-10, 1)
    a1.set_xlim(0, 200)
    a1.set_xlabel("modulation frequency (GHz)")
    a1.set_ylabel("EO response (dB)")
    a1.set_title("electrode length $L$ = 1 cm", fontsize=10)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, frameon=False, fontsize=9)

    lengths = np.round(np.arange(2.0, 20.01, 0.5), 2)
    f3s = np.array([f3db(F, eo_response(F, L * 1e-3, nrf, ng, a, z0)) for L in lengths])
    vpi = vpi_l / (lengths / 10)
    target = 67.0
    ok = f3s >= target * 1e9
    Ld = float(lengths[ok].max())
    f3d = float(f3s[lengths == Ld][0]) / 1e9
    vd = vpi_l / (Ld / 10)
    a2.plot(f3s / 1e9, vpi, "-", color=style.COLOR_STEEL, lw=1.8, label="electrode length 2–20 mm")
    a2.axvline(target, color=style.COLOR_REFERENCE, lw=0.8, ls="--", label="bandwidth target 67 GHz")
    a2.plot([f3d], [vd], "o", color=style.COLOR_ROSE, ms=6, label=f"design: $L$ = {Ld:.1f} mm, $V_\\pi$ = {vd:.2f} V, {f3d:.0f} GHz")
    a2.set_xlabel("EO 3 dB bandwidth (GHz)")
    a2.set_ylabel("$V_\\pi$ (V)")
    a2.set_title("longer electrode: lower $V_\\pi$, lower bandwidth", fontsize=10)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig6_bandwidth.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    numbers["design"] = {"L_mm": Ld, "f3db_GHz": f3d, "vpi_V": vd}
    for Lmm in (2.0, 10.0, 20.0):
        i = int(np.argmin(np.abs(lengths - Lmm)))
        numbers["design"][f"f3db_at_{Lmm:.0f}mm"] = float(f3s[i] / 1e9)
        numbers["design"][f"vpi_at_{Lmm:.0f}mm"] = float(vpi[i])
    numbers["velocity_only_formula_1cm_GHz"] = 1.392 * C0 / (np.pi * 0.01 * abs(nrf - ng)) / 1e9


# ---------------------------------------------------------------- Fig. 5 (FDTD switching)
def fig_fdtd(out, numbers):
    dp = yaml.safe_load(open(DP))
    dn_vpi = dp["meep_2d_mzm"]["delta_n_vpi"]
    pts = []
    for f in sorted((SP / "sweep").glob("tfln_mzm_dn*.npz")) + [SP / "baseline" / "tfln_mzm_vpi.npz"]:
        d = np.load(f)
        meta = json.load(open(f.with_suffix(".json")))
        wl = d["wavelengths_um"]
        i = int(np.argmin(np.abs(wl - 1.35)))
        pts.append((meta.get("delta_n", meta.get("delta_n_vpi")), abs(d["S31"][i]) ** 2, abs(d["S41"][i]) ** 2))
    pts = np.array(sorted(pts))
    v = np.linspace(0, 1.1, 300)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(v, np.sin(np.pi * v / 2) ** 2, color=style.COLOR_REFERENCE, lw=1.0, label="ideal: $\\sin^2(\\pi V/2V_\\pi)$, $\\cos^2(\\pi V/2V_\\pi)$")
    ax.plot(v, np.cos(np.pi * v / 2) ** 2, color=style.COLOR_REFERENCE, lw=1.0)
    ax.plot(pts[:, 0] / dn_vpi, pts[:, 1], "o", color=style.COLOR_INDIGO, ms=6, label="FDTD, bar output")
    ax.plot(pts[:, 0] / dn_vpi, pts[:, 2], "s", color=style.COLOR_TEAL, ms=6, label="FDTD, cross output")
    ax.set_xlabel("equivalent drive $V/V_\\pi$ (index change $\\Delta n/\\Delta n_\\pi$)")
    ax.set_ylabel("output power at 1.35 µm")
    ax.set_xlim(-0.03, 1.1)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig5_fdtd_switching.png", dpi=200)
    plt.close(fig)
    numbers["fdtd"] = {"points": pts.tolist(), "dn_vpi": dn_vpi}


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA)
    numbers = {k: (float(d[k]) if d[k].ndim == 0 else None) for k in d.files if d[k].ndim == 0}
    numbers["gap_sweep"] = d["gap_sweep"].tolist()
    numbers["box_sweep"] = d["box_sweep"].tolist()
    numbers["box_velocity_f3db_1cm_GHz"] = [1.392 * C0 / (np.pi * 0.01 * abs(r[1] - float(d["ng_dispersive"]))) / 1e9
                                            for r in d["box_sweep"]]
    numbers["alpha_db_per_cm_100GHz"] = float(np.interp(100e9, d["f_hz"], d["alpha_db_per_cm"]))
    numbers["lumped_rc_GHz_1cm"] = 1 / (2 * np.pi * 50 * float(d["C_pF_per_m"]) * 1e-12 * 0.01) / 1e9
    ne = 2.1437460671713318
    numbers["dn_1V_5um"] = 0.5 * ne**3 * 30.9e-12 * (1 / 5e-6)
    numbers["n3r_pm_per_V"] = ne**3 * 30.9
    fig_schematic(d, out)
    fig_transfer(out)
    fig_overlap(d, out)
    fig_gap(d, out)
    fig_bandwidth(d, out, numbers)
    fig_fdtd(out, numbers)
    numbers["design_point"] = yaml.safe_load(open(DP))
    txt = json.dumps(numbers, indent=1, default=float)
    print(txt)
    (out / "numbers.json").write_text(txt)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "blog_figures_out/mach_zehnder_modulators")
