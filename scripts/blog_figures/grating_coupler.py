"""Figures and numbers for the blog article "Grating Couplers: Phase Matching,
Directionality and Mode Overlap in FDTD".

Reads only cached results of the SiN grating study (meta/data/...) and the design
point meta/data/sin_grating_conventional.yaml -- no FDTD is run and meep is never
imported. The fiber-overlap formula is re-implemented here (identical to
pic_toolkit.meep_sim.sin_grating.eta_overlap, which imports meep at module level)
and the recomputed efficiency is checked against the saved design point.

Usage (from the repo root):
    conda run -n mp python scripts/blog_figures/grating_coupler.py <out_dir>
"""

import glob
import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import yaml
from scipy.optimize import minimize

from pic_toolkit import style

REPO = Path(__file__).resolve().parents[2]
META = REPO / "meta" / "data"
PHASE_SIGN = +1  # forward-wave DFT phase grows with x (verified in meta notebook 01, Section 2)

style.apply_style()
assert "meep" not in sys.modules


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Fiber overlap -- same formula as sin_grating.fiber_mode / eta_overlap / best_x0
# ---------------------------------------------------------------------------
def fiber_mode(x, x0, wl, p):
    th = math.radians(p["fiber_angle_deg"])
    w0 = p["mfd_um"] / 2
    k = 2 * np.pi / wl * p["n_top"]
    ey = np.exp(-(((x - x0) * math.cos(th)) ** 2) / w0**2) * np.exp(PHASE_SIGN * 1j * k * math.sin(th) * (x - x0))
    return ey, -p["n_top"] * math.cos(th) * ey


def eta_overlap(x, ez, hx, wl, p_in, p, x0):
    ey_s, hx_s = -ez, hx  # physical Ey = -Ez(Meep) in the rotated x-z frame
    ey_f, hx_f = fiber_mode(x, x0, wl, p)
    dx = x[1] - x[0]
    X = np.sum(ey_s * np.conj(hx_f) + np.conj(ey_f) * hx_s) * dx
    p_f = abs(np.real(np.sum(ey_f * np.conj(hx_f))) * dx)
    return float(np.abs(X) ** 2 / (4 * p_f * p_in))


def best_x0(r, i):
    x = r["x_up"]
    grid = np.arange(x[0] + 2.0, x[-1] - 2.0, 0.05)
    etas = [eta_overlap(x, r["ez_up"][i], r["hx_up"][i], r["wl"][i], r["p_in"][i], r["params"], x0) for x0 in grid]
    k = int(np.argmax(etas))
    return float(grid[k]), float(etas[k])


def decay_length(r, i):
    """1/e decay length of the emitted |Ey|^2 along the near-field line (as in meta notebook 03)."""
    x, I = r["x_up"], np.abs(r["ez_up"][i]) ** 2
    inside = (x > 0) & (x < r["dom"]["x_g1"])
    k = int(np.argmax(np.where(inside, I, 0)))
    sel = inside & (x > x[k]) & (I > 1e-3 * I[k])
    return float(-1 / np.polyfit(x[sel], np.log(I[sel]), 1)[0])


def report(r, wl0):
    i = int(np.argmin(np.abs(r["wl"] - wl0)))
    x0, eta = best_x0(r, i)
    spec = np.array([eta_overlap(r["x_up"], r["ez_up"][j], r["hx_up"][j], r["wl"][j], r["p_in"][j], r["params"], x0)
                     for j in range(len(r["wl"]))])
    up, down = float(r["up"][i]), float(r["down"][i])
    return dict(i=i, x0=x0, eta=eta, eta_db=10 * np.log10(eta), up=up, down=down, R=float(r["R"][i]),
                T=float(r["T"][i]), directionality=up / (up + down), mode_match=eta / up,
                L_c=decay_length(r, i), spec=spec)


def band_1db(wl, spec):
    """Width of the region where eta is within 1 dB of its peak (linear interpolation)."""
    o = np.argsort(wl)
    wl, s = wl[o], 10 * np.log10(spec[o])
    lvl = s.max() - 1
    above = s >= lvl
    idx = np.where(above)[0]
    lo, hi = idx[0], idx[-1]
    wl_lo = np.interp(lvl, [s[lo - 1], s[lo]], [wl[lo - 1], wl[lo]]) if lo > 0 else wl[0]
    wl_hi = np.interp(lvl, [s[hi + 1], s[hi]], [wl[hi + 1], wl[hi]]) if hi < len(wl) - 1 else wl[-1]
    return wl_lo, wl_hi


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numbers = {}
    dp = yaml.safe_load((META / "sin_grating_conventional.yaml").read_text())
    conv = load(META / "step2" / "final_grid_res160.pkl")
    asi = load(META / "step4_asi_overlay" / "uniform_TE_res160.pkl")
    p = conv["params"]
    n_box, wl0, P, th = p["n_box"], p["wl0_um"], p["period_um"], p["fiber_angle_deg"]

    # ---- numbers: conventional SiN optimum and a-Si overlay, recomputed from the cache
    rc = report(conv, 1.35)
    ra = report(asi, 1.31)
    assert abs(rc["eta"] - dp["measured"]["eta"]) < 2e-3, (rc["eta"], dp["measured"]["eta"])
    for name, r, d in (("conventional", rc, conv), ("asi_overlay", ra, asi)):
        lo, hi = band_1db(d["wl"], r["spec"])
        numbers[name] = {k: v for k, v in r.items() if k not in ("spec", "i")}
        numbers[name].update(band_1db_nm=(hi - lo) * 1000, wl_peak_um=float(d["wl"][np.argmax(r["spec"])]))

    # ---- analytic: best overlap of an exponential emission profile with the fiber Gaussian
    xs = np.linspace(-60, 120, 60001)
    dx = xs[1] - xs[0]
    w0 = p["mfd_um"] / 2

    def neg_ov(q):
        Lc, x0 = q
        a = np.where(xs > 0, np.exp(-xs / (2 * Lc)), 0)
        b = np.exp(-((xs - x0) ** 2) / w0**2)
        return -(np.sum(a * b) * dx) ** 2 / (np.sum(a * a) * dx * np.sum(b * b) * dx)

    best = minimize(neg_ov, [3.4, 3.0], method="Nelder-Mead")
    at_Lc = -minimize(lambda q: neg_ov([rc["L_c"], q[0]]), [8.0], method="Nelder-Mead").fun
    numbers["exp_gauss_overlap"] = dict(max=float(-best.fun), Lc_opt_um=float(best.x[0]),
                                        w0_over_Lc=float(w0 / best.x[0]), at_design_Lc=float(at_Lc))

    # ---- Fig. 1: layer stack (x-z), labels outside the structure
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x_a, x_b = -2.0, 9.0
    z_sub, z_box, z_core, z_top = -0.8, 0.0, 1.6, 2.0  # schematic heights (not to scale)
    ax.add_patch(Rectangle((x_a, z_sub), x_b - x_a, z_box - z_sub, color="#8c8c8c"))
    ax.add_patch(Rectangle((x_a, z_box), x_b - x_a, z_core - z_box, color="#dfe7ef"))
    ax.add_patch(Rectangle((x_a, z_core), x_b - x_a, z_top - z_core, color=style.COLOR_STEEL))
    per, x_g0 = 0.9, 1.0
    for k in range(8):  # grooves (schematic period)
        g0 = x_g0 + k * per + 0.45
        ax.add_patch(Rectangle((g0, z_top - 0.2), 0.45, 0.2, color="#eef3f7"))
    ax.add_patch(Rectangle((x_a, z_top), x_b - x_a, 1.6, color="#eef3f7", zorder=0))
    # fiber, tilted by theta from the vertical, above the grating
    t = math.radians(th)
    cx, cz = 4.6, 3.6
    hw, L = 0.9, 2.2
    ux, uz = math.sin(t), math.cos(t)
    nx, nz = math.cos(t), -math.sin(t)
    corners = [(cx - hw * nx, cz - hw * nz), (cx + hw * nx, cz + hw * nz),
               (cx + hw * nx + L * ux, cz + hw * nz + L * uz), (cx - hw * nx + L * ux, cz - hw * nz + L * uz)]
    ax.add_patch(Polygon(corners, closed=True, color="#f2d7a6"))
    # input arrow and its label sit left of the structure, aimed at the core
    ax.annotate("", xy=(x_a - 0.1, (z_core + z_top) / 2), xytext=(x_a - 2.3, (z_core + z_top) / 2),
                arrowprops=dict(arrowstyle="->", color=style.COLOR_ANNOTATION, lw=1.2))
    # labels: all to the right of the structure, with leader lines through open space
    xl = x_b + 0.4
    for zc, txt in ((4.6, f"SMF-28 fiber, MFD {p['mfd_um']} µm, tilted {th:.0f}° (in oxide)"),
                    (2.7, f"oxide top cladding, n = {p['n_top']}"),
                    (1.9, f"SiN core {p['core_um']*1000:.0f} nm, n = {p['n_core']}; grating: P = {P*1000:.0f} nm,\n"
                          f"tooth {p['widths_um'][0]*1000:.0f} nm, etch {p['etch_um']*1000:.0f} nm, {p['n_periods']} periods"),
                    (0.8, f"buried oxide (BOX) {p['box_um']} µm, n = {n_box}"),
                    (-0.4, f"Si substrate, n = {p['n_sub']}")):
        ax.text(xl, zc, txt, va="center", fontsize=8)
    ax.text(x_a - 2.3, (z_core + z_top) / 2 + 0.25, "guided\nmode in", fontsize=8,
            color=style.COLOR_ANNOTATION, va="bottom")
    ax.set_xlim(x_a - 2.5, x_b + 9.5)
    ax.set_ylim(z_sub, 6.2)
    ax.axis("off")
    ax.text(x_a, z_sub - 0.35, "Schematic, not to scale.", fontsize=7, color=style.COLOR_REFERENCE)
    fig.tight_layout()
    fig.savefig(out / "fig1_stack.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 2: phase matching -- emission angle vs wavelength (fixed Bloch index)
    summ = json.loads((META / "step1" / "summary.json").read_text())
    nB = summ["n_bloch_ref"]
    wl = np.linspace(1.28, 1.42, 200)
    ang = np.degrees(np.arcsin((nB - wl / P) / n_box))
    numbers["phase_matching"] = dict(n_B=nB, P_um=P, theta_at_wl0=float(np.degrees(np.arcsin((nB - wl0 / P) / n_box))),
                                     dtheta_dlambda_deg_per_nm=float(np.gradient(ang, wl * 1000).mean()),
                                     P_from_formula=float(wl0 / (nB - n_box * math.sin(math.radians(th)))))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(wl * 1000, ang, color=style.COLOR_STEEL)
    ax.axhline(th, color=style.COLOR_REFERENCE, ls="--", lw=0.9)
    ax.axvline(wl0 * 1000, color=style.COLOR_ANNOTATION, ls=":", lw=0.9)
    ax.plot([wl0 * 1000], [th], "o", color=style.COLOR_ROSE, zorder=5)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("emission angle θ in oxide (deg)")
    ax.set_xlim(1280, 1420)
    ax.text(1283, th + 0.25, f"fiber angle {th:.0f}°", fontsize=8, color=style.COLOR_REFERENCE, va="bottom")
    fig.tight_layout()
    fig.savefig(out / "fig2_phase_matching.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 3: field map of the conventional optimum at 1.35 um
    fm, em = conv["field_map"], conv["eps_map"]
    d = conv["dom"]
    ext = (d["x_left"], d["x_right"], d["z_sub0"], d["z_top"])
    f = np.real(fm) if np.iscomplexobj(fm) else fm
    vmax = np.percentile(np.abs(f), 99.5)
    fig, ax = plt.subplots(figsize=(9, 3.9))
    im = ax.imshow(f.T, origin="lower", extent=ext, cmap=style.CMAP_FIELD, vmin=-vmax, vmax=vmax, aspect="equal")
    ax.contour(em.T, levels=[(n_box**2 + p["n_core"] ** 2) / 2, (n_box**2 + p["n_sub"] ** 2) / 2], extent=ext,
               colors=style.COLOR_REFERENCE, linewidths=0.6)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("z (µm)")
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Re($E_y$) (arb. units)" if np.iscomplexobj(fm) else "field (arb. units)")
    fig.tight_layout()
    fig.savefig(out / "fig3_field.png", dpi=200)
    plt.close(fig)
    numbers["field_map_complex"] = bool(np.iscomplexobj(fm))

    # ---- Fig. 4: where the power goes -- conventional SiN vs a-Si overlay
    cats = ["up", "down", "T", "R"]
    labels = ["radiated up", "radiated down\n(substrate)", "transmitted\n(not radiated)", "reflected"]
    xpos = np.arange(len(cats))
    wbar = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for k, (r, name, col) in enumerate(((rc, f"SiN single-etch (λ = 1.35 µm)", style.COLOR_STEEL),
                                        (ra, f"a-Si overlay on SiN (λ = 1.31 µm)", style.COLOR_TEAL))):
        vals = [r[c] for c in cats]
        bars = ax.bar(xpos + (k - 0.5) * wbar, vals, wbar, color=col,
                      label=f"{name}: η = {r['eta']:.2f} ({r['eta_db']:.2f} dB), D = {r['directionality']:.2f}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("fraction of input power")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig4_power_budget.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 5: BOX thickness sweep (res 40)
    rows = []
    for fpath in glob.glob(str(META / "step1" / "box_res40" / "box_*.pkl")):
        t_box = float(re.search(r"box_([\d.]+)\.pkl", fpath).group(1))
        r = load(fpath)
        rows.append((t_box, r["up"], r["eta"], r["down"]))
    rows = np.array(sorted(rows))
    pred = wl0 / (2 * n_box * math.cos(math.radians(th)))
    numbers["box_sweep"] = dict(n_points=int(len(rows)), up_min=float(rows[:, 1].min()), up_max=float(rows[:, 1].max()),
                                period_predicted_um=pred, period_fitted_um=summ["box_period_fit_um"],
                                box_chosen_um=summ["box_um"])
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(rows[:, 0], rows[:, 1], "-o", ms=3, color=style.COLOR_STEEL, label="radiated up")
    ax.plot(rows[:, 0], rows[:, 2], "-s", ms=3, color=style.COLOR_TEAL, label="coupling efficiency η")
    ax.axvline(summ["box_um"], color=style.COLOR_ROSE, lw=1.2, label=f"chosen BOX {summ['box_um']} µm")
    ax.axvline(4.5, color=style.COLOR_REFERENCE, ls=":", lw=1, label="standard SiN-MPW BOX 4.5 µm")
    ax.set_xlabel("BOX thickness (µm)")
    ax.set_ylabel("fraction of input power")
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig5_box_sweep.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 6: coupling-efficiency spectrum of the conventional optimum
    o = np.argsort(conv["wl"])
    wlc, sc = conv["wl"][o] * 1000, 10 * np.log10(rc["spec"][o])
    lo, hi = band_1db(conv["wl"], rc["spec"])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.axvspan(lo * 1000, hi * 1000, color=style.COLOR_REFERENCE, alpha=0.15, lw=0,
               label=f"1-dB band ({(hi - lo) * 1000:.0f} nm)")
    ax.plot(wlc, sc, "-o", ms=3, color=style.COLOR_STEEL, label="coupling efficiency (fiber fixed)")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("coupling efficiency (dB)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig6_spectrum.png", dpi=200)
    plt.close(fig)

    print(json.dumps(numbers, indent=2, default=float))
    return numbers


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "blog_figures_out")
