"""Figures and numbers for the blog article "Ring Resonators in Silicon
Photonics: Resonance, Coupling, and Spectral Response".

Reads only saved artifacts (si/data/design_points/{racetrack,add_drop_ring}.yaml,
si/data/sparams/racetrack/{baseline,sweep}, si/data/sparams/add_drop_ring/narrowband,
and the racetrack field snapshots cached by notebook 05 in si/data/field_cache/)
-- no FDTD is run and meep is never imported.

Unlike the notebooks' single-resonance fits (constant n_eff in the round-trip
phase), the fits here take the phase slope from the group index measured from
the free spectral range, and resolve the coupling/loss ambiguity with (i) the
S21 phase response and (ii) the conductivity-set lower bound on the loss.

Usage (from the repo root):
    conda run -n mp python scripts/blog_figures/article3_ring.py <out_dir>
"""

import glob
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pic_toolkit.slab_modes import slab_modes  # noqa: E402  (1D slab solver, meep-free)
from pic_toolkit import ring_fit as rf, style  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SP = REPO / "si" / "data" / "sparams"

style.apply_style()
assert "meep" not in sys.modules


def load(path):
    d = np.load(path)
    wl = d["wavelengths_um"]
    o = np.argsort(wl)
    return {k: d[k][o] for k in d.files if d[k].shape == wl.shape}


class _Snapshot:
    """Stand-in for meep_sim.racetrack.FieldSnapshot, so the cached pickles load without meep."""


class _SnapshotUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "FieldSnapshot":
            return _Snapshot
        return super().find_class(module, name)


def load_snapshots(name):
    """(params, [FieldSnapshot-like]) cached by notebook 05, Section 11 (regime = under/critical/over).
    The last snapshot is off resonance; the others scan the resonance."""
    with open(REPO / "si" / "data" / "field_cache" / f"racetrack_{name}_snapshots.pkl", "rb") as f:
        d = _SnapshotUnpickler(f).load()
    return d["key"]["params"], d["value"]


def racetrack_layout(p):
    """Bus position and x-extents of racetrack._compute_domain (kept meep-free here)."""
    half_length = p["coupling_length_um"] / 2 + p["radius_um"] + p["wg_width_um"] / 2
    return dict(bus_y=p["radius_um"] + p["gap_um"] / 2 + p["wg_width_um"] / 2, half_length=half_length,
                source_offset=half_length + p["margin_um"] + 1.5)


def bus_cut(snap, p):
    """x, and the field along the bus core center, divided by the incident amplitude
    (sqrt(2) x rms over the bus core between the source and the racetrack)."""
    lay = racetrack_layout(p)
    x = np.linspace(snap.extent_um[0], snap.extent_um[1], snap.field.shape[0])
    y = np.linspace(snap.extent_um[2], snap.extent_um[3], snap.field.shape[1])
    up = (x > -lay["source_offset"] + 0.2) & (x < -lay["half_length"] - 0.2)
    core = np.abs(y - lay["bus_y"]) < p["wg_width_um"] / 2
    inc = np.sqrt(2 * np.mean(np.abs(snap.field[np.ix_(up, core)]) ** 2))
    down = (x > lay["half_length"] + 0.2) & (x < lay["source_offset"])
    band = np.abs(y - lay["bus_y"]) < p["wg_width_um"] / 2 + p["gap_um"] / 2 + 0.25
    ratio = (np.mean(np.sum(np.abs(snap.field[np.ix_(down, band)]) ** 2, axis=1))
             / np.mean(np.sum(np.abs(snap.field[np.ix_(up, band)]) ** 2, axis=1)))
    return x, inc, snap.field[:, np.argmin(np.abs(y - lay["bus_y"]))] / inc, float(ratio)


def field_panel(ax, snap, p, inc, vmax):
    ax.imshow((snap.field.real / inc).T, origin="lower", extent=snap.extent_um, cmap=style.CMAP_FIELD,
              vmin=-vmax, vmax=vmax, aspect="equal")
    ax.contour(snap.eps.T, levels=[(snap.eps.min() + snap.eps.max()) / 2], extent=snap.extent_um,
               colors=style.COLOR_REFERENCE, linewidths=0.6)
    d = p["dpml_um"]
    x0, x1, y0, y1 = snap.extent_um
    for xa, xb, ya, yb in ((x0, x0 + d, y0, y1), (x1 - d, x1, y0, y1), (x0, x1, y0, y0 + d), (x0, x1, y1 - d, y1)):
        ax.fill_between([xa, xb], ya, yb, color=style.COLOR_REFERENCE, alpha=0.25, lw=0)
    ax.set_xlabel("x (µm)")


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numbers = {}
    rt = yaml.safe_load((REPO / "si" / "data" / "design_points" / "racetrack.yaml").read_text())["parameters"]
    R, gap, w = rt["radius_um"], rt["gap_um"], rt["wg_width_um"]
    ad_lc = yaml.safe_load((REPO / "si" / "data" / "design_points" / "add_drop_ring.yaml").read_text())["parameters"][
        "coupling_length_um"]

    # ---- Fig. 1: geometry -- all-pass racetrack and add-drop ring (outlines from parameters)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [rt["coupling_length_um"] + 2 * R + 19,
                                                                                  ad_lc + 2 * R + 19]})
    for ax, drop in zip(axes, (False, True)):
        Lc = ad_lc if drop else rt["coupling_length_um"]
        th = np.linspace(0, np.pi, 200)
        top = R + 0.0
        xs = np.concatenate([np.linspace(-Lc / 2, Lc / 2, 50), Lc / 2 + R * np.sin(th),
                             np.linspace(Lc / 2, -Lc / 2, 50), -Lc / 2 - R * np.sin(th)])
        ys = np.concatenate([np.full(50, top), R * np.cos(th), np.full(50, -top), -R * np.cos(th)])
        ax.plot(xs, ys, color=style.COLOR_STEEL, lw=3.2, solid_capstyle="butt")
        yb = top + w + gap
        ax.plot([-R - Lc / 2 - 3, R + Lc / 2 + 3], [yb, yb], color=style.COLOR_STEEL, lw=3.2)
        ax.text(-R - Lc / 2 - 3.3, yb, "in", ha="right", va="center", fontsize=8)
        ax.text(R + Lc / 2 + 3.3, yb, "through", ha="left", va="center", fontsize=8)
        if drop:
            ax.plot([-R - Lc / 2 - 3, R + Lc / 2 + 3], [-yb, -yb], color=style.COLOR_STEEL, lw=3.2)
            ax.text(-R - Lc / 2 - 3.3, -yb, "drop", ha="right", va="center", fontsize=8)
            ax.text(R + Lc / 2 + 3.3, -yb, "add", ha="left", va="center", fontsize=8)
        ax.annotate("", xy=(-Lc / 2, yb + 1.4), xytext=(Lc / 2, yb + 1.4),
                    arrowprops=dict(arrowstyle="<->", color="k", lw=0.7))
        ax.text(0, yb + 1.7, f"coupling section $L_c$, gap {gap*1000:.0f} nm", ha="center", va="bottom", fontsize=8)
        # radius drawn from the centre of the right semicircle (x = Lc/2) to its outer edge
        ax.annotate("", xy=(Lc / 2, 0), xytext=(Lc / 2 + R, 0), arrowprops=dict(arrowstyle="<->", color="k", lw=0.7))
        ax.plot([Lc / 2], [0], "+", color="k", ms=5)
        ax.text(Lc / 2 + R / 2 - 0.6, -0.5, f"R = {R:.0f} µm", ha="center", va="top", fontsize=8)
        ax.set_title((f"add-drop (two buses), $L_c$ = {Lc:g} µm" if drop
                      else f"all-pass (one bus), $L_c$ = {Lc:g} µm"), fontsize=10)
        ax.set_xlim(-R - Lc / 2 - 9, R + Lc / 2 + 10)
        ax.set_ylim(-yb - 3.5 if drop else -R - 2.5, yb + 4.2)
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")
    fig.tight_layout()
    fig.savefig(out / "fig1_geometry.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 3: full-band all-pass spectrum (Lc = 3 um baseline), resonances and FSR
    b = load(SP / "racetrack" / "baseline" / "racetrack_baseline.npz")
    wl, T = b["wavelengths_um"], abs(b["s21"]) ** 2
    L3 = 2 * np.pi * R + 2 * 3.0
    ng, fsr_info = rf.group_index_from_fsr(wl, T, L3)
    lam = np.array(fsr_info["resonances_um"])
    neff_straight = slab_modes(None, n_modes=1)[0][0]
    numbers["fsr"] = dict(**fsr_info, n_g=ng, L_um=L3, n_eff_straight_slab=float(neff_straight),
                          m_estimate=float(neff_straight * L3 / lam[len(lam) // 2]))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.plot(wl * 1000, T, color=style.COLOR_STEEL)
    for a_, b_ in zip(lam[:-1], lam[1:]):
        ax.annotate("", xy=(a_ * 1000, 1.04), xytext=(b_ * 1000, 1.04),
                    arrowprops=dict(arrowstyle="<->", color=style.COLOR_REFERENCE, lw=0.7))
        ax.text((a_ + b_) / 2 * 1000, 1.06, f"{(b_ - a_) * 1000:.1f}", ha="center", va="bottom", fontsize=7,
                color=style.COLOR_REFERENCE)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("through-port power $|S_{21}|^2$")
    ax.set_ylim(0, 1.15)
    ax.text(1301, 1.06, "FSR (nm):", fontsize=7, color=style.COLOR_REFERENCE, va="bottom")
    fig.tight_layout()
    fig.savefig(out / "fig3_spectrum.png", dpi=200)
    plt.close(fig)

    # ---- loss expected from the ring's conductivity (perturbation theory with the slab solver)
    alpha_abs = rf.absorption_floor(rt["ring_conductivity"], 1.31)
    numbers["absorption_floor"] = dict(alpha_per_um=alpha_abs, dB_per_cm=alpha_abs * 1e4 * 4.343)

    # ---- narrowband sweep: fit each resonance with FSR fixed by n_g; both (t, a) branches
    rows = []
    for p in sorted(glob.glob(str(SP / "racetrack" / "sweep" / "*.npz"))):
        Lc = float(re.search(r"length([\d.]+)\.npz", p).group(1))
        d = load(p)
        passed = json.loads(Path(p[:-4] + ".json").read_text())["validation"]["passed"]
        L = 2 * np.pi * R + 2 * Lc
        f = rf.fit_all_pass(d["wavelengths_um"], d["s21"], L, ng)
        regime, chosen, reason = rf.choose_branch(f, alpha_abs)
        rows.append(dict(Lc=Lc, L=L, wl_res=f["lambda_min_um"], T_min=f["T_min"], fwhm_nm=f["fwhm_nm"],
                         fwhm_formula_nm=chosen["fwhm_formula_nm"], Q_L=f["Q_L"], Q_i=chosen["Q_i"],
                         phase_excursion_rad=f["phase_swing_rad"], regime=regime, reason=reason,
                         validation_passed=passed, chosen=chosen, branches=f["branches"]))
    rows.sort(key=lambda r: r["Lc"])
    numbers["sweep"] = rows

    # ---- Fig. 4: one resonance (Lc = 4 um): power and phase, data vs both branches
    d = load(SP / "racetrack" / "sweep" / "racetrack_coupling_length4.0.npz")
    wn, s = d["wavelengths_um"], d["s21"]
    row = [r for r in rows if r["Lc"] == 4.0][0]
    nu, fsr_nu = 1 / wn, 1 / (ng * row["L"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.9))
    a1.plot(wn * 1000, abs(s) ** 2, "o", ms=2.5, color=style.COLOR_STEEL, label="FDTD")
    a2.plot(wn * 1000, rf.detrended_phase(wn, s), "o", ms=2.5, color=style.COLOR_STEEL, label="FDTD")
    for name, ls, col in (("under", "-", style.COLOR_REFERENCE), ("over", "--", style.COLOR_ROSE)):
        br = row["branches"][name]
        m = rf.all_pass(nu, br["t"], br["a"], 1 / br["lambda_res_um"], fsr_nu, br["scale"])
        lab = (f"under-coupled fit: t = {br['t']:.3f} > a = {br['a']:.3f}" if name == "under"
               else f"over-coupled fit: t = {br['t']:.3f} < a = {br['a']:.3f}")
        a1.plot(wn * 1000, abs(m) ** 2, ls, color=col, lw=1.2, label=lab)
        a2.plot(wn * 1000, rf.detrended_phase(wn, m), ls, color=col, lw=1.2, label=lab)
    a1.set_xlim(1305.5, 1311.5)
    a2.set_xlim(1305.5, 1311.5)
    a1.set_xlabel("wavelength (nm)")
    a2.set_xlabel("wavelength (nm)")
    a1.set_ylabel("through-port power")
    a2.set_ylabel("phase of $S_{21}$, detrended (rad)")
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "fig4_resonance_fit.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 5: coupling regimes -- on-resonance transmission vs kappa^2 (chosen branch per point)
    good = [r for r in rows if r["validation_passed"]]
    a_lo = float(min(r["chosen"]["a"] for r in good))
    a_hi = float(max(r["chosen"]["a"] for r in good))
    k2 = np.linspace(0, 0.4, 400)
    t = np.sqrt(1 - k2)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for a_edge, ls in ((a_hi, "-"), (a_lo, "--")):
        ax.plot(k2, ((t - a_edge) / (1 - t * a_edge)) ** 2, ls, color=style.COLOR_REFERENCE, lw=1,
                label=f"all-pass theory, a = {a_edge:.3f}")
    # label offsets (points) where neighbouring markers would otherwise collide
    label_offset = {3.5: (-4, -14), 4.0: (7, -3), 9.0: (6, 5), 10.0: (-17, 6), 11.0: (4, 6)}
    regime_col = {"under-coupled": style.COLOR_STEEL, "critically coupled": style.COLOR_ROSE,
                  "over-coupled": style.COLOR_CYCLE[1]}
    for r in rows:
        c = r["chosen"]
        star = r["Lc"] == rt["coupling_length_um"]
        col = regime_col[c["regime"]]
        ax.plot(c["kappa2"], r["T_min"], "*" if star else "o", ms=12 if star else 6, color=col,
                markerfacecolor=col if r["validation_passed"] else "none")
        ax.annotate(f"{r['Lc']:g}", (c["kappa2"], r["T_min"]), textcoords="offset points",
                    xytext=label_offset.get(r["Lc"], (7, 4)), fontsize=7, color="0.3")
    ax.plot([], [], "o", color=regime_col["under-coupled"], label="FDTD, under-coupled (label: $L_c$ in µm)")
    ax.plot([], [], "o", color=regime_col["critically coupled"], label="FDTD, critically coupled ($|t-a|$ < 0.005)")
    ax.plot([], [], "o", color=regime_col["over-coupled"], label="FDTD, over-coupled")
    ax.plot([], [], "*", ms=10, color=style.COLOR_ROSE, label=f"design point, $L_c$ = {rt['coupling_length_um']:g} µm")
    ax.plot([], [], "o", color="0.5", markerfacecolor="none", label="failed a validation check")
    ax.set_xlabel(r"power cross-coupling $\kappa^2 = 1 - t^2$")
    ax.set_ylabel("on-resonance through power $T_{min}$")
    ax.set_xlim(0, 0.4)
    ax.set_ylim(-0.03, 1)
    ax.text(0.03, 0.88, "under-coupled (t > a)", fontsize=8)
    ax.text(0.39, 0.88, "over-coupled (t < a)", fontsize=8, ha="right")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig5_regimes.png", dpi=200)
    plt.close(fig)
    numbers["regimes"] = dict(a_min=a_lo, a_max=a_hi, kappa2_critical_range=[1 - a_hi**2, 1 - a_lo**2])

    # ---- Fig. 2 and Fig. 6: Re(Hz) field snapshots cached by notebook 05, Section 11.
    # On resonance = the scanned wavelength with the lowest bus transmission in the field itself.
    regimes = []
    for name, label in (("under", "under-coupled"), ("critical", "critically coupled")):
        p, snaps = load_snapshots(name)
        cuts = [bus_cut(sn, p) for sn in snaps]
        k = int(np.argmin([c_[3] for c_ in cuts[:-1]]))
        row = [r for r in rows if r["Lc"] == p["coupling_length_um"]][0]
        regimes.append(dict(name=name, label=label, p=p, on=snaps[k], off=snaps[-1], cut_on=cuts[k],
                            cut_off=cuts[-1], row=row))
    numbers["fields"] = [dict(regime=g["name"], Lc=g["p"]["coupling_length_um"], wl_on=g["on"].wavelength_um,
                              wl_off=g["off"].wavelength_um, bus_ratio_on=g["cut_on"][3],
                              bus_ratio_off=g["cut_off"][3], T_min_sweep=g["row"]["T_min"],
                              ring_peak_over_incident=float(np.max(np.abs(g["on"].field)) / g["cut_on"][1]))
                         for g in regimes]

    # Fig. 2: the critically coupled ring on and off resonance (one color scale).
    crit = regimes[1]
    vmax = max(np.max(np.abs(crit[k_].field.real)) / crit[c_][1] for k_, c_ in (("on", "cut_on"), ("off", "cut_off")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, k_, c_, title in ((axes[0], "on", "cut_on", "on resonance"), (axes[1], "off", "cut_off", "off resonance")):
        field_panel(ax, crit[k_], crit["p"], crit[c_][1], vmax)
        ax.set_title(f"{title}, {crit[k_].wavelength_um * 1000:.1f} nm", fontsize=10)
    axes[0].set_ylabel("y (µm)")
    fig.tight_layout()
    fig.savefig(out / "fig2_field.png", dpi=200)
    plt.close(fig)

    # Fig. 6: on resonance, under-coupled vs critically coupled (shared scale) + the field along the bus.
    vmax = max(np.max(np.abs(g["on"].field.real)) / g["cut_on"][1] for g in regimes)
    widths = [g["on"].extent_um[1] - g["on"].extent_um[0] for g in regimes]
    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=widths, height_ratios=[1, 0.85])
    line_col = (style.COLOR_STEEL, style.COLOR_ROSE)
    ax_line = fig.add_subplot(gs[1, :])
    for j, (g, col) in enumerate(zip(regimes, line_col)):
        ax = fig.add_subplot(gs[0, j])
        field_panel(ax, g["on"], g["p"], g["cut_on"][1], vmax)
        c = g["row"]["chosen"]
        ax.set_title(f"{g['label']}\n$L_c$ = {g['p']['coupling_length_um']:g} µm, $T_{{min}}$ = {g['row']['T_min']:.2g}",
                     fontsize=9)
        if j == 0:
            ax.set_ylabel("y (µm)")
        x, _, cut, _ = g["cut_on"]
        keep = np.abs(x) < racetrack_layout(g["p"])["source_offset"]
        ax_line.plot(x[keep] - racetrack_layout(g["p"])["half_length"], np.abs(cut[keep]), color=col, lw=1.1,
                     label=f"{g['label']}, $L_c$ = {g['p']['coupling_length_um']:g} µm")
        ax_line.axhline(np.sqrt(g["row"]["T_min"]), color=col, ls=":", lw=0.9)
    ax_line.set_xlabel("x (µm), measured from the end of the racetrack")
    ax_line.set_ylabel("|Hz| / incident,\nalong the bus")
    ax_line.set_ylim(0, 1.6)
    ax_line.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig6_regime_fields.png", dpi=200)
    plt.close(fig)

    # ---- Fig. 7: add-drop ring (Lc = 4 um, narrowband): through and drop, joint 4-port fit
    ad = load(SP / "add_drop_ring" / "narrowband" / "add_drop_ring_couplinglength4.0_narrowband.npz")
    wa = ad["wavelengths_um"]
    Tt, Td = abs(ad["S_input_through"]) ** 2, abs(ad["S_input_drop"]) ** 2
    La = 2 * np.pi * R + 2 * 4.0
    fit = rf.fit_add_drop(wa, ad["S_input_through"], ad["S_input_drop"], ad["S_add_drop"], ad["S_add_through"], La, ng)
    numbers["add_drop"] = fit
    m = rf.add_drop(1 / wa, fit["t1"], fit["t2"], fit["a"], 1 / fit["lambda_res_um"], 1 / (ng * La))
    thr, drp = abs(m["through"]) ** 2, abs(m["drop"]) ** 2
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(wa * 1000, Tt, "o", ms=2.5, color=style.COLOR_INDIGO, label="FDTD, through")
    ax.plot(wa * 1000, Td, "s", ms=2.5, color=style.COLOR_TEAL, label="FDTD, drop")
    ax.plot(wa * 1000, thr, color=style.COLOR_REFERENCE, lw=1, label="add-drop model (joint fit)")
    ax.plot(wa * 1000, drp, color=style.COLOR_REFERENCE, lw=1)
    ax.set_xlim(1351.5, 1355.5)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("power fraction")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig7_add_drop.png", dpi=200)
    plt.close(fig)

    print(json.dumps(numbers, indent=1, default=float))
    return numbers


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "blog_figures_out")
