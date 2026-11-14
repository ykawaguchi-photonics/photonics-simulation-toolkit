"""Solver data for the blog article "Thermo-Optic Phase Shifters".

Re-runs the solves of si_active/notebooks/01_thermo_optic_phase_shifter.ipynb with the
same modules and parameters (2D finite-volume heat conduction + MPB cross-section mode,
400 nm x 220 nm Si strip / rib at 1.31 um), and adds two studies used by the article:
- a BOX-thickness sweep (1, 2, 3 um), to compare the power / speed / crosstalk trade-off
  with the one-layer slab model of the article;
- the lateral temperature profile at the waveguide height, for the crosstalk decay length.

Saves si_active/data/blog_figures/thermo_optic_phase_shifters.npz. No FDTD.

Usage (from the repo root, tens of minutes on one core):
    conda run -n mp python scripts/blog_figures/thermo_optic_phase_shifters_data.py
"""

from pathlib import Path

import numpy as np

from pic_toolkit.active import si_xsection as xs
from pic_toolkit.active import thermal as th
from pic_toolkit.active import thermo_optic as to

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "si_active" / "data" / "blog_figures" / "thermo_optic_phase_shifters.npz"

WL0 = 1.31
DNDT_SI = xs.si_dndT(WL0)                 # Frey 2006, 1.936e-4 1/K
WG = 0.40
STACK = dict(th.DEFAULT_STACK)
TRENCH_D3 = [(3.0, 4.5), (-4.5, -3.0)]
FREQS = np.geomspace(1e2, 1e7, 101)
RECT_CODE = {"substrate": 0, "core": 1, "slab": 2, "heater": 3, "heater_r": 3, "heater_l": 3, "trench": 4}


def doped_loss_db_per_cm(mode, wg_width_um, buffer_um, kappa, wl_um, doped_width=1.0, slab=0.09):
    """Modal loss from N++ strips (imaginary index kappa) in the slab, by the same
    fixed-frequency perturbation formula as the phase (notebook 01, section 6)."""
    X, Y = np.meshgrid(mode.x_um, mode.y_um)
    half = wg_width_um / 2
    dop = (np.abs(X) >= half + buffer_um) & (np.abs(X) <= half + buffer_um + doped_width) & (Y >= 0) & (Y <= slab)
    eps = np.where(mode.is_si, mode.n_si ** 2, mode.n_ox ** 2)
    den = np.sum(eps * mode.E2)
    im_dneff = mode.n_g * np.sum((mode.n_si * kappa * mode.E2)[dop]) / den
    return 4.343 * 2 * (2 * np.pi / (wl_um * 1e-4)) * im_dneff


def rect_table(prob):
    return np.array([[r.x0, r.x1, r.y0, r.y1, RECT_CODE.get(r.name, 9)] for r in prob.rects])


def lateral_profile(wide_stack):
    """Temperature at the Si-layer height vs lateral distance, TiN heater with NO neighbour
    waveguides (their higher conductivity would put kinks in the profile)."""
    prob = th.tin_heater_strip(wg_width_um=WG, stack=wide_stack)
    T = th.solve_steady(prob, 1.0)
    xp = np.linspace(0, 30, 301)
    return {"profile_x_um": xp,
            "profile_T": th.interpolator(prob, T)(np.column_stack([np.full_like(xp, 0.11), xp]))}


def main():
    data = {"wl0_um": WL0, "dndT_si": DNDT_SI, "wg_width_um": WG,
            **{f"stack_{k}": v for k, v in STACK.items()}}

    strip = xs.solve_mode({"wl0_um": WL0, "wg_width_um": WG})
    p_rib = {"wl0_um": WL0, "wg_width_um": WG, "slab_um": 0.09, "slab_half_width_um": 3.0}
    rib = xs.solve_mode(p_rib)
    for name, m in (("strip", strip), ("rib", rib)):
        data.update({f"{name}_x": m.x_um, f"{name}_y": m.y_um, f"{name}_E2": m.E2 / m.E2.max(),
                     f"{name}_n_eff": m.n_eff, f"{name}_n_g": m.n_g})
    print(f"strip n_eff {strip.n_eff:.4f} n_g {strip.n_g:.4f}; rib n_eff {rib.n_eff:.4f} n_g {rib.n_g:.4f}")

    # Three design variants (design point of notebook 01), with the small-signal frequency response.
    variants = {"tin": (th.tin_heater_strip(wg_width_um=WG, stack=STACK), strip),
                "tin_trench": (th.tin_heater_strip(wg_width_um=WG, stack=STACK, trenches=TRENCH_D3), strip),
                "n_doped": (th.doped_si_heater_rib(wg_width_um=WG, buffer_um=0.8, stack=STACK), rib)}
    for key, (prob, mode) in variants.items():
        r = to.evaluate(prob, mode, WL0, DNDT_SI, frequency=True, freqs_hz=FREQS)
        d_point = DNDT_SI * to.point_reader(prob, 0.11)(r.T)
        data.update({f"{key}_x": prob.x, f"{key}_y": prob.y, f"{key}_T": r.T, f"{key}_rects": rect_table(prob),
                     f"{key}_p_pi_mW": r.p_pi_mW, f"{key}_tau_us": r.tau_us, f"{key}_f3db_kHz": r.f3db_kHz,
                     f"{key}_t": r.times_s, f"{key}_step": r.step, f"{key}_f": r.freqs_hz, f"{key}_H": r.H,
                     f"{key}_dneff_per_W_per_m": r.dneff_per_W_per_m,
                     f"{key}_p_pi_point_mW": WL0 * 1e-6 / (2 * d_point) * 1e3})
        print(f"{key:10s}: P_pi {r.p_pi_mW:6.2f} mW (point metric {data[key + '_p_pi_point_mW']:.2f}), "
              f"tau {r.tau_us:5.2f} us, f3dB {r.f3db_kHz:5.1f} kHz, 1/(2 pi tau) {1e3 / (2 * np.pi * r.tau_us):5.1f} kHz")

    # Trench distance sweep (TiN heater).
    trench_d = np.array([2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0])
    tr = [to.evaluate(th.tin_heater_strip(wg_width_um=WG, stack=STACK, trenches=[(d, d + 1.5), (-d - 1.5, -d)]),
                      strip, WL0, DNDT_SI, steps_per_decade=60) for d in trench_d]
    data.update({"trench_d_um": trench_d, "trench_p_pi_mW": np.array([r.p_pi_mW for r in tr]),
                 "trench_tau_us": np.array([r.tau_us for r in tr])})
    for d, r in zip(trench_d, tr):
        print(f"trench d = {d:3.1f} um: P_pi {r.p_pi_mW:5.2f} mW, tau {r.tau_us:5.2f} us, P_pi tau {r.p_pi_tau:5.0f}")

    # N++ buffer sweep: P_pi, tau, loss (Drude lambda^2 scaling of Jacques' kappa at 1.55 um: an estimate).
    alpha_131 = 4 * np.pi * 0.137 / 1.55e-4 * (WL0 / 1.55) ** 2           # 1/cm
    kappa_131 = alpha_131 * WL0 * 1e-4 / (4 * np.pi)
    buffers = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2])
    bu = [to.evaluate(th.doped_si_heater_rib(wg_width_um=WG, buffer_um=b, stack=STACK), rib, WL0, DNDT_SI,
                      steps_per_decade=60) for b in buffers]
    loss = np.array([doped_loss_db_per_cm(rib, WG, b, kappa_131, WL0) for b in buffers])
    data.update({"buffer_um": buffers, "buffer_p_pi_mW": np.array([r.p_pi_mW for r in bu]),
                 "buffer_tau_us": np.array([r.tau_us for r in bu]), "buffer_loss_db_per_cm": loss})
    for b, r, l in zip(buffers, bu, loss):
        print(f"N++ b = {b:3.1f} um: P_pi {r.p_pi_mW:5.2f} mW, tau {r.tau_us:4.2f} us, loss {l:.3g} dB/cm")

    # Crosstalk vs pitch (wide domain), without and with trenches; lateral temperature profile.
    pitches = np.array([3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0])
    wide = {**STACK, "half_width_um": 150}
    prob = th.tin_heater_strip(wg_width_um=WG, neighbors_x_um=pitches, stack=wide)
    T_wide = th.solve_steady(prob, 1.0)
    data["xt_pitch_um"] = pitches
    data["xt_none"] = to.crosstalk(prob, strip, DNDT_SI, pitches, T=T_wide)
    data.update(lateral_profile(wide))
    p_tr = pitches[pitches >= 5]
    prob_t = th.tin_heater_strip(wg_width_um=WG, neighbors_x_um=p_tr, stack=wide, trenches=TRENCH_D3)
    data["xt_trench_pitch_um"] = p_tr
    data["xt_trench"] = to.crosstalk(prob_t, strip, DNDT_SI, p_tr)
    print("crosstalk, no trench:", np.round(100 * data["xt_none"], 2))
    print("crosstalk, trench   :", np.round(100 * data["xt_trench"], 2))

    # BOX thickness sweep (TiN heater 2 um above the Si; top oxide 2.28 um above the Si in every case).
    boxes = np.array([1.0, 2.0, 3.0])
    bp, bt, bx5, bx10 = [], [], [], []
    for b in boxes:
        s = {**STACK, "box_um": b}
        r = to.evaluate(th.tin_heater_strip(wg_width_um=WG, stack=s), strip, WL0, DNDT_SI, steps_per_decade=60)
        pw = th.tin_heater_strip(wg_width_um=WG, neighbors_x_um=[5.0, 10.0], stack={**s, "half_width_um": 150})
        x5, x10 = to.crosstalk(pw, strip, DNDT_SI, [5.0, 10.0])
        bp.append(r.p_pi_mW); bt.append(r.tau_us); bx5.append(x5); bx10.append(x10)
        print(f"BOX {b:.1f} um: P_pi {r.p_pi_mW:5.2f} mW, tau {r.tau_us:5.2f} us, crosstalk 5 um {100 * x5:5.2f} %, "
              f"10 um {100 * x10:5.2f} %")
    data.update({"box_um": boxes, "box_p_pi_mW": np.array(bp), "box_tau_us": np.array(bt),
                 "box_xt5": np.array(bx5), "box_xt10": np.array(bx10)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **data)
    print("saved", OUT.relative_to(REPO))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--profile-only"]:     # refresh only the lateral profile in an existing file
        d = dict(np.load(OUT))
        d.update(lateral_profile({**STACK, "half_width_um": 150}))
        np.savez_compressed(OUT, **d)
        print("updated profile in", OUT.relative_to(REPO))
    else:
        main()
