"""Solver data for the blog article "Mach-Zehnder Modulators".

Re-runs the cross-section and electrode solves of tfln/notebooks/01_tfln_mzm.ipynb
(x-cut thin-film lithium niobate MZM) with the same modules and parameters,
and saves what the figures need to si/data/blog_figures/mach_zehnder_modulators.npz:
- the TE optical mode of the rib (MPB, anisotropic LN) and the electrode field
  around one gap (finite-volume electrostatics), for the overlap picture;
- the gap sweep (V_pi L, overlap factor, mode intensity at the electrode corner);
- the RF line constants (n_RF, Z0) and the conductor loss alpha(f);
- the BOX-thickness sweep of n_RF and Z0.
No FDTD time stepping happens here (MPB + electrostatics only); the numbers
reproduce notebook tfln/01's saved outputs (sections 3, 5, 6.5, 9).

Usage (from the repo root, a few minutes on one core):
    conda run -n mp python scripts/blog_figures/mach_zehnder_modulators_data.py
"""

from pathlib import Path

import numpy as np

from pic_toolkit.meep_sim import tfln_electrostatics as es
from pic_toolkit.meep_sim import tfln_mzm as tm
from pic_toolkit.meep_sim import tfln_waveguide as tw

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "si" / "data" / "blog_figures" / "mach_zehnder_modulators.npz"


def main():
    optical, electrical = tm.combined_params()
    wl0 = optical["wl0_um"]
    r33 = tm.R33_CLASSIC_M_PER_V
    slab_t = optical["film_um"] - optical["etch_um"]
    data = {"wl0_um": wl0, "slab_t_um": slab_t, "film_um": optical["film_um"],
            "rib_width_um": optical["rib_width_um"], "box_um": electrical["box_um"],
            "gap_um": electrical["gap_um"], "signal_width_um": electrical["signal_width_um"],
            "metal_um": electrical["metal_um"], "r33_pm_per_V": r33 * 1e12}

    # optical mode and electrode field (sections 3 and 5)
    mode = tw.solve_mode(optical)
    fld = es.solve_field(electrical, voltage=1.0)
    ov = tm.overlap(mode, fld, optical, electrical, arm=0, r33=r33)
    data.update(mode_x_um=mode.x_um, mode_y_um=mode.y_um,
                mode_I=np.abs(mode.Ex) ** 2 + np.abs(mode.Ey) ** 2 + np.abs(mode.Ez) ** 2,
                mode_neff=mode.n_eff, mode_ng=mode.n_g, mode_te_fraction=mode.te_fraction,
                wg_center_um=es.electrode_layout(electrical)["wg_centers"][0],
                gamma=abs(ov.gamma_rigorous), dn_dV=abs(ov.dn_dV_rigorous),
                vpi_l_V_cm=tm.vpi_l_V_cm(ov.dn_dV_rigorous, wl0))
    selx = (fld.xc_um > -20) & (fld.xc_um < 0)
    sely = (fld.yc_um > -2.5) & (fld.yc_um < 3.0)
    data.update(field_x_um=fld.xc_um[selx], field_y_um=fld.yc_um[sely],
                field_Ex=fld.Ex[np.ix_(sely, selx)], field_Ey=fld.Ey[np.ix_(sely, selx)],
                C_pF_per_m=fld.capacitance_F_per_m * 1e12)
    print(f"n_eff {mode.n_eff:.4f}, n_g {mode.n_g:.4f}, Gamma {abs(ov.gamma_rigorous):.3f}, "
          f"V_pi L {data['vpi_l_V_cm']:.2f} V cm")

    # gap sweep (section 6.5)
    rows = []
    for g in (3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
        op_g, el_g = tm.combined_params(optical, {**electrical, "gap_um": g})
        m_g = tw.solve_mode(op_g)
        f_g = es.solve_field(el_g)
        o_g = tm.overlap(m_g, f_g, op_g, el_g, arm=0, r33=r33)
        corner = tw.mode_intensity_at(m_g, g / 2, slab_t)
        rows.append((g, tm.vpi_l_V_cm(o_g.dn_dV_rigorous, wl0), abs(o_g.gamma_rigorous), corner))
        print(f"gap {g}: V_pi L {rows[-1][1]:.2f} V cm, Gamma {rows[-1][2]:.3f}, corner {corner:.1e}")
    data["gap_sweep"] = np.array(rows)   # cols: gap, V_pi L, Gamma, corner intensity / peak

    # RF line constants and conductor loss (section 9.1)
    rf = es.rf_line_constants(electrical)
    f_hz = np.linspace(0.5e9, 200e9, 400)
    loss = es.conductor_loss(electrical, f_hz)
    ng = tw.group_index_dispersive(optical)
    data.update(n_rf=rf["n_rf"], z0_ohm=rf["Z0_ohm"], C_air_pF_per_m=rf["C_air_F_per_m"] * 1e12,
                f_hz=f_hz, alpha_np_per_m=loss["alpha_np_per_m"], alpha_db_per_cm=loss["alpha_db_per_cm"],
                ng_dispersive=ng["n_g_dispersive"], sigma_metal=electrical["sigma_metal_S_per_m"])
    print(f"n_RF {rf['n_rf']:.3f}, Z0 {rf['Z0_ohm']:.1f} ohm, n_g (dispersive) {ng['n_g_dispersive']:.3f}")

    # BOX sweep (section 9.2)
    box = []
    for b in (1.0, 1.5, 2.0, 3.0, 4.7):
        r_b = es.rf_line_constants({**electrical, "box_um": b})
        box.append((b, r_b["n_rf"], r_b["Z0_ohm"]))
    data["box_sweep"] = np.array(box)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **data)
    print("saved", OUT)


if __name__ == "__main__":
    main()
