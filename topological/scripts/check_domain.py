"""Single-frequency domain check for the strip -> valley-Hall wall transition (notebook 03).

Before the length-scale-constrained re-optimization, the domain rules of CLAUDE.md §7 are
checked on one model: the optimized valley-Hall transition v2_valley_gm36_s1r (strongly
scattering junction), gap-matched crystal, 36/a, at f = 0.237 c/a (lambda0 = 4.22 a = 1550 nm;
in Si lambda0/n = 1.23 a, lambda0/2 = 2.11 a). One setting is changed at a time from the
optimization model of scripts/topopt_transition_v2.py:
  * objective plane -> right PML distance and right PML thickness (crystal into the PML);
  * source -> left PML and source -> crystal face (strip in air cladding);
  * y boundary: Si pad + PML, the crystal ending at a PML in Si (no pad), or the crystal running
    into the PML; rows per side.

For every configuration: forward flux over the full height at the objective plane and at a
plane 2 a before it (T, T_up; equal if the plane is clean), the flux returning through the
input plane (back, incident subtracted), the flux through the top / bottom lines just inside
the y boundary (bulk), the balance T + back + bulk - 1, the field at the domain edge
(the largest |Hz|^2 in the last lattice constant of the objective line / its maximum), and the field energy at the end of
the run relative to its peak (a PML that amplifies an evanescent tail does not decay).

Usage (from topological/, mp-mpi env): mpirun -np 12 python scripts/check_domain.py [NAME ...]
Writes data/coupling/check_domain.json (one entry per configuration, cached).
"""
import json
import sys
from pathlib import Path

import meep as mp
import numpy as np

from pic_toolkit.meep_sim import topological_pc as tp

mp.verbosity(0)
ROOT = Path(__file__).resolve().parents[1] / "data"
OUT = ROOT / "coupling" / "check_domain.json"
F0, RES = 0.237, 36
LAM = 1 / F0  # lambda0 in units of a

# the optimization model of topopt_transition_v2.py (lattice units a)
BASE = dict(dpml_l=1.5, src_pml=0.5, src_face=1.5, x_t=9.0, obj_pml=1.0, dpml_r=2.5,
            pad_y=0.5, dpml_y=1.5, rows=12, crystal_in_pml=False)
H = LAM / 2  # lambda0 / 2 = 2.11 a
CONFIGS = {
    "base": {},
    # right side: objective plane -> PML, PML thickness
    "obj_pml_half": dict(obj_pml=H),
    "obj_pml_1": dict(obj_pml=LAM),
    "obj_pml_half_dpml_1": dict(obj_pml=H, dpml_r=LAM),
    # left side: source -> PML, source -> face (right side at lambda0/2)
    "src_pml_half": dict(obj_pml=H, src_pml=H),
    "src_pml_1": dict(obj_pml=H, src_pml=LAM),
    "src_pml_half_face_1": dict(obj_pml=H, src_pml=H, src_face=LAM),
    # y boundary (right side at lambda0/2). The crystal running into the y PML diverged in a
    # low-resolution smoke test (8/a, 6 rows: fields ~1e95 at t = 4000; the in-gap evanescent
    # Bloch waves grow in the PML); one run checks it at 36/a. No pad: the crystal ends at the
    # PML, which lies in plain Si.
    "y_crystal_into_pml": dict(obj_pml=H, pad_y=0.0, dpml_y=H, crystal_in_pml=True),
    "y_nopad": dict(obj_pml=H, pad_y=0.0, dpml_y=H),
    "y_nopad_r20": dict(obj_pml=H, pad_y=0.0, dpml_y=H, rows=20),
    "y_pad_r20": dict(obj_pml=H, rows=20),
    # all rules: PML lambda0/2 thick and lambda0/2 from every monitor, source lambda0 from the face
    # and lambda0/2 from its PML, no y pad (crystal ends at the Si PML), 20 rows (4.1 lambda0 per side)
    "rules_r20": dict(obj_pml=H, dpml_r=H, dpml_l=H, src_pml=H, src_face=LAM, pad_y=0.0, dpml_y=H, rows=20),
    "rules_r20_dpml_1": dict(obj_pml=H, dpml_r=LAM, dpml_l=LAM, src_pml=H, src_face=LAM, pad_y=0.0,
                             dpml_y=LAM, rows=20),
    # the layout of run_coupling.py / tp.run_wg_coupling (gm_valley_*_r36n20): source 0.5 a from a 3 a
    # PML, incident / back plane 1 a after the source, Si pad 1.5 a + 3 a PML, right PML 4 a from x = 25
    "coupling_model": dict(dpml_l=3.0, src_pml=0.5, src_face=3.5, x_inc=-2.5, obj_pml=16.0, dpml_r=4.0,
                           pad_y=1.5, dpml_y=3.0, rows=20),
    "rules_r20_mode": dict(obj_pml=H, dpml_r=H, dpml_l=H, src_pml=H, src_face=LAM, pad_y=0.0, dpml_y=H, rows=20),
    # right end at 20 rows (rules otherwise): T at the fixed x_t against the objective -> PML distance
    # (coupling_model, PML at x = 25: 0.855; rules, x_t + lambda0/2: 0.912) and the PML thickness
    **{f"xr_{k}": dict(obj_pml=d, dpml_r=t, dpml_l=H, src_pml=H, src_face=LAM, pad_y=0.0, dpml_y=H, rows=20)
       for k, d, t in (("d1.5h", 1.5 * H, H), ("d2h", 2 * H, H), ("d3h", 3 * H, H), ("d16", 16.0, H),
                       ("h_t4h", H, 4 * H), ("d16_t4h", 16.0, 4 * H))},
    "rules_r30": dict(obj_pml=H, dpml_r=H, dpml_l=H, src_pml=H, src_face=LAM, pad_y=0.0, dpml_y=H, rows=30),
}


def run(cfg: dict, design: dict) -> dict:
    p = {**BASE, **cfg}
    w = tp.strip_width("valley_te")
    x_src = -p["src_face"]
    x_left = x_src - p["src_pml"] - p["dpml_l"]
    x_inc = p.get("x_inc", x_src / 2)  # incident / back plane: default halfway between the source and the face
    x_pml_r = p["x_t"] + p["obj_pml"]
    x_right = x_pml_r + p["dpml_r"]
    y_cr = p["rows"] * tp.ROW_H
    sy = 2 * (y_cr + p["pad_y"] + p["dpml_y"])
    sx, cx = x_right - x_left, (x_right + x_left) / 2
    y_in = y_cr + p["pad_y"] - 0.05  # top / bottom lines just inside the y boundary
    n_geom = p["rows"] + (int(np.ceil(p["dpml_y"] / tp.ROW_H)) + 1 if p["crystal_in_pml"] else 0)
    bounds = [mp.PML(p["dpml_y"], direction=mp.Y), mp.PML(p["dpml_l"], direction=mp.X, side=mp.Low),
              mp.PML(p["dpml_r"], direction=mp.X, side=mp.High)]
    src = [mp.EigenModeSource(mp.GaussianSource(F0, fwidth=0.03), center=mp.Vector3(x_src, 0),
                              size=mp.Vector3(0, sy - 2 * p["dpml_y"]), eig_band=1, eig_parity=mp.TE)]
    line = lambda x: mp.FluxRegion(center=mp.Vector3(x, 0), size=mp.Vector3(0, 2 * y_in))
    kw = dict(cell_size=mp.Vector3(sx, sy), geometry_center=mp.Vector3(cx, 0), sources=src,
              boundary_layers=bounds, resolution=RES)

    ref = mp.Simulation(geometry=[mp.Block(size=mp.Vector3(mp.inf, w), material=tp.HOST)], **kw)
    f_inc = ref.add_flux(F0, 0, 1, line(x_inc))
    m_inc = ref.add_mode_monitor(F0, 0, 1, mp.ModeRegion(center=mp.Vector3(x_inc, 0), size=mp.Vector3(0, 2 * y_in)))
    ref.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-9, minimum_run_time=50))
    p_inc = mp.get_fluxes(f_inc)[0]
    # the mode power (run_wg_coupling's normalization) next to the flux of the straight strip
    p_mode = abs(ref.get_eigenmode_coefficients(m_inc, [1], eig_parity=mp.TE).alpha[0, 0, 0]) ** 2
    inc_data = ref.get_flux_data(f_inc)
    ref.reset_meep()

    geom = tp.coupler_geometry("valley_te", w, x_left, x_right, n_geom)
    mg = mp.MaterialGrid(mp.Vector3(*design["weights"].shape), mp.air, tp.HOST, weights=design["weights"],
                         do_averaging=True, beta=np.inf)
    geom.append(mp.Block(center=mp.Vector3(*design["center"]), size=mp.Vector3(*design["size"]), material=mg))
    sim = mp.Simulation(geometry=geom, **kw)
    back = sim.add_flux(F0, 0, 1, line(x_inc))
    sim.load_minus_flux_data(back, inc_data)
    t_obj = sim.add_flux(F0, 0, 1, line(p["x_t"]))
    t_up = sim.add_flux(F0, 0, 1, line(p["x_t"] - 2.0))
    span, mid = mp.Vector3(p["x_t"] - x_inc, 0), (p["x_t"] + x_inc) / 2
    top = sim.add_flux(F0, 0, 1, mp.FluxRegion(center=mp.Vector3(mid, y_in), size=span))
    bot = sim.add_flux(F0, 0, 1, mp.FluxRegion(center=mp.Vector3(mid, -y_in), size=span, weight=-1))
    edge = sim.add_dft_fields([mp.Hz], F0, 0, 1, center=mp.Vector3(p["x_t"], 0),
                              size=mp.Vector3(0, sy - 2 * p["dpml_y"]))
    box = mp.Volume(center=mp.Vector3((x_left + p["dpml_l"] + x_pml_r) / 2, 0),
                    size=mp.Vector3(x_pml_r - x_left - p["dpml_l"], 2 * (y_cr + p["pad_y"])))
    energy = []
    sim.run(mp.at_every(25, lambda s: energy.append(s.field_energy_in_box(box=box))),
            until_after_sources=mp.stop_when_dft_decayed(tol=1e-9, minimum_run_time=400, maximum_run_time=4000))
    hz = np.abs(np.squeeze(sim.get_dft_array(edge, mp.Hz, 0))) ** 2
    T, Tu = mp.get_fluxes(t_obj)[0] / p_inc, mp.get_fluxes(t_up)[0] / p_inc
    bk = -mp.get_fluxes(back)[0] / p_inc
    bulk = (mp.get_fluxes(top)[0] + mp.get_fluxes(bot)[0]) / p_inc
    e = np.asarray(energy)
    return {**p, "T": T, "T_up": Tu, "back": bk, "bulk": bulk, "balance": T + bk + bulk - 1,
            "edge_field": float(hz[-RES:].max() / hz.max()), "energy_end_over_peak": float(e[-1] / e.max()),
            "t_end": sim.meep_time(), "p_inc_over_mode": p_inc / p_mode, "cell_um2": sx * sy * (tp.A_NM / 1000) ** 2}


if __name__ == "__main__":
    tp.set_gap_matched()
    d = np.load(ROOT / "topopt" / "v2_valley_gm36_s1r_design.npz")
    design = {"weights": np.clip(d["rho"], 0, 1), "center": tuple(d["center"]), "size": tuple(d["size"])}
    r = json.loads(OUT.read_text()) if OUT.exists() else {}
    # the diverging case runs to maximum_run_time: last
    for name in sys.argv[1:] or sorted(CONFIGS, key=lambda n: n == "y_crystal_into_pml"):
        if name in r:
            continue
        o = run(CONFIGS[name], design)
        if mp.am_master():
            print(f"{name}: T {o['T']:.4f} (2 a before {o['T_up']:.4f}), back {o['back']:.4f}, bulk {o['bulk']:.4f}, "
                  f"balance {o['balance']:+.4f}, edge |Hz|^2 {o['edge_field']:.1e}, "
                  f"energy end/peak {o['energy_end_over_peak']:.1e}, t {o['t_end']:.0f}, "
                  f"flux / mode power {o['p_inc_over_mode']:.4f}", flush=True)
            r[name] = o
            OUT.write_text(json.dumps(r, indent=1))
