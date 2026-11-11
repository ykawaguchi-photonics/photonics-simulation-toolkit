"""Strip waveguide -> TE topological wall runs for notebook 03, cached in data/coupling/.

Cases:
  strip            straight 500 nm strip (spin-Hall units, w = 0.79 a0)
  spin_butt        strip butt-coupled to the spin-Hall TE wall
  spin_topopt      ... with the optimized transition (data/topopt/spin_design.npz)
  spin_topopt_vp   ... with the transition optimized by the valley procedure
  valley_butt      strip butt-coupled to the bearded valley TE wall
  valley_ld6       line-defect coupler, L = 6 (Chen 2022 / Yoshimi 2023)
  valley_odd       butt coupling, first odd strip mode as input
  valley_topopt    ... with the optimized transition (data/topopt/valley_design.npz)
  spin_topopt2, valley_topopt2   gap-matched re-optimization (scripts/topopt_transition_v2.py,
                   data/topopt/v2_spin_gm32_s1 / v2_valley_gm36_s1r): run as gm_..._rRESnROWS;
                   ..._topopt2c: the same with every feature below 100 nm removed;
                   valley_topopt2c80: below 80 nm removed;
  valley_topopt3   re-optimized with 80 nm solid/void length-scale constraints (v2_valley_gm36_ls80);
                   valley_topopt3c80: the same after the 80 nm cleaning (check that it is unchanged)
  valley_topopt4   80 nm by the robust formulation, domain in wavelengths: coarse 24/a run from
                   gray + noise (v2_valley_c24_rob80_s1), refined at 36/a (v2_valley_gm36_rob80);
                   valley_topopt4c80: the same after the 80 nm cleaning
Field maps are stored at every second grid point.

Convergence variants: CASE_rRESnROWS (e.g. spin_topopt_r64n20) runs CASE at that resolution
and number of rows per side, cached in data/coupling/conv/.
Gap-matched variants: gm_CASE_rRESnROWS (e.g. gm_spin_butt_r48n10) run the gap-matched pair
(tp.set_gap_matched(), both shared TE gaps 4.3 %), cached in data/coupling/gm/; optional
y-offset suffix _yOFFSET (e.g. gm_spin_butt_r48n10_y-0.3).

Usage (mp env, from topological/):  python scripts/run_coupling.py CASE [CASE ...]
  under MPI: mpirun -np N python scripts/run_coupling.py CASE ...  (mp-mpi env)
"""
import re
import sys
from pathlib import Path

import meep as mp
import numpy as np

from pic_toolkit.meep_sim import topological_pc as tp

mp.verbosity(0)
ROOT = Path(__file__).resolve().parents[1] / "data"
DATA = ROOT / "coupling"


def design(name: str) -> dict:
    d = np.load(ROOT / "topopt" / f"{name}.npz")
    return {"weights": d["weights"], "center": tuple(d["center"]), "size": tuple(d["size"])}


def design_v2(name: str, tag: str, clean_nm: float | None = None) -> dict:
    """The v2 design as its filtered density: inserted at beta = inf it is the binary level set
    rho = 0.5 with subpixel boundary, the structure the optimizer's beta = inf iterations saw.
    clean_nm: features below clean_nm removed (clean_100nm), the strictly manufacturable variant."""
    import meep.adjoint as mpa
    d = np.load(ROOT / "topopt" / f"v2_{name}_{tag}_design.npz")
    if "rho" in d.files:
        rho = d["rho"]
    else:
        L, res = d["size"], int(d["res"])
        n = [int(round(s * res)) + 1 for s in L]
        rho = np.asarray(mpa.conic_filter(d["x"].reshape(*n), float(d["filter_radius"]), L[0], L[1], res))
    w = np.clip(rho, 0, 1)
    if clean_nm:
        w = clean_100nm(w, d["size"][0] * (tp.A0_NM if name == "spin" else tp.A_NM), min_nm=clean_nm)
    return {"weights": w, "center": tuple(d["center"]), "size": tuple(d["size"])}


def clean_100nm(rho: np.ndarray, side_nm: float, up: int = 4, min_nm: float = 100.0) -> np.ndarray:
    """Binary level set of rho on a `up`-times finer grid with every feature below min_nm removed:
    morphological opening (Si features) then closing (gaps) with a min_nm disk. The grid is
    padded by edge replication so the region border (strip / crystal outside) is not eroded."""
    from scipy import ndimage
    b = ndimage.zoom(rho, up, order=1) > 0.5
    rad = min_nm / 2 / (side_nm / b.shape[0])
    k = int(rad) + 1
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    disk = xx ** 2 + yy ** 2 <= rad ** 2
    bp = np.pad(b, 2 * k, mode="edge")
    bp = ndimage.binary_closing(ndimage.binary_opening(bp, disk), disk)
    return bp[2 * k:-2 * k, 2 * k:-2 * k].astype(float)


# gap-matched runs (all gm_..._rRESnROWS cases; the r48n10/n15 butt checks were cached before
# this was added and use the default plane): objective plane and field frequencies of the
# re-optimization
GM_KW = {"spin": dict(x_t=7.0, field_freqs=(0.4015, 0.404, 0.4065)),
         "valley": dict(x_t=9.0, field_freqs=(0.2355, 0.237, 0.2385))}

CASES = {
    "spin_butt": lambda **kw: tp.run_wg_coupling("spin_te", **kw),
    "spin_topopt": lambda **kw: tp.run_wg_coupling("spin_te", design=design("spin_design"), **kw),
    "spin_topopt_vp": lambda **kw: tp.run_wg_coupling("spin_te", design=design("spin_valleyprocedure_design"), **kw),
    "valley_butt": lambda **kw: tp.run_wg_coupling("valley_te", **kw),
    # the filled section rings (its even mode cannot enter the odd edge mode): longer run
    "valley_ld6": lambda **kw: tp.run_wg_coupling("valley_te", fill_len=6.0, **{"max_run": 12000.0, **kw}),
    "valley_odd": lambda **kw: tp.run_wg_coupling("valley_te", eig_band=2, **kw),
    "valley_topopt": lambda **kw: tp.run_wg_coupling("valley_te", design=design("valley_design"), **kw),
    "spin_topopt2": lambda **kw: tp.run_wg_coupling("spin_te", design=design_v2("spin", "gm32_s1"), **kw),
    "valley_topopt2": lambda **kw: tp.run_wg_coupling("valley_te", design=design_v2("valley", "gm36_s1r"), **kw),
    "spin_topopt2c": lambda **kw: tp.run_wg_coupling("spin_te", design=design_v2("spin", "gm32_s1", 100), **kw),
    "valley_topopt2c": lambda **kw: tp.run_wg_coupling("valley_te", design=design_v2("valley", "gm36_s1r", 100), **kw),
    "valley_topopt2c80": lambda **kw: tp.run_wg_coupling("valley_te", design=design_v2("valley", "gm36_s1r", 80), **kw),
    "valley_topopt3": lambda **kw: tp.run_wg_coupling("valley_te", design=design_v2("valley", "gm36_ls80"), **kw),
    "valley_topopt3c80": lambda **kw: tp.run_wg_coupling("valley_te", design=design_v2("valley", "gm36_ls80", 80), **kw),
    "valley_topopt4": lambda **kw: tp.run_wg_coupling("valley_te", design=design_v2("valley", "gm36_rob80"), **kw),
    "valley_topopt4c80": lambda **kw: tp.run_wg_coupling("valley_te", design=design_v2("valley", "gm36_rob80", 80), **kw),
}

if __name__ == "__main__":
    for case in sys.argv[1:]:
        gm = case.startswith("gm_")
        if gm:
            tp.set_gap_matched()
        m = re.fullmatch(r"(?:gm_)?(\w+?)_r(\d+)n(\d+)(?:_y([-+.\d]+))?", case)
        base, kw, out = (m[1], dict(resolution=int(m[2]), n_rows=int(m[3])), DATA / ("gm" if gm else "conv")) \
            if m else (case, {}, DATA)
        if m and m[4]:
            kw["y_strip"] = float(m[4])
        if gm and m:  # objective plane and field frequencies of the re-optimization
            kw.update(GM_KW[base.split("_")[0]])
        out.mkdir(parents=True, exist_ok=True)
        p = out / f"{case}.npz"
        if p.exists():
            continue
        if case == "strip":
            o = tp.run_straight_strip(tp.strip_width("spin_te"))
        else:
            o = CASES[base](**kw)
            o["hz"] = o["hz"][:, ::2, ::2].astype(np.complex64)
            o["eps"] = o["eps"][::2, ::2].astype(np.float32)
        if mp.am_master():
            np.savez_compressed(p, **o)
            print("done", p.name, f"t_end {o.get('t_end', 0):.0f}", flush=True)
