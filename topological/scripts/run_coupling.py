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
Field maps are stored at every second grid point.

Usage (mp env, from topological/):  python scripts/run_coupling.py CASE [CASE ...]
"""
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


CASES = {
    "spin_butt": lambda: tp.run_wg_coupling("spin_te"),
    "spin_topopt": lambda: tp.run_wg_coupling("spin_te", design=design("spin_design")),
    "spin_topopt_vp": lambda: tp.run_wg_coupling("spin_te", design=design("spin_valleyprocedure_design")),
    "valley_butt": lambda: tp.run_wg_coupling("valley_te"),
    # the filled section rings (its even mode cannot enter the odd edge mode): longer run
    "valley_ld6": lambda: tp.run_wg_coupling("valley_te", fill_len=6.0, max_run=12000.0),
    "valley_odd": lambda: tp.run_wg_coupling("valley_te", eig_band=2),
    "valley_topopt": lambda: tp.run_wg_coupling("valley_te", design=design("valley_design")),
}

if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    for case in sys.argv[1:]:
        p = DATA / f"{case}.npz"
        if p.exists():
            continue
        if case == "strip":
            o = tp.run_straight_strip(tp.strip_width("spin_te"))
        else:
            o = CASES[case]()
            o["hz"] = o["hz"][:, ::2, ::2].astype(np.complex64)
            o["eps"] = o["eps"][::2, ::2].astype(np.float32)
        np.savez_compressed(p, **o)
        print("done", p.name, flush=True)
