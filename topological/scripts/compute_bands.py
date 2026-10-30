"""MPB data for notebook 01: bulk bands of the eight unit cells (fine Gamma-M-K-Gamma
sweep, p/d character for the spin-Hall cells), ribbon edge bands of the four walls (fine
kx sweep) and the mirror parity of the valley TE edge mode. Cached in data/bands/.

Usage (mp env, from topological/):  python scripts/compute_bands.py [bulk|ribbon|parity]
"""
import json
import sys
from pathlib import Path

import meep as mp
import numpy as np

from pic_toolkit.meep_sim import topological_pc as tp

mp.verbosity(0)
DATA = Path(__file__).resolve().parents[1] / "data" / "bands"
N_INTERP = 40
BULK = {"spin_tm": ("trivial", "honeycomb", "topological"), "spin_te": ("trivial", "honeycomb", "topological"),
        "valley_te": ("symmetric", "vpc1", "vpc2"), "valley_tm": ("symmetric", "vpc1", "vpc2")}
RES_BULK = {"spin_tm": 48, "spin_te": 48, "valley_te": 48, "valley_tm": 48}
NB_BULK = {"spin_tm": 8, "spin_te": 8, "valley_te": 6, "valley_tm": 6}
# ribbons at the resolution of the matching FDTD runs (notebook 02)
RIB = {"spin_tm": dict(resolution=32, num_bands=60), "spin_te": dict(resolution=32, num_bands=60),
       "valley_te": dict(resolution=24, num_bands=24), "valley_tm": dict(resolution=24, num_bands=24)}

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "bulk"
    DATA.mkdir(parents=True, exist_ok=True)
    if what == "bulk":
        ks, dist, ticks = tp.k_path([tp.GAMMA, tp.M_PT, tp.K_ADJ, tp.GAMMA], N_INTERP)
        for kind, variants in BULK.items():
            for v in variants:
                p = DATA / f"bulk_{kind}_{v}.npz"
                if p.exists():
                    continue
                o = tp.bulk_bands(kind, v, ks, RES_BULK[kind], NB_BULK[kind], character=kind.startswith("spin"))
                np.savez(p, dist=dist, ticks=ticks, resolution=RES_BULK[kind], **o)
                print("done", p.name, flush=True)
    elif what == "ribbon":
        kinds = sys.argv[2].split(",") if len(sys.argv) > 2 else list(RIB)
        for kind in kinds:
            p = DATA / f"ribbon_{kind}.npz"
            if p.exists():
                continue
            o = tp.ribbon_bands(kind, np.linspace(0, 0.5, 51), n_rows=8, **RIB[kind])
            np.savez(p, n_rows=8, **RIB[kind], **o)
            print("done", p.name, flush=True)
    elif what == "parity":
        par = tp.wall_mirror_parity("valley_te", 0.32, resolution=24, num_bands=24, f_window=(0.22, 0.255))
        (DATA / "parity_valley_te_kx032.json").write_text(json.dumps(par, indent=1))
        print("done parity", par, flush=True)
