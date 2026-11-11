"""Bulk gap and ribbon edge band of the gap-matched pair (tp.set_gap_matched(): spin-Hall
a0/R = 2.8 / 3.2, valley-Hall delta = 0.115), at the resolution of the FDTD runs, for the
gap-matched coupling comparison of notebook 03. Cached in data/bands/gm_*.npz.

Usage (mp env, from topological/):  python scripts/gapmatched_bands.py spin_te|valley_te RES
"""
import sys
from pathlib import Path

import meep as mp
import numpy as np

from pic_toolkit.meep_sim import topological_pc as tp

mp.verbosity(0)
tp.set_gap_matched()
DATA = Path(__file__).resolve().parents[1] / "data" / "bands"
VARIANTS = {"spin_te": ("topological", "trivial"), "valley_te": ("vpc1",)}
NB = {"spin_te": 60, "valley_te": 24}

if __name__ == "__main__":
    kind, res = sys.argv[1], int(sys.argv[2])
    kp = mp.interpolate(12, [tp.GAMMA, tp.M_PT, tp.K_ADJ, tp.GAMMA])
    gaps = []
    for v in VARIANTS[kind]:
        g = tp.complete_gaps(tp.bulk_bands(kind, v, kp, resolution=res, num_bands=8)["freqs"])
        f0 = 0.41 if kind == "spin_te" else 0.236
        gaps.append(min(g, key=lambda x: abs(x["center"] - f0)))
    lo, hi = max(g["lo"] for g in gaps), min(g["hi"] for g in gaps)
    print(f"{kind} res {res}: shared gap {lo:.4f}-{hi:.4f} ({(hi - lo) / ((hi + lo) / 2):.2%})", flush=True)
    kxs = np.linspace(0, 0.5, 51)
    rb = tp.ribbon_bands(kind, kxs, n_rows=8, resolution=res, num_bands=NB[kind])
    np.savez(DATA / f"gm_ribbon_{kind}_r{res}.npz", **rb, gap=np.array([lo, hi]), resolution=res)
    print("saved", flush=True)
