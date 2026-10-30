"""Convergence of the spin-Hall wall runs (notebook 02) with the number of rows per side:
S+ only at the chosen source (S- follows by the mirror), n_rows = 10 ... 40, all else as in
WALL_DEFAULTS. Stores the box power split, the part of each side-plane flux within
|y| < 1.5 of the wall (guided), and the x-averaged |E|^2 profile across the wall.

Usage (mp env, from topological/):
    python scripts/scan_spin_rows.py KIND [N_ROWS ...]
    e.g. python scripts/scan_spin_rows.py spin_te 10 15 20 30 40
"""
import sys
from pathlib import Path

import meep as mp
import numpy as np

from pic_toolkit.meep_sim import topological_pc as tp

mp.verbosity(0)
DATA = Path(__file__).resolve().parents[1] / "data" / "walls"
RUNS = {"spin_tm": 0.457, "spin_te": 0.405}
X_LINE = 8.0

if __name__ == "__main__":
    kind = sys.argv[1]
    rows = [int(r) for r in sys.argv[2:]] or [10, 15, 20, 30, 40]
    out = DATA / f"{kind}_rows_scan.npz"
    done = dict(np.load(out)) if out.exists() else {}
    for n in rows:
        if f"n{n}_P" in done:
            continue
        o = tp.run_domain_wall(kind, +1, RUNS[kind], n_rows=n)
        P = tp.box_fluxes(o, X_LINE, o["dpml"])
        x, y, Sx = o["x"], o["y"], o["Sx"]
        dy = y[1] - y[0]
        il, ir = np.argmin(np.abs(x + X_LINE)), np.argmin(np.abs(x - X_LINE))
        wall = np.abs(y) < 1.5
        guided = np.array([-Sx[il, wall].sum() * dy, Sx[ir, wall].sum() * dy])
        ix = np.abs(x) <= o["n_x"] / 2
        done.update({f"n{n}_P": np.array([P["left"], P["right"], P["top"], P["bottom"]]),
                     f"n{n}_guided": guided, f"n{n}_y": y[::2],
                     f"n{n}_E2y": o["E2"][ix].mean(axis=0)[::2]})
        tot = sum(P.values())
        print(f"{kind} n_rows {n}: eta {(P['left'] - P['right']) / (P['left'] + P['right']):+.4f}, "
              f"bulk {(P['top'] + P['bottom']) / tot:.4f}, P_tot {tot:.4f}, "
              f"guided L/R {guided[0] / tot:.3f}/{guided[1] / tot:.4f}", flush=True)
        np.savez(out, **done)
