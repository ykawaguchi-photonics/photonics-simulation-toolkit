"""FDTD check of chiral source positions on the spin-Hall walls (notebook 02): S+ only at
candidate points on the two vertical mirror lines of the wall, each placed at x = 0 (the
lattice shifted by x_shift) so the sample is mirror-symmetric about the source; S- follows
by the mirror. Candidates: high |C| points of the edge-mode chirality map
(scripts/chirality_maps.py), including the inclusion (cluster-centre) position.

Usage (mp env, from topological/):
    python scripts/scan_spin_sources.py
"""
from pathlib import Path

import meep as mp
import numpy as np

from pic_toolkit.meep_sim import topological_pc as tp

mp.verbosity(0)
DATA = Path(__file__).resolve().parents[1] / "data" / "walls"
RUNS = {"spin_tm": 0.457, "spin_te": 0.405}
# (map x of the mirror line, y): map x = 0 -> x_shift 0, map x = -1/2 -> x_shift 1/2
CANDIDATES = {
    "spin_tm": [(0.0, -0.4330), (0.0, -0.3403), (0.0, -0.2784), (0.0, 0.0),
                (-0.5, -0.3093), (-0.5, 0.4330), (-0.5, 0.5567)],
    "spin_te": [(0.0, -0.4330), (0.0, -0.3712), (0.0, 0.0619),
                (-0.5, 0.4330), (-0.5, 0.4639), (-0.5, 0.4949)],
}

if __name__ == "__main__":
    out = DATA / "spin_source_scan.npz"
    done = dict(np.load(out)) if out.exists() else {}
    for kind, pts in CANDIDATES.items():
        for xm, y in pts:
            key = f"{kind}_x{xm:+.1f}_y{y:+.4f}"
            if key in done:
                continue
            o = tp.run_domain_wall(kind, +1, RUNS[kind], source_at=((0.0, y), -xm))
            P = tp.box_fluxes(o, 8.0, o["dpml"])
            done[key] = np.array([P["left"], P["right"], P["top"], P["bottom"]])
            tot = sum(P.values())
            print(f"{key}: eta {(P['left'] - P['right']) / (P['left'] + P['right']):+.4f}, "
                  f"bulk {(P['top'] + P['bottom']) / tot:.3f}, P_tot {tot:.4f}", flush=True)
            np.savez(out, **done)
