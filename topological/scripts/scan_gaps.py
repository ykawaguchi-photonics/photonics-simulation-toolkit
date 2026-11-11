"""Shared TE gap of the spin-Hall pair versus deformation, and TE gap of the valley-Hall
crystal versus size asymmetry delta (notebook 03, gap-matched comparison of the two feeds).

spin:   a0/R pairs (topological, trivial) around the honeycomb value 3; current 2.9 / 3.125,
        Parappurath et al. (Sci. Adv. 2020) 2.75 / 3.29.  Shared gap = overlap of the two
        crystals' complete gaps near 0.40 c/a0.
valley: delta (sizes s(1 -/+ delta)); current 0.2.  VPC1 and VPC2 have the same bands, so the
        gap of VPC1 is the shared gap.
Gamma-M-K-Gamma path, resolution RES. Results appended to data/bands/gap_scan.json.

Usage (mp env, from topological/):  python scripts/scan_gaps.py spin|valley [VALUES ...]
    spin values as topo:triv, e.g. 2.9:3.125 2.8:3.2;  valley values as delta, e.g. 0.1
"""
import json
import sys
from pathlib import Path

import meep as mp
import numpy as np
from meep import mpb

from pic_toolkit.meep_sim import topological_pc as tp

mp.verbosity(0)
OUT = Path(__file__).resolve().parents[1] / "data" / "bands" / "gap_scan.json"
RES, N_INTERP = 32, 12
KPTS = mp.interpolate(N_INTERP, [tp.GAMMA, tp.M_PT, tp.K_ADJ, tp.GAMMA])


def bands(shapes):
    ms = mpb.ModeSolver(geometry_lattice=tp.LATTICE, geometry=tp.to_objects(shapes, tp.AIR, tp.LATTICE),
                        default_material=tp.HOST, resolution=RES, num_bands=8, k_points=KPTS)
    ms.run_te()
    return np.asarray(ms.all_freqs)


def gap_near(freqs, f0):
    gs = [g for g in tp.complete_gaps(freqs) if g["lo"] < f0 + 0.08 and g["hi"] > f0 - 0.08]
    return min(gs, key=lambda g: abs(g["center"] - f0)) if gs else None


if __name__ == "__main__":
    which, vals = sys.argv[1], sys.argv[2:]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for v in vals:
        key = f"{which}_{v}_r{RES}"
        if key in res:
            continue
        if which == "spin":
            rt, rv = map(float, v.split(":"))
            g = [gap_near(bands(tp.spin_shapes(1 / r, "spin_te")), 0.41) for r in (rt, rv)]
            if None in g:
                out = {"shared": None}
            else:
                lo, hi = max(g[0]["lo"], g[1]["lo"]), min(g[0]["hi"], g[1]["hi"])
                out = {"topo": [g[0]["lo"], g[0]["hi"]], "triv": [g[1]["lo"], g[1]["hi"]],
                       "shared": [lo, hi] if hi > lo else None,
                       "rel": (hi - lo) / ((hi + lo) / 2) if hi > lo else 0.0}
        else:
            d = float(v)
            s = tp.VALLEY_TE_S
            g = gap_near(bands(tp.valley_site_shapes("valley_te", s * (1 - d), s * (1 + d))), 0.24)
            out = {"shared": [g["lo"], g["hi"]], "rel": g["width"] / g["center"]} if g else {"shared": None}
        res[key] = out
        OUT.write_text(json.dumps(res, indent=1))
        print(key, out, flush=True)
