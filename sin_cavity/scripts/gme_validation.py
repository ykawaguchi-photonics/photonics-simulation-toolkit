"""Guided-mode expansion (legume) check of the L3 designs: Si literature cases and the SiN designs.

Si (n 3.46, d = 0.55a, r = 0.25a): plain L3, Akahane S1 = 0.15a, Minkov-Savona shifts
(0.327, 0.257, 0.116)a, compared with Akahane 2003 and Minkov & Savona 2014. SiN (n 2.0, 200 nm,
a = 280 nm, r = 0.30a): the four designs also run in 3D Meep (notebook 02). The defect mode is the
lowest mode in the window below that has no near-degenerate partner: the supercell's folded
band-edge modes come in pairs (|df| < 5e-4 c/a) and can have a higher GME Q than the defect mode.

Results: data/l3/gme_validation.npz.

Usage (legume env, from sin_cavity/): OMP_NUM_THREADS=2 python scripts/gme_validation.py
"""
from pathlib import Path

import numpy as np

from gme_l3 import l3_gme

DATA = Path(__file__).resolve().parents[1] / "data" / "l3"
MS3 = (0.327, 0.257, 0.116)
CASES = {
    # name: (eps, d/a, r/a, shifts, sigma, window c/a)
    "si_plain": (3.46 ** 2, 0.55, 0.25, (), 0.26, (0.252, 0.27)),
    "si_s1": (3.46 ** 2, 0.55, 0.25, (0.15,), 0.26, (0.252, 0.27)),
    "si_ms3": (3.46 ** 2, 0.55, 0.25, MS3, 0.26, (0.252, 0.27)),
    "sin_plain": (4.0, 0.2 / 0.28, 0.30, (), 0.42, (0.415, 0.44)),
    "sin_s25": (4.0, 0.2 / 0.28, 0.30, (0.25,), 0.42, (0.415, 0.44)),
    "sin_s25_20": (4.0, 0.2 / 0.28, 0.30, (0.25, 0.20), 0.42, (0.415, 0.44)),
    "sin_ms3": (4.0, 0.2 / 0.28, 0.30, MS3, 0.42, (0.415, 0.44)),
}

if __name__ == "__main__":
    out = {}
    for name, (eps, d, r, shifts, sigma, (lo, hi)) in CASES.items():
        f, q = l3_gme(eps, d, r, shifts, sigma=sigma)
        alone = np.array([np.sum(np.abs(f - fi) < 5e-4) == 1 for fi in f])
        i = np.flatnonzero((f >= lo) & (f <= hi) & alone)[np.argmin(f[(f >= lo) & (f <= hi) & alone])]
        out[f"{name}_f"], out[f"{name}_Q"] = f[i], q[i]
        print(f"{name}: f {f[i]:.4f} c/a, Q {q[i]:.3g}   (all: "
              + " ".join(f"{a:.4f}/{b:.3g}" for a, b in sorted(zip(f, q))) + ")", flush=True)
    np.savez(DATA / "gme_validation.npz", **out)
