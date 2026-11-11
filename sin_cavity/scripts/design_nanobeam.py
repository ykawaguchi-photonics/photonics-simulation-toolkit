"""Own SiN nanobeam design after Quan & Loncar (Opt. Express 19, 18529, 2011), 637 nm.

Same beam as McCutcheon & Loncar (w 300 nm, t 200 nm, n 2.0), constant period a = 250 nm, zero
cavity length (innermost holes at x = +-a/2). MPB band edges of one beam period vs hole radius:
- centre radius r_c: dielectric band edge f1(r_c) = f_res (the mode forms at that band edge);
- mirror radius r_m: largest mirror strength gamma at f_res;
- taper: filling fraction pi r^2 / (a w) quadratic from r_c to r_m over N segments (mirror
  strength rising about linearly, Gaussian envelope), then n_mirror holes at r_m.
Writes data/nanobeam/design_band_edges.npz and design_N<N>.npy (x, r per hole, x > 0, um).

Usage (mp env, from sin_cavity/, PYTHONPATH=../src): python scripts/design_nanobeam.py
"""
from pathlib import Path

import meep as mp
import numpy as np

from pic_toolkit.meep_sim import phc_cavity as pc

mp.verbosity(0)
DATA = Path(__file__).resolve().parents[1] / "data" / "nanobeam"
A, W, T = 0.250, 0.300, 0.200
F_RES = 1 / pc.LAMBDA0
R = np.round(np.arange(0.050, 0.1001, 0.005), 4)

if __name__ == "__main__":
    edges = np.array([pc.nanobeam_band_edges(A, r, W, T, resolution=16) for r in R])
    gam = np.array([pc.mirror_strength(f1, f2, F_RES) for f1, f2 in edges])
    for r, (f1, f2), g in zip(R, edges, gam):
        print(f"r {r * 1e3:.0f} nm: band edges {1e3 / f2:.0f}-{1e3 / f1:.0f} nm, gamma {g:.3f}", flush=True)
    np.savez(DATA / "design_band_edges.npz", r=R, f1=edges[:, 0], f2=edges[:, 1], gamma=gam, a=A, w=W, t=T)
    r_c = float(np.interp(F_RES, edges[:, 0], R))       # f1 rises with r
    r_m = float(R[np.argmax(gam)])
    print(f"r_c {r_c * 1e3:.1f} nm (f1 = f_res), r_m {r_m * 1e3:.1f} nm (gamma {gam.max():.3f})")
    ff_c, ff_m = np.pi * r_c ** 2 / (A * W), np.pi * r_m ** 2 / (A * W)
    for n_taper in (15, 20):
        i = np.arange(n_taper + 1)
        ff = ff_c + (ff_m - ff_c) * (i / n_taper) ** 2
        radii = np.concatenate([np.sqrt(ff * A * W / np.pi), np.full(10, r_m)])
        x = A / 2 + A * np.arange(len(radii))
        np.save(DATA / f"design_N{n_taper}.npy", np.column_stack([x, radii]))
        print(f"N{n_taper}: {len(radii)} holes per side, beam {2 * x[-1]:.2f} um")
