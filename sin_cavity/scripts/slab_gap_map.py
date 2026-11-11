"""TE-like gap map of a triangular air-hole lattice in a suspended 200 nm SiN membrane (MPB).

For each (r/a, a) the z-even bands along Gamma-M-K-Gamma; the guided gap is between the top of
band 1 and the bottom of band 2, both taken only below the light line (f < |k|, c/a units).
Results: data/bands/slab_gap.npz.

Usage (mp env, from sin_cavity/): python scripts/slab_gap_map.py
"""
from pathlib import Path

import meep as mp
import numpy as np

from pic_toolkit.meep_sim import phc_cavity as pc

mp.verbosity(0)
DATA = Path(__file__).resolve().parents[1] / "data" / "bands"

R_A = np.round(np.arange(0.20, 0.421, 0.02), 3)
A_UM = np.array([0.24, 0.26, 0.28, 0.30, 0.32])

if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    G, M, K = mp.Vector3(), mp.Vector3(0.5, 0), mp.Vector3(-1 / 3, 1 / 3)
    kpts = mp.interpolate(8, [G, M, K, G])
    # |k| in units of 2pi/a for the triangular reciprocal basis used by pc.slab_te_bands.
    b1, b2 = np.array([1 / np.sqrt(3), 1.0]), np.array([1 / np.sqrt(3), -1.0])
    kmag = np.array([np.linalg.norm(k.x * b1 + k.y * b2) for k in kpts])
    bands = np.zeros((len(R_A), len(A_UM), len(kpts), 4))
    lo = np.full((len(R_A), len(A_UM)), np.nan)
    hi = np.full_like(lo, np.nan)
    for i, r_a in enumerate(R_A):
        for j, a in enumerate(A_UM):
            f = pc.slab_te_bands(r_a, pc.T_SLAB / a, kpts, resolution=16)
            bands[i, j] = f
            guided = f < kmag[:, None]
            b1_max = f[guided[:, 0], 0].max()
            b2_min = f[guided[:, 1], 1].min() if guided[:, 1].any() else np.nan
            lo[i, j], hi[i, j] = b1_max, b2_min
            print(f"r/a {r_a:.2f} a {a * 1e3:.0f} nm: gap {b1_max:.4f}-{b2_min:.4f} c/a "
                  f"({200 * (b2_min - b1_max) / (b2_min + b1_max):.1f} %), "
                  f"lambda {a / b2_min * 1e3:.0f}-{a / b1_max * 1e3:.0f} nm", flush=True)
    np.savez(DATA / "slab_gap.npz", r_a=R_A, a_um=A_UM, bands=bands, kmag=kmag,
             gap_lo=lo, gap_hi=hi, k=np.array([[k.x, k.y] for k in kpts]))
