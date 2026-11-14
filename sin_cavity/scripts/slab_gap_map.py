"""Guided gap map of a triangular air-hole lattice in a suspended 200 nm SiN membrane (MPB).

For each (r/a, a) the TE-like (z-even) bands along Gamma-M-K-Gamma; the guided gap is between
the top of band 1 and the bottom of band 2, both taken only below the light line (f < |k|, c/a
units). The TM-like (z-odd) bands are computed the same way for a = 280 nm at every r/a, for
comparison (a negative "gap" means the bands overlap). Results: data/bands/slab_gap.npz.

Usage (mp env, from sin_cavity/): python scripts/slab_gap_map.py
"""
from pathlib import Path

import meep as mp
import numpy as np

from pic_toolkit.meep_sim import phc_cavity as pc

mp.verbosity(0)
DATA = Path(__file__).resolve().parents[1] / "data" / "bands"

R_A = np.round(np.arange(0.20, 0.4201, 0.01), 3)
A_UM = np.array([0.24, 0.26, 0.28, 0.30, 0.32])
A_TM = 0.28               # lattice constant of the TM-like comparison


def guided_gap(f, kmag):
    # Top of band 1 and bottom of band 2, each only where the band lies below the light line.
    guided = f < kmag[:, None]
    b1_max = f[guided[:, 0], 0].max()
    b2_min = f[guided[:, 1], 1].min() if guided[:, 1].any() else np.nan
    return b1_max, b2_min


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
    bands_tm = np.zeros((len(R_A), len(kpts), 4))
    lo_tm, hi_tm = np.full(len(R_A), np.nan), np.full(len(R_A), np.nan)
    for i, r_a in enumerate(R_A):
        for j, a in enumerate(A_UM):
            f = pc.slab_bands(r_a, pc.T_SLAB / a, kpts, resolution=16, parity="zeven")
            bands[i, j] = f
            lo[i, j], hi[i, j] = guided_gap(f, kmag)
            print(f"TE r/a {r_a:.2f} a {a * 1e3:.0f} nm: gap {lo[i, j]:.4f}-{hi[i, j]:.4f} c/a "
                  f"({200 * (hi[i, j] - lo[i, j]) / (hi[i, j] + lo[i, j]):.1f} %)", flush=True)
        f = pc.slab_bands(r_a, pc.T_SLAB / A_TM, kpts, resolution=16, parity="zodd")
        bands_tm[i] = f
        lo_tm[i], hi_tm[i] = guided_gap(f, kmag)
        print(f"TM r/a {r_a:.2f} a {A_TM * 1e3:.0f} nm: band 1 top {lo_tm[i]:.4f}, band 2 bottom "
              f"{hi_tm[i]:.4f} c/a", flush=True)
    np.savez(DATA / "slab_gap.npz", r_a=R_A, a_um=A_UM, bands=bands, kmag=kmag,
             gap_lo=lo, gap_hi=hi, a_tm=A_TM, bands_tm=bands_tm, gap_lo_tm=lo_tm, gap_hi_tm=hi_tm, k=np.array([[k.x, k.y] for k in kpts]))
