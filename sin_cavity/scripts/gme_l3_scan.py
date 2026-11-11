"""GME scan of the SiN L3 end-hole shifts (S1, S2, S3) at a = 280 nm, r/a = 0.30, 200 nm slab.

The defect mode is taken as the lowest mode in f = 0.415-0.44 c/a (above the supercell's
band-edge modes, which stay below 0.414 c/a for these shifts). Results: data/l3/gme_scan<name>.npz.
Usage (legume env, from sin_cavity/): python scripts/gme_l3_scan.py [_ext]
"""
from itertools import product
from pathlib import Path

import numpy as np

from gme_l3 import l3_gme

DATA = Path(__file__).resolve().parents[1] / "data" / "l3"
GRIDS = {"": (np.arange(0.15, 0.401, 0.05), np.arange(0.0, 0.301, 0.05), np.arange(0.0, 0.151, 0.05)),
         # second pass: the first grid peaked at its S2 edge (0.30)
         "_ext": (np.arange(0.30, 0.451, 0.05), np.arange(0.30, 0.501, 0.05), np.arange(0.10, 0.251, 0.05))}

if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    S1, S2, S3 = (np.round(g, 3) for g in GRIDS[name])
    f_out = np.full((len(S1), len(S2), len(S3)), np.nan)
    q_out = np.full_like(f_out, np.nan)
    for (i, s1), (j, s2), (k, s3) in product(enumerate(S1), enumerate(S2), enumerate(S3)):
        f, q = l3_gme(4.0, 0.200 / 0.280, 0.30, (s1, s2, s3))
        sel = np.where((f > 0.415) & (f < 0.44))[0]
        if sel.size:
            m = sel[np.argmin(f[sel])]
            f_out[i, j, k], q_out[i, j, k] = f[m], q[m]
        print(f"S {s1:.2f} {s2:.2f} {s3:.2f}: f {f_out[i, j, k]:.4f} c/a, Q {q_out[i, j, k]:.4g}",
              flush=True)
        np.savez(DATA / f"gme_scan{name}.npz", S1=S1, S2=S2, S3=S3, f=f_out, Q=q_out)
