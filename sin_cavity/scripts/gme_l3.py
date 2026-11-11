"""Guided-mode expansion (legume) of an L3 cavity: frequency and radiative Q of the modes.

Supercell of nx x ny periods of the triangular lattice (units of a), slab of thickness d and
permittivity eps in air, three missing holes, end-hole shifts S1..S3 (units of a, outward along
x). Q from the GME radiative loss at Gamma; in-plane loss is not captured (periodic supercell).
The supercell's folded band-edge modes appear as near-degenerate pairs below the defect mode.

Usage (legume env, from sin_cavity/):
    python scripts/gme_l3.py eps=4.0 d=0.7143 r=0.30 shifts=0.25,0.20 [gmax=2.01 nx=16 ny=10
                             sigma=0.42 neig=8]
"""
import sys
import time

import legume
import numpy as np

legume.set_backend("numpy")


def l3_gme(eps, d, r, shifts=(), gmax=2.01, nx=16, ny=10, neig=8, sigma=0.42):
    """Frequencies (c/a) and radiative Q of the `neig` modes nearest `sigma` at Gamma."""
    lat = legume.Lattice([nx, 0], [0, ny * np.sqrt(3) / 2])
    phc = legume.PhotCryst(lat)
    phc.add_layer(d=d, eps_b=eps)
    for iy in range(-ny // 2, ny // 2):
        for ix in range(-nx // 2, nx // 2):
            x, y = ix + 0.5 * (iy % 2), iy * np.sqrt(3) / 2
            if iy == 0 and ix in (-1, 0, 1):
                continue
            if iy == 0 and abs(ix) >= 2 and abs(ix) - 2 < len(shifts):
                x += np.sign(ix) * shifts[abs(ix) - 2]
            phc.add_shape(legume.Circle(eps=1, x_cent=x, y_cent=y, r=r))
    gme = legume.GuidedModeExp(phc, gmax=gmax)
    gme.run(kpoints=np.array([[0.0], [0.0]]), gmode_inds=[0], numeig=neig, compute_im=False,
            eig_sigma=sigma, verbose=False)
    f = gme.freqs[0]
    rad = gme.compute_rad(kind=0, minds=range(neig))[0]
    return f, f / (2 * np.maximum(rad, 1e-30))


if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:])
    shifts = [float(s) for s in kw["shifts"].split(",")] if kw.get("shifts") else []
    t0 = time.time()
    f, q = l3_gme(float(kw.get("eps", 4.0)), float(kw["d"]), float(kw["r"]), shifts,
                  float(kw.get("gmax", 2.01)), int(kw.get("nx", 16)), int(kw.get("ny", 10)),
                  int(kw.get("neig", 8)), float(kw.get("sigma", 0.42)))
    for i in range(len(f)):
        print(f"mode {i:2d}: f {f[i]:.5f} c/a, Q {q[i]:.4g}")
    print(f"time {time.time() - t0:.0f} s")
