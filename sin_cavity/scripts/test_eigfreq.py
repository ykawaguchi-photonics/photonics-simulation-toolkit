"""Test of Meep's frequency-domain eigensolver (solve_eigfreq) on the plain L3 at 32 px/um.

Reference (Harminv ring-down, same geometry and grid): f 1.5102 c/um (662 nm), Q 506.
Warm start: a narrowband pulse on GUESS_F rings down for `ring` first, so the eigensolver starts
from fields already close to the mode; the inner CW solves use a loose tolerance `cwtol`
(inverse iteration tolerates inexact solves).
Usage (mp-mpi env, from sin_cavity/, PYTHONPATH=../src):
    mpirun -np 12 python scripts/test_eigfreq.py GUESS_F [res] [tol] [cwtol] [ring]
"""
import sys
import time

import meep as mp

from pic_toolkit.meep_sim import phc_cavity as pc

guess = float(sys.argv[1])
res = float(sys.argv[2]) if len(sys.argv) > 2 else 32
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6
cwtol = float(sys.argv[4]) if len(sys.argv) > 4 else tol * 1e-3
ring = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
a, r_a, dpml = 0.28, 0.30, 0.5
holes = pc.l3_holes(a, r_a * a, 14, 10, ())
cell = pc.l3_cell(a, 14, 10, pc.T_SLAB, pad_z=0.6, dpml=dpml)
src = [mp.Source(mp.GaussianSource(guess, fwidth=0.1) if ring else mp.ContinuousSource(guess), component=mp.Ey, center=mp.Vector3())]
sim = mp.Simulation(cell_size=cell, geometry=pc.l3_geometry(holes, pc.T_SLAB), sources=src,
                    boundary_layers=[mp.PML(dpml)], resolution=res, symmetries=pc.TE_SYMMETRIES,
                    eps_averaging=True, force_complex_fields=True)
t0 = time.time()
if ring:
    sim.run(until_after_sources=ring)
    # solve_cw needs a non-zero current: switch to a CW source at the guess after the ring-down.
    sim.change_sources([mp.Source(mp.ContinuousSource(guess), component=mp.Ey, center=mp.Vector3())])
else:
    sim.init_sim()
t1 = time.time()
f = sim.solve_eigfreq(tol=tol, maxiters=30, guessfreq=guess, cwtol=cwtol, cwmaxiters=3000, L=10)
if mp.am_master():
    print(f"guess {guess:.4f}: eigfreq {f.real:.5f}{f.imag:+.3e}j -> lambda {1e3 / f.real:.1f} nm, "
          f"Q {f.real / (-2 * f.imag):.4g}, ring-down {t1 - t0:.0f} s + eigensolver {time.time() - t1:.0f} s",
          flush=True)
