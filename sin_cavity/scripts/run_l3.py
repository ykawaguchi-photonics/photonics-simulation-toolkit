"""3D ring-down of an L3 cavity in the suspended SiN membrane (Q, f, mode volume).

Results go to data/l3/<tag>.npz (skipped if present).

Usage (mp-mpi env, from sin_cavity/, PYTHONPATH=../src):
    mpirun -np 12 python scripts/run_l3.py TAG a=0.28 r_a=0.30 [key=value ...]
    keys: a (um), r_a, shifts (comma list in units of a, S1x,S2x,...), nx, ny (periods / rows
          around the cavity), res (px/um), pad (um), dpml (um), ring (time), t (um),
          fcen (c/um, default 1/0.637), df, pick (q|lowest|amp: mode choice, default q =
          highest-Q mode = fundamental L3 mode), ref (tag of a broadband run: narrowband re-run on its mode,
          df default 0.03)
"""
import sys
from pathlib import Path

import meep as mp
import numpy as np

from pic_toolkit.meep_sim import phc_cavity as pc

mp.verbosity(0)
DATA = Path(__file__).resolve().parents[1] / "data" / "l3"

if __name__ == "__main__":
    tag = sys.argv[1]
    kw = dict(arg.split("=") for arg in sys.argv[2:])
    out_path = DATA / f"{tag}.npz"
    if out_path.exists():
        sys.exit(0)
    a, r_a = float(kw["a"]), float(kw["r_a"])
    shifts = tuple(float(s) for s in kw["shifts"].split(",")) if kw.get("shifts") else ()
    nx, ny = int(kw.get("nx", 14)), int(kw.get("ny", 10))
    res = float(kw.get("res", 64))
    pad, dpml = float(kw.get("pad", 0.6)), float(kw.get("dpml", 0.5))
    ring = float(kw.get("ring", 200))
    t = float(kw.get("t", pc.T_SLAB))
    fcen, df = float(kw.get("fcen", 1 / pc.LAMBDA0)), float(kw.get("df", 0.3))
    if "ref" in kw:
        # Second pass: narrowband pulse on the mode found by a broadband run (data/l3/<ref>.npz),
        # so the field used for V holds that mode only.
        fcen, df = float(np.load(DATA / f"{kw['ref']}.npz")["f"]), float(kw.get("df", 0.03))
    holes = pc.l3_holes(a, r_a * a, nx, ny, shifts)
    cell = pc.l3_cell(a, nx, ny, t, pad_z=pad, dpml=dpml)
    o = pc.run_cavity(pc.l3_geometry(holes, t), cell, fcen, df, res, dpml=dpml, ring_time=ring,
                        pick=kw.get("pick", "q"))
    if mp.am_master():
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, holes=holes, a=a, r_a=r_a, shifts=np.array(shifts), nx=nx, ny=ny, t=t,
                            pad=pad, dpml=dpml, **o)
        print(tag, "modes (f, Q, |amp|, err):", np.array2string(o["modes"][:4].real, precision=4))
        if "Q" in o:
            print(f"{tag}: lambda {1e3 / o['f']:.1f} nm, Q {o['Q']:.3g}, V {o['V']:.3f} (lambda/n)^3, "
                  f"F {o['purcell']:.0f}")
