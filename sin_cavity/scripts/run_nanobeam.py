"""3D ring-down of a SiN nanobeam cavity (Q, f, mode volume); one geometry per call.

Designs: "mccutcheon" (McCutcheon & Loncar 2008, pc.MCCUTCHEON), with optional overrides.
Results go to data/nanobeam/<tag>.npz (skipped if present).

Usage (mp-mpi env, from sin_cavity/, PYTHONPATH=../src):
    mpirun -np 12 python scripts/run_nanobeam.py TAG [key=value ...]
    keys: gap (um), gap_mode (section|edge), res (px/um), n_mirror, n_taper (2-4, inner holes
          of the McCutcheon taper dropped from the outside), pad (um), dpml (um), ring (time),
          w (um), t (um), dr (um, added to every radius), nosym (1: no mirror symmetries),
          design (name of an own design, data/nanobeam/design_<name>.npy; replaces the
          McCutcheon holes, gap/n_mirror/n_taper ignored)
"""
import sys
from pathlib import Path

import meep as mp
import numpy as np

from pic_toolkit.meep_sim import phc_cavity as pc

mp.verbosity(0)
DATA = Path(__file__).resolve().parents[1] / "data" / "nanobeam"

if __name__ == "__main__":
    tag = sys.argv[1]
    kw = dict(a.split("=") for a in sys.argv[2:])
    out_path = DATA / f"{tag}.npz"
    if out_path.exists():
        sys.exit(0)
    d = pc.MCCUTCHEON
    gap = float(kw.get("gap", 0.095))
    gap_mode = kw.get("gap_mode", "section")
    res = float(kw.get("res", 64))
    n_mirror = int(kw.get("n_mirror", d["n_mirror"]))
    n_taper = int(kw.get("n_taper", len(d["taper"])))
    pad = float(kw.get("pad", 0.6))
    dpml = float(kw.get("dpml", 0.5))
    ring = float(kw.get("ring", 300))
    w, t = float(kw.get("w", d["w"])), float(kw.get("t", d["t"]))
    dr = float(kw.get("dr", 0.0))
    # Fewer taper holes: keep the outer ones (closest to the mirror), drop the inner ones.
    taper = [(a, r + dr) for a, r in d["taper"][len(d["taper"]) - n_taper:]]
    mirror = (d["mirror"][0], d["mirror"][1] + dr)
    if "design" in kw:
        # Own design: (x, r) per hole from scripts/design_nanobeam.py (data/nanobeam/design_<name>.npy).
        holes = np.load(DATA / f"design_{kw['design']}.npy") + np.array([0.0, dr])
    else:
        holes = pc.nanobeam_holes(taper, mirror, n_mirror, gap, gap_mode)
    cell = pc.nanobeam_cell(holes, w, t, pad_yz=pad, dpml=dpml)
    geo = pc.nanobeam_geometry(holes, w, t)
    fcen, df = 1 / pc.LAMBDA0, 0.25
    o = pc.run_cavity(geo, cell, fcen, df, res, dpml=dpml, ring_time=ring,
                      symmetries=[] if kw.get("nosym") == "1" else None)
    if mp.am_master():
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, holes=holes, w=w, t=t, gap=gap, gap_mode=gap_mode, pad=pad, dpml=dpml,
                            n_mirror=n_mirror, n_taper=n_taper, dr=dr, **o)
        m = o["modes"]
        print(tag, "modes (f, Q, |amp|, err):", np.array2string(m[:4], precision=4))
        if "Q" in o:
            print(f"{tag}: lambda {1e3 / o['f']:.1f} nm, Q {o['Q']:.3g}, V {o['V']:.3f} (lambda/n)^3, "
                  f"F {o['purcell']:.0f}")
