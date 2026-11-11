"""Mesh check of a finished v2 transition design (notebook 03): the binary design of
data/topopt/v2_{NAME}_{TAG}_design.npz re-evaluated (forward run only) at another resolution,
with the same rows, band, design region, objective plane and objective (all forward power) as
the optimization.

The design is carried to the new grid as its filtered density (conic filter on the
optimization grid), linearly interpolated, and evaluated at beta = inf with subpixel smoothing:
the structure is the binary level set rho = 0.5 with its boundary placed below the pixel, the
same representation the optimizer's beta = inf iterations used (at the optimization resolution
this reproduces the stage's best J). The pixel-thresholded variant (rho > 0.5 per pixel, a
staircase of the design grid) is evaluated as well.

Usage (from topological/, mp-mpi env):
    mpirun -np 12 python scripts/eval_design_res.py NAME TAG RES
Appends {res: J} to data/topopt/v2_{NAME}_{TAG}_meshcheck.json.
"""
import json
import sys
from pathlib import Path

import meep as mp
import meep.adjoint as mpa
import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import topopt_transition_v2 as v2  # noqa: E402
from pic_toolkit.meep_sim import topological_pc as tp  # noqa: E402

if __name__ == "__main__":
    name, tag, res = sys.argv[1], sys.argv[2], int(sys.argv[3])
    stem = v2.DATA / f"v2_{name}_{tag}"
    d = np.load(f"{stem}_design.npz")
    c = v2.SETUP[name]
    if bool(d["gm"]):
        tp.set_gap_matched()
    v2.DES_C, v2.DES_S = tuple(d["center"]), tuple(d["size"])
    v2.X_T = float(d["x_t"])
    v2.X_PML_R = v2.X_T + 1.0
    res0, rows, freqs = int(d["res"]), int(d["n_rows"]), np.asarray(d["freqs"])
    nx0, ny0 = int(round(v2.DES_S[0] * res0)) + 1, int(round(v2.DES_S[1] * res0)) + 1
    rho = np.asarray(mpa.conic_filter(d["x"].reshape(nx0, ny0), float(d["filter_radius"]),
                                      v2.DES_S[0], v2.DES_S[1], res0))
    opt, dv, _, (nx, ny), _, _ = v2.problem(c, res, rows, float(d["y_strip"]), freqs,
                                            bool(d["full"]))
    rho1 = np.clip(zoom(rho, (nx / nx0, ny / ny0), order=1)[:nx, :ny], 0, 1)
    dv.beta = np.inf
    j = float(np.real(opt([rho1.flatten()], need_gradient=False)[0]))
    j_px = float(np.real(opt([(rho1 > 0.5).astype(float).flatten()], need_gradient=False)[0]))
    v2.log(f"{name} {tag}: res {res0} -> {res}, rows {rows}: J level set = {j:.4f}, pixel threshold = {j_px:.4f}")
    if mp.am_master():
        out = Path(f"{stem}_meshcheck.json")
        r = json.loads(out.read_text()) if out.exists() else {}
        r[str(res)] = {"level_set": j, "pixel": j_px}
        out.write_text(json.dumps(r, indent=1))
