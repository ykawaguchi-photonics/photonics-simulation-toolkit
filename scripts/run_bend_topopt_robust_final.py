"""Final deliverable pipeline for `04b_bend_topopt_robust.ipynb`, at the
configuration the Tidy3D-tutorial-informed study settled on:

    design region  2.0um  (1.48 wavelengths)   <- where topology optimization
                                                  beats a conventional bend by
                                                  27 points (+56% relative)
    design_grid_n  81     (25nm pitch)          <- 161 measured worse
    resolution     30 px/um                     <- 20 is under-resolved in Si
    optimizer      Adam, learning_rate 0.2      <- best of MMA/Adam sweep here
    target bias    +/-10nm

Runs, in order, saving each stage as it completes:
  1. calibrate eta_e / eta_d at this configuration (never reused across
     configurations -- the eta<->nm mapping depends on grid and filter)
  2. BASELINE: single-objective optimization (the "before robustness" design)
  3. ROBUST:   3-way averaged objective from the SAME start and budget, so the
     only difference between the two is the objective itself
  4. measure both designs' transmission under the calibrated +/-10nm bias

Usage:
    /path/to/envs/mp/bin/python -u scripts/run_bend_topopt_robust_final.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "data" / "sparams" / "bend_topopt_robust" / "final"

from pic_toolkit.meep_sim import bend_topopt_robust as btr  # noqa: E402


def measure_bias_triplet(params: dict, x_opt: np.ndarray, resolution: int) -> dict:
    """Honest two-port transmission of one design rendered at eta_i / eta_e /
    eta_d -- i.e. as drawn, 10nm thinner, and 10nm thicker."""
    n = params["design_grid_n"]
    final = {**params, "beta": params["beta_schedule"][-1]}
    val_params = {**params, "resolution": resolution}
    out = {}
    for label, eta in (("nominal", params["eta_i"]),
                       ("eroded", params["eta_e"]),
                       ("dilated", params["eta_d"])):
        w = np.array(btr._mapping_robust(x_opt, final, eta)).reshape(n, n)
        r = btr.simulate_baseline_compact(val_params, weights=w)
        i_mid = len(r.s_matrix["21"]) // 2
        out[label] = float(np.abs(r.s_matrix["21"][i_mid]) ** 2)
        print(f"    {label}: T21={out[label]:.4f}", flush=True)
    out["max_abs_delta"] = max(abs(out["dilated"] - out["nominal"]),
                               abs(out["eroded"] - out["nominal"]))
    out["relative_spread"] = out["max_abs_delta"] / out["nominal"]
    print(f"    max|delta T|={out['max_abs_delta']:.4f}  "
          f"relative spread={out['relative_spread']:.4%}", flush=True)
    return out


def main(seed: int = 0) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if seed == 0 else f"_seed{seed}"
    params = dict(btr.DEFAULT_PARAMS)
    dom = btr._domain_compact(params)
    corner_xy = (dom["x0"], dom["y0"])
    validation_resolution = 40  # one step above the training resolution, for the
                                # reported fabrication-bias numbers only

    print(f"region={params['design_region_x_um']}um  grid_n={params['design_grid_n']}  "
          f"resolution={params['resolution']}  optimizer={params['optimizer']}"
          f"(lr={params['learning_rate']})  cell={dom['cell_x_um']:.2f}um", flush=True)

    # --- 1. calibration -----------------------------------------------------
    cal_path = _OUT / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
        print(f"calibration cached: eta_e={cal['eta_e']:.5f} eta_d={cal['eta_d']:.5f}", flush=True)
    else:
        t0 = time.time()
        eta_d = btr.find_eta_for_bias(params, beta=32.0, target_bias_um=+params["target_bias_um"])
        eta_e = btr.find_eta_for_bias(params, beta=32.0, target_bias_um=-params["target_bias_um"])
        eta_d4 = btr.find_eta_for_bias(params, beta=4.0, target_bias_um=+params["target_bias_um"])
        eta_e4 = btr.find_eta_for_bias(params, beta=4.0, target_bias_um=-params["target_bias_um"])
        cal = {"eta_e": eta_e, "eta_d": eta_d, "eta_e_beta4": eta_e4, "eta_d_beta4": eta_d4,
               "minutes": (time.time() - t0) / 60}
        cal_path.write_text(json.dumps(cal, indent=2))
        print(f"calibrated eta_e={eta_e:.5f} eta_d={eta_d:.5f} "
              f"(beta=4 check: {eta_e4:.5f}/{eta_d4:.5f}, "
              f"delta {abs(eta_e4-eta_e):.5f}/{abs(eta_d4-eta_d):.5f})", flush=True)
    params["eta_e"], params["eta_d"] = cal["eta_e"], cal["eta_d"]

    init = btr.build_corner_biased_random_density(
        params, seed=seed, sigma_um=params["baseline_corner_bias_sigma_um"], corner_xy=corner_xy
    )

    # --- 2. baseline: single objective --------------------------------------
    base_path = _OUT / f"baseline_single_objective{suffix}.npz"
    if base_path.exists():
        bz = np.load(base_path)
        print(f"baseline cached (T21={float(bz['T21_nominal']):.4f})", flush=True)
    else:
        t0 = time.time()
        print(f"=== baseline: single-objective (seed {seed}, {params['optimizer']}) ===", flush=True)
        # MUST honour params["optimizer"] -- an earlier version hardcoded Adam
        # here while the robust run below read the setting, so the two sides of
        # the comparison silently used different optimizers.
        single_fn = (btr.run_adam_optimization_compact if params["optimizer"] == "adam"
                     else btr.run_adjoint_optimization_compact)
        opt = single_fn(params, init_weights=init)
        tri = measure_bias_triplet(params, opt.x_opt, validation_resolution)
        sm = btr.structure_metrics(opt.final_weights_binarized)
        np.savez(base_path, x_opt=opt.x_opt,
                 evaluation_history=opt.evaluation_history,
                 final_weights_binarized=opt.final_weights_binarized,
                 final_weights_continuous=opt.final_weights_continuous,
                 T21_nominal=tri["nominal"], T21_eroded=tri["eroded"],
                 T21_dilated=tri["dilated"], max_abs_delta=tri["max_abs_delta"],
                 relative_spread=tri["relative_spread"],
                 minutes=(time.time() - t0) / 60,
                 **{f"struct_{k}": v for k, v in sm.items()})
        print(f"baseline saved ({(time.time()-t0)/60:.1f} min), structure={sm}", flush=True)

    # --- 3. robust: 3-way averaged objective, same start --------------------
    rob_path = _OUT / f"robust_3way{suffix}.npz"
    if rob_path.exists():
        print("robust cached", flush=True)
    else:
        t0 = time.time()
        print("=== robust: 3-way averaged objective (same start, same budget) ===", flush=True)
        rob = btr.run_robust_adjoint_optimization_compact(params, init_weights=init)
        tri = measure_bias_triplet(params, rob.x_opt, validation_resolution)
        sm = btr.structure_metrics(rob.final_weights_binarized["nominal"])
        np.savez(rob_path, x_opt=rob.x_opt,
                 evaluation_history_avg=rob.evaluation_history_avg,
                 evaluation_history_nominal=rob.evaluation_history_nominal,
                 evaluation_history_eroded=rob.evaluation_history_eroded,
                 evaluation_history_dilated=rob.evaluation_history_dilated,
                 final_weights_binarized_nominal=rob.final_weights_binarized["nominal"],
                 final_weights_continuous_nominal=rob.final_weights_continuous["nominal"],
                 final_weights_continuous_eroded=rob.final_weights_continuous["eroded"],
                 final_weights_continuous_dilated=rob.final_weights_continuous["dilated"],
                 T21_nominal=tri["nominal"], T21_eroded=tri["eroded"],
                 T21_dilated=tri["dilated"], max_abs_delta=tri["max_abs_delta"],
                 relative_spread=tri["relative_spread"],
                 minutes=(time.time() - t0) / 60,
                 **{f"struct_{k}": v for k, v in sm.items()})
        print(f"robust saved ({(time.time()-t0)/60:.1f} min), structure={sm}", flush=True)

    # --- 4. headline --------------------------------------------------------
    b = np.load(base_path)
    r = np.load(rob_path)
    print("\n=== HEADLINE (2.0um, res30 training / res40 validation, +/-10nm) ===", flush=True)
    print(f"  single objective : nominal={float(b['T21_nominal']):.4f}  "
          f"max|dT|={float(b['max_abs_delta']):.4f}  spread={float(b['relative_spread']):.4%}", flush=True)
    print(f"  3-way averaged   : nominal={float(r['T21_nominal']):.4f}  "
          f"max|dT|={float(r['max_abs_delta']):.4f}  spread={float(r['relative_spread']):.4%}", flush=True)
    improvement = (float(b["max_abs_delta"]) - float(r["max_abs_delta"])) / float(b["max_abs_delta"])
    print(f"  robustness gain  : {improvement:+.1%} reduction in worst-case sensitivity", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
