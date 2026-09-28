"""Apples-to-apples control for `04b_bend_topopt_robust.ipynb`'s headline claim.

The notebook's before/after pair differs in TWO ways at once, by the study's own
design: the "before" baseline is corner-biased-random-started and
single-objective, while the "after" robust design is Euler-warm-started and
3-way averaged. So its headline improvement conflates "the averaged objective
helped" with "the Euler warm start helped".

This script runs the missing control: SINGLE-objective (non-robust) optimization
from the SAME Euler warm start, in the SAME compact domain, with the SAME
budget. Comparing it against the robust design isolates the averaged
objective's own contribution, since the only remaining difference is the
objective itself.

Usage (repo root):
    /path/to/envs/mp/bin/python -u scripts/run_bend_topopt_robust_euler_control.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "data" / "sparams" / "bend_topopt_robust" / "euler_control"

from pic_toolkit.meep_sim import bend_topopt, bend_topopt_robust as btr  # noqa: E402


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    params = dict(btr.DEFAULT_PARAMS)
    dom = btr._domain_compact(params)

    # eta_e/eta_d are needed only to MEASURE the +/-10nm sensitivity afterwards,
    # not to optimize -- this control deliberately optimizes the nominal
    # rendering alone.
    eta_d = btr.find_eta_for_bias(params, beta=32.0, target_bias_um=+params["target_bias_um"])
    eta_e = btr.find_eta_for_bias(params, beta=32.0, target_bias_um=-params["target_bias_um"])
    print(f"calibrated eta_e={eta_e:.5f} eta_d={eta_d:.5f}", flush=True)

    euler_params = {**params, "design_region_center": (dom["x0"], dom["y0"])}
    init = bend_topopt.build_euler_initial_density(euler_params, params["init_bend_radius_um"])
    print(f"Euler warm start fill fraction: {init.mean():.4f}", flush=True)

    t0 = time.time()
    opt = btr.run_adjoint_optimization_compact(params, init_weights=init)
    print(f"single-objective optimization finished in {(time.time()-t0)/60:.1f} min "
          f"(J: {opt.evaluation_history[0]:.4f} -> {opt.evaluation_history[-1]:.4f})", flush=True)

    # Measure at the SAME resolution the notebook chose for its reported numbers.
    val_params = {**params, "resolution": 40}
    results = {}
    for label, eta in (("nominal", params["eta_i"]), ("eroded", eta_e), ("dilated", eta_d)):
        weights = np.array(btr._mapping_robust(
            opt.x_opt, {**params, "beta": params["beta_schedule"][-1]}, eta
        )).reshape(params["design_grid_n"], params["design_grid_n"])
        r = btr.simulate_baseline_compact(val_params, weights=weights)
        i_mid = len(r.s_matrix["21"]) // 2
        results[label] = float(np.abs(r.s_matrix["21"][i_mid]) ** 2)
        print(f"  {label}: T={results[label]:.4f}", flush=True)

    max_delta = max(abs(results["dilated"] - results["nominal"]),
                    abs(results["eroded"] - results["nominal"]))
    print(f"max|delta T| = {max_delta:.4f}  "
          f"relative spread = {max_delta / results['nominal']:.4%}", flush=True)

    np.savez(
        _OUT / f"euler_control_L{params['design_region_x_um']:g}_n{params['design_grid_n']}.npz",
        evaluation_history=opt.evaluation_history,
        x_opt=opt.x_opt,
        final_weights_binarized=opt.final_weights_binarized,
        T_nominal=results["nominal"], T_eroded=results["eroded"], T_dilated=results["dilated"],
        max_abs_delta=max_delta,
        eta_e=eta_e, eta_d=eta_d,
        validation_resolution=40,
    )
    print("saved euler-control artifact", flush=True)


if __name__ == "__main__":
    main()
