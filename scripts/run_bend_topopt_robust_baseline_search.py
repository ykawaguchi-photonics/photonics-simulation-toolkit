"""Standalone runner for `04b_bend_topopt_robust.ipynb`'s corner-biased-random
baseline search (the non-Euler-warm-started "before robustness" design family).

Why a script instead of a notebook cell: running all 5 seeds inside ONE
notebook cell hit `nbconvert`'s PER-CELL `ExecutePreprocessor.timeout` and lost
every completed seed with it (a timeout kills the kernel connection, so
`--allow-errors` does not save the outputs either -- 3 hours of compute
discarded). This script instead saves each seed's result to its own `.npz` the
moment that seed finishes, so a timeout, crash, or Ctrl-C never costs more than
the seed in flight, and re-running skips seeds already on disk. The notebook
then only LOADS these artifacts, which is fast and timeout-proof -- the same
split the repo already uses for its other expensive sweeps (see
`scripts/run_mzi_sweep_once.sh`).

Usage (from the repo root, in the `mp` conda env):
    conda run -n mp python scripts/run_bend_topopt_robust_baseline_search.py          # all 5 seeds
    conda run -n mp python scripts/run_bend_topopt_robust_baseline_search.py 0        # just seed 0
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_OUT_DIR = _REPO / "data" / "sparams" / "bend_topopt_robust" / "baseline_search"

from pic_toolkit.meep_sim import bend_topopt_robust as btr  # noqa: E402  (after path setup)


def artifact_name(seed: int, params: dict) -> str:
    """Filename encodes the design region and grid, so results from a different
    region/grid can never be silently mistaken for this configuration's own (the
    3.0um/121 sweep that preceded the 4.5um/181 one produced same-named files
    holding incompatible array shapes)."""
    return (f"seed{seed}"
            f"_L{params['design_region_x_um']:g}"
            f"_n{params['design_grid_n']}.npz")


def run_seed(seed: int, params: dict, corner_xy: tuple, sigma_um: float) -> None:
    out_npz = _OUT_DIR / artifact_name(seed, params)
    if out_npz.exists():
        cached = np.load(out_npz)
        print(f"[seed {seed}] already on disk (T21={float(cached['T21']):.4f}) -- skipping", flush=True)
        return

    t0 = time.time()
    print(f"[seed {seed}] === starting at {time.strftime('%H:%M:%S')} ===", flush=True)
    init = btr.build_corner_biased_random_density(
        params, seed=seed, sigma_um=sigma_um, corner_xy=corner_xy
    )
    print(f"[seed {seed}] initial density fill fraction: {init.mean():.4f}", flush=True)

    opt_result = btr.run_adjoint_optimization_compact(params, init_weights=init)
    t_opt = time.time() - t0
    print(f"[seed {seed}] optimization finished in {t_opt / 60:.1f} min "
          f"(J: {opt_result.evaluation_history[0]:.4f} -> {opt_result.evaluation_history[-1]:.4f})",
          flush=True)

    val = btr.simulate_baseline_compact(params, weights=opt_result.final_weights_binarized)
    i_mid = len(val.s_matrix["21"]) // 2
    t21 = float(np.abs(val.s_matrix["21"][i_mid]) ** 2)
    print(f"[seed {seed}] honest T21={t21:.4f} at {val.wavelengths_um[i_mid]:.4f}um "
          f"-- total {(time.time() - t0) / 60:.1f} min", flush=True)

    np.savez(
        out_npz,
        seed=seed,
        evaluation_history=opt_result.evaluation_history,
        final_weights_binarized=opt_result.final_weights_binarized,
        final_weights_continuous=opt_result.final_weights_continuous,
        x_opt=opt_result.x_opt,
        wavelengths_um=val.wavelengths_um,
        freqs=val.freqs,
        s11=val.s_matrix["11"], s12=val.s_matrix["12"],
        s21=val.s_matrix["21"], s22=val.s_matrix["22"],
        T21=t21,
        initial_fill_fraction=init.mean(),
        wall_time_minutes=(time.time() - t0) / 60,
    )
    print(f"[seed {seed}] saved {out_npz}", flush=True)


def main(seeds: list[int]) -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = dict(btr.DEFAULT_PARAMS)
    dom = btr._domain_compact(params)
    corner_xy = (dom["x0"], dom["y0"])
    sigma_um = params["baseline_corner_bias_sigma_um"]

    print(f"design_grid_n={params['design_grid_n']}  "
          f"design_region={params['design_region_x_um']}x{params['design_region_y_um']}um  "
          f"cell_side={dom['cell_x_um']:.2f}um  resolution={params['resolution']}px/um", flush=True)
    print(f"beta_schedule={params['beta_schedule']}  iters_per_stage={params['iters_per_stage']}  "
          f"corner_bias_sigma_um={sigma_um}  corner_xy=({corner_xy[0]:.3f},{corner_xy[1]:.3f})", flush=True)

    for seed in seeds:
        run_seed(seed, params, corner_xy, sigma_um)

    print("=== all requested seeds complete ===", flush=True)


if __name__ == "__main__":
    requested = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
    main(requested)
