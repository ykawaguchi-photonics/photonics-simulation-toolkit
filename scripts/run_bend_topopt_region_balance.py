"""Find the design-region size that satisfies BOTH of the user's requirements
at once: nominal transmission >=75% (ideally >=80%), AND a real margin over a
conventional, un-optimized Euler bend at the same footprint.

Context this corrects for: the 2.0um region that earlier looked best (a 27
point / +56% advantage over conventional) tops out around 0.67-0.69 nominal
once measured honestly (MMA, trained at resolution=30, re-validated at
resolution=40) -- comfortably ahead of a conventional bend there (0.4668),
but well under 0.75. The 4.5um region reaches ~0.96 but topology optimization
does not beat a conventional bend there at all (0.9638 vs 0.9677 at
resolution=40). This sweeps the region sizes in between at CONSISTENT,
correct settings (the two mistakes made earlier -- comparing across different
optimizers, and reading resolution=20 numbers as if they were resolution=40
ones -- are both avoided here by construction) to find where the two curves
actually cross the target band.

Every run: MMA (this module's default, not Adam -- see docs/troubleshooting_
log.md for why a resolution-tuned Adam learning rate is not trustworthy
across regions), corner-biased random start (seed 0), trained at
resolution=30, RE-validated at resolution=40 (one step finer than training,
never the training resolution itself).

Usage:
    /path/to/envs/mp/bin/python -u scripts/run_bend_topopt_region_balance.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "data" / "sparams" / "bend_topopt_robust" / "region_balance"
_CONV_REF = (_REPO / "data" / "sparams" / "bend_topopt_robust"
             / "conventional_reference" / "conventional_reference.json")

from pic_toolkit.meep_sim import bend_topopt, bend_topopt_robust as btr  # noqa: E402

TRAIN_RESOLUTION = 30
VALIDATE_RESOLUTION = 40
_EULER_FOOTPRINT_PER_RADIUS = 1.87


def grid_for_region(region_um: float) -> int:
    return int(round(region_um / 0.025)) + 1


def conventional_at(region_um: float) -> float:
    """Reuse the cached conventional-reference measurement if this exact
    region was already in that sweep; otherwise measure it fresh, at the
    SAME resolution=40 those were recorded at."""
    if _CONV_REF.exists():
        for row in json.loads(_CONV_REF.read_text()):
            if abs(row["region_um"] - region_um) < 1e-9:
                return row["T21"]
    grid_n = grid_for_region(region_um)
    params = {**btr.DEFAULT_PARAMS, "design_region_x_um": region_um,
              "design_region_y_um": region_um, "design_grid_n": grid_n}
    dom = btr._domain_compact(params)
    r_max = (region_um / 2) / _EULER_FOOTPRINT_PER_RADIUS
    radius = round(r_max * 0.92, 3)
    init = bend_topopt.build_euler_initial_density(
        {**params, "design_region_center": (dom["x0"], dom["y0"])}, radius
    )
    val = btr.simulate_baseline_compact({**params, "resolution": VALIDATE_RESOLUTION}, weights=init)
    i_mid = len(val.s_matrix["21"]) // 2
    return float(np.abs(val.s_matrix["21"][i_mid]) ** 2)


def run_region(region_um: float, seed: int = 0) -> dict:
    tag = f"L{region_um:g}_seed{seed}"
    out_npz = _OUT / f"{tag}.npz"
    grid_n = grid_for_region(region_um)

    if out_npz.exists():
        z = np.load(out_npz)
        result = {"region_um": region_um, "T21": float(z["T21"]),
                  "conventional": float(z["conventional"]),
                  "components": int(z["struct_n_components"]),
                  "holes": int(z["struct_enclosed_holes"]),
                  "perim_over_area": float(z["struct_perimeter_over_area"])}
        print(f"[{tag}] cached: T21={result['T21']:.4f} vs conventional={result['conventional']:.4f}",
              flush=True)
        return result

    params = {
        **btr.DEFAULT_PARAMS,
        "design_region_x_um": region_um, "design_region_y_um": region_um,
        "design_grid_n": grid_n, "resolution": TRAIN_RESOLUTION,
    }
    dom = btr._domain_compact(params)
    init = btr.build_corner_biased_random_density(
        params, seed=seed, sigma_um=params["baseline_corner_bias_sigma_um"],
        corner_xy=(dom["x0"], dom["y0"]),
    )

    t0 = time.time()
    print(f"[{tag}] start (train res={TRAIN_RESOLUTION}, grid_n={grid_n}, "
          f"pitch={1000*region_um/(grid_n-1):.1f}nm)", flush=True)
    opt = btr.run_adjoint_optimization_compact(params, init_weights=init)
    print(f"[{tag}] optimized in {(time.time()-t0)/60:.1f} min "
          f"(J: {opt.evaluation_history[0]:.4f} -> {opt.evaluation_history[-1]:.4f})", flush=True)

    val = btr.simulate_baseline_compact({**params, "resolution": VALIDATE_RESOLUTION},
                                        weights=opt.final_weights_binarized)
    i_mid = len(val.s_matrix["21"]) // 2
    t21 = float(np.abs(val.s_matrix["21"][i_mid]) ** 2)
    conv = conventional_at(region_um)
    sm = btr.structure_metrics(opt.final_weights_binarized)
    print(f"[{tag}] RESULT (validated at res={VALIDATE_RESOLUTION}):  "
          f"topology={t21:.4f}  conventional={conv:.4f}  "
          f"advantage={t21-conv:+.4f} ({100*(t21-conv)/conv:+.1f}%)  "
          f"[{(time.time()-t0)/60:.1f} min total]", flush=True)

    np.savez(out_npz, region_um=region_um, seed=seed,
             evaluation_history=opt.evaluation_history,
             x_opt=opt.x_opt, final_weights_binarized=opt.final_weights_binarized,
             T21=t21, conventional=conv,
             **{f"struct_{k}": v for k, v in sm.items()})
    return {"region_um": region_um, "T21": t21, "conventional": conv,
            "components": sm["n_components"], "holes": sm["enclosed_holes"],
            "perim_over_area": sm["perimeter_over_area"]}


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_region(r) for r in (2.5, 3.0, 3.5, 4.05)]

    print("\n=== region balance: topology vs conventional, both at resolution=40 ===", flush=True)
    print(f"{'region':>8}  {'topology':>9}  {'conventional':>12}  {'advantage':>10}  "
          f">=0.75  >=0.80  beats_conv", flush=True)
    for r in rows:
        t21, conv = r["T21"], r["conventional"]
        print(f"{r['region_um']:>7.2f}u  {t21:>9.4f}  {conv:>12.4f}  {t21-conv:>+10.4f}  "
              f"{'yes' if t21>=0.75 else 'no ':>6}  {'yes' if t21>=0.80 else 'no ':>6}  "
              f"{'yes' if t21>conv else 'no'}", flush=True)


if __name__ == "__main__":
    main()
