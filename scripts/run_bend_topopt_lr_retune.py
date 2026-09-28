"""Re-tune the optimizer at the FINAL training resolution.

Adam's learning rate was swept at `resolution=20` (best: lr=0.2, honest
T21=0.7560 at 2.0um) and the training resolution was raised to 30
independently (measured under-resolution at 20). Combining the two dropped
the result to 0.6040, with the design noticeably more filled-in (fill
fraction 0.48 vs 0.33, 3 connected components vs 7) -- the signature of an
overshooting step, which is expected: the adjoint gradient's magnitude
depends on the discretization, so a learning rate tuned at one resolution
does not carry to another.

This sweeps the optimizer at resolution=30 and validates every candidate at
the SAME resolution (40), so the numbers are comparable to each other and to
the final pipeline's.

Usage:
    /path/to/envs/mp/bin/python -u scripts/run_bend_topopt_lr_retune.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "data" / "sparams" / "bend_topopt_robust" / "lr_retune"

from pic_toolkit.meep_sim import bend_topopt_robust as btr  # noqa: E402

VALIDATION_RESOLUTION = 40


def run(label: str, overrides: dict) -> dict:
    out_npz = _OUT / f"{label}.npz"
    if out_npz.exists():
        z = np.load(out_npz)
        print(f"[{label}] cached: T21={float(z['T21']):.4f}", flush=True)
        return {"label": label, "T21": float(z["T21"]),
                "fill": float(z["struct_fill_fraction"]),
                "components": int(z["struct_n_components"])}

    params = {**btr.DEFAULT_PARAMS, **overrides}
    dom = btr._domain_compact(params)
    init = btr.build_corner_biased_random_density(
        params, seed=0, sigma_um=params["baseline_corner_bias_sigma_um"],
        corner_xy=(dom["x0"], dom["y0"]),
    )

    t0 = time.time()
    print(f"[{label}] start ({params['optimizer']}, lr={params.get('learning_rate')}, "
          f"res={params['resolution']})", flush=True)
    run_fn = (btr.run_adam_optimization_compact if params["optimizer"] == "adam"
              else btr.run_adjoint_optimization_compact)
    opt = run_fn(params, init_weights=init)

    val = btr.simulate_baseline_compact({**params, "resolution": VALIDATION_RESOLUTION},
                                        weights=opt.final_weights_binarized)
    i_mid = len(val.s_matrix["21"]) // 2
    t21 = float(np.abs(val.s_matrix["21"][i_mid]) ** 2)
    sm = btr.structure_metrics(opt.final_weights_binarized)
    print(f"[{label}] T21={t21:.4f} (validated at res={VALIDATION_RESOLUTION})  "
          f"fill={sm['fill_fraction']:.3f} components={sm['n_components']}  "
          f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    np.savez(out_npz, x_opt=opt.x_opt,
             evaluation_history=opt.evaluation_history,
             final_weights_binarized=opt.final_weights_binarized,
             final_weights_continuous=opt.final_weights_continuous,
             T21=t21, **{f"struct_{k}": v for k, v in sm.items()})
    return {"label": label, "T21": t21, "fill": sm["fill_fraction"],
            "components": sm["n_components"]}


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    # MMA has no learning rate to mis-tune -- the control.
    rows.append(run("res30_mma", {"optimizer": "mma"}))
    for lr in (0.02, 0.05, 0.1, 0.2):
        rows.append(run(f"res30_adam_lr{lr:g}", {"optimizer": "adam", "learning_rate": lr}))

    (_OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    print("\n=== resolution=30, all validated at resolution=40 ===", flush=True)
    for r in sorted(rows, key=lambda d: -d["T21"]):
        print(f"  {r['label']:22s} T21={r['T21']:.4f}  fill={r['fill']:.3f}  "
              f"components={r['components']}", flush=True)


if __name__ == "__main__":
    main()
