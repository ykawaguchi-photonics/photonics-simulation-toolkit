"""Does a smaller optimizer step find a genuinely higher-transmission (and
therefore more fabrication-sensitive -- higher "Q") nominal design, at the
region size (2.5um) the footprint sweep settled on?

Motivation (the user's own framing): topology optimization alone tends to
discover solutions that lean on delicate interference to reach high nominal
transmission -- exactly the kind of solution a large optimizer step is likely
to step past, landing instead in a coarser, blander, lower-transmission basin.
If the nominal design found here is already low-transmission/low-"Q", a
subsequent fabrication-robust redesign has nothing interesting to fix. So
before building the robust design on top of whatever nominal solution MMA's
default step happens to find, check whether a smaller step finds something
genuinely better -- and, if it does, whether that better design is ALSO more
fabrication-sensitive (the point being made).

Every run: single-objective (not yet the 3-way robust average -- that
question is separate and comes after this one is settled), corner-biased
random start (seed 0), 2.5um region, trained at resolution=30, validated at
resolution=40 (never the training resolution itself).

Usage:
    /path/to/envs/mp/bin/python -u scripts/run_bend_topopt_step_size_study.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "data" / "sparams" / "bend_topopt_robust" / "step_size_study"

from pic_toolkit.meep_sim import bend_topopt_robust as btr  # noqa: E402

VALIDATE_RESOLUTION = 40


def run(label: str, optimizer: str, step_or_lr: float) -> dict:
    out_npz = _OUT / f"{label}.npz"
    params = dict(btr.DEFAULT_PARAMS)
    if optimizer == "mma":
        params["init_step"] = step_or_lr
    else:
        params["optimizer"] = "adam"
        params["learning_rate"] = step_or_lr

    if out_npz.exists():
        z = np.load(out_npz)
        result = {"label": label, "T21": float(z["T21"]), "R11": float(z["R11"]),
                  "components": int(z["struct_n_components"]),
                  "fill": float(z["struct_fill_fraction"]),
                  "bias_sensitivity_proxy": float(z["bias_sensitivity_proxy"])}
        print(f"[{label}] cached: T21={result['T21']:.4f}  "
              f"bias_proxy={result['bias_sensitivity_proxy']:.4f}", flush=True)
        return result

    dom = btr._domain_compact(params)
    init = btr.build_corner_biased_random_density(
        params, seed=0, sigma_um=params["baseline_corner_bias_sigma_um"],
        corner_xy=(dom["x0"], dom["y0"]),
    )

    t0 = time.time()
    print(f"[{label}] start ({optimizer}, {step_or_lr:g})", flush=True)
    run_fn = btr.run_adam_optimization_compact if optimizer == "adam" else btr.run_adjoint_optimization_compact
    opt = run_fn(params, init_weights=init)

    val = btr.simulate_baseline_compact({**params, "resolution": VALIDATE_RESOLUTION},
                                        weights=opt.final_weights_binarized)
    i_mid = len(val.s_matrix["21"]) // 2
    t21 = float(np.abs(val.s_matrix["21"][i_mid]) ** 2)
    r11 = float(np.abs(val.s_matrix["11"][i_mid]) ** 2)
    sm = btr.structure_metrics(opt.final_weights_binarized)

    # Cheap proxy for fabrication sensitivity ("Q"), without a full eta
    # calibration: re-filter the SAME frozen mask at a sharp beta but with a
    # deliberately displaced density threshold (eta 0.4 / 0.6 instead of the
    # calibrated +/-10nm eta_e/eta_d), and see how much T21 swings. Not the
    # calibrated +/-10nm number itself -- just a fast relative ranking across
    # many candidates before committing to the expensive real calibration on
    # the winner.
    fine_mask, fine_params = btr.apply_bias_to_frozen_mask(
        opt.final_weights_binarized, params, eta=0.4, beta=64.0
    )
    fine_mask2, _ = btr.apply_bias_to_frozen_mask(
        opt.final_weights_binarized, params, eta=0.6, beta=64.0
    )
    val_params = {**fine_params, "resolution": VALIDATE_RESOLUTION}
    v1 = btr.simulate_baseline_compact(val_params, weights=fine_mask)
    v2 = btr.simulate_baseline_compact(val_params, weights=fine_mask2)
    t_a = float(np.abs(v1.s_matrix["21"][i_mid]) ** 2)
    t_b = float(np.abs(v2.s_matrix["21"][i_mid]) ** 2)
    bias_proxy = max(abs(t_a - t21), abs(t_b - t21))

    print(f"[{label}] T21={t21:.4f}  R11={r11:.4f}  radiated={1-t21-r11:.4f}  "
          f"components={sm['n_components']}  fill={sm['fill_fraction']:.3f}  "
          f"bias_sensitivity_proxy={bias_proxy:.4f}  [{(time.time()-t0)/60:.1f} min]", flush=True)

    np.savez(out_npz, label=label, optimizer=optimizer, step_or_lr=step_or_lr,
             x_opt=opt.x_opt, final_weights_binarized=opt.final_weights_binarized,
             evaluation_history=opt.evaluation_history,
             T21=t21, R11=r11, bias_sensitivity_proxy=bias_proxy,
             **{f"struct_{k}": v for k, v in sm.items()})
    return {"label": label, "T21": t21, "R11": r11, "components": sm["n_components"],
            "fill": sm["fill_fraction"], "bias_sensitivity_proxy": bias_proxy}


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    # MMA's init_step, progressively smaller than the 0.02 default.
    for step in (0.02, 0.01, 0.005, 0.002):
        rows.append(run(f"mma_step{step:g}", "mma", step))
    # Adam at correspondingly small learning rates (never revisit lr=0.2 --
    # already shown to overshoot at this training resolution).
    for lr in (0.02, 0.01, 0.005):
        rows.append(run(f"adam_lr{lr:g}", "adam", lr))

    (_OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    print("\n=== step-size study, 2.5um, all validated at resolution=40 ===", flush=True)
    for r in sorted(rows, key=lambda d: -d["T21"]):
        print(f"  {r['label']:16s}  T21={r['T21']:.4f}  bias_proxy={r['bias_sensitivity_proxy']:.4f}  "
              f"components={r['components']}  fill={r['fill']:.3f}", flush=True)


if __name__ == "__main__":
    main()
