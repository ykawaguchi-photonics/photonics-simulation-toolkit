"""Sweep runner for `04b_bend_topopt_robust.ipynb`'s Tidy3D-tutorial-informed
re-examination: initial condition x design-region size x FDTD resolution.

Motivation: the designs this notebook produced so far look like an ordinary
smooth bend, not a topology-optimized structure. Measured on the saved
designs (connected components / enclosed holes / perimeter-to-area):

    Euler warm start, 4.5um : 1 component, 0 holes, 0.120
    corner-biased random, 4.5um: 2-4 components, 0 holes, 0.128-0.138
    corner-biased random, 3.0um: 1-2 components, 0 holes, 0.121-0.133

i.e. shrinking the region alone did NOT produce freeform structure -- the
corner-biased initializer seeds a bend shape, so the optimizer just refines
one. The Tidy3D tutorial instead starts from a UNIFORM 0.5 gray field and
lets the optimizer discover the structure. This script measures that, plus
the region-size and FDTD-resolution axes, on the same footing.

Structure metrics are recorded alongside transmission for every run, because
"did topology optimization actually earn its name here" is a first-class
question for this study, not an afterthought.

Each configuration is saved to its own `.npz` as soon as it finishes, and
existing files are skipped, so a crash/timeout costs at most the run in
flight (see docs/troubleshooting_log.md for why that matters here).

Usage (repo root, mp env interpreter directly -- `conda run` buffers output):
    /path/to/envs/mp/bin/python -u scripts/run_bend_topopt_init_comparison.py A
    /path/to/envs/mp/bin/python -u scripts/run_bend_topopt_init_comparison.py B
    /path/to/envs/mp/bin/python -u scripts/run_bend_topopt_init_comparison.py C
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "data" / "sparams" / "bend_topopt_robust" / "init_comparison"

from pic_toolkit.meep_sim import bend_topopt, bend_topopt_robust as btr  # noqa: E402


structure_metrics = btr.structure_metrics  # single definition lives in the module


def make_initial_density(kind: str, params: dict, dom: dict, seed: int) -> np.ndarray | None:
    """`None` means "let the optimizer start from a uniform init_density gray
    field" -- run_adjoint_optimization_compact's own default, and the Tidy3D
    tutorial's choice ("initially uniform with halfway values of 0.5")."""
    if kind == "uniform":
        return None
    if kind == "corner":
        return btr.build_corner_biased_random_density(
            params, seed=seed, sigma_um=params["baseline_corner_bias_sigma_um"],
            corner_xy=(dom["x0"], dom["y0"]),
        )
    if kind == "euler":
        return bend_topopt.build_euler_initial_density(
            {**params, "design_region_center": (dom["x0"], dom["y0"])},
            params["init_bend_radius_um"],
        )
    raise ValueError(f"unknown init kind: {kind}")


def run_one(kind: str, region_um: float, resolution: int, seed: int, grid_n: int,
            penalty_strength: float = 0.0, penalty_eta_e: float = 0.75,
            beta_schedule: list | None = None, filter_radius_um: float | None = None,
            iters_per_stage: int | None = None, init_step: float | None = None,
            optimizer: str = "mma", learning_rate: float | None = None) -> None:
    tag = f"{kind}_L{region_um:g}_n{grid_n}_res{resolution}_seed{seed}"
    if penalty_strength > 0:
        tag += f"_pen{penalty_strength:g}eta{penalty_eta_e:g}"
    if beta_schedule is not None:
        tag += f"_beta{'-'.join(str(int(b)) for b in beta_schedule)}"
    if filter_radius_um is not None:
        tag += f"_R{filter_radius_um:g}"
    if iters_per_stage is not None:
        tag += f"_it{iters_per_stage}"
    if init_step is not None:
        tag += f"_step{init_step:g}"
    if optimizer != "mma":
        tag += f"_{optimizer}"
        if learning_rate is not None:
            tag += f"lr{learning_rate:g}"
    out_npz = _OUT / f"{tag}.npz"
    if out_npz.exists():
        z = np.load(out_npz)
        print(f"[{tag}] already on disk (T21={float(z['T21']):.4f}) -- skipping", flush=True)
        return

    params = {
        **btr.DEFAULT_PARAMS,
        "design_region_x_um": region_um,
        "design_region_y_um": region_um,
        "design_grid_n": grid_n,
        "resolution": resolution,
        "penalty_strength": penalty_strength,
        "penalty_eta_e": penalty_eta_e,
    }
    if beta_schedule is not None:
        params["beta_schedule"] = list(beta_schedule)
    if filter_radius_um is not None:
        params["filter_radius_um"] = filter_radius_um
    if iters_per_stage is not None:
        params["iters_per_stage"] = iters_per_stage
    if init_step is not None:
        params["init_step"] = init_step
    if learning_rate is not None:
        params["learning_rate"] = learning_rate
    dom = btr._domain_compact(params)

    t0 = time.time()
    print(f"[{tag}] === start {time.strftime('%H:%M:%S')} "
          f"(cell {dom['cell_x_um']:.2f}um, pitch {1000*region_um/(grid_n-1):.1f}nm) ===", flush=True)

    init = make_initial_density(kind, params, dom, seed)
    print(f"[{tag}] initial fill: "
          f"{'uniform ' + str(params['init_density']) if init is None else f'{init.mean():.4f}'}", flush=True)

    run_fn = (btr.run_adam_optimization_compact if optimizer == "adam"
              else btr.run_adjoint_optimization_compact)
    opt = run_fn(params, init_weights=init)
    t_opt = (time.time() - t0) / 60
    print(f"[{tag}] optimized in {t_opt:.1f} min "
          f"(J: {opt.evaluation_history[0]:.4f} -> {opt.evaluation_history[-1]:.4f})", flush=True)

    val = btr.simulate_baseline_compact(params, weights=opt.final_weights_binarized)
    i_mid = len(val.s_matrix["21"]) // 2
    t21 = float(np.abs(val.s_matrix["21"][i_mid]) ** 2)
    r11 = float(np.abs(val.s_matrix["11"][i_mid]) ** 2)
    sm = structure_metrics(opt.final_weights_binarized)
    print(f"[{tag}] honest T21={t21:.4f}  R11={r11:.4f}  radiated={1-t21-r11:.4f}", flush=True)
    print(f"[{tag}] structure: components={sm['n_components']} holes={sm['enclosed_holes']} "
          f"perim/area={sm['perimeter_over_area']:.3f} fill={sm['fill_fraction']:.3f}", flush=True)

    np.savez(
        out_npz,
        init_kind=kind, region_um=region_um, resolution=resolution, seed=seed, grid_n=grid_n,
        evaluation_history=opt.evaluation_history,
        final_weights_binarized=opt.final_weights_binarized,
        final_weights_continuous=opt.final_weights_continuous,
        x_opt=opt.x_opt,
        wavelengths_um=val.wavelengths_um,
        s11=val.s_matrix["11"], s12=val.s_matrix["12"],
        s21=val.s_matrix["21"], s22=val.s_matrix["22"],
        T21=t21, R11=r11,
        wall_time_minutes=(time.time() - t0) / 60,
        **{f"struct_{k}": v for k, v in sm.items()},
    )
    print(f"[{tag}] saved", flush=True)


def grid_for_region(region_um: float) -> int:
    """Keep the verified 25nm design-grid pitch at any region size."""
    return int(round(region_um / 0.025)) + 1


def main(experiment: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)

    if experiment == "A":
        # Initial condition, held at the current region/resolution.
        for kind in ("uniform", "corner", "euler"):
            run_one(kind, 4.5, 20, seed=0, grid_n=181)
    elif experiment == "B":
        # Region size. Uses the corner-biased start, NOT the tutorial's uniform
        # gray one: experiment A measured that uniform gray + NLopt MMA lands in
        # a half-filled maze on this problem (honest T21 0.55, and only 0.69 even
        # with the beta ramp doubled), so sweeping region size from that start
        # would compare failures rather than designs.
        #
        # The question this sweep asks: every successful run so far converged to
        # a plain smooth bend, because at 4.5um a plain bend IS the optimum
        # (96-97%). Shrinking the footprint should eventually make a plain bend
        # insufficient -- and that is where non-trivial, genuinely
        # topology-optimized geometry has something to win.
        for region in (4.05, 3.0, 2.5, 2.0):
            run_one("corner", region, 20, seed=0, grid_n=grid_for_region(region))
    elif experiment == "C":
        # FDTD resolution (silicon-wavelength sampling). Corner start, not the
        # tutorial's uniform gray one -- experiment A measured that uniform gray
        # fails here, so a resolution study from it would just compare failures.
        # resolution=20 gives only ~10 points per wavelength INSIDE silicon
        # (lambda_Si = 1.35/2.7 = 0.5um), against the usual >=20 rule of thumb
        # and the Tidy3D tutorial's >=50 per vacuum wavelength.
        for res in (30, 40):
            run_one("corner", 4.5, res, seed=0, grid_n=181)
    elif experiment == "D":
        # Minimum-length-scale penalty. Measured first (see
        # docs/simulation_settings_record.md): at the tutorial's own setting
        # (minimum length == filter radius, penalty_eta_e=0.75) our designs
        # already comply by ~15 orders of magnitude -- the conic filter alone
        # satisfies it, so the penalty is inert there and optimizing with it
        # would prove nothing. The informative experiment is to demand a
        # minimum length LARGER than the filter radius -- i.e. "what does a
        # 310nm-minimum-feature fab constraint cost in transmission?" --
        # which is what penalty_eta_e=0.95 encodes.
        for strength in (5e5,):
            run_one("uniform", 4.5, 20, seed=0, grid_n=181,
                    penalty_strength=strength, penalty_eta_e=0.95)
    elif experiment == "E":
        # Binarization pressure. Experiment A found that the uniform-gray start
        # (the Tidy3D tutorial's own choice) DOES produce genuinely freeform
        # structure -- 8 components, 7 enclosed holes -- but never binarizes:
        # 80% of its pixels are still gray after beta=32, leaving a half-filled
        # gradient-index slab that transmits only 0.55 honestly. The standard
        # remedy is more binarization pressure, so extend the beta ramp.
        for schedule in ([4.0, 8.0, 16.0, 32.0, 64.0, 128.0],):
            run_one("uniform", 4.5, 20, seed=0, grid_n=181, beta_schedule=schedule)
    elif experiment == "F":
        # Can optimizer-side tuning lift the 2.0um (1.48 wavelength) design
        # above its measured 0.740? That region is the one that finally
        # produces genuinely topology-optimized geometry (6 components,
        # perimeter/area 0.237 vs a plain bend's 0.121), so it is worth
        # pushing before trading it away for a bigger, duller design region.
        # Knobs, roughly in expected order of impact at this size:
        R = 2.0
        # 1. Filter radius. At a 2.0um region the default 0.2um filter radius
        #    is 10% of the whole design -- it may simply forbid the features a
        #    tight bend needs. (Trade-off: it IS the minimum printable feature.)
        for radius in (0.1, 0.15):
            run_one("corner", R, 20, seed=0, grid_n=81, filter_radius_um=radius)
        # 2. Design-grid density (the "grid count" knob).
        for n in (161,):
            run_one("corner", R, 20, seed=0, grid_n=n)
        # 3. Iteration budget -- saturation was measured at 4.5um, but this is
        #    a harder problem and may still be climbing.
        run_one("corner", R, 20, seed=0, grid_n=81, iters_per_stage=40)
        # 4. MMA's step size (its closest analogue to a learning rate).
        for step in (0.01, 0.05):
            run_one("corner", R, 20, seed=0, grid_n=81, init_step=step)
        # 5. Adam with an actual learning rate, as the Tidy3D tutorial uses.
        for lr in (0.02, 0.05, 0.2):
            run_one("corner", R, 20, seed=0, grid_n=81, optimizer="adam", learning_rate=lr)
        # 6. Different basins -- only seed 0 has been tried at this size.
        for s in (1, 2):
            run_one("corner", R, 20, seed=s, grid_n=81)
    else:
        raise SystemExit(f"unknown experiment {experiment!r}; expected A, B, C, D, E or F")

    print(f"=== experiment {experiment} complete ===", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A")
