"""L-BFGS-B optimizer for the WDM lattice objectives -- an alternative to
`adam.py`/`adam_nd.py`'s hand-rolled first-order Adam.

Adam-vs-L-BFGS-B is not a style preference here: this optimization problem
(a handful of scalar geometry parameters, a smooth loss, EXACT gradients via
JAX autodiff through a purely interpolation-based surrogate -- no
minibatch/measurement noise anywhere) is exactly the regime quasi-Newton
methods are built for, and an earlier 1000-iteration Adam run on this same
objective showed "a slow, genuine residual drift... a shallow-landscape
effect" late in the run -- the kind of slow terminal convergence L-BFGS's
curvature estimate is designed to fix (see `docs/simulation_settings_record.md`'s
"`circuits/` MUX2 (N=4 lattice) geometry-optimization program" section).

Uses `scipy.optimize.minimize(method="L-BFGS-B")` (scipy is already a
dependency via `scipy.signal.find_peaks` elsewhere in this optimization
layer) driven by `jax.value_and_grad`, rather than a hand-rolled quasi-Newton
loop -- unlike Adam's simple per-iteration update, L-BFGS-B's own curvature
bookkeeping (the two-loop recursion, Wolfe line search) is exactly the kind
of easy-to-get-subtly-wrong logic worth getting from a well-tested library
instead of hand-rolling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import numpy as np
from scipy.optimize import minimize


@dataclass
class LBFGSConfig:
    max_iterations: int = 300
    bounds: list = field(default_factory=list)       # list of (lo, hi), one per parameter
    param_names: list = field(default_factory=list)  # for readable logging, e.g. ["L_upper", "delta_L_delay_um", "Lc0", ...]
    ftol: float = 1e-12
    gtol: float = 1e-10


def run_lbfgs(loss_fn, metrics_fn, params0: tuple, config: LBFGSConfig) -> list[dict]:
    """Same calling convention as `adam_nd.run_adam_nd`: `loss_fn`/
    `metrics_fn` are called as `fn(*params)` (N positional args), so no
    adapter is needed for `objective.make_objective_with_couplers`/
    `make_objective_with_couplers_and_arms`. Returns one dict per outer
    L-BFGS-B step (row 0 = the initial point, before any update) in the same
    shape `run_adam_nd` returns, so both optimizers' histories can be plotted
    with the same code."""
    n = len(params0)
    assert len(config.bounds) == n, "bounds must have one (lo, hi) entry per parameter"
    names = config.param_names or [f"theta{i}" for i in range(n)]

    value_and_grad = jax.value_and_grad(lambda p: loss_fn(*p))
    rows = []

    def _record(params_vec):
        params = [float(p) for p in params_vec]
        loss_val, grads = value_and_grad(np.asarray(params))
        metrics = metrics_fn(*params)
        row = {"iteration": len(rows), "objective": float(loss_val)}
        for i, name in enumerate(names):
            row[name] = params[i]
            row[f"grad_{name}"] = float(grads[i])
        row.update(metrics)
        rows.append(row)

    def _fun(params_vec):
        loss_val, grads = value_and_grad(np.asarray(params_vec))
        return float(loss_val), np.asarray(grads, dtype=np.float64)

    _record(np.asarray(params0, dtype=np.float64))
    result = minimize(
        _fun, x0=np.asarray(params0, dtype=np.float64), jac=True, method="L-BFGS-B",
        bounds=config.bounds, callback=lambda xk: _record(xk),
        options={"maxiter": config.max_iterations, "ftol": config.ftol, "gtol": config.gtol},
    )
    last_x = np.array([rows[-1][name] for name in names])
    if not np.allclose(last_x, result.x, atol=1e-9):
        # scipy's callback is not guaranteed across versions to fire on the exact
        # final accepted point -- append it explicitly if the last recorded row
        # doesn't already match, so the returned history always ends where
        # `result.x` actually landed.
        _record(result.x)
    return rows


def sample_multistart_points(params0: tuple, bounds: list, n_starts: int, seed: int = 0) -> list[tuple]:
    """`params0` (the mandatory current-design point) is always included
    first -- matching `adam.py`/`adam_nd.py`'s own "never randomize the
    initial point" convention -- followed by `n_starts - 1` additional
    points drawn uniformly within `bounds`."""
    assert n_starts >= 1
    rng = np.random.default_rng(seed)
    starts = [tuple(float(p) for p in params0)]
    for _ in range(n_starts - 1):
        starts.append(tuple(float(rng.uniform(lo, hi)) for lo, hi in bounds))
    return starts


def run_lbfgs_multistart(loss_fn, metrics_fn, params0_list: list[tuple], config: LBFGSConfig) -> dict:
    """Runs `run_lbfgs` independently from each point in `params0_list` and
    returns the run with the lowest final objective, alongside every run's
    own final loss -- a lightweight guard against the shallow-landscape
    local-minima risk noted in stage1's own commentary: a single optimizer
    run from one starting point has no way to tell whether it landed in a
    shallow local trap or the true optimum."""
    runs = [run_lbfgs(loss_fn, metrics_fn, params0, config) for params0 in params0_list]
    final_objectives = [rows[-1]["objective"] for rows in runs]
    best_idx = int(np.argmin(final_objectives))
    return {
        "best_rows": runs[best_idx],
        "best_start_index": best_idx,
        "all_final_objectives": final_objectives,
        "all_runs": runs,
    }
