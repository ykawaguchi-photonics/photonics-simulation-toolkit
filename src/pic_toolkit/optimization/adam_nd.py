"""N-parameter generalization of `adam.py`'s hand-rolled Adam -- needed once
coupler lengths join `L_upper`/`delta_L_delay_um` as decision variables
(`adam.py`'s `run_adam` is hardcoded to exactly 2 parameters). Kept as a
separate module rather than rewriting `adam.py` in place, so the already-
validated stage-1/stage-2 notebooks (built against the 2-parameter
`run_adam`/`AdamConfig`) keep working unmodified -- same "duplication over a
risky shared rewrite" convention used elsewhere in this optimization layer.

Same algorithm as `adam.py`: hand-rolled bias-corrected Adam, `jnp.clip` to
bounds immediately after each step, one row of history per iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import numpy as np


@dataclass
class AdamConfigND:
    lr: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_iterations: int = 300
    bounds: list = field(default_factory=list)      # list of (lo, hi), one per parameter
    param_names: list = field(default_factory=list)  # for readable logging, e.g. ["L_upper", "delta_L_delay_um", "Lc0", ...]


def run_adam_nd(loss_fn, metrics_fn, params0: tuple, config: AdamConfigND) -> list[dict]:
    """params0: initial parameter vector (mandatory current-design values,
    never randomized), same length as `config.bounds`/`config.param_names`.
    `loss_fn`/`metrics_fn` are called as `fn(*params)` (N positional args) --
    this is exactly how `objective.make_objective_with_couplers`'s
    `loss_fn(L_upper, delta_L_delay_um, *Lc_vars)` is already shaped, so no
    adapter is needed. Returns one dict per iteration (iteration 0 = the
    initial point, before any update)."""
    n = len(params0)
    assert len(config.bounds) == n, "bounds must have one (lo, hi) entry per parameter"
    names = config.param_names or [f"theta{i}" for i in range(n)]

    params = [float(p) for p in params0]
    m = [0.0] * n
    v = [0.0] * n

    value_and_grad = jax.value_and_grad(loss_fn, argnums=tuple(range(n)))
    rows = []

    for t in range(config.max_iterations + 1):
        loss_val, grads = value_and_grad(*params)
        loss_val = float(loss_val)
        grads = [float(g) for g in grads]
        metrics = metrics_fn(*params)

        row = {"iteration": t, "objective": loss_val}
        for i, name in enumerate(names):
            row[name] = params[i]
            row[f"grad_{name}"] = grads[i]
        row.update(metrics)
        rows.append(row)

        if t == config.max_iterations:
            break

        for i in range(n):
            m[i] = config.beta1 * m[i] + (1 - config.beta1) * grads[i]
            v[i] = config.beta2 * v[i] + (1 - config.beta2) * grads[i] ** 2
            m_hat = m[i] / (1 - config.beta1 ** (t + 1))
            v_hat = v[i] / (1 - config.beta2 ** (t + 1))
            params[i] = params[i] - config.lr * m_hat / (np.sqrt(v_hat) + config.epsilon)
            lo, hi = config.bounds[i]
            params[i] = float(np.clip(params[i], lo, hi))

    return rows
