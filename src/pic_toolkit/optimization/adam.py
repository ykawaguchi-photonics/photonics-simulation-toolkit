"""Hand-rolled Adam optimizer for the MUX2 stage-1 objective. Gradients via
`jax.value_and_grad`; bounds enforced by clipping immediately after each
raw Adam step (not a projected/mirror-descent step).

A hand-rolled loop (rather than `optax.adam`, which is installed but unused
elsewhere in this repo) keeps the per-iteration state and post-step clipping
fully visible/inspectable, matching this codebase's general style
(`circuits/mzi_lattice.py` favors transparent, heavily-commented numerics
over opaque library calls even where a library exists). `optax.adam` is a
drop-in alternative if preferred later.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np

from .circuit_diff import REFERENCE_ARM_LENGTH_UM


@dataclass
class AdamConfig:
    lr: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_iterations: int = 300
    # delta_L_delay_um bounds MUST stay inside the arm library's own length
    # range (jnp.interp flat-extrapolates silently outside it, which would
    # let Adam "optimize" against untrustworthy extrapolated data).
    bounds_L_upper: tuple = (16.0, 36.0)
    bounds_delta_L_delay: tuple = (14.0, 34.0)
    # L_lower = reference_arm_length_um + delta_L_delay_um -- 26.0 for stage-1's
    # radius_um=5.0 arm footprint (2*lead_len_um+4*radius_um); stage-2's
    # radius_um=2.5 arm footprint is 16.0. Purely a reporting/logging constant --
    # does not affect the optimization itself (that's driven by delta_L_delay_um).
    reference_arm_length_um: float = REFERENCE_ARM_LENGTH_UM


def run_adam(loss_fn, metrics_fn, params0: tuple, config: AdamConfig) -> list[dict]:
    """params0: (L_upper0, delta_L_delay_um0) -- the mandatory current-design
    initial point (never randomized). Returns one dict per iteration
    (iteration 0 = the initial point, before any update)."""
    L_upper, delta_L_delay_um = float(params0[0]), float(params0[1])
    lo_u, hi_u = config.bounds_L_upper
    lo_d, hi_d = config.bounds_delta_L_delay

    value_and_grad = jax.value_and_grad(loss_fn, argnums=(0, 1))

    m_u = m_d = v_u = v_d = 0.0
    rows = []

    for t in range(config.max_iterations + 1):
        loss_val, (grad_u, grad_d) = value_and_grad(L_upper, delta_L_delay_um)
        loss_val, grad_u, grad_d = float(loss_val), float(grad_u), float(grad_d)
        metrics = metrics_fn(L_upper, delta_L_delay_um)

        L_lower = config.reference_arm_length_um + delta_L_delay_um
        rows.append({
            "iteration": t,
            "L_upper": L_upper,
            "delta_L_delay_um": delta_L_delay_um,
            "L_lower": L_lower,
            "Delta_L": L_lower - L_upper,
            "objective": loss_val,
            "grad_L_upper": grad_u,
            "grad_delta_L_delay_um": grad_d,
            **metrics,
        })

        if t == config.max_iterations:
            break

        # Adam update (bias-corrected), t is 0-indexed here so use t+1 for correction.
        m_u = config.beta1 * m_u + (1 - config.beta1) * grad_u
        m_d = config.beta1 * m_d + (1 - config.beta1) * grad_d
        v_u = config.beta2 * v_u + (1 - config.beta2) * grad_u ** 2
        v_d = config.beta2 * v_d + (1 - config.beta2) * grad_d ** 2

        m_u_hat = m_u / (1 - config.beta1 ** (t + 1))
        m_d_hat = m_d / (1 - config.beta1 ** (t + 1))
        v_u_hat = v_u / (1 - config.beta2 ** (t + 1))
        v_d_hat = v_d / (1 - config.beta2 ** (t + 1))

        L_upper = L_upper - config.lr * m_u_hat / (np.sqrt(v_u_hat) + config.epsilon)
        delta_L_delay_um = delta_L_delay_um - config.lr * m_d_hat / (np.sqrt(v_d_hat) + config.epsilon)

        L_upper = float(np.clip(L_upper, lo_u, hi_u))
        delta_L_delay_um = float(np.clip(delta_L_delay_um, lo_d, hi_d))

    return rows
