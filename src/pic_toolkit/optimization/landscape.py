"""2D objective-landscape evaluation over (L_upper, delta_L_delay_um), for
the diagnostic contour plot -- `jax.vmap` over a meshgrid (loss_fn is pure
JAX, so this vmaps cleanly instead of a slow Python double loop)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def evaluate_grid(loss_fn, L_upper_range: tuple, delta_L_delay_range: tuple,
                   n_upper: int = 41, n_lower: int = 41) -> dict:
    """L_upper_range/delta_L_delay_range: (min, max) tuples. Returns
    {"L_upper": (n_upper,), "delta_L_delay_um": (n_lower,), "loss": (n_upper,
    n_lower)}."""
    L_upper_vals = jnp.linspace(*L_upper_range, n_upper)
    delta_L_vals = jnp.linspace(*delta_L_delay_range, n_lower)

    grid_fn = jax.vmap(jax.vmap(loss_fn, in_axes=(None, 0)), in_axes=(0, None))
    loss_grid = grid_fn(L_upper_vals, delta_L_vals)

    return {
        "L_upper": np.asarray(L_upper_vals),
        "delta_L_delay_um": np.asarray(delta_L_vals),
        "loss": np.asarray(loss_grid),
    }
