"""Differentiable (JAX/adjoint) optimization layer for MUX2 stage-1 arm
lengths -- L_upper (reference arm) and L_lower (delay arm).

Architectural rule: everything here is meep-free (same rule as
`pic_toolkit.models`), and every function whose output depends on
`L_upper`/`delta_L_delay_um` uses `jax.numpy`, not plain `numpy`, so
`jax.grad`/`jax.value_and_grad` can differentiate through it. This is a new
layer alongside `pic_toolkit.models`/`pic_toolkit.circuits` -- it does not
modify either.
"""
