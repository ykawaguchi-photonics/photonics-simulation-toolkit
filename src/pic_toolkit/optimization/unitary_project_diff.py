"""JAX-differentiable passivity safeguard for the optimization path, playing
the same role `pic_toolkit.circuits.mzi_lattice.unitary_project` plays for
the non-differentiable "real circuit" path: cascading 3+ real (FDTD-
measurement-noisy) components in one `sax.circuit()` was found to violate
passivity by up to ~26% without some correction (see that function's own
docstring for the full investigation).

This is NOT a reimplementation of that function's SVD-based nearest-unitary
projection. An `jnp.linalg.svd`-based version was tried first and is
numerically unstable here: differentiating through SVD blows up when two
singular values are close together (the gradient involves a
`1/(sigma_i^2 - sigma_j^2)` term), and every component this optimization
targets is a well-designed, near-lossless passive device -- i.e. already
close to unitary, which means ALL of its singular values already sit close
to 1 (near-degenerate by construction). Confirmed empirically: `jax.grad`
through the full MUX2 objective returned NaN gradients starting at
iteration 0, specifically because the objective samples wavelengths right
at each channel's transmission PEAK -- exactly where the underlying S-matrix
is closest to exactly unitary (minimum loss), i.e. exactly the worst case
for this instability.

Instead, this enforces the concrete property the passivity investigation
actually cared about (no port emits more power than it received) via a
simple per-row power clamp: `scale_i = 1/sqrt(max(row_power_i, 1))`,
`M'_ij = M_ij * scale_i`. This only shrinks rows whose power already exceeds
1 (leaving already-passive rows untouched) and uses only elementwise
abs/sum/sqrt/divide -- no eigendecomposition, so no degenerate-eigenvalue
gradient blowup. It does not restore exact row-to-row orthogonality the way
the true unitary projection does, but the optimization objective only reads
specific transmission/crosstalk magnitudes, not the full matrix's unitarity,
so this is sufficient for the differentiable path. The exact, numpy-based
`unitary_project` is still used unchanged for every non-differentiable
("real circuit") evaluation in this project, including the final validation.
"""

from __future__ import annotations

import jax.numpy as jnp


def unitary_project_diff(model_fn, port_names: tuple):
    """Wrap a SAX model function (whose output may be plain numpy -- e.g. a
    fixed/non-optimized coupler -- or jnp with traced values, e.g. the
    differentiable arm/waveguide models) so its SDict's per-row power is
    clamped to at most 1, applied per-wavelength-point. See module docstring
    for why this replaces an SVD-based unitary projection here."""

    def wrapped(wl=1.35):
        s = model_fn(wl=wl)
        wl_arr = jnp.atleast_1d(jnp.asarray(wl, dtype=jnp.float32))
        n_wl = wl_arr.shape[0]

        columns = {}
        for p1 in port_names:
            row = []
            for p2 in port_names:
                v = jnp.atleast_1d(jnp.asarray(s.get((p1, p2), 0j), dtype=jnp.complex64))
                v = v if v.shape[0] == n_wl else jnp.full((n_wl,), v[0])
                row.append(v)
            columns[p1] = jnp.stack(row, axis=-1)  # (n_wl, k) -- row p1's entries over all p2

        scalar_in = jnp.ndim(wl) == 0
        out = {}
        for p1 in port_names:
            row = columns[p1]
            row_power = jnp.sum(jnp.abs(row) ** 2, axis=-1)  # (n_wl,)
            scale = 1.0 / jnp.sqrt(jnp.clip(row_power, 1.0, None))
            for j, p2 in enumerate(port_names):
                val = row[:, j] * scale
                out[(p1, p2)] = val[0] if scalar_in else val
        return out

    return wrapped
