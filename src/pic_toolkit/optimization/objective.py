"""Configurable WDM objective for the MUX2 stage-1 lattice filter, evaluated
against `circuit_diff.build_diff_lattice_circuit`'s differentiable surrogate.

Channel center wavelengths are fixed (found once from the initial design,
never re-centered on the current Delta_L during optimization) -- a moving
target would make the objective trivially satisfiable by any Delta_L and
defeat the stated goal of restoring the intended WDM channel wavelengths
despite FDTD-induced drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from . import circuit_diff


@dataclass
class ChannelSpec:
    name: str
    center_wl_um: float
    half_bandwidth_um: float
    n_samples: int
    desired_port: str      # e.g. "out_top" (bar / channel 1)
    undesired_port: str    # e.g. "out_bot" (cross / channel 2)

    def sample_wavelengths_um(self) -> jnp.ndarray:
        if self.n_samples == 1:
            return jnp.array([self.center_wl_um])
        return jnp.linspace(
            self.center_wl_um - self.half_bandwidth_um,
            self.center_wl_um + self.half_bandwidth_um,
            self.n_samples,
        )


@dataclass
class ObjectiveWeights:
    w_transmission: float = 1.0
    w_crosstalk: float = 1.0
    w_insertion_loss: float = 0.5
    w_bandwidth: float = 0.25
    il_ref_db: float = 1.0


def find_channel_centers_um(circuit_fn, wl_scan_um) -> tuple[float, float]:
    """Runs the given circuit over `wl_scan_um` and returns (lambda_1,
    lambda_2): the bar-port and cross-port transmission peak wavelengths,
    via scipy.signal.find_peaks -- same technique `wdm_sax.ipynb` cell 24
    uses. Intended to be called ONCE, at the initial design, to freeze the
    channel targets for the whole optimization."""
    import numpy as np
    from scipy.signal import find_peaks

    wl_np = np.asarray(wl_scan_um)
    S = circuit_fn(wl=jnp.asarray(wl_np))
    bar = np.abs(np.asarray(S[("in_top", "out_top")])) ** 2
    cross = np.abs(np.asarray(S[("in_top", "out_bot")])) ** 2

    bar_peaks, _ = find_peaks(bar, prominence=0.05)
    cross_peaks, _ = find_peaks(cross, prominence=0.05)
    if len(bar_peaks) == 0 or len(cross_peaks) == 0:
        raise RuntimeError(
            "find_channel_centers_um: could not find a bar-port and cross-port peak in "
            f"the scanned range [{wl_np.min()}, {wl_np.max()}]um -- widen wl_scan_um."
        )
    lambda_1 = float(wl_np[bar_peaks[np.argmax(bar[bar_peaks])]])
    lambda_2 = float(wl_np[cross_peaks[np.argmax(cross[cross_peaks])]])
    return lambda_1, lambda_2


def _cluster_peaks(wl_np, values, peak_idx, tol_um: float = 0.002) -> list[float]:
    """Groups peak indices that sit within tol_um of each other into single
    cluster centers (mean wavelength) -- a maximally-flat lattice filter's
    near-flat-top passband registers several adjacent samples as separate
    `find_peaks` detections; without this, downstream nearest-to-target
    selection could lock onto one edge of the flat-top rather than its
    center."""
    if len(peak_idx) == 0:
        return []
    wls = sorted(float(wl_np[i]) for i in peak_idx)
    clusters, current = [], [wls[0]]
    for w in wls[1:]:
        if w - current[-1] <= tol_um:
            current.append(w)
        else:
            clusters.append(current)
            current = [w]
    clusters.append(current)
    return [sum(c) / len(c) for c in clusters]


def ideal_channel_centers_um(kappas: list[float], signs: list[int], delta_L_um: float,
                              n_eff0: float, n_g0: float, wl0_um: float,
                              target_bar_wl_um: float, target_cross_wl_um: float,
                              wl_scan_um=None) -> tuple[float, float]:
    """Derives the two channel target wavelengths (lambda_1=bar, lambda_2=
    cross) from the IDEAL analytic lattice circuit's own periodic peak
    structure -- NOT from wherever the real/FDTD-drifted circuit happens to
    peak (that's what `find_channel_centers_um` above does, and using it to
    set optimization targets was found to anchor the objective to arbitrary
    drift rather than design intent, defeating the point of the
    optimization).

    The lattice filter's bar/cross response is periodic with period FSR
    (`mzi_lattice.delta_L_um_to_fsr`), so multiple valid channel pairs exist
    within any wavelength range spanning more than one FSR. This returns
    the pair nearest the caller-supplied `target_bar_wl_um`/
    `target_cross_wl_um` seed guesses (e.g. from inspecting the ideal
    spectrum once, or a known design intent), not simply the tallest peak
    anywhere in range -- callers pick which periodic repeat they want."""
    import numpy as np

    from ..circuits import mzi_lattice as ml
    from scipy.signal import find_peaks

    if wl_scan_um is None:
        wl_scan_um = np.linspace(1.30, 1.40, 4000)
    wl_np = np.asarray(wl_scan_um)

    circuit_ideal, _ = ml.build_lattice_circuit(kappas, signs, delta_L_um, n_eff0, n_g0, wl0_um)
    S = circuit_ideal(wl=jnp.asarray(wl_np))
    bar = np.abs(np.asarray(S[("in_top", "out_top")])) ** 2
    cross = np.abs(np.asarray(S[("in_top", "out_bot")])) ** 2

    bar_peaks, _ = find_peaks(bar, prominence=0.3)
    cross_peaks, _ = find_peaks(cross, prominence=0.3)
    bar_clusters = _cluster_peaks(wl_np, bar, bar_peaks)
    cross_clusters = _cluster_peaks(wl_np, cross, cross_peaks)
    if not bar_clusters or not cross_clusters:
        raise RuntimeError(
            "ideal_channel_centers_um: could not find a bar-port and cross-port peak in "
            f"the scanned range [{wl_np.min()}, {wl_np.max()}]um -- widen wl_scan_um."
        )
    lambda_1 = min(bar_clusters, key=lambda w: abs(w - target_bar_wl_um))
    lambda_2 = min(cross_clusters, key=lambda w: abs(w - target_cross_wl_um))
    return lambda_1, lambda_2


def ideal_tree_channel_centers_um(kappas: list[float], signs: list[int],
                                   delta_L_1_um: float, delta_L_2_down_um: float,
                                   n_eff0: float, n_g0: float, wl0_um: float,
                                   quarter_wave_shift_up: bool = True,
                                   wl_scan_um=None) -> dict:
    """Derives all 4 MUX4-tree channel target wavelengths (ch1..ch4) from the
    IDEAL 3-stage tree's own periodic response (`mux4_tree.
    build_ideal_mux4_tree_circuit`) -- the tree-level analog of
    `ideal_channel_centers_um` above. Each channel's target is simply its
    OWN tallest peak in the ideal tree spectrum (unlike the single-stage
    case, the tree's per-channel passbands are well-separated by
    construction -- no target-wavelength seed needed to disambiguate a
    periodic repeat)."""
    import numpy as np
    from scipy.signal import find_peaks

    from ..circuits import mux4_tree as mt

    if wl_scan_um is None:
        wl_scan_um = np.linspace(1.30, 1.40, 8000)
    wl_np = np.asarray(wl_scan_um)

    circuit_ideal, _ = mt.build_ideal_mux4_tree_circuit(
        delta_L_1_um, delta_L_2_down_um, n_eff0, n_g0, wl0_um,
        quarter_wave_shift_up=quarter_wave_shift_up, n_couplers=len(kappas),
    )
    S = circuit_ideal(wl=jnp.asarray(wl_np))

    centers = {}
    for ch in ("ch1", "ch2", "ch3", "ch4"):
        power = np.abs(np.asarray(S[("in_top", ch)])) ** 2
        peaks, _ = find_peaks(power, prominence=0.3)
        clusters = _cluster_peaks(wl_np, power, peaks)
        if not clusters:
            raise RuntimeError(f"ideal_tree_channel_centers_um: no peak found for {ch} in "
                                f"[{wl_np.min()}, {wl_np.max()}]um -- widen wl_scan_um.")
        # one dominant passband per channel in this tree -- take the tallest cluster.
        best = max(clusters, key=lambda w: power[int(np.argmin(np.abs(wl_np - w)))])
        centers[ch] = best
    return centers


def make_objective(channels: list[ChannelSpec], weights: ObjectiveWeights, coupling_lengths_um: list[float],
                    arm_library_obj=None):
    """Returns (loss_fn, metrics_fn).

    loss_fn(L_upper, delta_L_delay_um) -> scalar, JAX-differentiable.
    metrics_fn(L_upper, delta_L_delay_um) -> dict of per-channel raw
    transmission/crosstalk/insertion-loss numbers (concrete floats, NOT
    passed through jax.grad) -- for iteration logging.

    Per-channel loss (sampled at N_c wavelengths within +-half_bandwidth_um
    of the channel's fixed center):
        w_T  * mean((1 - T_desired)^2)
      + w_X  * mean(T_undesired^2)
      + w_IL * mean((IL/il_ref_db)^2),  IL = -10*log10(T_desired)
      + w_BW * Var(T_desired)
    """

    def _transmissions(L_upper, delta_L_delay_um, channel: ChannelSpec):
        circuit_fn, _info = circuit_diff.build_diff_lattice_circuit(
            L_upper, delta_L_delay_um, coupling_lengths_um, arm_library_obj=arm_library_obj,
        )
        wl = channel.sample_wavelengths_um()
        S = circuit_fn(wl=wl)
        T_desired = jnp.abs(S[("in_top", channel.desired_port)]) ** 2
        T_undesired = jnp.abs(S[("in_top", channel.undesired_port)]) ** 2
        return T_desired, T_undesired

    def loss_fn(L_upper, delta_L_delay_um):
        total = 0.0
        for channel in channels:
            T_desired, T_undesired = _transmissions(L_upper, delta_L_delay_um, channel)
            T_desired_clamped = jnp.clip(T_desired, 1e-6, None)
            il_db = -10.0 * jnp.log10(T_desired_clamped)

            term_transmission = weights.w_transmission * jnp.mean((1.0 - T_desired) ** 2)
            term_crosstalk = weights.w_crosstalk * jnp.mean(T_undesired ** 2)
            term_il = weights.w_insertion_loss * jnp.mean((il_db / weights.il_ref_db) ** 2)
            term_bandwidth = weights.w_bandwidth * jnp.var(T_desired)

            total = total + term_transmission + term_crosstalk + term_il + term_bandwidth
        return total

    def metrics_fn(L_upper, delta_L_delay_um) -> dict:
        import numpy as np

        out = {}
        for channel in channels:
            T_desired, T_undesired = _transmissions(L_upper, delta_L_delay_um, channel)
            T_desired_np, T_undesired_np = np.asarray(T_desired), np.asarray(T_undesired)
            il_db = -10.0 * np.log10(np.clip(T_desired_np, 1e-9, None))
            out[f"{channel.name}_transmission"] = float(np.mean(T_desired_np))
            out[f"{channel.name}_crosstalk"] = float(np.mean(T_undesired_np))
            out[f"{channel.name}_insertion_loss_db"] = float(np.mean(il_db))
            out[f"{channel.name}_flatness_var"] = float(np.var(T_desired_np))
        return out

    return loss_fn, metrics_fn


def make_objective_with_couplers(channels: list[ChannelSpec], weights: ObjectiveWeights,
                                  n_couplers: int, arm_library_obj=None, coupler_library_obj=None):
    """Coupler-inclusive twin of `make_objective`: `loss_fn`/`metrics_fn` take
    `(L_upper, delta_L_delay_um, Lc_0, ..., Lc_{n_couplers-1})` -- each
    coupler length is now an independent differentiable variable (via
    `circuit_diff.build_diff_lattice_circuit_variable_couplers`), not a fixed
    closed-over list. Same per-channel loss formula as `make_objective`."""

    def _transmissions(L_upper, delta_L_delay_um, coupling_lengths_um, channel: ChannelSpec):
        circuit_fn, _info = circuit_diff.build_diff_lattice_circuit_variable_couplers(
            L_upper, delta_L_delay_um, coupling_lengths_um,
            arm_library_obj=arm_library_obj, coupler_library_obj=coupler_library_obj,
        )
        wl = channel.sample_wavelengths_um()
        S = circuit_fn(wl=wl)
        T_desired = jnp.abs(S[("in_top", channel.desired_port)]) ** 2
        T_undesired = jnp.abs(S[("in_top", channel.undesired_port)]) ** 2
        return T_desired, T_undesired

    def loss_fn(L_upper, delta_L_delay_um, *Lc_vars):
        assert len(Lc_vars) == n_couplers
        total = 0.0
        for channel in channels:
            T_desired, T_undesired = _transmissions(L_upper, delta_L_delay_um, list(Lc_vars), channel)
            T_desired_clamped = jnp.clip(T_desired, 1e-6, None)
            il_db = -10.0 * jnp.log10(T_desired_clamped)

            term_transmission = weights.w_transmission * jnp.mean((1.0 - T_desired) ** 2)
            term_crosstalk = weights.w_crosstalk * jnp.mean(T_undesired ** 2)
            term_il = weights.w_insertion_loss * jnp.mean((il_db / weights.il_ref_db) ** 2)
            term_bandwidth = weights.w_bandwidth * jnp.var(T_desired)

            total = total + term_transmission + term_crosstalk + term_il + term_bandwidth
        return total

    def metrics_fn(L_upper, delta_L_delay_um, *Lc_vars) -> dict:
        import numpy as np

        assert len(Lc_vars) == n_couplers
        out = {}
        for channel in channels:
            T_desired, T_undesired = _transmissions(L_upper, delta_L_delay_um, list(Lc_vars), channel)
            T_desired_np, T_undesired_np = np.asarray(T_desired), np.asarray(T_undesired)
            il_db = -10.0 * np.log10(np.clip(T_desired_np, 1e-9, None))
            out[f"{channel.name}_transmission"] = float(np.mean(T_desired_np))
            out[f"{channel.name}_crosstalk"] = float(np.mean(T_undesired_np))
            out[f"{channel.name}_insertion_loss_db"] = float(np.mean(il_db))
            out[f"{channel.name}_flatness_var"] = float(np.var(T_desired_np))
        return out

    return loss_fn, metrics_fn


def full_spectrum_ideal_bar(kappas: list[float], signs: list[int], delta_L_um: float,
                             n_eff0: float, n_g0: float, wl0_um: float, wl_grid_um) -> "np.ndarray":
    """Precomputes the ideal analytic design's bar-port power spectrum over
    `wl_grid_um` -- the fixed target `make_full_spectrum_objective` matches
    against. A plain numpy array, not JAX-traced: the ideal model depends
    only on the fixed design constants (kappas/signs/delta_L_um/n_eff0/n_g0/
    wl0_um), never on the optimization variables, so this only needs
    computing once, outside the loss function."""
    import numpy as np

    from ..circuits import mzi_lattice as ml

    circuit_ideal, _ = ml.build_lattice_circuit(kappas, signs, delta_L_um, n_eff0, n_g0, wl0_um)
    S = circuit_ideal(wl=jnp.asarray(wl_grid_um))
    return np.abs(np.asarray(S[("in_top", "out_top")])) ** 2


def make_full_spectrum_objective(n_couplers: int, wl_grid_um, ideal_bar_spectrum,
                                  arm_library_obj=None, coupler_library_obj=None):
    """Whole-band RMS-deviation objective: matches the real/FDTD-surrogate
    circuit's bar-port transmission to `ideal_bar_spectrum` (see
    `full_spectrum_ideal_bar`) at EVERY wavelength in `wl_grid_um`, not just
    narrow windows around each channel's target peak
    (`make_objective`/`make_objective_with_couplers`'s `ChannelSpec`
    windows). A peak-window objective can score well right at the target
    wavelengths while the rest of the passband stays mismatched; this
    objective can't be satisfied that way -- it only improves by making the
    real spectrum's whole shape track the ideal design's shape.

    Same `(L_upper, delta_L_delay_um, *Lc_vars)` calling convention as
    `make_objective_with_couplers` -- drop-in compatible with
    `adam_nd.run_adam_nd`/`lbfgs.run_lbfgs`/`lbfgs.run_lbfgs_multistart`,
    no changes needed there. Builds the real circuit via `circuit_diff.
    build_diff_lattice_circuit_variable_couplers` ONCE per evaluation (a
    single `wl_grid_um` array), cheaper per call than the channel-window
    objectives' one-circuit-build-per-channel loop.

    `loss_fn` returns the MSE (mean squared deviation), not the RMS itself
    -- minimizing MSE is equivalent to minimizing RMS (sqrt is monotonic)
    and avoids a sqrt-near-zero gradient wrinkle once the match is close;
    `metrics_fn` reports the actual `rms_deviation` (linear power units) for
    readability, plus `max_abs_deviation`."""
    ideal_bar_spectrum_jnp = jnp.asarray(ideal_bar_spectrum)

    def _bar(L_upper, delta_L_delay_um, coupling_lengths_um):
        circuit_fn, _info = circuit_diff.build_diff_lattice_circuit_variable_couplers(
            L_upper, delta_L_delay_um, coupling_lengths_um,
            arm_library_obj=arm_library_obj, coupler_library_obj=coupler_library_obj,
        )
        S = circuit_fn(wl=wl_grid_um)
        return jnp.abs(S[("in_top", "out_top")]) ** 2

    def loss_fn(L_upper, delta_L_delay_um, *Lc_vars):
        assert len(Lc_vars) == n_couplers
        bar_real = _bar(L_upper, delta_L_delay_um, list(Lc_vars))
        return jnp.mean((bar_real - ideal_bar_spectrum_jnp) ** 2)

    def metrics_fn(L_upper, delta_L_delay_um, *Lc_vars) -> dict:
        import numpy as np

        assert len(Lc_vars) == n_couplers
        bar_real = np.asarray(_bar(L_upper, delta_L_delay_um, list(Lc_vars)))
        diff = bar_real - np.asarray(ideal_bar_spectrum)
        return {
            "rms_deviation": float(np.sqrt(np.mean(diff ** 2))),
            "max_abs_deviation": float(np.max(np.abs(diff))),
            "mean_bar": float(np.mean(bar_real)),
        }

    return loss_fn, metrics_fn


def full_spectrum_ideal_bar_cross(kappas: list[float], signs: list[int], delta_L_um: float,
                                   n_eff0: float, n_g0: float, wl0_um: float, wl_grid_um) -> tuple:
    """Twin of `full_spectrum_ideal_bar` that also returns the ideal cross-port
    power spectrum -- the fixed target `make_full_spectrum_objective_2d`
    matches both ports against. Builds the ideal circuit once and reads off
    both S-entries; still a plain numpy result, not JAX-traced, for the same
    reason `full_spectrum_ideal_bar` is."""
    import numpy as np

    from ..circuits import mzi_lattice as ml

    circuit_ideal, _ = ml.build_lattice_circuit(kappas, signs, delta_L_um, n_eff0, n_g0, wl0_um)
    S = circuit_ideal(wl=jnp.asarray(wl_grid_um))
    bar = np.abs(np.asarray(S[("in_top", "out_top")])) ** 2
    cross = np.abs(np.asarray(S[("in_top", "out_bot")])) ** 2
    return bar, cross


def make_full_spectrum_objective_2d(n_couplers: int, wl_grid_um, ideal_bar_spectrum, ideal_cross_spectrum,
                                     arm_library_obj=None, coupler_gap_library_obj=None,
                                     w_bar: float = 1.0, w_cross: float = 1.0):
    """2D (gap+length) + bar-and-cross twin of `make_full_spectrum_objective`:
    matches the real/FDTD-surrogate circuit's BOTH bar-port and cross-port
    power to `ideal_bar_spectrum`/`ideal_cross_spectrum` (see
    `full_spectrum_ideal_bar_cross`) at EVERY wavelength in `wl_grid_um`,
    using the 2D coupler surrogate (`circuit_diff.
    build_diff_lattice_circuit_variable_couplers_2d`) so gap is a free
    variable per coupler alongside length -- same search space stage 1's
    `make_objective_with_couplers_2d` used, but scored against the whole
    spectral shape instead of narrow per-channel windows.

    Same `(L_upper, delta_L_delay_um, Lc_0, ..., Lc_{n-1}, gap_0, ...,
    gap_{n-1})` calling convention as `make_objective_with_couplers_2d` --
    drop-in compatible with `lbfgs.run_lbfgs_multistart`.

    `loss_fn` returns `w_bar * MSE(bar) + w_cross * MSE(cross)` (MSE, not
    RMS, for the same near-zero-gradient reason `make_full_spectrum_objective`
    avoids the sqrt); `metrics_fn` reports the actual `rms_deviation_bar`/
    `rms_deviation_cross` (linear power units) plus `max_abs_deviation_bar`/
    `max_abs_deviation_cross` for readability."""
    ideal_bar_jnp = jnp.asarray(ideal_bar_spectrum)
    ideal_cross_jnp = jnp.asarray(ideal_cross_spectrum)

    def _bar_cross(L_upper, delta_L_delay_um, coupling_lengths_um, gaps_um):
        circuit_fn, _info = circuit_diff.build_diff_lattice_circuit_variable_couplers_2d(
            L_upper, delta_L_delay_um, coupling_lengths_um, gaps_um,
            arm_library_obj=arm_library_obj, coupler_gap_library_obj=coupler_gap_library_obj,
        )
        S = circuit_fn(wl=wl_grid_um)
        bar = jnp.abs(S[("in_top", "out_top")]) ** 2
        cross = jnp.abs(S[("in_top", "out_bot")]) ** 2
        return bar, cross

    def loss_fn(L_upper, delta_L_delay_um, *rest):
        assert len(rest) == 2 * n_couplers
        Lc_vars, gap_vars = list(rest[:n_couplers]), list(rest[n_couplers:])
        bar_real, cross_real = _bar_cross(L_upper, delta_L_delay_um, Lc_vars, gap_vars)
        return (w_bar * jnp.mean((bar_real - ideal_bar_jnp) ** 2)
                + w_cross * jnp.mean((cross_real - ideal_cross_jnp) ** 2))

    def metrics_fn(L_upper, delta_L_delay_um, *rest) -> dict:
        import numpy as np

        assert len(rest) == 2 * n_couplers
        Lc_vars, gap_vars = list(rest[:n_couplers]), list(rest[n_couplers:])
        bar_real, cross_real = _bar_cross(L_upper, delta_L_delay_um, Lc_vars, gap_vars)
        bar_real, cross_real = np.asarray(bar_real), np.asarray(cross_real)
        diff_bar = bar_real - np.asarray(ideal_bar_spectrum)
        diff_cross = cross_real - np.asarray(ideal_cross_spectrum)
        return {
            "rms_deviation_bar": float(np.sqrt(np.mean(diff_bar ** 2))),
            "rms_deviation_cross": float(np.sqrt(np.mean(diff_cross ** 2))),
            "max_abs_deviation_bar": float(np.max(np.abs(diff_bar))),
            "max_abs_deviation_cross": float(np.max(np.abs(diff_cross))),
            "mean_bar": float(np.mean(bar_real)),
            "mean_cross": float(np.mean(cross_real)),
        }

    return loss_fn, metrics_fn


def make_objective_with_couplers_and_arms(channels: list[ChannelSpec], weights: ObjectiveWeights,
                                           n_couplers: int, arm_library_obj=None, coupler_library_obj=None):
    """Arm-inclusive twin of `make_objective_with_couplers`: `loss_fn`/
    `metrics_fn` take `(L_upper, delta_L_0, ..., delta_L_{n_couplers-2},
    Lc_0, ..., Lc_{n_couplers-1})` -- each arm-PAIR now gets its own
    independent `delta_L` (via
    `circuit_diff.build_diff_lattice_circuit_variable_couplers_and_arms`)
    instead of one shared `delta_L_delay_um`, on top of each coupler's own
    independent length. Same per-channel loss formula as `make_objective`/
    `make_objective_with_couplers`. See
    `build_diff_lattice_circuit_variable_couplers_and_arms`'s docstring for
    why this extra per-stage freedom exists (an empirical test of whether a
    differential phase bias can substitute for a coupler sign no length
    tuning alone can reach)."""
    n_arms = n_couplers - 1

    def _transmissions(L_upper, delta_L_list, coupling_lengths_um, channel: ChannelSpec):
        circuit_fn, _info = circuit_diff.build_diff_lattice_circuit_variable_couplers_and_arms(
            L_upper, delta_L_list, coupling_lengths_um,
            arm_library_obj=arm_library_obj, coupler_library_obj=coupler_library_obj,
        )
        wl = channel.sample_wavelengths_um()
        S = circuit_fn(wl=wl)
        T_desired = jnp.abs(S[("in_top", channel.desired_port)]) ** 2
        T_undesired = jnp.abs(S[("in_top", channel.undesired_port)]) ** 2
        return T_desired, T_undesired

    def loss_fn(L_upper, *rest):
        assert len(rest) == n_arms + n_couplers
        delta_L_list, Lc_vars = list(rest[:n_arms]), list(rest[n_arms:])
        total = 0.0
        for channel in channels:
            T_desired, T_undesired = _transmissions(L_upper, delta_L_list, Lc_vars, channel)
            T_desired_clamped = jnp.clip(T_desired, 1e-6, None)
            il_db = -10.0 * jnp.log10(T_desired_clamped)

            term_transmission = weights.w_transmission * jnp.mean((1.0 - T_desired) ** 2)
            term_crosstalk = weights.w_crosstalk * jnp.mean(T_undesired ** 2)
            term_il = weights.w_insertion_loss * jnp.mean((il_db / weights.il_ref_db) ** 2)
            term_bandwidth = weights.w_bandwidth * jnp.var(T_desired)

            total = total + term_transmission + term_crosstalk + term_il + term_bandwidth
        return total

    def metrics_fn(L_upper, *rest) -> dict:
        import numpy as np

        assert len(rest) == n_arms + n_couplers
        delta_L_list, Lc_vars = list(rest[:n_arms]), list(rest[n_arms:])
        out = {}
        for channel in channels:
            T_desired, T_undesired = _transmissions(L_upper, delta_L_list, Lc_vars, channel)
            T_desired_np, T_undesired_np = np.asarray(T_desired), np.asarray(T_undesired)
            il_db = -10.0 * np.log10(np.clip(T_desired_np, 1e-9, None))
            out[f"{channel.name}_transmission"] = float(np.mean(T_desired_np))
            out[f"{channel.name}_crosstalk"] = float(np.mean(T_undesired_np))
            out[f"{channel.name}_insertion_loss_db"] = float(np.mean(il_db))
            out[f"{channel.name}_flatness_var"] = float(np.var(T_desired_np))
        return out

    return loss_fn, metrics_fn


def make_objective_with_couplers_2d(channels: list[ChannelSpec], weights: ObjectiveWeights,
                                     n_couplers: int, arm_library_obj=None, coupler_gap_library_obj=None):
    """Gap-and-length-inclusive twin of `make_objective_with_couplers`:
    `loss_fn`/`metrics_fn` take `(L_upper, delta_L_delay_um, Lc_0, ...,
    Lc_{n_couplers-1}, gap_0, ..., gap_{n_couplers-1})` -- each coupler now
    has BOTH its length AND its gap as independent differentiable variables
    (via `circuit_diff.build_diff_lattice_circuit_variable_couplers_2d` /
    `coupler_library_2d`'s 2D interpolation), instead of a fixed gap=0.2um.
    Same per-channel loss formula as `make_objective_with_couplers`. See
    `build_diff_lattice_circuit_variable_couplers_2d`'s docstring for why
    this extra freedom exists (the 2D coupler sweep found dispersion spikes
    sharply near each gap's own coupling extremum, and the production
    design's `Lc=26um` coupler happens to sit almost exactly on `gap=0.2`'s
    own extremum -- a different (gap, length) pair reaching the same target
    kappa away from any extremum was shown, by hand, to cut that dispersion
    residual ~380x; this lets the optimizer search for such combinations
    directly instead of only trying the ones a human already found)."""

    def _transmissions(L_upper, delta_L_delay_um, coupling_lengths_um, gaps_um, channel: ChannelSpec):
        circuit_fn, _info = circuit_diff.build_diff_lattice_circuit_variable_couplers_2d(
            L_upper, delta_L_delay_um, coupling_lengths_um, gaps_um,
            arm_library_obj=arm_library_obj, coupler_gap_library_obj=coupler_gap_library_obj,
        )
        wl = channel.sample_wavelengths_um()
        S = circuit_fn(wl=wl)
        T_desired = jnp.abs(S[("in_top", channel.desired_port)]) ** 2
        T_undesired = jnp.abs(S[("in_top", channel.undesired_port)]) ** 2
        return T_desired, T_undesired

    def loss_fn(L_upper, delta_L_delay_um, *rest):
        assert len(rest) == 2 * n_couplers
        Lc_vars, gap_vars = list(rest[:n_couplers]), list(rest[n_couplers:])
        total = 0.0
        for channel in channels:
            T_desired, T_undesired = _transmissions(L_upper, delta_L_delay_um, Lc_vars, gap_vars, channel)
            T_desired_clamped = jnp.clip(T_desired, 1e-6, None)
            il_db = -10.0 * jnp.log10(T_desired_clamped)

            term_transmission = weights.w_transmission * jnp.mean((1.0 - T_desired) ** 2)
            term_crosstalk = weights.w_crosstalk * jnp.mean(T_undesired ** 2)
            term_il = weights.w_insertion_loss * jnp.mean((il_db / weights.il_ref_db) ** 2)
            term_bandwidth = weights.w_bandwidth * jnp.var(T_desired)

            total = total + term_transmission + term_crosstalk + term_il + term_bandwidth
        return total

    def metrics_fn(L_upper, delta_L_delay_um, *rest) -> dict:
        import numpy as np

        assert len(rest) == 2 * n_couplers
        Lc_vars, gap_vars = list(rest[:n_couplers]), list(rest[n_couplers:])
        out = {}
        for channel in channels:
            T_desired, T_undesired = _transmissions(L_upper, delta_L_delay_um, Lc_vars, gap_vars, channel)
            T_desired_np, T_undesired_np = np.asarray(T_desired), np.asarray(T_undesired)
            il_db = -10.0 * np.log10(np.clip(T_desired_np, 1e-9, None))
            out[f"{channel.name}_transmission"] = float(np.mean(T_desired_np))
            out[f"{channel.name}_crosstalk"] = float(np.mean(T_undesired_np))
            out[f"{channel.name}_insertion_loss_db"] = float(np.mean(il_db))
            out[f"{channel.name}_flatness_var"] = float(np.var(T_desired_np))
        return out

    return loss_fn, metrics_fn
