"""JAX-differentiable 2D (gap_um, coupling_length_um) coupler interpolation
surrogate -- extends `coupler_library.py`'s existing 1D (length-only, fixed
gap=0.2) surrogate to a genuine second free variable, using the new
`data/sparams/coupler/sweep_2d/` grid (gaps 0.08-0.35um).

Architectural rule: meep-free, matching `coupler_library.py`'s own rule.

Design: reuses `coupler_library.py`'s existing, already-validated per-gap
length+wavelength de-winding/interpolation UNCHANGED -- one
`CouplerLengthLibrary` per characterized gap value (built via
`coupler_library.load_coupler_length_library(library_dirs=..., gap_um_
filter=...)`, which is what that module's own gap-filtering extension exists
for). This module only adds ONE new outer interpolation stage, across the
small, discrete set of characterized gaps, via `jnp.interp` on the resulting
per-gap values -- exactly the same "amplitude+phase, not real/imag" pattern
already used for the length and wavelength axes, since a naive real/imag
interpolation was already found (in `coupler_library.py`'s own development)
to underestimate magnitude wherever phase winds quickly.

Why NOT a single joint (gap, length) 2D de-winding pass instead: the gap
sweep grid is RAGGED, not rectangular -- gaps 0.08/0.10/0.12 use a denser,
shorter length axis (they oscillate over much shorter lengths, confirmed
empirically: gap=0.08 already sits past its own coupling peak by
`Lc=10um`) than gaps 0.15-0.35's original 7-point, 2-26um axis. Reusing the
per-gap 1D loader sidesteps ever needing a common length grid across gaps;
each gap's own length axis, and its own already-validated de-winding trend
multiplier, stays exactly as `coupler_library.py` computed it.

Whether phase needs de-winding across the GAP axis too (as it clearly does
across length) is checked empirically, not assumed: this module's own
`gap_phase_dewinding_report()` measures how much each S-entry's absolute
phase (at a fixed representative length/wavelength) actually varies across
the characterized gap axis, before deciding whether `jnp.unwrap` alone
(cheap, no trend-fitting) is sufficient or a fuller de-winding pass
(mirroring the length-axis one) would be needed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

assert "meep" not in sys.modules, "pic_toolkit.optimization.coupler_library_2d must never coexist with a meep import"

from . import coupler_library as cl  # noqa: E402  (after the meep-free assert, matching coupler_library.py's own ordering)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SWEEP_1D_DIR = _REPO_ROOT / "data" / "sparams" / "coupler" / "sweep"       # gap=0.20 only (implicit)
_SWEEP_2D_DIR = _REPO_ROOT / "data" / "sparams" / "coupler" / "sweep_2d"    # gaps 0.08-0.35 (explicit gap_um)
_DEDUP_TOL_GAP_UM = 1e-6


@dataclass
class CouplerGapLengthLibrary:
    gap_um: np.ndarray                 # sorted ascending, shape (n_gap,)
    per_gap: dict                      # {gap_um_value: coupler_library.CouplerLengthLibrary}
    port_names: tuple


def _discover_gaps() -> list[float]:
    """Every distinct gap_um present across both sweep directories -- 0.20
    from the original single-gap sweep/ (implicit, no gap_um in its own
    metadata, defaulted by `coupler_library._load_raw_artifact`), plus every
    explicit gap_um recorded in sweep_2d/'s own artifact metadata."""
    import json

    gaps = {0.20}
    for npz_path in sorted(_SWEEP_2D_DIR.glob("*.npz")):
        stem = Path(str(npz_path)[: -len(".npz")])
        json_path = stem.parent / (stem.name + ".json")
        if not json_path.exists():
            continue
        meta = json.loads(json_path.read_text())
        if "gap_um" in meta:
            gaps.add(float(meta["gap_um"]))
    gaps -= _EXCLUDED_GAPS
    return sorted(gaps)


# gap_um=0.08 and 0.10 are excluded from auto-discovery (not just truncated, unlike
# _MAX_LENGTH_BY_GAP's tighter-but-usable truncation above): at BOTH gaps, S_through's
# phase vs. length isn't a single, slowly-varying trend the way it is for a plain arm or
# for the wider gaps (0.12-0.35) that never complete a full coupling oscillation within
# the swept range -- it's the phase of a SUM of two different-propagation-constant
# supermodes (even/odd), which genuinely produces fast, near-discontinuous phase jumps
# right at the beat nulls, confirmed empirically: even 1um-step sampling (already 4x
# denser than the original 4um-step sweep) still hit residual jumps pegged at ~180deg
# (the fundamental np.unwrap ambiguity point) for EVERY candidate trend multiplier, at
# multiple different (length, wavelength) locations -- not a sampling-density problem
# fixable by yet finer steps alone, but a real limitation of this module's single-linear-
# trend de-winding model for a two-supermode-interference S-parameter. gap_um=0.12 and
# 0.15 ALSO turn over within the swept range but were successfully de-wound once
# densified to a uniform ~1um step -- the beat effect there is evidently gentler/slower.
# A proper fix (tracking the two supermodes' own phases separately, rather than S_through/
# S_cross's magnitude+phase) is future work, not attempted here.
_EXCLUDED_GAPS = {0.08, 0.10}


def load_coupler_gap_length_library(gaps: list[float] | None = None) -> CouplerGapLengthLibrary:
    """Loads one `CouplerLengthLibrary` per characterized gap (auto-
    discovered from `sweep_2d/`'s own metadata, plus the always-present
    gap=0.20 from the original `sweep/` directory) via `coupler_library.
    load_coupler_length_library`'s existing, unmodified per-gap loading
    path. `gaps`, if given, restricts to a subset (e.g. while a sweep is
    still in progress and only some gaps are fully populated yet).

    `_MAX_LENGTH_BY_GAP` truncates the tightest gaps (0.08/0.10um) to their
    first coupling-oscillation period only -- confirmed empirically (see
    `coupler_library.load_coupler_length_library`'s own `max_coupling_
    length_um` docstring) that kappa completes a full 0->1->0 cycle and
    starts a second one within the swept range at these gaps, and the
    second cycle's tail has a genuine phase-winding problem right at the
    edge of what even 1um-step sampling can safely de-wind. A real design
    has no reason to operate a coupler at its second oscillation period
    anyway, so this data is excluded rather than chased with ever-finer
    sampling."""
    gap_values = _discover_gaps() if gaps is None else sorted(gaps)
    library_dirs = [_SWEEP_1D_DIR, _SWEEP_2D_DIR]
    max_length_by_gap = {0.08: 16.0, 0.10: 22.0}

    per_gap = {}
    port_names = None
    for gap in gap_values:
        lib = cl.load_coupler_length_library(
            library_dirs=library_dirs, gap_um_filter=gap,
            max_coupling_length_um=max_length_by_gap.get(gap),
        )
        per_gap[gap] = lib
        if port_names is None:
            port_names = lib.port_names
        elif port_names != lib.port_names:
            raise RuntimeError(f"coupler_library_2d: gap={gap}um has port_names={lib.port_names}, "
                                f"expected {port_names} (mismatched across gaps).")

    return CouplerGapLengthLibrary(gap_um=np.array(gap_values), per_gap=per_gap, port_names=port_names)


def gap_phase_dewinding_report(library: CouplerGapLengthLibrary | None = None,
                                wl_um: float = 1.35) -> dict:
    """Empirically checks how much each S-entry's absolute phase varies
    across the characterized gap axis, at a fixed representative wavelength
    and (per-gap) its own shortest characterized length -- the length where
    cross-gap comparison is least confounded by each gap's own very
    different length-axis phase winding. Reports the raw phase span (deg)
    and the max step-to-step jump between adjacent characterized gaps, so a
    caller can judge whether plain `jnp.unwrap` (no trend removal) is
    sufficient, the way `coupler_diff_2d` currently assumes, or whether a
    fuller de-winding pass (mirroring the length axis) is actually needed."""
    if library is None:
        library = load_coupler_gap_length_library()

    report = {}
    for entry in cl._S_ENTRIES:
        phases_deg = []
        for gap in library.gap_um:
            lib = library.per_gap[float(gap)]
            Lc0 = float(lib.Lc_um[0])  # shortest characterized length for this gap
            s = cl.coupler_diff(np.array([wl_um]), Lc0, library=lib)
            phases_deg.append(float(np.degrees(np.angle(np.asarray(s[_entry_port_pair(entry, lib.port_names)])[0]))))
        unwrapped = np.degrees(np.unwrap(np.radians(phases_deg)))
        step_jumps = np.abs(np.diff(unwrapped))
        report[entry] = {
            "phase_deg_per_gap": phases_deg,
            "total_span_deg": float(np.max(unwrapped) - np.min(unwrapped)),
            "max_step_jump_deg": float(np.max(step_jumps)) if step_jumps.size else 0.0,
        }
    return report


def _entry_port_pair(entry: str, port_names: tuple) -> tuple:
    in_top, in_bot, out_top, out_bot = port_names
    return {
        "S_through": (in_top, out_top), "S_cross": (in_top, out_bot), "S_reflect": (in_top, in_top),
        "S_through_bot": (in_bot, out_bot), "S_cross_bot": (in_bot, out_top), "S_reflect_bot": (in_bot, in_bot),
    }[entry]


def coupler_diff_2d(wl, coupling_length_um, gap_um, library: CouplerGapLengthLibrary | None = None) -> dict:
    """SAX model fn: wl (array-like, usually NOT traced), coupling_length_um
    AND gap_um (each a scalar, MAY be a JAX tracer -- both are now free
    optimization variables) -> SDict over the 4 ports. Evaluates
    `coupler_library.coupler_diff` (unchanged) at every characterized gap,
    then interpolates those results across the gap axis via `jnp.interp` on
    magnitude and unwrapped phase separately (same convention as every other
    axis in this optimization layer). Silently flat-extrapolates outside
    [library.gap_um.min(), .max()] (jnp.interp's own behavior, same caveat
    as the length/wavelength axes) -- callers' bounds must stay inside that
    range, and outside each per-gap library's own [Lc_um.min(), .max()] too.
    """
    if library is None:
        library = load_coupler_gap_length_library()

    # Force wl to at least 1D for the stack/vmap below regardless of whether the caller
    # passed a scalar (e.g. wl=1.35, as circuit_diff.py's per-instance closures do) or an
    # array -- `cl.coupler_diff` itself returns a 0-d result for a scalar wl, and stacking
    # N such 0-d per-gap results gives shape (n_gap,), not (n_gap, n_wl), which silently
    # confuses the wl-axis vmap below. Squeeze back to scalar at the end if that's what
    # was asked for.
    scalar_in = jnp.ndim(wl) == 0
    wl_arr = jnp.atleast_1d(jnp.asarray(wl, dtype=jnp.float32))

    gap_grid = jnp.asarray(library.gap_um)
    port_names = library.port_names
    per_gap_libs = [library.per_gap[float(g)] for g in library.gap_um]

    per_gap_results = [cl.coupler_diff(wl_arr, coupling_length_um, library=lib) for lib in per_gap_libs]

    out = {}
    for key in per_gap_results[0]:
        stacked = jnp.stack([r[key] for r in per_gap_results], axis=0)  # (n_gap, n_wl), guaranteed 2D
        mag = jnp.abs(stacked)
        phase = jnp.unwrap(jnp.angle(stacked), axis=0)

        mag_interp = jax.vmap(lambda col: jnp.interp(gap_um, gap_grid, col), in_axes=1)(mag)   # (n_wl,)
        phase_interp = jax.vmap(lambda col: jnp.interp(gap_um, gap_grid, col), in_axes=1)(phase)
        val = mag_interp * jnp.exp(1j * phase_interp)
        out[key] = val[0] if scalar_in else val
    return out


def fit_quality_report_gap_axis(library: CouplerGapLengthLibrary | None = None, wl_um: float = 1.35) -> dict:
    """Leave-one-out cross-validation over the GAP axis (interior gaps only):
    for each interior characterized gap, interpolate `coupler_diff_2d` from
    its neighboring gaps only (excluding that gap's own library) and compare
    to the true (held-out) value, at a length common to both neighbors'
    characterized ranges. Reports max relative magnitude error and max phase
    error (deg) per S-entry -- the gap-axis analog of `coupler_library.
    fit_quality_report`'s existing length-axis check."""
    if library is None:
        library = load_coupler_gap_length_library()

    gaps = sorted(library.gap_um)
    n_gap = len(gaps)
    report = {entry: {"rel_mag_errs": [], "phase_errs_deg": []} for entry in cl._S_ENTRIES}

    for i in range(1, n_gap - 1):
        held_out_gap = gaps[i]
        neighbor_gaps = gaps[:i] + gaps[i + 1:]
        held_lib = library.per_gap[float(held_out_gap)]
        # A length inside the held-out gap's own characterized range, that also lies inside
        # every neighbor's own range (so no length-axis extrapolation confounds this check).
        lo = max(library.per_gap[float(g)].Lc_um.min() for g in [held_out_gap] + neighbor_gaps)
        hi = min(library.per_gap[float(g)].Lc_um.max() for g in [held_out_gap] + neighbor_gaps)
        if lo >= hi:
            continue
        Lc_probe = (lo + hi) / 2.0

        loo_library = CouplerGapLengthLibrary(
            gap_um=np.array(neighbor_gaps),
            per_gap={g: library.per_gap[float(g)] for g in neighbor_gaps},
            port_names=library.port_names,
        )
        true_s = cl.coupler_diff(np.array([wl_um]), Lc_probe, library=held_lib)
        pred_s = coupler_diff_2d(np.array([wl_um]), Lc_probe, held_out_gap, library=loo_library)

        for entry in cl._S_ENTRIES:
            key = _entry_port_pair(entry, library.port_names)
            true_val, pred_val = np.asarray(true_s[key])[0], np.asarray(pred_s[key])[0]
            rel_mag_err = abs(abs(pred_val) - abs(true_val)) / max(abs(true_val), 1e-9)
            phase_err_deg = float(np.degrees(abs(np.angle(np.exp(1j * (np.angle(pred_val) - np.angle(true_val)))))))
            report[entry]["rel_mag_errs"].append(float(rel_mag_err))
            report[entry]["phase_errs_deg"].append(phase_err_deg)

    summary = {}
    for entry, vals in report.items():
        summary[entry] = {
            "max_relative_magnitude_error": float(np.max(vals["rel_mag_errs"])) if vals["rel_mag_errs"] else float("nan"),
            "max_phase_error_deg": float(np.max(vals["phase_errs_deg"])) if vals["phase_errs_deg"] else float("nan"),
            "n_holdout_gaps": len(vals["rel_mag_errs"]),
        }
    return summary
