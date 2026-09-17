"""JAX-differentiable arm-length interpolation library for the `mzi_arm`
"jog" delay arm -- the arm-length axis this repo's `models/mzi_arm.py` has
never interpolated over (only exact-grid-match via `mzi_arm_at_sweep_point`).

Architectural rule: this file NEVER imports meep (a plain `import
gdsfactory` is fine -- gdsfactory itself never imports meep). It reads the
same cached `.npz`/`.json` sweep/baseline artifacts `models/mzi_arm.py`
reads, but does NOT modify that file; this is a separate, differentiable
consumer of the same on-disk data for the optimization layer only.

Phase de-winding (why this is needed): the arm's S-parameter phase winds
very fast with physical length (~640 deg/um, confirmed numerically -- see
docs/simulation_settings_record.md-style investigation in the optimization
plan) -- far too fast to `np.unwrap` naively across a sparse, irregularly
spaced length grid. Before unwrapping across the LENGTH axis, an analytic
propagation trend (`k * 2*pi*n_eff*delta_L_um/wl`, k=1 for the through/cross
terms S12/S21, k=2 for a round-trip guess on the reflection terms S11/S22)
is subtracted; what's left is a small, slowly varying residual that unwraps
cleanly. The best-fitting k in {0, 1, 2} is chosen automatically per
S-entry by minimizing the residual's worst per-step jump, and the result is
asserted to stay under 90 degrees/step -- a real de-winding failure raises
loudly rather than silently interpolating garbage.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .. import design_points, sparams

assert "meep" not in sys.modules, "pic_toolkit.optimization.arm_library must never coexist with a meep import"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WAVEGUIDE_DESIGN_POINT = _REPO_ROOT / "data" / "design_points" / "waveguide.yaml"

_LIBRARY_DIRS_BY_RADIUS = {
    5.0: [
        _REPO_ROOT / "data" / "sparams" / "mzi_arm" / "sweep",
        _REPO_ROOT / "data" / "sparams" / "mzi_arm" / "baseline",
        _REPO_ROOT / "data" / "sparams" / "mzi_arm" / "sweep_dense2um",
    ],
    2.5: [
        _REPO_ROOT / "data" / "sparams" / "mzi_arm" / "sweep_r2.5",
        _REPO_ROOT / "data" / "sparams" / "mzi_arm" / "sweep_r2p5_dense",
    ],
}

_S_ENTRIES = ("11", "12", "21", "22")
_DEDUP_TOL_UM = 1e-3
_MAX_RESIDUAL_STEP_DEG = 90.0
_CANDIDATE_TREND_MULTIPLIERS = (0, 1, 2)

# Fixed geometry defaults this optimization targets -- matches
# meep_sim.mzi_arm.DEFAULT_PARAMS's own radius_um/bend_type/wg_width_um.
# Only used to recompute delta_L_um for artifacts (sweep points) whose own
# saved metadata doesn't already carry it -- see _bend_length_um's docstring.
_RADIUS_UM = 5.0
_WG_WIDTH_UM = 0.5
_BEND_TYPE = "euler"


def _bend_length_um(radius_um: float = _RADIUS_UM, wg_width_um: float = _WG_WIDTH_UM,
                     bend_type: str = _BEND_TYPE) -> float:
    """Meep-free reimplementation of `meep_sim.mzi_arm._one_bend`'s bend-length
    lookup (gdsfactory only, no meep) -- needed because `sweep/` artifacts'
    own saved metadata only records `extra_straight_um`/`radius_um`, not the
    derived `delta_L_um` (only the `baseline/` artifact's metadata has that
    field precomputed). Importing `meep_sim.mzi_arm` itself would pull in
    `import meep as mp` at module load time, breaking every downstream
    `assert "meep" not in sys.modules` guard in `models/*.py` -- so this
    duplicates the tiny gdsfactory-only calculation instead, same
    "duplication over a meep-boundary-crossing shared helper" convention
    `models/coupler.py` already established relative to `meep_sim/coupler.py`.
    """
    import gdsfactory as gf

    xs = gf.cross_section.strip(width=wg_width_um, radius=radius_um, radius_min=radius_um)
    bend_fn = gf.components.bend_circular if bend_type == "circular" else gf.components.bend_euler
    bend = bend_fn(radius=radius_um, angle=90, cross_section=xs, allow_min_radius_violation=True)
    return float(bend.info["length"])


def _delta_L_um_from_extra_straight(extra_straight_um: float, radius_um: float = _RADIUS_UM,
                                     wg_width_um: float = _WG_WIDTH_UM, bend_type: str = _BEND_TYPE) -> float:
    """Meep-free mirror of `meep_sim.mzi_arm.delta_L_um_from_extra` -- see
    `_bend_length_um`'s docstring for why this is duplicated here rather
    than imported."""
    return 4 * _bend_length_um(radius_um, wg_width_um, bend_type) - 4 * radius_um + 2 * extra_straight_um


def _artifact_delta_L_um(metadata: dict) -> float:
    if "delta_L_um" in metadata:
        return float(metadata["delta_L_um"])
    geometry = metadata["geometry_params"]
    return _delta_L_um_from_extra_straight(
        geometry["extra_straight_um"],
        radius_um=geometry.get("radius_um", _RADIUS_UM),
        wg_width_um=geometry.get("wg_width_um", _WG_WIDTH_UM),
        bend_type=geometry.get("bend_type", _BEND_TYPE),
    )


def _load_n_eff() -> float:
    design_point = design_points.load_design_point(_WAVEGUIDE_DESIGN_POINT)
    return float(design_point["fitted_model"]["n_eff"])


@dataclass
class ArmLengthLibrary:
    delta_L_um: np.ndarray             # sorted ascending, shape (n_length,)
    wl_grid_um: np.ndarray             # sorted ascending, shape (n_wl,)
    magnitude: dict                    # {"11"/"12"/"21"/"22": jnp.ndarray (n_length, n_wl)}
    dewound_phase: dict                # same keys/shape -- residual phase after trend removal, unwrapped over length
    trend_multiplier: dict             # {"11"/"12"/"21"/"22": int in {0,1,2}} -- chosen automatically
    n_eff: float
    port_names: tuple
    source_stems: list                 # for provenance/debugging -- one Path per length point
    radius_um: float                   # bend radius this library's arms share (5.0 or 2.5)


def _collect_artifact_stems(library_dirs: list[Path]) -> list[Path]:
    # Only per-point artifacts (mzi_arm_baseline / mzi_arm_extra_straight<X>),
    # each with a sibling .json -- excludes summary-only files like
    # mzi_arm_sweep_summary.npz (no matching .json, not a real S-param artifact).
    # NOTE: deliberately not Path.with_suffix() -- these stems embed float
    # parameter values (e.g. "...extra_straight0.8640000000000008"), and
    # with_suffix() only strips/replaces text after the LAST '.', which would
    # mangle such a stem (same footgun sparams.save_artifact's own docstring
    # warns about).
    stems = []
    for d in library_dirs:
        if not d.exists():
            continue
        for npz_path in sorted(d.glob("*.npz")):
            stem_str = str(npz_path)[: -len(".npz")]
            if Path(stem_str + ".json").exists():
                stems.append(Path(stem_str))
    return stems


def load_arm_length_library(radius_um: float = 5.0) -> ArmLengthLibrary:
    """radius_um selects which bend-radius arm family to load (5.0 = stage-1's
    production geometry, 2.5 = the MUX4 tree's stage-2 geometry) -- these are
    physically distinct arms (different port-to-port footprint) and must
    never be mixed into one interpolation table."""
    if radius_um not in _LIBRARY_DIRS_BY_RADIUS:
        raise ValueError(f"No known library directories for radius_um={radius_um}. "
                          f"Available: {sorted(_LIBRARY_DIRS_BY_RADIUS)}")
    library_dirs = _LIBRARY_DIRS_BY_RADIUS[radius_um]

    stems = _collect_artifact_stems(library_dirs)
    if not stems:
        raise FileNotFoundError(
            f"No mzi_arm artifacts found under {[str(d) for d in library_dirs]}. "
            "Run the relevant dense-sweep notebook first."
        )

    entries = []  # (delta_L_um, artifact, stem)
    for stem in stems:
        artifact = sparams.load_artifact(stem)
        delta_L_um = _artifact_delta_L_um(artifact.metadata)
        entries.append((delta_L_um, artifact, stem))

    # Dedup by delta_L_um (within tolerance) -- e.g. 15.0 appears in both
    # baseline/ and sweep/ (extra_straight_um=0.864 vs 0.8640000000000008).
    entries.sort(key=lambda e: e[0])
    deduped = []
    for delta_L_um, artifact, stem in entries:
        if deduped and abs(delta_L_um - deduped[-1][0]) < _DEDUP_TOL_UM:
            continue
        deduped.append((delta_L_um, artifact, stem))

    wl_grid_um = np.sort(deduped[0][1].wavelengths_um)
    wl_order_ref = np.argsort(deduped[0][1].wavelengths_um)
    for delta_L_um, artifact, stem in deduped:
        wl_sorted = np.sort(artifact.wavelengths_um)
        if not np.allclose(wl_sorted, wl_grid_um, rtol=1e-6):
            raise RuntimeError(
                f"Artifact at {stem} (delta_L_um={delta_L_um:.4f}) has a different wavelength "
                f"grid than the library's reference grid -- arm_library assumes every artifact "
                "shares one common wl grid (verified true for the existing 03_mzi_arm.ipynb sweep "
                "+ 03b_mzi_arm_dense_sweep.ipynb outputs; re-check if this fires)."
            )

    delta_L_arr = np.array([e[0] for e in deduped])
    n_length, n_wl = len(deduped), len(wl_grid_um)
    port_names = deduped[0][1].port_names

    n_eff = _load_n_eff()
    magnitude, dewound_phase, trend_multiplier = {}, {}, {}

    for entry in _S_ENTRIES:
        raw = np.zeros((n_length, n_wl), dtype=complex)
        for i, (delta_L_um, artifact, stem) in enumerate(deduped):
            order = np.argsort(artifact.wavelengths_um)
            raw[i, :] = artifact.s_matrix[entry][order]

        mag = np.abs(raw)
        raw_phase = np.angle(raw)

        best_m, best_residual, best_score = None, None, np.inf
        for m in _CANDIDATE_TREND_MULTIPLIERS:
            trend = m * 2.0 * np.pi * n_eff * delta_L_arr[:, None] / wl_grid_um[None, :]
            wrapped_residual = np.angle(np.exp(1j * (raw_phase - trend)))
            unwrapped_residual = np.unwrap(wrapped_residual, axis=0)
            step_jumps = np.abs(np.diff(unwrapped_residual, axis=0))
            score = float(np.max(step_jumps)) if step_jumps.size else 0.0
            if score < best_score:
                best_m, best_residual, best_score = m, unwrapped_residual, score

        if np.degrees(best_score) >= _MAX_RESIDUAL_STEP_DEG:
            worst = np.unravel_index(np.argmax(np.abs(np.diff(best_residual, axis=0))), (n_length - 1, n_wl))
            raise RuntimeError(
                f"arm_library: S{entry} de-winding failed -- best trend multiplier k={best_m} still "
                f"leaves a {np.degrees(best_score):.1f} deg/step residual jump (limit "
                f"{_MAX_RESIDUAL_STEP_DEG} deg) between delta_L_um="
                f"{delta_L_arr[worst[0]]:.3f} and {delta_L_arr[worst[0] + 1]:.3f} at wl="
                f"{wl_grid_um[worst[1]]:.4f}um. The library is too sparse there, or this S-entry's "
                "phase doesn't follow the assumed propagation trend -- do not trust interpolation "
                "across this gap without adding more FDTD points."
            )

        magnitude[entry] = jnp.asarray(mag)
        dewound_phase[entry] = jnp.asarray(best_residual)
        trend_multiplier[entry] = best_m

    return ArmLengthLibrary(
        delta_L_um=delta_L_arr, wl_grid_um=wl_grid_um,
        magnitude=magnitude, dewound_phase=dewound_phase, trend_multiplier=trend_multiplier,
        n_eff=n_eff, port_names=port_names, source_stems=[e[2] for e in deduped],
        radius_um=radius_um,
    )


def _interp_along_length(query_delta_L_um, delta_L_grid, table_2d):
    """table_2d: (n_length, n_wl). Returns (n_wl,) interpolated at
    query_delta_L_um -- differentiable w.r.t. query_delta_L_um via jnp.interp,
    vmapped across the wavelength columns (jnp.interp itself only takes 1D
    xp/fp sharing one x-query)."""
    import jax

    return jax.vmap(lambda col: jnp.interp(query_delta_L_um, delta_L_grid, col), in_axes=1)(table_2d)


def _interp_complex_jnp(wl, wl_grid, s_grid):
    """jnp twin of models.mzi_arm._interp_complex's magnitude+unwrapped-phase
    convention, for the second (wavelength-axis) interpolation stage."""
    wl = jnp.asarray(wl, dtype=jnp.float32)
    magnitude = jnp.interp(wl, wl_grid, jnp.abs(s_grid))
    phase = jnp.interp(wl, wl_grid, jnp.unwrap(jnp.angle(s_grid)))
    return magnitude * jnp.exp(1j * phase)


def mzi_arm_diff(wl, delta_L_delay_um, library: ArmLengthLibrary | None = None) -> dict:
    """SAX model fn: wl (array-like, usually NOT traced), delta_L_delay_um
    (scalar, MAY be a JAX tracer -- the delay-arm decision variable) -> SDict
    over (o1, o2). Two-stage interpolation: (1) across LENGTH at each of the
    library's own wavelength-grid bins (differentiable w.r.t.
    delta_L_delay_um), re-adding the analytic trend removed at library-build
    time; (2) across WAVELENGTH to the caller's requested `wl`. Silently
    flat-extrapolates outside [library.delta_L_um.min(), .max()] (jnp.interp's
    own behavior) -- callers (AdamConfig bounds) must stay inside that range.
    """
    if library is None:
        library = load_arm_length_library()

    delta_L_grid = jnp.asarray(library.delta_L_um)
    wl_grid = jnp.asarray(library.wl_grid_um)
    o1, o2 = library.port_names

    s_at_wlgrid = {}
    for entry in _S_ENTRIES:
        mag_at_wlgrid = _interp_along_length(delta_L_delay_um, delta_L_grid, library.magnitude[entry])
        dewound_at_wlgrid = _interp_along_length(delta_L_delay_um, delta_L_grid, library.dewound_phase[entry])
        m = library.trend_multiplier[entry]
        trend_at_wlgrid = m * 2.0 * jnp.pi * library.n_eff * delta_L_delay_um / wl_grid
        phase_at_wlgrid = dewound_at_wlgrid + trend_at_wlgrid
        s_at_wlgrid[entry] = mag_at_wlgrid * jnp.exp(1j * phase_at_wlgrid)

    return {
        (o1, o1): _interp_complex_jnp(wl, wl_grid, s_at_wlgrid["11"]),
        (o1, o2): _interp_complex_jnp(wl, wl_grid, s_at_wlgrid["12"]),
        (o2, o1): _interp_complex_jnp(wl, wl_grid, s_at_wlgrid["21"]),
        (o2, o2): _interp_complex_jnp(wl, wl_grid, s_at_wlgrid["22"]),
    }


def fit_quality_report(library: ArmLengthLibrary | None = None) -> dict:
    """Leave-one-out cross-validation over the library's own (already
    de-wound) tables: for each interior length point, interpolate from its
    neighbors only and compare to the true (held-out) value. Reports max
    relative magnitude error and max phase error (deg), per S-entry, at the
    held-out point's own wavelength grid. Interior points only (index 1..
    n-2) -- edge points can't be honestly held out without extrapolating."""
    if library is None:
        library = load_arm_length_library()

    delta_L = np.asarray(library.delta_L_um)
    n_length = len(delta_L)
    report = {}
    for entry in _S_ENTRIES:
        mag_table = np.asarray(library.magnitude[entry])
        phase_table = np.asarray(library.dewound_phase[entry])
        m = library.trend_multiplier[entry]

        mag_errs, phase_errs = [], []
        for i in range(1, n_length - 1):
            keep = np.array([j for j in range(n_length) if j != i])
            mag_pred = np.array([np.interp(delta_L[i], delta_L[keep], mag_table[keep, j])
                                  for j in range(mag_table.shape[1])])
            phase_pred = np.array([np.interp(delta_L[i], delta_L[keep], phase_table[keep, j])
                                    for j in range(phase_table.shape[1])])
            mag_true, phase_true = mag_table[i, :], phase_table[i, :]

            rel_mag_err = np.abs(mag_pred - mag_true) / np.clip(mag_true, 1e-9, None)
            phase_err_deg = np.degrees(np.abs(np.angle(np.exp(1j * (phase_pred - phase_true)))))
            mag_errs.append(np.max(rel_mag_err))
            phase_errs.append(np.max(phase_err_deg))

        report[entry] = {
            "max_relative_magnitude_error": float(np.max(mag_errs)) if mag_errs else float("nan"),
            "max_phase_error_deg": float(np.max(phase_errs)) if phase_errs else float("nan"),
            "trend_multiplier": m,
            "n_holdout_points": len(mag_errs),
        }
    return report
