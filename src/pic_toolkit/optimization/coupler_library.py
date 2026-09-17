"""JAX-differentiable coupler-length interpolation library -- the coupling-
length analog of `arm_library.py`, extending the toolkit's `coupling_length_um`
axis (previously only exact-grid-matchable via `models.coupler.
coupler_at_sweep_point`) to a continuous, gradient-friendly interpolant.

Architectural rule: this file NEVER imports meep (matches `arm_library.py`'s
own rule). It reads the same cached sweep artifacts `models/coupler.py`
reads, via a small meep-free loader mirroring that module's own
`_load_coupler_artifact` (the coupler's 4-port S-matrix doesn't fit
`pic_toolkit.sparams`'s hardcoded 2x2 schema, so neither module uses it --
same "duplication over a schema-mismatched shared helper" convention
`models/coupler.py` already established).

The cross-vs-through RELATIVE phase stays within a fraction of a degree of
+90deg at every swept point (a directional coupler's coupled-mode phase
relationship is set by the coupling geometry, not the coupling length --
length only changes how MUCH power couples, never the phase sign; this is
why length tuning alone can never fix a stage that needs the opposite
sign). But each S-entry's OWN ABSOLUTE phase still winds substantially with
`coupling_length_um`, exactly like the arm's phase winds with `delta_L_um`
(more coupling length is still more propagation length) -- confirmed
empirically: a first version of this module skipped de-winding and its own
leave-one-out check immediately caught ~180deg errors. So this reuses
`arm_library.py`'s exact de-winding approach (subtract an analytic
propagation trend before unwrapping across length, re-add it at query time,
automatically picking the best-fitting trend multiplier per S-entry) rather
than assuming it away a second time. Coupling MAGNITUDE vs. length is smooth
and monotonic (kappa rises from 0.046 at 2um to 0.954 at 26um, no
oscillation), so magnitude interpolation itself needs no such correction.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

assert "meep" not in sys.modules, "pic_toolkit.optimization.coupler_library must never coexist with a meep import"

from .. import design_points  # noqa: E402  (after the meep-free assert, matching arm_library.py's own ordering)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WAVEGUIDE_DESIGN_POINT = _REPO_ROOT / "data" / "design_points" / "waveguide.yaml"
_LIBRARY_DIRS = [
    _REPO_ROOT / "data" / "sparams" / "coupler" / "sweep",
    # NOTE: data/sparams/coupler/selected/ (the 13.848um 50:50 point) is
    # deliberately NOT merged in -- it was re-simulated in a separate run
    # from the uniform 7-point sweep and was found to carry a spurious
    # ~96deg absolute phase offset relative to its 14.0um neighbor despite
    # being only 0.15um away (confirmed by direct inspection: two physically
    # near-identical lengths cannot legitimately differ in S_through phase
    # by that much). Likely a different monitor/reference-plane convention
    # between the two runs, not a real physical effect. The 7-point sweep
    # alone already covers the CHOSEN_LC range (2-26um) at a consistent
    # reference, which is what actually matters for this interpolation.
]
_S_ENTRIES = ("S_through", "S_cross", "S_reflect", "S_through_bot", "S_cross_bot", "S_reflect_bot")
_DEDUP_TOL_UM = 1e-3
_MAX_RESIDUAL_STEP_DEG = 90.0
_CANDIDATE_TREND_MULTIPLIERS = (0, 1, 2)
_NULL_MAGNITUDE_THRESHOLD = 0.06  # below this, phase is unreliable (at/near an actual coupling null, a real
                                    # branch-point zero-crossing of this S-entry, not simulation noise) --
                                    # excluded from the de-winding quality check only, not from the
                                    # interpolation table itself (confirmed: S_through at Lc=26um has a
                                    # genuine null around wl=1.392-1.393um, |S_through| dropping to ~0.002
                                    # then recovering -- phase near such a zero is ill-defined regardless of
                                    # interpolation method, and its negligible magnitude there means it
                                    # barely affects any reconstructed value anyway).


def _load_n_eff() -> float:
    design_point = design_points.load_design_point(_WAVEGUIDE_DESIGN_POINT)
    return float(design_point["fitted_model"]["n_eff"])


def _load_raw_artifact(path_stem: Path) -> dict:
    """Meep-free mirror of models.coupler._load_coupler_artifact / meep_sim.
    coupler's own save format -- see module docstring for why this isn't
    shared via pic_toolkit.sparams."""
    npz_path = path_stem.parent / (path_stem.name + ".npz")
    json_path = path_stem.parent / (path_stem.name + ".json")
    data = np.load(npz_path)
    metadata = json.loads(json_path.read_text())
    coupling_length_um = metadata.get("coupling_length_um")
    if coupling_length_um is None:
        coupling_length_um = metadata["geometry_params"]["coupling_length_um"]
    # gap_um is a new metadata field (coupler_library_2d.py's sweep_2d/ artifacts) -- the
    # original single-gap sweep/ artifacts never recorded it, since gap_um was always the
    # fixed DEFAULT_PARAMS value (0.2) for every one of those artifacts.
    gap_um = metadata.get("gap_um", 0.2)
    return {
        "coupling_length_um": float(coupling_length_um),
        "gap_um": float(gap_um),
        "wavelengths_um": data["wavelengths_um"],
        **{entry: data[entry] for entry in _S_ENTRIES},
        "port_names": tuple(metadata["port_names"]),
    }


@dataclass
class CouplerLengthLibrary:
    Lc_um: np.ndarray                  # sorted ascending, shape (n_length,)
    wl_grid_um: np.ndarray             # sorted ascending, shape (n_wl,)
    magnitude: dict                    # {S_entry: jnp.ndarray (n_length, n_wl)}
    dewound_phase: dict                # {S_entry: jnp.ndarray (n_length, n_wl)} -- residual after trend removal
    trend_multiplier: dict             # {S_entry: int in {0,1,2}} -- chosen automatically
    n_eff: float
    port_names: tuple
    source_stems: list


def _collect_artifact_stems(library_dirs: list[Path] | None = None) -> list[Path]:
    dirs = _LIBRARY_DIRS if library_dirs is None else library_dirs
    stems = []
    for d in dirs:
        if not d.exists():
            continue
        for npz_path in sorted(d.glob("*.npz")):
            stem_str = str(npz_path)[: -len(".npz")]
            if Path(stem_str + ".json").exists():
                stems.append(Path(stem_str))
    return stems


def load_coupler_length_library(library_dirs: list[Path] | None = None,
                                 gap_um_filter: float | None = None,
                                 max_coupling_length_um: float | None = None) -> CouplerLengthLibrary:
    """`library_dirs`/`gap_um_filter`/`max_coupling_length_um` are optional,
    backward-compatible additions -- every existing caller (passing none of
    them) gets exactly the original behavior (scan `_LIBRARY_DIRS`, no gap
    filtering, no length truncation, since every artifact there shares the
    same implicit gap_um=0.2 and the original 7-point sweep never had a
    winding problem). `coupler_library_2d.py` uses all three: it scans a
    directory containing MULTIPLE gaps' artifacts and loads one gap's worth
    at a time via `gap_um_filter`, reusing this exact, already-validated
    length+wavelength de-winding logic unchanged for each gap.
    `max_coupling_length_um` truncates to the first coupling oscillation
    period only -- at the tightest characterized gaps (0.08/0.10um), the
    coupling ratio completes a full 0->1->0 cycle and starts a SECOND one
    within the swept length range; phase winding in that second cycle's
    tail was found to occasionally exceed even a 1um sampling step's safe
    de-winding margin (right at the edge of numerical ambiguity, not fixed
    by finer sampling alone), and a real design has no reason to operate a
    coupler at its second oscillation period anyway (longer coupling run,
    more loss/back-reflection risk, no coupling-ratio benefit over the
    first period) -- so that data is deliberately excluded rather than
    chased further."""
    stems = _collect_artifact_stems(library_dirs)
    if not stems:
        dirs = _LIBRARY_DIRS if library_dirs is None else library_dirs
        raise FileNotFoundError(
            f"No coupler artifacts found under {[str(d) for d in dirs]}. "
            "Run notebooks/06_directional_coupler.ipynb first."
        )

    entries = [(_load_raw_artifact(stem), stem) for stem in stems]
    if gap_um_filter is not None:
        entries = [(a, s) for a, s in entries if abs(a["gap_um"] - gap_um_filter) < _DEDUP_TOL_UM]
    if max_coupling_length_um is not None:
        entries = [(a, s) for a, s in entries if a["coupling_length_um"] <= max_coupling_length_um]
        if not entries:
            raise FileNotFoundError(f"No coupler artifacts found with gap_um={gap_um_filter} "
                                     f"under {[str(d) for d in (library_dirs or _LIBRARY_DIRS)]}.")
    entries.sort(key=lambda e: e[0]["coupling_length_um"])
    deduped = []
    for artifact, stem in entries:
        if deduped and abs(artifact["coupling_length_um"] - deduped[-1][0]["coupling_length_um"]) < _DEDUP_TOL_UM:
            continue
        deduped.append((artifact, stem))

    wl_grid_um = np.sort(deduped[0][0]["wavelengths_um"])
    for artifact, stem in deduped:
        wl_sorted = np.sort(artifact["wavelengths_um"])
        if not np.allclose(wl_sorted, wl_grid_um, rtol=1e-6):
            raise RuntimeError(
                f"Artifact at {stem} (coupling_length_um={artifact['coupling_length_um']:.4f}) has a "
                "different wavelength grid than the library's reference grid."
            )

    Lc_arr = np.array([e[0]["coupling_length_um"] for e in deduped])
    n_length, n_wl = len(deduped), len(wl_grid_um)
    port_names = deduped[0][0]["port_names"]
    n_eff = _load_n_eff()

    magnitude, dewound_phase, trend_multiplier = {}, {}, {}
    for entry in _S_ENTRIES:
        raw = np.zeros((n_length, n_wl), dtype=complex)
        for i, (artifact, stem) in enumerate(deduped):
            order = np.argsort(artifact["wavelengths_um"])
            raw[i, :] = artifact[entry][order]

        mag = np.abs(raw)
        raw_phase = np.angle(raw)

        # Wavelength bins where EITHER endpoint of a length-step is near a
        # coupling null (this S-entry's own power routed almost entirely
        # elsewhere) have simulation-noise-dominated phase there -- exclude
        # those steps from the residual-quality check (not from the
        # interpolation table itself; their tiny magnitude means their
        # phase barely affects the reconstructed complex value anyway).
        well_conditioned_step = (mag[:-1, :] > _NULL_MAGNITUDE_THRESHOLD) & (mag[1:, :] > _NULL_MAGNITUDE_THRESHOLD)

        best_m, best_residual, best_score = None, None, np.inf
        for m in _CANDIDATE_TREND_MULTIPLIERS:
            trend = m * 2.0 * np.pi * n_eff * Lc_arr[:, None] / wl_grid_um[None, :]
            wrapped_residual = np.angle(np.exp(1j * (raw_phase - trend)))
            unwrapped_residual = np.unwrap(wrapped_residual, axis=0)
            step_jumps = np.abs(np.diff(unwrapped_residual, axis=0))
            masked_jumps = step_jumps[well_conditioned_step]
            score = float(np.max(masked_jumps)) if masked_jumps.size else 0.0
            if score < best_score:
                best_m, best_residual, best_score = m, unwrapped_residual, score

        if np.degrees(best_score) >= _MAX_RESIDUAL_STEP_DEG:
            step_jumps = np.abs(np.diff(best_residual, axis=0))
            step_jumps_masked = np.where(well_conditioned_step, step_jumps, -1)
            worst = np.unravel_index(np.argmax(step_jumps_masked), (n_length - 1, n_wl))
            raise RuntimeError(
                f"coupler_library: {entry} de-winding failed -- best trend multiplier k={best_m} still "
                f"leaves a {np.degrees(best_score):.1f} deg/step residual jump (limit "
                f"{_MAX_RESIDUAL_STEP_DEG} deg) between coupling_length_um="
                f"{Lc_arr[worst[0]]:.3f} and {Lc_arr[worst[0] + 1]:.3f} at wl={wl_grid_um[worst[1]]:.4f}um "
                "(excluding near-null-magnitude bins, so this is a genuine winding-rate problem)."
            )

        magnitude[entry] = jnp.asarray(mag)
        dewound_phase[entry] = jnp.asarray(best_residual)
        trend_multiplier[entry] = best_m

    return CouplerLengthLibrary(
        Lc_um=Lc_arr, wl_grid_um=wl_grid_um, magnitude=magnitude, dewound_phase=dewound_phase,
        trend_multiplier=trend_multiplier, n_eff=n_eff,
        port_names=port_names, source_stems=[e[1] for e in deduped],
    )


def _interp_along_length(query_Lc_um, Lc_grid, table_2d):
    return jax.vmap(lambda col: jnp.interp(query_Lc_um, Lc_grid, col), in_axes=1)(table_2d)


def _interp_complex_jnp(wl, wl_grid, s_grid):
    wl = jnp.asarray(wl, dtype=jnp.float32)
    magnitude = jnp.interp(wl, wl_grid, jnp.abs(s_grid))
    phase = jnp.interp(wl, wl_grid, jnp.unwrap(jnp.angle(s_grid)))
    return magnitude * jnp.exp(1j * phase)


def coupler_diff(wl, coupling_length_um, library: CouplerLengthLibrary | None = None) -> dict:
    """SAX model fn: wl (array-like, usually NOT traced), coupling_length_um
    (scalar, MAY be a JAX tracer -- a coupler-length decision variable) ->
    SDict over the 4 ports (in_top, in_bot, out_top, out_bot). Two-stage
    interpolation exactly like `arm_library.mzi_arm_diff`: (1) across LENGTH
    at each fixed wavelength-grid bin (differentiable w.r.t.
    coupling_length_um); (2) across WAVELENGTH to the caller's requested wl.
    Silently flat-extrapolates outside [library.Lc_um.min(), .max()] --
    callers' bounds must stay inside that range."""
    if library is None:
        library = load_coupler_length_library()

    Lc_grid = jnp.asarray(library.Lc_um)
    wl_grid = jnp.asarray(library.wl_grid_um)
    in_top, in_bot, out_top, out_bot = library.port_names

    s_at_wlgrid = {}
    for entry in _S_ENTRIES:
        mag_at_wlgrid = _interp_along_length(coupling_length_um, Lc_grid, library.magnitude[entry])
        dewound_at_wlgrid = _interp_along_length(coupling_length_um, Lc_grid, library.dewound_phase[entry])
        m = library.trend_multiplier[entry]
        trend_at_wlgrid = m * 2.0 * jnp.pi * library.n_eff * coupling_length_um / wl_grid
        phase_at_wlgrid = dewound_at_wlgrid + trend_at_wlgrid
        s_at_wlgrid[entry] = mag_at_wlgrid * jnp.exp(1j * phase_at_wlgrid)

    def _s(entry):
        return _interp_complex_jnp(wl, wl_grid, s_at_wlgrid[entry])

    s_through, s_cross, s_r = _s("S_through"), _s("S_cross"), _s("S_reflect")
    s_through_bot, s_cross_bot, s_r_bot = _s("S_through_bot"), _s("S_cross_bot"), _s("S_reflect_bot")

    return {
        (in_top, out_top): s_through, (out_top, in_top): s_through,
        (in_top, out_bot): s_cross, (out_bot, in_top): s_cross,
        (in_top, in_top): s_r,
        (in_bot, out_bot): s_through_bot, (out_bot, in_bot): s_through_bot,
        (in_bot, out_top): s_cross_bot, (out_top, in_bot): s_cross_bot,
        (in_bot, in_bot): s_r_bot,
    }


def fit_quality_report(library: CouplerLengthLibrary | None = None) -> dict:
    """Leave-one-out cross-validation over the library's own tables (interior
    points only). Reports max relative magnitude error and max phase error
    (deg) per S-entry."""
    if library is None:
        library = load_coupler_length_library()

    Lc = np.asarray(library.Lc_um)
    n_length = len(Lc)
    report = {}
    for entry in _S_ENTRIES:
        mag_table = np.asarray(library.magnitude[entry])
        phase_table = np.asarray(library.dewound_phase[entry])

        mag_errs, phase_errs = [], []
        for i in range(1, n_length - 1):
            keep = np.array([j for j in range(n_length) if j != i])
            mag_pred = np.array([np.interp(Lc[i], Lc[keep], mag_table[keep, j]) for j in range(mag_table.shape[1])])
            phase_pred = np.array([np.interp(Lc[i], Lc[keep], phase_table[keep, j]) for j in range(phase_table.shape[1])])
            mag_true, phase_true = mag_table[i, :], phase_table[i, :]

            rel_mag_err = np.abs(mag_pred - mag_true) / np.clip(mag_true, 1e-9, None)
            phase_err_deg = np.degrees(np.abs(np.angle(np.exp(1j * (phase_pred - phase_true)))))
            mag_errs.append(np.max(rel_mag_err))
            phase_errs.append(np.max(phase_err_deg))

        report[entry] = {
            "max_relative_magnitude_error": float(np.max(mag_errs)) if mag_errs else float("nan"),
            "max_phase_error_deg": float(np.max(phase_errs)) if phase_errs else float("nan"),
            "trend_multiplier": library.trend_multiplier[entry],
            "n_holdout_points": len(mag_errs),
        }
    return report
