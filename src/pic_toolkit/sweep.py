"""Generic parameter-sweep driver, reused by any component's simulate_baseline
function. No meep import here -- this only orchestrates calls and file I/O.

Rule 6 (never sweep before baseline validation) is enforced by the notebooks
that call this -- e.g. by asserting the baseline's validation passed before
this module is ever invoked -- not by this module itself, which has no way
to know what a human has actually inspected.
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

from . import checks, sparams


def _default_validate(result, **check_kwargs):
    """Default `validate_fn` for grid_sweep -- the generic 2-port checks,
    exactly what every caller got before `validate_fn` existed."""
    return checks.run_all_checks(result.s_matrix, **check_kwargs)


_MATERIAL_KEYS = ("core_index", "clad_index")


def _reusable(artifact_stem: Path, run_params: dict, check_kwargs: dict) -> dict | None:
    """The stored validation of an existing artifact recorded with exactly
    these geometry parameters (every `_um` key, none missing), resolution and
    check tolerances; None if there is no such artifact."""
    from .artifact_cache import _same, artifact_exists

    if not artifact_exists(artifact_stem):
        return None
    meta = sparams.load_artifact(artifact_stem).metadata
    stored = meta.get("geometry_params") or {}
    wanted = {k: v for k, v in run_params.items() if k.endswith("_um")}
    if set(stored) != set(wanted) or not all(_same(stored[k], wanted[k]) for k in wanted):
        return None
    if not _same(meta.get("meep_resolution"), run_params.get("resolution")):
        return None
    # Artifacts written before material_params was recorded carry no indices; they keep
    # the old geometry-only match. Newer ones must also match the indices, since a
    # changed core_index alone (e.g. a recalibrated 2D index) changes every result.
    stored_mat = meta.get("material_params")
    if stored_mat is not None and not all(
            k in run_params and _same(stored_mat[k], run_params[k]) for k in stored_mat):
        return None
    if meta.get("check_kwargs", {}) != check_kwargs:
        return None
    return meta["validation"]


def grid_sweep(simulate_fn, base_params: dict, param_grid: dict, artifact_dir: Path,
               component_name: str, check_kwargs: dict | None = None, validate_fn=None,
               reuse_existing: bool = False) -> list[dict]:
    """Run simulate_fn over the Cartesian product of param_grid's values.

    param_grid values are merged into base_params (overriding matching keys)
    for each combination. Saves one artifact per run and returns one manifest
    row per run (also written to disk by save_manifest). check_kwargs is
    passed through to `validate_fn` -- e.g. a component with a known loss
    channel (a ring's bend radiation) can pass max_loss_fraction so every
    sweep point is validated against the SAME physically-appropriate
    tolerance used for its baseline, not the generic default.

    `validate_fn`, if given, replaces the default `checks.run_all_checks`
    call -- signature `validate_fn(result, **check_kwargs) -> dict` (with a
    `"passed"` key), called with the full `simulate_fn` result (not just
    `result.s_matrix`), so a component whose checks need more than the
    s_matrix alone (e.g. racetrack.py's resonance-aware energy check, which
    also needs `result.wavelengths_um` to tell on-resonance points from
    off-resonance ones) can use it. Every OTHER component keeps using the
    default (`validate_fn=None`), which behaves identically to the
    `checks.run_all_checks(result.s_matrix, **check_kwargs)` call this
    replaced -- this parameter is purely additive.

    `reuse_existing=True` loads a point instead of simulating it when its
    artifact already exists with exactly the same geometry parameters,
    resolution and check tolerances (see `_reusable`), so a notebook can be
    re-executed without repeating hours of FDTD.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    check_kwargs = check_kwargs or {}
    validate_fn = validate_fn or _default_validate

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    rows = []
    for i, combo in enumerate(combos, start=1):
        label = ", ".join(f"{k}={v}" for k, v in zip(keys, combo))
        print(f"[{i}/{len(combos)}  {100 * i // len(combos)}%] running {component_name} ({label}) ...", flush=True)

        run_params = {**base_params, **dict(zip(keys, combo))}
        tag = "_".join(f"{k.replace('_um', '')}{v}" for k, v in zip(keys, combo))
        artifact_stem = artifact_dir / f"{component_name}_{tag}"

        validation = _reusable(artifact_stem, run_params, check_kwargs) if reuse_existing else None
        if validation is not None:
            print("    reusing the existing artifact (same geometry, resolution and tolerances)", flush=True)
        else:
            result = simulate_fn(run_params)
            validation = validate_fn(result, **check_kwargs)
            metadata = {
                "component": component_name,
                "swept_params": dict(zip(keys, combo)),
                "geometry_params": {k: v for k, v in run_params.items() if k.endswith("_um")},
                "meep_resolution": run_params.get("resolution"),
                "material_params": {k: run_params[k] for k in _MATERIAL_KEYS if k in run_params},
                "meep_version": result.sim_params.get("meep_version"),
                "creation_date": result.sim_params.get("creation_date"),
                "check_kwargs": check_kwargs,
                "validation": validation,
            }
            sparams.save_artifact(
                artifact_stem, result.wavelengths_um, result.freqs,
                result.s_matrix, result.port_names, metadata,
            )

        rows.append({
            **dict(zip(keys, combo)),
            "wl_min_um": run_params["wl_min_um"],
            "wl_max_um": run_params["wl_max_um"],
            "n_freq": run_params["n_freq"],
            "artifact_path": str(artifact_stem),
            "validation_passed": validation["passed"],
        })

    return rows


def save_manifest(rows: list[dict], path: Path) -> None:
    """Rows from different grid_sweep calls (e.g. a 1D radius sweep and a 2D
    radius x gap sweep) can have different key sets -- use the union of all
    columns seen, in first-appearance order, rather than assuming every row
    matches the first one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
