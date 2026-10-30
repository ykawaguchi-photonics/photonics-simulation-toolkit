"""Cache-or-run helpers for notebooks that re-execute expensive FDTD cells.

Meep-free. A cached artifact is reused only if every parameter it recorded
matches the parameters the caller is about to simulate; otherwise the caller
re-simulates. This keeps a full headless re-run of a notebook cheap without
ever silently reusing data from a different geometry or wavelength grid.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

# metadata blocks that record simulation inputs (as written by sparams.save_artifact /
# add_drop_ring.save_artifact / sweep.grid_sweep)
_PARAM_BLOCKS = ("geometry_params", "material_params", "simulation_params", "swept_params")


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def metadata_mismatches(metadata: dict, params: dict) -> list[str]:
    """Parameters recorded in an artifact's metadata that differ from `params`
    (keys absent from `params` are ignored). Empty list = the artifact matches."""
    bad = []
    for block in _PARAM_BLOCKS:
        for k, v in (metadata.get(block) or {}).items():
            if k in params and not _same(v, params[k]):
                bad.append(f"{block}.{k}: cached {v!r} vs requested {params[k]!r}")
    res = metadata.get("meep_resolution")
    if res is not None and "resolution" in params and not _same(res, params["resolution"]):
        bad.append(f"meep_resolution: cached {res!r} vs requested {params['resolution']!r}")
    return bad


def artifact_exists(path_stem) -> bool:
    stem = Path(path_stem)
    return (stem.parent / (stem.name + ".npz")).exists() and (stem.parent / (stem.name + ".json")).exists()


def cached_pickle(path, key: dict, compute):
    """Return compute() and store it at `path` together with `key`; on later calls return the
    stored value if its key is identical, otherwise recompute and overwrite."""
    path = Path(path)
    if path.exists():
        with open(path, "rb") as f:
            stored = pickle.load(f)
        if stored.get("key") == key:
            print(f"loaded cached result: {path}")
            return stored["value"]
        print(f"cache key changed, recomputing: {path}")
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"key": key, "value": value}, f)
    return value
