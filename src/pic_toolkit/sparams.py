"""S-parameter artifact storage: one .npz (numeric arrays) + one .json
(metadata) sharing a filename stem. No meep import -- this module only does
file I/O and is shared by both the Meep side (writer) and the SAX side
(reader).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SparamArtifact:
    wavelengths_um: np.ndarray
    freqs: np.ndarray
    s_matrix: dict            # {"11": arr, "12": arr, "21": arr, "22": arr}, complex
    port_names: tuple
    metadata: dict             # component, geometry/material params, polarization,
                                # resolution, sim params, meep version, date, validation


def save_artifact(path_stem: Path, wavelengths_um, freqs, s_matrix, port_names, metadata) -> None:
    # NOTE: deliberately NOT Path.with_suffix() -- it replaces everything after
    # the LAST '.' in the stem, which silently truncates (and collides) stems
    # that embed a float parameter value, e.g. "ring_radius2.2" -> "ring_radius2".
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        path_stem.parent / (path_stem.name + ".npz"),
        wavelengths_um=wavelengths_um,
        freqs=freqs,
        s11=s_matrix["11"],
        s12=s_matrix["12"],
        s21=s_matrix["21"],
        s22=s_matrix["22"],
    )
    full_metadata = {**metadata, "port_names": list(port_names)}
    (path_stem.parent / (path_stem.name + ".json")).write_text(json.dumps(full_metadata, indent=2))


def to_sdict(s_matrix: dict, port_names: tuple) -> dict:
    """Convert the on-disk {"11"/"12"/"21"/"22": array} convention into SAX's
    SDict convention, keyed by (port_in, port_out) name tuples. The on-disk
    schema itself is left unchanged (models/*.py still depend on it) -- this
    is only a display/interchange helper, de-duplicating the identical
    tuple-key construction that used to be copy-pasted inside each
    models/<name>.py's own model function.
    """
    o1, o2 = port_names
    return {
        (o1, o1): s_matrix["11"],
        (o1, o2): s_matrix["12"],
        (o2, o1): s_matrix["21"],
        (o2, o2): s_matrix["22"],
    }


def load_artifact(path_stem: Path) -> SparamArtifact:
    path_stem = Path(path_stem)
    npz_path = path_stem.parent / (path_stem.name + ".npz")
    json_path = path_stem.parent / (path_stem.name + ".json")
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"S-parameter artifact not found at {path_stem}(.npz/.json). "
            "Run notebooks/01_waveguide_baseline.ipynb first to generate it."
        )

    data = np.load(npz_path)
    metadata = json.loads(json_path.read_text())
    s_matrix = {
        "11": data["s11"], "12": data["s12"],
        "21": data["s21"], "22": data["s22"],
    }
    return SparamArtifact(
        wavelengths_um=data["wavelengths_um"],
        freqs=data["freqs"],
        s_matrix=s_matrix,
        port_names=tuple(metadata["port_names"]),
        metadata=metadata,
    )
