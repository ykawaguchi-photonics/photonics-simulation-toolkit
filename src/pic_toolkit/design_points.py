"""A design point is the single source of truth for a component's selected
physical parameters. Both the SAX model and (later) the gdsfactory layout
must derive their parameters from this record, never from each other.

Intentionally simple: one human-readable YAML file per component.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def save_design_point(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(record, sort_keys=False))


def load_design_point(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Design point not found at {path}. "
            "Run notebooks/01_waveguide_baseline.ipynb first to generate it "
            "(it saves the validated S-parameter artifact and this design point)."
        )
    return yaml.safe_load(path.read_text())
