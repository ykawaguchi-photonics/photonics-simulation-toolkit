"""Toolkit-wide default physical/simulation parameters.

Every component's own meep_sim/<name>.py DEFAULT_PARAMS starts from
GLOBAL_PARAMS (via `{**GLOBAL_PARAMS, ...}`) and overrides only what that
specific component's physics genuinely requires -- see each module's own
comments for why an override exists. Values that drifted apart across
components without a physical reason (e.g. a copy-pasted bulk-silicon index
in one module while every other module used the toolkit's 2D effective-index
convention) are bugs, not intentional per-component choices; this module is
the single place to fix that going forward.
"""

from __future__ import annotations

GLOBAL_PARAMS = {
    "wg_width_um": 0.5,     # waveguide core width -- matches racetrack.py/coupler.py/
                            # mzi.py/bend_topopt.py's own standalone DEFAULT_PARAMS
                            # (see docs/simulation_settings_record.md's width-
                            # unification note for why 0.5 was picked over 0.45)
    "core_index": 2.7,      # 2D effective index (TE), NOT bulk silicon (~3.45)
    "clad_index": 1.44,     # silicon dioxide
    "polarization": "TE",
    "wl_min_um": 1.30,      # O-band
    "wl_max_um": 1.40,
    "resolution": 20,       # pixels/um
    "dpml_um": 1.0,         # PML thickness
}
