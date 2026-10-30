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
                            # (0.5 rather than 0.45, unified across components)
    "core_index": 2.7,      # 2D effective index (TE), NOT bulk silicon (~3.45)
    "clad_index": 1.44,     # silicon dioxide
    "polarization": "TE",
    "wl_min_um": 1.30,      # O-band
    "wl_max_um": 1.40,
    "resolution": 20,       # pixels/um
    "dpml_um": 1.0,         # PML thickness
}

# SiN track (sin/notebooks/): 400 nm LPCVD SiN (n = 2.0, as in meta_sim/sin_grating.py)
# in SiO2, 0.9 um wide, collapsed to the plane by the n_eff-matched effective-index
# method at the band center: the 2D core index is chosen so that the 2D lateral TE0
# mode has the n_eff of the real 900 x 400 nm cross-section (MPB: 1.6753), i.e.
# slab_modes.core_index_for_neff(1.6753, 0.9) = 1.7650. The plain slab index
# (te_slab_index = 1.7786) overestimates the mode index and so underestimates bend
# radiation. The SiN track compares bends against a Si reference calibrated the same
# way (core 2.944), NOT against the Si track's simplified 2.7. Same O-band window as
# Si. Rationale and sources: sin/literature.md and docs/simulation_settings_record.md
# (SiN track section).
SIN_GLOBAL_PARAMS = {
    **GLOBAL_PARAMS,
    "wg_width_um": 0.9,      # 400 nm SiN strip width used by Toronto-IME (literature.md);
                             # carries TE0 and a weakly guided TE1 in this 2D model
    "core_index": 1.7650,    # n_eff-matched 2D index of the 900x400 nm SiN strip, NOT bulk SiN (2.0)
    "sin_thickness_um": 0.4, # recorded for provenance; the 2D solver only sees core_index
}

# Si reference for the SiN track's comparisons: a 500 x 220 nm Si strip (n = 3.50) in
# SiO2, calibrated exactly like SIN_GLOBAL_PARAMS (MPB cross-section n_eff 2.671 ->
# core_index_for_neff = 2.944). Only the SiN track uses it; the Si track (si/notebooks/)
# keeps GLOBAL_PARAMS.
SI_REF_PARAMS = {
    **GLOBAL_PARAMS,
    "core_index": 2.944,
    "si_thickness_um": 0.22,
}
