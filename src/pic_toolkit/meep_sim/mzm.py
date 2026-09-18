"""Meep simulation of a Mach-Zehnder Modulator (MZM): two `coupler.py` 50:50
stages (splitter, combiner) joined by two arms of IDENTICAL physical length,
carrying a push-pull electro-optic index perturbation Delta_n(x) rather than
`mzi.py`'s geometric path-length difference. Where `mzi.py` is the passive
device coupler.py's own docstring anticipates, this module is the ACTIVE one
`mzi.py`'s own docstring anticipates in turn (see its module docstring's
"Unlike a Mach-Zehnder MODULATOR" paragraph): both arms stay the same length,
and the controllable phase shift comes entirely from Delta_n, not geometry.

Simulation model: 2D effective-index cross-section (x = propagation, y =
transverse), the same convention as `waveguide.py`/`coupler.py`/`mzi.py` --
`core_index=2.7` is this toolkit's shared effective index, not bulk
silicon's real refractive index (~3.45; see `grating_coupler.py`, the one
module that resolves the real x-z layer stack instead). Material is
non-dispersive throughout (`mp.Medium(index=...)`, constant across the
analyzed band) -- no wavelength-dependent Sellmeier/Lorentzian model is used
anywhere in this toolkit.

**Vpi here means a Delta_n-defined (Delta_n-equivalent) operating point,
not a real device's measured Vpi.** The push-pull operating point is
n_upper = core_index + Delta_n/2, n_lower = core_index - Delta_n/2, with
Delta_n_vpi the TOTAL index difference between arms at which this design is
meant to deliver a full pi phase swing. The arm length needed for that is
then DERIVED (see `plateau_length_um`/`arm_length_um`) from an idealized
formula, Delta_n_eff = Gamma_eff * Delta_n_material, corrected by an
effective phase-response-factor PRIOR (`phase_factor_prior`) carried over
from `PIC_components/MZM/03_mzm_design_v4.ipynb`'s own embedded-device Meep
calibration (that notebook's Delta_n regime differs by 10x, so this is a
starting estimate, not an assumed-exact value). Gamma_eff is NOT a
0<=Gamma<=1 mode-confinement factor -- it also absorbs whatever modeling
error the idealized formula itself carries, which is why
`phase_factor_prior=1.02` is allowed to exceed 1 (see its own
DEFAULT_PARAMS comment for why an actual confinement-factor measurement was
abandoned in favor of this looser, explicitly-not-a-confinement-factor
prior). notebooks/08_mzm.ipynb's Section 7 Meep-validates how close the
derived length actually lands to a true pi swing and reports the REAL
phase-response factor without forcing agreement, exactly as
03_mzm_design_v4.ipynb's own Section 7a does.

This module does not model the electrical (voltage -> carrier density ->
Delta_n) mechanism -- Delta_n is imposed directly as a numerical stand-in,
same scope limitation racetrack.py/coupler.py's own electro-optic-adjacent
components share.

**Reuse, not re-derivation.** The splitter/combiner stages reuse
`coupler.build_geometry`/`build_gf_component` verbatim (translated into
place, same technique `mzi.py` already uses). The optional geometric
path-length bias (`delta_L_um`, mirroring `mzi.py`'s own device for a
combined active+passive check) reuses `mzi.solve_delay_arm` and
`mzi._build_delay_arm_blocks`/`mzi._delay_arm_path` directly rather than
re-implementing this delicate rotated-Block bend construction a second time:
an axis-aligned-Block reimplementation of this same bump was tried in
`PIC_components/MZM/03_mzm_design_v5.ipynb` and found to leave an actual
GAP in the waveguide core at this bend's slope (adjacent segments' vertical
spans failing to even overlap) -- `mzi._build_delay_arm_blocks`'s
tangent-rotated construction was the fix used there (via a gdsfactory
perpendicular-sweep workaround in that standalone notebook; here, reusing
the toolkit's own already-correct native implementation directly is
possible and preferable). The push-pull Delta_n overlay itself is new to
this module: an erf-ramped index perturbation applied to whichever geometry
(straight or delay-arm-bent) each arm actually has, positioned so it decays
to ~0 well before either coupler stage.

**TE/TM**: `eig_parity=mp.TE` is forced at every source/monitor, following
`coupler.py`'s already-correct convention (CLAUDE.md Sec. 5) -- `mzi.py`
still uses `mp.NO_PARITY` as an un-migrated legacy default; this module,
being new, does not inherit that.

IMPORTANT ARCHITECTURAL NOTE: same reasoning as `coupler.py`/`mzi.py` -- this
is a 4-port device (single-excitation results are S11/S21/S31/S41; the
canonical baseline additionally excites in_bot for S22/S12/S32/S42 and a
genuine reciprocity check), so `pic_toolkit.checks`/`pic_toolkit.sparams`/
`pic_toolkit.sweep.grid_sweep` (all hardcoded to a 2x2 s_matrix) don't apply.
This module is self-contained (own result dataclass, own physical
validation, own artifact save/load), matching `coupler.py`/`mzi.py`'s
precedent. `pic_toolkit.viz` and `pic_toolkit.sweep.save_manifest` ARE
generic enough to reuse as-is (see notebooks/08_mzm.ipynb).

Only this file (plus `waveguide.py`, `bend.py`, `bend_topopt.py`,
`racetrack.py`, `coupler.py`, `mzi.py`) imports meep.
"""

from __future__ import annotations

import contextlib
import io
import json
import platform
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import meep as mp
import numpy as np
from scipy.special import erf

from . import coupler, gds_import
from .mzi import solve_delay_arm, _build_delay_arm_blocks, _delay_arm_path

mp.verbosity(0)


@contextlib.contextmanager
def _quiet_meep():
    """See coupler.py's identical helper -- silences Meep's C++-layer stdout/
    stderr chatter around init_sim()/run(), without swallowing real Python
    exceptions."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
# User-adjustable parameters.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # --- splitter/combiner stage: reused verbatim from coupler.py's own
    # DEFAULT_PARAMS / data/design_points/coupler.yaml (the validated 50:50
    # point from notebooks/06_directional_coupler.ipynb) -- both MZM stages
    # use this exact design, unmodified, same convention as mzi.py.
    "coupling_length_um": 13.848128928800111,
    "gap_um": 0.2,
    "wg_width_um": 0.5,
    "wg_height_um": 0.22,
    "core_index": 2.7,
    "clad_index": 1.44,
    "sbend_len_um": 6.0,
    "wide_sep_um": 2.0,
    "lead_len_um": 2.0,
    "n_seg": 24,               # coupler stages ONLY -- see mzi.py's identical note.
    "mon_size_um": 1.2,
    "margin_um": 0.7,
    "wl_min_um": 1.3,
    "wl_max_um": 1.4,
    "n_freq": 101,
    "wl0_um": 1.35,            # follows the reused coupler stage's own validated
                               # wavelength (its 50:50 split was Meep-verified
                               # specifically at 1.35um) -- NOT the 1311nm CWDM
                               # channel PIC_components/MZM's own notebooks used.
    "resolution": 25,
    "dpml_um": 1.0,

    # --- push-pull phase shifter: Vpi is DEFINED, not measured. Delta_n_vpi
    # is the TOTAL index difference between arms (n_upper - n_lower) at which
    # this design is meant to deliver a full pi push-pull phase swing:
    #   n_upper = core_index + Delta_n/2, n_lower = core_index - Delta_n/2.
    "delta_n_vpi": 0.002,
    "sigma_um": 3.0,           # erf ramp width -- ramps Delta_n(x) up/down gradually
                               # rather than as a hard step, suppressing reflection at
                               # the perturbation's own edges (same technique
                               # PIC_components/MZM's notebooks already validated).
    "margin_sigma": 4.0,       # erf transition margin (in units of sigma_um) kept
                               # inactive at each arm end, so Delta_n ~ 0 well before
                               # reaching either coupler stage.
    "phase_factor_prior": 1.02,  # effective phase-response-factor PRIOR used only to
                               # SIZE the arm length -- carried over from
                               # PIC_components/MZM/03_mzm_design_v4.ipynb's own
                               # embedded-device Meep calibration (Gamma_eff~1.02 at
                               # that notebook's much larger push-pull differential,
                               # 0.02 vs. this module's 0.002) since no better a
                               # priori estimate exists (an isolated-waveguide
                               # confinement measurement was tried there and found
                               # unreliable -- Gamma above the physical bound of 1,
                               # sign-unstable vs. resolution -- and dropped).
                               # NOT a 0<=Gamma<=1 mode-confinement factor: an actual
                               # confinement measurement was exactly what got dropped
                               # above for being unreliable, so this PRIOR is instead
                               # an effective phase-response factor in
                               # Delta_n_eff = Gamma_eff * Delta_n_material, which is
                               # why >1 is an allowed value here rather than a bug --
                               # it also absorbs the idealized sizing formula's own
                               # modeling error, not just mode overlap.
                               # notebooks/08_mzm.ipynb Meep-measures the REAL
                               # phase-response factor at this module's own operating
                               # point and reports it without forcing agreement.

    # --- path-length bias (optional; delta_L_um=0.0 is the pure-modulator
    # case). Reuses mzi.py's own delay-arm technique and DEFAULT_PARAMS values
    # verbatim when active, so a combined active+passive device is directly
    # comparable to the passive-only mzi.py baseline.
    "bump_n_seg": 200,
    "delay_region_len_um": 100.0,   # window (at the start of the delay arm) the
                                    # bump is built over -- longer than mzi.py's own
                                    # default (50um) because this module's arm is
                                    # already long enough that a longer window costs
                                    # nothing in extra footprint (see build_geometry)
                                    # while keeping the bend gentle: at
                                    # delta_L_um=10, 100um gives a ~33 degree max
                                    # slope (min radius of curvature ~24um) versus a
                                    # 52-degree/~3.7um-radius bend confirmed to
                                    # produce an actual construction defect at a
                                    # shorter window (see module docstring).
    "delta_L_um": 0.0,

    # --- FDTD run-length control -- same long-domain floor mzi.py's own
    # (much shorter) delay arm already needed; this module's arm is longer
    # still, so the floor matters even more here. See mzi.py's
    # min_sim_time_factor comment for the documented failure mode this
    # guards against (mp.stop_when_dft_decayed's default can be satisfied by
    # a quiet early transient before a pulse reaches a far monitor).
    "dft_decay_tol": 1e-3,
    "min_sim_time_factor": 2.0,
    "max_sim_time_factor": 4.0,
}


def plateau_length_um(params: dict) -> float:
    """Idealized-then-phase-factor-corrected plateau length for a pi
    push-pull swing at Delta_n = delta_n_vpi:
        pi = Gamma_eff * (2*pi/wl0) * Delta_n * L  =>  L = wl0 / (2*Gamma_eff*Delta_n)
    Gamma_eff (Delta_n_eff = Gamma_eff * Delta_n_material) is an effective
    phase-response factor, NOT a 0<=Gamma<=1 mode-confinement factor -- see
    `phase_factor_prior`'s own DEFAULT_PARAMS comment for why it's allowed to
    exceed 1. Uses phase_factor_prior as the best available estimate for
    sizing -- not re-derived from a measurement in this module itself.
    """
    return params["wl0_um"] / (2.0 * params["phase_factor_prior"] * params["delta_n_vpi"])


def arm_length_um(params: dict) -> float:
    """Total arm length actually built: the phase-active plateau plus an
    inactive erf-ramp margin at each end (see margin_sigma/sigma_um)."""
    return plateau_length_um(params) + 2 * params["margin_sigma"] * params["sigma_um"]


def build_index_perturbation(x, x1, x2, delta_n_max, sigma):
    """Smooth (erf) Delta_n(x) profile: ~0 outside [x1, x2], ~delta_n_max
    inside, with transition width `sigma` at each edge -- avoids the
    reflection a hard step discontinuity in Delta_n would cause."""
    return 0.5 * delta_n_max * (erf((x - x1) / sigma) - erf((x - x2) / sigma))


@dataclass
class PermittivityMap:
    eps: np.ndarray
    extent_um: tuple


@dataclass
class FieldSnapshot:
    field: np.ndarray
    eps: np.ndarray
    extent_um: tuple
    component: str


@dataclass
class MZMResult:
    wavelengths_um: np.ndarray
    freqs: np.ndarray
    S11: np.ndarray               # excite port1 (in_top) -> reflection back into port1
    S21: np.ndarray               # excite port1 -> leaked out port2 (in_bot)
    S31: np.ndarray               # excite port1 -> out_top ("through"/bar)
    S41: np.ndarray               # excite port1 -> out_bot ("cross")
    S22: np.ndarray = None        # only set when the baseline (dual-excitation) run is used
    S12: np.ndarray = None
    S32: np.ndarray = None
    S42: np.ndarray = None
    delta_n_lower: float = 0.0
    delta_n_upper: float = 0.0
    port_names: tuple = ("in_top", "in_bot", "out_top", "out_bot")
    permittivity: PermittivityMap = None
    field_snapshot: FieldSnapshot = None
    sim_params: dict = field(default_factory=dict)


def _translate_block(block, dx):
    """coupler.build_geometry(params) always returns geometry centered at
    x=0; translate its mp.Block list by dx to place a splitter/combiner
    stage at this MZM's own stage centers -- same helper mzi.py uses."""
    c = block.center
    return mp.Block(size=block.size, center=mp.Vector3(c.x + dx, c.y, c.z), material=block.material)


def _compute_domain(params: dict) -> dict:
    """Derive every x-position, the two coupler stages' centers, the arm
    span (from the Vpi-derived arm_length_um), the erf-perturbation plateau
    boundaries, the delay-window bounds and solved bump amplitude, and the
    cell size -- shared by build_geometry, port/monitor placement, and
    get_permittivity_map, so all three always agree (same role as
    coupler.py/mzi.py's own _compute_domain).

    Like mzi.py, cell_y is NOT symmetric about y=0 when a delay-arm bump is
    present (delta_L_um > 0): the cell tightly wraps the reference arm's top
    edge to the bump's peak, rather than mirroring the bump's extent onto
    the empty top half too. `y_center` must be passed as every
    mp.Simulation's `geometry_center`.
    """
    coupler_device_len = 2 * params["sbend_len_um"] + params["coupling_length_um"]
    arm_span = arm_length_um(params)
    total_device_len = 2 * coupler_device_len + arm_span

    x_c1_center = -total_device_len / 2 + coupler_device_len / 2
    x_c2_center = total_device_len / 2 - coupler_device_len / 2
    x_arm_L = -arm_span / 2
    x_arm_R = arm_span / 2

    margin = params["margin_sigma"] * params["sigma_um"]
    plateau_x1 = x_arm_L + margin
    plateau_x2 = x_arm_R - margin

    delay_len = min(params["delay_region_len_um"], arm_span)
    x_delay_L = x_arm_L
    x_delay_R = x_arm_L + delay_len

    wide_half = params["wide_sep_um"] / 2
    width = params["wg_width_um"]
    bend_amplitude_um = solve_delay_arm(params["delta_L_um"], delay_len) if params["delta_L_um"] > 0 else 0.0

    lead, dpml, margin_um = params["lead_len_um"], params["dpml_um"], params["margin_um"]
    cell_x = total_device_len + 2 * (lead + dpml + margin_um)

    y_top = wide_half + width / 2
    y_bottom = -(wide_half + bend_amplitude_um) - width / 2
    cell_y = (y_top - y_bottom) + 2 * (dpml + margin_um)
    y_center = (y_top + y_bottom) / 2

    mon_size = params["mon_size_um"]
    top_clearance = (y_center + cell_y / 2 - dpml) - (wide_half + mon_size / 2)
    neighbor_clearance = params["wide_sep_um"] - mon_size
    if neighbor_clearance <= 0 or top_clearance <= 0:
        raise ValueError(
            f"mon_size_um={mon_size} doesn't fit: neighbor clearance={neighbor_clearance:.3f}um, "
            f"PML clearance={top_clearance:.3f}um (both must be > 0). Shrink mon_size_um or "
            f"grow wide_sep_um/margin_um."
        )

    return dict(
        coupler_device_len=coupler_device_len, arm_span=arm_span, total_device_len=total_device_len,
        x_c1_center=x_c1_center, x_c2_center=x_c2_center, x_arm_L=x_arm_L, x_arm_R=x_arm_R,
        plateau_x1=plateau_x1, plateau_x2=plateau_x2, x_delay_L=x_delay_L, x_delay_R=x_delay_R,
        delay_len=delay_len, wide_half=wide_half, bend_amplitude_um=bend_amplitude_um,
        cell_x=cell_x, cell_y=cell_y, y_center=y_center, mon_size=mon_size,
    )


def _material_fn(core_index, x1, x2, delta_n, sigma):
    def f(p):
        return mp.Medium(index=core_index + build_index_perturbation(p.x, x1, x2, delta_n, sigma))
    return f


def build_geometry(params: dict, delta_n_lower: float = 0.0, delta_n_upper: float = 0.0) -> list:
    """Two coupler stages (splitter, combiner), each coupler.build_geometry's
    own validated 50:50 shape translated into place, joined by two arms of
    IDENTICAL physical length extending directly from the coupler ports' own
    wide_sep_um separation: the top arm carries delta_n_upper, the bottom arm
    carries delta_n_lower, each as an erf-ramped overlay (build_index_
    perturbation) confined to the plateau region (px1, px2) -- ~0 by
    construction at either coupler stage. When delta_L_um > 0, the bottom arm
    additionally carries mzi.py's own raised-cosine delay-arm bump over a
    delay_region_len_um-long window at the arm's start; the Delta_n overlay
    tracks the GLOBAL x-coordinate regardless of the physical y-path the arm
    follows through that window, so it stays correctly calibrated either way.
    """
    dom = _compute_domain(params)
    core = mp.Medium(index=params["core_index"])
    width = params["wg_width_um"]

    stage_blocks = coupler.build_geometry(params)
    left = [_translate_block(b, dom["x_c1_center"]) for b in stage_blocks]
    right = [_translate_block(b, dom["x_c2_center"]) for b in stage_blocks]
    geometry = left + right

    wide_half = dom["wide_half"]
    x_arm_L, x_arm_R = dom["x_arm_L"], dom["x_arm_R"]
    px1, px2 = dom["plateau_x1"], dom["plateau_x2"]
    sigma = params["sigma_um"]
    core_index = params["core_index"]

    # --- Top arm: straight over its full length -------------------------------------
    geometry.append(mp.Block(size=mp.Vector3(x_arm_R - x_arm_L, width, mp.inf),
                              center=mp.Vector3(0, wide_half, 0), material=core))
    if delta_n_upper:
        geometry.append(mp.Block(size=mp.Vector3(x_arm_R - x_arm_L, width, mp.inf),
                                  center=mp.Vector3(0, wide_half, 0),
                                  material=_material_fn(core_index, px1, px2, delta_n_upper, sigma)))

    # --- Bottom arm: optional delay-window bump (mzi.py's own construction), then
    # straight remainder --------------------------------------------------------------
    x_delay_L, x_delay_R = dom["x_delay_L"], dom["x_delay_R"]
    bump_amp = dom["bend_amplitude_um"]
    geometry += _build_delay_arm_blocks(x_delay_L, x_delay_R, -wide_half, bump_amp, width, core,
                                         params["bump_n_seg"])
    if delta_n_lower:
        geometry += _build_delay_arm_blocks(x_delay_L, x_delay_R, -wide_half, bump_amp, width,
                                             _material_fn(core_index, px1, px2, delta_n_lower, sigma),
                                             params["bump_n_seg"])
    if x_delay_R < x_arm_R:
        rest_len = x_arm_R - x_delay_R
        rest_center = (x_delay_R + x_arm_R) / 2
        geometry.append(mp.Block(size=mp.Vector3(rest_len, width, mp.inf),
                                  center=mp.Vector3(rest_center, -wide_half, 0), material=core))
        if delta_n_lower:
            geometry.append(mp.Block(size=mp.Vector3(rest_len, width, mp.inf),
                                      center=mp.Vector3(rest_center, -wide_half, 0),
                                      material=_material_fn(core_index, px1, px2, delta_n_lower, sigma)))

    return geometry


def build_gf_component(params: dict):
    """gdsfactory Component for geometry display/comparison (Section 3/4 of
    notebooks/08_mzm.ipynb) -- traces the same shape build_geometry()'s
    native mp.Block construction uses, reusing coupler.build_gf_component's
    stage shape and mzi._delay_arm_path's bump curve directly. gdsfactory has
    no notion of a spatially-varying index, so this shows the WAVEGUIDE SHAPE
    only, not the Delta_n perturbation -- same limitation the native
    permittivity-map preview does not have (it uses real mp.Medium functions)."""
    import gdsfactory as gf

    dom = _compute_domain(params)
    cross_section = gf.cross_section.strip(width=params["wg_width_um"])

    top, bot = coupler.build_gf_component(params)
    combined = gf.Component()
    for stage_x in (dom["x_c1_center"], dom["x_c2_center"]):
        for stage_comp in (top, bot):
            ref = combined.add_ref(stage_comp)
            ref.move((stage_x, 0))

    wide_half = dom["wide_half"]
    x_arm_L, x_arm_R = dom["x_arm_L"], dom["x_arm_R"]
    x_delay_L, x_delay_R = dom["x_delay_L"], dom["x_delay_R"]

    ref_path = gf.Path(np.array([[x_arm_L, wide_half], [x_arm_R, wide_half]]))
    combined.add_ref(gf.path.extrude(ref_path, cross_section=cross_section))

    delay_path = _delay_arm_path(x_delay_L, x_delay_R, -wide_half, dom["bend_amplitude_um"])
    combined.add_ref(gf.path.extrude(delay_path, cross_section=cross_section))
    if x_delay_R < x_arm_R:
        rest_path = gf.Path(np.array([[x_delay_R, -wide_half], [x_arm_R, -wide_half]]))
        combined.add_ref(gf.path.extrude(rest_path, cross_section=cross_section))

    return combined


def build_geometry_from_gds(params: dict) -> list:
    """Meep geometry sourced from `build_gf_component` via `gds_import` --
    display/comparison only (see get_permittivity_map's use_native_geometry
    docstring); NOT the default for simulate()."""
    core = mp.Medium(index=params["core_index"])
    combined = build_gf_component(params)
    return gds_import.gds_component_to_prisms(combined, material=core)


def _ports(params: dict) -> dict:
    dom = _compute_domain(params)
    half = dom["total_device_len"] / 2
    wide_half = dom["wide_half"]
    return {
        "in_top": mp.Vector3(-half, wide_half),
        "in_bot": mp.Vector3(-half, -wide_half),
        "out_top": mp.Vector3(half, wide_half),
        "out_bot": mp.Vector3(half, -wide_half),
    }


def _leads(params: dict) -> list:
    """Straight extension blocks from each port to the cell edge -- same role
    as coupler.py/mzi.py's own _leads."""
    dom = _compute_domain(params)
    core = mp.Medium(index=params["core_index"])
    width = params["wg_width_um"]
    x_edge = dom["cell_x"] / 2
    ports = _ports(params)
    blocks = []
    for name, x_sign in [("in_top", -1), ("in_bot", -1), ("out_top", 1), ("out_bot", 1)]:
        p = ports[name]
        x_far = x_sign * x_edge
        blocks.append(mp.Block(
            size=mp.Vector3(abs(x_far - p.x), width, mp.inf),
            center=mp.Vector3((x_far + p.x) / 2, p.y, 0),
            material=core,
        ))
    return blocks


def get_permittivity_map(params: dict, delta_n_lower: float = 0.0, delta_n_upper: float = 0.0,
                          use_native_geometry: bool = True) -> PermittivityMap:
    """Permittivity map WITHOUT running any FDTD timestepping -- inspect both
    coupler stages and the push-pull arm perturbation before paying for an
    expensive simulation, same discipline as every other component.

    `use_native_geometry=True` (the default) uses `build_geometry` -- the
    mp.Medium-function construction every validated result is measured
    against (and the only one that can show the real Delta_n(x) profile, not
    just waveguide shape). `use_native_geometry=False` uses the
    gdsfactory-derived `build_geometry_from_gds` instead, for a
    geometry-display/comparison view.
    """
    dom = _compute_domain(params)
    cell = mp.Vector3(dom["cell_x"], dom["cell_y"], 0)
    geom_center = mp.Vector3(0, dom["y_center"], 0)
    clad = mp.Medium(index=params["clad_index"])
    geometry = (build_geometry(params, delta_n_lower, delta_n_upper) if use_native_geometry
                else build_geometry_from_gds(params))
    sim = mp.Simulation(
        cell_size=cell, boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=geometry + _leads(params),
        default_material=clad, resolution=params["resolution"],
        geometry_center=geom_center,
    )
    with _quiet_meep():
        sim.init_sim()
    eps = sim.get_epsilon()
    extent = (-cell.x / 2, cell.x / 2, dom["y_center"] - cell.y / 2, dom["y_center"] + cell.y / 2)
    return PermittivityMap(eps=eps, extent_um=extent)


def _make_simulation(params: dict, delta_n_lower: float, delta_n_upper: float, launch_from: str,
                      capture_dft: bool):
    dom = _compute_domain(params)
    cell = mp.Vector3(dom["cell_x"], dom["cell_y"], 0)
    geom_center = mp.Vector3(0, dom["y_center"], 0)
    clad = mp.Medium(index=params["clad_index"])
    fcen = 1.0 / ((params["wl_min_um"] + params["wl_max_um"]) / 2)
    fwidth = 1.0 / params["wl_min_um"] - 1.0 / params["wl_max_um"]
    ports = _ports(params)
    src_port = ports[launch_from]
    other_in = "in_bot" if launch_from == "in_top" else "in_top"

    source = mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
        center=mp.Vector3(src_port.x - 0.5, src_port.y),
        size=mp.Vector3(0, dom["mon_size"], 0),
        eig_band=1, eig_parity=mp.TE, eig_match_freq=True,
        eig_kpoint=mp.Vector3(1, 0, 0),
    )
    sim = mp.Simulation(
        cell_size=cell, boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=build_geometry(params, delta_n_lower, delta_n_upper) + _leads(params), sources=[source],
        default_material=clad, resolution=params["resolution"],
        geometry_center=geom_center,
    )
    mon_self = sim.add_mode_monitor(fcen, fwidth, params["n_freq"],
        mp.ModeRegion(center=src_port, size=mp.Vector3(0, dom["mon_size"])))
    # Monitors the OTHER (non-excited) input port's backward-traveling component -- power
    # that couples/reflects all the way back out the "wrong" input side rather than through
    # either output. Needed for a genuine S21/S12 reciprocity check (mzi.py's own
    # simulate_baseline needs the identical monitor for the same reason); omitting it
    # doesn't just under-measure S21/S12, it silently leaves them at exactly 0+0j, which
    # would make reciprocity_check trivially "pass" without testing anything real.
    mon_in_other = sim.add_mode_monitor(fcen, fwidth, params["n_freq"],
        mp.ModeRegion(center=ports[other_in], size=mp.Vector3(0, dom["mon_size"])))
    mon_out_top = sim.add_mode_monitor(fcen, fwidth, params["n_freq"],
        mp.ModeRegion(center=ports["out_top"], size=mp.Vector3(0, dom["mon_size"])))
    mon_out_bot = sim.add_mode_monitor(fcen, fwidth, params["n_freq"],
        mp.ModeRegion(center=ports["out_bot"], size=mp.Vector3(0, dom["mon_size"])))

    dft_obj = None
    if capture_dft:
        dft_obj = sim.add_dft_fields([mp.Ez, mp.Hz], fcen, fcen, 1, center=geom_center, size=cell)
    return sim, mon_self, mon_in_other, mon_out_top, mon_out_bot, dft_obj, dom


def _run_one_excitation(params: dict, delta_n_lower: float, delta_n_upper: float, launch_from: str,
                         capture_dft: bool):
    """Excite `launch_from` and measure self (reflection), out_top, out_bot
    mode coefficients -- self-normalized against this same run's own
    excited-port monitor, same as coupler.py/mzi.py (this device is not
    resonant). Minimum FDTD run time is computed here (not a fixed
    DEFAULT_PARAMS constant) since it must scale with this device's own
    (Vpi-derived) cell_x -- see DEFAULT_PARAMS["min_sim_time_factor"]'s
    docstring."""
    sim, mon_self, mon_in_other, mon_out_top, mon_out_bot, dft_obj, dom = _make_simulation(
        params, delta_n_lower, delta_n_upper, launch_from, capture_dft)
    min_sim_time = params["min_sim_time_factor"] * params["core_index"] * (dom["cell_x"] + params["delta_L_um"])
    max_sim_time = params["max_sim_time_factor"] * min_sim_time
    with _quiet_meep():
        sim.run(until_after_sources=mp.stop_when_dft_decayed(
            tol=params["dft_decay_tol"], minimum_run_time=min_sim_time, maximum_run_time=max_sim_time,
        ))
        res_self = sim.get_eigenmode_coefficients(mon_self, [1], eig_parity=mp.TE)
        res_other = sim.get_eigenmode_coefficients(mon_in_other, [1], eig_parity=mp.TE)
        res_top = sim.get_eigenmode_coefficients(mon_out_top, [1], eig_parity=mp.TE)
        res_bot = sim.get_eigenmode_coefficients(mon_out_bot, [1], eig_parity=mp.TE)
        freqs = np.array(mp.get_flux_freqs(mon_self))
        if capture_dft:
            ez = sim.get_dft_array(dft_obj, mp.Ez, 0)
            hz = sim.get_dft_array(dft_obj, mp.Hz, 0)
            eps = sim.get_array(component=mp.Dielectric)

    a_self, a_other, a_top, a_bot = (res_self.alpha[0, :, :], res_other.alpha[0, :, :],
                                      res_top.alpha[0, :, :], res_bot.alpha[0, :, :])
    incident = a_self[:, 0]
    # The other input port's own "backward" (index 1) direction is the correct sign for
    # power leaking OUT of the device there -- both input ports launch in the +x direction
    # (see _ports/_make_simulation), so leaked power arriving at the other input port is,
    # like this run's own reflection, travelling in -x. Same convention as mzi.py.
    cross_input = a_other[:, 1]

    field_snapshot = None
    if capture_dft:
        extent = (-dom["cell_x"] / 2, dom["cell_x"] / 2,
                  dom["y_center"] - dom["cell_y"] / 2, dom["y_center"] + dom["cell_y"] / 2)
        if np.max(np.abs(ez)) >= np.max(np.abs(hz)):
            field, component = ez, "Ez"
        else:
            field, component = hz, "Hz"
        field_snapshot = FieldSnapshot(field=field, eps=eps, extent_um=extent, component=component)
    return freqs, incident, a_self[:, 1], cross_input, a_top[:, 0], a_bot[:, 0], field_snapshot


def simulate(params: dict | None = None, delta_n_lower: float = 0.0, delta_n_upper: float = 0.0,
             capture_dft: bool = False) -> MZMResult:
    """Single-excitation (in_top / port 1) characterization at one push-pull
    operating point (delta_n_lower, delta_n_upper) -- the workhorse call for
    the Delta_n sweep (Section 10) and the delta_L_um path-bias check
    (Section 12), where many points are needed and a full dual-excitation
    reciprocity check per point would be prohibitively expensive on this
    device's already-long domain. See `simulate_baseline` for the one
    canonical, fully-validated (dual-excitation) operating point.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    freqs, inc, refl, cross, top, bot, field_snapshot = _run_one_excitation(
        params, delta_n_lower, delta_n_upper, "in_top", capture_dft)
    permittivity = get_permittivity_map(params, delta_n_lower, delta_n_upper) if capture_dft else None
    sim_params = {**params, "meep_version": mp.__version__,
                  "python_version": platform.python_version(), "creation_date": date.today().isoformat()}
    return MZMResult(
        wavelengths_um=1.0 / freqs, freqs=freqs,
        S11=refl / inc, S21=cross / inc, S31=top / inc, S41=bot / inc,
        delta_n_lower=delta_n_lower, delta_n_upper=delta_n_upper,
        permittivity=permittivity, field_snapshot=field_snapshot, sim_params=sim_params,
    )


def simulate_baseline(params: dict | None = None, delta_n_lower: float | None = None,
                       delta_n_upper: float | None = None) -> MZMResult:
    """Full 4-port characterization at the canonical push-pull-ON operating
    point (default: delta_n_lower=-delta_n_vpi/2, delta_n_upper=+delta_n_vpi/2,
    i.e. the Vpi design point itself): excite in_top (port 1) AND in_bot
    (port 2) independently, enabling a genuine Lorentz-reciprocity check
    (S21 vs S12) the same way mzi.py's own baseline does. This is the one
    operating point this module validates at full (strict-tolerance) rigor;
    the Delta_n sweep and delta_L_um check (Sections 10/12) use the cheaper
    single-excitation `simulate` instead.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    if delta_n_lower is None:
        delta_n_lower = -params["delta_n_vpi"] / 2
    if delta_n_upper is None:
        delta_n_upper = params["delta_n_vpi"] / 2

    freqs, inc1, refl1, cross1, top1, bot1, field_snapshot = _run_one_excitation(
        params, delta_n_lower, delta_n_upper, "in_top", True)
    S11 = refl1 / inc1
    S21 = cross1 / inc1   # port1 -> leaked out port2 (in_bot), not either output
    S31 = top1 / inc1
    S41 = bot1 / inc1

    _, inc2, refl2, cross2, top2, bot2, _ = _run_one_excitation(
        params, delta_n_lower, delta_n_upper, "in_bot", False)
    # in_bot is "port 2"; its own reflection is S22, and its through/cross responses
    # land on the SAME two output ports port-1's did -- from port 2's perspective
    # out_bot is "through" and out_top is "cross", the mirror of port 1's labeling.
    S22 = refl2 / inc2
    S12 = cross2 / inc2   # port2 -> leaked out port1 (in_top)
    S32 = top2 / inc2   # port2 -> out_top ("cross" from port 2)
    S42 = bot2 / inc2   # port2 -> out_bot ("through" from port 2)

    permittivity = get_permittivity_map(params, delta_n_lower, delta_n_upper)
    sim_params = {**params, "meep_version": mp.__version__,
                  "python_version": platform.python_version(), "creation_date": date.today().isoformat()}
    return MZMResult(
        wavelengths_um=1.0 / freqs, freqs=freqs,
        S11=S11, S21=S21, S31=S31, S41=S41,
        S22=S22, S12=S12, S32=S32, S42=S42,
        delta_n_lower=delta_n_lower, delta_n_upper=delta_n_upper,
        permittivity=permittivity, field_snapshot=field_snapshot, sim_params=sim_params,
    )


# ---------------------------------------------------------------------------
# Local validation -- deliberately NOT pic_toolkit.checks (2-port hardcoded),
# same rationale as coupler.py/mzi.py.
# ---------------------------------------------------------------------------
def energy_conservation_check(result: MZMResult, tol: float = 0.02, max_expected_loss: float = 0.05) -> dict:
    """|S11|^2+|S21|^2+|S31|^2+|S41|^2 ~= 1 (likewise for the port-2
    excitation, if present), within max_expected_loss. Includes S21/S12
    (power leaked out the OTHER input port, not either output) alongside
    self-reflection and the two through/cross outputs -- see mzi.py's
    identical check for why this channel matters for a two-coupler-stage
    device. For a `simulate()`-only (single-excitation) result, S21 IS a
    real measurement (both excitations always measure their own leaked-
    power monitor); only S12 requires the second excitation."""
    def _dev(*arrs):
        total = sum(np.abs(a) ** 2 for a in arrs)
        lower = 1.0 - max_expected_loss
        return float(np.max(np.maximum(total - 1.0, lower - total)))

    dev1 = max(0.0, _dev(result.S11, result.S21, result.S31, result.S41))
    if result.S22 is not None:
        dev2 = max(0.0, _dev(result.S22, result.S12, result.S32, result.S42))
        max_dev = max(dev1, dev2)
        return {"max_deviation_port1": dev1, "max_deviation_port2": dev2, "max_deviation": max_dev,
                "max_expected_loss": max_expected_loss, "tolerance": tol, "passed": max_dev < tol}
    return {"max_deviation_port1": dev1, "max_deviation": dev1,
            "max_expected_loss": max_expected_loss, "tolerance": tol, "passed": dev1 < tol}


def passivity_check(result: MZMResult, tol: float = 0.01) -> dict:
    """A passive device can never amplify: every power fraction <= 1, within
    a small numerical margin."""
    vals = [result.S11, result.S21, result.S31, result.S41]
    if result.S22 is not None:
        vals += [result.S22, result.S12, result.S32, result.S42]
    max_mag = max(float(np.max(np.abs(v))) for v in vals)
    return {"max_power_fraction": max_mag ** 2, "tolerance": tol, "passed": max_mag < 1.0 + tol}


def reciprocity_check(result: MZMResult, tol: float = 0.02) -> dict:
    """Genuine Lorentz reciprocity requires S_ij = S_ji. With push-pull
    Delta_n symmetric about the arms' centerline (delta_n_lower = -delta_n_upper),
    the device has no top<->bottom mirror symmetry to exploit the way
    coupler.py's passive, symmetric device can -- like mzi.py, only the
    (port1, port2) pair is directly checkable from two independent
    excitations: S21 (excite port1, measure leaked power at port2) vs. S12
    (excite port2, measure leaked power at port1). Only meaningful when the
    baseline (dual-excitation) result is passed in -- a `simulate()`-only
    (single-excitation) result has a real S21 but no S12 at all (S12 stays
    the MZMResult dataclass default, None, since the second excitation never
    ran); callers should only run this against `simulate_baseline()`'s
    result (S22 is not None)."""
    if result.S22 is None:
        return {"max_diff": 0.0, "tolerance": tol, "passed": True, "note": "not measured (single-excitation result)"}
    d = float(np.max(np.abs(result.S21 - result.S12)))
    return {"max_diff": d, "tolerance": tol, "passed": d < tol}


def run_all_checks(result: MZMResult, energy_tol: float = 0.03, reciprocity_tol: float = 0.03,
                    passivity_tol: float = 0.01, max_expected_loss: float = 0.05) -> dict:
    energy = energy_conservation_check(result, tol=energy_tol, max_expected_loss=max_expected_loss)
    reciprocity = reciprocity_check(result, tol=reciprocity_tol)
    passivity = passivity_check(result, tol=passivity_tol)
    return {
        "energy_conservation": energy, "reciprocity": reciprocity, "passivity": passivity,
        "passed": bool(energy["passed"] and reciprocity["passed"] and passivity["passed"]),
    }


# ---------------------------------------------------------------------------
# Local artifact save/load -- own .npz+.json sidecar pair, same convention as
# coupler.py/mzi.py.
# ---------------------------------------------------------------------------
@dataclass
class MZMArtifact:
    wavelengths_um: np.ndarray
    freqs: np.ndarray
    S11: np.ndarray
    S21: np.ndarray
    S31: np.ndarray
    S41: np.ndarray
    S22: np.ndarray
    S12: np.ndarray
    S32: np.ndarray
    S42: np.ndarray
    port_names: tuple
    metadata: dict


def save_artifact(path_stem, result: MZMResult, metadata: dict) -> None:
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    zeros = np.zeros_like(result.S11)
    np.savez(
        path_stem.parent / (path_stem.name + ".npz"),
        wavelengths_um=result.wavelengths_um, freqs=result.freqs,
        S11=result.S11, S21=result.S21, S31=result.S31, S41=result.S41,
        S22=result.S22 if result.S22 is not None else zeros,
        S12=result.S12 if result.S12 is not None else zeros,
        S32=result.S32 if result.S32 is not None else zeros,
        S42=result.S42 if result.S42 is not None else zeros,
    )
    full_metadata = {**metadata, "port_names": list(result.port_names),
                      "delta_n_lower": result.delta_n_lower, "delta_n_upper": result.delta_n_upper}
    (path_stem.parent / (path_stem.name + ".json")).write_text(json.dumps(full_metadata, indent=2, default=str))


def load_artifact(path_stem) -> MZMArtifact:
    path_stem = Path(path_stem)
    npz_path = path_stem.parent / (path_stem.name + ".npz")
    json_path = path_stem.parent / (path_stem.name + ".json")
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"MZM artifact not found at {path_stem}(.npz/.json). "
            "Run notebooks/08_mzm.ipynb first to generate it."
        )
    data = np.load(npz_path)
    metadata = json.loads(json_path.read_text())
    return MZMArtifact(
        wavelengths_um=data["wavelengths_um"], freqs=data["freqs"],
        S11=data["S11"], S21=data["S21"], S31=data["S31"], S41=data["S41"],
        S22=data["S22"], S12=data["S12"], S32=data["S32"], S42=data["S42"],
        port_names=tuple(metadata["port_names"]), metadata=metadata,
    )
