"""Meep simulation of a uniform-period, partial-etch grating coupler: the
vertical (fiber-to-chip) transition between a horizontally-guided waveguide
mode and a tilted free-space beam coupled into an optical fiber.

AXIS CONVENTION -- the one deliberate departure every other module in this
toolkit avoids: Meep's own computational (x, y) are used here as physical
(propagation-x, vertical-z), NOT (propagation-x, in-plane transverse-y) like
every other `meep_sim/*.py` module. Consequently Meep's own invariant/
extrusion axis (always internally called "z" regardless of what physical
meaning we assign x/y) represents physical y -- the real in-plane lateral
direction, which this 2D cross-section assumes the device is uniform along.

WHY this departure is physically necessary (not a style choice): a grating
coupler's whole purpose is coupling a horizontally-guided mode to a
vertically-radiating beam. That radiation is set by the REAL vertical layer
stack (BOX thickness, Si device-layer thickness, partial-etch grating-tooth
depth, top cladding) -- exactly the dimension every other module's 2D
effective-index convention (`core_index=2.7`, collapsing z into one number)
throws away. This module instead resolves that stack directly, with REAL
bulk indices (`core_index=3.45` for Si, not the toolkit's effective 2.7).

STRUCTURAL SIMPLIFICATION this buys: because the x-z cross-section is
translationally invariant along y, a width TAPER is invisible to this 2D
simulation (a taper only changes the device's extent in y). So
`build_geometry()` below never needs taper-shaped blocks -- the input lead
and the grating region differ only in whether etched teeth are present.
**This also means `coupling_efficiency` below is a y-invariant idealization**
(as if the aperture were infinitely wide/uniform in y): it is good for tuning
period / duty-cycle / etch-depth / fiber-angle (genuinely x-z-plane physics --
the fiber tilt lies in the plane containing propagation and vertical), but it
does NOT capture the real device's taper adiabaticity loss or finite-
aperture-width beam mismatch in y. Same "deliberate Phase-1 simplification,
not an oversight" status as every other module's own approximations -- see
`waveguide.py`'s docstring for the precedent.

NO `build_geometry_from_gds()` HERE -- the one place this module breaks the
toolkit's GDSFactory-first pattern (see CLAUDE.md section 4). A GDS layer
only encodes an x-y footprint; it cannot represent this module's z-stack.
The x-y layout (`grating_coupler_gds.py`, meep-free) is therefore a SEPARATE
projection of the same design parameters, not a shared geometry source with
this file -- it is a real tapeout export, never fed into this module's FDTD.

TE/TM -- ROTATED-AXIS GOTCHA, EMPIRICALLY CONFIRMED (see
docs/simulation_settings_record.md's `grating_coupler.py` section for the
full measurement): in every OTHER module, `eig_parity=mp.TE` is the fix for
the toolkit's well-known `NO_PARITY` bug (see CLAUDE.md section 5) because
their invariant Meep-z axis is physical z, and standard photonics "TE" (E
confined to the wafer plane, i.e. no vertical E component) means Ez(Meep)=0,
Meep's own definition of TE. HERE the invariant Meep-z axis is physical y
instead, so standard photonics "TE" (E along the in-plane lateral direction,
i.e. E along physical y = along Meep's invariant axis) means E has a
NONZERO Meep-z component -- which is Meep's own definition of **TM**, not
TE. Directly measured via `Simulation.get_eigenmode()` on this module's own
220nm-Si-slab cross-section at the O-band center (1.35um): Meep-labelled
`mp.TM`, band 1 gives n_eff=2.91 with dominant fields Ez(Meep)/Hy(Meep) (E
along the invariant axis, physically-TE); Meep-labelled `mp.TE`, band 1 gives
n_eff=2.27 with dominant fields Ey(Meep)/Hz(Meep) (physically-TM). **This
module therefore passes `eig_parity=mp.TM` everywhere** to select the
physically-TE fundamental mode -- the opposite label from every other module,
for a physically consistent reason, not an inconsistency. Field snapshots
capture Ez and Hy (this module's analog of every other module's Ez/Hz check)
and report whichever dominates.

Only this file (plus `gds_import.py`, and NOT `grating_coupler_gds.py`,
which is deliberately meep-free) imports meep.
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

from ..params import GLOBAL_PARAMS

mp.verbosity(0)


@contextlib.contextmanager
def _quiet_meep():
    """See waveguide.py's identical helper -- silences Meep's C++-layer
    stdout/stderr chatter around init_sim()/run(), without swallowing real
    Python exceptions."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
# User-adjustable parameters.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    **GLOBAL_PARAMS,               # wg_width_um=0.5, polarization="TE",
                                    # wl_min_um=1.30/wl_max_um=1.40 (O-band,
                                    # kept unchanged per user request),
                                    # dpml_um=1.0 -- all reused as-is.
    "core_index": 3.45,            # OVERRIDE: real bulk Si, not the toolkit's
                                    # 2D effective index 2.7 -- see module docstring.
    "substrate_index": 3.45,       # NEW: Si handle wafer below the BOX. Numerically
                                    # equal to core_index (both bulk Si) but kept as
                                    # its own name -- distinct physical role.
    "resolution": 40,              # OVERRIDE: GLOBAL_PARAMS' 20 px/um cannot resolve
                                    # sub-micron grating teeth / a ~70nm partial etch.
                                    # A resolution-convergence check is part of the
                                    # notebook's own sweep section, not asserted here.

    # --- vertical layer stack (generic MPW-typical SOI values) ---
    "si_thickness_um": 0.22,
    "box_thickness_um": 2.0,
    "clad_thickness_um": 2.0,
    "substrate_thickness_um": 1.5,  # substrate modeled INSIDE the cell (excl. PML) --
                                     # just needs to comfortably hold mon_down clear of
                                     # both the BOX interface and the PML.

    # --- grating geometry ---
    "etch_depth_um": 0.14,         # PROMOTED from a 6x6 etch_depth_um x duty_cycle grid
                                    # search (notebooks/09_grating_coupler.ipynb Section
                                    # 5.3) at n_periods=22, resolution=40 -- the highest
                                    # MEAN coupling_efficiency across the full analyzed
                                    # O-band of the 36 points tried (28.6%, peak 40.7% at
                                    # 1.3475um, close to the true band center). Selected on
                                    # the band-average, NOT the peak: a prior pass promoted
                                    # on peak alone (etch_depth_um=0.16/duty_cycle=0.6) and
                                    # only afterward found that point's spectrum had a
                                    # near-zero null almost exactly at the O-band center and
                                    # a high plateau confined to a narrow sub-band -- see
                                    # docs/simulation_settings_record.md for the full
                                    # comparison. This point's `energy_budget_check` PASSES
                                    # (unlike the peak-selected point, which failed it) -- a
                                    # materially more usable design, not just a different
                                    # number. NOTE: `coupling_efficiency_overlap()` itself
                                    # was also fixed (a real power-normalization bug -- see
                                    # that function's own docstring and
                                    # docs/troubleshooting_log.md) after this point was first
                                    # promoted; the 28.6%/40.7% figures here are the
                                    # corrected, re-measured values, not the original
                                    # (invalid, too-high) ones this point was first selected
                                    # under -- re-measurement confirmed it is STILL the best
                                    # of the 36 points by the corrected metric too.
    "period_um": 0.5094442234013399,  # PROMOTED: derive_grating_period()'s phase-matching
                                    # estimate at this module's n_eff_grating_guess/
                                    # fiber_angle_deg -- NOT re-tuned by the etch_depth_um
                                    # x duty_cycle sweep above (period sets phase-matching,
                                    # a separate axis). This promoted point's peak already
                                    # lands close to the O-band center (1.3475um of
                                    # 1.30-1.40um), so a follow-up period fine-sweep is lower
                                    # priority than it was for the earlier peak-only-selected
                                    # point -- see docs/simulation_settings_record.md. Pass
                                    # period_um=None to re-derive from scratch instead of
                                    # this promoted value.
    "duty_cycle": 0.4,              # PROMOTED alongside etch_depth_um above.
    "n_periods": 22,
    "fiber_angle_deg": 10.0,       # tilt in the x-z (simulated) plane, suppresses 2nd-
                                    # order Bragg back-reflection. Deliberately NOT
                                    # gdsfactory's own stock default (15deg) -- 8-10deg
                                    # is the more commonly cited range in the literature.
    "n_eff_grating_guess": 2.9,    # seed for derive_grating_period() -- the MEASURED
                                    # (not guessed) n_eff of the UNETCHED 220nm Si slab's
                                    # fundamental physically-TE mode at the O-band center
                                    # (see module docstring's TE/TM measurement). The
                                    # real (partially-etched) grating region's average
                                    # index is somewhat lower, refined by the notebook's
                                    # own sweep.

    # --- x-y layout metadata -- UNUSED by this module's x-z geometry (y-invariant,
    # see module docstring); passed through to grating_coupler_gds.py so the two
    # projections of the same design stay in sync. ---
    "width_grating_um": 11.0,
    "taper_length_um": 20.0,

    # --- fiber coupling-efficiency overlap ---
    "fiber_mfd_um": 9.2,           # approximate SMF-28 mode-field diameter near the
                                    # O-band center (~1.31um) -- smaller than the
                                    # commonly-quoted 10.4um C-band (~1.55um) value.

    "n_freq": 41,
    "z_mon_up_offset_um": 0.5,     # near-field/flux monitor height above the grating's
                                    # unetched Si top surface.
    "margin_um": 0.5,              # generic clearance: structure/monitor to PML.
    "input_lead_um": 2.0,          # straight input lead before the grating, holding
                                    # the source and reflection monitor.
    "min_sim_time_factor": 2.0,    # minimum_run_time = this * core_index * cell_x --
                                    # same long-cell floor discipline as mzi.py/
                                    # spiral_gds.py (CLAUDE.md section 7).
}


@dataclass
class PermittivityMap:
    eps: np.ndarray
    extent_um: tuple


@dataclass
class FieldSnapshot:
    field: np.ndarray
    eps: np.ndarray
    extent_um: tuple
    component: str      # "Ez" or "Hy" -- see module docstring's TE/TM measurement.


@dataclass
class NearFieldSnapshot:
    """Complex tangential E and H fields along the near-field monitor line, one
    row per analyzed frequency -- the raw input to coupling_efficiency_overlap().
    Both are needed for a power-normalized (not just shape-normalized) overlap --
    see that function's docstring."""
    field: np.ndarray       # complex Ez(Meep)=Ey(physical), shape (n_freq, n_x)
    hx: np.ndarray          # complex Hx(Meep)=Hx(physical), shape (n_freq, n_x)
    x_um: np.ndarray        # shape (n_x,), absolute x positions along the monitor line
    wavelengths_um: np.ndarray  # shape (n_freq,)
    component: str


@dataclass
class GratingCouplerResult:
    wavelengths_um: np.ndarray
    freqs: np.ndarray
    reflection: np.ndarray          # |S11|^2, guided mode reflected back into the input
    upward_power: np.ndarray        # fraction of incident power radiated upward
    downward_power: np.ndarray      # fraction of incident power radiated into the substrate
    coupling_efficiency: np.ndarray  # fiber-mode overlap efficiency (see module docstring)
    port_names: tuple = ("o1",)     # ONE guided port -- everything else is radiation,
                                     # not a second bound port (see simulate_baseline).
    permittivity: PermittivityMap = None
    field_snapshot: FieldSnapshot = None
    near_field: NearFieldSnapshot = None
    sim_params: dict = field(default_factory=dict)


@dataclass
class FiberIncidenceResult:
    """Result of `simulate_fiber_incidence()` -- the REVERSE direction (fiber
    -> grating -> waveguide), a qualitative confirmation only. See that
    function's docstring for why `guided_mode_power_relative` is uncalibrated
    (not a percentage efficiency, unlike GratingCouplerResult's fields)."""
    freqs: np.ndarray
    wavelengths_um: np.ndarray
    guided_mode_power_relative: np.ndarray  # |alpha[0,:,1]|**2, UNCALIBRATED
    permittivity: PermittivityMap
    field_snapshot: FieldSnapshot
    sim_params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Domain / geometry
# ---------------------------------------------------------------------------
def derive_grating_period(params: dict) -> float:
    """First-order grating phase-matching estimate:

        period_um = wl_center_um / (n_eff_guess - clad_index * sin(fiber_angle))

    `n_eff_guess` defaults to the MEASURED unetched-slab n_eff (see
    DEFAULT_PARAMS["n_eff_grating_guess"]'s comment) -- a starting point only,
    refined by the notebook's own empirical etch_depth_um x duty_cycle sweep
    (the real, partially-etched grating region's average index differs from
    the unetched slab this estimate is seeded from)."""
    wl_center_um = (params["wl_min_um"] + params["wl_max_um"]) / 2
    theta = np.radians(params["fiber_angle_deg"])
    return wl_center_um / (params["n_eff_grating_guess"] - params["clad_index"] * np.sin(theta))


def derive_domain(params: dict) -> dict:
    """Single source of truth for every x/z position and the cell size --
    shared by build_geometry, get_permittivity_map, and _make_simulation, so
    all three always agree (same role as every other module's own
    _compute_domain/_domain).

    x layout (grating starts at x=0, matching this module's own convention --
    NOT the toolkit's usual "device centered at x=0"):
        x_grating_start=0 -- n_periods*period_um --> x_grating_end
        x_src (source) < x_refl_mon (reflection monitor) < x_grating_start
        x_left = x_src - margin_um ; x_right = x_grating_end + margin_um
        cell_x = (x_right - x_left) + 2*dpml_um

    z layout (physical vertical, stored in Meep's own y slot):
        z_box_bot=0 -- box_thickness_um --> z_box_top -- si_thickness_um --> z_si_top
        z_etch_top = z_si_top - etch_depth_um (grating trenches only)
        z_mon_up = z_si_top + z_mon_up_offset_um
        z_sub_bot = z_box_bot - substrate_thickness_um ; z_mon_down = z_sub_bot + margin_um
        (top cladding has no explicit block -- default_material=clad fills it)
    """
    period_um = params["period_um"] if params["period_um"] is not None else derive_grating_period(params)

    x_grating_start = 0.0
    x_grating_end = x_grating_start + params["n_periods"] * period_um
    x_grating_center = (x_grating_start + x_grating_end) / 2
    x_src = x_grating_start - params["input_lead_um"]
    x_refl_mon = x_src + 0.3
    x_left = x_src - params["margin_um"]
    x_right = x_grating_end + params["margin_um"]
    cell_x = (x_right - x_left) + 2 * params["dpml_um"]
    x_center = (x_left + x_right) / 2

    z_box_bot = 0.0
    z_box_top = z_box_bot + params["box_thickness_um"]
    z_si_top = z_box_top + params["si_thickness_um"]
    z_etch_top = z_si_top - params["etch_depth_um"]
    z_clad_top = z_si_top + params["clad_thickness_um"]
    z_mon_up = z_si_top + params["z_mon_up_offset_um"]
    z_sub_bot = z_box_bot - params["substrate_thickness_um"]
    z_mon_down = z_sub_bot + params["margin_um"]
    cell_z = (z_clad_top - z_sub_bot) + 2 * params["dpml_um"]
    z_center = (z_clad_top + z_sub_bot) / 2

    # Vertical mode-monitor span for the source/reflection monitor -- a bit into the
    # BOX and a bit into the top cladding, comfortably catching the guided mode's
    # evanescent tails without extending all the way to the PML.
    z_mode_mon_bot = z_box_bot + 0.3
    z_mode_mon_top = z_si_top + 0.5
    z_mode_mon_center = (z_mode_mon_bot + z_mode_mon_top) / 2
    mode_mon_height = z_mode_mon_top - z_mode_mon_bot

    return dict(
        period_um=period_um,
        x_grating_start=x_grating_start, x_grating_end=x_grating_end, x_grating_center=x_grating_center,
        x_grating_span=x_grating_end - x_grating_start,
        x_src=x_src, x_refl_mon=x_refl_mon, x_left=x_left, x_right=x_right,
        cell_x=cell_x, x_center=x_center,
        z_box_bot=z_box_bot, z_box_top=z_box_top, z_si_top=z_si_top, z_etch_top=z_etch_top,
        z_clad_top=z_clad_top, z_mon_up=z_mon_up, z_sub_bot=z_sub_bot, z_mon_down=z_mon_down,
        cell_z=cell_z, z_center=z_center,
        z_mode_mon_center=z_mode_mon_center, mode_mon_height=mode_mon_height,
    )


def build_geometry(params: dict) -> list:
    """Ordered mp.Block list (later blocks override earlier -- standard Meep
    compositing, same convention every other module relies on): substrate,
    BOX, then one full-thickness Si block spanning the ENTIRE cell in x (input
    lead + grating base + right-of-grating stub -- extending Si to the cell
    edges is the same "input/output access stub" simplification
    waveguide.py's build_geometry_from_gds already uses), then n_periods
    clad-filled trenches punched into the grating region only. Top cladding
    has no explicit block: `default_material=clad` in `_base_simulation_kwargs`
    fills everything above z_si_top for free.

    No `build_geometry_from_gds()` counterpart -- see module docstring."""
    dom = derive_domain(params)
    substrate = mp.Medium(index=params["substrate_index"])
    core = mp.Medium(index=params["core_index"])
    clad = mp.Medium(index=params["clad_index"])

    sub_size_z = 2 * (params["substrate_thickness_um"] + params["dpml_um"] + params["margin_um"])
    geometry = [
        mp.Block(size=mp.Vector3(mp.inf, sub_size_z, mp.inf),
                  center=mp.Vector3(0, dom["z_box_bot"] - sub_size_z / 2, 0), material=substrate),
        mp.Block(size=mp.Vector3(mp.inf, dom["z_box_top"] - dom["z_box_bot"], mp.inf),
                  center=mp.Vector3(0, (dom["z_box_bot"] + dom["z_box_top"]) / 2, 0), material=clad),
        mp.Block(size=mp.Vector3(mp.inf, dom["z_si_top"] - dom["z_box_top"], mp.inf),
                  center=mp.Vector3(0, (dom["z_box_top"] + dom["z_si_top"]) / 2, 0), material=core),
    ]

    tooth_width = dom["period_um"] * params["duty_cycle"]
    trench_width = dom["period_um"] * (1 - params["duty_cycle"])
    etch_center_z = (dom["z_etch_top"] + dom["z_si_top"]) / 2
    etch_height = dom["z_si_top"] - dom["z_etch_top"]
    for i in range(params["n_periods"]):
        x0 = dom["x_grating_start"] + i * dom["period_um"] + tooth_width
        x1 = x0 + trench_width
        if trench_width <= 0:
            continue
        geometry.append(mp.Block(
            size=mp.Vector3(x1 - x0, etch_height, mp.inf),
            center=mp.Vector3((x0 + x1) / 2, etch_center_z, 0),
            material=clad,
        ))
    return geometry


def _base_simulation_kwargs(params: dict, dom: dict) -> dict:
    clad = mp.Medium(index=params["clad_index"])
    cell = mp.Vector3(dom["cell_x"], dom["cell_z"], 0)
    geometry_center = mp.Vector3(dom["x_center"], dom["z_center"], 0)
    return dict(
        cell_size=cell, boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=build_geometry(params), default_material=clad,
        resolution=params["resolution"], geometry_center=geometry_center,
    )


def get_permittivity_map(params: dict) -> PermittivityMap:
    """Permittivity map WITHOUT running any FDTD timestepping -- inspect the
    layer stack and grating teeth before paying for an expensive simulation,
    same discipline as every other module."""
    dom = derive_domain(params)
    sim = mp.Simulation(**_base_simulation_kwargs(params, dom))
    with _quiet_meep():
        sim.init_sim()
    eps = sim.get_epsilon()
    gc = mp.Vector3(dom["x_center"], dom["z_center"], 0)
    cell = mp.Vector3(dom["cell_x"], dom["cell_z"], 0)
    extent = (gc.x - cell.x / 2, gc.x + cell.x / 2, gc.y - cell.y / 2, gc.y + cell.y / 2)
    return PermittivityMap(eps=eps, extent_um=extent)


# ---------------------------------------------------------------------------
# Source / monitors / run
# ---------------------------------------------------------------------------
def _make_simulation(params: dict, capture_dft: bool, capture_near_field: bool, field_freq: float | None = None):
    dom = derive_domain(params)
    fcen = 1.0 / ((params["wl_min_um"] + params["wl_max_um"]) / 2)
    fwidth = 1.0 / params["wl_min_um"] - 1.0 / params["wl_max_um"]

    source = mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
        center=mp.Vector3(dom["x_src"], dom["z_mode_mon_center"]),
        size=mp.Vector3(0, dom["mode_mon_height"], 0),
        eig_band=1,
        eig_parity=mp.TM,   # selects the PHYSICALLY-TE mode in this rotated-axis
                             # convention -- see module docstring's measurement.
        eig_match_freq=True,
        eig_kpoint=mp.Vector3(1, 0, 0),
    )

    sim = mp.Simulation(sources=[source], **_base_simulation_kwargs(params, dom))

    mon_refl = sim.add_mode_monitor(
        fcen, fwidth, params["n_freq"],
        mp.ModeRegion(center=mp.Vector3(dom["x_refl_mon"], dom["z_mode_mon_center"]),
                      size=mp.Vector3(0, dom["mode_mon_height"])),
    )
    mon_up = sim.add_flux(
        fcen, fwidth, params["n_freq"],
        mp.FluxRegion(center=mp.Vector3(dom["x_grating_center"], dom["z_mon_up"]),
                      size=mp.Vector3(dom["x_grating_span"], 0)),
    )
    mon_down = sim.add_flux(
        fcen, fwidth, params["n_freq"],
        mp.FluxRegion(center=mp.Vector3(dom["x_center"], dom["z_mon_down"]),
                      size=mp.Vector3(dom["x_right"] - dom["x_left"], 0)),
    )

    dft_obj = None
    if capture_dft:
        # Whole-cell snapshot at a single frequency, for the field-distribution
        # plot -- Ez/Hy are this module's dominant-component pair (see module
        # docstring's TE/TM measurement), not Ez/Hz. Defaults to the O-band
        # CENTER frequency (fcen), but `field_freq` lets a caller request the
        # snapshot at any other frequency instead (e.g. a design's own
        # peak-coupling-efficiency wavelength, which need not be fcen -- see
        # docs/simulation_settings_record.md's note on the promoted point's
        # sharply asymmetric spectrum).
        snap_freq = field_freq if field_freq is not None else fcen
        dft_obj = sim.add_dft_fields(
            [mp.Ez, mp.Hy], snap_freq, snap_freq, 1,
            center=mp.Vector3(dom["x_center"], dom["z_center"]),
            size=mp.Vector3(dom["cell_x"], dom["cell_z"]),
        )

    near_field_obj = None
    if capture_near_field:
        # Tangential E (Ez=phys Ey) AND H (Hx=phys Hx) along the near-field
        # line, at EVERY analyzed frequency -- both feed coupling_efficiency_
        # overlap()'s power-normalized reciprocity overlap (see that
        # function's docstring; Hx is the component paired with Ez in this
        # module's Poynting-flux formula, Sz_phys = Re(Ez*conj(Hx)) -- same
        # mapping validated via the diagnostic Poynting-vector check recorded
        # in docs/troubleshooting_log.md).
        near_field_obj = sim.add_dft_fields(
            [mp.Ez, mp.Hx], fcen, fwidth, params["n_freq"],
            center=mp.Vector3(dom["x_grating_center"], dom["z_mon_up"]),
            size=mp.Vector3(dom["x_grating_span"], 0),
        )

    return sim, mon_refl, mon_up, mon_down, dft_obj, near_field_obj, dom


def fiber_gaussian_field(x_um: np.ndarray, x0_um: float, wl_um: float, mfd_um: float,
                          fiber_angle_deg: float, clad_index: float) -> np.ndarray:
    """Tilted Gaussian field profile of the fiber mode, evaluated on the
    near-field monitor line (in the x-z plane, at fixed z):

        E_fiber(x) = exp(-[(x-x0)*cos(theta)]^2 / w0^2) * exp(-i*k0*sin(theta)*(x-x0))

    `cos(theta)` foreshortens the fiber's circular Gaussian spot onto the
    tilted monitor plane; the phase term encodes the fiber's own tilted
    propagation direction (the same phase-matching condition
    derive_grating_period uses). The fiber mode is assumed to propagate
    through the top oxide cladding (index ~ clad_index) on its way to/from
    the chip surface -- an approximation (the fiber's own core index differs
    slightly), noted here rather than hidden."""
    theta = np.radians(fiber_angle_deg)
    w0 = mfd_um / 2.0
    k0 = 2 * np.pi / wl_um * clad_index
    dx = x_um - x0_um
    return np.exp(-((dx * np.cos(theta)) ** 2) / w0 ** 2) * np.exp(-1j * k0 * np.sin(theta) * dx)


def coupling_efficiency_overlap(x_um: np.ndarray, e_sim: np.ndarray, hx_sim: np.ndarray,
                                 wl_um: float, params: dict) -> float:
    """Power-normalized reciprocity overlap between the simulated near-field
    and the fiber's assumed Gaussian mode, at one wavelength:

        eta = |integral(Ey_sim*conj(Hx_fiber) + conj(Ey_fiber)*Hx_sim dx)|^2
              / (8 * P_sim * P_fiber)

        P_sim   = |Re(integral(Ey_sim * conj(Hx_sim) dx))|
        P_fiber = |Re(integral(Ey_fiber * conj(Hx_fiber) dx))|

    This is the standard reciprocity/mode-overlap formula (Snyder & Love;
    the same formula commercial mode solvers use for waveguide/fiber
    coupling-efficiency calculations). All quantities above are in the
    PHYSICAL (x, y, z) frame, `Ey_sim = -e_sim` (see the HANDEDNESS NOTE
    below for why the minus sign is required), `Hx_sim = hx_sim` (no sign
    change -- see below), `Hx_fiber = -clad_index*cos(theta)*Ey_fiber` from
    the plane-wave/paraxial Maxwell relation H = n*(k_hat x E) in Meep's own
    c=1 unit convention (impedance of a medium of index n is 1/n).

    HANDEDNESS NOTE (the actual bug behind an earlier, nearly-silent
    version of this fix that produced eta~=0 via near-total cancellation --
    see docs/troubleshooting_log.md): this module's own stated axis mapping,
    "Meep's invariant z axis represents physical y" (module docstring),
    swaps exactly two axes (Meep_y<->phys_z, Meep_z<->phys_y) relative to
    Meep's x. A single-axis-pair swap of a right-handed frame is
    orientation-REVERSING (e.g. x_meep x z_meep = -y_meep, not +y_meep), so
    naively reading `e_sim` (raw Ez[Meep]) as physical Ey without a sign
    correction silently breaks any cross-product-based (Poynting/reciprocity)
    calculation -- confirmed by re-deriving the physical Poynting vector
    directly in a genuine right-handed physical frame and finding
    Ey(phys) = -Ez(Meep), while Hx(phys) = Hx(Meep) and Hz(phys) = Hy(Meep)
    both need NO flip (only the axis being swapped, physical y, picks up the
    sign). Plain field-MAGNITUDE quantities (`field_snapshot`, the `Ez`/`Hy`
    dominant-component check in Section 5.2, `passivity_check`, flux
    monitors, etc.) were never affected -- Meep's own `add_flux`/
    `get_eigenmode_coefficients` compute physical power directly in Meep's
    native frame and never go through this relabeling. This bug only ever
    affected a from-scratch reciprocity/cross-product calculation built by
    hand on top of the raw Ez/Hx/Hy arrays, i.e. only this function.

    FIXES a real bug in the prior Phase-1 formula (shape-only overlap,
    normalized by integral(|E_sim|^2 dx) instead of by the field's own
    measured power): that version was NOT bounded by the actual radiated
    power (Cauchy-Schwarz only bounds it to <=1 relative to E_sim's own
    intensity-derived normalization, not relative to `upward_power`), and
    was empirically found to exceed `upward_power` by a wide margin at the
    prior promoted design's own peak wavelength (coupling_efficiency=77%
    vs. upward_power=38% -- physically impossible, since coupling_efficiency
    can be at most the fraction of incident power that reaches the near-
    field plane at all). See docs/troubleshooting_log.md for the full
    diagnostic (Poynting-flux height-invariance check ruling out evanescent
    contamination, then a restricted-fiber-window sensitivity test that
    still exceeded `upward_power` regardless of window choice, isolating the
    bug to the normalization itself, not the fiber-position assumption).
    This reciprocity form is properly bounded by construction (Cauchy-
    Schwarz on the true power cross-term), so `coupling_efficiency` can
    never exceed `upward_power` again.

    The fiber's assumed lateral position `x0_um` is still the intensity-
    weighted centroid of |E_sim|^2 along the monitor line (unchanged from
    the prior version -- a separate, already-documented idealization, not
    what this fix addresses; see docs/simulation_settings_record.md)."""
    ey_sim = -e_sim  # physical Ey -- see HANDEDNESS NOTE above
    intensity = np.abs(e_sim) ** 2
    x0_um = np.sum(x_um * intensity) / np.sum(intensity)
    theta = np.radians(params["fiber_angle_deg"])
    e_fiber = fiber_gaussian_field(x_um, x0_um, wl_um, params["fiber_mfd_um"],
                                    params["fiber_angle_deg"], params["clad_index"])
    hx_fiber = -params["clad_index"] * np.cos(theta) * e_fiber

    numerator = np.abs(np.trapezoid(
        ey_sim * np.conj(hx_fiber) + np.conj(e_fiber) * hx_sim, x_um)) ** 2
    p_sim = abs(np.trapezoid(ey_sim * np.conj(hx_sim), x_um).real)
    p_fiber = abs(np.trapezoid(e_fiber * np.conj(hx_fiber), x_um).real)
    return float(numerator / (8 * p_sim * p_fiber))


def sweep_etch_duty(base_params: dict, etch_depth_values, duty_cycle_values) -> list[dict]:
    """Cartesian-product grid search over (etch_depth_um, duty_cycle). Scores
    each point by BOTH its PEAK coupling_efficiency over the analyzed band
    AND its MEAN coupling_efficiency across that same band -- a point with a
    high peak can still have a near-zero response over most of the band (a
    sharply asymmetric spectrum with the peak concentrated in a narrow
    sub-band, empirically observed for at least one promoted point -- see
    docs/simulation_settings_record.md), so `coupling_efficiency_mean` is the
    more honest "how good is this design across the WHOLE O-band" metric,
    while `coupling_efficiency_max` stays useful context (achievable ceiling,
    if the band were re-centered via a period fine-sweep). This does NOT
    reuse `pic_toolkit.sweep.grid_sweep` (its `simulate_fn`/`validate_fn`
    contract is built around a pass/fail check plus a hardcoded
    `pic_toolkit.sparams` 2-port artifact write, neither of which fits this
    module's own local result shape -- same "own local infra, don't force
    the shared one" precedent as coupler.py/mzi.py already establish for a
    mismatched result shape). Returns one dict per grid point (NOT persisted
    to disk here -- the notebook decides what to keep and calls
    `save_artifact` on the winner)."""
    rows = []
    total = len(etch_depth_values) * len(duty_cycle_values)
    i = 0
    for etch_depth_um in etch_depth_values:
        for duty_cycle in duty_cycle_values:
            i += 1
            print(f"[{i}/{total}] etch_depth_um={etch_depth_um}, duty_cycle={duty_cycle} ...", flush=True)
            run_params = {**base_params, "etch_depth_um": etch_depth_um, "duty_cycle": duty_cycle}
            result = simulate_baseline(run_params)
            checks = run_all_checks(result)
            rows.append({
                "etch_depth_um": etch_depth_um, "duty_cycle": duty_cycle,
                "coupling_efficiency_max": float(result.coupling_efficiency.max()),
                "coupling_efficiency_argmax_wl_um": float(
                    result.wavelengths_um[result.coupling_efficiency.argmax()]),
                "coupling_efficiency_mean": float(result.coupling_efficiency.mean()),
                "passivity_passed": checks["passivity"]["passed"],
            })
    return rows


def _run(params: dict, capture_dft: bool = True, capture_near_field: bool = True, field_freq: float | None = None):
    """Single excitation: launch from the input waveguide toward the grating.
    UNLIKE every other module, there is no second guided port to excite from
    independently -- the "other side" is free-space radiation, not a bound
    mode -- so simulate_baseline() below does not run a second direction."""
    sim, mon_refl, mon_up, mon_down, dft_obj, near_field_obj, dom = _make_simulation(
        params, capture_dft, capture_near_field, field_freq=field_freq)
    min_sim_time = params["min_sim_time_factor"] * params["core_index"] * dom["cell_x"]
    with _quiet_meep():
        sim.run(until_after_sources=mp.stop_when_dft_decayed(minimum_run_time=min_sim_time))
        res_refl = sim.get_eigenmode_coefficients(mon_refl, [1], eig_parity=mp.TM)
        freqs = np.array(mp.get_flux_freqs(mon_refl))
        flux_up = np.array(mp.get_fluxes(mon_up))
        flux_down = np.array(mp.get_fluxes(mon_down))

        field_snapshot = None
        if capture_dft:
            ez = sim.get_dft_array(dft_obj, mp.Ez, 0)
            hy = sim.get_dft_array(dft_obj, mp.Hy, 0)
            eps = sim.get_array(component=mp.Dielectric)

        near_field = None
        if capture_near_field:
            n_freq = params["n_freq"]
            near = np.stack([sim.get_dft_array(near_field_obj, mp.Ez, i) for i in range(n_freq)])
            near_hx = np.stack([sim.get_dft_array(near_field_obj, mp.Hx, i) for i in range(n_freq)])
            n_x = near.shape[1]
            x_um = np.linspace(dom["x_grating_start"], dom["x_grating_end"], n_x)

    a_refl = res_refl.alpha[0, :, :]     # [:,0]=incident (toward grating), [:,1]=reflected
    incident_power = np.abs(a_refl[:, 0]) ** 2
    reflected_power = np.abs(a_refl[:, 1]) ** 2

    if capture_dft:
        if np.max(np.abs(ez)) >= np.max(np.abs(hy)):
            field, component = ez, "Ez"
        else:
            field, component = hy, "Hy"
        cell = mp.Vector3(dom["cell_x"], dom["cell_z"], 0)
        gc = mp.Vector3(dom["x_center"], dom["z_center"], 0)
        extent = (gc.x - cell.x / 2, gc.x + cell.x / 2, gc.y - cell.y / 2, gc.y + cell.y / 2)
        field_snapshot = FieldSnapshot(field=field, eps=eps, extent_um=extent, component=component)

    if capture_near_field:
        near_field = NearFieldSnapshot(
            field=near, hx=near_hx, x_um=x_um, wavelengths_um=1.0 / freqs, component="Ez",
        )

    return freqs, incident_power, reflected_power, flux_up, flux_down, field_snapshot, near_field, dom


def simulate_baseline(params: dict | None = None, field_snapshot_wl_um: float | None = None) -> GratingCouplerResult:
    """Full baseline characterization: launch the guided mode toward the
    grating, measure reflection back into the waveguide plus upward/downward
    radiated power (self-normalized against this run's own incident mode
    coefficient, same "self-normalize against a run's own incident power"
    discipline as coupler.py/mzi.py), and compute the fiber coupling
    efficiency via `coupling_efficiency_overlap` at every analyzed
    wavelength.

    `field_snapshot_wl_um`, if given, captures `field_snapshot` at that exact
    wavelength instead of the O-band center (the default) -- useful for
    inspecting the field at a design's own peak-coupling-efficiency
    wavelength, which is not necessarily the band center (see module
    docstring / docs/simulation_settings_record.md)."""
    dom_params = {**DEFAULT_PARAMS, **(params or {})}
    resolved_period = dom_params["period_um"] if dom_params["period_um"] is not None else derive_grating_period(dom_params)
    resolved = {**dom_params, "period_um": resolved_period}
    field_freq = 1.0 / field_snapshot_wl_um if field_snapshot_wl_um is not None else None

    freqs, incident_power, reflected_power, flux_up, flux_down, field_snapshot, near_field, dom = _run(
        resolved, field_freq=field_freq)

    reflection = reflected_power / incident_power
    upward_power = flux_up / incident_power
    # Sign convention: Meep's AUTOMATIC flux direction is not guaranteed to align with
    # "physically downward" -- taking the magnitude and relying on passivity_check /
    # energy_budget_check to catch anything unphysical, same "measure, then validate"
    # discipline as everywhere else in this toolkit, documented rather than asserted.
    downward_power = np.abs(flux_down) / incident_power

    coupling_efficiency = np.array([
        coupling_efficiency_overlap(near_field.x_um, near_field.field[i], near_field.hx[i],
                                     near_field.wavelengths_um[i], resolved)
        for i in range(len(freqs))
    ])

    permittivity = get_permittivity_map(resolved)
    sim_params = {
        **resolved,
        "meep_version": mp.__version__,
        "python_version": platform.python_version(),
        "creation_date": date.today().isoformat(),
    }
    return GratingCouplerResult(
        wavelengths_um=1.0 / freqs, freqs=freqs,
        reflection=reflection, upward_power=upward_power, downward_power=downward_power,
        coupling_efficiency=coupling_efficiency,
        permittivity=permittivity, field_snapshot=field_snapshot, near_field=near_field,
        sim_params=sim_params,
    )


def simulate_fiber_incidence(params: dict | None = None, wl_um: float | None = None) -> FiberIncidenceResult:
    """REVERSE direction: launch a tilted Gaussian beam FROM the fiber side
    (free space, at this design's own `fiber_angle_deg`), aimed down at the
    grating, and check it couples into the waveguide's guided mode --
    qualitative confirmation only, not the calibrated quantitative
    `reciprocity_check` still flagged as a further step in `run_all_checks()`'s
    own docstring. Uses `mp.GaussianBeam2DSource` (Meep's true 2D closed-form
    tilted-beam source -- NOT `GaussianBeamSource`/`GaussianBeam3DSource`,
    which deprecation-warn or use only an approximate cross-section in 2D).

    UNCALIBRATED: `GaussianBeam2DSource` normalizes via `beam_E0` (peak
    E-field amplitude), unlike `EigenModeSource(eig_match_freq=True)`'s
    automatic unit-incident-power normalization used everywhere else in this
    module -- there is no independent measurement of how much power this
    source actually launches. `guided_mode_power_relative` is therefore a
    relative/trend signal only, never a percentage efficiency; do not compare
    it numerically against `GratingCouplerResult.coupling_efficiency`.

    RECIPROCITY SIGN: the OUTGOING beam (grating -> fiber, see
    `derive_grating_period`/the permittivity-map fiber-tilt overlay) travels
    toward physical (+sin(theta), +cos(theta)) -- tilting toward +x as height
    increases. The reverse (fiber -> grating) ray, retracing that SAME
    physical path backward, must travel the OPPOSITE direction:
    (-sin(theta), -cos(theta)). Physical z = Meep y DIRECTLY here (no sign
    flip -- confirmed from `build_geometry()`'s own code, which uses physical
    z-values as Meep-y coordinates verbatim; contrast with physical y = Meep
    z, which DOES need a flip -- see `coupling_efficiency_overlap()`'s
    HANDEDNESS NOTE for that unrelated, already-fixed case), so `beam_kdir`
    below is `(-sin(theta), -cos(theta))` in Meep coordinates, not
    `(+sin(theta), -cos(theta))`.

    The beam's focus is placed at the grating surface such that the beam's
    OWN left (-x) edge -- the 1/e field radius `beam_w0`, horizontally
    foreshortened by the tilt to `beam_w0/cos(theta)` (same foreshortening
    convention `fiber_gaussian_field` already uses) -- lands exactly at
    `x_grating_start`. This is NOT an arbitrary choice: `simulate_baseline()`'s
    own OUTcoupling result shows the radiated near-field intensity (and hence,
    by reciprocity, this design's own receiving sensitivity) is concentrated
    in the first few periods near `x_grating_start`, not the aperture's
    geometric center -- see `coupling_efficiency_overlap()`'s docstring on why
    the near-field's own intensity centroid sits near the aperture start, not
    its center. Aiming the incident beam's edge (not its center) at
    `x_grating_start` keeps the beam's illuminated footprint over that same
    high-sensitivity region instead of mostly overshooting it toward +x (an
    earlier, geometric-center-aimed version measurably under-coupled -- see
    docs/troubleshooting_log.md)."""
    dom_params = {**DEFAULT_PARAMS, **(params or {})}
    resolved_period = dom_params["period_um"] if dom_params["period_um"] is not None else derive_grating_period(dom_params)
    resolved = {**dom_params, "period_um": resolved_period}
    dom = derive_domain(resolved)

    fcen = 1.0 / ((resolved["wl_min_um"] + resolved["wl_max_um"]) / 2)
    fwidth = 1.0 / resolved["wl_min_um"] - 1.0 / resolved["wl_max_um"]
    field_freq = 1.0 / wl_um if wl_um is not None else fcen
    theta = np.radians(resolved["fiber_angle_deg"])

    z_beam_src = dom["z_clad_top"] - resolved["margin_um"]
    beam_x0 = mp.Vector3(0, dom["z_si_top"] - z_beam_src, 0)
    beam_kdir = mp.Vector3(-np.sin(theta), -np.cos(theta), 0)
    beam_w0 = resolved["fiber_mfd_um"] / 2

    # Left (-x) edge of the beam's footprint at the grating surface lands at
    # x_grating_start -- see docstring. beam_x0 has no x-offset (0 above), so
    # the absolute focus x-coordinate equals the source center's own x.
    x_focus = dom["x_grating_start"] + beam_w0 / np.cos(theta)

    source = mp.GaussianBeam2DSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
        center=mp.Vector3(x_focus, z_beam_src, 0),
        size=mp.Vector3(dom["x_grating_span"], 0, 0),
        beam_x0=beam_x0, beam_kdir=beam_kdir, beam_w0=beam_w0,
        beam_E0=mp.Vector3(0, 0, 1),  # pure Meep-z (invariant-axis) polarization --
                                        # this module's physically-TE family, same as
                                        # eig_parity=mp.TM everywhere else.
    )
    sim = mp.Simulation(sources=[source], **_base_simulation_kwargs(resolved, dom))

    mon_refl = sim.add_mode_monitor(
        fcen, fwidth, resolved["n_freq"],
        mp.ModeRegion(center=mp.Vector3(dom["x_refl_mon"], dom["z_mode_mon_center"]),
                      size=mp.Vector3(0, dom["mode_mon_height"])),
    )
    dft_obj = sim.add_dft_fields(
        [mp.Ez, mp.Hy], field_freq, field_freq, 1,
        center=mp.Vector3(dom["x_center"], dom["z_center"]),
        size=mp.Vector3(dom["cell_x"], dom["cell_z"]),
    )

    min_sim_time = resolved["min_sim_time_factor"] * resolved["core_index"] * dom["cell_x"]
    with _quiet_meep():
        sim.run(until_after_sources=mp.stop_when_dft_decayed(minimum_run_time=min_sim_time))
        res_refl = sim.get_eigenmode_coefficients(mon_refl, [1], eig_parity=mp.TM)
        freqs = np.array(mp.get_flux_freqs(mon_refl))
        ez = sim.get_dft_array(dft_obj, mp.Ez, 0)
        hy = sim.get_dft_array(dft_obj, mp.Hy, 0)
        eps = sim.get_array(component=mp.Dielectric)

    # alpha[:,1] = mode traveling -x (away from the grating, out through the
    # input lead) -- the physically relevant "coupled into the waveguide"
    # signal here, same index as the forward run's "reflection" term because
    # it's the same direction of travel, not because it means the same thing
    # (see this function's own docstring).
    a_refl = res_refl.alpha[0, :, :]
    guided_mode_power_relative = np.abs(a_refl[:, 1]) ** 2

    if np.max(np.abs(ez)) >= np.max(np.abs(hy)):
        field, component = ez, "Ez"
    else:
        field, component = hy, "Hy"
    cell = mp.Vector3(dom["cell_x"], dom["cell_z"], 0)
    center = mp.Vector3(dom["x_center"], dom["z_center"], 0)
    extent = (center.x - cell.x / 2, center.x + cell.x / 2, center.y - cell.y / 2, center.y + cell.y / 2)
    field_snapshot = FieldSnapshot(field=field, eps=eps, extent_um=extent, component=component)

    permittivity = get_permittivity_map(resolved)
    sim_params = {
        **resolved,
        "meep_version": mp.__version__,
        "python_version": platform.python_version(),
        "creation_date": date.today().isoformat(),
    }
    return FiberIncidenceResult(
        freqs=freqs, wavelengths_um=1.0 / freqs,
        guided_mode_power_relative=guided_mode_power_relative,
        permittivity=permittivity, field_snapshot=field_snapshot, sim_params=sim_params,
    )


# ---------------------------------------------------------------------------
# Local validation -- deliberately NOT pic_toolkit.checks (hardcoded 2-port
# S-matrix shape), same rationale as coupler.py/mzi.py. A grating coupler
# radiates into a continuum and always leaks some power sideways/near the PML
# corners (finite-aperture diffraction beyond the flux monitors' finite
# span), so a strict energy-conservation check is not honest here -- see the
# two-tier design below.
# ---------------------------------------------------------------------------
def passivity_check(result: GratingCouplerResult, tol: float = 0.02) -> dict:
    """Primary, must-pass check: a passive device can never emit more power
    than it received. reflection + upward + downward must not exceed 1 (+
    a small numerical margin) at any wavelength -- the fundamental
    "did the simulation do something unphysical" check, same status as
    checks.passivity_check/coupler.py's own passivity_check."""
    total = result.reflection + result.upward_power + result.downward_power
    max_total = float(np.max(total))
    return {"max_total_power_fraction": max_total, "tolerance": tol, "passed": max_total < 1.0 + tol}


def energy_budget_check(result: GratingCouplerResult, max_unaccounted_fraction: float = 0.15) -> dict:
    """Secondary, INFORMATIVE check, not a hard gate: reflection + upward +
    downward should not fall too far short of 1, either -- but unlike a
    bounded 2-port/4-port device, some shortfall here is EXPECTED (sideways
    leakage past the flux monitors' finite span, finite-aperture diffraction)
    rather than a defect, so `max_unaccounted_fraction` is deliberately
    generous. Named `energy_budget_check`, not `energy_conservation_check`,
    to signal this weaker guarantee explicitly."""
    total = result.reflection + result.upward_power + result.downward_power
    min_total = float(np.min(total))
    lower_bound = 1.0 - max_unaccounted_fraction
    return {
        "min_total_power_fraction": min_total, "max_unaccounted_fraction": max_unaccounted_fraction,
        "passed": min_total > lower_bound,
    }


def run_all_checks(result: GratingCouplerResult, passivity_tol: float = 0.02,
                    max_unaccounted_fraction: float = 0.15) -> dict:
    """NOTE: no CALIBRATED reciprocity_check here. `simulate_fiber_incidence()`
    now launches a tilted Gaussian beam FROM the fiber side and reports a
    guided-mode signal, but that signal is amplitude- not power-normalized
    (see its own docstring) -- a QUALITATIVE confirmation only, not a
    quantitative cross-check against this function's own `coupling_efficiency`.
    A genuine calibrated reciprocity check (e.g. independently normalizing the
    fiber-incidence run's own launched power, then comparing directly against
    `coupling_efficiency`) remains a further, not-yet-done step."""
    passivity = passivity_check(result, tol=passivity_tol)
    budget = energy_budget_check(result, max_unaccounted_fraction=max_unaccounted_fraction)
    return {"passivity": passivity, "energy_budget": budget, "passed": bool(passivity["passed"])}


# ---------------------------------------------------------------------------
# Local artifact save/load -- own .npz+.json sidecar pair, same convention as
# coupler.py/mzi.py (this device's result shape doesn't fit pic_toolkit.sparams'
# hardcoded 2-port {"11".."22"} s_matrix any more than theirs did).
# ---------------------------------------------------------------------------
@dataclass
class GratingCouplerArtifact:
    wavelengths_um: np.ndarray
    freqs: np.ndarray
    reflection: np.ndarray
    upward_power: np.ndarray
    downward_power: np.ndarray
    coupling_efficiency: np.ndarray
    port_names: tuple
    metadata: dict


def save_artifact(path_stem, result: GratingCouplerResult, metadata: dict) -> None:
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path_stem.parent / (path_stem.name + ".npz"),
        wavelengths_um=result.wavelengths_um, freqs=result.freqs,
        reflection=result.reflection, upward_power=result.upward_power,
        downward_power=result.downward_power, coupling_efficiency=result.coupling_efficiency,
    )
    full_metadata = {**metadata, "port_names": list(result.port_names)}
    (path_stem.parent / (path_stem.name + ".json")).write_text(json.dumps(full_metadata, indent=2, default=str))


def load_artifact(path_stem) -> GratingCouplerArtifact:
    path_stem = Path(path_stem)
    npz_path = path_stem.parent / (path_stem.name + ".npz")
    json_path = path_stem.parent / (path_stem.name + ".json")
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"Grating coupler artifact not found at {path_stem}(.npz/.json). "
            "Run notebooks/08_grating_coupler.ipynb first to generate it."
        )
    data = np.load(npz_path)
    metadata = json.loads(json_path.read_text())
    return GratingCouplerArtifact(
        wavelengths_um=data["wavelengths_um"], freqs=data["freqs"],
        reflection=data["reflection"], upward_power=data["upward_power"],
        downward_power=data["downward_power"], coupling_efficiency=data["coupling_efficiency"],
        port_names=tuple(metadata["port_names"]), metadata=metadata,
    )
