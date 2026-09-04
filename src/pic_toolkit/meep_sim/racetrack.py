"""Meep simulation of an all-pass racetrack resonator: a bus waveguide
evanescently coupled to a RACETRACK ring (two half-circles joined by two
straight segments), rather than a plain circular ring.

This is a deliberate redesign of `the since-removed ring.py`'s device, not a variant of it
(this file is independent -- `the since-removed ring.py` is untouched). `the since-removed ring.py`'s
critical-coupling search swept the bus-ring GAP down to 0.13um and hit a
resolution/reciprocity wall with no turnover found (see that module's
`DEFAULT_PARAMS["gap_um"]` docstring). Here the gap is fixed at a
comfortable, easily-resolved 200nm, and the coupling-strength knob is
instead the length of the straight segment that runs parallel to the bus
(`coupling_length_um`) -- more interaction length, not a smaller gap, buys
stronger coupling.

Geometry: the racetrack annulus is built with the SAME "later object in the
geometry list carves a hole" trick `the since-removed ring.py` uses (there: two concentric
Cylinders; here: two concentric "stadium" outlines, each a Block spanning
the straight section plus a Cylinder at each end). At `coupling_length_um =
0` the two end-cylinders of each stadium coincide exactly, and the geometry
degenerates to a plain circular ring -- a useful sanity check, though not a
numeric match to `the since-removed ring.py`'s artifacts (different radius/gap there).

Round-trip length is now `L = 2*pi*radius_um + 2*coupling_length_um` (both
straight segments contribute, even though only the one nearest the bus
couples).

Geometry source: `build_gf_component()` builds the bus and ring as TWO
SEPARATE gdsfactory Components (not one combined Component) -- the ring
needs `D_conductivity=ring_conductivity` (lossy) while the bus stays
lossless, and `gds_component_to_prisms()` only accepts one `material=` per
call. The ring itself is a single CLOSED gdsfactory Path (straight -> 180-
degree arc -> straight -> 180-degree arc back to the start), extruded to
`wg_width_um` -- a genuinely simpler construction than native
`build_geometry()`'s outer+inner stadium-carving pair, since the extruded
closed loop is already hollow (confirmed empirically: KLayout represents it
as one self-touching "keyhole" polygon, `poly.holes()` reports 0, but the
interior IS correctly cladding-index when rendered). `build_geometry_from_gds()`
converts both Components via `gds_import.py` and is the default
`simulate_baseline()`/`_reference_incident()`/etc. use (`use_native_geometry=
True` selects the native path instead, kept for comparison). See
`_ring_gf_component()`'s docstring for a load-bearing gotcha:
`gf.path.straight(length=0)` is NOT a no-op and corrupts the
`coupling_length_um=0` degenerate (plain-ring) case if called instead of
omitted.

Only this file (plus `waveguide.py`, `bend.py`, `bend_topopt.py`, `the since-removed ring.py`)
imports meep.
"""

from __future__ import annotations

import contextlib
import io
import platform
from dataclasses import dataclass, field
from datetime import date

import meep as mp
import numpy as np

from . import gds_import

mp.verbosity(0)


@contextlib.contextmanager
def _quiet_meep():
    """Meep's C++ layer writes some messages (e.g. "grid volume is not an
    integer number of pixels" cell-rounding warnings) straight to stdout/
    stderr regardless of mp.verbosity() -- silence those around the actual
    init_sim()/run() calls below. Redirecting output doesn't affect Python
    exceptions, so a genuine error still propagates and is still visible.
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield

# Same source-bandwidth-margin fix as the since-removed ring.py: a Gaussian source's spectral
# power falls off toward the edges of its own (fcen, fwidth) window, so the
# SOURCE spectrum is made wider than the ANALYZED range while the analyzed
# (monitor) frequency grid stays at exactly wl_min..wl_max.
_SOURCE_BANDWIDTH_MARGIN = 2.0

# ---------------------------------------------------------------------------
# User-adjustable parameters.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "radius_um": 5.0,          # half-circle (centerline) radius -- kept at the since-removed ring.py's
                               # realistic, already-validated 5.0um. NOT swept here --
                               # coupling_length_um is the swept knob (see module docstring).
    "gap_um": 0.2,             # bus-racetrack coupling gap (edge to edge). FIXED at a
                               # comfortable 200nm, deliberately not pushed smaller --
                               # the since-removed ring.py's small-gap sweep hit a resolution/reciprocity
                               # wall below ~0.12um with no critical-coupling turnover
                               # found. Coupling strength is controlled by
                               # coupling_length_um instead.
    "coupling_length_um": 3.0, # length of the straight racetrack segment running
                               # parallel to the bus -- the primary coupling-strength
                               # knob. A first sweep across 0-5um (see Section 10) found
                               # this to be the closest approach to critical coupling,
                               # so it is used directly as the Section-5 baseline rather
                               # than an arbitrary middle-of-sweep value.
    "wg_width_um": 0.5,
    "wg_height_um": 0.22,
    "core_index": 2.7,
    "clad_index": 1.44,
    "ring_conductivity": 0.001,  # same rationale as the since-removed ring.py: a realistic radius has
                               # negligible geometric bend-radiation loss on its own, so
                               # a small intrinsic material loss is needed both for a
                               # meaningful critical-coupling target and for FDTD
                               # ring-down to converge in practical time.
    "wl_min_um": 1.3,
    "wl_max_um": 1.4,
    "n_freq": 301,             # dense grid, same as the since-removed ring.py -- resonance dips are only
                               # a few nm wide. Raised from 101 to 301 (found during the GDSFactory
                               # migration): the sharp, narrow on-resonance REFLECTION peaks (S11/S22,
                               # not just the transmission dip Section 8 already discusses) were
                               # severely under-sampled at 101 points/100nm, making the S11=S22 symmetry
                               # check highly sensitive to run-to-run floating-point/threading noise in
                               # exactly which grid point happened to land nearest each peak (observed:
                               # max|S11-S22| varying from 0.004 to 0.29 between two otherwise-identical
                               # runs at n_freq=101). Raising n_freq costs essentially nothing here --
                               # per-timestep DFT bookkeeping is negligible next to this device's
                               # dominant cost (GDS geometry setup, ~85s; the actual FDTD run itself is
                               # only ~4s even at min_sim_time=7000) -- so there is no real reason to
                               # tolerate the coarser grid's noise instead of just resolving it.
    "resolution": 25,          # pixels/um -- slightly coarser than the since-removed ring.py's 30 (a
                               # controlled test found spatial resolution changes the
                               # measured S-parameters by well under 1% -- resolution was
                               # NOT the dominant source of the |S|^2>1 bug, see
                               # min_sim_time below), traded down to afford the much
                               # longer min_sim_time this fix needs. Still resolves the
                               # 200nm gap with 5 grid points, comparable margin to
                               # bend.py's own resolution=25 for its curved geometry.
    "dpml_um": 1.0,
    "margin_um": 0.5,
    "dft_decay_tol": 1e-3,     # same as the since-removed ring.py -- NOT tightened. stop_when_dft_decayed's
                               # change/maxchange test is biased to register "plateaued"
                               # early (maxchange is fixed at the initial pulse's large
                               # transient, so a much smaller later change already looks
                               # small by comparison) regardless of tol -- tightening tol
                               # alone was tried and made runs dramatically slower without
                               # reliably fixing the |S|^2>1 symptom. min_sim_time below
                               # is the actual fix.
    "min_sim_time": 7000,      # floor on every FDTD run's length (meep time units, after
                               # the source has finished). A controlled scan (1500/3000/
                               # 6000/10000) showed the passivity violation shrinking
                               # monotonically and PLATEAUING at physically valid values
                               # (max|S_ij|^2 comfortably <= 1) from 6000 onward -- 1500
                               # (an earlier, insufficient attempt) was nowhere near
                               # enough. This is the actual fix: a longer coupling length
                               # couples into a higher-loaded-Q resonance whose ring-down
                               # decays slowly, and stop_when_dft_decayed's tol alone
                               # doesn't guarantee waiting that long.
    "max_sim_time": 25000,     # hard cap, well above min_sim_time so it is never the
                               # binding constraint in practice -- just a backstop.
                               # NOTE (see notebook Section 8): a later re-validation pass first
                               # (wrongly) suspected min_sim_time=7000 was ALSO too short to
                               # satisfy energy_conservation (|S11|^2+|S21|^2 close to 1),
                               # since that check still failed badly after fixing an unrelated
                               # reference-normalization bug. It was not a convergence problem --
                               # the saved S-parameters never exceed 1 anywhere (which
                               # under-convergence, unlike real physical loss, CAN produce and
                               # did in a deliberately-short min_sim_time=500 diagnostic used only
                               # to test that hypothesis), and the large deviations are clean,
                               # narrow, Lorentzian-shaped dips sitting exactly on resonance --
                               # genuine near-critical-coupling loss (this device dissipates up to
                               # ~99% of input power into ring_conductivity right at resonance,
                               # which is the whole point of a critical-coupling search), not a
                               # numerical artifact. min_sim_time=7000 needed no further increase;
                               # the actual fix was `energy_conservation_check` below, a
                               # resonance-aware validator that stops applying an off-resonance
                               # lossless expectation to the resonance dip itself.
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
    component: str
    wavelength_um: float = None  # which wavelength this snapshot was captured at -- None
                                  # only for backward-compatible/single-snapshot callers


@dataclass
class BaselineResult:
    wavelengths_um: np.ndarray
    freqs: np.ndarray
    s_matrix: dict
    port_names: tuple = ("o1", "o2")
    permittivity: PermittivityMap = None
    sim_params: dict = field(default_factory=dict)


def _compute_domain(params: dict) -> dict:
    """Derive the simulation domain and part positions from radius/gap/width/
    coupling_length. The y-direction stack (bus/gap/ring cross-section) is
    IDENTICAL to the since-removed ring.py's -- unaffected by coupling_length_um. Only the
    x-direction footprint grows, by coupling_length_um, to fit the racetrack's
    straight sections."""
    radius = params["radius_um"]
    gap = params["gap_um"]
    width = params["wg_width_um"]
    Lc = params["coupling_length_um"]
    dpml = params["dpml_um"]
    margin = params["margin_um"]

    cell_y = 2 * radius + gap + 2 * width + 2 * dpml + 2 * margin
    bus_y = radius + gap / 2 + width / 2
    ring_center_y = -(width / 2 + gap / 2)

    ring_outer_r = radius + width / 2
    ring_inner_r = radius - width / 2

    # Ports must clear the racetrack's full footprint -- half-length Lc/2 (the
    # straight section) plus the end-cap radius -- with the same clearance
    # margin the since-removed ring.py used for a plain circular ring's radius alone.
    half_length = Lc / 2 + ring_outer_r
    port_offset = half_length + margin + 1.0
    source_offset = port_offset + 0.5
    cell_x = 2 * (source_offset + dpml + 0.5)

    bus_mode_half_height = width / 2 + 0.6 * gap

    return {
        "cell_x": cell_x, "cell_y": cell_y,
        "bus_y": bus_y, "ring_center_y": ring_center_y,
        "ring_outer_r": ring_outer_r, "ring_inner_r": ring_inner_r,
        "half_length": half_length,
        "port_offset": port_offset, "source_offset": source_offset,
        "bus_mode_size_y": 2 * bus_mode_half_height,
    }


def build_geometry(params: dict) -> list:
    """Bus waveguide + racetrack ring (outer "stadium" of core material, with
    a smaller inner stadium of cladding material placed AFTER it to carve out
    the annulus -- the same later-object-precedence trick the since-removed ring.py uses, just
    generalized from a Cylinder pair to a Block+2*Cylinder stadium pair).

    Each stadium = a Block spanning the straight section (length Lc, height
    2*R) unioned with a full Cylinder of radius R centered at each end -- the
    Cylinders' "extra" halves that would poke into the straight section are
    already covered by the Block, so this union is exactly a stadium outline
    with no extra geometry logic needed. At coupling_length_um=0 the two
    end-Cylinders of each stadium coincide and the racetrack degenerates to a
    plain circular ring.
    """
    dom = _compute_domain(params)
    core = mp.Medium(index=params["core_index"])
    ring_core = mp.Medium(index=params["core_index"], D_conductivity=params["ring_conductivity"])
    clad = mp.Medium(index=params["clad_index"])
    width = params["wg_width_um"]
    Lc = params["coupling_length_um"]
    cy = dom["ring_center_y"]
    r_outer = dom["ring_outer_r"]
    r_inner = dom["ring_inner_r"]

    bus = mp.Block(
        size=mp.Vector3(mp.inf, width, mp.inf),
        center=mp.Vector3(0, dom["bus_y"]),
        material=core,
    )

    outer_parts = []
    inner_parts = []
    if Lc > 0:
        outer_parts.append(mp.Block(
            size=mp.Vector3(Lc, 2 * r_outer, mp.inf), center=mp.Vector3(0, cy), material=ring_core
        ))
        inner_parts.append(mp.Block(
            size=mp.Vector3(Lc, 2 * r_inner, mp.inf), center=mp.Vector3(0, cy), material=clad
        ))
    for sign in (-1, +1):
        outer_parts.append(mp.Cylinder(
            radius=r_outer, height=mp.inf, center=mp.Vector3(sign * Lc / 2, cy), material=ring_core
        ))
        inner_parts.append(mp.Cylinder(
            radius=r_inner, height=mp.inf, center=mp.Vector3(sign * Lc / 2, cy), material=clad
        ))

    return [bus, *outer_parts, *inner_parts]


def _bus_gf_component(params: dict):
    """The bus as a gdsfactory Component: a single straight spanning the full
    cell_x width at bus_y. Factored out of build_gf_component so
    _reference_incident can reuse the EXACT same bus discretization its
    bus-only reference run needs -- see build_geometry_from_gds's docstring
    for why a mismatched bus discretization there would reintroduce the
    left/right reference-amplitude bug this module's docstring already
    documents once."""
    import gdsfactory as gf

    dom = _compute_domain(params)
    cross_section = gf.cross_section.strip(width=params["wg_width_um"])
    straight = gf.components.straight(length=dom["cell_x"], cross_section=cross_section)

    centered = gf.Component()
    ref = centered.add_ref(straight)
    ref.move((-dom["cell_x"] / 2, dom["bus_y"]))
    centered.add_ports(ref.ports)
    return centered


def _ring_gf_component(params: dict):
    """The racetrack ring as ONE closed-loop gdsfactory Path (straight ->
    180-degree arc -> straight -> 180-degree arc, back to the start),
    extruded to wg_width_um -- a single annular Component, unlike
    build_geometry's native outer+inner stadium-carving pair. KLayout
    represents the resulting annulus as one self-touching "keyhole" simple
    polygon (`poly.holes()` reports 0), NOT a polygon-with-holes structure --
    confirmed empirically (direct epsilon sampling through
    gds_import.get_permittivity_map) that the interior is correctly hollow
    regardless, so gds_import.gds_component_to_prisms needs no changes.

    IMPORTANT: `gf.path.straight(length=0)` is NOT a geometric no-op -- at
    coupling_length_um=0 it corrupts the closed loop into a self-intersecting
    shape roughly 3.7x too wide (confirmed empirically: bbox width ~38.8um
    instead of the correct ~10.5um at radius_um=5). The `Lc > 0` branch below
    must omit the straight() calls entirely in the degenerate case, mirroring
    build_geometry's own `if Lc > 0:` branch.

    The closed loop's start and end point are coincident (a degenerate,
    meaningless "port" at the seam) -- this Component deliberately does NOT
    call add_ports(). Do not "fix" this by adding one back; there is no
    physically meaningful port on a closed ring.
    """
    import gdsfactory as gf

    dom = _compute_domain(params)
    radius = params["radius_um"]
    Lc = params["coupling_length_um"]
    cross_section = gf.cross_section.strip(width=params["wg_width_um"])

    if Lc > 0:
        path = gf.path.straight(length=Lc)
        path += gf.path.arc(radius=radius, angle=-180)
        path += gf.path.straight(length=Lc)
        path += gf.path.arc(radius=radius, angle=-180)
    else:
        path = gf.path.arc(radius=radius, angle=-180)
        path += gf.path.arc(radius=radius, angle=-180)

    ring = gf.path.extrude(path, cross_section=cross_section)

    # The closed centerline's own bbox is always x in [-radius, Lc+radius],
    # y in [-2*radius, 0] -- i.e. centered at (Lc/2, -radius) -- a fixed
    # topological consequence of the segment sequence above, confirmed
    # numerically at both Lc=3 and the degenerate Lc=0 case.
    centered = gf.Component()
    ref = centered.add_ref(ring)
    ref.move((-Lc / 2, dom["ring_center_y"] + radius))
    return centered


def build_gf_component(params: dict) -> tuple:
    """Returns (bus, ring): two SEPARATE gdsfactory Components in the shared
    _compute_domain() coordinate frame. Kept separate (not merged into one
    Component) because they need DIFFERENT Meep materials once converted --
    plain `core` for the bus vs. lossy `ring_core` (D_conductivity=
    ring_conductivity) for the ring -- see build_geometry_from_gds."""
    return _bus_gf_component(params), _ring_gf_component(params)


def build_geometry_from_gds(params: dict) -> list:
    """Meep geometry sourced from build_gf_component via gds_import -- the
    default simulation path (see use_native_geometry flags below). Simpler
    than native build_geometry: the ring's closed-loop Path extrusion is
    already hollow, so no separate outer-stadium/inner-stadium carving pair
    is needed."""
    core = mp.Medium(index=params["core_index"])
    ring_core = mp.Medium(index=params["core_index"], D_conductivity=params["ring_conductivity"])
    bus_gf, ring_gf = build_gf_component(params)
    return [
        *gds_import.gds_component_to_prisms(bus_gf, material=core),
        *gds_import.gds_component_to_prisms(ring_gf, material=ring_core),
    ]


def get_permittivity_map(params: dict, use_native_geometry: bool = False) -> PermittivityMap:
    """Permittivity map WITHOUT running any FDTD timestepping -- inspect the
    geometry (racetrack shape, gap, bus alignment, PML clearance) before
    paying for an expensive simulation. `use_native_geometry=True` uses
    build_geometry (native mp.Block/mp.Cylinder); the default (False) uses
    build_geometry_from_gds (GDSFactory-derived) -- see module docstring."""
    dom = _compute_domain(params)
    cell = mp.Vector3(dom["cell_x"], dom["cell_y"], 0)
    clad = mp.Medium(index=params["clad_index"])
    geometry = build_geometry(params) if use_native_geometry else build_geometry_from_gds(params)

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=geometry,
        default_material=clad,
        resolution=params["resolution"],
    )
    with _quiet_meep():
        sim.init_sim()
    eps = sim.get_epsilon()
    extent = (-cell.x / 2, cell.x / 2, -cell.y / 2, cell.y / 2)
    return PermittivityMap(eps=eps, extent_um=extent)


def _make_simulation(params: dict, source_x: float, direction_sign: float, field_snapshot_freqs=None,
                      use_native_geometry: bool = False):
    """`field_snapshot_freqs`, if given, is a list of Meep FREQUENCIES (not
    wavelengths -- 1/wavelength_um) at which to additionally capture a
    steady-state Hz field snapshot via `add_dft_fields`. `None` (the default)
    captures no field at all -- most calls (both reference runs, and the
    `o2` excitation run) don't need one; only a dedicated field-snapshot run
    (see `capture_field_snapshots`) passes a real list.
    """
    dom = _compute_domain(params)
    cell = mp.Vector3(dom["cell_x"], dom["cell_y"], 0)
    clad = mp.Medium(index=params["clad_index"])
    fcen = 1.0 / ((params["wl_min_um"] + params["wl_max_um"]) / 2)
    analysis_fwidth = 1.0 / params["wl_min_um"] - 1.0 / params["wl_max_um"]
    source_fwidth = analysis_fwidth * _SOURCE_BANDWIDTH_MARGIN
    mon_size_y = dom["bus_mode_size_y"]
    bus_y = dom["bus_y"]

    source = mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=source_fwidth),
        center=mp.Vector3(source_x, bus_y),
        size=mp.Vector3(0, mon_size_y, 0),
        eig_band=1,
        eig_parity=mp.TE,
        eig_match_freq=True,
        eig_kpoint=mp.Vector3(direction_sign, 0, 0),
    )

    geometry = build_geometry(params) if use_native_geometry else build_geometry_from_gds(params)
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=geometry,
        sources=[source],
        default_material=clad,
        resolution=params["resolution"],
    )

    port1_x = -dom["port_offset"]
    port2_x = dom["port_offset"]
    mon1 = sim.add_mode_monitor(
        fcen, analysis_fwidth, params["n_freq"],
        mp.ModeRegion(center=mp.Vector3(port1_x, bus_y), size=mp.Vector3(0, mon_size_y)),
    )
    mon2 = sim.add_mode_monitor(
        fcen, analysis_fwidth, params["n_freq"],
        mp.ModeRegion(center=mp.Vector3(port2_x, bus_y), size=mp.Vector3(0, mon_size_y)),
    )

    dft_obj = None
    if field_snapshot_freqs:
        # `freq=` (an explicit list) rather than the usual (fcen, df, nfreq) triple --
        # lets this capture the field at ARBITRARY, independently-chosen frequencies
        # (e.g. the resonance dip and an off-resonance point, see
        # capture_field_snapshots) rather than only ever the band center.
        dft_obj = sim.add_dft_fields(
            [mp.Hz], fcen, analysis_fwidth, len(field_snapshot_freqs),
            freq=list(field_snapshot_freqs), center=mp.Vector3(), size=cell,
        )

    return sim, mon1, mon2, dft_obj


def _reference_incident(params: dict, launch_from: str = "o1", use_native_geometry: bool = False):
    """Bus-only (no racetrack) reference simulation, used to normalize the
    with-ring runs below -- same rationale as the since-removed ring.py's _reference_incident:
    a resonant structure can leave energy circulating long after the source
    pulse itself has finished, contaminating self-normalization.

    `launch_from` selects which side the reference source/monitor sit on --
    "o1" (left, +x-launched) or "o2" (right, -x-launched), mirroring
    `_run_one_direction`'s own convention. A SEPARATE reference is required
    for each direction: although the bus-only geometry and domain are exactly
    x-mirror-symmetric, `source_offset`/`port_offset` are not exact multiples
    of the Yee grid spacing (1/resolution) -- confirmed empirically, a single
    left-launched reference under- or over-estimates the true right-launched
    incident amplitude by roughly 8-30% (shrinking, but not vanishing, as
    resolution increases from 15 to 25). An earlier version of this function
    took no `launch_from` argument and always launched from the left,
    reusing that one reference for BOTH `simulate_baseline`'s excitation
    directions -- this silently fed S22/S12 a systematically wrong
    denominator and was the dominant cause of a large (50-60% of total
    power) energy-conservation/reciprocity violation that looked, from the
    validation numbers alone, like a convergence problem but was not: the
    deviation did not shrink between `min_sim_time=500` and `min_sim_time=7000`,
    which a genuine ring-down under-convergence would have.

    The bus-only geometry here must be discretized THE SAME WAY the with-ring
    run's own bus is -- when `use_native_geometry=False` (the default), that
    means sourcing it from `_bus_gf_component` via gds_import, not an
    idealized `mp.Block`. A mismatched discretization here would reintroduce
    a bus-specific version of exactly the left/right reference-amplitude bug
    documented above: the reference run's incident amplitude would be
    measured on a subtly different bus than the one the with-ring run
    actually has, silently feeding S-parameters a wrong denominator again.
    """
    dom = _compute_domain(params)
    cell = mp.Vector3(dom["cell_x"], dom["cell_y"], 0)
    clad = mp.Medium(index=params["clad_index"])
    core = mp.Medium(index=params["core_index"])
    fcen = 1.0 / ((params["wl_min_um"] + params["wl_max_um"]) / 2)
    analysis_fwidth = 1.0 / params["wl_min_um"] - 1.0 / params["wl_max_um"]
    source_fwidth = analysis_fwidth * _SOURCE_BANDWIDTH_MARGIN
    mon_size_y = dom["bus_mode_size_y"]
    bus_y = dom["bus_y"]

    if launch_from == "o1":
        source_x = -dom["source_offset"]
        mon_x = -dom["port_offset"]
        direction_sign = +1
        fwd_index = 0  # forward (+x) component at the LEFT monitor is "incident" here
    else:
        source_x = dom["source_offset"]
        mon_x = dom["port_offset"]
        direction_sign = -1
        fwd_index = 1  # backward (-x) component at the RIGHT monitor is "incident" here

    if use_native_geometry:
        bus_only_geometry = [mp.Block(
            size=mp.Vector3(mp.inf, params["wg_width_um"], mp.inf),
            center=mp.Vector3(0, bus_y), material=core,
        )]
    else:
        bus_only_geometry = gds_import.gds_component_to_prisms(_bus_gf_component(params), material=core)
    source = mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=source_fwidth),
        center=mp.Vector3(source_x, bus_y),
        size=mp.Vector3(0, mon_size_y, 0),
        eig_band=1, eig_parity=mp.TE, eig_match_freq=True,
        eig_kpoint=mp.Vector3(direction_sign, 0, 0),
    )
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=bus_only_geometry,
        sources=[source],
        default_material=clad,
        resolution=params["resolution"],
    )
    mon = sim.add_mode_monitor(
        fcen, analysis_fwidth, params["n_freq"],
        mp.ModeRegion(center=mp.Vector3(mon_x, bus_y), size=mp.Vector3(0, mon_size_y)),
    )
    with _quiet_meep():
        sim.run(until_after_sources=mp.stop_when_dft_decayed(
            tol=params["dft_decay_tol"], minimum_run_time=params["min_sim_time"],
            maximum_run_time=params["max_sim_time"],
        ))
        res = sim.get_eigenmode_coefficients(mon, [1], eig_parity=mp.TE)
        freqs = np.array(mp.get_flux_freqs(mon))
    incident = res.alpha[0, :, fwd_index]
    return freqs, incident


def _run_one_direction(params: dict, launch_from: str, use_native_geometry: bool = False):
    """S-parameter-only excitation run -- no field capture. Field snapshots are
    handled entirely separately by `capture_field_snapshots`, because the
    wavelengths worth snapshotting (particularly the resonance dip) are only
    known AFTER a run like this one has already measured S(wavelength); DFT
    capture frequencies must be fixed before `sim.run()`, so there's no way to
    usefully fold field capture into the same run that discovers where the
    dip is.
    """
    dom = _compute_domain(params)
    if launch_from == "o1":
        source_x = -dom["source_offset"]
        direction_sign = +1
    else:
        source_x = dom["source_offset"]
        direction_sign = -1

    sim, mon1, mon2, _ = _make_simulation(params, source_x, direction_sign, field_snapshot_freqs=None,
                                            use_native_geometry=use_native_geometry)
    with _quiet_meep():
        sim.run(until_after_sources=mp.stop_when_dft_decayed(
            tol=params["dft_decay_tol"], minimum_run_time=params["min_sim_time"],
            maximum_run_time=params["max_sim_time"],
        ))
        res1 = sim.get_eigenmode_coefficients(mon1, [1], eig_parity=mp.TE)
        res2 = sim.get_eigenmode_coefficients(mon2, [1], eig_parity=mp.TE)
        freqs = np.array(mp.get_flux_freqs(mon1))

    a1 = res1.alpha[0, :, :]
    a2 = res2.alpha[0, :, :]
    return freqs, a1, a2


def capture_field_snapshots(params: dict | None, wavelengths_um: list, use_native_geometry: bool = False) -> list:
    """Run one additional o1-launched excitation purely to capture Hz field
    snapshots at the given wavelengths -- e.g. the resonance dip and a clean,
    off-resonance wavelength, so the two can be compared side by side
    (notebook Section 6, "field localized in the ring" vs. "field passing
    straight through"). Deliberately separate from `simulate_baseline()`:
    the interesting wavelengths (the dip especially) are only known AFTER
    `simulate_baseline()`'s own S(wavelength) sweep has already run, and DFT
    capture frequencies must be fixed before `sim.run()` starts, so there is
    no way to capture them within that same run.

    Captures Hz, not Ez -- see `_make_simulation`'s `eig_parity=mp.TE`: TE
    (Meep's own 2D convention: `Ex`,`Ey`,`Hz` nonzero, `Ez`,`Hx`,`Hy` ~0) is
    forced EXPLICITLY here, not discovered by checking what `eig_parity=
    mp.NO_PARITY` happens to prefer -- a direct MPB query at this device's
    width/index (0.5um core, 2.7/1.44 index contrast) finds `NO_PARITY`'s
    `eig_band=1` is actually the TM mode (n_eff=2.509), not TE (n_eff=2.409,
    the lower of the two here) -- so relying on `NO_PARITY` would have
    silently launched/measured TM, not TE (this is exactly the bug found and
    fixed in `waveguide.py`/`bend.py`/`bend_topopt.py`, all of which used to
    rely on `NO_PARITY`; `racetrack.py` never did -- `eig_parity=mp.TE` has
    been explicit here throughout). Capturing Ez under `eig_parity=mp.TE` is
    NOT an option: Meep's DFT field array degenerates to an empty/zero
    result for the OTHER field family when `eig_parity` selects one specific
    one -- `coupler.py` hit exactly this once it was also fixed to force
    `eig_parity=mp.TE` (it too used to rely on `NO_PARITY`), and now also
    captures both `Ez`/`Hz` and reports whichever is actually dominant
    (confirmed there too: `Ez` exactly 0, `Hz` dominant) rather than
    hardcoding either one.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    freqs_wanted = [1.0 / wl for wl in wavelengths_um]

    dom = _compute_domain(params)
    source_x = -dom["source_offset"]
    sim, mon1, mon2, dft_obj = _make_simulation(params, source_x, direction_sign=+1, field_snapshot_freqs=freqs_wanted,
                                                  use_native_geometry=use_native_geometry)
    with _quiet_meep():
        sim.run(until_after_sources=mp.stop_when_dft_decayed(
            tol=params["dft_decay_tol"], minimum_run_time=params["min_sim_time"],
            maximum_run_time=params["max_sim_time"],
        ))
        eps = sim.get_array(component=mp.Dielectric)
        hz_arrays = [sim.get_dft_array(dft_obj, mp.Hz, i) for i in range(len(freqs_wanted))]

    extent = (-dom["cell_x"] / 2, dom["cell_x"] / 2, -dom["cell_y"] / 2, dom["cell_y"] / 2)
    return [
        FieldSnapshot(field=hz, eps=eps, extent_um=extent, component="Hz", wavelength_um=wl)
        for hz, wl in zip(hz_arrays, wavelengths_um)
    ]


def simulate_baseline(params: dict | None = None, use_native_geometry: bool = False) -> BaselineResult:
    """Full two-port characterization: excite each port independently to
    measure the complete 2x2 S-matrix from genuinely independent data --
    same philosophy as the since-removed ring.py.

    Normalized against a separate bus-only reference run (_reference_incident)
    rather than self-normalized, for the same reason as the since-removed ring.py: a resonant
    structure's own monitor can be contaminated by recirculating energy.

    A SEPARATE reference is run for each excitation direction (see
    `_reference_incident`'s docstring) -- reusing a single left-launched
    reference for both directions was tried first and produced a large,
    spurious energy-conservation/reciprocity violation concentrated entirely
    in S22/S12 (the direction whose reference was silently wrong).

    `use_native_geometry=True` uses build_geometry (native mp.Block/
    mp.Cylinder); the default (False) uses build_geometry_from_gds
    (GDSFactory-derived) -- see module docstring.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}

    ref_freqs_L, incident_ref_L = _reference_incident(params, launch_from="o1", use_native_geometry=use_native_geometry)

    freqs, a1_fwd_run, a2_fwd_run = _run_one_direction(params, launch_from="o1", use_native_geometry=use_native_geometry)
    assert np.allclose(ref_freqs_L, freqs), "reference and with-ring runs must share the same frequency grid"
    reflected_1 = a1_fwd_run[:, 1]
    transmitted_21 = a2_fwd_run[:, 0]
    s11 = reflected_1 / incident_ref_L
    s21 = transmitted_21 / incident_ref_L

    ref_freqs_R, incident_ref_R = _reference_incident(params, launch_from="o2", use_native_geometry=use_native_geometry)
    _, a1_bwd_run, a2_bwd_run = _run_one_direction(params, launch_from="o2", use_native_geometry=use_native_geometry)
    assert np.allclose(ref_freqs_R, freqs), "reference and with-ring runs must share the same frequency grid"
    reflected_2 = a2_bwd_run[:, 0]
    transmitted_12 = a1_bwd_run[:, 1]
    s22 = reflected_2 / incident_ref_R
    s12 = transmitted_12 / incident_ref_R

    permittivity = get_permittivity_map(params, use_native_geometry=use_native_geometry)

    sim_params = {
        **params,
        "meep_version": mp.__version__,
        "python_version": platform.python_version(),
        "creation_date": date.today().isoformat(),
    }

    return BaselineResult(
        wavelengths_um=1.0 / freqs,
        freqs=freqs,
        s_matrix={"11": s11, "12": s12, "21": s21, "22": s22},
        permittivity=permittivity,
        sim_params=sim_params,
    )


# ---------------------------------------------------------------------------
# Resonator-specific energy-conservation check.
#
# WHY THIS EXISTS, instead of just calling checks.energy_conservation_check:
# that function applies ONE tolerance band, uniformly, to |S11|^2+|S21|^2 at
# every wavelength. That is the right model for a non-resonant device (a
# straight waveguide, a bend, a directional coupler -- loss should be roughly
# flat across the band), but it is the WRONG model for a resonator like this
# racetrack. At and near each resonance wavelength, a near-critical-coupled
# ring/racetrack is SUPPOSED to dissipate close to 100% of the input power
# into its own intrinsic loss channel (here, `ring_conductivity`) -- that is
# the entire physical mechanism critical-coupling search (Section 10) is
# looking for, not a defect. Everywhere else in the band (off resonance), the
# bus is only weakly perturbed by the ring and transmission should be close
# to lossless. A single global tolerance can't express "tight off resonance,
# loose on resonance" -- either it's tight enough to correctly catch bugs off
# resonance and then wrongly flags every genuine critical-coupling dip as a
# failure (this is what happened before this function existed -- see the
# notebook's Section 8 discussion of a "second bug" that, on closer
# inspection using an expert's challenge on exactly this point, turned out to
# be genuine physics, not a bug), or it's loosened enough to tolerate the
# on-resonance dip and then can no longer catch a genuine broadband
# normalization/convergence bug (which corrupts the WHOLE spectrum, not just
# a narrow dip near resonance).
# ---------------------------------------------------------------------------
def _find_resonance_dip_spans(total: np.ndarray, baseline: float, min_dip_depth: float) -> np.ndarray:
    """Identify which wavelength INDICES belong to a resonance dip, without
    any assumption about how many resonances there are or where they sit.

    Method: find local minima of `total` (points strictly no higher than both
    neighbors) that are at least `min_dip_depth` below the spectrum's own
    median (a robust stand-in for "off-resonance level" that isn't thrown off
    by a handful of narrow dips). From each such minimum, walk outward in
    both directions while `total` keeps recovering (weakly monotonically) --
    this traces out the FULL shoulder of a genuine Lorentzian-shaped
    resonance dip, not just its narrow bottom, and stops naturally wherever
    the trend reverses (e.g. at a saddle point between two nearby
    resonances, or once the off-resonance baseline is reached). No fixed
    linewidth or frequency window is assumed -- confirmed against this
    racetrack's actual data to correctly span a dip's full ~1.7nm shoulder
    (17 grid points at this notebook's narrowband n_freq=201/20nm sweep
    resolution) rather than just the 1-2 points nearest the exact minimum.
    """
    n = len(total)
    in_dip = np.zeros(n, dtype=bool)
    is_local_min = np.zeros(n, dtype=bool)
    is_local_min[1:-1] = (total[1:-1] <= total[:-2]) & (total[1:-1] <= total[2:])
    minima = [i for i in np.where(is_local_min)[0] if total[i] < baseline - min_dip_depth]

    for m in minima:
        i = m
        while i > 0 and total[i - 1] >= total[i] - 1e-9:
            i -= 1
        j = m
        while j < n - 1 and total[j + 1] >= total[j] - 1e-9:
            j += 1
        in_dip[i:j + 1] = True
    return in_dip


def energy_conservation_check(s_matrix: dict, wavelengths_um: np.ndarray, off_resonance_tol: float = 0.02,
                               min_dip_depth: float = 0.1, passivity_tol: float = 0.01) -> dict:
    """Resonance-aware analog of checks.energy_conservation_check, for a
    single-bus (all-pass) resonator where a large, narrow on-resonance power
    deficit is expected physics, not a bug (see module-level comment above).

    For each port (1 and 2), the spectrum is split into two populations:
      - OFF-RESONANCE points (everywhere NOT inside a detected resonance dip,
        see `_find_resonance_dip_spans`) must sit within `off_resonance_tol`
        of perfectly lossless (|S11|^2+|S21|^2 ~= 1). This is the part of the
        check that actually catches a real bug: a broadband normalization or
        convergence error corrupts every wavelength, not just a narrow dip,
        so it always shows up here regardless of how deep any individual
        resonance happens to be.
      - ON-RESONANCE (dip) points are held to a much weaker bar: only that
        they stay physical (|S11|^2+|S21|^2 <= 1+passivity_tol everywhere,
        checked across the FULL spectrum, not just off-resonance points, so
        this still catches an amplification-type bug regardless of where it
        occurs). No minimum-loss requirement is imposed on dip points --
        near-total loss AT resonance is the intended result of a
        near-critical-coupled design, not something to flag.

    `wavelengths_um` is accepted (not just s_matrix) to keep this function's
    signature self-documenting about what it needs, even though the current
    implementation only uses array length/ordering, not the actual values --
    dip detection here works in INDEX space (assumes wavelengths_um is
    monotonic, true for every artifact this toolkit produces).
    """
    total1 = np.abs(s_matrix["11"]) ** 2 + np.abs(s_matrix["21"]) ** 2
    total2 = np.abs(s_matrix["22"]) ** 2 + np.abs(s_matrix["12"]) ** 2
    assert len(total1) == len(wavelengths_um), "s_matrix and wavelengths_um must have matching length"

    def _check_one(total):
        baseline = float(np.median(total))
        in_dip = _find_resonance_dip_spans(total, baseline, min_dip_depth)
        off_res = total[~in_dip]
        off_resonance_deviation = float(np.max(np.abs(off_res - 1.0))) if len(off_res) else 0.0
        # physicality (no amplification) is checked across the WHOLE spectrum, dip or not
        max_physicality_excess = max(0.0, float(np.max(total)) - (1.0 + passivity_tol))
        return {
            "off_resonance_baseline": baseline,
            "n_off_resonance_points": int((~in_dip).sum()),
            "n_resonance_points": int(in_dip.sum()),
            "off_resonance_deviation": off_resonance_deviation,
            "max_physicality_excess": max_physicality_excess,
            "passed": bool(off_resonance_deviation < off_resonance_tol and max_physicality_excess < 1e-6),
        }

    port1 = _check_one(total1)
    port2 = _check_one(total2)
    return {
        "port1": port1,
        "port2": port2,
        "off_resonance_tol": off_resonance_tol,
        "min_dip_depth": min_dip_depth,
        "passivity_tol": passivity_tol,
        "passed": bool(port1["passed"] and port2["passed"]),
    }
