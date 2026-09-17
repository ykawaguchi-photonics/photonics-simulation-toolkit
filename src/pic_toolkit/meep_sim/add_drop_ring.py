"""Meep simulation of an ADD-DROP ring resonator: the same RACETRACK ring
construction as `racetrack.py` (two half-circles joined by two straight
segments), but now with a bus waveguide at BOTH straight segments instead of
just one -- giving 4 physical ports instead of 2.

Geometric reuse of racetrack.py: `racetrack.py`'s ring already has two
straight segments of equal length `coupling_length_um` (that is what makes
it a "racetrack" and not a plain ring) -- only ONE of them sits near a bus
there. Here, a SECOND bus is placed mirrored across the ring's own center
line (`ring_center_y`), coupled to the other straight segment, using the
SAME `gap_um`/`coupling_length_um` (both couplers are symmetric by design --
this device does not support independent left/right coupling). The ring
geometry construction itself (stadium-carving for the native path, closed
Path extrusion for the gdsfactory path) is unchanged from racetrack.py.

Ports: `input`/`through` on the first ("input") bus, `add`/`drop` on the
second ("add/drop") bus. Physical layout (mirrors racetrack's own
port1=left/port2=right convention): `input` and `drop` sit on the -x side,
`through` and `add` sit on the +x side. This follows from co-directional
coupling -- light launched rightward into `input` couples into the ring
moving rightward at that straight section, hence LEFTWARD at the opposite
(mirrored) straight section, which co-directionally couples into the
add/drop bus moving leftward -- i.e. towards its -x end, `drop`. This is a
geometric hypothesis, confirmed empirically in the notebook's field
snapshots (Section 11) by checking which physical port actually lights up
when only `input` is excited on resonance.

This module is DELIBERATELY self-contained -- it does not import
`racetrack.py` (whose ring-building helpers are private/underscore-prefixed
and tied to that module's own single-bus assumptions) and does not route
through `pic_toolkit.sparams`/`pic_toolkit.checks` (hardcoded to a 2x2
S-matrix). This mirrors the precedent already set by `coupler.py`/`mzm.py`/
`mzi.py`, the toolkit's other genuinely-N-port devices: each keeps its own
local artifact I/O and its own local energy/reciprocity/passivity checks.

Normalization: like `racetrack.py` (and UNLIKE `coupler.py`), this is a
resonant structure, so self-normalization against a run's own incident
coefficient is unsafe (recirculating energy in the ring can contaminate a
monitor long after the source pulse itself has finished). Every excitation
run below is normalized against a SEPARATE bus-only reference run, launched
from the same physical port in the same direction -- exactly racetrack's own
hard-learned fix (an earlier version of that module reused one reference for
both directions and silently fed the wrong denominator to S22/S12).

Critical-coupling search knob: SAME choice as racetrack.py -- `gap_um` is
fixed (0.20um, comfortably above racetrack's own documented ~0.12um
resolution/reciprocity wall) and `coupling_length_um` is swept. A `gap_um`
sweep was tried FIRST for this module's notebook and abandoned: it ran
cleanly (validation passing at every point except the very weakest
coupling) but showed a MONOTONIC through-port-extinction trend across a
resolution-safe 0.15-0.35um range with no interior turnover -- the true
critical-coupling gap apparently sits below 0.15um, in the same territory
`racetrack.py`'s own predecessor (`ring.py`, since removed) already
documented hitting a wall on. A gap sweep also makes the measured
S-parameters progressively more sensitive to grid resolution as the gap
shrinks; `coupling_length_um` avoids that entirely (see
`docs/simulation_settings_record.md`'s `add_drop_ring.py` section for the
full record of both sweeps).
"""

from __future__ import annotations

import contextlib
import io
import platform
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import meep as mp
import numpy as np

from . import gds_import

mp.verbosity(0)


@contextlib.contextmanager
def _quiet_meep():
    """See racetrack.py's identical helper: Meep's C++ layer writes some
    messages straight to stdout/stderr regardless of mp.verbosity() --
    silence those around sim.init_sim()/sim.run(). A genuine Python
    exception still propagates normally."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


# Same source-bandwidth-margin fix as racetrack.py/the since-removed ring.py.
_SOURCE_BANDWIDTH_MARGIN = 2.0

# ---------------------------------------------------------------------------
# User-adjustable parameters. Same defaults and same sweep-knob choice as
# racetrack.py's DEFAULT_PARAMS -- gap_um fixed, coupling_length_um swept
# (see module docstring for why a gap sweep was tried and abandoned here).
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "radius_um": 5.0,
    "gap_um": 0.20,              # FIXED throughout -- same value and rationale as
                                  # racetrack.py (both couplers share it, symmetric by design).
    "coupling_length_um": 3.0,   # baseline/starting value -- SWEPT in the notebook's
                                  # Section 9 critical-coupling search (both couplers
                                  # move together, symmetric by design -- both straight
                                  # segments always share this same length).
    "wg_width_um": 0.5,
    "wg_height_um": 0.22,
    "core_index": 2.7,
    "clad_index": 1.44,
    "ring_conductivity": 0.001,  # same rationale as racetrack.py: negligible geometric
                                  # bend-radiation loss at this radius, so a small
                                  # intrinsic material loss is needed for a meaningful
                                  # critical-coupling target and for FDTD ring-down to
                                  # converge in practical time.
    "wl_min_um": 1.3,
    "wl_max_um": 1.4,
    "n_freq": 301,
    "resolution": 25,
    "dpml_um": 1.0,
    "margin_um": 0.5,
    "dft_decay_tol": 1e-3,
    "min_sim_time": 7000,
    "max_sim_time": 25000,
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
    wavelength_um: float = None


@dataclass
class AddDropRingResult:
    wavelengths_um: np.ndarray
    freqs: np.ndarray
    # From exciting "input" (launched +x into the input bus):
    S_input_refl: np.ndarray = None     # input -> input (reflection)
    S_input_through: np.ndarray = None  # input -> through
    S_input_drop: np.ndarray = None     # input -> drop
    S_input_add: np.ndarray = None      # input -> add (direct bus-to-bus leakage)
    # From exciting "add" (launched -x into the add/drop bus):
    S_add_refl: np.ndarray = None       # add -> add (reflection)
    S_add_drop: np.ndarray = None       # add -> drop
    S_add_through: np.ndarray = None    # add -> through (direct bus-to-bus leakage)
    S_add_input: np.ndarray = None      # add -> input
    port_names: tuple = ("input", "through", "add", "drop")
    permittivity: PermittivityMap = None
    field_snapshot: FieldSnapshot = None
    sim_params: dict = field(default_factory=dict)


@dataclass
class AddDropRingArtifact:
    wavelengths_um: np.ndarray
    freqs: np.ndarray
    S_input_refl: np.ndarray
    S_input_through: np.ndarray
    S_input_drop: np.ndarray
    S_input_add: np.ndarray
    S_add_refl: np.ndarray
    S_add_drop: np.ndarray
    S_add_through: np.ndarray
    S_add_input: np.ndarray
    port_names: tuple
    metadata: dict


def _compute_domain(params: dict) -> dict:
    """Racetrack's domain math, extended with a SECOND (add/drop) bus placed
    mirrored across the ring's own center line -- `drop_bus_y = 2*ring_center_y
    - bus_y` -- reproducing the SAME `gap_um` clearance on the opposite side
    of the ring (this is an exact geometric consequence of the ring being a
    stadium centered at `ring_center_y`, independent of any cell-sizing
    choice below).

    `cell_y` must now reserve dpml+margin clearance beyond BOTH buses'
    outer edges (racetrack.py only ever needed this on one side -- the far
    side there was empty cladding out to the PML). Sized generously here
    (confirmed/tightened empirically via the notebook's Section 4
    permittivity-map inspection, per CLAUDE.md's "measure first" tolerance
    discipline) rather than derived to the exact minimum.
    """
    radius = params["radius_um"]
    gap = params["gap_um"]
    width = params["wg_width_um"]
    Lc = params["coupling_length_um"]
    dpml = params["dpml_um"]
    margin = params["margin_um"]

    bus_y = radius + gap / 2 + width / 2
    ring_center_y = -(width / 2 + gap / 2)
    drop_bus_y = 2 * ring_center_y - bus_y

    ring_outer_r = radius + width / 2
    ring_inner_r = radius - width / 2

    # The Meep cell is centered at y=0, but the geometry itself is NOT
    # symmetric about y=0 (ring_center_y != 0, so bus_y and drop_bus_y sit
    # at different distances from the origin) -- sizing cell_y from the
    # TOTAL span between the two buses (as a naive extension of racetrack's
    # single-bus formula would) can under-allocate clearance on whichever
    # side happens to sit farther from y=0. Instead, take the larger of the
    # two buses' own distance-from-center, and require dpml+margin clearance
    # beyond THAT -- guarantees full clearance on both sides (verified
    # numerically: the nearer side ends up with extra, harmless slack).
    half_extent_needed = max(bus_y, abs(drop_bus_y)) + width / 2 + dpml + margin
    cell_y = 2 * half_extent_needed

    half_length = Lc / 2 + ring_outer_r
    port_offset = half_length + margin + 1.0
    source_offset = port_offset + 0.5
    cell_x = 2 * (source_offset + dpml + 0.5)

    bus_mode_half_height = width / 2 + 0.6 * gap

    return {
        "cell_x": cell_x, "cell_y": cell_y,
        "bus_y": bus_y, "drop_bus_y": drop_bus_y, "ring_center_y": ring_center_y,
        "ring_outer_r": ring_outer_r, "ring_inner_r": ring_inner_r,
        "half_length": half_length,
        "port_offset": port_offset, "source_offset": source_offset,
        "bus_mode_size_y": 2 * bus_mode_half_height,
    }


def build_geometry(params: dict) -> list:
    """Native mp.Block/mp.Cylinder geometry: racetrack's outer+inner stadium
    carving pair (unchanged), plus TWO bus mp.Blocks -- one at `bus_y`
    (input bus) and one at `drop_bus_y` (add/drop bus)."""
    dom = _compute_domain(params)
    core = mp.Medium(index=params["core_index"])
    ring_core = mp.Medium(index=params["core_index"], D_conductivity=params["ring_conductivity"])
    clad = mp.Medium(index=params["clad_index"])
    width = params["wg_width_um"]
    Lc = params["coupling_length_um"]
    cy = dom["ring_center_y"]
    r_outer = dom["ring_outer_r"]
    r_inner = dom["ring_inner_r"]

    input_bus = mp.Block(
        size=mp.Vector3(mp.inf, width, mp.inf),
        center=mp.Vector3(0, dom["bus_y"]),
        material=core,
    )
    drop_bus = mp.Block(
        size=mp.Vector3(mp.inf, width, mp.inf),
        center=mp.Vector3(0, dom["drop_bus_y"]),
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

    return [input_bus, drop_bus, *outer_parts, *inner_parts]


def _bus_gf_component(params: dict, y_um: float):
    """A single straight bus at an arbitrary y -- generalizes racetrack's
    `_bus_gf_component` (hardcoded to `bus_y`) so both the input bus and the
    add/drop bus can share this one helper."""
    import gdsfactory as gf

    dom = _compute_domain(params)
    cross_section = gf.cross_section.strip(width=params["wg_width_um"])
    straight = gf.components.straight(length=dom["cell_x"], cross_section=cross_section)

    centered = gf.Component()
    ref = centered.add_ref(straight)
    ref.move((-dom["cell_x"] / 2, y_um))
    centered.add_ports(ref.ports)
    return centered


def _ring_gf_component(params: dict):
    """Identical construction to racetrack.py's `_ring_gf_component` --
    closed racetrack loop (straight -> 180-degree arc -> straight ->
    180-degree arc), no ports (see that module's docstring for the
    `gf.path.straight(length=0)` gotcha at the degenerate coupling_length_um=0
    case, which is handled the same way here)."""
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

    centered = gf.Component()
    ref = centered.add_ref(ring)
    ref.move((-Lc / 2, dom["ring_center_y"] + radius))
    return centered


def build_gf_component(params: dict) -> tuple:
    """Returns (input_bus, drop_bus, ring): THREE separate gdsfactory
    Components -- the ring needs the lossy `ring_core` material once
    converted, both buses stay lossless `core`."""
    dom = _compute_domain(params)
    return (
        _bus_gf_component(params, dom["bus_y"]),
        _bus_gf_component(params, dom["drop_bus_y"]),
        _ring_gf_component(params),
    )


def build_geometry_from_gds(params: dict) -> list:
    """Meep geometry sourced from build_gf_component via gds_import -- the
    default simulation path, mirroring racetrack.py's own default."""
    core = mp.Medium(index=params["core_index"])
    ring_core = mp.Medium(index=params["core_index"], D_conductivity=params["ring_conductivity"])
    input_bus_gf, drop_bus_gf, ring_gf = build_gf_component(params)
    return [
        *gds_import.gds_component_to_prisms(input_bus_gf, material=core),
        *gds_import.gds_component_to_prisms(drop_bus_gf, material=core),
        *gds_import.gds_component_to_prisms(ring_gf, material=ring_core),
    ]


def get_permittivity_map(params: dict, use_native_geometry: bool = False) -> PermittivityMap:
    """Permittivity map WITHOUT running any FDTD timestepping -- inspect
    both couplers' gaps, bus alignment, and PML clearance before paying for
    an expensive simulation."""
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


def _ports(params: dict) -> dict:
    """Physical port locations -- see module docstring for the input/
    through/add/drop x-side hypothesis (verified empirically in the
    notebook, not assumed here)."""
    dom = _compute_domain(params)
    return {
        "input": mp.Vector3(-dom["port_offset"], dom["bus_y"]),
        "through": mp.Vector3(dom["port_offset"], dom["bus_y"]),
        "drop": mp.Vector3(-dom["port_offset"], dom["drop_bus_y"]),
        "add": mp.Vector3(dom["port_offset"], dom["drop_bus_y"]),
    }


def _make_simulation(params: dict, launch_from: str, field_snapshot_freqs=None,
                      use_native_geometry: bool = False):
    """Build one 4-monitor simulation, sourced from `launch_from`
    ("input" -> launched +x into the input bus, "add" -> launched -x into
    the add/drop bus -- the only two excitations `simulate_baseline` uses,
    mirroring coupler.py's "excite the two input-side ports only"
    convention). Returns (sim, monitors dict keyed by port name, dft_obj).

    The SOURCE sits at `source_offset` (farther from the ring than
    `port_offset`) while the MONITORS sit at `port_offset` -- same
    source/monitor separation racetrack.py uses, so the excited pulse has
    room to be recorded distinctly rather than sourced and monitored at the
    exact same point.
    """
    dom = _compute_domain(params)
    cell = mp.Vector3(dom["cell_x"], dom["cell_y"], 0)
    clad = mp.Medium(index=params["clad_index"])
    fcen = 1.0 / ((params["wl_min_um"] + params["wl_max_um"]) / 2)
    analysis_fwidth = 1.0 / params["wl_min_um"] - 1.0 / params["wl_max_um"]
    source_fwidth = analysis_fwidth * _SOURCE_BANDWIDTH_MARGIN
    mon_size_y = dom["bus_mode_size_y"]
    ports = _ports(params)

    if launch_from == "input":
        source_center = mp.Vector3(-dom["source_offset"], dom["bus_y"])
        direction_sign = +1
    elif launch_from == "add":
        source_center = mp.Vector3(dom["source_offset"], dom["drop_bus_y"])
        direction_sign = -1
    else:
        raise ValueError(f"launch_from must be 'input' or 'add', got {launch_from!r}")

    source = mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=source_fwidth),
        center=source_center,
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

    monitors = {
        name: sim.add_mode_monitor(
            fcen, analysis_fwidth, params["n_freq"],
            mp.ModeRegion(center=pos, size=mp.Vector3(0, mon_size_y)),
        )
        for name, pos in ports.items()
    }

    dft_obj = None
    if field_snapshot_freqs:
        dft_obj = sim.add_dft_fields(
            [mp.Hz], fcen, analysis_fwidth, len(field_snapshot_freqs),
            freq=list(field_snapshot_freqs), center=mp.Vector3(), size=cell,
        )

    return sim, monitors, dft_obj


def _reference_incident(params: dict, launch_from: str, use_native_geometry: bool = False):
    """Bus-only (no ring) reference simulation for ONE bus, launched from
    ONE side -- required because this is a resonant structure (see module
    docstring). `launch_from="input"` references the input bus, launched
    +x; `launch_from="add"` references the add/drop bus, launched -x --
    exactly matching `simulate_baseline`'s two excitations below, each
    normalized against its own bus's own reference."""
    dom = _compute_domain(params)
    cell = mp.Vector3(dom["cell_x"], dom["cell_y"], 0)
    clad = mp.Medium(index=params["clad_index"])
    core = mp.Medium(index=params["core_index"])
    fcen = 1.0 / ((params["wl_min_um"] + params["wl_max_um"]) / 2)
    analysis_fwidth = 1.0 / params["wl_min_um"] - 1.0 / params["wl_max_um"]
    source_fwidth = analysis_fwidth * _SOURCE_BANDWIDTH_MARGIN
    mon_size_y = dom["bus_mode_size_y"]
    ports = _ports(params)

    if launch_from == "input":
        bus_y = dom["bus_y"]
        source_x = -dom["source_offset"]
        mon_x = ports["input"].x
        direction_sign = +1
        fwd_index = 0  # forward (+x) component at the LEFT monitor is "incident"
    elif launch_from == "add":
        bus_y = dom["drop_bus_y"]
        source_x = dom["source_offset"]
        mon_x = ports["add"].x
        direction_sign = -1
        fwd_index = 1  # backward (-x) component at the RIGHT monitor is "incident"
    else:
        raise ValueError(f"launch_from must be 'input' or 'add', got {launch_from!r}")

    if use_native_geometry:
        bus_only_geometry = [mp.Block(
            size=mp.Vector3(mp.inf, params["wg_width_um"], mp.inf),
            center=mp.Vector3(0, bus_y), material=core,
        )]
    else:
        bus_only_geometry = gds_import.gds_component_to_prisms(
            _bus_gf_component(params, bus_y), material=core
        )
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


def _run_one_excitation(params: dict, launch_from: str, field_snapshot_freqs=None,
                         use_native_geometry: bool = False):
    """Excite one port, measure the complex mode coefficient at all 4
    ports. Returns (freqs, coeffs) where coeffs is a dict port_name ->
    complex array, each the OUTGOING-from-the-domain component at that
    port's monitor.

    Index convention: `input`/`drop` sit at the -x edge of the domain
    (x=-port_offset) and `through`/`add` sit at the +x edge
    (x=+port_offset) -- see module docstring. In the with-ring run's
    steady state, ANY signal reaching one of these monitors is, by
    definition, traveling AWAY from the device core (nothing legitimate
    re-enters from the PML) -- so every port's meaningful coefficient is
    the one pointing outward at ITS OWN side: backward/-x (mode index 1)
    for the two -x-side ports, forward/+x (index 0) for the two +x-side
    ports. This holds for BOTH excitations (verified against the expected
    physical routing: launching "input" [+x] drives the ring's
    add/drop-side segment in -x, so `drop` -- not `add` -- is where that
    energy actually exits; launching "add" [-x] drives the ring's
    input-side segment in +x, so `through` -- not `input` -- is where that
    energy exits). This is NOT simply "index 1 at the excited port, index 0
    elsewhere" (racetrack.py's 2-port rule) -- that rule only happens to
    match the `input`-excitation case; for the `add` excitation (launched
    -x) the self-reflection is actually the FORWARD (+x, index 0)
    component, since `add` itself sits on the +x side."""
    sim, monitors, dft_obj = _make_simulation(
        params, launch_from, field_snapshot_freqs=field_snapshot_freqs,
        use_native_geometry=use_native_geometry,
    )
    with _quiet_meep():
        sim.run(until_after_sources=mp.stop_when_dft_decayed(
            tol=params["dft_decay_tol"], minimum_run_time=params["min_sim_time"],
            maximum_run_time=params["max_sim_time"],
        ))
        results = {
            name: sim.get_eigenmode_coefficients(mon, [1], eig_parity=mp.TE)
            for name, mon in monitors.items()
        }
        freqs = np.array(mp.get_flux_freqs(monitors[launch_from]))
        eps = sim.get_array(component=mp.Dielectric) if dft_obj is not None else None
        hz_arrays = (
            [sim.get_dft_array(dft_obj, mp.Hz, i) for i in range(len(field_snapshot_freqs))]
            if dft_obj is not None else None
        )

    coeffs = {}
    for name, res in results.items():
        idx = 1 if name in ("input", "drop") else 0
        coeffs[name] = res.alpha[0, :, idx]

    return freqs, coeffs, eps, hz_arrays


def capture_field_snapshots(params: dict | None, wavelengths_um: list,
                             launch_from: str = "input", use_native_geometry: bool = False) -> list:
    """Run one additional excitation purely to capture Hz field snapshots at
    the given wavelengths (e.g. an on-resonance and an off-resonance
    wavelength) -- same rationale/approach as racetrack.py's
    capture_field_snapshots (DFT capture frequencies must be fixed before
    sim.run(), so this can't be folded into simulate_baseline's own runs,
    whose interesting wavelengths are only known AFTER that spectrum has
    been measured)."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    freqs_wanted = [1.0 / wl for wl in wavelengths_um]
    dom = _compute_domain(params)

    _, _, eps, hz_arrays = _run_one_excitation(
        params, launch_from, field_snapshot_freqs=freqs_wanted, use_native_geometry=use_native_geometry,
    )
    extent = (-dom["cell_x"] / 2, dom["cell_x"] / 2, -dom["cell_y"] / 2, dom["cell_y"] / 2)
    return [
        FieldSnapshot(field=hz, eps=eps, extent_um=extent, component="Hz", wavelength_um=wl)
        for hz, wl in zip(hz_arrays, wavelengths_um)
    ]


def simulate_baseline(params: dict | None = None, use_native_geometry: bool = False) -> AddDropRingResult:
    """Full 4-port characterization from exactly 2 independent excitations
    ("input", "add" -- the two physically-input-side ports), mirroring
    coupler.py's established "excite 2 of 4, verify the rest via
    reciprocity" convention. Each excitation measures all 4 ports at once
    (4 mode monitors per run). Normalized against a SEPARATE, direction- and
    bus-matched bus-only reference run each (see `_reference_incident`'s
    docstring) -- required for a resonant structure, unlike coupler.py's
    self-normalization.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}

    ref_freqs_in, incident_in = _reference_incident(params, launch_from="input", use_native_geometry=use_native_geometry)
    freqs, coeffs_in, _, _ = _run_one_excitation(params, launch_from="input", use_native_geometry=use_native_geometry)
    assert np.allclose(ref_freqs_in, freqs), "reference and with-ring runs must share the same frequency grid"

    ref_freqs_add, incident_add = _reference_incident(params, launch_from="add", use_native_geometry=use_native_geometry)
    _, coeffs_add, _, _ = _run_one_excitation(params, launch_from="add", use_native_geometry=use_native_geometry)
    assert np.allclose(ref_freqs_add, freqs), "reference and with-ring runs must share the same frequency grid"

    permittivity = get_permittivity_map(params, use_native_geometry=use_native_geometry)
    sim_params = {
        **params,
        "meep_version": mp.__version__,
        "python_version": platform.python_version(),
        "creation_date": date.today().isoformat(),
    }

    return AddDropRingResult(
        wavelengths_um=1.0 / freqs,
        freqs=freqs,
        S_input_refl=coeffs_in["input"] / incident_in,
        S_input_through=coeffs_in["through"] / incident_in,
        S_input_drop=coeffs_in["drop"] / incident_in,
        S_input_add=coeffs_in["add"] / incident_in,
        S_add_refl=coeffs_add["add"] / incident_add,
        S_add_drop=coeffs_add["drop"] / incident_add,
        S_add_through=coeffs_add["through"] / incident_add,
        S_add_input=coeffs_add["input"] / incident_add,
        permittivity=permittivity,
        sim_params=sim_params,
    )


# ---------------------------------------------------------------------------
# Resonance-aware energy-conservation check -- 4-port analog of
# racetrack.energy_conservation_check. See that module's extensive
# module-level comment for the full rationale (tight off-resonance, loose
# on-resonance); the only change here is summing 4 power terms per
# excitation instead of 2.
# ---------------------------------------------------------------------------
def _find_resonance_dip_spans(total: np.ndarray, baseline: float, min_dip_depth: float) -> np.ndarray:
    """Identical algorithm to racetrack._find_resonance_dip_spans -- find
    local minima at least `min_dip_depth` below the spectrum's own median,
    then walk outward while the spectrum keeps weakly recovering."""
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


def energy_conservation_check(result: AddDropRingResult, off_resonance_tol: float = 0.02,
                               min_dip_depth: float = 0.1, passivity_tol: float = 0.01) -> dict:
    """For each of the 2 excitations, sum |S|^2 over all 4 measured ports
    (reflection + through + drop + cross-bus leakage). Off-resonance points
    must sit within `off_resonance_tol` of lossless; on-resonance (dip)
    points are only held to a physicality bound (no amplification) -- a
    near-critical-coupled add-drop ring is SUPPOSED to dump most power into
    `ring_conductivity` right at resonance."""
    total_in = (np.abs(result.S_input_refl) ** 2 + np.abs(result.S_input_through) ** 2
                + np.abs(result.S_input_drop) ** 2 + np.abs(result.S_input_add) ** 2)
    total_add = (np.abs(result.S_add_refl) ** 2 + np.abs(result.S_add_drop) ** 2
                 + np.abs(result.S_add_through) ** 2 + np.abs(result.S_add_input) ** 2)

    def _check_one(total):
        baseline = float(np.median(total))
        in_dip = _find_resonance_dip_spans(total, baseline, min_dip_depth)
        off_res = total[~in_dip]
        off_resonance_deviation = float(np.max(np.abs(off_res - 1.0))) if len(off_res) else 0.0
        max_physicality_excess = max(0.0, float(np.max(total)) - (1.0 + passivity_tol))
        return {
            "off_resonance_baseline": baseline,
            "n_off_resonance_points": int((~in_dip).sum()),
            "n_resonance_points": int(in_dip.sum()),
            "off_resonance_deviation": off_resonance_deviation,
            "max_physicality_excess": max_physicality_excess,
            "passed": bool(off_resonance_deviation < off_resonance_tol and max_physicality_excess < 1e-6),
        }

    excite_input = _check_one(total_in)
    excite_add = _check_one(total_add)
    return {
        "excite_input": excite_input,
        "excite_add": excite_add,
        "off_resonance_tol": off_resonance_tol,
        "min_dip_depth": min_dip_depth,
        "passivity_tol": passivity_tol,
        "passed": bool(excite_input["passed"] and excite_add["passed"]),
    }


def reciprocity_check(result: AddDropRingResult, tol: float = 0.35) -> dict:
    """The two couplers are geometrically identical (symmetric design), so
    the SAME physical transfer function measured from two INDEPENDENT
    excitation runs should agree: |input->through| vs |add->drop|,
    |input->drop| vs |add->through|, and |input->input| (reflection) vs
    |add->add|. This genuinely tests the simulation (both quantities come
    from separate FDTD runs), mirroring coupler.py's own reciprocity check
    philosophy for a device where only 2 of 4 ports are ever excited.

    `tol` defaults loose (0.35), matching racetrack.py's own
    RECIPROCITY_TOL: a narrow, high-Q resonance sampled by a finite `n_freq`
    grid makes a POINT-WISE max-difference check between two independently
    (and not identically) converged FDTD runs highly sensitive to exactly
    which grid point happens to land nearest each run's own resonance dip --
    confirmed empirically here too (a coarse `n_freq=11` smoke test showed
    max_diff swinging between 0.02 and 0.12 run-to-run at otherwise-identical
    settings). This is a sampling-density effect, not evidence of a port/
    direction bug -- see racetrack.py's own settings-record entry for the
    identical phenomenon."""
    diff_through_drop = float(np.max(np.abs(
        np.abs(result.S_input_through) - np.abs(result.S_add_drop)
    )))
    diff_drop_through = float(np.max(np.abs(
        np.abs(result.S_input_drop) - np.abs(result.S_add_through)
    )))
    diff_refl = float(np.max(np.abs(
        np.abs(result.S_input_refl) - np.abs(result.S_add_refl)
    )))
    max_diff = max(diff_through_drop, diff_drop_through, diff_refl)
    return {
        "diff_input_through_vs_add_drop": diff_through_drop,
        "diff_input_drop_vs_add_through": diff_drop_through,
        "diff_input_refl_vs_add_refl": diff_refl,
        "max_diff": max_diff,
        "tolerance": tol,
        "passed": max_diff < tol,
    }


def passivity_check(result: AddDropRingResult, tol: float = 0.01) -> dict:
    """No |S_ij| may exceed 1 (within numerical margin) -- a passive device
    can never amplify, regardless of loss."""
    all_s = [
        result.S_input_refl, result.S_input_through, result.S_input_drop, result.S_input_add,
        result.S_add_refl, result.S_add_drop, result.S_add_through, result.S_add_input,
    ]
    max_mag = max(float(np.max(np.abs(s))) for s in all_s)
    return {
        "max_magnitude": max_mag,
        "tolerance": tol,
        "passed": max_mag < 1.0 + tol,
    }


def run_all_checks(result: AddDropRingResult, off_resonance_tol: float = 0.02, min_dip_depth: float = 0.1,
                    reciprocity_tol: float = 0.35, passivity_tol: float = 0.01) -> dict:
    """Composite validator -- flat kwargs (not a nested dict) so this can be
    called the same way at both the baseline and every sweep point, mirroring
    racetrack.py's own `racetrack_validate(result, **check_kwargs)` shape."""
    energy = energy_conservation_check(result, off_resonance_tol=off_resonance_tol,
                                        min_dip_depth=min_dip_depth, passivity_tol=passivity_tol)
    reciprocity = reciprocity_check(result, tol=reciprocity_tol)
    passivity = passivity_check(result, tol=passivity_tol)
    return {
        "energy_conservation": energy,
        "reciprocity": reciprocity,
        "passivity": passivity,
        "passed": bool(energy["passed"] and reciprocity["passed"] and passivity["passed"]),
    }


# ---------------------------------------------------------------------------
# Own local artifact I/O -- pic_toolkit.sparams is hardcoded to a 2x2
# S-matrix, so (per coupler.py/mzm.py's precedent) this module keeps its own
# .npz+.json pair. ALL 8 complex arrays are saved as raw complex128 (phase
# preserved natively, per CLAUDE.md's "persist complex S-parameters, not
# just power" rule).
# ---------------------------------------------------------------------------
_ARRAY_FIELDS = (
    "S_input_refl", "S_input_through", "S_input_drop", "S_input_add",
    "S_add_refl", "S_add_drop", "S_add_through", "S_add_input",
)


def save_artifact(path_stem, result: AddDropRingResult, metadata: dict) -> None:
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)

    arrays = {"wavelengths_um": result.wavelengths_um, "freqs": result.freqs}
    arrays.update({name: getattr(result, name) for name in _ARRAY_FIELDS})
    np.savez(path_stem.parent / (path_stem.name + ".npz"), **arrays)

    full_metadata = {**metadata, "port_names": list(result.port_names)}
    (path_stem.parent / (path_stem.name + ".json")).write_text(
        __import__("json").dumps(full_metadata, indent=2)
    )


def load_artifact(path_stem) -> AddDropRingArtifact:
    import json

    path_stem = Path(path_stem)
    npz_path = path_stem.parent / (path_stem.name + ".npz")
    json_path = path_stem.parent / (path_stem.name + ".json")
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"add-drop-ring artifact not found at {path_stem}(.npz/.json).")

    data = np.load(npz_path)
    metadata = json.loads(json_path.read_text())
    return AddDropRingArtifact(
        wavelengths_um=data["wavelengths_um"],
        freqs=data["freqs"],
        **{name: data[name] for name in _ARRAY_FIELDS},
        port_names=tuple(metadata.get("port_names", ("input", "through", "add", "drop"))),
        metadata=metadata,
    )
