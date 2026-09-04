"""Meep simulation of an MZI delay arm built from 4 chained `radius_um` bends
(Euler by default) plus a variable straight-waveguide insertion -- a second,
independent way to realize a delay arm, alongside `mzi.py`'s own smooth
raised-cosine "bump" (see that module's docstring). Shares `bend.py`'s own
module structure closely: this is, like a 90-degree bend, a 2-port device
whose interior is non-collinear but whose two ports sit on the same axis.

Geometry ("jog"): 4 `gf.components.bend_euler(radius=radius_um, angle=90)`
(or `bend_circular`, via `params["bend_type"]`) chained via `.connect()`,
alternating handedness so the path goes `+x -> +y -> +x -> -y -> +x` -- an
"up-then-down" detour that returns to the same y and heading. Each bend
displaces `(radius_um, +-radius_um)`, so 4 bends span exactly `4*radius_um`
in x and net 0 in y (ports stay collinear -- verified against gdsfactory
directly: `dx=4*radius_um`, `dy=0` exactly, for any insertion length). A
straight-waveguide segment is inserted symmetrically right after bend 1 and
right after bend 3 (both while the path is momentarily heading in `+y`/`-y`),
lengthening the jog without changing its x-extent or breaking port
collinearity. Short straight leads (`lead_len_um`) at each end give the
source/monitor planes clean straight waveguide to sit on (same role as
`bend.py`'s `arm_length_um`) -- they add equally to the actual path and to
the straight-line reference below, so they drop out of `delta_L_um`.

`delta_L_um` (extra centerline length vs. a straight reference spanning the
same two ports, matching `mzi.py`'s own `delta_L_um` convention) is a closed
form: `delta_L_um = 4*one_bend_length - 4*radius_um + 2*extra_straight_um`,
where `one_bend_length` is read directly from the built bend component's own
`.info["length"]` (not hardcoded -- differs between `bend_type="euler"`
(longer, clothoid) and `"circular"` (exact quarter-circle, `pi/2*radius_um`),
see `extra_straight_um_for_delta_L`/`_domain`). At `radius_um=5.0,
bend_type="euler"` (this module's default), `one_bend_length=8.318um`, so
even the bare jog (`extra_straight_um=0`) already provides
`delta_L_um=13.274um` -- just under `07_mzi.ipynb`/the SAX-designed lattice
notebook's `delta_L_um=15.0` target, reached with `extra_straight_um~=0.863`.

Simulation model: 2D effective-index cross-section, same convention as
`bend.py`/`waveguide.py` (TE, `eig_parity=mp.TE` forced explicitly at every
source/monitor per `CLAUDE.md` Sec 5). Only this file (plus `waveguide.py`,
`bend.py`, `bend_topopt.py`, `racetrack.py`, `coupler.py`, `mzi.py`,
`spiral.py`, `spiral_gds.py`, `gds_import.py`) imports meep.
"""

from __future__ import annotations

import contextlib
import io
import platform
from dataclasses import dataclass, field
from datetime import date

import meep as mp
import numpy as np

from ..params import GLOBAL_PARAMS
from . import gds_import


@contextlib.contextmanager
def _quiet_meep():
    """See bend.py's identical helper -- silences Meep's C++-layer stdout/
    stderr chatter, without swallowing real Python exceptions."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
# User-adjustable parameters.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    **GLOBAL_PARAMS,
    "resolution": 40,           # pixels/um -- finer than the toolkit-wide default, same
                                 # justification as bend.py: need to resolve the bend curvature.
    "radius_um": 5.0,           # bend (centerline) radius for all 4 bends.
    "bend_type": "euler",       # "euler" (clothoid, default) or "circular" -- see bend.py's
                                 # own Euler-vs-circular crossover note (docs/simulation_
                                 # settings_record.md); radius_um=5.0 is comfortably above it.
    "extra_straight_um": 0.864, # straight-waveguide length inserted after bend 1 AND after
                                 # bend 3 (symmetric) -- the representative baseline value here
                                 # lands delta_L_um at exactly 15.0, matching the SAX-designed
                                 # lattice-filter notebook's delta_L_um.
    "lead_len_um": 3.0,         # straight lead at each of the 2 outer ports, for clean
                                 # source/monitor placement (same role as bend.py's arm_length_um;
                                 # cancels out of delta_L_um -- see module docstring).
    "n_freq": 21,
    "margin_um": 1.5,
    "port_offset_um": 1.0,      # ports sit this far, along each outer lead, from its bend tangent point
    "source_offset_um": 0.5,    # source planes sit this much further out than the ports
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


@dataclass
class BaselineResult:
    wavelengths_um: np.ndarray
    freqs: np.ndarray
    s_matrix: dict
    port_names: tuple = ("o1", "o2")
    permittivity: PermittivityMap = None
    field_snapshot: FieldSnapshot = None
    sim_params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _one_bend(params: dict):
    """Build the single reusable 90-degree bend (Euler or circular) all 4
    copies in the jog share, plus the cross-section it was built with."""
    import gdsfactory as gf

    radius = params["radius_um"]
    xs = gf.cross_section.strip(width=params["wg_width_um"], radius=radius, radius_min=radius)
    bend_fn = gf.components.bend_circular if params["bend_type"] == "circular" else gf.components.bend_euler
    bend = bend_fn(radius=radius, angle=90, cross_section=xs, allow_min_radius_violation=True)
    return bend, xs


def delta_L_um_from_extra(extra_straight_um: float, params: dict) -> float:
    """delta_L_um for a given extra_straight_um -- see module docstring."""
    bend, _ = _one_bend(params)
    radius = params["radius_um"]
    return 4 * bend.info["length"] - 4 * radius + 2 * extra_straight_um


def extra_straight_um_for_delta_L(delta_L_um: float, params: dict) -> float:
    """Inverse of delta_L_um_from_extra -- the extra_straight_um needed to hit
    a target delta_L_um, for use when specifying a sweep directly in delta_L_um
    terms (the quantity that actually matters). Raises ValueError if
    delta_L_um is below this geometry's floor (the bare 4-bend jog, extra_
    straight_um=0) rather than silently clamping."""
    bend, _ = _one_bend(params)
    radius = params["radius_um"]
    floor = 4 * bend.info["length"] - 4 * radius
    extra = (delta_L_um - floor) / 2.0
    if extra < 0:
        raise ValueError(
            f"delta_L_um={delta_L_um} is below this geometry's floor of {floor:.4f}um "
            f"(radius_um={radius}, bend_type={params['bend_type']!r}) -- increase delta_L_um, "
            "reduce radius_um, or accept a lower floor with a different bend_type."
        )
    return extra


def _build_raw_gf_component(params: dict):
    """The jog + 2 leads, at gdsfactory's own natural placement (o1 near the
    origin, shape extending mostly toward +y) -- NOT yet centered; see
    build_gf_component for the centering step (same 2-stage pattern as
    bend.py's own build_gf_component, for the same documented reason:
    mp.Simulation's geometry_center measurably corrupts S-parameters, see
    docs/simulation_settings_record.md)."""
    import gdsfactory as gf

    bend, xs = _one_bend(params)
    extra = params["extra_straight_um"]
    straight_len = max(extra, 1e-3)  # gf.components.straight requires length > 0
    lead_len = params["lead_len_um"]

    c = gf.Component()
    lead_in = c.add_ref(gf.components.straight(length=lead_len, cross_section=xs))
    b1 = c.add_ref(bend)
    b1.connect("o1", lead_in.ports["o2"])
    s1 = c.add_ref(gf.components.straight(length=straight_len, cross_section=xs))
    s1.connect("o1", b1.ports["o2"])
    b2 = c.add_ref(bend)
    b2.mirror_x()
    b2.connect("o1", s1.ports["o2"])
    b3 = c.add_ref(bend)
    b3.mirror_x()
    b3.connect("o1", b2.ports["o2"])
    s2 = c.add_ref(gf.components.straight(length=straight_len, cross_section=xs))
    s2.connect("o1", b3.ports["o2"])
    b4 = c.add_ref(bend)
    b4.connect("o1", s2.ports["o2"])
    lead_out = c.add_ref(gf.components.straight(length=lead_len, cross_section=xs))
    lead_out.connect("o1", b4.ports["o2"])

    c.add_port("o1", port=lead_in.ports["o1"])
    c.add_port("o2", port=lead_out.ports["o2"])
    return c


def _domain(params: dict) -> dict:
    """Derive the centered gdsfactory component, both ports' positions, and
    the cell size -- shared by build_gf_component, get_permittivity_map, and
    the simulation helpers below, so all three always agree (same role as
    every other pic_toolkit component's own _domain)."""
    raw = _build_raw_gf_component(params)
    dbu = raw.kcl.dbu
    bbox = raw.bbox()
    bbox_cx = (bbox.left + bbox.right) / 2 * dbu
    bbox_cy = (bbox.bottom + bbox.top) / 2 * dbu

    port_map = dict(raw.ports.items()) if hasattr(raw.ports, "items") else {p.name: p for p in raw.ports}
    o1_native, o2_native = port_map["o1"], port_map["o2"]
    o1_port = mp.Vector3(o1_native.center[0] * dbu - bbox_cx, o1_native.center[1] * dbu - bbox_cy)
    o2_port = mp.Vector3(o2_native.center[0] * dbu - bbox_cx, o2_native.center[1] * dbu - bbox_cy)

    dpml, margin = params["dpml_um"], params["margin_um"]
    cell_x = (bbox.right - bbox.left) * dbu + 2 * (dpml + margin)
    cell_y = (bbox.top - bbox.bottom) * dbu + 2 * (dpml + margin)

    return dict(
        raw=raw, dbu=dbu, shift=mp.Vector3(-bbox_cx, -bbox_cy),
        o1_port=o1_port, o2_port=o2_port, cell_x=cell_x, cell_y=cell_y,
        delta_L_um=delta_L_um_from_extra(params["extra_straight_um"], params),
    )


def build_gf_component(params: dict):
    """The centered jog component -- gdsfactory geometry translated so its
    own tight bounding box is centered on the origin (see _domain's
    docstring for why, and bend.py's build_gf_component for the identical
    pattern)."""
    import gdsfactory as gf

    dom = _domain(params)
    centered = gf.Component()
    ref = centered.add_ref(dom["raw"])
    ref.move((dom["shift"].x, dom["shift"].y))
    centered.add_ports(ref.ports)
    return centered


def build_geometry_from_gds(params: dict) -> list:
    core = mp.Medium(index=params["core_index"])
    gf_component = build_gf_component(params)
    return gds_import.gds_component_to_prisms(gf_component, material=core)


def _cell_and_clad(params: dict):
    dom = _domain(params)
    cell = mp.Vector3(dom["cell_x"], dom["cell_y"], 0)
    clad = mp.Medium(index=params["clad_index"])
    return cell, clad


def get_permittivity_map(params: dict) -> PermittivityMap:
    """Permittivity map WITHOUT running any FDTD timestepping -- inspect the
    jog geometry (bend radii, strand/leg clearance, port alignment) before
    paying for an expensive simulation, same discipline as every other
    component."""
    cell, clad = _cell_and_clad(params)
    sim = mp.Simulation(
        cell_size=cell, boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=build_geometry_from_gds(params),
        default_material=clad, resolution=params["resolution"],
    )
    with _quiet_meep():
        sim.init_sim()
    eps = sim.get_epsilon()
    extent = (-cell.x / 2, cell.x / 2, -cell.y / 2, cell.y / 2)
    return PermittivityMap(eps=eps, extent_um=extent)


def _port_planes(params: dict):
    """Reference-plane centers/sizes for port o1 and o2 -- both leads are
    x-aligned, so both monitors are vertical (x-normal) lines, exactly like
    waveguide.py's simpler collinear case."""
    dom = _domain(params)
    mon_size = min(1.5, dom["cell_y"] - 2 * params["dpml_um"] - 0.2)
    o1, o2 = dom["o1_port"], dom["o2_port"]
    port_off = params["port_offset_um"]
    # o1's lead points -x outward (entry is +x); o2's lead points +x outward (entry is -x).
    port1_center = mp.Vector3(o1.x + port_off, o1.y)
    port2_center = mp.Vector3(o2.x - port_off, o2.y)
    size = mp.Vector3(0, mon_size, 0)
    return port1_center, size, port2_center, size


def _make_simulation(params: dict, launch_from: str, capture_dft: bool):
    """Build one Simulation with an eigenmode source just outside port o1 or
    o2, launching INTO the device -- same convention as every other
    component's own _make_simulation."""
    cell, clad = _cell_and_clad(params)
    dom = _domain(params)
    fcen = 1.0 / ((params["wl_min_um"] + params["wl_max_um"]) / 2)
    fwidth = 1.0 / params["wl_min_um"] - 1.0 / params["wl_max_um"]
    port1_center, port1_size, port2_center, port2_size = _port_planes(params)

    source_off = params["source_offset_um"]
    if launch_from == "o1":
        src_center = mp.Vector3(port1_center.x - source_off, port1_center.y)
        direction_sign = 1
    else:
        src_center = mp.Vector3(port2_center.x + source_off, port2_center.y)
        direction_sign = -1

    source = mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
        center=src_center, size=port1_size,
        eig_band=1, eig_parity=mp.TE, eig_match_freq=True,
        eig_kpoint=mp.Vector3(direction_sign, 0, 0),
    )
    sim = mp.Simulation(
        cell_size=cell, boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=build_geometry_from_gds(params), sources=[source],
        default_material=clad, resolution=params["resolution"],
    )
    mon1 = sim.add_mode_monitor(fcen, fwidth, params["n_freq"], mp.ModeRegion(center=port1_center, size=port1_size))
    mon2 = sim.add_mode_monitor(fcen, fwidth, params["n_freq"], mp.ModeRegion(center=port2_center, size=port2_size))

    dft_obj = None
    if capture_dft:
        # Capture both Ez and Hz -- don't assume inheritance of the "TE -> Ez" convention
        # from other components, same discipline as bend.py.
        dft_obj = sim.add_dft_fields([mp.Ez, mp.Hz], fcen, fcen, 1, center=mp.Vector3(), size=cell)
    return sim, mon1, mon2, dft_obj, dom


def _run_one_direction(params: dict, launch_from: str, capture_dft: bool):
    sim, mon1, mon2, dft_obj, dom = _make_simulation(params, launch_from, capture_dft)
    min_sim_time = 2.0 * params["core_index"] * dom["cell_x"]
    with _quiet_meep():
        sim.run(until_after_sources=mp.stop_when_dft_decayed(minimum_run_time=min_sim_time))
        res1 = sim.get_eigenmode_coefficients(mon1, [1], eig_parity=mp.TE)
        res2 = sim.get_eigenmode_coefficients(mon2, [1], eig_parity=mp.TE)
        freqs = np.array(mp.get_flux_freqs(mon1))
        if capture_dft:
            ez = sim.get_dft_array(dft_obj, mp.Ez, 0)
            hz = sim.get_dft_array(dft_obj, mp.Hz, 0)
            eps = sim.get_array(component=mp.Dielectric)

    a1 = res1.alpha[0, :, :]  # [:,0]=+x, [:,1]=-x
    a2 = res2.alpha[0, :, :]

    field_snapshot = None
    if capture_dft:
        cell = mp.Vector3(dom["cell_x"], dom["cell_y"], 0)
        extent = (-cell.x / 2, cell.x / 2, -cell.y / 2, cell.y / 2)
        if np.max(np.abs(ez)) >= np.max(np.abs(hz)):
            field, component = ez, "Ez"
        else:
            field, component = hz, "Hz"
        field_snapshot = FieldSnapshot(field=field, eps=eps, extent_um=extent, component=component)
    return freqs, a1, a2, field_snapshot


def simulate_baseline(params: dict | None = None) -> BaselineResult:
    """Full two-port characterization -- excite o1 then o2 independently, same
    "don't assume S12=S21, measure it" discipline as every other component.

    o1's lead points -x outward (entry is +x): incident=a1[:,0], reflected=a1[:,1].
    o2's lead points +x outward (entry is -x): incident=a2[:,1], reflected=a2[:,0].
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    mp.verbosity(0)

    freqs, a1_fwd, a2_fwd, field_snapshot = _run_one_direction(params, "o1", capture_dft=True)
    incident_1 = a1_fwd[:, 0]
    reflected_1 = a1_fwd[:, 1]
    transmitted_21 = a2_fwd[:, 0]
    s11 = reflected_1 / incident_1
    s21 = transmitted_21 / incident_1

    _, a1_bwd, a2_bwd, _ = _run_one_direction(params, "o2", capture_dft=False)
    incident_2 = a2_bwd[:, 1]
    reflected_2 = a2_bwd[:, 0]
    transmitted_12 = a1_bwd[:, 1]
    s22 = reflected_2 / incident_2
    s12 = transmitted_12 / incident_2

    permittivity = get_permittivity_map(params)
    sim_params = {**params, "meep_version": mp.__version__,
                  "python_version": platform.python_version(), "creation_date": date.today().isoformat(),
                  "delta_L_um": delta_L_um_from_extra(params["extra_straight_um"], params)}
    return BaselineResult(
        wavelengths_um=1.0 / freqs, freqs=freqs,
        s_matrix={"11": s11, "12": s12, "21": s21, "22": s22},
        permittivity=permittivity, field_snapshot=field_snapshot, sim_params=sim_params,
    )
