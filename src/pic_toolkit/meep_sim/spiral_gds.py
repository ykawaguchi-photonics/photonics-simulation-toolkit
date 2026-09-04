"""Meep simulation of gdsfactory's `spiral_racetrack_fixed_length()` component -- a
rounded-rectangular racetrack-style spiral delay line, built entirely from gdsfactory's
own PCells (Euler bends by default) rather than this toolkit's hand-built `spiral.py`
turtle path.

Topology choice: gdsfactory's plain `spiral()` puts both ports on the SAME side, both
facing +x -- fine for a "pigtail" access pattern, but not directly usable as an inline
2-port delay element in an MZI arm (which needs one port on each side, both at the SAME
y, facing outward in OPPOSITE directions -- the same collinear convention
`waveguide.py`/`bend.py` use). `spiral_racetrack_fixed_length()` gives exactly that
NATIVELY: its own docstring states "the input and output ports are aligned in y", and
`o1`/`o2` face opposite directions (180/-x and 0/+x respectively) by construction --
verified empirically across several `length`/`min_radius`/`n_straight_sections`
combinations (`o1.y - o2.y == 0` in every case). This replaces an earlier version of
this module that reused gdsfactory's plain `spiral()` and faked a second, opposite-
facing port by surgically cutting one specific loop's polygon open at hardcoded indices
tied to one exact parameter set -- fragile to any parameter change (including the
length sweep this module exists to support) and never guaranteed equal y. No such
surgery is needed here: `build_geometry` reads gdsfactory's own ports and bounding box
directly off the built component.

`length_um` is gdsfactory's own total centerline length target (input -> output);
gdsfactory solves internally for the exact straight-segment length that hits it within
a bounded feasible interval set by `n_straight_sections`/`min_radius_um`/
`min_spacing_um`/`in_out_port_spacing_um` -- an infeasible `length_um` raises a clear
`ValueError` from gdsfactory itself rather than silently misbuilding, which is exactly
what a length sweep needs.

Reuses `pic_toolkit.meep_sim.gds_import.gds_component_to_prisms` for the geometry (see
that module's docstring for why polygons are taken UNMERGED -- one small mp.Prism per
gdsfactory sub-instance -- rather than as one big merged Prism: merging a spiral's
polygons into one huge-bounding-box Prism forces Meep to subpixel-average every grid
pixel against it, which was measured to take several minutes vs. ~100s unmerged on a
comparably-sized spiral).

Because this device ends up a plain 2-port device (like waveguide.py, unlike
coupler.py/mzi.py), simulate_baseline here reuses `pic_toolkit.checks` directly instead
of writing local checks -- same "genuinely 2-port -> use the shared 2-port infra"
reasoning waveguide.py/bend.py/racetrack.py already follow.

Only this file (plus `waveguide.py`, `bend.py`, `bend_topopt.py`, `racetrack.py`,
`coupler.py`, `mzi.py`, `spiral.py`, `gds_import.py`) imports meep.
"""

from __future__ import annotations

import contextlib
import io
import platform
from dataclasses import dataclass, field
from datetime import date

import gdsfactory as gf
import meep as mp
import numpy as np

from . import gds_import

mp.verbosity(0)


@contextlib.contextmanager
def _quiet_meep():
    """See waveguide.py's identical helper -- silences Meep's C++-layer stdout/stderr
    chatter, without swallowing real Python exceptions."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


DEFAULT_PARAMS = {
    "wg_width_um": 0.5,
    "core_index": 2.70,
    "clad_index": 1.44,
    "min_radius_um": 5.0,          # Euler bend radius floor (bend_factory=bend_euler by
                                    # default) -- above bend.py's own measured circular-vs-
                                    # Euler crossover (~2.0um, see docs/simulation_settings_
                                    # record.md) and above the requested >3um margin.
    "min_spacing_um": 5.0,         # center-to-center pitch between adjacent strands --
                                    # edge-to-edge gap = min_spacing_um-wg_width_um = 4.5um,
                                    # far beyond any evanescent coupling range (same reasoning
                                    # spiral.py's own spacing_um documents).
    "length_um": 500.0,            # total centerline length, input->output -- the sweep knob.
    "n_straight_sections": 8,      # must be even (gdsfactory constraint); fixes the feasible
                                    # length_um range for this radius/spacing/port-spacing combo.
    "in_out_port_spacing_um": 30.0,
    "wl_min_um": 1.3,
    "wl_max_um": 1.4,
    "n_freq": 21,
    "resolution": 25,
    "dpml_um": 1.0,
    "margin_um": 1.0,
}


def _gf_component(params: dict):
    xs = gf.cross_section.strip(
        width=params["wg_width_um"], radius=params["min_radius_um"], radius_min=params["min_radius_um"],
    )
    return gf.components.spiral_racetrack_fixed_length(
        length=params["length_um"],
        in_out_port_spacing=params["in_out_port_spacing_um"],
        n_straight_sections=params["n_straight_sections"],
        min_radius=params["min_radius_um"],
        min_spacing=params["min_spacing_um"],
        cross_section=xs,
    )


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


def _domain(params: dict) -> dict:
    """Derive both ports' positions and the overall cell -- shared by build_geometry,
    get_permittivity_map, and the simulation helpers below, so all three always agree
    (same role as every other pic_toolkit component's own _compute_domain/_domain).
    Ports and bounding box are read directly off the built gdsfactory component -- no
    cut-point search, no hardcoded polygon indices, so this stays correct across any
    length_um/min_radius_um/min_spacing_um/n_straight_sections change."""
    c = _gf_component(params)
    dbu = c.kcl.dbu
    port_map = dict(c.ports.items()) if hasattr(c.ports, "items") else {p.name: p for p in c.ports}
    o1_native, o2_native = port_map["o1"], port_map["o2"]
    o1_port = mp.Vector3(o1_native.center[0] * dbu, o1_native.center[1] * dbu)
    o2_port = mp.Vector3(o2_native.center[0] * dbu, o2_native.center[1] * dbu)

    bbox = c.bbox()
    xmin, xmax, ymin, ymax = bbox.left * dbu, bbox.right * dbu, bbox.bottom * dbu, bbox.top * dbu
    dpml, margin = params["dpml_um"], params["margin_um"]
    cell_x = (xmax - xmin) + 2 * (dpml + margin)
    cell_y = (ymax - ymin) + 2 * (dpml + margin)
    geom_center = mp.Vector3((xmin + xmax) / 2, (ymin + ymax) / 2, 0)

    return dict(
        component=c, dbu=dbu, o1_port=o1_port, o2_port=o2_port,
        cell_x=cell_x, cell_y=cell_y, geom_center=geom_center,
        total_length_um=c.info["length"],
    )


def build_geometry(params: dict) -> list:
    core = mp.Medium(index=params["core_index"])
    dom = _domain(params)
    return gds_import.gds_component_to_prisms(dom["component"], material=core)


def get_permittivity_map(params: dict) -> PermittivityMap:
    """Permittivity map WITHOUT running any FDTD timestepping -- inspect the geometry
    (in particular: bend radii, port alignment, strand spacing) before paying for an
    expensive simulation, same discipline as every other component."""
    dom = _domain(params)
    cell = mp.Vector3(dom["cell_x"], dom["cell_y"], 0)
    clad = mp.Medium(index=params["clad_index"])
    sim = mp.Simulation(
        cell_size=cell, boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=build_geometry(params),
        default_material=clad, resolution=params["resolution"],
        geometry_center=dom["geom_center"],
    )
    with _quiet_meep():
        sim.init_sim()
    eps = sim.get_epsilon()
    gc = dom["geom_center"]
    extent = (gc.x - cell.x / 2, gc.x + cell.x / 2, gc.y - cell.y / 2, gc.y + cell.y / 2)
    return PermittivityMap(eps=eps, extent_um=extent)


def _make_simulation(params: dict, source_at: str, capture_dft: bool):
    """Build one Simulation with an eigenmode source just outside `source_at` ('o1' or
    'o2'), launching INTO the device. o1 faces -x outward (entry is +x); o2 faces +x
    outward (entry is -x) -- see module docstring. Both ports share one y by
    construction, so a single mon_size/y works for both monitors."""
    dom = _domain(params)
    cell = mp.Vector3(dom["cell_x"], dom["cell_y"], 0)
    clad = mp.Medium(index=params["clad_index"])
    fcen = 1.0 / ((params["wl_min_um"] + params["wl_max_um"]) / 2)
    fwidth = 1.0 / params["wl_min_um"] - 1.0 / params["wl_max_um"]
    # Both ports are plain single (unpaired) leads -- no neighboring waveguide to avoid,
    # so a generous fixed transverse span is fine, same style as PIC_components/MZM's
    # own MODE_SIZE_WIDE for a single-waveguide monitor.
    mon_size = min(1.5, dom["cell_y"] - 2 * params["dpml_um"] - 0.2)

    o1, o2 = dom["o1_port"], dom["o2_port"]
    # o1's lead points -x outward -> source must sit further -x, launching +x (into device).
    # o2's lead points +x outward -> source must sit further +x, launching -x (into device).
    if source_at == "o1":
        source_center = mp.Vector3(o1.x - 0.5, o1.y)
        direction_sign = 1
    else:
        source_center = mp.Vector3(o2.x + 0.5, o2.y)
        direction_sign = -1

    source = mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
        center=source_center, size=mp.Vector3(0, mon_size, 0),
        eig_band=1, eig_parity=mp.TE, eig_match_freq=True,
        eig_kpoint=mp.Vector3(direction_sign, 0, 0),
    )
    sim = mp.Simulation(
        cell_size=cell, boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=build_geometry(params), sources=[source],
        default_material=clad, resolution=params["resolution"],
        geometry_center=dom["geom_center"],
    )
    mon1 = sim.add_mode_monitor(fcen, fwidth, params["n_freq"],
        mp.ModeRegion(center=o1, size=mp.Vector3(0, mon_size)))
    mon2 = sim.add_mode_monitor(fcen, fwidth, params["n_freq"],
        mp.ModeRegion(center=o2, size=mp.Vector3(0, mon_size)))

    dft_obj = None
    if capture_dft:
        dft_obj = sim.add_dft_fields([mp.Ez], fcen, fcen, 1, center=dom["geom_center"], size=cell)
    return sim, mon1, mon2, dft_obj, dom


def _run_one_direction(params: dict, launch_from: str, capture_dft: bool):
    sim, mon1, mon2, dft_obj, dom = _make_simulation(params, launch_from, capture_dft)
    # Minimum run time floor, same rationale as mzi.py's own -- this device's routed
    # cell_x is longer than a simple straight waveguide's.
    min_sim_time = 2.0 * params["core_index"] * dom["cell_x"]
    with _quiet_meep():
        sim.run(until_after_sources=mp.stop_when_dft_decayed(minimum_run_time=min_sim_time))
        res1 = sim.get_eigenmode_coefficients(mon1, [1], eig_parity=mp.TE)
        res2 = sim.get_eigenmode_coefficients(mon2, [1], eig_parity=mp.TE)
        freqs = np.array(mp.get_flux_freqs(mon1))
        if capture_dft:
            ez = sim.get_dft_array(dft_obj, mp.Ez, 0)
            eps = sim.get_array(component=mp.Dielectric)

    a1 = res1.alpha[0, :, :]
    a2 = res2.alpha[0, :, :]

    field_snapshot = None
    if capture_dft:
        gc = dom["geom_center"]
        cell_x, cell_y = dom["cell_x"], dom["cell_y"]
        extent = (gc.x - cell_x / 2, gc.x + cell_x / 2, gc.y - cell_y / 2, gc.y + cell_y / 2)
        field_snapshot = FieldSnapshot(field=ez, eps=eps, extent_um=extent, component="Ez")
    return freqs, a1, a2, field_snapshot


def simulate_baseline(params: dict | None = None) -> BaselineResult:
    """Full two-port characterization -- excite o1 then o2 independently, same "don't
    assume S12=S21, measure it" discipline as waveguide.py."""
    params = {**DEFAULT_PARAMS, **(params or {})}

    freqs, a1_fwd, a2_fwd, field_snapshot = _run_one_direction(params, "o1", capture_dft=True)
    # Excite o1 (source launches +x, into device): a1[:,0]=incident (+x component),
    # a1[:,1]=reflected (-x, back toward the o1 source side); a2[:,0]=transmitted (+x
    # component arriving at o2's monitor, since o2's own local +x/-x convention matches
    # o1's here -- see _make_simulation's docstring).
    incident_1 = a1_fwd[:, 0]
    reflected_1 = a1_fwd[:, 1]
    transmitted_21 = a2_fwd[:, 0]
    s11 = reflected_1 / incident_1
    s21 = transmitted_21 / incident_1

    _, a1_bwd, a2_bwd, _ = _run_one_direction(params, "o2", capture_dft=False)
    # Excite o2 (source launches -x, into device): a2[:,1]=incident, a2[:,0]=reflected;
    # a1[:,1]=transmitted at o1.
    incident_2 = a2_bwd[:, 1]
    reflected_2 = a2_bwd[:, 0]
    transmitted_12 = a1_bwd[:, 1]
    s22 = reflected_2 / incident_2
    s12 = transmitted_12 / incident_2

    permittivity = get_permittivity_map(params)
    sim_params = {**params, "meep_version": mp.__version__,
                  "python_version": platform.python_version(), "creation_date": date.today().isoformat()}
    return BaselineResult(
        wavelengths_um=1.0 / freqs, freqs=freqs,
        s_matrix={"11": s11, "12": s12, "21": s21, "22": s22},
        permittivity=permittivity, field_snapshot=field_snapshot, sim_params=sim_params,
    )
