"""Meep simulation of a straight dielectric strip waveguide.

This is the only file in the toolkit (besides gds_import.py) that imports
meep. Everything downstream (SAX models, circuits) reads cached results
produced here and never imports this module.

Simulation model: 2D effective-index cross-section (x = propagation, y =
transverse). This ignores vertical (z) confinement -- a real ridge waveguide
is approximated by a single dielectric slab of constant index. That's a
deliberate Phase-1 simplification, not an oversight.

TE/TM correction: every eigenmode source/monitor now passes
`eig_parity=mp.TE` explicitly, forcing the genuinely-TE mode (see
docs/simulation_settings_record.md for the investigation that found
`NO_PARITY` was silently simulating TM). Both Ez and Hz DFT fields are
captured for the field snapshot; expect Hz to dominate now that TE is
genuinely forced.

Geometry source: build_gf_component() builds the waveguide as a gdsfactory
Component; build_geometry_from_gds() converts it into Meep mp.Prism geometry
via gds_import.py. This is the default geometry simulate_baseline() uses --
the same gdsfactory Component is what would be exported to GDS for a real
tapeout, so the simulated geometry and the layout geometry are never built
twice. build_geometry() (the original native mp.Block) is kept only as a
Phase-1 reference path, selectable via use_native_geometry=True.
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
    """Meep's C++ layer writes some messages (e.g. "grid volume is not an
    integer number of pixels" cell-rounding warnings) straight to stdout/
    stderr regardless of mp.verbosity() -- silence those around the actual
    init_sim()/run() calls below. Redirecting output doesn't affect Python
    exceptions, so a genuine error still propagates and is still visible.
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield

# ---------------------------------------------------------------------------
# User-adjustable parameters. Everything a notebook user is expected to touch
# lives here, separated from the simulation mechanics below.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    **GLOBAL_PARAMS,
    "n_freq": 21,
    "cell_y_um": 4.0,           # simulation cell width (transverse direction)
    "length_um": 4.0,           # characterized waveguide length (port-to-port);
                                 # swept in notebooks/01_waveguide_baseline.ipynb
                                 # to fit n_eff/loss via the cutback method
    "source_margin_um": 0.5,    # gap between port reference plane and source plane
    "pml_margin_um": 0.5,       # gap between source plane and PML inner edge
}


def derive_lengths(params: dict) -> dict:
    """Derive port_offset_um/source_offset_um/cell_x_um from length_um so
    they never drift out of sync when length_um is swept (port reference
    planes sit at the two ends of the characterized length; source planes
    and the PML are placed a fixed clearance further out). With the Phase-1
    defaults (length_um=4.0, source_margin_um=0.5, pml_margin_um=0.5,
    dpml_um=1.0) this reproduces the module's original fixed constants
    exactly: port_offset_um=2.0, source_offset_um=2.5, cell_x_um=8.0."""
    port_offset_um = params["length_um"] / 2
    source_offset_um = port_offset_um + params["source_margin_um"]
    cell_x_um = 2 * (source_offset_um + params["pml_margin_um"] + params["dpml_um"])
    return {**params, "port_offset_um": port_offset_um,
            "source_offset_um": source_offset_um, "cell_x_um": cell_x_um}


@dataclass
class PermittivityMap:
    eps: np.ndarray          # shape (nx, ny), real permittivity
    extent_um: tuple         # (xmin, xmax, ymin, ymax) for imshow


@dataclass
class FieldSnapshot:
    field: np.ndarray        # complex Ez (or Hz) DFT field at the center wavelength, shape (nx, ny)
    eps: np.ndarray          # permittivity over the same grid, for overlay
    extent_um: tuple
    component: str            # "Ez" or "Hz"


@dataclass
class BaselineResult:
    wavelengths_um: np.ndarray
    freqs: np.ndarray
    s_matrix: dict            # {"11": arr, "12": arr, "21": arr, "22": arr}, complex, one value per freq
    port_names: tuple = ("o1", "o2")
    permittivity: PermittivityMap = None
    field_snapshot: FieldSnapshot = None
    sim_params: dict = field(default_factory=dict)


def build_geometry(params: dict) -> list:
    """Phase-1 native geometry: an infinite (cell-spanning) dielectric strip.
    Kept as the original meep-only reference path -- build_geometry_from_gds
    below is the default simulate_baseline now uses; this native path is
    still exercised (via use_native_geometry=True) purely so the notebook can
    show the two geometry sources agree."""
    core = mp.Medium(index=params["core_index"])
    return [
        mp.Block(
            size=mp.Vector3(mp.inf, params["wg_width_um"], mp.inf),
            center=mp.Vector3(),
            material=core,
        )
    ]


def _straight_centered(length_um: float, wg_width_um: float):
    """A gdsfactory straight waveguide of the given length, centered on the
    origin (gdsfactory's own gf.components.straight spans x=[0, length], not
    x=[-length/2, +length/2])."""
    import gdsfactory as gf

    cross_section = gf.cross_section.strip(width=wg_width_um)
    straight = gf.components.straight(length=length_um, cross_section=cross_section)
    centered = gf.Component()
    ref = centered.add_ref(straight)
    ref.move((-length_um / 2, 0))
    centered.add_ports(ref.ports)  # a Component's own .ports are NOT inherited from
                                    # its references automatically -- without this,
                                    # `centered.ports` would be empty even though the
                                    # underlying geometry has two well-defined ports.
    return centered


def build_gf_component(params: dict):
    """The DUT itself: exactly length_um of straight waveguide -- what
    Sections 2-3 of notebooks/01_waveguide_baseline.ipynb display, and
    what would be exported to GDS for a real tapeout."""
    return _straight_centered(params["length_um"], params["wg_width_um"])


def build_geometry_from_gds(params: dict) -> list:
    """Meep geometry for FDTD -- the default simulation path (see
    simulate_baseline's use_native_geometry flag). Uses the SAME
    cross-section as build_gf_component, but extended out to the cell edges
    (length=cell_x_um) so the EigenModeSource, placed at +/-source_offset_um
    outside the length_um span, still sits on core material rather than in
    cladding -- standard FDTD input/output access-waveguide stubs, not part
    of the exported DUT. Over the characterized span |x| < port_offset_um
    this is pixel-identical to build_gf_component's DUT (only the padding
    outside the ports differs) -- see the notebook's Step 4.1 comparison
    against build_geometry()'s native, genuinely-infinite Phase-1 model,
    which also differs from both only in that same outside-the-ports region.
    A straight waveguide is a single rectangle, so this doesn't hit the
    unmerged-Prism performance trap gds_import.py's docstring warns about
    for more complex layouts (spiral, mzi)."""
    core = mp.Medium(index=params["core_index"])
    sim_component = _straight_centered(params["cell_x_um"], params["wg_width_um"])
    return gds_import.gds_component_to_prisms(sim_component, material=core)


def _cell_and_clad(params: dict):
    cell = mp.Vector3(params["cell_x_um"], params["cell_y_um"], 0)
    clad = mp.Medium(index=params["clad_index"])
    return cell, clad


def get_permittivity_map(params: dict, use_native_geometry: bool = False) -> PermittivityMap:
    """Return the permittivity map WITHOUT running any FDTD timestepping.

    This is intentionally cheap: it lets a user visually inspect the geometry
    before committing to an expensive simulation run. use_native_geometry
    selects build_geometry (Phase-1 mp.Block) vs. the default
    build_geometry_from_gds, so the notebook can render both and compare.
    """
    cell, clad = _cell_and_clad(params)
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
    extent = (
        -cell.x / 2, cell.x / 2,
        -cell.y / 2, cell.y / 2,
    )
    return PermittivityMap(eps=eps, extent_um=extent)


def _make_simulation(params: dict, source_x: float, direction_sign: float, capture_dft: bool,
                      use_native_geometry: bool = False):
    """Build one Simulation with an eigenmode source at source_x launching
    in the direction given by direction_sign (+1 or -1 along x)."""
    cell, clad = _cell_and_clad(params)
    fcen = 1.0 / ((params["wl_min_um"] + params["wl_max_um"]) / 2)
    fwidth = 1.0 / params["wl_min_um"] - 1.0 / params["wl_max_um"]
    mon_size_y = params["cell_y_um"] - 2 * params["dpml_um"]

    source = mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
        center=mp.Vector3(source_x, 0),
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

    freqs_hz = np.linspace(
        1.0 / params["wl_max_um"], 1.0 / params["wl_min_um"], params["n_freq"]
    )
    port1_x = -params["port_offset_um"]
    port2_x = params["port_offset_um"]
    mon1 = sim.add_mode_monitor(
        fcen, fwidth, params["n_freq"],
        mp.ModeRegion(center=mp.Vector3(port1_x, 0), size=mp.Vector3(0, mon_size_y)),
    )
    mon2 = sim.add_mode_monitor(
        fcen, fwidth, params["n_freq"],
        mp.ModeRegion(center=mp.Vector3(port2_x, 0), size=mp.Vector3(0, mon_size_y)),
    )

    dft_obj = None
    if capture_dft:
        # Steady-state complex field at the CENTER wavelength only, for the
        # single field-distribution plot shown in the notebook. Both Ez and
        # Hz are captured -- see module docstring's TE/TM correction note --
        # and _run_one_direction picks whichever actually dominates rather
        # than assuming Ez.
        dft_obj = sim.add_dft_fields(
            [mp.Ez, mp.Hz], fcen, fcen, 1, center=mp.Vector3(), size=cell
        )

    return sim, mon1, mon2, dft_obj


def _run_one_direction(params: dict, launch_from: str, capture_dft: bool,
                        use_native_geometry: bool = False):
    """Excite from port 1 ('o1') or port 2 ('o2') and return complex mode
    coefficients [forward, backward] at both port reference planes."""
    if launch_from == "o1":
        source_x = -params["source_offset_um"]
        direction_sign = +1
    else:
        source_x = params["source_offset_um"]
        direction_sign = -1

    sim, mon1, mon2, dft_obj = _make_simulation(params, source_x, direction_sign, capture_dft,
                                                 use_native_geometry=use_native_geometry)
    # Wrapped broadly, not just around sim.run(): the pixel-rounding warning
    # (see _quiet_meep) also fires from get_eigenmode_coefficients()'s own
    # internal MPB solve, not only from sim.run()/init_sim().
    with _quiet_meep():
        sim.run(until_after_sources=mp.stop_when_dft_decayed())
        res1 = sim.get_eigenmode_coefficients(mon1, [1], eig_parity=mp.TE)
        res2 = sim.get_eigenmode_coefficients(mon2, [1], eig_parity=mp.TE)
        freqs = np.array(mp.get_flux_freqs(mon1))
        if capture_dft:
            ez = sim.get_dft_array(dft_obj, mp.Ez, 0)
            hz = sim.get_dft_array(dft_obj, mp.Hz, 0)
            eps = sim.get_array(component=mp.Dielectric)

    # alpha shape: (bands=1, nfreq, 2) -> [:, :, 0]=+x coefficient, [:, :, 1]=-x coefficient
    a1 = res1.alpha[0, :, :]
    a2 = res2.alpha[0, :, :]

    field_snapshot = None
    if capture_dft:
        cell, _ = _cell_and_clad(params)
        extent = (-cell.x / 2, cell.x / 2, -cell.y / 2, cell.y / 2)
        if np.max(np.abs(ez)) >= np.max(np.abs(hz)):
            field, component = ez, "Ez"
        else:
            field, component = hz, "Hz"
        field_snapshot = FieldSnapshot(field=field, eps=eps, extent_um=extent, component=component)

    return freqs, a1, a2, field_snapshot


def simulate_baseline(params: dict | None = None, use_native_geometry: bool = False) -> BaselineResult:
    """Run the full two-port characterization of the straight waveguide.

    Physically: two independent FDTD runs (excite port 1, excite port 2) are
    needed to measure the FULL 2x2 S-matrix rather than assuming S12=S21 and
    S22=S11 from geometric symmetry -- that assumption is instead VALIDATED
    by the reciprocity check downstream, using genuinely independent data.

    use_native_geometry=False (default): geometry comes from build_gf_component
    via build_geometry_from_gds, so a gdsfactory Component is the single
    source of truth shared by the GDS layout and this simulation.
    use_native_geometry=True: the original Phase-1 mp.Block path, kept so the
    notebook can show both geometry sources agree.
    """
    params = derive_lengths({**DEFAULT_PARAMS, **(params or {})})

    freqs, a1_fwd_run, a2_fwd_run, field_snapshot = _run_one_direction(
        params, launch_from="o1", capture_dft=True, use_native_geometry=use_native_geometry
    )
    # Run A (excite o1, +x): a1[:,0]=incident, a1[:,1]=reflected at port1;
    #                        a2[:,0]=transmitted, a2[:,1]=backscatter at port2 (~0)
    incident_1 = a1_fwd_run[:, 0]
    reflected_1 = a1_fwd_run[:, 1]
    transmitted_21 = a2_fwd_run[:, 0]

    s11 = reflected_1 / incident_1
    s21 = transmitted_21 / incident_1

    _, a1_bwd_run, a2_bwd_run, _ = _run_one_direction(
        params, launch_from="o2", capture_dft=False, use_native_geometry=use_native_geometry
    )
    # Run B (excite o2, -x): a2[:,1]=incident, a2[:,0]=reflected at port2;
    #                        a1[:,1]=transmitted, a1[:,0]=backscatter at port1 (~0)
    incident_2 = a2_bwd_run[:, 1]
    reflected_2 = a2_bwd_run[:, 0]
    transmitted_12 = a1_bwd_run[:, 1]

    s22 = reflected_2 / incident_2
    s12 = transmitted_12 / incident_2

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
        field_snapshot=field_snapshot,
        sim_params=sim_params,
    )
