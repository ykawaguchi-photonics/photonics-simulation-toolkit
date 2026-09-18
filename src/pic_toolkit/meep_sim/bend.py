"""Meep simulation of a 90-degree bent silicon strip waveguide.

This is the third component in the toolkit, and the first with a genuinely
non-collinear two-port layout: light enters horizontally (traveling in +x)
through port `o1` and exits vertically (traveling in +y) through port `o2`.

Two geometry families share this module, selected purely by `radius_um`:

* `radius_um == 0`  -> a SHARP 90-degree corner, built from two overlapping
  rectangular blocks (the "worst case" bend -- all the mode mismatch and
  radiation happens right at the corner).
* `radius_um  > 0`  -> a ROUNDED bend, built as a single `mp.Prism` whose
  centerline is a quarter circle of the given radius (adiabatic limit as
  radius grows).

Treating `radius_um=0` as just the smallest point on the same sweep (rather
than a separate notebook/module) is deliberate: it lets the whole "how much
radius do I need at a given wavelength" question be answered from ONE
consistent sweep over `simulate_baseline(params)`, exactly as the
since-removed `meep_sim/ring.py` swept `gap_um`.

Simulation model: 2D effective-index cross-section, same convention as
`waveguide.py` and every other component (TE, Hz out-of-plane, no vertical
confinement).

TE/TM correction (both this file and waveguide.py): every eigenmode
source/monitor now passes `eig_parity=mp.TE` explicitly, forcing the
genuinely-TE mode -- see docs/simulation_settings_record.md for the MPB
investigation that found the previous `NO_PARITY` setting was silently
simulating TM instead.

`_make_simulation` captures both `mp.Ez` and `mp.Hz` DFT fields whenever
`capture_dft=True`, and `_run_one_direction` picks whichever actually
dominates for `FieldSnapshot.component` -- expected to be Hz now that TE is
forced (re-verify after re-running; don't assume). This is one of several
files that import meep (see gds_import.py's docstring for the full list).

Geometry source: build_gf_component() builds the bend as a gdsfactory
Component -- either gf.components.bend_circular (a true constant-radius arc,
matching build_geometry's native rounded bend up to ordinary polygon-
discretization noise between the two curve representations) or
gf.components.bend_euler (a clothoid/Euler-spiral bend whose curvature varies
smoothly from zero to 1/radius_um -- a genuinely DIFFERENT shape from a
circular arc of the same nominal radius, not just a re-discretization of it;
see notebooks/02_bent_waveguide.ipynb's Section 3 comparison), selected by
params["bend_type"]. build_geometry_from_gds() converts it to Meep geometry
via gds_import.py, and is the default simulate_baseline() uses. Unlike
waveguide.py, no separate longer "simulation-only" lead is needed here:
arm_length_um already reaches out to _domain()'s x_start/y_end, which the
port/source/cell-sizing logic below is built around, so the same geometry
serves both the exportable DUT and the FDTD run.
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
# User-adjustable parameters.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    **GLOBAL_PARAMS,
    "resolution": 40,        # pixels/um -- finer than the toolkit-wide default (need to
                              # resolve the arc); see docs/simulation_settings_record.md
                              # for the resolution-convergence study behind this value.
    "radius_um": 2.0,        # bend (centerline) radius; 0 = sharp corner (native path only --
                              # gdsfactory's bend_circular/bend_euler both require radius > 0)
    "arm_length_um": 3.0,    # straight arm length from the corner/arc tangent point
    "bend_type": "circular", # "circular" (constant-radius arc) or "euler" (clothoid bend) --
                              # only used by the gdsfactory-sourced geometry path; the native
                              # build_geometry() path always builds a true circular arc
    "n_freq": 21,
    "margin_um": 1.5,        # clearance between structure/ports and the PML
    "port_offset_um": 1.0,   # ports sit this far (along each arm) from the corner/tangent point
    "source_offset_um": 0.5, # source planes sit this much further out than the ports
}


@dataclass
class PermittivityMap:
    eps: np.ndarray          # shape (nx, ny), real permittivity
    extent_um: tuple         # (xmin, xmax, ymin, ymax) for imshow


@dataclass
class FieldSnapshot:
    field: np.ndarray        # complex DFT field (whichever of Ez/Hz dominates -- see
                               # _run_one_direction) at the center wavelength, shape (nx, ny)
    eps: np.ndarray          # permittivity over the same grid, for overlay
    extent_um: tuple
    component: str            # "Ez" or "Hz", whichever was actually dominant


@dataclass
class BaselineResult:
    wavelengths_um: np.ndarray
    freqs: np.ndarray
    s_matrix: dict            # {"11": arr, "12": arr, "21": arr, "22": arr}, complex
    port_names: tuple = ("o1", "o2")
    permittivity: PermittivityMap = None
    field_snapshot: FieldSnapshot = None
    sim_params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _domain(params: dict):
    """Compute cell size and the two arm reference points, shared by every
    function below so the geometry, ports, and monitors all agree on the
    exact same coordinates regardless of radius.

    Layout (fixed for every radius, sharp or rounded): the corner nominally
    sits at the origin. The horizontal arm extends in -x from the corner (or
    from the arc's tangent point, for a rounded bend); the vertical arm
    extends in +y. `arm_length_um` is measured from the tangent point, so the
    two arms have the same physical length in the sharp (radius=0) and
    rounded cases -- only the corner itself changes shape.

    NOTE: an earlier, off-origin tight-bounding-box version of this cell
    (shifting the cell via mp.Simulation's `geometry_center`) measurably
    corrupted the S-parameters and was reverted -- see
    docs/simulation_settings_record.md for the investigation.

    FIX (implemented here, per the note above): instead of `geometry_center`,
    every ABSOLUTE coordinate (structure, ports, sources, monitors) is
    translated by a fixed `shift`, so the structure's tight bounding box
    (x in [x_start, +half_w], y in [-half_w, y_end] in the old, unshifted
    frame -- exact for both the sharp corner and the rounded arc, since the
    arc's x-coordinate is monotonic between its two tangent points) lands
    centered on Meep's own implicit origin. `cell_size` alone is then sized
    to that tight box -- no `geometry_center` argument anywhere. Because the
    L-shape has 90-degree rotational symmetry (equal arm length and width on
    both arms), the tight cell always comes out SQUARE; if a future edit
    makes `cell.x != cell.y`, that's a sign something has diverged. `shift`
    is returned so every other function in this module (which all used to
    independently recompute `arc_center`/`x_start`/`y_end`, a latent
    duplication risk) can derive its own coordinates from this single
    source of truth instead.
    """
    radius = params["radius_um"]
    arm = params["arm_length_um"]
    half_w = params["wg_width_um"] / 2
    pad = params["margin_um"] + params["dpml_um"]

    # shift.y is the (shifted) y-centerline of the horizontal arm; shift.x is
    # the (shifted) x-centerline of the vertical arm -- anything that used to
    # sit at the old hardcoded 0 on either arm's centerline now sits at the
    # corresponding shift component instead.
    shift_mag = (radius + arm - half_w) / 2
    shift = mp.Vector3(shift_mag, -shift_mag)

    # Arc/turn center: for radius=0 this collapses to the (shifted) corner.
    arc_center = mp.Vector3(-radius, radius) + shift

    x_start = arc_center.x - arm  # far (source/PML-facing) end of horizontal arm
    y_end = arc_center.y + arm    # far (source/PML-facing) end of vertical arm

    # Tight, asymmetric cell: half_w+pad of clearance on the two "empty"
    # sides (no longer mirrored out to match the arm+pad reach on the other
    # two sides), full arm+pad on the two arm/PML-facing sides.
    cell_side = radius + arm + half_w + 2 * pad
    cell_side = max(cell_side, 2 * (half_w + pad))
    cell = mp.Vector3(cell_side, cell_side, 0)

    return cell, arc_center, x_start, y_end, shift


def sharp_bend_geometry(params: dict, medium) -> list:
    """L-shaped bend from two overlapping rectangular blocks (radius=0)."""
    arm = params["arm_length_um"]
    width = params["wg_width_um"]
    _, arc_center, x_start, y_end, _ = _domain(params)
    horizontal_arm = mp.Block(
        center=mp.Vector3((x_start + arc_center.x) / 2, arc_center.y),
        size=mp.Vector3(arm, width, mp.inf),
        material=medium,
    )
    vertical_arm = mp.Block(
        center=mp.Vector3(arc_center.x, (arc_center.y + y_end) / 2),
        size=mp.Vector3(width, arm, mp.inf),
        material=medium,
    )
    return [horizontal_arm, vertical_arm]


def rounded_bend_vertices(params: dict, n_arc: int = 80) -> list:
    """Polygon outline of a rounded 90-degree bend (quarter-circle centerline)."""
    radius = params["radius_um"]
    width = params["wg_width_um"]
    half_w = width / 2

    if radius <= half_w:
        raise ValueError("radius_um must be greater than wg_width_um / 2")

    _, arc_center, x_start, y_end, shift = _domain(params)
    arc_center_x, arc_center_y = arc_center.x, arc_center.y

    # theta = -90 deg -> (arc_center_x, shift.y)   [end of horizontal arm]
    # theta =   0 deg -> (shift.x, arc_center_y)   [start of vertical arm]
    # The theta=-90deg endpoint is dropped: it coincides (up to floating-point
    # noise in cos(-pi/2)) with the explicit tangent-point vertices appended
    # around it below, and keeping both created a redundant, near-zero-length
    # polygon edge -- for some radii (e.g. 3.0, 4.0um at this module's default
    # arm_length_um) the duplicate is exact enough that Meep's Prism
    # construction fails outright ("degenerate plane in
    # project_point_into_plane"). Dropping it here is exact for every radius,
    # not just the ones where the float coincidence happens to miss.
    thetas = np.linspace(-np.pi / 2, 0.0, n_arc)[1:]
    outer_r = radius + half_w
    inner_r = radius - half_w

    outer_arc = [
        mp.Vector3(arc_center_x + outer_r * np.cos(t), arc_center_y + outer_r * np.sin(t))
        for t in thetas
    ]
    inner_arc = [
        mp.Vector3(arc_center_x + inner_r * np.cos(t), arc_center_y + inner_r * np.sin(t))
        for t in thetas
    ]

    vertices = []
    vertices.append(mp.Vector3(x_start, shift.y - half_w))
    vertices.append(mp.Vector3(arc_center_x, shift.y - half_w))
    vertices.extend(outer_arc)
    vertices.append(mp.Vector3(shift.x + half_w, y_end))
    vertices.append(mp.Vector3(shift.x - half_w, y_end))
    vertices.extend(list(reversed(inner_arc)))
    vertices.append(mp.Vector3(arc_center_x, shift.y + half_w))
    vertices.append(mp.Vector3(x_start, shift.y + half_w))
    return vertices


def rounded_bend_geometry(params: dict, medium, n_arc: int = 80) -> list:
    vertices = rounded_bend_vertices(params, n_arc=n_arc)
    prism = mp.Prism(
        vertices=vertices,
        height=mp.inf,
        axis=mp.Vector3(0, 0, 1),
        material=medium,
    )
    return [prism]


def build_geometry(params: dict) -> list:
    """Phase-1 native geometry: sharp corner if radius_um == 0, else a
    rounded quarter-circle bend. Kept as the original meep-only reference
    path -- build_geometry_from_gds below is the default simulate_baseline
    now uses; this native path is still exercised (via
    use_native_geometry=True) purely so the notebook can show the two
    geometry sources agree (for the circular case; see build_gf_component's
    docstring for why the euler case is expected to genuinely differ)."""
    core = mp.Medium(index=params["core_index"])
    if params["radius_um"] <= 0:
        return sharp_bend_geometry(params, core)
    return rounded_bend_geometry(params, core)


def build_gf_component(params: dict):
    """Build the bend + two straight arms as a gdsfactory Component. Ports o1
    (horizontal arm, facing -x, light travels +x into the bend) and o2
    (vertical arm, facing +y, light exits +y) come from gf.components.
    bend_circular or bend_euler (params["bend_type"]) at the given radius_um,
    angle=90, connected to two arm_length_um-long gf.components.straight()
    leads via ComponentReference.connect().

    gdsfactory's bend has its horizontal tangent point (o1) at the origin and
    its arc/turn center at (0, radius_um) -- offset by (-radius_um, 0) from
    this module's native (unshifted) convention (arc center at
    (-radius_um, radius_um), see _domain()'s docstring), which itself is then
    translated by _domain()'s own `shift` to land in the tight, asymmetric
    cell. Combining both -- `(-radius_um + shift.x, shift.y)` below -- makes
    gdsfactory's geometry coincide exactly with `_domain()`/`_port_planes()`'s
    (shifted) coordinates, regardless of which geometry source is active.

    allow_min_radius_violation=True bypasses gdsfactory's default PDK
    minimum-bend-radius DRC check -- deliberate here, since this notebook's
    whole point is to characterize loss at radii a real PDK might reject as
    too tight.
    """
    import gdsfactory as gf

    radius = params["radius_um"]
    arm = params["arm_length_um"]
    _, _, _, _, shift = _domain(params)
    cross_section = gf.cross_section.strip(width=params["wg_width_um"])
    bend_fn = gf.components.bend_circular if params["bend_type"] == "circular" else gf.components.bend_euler
    bend_component = bend_fn(radius=radius, angle=90, cross_section=cross_section,
                              allow_min_radius_violation=True)
    lead = gf.components.straight(length=arm, cross_section=cross_section)

    raw = gf.Component()
    bend_ref = raw.add_ref(bend_component)
    lead_in = raw.add_ref(lead)
    lead_in.connect("o2", bend_ref.ports["o1"])
    lead_out = raw.add_ref(lead)
    lead_out.connect("o1", bend_ref.ports["o2"])
    raw.add_port("o1", port=lead_in.ports["o1"])
    raw.add_port("o2", port=lead_out.ports["o2"])

    centered = gf.Component()
    ref = centered.add_ref(raw)
    ref.move((-radius + shift.x, shift.y))
    centered.add_ports(ref.ports)
    return centered


def build_geometry_from_gds(params: dict) -> list:
    """Meep geometry sourced from build_gf_component via gds_import -- the
    default simulation path (see simulate_baseline's use_native_geometry
    flag). Bend + 2 leads are 3 separate, small-bounding-box polygons (via
    gds_component_to_prisms's unmerged-per-reference conversion), so this
    doesn't hit the single-giant-Prism performance trap gds_import.py's
    docstring warns about for more complex layouts (spiral, mzi)."""
    core = mp.Medium(index=params["core_index"])
    gf_component = build_gf_component(params)
    return gds_import.gds_component_to_prisms(gf_component, material=core)


def _cell_and_clad(params: dict):
    cell, *_ = _domain(params)
    clad = mp.Medium(index=params["clad_index"])
    return cell, clad


def get_permittivity_map(params: dict, use_native_geometry: bool = False) -> PermittivityMap:
    """Return the permittivity map WITHOUT running any FDTD timestepping.
    use_native_geometry selects build_geometry (Phase-1 native) vs. the
    default build_geometry_from_gds, so the notebook can render both and
    compare."""
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
    extent = (-cell.x / 2, cell.x / 2, -cell.y / 2, cell.y / 2)
    return PermittivityMap(eps=eps, extent_um=extent)


def _port_planes(params: dict):
    """Reference-plane centers/sizes for port o1 (horizontal arm, x-normal)
    and port o2 (vertical arm, y-normal), plus a monitor cross-section size
    wide enough to catch the mode but short enough to stay off the other arm.
    """
    _, _, x_start, y_end, shift = _domain(params)
    radius = params["radius_um"]
    mon_span = min(
        params["wg_width_um"] + 2 * params["margin_um"],
        2 * radius + params["wg_width_um"],
    ) if radius > 0 else params["wg_width_um"] + 2 * params["margin_um"]
    # Keep it simple and generous but bounded by the arm length so it can't
    # reach into the turn itself.
    mon_span = min(mon_span, 2 * (params["arm_length_um"] - params["port_offset_um"]) * 0.9)
    mon_span = max(mon_span, 3 * params["wg_width_um"])

    port1_x = x_start + params["port_offset_um"]
    port2_y = y_end - params["port_offset_um"]

    port1_center = mp.Vector3(port1_x, shift.y)
    port1_size = mp.Vector3(0, mon_span, 0)
    port2_center = mp.Vector3(shift.x, port2_y)
    port2_size = mp.Vector3(mon_span, 0, 0)
    return port1_center, port1_size, port2_center, port2_size


_REFERENCE_CACHE: dict = {}


def _reference_incident(params: dict, launch_from: str) -> np.ndarray:
    """Measure the TRUE incident mode-coefficient spectrum for a given launch
    direction using a straight-waveguide reference simulation -- same
    resolution, cell, PML, and source/monitor offsets as the actual bend run,
    just with the bend geometry replaced by a straight `mp.Block` along the
    launch axis.

    Kept as a documented dead end -- NOT used by simulate_baseline (see its
    docstring and docs/simulation_settings_record.md for why an independent
    reference made results worse here, unlike racetrack.py's analogous use
    of the same technique). `minimum_run_time=1000` is required for this
    function specifically: an unobstructed straight line decays too fast for
    Meep's default stopping criterion, giving non-physical coefficients if
    stopped early. Results are cached per (launch_from, radius_um,
    resolution, ...) since bend_type doesn't affect a straight reference.
    """
    cache_key = (
        launch_from, params["radius_um"], params["arm_length_um"], params["wg_width_um"],
        params["core_index"], params["clad_index"], params["wl_min_um"], params["wl_max_um"],
        params["n_freq"], params["resolution"], params["dpml_um"], params["margin_um"],
        params["port_offset_um"], params["source_offset_um"],
    )
    if cache_key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[cache_key]

    cell, clad = _cell_and_clad(params)
    fcen = 1.0 / ((params["wl_min_um"] + params["wl_max_um"]) / 2)
    fwidth = 1.0 / params["wl_min_um"] - 1.0 / params["wl_max_um"]
    _, _, x_start, y_end, shift = _domain(params)
    port1_center, port1_size, port2_center, port2_size = _port_planes(params)
    source_off = params["source_offset_um"]
    core = mp.Medium(index=params["core_index"])
    width = params["wg_width_um"]

    # NOTE: this function is currently unused (see docstring above) --
    # updated for consistency with _domain()'s shifted convention, but not
    # re-verified against the new asymmetric domain. Re-test before
    # un-deprecating it.
    if launch_from == "o1":
        src_center = mp.Vector3(x_start + source_off, shift.y)
        src_size = port1_size
        eig_kpoint = mp.Vector3(1, 0, 0)
        mon_center, mon_size = port1_center, port1_size
        straight = mp.Block(size=mp.Vector3(mp.inf, width, mp.inf), center=mp.Vector3(0, shift.y), material=core)
    else:
        src_center = mp.Vector3(shift.x, y_end - source_off)
        src_size = port2_size
        eig_kpoint = mp.Vector3(0, -1, 0)
        mon_center, mon_size = port2_center, port2_size
        straight = mp.Block(size=mp.Vector3(width, mp.inf, mp.inf), center=mp.Vector3(shift.x, 0), material=core)

    source = mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
        center=src_center, size=src_size,
        eig_band=1, eig_parity=mp.TE, eig_match_freq=True,
        eig_kpoint=eig_kpoint,
    )
    sim = mp.Simulation(
        cell_size=cell, boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=[straight], sources=[source], default_material=clad,
        resolution=params["resolution"],
    )
    mon = sim.add_mode_monitor(fcen, fwidth, params["n_freq"], mp.ModeRegion(center=mon_center, size=mon_size))
    with _quiet_meep():
        sim.run(until_after_sources=mp.stop_when_dft_decayed(minimum_run_time=1000))
        res = sim.get_eigenmode_coefficients(mon, [1], eig_parity=mp.TE)
    # Monitor-index convention matches _run_one_direction's: index 0 = the
    # monitor's own +normal direction, index 1 = -normal. o1 launches in +x
    # (the monitor's own +normal), so its incident coefficient is index 0;
    # o2 launches in -y (opposite the monitor's +y normal), so index 1. A
    # straight, object-free line has no reflection to conflate with the
    # other index either way.
    index = 0 if launch_from == "o1" else 1
    incident = res.alpha[0, :, index]
    _REFERENCE_CACHE[cache_key] = incident
    return incident


def _make_simulation(params: dict, launch_from: str, capture_dft: bool,
                      use_native_geometry: bool = False):
    """Build one Simulation with an eigenmode source at port o1 (launching in
    +x) or port o2 (launching in -y), plus mode monitors at both ports."""
    cell, clad = _cell_and_clad(params)
    fcen = 1.0 / ((params["wl_min_um"] + params["wl_max_um"]) / 2)
    fwidth = 1.0 / params["wl_min_um"] - 1.0 / params["wl_max_um"]

    _, _, x_start, y_end, shift = _domain(params)
    port1_center, port1_size, port2_center, port2_size = _port_planes(params)

    source_off = params["source_offset_um"]
    if launch_from == "o1":
        src_center = mp.Vector3(x_start + source_off, shift.y)
        src_size = port1_size
        eig_kpoint = mp.Vector3(1, 0, 0)
    else:
        src_center = mp.Vector3(shift.x, y_end - source_off)
        src_size = port2_size
        eig_kpoint = mp.Vector3(0, -1, 0)

    source = mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
        center=src_center,
        size=src_size,
        eig_band=1,
        eig_parity=mp.TE,
        eig_match_freq=True,
        eig_kpoint=eig_kpoint,
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

    mon1 = sim.add_mode_monitor(
        fcen, fwidth, params["n_freq"],
        mp.ModeRegion(center=port1_center, size=port1_size),
    )
    mon2 = sim.add_mode_monitor(
        fcen, fwidth, params["n_freq"],
        mp.ModeRegion(center=port2_center, size=port2_size),
    )

    dft_obj = None
    if capture_dft:
        # Capture BOTH Ez and Hz: this toolkit's "TE -> visualize Ez"
        # convention (stated in this module's docstring) was copied from
        # waveguide.py without device-specific verification -- racetrack.py's
        # own development history found the OPPOSITE (Hz dominant, |Hz| ~1e7x
        # |Ez|) for its device, and warns every component needs its own
        # check, not an assumed inheritance. _run_one_direction below picks
        # whichever of the two actually dominates.
        dft_obj = sim.add_dft_fields(
            [mp.Ez, mp.Hz], fcen, fcen, 1, center=mp.Vector3(), size=cell
        )

    return sim, mon1, mon2, dft_obj


def _run_one_direction(params: dict, launch_from: str, capture_dft: bool,
                        use_native_geometry: bool = False):
    """Excite from port o1 or o2 and return complex mode coefficients
    [forward, backward] at both port reference planes.

    Mode-monitor convention (Meep): index 0 of the last axis is the
    coefficient for propagation along the monitor's own +normal direction,
    index 1 for -normal. Port o1's monitor normal is +x; port o2's monitor
    normal is +y.
    """
    sim, mon1, mon2, dft_obj = _make_simulation(params, launch_from, capture_dft,
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

    a1 = res1.alpha[0, :, :]  # [:, 0]=+x (forward), [:, 1]=-x (backward)
    a2 = res2.alpha[0, :, :]  # [:, 0]=+y (forward),  [:, 1]=-y (backward)

    field_snapshot = None
    if capture_dft:
        cell, _ = _cell_and_clad(params)
        extent = (-cell.x / 2, cell.x / 2, -cell.y / 2, cell.y / 2)
        # Pick whichever of Ez/Hz actually carries the mode -- see
        # _make_simulation's dft_fields comment for why this isn't assumed.
        if np.max(np.abs(ez)) >= np.max(np.abs(hz)):
            field, component = ez, "Ez"
        else:
            field, component = hz, "Hz"
        field_snapshot = FieldSnapshot(field=field, eps=eps, extent_um=extent, component=component)

    return freqs, a1, a2, field_snapshot


def simulate_baseline(params: dict | None = None, use_native_geometry: bool = False) -> BaselineResult:
    """Run the full two-port characterization of the 90-degree bend.

    Two independent FDTD runs (excite o1, excite o2) measure the full 2x2
    S-matrix directly, exactly as in waveguide.py -- important here because,
    unlike the straight waveguide, a bend has no symmetry that would let
    S12=S21 / S22=S11 be assumed a priori.

    Run A (excite o1, launch +x): a1[:,1]=reflected at o1; a2[:,0]=transmitted (+y) at o2.
    Run B (excite o2, launch -y): a2[:,0]=reflected at o2; a1[:,1]=transmitted (-x) at o1.

    Normalization: self-normalized against this SAME run's own in-situ
    "incident" coefficient at the source-side port -- an independently
    measured reference (_reference_incident) was tried and made results
    worse; see docs/simulation_settings_record.md for the full comparison.

    use_native_geometry=False (default): geometry comes from build_gf_component
    (params["bend_type"]) via build_geometry_from_gds. use_native_geometry=True:
    the original Phase-1 native path (always a true circular arc, or a sharp
    corner at radius_um=0).
    """
    params = {**DEFAULT_PARAMS, **(params or {})}

    mp.verbosity(0)

    freqs, a1_run_a, a2_run_a, field_snapshot = _run_one_direction(
        params, launch_from="o1", capture_dft=True, use_native_geometry=use_native_geometry
    )
    incident_1 = a1_run_a[:, 0]
    reflected_1 = a1_run_a[:, 1]
    transmitted_21 = a2_run_a[:, 0]

    s11 = reflected_1 / incident_1
    s21 = transmitted_21 / incident_1

    _, a1_run_b, a2_run_b, _ = _run_one_direction(
        params, launch_from="o2", capture_dft=False, use_native_geometry=use_native_geometry
    )
    incident_2 = a2_run_b[:, 1]
    reflected_2 = a2_run_b[:, 0]
    transmitted_12 = a1_run_b[:, 1]

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
