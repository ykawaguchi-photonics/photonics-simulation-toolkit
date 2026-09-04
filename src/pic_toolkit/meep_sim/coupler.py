"""Meep simulation of a directional coupler: two waveguides adiabatically
S-bent from a wide, uncoupled separation down to a tight edge-to-edge gap for
a straight coupling run, then S-bent back out. This is the MZI's splitter/
combiner building block -- a genuinely 4-port device (2 in / 2 out), unlike
every other component in this toolkit so far (all 2-port).

Geometry and physics are adapted from `PIC_components/MZM/03_mzm_design_v4.ipynb`
Section 5 ("Directional coupler -- building block for the 3 dB splitter/
combiner"), which already validated this S-bend/coupling-run design standalone.
Here it is re-expressed in this toolkit's `params`-dict / `DEFAULT_PARAMS` /
`build_geometry(params)` / `simulate_baseline(params)` convention, matching
`racetrack.py`/`the since-removed ring.py`.

IMPORTANT ARCHITECTURAL NOTE: `pic_toolkit.checks`, `pic_toolkit.sparams`, and
`pic_toolkit.sweep.grid_sweep` are all hardcoded to a 2-port S-matrix (keys
"11"/"12"/"21"/"22" -- confirmed by reading their source). A directional
coupler cannot be represented in that shape. Rather than modify that shared,
widely-used infra for one new 4-port component, this module is deliberately
self-contained: its own result dataclass (`CouplerResult`), its own physical
validation functions (`energy_conservation_check`, `reciprocity_check`,
`passivity_check`, `run_all_checks`), and its own artifact save/load
(`save_artifact`/`load_artifact`), all shaped for (T_through, T_cross,
R_reflect) x (in_top, in_bot) rather than a 2x2 s_matrix. `pic_toolkit.viz`
(`plot_permittivity`/`plot_field`) and `pic_toolkit.sweep.save_manifest` ARE
generic enough to reuse as-is (see notebooks/06_directional_coupler.ipynb).

Only this file (plus `waveguide.py`, `bend.py`, `bend_topopt.py`, `the since-removed ring.py`,
`racetrack.py`) imports meep.

**TE/TM correction:** every `eig_parity=` here used to be `mp.NO_PARITY`, on the
premise (same one found and fixed in `waveguide.py`/`bend.py`/`bend_topopt.py`) that
this device's dominant field is `Ez`. It is not: a direct check confirmed
`eig_parity=mp.TE` gives a genuine guided mode with `Ez` EXACTLY zero and `Hz`
dominant (max|Hz|~62, matching the rigorous 2D result that TE/TM never mix in an
isotropic medium), while `eig_parity=mp.NO_PARITY` was silently selecting the TM mode
(`Ez`-dominant, `Hz`~1e-5) instead -- the earlier "switch field capture to Ez because
that's what's non-empty under NO_PARITY" fix (referenced from `racetrack.py`'s own
docstring) papered over the real issue rather than resolving it: this component was
actually simulating TM, not TE, inconsistent with every other component in the
toolkit. `eig_parity` is now forced to `mp.TE` at the source and all three mode
monitors, and the DFT field capture records whichever of `Ez`/`Hz` is actually
dominant rather than assuming `Ez` -- see `_run_one_excitation`. Because the actual
launched/measured polarization changed, `T_through`/`T_cross`/`R_reflect` and the
tuned `coupling_length_um` (originally found under the wrong polarization) needed
re-measuring from scratch, not just a documentation fix.
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

from . import gds_import

mp.verbosity(0)


@contextlib.contextmanager
def _quiet_meep():
    """Meep's C++ layer writes some messages straight to stdout/stderr
    regardless of mp.verbosity() -- silence those around the actual
    init_sim()/run() calls below. Redirecting output doesn't affect Python
    exceptions, so a genuine error still propagates and is still visible.
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
# User-adjustable parameters.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "coupling_length_um": 13.0,   # length of the straight, tight-gap coupling run -- the
                                  # primary coupling-strength knob. A representative starting
                                  # point expected to land close to 50:50 (at gap_um=0.2,
                                  # wl0_um=1.35) -- 06_directional_coupler.ipynb's Section 10
                                  # sweep + Section 11 interpolation is what actually
                                  # (re)discovers and saves the precise 50:50 value to
                                  # data/design_points/coupler.yaml; see
                                  # docs/simulation_settings_record.md for the TE/TM
                                  # mode-mislabeling bug this value was re-measured after.
    "gap_um": 0.2,                # coupling-region edge-to-edge gap. Matches racetrack.py's
                                  # already-validated choice at this same resolution=25 (5 grid
                                  # points across the gap), rather than the MZM reference
                                  # notebook's tighter 0.15um (which needed resolution=64) --
                                  # deliberately not pushed smaller, same rationale as
                                  # racetrack.py's (and the since-removed ring.py's) small-gap
                                  # resolution/reciprocity wall.
    "wg_width_um": 0.5,           # matches every other pic_toolkit component -- the toolkit
                                  # standardized on 0.5 (see docs/simulation_settings_record.md's
                                  # width-unification note; waveguide.py/bend.py/grating_coupler.py
                                  # inherit this same value from params.GLOBAL_PARAMS).
    "wg_height_um": 0.22,         # vertical thickness of the real 3D device this 2D model
                                  # approximates -- metadata only, same convention as every
                                  # other component.
    "core_index": 2.7,            # matches every other pic_toolkit component; also matches
                                  # PIC_components/MZM's N_EFF_SI=2.70.
    "clad_index": 1.44,           # silicon dioxide, same as every other component (also
                                  # matches PIC_components/MZM's N_CLAD).
    "sbend_len_um": 6.0,          # adiabatic S-bend length connecting the wide (uncoupled)
                                  # input/output spacing to the tight coupling gap. Reused
                                  # directly from PIC_components/MZM's already-working design.
    "wide_sep_um": 2.0,           # separation the S-bends widen out to at each end, so the two
                                  # waveguides don't interact outside coupling_length_um. Same
                                  # value as the MZM reference.
    "lead_len_um": 2.0,           # straight lead length from each port out toward the PML,
                                  # same role as racetrack.py's port/source-offset margins.
    "n_seg": 24,                  # thin Blocks approximating each S-bend's raised-cosine
                                  # profile -- same value as the MZM reference, not re-tuned.
    "mon_size_um": 1.2,           # mode-monitor/source span at every port. At the ports'
                                  # wide_sep_um=2.0 separation, two monitors this size
                                  # (centered at +-1.0) span [0.4,1.6]/[-1.6,-0.4] -- clear of
                                  # each other AND, with margin_um=0.7 below, clear of the PML
                                  # (physical half-extent 1.95um) by ~0.35um on each side.
    "wl_min_um": 1.3,             # O-band, same convention as every other component (NOT the
                                  # MZM reference's single-CWDM-channel 1.261-1.361um band).
    "wl_max_um": 1.4,
    "n_freq": 101,                # matches every other component's default density. A
                                  # directional coupler's spectral response is broad/smooth (no
                                  # narrow resonance dip to miss), so unlike racetrack.py's
                                  # Section 10, this is not expected to need an especially fine
                                  # grid.
    "wl0_um": 1.35,                # single "design" wavelength the 50:50 interpolation is
                                  # solved at -- center of wl_min_um-wl_max_um, matching
                                  # models/racetrack.py's own default readout wavelength.
    "resolution": 25,              # pixels/um -- matches racetrack.py, already confirmed (at
                                  # this exact gap_um=0.2) to give 5 grid points across the gap
                                  # and sub-1% sensitivity to resolution.
    "dpml_um": 1.0,
    "margin_um": 0.7,             # clearance from the wide-separation waveguides/monitors to
                                  # the PML -- larger than racetrack.py's 0.5 because this
                                  # device has TWO parallel waveguides (not one bus) that both
                                  # need monitor clearance from the PML at the port planes; see
                                  # mon_size_um's comment for the numeric margin this buys.
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
class CouplerResult:
    wavelengths_um: np.ndarray
    freqs: np.ndarray
    T_through: np.ndarray       # excite in_top -> |S(out_top<-in_top)|^2
    T_cross: np.ndarray         # excite in_top -> |S(out_bot<-in_top)|^2
    R_reflect: np.ndarray       # excite in_top -> |S(in_top<-in_top)|^2 (backreflection)
    T_through_bot: np.ndarray   # excite in_bot -> |S(out_bot<-in_bot)|^2 (symmetry check)
    T_cross_bot: np.ndarray     # excite in_bot -> |S(out_top<-in_bot)|^2
    R_reflect_bot: np.ndarray   # excite in_bot -> |S(in_bot<-in_bot)|^2
    # Complex counterparts of the 6 power fields above (same excite/measure pairing,
    # just kept as the genuine complex ratio instead of |.|^2) -- see CLAUDE.md Sec 7's
    # "persist complex S-parameters, not just power" note for why these were added.
    # T_* above are exactly np.abs(S_*)**2, so both stay numerically consistent by
    # construction.
    S_through: np.ndarray = None
    S_cross: np.ndarray = None
    S_reflect: np.ndarray = None
    S_through_bot: np.ndarray = None
    S_cross_bot: np.ndarray = None
    S_reflect_bot: np.ndarray = None
    port_names: tuple = ("in_top", "in_bot", "out_top", "out_bot")
    permittivity: PermittivityMap = None
    field_snapshot: FieldSnapshot = None
    sim_params: dict = field(default_factory=dict)


def s_bend_blocks(x0, x1, y0, y1, width, material, n_seg=24):
    """Adiabatic S-bend as n_seg thin axis-aligned Blocks along a raised-cosine
    y(x) profile -- reused verbatim from
    PIC_components/MZM/03_mzm_design_v4.ipynb's coupler geometry, which
    already validated this avoids the strong scattering a sharp-cornered bend
    would cause.
    """
    xs = np.linspace(x0, x1, n_seg + 1)
    blocks = []
    for i in range(n_seg):
        xa, xb = xs[i], xs[i + 1]
        ta, tb = (xa - x0) / (x1 - x0), (xb - x0) / (x1 - x0)
        ya = y0 + (y1 - y0) * 0.5 * (1 - np.cos(np.pi * ta))
        yb = y0 + (y1 - y0) * 0.5 * (1 - np.cos(np.pi * tb))
        blocks.append(mp.Block(
            size=mp.Vector3(xb - xa, width, mp.inf),
            center=mp.Vector3((xa + xb) / 2, (ya + yb) / 2, 0),
            material=material,
        ))
    return blocks


def _compute_domain(params: dict) -> dict:
    """Derive every x-position, the coupled/wide half-separations, and the
    cell size, shared by build_geometry, port/monitor placement, and
    get_permittivity_map, so all three always agree.
    """
    Lc, gap, width = params["coupling_length_um"], params["gap_um"], params["wg_width_um"]
    sbend, wide_sep = params["sbend_len_um"], params["wide_sep_um"]
    lead, dpml, margin = params["lead_len_um"], params["dpml_um"], params["margin_um"]

    device_len = 2 * sbend + Lc
    x_L = -device_len / 2
    x_c0 = x_L + sbend
    x_c1 = x_c0 + Lc
    x_R = x_c1 + sbend

    couple_half = (width + gap) / 2
    wide_half = wide_sep / 2

    cell_x = device_len + 2 * (lead + dpml + margin)
    cell_y = wide_sep + width + 2 * (dpml + margin)

    # Single monitor size for every port (self and both outputs): at the ports'
    # wide_sep separation, two neighboring monitors of this size must not overlap
    # each other, and must also stay clear of the PML -- both checked explicitly
    # here (not just assumed from DEFAULT_PARAMS) since build_geometry/
    # get_permittivity_map/_make_simulation all call this for ANY params dict,
    # including sweep overrides that change wide_sep_um/wg_width_um/margin_um.
    mon_size = params["mon_size_um"]
    phys_half_extent = cell_y / 2 - dpml
    neighbor_clearance = wide_sep - mon_size
    pml_clearance = phys_half_extent - (wide_half + mon_size / 2)
    if neighbor_clearance <= 0 or pml_clearance <= 0:
        raise ValueError(
            f"mon_size_um={mon_size} doesn't fit: neighbor clearance={neighbor_clearance:.3f}um, "
            f"PML clearance={pml_clearance:.3f}um (both must be > 0). Shrink mon_size_um or "
            f"grow wide_sep_um/margin_um."
        )

    return dict(device_len=device_len, x_L=x_L, x_c0=x_c0, x_c1=x_c1, x_R=x_R,
                couple_half=couple_half, wide_half=wide_half,
                cell_x=cell_x, cell_y=cell_y,
                mon_size_self=mon_size, mon_size_out=mon_size)


def build_geometry(params: dict) -> list:
    """Directional coupler geometry only (no leads/PML) -- two waveguides
    adiabatically S-bent from a wide, uncoupled separation (wide_sep_um) down
    to a tight edge-to-edge gap (gap_um) for a straight run of
    coupling_length_um, then S-bent back out. Reused near-verbatim from
    PIC_components/MZM's build_coupler.
    """
    dom = _compute_domain(params)
    core = mp.Medium(index=params["core_index"])
    width, Lc, n_seg = params["wg_width_um"], params["coupling_length_um"], params["n_seg"]
    x_L, x_c0, x_c1, x_R = dom["x_L"], dom["x_c0"], dom["x_c1"], dom["x_R"]
    couple_half, wide_half = dom["couple_half"], dom["wide_half"]

    blocks = []
    blocks += s_bend_blocks(x_L, x_c0, wide_half, couple_half, width, core, n_seg)
    blocks += s_bend_blocks(x_L, x_c0, -wide_half, -couple_half, width, core, n_seg)
    blocks.append(mp.Block(size=mp.Vector3(Lc, width, mp.inf),
                            center=mp.Vector3((x_c0 + x_c1) / 2, couple_half, 0), material=core))
    blocks.append(mp.Block(size=mp.Vector3(Lc, width, mp.inf),
                            center=mp.Vector3((x_c0 + x_c1) / 2, -couple_half, 0), material=core))
    blocks += s_bend_blocks(x_c1, x_R, couple_half, wide_half, width, core, n_seg)
    blocks += s_bend_blocks(x_c1, x_R, -couple_half, -wide_half, width, core, n_seg)
    return blocks


def _s_bend_path_points(x0, x1, y0, y1, n=200):
    """(x, y) samples of the exact same raised-cosine S-bend profile
    `s_bend_blocks()` above already uses, as a continuous curve instead of
    `n_seg` discrete axis-aligned Blocks."""
    xs = np.linspace(x0, x1, n)
    t = (xs - x0) / (x1 - x0)
    ys = y0 + (y1 - y0) * 0.5 * (1 - np.cos(np.pi * t))
    return xs, ys


def _waveguide_path(x_L, x_c0, x_c1, x_R, y_wide, y_couple, n_seg=200):
    """One continuous point array for a single waveguide of the coupler:
    S-bend in (wide->couple), straight coupling run, S-bend out
    (couple->wide) -- built as ONE combined array (not 3 separately-built
    `gf.Path` objects joined with `+`) so the absolute (x, y) coordinates
    match `build_geometry()`'s native construction exactly. `gf.Path`'s own
    `+` operator re-orients the appended segment to match end tangents,
    which does not preserve exact absolute coordinates the way this module's
    native mp.Block placement (at fixed, independently-computed positions)
    requires -- confirmed empirically while developing this function."""
    import gdsfactory as gf

    xs1, ys1 = _s_bend_path_points(x_L, x_c0, y_wide, y_couple, n_seg)
    xs2, ys2 = np.linspace(x_c0, x_c1, 20), np.full(20, y_couple)
    xs3, ys3 = _s_bend_path_points(x_c1, x_R, y_couple, y_wide, n_seg)
    xs = np.concatenate([xs1, xs2[1:], xs3[1:]])
    ys = np.concatenate([ys1, ys2[1:], ys3[1:]])
    return gf.Path(np.column_stack([xs, ys]))


def build_gf_component(params: dict) -> tuple:
    """Returns (top, bottom): two SEPARATE gdsfactory Components, each
    tracing the exact same raised-cosine S-bend + straight-coupling-run
    profile `build_geometry()`'s native mp.Block construction uses (see
    `_waveguide_path`'s docstring) -- not gdsfactory's own built-in
    bend/coupler primitives, which trace a different curve family and would
    require re-tuning `coupling_length_um`'s already-validated 50:50 value.
    Kept as two separate Components (never touch), mirroring `racetrack.py`'s
    bus/ring split."""
    import gdsfactory as gf

    dom = _compute_domain(params)
    cross_section = gf.cross_section.strip(width=params["wg_width_um"])
    x_L, x_c0, x_c1, x_R = dom["x_L"], dom["x_c0"], dom["x_c1"], dom["x_R"]
    couple_half, wide_half = dom["couple_half"], dom["wide_half"]

    top_path = _waveguide_path(x_L, x_c0, x_c1, x_R, wide_half, couple_half)
    bot_path = _waveguide_path(x_L, x_c0, x_c1, x_R, -wide_half, -couple_half)
    top = gf.path.extrude(top_path, cross_section=cross_section)
    bot = gf.path.extrude(bot_path, cross_section=cross_section)
    return top, bot


def build_geometry_from_gds(params: dict) -> list:
    """Meep geometry sourced from `build_gf_component` via `gds_import` --
    reproduces the exact same curve as `build_geometry()` (see
    `_waveguide_path`'s docstring), just via a gdsfactory Component/GDS
    round-trip instead of directly-placed mp.Block objects. NOT the default
    for `simulate_baseline()` -- see that function's `use_native_geometry`
    docstring for why (this module is shared with
    `06_directional_coupler.ipynb`'s already-validated results)."""
    core = mp.Medium(index=params["core_index"])
    top, bot = build_gf_component(params)
    return [
        *gds_import.gds_component_to_prisms(top, material=core),
        *gds_import.gds_component_to_prisms(bot, material=core),
    ]


def _ports(params: dict) -> dict:
    dom = _compute_domain(params)
    wide_half = dom["wide_half"]
    return {
        "in_top": mp.Vector3(dom["x_L"], wide_half),
        "in_bot": mp.Vector3(dom["x_L"], -wide_half),
        "out_top": mp.Vector3(dom["x_R"], wide_half),
        "out_bot": mp.Vector3(dom["x_R"], -wide_half),
    }


def _leads(params: dict) -> list:
    """Straight extension blocks from each port to the cell edge -- ports sit
    at the coupler's own S-bend boundary (x_L/x_R), short of the PML, so
    source/monitor planes need continuous waveguide underneath them.
    """
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


def get_permittivity_map(params: dict, use_native_geometry: bool = True) -> PermittivityMap:
    """Permittivity map WITHOUT running any FDTD timestepping -- inspect the
    S-bend taper, gap, and 4-port layout before paying for an expensive
    simulation, same discipline as every other component.

    `use_native_geometry=True` (the default) uses `build_geometry` (the
    original mp.Block construction, what every validated result in this
    toolkit was measured against). `use_native_geometry=False` uses the
    gdsfactory-derived `build_geometry_from_gds` instead -- pass this
    explicitly for a geometry-display/comparison view (see
    `07_mzi.ipynb`'s Section 3/4); it is NOT the default here because this
    module's geometry is shared with `06_directional_coupler.ipynb`'s
    already-validated `coupling_length_um`, and changing the default
    geometry source is out of scope for that notebook.
    """
    dom = _compute_domain(params)
    cell = mp.Vector3(dom["cell_x"], dom["cell_y"], 0)
    clad = mp.Medium(index=params["clad_index"])
    geometry = build_geometry(params) if use_native_geometry else build_geometry_from_gds(params)
    sim = mp.Simulation(
        cell_size=cell, boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=geometry + _leads(params),
        default_material=clad, resolution=params["resolution"],
    )
    with _quiet_meep():
        sim.init_sim()
    eps = sim.get_epsilon()
    extent = (-cell.x / 2, cell.x / 2, -cell.y / 2, cell.y / 2)
    return PermittivityMap(eps=eps, extent_um=extent)


def _make_simulation(params: dict, launch_from: str, capture_dft: bool):
    dom = _compute_domain(params)
    cell = mp.Vector3(dom["cell_x"], dom["cell_y"], 0)
    clad = mp.Medium(index=params["clad_index"])
    fcen = 1.0 / ((params["wl_min_um"] + params["wl_max_um"]) / 2)
    fwidth = 1.0 / params["wl_min_um"] - 1.0 / params["wl_max_um"]
    ports = _ports(params)
    src_port = ports[launch_from]

    source = mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
        center=mp.Vector3(src_port.x - 0.5, src_port.y),
        size=mp.Vector3(0, dom["mon_size_self"], 0),
        eig_band=1, eig_parity=mp.TE, eig_match_freq=True,
        eig_kpoint=mp.Vector3(1, 0, 0),
    )
    sim = mp.Simulation(
        cell_size=cell, boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=build_geometry(params) + _leads(params), sources=[source],
        default_material=clad, resolution=params["resolution"],
    )
    mon_self = sim.add_mode_monitor(fcen, fwidth, params["n_freq"],
        mp.ModeRegion(center=src_port, size=mp.Vector3(0, dom["mon_size_self"])))
    mon_out_top = sim.add_mode_monitor(fcen, fwidth, params["n_freq"],
        mp.ModeRegion(center=ports["out_top"], size=mp.Vector3(0, dom["mon_size_out"])))
    mon_out_bot = sim.add_mode_monitor(fcen, fwidth, params["n_freq"],
        mp.ModeRegion(center=ports["out_bot"], size=mp.Vector3(0, dom["mon_size_out"])))

    dft_obj = None
    if capture_dft:
        dft_obj = sim.add_dft_fields([mp.Ez, mp.Hz], fcen, fcen, 1, center=mp.Vector3(), size=cell)
    return sim, mon_self, mon_out_top, mon_out_bot, dft_obj


def _run_one_excitation(params: dict, launch_from: str, capture_dft: bool):
    """Excite `launch_from` and measure self (reflection), out_top, out_bot
    mode coefficients. Self-normalized against this SAME run's own excited-
    port monitor (index 0 = forward/incident) -- unlike racetrack.py (or the
    since-removed ring.py), this device is NOT resonant, so there is no recirculating energy to
    contaminate self-normalization and no separate bus-only reference run is
    needed.
    """
    sim, mon_self, mon_out_top, mon_out_bot, dft_obj = _make_simulation(params, launch_from, capture_dft)
    with _quiet_meep():
        sim.run(until_after_sources=mp.stop_when_dft_decayed())
        res_self = sim.get_eigenmode_coefficients(mon_self, [1], eig_parity=mp.TE)
        res_top = sim.get_eigenmode_coefficients(mon_out_top, [1], eig_parity=mp.TE)
        res_bot = sim.get_eigenmode_coefficients(mon_out_bot, [1], eig_parity=mp.TE)
        freqs = np.array(mp.get_flux_freqs(mon_self))
        if capture_dft:
            ez = sim.get_dft_array(dft_obj, mp.Ez, 0)
            hz = sim.get_dft_array(dft_obj, mp.Hz, 0)
            eps = sim.get_array(component=mp.Dielectric)

    a_self, a_top, a_bot = res_self.alpha[0, :, :], res_top.alpha[0, :, :], res_bot.alpha[0, :, :]
    incident = a_self[:, 0]

    field_snapshot = None
    if capture_dft:
        dom = _compute_domain(params)
        extent = (-dom["cell_x"] / 2, dom["cell_x"] / 2, -dom["cell_y"] / 2, dom["cell_y"] / 2)
        if np.max(np.abs(ez)) >= np.max(np.abs(hz)):
            field, component = ez, "Ez"
        else:
            field, component = hz, "Hz"
        field_snapshot = FieldSnapshot(field=field, eps=eps, extent_um=extent, component=component)
    return freqs, incident, a_self[:, 1], a_top[:, 0], a_bot[:, 0], field_snapshot


def simulate_baseline(params: dict | None = None) -> CouplerResult:
    """Full 4-port characterization: excite in_top and in_bot independently.
    The coupler is geometrically mirror-symmetric top<->bottom, so these two
    genuinely independent runs both fully characterize the device AND provide
    the reciprocity/symmetry check `reciprocity_check` below relies on --
    same "excite every port independently, don't assume symmetry" philosophy
    as racetrack.py/bend.py (and the since-removed ring.py), just with 2 of this device's 4 ports
    rather than all 4 (the other 2, out_top/out_bot, are output-only in the
    50:50-splitter use case this component targets).
    """
    params = {**DEFAULT_PARAMS, **(params or {})}

    freqs, inc_t, refl_t, out_top_t, out_bot_t, field_snapshot = _run_one_excitation(params, "in_top", True)
    S_through = out_top_t / inc_t
    S_cross = out_bot_t / inc_t
    S_reflect = refl_t / inc_t
    T_through = np.abs(S_through) ** 2
    T_cross = np.abs(S_cross) ** 2
    R_reflect = np.abs(S_reflect) ** 2

    _, inc_b, refl_b, out_top_b, out_bot_b, _ = _run_one_excitation(params, "in_bot", False)
    S_through_bot = out_bot_b / inc_b
    S_cross_bot = out_top_b / inc_b
    S_reflect_bot = refl_b / inc_b
    T_through_bot = np.abs(S_through_bot) ** 2
    T_cross_bot = np.abs(S_cross_bot) ** 2
    R_reflect_bot = np.abs(S_reflect_bot) ** 2

    permittivity = get_permittivity_map(params)
    sim_params = {**params, "meep_version": mp.__version__,
                  "python_version": platform.python_version(), "creation_date": date.today().isoformat()}
    return CouplerResult(
        wavelengths_um=1.0 / freqs, freqs=freqs,
        T_through=T_through, T_cross=T_cross, R_reflect=R_reflect,
        T_through_bot=T_through_bot, T_cross_bot=T_cross_bot, R_reflect_bot=R_reflect_bot,
        S_through=S_through, S_cross=S_cross, S_reflect=S_reflect,
        S_through_bot=S_through_bot, S_cross_bot=S_cross_bot, S_reflect_bot=S_reflect_bot,
        permittivity=permittivity, field_snapshot=field_snapshot, sim_params=sim_params,
    )


# ---------------------------------------------------------------------------
# Local validation -- deliberately NOT pic_toolkit.checks (2-port hardcoded).
# ---------------------------------------------------------------------------
def energy_conservation_check(result: CouplerResult, tol: float = 0.02, max_expected_loss: float = 0.05) -> dict:
    """T_through + T_cross + R_reflect + expected_loss ~= 1, checked for BOTH
    excitation directions -- 4-port analog of checks.energy_conservation_check's
    |S11|^2+|S21|^2 ~= 1-loss. max_expected_loss plays the same role as that
    function's max_loss_fraction (a small, honest radiative-loss budget at the
    S-bends -- no lossy medium is used here, so this should stay small).
    """
    def _dev(T, Tc, R):
        total = T + Tc + R
        lower = 1.0 - max_expected_loss
        return float(np.max(np.maximum(total - 1.0, lower - total)))

    dev_top = max(0.0, _dev(result.T_through, result.T_cross, result.R_reflect))
    dev_bot = max(0.0, _dev(result.T_through_bot, result.T_cross_bot, result.R_reflect_bot))
    max_dev = max(dev_top, dev_bot)
    return {
        "max_deviation_in_top": dev_top, "max_deviation_in_bot": dev_bot, "max_deviation": max_dev,
        "max_expected_loss": max_expected_loss, "tolerance": tol, "passed": max_dev < tol,
    }


def passivity_check(result: CouplerResult, tol: float = 0.01) -> dict:
    """A passive device can never amplify: every power fraction <= 1, within a
    small numerical margin -- the more fundamental "did the simulation do
    something unphysical" check, same role as checks.passivity_check.
    """
    vals = [result.T_through, result.T_cross, result.R_reflect,
            result.T_through_bot, result.T_cross_bot, result.R_reflect_bot]
    max_mag = max(float(np.max(v)) for v in vals)
    return {"max_power_fraction": max_mag, "tolerance": tol, "passed": max_mag < 1.0 + tol}


def reciprocity_check(result: CouplerResult, tol: float = 0.02) -> dict:
    """This coupler is geometrically mirror-symmetric top<->bottom, so exciting
    in_top and in_bot are physically equivalent, independent measurements --
    T_through(in_top->out_top) should equal T_through(in_bot->out_bot),
    likewise for T_cross and R_reflect. This checks SYMMETRY between two
    independent excitations of a 4-port device (not literal S_ij=S_ji), the
    practically-checkable analog of checks.reciprocity_check given
    simulate_baseline()'s documented 2-excitation scope.
    """
    d_through = float(np.max(np.abs(result.T_through - result.T_through_bot)))
    d_cross = float(np.max(np.abs(result.T_cross - result.T_cross_bot)))
    d_r = float(np.max(np.abs(result.R_reflect - result.R_reflect_bot)))
    max_diff = max(d_through, d_cross, d_r)
    return {
        "max_diff_through": d_through, "max_diff_cross": d_cross, "max_diff_reflect": d_r,
        "max_diff": max_diff, "tolerance": tol, "passed": max_diff < tol,
    }


def run_all_checks(result: CouplerResult, energy_tol: float = 0.03, reciprocity_tol: float = 0.03,
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
# pic_toolkit.sparams but shaped for this device's 4 power-transmission
# curves rather than a 2x2 complex s_matrix.
# ---------------------------------------------------------------------------
@dataclass
class CouplerArtifact:
    wavelengths_um: np.ndarray
    freqs: np.ndarray
    T_through: np.ndarray
    T_cross: np.ndarray
    R_reflect: np.ndarray
    T_through_bot: np.ndarray
    T_cross_bot: np.ndarray
    R_reflect_bot: np.ndarray
    S_through: np.ndarray
    S_cross: np.ndarray
    S_reflect: np.ndarray
    S_through_bot: np.ndarray
    S_cross_bot: np.ndarray
    S_reflect_bot: np.ndarray
    port_names: tuple
    metadata: dict


def save_artifact(path_stem, result: CouplerResult, metadata: dict) -> None:
    """Local, 4-port-shaped analog of pic_toolkit.sparams.save_artifact -- that
    module is hardcoded to a 2-port s_matrix ('11'/'12'/'21'/'22') and can't
    represent this device's (T_through, T_cross, R_reflect) x (in_top, in_bot)
    shape, so this is a parallel, independent implementation, not a wrapper
    around it. Same .npz (arrays) + .json (metadata) sidecar-pair convention.

    Saves both the power fields (T_*, kept for backward compatibility -- exactly
    np.abs(S_*)**2) and the complex S_* fields (magnitude AND phase -- see
    CLAUDE.md Sec 7's "persist complex S-parameters" note for why these exist).
    """
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path_stem.parent / (path_stem.name + ".npz"),
        wavelengths_um=result.wavelengths_um, freqs=result.freqs,
        T_through=result.T_through, T_cross=result.T_cross, R_reflect=result.R_reflect,
        T_through_bot=result.T_through_bot, T_cross_bot=result.T_cross_bot, R_reflect_bot=result.R_reflect_bot,
        S_through=result.S_through, S_cross=result.S_cross, S_reflect=result.S_reflect,
        S_through_bot=result.S_through_bot, S_cross_bot=result.S_cross_bot, S_reflect_bot=result.S_reflect_bot,
    )
    full_metadata = {**metadata, "port_names": list(result.port_names)}
    (path_stem.parent / (path_stem.name + ".json")).write_text(json.dumps(full_metadata, indent=2, default=str))


def load_artifact(path_stem) -> CouplerArtifact:
    path_stem = Path(path_stem)
    npz_path = path_stem.parent / (path_stem.name + ".npz")
    json_path = path_stem.parent / (path_stem.name + ".json")
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"Coupler artifact not found at {path_stem}(.npz/.json). "
            "Run notebooks/06_directional_coupler.ipynb first to generate it."
        )
    data = np.load(npz_path)
    metadata = json.loads(json_path.read_text())
    return CouplerArtifact(
        wavelengths_um=data["wavelengths_um"], freqs=data["freqs"],
        T_through=data["T_through"], T_cross=data["T_cross"], R_reflect=data["R_reflect"],
        T_through_bot=data["T_through_bot"], T_cross_bot=data["T_cross_bot"], R_reflect_bot=data["R_reflect_bot"],
        S_through=data["S_through"], S_cross=data["S_cross"], S_reflect=data["S_reflect"],
        S_through_bot=data["S_through_bot"], S_cross_bot=data["S_cross_bot"], S_reflect_bot=data["S_reflect_bot"],
        port_names=tuple(metadata["port_names"]), metadata=metadata,
    )
