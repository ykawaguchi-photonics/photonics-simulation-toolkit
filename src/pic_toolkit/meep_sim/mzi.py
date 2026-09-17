"""Meep simulation of a passive Mach-Zehnder interferometer (MZI): two 50:50
directional couplers (splitter, combiner) joined by two arms of DIFFERENT
optical path length, ΔL. This is the interferometric device
`meep_sim/coupler.py`'s own docstring anticipates -- that module measures
only |S|^2 (T_through/T_cross/R_reflect), which is "insufficient for a real
interferometric MZI, which needs the coupler's +-90deg through/cross phase
relationship." This module closes that gap: it reuses the coupler's already-
validated geometry and 50:50 design point verbatim, and records COMPLEX
mode-coefficient ratios (not just power) so the arm interference is captured
honestly.

Port numbering follows standard S-parameter convention: port 1 = in_top,
port 2 = in_bot, port 3 = out_top, port 4 = out_bot. S31/S41 are the
through/cross response to a port-1 excitation; S32/S42 are the same for a
port-2 excitation -- an independent run, not derived from the first, since
(unlike coupler.py's own symmetric device) this MZI's two arms are
deliberately DIFFERENT and so is NOT top<->bottom mirror-symmetric once
delta_L_um > 0; see reciprocity_check's docstring for what IS still checkable
across the two runs.

Unlike a Mach-Zehnder MODULATOR (see `PIC_components/MZM/03_mzm_design_v4.ipynb`),
which keeps both arms the SAME physical length and modulates via an index
perturbation (Delta n) on one or both arms, this is a fully passive device:
both arms have identical width/material (no index perturbation anywhere) and
the interference is created purely by a geometric path-length difference
Delta L. Since Delta_phi(lambda) = 2*pi*n_eff/lambda * Delta_L, only the
CENTERLINE length difference matters -- not the raw (x,y) footprint -- so
the two arms are built to start and end at IDENTICAL coordinates (the
coupler stages' own ports never move) and the delay arm accumulates its
extra length via a single, large, smoothly-curving raised-cosine bump
spanning the whole arm span. See `solve_delay_arm` for how the bump's
amplitude is chosen, and DEFAULT_PARAMS["delta_L_um"]'s comment for why a
single large bump beats a serpentine of many small ones here.

IMPORTANT ARCHITECTURAL NOTE: same reasoning as `coupler.py` -- this is a
4-port device with no natural 2x2 S-matrix, so `pic_toolkit.checks`,
`pic_toolkit.sparams`, and `pic_toolkit.sweep.grid_sweep` don't apply here
either. This module is self-contained (own result dataclass, own physical
validation, own artifact save/load), matching `coupler.py`'s precedent.
`pic_toolkit.viz` and `pic_toolkit.sweep.save_manifest` ARE generic enough
to reuse as-is (see notebooks/07_mzi.ipynb).

Only this file (plus `waveguide.py`, `bend.py`, `bend_topopt.py`, `the since-removed ring.py`,
`racetrack.py`, `coupler.py`) imports meep.

**TE/TM correction:** this module had the same bug `coupler.py`'s own docstring
documents fixing -- `eig_parity=` at the source and all 4 mode monitors was
`mp.NO_PARITY`, and DFT capture recorded only `Ez`, on the unverified premise that
`Ez` is this device's dominant field. It is not, for the same reason as `coupler.py`:
`eig_parity=mp.NO_PARITY` silently selects the TM mode here too. `eig_parity` is now
forced to `mp.TE` at the source and all four mode monitors (`_make_simulation`,
`_run_one_excitation`), and the DFT field capture records whichever of `Ez`/`Hz` is
actually dominant rather than assuming `Ez`. Unlike `coupler.py`, this module's own
`coupling_length_um` default was already re-synced to match `coupler.py`'s
already-TE-corrected value (13.848 um) -- see
`docs/simulation_settings_record.md`'s `coupler.py`/`mzi.py` section -- so this fix
does not change that constant; it corrects the polarization this module launches and
measures with independently of it, which changes the DFT field snapshot's reported
dominant component and this module's own measured energy/reciprocity residuals.
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

from . import coupler, gds_import

mp.verbosity(0)

# Same source-bandwidth-margin fix as racetrack.py: a Gaussian source's spectral
# power falls off toward the edges of its own (fcen, fwidth) window, so the mode
# decomposition there is noisier (small signal, same discretization/PML residual
# noise) -- this device's near-total interferometric cancellation at delta_L_um=0
# makes it far more sensitive to that edge noise than coupler.py's own single-stage
# device ever was. The SOURCE spectrum is made wider than the ANALYZED range while
# the analyzed (monitor) frequency grid stays at exactly wl_min..wl_max.
_SOURCE_BANDWIDTH_MARGIN = 2.0


@contextlib.contextmanager
def _quiet_meep():
    """See coupler.py's identical helper -- silences Meep's C++-layer stdout/
    stderr chatter around init_sim()/run(), without swallowing real Python
    exceptions.
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
# User-adjustable parameters.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # --- splitter/combiner stage: reused verbatim from coupler.py's own
    # DEFAULT_PARAMS / data/design_points/coupler.yaml (the validated 50:50
    # point from notebooks/06_directional_coupler.ipynb) -- both MZI stages
    # use this exact design, unmodified. Re-synced after coupler.py's own
    # resolution=25->40 re-measurement (see docs/simulation_settings_record.md);
    # 07_mzi.ipynb needs a re-run against this update.
    "coupling_length_um": 13.708639452498595,
    "gap_um": 0.2,
    "wg_width_um": 0.5,
    "wg_height_um": 0.22,
    "core_index": 2.7,
    "clad_index": 1.44,
    "sbend_len_um": 6.0,
    "wide_sep_um": 2.0,       # coupler ports' own separation -- UNCHANGED, since
                              # the MZI's overall in/out port positions are exactly
                              # this stage's in_top/in_bot (left stage) and
                              # out_top/out_bot (right stage) ports.
    "lead_len_um": 2.0,
    "n_seg": 24,              # coupler stages ONLY -- passed straight to
                              # coupler.build_geometry, matches the already-validated
                              # 06_directional_coupler.ipynb design exactly. NOT reused
                              # for the delay arm's own (much larger) bump -- see
                              # bump_n_seg below.
    "mon_size_um": 1.2,
    "margin_um": 0.7,
    "wl_min_um": 1.3,
    "wl_max_um": 1.4,
    "n_freq": 101,
    "resolution": 40,              # was 25 -- a direct resolution=25 vs. 40 comparison at this
                                  # module's own baseline delta_L_um (after the TE/TM eig_parity
                                  # fix) showed a dramatic difference, not a minor refinement:
                                  # energy-conservation deviation 3.79%->0.0%, reciprocity ~17x
                                  # tighter (1.92%->0.11%), and a real passivity violation
                                  # (1.0231, needed a loosened 0.03 tolerance) disappearing
                                  # entirely (0.991, comfortably under 1) -- see
                                  # docs/simulation_settings_record.md for the full comparison.
                                  # resolution=25 is not adequate for this device's delay-arm
                                  # bump geometry.
    "dpml_um": 1.0,

    # --- arm section: new to this module. Both arms extend directly from the
    # coupler stages' own out_top/out_bot ports at wide_sep_um separation --
    # no separate widen/narrow transition is needed (an earlier version added
    # one, purely to create clearance for the delay arm's excursion; that's
    # unnecessary now that the bump bulges AWAY from the centerline rather
    # than toward the reference arm, and removing it also removes 4 S-bend
    # junctions -- 4 fewer places to scatter/reflect).
    "bump_n_seg": 200,             # thin Blocks approximating the delay arm's single
                                   # raised-cosine bump, EACH ROTATED to its own local
                                   # tangent direction (see _build_delay_arm_blocks) so
                                   # every segment's true width stays constant even at
                                   # this bump's much steeper local slopes -- NOT the
                                   # coupler's own axis-aligned n_seg (fine there, since
                                   # its S-bend's slope stays small). 200 keeps
                                   # individual segments well under a wavelength even
                                   # for the largest bump.
    "delay_region_len_um": 50.0,  # straight x-span shared by BOTH arms. Fixed
                                   # regardless of delta_L_um -- only what happens
                                   # INSIDE this span differs between the two arms
                                   # (see build_geometry). Longer than a first attempt
                                   # at 30um: for a fixed delta_L_um, a single bump's
                                   # radius of curvature scales as
                                   # ~delay_region_len_um^1.5 (see delta_L_um's own
                                   # comment below), so a longer span keeps the
                                   # worst-case (delta_L_um=20) bend gentler -- radius
                                   # ~5.5um at 50um vs. ~2.4um at 30um -- at the cost of
                                   # a bigger PIC footprint.
    "delta_L_um": 10.0,           # baseline arm path-length difference (Sections 1/9
                                   # of the notebook); swept 0-20um in Section 10.
                                   # Realized as a SINGLE raised-cosine bump on the
                                   # delay arm (see solve_delay_arm), bulging away
                                   # from the device centerline by an amplitude
                                   # solved to add exactly delta_L_um of centerline
                                   # length over delay_region_len_um -- e.g. ~22.8um
                                   # at delta_L_um=20 with the default
                                   # delay_region_len_um=50. A serpentine of many
                                   # short-period bumps was tried first and rejected:
                                   # bend loss is set by radius of curvature
                                   # (~half_period^2/amplitude for a raised-cosine),
                                   # and packing many periods into a fixed span
                                   # collapses that radius far faster than the
                                   # smaller per-period amplitude helps (a 9-period
                                   # serpentine for this same 20um over 30um measured
                                   # a ~0.3um radius vs. a single bump's ~2.4-5.5um --
                                   # comparable to bend.py's own already-validated
                                   # radius_um=2.0). cell_y (see _compute_domain) grows
                                   # with this amplitude, so the simulation domain is
                                   # NOT identical across the delta_L_um sweep -- unlike
                                   # every other pic_toolkit component's sweep.

    # --- FDTD run-length control.
    "dft_decay_tol": 1e-3,        # same default as racetrack.py.
    "min_sim_time_factor": 2.0,   # minimum_run_time = this * core_index * (cell_x + delta_L_um)
                                   # -- i.e. at least ~2 optical transit times of the
                                   # full device, PLUS the delay arm's own extra physical
                                   # length (delta_L_um), which cell_x alone doesn't
                                   # capture. Needed because this device's cell_x
                                   # is far longer than any other pic_toolkit
                                   # component's: PIC_components/MZM/03_mzm_design_v4's
                                   # coupler+arm+coupler notebook hit a real bug on a
                                   # long domain -- mp.stop_when_fields_decayed's decay
                                   # check can be satisfied by a quiet EARLY transient
                                   # at the monitor point, before the pulse has even
                                   # arrived there, silently leaving the far output
                                   # monitor's DFT at exactly 0+0j. A minimum-run-time
                                   # floor (there: 2*n_eff*cell_x) fixed it -- meep's
                                   # own `minimum_run_time` kwarg to
                                   # stop_when_dft_decayed (already used by
                                   # racetrack.py) provides the same floor natively.
    "max_sim_time_factor": 4.0,   # hard cap = this * min_sim_time, just a backstop
                                   # (this device is non-resonant, so no long ring-down
                                   # like racetrack.py's is expected).
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
class MZIResult:
    wavelengths_um: np.ndarray
    freqs: np.ndarray
    S11: np.ndarray   # excite port1 (in_top) -> reflection back into port1
    S21: np.ndarray   # excite port1 -> leaked out port2 (in_bot), not either output
    S31: np.ndarray   # excite port1 -> out_top ("through")
    S41: np.ndarray   # excite port1 -> out_bot ("cross")
    S22: np.ndarray   # excite port2 (in_bot) -> reflection back into port2
    S12: np.ndarray   # excite port2 -> leaked out port1 (in_top), not either output
    S32: np.ndarray   # excite port2 -> out_top ("cross")
    S42: np.ndarray   # excite port2 -> out_bot ("through")
    port_names: tuple = ("in_top", "in_bot", "out_top", "out_bot")
    permittivity: PermittivityMap = None
    field_snapshot: FieldSnapshot = None
    sim_params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Delay-arm geometry: a SINGLE raised-cosine bump (two mirrored
# coupler.s_bend_blocks calls, up then back down) spanning the whole
# delay_region_len_um span, that returns to its own baseline y at both ends --
# so the arm's overall endpoints never move, only its centerline length
# changes. The bump bulges AWAY from the device centerline (not toward the
# reference arm), so its amplitude never threatens to collide with the
# reference arm -- it only needs cell_y (computed in _compute_domain) to grow
# to fit it.
#
# An earlier version of this module used a SERPENTINE of many short-period
# bumps instead, reasoning that many small-amplitude wiggles would look
# "gentler" than one big one. That reasoning was wrong: bend loss is set by
# RADIUS OF CURVATURE, not amplitude, and radius of curvature for a
# raised-cosine bump of amplitude A over half-period h scales as ~h^2/A --
# cramming many periods into a fixed span makes each period's h tiny, and the
# radius collapses far faster than the smaller per-period amplitude helps
# (worked example: 9 periods over ~30um to reach delta_L_um=10 gave a ~0.3um
# radius, whereas one single bump over the same 30um span gives ~3um --
# roughly the toolkit's own already-validated bend.py radius_um=2.0). A single
# long bump is therefore the lower-loss choice even though its own amplitude
# is much larger in absolute terms.
# ---------------------------------------------------------------------------
def _raised_cosine_arc_length(amplitude_um: float, bump_len_um: float, n_sample: int = 200) -> float:
    """Arc length of one full up-and-back-down raised-cosine bump of the
    given lateral amplitude and total x-span, built from two mirrored
    raised-cosine halves (exactly what two back-to-back coupler.s_bend_blocks
    calls trace out). Pure numpy trapezoidal integration -- no scipy, matching
    this toolkit's existing convention (see checks.py's module docstring).
    """
    if amplitude_um == 0:
        return bump_len_um
    h = bump_len_um / 2.0
    t = np.linspace(0.0, h, n_sample)
    dydt = amplitude_um * np.pi / (2.0 * h) * np.sin(np.pi * t / h)
    half_arc = np.trapz(np.sqrt(1.0 + dydt ** 2), t)
    return float(2.0 * half_arc)


def solve_delay_arm(delta_L_um: float, delay_region_len_um: float, max_search_amplitude_um: float = 500.0) -> float:
    """Solve, by bisection, the single bump amplitude whose raised-cosine arc
    length exceeds delay_region_len_um by exactly delta_L_um. Arc length grows
    monotonically with amplitude, so bisection within
    [0, max_search_amplitude_um] is well-posed; max_search_amplitude_um is
    just a generous search ceiling; the toolkit's biggest CURRENT default
    (delta_L_um=20, delay_region_len_um=30) only needs ~15.6um.

    delta_L_um <= 0 returns 0.0 -- the trivial case, a plain straight arm.
    """
    if delta_L_um <= 0:
        return 0.0

    target_total = delay_region_len_um + delta_L_um
    lo, hi = 0.0, max_search_amplitude_um
    if _raised_cosine_arc_length(hi, delay_region_len_um) < target_total:
        raise ValueError(
            f"Could not realize delta_L_um={delta_L_um} over delay_region_len_um={delay_region_len_um} "
            f"within max_search_amplitude_um={max_search_amplitude_um}. Increase delay_region_len_um "
            "or max_search_amplitude_um."
        )
    for _ in range(60):
        mid = (lo + hi) / 2.0
        total = _raised_cosine_arc_length(mid, delay_region_len_um)
        if total < target_total:
            lo = mid
        else:
            hi = mid
    return hi


def _build_delay_arm_blocks(x_start, x_end, baseline_y, amplitude_um, width, material, n_seg):
    """The delay (long) arm: a single raised-cosine bump from baseline_y,
    bulging AWAY from the device centerline (y=0) by amplitude_um, and back to
    baseline_y -- so it can never cross into the reference arm on the other
    side of the centerline regardless of amplitude.

    Built as n_seg small Blocks, each ROTATED to align with the centerline's
    OWN local tangent direction at its midpoint (via mp.Block's e1/e2 basis
    vectors) -- NOT coupler.s_bend_blocks' AXIS-ALIGNED stacked Blocks. Axis-
    aligned stacking only holds the y-EXTENT of each segment constant, not the
    true width perpendicular to the local propagation direction; that's an
    excellent approximation for coupler.py's own S-bends (max slope dy/dx
    ~0.17, under 2% width error) but breaks down for this bump, whose slope
    approaches order 1 near its midpoint at large delta_L_um: confirmed the
    effective width there would be width*cos(atan(dy/dx)), less than half the
    intended width_um at this module's largest bumps -- a real, physically
    wrong waveguide, not merely a rendering artifact. Rotating each segment
    fixes that directly, at the cost of small (sub-pixel-scale, same order as
    every other piecewise-taper approximation in this toolkit) notches at
    each segment-to-segment joint where consecutive orientations differ
    slightly.

    A first version of this fix used a single big mp.Prism (an offset curve:
    centerline samples pushed +-width/2 along their local normal, joined into
    one polygon) instead of many small rotated Blocks. That was geometrically
    correct but far too slow: a single Prism whose bounding box spans nearly
    the WHOLE simulation cell forces Meep's subpixel-averaging to test EVERY
    grid pixel against it (no cheap bounding-box rejection), confirmed to
    take minutes just for get_permittivity_map (no timestepping at all) at
    this module's default resolution -- versus seconds for many small
    Blocks, each with its own tiny local bounding box that lets Meep quickly
    skip pixels far from it, exactly like coupler.s_bend_blocks already
    relies on.

    amplitude_um=0 (zero slope everywhere) degenerates exactly to the same
    plain straight Block the reference arm uses.
    """
    if amplitude_um == 0:
        length = x_end - x_start
        return [mp.Block(size=mp.Vector3(length, width, mp.inf),
                          center=mp.Vector3((x_start + x_end) / 2, baseline_y, 0), material=material)]

    bump_len = x_end - x_start
    away_from_center = -amplitude_um if baseline_y <= 0 else amplitude_um

    xs = np.linspace(x_start, x_end, n_seg + 1)
    u = (xs - x_start) / bump_len
    ys = baseline_y + away_from_center * 0.5 * (1 - np.cos(2 * np.pi * u))

    blocks = []
    for i in range(n_seg):
        dx, dy = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
        seg_len = float(np.hypot(dx, dy))
        tangent = mp.Vector3(dx, dy, 0) / seg_len
        normal = mp.Vector3(-tangent.y, tangent.x, 0)
        blocks.append(mp.Block(
            size=mp.Vector3(seg_len, width, mp.inf),
            center=mp.Vector3((xs[i] + xs[i + 1]) / 2, (ys[i] + ys[i + 1]) / 2, 0),
            e1=tangent, e2=normal, e3=mp.Vector3(0, 0, 1),
            material=material,
        ))
    return blocks


def _translate_block(block, dx):
    """coupler.build_geometry(params) always returns geometry centered at
    x=0; translate its mp.Block list by dx to place a splitter/combiner
    stage at this MZI's own stage centers. Assumes every object
    coupler.build_geometry returns is an mp.Block (true as of this writing --
    see that module's build_geometry)."""
    c = block.center
    return mp.Block(size=block.size, center=mp.Vector3(c.x + dx, c.y, c.z), material=block.material)


def _compute_domain(params: dict) -> dict:
    """Derive every x-position, the two coupler stages' centers, the arm
    span, the solved delay-arm bump amplitude, and the cell size -- shared by
    build_geometry, port/monitor placement, and get_permittivity_map, so all
    three always agree (same role as coupler.py's own _compute_domain).

    Unlike every other pic_toolkit component, cell_y depends on delta_L_um:
    the delay arm's single bump bulges away from the device centerline (see
    _build_delay_arm_blocks), so the cell must grow to keep the bump clear of
    the PML as delta_L_um (and therefore the bump's amplitude) grows. That
    growth is asymmetric -- only the bump side needs it, the reference arm's
    side never does -- so the cell is NOT centered on y=0 the way every other
    pic_toolkit component's is: `y_center` shifts it to tightly wrap the
    actual geometry (reference arm's top edge to the bump's peak, plus
    dpml_um+margin_um on each side) rather than mirroring the bump's extent
    onto the empty top half too. `y_center` must be passed as every
    mp.Simulation's `geometry_center` (see _make_simulation/get_permittivity_map)
    -- the geometry itself is NOT translated, only the cell/PML placement
    around it.
    """
    coupler_device_len = 2 * params["sbend_len_um"] + params["coupling_length_um"]
    arm_span = params["delay_region_len_um"]
    total_device_len = 2 * coupler_device_len + arm_span

    x_c1_center = -total_device_len / 2 + coupler_device_len / 2
    x_c2_center = total_device_len / 2 - coupler_device_len / 2
    x_arm_L = -arm_span / 2
    x_arm_R = arm_span / 2

    wide_half = params["wide_sep_um"] / 2
    width = params["wg_width_um"]

    bend_amplitude_um = solve_delay_arm(params["delta_L_um"], params["delay_region_len_um"])

    lead, dpml, margin = params["lead_len_um"], params["dpml_um"], params["margin_um"]
    cell_x = total_device_len + 2 * (lead + dpml + margin)

    y_top = wide_half + width / 2       # reference (top) arm's outer edge
    y_bottom = -(wide_half + bend_amplitude_um) - width / 2  # delay arm's bump peak outer edge
    cell_y = (y_top - y_bottom) + 2 * (dpml + margin)
    y_center = (y_top + y_bottom) / 2

    mon_size = params["mon_size_um"]
    # Clearance checks stay relative to y_center (the ports sit at wide_half
    # above/below it) -- unaffected by the bump's extent below.
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
        wide_half=wide_half, bend_amplitude_um=bend_amplitude_um,
        cell_x=cell_x, cell_y=cell_y, y_center=y_center, mon_size=mon_size,
    )


def build_geometry(params: dict) -> list:
    """Two coupler stages (splitter, combiner), each coupler.build_geometry's
    own validated 50:50 shape translated into place, joined by two arms
    extending directly from the coupler ports' own wide_sep_um separation: a
    plain straight reference arm (top) and a single raised-cosine bump delay
    arm (bottom, bulging away from the centerline) carrying the extra
    delta_L_um. Both arms share identical start/end coordinates regardless of
    delta_L_um -- only the path taken between them differs.
    """
    dom = _compute_domain(params)
    core = mp.Medium(index=params["core_index"])
    width = params["wg_width_um"]

    stage_blocks = coupler.build_geometry(params)
    left = [_translate_block(b, dom["x_c1_center"]) for b in stage_blocks]
    right = [_translate_block(b, dom["x_c2_center"]) for b in stage_blocks]

    wide_half = dom["wide_half"]
    x_arm_L, x_arm_R = dom["x_arm_L"], dom["x_arm_R"]

    arm_blocks = [
        # reference (top) arm: straight
        mp.Block(size=mp.Vector3(x_arm_R - x_arm_L, width, mp.inf),
                  center=mp.Vector3(0, wide_half, 0), material=core),
    ]
    # delay (bottom) arm: single bump carrying delta_L_um extra length
    arm_blocks += _build_delay_arm_blocks(x_arm_L, x_arm_R, -wide_half, dom["bend_amplitude_um"],
                                           width, core, params["bump_n_seg"])

    return left + right + arm_blocks


def _delay_arm_path(x_start, x_end, baseline_y, amplitude_um, n_seg=400):
    """Point array for the delay arm: the exact same single raised-cosine
    bump `_build_delay_arm_blocks()` traces via many small rotated Blocks
    above, as one continuous curve instead. `amplitude_um=0` degenerates to
    a plain straight line, matching that function's own degenerate case."""
    import gdsfactory as gf

    if amplitude_um == 0:
        return gf.Path(np.array([[x_start, baseline_y], [x_end, baseline_y]]))
    bump_len = x_end - x_start
    away_from_center = -amplitude_um if baseline_y <= 0 else amplitude_um
    xs = np.linspace(x_start, x_end, n_seg)
    u = (xs - x_start) / bump_len
    ys = baseline_y + away_from_center * 0.5 * (1 - np.cos(2 * np.pi * u))
    return gf.Path(np.column_stack([xs, ys]))


def build_gf_component(params: dict):
    """One combined gdsfactory Component: both coupler stages (reusing
    `coupler.build_gf_component`'s own S-bend/coupling-run profile,
    translated into place) plus the two arms -- a straight reference arm and
    a raised-cosine-bump delay arm (`_delay_arm_path`) -- tracing the exact
    same curves `build_geometry()`'s native mp.Block construction uses."""
    import gdsfactory as gf

    dom = _compute_domain(params)
    cross_section = gf.cross_section.strip(width=params["wg_width_um"])

    top, bot = coupler.build_gf_component(params)  # one stage design, referenced twice below
    combined = gf.Component()
    for stage_x in (dom["x_c1_center"], dom["x_c2_center"]):
        for stage_comp in (top, bot):
            ref = combined.add_ref(stage_comp)
            ref.move((stage_x, 0))

    wide_half = dom["wide_half"]
    x_arm_L, x_arm_R = dom["x_arm_L"], dom["x_arm_R"]

    ref_path = gf.Path(np.array([[x_arm_L, wide_half], [x_arm_R, wide_half]]))
    combined.add_ref(gf.path.extrude(ref_path, cross_section=cross_section))

    delay_path = _delay_arm_path(x_arm_L, x_arm_R, -wide_half, dom["bend_amplitude_um"])
    combined.add_ref(gf.path.extrude(delay_path, cross_section=cross_section))

    return combined


def build_geometry_from_gds(params: dict) -> list:
    """Meep geometry sourced from `build_gf_component` via `gds_import` --
    reproduces the exact same curves as `build_geometry()` (both coupler
    stages' S-bends and the delay arm's raised-cosine bump), just via a
    gdsfactory Component/GDS round-trip instead of directly-placed mp.Block
    objects. NOT the default for `simulate_baseline()` -- see
    `get_permittivity_map`'s `use_native_geometry` docstring for why."""
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
    as coupler.py's _leads."""
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
    """Permittivity map WITHOUT running any FDTD timestepping -- inspect both
    coupler stages and the reference-vs-bump arm asymmetry before paying for
    an expensive simulation, same discipline as every other component.

    `use_native_geometry=True` (the default) uses `build_geometry` -- the
    original mp.Block construction every validated result in this notebook
    was measured against. `use_native_geometry=False` uses the
    gdsfactory-derived `build_geometry_from_gds` instead, for a
    geometry-display/comparison view (see notebooks/07_mzi.ipynb's
    Section 3/4) -- not the default, since `simulate_baseline()` itself is
    intentionally left on the native path (see `coupler.get_permittivity_
    map`'s docstring for the same reasoning, shared by this module).
    """
    dom = _compute_domain(params)
    cell = mp.Vector3(dom["cell_x"], dom["cell_y"], 0)
    geom_center = mp.Vector3(0, dom["y_center"], 0)
    clad = mp.Medium(index=params["clad_index"])
    geometry = build_geometry(params) if use_native_geometry else build_geometry_from_gds(params)
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


def _make_simulation(params: dict, launch_from: str, capture_dft: bool):
    dom = _compute_domain(params)
    cell = mp.Vector3(dom["cell_x"], dom["cell_y"], 0)
    geom_center = mp.Vector3(0, dom["y_center"], 0)
    clad = mp.Medium(index=params["clad_index"])
    fcen = 1.0 / ((params["wl_min_um"] + params["wl_max_um"]) / 2)
    analysis_fwidth = 1.0 / params["wl_min_um"] - 1.0 / params["wl_max_um"]
    source_fwidth = analysis_fwidth * _SOURCE_BANDWIDTH_MARGIN
    ports = _ports(params)
    src_port = ports[launch_from]
    other_in = "in_bot" if launch_from == "in_top" else "in_top"

    source = mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=source_fwidth),
        center=mp.Vector3(src_port.x - 0.5, src_port.y),
        size=mp.Vector3(0, dom["mon_size"], 0),
        eig_band=1, eig_parity=mp.TE, eig_match_freq=True,
        eig_kpoint=mp.Vector3(1, 0, 0),
    )
    sim = mp.Simulation(
        cell_size=cell, boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=build_geometry(params) + _leads(params), sources=[source],
        default_material=clad, resolution=params["resolution"],
        geometry_center=geom_center,
    )
    mon_self = sim.add_mode_monitor(fcen, analysis_fwidth, params["n_freq"],
        mp.ModeRegion(center=src_port, size=mp.Vector3(0, dom["mon_size"])))
    # Monitors the OTHER (non-excited) input port's backward-traveling component --
    # i.e. power that couples/reflects all the way back out the "wrong" input side
    # rather than through either output. coupler.py's own device never needed this
    # (validated to have negligible input-to-input leakage at wide_sep_um separation),
    # but this MZI's two coupler stages in series couple measurably into it --
    # omitting this channel was confirmed, empirically, to leave a large,
    # spurious-looking "missing energy" gap in energy_conservation_check.
    mon_in_other = sim.add_mode_monitor(fcen, analysis_fwidth, params["n_freq"],
        mp.ModeRegion(center=ports[other_in], size=mp.Vector3(0, dom["mon_size"])))
    mon_out_top = sim.add_mode_monitor(fcen, analysis_fwidth, params["n_freq"],
        mp.ModeRegion(center=ports["out_top"], size=mp.Vector3(0, dom["mon_size"])))
    mon_out_bot = sim.add_mode_monitor(fcen, analysis_fwidth, params["n_freq"],
        mp.ModeRegion(center=ports["out_bot"], size=mp.Vector3(0, dom["mon_size"])))

    dft_obj = None
    if capture_dft:
        dft_obj = sim.add_dft_fields([mp.Ez, mp.Hz], fcen, fcen, 1, center=geom_center, size=cell)
    return sim, mon_self, mon_in_other, mon_out_top, mon_out_bot, dft_obj, dom


def _run_one_excitation(params: dict, launch_from: str, capture_dft: bool):
    """Excite `launch_from` and measure self (reflection), out_top, out_bot
    mode coefficients -- self-normalized against this same run's own excited-
    port monitor, same as coupler.py (this device is not resonant, so there
    is no recirculating energy to contaminate self-normalization).

    Unlike every other pic_toolkit component, the minimum FDTD run time is
    computed here (not a fixed DEFAULT_PARAMS constant), since it must scale
    with THIS device's cell_x -- see DEFAULT_PARAMS["min_sim_time_factor"]'s
    docstring for the documented failure mode this guards against. It also
    adds delta_L_um itself, on the reasoning that the delay arm's bump is
    PHYSICALLY LONGER than cell_x alone would suggest. This did NOT, on its
    own, fully close an observed gap: at large delta_L_um (confirmed at
    delta_L_um=20, resolution=25) energy_conservation_check and
    reciprocity_check (S21 vs S12, which must match regardless of delta_L_um
    by genuine Lorentz reciprocity) both still show a real, if modest
    (~5-6%), residual -- present with or without this term. Kept anyway since
    it's independently well-motivated (the estimate IS more accurate with
    it), but the residual gap itself is still an open item -- see
    DEFAULT_PARAMS["delta_L_um"]'s comment and the Section 8 discussion in
    notebooks/07_mzi.ipynb for what's been ruled out (bend-loss geometry,
    reciprocity-check logic) and what's still worth checking (this module's
    own reduced testing used a coarse n_freq=7; the notebook's real
    n_freq=101 averages over far more points, and Section 7/8's full spectrum
    plot is the right place to judge whether this is broadband loss or a
    handful of noisy frequency points).
    """
    sim, mon_self, mon_in_other, mon_out_top, mon_out_bot, dft_obj, dom = _make_simulation(
        params, launch_from, capture_dft)
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
    # The other input port's own "backward" (index 1) direction is the correct sign
    # for power leaking OUT of the device there -- both input ports share the same
    # +x forward / -x backward convention (see _ports/_make_simulation: both in_top
    # and in_bot launch in the +x direction), so leaked power arriving at the other
    # input port is, like this run's own reflection, travelling in -x.
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


def simulate_baseline(params: dict | None = None) -> MZIResult:
    """Full 4-port characterization: excite in_top (port 1) and in_bot
    (port 2) independently, recording COMPLEX mode-coefficient ratios (not
    power) -- this is the fix for the gap coupler.py's own docstring flags.
    S31/S41 (from the port-1 run) directly give this notebook's Section 2
    |S31|^2, |S41|^2, and phase results. The port-2 run is not just a
    convenience baseline: it is independent data used by reciprocity_check
    below to test genuine Lorentz reciprocity (S21 vs S12) -- see that
    function's docstring for why this device's asymmetric arms rule out the
    top<->bottom mirror-symmetry checks coupler.py could otherwise do.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}

    freqs, inc1, refl1, cross1, top1, bot1, field_snapshot = _run_one_excitation(params, "in_top", True)
    S11 = refl1 / inc1
    S21 = cross1 / inc1  # port1 -> leaked out port2 (in_bot), instead of either output
    S31 = top1 / inc1
    S41 = bot1 / inc1

    _, inc2, refl2, cross2, top2, bot2, _ = _run_one_excitation(params, "in_bot", False)
    S22 = refl2 / inc2
    S12 = cross2 / inc2  # port2 -> leaked out port1 (in_top)
    S32 = top2 / inc2
    S42 = bot2 / inc2

    permittivity = get_permittivity_map(params)
    sim_params = {**params, "meep_version": mp.__version__,
                  "python_version": platform.python_version(), "creation_date": date.today().isoformat()}
    return MZIResult(
        wavelengths_um=1.0 / freqs, freqs=freqs,
        S11=S11, S21=S21, S31=S31, S41=S41, S22=S22, S12=S12, S32=S32, S42=S42,
        permittivity=permittivity, field_snapshot=field_snapshot, sim_params=sim_params,
    )


# ---------------------------------------------------------------------------
# Arm dispersion -- n_eff(wavelength) via a cheap MPB eigenmode solve (no FDTD
# timestepping), used by the notebook to cross-check the group index n_g
# extracted from the delta_L sweep's own measured FSR.
# ---------------------------------------------------------------------------
def compute_arm_dispersion(params: dict, wavelengths_um) -> dict:
    """Solve for the arm waveguide's fundamental TE mode at each wavelength in
    `wavelengths_um`, using the SAME cross-section (wg_width_um, core_index,
    clad_index) and the SAME resolution/dpml_um this module's own FDTD runs
    use, so the result is directly comparable to them.

    Returns n_eff(wavelength) -- the phase index, phi = 2*pi*n_eff*L/wavelength
    -- and, for free from the same MPB solve, the group velocity MPB itself
    reports for that mode -> n_g_direct = 1/group_velocity. n_g_direct is an
    INDEPENDENT way to get the group index, alongside the finite-difference
    n_g = n_eff - wavelength*d(n_eff)/d(wavelength) a caller can compute from
    the returned n_eff array -- the two should agree.
    """
    core = mp.Medium(index=params["core_index"])
    clad = mp.Medium(index=params["clad_index"])
    cell_y = 6.0  # transverse extent with room for the PML and the mode's evanescent tail
    sim = mp.Simulation(
        cell_size=mp.Vector3(0, cell_y, 0),
        resolution=params["resolution"],
        geometry=[mp.Block(size=mp.Vector3(mp.inf, params["wg_width_um"], mp.inf), material=core)],
        default_material=clad,
        boundary_layers=[mp.PML(params["dpml_um"], direction=mp.Y)],
    )
    with _quiet_meep():
        sim.init_sim()

    vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, cell_y, 0))
    wavelengths_um = np.asarray(wavelengths_um, dtype=float)
    n_eff = np.empty_like(wavelengths_um)
    n_g_direct = np.empty_like(wavelengths_um)
    for i, wl in enumerate(wavelengths_um):
        freq = 1.0 / wl
        with _quiet_meep():
            mode = sim.get_eigenmode(freq, mp.X, vol, 1, mp.Vector3(3, 0, 0),
                                      match_frequency=True, parity=mp.TE)
        n_eff[i] = mode.k.x / freq
        n_g_direct[i] = 1.0 / mode.group_velocity
    return dict(wavelengths_um=wavelengths_um, n_eff=n_eff, n_g_direct=n_g_direct)


# ---------------------------------------------------------------------------
# Local validation -- deliberately NOT pic_toolkit.checks (2-port hardcoded),
# same rationale as coupler.py. Complex-valued analogs of coupler.py's own
# energy_conservation_check/passivity_check/reciprocity_check.
# ---------------------------------------------------------------------------
def energy_conservation_check(result: MZIResult, tol: float = 0.02, max_expected_loss: float = 0.05) -> dict:
    """|S11|^2+|S21|^2+|S31|^2+|S41|^2 ~= 1 (likewise for the port-2 excitation),
    within max_expected_loss -- 4-port analog of checks.energy_conservation_check.

    Includes S21/S12 (power leaked out the OTHER input port, not either output)
    alongside self-reflection and the two through/cross outputs -- coupler.py's
    single-stage device never needed this fourth channel (validated to have
    negligible input-to-input leakage), but this MZI's two coupler stages in
    series (plus the delay arm's own bump) couple measurably into it; omitting
    it understates the device's real energy budget.
    """
    def _dev(*arrs):
        total = sum(np.abs(a) ** 2 for a in arrs)
        lower = 1.0 - max_expected_loss
        return float(np.max(np.maximum(total - 1.0, lower - total)))

    dev1 = max(0.0, _dev(result.S11, result.S21, result.S31, result.S41))
    dev2 = max(0.0, _dev(result.S22, result.S12, result.S32, result.S42))
    max_dev = max(dev1, dev2)
    return {
        "max_deviation_port1": dev1, "max_deviation_port2": dev2, "max_deviation": max_dev,
        "max_expected_loss": max_expected_loss, "tolerance": tol, "passed": max_dev < tol,
    }


def passivity_check(result: MZIResult, tol: float = 0.01) -> dict:
    """A passive device can never amplify: every power fraction <= 1, within
    a small numerical margin -- same role as coupler.py's passivity_check."""
    vals = [result.S11, result.S21, result.S31, result.S41, result.S22, result.S12, result.S32, result.S42]
    max_mag = max(float(np.max(np.abs(v))) for v in vals)
    return {"max_power_fraction": max_mag ** 2, "tolerance": tol, "passed": max_mag < 1.0 + tol}


def reciprocity_check(result: MZIResult, tol: float = 0.02) -> dict:
    """Genuine Lorentz reciprocity for a linear passive device requires
    S_ij = S_ji for every port pair (i, j) -- but with only two independent
    excitations (port1, port2), the only pair where BOTH directions were
    actually measured is (1, 2) itself: S21 (excite port1, measure port2) vs.
    S12 (excite port2, measure port1). This is checked as complex numbers
    (magnitude AND phase), a strictly stronger test than coupler.py could do
    with power-only data.

    An earlier version of this check ALSO compared S31 to S42, S41 to S32,
    and S11 to S22 -- reasoning (correctly, for coupler.py's own symmetric
    device) that top<->bottom mirror symmetry should make those equal. That
    reasoning does NOT carry over here: this MZI's two arms are deliberately
    DIFFERENT (one carries delta_L_um, the other doesn't), so the device is
    NOT mirror-symmetric top<->bottom once delta_L_um > 0, and comparing
    those pairs was testing a symmetry this device was never meant to have,
    not an actual reciprocity violation -- confirmed empirically: those
    differences grew to ~1.6-1.8 (meaningless for quantities bounded by 1) as
    soon as delta_L_um left the symmetric delta_L_um=0 case, while S21 vs S12
    stayed small (<0.005) throughout, exactly as genuine reciprocity predicts
    regardless of the arms' asymmetry.
    """
    d_input_leak = float(np.max(np.abs(result.S21 - result.S12)))
    return {
        "max_diff_input_leak": d_input_leak,
        "max_diff": d_input_leak, "tolerance": tol, "passed": d_input_leak < tol,
    }


def run_all_checks(result: MZIResult, energy_tol: float = 0.03, reciprocity_tol: float = 0.03,
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
# coupler.py but storing COMPLEX S-parameters rather than power fractions.
# ---------------------------------------------------------------------------
@dataclass
class MZIArtifact:
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


def save_artifact(path_stem, result: MZIResult, metadata: dict) -> None:
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path_stem.parent / (path_stem.name + ".npz"),
        wavelengths_um=result.wavelengths_um, freqs=result.freqs,
        S11=result.S11, S21=result.S21, S31=result.S31, S41=result.S41,
        S22=result.S22, S12=result.S12, S32=result.S32, S42=result.S42,
    )
    full_metadata = {**metadata, "port_names": list(result.port_names)}
    (path_stem.parent / (path_stem.name + ".json")).write_text(json.dumps(full_metadata, indent=2, default=str))


def load_artifact(path_stem) -> MZIArtifact:
    path_stem = Path(path_stem)
    npz_path = path_stem.parent / (path_stem.name + ".npz")
    json_path = path_stem.parent / (path_stem.name + ".json")
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"MZI artifact not found at {path_stem}(.npz/.json). "
            "Run notebooks/07_mzi.ipynb first to generate it."
        )
    data = np.load(npz_path)
    metadata = json.loads(json_path.read_text())
    return MZIArtifact(
        wavelengths_um=data["wavelengths_um"], freqs=data["freqs"],
        S11=data["S11"], S21=data["S21"], S31=data["S31"], S41=data["S41"],
        S22=data["S22"], S12=data["S12"], S32=data["S32"], S42=data["S42"],
        port_names=tuple(metadata["port_names"]), metadata=metadata,
    )
