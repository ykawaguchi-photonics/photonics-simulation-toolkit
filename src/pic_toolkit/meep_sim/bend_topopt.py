"""Meep adjoint (gradient-based) topology optimization of a 90-degree silicon
waveguide bend.

This is the fourth "LEGO block" in the toolkit, and the first that does NOT
pick a design from a small parametric family (a radius, a gap) -- instead, a
freeform pixel density inside a fixed design-region footprint is optimized
directly against an adjoint-computed gradient (`meep.adjoint` + `autograd` +
`nlopt`). The optimization itself is genuinely expensive (one forward + one
adjoint FDTD run per iteration), so unlike `bend.py`'s radius sweep, this
module does NOT sweep a family of independent baselines -- there is one
optimization run, and its own per-iteration objective history (Section 6 of
the notebook) IS this component's "sweep" axis (iteration, not a parameter
grid).

Two genuinely different kinds of simulation live here, both built on the same
domain layout (straight input/output arms meeting a MaterialGrid design
region at the bottom-right corner, adapted from the standalone
`inverse_design_silicon_waveguide_bend_meep_adjoint_v2.ipynb` notebook this
module ports and integrates):

* `build_optimization_problem()` / `run_adjoint_optimization()` -- the
  adjoint optimization itself. Objective is `|c_out / c_in|^2` (normalized by
  the reference input coefficient, unlike the source notebook's raw
  `|c_top|^2`) at a single design wavelength, filtered (`mpa.conic_filter`,
  minimum feature size) and projected (`mpa.tanh_projection`, binarization)
  before every evaluation.
* `simulate_baseline()` -- a PLAIN forward two-port FDTD extraction (no
  adjoint, no autograd), structurally a direct port of `bend.py`'s two-
  independent-direction eigenmode-decomposition method, run on a FROZEN
  design (`weights`). This is what replaces the source notebook's guessed
  `R_final = 0.015` placeholder with an honestly measured S11/S22 -- it is
  the only thing this module's results are trusted from, exactly per the
  toolkit's "sanity-check before trusting anything" rule.

Same 2D effective-index convention as every other `meep_sim` module. Only
this file (plus `waveguide.py`, `bend.py`, `the since-removed ring.py`)
imports meep; this is also the only file that imports `meep.adjoint`,
`autograd`, and `nlopt`.

**TE/TM correction (found while auditing `waveguide.py`/`bend.py`, fixed
here too):** `eig_parity` is now forced to `mp.TE` everywhere a mode is
launched or decomposed (`_make_simulation`, `_run_one_direction`, and the
adjoint source/objective in `build_optimization_problem`), and the DFT
field capture records whichever of Ez/Hz is actually dominant rather than
assuming Ez. See docs/simulation_settings_record.md for the MPB
investigation behind this fix.
"""

from __future__ import annotations

import contextlib
import io
import platform
from dataclasses import dataclass, field
from datetime import date

import meep as mp
import meep.adjoint as mpa
import numpy as np
import nlopt
from autograd import numpy as npa
from autograd import tensor_jacobian_product

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

# ---------------------------------------------------------------------------
# User-adjustable parameters.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "wg_width_um": 0.5,          # waveguide core width
    "core_index": 2.7,           # silicon (same effective index as every other component)
    "clad_index": 1.44,          # silicon dioxide
    "cell_x_um": 9.8,             # domain size -- grown to keep the same PML clearance
    "cell_y_um": 8.8,             # around the design region below (see its comment)
    "dpml_um": 1.0,
    "design_region_center": (1.0, -0.5),  # bottom-right corner, where the input
                                           # (horizontal) and output (vertical) arms meet
    "design_region_x_um": 4.5,   # sized to comfortably contain a full-radius R=1um Euler
    "design_region_y_um": 4.5,   # (clothoid) warm-start curve -- see docs/simulation_
                                  # settings_record.md for why a blank uniform-gray guess
                                  # failed here and a warm start was needed instead.
    "init_bend_radius_um": 1.0,  # minimum radius of the Euler-spiral warm-start curve
                                  # build_euler_initial_density() rasterizes as the
                                  # optimizer's starting point. A pure Euler spiral's
                                  # footprint at a given minimum radius is ~1.87x that
                                  # radius (not 1x like a circular arc), which is why the
                                  # design region above is sized bigger than radius_um.
    "design_grid_n": 31,         # Nx=Ny=31 -> 961 design variables, grid pitch 0.15um --
                                  # a warm-started run is refining an already-good shape,
                                  # not discovering one from scratch, so it needs less
                                  # budget per variable than a blank-start run would.
    "resolution": 20,            # pixels/um
    "wavelength_um": 1.35,       # single design wavelength the adjoint objective targets
                                  # (center of the O-band validation range below)
    "source_fwidth_frac": 0.2,   # adjoint source df = 0.2*fcen (source notebook's value)
    "source_inset_um": 0.3,      # source sits this far inside the PML boundary
    "port_gap_um": 0.7,          # o1/o2 reference planes sit this much further from
                                  # the source than the source itself sits from PML --
                                  # o1 lands exactly at the source notebook's x=-1.0;
                                  # o2 is placed the same way (NOT the source notebook's
                                  # y=1.2, which left no room for a reverse-direction
                                  # source -- see _domain()) so a genuine two-port
                                  # validation is possible, not just one-directional T.
    "mode_size_um": 2.0,         # source/monitor plane extent, transverse to propagation
    "filter_radius_um": 0.2,     # conic filter radius -> approximate minimum feature size
    "eta": 0.5,                  # tanh projection midpoint
    "beta_schedule": [4.0, 8.0, 16.0, 32.0],  # tanh-projection sharpness, run in
                                  # increasing stages (beta continuation) -- starting soft
                                  # and progressively forcing binarization stabilizes
                                  # convergence; see docs/simulation_settings_record.md
                                  # for why (and for the resulting per-stage objective
                                  # dips visible in Section 6's convergence plot).
    "iters_per_stage": 20,       # NLopt MMA evaluations per beta stage -> 80 total.
    "adjoint_minimum_run_time": 50,  # floor on each forward/adjoint FDTD run's length
                                  # (meep time units), passed to OptimizationProblem --
                                  # prevents a too-early stop (governed by decay_by alone)
                                  # from making individual iterations' objective/gradient
                                  # noisy, another source of the same instability.
    "init_density": 0.5,         # uniform initial guess (fully unbiased 50/50 density)
    "init_step": 0.02,           # NLopt MMA initial step size
    "wl_min_um": 1.3,            # validation band (Section 8's honest two-port check) --
    "wl_max_um": 1.4,            # O-band, same convention as every other component
    "n_freq": 21,                # matches waveguide.py/bend.py's coarse validation grid
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


@dataclass
class OptimizationResult:
    evaluation_history: np.ndarray   # objective J per NLopt iteration
    x_opt: np.ndarray                # raw (pre-filter/projection) optimizer output
    final_weights_continuous: np.ndarray  # (n, n) filtered + projected, NOT thresholded
    final_weights_binarized: np.ndarray   # (n, n) thresholded at 0.5 -- the actual design
    sim_params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared domain/geometry helpers.
# ---------------------------------------------------------------------------
def _domain(params: dict) -> dict:
    """Reference-plane and source-plane coordinates, derived from the cell/
    design-region layout so the optimization objective (build_optimization_
    problem) and the honest validation (simulate_baseline) measure at
    EXACTLY the same two planes.

    o1 (input, horizontal arm, x-normal) reproduces the source notebook's
    x=-1.0 exactly. o2 (output, vertical arm, y-normal) does NOT reuse the
    source notebook's y=1.2 -- that location left no physical-region room
    for a reverse-direction (o2 -> o1) source before hitting the PML, and a
    genuine 2x2 S-matrix needs both directions excited independently. o2 is
    instead placed the same way o1 is: `port_gap_um` beyond a source that
    itself sits `source_inset_um` inside the PML boundary.
    """
    Sx, Sy = params["cell_x_um"], params["cell_y_um"]
    dpml = params["dpml_um"]
    x0, y0 = params["design_region_center"]
    inset = params["source_inset_um"]
    gap = params["port_gap_um"]

    src1_x = -Sx / 2 + dpml + inset
    port1_x = src1_x + gap
    src2_y = Sy / 2 - dpml - inset
    port2_y = src2_y - gap

    return {"x0": x0, "y0": y0, "src1_x": src1_x, "port1_x": port1_x,
            "src2_y": src2_y, "port2_y": port2_y}


def _design_region(params: dict, weights=None):
    """Build the MaterialGrid + its DesignRegion, frozen at `weights` (a
    uniform `init_density` guess by default -- the pre-optimization state)."""
    x0, y0 = params["design_region_center"]
    Lx, Ly = params["design_region_x_um"], params["design_region_y_um"]
    n = params["design_grid_n"]
    core = mp.Medium(index=params["core_index"])
    clad = mp.Medium(index=params["clad_index"])

    design_variables = mp.MaterialGrid(mp.Vector3(n, n, 0), clad, core, grid_type="U_MEAN")
    if weights is None:
        weights = params["init_density"] * np.ones((n, n))
    design_variables.update_weights(np.asarray(weights, dtype=float).reshape(n, n))

    design_region = mpa.DesignRegion(
        design_variables, volume=mp.Volume(center=mp.Vector3(x0, y0, 0), size=mp.Vector3(Lx, Ly, 0))
    )
    return design_variables, design_region


def build_geometry(params: dict, weights=None) -> list:
    """Straight input (horizontal) and output (vertical) arms, oversized so
    they extend THROUGH the bottom-right design region -- the design region
    is listed last, so Meep's later-object precedence lets its MaterialGrid
    density override the arms wherever they overlap it. Everywhere else, the
    arms are the only material -- ordinary single-material silicon wire."""
    core = mp.Medium(index=params["core_index"])
    w = params["wg_width_um"]
    Sx, Sy = params["cell_x_um"], params["cell_y_um"]
    x0, y0 = params["design_region_center"]
    Lx, Ly = params["design_region_x_um"], params["design_region_y_um"]

    in_x1, in_x2 = -Sx / 2, x0 + Lx / 2
    in_wg = mp.Block(
        center=mp.Vector3((in_x1 + in_x2) / 2, y0),
        size=mp.Vector3(in_x2 - in_x1, w),
        material=core,
    )
    out_y1, out_y2 = y0 - Ly / 2, Sy / 2
    out_wg = mp.Block(
        center=mp.Vector3(x0, (out_y1 + out_y2) / 2),
        size=mp.Vector3(w, out_y2 - out_y1),
        material=core,
    )

    design_variables, design_region = _design_region(params, weights)
    design_block = mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    )
    return [in_wg, out_wg, design_block]


def _cell_and_clad(params: dict):
    cell = mp.Vector3(params["cell_x_um"], params["cell_y_um"], 0)
    clad = mp.Medium(index=params["clad_index"])
    return cell, clad


def get_permittivity_map(params: dict, weights=None) -> PermittivityMap:
    """Return the permittivity map WITHOUT running any FDTD timestepping,
    exactly as in every other component -- `weights=None` shows the uniform
    (pre-optimization) design region; pass the optimized weights to inspect
    the final design instead."""
    cell, clad = _cell_and_clad(params)
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=build_geometry(params, weights),
        default_material=clad,
        resolution=params["resolution"],
        eps_averaging=False,  # matches the source notebook -- subpixel averaging
                               # interacts poorly with MaterialGrid's own interpolation
    )
    with _quiet_meep():
        sim.init_sim()
    eps = sim.get_epsilon()
    extent = (-cell.x / 2, cell.x / 2, -cell.y / 2, cell.y / 2)
    return PermittivityMap(eps=eps, extent_um=extent)


# ---------------------------------------------------------------------------
# Plain forward two-port validation (simulate_baseline) -- structurally a
# direct port of bend.py's extraction method, just built on a frozen
# MaterialGrid geometry instead of a parametric arc.
# ---------------------------------------------------------------------------
def _make_simulation(params: dict, weights, launch_from: str, capture_dft: bool):
    cell, clad = _cell_and_clad(params)
    fcen = 1.0 / params["wavelength_um"]
    fwidth = 1.0 / params["wl_min_um"] - 1.0 / params["wl_max_um"]
    dom = _domain(params)
    mode_size = params["mode_size_um"]

    if launch_from == "o1":
        src_center = mp.Vector3(dom["src1_x"], dom["y0"])
        src_size = mp.Vector3(0, mode_size, 0)
        eig_kpoint = mp.Vector3(1, 0, 0)
    else:
        src_center = mp.Vector3(dom["x0"], dom["src2_y"])
        src_size = mp.Vector3(mode_size, 0, 0)
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

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=build_geometry(params, weights),
        sources=[source],
        default_material=clad,
        resolution=params["resolution"],
        eps_averaging=False,
    )

    port1_center = mp.Vector3(dom["port1_x"], dom["y0"])
    port1_size = mp.Vector3(0, mode_size, 0)
    port2_center = mp.Vector3(dom["x0"], dom["port2_y"])
    port2_size = mp.Vector3(mode_size, 0, 0)

    mon1 = sim.add_mode_monitor(
        fcen, fwidth, params["n_freq"], mp.ModeRegion(center=port1_center, size=port1_size)
    )
    mon2 = sim.add_mode_monitor(
        fcen, fwidth, params["n_freq"], mp.ModeRegion(center=port2_center, size=port2_size)
    )

    dft_obj = None
    if capture_dft:
        dft_obj = sim.add_dft_fields([mp.Ez, mp.Hz], fcen, fcen, 1, center=mp.Vector3(), size=cell)

    return sim, mon1, mon2, dft_obj


def _run_one_direction(params: dict, weights, launch_from: str, capture_dft: bool):
    sim, mon1, mon2, dft_obj = _make_simulation(params, weights, launch_from, capture_dft)
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
        if np.max(np.abs(ez)) >= np.max(np.abs(hz)):
            field, component = ez, "Ez"
        else:
            field, component = hz, "Hz"
        field_snapshot = FieldSnapshot(field=field, eps=eps, extent_um=extent, component=component)

    return freqs, a1, a2, field_snapshot


def simulate_baseline(params: dict | None = None, weights=None) -> BaselineResult:
    """Honest two-port characterization of a FROZEN design (`weights`) --
    two independent FDTD runs (excite o1, excite o2), exactly as in
    bend.py, so S12=S21/S22=S11 are checked, not assumed. This is the only
    source of truth this module saves or trusts; the optimizer's own
    per-iteration objective (run_adjoint_optimization) is a cheap proxy,
    not a substitute for this."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    mp.verbosity(0)

    freqs, a1_run_a, a2_run_a, field_snapshot = _run_one_direction(
        params, weights, launch_from="o1", capture_dft=True
    )
    incident_1 = a1_run_a[:, 0]
    reflected_1 = a1_run_a[:, 1]
    transmitted_21 = a2_run_a[:, 0]
    s11 = reflected_1 / incident_1
    s21 = transmitted_21 / incident_1

    _, a1_run_b, a2_run_b, _ = _run_one_direction(
        params, weights, launch_from="o2", capture_dft=False
    )
    incident_2 = a2_run_b[:, 1]
    reflected_2 = a2_run_b[:, 0]
    transmitted_12 = a1_run_b[:, 1]
    s22 = reflected_2 / incident_2
    s12 = transmitted_12 / incident_2

    permittivity = get_permittivity_map(params, weights)

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


# ---------------------------------------------------------------------------
# Adjoint topology optimization.
# ---------------------------------------------------------------------------
def build_optimization_problem(params: dict):
    """Assemble the mpa.OptimizationProblem: the same input/output arms as
    build_geometry(), a MaterialGrid design region, and EigenmodeCoefficient
    monitors at the SAME o1/o2 reference planes simulate_baseline() uses --
    so the optimization objective and the final honest validation measure at
    identical locations. Objective is `|c_out / c_in|^2`, normalized by the
    reference input coefficient (unlike the source notebook's raw
    `|c_top|^2`) so it approximates true transmission, not an arbitrary
    unnormalized power.

    Returns (opt, design_variables, design_region) -- design_variables is
    what `nlopt`'s objective callback updates every iteration.
    """
    dom = _domain(params)
    fcen = 1.0 / params["wavelength_um"]
    df = params["source_fwidth_frac"] * fcen
    mode_size = params["mode_size_um"]
    core = mp.Medium(index=params["core_index"])
    w = params["wg_width_um"]
    Sx, Sy = params["cell_x_um"], params["cell_y_um"]
    x0, y0 = params["design_region_center"]
    Lx, Ly = params["design_region_x_um"], params["design_region_y_um"]

    design_variables, design_region = _design_region(params)

    in_x1, in_x2 = -Sx / 2, x0 + Lx / 2
    in_wg = mp.Block(
        center=mp.Vector3((in_x1 + in_x2) / 2, y0),
        size=mp.Vector3(in_x2 - in_x1, w),
        material=core,
    )
    out_y1, out_y2 = y0 - Ly / 2, Sy / 2
    out_wg = mp.Block(
        center=mp.Vector3(x0, (out_y1 + out_y2) / 2),
        size=mp.Vector3(w, out_y2 - out_y1),
        material=core,
    )
    design_block = mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    )

    source = mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=df),
        center=mp.Vector3(dom["src1_x"], dom["y0"]),
        size=mp.Vector3(0, mode_size, 0),
        eig_band=1,
        eig_parity=mp.TE,
        eig_match_freq=True,
        direction=mp.NO_DIRECTION,
        eig_kpoint=mp.Vector3(1, 0, 0),
    )

    sim = mp.Simulation(
        cell_size=mp.Vector3(Sx, Sy, 0),
        boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=[in_wg, out_wg, design_block],
        sources=[source],
        resolution=params["resolution"],
        eps_averaging=False,
    )

    TE_in = mpa.EigenmodeCoefficient(
        sim, mp.Volume(center=mp.Vector3(dom["port1_x"], dom["y0"]), size=mp.Vector3(0, mode_size, 0)),
        mode=1, eig_parity=mp.TE,
    )
    TE_out = mpa.EigenmodeCoefficient(
        sim, mp.Volume(center=mp.Vector3(dom["x0"], dom["port2_y"]), size=mp.Vector3(mode_size, 0, 0)),
        mode=1, eig_parity=mp.TE,
    )

    def objective_function(c_out, c_in):
        return npa.abs(c_out / c_in) ** 2

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=objective_function,
        objective_arguments=[TE_out, TE_in],
        design_regions=[design_region],
        fcen=fcen,
        df=0,
        nf=1,
        minimum_run_time=params["adjoint_minimum_run_time"],
    )
    return opt, design_variables, design_region


def build_euler_initial_density(params: dict, radius_um: float) -> np.ndarray:
    """Rasterize a symmetric 90-degree Euler (clothoid) bend of true minimum
    radius `radius_um` onto the design region's (design_grid_n x
    design_grid_n) density grid -- a warm-start initial guess for
    run_adjoint_optimization(init_weights=...), instead of a uniform gray
    density (see design_region_x_um's comment in DEFAULT_PARAMS for why).

    The centerline is built by numerically integrating the standard
    symmetric Euler-spiral curvature profile (ramps linearly 0 -> 1/
    radius_um over the first half-arc-length, then back to 0 over the
    second half, sweeping a total 90-degree turn -- Fresnel integrals have
    no elementary closed form, hence numeric). It is placed so it starts
    EXACTLY on the input arm's line (y=y0, heading +x) and ends EXACTLY on
    the output arm's line (x=x0, heading +y) -- both by construction, not
    approximation. Straight connector segments fill the design region on
    both sides of the curve out to its boundary, because build_geometry()'s
    input/output arms extend THROUGH the design region and the MaterialGrid
    overrides them wherever it's defined -- without these connectors the
    density would cut a gap in the arms rather than continue them.

    Raises ValueError if the curve doesn't fit inside the design region
    (see design_region_x_um/y_um and init_bend_radius_um's comments).

    NOTE this is a warm start, not a guarantee: run_adjoint_optimization()
    still passes every candidate through _mapping()'s conic filter + tanh
    projection before simulating it, which will smooth/gray this mask's
    sharp edges somewhat on the very first evaluation (over ~filter_
    radius_um), not reproduce it pixel-exact.
    """
    x0, y0 = params["design_region_center"]
    Lx, Ly = params["design_region_x_um"], params["design_region_y_um"]
    n = params["design_grid_n"]
    w = params["wg_width_um"]

    L = np.pi * radius_um / 2  # half-arc-length: theta(L) = L/(2*radius_um) = pi/4
    s = np.linspace(0, 2 * L, 4000)
    kappa = np.where(s <= L, s / (L * radius_um), (2 * L - s) / (L * radius_um))
    theta = np.concatenate([[0.0], np.cumsum((kappa[1:] + kappa[:-1]) / 2 * np.diff(s))])
    dx = np.concatenate([[0.0], np.cumsum((np.cos(theta[1:]) + np.cos(theta[:-1])) / 2 * np.diff(s))])
    dy = np.concatenate([[0.0], np.cumsum((np.sin(theta[1:]) + np.sin(theta[:-1])) / 2 * np.diff(s))])

    curve_x = x0 - dx[-1] + dx  # ends exactly at x0
    curve_y = y0 + dy           # starts exactly at y0

    if curve_x[0] < x0 - Lx / 2 or curve_y[-1] > y0 + Ly / 2:
        raise ValueError(
            f"Euler bend at radius_um={radius_um} (footprint {dx[-1]:.3f}um) does "
            f"not fit inside the {Lx}x{Ly}um design region -- widen it or use a "
            "smaller radius_um."
        )

    left_run_x = np.linspace(x0 - Lx / 2, curve_x[0], 50)
    left_run_y = np.full_like(left_run_x, y0)
    top_run_y = np.linspace(curve_y[-1], y0 + Ly / 2, 50)
    top_run_x = np.full_like(top_run_y, x0)

    path_x = np.concatenate([left_run_x, curve_x, top_run_x])
    path_y = np.concatenate([left_run_y, curve_y, top_run_y])

    grid_x = np.linspace(x0 - Lx / 2, x0 + Lx / 2, n)
    grid_y = np.linspace(y0 - Ly / 2, y0 + Ly / 2, n)
    gx, gy = np.meshgrid(grid_x, grid_y, indexing="ij")
    dist2 = (gx[..., None] - path_x) ** 2 + (gy[..., None] - path_y) ** 2
    min_dist = np.sqrt(dist2.min(axis=-1))
    return np.where(min_dist <= w / 2, 1.0, 0.0)


def _mapping(x, params: dict):
    """Conic filter (minimum feature size) + tanh projection (binarization
    pressure), applied before every objective evaluation -- an addition
    over the source notebook, which optimized the raw pixel grid directly."""
    n = params["design_grid_n"]
    Lx, Ly = params["design_region_x_um"], params["design_region_y_um"]
    # meep.adjoint.mesh_grid derives its own Nx = round(Lx*resolution)+1 --
    # (n-1)/Lx is what makes that come back out to n, matching design_grid_n.
    design_region_resolution = (n - 1) / Lx
    filtered = mpa.conic_filter(
        x.reshape(n, n), params["filter_radius_um"], Lx, Ly, design_region_resolution
    )
    projected = mpa.tanh_projection(filtered, params["beta"], params["eta"])
    return projected.flatten()


def run_adjoint_optimization(
    params: dict | None = None, init_weights: np.ndarray | None = None
) -> OptimizationResult:
    """NLopt LD_MMA loop over build_optimization_problem()'s objective,
    filtered/projected by _mapping() every iteration. Prints per-iteration
    progress in the same `[i/N  P%]` style as sweep.grid_sweep -- this
    module has no parameter-grid sweep; the per-iteration objective history
    returned here IS its sweep-equivalent axis.

    Runs `beta_schedule` as successive BETA CONTINUATION stages -- each stage
    re-optimizes from the previous stage's result at a higher tanh-projection
    sharpness, rather than optimizing at one fixed (and, empirically, less
    stable) beta for the whole budget. `evaluation_history` is the
    concatenation of every stage's objective values, in run order.

    `init_weights`, if given (any shape that flattens to design_grid_n**2 --
    e.g. build_euler_initial_density()'s (n, n) array), is the starting
    point instead of a uniform `init_density` gray field. See
    design_region_x_um's comment in DEFAULT_PARAMS for why a warm start is
    used here, unlike every other component.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    mp.verbosity(0)

    opt, design_variables, design_region = build_optimization_problem(params)
    n = params["design_grid_n"]
    n_params = n * n
    beta_schedule = params["beta_schedule"]
    iters_per_stage = params["iters_per_stage"]
    n_iterations = len(beta_schedule) * iters_per_stage

    evaluation_history = []
    x_cur = (
        np.asarray(init_weights, dtype=float).flatten()
        if init_weights is not None
        else params["init_density"] * np.ones(n_params)
    )

    for beta in beta_schedule:
        stage_params = {**params, "beta": beta}

        def nlopt_objective(x, gradient, stage_params=stage_params):
            mapped = _mapping(x, stage_params)
            with _quiet_meep():
                f0, dJ_du = opt([mapped])
            f0 = np.real(f0[0]) if hasattr(f0, "__len__") else float(np.real(f0))
            if gradient.size > 0:
                dJ_du = np.squeeze(dJ_du)
                gradient[:] = tensor_jacobian_product(_mapping, 0)(x, stage_params, dJ_du)
            evaluation_history.append(float(f0))
            i = len(evaluation_history)
            print(f"[{i}/{n_iterations}  {100 * i // n_iterations}%]  beta={beta:g}  J={float(f0):.6f}",
                  flush=True)
            return float(f0)

        solver = nlopt.opt(nlopt.LD_MMA, n_params)
        solver.set_lower_bounds(0.0)
        solver.set_upper_bounds(1.0)
        solver.set_initial_step(params["init_step"])
        solver.set_max_objective(nlopt_objective)
        solver.set_maxeval(iters_per_stage)
        x_cur = solver.optimize(x_cur)

    final_params = {**params, "beta": beta_schedule[-1]}
    final_weights_continuous = np.array(_mapping(x_cur, final_params)).reshape(n, n)
    final_weights_binarized = np.where(final_weights_continuous >= 0.5, 1.0, 0.0)

    sim_params = {
        **params,
        "meep_version": mp.__version__,
        "python_version": platform.python_version(),
        "creation_date": date.today().isoformat(),
    }

    return OptimizationResult(
        evaluation_history=np.array(evaluation_history),
        x_opt=x_cur,
        final_weights_continuous=final_weights_continuous,
        final_weights_binarized=final_weights_binarized,
        sim_params=sim_params,
    )
