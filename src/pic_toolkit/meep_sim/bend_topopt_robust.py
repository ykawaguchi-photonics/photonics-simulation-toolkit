"""Fabrication-bias-robust adjoint topology optimization of a 90-degree
silicon waveguide bend -- the fifth component in the toolkit, and a direct
extension of the fourth (`bend_topopt.py`, which this module imports from
read-only and never modifies).

`bend_topopt.py`'s optimizer only ever sees ONE rendering of the design
variables per iteration (`eta=0.5`). Nothing prevents the optimizer from
converging onto a shape whose transmission is a knife-edge function of the
exact Si/clad boundary position -- a real fabrication process (lithography/
etch bias) never reproduces that boundary exactly, so a knife-edge design is
a real risk, not a hypothetical one. This module instead averages the
adjoint objective (and its gradient) over THREE renderings of the SAME raw
design variables every iteration -- nominal (`eta_i=0.5`), eroded (`eta_e`,
Si shrinks), and dilated (`eta_d`, Si grows) -- so the optimizer is pushed
toward shapes whose transmission is stable across a small manufacturing
bias, not just optimal at one exact boundary position. This is the standard
"eroded/intermediate/dilated" robust-topology-optimization technique
(Sigmund/Lazarov/Wang); see `04b_bend_topopt_robust.ipynb` for the
calibration this module's `eta_e`/`eta_d` were measured against and the
resulting fabrication-tolerance comparison against a fresh,
non-Euler-warm-started baseline design.

**Own, compact domain -- NOT `bend_topopt.py`'s.** An earlier revision of
this module reused `bend_topopt.build_optimization_problem` unmodified
(calling it 3 times), which worked because `eta` plays no role in that
function's construction -- only the domain/geometry did. This revision
changes the domain/geometry itself (a tight, `bend.py`-style asymmetric box
sized to the structure's own footprint, instead of `bend_topopt.py`'s
symmetric-clearance box, which wasted ~32% of its cell area), so that reuse
path no longer applies. Every function below whose name ends in `_compact`
is a deliberate, documented duplication of a `bend_topopt.py` function,
rewritten against this module's own `_domain_compact()` instead of
`bend_topopt._domain()`. Everything that does NOT depend on domain/cell
geometry (`_mapping_robust`, `render_boundary_position_um`,
`calibrate_eta_bias`, `find_eta_for_bias`, `apply_bias_to_frozen_mask`, and
the imports below) is reused unchanged. `bend_topopt.py` itself is NEVER
edited.

**Why three independent `mp.Simulation`s per iteration, not one:** a
`MaterialGrid` cannot be shared live across independently-timestepped
simulations, and nominal/eroded/dilated are physically different rendered
structures -- so this is genuinely three forward + three adjoint FDTD
solves per iteration, ~3x one single-objective run's per-iteration cost,
budgeted for explicitly rather than hidden.

Same 2D effective-index convention, same corrected `eig_parity=mp.TE`
everywhere (inherited unchanged from the imported functions below).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import date

import meep as mp
import meep.adjoint as mpa
import numpy as np
import nlopt
import scipy.optimize
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
from autograd import numpy as npa
from autograd import grad as _autograd_grad
from autograd import tensor_jacobian_product

from .bend_topopt import (
    DEFAULT_PARAMS as _NOMINAL_DEFAULT_PARAMS,
    build_euler_initial_density,
    _design_region,
    _quiet_meep,
    PermittivityMap,
    FieldSnapshot,
    BaselineResult,
    OptimizationResult,
)

mp.verbosity(0)

# ---------------------------------------------------------------------------
# User-adjustable parameters -- a frozen copy of bend_topopt.DEFAULT_PARAMS
# (so a later edit to that module's defaults never silently drifts this one,
# same idiom mzm.py uses when it borrows constants from mzi.py), with
# `cell_x_um`/`cell_y_um`/`design_region_center` REMOVED (this module's
# compact domain computes these fresh every call via `_domain_compact()` --
# keeping the old fixed values around would be misleading, since nothing
# here reads them), plus the robustness- and compact-domain-specific keys.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    k: v for k, v in _NOMINAL_DEFAULT_PARAMS.items()
    if k not in ("cell_x_um", "cell_y_um", "design_region_center")
}
DEFAULT_PARAMS.update({
    "design_region_x_um": 2.5,   # 1.85 wavelengths. A first footprint sweep at resolution=20
    "design_region_y_um": 2.5,   # (contaminated by that resolution's own inadequacy -- see
                                  # design_grid_n's and resolution's own comments) picked 2.0um;
                                  # re-measured at the corrected settings (MMA, trained at
                                  # resolution=30, validated at resolution=40, corner-biased
                                  # start) against a conventional, un-optimized Euler bend of the
                                  # largest radius that fits each footprint:
                                  #
                                  #   region   conventional   topology-opt   advantage
                                  #   4.05um      0.9380         0.9412        +0.3%
                                  #   3.50um      0.8699         0.8897        +2.3%
                                  #   3.00um      0.7613         0.8678        +14.0%
                                  #   2.50um      0.6254         0.8163        +30.5%
                                  #   2.00um      0.4668         ~0.68-0.69    +46-48%
                                  #
                                  # 2.0um has the largest RELATIVE advantage, but its absolute
                                  # nominal transmission (~0.68) sits under a 0.75 target with
                                  # no margin, and -- importantly for a fabrication-ROBUSTNESS
                                  # study specifically -- a nominal design that already performs
                                  # poorly has correspondingly less room to demonstrate a
                                  # convincing "before/after" robustness story. 2.5um clears 0.80
                                  # while keeping a large (30.5%) margin over a conventional
                                  # bend, and was the smallest region to do so -- the balance
                                  # point between "small enough that topology optimization has
                                  # real work to do" and "high enough absolute performance for
                                  # the robustness study built on top of it to matter."
    "margin_um": 1.0,            # NEW -- clearance on the two "empty" sides of the compact
                                  # domain (see _domain_compact()), bend.py's own margin_um role.
    "arm_lead_um": 0.4,          # NEW -- straight-waveguide length between the port reference
                                  # plane and the design region's own edge (mode-settling
                                  # buffer). The OLD domain (inherited from bend_topopt.py's
                                  # symmetric-clearance box) left an unintentional ~1.65um
                                  # buffer here on ALL four sides, which was the actual cause of
                                  # its ~32% wasted cell area -- this makes the buffer explicit
                                  # and small on the two sides that need it, and drops it (down
                                  # to just dpml_um+margin_um) on the two that don't.
    "resolution": 30,             # px/um, OVERRIDES bend_topopt.py's 20. Measured: at 20 the
                                  # Yee grid carries only ~10 points per wavelength INSIDE
                                  # silicon (lambda_Si = 1.35/2.7 = 0.5um), under the usual >=20
                                  # rule of thumb. A resolution sweep on an otherwise identical
                                  # run gave honest T21 = 0.9440 (res 20) / 0.9585 (res 30) /
                                  # 0.9638 (res 40), and shrank the gap between the optimizer's
                                  # own objective and the honest two-port measurement from 0.032
                                  # to 0.011 at res 30 (no further at 40). 30 is the knee: +1.45
                                  # points over res 20 for 3.4x the cost, while 40 adds only
                                  # +0.53 more for another 3x on top.
    "design_grid_n": 101,          # 25nm pitch (2.5um/100) -- the same pitch verified at every
                                  # other region tried. At the smaller 2.0um region a FINER grid
                                  # (161, 12.5nm) measured WORSE (honest T21 0.7070 vs 0.7560 at
                                  # 81) and produced a blobbier design (perimeter/area 0.098 vs
                                  # 0.270), since the fixed 0.2um filter then smooths over
                                  # proportionally more pixels -- a caution against assuming
                                  # finer-is-better if this region size is revisited.
                                  # Why 25nm pitch at all: mpa.conic_filter
                                  # evaluates its own continuous kernel ONLY at the design grid's
                                  # own sample points, with no internal upsampling. At
                                  # bend_topopt.py's own design_grid_n=31 (150nm pitch) and
                                  # filter_radius_um=0.2, the filter's transition zone spanned
                                  # barely ~1.3 grid cells (measured: exactly 2 non-saturated
                                  # samples at a test edge) -- at beta=32 (the schedule's
                                  # sharpest, final stage), ANY eta between roughly 0.13 and 0.87
                                  # rendered an IDENTICAL, exactly-zero boundary shift. ~8 samples
                                  # across filter_radius_um's transition (the 25nm-pitch choice
                                  # here) is what restored a real, well-conditioned eta<->nm
                                  # relationship. This is free: FDTD cost is governed by
                                  # params["resolution"] (the fixed Yee grid), not design_grid_n.
    "init_bend_radius_um": 0.6,   # Euler warm-start radius for the ROBUST optimization's start
                                  # (at 2.5um, Lx/2=1.25um limits footprint=1.87*radius<=1.25,
                                  # i.e. radius<=0.668; 0.6 leaves real margin, same 0.92-of-max
                                  # ratio the conventional-Euler-reference measurement used).
                                  # (Section 7). build_euler_initial_density()'s curve is
                                  # anchored at the design region's own center (= the bend's
                                  # corner point), so it only ever occupies ONE quadrant of the
                                  # design region -- the real constraint is footprint (~1.87*
                                  # radius_um) <= design_region_x_um/2, NOT the full
                                  # design_region_x_um (confirmed directly: radius_um=1.0, the
                                  # value that fits comfortably in bend_topopt.py's own 4.5um
                                  # region (Lx/2=2.25 vs footprint 1.87), raises ValueError here
                                  # at 3.0um (Lx/2=1.5 vs footprint 1.87) -- the naive "does
                                  # 1.87*radius fit in Lx" check is wrong for this curve's own
                                  # anchoring). At this 4.5um region, radius_um=1.0 fits
                                  # (Lx/2=2.25 vs footprint 1.87, margin 0.38um) -- the same
                                  # value and margin notebook 04 itself uses; re-verify in the
                                  # notebook regardless (Section 9's own STOP checkpoint).
    "eta_i": 0.5,                 # intermediate/nominal threshold -- identical role to
                                  # bend_topopt.py's single "eta"
    "eta_e": None,                # erosion threshold (Si shrinks, NEGATIVE bias_um) -- None
                                  # until calibrated against target_bias_um via
                                  # find_eta_for_bias(); see the notebook's calibration section
    "eta_d": None,                # dilation threshold (Si grows, POSITIVE bias_um) -- same,
                                  # calibrated separately (target_bias_um negated)
    "target_bias_um": 0.010,      # +/-10nm fabrication bias eta_e/eta_d are calibrated to reach
                                  # (was +/-5nm -- too small an effect to be a meaningful test).
    "calibration_resolution": 200,  # px/um for calibrate_eta_bias's structure-only MaterialGrid
                                  # rendering (NOT the FDTD timestepping resolution below) --
                                  # far finer than params["resolution"]=20 because this is a
                                  # cheap, one-off geometric measurement, not a timestepped sim.
    "baseline_transmission_gate": 0.90,  # the 5-random-seed baseline search (Section 5 of the
                                  # notebook) must produce at least one design above this
                                  # transmission before the notebook proceeds to calibration/
                                  # robust optimization -- a hard gate, not a soft target; see
                                  # the notebook's own STOP checkpoint for what happens if none
                                  # of the 5 seeds clears it.
    "adjoint_maximum_run_time": 500,  # HANG GUARD on each forward/adjoint FDTD run, not a
                                  # convergence knob. A mid-optimization density can form a
                                  # high-Q resonant trap whose fields never satisfy
                                  # stop_when_dft_decayed()'s criterion; uncapped, one such
                                  # iteration ran 2+ hours at 100% CPU with zero progress and
                                  # silently consumed a 3-hour headless notebook run. Healthy runs on this domain
                                  # converge by t~200, and the cell's own one-way optical
                                  # transit is only ~24 meep time units, so 500 is ~2.5x normal
                                  # convergence / ~20 transits -- generous enough never to
                                  # truncate a well-behaved run, bounded enough that a pathology
                                  # costs one noisy iteration instead of the whole job.
    "validation_maximum_run_time": 1000,  # same guard for the honest two-port validation runs,
                                  # which are few and one-off, so they get a looser cap. If it
                                  # ever bites, checks.run_all_checks's energy-conservation test
                                  # is what surfaces the truncation rather than it passing quietly.
    "optimizer": "mma",          # "mma" or "adam". Adam (the Tidy3D tutorial's optimizer) is
                                  # implemented and available, but MMA measured better once the
                                  # comparison was made fairly. The trap worth recording: a
                                  # first sweep at resolution=20 made Adam(lr=0.2) look best
                                  # (0.7560 vs MMA's 0.7401) -- but BOTH were validated at
                                  # resolution=20, which over-reports this small, multi-component
                                  # 2.0um design by ~7 points (re-validating that same design
                                  # gives 0.6848 / 0.6945 / 0.6752 at resolution 30 / 40 / 50).
                                  # Re-swept at resolution=30 and validated at 40, the ordering
                                  # inverts: MMA 0.6870, Adam 0.6701 (lr=0.05), 0.6571 (0.1),
                                  # 0.6565 (0.02), 0.6143 (0.2). Adam's learning rate is also not
                                  # resolution-invariant -- the adjoint gradient's scale depends
                                  # on the discretization, so lr tuned at one resolution
                                  # overshoots at another (visible as fill fraction climbing
                                  # 0.299 -> 0.371 -> 0.440 -> 0.472 with lr).
    "learning_rate": 0.05,       # Adam step size, used only by run_adam_optimization_compact
                                  # (NLopt MMA has no learning rate -- its closest analogue is
                                  # init_step below). The Tidy3D tutorial uses Adam at lr=1.0,
                                  # but its design variables and gradient scaling differ from
                                  # this module's, so treat that value as a starting point to
                                  # sweep rather than a transferable constant.
    "adam_beta1": 0.9,           # standard Adam momentum terms
    "adam_beta2": 0.999,
    "adam_eps": 1e-8,
    "penalty_strength": 0.0,     # weight on minimum_length_penalty() in the objective.
                                  # 0.0 = OFF (the conic filter's implicit smoothing is the
                                  # only length-scale control, this module's original
                                  # behaviour). The Tidy3D topology-optimization tutorial
                                  # instead subtracts an erosion-dilation penalty from
                                  # transmission to enforce a minimum printable feature size
                                  # explicitly; set this > 0 to do the same here. Calibrate
                                  # it against the measured penalty magnitude at iteration 1
                                  # rather than guessing -- the transmission term is O(1)
                                  # while the raw constraint value is typically far smaller.
    "penalty_eta_e": 0.75,        # erosion threshold for the geometric constraint. 0.75
                                  # makes the enforced minimum feature size exactly equal to
                                  # filter_radius_um (see minimum_length_um()), which is the
                                  # tutorial's own choice; the void-phase constraint uses
                                  # 1-penalty_eta_e symmetrically.
    "penalty_c": 1000.0,          # constraint decay-rate parameter (meep.adjoint documents a
                                  # usable range of 1e0-1e8). Affects how sharply the
                                  # indicator localizes to violating pixels, not the length
                                  # scale itself.
    "baseline_corner_bias_sigma_um": 0.2,  # Gaussian width for build_corner_biased_random_
                                  # density()'s silicon-probability envelope -- set equal to
                                  # filter_radius_um as a principled starting point (a
                                  # perturbation length scale narrower than the filter that
                                  # will process it can't survive filtering any better than
                                  # pure per-pixel noise did; see that function's own
                                  # docstring). Re-verify (visually + via the measured fill
                                  # fraction) before trusting it, per this module's own
                                  # "measure, don't assume" discipline.
})


@dataclass
class RobustOptimizationResult:
    evaluation_history_nominal: np.ndarray
    evaluation_history_eroded: np.ndarray
    evaluation_history_dilated: np.ndarray
    evaluation_history_avg: np.ndarray   # objective NLopt actually maximized, per iteration
    x_opt: np.ndarray                    # raw (pre-filter/projection) optimizer output
    final_weights_continuous: dict       # {"nominal"/"eroded"/"dilated": (n,n) array}, NOT thresholded
    final_weights_binarized: dict        # same keys, thresholded at 0.5
    sim_params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Filtering/projection with an explicit eta -- lets the SAME raw x be
# projected at three different thresholds every iteration. Domain-agnostic
# (only reads design_grid_n/design_region_x_um/design_region_y_um/
# filter_radius_um) -- reused unchanged from the prior revision.
# ---------------------------------------------------------------------------
def _mapping_robust(x, params: dict, eta: float) -> np.ndarray:
    """Same conic_filter (minimum feature size) + tanh_projection pipeline as
    bend_topopt._mapping, but with `eta` as an explicit argument instead of
    reading params["eta"] -- deliberately duplicated (not imported) from that
    function, which hardcodes a single params["eta"] internally and has no
    way to take a per-call threshold.

    Clips to [0,1] (autograd-differentiable `npa.clip`) before returning --
    unlike bend_topopt._mapping, this module explores eta far from 0.5
    (eta_e/eta_d), where tanh_projection's floating-point output can
    overshoot [0,1] by a machine-epsilon-scale amount and trip mp.
    MaterialGrid's "weights must be in [0,1]" warning on update_weights().
    The clip has no effect on the true (already-saturated) gradient in that
    region."""
    n = params["design_grid_n"]
    Lx, Ly = params["design_region_x_um"], params["design_region_y_um"]
    design_region_resolution = (n - 1) / Lx
    filtered = mpa.conic_filter(
        x.reshape(n, n), params["filter_radius_um"], Lx, Ly, design_region_resolution
    )
    projected = mpa.tanh_projection(filtered, params["beta"], eta)
    projected = npa.clip(projected, 0.0, 1.0)
    return projected.flatten()


# ---------------------------------------------------------------------------
# Minimum-length-scale penalty (Zhou/Lazarov/Wang/Sigmund 2015 geometric
# constraints), the same mechanism the Tidy3D topology-optimization tutorial
# applies as its "erosion-dilation penalty". meep.adjoint ships these
# ready-made and autograd-differentiable, so the objective can enforce a
# minimum printable feature size explicitly instead of leaving it entirely to
# the conic filter's implicit smoothing.
# ---------------------------------------------------------------------------
def minimum_length_um(params: dict) -> float:
    """The minimum feature size the penalty's `penalty_eta_e` actually
    enforces, given this module's own `filter_radius_um`. Inverse of
    mpa.get_conic_radius_from_eta_e (which solves the other direction):

        eta_e in [0.5, 0.75):  b = 2 * R * sqrt(eta_e - 0.5)
        eta_e in [0.75, 1.0]:  b = R * (2 - 2*sqrt(1 - eta_e))

    so the default `penalty_eta_e = 0.75` enforces b = R exactly -- minimum
    feature size equal to the filter radius, matching the Tidy3D tutorial's
    own choice.
    """
    r = params["filter_radius_um"]
    eta_e = params["penalty_eta_e"]
    if 0.5 <= eta_e < 0.75:
        return float(2 * r * np.sqrt(eta_e - 0.5))
    if 0.75 <= eta_e <= 1.0:
        return float(r * (2 - 2 * np.sqrt(1 - eta_e)))
    raise ValueError("penalty_eta_e must be in [0.5, 1.0]")


def minimum_length_penalty(x, params: dict, beta: float):
    """Solid- and void-phase geometric constraints summed into a single
    scalar penalty, differentiable by autograd all the way back to the raw
    design variables `x`. Built from the SAME conic filter and tanh
    projection `_mapping_robust` uses, passed to mpa.constraint_solid /
    mpa.constraint_void as their `filter_f` / `threshold_f` handles, so the
    penalty and the geometry it penalizes cannot drift apart.

    Returns a small non-negative number (0 when no feature violates the
    length scale), which `run_adjoint_optimization_compact` SUBTRACTS from
    the transmission objective, scaled by `params["penalty_strength"]`.
    """
    n = params["design_grid_n"]
    Lx, Ly = params["design_region_x_um"], params["design_region_y_um"]
    design_region_resolution = (n - 1) / Lx

    def filter_f(a):
        return mpa.conic_filter(
            a.reshape(n, n), params["filter_radius_um"], Lx, Ly, design_region_resolution
        )

    def threshold_f(a):
        return mpa.tanh_projection(a, beta, params["eta_i"])

    eta_e = params["penalty_eta_e"]
    c = params["penalty_c"]
    g_solid = mpa.constraint_solid(x, c, eta_e, filter_f, threshold_f, design_region_resolution)
    g_void = mpa.constraint_void(x, c, 1.0 - eta_e, filter_f, threshold_f, design_region_resolution)
    return g_solid + g_void


# ---------------------------------------------------------------------------
# Structure metrics -- "did topology optimization actually earn its name
# here?" A plain smooth bend is one connected blob with no enclosed voids and
# a low perimeter-to-area ratio; a genuinely freeform design has detached
# pieces, enclosed holes, and/or far more boundary per unit area. Measured
# alongside transmission because a design that merely reproduces a
# conventional waveguide is a different (and less interesting) outcome even
# when its transmission is excellent.
# ---------------------------------------------------------------------------
def structure_metrics(binarized: np.ndarray) -> dict:
    solid = np.asarray(binarized) >= 0.5
    n_components = int(ndimage.label(solid)[1])
    voids, n_voids = ndimage.label(~solid)
    border_labels = set(voids[0, :]) | set(voids[-1, :]) | set(voids[:, 0]) | set(voids[:, -1])
    enclosed_holes = int(sum(1 for i in range(1, n_voids + 1) if i not in border_labels))
    perimeter = int((solid[:-1, :] != solid[1:, :]).sum() + (solid[:, :-1] != solid[:, 1:]).sum())
    area = int(solid.sum())
    return {
        "fill_fraction": float(solid.mean()),
        "n_components": n_components,
        "enclosed_holes": enclosed_holes,
        "perimeter_over_area": float(perimeter / max(area, 1)),
    }


# ---------------------------------------------------------------------------
# Calibration: map a physical +/-nm bias onto an eta threshold, by rendering
# a synthetic straight-edge test pattern through a REAL mp.MaterialGrid
# (structure only, no timestepping) at fine resolution and measuring where
# the resulting permittivity actually crosses the core/clad midpoint.
# Domain-agnostic -- reused unchanged from the prior revision.
# ---------------------------------------------------------------------------
def render_boundary_position_um(
    density_2d: np.ndarray,
    params: dict,
    beta: float,
    eta: float,
    edge_axis: str = "x",
    calibration_resolution: float | None = None,
) -> float:
    """Project `density_2d` (a RAW, unfiltered (n,n) test pattern) through
    _mapping_robust(..., eta) at the given beta, render it into a real
    mp.MaterialGrid inside a minimal structure-only mp.Simulation (init_sim()
    only -- no sim.run(), so this is cheap regardless of calibration_
    resolution), and locate the physical coordinate along `edge_axis` where
    the RENDERED epsilon crosses the core/clad midpoint. This measures what
    the real FDTD grid would actually see for this eta -- not a numpy-side
    approximation of the filter/projection curve alone -- because it goes
    through mp.MaterialGrid's own bilinear interpolation exactly as
    build_geometry_compact()'s design_block does.
    """
    calibration_resolution = calibration_resolution or params["calibration_resolution"]
    n = params["design_grid_n"]
    Lx, Ly = params["design_region_x_um"], params["design_region_y_um"]
    core = mp.Medium(index=params["core_index"])
    clad = mp.Medium(index=params["clad_index"])

    stage_params = {**params, "beta": beta}
    projected = _mapping_robust(np.asarray(density_2d, dtype=float).flatten(), stage_params, eta)
    projected = np.asarray(projected).reshape(n, n)

    design_variables = mp.MaterialGrid(mp.Vector3(n, n, 0), clad, core, grid_type="U_MEAN")
    design_variables.update_weights(projected)
    block = mp.Block(center=mp.Vector3(0, 0, 0), size=mp.Vector3(Lx, Ly, 0), material=design_variables)

    sim = mp.Simulation(
        cell_size=mp.Vector3(Lx, Ly, 0),
        resolution=calibration_resolution,
        geometry=[block],
        default_material=clad,
        eps_averaging=False,
    )
    with _quiet_meep():
        sim.init_sim()
    eps = sim.get_epsilon()

    core_eps = params["core_index"] ** 2
    clad_eps = params["clad_index"] ** 2
    mid_eps = 0.5 * (core_eps + clad_eps)

    nx, ny = eps.shape
    xs = np.linspace(-Lx / 2, Lx / 2, nx)
    ys = np.linspace(-Ly / 2, Ly / 2, ny)

    if edge_axis == "x":
        cut, coords = eps[:, ny // 2], xs
    else:
        cut, coords = eps[nx // 2, :], ys

    sign = np.sign(cut - mid_eps)
    crossings = np.where(np.diff(sign) != 0)[0]
    if len(crossings) == 0:
        raise RuntimeError(
            f"render_boundary_position_um: no epsilon-midpoint crossing found along "
            f"{edge_axis} for eta={eta}, beta={beta} -- check density_2d/eta_bracket range."
        )
    i0 = crossings[len(crossings) // 2]  # nearest-to-center crossing if more than one
    c0, c1 = coords[i0], coords[i0 + 1]
    v0, v1 = cut[i0], cut[i0 + 1]
    frac = (mid_eps - v0) / (v1 - v0)
    return float(c0 + frac * (c1 - c0))


def calibrate_eta_bias(
    params: dict,
    beta: float,
    eta_candidates: np.ndarray,
    edge_axis: str = "x",
    calibration_resolution: float | None = None,
) -> np.ndarray:
    """Build a synthetic straight-edge test pattern (density=0 on one side,
    1 on the other, split at the design region's own center along
    `edge_axis`, on the SAME design_grid_n grid this module optimizes on),
    and measure render_boundary_position_um(...) at eta=0.5 (reference) and
    at each of `eta_candidates`. Returns bias_um[i], the signed physical
    boundary shift in the "core grows/shrinks" sense: dilation (Si grows) =
    POSITIVE, erosion (Si shrinks) = NEGATIVE.

    Sign derivation (measured, not assumed): the test pattern below puts
    core (density=1) on the coord>=0 side. A HIGHER eta always shrinks
    whichever region is classified as density=1 (tanh_projection's
    threshold), so eta>0.5 erodes the core -- since core occupies
    [boundary, +inf), eroding it means the boundary must move in the
    POSITIVE direction (measured directly: eta=0.6 renders a more positive
    raw crossing than eta=0.5 does). Raw `position(eta) - position(0.5)` is
    therefore POSITIVE for erosion -- the opposite of the "dilation=+"
    convention `target_bias_um` uses -- so the value returned below is
    negated relative to that raw measured crossing shift.

    Valid as a LOCAL model of the actual bend's boundary wherever the real
    boundary's curvature radius is much larger than filter_radius_um -- true
    here (bend radius O(1um) >> filter_radius_um=0.2um) -- so a straight-edge
    calibration transfers to the real curved design region.
    """
    n = params["design_grid_n"]
    Lx, Ly = params["design_region_x_um"], params["design_region_y_um"]
    grid_x = np.linspace(-Lx / 2, Lx / 2, n)
    grid_y = np.linspace(-Ly / 2, Ly / 2, n)
    gx, gy = np.meshgrid(grid_x, grid_y, indexing="ij")
    test_pattern = np.where((gx if edge_axis == "x" else gy) >= 0, 1.0, 0.0)

    baseline = render_boundary_position_um(test_pattern, params, beta, 0.5, edge_axis, calibration_resolution)
    eta_candidates = np.atleast_1d(np.asarray(eta_candidates, dtype=float))
    biases = np.empty(len(eta_candidates))
    for i, eta in enumerate(eta_candidates):
        pos = render_boundary_position_um(test_pattern, params, beta, float(eta), edge_axis, calibration_resolution)
        biases[i] = -(pos - baseline)  # negated -- see sign derivation above
    return biases


def find_eta_for_bias(
    params: dict,
    beta: float,
    target_bias_um: float,
    eta_bracket: tuple = (0.05, 0.95),
    edge_axis: str = "x",
    calibration_resolution: float | None = None,
) -> float:
    """Root-find (scipy.optimize.brentq) the eta achieving `target_bias_um`
    on the same straight-edge test calibrate_eta_bias uses. Sign convention:
    a HIGHER eta classifies less material as core (erosion, negative bias);
    a LOWER eta classifies more (dilation, positive bias) -- so eta_bracket
    must straddle 0.5 in the direction target_bias_um requires."""

    def f(eta):
        return calibrate_eta_bias(
            params, beta, np.array([eta]), edge_axis, calibration_resolution
        )[0] - target_bias_um

    return float(scipy.optimize.brentq(f, eta_bracket[0], eta_bracket[1]))


# ---------------------------------------------------------------------------
# Compact domain -- bend.py-style tight, asymmetric box sized to the
# structure's own footprint (input arm reach + design region + output arm
# reach), instead of bend_topopt.py's symmetric-clearance box (which wasted
# ~32% of its cell area: a uniform clearance around the design region's
# bounding box on all 4 sides, rather than a minimal clearance only on the
# 2 sides that carry no arm). See this module's own docstring.
# ---------------------------------------------------------------------------
def _domain_compact(params: dict) -> dict:
    """Compute the compact cell size, design-region center, and port/source
    coordinates, mirroring bend.py's own `_domain()` pattern (tight box =
    structure's own reach + minimal necessary PML/margin clearance, square
    cell exploiting the L-shape's symmetry) but parametrized by "design
    region + straight lead" instead of "arc radius + straight arm".

    Layout: the input (horizontal) arm reaches from the cell's left edge to
    the design region's right edge; the output (vertical) arm reaches from
    the design region's bottom edge to the cell's top edge. The two "arm"
    sides (left, top) need `dpml_um + source_inset_um + port_gap_um +
    arm_lead_um` of reach (PML + source/port placement + a short buffer
    before the design region). The two "empty" sides (right, bottom -- no
    arm continues past the design region there) need only `dpml_um +
    margin_um`. Requires design_region_x_um == design_region_y_um (the
    square-cell symmetry this shares with bend.py breaks otherwise).
    """
    Lx, Ly = params["design_region_x_um"], params["design_region_y_um"]
    if abs(Lx - Ly) > 1e-9:
        raise ValueError(
            f"_domain_compact requires a square design region (got {Lx}x{Ly}um) -- "
            "the compact cell's square-symmetry formula assumes design_region_x_um "
            "== design_region_y_um."
        )
    half_w = params["wg_width_um"] / 2
    dpml = params["dpml_um"]

    reach_arm = dpml + params["source_inset_um"] + params["port_gap_um"] + params["arm_lead_um"]
    reach_empty = dpml + params["margin_um"]

    cell_side = reach_arm + Lx + reach_empty
    cell_side = max(cell_side, 2 * (half_w + reach_empty))  # degeneracy guard, mirrors bend.py

    x0 = (reach_arm - reach_empty) / 2
    y0 = -(reach_arm - reach_empty) / 2

    src1_x = -cell_side / 2 + dpml + params["source_inset_um"]
    port1_x = src1_x + params["port_gap_um"]
    src2_y = cell_side / 2 - dpml - params["source_inset_um"]
    port2_y = src2_y - params["port_gap_um"]

    mode_size = params["mode_size_um"]
    available = cell_side - 2 * dpml
    if mode_size > available:
        raise ValueError(
            f"mode_size_um={mode_size} does not fit inside cell_side={cell_side:.3f}um "
            f"(dpml_um={dpml} on each side leaves only {available:.3f}um) -- reduce "
            "mode_size_um or increase margin_um/arm_lead_um/design_region_x_um."
        )

    return {
        "x0": x0, "y0": y0,
        "cell_x_um": cell_side, "cell_y_um": cell_side,
        "src1_x": src1_x, "port1_x": port1_x,
        "src2_y": src2_y, "port2_y": port2_y,
        "reach_arm_um": reach_arm, "reach_empty_um": reach_empty,
    }


def build_geometry_compact(params: dict, weights=None) -> list:
    """Compact-domain analog of bend_topopt.build_geometry -- straight input
    (horizontal) and output (vertical) arms sized/positioned by
    _domain_compact() instead of bend_topopt.py's own symmetric-clearance
    domain. Deliberately duplicated (not imported) from build_geometry,
    which hardcodes bend_topopt._domain(); see this module's own docstring
    for why the domain itself had to change."""
    dom = _domain_compact(params)
    core = mp.Medium(index=params["core_index"])
    w = params["wg_width_um"]
    Sx = Sy = dom["cell_x_um"]
    x0, y0 = dom["x0"], dom["y0"]
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

    design_variables, design_region = _design_region(
        {**params, "design_region_center": (x0, y0)}, weights
    )
    design_block = mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    )
    return [in_wg, out_wg, design_block]


def _cell_and_clad_compact(params: dict):
    dom = _domain_compact(params)
    cell = mp.Vector3(dom["cell_x_um"], dom["cell_y_um"], 0)
    clad = mp.Medium(index=params["clad_index"])
    return cell, clad, dom


def get_permittivity_map_compact(params: dict, weights=None) -> PermittivityMap:
    """Compact-domain analog of bend_topopt.get_permittivity_map -- returns
    the permittivity map WITHOUT running any FDTD timestepping."""
    cell, clad, _ = _cell_and_clad_compact(params)
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=build_geometry_compact(params, weights),
        default_material=clad,
        resolution=params["resolution"],
        eps_averaging=False,
    )
    with _quiet_meep():
        sim.init_sim()
    eps = sim.get_epsilon()
    extent = (-cell.x / 2, cell.x / 2, -cell.y / 2, cell.y / 2)
    return PermittivityMap(eps=eps, extent_um=extent)


# ---------------------------------------------------------------------------
# Corner-biased random initial density -- the 5-random-seed baseline search's
# starting point (Section 5 of the notebook). NOT pure per-pixel i.i.d. noise:
# measured directly that such noise
# collapses through mpa.conic_filter to a near-constant ~0.5 field (std drops
# ~10x), then mpa.tanh_projection re-fragments that tiny residual into a
# spatially-UNCORRELATED binary texture -- effectively as uninformative a
# starting point as the already-documented-to-fail blank/uniform start, just
# noisier-looking. This generator instead makes each pixel's SILICON
# PROBABILITY a Gaussian function of its perpendicular distance from the
# sharp (radius=0) 90-degree corner path connecting the two ports (the same
# corner `bend.py`'s own `sharp_bend_geometry` uses, NOT the smooth Euler
# curve `build_euler_initial_density` rasterizes) -- concentrating density
# near a physically sensible path while keeping every pixel's OWN value a
# genuine, independent random draw (real seed-to-seed diversity), not a
# fixed deterministic shape.
# ---------------------------------------------------------------------------
def build_corner_biased_random_density(
    params: dict, seed: int, sigma_um: float, corner_xy: tuple,
) -> np.ndarray:
    """Bernoulli-sample an (n,n) density where pixel (x,y)'s probability of
    being silicon is `exp(-dist(x,y)**2 / (2*sigma_um**2))`, `dist` the
    perpendicular distance to the L-shaped path (horizontal segment from the
    design region's left edge to `corner_xy`, vertical segment from
    `corner_xy` to the design region's top edge -- `corner_xy` is the
    compact domain's own `(x0, y0)` from `_domain_compact`, i.e. the point
    where the input and output arms meet). Near the path (dist ~ 0),
    probability -> 1 (almost always silicon); far away, probability -> 0
    (almost always cladding); at dist ~ sigma_um, probability ~ 0.5 (a
    genuinely random fringe band). `sigma_um` sets how tightly density
    concentrates around the path -- too small approaches a deterministic
    sharp-corner shape (little real randomness left); too large approaches
    uniform ~1 fill (a different, but similarly uninformative, degenerate
    case). Verify the rendered density visually before trusting it (same
    "measure, don't assume" discipline as every other parameter here).
    """
    n = params["design_grid_n"]
    Lx, Ly = params["design_region_x_um"], params["design_region_y_um"]
    x0, y0 = corner_xy

    grid_x = np.linspace(x0 - Lx / 2, x0 + Lx / 2, n)
    grid_y = np.linspace(y0 - Ly / 2, y0 + Ly / 2, n)
    gx, gy = np.meshgrid(grid_x, grid_y, indexing="ij")

    # Distance to the horizontal segment (y=y0, x in [x0-Lx/2, x0]): straight
    # perpendicular distance |gy-y0| where gx is within the segment's x-range
    # (gx<=x0), else distance to the segment's near endpoint (the corner).
    dist_horiz = np.where(gx <= x0, np.abs(gy - y0), np.sqrt((gx - x0) ** 2 + (gy - y0) ** 2))
    # Distance to the vertical segment (x=x0, y in [y0, y0+Ly/2]): symmetric.
    dist_vert = np.where(gy >= y0, np.abs(gx - x0), np.sqrt((gx - x0) ** 2 + (gy - y0) ** 2))
    dist = np.minimum(dist_horiz, dist_vert)

    silicon_probability = np.exp(-(dist ** 2) / (2 * sigma_um ** 2))
    rng = np.random.default_rng(seed)
    return (rng.uniform(0.0, 1.0, (n, n)) < silicon_probability).astype(float)


# ---------------------------------------------------------------------------
# Plain forward two-port validation (compact domain) -- structurally a
# direct port of bend_topopt.simulate_baseline, just built on
# _domain_compact()/build_geometry_compact() instead of bend_topopt._domain/
# build_geometry.
# ---------------------------------------------------------------------------
def _make_simulation_compact(params: dict, weights, launch_from: str, capture_dft: bool):
    cell, clad, dom = _cell_and_clad_compact(params)
    fcen = 1.0 / params["wavelength_um"]
    fwidth = 1.0 / params["wl_min_um"] - 1.0 / params["wl_max_um"]
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
        geometry=build_geometry_compact(params, weights),
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


def _run_one_direction_compact(params: dict, weights, launch_from: str, capture_dft: bool):
    sim, mon1, mon2, dft_obj = _make_simulation_compact(params, weights, launch_from, capture_dft)
    with _quiet_meep():
        # maximum_run_time is a HANG GUARD, not a convergence setting: a
        # mid-optimization design can form a high-Q resonant trap whose fields
        # effectively never satisfy the decay criterion, and with no cap
        # `stop_when_dft_decayed()` then runs forever (measured: one adjoint
        # iteration burned 2+ hours at 100% CPU with no progress, which is also
        # what silently ate a 3-hour headless notebook run). A healthy run here converges by
        # t~200 (verified: identical S-parameters at minimum_run_time 0/50/
        # 200/400), so this cap is ~5x normal and only ever bites a pathology.
        # If it does bite, the energy-conservation check will show it rather
        # than the truncation passing silently.
        sim.run(until_after_sources=mp.stop_when_dft_decayed(
            maximum_run_time=params["validation_maximum_run_time"]))
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
        cell, _, _ = _cell_and_clad_compact(params)
        extent = (-cell.x / 2, cell.x / 2, -cell.y / 2, cell.y / 2)
        if np.max(np.abs(ez)) >= np.max(np.abs(hz)):
            field, component = ez, "Ez"
        else:
            field, component = hz, "Hz"
        field_snapshot = FieldSnapshot(field=field, eps=eps, extent_um=extent, component=component)

    return freqs, a1, a2, field_snapshot


def simulate_baseline_compact(params: dict | None = None, weights=None) -> BaselineResult:
    """Honest two-port characterization of a FROZEN design (`weights`) on
    the compact domain -- two independent FDTD runs (excite o1, excite o2),
    exactly as bend_topopt.simulate_baseline does. This is the only source
    of truth this module trusts; the optimizer's own per-iteration objective
    is a cheap proxy, not a substitute for this."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    mp.verbosity(0)

    freqs, a1_run_a, a2_run_a, field_snapshot = _run_one_direction_compact(
        params, weights, launch_from="o1", capture_dft=True
    )
    incident_1 = a1_run_a[:, 0]
    reflected_1 = a1_run_a[:, 1]
    transmitted_21 = a2_run_a[:, 0]
    s11 = reflected_1 / incident_1
    s21 = transmitted_21 / incident_1

    _, a1_run_b, a2_run_b, _ = _run_one_direction_compact(
        params, weights, launch_from="o2", capture_dft=False
    )
    incident_2 = a2_run_b[:, 1]
    reflected_2 = a2_run_b[:, 0]
    transmitted_12 = a1_run_b[:, 1]
    s22 = reflected_2 / incident_2
    s12 = transmitted_12 / incident_2

    permittivity = get_permittivity_map_compact(params, weights)

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
# Adjoint optimization problem construction (compact domain).
# ---------------------------------------------------------------------------
def build_optimization_problem_compact(params: dict):
    """Compact-domain analog of bend_topopt.build_optimization_problem --
    same objective (`|c_out/c_in|^2`, normalized transmission at a single
    design wavelength) and TE-forced eigenmode source/monitors, built on
    _domain_compact()/_design_region() instead of bend_topopt._domain().
    Returns (opt, design_variables, design_region)."""
    dom = _domain_compact(params)
    fcen = 1.0 / params["wavelength_um"]
    df = params["source_fwidth_frac"] * fcen
    mode_size = params["mode_size_um"]
    core = mp.Medium(index=params["core_index"])
    w = params["wg_width_um"]
    x0, y0 = dom["x0"], dom["y0"]
    Sx = Sy = dom["cell_x_um"]
    Lx, Ly = params["design_region_x_um"], params["design_region_y_um"]

    design_variables, design_region = _design_region({**params, "design_region_center": (x0, y0)})

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
        maximum_run_time=params["adjoint_maximum_run_time"],  # hang guard -- see
        # _run_one_direction_compact's own note on why this is not optional here.
    )
    return opt, design_variables, design_region


def build_robust_optimization_problem_compact(params: dict) -> dict:
    """Three independent calls to build_optimization_problem_compact --
    eta plays no role in that function's construction (see this module's
    docstring), so nominal/eroded/dilated only differ in which
    _mapping_robust(..., eta) output is fed into each one's opt(...) call,
    not in how each mp.Simulation/MaterialGrid/DesignRegion is built.
    Returns {"nominal": (opt, design_variables, design_region), "eroded":
    (...), "dilated": (...)}."""
    return {
        "nominal": build_optimization_problem_compact(params),
        "eroded": build_optimization_problem_compact(params),
        "dilated": build_optimization_problem_compact(params),
    }


# ---------------------------------------------------------------------------
# Single-objective (non-robust) adjoint optimization on the compact domain --
# used ONLY by the 5-random-seed baseline search (deliberately NOT
# Euler-warm-started, NOT the 3-way averaged objective): this is the "before
# robustness" design the +/-10nm fabrication-sensitivity comparison measures
# against, replacing the earlier revision's comparison against notebook 04's
# own (Euler-warm-started, and therefore already somewhat robust "by luck")
# design.
# ---------------------------------------------------------------------------
def run_adjoint_optimization_compact(
    params: dict | None = None, init_weights: np.ndarray | None = None,
) -> OptimizationResult:
    """Same NLopt LD_MMA / beta-continuation loop shape as bend_topopt.
    run_adjoint_optimization, built on build_optimization_problem_compact
    instead of the plain bend_topopt.build_optimization_problem. Always
    projects at eta_i=0.5 (single objective, no robustness averaging) --
    `params["eta_e"]`/`params["eta_d"]` need not be calibrated to call this.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    mp.verbosity(0)

    opt, design_variables, design_region = build_optimization_problem_compact(params)
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
            mapped = _mapping_robust(x, stage_params, params["eta_i"])
            with _quiet_meep():
                f0, dJ_du = opt([mapped])
            f0 = np.real(f0[0]) if hasattr(f0, "__len__") else float(np.real(f0))
            if gradient.size > 0:
                dJ_du = np.squeeze(dJ_du)
                gradient[:] = tensor_jacobian_product(_mapping_robust, 0)(
                    x, stage_params, params["eta_i"], dJ_du
                )

            # Optional minimum-length-scale penalty (off unless penalty_strength>0).
            # Purely a function of x through the same filter/projection, so its
            # gradient comes straight from autograd -- no extra FDTD solve.
            penalty = 0.0
            if params["penalty_strength"] > 0:
                penalty = float(minimum_length_penalty(x, stage_params, beta))
                if gradient.size > 0:
                    gradient[:] -= params["penalty_strength"] * np.asarray(
                        _autograd_grad(minimum_length_penalty, 0)(x, stage_params, beta)
                    )
            f_total = f0 - params["penalty_strength"] * penalty

            evaluation_history.append(float(f_total))
            i = len(evaluation_history)
            suffix = (f"  (transmission={f0:.6f} penalty={penalty:.3e})"
                      if params["penalty_strength"] > 0 else "")
            print(f"[{i}/{n_iterations}  {100 * i // n_iterations}%]  beta={beta:g}  "
                  f"J={float(f_total):.6f}{suffix}", flush=True)
            return float(f_total)

        solver = nlopt.opt(nlopt.LD_MMA, n_params)
        solver.set_lower_bounds(0.0)
        solver.set_upper_bounds(1.0)
        solver.set_initial_step(params["init_step"])
        solver.set_max_objective(nlopt_objective)
        solver.set_maxeval(iters_per_stage)
        x_cur = solver.optimize(x_cur)

    final_params = {**params, "beta": beta_schedule[-1]}
    final_weights_continuous = np.array(_mapping_robust(x_cur, final_params, params["eta_i"])).reshape(n, n)
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


# ---------------------------------------------------------------------------
# Adam alternative to NLopt MMA -- the optimizer the Tidy3D topology-
# optimization tutorial uses (Adam, learning rate 1.0). NLopt's MMA has no
# learning rate at all (its closest analogue is `init_step`), so comparing
# "learning rate" against that tutorial requires an actual Adam loop.
#
# `optimization/adam.py` and `adam_nd.py` already exist in this toolkit but
# are NOT reusable here: they are JAX-based, take the gradient via
# jax.value_and_grad rather than accepting one, assume a handful of SCALAR
# parameters (not a 6561-to-32761-element design array), and `adam.py` cannot
# even be imported alongside meep (its import chain asserts meep is absent).
# Only the update rule itself carries over, which is the ten lines below.
# ---------------------------------------------------------------------------
def run_adam_optimization_compact(
    params: dict | None = None, init_weights: np.ndarray | None = None,
) -> OptimizationResult:
    """Same objective, mapping, beta continuation and penalty as
    run_adjoint_optimization_compact, but driven by Adam with an explicit
    `learning_rate` instead of NLopt MMA. Design variables are clipped back
    into [0,1] after every step (MMA enforced those as box constraints)."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    mp.verbosity(0)

    opt, _, _ = build_optimization_problem_compact(params)
    n = params["design_grid_n"]
    n_params = n * n
    beta_schedule = params["beta_schedule"]
    iters_per_stage = params["iters_per_stage"]
    n_iterations = len(beta_schedule) * iters_per_stage
    lr = params["learning_rate"]
    b1, b2, eps = params["adam_beta1"], params["adam_beta2"], params["adam_eps"]

    x_cur = (
        np.asarray(init_weights, dtype=float).flatten()
        if init_weights is not None
        else params["init_density"] * np.ones(n_params)
    )
    m = np.zeros(n_params)
    v = np.zeros(n_params)
    step = 0
    evaluation_history = []

    for beta in beta_schedule:
        stage_params = {**params, "beta": beta}
        for _ in range(iters_per_stage):
            step += 1
            mapped = _mapping_robust(x_cur, stage_params, params["eta_i"])
            with _quiet_meep():
                f0, dJ_du = opt([mapped])
            f0 = np.real(f0[0]) if hasattr(f0, "__len__") else float(np.real(f0))
            gradient = np.asarray(tensor_jacobian_product(_mapping_robust, 0)(
                x_cur, stage_params, params["eta_i"], np.squeeze(dJ_du)
            ))

            penalty = 0.0
            if params["penalty_strength"] > 0:
                penalty = float(minimum_length_penalty(x_cur, stage_params, beta))
                gradient = gradient - params["penalty_strength"] * np.asarray(
                    _autograd_grad(minimum_length_penalty, 0)(x_cur, stage_params, beta)
                )
            f_total = f0 - params["penalty_strength"] * penalty

            # Adam, ASCENDING (this is a maximization), then clip to the box
            # MMA enforced natively.
            m = b1 * m + (1 - b1) * gradient
            v = b2 * v + (1 - b2) * gradient ** 2
            m_hat = m / (1 - b1 ** step)
            v_hat = v / (1 - b2 ** step)
            x_cur = np.clip(x_cur + lr * m_hat / (np.sqrt(v_hat) + eps), 0.0, 1.0)

            evaluation_history.append(float(f_total))
            print(f"[{step}/{n_iterations}  {100 * step // n_iterations}%]  beta={beta:g}  "
                  f"J={f_total:.6f}  (adam lr={lr:g})", flush=True)

    final_params = {**params, "beta": beta_schedule[-1]}
    final_weights_continuous = np.array(_mapping_robust(x_cur, final_params, params["eta_i"])).reshape(n, n)
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


# ---------------------------------------------------------------------------
# Robust adjoint optimization (compact domain): three independent
# OptimizationProblems, same raw design variables, averaged objective/
# gradient each iteration.
# ---------------------------------------------------------------------------
def run_robust_adjoint_optimization_compact(
    params: dict | None = None, init_weights: np.ndarray | None = None,
) -> RobustOptimizationResult:
    """Same NLopt LD_MMA / beta-continuation loop shape as
    run_adjoint_optimization_compact, but every iteration projects the SAME
    raw x at eta_i/eta_e/eta_d, evaluates all three OptimizationProblems
    (from build_robust_optimization_problem_compact), and averages their
    (objective, gradient) pairs -- exact by linearity of differentiation,
    not an approximation -- before handing NLopt a single averaged scalar
    and a single averaged gradient vector. NLopt itself never sees the
    three sub-problems; it only ever sees one objective.

    Requires params["eta_e"]/params["eta_d"] to already be calibrated (see
    find_eta_for_bias) -- raises if either is still None.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    if params["eta_e"] is None or params["eta_d"] is None:
        raise ValueError(
            "params['eta_e']/['eta_d'] must be calibrated (see find_eta_for_bias) "
            "before calling run_robust_adjoint_optimization_compact."
        )
    mp.verbosity(0)

    problems = build_robust_optimization_problem_compact(params)
    opt_n, _, _ = problems["nominal"]
    opt_e, _, _ = problems["eroded"]
    opt_d, _, _ = problems["dilated"]

    n = params["design_grid_n"]
    n_params = n * n
    beta_schedule = params["beta_schedule"]
    iters_per_stage = params["iters_per_stage"]
    n_iterations = len(beta_schedule) * iters_per_stage

    eval_hist_n, eval_hist_e, eval_hist_d, eval_hist_avg = [], [], [], []
    x_cur = (
        np.asarray(init_weights, dtype=float).flatten()
        if init_weights is not None
        else params["init_density"] * np.ones(n_params)
    )

    def _scalar(f0):
        return np.real(f0[0]) if hasattr(f0, "__len__") else float(np.real(f0))

    use_adam = params.get("optimizer", "mma") == "adam"
    adam_m = np.zeros(n_params)
    adam_v = np.zeros(n_params)
    adam_step = 0

    for beta in beta_schedule:
        stage_params = {**params, "beta": beta}

        def nlopt_objective(x, gradient, stage_params=stage_params):
            mapped_n = _mapping_robust(x, stage_params, params["eta_i"])
            mapped_e = _mapping_robust(x, stage_params, params["eta_e"])
            mapped_d = _mapping_robust(x, stage_params, params["eta_d"])

            with _quiet_meep():
                f0_n, dJ_n = opt_n([mapped_n])
                f0_e, dJ_e = opt_e([mapped_e])
                f0_d, dJ_d = opt_d([mapped_d])

            f0_n, f0_e, f0_d = _scalar(f0_n), _scalar(f0_e), _scalar(f0_d)
            f_avg = (f0_n + f0_e + f0_d) / 3.0

            if gradient.size > 0:
                g_n = tensor_jacobian_product(_mapping_robust, 0)(
                    x, stage_params, params["eta_i"], np.squeeze(dJ_n)
                )
                g_e = tensor_jacobian_product(_mapping_robust, 0)(
                    x, stage_params, params["eta_e"], np.squeeze(dJ_e)
                )
                g_d = tensor_jacobian_product(_mapping_robust, 0)(
                    x, stage_params, params["eta_d"], np.squeeze(dJ_d)
                )
                gradient[:] = (g_n + g_e + g_d) / 3.0

            eval_hist_n.append(f0_n)
            eval_hist_e.append(f0_e)
            eval_hist_d.append(f0_d)
            eval_hist_avg.append(f_avg)
            i = len(eval_hist_avg)
            print(
                f"[{i}/{n_iterations}  {100 * i // n_iterations}%]  beta={beta:g}  "
                f"J_avg={f_avg:.6f}  (nominal={f0_n:.6f} eroded={f0_e:.6f} dilated={f0_d:.6f})",
                flush=True,
            )
            return f_avg

        if use_adam:
            # Same averaged objective/gradient, driven by Adam instead of MMA
            # (see run_adam_optimization_compact for why this module carries
            # its own Adam). nlopt_objective fills `gradient` in place, so it
            # is reused verbatim -- the two optimizers differ only in what they
            # do with that gradient.
            lr = params["learning_rate"]
            b1, b2, eps = params["adam_beta1"], params["adam_beta2"], params["adam_eps"]
            for _ in range(iters_per_stage):
                adam_step += 1
                gradient = np.zeros(n_params)
                nlopt_objective(x_cur, gradient)
                adam_m = b1 * adam_m + (1 - b1) * gradient
                adam_v = b2 * adam_v + (1 - b2) * gradient ** 2
                m_hat = adam_m / (1 - b1 ** adam_step)
                v_hat = adam_v / (1 - b2 ** adam_step)
                x_cur = np.clip(x_cur + lr * m_hat / (np.sqrt(v_hat) + eps), 0.0, 1.0)
        else:
            solver = nlopt.opt(nlopt.LD_MMA, n_params)
            solver.set_lower_bounds(0.0)
            solver.set_upper_bounds(1.0)
            solver.set_initial_step(params["init_step"])
            solver.set_max_objective(nlopt_objective)
            solver.set_maxeval(iters_per_stage)
            x_cur = solver.optimize(x_cur)

    final_params = {**params, "beta": beta_schedule[-1]}
    final_weights_continuous = {
        "nominal": np.array(_mapping_robust(x_cur, final_params, params["eta_i"])).reshape(n, n),
        "eroded": np.array(_mapping_robust(x_cur, final_params, params["eta_e"])).reshape(n, n),
        "dilated": np.array(_mapping_robust(x_cur, final_params, params["eta_d"])).reshape(n, n),
    }
    final_weights_binarized = {
        k: np.where(v >= 0.5, 1.0, 0.0) for k, v in final_weights_continuous.items()
    }

    sim_params = {
        **params,
        "meep_version": mp.__version__,
        "python_version": platform.python_version(),
        "creation_date": date.today().isoformat(),
    }

    return RobustOptimizationResult(
        evaluation_history_nominal=np.array(eval_hist_n),
        evaluation_history_eroded=np.array(eval_hist_e),
        evaluation_history_dilated=np.array(eval_hist_d),
        evaluation_history_avg=np.array(eval_hist_avg),
        x_opt=x_cur,
        final_weights_continuous=final_weights_continuous,
        final_weights_binarized=final_weights_binarized,
        sim_params=sim_params,
    )


# ---------------------------------------------------------------------------
# Kept for backward compatibility / external callers with an already-
# binarized frozen mask on a DIFFERENT (coarser) grid than this module's own
# design_grid_n -- not used by the current notebook flow (the new baseline
# is optimized directly on this module's own grid, so its final_weights_*
# can be re-projected at eta_e/eta_d directly via _mapping_robust, no
# upsampling needed), but still correct and potentially useful standalone.
# ---------------------------------------------------------------------------
def apply_bias_to_frozen_mask(
    weights_binary: np.ndarray, params: dict, eta: float, beta: float = 64.0,
    upsampled_grid_n: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Bilinearly upsamples the already-binarized `weights_binary` (any
    shape/grid) onto a fine `upsampled_grid_n`x`upsampled_grid_n` grid
    (default: params["design_grid_n"]), THEN re-filters/re-projects that
    upsampled continuous mask through _mapping_robust(..., eta) at a sharp
    beta. Upsampling first is required whenever the mask's own native grid
    is too coarse for conic_filter's transition zone to be well-resolved
    (see design_grid_n's own comment in DEFAULT_PARAMS for the mechanism).

    Valid wherever the mask's boundary curvature radius is much larger than
    filter_radius_um; potentially less faithful right at a concave corner or
    already-thin feature (visually inspect the result before trusting
    downstream transmission numbers).

    Returns (biased_fine_mask, fine_params) -- fine_params is `params` with
    `design_grid_n` overridden to `upsampled_grid_n`, which MUST be passed
    to simulate_baseline_compact/get_permittivity_map_compact alongside
    biased_fine_mask. Does NOT re-threshold to hard 0/1.
    """
    n_fine = upsampled_grid_n or params["design_grid_n"]
    n_coarse = weights_binary.shape[0]

    coarse_axis = np.linspace(0.0, 1.0, n_coarse)
    interp = RegularGridInterpolator(
        (coarse_axis, coarse_axis), np.asarray(weights_binary, dtype=float), method="linear"
    )
    fine_axis = np.linspace(0.0, 1.0, n_fine)
    fx, fy = np.meshgrid(fine_axis, fine_axis, indexing="ij")
    upsampled = interp(np.stack([fx.ravel(), fy.ravel()], axis=-1)).reshape(n_fine, n_fine)

    fine_params = {**params, "design_grid_n": n_fine, "beta": beta}
    mapped = _mapping_robust(upsampled.flatten(), fine_params, eta)
    return np.array(mapped).reshape(n_fine, n_fine), fine_params


# ---------------------------------------------------------------------------
# Raster (pixel-density) -> GDS. No existing path in this toolkit goes this
# direction: `gds_import.py` converts GDS -> meep prisms, and every other
# `meep_sim` module's `build_gf_component()` starts from parametric
# gdsfactory primitives (rectangles, bends, bezier curves), never a pixel
# array, so a topology-optimized design has never needed a GDS export before.
# Built fresh here on `skimage.measure.find_contours` (marching squares).
# ---------------------------------------------------------------------------
def _contour_signed_area(contour: np.ndarray) -> float:
    """Shoelace formula on a closed (row, col) contour from
    skimage.measure.find_contours. Verified directly on a synthetic solid
    annulus (a filled ring with a hole in the middle), not assumed from the
    docs: `find_contours` returns the OUTER solid boundary with NEGATIVE
    signed area under this formula, and the boundary of an ENCLOSED HOLE
    with POSITIVE signed area."""
    r, c = contour[:, 0], contour[:, 1]
    return 0.5 * float(np.sum(r[:-1] * c[1:] - r[1:] * c[:-1]) + (r[-1] * c[0] - r[0] * c[-1]))


def _rasterize_full_device(
    weights_binary: np.ndarray, params: dict, dom: dict, margin_um: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize the WHOLE compact-domain device -- straight input/output
    arms (matching build_geometry_compact's own in_wg/out_wg mp.Block
    definitions) PLUS the design region's own binarized density -- onto one
    combined pixel grid at the design grid's own pitch, extended by
    `margin_um` of cladding on every side.

    Required, not cosmetic: tracing the design-region density in isolation
    breaks for a genuinely freeform design, because its material can reach
    ALL FOUR edges of the design region, not just the two "arm" edges
    (measured directly on the final robust design: solid pixels present on
    every one of its four border rows/columns). A solid region touching the
    array's own border does not close into a simple marching-squares contour
    loop, which silently corrupted the solid/hole split (measured: 5 solid +
    5 "hole" contours recovered where the true design has 6 solid components
    and exactly 1 enclosed hole). Embedding the density in the full device
    (arms extending well past the design region, then a cladding margin
    beyond THAT) guarantees every solid region is fully surrounded by
    cladding somewhere in the canvas, so every contour closes correctly.
    """
    n = params["design_grid_n"]
    Lx, Ly = params["design_region_x_um"], params["design_region_y_um"]
    pitch = Lx / (n - 1)
    Sx = Sy = dom["cell_x_um"]
    x0, y0 = dom["x0"], dom["y0"]
    w = params["wg_width_um"]

    half_extent = Sx / 2 + margin_um
    n_canvas = int(round(2 * half_extent / pitch)) + 1
    grid = np.linspace(-half_extent, half_extent, n_canvas)
    gx, gy = np.meshgrid(grid, grid, indexing="ij")

    canvas = np.zeros((n_canvas, n_canvas))
    in_x1, in_x2 = -Sx / 2, x0 + Lx / 2
    canvas[(gx >= in_x1) & (gx <= in_x2) & (np.abs(gy - y0) <= w / 2)] = 1.0
    out_y1, out_y2 = y0 - Ly / 2, Sy / 2
    canvas[(np.abs(gx - x0) <= w / 2) & (gy >= out_y1) & (gy <= out_y2)] = 1.0

    design_x = np.linspace(x0 - Lx / 2, x0 + Lx / 2, n)
    design_y = np.linspace(y0 - Ly / 2, y0 + Ly / 2, n)
    interp = RegularGridInterpolator(
        (design_x, design_y), np.asarray(weights_binary, dtype=float),
        method="nearest", bounds_error=False, fill_value=0.0,
    )
    in_region = (np.abs(gx - x0) <= Lx / 2 + 1e-9) & (np.abs(gy - y0) <= Ly / 2 + 1e-9)
    canvas[in_region] = interp(np.stack([gx[in_region], gy[in_region]], axis=-1))

    return canvas, grid


def density_to_gds_component(
    weights_binary: np.ndarray, params: dict, dom: dict,
    layer: tuple = (1, 0), margin_um: float = 0.3,
):
    """Rasterize the full device (`_rasterize_full_device` -- arms + design
    region density + cladding margin) and trace its silicon/cladding
    boundary (marching squares, `skimage.measure.find_contours` at level
    0.5) into a `gdsfactory.Component` on `layer`, in the same physical (um)
    coordinate frame `_domain_compact` uses (`dom` = its own return value).

    Handles enclosed holes correctly rather than filling them in: `find_
    contours` returns one closed loop per solid-void boundary, INCLUDING the
    boundary of any hole enclosed within a solid region, and the two wind in
    opposite directions (signed area, `_contour_signed_area`, is NEGATIVE for
    an outer solid boundary and POSITIVE for a hole here -- verified directly
    against a synthetic solid-annulus test pattern, not assumed). Outer
    boundaries are added as polygons on `layer`; if any hole boundaries
    exist, their own polygons are unioned and then removed from the solid
    via `gf.boolean` ("not"), rather than silently rendering the hole as
    solid material.
    """
    import gdsfactory as gf
    from skimage import measure

    canvas, grid = _rasterize_full_device(weights_binary, params, dom, margin_um=margin_um)
    n_canvas = len(grid)
    pitch = grid[1] - grid[0]

    def to_um(contour: np.ndarray) -> list[tuple[float, float]]:
        # contour columns are (row, col) in array-INDEX space; this canvas is
        # built with np.meshgrid(..., indexing="ij"), so axis 0 (row) is the
        # x-direction and axis 1 (col) is the y-direction.
        row, col = contour[:, 0], contour[:, 1]
        xs = grid[0] + row * pitch
        ys = grid[0] + col * pitch
        return list(zip(xs.tolist(), ys.tolist()))

    contours = measure.find_contours(canvas, level=0.5)
    solids = [c for c in contours if _contour_signed_area(c) < 0 and len(c) >= 3]
    holes = [c for c in contours if _contour_signed_area(c) > 0 and len(c) >= 3]

    solid_component = gf.Component()
    for c in solids:
        solid_component.add_polygon(to_um(c), layer=layer)

    if not holes:
        return solid_component

    hole_component = gf.Component()
    for c in holes:
        hole_component.add_polygon(to_um(c), layer=layer)
    return gf.boolean(solid_component, hole_component, operation="not", layer=layer)
