"""Re-optimization of the strip -> TE-wall transition (notebook 03, Stage C, 2026-11).

Same objective, design region, filter and validation representation as
scripts/topopt_transition.py (which reproduces the cached first designs), with what the
convergence study changed:
  * resolution and rows per side are arguments (spin TE: 10 rows lost 15 % into the bulk and
    the band average moved 3.6 pt between 48 and 64/a0, so the first spin design was tuned on
    a model that was too small);
  * the valley procedure for both walls: MaterialGrid with subpixel averaging, beta 8 -> 16 ->
    32 -> 64 -> inf, decay_by 1e-9 (accurate adjoint gradients), MMA initial step 0.05 at inf;
  * strip lateral offset y_strip (sweep: spin butt coupling 7.8 % at 0 -> 29.7 % at -0.3 a0);
  * band and seed as arguments; gray + smooth noise start;
  * MPI-safe: only the master writes files and prints;
  * each beta stage stops early once it has saturated (--patience: the stage's best J has not
    risen by >= --tol relative for that many iterations, after at least --min-stage iterations);
    the --iters numbers are then upper limits. In the first valley run (72 iterations) the stages
    saturated after ~6 (beta 8, 16), 1-2 (32, 64) and ~10 (inf) iterations;
  * --lengthscale-nm L: minimum length L for solid and void, imposed by the geometric constraints
    of Zhou et al. (CMAME 293, 266 (2015); mpa.constraint_solid / constraint_void) as MMA
    inequality constraints in the stages with beta >= --constraint-beta. The filter radius is set
    from L (eta_e 0.75), c = R^4 res^2 (Zhou's c = R^4 in pixels, converted to the gradient per
    lattice constant), and each constraint is log10(g / EPS_G) <= 0 (EPS_G 1e-8: an 80 nm line or
    gap gives g <= 1e-12, a 60 nm one ~3e-5). With --resume the start design is first projected
    onto the constraints (the design closest to the resumed one's binary level set, no FDTD), and
    only designs with g <= 10 EPS_G count as the best. The first constrained stage also starts from
    that projection when its start is infeasible, and an infeasible iteration does not count
    towards --patience;
  * --domain-rules: the domain sized in wavelengths (CLAUDE.md section 7; single-frequency check
    scripts/check_domain.py, 2026-11-07). With lambda0 = 1/fcen: every PML lambda0/2 thick, the
    right PML lambda0/2 behind the objective plane, the source lambda0 before the crystal face and
    lambda0/2 from the left PML, no y pad (the crystal ends at the PML, which lies in Si), and the
    right PML --obj-pml-lam (default 2) lambda0 behind the objective plane: the junction's near
    field reaches several lambda0 down the wall, and a PML inside it absorbs part of what would
    return, so T rises (s1r at 0.237, 20 rows, x_t 9: PML lambda0/2 behind 0.912, 1.5 lambda0 0.859,
    x = 25 0.849; check_domain.py xr_*). The default model (PML 1 a behind the objective plane, source 0.5 a from the PML) read the
    valley-Hall transitions 0.87 / 0.52 against 0.77 / 0.16 in the coupling runs;
  * --eval-only (with --resume): evaluate the resumed design (beta = inf level set) and stop;
  * --noise A: amplitude of the smooth noise on the gray start (0: uniform gray 0.5);
  * --robust: robust formulation for the minimum length scale (Wang, Lazarov & Sigmund, Struct.
    Multidisc. Optim. 43, 767 (2011)): every iteration evaluates the eroded, nominal and dilated
    designs (MaterialGrid threshold eta = ETA_E, 0.5, 1 - ETA_E on the same filtered field, filter
    radius from --lengthscale-nm) and maximizes their soft minimum
    -(1/k) log(mean(exp(-k J_i))) (--softmin-k). Features below the length scale vanish or merge in
    the eroded / dilated designs, so they are avoided while the topology forms; the Zhou constraints
    are not imposed (only reported). The three length-scale-constrained runs before this
    (gm36_ls80r, c24_ls80_s1/s1c, c24_ls80_s2) projected an unconstrained (gray) design onto the
    constraints and lost it (J 0.79-0.93 gray -> 0.03-0.06 feasible);
  * --resume across resolutions: the resumed design variables are interpolated onto this run's
    design grid (coarse-to-fine: a 24/a design as the start of a 36/a run).

Usage (from topological/, mp-mpi env):
    mpirun -np 12 python scripts/topopt_transition_v2.py NAME TAG [--res R] [--rows N]
        [--y-strip Y] [--band LO HI NF] [--seed S] [--iters N8 N16 N32 N64 NINF]
    e.g. spin  s32n15_y03   --res 32 --rows 15 --y-strip -0.3
         valley seed2       --seed 2
Writes data/topopt/v2_{NAME}_{TAG}_history.npz and v2_{NAME}_{TAG}_design.npz.
"""
import argparse
import time
from pathlib import Path

import autograd.numpy as npa
import meep as mp
import meep.adjoint as mpa
import nlopt
import numpy as np
from autograd import tensor_jacobian_product
from scipy.ndimage import gaussian_filter

from pic_toolkit.meep_sim import topological_pc as tp

mp.verbosity(0)
DATA = Path(__file__).resolve().parents[1] / "data" / "topopt"

SETUP = {
    "spin": dict(kind="spin_te", res=32, n_rows=15, band=(0.4045, 0.4075, 3), fcen=0.406, fwidth=0.03,
                 a_nm=tp.A0_NM, y_wall=3 * tp.ROW_H),
    "valley": dict(kind="valley_te", res=24, n_rows=10, band=(0.232, 0.2415, 5), fcen=0.2368, fwidth=0.03,
                   a_nm=tp.A_NM, y_wall=4.5),
}
DES_C, DES_S = (2.0, 0.0), (4.0, 4.0)
X_LEFT, X_PML_R, X_T, DPML_X, DPML_RIGHT, DPML_Y, PAD_Y = -3.5, 13.0, 12.0, 1.5, 2.5, 1.5, 0.5
X_SRC, X_INC = -1.5, -0.8
STAGES = [(8, 12), (16, 12), (32, 12), (64, 12), (np.inf, 24)]
ETA_E, EPS_G, BETA_G = 0.75, 1e-8, 64  # length-scale constraints: erosion threshold, bound, projection beta at inf


def apply_domain_rules(c: dict, obj_pml_lam: float = 2.0) -> None:
    """Domain in wavelengths (see the docstring); X_T must already be set. obj_pml_lam: objective
    plane -> right PML in lambda0."""
    global X_LEFT, X_PML_R, DPML_X, DPML_RIGHT, DPML_Y, PAD_Y, X_SRC, X_INC
    half = 0.5 / c["fcen"]  # lambda0 / 2 (a units)
    X_SRC, X_INC = -2 * half, -half
    X_LEFT, DPML_X = X_SRC - 2 * half, half
    X_PML_R, DPML_RIGHT, DPML_Y, PAD_Y = X_T + 2 * half * obj_pml_lam, half, half, 0.0


def log(*a):
    if mp.am_master():
        print(*a, flush=True)


def problem(c: dict, res: int, n_rows: int, y_strip: float, freqs: np.ndarray, full: bool = False,
            decay: float = 1e-9, min_run: float = 400, min_nm: float = 100.0):
    sx, cx = X_PML_R + DPML_RIGHT - X_LEFT, (X_PML_R + DPML_RIGHT + X_LEFT) / 2
    sy = 2 * (n_rows * tp.ROW_H + PAD_Y + DPML_Y)
    w = tp.strip_width(c["kind"])
    nx, ny = int(round(DES_S[0] * res)) + 1, int(round(DES_S[1] * res)) + 1
    bounds = [mp.PML(DPML_Y, direction=mp.Y), mp.PML(DPML_X, direction=mp.X, side=mp.Low),
              mp.PML(DPML_RIGHT, direction=mp.X, side=mp.High)]
    src = [mp.EigenModeSource(mp.GaussianSource(c["fcen"], fwidth=c["fwidth"]), center=mp.Vector3(X_SRC, 0),
                              size=mp.Vector3(0, sy - 2 * DPML_Y), eig_band=1, eig_parity=mp.TE)]
    # incident power: the same strip, straight through the cell
    ref = mp.Simulation(cell_size=mp.Vector3(sx, sy), geometry_center=mp.Vector3(cx, 0),
                        geometry=[mp.Block(size=mp.Vector3(mp.inf, w), center=mp.Vector3(0, y_strip), material=tp.HOST)],
                        sources=src, boundary_layers=bounds, resolution=res)
    fl = ref.add_flux(freqs, mp.FluxRegion(center=mp.Vector3(X_INC, 0), size=mp.Vector3(0, sy - 2 * DPML_Y)))
    ref.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-9, minimum_run_time=50))
    p_inc = np.asarray(mp.get_fluxes(fl))
    dv = mp.MaterialGrid(mp.Vector3(nx, ny), mp.air, tp.HOST, grid_type="U_MEAN", do_averaging=True, beta=8, eta=0.5)
    region = mpa.DesignRegion(dv, volume=mp.Volume(center=mp.Vector3(*DES_C), size=mp.Vector3(*DES_S)))
    geom = tp.coupler_geometry(c["kind"], w, X_LEFT, X_PML_R + DPML_RIGHT, n_rows, y_strip=y_strip)
    geom.append(mp.Block(center=region.center, size=region.size, material=dv))
    sim = mp.Simulation(cell_size=mp.Vector3(sx, sy), geometry_center=mp.Vector3(cx, 0), geometry=geom,
                        sources=src, boundary_layers=bounds, resolution=res)
    # objective line: the edge channel |y| < y_wall, or (full) the whole height inside the y PML --
    # inside the shared gap only the edge mode carries power forward, and the fixed window holds
    # only 63 % (spin) / 87 % (valley) of the edge mode's own flux (notebook 02 wall runs)
    h = sy - 2 * DPML_Y - 0.4 if full else 2 * c["y_wall"]
    line = mp.Volume(center=mp.Vector3(X_T, 0), size=mp.Vector3(0, h))
    mons = [mpa.FourierFields(sim, line, mp.Ey), mpa.FourierFields(sim, line, mp.Hz)]

    def J(ey, hz):
        n = min(ey.shape[1], hz.shape[1])
        return npa.mean(npa.real(npa.sum(npa.conj(ey[:, :n]) * hz[:, :n], axis=1)) / res / p_inc)

    opt = mpa.OptimizationProblem(simulation=sim, objective_functions=[J], objective_arguments=mons,
                                  design_regions=[region], frequencies=list(freqs), decay_by=decay,
                                  minimum_run_time=min_run, maximum_run_time=3000)
    radius = mpa.get_conic_radius_from_eta_e(min_nm / c["a_nm"], ETA_E)
    filt = lambda x: mpa.conic_filter(x.reshape(nx, ny), radius, DES_S[0], DES_S[1], res)
    return opt, dv, filt, (nx, ny), radius, w


def lengthscale_constraints(filt, radius: float, res: int):
    """g_solid(x, beta), g_void(x, beta) of Zhou et al. 2015 and their autograd gradients."""
    from autograd import grad
    c0 = (radius * res) ** 4 / res ** 2
    thr = lambda b: (lambda a: mpa.tanh_projection(a, min(b, BETA_G), 0.5))
    gs = lambda x, b: mpa.constraint_solid(x, c0, ETA_E, filt, thr(b), res)
    gv = lambda x, b: mpa.constraint_void(x, c0, 1 - ETA_E, filt, thr(b), res)
    return [(gs, grad(gs)), (gv, grad(gv))]


def log_constraint(g, dg, beta):
    """nlopt constraint log10(g / EPS_G) <= 0 (well scaled for MMA from far outside the bound)."""
    def cf(v, gr):
        val = float(g(v, beta))
        if gr.size > 0:
            gr[:] = dg(v, beta) / ((val + 1e-30) * np.log(10))
        return np.log10(val + 1e-30) - np.log10(EPS_G)
    return cf


def project_feasible(x0, target, filt, cons, n_iter: int = 150):
    """The design whose projected density is closest to `target` under the length-scale constraints
    (MMA, no FDTD; deterministic, so every MPI rank gets the same result)."""
    from autograd import grad
    obj = lambda v: npa.mean((mpa.tanh_projection(filt(v), BETA_G, 0.5).flatten() - target) ** 2)
    gobj = grad(obj)

    def f(v, gr):
        if gr.size > 0:
            gr[:] = -gobj(v)
        return -float(obj(v))

    s = nlopt.opt(nlopt.LD_MMA, x0.size)
    s.set_lower_bounds(0); s.set_upper_bounds(1); s.set_max_objective(f); s.set_maxeval(n_iter)
    for g, dg in cons:
        s.add_inequality_constraint(log_constraint(g, dg, BETA_G), 0.0)
    return s.optimize(x0.copy())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=list(SETUP))
    ap.add_argument("tag")
    ap.add_argument("--res", type=int)
    ap.add_argument("--rows", type=int)
    ap.add_argument("--y-strip", type=float, default=0.0)
    ap.add_argument("--band", type=float, nargs=3, metavar=("LO", "HI", "NF"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gm", action="store_true", help="gap-matched pair (tp.set_gap_matched)")
    ap.add_argument("--region-um", type=float, help="square design region of this side (um) from the face")
    ap.add_argument("--xt", type=float, help="objective plane x (lattice units); PML starts 1 later")
    ap.add_argument("--full", action="store_true", help="objective over the full height (all forward power)")
    ap.add_argument("--domain-rules", action="store_true", help="domain sized in wavelengths (see docstring)")
    ap.add_argument("--obj-pml-lam", type=float, default=2.0, help="with --domain-rules: objective -> right PML (lambda0)")
    ap.add_argument("--robust", action="store_true", help="eroded / nominal / dilated soft-min objective")
    ap.add_argument("--softmin-k", type=float, default=30.0)
    ap.add_argument("--noise", type=float, default=0.15, help="smooth-noise amplitude on the gray start (0: uniform)")
    ap.add_argument("--eval-only", action="store_true", help="with --resume: evaluate that design and stop")
    ap.add_argument("--decay", type=float, default=1e-9)
    ap.add_argument("--min-run", type=float, default=400)
    ap.add_argument("--resume", metavar="TAG",
                    help="start from the design of v2_NAME_TAG (its best beta=inf design, else its last); "
                         "use with --iters 0 0 0 0 N for a few more beta=inf iterations")
    ap.add_argument("--patience", type=int, default=3, help="0: always run --iters per stage")
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--min-stage", type=int, default=3)
    ap.add_argument("--lengthscale-nm", type=float,
                    help="minimum solid and void length (nm): filter from it + Zhou length-scale constraints")
    ap.add_argument("--constraint-beta", type=float, default=32, help="constraints in stages with beta >= this")
    ap.add_argument("--iters", type=int, nargs=5, metavar="N",
                    help="iterations at beta 8, 16, 32, 64, inf (default 12 12 12 12 24)")
    a = ap.parse_args()
    stages = [(b, n) for (b, _), n in zip(STAGES, a.iters)] if a.iters else STAGES
    c = SETUP[a.name]
    if a.gm:
        tp.set_gap_matched()
    if a.region_um:
        L = a.region_um * 1000 / c["a_nm"]
        DES_C, DES_S = (L / 2, 0.0), (L, L)
    if a.xt:
        X_T, X_PML_R = a.xt, a.xt + 1.0
    if a.domain_rules:
        apply_domain_rules(c, a.obj_pml_lam)
    res, rows = a.res or c["res"], a.rows or c["n_rows"]
    lo, hi, nf = a.band or c["band"]
    freqs = np.linspace(lo, hi, int(nf))
    stem = DATA / f"v2_{a.name}_{a.tag}"
    log(f"{a.name} {a.tag}: res {res}, rows {rows}, y_strip {a.y_strip}, freqs {np.round(freqs, 4)}, seed {a.seed}")

    log(f"  gm {a.gm}, design region {DES_S[0]:.2f} x {DES_S[1]:.2f} at {DES_C}, x_t {X_T}, full {a.full}, "
        f"decay {a.decay}, min_run {a.min_run}")
    log(f"  domain: source {X_SRC:.2f}, left PML {X_LEFT:.2f}+{DPML_X:.2f}, right PML {X_PML_R:.2f}+{DPML_RIGHT:.2f}, "
        f"y PML {DPML_Y:.2f}, pad {PAD_Y:.2f} (a units)")
    min_nm = a.lengthscale_nm or 100.0
    opt, dv, filt, (nx, ny), radius, w = problem(c, res, rows, a.y_strip, freqs, a.full, a.decay, a.min_run, min_nm)
    cons = lengthscale_constraints(filt, radius, res) if a.lengthscale_nm and not a.robust else []
    gchk = lengthscale_constraints(filt, radius, res) if a.robust else cons  # reported only (robust)
    etas = (ETA_E, 0.5, 1 - ETA_E) if a.robust else (0.5,)  # eroded, nominal, dilated
    if a.robust:
        log(f"  robust: eta {etas}, filter radius {radius:.4f} a ({min_nm:.0f} nm), soft-min k {a.softmin_k}")
    feasible = lambda v, b: all(g(v, b) <= 10 * EPS_G for g, _ in cons)
    # gray + smooth noise (identical on every rank: same seed); breaks the valley wall's mirror symmetry
    noise = gaussian_filter(np.random.default_rng(a.seed).standard_normal((nx, ny)), 3)
    x = np.clip(0.5 + a.noise * noise / np.abs(noise).max(), 0, 1).flatten()
    if a.resume:
        d0 = np.load(DATA / f"v2_{a.name}_{a.resume}_design.npz")
        x = np.asarray(d0["x"], dtype=float)
        n0 = tuple(int(round(L * int(d0["res"]))) + 1 for L in d0["size"])
        if n0 != (nx, ny):  # another resolution: interpolate onto this design grid
            from scipy.interpolate import RegularGridInterpolator
            ip = RegularGridInterpolator((np.linspace(0, 1, n0[0]), np.linspace(0, 1, n0[1])), x.reshape(n0))
            gx, gy = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing="ij")
            x = ip(np.stack([gx, gy], -1))
            log(f"  design variables interpolated from {n0[0]} x {n0[1]} to {nx} x {ny}")
        x = x.flatten()
        log(f"  resumed from v2_{a.name}_{a.resume} (its final binary J {float(d0['j_final']):.4f})")
        if a.eval_only:
            dv.beta = np.inf
            j = float(np.real(opt([np.asarray(d0["rho"]).flatten()], need_gradient=False)[0]))
            log(f"  {a.resume} in this model: J = {j:.4f} (level set)")
            raise SystemExit
        if cons:
            # target: the resumed binary level set (re-filtered on this grid when interpolated)
            rho0 = np.asarray(d0["rho"]) if n0 == (nx, ny) else np.asarray(filt(x))
            x = project_feasible(x, (rho0 > 0.5).astype(float).flatten(), filt, cons)
            log(f"  projected onto the {min_nm:.0f} nm constraints: g_solid {cons[0][0](x, np.inf):.1e}, "
                f"g_void {cons[1][0](x, np.inf):.1e}")
    mapping = lambda v, b: filt(v).flatten()
    hist = {"J": [], "beta": [], "t": []}
    best = [-np.inf, x.copy()]
    t0 = time.time()
    for beta, nit in stages:
        if nit == 0:  # (nlopt reads maxeval 0 as unlimited)
            continue
        if cons and beta >= a.constraint_beta and not feasible(x, beta):
            # first constrained stage: start from a feasible design. Otherwise MMA does not reach
            # g <= 10 EPS_G within a stage (c24_ls80_s1: g ~2e-2 for 9 iterations), no design of the
            # stage counts, and every constrained stage restarts from the unconstrained design.
            x = project_feasible(x, (np.asarray(filt(x)) > 0.5).astype(float).flatten(), filt, cons)
            log(f"  beta {beta}: projected onto the {min_nm:.0f} nm constraints: g_solid {cons[0][0](x, beta):.1e}, "
                f"g_void {cons[1][0](x, beta):.1e}")
        stage = {"best": -np.inf, "stall": 0, "n": 0, "v": x.copy(), "vbest": -np.inf}

        def f(v, grad, beta=beta, stage=stage):
            dv.beta = beta
            vals, grads = [], []
            for eta in etas:
                dv.eta = eta
                f0, dJ = opt([mapping(v, beta)])
                dJ = np.asarray(dJ)
                dJ = dJ.sum(axis=-1) if dJ.ndim > 1 else dJ
                vals.append(float(np.real(f0)))
                if grad.size > 0:
                    grads.append(np.asarray(tensor_jacobian_product(mapping, 0)(v, beta, dJ)))
            dv.eta = 0.5
            vals_a = np.asarray(vals)
            e = np.exp(-a.softmin_k * (vals_a - vals_a.min()))
            val = float(vals_a.min() - np.log(e.mean()) / a.softmin_k)  # soft minimum (= J if not robust)
            if grad.size > 0:
                grad[:] = sum(wi * gi for wi, gi in zip(e / e.sum(), grads))
            hist.setdefault("J_each", []).append(vals)
            gvals = [float(g(v, beta)) for g, _ in cons]
            if beta == np.inf and val > best[0] and (not cons or feasible(v, beta)):
                best[0], best[1] = val, v.copy()
            for k, vv in (("J", val), ("beta", beta), ("t", time.time() - t0)):
                hist[k].append(vv)
            if mp.am_master():
                np.savez(f"{stem}_history.npz", **hist, x=v, best_x=best[1], freqs=freqs, res=res, n_rows=rows,
                         y_strip=a.y_strip, seed=a.seed, gm=a.gm, full=a.full, des_c=DES_C, des_s=DES_S, x_t=X_T,
                         domain_rules=a.domain_rules)
            log(f"iter {len(hist['J'])} beta {beta} J {val:.4f}"
                + (" (eroded / nominal / dilated " + " / ".join(f"{q:.4f}" for q in vals) + ")" if a.robust else "")
                + (f" g_solid {gvals[0]:.1e} g_void {gvals[1]:.1e}" if cons else "")
                + f" ({(time.time() - t0) / 60:.0f} min)")
            stage["n"] += 1
            if val > stage["vbest"] and (not cons or beta < a.constraint_beta or feasible(v, beta)):
                stage["v"], stage["vbest"] = v.copy(), val  # the stage's best design: the next stage starts from it
            if val > stage["best"] * (1 + a.tol) if stage["best"] > 0 else val > stage["best"]:
                stage["best"], stage["stall"] = max(val, stage["best"]), 0
            else:
                stage["best"], stage["stall"] = max(val, stage["best"]), stage["stall"] + 1
            if cons and beta >= a.constraint_beta and not feasible(v, beta):
                stage["stall"] = 0  # an infeasible design does not count as saturation
            if a.patience and stage["n"] >= a.min_stage and stage["stall"] >= a.patience:
                log(f"  beta {beta}: saturated after {stage['n']} iterations (stage best J {stage['best']:.4f})")
                raise nlopt.ForcedStop
            return val

        s = nlopt.opt(nlopt.LD_MMA, x.size)
        s.set_lower_bounds(0); s.set_upper_bounds(1); s.set_max_objective(f); s.set_maxeval(nit)
        if beta == np.inf:
            s.set_initial_step(0.05)
        if beta >= a.constraint_beta:
            for g, dg in cons:
                s.add_inequality_constraint(log_constraint(g, dg, beta), 0.0)
        try:
            x[:] = s.optimize(x)
        except nlopt.ForcedStop:
            x[:] = stage["v"]  # as nlopt does on a normal finish: continue from the stage's best design
        if cons and beta >= a.constraint_beta:
            x[:] = stage["v"]  # the best design that satisfies the length-scale constraints

    xf = best[1] if np.isfinite(best[0]) else x
    # the binary design is the level set rho = 0.5 of the filtered density, evaluated at beta = inf
    # with subpixel smoothing (boundary below the pixel; what the beta = inf iterations optimized).
    # The per-pixel threshold w_bin (a staircase of the design grid) is evaluated for comparison.
    rho = np.asarray(filt(xf))
    w_bin = (rho > 0.5).astype(float)
    dv.beta = np.inf
    j_rob = []
    for eta in etas:  # (robust) the eroded and dilated designs of the final one; the nominal last
        dv.eta = eta
        j_rob.append(float(np.real(opt([rho.flatten()], need_gradient=False)[0])))
    dv.eta = 0.5
    j_bin = j_rob[len(etas) // 2]
    j_px = float(np.real(opt([w_bin.flatten()], need_gradient=False)[0]))
    log(f"final binary design: J = {j_bin:.4f} (level set), {j_px:.4f} (pixel threshold)"
        + (f"; eroded {j_rob[0]:.4f}, dilated {j_rob[2]:.4f}" if a.robust else "")
        + (f"; g_solid {gchk[0][0](xf, np.inf):.1e}, g_void {gchk[1][0](xf, np.inf):.1e}" if gchk else ""))
    if mp.am_master():
        np.savez(f"{stem}_design.npz", x=xf, rho=rho, weights=w_bin, j_final=j_bin, j_pixel=j_px, center=DES_C, size=DES_S, res=res,
                 n_rows=rows, freqs=freqs, filter_radius=radius, min_len=min_nm / c["a_nm"], w=w,
                 lengthscale_nm=a.lengthscale_nm or 0.0, g_solid=gchk[0][0](xf, np.inf) if gchk else -1.0,
                 g_void=gchk[1][0](xf, np.inf) if gchk else -1.0, robust=a.robust, j_robust=j_rob,
                 y_strip=a.y_strip, seed=a.seed, gm=a.gm, full=a.full, x_t=X_T, domain_rules=a.domain_rules,
                 noise=a.noise)
