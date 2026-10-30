"""Adjoint topology optimization of the strip -> TE-wall transition (notebook 03).

The cached results in data/topopt/ were produced with these settings; re-running takes
several hours per design (one forward + adjoint FDTD per iteration).

Common: reduced model (crystal to x = 13, PML beyond, strip from the left), design region
x in [0, 4], y in [-2, 2] in lattice units, MaterialGrid between air and eps_d, conic
filter for a 100 nm minimum length scale (spin: a0 = 632 nm; valley: a = 367 nm),
objective = mean over the listed frequencies of the flux through the edge-channel line
normalized to the incident strip-mode power (Meep flux convention, Re(E* x H)), nlopt MMA.

spin   resolution 24/a0, 6 + 6 rows, frequencies 0.4045 / 0.406 / 0.4075 c/a0, gray start;
       stage 1: tanh projection beta 8 -> 64 (15 iterations each, MaterialGrid without
       averaging); stage 2: Meep's smoothed projection (Hammond et al.) beta 128 -> 256 ->
       inf (20 each); final design = 0.5 level set of the filtered field. DFT decay 1e-5.
valley resolution 24/a, 10 + 10 rows, frequencies 0.232 ... 0.2415 c/a (5), gray start
       plus small smooth random noise (the wall is mirror-symmetric and its edge mode odd,
       so the design must break the symmetry); MaterialGrid with subpixel averaging and
       beta 8 -> 16 -> 32 -> 64 -> inf (12, 12, 12, 12, 24 iterations; MMA initial step 0.05
       at inf), i.e. the representation used in the validation; DFT decay 1e-9 (accurate
       adjoint gradients); best binary iterate kept.

Usage (mp env, from topological/):  python scripts/topopt_transition.py spin|valley [MAX_HOURS]
"""
import sys
import time
from pathlib import Path

import autograd.numpy as npa
import meep as mp
import meep.adjoint as mpa
import nlopt
import numpy as np
from autograd import tensor_jacobian_product

from pic_toolkit.meep_sim import topological_pc as tp

mp.verbosity(0)
DATA = Path(__file__).resolve().parents[1] / "data" / "topopt"

SETUP = {
    "spin": dict(kind="spin_te", res=24, n_rows=6, freqs=[0.4045, 0.406, 0.4075], fcen=0.406, fwidth=0.03,
                 a_nm=tp.A0_NM, y_wall=3 * tp.ROW_H, decay=1e-5, min_run=250),
    "valley": dict(kind="valley_te", res=24, n_rows=10, freqs=[0.232, 0.2345, 0.2368, 0.239, 0.2415],
                   fcen=0.2368, fwidth=0.03, a_nm=tp.A_NM, y_wall=4.5, decay=1e-9, min_run=400),
}
DES_C, DES_S = (2.0, 0.0), (4.0, 4.0)
X_LEFT, X_PML_R, X_T, DPML_X, DPML_RIGHT, DPML_Y, PAD_Y = -3.5, 13.0, 12.0, 1.5, 2.5, 1.5, 0.5
X_SRC, X_INC = -1.5, -0.8


def problem(name: str, averaging: bool, beta0: float):
    c = SETUP[name]
    res, freqs = c["res"], np.array(c["freqs"])
    sx, cx = X_PML_R + DPML_RIGHT - X_LEFT, (X_PML_R + DPML_RIGHT + X_LEFT) / 2
    sy = 2 * (c["n_rows"] * tp.ROW_H + PAD_Y + DPML_Y)
    w = tp.strip_width(c["kind"])
    nx, ny = int(round(DES_S[0] * res)) + 1, int(round(DES_S[1] * res)) + 1
    bounds = [mp.PML(DPML_Y, direction=mp.Y), mp.PML(DPML_X, direction=mp.X, side=mp.Low),
              mp.PML(DPML_RIGHT, direction=mp.X, side=mp.High)]
    src = [mp.EigenModeSource(mp.GaussianSource(c["fcen"], fwidth=c["fwidth"]), center=mp.Vector3(X_SRC, 0),
                              size=mp.Vector3(0, sy - 2 * DPML_Y), eig_band=1, eig_parity=mp.TE)]
    # incident power: straight strip through the cell
    ref = mp.Simulation(cell_size=mp.Vector3(sx, sy), geometry_center=mp.Vector3(cx, 0),
                        geometry=[mp.Block(size=mp.Vector3(mp.inf, w), material=tp.HOST)], sources=src,
                        boundary_layers=bounds, resolution=res)
    fl = ref.add_flux(freqs, mp.FluxRegion(center=mp.Vector3(X_INC, 0), size=mp.Vector3(0, sy - 2 * DPML_Y)))
    ref.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-9 if averaging else 1e-8, minimum_run_time=50))
    p_inc = np.asarray(mp.get_fluxes(fl))
    kw = dict(do_averaging=True, beta=beta0, eta=0.5) if averaging else {}
    dv = mp.MaterialGrid(mp.Vector3(nx, ny), mp.air, tp.HOST, grid_type="U_MEAN", **kw)
    region = mpa.DesignRegion(dv, volume=mp.Volume(center=mp.Vector3(*DES_C), size=mp.Vector3(*DES_S)))
    geom = tp.coupler_geometry(c["kind"], w, X_LEFT, X_PML_R + DPML_RIGHT, c["n_rows"])
    geom.append(mp.Block(center=region.center, size=region.size, material=dv))
    sim = mp.Simulation(cell_size=mp.Vector3(sx, sy), geometry_center=mp.Vector3(cx, 0), geometry=geom,
                        sources=src, boundary_layers=bounds, resolution=res)
    line = mp.Volume(center=mp.Vector3(X_T, 0), size=mp.Vector3(0, 2 * c["y_wall"]))
    mons = [mpa.FourierFields(sim, line, mp.Ey), mpa.FourierFields(sim, line, mp.Hz)]

    def J(ey, hz):
        n = min(ey.shape[1], hz.shape[1])
        return npa.mean(npa.real(npa.sum(npa.conj(ey[:, :n]) * hz[:, :n], axis=1)) / res / p_inc)

    opt = mpa.OptimizationProblem(simulation=sim, objective_functions=[J], objective_arguments=mons,
                                  design_regions=[region], frequencies=list(freqs), decay_by=c["decay"],
                                  minimum_run_time=c["min_run"], maximum_run_time=3000 if averaging else 1500)
    radius = mpa.get_conic_radius_from_eta_e(100.0 / c["a_nm"], 0.75)
    filt = lambda x: mpa.conic_filter(x.reshape(nx, ny), radius, DES_S[0], DES_S[1], res)
    return opt, dv, filt, (nx, ny), radius


def run_stages(opt, stages, x, mapping, hist, deadline, path, extra, dv=None):
    best = [-np.inf, x.copy()]
    t0 = time.time()
    for beta, nit in stages:
        if time.time() > deadline:
            break

        def f(v, grad):
            if time.time() > deadline:
                raise nlopt.ForcedStop
            if dv is not None:
                dv.beta = beta
            f0, dJ = opt([mapping(v, beta)])
            dJ = np.asarray(dJ)
            dJ = dJ.sum(axis=-1) if dJ.ndim > 1 else dJ
            if grad.size > 0:
                grad[:] = tensor_jacobian_product(mapping, 0)(v, beta, dJ)
            val = float(np.real(f0))
            if beta == np.inf and val > best[0]:
                best[0], best[1] = val, v.copy()
            for k, vv in (("J", val), ("beta", beta), ("t", time.time() - t0)):
                hist[k].append(vv)
            np.savez(path, **hist, x=v, **extra)
            print(f"iter {len(hist['J'])} beta {beta} J {val:.4f}", flush=True)
            return val

        s = nlopt.opt(nlopt.LD_MMA, x.size)
        s.set_lower_bounds(0); s.set_upper_bounds(1); s.set_max_objective(f); s.set_maxeval(nit)
        if dv is not None and beta == np.inf:
            s.set_initial_step(0.05)
        try:
            x[:] = s.optimize(x)
        except nlopt.ForcedStop:
            break
    return x, best


if __name__ == "__main__":
    name = sys.argv[1]
    deadline = time.time() + (float(sys.argv[2]) if len(sys.argv) > 2 else 12.0) * 3600
    tag = f"{name}_rerun"
    if name == "spin":
        opt, dv, filt, (nx, ny), radius = problem("spin", averaging=False, beta0=0)
        x = 0.5 * np.ones(nx * ny)
        h1 = {"J": [], "beta": [], "t": []}
        tanh_map = lambda v, b: mpa.tanh_projection(filt(v), b, 0.5).flatten()
        x, _ = run_stages(opt, [(8, 15), (16, 15), (32, 15), (64, 15)], x, tanh_map, h1, deadline,
                          DATA / f"{tag}_stage1_history.npz", {})
        h2 = {"J": [], "beta": [], "t": []}
        smooth_map = lambda v, b: mpa.smoothed_projection(filt(v), b, 0.5, SETUP["spin"]["res"]).flatten()
        x, best = run_stages(opt, [(128, 20), (256, 20), (np.inf, 20)], x, smooth_map, h2, deadline,
                             DATA / f"{tag}_stage2_history.npz", {})
        xf = x
    else:
        opt, dv, filt, (nx, ny), radius = problem("valley", averaging=True, beta0=8)
        from scipy.ndimage import gaussian_filter
        noise = gaussian_filter(np.random.default_rng(1).standard_normal((nx, ny)), 3)
        x = np.clip(0.5 + 0.15 * noise / np.abs(noise).max(), 0, 1).flatten()
        h = {"J": [], "beta": [], "t": []}
        x, best = run_stages(opt, [(8, 12), (16, 12), (32, 12), (64, 12), (np.inf, 24)], x,
                             lambda v, b: filt(v).flatten(), h, deadline, DATA / f"{tag}_history.npz", {}, dv=dv)
        xf = best[1] if np.isfinite(best[0]) else x
    w_bin = (np.asarray(filt(xf)) > 0.5).astype(float)
    if dv is not None and name == "valley":
        dv.beta = np.inf
    j_bin = float(np.real(opt([w_bin.flatten()], need_gradient=False)[0]))
    print(f"final binary design: J = {j_bin:.4f}", flush=True)
    np.savez(DATA / f"{tag}_design.npz", x=xf, weights=w_bin, j_final=j_bin, center=DES_C, size=DES_S,
             res=SETUP[name]["res"], freqs=SETUP[name]["freqs"], filter_radius=radius,
             min_len=100.0 / SETUP[name]["a_nm"], w=tp.strip_width(SETUP[name]["kind"]))
