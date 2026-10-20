"""SiN grating coupler in Meep 2D x-z: uniform (1 atom/period) and meta (n atoms/period).

**Modeling assumption (deviates from the toolkit's 2D effective-index default).** Like
`grating_coupler.py`, this module resolves the real vertical x-z layer stack with real
bulk indices (SiN 2.0, SiO2 1.44, Si 3.45), non-dispersive across the band, and assumes
the device is invariant along y (no taper, infinitely wide grating). Used by
`meta/notebooks/01`-`03`; parameter rationale in `docs/simulation_settings_record.md`.

Cross-section (physical x = propagation, z = vertical; Meep's (x, y) = physical (x, z),
Meep's invariant axis = physical y), bottom to top: Si substrate / BOX / SiN core /
top cladding. Physical TE (E along y, i.e. along Meep's invariant axis) is Meep's
`mp.TM` parity, the same rotated-axis convention as `pic_toolkit.meep_sim.grating_coupler`.

One period holds `n_atoms` meta-atoms, each an unetched SiN tooth followed by a
partially etched groove: widths (w_sin_1, w_gap_1, ..., w_sin_n, w_gap_n), plus one
shared etch depth. The period P is fixed; any slack P - sum(widths) is added to the
last groove. n_atoms = 1 is the conventional grating.

Two geometry representations of the same design:
- "block": mp.Block objects with Meep's subpixel averaging (used for final evaluation);
- "grid":  an mp.MaterialGrid whose density is the exact per-cell fill fraction of
  each material, written with autograd so it is differentiable in the widths and the
  etch depth. Physical-TE E lies along y, parallel to every interface, so the correct
  subpixel average is the arithmetic mean of epsilon, which is exactly what
  MaterialGrid's linear interpolation between the two media gives.

Coupling efficiency (reciprocity overlap on a line `z_mon_up_offset_um` above the SiN):
    eta = |X|^2 / (4 * P_fib * P_in),  X = integral(Ey_s conj(Hx_f) + conj(Ey_f) Hx_s) dx
with P_in the incident guided-mode power (|alpha_+|^2) and P_fib the fiber mode's
power through the line. eta <= up-going power fraction by Cauchy-Schwarz.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import pickle
import sys
import time
from pathlib import Path

import autograd.numpy as npa
import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd.core import make_vjp
from scipy.optimize import brentq

mp.verbosity(0)

DEFAULT_PARAMS = {
    # wavelengths
    "wl0_um": 1.35, "wl_min_um": 1.30, "wl_max_um": 1.40, "n_freq": 21,
    # materials (non-dispersive)
    "n_core": 2.0, "n_box": 1.44, "n_top": 1.44, "n_sub": 3.45,
    # vertical stack
    "core_um": 0.40, "box_um": 2.0, "top_um": 2.5, "sub_um": 1.0,
    # grating
    "period_um": 0.90, "n_periods": 22, "n_atoms": 1,
    "widths_um": [0.45, 0.45], "etch_um": 0.20,
    # fiber
    "fiber_angle_deg": 10.0, "mfd_um": 9.2,
    # simulation
    "resolution": 40, "dpml_um": 1.0, "margin_um": 0.5, "lead_um": 3.0, "tail_um": 2.0,
    "z_mon_up_offset_um": 1.0, "mode_span_um": 1.0, "run_time_factor": 2.0,
    "design_res_factor": 1, "eps_averaging": True,
}


@contextlib.contextmanager
def quiet():
    """Silence Meep's chatter for one run, including the C++ library's direct writes to
    the stdout file descriptor (e.g. MaterialGrid's 'vol mismatch' notices)."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


def silent(fn):
    """Decorator: run the whole function inside `quiet()` (nesting is harmless)."""
    def wrapper(*a, **k):
        with quiet():
            return fn(*a, **k)
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


def cached(path, fn, force: bool = False):
    """Load a pickled result from `path` if it exists, otherwise compute fn() and save it.
    Lets long notebook sections resume after an interruption."""
    path = Path(path)
    if path.exists() and not force:
        with open(path, "rb") as f:
            return pickle.load(f)
    out = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def resolve(params: dict | None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    assert len(p["widths_um"]) == 2 * p["n_atoms"], "need one SiN width and one groove width per atom"
    return p


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------
def domain(p: dict) -> dict:
    """All x/z positions. x = 0 at the grating start; z = 0 at the BOX bottom."""
    x_g0, x_g1 = 0.0, p["n_periods"] * p["period_um"]
    x_src = x_g0 - p["lead_um"] + 0.5
    x_refl = x_src + 0.5
    x_trans = x_g1 + p["tail_um"] - 0.5
    x_left, x_right = x_g0 - p["lead_um"], x_g1 + p["tail_um"]
    z_box1 = p["box_um"]
    z_core1 = z_box1 + p["core_um"]
    z_top = z_core1 + p["top_um"]
    z_sub0 = -p["sub_um"]
    dpml = p["dpml_um"]
    return dict(
        x_g0=x_g0, x_g1=x_g1, x_src=x_src, x_refl=x_refl, x_trans=x_trans, x_left=x_left, x_right=x_right,
        z_box0=0.0, z_box1=z_box1, z_core1=z_core1, z_top=z_top, z_sub0=z_sub0,
        z_up=z_core1 + p["z_mon_up_offset_um"], z_down=-0.5 * p["sub_um"],
        z_mode=(z_box1 + z_core1) / 2, mode_h=p["core_um"] + 2 * p["mode_span_um"],
        cell_x=round(((x_right - x_left) + 2 * dpml) * p["resolution"]) / p["resolution"],
        cell_z=round(((z_top - z_sub0) + 2 * dpml) * p["resolution"]) / p["resolution"],
        x_c=(x_left + x_right) / 2, z_c=(z_top + z_sub0) / 2,
    )


def groove_intervals(p: dict) -> list[tuple[float, float]]:
    """Absolute x intervals of every etched groove. Each period: tooth, groove, tooth,
    groove, ...; the slack P - sum(widths) widens the period's last groove."""
    w = np.asarray(p["widths_um"], dtype=float)
    P = p["period_um"]
    slack = P - w.sum()
    out = []
    for k in range(p["n_periods"]):
        x = k * P
        for a in range(p["n_atoms"]):
            x += w[2 * a]
            g = w[2 * a + 1] + (slack if a == p["n_atoms"] - 1 else 0.0)
            out.append((x, x + g))
            x += g
    return out


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def media(p):
    return (mp.Medium(index=p["n_core"]), mp.Medium(index=p["n_box"]), mp.Medium(index=p["n_top"]),
            mp.Medium(index=p["n_sub"]))


def stack_blocks(p: dict, d: dict) -> list:
    """Substrate (through the bottom PML), BOX, and a full-length SiN core."""
    core, box, _, sub = media(p)
    big = d["cell_x"] + 10
    h_sub = p["sub_um"] + p["dpml_um"] + 1.0
    return [
        mp.Block(size=mp.Vector3(big, h_sub), center=mp.Vector3(d["x_c"], d["z_box0"] - h_sub / 2), material=sub),
        mp.Block(size=mp.Vector3(big, p["box_um"]), center=mp.Vector3(d["x_c"], p["box_um"] / 2), material=box),
        mp.Block(size=mp.Vector3(big, p["core_um"]), center=mp.Vector3(d["x_c"], (d["z_box1"] + d["z_core1"]) / 2),
                 material=core),
    ]


def block_geometry(p: dict, d: dict) -> list:
    """Stack plus one top-cladding block per groove, `etch_um` deep from the SiN top."""
    _, _, top, _ = media(p)
    geo = stack_blocks(p, d)
    e = p["etch_um"]
    for x0, x1 in groove_intervals(p):
        if x1 - x0 > 0:
            geo.append(mp.Block(size=mp.Vector3(x1 - x0, e), center=mp.Vector3((x0 + x1) / 2, d["z_core1"] - e / 2),
                                material=top))
    return geo


def design_region_box(p: dict, d: dict) -> dict:
    """Design region: the grating's x span (+ 2 cells each side) times the SiN layer
    (+ 2 cells above and below), and its MaterialGrid point coordinates."""
    dx = 1.0 / (p["resolution"] * p["design_res_factor"])
    pad = 2.0 / p["resolution"]
    x0, x1 = d["x_g0"] - pad, d["x_g1"] + pad
    z0, z1 = d["z_box1"] - pad, d["z_core1"] + pad
    nx = int(round((x1 - x0) / dx)) + 1
    nz = int(round((z1 - z0) / dx)) + 1
    return dict(x0=x0, x1=x1, z0=z0, z1=z1, nx=nx, nz=nz,
                xs=np.linspace(x0, x1, nx), zs=np.linspace(z0, z1, nz), dx=(x1 - x0) / (nx - 1), dz=(z1 - z0) / (nz - 1))


def _fill(c, h, a, b):
    """Fraction of each cell [c - h/2, c + h/2] covered by the interval [a, b]
    (piecewise linear in a and b, so autograd gives a continuous derivative)."""
    lo = npa.maximum(c - h / 2, a)
    hi = npa.minimum(c + h / 2, b)
    return npa.clip(hi - lo, 0.0, None) / h


def density(theta, p: dict, d: dict, reg: dict):
    """MaterialGrid weights (nx, nz) for parameters theta = (*widths_um, etch_um):
    rho = f_core(z) - g(x) * f_etch(z), with g the groove fill fraction along x."""
    n = p["n_atoms"]
    w = theta[: 2 * n]
    etch = theta[2 * n]
    P = p["period_um"]
    slack = P - npa.sum(w)
    xs, zs = reg["xs"], reg["zs"]
    g = npa.zeros(xs.shape)
    for k in range(p["n_periods"]):
        x = k * P
        for a in range(n):
            x = x + w[2 * a]
            gw = w[2 * a + 1] + (slack if a == n - 1 else 0.0)
            g = g + _fill(xs, reg["dx"], x, x + gw)
            x = x + gw
    f_core = _fill(zs, reg["dz"], d["z_box1"], d["z_core1"])
    f_etch = _fill(zs, reg["dz"], d["z_core1"] - etch, d["z_core1"])
    return npa.clip(f_core[None, :] - g[:, None] * f_etch[None, :], 0.0, 1.0)


def theta_of(p: dict) -> np.ndarray:
    return np.array([*p["widths_um"], p["etch_um"]], dtype=float)


def grid_geometry(p: dict, d: dict, weights=None):
    """Stack plus a MaterialGrid design block over the grating (clad <-> SiN)."""
    assert p["n_top"] == p["n_box"], "the grid representation needs the same cladding above and below the core"
    core, box, _, _ = media(p)
    reg = design_region_box(p, d)
    mg = mp.MaterialGrid(mp.Vector3(reg["nx"], reg["nz"]), box, core, grid_type="U_DEFAULT", do_averaging=False)
    if weights is None:
        weights = density(theta_of(p), p, d, reg)
    mg.update_weights(np.asarray(weights, dtype=float))
    center = mp.Vector3((reg["x0"] + reg["x1"]) / 2, (reg["z0"] + reg["z1"]) / 2)
    size = mp.Vector3(reg["x1"] - reg["x0"], reg["z1"] - reg["z0"])
    dr = mpa.DesignRegion(mg, volume=mp.Volume(center=center, size=size))
    geo = stack_blocks(p, d) + [mp.Block(center=center, size=size, material=mg)]
    return geo, mg, dr, reg


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _freqs(p):
    """Frequency grid centred exactly on 1/wl0 (so the design wavelength is a sample)
    and wide enough to cover [wl_min, wl_max]."""
    f0 = 1 / p["wl0_um"]
    half = max(f0 - 1 / p["wl_max_um"], 1 / p["wl_min_um"] - f0)
    return f0, 2 * half


def make_sim(p: dict, geometry: list) -> tuple[mp.Simulation, dict]:
    d = domain(p)
    fcen, fw = _freqs(p)
    src = mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=fw),
        center=mp.Vector3(d["x_src"], d["z_mode"]), size=mp.Vector3(0, d["mode_h"]),
        eig_band=1, eig_parity=mp.TM, eig_match_freq=True, eig_kpoint=mp.Vector3(1, 0, 0))
    _, _, top, _ = media(p)
    sim = mp.Simulation(cell_size=mp.Vector3(d["cell_x"], d["cell_z"]), boundary_layers=[mp.PML(p["dpml_um"])],
                        geometry=geometry, sources=[src], default_material=top, resolution=p["resolution"],
                        geometry_center=mp.Vector3(d["x_c"], d["z_c"]), eps_averaging=p["eps_averaging"])
    return sim, d


def up_line(d: dict) -> mp.Volume:
    """Near-field line above the SiN, spanning the whole interior in x."""
    return mp.Volume(center=mp.Vector3(d["x_c"], d["z_up"]), size=mp.Vector3(d["x_right"] - d["x_left"], 0))


@silent
def run_forward(p: dict | None = None, geometry_kind: str = "block", weights=None, field_map: bool = False,
                waveguide_probes: bool = False) -> dict:
    """Broadband forward run: reflection, transmission, up/down power, near field on the
    up line (Ez, Hx at every frequency), incident mode power. Returns plain arrays.
    `waveguide_probes` adds an Ez line along the core centre (phase-slope n_eff) and a
    vertical flux line at the transmission plane (checks |alpha|^2 against the flux)."""
    p = resolve(p)
    d = domain(p)
    if geometry_kind == "block":
        geo = block_geometry(p, d)
    else:
        geo, _, _, _ = grid_geometry(p, d, weights)
    sim, d = make_sim(p, geo)
    fcen, fw = _freqs(p)
    nf = p["n_freq"]
    mode_vol = lambda x: mp.Volume(center=mp.Vector3(x, d["z_mode"]), size=mp.Vector3(0, d["mode_h"]))
    m_refl = sim.add_mode_monitor(fcen, fw, nf, mp.ModeRegion(volume=mode_vol(d["x_refl"])))
    m_trans = sim.add_mode_monitor(fcen, fw, nf, mp.ModeRegion(volume=mode_vol(d["x_trans"])))
    f_up = sim.add_flux(fcen, fw, nf, mp.FluxRegion(volume=up_line(d)))
    f_down = sim.add_flux(fcen, fw, nf, mp.FluxRegion(center=mp.Vector3(d["x_c"], d["z_down"]),
                                                     size=mp.Vector3(d["x_right"] - d["x_left"], 0)))
    near = sim.add_dft_fields([mp.Ez, mp.Hx], fcen, fw, nf, where=up_line(d))
    probes = None
    if waveguide_probes:
        core_line = sim.add_dft_fields([mp.Ez], fcen, fw, nf, center=mp.Vector3(d["x_c"], d["z_mode"]),
                                       size=mp.Vector3(d["x_right"] - d["x_left"], 0))
        f_vert = sim.add_flux(fcen, fw, nf, mp.FluxRegion(center=mp.Vector3(d["x_trans"], d["z_c"]),
                                                         size=mp.Vector3(0, d["z_top"] - d["z_sub0"])))
        probes = (core_line, f_vert)
    fmap = None
    if field_map:
        fmap = sim.add_dft_fields([mp.Ez], p["wl0_um"] ** -1, 0, 1, center=mp.Vector3(d["x_c"], d["z_c"]),
                                  size=mp.Vector3(d["cell_x"] - 2 * p["dpml_um"], d["cell_z"] - 2 * p["dpml_um"]))
    t_run = p["run_time_factor"] * p["n_core"] * d["cell_x"]
    t0 = time.perf_counter()
    with quiet():
        sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-6, minimum_run_time=t_run))
    wall = time.perf_counter() - t0
    freqs = np.array(mp.get_flux_freqs(f_up))
    with quiet():
        a_r = sim.get_eigenmode_coefficients(m_refl, [1], eig_parity=mp.TM).alpha[0]
        a_t = sim.get_eigenmode_coefficients(m_trans, [1], eig_parity=mp.TM).alpha[0]
    p_in = np.abs(a_r[:, 0]) ** 2
    ez = np.stack([sim.get_dft_array(near, mp.Ez, i) for i in range(nf)])
    hx = np.stack([sim.get_dft_array(near, mp.Hx, i) for i in range(nf)])
    xs, _, _, _ = sim.get_array_metadata(dft_cell=near)
    out = dict(
        params=p, dom=d, freqs=freqs, wl=1 / freqs, p_in=p_in,
        R=np.abs(a_r[:, 1]) ** 2 / p_in, T=np.abs(a_t[:, 0]) ** 2 / p_in,
        up=np.array(mp.get_fluxes(f_up)) / p_in, down=-np.array(mp.get_fluxes(f_down)) / p_in,
        x_up=np.asarray(xs), ez_up=ez, hx_up=hx, wall_s=wall, steps=sim.timestep(),
    )
    if probes is not None:
        core_line, f_vert = probes
        out["ez_core"] = np.stack([sim.get_dft_array(core_line, mp.Ez, i) for i in range(nf)])
        out["x_core"] = np.asarray(sim.get_array_metadata(dft_cell=core_line)[0])
        out["flux_vert"] = np.array(mp.get_fluxes(f_vert)) / p_in
        out["a_trans"] = np.abs(a_t[:, 0]) ** 2 / p_in
    if field_map:
        out["field_map"] = sim.get_dft_array(fmap, mp.Ez, 0)
        out["eps_map"] = sim.get_array(center=mp.Vector3(d["x_c"], d["z_c"]),
                                       size=mp.Vector3(d["cell_x"] - 2 * p["dpml_um"], d["cell_z"] - 2 * p["dpml_um"]),
                                       component=mp.Dielectric)
    return out


# ---------------------------------------------------------------------------
# Slab modes
# ---------------------------------------------------------------------------
def te_slab_neff(layers, n_bot: float, n_top: float, wl: float) -> float:
    """Fundamental TE n_eff of a planar stack (transfer matrix, exact). `layers` is a
    list of (index, thickness_um) from bottom to top between two semi-infinite claddings."""
    k0 = 2 * np.pi / wl

    def disp(neff):
        gb, gt = k0 * np.sqrt(neff**2 - n_bot**2), k0 * np.sqrt(neff**2 - n_top**2)
        v = np.array([1.0, gb])
        for n, t in layers:
            q2 = k0**2 * (n**2 - neff**2)
            if q2 > 0:
                q = np.sqrt(q2)
                M = np.array([[np.cos(q * t), np.sin(q * t) / q], [-q * np.sin(q * t), np.cos(q * t)]])
            else:
                q = np.sqrt(-q2)
                M = np.array([[np.cosh(q * t), np.sinh(q * t) / q], [q * np.sinh(q * t), np.cosh(q * t)]])
            v = M @ v
        return v[1] + gt * v[0]

    grid = np.linspace(max(n_bot, n_top) + 1e-9, max(n for n, _ in layers) - 1e-9, 4000)
    f = [disp(x) for x in grid]
    roots = [brentq(disp, grid[i], grid[i + 1]) for i in range(len(grid) - 1) if np.sign(f[i]) != np.sign(f[i + 1])]
    return max(roots)


@silent
def meep_slab_neff(p: dict | None = None, parity=mp.TM) -> tuple[float, np.ndarray, np.ndarray]:
    """n_eff of the unetched core from Meep's eigensolver on the FDTD grid at wl0, plus
    the mode's Ez and Ey along z (to identify which polarization it is)."""
    p = resolve({**(p or {}), "n_periods": 0})
    d = domain(p)
    sim, d = make_sim(p, stack_blocks(p, d))
    f0 = 1 / p["wl0_um"]
    # Same vertical span as the mode source: a span reaching the Si substrate would return
    # the substrate's own slab mode (n_eff ~ 3.39) instead of the SiN core mode.
    vol = mp.Volume(center=mp.Vector3(d["x_src"], d["z_mode"]), size=mp.Vector3(0, d["mode_h"]))
    with quiet():
        sim.init_sim()
        em = sim.get_eigenmode(f0, mp.X, vol, 1, mp.Vector3(1.7 * f0), parity=parity)
    zs = np.linspace(d["z_mode"] - d["mode_h"] / 2, d["z_mode"] + d["mode_h"] / 2, 400)
    ez = np.array([abs(em.amplitude(mp.Vector3(d["x_src"], z), mp.Ez)) for z in zs])
    ey = np.array([abs(em.amplitude(mp.Vector3(d["x_src"], z), mp.Ey)) for z in zs])
    return float(em.k.x / f0), zs, np.vstack([ez, ey])


# ---------------------------------------------------------------------------
# Fiber overlap
# ---------------------------------------------------------------------------
PHASE_SIGN = +1   # forward-wave DFT phase grows with x (exp(+i k x)); verified in 01, Section 2


def fiber_mode(x, x0, wl, p, angle_deg=None, mfd=None, n_clad=None):
    """Physical (Ey, Hx) of a tilted Gaussian fiber beam travelling up and toward +x
    (for a positive angle), evaluated on the up line. Hx = -n cos(theta) Ey."""
    th = math.radians(p["fiber_angle_deg"] if angle_deg is None else angle_deg)
    w0 = (p["mfd_um"] if mfd is None else mfd) / 2
    n = p["n_top"] if n_clad is None else n_clad
    k = 2 * np.pi / wl * n
    ey = np.exp(-(((x - x0) * math.cos(th)) ** 2) / w0**2) * np.exp(PHASE_SIGN * 1j * k * math.sin(th) * (x - x0))
    return ey, -n * math.cos(th) * ey


def eta_overlap(x, ez, hx, wl, p_in, p, x0, **fiber_kw):
    """Coupling efficiency to the fiber mode centred at x0 (see module docstring).
    Physical Ey = -Ez(Meep) (one axis pair is swapped), Hx = Hx(Meep)."""
    ey_s, hx_s = -ez, hx
    ey_f, hx_f = fiber_mode(x, x0, wl, p, **fiber_kw)
    dx = x[1] - x[0]
    X = np.sum(ey_s * np.conj(hx_f) + np.conj(ey_f) * hx_s) * dx
    p_f = abs(np.real(np.sum(ey_f * np.conj(hx_f))) * dx)
    return float(np.abs(X) ** 2 / (4 * p_f * p_in))


def best_x0(res: dict, i_wl: int | None = None, x0_grid=None, **fiber_kw):
    """Fiber position (on the up line) that maximises eta at one wavelength."""
    p = res["params"]
    if i_wl is None:
        i_wl = int(np.argmin(np.abs(res["wl"] - p["wl0_um"])))
    x = res["x_up"]
    if x0_grid is None:
        x0_grid = np.arange(x[0] + 2.0, x[-1] - 2.0, 0.05)
    etas = [eta_overlap(x, res["ez_up"][i_wl], res["hx_up"][i_wl], res["wl"][i_wl], res["p_in"][i_wl], p, x0, **fiber_kw)
            for x0 in x0_grid]
    k = int(np.argmax(etas))
    return float(x0_grid[k]), float(etas[k])


def eta_spectrum(res: dict, x0: float, **fiber_kw) -> np.ndarray:
    """eta(lambda) for a fiber held fixed at x0."""
    p = res["params"]
    return np.array([eta_overlap(res["x_up"], res["ez_up"][i], res["hx_up"][i], res["wl"][i], res["p_in"][i], p, x0,
                                 **fiber_kw) for i in range(len(res["wl"]))])


# ---------------------------------------------------------------------------
# Adjoint: eta(theta) and its gradient at the design wavelength
# ---------------------------------------------------------------------------
class AdjointProblem:
    """mpa.OptimizationProblem on the grid geometry; objective = eta at wl0 for a fiber
    at `x0` (updated by the caller); gradient chained from d eta/d rho to d eta/d theta."""

    @silent
    def __init__(self, params: dict, p_in: float):
        p = resolve(params)
        self.p, self.d = p, domain(p)
        geo, self.mg, self.dr, self.reg = grid_geometry(p, self.d)
        sim, _ = make_sim(p, geo)
        line = up_line(self.d)
        self.ff_ez = mpa.FourierFields(sim, line, mp.Ez)
        self.ff_hx = mpa.FourierFields(sim, line, mp.Hx)
        self.p_in = float(p_in)
        self.x0 = None
        self.f0 = 1 / p["wl0_um"]
        with quiet():
            sim.init_sim()
            xs, _, _, _ = sim.get_array_metadata(vol=line)
        self.x = np.asarray(xs)

        def J(ez, hx):
            ey_s = -npa.ravel(ez)
            hx_s = npa.ravel(hx)
            ey_f, hx_f = fiber_mode(self.x, self.x0, p["wl0_um"], p)
            dx = self.x[1] - self.x[0]
            X = npa.sum(ey_s * np.conj(hx_f) + np.conj(ey_f) * hx_s) * dx
            p_f = abs(np.real(np.sum(ey_f * np.conj(hx_f))) * dx)
            return npa.abs(X) ** 2 / (4 * p_f * self.p_in)

        t_min = p["run_time_factor"] * p["n_core"] * self.d["cell_x"]
        self.opt = mpa.OptimizationProblem(simulation=sim, objective_functions=[J],
                                           objective_arguments=[self.ff_ez, self.ff_hx], design_regions=[self.dr],
                                           frequencies=[self.f0], minimum_run_time=t_min, decay_by=1e-6)

    def near_field(self):
        """Ez, Hx on the up line from the last forward run (for re-optimising x0)."""
        return np.ravel(self.ff_ez._eval), np.ravel(self.ff_hx._eval)

    def value(self, theta) -> float:
        """eta only (forward run, no adjoint), e.g. for finite-difference checks."""
        rho = density(np.asarray(theta, dtype=float), self.p, self.d, self.reg)
        with quiet():
            f, _ = self.opt([np.asarray(rho).ravel()], need_gradient=False)
        return float(np.real(f[0] if np.ndim(f) else f))

    def value_and_grad(self, theta):
        back, rho = make_vjp(lambda t: density(t, self.p, self.d, self.reg), np.asarray(theta, dtype=float))
        with quiet():
            f, g = self.opt([np.asarray(rho).ravel()])
        g = np.asarray(g).reshape(rho.shape)
        return float(np.real(f[0] if np.ndim(f) else f)), np.asarray(back(g))


# ---------------------------------------------------------------------------
# Constraints and Adam
# ---------------------------------------------------------------------------
def project(theta, p: dict, w_min: float = 0.10, etch_bounds=(0.02, None)) -> np.ndarray:
    """Euclidean projection of the widths onto {w_i >= w_min, sum(w) <= P}; etch clipped
    to [etch_min, core thickness]."""
    n = p["n_atoms"]
    w = np.maximum(np.asarray(theta[: 2 * n], dtype=float), w_min)
    P = p["period_um"]
    if w.sum() > P:
        # project onto {sum = P, w >= w_min}: shift by a common tau, clip, repeat (simplex projection)
        lo, hi = 0.0, float(w.max())
        for _ in range(100):
            tau = (lo + hi) / 2
            s = np.maximum(w - tau, w_min).sum()
            lo, hi = (tau, hi) if s > P else (lo, tau)
        w = np.maximum(w - hi, w_min)
    e_lo, e_hi = etch_bounds
    etch = float(np.clip(theta[2 * n], e_lo, p["core_um"] if e_hi is None else e_hi))
    return np.array([*w, etch])


def adam(value_and_grad, theta0, p: dict, lr=0.01, n_iter=40, beta1=0.9, beta2=0.999, eps=1e-8, w_min=0.10,
         callback=None) -> list[dict]:
    """Bias-corrected Adam ascent on eta, with a projection onto the constraints after
    every step (same algorithm as pic_toolkit.optimization.adam_nd, gradient supplied)."""
    theta = project(theta0, p, w_min)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    hist = []
    for t in range(n_iter + 1):
        f, g = value_and_grad(theta)
        hist.append({"iter": t, "eta": f, "theta": theta.copy(), "grad": g.copy()})
        if callback:
            callback(hist[-1])
        if t == n_iter:
            break
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        mh, vh = m / (1 - beta1 ** (t + 1)), v / (1 - beta2 ** (t + 1))
        theta = project(theta + lr * mh / (np.sqrt(vh) + eps), p, w_min)
    return hist


def optimize(params: dict, theta0, checkpoint, n_iter: int = 40, lr0: float = 0.02, lr1: float = 0.002,
             w_min: float = 0.10, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, log=print) -> dict:
    """Adam ascent of eta(theta) at wl0 with the adjoint gradient, resumable.

    - Constraints: projection onto {w_i >= w_min, sum(w) <= P, etch in [0.02, core]} after every step.
    - Learning rate: cosine decay from lr0 to lr1 (um per step, Adam-normalised).
    - Fiber position: re-optimised after every forward run from the near field the adjoint
      run already computed, and used for the next iteration's objective.
    - Every iteration is checkpointed to `checkpoint` (pickle); calling again resumes.
    """
    p = resolve(params)
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        with open(checkpoint, "rb") as f:
            st = pickle.load(f)
    else:
        seed = run_forward({**p, "widths_um": list(theta0[:-1]), "etch_um": float(theta0[-1])}, "grid")
        i0 = int(np.argmin(np.abs(seed["wl"] - p["wl0_um"])))
        x0, _ = best_x0(seed)
        st = {"theta": project(theta0, p, w_min), "m": np.zeros(len(theta0)), "v": np.zeros(len(theta0)),
              "t": 0, "x0": x0, "p_in": float(seed["p_in"][i0]), "history": []}
    ap = AdjointProblem({**p, "widths_um": list(st["theta"][:-1]), "etch_um": float(st["theta"][-1])}, st["p_in"])
    x_grid = np.arange(ap.x[0] + 2.0, ap.x[-1] - 2.0, 0.05)
    while st["t"] <= n_iter:
        t0 = time.perf_counter()
        ap.x0 = st["x0"]
        f, g = ap.value_and_grad(st["theta"])
        ez, hx = ap.near_field()
        etas = [eta_overlap(ap.x, ez, hx, p["wl0_um"], st["p_in"], p, x0) for x0 in x_grid]
        k = int(np.argmax(etas))
        st["history"].append({"iter": st["t"], "eta": f, "eta_best_x0": float(etas[k]), "x0": st["x0"],
                              "theta": st["theta"].copy(), "grad": g.copy(), "wall_s": time.perf_counter() - t0})
        log(f"iter {st['t']:>3}: eta {f:.4f} (best-x0 {etas[k]:.4f}), theta {np.round(st['theta'], 4)}, "
            f"|g| {np.linalg.norm(g):.3f}, {time.perf_counter() - t0:.0f} s")
        st["x0"] = float(x_grid[k])
        if st["t"] == n_iter:
            st["t"] += 1
        else:
            lr = lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * st["t"] / n_iter))
            st["m"] = beta1 * st["m"] + (1 - beta1) * g
            st["v"] = beta2 * st["v"] + (1 - beta2) * g**2
            mh = st["m"] / (1 - beta1 ** (st["t"] + 1))
            vh = st["v"] / (1 - beta2 ** (st["t"] + 1))
            st["theta"] = project(st["theta"] + lr * mh / (np.sqrt(vh) + eps), p, w_min)
            st["t"] += 1
        with open(checkpoint, "wb") as fh:
            pickle.dump(st, fh)
    best = max(st["history"], key=lambda h: h["eta_best_x0"])
    return {**st, "best": best}
