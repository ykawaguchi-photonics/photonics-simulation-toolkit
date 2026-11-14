"""2D finite-volume heat conduction in a waveguide cross-section (si_active track).

Solves, per unit length along the waveguide (the cross-section is uniform along z),

    rho c_p dT/dt = d/dx( k dT/dx ) + d/dy( k dT/dy ) + q(x, y)

for the temperature RISE dT above ambient. Boundaries:

- bottom (deep in the Si substrate): isothermal, dT = 0 (Dirichlet);
- top (oxide/air surface): convection, flux = h dT (Robin);
- left/right: adiabatic (zero flux), the natural boundary of the finite-volume form.

These match the Jacques et al. 2019 setup (si_active/literature.md). Because the problem is
linear, every result scales with the dissipated power per unit length P' = P / L. That is
why, in a cross-section that is uniform along the heater, P_pi does not depend on L.

Discretization: node-based finite volumes on a non-uniform rectilinear grid, the same
scheme as `meep_sim/tfln_electrostatics.py` (there for the Laplace equation): every
material interface lies on a grid line, conductivity and heat capacity are held per cell,
and the face conductance between two nodes takes the two adjacent cells' k weighted by
their share of the face. Heat sources are uniform over the cells of the heater region;
each cell's power goes to its four corner nodes in equal parts.

Units: coordinates in um; k in W/(m K); rho c_p in J/(m^3 K); P' in W/m; h in W/(m^2 K).
No meep import -- pure numpy/scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Thermal materials. Values: Jacques et al., Opt. Express 27, 10456 (2019), Table 3
# (read through the Tidy3D reproduction, see si_active/literature.md).
# The thin-film Si values (90 / 55 / 25) include phonon-boundary scattering and doping.
# ---------------------------------------------------------------------------
THERMAL_MATERIALS = {
    "si_substrate": {"k": 148.0, "rho": 2330.0, "cp": 711.0},
    "si_wg":        {"k": 90.0,  "rho": 2330.0, "cp": 711.0},   # 220 nm strip / rib core
    "si_slab":      {"k": 55.0,  "rho": 2330.0, "cp": 711.0},   # 90 nm slab
    "si_doped":     {"k": 25.0,  "rho": 2330.0, "cp": 711.0},   # N++ heater strips
    "sio2":         {"k": 1.38,  "rho": 2203.0, "cp": 709.0},
    "tin":          {"k": 28.0,  "rho": 5240.0, "cp": 598.0},
    "air":          {"k": 0.026, "rho": 1.177,  "cp": 1006.0},  # filled trenches (no convection inside)
}


@dataclass
class Rect:
    """Axis-aligned region of one material. Later rects override earlier ones."""
    x0: float
    x1: float
    y0: float
    y1: float
    material: str
    heater: bool = False   # True: dissipates the heater power (uniform per area)
    name: str = ""


def _graded_axis(fine_lo, fine_hi, h_fine, growth, h_max, outer_lo, outer_hi, breakpoints):
    """Node coordinates: uniform h_fine on [fine_lo, fine_hi], geometric growth outward to
    [outer_lo, outer_hi], every breakpoint inserted as a node (copied from
    tfln_electrostatics so this module has no meep_sim dependency)."""
    fine_lo, fine_hi = max(fine_lo, outer_lo), min(fine_hi, outer_hi)
    core = np.arange(fine_lo, fine_hi + 0.5 * h_fine, h_fine)
    up, h, x = [], h_fine, core[-1]
    while x < outer_hi:
        h = min(h * growth, h_max); x = x + h; up.append(min(x, outer_hi))
    down, h, x = [], h_fine, core[0]
    while x > outer_lo:
        h = min(h * growth, h_max); x = x - h; down.append(max(x, outer_lo))
    # Fixed nodes (outer bounds and material interfaces) are always kept; a graded node is
    # dropped when it falls within 0.3 h_fine of a fixed node or of the previous kept node.
    fixed = np.unique(np.round([outer_lo, outer_hi, *[b for b in breakpoints if outer_lo <= b <= outer_hi]], 9))
    graded = np.concatenate([down[::-1], core, up])
    tol = 0.3 * h_fine
    graded = graded[np.min(np.abs(graded[:, None] - fixed[None, :]), axis=1) > tol]
    nodes = np.unique(np.concatenate([fixed, graded]))
    keep = [nodes[0]]
    for v in nodes[1:]:
        if v - keep[-1] > tol or np.any(np.isclose(v, fixed)):
            if np.any(np.isclose(keep[-1], fixed)) or v - keep[-1] > tol:
                keep.append(v)
            else:              # previous was a graded node too close to a fixed one: replace it
                keep[-1] = v
    return np.array(keep)


@dataclass
class Domain:
    """Outer box and grid controls. The bottom edge is the isothermal heat sink."""
    x_min: float
    x_max: float
    y_min: float               # bottom (isothermal)
    y_max: float               # top surface (convection)
    background: str = "sio2"
    h_conv: float = 10.0       # W/(m^2 K) at the top surface (Jacques 2019)
    fine_x: tuple = (-3.0, 3.0)
    fine_y: tuple = (-2.5, 2.6)
    h_fine: float = 0.02       # um
    growth: float = 1.12
    h_max: float = 0.5         # um


@dataclass
class Problem:
    x: np.ndarray              # node coordinates (um)
    y: np.ndarray
    k: np.ndarray              # (ny-1, nx-1) per cell
    rho_cp: np.ndarray
    heat_mask: np.ndarray      # (ny-1, nx-1) bool
    material: np.ndarray       # (ny-1, nx-1) material names
    domain: Domain
    rects: list = field(default_factory=list)

    @property
    def xc(self):
        return 0.5 * (self.x[1:] + self.x[:-1])

    @property
    def yc(self):
        return 0.5 * (self.y[1:] + self.y[:-1])

    @property
    def heater_area_um2(self) -> float:
        a = np.diff(self.y)[:, None] * np.diff(self.x)[None, :]
        return float(np.sum(a[self.heat_mask]))


def build_problem(rects: list[Rect], domain: Domain, extra_x=(), extra_y=()) -> Problem:
    """Grid with every rect edge on a grid line, and per-cell material properties."""
    xb = [v for r in rects for v in (r.x0, r.x1)] + list(extra_x)
    yb = [v for r in rects for v in (r.y0, r.y1)] + list(extra_y)
    x = _graded_axis(*domain.fine_x, domain.h_fine, domain.growth, domain.h_max,
                     domain.x_min, domain.x_max, xb)
    y = _graded_axis(*domain.fine_y, domain.h_fine, domain.growth, domain.h_max,
                     domain.y_min, domain.y_max, yb)
    X, Y = np.meshgrid(0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1]))
    mat = np.full(X.shape, domain.background, dtype=object)
    heat = np.zeros(X.shape, dtype=bool)
    for r in rects:
        m = (X > r.x0) & (X < r.x1) & (Y > r.y0) & (Y < r.y1)
        mat[m] = r.material
        heat[m] = r.heater
    k = np.vectorize(lambda s: THERMAL_MATERIALS[s]["k"])(mat).astype(float)
    rc = np.vectorize(lambda s: THERMAL_MATERIALS[s]["rho"] * THERMAL_MATERIALS[s]["cp"])(mat).astype(float)
    if not heat.any():
        raise ValueError("no heater cells: at least one Rect needs heater=True")
    return Problem(x=x, y=y, k=k, rho_cp=rc, heat_mask=heat, material=mat, domain=domain, rects=list(rects))


@dataclass
class System:
    """Assembled linear system on the FREE nodes (all but the isothermal bottom row)."""
    K: sp.csr_matrix           # W/(m K) per unit length
    C: np.ndarray              # J/(m K) per unit length (lumped, diagonal)
    b_unit: np.ndarray         # W/m nodal source for P' = 1 W/m
    free: np.ndarray           # bool over all nodes (ny*nx)
    shape: tuple               # (ny, nx)


def assemble(prob: Problem) -> System:
    x, y = prob.x, prob.y
    nx, ny = len(x), len(y)
    hx, hy = np.diff(x), np.diff(y)                       # um
    # Face conductances (W/(m K)): east face of node (j,i) spans half of cell rows j-1 and j.
    k_pad = np.zeros((ny + 1, nx - 1)); k_pad[1:-1] = prob.k * hy[:, None]
    ge = 0.5 * (k_pad[:-1] + k_pad[1:]) / hx[None, :]     # (ny, nx-1), um/um -> dimensionless * k
    k_pad = np.zeros((ny - 1, nx + 1)); k_pad[:, 1:-1] = prob.k * hx[None, :]
    gn = 0.5 * (k_pad[:, :-1] + k_pad[:, 1:]) / hy[:, None]   # (ny-1, nx)

    idx = np.arange(nx * ny).reshape(ny, nx)
    rows, cols, vals = [], [], []
    diag = np.zeros((ny, nx))
    a, b = idx[:, :-1].ravel(), idx[:, 1:].ravel(); g = ge.ravel()
    rows += [a, b]; cols += [b, a]; vals += [-g, -g]
    diag[:, :-1] += ge; diag[:, 1:] += ge
    a, b = idx[:-1, :].ravel(), idx[1:, :].ravel(); g = gn.ravel()
    rows += [a, b]; cols += [b, a]; vals += [-g, -g]
    diag[:-1, :] += gn; diag[1:, :] += gn
    # Convection at the top row: h * (half the adjacent face widths), um -> m
    wtop = np.zeros(nx); wtop[:-1] += 0.5 * hx; wtop[1:] += 0.5 * hx
    diag[-1, :] += prob.domain.h_conv * wtop * 1e-6
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2)
    K = K + sp.diags(diag.ravel())

    area = hy[:, None] * hx[None, :] * 1e-12              # m^2 per cell
    cap_cell = prob.rho_cp * area                         # J/(m K)
    C = np.zeros((ny, nx))
    for sj, si in ((slice(None, -1), slice(None, -1)), (slice(None, -1), slice(1, None)),
                   (slice(1, None), slice(None, -1)), (slice(1, None), slice(1, None))):
        C[sj, si] += 0.25 * cap_cell
    # Unit source: P' = 1 W/m spread uniformly (per area) over the heater cells.
    a_heat = np.sum(area[prob.heat_mask])
    q_cell = np.where(prob.heat_mask, area / a_heat, 0.0)  # W/m per cell, sums to 1
    bnode = np.zeros((ny, nx))
    for sj, si in ((slice(None, -1), slice(None, -1)), (slice(None, -1), slice(1, None)),
                   (slice(1, None), slice(None, -1)), (slice(1, None), slice(1, None))):
        bnode[sj, si] += 0.25 * q_cell

    free = np.ones((ny, nx), dtype=bool); free[0, :] = False   # isothermal bottom
    f = free.ravel()
    return System(K=K[f][:, f].tocsc(), C=C.ravel()[f], b_unit=bnode.ravel()[f], free=f, shape=(ny, nx))


def _full(sysm: System, Tf: np.ndarray) -> np.ndarray:
    T = np.zeros(sysm.free.size, dtype=Tf.dtype)
    T[sysm.free] = Tf
    return T.reshape(sysm.shape)


def solve_steady(prob: Problem, power_per_length_W_per_m: float = 1.0, sysm: System | None = None) -> np.ndarray:
    """Steady temperature rise (K) on the nodes, shape (ny, nx), for P' in W/m."""
    sysm = sysm or assemble(prob)
    Tf = spla.spsolve(sysm.K, sysm.b_unit * power_per_length_W_per_m)
    return _full(sysm, Tf)


def heat_balance(prob: Problem, T: np.ndarray, power_per_length_W_per_m: float) -> dict:
    """Energy check: power leaving through the bottom (into the sink) and the top
    (convection) must equal the input P'. Bottom flux is read from the first row of
    vertical face conductances."""
    hx, hy = np.diff(prob.x), np.diff(prob.y)
    k_pad = np.zeros((len(prob.y) - 1, len(prob.x) + 1)); k_pad[:, 1:-1] = prob.k * hx[None, :]
    gn0 = 0.5 * (k_pad[0, :-1] + k_pad[0, 1:]) / hy[0]
    p_bottom = float(np.sum(gn0 * (T[1, :] - T[0, :])))
    wtop = np.zeros(len(prob.x)); wtop[:-1] += 0.5 * hx; wtop[1:] += 0.5 * hx
    p_top = float(np.sum(prob.domain.h_conv * wtop * 1e-6 * T[-1, :]))
    return {"p_in": power_per_length_W_per_m, "p_bottom": p_bottom, "p_top": p_top,
            "rel_error": (p_bottom + p_top - power_per_length_W_per_m) / power_per_length_W_per_m}


def interpolator(prob: Problem, T: np.ndarray):
    """(y_um, x_um) -> temperature rise, bilinear on the nodes."""
    return RegularGridInterpolator((prob.y, prob.x), T, bounds_error=False, fill_value=None)


def frequency_response(prob: Problem, freqs_hz, weight_fn, sysm: System | None = None) -> np.ndarray:
    """Small-signal response of a scalar readout to a sinusoidal heater power:
    solve (K + i w C) T = b for each frequency and return weight_fn(T_full) / weight_fn(T_dc).
    `weight_fn` maps a full (ny, nx) node temperature array to a scalar (e.g. a
    mode-weighted Delta n_eff); it must be linear in T."""
    sysm = sysm or assemble(prob)
    dc = weight_fn(_full(sysm, spla.spsolve(sysm.K, sysm.b_unit)))
    out = []
    for f in np.atleast_1d(freqs_hz):
        A = (sysm.K + sp.diags(1j * 2 * np.pi * f * sysm.C)).tocsc()
        out.append(weight_fn(_full(sysm, spla.spsolve(A, sysm.b_unit.astype(complex)))) / dc)
    return np.array(out)


def step_response(prob: Problem, t_start_s: float, t_end_s: float, weight_fn, steps_per_decade: int = 200,
                  sysm: System | None = None):
    """Turn-on step response of a linear scalar readout, normalized to its steady value,
    for a P' = 1 W/m step at t = 0 from dT = 0. Implicit (backward-Euler) stepping:
    unconditionally stable for this stiff system. The step grows with time: within each
    decade [t_k, 10 t_k] the step is fixed at 9 t_k / steps_per_decade, so dt/t stays
    between 0.0045 and 0.045 for the default 200 (first decade from 0 to t_start uses
    t_start / 50). Only one sparse LU factorization is held at a time.
    Returns (times_s, response)."""
    sysm = sysm or assemble(prob)
    dc = weight_fn(_full(sysm, spla.spsolve(sysm.K, sysm.b_unit)))
    Tf = np.zeros_like(sysm.b_unit)
    stages = [(0.0, t_start_s, t_start_s / 50)]
    t = t_start_s
    while t < t_end_s * (1 - 1e-12):
        t_next = min(10 * t, t_end_s)
        stages.append((t, t_next, 9 * t / steps_per_decade))
        t = t_next
    times, out = [], []
    for a, b, dt in stages:
        n = max(1, int(round((b - a) / dt)))
        dt = (b - a) / n
        lu = spla.splu((sp.diags(sysm.C / dt) + sysm.K).tocsc())
        for i in range(1, n + 1):
            Tf = lu.solve(sysm.C / dt * Tf + sysm.b_unit)
            times.append(a + i * dt)
            out.append(weight_fn(_full(sysm, Tf)) / dc)
        del lu
    return np.array(times), np.array(out)


def time_constant(times_s, response) -> float:
    """1/e time constant of a normalized turn-on response: first time the response reaches
    1 - 1/e (linear interpolation). For a single-pole system this is its tau."""
    target = 1 - np.exp(-1)
    times_s, response = np.asarray(times_s), np.asarray(response)
    i = int(np.argmax(response >= target))
    if response[i] < target:
        return float("nan")
    if i == 0:
        return float(times_s[0])
    t0, t1, r0, r1 = times_s[i - 1], times_s[i], response[i - 1], response[i]
    return float(t0 + (target - r0) * (t1 - t0) / (r1 - r0))


def bandwidth_3db(freqs_hz, response) -> float:
    """First frequency where |H| drops below 1/sqrt(2) (log-linear interpolation)."""
    mag = np.abs(np.asarray(response))
    f = np.asarray(freqs_hz)
    i = int(np.argmax(mag < 1 / np.sqrt(2)))
    if mag[i] >= 1 / np.sqrt(2):
        return float("nan")
    lf0, lf1 = np.log(f[i - 1]), np.log(f[i])
    m0, m1 = mag[i - 1], mag[i]
    return float(np.exp(lf0 + (1 / np.sqrt(2) - m0) * (lf1 - lf0) / (m1 - m0)))


# ---------------------------------------------------------------------------
# Cross-section builders. y = 0 is the BOX top (bottom of the Si layer).
# ---------------------------------------------------------------------------
DEFAULT_STACK = {
    "si_um": 0.22,           # SOI device layer
    "slab_um": 0.09,         # rib slab (partial etch)
    "box_um": 2.0,           # buried oxide
    "clad_top_um": 2.5,      # top-oxide surface height above the BOX top
    "substrate_um": 6.0,     # Si substrate kept in the model below the BOX; its bottom is isothermal
    "half_width_um": 20.0,   # lateral half-width (adiabatic sides)
    "h_conv": 10.0,          # W/(m^2 K)
}


def _domain(stack: dict, **grid) -> Domain:
    return Domain(x_min=-stack["half_width_um"], x_max=stack["half_width_um"],
                  y_min=-stack["box_um"] - stack["substrate_um"], y_max=stack["clad_top_um"],
                  h_conv=stack["h_conv"], **grid)


def _substrate(stack: dict) -> Rect:
    w = stack["half_width_um"]
    return Rect(-w, w, -stack["box_um"] - stack["substrate_um"], -stack["box_um"], "si_substrate", name="substrate")


def _trenches(stack: dict, trenches) -> list[Rect]:
    """Air-filled deep trenches through the top oxide and the BOX, down to the substrate.
    `trenches` = iterable of (x0, x1)."""
    return [Rect(a, b, -stack["box_um"], stack["clad_top_um"], "air", name="trench") for a, b in trenches]


def tin_heater_strip(wg_width_um=0.5, heater_width_um=2.0, heater_thickness_um=0.2, heater_gap_um=2.0,
                     heater_x_um=0.0, stack: dict | None = None, trenches=(), neighbors_x_um=(), **grid) -> Problem:
    """Si strip waveguide with a TiN heater above it (Jacques 2019 'TiN' design).
    `heater_gap_um` is the oxide thickness between the Si top and the heater bottom.
    `neighbors_x_um`: extra (unheated) strip waveguides, for thermal crosstalk."""
    s = {**DEFAULT_STACK, **(stack or {})}
    hb = s["si_um"] + heater_gap_um
    if hb + heater_thickness_um > s["clad_top_um"] + 1e-9:
        raise ValueError("heater sticks out of the top oxide")
    rects = [_substrate(s)]
    for xc in (0.0, *neighbors_x_um):
        rects.append(Rect(xc - wg_width_um / 2, xc + wg_width_um / 2, 0.0, s["si_um"], "si_wg", name="core"))
    rects.append(Rect(heater_x_um - heater_width_um / 2, heater_x_um + heater_width_um / 2, hb, hb + heater_thickness_um,
                      "tin", heater=True, name="heater"))
    rects += _trenches(s, trenches)
    extra_x = [x + d for x in neighbors_x_um for d in (-0.5, 0.5)]
    return build_problem(rects, _domain(s, **grid), extra_x=extra_x)


def doped_si_heater_rib(wg_width_um=0.5, buffer_um=0.8, doped_width_um=1.0, stack: dict | None = None,
                        trenches=(), neighbors_x_um=(), **grid) -> Problem:
    """Si rib waveguide with N++ doped heater strips in the slab on both sides (Jacques 2019
    'N++' design): slab from the rib edge out to buffer + doped width; the doped strips start
    `buffer_um` from each rib edge. Both strips dissipate (series current path)."""
    s = {**DEFAULT_STACK, **(stack or {})}
    half = wg_width_um / 2
    slab_half = half + buffer_um + doped_width_um
    rects = [_substrate(s),
             Rect(-slab_half, slab_half, 0.0, s["slab_um"], "si_slab", name="slab"),
             Rect(-half, half, 0.0, s["si_um"], "si_wg", name="core"),
             Rect(half + buffer_um, slab_half, 0.0, s["slab_um"], "si_doped", heater=True, name="heater_r"),
             Rect(-slab_half, -half - buffer_um, 0.0, s["slab_um"], "si_doped", heater=True, name="heater_l")]
    for xc in neighbors_x_um:
        rects.append(Rect(xc - half, xc + half, 0.0, s["si_um"], "si_wg", name="core"))
    rects += _trenches(s, trenches)
    return build_problem(rects, _domain(s, **grid))


def layered_sheet_heater(stack: dict | None = None, heater_y_um=2.22, heater_thickness_um=0.2, **grid) -> Problem:
    """Validation case: a heater sheet spanning the FULL width with no waveguide and no
    convection -- a 1D series of layers whose exact solution is dT = P'/W * sum(t_i / k_i)
    (see `layered_sheet_exact`)."""
    s = {**DEFAULT_STACK, **(stack or {}), "h_conv": 0.0}
    w = s["half_width_um"]
    rects = [_substrate(s), Rect(-w, w, heater_y_um, heater_y_um + heater_thickness_um, "tin", heater=True, name="sheet")]
    return build_problem(rects, _domain(s, **grid))


def layered_sheet_exact(stack: dict | None = None, heater_y_um=2.22, heater_thickness_um=0.2,
                        power_per_length_W_per_m=1.0) -> dict:
    """Exact 1D solution for `layered_sheet_heater`: all heat flows down. Heat flux
    q'' = P'/W. Temperature at y: integrate t/k from the sink to y. Inside the heater the
    flux grows linearly (uniform generation), which adds t_h/(2 k_h) at its top half."""
    s = {**DEFAULT_STACK, **(stack or {})}
    W = 2 * s["half_width_um"] * 1e-6
    qf = power_per_length_W_per_m / W
    k_si, k_ox, k_tin = (THERMAL_MATERIALS[m]["k"] for m in ("si_substrate", "sio2", "tin"))
    r_sub = s["substrate_um"] * 1e-6 / k_si
    r_ox = (s["box_um"] + heater_y_um) * 1e-6 / k_ox           # BOX + oxide up to the heater bottom
    T_heater_bottom = qf * (r_sub + r_ox)
    T_y0 = qf * (r_sub + s["box_um"] * 1e-6 / k_ox)              # at the BOX top (y = 0)
    T_heater_top = T_heater_bottom + qf * heater_thickness_um * 1e-6 / (2 * k_tin)
    return {"T_box_top": T_y0, "T_heater_bottom": T_heater_bottom, "T_heater_top": T_heater_top}
