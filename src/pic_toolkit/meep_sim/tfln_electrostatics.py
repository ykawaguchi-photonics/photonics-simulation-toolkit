"""Quasi-static electrode field and RF line constants for the TFLN modulator
cross-section -- the electrical half of `tfln_mzm.py`.

Solves the anisotropic Laplace equation

    d/dx( eps_x dphi/dx ) + d/dy( eps_y dphi/dy ) = 0

on the lab-frame cross-section (x lateral, y vertical; same frame as
`tfln_waveguide.py`) with the metal electrodes as fixed-potential (Dirichlet)
regions and phi = 0 on a far outer boundary. Why quasi-static is enough: the
cross-section (~tens of um) is far smaller than the RF wavelength (several mm
at 100 GHz), so at every instant the transverse field has the electrostatic
shape for the instantaneous voltage -- the quasi-TEM picture notebook 11
section 9 builds the traveling-wave analysis on.

Discretization: node-based finite volumes on a NON-UNIFORM rectilinear grid
(fine around the waveguides/electrode edges, geometrically growing to the far
boundary), with every material interface on a grid line and permittivity held
per cell. The flux through each dual-cell face uses the two adjacent cells'
permittivities weighted by their share of that face, which handles
the large RF permittivity jumps (LN ~27-45 vs air 1) exactly at the
interfaces. No meep import -- pure numpy/scipy.

Electrode layout: ground-signal-ground (GSG) coplanar waveguide sitting on the
etched LN slab, one rib waveguide centered in EACH gap. The signal-to-ground
field points in opposite lateral directions in the two gaps, so a single drive
voltage produces opposite index changes in the two MZM arms -- push-pull for
free (notebook 11 section 6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator

EPS0 = 8.8541878128e-12   # F/m
C0 = 299792458.0          # m/s

DEFAULT_PARAMS = {
    # --- layer stack (must match tfln_waveguide.DEFAULT_PARAMS) ---
    "film_um": 0.6,
    "etch_um": 0.3,
    "rib_width_um": 1.0,
    "box_um": 2.0,              # buried oxide, then the Si handle below it. The RF field
                                # reaches far deeper than the optical mode, so (unlike
                                # the MPB cell) the Si handle is part of this solve; a
                                # thinner BOX pulls the RF index up toward the optical
                                # group index -- notebook 11 section 9 sweeps it
    "substrate_um": 500.0,      # high-resistivity Si handle (treated as a lossless dielectric)
    # --- electrodes (GSG, gold, on the etched slab) ---
    "gap_um": 5.0,              # signal-to-ground edge-to-edge gap (each side)
    "signal_width_um": 20.0,
    "ground_width_um": 100.0,
    "metal_um": 1.0,
    "sigma_metal_S_per_m": 4.1e7,  # evaporated gold (bulk Au is ~4.5e7 S/m)
    "metal_inset_um": 0.0,      # receding every metal surface by this much -- used ONLY
                                # by conductor_loss (Wheeler's incremental-inductance rule)
    "loss_inset_um": 0.0,       # if > 0, the grid also carries the receded-metal interface
                                # lines, so the receded and un-receded solves share ONE grid
    # --- RF relative permittivities. LN values: Nature Materials 2025
    # (BTO/LN permittivity and Pockels coefficients, MHz-330 GHz): eps_c ~ 27 along
    # the optic axis, eps_a ~ 45 perpendicular to it.
    "eps_ln_c": 27.0,           # along crystal Z = lab lateral x (x-cut)
    "eps_ln_a": 45.0,           # along crystal X = lab vertical y
    "eps_sio2": 3.9,
    "eps_si": 11.7,
    # --- grid ---
    "h_fine_um": 0.025,         # cell size in the fine window around the gaps
    "growth": 1.15,             # geometric growth ratio of cells outside the window
    "h_max_um": 50.0,
    "outer_lateral_um": 1500.0, # far boundary (phi=0) half-width
    "outer_up_um": 1500.0,      # far boundary height above the slab
}


def _graded_axis(fine_lo, fine_hi, h_fine, growth, h_max, outer_lo, outer_hi, breakpoints):
    """Node coordinates: uniform `h_fine` on [fine_lo, fine_hi], geometric growth
    outward to [outer_lo, outer_hi], with every breakpoint inserted as a node so
    each material/metal interface lies exactly on a grid line."""
    core = np.arange(fine_lo, fine_hi + 0.5 * h_fine, h_fine)
    up, h, x = [], h_fine, core[-1]
    while x < outer_hi:
        h = min(h * growth, h_max); x = x + h; up.append(min(x, outer_hi))
    down, h, x = [], h_fine, core[0]
    while x > outer_lo:
        h = min(h * growth, h_max); x = x - h; down.append(max(x, outer_lo))
    nodes = np.concatenate([down[::-1], core, up])
    nodes = np.unique(np.concatenate([nodes, [b for b in breakpoints if outer_lo <= b <= outer_hi]]))
    # drop near-duplicate nodes created by inserting breakpoints next to existing ones
    keep = np.concatenate([[True], np.diff(nodes) > 0.2 * h_fine])
    return nodes[keep]


def electrode_layout(params: dict) -> dict:
    """Lateral extents of the three electrodes and the two waveguide centers."""
    g, ws, wg = params["gap_um"], params["signal_width_um"], params["ground_width_um"]
    sig = (-ws / 2, ws / 2)
    gnd_l = (-ws / 2 - g - wg, -ws / 2 - g)
    gnd_r = (ws / 2 + g, ws / 2 + g + wg)
    wg_centers = (-(ws / 2 + g / 2), ws / 2 + g / 2)
    return {"signal": sig, "ground_left": gnd_l, "ground_right": gnd_r, "wg_centers": wg_centers}


def build_grid(params: dict):
    lay = electrode_layout(params)
    slab_t = params["film_um"] - params["etch_um"]
    xc = lay["wg_centers"]
    wr = params["rib_width_um"] / 2
    x_break = [*lay["signal"], *lay["ground_left"], *lay["ground_right"],
               xc[0] - wr, xc[0] + wr, xc[1] - wr, xc[1] + wr]
    d = params["loss_inset_um"]
    if d > 0:
        x_break += [e + sgn * d for pair in (lay["signal"], lay["ground_left"], lay["ground_right"])
                    for e, sgn in zip(pair, (1, -1))]
    edge = params["signal_width_um"] / 2 + params["gap_um"] + 2.0
    x = _graded_axis(-edge, edge, params["h_fine_um"], params["growth"], params["h_max_um"],
                     -params["outer_lateral_um"], params["outer_lateral_um"], x_break)
    y_break = [-params["box_um"] - params["substrate_um"], -params["box_um"], 0.0, slab_t,
               params["film_um"], slab_t + params["metal_um"]]
    if d > 0:
        y_break += [slab_t + d, slab_t + params["metal_um"] - d]
    y = _graded_axis(-1.0, slab_t + params["metal_um"] + 1.0, params["h_fine_um"], params["growth"],
                     params["h_max_um"], -params["box_um"] - params["substrate_um"] - params["outer_up_um"],
                     params["outer_up_um"], y_break)
    return x, y


def cell_permittivity(params: dict, x: np.ndarray, y: np.ndarray, dielectric: bool = True):
    """Per-cell (eps_x, eps_y) on the cells between nodes, shape (ny-1, nx-1).
    `dielectric=False` sets every cell to 1 (vacuum) -- the reference geometry
    for the quasi-TEM n_RF / Z0 extraction."""
    xm, ym = 0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1])
    X, Y = np.meshgrid(xm, ym)
    ex = np.ones_like(X); ey = np.ones_like(X)
    if not dielectric:
        return ex, ey
    slab_t = params["film_um"] - params["etch_um"]
    lay = electrode_layout(params)
    sub = Y < -params["box_um"]
    box = (Y >= -params["box_um"]) & (Y < 0)
    ln = (Y >= 0) & (Y < slab_t)
    for c in lay["wg_centers"]:
        ln |= (Y >= 0) & (Y < params["film_um"]) & (np.abs(X - c) < params["rib_width_um"] / 2)
    for arr, val in ((ex, params["eps_si"]), (ey, params["eps_si"])):
        arr[sub] = val
    ex[box] = ey[box] = params["eps_sio2"]
    ex[ln] = params["eps_ln_c"]
    ey[ln] = params["eps_ln_a"]
    return ex, ey


def _metal_mask(params: dict, x: np.ndarray, y: np.ndarray):
    """Node masks (ny, nx): which nodes are inside the signal / either ground."""
    slab_t = params["film_um"] - params["etch_um"]
    lay = electrode_layout(params)
    d = params["metal_inset_um"]
    X, Y = np.meshgrid(x, y)
    in_y = (Y >= slab_t + d - 1e-9) & (Y <= slab_t + params["metal_um"] - d + 1e-9)
    inside = lambda a, b: in_y & (X >= a + d - 1e-9) & (X <= b - d + 1e-9)
    return inside(*lay["signal"]), inside(*lay["ground_left"]) | inside(*lay["ground_right"])


@dataclass
class FieldSolution:
    x_um: np.ndarray            # node coordinates
    y_um: np.ndarray
    phi: np.ndarray             # (ny, nx) potential [V] for `voltage` on the signal
    xc_um: np.ndarray           # cell-center coordinates
    yc_um: np.ndarray
    Ex: np.ndarray              # (ny-1, nx-1) cell-centered field [V/m]
    Ey: np.ndarray
    voltage: float
    capacitance_F_per_m: float  # signal-to-ground C per unit length (energy method)
    dielectric: bool

    def interpolator(self, component: str = "Ex"):
        """(x_um, y_um) -> field, for overlap integrals on another (e.g. MPB) grid."""
        data = self.Ex if component == "Ex" else self.Ey
        return RegularGridInterpolator((self.yc_um, self.xc_um), data, bounds_error=False, fill_value=0.0)


def solve_field(params: dict | None = None, voltage: float = 1.0, dielectric: bool = True) -> FieldSolution:
    params = {**DEFAULT_PARAMS, **(params or {})}
    x, y = build_grid(params)
    nx, ny = len(x), len(y)
    hx, hy = np.diff(x), np.diff(y)
    ex, ey = cell_permittivity(params, x, y, dielectric)

    sig, gnd = _metal_mask(params, x, y)
    fixed = sig | gnd
    fixed[0, :] = fixed[-1, :] = fixed[:, 0] = fixed[:, -1] = True   # far boundary phi = 0
    phi_fixed = np.where(sig, voltage, 0.0)

    # Face conductances between node (j,i) and its east / north neighbor.
    # East face of node (j,i): between x_i and x_{i+1}, spanning half of cell row j-1
    # (below) and half of cell row j (above).
    ex_pad = np.zeros((ny + 1, nx - 1)); ex_pad[1:-1] = ex * hy[:, None]
    ge = 0.5 * (ex_pad[:-1] + ex_pad[1:]) / hx[None, :]          # (ny, nx-1)
    ey_pad = np.zeros((ny - 1, nx + 1)); ey_pad[:, 1:-1] = ey * hx[None, :]
    gn = 0.5 * (ey_pad[:, :-1] + ey_pad[:, 1:]) / hy[:, None]    # (ny-1, nx)

    idx = np.arange(nx * ny).reshape(ny, nx)
    rows, cols, vals = [], [], []
    diag = np.zeros((ny, nx))
    # east-west couplings
    a, b = idx[:, :-1].ravel(), idx[:, 1:].ravel(); g = ge.ravel()
    rows += [a, b]; cols += [b, a]; vals += [-g, -g]
    np.add.at(diag, (slice(None), slice(None, -1)), ge); np.add.at(diag, (slice(None), slice(1, None)), ge)
    # north-south couplings
    a, b = idx[:-1, :].ravel(), idx[1:, :].ravel(); g = gn.ravel()
    rows += [a, b]; cols += [b, a]; vals += [-g, -g]
    np.add.at(diag, (slice(None, -1), slice(None)), gn); np.add.at(diag, (slice(1, None), slice(None)), gn)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2)
    A = A + sp.diags(diag.ravel())

    free = ~fixed.ravel()
    rhs = -A[free][:, ~free] @ phi_fixed.ravel()[~free]
    phi = phi_fixed.ravel().copy()
    phi[free] = spla.spsolve(A[free][:, free].tocsc(), rhs)
    phi = phi.reshape(ny, nx)

    # Cell-centered field (um -> m): average the two edge differences of each cell.
    dpx = np.diff(phi, axis=1) / (hx[None, :] * 1e-6)            # (ny, nx-1)
    dpy = np.diff(phi, axis=0) / (hy[:, None] * 1e-6)            # (ny-1, nx)
    Ex = -0.5 * (dpx[:-1] + dpx[1:])
    Ey = -0.5 * (dpy[:, :-1] + dpy[:, 1:])
    # Energy per unit length W = 1/2 eps0 sum(eps|E|^2 dA) = 1/2 C V^2
    area = (hy[:, None] * hx[None, :]) * 1e-12
    W = 0.5 * EPS0 * np.sum((ex * Ex ** 2 + ey * Ey ** 2) * area)
    C = 2 * W / voltage ** 2
    return FieldSolution(x_um=x, y_um=y, phi=phi, xc_um=0.5 * (x[1:] + x[:-1]), yc_um=0.5 * (y[1:] + y[:-1]),
                         Ex=Ex, Ey=Ey, voltage=voltage, capacitance_F_per_m=C, dielectric=dielectric)


def cpw_capacitance_analytic(signal_width_um: float, gap_um: float, eps_r: float = 1.0) -> float:
    """Conformal-mapping capacitance per unit length of a zero-thickness coplanar
    waveguide with infinitely wide grounds in a homogeneous medium eps_r:
        C = 4 eps0 eps_r K(k) / K(k'),  k = w / (w + 2g),  k' = sqrt(1 - k^2)
    (the factor 4 = 2 for the two half-spaces x 2 for the two gaps). Used only to
    validate `solve_field` (dielectric=False, thin metal, wide grounds)."""
    from scipy.special import ellipk
    k = signal_width_um / (signal_width_um + 2 * gap_um)
    return 4 * EPS0 * eps_r * ellipk(k ** 2) / ellipk(1 - k ** 2)


def rf_line_constants(params: dict | None = None) -> dict:
    """Quasi-TEM RF index and characteristic impedance from two electrostatic solves:
        n_RF = sqrt(C / C_air),   Z0 = 1 / (c0 sqrt(C C_air)).
    Why this works: in a quasi-TEM line the per-length inductance L does not depend
    on the dielectrics, so L = 1/(c0^2 C_air) from the vacuum solve; then
    v = 1/sqrt(L C) and Z0 = sqrt(L / C). Metal (conductor) loss is NOT part of
    this solve -- see tfln_mzm.eo_response for how it is added."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    sol = solve_field(params, dielectric=True)
    sol_air = solve_field(params, dielectric=False)
    C, Ca = sol.capacitance_F_per_m, sol_air.capacitance_F_per_m
    return {"C_F_per_m": C, "C_air_F_per_m": Ca, "n_rf": float(np.sqrt(C / Ca)),
            "Z0_ohm": float(1.0 / (C0 * np.sqrt(C * Ca))), "solution": sol}


MU0 = 4e-7 * np.pi


def surface_resistance(f_hz, sigma: float):
    """Skin-effect surface resistance R_s = sqrt(pi f mu0 / sigma) = 1 / (sigma delta),
    delta = sqrt(1 / (pi f mu0 sigma)) the skin depth. This is the standard
    alpha ~ sqrt(f) conductor-loss model. It assumes the skin depth is smaller than
    the metal thickness (above ~6 GHz for 1 um of gold); below that the current
    spreads through the whole conductor, the line turns partly resistive (RC-like),
    and this model underestimates the loss -- notebook 11 flags that range."""
    f_hz = np.asarray(f_hz, dtype=float)
    return np.sqrt(np.pi * f_hz * MU0 / sigma)


def skin_depth_um(f_hz, sigma: float):
    return 1e6 * np.sqrt(1.0 / (np.pi * np.asarray(f_hz, dtype=float) * MU0 * sigma))


def conductor_loss(params: dict | None = None, f_hz=None, inset_um: float = 0.05) -> dict:
    """Conductor (ohmic) loss of the CPW from Wheeler's incremental-inductance rule:
        alpha_c = R_s / (2 Z0 mu0) * dL/dn,
    where dL/dn is how much the per-length inductance grows when every metal surface
    recedes by dn (the field penetrates the skin). L comes from vacuum solves alone,
    L = 1/(c0^2 C_air), so dL/dn is a finite difference of two electrostatic solves
    on one shared grid. Returns alpha in Np/m and dB/cm at each f, plus dL/dn.
    Dielectric loss (LN, SiO2) and radiation/substrate leakage are not included."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    f_hz = np.asarray(f_hz if f_hz is not None else np.linspace(1e9, 200e9, 200), dtype=float)
    base = {**params, "loss_inset_um": inset_um}
    Ca0 = solve_field({**base, "metal_inset_um": 0.0}, dielectric=False).capacitance_F_per_m
    Ca1 = solve_field({**base, "metal_inset_um": inset_um}, dielectric=False).capacitance_F_per_m
    dL_dn = (1.0 / (C0 ** 2 * Ca1) - 1.0 / (C0 ** 2 * Ca0)) / (inset_um * 1e-6)   # H/m per m
    Z0 = rf_line_constants(params)["Z0_ohm"]
    Rs = surface_resistance(f_hz, params["sigma_metal_S_per_m"])
    alpha_np = Rs / (2 * Z0 * MU0) * dL_dn
    return {"f_hz": f_hz, "alpha_np_per_m": alpha_np, "alpha_db_per_cm": alpha_np * 8.686 / 100,
            "dL_dn_H_per_m2": dL_dn, "Z0_ohm": Z0}
