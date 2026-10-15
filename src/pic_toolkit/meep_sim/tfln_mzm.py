"""x-cut thin-film lithium niobate (TFLN) electro-optic Mach-Zehnder modulator:
the layer that connects the optical cross-section (`tfln_waveguide.py`), the
electrode field (`tfln_electrostatics.py`), a 2D top-view FDTD check that reuses
`mzm.py` / `coupler.py` unchanged, and a traveling-wave RF bandwidth model.

Chain of physics (notebook 11 follows it section by section):

1. Pockels effect in LN, for the field along the optic axis (lab x for x-cut TE):
       Delta(1/n_e^2) = r33 E_x   =>   Delta n_e = -1/2 n_e^3 r33 E_x
   (equivalently Delta eps_xx = -n_e^4 r33 E_x).
2. The guided mode only partly overlaps the electrode field, so the modal index
   changes by
       dn_eff/dV = 1/2 n_e^3 r33 Gamma / g ,
   with Gamma the electro-optic overlap factor (Gamma=1 for a mode fully inside a
   uniform field V/g). Gamma is computed two ways (`overlap`): the common
   intensity-weighted textbook form, and the rigorous first-order perturbation
   form  dn_eff = (n_g/2) * int(Delta eps |E_x|^2) / int(eps |E|^2). A third,
   fully independent value comes from re-solving the mode in MPB with the
   field-dependent permittivity actually inserted (`delta_neff_direct`).
3. Push-pull: the GSG field points opposite ways in the two gaps, so the two
   arms change by +/- dn and the relative phase is
       Delta phi = (2 pi / lambda) * 2 |dn_eff/dV| V L   =>   V_pi L = lambda / (4 |dn_eff/dV|).
4. Traveling-wave electrode: the optical wave rides along a ~1 cm RF line;
   bandwidth is set by velocity mismatch (n_RF vs optical n_g), conductor loss and
   impedance mismatch (`eo_response`).

The 2D top-view FDTD (section 8) cannot afford a 1 cm arm, so -- exactly like
notebook 08 -- it applies an index change Delta n on a shorter arm whose
Delta n * L product equals the real device's at V_pi; it verifies the
interferometer (couplers + arms + phase-to-intensity transfer) in the
real TFLN effective indices, not the Pockels physics itself.

Only `meep` is imported here through the helper modules (no FDTD timestepping
except through `coupler.simulate_baseline` / `mzm.simulate`).
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import meep as mp
import numpy as np

from . import coupler, mzm
from . import tfln_electrostatics as es
from . import tfln_waveguide as tw

mp.verbosity(0)

# Pockels coefficient r33 of congruent LN. 30.9 pm/V is the classic bulk value used in
# most TFLN design papers; 26.9 pm/V is the recent broadband (100 MHz - 330 GHz)
# measurement (Nature Materials 2025). The notebook reports both.
R33_CLASSIC_M_PER_V = 30.9e-12
R33_NATMAT2025_M_PER_V = 26.9e-12


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def combined_params(optical: dict | None = None, electrical: dict | None = None) -> tuple[dict, dict]:
    """Merged (optical, electrical) parameter dicts that share one layer stack: the
    film / etch / rib width are taken from the optical side and copied into the
    electrical side, so the two solvers can never silently disagree on geometry."""
    op = {**tw.DEFAULT_PARAMS, **(optical or {})}
    el = {**es.DEFAULT_PARAMS, **(electrical or {})}
    for k in ("film_um", "etch_um", "rib_width_um"):
        el[k] = op[k]
    if op["cell_lateral_um"] < el["gap_um"] + 2.0:
        op["cell_lateral_um"] = el["gap_um"] + 3.0
    return op, el


# ---------------------------------------------------------------------------
# Electro-optic overlap
# ---------------------------------------------------------------------------
@dataclass
class OverlapResult:
    arm: int                    # 0 = left gap, 1 = right gap
    gamma_simple: float         # (g/V) int_LN E_x |E|^2 / int |E|^2
    gamma_rigorous: float       # Gamma that reproduces dn_dV_rigorous in the textbook formula
    dn_dV_simple: float         # 1/2 n_e^3 r33 gamma_simple / g    [1/V], signed
    dn_dV_rigorous: float       # first-order perturbation theory   [1/V], signed
    r33_m_per_V: float
    gap_um: float


def _rf_field_on_mode_grid(mode: tw.ModeResult, field: es.FieldSolution, x_center_um: float):
    """Lateral electrode field E_x [V/m per V applied] sampled on the MPB mode grid,
    with the mode (solved centered at x=0) shifted to the waveguide's real position."""
    X, Y = np.meshgrid(mode.x_um + x_center_um, mode.y_um)
    pts = np.stack([Y.ravel(), X.ravel()], axis=-1)
    return field.interpolator("Ex")(pts).reshape(X.shape) / field.voltage


def overlap(mode: tw.ModeResult, field: es.FieldSolution, optical: dict, electrical: dict,
            arm: int = 0, r33: float = R33_CLASSIC_M_PER_V) -> OverlapResult:
    m = tw.materials(optical)
    ne = m["n_e"]
    g = electrical["gap_um"] * 1e-6
    xc = es.electrode_layout(electrical)["wg_centers"][arm]
    Erf = _rf_field_on_mode_grid(mode, field, xc)                      # V/m per V
    maps = tw.eps_diag_maps(optical, mode.x_um, mode.y_um)
    ln = maps["is_ln"]
    Ix, Iy, Iz = np.abs(mode.Ex) ** 2, np.abs(mode.Ey) ** 2, np.abs(mode.Ez) ** 2
    I = Ix + Iy + Iz
    # Textbook (intensity-weighted) overlap. Sign kept: it tells which arm goes up.
    gamma_simple = g * np.sum(Erf * I * ln) / np.sum(I)
    dn_simple = -0.5 * ne ** 3 * r33 * gamma_simple / g
    # Rigorous first-order perturbation: Delta eps_xx = -n_e^4 r33 E_x inside LN only;
    # d(n_eff) = (n_g / 2) * int(Delta eps_xx |E_x|^2) / int(sum eps_ii |E_i|^2).
    num = np.sum(-ne ** 4 * r33 * Erf * Ix * ln)
    den = np.sum(maps["eps_x"] * Ix + maps["eps_y"] * Iy + maps["eps_z"] * Iz)
    dn_rig = 0.5 * mode.n_g * num / den
    gamma_rig = -dn_rig / (0.5 * ne ** 3 * r33 / g)
    return OverlapResult(arm=arm, gamma_simple=float(gamma_simple), gamma_rigorous=float(gamma_rig),
                         dn_dV_simple=float(dn_simple), dn_dV_rigorous=float(dn_rig), r33_m_per_V=r33,
                         gap_um=electrical["gap_um"])


def delta_neff_direct(field: es.FieldSolution, optical: dict, electrical: dict, voltage: float = 50.0,
                      arm: int = 0, r33: float = R33_CLASSIC_M_PER_V, resolution: int | None = None) -> dict:
    """Independent check of `overlap`: insert Delta eps_xx(x, y) = -n_e^4 r33 E_x(x, y) V
    directly into the MPB permittivity (an MPB material function over the LN blocks)
    and re-solve at +V and -V. The central difference cancels any even-order
    discretization offset; the large `voltage` only lifts the shift well above the
    solver tolerance -- the Pockels effect itself is linear in V."""
    m = tw.materials(optical)
    ne2, no2 = m["n_e"] ** 2, m["n_o"] ** 2
    xc = es.electrode_layout(electrical)["wg_centers"][arm]
    interp = field.interpolator("Ex")
    _, _, y_off = tw._mpb_cell(optical)

    def ln_fn(sign):
        def f(p):
            # MPB point -> lab (x lateral = p.y, y vertical = p.x + offset)
            Ex = interp([[p.x + y_off, p.y + xc]])[0] / field.voltage
            return mp.Medium(epsilon_diag=mp.Vector3(no2, ne2 - ne2 ** 2 * r33 * Ex * sign * voltage, no2))
        return f

    n_plus = tw.solve_mode(optical, ln_material=ln_fn(+1), resolution=resolution).n_eff
    n_minus = tw.solve_mode(optical, ln_material=ln_fn(-1), resolution=resolution).n_eff
    return {"dn_dV": (n_plus - n_minus) / (2 * voltage), "n_plus": n_plus, "n_minus": n_minus,
            "voltage": voltage}


def vpi_l_V_cm(dn_dV: float, wl_um: float) -> float:
    """Push-pull V_pi * L in V*cm: Delta phi = (2 pi / lambda) (2 |dn/dV|) V L = pi."""
    return (wl_um * 1e-6) / (4 * abs(dn_dV)) * 100.0


# ---------------------------------------------------------------------------
# Traveling-wave EO frequency response
# ---------------------------------------------------------------------------
def eo_response(f_hz, length_m: float, n_rf: float, n_g: float, alpha_np_per_m, z0_ohm: float,
                zs_ohm: float = 50.0, zl_ohm: float = 50.0) -> np.ndarray:
    """Small-signal EO response m(f), normalized to its DC value (m(0)=1).

    The line voltage is a forward wave plus the wave reflected at the load,
        V(z) = V+ ( e^{-gamma z} + Gamma_L e^{-2 gamma L} e^{+gamma z} ),
        gamma = alpha + j omega n_rf / c,  Gamma_L = (Z_L - Z0)/(Z_L + Z0),
    with V+ fixed by the source (EMF V_s, impedance Z_s) at z = 0. A photon entering
    at t0 reaches z at t0 + z n_g / c, so the phase it accumulates is
        Phi(omega) ~ int_0^L V(z) e^{+j omega n_g z / c} dz,
    integrated in closed form below. m(f) = |Phi(f)| / |Phi(0)|; Phi(0) is the
    lossless DC case, a uniform V = V_s Z_L / (Z_L + Z_s). EO "3 dB bandwidth" in
    notebook 11 is the frequency where 20 log10 m = -3 dB (the electrical-RF
    convention used in most TFLN papers)."""
    f = np.asarray(f_hz, dtype=float)
    w = 2 * np.pi * f
    a = np.broadcast_to(np.asarray(alpha_np_per_m, dtype=float), f.shape)
    L = length_m
    gam = a + 1j * w * n_rf / es.C0
    bo = w * n_g / es.C0
    GL = (zl_ohm - z0_ohm) / (zl_ohm + z0_ohm)
    eL = np.exp(-2 * gam * L)
    v_plus = 1.0 / ((1 + GL * eL) + zs_ohm / z0_ohm * (1 - GL * eL))      # per unit V_s

    def integ(k):          # int_0^L e^{k z} dz, safe at k -> 0
        k = np.asarray(k, dtype=complex)
        small = np.abs(k * L) < 1e-9
        out = np.empty_like(k)
        out[~small] = (np.exp(k[~small] * L) - 1) / k[~small]
        out[small] = L
        return out

    phi = v_plus * (integ(-gam + 1j * bo) + GL * eL * integ(gam + 1j * bo))
    phi0 = zl_ohm / (zl_ohm + zs_ohm) * L
    return np.abs(phi) / abs(phi0)


def f3db_hz(f_hz, m, level_db: float = -3.0) -> float:
    """First frequency where 20 log10 m(f) falls below `level_db` (linear interp)."""
    db = 20 * np.log10(np.maximum(m, 1e-12))
    below = np.nonzero(db < level_db)[0]
    if len(below) == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float(f_hz[0])
    return float(np.interp(level_db, [db[i], db[i - 1]], [f_hz[i], f_hz[i - 1]]))


def velocity_mismatch_f3db_hz(length_m: float, n_rf: float, n_g: float) -> float:
    """Lossless, matched-line limit: m = |sinc(u)|, u = pi f L |n_rf - n_g| / c, and
    20 log10 |sin u / u| = -3 dB at u = 1.392  =>  f = 1.392 c / (pi L |n_rf - n_g|)."""
    return 1.392 * es.C0 / (np.pi * length_m * abs(n_rf - n_g))


# ---------------------------------------------------------------------------
# 2D top-view helpers (EIM indices, supermodes, Meep parameter sets)
# ---------------------------------------------------------------------------
def eim_waveguide_neff(core_index: float, clad_index: float, width_um: float, wl_um: float,
                       resolution: int = 40, gap_um: float | None = None, band: int = 1) -> dict:
    """n_eff and n_g of the 2D top-view (EIM) waveguide -- or, with `gap_um`, of the
    band-th supermode of a symmetric pair -- from Meep's own eigenmode solver on a
    1D transverse cell, with parity TE: the SAME solver Meep's FDTD sources/monitors
    use, so it is directly comparable to the FDTD runs (pattern of
    mzi.compute_arm_dispersion)."""
    core, clad = mp.Medium(index=core_index), mp.Medium(index=clad_index)
    if gap_um is None:
        geom = [mp.Block(size=mp.Vector3(mp.inf, width_um, mp.inf), material=core)]
    else:
        c = (gap_um + width_um) / 2
        geom = [mp.Block(size=mp.Vector3(mp.inf, width_um, mp.inf), center=mp.Vector3(0, s * c), material=core)
                for s in (-1, 1)]
    cell_y = 12.0 + (gap_um or 0.0)
    sim = mp.Simulation(cell_size=mp.Vector3(0, cell_y, 0), resolution=resolution, geometry=geom,
                        default_material=clad, boundary_layers=[mp.PML(1.0, direction=mp.Y)])
    with _quiet():
        sim.init_sim()
        mode = sim.get_eigenmode(1.0 / wl_um, mp.X, mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, cell_y, 0)),
                                 band, mp.Vector3(core_index / wl_um, 0, 0), match_frequency=True, parity=mp.TE)
    f = 1.0 / wl_um
    return {"n_eff": mode.k.x / f, "n_g": 1.0 / mode.group_velocity}


def supermode_coupling_length_um(core_index, clad_index, width_um, gap_um, wl_um, resolution: int = 40) -> dict:
    """Straight-section 50:50 length of a symmetric directional coupler from its two
    supermodes: power fully transfers after L_x = lambda / (2 (n_even - n_odd)), so a
    3 dB split needs L_50 = L_x / 2. The S-bends also couple a little, so this is a
    SEED for the FDTD search, not the answer."""
    ne_ = eim_waveguide_neff(core_index, clad_index, width_um, wl_um, resolution, gap_um, band=1)["n_eff"]
    no_ = eim_waveguide_neff(core_index, clad_index, width_um, wl_um, resolution, gap_um, band=2)["n_eff"]
    lx = wl_um / (2 * (ne_ - no_))
    return {"n_even": ne_, "n_odd": no_, "L_cross_um": lx, "L_50_um": lx / 2}


def phase_response_factor(core_index, clad_index, width_um, wl_um, dn: float = 1e-3, resolution: int = 40) -> float:
    """d n_eff / d n_core of the 2D EIM waveguide (central difference). mzm.py raises
    only the CORE index by Delta n, and in this low-contrast guide part of the mode
    sits in the cladding, so n_eff moves by less than Delta n. This replaces mzm.py's
    empirically-carried `phase_factor_prior` with a computed value."""
    up = eim_waveguide_neff(core_index + dn, clad_index, width_um, wl_um, resolution)["n_eff"]
    dw = eim_waveguide_neff(core_index - dn, clad_index, width_um, wl_um, resolution)["n_eff"]
    return (up - dw) / (2 * dn)


def coupler_params(core_index: float, clad_index: float, **overrides) -> dict:
    """coupler.py parameter set in TFLN effective indices. Defaults reflect the low
    index contrast (~0.2): a wider gap, wider arm separation and longer, gentler
    S-bends than the toolkit's 2.7/1.44 couplers; see notebook 11 section 7."""
    p = {**coupler.DEFAULT_PARAMS,
         "core_index": core_index, "clad_index": clad_index,
         "wg_width_um": 1.0, "wg_height_um": tw.DEFAULT_PARAMS["film_um"],
         "gap_um": 0.8, "wide_sep_um": 6.0, "sbend_len_um": 40.0, "lead_len_um": 3.0,
         "mon_size_um": 3.0, "margin_um": 1.5, "resolution": 30, "n_freq": 41}
    p.update(overrides)
    return p


def mzm_meep_params(cp: dict, delta_n_vpi: float, phase_factor: float, **overrides) -> dict:
    """mzm.py parameter set for the TFLN 2D top-view MZM: the coupler stage `cp`
    (TFLN indices + the FDTD-confirmed 50:50 length), and a computed (not
    carried-over) phase-response factor. mzm.py itself is used unmodified."""
    keys = ("coupling_length_um", "gap_um", "wg_width_um", "wg_height_um", "core_index", "clad_index",
            "sbend_len_um", "wide_sep_um", "lead_len_um", "n_seg", "mon_size_um", "margin_um", "resolution")
    p = {**mzm.DEFAULT_PARAMS, **{k: cp[k] for k in keys},
         "delta_n_vpi": delta_n_vpi, "phase_factor_prior": phase_factor, "n_freq": 41}
    p.update(overrides)
    return p


def monitor_safe_length(cp: dict, length_um: float) -> float:
    """Nudge a coupler's `coupling_length_um` (by at most half a pixel) so the
    output mode monitors sit mid-pixel rather than exactly on a grid line.

    Why it was introduced: at L = 42.40 um (resolution 30) a +1.4e-14 um
    floating-point hairline between coupler.py's last S-bend block and its lead
    block fell exactly on a grid line, Meep's subpixel averaging left a one-pixel
    permittivity hole at the port plane, and the mode monitor there lost ~47 % of
    the power. coupler/mzi/mzm `_leads` now overlap the S-bend by one pixel, which
    removes that hole at its source, so this nudge is no longer required. It is kept
    because notebooks/11_tfln_mzm.ipynb's recorded run used it (see
    docs/troubleshooting_log.md)."""
    res = cp["resolution"]
    offset = 2 * cp["sbend_len_um"] + cp["lead_len_um"] + cp["dpml_um"] + cp["margin_um"]
    pos = (offset + length_um) * res
    return float(length_um + (np.floor(pos) + 0.5 - pos) / res)
