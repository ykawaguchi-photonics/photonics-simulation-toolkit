"""Cross-section (MPB) mode solver for Si strip and rib waveguides in SiO2 -- the optical
half of the si_active track.

**Deviation from the toolkit-wide modeling convention.** The `si/` track uses a 2D top
view with an effective core index of 2.7. Thermo-optic and electro-optic tuning act
through the real cross-section (where the heat and the carriers sit relative to the
mode), so this module solves the REAL cross-section with bulk indices:

- Si: Frey, Leviton & Madison, Proc. SPIE 6273 (2006), Table 2 at 295 K, linear
  interpolation in wavelength (1.1-1.5 um); dn/dT from their Table 4.
- SiO2: Malitson (1965), from `tfln_waveguide.silica_index`.

Each index is evaluated once at the design wavelength and held constant (the toolkit's
non-dispersive convention); the group index comes from MPB's group velocity, i.e. the
WAVEGUIDE dispersion only. The material dispersion of Si adds about 0.1-0.2 to n_g at
1.31 um; `group_index_material_dispersive` returns the full value by finite differences
of n_eff solved with the wavelength-dependent indices.

**Coordinates.** Lab frame in every public function: x lateral, y vertical with y = 0 at
the BOX top (bottom of the Si layer), z propagation. As in `tfln_waveguide.py`, the MPB
lattice is rotated (MPB x = lab y, MPB y = lab x) so that MPB's mirror plane (its own y)
is the lateral mirror of the waveguide; the TE family (lateral E, even) is then MPB
`ODD_Y`, passed explicitly -- see CLAUDE.md section 5.

Only meep / meep.mpb are imported (no FDTD time stepping).
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import meep as mp
import numpy as np
from meep import mpb

from ..meep_sim.tfln_waveguide import silica_index

mp.verbosity(0)

TE_PARITY = mp.ODD_Y
TM_PARITY = mp.EVEN_Y

# Frey et al. 2006, 295 K. Table 2 (absolute index) and Table 4 (dn/dT, 1/K).
_FREY_WL = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
_FREY_N = np.array([3.54394, 3.52253, 3.50616, 3.49335, 3.48314])
_FREY_DNDT = np.array([2.15e-4, 2.01e-4, 1.94e-4, 1.90e-4, 1.87e-4])
SIO2_DNDT = 1.0e-5   # Jacques et al. 2019 Table 3; its share of Delta n_eff is reported, not assumed small


def si_index(wl_um: float) -> float:
    if not (_FREY_WL[0] <= wl_um <= 1.6):
        raise ValueError("Si index table covers 1.1-1.5 um (linear extrapolation allowed to 1.6)")
    if wl_um <= _FREY_WL[-1]:
        return float(np.interp(wl_um, _FREY_WL, _FREY_N))
    s = (_FREY_N[-1] - _FREY_N[-2]) / 0.1
    return float(_FREY_N[-1] + s * (wl_um - _FREY_WL[-1]))


def si_dndT(wl_um: float) -> float:
    if wl_um <= _FREY_WL[-1]:
        return float(np.interp(wl_um, _FREY_WL, _FREY_DNDT))
    s = (_FREY_DNDT[-1] - _FREY_DNDT[-2]) / 0.1
    return float(_FREY_DNDT[-1] + s * (wl_um - _FREY_WL[-1]))


DEFAULT_PARAMS = {
    "wl0_um": 1.31,
    "wg_width_um": 0.40,      # chosen in notebook si_active/01 (single-mode TE at 1.31 um)
    "si_um": 0.22,
    "slab_um": 0.0,           # 0 = strip; 0.09 = standard rib
    "slab_half_width_um": 2.0,  # rib slab lateral half-extent (ignored for strip)
    "box_cell_um": 1.5,       # oxide kept below y = 0 inside the MPB cell
    "top_cell_um": 1.5,       # oxide kept above the Si top inside the MPB cell
    "cell_lateral_um": 4.0,
    "resolution": 64,         # px/um; convergence checked in notebook si_active/01
    "n_si": None,             # override (benchmark at 1.55 um uses Jacques' 3.476)
    "n_ox": None,
}


def materials(params: dict) -> dict:
    wl = params["wl0_um"]
    return {"n_si": params["n_si"] or si_index(wl), "n_ox": params["n_ox"] or silica_index(wl)}


def _cell(params: dict):
    sv = params["box_cell_um"] + params["si_um"] + params["top_cell_um"]
    y_offset = -params["box_cell_um"] + sv / 2      # lab y = mpb_x + y_offset
    sl = max(params["cell_lateral_um"], 2 * params["slab_half_width_um"] + 1.0 if params["slab_um"] > 0 else 0)
    return sv, sl, y_offset


def build_cross_section(params: dict, n_si: float, n_ox: float) -> tuple[list, mp.Medium]:
    sv, sl, y0 = _cell(params)
    si = mp.Medium(index=n_si)
    blocks = []
    if params["slab_um"] > 0:
        blocks.append(mp.Block(size=mp.Vector3(params["slab_um"], 2 * params["slab_half_width_um"], mp.inf),
                               center=mp.Vector3(params["slab_um"] / 2 - y0, 0), material=si))
    blocks.append(mp.Block(size=mp.Vector3(params["si_um"], params["wg_width_um"], mp.inf),
                           center=mp.Vector3(params["si_um"] / 2 - y0, 0), material=si))
    return blocks, mp.Medium(index=n_ox)


def lab_grid(params: dict, shape: tuple):
    sv, sl, y0 = _cell(params)
    nv, nl = shape
    y = -sv / 2 + np.arange(nv) * sv / nv + y0
    x = -sl / 2 + np.arange(nl) * sl / nl
    return x, y


def si_mask(params: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Analytic (un-averaged) Si region on a lab grid, shape (len(y), len(x))."""
    X, Y = np.meshgrid(x, y)
    core = (np.abs(X) <= params["wg_width_um"] / 2) & (Y >= 0) & (Y <= params["si_um"])
    if params["slab_um"] > 0:
        core |= (np.abs(X) <= params["slab_half_width_um"]) & (Y >= 0) & (Y <= params["slab_um"])
    return core


@dataclass
class ModeResult:
    wl_um: float
    n_eff: float
    n_g: float                  # waveguide-dispersion group index (constant material indices)
    x_um: np.ndarray
    y_um: np.ndarray
    E2: np.ndarray              # |E|^2, shape (len(y), len(x)), arbitrary normalization
    is_si: np.ndarray
    n_si: float
    n_ox: float
    te_fraction: float          # sum|E_lateral|^2 / sum|E|^2
    resolution: int


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def solve_mode(params: dict | None = None, band: int = 1, parity=TE_PARITY, n_si: float | None = None,
               n_ox: float | None = None, resolution: int | None = None) -> ModeResult:
    """Mode `band` of the given parity at fixed frequency (MPB find_k)."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    m = materials(params)
    n_si = n_si or m["n_si"]; n_ox = n_ox or m["n_ox"]
    res = resolution or params["resolution"]
    wl = params["wl0_um"]; f = 1.0 / wl
    sv, sl, _ = _cell(params)
    geom, default = build_cross_section(params, n_si, n_ox)
    ms = mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(sv, sl, 0)), geometry=geom,
                        default_material=default, resolution=res, num_bands=band)
    with _quiet():
        k = ms.find_k(parity, f, band, band, mp.Vector3(0, 0, 1), 1e-9, 0.9 * n_si * f, n_ox * f, n_si * f)[0]
        vg = ms.compute_group_velocity_component(mp.Vector3(0, 0, 1))[band - 1]
        E = np.asarray(ms.get_efield(band, bloch_phase=False)[:, :, 0, :])
    Ey, Ex, Ez = E[..., 0], E[..., 1], E[..., 2]          # MPB (x,y,z) = lab (y,x,z)
    E2 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2
    x, y = lab_grid(params, E2.shape)
    return ModeResult(wl_um=wl, n_eff=float(k / f), n_g=float(1.0 / vg), x_um=x, y_um=y, E2=E2,
                      is_si=si_mask(params, x, y), n_si=n_si, n_ox=n_ox,
                      te_fraction=float(np.sum(np.abs(Ex) ** 2) / np.sum(E2)), resolution=res)


def guided_te_bands(params: dict | None = None, max_bands: int = 3) -> list[float]:
    """n_eff of every guided TE band (n_eff above the oxide index), up to max_bands."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    out = []
    for b in range(1, max_bands + 1):
        try:
            m = solve_mode(params, band=b)
        except Exception:
            break
        if m.n_eff <= m.n_ox + 1e-4:
            break
        out.append(m.n_eff)
    return out


def perturbation_dneff(mode: ModeResult, dn_si_map: np.ndarray, dn_ox_map: np.ndarray) -> dict:
    """First-order Delta n_eff at FIXED frequency for a small index change Delta n(x, y):

        Delta n_eff = n_g * sum(n Delta n |E|^2) / sum(eps |E|^2)

    (Delta eps = 2 n Delta n; MPB's perturbation formula Delta w / w = -1/2 <E|Delta eps|E>/<E|eps|E>
    at fixed k, converted to fixed w with the group velocity: Delta k = -Delta w / v_g,
    hence the factor n_g, not n_eff). The maps are on the mode's lab grid. Returns the total
    and the Si / oxide shares."""
    eps = np.where(mode.is_si, mode.n_si ** 2, mode.n_ox ** 2)
    den = np.sum(eps * mode.E2)
    num_si = np.sum(np.where(mode.is_si, mode.n_si * dn_si_map, 0.0) * mode.E2)
    num_ox = np.sum(np.where(mode.is_si, 0.0, mode.n_ox * dn_ox_map) * mode.E2)
    # A complex map (a phasor temperature in the frequency response) gives a complex
    # Delta n_eff: keep it complex, never cast it to float.
    dneff = mode.n_g * (num_si + num_ox) / den
    total = num_si + num_ox
    return {"dneff": complex(dneff) if np.iscomplexobj(dneff) else float(dneff),
            "si_share": float(np.real(num_si / total)) if total else float("nan"),
            "confinement_si": float(np.sum(np.where(mode.is_si, eps * mode.E2, 0)) / den)}


def thermo_optic_dneff(mode: ModeResult, temperature_fn, dndT_si: float, dndT_ox: float = SIO2_DNDT) -> dict:
    """Delta n_eff for a temperature-rise field. `temperature_fn((y, x) points) -> dT`
    (e.g. thermal.interpolator). Uses `perturbation_dneff`."""
    X, Y = np.meshgrid(mode.x_um, mode.y_um)
    dT = np.asarray(temperature_fn(np.column_stack([Y.ravel(), X.ravel()]))).reshape(X.shape)
    return {**perturbation_dneff(mode, dndT_si * dT, dndT_ox * dT), "dT_peak_in_si": float(np.abs(dT[mode.is_si]).max())}


def group_index_material_dispersive(params: dict | None = None, dwl_um: float = 0.01) -> float:
    """Group index including Si and SiO2 material dispersion, n_g = n_eff - lambda dn_eff/dlambda,
    by central differences of n_eff solved with the wavelength-dependent indices."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    wl = params["wl0_um"]
    n = {}
    for s in (-1, 0, 1):
        w = wl + s * dwl_um
        n[s] = solve_mode({**params, "wl0_um": w, "n_si": None, "n_ox": None}).n_eff
    return float(n[0] - wl * (n[1] - n[-1]) / (2 * dwl_um))
