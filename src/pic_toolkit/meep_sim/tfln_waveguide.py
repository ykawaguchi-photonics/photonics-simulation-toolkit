"""Cross-section (MPB) mode solver for an x-cut thin-film lithium niobate (TFLN)
rib waveguide -- the optical half of `tfln_mzm.py`'s electro-optic modulator.

**Deviation from the toolkit-wide modeling convention (see README "Modeling
assumptions").** Every other module except `grating_coupler.py` simulates a 2D
top view with a single isotropic effective index (`core_index=2.7`). An
electro-optic modulator cannot be modeled that way: its figure of merit (the
overlap between the optical mode and the electrode field inside the
electro-optic crystal) only exists in the real cross-section. This module
therefore:

- uses REAL bulk indices: lithium niobate (Zelmon et al. 1997 Sellmeier,
  congruent LN) and fused silica (Malitson 1965), each evaluated ONCE at
  `wl0_um` and held constant across the band -- still non-dispersive, like
  every other module;
- models lithium niobate's UNIAXIAL anisotropy with
  `mp.Medium(epsilon_diag=...)` -- the first anisotropic material in the
  toolkit;
- solves the 2D cross-section with MPB (`meep.mpb.ModeSolver`), not FDTD.

The 2D top-view FDTD that `tfln_mzm.py` runs afterwards uses an
effective-index (EIM) reduction of this cross-section (`eim_indices`), whose
own error against the MPB result is measured in notebooks/11_tfln_mzm.ipynb
rather than assumed small.

**Coordinates.** Lab frame used everywhere in this module's public API and in
the notebook: `x` = lateral (in the chip plane, across the waveguide), `y` =
vertical (y=0 at the buried-oxide / LN interface, LN above), `z` =
propagation. x-cut LN propagating along crystal Y means:

    lab x (lateral)     <-> crystal Z (optic / c axis)  -> n_e, r33
    lab y (vertical)    <-> crystal X                   -> n_o
    lab z (propagation) <-> crystal Y                   -> n_o

so the TE mode (E mostly lateral) sees n_e and, with lateral electrodes, the
largest Pockels coefficient r33.

MPB internally can only exploit a mirror symmetry along ITS OWN y axis, and
this cross-section is mirror-symmetric only laterally (substrate below, air
above). So the MPB lattice is ROTATED relative to the lab frame:
MPB x = lab y (vertical), MPB y = lab x (lateral). Every public function
converts back, so callers never see MPB's axes.

**TE/TM -- a sharper version of the toolkit's recurring gotcha (CLAUDE.md
section 5).** In x-cut TFLN the fundamental TE and TM modes are nearly
degenerate (n_eff differ by ~1e-3 at this module's default geometry), so a
parity-free solve (`mp.NO_PARITY`) can land on either. TE is the mode whose
dominant lateral E-field is EVEN under the lateral mirror -- which, because
the mirror also flips the lateral field component's sign, MPB labels
`mp.ODD_Y` in this rotated lattice. Every solve here passes that parity
explicitly and reports the TE fraction so it can be checked, never assumed.

Only `meep` / `meep.mpb` are imported (no FDTD timestepping happens here).
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import meep as mp
import numpy as np
from meep import mpb
from scipy.optimize import brentq

from ..params import GLOBAL_PARAMS

mp.verbosity(0)

# MPB parity that selects the physical TE family in this module's rotated lattice
# (see module docstring) -- one named constant so no call site re-derives it.
TE_PARITY = mp.ODD_Y
TM_PARITY = mp.EVEN_Y


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
# Materials -- Sellmeier models evaluated once at wl0 (non-dispersive across the
# band, the toolkit-wide convention).
# ---------------------------------------------------------------------------
def _sellmeier(wl_um: float, B, C) -> float:
    l2 = wl_um ** 2
    return float(np.sqrt(1.0 + sum(b * l2 / (l2 - c) for b, c in zip(B, C))))


def ln_ordinary_index(wl_um: float) -> float:
    """Congruent LiNbO3, ordinary ray (Zelmon, Small & Jundt, JOSA B 14, 3319, 1997)."""
    return _sellmeier(wl_um, (2.6734, 1.2290, 12.614), (0.01764, 0.05914, 474.60))


def ln_extraordinary_index(wl_um: float) -> float:
    """Congruent LiNbO3, extraordinary ray (Zelmon et al. 1997)."""
    return _sellmeier(wl_um, (2.9804, 0.5981, 8.9543), (0.02047, 0.0666, 416.08))


def silica_index(wl_um: float) -> float:
    """Fused silica (Malitson, JOSA 55, 1205, 1965)."""
    return _sellmeier(wl_um, (0.6961663, 0.4079426, 0.8974794),
                      (0.0684043 ** 2, 0.1162414 ** 2, 9.896161 ** 2))


# ---------------------------------------------------------------------------
# User-adjustable parameters.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "wl_min_um": GLOBAL_PARAMS["wl_min_um"],
    "wl_max_um": GLOBAL_PARAMS["wl_max_um"],
    "wl0_um": 1.35,              # same center wavelength as mzm.py / coupler.py
    # --- layer stack (typical commercial x-cut TFLN-on-insulator) ---
    "film_um": 0.6,              # LN film thickness (rib height)
    "etch_um": 0.3,              # rib etch depth -> slab thickness = film - etch
    "rib_width_um": 1.0,         # rib top width; vertical sidewalls (simplification)
    "box_um": 2.0,               # buried oxide kept INSIDE the MPB cell (the real
                                 # wafer's BOX is ~4.7 um; the TE mode's tail is
                                 # negligible ~2 um below the film -- checked in
                                 # the notebook by comparing against a deeper box)
    "air_um": 2.0,               # air cladding above the film inside the MPB cell
    "cell_lateral_um": 8.0,      # MPB cell width (must exceed the electrode gap so the
                                 # mode tail at the electrode edge can be read off)
    "resolution": 32,            # MPB pixels/um; converged value chosen in notebook 11
}


def materials(params: dict) -> dict:
    """Constant (wl0-evaluated) optical indices used by every solve."""
    wl0 = params["wl0_um"]
    return {"n_o": ln_ordinary_index(wl0), "n_e": ln_extraordinary_index(wl0),
            "n_sio2": silica_index(wl0)}


def ln_medium(params: dict, swap_axes: bool = False) -> mp.Medium:
    """Uniaxial LN as an MPB medium, in MPB's rotated axes (x=vertical,
    y=lateral, z=propagation): lateral = crystal c-axis -> n_e^2.
    `swap_axes=True` puts n_o on the lateral axis instead -- used only by the
    notebook's axis sanity check (the TE n_eff must move when it is swapped)."""
    m = materials(params)
    lat = m["n_o"] ** 2 if swap_axes else m["n_e"] ** 2
    return mp.Medium(epsilon_diag=mp.Vector3(m["n_o"] ** 2, lat, m["n_o"] ** 2))


def _mpb_cell(params: dict):
    """(size_vertical, size_lateral, vertical offset) of the MPB cell. The cell
    spans lab y in [-box_um, film_um + air_um]; MPB's cell is centered on 0, so
    lab y = mpb_x + y_offset."""
    sv = params["box_um"] + params["film_um"] + params["air_um"]
    y_offset = -params["box_um"] + sv / 2
    return sv, params["cell_lateral_um"], y_offset


def build_cross_section(params: dict, swap_axes: bool = False, ln_material=None) -> list:
    """Layer stack as MPB Blocks, in MPB coordinates. Later blocks override earlier
    ones (same layering idiom as grating_coupler.build_geometry): oxide below
    y=0, a full-width LN slab of thickness film-etch, then the rib. `ln_material`
    overrides the LN medium (tfln_mzm uses this to inject a field-dependent
    Pockels perturbation as an MPB material function)."""
    m = materials(params)
    sv, sl, y0 = _mpb_cell(params)
    ln = ln_material if ln_material is not None else ln_medium(params, swap_axes)
    slab_t = params["film_um"] - params["etch_um"]
    to_mpb = lambda y: y - y0  # lab vertical -> MPB x
    return [
        mp.Block(size=mp.Vector3(params["box_um"], mp.inf, mp.inf),
                 center=mp.Vector3(to_mpb(-params["box_um"] / 2), 0), material=mp.Medium(index=m["n_sio2"])),
        mp.Block(size=mp.Vector3(slab_t, mp.inf, mp.inf),
                 center=mp.Vector3(to_mpb(slab_t / 2), 0), material=ln),
        mp.Block(size=mp.Vector3(params["film_um"], params["rib_width_um"], mp.inf),
                 center=mp.Vector3(to_mpb(params["film_um"] / 2), 0), material=ln),
    ]


def lab_grid(params: dict, shape: tuple):
    """Lab-frame coordinate vectors (x lateral, y vertical) of an MPB field array of
    `shape` = (n_vertical, n_lateral). MPB samples x_i = -S/2 + i*S/N, where N is
    the array's own size -- MPB rounds S*resolution up to an FFT-friendly N, so N
    must be read off the array, never recomputed from the resolution."""
    sv, sl, y0 = _mpb_cell(params)
    nv, nl = shape
    y = -sv / 2 + np.arange(nv) * sv / nv + y0
    x = -sl / 2 + np.arange(nl) * sl / nl
    return x, y


def eps_diag_maps(params: dict, x: np.ndarray, y: np.ndarray, swap_axes: bool = False) -> dict:
    """Analytic (un-averaged) optical relative-permittivity maps on a lab grid,
    shape (len(y), len(x)): eps_x (lateral), eps_y (vertical), eps_z, plus a
    boolean `is_ln` mask -- the only region where the Pockels effect acts."""
    m = materials(params)
    X, Y = np.meshgrid(x, y)
    slab_t = params["film_um"] - params["etch_um"]
    in_slab = (Y >= 0) & (Y < slab_t)
    in_rib = (Y >= 0) & (Y < params["film_um"]) & (np.abs(X) <= params["rib_width_um"] / 2)
    is_ln = in_slab | in_rib
    is_ox = Y < 0
    base = np.where(is_ox, m["n_sio2"] ** 2, 1.0)
    lat = m["n_o"] ** 2 if swap_axes else m["n_e"] ** 2
    eps_x = np.where(is_ln, lat, base)
    eps_y = np.where(is_ln, m["n_o"] ** 2, base)
    eps_z = np.where(is_ln, m["n_o"] ** 2, base)
    return {"eps_x": eps_x, "eps_y": eps_y, "eps_z": eps_z, "is_ln": is_ln}


@dataclass
class ModeResult:
    wl_um: float
    n_eff: float
    n_g: float
    x_um: np.ndarray            # lateral grid
    y_um: np.ndarray            # vertical grid
    Ex: np.ndarray              # lab lateral E, shape (len(y), len(x)), complex
    Ey: np.ndarray              # lab vertical E
    Ez: np.ndarray              # lab longitudinal E
    te_fraction: float          # sum|Ex|^2 / sum|E|^2 -- ~1 for a genuine TE mode
    parity: str
    resolution: int


def solve_mode(params: dict | None = None, wl_um: float | None = None, band: int = 1,
               parity=TE_PARITY, resolution: int | None = None, swap_axes: bool = False,
               ln_material=None, num_bands: int | None = None) -> ModeResult:
    """Fundamental (band=1) mode of the rib at fixed frequency: MPB `find_k` for the
    propagation constant, then the E-field and MPB's own group velocity.

    `parity` defaults to TE (see module docstring); pass TM_PARITY or
    mp.NO_PARITY only to demonstrate the near-degeneracy."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    wl = wl_um or params["wl0_um"]
    res = resolution or params["resolution"]
    m = materials(params)
    f = 1.0 / wl
    sv, sl, _ = _mpb_cell(params)
    ms = mpb.ModeSolver(
        geometry_lattice=mp.Lattice(size=mp.Vector3(sv, sl, 0)),
        geometry=build_cross_section(params, swap_axes=swap_axes, ln_material=ln_material),
        resolution=res, num_bands=num_bands or band,
    )
    n_hi = max(m["n_e"], m["n_o"])
    with _quiet():
        ks = ms.find_k(parity, f, band, band, mp.Vector3(0, 0, 1), 1e-8,
                       0.95 * n_hi * f, 1.0 * f, 1.0 * n_hi * f)
        k = ks[0]
        vg = ms.compute_group_velocity_component(mp.Vector3(0, 0, 1))[band - 1]
        E = ms.get_efield(band, bloch_phase=False)[:, :, 0, :]
    # MPB arrays are indexed (mpb_x, mpb_y) = (vertical, lateral) -- already the
    # (len(y), len(x)) lab layout; MPB components (x, y, z) = lab (y, x, z).
    Ey_lab, Ex_lab, Ez_lab = E[..., 0], E[..., 1], E[..., 2]
    x, y = lab_grid(params, Ex_lab.shape)
    p = [np.sum(np.abs(c) ** 2) for c in (Ex_lab, Ey_lab, Ez_lab)]
    pname = {TE_PARITY: "TE (MPB ODD_Y)", TM_PARITY: "TM (MPB EVEN_Y)"}.get(parity, "NO_PARITY")
    return ModeResult(wl_um=wl, n_eff=float(k / f), n_g=float(1.0 / vg), x_um=x, y_um=y,
                      Ex=np.asarray(Ex_lab), Ey=np.asarray(Ey_lab), Ez=np.asarray(Ez_lab), te_fraction=float(p[0] / sum(p)),
                      parity=pname, resolution=res)


def mode_intensity_at(mode: ModeResult, x_um: float, y_um: float) -> float:
    """|E|^2 at a lab point, normalized to the mode's peak |E|^2 -- the notebook's
    proxy for optical absorption by a metal electrode edge placed there."""
    I = np.abs(mode.Ex) ** 2 + np.abs(mode.Ey) ** 2 + np.abs(mode.Ez) ** 2
    ix = int(np.argmin(np.abs(mode.x_um - x_um)))
    iy = int(np.argmin(np.abs(mode.y_um - y_um)))
    return float(I[iy, ix] / I.max())


# ---------------------------------------------------------------------------
# Effective-index (EIM) reduction for the 2D top-view FDTD.
# ---------------------------------------------------------------------------
def slab_te_neff(n_film: float, t_um: float, n_sub: float, n_cover: float, wl_um: float) -> float:
    """Fundamental TE mode (E parallel to the layers) of an asymmetric 3-layer slab,
    from the standard dispersion relation
        tan(kappa t) = kappa (gs + gc) / (kappa^2 - gs gc),
    kappa = k0 sqrt(n_f^2 - n^2), gs/gc = k0 sqrt(n^2 - n_s/c^2). Only the in-plane
    (lateral) permittivity enters a TE slab mode, so n_film = n_e for x-cut TE."""
    k0 = 2 * np.pi / wl_um

    def f(n):
        kap = k0 * np.sqrt(n_film ** 2 - n ** 2)
        gs = k0 * np.sqrt(n ** 2 - n_sub ** 2)
        gc = k0 * np.sqrt(n ** 2 - n_cover ** 2)
        # phase form, continuous on the fundamental-mode branch: kappa t = atan(gs/kap) + atan(gc/kap)
        return kap * t_um - np.arctan(gs / kap) - np.arctan(gc / kap)

    lo, hi = max(n_sub, n_cover) + 1e-9, n_film - 1e-9
    return float(brentq(f, lo, hi))


def eim_indices(params: dict | None = None, wl_um: float | None = None) -> dict:
    """Core / cladding indices for the 2D top view: the TE slab index of the full
    film (rib region) and of the etched slab (everywhere else)."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    wl = wl_um or params["wl0_um"]
    m = materials(params)
    core = slab_te_neff(m["n_e"], params["film_um"], m["n_sio2"], 1.0, wl)
    clad = slab_te_neff(m["n_e"], params["film_um"] - params["etch_um"], m["n_sio2"], 1.0, wl)
    return {"core_index": core, "clad_index": clad}


def group_index_dispersive(params: dict | None = None, dwl_um: float = 0.01, resolution: int | None = None) -> dict:
    """Group index INCLUDING lithium niobate / silica material dispersion,
        n_g = n_eff - lambda d(n_eff)/d(lambda),
    from two MPB solves at wl0 -/+ dwl with every Sellmeier index re-evaluated at
    that wavelength. Everything else in this module holds the indices fixed at wl0
    (the toolkit's non-dispersive convention), which drops LN's own material
    dispersion from n_g; the RF velocity-matching analysis (notebook 11 section 9)
    is the one place where that omission would change a conclusion, so this
    single quantity is computed dispersively and labeled as such."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    wl0 = params["wl0_um"]
    n = {}
    for s in (-1, 1):
        wl = wl0 + s * dwl_um
        n[s] = solve_mode({**params, "wl0_um": wl}, wl_um=wl, resolution=resolution).n_eff
    n0 = solve_mode(params, resolution=resolution)
    dn_dwl = (n[1] - n[-1]) / (2 * dwl_um)
    return {"n_eff": n0.n_eff, "n_g_dispersive": n0.n_eff - wl0 * dn_dwl,
            "n_g_nondispersive": n0.n_g, "dn_dwl_per_um": dn_dwl}


def guided_modes(params: dict | None = None, bands_per_parity: int = 3, resolution: int | None = None) -> list:
    """Every mode (up to `bands_per_parity` in each mirror family) that is actually
    GUIDED by the rib, i.e. n_eff above the etched slab's own TE index (anything
    below it leaks sideways into the slab). Each entry: dict(n_eff, te_fraction,
    family). Used for the single-mode width choice.

    The two mirror families are solved SEPARATELY. TE1 (lateral E-field odd across
    the rib) is EVEN under MPB's mirror -- the same family as TM0 -- so a TE-parity
    (ODD_Y) solve alone would miss it; and a parity-free solve is unreliable here
    because TE0/TM0 are nearly degenerate and its root search can land on either.
    Each guided band is then labeled by its TE fraction."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    n_slab = eim_indices(params)["clad_index"]
    out = []
    for parity, fam in ((TE_PARITY, "ODD_Y"), (TM_PARITY, "EVEN_Y")):
        for b in range(1, bands_per_parity + 1):
            m = solve_mode(params, band=b, num_bands=b, parity=parity, resolution=resolution)
            if m.n_eff > n_slab:
                out.append({"n_eff": m.n_eff, "te_fraction": m.te_fraction, "family": fam, "band": b})
    return sorted(out, key=lambda d: -d["n_eff"])
