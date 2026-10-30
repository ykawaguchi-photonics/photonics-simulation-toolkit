"""1D finite-difference mode solver for the toolkit's 2D effective-index
cross-section (one waveguide or a symmetric pair of waveguides).

Meep-free. Solves Meep's 2D TE family (Hz out of plane, E in plane): Hz and
(1/eps) dHz/dy are continuous across each interface, so the operator is

    eps d/dy[(1/eps) dH/dy] + k0^2 eps H = beta^2 H,

discretized on a staggered grid and symmetrized. At the toolkit's default
cross-section (w = 0.5 um, n = 2.7 / 1.44, 1.35 um) it gives n_eff = 2.4184 and
n_g = 2.928, matching MPB's 2.4148 / 2.9289 (si/notebooks/07_mzi.ipynb).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

N_CORE = 2.7
N_CLAD = 1.44


def slab_modes(gap, width=0.5, n_core=N_CORE, n_clad=N_CLAD, wl=1.35, dy=0.001, span=6.0, n_modes=2):
    """Guided TE (Hz) modes of one slab (gap=None) or of two identical slabs separated by
    an edge-to-edge `gap` (um). Returns (n_eff array sorted descending, y, Hz profiles,
    eps profile); profiles are normalized to max |Hz| = 1."""
    y = np.arange(-span / 2, span / 2, dy)
    k0 = 2 * np.pi / wl

    def eps(yy):
        if gap is None:
            inside = np.abs(yy) <= width / 2
        else:
            inside = np.abs(np.abs(yy) - (gap + width) / 2) <= width / 2
        return np.where(inside, n_core**2, n_clad**2)

    e = eps(y)
    inv_half = 1 / eps(y + dy / 2)
    main = -(inv_half + np.roll(inv_half, 1)) / dy**2
    off = inv_half[:-1] / dy**2
    K = diags([off, main, off], [-1, 0, 1])
    s = np.sqrt(e)
    A = diags(s) @ K @ diags(s) + diags(k0**2 * e)
    vals, vecs = eigsh(A, k=n_modes, which="LA")
    order = np.argsort(-vals)
    n_eff = np.sqrt(vals[order]) / k0
    H = s[:, None] * vecs[:, order]  # undo the symmetrizing scaling
    H /= np.abs(H).max(axis=0)
    return n_eff, y, H, e


def n_eff_single(wl=1.35, **kw):
    """Effective index of one waveguide."""
    return float(slab_modes(None, wl=wl, n_modes=1, **kw)[0][0])


def group_index(wl=1.35, dwl=0.005, **kw):
    """n_g = n_eff - wl dn_eff/dwl (central difference)."""
    n0 = n_eff_single(wl, **kw)
    dn = (n_eff_single(wl + dwl, **kw) - n_eff_single(wl - dwl, **kw)) / (2 * dwl)
    return float(n0 - wl * dn)


def dneff_dn_core(wl=1.35, dn=1e-3, n_core=N_CORE, **kw):
    """Response of the mode index to a change of the core index, dn_eff/dn_core."""
    return float((n_eff_single(wl, n_core=n_core + dn, **kw) - n_eff_single(wl, n_core=n_core, **kw)) / dn)


def te_slab_index(thickness_um, n_film, n_clad=N_CLAD, wl=1.35):
    """TE0 effective index of a symmetric planar film (E parallel to the layers),
    from the even-mode condition tan(u) = sqrt(V^2 - u^2) / u with
    u = k0 t/2 sqrt(n_film^2 - n_eff^2) and V = k0 t/2 sqrt(n_film^2 - n_clad^2).

    This is the vertical step of the effective-index method: it turns a real
    film thickness into the 2D core index used in the plane. 220 nm Si (3.48)
    in SiO2 at 1.35 um gives 2.947; 400 nm SiN (2.0) gives 1.779.
    """
    from scipy.optimize import brentq

    k0 = 2 * np.pi / wl
    V = k0 * thickness_um / 2 * np.sqrt(n_film**2 - n_clad**2)
    u = brentq(lambda u: np.tan(u) - np.sqrt(V**2 - u**2) / u, 1e-12, min(V, np.pi / 2 - 1e-12))
    return float(np.sqrt(n_film**2 - (u / (k0 * thickness_um / 2)) ** 2))


def core_index_for_neff(n_eff_target, width, n_clad=N_CLAD, wl=1.35, span=8.0):
    """2D core index whose lateral TE0 mode has effective index `n_eff_target`.

    The calibration step of the n_eff-matched effective-index method: the target is the
    fundamental mode index of the real cross-section (a 2D cross-section mode solve,
    e.g. MPB), and the returned core index makes the 2D model propagate, and bend, a mode
    with the same n_eff and the same lateral decay into the cladding. The plain slab
    method (`te_slab_index`) overestimates n_eff by about 0.01-0.03 for strip guides.
    """
    from scipy.optimize import brentq

    return float(brentq(lambda n: n_eff_single(wl, width=width, n_core=n, n_clad=n_clad, span=span) - n_eff_target,
                        n_eff_target, 4.0, xtol=1e-6))
