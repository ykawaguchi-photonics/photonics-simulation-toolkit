"""Coupled-mode (coupling-matrix) fits of ring and racetrack resonator spectra.

Meep-free; used by si/notebooks/05_racetrack_resonator.ipynb,
si/notebooks/10_add_drop_ring_resonator.ipynb and scripts/blog_figures/.

Conventions (Yariv, Electron. Lett. 36, 321 (2000); Bogaerts et al., Laser
Photonics Rev. 6, 47 (2012)):
- t, kappa: field self-/cross-coupling of a bus-ring coupler, t^2 + kappa^2 = 1.
- a: single-pass field transmission of the ring, a^2 = exp(-alpha L)
  (alpha = power attenuation per um, L = round-trip length).
- Round-trip phase near a resonance at nu0 = 1/lambda_res:
  phi = 2 pi (nu - nu0) / FSR_nu with FSR_nu = 1 / (n_g L). The slope uses the
  GROUP index measured from the free spectral range. A single resonance does not
  determine n_eff (it is degenerate with the azimuthal order m), so n_eff is
  not fitted.

All-pass ring:  S21 = (t - a e^{i phi}) / (1 - t a e^{i phi}).
Add-drop ring (t1 = input-bus coupler, t2 = drop-bus coupler, D = 1 - t1 t2 a e^{i phi}):
    input -> through = (t1 - t2 a e^{i phi}) / D
    add   -> drop    = (t2 - t1 a e^{i phi}) / D
    input -> drop = add -> through = -kappa1 kappa2 sqrt(a) e^{i phi/2} / D

The through-port power of an all-pass ring is symmetric in t <-> a (McKinnon et
al., Opt. Express 17, 18971 (2009)), so both branches are fitted and the physical
one is chosen from (i) the S21 phase, which swings by much less than pi across an
under-coupled resonance and by 2 pi across an over-coupled one (Heebner et al.,
IEEE JQE 40, 726 (2004)), and (ii) the loss floor set by the ring material's own
absorption.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from .slab_modes import dneff_dn_core


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def round_trip_phase(nu, nu0, fsr_nu):
    return 2 * np.pi * (np.asarray(nu) - nu0) / fsr_nu


def all_pass(nu, t, a, nu0, fsr_nu, scale=1.0):
    """Complex all-pass transmission; `scale` absorbs a small off-resonance insertion loss."""
    e = np.exp(1j * round_trip_phase(nu, nu0, fsr_nu))
    return np.sqrt(scale) * (t - a * e) / (1 - t * a * e)


def add_drop(nu, t1, t2, a, nu0, fsr_nu):
    """Complex add-drop transfer functions: dict with 'through', 'drop', 'add_drop', 'add_through'."""
    phi = round_trip_phase(nu, nu0, fsr_nu)
    e = np.exp(1j * phi)
    D = 1 - t1 * t2 * a * e
    k1, k2 = np.sqrt(1 - t1**2), np.sqrt(1 - t2**2)
    drop = -k1 * k2 * np.sqrt(a) * np.exp(1j * phi / 2) / D
    return {"through": (t1 - t2 * a * e) / D, "drop": drop, "add_drop": (t2 - t1 * a * e) / D, "add_through": drop}


# ---------------------------------------------------------------------------
# Spectral features
# ---------------------------------------------------------------------------
def _sorted(wl, *arrays):
    o = np.argsort(wl)
    return (np.asarray(wl)[o], *[np.asarray(x)[o] for x in arrays])


def group_index_from_fsr(wl, T, L, prominence=0.1):
    """Group index from the spacing of adjacent transmission dips, n_g = l1 l2 / (dl L).
    Returns (median n_g, dict with resonance wavelengths, FSRs and per-pair n_g)."""
    wl, T = _sorted(wl, T)
    pk, _ = find_peaks(-T, prominence=prominence)
    lam = wl[pk]
    if len(lam) < 2:
        raise ValueError("need at least two resonances in the band to measure the FSR")
    pairs = lam[:-1] * lam[1:] / np.diff(lam) / L
    return float(np.median(pairs)), {"resonances_um": lam.tolist(), "fsr_nm": (np.diff(lam) * 1e3).tolist(),
                                     "n_g_pairs": pairs.tolist(), "grid_spacing_nm": float(np.median(np.diff(wl)) * 1e3)}


def absorption_floor(sigma, wl, n_core=2.7, **slab_kw):
    """Modal power attenuation (1/um) caused by a Meep D_conductivity `sigma` on the core alone.
    Meep: eps(omega) = eps (1 + i sigma/omega) -> n'' = n sigma / (2 omega) with omega = 2 pi / wl
    (c = 1, um units); the mode sees dn_eff/dn_core of it; alpha = 2 k0 n''_eff."""
    k0 = 2 * np.pi / wl
    n_imag_core = n_core * sigma / (2 * k0)
    return float(2 * k0 * dneff_dn_core(wl, n_core=n_core, **slab_kw) * n_imag_core)


def detrended_phase(wl, s):
    """Unwrapped phase of s minus the straight line through its end points, returned in the
    same order as the input `wl` (unwrapping is done in ascending-wavelength order)."""
    wl = np.asarray(wl)
    o = np.argsort(wl)
    ws, ss = wl[o], np.asarray(s)[o]
    ph = np.unwrap(np.angle(ss))
    det = ph - np.polyval(np.polyfit(ws[[0, -1]], ph[[0, -1]], 1), ws)
    out = np.empty_like(det)
    out[o] = det
    return out


def half_depth_fwhm(wl, T):
    """Full width of the deepest dip at half depth, (1 + T_min)/2, from the sampled points.
    Only the contiguous run of points below half depth around the minimum counts, so a
    second resonance in the window (e.g. one clipped at its edge) does not widen it."""
    wl, T = _sorted(wl, T)
    i = int(np.argmin(T))
    below = T < (1 + T[i]) / 2
    lo = i
    while lo > 0 and below[lo - 1]:
        lo -= 1
    hi = i
    while hi < len(T) - 1 and below[hi + 1]:
        hi += 1
    return float(wl[hi] - wl[lo]), float(wl[i]), float(T[i])


# ---------------------------------------------------------------------------
# All-pass fit
# ---------------------------------------------------------------------------
def fit_all_pass(wl, s21, L, n_g):
    """Fit one all-pass resonance with the FSR fixed by n_g. Returns a dict with both
    branches ('under': t > a seed, 'over': t < a seed), the measured T_min / FWHM / Q_L,
    and the detrended phase swing of the data."""
    wl, s21 = _sorted(wl, s21)
    T = np.abs(s21) ** 2
    nu, fsr_nu = 1 / wl, 1 / (n_g * L)
    i = int(np.argmin(T))
    branches = {}
    for name, (t0, a0) in (("under", (0.98, 0.93)), ("over", (0.93, 0.98))):
        r = least_squares(lambda q: np.abs(all_pass(nu, q[0], q[1], q[2], fsr_nu, q[3])) ** 2 - T,
                          [t0, a0, nu[i], 1.0],
                          bounds=([0.5, 0.5, nu[i] - 0.002, 0.9], [0.99999, 0.99999, nu[i] + 0.002, 1.05]))
        t, a, nu0, sc = r.x
        model = all_pass(nu, t, a, nu0, fsr_nu, sc)
        branches[name] = {"t": float(t), "a": float(a), "kappa": float(np.sqrt(1 - t**2)), "kappa2": float(1 - t**2),
                          "alpha_per_um": float(-2 * np.log(a) / L), "lambda_res_um": float(1 / nu0),
                          "scale": float(sc), "rms": float(np.sqrt(np.mean(r.fun**2))),
                          "phase_swing_rad": float(np.ptp(detrended_phase(wl, model)))}
    fwhm, lam_min, T_min = half_depth_fwhm(wl, T)
    return {"L_um": float(L), "n_g": float(n_g), "T_min": T_min, "lambda_min_um": lam_min,
            "fwhm_nm": fwhm * 1e3, "Q_L": lam_min / fwhm,
            "phase_swing_rad": float(np.ptp(detrended_phase(wl, s21))), "branches": branches}


CRITICAL_T_A_TOL = 0.005  # |t - a| below this (T_min below ~1e-3 here) counts as critically coupled


def choose_branch(fit, alpha_floor=None):
    """Pick the physical branch of `fit_all_pass` and add derived quantities.
    Criteria: the measured phase swing (< pi -> under-coupled, otherwise over-coupled),
    cross-checked against the loss floor (a branch with alpha below the material
    absorption is unphysical). Returns (name, branch dict with Q_i / Q_c / FWHM formula, reason)."""
    swing = fit["phase_swing_rad"]
    by_phase = "under" if swing < np.pi else "over"
    reasons = [f"measured phase swing {swing:.2f} rad {'<' if by_phase == 'under' else '>='} pi -> {by_phase}-coupled"]
    if alpha_floor is not None:
        other = "over" if by_phase == "under" else "under"
        a_other = fit["branches"][other]["alpha_per_um"]
        a_chosen = fit["branches"][by_phase]["alpha_per_um"]
        if a_other < alpha_floor:
            reasons.append(f"{other}-coupled branch alpha {a_other:.2e} < absorption floor {alpha_floor:.2e} /um (unphysical)")
        if a_chosen < alpha_floor:
            reasons.append(f"WARNING: chosen branch alpha {a_chosen:.2e} is below the absorption floor")
    b = dict(fit["branches"][by_phase])
    at = b["t"] * b["a"]
    lam = b["lambda_res_um"]
    b["fwhm_formula_nm"] = (1 - at) * lam**2 / (np.pi * fit["n_g"] * fit["L_um"] * np.sqrt(at)) * 1e3
    b["Q_i"] = 2 * np.pi * fit["n_g"] / (lam * b["alpha_per_um"])
    b["Q_c"] = 1 / (1 / fit["Q_L"] - 1 / b["Q_i"]) if fit["Q_L"] < b["Q_i"] else float("inf")
    b["regime"] = f"{by_phase}-coupled"
    if abs(b["t"] - b["a"]) < CRITICAL_T_A_TOL:
        # Both branches coincide at t = a, and the phase jumps by ~pi where |S21| passes
        # through zero, so the phase swing no longer decides anything.
        b["regime"] = "critically coupled"
        reasons.append(f"|t - a| = {abs(b['t'] - b['a']):.4f} < {CRITICAL_T_A_TOL} -> critically coupled "
                       "(both branches coincide)")
    return by_phase, b, "; ".join(reasons)


def critical_kappa2(a):
    """Power cross-coupling at which an all-pass ring with single-pass transmission a is critically coupled (t = a)."""
    return float(1 - a**2)


# ---------------------------------------------------------------------------
# Add-drop fit
# ---------------------------------------------------------------------------
def fit_add_drop(wl, through, drop, add_drop_s, add_through, L, n_g, seeds=(0.9, 0.95, 0.98)):
    """Joint fit of the four measured add-drop transfer functions (complex or power arrays)
    with the FSR fixed by n_g. Multi-started over (t1, t2, a). Returns a dict."""
    wl, *arrs = _sorted(wl, through, drop, add_drop_s, add_through)
    P = [np.abs(x) ** 2 for x in arrs]
    nu, fsr_nu = 1 / wl, 1 / (n_g * L)
    i = int(np.argmin(P[0]))

    def resid(q):
        m = add_drop(nu, q[0], q[1], q[2], q[3], fsr_nu)
        return np.concatenate([np.abs(m[k]) ** 2 - p for k, p in zip(("through", "drop", "add_drop", "add_through"), P)])

    best = min((least_squares(resid, [x1, x2, x3, nu[i]],
                              bounds=([0.5, 0.5, 0.5, nu[i] - 0.001], [0.99999, 0.99999, 0.99999, nu[i] + 0.001]))
                for x1 in seeds for x2 in seeds for x3 in seeds), key=lambda r: r.cost)
    t1, t2, a, nu0 = best.x
    fwhm, lam_min, T_min = half_depth_fwhm(wl, P[0])
    return {"t1": float(t1), "t2": float(t2), "a": float(a), "kappa1": float(np.sqrt(1 - t1**2)),
            "kappa2": float(np.sqrt(1 - t2**2)), "kappa1_2": float(1 - t1**2), "kappa2_2": float(1 - t2**2),
            "alpha_per_um": float(-2 * np.log(a) / L), "lambda_res_um": float(1 / nu0), "n_g": float(n_g),
            "L_um": float(L), "rms": float(np.sqrt(np.mean(best.fun**2))),
            "T_through_min": T_min, "T_drop_max": float(P[1].max()), "T_add_drop_min": float(P[2].min()),
            "fwhm_nm": fwhm * 1e3, "Q_L": lam_min / fwhm,
            "regime": "under-coupled" if t1 > a * t2 else "over-coupled",
            "critical_condition": f"t1 = a*t2: t1 = {t1:.4f}, a*t2 = {a * t2:.4f}"}
