"""Analytic (meep-free) SAX model of a cascaded Mach-Zehnder lattice filter --
`N` ideal directional couplers joined by `N-1` delay-arm pairs, all sharing one
`delta_L_um`, the classical uniform-delay transversal/FIR-lattice architecture
used for CWDM (de)multiplexers (see e.g. the Luceda "mux2" tutorial). This is
the first module in this repo to actually call `sax.circuit()` -- everything
else in `pic_toolkit.models` only produces SAX-*shaped* SDicts by hand.

Two building blocks:
- `ideal_coupler_model` -- a lossless 2x2 directional coupler, purely analytic
  (no cached FDTD artifact): through amplitude `sqrt(1-kappa)`, cross
  amplitude `sign*1j*sqrt(kappa)`. The `sign` (+-1) is NOT a free/arbitrary
  choice -- see `synthesize_maximally_flat_kappas` below for why some stages
  of a synthesized lattice genuinely need `sign=-1` (a real, physical
  either-way coupler-layout choice), not just `+1`.
- `ideal_arm_model` -- analytic 2-port phase delay, `S21=exp(2j*pi*n_eff(wl)*
  length_um/wl)`, using a first-order dispersion correction so `n_eff(wl)`
  (NOT `n_g`) drives the phase, with `n_g` only setting the correction's
  slope -- `n_g=n_eff-wl*dn_eff/dwl` inverted at `wl0_um`.

`synthesize_maximally_flat_kappas(n_couplers)` derives the coupling ratios an
N-coupler lattice needs for a maximally-flat HALFBAND response, via the
classical two-port paraunitary "layer-peeling" synthesis (Jinguji-style).
`n_couplers` must be even (`N=2K`): the target bar-port power
`|A(phi)|^2 = P_K(sin^2(phi/2))` (phi = 2*pi*n_eff*delta_L/wl) uses the
maximally-flat HALFBAND polynomial `P_K(x) = sum_{k=K}^{2K-1} C(2K-1,k)
x^k (1-x)^(2K-1-k)` -- the regularized-incomplete-beta / "smoothstep" family
(`P_1(x)=x`; `P_2(x)=3x^2-2x^3`, the classic cubic smoothstep) -- chosen
specifically because it is flat (many vanishing derivatives) at BOTH phi=0
AND phi=pi while crossing exactly 0.5 at the SAME phi=pi/2 point regardless
of K (the defining halfband property). This -- not a plain
`((1-w)/2)^(N-1)` binomial target -- is what actually widens the passband as
N grows: an earlier version of this module used that simpler binomial target
and it was WRONG for this purpose -- it is maximally flat only exactly AT
its single peak, and its -1dB bandwidth actually SHRINKS with N (verified
numerically), the opposite of what a WDM channel filter needs. The halfband
target keeps the same -3dB crossover for every N (by construction) while
widening the -1dB (near-flat-top) region as N grows -- confirmed
numerically for N=2 vs N=4 (-1dB width grows from ~30% to ~35% of one FSR,
see `circuits/wdm_mux_mzi_lattice_sax.ipynb` Section 5).

Synthesis, once the target is right, reuses the SAME machinery either way:

1. `A(w)` itself (not just `|A|^2`) is recovered from the target
   `|A(phi)|^2` polynomial by the SAME spectral-factorization technique used
   for step 2 below (`_factor_real`): expand `P_K(sin^2(phi/2))` as an
   ordinary real polynomial in `w` (`w=e^{-i*phi}`, multiplied through by
   `w^(N-1)` to clear the Laurent form), then keep only the roots with
   `|root|<1` (plus half of any root exactly on the unit circle).
2. Its lossless complement `B` (`|A|^2+|B|^2=1`) is found the same way, from
   `Q(w) = w^(N-1) - A(w)*A_rev(w)` (A_rev = A's reverse-coefficient
   polynomial) -- a real, self-inversive polynomial of degree `2(N-1)`; the
   same inside-root selection reproduces `B` as `1j * (real polynomial)` --
   the specific structure this recursion's own `A` real / `B`
   purely-imaginary invariant requires (NOT an arbitrary phase choice --
   multiplying by any other unit-modulus factor breaks it).
3. A backward recursion ("layer peeling") extracts each stage's `kappa_i`
   from the leading coefficients of the current-order `(A,B)` pair, then
   deflates to one order lower. At each step BOTH `sign=+1` and `sign=-1`
   deflations are tried and whichever leaves a numerically-zero leftover
   coefficient is kept -- this per-stage sign is not free: forcing `sign=+1`
   everywhere (the naive assumption) silently produces a WRONG design for
   `N>=3` (confirmed while building this: N=2 happens to only ever need
   `sign=+1`, which is what made the bug easy to miss at first).

The whole synthesis is verified, not just trusted: reconstructing `(A,B)`
forward from the peeled `(kappa_i, sign_i)` sequence must reproduce the
target to near machine precision, checked every call. See
`docs/simulation_settings_record.md`'s `circuits/mzi_lattice.py` section for
the full derivation history (including the binomial-target dead end above).

This module never imports meep.
"""

from __future__ import annotations

import numpy as np
import sax


def ideal_coupler_model(kappa: float = 0.5, sign: int = 1):
    """Ideal lossless 2x2 directional coupler SAX model. Ports: in_top, in_bot
    (left side), out_top, out_bot (right side). Through: in_top->out_top,
    in_bot->out_bot, amplitude sqrt(1-kappa). Cross: in_top->out_bot,
    in_bot->out_top, amplitude sign*1j*sqrt(kappa) -- the standard +-90deg
    phase a lossless 3dB-style coupler must carry to stay unitary."""

    def model(wl=1.35):
        wl = np.asarray(wl, dtype=float)
        ones = np.ones_like(wl)
        t = np.sqrt(1.0 - kappa) * ones
        c = (sign * 1j * np.sqrt(kappa)) * ones
        return sax.reciprocal({
            ("in_top", "out_top"): t, ("in_bot", "out_bot"): t,
            ("in_top", "out_bot"): c, ("in_bot", "out_top"): c,
        })

    return model


def ideal_arm_model(length_um: float, n_eff0: float, n_g0: float, wl0_um: float):
    """Ideal 2-port phase-delay SAX model for one lattice arm. Phase uses
    `n_eff(wl)`, NOT `n_g`, with a first-order dispersion correction so the
    resulting FSR matches the `n_g`-based design target across the plotted
    band: n_eff(wl) = n_eff0 + (n_eff0-n_g0)/wl0_um*(wl-wl0_um), from
    inverting n_g = n_eff - wl*dn_eff/dwl at wl0_um."""
    dn_dwl = (n_eff0 - n_g0) / wl0_um

    def model(wl=1.35):
        wl = np.asarray(wl, dtype=float)
        n_eff = n_eff0 + dn_dwl * (wl - wl0_um)
        phase = 2.0 * np.pi * n_eff * length_um / wl
        s21 = np.exp(1j * phase)
        return sax.reciprocal({("in", "out"): s21})

    return model


def fsr_to_delta_L_um(fsr_um: float, wl0_um: float, n_g: float) -> float:
    """L = wl0^2 / (n_g * FSR) -- the standard FSR<->arm-length-difference
    relation (see e.g. 07_mzi.ipynb Section 10-11, which uses this same
    formula inline; factored out here since this is the first reusable copy
    of it)."""
    return wl0_um**2 / (n_g * fsr_um)


def delta_L_um_to_fsr(delta_L_um: float, wl0_um: float, n_g: float) -> float:
    """Inverse of `fsr_to_delta_L_um` -- FSR that a given delta_L produces."""
    return wl0_um**2 / (n_g * delta_L_um)


def _forward_response(entries: list[tuple[float, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Build the (A, B) polynomial-coefficient pair (low->high degree) that
    `entries` (a list of (kappa, sign) pairs, one per coupler, in physical
    application order) produces. A is always real, B always purely
    imaginary, for real kappas -- an exact invariant of this recursion (see
    module docstring), and the basis of the self-check below."""
    kap0, sign0 = entries[0]
    c0 = np.sqrt(1.0 - kap0)
    s0 = sign0 * np.sqrt(kap0)
    A = np.array([c0], dtype=complex)
    B = np.array([1j * s0], dtype=complex)
    for kap, sign in entries[1:]:
        c = np.sqrt(1.0 - kap)
        s = sign * np.sqrt(kap)
        A_pad = np.concatenate([A, [0.0]])
        B_shift = np.concatenate([[0.0], B])
        A, B = c * A_pad + 1j * s * B_shift, 1j * s * A_pad + c * B_shift
    return A, B


def _target_halfband_A_mag2(n_couplers: int) -> np.ndarray:
    """w^(n_couplers-1) * |A(w)|^2 (low->high, real, palindromic) for the
    maximally-flat halfband target P_K(sin^2(phi/2)), K=n_couplers/2 (see
    module docstring). Built by expanding P_K(x) = sum_{k=K}^{2K-1}
    C(2K-1,k) x^k (1-x)^(2K-1-k) with x=sin^2(phi/2) written as
    `w*x(w) = (-1+2w-w^2)/4` and `w*(1-x(w)) = w - w*x(w)`, so
    `w^(2K-1)*P_K(x(w)) = sum C(2K-1,k) (w*x)^k (w*(1-x))^(2K-1-k)` is an
    ordinary (non-Laurent) polynomial directly."""
    from math import comb
    if n_couplers % 2 != 0:
        raise ValueError(
            f"n_couplers={n_couplers} is odd -- this halfband synthesis only covers even "
            "n_couplers (K=n_couplers/2); an odd-order maximally-flat design needs the more "
            "general (non-halfband) synthesis this module does not implement."
        )
    K = n_couplers // 2
    deg = 2 * K - 1
    w_x = np.array([-1.0, 2.0, -1.0]) / 4.0  # w * x(w)
    w_1mx = np.pad(np.array([0.0, 1.0]), (0, len(w_x) - 2)) - w_x  # w * (1 - x(w))
    total = np.zeros(1)
    for k in range(K, 2 * K):
        term = np.convolve(_poly_pow(w_x, k), _poly_pow(w_1mx, deg - k)) * comb(deg, k)
        if len(term) > len(total):
            total = np.pad(total, (0, len(term) - len(total)))
        else:
            term = np.pad(term, (0, len(total) - len(term)))
        total = total + term
    return total


def _poly_pow(coeffs: np.ndarray, power: int) -> np.ndarray:
    result = np.array([1.0])
    for _ in range(power):
        result = np.convolve(result, coeffs)
    return result


def _factor_real(poly: np.ndarray, degree: int, make_imaginary: bool) -> np.ndarray:
    """Spectral factorization shared by both A's and B's derivation: given a
    real, self-inversive (palindromic) polynomial `poly` (low->high, degree
    `2*degree`, non-negative on |w|=1), return the one-sided real factor of
    degree `degree` (times 1j if `make_imaginary`) whose |.|^2 on |w|=1
    reproduces `poly/w^degree`. Keeps roots strictly inside the unit circle
    (plus half of any cluster sitting exactly on it, within a tolerance loose
    enough to absorb the numerical spread `numpy.roots` gives near a
    high-multiplicity root -- the flat-halfband targets this module uses
    have exactly that at w=-1) -- this automatically keeps the result both
    real (a conjugate-root pair shares the same magnitude, so it's kept or
    dropped as a pair) and correctly normalized (exactly one root per
    reciprocal pair)."""
    if np.allclose(poly, 0):
        return np.zeros(degree + 1, dtype=complex)
    roots = np.roots(poly[::-1])
    tol = 0.02
    inside, oncircle = [], []
    for r in roots:
        ar = abs(r)
        if ar < 1 - tol:
            inside.append(r)
        elif ar <= 1 + tol:
            oncircle.append(r)
    oncircle.sort(key=lambda z: (round(z.real, 3), round(z.imag, 3)))
    kept_oncircle = []
    i = 0
    while i < len(oncircle):
        j = i
        while j < len(oncircle) and abs(oncircle[j] - oncircle[i]) < 0.05:
            j += 1
        cluster = oncircle[i:j]
        kept_oncircle.extend(cluster[: len(cluster) // 2])
        i = j
    chosen = inside + kept_oncircle
    if len(chosen) != degree:
        raise RuntimeError(
            f"spectral factorization root count mismatch: got {len(chosen)}, need {degree} "
            f"(inside={len(inside)}, on-circle kept={len(kept_oncircle)}/{len(oncircle)})"
        )
    factor = np.real(np.poly(chosen)[::-1])
    factor_rev = factor[::-1]
    prod = np.convolve(factor, factor_rev)
    wtest = 2.0  # any point off the unit circle and off factor's own roots
    g2 = np.polyval(poly[::-1], wtest) / np.polyval(prod[::-1], wtest)
    if g2 < 0:
        raise RuntimeError(f"spectral factorization scale g^2={g2:.3e} < 0 -- wrong root branch")
    result = np.sqrt(g2) * factor.astype(complex)
    return 1j * result if make_imaginary else result


def _peel(A: np.ndarray, B: np.ndarray) -> list[tuple[float, int]]:
    """Backward layer-peeling recursion: extract (kappa_i, sign_i) one stage
    at a time from the highest order down, deflating by one degree each
    step. `sign_i` is picked, per step, as whichever of +1/-1 leaves the
    dropped (highest-order) coefficient nearest zero -- see module docstring
    for why this per-stage choice is necessary, not optional."""
    n_couplers = len(A)
    entries: list[tuple[float, int]] = []
    A_cur, B_cur = A.copy(), B.copy()
    for _ in range(n_couplers, 1, -1):
        a0, b0 = A_cur[0], B_cur[0]
        denom = abs(a0) ** 2 + abs(b0) ** 2
        kap = float(np.clip(abs(b0) ** 2 / denom, 0.0, 1.0))
        c = np.sqrt(1.0 - kap)
        best = None
        for sign in (1, -1):
            s = sign * np.sqrt(kap)
            A_full = c * A_cur - 1j * s * B_cur
            B_full = -1j * s * A_cur + c * B_cur
            dropped = abs(A_full[-1]) + abs(B_full[0])
            if best is None or dropped < best[0]:
                best = (dropped, sign, A_full, B_full)
        _, sign, A_full, B_full = best
        entries.append((kap, sign))
        A_cur, B_cur = A_full[:-1], B_full[1:]
    a0, b0 = A_cur[0], B_cur[0]
    denom = abs(a0) ** 2 + abs(b0) ** 2
    kap0 = float(np.clip(abs(b0) ** 2 / denom, 0.0, 1.0))
    # b0 should equal sign0 * 1j * sqrt(kap0); sign0 irrelevant (b0~0) when kap0~0.
    sign0 = -1 if (kap0 > 1e-12 and (b0 / (1j * np.sqrt(kap0))).real < 0) else 1
    entries.append((kap0, sign0))
    return list(reversed(entries))


def synthesize_maximally_flat_kappas(n_couplers: int) -> tuple[list[float], list[int]]:
    """Synthesize the (kappa_i, sign_i) sequence for an n_couplers-coupler
    maximally-flat lattice (see module docstring for the full method).
    Returns (kappas, signs) -- kappas are the physical power-coupling ratios
    (what to report/plot), signs are needed alongside them to build a
    correct `ideal_coupler_model` for each stage.

    Self-verified every call: forward-reconstructing (A, B) from the peeled
    sequence must reproduce the target to near machine precision, or this
    raises -- a synthesis bug fails loudly rather than silently returning a
    wrong design (same discipline as this repo's other tolerance/validation
    checks, see CLAUDE.md Sec 6)."""
    if n_couplers < 2:
        raise ValueError(f"n_couplers must be >= 2, got {n_couplers}")
    m = n_couplers - 1
    A_mag2 = _target_halfband_A_mag2(n_couplers)
    A_target = _factor_real(A_mag2, m, make_imaginary=False)
    if A_target[0].real < 0:  # normalize sign: this recursion needs A[0]=c0>=0
        A_target = -A_target
    Q = np.zeros(2 * m + 1)
    Q[m] = 1.0
    Q -= np.convolve(A_target.real, A_target.real[::-1])
    B_target = _factor_real(Q, m, make_imaginary=True)
    entries = _peel(A_target, B_target)
    A_check, B_check = _forward_response(entries)
    err = max(np.max(np.abs(A_target - A_check)), np.max(np.abs(B_target - B_check)))
    # Tolerance loosened beyond machine precision because `numpy.roots` is only accurate to
    # roughly eps**(1/multiplicity) near the high-multiplicity root these halfband targets
    # have at w=-1 (~1e-4 for a quadruple root at N=4, empirically) -- still tight enough to
    # catch a real synthesis bug (those showed residuals of order 1, not 1e-4).
    if err > 1e-3:
        raise RuntimeError(
            f"synthesize_maximally_flat_kappas(n_couplers={n_couplers}) failed its own "
            f"forward-reconstruction self-check (residual={err:.3e}, expected <1e-3) -- "
            "the synthesized kappas/signs do not reproduce the intended maximally-flat "
            "response; do not trust this design."
        )
    kappas = [float(kap) for kap, _ in entries]
    signs = [int(sign) for _, sign in entries]
    return kappas, signs


def build_lattice_circuit(
    kappas: list[float], signs: list[int], delta_L_um: float,
    n_eff0: float, n_g0: float, wl0_um: float,
):
    """Build the sax.circuit() for an N-coupler lattice: N couplers, N-1
    arm-pairs (reference arm length_um=0, delay arm length_um=delta_L_um --
    only the DIFFERENCE matters for this ideal/lossless model, so the
    reference arm's absolute length is arbitrary). Exposes the overall
    in_top/in_bot (leftmost coupler) and out_top/out_bot (rightmost coupler)
    ports. Returns (circuit_fn, info) exactly as `sax.circuit` does."""
    if len(kappas) != len(signs):
        raise ValueError("kappas and signs must have the same length")
    n = len(kappas)
    models = {}
    instances = {}
    connections = {}
    for i, (kap, sign) in enumerate(zip(kappas, signs)):
        name = f"coupler{i}"
        models[name] = ideal_coupler_model(kap, sign)
        instances[name] = {"component": name, "settings": {}}
    for i in range(n - 1):
        ref_name, delay_name = f"arm_ref{i}", f"arm_delay{i}"
        models[ref_name] = ideal_arm_model(0.0, n_eff0, n_g0, wl0_um)
        models[delay_name] = ideal_arm_model(delta_L_um, n_eff0, n_g0, wl0_um)
        instances[ref_name] = {"component": ref_name, "settings": {}}
        instances[delay_name] = {"component": delay_name, "settings": {}}
        connections[f"coupler{i},out_top"] = f"{ref_name},in"
        connections[f"coupler{i},out_bot"] = f"{delay_name},in"
        connections[f"{ref_name},out"] = f"coupler{i + 1},in_top"
        connections[f"{delay_name},out"] = f"coupler{i + 1},in_bot"

    netlist = {
        "instances": instances,
        "connections": connections,
        "ports": {
            "in_top": "coupler0,in_top", "in_bot": "coupler0,in_bot",
            "out_top": f"coupler{n - 1},out_top", "out_bot": f"coupler{n - 1},out_bot",
        },
    }
    return sax.circuit(netlist, models=models)


def unitary_project(model_fn, port_names: tuple):
    """Wrap a SAX model function so its returned SDict, treated as a square
    matrix over `port_names` (missing pairs default to 0), is replaced by the
    NEAREST exactly-unitary matrix (SVD projection: `M = U*Sigma*Vh` ->
    `M_unitary = U*Vh`), applied per-wavelength-point.

    Why this exists: a real, FDTD-measured component (e.g. `models.coupler.
    coupler`, `models.mzi_arm.mzi_arm`) is only unitary up to measurement
    noise -- each individually passes its own validation (a few percent
    energy-conservation deviation, well within that component's own
    tolerance). Composing 2 such components via `sax.circuit()` stays honest
    (measured directly composing `circuits/mzi_real_fdtd_sax.ipynb`'s N=2
    real MZI: total scattered power ~0.98, consistent with real loss). But
    composing 3+ in series (an N=4 real lattice) was found to VIOLATE
    passivity outright -- total scattered power exceeding 1 by up to ~26% at
    some wavelengths, worsening with each additional stage (~0.98 at N=2,
    ~1.01 at N=3, ~1.04 at N=4, confirmed even with 4 IDENTICAL, individually-
    well-validated 50:50 couplers, ruling out any one component's own
    kappa/geometry as the cause). Root cause: a component's per-excitation
    row satisfying `sum_j|S_ij|^2~=1` (what each component's own validation
    checks) is necessary but NOT sufficient for its full S-matrix to be
    unitary -- row *orthogonality* (`row_i . conj(row_j) = 0` for `i!=j`) can
    still be measurably violated by FDTD noise, and `sax.circuit()`'s exact
    (not naive series-multiplication) solver amplifies that violation with
    each additional cascaded stage. Confirmed by isolation: projecting
    `mzi_arm` alone onto the nearest unitary matrix (it has by far the
    largest individual imperfection, ~6% reflection) already fixed the N=3/
    N=4 passivity violation; projecting both `mzi_arm` and `coupler` gives
    the cleanest result (N=4 total power stayed in [0.82, 1.00] across a full
    wavelength sweep, vs. up to 1.26 unprojected). See
    `docs/simulation_settings_record.md`'s `circuits/mzi_lattice.py`
    (`unitary_project`) section for the full investigation."""
    import numpy as np

    def wrapped(wl=1.35):
        s = model_fn(wl=wl)
        wl_arr = np.atleast_1d(np.asarray(wl, dtype=float))
        n_wl = len(wl_arr)
        k = len(port_names)
        M = np.zeros((n_wl, k, k), dtype=complex)
        for i, p1 in enumerate(port_names):
            for j, p2 in enumerate(port_names):
                v = np.atleast_1d(np.asarray(s.get((p1, p2), 0j), dtype=complex))
                M[:, i, j] = v if len(v) == n_wl else np.full(n_wl, v[0])
        U, _sigma, Vh = np.linalg.svd(M)
        M_unitary = U @ Vh
        scalar_in = np.ndim(wl) == 0
        out = {}
        for i, p1 in enumerate(port_names):
            for j, p2 in enumerate(port_names):
                val = M_unitary[:, i, j]
                out[(p1, p2)] = complex(val[0]) if scalar_in else val
        return out

    return wrapped
