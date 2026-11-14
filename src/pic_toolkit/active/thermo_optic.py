"""Thermo-optic figures of merit: couples the 2D heat solve (`thermal`) to the MPB mode
(`si_xsection`). Everything is per unit power per unit length (P' = 1 W/m) and linear.

    Delta n_eff(P')   from si_xsection.thermo_optic_dneff on the solved temperature field
    P_pi = wl / (2 * dn_eff/dP')            [W]  -- the heater length cancels (see thermal.py)
    tau              1/e rise of the mode-weighted Delta n_eff after a power step
    f_3dB            from the small-signal frequency response of the same readout
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import si_xsection as xs
from . import thermal as th


def dneff_reader(prob: th.Problem, mode: xs.ModeResult, dndT_si: float, x_shift_um: float = 0.0,
                 dndT_ox: float = xs.SIO2_DNDT):
    """Linear map: node temperature array (ny, nx) -> Delta n_eff of `mode` placed at lateral
    position x_shift_um (a neighbor waveguide for crosstalk; the mode itself is solved at x = 0)."""
    X, Y = np.meshgrid(mode.x_um, mode.y_um)
    pts = np.column_stack([Y.ravel(), X.ravel() + x_shift_um])

    def read(T):
        it = th.interpolator(prob, T)
        return xs.thermo_optic_dneff(mode, lambda _p: it(pts), dndT_si, dndT_ox)["dneff"]
    return read


def point_reader(prob: th.Problem, y_um: float, x_um: float = 0.0):
    """Temperature at one point (the metric of Jacques 2019 / the Tidy3D reproduction)."""
    return lambda T: float(th.interpolator(prob, T)([[y_um, x_um]])[0])


@dataclass
class HeaterResult:
    p_pi_mW: float
    dneff_per_W_per_m: float
    tau_us: float | None
    f3db_kHz: float | None
    T: np.ndarray                  # steady node temperature rise for P' = 1 W/m
    times_s: np.ndarray | None = None
    step: np.ndarray | None = None
    freqs_hz: np.ndarray | None = None
    H: np.ndarray | None = None

    @property
    def p_pi_tau(self) -> float:
        return self.p_pi_mW * self.tau_us if self.tau_us is not None else float("nan")


def evaluate(prob: th.Problem, mode: xs.ModeResult, wl_um: float, dndT_si: float, transient: bool = True,
             frequency: bool = False, steps_per_decade: int = 100, freqs_hz=None) -> HeaterResult:
    sysm = th.assemble(prob)
    T = th.solve_steady(prob, 1.0, sysm)
    read = dneff_reader(prob, mode, dndT_si)
    d = read(T)
    out = HeaterResult(p_pi_mW=wl_um * 1e-6 / (2 * d) * 1e3, dneff_per_W_per_m=d, tau_us=None, f3db_kHz=None, T=T)
    if transient:
        t, r = th.step_response(prob, 1e-8, 1e-3, read, steps_per_decade=steps_per_decade, sysm=sysm)
        out.times_s, out.step, out.tau_us = t, r, th.time_constant(t, r) * 1e6
    if frequency:
        f = np.geomspace(1e3, 1e7, 61) if freqs_hz is None else np.asarray(freqs_hz)
        H = th.frequency_response(prob, f, read, sysm=sysm)
        out.freqs_hz, out.H, out.f3db_kHz = f, H, th.bandwidth_3db(f, H) / 1e3
    return out


def crosstalk(prob: th.Problem, mode: xs.ModeResult, dndT_si: float, pitches_um, T=None) -> np.ndarray:
    """Phase on a neighbor waveguide at each pitch, relative to the heated waveguide's own
    phase (both read with the same mode overlap). The neighbor cores must be part of `prob`."""
    T = th.solve_steady(prob, 1.0) if T is None else T
    d0 = dneff_reader(prob, mode, dndT_si)(T)
    return np.array([dneff_reader(prob, mode, dndT_si, p)(T) / d0 for p in pitches_um])


def ring_tuning(p_pi_mW: float, radius_um: float, wl_um: float, n_g: float) -> dict:
    """Heater over the whole ring: the round trip shifts by 2 pi at P = 2 P_pi, which moves
    the resonance by one FSR, whatever the radius (the ring's length cancels as for P_pi).
    Uses the straight-waveguide cross-section for the bent ring (an approximation)."""
    L = 2 * np.pi * radius_um
    fsr_nm = wl_um ** 2 / (n_g * L) * 1e3
    return {"radius_um": radius_um, "fsr_nm": fsr_nm, "p_fsr_mW": 2 * p_pi_mW,
            "nm_per_mW": fsr_nm / (2 * p_pi_mW)}
