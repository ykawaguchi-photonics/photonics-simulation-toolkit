"""Physical sanity checks for a two-port S-matrix. Pure numpy -- no meep import,
so these can run identically on freshly-simulated or later-loaded/cached data.
"""

from __future__ import annotations

import numpy as np


def energy_conservation_check(s_matrix: dict, tol: float = 1e-2, max_loss_fraction: float = 0.0) -> dict:
    """For a passive 2-port device, power in must equal power out plus
    whatever is lost to a real channel outside the 2-port model:
    |S11|^2 + |S21|^2 = 1 - loss (likewise for port 2).

    max_loss_fraction=0 (the default) means "expect no loss channel" -- a
    lossless device (the Phase-1 straight waveguide) should show no deficit
    from 1 beyond `tol`, and this reduces exactly to a strict |total-1| check.
    A component with a REAL, expected loss mechanism (e.g. a small-radius
    ring resonator's bend-radiation loss) can declare how much deficit is
    physically expected via max_loss_fraction, rather than that deficit being
    mistaken for simulation error -- deviation is then measured against the
    allowed band [1-max_loss_fraction, 1], not against a strict target of 1.
    """
    total_1 = np.abs(s_matrix["11"]) ** 2 + np.abs(s_matrix["21"]) ** 2
    total_2 = np.abs(s_matrix["22"]) ** 2 + np.abs(s_matrix["12"]) ** 2
    lower_bound = 1.0 - max_loss_fraction

    def band_deviation(total):
        return float(np.max(np.maximum(total - 1.0, lower_bound - total)))

    max_dev_1 = max(0.0, band_deviation(total_1))
    max_dev_2 = max(0.0, band_deviation(total_2))
    max_dev = max(max_dev_1, max_dev_2)
    return {
        "max_deviation_port1": max_dev_1,
        "max_deviation_port2": max_dev_2,
        "max_deviation": max_dev,
        "max_loss_fraction_allowed": max_loss_fraction,
        "tolerance": tol,
        "passed": max_dev < tol,
    }


def passivity_check(s_matrix: dict, tol: float = 1e-2) -> dict:
    """A passive device can never amplify: every |S_ij| <= 1, within a small
    numerical margin. Unlike energy_conservation_check, this holds regardless
    of whether the device is lossless or has a real loss channel, so it's the
    more fundamental "did the simulation do something unphysical" check.
    """
    max_mag = max(float(np.max(np.abs(s_matrix[k]))) for k in ["11", "12", "21", "22"])
    return {
        "max_magnitude": max_mag,
        "tolerance": tol,
        "passed": max_mag < 1.0 + tol,
    }


def reciprocity_check(s_matrix: dict, tol: float = 1e-2) -> dict:
    """A passive, reciprocal device must satisfy S12 = S21 (and S11=S22 for a
    symmetric straight waveguide). S12 and S21 here come from two INDEPENDENT
    excitation runs, so this genuinely tests the simulation rather than
    restating an assumption.
    """
    diff_12_21 = np.max(np.abs(s_matrix["12"] - s_matrix["21"]))
    diff_11_22 = np.max(np.abs(s_matrix["11"] - s_matrix["22"]))
    max_diff = float(max(diff_12_21, diff_11_22))
    return {
        "max_diff_s12_s21": float(diff_12_21),
        "max_diff_s11_s22": float(diff_11_22),
        "max_diff": max_diff,
        "tolerance": tol,
        "passed": max_diff < tol,
    }


def run_all_checks(s_matrix: dict, energy_tol: float = 1e-2, reciprocity_tol: float = 1e-2,
                    passivity_tol: float = 1e-2, max_loss_fraction: float = 0.0) -> dict:
    energy = energy_conservation_check(s_matrix, tol=energy_tol, max_loss_fraction=max_loss_fraction)
    reciprocity = reciprocity_check(s_matrix, tol=reciprocity_tol)
    passivity = passivity_check(s_matrix, tol=passivity_tol)
    return {
        "energy_conservation": energy,
        "reciprocity": reciprocity,
        "passivity": passivity,
        "passed": bool(energy["passed"] and reciprocity["passed"] and passivity["passed"]),
    }
