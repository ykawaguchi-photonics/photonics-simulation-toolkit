"""Mode-solver data for the blog article "Silicon Waveguides".

Solves the real 3D cross-section of a silicon strip waveguide (Si core fully
clad in SiO2) with MPB and saves the results to
si/data/blog_figures/silicon_waveguides_mpb.npz, which silicon_waveguides.py reads.
No FDTD time stepping happens here (MPB is a frequency-domain eigensolver).

What is solved:
- width sweep (220 nm thick core) at 1.35 and 1.55 um: every guided mode of
  both mirror families, with n_eff, n_g and the share of transverse E that is
  lateral (TE-like > 0.5);
- wavelength sweep of the 500 x 220 nm TE-like mode, with non-dispersive
  indices (3.48 / 1.44, the toolkit's 3D-track values) and with material
  dispersion (Li 1980 for Si, Malitson 1965 for SiO2);
- the TE-like and TM-like mode fields of 500 x 220 nm at 1.35 um.

MPB lattice: MPB x = vertical, MPB y = lateral, k along z. As in
meep_sim/tfln_waveguide.py, TE-like modes (lateral E even across the core)
are MPB's ODD_Y family; TM0 and TE1 are in EVEN_Y.

Usage (from the repo root, ~5 min on one core):
    conda run -n mp python scripts/blog_figures/silicon_waveguides_data.py
"""

import contextlib
import io
from pathlib import Path

import meep as mp
import numpy as np
from meep import mpb

mp.verbosity(0)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "si" / "data" / "blog_figures" / "silicon_waveguides_mpb.npz"

N_SI, N_OX = 3.48, 1.44          # 3D-track values (3d/notebooks/01b_waveguide_3d_fdtdx.ipynb)
H_UM = 0.22
CELL_V, CELL_L = 3.0, 4.0         # MPB cell (vertical, lateral), um
RES_SWEEP, RES_FINE = 32, 64      # res 64 reproduces the converged 2.6460 of notebook 01b


def n_si_li(wl_um):
    """Room-temperature Si index, H. H. Li, J. Phys. Chem. Ref. Data 9, 561 (1980)."""
    l2 = wl_um**2
    return np.sqrt(11.6858 + 0.939816 / l2 + 0.00810461 * 1.1071**2 / (l2 - 1.1071**2))


def n_sio2_malitson(wl_um):
    """Fused silica, I. H. Malitson, J. Opt. Soc. Am. 55, 1205 (1965)."""
    l2 = wl_um**2
    return np.sqrt(1 + 0.6961663 * l2 / (l2 - 0.0684043**2) + 0.4079426 * l2 / (l2 - 0.1162414**2)
                   + 0.8974794 * l2 / (l2 - 9.896161**2))


def solver(w_um, n_si=N_SI, n_ox=N_OX, res=RES_SWEEP, num_bands=1):
    geom = [mp.Block(size=mp.Vector3(H_UM, w_um, mp.inf), material=mp.Medium(index=n_si))]
    return mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(CELL_V, CELL_L, 0)),
                          geometry=geom, default_material=mp.Medium(index=n_ox),
                          resolution=res, num_bands=num_bands)


def solve(w_um, wl_um, parity, band, n_si=N_SI, n_ox=N_OX, res=RES_SWEEP, fields=False):
    """One band at fixed frequency. Returns None if the band is not guided."""
    f = 1.0 / wl_um
    ms = solver(w_um, n_si, n_ox, res, band)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            k = ms.find_k(parity, f, band, band, mp.Vector3(0, 0, 1), 1e-8,
                          0.8 * n_si * f, 1.0005 * n_ox * f, n_si * f)[0]
            vg = ms.compute_group_velocity_component(mp.Vector3(0, 0, 1))[band - 1]
            E = np.asarray(ms.get_efield(band, bloch_phase=False)[:, :, 0, :])
    except Exception:
        return None
    n_eff = k / f
    if n_eff < n_ox * 1.001:
        return None
    pv, pl = np.sum(np.abs(E[..., 0]) ** 2), np.sum(np.abs(E[..., 1]) ** 2)
    out = {"n_eff": float(n_eff), "n_g": float(1.0 / vg), "te_share": float(pl / (pl + pv))}
    if fields:
        out["E"] = E
    return out


def width_sweep(wl_um, widths):
    rows = []
    for w in widths:
        for parity, fam in ((mp.ODD_Y, 0), (mp.EVEN_Y, 1)):
            for band in (1, 2, 3):
                r = solve(w, wl_um, parity, band)
                if r is None:
                    break
                rows.append([w, fam, band, r["n_eff"], r["n_g"], r["te_share"]])
        print(f"  wl {wl_um} w {w:.3f}: {sum(1 for x in rows if x[0] == w)} guided bands")
    return np.array(rows)


def main():
    widths = np.round(np.arange(0.20, 0.8001, 0.025), 4)
    data = {"widths_um": widths, "h_um": H_UM, "n_si": N_SI, "n_ox": N_OX,
            "res_sweep": RES_SWEEP, "res_fine": RES_FINE}
    for wl in (1.35, 1.55):
        print(f"width sweep at {wl} um")
        data[f"wsweep_{int(wl * 100)}"] = width_sweep(wl, widths)  # cols: w, family, band, n_eff, n_g, te_share

    print("wavelength sweep, 500 x 220 nm TE0")
    wls = np.round(np.arange(1.26, 1.6201, 0.02), 4)
    nd = [solve(0.5, wl, mp.ODD_Y, 1, res=RES_FINE) for wl in wls]
    data["wl_um"] = wls
    data["neff_nondisp"] = np.array([r["n_eff"] for r in nd])
    data["ng_nondisp"] = np.array([r["n_g"] for r in nd])
    # with material dispersion: n_eff(wl) from indices evaluated at each wl, n_g by central difference
    dwl = 0.005
    neff_d, ng_d = [], []
    for wl in wls:
        n0 = solve(0.5, wl, mp.ODD_Y, 1, n_si_li(wl), n_sio2_malitson(wl), RES_FINE)["n_eff"]
        np_ = solve(0.5, wl + dwl, mp.ODD_Y, 1, n_si_li(wl + dwl), n_sio2_malitson(wl + dwl), RES_FINE)["n_eff"]
        nm_ = solve(0.5, wl - dwl, mp.ODD_Y, 1, n_si_li(wl - dwl), n_sio2_malitson(wl - dwl), RES_FINE)["n_eff"]
        neff_d.append(n0)
        ng_d.append(n0 - wl * (np_ - nm_) / (2 * dwl))
    data["neff_disp"], data["ng_disp"] = np.array(neff_d), np.array(ng_d)

    print("mode fields, 500 x 220 nm at 1.35 um")
    te = solve(0.5, 1.35, mp.ODD_Y, 1, res=RES_FINE, fields=True)
    tm = solve(0.5, 1.35, mp.EVEN_Y, 1, res=RES_FINE, fields=True)
    data["E_te0"], data["E_tm0"] = te["E"], tm["E"]
    data["neff_te0"], data["neff_tm0"] = te["n_eff"], tm["n_eff"]
    data["ng_te0"], data["te_share_te0"], data["te_share_tm0"] = te["n_g"], te["te_share"], tm["te_share"]
    data["cell_um"] = np.array([CELL_V, CELL_L])
    print(f"TE0 n_eff {te['n_eff']:.4f} (notebook 01b reference 2.6460), n_g {te['n_g']:.3f}; "
          f"TM0 n_eff {tm['n_eff']:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **data)
    print("saved", OUT)


if __name__ == "__main__":
    main()
