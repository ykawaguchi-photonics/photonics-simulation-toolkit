# si_active: silicon active devices on the real cross-section

The `si/` track models components in a 2D top view with an effective core index of 2.7.
Active devices act through the **real cross-section**: where the heat or the carriers sit
relative to the optical mode. This track therefore solves the 220 nm SOI cross-section
with bulk indices (Si: Frey et al. 2006; SiO₂: Malitson 1965) in MPB, and couples it to
electrical or thermal solvers on the same cross-section.

| Notebook | Device | Status |
|---|---|---|
| `notebooks/01_thermo_optic_phase_shifter.ipynb` | TiN and N++ heaters: P_π, speed, trenches, crosstalk, ring tuning; Jacques et al. 2019 benchmark | done |
| `02_ring_modulator` | PN depletion ring modulator: pm/V, C_j, RC, CMT S21, optical peaking; Müller et al. 2014 benchmark | planned |
| `03_depletion_mzm` | traveling-wave depletion MZM (replaces `si/08`'s imposed Δn) | planned |
| `04_ge_photodetector` | Ge-on-Si PD: responsivity, transit/RC bandwidth | planned |
| `05_ring_wavelength_locking` | heater + monitor PD + controller; thermal bistability | planned |

Shared code: `src/pic_toolkit/active/` (`thermal.py` 2D finite-volume heat solver,
`si_xsection.py` MPB cross-section and perturbation overlaps, `thermo_optic.py` heater
figures of merit). SAX model: `models/thermo_optic_phase_shifter.py`. Data and design points
stay in `si_active/data/`.

Common assumptions: O-band design wavelength 1.31 µm (benchmarks at the reference's own
wavelength), 220 nm Si / 2 µm BOX, optics non-dispersive at λ₀ except a separately reported
material-dispersive group index; no TCAD (carriers by the depletion approximation in 02-03).
Sources and their tiers: [literature.md](literature.md).
