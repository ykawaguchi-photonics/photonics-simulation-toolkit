# Literature notes: Si active devices (thermo-optic heaters, ring modulators)

Collected before the `si_active/` simulations, to set the platform parameters, the
material constants and the benchmarks. Each source has a tier label, following
CLAUDE.md §10:

- **A**: well-cited review, or a high-impact peer-reviewed journal from an established
  group;
- **B**: other peer-reviewed work (incl. conference proceedings);
- **C**: an unrefereed preprint, a vendor example, or a source read only as an abstract or
  snippet. These are used as concept sources only, never for a number on their own.

DOIs were checked with Crossref (2026-11-13). "Read" says how much of the source was read.

## 1. Thermo-optic phase shifters (notebook 01)

### Benchmarks

| Tier | Reference | Read | What it gives |
|---|---|---|---|
| A | M. Jacques, A. Samani, E. El-Fiky, D. Patel, Z. Xing, D. V. Plant, "Optimization of thermo-optic phase-shifter design and mitigation of thermal crosstalk on the SOI platform", *Opt. Express* 27, 10456 (2019), doi:10.1364/OE.27.010456 (168 citations) | abstract | **measured**: TiN heater P_π = 21.4 mW, τ = 5.6 µs; N++ Si heater P_π = 22.8 mW, τ = 2.2 µs; deep trenches are the best standard-process way to cut thermal crosstalk |
| C | Flexcompute, "Transient heat analysis of thermo-optic phase-shifters" (Tidy3D example notebook `TransientThermoOpticShifter.ipynb`, reproduces Jacques 2019) | full notebook source | the **full geometry and material table of Jacques 2019** (their Table 3), and the paper's **calculated** P_π = 23.9 mW (TiN) and 25.2 mW (N++), in a **2D cross-section** heat solve (see below). Used because the paper's full text could not be fetched; every number is attributed to Jacques Table 3 in the notebook |
| A | N. C. Harris et al., "Efficient, compact and low loss thermo-optic phase shifter in silicon", *Opt. Express* 22, 10487 (2014), doi:10.1364/OE.22.010487 (376 citations) | full text (arXiv:1410.3616) | doped-Si heater in the waveguide, 61.6 µm long, OpSIS 220 nm SOI, oxide clad: P_π = 24.77 mW, τ = 2.69 µs, 130 kHz, IL 0.23 dB; COMSOL 3D, dn/dT = 1.86e-4 K⁻¹; air cladding instead of oxide gave up to 25 % more phase. Table 1 compares heaters by P_π·τ (their device: 66.9 mW·µs) |
| B (via Harris Table 1) | M. R. Watts et al., "Adiabatic thermo-optic Mach–Zehnder switch", *Opt. Lett.* 38, 733 (2013), doi:10.1364/OL.38.000733 | table entry only | Si heater, air clad: P_π = 12.7 mW, τ = 2.4 µs (P_π·τ = 30.5 mW·µs) |
| C | P. Dong et al., "Thermally tunable silicon racetrack resonators with ultralow tuning power", *Opt. Express* 18, 20298 (2010), doi:10.1364/OE.18.020298 | title only | substrate undercut cuts tuning power by an order of magnitude at the cost of speed; concept only (undercut needs a 3D model and is not simulated) |

### Jacques 2019 geometry and materials (via the Tidy3D reproduction)

| Item | Value |
|---|---|
| Wavelength | 1.55 µm |
| Si layer, strip | 220 nm thick, 500 nm wide |
| Rib (N++ design) | 90 nm slab, slab width = wg + 2 × 0.8 µm; N++ strips 1 µm wide in the slab, starting 0.8 µm from each rib edge |
| BOX | 2 µm |
| TiN heater | 2 µm wide, 200 nm thick, its bottom 2 µm above the Si top (y = 2.22 µm) |
| Top oxide | to y = 2.5 µm above the BOX top; air above |
| Domain | 40 µm wide, from y = −8 µm (6 µm of Si substrate) to the cladding top |
| Boundaries | bottom fixed at 300 K; convection 10 W/(m²K) at the oxide/air surface |
| Length | 320 µm (2D: the cross-section is uniform along the heater) |
| k (W/(m·K)) | bulk Si 148, Si strip 90, Si slab 55, N++ Si 25, SiO₂ 1.38, TiN 28, air 0.026 |
| ρ (kg/m³), c_p (J/(kg·K)) | Si 2330, 711; SiO₂ 2203, 709; TiN 5240, 598 |
| dn/dT (K⁻¹) | Si 1.8e-4, SiO₂ 1e-5 |
| Phase metric | ΔΦ = 2πL/λ · dn_Si/dT · ΔT at the waveguide centre (a point temperature, not a mode overlap) |

The reduced k of the thin Si layers (90 and 55 W/(m·K) instead of 148) is phonon-boundary
scattering in sub-µm films; the values come from Jacques Table 3 and are kept as given.

### Material constants used in the O-band design

| Tier | Reference | Read | Value used |
|---|---|---|---|
| B | B. J. Frey, D. B. Leviton, T. J. Madison, "Temperature-dependent refractive index of silicon and germanium", *Proc. SPIE* 6273, 62732J (2006) (arXiv:physics/0606168) | full text | Si at 295 K: **dn/dT = 1.94e-4 K⁻¹ at 1.3 µm**, 1.90e-4 at 1.4 µm, 1.87e-4 at 1.5 µm (Table 4); n = 3.50616 at 1.3 µm, 3.49335 at 1.4 µm, 3.48314 at 1.5 µm (Table 2). Linear interpolation gives 1.936e-4 and 3.5049 at 1.31 µm |
| A | J. Komma et al., "Thermo-optic coefficient of silicon at 1550 nm and cryogenic temperatures", *Appl. Phys. Lett.* 101, 041905 (2012), doi:10.1063/1.4738989 | title/abstract | ≈1.8e-4 K⁻¹ at 1550 nm, room temperature: consistent with Frey's 1.87e-4 at 1.5 µm |
| — | I. H. Malitson, *JOSA* 55, 1205 (1965) | (already in `tfln_waveguide.py`) | SiO₂ index |
| A | Jacques 2019 Table 3 (above) | via Tidy3D | SiO₂ dn/dT = 1e-5 K⁻¹; thermal constants above |

### Hypotheses for notebook 01

1. In a 2D cross-section the temperature rise scales as the power **per unit length**, so
   P_π does not depend on the heater length. A ring tuned over one full FSR needs 2 P_π
   whatever its radius (heater over the whole ring).
2. Our FV solve with Jacques' geometry should land near the paper's calculated 23.9 / 25.2 mW
   and within ~15 % of the measured 21.4 / 22.8 mW.
3. Mode-weighted Δn_eff (the physically correct metric) differs from the point-temperature
   metric by the mode's overlap with the hot core and n_g/n_eff effects; expect a few-%
   to ~10 % difference.
4. TiN above the waveguide is slower (heat crosses 2 µm of oxide); a doped-Si heater in the
   slab is ~2–3× faster at similar P_π (Jacques: 5.6 vs 2.2 µs).
5. Deep trenches beside the heater raise efficiency and cut crosstalk, but slow the heater.

## 2. Ring modulators (notebook 02) — collected for the plan, detailed in 02

| Tier | Reference | Read | What it gives |
|---|---|---|---|
| A | R. A. Soref, B. R. Bennett, "Electrooptical effects in silicon", *IEEE JQE* 23, 123 (1987), doi:10.1109/JQE.1987.1073206 | — | plasma dispersion, origin |
| A | M. Nedeljkovic, R. Soref, G. Z. Mashanovich, "Free-carrier electrorefraction and electroabsorption modulation predictions for silicon over the 1–14 µm infrared wavelength range", *IEEE Photon. J.* 3, 1171 (2011), doi:10.1109/JPHOT.2011.2171930 | — | Δn, Δα coefficients at 1.3 and 1.55 µm (to be read from the paper before use) |
| A | J. Müller et al., "Optical peaking enhancement in high-speed ring modulators", *Sci. Rep.* 4, 6310 (2014), doi:10.1038/srep06310 | full text | **benchmark**: R = 10.3 µm, 400 nm rib in 220 nm SOI with 90 nm slab; p⁻ 3e17 / n⁻ 1e17 cm⁻³, 50 nm overlap, p⁺/n⁺ 1e20; C_j = 17 fF at −1 V, R_s = 215 Ω, RC bandwidth 43 GHz; Q 3500–5000; detuning-induced optical peaking raises the EO bandwidth from 30 to 40 GHz |
| A | W. D. Sacher, J. K. S. Poon, "Dynamics of microring resonator modulators", *Opt. Express* 16, 15741 (2008), doi:10.1364/OE.16.015741 | — | time-domain CMT, photon-lifetime limit |
| A | Q. Xu et al., "Micrometre-scale silicon electro-optic modulator", *Nature* 435, 325 (2005), doi:10.1038/nature03569 | — | first Si ring modulator |
| A | G. T. Reed et al., "Silicon optical modulators", *Nat. Photon.* 4, 518 (2010), doi:10.1038/nphoton.2010.179 | — | review |
| A | W. Bogaerts et al., "Silicon microring resonators", *Laser Photon. Rev.* 6, 47 (2012), doi:10.1002/lpor.201100017 | — | review |
| A | E. Timurdogan et al., "An ultralow power athermal silicon modulator", *Nat. Commun.* 5, 4008 (2014), doi:10.1038/ncomms5008 | — | vertical junction, athermal operation |
| A | Y. Yuan et al., "A 5 × 200 Gbps microring modulator silicon chip empowered by two-segment Z-shape junctions", *Nat. Commun.* 15, 918 (2024), doi:10.1038/s41467-024-45301-3 | snippet | state of the art for CPO |
| A | K. Padmaraju, K. Bergman, "Resolving the thermal challenges for silicon microring resonator devices", *Nanophotonics* 3, 269 (2014), doi:10.1515/nanoph-2013-0013 | — | thermal tuning and locking (notebook 05) |
| A | T. Carmon, L. Yang, K. J. Vahala, "Dynamical thermal behavior and thermal self-stability of microcavities", *Opt. Express* 12, 4742 (2004), doi:10.1364/OPEX.12.004742 | — | thermal bistability (notebook 05) |
