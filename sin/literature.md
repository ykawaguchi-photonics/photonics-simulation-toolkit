# Literature notes: 400 nm SiN strip waveguides and bends

Collected before the SiN-track simulations, to set the platform parameters and the
hypotheses for `si/notebooks/01`–`02`. Each source has a tier label, following CLAUDE.md §10:

- **A**: well-cited review, or a high-impact peer-reviewed journal from an established
  group;
- **B**: other peer-reviewed work;
- **C**: an unrefereed preprint, or a source read only as an abstract or snippet. These are
  used as concept sources only, never for a number.

The platform choices rest on tier A/B sources. Where only the abstract was read, the entry
says so.

## Platform: thickness, width, cladding

| Tier | Reference | What it gives |
|---|---|---|
| A | Rahim et al., "Expanding the silicon photonics portfolio with silicon nitride photonic integrated circuits", *J. Lightwave Technol.* 35, 639 (2017) (Ghent-imec: Roelkens, Baets) | survey of SiN platforms (thickness, BOX, foundries); already used in `meta/literature.md` |
| A | Blumenthal, Heideman, Geuzebroek, Leinse, Roeloffzen, "Silicon nitride in silicon photonics", *Proc. IEEE* 106, 2209 (2018) | review of low-, medium- and high-confinement SiN; background only (full text not read here) |
| B | Muñoz et al., "Silicon nitride photonic integration platforms for visible, near-infrared and mid-infrared applications", *Sensors* 17, 2088 (2017), PMC5620990 | platform table: **400 nm LPCVD SiN**: IME 1000 nm wide (0.32 dB/cm at 1270 nm), **Toronto-IME (Poon) 900 nm wide (0.34 dB/cm at 1270 nm)**, IME 700 nm wide; 300 nm CNM 1000 nm wide with R = 150 µm "negligible bend loss" (no curve). Thick SiN (700–950 nm) platforms list minimum radii of 20–230 µm, thin (≤ 200 nm) ones 50–2000 µm |
| B | Applied Nanotools SiN process (CMC Microsystems page); stack confirmed in Pita Ruiz et al. 2025 below | 400 nm SiN / 4.5 µm BOX / 3 µm SiO₂ top cladding, e-beam lithography: the same stack as `meta/` |

**Choice:** 400 nm SiN (n = 2.0, non-dispersive, as in `meta_sim/sin_grating.py`), fully clad in
SiO₂ (1.44), **width 0.9 µm** (Toronto-IME). It is a standard MPW stack (Applied Nanotools,
IME), with no custom process.

## Bend loss of ~400 nm SiN

| Tier | Reference | Structure | Result |
|---|---|---|---|
| A/B | Pita Ruiz, Dalvand, Ménard, "Inverse-designed 90-degree silicon nitride bends for the C band", *J. Lightwave Technol.* 43, 5804 (2025) (abstract) | 850 × 400 nm SiN in SiO₂, e-beam, C band; footprints 6 × 6 to 21 × 21 µm². The group's indices are SiN 1.997 and SiO₂ 1.441 at 1550 nm (Nature Commun. 2025, PMC12540863) | **circular R = 6 µm: 2.88 dB (sim.) / 2.86 dB (meas.) per 90°**; Euler of the same radius worse (3.46 / 3.44 dB); freeform bends reach 0.08–0.11 dB at R = 16–21 µm |
| B | Song et al. (imec), "Advanced bends and micro ring resonators in silicon nitride photonic waveguides for C-band", *IEEE Photonics J.* 16(6) (2024), doi:10.1109/JPHOT.2024.3509312 (abstract and search snippet only) | SiN strip, C band (thickness not confirmed from the full text) | advanced 25 µm bend ≈ 0.025 dB/90°, **about the same as a circular 50 µm bend** |
| B | Vogelbacher et al., "Analysis of silicon nitride partial Euler waveguide bends", *Opt. Express* 27, 31394 (2019) | SiN, 850 nm | concept: an Euler bend trades the straight-to-bend transition loss against radiation from its tighter peak curvature |
| C | "Low-loss silicon nitride bent waveguides at O-band with modified Hermite curves", *Photonics* 13, 175 (2026) (search snippet; simulation only) | SiN, 1311 nm | circular reference ≈ 0.08 / 0.02 / 0.006 / 0.001 dB at R = 15 / 20 / 25 / 30 µm (concept only) |

| B | Bi et al., "Integrated silicon nitride devices via inverse design", *Nature Commun.* (2025), PMC12540863 (full text read) | the same 400 nm SiN platform | states that an 850 nm wide strip needs R ≈ 50 µm for "negligible" bend loss (no number) |

## Si bend loss: oxide-clad 220 nm strip

Only oxide-clad references are used to check the Si reference against.

| Tier | Reference | Structure | Result |
|---|---|---|---|
| A | Bahadori, Nikdast, Cheng, Bergman, "Universal design of waveguide bends in silicon-on-insulator photonics platform", *J. Lightwave Technol.* 37, 3044 (2019) (full text read) | 400 × 220 nm Si in SiO₂, TE₀, 3D FDTD, ~1.55 µm (Si 3.5, SiO₂ 1.45) | circular 90° bend fitted as arc radiation **181.98 R^−2.49 dB/cm** + mode mismatch **0.1315 R^−2.37 dB per junction**: 0.29 / 0.12 / 0.061 / 0.025 / 0.008 dB at R = 1 / 1.5 / 2 / 3 / 5 µm. Their directly simulated circular bend at R = 5 µm: 0.0125 dB. Mismatch dominates at small R |
| A (theory) | D. Marcuse, "Bending losses of the asymmetric slab waveguide", *Bell Syst. Tech. J.* 50, 2551 (1971) | 2D slab bend, exact large-R asymptotics | radiation loss ∝ exp(−cR) with **c = 2γ³/(3β²)**; used in notebook 02 §9.5 to predict the Si/SiN radius ratio |
| A (not comparable) | Vlasov and McNab, "Losses in single-mode silicon-on-insulator strip waveguides and bends", *Opt. Express* 12, 1622 (2004) (full text read) | 445 × 220 nm Si **between two 2 µm air trenches, no top oxide**, 1.5 µm | 0.086 dB at R = 1 µm, 0.013 dB at R = 2 µm. Air cladding confines more strongly, so this is not used as a reference for oxide-clad strips |

## Hypotheses for this study

1. The 2D core index is the TE₀ slab index of the 400 nm film:
   `slab_modes.te_slab_index(0.4, 2.0, 1.44, 1.35)` = **1.779**. The same function gives
   2.947 for 220 nm Si. The Si track's 2.7 is a simplified index that corresponds to a
   ~158 nm film, so it is *less* confined than real 220 nm Si.
2. At w = 0.9 µm the lateral TE₀ index is ≈ 1.69, against 2.42 for the Si track. The
   lateral contrast to the cladding (Δn ≈ 0.25 vs ≈ 0.98) and the evanescent decay
   constant (≈ 4.1 vs ≈ 9.0 µm⁻¹) are both much smaller. In this 2D model a weakly guided
   TE₁ also exists at 0.9 µm (cutoff ≈ 0.62–0.67 µm across the band).
3. Bend loss is therefore radiation-limited out to much larger radii:
   - **≈ 3 dB per 90° at R ≈ 5–6 µm**: Pita Ruiz 2025 at C band. The O band is somewhat
     lower loss for the same geometry.
   - **≤ 0.01 dB per 90° at R ≈ 20–40 µm**, against ≈ 2–3 µm for the Si track. That is a
     ratio of roughly 10×.
4. As in Si, a gdsfactory Euler bend of the same footprint loses to the circular bend at
   small R, because its peak curvature is tighter. It can win only where the
   straight-to-arc mode mismatch, not radiation, sets the loss.

### Revision after the first run (2026-10-29)

The first version of the comparison used hypothesis 1 and the Si track (core 2.7) as the Si
partner. Checking it against the sources above showed two problems.

- **The Si track is not a valid Si reference.** Its lateral n_eff is 2.418, while a real
  500 × 220 nm strip has 2.671 (MPB cross-section solve). Its bends lose several times
  more than Bahadori's 3D FDTD, even though Bahadori's 400 nm strip is the weaker guide.
  The Si-vs-SiN radius ratio therefore came out too small (6–9×).
- **The slab-EIM index overestimates n_eff.** For SiN it gives 1.6878 against the real
  1.6753. That is 5 % too much index excess over the cladding, so it predicts too little
  radiation.

Revised approach (notebook 01 §1.1, notebook 02 §9):
- Both platforms use the **n_eff-matched** 2D core index: SiN 1.7650, Si reference 2.944.
- The calibrated model is first benchmarked at the publications' own geometry and
  wavelength: Bahadori (Si 400 × 220 nm) and Pita Ruiz (SiN 850 × 400 nm, R = 6 µm).

A 2D effective-index model is not a 3D solver: a full 3D bend radiates also into the
cladding above and below the film, and the EIM lateral contrast is approximate. Compare
with the literature on the factor level, not the third digit.

### Outcome (2026-10-29, calibrated runs)

All numbers are at the band centre, from notebook 02 §9; all 45 FDTD runs passed the
S-parameter checks.

- **Benchmarks.** The calibrated 2D model *overestimates* bend loss against 3D results, by a
  similar factor on both platforms.
  - Si 400 × 220 nm (Bahadori): 1.38 / 0.43 / 0.19 / 0.065 / 0.023 dB at R = 1 / 1.5 / 2 /
    3 / 5 µm. That is 2.6–4.7× their fitted curve, and 1.8× their directly simulated
    0.0125 dB at 5 µm.
  - SiN 850 × 400 nm, R = 6 µm (Pita Ruiz): circular 5.2 dB, Euler 6.8 dB, against
    2.88 / 3.46 dB (3D simulation) and 2.86 / 3.44 dB (measured), i.e. 1.8× and 2.0×. The
    Euler-worse-than-circular ordering is reproduced (ratio 1.31, against 1.20).
  - Because loss falls exponentially with R, a factor of 2–3 in loss is a modest shift in
    radius. Read off the curves, it is about +1 µm (≈ 1.5×) for Si at 2–3 µm and about
    +1.5 µm (≈ 1.25×) for SiN at 6 µm. The 2D radii are therefore conservative.
- **Hypothesis 1** was replaced by the n_eff-matched core (see the revision above).
- **Hypothesis 3** holds.
  - At 1.35 µm the SiN Euler bend reaches 0.1 dB at 13.8 µm and 0.01 dB at 20.1 µm; the
    circular bend reaches 0.1 dB at 20.1 µm.
  - The calibrated Si reference reaches the same levels at 1.71 / 2.51 / 2.62 µm.
  - The ratio is **7.7–8.1×** at every level that both reach. Marcuse's
    c = 2γ³/(3β²) predicts 7.1× (c = 4.95 vs 0.694 µm⁻¹).
- **Hypothesis 4** holds. The Euler bend is worse than the circular one up to R = 10 µm in
  SiN (1.25 µm in Si) and better beyond.
- **New effect.** At 1.35 µm the 500 nm Si strip guides TE1 (2D n_eff 1.828). The circular
  bend's junctions couple TE0↔TE1, and its loss ripples with period 2L_B/π = 1.02 µm in R.
  The ripple is not seen in the single-mode 400 nm Bahadori geometry, or in Euler bends.

