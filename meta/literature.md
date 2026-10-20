# Literature notes: SiN grating couplers and meta-gratings

Collected before any simulation, to set the hypotheses for this study. Each source is
labelled by tier, following CLAUDE.md §10:

- **A**: well-cited review, or a high-impact peer-reviewed journal from an established
  group;
- **B**: other peer-reviewed work;
- **C**: unrefereed preprint, used as a concept source only (never for a number).

Numbers and design rules in this study rest on tier A/B sources.

## Reviews and platform overviews (tier A)

| Reference | Group | Use here |
|---|---|---|
| Marchetti, Lacava, Carroll, Gradkowski, Minzioni, "Coupling strategies for silicon photonics integrated chips [Invited]", *Photonics Research* 7, 201 (2019) | Pavia / Tyndall | fiber-chip coupler overview: out-of-plane (grating) couplers trade efficiency and bandwidth for wafer-scale testing; design principles |
| Rahim et al., "Expanding the silicon photonics portfolio with silicon nitride photonic integrated circuits", *J. Lightwave Technol.* (2017) | Ghent-imec (Roelkens, Van Thourhout, Baets) | survey of SiN platforms (thicknesses, BOX, foundries) |

## SiN grating couplers

| Tier | Reference | Stack | Per period | Result | Notes |
|---|---|---|---|---|---|
| B | Korček et al., *Sci. Rep.* 13 (2023), PMC10576773 (Cheben/Benedikovič group) | SiN 400 nm, full etch, BOX 6 µm, air top, 1.55 µm, SMF-28 (MFD 10.4 µm) | 1 tooth + 1 trench, Λ = 860 nm, DC 70 %; SWG variant; 7-period apodized variant | uniform −3.6 dB sim / −3.9 dB meas., directionality 72 %; apodized −2.7 / −2.9 dB | fabricated and measured; min feature 100 nm (DUV); Lumerical 2D/3D FDTD, **uniform 10 nm mesh**. SiN worse than SOI: moderate index contrast → weak scattering per period → poorer overlap and directionality. **Reproduced in Step 1: −3.67 dB, 73 %.** |
| B | Fraser et al., *Sci. Rep.* 14 (2024), PMC11109221 (Ye / Cheben) | SiN 400 + 50 nm oxide + α-Si 220 nm, air top, BOX 4.5 µm, 1.31 µm | 1 tooth + 1 trench, Λ 600–700 nm (1st order); transverse SWG for apodization | −1.3 dB (3D sim.), directionality 85 % | simulation only; GA + Lumerical 2D; min feature 120 nm. The α-Si overlay breaks the vertical symmetry |
| B | Vitali et al., *Sci. Rep.* 12 (2022), PMC9474549 (Southampton: Petropoulos, Gardes) | SiN 150–500 nm, dual-level grating | two scattering levels | −0.45 dB (400 nm SiN, sim., no reflector) | PSO. Single-level partial-etch SiN needs > 800 nm thickness for > 80 % directionality |
| B (secondary, not yet checked against the original) | Hong et al. (2009): fully etched grating on 400 nm SiN | | | predicted ≈ −0.4 dB, measured ≈ −1.2 dB (C–L band) | gap between prediction and measurement is a reminder to compare like with like |
| C | arXiv 2604.28058 (inverse design, multimode GCs) | SiN/air, SiN/SiO₂ | density-based adjoint | ≈ 66 % / 60 % mean | concept only |

## Several elements per period

| Tier | Reference | Idea | Relevance |
|---|---|---|---|
| A | Benedikovic et al., *Opt. Lett.* 40, 4190 (2015); L-shaped, *Opt. Lett.* (2017) (Cheben, Halir, Ortega-Moñux) | interleaved deep + shallow trenches (scatterers at two heights) | blazing: up to 98 % directionality, −1.3 dB on 220 nm SOI. Needs two etch depths |
| A | Sell et al., *Nano Lett.* (2017) (Fan, Stanford) | freeform multi-element metagratings, adjoint topology optimization | free-space deflection; establishes the adjoint method for multi-element periods |
| C | arXiv 2607.01477 (SOI meta-grating) | two ridges per super-period (a ≠ b); a = b is a non-radiating SWG, a small asymmetry creates a radiating super-period | concept only: continuous weak coupling-strength control |

## Hypotheses for this study

1. With **one shared etch depth**, rearranging meta-atoms in the plane changes the
   period's in-plane Fourier content (grating strength, Bragg back-reflection, parasitic
   orders) but not the up/down asymmetry, which comes from the vertical profile and the
   BOX/substrate reflection (supported by the tier-A blazing work, which needs two
   heights). Expected gains are in **mode match and reflection**, not directionality.
2. SiN's long first-order period (≈ 0.9 µm vs 0.51 µm for Si) leaves room for three
   atoms with ≥ 100 nm features; Si does not.
3. Large directionality gains would need vertical asymmetry: per-atom etch depths
   (dual-level, interleaved) or an overlay.

## Standard platform values used

- BOX: chosen interference peak nearest a standard value: imec 200 mm pilot line 2.9 µm
  (2.5 µm thermal + 0.4 µm HDP), standard thermal-oxide wafers 3 µm, Applied Nanotools
  400 nm SiN MPW 4.5 µm.
- Minimum feature 100 nm (DUV, as in Korček 2023).
- SiN 400 nm (Korček, Fraser, Applied Nanotools).

## a-Si overlay and dual polarization (Step 4)

| Tier | Reference | Use here |
|---|---|---|
| B | Fraser et al., *Sci. Rep.* 14 (2024), PMC11109221 (Cheben group) | design of Step 4: SiN 400 nm / 50 nm buffer / a-Si 220 nm full etch, Λ 700 nm, DC 28 %, 31 periods, BOX 4.5 µm, air, 1.31 µm, SMF-28 at −4°; reported −1.7 dB, directionality 93 % (Step 4: −1.66 dB, 0.93) |
| A | Kohli et al., *ACS Photonics* 10, 3366 (2023), PMC10515627 (ETH Zurich, Leuthold; Ligentec) | one a-Si-overlay grating for TE and TM on 800 nm SiN at the same fiber angle (Λ 765 nm): measured TE −3.0 dB, TM −2.3 dB, PDL < 1 dB over 40 nm; TE/TM matching was not reached where the foundry fixes the SiN/BOX thickness |
| B | Polarization-independent micro-antenna with an SWG metamaterial, *Sci. Rep.* (2025) | TE/TM Bloch-index matching by engineering the birefringence with a sub-wavelength metamaterial |

