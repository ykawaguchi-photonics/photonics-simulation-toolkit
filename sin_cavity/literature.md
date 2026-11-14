# Literature: SiN photonic-crystal nanocavities in the visible

Tiers: **review**, **high-impact** (Nature family, Science, PRL), **peer-reviewed** (Opt. Express,
Opt. Lett., APL, Sci. Rep. …), **preprint** (concept only, never the basis for a number).
Numbers marked "sim." are calculated, "meas." are measured.

## Concept

| Source | Tier | What it sets |
|---|---|---|
| S. Noda, M. Fujita, T. Asano, "Spontaneous-emission control by photonic crystals and nanocavities", Nat. Photon. 1, 449 (2007) | review | The study's starting point: 2D PhC slab nanocavities (L3, double heterostructure) with Q ~ 10^6 in Si, Purcell enhancement F = (3/4π²)(λ/n)³ Q/V and emission inhibition inside the bandgap. |
| Y. Akahane, T. Asano, B.-S. Song, S. Noda, "High-Q photonic nanocavity in a two-dimensional photonic crystal", Nature 425, 944 (2003) | high-impact | L3 cavity (three missing holes) in a triangular hole lattice; shifting the two end holes outward by ≈ 0.15a raises Q from ≈ 5×10³ to 4.5×10⁴ (meas.) by making the field envelope gentler (fewer components inside the light cone). |
| B.-S. Song, S. Noda, T. Asano, Y. Akahane, "Ultra-high-Q photonic double-heterostructure nanocavity", Nat. Mater. 4, 207 (2005) | high-impact | Heterostructure cavity: a short section of larger lattice constant in a line-defect waveguide, Q ≈ 6×10⁵ (meas.) in Si. |
| L. C. Andreani, D. Gerace, "Photonic-crystal slabs with a triangular lattice of triangular holes investigated using a guided-mode expansion method", Phys. Rev. B 73, 235114 (2006) | peer-reviewed | Guided-mode expansion (GME): the field is expanded in the guided modes of the unpatterned slab, and the radiative loss (Q) follows from perturbation theory. Basis of `legume`, used here to screen L3 shifts. |
| M. Minkov, V. Savona, "Automated optimization of photonic crystal slab cavities", Sci. Rep. 4, 5124 (2014) | peer-reviewed | L3 in Si (n 3.46, d = 0.55a, r = 0.25a): Akahane-type shifts Q = 2.6×10⁵ (sim., Meep FDTD); three optimized end-hole shifts (0.327a, 0.257a, 0.116a) Q = 1.6×10⁶ (sim., Meep), V = 0.94 (λ/n)³. Sets the Si reference and the shift parametrization. |

## SiN cavities in the visible

| Source | Tier | What it sets |
|---|---|---|
| M. W. McCutcheon, M. Lončar, "Design of a silicon nitride photonic crystal nanocavity with a quality factor of one million for coupling to a diamond nanocrystal", Opt. Express 16, 19136 (2008) (arXiv 0809.5066) | peer-reviewed | **Reproduction target.** SiN (n = 2.0) ridge 200 nm × 300 nm in air, λ = 637 nm (NV ZPL), circular holes, mirror a = 250 nm, r = 70 nm (r/a 0.28), 17 mirror periods; 4-hole taper toward the centre (a, r) = (250, 66) → (238, 62) → (226, 58) → (214, 55) nm; cavity gap s ≈ 95 nm. Q = 2.3×10⁵, V = 0.55 (λ/n)³ (sim., Lumerical 3D FDTD, 5 nm mesh at the centre, 10 nm elsewhere). 2/3/4-hole tapers: Q 1.3×10⁴ / 2.3×10⁴ / 2.3×10⁵; 11 mirror periods halve Q. The definition of s is not stated ("the gap s between two taper/mirror sections"); the edge-to-edge reading reproduces V. Q vs s is sharply peaked (values read from the vector data of their Fig. 4): s = 80/90/93/95/97/100/110 nm → Q ≈ 1.2×10⁴ / 7.7×10⁴ / 1.6×10⁵ / 2.3×10⁵ / 1.9×10⁵ / 1.1×10⁵ / 2.1×10⁴, V ≈ 0.46/0.52/–/0.55/–/0.58/0.64 (λ/n)³, so a few nm of s (or of effective geometry) moves Q several-fold. The arXiv title says "ultrahigh Q"; the published title says "one million" for a variant. |
| M. Khan, T. Babinec, M. W. McCutcheon, P. Deotare, M. Lončar, "Fabrication and characterization of high-quality-factor silicon nitride nanobeam cavities", Opt. Lett. 36, 421 (2011) (arXiv 1010.6287) | peer-reviewed | Fabricated SiN nanobeams, Q up to 5.5×10⁴ (meas., spectrometer-limited), resonances tunable over 600–700 nm by scaling. The fabricated design is the McCutcheon–Lončar 4-hole taper, for which the paper gives V_m = 0.55 (λ/n)³ (sim.); V is not measured. Fabrication sets the realistic ceiling: designs above ~10⁵ are disorder-limited. |
| M. Barth, J. Kouba, J. Stingl, B. Löchel, O. Benson, "Modification of visible spontaneous emission with silicon nitride photonic crystal nanocavities", Opt. Express 15, 17231 (2007); M. Barth et al., "Emission properties of high-Q silicon nitride photonic crystal heterostructure cavities", APL 93, 021112 (2008) | peer-reviewed | Suspended SiN 2D PhC slab cavities in the visible: Q ≈ 3.4×10³ (meas., heterostructure). Other groups report SiN L3 Q 1.5–2.7×10³ (meas.). The 2D slab is the low-Q route in SiN. |
| Q. Quan, M. Lončar, "Deterministic design of wavelength scale, ultra-high Q photonic crystal nanobeam cavities", Opt. Express 19, 18529 (2011) (arXiv 1108.2675) | peer-reviewed | Design recipe: zero cavity length, constant period, hole filling fraction tapered quadratically (mirror strength rising linearly → Gaussian envelope), ≥ 15 taper segments per side plus extra full-strength mirrors. Si examples only (Q > 10⁸ sim.). Used for the own SiN design. |

## Platform and parameters

- **Membrane:** stoichiometric LPCVD Si₃N₄, 200 nm, suspended in air (KOH or HF undercut). 200 nm
  is the thickness used by McCutcheon & Lončar and a standard LPCVD membrane thickness; it is not
  the toolkit's 400 nm SiN track (which is SiO₂-clad and used at 1550 nm).
- **Index:** n = 2.0, non-dispersive, lossless (McCutcheon & Lončar). LPCVD Si₃N₄ is ≈ 2.02 at
  637 nm; absorption in the visible is low enough not to limit Q at 10⁵ (Khan 2011).
- **Minimum feature:** e-beam lithography, holes ≥ 100 nm diameter and walls ≥ 50 nm.

## Hypotheses for this study

1. The McCutcheon–Lončar nanobeam reproduces Q ≈ 2×10⁵ and V ≈ 0.55 (λ/n)³ in 3D Meep.
2. An L3 cavity in the same 200 nm SiN membrane has a small TE-like gap (n = 2.0) and reaches
   only Q ~ 10³–10⁴ even with end-hole shifts, in line with the measured 1.5–3.4×10³.
3. The nanobeam's Purcell factor is 10–100× the L3's, which is why SiN cavities for emitters are
   nanobeams rather than the 2D slabs of the Noda review.
