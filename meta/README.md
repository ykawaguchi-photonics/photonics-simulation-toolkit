# Meta-grating study: SiN grating coupler, conventional vs meta-atoms

**Question:** can a meta-grating (two or three tooth/groove "meta-atoms" per period) couple a
SiN waveguide to an SMF-28 fiber more efficiently than a conventional grating (one
tooth/groove per period), when both are optimized the same way?

**Model:** Meep 2D x-z, SiN 400 nm (n = 2.0) / BOX / Si substrate, oxide top cladding,
TE at 1.35 µm, SMF-28 (MFD 9.2 µm) at 10°. Real bulk indices, non-dispersive,
y-invariant. Both designs share one period P fixed by the first-order phase-matching
condition, with every width ≥ 100 nm and the widths summing to at most P.
Optimization: Meep adjoint (∂η/∂ρ on a MaterialGrid) chained through a differentiable
fill-fraction parametrization, and Adam with a projection onto the constraints.

| Notebook | Content | Result |
|---|---|---|
| `notebooks/01_sin_validation_mesh.ipynb` | slab mode, propagation and normalization, overlap definition, domain, mesh convergence, design period, BOX (standing wave, standard value), smoothness, adjoint vs finite differences, literature benchmark | res 60 for optimization, res 160 for evaluation; P = 0.926 µm; BOX 4.58 µm (nearest interference peak to the 4.5 µm standard); Korček et al. 2023 reproduced within 0.1 dB |
| `notebooks/02_uniform_adam.ipynb` | conventional grating (1 atom/period): landscape map + Adam from three starts | −4.14 dB (tooth 0.53 µm, etch 203 nm), directionality 0.73 |
| `notebooks/03_metagrating_adam.ipynb` | meta gratings with 3 and 2 atoms/period: screening + Adam refinement, comparison with the conventional grating, conclusion | 2 atoms −5.04 dB (−0.90 dB), 3 atoms −7.49 dB (−3.35 dB); same directionality, weaker grating |
| `notebooks/04_asi_overlay.ipynb` | grating in an a-Si overlay on SiN (design of Fraser et al. 2024): modes, mesh, coupling, buffer, TE vs TM | −1.66 dB at 1.31 µm, directionality 0.93 (vertical asymmetry); TM −6.4 dB at its own angle |

**Conclusion.** With one shared etch depth and ≥ 100 nm features on 400 nm SiN, meta-atoms
do not raise the coupling efficiency. Directionality is unchanged (all atoms scatter at
the same height), and the small SiN/oxide index contrast averages the sub-wavelength
features away, so the grating's radiating first-order component weakens. Meta-atoms would
need vertical asymmetry (per-atom etch depths, an overlay) or a higher-contrast platform
to help. Step 4 confirms the overlay route: a grating etched into an a-Si overlay
reaches directionality 0.93 and −1.66 dB.

Code: `src/pic_toolkit/meep_sim/sin_grating.py` and `asi_grating.py` (Step 4). Literature (tiered by source quality):
`literature.md`. Run in the `mp` conda env from `meta/notebooks/`.

`data/` holds every cached run (`step1`, `step2`, `step3_3atoms`, `step3_2atoms`,
`step4_asi_overlay`) and the
three design points (`sin_grating_{conventional,meta2,meta3}.yaml`). The notebooks reload
these caches, so a re-run takes minutes; deleting them re-runs the full study (several
days of single-core FDTD). The res-160 field maps in the final-design caches are stored
at every fourth grid point to keep the files small; all numbers are unaffected.

## Parameter notes

- With P fixed and the slack P − Σw added to the last groove, that groove's own width
  has no effect (its derivative is exactly zero). Free parameters: 2 for the
  conventional grating (tooth width, etch), 4 for two atoms (3 widths, etch), 6 for three
  atoms (5 widths, etch).
- The near-field line sits 1.0 µm above the SiN. At 0.5 µm the guided mode's
  evanescent tail dominates the phase there (apparent emission angle 28° instead of 7°);
  from 1.0 µm up the angle is stable. η itself barely changes, since the overlap
  integral rejects the evanescent part.
- The eigenmode source/monitor span is core ± 1 µm. A span that reaches the Si
  substrate returns the substrate's slab mode (n_eff ≈ 3.39).
- Independent runs in parallel processes give no speedup on this machine (4 runs at
  once take 4× longer each), so everything runs sequentially.
