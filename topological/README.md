# Dielectric topological photonic crystals: spin-Hall and valley-Hall

**Question:** how do the two all-dielectric routes to topological edge states — the
spin-Hall crystal of Wu & Hu (C6 deformation, band inversion at Γ) and the valley-Hall
crystal (broken inversion, gaps at K) — compare as pillar (TM) and air-hole (TE)
crystals, as one-way waveguides excited by circularly polarized sources, and as
waveguides fed from a standard 500 nm silicon strip?

**Model:** Meep / MPB 2D, non-dispersive, silicon εd = 11.7 and air. This deliberately
uses the bulk silicon index of the Wu–Hu paper instead of the toolkit's 2D effective index
(see the module docstring). Spin-Hall lengths in a0, valley-Hall lengths in a = a0/√3;
at 1550 nm a0 = 632 nm and a = 367 nm.

| Notebook | Content | Key results |
|---|---|---|
| `notebooks/01_unit_cells_and_edge_states.ipynb` | permittivity maps of the eight unit cells; bulk bands (fine Γ–M–K sweep, points; p/d character for spin-Hall); ribbon edge bands of the four walls | shared bulk gaps: spin TM 4.1 % (centre 0.4624 c/a0, Wu–Hu 0.4629), spin TE 3.1 %, valley TE 6.8 % at K, valley TM 8.7 %; one edge band crosses each gap (v_g 0.11–0.21 c) |
| `notebooks/02_domain_walls_chiral_excitation.ipynb` | four domain walls driven by S± circular dipoles; Re Hz / Re Ez and \|E\|² maps; power split and directionality | source placed from the edge mode's local circular polarization (MPB) and coupling strength; one-way excitation: directionality 0.994 (spin TM), 0.990 (spin TE; converged only at 64/a0 and 20 rows, drive 0.004 above the gap edge), 0.997 (valley TE), 0.998 (valley TM); source at a C2 centre gives 0 |
| `notebooks/03_waveguide_to_topological_coupling.ipynb` | 500 nm strip → TE walls: butt coupling, line defect, parity rule; topology-optimized transitions (history, structures, before/after) | edge-channel coupling, butt → optimized: spin 5.9 % → 48.7 % (band average, ≈ 11 nm), valley 0 % → 77.1 % (≈ 62 nm, peak 80.9 %, R 2.4 %); 100 nm minimum features |

Code: `src/pic_toolkit/meep_sim/topological_pc.py` (crystals, unit cells, ribbons, walls,
strip feed). Long runs: `scripts/compute_bands.py`, `scripts/chirality_maps.py`, `scripts/scan_spin_sources.py`, `scripts/scan_spin_rows.py`, `scripts/run_walls.py`,
`scripts/run_coupling.py`, `scripts/topopt_transition.py` (run from this folder in the
`mp` env). Literature (tiered by source quality): `literature.md`. Parameter history:
`docs/simulation_settings_record.md`, section `meep_sim/topological_pc.py`.

`data/` holds every cached run (`bands/`, `walls/`, `coupling/`, `topopt/`) and the two
design points `topopt_{spin,valley}_transition.yaml`. The notebooks reload these caches, so
a re-run takes minutes; deleting them re-runs the study (bands ≈ 1 h, walls ≈ 4 h,
coupling ≈ 3 h, each topology optimization 6–12 h of single-core FDTD). Field maps are
stored at every second grid point; all power numbers are evaluated at full resolution.

## Notes

- Spin-Hall walls: topological crystal (a0/R = 2.9) below, trivial (3.125) above; the
  spin-Hall TE edge bands cross at kx = 0 with a small anticrossing, so drive frequencies
  sit on the lower branch.
- Valley-Hall walls are bearded with the large inclusions facing; the wall is
  mirror-symmetric about its line and C2-symmetric about every point of the line at
  x = k/2. Chiral sources therefore cannot sit on that line; they go where the edge mode
  is circularly polarized and strong (here near the centre of a small inclusion in the
  second row), and the even fundamental strip mode cannot couple to the (odd) TE edge mode without breaking the
  mirror symmetry.
