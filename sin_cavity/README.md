# SiN photonic-crystal nanocavities in the visible: L3 vs nanobeam

**Question:** Noda, Fujita & Asano (*Nat. Photon.* 1, 449 (2007)) control spontaneous emission
with silicon photonic-crystal nanocavities (Q ~ 10⁶). How far do the same cavities go in
silicon nitride at 637 nm (NV zero-phonon line), where n = 2.0 instead of 3.5: a 2D-slab L3
cavity, or a 1D nanobeam?

**Model:** Meep / MPB 3D, suspended 200 nm Si₃N₄ membrane in air, bulk n = 2.0, non-dispersive
and lossless (not the toolkit's 2D effective index: slab-cavity Q is set by out-of-plane
radiation). Q from Harminv ring-down, V = ∫ε|E|²/max(ε|E|²) in (λ/n)³, Purcell
F = 3/(4π²) Q/V. GME screening with `legume`.

| Notebook | Content | Key results |
|---|---|---|
| `notebooks/01_bands_and_mirrors.ipynb` | guided TE-like gap of the triangular hole lattice in the membrane; nanobeam band edges and mirror strength; taper designs | slab gap up to 16 % (11–12 % at r/a = 0.30); nanobeam mirror strongest at r = 70 nm (γ 0.076), centre hole 92 nm for the Quan–Lončar taper |
| `notebooks/02_l3_cavity.ipynb` | plain L3, end-hole shifts (FDTD), three-shift GME screen (validated on the Si literature), final design, ±5 nm tolerance | final L3 (a 266.6 nm, shifts 0.35/0.35/0.20 a): λ 642 nm, Q 3.0×10³, V 1.13 (λ/n)³, F ≈ 200; Q 1.8–4.6×10³ under ±5 nm |
| `notebooks/03_nanobeam_cavity.ipynb` | McCutcheon & Lončar 2008 reproduction; own Quan–Lončar taper; ±5 nm tolerance; L3 vs nanobeam | reproduction: peak Q 2.2×10⁵ (paper 2.3×10⁵), V 0.45–0.65 (paper 0.46–0.64), peak +10 nm in s, λ +2.5 %; own N = 15 taper Q ~10⁶–10⁷ (order of magnitude), 1.4×10⁶ retuned to 637 nm; worst case under ±5 nm Q 9×10⁴ |

Code: `src/pic_toolkit/meep_sim/phc_cavity.py`. Long runs (from this folder):
`scripts/run_l3.py`, `scripts/run_nanobeam.py` (mp-mpi env, 12 MPI ranks, launched through
`../scripts/cpu_slot.sh` by the `scripts/queue_*.sh` files), `scripts/slab_gap_map.py`,
`scripts/design_nanobeam.py` (mp env), `scripts/gme_l3_scan.py`, `scripts/gme_validation.py`
(legume env). Literature (tiered): `literature.md`. Parameter history:
`docs/simulation_settings_record.md`, section `meep_sim/phc_cavity.py`.

`data/` holds every cached run (`bands/`, `l3/`, `nanobeam/`) and the figures. The notebooks only
read these caches; each 3D run takes 1–3 h on 12 cores at 64 px/µm.

## Notes

- Own-nanobeam Q values above 10⁶ are extrapolated by Harminv from a ring-down much shorter than
  the decay time: order of magnitude only. Their V values (0.9–1.65) include an admixture of the
  next even mode and are uncertain by a few tens of percent; a narrowband re-run would settle them.
- The L3 tolerance and nanobeam tolerance runs are at 32 and 64 px/µm respectively, each compared
  with a nominal run at the same resolution.
