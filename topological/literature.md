# Literature: dielectric topological photonic crystals (spin-Hall and valley-Hall)

Sources are tiered by weight (CLAUDE.md §10): numbers and design rules rest on the
high-impact peer-reviewed sources; preprints and vendor examples are used for concepts or
cross-checks only.

| Source | Tier | Used for |
|---|---|---|
| L.-H. Wu, X. Hu, "Scheme for achieving a topological photonic crystal by using dielectric material", *Phys. Rev. Lett.* 114, 223901 (2015) | high-impact peer-reviewed | spin-Hall pillars: εd = 11.7, d = 0.24 a0 (Fig. 5), a0/R = 3.125 / 2.9; reference gap 5.47 THz at 138.77 THz for a0 = 1 µm |
| T. Ozawa et al., "Topological photonics", *Rev. Mod. Phys.* 91, 015006 (2019) | review | context: crystalline-symmetry and valley topological photonics |
| S. Barik et al., "A topological quantum optics interface", *Science* 359, 666 (2018) | high-impact peer-reviewed | spin-Hall design with triangular holes in a GaAs membrane |
| N. Parappurath et al., "Direct observation of topological edge states in silicon photonic crystals", *Sci. Adv.* 6, eaaw4137 (2020) | high-impact peer-reviewed | spin-Hall triangular holes in 220 nm Si: a0 = 800 nm, side 250 nm (0.3125 a0), a0/R = 3.29 / 2.75; slab-calculated TE gaps 4.7 % / 7.8 % |
| Flexcompute Tidy3D example "TopoQuantumPhC" (reproduction of Barik 2018) | other (vendor example) | cross-check of the Barik geometry: side 0.315 a0, a0/R = 3.19 / 2.86, one vertex toward the cluster centre |
| X.-T. He et al., "A silicon-on-insulator slab for topological valley transport", *Nat. Commun.* 10, 872 (2019) | high-impact peer-reviewed | valley-Hall SOI crystal, bearded interface stacked with the larger holes, chiral source in a bearded hole (Dx = a/2, Dy = a/(2√3) from the hexagon centre), 1.7 µm strip butt-coupled to the interface |
| M. Shalaev et al., "Robust topologically protected transport in photonic crystals at telecommunication wavelengths", *Nat. Nanotechnol.* 14, 31 (2019); arXiv:1712.07284 | high-impact peer-reviewed | valley-Hall inverted triangular holes 0.4 a / 0.6 a, a = 423 nm, 2D model with n_eff = 2.965, measured gap ≈ 1420–1510 nm |
| Z. Lan, J. W. You, Q. Ren, W. E. I. Sha, N. C. Panoiu, "Second-harmonic generation via double topological valley-Hall kink modes in all-dielectric photonic crystals", *Phys. Rev. A* 103, L041502 (2021) | other peer-reviewed | valley-Hall pillars: honeycomb of rods r = 0.18 a / 0.22 a, TM, Dirac point between bands 1 and 2 |
| X.-D. Chen, F.-L. Zhao, M. Chen, J.-W. Dong, "Valley-contrasting physics in all-dielectric photonic crystals", *Phys. Rev. B* 96, 020202(R) (2017) | other peer-reviewed | context: valley-contrasting orbital angular momentum and chiral excitation in all-dielectric rod crystals |
| L. Chen et al., "Efficient light coupling between conventional silicon photonic waveguides and quantum valley-Hall topological interfaces", *Opt. Express* 30, 2517 (2022) | other peer-reviewed | line-defect coupler: ≈ 94 % (2D) for a bearded interface |
| M. J. Mehrabad et al., "Chiral topological photonics with an embedded quantum emitter", *Optica* 7, 1690 (2020) | high-impact peer-reviewed | line-defect coupler to a bearded interface, ≈ 31 % in 3D (as cited by Yoshimi et al.) |
| H. Yoshimi et al., "Efficient light couplers to topological slow light waveguides in valley photonic crystals", arXiv:2311.13965 (2023) | preprint (established group) | triangular holes, end face through the hole centres; butt coupling < 10 %, line defect L = 6a > 80 % (3D); bearded interface described as glide-symmetric |
| P. Lodahl et al., "Chiral quantum optics", *Nature* 541, 473 (2017) | review | a circular dipole couples directionally to a guided mode where the mode's local field is circularly polarized; directionality = local degree of circular polarization |
| I. Söllner et al., "Deterministic photon–emitter coupling in chiral photonic circuits", *Nat. Nanotechnol.* 10, 775 (2015) | high-impact peer-reviewed | chiral points of a glide-plane PhC waveguide: directionality and coupling strength vary across the unit cell, so the emitter position is chosen from the mode field |
| N. V. Hauff et al., "Chiral quantum optics in broken-symmetry and topological photonic crystal waveguides", *Phys. Rev. Research* 4, 023082 (2022) | other peer-reviewed | maps of directionality and coupling over the unit cell of valley-Hall and spin-Hall waveguides; high directionality and strong coupling are generally not at the same point |
| A. M. Hammond et al., "High-performance hybrid time/frequency-domain topology optimization for large-scale photonics inverse design", *Opt. Express* 30, 4467 (2022) | other peer-reviewed | the Meep adjoint (density-based) topology optimization used here |
| Meep documentation, `meep.adjoint.smoothed_projection` (subpixel-smoothed projection, Hammond et al.) | software documentation | projection that keeps gradients valid up to β = ∞, used to binarize the designs |

## Hypotheses set for the study

1. The spin-Hall crystals open a gap at Γ with inverted p/d order between the shrunken and
   expanded clusters; the honeycomb limit has a fourfold degeneracy at Γ (Wu & Hu).
2. Triangular holes give complete TE gaps on both sides of the spin-Hall transition
   (Barik, Parappurath).
3. The valley-Hall crystals have a Dirac point at K in the symmetric limit and a gap at K
   that grows with the size difference (He, Shalaev, Lan).
4. A circularly polarized source at the pseudospin / valley vortex point excites the edge
   state in one direction (Wu & Hu Fig. 5, He Fig. 3). The best position is where the edge
   mode's in-plane field is circularly polarized *and* strong (Lodahl, Söllner, Hauff), which
   need not be the symmetry point.
5. A strip butt-coupled to a valley interface couples weakly (Yoshimi) and a line-defect
   section raises it (Chen, Yoshimi).
