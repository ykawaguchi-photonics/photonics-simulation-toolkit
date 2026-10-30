# SiN track (Meep 2D)

The silicon-nitride counterparts of the toolkit's components, simulated in Meep 2D with
the same code as the Si track. The Si track lives in `si/notebooks/` and `si/data/`. This
track changes only the parameter set (`pic_toolkit.params.SIN_GLOBAL_PARAMS`).

The bend is compared not with the Si track but with a **Si reference**
(`params.SI_REF_PARAMS`). The Si track's simplified core index 2.7 gives a lateral n_eff of
2.418, well below the 2.671 of a real 500 × 220 nm strip. Both SiN and the Si reference are
calibrated the same way: the 2D core index is chosen so that the 2D mode has the MPB n_eff of
the real cross-section. Both are then benchmarked against published 3D results, at the
publications' own geometry.

The platform is a 400 nm SiN film (n = 2.0) clad in SiO₂ (1.44), a 0.9 µm wide strip, and
the O band (1.30–1.40 µm, centre 1.35 µm), the same band as the Si track. Sources and the
hypotheses they set are in [`literature.md`](literature.md).

## Notebooks

| Notebook | Component | Main result | Si counterpart |
|---|---|---|---|
| `notebooks/01_sin_waveguide_baseline.ipynb` | 0.9 µm SiN strip: n_eff-matched 2D core (MPB cross-section), TE1 cutoff, FDTD baseline, cutback, MPB, mode comparison with the Si track and the Si reference | n_eff 1.6855 (cutback) / 1.6751 (2D MPB) / 1.6753 (3D cross-section target), n_g 1.803; lateral Δn 0.235 vs 1.231 (Si reference) | `si/notebooks/01_waveguide_baseline.ipynb` |
| `notebooks/02_sin_bent_waveguide.ipynb` | 90° bend, circular and Euler, R = 5–30 µm sweep, SAX fit; benchmarks against Bahadori 2019 (Si) and Pita Ruiz 2025 (SiN); comparison with the Si reference | design point Euler R = 20 µm, 0.010 dB per 90°; equal loss needs 7.7–8.1× the Si radius (Marcuse predicts 7.1×) | `si/notebooks/02_bent_waveguide.ipynb` |

Both notebooks run end to end with "Run All" from `sin/notebooks/`. The bend sweep is
read from cached artifacts (`reuse_existing=True`; a changed core or cladding index
invalidates them). Without the cache, the 45 SiN, Si-reference and benchmark runs take about
4 h as 8 parallel processes; the R = 30 µm Euler point alone takes 3.5 h. Each notebook still runs its
own representative baseline live, so notebook 02 takes about 40 min.

## How this track differs from the Si track

| | Si track (`si/notebooks/`, `si/data/`) | SiN track (`sin/`) |
|---|---|---|
| film | Si, simplified 2D core index 2.7 | 400 nm SiN, 2D core index **1.7650**, n_eff-matched to the MPB cross-section (`slab_modes.core_index_for_neff`) |
| strip width | 0.5 µm | 0.9 µm (TE1 cutoff 0.64–0.69 µm in this 2D model, so a weak TE1 is also guided) |
| lateral TE0 n_eff | 2.418 (real 500 × 220 nm strip: 2.671) | 1.675 |
| mode decay length outside the core | 0.11 µm (Si reference 0.095 µm) | 0.25 µm |
| waveguide cell height | `cell_y_um` 4.0 | 6.0 (at 4.0 the PML clipped the tail: \|S21\|² 0.9997) |
| bend margin to the PML | 1.5 µm | 2.0 µm |
| bend radii swept | 1.0–3.0 µm | 5–30 µm |
| design points | `si/data/design_points/` | `sin/data/design_points/` |
| SAX models | `models.waveguide`, `models.bend` | `models.sin_waveguide`, `models.sin_bend` (same fitted-model form) |

The resolution is the same in both tracks: 20 px/µm for the waveguide, 40 px/µm for the
bend. SiN data never goes into `si/data/design_points/`, so a Si SAX model can never pick
up a SiN number by accident (the same rule as the 3D track).

## Bend comparison (notebook 02 §9)

**Benchmarks first.** The calibrated 2D model was run at two published geometries.

| | Published | This model | Ratio |
|---|---|---|---|
| Si 400 × 220 nm, circular, 1.55 µm (Bahadori et al., JLT 2019, 3D FDTD) | 0.29 / 0.061 / 0.0084 dB at R = 1 / 2 / 5 µm (their fit); 0.0125 dB at 5 µm (direct) | 1.38 / 0.19 / 0.023 dB | 2.6–4.7× (fit), 1.8× (direct) |
| SiN 850 × 400 nm, R = 6 µm, 1.55 µm (Pita Ruiz et al., JLT 2025) | circular 2.88 / Euler 3.46 dB (3D sim.), 2.86 / 3.44 dB (measured) | 5.2 / 6.8 dB | 1.8× / 2.0× |

The 2D model therefore *overestimates* bend loss, by a similar factor on both platforms.
Since loss falls exponentially with R, this shifts the radius for a given loss by about
1.5× (Si) and 1.25× (SiN); the radii below are conservative.

**Si reference vs SiN at 1.35 µm** (radius beyond which the loss stays below the target):

| loss per 90° | Si reference | SiN | ratio |
|---|---|---|---|
| Euler ≤ 0.1 dB | 1.71 µm | 13.8 µm | 8.1× |
| Euler ≤ 0.01 dB | 2.51 µm | 20.1 µm | 8.0× |
| circular ≤ 0.1 dB | 2.62 µm | 20.1 µm | 7.7× |

**Why about 8×.** The ratio follows from the weaker lateral confinement: decay constant
γ = 3.99 vs 10.47 µm⁻¹. Marcuse's slab bend loss ∝ exp(−2γ³R/(3β²)) predicts 7.1×. Plotted
against the WKB tunnelling exponent Φ to the radiation caustic (exact cylindrical barrier),
the Euler bends of both platforms fall in the same range: 0.01 dB at Φ ≈ 13 (SiN) and 19
(Si), against 20 µm and 2.5 µm on the radius axis. Φ removes most of the 8×, but not all of
it; the prefactor of e^−Φ differs between the platforms.

**Other findings.**
- The circular SiN bend flattens above about 15 µm (0.039 dB at 30 µm). That is the
  straight-to-arc junction loss, which the Euler bend removes.
- The circular Si-reference loss ripples with a period of about 1 µm in R. The 500 nm strip
  guides TE1 at 1.35 µm, and the junctions excite TE0/TE1 beating (2L_B/π = 1.02 µm).

**Caveats.**
- This is a 2D effective-index model. A real 3D bend also radiates into the substrate and
  the upper cladding, and the benchmarks show a systematic error of 2–3× in loss.
- Compare absolute numbers with the literature at the factor level only. The SiN/Si ratio
  is the more robust result.
- The Si track (core 2.7) is kept only as a grey reference curve. Its bends lose more than
  the 3D literature.

## Layout

```
sin/
  literature.md            sources (tiered), platform choice, hypotheses
  notebooks/               01_sin_waveguide_baseline, 02_sin_bent_waveguide
  data/
    design_points/         sin_waveguide.yaml, sin_bend.yaml
    sparams/               sin_waveguide/{baseline,sweep}, sin_bend/{baseline,sweep},
                           si_ref_bend/{sweep,convergence}, benchmark/{si_400x220,sin_850x400}_c_band
    field_cache/           cached Euler bend field snapshots for notebook 02 §9.4
```

Simulation code is shared with the Si track: `src/pic_toolkit/meep_sim/waveguide.py` and
`bend.py`, driven by the SiN params dict. Parameter and tolerance rationale is in
`docs/simulation_settings_record.md`, section "SiN track".

## Running

```bash
conda activate mp
cd sin/notebooks
jupyter lab
```
