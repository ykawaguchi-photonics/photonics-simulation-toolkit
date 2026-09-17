# photonics-simulation-toolkit

A "LEGO block" toolkit for building, validating, and composing silicon
photonic integrated circuit (PIC) components with [Meep](https://meep.readthedocs.io/)
FDTD, [gdsfactory](https://gdsfactory.github.io/gdsfactory/), and
[SAX](https://flaport.github.io/sax/).

## What this is

Each component is built once, in isolation, and its validated result becomes
a reusable building block for the next layer:

1. **`notebooks/0N_*.ipynb`** — one notebook per PIC component. Builds the
   geometry (gdsfactory-first), runs a Meep FDTD simulation, characterizes
   it (S-parameters, resonance, extinction, Vπ, etc.), and saves the result
   as `data/design_points/<component>.yaml` — the single source of truth
   for that component's parameters and fitted/measured model.
2. **`src/pic_toolkit/models/<component>.py`** — a **Meep-free** twin of each
   component that reads its cached design point and serves it as a
   SAX-compatible model, so downstream circuit work never needs to re-run
   FDTD.
3. **`circuits/`** — SAX circuit-level compositions built purely from those
   cached component models, comparing an ideal analytic model against the
   genuinely FDTD-measured one.

Each notebook is written as a reader-facing deliverable — not personal lab
notes — with a consistent structure and explicit checkpoints before any
expensive or consequential step.

## Components

| Notebook | Component |
|---|---|
| `01_waveguide_baseline.ipynb` | Straight waveguide baseline (TE mode, cross-section characterization) |
| `02_bent_waveguide.ipynb` | Bent waveguide (radius sweep, circular vs. Euler bends) |
| `03_mzi_arm.ipynb` | MZI delay arm (jog geometry) |
| `03b_mzi_arm_dense_sweep.ipynb` | Densified ΔL sweep for the delay arm |
| `04_bend_topology_optimization.ipynb` | Adjoint topology optimization of a waveguide bend |
| `05_racetrack_resonator.ipynb` | Racetrack resonator |
| `06_directional_coupler.ipynb` | Directional coupler (baseline) |
| `06b_directional_coupler_gap_sweep.ipynb` | 2D gap × length sweep for the coupler |
| `07_mzi.ipynb` | Passive Mach-Zehnder interferometer |
| `08_mzm.ipynb` | Mach-Zehnder modulator (push-pull Vπ) |
| `09_grating_coupler.ipynb` | Grating coupler (x-z cross-section) |
| `10_add_drop_ring_resonator.ipynb` | Add-drop ring resonator |

## Circuit-level compositions

Under `circuits/` — never imports Meep, only wires together already-cached
component models:

- **`wdm_sax.ipynb`** — single-stage cascaded-coupler lattice filter (N=2 vs.
  N=4 couplers), built both as an ideal analytic model and from genuinely
  FDTD-measured component S-parameters, with GDS layout export.
- **`wdm_mux4_sax.ipynb`** — 4-channel WDM demultiplexer using a cascaded
  binary-tree ("Mux4") topology built from the `wdm_sax` lattice as its
  per-node building block, with GDS layout export.

See [circuits/README.md](circuits/README.md) for the full derivation.

## Repository structure

```
notebooks/            Component-building notebooks (01-10, + 03b/06b)
circuits/              SAX circuit-level compositions + GDS exports
src/pic_toolkit/
  meep_sim/            Meep FDTD simulation modules (one per component)
  models/               Meep-free SAX-compatible twins, read cached design points
  circuits/             SAX circuit building blocks (lattice, mux tree)
  params.py, sweep.py, sparams.py, checks.py, design_points.py, style.py, viz.py
data/
  design_points/        Per-component YAML — the source of truth
  sparams/               Cached S-parameter artifacts (.npz + .json) per component
docs/
  simulation_settings_record.md   Parameter/tolerance rationale per component
  troubleshooting_log.md          Process/workflow issues and fixes
scripts/               One-off sweep runner scripts
```

## Setup

```bash
conda env create -f environment.yml
conda activate mp
```

This installs Python 3.10, `pymeep`, `sax`, `gdsfactory`, and an editable
install of this package (`pic_toolkit`).

## Usage

```bash
conda activate mp
jupyter lab notebooks/
```

Notebooks are meant to be run cell-by-cell, not with "Run All" — each has
explicit **STOP** checkpoints before any expensive FDTD run or before saving
a new design point. See [CLAUDE.md](CLAUDE.md) for the full set of
conventions (notebook structure, shared plotting style, geometry-construction
pattern, and known simulation gotchas) before editing any notebook or
`src/pic_toolkit/meep_sim/*.py` module.

## Tech stack

- [Meep](https://meep.readthedocs.io/) (`pymeep`) — FDTD electromagnetic simulation, including adjoint topology optimization (`autograd` + `nlopt`)
- [gdsfactory](https://gdsfactory.github.io/gdsfactory/) — parametric GDS layout generation
- [SAX](https://flaport.github.io/sax/) — circuit-level S-parameter composition

## More documentation

- [CLAUDE.md](CLAUDE.md) — repo conventions for notebooks and simulation modules
- [docs/simulation_settings_record.md](docs/simulation_settings_record.md) — why each non-obvious parameter/tolerance is what it is
- [docs/troubleshooting_log.md](docs/troubleshooting_log.md) — process/workflow issues hit during development
- [circuits/README.md](circuits/README.md) — circuit-level composition details
