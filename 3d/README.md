# 3D track (fdtdx)

True-3D FDTD counterparts of selected toolkit components, run with
[fdtdx](https://github.com/ymahlau/fdtdx), a JAX-based 3D FDTD solver. The 2D
track in `si/notebooks/` simulates an effective-index plane; this track puts the
real 3D cross-section on the grid.

## Notebooks

| Notebook | Component | Kernel | 2D counterpart |
|---|---|---|---|
| `notebooks/01b_waveguide_3d_fdtdx.ipynb` | 500 × 220 nm Si strip: TE-like mode, converged n_eff, 3D FDTD accuracy on a matched grid | `fdtdx` | `si/notebooks/01_waveguide_baseline.ipynb` |
| `notebooks/09b_grating_coupler_3d.ipynb` | Fan-shaped focusing grating coupler to a 2.5 µm lensed fiber: η = −1.49 dB at 1.35 µm, GDS export, design point | `mp` | `si/notebooks/09_grating_coupler.ipynb` |

09b loads the final device's saved 3D results from
`data/results/grating_coupler_fan_3d/` and needs `gdsfactory`, so it runs in the
2D track's `mp` env. It does not re-run the 3D simulation.

## How this track differs from the 2D track

| | 2D Si track (`si/notebooks/`, `si/data/`) | 3D track (`3d/`) |
|---|---|---|
| geometry | 2D effective-index plane, `core_index=2.7` | real 3D cross-section, bulk Si 3.45–3.48 / SiO₂ 1.44 |
| solver | Meep (`src/pic_toolkit/meep_sim/`) | fdtdx (`src/pic_toolkit/fdtdx_sim/`) |
| env | `mp` (Python 3.10) | `fdtdx` (Python 3.12) |
| geometry on the grid | subpixel averaging | staircased: dimensions pinned to whole cells, centering asserted |
| design points | `si/data/design_points/`, read by the SAX `models/` | `3d/data/design_points/`, not (yet) read by any SAX model |

Materials are non-dispersive in both tracks. The 3D design points are kept
apart on purpose: their numbers belong to a different model, and the 2D SAX
models must never pick them up by accident.

## Layout

```
3d/
  notebooks/              01b, 09b
  data/
    design_points/        waveguide_3d.yaml, grating_coupler_fan_3d.yaml
    results/              saved 3D FDTD results that notebooks load (no re-run)
    gds/                  fan_grating_coupler_3d.gds
  environment.yml         the `fdtdx` env
```

Simulation code lives in `src/pic_toolkit/fdtdx_sim/`, so it shares the
toolkit's `style.py` and `design_points.py`. Parameter and tolerance rationale
is in `docs/simulation_settings_record.md` (sections `fdtdx_sim/waveguide_3d.py`
and `grating_coupler_fan_3d`).

## Setup

```bash
conda env create -f 3d/environment.yml     # creates the `fdtdx` env
conda activate fdtdx
jupyter lab 3d/notebooks/
```

Run each notebook from `3d/notebooks/` (paths are relative to it), cell by cell,
not with "Run All". A 3D run takes several minutes or more on a CPU.
