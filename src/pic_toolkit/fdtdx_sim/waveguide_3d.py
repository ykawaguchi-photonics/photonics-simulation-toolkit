"""3D SOI strip waveguide in fdtdx: simulation builder and cross-section mode helpers.

**Modeling assumption (deviates from the toolkit's 2D default).** A real 3D cell:
500 x 220 nm Si core (bulk n = 3.48) fully clad in SiO2 (n = 1.44), PML on all six
faces, propagation along x. Materials are non-dispersive (one constant index across
the analyzed band), as everywhere else in the toolkit.

**TE convention.** fdtdx's `filter_pol="te"` keeps modes whose E field lies mainly
along the first transverse axis, which is y for propagation along x. That is the
Ey-dominant TE-like mode (Meep's `EVEN_Z`), used at the source and every detector.

**Grid.** fdtdx staircases: each cell is either core or cladding (no subpixel
averaging), and object sizes/positions are rounded to whole cells. Check what the
grid actually holds with `core_on_grid` before trusting a comparison with the
nominal 500 x 220 nm structure.

fdtdx works in SI units (metres, seconds). Resolutions here are in px/um, converted
to a grid spacing of 1/res um. Parameter rationale: `docs/simulation_settings_record.md`,
section `fdtdx_sim/waveguide_3d.py`.
"""

from __future__ import annotations

import math
import time

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx
from fdtdx.core.physics.modes import compute_mode

UM = 1e-6
CORE_INDEX = 3.48
CLAD_INDEX = 1.44
WG_WIDTH_UM = 0.5
WG_HEIGHT_UM = 0.22
WL0_UM = 1.35
DPML_UM = 1.0
N_G = 4.2                 # group index of this strip near 1.35 um (sets the run length)
SRC_BW = 0.2              # pulse bandwidth as a fraction of the center frequency

F0 = fdtdx.constants.c / (WL0_UM * UM)


def pulse_sigma_t() -> float:
    """Temporal std-dev of fdtdx's GaussianPulseProfile (peak sits at 6 sigma)."""
    return 1.0 / (2 * math.pi * SRC_BW * F0)


def full_run_time(lx: float, dpml: float = DPML_UM) -> float:
    """Pulse duration (12 sigma) + two group-delay transits of the whole cell, in s.

    fdtdx runs a fixed number of steps (there is no decay-based stop condition), so the
    run must be long enough for the pulse to cross the cell and leave it.
    """
    cell_lx = (lx + 2 * dpml) * UM
    return 12 * pulse_sigma_t() + 2 * N_G * cell_lx / fdtdx.constants.c


def detector_x_um(lx: float, n_det: int) -> list[float]:
    """Mode-overlap detector x positions (relative to cell center), evenly spaced
    from 1 um after the source plane to 0.5 um before the far end of the interior."""
    x_src = -0.5 * lx + 0.5
    x0, x1 = x_src + 0.5, 0.5 * lx - 0.5
    if n_det == 1:
        return [x1]
    return [x0 + i * (x1 - x0) / (n_det - 1) for i in range(n_det)]


def make_sim(res: float, lx: float, sy: float, sz: float, *, sim_time: float,
             n_det: int = 2, field_slices: bool = False, dpml: float = DPML_UM, key=None):
    """Build and place the 3D waveguide. Returns (objects, arrays, params, config, det_names).

    lx/sy/sz are interior (non-PML) extents in um; PML of `dpml` um is added on every face.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    spacing = UM / res
    n_pml = int(round(dpml * res))
    config = fdtdx.SimulationConfig(time=sim_time, resolution=spacing, backend="gpu",  # falls back to cpu
                                    dtype=jnp.float32, courant_factor=0.99)

    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(
        partial_real_shape=((lx + 2 * dpml) * UM, (sy + 2 * dpml) * UM, (sz + 2 * dpml) * UM),
        material=fdtdx.Material(permittivity=CLAD_INDEX**2),
    )
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=n_pml, boundary_type="pml")
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    core = fdtdx.UniformMaterialObject(
        name="core",
        partial_real_shape=(None, WG_WIDTH_UM * UM, WG_HEIGHT_UM * UM),
        material=fdtdx.Material(permittivity=CORE_INDEX**2),
    )
    objects.append(core)
    constraints.append(core.place_at_center(volume, axes=(1, 2)))

    wave = fdtdx.WaveCharacter(wavelength=WL0_UM * UM)
    source = fdtdx.ModePlaneSource(
        name="source",
        partial_grid_shape=(1, None, None),
        partial_real_shape=(None, sy * UM, sz * UM),
        direction="+",
        mode_index=0,
        filter_pol="te",
        wave_character=wave,
        temporal_profile=fdtdx.GaussianPulseProfile(
            center_wave=wave,
            spectral_width=fdtdx.WaveCharacter(frequency=SRC_BW * F0),
        ),
    )
    objects.append(source)
    x_src = -0.5 * lx + 0.5
    constraints += [source.place_at_center(volume, axes=(1, 2)),
                    source.place_relative_to(volume, axes=0, own_positions=0, other_positions=0,
                                             margins=x_src * UM)]

    det_names = []
    for i, xd in enumerate(detector_x_um(lx, n_det)):
        det = fdtdx.ModeOverlapDetector(
            name=f"mode_{i}",
            partial_grid_shape=(1, None, None),
            partial_real_shape=(None, sy * UM, sz * UM),
            direction="+",
            mode_index=0,
            filter_pol="te",
            wave_characters=(wave,),
        )
        objects.append(det)
        det_names.append(det.name)
        constraints += [det.place_at_center(volume, axes=(1, 2)),
                        det.place_relative_to(volume, axes=0, own_positions=0, other_positions=0,
                                              margins=xd * UM)]

    if field_slices:
        # Ey phasor on the x-y mid-plane (z = 0) and Ey/Ez on a y-z cross-section.
        xy = fdtdx.PhasorDetector(name="phasor_xy", wave_characters=(wave,), components=("Ey",),
                                  partial_grid_shape=(None, None, 1),
                                  partial_real_shape=(lx * UM, sy * UM, None), plot=False)
        yz = fdtdx.PhasorDetector(name="phasor_yz", wave_characters=(wave,), components=("Ey", "Ez"),
                                  partial_grid_shape=(1, None, None),
                                  partial_real_shape=(None, sy * UM, sz * UM), plot=False)
        objects += [xy, yz]
        constraints += [xy.place_at_center(volume, axes=(0, 1, 2)),
                        yz.place_at_center(volume, axes=(1, 2)),
                        yz.place_relative_to(volume, axes=0, own_positions=0, other_positions=0,
                                             margins=0.5 * UM)]

    key, sub = jax.random.split(key)
    objects_c, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=sub)
    return objects_c, arrays, params, config, det_names


def build_and_run(res, lx, sy, sz, *, sim_time, n_det=2, field_slices=False, dpml=DPML_UM, seed=0):
    """Place objects, solve the source/detector modes (apply_params), compile and run.

    Returns (arrays_after_run, objects, config, det_names, timings). Compile time is kept
    separate from run time, because XLA compiles once per grid shape.
    """
    key = jax.random.PRNGKey(seed)
    t0 = time.perf_counter()
    objects, arrays, params, config, det_names = make_sim(
        res, lx, sy, sz, sim_time=sim_time, n_det=n_det, field_slices=field_slices, dpml=dpml, key=key)
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)  # mode solves happen here
    jax.block_until_ready(arrays)
    t_setup = time.perf_counter() - t0

    def sim_fn(arrays, key):
        _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key)
        return arrays

    t1 = time.perf_counter()
    # donate_argnums=0 lets XLA reuse the input field buffers for the output, so the input
    # and output ArrayContainers are not both alive at once.
    compiled = jax.jit(sim_fn, donate_argnums=0).lower(arrays, key).compile()
    t_compile = time.perf_counter() - t1

    t2 = time.perf_counter()
    arrays = compiled(arrays, key)
    jax.block_until_ready(arrays)
    t_run = time.perf_counter() - t2
    timings = {"t_setup_s": t_setup, "t_compile_s": t_compile, "t_run_s": t_run,
               "time_steps": config.time_steps_total, "dt_s": config.time_step_duration}
    return arrays, objects, config, det_names, timings


def grid_voxels(res: float, lx: float, sy: float, sz: float, dpml: float = DPML_UM) -> int:
    """Total cell count of the grid, PML included."""
    d = 2 * dpml
    return int(round((lx + d) * res)) * int(round((sy + d) * res)) * int(round((sz + d) * res))


def core_on_grid(arrays, res: float) -> dict:
    """What the grid actually holds: the core's size in cells along the center lines, and
    how many cladding cells sit on each side (equal counts = core exactly centered)."""
    eps = 1.0 / np.asarray(arrays.inv_permittivities[0])
    nx, ny, nz = eps.shape
    cy = np.where(eps[nx // 2, :, nz // 2] > 5)[0]
    cz = np.where(eps[nx // 2, ny // 2, :] > 5)[0]
    return {"w_nm": cy.size * 1000 / res, "h_nm": cz.size * 1000 / res, "cells": (cy.size, cz.size),
            "y_sides": (int(cy.min()), int(ny - 1 - cy.max())), "z_sides": (int(cz.min()), int(nz - 1 - cz.max()))}


def is_centered(g: dict) -> bool:
    """True when `core_on_grid` reports equal cladding counts on both sides in y and in z."""
    return g["y_sides"][0] == g["y_sides"][1] and g["z_sides"][0] == g["z_sides"][1]


def upper_share(ey, ez) -> float:
    """Fraction of |Ey|^2 + |Ez|^2 above the plane's z midplane (0.5 for a z-symmetric mode)."""
    e2 = np.abs(ey) ** 2 + np.abs(ez) ** 2
    h = e2.shape[1] // 2
    up, lo = e2[:, h + e2.shape[1] % 2:].sum(), e2[:, :h].sum()
    return float(up / (up + lo))


def solve_plane_modes(inv_eps_plane, inv_mu, spacing_m: float) -> dict:
    """TE-like and TM-like fundamental modes of one cross-section (shape (n_comp, 1, ny, nz)),
    solved with the same tidy3d mode solver fdtdx uses for its sources and detectors."""
    out = {}
    for label, pol in (("TE-like", "te"), ("TM-like", "tm")):
        E, _, neff = compute_mode(frequency=F0, inv_permittivities=inv_eps_plane, inv_permeabilities=inv_mu,
                                  resolution=spacing_m, direction="+", mode_index=0, filter_pol=pol)
        E = np.asarray(E)[:, 0]      # (3, ny, nz) in physical (Ex, Ey, Ez) order
        out[label] = {"n_eff": float(np.real(neff)), "Ey": np.abs(E[1]), "Ez": np.abs(E[2]),
                      "upper_share": upper_share(E[1], E[2])}
    return out


def fine_grid_modes(res: int, sy: float, sz: float, w_um: float = WG_WIDTH_UM, h_um: float = WG_HEIGHT_UM) -> dict:
    """Modes of a purpose-built sy x sz um cross-section whose core edges fall exactly on
    cell boundaries (w_um * res and h_um * res should be integers), solved by tidy3d."""
    ny, nz = int(round(sy * res)), int(round(sz * res))
    y = (np.arange(ny) + 0.5) / res - sy / 2
    z = (np.arange(nz) + 0.5) / res - sz / 2
    core = (np.abs(y)[:, None] < w_um / 2) & (np.abs(z)[None, :] < h_um / 2)
    eps2d = np.where(core, CORE_INDEX**2, CLAD_INDEX**2)
    inv_eps = jnp.asarray(1.0 / eps2d, dtype=jnp.float32)[None, None]   # (1, nx=1, ny, nz)
    return solve_plane_modes(inv_eps, 1.0, UM / res)
