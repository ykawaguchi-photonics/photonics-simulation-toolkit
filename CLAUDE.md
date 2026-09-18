# Agent guide: `photonics-simulation-toolkit`

This file is loaded automatically by Claude Code at the start of any session
working in this directory. It captures conventions worked out (often the hard
way) while building and revising the notebooks in `notebooks/`. Read this
before editing any notebook or `src/pic_toolkit/meep_sim/*.py` module.

For the concrete, per-bug history behind these conventions, see
**`docs/troubleshooting_log.md`** (process/workflow issues actually hit) and
**`docs/simulation_settings_record.md`** (per-component parameter/tolerance
rationale). Add to those files, don't re-litigate their content in a notebook.

## 1. What this repo is

A component-based toolkit: each `notebooks/0N_*.ipynb` builds and validates one
PIC component via MEEP FDTD, saves `data/design_points/<component>.yaml`
(the single source of truth for that component's parameters plus its
fitted/measured model), and a matching `src/pic_toolkit/models/<component>.py`
reads that YAML in a **meep-free** process to serve as a SAX-compatible
component model. See `README.md` / `circuits/README.md` for the component
list.

## 2. Audience and tone for notebooks

Notebooks are external-reader-facing deliverables (a PIC-engineer portfolio
audience), not personal lab notes. Write for someone who wasn't in the room.

- **Do not narrate debugging/tuning history in the notebook body.** No "a real
  bug was found and fixed here", "an earlier version tried X and abandoned
  it", "measured ~5% during this notebook's own development". State the
  physics/method plainly and point to `docs/simulation_settings_record.md`
  for the full history. That file is organized one `## module.py` section
  per component, each with a parameter table then bolded-lead-in prose
  subsections.
- Section markdown should explain *why* a step matters physically, not
  restate what the code obviously does.
- Keep a consistent section skeleton: numbered `## N. Title` headers, a
  `> **STOP -- ...**` checkpoint blockquote before/after any expensive or
  consequential step (running real FDTD, saving a design point), and
  `**Do not use "Run All."**` stated once near the top.
- Code cells should carry explanatory comments (what this block does and
  why), independent of the markdown above them -- don't rely on the reader
  having just read the prose immediately above.

## 3. Shared plotting style -- `pic_toolkit/style.py`

Every notebook's import cell does:
```python
from pic_toolkit import ..., style
style.apply_style()
```
`viz.plot_permittivity`/`plot_field`/`plot_sparams` already default to this
palette -- no extra work needed for those. For any plot NOT already routed
through `pic_toolkit.viz`, apply these colors explicitly rather than leaving
matplotlib defaults:

| Token | Role |
|---|---|
| `COLOR_STEEL` | baseline / measured data (single series) |
| `COLOR_CYCLE` | any plot with 3+ data series (sweeps, multi-type comparisons) |
| `COLOR_REFERENCE` (gray) | fit/theory overlay lines, PML shading, residuals |
| `COLOR_SEAGREEN` + `CMAP_PERMITTIVITY_OPTIMIZED` ("Greens") | the optimized/final design only -- keep green exclusive to this role |
| `COLOR_ROSE` | sparing one-off emphasis (a single selected point), never a full series |
| `COLOR_ANNOTATION` (crimson) | port/monitor marker lines |
| `CMAP_PERMITTIVITY_BASELINE` ("Blues") | baseline/initial-guess permittivity maps |
| `CMAP_FIELD` ("RdBu") | signed Re(field) plots -- the community-standard convention, keep it |
| `CMAP_DENSITY` ("viridis") | unsigned, continuous 0-1 data (e.g. topology-optimization design density) |

## 4. GDSFactory-first geometry pattern

Preferred shape per component module:
`build_gf_component(params)` (gdsfactory `Component`) ->
`build_geometry_from_gds(params)` (via `gds_import.gds_component_to_prisms`) ->
consumed by `get_permittivity_map`/`simulate_baseline` via a
`use_native_geometry: bool` flag.

**Caution -- axis-aligned Blocks vs. gdsfactory paths are NOT automatically
pixel-identical.** If a module's native geometry is built from many small
discrete Meep `Block`s along a curve, especially *axis-aligned* ones (width
held constant in one coordinate rather than perpendicular to the local
tangent), a gdsfactory path retracing the same nominal curve will disagree at
the sub-pixel/edge level by a few percent -- this is expected, not a
construction bug (see `docs/troubleshooting_log.md`). Before defaulting
`simulate_baseline()` to a new gdsfactory path:
1. Run the native-vs-gds permittivity-map diff (`max`/`mean`/`fraction of
   pixels differing by >0.01`) and explain any discrepancy rather than
   silently accepting or hiding it.
2. **Check whether the module is shared by more than one notebook** (e.g.
   `coupler.py` also backs `06_directional_coupler.ipynb`, not just
   `07_mzi.ipynb`). Don't flip a shared module's simulated-geometry default
   without re-validating every notebook that depends on it.
3. When in doubt, keep `simulate_baseline()` on the already-validated native
   path and use the gdsfactory path only for the layout-display cell
   (`get_permittivity_map(params, use_native_geometry=False)` called
   explicitly just for that one display cell).

## 5. TE/TM polarization -- the recurring gotcha

`eig_parity=mp.NO_PARITY` on an eigenmode source/monitor does **not** mean
"TE" -- MPB picks whichever polarization family has the higher effective
index at `eig_band=1`, and for most of this toolkit's cross-sections that's
TM, not TE. This was found and independently fixed in `waveguide.py`,
`bend.py`, and `bend_topopt.py` (see `docs/simulation_settings_record.md`'s
"Shared: TE/TM mode-parity bug" section for the MPB numbers);
`racetrack.py`/`coupler.py` never relied on `NO_PARITY` in the first place.

**For any new component**: pass `eig_parity=mp.TE` explicitly at every source
and monitor. Capture both `Ez` and `Hz` DFT fields and report whichever
actually dominates -- don't assume `Ez` (once TE is genuinely forced, `Hz`
usually dominates in this toolkit's convention).

## 6. `docs/simulation_settings_record.md` -- read and update this, not the notebook

Single home for *why* a parameter/tolerance is what it is: convergence
studies, bugs found and fixed, approaches tried and abandoned, and the
empirical basis for every non-obvious tolerance constant. One `## module.py`
section per component (parameter table, then bolded-lead-in prose
subsections).

**Tolerance-setting discipline: measure first, then set the tolerance with
real margin above the observed worst case.** Don't guess a tight value and
hope, and don't loosen blindly past what was actually observed. When a
device chains multiple already-validated sub-components in series (e.g. an
MZI's two coupler stages), expect a somewhat larger accumulated residual than
any single sub-component's own budget, and size tolerances accordingly once
measured -- not before.

**A notebook with no cached outputs is a real risk sign.** Its default
parameters/tolerances have not been empirically exercised, no matter how
confident the comments sound. Run it fully before trusting anything it
claims about "measured during development" (see `docs/troubleshooting_log.md`
for a case where this exact situation hid a stale parameter *and* an
untested tolerance at once).

## 7. MEEP simulation-setting cautions

- `mp.stop_when_dft_decayed()`'s default decay criterion can be satisfied by
  a quiet *early* transient before a pulse has actually reached a far
  monitor, especially on long simulation cells -- silently leaving that
  monitor's DFT at exactly 0+0j. Always pass an explicit `minimum_run_time`
  floor scaled to the domain's own optical transit time for any new
  long-cell component.
- A resonant/resonator-like component needs an energy-conservation check
  that is **tight off-resonance and loose (physicality-only) on-resonance**
  -- a single flat tolerance either misses real bugs off-resonance or
  wrongly flags the intended on-resonance loss as a failure. See
  `racetrack.py`'s `energy_conservation_check` for the reusable pattern.
- An independently-measured reference-normalization scheme is not
  automatically better than self-normalizing against a run's own incident
  coefficient -- it can make things *worse* if the port planes sit too close
  to a scattering feature (documented dead end: `bend.py`'s
  `_reference_incident`). Measure before/after; don't add one on faith.
- `curve_fit`/nonlinear refinement over a resonance or interference lineshape
  is fragile when the round-trip phase is highly sensitive to the fitted
  index (a full 2π wrap over a tiny parameter change). Prefer extracting a
  closed-form seed from measured lineshape features (dip depth, FWHM,
  resonance location) first; fit the real-valued power spectrum rather than
  complex `(Re, Im)` values; pass an explicit `x_scale` when parameters span
  very different magnitudes; keep the closed-form seed outright if
  refinement ever makes the residual worse (see
  `docs/troubleshooting_log.md`).
- **Headless verification in this environment is slow and highly variable.**
  Some components' `get_permittivity_map` alone can take 20+ minutes, and a
  full `simulate_baseline()` + sweep can take hours. Use a generous
  `--ExecutePreprocessor.timeout` (hours, not nbconvert's default) when
  running `jupyter nbconvert --to notebook --execute` in the background, and
  expect to wait. Always run from an explicit, verified working directory
  (see `docs/troubleshooting_log.md` -- a background shell's cwd can
  silently revert between tool calls).
- **Persist complex S-parameters (magnitude AND phase), not just power
  fractions, even when the current notebook's own use case only needs
  power.** `coupler.py`'s original `simulate_baseline()` computed genuine
  complex mode-coefficient ratios internally but reduced them to
  `np.abs(...)**2` before they ever reached `CouplerResult`/the saved
  artifact -- fine for that notebook's own standalone splitter-ratio
  characterization, but it meant the coupler's real, FDTD-measured phase was
  unrecoverable later without a full re-simulation, exactly when a
  downstream SAX composition into a genuinely interferometric circuit needed
  it (an MZI's transmission depends on the coupler's own through/cross phase
  relationship, not just its power split). Keep the complex ratio and save
  it alongside any derived power/magnitude fields -- nearly free to keep,
  expensive to reconstruct later.

## 8. Verification checklist for any new/edited notebook

1. Byte-compile / `ast.parse` every edited code cell before running anything
   expensive.
2. Run the full notebook headlessly in the `mp` conda env
   (`conda run -n mp jupyter nbconvert --to notebook --execute ...`).
3. Confirm the resulting `data/design_points/<component>.yaml` has the
   expected keys.
4. Confirm the corresponding `models/<component>.py` SAX function still
   works standalone in a process where `meep` was never imported
   (`assert "meep" not in sys.modules`).
5. If a notebook shares a `meep_sim` module with another notebook, confirm
   the other notebook's own validated behavior is unaffected (see §4).
