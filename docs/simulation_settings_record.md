# Simulation Settings Record

This document is the single home for *why* each component's FDTD/adjoint
settings are what they are: convergence studies, bugs found and fixed, and
approaches that were tried and abandoned. The notebooks and source docstrings
intentionally stay short and point here instead of repeating this history
inline — read this file when you need the full story behind a parameter.

## Shared: width unification (500nm)

`waveguide.py`/`bend.py`/`grating_coupler.py` build their `DEFAULT_PARAMS` from
the shared `GLOBAL_PARAMS` (`src/pic_toolkit/params.py`), which used to set
`wg_width_um=0.45`. `racetrack.py`/`coupler.py`/`mzi.py`/`bend_topopt.py` each
have their own fully standalone `DEFAULT_PARAMS` (no `GLOBAL_PARAMS` import at
all) and independently hardcoded `wg_width_um=0.5` — a silent fork `params.py`'s
own module docstring already names as a bug class ("values that drifted apart
across components without a physical reason... are bugs, not intentional
per-component choices"), just not caught until a direct comparison across
notebooks surfaced it.

**Fix: converge on 500nm.** Chosen over 450nm purely on cost: `racetrack.py`/
`coupler.py`/`mzi.py`/`bend_topopt.py` already had expensive, already-validated
measurements pinned to 500nm (a coupled-mode-theory resonance fit, a directional
coupler's 50:50 coupling length, an MZI's chained delay-arm sweep, and a full
adjoint topology optimization, respectively) — re-measuring those would mean
redoing the toolkit's most expensive notebooks. Converging the other direction
only required re-running `01_waveguide_baseline.ipynb` and
`02_bent_waveguide.ipynb`, both lightweight single-component baselines.
500nm×220nm is also the more commonly cited canonical SOI strip-waveguide
cross-section.

**What changed:** `GLOBAL_PARAMS["wg_width_um"]` → `0.5`; `waveguide.py`/
`bend.py`/`grating_coupler.py` inherit it automatically (no per-module edits).
`spiral.py`/`spiral_gds.py` independently hardcoded `0.45` too and were fixed to
`0.5` for the same reason — free changes, since neither has a notebook or a
saved `data/design_points/*.yaml` yet. `notebooks/01_waveguide_baseline.ipynb`
and `02_bent_waveguide.ipynb` were re-run end-to-end at the new width, and
`data/design_points/waveguide.yaml`/`bend.yaml` regenerated with fresh fits (see
those modules' own sections below for the updated numbers).
`racetrack.py`/`coupler.py`/`mzi.py`/`bend_topopt.py` and their design points
were already at 500nm and were not touched.

## Shared: TE/TM mode-parity bug

Every eigenmode source/monitor in this toolkit used to pass
`eig_parity=mp.NO_PARITY`, which lets Meep's MPB solver pick whichever
polarization family has the higher effective index at `eig_band=1` —
**not** necessarily TE, despite every parameter/docstring in the toolkit
claiming "TE".

Direct MPB queries (`Simulation.get_eigenmode`, no timestepping) on this
toolkit's cross-section (`wg_width_um=0.5`, `core_index=2.7`,
`clad_index=1.44` — every component now shares this exact cross-section, see
the width-unification note above) found:

| Family | Fields | n_eff |
|---|---|---|
| TE (`mp.TE` / `mp.EVEN_Z`) | Ex, Ey, Hz | 2.4094 |
| TM (`mp.TM` / `mp.ODD_Z`) | Ez, Hx, Hy | **2.5085** (higher — silently selected) |

(Re-measured after the width-unification pass below moved `waveguide.py`/`bend.py`
from `wg_width_um=0.45` to `0.5` — the original 0.45µm measurement found the same
TM-over-TE ordering, just with different absolute n_eff values: TE=2.3552,
TM=2.4847.)

A width sweep (0.40–5.00 µm) confirmed TM stays dominant at every practical
width, so this is not fixable by re-choosing `wg_width_um` — it's an
inherent property of a 2D effective-index slab with no vertical (thickness)
confinement, unlike a real 3D ridge waveguide's aspect ratio, which is what
makes TE0 fundamental in practice.

**Fix:** every eigenmode source/monitor across `waveguide.py`, `bend.py`, and
`bend_topopt.py` now passes `eig_parity=mp.TE` explicitly. Both Ez and Hz DFT
fields are captured; whichever actually dominates is used for the field
snapshot (don't assume Ez, even though the "TE" label might suggest it — Hz
is the one that dominates once TE is genuinely forced).

## `waveguide.py` (straight waveguide baseline)

| Parameter | Value | Note |
|---|---|---|
| `resolution` | 20 px/µm | Toolkit-wide default (`GLOBAL_PARAMS`). |
| `dpml_um` | 1.0 µm | PML thickness. |
| `wl_min_um` / `wl_max_um` | 1.30 / 1.40 µm | O-band. |
| `core_index` | 2.7 | *Not* bulk silicon (~3.45) — this toolkit's
  effective-index convention. |
| `length_um`, margins | see `derive_lengths()` | Port/source/PML offsets are
  *derived* from `length_um` so they never drift out of sync when it's swept. |
| Fitted `n_eff` | 2.4538 | Cutback-method phase fit at `wg_width_um=0.5` (re-measured
  after the width-unification pass below; max phase-residual 1.4e-6 rad). |

**Cutback loss fit removed:** the length-sweep cutback method's amplitude
channel (fit to `ln|S21|` vs. length) measured ~0.05 dB/cm — this is FDTD's
own discretization noise floor at `resolution=20`, not a real propagation
loss, since this cross-section has no absorption mechanism at all. The
baseline notebook and `models/waveguide.py` now fit and report **phase only**
(→ `n_eff`); a meaningful loss number requires a lossy material model (see
`racetrack.py`'s `ring_conductivity`/`D_conductivity` trick) or a full 3D
simulation.

## `bend.py` (90° bend baseline)

| Parameter | Value | Note |
|---|---|---|
| `resolution` | **40 px/µm** | See convergence study below. |
| `radius_um` | 2.0 | Representative default; swept 1.0–3.0 µm elsewhere. |
| `margin_um` | 1.5 µm | Clearance between structure/ports and PML. |

**Resolution convergence study.** `bend.py` needs a finer grid than the
toolkit-wide default (20 px/µm) to resolve the arc. A convergence sweep
(25 → 40 → 60 px/µm) at the representative radius, run at the prior
`wg_width_um=0.45` (this was a one-off side investigation, not reproduced by
the notebook's own cells, and was NOT re-run after the width-unification pass
below moved this module to `wg_width_um=0.5` — the numbers here predate that
change) found:

| Resolution | Insertion loss |
|---|---|
| 25 px/µm | 0.32 dB (not converged) |
| 40 px/µm | 0.4639 dB |
| 60 px/µm | 0.4629 dB (~converged) |

25 px/µm is not converged — the loss estimate moves by 0.14 dB going to 40.
Between 40 and 60 the estimate holds nearly flat (Δ ≈ 0.001 dB), so **40
px/µm** is used: the cheapest resolution in the converged plateau. Arc-
resolution requirements are governed by `radius_um`/`resolution`, not core
width, so this choice is expected to still hold at `wg_width_um=0.5` — but
that expectation has not been independently re-verified with a fresh
convergence sweep at the new width.

**Off-origin cell regression (fixed).** An earlier, tighter-bounding-box
version of the simulation cell shifted it via `mp.Simulation`'s
`geometry_center` to crop unused space. This measurably corrupted the
S-parameters — reciprocity's `max|S12−S21|` grew from ~0.003 to ~0.075 at
`resolution=25`, confirmed by a controlled before/after comparison against
an on-disk artifact, purely from passing a non-zero `geometry_center`
(independent of cell size). **Fix:** every absolute coordinate is instead
translated by a fixed `shift` so the tight bounding box lands on Meep's own
implicit origin, with no `geometry_center` argument anywhere (see `_domain()`
in `bend.py`).

**Independent-reference normalization (tried, reverted).** `simulate_baseline`
self-normalizes each run against its own in-situ incident coefficient. An
independently-measured straight-waveguide reference
(`_reference_incident`, Meep's own officially recommended technique) was
tried instead, expecting it to cancel a systematic bias (`|S22|²+|S12|²`
running 5–14% high on the y-launch port). It made results *worse*: energy
conservation deviation grew from ~0.09–0.14 to ~0.36–0.44 and passivity
started failing outright — `port_offset_um=1.0 µm` places the port monitor
too close to the bend's tangent point for "measure in isolation, divide" to
hold; near-field content from the bend contaminates the in-situ measurement
in a way that happens to cancel against itself but not against an
independent, perturbation-free reference. `_reference_incident` is kept in
the codebase, unused, as a documented dead end — the real fix would be
moving the port planes further from the bend, not swapping the
normalization method.

`_reference_incident` also needs `minimum_run_time=1000` (not Meep's default
decay-based stopping): an unobstructed straight line has nothing to scatter,
so the default criterion is satisfied almost immediately (empirically
t≈220 vs. t≈875+ actually needed), producing non-physical (sign-alternating)
per-frequency coefficients if stopped early.

**Bend-loss fit (Section 5): why not linear.** The 9-point radius sweep
(1.0–3.0 µm, 0.25 µm steps) shows loss falling steeply at small radius and
flattening near a noise floor at large radius — visibly not a straight line.
The model now fits `loss_dB(r) = floor + amplitude·exp(−r/decay_um)` per
`bend_type` via `scipy.optimize.curve_fit`, which both matches the data
shape and extrapolates sensibly; phase remains a linear fit (arc length is
already ≈ linear in radius).

**Circular vs. Euler crossover.** gdsfactory's `bend_euler(...,
with_arc_floorplan=True)` (the default) keeps the Euler bend's overall
footprint the same as a circular bend of the same nominal `radius`, which
forces the Euler curve's peak curvature *tighter* than `radius_um` at small
radii — so Euler is not simply "better" everywhere. The 9-point sweep (at the
current `wg_width_um=0.5`; see the width-unification note above) pins the
crossover between `radius_um=1.5` (Euler still worse: 0.497 dB vs. circular's
0.307 dB) and `radius_um=1.75` (Euler clearly better: 0.227 dB vs. 0.292 dB).
Don't trust the Euler fit below `radius_um≈1.5–1.75` as if Euler were the
better choice there. (At the prior `wg_width_um=0.45`, the same measurement
pinned the crossover slightly higher, between `radius_um=1.75`/`2.0` — the
qualitative conclusion, that Euler only wins from some finite radius up, is
unchanged, but the exact crossover point is width-dependent, so re-measure it
if `wg_width_um` changes again.)

**Validation tolerances** (`ENERGY_TOL=0.08`, `RECIPROCITY_TOL=0.10`,
`PASSIVITY_TOL=0.02`, `MAX_LOSS_FRACTION=0.15`), and the sweep-loosened
variants (`energy_tol=0.15`, `reciprocity_tol=0.15`, `passivity_tol=0.05`,
`max_loss_fraction=0.5`), were tuned empirically: `ENERGY_TOL` was lowered
from a pre-TE-fix `0.12` to `0.08`, comfortably above the ~0.045 actually
observed after the fix; the sweep tolerances are loosened to cover
`radius_um=1.0`'s expected higher loss.

## `bend_topopt.py` (adjoint topology optimization)

| Parameter | Value | Note |
|---|---|---|
| `resolution` | 20 px/µm | Same as `bend.py`'s baseline resolution family. |
| `design_grid_n` | 31 | → 961 design variables; grid pitch 0.15 µm. |
| `filter_radius_um` | 0.2 µm | Conic filter → approximate minimum feature size. |
| `eta` | 0.5 | tanh-projection midpoint. |
| `beta_schedule` | `[4.0, 8.0, 16.0, 32.0]` | Beta continuation — see below. |
| `iters_per_stage` | 20 | → 80 total NLopt evaluations. |
| `adjoint_minimum_run_time` | 50 | Floor on each forward/adjoint FDTD run's length. |

**Warm-start history (why the design region is 4.5×4.5 µm and not blank).**
A blank, uniform-gray initial guess (this toolkit's convention for every
other component) does not work well for this bend: after fixing the TE/TM
bug, an 80-iteration / 625-variable optimization from uniform gray converged
to a design that failed `checks.run_all_checks` (2.58 dB loss). Widening the
blank-start footprint to 2.4×2.4 µm (2401 variables) made it *worse*
(passivity 0.75 → 0.51 — too many free variables for the same iteration
budget to discover structure from scratch); coarsening the grid back down at
the original footprint didn't help either. The fix that worked: don't ask
the optimizer to *discover* a bend from nothing — warm-start it from an
already-good Euler-curve shape (`bend.py`'s own low-loss family,
`build_euler_initial_density()`) and let gradient ascent *refine* it. This
needs a design region big enough to hold the reference shape without
clipping it — a pure Euler spiral's footprint at a given minimum radius is
~1.87× that radius (not 1× like a circular arc), which is why
`design_region_x_um`/`design_region_y_um=4.5` and `init_bend_radius_um=1.0`
are sized the way they are.

**Objective drops at beta-stage boundaries (iterations 20/40/60).** The
optimization runs in 4 stages of increasing `beta` (tanh-projection
sharpness): 4 → 8 → 16 → 32, `iters_per_stage=20` each. A single fixed
high `beta` from the start made early iterations erratic, so the schedule
starts soft and progressively sharpens. The visible objective dip at each
stage boundary is a direct consequence of that: at the boundary, `beta`
jumps discretely while the raw optimizer variable `x_cur` carries over
unchanged from the previous stage. Since `mpa.tanh_projection` is much
sharper at higher `beta`, the *same* `x_cur` maps to a visibly more
binarized density immediately after the jump, producing a discontinuous
change in the projected geometry and hence the objective. This is expected,
benign behavior — not a bug — and the schedule is kept as-is because it
stabilizes what would otherwise be erratic early convergence. The honest,
trusted metric is the frozen-design S-parameter measurement (`simulate_
baseline`), not this per-iteration proxy curve.

**Reciprocity tolerance.** `checks.run_all_checks`'s `RECIPROCITY_TOL` is
loosened for this notebook relative to `bend.py`'s baseline, since a
freeform, only-approximately-symmetric optimized structure is not expected
to satisfy `S11=S22`/`S12=S21` as tightly as an exactly-symmetric parametric
bend does.

## `racetrack.py` (all-pass racetrack resonator)

`racetrack.py` never relied on `eig_parity=mp.NO_PARITY` (unlike the other three
modules above) — `eig_parity=mp.TE` was forced from the start, after the same MPB
check described above was run for this device's own cross-section
(`wg_width_um=0.5`, `core_index=2.7`, `clad_index=1.44`): `NO_PARITY` would have
picked TM (n_eff=2.509) over TE (n_eff=2.409) here too, so the explicit `mp.TE`
was necessary from day one, not a later fix.

| Parameter | Value | Note |
|---|---|---|
| `resolution` | 25 px/µm | Traded down from an earlier 30 px/µm — a controlled test found resolution changes the measured S-parameters by well under 1% here, freeing budget for `min_sim_time` below, which is what actually mattered. |
| `min_sim_time` | 7000 | Floor on every FDTD run's length — see convergence study below. |
| `n_freq` | 301 (baseline), 402 (sweep, narrowband) | See resonance-sampling study below. |
| `ring_conductivity` | 0.001 | Intrinsic material loss standing in for a real waveguide's propagation loss — needed both for a meaningful critical-coupling target and for the FDTD ring-down to converge in practical time. |

**FDTD convergence time.** An early version measured `|S|^2` **above 1** at several
sweep points — a physical impossibility for a passive device. Resolution
(30/45/60 px/µm) was ruled out first (changed the bus-only reference by well under
1%). The actual cause: `mp.stop_when_dft_decayed()`'s plateau-detection test can
register "converged" well before a high-loaded-Q resonance has actually rung down.
A controlled scan of `min_sim_time` (1500/3000/6000/10000) showed the violation
shrinking monotonically and plateauing at physically valid values from 6000
onward — `min_sim_time=7000` is the fix, confirmed at this module's actual
`resolution=25`. Tightening `dft_decay_tol` alone was tried first and only made
runs slower without fixing the symptom. This value is specific to this device's
ring-down time, not a universal constant — re-run the same scan for a
meaningfully different `ring_conductivity`, gap, or radius.

**Left/right reference normalization.** `_reference_incident()` originally
launched its bus-only reference from the left only, and `simulate_baseline()`
reused that one reference for both excitation directions — assuming a
left-launched `EigenModeSource` injects exactly the same power a right-launched
one would. That's true by the bus's exact mirror symmetry in principle, but not
in practice, because `source_offset`/`port_offset` aren't exact multiples of the
Yee grid spacing: a measured ~8-10% amplitude mismatch at `resolution=25` was
silently feeding `S22`/`S12` the wrong denominator and was the main cause of
`reciprocity_check` failing. **Fix:** `_reference_incident(params,
launch_from=...)` now runs a genuinely separate, direction-matched reference for
each side — a general fix applicable to any left/right-symmetric bus geometry.

**Resonance-aware energy conservation.** The generic 2-port
`checks.energy_conservation_check` applies one tolerance band to
`|S11|^2+|S21|^2` across the whole spectrum — correct for a non-resonant device,
but wrong for a resonator, where a near-critical-coupled ring is *supposed* to
dissipate close to 100% of input power right at resonance while staying lossless
everywhere else. An early pass here saw this large on-resonance deviation and,
without plotting the full spectrum, wrongly concluded it must be a further
convergence bug — it wasn't: the saved S-parameters never exceeded 1 anywhere,
and the deviations were narrow, smooth, Lorentzian-shaped dips sitting exactly on
resonance. **Fix:** `racetrack.energy_conservation_check` automatically finds
each resonance dip's full shoulder and applies a **tight** (`off_resonance_tol
=0.02`) lossless requirement off resonance (the part that actually catches a
broadband normalization/convergence bug) and only a **physicality** bound
(no amplification) on resonance, where near-total loss is the intended result.
This "tight off resonance / loose on resonance" split is a property of resonators
in general — reuse the pattern for any new resonant component.

**Resonance sampling (`n_freq`) and the `S11=S22` symmetry check.** At the
original `n_freq=101` (~1nm/point over the 100nm O-band), this device's small
on-resonance reflection peaks (`S11`/`S22`) were severely under-sampled, making
bin-by-bin `S11=S22` comparison sensitive to exactly where each peak's few sample
points landed (`max|S11-S22|` ranged 0.004-0.29 across otherwise-identical runs).
Raising `n_freq` to 301 fixed this cleanly in an isolated script (~0.013), but
running the same code through a notebook kernel reproduced the original ~0.29
value regardless of the `n_freq` fix — an unresolved, execution-context-dependent
symptom isolated to this one secondary sub-check (`S11=S22` is a structural
expectation from this device's mirror symmetry, not a fundamental law the way
`S12=S21` is; true reciprocity stayed small and consistent, ~0.03, throughout).
`RECIPROCITY_TOL=0.35` is set with real margin above the observed 0.29 worst
case rather than chasing this further through expensive re-runs. Because this
racetrack's resonances (FSR~18nm) sit close together relative to the 100nm O-band,
even at `n_freq=301` the baseline grid still under-resolves the true dip depth —
Section 9's narrowband sweep (`1.3050-1.3200um`, `n_freq=402`, focused on one
resonance) is what the design-selection and field-visualization sections actually
rely on for precise resonance location and depth, not the baseline.

**CMT fit — mode-number aliasing in `scipy.optimize.curve_fit`.** The closed-form
extraction (Bogaerts et al. 2012 Eqs. 3/7/12, from the measured dip wavelength,
depth, and FWHM) already lands very close to the true `(kappa, alpha, n_eff)` —
but refining it with `curve_fit` on the *complex* `(Re, Im)` S21 initially
diverged to the parameter bounds (`kappa=1`, `n_eff` pinned at its search-window
edge) instead of improving the seed. Root cause: with round-trip length `L~35-40
µm`, the round-trip phase `2*pi*n_eff*L/wl` is extremely sensitive to `n_eff` (a
full 2π wrap needs only `dn_eff ~ wl/L`), so the complex-domain residual has many
aliased local optima within the seed's search window that a linearized
Levenberg-Marquardt/trust-region step can jump into; the mismatched natural
scales of `kappa` (~0.1-1), `alpha` (~1e-3), and `n_eff` (~2.4) made this worse.
**Fix:** fit the real-valued, non-oscillatory `|S21|^2` power spectrum instead of
the complex value, pass an explicit `x_scale` matching each parameter's natural
magnitude, and keep the closed-form seed outright if the refined fit's residual
is ever worse than the seed's own (confirmed this combination reliably recovers
the seed's already-good answer rather than diverging).

## `add_drop_ring.py` (add-drop ring resonator)

Reuses `racetrack.py`'s exact stadium-carving ring construction (a racetrack
already has two straight coupling segments; only one is coupled to a bus
there) and adds a SECOND bus mirrored across the ring's own center line,
coupled to the other straight segment with the same `gap_um`/
`coupling_length_um` (symmetric-coupling design decision). Self-contained --
does not import `racetrack.py` and does not route through
`pic_toolkit.sparams`/`pic_toolkit.checks`/`pic_toolkit.sweep.grid_sweep`
(all hardcoded to a 2x2 S-matrix), following the precedent already set by
`coupler.py`/`mzm.py`/`mzi.py`.

| Parameter | Value | Note |
|---|---|---|
| `gap_um` | 0.20 (fixed) | Same value and rationale as `racetrack.py` -- see the gap-sweep dead end below for why this is fixed, not swept, here too. |
| `coupling_length_um` | 3.0 (baseline), swept 1.0-5.0 | The critical-coupling search knob, same choice as `racetrack.py`, for the same reason. |
| `resolution`, `min_sim_time`, `ring_conductivity` | Same as `racetrack.py` | Started from racetrack's already-validated values; no resolution-sensitivity concern here since `gap_um` never changes. |
| `reciprocity_tol` | 0.35 | Same value and same rationale as `racetrack.py`'s `RECIPROCITY_TOL` -- see below. |

**Dead end: a `gap_um` sweep was tried first.** The initial design for this
module's critical-coupling search swept `gap_um` (0.15-0.35um, `coupling_
length_um` fixed) instead of `coupling_length_um`, reasoning that with two
*symmetric*, equal couplers a lossless add-drop ring shows full
through-port extinction at resonance for ANY equal `kappa1=kappa2`
independent of gap, so the ring's own intrinsic loss should pick out a
specific best gap. **This ran cleanly** (every point validated except the
weakest, 0.35um -- see below) but the extinction/drop-transfer trend was
**monotonic across the entire swept range, with no interior turnover**:

| `gap_um` | through-port extinction | drop-port peak |
|---|---|---|
| 0.15 | 0.903 | 0.464 |
| 0.20 | 0.626 | 0.184 |
| 0.25 | 0.307 | 0.051 |
| 0.30 | 0.266 | 0.023 |
| 0.35 | 0.105 | 0.004 |

The true critical-coupling gap therefore sits below 0.15um -- inside
territory `racetrack.py`'s own predecessor (`ring.py`, since removed)
already documented hitting a resolution/reciprocity wall on (no turnover
found below ~0.12-0.13um). Rather than chase that wall for a second
component, the swept knob was switched to `coupling_length_um` (fixed
`gap_um=0.20`) -- `racetrack.py`'s own original choice, and for the same
reason: it also has a bonus the all-pass device's own docstring doesn't
emphasize as strongly here -- a gap sweep makes the measured S-parameters
progressively MORE sensitive to grid resolution as the gap shrinks (fewer
pixels resolve it), while `coupling_length_um` changes the coupling REGION
LENGTH, a distance the grid resolves exactly as well regardless of its
value. The abandoned gap-sweep artifacts are not deleted from
`data/sparams/add_drop_ring/sweep/` -- they remain a valid, honestly-run
(if ultimately not selected) result.

**`gap_um=0.35`'s validation failure (energy-conservation, not a bug).**
The weakest-coupling gap-sweep point failed the automated energy check
(`off_resonance_deviation` ~0.07-0.09 against a 0.02 tolerance) despite a
near-perfect median baseline (~0.9998-1.0001). Root cause: at this device's
weakest tested coupling, the resonance dip is extremely shallow (extinction
~10%, barely above `energy_conservation_check`'s own `min_dip_depth=0.1`
detection threshold), so `_find_resonance_dip_spans` inconsistently
classifies genuinely resonance-adjacent points as "off-resonance," which
then wrongly get held to the tight lossless bound. Not investigated further
since this sweep point was abandoned anyway (see above) -- flagged here so
a future very-weak-coupling sweep point isn't mistaken for a real bug
without first checking whether this is the cause.

**The `coupling_length_um` sweep found a genuine interior turnover.**
Unlike the abandoned gap sweep, extinction/drop-transfer rose across
`[1.0, 2.0, 3.0, 4.0]` um then FELL at `5.0` um (extinction: 0.36, 0.53,
0.63, 0.77, 0.73) -- a real peak, not an edge artifact, confirming
`coupling_length_um` is the right knob for this device. `coupling_length_um
=2.0`'s point failed the (tight, 0.02) off-resonance energy tolerance by a
small margin (0.0246) -- not investigated further since it isn't the
selected point and the deviation is far smaller than the genuine
weak-coupling failure documented above. The selected point
(`coupling_length_um=4.0`, `gap_um=0.2`) passed all three checks cleanly.

**CMT fit quality at the selected point was initially fair, not excellent
(RMS=0.097), then substantially improved by switching to a narrowband,
single-resonance fit (see below).** The original 4-parameter
(`kappa1`,`kappa2`,`alpha`,`n_eff`) fit to the full 100nm baseline band
reproduced the correct resonance SHAPE and spacing (FSR) but visibly
underestimated the FDTD dip depth and drop-peak height at several
resonances (`kappa1=0.906, kappa2=0.230` -- notably asymmetric despite the
symmetric-gap design).

**Dead end: a symmetric (`kappa1=kappa2`) full-band fit is WORSE, not
better, and the fit is provably not seed-sensitive.** Since both couplers
share the same `gap_um`/`coupling_length_um`, a natural refinement is to
constrain `kappa1=kappa2` (3 free parameters instead of 4), fit jointly to
both couplers' measured spectra (`input_through`+`add_drop` as one
"through-type" dataset, `input_drop`+`add_through` as one "drop-type"
dataset). Tried first on the full 100nm baseline band: RMS residual got
WORSE (0.097 -> 0.112), and -- the important diagnostic -- **10 different
`kappa` seeds spanning 0.2-0.99 all converged to the exact same optimum**
(`kappa=0.245`). Seed-independence like this rules out a seeding/local-minimum
problem: the root cause is that individual resonance dip depths visibly vary
across the 100nm band (confirmed by eye on an overlay plot -- some dips
reach ~0.6, others only ~0.85, within the same spectrum), which no single
constant-`(kappa,alpha,n_eff)` model, symmetric or not, can reproduce
everywhere at once.

**Fix: fit to a single, finely-resolved resonance, not the full band --
exactly racetrack.py's own precedent.** A dedicated narrowband,
high-density re-run at the selected `coupling_length_um=4.0`
(`wl_min_um=1.350`, `wl_max_um=1.360`, `n_freq=401`, bracketing one
resonance located from the Section 9 sweep) was simulated and validated
(all checks passed) before re-fitting. The symmetric model, fit only to
this one resonance, dropped the RMS residual to **0.0057 (input coupler
alone), 0.0061 (add/drop coupler alone), 0.017 (both couplers combined)** --
a 5-20x improvement over the full-band symmetric attempt, and the fitted
curves visually hug the FDTD data closely across the whole narrowband
window (both shoulders and the dip/peak center). This confirms the
per-resonance-varying full-band data, not the symmetric-coupling
assumption itself, was the real obstacle.

**A real (not just full-band-averaging) discrepancy between the two
couplers persists even on this single, well-resolved resonance.**
Input-coupler-alone: `kappa=0.232`. Add/drop-coupler-alone: `kappa=0.266`
-- a ~13% relative difference, much smaller than the full-band fit's ~4x
discrepancy (0.906 vs 0.230), but not zero, and the raw reciprocity
diagnostic (`diff_input_through_vs_add_drop`) is still ~0.21 on this
narrowband run alone (comparable to the full-band value, ~0.15-0.23 across
different sweep points) despite both excitations independently passing the
tight, resonance-aware energy-conservation check. Not root-caused further
here (would need e.g. an even longer `min_sim_time` specifically for this
comparison, or a dedicated investigation into whether launching from the +x
vs -x side of an otherwise-mirror-symmetric domain introduces a small
systematic difference) -- flagged as a known, disclosed limitation. The
SAVED `fitted_model` at this stage used the COMBINED (both-couplers-stacked)
symmetric fit (`kappa=0.250`, RMS=0.017) as the best-available compromise --
superseded by the independent-`kappa1`/`kappa2` fit below.

**Final fix: stop constraining `kappa1=kappa2` at all -- fit both
independently on the narrowband data.** The two directions of propagation
through the ring are not physically identical (different launch side,
different net path around the loop before reaching each measurement
plane), so there's no first-principles reason the *measured* coupling
strengths must come out equal even for a geometrically symmetric device --
this matches the ~13% discrepancy already seen between the two
single-coupler-alone symmetric fits above. Fit jointly to all 4 measured
transfer functions (`input->through`, `add->drop`, `input->drop`,
`add->through`) with independent `kappa1`,`kappa2`, multi-started from 5
seed pairs spanning both `kappa1>kappa2` and `kappa1<kappa2` (each crossed
with 3 candidate mode numbers, 15 total): **every seed converged to the
same answer** (`kappa1=0.274, kappa2=0.229`, RMS=0.0081) regardless of which
direction the seed asymmetry favored -- the same seed-independence
diagnostic used above to rule out a fitting artifact, this time confirming
the asymmetry itself IS real and data-supported, not an artifact of seed
choice. RMS residual is roughly HALF the `kappa1=kappa2`-constrained fit's
0.0171 on the identical narrowband data. This is now the actual SAVED
`fitted_model`.

**Mode-number ambiguity, disclosed rather than resolved.** Across the 3
candidate mode numbers (`m`, `m-1`, `m+1`), residuals differed by <0.001% --
statistically indistinguishable, expected for a single-resonance fit (only
a multi-resonance fit constrains the ABSOLUTE mode order, and the full-band
data has its own per-resonance-variation problem documented above). The
seeded mode number (`n_eff=2.438`, consistent with the earlier symmetric
fits and reasonably close to `racetrack.py`'s own validated ~2.36-2.41 range
for the same cross-section) is kept rather than chasing a meaningless
residual difference toward a neighboring candidate (`n_eff=2.40` or `2.47`).

**Cell sizing for an asymmetric-about-origin geometry.** `racetrack.py`'s
own `cell_y` formula assumes only ONE bus needs dpml+margin clearance (the
far side is empty cladding out to the PML). Naively extending that formula
by adding one bus-width's worth of space undersized the domain here: the
ring's center line (`ring_center_y`) is not at `y=0`, so the two buses
(mirrored across `ring_center_y`, not across `y=0`) sit at different
distances from the Meep cell's own center. Sizing `cell_y` from the two
buses' *total span* put barely-insufficient clearance (1.4um against a
required 1.5um dpml+margin) on whichever bus happened to sit farther from
`y=0`. **Fix:** `cell_y` is instead sized from `2 * (max(bus_y,
abs(drop_bus_y)) + wg_width_um/2 + dpml_um + margin_um)` -- clearance
guaranteed on both sides, the nearer side simply gets harmless extra slack.

**Source/monitor separation.** An early version placed the `EigenModeSource`
directly at the same location as the self/reflection mode monitor for
whichever port was being excited (`port_offset`, not `source_offset`),
leaving no propagation distance between them. Fixed to source from
`source_offset` (as `racetrack.py` does), monitor at `port_offset`.

**Mode-coefficient index convention is NOT racetrack's simple "self=backward,
other=forward" rule.** Naively porting `racetrack.py`'s two-port index
rule (self-port -> mode index 1, every other port -> index 0) to 4 ports
produced badly wrong results for the `add` excitation specifically (a
resonance-scale ~72% energy-conservation deficit and >0.85 reciprocity
mismatch, versus a normal few-percent deviation for the `input` excitation)
-- because that rule silently assumes the excited port is always on the -x
side, true for every 2-port device in this toolkit so far but not for
`add` (which sits on the +x side). **The correct, general rule**: `input`
and `drop` sit at the domain's -x edge and `through`/`add` sit at the +x
edge (see module docstring); in the with-ring run's steady state, ANY
energy reaching one of these edge monitors is -- by definition of a
correctly-absorbing PML -- traveling AWAY from the device on that edge's
own side. So the coefficient to keep is simply the backward (-x, mode
index 1) component for `input`/`drop`, and the forward (+x, index 0)
component for `through`/`add`, **regardless of which port was excited**.
This was verified two ways: it makes the two independent excitation runs'
reciprocity check pass (a real, both-runs-must-agree test, not just an
internal consistency check), and it matches a first-principles ring-
circulation argument (launching `input` [+x] drives the ring's input-side
segment in +x, which forces the opposite (add/drop-side) segment to move
in -x -- i.e. towards `drop`, not `add`).

**`reciprocity_tol=0.35`, not a tighter default.** Same phenomenon
`racetrack.py` already documented above (resonance-sampling and the
`S11=S22` check): a narrow, high-Q resonance sampled by a finite `n_freq`
grid makes a point-wise max-difference reciprocity check between two
*independently* FDTD-converged runs highly sensitive to exactly which grid
point lands nearest each run's own dip. Confirmed directly here at a
coarse `n_freq=11` smoke-test grid: the SAME two-excitation reciprocity
comparison swung between `max_diff=0.02` (passing a tight 0.05 tolerance)
and `max_diff=0.12` (failing it) between otherwise-identical runs differing
only in `min_sim_time`/`resolution` -- not evidence of a remaining port/
direction bug, just this toolkit's already-known sampling-density
sensitivity. Reused `racetrack.py`'s own `RECIPROCITY_TOL=0.35` value
directly rather than re-deriving a new one from scratch.

## `coupler.py` / `mzi.py` (directional coupler, passive Mach-Zehnder interferometer)

These two modules are the toolkit's first genuinely 4-port devices. `coupler.py`
builds a directional coupler (two waveguides S-bent from a wide separation down to a
tight coupling gap); `mzi.py` reuses that same, already-validated 50:50 design for
both stages of a passive Mach-Zehnder interferometer, joined by a straight reference
arm and a raised-cosine-bump delay arm carrying the extra path length `delta_L_um`.

| Parameter | Value | Note |
|---|---|---|
| `coupler.coupling_length_um` | 13.708639 µm | The 50:50 point at `resolution=40` (see the resolution=25→40 update below); was 13.848 µm at `resolution=25`. |
| `coupler.gap_um` | 0.2 µm | Matches `racetrack.py`'s choice; resolution sensitivity re-checked below (`resolution=40`, not `25`). |
| `coupler.n_seg` | 24 | Axis-aligned blocks approximating each S-bend (max slope ~0.17, <2% width error — see below). |
| `mzi.bump_n_seg` | 200 | Finer than the coupler's `n_seg` — the delay arm's bump offset can be an order of magnitude larger. |
| `mzi.min_sim_time_factor` | 2.0 | Scales with `cell_x`, not a fixed constant — see below. |

**TE/TM mode-mislabeling bug in `coupler.py` (fixed).** Every `eig_parity=` here used
to be `mp.NO_PARITY`, on the premise that this device's dominant field is `Ez`. A
direct check found the opposite: `eig_parity=mp.TE` gives a genuine guided mode with
`Ez` exactly zero and `Hz` dominant, while `NO_PARITY` was silently selecting the TM
mode instead — inconsistent with every other component in the toolkit, and worse, an
earlier "fix" that just switched field *capture* to whichever of `Ez`/`Hz` happened to
be non-empty had papered over the real issue without resolving it. `eig_parity` is now
forced to `mp.TE` at the source and all monitors. Because the actual launched/measured
polarization changed, `coupling_length_um` needed re-measuring from scratch (the
original 9.9 µm value was tuned under the wrong, TM polarization; genuine TE couples
more slowly per unit length here, hence the longer ~13.8 µm result) — not just a
documentation fix.

**Sweep range widened, monotonicity guard added (`06_directional_coupler.ipynb`).**
The original Section 10 sweep, `[2, 4, 6, 8, 10, 12, 15]` µm, stayed entirely below
the beat curve's first peak ($\kappa L_c=\pi/2$), so `np.interp(0.5, T_cross, Ls)`
was always safe without checking. A later pass widened the range to
`[2, 6, 10, 14, 18, 22, 26]` µm specifically to see the curve turn over past that
peak (confirmed: `T_cross` reaches 0.954 at 26 µm, still short of a full period but
well past the 50:50 crossing) — this makes an unchecked `np.interp` genuinely risky
in general (a wider or differently-spaced future sweep could sample past the peak
and silently corrupt the interpolation), so an explicit monotonicity check
(`np.diff(T_cross_at_wl0) > 0`) now gates the interpolation, falling back to the
longest monotonic-increasing prefix and printing a warning if it ever fires. It did
not fire for this range: `T_cross_at_wl0` stayed strictly increasing across all 7
points, and the interpolated result (13.848 µm) landed within 0.03 µm of the prior
narrower-sweep's 13.873 µm value. `mzi.py`'s `DEFAULT_PARAMS["coupling_length_um"]`
was re-synced to 13.848 µm to match; `07_mzi.ipynb` and `data/design_points/mzi.yaml`
were deliberately **not** re-run/regenerated for this shift (see
`docs/troubleshooting_log.md`), so that notebook's own cached outputs still reflect
the prior 13.873 µm value pending a future re-run.

**TE/TM mode-mislabeling bug in `mzi.py` itself (fixed).** Independent of the
`coupler.py` bug above -- `mzi.py` has its own `_make_simulation`/
`_run_one_excitation` (it does not delegate mode launch/measurement to
`coupler.py`) -- the same `eig_parity=mp.NO_PARITY` pattern was still present at
this module's own source and all four mode monitors, with DFT capture recording
only `Ez`. Fixed the same way: `eig_parity` forced to `mp.TE` everywhere, DFT
capture now records both `Ez`/`Hz` and the field snapshot reports whichever
actually dominates -- confirmed `Hz` dominant (`Ez` exactly 0), same convention as
`coupler.py`. This does *not* change `coupling_length_um` (that constant is
`coupler.py`'s own, already re-measured under TE and re-synced into `mzi.py`'s
`DEFAULT_PARAMS` per the "Sweep range widened..." entry above) -- it corrects the
polarization `mzi.py` itself launches and measures with. This also resolves the
previously-deferred re-run: `07_mzi.ipynb` and `data/design_points/mzi.yaml` were
re-executed as part of this fix, so they are no longer stale relative to
`mzi.py`'s `coupling_length_um` default.

**Consequences of the re-run, measured directly.** Across the full Section 10
`delta_L_um` sweep (`[2.5, 5.0, 7.5, 10.0, 12.5, 15.0]`), the energy-budget
residual is **non-monotonic** in `delta_L_um` under genuine TE physics (0-3.8%,
worst at `delta_L_um=10.0` -- this notebook's own baseline point) -- unlike the
prior (TM-under-`NO_PARITY`) measurement's clean growth-with-length trend, which
does not survive the fix. `PASSIVITY_TOL` (baseline/selected re-validation) had to
be loosened from `coupler.py`'s single-stage `0.01` to `0.03` to match: the
strict budget is now measurably violated (`max_power_fraction` up to `1.0231`,
same worst point, `delta_L_um=10.0`), a small, non-monotonic overshoot
comparable in size to the energy-budget residual -- consistent with (though not
identical in mechanism to) this toolkit's own documented precedent for cascaded
real-FDTD devices showing small non-physical passivity overshoots
(`circuits/mzi_lattice.py`'s section below), not evidence of a bug. Both energy
and passivity worst-casing at the SAME point (`delta_L_um=10.0`, coincidentally
also the baseline default) is consistent with ordinary discretization/mode-overlap
noise rather than a systematic length-dependent effect.

**Design selection changed as a direct result: `SELECTED_DELTA_L_UM` moved from
`15.0` to `7.5`.** Section 11's selection logic (unchanged: among validated points
with `n_peaks>=2`, pick max extinction) is genuinely sensitive to the corrected
physics -- under TE, `delta_L_um=7.5` now measures the deepest extinction
(35.83dB) of the 3 points with a measurable in-band FSR (`7.5`/`12.5`/`15.0`),
displacing the prior selection. `data/design_points/mzi.yaml` and
`data/sparams/mzi/selected/` now reflect `delta_L_um=7.5`. This does NOT affect
`circuits/wdm_mux4_sax.ipynb`, which never reads the "selected" design point --
it calls `models.mzi.mzi_at_sweep_point(15.0, ...)`/`mzi_at_sweep_point(7.5, ...)`
directly by name against Section 10's own sweep artifacts (both regenerated fresh
by this same re-run), independent of which point Section 11 itself prefers.

**UPDATE: `resolution=25` (both `coupler.py` and `mzi.py`) was itself found to be
substantially under-converged, superseding everything above that was measured at
that resolution — now `resolution=40`.** While investigating a real-arm
recalibration for `circuits/wdm_mux4_sax.ipynb`, `mzi.py`'s baseline
(`delta_L_um=10.0`) was directly compared at `resolution=25` vs. `resolution=40`:
energy-conservation deviation dropped from 3.79% to **0.0%**, reciprocity tightened
~17x (1.92%→0.11%), and the passivity budget's real violation (`max_power_fraction
=1.0231`, needing `PASSIVITY_TOL` loosened to 0.03) disappeared entirely (`0.991`,
comfortably under 1) — not a minor refinement, a qualitatively different, far
better-converged answer (the through/cross power split itself changed from
4.8%/88.6% to 42.6%/55.3% at this one point). `coupler.py`'s own default was
re-checked the same way and changed too, though far less dramatically (this
simpler single-stage device is much better-behaved): the 50:50 coupling length
moved from `13.848µm` to **`13.708639452498595µm`** (a clean, monotonic beat curve,
no turnover, all 7 sweep points passing cleanly) — re-synced into `mzi.py`'s own
`DEFAULT_PARAMS`. Both modules' `DEFAULT_PARAMS["resolution"]` are now `40`.
`coupler.gap_um=0.2µm`'s own "already confirmed... sub-1% sensitivity to
resolution" claim in the table above was made under the pre-TE-fix physics and
does not hold under genuine TE. Re-quantified via `06b_directional_coupler_gap_
sweep.ipynb`'s full 5×7 `(gap_um, coupling_length_um)` grid at `resolution=40`
(was `resolution=25`): the "generic" baseline noise floor tightened noticeably
(dispersion-residual std 0.03–0.5° → 0.00–0.03° across 33 of the 35 grid points),
confirming the coupler *is* resolution-sensitive at `resolution=25`, consistent
with `mzi.py`'s own much larger effect. The two genuine outliers — dispersion
residual spiking at each gap's own coupling extremum (`gap=0.20,Lc=26`:
41.5°→45.48°; `gap=0.15,Lc=18`: 44.0°→44.87°) — stayed essentially unchanged in
magnitude at the finer resolution, confirming that specific spike is real,
resolution-independent physics (`d(kappa)/d(Lc)≈0` there, maximally sensitive to
any perturbation), not a numerical artifact — see
`06b_directional_coupler_gap_sweep.ipynb`'s own Section 8 for the full picture.
`notebooks/07_mzi.ipynb`'s baseline/sweep/selection above (delta_L_um=7.5 selected
under `resolution=25`) all needed re-running at `resolution=40` — **now done.**
Every specific number above this UPDATE (residuals, the 7.5 selection, the 13.848
coupling length) is historical, superseded by this re-run.

**`resolution=40` re-run results (`07_mzi.ipynb`, complete).** Baseline
(`delta_L_um=10.0`, `coupling_length_um=13.708639452498595`): energy deviation
0.0%, reciprocity 0.04%, passivity 0.990 — all comfortably inside even a single
`coupler.py` stage's own strict budget (`energy_tol`/`reciprocity_tol`/
`passivity_tol` = 0.03/0.03/0.01), though the notebook itself keeps the wider
cascaded-device tolerances (`ENERGY_TOL=0.08`, `RECIPROCITY_TOL=0.05`,
`PASSIVITY_TOL=0.03`) rather than re-running again purely to tighten thresholds
with no failing case. Full `delta_L_um` sweep, all 6/6 points passing:

| `delta_L_um` | energy dev. | reciprocity | passivity (`max_power_fraction`) |
|---|---|---|---|
| 2.5 | 0.0% | 0.03% | 0.994 |
| 5.0 | 1.59% | 2.35% | 1.008 |
| 7.5 | 0.62% | 0.03% | 0.981 |
| 10.0 (baseline) | 0.0% | 0.04% | 0.990 |
| 12.5 (selected) | 0.0% | 0.04% | 0.990 |
| 15.0 | 1.37% | 1.94% | 1.003 |

Worst case at `delta_L_um=5.0` (previously not even distinguishable as a
worst-case under `resolution=25`'s much larger, non-monotonic residuals) — every
value, including the two passivity figures marginally over 1.0 (`5.0`, `15.0`),
would clear a strict `passivity_tol=0.01` budget too.

**Design selection changed again, for a third time (physically real, not a
symptom of the bug this time): `SELECTED_DELTA_L_UM` moved from `7.5`
(`resolution=25`, TE-fixed) to `12.5`.** Section 11's logic is unchanged
(among validated points with `n_peaks>=2`, pick max extinction); under
`resolution=40`, `delta_L_um=12.5` has the deepest extinction (36.94dB) of the
now-3 points with a measurable in-band FSR (`12.5`/`15.0`, plus `10.0` which
narrowly missed `n_peaks>=2`). `data/design_points/mzi.yaml` and
`data/sparams/mzi/selected/` reflect `delta_L_um=12.5`. As before, this does
NOT affect `circuits/wdm_mux4_sax.ipynb`, which reads specific sweep points by
name (`mzi_at_sweep_point`), not the "selected" design point — but that
notebook's own `DELTA_L_1_UM=15.0`/`DELTA_L_2_DOWN_UM=7.5` reference choices,
and the entire stage2_up recalibration search attempted at `resolution=25`
(see `circuits/mux4_tree.py`'s own section below), all need re-doing against
this `resolution=40` data — not yet done as of this entry.

**Delay-arm design history: single bump vs. serpentine.** An earlier version of
`mzi.py` used a serpentine of many short-period bumps, reasoning that many
small-amplitude wiggles would look "gentler" than one large one. That reasoning was
wrong: for a raised-cosine bend, radius of curvature scales like
(half-period)²/amplitude, so packing many periods into a fixed span shrinks the radius
far faster than the smaller per-period amplitude helps — a 9-period serpentine for
`delta_L_um=10` measured a radius of curvature of only ~0.3 µm (a large, spurious
broadband power deficit in `energy_conservation_check` to match), whereas a single
bump over the same span needs a much larger absolute amplitude but a far gentler
~3 µm radius, comparable to `bend.py`'s own already-validated `radius_um=2.0`. A
separate widen/narrow transition at the arm ends (added purely to give the bump
clearance) was later removed too, once the bump was changed to bulge *away* from the
centerline rather than toward the reference arm — removing it also removes 4 S-bend
junctions worth of scattering.

**Delay-arm geometry: rotated Blocks, not axis-aligned ones or a single Prism.**
`_build_delay_arm_blocks` rotates each small Block to its own local tangent
direction, unlike the coupler's own axis-aligned S-bend blocks. Axis-aligned stacking
only holds the *y-extent* of each segment constant, not the true width perpendicular
to the local propagation direction — an excellent approximation for the coupler's
gentle S-bends (max slope ~0.17, confirmed under 2% width error) but wrong for this
bump, whose slope approaches order 1 near its midpoint at large `delta_L_um`
(confirmed: the effective width there would drop to less than half the intended
value without rotation — a real wrong waveguide, not a rendering artifact). A single
big `mp.Prism` (an offset curve joined into one polygon) was tried first instead of
many small rotated Blocks — geometrically correct, but far too slow: a Prism whose
bounding box spans nearly the whole simulation cell forces Meep's subpixel-averaging
to test every grid pixel against it, confirmed to take minutes just for
`get_permittivity_map` versus seconds for many small Blocks with their own tiny local
bounding boxes.

**`min_sim_time_factor` and the long-domain decay-detection bug.** This device's
simulation cell is far longer than any other component's (two coupler stages plus a
delay arm), and `PIC_components/MZM/03_mzm_design_v4.ipynb` had already documented a
real bug on long domains: `mp.stop_when_dft_decayed()`'s decay check can be satisfied
by a quiet *early* transient at the monitor point, before the pulse has even arrived
there, silently leaving the far output monitor's DFT at exactly 0+0j. A
`minimum_run_time` floor scaled to the domain's own optical transit time
(`min_sim_time_factor * core_index * (cell_x + delta_L_um)`) fixes this — the
`+ delta_L_um` term accounts for the delay arm's bump being physically longer than
`cell_x` alone suggests. This did not, on its own, fully close an observed gap: at
large `delta_L_um` (confirmed at `delta_L_um=20`, `resolution=25`) energy conservation
and reciprocity (`S21` vs. `S12`, which must match regardless of `delta_L_um` by
genuine Lorentz reciprocity) both still show a real, if modest (~5-6%), residual,
present with or without this term — kept anyway since the estimate is independently
more accurate with it, but the residual itself is an open item, not fully explained by
bend-loss geometry or reciprocity-check logic alone.

**Baseline/sweep tolerances (Section 8/10/11).** `ENERGY_TOL`/`MAX_EXPECTED_LOSS`/
`RECIPROCITY_TOL` are loosened beyond a single `coupler.py` stage's own budget since
this device chains two stages plus a delay-arm bump in series. Measured directly (at
the corrected `coupling_length_um=13.873`, `resolution=25`): the baseline
`delta_L_um=10` shows a ~3.9% energy-budget residual, growing to ~6.7% at the larger
`delta_L_um` Section 11's sweep selects — both comfortably inside
`MAX_EXPECTED_LOSS=0.08`, but well above a single coupler stage's own tight ~2-3%
budget. `ENERGY_TOL` is set to `0.08` (matching `MAX_EXPECTED_LOSS` itself) to cover
the full range with real margin above the largest value actually measured, rather
than a tighter, untested guess — reciprocity and passivity stay comfortably clean
throughout. This growth-with-`delta_L_um` pattern is consistent with the larger,
still-open residual documented above at `delta_L_um=20` (the same class of
long-domain effect). `delta_L_um=5`/`15` (the sweep's other two points) were not
individually re-checked at full resolution beyond the sweep's own (loosened)
tolerances.

**Fringe-detection bug in Section 10's FSR measurement (found and fixed).** The FSR
this notebook reports is *measured* directly from each swept point's simulated
`|S41|^2` spectrum (mean spacing between detected fringe maxima) — there is no
closed-form FSR formula anywhere in this notebook to get wrong; the bug was in the
*peak detector* underneath that measurement. An earlier `_local_maxima_indices` had
no minimum-prominence threshold at all, on the reasoning that "this device's spectra
are smooth FDTD output ... so no prominence/smoothing threshold is needed." That
reasoning was wrong: at `delta_L_um=5`, the true FSR is larger than the 100nm
analyzed O-band, so the spectrum genuinely has only ONE broad interference maximum
in-band — but ~1e-4-scale FDTD numerical noise sitting on that maximum's own
near-flat top was enough to trip the naive detector into reporting it as TWO separate
peaks 3 grid points (~3nm) apart, producing a nonsensical `FSR=0.0031um` — wildly
inconsistent with (and, backwards from) `delta_L_um=10`/`15`'s own genuine,
theory-consistent FSR values (`0.0689um`/`0.0430um`, correctly *decreasing* with
increasing `delta_L_um`, as `FSR ≈ lambda^2/(n_g * delta_L)` predicts). Fixed by
switching to `scipy.signal.find_peaks(y, prominence=0.1*(max(y)-min(y)))` — a
prominence of 10% of the spectrum's own full dynamic range comfortably rejects the
noise-scale wobble while keeping every genuine fringe (whose prominence spans nearly
the full range). After the fix, `delta_L_um=5` correctly reports `n_peaks=1`,
`FSR=NaN` (not measurable — the true FSR exceeds the analyzed band, honestly
reported as undefined rather than a wrong number); `delta_L_um=10`/`15` are
unaffected (their peaks were always genuine). Section 11's design selection
(`n_peaks >= 2` among validated points, then max extinction) still selects
`delta_L_um=15` either way — that outcome happened not to depend on the bug this
time, but the bug was real and would not be safe to assume harmless in general (e.g.
a narrower analyzed band or a different `delta_L_um` sweep could easily have let a
spurious `n_peaks>=2` point win on extinction).

**Coupler-vs-gdsfactory permittivity comparison (Section 4, `07_mzi.ipynb`).** The
gdsfactory-derived geometry (added for the layout display) sweeps its cross-section
perpendicular to the local tangent — the more physically exact representation for a
curved waveguide — whereas the native construction's axis-aligned S-bend Blocks only
hold the *y*-extent constant (the same ~2%-at-most approximation noted above for
`coupler.n_seg`). The two are therefore expected to disagree at the sub-pixel level
right at the S-bend edges (confirmed: ~1-2% of pixels differ by a non-trivial amount,
concentrated exactly there), not because either construction is wrong, but because
they trace genuinely different polygons for the same nominal curve.
`simulate_baseline()` keeps using the native construction regardless, since every
validated result in this notebook and `06_directional_coupler.ipynb`'s was measured
against it — the gdsfactory path is for the layout view and this comparison only.

## `grating_coupler.py` (uniform partial-etch grating coupler)

The toolkit's first genuinely non-in-plane device: an x-z (propagation, vertical)
2D cross-section with a real vertical layer stack, rather than the x-y in-plane
2D effective-index convention every other component uses. See the module's own
docstring for the full rationale; this section covers the empirically-measured
settings.

| Parameter | Value | Note |
|---|---|---|
| `resolution` | 40 px/µm | Overridden from `GLOBAL_PARAMS`' 20 — sub-micron teeth/etch need it. Not yet re-checked against a higher resolution (see below). |
| `core_index` / `substrate_index` | 3.45 | Real bulk Si, not the toolkit's 2D effective 2.7 — see module docstring. |
| `si_thickness_um` / `box_thickness_um` / `clad_thickness_um` | 0.22 / 2.0 / 2.0 µm | Generic MPW-typical SOI values (no specific foundry PDK targeted, per user request). |
| `etch_depth_um` | 0.14 µm | Promoted from the sweep below (selected on O-band-mean efficiency). |
| `duty_cycle` | 0.4 | Promoted from the sweep below (selected on O-band-mean efficiency). |
| `period_um` | 0.5094 µm | Promoted; phase-matching estimate, NOT re-tuned by the sweep below (see caveat). |
| `fiber_angle_deg` | 10.0 | Within the commonly-cited 8–10° range; deliberately not gdsfactory's own stock 15° default. |
| `fiber_mfd_um` | 9.2 µm | Approximate SMF-28 mode-field diameter near the O-band center (~1.31µm), not the commonly-quoted 10.4µm C-band value. |

**Rotated-axis TE/TM measurement.** Every other module fixes the toolkit's
`NO_PARITY` bug with `eig_parity=mp.TE`, because their invariant Meep-z axis is
physical z and standard photonics "TE" (no vertical E component) is exactly
Meep's own `Ez(Meep)=0` definition of TE. This module's invariant Meep-z axis is
physical y instead (see module docstring), so standard photonics "TE" (E along
the in-plane lateral direction, i.e. along Meep's invariant axis) means a
NONZERO Meep-z component — Meep's own definition of TM, not TE. Directly measured
via `Simulation.get_eigenmode()` on this module's own 220nm-Si-slab cross-section
at the O-band center (1.35µm, `resolution=100` for this one-off MPB solve):

| Meep parity | band 1 n_eff | Dominant fields |
|---|---|---|
| `mp.TE` | 2.27 | Ey(Meep), Hz(Meep) — physically TM |
| `mp.TM` | **2.91** | Ez(Meep), Hy(Meep) — physically TE |

`n_eff=2.91` also seeded `DEFAULT_PARAMS["n_eff_grating_guess"]` (measured, not a
literature placeholder) for `derive_grating_period()`'s phase-matching estimate.
This module passes `eig_parity=mp.TM` everywhere — the opposite label from every
other module, for a physically consistent reason (see module docstring), not an
inconsistency.

**Fiber lateral-position choice in `coupling_efficiency_overlap`.** Using the
aperture's geometric center as the assumed fiber position measurably understates
achievable efficiency: at a first (`n_periods=12`) smoke-test point, the
geometric-center overlap measured 1.4%, the near-field's own intensity centroid
measured 1.5%, and a brute-force scan over candidate fiber positions found 2.7%
achievable at that same point — confirming the radiated near-field's energy
centroid is measurably offset from the aperture's geometric center (expected: a
uniform grating's per-period leakage rate isn't spatially uniform). The intensity
centroid (cheap, no per-point scan) is used as the Phase-1 compromise — closer to
the achievable ceiling than the geometric center, without the cost of a full scan.

**`coupling_efficiency_overlap()` formula fix — power-normalized reciprocity
overlap, replacing an unbounded shape-only overlap.** The original formula
normalized by `integral(|E_sim|^2 dx)` — an E-field-only SHAPE-match fidelity,
NOT a genuine power-normalized efficiency, and empirically found to exceed
the physical ceiling (`upward_power`, the actual measured fraction of
incident power radiated upward) by a wide margin: e.g. the (now superseded)
`etch_depth_um=0.16`/`duty_cycle=0.6` point reported `coupling_efficiency=
77%` at its own peak wavelength while `upward_power` there was only 38% —
physically impossible, since `coupling_efficiency` cannot exceed the
fraction of power that reaches the near-field plane at all. Full diagnostic
story (evanescent-contamination check ruled out, restricted-fiber-window
sensitivity test that isolated the bug to the metric's own normalization,
and a handedness bug in the FIRST fix attempt) recorded in
`docs/troubleshooting_log.md`.

**Every `coupling_efficiency` number in this section below predates this
fix** (computed with the old, unbounded shape-only formula) and is
**superseded** — kept as a historical record (per this file's own
"approaches tried and abandoned" purpose), not as a currently-trusted
result. The module now computes a standard E×H reciprocity/mode-overlap
integral (Snyder & Love; the same formula commercial mode solvers use),
requiring the near-field monitor to also capture `Hx` (physical Hx, no
axis-relabeling sign issue — only the physical-y-aligned component,
`Ez[Meep]`, needed a sign correction; see `coupling_efficiency_overlap()`'s
own "HANDEDNESS NOTE" docstring section for the full derivation):

$$
\eta(\lambda) = \frac{\left|\int\left[E_{y,sim}(x)H_{x,fiber}^{*}(x) + E_{y,fiber}^{*}(x)H_{x,sim}(x)\right]dx\right|^2}{8\,P_{sim}\,P_{fiber}}
$$

with $P_{sim}=\left|\mathrm{Re}\int E_{y,sim}H_{x,sim}^{*}dx\right|$,
$P_{fiber}=\left|\mathrm{Re}\int E_{y,fiber}H_{x,fiber}^{*}dx\right|$, and
$H_{x,fiber}=-n_{clad}\cos\theta\,E_{y,fiber}$ from the plane-wave Maxwell
relation $H=n(\hat{k}\times E)$ in Meep's own $c=1$ unit convention. This is
now properly bounded by construction (Cauchy-Schwarz on the true power
cross-term) — `coupling_efficiency` can no longer exceed `upward_power`.
The full 36-point sweep was re-measured with this corrected formula (pass 4
below); see that pass for the current, trusted numbers.

**Etch-depth x duty-cycle sweep (Section 5.3, `notebooks/09_grating_coupler.ipynb`).**
Four passes so far, each widening the grid and/or fixing a bug in the scoring
itself:

1. A 9-point grid (`etch_depth_um` ∈ {0.05, 0.07, 0.09} µm x `duty_cycle` ∈
   {0.3, 0.5, 0.7}), scored by PEAK `coupling_efficiency`. Promoted
   `etch_depth_um=0.09`/`duty_cycle=0.3`, peak 22.8%.
2. A 30-point grid (`etch_depth_um` ∈ {0.08, 0.10, 0.12, 0.14, 0.16} µm x
   `duty_cycle` ∈ {0.2, ..., 0.7}), still scored by PEAK. Promoted
   `etch_depth_um=0.16`/`duty_cycle=0.6`, peak 84.9% at `wl=1.3966µm`. Re-run
   at full precision: `passivity_check` passed (max total power fraction
   0.953); `energy_budget_check` did NOT (min total power fraction 0.814, vs.
   the 0.85 floor). **Investigating that shortfall found the real problem
   with a peak-only metric**: this point's spectrum turned out sharply
   asymmetric — a near-zero null almost exactly at the O-band CENTER
   wavelength (`fcen=1.35µm`: `coupling_efficiency=0.3%`, `reflection=62.6%`,
   a dense standing-wave pattern visible in the field snapshot, which is
   ALWAYS captured at `fcen`) and a 70–85% plateau confined to a narrow
   sub-band near 1.385–1.402µm. The 84.9% headline number was only true in
   that narrow sub-band, not representative of this design's O-band-average
   performance — this promoted point was subsequently found to be a poor
   practical design despite its high peak, motivating pass 3 below.
3. **A 36-point grid, `etch_depth_um` ∈ {0.08, 0.10, 0.12, 0.14, 0.16, 0.18} µm
   x `duty_cycle` ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7}, now scored by MEAN
   `coupling_efficiency` across the full analyzed O-band** (`grating_coupler.
   sweep_etch_duty()` now records both `coupling_efficiency_max` and
   `coupling_efficiency_mean` per point; `pic_toolkit.meep_sim.grating_coupler.
   simulate_baseline()` also gained an optional `field_snapshot_wl_um`
   argument, so a field snapshot can be captured at any specific wavelength,
   not only `fcen` — used below to inspect the promoted point at its own
   peak-efficiency wavelength instead of the band center). At `n_periods=22`,
   `resolution=40`, period held fixed at the phase-matching estimate
   (0.5094µm); all 36 points passed `passivity_check`.
   **⚠ SUPERSEDED by pass 4 below**: the table and analysis immediately
   following used `coupling_efficiency_overlap()`'s buggy, unbounded
   shape-only formula (see the fix note above and `docs/troubleshooting_log.md`)
   — every percentage in this pass 3 sub-section is too high and kept only as
   a historical record of the mean-vs-peak selection methodology, which
   pass 4 re-confirms.

| etch_depth_um | duty_cycle | mean coupling_efficiency | peak coupling_efficiency | peak wavelength (µm) |
|---|---|---|---|---|
| 0.14 | 0.4 | **54.50%** | 77.16% | 1.3475 |
| 0.16 | 0.5 | 50.57% | 81.23% | 1.3376 |
| 0.14 | 0.3 | 48.87% | 77.54% | 1.3303 |
| 0.12 | 0.3 | 45.13% | 76.79% | 1.3781 |
| 0.12 | 0.2 | 42.21% | 75.03% | 1.3601 |
| 0.14 | 0.5 | 40.41% | 80.30% | 1.3781 |
| 0.12 | 0.4 | 37.26% | 78.08% | 1.3939 |
| 0.18 | 0.6 | 36.27% | 84.03% | 1.3755 |
| 0.16 | 0.4 | 32.79% | 77.62% | 1.3017 |
| 0.18 | 0.5 | 31.40% | 84.00% | 1.3017 |
| 0.14 | 0.2 | 31.12% | 81.92% | 1.3064 |
| 0.16 | 0.6 | 26.54% | 84.89% | 1.3966 |
| 0.10 | 0.2 | 19.52% | 74.38% | 1.4020 |
| 0.16 | 0.3 | 17.38% | 42.29% | 1.3017 |
| 0.10 | 0.3 | 16.29% | 60.01% | 1.4020 |
| 0.12 | 0.5 | 14.85% | 69.54% | 1.4020 |
| 0.18 | 0.7 | 14.77% | 18.19% | 1.3087 |
| 0.16 | 0.7 | 14.51% | 16.11% | 1.3351 |
| 0.14 | 0.7 | 11.83% | 13.84% | 1.3652 |
| 0.14 | 0.6 | 11.17% | 51.32% | 1.4020 |
| 0.18 | 0.4 | 11.09% | 16.30% | 1.3017 |
| 0.10 | 0.4 | 9.24% | 39.53% | 1.4020 |
| 0.12 | 0.7 | 7.41% | 11.79% | 1.3860 |
| 0.18 | 0.3 | 7.11% | 11.81% | 1.3017 |
| 0.16 | 0.2 | 6.03% | 9.70% | 1.3017 |
| 0.12 | 0.6 | 4.93% | 7.87% | 1.3576 |
| 0.10 | 0.7 | 3.50% | 9.58% | 1.4020 |
| 0.10 | 0.6 | 3.04% | 5.72% | 1.3807 |
| 0.18 | 0.2 | 2.90% | 6.79% | 1.3017 |
| 0.08 | 0.3 | 2.57% | 6.02% | 1.3703 |
| 0.10 | 0.5 | 2.36% | 12.08% | 1.4020 |
| 0.08 | 0.2 | 2.06% | 8.36% | 1.4020 |
| 0.08 | 0.4 | 1.39% | 4.18% | 1.4020 |
| 0.08 | 0.6 | 1.25% | 3.70% | 1.3993 |
| 0.08 | 0.7 | 0.88% | 5.15% | 1.4020 |
| 0.08 | 0.5 | 0.58% | 0.96% | 1.4020 |

`etch_depth_um=0.14`/`duty_cycle=0.4` was promoted into `DEFAULT_PARAMS` --
highest mean (54.5%), a peak (77.2%) landing close to the true O-band center
(1.3475µm of 1.30–1.40µm, vs. the pass-2 point's peak sitting at the very
edge), a clean unimodal spectrum shape (rises smoothly from ~37% at 1.30µm to
77% at 1.3475µm and back down to ~32% at 1.40µm -- no null), and, unlike the
pass-2 point, BOTH checks pass at full precision: `passivity_check` (max total
power fraction 1.017) and `energy_budget_check` (min total power fraction
0.963, comfortably above the 0.85 floor). At the peak wavelength itself:
reflection=1.2%, upward=38.3%, downward=61.8% -- note `coupling_efficiency`
(77.2%) is NOT bounded by `upward_power` (38.3%) here, because
`coupling_efficiency_overlap()` is a normalized SHAPE-overlap fidelity
(Cauchy-Schwarz-bounded to [0,1] by construction, independent of the near
field's absolute amplitude), not an absolute-power-normalized quantity --
still the documented Phase-1 simplification noted in that function's own
docstring, not a new issue introduced by this sweep.

**Two open observations, not yet resolved (documented rather than hidden):**
1. **Neither the winning `etch_depth_um` (0.14) nor `duty_cycle` (0.4) sits at
   the edge of the grid actually tried** (the full 6x6 range spans
   0.08–0.18µm x 0.2–0.7) -- unlike both prior passes' promoted points, this
   one is an interior optimum, a mild positive signal that the grid resolution
   is adequate near this point (though still not a formal convergence check).
2. **Mean vs. peak can disagree sharply, and the biggest peak in the grid
   (0.16/0.6, 84.9%) has one of the WORST means (26.5%)** -- a reminder that
   `coupling_efficiency_max` alone is a poor design-selection metric for this
   device (see pass 2's history above); any future sweep here should keep
   recording and selecting on `coupling_efficiency_mean`, not revert to peak
   alone.

4. **Same 36-point grid, re-measured with the CORRECTED (power-normalized
   reciprocity) `coupling_efficiency_overlap()`** — see the fix note above
   and `docs/troubleshooting_log.md` for the full diagnostic story (a user
   noticing the field snapshot didn't visually look like a strong radiated
   beam, then three rounds of increasingly specific follow-up questions that
   isolated first a wavelength-capture issue, then ruled out evanescent
   contamination, then a literature comparison + restricted-window test that
   pinned the bug to the metric's own normalization, then a handedness bug
   in the first fix attempt). Same grid, same `n_periods=22`/`resolution=40`,
   period still held fixed; all 36 points again passed `passivity_check`.

| etch_depth_um | duty_cycle | mean coupling_efficiency | peak coupling_efficiency | peak wavelength (µm) |
|---|---|---|---|---|
| 0.14 | 0.4 | **28.58%** | 40.68% | 1.3475 |
| 0.16 | 0.5 | 26.11% | 42.16% | 1.3376 |
| 0.14 | 0.3 | 25.64% | 40.57% | 1.3303 |
| 0.12 | 0.3 | 23.73% | 40.58% | 1.3781 |
| 0.12 | 0.2 | 22.08% | 39.46% | 1.3601 |
| 0.14 | 0.5 | 21.07% | 42.04% | 1.3781 |
| 0.12 | 0.4 | 19.62% | 41.15% | 1.3939 |
| 0.18 | 0.6 | 18.56% | 42.67% | 1.3755 |
| 0.16 | 0.4 | 17.16% | 40.19% | 1.3017 |
| 0.14 | 0.2 | 16.24% | 42.33% | 1.3064 |
| 0.18 | 0.5 | 16.16% | 43.01% | 1.3017 |
| 0.16 | 0.6 | 13.70% | 43.30% | 1.3966 |
| 0.10 | 0.2 | 10.26% | 39.01% | 1.4020 |
| 0.16 | 0.3 | 9.18% | 22.20% | 1.3017 |
| 0.10 | 0.3 | 8.59% | 31.49% | 1.4020 |
| 0.18 | 0.7 | 7.97% | 9.86% | 1.3087 |
| 0.16 | 0.7 | 7.89% | 8.86% | 1.3376 |
| 0.12 | 0.5 | 7.78% | 36.44% | 1.4020 |
| 0.14 | 0.7 | 6.45% | 7.68% | 1.3626 |
| 0.14 | 0.6 | 5.87% | 26.56% | 1.4020 |
| 0.18 | 0.4 | 5.82% | 8.62% | 1.3017 |
| 0.10 | 0.4 | 4.86% | 20.82% | 1.4020 |
| 0.12 | 0.7 | 4.02% | 6.48% | 1.3860 |
| 0.18 | 0.3 | 3.77% | 6.31% | 1.3017 |
| 0.16 | 0.2 | 3.19% | 5.18% | 1.3017 |
| 0.12 | 0.6 | 2.60% | 4.16% | 1.3576 |
| 0.10 | 0.7 | 1.87% | 5.20% | 1.4020 |
| 0.10 | 0.6 | 1.59% | 3.01% | 1.3807 |
| 0.18 | 0.2 | 1.57% | 3.64% | 1.3017 |
| 0.08 | 0.3 | 1.37% | 3.21% | 1.3703 |
| 0.10 | 0.5 | 1.23% | 6.37% | 1.4020 |
| 0.08 | 0.2 | 1.10% | 4.40% | 1.4020 |
| 0.08 | 0.4 | 0.73% | 2.20% | 1.4020 |
| 0.08 | 0.6 | 0.65% | 1.94% | 1.3993 |
| 0.08 | 0.7 | 0.46% | 2.74% | 1.4020 |
| 0.08 | 0.5 | 0.30% | 0.51% | 1.4020 |

**`etch_depth_um=0.14`/`duty_cycle=0.4` is STILL the winner by mean
efficiency after the fix** — same grid point as pass 3's promotion, now with
honest numbers: mean 28.6% (vs. pass 3's invalid 54.5%), peak 40.7% at
1.3475µm (vs. pass 3's invalid 77.2%). At the peak wavelength: reflection=
1.2%, upward=38.3%, downward=61.8%, `coupling_efficiency=40.7%` — now
correctly close to (just 2.4 points above) the physical ceiling
`upward_power=38.3%`, the small residual consistent with the reciprocity
formula's own remaining approximations (paraxial/plane-wave `H_fiber` model,
finite-aperture window truncation), not a further bug. `passivity_check` /
`energy_budget_check` still both pass (max total power fraction 1.017, min
0.963) — unaffected by the fix, since those come from flux monitors that
never routed through the buggy formula.

**Comparison to literature**, now meaningful for the first time: Yang et al.
(Sci. Rep. 13:18112, 2023) report 23% for a 0°-normal-incidence uniform
grating coupler on the same 220nm-Si/2µm-BOX stack (single etch, no bottom
reflector), and 50% (2D-FDTD)/26% (measured) for their inverse-designed,
apodized version. This module's own design is UNIFORM (not apodized) but
TILTED (`fiber_angle_deg=10°`, vs. their 0°) — a tilt specifically chosen to
suppress the 2nd-order Bragg back-reflection that Yang et al. identify as
the dominant loss mechanism for 0°-normal uniform gratings. Our measured
reflection at the promoted point's peak (1.2%) is indeed far lower than a
0°-normal uniform grating's typical back-reflection, and our efficiency
(28.6% mean / 40.7% peak) now sits sensibly BETWEEN their uniform (23%) and
apodized (50%/26%) numbers — a physically coherent result: better than a
non-tilted uniform grating (tilt suppresses its main loss channel) but worse
than a genuinely apodized design (still a fixed leakage factor per period,
not shaped to match the Gaussian fiber mode). This ordering is the kind of
sanity check the pre-fix numbers (77%, exceeding even the apodized design's
measured 26%, from a UNIFORM grating) could never have passed.

**No resolution-convergence study yet.** `resolution=40` was carried over
unchanged from `bend.py`'s own already-converged choice (see that section
above), on the reasoning that this module's features (sub-micron teeth,
~100-200nm partial etch) are of comparable scale — not verified empirically for THIS
module's own geometry/monitors (in particular, the near-field overlap
integral's sensitivity to resolution near a sharp etched edge is untested).
Flagged as a documented gap, not asserted as converged.

**`simulate_fiber_incidence()` (reverse-direction, fiber -> waveguide) —
`GaussianBeam2DSource` is a poor fit to this grating's real near-field; a
literal time-reversal test is the trustworthy ground truth.** Full diagnostic
story in `docs/troubleshooting_log.md`; summary here. Sweeping the idealized
`GaussianBeam2DSource`'s position and tilt sign gave confusing, seemingly
contradictory results (a reversed tilt coupling 34-54x MORE strongly than the
"reciprocity-correct" tilt, at a fixed position) that flatly contradicted a
simple analytic momentum-matching derivation. Resolved by injecting the
ACTUAL simulated OUTcoupling near-field, complex-conjugated (= time reversal),
as a custom `mp.Source(amp_func=...)` at the same near-field plane: near-
perfect reconstruction of the -x guided mode (497,000:1 directional ratio),
confirming both time-reversal symmetry and `derive_grating_period()`'s design
(independently cross-checked via the OUTcoupling near-field's own measured
phase slope: 10.17° vs. the 10.0° design target). A follow-up test (idealized
Gaussian AMPLITUDE at the near-field's own concentration zone, `x≈2µm`,
combined with the REAL time-reversed PHASE) still reconstructed the correct
DIRECTION cleanly (6538:1) but only ~4% of the full magnitude — showing phase
fidelity governs direction, amplitude-shape fidelity governs magnitude.
**Practical implication:** `simulate_fiber_incidence()`'s `GaussianBeam2DSource`-
based `guided_mode_power_relative` should be read as a qualitative,
order-of-magnitude-uncertain signal, not just because of its known amplitude
mis-calibration (see the function's own docstring) but ALSO because a plain
tilted Gaussian is a genuinely poor model for what this specific grating's
mode actually looks like — matching real hardware practice (grating couplers
commonly use microlensed, not plain-cleaved, fibers, and are known to be
sensitive to fiber lateral position) for the same underlying reason.

## `models/mzi_arm.py` / `models/coupler.py` (SAX interpolation over a rotating phase)

**Real/imaginary-part linear interpolation silently underestimates magnitude when
phase winds fast between grid points.** `models/mzi.py`/`models/bend_topopt.py`
established the pattern every interpolated-artifact SAX model in this repo followed:
interpolate `S.real` and `S.imag` independently via `np.interp`. This is only safe
when the phase change between adjacent wavelength-grid points stays well under a
fraction of a full turn -- linearly interpolating real/imaginary parts across a
larger step cuts the chord across the true circular arc traced by a rotating complex
phasor, dipping toward (or through) zero magnitude between the actual sample points,
even though the true magnitude barely changes there.

**How this was found**: composing `models/mzi_arm.py` (delta_L_um=15, on top of a
~26um lead/bend span -- long enough that S21's phase winds by ~114deg between its
own artifact's `n_freq=21` wavelength samples, confirmed directly via
`np.diff(np.unwrap(np.angle(s21)))`) into `circuits/mzi_real_fdtd_sax.ipynb`'s SAX
circuit produced a spurious total-scattered-power deficit up to ~40% (`bar+cross+
refl+leak` should stay near 1, minus real per-component loss of a few percent) at
~1724/2000 swept wavelength points -- far too pervasive and periodic (dips spaced at
exactly the artifact's own 5nm grid spacing, confirmed by direct isolation: freezing
`mzi_arm`'s S-parameters to a constant mid-band value made the deficit vanish
entirely) to be genuine resonant-cavity physics from the components' own small
reflections. **Fix**: both `models/mzi_arm.py`'s and `models/coupler.py`'s
`_interp_complex` now interpolate magnitude and *unwrapped* phase separately instead
of real/imaginary parts -- confirmed to remove the deficit entirely (post-fix
`total4` stayed in `[0.946, 1.027]` across the same 2000-point sweep, zero points
below 0.9). `models/mzi.py`/`models/bend_topopt.py`/`models/racetrack.py` were not
touched (out of scope for this task, and not yet observed to hit this failure mode --
their own devices' path lengths may keep them in the safe regime), but any future
interpolated SAX model over a long/dispersive device should use magnitude+phase
interpolation from the start, not real/imaginary parts.

**Secondary finding, still under this same investigation**: even after the fix, the
composed real-FDTD MZI's measured FSR (~33nm) came out shorter than the simple
`FSR=wl0_um^2/(n_g*delta_L_um)` arm-only prediction (~44nm using the FDTD-fringe-fit
`n_g`, ~41.5nm using the MPB-dispersion `n_g`) -- plausibly because the directional
coupler's own ~26um physical path contributes real, measured wavelength-dependent
phase that the arm-only formula (and the ideal, dispersion-free couplers
`circuits/mzi_lattice.py` used) does not account for. Flagged as a real, physically
plausible effect of using genuinely measured components rather than idealized ones,
not re-investigated further here.

## `circuits/mzi_lattice.py` (`unitary_project` -- cascaded real-data circuits amplify per-component noise)

**Cascading 3+ real, FDTD-measured components in one `sax.circuit()` can violate
passivity outright, even when every individual component passes its own
validation.** Building a real (not ideal) N=4 coupled-lattice MZI from `models.
coupler.coupler` (4 real 50:50-ish couplers at different `coupling_length_um`) and
`models.mzi_arm.mzi_arm` produced total scattered power exceeding 1 by up to ~26% at
some wavelengths -- non-physical for any passive circuit. Isolated by testing with
`n=2,3,4` IDENTICAL, individually-well-validated 50:50 couplers (ruling out any one
component's own kappa or geometry as the cause): total power was ~0.98 at N=2 (fine,
consistent with real loss), ~1.01 at N=3, ~1.04 at N=4 -- growing with cascade depth,
present even in the simplest possible case.

**Root cause**: a component's own validation checks that each excitation ROW
satisfies `sum_j|S_ij|^2~=1` (energy conservation per port) -- necessary, but NOT
sufficient for the FULL S-matrix to be unitary. Row *orthogonality*
(`row_i . conj(row_j) = 0` for `i != j`) can still be measurably violated by ordinary
FDTD measurement noise without tripping any single-row energy check.
`sax.circuit()`'s exact circuit solver (not a naive per-stage power multiplication)
amplifies that latent non-orthogonality with each additional cascaded stage --
confirmed by testing `mzi_arm` in isolation (renormalizing just its per-row power
budget to sum to exactly 1 did NOT fix the cascading violation) vs. a full SVD-based
projection onto the nearest unitary matrix (`M = U*Sigma*Vh -> M_unitary = U*Vh`),
which DID fix it. `mzi_arm` (by far the largest individual imperfection of the two
components, ~6% reflection) was the dominant contributor: projecting it alone already
resolved the N=3/N=4 violation; projecting both `mzi_arm` and `coupler` gives the
cleanest result (N=4 total power stayed in `[0.82, 1.00]` across a full wavelength
sweep, vs. up to `1.26` unprojected).

**Fix**: `circuits/mzi_lattice.py::unitary_project(model_fn, port_names)` wraps any
SAX model function so its returned SDict (interpreted as a square matrix over
`port_names`, missing pairs defaulting to 0) is replaced by the nearest exactly-unitary
matrix at each wavelength point, before it's used in a cascaded circuit. Used in
`circuits/mzi_real_fdtd_sax.ipynb`'s N=2 and N=4 real-FDTD lattice sections. Not
applied retroactively to `models/coupler.py`/`models/mzi_arm.py` themselves -- those
stay as the honest, unmodified measured data; projection is a composition-time
concern, applied only where multiple real components are cascaded together.

## `circuits/mzi_lattice.py` (analytic SAX cascaded-MZI lattice filter)

| Parameter | Value | Note |
|---|---|---|
| `wl0_um` | 1.35 | Matches this toolkit's own characterization wavelength. |
| `n_eff0` | 2.4085 | Phase index at `wl0_um`, from `07_mzi.ipynb`'s MPB arm dispersion (Section 10-11). |
| `n_g0` | 2.763 | Group index at `wl0_um`, from `07_mzi.ipynb`'s FDTD S21 fringe-spacing fit (`N_G_FDTD_FIT`) — chosen over the MPB-dispersion group-index estimate (`~2.928`) because the FSR-fit value reflects the full round-trip through both couplers' S-bends, not just a straight arm segment. |
| `delta_L_um` | 15.0 | Design target itself (not derived from a fixed FSR) — chosen to land exactly on `07_mzi.ipynb`'s own already-simulated `delta_L_um` sweep point (`data/sparams/mzi/sweep/mzi_deltaL15.0.json`), per direct user request, so a later FDTD-based comparison can reuse that existing run. Gives `FSR≈43.97nm`, close to the ~40nm originally requested. |
| root-classification tolerance | `0.02` (magnitude), `0.05` (cluster) | See "Numerical tolerance" below. |
| self-check tolerance | `1e-3` | See "Numerical tolerance" below. |

**The binomial-target dead end.** The first version of `synthesize_maximally_flat_kappas`
targeted the standard Butterworth-style maximally-flat FIR shape,
`|A(phi)|^2=((1-cos(phi))/2)^(N-1)` — flat (many vanishing derivatives) at a single
point, `phi=pi`. This is the textbook "maximally flat" filter and the synthesis
(spectral factorization + layer-peeling, described in the module docstring) reproduced
it exactly (verified via forward-reconstruction to machine precision for N=2..5). But
numerically sweeping the resulting circuit showed its -1dB bandwidth **shrinks** with N
(N=2: 30.5% of FSR; N=4: 17.8%) — raising *any* smooth bump to a higher power makes it
*narrower*, not flatter, which is the opposite of what a WDM channel filter needs. This
was caught by directly measuring -1dB/-3dB bandwidth as a fraction of FSR before
trusting the result, not by inspecting the synthesis math alone.

**The fix: a halfband target, not a single-point-flat one.** A wide, flat *passband*
(not just an infinitesimally flat point) is a maximally-flat **halfband filter**
target: `|A(phi)|^2 = P_K(sin^2(phi/2))`, `P_K(x) = sum_{k=K}^{2K-1} C(2K-1,k) x^k
(1-x)^(2K-1-k)` (`N=2K` couplers) — flat at *both* `phi=0` and `phi=pi` while crossing
exactly 0.5 at `phi=pi/2` for every K, so every order shares the same -3dB crossover and
only the -1dB (near-flat-top) region widens with N. `P_1(x)=x` (trivial, N=2, identical
to the plain MZI already built in `07_mzi.ipynb`); `P_2(x)=3x^2-2x^3` (N=4, the classic
cubic "smoothstep" — independent confirmation this is the standard, correct
construction, not a bespoke one). Re-measured: N=4's -1dB width is 34.6% of FSR vs.
N=2's 30.5%, with near-matching -3dB crossovers (47.6% vs. 50.7%) — the intended result.
This synthesis only covers even `n_couplers` (`K=n_couplers/2`); an odd-order design
(e.g. Luceda's real 5-coupler filter) needs the more general non-halfband synthesis this
module does not implement, which is also why that design's real per-stage non-uniform
`delta_L` (see its own literature source) isn't reproduced here.

**A synthesized coupler can legitimately need `sign=-1`.** `ideal_coupler_model`'s
cross-port phase is `sign*i*sqrt(kappa)`. Forcing `sign=+1` at every stage (the naive
assumption) silently produces a WRONG design for `N>=3` — confirmed by hand-tracing N=3
of the (now-abandoned) binomial-target case, where the layer-peeling recursion, run with
the wrong fixed sign, produced kappas that looked plausible (all in `[0,1]`) but whose
forward-reconstruction did not match the target at all (residual ~1, not ~0). `N=2`
happens to only ever need `sign=+1`, which is exactly what let this bug hide during
initial testing — the fix (try both `sign=+1` and `sign=-1` at each peeling step, keep
whichever leaves the dropped/deflated coefficient nearest zero) was found by comparing
the deflation residual, not by inspecting the kappa values.

**A synthesized N=4 halfband design has an exact `kappa_1≈0`.** Not a bug: the second
coupler's power-coupling ratio in the exact synthesis is `~3e-7` (rounds to 0), meaning
this specific 4-coupler maximally-flat halfband design is, in effect, only 3 couplers'
worth of independent structure. Confirmed via the same forward-reconstruction self-check
as every other N (residual within tolerance) — an honest result of the exact math, kept
as-is rather than treated as suspicious.

**Numerical tolerance for spectral factorization.** `numpy.roots` is only accurate to
roughly `eps^(1/multiplicity)` near a high-multiplicity root — and the halfband targets
above have exactly that at `w=-1` (multiplicity 4 for N=4), giving root-position noise
of order `1e-4`, not machine precision. The root-vs-unit-circle classification tolerance
was loosened from an initial `1e-6` (fine for N=2's cleanly-separated real roots, too
tight for N=4's clustered near-degenerate ones — raised a spurious root-count-mismatch
error) to `0.02` (magnitude) / `0.05` (cluster-merge distance), and the
forward-reconstruction self-check tolerance from `1e-8` to `1e-3` to match — still tight
enough to catch a genuine synthesis bug (the sign-convention bug above produced
residuals of order 1, four orders of magnitude above this floor).

## `circuits/mux4_tree.py` (binary-tree Mux4 WDM demultiplexer)

> **Notebook consolidation note (post-hoc):** every notebook named below in
> this section and the next (`wdm_mux_mzi_lattice_sax.ipynb`,
> `mzi_real_fdtd_sax.ipynb`, `wdm_mux4_tree_sax.ipynb`,
> `mux4_tree_real_fdtd_sax.ipynb`, `mux4_tree_n4_real_fdtd_sax.ipynb`,
> `mux4_tree_n4_sax.ipynb`) has since been deleted and its content merged
> into exactly 2 final notebooks: **`circuits/wdm_sax.ipynb`** (the
> single-stage lattice, N=2 and N=4, ideal + real-FDTD + GDS) and
> **`circuits/wdm_mux4_sax.ipynb`** (the 2-stage Mux4 tree, same structure,
> GDS export N=4-only). All the numbers, findings, and derivations below
> remain accurate as history — they now live in those 2 files instead.

| Parameter | Value | Note |
|---|---|---|
| `delta_L_1_um` (stage 1) | 15.0 | Reuses `circuits/mzi_lattice.py`'s own reference sweep point unchanged — same value both `wdm_mux_mzi_lattice_sax.ipynb` and `mzi_real_fdtd_sax.ipynb` already use. |
| `delta_L_2_um` (stage 2, "down") | 7.5 | The only sibling of 15.0 already present in `07_mzi.ipynb`'s 6-point sweep (`[2.5,5,7.5,10,12.5,15]`) satisfying the standard binary-tree `delta_L_1 = 2·delta_L_2` FSR-halving relation. |
| `delta_L_2_up_um` (stage 2, "up") | `delta_L_2_um + quarter_wave_length_um(n_eff0, wl0_um)` ≈ 7.6401287 | New sweep point — see below. |
| `quarter_wave_length_um` | `wl0_um/(4·n_eff0)` ≈ 0.1401287 µm | Universal (base-`delta_L`-independent) constant; see derivation below. |

**This topology is a binary tree, not a lattice — do not confuse `mux4_tree.py` with
`mzi_lattice.py`'s own synthesis machinery.** A lattice (`circuits/mzi_lattice.py`,
`wdm_mux_mzi_lattice_sax.ipynb`) cascades N couplers sharing one `delta_L_um` to widen
a single flat-top passband. This tree instead cascades 3 *independent*, ordinary
50:50 MZIs (`kappas=[0.5,0.5]`, no lattice synthesis needed for any of them) in a
binary-split arrangement — stage 1 splits the input into an "up"/"down" branch, and
one stage-2 MZI per branch splits each again, giving 4 *simultaneous* channel
outputs, matching Luceda Photonics' `muxN` training reference
(academy.lucedaphotonics.com/training/topical_training/wdm_transmitter_mzi/muxn).

**Naively giving both stage-2 branches the same `delta_L_2` does not work.** Verified
numerically (built the actual ideal tree in SAX, both branches at `delta_L_2=7.5`, no
correction): the "down" branch (fed by stage 1's cross output) comes out clean, but
the "up" branch's two channels are capped at a **59.3% peak power** (degenerate double
peaks), not the ~35-85% (ideal) figures the corrected design reaches. Root cause: a
lossless coupler's cross port always picks up a ±90° phase relative to its own bar
port; stage 1's bar-output envelope reaches its own peak exactly where an
identically-parameterized stage 2 sits at its own ambiguous 50/50 crossover (not an
extremum), so the up branch's channel pair never resolves into two clean peaks.
Luceda's own reference design fixes this with a `center_wavelength` offset on
`stage_2_up` only — a free phase-tuning parameter their abstract model supports
directly. This toolkit's building blocks (`ideal_arm_model`, and per-artifact
`models.mzi.mzi_at_sweep_point`) expose only a physical arm-length difference, so the
fix here is instead a genuine additional length increment on `stage2_up`'s own arm:
setting the added differential phase to exactly a quarter-FSR (π/2),
`2π·n_eff0·Δ/wl0_um = π/2 ⟹ Δ = wl0_um/(4·n_eff0)`. Confirmed by direct numerical
sweep (all 4 sign/branch combinations tried; only "shift up only" gives all 4 channels
their full peak height simultaneously) — see `mux4_tree.py`'s own module docstring.
Ideal-model result (`wdm_mux4_tree_sax.ipynb`): total power stays unitary
(0.9999994-1.0000002 across 1.30-1.40µm); all 4 channels reach ≥0.9996 peak power at
≈1.3184/1.3396/1.3615/1.3843µm; extinction 35.1-84.7dB.

**First new Meep sweep point (`delta_L_um=7.6401287`, `stage2_up`) landed far short
of its π/2 target — a real, physically meaningful discrepancy, not noise.** Not
already present in `07_mzi.ipynb`'s 6-point sweep grid (spacing 2.5µm). Added via a
new run of `meep_sim.mzi.simulate_baseline` (same params/checks as that notebook's
own Section 10 sweep loop), saved to
`data/sparams/mzi/sweep/mzi_deltaL7.6401287108158605.npz/.json`. Measuring the real
S41 phase difference between this point and `stage2_down` (`delta_L=7.5`) directly
(`mux4_tree_real_fdtd_sax.ipynb` Section 3) gave only **+44.36°**, roughly half the
intended +90° — not the modest few-degree drift the `resolution=25` grid (~40nm
pixels, only ~3.5px for this ~140nm correction) and n_g-fit-vs-MPB dispersion
mismatch (~6%) would explain on their own.

**Root cause, confirmed by re-examining the existing 6-point sweep's own S41
phase**: the ideal linear model predicts `2π·n_eff0/wl0_um ≈ 11.21 rad/µm` of phase
accumulation, so the existing sweep's 2.5µm grid spacing is itself several full 2π
turns per step — every existing point is badly phase-aliased against its neighbors
(local finite-difference slopes between consecutive swept points flip sign
essentially at random: `+0.80, -0.43, -0.73, +0.78, -0.32 rad/µm`), so that grid
cannot calibrate anything finer than itself. But the ONE fine-spaced pair available
(`7.5` vs `7.6401287`, only `0.1401µm` apart, well inside one fringe) gives a real,
unaliased local slope of **5.5249 rad/µm** — only about half the ideal linear-model
slope. This is a genuine effect of the real arm geometry, not a bug: `mzi_arm`'s
raised-cosine delay bump changes SHAPE (bend curvature, not just arc length) as
`delta_L` grows, so the actual phase-vs-`delta_L` relationship near any operating
point is not simply `2π·n_eff0·Δ/wl0_um` — that formula is only exact for a
straight, dispersionless ADDED length, which this jog geometry is not.

**Second point, calibrated from that measured slope rather than the ideal
formula**: `delta_L_um = 7.5 + (π/2)/5.5249 ≈ 7.784312173071006`. If this lands
close to the intended +90° real phase difference (checked the same way, Section 3 of
the real-FDTD notebook), it supersedes the first point as `stage2_up`'s design value
throughout `mux4_tree_real_fdtd_sax.ipynb`; the first point is kept on disk (not
deleted) as the artifact underlying this very discovery. Per the user's own explicit
allowance ("追加sweepしてOkです" — additional sweeps are fine if the arm ΔL
calculation proves insufficient), this second point was run rather than accepting
the first point's badly-degraded (~3dB) channel extinction as final.

**Confirmed the ΔL geometry itself is realized exactly — the nonlinearity is
physical, not a construction bug.** Checked directly (not assumed): `mzi.py`'s
`solve_delay_arm` bisection realizes the requested `delta_L_um` as continuous
raised-cosine arc length to machine precision (error `0.0`–`2e-15`), and the
actual discretized geometry Meep receives (`bump_n_seg=200` chord-length
approximation) matches the requested value to within `~5e-4`–`1e-3µm` — three to
four orders of magnitude smaller than the effect being investigated. A broader
self-consistency check across all 8 known `mzi.py` sweep points (using the
measured fine-pair slope, `5.5249 rad/µm`, to predict the OTHER, 2.5µm-spaced
points' own raw phase) failed (`23°`–`148°` residuals) — ruling out "a single
constant effective index, different from `n_eff0`" as the explanation. The real
cause: the ideal `2π·n_eff0·Δ/wl0_um` formula assumes a straight, dispersionless
ADDED length, but `mzi.py`'s raised-cosine bump's curvature itself grows
nonlinearly with `delta_L_um` (amplitude scales notably faster than length near
this arm's operating range), so bend-induced effective-index shifts vary with
`delta_L_um` in a way no single corrected linear slope can capture globally —
only locally, over a small enough neighborhood (which is exactly why the second,
locally-calibrated point above still carried an ~8.86° residual, not zero).

**UPDATE 2 (superseded by UPDATE 3 below — flagged the need for a redo, but
the redo itself is in UPDATE 3).** Update 1 below (the `-11.2120 rad/µm` slope, the near-total
cross-port null found at `stage2_down`'s own `delta_L_um=7.5`, and the 6-point
scan of `delta_L_um∈[7.0,8.0]` that found no clean ~90°-separated healthy pair)
was entirely measured at `resolution=25` — before `resolution=25` itself was
found substantially under-converged for this exact delay-arm bump geometry (see
this file's `coupler.py`/`mzi.py` section, "`resolution=25`... was itself found to
be substantially under-converged"). Whether the null at `delta_L_um=7.5`, the sign
reversal, and the apparently chaotic point-to-point swings are genuine features of
this arm's phase response or were themselves resolution artifacts is now an open
question — not yet re-checked at `resolution=40`. Given `07_mzi.ipynb`'s own
`resolution=40` re-run also changed which `delta_L_um` gets auto-selected as
"best" (7.5 → 12.5, see the `coupler.py`/`mzi.py` section), this entire
stage2_down/stage2_up recalibration for `circuits/wdm_mux4_sax.ipynb` needs
re-doing from scratch against `resolution=40` data, not just re-verified.

**UPDATE 3 (RESOLVED — the null and the chaotic swings were entirely
`resolution=25` artifacts; the fix at `resolution=40` turned out to need no
new FDTD at all).** Re-checked `stage2_down`'s own `delta_L_um=7.5` in
`07_mzi.ipynb`'s fresh `resolution=40` sweep: `|S41|^2=0.379` — a completely
healthy value, nowhere near the old `resolution=25` near-null (`0.0005`).
Scanning the same 6-point sweep directly for a healthy partner near a `±90°`
offset (`build_mux4_tree_circuit`'s topology only needs the two stage2
instances' transmission ripples offset from each other — a `-90°` real
measured offset works exactly as well as `+90°`, just as a mirror) found one
immediately: `delta_L_um=10.0` (`|S41|^2=0.505`, phase offset from
`stage2_down` measured at `-94.69°`, only `4.69°` from the nearest `±90°`
target — versus UPDATE 1's `-123.6°` discrepancy at `resolution=25`).
Verified directly by building the actual N=2 real circuit
(`circuits/wdm_mux4_sax.ipynb`, re-run clean, 0 errors): total power stays
`0.983-1.000` across the band, and the 4 channel peaks land far more evenly
spaced (std `2.4nm`) than the next-nearest sweep-point alternative,
`delta_L_um=5.0` (std `11.3nm`, one 42nm gap) — confirming `10.0` is a
genuinely good, not just numerically-closest, choice. `stage2_up` is now
`delta_L_um=10.0` (an *existing* sweep point, not a bespoke calibrated
value) — no new FDTD simulation needed, no local-slope fitting, no
near-null reference. The whole UPDATE 1/2 local-calibration approach
(measure a fine-spaced neighbor pair, fit a slope, solve for a bespoke
length) is now understood to have been fighting a resolution artifact from
the start; the real fix was coarser and simpler once the actual physics was
resolved properly.

**UPDATE 1 (superseded — kept as history of what was tried under
`resolution=25`).** After `mzi.py`'s TE/TM `eig_parity` fix — see that
module's docstring and this file's `coupler.py`/`mzi.py` section — the entire
calibration above was re-measured, and the recalibrated value changed materially. Everything in this
subsection up to here describes the calibration as originally derived under the
pre-fix physics (`eig_parity=mp.NO_PARITY`, effectively TM); kept as history, not
current. Re-running the same procedure (same two-point local-slope method, same
`delta_L_um=7.5`/`7.6401287108158605` pair, now simulated under genuine
`eig_parity=mp.TE`) gives a materially different result: the measured local slope
is now **-11.2120 rad/µm** — its MAGNITUDE now closely matches the ideal
linear-model slope (`11.2097 rad/µm`, essentially exact agreement, unlike the old
~2x discrepancy), but its SIGN is reversed relative to the naive expectation that
phase should increase with `delta_L_um`. Solving for the length giving exactly
π/2 under this measured slope gives `delta_L_um = 7.5 + (π/2)/(-11.2120) ≈
7.359900150236218` (SHORTER than `stage2_down`, unlike the old calibration's
longer value) — simulated and saved
(`data/sparams/mzi/sweep/mzi_deltaL7.359900150236218.npz/.json`, energy/
reciprocity/passivity all pass). `circuits/wdm_mux4_sax.ipynb`'s
`DELTA_L_2_UP_UM_N2_CAL` was updated to this new value. The sign reversal itself
is not further investigated here -- flagged as a genuine, measured, physically
real finding (not assumed away), consistent with the already-established fact
that this arm's raised-cosine bump does not follow the straight-line linear
phase model globally. The old point (`delta_L_um=7.784312173071006`) and its own
artifact are kept on disk, unused, for provenance.

**Does a flatter per-stage passband (N=4 couplers, not N=2) recover the lost
extinction, given the SAME phase-calibration residual?** Quantitatively confirmed
in the ideal SAX model (inject the exact measured residuals, `-8.86°`/`-45.64°`,
into `stage2_up`'s correction and measure worst-channel extinction, N=2 vs. N=4
via `build_ideal_mux4_tree_circuit`'s new `n_couplers` parameter):

| N (couplers/stage) | phase error | ch1 ext (dB) | ch2 ext (dB) | worst (dB) |
|---|---|---|---|---|
| 2 | 0° | 44.1 | 35.1 | 35.1 |
| 2 | -8.86° | 23.5 | 26.8 | 23.5 |
| 2 | -45.64° | 9.6 | 10.0 | 9.6 |
| 4 | 0° | 75.6 | 66.4 | 66.4 |
| 4 | -8.86° | 44.3 | 50.4 | 44.3 |
| 4 | -45.64° | 17.2 | 17.8 | 17.2 |

At every error level, N=4 beats N=2 by 8–31dB of worst-channel extinction — a
flat-top design's transmission derivative near its own crossover is much smaller
than a plain sinusoidal MZI's, so the same absolute phase error costs it far less
extinction. This motivated `circuits/mux4_tree_n4_real_fdtd_sax.ipynb`, a real-FDTD
rebuild of all 3 tree stages as N=4 lattices, per the user's approval to add
whatever new sweeps this required.

**A second, geometrically distinct real delay arm was needed for the N=4 rebuild.**
`models.mzi_arm` (4-Euler-bend jog, `03_mzi_arm.ipynb`) has a `delta_L_um` floor of
`13.272µm` at its own default `radius_um=5.0` — well above `stage2_down`/
`stage2_up`'s targets (`7.5`/`7.6401287µm`), geometrically unreachable there.
`radius_um=2.5` lowers the floor to `6.632µm` (comfortable margin below both
targets), while staying clear of the euler-vs-circular crossover (`~1.5–1.75µm`,
`bend.py` section above) where Euler stops being the better choice. Two new points
were simulated at this radius (`data/sparams/mzi_arm/sweep_r2.5/`,
`extra_straight_um=0.434`/`0.5040643554079294` for `delta_L_um=7.5`/`7.6401287`
respectively); `stage1` reuses the existing `radius_um=5.0`, `delta_L_um=15.0`
baseline design point unchanged (no new simulation needed there).

**This arm's phase-vs-length relationship is much closer to linear than `mzi.py`'s
bump — confirmed BEFORE committing to new simulations, not assumed.** Unlike the
bump (whose curvature itself changes shape with `delta_L_um`), `mzi_arm`'s 4 bends
have FIXED curvature at a given `radius_um`; only a straight segment's length
changes with `delta_L_um`. A self-consistency check across the existing
`radius_um=5.0` sweep (6 points, `delta_L_um=14`–`30µm`) found the SAME implied
slope correction (`+4.946°/µm`, i.e. `+0.77%` of the ideal `11.2097 rad/µm`) from
BOTH its 2.5µm-spaced and 5µm-spaced point pairs — internally consistent in a way
`mzi.py`'s own sweep never was, strong evidence this arm's phase-length relation
genuinely is linear (to within a small, constant correction, plausibly just
`n_eff0` itself being fit ~1% off at this exact cross-section). This is why the
N=4 rebuild's `stage2_up` uses the ORIGINAL ideal quarter-wave target
(`delta_L_um=7.6401287108158605`) directly, without the iterative recalibration
`mzi.py`'s bump arm needed.

**Confirmed: the near-linear arm needed no recalibration, and N=4 substantially
recovered real extinction.** `circuits/mux4_tree_n4_real_fdtd_sax.ipynb` measured
the new arm's real phase difference directly (same check as the N=2 notebook):
**+89.06° vs. the +90° target, an -0.94° residual** — an order of magnitude
tighter than the bump arm's best achieved result (-8.86°), confirming the
linearity prediction above without needing a second iteration. Real per-channel
extinction, N=2-per-stage vs. N=4-per-stage (both measured, not predicted):

| channel | N=2 real ext (dB) | N=4 real ext (dB) |
|---|---|---|
| ch1 | 7.3 | 11.6 |
| ch2 | 1.7 | 10.1 |
| ch3 | 9.9 | 16.3 |
| ch4 | 4.6 | 21.6 |

Every channel improved; the worst channel (ch2) improved by ~8.4dB. Still well
short of the ideal-model SAX prediction (~44dB at this same residual) — the real
N=4 lattice cascades 4 real couplers + 3 real arms per stage (vs. N=2's single
whole-MZI measurement or 2 couplers + 1 arm), so each stage now carries
substantially more accumulated real loss (`stage2_up`'s own bar+cross sum at
`wl0` measured `0.7448`, vs. `stage1`'s `0.9234` and `stage2_down`'s `0.9999` —
consistent with the achieved coupler kappas, `[0.9544, 0.0459, 0.6922, 0.5072]`,
deviating non-trivially from the ideal target `[0.9328, 0, 0.7503, 0.5006]` on
top of each component's own measured loss) on top of whatever residual
phase-calibration error remains. This is the expected, honest outcome — N=4 is a
real, substantial improvement over N=2 at the SAME calibration accuracy, not a
route to ideal-model performance.

**Consolidated into one N=4-only notebook, `circuits/mux4_tree_n4_sax.ipynb`.**
Per the user's decision to proceed with the N=4-per-stage design exclusively,
this single notebook now combines the ideal SAX model, the real-FDTD model +
ideal-vs-real comparison, and a full GDSfactory physical layout of the
assembled 3-stage tree — the working design going forward. The two N=2
notebooks and `mux4_tree_n4_real_fdtd_sax.ipynb` are kept unmodified as the
historical record of the investigation that motivated N=4.

**The GDS layout needed a genuinely new technique: branching, not series,
placement.** Every prior GDS-export cell in this repo (`mzi_real_fdtd_sax.
ipynb`'s `build_lattice_gds`, N=2 and N=4) only ever places coupler stages in
series along one straight line. This tree's actual branch point — `stage1`'s
two outputs feeding two separate, vertically-offset downstream lattice
devices — has no precedent here, and `gf.routing` (gdsfactory's auto-routing
API) was not used anywhere in this repo before this notebook.
`gf.routing.route_single_sbend(component, port1, port2)` (gdsfactory 8.32.2)
was the natural fit: a smooth S-bend between two arbitrary-offset,
opposite-facing ports, exactly the branch connection needed, with no custom
geometry math. Each per-stage lattice is placed via the existing
`build_lattice_gds` pattern (generalized to take an explicit arm-geometry +
reference-gap pair per stage, since `stage1`/`radius_um=5.0` and `stage2_up`,
`stage2_down`/`radius_um=2.5` have different fixed spans); `stage2_up`/
`stage2_down` are offset in y by half of `stage1`'s own measured `.bbox()`
height plus half of the downstream stage's own height plus a margin, not a
guessed constant.

**`route_single_sbend`'s strict port-orientation check needed a small,
justified snap.** It requires `port1`/`port2` orientations to differ by
EXACTLY 180° (tolerance 0.1°) — measured ports here differed by ~180.15°,
just over that tolerance. Root cause: `mzi_arm`'s own internal geometry chains
8 `.connect()` calls (2 leads + 4 bends + 2 straight segments), and each
accumulates a small KLayout fixed-point rounding error; by the time that
component is placed and its outer ports read back, ~0.08–0.15° of drift has
accumulated — physically meaningless (a sub-nanometer-scale effect at these
port sizes) but enough to trip a strict equality-style check. Fixed by
snapping each port's `orientation` to the nearest multiple of 90° immediately
before calling `route_single_sbend`. One subtlety: `component.ports["name"]`
returns a FRESH `Port` view on each access in this gdsfactory version —
mutating a re-fetched port has no lasting effect, so the snapped port object
must be captured once and reused for the actual routing call, not looked up
again by name.

## `optimization/arm_library.py` / `optimization/coupler_library.py` (length-axis interpolation for adjoint optimization)

**Interpolating S-parameters over a geometric length axis needs the same
magnitude+unwrapped-phase treatment as the wavelength axis (see this file's
`models/mzi_arm.py` / `models/coupler.py` section above), plus an analytic
propagation-trend removal step before unwrapping is even meaningful.**
`arm_library.py`'s `delta_L_um` axis and `coupler_library.py`'s
`coupling_length_um` axis both wind far too fast to unwrap directly across a
sparse, irregularly-spaced grid (confirmed: ~640deg/um for the arm). Fix:
subtract an analytic propagation trend (`k * 2*pi*n_eff*length_um/wl`, `k`
chosen automatically per S-entry from a small candidate set by minimizing
the worst residual step) before unwrapping across the length axis, and add
it back at query time. This mirrors `models/mzi_arm.py`'s own wavelength-axis
`_interp_complex` fix, just applied to a different axis.

**Coupler through/cross phase needs `k=1`; reflection terms need `k=2`; the
initial assumption that couplers needed no de-winding at all was wrong.**
Because the coupler's cross-vs-through *relative* phase stays pinned within
0.5° of +90° across the entire measured 2–26um sweep (a fixed property of
the coupled-mode coupling mechanism, not something length tuning can move —
see this file's "`circuits/` MUX2 (N=4 lattice) geometry-optimization
program" section below), a first version of `coupler_library.py` assumed no
length-axis winding needed correcting at all. Its own leave-one-out fit-quality check immediately
caught ~180deg phase errors: each S-entry's own *absolute* phase (not the
relative through/cross difference) still winds substantially with coupling
length, for the same physical reason the arm's phase winds with `delta_L_um`
— more coupling length is still more propagation length. The candidate-`k`
search picks `k=1` for `S_through`/`S_cross` (one-way propagation) and `k=2`
for `S_reflect` (round-trip), both using the same fitted-waveguide `n_eff`
from `data/design_points/waveguide.yaml` as the trend estimate (accurate
enough to de-wind by, despite the coupling region's own supermode index
technically differing from an isolated waveguide's).

**The `coupler_selected` 50:50 design-point artifact (13.848um) is excluded
from the length-axis library.** Direct comparison against its 14.0um
sweep-grid neighbor (only 0.15um away) showed a ~96° `S_through` phase
offset at a fixed wavelength — implausibly large for two such similar
lengths if both artifacts shared one consistent phase reference. Root cause
is a different monitor/reference-plane placement between the two
independent simulation runs (`06_directional_coupler.ipynb`'s Section-10
sweep vs. its own separate Section-11 50:50 re-simulation), not a real
physical discontinuity. The library uses only the uniform 7-point sweep
(`data/sparams/coupler/sweep/`), all from one consistent run.

**`S_through` has a genuine physical null (a true zero-crossing, not
simulation noise) near `coupling_length_um=26`, `wl≈1.392–1.393um`.**
`|S_through|` there drops to ~0.002 then recovers on either side —
consistent with the coupled-mode `cos(kappa*Lc)` transmission term crossing
zero at this length/wavelength combination. Phase is inherently ill-defined
at a magnitude null regardless of interpolation method; `coupler_library.py`
excludes wavelength bins where either endpoint of a length-step has
magnitude below a small threshold from the de-winding *quality check* only
(not from the interpolation table itself) — their negligible magnitude
means their phase barely affects any reconstructed value, and the affected
band sits well outside every channel wavelength this project's objective
actually samples (nearest channel target is 1.384um, whose own ±5nm
evaluation window ends at 1.389um).

## `circuits/` MUX2 (N=4 lattice) geometry-optimization program (2026-09-07 to 2026-09-14, paused/abandoned)

Eleven exploratory notebooks in `circuits/` (`wdm_stage1_adjoint_optimization.ipynb`,
`wdm_stage2_adjoint_optimization.ipynb`, `wdm_stage2_with_optimized_stage1.ipynb`,
`wdm_sax_optimization_v2.ipynb`, `wdm_sax_optimization_v2_fdtd_validation.ipynb`,
`wdm_component_sparam_diagnostic.ipynb`, `wdm_mux2_coupler_dispersion_diagnosis.ipynb`,
`wdm_mux2_optimization_dispersion_corrected.ipynb`, `wdm_mux2_optimization_2d_coupler.ipynb`,
`wdm_mux2_optimization_2d_coupler_full_spectrum.ipynb`,
`wdm_mux2_greedy_stepwise_redesign.ipynb`) tried to close the ideal-vs-real
gap `wdm_sax.ipynb`'s N=4 lattice shows in FDTD, using the JAX-differentiable
surrogate in `src/pic_toolkit/optimization/` (`circuit_diff.py`, `objective.py`,
`arm_library.py`, `coupler_library.py`, `lbfgs.py`, `adam.py`/`adam_nd.py`).
This work is paused with no optimized design adopted into production —
`wdm_sax.ipynb`/`wdm_mux4_sax.ipynb` still ship the original
`CHOSEN_LC=[26.0, 2.0, 18.0, 14.0]`, `gap_um=0.20` baseline. This entry
consolidates what the eleven notebooks found before they were removed; the
underlying measured data they produced (see "What's still on disk" below)
was left in place.

**Coupler stage 3's sign mismatch is a topological ceiling, not a tuning
problem.** The maximally-flat N=4 synthesis needs `signs=[+1,+1,+1,-1]`, but
every real directional coupler this platform has ever characterized (the
full measured 2–26um length range) shows the same `+1`-like cross/through
relative phase — `sin(kappa*L)`'s sign only flips well past the
"overcoupled" turnover point, far outside any length in this project's
libraries. No amount of arm- or coupler-length tuning, including letting
each arm-pair's `delta_L` vary independently (an 8-parameter variant that
was tried and made the objective *worse*, 1.609 vs. 0.656), reaches it. It
would need either new FDTD data deep in the overcoupled regime or a
non-mirror-symmetric coupler geometry — out of scope for a meep-free
circuits notebook.

**L-BFGS-B (quasi-Newton) clearly beats Adam for this problem class**: exact
JAX gradients through a pure-interpolation surrogate, smooth and
low-dimensional (6–8 scalars) — exactly quasi-Newton's regime. Head-to-head
from the same start, L-BFGS-B reached a meaningfully better optimum
(objective 0.656 vs. Adam's 1.524) in ~5x fewer iterations (45 steps vs.
250+). Every later notebook in this program used L-BFGS-B with multi-start,
not Adam.

**The decisive, repeated bug: every early optimization round targeted
channel-center wavelengths computed from the fully idealized (dispersion-
free) analytic model — a target the real, dispersive device could never
actually reach well.** A component-level complex-S-parameter diagnostic
(`wdm_component_sparam_diagnostic.ipynb`) confirmed real couplers are
genuinely, substantially dispersive (`ideal_coupler_model`'s flat-`kappa`
assumption is quantitatively wrong, not just a simplification — e.g. at
`Lc=18um`, `kappa(lambda)` swings ~28 percentage points across the band), on
top of the already-known coupler-3 sign ceiling. A 5-case component-swap
ablation (`wdm_mux2_coupler_dispersion_diagnosis.ipynb`,
`wdm_mux2_greedy_stepwise_redesign.ipynb`) then traced this project's whole
~21.5nm ideal-vs-real channel-wavelength offset almost entirely to coupler
dispersion alone (+21.4nm of the +21.5nm total) — arm/bend dispersion
matters in isolation (+8.3nm) but its marginal contribution once coupler
dispersion is already present nearly vanishes, correcting `wdm_sax.ipynb`'s
own original framing that blamed the two roughly equally. Fixing only the
optimization *target* — freezing `lambda_1`/`lambda_2` from a reference
circuit built with real, FDTD-measured coupler S-parameters plus a
bend-dispersion-corrected arm model, with the forward surrogate and
optimizer left unchanged — immediately produced a real, FDTD-validated
improvement: channel-1 transmission 54.6%→89.9%, crosstalk 38.7%→5.2%
(`wdm_mux2_optimization_dispersion_corrected.ipynb`).

**The best point-sampled result of the whole program**: extending the free
variables from coupler length only to a full 2D (gap × length) search
against that same corrected target reached 97.1% transmission, 0.71%
crosstalk, 0.13dB insertion loss, +0.4nm wavelength error
(`wdm_mux2_optimization_2d_coupler.ipynb`), validated against an
iteratively-densified 18-gap 2D interpolation library. An earlier attempt at
this same experiment against only 6 characterized gaps looked comparably
good in the surrogate (objective 0.137) but validated *worse* than the
length-only baseline in real FDTD (79.9% transmission, 16.3% crosstalk) —
traced to the sparse gap axis's own leave-one-out fit quality being too
poor (22% magnitude / 109deg phase error) to trust the optimizer's chosen
point. Lesson: densify and leave-one-out-validate an interpolation axis
*before* trusting an optimizer that can freely explore it.

**A stricter whole-spectrum shape-fidelity objective reintroduced the exact
same target-anchoring bug and failed outright**
(`wdm_mux2_optimization_2d_coupler_full_spectrum.ipynb`): its RMS-deviation
target was again the plain idealized (dispersion-free) spectrum, so the
optimizer faithfully found a design with a 46%-better whole-spectrum RMS
score against that *wrong* target — while its actual usable channel peak
landed 17.7nm off, transmission collapsed to 37.9%, and crosstalk rose to
60.6%. An objective's target definition matters at least as much as its
loss shape, and this project got bitten by the same mistake twice.

**Greedy, one-component-at-a-time coupler/arm selection (picking each
coupler's best real `(gap_um, Lc_um)` directly off the raw 2D-sweep data,
then the arm, no joint gradient optimization at all) is fast and directly
fixes wavelength alignment by construction** — best-in-program wavelength
error (−0.2nm) — **but does not dominate the joint 2D optimum**: it
underperforms on point transmission/crosstalk (89.3%/2.81% vs. 97.1%/0.71%)
and its whole-spectrum shape fidelity is actually *worse* than the
unoptimized production baseline, not better. Matching each coupler's own
kappa and local dispersion flatness in isolation says nothing about the
cascade's overall passband bandwidth (3.2nm vs. the joint optimum's
19.5nm) — a whole-circuit interaction only a joint (not greedy/local)
optimizer accounts for. Greedy and joint optimization solve different
problems; neither dominates the other.

**What's still on disk.** The differentiable surrogate/objective/optimizer
modules remain at `src/pic_toolkit/optimization/` and the FDTD-validated
artifacts each round produced remain cached under
`data/sparams/mzi_arm/optimized/` and `data/sparams/coupler/optimized/`
(`_v2` through `_v7`, plus `_up`/`_down`/`_with_couplers` variants for the
MUX4-tree stage-2 thread) and `data/sparams/mzi_arm/optimization_history*.csv`
— none of it is referenced by any notebook still in `circuits/` as of this
entry. If this program is resumed, `wdm_mux2_optimization_2d_coupler.ipynb`'s
2D (gap × length) joint result against the dispersion-corrected target is
the strongest point-performance candidate on record, but the whole-spectrum
and greedy experiments above show a real, unresolved point-performance-vs-
bandwidth tradeoff that would need a properly-scoped objective before
adopting anything into production.
