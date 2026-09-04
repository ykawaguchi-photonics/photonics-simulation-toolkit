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

## `coupler.py` / `mzi.py` (directional coupler, passive Mach-Zehnder interferometer)

These two modules are the toolkit's first genuinely 4-port devices. `coupler.py`
builds a directional coupler (two waveguides S-bent from a wide separation down to a
tight coupling gap); `mzi.py` reuses that same, already-validated 50:50 design for
both stages of a passive Mach-Zehnder interferometer, joined by a straight reference
arm and a raised-cosine-bump delay arm carrying the extra path length `delta_L_um`.

| Parameter | Value | Note |
|---|---|---|
| `coupler.coupling_length_um` | 13.848 µm | The 50:50 point, re-measured after the TE/TM fix below (widened-sweep re-check landed at 13.848 µm, a ~0.03 µm shift from the earlier 13.873 µm — see below). |
| `coupler.gap_um` | 0.2 µm | Matches `racetrack.py`'s already-validated choice at `resolution=25`. |
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
