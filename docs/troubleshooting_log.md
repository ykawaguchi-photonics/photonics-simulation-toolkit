# Troubleshooting Log

A chronological record of *process*-level problems I hit while building,
revising, and validating this toolkit's notebooks -- environment quirks,
workflow mistakes, and fitting/validation pitfalls that aren't specific to
any one component's physics (those live in
`docs/simulation_settings_record.md` instead). Each entry: what went wrong,
how I noticed it, and how I fixed it -- so I don't repeat the same mistake,
or at least recognize it faster, next time.

## A toolkit-wide default silently forked into two hardcoded values across newer modules

**Symptom:** comparing notebooks side by side, I noticed two different
waveguide widths in use: `waveguide.py`/`bend.py`/`grating_coupler.py` used
`wg_width_um=0.45` (inherited from the shared `GLOBAL_PARAMS` in
`src/pic_toolkit/params.py`), while `racetrack.py`/`coupler.py`/`mzi.py`/
`bend_topopt.py` each independently hardcoded `0.5` in their own standalone
`DEFAULT_PARAMS` dicts that never imported `GLOBAL_PARAMS` at all. Two more
unexecuted modules, `spiral.py`/`spiral_gds.py`, had also independently
hardcoded `0.45`. Nothing in the codebase cross-checks component parameters
against each other, so this went undetected through several components' worth
of otherwise-careful validation.

**Root cause:** `params.py`'s own module docstring already names this exact
failure mode ("values that drifted apart across components without a physical
reason... are bugs, not intentional per-component choices; this module is the
single place to fix that going forward") -- but I had written the four newer
modules with fully standalone `DEFAULT_PARAMS` instead of starting from
`{**GLOBAL_PARAMS, ...}`, bypassing the single source of truth the docstring
assumes every module uses.

**Fix:** I converged on `0.5` (see `docs/simulation_settings_record.md`'s
"Shared: width unification" note for the cost-based rationale -- 4 of 7
modules already had expensive, width-pinned measurements at `0.5`, so
converging in that direction only required re-running two lightweight
notebooks instead of the toolkit's most expensive ones). I updated
`GLOBAL_PARAMS["wg_width_um"]` to `0.5`; `waveguide.py`/`bend.py` picked it
up automatically, and I re-ran both end-to-end; `spiral.py`/
`spiral_gds.py`/`grating_coupler.py` were fixed for free (no notebook/design
point yet). Re-running `bend.py`'s notebook also surfaced a second, unrelated
latent bug: its loss-fit `curve_fit` seeded `p0`'s floor parameter from
`loss_db.min()` with no clamp, which crashed with "Initial guess is outside
of provided bounds" once one sweep point's measured loss dipped a hair below
0 dB (FDTD noise around a near-zero true loss at the largest swept radius) --
I fixed it by clamping the seed to `max(loss_db.min(), 0.0)`, matching the
model's own `floor>=0` physical constraint.

**Lesson:** a shared default that every *newer* module bypasses via its own
standalone `DEFAULT_PARAMS` provides no actual protection against drift --
grep for the literal value across every module periodically, don't trust that
a `GLOBAL_PARAMS` docstring's stated intent is actually being followed
everywhere. Also: re-running any notebook after a parameter change can surface
older, unrelated fragility (like an unclamped curve-fit seed) that a
never-quite-hit edge case had been hiding -- treat that as a real bug to fix
alongside the change, not as noise to work around.

## Stale `DEFAULT_PARAMS` value silently diverged from the design point it was supposed to mirror

**Symptom:** `07_mzi.ipynb` failed on its very first cell that does real work
-- a consistency assertion comparing `mzi.DEFAULT_PARAMS["coupling_length_um"]`
against `data/design_points/coupler.yaml`'s saved value. The two didn't
match: `mzi.py` had `9.895741297648652` hardcoded, while the actual validated
design point (after a TE/TM re-measurement, see
`simulation_settings_record.md`) was `13.873040884936595`.

**Root cause:** I had written `mzi.py`'s `DEFAULT_PARAMS` to "mirror
`coupler.py`'s own DEFAULT_PARAMS / design point verbatim" per its own
comment, but never updated it after `coupler.py`'s coupling length was
re-measured (a value I'd baked in at authoring time, not read live). Because
`07_mzi.ipynb` had **never been executed** in this repo (0 cached outputs
before this session), nothing had ever caught the drift.

**Fix:** I updated the hardcoded constant to match the live design point
exactly. **Lesson:** a notebook with no execution history is not neutral --
treat every one of its "known-good" defaults as unverified until a real run
proves otherwise, especially any hardcoded value that's supposed to mirror a
*different* module's own state.

## Tolerance constants that were never empirically exercised

**Symptom:** once I fixed the issue above, `07_mzi.ipynb` got further but
then failed its own `assert validation["passed"]` in Section 8 -- measured
energy deviation `0.0392`, tolerance `ENERGY_TOL=0.03`. After I widened it to
`0.05` and re-ran, Section 11's re-validation (at a *different*,
sweep-selected `delta_L_um`) failed again at `0.0667`.

**Root cause:** same underlying issue as above -- this notebook's tolerance
constants were written with specific numbers in their comments ("measured
~5% during this notebook's own development") that read as empirical, but the
notebook had never actually been run to produce those numbers. The real,
first-ever measurement (`0.0392` at `delta_L_um=10`, `0.0667` at the
sweep-selected `delta_L_um=15`) didn't match the aspirational comment.

**Fix:** I set `ENERGY_TOL=0.08` (matching `MAX_EXPECTED_LOSS`), i.e. real
margin above the largest value I actually measured across the whole
notebook, not a tighter guess. I updated the notebook markdown and
`simulation_settings_record.md` with the real numbers. **Lesson:** when a
tolerance constant's comment cites a specific measured number, do not trust
it until you've reproduced that number yourself in the current session --
comments describing "empirical" results can themselves be unverified.

## `curve_fit` diverging to parameter bounds despite a good starting seed

**Symptom** (`04_bend_topology_optimization` racetrack CMT fit, notebook04):
a closed-form initial guess for `(kappa, alpha, n_eff)` had a small residual
(~0.8% RMS) against the FDTD data, but refining it with
`scipy.optimize.curve_fit` on the complex `(Re, Im)` S21 values made the fit
*worse* -- converging to `kappa=1.0` (a bound) and `n_eff` pinned at the edge
of its search window, with 40-45x higher residual than the untouched seed.

**Root cause:** two compounding issues. (1) With round-trip length
`L ~ 30-40 um`, the round-trip phase `2*pi*n_eff*L/wavelength` is extremely
sensitive to `n_eff` -- a full 2π phase wrap needs only `d(n_eff) ~
wavelength/L`. The complex-domain residual therefore has many aliased local
optima within any reasonably-sized search window, and a linearized
trust-region step can jump straight into one of them. (2) `kappa` (~0.1-1),
`alpha` (~1e-3), and `n_eff` (~2.4) live on wildly different natural scales;
without explicit scaling, `curve_fit`'s default unit-step behavior takes a
badly-proportioned first step.

**Fix:** (a) I fit the real-valued, non-oscillatory `|S21|^2` power spectrum
instead of the complex value -- this removes the phase-aliasing problem
entirely since the fitted quantity no longer wraps. (b) I passed an explicit
`x_scale=[kappa_scale, alpha_scale, n_eff_scale]` matching each parameter's
natural magnitude. (c) as a hard safety net, I compared the refined fit's
residual against the seed's own residual and **kept the seed outright** if
refinement made things worse -- confirmed this combination reliably recovers
a good answer instead of silently shipping a diverged one.
**Lesson:** a good closed-form seed is not protection against a bad
refinement step; always guard the refinement with a residual comparison, and
be suspicious of fitting a complex/oscillatory quantity directly when a
real-valued, monotonic transform of it is available.

## GDSFactory-derived geometry not matching native `mp.Block` construction

**Symptom** (`coupler.py`/`mzi.py`, notebook06): a newly-added
gdsfactory-derived S-bend geometry, built to trace the *exact same*
raised-cosine analytic curve as the native construction, still showed a
non-trivial permittivity-map difference against the native version (max
`eps` diff ~2.3-3.5, ~1-2% of pixels differing by >0.01) -- concentrated
right at the S-bend edges, not spread uniformly or randomly.

**Investigation:** my first hypothesis was that native's block count
(`n_seg=24`) was too coarse a "staircase" approximation, so I tested it by
rebuilding native with `n_seg=400` -- the difference barely changed, which
ruled out discretization coarseness. I then ran a vertical-slice comparison
at a fixed `x`, which showed the two geometries' core regions agreeing
almost exactly in the *bulk*, with differences concentrated at the *edges*
-- consistent with a systematic edge/boundary effect, not a wrong curve.

**Root cause:** the native construction uses **axis-aligned** Blocks (width
held constant in the *y* direction only), which `coupler.py`'s own docstring
already documented as accurate to within ~2% of the true perpendicular width
at this device's gentle S-bend slope (~0.17). The gdsfactory path sweeps its
cross-section **perpendicular to the local tangent** -- the more physically
correct representation for a curved waveguide, but therefore a genuinely
different polygon at the sub-pixel level for the *same* nominal centerline.
I confirmed this further: `mzi.py`'s own delay-arm Blocks (already
tangent-rotated in the native code, unlike the coupler's S-bend Blocks)
showed *better* agreement with their own gdsfactory equivalent, consistent
with this explanation.

**Fix:** I did not force a match (that would mean either degrading the
gdsfactory path to the less-accurate axis-aligned convention, or touching an
already-validated native path used by another notebook). Instead, I
documented the expected discrepancy honestly in both the notebook and
`simulation_settings_record.md`, quantified it (max/mean diff, % of pixels
affected), and kept `simulate_baseline()` on the native path regardless --
the gdsfactory path is display-only. **Lesson:** "two constructions of the
same nominal curve" is not the same claim as "pixel-identical" once either
one uses axis-aligned segments along a bend -- check which convention each
side uses before being surprised by a mismatch.

## Background shell's working directory reverted between tool calls

**Symptom:** a `jupyter nbconvert` command that had worked from
`photonics-simulation-toolkit/` failed immediately afterward with
`pattern 'notebooks/07_mzi.ipynb' matched no files` -- and, worse, silently
created a new, empty `notebooks/` directory one level up (at the wrong cwd),
which looked at first like a clue.

**Root cause:** I had chained `cd photonics-simulation-toolkit && ...
run_in_background` in one call; the `cd` took effect for the backgrounded
process, but the persistent shell's cwd reverted to the parent directory
(`meep-course`) by the next tool call. My next command assumed the `cd` had
"stuck" and ran from the wrong directory.

**Fix:** I `rmdir`'d the stray empty directory (after confirming it was
empty), then re-ran with `cd <absolute path> && ...` bundled into the *same*
command invocation as the actual work, rather than relying on a previous
call's `cd` having persisted. **Lesson:** before launching a long background
job, `pwd` or otherwise confirm the working directory in the *same* turn,
and prefer bundling `cd` and the real command into one invocation over
depending on shell-state persistence across separate tool calls -- and
always check whether a "no files matched" failure created any stray
directories before assuming it was a no-op.

## Re-validating a shared module's design point without re-running every dependent notebook

**Symptom:** I widened `06_directional_coupler.ipynb`'s Section 10 sweep
range (from `[2,4,6,8,10,12,15]` to `[2,6,10,14,18,22,26]` µm) to
deliberately probe past the coupler's beat-curve peak. Re-running the
notebook shifted the interpolated 50:50 `coupling_length_um` from
`13.873040884936595` to `13.848128928800111` -- small, but `mzi.py`'s
`DEFAULT_PARAMS` hardcodes an exact copy of this value (the same mirroring
pattern already documented above), so leaving it unsynced would reintroduce
the very drift that earlier bug was about.

**Decision:** I updated `mzi.py`'s hardcoded default to the new value, but
chose **not** to re-run `07_mzi.ipynb` or regenerate
`data/design_points/mzi.yaml` as part of the same pass -- that notebook's
own cached outputs and saved design point still reflect the prior
`13.873040884936595` coupling length until it is next re-run. This was a
deliberate, scoped choice (I weighed it against the cost of a full MZI
re-run, which chains two coupler stages plus a delay-arm sweep), not an
oversight -- I flagged it in both `mzi.py`'s own comment and
`docs/simulation_settings_record.md`'s `coupler.py`/`mzi.py` section so a
future session doesn't assume `mzi.yaml` is current just because `mzi.py`'s
default is. **Lesson:** when a shared module's validated parameter changes,
syncing the *source-code default* and re-running *every* notebook that
consumes it are separable decisions -- do the first immediately (cheap,
prevents silent drift) and make the second an explicit, documented choice
rather than an assumption either way.

## Editing large, already-executed notebooks that exceed the Read-tool size limit

**Symptom:** several notebooks (01/02/03/04, each with real cached plot
outputs) grew past the ~25k-token limit the standard Read tool enforces,
making the normal Read-then-NotebookEdit workflow unusable without losing
the existing outputs.

**Fix, two techniques depending on whether the existing outputs are worth
keeping:**
1. **If the notebook is about to be fully re-executed anyway** (outputs will
   be regenerated regardless): strip all cell outputs first (`execution_count
   = None`, `outputs = []`) via a small script, which shrinks the file enough
   for a normal Read, then edit normally, then re-run headlessly at the end
   to regenerate fresh outputs.
2. **If existing outputs must be preserved** (e.g. only a source-comment
   change, no re-execution planned) or a single targeted **output** update is
   needed without re-running the whole (expensive) notebook: edit the
   notebook's JSON directly via a Python script (`json.load` /
   modify one cell's `source` list / `json.dump`), never touching other
   cells' `outputs`. For injecting a *fresh* output into one specific cell
   without re-running the full notebook, compute that cell's result in a
   standalone script (capturing stdout and, for a plot, a base64-encoded PNG
   via `fig.savefig` into a `BytesIO`), then construct a proper
   `stream`/`display_data` output dict and assign it to that cell only.

**Lesson:** never assume the interactive Read/NotebookEdit tool pairing is
the only option for a large notebook -- direct JSON manipulation is safe and
often necessary, as long as cell `id`s are used as the join key and the
surrounding `outputs`/`execution_count` structure is preserved exactly for
cells not being touched.

## `grating_coupler.py`'s `coupling_efficiency_overlap()` could exceed the physical power available to it

**Symptom:** reviewing the promoted design's field snapshot (Section 5.2's
plot, `etch_depth_um=0.16`/`duty_cycle=0.6` at the time), I noticed it
didn't visually look like ~70-80% of incident power was radiating upward --
no clean beam, mostly what looked like standing-wave interference. Rather
than dismiss that gut reaction, I pushed to investigate, and three
successive rounds of increasingly specific questions progressively isolated
the real bug:

1. *Is the field plot just captured at the wrong wavelength?* Yes,
   partially: the field snapshot is a single-frequency DFT always at the
   O-band CENTER wavelength, not the design's own peak-efficiency
   wavelength. Re-plotting at the actual peak wavelength (via a new
   `field_snapshot_wl_um` argument I added to `simulate_baseline()`) showed
   a much cleaner-looking beam -- but that only raised a sharper follow-up
   question.
2. *Is the monitor, sitting at a wavelength-scale distance, just picking up
   evanescent near-field energy as if it were real radiated power?* I
   tested this directly: I computed the physical Poynting vector
   `Sz_phys = Re(Ez[Meep]*conj(Hx[Meep]))` from an ad-hoc diagnostic capture
   of `[mp.Ez, mp.Hy, mp.Hx]` at the design's peak wavelength, then
   integrated it across the aperture at heights from 0.1um to 1.9um above
   the grating surface. The integrated flux stayed within 97-100% of the
   `mon_up` value at every height -- genuinely propagating power, not
   evanescent (which would have decayed sharply over a comparable distance,
   given the grating-period-scale ~0.08um decay length of the lowest
   non-propagating diffraction order). Ruled out.
3. *A uniform (non-apodized) grating coupler shouldn't beat ~23% (the
   literature comparison figure for a similar 220nm-Si/2um-BOX single-etch
   uniform grating, from Yang et al., Sci. Rep. 13:18112 2023) -- so why
   does mine report 77%?* I tried restricting the near-field integration
   window to fiber-MFD width, flush with the grating start, implemented as
   a standalone script reusing `coupling_efficiency_overlap`'s own math on
   a restricted window. Result: even the MOST conservative window choice
   (fixed fiber center, no re-optimization) gave 54%, still comfortably
   above `upward_power=38.3%` -- the actual measured fraction of incident
   power radiated upward at that wavelength (from the `mon_up` flux
   monitor). Since `coupling_efficiency` cannot physically exceed the
   fraction of power that reaches the near-field plane at all, this proved
   the bug was in the metric's own normalization, not the fiber-position or
   window-size assumptions.

**Root cause:** `coupling_efficiency_overlap()`'s original formula
normalized by `integral(|E_sim|^2 dx)` -- an E-field-only SHAPE-overlap
fidelity (Cauchy-Schwarz-bounded to <=1 relative to `E_sim`'s own intensity,
independent of its absolute amplitude/power), not a genuine power-normalized
mode-coupling efficiency. A near-field whose spatial SHAPE happens to
resemble the assumed Gaussian, but which carries little actual power (most
of which reflects or radiates before ever reaching that shape), could still
score arbitrarily high on this metric -- this is mathematically why it broke
the physical `<= upward_power` bound.

**Fix attempt #1 (properly bounded, but had ANOTHER bug):** I rewrote the
formula as a standard E×H reciprocity/mode-overlap integral (Snyder & Love;
the same formula commercial mode solvers use), which required the near-field
monitor to also capture `Hx` (I added this to `_make_simulation`'s
`near_field_obj` DFT and threaded it through
`NearFieldSnapshot`/`_run`/`simulate_baseline`). My first implementation of
this formula gave `eta~=0` at the promoted design's own peak wavelength --
clearly wrong, and I traced it to a real handedness bug: this module's own
stated axis convention ("Meep's invariant z axis represents physical y",
module docstring) swaps exactly two axes relative to Meep's x, which is an
ORIENTATION-REVERSING operation for a right-handed frame (`x_meep × z_meep =
-y_meep`, not `+y_meep`). Naively reading the raw `Ez[Meep]` DFT array as
physical `Ey` without a sign correction is silently wrong for any
cross-product-based (Poynting/reciprocity) calculation built on top of it,
even though it's completely harmless for plain field-MAGNITUDE uses (the
`field_snapshot` plots, `passivity_check`, and every flux-monitor-based
quantity `upward_power`/`downward_power`/`reflection`, which Meep computes
internally in its own native frame and never routes through this
relabeling). Re-deriving the physical Poynting vector directly in a genuine
right-handed physical frame, I confirmed `Ey(phys) = -Ez(Meep)`, while
`Hx(phys) = Hx(Meep)` and `Hz(phys) = Hy(Meep)` both need no flip -- only the
axis actually being swapped (physical y) picks up the sign. The symptom
(near-total numerator cancellation, `-23.03-11.89j` vs `+22.98+11.91j` in
the two cross-terms of a debug printout) was a dead giveaway once I looked
for it: two terms that should reinforce for a well-matched co-propagating
pair were instead almost perfectly canceling, exactly what a missing
relative sign flip between them produces.

**Fix (final):** I negated `e_sim` once (`ey_sim = -e_sim`) to obtain the
true physical `Ey`, then used it consistently throughout the reciprocity
formula -- see `coupling_efficiency_overlap()`'s own docstring (its
"HANDEDNESS NOTE" section) for the full derivation; I kept it there rather
than only here since any future reader modifying that function needs to
re-derive this correctly. I verified it against the diagnostic Poynting
data: corrected `eta` came out to ~41% at the design's own peak wavelength,
close to (and no longer wildly exceeding) `upward_power=38.3%` -- the small
residual (~2.5 points) is consistent with expected approximation error
(paraxial/plane-wave `H_fiber` model, finite-aperture window truncation),
not a remaining sign or normalization bug.

**Lesson:** (1) staying skeptical of a result that "doesn't look right"
physically, even after a plausible-sounding first explanation, is worth it
-- three rounds of increasingly specific questions here (wrong wavelength ->
evanescent contamination -> literature comparison + restricted-window test)
were needed to actually isolate the real bug, and each earlier "explanation"
I reached for (short-domain near-field, confirmed-propagating via
height-invariance) was TRUE but incomplete, not wrong. (2) Any hand-rolled
cross-product/Poynting-vector calculation built on top of a module that uses
a non-identity axis relabeling (this module's rotated x-z convention, or any
future one) must re-derive field-component sign conventions from scratch in
the TRUE physical frame -- don't assume a plain "Meep_axis_A =
physical_axis_B" statement in a docstring is safe to use directly on
vector/cross-product quantities without checking handedness; it's safe only
for scalar magnitude/intensity quantities.

**Addendum (recurred in an ad hoc diagnostic, caught before reporting):**
while verifying `simulate_fiber_incidence()`'s coupling direction, I noticed
the field snapshot's raw pattern looked like it was "propagating right,"
toward +x, rather than left into the input lead, and went back to re-check
the grating-period sign convention. A throwaway diagnostic script I wrote to
compute the x-direction Poynting flux used
`Sx = Re(Ez[Meep]*conj(Hy[Meep]))` -- missing the leading minus sign this
same bug class requires (Meep's own x-axis is untouched by the axis
relabeling, so `Sx_phys` should be read directly off Meep's native cross
product, `(E x H*)_x[Meep] = -Ez[Meep]*conj(Hy[Meep])`). The unsigned
version gave a result that flatly contradicted the already-trusted
`get_eigenmode_coefficients` measurement (which reports ~477x more power
toward -x than +x at the input-lead monitor -- correct, unaffected by this
bug since it's Meep's own internal calculation, not a hand-rolled one). The
contradiction itself was the tell; fixing the sign resolved it immediately
(input-lead-region net Sx flipped from +60.8 to -60.8, now agreeing with the
eigenmode-coefficient ratio). This confirms the lesson above isn't
hypothetical -- this exact mistake recurred within the same session, in a
five-minute verification script, despite having just written the general
warning about it. I now treat ANY new hand-rolled Poynting calculation on
this module's fields as guilty until proven consistent with an independent,
Meep-native measurement (a flux monitor or `get_eigenmode_coefficients`),
not as a one-time check to get right and then trust from memory next time.

## `GaussianBeam2DSource`-based fiber-incidence testing gave direction-flipping, seemingly-contradictory results -- resolved by time-reversal (phase-conjugation)

**Symptom:** while iterating on `simulate_fiber_incidence()`'s beam
position, I noticed the CENTERED default under-coupled and repositioned the
beam edge at `x_grating_start`, which helped (see the function's own
docstring). As a further trial, I reversed the fiber tilt sign and got a
wildly counter-intuitive result: the REVERSED tilt (physically mounting the
fiber the other way, `-fiber_angle_deg`) coupled ~34-54x MORE strongly into
the -x guided mode than the original, "correct by reciprocity"
`+fiber_angle_deg` tilt -- at the SAME beam position. This flatly
contradicted an earlier algebraic momentum-matching derivation I'd worked
out (grating equation `beta' = k_x,in +/- K_g`, solved for both signs of
`k_x,in`), which had predicted the reversed tilt should couple into the
OPPOSITE (+x) guided mode via the opposite diffraction order, not a
stronger -x signal.

**Root cause:** `mp.GaussianBeam2DSource`'s idealized analytic beam (a smooth,
paraxial, single-tilt-angle solution) is a POOR model for what this specific
grating's near-field actually requires, in two independent ways, both
confirmed by a rigorous ground-truth test (see Fix below):
1. **Amplitude:** the real radiated near-field (captured from
   `simulate_baseline()`'s own `near_field`) has a complex, multi-lobed
   envelope concentrated near the grating's input-side periods (see
   `coupling_efficiency_overlap()`'s docstring on why the intensity centroid
   sits near `x_grating_start`, not the aperture center) -- nothing like a
   single-lobed Gaussian, regardless of where that Gaussian is centered.
2. **Phase:** the real near-field's phase has grating-period-scale RIPPLES
   superimposed on the mean 10-degree tilt (visible as small oscillations
   around the linear fit in a phase-vs-x plot) -- `GaussianBeam2DSource`'s
   closed-form solution has none of this fine structure, only a smooth
   large-scale phase front.
Because the actual required field doesn't resemble a tilted Gaussian at ALL,
"how well a given `GaussianBeam2DSource` configuration happens to overlap
with it" is not a smooth, monotonic function of the beam's nominal tilt sign
or position -- so results from sweeping those parameters can look
non-monotonic or even sign-flipped versus naive momentum-matching intuition,
without any bug being present.

**Fix / validation method -- true time-reversal (phase-conjugation) test, not
an idealized beam model.** My own physical reasoning: this simulation has no
non-reciprocal materials, so time-reversal symmetry must hold EXACTLY. I
took the actual complex near-field from the OUTcoupling run
(`near_field.field` at the peak-efficiency frequency), complex-conjugated it
(= time reversal t -> -t for a frequency-domain phasor), and injected that
EXACT profile as a custom `mp.Source(component=mp.Ez, amp_func=...)` at the
same near-field plane (`z_mon_up`), propagating back down into the grating.
Result: near-PERFECT reconstruction of the -x guided mode --
`|alpha_-x|^2=461.9` vs `|alpha_+x|^2=0.0009`, a 497,000:1 ratio (vs the best
`GaussianBeam2DSource` config's ~479:1). This simultaneously confirmed: (a)
time-reversal symmetry holds as expected, (b) `derive_grating_period()`'s
design is correct -- independently cross-checked by fitting the OUTcoupling
near-field's own phase slope, giving a measured tilt of 10.17 degrees against
the 10.0-degree design target, and (c) every earlier confusing
`GaussianBeam2DSource` result was a modeling-fidelity artifact, not a period,
sign-convention, or handedness bug.

**Follow-up hybrid test, isolating amplitude vs. phase contribution:** I
built a second custom source using an IDEALIZED Gaussian amplitude (centered
at `x=2um`, the real near-field's own approximate energy concentration zone)
multiplied by the REAL time-reversed PHASE (kept from the actual simulated
near-field, ripples and all). Result: direction discrimination stayed
excellent (`|alpha_-x|^2=18.98` vs `|alpha_+x|^2=0.0029`, 6538:1), but
absolute magnitude dropped to ~4% of the full (real amplitude + real phase)
reconstruction. I concluded that getting the PHASE structure right (matching
the grating's actual local diffraction physics, not just the mean tilt
angle) is what mainly governs whether power couples into the correct
DIRECTION at all; getting the AMPLITUDE shape right (matching the real
near-field's own multi-lobed envelope, not just a Gaussian centered at a
good position) is what mainly governs the achievable MAGNITUDE. This is the
mechanistic reason real grating couplers often use microlensed fibers
(reshaping both the amplitude envelope AND the phase front, not just aiming
a plain cleaved fiber) and are known to be sensitive to fiber lateral
position -- a plain single-mode fiber's Gaussian mode is a fundamentally
imperfect match to this device's own natural near-field shape, which I
confirmed quantitatively here rather than just asserting from general
folklore.

**Lesson:** when a "reasonable" idealized source model gives a
counter-intuitive or seemingly bug-like result, don't trust hand-derived
momentum/phase-matching arguments built on that idealized model as the
arbiter of truth -- go straight to the ground truth the simulator can
actually provide (here, literal time-reversal of the real simulated field)
before concluding there's a sign error or design bug. The idealized model
being a poor fit is itself often the whole explanation, and time-reversal
symmetry (when no non-reciprocal materials are present) is a powerful,
cheap-to-apply consistency check for exactly this class of "does reciprocal
coupling actually work" question.

## Porting a 2-port mode-coefficient convention to a 4-port device silently assumed a coordinate frame

**Symptom:** while building `add_drop_ring.py` (notebook 10, a 4-port
extension of `racetrack.py`'s 2-port ring), a coarse smoke-test run showed
one of the two excitations (`add`) with a huge, obviously-wrong reflection
coefficient and roughly 70% of input power unaccounted for by the energy
check -- while the OTHER excitation (`input`) looked physically reasonable
(a few-percent deviation, plausible for a deliberately under-converged
smoke test). Reciprocity between the two excitations also failed by a wide
margin (~0.87 vs. a normal few-percent).

**Root cause:** I ported `racetrack.py`'s two-port mode-coefficient rule
("at the excited port, keep index 1 [backward]; at the other port, keep
index 0 [forward]") over unchanged, and it silently encodes an assumption
specific to that module's own geometry: the excited port is always on the
-x side of the domain. That's true for every 2-port device in this toolkit
so far, but not for the add-drop ring's `add` port, which sits on the +x
side. Applying the same rule there picked the WRONG traveling-wave component
for both the self-reflection and one of the two cross-bus transfers.

**Fix:** I replaced the "self vs. other" rule with a "which edge of the
domain is this monitor on" rule -- `input`/`drop` (both at the domain's -x
edge) always keep the backward (index 1) component, `through`/`add` (both
at the +x edge) always keep the forward (index 0) component, *regardless of
which port was excited*. This is a more fundamental characterization (any
signal reaching an edge monitor in steady state is, by construction of an
absorbing PML, moving away from the device on that edge's own side) and
happens to reduce to `racetrack.py`'s original rule exactly when the
excited port is on the -x side.

**Lesson:** a "self vs. other port" rule discovered from a 2-port device
may actually be a "which side of the domain" rule in disguise, indistinguishable
from the general case as long as every existing example happens to excite from
one particular side. Before reusing a coefficient-index/sign convention from
an existing module in a new topology, check whether the new device excites
from a side the old rule never had to handle -- and verify with an
independent cross-check (here, the two-excitation reciprocity check) rather
than trusting that the ported rule "looks similar enough."
