# circuits/

Circuit-level SAX simulations, composed from the cached, validated component
models in `src/pic_toolkit/models/` (e.g. `waveguide`, `racetrack`).

This directory never imports Meep and never triggers a new FDTD simulation —
it only wires together component models that have already been characterized
and cached under `data/sparams/` and `data/design_points/`.

Two notebooks, each following the same 7-section structure (Introduction →
analytical SAX model (N=2, N=4) → import S-parameters from FDTD simulation →
circuit design with SAX+FDTD → comparison → GDS export → summary):

- **`wdm_sax.ipynb`** — the single-stage cascaded-coupler **lattice filter**
  (2 vs. 4 couplers sharing one `delta_L_um`, one shared passband made
  flatter with more couplers). This is also the first module anywhere in this
  repo to actually call `sax.circuit()` (every `pic_toolkit.models` module
  only produces SAX-*shaped* data by hand). Both N=2 and N=4 are built as an
  ideal/analytic model (`src/pic_toolkit/circuits/mzi_lattice.py`) AND from
  genuinely FDTD-measured coupler (`models/coupler.py`)/delay-arm
  (`models/mzi_arm.py`) S-parameters (wrapped in `mzi_lattice.unitary_project`
  to keep a 3+-stage cascade physically unitary), with an honest ideal-vs-real
  comparison and matching GDS layout export (`gds/wdm_n2.gds`, `gds/wdm_n4.gds`)
  for both.

- **`wdm_mux4_sax.ipynb`** — 4-channel WDM demultiplexer using the
  cascaded-MZI **binary-tree** ("Mux4") topology instead of a single lattice:
  one stage-1 lattice splits the input into an "up"/"down" branch, and one
  stage-2 lattice per branch splits each again, giving 4 *simultaneous*
  channel outputs (see Luceda Photonics' `muxN` training reference). Reuses
  `wdm_sax.ipynb`'s lattice as its per-node building block
  (`src/pic_toolkit/circuits/mux4_tree.py`), documenting the quarter-wave
  arm-length correction the "up" branch needs that a naive FSR-halving design
  misses. Both N=2 and N=4 per-stage designs are built and compared, ideal
  and real-FDTD alike — real channel extinction is materially better at N=4
  (worst channel 1.7dB → 10.1dB), the concrete reason the GDS export (Section
  6, `gds/wdm_mux4_n4.gds`) covers **N=4 only**. That layout is also this
  repo's first *branching* (non-series) physical layout, using
  `gf.routing.route_single_sbend` to connect `stage1`'s two outputs to the
  two downstream lattice devices.

See `docs/simulation_settings_record.md` for the full derivation and
measurement history behind both notebooks (why N=4 was chosen for the tree's
physical layout, the real delay-arm phase-calibration investigation, etc.) —
that investigation was originally spread across several intermediate
notebooks (`wdm_mux_mzi_lattice_sax.ipynb`, `mzi_real_fdtd_sax.ipynb`,
`wdm_mux4_tree_sax.ipynb`, `mux4_tree_real_fdtd_sax.ipynb`,
`mux4_tree_n4_real_fdtd_sax.ipynb`, `mux4_tree_n4_sax.ipynb`), since
consolidated into the two notebooks above.

## Geometry-optimization effort (paused)

A separate, later effort tried to close `wdm_sax.ipynb`'s ideal-vs-real gap
by optimizing the N=4 lattice's arm and coupler lengths/gaps (JAX adjoint
gradients, L-BFGS-B, a component-level dispersion diagnostic, and a greedy
per-component redesign, across eleven exploratory notebooks). It is now
paused with no optimized design adopted into production — both notebooks
above still ship their original baseline geometry
(`CHOSEN_LC=[26.0, 2.0, 18.0, 14.0]`, `gap_um=0.20`). See
`docs/simulation_settings_record.md`'s "`circuits/` MUX2 (N=4 lattice)
geometry-optimization program" section for the full findings (the coupler-3
sign ceiling, the target-anchoring bug fixed twice, the best point-sampled
result reached: 97.1% transmission / 0.71% crosstalk) before those
notebooks were removed. The differentiable surrogate they used remains at
`src/pic_toolkit/optimization/`, and the FDTD artifacts they produced remain
cached under `data/sparams/{mzi_arm,coupler}/optimized/` — neither is
referenced by `wdm_sax.ipynb`/`wdm_mux4_sax.ipynb`.
