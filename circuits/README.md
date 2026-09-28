# circuits/

Circuit-level SAX simulations, composed from the cached, validated component
models in `src/pic_toolkit/models/` (e.g. `waveguide`, `racetrack`).

`wdm_sax.ipynb` and `wdm_mux4_sax.ipynb` never import Meep and never trigger
a new FDTD simulation — they only wire together component models that have
already been characterized and cached under `data/sparams/` and
`data/design_points/`. `mzi_fabrication_tolerance_sax.ipynb` is the one
exception (see below): it needs S-parameters at geometries that were never
separately cached, so it deliberately runs fresh, coupler-only FDTD.

The two WDM notebooks follow the same 7-section structure (Introduction →
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

- **`mzi_fabrication_tolerance_sax.ipynb`** — fabrication-tolerance study for
  `notebooks/07_mzi.ipynb`'s single, already-selected passive MZI design
  point (not the lattice/tree filters above). Waveguide width and coupler
  gap are modeled as a single anti-correlated critical-dimension (CD) bias
  `b` (`wg_width_um = nominal + b`, `gap_um = nominal - b`), not independent
  variables, swept deterministically over `+-10nm` in `5nm` steps. Each
  point composes the full MZI's response from a fresh, coupler-only FDTD run
  (`meep_sim/coupler.py` — cheap, since `gap_um` never touches the delay
  arm's own geometry) plus an MPB arm-dispersion solve, cascaded via
  `sax.circuit()`; the arm's `n_eff` needs a small calibrated dispersion
  correction (fit once against the nominal design's real full-MZI FDTD
  result) to get the interference phase right, because the 2D
  effective-index arm model alone does not capture the arm's full
  dispersion. This investigation also surfaced the same class of gap in
  `models/waveguide.py`'s own reference-arm model (used by both notebooks
  above); that gap has since been fixed with a real MPB-measured group
  index (see `models/waveguide.py`).

