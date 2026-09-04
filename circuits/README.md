# circuits/

Circuit-level SAX simulations, composed from the cached, validated component
models in `src/pic_toolkit/models/` (e.g. `waveguide`, `racetrack`).

This directory never imports Meep and never triggers a new FDTD simulation —
it only wires together component models that have already been characterized
and cached under `data/sparams/` and `data/design_points/`.

- `wdm_mux_mzi_lattice_sax.ipynb` — cascaded Mach-Zehnder lattice filter (WDM
  demultiplexer), 2 vs. 4-coupler comparison. Currently **analytic only**: its
  couplers/arms are ideal (`src/pic_toolkit/circuits/mzi_lattice.py`), not
  built from `src/pic_toolkit/models/`'s cached FDTD component models yet —
  see the notebook's own closing section for that deferred next step. This is
  also the first module anywhere in this repo to actually call `sax.circuit()`
  (every `pic_toolkit.models` module only produces SAX-*shaped* data by hand).
