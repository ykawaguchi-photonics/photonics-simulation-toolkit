"""Reusable engine behind the Mux4 (binary-tree cascaded-MZI) WDM
demultiplexer -- a DIFFERENT topology from `mzi_lattice.py`'s uniform-
`delta_L_um` lattice filter (N couplers in series, one shared passband).
This module instead builds a 2-stage BINARY TREE of 3 plain 50:50-coupler
MZIs: stage 1 splits the input into an "up"/"down" branch, and stage 2 (one
MZI per branch) splits each branch again, giving 4 distinct wavelength-
channel outputs -- the "Mux4" pattern (see e.g. the Luceda Photonics "muxN"
training reference). No lattice synthesis is needed here: every stage is an
ordinary `kappas=[0.5,0.5], signs=[1,1]` MZI, i.e. `mzi_lattice.py`'s own
trivial N=2 case, reused as-is via `build_lattice_circuit`.

**The quarter-wave correction.** Naively giving `stage2_up` and
`stage2_down` the SAME `delta_L_2` (half of `delta_L_1`, per the standard
FSR-halving rule) does NOT give 4 clean channels -- verified numerically
while building this module: the "down" branch (fed by stage 1's cross
output) comes out clean, but the "up" branch's two channels are capped at
~59% peak height (degenerate double peaks), because stage 1's bar-output
envelope peaks exactly where stage 2's own transfer function sits at its
ambiguous 50/50 point, not at an extremum. Luceda's own reference design
fixes this with a `center_wavelength` offset on `stage_2_up` only (a free
phase-tuning parameter their abstract model supports directly). Our
building blocks (`ideal_arm_model`, and real per-artifact `mzi_at_sweep_
point` models) have no such free parameter -- only a physical arm-length
difference -- so the fix here is a genuine, small ADDITIONAL length
increment on `stage2_up`'s arm only, `quarter_wave_length_um` below,
confirmed by direct numerical sweep (all 4 sign/branch combinations tried;
only "shift up only" gives all 4 channels their full peak height
simultaneously).

Never imports meep.
"""

from __future__ import annotations

import sax

from . import mzi_lattice as ml


def quarter_wave_length_um(n_eff0: float, wl0_um: float) -> float:
    """Universal physical arm-length increment (independent of whatever base
    `delta_L` it's added on top of) needed to add exactly pi/2 (a quarter-FSR)
    of DIFFERENTIAL phase at `wl0_um`: `wl0_um / (4*n_eff0)`, derived from
    setting `2*pi*n_eff0*delta/wl0_um = pi/2`. Equivalent to Luceda's "shift
    center_wavelength by FSR/4", linearized at `wl0_um`. This is how
    `stage_2_up` gets its quarter-FSR offset relative to `stage_2_down` in
    this module's tree -- see the module docstring for why a length
    increment (not a common-mode/global length change, which would only add
    an overall phase invisible to any leaf-port power measurement) is the
    only way to realize it with our fixed-arm-length building blocks."""
    return wl0_um / (4.0 * n_eff0)


def build_mux4_tree_circuit(stage1_fn, stage2_up_fn, stage2_down_fn):
    """Generic tree-composition wiring, agnostic to whether each `stage*_fn`
    is an ideal analytic circuit (`ml.build_lattice_circuit`'s return) or a
    real, single-measured-MZI-artifact model (e.g. `models.mzi.mzi_at_sweep_
    point`, wrapped in `ml.unitary_project`) -- each just needs to expose the
    standard 4-port shape (`in_top`, `in_bot`, `out_top`, `out_bot`).

    Topology: `in_top` (the only excited input) -> stage1 -> stage1's
    `out_top`/`out_bot` feed stage2_up's/stage2_down's own `in_top` -> each
    stage2's `out_top`/`out_bot` become 2 of the 4 final channels
    (`ch1`=stage2_up.out_top, `ch2`=stage2_up.out_bot,
    `ch3`=stage2_down.out_top, `ch4`=stage2_down.out_bot). `in_bot` (stage1's
    own second input) and both stage2 instances' own `in_bot` ports are left
    unconnected/idle -- exposed on the outer circuit for completeness (a real
    device's unused port), not physically driven in this demux's use case.

    A `sax.circuit()`'s returned callable is itself a valid SAX model
    function, so nesting works directly: this outer `sax.circuit()` call
    takes the 3 stage circuits as its `models` dict entries, no special
    handling needed."""
    netlist = {
        "instances": {
            "stage1": {"component": "stage1", "settings": {}},
            "stage2_up": {"component": "stage2_up", "settings": {}},
            "stage2_down": {"component": "stage2_down", "settings": {}},
        },
        "connections": {
            "stage1,out_top": "stage2_up,in_top",
            "stage1,out_bot": "stage2_down,in_top",
        },
        "ports": {
            "in_top": "stage1,in_top", "in_bot": "stage1,in_bot",
            "ch1": "stage2_up,out_top", "ch2": "stage2_up,out_bot",
            "ch3": "stage2_down,out_top", "ch4": "stage2_down,out_bot",
        },
    }
    models = {"stage1": stage1_fn, "stage2_up": stage2_up_fn, "stage2_down": stage2_down_fn}
    return sax.circuit(netlist, models=models)


def build_ideal_mux4_tree_circuit(
    delta_L_1_um: float, delta_L_2_um: float, n_eff0: float, n_g0: float, wl0_um: float,
    quarter_wave_shift_up: bool = True, n_couplers: int = 2,
):
    """Ideal/analytic Mux4 tree: 3 identical-`n_couplers` `ml.build_lattice_circuit`
    instances per stage. `n_couplers=2` (default) is the trivial case
    (`kappas=[0.5,0.5], signs=[1,1]` -- an ordinary 50:50 MZI, no lattice synthesis
    needed); `n_couplers=4` instead uses `ml.synthesize_maximally_flat_kappas(4)`'s
    halfband design at every one of the 3 stages -- a flatter per-stage passband,
    quantitatively confirmed (see `docs/simulation_settings_record.md`) to be
    substantially MORE ROBUST to a `stage2_up` phase-calibration error than the plain
    N=2 stage (e.g. an 8.86deg residual error costs N=2 ~12dB of worst-channel
    extinction but only ~22dB for N=4). `stage2_up` gets `quarter_wave_length_um`
    added to `delta_L_2_um`'s own arm-length difference unless
    `quarter_wave_shift_up=False`, which instead reproduces this module's own
    documented "broken baseline" (both stage-2 instances identical) for direct
    comparison -- see the module docstring."""
    if n_couplers == 2:
        kappas, signs = [0.5, 0.5], [1, 1]
    else:
        kappas, signs = ml.synthesize_maximally_flat_kappas(n_couplers)
    stage1_fn, _ = ml.build_lattice_circuit(kappas, signs, delta_L_1_um, n_eff0, n_g0, wl0_um)
    extra = quarter_wave_length_um(n_eff0, wl0_um) if quarter_wave_shift_up else 0.0
    stage2_up_fn, _ = ml.build_lattice_circuit(kappas, signs, delta_L_2_um + extra, n_eff0, n_g0, wl0_um)
    stage2_down_fn, _ = ml.build_lattice_circuit(kappas, signs, delta_L_2_um, n_eff0, n_g0, wl0_um)
    return build_mux4_tree_circuit(stage1_fn, stage2_up_fn, stage2_down_fn)
