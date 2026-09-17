"""Differentiable twin of `circuits/wdm_sax.ipynb`'s `build_real_lattice_circuit`
(cell 28), parameterized by the two optimization variables `L_upper` and
`delta_L_delay_um` instead of the fixed baseline design point.

Architectural rule: meep-free. Coupler models are reused UNCHANGED from
`pic_toolkit.models.coupler` (fixed -- not optimized in this task), wrapped
in `unitary_project_diff` for a consistent (jnp-typed, but zero-gradient)
treatment alongside the differentiable arm/waveguide models. `L_upper`/
`delta_L_delay_um` are bound into per-instance model functions via Python
closures (the same idiom `wdm_sax.ipynb` cell 28 already uses for per-
instance coupler lengths), NOT via `sax.circuit()` netlist `settings` --
this avoids relying on unverified support for tracer-valued instance
settings in SAX 0.14.2. All 3 ref_arm instances share one `L_upper` closure;
all 3 delay_arm instances share one `delta_L_delay_um` closure, matching the
physical requirement that a maximally-flat lattice filter's arm-pairs all
carry the same delta_L.
"""

from __future__ import annotations

import sax

from .. import design_points
from ..models import coupler as coupler_model
from .arm_library import ArmLengthLibrary, load_arm_length_library, mzi_arm_diff
from .coupler_library import CouplerLengthLibrary, coupler_diff, load_coupler_length_library
from .coupler_library_2d import CouplerGapLengthLibrary, coupler_diff_2d, load_coupler_gap_length_library
from .differentiable_waveguide import waveguide_diff
from .unitary_project_diff import unitary_project_diff

_REPO_ROOT = coupler_model._REPO_ROOT
REFERENCE_ARM_LENGTH_UM = 26.0  # matches wdm_sax.ipynb's own constant: 2*lead_len_um(3.0)+4*radius_um(5.0)


def resolve_chosen_coupling_lengths_um(n_couplers: int = 4) -> list[float]:
    """Reproduces `wdm_sax.ipynb` cell 26's `CHOSEN_LC`: the per-stage
    coupling_length_um values (from the existing coupler sweep artifacts)
    closest to the ideal maximally-flat kappas for an n_couplers lattice.
    Fixed/non-optimized throughout this module."""
    import numpy as np

    from ..circuits import mzi_lattice as ml

    kappas, _signs = ml.synthesize_maximally_flat_kappas(n_couplers)

    sweep_dir = _REPO_ROOT / "data" / "sparams" / "coupler" / "sweep"
    available_Lc = {}
    for p in sorted(sweep_dir.glob("coupler_couplinglength*.npz")):
        Lc = float(p.stem.replace("coupler_couplinglength", ""))
        d = np.load(p)
        design_point = design_points.load_design_point(_REPO_ROOT / "data" / "design_points" / "coupler.yaml")
        wl0_um = 1.35
        idx_wl0 = int(np.argmin(np.abs(d["wavelengths_um"] - wl0_um)))
        available_Lc[Lc] = float(d["T_cross"][idx_wl0])

    return [min(available_Lc, key=lambda Lc: abs(available_Lc[Lc] - target)) for target in kappas]


def build_diff_lattice_circuit(L_upper, delta_L_delay_um, coupling_lengths_um: list[float],
                                arm_library_obj: ArmLengthLibrary | None = None):
    """Differentiable twin of `build_real_lattice_circuit`. `L_upper`,
    `delta_L_delay_um`: scalars, may be JAX tracers. `coupling_lengths_um`:
    fixed list of per-stage coupling lengths (e.g. from
    `resolve_chosen_coupling_lengths_um`), one `None` entry meaning "use the
    selected 50:50 design point" exactly like the original. Returns
    (circuit_fn, info) exactly like `sax.circuit()`."""
    if arm_library_obj is None:
        arm_library_obj = load_arm_length_library()

    n = len(coupling_lengths_um)
    instances, connections, models = {}, {}, {}

    for i, Lc in enumerate(coupling_lengths_um):
        name = f"coupler{i}"
        instances[name] = {"component": name, "settings": {}}
        if Lc is None:
            models[name] = unitary_project_diff(coupler_model.coupler, ("in_top", "in_bot", "out_top", "out_bot"))
        else:
            models[name] = unitary_project_diff(
                (lambda L: (lambda wl=1.35: coupler_model.coupler_at_sweep_point(L, wl=wl)))(Lc),
                ("in_top", "in_bot", "out_top", "out_bot"),
            )

    ref_arm_model = unitary_project_diff(
        (lambda wl=1.35: waveguide_diff(wl, L_upper)), ("o1", "o2"),
    )
    delay_arm_model = unitary_project_diff(
        (lambda wl=1.35: mzi_arm_diff(wl, delta_L_delay_um, library=arm_library_obj)), ("o1", "o2"),
    )

    for i in range(n - 1):
        ref_name, delay_name = f"ref_arm{i}", f"delay_arm{i}"
        instances[ref_name] = {"component": "ref_arm", "settings": {}}
        instances[delay_name] = {"component": "delay_arm", "settings": {}}
        connections[f"coupler{i},out_top"] = f"{ref_name},o1"
        connections[f"coupler{i},out_bot"] = f"{delay_name},o1"
        connections[f"{ref_name},o2"] = f"coupler{i + 1},in_top"
        connections[f"{delay_name},o2"] = f"coupler{i + 1},in_bot"

    models["ref_arm"] = ref_arm_model
    models["delay_arm"] = delay_arm_model

    ports = {
        "in_top": "coupler0,in_top", "in_bot": "coupler0,in_bot",
        "out_top": f"coupler{n - 1},out_top", "out_bot": f"coupler{n - 1},out_bot",
    }
    netlist = {"instances": instances, "connections": connections, "ports": ports}
    return sax.circuit(netlist, models=models)


def build_diff_lattice_circuit_variable_couplers_and_arms(L_upper, delta_L_list: list, coupling_lengths_um: list,
                                                            arm_library_obj: ArmLengthLibrary | None = None,
                                                            coupler_library_obj: CouplerLengthLibrary | None = None):
    """Differentiable twin of `build_diff_lattice_circuit_variable_couplers`,
    but `delta_L_list` gives each of the N-1 arm-pairs its OWN independent
    delta_L (unlike every other builder in this module, where one shared
    `delta_L_delay_um` is bound into all arm-pair instances via a single
    closure, matching the production design's physical constraint that every
    arm-pair carries the same delay).

    This extra freedom exists to probe one specific question:
    `coupler_library.py`'s own docstring establishes that coupling LENGTH
    only changes how much power a coupler splits, never its cross-port phase
    SIGN (that sign is set by the coupling geometry itself, confirmed
    constant across the whole measured length range) -- so no coupler-length
    optimization can ever realize a stage that needs the opposite sign. A
    per-stage delta_L, however, changes each arm-pair's own phase bias at
    wl0, which is a physically realizable (if unconventional) lever no
    single shared delta_L has access to. Whether that lever can actually
    substitute for an unrealizable coupler sign is exactly what this
    builder's objective (`objective.make_objective_with_couplers_and_arms`)
    is used to test empirically -- this function makes no claim it works,
    only that the extra degrees of freedom are available to try."""
    if arm_library_obj is None:
        arm_library_obj = load_arm_length_library()
    if coupler_library_obj is None:
        coupler_library_obj = load_coupler_length_library()

    n = len(coupling_lengths_um)
    assert len(delta_L_list) == n - 1, f"delta_L_list must have {n - 1} entries (one per arm-pair), got {len(delta_L_list)}"
    instances, connections, models = {}, {}, {}

    for i, Lc in enumerate(coupling_lengths_um):
        name = f"coupler{i}"
        instances[name] = {"component": name, "settings": {}}
        models[name] = unitary_project_diff(
            (lambda L: (lambda wl=1.35: coupler_diff(wl, L, library=coupler_library_obj)))(Lc),
            ("in_top", "in_bot", "out_top", "out_bot"),
        )

    for i in range(n - 1):
        ref_name, delay_name = f"ref_arm{i}", f"delay_arm{i}"
        instances[ref_name] = {"component": ref_name, "settings": {}}
        instances[delay_name] = {"component": delay_name, "settings": {}}
        models[ref_name] = unitary_project_diff(
            (lambda wl=1.35: waveguide_diff(wl, L_upper)), ("o1", "o2"),
        )
        models[delay_name] = unitary_project_diff(
            (lambda dL: (lambda wl=1.35: mzi_arm_diff(wl, dL, library=arm_library_obj)))(delta_L_list[i]),
            ("o1", "o2"),
        )
        connections[f"coupler{i},out_top"] = f"{ref_name},o1"
        connections[f"coupler{i},out_bot"] = f"{delay_name},o1"
        connections[f"{ref_name},o2"] = f"coupler{i + 1},in_top"
        connections[f"{delay_name},o2"] = f"coupler{i + 1},in_bot"

    ports = {
        "in_top": "coupler0,in_top", "in_bot": "coupler0,in_bot",
        "out_top": f"coupler{n - 1},out_top", "out_bot": f"coupler{n - 1},out_bot",
    }
    netlist = {"instances": instances, "connections": connections, "ports": ports}
    return sax.circuit(netlist, models=models)


def build_diff_lattice_circuit_variable_couplers_2d(L_upper, delta_L_delay_um,
                                                      coupling_lengths_um: list, gaps_um: list,
                                                      arm_library_obj: ArmLengthLibrary | None = None,
                                                      coupler_gap_library_obj: CouplerGapLengthLibrary | None = None):
    """Differentiable twin of `build_diff_lattice_circuit_variable_couplers`,
    but each coupler's `gap_um` is now ALSO an independent differentiable
    variable (via `coupler_library_2d.coupler_diff_2d`'s 2D interpolation),
    not fixed at the production value of 0.2um. Motivated directly by the
    2D coupler sweep (`notebooks/06b_directional_coupler_gap_sweep.ipynb`):
    dispersion residual was found to spike sharply right at each gap's own
    coupling extremum, and the production design's `Lc=26um` (coupler0)
    happens to sit almost exactly at `gap=0.20`'s own extremum -- a
    DIFFERENT (gap, length) combination hitting the same target `kappa`
    while avoiding that extremum was shown to cut the dispersion residual by
    ~380x. Letting the optimizer choose gap jointly with length is what lets
    it find such combinations on its own instead of only being handed one
    by hand.

    `gaps_um` bounds MUST stay inside `coupler_gap_library_obj.gap_um`'s own
    characterized range (currently ~0.12-0.35um -- gap=0.08/0.10 are
    excluded from that library; see `coupler_library_2d._EXCLUDED_GAPS`'s
    own docstring for the two-supermode-beat de-winding limitation that
    caused that) to avoid extrapolation, exactly like every other axis in
    this optimization layer."""
    if arm_library_obj is None:
        arm_library_obj = load_arm_length_library()
    if coupler_gap_library_obj is None:
        coupler_gap_library_obj = load_coupler_gap_length_library()

    n = len(coupling_lengths_um)
    assert len(gaps_um) == n, f"gaps_um must have {n} entries (one per coupler), got {len(gaps_um)}"
    instances, connections, models = {}, {}, {}

    for i, (Lc, gap) in enumerate(zip(coupling_lengths_um, gaps_um)):
        name = f"coupler{i}"
        instances[name] = {"component": name, "settings": {}}
        models[name] = unitary_project_diff(
            (lambda L, g: (lambda wl=1.35: coupler_diff_2d(wl, L, g, library=coupler_gap_library_obj)))(Lc, gap),
            ("in_top", "in_bot", "out_top", "out_bot"),
        )

    ref_arm_model = unitary_project_diff(
        (lambda wl=1.35: waveguide_diff(wl, L_upper)), ("o1", "o2"),
    )
    delay_arm_model = unitary_project_diff(
        (lambda wl=1.35: mzi_arm_diff(wl, delta_L_delay_um, library=arm_library_obj)), ("o1", "o2"),
    )

    for i in range(n - 1):
        ref_name, delay_name = f"ref_arm{i}", f"delay_arm{i}"
        instances[ref_name] = {"component": "ref_arm", "settings": {}}
        instances[delay_name] = {"component": "delay_arm", "settings": {}}
        connections[f"coupler{i},out_top"] = f"{ref_name},o1"
        connections[f"coupler{i},out_bot"] = f"{delay_name},o1"
        connections[f"{ref_name},o2"] = f"coupler{i + 1},in_top"
        connections[f"{delay_name},o2"] = f"coupler{i + 1},in_bot"

    models["ref_arm"] = ref_arm_model
    models["delay_arm"] = delay_arm_model

    ports = {
        "in_top": "coupler0,in_top", "in_bot": "coupler0,in_bot",
        "out_top": f"coupler{n - 1},out_top", "out_bot": f"coupler{n - 1},out_bot",
    }
    netlist = {"instances": instances, "connections": connections, "ports": ports}
    return sax.circuit(netlist, models=models)


def build_diff_lattice_circuit_variable_couplers(L_upper, delta_L_delay_um, coupling_lengths_um: list,
                                                  arm_library_obj: ArmLengthLibrary | None = None,
                                                  coupler_library_obj: CouplerLengthLibrary | None = None):
    """Differentiable twin of `build_diff_lattice_circuit`, but
    `coupling_lengths_um` entries are themselves treated as differentiable
    variables (each may be a JAX tracer) via `coupler_library.coupler_diff`'s
    continuous length interpolation, instead of the fixed-length
    `coupler_at_sweep_point`/`coupler()` used above. Each of the N couplers
    gets its OWN independent length (unlike L_upper/delta_L_delay_um, which
    are shared across all arm-pairs) -- pass a list of N scalars, no `None`
    entries (there is no "selected design point" special case here, only
    continuous interpolation)."""
    if arm_library_obj is None:
        arm_library_obj = load_arm_length_library()
    if coupler_library_obj is None:
        coupler_library_obj = load_coupler_length_library()

    n = len(coupling_lengths_um)
    instances, connections, models = {}, {}, {}

    for i, Lc in enumerate(coupling_lengths_um):
        name = f"coupler{i}"
        instances[name] = {"component": name, "settings": {}}
        models[name] = unitary_project_diff(
            (lambda L: (lambda wl=1.35: coupler_diff(wl, L, library=coupler_library_obj)))(Lc),
            ("in_top", "in_bot", "out_top", "out_bot"),
        )

    ref_arm_model = unitary_project_diff(
        (lambda wl=1.35: waveguide_diff(wl, L_upper)), ("o1", "o2"),
    )
    delay_arm_model = unitary_project_diff(
        (lambda wl=1.35: mzi_arm_diff(wl, delta_L_delay_um, library=arm_library_obj)), ("o1", "o2"),
    )

    for i in range(n - 1):
        ref_name, delay_name = f"ref_arm{i}", f"delay_arm{i}"
        instances[ref_name] = {"component": "ref_arm", "settings": {}}
        instances[delay_name] = {"component": "delay_arm", "settings": {}}
        connections[f"coupler{i},out_top"] = f"{ref_name},o1"
        connections[f"coupler{i},out_bot"] = f"{delay_name},o1"
        connections[f"{ref_name},o2"] = f"coupler{i + 1},in_top"
        connections[f"{delay_name},o2"] = f"coupler{i + 1},in_bot"

    models["ref_arm"] = ref_arm_model
    models["delay_arm"] = delay_arm_model

    ports = {
        "in_top": "coupler0,in_top", "in_bot": "coupler0,in_bot",
        "out_top": f"coupler{n - 1},out_top", "out_bot": f"coupler{n - 1},out_bot",
    }
    netlist = {"instances": instances, "connections": connections, "ports": ports}
    return sax.circuit(netlist, models=models)
