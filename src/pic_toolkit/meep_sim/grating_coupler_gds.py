"""x-y top-down GDS layout for the grating coupler -- a SEPARATE projection
of the same design parameters `grating_coupler.py`'s x-z FDTD module uses,
not a shared geometry source with it (see that module's docstring: a GDS
layer is an x-y footprint only, it cannot encode a z-stack, so this file's
Component is never fed into any Meep simulation). This is the one component
in the toolkit whose `meep_sim/*.py` pair has no `build_geometry_from_gds()`
counterpart.

Deliberately meep-free (unlike every OTHER meep_sim/*.py module) -- this
Component is a genuine tapeout export, so a pure layout/tapeout workflow (or
CI) never needs to import meep to generate it.

Reuses gdsfactory's own `grating_coupler_rectangular` directly, same
precedent as notebooks/07_spiral_waveguide.ipynb using `gf.components.
spiral()` directly rather than hand-rolling a component.
"""

from __future__ import annotations

from pathlib import Path


def build_gf_component(params: dict):
    """The x-y tapeout layout: straight input waveguide -> taper ->
    n_periods uniform grating teeth. Uses the SAME period_um/duty_cycle/
    n_periods/width_grating_um/taper_length_um/fiber_angle_deg values
    `grating_coupler.py`'s DEFAULT_PARAMS uses, so the two projections stay
    in sync -- pass `grating_coupler.derive_domain(params)["period_um"]`
    (the resolved period, not the possibly-None DEFAULT_PARAMS value) as
    `params["period_um"]` before calling this.

    EBL-ready output, two features beyond the stock gdsfactory component:
    - `layer_grating="SHALLOW_ETCH"` puts every grating-tooth rectangle on
      its own layer, distinct from the input waveguide/taper (which stays on
      the cross-section's own WG layer) -- matches this device's own
      "partial-etch" framing (see `grating_coupler.py`'s module docstring),
      not `DEEP_ETCH`, and lets a mask engineer target the etch step to just
      the grating region without hand-picking polygons.
    - A GDS text label (non-drawn metadata, not a physical shape) is added
      above the component summarizing the fabrication parameters a mask
      engineer/EBL operator needs at a glance -- period, duty cycle, etch
      depth, period count -- so the file is self-documenting without a
      separate spec sheet.
    """
    import gdsfactory as gf

    cross_section = gf.cross_section.strip(width=params["wg_width_um"])
    component = gf.components.grating_coupler_rectangular(
        n_periods=params["n_periods"],
        period=params["period_um"],
        fill_factor=params["duty_cycle"],
        width_grating=params["width_grating_um"],
        length_taper=params["taper_length_um"],
        polarization="te",
        wavelength=(params["wl_min_um"] + params["wl_max_um"]) / 2,
        fiber_angle=params["fiber_angle_deg"],
        cross_section=cross_section,
        layer_grating="SHALLOW_ETCH",
    )

    # gdsfactory's stock components are cached/locked (the @cell decorator) --
    # add_label() raises LockedError on the original object, so an unlocked
    # copy is required before any post-processing modification.
    component = component.copy()
    fab_label = (
        f"GRATING: period={params['period_um'] * 1000:.1f}nm "
        f"duty_cycle={params['duty_cycle']:.2f} "
        f"etch_depth={params['etch_depth_um'] * 1000:.1f}nm "
        f"n_periods={params['n_periods']}"
    )
    component.add_label(text=fab_label, position=(component.x, component.ymax + 3), layer="TEXT")
    return component


def write_gds(component, path: Path) -> Path:
    """Thin wrapper around gdsfactory's own Component.write_gds -- new
    territory for this toolkit (no other notebook writes an actual .gds
    file to disk). `data/gds/<component>/` is the natural new sibling to
    `data/design_points/<component>.yaml` and `data/sparams/<component>/`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return component.write_gds(gdspath=path)
