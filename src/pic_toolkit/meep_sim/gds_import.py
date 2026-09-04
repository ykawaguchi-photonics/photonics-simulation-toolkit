"""Convert a gdsfactory Component's polygons into Meep geometry -- lets any
gdsfactory component (not just the toolkit's own hand-built spiral.py) be
inspected as a Meep permittivity map, e.g. gdsfactory.components.spiral()
itself (see notebooks/07_spiral_waveguide.ipynb's final section).

Each polygon is emitted as its own mp.Prism, deliberately NOT merged into one
big polygon per layer first: meep_sim/mzi.py's delay-arm bump already hit a
real performance trap this way (a single Prism whose bounding box spans
almost the whole simulation cell forces Meep's subpixel-averaging to test
EVERY grid pixel against it, with no cheap per-object bounding-box
rejection) -- confirmed again here empirically (get_polygons(merge=True) on
a modest 2-loop spiral took several minutes and was abandoned;
get_polygons(merge=False), one small Prism per gdsfactory sub-component
instance, each with its own small local bounding box, took ~100s instead).

Only this file (plus `waveguide.py`, `bend.py`, `bend_topopt.py`,
`racetrack.py`, `coupler.py`, `mzi.py`, `spiral.py`) imports meep.
"""

from __future__ import annotations

import contextlib
import io

import meep as mp


@contextlib.contextmanager
def _quiet_meep():
    """See spiral.py's identical helper -- silences Meep's C++-layer stdout/
    stderr chatter around init_sim(), without swallowing real Python
    exceptions."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def gds_component_to_prisms(component, layer: tuple = (1, 0), material=None) -> list:
    """One mp.Prism per UNMERGED polygon on `layer` (default: gdsfactory's
    standard WG/waveguide layer) -- see module docstring for why merge=False.
    `component` must be dup()-able (gdsfactory `@cell`-decorated components
    are cached/locked; `.dup()` gives an unlocked copy safe to query)."""
    dbu = component.kcl.dbu
    unlocked = component.dup()
    polys = unlocked.get_polygons(merge=False, by="tuple")[layer]
    prisms = []
    for poly in polys:
        vertices = [mp.Vector3(p.x * dbu, p.y * dbu, 0) for p in poly.each_point_hull()]
        prisms.append(mp.Prism(vertices=vertices, height=mp.inf, axis=mp.Vector3(0, 0, 1), material=material))
        for h in range(poly.holes()):
            hole_vertices = [mp.Vector3(p.x * dbu, p.y * dbu, 0) for p in poly.each_point_hole(h)]
            prisms.append(mp.Prism(vertices=hole_vertices, height=mp.inf, axis=mp.Vector3(0, 0, 1),
                                    material=mp.Medium(index=1.0)))  # holes cut back to vacuum/cladding below
    return prisms


def get_permittivity_map(component, core_index: float = 2.70, clad_index: float = 1.44,
                          resolution: int = 25, dpml_um: float = 1.0, margin_um: float = 1.0,
                          layer: tuple = (1, 0)):
    """Permittivity map of a gdsfactory Component WITHOUT running any FDTD
    timestepping -- same discipline as every other pic_toolkit component.
    Cell is sized/centered tightly around the component's own bounding box
    (same `geometry_center` technique meep_sim/mzi.py uses for its own
    delta_L_um-dependent asymmetric cell) plus dpml_um+margin_um on every
    side. Returns (eps, extent_um), same shape as spiral.py's own
    get_permittivity_map (this module doesn't have a PermittivityMap
    dataclass yet either -- see that module's docstring for why)."""
    core = mp.Medium(index=core_index)
    clad = mp.Medium(index=clad_index)
    # holes (if any) inside the WG polygon are cut back to clad, not vacuum --
    # override the material used for holes now that clad_index is known.
    prisms = gds_component_to_prisms(component, layer=layer, material=core)
    for p in prisms:
        if p.material.epsilon_diag.x == 1.0:
            p.material = clad

    dbu = component.kcl.dbu
    bbox = component.bbox()
    xmin, ymin, xmax, ymax = bbox.left * dbu, bbox.bottom * dbu, bbox.right * dbu, bbox.top * dbu
    cell = mp.Vector3((xmax - xmin) + 2 * (dpml_um + margin_um), (ymax - ymin) + 2 * (dpml_um + margin_um), 0)
    geom_center = mp.Vector3((xmin + xmax) / 2, (ymin + ymax) / 2, 0)

    sim = mp.Simulation(
        cell_size=cell, boundary_layers=[mp.PML(dpml_um)], geometry=prisms,
        default_material=clad, resolution=resolution, geometry_center=geom_center,
    )
    with _quiet_meep():
        sim.init_sim()
    eps = sim.get_epsilon()
    extent = (geom_center.x - cell.x / 2, geom_center.x + cell.x / 2,
              geom_center.y - cell.y / 2, geom_center.y + cell.y / 2)
    return eps, extent
