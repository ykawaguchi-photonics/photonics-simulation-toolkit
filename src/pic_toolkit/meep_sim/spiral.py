"""Meep geometry (permittivity map only, so far -- no simulate_baseline yet)
for a spiral delay waveguide, matching picwriter's Spiral component
(https://picwriter.readthedocs.io/en/latest/components/spirals.html): a
rounded-RECTANGULAR double spiral -- straight runs joined by quarter-circle
corners, spiraling inward from the input (top-left) for `num_loops` laps,
then a central U-turn, then spiraling back OUTWARD nested exactly
`spacing_um` between the inbound strands, ending at the output (top-right).
Built with a small turtle-graphics-style path builder (`_Turtle`) rather
than an offset-curve Prism, since the path is piecewise straight+circular-
arc, not one smooth curve.

GEOMETRIC NOTE on length vs. bend radius: this module originally targeted a
fixed total centerline length (~20um) at a >=5um bend radius. Neither holds
here: a genuine double spiral -- one inbound lap plus the matching outbound
lap nested inside it -- costs at least 4 quarter-circle corners' worth of
arc EACH way (8 quarter-circles total, i.e. 2 full circles) before any
straight segments are even added: >=2*pi*min_radius_um*2 =~ 37.7um at
min_radius_um=3.0 for just ONE loop, growing with num_loops. A ~20um length
budget can't fit this topology at any reasonable bend radius (confirmed down
to min_radius_um=2.0, where even num_loops=1 alone still needs >=31.4um).
Per direct user confirmation, this version drops the fixed-length target and
instead reports the length that falls out of `spiral_width_um` /
`spiral_height_um` / `num_loops` / `spacing_um` (`total_length_um` in
_domain's output) -- the double-spiral SHAPE is what matters here, not
hitting an arbitrary short length.

Simulation model: 2D effective-index cross-section, same convention as
`waveguide.py`/`bend.py` (TE, Ez out-of-plane, no vertical confinement).
Only this file (plus `waveguide.py`, `bend.py`, `bend_topopt.py`,
`racetrack.py`, `coupler.py`, `mzi.py`) imports meep.
"""

from __future__ import annotations

import contextlib
import io

import meep as mp
import numpy as np


@contextlib.contextmanager
def _quiet_meep():
    """See bend.py's identical helper -- silences Meep's C++-layer stdout/
    stderr chatter around init_sim(), without swallowing real Python
    exceptions."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
# User-adjustable parameters.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "wg_width_um": 0.5,           # matches bend.py's own width, per this module's spec
    "core_index": 2.70,           # silicon (effective index, matches every other component)
    "clad_index": 1.44,           # silicon dioxide
    "min_radius_um": 3.0,         # corner-radius floor, applied to EVERY corner (both the
                                   # inbound/outbound spiral corners and the central U-turn) --
                                   # relaxed from an initial 5.0um, confirmed OK by the user,
                                   # specifically to help fit a real double-spiral loop in less
                                   # overall length (see module docstring).
    "spacing_um": 6.0,             # center-to-center distance between adjacent nested strands.
                                   # The central U-turn's own radius is spacing_um/2, so this
                                   # must be >= 2*min_radius_um (_domain checks explicitly) --
                                   # same reasoning meep_sim/mzi.py's very first spiral-adjacent
                                   # attempt used for its own U-turn radius floor. Edge-to-edge
                                   # clearance is spacing_um-wg_width_um=5.5um at the default --
                                   # far beyond any evanescent coupling range for this index
                                   # contrast (this toolkit's own coupler.py needs only 0.2um
                                   # edge-to-edge for STRONG coupling); the radius floor, not
                                   # coupling avoidance, is what actually sets this minimum.
    "spiral_width_um": 30.0,      # outer bounding-box width of the inbound spiral's first lap
    "spiral_height_um": 25.0,     # outer bounding-box height of the inbound spiral's first lap
    "num_loops": 2,               # number of inbound laps before the central U-turn (the
                                   # outbound path retraces the same number back out, nested
                                   # spacing_um inside each inbound strand). _domain raises
                                   # ValueError if a lap's straight run would go non-positive at
                                   # the given spiral_width_um/spiral_height_um/spacing_um.
    "lead_len_um": 2.0,           # straight lead from each port to where the spiral proper begins
    "n_arc": 40,                  # points per quarter-circle corner.
    "resolution": 25,
    "dpml_um": 1.0,
    "margin_um": 1.0,
}


class _Turtle:
    """Minimal turtle-graphics path builder: track (x, y, heading) and emit
    mp.Block (straight run, ROTATED to the current heading via e1/e2 -- same
    constant-perpendicular-width technique meep_sim/mzi.py's delay arm uses,
    not axis-aligned stacking) / mp.Prism (quarter-circle corner, same
    offset-curve technique bend.py's rounded_bend_vertices uses) geometry as
    it goes. `sign` on turn() is +1 for a CCW (left) turn, -1 for CW (right).
    """

    def __init__(self, x, y, heading, width, material, n_arc):
        self.x, self.y, self.heading = x, y, heading
        self.width, self.material, self.n_arc = width, material, n_arc
        self.blocks = []

    def forward(self, length):
        if length <= 1e-9:
            return
        dx, dy = np.cos(self.heading), np.sin(self.heading)
        tangent = mp.Vector3(dx, dy, 0)
        normal = mp.Vector3(-dy, dx, 0)
        cx, cy = self.x + dx * length / 2, self.y + dy * length / 2
        self.blocks.append(mp.Block(
            size=mp.Vector3(length, self.width, mp.inf), center=mp.Vector3(cx, cy, 0),
            e1=tangent, e2=normal, e3=mp.Vector3(0, 0, 1), material=self.material,
        ))
        self.x, self.y = self.x + dx * length, self.y + dy * length

    def turn(self, radius, sign):
        half_w = self.width / 2
        normal_angle = self.heading + sign * np.pi / 2
        center_x = self.x + radius * np.cos(normal_angle)
        center_y = self.y + radius * np.sin(normal_angle)
        start_angle = self.heading - sign * np.pi / 2
        end_angle = start_angle + sign * np.pi / 2
        thetas = np.linspace(start_angle, end_angle, self.n_arc)
        outer_r, inner_r = radius + half_w, radius - half_w
        outer = [mp.Vector3(center_x + outer_r * np.cos(t), center_y + outer_r * np.sin(t)) for t in thetas]
        inner = [mp.Vector3(center_x + inner_r * np.cos(t), center_y + inner_r * np.sin(t)) for t in thetas]
        self.blocks.append(mp.Prism(
            vertices=outer + inner[::-1], height=mp.inf, axis=mp.Vector3(0, 0, 1), material=self.material,
        ))
        self.x = center_x + radius * np.cos(end_angle)
        self.y = center_y + radius * np.sin(end_angle)
        self.heading += sign * np.pi / 2


def _lap_lengths(width, height, spacing, n_sides):
    """Side lengths for a rounded-rectangle spiral, alternating
    width/height, each shrinking by `spacing` every 2 sides (one full lap)
    -- the standard rectangular-spiral turtle sequence: after sides i and
    i+2 (same axis, opposite direction), the path has moved inward by
    `spacing` on each of the two perpendicular sides visited in between, so
    length[i+2] = length[i] - spacing. Raises ValueError if any length would
    go non-positive (the spiral has run out of room)."""
    lengths = []
    for i in range(n_sides):
        base = width if i % 2 == 0 else height
        length = base - spacing * (i // 2)
        if length <= 0:
            raise ValueError(
                f"side {i} of the spiral would have non-positive length ({length:.3f}um) -- "
                f"reduce num_loops, or grow spiral_width_um/spiral_height_um relative to spacing_um."
            )
        lengths.append(length)
    return lengths


def _domain(params: dict) -> dict:
    """Derive the full inbound+outbound turtle path length and the cell size
    -- shared by build_geometry, _ports, and get_permittivity_map, so all
    three always agree (same role as every other pic_toolkit component's own
    _compute_domain/_domain). Also computes total_length_um (reported, not
    forced -- see module docstring)."""
    width, height = params["spiral_width_um"], params["spiral_height_um"]
    spacing, radius = params["spacing_um"], params["min_radius_um"]
    if spacing / 2 < radius:
        raise ValueError(
            f"spacing_um/2={spacing / 2}um is below min_radius_um={radius}um -- the central "
            "U-turn would violate the bend-radius floor. Increase spacing_um to at least "
            "2*min_radius_um."
        )
    n_sides = 4 * params["num_loops"]
    lengths = _lap_lengths(width, height, spacing, n_sides)

    total_straight = sum(lengths)
    corners_one_way = n_sides * (np.pi / 2) * radius
    u_turn_len = np.pi * radius
    total_length_um = 2 * total_straight + 2 * corners_one_way + u_turn_len + 2 * params["lead_len_um"]

    lead, dpml, margin = params["lead_len_um"], params["dpml_um"], params["margin_um"]
    cell_x = width + 2 * (lead + dpml + margin)
    cell_y = height + 2 * (lead + dpml + margin)

    return dict(
        lengths=lengths, n_sides=n_sides, radius=radius, spacing=spacing,
        width=width, height=height, lead=lead,
        cell_x=cell_x, cell_y=cell_y, total_length_um=total_length_um,
    )


def _run_turtle(params: dict) -> "_Turtle":
    """Inbound spiral (num_loops laps, turning clockwise, shrinking per
    _lap_lengths) from the input port, a central U-turn, then the outbound
    spiral (the SAME side-length sequence retraced in reverse order, still
    turning clockwise -- reversing a constant-handedness turtle's move
    sequence retraces a path offset by spacing_um from the forward one at
    every point, which is exactly the nesting a double spiral needs) back
    out to the output port. Shared by build_geometry and _ports so both
    always agree on where the turtle actually ends up."""
    dom = _domain(params)
    core = mp.Medium(index=params["core_index"])
    width, radius = params["wg_width_um"], dom["radius"]
    lengths = dom["lengths"]

    t = _Turtle(x=-dom["lead"], y=0, heading=0, width=width, material=core, n_arc=params["n_arc"])
    t.forward(dom["lead"])
    for length in lengths:
        t.forward(length)
        t.turn(radius, sign=-1)  # clockwise, spiraling inward

    # Central U-turn: 180 degrees, turning COUNTER-clockwise (opposite handedness
    # from the inbound spiral) -- found empirically (by rendering and comparing
    # several sign combinations) to be the connector that lets the outbound pass
    # nest evenly BETWEEN the inbound strands at every point, rather than retrace
    # on top of them. This matches the general rule that mirroring a path (here,
    # reversing direction of travel) also flips its handedness -- the outbound
    # pass is a mirror image of the inbound one, not a literal retrace.
    t.turn(radius, sign=1)
    t.turn(radius, sign=1)

    for length in reversed(lengths):
        t.forward(length)
        t.turn(radius, sign=1)  # counter-clockwise, spiraling back outward
    t.forward(dom["lead"])
    return t


def build_geometry(params: dict) -> list:
    """See _run_turtle for the actual path construction."""
    return _run_turtle(params).blocks


def _ports(params: dict) -> dict:
    dom = _domain(params)
    t = _run_turtle(params)
    return {"input": mp.Vector3(-dom["lead"], 0), "output": mp.Vector3(t.x, t.y)}


def _bounding_box(blocks: list, half_w: float) -> tuple:
    """Exact (xmin, xmax, ymin, ymax) of the built geometry -- needed because
    the spiral's actual footprint isn't simply centered on (0,0) or on a
    simple function of spiral_width_um/spiral_height_um (the U-turn and
    output lead both extend well past the nominal bounding box on one side;
    see build_geometry). mp.Block entries are ROTATED (via e1/e2, not
    axis-aligned), so their true extent needs all 4 corners, not just
    center+-size/2; mp.Prism entries already store their boundary directly
    as `vertices`.
    """
    xs, ys = [], []
    for b in blocks:
        if hasattr(b, "vertices"):
            xs += [v.x for v in b.vertices]
            ys += [v.y for v in b.vertices]
        else:
            c, e1, e2, half_len = b.center, b.e1, b.e2, b.size.x / 2
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    xs.append(c.x + s1 * half_len * e1.x + s2 * half_w * e2.x)
                    ys.append(c.y + s1 * half_len * e1.y + s2 * half_w * e2.y)
    return min(xs), max(xs), min(ys), max(ys)


def get_permittivity_map(params: dict):
    """Permittivity map WITHOUT running any FDTD timestepping -- inspect the
    spiral geometry before paying for an expensive simulation, same
    discipline as every other component. Returns (eps, extent_um) directly
    (this module doesn't have a PermittivityMap dataclass yet -- add one, and
    a simulate_baseline, if this component moves past the exploratory
    stage)."""
    blocks = build_geometry(params)
    half_w = params["wg_width_um"] / 2
    xmin, xmax, ymin, ymax = _bounding_box(blocks, half_w)
    dpml, margin = params["dpml_um"], params["margin_um"]
    cell = mp.Vector3((xmax - xmin) + 2 * (dpml + margin), (ymax - ymin) + 2 * (dpml + margin), 0)
    geom_center = mp.Vector3((xmin + xmax) / 2, (ymin + ymax) / 2, 0)
    clad = mp.Medium(index=params["clad_index"])
    sim = mp.Simulation(
        cell_size=cell, boundary_layers=[mp.PML(params["dpml_um"])],
        geometry=blocks,
        default_material=clad, resolution=params["resolution"],
        geometry_center=geom_center,
    )
    with _quiet_meep():
        sim.init_sim()
    eps = sim.get_epsilon()
    extent = (geom_center.x - cell.x / 2, geom_center.x + cell.x / 2,
              geom_center.y - cell.y / 2, geom_center.y + cell.y / 2)
    return eps, extent
