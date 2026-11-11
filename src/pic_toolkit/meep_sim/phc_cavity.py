"""Photonic-crystal nanocavities in a suspended SiN membrane (3D Meep / MPB).

Two cavity families in a 200 nm Si3N4 membrane in air, designed for the visible (NV zero-phonon
line, 637 nm):

* **Nanobeam**: a 1D row of circular holes in a ridge, tapered toward the centre
  (McCutcheon & Loncar, Opt. Express 16, 19136 (2008); Quan & Loncar, Opt. Express 19, 18529
  (2011)).
* **L3**: three missing holes in a triangular hole lattice, with outward shifts of the holes at
  the cavity ends (Akahane et al., Nature 425, 944 (2003); Minkov & Savona, Sci. Rep. 4, 5124
  (2014)).

**Deviation from the toolkit's modeling convention:** these are full 3D simulations with the
bulk index of SiN (n = 2.0, non-dispersive, lossless) and air, not the 2D effective-index
cross-section used by the Si and SiN tracks. The out-of-plane radiation that limits a
photonic-crystal cavity's Q cannot be represented in 2D.

Units: lengths in um, frequencies in c/um (f = 1/lambda[um]).

Fundamental cavity modes of both families are TE-like with Ey dominant at the centre; Ey is even
(as a scalar) about the x = 0, y = 0 and z = 0 planes. In Meep's vector convention this is
Mirror(X, +1), Mirror(Y, -1), Mirror(Z, +1) (`TE_SYMMETRIES`), which reduces the computed volume
eightfold.
"""
from __future__ import annotations

import math

import meep as mp
import numpy as np

N_SIN = 2.0
T_SLAB = 0.200          # membrane thickness (um)
LAMBDA0 = 0.637         # design wavelength (um), NV zero-phonon line
SIN = mp.Medium(index=N_SIN)

TE_SYMMETRIES = [mp.Mirror(mp.X, phase=+1), mp.Mirror(mp.Y, phase=-1), mp.Mirror(mp.Z, phase=+1)]

# McCutcheon & Loncar 2008: 300 nm wide ridge, mirror a = 250 nm / r = 70 nm (17 periods), and a
# four-hole taper whose (a, r) fall to (214, 55) nm next to the cavity (listed centre outward).
MCCUTCHEON = dict(
    w=0.300, t=0.200,
    taper=[(0.214, 0.055), (0.226, 0.058), (0.238, 0.062), (0.250, 0.066)],
    mirror=(0.250, 0.070), n_mirror=17,
)


# ----------------------------------------------------------------------------- nanobeam

def nanobeam_holes(taper: list[tuple[float, float]], mirror: tuple[float, float], n_mirror: int,
                   gap: float, gap_mode: str = "section") -> np.ndarray:
    """Hole centres x > 0 and radii (N, 2) of a symmetric nanobeam, centre outward.

    Each hole sits at the middle of its own cell of length a_i, so neighbouring holes are
    (a_i + a_{i+1}) / 2 apart. `gap` is the cavity length: with gap_mode "section" the two inner
    cells start gap/2 from the centre (inner hole at gap/2 + a_1/2); with "edge" it is the
    edge-to-edge distance between the two inner holes (inner hole at gap/2 + r_1).
    """
    cells = list(taper) + [mirror] * n_mirror
    a1, r1 = cells[0]
    x = gap / 2 + (a1 / 2 if gap_mode == "section" else r1)
    out = [(x, r1)]
    for (a_prev, _), (a, r) in zip(cells[:-1], cells[1:]):
        x += (a_prev + a) / 2
        out.append((x, r))
    return np.array(out)


def nanobeam_geometry(holes: np.ndarray, w: float, t: float = T_SLAB) -> list:
    """Ridge along x (infinite, it runs through the PML) with air holes at +-x."""
    geo = [mp.Block(size=mp.Vector3(mp.inf, w, t), material=SIN)]
    for x, r in holes:
        for sx in (+1, -1):
            geo.append(mp.Cylinder(radius=r, height=t, center=mp.Vector3(sx * x), material=mp.air))
    return geo


def nanobeam_cell(holes: np.ndarray, w: float, t: float = T_SLAB, pad_x: float = 0.5,
                  pad_yz: float = 0.6, dpml: float = 0.5) -> mp.Vector3:
    x_end = holes[-1, 0] + holes[-1, 1]
    return mp.Vector3(2 * (x_end + pad_x + dpml), w + 2 * (pad_yz + dpml), t + 2 * (pad_yz + dpml))


# ----------------------------------------------------------------------------- L3

def l3_holes(a: float, r: float, nx: int = 12, ny: int = 7, shifts: tuple = (),
             radii: dict | None = None) -> np.ndarray:
    """Hole centres and radii (N, 3: x, y, r) of an L3 cavity in a triangular lattice.

    Rows y = j * a * sqrt(3)/2 for |j| <= ny, holes at x = (i + j/2) a with |x| <= nx a. The three
    holes (-1, 0), (0, 0), (1, 0) are removed. `shifts[k]` moves the holes at (+-(k+2), 0) outward
    along x by shifts[k] * a (Minkov & Savona's S1x, S2x, ...). `radii` maps (i, j) -> r for
    individually resized holes (applied symmetrically by sign of i).
    """
    out = []
    for j in range(-ny, ny + 1):
        for i in range(-nx - abs(j), nx + abs(j) + 1):
            x = (i + j / 2) * a
            if abs(x) > nx * a + 1e-9:
                continue
            if j == 0 and abs(i) <= 1:
                continue
            rr = r
            if j == 0 and 2 <= abs(i) < 2 + len(shifts):
                x += np.sign(i) * shifts[abs(i) - 2] * a
            if radii and (abs(i), j) in radii:
                rr = radii[(abs(i), j)]
            out.append((x, j * a * math.sqrt(3) / 2, rr))
    return np.array(out)


def l3_geometry(holes: np.ndarray, t: float = T_SLAB) -> list:
    geo = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, t), material=SIN)]
    geo += [mp.Cylinder(radius=r, height=t, center=mp.Vector3(x, y), material=mp.air) for x, y, r in holes]
    return geo


def l3_cell(a: float, nx: int, ny: int, t: float = T_SLAB, pad_z: float = 0.6,
            dpml: float = 0.5) -> mp.Vector3:
    """Cell just covering the hole pattern in-plane (the slab and lattice run into the PML)."""
    return mp.Vector3(2 * (nx * a + dpml), 2 * (ny * a * math.sqrt(3) / 2 + dpml), t + 2 * (pad_z + dpml))


# ----------------------------------------------------------------------------- 3D cavity run

def run_cavity(geometry: list, cell: mp.Vector3, fcen: float, df: float, resolution: float,
               dpml: float = 0.5, pml_dirs: str = "xyz", ring_time: float = 300.0,
               symmetries: list | None = None, field_box: mp.Vector3 | None = None,
               core_box: mp.Vector3 = mp.Vector3(0.8, 0.6, 0.3), f_range: tuple | None = None,
               pick: str = "amp", min_amp: float = 0.05) -> dict:
    """Ring-down of the cavity: Gaussian Ey pulse at the centre, Harminv on Ey at the centre.

    Mode choice: Harminv modes with frequency in `f_range` (default: the source band) and
    amplitude >= `min_amp` x the largest; among them `pick="amp"` takes the strongest and
    `pick="lowest"` the lowest frequency, `pick="q"` the highest Q. For the L3, "q" is the
    robust choice: a broadband pulse can excite a higher-order mode more strongly than the
    fundamental, and low-Q band-edge modes can sit just below it in frequency; the fundamental
    has the highest Q of the defect modes.

    Returns the modes Harminv finds (f, Q, amplitude) and, for the chosen mode, the mode
    volume V = int eps|E|^2 / max(eps|E|^2) in (lambda/n)^3 with n = N_SIN, evaluated from
    two snapshots a quarter period apart after the ring-down (|E(t)|^2 + |E(t+T/4)|^2 is the
    time-independent |E|^2 amplitude of a single standing-wave mode). `field_box` limits the
    volume integral (default: the cell minus the PML).
    """
    pml = []
    for d, ax in (("x", mp.X), ("y", mp.Y), ("z", mp.Z)):
        if d in pml_dirs:
            pml.append(mp.PML(dpml, direction=ax))
    src = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ey, center=mp.Vector3())]
    sim = mp.Simulation(cell_size=cell, geometry=geometry, sources=src, boundary_layers=pml,
                        resolution=resolution, symmetries=symmetries if symmetries is not None else TE_SYMMETRIES,
                        eps_averaging=True)
    h = mp.Harminv(mp.Ey, mp.Vector3(), fcen, df)
    sim.run(mp.after_sources(h), until_after_sources=ring_time)
    modes = sorted(([m.freq, m.Q, abs(m.amp), m.err] for m in h.modes), key=lambda m: -m[2])
    out = dict(modes=np.array(modes) if modes else np.zeros((0, 4)), resolution=resolution,
               cell=np.array([cell.x, cell.y, cell.z]), ring_time=ring_time, fcen=fcen, df=df)
    lo, hi = f_range or (fcen - df / 2, fcen + df / 2)
    cand = [m for m in modes if lo <= m[0] <= hi and m[2] >= min_amp * modes[0][2]] if modes else []
    if not cand:
        return out
    chosen = {"lowest": lambda: min(cand, key=lambda m: m[0]),
              "q": lambda: max(cand, key=lambda m: m[1])}.get(pick, lambda: cand[0])()
    f0, q0 = chosen[0], chosen[1]
    box = field_box or mp.Vector3(cell.x - 2 * dpml * ("x" in pml_dirs), cell.y - 2 * dpml * ("y" in pml_dirs),
                                  cell.z - 2 * dpml * ("z" in pml_dirs))
    vol = mp.Volume(center=mp.Vector3(), size=box)
    # Never gather the full 3D field: under MPI get_array copies it to every rank, which ran the
    # 6 GB machine out of memory at 64 px/um. The integral is evaluated in place (distributed);
    # the maximum of eps|E|^2 lies in the cavity core, so only that small box is gathered; the
    # plots use 2D slices through the centre.
    core = mp.Volume(center=mp.Vector3(), size=mp.Vector3(min(core_box.x, box.x), min(core_box.y, box.y),
                                                          min(core_box.z, box.z)))
    xy = mp.Volume(center=mp.Vector3(), size=mp.Vector3(box.x, box.y, 0))
    xz = mp.Volume(center=mp.Vector3(), size=mp.Vector3(box.x, 0, box.z))
    cs = [mp.Dielectric, mp.Ex, mp.Ey, mp.Ez]
    w_fun = lambda r, eps, ex, ey, ez: eps * (abs(ex) ** 2 + abs(ey) ** 2 + abs(ez) ** 2)
    integral, w_core, e2_xy, e2_xz, ey_xy = 0.0, 0.0, 0.0, 0.0, None
    for k in range(2):
        if k:
            sim.run(until=0.25 / f0)
        integral += sim.integrate_field_function(cs, w_fun, vol).real
        eps_c = sim.get_array(component=mp.Dielectric, vol=core)
        w_core = w_core + eps_c * sum(np.abs(sim.get_array(component=c, vol=core)) ** 2 for c in cs[1:])
        e2_xy = e2_xy + sum(np.abs(sim.get_array(component=c, vol=xy)) ** 2 for c in cs[1:])
        e2_xz = e2_xz + sum(np.abs(sim.get_array(component=c, vol=xz)) ** 2 for c in cs[1:])
        if k == 0:
            ey_xy = sim.get_array(component=mp.Ey, vol=xy)
    V = integral / w_core.max()
    lam_n = (1 / f0) / N_SIN
    out.update(f=f0, Q=q0, V=V / lam_n ** 3, V_um3=V,
               purcell=3 / (4 * math.pi ** 2) * q0 / (V / lam_n ** 3),
               eps_mid=sim.get_array(component=mp.Dielectric, vol=xy), e2_mid=e2_xy, ey_mid=ey_xy,
               e2_xz=e2_xz, eps_xz=sim.get_array(component=mp.Dielectric, vol=xz),
               box=np.array([box.x, box.y, box.z]))
    return out


# ----------------------------------------------------------------------------- MPB

def slab_te_bands(r_a: float, t_a: float, k_points: list, resolution: int = 16, sz: float = 6.0,
                  num_bands: int = 4) -> np.ndarray:
    """TE-like (z-even) bands of a triangular air-hole lattice in a suspended slab, f in c/a."""
    from meep import mpb

    lat = mp.Lattice(size=mp.Vector3(1, 1, sz), basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
                     basis2=mp.Vector3(math.sqrt(3) / 2, -0.5))
    geo = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, t_a), material=SIN),
           mp.Cylinder(radius=r_a, height=t_a, material=mp.air)]
    ms = mpb.ModeSolver(geometry_lattice=lat, geometry=geo, k_points=k_points, resolution=resolution,
                        num_bands=num_bands)
    ms.run_zeven()
    return np.array(ms.all_freqs)


def nanobeam_band_edges(a: float, r: float, w: float, t: float = T_SLAB, resolution: int = 24,
                        sy: float = 2.0, sz: float = 2.0) -> tuple[float, float]:
    """Dielectric and air band edges (c/um) at the X point of a nanobeam unit cell.

    One period of the beam (length a, holes of radius r) in an air box sy x sz (um); TE-like
    (z-even), Ey even in y (MPB y-odd).
    """
    from meep import mpb

    lat = mp.Lattice(size=mp.Vector3(a, sy, sz))
    geo = [mp.Block(size=mp.Vector3(mp.inf, w, t), material=SIN),
           mp.Cylinder(radius=r, height=t, material=mp.air)]
    ms = mpb.ModeSolver(geometry_lattice=lat, geometry=geo, k_points=[mp.Vector3(0.5)],
                        resolution=int(round(resolution / a)), num_bands=2)
    ms.run_yodd_zeven()
    f1, f2 = ms.all_freqs[0]       # lattice given in um, so already c/um
    return f1, f2


def mirror_strength(f1: float, f2: float, f_res: float) -> float:
    """Quan & Loncar mirror strength gamma (0 when f_res lies outside the band gap)."""
    f0 = (f1 + f2) / 2
    g2 = ((f2 - f1) / (f2 + f1)) ** 2 - ((f_res - f0) / f0) ** 2
    return math.sqrt(g2) if g2 > 0 else 0.0
