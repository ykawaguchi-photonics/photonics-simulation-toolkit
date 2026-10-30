"""Dielectric topological photonic crystals: spin-Hall (Wu-Hu) and valley-Hall, TM pillars
and TE air holes, in MPB (unit cells, ribbons) and Meep (domain walls, strip feed).

Four crystals, all on the triangular Bravais lattice a1 = (1, 0), a2 = (1/2, sqrt(3)/2):

* ``spin_tm``  Wu & Hu, PRL 114, 223901 (2015): six dielectric rods (eps_d, diameter 0.24)
  on a hexagon of edge R around each lattice point, in air; TM (Ez). Lattice constant a0.
  a0/R = 3.125 (shrunken, trivial), 3 (honeycomb), 2.9 (expanded, topological).
* ``spin_te``  the inverted version: six equilateral-triangle air holes (side 0.3125 a0,
  one vertex toward the cluster centre; Barik et al. 2018, Parappurath et al. 2020) in an
  eps_d host; TE (Hz). Same a0/R values.
* ``valley_te`` honeycomb of inverted triangular air holes of two sizes (Shalaev et al.
  2019; He et al. 2019 use circles), s(1 -/+ delta) with s = 0.3125 sqrt(3) a (the spin-Hall
  hole), delta = 0.2; TE. Lattice constant a = a0 / sqrt(3) (honeycomb constant).
* ``valley_tm`` honeycomb of dielectric rods of two radii in air, r(1 -/+ delta) with
  r = 0.12 sqrt(3) a (the Wu-Hu rod), delta = 0.1, i.e. 0.187 a / 0.229 a (Lan et al.,
  PRA 103, L041502 (2021): 0.18 a / 0.22 a); TM.

Spin-Hall lengths are in a0, valley-Hall lengths in a; the symmetric limits of the two
families are the same honeycomb (a = a0 / sqrt(3)).

**Modeling assumptions (deviation from the toolkit default, CLAUDE.md section 1):** 2D,
non-dispersive, bulk eps_d = 11.7 (silicon) rather than the 2D effective index 2.7, as in
the Wu-Hu paper the spin-Hall rods reproduce. Air background/holes, eps = 1.

MPB conventions (checked numerically): geometric-object centres are in lattice
coordinates for the triangular lattice (``mp.cartesian_to_lattice``) and cartesian for a
rectangular supercell; ``get_epsilon()`` index (i, j) sits at lattice coordinates
(i/N - 1/2, j/N - 1/2).
"""

from __future__ import annotations

import numpy as np
import meep as mp
from meep import mpb

EPS_D = 11.7
EPS_A = 1.0
A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3) / 2])
ROW_H = np.sqrt(3) / 2
LATTICE = mp.Lattice(size=mp.Vector3(1, 1), basis1=mp.Vector3(*A1), basis2=mp.Vector3(*A2))
AIR = mp.Medium(epsilon=EPS_A)
HOST = mp.Medium(epsilon=EPS_D)

# High-symmetry points (reciprocal basis). K_ADJ is the corner adjacent to M_PT, so that
# Gamma-M-K_ADJ-Gamma follows the edge of the irreducible zone; K_PT is used for K-Gamma-M.
GAMMA = mp.Vector3()
M_PT = mp.Vector3(0.5, 0)
K_PT = mp.Vector3(-1 / 3, 1 / 3)
K_ADJ = mp.Vector3(1 / 3, -1 / 3)

# --- crystal parameters -----------------------------------------------------------------
SPIN_A0_OVER_R = {"trivial": 3.125, "honeycomb": 3.0, "topological": 2.9}
SPIN_ROD_D = 0.24                    # rod diameter / a0 (Wu-Hu Fig. 5)
SPIN_TRI_S = 0.3125                  # triangle side / a0 (Barik 2018, Parappurath 2020)
D_NN = 1 / np.sqrt(3)                # honeycomb nearest-neighbour distance / a
SITE_A = np.array([0.0, -D_NN / 2])  # valley sublattice A (down-pointing triangle)
SITE_B = np.array([0.0, +D_NN / 2])  # valley sublattice B (up-pointing triangle)
VALLEY_TE_S, VALLEY_TE_DELTA = SPIN_TRI_S * np.sqrt(3), 0.2
VALLEY_TM_R, VALLEY_TM_DELTA = (SPIN_ROD_D / 2) * np.sqrt(3), 0.1

KINDS = ("spin_tm", "spin_te", "valley_te", "valley_tm")
POLARIZATION = {"spin_tm": "TM", "spin_te": "TE", "valley_te": "TE", "valley_tm": "TM"}

# 1550 nm scales: spin-Hall TE edge wave at 0.408 c/a0, valley-Hall TE gap centre 0.2368 c/a
A0_NM = 0.408 * 1550.0
A_NM = 0.2368 * 1550.0


def valley_sizes(kind: str, variant: str) -> tuple[float, float]:
    """(size_A, size_B) of a valley crystal: triangle sides (TE) or rod radii (TM).
    VPC1: A small, B large; VPC2: swapped (inversion partner); symmetric: equal."""
    s, dl = (VALLEY_TE_S, VALLEY_TE_DELTA) if kind == "valley_te" else (VALLEY_TM_R, VALLEY_TM_DELTA)
    return {"vpc1": (s * (1 - dl), s * (1 + dl)), "vpc2": (s * (1 + dl), s * (1 - dl)),
            "symmetric": (s, s)}[variant]


# --- shapes -------------------------------------------------------------------------------

def triangle(center, s: float, angle: float) -> np.ndarray:
    """Equilateral triangle of side s centred at `center`, one vertex at polar `angle`."""
    ang = angle + np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    return np.asarray(center) + s / np.sqrt(3) * np.column_stack([np.cos(ang), np.sin(ang)])


def cluster_positions(R: float) -> np.ndarray:
    return R * np.column_stack([np.cos(np.arange(6) * np.pi / 3), np.sin(np.arange(6) * np.pi / 3)])


def spin_shapes(R: float, kind: str, center=(0.0, 0.0)) -> list[dict]:
    """The six inclusions of a spin-Hall cluster: rods (TM) or triangles pointing a vertex
    at the cluster centre (TE)."""
    c0 = np.asarray(center, float)
    out = []
    for k, p in enumerate(cluster_positions(R)):
        if kind == "spin_tm":
            out.append({"circle": c0 + p, "r": SPIN_ROD_D / 2})
        else:
            out.append({"poly": triangle(c0 + p, SPIN_TRI_S, k * np.pi / 3 + np.pi)})
    return out


def valley_site_shapes(kind: str, size_a: float, size_b: float, offset=(0.0, 0.0)) -> list[dict]:
    """A (down-pointing / rod) and B (up-pointing / rod) inclusions of a valley cell."""
    o = np.asarray(offset, float)
    if kind == "valley_te":
        return [{"poly": triangle(SITE_A + o, size_a, -np.pi / 2)},
                {"poly": triangle(SITE_B + o, size_b, np.pi / 2)}]
    return [{"circle": SITE_A + o, "r": size_a}, {"circle": SITE_B + o, "r": size_b}]


def inclusion_material(kind: str) -> mp.Medium:
    return HOST if kind.endswith("_tm") else AIR


def background_material(kind: str) -> mp.Medium:
    return AIR if kind.endswith("_tm") else HOST


def to_objects(shapes: list[dict], material: mp.Medium, lattice: mp.Lattice | None = None) -> list:
    """Meep/MPB objects; with `lattice`, coordinates are converted to lattice coordinates."""
    conv = (lambda x, y, z=0.0: mp.cartesian_to_lattice(mp.Vector3(x, y, z), lattice)) if lattice \
        else (lambda x, y, z=0.0: mp.Vector3(x, y, z))
    out = []
    for sh in shapes:
        if "circle" in sh:
            out.append(mp.Cylinder(radius=sh["r"], material=material, center=conv(*sh["circle"])))
        elif lattice is not None:
            out.append(mp.Prism(vertices=[conv(x, y) for x, y in sh["poly"]], height=mp.inf, material=material))
        else:
            # vertices at z = -5 so the extruded prism (height 10) is centred on the 2D plane
            out.append(mp.Prism(vertices=[mp.Vector3(x, y, -5) for x, y in sh["poly"]], height=10,
                                material=material))
    return out


# --- unit cells -----------------------------------------------------------------------

def cell_shapes(kind: str, variant: str) -> list[dict]:
    """variant: spin 'trivial' | 'honeycomb' | 'topological'; valley 'vpc1' | 'vpc2' | 'symmetric'."""
    if kind.startswith("spin"):
        return spin_shapes(1 / SPIN_A0_OVER_R[variant], kind)
    return valley_site_shapes(kind, *valley_sizes(kind, variant))


def cell_solver(kind: str, variant: str, resolution: int = 32, num_bands: int = 8,
                k_points: list | None = None) -> mpb.ModeSolver:
    return mpb.ModeSolver(geometry_lattice=LATTICE,
                          geometry=to_objects(cell_shapes(kind, variant), inclusion_material(kind), LATTICE),
                          default_material=background_material(kind), resolution=resolution,
                          num_bands=num_bands, k_points=k_points or [GAMMA])


def run_pol(ms: mpb.ModeSolver, kind: str, *band_funcs):
    (ms.run_tm if POLARIZATION[kind] == "TM" else ms.run_te)(*band_funcs)


def cell_epsilon(kind: str, variant: str, resolution: int = 64) -> np.ndarray:
    ms = cell_solver(kind, variant, resolution, 1)
    ms.init_params(mp.TM if POLARIZATION[kind] == "TM" else mp.TE, False)
    return np.asarray(ms.get_epsilon())


def cell_corner_grid(n: int, m_shift: int = 0, n_shift: int = 0):
    """Cartesian corner grids (n+1, n+1) for pcolormesh of a cell_epsilon array, shifted by
    m_shift*a1 + n_shift*a2."""
    u = (np.arange(n + 1) - 0.5) / n - 0.5 + m_shift
    v = (np.arange(n + 1) - 0.5) / n - 0.5 + n_shift
    U, V = np.meshgrid(u, v, indexing="ij")
    return U * A1[0] + V * A2[0], U * A1[1] + V * A2[1]


def k_path(points: list, n_interp: int = 16) -> tuple[list, np.ndarray, list]:
    """k-points along a polyline, cumulative cartesian path length (2*pi/a) and corner ticks."""
    ks = mp.interpolate(n_interp, points)
    kc = np.array([[k.x, k.y] for k in (mp.reciprocal_to_cartesian(k, LATTICE) for k in ks)])
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(kc, axis=0), axis=1))])
    return ks, dist, list(dist[:: n_interp + 1])


def angular_weights(f_periodic: np.ndarray, k_cart: np.ndarray, radius: float, n_theta: int = 96) -> dict:
    """Weights of the angular classes m mod 6 = 0 (s), +-1 (p), +-2 (d), 3 (f) of a scalar
    field (Ez or Hz) on a ring of `radius` around the cell centre; the field is rebuilt
    from MPB's periodic Bloch part times exp(i k.r)."""
    from scipy.ndimage import map_coordinates
    n = f_periodic.shape[0]
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    x, y = radius * np.cos(th), radius * np.sin(th)
    u, v = x - y / np.sqrt(3), 2 * y / np.sqrt(3)
    idx = np.vstack([(u + 0.5) * n, (v + 0.5) * n])
    re = map_coordinates(f_periodic.real, idx, order=3, mode="grid-wrap")
    im = map_coordinates(f_periodic.imag, idx, order=3, mode="grid-wrap")
    f = (re + 1j * im) * np.exp(2j * np.pi * np.column_stack([x, y]) @ k_cart)
    c = np.abs(np.fft.fft(f)) ** 2
    m = np.fft.fftfreq(n_theta, 1 / n_theta).astype(int) % 6
    w = {cls: c[np.isin(m, ms)].sum() for cls, ms in (("s", [0]), ("p", [1, 5]), ("d", [2, 4]), ("f", [3]))}
    tot = sum(w.values())
    return {k: (v / tot if tot > 0 else np.nan) for k, v in w.items()}


def bulk_bands(kind: str, variant: str, k_points: list, resolution: int = 32, num_bands: int = 8,
               character: bool = False) -> dict:
    """Bulk bands (c/a0 or c/a) and, for spin crystals with character=True, the s/p/d/f
    weights of Ez (TM) / Hz (TE) on the ring of cluster sites."""
    ms = cell_solver(kind, variant, resolution, num_bands, k_points)
    weights = []

    def record(ms_, band):
        kc = mp.reciprocal_to_cartesian(ms_.current_k, LATTICE)
        get = ms_.get_efield if POLARIZATION[kind] == "TM" else ms_.get_hfield
        f = np.asarray(get(band, bloch_phase=False))[..., 0, 2]
        w = angular_weights(f, np.array([kc.x, kc.y]), 1 / SPIN_A0_OVER_R[variant])
        weights.append([w["s"], w["p"], w["d"], w["f"]])

    run_pol(ms, kind, *([record] if character else []))
    out = {"freqs": np.asarray(ms.all_freqs)}
    if character:
        out["weights"] = np.asarray(weights).reshape(len(k_points), num_bands, 4)
    return out


def complete_gaps(freqs: np.ndarray, min_width: float = 1e-4) -> list[dict]:
    gaps = []
    for b in range(freqs.shape[1] - 1):
        lo, hi = freqs[:, b].max(), freqs[:, b + 1].min()
        if hi - lo > min_width:
            gaps.append({"below_band": b + 1, "lo": lo, "hi": hi, "center": (lo + hi) / 2, "width": hi - lo})
    return gaps


# --- domain walls ---------------------------------------------------------------------
#
# Rows of unit cells run along x. Spin walls: row j at y = (j + 1/2) ROW_H, shifted by
# ((j + 1) mod 2)/2 + x_shift, topological (a0/R = 2.9) for j < 0, trivial (3.125) for j >= 0; the
# atom of row -1 at x = 0 (x_shift = 0) or row 0 (x_shift = 1/2) is centred on x = 0. Valley walls: cell row n at y0 = n ROW_H,
# shifted by ((n mod 2) + 1)/2 + x_shift, sites at y0 -/+ d/2; VPC1 above y = 0, VPC2 below,
# so the large inclusions face each other across the wall (bearded interface). x_shift =
# -1/2 puts a bearded pair at x = 0 (source in the upper bearded inclusion).

def wall_shapes(kind: str, x_min: float, x_max: float, n_rows: int, x_shift: float = 0.0,
                x_face: float | None = None) -> list[dict]:
    """Inclusions of a wall sample with n_rows rows per side and centre x in [x_min, x_max).
    x_face (TE strip feed): spin -- only atoms with centre >= x_face + 1/2; valley -- holes
    with centre >= x_face, clipped at the face."""
    out = []
    if kind.startswith("spin"):
        lo = x_min if x_face is None else max(x_min, x_face + 0.5 - 1e-9)
        for j in range(-n_rows, n_rows):
            R = 1 / SPIN_A0_OVER_R["topological" if j < 0 else "trivial"]
            off = ((j + 1) % 2) * 0.5 + x_shift
            for cx in np.arange(np.ceil(lo - off) + off, x_max, 1.0):
                out += spin_shapes(R, kind, (cx, (j + 0.5) * ROW_H))
        return out
    s_up, s_low = valley_sizes(kind, "vpc1"), valley_sizes(kind, "vpc2")
    lo = x_min if x_face is None else max(x_min, x_face - 1e-9)
    for n in range(-n_rows, n_rows):
        y0 = n * ROW_H
        off = ((n % 2) + 1) / 2 + x_shift
        for xc in np.arange(np.ceil(lo - off) + off, x_max, 1.0):
            # site A (k = 0, down-pointing) below site B (k = 1, up-pointing) on a vertical bond;
            # sizes from VPC1 above the wall, from VPC2 below it
            for k, c in enumerate((np.array([xc, y0 - D_NN / 2]), np.array([xc, y0 + D_NN / 2]))):
                size = (s_up if c[1] > 0 else s_low)[k]
                if kind == "valley_tm":
                    out.append({"circle": c, "r": size})
                else:
                    out.append({"poly": triangle(c, size, np.pi / 2 if k == 1 else -np.pi / 2)})
    if x_face is not None:
        from shapely.geometry import Polygon, box
        clipped = []
        for sh in out:
            v = sh["poly"]
            if v[:, 0].min() < x_face:
                cl = Polygon(v).intersection(box(x_face, -1e3, 1e3, 1e3))
                if cl.is_empty or cl.area < 1e-6:
                    continue
                v = np.asarray(cl.exterior.coords)[:-1]
            clipped.append({"poly": v})
        out = clipped
    return out


# Chiral source positions (FDTD frame of run_domain_wall, source_position x_shift). Valley:
# chiral_hotspot on the ribbon edge mode at the drive frequency (scripts/chirality_maps.py,
# data/walls/chirality_maps.npz). Spin: the largest FDTD directionality among high-|C| points
# on the mirror lines (scripts/scan_spin_sources.py, data/walls/spin_source_scan.npz); there
# the S0-maximizing chiral_hotspot point was less directional in FDTD (spin_tm 0.982).
CHIRAL_HOTSPOTS: dict[str, tuple[float, float]] = {
    "spin_tm": (0.0, 0.0),         # air, on the wall midway between two topological clusters
    "spin_te": (0.0, -0.3712),     # silicon, 0.06 above the centre of the topological cluster
    "valley_te": (-0.5, -0.6186),  # air, small VPC2 hole of the second row below the wall
    "valley_tm": (-0.5, 0.6186),   # silicon, small VPC1 rod of the second row above the wall
}


def source_position(kind: str, at: str = "default") -> tuple[tuple[float, float], float]:
    """(source xy, lattice x_shift). at='inclusion': spin -- atom centre next to the wall;
    valley -- centre of the upper bearded inclusion (a C3 point that is not a twofold-rotation
    centre of the wall). at='hexcentre' (valley): hexagon centre on the wall, a C2 centre.
    at='default' / 'hotspot': CHIRAL_HOTSPOTS, the edge-mode chirality maximum (chiral_hotspot)."""
    x_shift = 0.0 if kind.startswith("spin") else -0.5
    if at in ("default", "hotspot"):
        return CHIRAL_HOTSPOTS[kind], x_shift
    if at == "hexcentre":
        return (0.0, 0.0), 0.0
    if kind.startswith("spin"):
        return (0.0, -0.5 * ROW_H), x_shift
    return (0.0, D_NN / 2), x_shift


def ribbon_bands(kind: str, kxs: np.ndarray, n_rows: int = 8, resolution: int = 32,
                 num_bands: int = 48) -> dict:
    """Periodic ribbon (one period along x, 2 n_rows rows): bands versus kx (2*pi/a) and,
    per band, the share of |Ez|^2 (TM) / |Hz|^2 (TE) within one row height of the wall at
    y = 0 and of the wrap-around wall at y = +-Ly/2."""
    Ly = 2 * n_rows * ROW_H
    shapes = wall_shapes(kind, -0.5, 0.5, n_rows)
    ms = mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(1, Ly)),
                        geometry=to_objects(shapes, inclusion_material(kind)),
                        default_material=background_material(kind), resolution=resolution,
                        num_bands=num_bands, k_points=[mp.Vector3(k) for k in kxs])
    loc = []

    def record(ms_, band):
        get = ms_.get_efield if POLARIZATION[kind] == "TM" else ms_.get_hfield
        u = np.abs(np.asarray(get(band, bloch_phase=False))[..., 0, 2]) ** 2
        ny = u.shape[1]
        y = (np.arange(ny) / ny - 0.5) * Ly
        prof = u.sum(axis=0)
        loc.append([prof[np.abs(y) < ROW_H].sum() / prof.sum(),
                    prof[Ly / 2 - np.abs(y) < ROW_H].sum() / prof.sum()])

    run_pol(ms, kind, record)
    return {"kx": np.asarray(kxs), "freqs": np.asarray(ms.all_freqs),
            "loc": np.asarray(loc).reshape(len(kxs), num_bands, 2)}


def wall_edge_mode(kind: str, freq: float, kx_guess: float, n_rows: int = 8, resolution: int | None = None,
                   x_shift: float | None = None, dk: float = 0.005, deg_tol: float = 2e-3) -> dict:
    """In-plane field of the wall edge mode at the drive frequency, on a periodic ribbon in the
    FDTD frame (same x_shift as `run_domain_wall`): E (TE) or H (TM), the field a circular
    dipole of that polarization couples to. kx is found by linear interpolation of the
    wall-localized band through `freq` on kx_guess + (-dk, 0, dk). The ribbon wraps round, so it
    has a second wall at y = +-Ly/2; modes within deg_tol of each other (spin-Hall: the two
    walls' modes are nearly degenerate and mix) are recombined into the one localized on y = 0
    (largest eigenvector of the |F|^2 weight within one row height of the wall).
    Returns x, y (cartesian grid), eps, F (nx, ny, 2) complex, kx, f, vg sign."""
    res = resolution or WALL_DEFAULTS[kind]["resolution"]
    xs = source_position(kind, "inclusion")[1] if x_shift is None else x_shift
    Ly = 2 * n_rows * ROW_H
    geom = to_objects(wall_shapes(kind, -0.5, 0.5, n_rows, xs), inclusion_material(kind))
    get = "get_efield" if POLARIZATION[kind] == "TE" else "get_hfield"

    def solve(kxs, num_bands, keep_fields):
        ms = mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(1, Ly)), geometry=geom,
                            default_material=background_material(kind), resolution=res,
                            num_bands=num_bands, k_points=[mp.Vector3(k) for k in kxs])
        fields = []

        def rec(ms_, band):
            if keep_fields:
                fields.append(np.asarray(getattr(ms_, get)(band, bloch_phase=False))[:, :, 0, :])

        run_pol(ms, kind, rec)
        return ms, np.asarray(ms.all_freqs).reshape(len(kxs), num_bands), fields

    # band count: enough to reach freq with margin (one extra band per 0.01 c/a of 2 Ly lines)
    _, f0s, _ = solve([kx_guess], 8, False)
    nb = 8
    while f0s[0, -1] < freq * 1.05:
        nb += 8
        _, f0s, _ = solve([kx_guess], nb, False)
    kxs = [kx_guess - dk, kx_guess, kx_guess + dk]
    _, fk, _ = solve(kxs, nb, False)
    # bands crossing freq on the scan; their kx at freq
    cross = []
    for b in range(nb):
        f = fk[:, b]
        for i in range(2):
            if (f[i] - freq) * (f[i + 1] - freq) <= 0 and f[i + 1] != f[i]:
                cross.append((kxs[i] + (freq - f[i]) / (f[i + 1] - f[i]) * dk, b, np.sign(f[i + 1] - f[i])))
    if not cross:
        raise RuntimeError(f"no band crosses f = {freq} near kx = {kx_guess}")
    # the crossing nearest the guess and any (near-degenerate) partner within 0.002; other
    # crossings in the window belong to the wrap-around wall
    k_near = min(cross, key=lambda c: abs(c[0] - kx_guess))[0]
    cross = [c for c in cross if abs(c[0] - k_near) < 2e-3]
    kx = float(np.mean([c[0] for c in cross]))
    ms, fq, fields = solve([kx], nb, True)
    bands = [b for b in range(nb) if abs(fq[0, b] - freq) < deg_tol]
    eps = np.asarray(ms.get_epsilon())
    nx, ny = eps.shape
    y = (np.arange(ny) / ny - 0.5) * Ly
    wall = (np.abs(y) < ROW_H)[None, :, None]
    Fs = [fields[b] for b in bands]
    W = np.array([[np.vdot(Fb * wall, Fc * wall) for Fc in Fs] for Fb in Fs])
    w, v = np.linalg.eigh(W)
    F = sum(c * Fb for c, Fb in zip(v[:, -1], Fs))
    total = sum(np.vdot(Fb, Fb).real for Fb in Fs) / len(Fs)
    vg = [c[2] for c in cross]
    return {"x": np.arange(nx) / nx - 0.5, "y": y, "eps": eps, "F": F[..., :2], "kx": kx,
            "f": float(np.mean(fq[0, bands])), "vg_sign": float(np.sign(np.sum(vg))),
            "wall_share": float(w[-1] / total), "n_combined": len(bands)}


def chirality(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S0 = |Fx|^2 + |Fy|^2 and C = 2 Im(Fx* Fy) / S0 of an in-plane field (..., 2)."""
    s0 = np.abs(F[..., 0]) ** 2 + np.abs(F[..., 1]) ** 2
    return s0, 2 * np.imag(np.conj(F[..., 0]) * F[..., 1]) / np.maximum(s0, 1e-30)


def chiral_hotspot(mode: dict, c_min: float = 0.98, edge_px: int = 2, y_max: float = 1.5 * ROW_H) -> dict:
    """Chiral source position: on a vertical mirror line of the wall (x = 0 or -1/2, so the
    mirror check P_left(S+) = P_right(S-) holds), within y_max of the wall, at least edge_px
    pixels from a dielectric interface (in-plane E jumps across it) and with |C| >= c_min, the
    point of largest S0, i.e. strongest coupling to the edge mode. A weakly coupled source
    loses directionality to bulk radiation, which leaves in both directions."""
    from scipy.ndimage import binary_erosion
    s0, C = chirality(mode["F"])
    eps, x = mode["eps"], mode["x"]
    solid = eps > 0.5 * (eps.max() + eps.min())
    st = np.ones((2 * edge_px + 1, 2 * edge_px + 1), bool)
    # periodic in x, so pad by wrapping before eroding both phases
    pad = lambda a: np.pad(a, ((edge_px, edge_px), (0, 0)), mode="wrap")
    inner = (binary_erosion(pad(solid), st) | binary_erosion(pad(~solid), st))[edge_px:-edge_px]
    near = np.broadcast_to(np.abs(mode["y"])[None, :] <= y_max, s0.shape)
    mirror = np.zeros_like(near)
    mirror[[np.argmin(np.abs(x)), np.argmin(np.abs(x + 0.5))], :] = True
    ok = near & mirror & inner & (np.abs(C) >= c_min)
    i, j = np.unravel_index(np.argmax(np.where(ok, s0, -1)), C.shape)
    return {"xy": (float(x[i]), float(mode["y"][j])), "C": float(C[i, j]),
            "s0_rel": float(s0[i, j] / s0[near].max()), "in_dielectric": bool(solid[i, j]),
            "ij": (int(i), int(j))}


def wall_mirror_parity(kind: str, kx: float, n_rows: int = 8, resolution: int = 32, num_bands: int = 24,
                       f_window: tuple = (0.0, 1.0)) -> list[dict]:
    """Ribbon modes at one kx with <F|M F>/<F|F> for the mirror y -> -y about the wall line,
    F = Ez (TM) or Hz (TE); +1 even, -1 odd."""
    Ly = 2 * n_rows * ROW_H
    ms = mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(1, Ly)),
                        geometry=to_objects(wall_shapes(kind, -0.5, 0.5, n_rows), inclusion_material(kind)),
                        default_material=background_material(kind), resolution=resolution,
                        num_bands=num_bands, k_points=[mp.Vector3(kx)])
    out = []

    def rec(ms_, band):
        get = ms_.get_efield if POLARIZATION[kind] == "TM" else ms_.get_hfield
        f = np.asarray(get(band, bloch_phase=False))[..., 0, 2]
        ny = f.shape[1]
        y = (np.arange(ny) / ny - 0.5) * Ly
        prof = (np.abs(f) ** 2).sum(0)
        jm = np.round((-y / Ly + 0.5) * ny).astype(int) % ny
        out.append({"f": float(ms_.freqs[band - 1]),
                    "wall_share": float(prof[np.abs(y) < ROW_H].sum() / prof.sum()),
                    "mirror": float(np.real(np.vdot(f, f[:, jm]) / np.vdot(f, f)))})

    run_pol(ms, kind, rec)
    return [o for o in out if f_window[0] < o["f"] < f_window[1]]


# Converged settings (S+ directionality within ~0.01, bulk loss < 1 %): spin TM 15 rows (10 rows
# lost 0.7 % through the bulk); spin TE 20 rows and resolution 64 (the drive frequency sits
# 0.0036 above the gap edge: eta 0.834 / 0.980 / 0.990 at resolution 32 / 48 / 64, and 15 % bulk
# loss at 10 rows); valley walls unchanged at resolution 36. scripts/scan_spin_rows.py.
WALL_DEFAULTS = {
    "spin_tm": dict(n_x=20, n_rows=15, resolution=32, dpml=3.0, pad_y=1.0, run_time=300.0),
    "spin_te": dict(n_x=20, n_rows=20, resolution=64, dpml=3.0, pad_y=1.0, run_time=300.0),
    "valley_te": dict(n_x=35, n_rows=17, resolution=24, dpml=4.0, pad_y=1.5, run_time=500.0),
    "valley_tm": dict(n_x=35, n_rows=17, resolution=24, dpml=4.0, pad_y=1.5, run_time=500.0),
}


def run_domain_wall(kind: str, spin: int, freq: float, source_at: str | tuple = "default", **kw) -> dict:
    """2D Meep CW run of a wall sample driven by a circularly polarized dipole: in-plane H,
    Hx + i*spin*Hy, for TM; in-plane E, Ex + i*spin*Ey, for TE; source_at is a
    source_position label or an explicit ((x, y), x_shift). The lattice continues
    through the x PML; along y the rows are followed by pad_y of background and PML.
    Returns complex fields (phasors) of all three field components, eps, and the
    time-averaged Poynting vector."""
    p = {**WALL_DEFAULTS[kind], **kw}
    sx = p["n_x"] + 2 * p["dpml"]
    sy = 2 * p["n_rows"] * ROW_H + 2 * (p["pad_y"] + p["dpml"])
    (sx_src, sy_src), x_shift = source_at if isinstance(source_at, tuple) else source_position(kind, source_at)
    shapes = wall_shapes(kind, -sx / 2 - 1, sx / 2 + 1, p["n_rows"], x_shift)
    src = mp.ContinuousSource(frequency=freq, width=20)
    c1, c2 = (mp.Hx, mp.Hy) if POLARIZATION[kind] == "TM" else (mp.Ex, mp.Ey)
    sources = [mp.Source(src, component=c1, center=mp.Vector3(sx_src, sy_src), amplitude=1.0),
               mp.Source(src, component=c2, center=mp.Vector3(sx_src, sy_src), amplitude=1j * spin)]
    sim = mp.Simulation(cell_size=mp.Vector3(sx, sy), geometry=to_objects(shapes, inclusion_material(kind)),
                        default_material=background_material(kind), sources=sources,
                        boundary_layers=[mp.PML(p["dpml"])], resolution=p["resolution"],
                        force_complex_fields=True)
    sim.run(until=p["run_time"])
    vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(sx, sy))
    comps = (mp.Ez, mp.Hx, mp.Hy) if POLARIZATION[kind] == "TM" else (mp.Hz, mp.Ex, mp.Ey)
    f = {name: sim.get_array(component=c, vol=vol) for name, c in zip(
        ("ez", "hx", "hy") if POLARIZATION[kind] == "TM" else ("hz", "ex", "ey"), comps)}
    if POLARIZATION[kind] == "TM":
        Sx, Sy = -0.5 * np.real(f["ez"] * np.conj(f["hy"])), 0.5 * np.real(f["ez"] * np.conj(f["hx"]))
        e2 = np.abs(f["ez"]) ** 2
    else:
        Sx, Sy = 0.5 * np.real(f["ey"] * np.conj(f["hz"])), -0.5 * np.real(f["ex"] * np.conj(f["hz"]))
        e2 = np.abs(f["ex"]) ** 2 + np.abs(f["ey"]) ** 2
    nx, ny = e2.shape
    return {**f, "E2": e2, "Sx": Sx, "Sy": Sy, "eps": sim.get_array(component=mp.Dielectric, vol=vol),
            "x": np.linspace(-sx / 2, sx / 2, nx), "y": np.linspace(-sy / 2, sy / 2, ny),
            "src_xy": np.array([sx_src, sy_src]), "freq": freq, "spin": spin, **p}


def box_fluxes(o: dict, x_line: float, dpml: float) -> dict:
    """Time-averaged power through a closed box: vertical lines at x = +-x_line and
    horizontal lines just inside the y PML (left/right positive outward)."""
    x, y = o["x"], o["y"]
    dx, dy = x[1] - x[0], y[1] - y[0]
    y_in = y.max() - dpml - 0.2
    jy = np.abs(y) <= y_in
    il, ir = np.argmin(np.abs(x - (o["src_xy"][0] - x_line))), np.argmin(np.abs(x - (o["src_xy"][0] + x_line)))
    jx = (x >= x[il]) & (x <= x[ir])
    jb, jt = np.argmin(np.abs(y + y_in)), np.argmin(np.abs(y - y_in))
    return {"left": -o["Sx"][il, jy].sum() * dy, "right": o["Sx"][ir, jy].sum() * dy,
            "bottom": -o["Sy"][jx, jb].sum() * dx, "top": o["Sy"][jx, jt].sum() * dx}


# --- strip waveguide -> TE wall ---------------------------------------------------------

W_STRIP_NM = 500.0

COUPLING_DEFAULTS = {
    "spin_te": dict(n_len=15, n_rows=10, resolution=32, l_wg=4.0, dpml=2.0, dpml_right=3.0, pad_y=1.0,
                    fcen=0.408, df=0.05, field_freqs=(0.405, 0.408, 0.4103), y_wall=3 * ROW_H,
                    min_run=500.0, max_run=1500.0),
    "valley_te": dict(n_len=25, n_rows=17, resolution=24, l_wg=4.0, dpml=3.0, dpml_right=4.0, pad_y=1.5,
                      fcen=0.2368, df=0.04, field_freqs=(0.232, 0.2368, 0.2415), y_wall=4.5,
                      min_run=600.0, max_run=3000.0),
}


def strip_width(kind: str) -> float:
    return W_STRIP_NM / (A0_NM if kind.startswith("spin") else A_NM)


def coupler_geometry(kind: str, w: float, x_left: float, x_right: float, n_rows: int,
                     fill_len: float = 0.0) -> list:
    """Host block for x >= 0, strip (width w, air cladding) from x_left to the face at x = 0,
    and the wall's holes (spin: whole atoms with centre >= 1/2; valley: lattice shifted by
    -1/2, holes clipped at the face). fill_len (valley): bearded-pair holes omitted over
    the first fill_len cells (line-defect coupler)."""
    geom = [mp.Block(size=mp.Vector3(x_right + 1, mp.inf), material=HOST, center=mp.Vector3((x_right + 1) / 2, 0)),
            mp.Block(size=mp.Vector3(-x_left + 1, w), material=HOST, center=mp.Vector3((x_left - 1) / 2, 0))]
    x_shift = 0.0 if kind.startswith("spin") else -0.5
    shapes = wall_shapes(kind, x_left, x_right + 1, n_rows, x_shift, x_face=0.0)
    if fill_len > 0:
        shapes = [sh for sh in shapes if not (abs(abs(sh["poly"].mean(0)[1]) - D_NN / 2) < 1e-6
                                              and sh["poly"].mean(0)[0] < fill_len - 1e-9)]
    return geom + to_objects(shapes, AIR)


def run_wg_coupling(kind: str, w: float | None = None, design: dict | None = None, fill_len: float = 0.0,
                    eig_band: int = 1, nfreq: int = 101, **kw) -> dict:
    """Strip mode `eig_band` (1 = fundamental) as a Gaussian pulse into the TE wall.

    Spectra normalized to the incident mode power: R (same mode back), back (all power
    returning through the input plane), T_wall (|y| < y_wall at x_t), T_all (full height),
    top / bottom (lines inside the y PML). `design` = {"weights", "center", "size"} is
    inserted last as a binary level set with subpixel smoothing (MaterialGrid beta = inf,
    averaging)."""
    p = {**COUPLING_DEFAULTS[kind], **kw}
    w = strip_width(kind) if w is None else w
    x_left, x_right = -p["l_wg"] - p["dpml"], p["n_len"] + p["dpml_right"]
    sx, cx = x_right - x_left, (x_right + x_left) / 2
    sy = 2 * (p["n_rows"] * ROW_H + p["pad_y"] + p["dpml"])
    x_t = p.get("x_t", p["n_len"] - 2.0)
    geom = coupler_geometry(kind, w, x_left, x_right, p["n_rows"], fill_len)
    if design is not None:
        wts = np.asarray(design["weights"], float)
        mg = mp.MaterialGrid(mp.Vector3(*wts.shape), mp.air, HOST, weights=wts, do_averaging=True, beta=np.inf)
        geom.append(mp.Block(center=mp.Vector3(*design["center"]), size=mp.Vector3(*design["size"]), material=mg))
    dpml, fcen, df = p["dpml"], p["fcen"], p["df"]
    x_src, x_r = x_left + dpml + 0.5, x_left + dpml + 1.5
    src = [mp.EigenModeSource(mp.GaussianSource(fcen, fwidth=df), center=mp.Vector3(x_src, 0),
                              size=mp.Vector3(0, sy - 2 * dpml), direction=mp.X, eig_band=eig_band,
                              eig_parity=mp.TE)]
    sim = mp.Simulation(cell_size=mp.Vector3(sx, sy), geometry_center=mp.Vector3(cx, 0), geometry=geom,
                        sources=src, resolution=p["resolution"],
                        boundary_layers=[mp.PML(dpml, direction=mp.Y), mp.PML(dpml, direction=mp.X, side=mp.Low),
                                         mp.PML(p["dpml_right"], direction=mp.X, side=mp.High)])
    y_in = sy / 2 - dpml - 0.2
    refl = sim.add_mode_monitor(fcen, df, nfreq, mp.ModeRegion(center=mp.Vector3(x_r, 0), size=mp.Vector3(0, sy - 2 * dpml)))
    back = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(x_r, 0), size=mp.Vector3(0, 2 * y_in)))
    tw = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(x_t, 0), size=mp.Vector3(0, 2 * p["y_wall"])))
    ta = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(x_t, 0), size=mp.Vector3(0, 2 * y_in)))
    span, mid = mp.Vector3(x_t - x_r, 0), (x_t + x_r) / 2
    top = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(mid, y_in), size=span))
    bot = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(mid, -y_in), size=span, weight=-1))
    fvol = mp.Volume(center=mp.Vector3(cx, 0), size=mp.Vector3(sx - dpml - p["dpml_right"], sy - 2 * dpml))
    dft = sim.add_dft_fields([mp.Hz], list(p["field_freqs"]), where=fvol)
    sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-6, minimum_run_time=p["min_run"],
                                                         maximum_run_time=p["max_run"]))
    freqs = np.asarray(mp.get_flux_freqs(ta))
    coef = sim.get_eigenmode_coefficients(refl, [eig_band], eig_parity=mp.TE).alpha[0]
    p_inc = np.abs(coef[:, 0]) ** 2
    return {"freqs": freqs, "p_inc": p_inc, "R": np.abs(coef[:, 1]) ** 2 / p_inc,
            "r_coef": coef[:, 1] / coef[:, 0],
            "back": 1 - np.asarray(mp.get_fluxes(back)) / p_inc,
            "T_wall": np.asarray(mp.get_fluxes(tw)) / p_inc, "T_all": np.asarray(mp.get_fluxes(ta)) / p_inc,
            "top": np.asarray(mp.get_fluxes(top)) / p_inc, "bottom": np.asarray(mp.get_fluxes(bot)) / p_inc,
            "field_freqs": np.asarray(p["field_freqs"]),
            "hz": np.stack([sim.get_dft_array(dft, mp.Hz, i) for i in range(len(p["field_freqs"]))]),
            "eps": sim.get_array(component=mp.Dielectric, vol=fvol),
            "extent": np.array([cx - (sx - dpml - p["dpml_right"]) / 2, cx + (sx - dpml - p["dpml_right"]) / 2,
                                -(sy / 2 - dpml), sy / 2 - dpml]),
            "w": w, "fill_len": fill_len, "eig_band": eig_band, "x_t": x_t, "t_end": sim.meep_time()}


def run_straight_strip(w: float, length: float = 16.0, resolution: int = 32, fcen: float = 0.408,
                       df: float = 0.05, nfreq: int = 51, dpml: float = 2.0) -> dict:
    """Straight strip (eps_d in air): fundamental TE mode power at two planes `length` apart
    (mode coefficients and flux), effective index of the launched mode, Hz cross-section."""
    sx, sy = length + 2 * dpml + 3, 6.0 + 2 * dpml
    x1, x2 = -sx / 2 + dpml + 1.5, sx / 2 - dpml - 1.0
    sim = mp.Simulation(cell_size=mp.Vector3(sx, sy), resolution=resolution, boundary_layers=[mp.PML(dpml)],
                        geometry=[mp.Block(size=mp.Vector3(mp.inf, w), material=HOST)],
                        sources=[mp.EigenModeSource(mp.GaussianSource(fcen, fwidth=df), center=mp.Vector3(x1 - 0.5),
                                                    size=mp.Vector3(0, sy - 2 * dpml), eig_band=1, eig_parity=mp.TE)])
    regs = [mp.ModeRegion(center=mp.Vector3(x), size=mp.Vector3(0, sy - 2 * dpml)) for x in (x1, x2)]
    m1, m2 = (sim.add_mode_monitor(fcen, df, nfreq, r) for r in regs)
    f1, f2 = (sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=r.center, size=r.size)) for r in regs)
    dft = sim.add_dft_fields([mp.Hz], [fcen], center=mp.Vector3(x2), size=mp.Vector3(0, sy - 2 * dpml))
    sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-8, minimum_run_time=100))
    freqs = np.asarray(mp.get_flux_freqs(f1))
    r1 = sim.get_eigenmode_coefficients(m1, [1], eig_parity=mp.TE)
    r2 = sim.get_eigenmode_coefficients(m2, [1], eig_parity=mp.TE)
    hz = sim.get_dft_array(dft, mp.Hz, 0)
    return {"freqs": freqs, "mode_in": np.abs(r1.alpha[0, :, 0]) ** 2, "mode_out": np.abs(r2.alpha[0, :, 0]) ** 2,
            "flux_in": np.asarray(mp.get_fluxes(f1)), "flux_out": np.asarray(mp.get_fluxes(f2)),
            "neff": np.array([v.x for v in r1.kpoints]) / freqs, "hz_profile": hz,
            "y": np.linspace(-(sy / 2 - dpml), sy / 2 - dpml, len(hz)), "w": w, "length": x2 - x1}


def slab_modes(w: float, f: float, n1: float = np.sqrt(EPS_D), n0: float = 1.0) -> list[tuple]:
    """Guided TE (Hz) modes of a symmetric slab of width w: [(parity, n_eff), ...]."""
    from scipy.optimize import brentq
    k0, out = 2 * np.pi * f, []
    for parity in ("even", "odd"):
        def g(neff):
            kap, gam = k0 * np.sqrt(n1 ** 2 - neff ** 2), k0 * np.sqrt(neff ** 2 - n0 ** 2)
            lhs = kap * np.tan(kap * w / 2) if parity == "even" else -kap / np.tan(kap * w / 2)
            return lhs - (n1 / n0) ** 2 * gam
        grid = np.linspace(n0 + 1e-6, n1 - 1e-6, 4000)
        v = np.array([g(n) for n in grid])
        out += [(parity, brentq(g, grid[i], grid[i + 1])) for i in range(len(grid) - 1)
                if np.sign(v[i]) != np.sign(v[i + 1]) and abs(v[i]) < 1e3 and abs(v[i + 1]) < 1e3]
    return sorted(out, key=lambda t: -t[1])


def geometry_mirror_check(kind: str, n_rows: int = 6) -> float:
    """Largest mismatch between the y-mirrored wall and the wall itself (0: mirror-symmetric)."""
    shapes = wall_shapes(kind, -4, 4, n_rows)
    keys = [np.sort(sh["poly"].round(6), axis=0) if "poly" in sh else np.array([*sh["circle"], sh["r"]])
            for sh in shapes]
    worst = 0.0
    for sh in shapes:
        if "poly" in sh:
            m = sh["poly"] * [1, -1]
            c = m.mean(0)
            key = np.sort(m, axis=0)
        else:
            c = sh["circle"] * [1, -1]
            key = np.array([*c, sh["r"]])
        if abs(c[0]) > 3 or abs(c[1]) > (n_rows - 1) * ROW_H:
            continue
        worst = max(worst, min(np.abs(key - k).max() for k in keys if k.shape == key.shape))
    return worst
