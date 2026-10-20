"""a-Si-overlay grating coupler on SiN in Meep 2D x-z (block geometry, TE and TM).

**Modeling assumption (deviates from the toolkit's 2D effective-index default).** The real
x-z layer stack with bulk indices at 1.31 um (those of Fraser et al., Sci. Rep. 2024),
non-dispersive, y-invariant. Used by `meta/notebooks/04`; parameter rationale in
`docs/simulation_settings_record.md`.

Cross-section (physical x = propagation, z = vertical; Meep's (x, y) = physical (x, z)),
bottom to top: Si substrate / BOX / SiN core / oxide buffer / a-Si grating teeth (full
etch, only over the grating) / top cladding. The input SiN waveguide carries no a-Si.

Polarization in this rotated-axis convention (same as pic_toolkit.meep_sim.grating_coupler):
- physical TE (E along y, the invariant axis) = Meep `mp.TM` parity: Ez(Meep) = -Ey, Hx;
- physical TM (H along y)                    = Meep `mp.TE` parity: Hz(Meep) = -Hy, Ex.

The overlap, fiber model and fiber-position search are the validated ones from
`pic_toolkit.meep_sim.sin_grating` (same definition of eta), extended here to TM.
"""

from __future__ import annotations

import math
import time

import meep as mp
import numpy as np

from pic_toolkit.meep_sim import sin_grating as sg

DEFAULT = {
    "wl0_um": 1.31, "wl_min_um": 1.21, "wl_max_um": 1.41, "n_freq": 41,
    # Fraser et al. (2024) indices at 1.31 um
    "n_core": 2.0017, "n_box": 1.4467, "n_buf": 1.4502, "n_asi": 3.5187, "n_top": 1.0, "n_sub": 3.5,
    "core_um": 0.40, "buf_um": 0.05, "asi_um": 0.22, "box_um": 4.5, "top_um": 2.5, "sub_um": 1.0,
    "period_um": 0.70, "duty": 0.28, "n_periods": 31,
    "fiber_angle_deg": -4.0, "mfd_um": 9.2, "pol": "TE",
    "resolution": 60, "dpml_um": 1.0, "lead_um": 3.0, "tail_um": 2.0, "z_mon_up_offset_um": 1.0,
    "mode_span_um": 1.0, "run_time_factor": 2.0, "eps_averaging": True,
}


def resolve(p):
    return {**DEFAULT, **(p or {})}


def domain(p):
    x_g0, x_g1 = 0.0, p["n_periods"] * p["period_um"]
    z_core1 = p["box_um"] + p["core_um"]
    z_buf1 = z_core1 + p["buf_um"]
    z_asi1 = z_buf1 + p["asi_um"]
    z_top = z_asi1 + p["top_um"]
    x_left, x_right = x_g0 - p["lead_um"], x_g1 + p["tail_um"]
    res, dpml = p["resolution"], p["dpml_um"]
    return dict(
        x_g0=x_g0, x_g1=x_g1, x_src=x_left + 0.5, x_refl=x_left + 1.0, x_trans=x_right - 0.5,
        x_left=x_left, x_right=x_right, z_box1=p["box_um"], z_core1=z_core1, z_buf1=z_buf1, z_asi1=z_asi1,
        z_top=z_top, z_sub0=-p["sub_um"], z_up=z_asi1 + p["z_mon_up_offset_um"], z_down=-0.5 * p["sub_um"],
        z_mode=(p["box_um"] + z_core1) / 2, mode_h=p["core_um"] + 2 * p["mode_span_um"],
        cell_x=round(((x_right - x_left) + 2 * dpml) * res) / res,
        cell_z=round(((z_top + p["sub_um"]) + 2 * dpml) * res) / res,
        x_c=(x_left + x_right) / 2, z_c=(z_top - p["sub_um"]) / 2,
    )


def geometry(p, d):
    """Substrate, BOX, SiN core and buffer over the whole length; a-Si teeth over the grating.
    Each period starts with its a-Si tooth (width duty * period)."""
    big = d["cell_x"] + 10
    h_sub = p["sub_um"] + p["dpml_um"] + 1.0
    m = lambda n: mp.Medium(index=n)
    geo = [mp.Block(size=mp.Vector3(big, h_sub), center=mp.Vector3(d["x_c"], -h_sub / 2), material=m(p["n_sub"])),
           mp.Block(size=mp.Vector3(big, p["box_um"]), center=mp.Vector3(d["x_c"], p["box_um"] / 2), material=m(p["n_box"])),
           mp.Block(size=mp.Vector3(big, p["core_um"]), center=mp.Vector3(d["x_c"], (d["z_box1"] + d["z_core1"]) / 2),
                    material=m(p["n_core"])),
           mp.Block(size=mp.Vector3(big, p["buf_um"]), center=mp.Vector3(d["x_c"], (d["z_core1"] + d["z_buf1"]) / 2),
                    material=m(p["n_buf"]))]
    w = p["duty"] * p["period_um"]
    for k in range(p["n_periods"]):
        x0 = k * p["period_um"]
        geo.append(mp.Block(size=mp.Vector3(w, p["asi_um"]), center=mp.Vector3(x0 + w / 2, (d["z_buf1"] + d["z_asi1"]) / 2),
                            material=m(p["n_asi"])))
    return geo


@sg.silent
def run_forward(p=None, field_map=False):
    """Broadband run for one polarization: R, T, up, down, near field on the up line,
    incident power. Same outputs as sin_grating.run_forward, plus 'pol'."""
    p = resolve(p)
    d = domain(p)
    te = p["pol"] == "TE"
    parity = mp.TM if te else mp.TE
    f0 = 1 / p["wl0_um"]
    fw = 2 * max(f0 - 1 / p["wl_max_um"], 1 / p["wl_min_um"] - f0)
    nf = p["n_freq"]
    src = mp.EigenModeSource(src=mp.GaussianSource(frequency=f0, fwidth=fw), center=mp.Vector3(d["x_src"], d["z_mode"]),
                             size=mp.Vector3(0, d["mode_h"]), eig_band=1, eig_parity=parity, eig_match_freq=True,
                             eig_kpoint=mp.Vector3(1, 0, 0))
    sim = mp.Simulation(cell_size=mp.Vector3(d["cell_x"], d["cell_z"]), boundary_layers=[mp.PML(p["dpml_um"])],
                        geometry=geometry(p, d), sources=[src], default_material=mp.Medium(index=p["n_top"]),
                        resolution=p["resolution"], geometry_center=mp.Vector3(d["x_c"], d["z_c"]),
                        eps_averaging=p["eps_averaging"])
    mode_vol = lambda x: mp.Volume(center=mp.Vector3(x, d["z_mode"]), size=mp.Vector3(0, d["mode_h"]))
    m_r = sim.add_mode_monitor(f0, fw, nf, mp.ModeRegion(volume=mode_vol(d["x_refl"])))
    m_t = sim.add_mode_monitor(f0, fw, nf, mp.ModeRegion(volume=mode_vol(d["x_trans"])))
    line = mp.Volume(center=mp.Vector3(d["x_c"], d["z_up"]), size=mp.Vector3(d["x_right"] - d["x_left"], 0))
    f_up = sim.add_flux(f0, fw, nf, mp.FluxRegion(volume=line))
    f_dn = sim.add_flux(f0, fw, nf, mp.FluxRegion(center=mp.Vector3(d["x_c"], d["z_down"]),
                                                  size=mp.Vector3(d["x_right"] - d["x_left"], 0)))
    comps = [mp.Ez, mp.Hx] if te else [mp.Hz, mp.Ex]
    near = sim.add_dft_fields(comps, f0, fw, nf, where=line)
    fmap = sim.add_dft_fields([comps[0]], f0, 0, 1, center=mp.Vector3(d["x_c"], d["z_c"]),
                              size=mp.Vector3(d["cell_x"] - 2 * p["dpml_um"], d["cell_z"] - 2 * p["dpml_um"])) if field_map else None
    t0 = time.perf_counter()
    sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-6, minimum_run_time=p["run_time_factor"] * p["n_core"] * d["cell_x"]))
    freqs = np.array(mp.get_flux_freqs(f_up))
    a_r = sim.get_eigenmode_coefficients(m_r, [1], eig_parity=parity).alpha[0]
    a_t = sim.get_eigenmode_coefficients(m_t, [1], eig_parity=parity).alpha[0]
    p_in = np.abs(a_r[:, 0]) ** 2
    c0 = np.stack([sim.get_dft_array(near, comps[0], i) for i in range(nf)])
    c1 = np.stack([sim.get_dft_array(near, comps[1], i) for i in range(nf)])
    out = dict(params=p, dom=d, freqs=freqs, wl=1 / freqs, p_in=p_in, R=np.abs(a_r[:, 1]) ** 2 / p_in,
               T=np.abs(a_t[:, 0]) ** 2 / p_in, up=np.array(mp.get_fluxes(f_up)) / p_in,
               down=-np.array(mp.get_fluxes(f_dn)) / p_in, x_up=np.asarray(sim.get_array_metadata(dft_cell=near)[0]),
               wall_s=time.perf_counter() - t0, pol=p["pol"])
    if te:   # physical (Ey, Hx) with Ey = -Ez(Meep): the form sin_grating's overlap expects (ez, hx)
        out["ez_up"], out["hx_up"] = c0, c1
    else:    # physical TM: Hy = -Hz(Meep), Ex = Ex(Meep); see eta_tm_overlap
        out["hz_up"], out["ex_up"] = c0, c1
    if field_map:
        out["field_map"] = sim.get_dft_array(fmap, comps[0], 0)
        out["eps_map"] = sim.get_array(center=mp.Vector3(d["x_c"], d["z_c"]), component=mp.Dielectric,
                                       size=mp.Vector3(d["cell_x"] - 2 * p["dpml_um"], d["cell_z"] - 2 * p["dpml_um"]))
    return out


def slab_neff_tmm(layers, n_bot, n_top, wl, pol="TE"):
    """Fundamental TE or TM n_eff of a planar stack (exact transfer matrix). `layers`:
    (index, thickness_um) from bottom to top between semi-infinite claddings. TM uses the
    tangential pair (Hy, (1/eps) dHy/dz)."""
    from scipy.optimize import brentq
    k0 = 2 * np.pi / wl
    w = (lambda n: 1.0) if pol == "TE" else (lambda n: 1.0 / n**2)

    def disp(neff):
        gb, gt = k0 * np.sqrt(neff**2 - n_bot**2), k0 * np.sqrt(neff**2 - n_top**2)
        v = np.array([1.0, gb * w(n_bot)])
        for n, t in layers:
            q2 = k0**2 * (n**2 - neff**2)
            a = w(n)
            if q2 > 0:
                q = np.sqrt(q2)
                M = np.array([[np.cos(q * t), np.sin(q * t) / (q * a)], [-q * a * np.sin(q * t), np.cos(q * t)]])
            else:
                q = np.sqrt(-q2)
                M = np.array([[np.cosh(q * t), np.sinh(q * t) / (q * a)], [q * a * np.sinh(q * t), np.cosh(q * t)]])
            v = M @ v
        return v[1] + gt * w(n_top) * v[0]

    grid = np.linspace(max(n_bot, n_top) + 1e-9, max(n for n, _ in layers) - 1e-9, 4000)
    f = [disp(x) for x in grid]
    return max(brentq(disp, grid[i], grid[i + 1]) for i in range(len(grid) - 1) if np.sign(f[i]) != np.sign(f[i + 1]))


@sg.silent
def slab_neff_meep(p=None, pol="TE"):
    """n_eff of the input SiN waveguide (no a-Si) from Meep's eigensolver at wl0, with the
    source's own span (core +- mode_span_um, clear of the Si substrate)."""
    p = resolve({**(p or {}), "n_periods": 0, "pol": pol})
    d = domain(p)
    sim = mp.Simulation(cell_size=mp.Vector3(d["cell_x"], d["cell_z"]), boundary_layers=[mp.PML(p["dpml_um"])],
                        geometry=geometry(p, d), default_material=mp.Medium(index=p["n_top"]),
                        resolution=p["resolution"], geometry_center=mp.Vector3(d["x_c"], d["z_c"]))
    sim.init_sim()
    f0 = 1 / p["wl0_um"]
    em = sim.get_eigenmode(f0, mp.X, mp.Volume(center=mp.Vector3(d["x_src"], d["z_mode"]), size=mp.Vector3(0, d["mode_h"])),
                           1, mp.Vector3(1.7 * f0), parity=mp.TM if pol == "TE" else mp.TE)
    return float(em.k.x / f0)


def eta_tm_overlap(x, hz, ex, wl, p_in, p, x0, angle_deg=None):
    """TM coupling to the fiber's other polarization (H along y). Upward S_z = Re(Ex Hy*);
    fiber: Hy_f Gaussian with tilt phase, Ex_f = Hy_f * cos(theta) / n (plane-wave relation,
    Meep units). eta = |X|^2 / (4 P_f P_in), X = integral(Ex_s Hy_f* + Ex_f* Hy_s) dx."""
    th = math.radians(p["fiber_angle_deg"] if angle_deg is None else angle_deg)
    n, w0 = p["n_top"], p["mfd_um"] / 2
    k = 2 * np.pi / wl * n
    hy_f = np.exp(-(((x - x0) * math.cos(th)) ** 2) / w0**2) * np.exp(sg.PHASE_SIGN * 1j * k * math.sin(th) * (x - x0))
    ex_f = hy_f * math.cos(th) / n
    hy_s = -hz
    dx = x[1] - x[0]
    X = np.sum(ex * np.conj(hy_f) + np.conj(ex_f) * hy_s) * dx
    p_f = abs(np.real(np.sum(ex_f * np.conj(hy_f))) * dx)
    return float(np.abs(X) ** 2 / (4 * p_f * p_in))


def eta_at(r, i, x0, angle_deg=None):
    p = r["params"]
    if r["pol"] == "TE":
        kw = {} if angle_deg is None else {"angle_deg": angle_deg}
        return sg.eta_overlap(r["x_up"], r["ez_up"][i], r["hx_up"][i], r["wl"][i], r["p_in"][i], p, x0, **kw)
    return eta_tm_overlap(r["x_up"], r["hz_up"][i], r["ex_up"][i], r["wl"][i], r["p_in"][i], p, x0, angle_deg)


def best(r, angle_deg=None, i=None):
    """Best fiber position at wl0 (or index i); returns (x0, eta)."""
    if i is None:
        i = int(np.argmin(np.abs(r["wl"] - r["params"]["wl0_um"])))
    xs = np.arange(r["x_up"][0] + 2.0, r["x_up"][-1] - 2.0, 0.05)
    e = [eta_at(r, i, x0, angle_deg) for x0 in xs]
    k = int(np.argmax(e))
    return float(xs[k]), float(e[k])


def summary(r, angle_deg=None):
    """eta at wl0 (best x0), spectrum at that fixed x0, 1-dB / 3-dB bands, power budget."""
    i0 = int(np.argmin(np.abs(r["wl"] - r["params"]["wl0_um"])))
    x0, eta = best(r, angle_deg)
    spec = np.array([eta_at(r, i, x0, angle_deg) for i in range(len(r["wl"]))])
    band = lambda db: 1e3 * np.ptp(r["wl"][spec >= spec.max() * 10 ** (-db / 10)])
    up, down = r["up"][i0], r["down"][i0]
    return {"eta": eta, "eta_db": 10 * np.log10(eta), "x0": x0, "spec": spec, "bw1_nm": band(1), "bw3_nm": band(3),
            "up": up, "down": down, "R": r["R"][i0], "T": r["T"][i0], "directionality": up / (up + down),
            "overlap": eta / up, "wall_s": r["wall_s"]}
