"""Edge-mode chirality maps for notebook 02: in-plane field of each wall's edge mode at the
drive frequency (periodic ribbon, FDTD frame), C = 2 Im(Fx* Fy) / (|Fx|^2 + |Fy|^2), the
predicted directionality of a circular dipole at each point, and the hotspot (chiral_hotspot).

Usage (mp env, from topological/):
    python scripts/chirality_maps.py
"""
from pathlib import Path

import meep as mp
import numpy as np

from pic_toolkit.meep_sim import topological_pc as tp

mp.verbosity(0)
DATA = Path(__file__).resolve().parents[1] / "data"
RUNS = {"spin_tm": 0.457, "spin_te": 0.405, "valley_te": 0.2368, "valley_tm": 0.265}


def kx_guess(kind: str, freq: float) -> float:
    """kx where the wall-localized (y = 0) ribbon band crosses freq, from the cached ribbon."""
    d = np.load(DATA / "bands" / f"ribbon_{kind}.npz")
    kx, F, L = d["kx"], d["freqs"], d["loc"]
    best = None
    for b in range(F.shape[1]):
        for i in range(len(kx) - 1):
            f0, f1 = F[i, b], F[i + 1, b]
            if (f0 - freq) * (f1 - freq) <= 0 and f1 != f0:
                k = kx[i] + (freq - f0) / (f1 - f0) * (kx[i + 1] - kx[i])
                if best is None or L[i, b, 0] > best[1]:
                    best = (k, L[i, b, 0])
    return best[0]


def at(mode: dict, C: np.ndarray, xy) -> float:
    x = (xy[0] + 0.5) % 1 - 0.5
    return float(C[np.argmin(np.abs(mode["x"] - x)), np.argmin(np.abs(mode["y"] - xy[1]))])


if __name__ == "__main__":
    out = {}
    for kind, freq in RUNS.items():
        m = tp.wall_edge_mode(kind, freq, kx_guess(kind, freq))
        s0, C = tp.chirality(m["F"])
        # eta(S+) = (P_left - P_right)/(...) = C of the left-going mode; the solved mode goes
        # left for vg < 0, otherwise its conjugate does
        C_left = C if m["vg_sign"] < 0 else -C
        hs = tp.chiral_hotspot(m)
        print(f"{kind}: kx {m['kx']:.4f}, f {m['f']:.4f}, vg {m['vg_sign']:+.0f}, "
              f"bands combined {m['n_combined']}, wall share {m['wall_share']:.2f}")
        print(f"   predicted eta(S+) at inclusion {at(m, C_left, tp.source_position(kind, 'inclusion')[0]):+.3f}"
              + (f", hexcentre {at(m, C_left, (0.0, 0.0)):+.3f}" if kind.startswith("valley") else ""))
        print(f"   hotspot {hs['xy'][0]:+.4f}, {hs['xy'][1]:+.4f}: eta(S+) {hs['C'] * (1 if m['vg_sign'] < 0 else -1):+.3f}, "
              f"S0/max {hs['s0_rel']:.2f}, in dielectric {hs['in_dielectric']}", flush=True)
        out.update({f"{kind}_{k}": v for k, v in dict(
            x=m["x"], y=m["y"], eps=m["eps"].astype(np.float32), F=m["F"].astype(np.complex64),
            kx=m["kx"], f=m["f"], vg_sign=m["vg_sign"], s0=s0.astype(np.float32),
            C_left=C_left.astype(np.float32), hotspot=np.array(hs["xy"]), hotspot_ij=np.array(hs["ij"])).items()})
    np.savez_compressed(DATA / "walls" / "chirality_maps.npz", **out)
