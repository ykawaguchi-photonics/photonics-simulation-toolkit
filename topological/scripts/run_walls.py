"""Domain-wall FDTD runs for notebook 02: circularly polarized dipole (S+ and S-) at the
wall of one crystal kind, one drive frequency.

The power split through a closed box is evaluated at full resolution here; the stored field
maps keep every second grid point to limit the file size.

SOURCE_AT (tp.source_position): default -- the edge-mode chirality hotspot
(tp.CHIRAL_HOTSPOTS, scripts/chirality_maps.py); inclusion -- cluster centre (spin) or bearded
inclusion centre (valley); hexcentre -- C2 centre on the valley wall.

Usage (mp env, from topological/):
    python scripts/run_walls.py KIND FREQ [SOURCE_AT]
    e.g. python scripts/run_walls.py valley_te 0.2368
         python scripts/run_walls.py valley_te 0.2368 hexcentre
"""
import sys
from pathlib import Path

import meep as mp
import numpy as np

from pic_toolkit.meep_sim import topological_pc as tp

mp.verbosity(0)
DATA = Path(__file__).resolve().parents[1] / "data" / "walls"
X_LINE = {"spin_tm": 8.0, "spin_te": 8.0, "valley_te": 14.0, "valley_tm": 14.0}


def cache_path(kind: str, freq: float, spin: int, source_at: str = "default") -> Path:
    tag = "" if source_at == "default" else f"_{source_at}"
    return DATA / f"{kind}_f{freq:.4f}_s{'p' if spin > 0 else 'm'}{tag}.npz"


if __name__ == "__main__":
    kind, freq = sys.argv[1], float(sys.argv[2])
    source_at = sys.argv[3] if len(sys.argv) > 3 else "default"
    DATA.mkdir(parents=True, exist_ok=True)
    for spin in (+1, -1):
        p = cache_path(kind, freq, spin, source_at)
        if p.exists():
            continue
        o = tp.run_domain_wall(kind, spin, freq, source_at=source_at)
        flux = tp.box_fluxes(o, X_LINE[kind], o["dpml"])
        comps = ("ez", "hx", "hy") if tp.POLARIZATION[kind] == "TM" else ("hz", "ex", "ey")
        sl = (slice(None, None, 2), slice(None, None, 2))
        np.savez_compressed(
            p, x=o["x"][::2], y=o["y"][::2], eps=o["eps"][sl].astype(np.float32),
            **{c: o[c][sl].astype(np.complex64) for c in comps},
            E2=o["E2"][sl].astype(np.float32), Sx=o["Sx"][sl].astype(np.float32),
            Sy=o["Sy"][sl].astype(np.float32), src_xy=o["src_xy"], freq=freq, spin=spin,
            **{f"P_{k}": v for k, v in flux.items()}, x_line=X_LINE[kind],
            resolution=o["resolution"], run_time=o["run_time"], n_x=o["n_x"], n_rows=o["n_rows"],
            dpml=o["dpml"], pad_y=o["pad_y"])
        print("done", p.name, flush=True)
