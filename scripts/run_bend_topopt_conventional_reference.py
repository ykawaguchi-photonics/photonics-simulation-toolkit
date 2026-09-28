"""What is topology optimization actually WORTH here? Measure a conventional,
un-optimized Euler bend at each design-region footprint, so the optimized
designs have something to be better *than*.

This exists because the optimized bends looked like ordinary waveguides and
the obvious reading was "topology optimization isn't earning its keep". But
"looks conventional" and "performs no better than conventional" are different
claims, and only the second one matters. At a generous footprint a plain bend
is already near-perfect, so there is nothing to win; the interesting question
is what happens as the footprint shrinks and a conventional bend starts to
fail.

The reference geometry is `bend_topopt.build_euler_initial_density` rasterized
into the same compact domain and simulated WITHOUT any optimization -- i.e.
exactly the classical low-loss bend `bend.py` builds, at that footprint. Its
radius is set to the largest that fits: the curve is anchored at the design
region's centre and occupies one quadrant, so footprint ~1.87*radius must fit
within design_region_x_um/2.

Usage:
    /path/to/envs/mp/bin/python -u scripts/run_bend_topopt_conventional_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "data" / "sparams" / "bend_topopt_robust" / "conventional_reference"

from pic_toolkit.meep_sim import bend_topopt, bend_topopt_robust as btr  # noqa: E402

_EULER_FOOTPRINT_PER_RADIUS = 1.87  # a pure Euler spiral's own footprint ratio


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    # Validated at resolution=40 -- the same resolution the optimized designs
    # report at. An earlier pass measured this reference at resolution=20, which
    # over-reports small, multi-component geometry by several points.
    for region in (2.0, 2.5, 3.0, 4.05, 4.5):
        grid_n = int(round(region / 0.025)) + 1
        params = {
            **btr.DEFAULT_PARAMS,
            "design_region_x_um": region,
            "design_region_y_um": region,
            "design_grid_n": grid_n,
        }
        dom = btr._domain_compact(params)

        # Largest Euler radius that fits this footprint, with a little margin.
        r_max = (region / 2) / _EULER_FOOTPRINT_PER_RADIUS
        radius = round(r_max * 0.92, 3)
        init = bend_topopt.build_euler_initial_density(
            {**params, "design_region_center": (dom["x0"], dom["y0"])}, radius
        )
        val = btr.simulate_baseline_compact({**params, "resolution": 40}, weights=init)
        i_mid = len(val.s_matrix["21"]) // 2
        t21 = float(np.abs(val.s_matrix["21"][i_mid]) ** 2)
        r11 = float(np.abs(val.s_matrix["11"][i_mid]) ** 2)
        rows.append({
            "region_um": region, "euler_radius_um": radius,
            "T21": t21, "R11": r11, "radiated": 1 - t21 - r11,
        })
        print(f"region={region:4.2f}um ({region/1.35:.2f} wavelengths)  "
              f"conventional Euler r={radius:.3f}um:  T21={t21:.4f}  radiated={1-t21-r11:.4f}",
              flush=True)

    (_OUT / "conventional_reference.json").write_text(json.dumps(rows, indent=2))
    print(f"saved {_OUT / 'conventional_reference.json'}", flush=True)


if __name__ == "__main__":
    main()
