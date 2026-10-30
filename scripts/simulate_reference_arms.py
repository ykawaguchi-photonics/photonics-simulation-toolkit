"""2D FDTD of the straight reference arms used by the real-component lattice
circuits in circuits/wdm_sax.ipynb and circuits/wdm_mux4_sax.ipynb.

A lattice stage's two arms must be described consistently: the delay arm is
the FDTD-measured jog of meep_sim/mzi_arm.py (resolution 40, S21 between
mode-monitor planes 1 um inside each port). The reference arm is therefore
simulated here with the same solver settings -- a straight waveguide whose
monitor-plane separation equals the jog's (4 * radius + 2 * lead - 2 *
port_offset: 24 um for radius 5, 14 um for radius 2.5), at the same
resolution, band and frequency sampling. Any phase that both arms share (the
extra lead length up to the ports, coupler reference planes) then cancels in
the interference, and the grid's numerical dispersion is the same for both.

Saved to si/data/sparams/waveguide/reference_arm_res40/waveguide_length<L>.{npz,json};
read by pic_toolkit.models.mzi_arm.straight_reference().

Usage (from the repo root, a few minutes on one core):
    conda run -n mp python scripts/simulate_reference_arms.py
"""

from pathlib import Path

from pic_toolkit import checks, sparams
from pic_toolkit.meep_sim import mzi_arm, waveguide

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "si" / "data" / "sparams" / "waveguide" / "reference_arm_res40"


def monitor_separation_um(radius_um: float) -> float:
    p = {**mzi_arm.DEFAULT_PARAMS, "radius_um": radius_um}
    p1, _, p2, _ = mzi_arm._port_planes(p)
    return round(float(p2.x - p1.x), 6)


def main():
    arm = mzi_arm.DEFAULT_PARAMS
    for radius in (5.0, 2.5):
        length = monitor_separation_um(radius)
        params = {"length_um": length, "resolution": arm["resolution"], "n_freq": arm["n_freq"],
                  "wl_min_um": arm["wl_min_um"], "wl_max_um": arm["wl_max_um"]}
        print(f"radius {radius} um -> straight reference over {length} um at resolution {params['resolution']}")
        result = waveguide.simulate_baseline(params)
        validation = checks.run_all_checks(result.s_matrix)
        assert validation["passed"], validation
        metadata = {
            "component": "waveguide_reference_arm",
            "purpose": f"straight reference for mzi_arm radius_um={radius} (same monitor-plane separation)",
            "geometry_params": {"wg_width_um": result.sim_params["wg_width_um"], "length_um": length},
            "material_params": {"core_index": result.sim_params["core_index"],
                                "clad_index": result.sim_params["clad_index"]},
            "meep_resolution": result.sim_params["resolution"],
            "simulation_params": {k: result.sim_params[k] for k in
                                  ["wl_min_um", "wl_max_um", "n_freq", "dpml_um", "cell_x_um", "cell_y_um",
                                   "port_offset_um", "source_offset_um"]},
            "meep_version": result.sim_params["meep_version"],
            "validation": validation,
        }
        sparams.save_artifact(OUT_DIR / f"waveguide_length{length}", result.wavelengths_um, result.freqs,
                              result.s_matrix, ("o1", "o2"), metadata)
        print("  saved, validation passed")


if __name__ == "__main__":
    main()
