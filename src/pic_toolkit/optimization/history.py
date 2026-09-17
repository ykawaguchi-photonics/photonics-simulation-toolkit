"""Plotting helpers for the Adam optimization trajectory and the 2D
objective landscape, using `pic_toolkit.style`'s existing palette
conventions (COLOR_ROSE for the initial/single-point emphasis, COLOR_SEAGREEN
for the optimized/final design)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .. import style


def plot_trajectory(rows: list[dict]):
    """4-panel: L_upper, L_lower, Delta_L, objective vs. iteration."""
    it = [r["iteration"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        ("L_upper", "L_upper (um)"),
        ("L_lower", "L_lower (um)"),
        ("Delta_L", "Delta_L = L_lower - L_upper (um)"),
        ("objective", "objective"),
    ]
    for ax, (key, ylabel) in zip(axes.flat, panels):
        ax.plot(it, [r[key] for r in rows], color=style.COLOR_STEEL)
        ax.scatter([it[0]], [rows[0][key]], color=style.COLOR_ROSE, zorder=5, label="initial")
        ax.scatter([it[-1]], [rows[-1][key]], color=style.COLOR_SEAGREEN, zorder=5, label="final")
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_landscape_with_trajectory(grid: dict, rows: list[dict] | None = None):
    """Contour of grid["loss"] over (L_upper, delta_L_delay_um), with the
    optimization trajectory overlaid (if given) and the initial/final points
    marked."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    L_upper_vals, delta_L_vals, loss = grid["L_upper"], grid["delta_L_delay_um"], grid["loss"]
    cs = ax.contourf(L_upper_vals, delta_L_vals, loss.T, levels=30, cmap=style.CMAP_DENSITY)
    fig.colorbar(cs, ax=ax, label="objective")

    if rows is not None:
        traj_u = [r["L_upper"] for r in rows]
        traj_d = [r["delta_L_delay_um"] for r in rows]
        ax.plot(traj_u, traj_d, color="white", lw=1.5, alpha=0.85, label="Adam trajectory")
        ax.scatter([traj_u[0]], [traj_d[0]], color=style.COLOR_ROSE, s=60, zorder=5, label="initial design")
        ax.scatter([traj_u[-1]], [traj_d[-1]], color=style.COLOR_SEAGREEN, s=60, zorder=5, label="optimized design")

    ax.set_xlabel("L_upper (um)")
    ax.set_ylabel("delta_L_delay_um")
    ax.set_title("MUX2 stage-1 objective landscape")
    ax.legend()
    fig.tight_layout()
    return fig
