#!/usr/bin/env python3
"""
Plot circulation diagnostics from checkpoints.

Droplet border is always plotted as the interface contour phi = 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _vorticity(U: np.ndarray, dx: float, dy: float) -> np.ndarray:
    u = U[..., 0]
    v = U[..., 1]
    dvdx = np.zeros_like(v)
    dudy = np.zeros_like(u)

    dvdx[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * dx)
    dvdx[0, :] = (v[1, :] - v[0, :]) / dx
    dvdx[-1, :] = (v[-1, :] - v[-2, :]) / dx

    dudy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dy)
    dudy[:, 0] = (u[:, 1] - u[:, 0]) / dy
    dudy[:, -1] = (u[:, -1] - u[:, -2]) / dy

    return dvdx - dudy


def _plot_checkpoint(checkpoint_path: Path, experiment_name: str, y_max: float, out_dir: Path) -> Path:
    data = np.load(checkpoint_path)
    phi = data["phi"]
    U = data["U"]

    Nx, Ny = phi.shape
    dx = 1.0 / Nx
    dy = 1.0 / Ny
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")

    omega = _vorticity(U, dx, dy)
    vmax = np.percentile(np.abs(omega), 99.0)
    if vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    im = ax.pcolormesh(X, Y, omega, cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)

    # Required convention: droplet border is always the phi=0 interface.
    ax.contour(X, Y, phi, levels=[0.0], colors="k", linewidths=1.4)

    u = U[..., 0]
    v = U[..., 1]
    ax.streamplot(
        x[::2], y[::2], u[::2, ::2].T, v[::2, ::2].T,
        color="k", density=0.9, linewidth=0.5, arrowsize=0.6
    )

    step = int(checkpoint_path.stem.split("_")[-1])
    ax.set_title(f"Circulation/Vorticity | {experiment_name} | step {step}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, y_max)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("vorticity")

    out_path = out_dir / f"circulation_step_{step:06d}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot circulation from checkpoint files.")
    parser.add_argument("--experiment-dir", required=True, help="Experiment directory with checkpoints/")
    parser.add_argument("--steps", nargs="+", type=int, required=True, help="Checkpoint steps to plot")
    parser.add_argument("--y-max", type=float, default=0.25, help="Upper y-limit for plots")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <experiment-dir>/diagnostics/circulation_patterns)",
    )
    args = parser.parse_args()

    exp = Path(args.experiment_dir).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else exp / "diagnostics" / "circulation_patterns"
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in args.steps:
        cp = exp / "checkpoints" / f"checkpoint_{step:06d}.npz"
        if not cp.exists():
            raise FileNotFoundError(f"Missing checkpoint: {cp}")
        out_path = _plot_checkpoint(cp, exp.name, args.y_max, out_dir)
        print(out_path)


if __name__ == "__main__":
    main()

