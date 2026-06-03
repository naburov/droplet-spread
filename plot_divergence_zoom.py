#!/usr/bin/env python3
"""
Plot divergence from experiment checkpoints: full domain, zoom-ins, and 10x10 grid with values.

Runs for every checkpoint in the experiment (checkpoint_*.npz).

Usage (with project env and PYTHONPATH=src):
  cd /path/to/droplet_spreading_modeling
  PYTHONPATH=src python plot_divergence_zoom.py /path/to/experiment_20260201_143811

Outputs in experiment_dir/divergence_zoom/ (one set per checkpoint step):
  - divergence_full_step000000.png, divergence_full_step000020.png, ...
  - divergence_zoom_bottom_left_step000000.png, ...
  - divergence_zoom_around_max_step000000.png, ...
  - divergence_10x10_values_step000000.png, divergence_10x10_around_max_step000000.png
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.physics.fluid_dynamics import jax_check_continuity
from src.simulation.geometry import TiltedGeometry


def load_experiment_config(exp_dir):
    path = os.path.join(exp_dir, "simulation_parameters.json")
    with open(path) as f:
        return json.load(f)


def build_geometry_from_config(config):
    grid = config["grid_params"]
    Nx, Ny = grid["Nx"], grid["Ny"]
    Lx, Ly = grid["Lx"], grid["Ly"]
    dx, dy = Lx / Nx, Ly / Ny
    geom_cfg = config.get("geometry", config.get("initial_conditions", {}).get("geometry", {}))
    if geom_cfg.get("type") == "tilted":
        degree = float(geom_cfg.get("degree", 10.0))
        origin = geom_cfg.get("origin", "bottom_left")
        geometry = TiltedGeometry(Nx, Ny, dx, dy, Lx, Ly, angle_degrees=degree, origin=origin)
    else:
        from src.simulation.geometry import FlatGeometry
        geometry = FlatGeometry(Nx, Ny)
    return geometry, dx, dy, Nx, Ny


def compute_divergence(U, dx, dy, geometry):
    div_field, max_div, mean_div = jax_check_continuity(U, dx, dy, geometry.f_1_grid)
    return np.array(div_field), float(max_div), float(mean_div)


def plot_full_domain(div, dx, dy, step, out_dir):
    """Full domain divergence (imshow)."""
    Nx, Ny = div.shape
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    extent = [0, Nx * dx, 0, Ny * dy]
    v = np.abs(div).max()
    im = ax.imshow(
        div.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
    )
    plt.colorbar(im, ax=ax, label="divergence")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Divergence full domain (step {step})")
    plt.tight_layout()
    path = os.path.join(out_dir, f"divergence_full_step{step:06d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_zoom(div, dx, dy, step, out_dir, i_lo, i_hi, j_lo, j_hi, title_suffix):
    """Zoom into a rectangular region."""
    patch = div[i_lo:i_hi, j_lo:j_hi]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    v = np.abs(patch).max() or 1.0
    im = ax.imshow(
        patch.T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
    )
    plt.colorbar(im, ax=ax, label="divergence")
    ax.set_xlabel("i (x index)")
    ax.set_ylabel("j (y index)")
    ax.set_title(f"Divergence zoom {title_suffix} (step {step}), i=[{i_lo},{i_hi}) j=[{j_lo},{j_hi})")
    plt.tight_layout()
    path = os.path.join(out_dir, f"divergence_zoom_{title_suffix}_step{step:06d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_10x10_values(div, step, out_dir, i0=0, j0=0, suffix="values"):
    """10x10 grid with numeric values in each cell (bottom-left corner by default)."""
    # div[i,j]: i=x, j=y. Show div[i0:i0+10, j0:j0+10] with j increasing upward (origin=lower)
    patch = div[i0 : i0 + 10, j0 : j0 + 10]  # (10, 10)
    # For imshow with origin='lower', row 0 is bottom -> j index 0 at bottom
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    v = np.abs(patch).max() or 1.0
    im = ax.imshow(
        patch.T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
    )
    # Add text in each cell (value with 2 decimals)
    for ii in range(patch.shape[0]):
        for jj in range(patch.shape[1]):
            val = patch[ii, jj]
            ax.text(
                ii,
                jj,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if np.abs(val) < v * 0.5 else "white",
            )
    plt.colorbar(im, ax=ax, label="divergence")
    ax.set_xlabel("i (x index)")
    ax.set_ylabel("j (y index)")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(np.arange(i0, i0 + 10))
    ax.set_yticklabels(np.arange(j0, j0 + 10))
    ax.set_title(f"Divergence 10x10 {suffix} (step {step}), i=[{i0},{i0+10}) j=[{j0},{j0+10})")
    plt.tight_layout()
    path = os.path.join(out_dir, f"divergence_10x10_{suffix}_step{step:06d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if not exp_dir or not os.path.isdir(exp_dir):
        print("Usage: python plot_divergence_zoom.py /path/to/experiment_dir")
        sys.exit(1)

    config = load_experiment_config(exp_dir)
    geometry, dx, dy, Nx, Ny = build_geometry_from_config(config)
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    out_dir = os.path.join(exp_dir, "divergence_zoom")
    os.makedirs(out_dir, exist_ok=True)

    steps = []
    for name in os.listdir(checkpoints_dir):
        if name.startswith("checkpoint_") and name.endswith(".npz"):
            try:
                steps.append(int(name.replace("checkpoint_", "").replace(".npz", "")))
            except ValueError:
                pass
    steps = sorted(steps)
    if not steps:
        print(f"No checkpoints found in {checkpoints_dir}")
        return
    print(f"Found {len(steps)} checkpoints: steps {steps[0]}..{steps[-1]}")

    for step in steps:
        cpath = os.path.join(checkpoints_dir, f"checkpoint_{step:06d}.npz")
        if not os.path.isfile(cpath):
            continue
        data = np.load(cpath, allow_pickle=True)
        U = data["U"]
        div, max_div, mean_div = compute_divergence(U, dx, dy, geometry)
        print(f"Step {step}: max_div={max_div:.4f}, mean_div={mean_div:.4f}")

        # Full domain
        plot_full_domain(div, dx, dy, step, out_dir)

        # Zoom bottom-left (inlet + no_slip corner): first 20 x 20
        plot_zoom(div, dx, dy, step, out_dir, 0, 20, 0, 20, "bottom_left")

        # Zoom around global max
        abs_div = np.abs(div)
        jmax, imax = np.unravel_index(np.argmax(abs_div), abs_div.shape)
        i_lo = max(0, imax - 10)
        i_hi = min(Nx, imax + 10)
        j_lo = max(0, jmax - 10)
        j_hi = min(Ny, jmax + 10)
        plot_zoom(div, dx, dy, step, out_dir, i_lo, i_hi, j_lo, j_hi, "around_max")

        # 10x10 grid with values: bottom-left corner (inlet + no_slip)
        plot_10x10_values(div, step, out_dir, i0=0, j0=0, suffix="values")

        # 10x10 at the location of max divergence
        if i_hi - i_lo >= 10 and j_hi - j_lo >= 10:
            plot_10x10_values(div, step, out_dir, i0=i_lo, j0=j_lo, suffix="around_max")

    print(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
