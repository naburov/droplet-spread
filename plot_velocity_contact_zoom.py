#!/usr/bin/env python3
"""
Plot velocity at contact points from experiment checkpoints: full domain, zoom at surface,
zoom around contact line, and 10x10 grid with u,v values.

Runs for every checkpoint saved in the experiment (checkpoint_*.npz).

Contact points: surface row (j=0) where the interface is present (phi crosses zero or
|phi| < 0.5 with non-trivial gradient).

Usage (with project env and PYTHONPATH=src):
  cd /path/to/droplet_spreading_modeling
  PYTHONPATH=src python plot_velocity_contact_zoom.py /path/to/experiment_20260201_143811

Outputs in experiment_dir/velocity_contact_zoom/ (one set per checkpoint step):
  - velocity_full_step000000.png, velocity_full_step000020.png, ...
  - velocity_zoom_surface_step000000.png, ...
  - velocity_zoom_around_contact_step000000.png, ...
  - velocity_10x10_surface_values_step000000.png, ...
  - velocity_10x10_around_contact_step000000.png (when contact exists)
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation.geometry import TiltedGeometry, FlatGeometry


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
        geometry = FlatGeometry(Nx, Ny)
    return geometry, dx, dy, Nx, Ny


def contact_mask_surface(phi, dx, dy, grad_thresh=1e-3):
    """Boolean mask (Nx,) for surface (j=0) cells where the interface is present."""
    phi_bottom = phi[:, 0]
    phi_above = phi[:, 1] if phi.shape[1] > 1 else phi_bottom
    phi_crosses = (phi_bottom * phi_above) < 0
    phi_near_zero = np.abs(phi_bottom) < 0.5
    # Gradient magnitude at j=0 (simple one-sided)
    grad_x = np.zeros_like(phi_bottom)
    grad_x[1:-1] = (phi[2, 0] - phi[:-2, 0]) / (2 * dx)
    grad_x[0] = (phi[1, 0] - phi[0, 0]) / dx
    grad_x[-1] = (phi[-1, 0] - phi[-2, 0]) / dx
    grad_y = (phi[:, 1] - phi[:, 0]) / dy if phi.shape[1] > 1 else np.zeros_like(phi_bottom)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    has_interface = grad_mag > grad_thresh
    return (phi_crosses | phi_near_zero) & has_interface


def plot_full_domain(U, phi, dx, dy, step, out_dir):
    """Full domain: speed (imshow) and quiver (subsampled)."""
    Nx, Ny = U.shape[:2]
    speed = np.sqrt(U[..., 0] ** 2 + U[..., 1] ** 2)
    extent = [0, Nx * dx, 0, Ny * dy]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    im = ax.imshow(
        speed.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="|U|")
    # Quiver every few points
    skip = max(1, Nx // 12)
    skipy = max(1, Ny // 12)
    ii = np.arange(0, Nx, skip)
    jj = np.arange(0, Ny, skipy)
    I, J = np.meshgrid(ii, jj)
    u = U[I, J, 0]
    v = U[I, J, 1]
    x = I * dx + dx / 2
    y = J * dy + dy / 2
    ax.quiver(x, y, u, v, scale=None, color="white", alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Velocity full domain (step {step})")
    plt.tight_layout()
    path = os.path.join(out_dir, f"velocity_full_step{step:06d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_zoom_surface(U, phi, dx, dy, step, out_dir, j_hi=12):
    """Zoom into bottom (surface) strip: first Nx x j_hi cells."""
    Nx, Ny = U.shape[:2]
    j_hi = min(j_hi, Ny)
    patch_u = U[:Nx, 0:j_hi, 0]
    patch_v = U[:Nx, 0:j_hi, 1]
    speed = np.sqrt(patch_u**2 + patch_v**2)
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    extent = [0, Nx * dx, 0, j_hi * dy]
    im = ax.imshow(
        speed.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="|U|")
    # Quiver
    skip = max(1, Nx // 20)
    ii = np.arange(0, Nx, skip)
    jj = np.arange(0, j_hi)
    I, J = np.meshgrid(ii, jj)
    u = U[I, J, 0]
    v = U[I, J, 1]
    x = I * dx + dx / 2
    y = J * dy + dy / 2
    ax.quiver(x, y, u, v, scale=None, color="white", alpha=0.9)
    ax.axhline(0.5 * dy, color="gray", ls="--", alpha=0.7, label="j=0 (surface)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Velocity zoom surface (step {step}), j in [0, {j_hi})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(out_dir, f"velocity_zoom_surface_step{step:06d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_zoom_around_contact(U, phi, dx, dy, step, out_dir, contact_mask, half_width=15, j_hi=15):
    """Zoom around contact line: i around center of contact, j in [0, j_hi]."""
    Nx, Ny = U.shape[:2]
    i_inds = np.where(contact_mask)[0]
    if len(i_inds) == 0:
        i_center = Nx // 2
    else:
        i_center = int(np.mean(i_inds))
    i_lo = max(0, i_center - half_width)
    i_hi = min(Nx, i_center + half_width)
    j_hi = min(j_hi, Ny)
    patch_u = U[i_lo:i_hi, 0:j_hi, 0]
    patch_v = U[i_lo:i_hi, 0:j_hi, 1]
    speed = np.sqrt(patch_u**2 + patch_v**2)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    extent = [i_lo * dx, i_hi * dx, 0, j_hi * dy]
    im = ax.imshow(
        speed.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="|U|")
    skip = max(1, (i_hi - i_lo) // 10)
    ii = np.arange(i_lo, i_hi, skip)
    jj = np.arange(0, j_hi)
    I, J = np.meshgrid(ii, jj)
    u = U[I, J, 0]
    v = U[I, J, 1]
    x = I * dx + dx / 2
    y = J * dy + dy / 2
    ax.quiver(x, y, u, v, scale=None, color="white", alpha=0.9)
    # Mark contact columns in surface row (axes coords: y from 0 to 1/j_hi)
    contact_in_patch = contact_mask[i_lo:i_hi]
    for i in range(i_hi - i_lo):
        if contact_in_patch[i]:
            ax.axvspan(i_lo * dx + i * dx, i_lo * dx + (i + 1) * dx, ymin=0, ymax=1.0 / j_hi, color="red", alpha=0.4)
    ax.axhline(0.5 * dy, color="gray", ls="--", alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Velocity zoom around contact (step {step})")
    plt.tight_layout()
    path = os.path.join(out_dir, f"velocity_zoom_around_contact_step{step:06d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_10x10_values(U, step, out_dir, i0=0, j0=0, suffix="values"):
    """10x10 grid with u and v in each cell (surface / bottom-left by default)."""
    patch_u = U[i0 : i0 + 10, j0 : j0 + 10, 0]
    patch_v = U[i0 : i0 + 10, j0 : j0 + 10, 1]
    speed = np.sqrt(patch_u**2 + patch_v**2)
    vmax = speed.max() or 1.0
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(
        speed.T,
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=vmax,
    )
    for ii in range(10):
        for jj in range(10):
            u, v = patch_u[ii, jj], patch_v[ii, jj]
            ax.text(
                ii,
                jj,
                f"{u:.2f}\n{v:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if speed[ii, jj] > vmax * 0.4 else "black",
            )
    plt.colorbar(im, ax=ax, label="|U|")
    ax.set_xlabel("i (x index)")
    ax.set_ylabel("j (y index)")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(np.arange(i0, i0 + 10))
    ax.set_yticklabels(np.arange(j0, j0 + 10))
    ax.set_title(f"Velocity 10x10 {suffix} (step {step}), u,v per cell, i=[{i0},{i0+10}) j=[{j0},{j0+10})")
    plt.tight_layout()
    path = os.path.join(out_dir, f"velocity_10x10_{suffix}_step{step:06d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if not exp_dir or not os.path.isdir(exp_dir):
        print("Usage: python plot_velocity_contact_zoom.py /path/to/experiment_dir")
        sys.exit(1)

    config = load_experiment_config(exp_dir)
    geometry, dx, dy, Nx, Ny = build_geometry_from_config(config)
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    out_dir = os.path.join(exp_dir, "velocity_contact_zoom")
    os.makedirs(out_dir, exist_ok=True)

    # Discover all saved checkpoints
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
        phi = data["phi"]
        contact = contact_mask_surface(phi, dx, dy)
        n_contact = np.sum(contact)
        print(f"Step {step}: contact points at j=0: {n_contact}")

        plot_full_domain(U, phi, dx, dy, step, out_dir)
        plot_zoom_surface(U, phi, dx, dy, step, out_dir)
        plot_zoom_around_contact(U, phi, dx, dy, step, out_dir, contact)

        plot_10x10_values(U, step, out_dir, i0=0, j0=0, suffix="surface_values")

        i_inds = np.where(contact)[0]
        if len(i_inds) > 0:
            i_center = int(np.mean(i_inds))
            i_lo = max(0, i_center - 5)
            if i_lo + 10 <= Nx:
                plot_10x10_values(U, step, out_dir, i0=i_lo, j0=0, suffix="around_contact")

    print(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
