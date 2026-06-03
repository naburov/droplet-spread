#!/usr/bin/env python3
"""
Plot velocity profiles from a simulation checkpoint.

Usage:
  python scripts/plot_checkpoint_velocity.py <checkpoint.npz> [--out velocity_profiles.png]
  python scripts/plot_checkpoint_velocity.py <experiment_dir> --step <N> [--out ...]

Example:
  python scripts/plot_checkpoint_velocity.py experiment_upstream_staggered_64/checkpoints/checkpoint_000010.npz
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root so we can import src
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.visualization.checkpointing import load_checkpoint


def plot_velocity_profiles(checkpoint_path, out_path=None):
    """Load checkpoint and plot u/v profiles (bottom slice, mid-x slice, 2D magnitude)."""
    if not os.path.isfile(checkpoint_path):
        # Allow path without .npz
        if os.path.isfile(checkpoint_path + ".npz"):
            checkpoint_path = checkpoint_path + ".npz"
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data = load_checkpoint(checkpoint_path)
    U = np.asarray(data["U"])
    step = data["step"]
    Nx, Ny = U.shape[0], U.shape[1]
    dx = 1.0 / Nx  # assume Lx=Ly=1
    dy = 1.0 / Ny
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy

    u = U[..., 0]
    v = U[..., 1]
    u_bottom = u[:, 1]   # first fluid row above bottom (j=1)
    v_bottom = v[:, 1]

    # Stats (useful to check v == 0 or NaNs)
    print(f"Checkpoint step {step} shape {U.shape}")
    print(f"  u: min={u.min():.6f} max={u.max():.6f} mean={u.mean():.6f}")
    print(f"  v: min={v.min():.6f} max={v.max():.6f} mean={v.mean():.6f}")
    print(f"  u_bottom (j=1): min={u_bottom.min():.6f} max={u_bottom.max():.6f}")
    print(f"  v_bottom (j=1): min={v_bottom.min():.6f} max={v_bottom.max():.6f}")
    if np.any(~np.isfinite(U)):
        print("  WARNING: non-finite values in U")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1) u at bottom (j=1) – contact line / inlet profile
    ax = axes[0, 0]
    ax.plot(x, u_bottom, "b-", linewidth=2, label="u at y=dy")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("u just above bottom (j=1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) v at bottom
    ax = axes[0, 1]
    ax.plot(x, v_bottom, "b-", linewidth=2, label="v at y=dy")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_title("v just above bottom (j=1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3) u and v at mid-x
    i_mid = Nx // 2
    ax = axes[1, 0]
    ax.plot(y, u[i_mid, :], "b-", label="u", linewidth=2)
    ax.plot(y, v[i_mid, :], "r-", label="v", linewidth=2)
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("y")
    ax.set_ylabel("velocity")
    ax.set_title(f"u, v at x={x[i_mid]:.3f} (mid)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4) 2D velocity magnitude
    ax = axes[1, 1]
    mag = np.sqrt(u**2 + v**2)
    im = ax.imshow(
        mag.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="equal",
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="|U|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Velocity magnitude")

    plt.suptitle(f"Velocity profiles (step {step})")
    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(checkpoint_path),
            f"velocity_profiles_step{step:06d}.png",
        )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot velocity profiles from a checkpoint")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint .npz or experiment dir (use --step with dir)",
    )
    parser.add_argument("--step", type=int, default=None, help="Step number when checkpoint is experiment dir")
    parser.add_argument("--out", "-o", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    path = args.checkpoint
    if os.path.isdir(path):
        if args.step is None:
            print("Error: when checkpoint is a directory, pass --step N")
            sys.exit(1)
        path = os.path.join(path, "checkpoints", f"checkpoint_{args.step:06d}.npz")
    plot_velocity_profiles(path, out_path=args.out)


if __name__ == "__main__":
    main()
