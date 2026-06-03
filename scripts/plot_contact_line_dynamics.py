#!/usr/bin/env python3
"""
Plot contact line dynamics: interface height vs x at three timestamps
(beginning, middle, end) on one chart with distinct line styles.
Saves to experiment_dir/contact_line_dynamics.png (experiment root).
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required for plotting")
    sys.exit(1)

# Allow running from repo root (PYTHONPATH=.) without pulling in full src.visualization
REPO = Path(__file__).resolve().parent.parent
_checkpointing_path = REPO / "src" / "visualization" / "checkpointing.py"
_spec = importlib.util.spec_from_file_location("checkpointing", _checkpointing_path)
_checkpointing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_checkpointing)
load_checkpoint = _checkpointing.load_checkpoint


def _grid_like_plot_utils(Nx, Ny, Lx, Ly):
    """Same grid as plot_layout.prepare_joint_plot_data: x, eta, X, Eta."""
    x = np.linspace(0, Lx, Nx)
    eta = np.linspace(0, Ly, Ny)
    X, Eta = np.meshgrid(x, eta)
    return x, eta, X, Eta


def _phi0_contour_surface(phi, X, Eta):
    """
    Extract phi=0 contour (same as plot utils: contour(X, Eta, phi.T, levels=[0])).
    Returns the free-surface part (y > 0) as (x_phys, y_phys).
    """
    phi = np.asarray(phi)
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(X, Eta, phi.T, levels=[0.0])
    plt.close(fig_tmp)
    segs = cs.allsegs[0] if len(cs.levels) > 0 and cs.allsegs else []
    if not segs:
        return np.array([]), np.array([])
    # Take the segment with largest y extent (droplet); keep only free surface (y > small)
    best = max(segs, key=lambda s: np.max(s[:, 1]) - np.min(s[:, 1]) if len(s) > 0 else 0)
    best = np.asarray(best)
    if len(best) == 0:
        return np.array([]), np.array([])
    y_eps = 1e-6 * (np.nanmax(best[:, 1]) - np.nanmin(best[:, 1]) + 1e-12)
    mask = best[:, 1] >= y_eps
    if not np.any(mask):
        mask = best[:, 1] >= 0
    x_phys = best[mask, 0]
    y_phys = best[mask, 1]
    # Sort by x for a clean left-to-right curve
    order = np.argsort(x_phys)
    x_phys = x_phys[order]
    y_phys = y_phys[order]
    # Extend to substrate (y=0) so contour meets the surface line at both contact points
    if len(x_phys) >= 2:
        x_phys = np.concatenate([[x_phys[0]], x_phys, [x_phys[-1]]])
        y_phys = np.concatenate([[0.0], y_phys, [0.0]])
    return x_phys, y_phys


def main():
    parser = argparse.ArgumentParser(
        description="Plot contact line dynamics (begin/mid/end). Output: <experiment_dir>/contact_line_dynamics.png",
    )
    parser.add_argument(
        "experiment_dir",
        nargs="?",
        default=None,
        help="Experiment directory (e.g. experiment_20260217_091150). Required unless default exists.",
    )
    args = parser.parse_args()
    exp_dir = Path(args.experiment_dir) if args.experiment_dir else REPO / "experiment_20260217_091150"
    if not exp_dir.is_dir():
        print(f"Experiment directory not found: {exp_dir}")
        print("Usage: PYTHONPATH=. python scripts/plot_contact_line_dynamics.py <experiment_dir>")
        sys.exit(1)
    out_path = exp_dir / "contact_line_dynamics.png"
    print(f"Experiment: {exp_dir.resolve()}")
    print(f"Output:     {out_path.resolve()}")

    params_path = exp_dir / "simulation_parameters.json"
    stats_path = exp_dir / "statistics.csv"
    checkpoint_dir = exp_dir / "checkpoints"

    if not params_path.exists():
        print(f"Missing {params_path}")
        sys.exit(1)
    with open(params_path) as f:
        params = json.load(f)
    grid = params.get("grid_params", {})
    Nx = grid.get("Nx", 128)
    Ny = grid.get("Ny", 128)
    Lx = grid.get("Lx", 1.0)
    Ly = grid.get("Ly", 1.0)
    dx = Lx / Nx
    dy = Ly / Ny

    if not stats_path.exists():
        print(f"Missing {stats_path}")
        sys.exit(1)
    df = pd.read_csv(stats_path)
    steps = df["step"].values
    times = df["time"].values

    cands = sorted([int(p.stem.split("_")[1]) for p in Path(checkpoint_dir).glob("checkpoint_*.npz")])
    if not cands:
        print("No checkpoints found in", checkpoint_dir)
        sys.exit(1)
    n_c = len(cands)
    step_initial = cands[0]
    step_beginning = cands[n_c // 4] if n_c >= 4 else cands[0]
    step_mid = cands[n_c // 2]
    step_end = cands[-1]

    def load_phi(step):
        p = checkpoint_dir / f"checkpoint_{step:06d}.npz"
        if not p.exists():
            return None, None, None
        ck = load_checkpoint(str(p))
        return ck["phi"], float(ck.get("time", 0)), step

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Substrate (y=0)
    ax.axhline(0.0, color="gray", linestyle="-", linewidth=1.5, zorder=0)
    ax.fill_between([0, Lx], 0, -0.02 * Ly, color="gray", alpha=0.3, zorder=0)

    segments = [
        (step_initial, "Initial", "solid", "#4a4a4a"),
        (step_beginning, "Beginning", "dashed", "#1f77b4"),
        (step_mid, "Middle", "dotted", "#ff7f0e"),
        (step_end, "End", (0, (3, 1, 1, 1)), "#2ca02c"),
    ]
    lw = 1.0
    x_gr, eta_gr, X_gr, Eta_gr = _grid_like_plot_utils(Nx, Ny, Lx, Ly)
    for step, label, linestyle, color in segments:
        phi, time, _ = load_phi(step)
        if phi is None:
            continue
        phi = np.asarray(phi)
        x_phys, y_phys = _phi0_contour_surface(phi, X_gr, Eta_gr)
        if len(x_phys) == 0:
            continue
        time_val = times[np.argmin(np.abs(steps - step))]
        ax.plot(x_phys, y_phys, linestyle=linestyle, color=color, linewidth=lw,
                label=f"{label} (step {step}, t={time_val:.2e})", zorder=2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Contact line dynamics")
    ax.set_xlim(0.3, 0.7)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02 * Ly, None)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
