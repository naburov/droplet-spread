#!/usr/bin/env python3
"""
Check a simulation checkpoint for anomalies (NaN, Inf, extreme values, mass, divergence, etc.).

Usage:
  python scripts/check_checkpoint_anomalies.py <checkpoint.npz>
  python scripts/check_checkpoint_anomalies.py <experiment_dir> --step <N>
"""

import argparse
import os
import sys

import numpy as np

# Add project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.visualization.checkpointing import load_checkpoint


def check_checkpoint(checkpoint_path):
    """Load checkpoint and report anomalies."""
    if not os.path.isfile(checkpoint_path):
        if os.path.isfile(checkpoint_path + ".npz"):
            checkpoint_path = checkpoint_path + ".npz"
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data = load_checkpoint(checkpoint_path)
    step = data["step"]
    phi = np.asarray(data["phi"])
    U = np.asarray(data["U"])
    P = np.asarray(data["P"])

    Nx, Ny = phi.shape[0], phi.shape[1]
    dx, dy = 1.0 / Nx, 1.0 / Ny
    Lx, Ly = 1.0, 1.0

    anomalies = []
    ok = []

    # ----- Finite -----
    for name, arr in [("phi", phi), ("U", U), ("P", P)]:
        if not np.all(np.isfinite(arr)):
            n_bad = np.sum(~np.isfinite(arr))
            anomalies.append(f"{name}: {n_bad} non-finite values (NaN/Inf)")
        else:
            ok.append(f"{name}: all finite")

    # ----- Phase field range (typically [-1, 1] or similar) -----
    phi_min, phi_max = float(phi.min()), float(phi.max())
    if phi_min < -1.5 or phi_max > 1.5:
        anomalies.append(f"phi: out-of-range min={phi_min:.4f} max={phi_max:.4f} (expected ~[-1,1])")
    else:
        ok.append(f"phi: range [{phi_min:.4f}, {phi_max:.4f}]")

    # ----- Velocity magnitude -----
    u, v = U[..., 0], U[..., 1]
    mag = np.sqrt(u**2 + v**2)
    mag_max = float(mag.max())
    if not np.isfinite(mag_max) or mag_max > 1e6:
        anomalies.append(f"U: extreme magnitude max={mag_max}")
    else:
        ok.append(f"U: |U|_max={mag_max:.6f}")

    # ----- Bottom row (no-slip: u[0]=0, v[0]=0 at wall) -----
    u_bottom = u[:, 1]   # first fluid row
    v_bottom = v[:, 1]
    u_wall = u[:, 0]
    v_wall = v[:, 0]
    if np.any(np.abs(u_wall) > 1e-6) or np.any(np.abs(v_wall) > 1e-6):
        anomalies.append(
            f"U at wall (j=0): non-zero u_max={np.abs(u_wall).max():.2e} v_max={np.abs(v_wall).max():.2e} (no-slip expects 0)"
        )
    else:
        ok.append("U at wall (j=0): zero (no-slip OK)")

    # ----- Inlet profile (left column): expect variation in y for linear profile -----
    u_left = u[0, :]
    u_left_std = float(np.std(u_left))
    if u_left_std < 1e-10 and Nx * Ny > 1:
        anomalies.append("u at left (inlet): constant across y (std≈0) — linear profile may not be applied")
    else:
        ok.append(f"u at inlet: std={u_left_std:.6f} (varying in y)")

    # ----- Pressure -----
    p_min, p_max = float(P.min()), float(P.max())
    if not np.isfinite(p_min) or not np.isfinite(p_max):
        anomalies.append("P: non-finite")
    elif np.abs(p_max - p_min) > 1e10:
        anomalies.append(f"P: very large range [{p_min:.2e}, {p_max:.2e}]")
    else:
        ok.append(f"P: range [{p_min:.4f}, {p_max:.4f}]")

    # ----- Mass (integral of (1+phi)/2 for liquid volume; or sum phi) -----
    # Phase field: phi ≈ +1 liquid, ≈ -1 gas; (1+phi)/2 is liquid fraction
    liquid_frac = np.clip((1 + phi) / 2, 0, 1)
    mass_liquid = float(np.sum(liquid_frac) * dx * dy)
    ok.append(f"liquid volume ( (1+phi)/2 sum ) = {mass_liquid:.6f}")

    # ----- Staggered faces if present -----
    if "u_face" in data:
        u_face = np.asarray(data["u_face"])
        v_face = np.asarray(data["v_face"])
        if not np.all(np.isfinite(u_face)):
            anomalies.append("u_face: non-finite")
        else:
            ok.append(f"u_face: finite shape {u_face.shape}")
        if not np.all(np.isfinite(v_face)):
            anomalies.append("v_face: non-finite")
        else:
            ok.append(f"v_face: finite shape {v_face.shape}")

    # ----- Simple divergence check (cell-centered, interior) -----
    if Nx >= 3 and Ny >= 3:
        du_dx = (u[1:-1, 1:-1] - u[:-2, 1:-1]) / dx
        dv_dy = (v[1:-1, 1:-1] - v[1:-1, :-2]) / dy
        div = du_dx + dv_dy
        div_max = float(np.abs(div).max())
        div_mean = float(np.abs(div).mean())
        if div_max > 1.0:
            anomalies.append(f"|div U|: max={div_max:.4f} mean={div_mean:.4f} (large divergence)")
        else:
            ok.append(f"|div U|: max={div_max:.6f} mean={div_mean:.6f}")

    # ----- Report -----
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Step: {step}  shape: {Nx} x {Ny}")
    print()
    for s in ok:
        print(f"  OK  {s}")
    for s in anomalies:
        print(f"  !!  {s}")
    print()
    if anomalies:
        print(f"Summary: {len(anomalies)} anomaly(ies) found.")
    else:
        print("Summary: no anomalies detected.")
    return len(anomalies)


def main():
    parser = argparse.ArgumentParser(description="Check checkpoint for anomalies")
    parser.add_argument("checkpoint", type=str, help="Path to .npz or experiment dir")
    parser.add_argument("--step", type=int, default=None, help="Step when checkpoint is experiment dir")
    args = parser.parse_args()

    path = args.checkpoint
    if os.path.isdir(path):
        if args.step is None:
            print("Error: when checkpoint is a directory, pass --step N")
            sys.exit(1)
        path = os.path.join(path, "checkpoints", f"checkpoint_{args.step:06d}.npz")

    n = check_checkpoint(path)
    sys.exit(1 if n > 0 else 0)


if __name__ == "__main__":
    main()
