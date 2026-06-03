#!/usr/bin/env python3
"""
Diagnose top-adjacent row (j=-2) velocity onset over 1-2 steps.

Logs per stage:
  - pre_predictor
  - post_predictor
  - post_ppe
  - post_full_step

for each step, focusing on j=-2 (and neighbors j=-3, j=-1 for comparison).
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulation.two_phase import TwoPhaseSimulation
from numerics.staggered_mac import grad_p_to_faces
from physics.properties import jax_calculate_density


def _inv_rho_faces(phi, rho1, rho2):
    rho_cc = np.array(jax_calculate_density(jnp.array(phi), rho1, rho2))
    rho_cc = np.maximum(rho_cc, 1e-6)
    inv_cc = 1.0 / rho_cc
    Nx, Ny = rho_cc.shape
    inv_u = np.zeros((Nx + 1, Ny), dtype=np.float64)
    inv_v = np.zeros((Nx, Ny + 1), dtype=np.float64)
    inv_u[1:Nx, :] = 0.5 * (inv_cc[1:, :] + inv_cc[:-1, :])
    inv_u[0, :] = inv_cc[0, :]
    inv_u[Nx, :] = inv_cc[-1, :]
    inv_v[:, 1:Ny] = 0.5 * (inv_cc[:, 1:] + inv_cc[:, :-1])
    inv_v[:, 0] = inv_cc[:, 0]
    inv_v[:, Ny] = inv_cc[:, -1]
    return inv_cc, inv_u, inv_v


def _row_stats(arr_1d: np.ndarray) -> Dict[str, float]:
    d1 = np.diff(arr_1d)
    d2 = np.diff(arr_1d, n=2) if arr_1d.size >= 3 else np.array([0.0])
    return {
        "max_abs": float(np.max(np.abs(arr_1d))),
        "mean_abs": float(np.mean(np.abs(arr_1d))),
        "max_abs_d1": float(np.max(np.abs(d1))) if d1.size > 0 else 0.0,
        "max_abs_d2": float(np.max(np.abs(d2))) if d2.size > 0 else 0.0,
        "argmax_abs_d2": int(np.argmax(np.abs(d2))) if d2.size > 0 else -1,
    }


def _snapshot(sim: TwoPhaseSimulation, step_idx: int, stage: str, focus_n: int = 12) -> Dict:
    U = np.array(sim.state.U)
    u = np.array(sim.state.u_face)
    v = np.array(sim.state.v_face)
    phi = np.array(sim.state.phi)
    P = np.array(sim.state.P)
    Nx, Ny = phi.shape
    j_top = Ny - 1
    j_top_adj = Ny - 2
    j_top_adj2 = Ny - 3

    mag = np.sqrt(U[..., 0] ** 2 + U[..., 1] ** 2)
    inv_cc, inv_u, _ = _inv_rho_faces(phi, sim.state.rho1, sim.state.rho2)
    dpdx_face, _ = grad_p_to_faces(jnp.array(P), float(sim.state.dx), float(sim.state.dy))
    dpdx_face = np.array(dpdx_face)
    ax_u = -inv_u * dpdx_face

    row_u_top_adj = u[:, j_top_adj]
    row_Umag_top_adj = mag[:, j_top_adj]
    row_ax_top_adj = ax_u[:, j_top_adj]
    row_dpdx_top_adj = dpdx_face[:, j_top_adj]
    row_inv_u_top_adj = inv_u[:, j_top_adj]

    # First cells where user sees kink.
    k = min(focus_n, row_u_top_adj.shape[0])
    first_slice = slice(0, k)

    out = {
        "step_index": int(step_idx),
        "stage": stage,
        "time": float(sim.state.t),
        "dt": float(sim.state.dt),
        "j_top": int(j_top),
        "j_top_adj": int(j_top_adj),
        "u_top_row_max_abs": float(np.max(np.abs(u[:, j_top]))),
        "u_top_adj_row_max_abs": float(np.max(np.abs(row_u_top_adj))),
        "U_top_adj_row_max_abs": float(np.max(np.abs(row_Umag_top_adj))),
        "U_top_adj_row_argmax": int(np.argmax(np.abs(row_Umag_top_adj))),
        "U_top_adj2_row_max_abs": float(np.max(np.abs(mag[:, j_top_adj2]))),
        "U_bottom_row_max_abs": float(np.max(np.abs(mag[:, 0]))),
        "U_first_row_max_abs": float(np.max(np.abs(mag[:, 1]))),
        "inv_rho_top_adj_u_mean": float(np.mean(row_inv_u_top_adj)),
        "inv_rho_top_adj_u_max": float(np.max(row_inv_u_top_adj)),
        "dpdx_top_adj_max_abs": float(np.max(np.abs(row_dpdx_top_adj))),
        "ax_top_adj_max_abs": float(np.max(np.abs(row_ax_top_adj))),
        "u_top_adj_stats": _row_stats(row_u_top_adj),
        "Umag_top_adj_stats": _row_stats(row_Umag_top_adj),
        "dpdx_top_adj_stats": _row_stats(row_dpdx_top_adj),
        "ax_top_adj_stats": _row_stats(row_ax_top_adj),
        "samples_first_cells": {
            "u_top_adj": row_u_top_adj[first_slice].tolist(),
            "Umag_top_adj": row_Umag_top_adj[first_slice].tolist(),
            "dpdx_top_adj": row_dpdx_top_adj[first_slice].tolist(),
            "inv_u_top_adj": row_inv_u_top_adj[first_slice].tolist(),
            "ax_top_adj": row_ax_top_adj[first_slice].tolist(),
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="Debug top-adjacent row onset over 1-2 steps")
    ap.add_argument("--config", required=True, help="Path to config json")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint npz")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.config, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("restart", {})["restart_from"] = args.checkpoint

    sim = TwoPhaseSimulation(cfg, output_dir=args.output_dir)

    snaps: List[Dict] = []
    for n in range(int(args.steps)):
        sim._compute_cfl_dt()
        snaps.append(_snapshot(sim, n, "pre_predictor"))

        sim.state.t += sim.state.dt
        sim._predictor_step()
        snaps.append(_snapshot(sim, n, "post_predictor"))

        sim._corrector_step()
        snaps.append(_snapshot(sim, n, "post_ppe"))

        sim._phase_update()
        sim._pressure_update()
        snaps.append(_snapshot(sim, n, "post_full_step"))

        sim.state.step += 1

    # Write JSON (full detail)
    json_path = os.path.join(args.output_dir, "top_row_onset_debug.json")
    with open(json_path, "w") as f:
        json.dump(snaps, f, indent=2)

    # Write compact CSV
    csv_path = os.path.join(args.output_dir, "top_row_onset_debug.csv")
    fieldnames = [
        "step_index",
        "stage",
        "time",
        "dt",
        "u_top_row_max_abs",
        "u_top_adj_row_max_abs",
        "U_top_adj_row_max_abs",
        "U_top_adj_row_argmax",
        "U_top_adj2_row_max_abs",
        "U_bottom_row_max_abs",
        "U_first_row_max_abs",
        "inv_rho_top_adj_u_mean",
        "inv_rho_top_adj_u_max",
        "dpdx_top_adj_max_abs",
        "ax_top_adj_max_abs",
        "u_top_adj_stats.max_abs_d2",
        "Umag_top_adj_stats.max_abs_d2",
        "dpdx_top_adj_stats.max_abs_d2",
        "ax_top_adj_stats.max_abs_d2",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in snaps:
            row = dict(s)
            row["u_top_adj_stats.max_abs_d2"] = s["u_top_adj_stats"]["max_abs_d2"]
            row["Umag_top_adj_stats.max_abs_d2"] = s["Umag_top_adj_stats"]["max_abs_d2"]
            row["dpdx_top_adj_stats.max_abs_d2"] = s["dpdx_top_adj_stats"]["max_abs_d2"]
            row["ax_top_adj_stats.max_abs_d2"] = s["ax_top_adj_stats"]["max_abs_d2"]
            w.writerow({k: row.get(k, "") for k in fieldnames})

    print("Saved JSON:", json_path)
    print("Saved CSV :", csv_path)
    print("\nQuick summary:")
    for s in snaps:
        print(
            f"step={s['step_index']} {s['stage']:14s} "
            f"|U|_j-2_max={s['U_top_adj_row_max_abs']:.6f} "
            f"argmax_i={s['U_top_adj_row_argmax']:3d} "
            f"max|dpdx|={s['dpdx_top_adj_max_abs']:.3e} "
            f"max|ax|={s['ax_top_adj_max_abs']:.3e}"
        )


if __name__ == "__main__":
    main()
