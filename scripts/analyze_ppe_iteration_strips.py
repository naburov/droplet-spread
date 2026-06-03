#!/usr/bin/env python3
"""
Analyze PPE iteration-by-iteration divergence localization on staggered (MAC) grid.

This script replays the staggered PPE loop from a checkpoint and logs, for each PPE
iteration:
  - global/integral divergence metrics
  - hotspot cell index (i, j)
  - face-contribution split at hotspot: du = u_e - u_w, dv = v_n - v_s
  - top-strip / corner-strip diagnostics (for boundary-localized issues)

Usage:
  PYTHONPATH=src python scripts/analyze_ppe_iteration_strips.py \
    --config configs/static_droplet_contact_angle/contact_angle_60.json \
    --checkpoint experiment_x/checkpoints/checkpoint_000020.npz \
    --output_dir tmp_probe/ppe_strip_analysis \
    --max_iters 80
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, Any

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulation.state import SimulationState
from solvers.ppe_bc_derivation import derive_ppe_bcs_from_config
from numerics.staggered_utils import to_staggered, to_collocated
from numerics.staggered_mac import divergence as mac_divergence, grad_p_to_faces
from numerics.finite_differences import jax_divergence, jax_gradient


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _hotspot_metrics(u_face: np.ndarray, v_face: np.ndarray, div: np.ndarray) -> Dict[str, Any]:
    ad = np.abs(div)
    i, j = np.unravel_index(np.argmax(ad), ad.shape)
    u_w = float(u_face[i, j])
    u_e = float(u_face[i + 1, j])
    v_s = float(v_face[i, j])
    v_n = float(v_face[i, j + 1])
    du = u_e - u_w
    dv = v_n - v_s
    return {
        "hot_i": int(i),
        "hot_j": int(j),
        "hot_abs_div": float(ad[i, j]),
        "hot_div": float(div[i, j]),
        "hot_du": float(du),
        "hot_dv": float(dv),
        "hot_u_w": u_w,
        "hot_u_e": u_e,
        "hot_v_s": v_s,
        "hot_v_n": v_n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PPE strip/corner divergence analyzer.")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .npz")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--max_iters", type=int, default=80, help="Max PPE iterations to replay")
    parser.add_argument("--dt", type=float, default=None, help="Optional dt override")
    args = parser.parse_args()

    _ensure_dir(args.output_dir)

    with open(args.config, "r") as f:
        config = json.load(f)

    state = SimulationState.from_config(config, restart_from=args.checkpoint)
    if args.dt is not None:
        state.dt = float(args.dt)

    dt = float(state.dt)
    dx = float(state.dx)
    dy = float(state.dy)
    geometry = state.geometry
    correction_solver = state.correction_solver
    velocity_bc_manager = state.velocity_bc
    ppe_bcs = derive_ppe_bcs_from_config(config)

    Uj = jnp.array(state.U) if not isinstance(state.U, jnp.ndarray) else state.U
    if getattr(state, "u_face", None) is not None and getattr(state, "v_face", None) is not None:
        u_face = jnp.array(state.u_face)
        v_face = jnp.array(state.v_face)
    else:
        u_face, v_face = to_staggered(Uj)

    flat_geometry = not (geometry is not None and getattr(geometry, "has_geometry", False))
    div_history = []
    rows = []

    ppe_cfg = config.get("solver_params", {}).get("ppe", {})
    div_threshold = float(ppe_cfg.get("div_threshold", 0.01))
    max_div_threshold = float(ppe_cfg.get("max_div_threshold", 0.15))
    under_relaxation = float(ppe_cfg.get("under_relaxation", 1.0))

    for it in range(int(args.max_iters)):
        # Apply BCs before assembling PPE RHS.
        if flat_geometry and hasattr(velocity_bc_manager, "apply_to_faces"):
            u_face, v_face = velocity_bc_manager.apply_to_faces(
                u_face, v_face, dx, dy, psi=state.psi, geometry=geometry, phi=state.phi
            )
        else:
            U_cc = to_collocated(u_face, v_face)
            U_cc = velocity_bc_manager.apply_boundary_conditions(
                U_cc, dx, dy, use_jax=True, psi=state.psi, geometry=geometry
            )
            u_face, v_face = to_staggered(U_cc)

        if flat_geometry:
            div = mac_divergence(u_face, v_face, dx, dy)
        else:
            U_cc = to_collocated(u_face, v_face)
            div = jax_divergence(U_cc, dx, dy, geometry.f_1_grid)

        div_np = np.array(div)
        ad = np.abs(div_np)
        interior = ad[1:-1, 1:-1]
        max_div = float(np.max(ad))
        mean_div = float(np.mean(ad))
        max_div_interior = float(np.max(interior)) if interior.size > 0 else max_div

        # Boundary strip diagnostics (problem area: top-left / top strip).
        Ny = div_np.shape[1]
        strip_top_m1 = ad[:, Ny - 2] if Ny >= 2 else ad[:, -1]
        strip_top = ad[:, Ny - 1]
        tl5 = ad[:5, -5:]
        tr5 = ad[-5:, -5:]

        metrics = _hotspot_metrics(np.array(u_face), np.array(v_face), div_np)
        row = {
            "iter": it,
            "max_div": max_div,
            "mean_div": mean_div,
            "max_div_interior": max_div_interior,
            "top_strip_m1_max": float(np.max(strip_top_m1)),
            "top_strip_max": float(np.max(strip_top)),
            "tl5_max": float(np.max(tl5)),
            "tl5_mean": float(np.mean(tl5)),
            "tr5_max": float(np.max(tr5)),
            "tr5_mean": float(np.mean(tr5)),
            **metrics,
        }
        rows.append(row)
        div_history.append(div_np)

        # Convergence check.
        if max_div_interior <= max_div_threshold and mean_div <= div_threshold:
            break

        rhs_np = (1.0 / dt) * div_np
        if all(ppe_bcs.get(side) == "neumann" for side in ("left", "right", "bottom", "top")):
            rhs_np = rhs_np - np.mean(rhs_np)
        if ppe_bcs.get("left") == "dirichlet":
            rhs_np[0, :] = 0.0
        if ppe_bcs.get("right") == "dirichlet":
            rhs_np[-1, :] = 0.0
        if ppe_bcs.get("bottom") == "dirichlet":
            rhs_np[:, 0] = 0.0
        if ppe_bcs.get("top") == "dirichlet":
            rhs_np[:, -1] = 0.0
        rhs_np = np.nan_to_num(rhs_np, nan=0.0, posinf=0.0, neginf=0.0)

        p_corr = correction_solver.solve(rhs_np)
        p_corr = np.array(p_corr)
        if not np.all(np.isfinite(p_corr)):
            print(f"Stopping at iter {it}: non-finite pressure correction.")
            break

        if flat_geometry:
            dpdx_face, dpdy_face = grad_p_to_faces(jnp.array(p_corr), dx, dy)
            u_face = u_face - under_relaxation * dt * dpdx_face
            v_face = v_face - under_relaxation * dt * dpdy_face
        else:
            grad_p = jax_gradient(jnp.array(p_corr), dx, dy, geometry.f_1_grid)
            U_cc = to_collocated(u_face, v_face)
            U_cc = U_cc - under_relaxation * dt * grad_p
            u_face, v_face = to_staggered(U_cc)

        # Re-apply BCs after correction.
        if flat_geometry and hasattr(velocity_bc_manager, "apply_to_faces"):
            u_face, v_face = velocity_bc_manager.apply_to_faces(
                u_face, v_face, dx, dy, psi=state.psi, geometry=geometry, phi=state.phi
            )
        else:
            U_cc = to_collocated(u_face, v_face)
            U_cc = velocity_bc_manager.apply_boundary_conditions(
                U_cc, dx, dy, use_jax=True, psi=state.psi, geometry=geometry
            )
            u_face, v_face = to_staggered(U_cc)

    csv_path = os.path.join(args.output_dir, "ppe_iteration_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        os.path.join(args.output_dir, "ppe_iteration_divergence_maps.npz"),
        div_maps=np.array(div_history, dtype=np.float64),
    )

    summary = {
        "iterations_recorded": len(rows),
        "flat_geometry": flat_geometry,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "dt": dt,
        "csv": csv_path,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved metrics: {csv_path}")
    print(f"Saved maps: {os.path.join(args.output_dir, 'ppe_iteration_divergence_maps.npz')}")
    if rows:
        last = rows[-1]
        print(
            "Last iter:",
            f"iter={last['iter']}",
            f"max_div={last['max_div']:.6e}",
            f"max_div_interior={last['max_div_interior']:.6e}",
            f"hot=({last['hot_i']},{last['hot_j']})",
            f"du={last['hot_du']:.6e}",
            f"dv={last['hot_dv']:.6e}",
        )


if __name__ == "__main__":
    main()

