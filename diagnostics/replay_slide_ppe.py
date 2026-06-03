#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from numerics.staggered_mac import divergence as mac_divergence
from simulation.state import SimulationState
from solvers.ppe import ppe_solve_staggered
from solvers.ppe_bc_derivation import derive_ppe_bcs_from_config


def div_stats(u_face, v_face, dx, dy):
    div = np.asarray(mac_divergence(u_face, v_face, dx, dy), dtype=np.float64)
    abs_div = np.abs(div)
    interior = abs_div[1:-1, 1:-1] if min(abs_div.shape) > 2 else abs_div
    argmax = np.unravel_index(np.argmax(abs_div), abs_div.shape)
    return {
        "global_max": float(abs_div.max()),
        "global_mean": float(abs_div.mean()),
        "interior_max": float(interior.max()),
        "argmax": (int(argmax[0]), int(argmax[1])),
        "right_col_max": float(abs_div[-1, :].max()),
        "bottom_row_max": float(abs_div[:, 0].max()),
        "top_row_max": float(abs_div[:, -1].max()),
    }


def print_stats(label, stats):
    print(
        f"{label:12s} "
        f"global_max={stats['global_max']:.6f} "
        f"interior_max={stats['interior_max']:.6f} "
        f"mean={stats['global_mean']:.6f} "
        f"right_col={stats['right_col_max']:.6f} "
        f"bottom_row={stats['bottom_row_max']:.6f} "
        f"top_row={stats['top_row_max']:.6f} "
        f"argmax={stats['argmax']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--force-iterations", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    state = SimulationState.from_config(cfg, restart_from=args.checkpoint)
    dx = float(state.dx)
    dy = float(state.dy)

    u0 = np.asarray(state.u_face, dtype=np.float64)
    v0 = np.asarray(state.v_face, dtype=np.float64)
    print(f"Checkpoint: {args.checkpoint}")
    print_stats("checkpoint", div_stats(u0, v0, dx, dy))

    u_bc, v_bc = state.velocity_bc.apply_to_faces(
        state.u_face, state.v_face, dx, dy, geometry=state.geometry, phi=state.phi
    )
    u_bc = np.asarray(u_bc, dtype=np.float64)
    v_bc = np.asarray(v_bc, dtype=np.float64)
    print_stats("after_bc", div_stats(u_bc, v_bc, dx, dy))

    ppe_bcs = derive_ppe_bcs_from_config(cfg)
    U_after, info = ppe_solve_staggered(
        state.U,
        dx,
        dy,
        float(state.dt),
        state.geometry,
        correction_solver=state.correction_solver,
        velocity_bc_manager=state.velocity_bc,
        ppe_bcs=ppe_bcs,
        psi=None,
        max_div_threshold=(-1.0 if args.force_iterations else 0.05),
        mean_div_threshold=(-1.0 if args.force_iterations else 0.1),
        max_iterations=args.iterations,
        phi=state.phi,
        rho1=float(state.rho1),
        rho2=float(state.rho2),
        u_face_in=u_bc,
        v_face_in=v_bc,
    )

    u_out = np.asarray(info["u_face_out"], dtype=np.float64)
    v_out = np.asarray(info["v_face_out"], dtype=np.float64)
    print_stats("after_ppe", div_stats(u_out, v_out, dx, dy))
    print(
        f"iterations={info['iterations']} "
        f"ppe_div_after_max={info['div_after_max']:.6f} "
        f"ppe_div_after_max_interior={info['div_after_max_interior']:.6f}"
    )


if __name__ == "__main__":
    main()
