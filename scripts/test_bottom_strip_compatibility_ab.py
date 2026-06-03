#!/usr/bin/env python3
"""
A/B test for bottom/contact-line strip compatibility in staggered PPE replay.

This script does NOT modify solver code. It replays PPE iterations from a checkpoint
and optionally applies a bottom-strip compatibility update near contact-line x-range:

  div(i,0) = (u[i+1,0]-u[i,0])/dx + (v[i,1]-v[i,0])/dy ~= 0
  => v[i,1] = v[i,0] - (dy/dx) * (u[i+1,0]-u[i,0])
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, Any, Tuple

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulation.state import SimulationState
from solvers.ppe_bc_derivation import derive_ppe_bcs_from_config
from numerics.staggered_utils import to_staggered, to_collocated
from numerics.staggered_mac import divergence as mac_divergence, grad_p_to_faces


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _contact_span(phi: np.ndarray, pad: int) -> Tuple[int, int]:
    phi0 = phi[:, 0]
    phi1 = phi[:, 1]
    contact_mask = ((phi0 * phi1) < 0.0) | (np.abs(phi0) < 0.5)
    idx = np.where(contact_mask)[0]
    if idx.size == 0:
        return (0, phi.shape[0] - 1)
    lo = max(0, int(idx.min()) - pad)
    hi = min(phi.shape[0] - 1, int(idx.max()) + pad)
    return lo, hi


def _apply_bottom_contact_strip_compat(
    u_face: jnp.ndarray,
    v_face: jnp.ndarray,
    dx: float,
    dy: float,
    i_lo: int,
    i_hi: int,
) -> jnp.ndarray:
    # Apply only for bottom-adjacent row (j=1) in the contact-line strip.
    for i in range(i_lo, i_hi + 1):
        v_new = v_face[i, 0] - (dy / dx) * (u_face[i + 1, 0] - u_face[i, 0])
        v_face = v_face.at[i, 1].set(v_new)
    return v_face


def _metrics(div: np.ndarray, i_lo: int, i_hi: int) -> Dict[str, Any]:
    ad = np.abs(div)
    i, j = np.unravel_index(np.argmax(ad), ad.shape)
    contact_slice = ad[i_lo:i_hi + 1, :]
    m = {
        "max_div": float(np.max(ad)),
        "mean_div": float(np.mean(ad)),
        "max_div_interior": float(np.max(ad[1:-1, 1:-1])),
        "hot_i": int(i),
        "hot_j": int(j),
        "hot_abs_div": float(ad[i, j]),
        "row0_max": float(np.max(ad[:, 0])),
        "row1_max": float(np.max(ad[:, 1])),
        "row2_max": float(np.max(ad[:, 2])),
        "contact_row0_max": float(np.max(ad[i_lo:i_hi + 1, 0])),
        "contact_row1_max": float(np.max(ad[i_lo:i_hi + 1, 1])),
        "contact_row2_max": float(np.max(ad[i_lo:i_hi + 1, 2])),
        "contact_band_max": float(np.max(contact_slice[:, :4])),
    }
    return m


def _replay(config: Dict[str, Any], checkpoint: str, max_iters: int, dt_override: float, enable_bottom_compat: bool, pad: int):
    state = SimulationState.from_config(config, restart_from=checkpoint)
    if dt_override is not None:
        state.dt = float(dt_override)

    dt = float(state.dt)
    dx = float(state.dx)
    dy = float(state.dy)
    correction_solver = state.correction_solver
    velocity_bc_manager = state.velocity_bc
    ppe_bcs = derive_ppe_bcs_from_config(config)

    if getattr(state, "u_face", None) is not None and getattr(state, "v_face", None) is not None:
        u_face = jnp.array(state.u_face)
        v_face = jnp.array(state.v_face)
    else:
        u_face, v_face = to_staggered(jnp.array(state.U))

    phi_np = np.array(state.phi)
    i_lo, i_hi = _contact_span(phi_np, pad=pad)
    rows = []

    ppe_cfg = config.get("solver_params", {}).get("ppe", {})
    div_threshold = float(ppe_cfg.get("div_threshold", 0.01))
    max_div_threshold = float(ppe_cfg.get("max_div_threshold", 0.15))
    under_relaxation = float(ppe_cfg.get("under_relaxation", 1.0))

    for it in range(int(max_iters)):
        u_face, v_face = velocity_bc_manager.apply_to_faces(
            u_face, v_face, dx, dy, psi=state.psi, geometry=state.geometry
        )
        if enable_bottom_compat:
            v_face = _apply_bottom_contact_strip_compat(u_face, v_face, dx, dy, i_lo, i_hi)

        div = np.array(mac_divergence(u_face, v_face, dx, dy))
        row = {"iter": it, **_metrics(div, i_lo, i_hi)}
        rows.append(row)
        if row["max_div_interior"] <= max_div_threshold and row["mean_div"] <= div_threshold:
            break

        rhs = (1.0 / dt) * div
        if all(ppe_bcs.get(side) == "neumann" for side in ("left", "right", "bottom", "top")):
            rhs = rhs - np.mean(rhs)
        for side in ("left", "right", "bottom", "top"):
            if ppe_bcs.get(side) == "dirichlet":
                if side == "left":
                    rhs[0, :] = 0.0
                elif side == "right":
                    rhs[-1, :] = 0.0
                elif side == "bottom":
                    rhs[:, 0] = 0.0
                elif side == "top":
                    rhs[:, -1] = 0.0

        p_corr = np.array(correction_solver.solve(np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)))
        dpdx_face, dpdy_face = grad_p_to_faces(jnp.array(p_corr), dx, dy)
        u_face = u_face - under_relaxation * dt * dpdx_face
        v_face = v_face - under_relaxation * dt * dpdy_face

        u_face, v_face = velocity_bc_manager.apply_to_faces(
            u_face, v_face, dx, dy, psi=state.psi, geometry=state.geometry
        )
        if enable_bottom_compat:
            v_face = _apply_bottom_contact_strip_compat(u_face, v_face, dx, dy, i_lo, i_hi)

    return rows, (i_lo, i_hi)


def main() -> None:
    p = argparse.ArgumentParser(description="A/B test bottom strip compatibility")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_iters", type=int, default=80)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--contact_pad", type=int, default=2)
    args = p.parse_args()

    _ensure_dir(args.output_dir)
    with open(args.config, "r") as f:
        cfg = json.load(f)

    summary = {}
    for mode, enabled in (("off", False), ("on", True)):
        rows, span = _replay(
            cfg, args.checkpoint, args.max_iters, args.dt, enabled, args.contact_pad
        )
        mode_dir = os.path.join(args.output_dir, mode)
        _ensure_dir(mode_dir)
        csv_path = os.path.join(mode_dir, "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        summary[mode] = {"last": rows[-1], "span": span, "csv": csv_path}

    def fv(mode: str, key: str) -> float:
        return float(summary[mode]["last"][key])

    print("\n=== Bottom strip compatibility A/B ===")
    print("contact span i:", summary["off"]["span"])
    for key in (
        "max_div",
        "max_div_interior",
        "row1_max",
        "contact_row1_max",
        "contact_band_max",
    ):
        print(f"{key:18s} off={fv('off', key):12.6g}  on={fv('on', key):12.6g}")
    print(
        "hotspot:",
        f"off=({summary['off']['last']['hot_i']},{summary['off']['last']['hot_j']})",
        f"on=({summary['on']['last']['hot_i']},{summary['on']['last']['hot_j']})",
    )


if __name__ == "__main__":
    main()

