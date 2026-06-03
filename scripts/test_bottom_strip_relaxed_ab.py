#!/usr/bin/env python3
"""
A/B test: hard vs relaxed+gated bottom strip compatibility.
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
from numerics.staggered_utils import to_staggered
from numerics.staggered_mac import divergence as mac_divergence, grad_p_to_faces


def _contact_span(phi: np.ndarray, pad: int) -> Tuple[int, int]:
    phi0 = phi[:, 0]
    phi1 = phi[:, 1]
    mask = ((phi0 * phi1) < 0.0) | (np.abs(phi0) < 0.5)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return (0, phi.shape[0] - 1)
    return max(0, int(idx.min()) - pad), min(phi.shape[0] - 1, int(idx.max()) + pad)


def _apply_hard(
    u_face: jnp.ndarray, v_face: jnp.ndarray, dx: float, dy: float, i_lo: int, i_hi: int
) -> jnp.ndarray:
    for i in range(i_lo, i_hi + 1):
        v_target = v_face[i, 0] - (dy / dx) * (u_face[i + 1, 0] - u_face[i, 0])
        v_face = v_face.at[i, 1].set(v_target)
    return v_face


def _apply_hard_guarded(
    u_face: jnp.ndarray,
    v_face: jnp.ndarray,
    dx: float,
    dy: float,
    i_lo: int,
    i_hi: int,
    du_cap: float,
) -> jnp.ndarray:
    for i in range(i_lo, i_hi + 1):
        du = u_face[i + 1, 0] - u_face[i, 0]
        if jnp.abs(du) > du_cap:
            continue
        v_target = v_face[i, 0] - (dy / dx) * du
        v_face = v_face.at[i, 1].set(v_target)
    return v_face


def _apply_relaxed_gated(
    u_face: jnp.ndarray,
    v_face: jnp.ndarray,
    phi: np.ndarray,
    dx: float,
    dy: float,
    i_lo: int,
    i_hi: int,
    beta: float,
    phi_cut: float,
    exclude_side: int,
) -> jnp.ndarray:
    Nx = phi.shape[0]
    for i in range(i_lo, i_hi + 1):
        if i < exclude_side or i > Nx - 1 - exclude_side:
            continue
        # Gate to interface band only; do not force deep bulk cells.
        if abs(phi[i, 0]) > phi_cut and abs(phi[i, 1]) > phi_cut:
            continue
        v_target = v_face[i, 0] - (dy / dx) * (u_face[i + 1, 0] - u_face[i, 0])
        v_new = (1.0 - beta) * v_face[i, 1] + beta * v_target
        v_face = v_face.at[i, 1].set(v_new)
    return v_face


def _apply_capped_gated(
    u_face: jnp.ndarray,
    v_face: jnp.ndarray,
    phi: np.ndarray,
    dx: float,
    dy: float,
    i_lo: int,
    i_hi: int,
    phi_cut: float,
    exclude_side: int,
    delta_cap: float,
) -> jnp.ndarray:
    Nx = phi.shape[0]
    for i in range(i_lo, i_hi + 1):
        if i < exclude_side or i > Nx - 1 - exclude_side:
            continue
        if abs(phi[i, 0]) > phi_cut and abs(phi[i, 1]) > phi_cut:
            continue
        v_target = v_face[i, 0] - (dy / dx) * (u_face[i + 1, 0] - u_face[i, 0])
        dv = v_target - v_face[i, 1]
        dv = jnp.clip(dv, -delta_cap, delta_cap)
        v_face = v_face.at[i, 1].set(v_face[i, 1] + dv)
    return v_face


def _metrics(div: np.ndarray, i_lo: int, i_hi: int) -> Dict[str, Any]:
    ad = np.abs(div)
    i, j = np.unravel_index(np.argmax(ad), ad.shape)
    return {
        "max_div": float(np.max(ad)),
        "mean_div": float(np.mean(ad)),
        "max_div_interior": float(np.max(ad[1:-1, 1:-1])),
        "row1_max": float(np.max(ad[:, 1])),
        "contact_row1_max": float(np.max(ad[i_lo:i_hi + 1, 1])),
        "contact_band_max": float(np.max(ad[i_lo:i_hi + 1, :4])),
        "hot_i": int(i),
        "hot_j": int(j),
        "hot_abs_div": float(ad[i, j]),
    }


def _replay(cfg: Dict[str, Any], checkpoint: str, max_iters: int, dt: float, mode: str, beta: float, phi_cut: float, exclude_side: int, pad: int, delta_cap: float, du_cap: float):
    state = SimulationState.from_config(cfg, restart_from=checkpoint)
    if dt is not None:
        state.dt = float(dt)

    # Disable integrated bottom strip rule so this script compares manual variants only.
    state.velocity_bc.config.setdefault("boundary_conditions", {}).setdefault("velocity", {})[
        "bottom_strip_compatibility"
    ] = False

    u_face = jnp.array(state.u_face) if state.u_face is not None else to_staggered(jnp.array(state.U))[0]
    v_face = jnp.array(state.v_face) if state.v_face is not None else to_staggered(jnp.array(state.U))[1]
    phi_np = np.array(state.phi)
    i_lo, i_hi = _contact_span(phi_np, pad=pad)

    dtv = float(state.dt)
    dx = float(state.dx)
    dy = float(state.dy)
    ppe_bcs = derive_ppe_bcs_from_config(cfg)
    under_relax = float(cfg.get("solver_params", {}).get("ppe", {}).get("under_relaxation", 1.0))
    rows = []

    for it in range(int(max_iters)):
        u_face, v_face = state.velocity_bc.apply_to_faces(
            u_face, v_face, dx, dy, psi=state.psi, geometry=state.geometry, phi=state.phi
        )
        if mode == "hard":
            v_face = _apply_hard(u_face, v_face, dx, dy, i_lo, i_hi)
        elif mode == "hard_guarded":
            v_face = _apply_hard_guarded(u_face, v_face, dx, dy, i_lo, i_hi, du_cap)
        elif mode == "relaxed":
            v_face = _apply_relaxed_gated(u_face, v_face, phi_np, dx, dy, i_lo, i_hi, beta, phi_cut, exclude_side)
        elif mode == "capped":
            v_face = _apply_capped_gated(u_face, v_face, phi_np, dx, dy, i_lo, i_hi, phi_cut, exclude_side, delta_cap)

        div = np.array(mac_divergence(u_face, v_face, dx, dy))
        row = {"iter": it, **_metrics(div, i_lo, i_hi)}
        rows.append(row)

        rhs = (1.0 / dtv) * div
        if all(ppe_bcs.get(s) == "neumann" for s in ("left", "right", "bottom", "top")):
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

        p_corr = np.array(state.correction_solver.solve(np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)))
        dpdx, dpdy = grad_p_to_faces(jnp.array(p_corr), dx, dy)
        u_face = u_face - under_relax * dtv * dpdx
        v_face = v_face - under_relax * dtv * dpdy

    return rows, (i_lo, i_hi)


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B test relaxed bottom strip compatibility")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_iters", type=int, default=12)
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--phi_cut", type=float, default=0.8)
    ap.add_argument("--exclude_side", type=int, default=2)
    ap.add_argument("--contact_pad", type=int, default=2)
    ap.add_argument("--delta_cap", type=float, default=20.0)
    ap.add_argument("--du_cap", type=float, default=80.0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = json.load(open(args.config))

    summary = {}
    for mode in ("hard", "hard_guarded", "capped", "relaxed"):
        rows, span = _replay(
            cfg, args.checkpoint, args.max_iters, args.dt, mode,
            args.beta, args.phi_cut, args.exclude_side, args.contact_pad, args.delta_cap, args.du_cap
        )
        mode_dir = os.path.join(args.output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        csv_path = os.path.join(mode_dir, "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        summary[mode] = {"last": rows[-1], "span": span}

    print("\n=== Bottom compatibility: hard vs hard-guarded vs capped vs relaxed ===")
    print("contact span:", summary["hard"]["span"])
    print(
        f"beta={args.beta} phi_cut={args.phi_cut} exclude_side={args.exclude_side} "
        f"delta_cap={args.delta_cap} du_cap={args.du_cap}"
    )
    for k in ("max_div", "max_div_interior", "row1_max", "contact_row1_max", "contact_band_max"):
        print(
            f"{k:18s} hard={float(summary['hard']['last'][k]):12.6g}  "
            f"hardG={float(summary['hard_guarded']['last'][k]):12.6g}  "
            f"capped={float(summary['capped']['last'][k]):12.6g}  "
            f"relaxed={float(summary['relaxed']['last'][k]):12.6g}"
        )
    print(
        "hotspot:",
        f"hard=({summary['hard']['last']['hot_i']},{summary['hard']['last']['hot_j']})",
        f"hardG=({summary['hard_guarded']['last']['hot_i']},{summary['hard_guarded']['last']['hot_j']})",
        f"capped=({summary['capped']['last']['hot_i']},{summary['capped']['last']['hot_j']})",
        f"relaxed=({summary['relaxed']['last']['hot_i']},{summary['relaxed']['last']['hot_j']})",
    )


if __name__ == "__main__":
    main()

