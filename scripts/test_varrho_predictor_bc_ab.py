#!/usr/bin/env python3
"""
A/B test for variable-density pressure coupling near bottom corners/contact line.

Compares:
  - legacy_cell_grad_corner_gauge:
      predictor uses cell-gradient -grad(P)/rho averaged to faces,
      variable-coefficient PPE gauge pinned at corner.
  - fixed_face_grad_center_gauge:
      predictor uses face-consistent -(1/rho_face)grad_face(P),
      variable-coefficient PPE gauge pinned at domain center.
"""

import argparse
import csv
import importlib
import json
import os
import sys
from typing import Dict, List

import jax.numpy as jnp
import numpy as np
import scipy.sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulation.two_phase import TwoPhaseSimulation
import solvers.staggered_velocity as sv
from numerics.finite_differences import jax_gradient
from numerics.staggered_mac import divergence as mac_divergence

ppe_mod = importlib.import_module("solvers.ppe")


def _legacy_predictor_cell_grad(
    u,
    v,
    surface_tension,
    dt: float,
    dx: float,
    dy: float,
    Re2: float,
    Fr: float,
    g: float,
    include_gravity: bool = False,
    include_advection: bool = True,
    P=None,
    phi=None,
    rho1=None,
    rho2=None,
    geometry=None,
):
    """Legacy predictor branch: cell-gradient -grad(P)/rho then average to faces."""
    from numerics.staggered_mac import advect_u, advect_v, laplacian_u, laplacian_v
    from physics.properties import jax_calculate_density

    if include_advection:
        Au = advect_u(u, v, dx, dy)
        Av = advect_v(u, v, dx, dy)
    else:
        Au = jnp.zeros_like(u)
        Av = jnp.zeros_like(v)

    nu = 1.0 / max(Re2, 1e-6)
    Lu = laplacian_u(u, dx, dy)
    Lv = laplacian_v(v, dx, dy)
    u_star = u + dt * (-Au + nu * Lu)
    v_star = v + dt * (-Av + nu * Lv)

    _ = surface_tension

    if P is not None and phi is not None and rho1 is not None and rho2 is not None and geometry is not None:
        Nx, Ny = u.shape[0] - 1, u.shape[1]
        rho = jax_calculate_density(phi, rho1, rho2)
        rho = jnp.maximum(rho, 1e-6)
        grad_P = jax_gradient(P, dx, dy, geometry.f_1_grid)
        acc = -grad_P / rho[..., jnp.newaxis]
        ax = acc[..., 0]
        ay = acc[..., 1]
        ax_u = jnp.zeros((Nx + 1, Ny), dtype=ax.dtype)
        ax_u = ax_u.at[1:Nx, :].set(0.5 * (ax[1:, :] + ax[:-1, :]))
        ay_v = jnp.zeros((Nx, Ny + 1), dtype=ay.dtype)
        ay_v = ay_v.at[:, 1:Ny].set(0.5 * (ay[:, 1:] + ay[:, :-1]))
        u_star = u_star + dt * ax_u
        v_star = v_star + dt * ay_v

    if include_gravity:
        gravity_v = (1.0 / max(Fr, 1e-6)) * g * jnp.ones_like(v_star)
        v_star = v_star + dt * gravity_v

    return u_star, v_star


def _build_varcoef_matrix_corner_gauge(inv_rho_u_face, inv_rho_v_face, dx, dy, ppe_bcs):
    """Legacy var-coeff matrix builder with corner gauge pin."""
    Nx = inv_rho_v_face.shape[0]
    Ny = inv_rho_u_face.shape[1]
    dx2 = dx * dx
    dy2 = dy * dy

    bcs = ppe_bcs or {}
    left_bc = bcs.get("left", "neumann")
    right_bc = bcs.get("right", "neumann")
    bottom_bc = bcs.get("bottom", "neumann")
    top_bc = bcs.get("top", "neumann")
    all_neumann = all(bc == "neumann" for bc in (left_bc, right_bc, bottom_bc, top_bc))

    A = scipy.sparse.lil_matrix((Nx * Ny, Nx * Ny), dtype=np.float64)

    def idx(i, j):
        return j * Nx + i

    for i in range(Nx):
        for j in range(Ny):
            on_left = i == 0
            on_right = i == Nx - 1
            on_bottom = j == 0
            on_top = j == Ny - 1
            is_dirichlet = (
                (on_left and left_bc == "dirichlet")
                or (on_right and right_bc == "dirichlet")
                or (on_bottom and bottom_bc == "dirichlet")
                or (on_top and top_bc == "dirichlet")
            )
            k = idx(i, j)
            if is_dirichlet:
                A[k, k] = 1.0
                continue

            beta_w = 0.0 if on_left else float(inv_rho_u_face[i, j])
            beta_e = 0.0 if on_right else float(inv_rho_u_face[i + 1, j])
            beta_s = 0.0 if on_bottom else float(inv_rho_v_face[i, j])
            beta_n = 0.0 if on_top else float(inv_rho_v_face[i, j + 1])

            diag = 0.0
            if i > 0:
                A[k, idx(i - 1, j)] = beta_w / dx2
                diag -= beta_w / dx2
            if i < Nx - 1:
                A[k, idx(i + 1, j)] = beta_e / dx2
                diag -= beta_e / dx2
            if j > 0:
                A[k, idx(i, j - 1)] = beta_s / dy2
                diag -= beta_s / dy2
            if j < Ny - 1:
                A[k, idx(i, j + 1)] = beta_n / dy2
                diag -= beta_n / dy2
            A[k, k] = diag

    gauge_ij = None
    if all_neumann:
        # Legacy gauge pin at corner.
        gauge_ij = (0, 0)
        k0 = idx(0, 0)
        A[k0, :] = 0.0
        A[k0, k0] = 1.0

    return A.tocsr(), all_neumann, gauge_ij


def _contact_window(phi: np.ndarray):
    phi_bottom = phi[:, 0]
    phi_above = phi[:, 1]
    contact = ((phi_bottom * phi_above) < 0.0) | (np.abs(phi_bottom) < 0.5)
    idx = np.where(contact)[0]
    if idx.size == 0:
        return 0, phi.shape[0] - 1
    return max(0, int(idx.min()) - 5), min(phi.shape[0] - 1, int(idx.max()) + 5)


def _run_scenario(cfg: Dict, output_dir: str, steps: int, mode: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    sim = TwoPhaseSimulation(cfg, output_dir=output_dir)

    orig_predictor = sv.staggered_predictor_step
    orig_builder = ppe_mod._build_variable_coefficient_ppe_matrix

    if mode == "legacy_cell_grad_corner_gauge":
        sv.staggered_predictor_step = _legacy_predictor_cell_grad
        ppe_mod._build_variable_coefficient_ppe_matrix = _build_varcoef_matrix_corner_gauge
    elif mode == "fixed_face_grad_center_gauge":
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")

    rows: List[Dict] = []
    try:
        for n in range(int(steps)):
            sim.step()
            u = np.array(sim.state.u_face)
            v = np.array(sim.state.v_face)
            phi = np.array(sim.state.phi)
            div = np.array(mac_divergence(jnp.array(u), jnp.array(v), sim.state.dx, sim.state.dy))
            ad = np.abs(div)

            lo, hi = _contact_window(phi)
            side = min(16, u.shape[0] // 8)
            row1 = np.abs(u[:, 1]) if u.shape[1] > 1 else np.abs(u[:, 0])
            rows.append(
                {
                    "n": n,
                    "step": int(sim.state.step),
                    "time": float(sim.state.t),
                    "dt": float(sim.state.dt),
                    "row1_u_global_max": float(np.max(row1)),
                    "row1_u_left_corner_max": float(np.max(row1[:side])),
                    "row1_u_right_corner_max": float(np.max(row1[-side:])),
                    "row1_u_contact_win_max": float(np.max(row1[lo : hi + 1])),
                    "div_row0_max": float(np.max(ad[:, 0])),
                    "div_row1_max": float(np.max(ad[:, 1])) if ad.shape[1] > 1 else 0.0,
                    "div_max": float(np.max(ad)),
                    "ppe_iters": float((sim._last_ppe_info or {}).get("iterations", 0.0)),
                }
            )
            sim.state.step += 1
    finally:
        sv.staggered_predictor_step = orig_predictor
        ppe_mod._build_variable_coefficient_ppe_matrix = orig_builder

    csv_path = os.path.join(output_dir, "per_step.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    if not rows:
        return {"mode": mode, "steps_done": 0, "per_step_csv": csv_path}

    summary = {
        "mode": mode,
        "steps_done": len(rows),
        "row1_u_global_peak": float(max(r["row1_u_global_max"] for r in rows)),
        "row1_u_corner_peak": float(
            max(max(r["row1_u_left_corner_max"], r["row1_u_right_corner_max"]) for r in rows)
        ),
        "row1_u_contact_peak": float(max(r["row1_u_contact_win_max"] for r in rows)),
        "div_row0_peak": float(max(r["div_row0_max"] for r in rows)),
        "div_peak": float(max(r["div_max"] for r in rows)),
        "mean_ppe_iters": float(sum(r["ppe_iters"] for r in rows) / len(rows)),
        "per_step_csv": csv_path,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="A/B test var-rho predictor + PPE gauge BC behavior")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--steps", type=int, default=40)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.config, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("restart", {})["restart_from"] = args.checkpoint

    scenarios = ["legacy_cell_grad_corner_gauge", "fixed_face_grad_center_gauge"]
    summaries = []
    for mode in scenarios:
        sc_dir = os.path.join(args.output_dir, mode)
        print(f"\n=== Running {mode} ===")
        summaries.append(_run_scenario(cfg, sc_dir, int(args.steps), mode))

    summary_path = os.path.join(args.output_dir, "ab_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"config": args.config, "checkpoint": args.checkpoint, "results": summaries}, f, indent=2)

    print("\n=== A/B summary ===")
    for s in summaries:
        if s.get("steps_done", 0) <= 0:
            print(f"{s['mode']}: no steps")
            continue
        print(
            f"{s['mode']:32s} "
            f"row1_corner_peak={s['row1_u_corner_peak']:.3e} "
            f"row1_contact_peak={s['row1_u_contact_peak']:.3e} "
            f"div_row0_peak={s['div_row0_peak']:.3e} "
            f"div_peak={s['div_peak']:.3e}"
        )
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
