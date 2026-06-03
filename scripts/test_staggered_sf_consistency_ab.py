#!/usr/bin/env python3
"""
A/B test for staggered predictor SF coupling consistency.

Compares:
  - legacy_sf_plus: historical staggered coupling (+SF on faces)
  - consistent_sf_rho: collocated-consistent coupling (-SF/rho on faces)

Runs each scenario from the same checkpoint and logs per-step diagnostics.
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

from numerics.staggered_mac import advect_u, advect_v, divergence as mac_divergence, laplacian_u, laplacian_v
from simulation.two_phase import TwoPhaseSimulation
import solvers.staggered_velocity as sv


def _legacy_predictor_step(
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
    """Legacy staggered predictor branch (+SF), kept for controlled A/B only."""
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

    fx = surface_tension[..., 0]
    fy = surface_tension[..., 1]
    Nx, Ny = fx.shape

    fx_u = jnp.zeros((Nx + 1, Ny), dtype=fx.dtype)
    fx_u = fx_u.at[1:Nx, :].set(0.5 * (fx[1:, :] + fx[:-1, :]))
    fy_v = jnp.zeros((Nx, Ny + 1), dtype=fy.dtype)
    fy_v = fy_v.at[:, 1:Ny].set(0.5 * (fy[:, 1:] + fy[:, :-1]))
    u_star = u_star + dt * fx_u
    v_star = v_star + dt * fy_v

    if P is not None and phi is not None and rho1 is not None and rho2 is not None and geometry is not None:
        from physics.properties import jax_calculate_density
        from numerics.finite_differences import jax_gradient

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


def _liquid_com_y(phi: np.ndarray, dy: float) -> float:
    y = (np.arange(phi.shape[1]) + 0.5) * dy
    w = np.clip((1.0 - phi) * 0.5, 0.0, 1.0)  # liquid fraction
    m = np.sum(w)
    if m <= 0.0:
        return float("nan")
    return float(np.sum(w * y[None, :]) / m)


def _run_scenario(cfg: Dict, output_dir: str, steps: int, mode: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    sim = TwoPhaseSimulation(cfg, output_dir=output_dir)

    original_predictor = sv.staggered_predictor_step
    if mode == "legacy_sf_plus":
        sv.staggered_predictor_step = _legacy_predictor_step
    elif mode == "consistent_sf_rho":
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

            rows.append(
                {
                    "n": n,
                    "step": int(sim.state.step),
                    "time": float(sim.state.t),
                    "dt": float(sim.state.dt),
                    "max_div": float(np.max(ad)),
                    "mean_div": float(np.mean(ad)),
                    "row1_max_div": float(np.max(ad[:, 1])),
                    "strip_max_u_j1_j3": float(np.max(np.abs(u[:, 1:4]))),
                    "strip_max_v_j1_j3": float(np.max(np.abs(v[:, 1:4]))),
                    "com_y": _liquid_com_y(phi, float(sim.state.dy)),
                }
            )
            sim.state.step += 1
    finally:
        sv.staggered_predictor_step = original_predictor

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
        "max_div_peak": float(max(r["max_div"] for r in rows)),
        "row1_div_peak": float(max(r["row1_max_div"] for r in rows)),
        "strip_max_u_peak": float(max(r["strip_max_u_j1_j3"] for r in rows)),
        "strip_max_u_end": float(rows[-1]["strip_max_u_j1_j3"]),
        "strip_max_v_peak": float(max(r["strip_max_v_j1_j3"] for r in rows)),
        "com_y_start": float(rows[0]["com_y"]),
        "com_y_end": float(rows[-1]["com_y"]),
        "com_y_delta": float(rows[-1]["com_y"] - rows[0]["com_y"]),
        "avg_dt": float(sum(r["dt"] for r in rows) / len(rows)),
        "per_step_csv": csv_path,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="A/B test staggered SF consistency")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--steps", type=int, default=50)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.config, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("restart", {})["restart_from"] = args.checkpoint

    scenarios = ["legacy_sf_plus", "consistent_sf_rho"]
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
            f"{s['mode']:18s} "
            f"strip_u_end={s['strip_max_u_end']:.3e} "
            f"strip_u_peak={s['strip_max_u_peak']:.3e} "
            f"max_div_peak={s['max_div_peak']:.3e} "
            f"com_dy={s['com_y_delta']:.3e}"
        )
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()

