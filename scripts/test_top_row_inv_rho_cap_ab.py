#!/usr/bin/env python3
"""
A/B debug for top-adjacent row amplification with optional inv_rho cap
applied only in the staggered predictor pressure term.
"""

import argparse
import importlib
import json
import os
import sys
from typing import Dict, List, Optional

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulation.two_phase import TwoPhaseSimulation
from numerics.staggered_mac import grad_p_to_faces

sv = importlib.import_module("solvers.staggered_velocity")


def _make_capped_predictor(inv_rho_cap: Optional[float]):
    """Create predictor variant with optional inv_rho cap."""

    def _predictor(
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
        from numerics.staggered_mac import advect_u, advect_v, laplacian_u, laplacian_v
        from physics.properties import jax_calculate_density
        from numerics.finite_differences import jax_gradient

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

            if getattr(geometry, "has_geometry", False):
                grad_P = jax_gradient(P, dx, dy, geometry.f_1_grid)
                acc = -grad_P / rho[..., jnp.newaxis]
                ax = acc[..., 0]
                ay = acc[..., 1]
                ax_u = jnp.zeros((Nx + 1, Ny), dtype=ax.dtype)
                ax_u = ax_u.at[1:Nx, :].set(0.5 * (ax[1:, :] + ax[:-1, :]))
                ay_v = jnp.zeros((Nx, Ny + 1), dtype=ay.dtype)
                ay_v = ay_v.at[:, 1:Ny].set(0.5 * (ay[:, 1:] + ay[:, :-1]))
            else:
                inv_rho = 1.0 / rho
                if inv_rho_cap is not None:
                    inv_rho = jnp.minimum(inv_rho, float(inv_rho_cap))
                inv_rho_u = jnp.zeros((Nx + 1, Ny), dtype=inv_rho.dtype)
                inv_rho_v = jnp.zeros((Nx, Ny + 1), dtype=inv_rho.dtype)
                inv_rho_u = inv_rho_u.at[1:Nx, :].set(0.5 * (inv_rho[1:, :] + inv_rho[:-1, :]))
                inv_rho_u = inv_rho_u.at[0, :].set(inv_rho[0, :])
                inv_rho_u = inv_rho_u.at[Nx, :].set(inv_rho[-1, :])
                inv_rho_v = inv_rho_v.at[:, 1:Ny].set(0.5 * (inv_rho[:, 1:] + inv_rho[:, :-1]))
                inv_rho_v = inv_rho_v.at[:, 0].set(inv_rho[:, 0])
                inv_rho_v = inv_rho_v.at[:, Ny].set(inv_rho[:, -1])
                dpdx_face, dpdy_face = grad_p_to_faces(P, dx, dy)
                ax_u = -inv_rho_u * dpdx_face
                ay_v = -inv_rho_v * dpdy_face

            u_star = u_star + dt * ax_u
            v_star = v_star + dt * ay_v

        if include_gravity:
            gravity_v = (1.0 / max(Fr, 1e-6)) * g * jnp.ones_like(v_star)
            v_star = v_star + dt * gravity_v

        return u_star, v_star

    return _predictor


def _rowmax(sim: TwoPhaseSimulation) -> Dict[str, float]:
    U = np.array(sim.state.U)
    Ny = U.shape[1]
    mag = np.sqrt(U[..., 0] ** 2 + U[..., 1] ** 2)
    return {
        "U_jm3_max": float(np.max(mag[:, Ny - 3])),
        "U_jm2_max": float(np.max(mag[:, Ny - 2])),
        "U_jm1_max": float(np.max(mag[:, Ny - 1])),
    }


def _run_case(cfg: Dict, checkpoint: str, output_dir: str, inv_rho_cap: Optional[float], steps: int) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    cfg_local = json.loads(json.dumps(cfg))
    cfg_local.setdefault("restart", {})["restart_from"] = checkpoint

    sim = TwoPhaseSimulation(cfg_local, output_dir=output_dir)
    orig = sv.staggered_predictor_step
    sv.staggered_predictor_step = _make_capped_predictor(inv_rho_cap)
    logs: List[Dict] = []
    try:
        for k in range(int(steps)):
            sim._compute_cfl_dt()
            pre = _rowmax(sim)
            sim.state.t += sim.state.dt
            sim._predictor_step()
            post_pred = _rowmax(sim)
            sim._corrector_step()
            post_ppe = _rowmax(sim)
            logs.append(
                {
                    "step": int(sim.state.step),
                    "k": k,
                    "dt": float(sim.state.dt),
                    "pre_jm2": pre["U_jm2_max"],
                    "post_pred_jm2": post_pred["U_jm2_max"],
                    "post_ppe_jm2": post_ppe["U_jm2_max"],
                    "pre_jm1": pre["U_jm1_max"],
                    "post_ppe_jm1": post_ppe["U_jm1_max"],
                    "ppe_iters": float((sim._last_ppe_info or {}).get("iterations", 0.0)),
                }
            )
            sim._phase_update()
            sim._pressure_update()
            sim.state.step += 1
    finally:
        sv.staggered_predictor_step = orig

    return {
        "cap": None if inv_rho_cap is None else float(inv_rho_cap),
        "rows": logs,
        "jm2_pred_gain_mean": float(np.mean([r["post_pred_jm2"] - r["pre_jm2"] for r in logs])) if logs else 0.0,
        "jm2_ppe_net_mean": float(np.mean([r["post_ppe_jm2"] - r["pre_jm2"] for r in logs])) if logs else 0.0,
        "jm2_post_ppe_max": float(np.max([r["post_ppe_jm2"] for r in logs])) if logs else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="A/B top-row test with predictor inv_rho cap")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--caps", default="none,400,200,100")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    caps = []
    for tok in [t.strip().lower() for t in args.caps.split(",")]:
        if tok in ("none", "null", ""):
            caps.append(None)
        else:
            caps.append(float(tok))

    results = []
    for cap in caps:
        tag = "none" if cap is None else f"cap_{int(cap)}"
        print(f"\n=== Running {tag} ===")
        out = _run_case(
            cfg=cfg,
            checkpoint=args.checkpoint,
            output_dir=os.path.join(args.output_dir, tag),
            inv_rho_cap=cap,
            steps=int(args.steps),
        )
        results.append(out)

    out_path = os.path.join(args.output_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump({"config": args.config, "checkpoint": args.checkpoint, "results": results}, f, indent=2)

    print("\n=== Summary ===")
    for r in results:
        cap = "none" if r["cap"] is None else str(int(r["cap"]))
        print(
            f"cap={cap:>4s} "
            f"jm2_pred_gain_mean={r['jm2_pred_gain_mean']:.6e} "
            f"jm2_ppe_net_mean={r['jm2_ppe_net_mean']:.6e} "
            f"jm2_post_ppe_max={r['jm2_post_ppe_max']:.6e}"
        )
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
