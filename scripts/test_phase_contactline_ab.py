#!/usr/bin/env python3
"""
A/B test for two contact-line hypotheses:
  1) Phase clipping timing: clip before BC (baseline) vs clip after BC.
  2) Contact-angle gradient location: grad at j=1 (baseline) vs grad at j=0.

Runs from a checkpoint and compares stability/contact-line diagnostics.
"""

import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulation.two_phase import TwoPhaseSimulation
from numerics.staggered_mac import divergence as mac_divergence
from numerics.finite_differences import jax_gradient, jax_laplacian, jax_norm
from physics.properties import jax_df_2, jax_calculate_density
import physics.phase_field as phase_field_mod
import boundary_conditions.contact_angle_bc as cab_mod


def _contact_span(phi: np.ndarray, pad: int = 2) -> Optional[Tuple[int, int]]:
    phi0 = phi[:, 0]
    phi1 = phi[:, 1]
    mask = ((phi0 * phi1) < 0.0) | (np.abs(phi0) < 0.5)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    return max(0, int(idx.min()) - pad), min(phi.shape[0] - 1, int(idx.max()) + pad)


def _liquid_com_y(phi: np.ndarray, dy: float) -> float:
    y = (np.arange(phi.shape[1]) + 0.5) * dy
    w = np.clip((1.0 - phi) * 0.5, 0.0, 1.0)  # liquid fraction
    m = np.sum(w)
    if m <= 0:
        return float("nan")
    return float(np.sum(w * y[None, :]) / m)


def _sf_pg_alignment(phi: np.ndarray, P: np.ndarray, sf: np.ndarray, dx: float, dy: float, f_1_grid, rho1: float, rho2: float) -> float:
    rho = np.array(jax_calculate_density(jnp.array(phi), rho1, rho2))
    inv_rho = 1.0 / np.maximum(rho, 1e-6)
    gradP = np.array(jax_gradient(jnp.array(P), dx, dy, f_1_grid))
    a_pg = -gradP * inv_rho[..., None]
    a_st = -sf * inv_rho[..., None]
    mask = np.abs(phi) < 0.5
    if not np.any(mask):
        return float("nan")
    v1 = a_st[mask]
    v2 = a_pg[mask]
    num = np.sum(v1 * v2, axis=1)
    den = np.maximum(np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), 1e-12)
    return float(np.mean(num / den))


def _jax_update_phase_no_clip(phi, U, current_dt, dx, dy, Pe, epsilon, contact_angle, f_1_grid, f_2_grid, lambda_willmore=0.0, epsilon_willmore=0.0):
    # Mirror original update but do not clip; used only for controlled A/B.
    grad_phi = jax_gradient(phi, dx, dy, f_1_grid)
    convective_term = U[..., 0] * grad_phi[..., 0] + U[..., 1] * grad_phi[..., 1]
    lap_phi = jax_laplacian(phi, dx, dy, f_1_grid, f_2_grid)
    mu_ch = jax_df_2(phi) - epsilon**2 * lap_phi
    from boundary_conditions.chemical_potential_bc import jax_apply_chemical_potential_zero_flux_bc
    mu_ch = jax_apply_chemical_potential_zero_flux_bc(mu_ch, dx, dy)

    willmore_active = (lambda_willmore > 0) & (epsilon_willmore > 0)
    mu_willmore = phase_field_mod.jax_willmore_chemical_potential(phi, dx, dy, f_1_grid, f_2_grid, epsilon_willmore)
    mu_willmore = jax_apply_chemical_potential_zero_flux_bc(mu_willmore, dx, dy)
    mu_total = mu_ch + jnp.where(willmore_active, lambda_willmore * mu_willmore, 0.0)

    lagrange_multiplier = jnp.mean(mu_total)
    source_term = -1 / Pe * (mu_total - lagrange_multiplier)
    rhs_phi = -convective_term + source_term
    return phi + current_dt * rhs_phi


@contextmanager
def _patched_phase_contact_logic(clip_after_bc: bool, simple_grad_j0: bool):
    orig_simple = cab_mod.ContactAngleBoundaryCondition._simple_contact_angle_jax
    orig_update = phase_field_mod.PhaseFieldSolver.update

    def simple_j0(self, phi, dx, dy, geometry, psi=None, U=None, bottom_velocity_bc="no_slip"):
        theta_effective = self._get_effective_contact_angle_jax(psi, U=U, contact_line_velocity=None, bottom_velocity_bc=bottom_velocity_bc)
        grad_phi = jax_gradient(phi, dx, dy, geometry.f_1_grid)
        grad_phi_x = grad_phi[:, 0, 0]
        grad_phi_y = grad_phi[:, 0, 1]
        norm_grad_phi = jnp.sqrt(grad_phi_x**2 + grad_phi_y**2)
        norm_grad_phi = jnp.where(norm_grad_phi < 1e-10, 1e-10, norm_grad_phi)
        normal_derivative = -jnp.cos(theta_effective) * norm_grad_phi

        phi_bottom = phi[:, 0]
        phi_above = phi[:, 1]
        phi_crosses_zero = (phi_bottom * phi_above) < 0
        phi_near_zero = jnp.abs(phi_bottom) < 0.5
        has_interface = norm_grad_phi > 1e-3
        contact_mask = (phi_crosses_zero | phi_near_zero) & has_interface

        phi_bottom_advected = phi[:, 0]
        phi_contact_target = phi[:, 1] - normal_derivative * dy
        phi_neumann_target = phi[:, 1]

        alpha = jnp.clip(self.contact_angle_relaxation, 0.0, 1.0)
        phi_contact_blended = (1.0 - alpha) * phi_bottom_advected + alpha * phi_contact_target
        phi_neumann_blended = (1.0 - alpha) * phi_bottom_advected + alpha * phi_neumann_target

        phi_bottom_new = jnp.where(contact_mask, phi_contact_blended, phi_neumann_blended)
        phi_new = phi.at[:, 0].set(phi_bottom_new)
        phi_new = phi_new.at[:, -1].set(phi_new[:, -2])
        phi_new = phi_new.at[0, :].set(phi_new[1, :])
        phi_new = phi_new.at[-1, :].set(phi_new[-2, :])
        return phi_new

    def update_clip_after(self, phi, U, current_dt, dx, dy, geometry, use_jax=True, psi=None):
        phi_new = _jax_update_phase_no_clip(
            phi,
            U,
            current_dt,
            dx,
            dy,
            self.Pe,
            self.epsilon,
            self.contact_angle,
            geometry.f_1_grid,
            geometry.f_2_grid,
            lambda_willmore=self.lambda_willmore,
            epsilon_willmore=self.epsilon_willmore,
        )
        if self.advection_bc_manager is not None:
            phi_new = self.advection_bc_manager.apply_boundary_conditions(
                phi_new, U, current_dt, dx, dy, use_jax=True, geometry=geometry
            )
        if self.bc_manager is not None:
            phi_new = self.bc_manager.apply_boundary_conditions(
                phi_new, dx, dy, use_jax=True, psi=psi, U=U, geometry=geometry
            )
        return jnp.clip(phi_new, -1.0, 1.0)

    try:
        if simple_grad_j0:
            cab_mod.ContactAngleBoundaryCondition._simple_contact_angle_jax = simple_j0
        if clip_after_bc:
            phase_field_mod.PhaseFieldSolver.update = update_clip_after
        yield
    finally:
        cab_mod.ContactAngleBoundaryCondition._simple_contact_angle_jax = orig_simple
        phase_field_mod.PhaseFieldSolver.update = orig_update


def _run(cfg: Dict, output_dir: str, steps: int, clip_after_bc: bool, simple_grad_j0: bool) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    with _patched_phase_contact_logic(clip_after_bc=clip_after_bc, simple_grad_j0=simple_grad_j0):
        sim = TwoPhaseSimulation(cfg, output_dir=output_dir)
        rows: List[Dict] = []
        for n in range(int(steps)):
            sim.step()
            u = np.array(sim.state.u_face)
            v = np.array(sim.state.v_face)
            phi = np.array(sim.state.phi)
            P = np.array(sim.state.P)
            sf = np.array(sim.state.compute_surface_tension())
            div = np.array(mac_divergence(jnp.array(u), jnp.array(v), sim.state.dx, sim.state.dy))
            ad = np.abs(div)
            span = _contact_span(phi, pad=2)
            contact_row1 = float("nan")
            if span is not None:
                lo, hi = span
                contact_row1 = float(np.max(ad[lo : hi + 1, 1]))

            rows.append(
                {
                    "n": n,
                    "step": int(sim.state.step),
                    "time": float(sim.state.t),
                    "dt": float(sim.state.dt),
                    "max_div": float(np.max(ad)),
                    "row1_max_div": float(np.max(ad[:, 1])),
                    "contact_row1_max_div": contact_row1,
                    "strip_max_u_j1_j3": float(np.max(np.abs(u[:, 1:4]))),
                    "strip_max_v_j1_j3": float(np.max(np.abs(v[:, 1:4]))),
                    "com_y": _liquid_com_y(phi, float(sim.state.dy)),
                    "sf_pg_cos_interface": _sf_pg_alignment(
                        phi,
                        P,
                        sf,
                        float(sim.state.dx),
                        float(sim.state.dy),
                        sim.state.geometry.f_1_grid,
                        float(sim.state.rho1),
                        float(sim.state.rho2),
                    ),
                }
            )
            sim.state.step += 1

    csv_path = os.path.join(output_dir, "per_step.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    if not rows:
        return {"steps_done": 0, "per_step_csv": csv_path}
    return {
        "steps_done": len(rows),
        "max_div_peak": float(max(r["max_div"] for r in rows)),
        "row1_div_peak": float(max(r["row1_max_div"] for r in rows)),
        "contact_row1_div_peak": float(np.nanmax([r["contact_row1_max_div"] for r in rows])),
        "strip_u_peak": float(max(r["strip_max_u_j1_j3"] for r in rows)),
        "strip_u_end": float(rows[-1]["strip_max_u_j1_j3"]),
        "com_y_delta": float(rows[-1]["com_y"] - rows[0]["com_y"]),
        "sf_pg_cos_mean": float(np.nanmean([r["sf_pg_cos_interface"] for r in rows])),
        "avg_dt": float(sum(r["dt"] for r in rows) / len(rows)),
        "per_step_csv": csv_path,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="A/B test phase clipping and contact-angle gradient location")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.config, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("restart", {})["restart_from"] = args.checkpoint

    scenarios = [
        ("baseline", False, False),
        ("clip_after_bc", True, False),
        ("simple_grad_j0", False, True),
        ("clip_after_bc_and_grad_j0", True, True),
    ]

    results = []
    for name, clip_after, grad_j0 in scenarios:
        print(f"\n=== Running {name} ===")
        out = _run(
            cfg=json.loads(json.dumps(cfg)),
            output_dir=os.path.join(args.output_dir, name),
            steps=int(args.steps),
            clip_after_bc=clip_after,
            simple_grad_j0=grad_j0,
        )
        out["scenario"] = name
        out["clip_after_bc"] = clip_after
        out["simple_grad_j0"] = grad_j0
        results.append(out)

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "steps": int(args.steps),
        "results": results,
    }
    out_json = os.path.join(args.output_dir, "ab_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    for r in results:
        if r.get("steps_done", 0) <= 0:
            print(f"{r['scenario']}: no data")
            continue
        print(
            f"{r['scenario']:28s} "
            f"row1_peak={r['row1_div_peak']:.3e} "
            f"strip_u_end={r['strip_u_end']:.3e} "
            f"cos_sf_pg={r['sf_pg_cos_mean']:.3f} "
            f"com_dy={r['com_y_delta']:.3e}"
        )
    print("Saved:", out_json)


if __name__ == "__main__":
    main()

