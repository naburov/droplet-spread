#!/usr/bin/env python3
"""
One-step A/B for contact-line phase update hypotheses (no PPE):
  1) clip timing: before BC (baseline) vs after BC
  2) simple contact-angle gradient location: j=1 (baseline) vs j=0
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulation.state import SimulationState
from numerics.finite_differences import jax_gradient, jax_laplacian
from physics.properties import jax_df_2
from physics.surface_tension import jax_curvature
import physics.phase_field as phase_field_mod
import boundary_conditions.contact_angle_bc as cab_mod


def _contact_pair(phi: np.ndarray) -> Optional[Tuple[int, int]]:
    mask = ((phi[:, 0] * phi[:, 1]) < 0.0) | (np.abs(phi[:, 0]) < 0.5) | (np.abs(phi[:, 1]) < 0.5)
    idx = np.where(mask)[0]
    if idx.size < 2:
        return None
    return int(idx.min()), int(idx.max())


def _liquid_mass_proxy(phi: np.ndarray, dx: float, dy: float) -> float:
    c_liq = np.clip((1.0 - phi) * 0.5, 0.0, 1.0)
    return float(np.sum(c_liq) * dx * dy)


def _jax_update_phase_no_clip(phi, U, current_dt, dx, dy, Pe, epsilon, contact_angle, f_1_grid, f_2_grid, lambda_willmore=0.0, epsilon_willmore=0.0):
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
    rhs_phi = -convective_term - (1.0 / Pe) * (mu_total - lagrange_multiplier)
    return phi + current_dt * rhs_phi


@contextmanager
def _patch(clip_after_bc: bool, simple_grad_j0: bool):
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


def _evaluate(state: SimulationState, phi0: np.ndarray, phi1: np.ndarray) -> Dict:
    dx, dy = float(state.dx), float(state.dy)
    grad1 = np.array(jax_gradient(jnp.array(phi1), dx, dy, state.geometry.f_1_grid))
    kappa1 = np.array(jax_curvature(jnp.array(phi1), dx, dy, state.geometry.f_1_grid))
    sf1 = np.array(
        state.surface_tension_solver.apply_boundary_conditions(
            state.surface_tension_solver.calculate_force(
                jnp.array(phi1), dx, dy, state.geometry, use_jax=True, interface_mask=None
            ),
            jnp.array(phi1),
            use_jax=True,
            geometry=state.geometry,
            dx=dx,
            dy=dy,
        )
    )

    cp = _contact_pair(phi1)
    contact_rows = {}
    if cp is not None:
        for tag, i in (("left", cp[0]), ("right", cp[1])):
            j = int(np.argmin(np.abs(phi1[i, :6])))
            gx = float(grad1[i, j, 0])
            gy = float(grad1[i, j, 1])
            gmag = float(np.hypot(gx, gy))
            nx = gx / max(gmag, 1e-12)
            ny = gy / max(gmag, 1e-12)
            contact_rows[tag] = {
                "i": int(i),
                "j": int(j),
                "phi": float(phi1[i, j]),
                "grad_mag": gmag,
                "n_x": float(nx),
                "n_y": float(ny),
                "kappa": float(kappa1[i, j]),
                "sf_mag": float(np.linalg.norm(sf1[i, j])),
            }

    bottom_band = slice(0, 4)
    return {
        "mass_proxy_before": _liquid_mass_proxy(phi0, dx, dy),
        "mass_proxy_after": _liquid_mass_proxy(phi1, dx, dy),
        "mass_proxy_delta": _liquid_mass_proxy(phi1, dx, dy) - _liquid_mass_proxy(phi0, dx, dy),
        "phi_l2_change": float(np.sqrt(np.mean((phi1 - phi0) ** 2))),
        "phi_bottom_tv": float(np.mean(np.abs(np.diff(phi1[:, 0])))),
        "kappa_max_bottom4": float(np.max(np.abs(kappa1[:, bottom_band]))),
        "kappa_mean_bottom4": float(np.mean(np.abs(kappa1[:, bottom_band]))),
        "sf_max_bottom4": float(np.max(np.linalg.norm(sf1[:, bottom_band, :], axis=-1))),
        "contact_points": contact_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="One-step phase/contact-line A/B")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.config, "r") as f:
        cfg = json.load(f)
    state = SimulationState.from_config(cfg, restart_from=args.checkpoint)

    phi0 = np.array(state.phi)
    U0 = np.array(state.U)
    dt = float(state.dt)

    scenarios = [
        ("baseline", False, False),
        ("clip_after_bc", True, False),
        ("simple_grad_j0", False, True),
        ("clip_after_bc_and_grad_j0", True, True),
    ]
    results = []
    phi_store = {}

    for name, clip_after, grad_j0 in scenarios:
        with _patch(clip_after_bc=clip_after, simple_grad_j0=grad_j0):
            phi1 = np.array(
                state.phase_solver.update(
                    jnp.array(phi0),
                    jnp.array(U0),
                    dt,
                    float(state.dx),
                    float(state.dy),
                    state.geometry,
                    use_jax=True,
                    psi=state.psi,
                )
            )
        phi_store[name] = phi1
        evals = _evaluate(state, phi0, phi1)
        evals["scenario"] = name
        evals["clip_after_bc"] = clip_after
        evals["simple_grad_j0"] = grad_j0
        results.append(evals)

    baseline = phi_store["baseline"]
    for r in results:
        name = r["scenario"]
        r["phi_diff_vs_baseline_l2"] = float(np.sqrt(np.mean((phi_store[name] - baseline) ** 2)))
        r["phi_diff_vs_baseline_maxabs"] = float(np.max(np.abs(phi_store[name] - baseline)))

    out = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "dt_used": dt,
        "results": results,
    }
    out_path = os.path.join(args.output_dir, "one_step_ab_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("Saved:", out_path)
    print("\nOne-step comparison:")
    for r in results:
        print(
            f"{r['scenario']:28s} "
            f"dphi_l2={r['phi_l2_change']:.3e} "
            f"kappa_bottom4_max={r['kappa_max_bottom4']:.3e} "
            f"sf_bottom4_max={r['sf_max_bottom4']:.3e} "
            f"vs_base_maxabs={r['phi_diff_vs_baseline_maxabs']:.3e}"
        )


if __name__ == "__main__":
    main()

