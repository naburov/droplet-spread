"""Runtime chainsaw / contact-line diagnostics (ghost row, div strips, contact mask)."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from solvers.ppe_utils import check_divergence


def _alternating_mode_1d(arr: np.ndarray) -> float:
    arr = np.asarray(arr).ravel()
    if arr.size == 0:
        return 0.0
    signs = (-1.0) ** np.arange(arr.size, dtype=np.float64)
    return float(np.mean(arr * signs))


def _contact_mask_stats(phi: np.ndarray, dx: float, dy: float, bc) -> dict[str, float]:
    """Mirror ghost-cell contact_weight logic in NumPy (for diagnostics only)."""
    from numerics.finite_differences import jax_gradient

    phi_j = jnp.asarray(phi)
    f_1 = jnp.zeros((phi.shape[0], phi.shape[1]))
    grad = np.asarray(jax_gradient(phi_j, dx, dy, f_1))
    phi_x = grad[:, 0, 0]
    phi_y = grad[:, 0, 1]
    norm_grad = np.sqrt(phi_x**2 + phi_y**2)
    norm_grad = np.maximum(norm_grad, 1e-10)

    phi_bottom = phi[:, 0]
    phi_above = phi[:, 1]
    phi_crosses = (phi_bottom * phi_above) < 0.0
    phi_near = (np.abs(phi_bottom) < 0.5) | (np.abs(phi_above) < 0.5)
    hard = (phi_crosses | phi_near) & (norm_grad > 1e-3)

    band = max(float(bc.contact_mask_soft_band), 1e-12)
    gscale = max(float(bc.contact_mask_grad_scale), 1e-12)
    strength = np.minimum(np.abs(phi_bottom), np.abs(phi_above))
    w_phi = np.clip((band - strength) / band, 0.0, 1.0)
    w_grad = np.clip((norm_grad - 1e-3) / gscale, 0.0, 1.0)
    soft = w_phi * w_grad
    use_soft = bc.contact_mask_soft_band > 0.0
    weight = soft if use_soft else hard.astype(np.float64)
    if bc.contact_angle_full_wall or bc.contact_angle_ghost_law_code == 1:
        weight = np.ones_like(weight)

    return {
        "contact_cells": float(np.sum(weight > 0.05)),
        "contact_weight_min": float(weight.min()),
        "contact_weight_max": float(weight.max()),
        "contact_weight_mean": float(weight.mean()),
        "contact_weight_alt": _alternating_mode_1d(weight),
    }


def _divergence_strips(U: np.ndarray, dx: float, dy: float, geometry) -> dict[str, float]:
    div, max_div, mean_div, max_div_interior = check_divergence(U, dx, dy, geometry)
    div_np = np.asarray(div)
    out = {
        "div_max": float(max_div),
        "div_mean": float(mean_div),
        "div_max_interior": float(max_div_interior),
    }
    ny = div_np.shape[1]
    out["div_max_bottom_strip"] = float(np.max(np.abs(div_np[:, : min(4, ny)])))
    out["div_max_top_strip"] = float(np.max(np.abs(div_np[:, max(0, ny - 4) :])))
    out["div_max_left_strip"] = float(np.max(np.abs(div_np[:4, :])))
    out["div_max_right_strip"] = float(np.max(np.abs(div_np[-4:, :])))
    return out


def _striping_score(phi: np.ndarray) -> float:
    mask = np.abs(phi) < 0.9
    if not mask.any():
        return 0.0
    d2 = phi[2:, :] - 2.0 * phi[1:-1, :] + phi[:-2, :]
    m = mask[1:-1, :]
    values = np.abs(d2[m])
    return float(np.percentile(values, 95)) if values.size else 0.0


def collect_chainsaw_diagnostics(simulation) -> dict[str, Any]:
    """Collect diagnostics from a running TwoPhaseSimulation instance."""
    state = simulation.state
    phi = np.asarray(state.phi)
    U = np.asarray(state.U)
    dx, dy = float(state.dx), float(state.dy)
    geometry = state.geometry
    step = int(state.step)
    t = float(state.t)

    row: dict[str, Any] = {"step": step, "t": t, "strip95": _striping_score(phi)}

    # Ghost row (prefer in-step capture from last CH solve; else rebuild at checkpoint)
    phase_solver = state.phase_solver
    instep = getattr(phase_solver, "_last_ghost_row_instep", None)
    pf_bc = phase_solver.bc_manager
    ca_bc = pf_bc.contact_angle_bc
    if instep is not None:
        row.update(instep)
        row["ghost_from_instep"] = True
        row["phi_wall_alt"] = instep.get("phi0_alt", _alternating_mode_1d(phi[:, 0]))
        row["phi_above_alt"] = instep.get("phi1_alt", _alternating_mode_1d(phi[:, 1]))
    elif getattr(ca_bc, "method", "") == "ghost_cell" or str(
        simulation.config.get("solver_params", {}).get("phase_field_solver", "")
    ).lower() == "ghost_cell":
        bottom_vel = (
            simulation.config.get("boundary_conditions", {})
            .get("velocity", {})
            .get("bottom", "no_slip")
        )
        ghost = np.asarray(
            ca_bc.build_bottom_ghost_row_jax(
                jnp.asarray(phi),
                dx,
                dy,
                geometry,
                psi=jnp.asarray(state.psi) if state.psi is not None else None,
                U=jnp.asarray(U),
                bottom_velocity_bc=bottom_vel,
            )
        )
        ghost_delta = ghost - phi[:, 1]
        row.update(
            {
                "ghost_delta_max": float(np.max(np.abs(ghost_delta))),
                "ghost_delta_mean": float(np.mean(np.abs(ghost_delta))),
                "ghost_delta_alt": _alternating_mode_1d(ghost_delta),
                "phi0_alt": _alternating_mode_1d(phi[:, 0]),
                "phi1_alt": _alternating_mode_1d(phi[:, 1]),
                "phi_wall_alt": _alternating_mode_1d(phi[:, 0]),
                "phi_above_alt": _alternating_mode_1d(phi[:, 1]),
                "ghost_from_instep": False,
            }
        )
    if getattr(ca_bc, "method", "") == "ghost_cell" or instep is not None:
        row.update(_contact_mask_stats(phi, dx, dy, ca_bc))

    # Surface tension bottom row
    st = np.asarray(state.compute_surface_tension())
    row["st_max"] = float(np.max(st))
    row["st_wall_alt"] = _alternating_mode_1d(st[:, 0, 0])
    row["st_above_alt"] = _alternating_mode_1d(st[:, 0, 1] if st.shape[1] > 1 else st[:, 0, 0])

    # Curvature at wall
    from physics.surface_tension import jax_curvature

    kappa = np.asarray(jax_curvature(jnp.asarray(phi), dx, dy, geometry.f_1_grid))
    row["kappa_wall_max"] = float(np.max(np.abs(kappa[:, 0])))
    row["kappa_wall_alt"] = _alternating_mode_1d(kappa[:, 0])

    # PPE info from last step if available
    ppe = getattr(simulation, "_last_ppe_info", None) or {}
    row["ppe_div_after_max"] = float(ppe.get("div_after_max", np.nan))
    row["ppe_div_after_mean"] = float(ppe.get("div_after_mean", np.nan))

    row.update(_divergence_strips(U, dx, dy, geometry))

    cl_mask = (np.abs(phi[:, 0]) < 0.5) | (np.abs(phi[:, 1]) < 0.5)
    if np.any(cl_mask):
        div_np = np.asarray(check_divergence(U, dx, dy, geometry)[0])
        row["div_max_contact_neighborhood"] = float(
            np.max(np.abs(div_np[cl_mask, : min(4, phi.shape[1])]))
        )
    else:
        row["div_max_contact_neighborhood"] = 0.0

    return row


def append_chainsaw_diagnostics_csv(simulation, output_dir: str) -> None:
    row = collect_chainsaw_diagnostics(simulation)
    path = Path(output_dir) / "chainsaw_diagnostics.csv"
    write_header = not path.is_file()
    fieldnames = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
