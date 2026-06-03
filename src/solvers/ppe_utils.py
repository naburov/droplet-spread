"""
Common utilities for PPE (Pressure Projection Equation) methods.

Simplified and split into clear steps.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from physics.fluid_dynamics import jax_check_continuity
from numerics.finite_differences import jax_dx, jax_dy, jax_divergence, jax_gradient
from numerics.poisson_solvers import solve_poisson_pyamg


def check_divergence(U, dx, dy, geometry):
    """Check divergence (terrain). Curvilinear (x, eta): fluid-only grid.
    Returns (divergence, max_div, mean_div, max_div_interior).
    """
    divergence, max_div, mean_div = jax_check_continuity(U, dx, dy, geometry.f_1_grid)
    div_np = np.array(divergence)
    interior = div_np[1:-1, 1:-1]
    max_div_interior = float(np.max(np.abs(interior))) if interior.size > 0 else float(max_div)
    return divergence, max_div, mean_div, max_div_interior


def to_numpy(array):
    """Convert JAX array to NumPy array."""
    return np.array(array)


def to_jax(array):
    """Convert NumPy array to JAX array."""
    return jnp.array(array)


def compute_divergence_for_solve(U, dx, dy, dt, divergence, psi, geometry, ppe_bcs):
    """Step 1: Compute and prepare divergence for pressure correction solve. Grid is fluid-only."""
    div = np.array(divergence) / dt
    # Mean-zero RHS only when all BCs are Neumann (otherwise pressure is unique).
    if ppe_bcs is not None and all(bc == 'neumann' for bc in ppe_bcs.values()):
        div = div - np.mean(div)

    # Dirichlet pressure (e.g. outlet p' = 0): RHS at that boundary is the prescribed value.
    if ppe_bcs is not None:
        Nx, Ny = div.shape
        if ppe_bcs.get("left") == "dirichlet":
            div[0, :] = 0.0
        if ppe_bcs.get("right") == "dirichlet":
            div[Nx - 1, :] = 0.0
        if ppe_bcs.get("bottom") == "dirichlet":
            div[:, 0] = 0.0
        if ppe_bcs.get("top") == "dirichlet":
            div[:, Ny - 1] = 0.0

    return div


def solve_pressure_correction(div, correction_solver, geometry, dy, psi,
                              U_star, velocity_bc_manager, dx, dt):
    """Solve for pressure correction. Grid is fluid-only (bottom-aligned)."""
    if correction_solver is None:
        ppe_bcs = {"top": "neumann", "bottom": "neumann", "left": "neumann", "right": "neumann"}
        return solve_poisson_pyamg(div, dx, dy, ppe_bcs=ppe_bcs)

    # Inlet compatibility: Neumann pressure BC consistent with Dirichlet velocity
    compatibility_term = None
    if U_star is not None and velocity_bc_manager is not None:
        vel_bc_cfg = velocity_bc_manager.config.get("boundary_conditions", {}).get("velocity", {})
        if vel_bc_cfg.get("left") == "dirichlet":
            Ny = div.shape[1]
            u_in_profile = velocity_bc_manager.get_inlet_profile(Ny, dy)
            if u_in_profile is not None:
                u_star_left = U_star[0, :, 0] if isinstance(U_star, np.ndarray) else np.array(U_star[0, :, 0])
                compatibility_term = (dx / dt) * (u_star_left - u_in_profile)

    correction_solver.set_rhs(div)
    p_correction = correction_solver.solve()

    if compatibility_term is not None:
        p_correction[0, :] = p_correction[1, :] - compatibility_term

    return p_correction


def apply_pressure_correction_to_velocity(U, p_correction, dt, dx, dy, geometry, psi,
                                         under_relaxation=1.0):
    """Step 3: Apply pressure correction to velocity. Grid is fluid-only (bottom-aligned)."""
    grad_p = jax_gradient(p_correction, dx, dy, geometry.f_1_grid)
    dp_dx, dp_dy = grad_p[..., 0], grad_p[..., 1]
    U = U.at[..., 0].set(U[..., 0] - under_relaxation * dt * dp_dx)
    U = U.at[..., 1].set(U[..., 1] - under_relaxation * dt * dp_dy)
    return U


def correction_step(U, dx, dy, dt, geometry, correction_solver=None, div=None, ppe_bcs=None,
                   psi=None, U_star=None, velocity_bc_manager=None):
    """Legacy correction_step. geometry: from state; solid derived from geometry."""
    if div is None:
        div = jax_divergence(U, dx, dy, geometry.f_1_grid) / dt
    div_for_solve = compute_divergence_for_solve(U, dx, dy, dt, div, psi, geometry, ppe_bcs)
    p_correction = solve_pressure_correction(
        div_for_solve, correction_solver, geometry, dy, psi,
        U_star, velocity_bc_manager, dx, dt
    )
    U = apply_pressure_correction_to_velocity(U, p_correction, dt, dx, dy, geometry, psi, under_relaxation=1.0)
    return U, p_correction


def apply_pressure_correction(U, p_correction, dt, dx, dy, geometry, ppe_bcs=None):
    """Legacy: apply correction. geometry: from state; solid derived from geometry."""
    return apply_pressure_correction_to_velocity(U, p_correction, dt, dx, dy, geometry, psi=None)
