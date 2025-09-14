"""
Projection methods for incompressible flow.

This module contains implementations of projection methods
for enforcing incompressibility in fluid simulations.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from numerics.finite_differences import (
    divergence, jax_divergence,
    gradient, jax_gradient
)
from numerics.poisson_solvers import solve_poisson_pyamg


def correction_step(U, dx, dy, dt, correction_solver=None, div=None):
    """PPE method for incompressible flow.
    
    Args:
        U (np.ndarray): Velocity field.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        dt (float): Time step.
        correction_solver: Solver for pressure correction.
        div (np.ndarray, optional): Precomputed divergence.
    
    Returns:
        tuple: (corrected_velocity, pressure_correction)
    """
    if div is None:
        div = divergence(U, dx, dy) / dt
    
    div = div - np.mean(div)

    if correction_solver is not None:
        correction_solver.set_rhs(div)
        correction_solver.solve()
        p_correction = correction_solver.get_solution()
    else:
        p_correction = solve_poisson_pyamg(div, dx, dy)

    # Apply pressure correction (using JAX derivatives exactly like original)
    from numerics.finite_differences import jax_dx, jax_dy
    if hasattr(U, 'at'):  # JAX array
        U = U.at[..., 0].set(U[..., 0] - dt * jax_dx(p_correction, h=dx))
        U = U.at[..., 1].set(U[..., 1] - dt * jax_dy(p_correction, h=dy))
    else:  # NumPy array - convert to JAX for derivative calculation
        import jax.numpy as jnp
        p_correction_jax = jnp.array(p_correction)
        U_jax = jnp.array(U)
        U_jax = U_jax.at[..., 0].set(U_jax[..., 0] - dt * jax_dx(p_correction_jax, h=dx))
        U_jax = U_jax.at[..., 1].set(U_jax[..., 1] - dt * jax_dy(p_correction_jax, h=dy))
        U = np.array(U_jax)
    
    return U, p_correction


@jit
def jax_correction_step(U, dx, dy, dt, div=None):
    """JAX-compiled version of correction step."""
    if div is None:
        div = jax_divergence(U, dx, dy) / dt
    
    div = div - jnp.mean(div)
    
    # For JAX version, we would need a JAX-compatible solver
    # This is a placeholder - would need proper implementation
    p_correction = jnp.zeros_like(div)
    
    # Apply pressure correction
    grad_p = jax_gradient(p_correction, dx, dy)
    U = U.at[..., 0].set(U[..., 0] - dt * grad_p[..., 0])
    U = U.at[..., 1].set(U[..., 1] - dt * grad_p[..., 1])
    
    return U, p_correction


def damp_divergence(U, dx, dy, xi, dt):
    """Damp divergence using artificial viscosity.
    
    Args:
        U (np.ndarray): Velocity field.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        xi (float): Damping coefficient.
        dt (float): Time step.
    
    Returns:
        np.ndarray: Damped velocity field.
    """
    div = divergence(U, dx, dy)
    div_grad = gradient(div, dx, dy)
    U[..., 0] -= xi * dt * div_grad[..., 0]
    U[..., 1] -= xi * dt * div_grad[..., 1]
    return U


@jit
def jax_damp_divergence(U, dx, dy, xi, dt):
    """JAX-compiled version of divergence damping."""
    div = jax_divergence(U, dx, dy)
    div_grad = jax_gradient(div, dx, dy)
    U = U.at[..., 0].set(U[..., 0] - xi * dt * div_grad[..., 0])
    U = U.at[..., 1].set(U[..., 1] - xi * dt * div_grad[..., 1])
    return U


def ppe(U, dx, dy, dt, correction_solver=None, div_threshold=0.05):
    """Pressure projection method for incompressible flow.
    
    This function exactly matches the original implementation.
    
    Args:
        U (np.ndarray): Velocity field.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        dt (float): Time step.
        correction_solver: Solver for pressure correction.
        div_threshold (float): Divergence threshold for convergence.
    
    Returns:
        np.ndarray: Projected velocity field.
    """
    max_div_threshold = 0.05
    
    U, solution = correction_step(U, dx, dy, dt, correction_solver=correction_solver)
    
    # Apply velocity boundary conditions after correction
    from physics.fluid_dynamics import apply_velocity_boundary_conditions
    U = apply_velocity_boundary_conditions(U, 0.01, dy)
    
    # Local corrections (exactly like original)
    from physics.fluid_dynamics import check_continuity
    divergence, max_div, mean_div = check_continuity(U, dx, dy)
    
    if mean_div > div_threshold:
        half = int(U.shape[1] / 2)
        to_replace = int(U.shape[1] * 0.4)
        count = 0
        while mean_div > div_threshold:
            U, solution = correction_step(U, dx, dy, dt, correction_solver=correction_solver, div=divergence/dt)
            U = apply_velocity_boundary_conditions(U, 0.01, dy)
            divergence, max_div, mean_div = check_continuity(U, dx, dy)
            if count % 20 == 0:
                import sys
                sys.stdout.write(f"\rMax|mean div: {max_div:.6f}  | {mean_div:.6f}")
            if max_div < max_div_threshold:
                break
            count += 1
        import sys
        sys.stdout.write(f"\nCorrected in {count} iterations \n")
    
    return U


@jit
def jax_ppe(U, dx, dy, dt, div_threshold=0.05):
    """JAX-compiled version of PPE method."""
    # This would need proper JAX implementation
    # For now, just return the input
    return U
