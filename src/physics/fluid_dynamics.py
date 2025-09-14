"""
Fluid dynamics equations for droplet spreading simulation.

This module contains the Navier-Stokes equations, continuity equation,
and related fluid dynamics functionality.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from numerics.finite_differences import (
    gradient, jax_gradient,
    divergence, jax_divergence,
    laplacian, jax_laplacian
)
from physics.properties import (
    calculate_density, calculate_reynolds_number,
    jax_calculate_density, jax_calculate_reynolds_number
)


def compute_viscous_term(U, dx, dy, Re):
    """Simplified viscous term for constant viscosity: (1/Re) * ∇²U.
    
    Args:
        U (np.ndarray): Velocity field (shape: (Nx, Ny, 2)).
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        Re (float or np.ndarray): Reynolds number.
    
    Returns:
        np.ndarray: Viscous term.
    """
    viscous_term = np.zeros_like(U)
    viscous_term[..., 0] = laplacian(U[..., 0], dx, dy) / Re
    viscous_term[..., 1] = laplacian(U[..., 1], dx, dy) / Re
    return viscous_term


@jit
def jax_compute_viscous_term(U, dx, dy, Re):
    """JAX-compiled version of viscous term calculation.
    
    Args:
        U (jnp.ndarray): Velocity field (shape: (Nx, Ny, 2)).
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        Re (float or jnp.ndarray): Reynolds number.
    
    Returns:
        jnp.ndarray: Viscous term.
    """
    return jnp.stack([jax_laplacian(U[..., 0], dx, dy) / Re, 
                      jax_laplacian(U[..., 1], dx, dy) / Re], axis=-1)


def check_continuity(U, dx, dy):
    """
    Check continuity equation condition (∇·U = 0).
    Returns the divergence field and maximum absolute divergence.
    
    Args:
        U (np.ndarray or jnp.ndarray): Velocity field (shape: (Nx, Ny, 2)).
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        tuple: (divergence, max_div, mean_div)
    """
    # Use JAX divergence for JAX arrays, NumPy divergence for NumPy arrays
    if hasattr(U, 'at'):  # JAX array
        from numerics.finite_differences import jax_divergence
        import jax.numpy as jnp
        divergence_field = jax_divergence(U, dx, dy)
        max_div = jnp.max(jnp.abs(divergence_field))
        mean_div = jnp.mean(jnp.abs(divergence_field))
    else:  # NumPy array
        divergence_field = divergence(U, dx, dy)
        max_div = np.max(np.abs(divergence_field))
        mean_div = np.mean(np.abs(divergence_field))
    
    return divergence_field, max_div, mean_div


@jit
def jax_check_continuity(U, dx, dy):
    """
    JAX-compiled version of continuity check.
    
    Args:
        U (jnp.ndarray): Velocity field (shape: (Nx, Ny, 2)).
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        tuple: (divergence, max_div, mean_div)
    """
    # Calculate divergence: du/dx + dv/dy
    u_x = jax_divergence(U, dx, dy)[..., 0]
    v_y = jax_divergence(U, dx, dy)[..., 1]
    
    divergence_field = u_x + v_y
    max_div = jnp.max(jnp.abs(divergence_field))
    mean_div = jnp.mean(jnp.abs(divergence_field))
    
    return divergence_field, max_div, mean_div


def update_velocity(U, p, surface_tension, current_dt, dx, dy, rho1, rho2, 
                   Re1, Re2, Fr, g, phi, include_gravity=False):
    """Update the velocity field U based on the phase field phi.
    
    Args:
        U (np.ndarray): Velocity field (shape: (Nx, Ny, 2)).
        p (np.ndarray): Pressure field (shape: (Nx, Ny)).
        surface_tension (np.ndarray): Surface tension force (shape: (Nx, Ny, 2)).
        current_dt (float): Current time step.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        rho1 (float): Density of phase 1.
        rho2 (float): Density of phase 2.
        Re1 (float): Reynolds number of phase 1.
        Re2 (float): Reynolds number of phase 2.
        Fr (float): Froude number.
        g (float): Gravitational acceleration.
        phi (np.ndarray): Phase field.
        include_gravity (bool): Whether to include gravity.
    
    Returns:
        np.ndarray: Updated velocity field.
    """
    # Calculate the Reynolds number and density
    Re = calculate_reynolds_number(phi, Re1, Re2)
    rho = calculate_density(phi, rho1, rho2)
    rho_stacked = np.stack([rho, rho], axis=-1) + 1e-6

    # Calculate gradients and terms
    grad_U = gradient(U, dx, dy)
    p_grad = gradient(p, dx, dy)
    
    # Calculate viscous term with proper scaling
    viscous_term = compute_viscous_term(U, dx, dy, Re)
    
    # Calculate convective term (in conservative form)
    convective_term = np.zeros_like(U)
    convective_term[..., 0] = (U[..., 0] * grad_U[..., 0, 0] + U[..., 1] * grad_U[..., 0, 1])
    convective_term[..., 1] = (U[..., 0] * grad_U[..., 1, 0] + U[..., 1] * grad_U[..., 1, 1])

    # Combine terms with proper density scaling
    rhs_U = (
        -p_grad / rho_stacked +  # Pressure term
        viscous_term / rho_stacked +  # Viscous term
        -surface_tension / rho_stacked +  # Surface tension
        -convective_term  # Convective term (already includes velocity)
    )

    # Add gravity if included
    if include_gravity:
        rhs_U += (1 / Fr) * np.stack([np.zeros_like(U[..., 0]), -np.ones_like(U[..., 1])], axis=-1)

    # Update velocity field using explicit Euler
    U = U + current_dt * rhs_U
    
    return U


@jit
def jax_update_velocity(U, p, surface_tension, current_dt, dx, dy, rho1, rho2, 
                       Re1, Re2, Fr, g, phi, include_gravity=False):
    """JAX-compiled version of velocity update.
    
    Args:
        U (jnp.ndarray): Velocity field (shape: (Nx, Ny, 2)).
        p (jnp.ndarray): Pressure field (shape: (Nx, Ny)).
        surface_tension (jnp.ndarray): Surface tension force (shape: (Nx, Ny, 2)).
        current_dt (float): Current time step.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        rho1 (float): Density of phase 1.
        rho2 (float): Density of phase 2.
        Re1 (float): Reynolds number of phase 1.
        Re2 (float): Reynolds number of phase 2.
        Fr (float): Froude number.
        g (float): Gravitational acceleration.
        phi (jnp.ndarray): Phase field.
        include_gravity (bool): Whether to include gravity.
    
    Returns:
        jnp.ndarray: Updated velocity field.
    """
    # Calculate the Reynolds number and density
    Re = jax_calculate_reynolds_number(phi, Re1, Re2)
    rho = jax_calculate_density(phi, rho1, rho2)
    rho_stacked = jnp.stack([rho, rho], axis=-1) + 1e-6

    # Calculate gradients and terms
    grad_U = jax_gradient(U, dx, dy)
    p_grad = jax_gradient(p, dx, dy)
    
    # Calculate viscous term with proper scaling
    viscous_term = jax_compute_viscous_term(U, dx, dy, Re)
    
    # Calculate convective term (in conservative form)
    convective_term = jnp.stack(
        [
            U[..., 0] * grad_U[..., 0, 0] + U[..., 1] * grad_U[..., 0, 1],
            U[..., 0] * grad_U[..., 1, 0] + U[..., 1] * grad_U[..., 1, 1]
        ],
        axis=-1
    )

    # Combine terms with proper density scaling
    rhs_U = (
        -p_grad / rho_stacked +  # Pressure term
        viscous_term / rho_stacked +  # Viscous term
        -surface_tension / rho_stacked +  # Surface tension
        -convective_term  # Convective term (already includes velocity)
    )

    # Add gravity if included (using JAX conditional)
    gravity_term = (1 / Fr) * jnp.stack([jnp.zeros_like(U[..., 0]), -jnp.ones_like(U[..., 1])], axis=-1)
    rhs_U = jnp.where(include_gravity, rhs_U + gravity_term, rhs_U)

    # Update velocity field using explicit Euler
    U = U + current_dt * rhs_U
    
    return U


def apply_velocity_boundary_conditions(U, beta, dy):
    """Apply physically appropriate boundary conditions to velocity field.
    
    Args:
        U (np.ndarray or jnp.ndarray): Velocity field (shape: (Nx, Ny, 2)).
        beta (float): Slip parameter.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        np.ndarray or jnp.ndarray: Velocity with boundary conditions applied.
    """
    # Check if U is a JAX array
    if hasattr(U, 'at'):  # JAX array
        # Bottom boundary (solid wall): Slip condition
        U = U.at[:, 0, 1].set(0.0)  # y-component
        U = U.at[:, 0, 0].set(U[:, 1, 0] - dy * 1/beta * U[:, 1, 0])  # x-component

        # Top boundary (open atmosphere): Zero-gradient condition
        U = U.at[:, -1, :].set(U[:, -2, :])

        # Left and right boundaries: Zero-gradient condition
        U = U.at[0, :, :].set(U[1, :, :])
        U = U.at[-1, :, :].set(U[-2, :, :])
    else:  # NumPy array
        # Make a copy to avoid modifying the original array
        U_new = U.copy()
        
        # Bottom boundary (solid wall): Slip condition
        U_new[:, 0, 1] = 0.0  # y-component
        U_new[:, 0, 0] = U_new[:, 1, 0] - dy * 1/beta * U_new[:, 1, 0]  # x-component

        # Top boundary (open atmosphere): Zero-gradient condition
        U_new[:, -1, :] = U_new[:, -2, :]

        # Left and right boundaries: Zero-gradient condition
        U_new[0, :, :] = U_new[1, :, :]
        U_new[-1, :, :] = U_new[-2, :, :]
        
        U = U_new

    return U


@jit
def jax_apply_velocity_boundary_conditions(U, beta, dy):
    """JAX-compiled version of velocity boundary conditions.
    
    Args:
        U (jnp.ndarray): Velocity field (shape: (Nx, Ny, 2)).
        beta (float): Slip parameter.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        jnp.ndarray: Velocity with boundary conditions applied.
    """
    # Bottom boundary (solid wall): Slip condition
    U = U.at[:, 0, 1].set(0.0)
    U = U.at[:, 0, 0].set(U[:, 1, 0] - dy * 1/beta * U[:, 1, 0])

    # Top boundary (open atmosphere): Zero-gradient condition
    U = U.at[:, -1, :].set(U[:, -2, :])
    
    # Left and right boundaries: Zero-gradient condition
    U = U.at[0, :, :].set(U[1, :, :])
    U = U.at[-1, :, :].set(U[-2, :, :])
    
    return U


class FluidDynamicsSolver:
    """Fluid dynamics solver class for droplet spreading simulation."""
    
    def __init__(self, rho1, rho2, Re1, Re2, Fr, g):
        """Initialize the fluid dynamics solver.
        
        Args:
            rho1 (float): Density of phase 1.
            rho2 (float): Density of phase 2.
            Re1 (float): Reynolds number of phase 1.
            Re2 (float): Reynolds number of phase 2.
            Fr (float): Froude number.
            g (float): Gravitational acceleration.
        """
        self.rho1 = rho1
        self.rho2 = rho2
        self.Re1 = Re1
        self.Re2 = Re2
        self.Fr = Fr
        self.g = g
    
    def update_velocity(self, U, p, surface_tension, current_dt, dx, dy, 
                       phi, include_gravity=False, use_jax=False):
        """Update the velocity field.
        
        Args:
            U: Velocity field array.
            p: Pressure field array.
            surface_tension: Surface tension force array.
            current_dt: Current time step.
            dx: Grid spacing in x-direction.
            dy: Grid spacing in y-direction.
            phi: Phase field array.
            include_gravity: Whether to include gravity.
            use_jax: Whether to use JAX-compiled functions.
        
        Returns:
            Updated velocity field.
        """
        if use_jax:
            return jax_update_velocity(U, p, surface_tension, current_dt, dx, dy,
                                     self.rho1, self.rho2, self.Re1, self.Re2,
                                     self.Fr, self.g, phi, include_gravity)
        else:
            return update_velocity(U, p, surface_tension, current_dt, dx, dy,
                                 self.rho1, self.rho2, self.Re1, self.Re2,
                                 self.Fr, self.g, phi, include_gravity)
    
    def check_continuity(self, U, dx, dy, use_jax=False):
        """Check continuity equation.
        
        Args:
            U: Velocity field array.
            dx: Grid spacing in x-direction.
            dy: Grid spacing in y-direction.
            use_jax: Whether to use JAX-compiled functions.
        
        Returns:
            Tuple of (divergence, max_div, mean_div).
        """
        if use_jax:
            return jax_check_continuity(U, dx, dy)
        else:
            return check_continuity(U, dx, dy)
