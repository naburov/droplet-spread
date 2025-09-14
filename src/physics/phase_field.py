"""
Phase field equations and boundary conditions.

This module contains the phase field evolution equations and
boundary condition implementations for the droplet spreading simulation.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from numerics.finite_differences import (
    gradient, jax_gradient, 
    laplacian, jax_laplacian
)
from physics.properties import df_2, jax_df_2


def update_phase(phi, U, current_dt, dx, dy, Pe, epsilon, contact_angle):
    """Update the phase field using the explicit Euler method with interface thickness control.
    
    Args:
        phi (np.ndarray): Current phase field (shape: (Nx, Ny)).
        U (np.ndarray): Velocity field (shape: (Nx, Ny, 2)).
        current_dt (float): Current time step.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        Pe (float): Peclet number.
        epsilon (float): Interface thickness parameter.
        contact_angle (float): Contact angle in degrees.
    
    Returns:
        np.ndarray: Updated phase field.
    """
    # Step 1: Calculate the gradient of phi
    grad_phi = gradient(phi, dx, dy)  # Shape: (Nx, Ny, 2)

    # Step 2: Calculate the convective term
    convective_term = np.zeros_like(phi)  # Shape: (Nx, Ny)
    convective_term += U[..., 0] * grad_phi[..., 0] + U[..., 1] * grad_phi[..., 1]

    # Step 3: Calculate the Laplacian of phi
    lap_phi = laplacian(phi, dx, dy)

    # Step 4: Calculate stabilized chemical potential with interface thickness control
    chemical_potential = df_2(phi) - epsilon**2 * lap_phi
    lagrange_multiplier = np.mean(chemical_potential)
    source_term = -1/Pe * (chemical_potential - lagrange_multiplier)

    # Step 5: Right-hand side of the phase equation
    rhs_phi = -convective_term + source_term

    # Step 6: Update phase field
    phi = phi + current_dt * rhs_phi

    # Step 7: Apply boundary conditions and maintain phase field bounds
    phi = apply_contact_angle_boundary_conditions(phi, dx, dy, contact_angle=contact_angle)
    
    # Ensure phi stays within physical bounds [-1, 1]
    phi = np.clip(phi, -1.0, 1.0)

    return phi


@jit
def jax_update_phase(phi, U, current_dt, dx, dy, Pe, epsilon, contact_angle):
    """JAX-compiled version of phase field update.
    
    Args:
        phi (jnp.ndarray): Current phase field (shape: (Nx, Ny)).
        U (jnp.ndarray): Velocity field (shape: (Nx, Ny, 2)).
        current_dt (float): Current time step.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        Pe (float): Peclet number.
        epsilon (float): Interface thickness parameter.
        contact_angle (float): Contact angle in degrees.
    
    Returns:
        jnp.ndarray: Updated phase field.
    """
    # Step 1: Calculate the gradient of phi
    grad_phi = jax_gradient(phi, dx, dy)  # Shape: (Nx, Ny, 2)

    # Step 2: Calculate the convective term
    convective_term = U[..., 0] * grad_phi[..., 0] + U[..., 1] * grad_phi[..., 1]

    # Step 3: Calculate the Laplacian of phi
    lap_phi = jax_laplacian(phi, dx, dy)

    # Step 4: Calculate stabilized chemical potential with interface thickness control
    chemical_potential = jax_df_2(phi) - epsilon**2 * lap_phi
    lagrange_multiplier = jnp.mean(chemical_potential)
    source_term = -1/Pe * (chemical_potential - lagrange_multiplier)

    # Step 5: Right-hand side of the phase equation
    rhs_phi = -convective_term + source_term

    # Step 6: Update phase field
    phi = phi + current_dt * rhs_phi

    # Step 7: Apply boundary conditions and maintain phase field bounds
    phi = jax_apply_contact_angle_boundary_conditions(phi, dx, dy, contact_angle=contact_angle)
    
    # Ensure phi stays within physical bounds [-1, 1]
    phi = jnp.clip(phi, -1.0, 1.0)

    return phi


def apply_contact_angle_boundary_conditions(phi, dx, dy, contact_angle=90):
    """Apply contact angle boundary conditions to the phase field.
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        contact_angle (float): Contact angle in degrees (default: 90).
    
    Returns:
        np.ndarray: Phase field with contact angle boundary conditions applied.
    """
    # Create a copy of the phase field to avoid modifying the original
    phi_new = phi.copy()
    
    # Convert contact angle to radians
    theta = (180 - contact_angle) * np.pi / 180
    
    # Calculate the gradient of phi at the bottom boundary
    grad_phi_x = gradient(phi, dx, dy)[:, 1, 0]  # x-component of gradient at y=1
    grad_phi_y = gradient(phi, dx, dy)[:, 1, 1]  # y-component of gradient at y=1
    
    # Calculate the norm of the gradient
    norm_grad_phi = np.sqrt(grad_phi_x**2 + grad_phi_y**2)
    
    # Avoid division by zero
    norm_grad_phi[norm_grad_phi < 1e-10] = 1e-10
    
    # Calculate the normal derivative based on the contact angle
    normal_derivative = -np.cos(theta) * norm_grad_phi
    
    # Apply the boundary condition at the bottom (y=0)
    phi_new[:, 0] = phi_new[:, 1] - normal_derivative * dy
    
    # Apply Neumann boundary conditions (zero gradient) at other boundaries
    phi_new[:, -1] = phi_new[:, -2]  # Top boundary
    phi_new[0, :] = phi_new[1, :]    # Left boundary
    phi_new[-1, :] = phi_new[-2, :]  # Right boundary
    
    return phi_new


@jit
def jax_apply_contact_angle_boundary_conditions(phi, dx, dy, contact_angle=90):
    """JAX-compiled version of contact angle boundary conditions.
    
    Args:
        phi (jnp.ndarray): Phase field (shape: (Nx, Ny)).
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        contact_angle (float): Contact angle in degrees (default: 90).
    
    Returns:
        jnp.ndarray: Phase field with contact angle boundary conditions applied.
    """
    # Convert contact angle to radians
    theta = (180 - contact_angle) * jnp.pi / 180
    
    # Calculate the gradient of phi at the bottom boundary
    grad_phi = jax_gradient(phi, dx, dy)
    grad_phi_x = grad_phi[:, 1, 0]  # x-component of gradient at y=1
    grad_phi_y = grad_phi[:, 1, 1]  # y-component of gradient at y=1
    
    # Calculate the norm of the gradient
    norm_grad_phi = jnp.sqrt(grad_phi_x**2 + grad_phi_y**2)
    
    # Avoid division by zero
    norm_grad_phi = jnp.where(norm_grad_phi < 1e-10, 1e-10, norm_grad_phi)
    
    # Calculate the normal derivative based on the contact angle
    normal_derivative = -jnp.cos(theta) * norm_grad_phi
    
    # Apply the boundary condition at the bottom (y=0)
    phi_new = phi.at[:, 0].set(phi[:, 1] - normal_derivative * dy)
    
    # Apply Neumann boundary conditions (zero gradient) at other boundaries
    phi_new = phi_new.at[:, -1].set(phi_new[:, -2])  # Top boundary
    phi_new = phi_new.at[0, :].set(phi_new[1, :])    # Left boundary
    phi_new = phi_new.at[-1, :].set(phi_new[-2, :])  # Right boundary
    
    return phi_new


def penalization(phi, alpha):
    """Apply penalization force if phi exceeds [-1, 1].
    
    Args:
        phi (np.ndarray): Phase field.
        alpha (float): Penalization strength.
    
    Returns:
        np.ndarray: Penalization force.
    """
    mask_pos = phi > 1.0
    mask_neg = phi < -1.0
    
    penalty = np.zeros_like(phi)
    penalty[mask_pos] = (phi[mask_pos] - 1.0)
    penalty[mask_neg] = (phi[mask_neg] + 1.0)
    
    return alpha * penalty


class PhaseFieldSolver:
    """Phase field solver class for droplet spreading simulation."""
    
    def __init__(self, Pe, epsilon, contact_angle):
        """Initialize the phase field solver.
        
        Args:
            Pe (float): Peclet number.
            epsilon (float): Interface thickness parameter.
            contact_angle (float): Contact angle in degrees.
        """
        self.Pe = Pe
        self.epsilon = epsilon
        self.contact_angle = contact_angle
    
    def update(self, phi, U, current_dt, dx, dy, use_jax=False):
        """Update the phase field.
        
        Args:
            phi: Phase field array.
            U: Velocity field array.
            current_dt: Current time step.
            dx: Grid spacing in x-direction.
            dy: Grid spacing in y-direction.
            use_jax: Whether to use JAX-compiled functions.
        
        Returns:
            Updated phase field.
        """
        if use_jax:
            return jax_update_phase(phi, U, current_dt, dx, dy, 
                                  self.Pe, self.epsilon, self.contact_angle)
        else:
            return update_phase(phi, U, current_dt, dx, dy, 
                              self.Pe, self.epsilon, self.contact_angle)
