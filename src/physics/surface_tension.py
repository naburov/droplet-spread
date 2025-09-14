"""
Surface tension calculations for droplet spreading simulation.

This module contains surface tension force calculations and
curvature computations for the phase field method.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from numerics.finite_differences import (
    gradient, jax_gradient,
    divergence, jax_divergence,
    norm, jax_norm
)
from physics.properties import (
    calculate_weber_number,
    jax_calculate_weber_number
)


def curvature(phi, dx, dy):
    """Calculate the curvature of the phase field phi.
    
    The curvature is defined as:
    K = ∇ · (∇φ / |∇φ|)
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        np.ndarray: Curvature field (shape: (Nx, Ny)).
    """
    # Step 1: Calculate the gradient of phi
    grad_phi = gradient(phi, dx, dy)  # Shape: (Nx, Ny, 2)

    # Step 2: Calculate the norm of the gradient
    norm_grad_phi = norm(grad_phi)  # Shape: (Nx, Ny)

    # Step 3: Avoid division by zero
    norm_grad_phi[norm_grad_phi == 0] = 1e-10

    # Step 4: Calculate the normalized gradient
    normalized_grad_phi = grad_phi / norm_grad_phi[..., np.newaxis]  # Shape: (Nx, Ny, 2)

    # Step 5: Calculate the divergence of the normalized gradient
    curvature_value = divergence(normalized_grad_phi, dx, dy)

    return curvature_value


@jit
def jax_curvature(phi, dx, dy):
    """JAX-compiled version of curvature calculation.
    
    Args:
        phi (jnp.ndarray): Phase field (shape: (Nx, Ny)).
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        jnp.ndarray: Curvature field (shape: (Nx, Ny)).
    """
    # Step 1: Calculate the gradient of phi
    grad_phi = jax_gradient(phi, dx, dy)  # Shape: (Nx, Ny, 2)

    # Step 2: Calculate the norm of the gradient
    norm_grad_phi = jax_norm(grad_phi)  # Shape: (Nx, Ny)

    # Step 3: Avoid division by zero
    norm_grad_phi = jnp.where(norm_grad_phi == 0, 1e-10, norm_grad_phi)

    # Step 4: Calculate the normalized gradient
    normalized_grad_phi = grad_phi / norm_grad_phi[..., jnp.newaxis]  # Shape: (Nx, Ny, 2)

    # Step 5: Calculate the divergence of the normalized gradient
    curvature_value = jax_divergence(normalized_grad_phi, dx, dy)

    return curvature_value


def improved_curvature(phi, dx, dy):
    """Calculate curvature with regularization to reduce numerical artifacts.
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        np.ndarray: Regularized curvature field (shape: (Nx, Ny)).
    """
    grad_phi = gradient(phi, dx, dy)
    grad_phi_magnitude = norm(grad_phi)
    
    # Add regularization to avoid division by zero
    reg = 1e-6
    grad_phi_magnitude = np.maximum(grad_phi_magnitude, reg)
    
    # Normalize gradient
    n_x = grad_phi[..., 0] / grad_phi_magnitude
    n_y = grad_phi[..., 1] / grad_phi_magnitude
    
    # Calculate divergence with higher-order scheme
    from numerics.finite_differences import numerical_derivative
    div_n = numerical_derivative(n_x, axis=0, h=dx) + numerical_derivative(n_y, axis=1, h=dy)
    
    return div_n


@jit
def jax_improved_curvature(phi, dx, dy):
    """JAX-compiled version of improved curvature calculation.
    
    Args:
        phi (jnp.ndarray): Phase field (shape: (Nx, Ny)).
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        jnp.ndarray: Regularized curvature field (shape: (Nx, Ny)).
    """
    grad_phi = jax_gradient(phi, dx, dy)
    grad_phi_magnitude = jax_norm(grad_phi)
    
    # Add regularization to avoid division by zero
    reg = 1e-6
    grad_phi_magnitude = jnp.maximum(grad_phi_magnitude, reg)
    
    # Normalize gradient
    n_x = grad_phi[..., 0] / grad_phi_magnitude
    n_y = grad_phi[..., 1] / grad_phi_magnitude
    
    # Calculate divergence with higher-order scheme
    from numerics.finite_differences import jax_dx, jax_dy
    div_n = jax_dx(n_x, h=dy) + jax_dy(n_y, h=dx)
    
    return div_n


def surface_tension_force(phi, epsilon, We1, We2, dx, dy):
    """Calculate the surface tension force based on the phase field.
    
    The surface tension force is given by:
    F_tension = (3√2 ε)/(4 We) ∇ · (∇φ/|∇φ|) |∇φ| ∇φ
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
        epsilon (float): Interface thickness parameter.
        We1 (float): Weber number for phase 1.
        We2 (float): Weber number for phase 2.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        np.ndarray: Surface tension force (shape: (Nx, Ny, 2)).
    """
    # Step 1: Calculate the curvature using the curvature function
    curvature_value = improved_curvature(phi, dx, dy)  # Shape: (M, N)
    curvature_value = np.stack([curvature_value, curvature_value], axis=-1)

    # Step 2: Calculate the gradient of phi
    grad_phi = gradient(phi, dx, dy)  # Shape: (M, N, 2)

    # Step 3: Calculate the norm of the gradient
    norm_grad_phi = norm(grad_phi)  # Shape: (M, N) 
    norm_grad_phi = np.stack([norm_grad_phi, norm_grad_phi], axis=-1)
    We = calculate_weber_number(phi, We1, We2)
    We = np.stack([We, We], axis=-1)

    # Step 4: Calculate the surface tension force
    tension_force = (3 * np.sqrt(2) * epsilon / (4 * We)) * curvature_value * norm_grad_phi * grad_phi

    return tension_force


@jit
def jax_surface_tension_force(phi, epsilon, We1, We2, dx, dy):
    """JAX-compiled version of surface tension force calculation.
    
    Args:
        phi (jnp.ndarray): Phase field (shape: (Nx, Ny)).
        epsilon (float): Interface thickness parameter.
        We1 (float): Weber number for phase 1.
        We2 (float): Weber number for phase 2.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        jnp.ndarray: Surface tension force (shape: (Nx, Ny, 2)).
    """
    # Step 1: Calculate the curvature using the curvature function
    curvature_value = jax_improved_curvature(phi, dx, dy)  # Shape: (M, N)
    curvature_value = jnp.stack([curvature_value, curvature_value], axis=-1)

    # Step 2: Calculate the gradient of phi
    grad_phi = jax_gradient(phi, dx, dy)  # Shape: (M, N, 2)

    # Step 3: Calculate the norm of the gradient
    norm_grad_phi = jax_norm(grad_phi)  # Shape: (M, N) 
    norm_grad_phi = jnp.stack([norm_grad_phi, norm_grad_phi], axis=-1)
    We = jax_calculate_weber_number(phi, We1, We2)
    We = jnp.stack([We, We], axis=-1)

    # Step 4: Calculate the surface tension force
    tension_force = (3 * jnp.sqrt(2) * epsilon / (4 * We)) * curvature_value * norm_grad_phi * grad_phi

    return tension_force


def apply_surface_tension_boundary_conditions(surface_tension, phi, contact_angle=60):
    """Apply boundary conditions to the surface tension force.
    
    Args:
        surface_tension (np.ndarray): Surface tension force (shape: (Nx, Ny, 2)).
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
        contact_angle (float): Contact angle in degrees.
    
    Returns:
        np.ndarray: Surface tension force with boundary conditions applied.
    """
    # Make a copy to avoid modifying the original
    sf = surface_tension.copy()
    
    # Bottom boundary (wall): Adjust normal component based on contact angle
    # and preserve tangential component
    theta = (180 - contact_angle) * np.pi / 180
    sf[:, 0, 1] = sf[:, 1, 1] * np.cos(theta)  # Normal component (y)
    sf[:, 0, 0] = sf[:, 1, 0]                  # Tangential component (x)
    
    # Top boundary (open): Zero gradient
    sf[:, -1, :] = sf[:, -2, :]
    
    # Left and right boundaries: Zero gradient
    sf[0, :, :] = sf[1, :, :]
    sf[-1, :, :] = sf[-2, :, :]
    
    return sf


@jit
def jax_apply_surface_tension_boundary_conditions(surface_tension, phi, contact_angle=60):
    """JAX-compiled version of surface tension boundary conditions.
    
    Args:
        surface_tension (jnp.ndarray): Surface tension force (shape: (Nx, Ny, 2)).
        phi (jnp.ndarray): Phase field (shape: (Nx, Ny)).
        contact_angle (float): Contact angle in degrees.
    
    Returns:
        jnp.ndarray: Surface tension force with boundary conditions applied.
    """
    # Bottom boundary (wall): Adjust normal component based on contact angle
    # and preserve tangential component
    theta = (180 - contact_angle) * jnp.pi / 180
    sf = surface_tension.at[:, 0, 1].set(surface_tension[:, 1, 1] * jnp.cos(theta))  # Normal component (y)
    sf = sf.at[:, 0, 0].set(surface_tension[:, 1, 0])                  # Tangential component (x)
    
    # Top boundary (open): Zero gradient
    sf = sf.at[:, -1, :].set(sf[:, -2, :])
    
    # Left and right boundaries: Zero gradient
    sf = sf.at[0, :, :].set(sf[1, :, :])
    sf = sf.at[-1, :, :].set(sf[-2, :, :])
    
    return sf


class SurfaceTensionSolver:
    """Surface tension solver class for droplet spreading simulation."""
    
    def __init__(self, epsilon, We1, We2, contact_angle):
        """Initialize the surface tension solver.
        
        Args:
            epsilon (float): Interface thickness parameter.
            We1 (float): Weber number for phase 1.
            We2 (float): Weber number for phase 2.
            contact_angle (float): Contact angle in degrees.
        """
        self.epsilon = epsilon
        self.We1 = We1
        self.We2 = We2
        self.contact_angle = contact_angle
    
    def calculate_force(self, phi, dx, dy, use_jax=False):
        """Calculate surface tension force.
        
        Args:
            phi: Phase field array.
            dx: Grid spacing in x-direction.
            dy: Grid spacing in y-direction.
            use_jax: Whether to use JAX-compiled functions.
        
        Returns:
            Surface tension force array.
        """
        if use_jax:
            return jax_surface_tension_force(phi, self.epsilon, self.We1, self.We2, dx, dy)
        else:
            return surface_tension_force(phi, self.epsilon, self.We1, self.We2, dx, dy)
    
    def apply_boundary_conditions(self, surface_tension, phi, use_jax=False):
        """Apply boundary conditions to surface tension force.
        
        Args:
            surface_tension: Surface tension force array.
            phi: Phase field array.
            use_jax: Whether to use JAX-compiled functions.
        
        Returns:
            Surface tension force with boundary conditions applied.
        """
        if use_jax:
            return jax_apply_surface_tension_boundary_conditions(surface_tension, phi, self.contact_angle)
        else:
            return apply_surface_tension_boundary_conditions(surface_tension, phi, self.contact_angle)
