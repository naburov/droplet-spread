#!/usr/bin/env python3
"""
Chemical Potential Contact-Angle Robin Boundary Condition Implementation.

Implements the contact-angle Robin BC for chemical potential as specified in bcs.md:
∂nφ = -(√2/η) cosθw (φ² - 1)

where:
- η = ε (interface thickness parameter)
- θw is the contact angle
- φ is the phase field
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jax_utils import jax_dx, jax_dy, jax_laplacian

@jit
def apply_chemical_potential_contact_angle_bc(phi, dx, dy, contact_angle, epsilon, strength=0.1):
    """Apply contact-angle Robin BC for chemical potential.
    
    Implements: ∂nφ = -(√2/η) cosθw (φ² - 1)
    
    Args:
        phi (jnp.ndarray): Phase field (shape: (Nx, Ny))
        dx (float): Grid spacing in x-direction
        dy (float): Grid spacing in y-direction  
        contact_angle (float): Contact angle in degrees
        epsilon (float): Interface thickness parameter (η)
        strength (float): Strength of the boundary condition (0-1)
    
    Returns:
        jnp.ndarray: Phase field with contact-angle Robin BC applied
    """
    # Convert contact angle to radians
    theta_w = contact_angle * jnp.pi / 180
    
    # Create a copy of phi
    phi_new = phi.copy()
    
    # Apply contact-angle Robin BC at bottom boundary (y=0)
    # Formula: ∂nφ = -(√2/η) cosθw (φ² - 1)
    # where ∂nφ ≈ (φ[:, 1] - φ[:, 0]) / dy
    
    # Get phi values at first interior node (y=1)
    phi_interior = phi[:, 1]
    
    # Compute the right-hand side: -(√2/η) cosθw (φ² - 1)
    sqrt2_over_eta = jnp.sqrt(2) / epsilon
    cos_theta_w = jnp.cos(theta_w)
    phi_squared_minus_one = phi_interior**2 - 1
    
    # Compute the normal derivative: ∂nφ = -(√2/η) cosθw (φ² - 1)
    normal_derivative = -sqrt2_over_eta * cos_theta_w * phi_squared_minus_one
    
    # Apply the boundary condition with strength factor: φ[:, 0] = φ[:, 1] - dy * ∂nφ * strength
    phi_new = phi_new.at[:, 0].set(phi_interior - dy * normal_derivative * strength)
    
    # Apply Neumann BC (∂nφ = 0) at other boundaries
    # Top boundary (y=Ny-1)
    phi_new = phi_new.at[:, -1].set(phi[:, -2])
    
    # Left boundary (x=0)  
    phi_new = phi_new.at[0, :].set(phi[1, :])
    
    # Right boundary (x=Nx-1)
    phi_new = phi_new.at[-1, :].set(phi[-2, :])
    
    return phi_new

@jit
def compute_chemical_potential(phi, epsilon, dx, dy):
    """Compute chemical potential: μ = f'(φ) - ε²∇²φ
    
    Args:
        phi (jnp.ndarray): Phase field (shape: (Nx, Ny))
        epsilon (float): Interface thickness parameter
        dx (float): Grid spacing in x-direction
        dy (float): Grid spacing in y-direction
    
    Returns:
        jnp.ndarray: Chemical potential (shape: (Nx, Ny))
    """
    # Compute f'(φ) = φ³ - φ
    f_prime = phi**3 - phi
    
    # Compute ∇²φ using existing function
    laplacian_phi = jax_laplacian(phi, dx, dy)
    
    # Chemical potential: μ = f'(φ) - ε²∇²φ
    mu = f_prime - epsilon**2 * laplacian_phi
    
    return mu

def test_chemical_potential_contact_angle_bc():
    """Test the chemical potential contact-angle Robin BC implementation."""
    print("🧪 Testing Chemical Potential Contact-Angle Robin BC")
    print("=" * 60)
    
    # Test parameters
    Nx, Ny = 32, 32
    dx, dy = 1.0/32, 1.0/32
    contact_angle = 120.0  # degrees
    epsilon = 0.02
    
    # Create a test phase field (simple droplet)
    x = jnp.linspace(0, 1, Nx)
    y = jnp.linspace(0, 1, Ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    
    # Create a circular droplet
    center_x, center_y = 0.5, 0.3
    radius = 0.2
    distance = jnp.sqrt((X - center_x)**2 + (Y - center_y)**2)
    phi_test = jnp.tanh((radius - distance) / (epsilon * 0.5))
    
    print(f"Test parameters:")
    print(f"  Grid: {Nx}x{Ny}")
    print(f"  Contact angle: {contact_angle}°")
    print(f"  Epsilon: {epsilon}")
    print(f"  Phase field range: [{jnp.min(phi_test):.3f}, {jnp.max(phi_test):.3f}]")
    
    # Apply the boundary condition
    phi_with_bc = apply_chemical_potential_contact_angle_bc(phi_test, dx, dy, contact_angle, epsilon)
    
    # Compute chemical potential
    mu = compute_chemical_potential(phi_with_bc, epsilon, dx, dy)
    
    print(f"\nResults:")
    print(f"  Phase field range after BC: [{jnp.min(phi_with_bc):.3f}, {jnp.max(phi_with_bc):.3f}]")
    print(f"  Chemical potential range: [{jnp.min(mu):.3f}, {jnp.max(mu):.3f}]")
    
    # Verify the boundary condition at bottom
    phi_bottom = phi_with_bc[:, 0]
    phi_interior = phi_with_bc[:, 1]
    
    # Compute normal derivative using forward difference: (φ_interior - φ_bottom) / dy
    normal_derivative = (phi_interior - phi_bottom) / dy
    
    # Expected normal derivative: -(√2/η) cosθw (φ² - 1)
    theta_w = contact_angle * jnp.pi / 180
    sqrt2_over_eta = jnp.sqrt(2) / epsilon
    cos_theta_w = jnp.cos(theta_w)
    expected_normal_derivative = -sqrt2_over_eta * cos_theta_w * (phi_interior**2 - 1)
    
    # Compare (use relative error for better assessment)
    error = jnp.abs(normal_derivative - expected_normal_derivative)
    relative_error = jnp.where(jnp.abs(expected_normal_derivative) > 1e-10, 
                              error / jnp.abs(expected_normal_derivative), 
                              error)
    max_error = jnp.max(error)
    mean_error = jnp.mean(error)
    max_relative_error = jnp.max(relative_error)
    mean_relative_error = jnp.mean(relative_error)
    
    print(f"\nBoundary condition verification:")
    print(f"  Max absolute error: {max_error:.2e}")
    print(f"  Mean absolute error: {mean_error:.2e}")
    print(f"  Max relative error: {max_relative_error:.2e}")
    print(f"  Mean relative error: {mean_relative_error:.2e}")
    print(f"  BC satisfied (abs): {max_error < 1e-6}")
    print(f"  BC satisfied (rel): {max_relative_error < 1e-3}")
    
    # Check other boundaries (should be Neumann: ∂nφ = 0)
    # Top boundary
    phi_top = phi_with_bc[:, -1]
    phi_top_interior = phi_with_bc[:, -2]
    top_normal_derivative = (phi_top - phi_top_interior) / dy
    top_error = jnp.max(jnp.abs(top_normal_derivative))
    
    # Left boundary
    phi_left = phi_with_bc[0, :]
    phi_left_interior = phi_with_bc[1, :]
    left_normal_derivative = (phi_left_interior - phi_left) / dx
    left_error = jnp.max(jnp.abs(left_normal_derivative))
    
    # Right boundary
    phi_right = phi_with_bc[-1, :]
    phi_right_interior = phi_with_bc[-2, :]
    right_normal_derivative = (phi_right - phi_right_interior) / dx
    right_error = jnp.max(jnp.abs(right_normal_derivative))
    
    print(f"\nOther boundaries (should be Neumann ∂nφ = 0):")
    print(f"  Top error: {top_error:.2e}")
    print(f"  Left error: {left_error:.2e}")
    print(f"  Right error: {right_error:.2e}")
    print(f"  All Neumann satisfied: {max(top_error, left_error, right_error) < 1e-10}")
    
    return phi_with_bc, mu, max_error < 1e-10

if __name__ == "__main__":
    test_chemical_potential_contact_angle_bc()
