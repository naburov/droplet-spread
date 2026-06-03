#!/usr/bin/env python3
"""
Test script to verify gravity implementation.
This script creates a simple test case to check if gravity is being applied correctly.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from physics.fluid_dynamics import jax_update_velocity
from physics.properties import jax_calculate_density
from simulation.initial_conditions import initialize_phase

def test_gravity():
    """Test gravity implementation with a simple case."""
    
    # Simple test parameters
    Nx, Ny = 64, 64
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / Nx, Ly / Ny
    
    # Physical parameters
    rho1 = 1.0      # Air density
    rho2 = 1000.0   # Water density
    Re1 = 10.0
    Re2 = 100.0
    Fr = 0.1
    g = -100.0      # Strong downward gravity
    include_gravity = True
    
    # Create a simple droplet in the center
    phi = initialize_phase(Nx, Ny, 0.2)
    phi = jnp.array(phi)
    
    # Initialize velocity field (initially at rest)
    U = jnp.zeros((Nx, Ny, 2))
    
    # Initialize pressure field
    P = jnp.zeros((Nx, Ny))
    
    # Create a simple surface tension force (zero for this test)
    surface_tension = jnp.zeros((Nx, Ny, 2))
    
    # Time step
    dt = 0.001
    
    print("=== Gravity Test ===")
    print(f"Gravity: g = {g}")
    print(f"Froude number: Fr = {Fr}")
    print(f"Include gravity: {include_gravity}")
    print(f"Gravity force magnitude: |g/Fr| = {abs(g/Fr)}")
    
    # Test velocity update with gravity
    print("\nBefore velocity update:")
    print(f"Velocity range: [{U.min():.6f}, {U.max():.6f}]")
    print(f"Y-velocity range: [{U[..., 1].min():.6f}, {U[..., 1].max():.6f}]")
    
    # Flat geometry for tests
    solid_mask = jnp.zeros((Nx, Ny), dtype=jnp.bool_)
    f_1_grid = jnp.zeros((Nx, Ny))
    f_2_grid = jnp.zeros((Nx, Ny))

    # Update velocity with gravity
    U_new = jax_update_velocity(
        U, P, surface_tension, dt, dx, dy,
        rho1, rho2, Re1, Re2, Fr, g, phi, solid_mask, f_1_grid, f_2_grid,
        include_gravity=include_gravity
    )
    
    print("\nAfter velocity update:")
    print(f"Velocity range: [{U_new.min():.6f}, {U_new.max():.6f}]")
    print(f"Y-velocity range: [{U_new[..., 1].min():.6f}, {U_new[..., 1].max():.6f}]")
    
    # Calculate the change in velocity
    delta_U = U_new - U
    print(f"\nVelocity change range: [{delta_U.min():.6f}, {delta_U.max():.6f}]")
    print(f"Y-velocity change range: [{delta_U[..., 1].min():.6f}, {delta_U[..., 1].max():.6f}]")
    
    # Expected gravity force per time step
    expected_gravity_force = (1 / Fr) * g * dt
    print(f"\nExpected gravity force per time step: {expected_gravity_force:.6f}")
    
    # Check if the velocity change matches expected gravity
    print(f"Actual Y-velocity change: {delta_U[..., 1].mean():.6f}")
    print(f"Expected Y-velocity change: {expected_gravity_force:.6f}")
    
    # Create a visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Phase field
    im1 = axes[0, 0].imshow(phi.T, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
    axes[0, 0].set_title('Phase Field (phi)')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Initial velocity
    im2 = axes[0, 1].imshow(U[..., 1].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdBu_r')
    axes[0, 1].set_title('Initial Y-Velocity')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Updated velocity
    im3 = axes[1, 0].imshow(U_new[..., 1].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdBu_r')
    axes[1, 0].set_title('Updated Y-Velocity')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Velocity change
    im4 = axes[1, 1].imshow(delta_U[..., 1].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdBu_r')
    axes[1, 1].set_title('Y-Velocity Change (Gravity Effect)')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('gravity_test.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved as 'gravity_test.png'")
    
    # Test without gravity for comparison
    print("\n=== Test without gravity ===")
    U_no_gravity = jax_update_velocity(
        U, P, surface_tension, dt, dx, dy,
        rho1, rho2, Re1, Re2, Fr, g, phi, solid_mask, f_1_grid, f_2_grid,
        include_gravity=False
    )
    
    delta_U_no_gravity = U_no_gravity - U
    print(f"Y-velocity change without gravity: {delta_U_no_gravity[..., 1].mean():.6f}")
    
    return U_new, delta_U

if __name__ == "__main__":
    test_gravity()
