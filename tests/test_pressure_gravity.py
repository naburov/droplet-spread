#!/usr/bin/env python3
"""
Test script to check if pressure is counteracting gravity.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from physics.fluid_dynamics import jax_update_velocity
from physics.surface_tension import SurfaceTensionSolver
from physics.pressure import PressureSolver
from physics.properties import jax_calculate_density
from simulation.initial_conditions import initialize_phase
from solvers.sparse_solver import SparseSolverWrapper

def test_pressure_gravity_interaction():
    """Test how pressure interacts with gravity."""
    
    # Simple test parameters
    Nx, Ny = 64, 64
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / Nx, Ly / Ny
    
    # Physical parameters
    rho1 = 1.0      # Air density
    rho2 = 1000.0   # Water density
    Re1 = 10.0
    Re2 = 100.0
    We1 = 0.1
    We2 = 10.0
    Fr = 0.1
    g = -10.0
    include_gravity = True
    epsilon = 0.05
    contact_angle = 120
    atm_pressure = 0.0
    
    # Create a simple droplet in the center
    phi = initialize_phase(Nx, Ny, 0.2)
    phi = jnp.array(phi)
    
    # Initialize velocity field (initially at rest)
    U = jnp.zeros((Nx, Ny, 2))
    
    # Initialize pressure field
    P = jnp.zeros((Nx, Ny))
    
    # Time step
    dt = 0.001
    
    print("=== Pressure-Gravity Interaction Test ===")
    print(f"Gravity: g = {g}")
    print(f"Froude number: Fr = {Fr}")
    
    # Flat geometry for tests
    solid_mask = jnp.zeros((Nx, Ny), dtype=jnp.bool_)
    f_1_grid = jnp.zeros((Nx, Ny))
    f_2_grid = jnp.zeros((Nx, Ny))

    # Test 1: Velocity update with gravity only (no pressure)
    print("\n--- Test 1: Gravity only (no pressure) ---")
    U_gravity_only = jax_update_velocity(
        U, P, jnp.zeros((Nx, Ny, 2)), dt, dx, dy,
        rho1, rho2, Re1, Re2, Fr, g, phi, solid_mask, f_1_grid, f_2_grid,
        include_gravity=include_gravity
    )
    
    print(f"Y-velocity change (gravity only): {U_gravity_only[..., 1].mean():.6f}")
    
    # Test 2: Calculate surface tension and pressure
    print("\n--- Test 2: With surface tension and pressure ---")
    
    # Calculate surface tension
    surface_tension_solver = SurfaceTensionSolver(epsilon, We1, We2, contact_angle)
    surface_tension = surface_tension_solver.calculate_force(phi, dx, dy, use_jax=True)
    surface_tension = surface_tension_solver.apply_boundary_conditions(surface_tension, phi, use_jax=True)
    
    print(f"Surface tension range: [{surface_tension.min():.6f}, {surface_tension.max():.6f}]")
    print(f"Y-surface tension range: [{surface_tension[..., 1].min():.6f}, {surface_tension[..., 1].max():.6f}]")
    
    # Calculate pressure
    pressure_solver = PressureSolver(rho1, rho2, g, atm_pressure)
    
    # Create sparse solver for pressure
    pressure_linear_solver = SparseSolverWrapper(Nx, Ny, dx, dy, "pyamg")
    pressure_linear_solver.set_top_boundary_condition("neumann")
    pressure_linear_solver.set_bottom_boundary_condition("neumann")
    pressure_linear_solver.set_left_boundary_condition("neumann")
    pressure_linear_solver.set_right_boundary_condition("neumann")
    pressure_linear_solver.create_sparse_matrix()
    
    P_new = pressure_solver.update_pressure(surface_tension, Nx, Ny, dx, dy, phi, pressure_linear_solver, use_jax=True)
    
    print(f"Pressure range: [{P_new.min():.6f}, {P_new.max():.6f}]")
    
    # Test 3: Velocity update with pressure
    U_with_pressure = jax_update_velocity(
        U, P_new, surface_tension, dt, dx, dy,
        rho1, rho2, Re1, Re2, Fr, g, phi, solid_mask, f_1_grid, f_2_grid,
        include_gravity=include_gravity
    )
    
    print(f"Y-velocity change (with pressure): {U_with_pressure[..., 1].mean():.6f}")
    
    # Test 4: Check pressure gradient
    from numerics.finite_differences import jax_gradient
    p_grad = jax_gradient(P_new, dx, dy)
    print(f"Pressure gradient range: [{p_grad.min():.6f}, {p_grad.max():.6f}]")
    print(f"Y-pressure gradient range: [{p_grad[..., 1].min():.6f}, {p_grad[..., 1].max():.6f}]")
    
    # Test 5: Check density
    rho = jax_calculate_density(phi, rho1, rho2)
    print(f"Density range: [{rho.min():.6f}, {rho.max():.6f}]")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Phase field
    im1 = axes[0, 0].imshow(phi.T, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
    axes[0, 0].set_title('Phase Field (phi)')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Density
    im2 = axes[0, 1].imshow(rho.T, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
    axes[0, 1].set_title('Density (rho)')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Pressure
    im3 = axes[0, 2].imshow(P_new.T, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdBu_r')
    axes[0, 2].set_title('Pressure')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Y-velocity change (gravity only)
    im4 = axes[1, 0].imshow((U_gravity_only - U)[..., 1].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdBu_r')
    axes[1, 0].set_title('Y-Velocity Change (Gravity Only)')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Y-velocity change (with pressure)
    im5 = axes[1, 1].imshow((U_with_pressure - U)[..., 1].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdBu_r')
    axes[1, 1].set_title('Y-Velocity Change (With Pressure)')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Y-pressure gradient
    im6 = axes[1, 2].imshow(p_grad[..., 1].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdBu_r')
    axes[1, 2].set_title('Y-Pressure Gradient')
    axes[1, 2].set_xlabel('X')
    axes[1, 2].set_ylabel('Y')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('pressure_gravity_test.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved as 'pressure_gravity_test.png'")
    
    return U_gravity_only, U_with_pressure, P_new, p_grad

if __name__ == "__main__":
    test_pressure_gravity_interaction()
