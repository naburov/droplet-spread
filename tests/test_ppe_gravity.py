#!/usr/bin/env python3
"""
Test script to check if PPE is counteracting gravity.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from physics.fluid_dynamics import jax_update_velocity, jax_check_continuity
from physics.surface_tension import SurfaceTensionSolver
from physics.pressure import PressureSolver
from physics.properties import jax_calculate_density
from simulation.initial_conditions import initialize_phase
from solvers.sparse_solver import SparseSolverWrapper
from solvers.projection_methods import ppe

def test_ppe_gravity_interaction():
    """Test how PPE affects gravity."""
    
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
    
    print("=== PPE-Gravity Interaction Test ===")
    print(f"Gravity: g = {g}")
    print(f"Froude number: Fr = {Fr}")
    
    # Calculate surface tension
    surface_tension_solver = SurfaceTensionSolver(epsilon, We1, We2, contact_angle)
    surface_tension = surface_tension_solver.calculate_force(phi, dx, dy, use_jax=True)
    surface_tension = surface_tension_solver.apply_boundary_conditions(surface_tension, phi, use_jax=True)
    
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
    
    # Flat geometry for tests
    solid_mask = jnp.zeros((Nx, Ny), dtype=jnp.bool_)
    f_1_grid = jnp.zeros((Nx, Ny))
    f_2_grid = jnp.zeros((Nx, Ny))

    # Test 1: Velocity update with gravity
    print("\n--- Test 1: Velocity update with gravity ---")
    U_after_velocity = jax_update_velocity(
        U, P_new, surface_tension, dt, dx, dy,
        rho1, rho2, Re1, Re2, Fr, g, phi, solid_mask, f_1_grid, f_2_grid,
        include_gravity=include_gravity
    )
    
    print(f"Y-velocity after velocity update: {U_after_velocity[..., 1].mean():.6f}")
    
    # Check continuity
    divergence, max_div, mean_div = jax_check_continuity(U_after_velocity, dx, dy, f_1_grid)
    print(f"Max divergence: {max_div:.6f}")
    print(f"Mean divergence: {mean_div:.6f}")
    
    # Test 2: Apply PPE correction
    print("\n--- Test 2: Apply PPE correction ---")
    
    # Create correction solver for PPE
    correction_solver = SparseSolverWrapper(Nx, Ny, dx, dy, "pyamg")
    correction_solver.set_top_boundary_condition("neumann")
    correction_solver.set_bottom_boundary_condition("neumann")
    correction_solver.set_left_boundary_condition("neumann")
    correction_solver.set_right_boundary_condition("neumann")
    correction_solver.create_sparse_matrix()
    
    # Apply PPE
    U_after_ppe = ppe(U_after_velocity, dx, dy, dt, correction_solver, 
                      div_threshold=0.01, max_div_threshold=1.0, mean_div_threshold=0.1)
    
    print(f"Y-velocity after PPE: {U_after_ppe[..., 1].mean():.6f}")
    
    # Check continuity after PPE
    divergence_after, max_div_after, mean_div_after = jax_check_continuity(U_after_ppe, dx, dy, f_1_grid)
    print(f"Max divergence after PPE: {max_div_after:.6f}")
    print(f"Mean divergence after PPE: {mean_div_after:.6f}")
    
    # Calculate changes
    delta_U_velocity = U_after_velocity - U
    delta_U_ppe = U_after_ppe - U_after_velocity
    delta_U_total = U_after_ppe - U
    
    print(f"\nY-velocity change from gravity: {delta_U_velocity[..., 1].mean():.6f}")
    print(f"Y-velocity change from PPE: {delta_U_ppe[..., 1].mean():.6f}")
    print(f"Total Y-velocity change: {delta_U_total[..., 1].mean():.6f}")
    
    # Test 3: Check if PPE is removing the gravity effect
    print(f"\nPPE effect on gravity: {delta_U_ppe[..., 1].mean():.6f}")
    print(f"Gravity effect: {delta_U_velocity[..., 1].mean():.6f}")
    print(f"Net effect: {delta_U_total[..., 1].mean():.6f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Phase field
    im1 = axes[0, 0].imshow(phi.T, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
    axes[0, 0].set_title('Phase Field (phi)')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Y-velocity after gravity
    im2 = axes[0, 1].imshow(U_after_velocity[..., 1].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdBu_r')
    axes[0, 1].set_title('Y-Velocity After Gravity')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Y-velocity after PPE
    im3 = axes[0, 2].imshow(U_after_ppe[..., 1].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdBu_r')
    axes[0, 2].set_title('Y-Velocity After PPE')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Gravity effect
    im4 = axes[1, 0].imshow(delta_U_velocity[..., 1].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdBu_r')
    axes[1, 0].set_title('Gravity Effect')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # PPE effect
    im5 = axes[1, 1].imshow(delta_U_ppe[..., 1].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdBu_r')
    axes[1, 1].set_title('PPE Effect')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Total effect
    im6 = axes[1, 2].imshow(delta_U_total[..., 1].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdBu_r')
    axes[1, 2].set_title('Total Effect')
    axes[1, 2].set_xlabel('X')
    axes[1, 2].set_ylabel('Y')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('ppe_gravity_test.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved as 'ppe_gravity_test.png'")
    
    return U_after_velocity, U_after_ppe, delta_U_velocity, delta_U_ppe

if __name__ == "__main__":
    test_ppe_gravity_interaction()
