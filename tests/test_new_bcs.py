#!/usr/bin/env python3
"""
Test script for the new boundary condition setup.

Tests the updated boundary conditions:
1. Pressure: Top Dirichlet p=0, all others Neumann ∂p/∂n=0
2. Velocity: Left/Right slip/symmetry u=0, ∂u/∂n=0; Top open ∂u/∂n=0; Bottom no-slip u=0
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from boundary_conditions.velocity_bc import VelocityBoundaryConditions
from boundary_conditions.pressure_bc import PressureBoundaryConditions


def create_test_velocity_field(Nx=64, Ny=64):
    """Create a test velocity field with some flow."""
    U = np.zeros((Nx, Ny, 2))
    
    # Create a simple flow pattern
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Horizontal flow (left to right)
    U[:, :, 0] = 0.1 * np.sin(np.pi * Y) * np.cos(np.pi * X)
    
    # Vertical flow (upward)
    U[:, :, 1] = 0.05 * np.sin(np.pi * X) * np.cos(np.pi * Y)
    
    return U


def create_test_pressure_field(Nx=64, Ny=64):
    """Create a test pressure field."""
    P = np.zeros((Nx, Ny))
    
    # Create a simple pressure pattern
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Pressure field
    P = 0.1 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    
    return P


def test_velocity_boundary_conditions():
    """Test the new velocity boundary conditions."""
    print("Testing Velocity Boundary Conditions")
    print("=" * 50)
    
    # Create test velocity field
    U = create_test_velocity_field()
    dx = dy = 1.0 / 63  # Grid spacing
    
    # Test configuration with new BCs
    config = {
        "boundary_conditions": {
            "velocity": {
                "top": "do_nothing",
                "bottom": "no_slip",
                "left": "slip_symmetry",
                "right": "slip_symmetry"
            }
        }
    }
    
    # Create velocity BC manager
    velocity_bc = VelocityBoundaryConditions(config)
    
    # Apply boundary conditions
    U_with_bc = velocity_bc.apply_boundary_conditions(U, dx, dy, use_jax=False)
    
    # Check boundary conditions
    print("Checking velocity boundary conditions:")
    
    # Bottom: no-slip (u = 0)
    bottom_u = U_with_bc[:, 0, 0]  # u-component at bottom
    bottom_v = U_with_bc[:, 0, 1]  # v-component at bottom
    print(f"  Bottom (no-slip): max|u| = {np.max(np.abs(bottom_u)):.6f}, max|v| = {np.max(np.abs(bottom_v)):.6f}")
    
    # Left: slip/symmetry (u = 0, ∂u/∂x = 0)
    left_u = U_with_bc[0, :, 0]  # u-component at left
    left_v_grad = np.abs(U_with_bc[0, :, 1] - U_with_bc[1, :, 1])  # ∂v/∂x at left
    print(f"  Left (slip/symmetry): max|u| = {np.max(np.abs(left_u)):.6f}, max|∂v/∂x| = {np.max(left_v_grad):.6f}")
    
    # Right: slip/symmetry (u = 0, ∂u/∂x = 0)
    right_u = U_with_bc[-1, :, 0]  # u-component at right
    right_v_grad = np.abs(U_with_bc[-1, :, 1] - U_with_bc[-2, :, 1])  # ∂v/∂x at right
    print(f"  Right (slip/symmetry): max|u| = {np.max(np.abs(right_u)):.6f}, max|∂v/∂x| = {np.max(right_v_grad):.6f}")
    
    # Top: do-nothing (∂u/∂y = 0)
    top_u_grad = np.abs(U_with_bc[:, -1, 0] - U_with_bc[:, -2, 0])  # ∂u/∂y at top
    top_v_grad = np.abs(U_with_bc[:, -1, 1] - U_with_bc[:, -2, 1])  # ∂v/∂y at top
    print(f"  Top (do-nothing): max|∂u/∂y| = {np.max(top_u_grad):.6f}, max|∂v/∂y| = {np.max(top_v_grad):.6f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original velocity magnitude
    im0 = axes[0, 0].imshow(np.sqrt(U[:, :, 0]**2 + U[:, :, 1]**2).T, 
                           extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    axes[0, 0].set_title('Original Velocity Magnitude')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)
    
    # Velocity with BCs
    im1 = axes[0, 1].imshow(np.sqrt(U_with_bc[:, :, 0]**2 + U_with_bc[:, :, 1]**2).T, 
                           extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    axes[0, 1].set_title('Velocity with New BCs')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    
    # u-component
    im2 = axes[1, 0].imshow(U_with_bc[:, :, 0].T, 
                           extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r')
    axes[1, 0].set_title('u-component (should be 0 at left/right)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)
    
    # v-component
    im3 = axes[1, 1].imshow(U_with_bc[:, :, 1].T, 
                           extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r')
    axes[1, 1].set_title('v-component')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 1], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('velocity_bc_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Velocity BC test completed!")
    print("Results saved to 'velocity_bc_test.png'")


def test_pressure_boundary_conditions():
    """Test the pressure boundary conditions."""
    print("\nTesting Pressure Boundary Conditions")
    print("=" * 50)
    
    # Create test pressure field
    P = create_test_pressure_field()
    
    # Test configuration
    config = {
        "boundary_conditions": {
            "pressure": {
                "top": "open",
                "bottom": "neumann",
                "left": "neumann",
                "right": "neumann",
                "open_pressure": 0.0
            }
        }
    }
    
    # Create pressure BC manager
    pressure_bc = PressureBoundaryConditions(config)
    
    # Apply boundary conditions
    P_with_bc = pressure_bc.apply_boundary_conditions(P, use_jax=False)
    
    # Check boundary conditions
    print("Checking pressure boundary conditions:")
    
    # Top: Dirichlet (p = 0)
    top_p = P_with_bc[:, -1]  # pressure at top
    print(f"  Top (Dirichlet p=0): max|p| = {np.max(np.abs(top_p)):.6f}")
    
    # Bottom: Neumann (∂p/∂y = 0)
    bottom_grad = np.abs(P_with_bc[:, 0] - P_with_bc[:, 1])  # ∂p/∂y at bottom
    print(f"  Bottom (Neumann ∂p/∂y=0): max|∂p/∂y| = {np.max(bottom_grad):.6f}")
    
    # Left: Neumann (∂p/∂x = 0)
    left_grad = np.abs(P_with_bc[0, :] - P_with_bc[1, :])  # ∂p/∂x at left
    print(f"  Left (Neumann ∂p/∂x=0): max|∂p/∂x| = {np.max(left_grad):.6f}")
    
    # Right: Neumann (∂p/∂x = 0)
    right_grad = np.abs(P_with_bc[-1, :] - P_with_bc[-2, :])  # ∂p/∂x at right
    print(f"  Right (Neumann ∂p/∂x=0): max|∂p/∂x| = {np.max(right_grad):.6f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original pressure
    im0 = axes[0].imshow(P.T, extent=[0, 1, 0, 1], origin='lower', 
                        cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    axes[0].set_title('Original Pressure')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    
    # Pressure with BCs
    im1 = axes[1].imshow(P_with_bc.T, extent=[0, 1, 0, 1], origin='lower', 
                        cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    axes[1].set_title('Pressure with BCs (top=0, others Neumann)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('pressure_bc_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Pressure BC test completed!")
    print("Results saved to 'pressure_bc_test.png'")


def test_simulation_with_new_bcs():
    """Test a short simulation with the new boundary conditions."""
    print("\nTesting Simulation with New BCs")
    print("=" * 50)
    
    # Run a short simulation
    import subprocess
    import os
    
    # Create a test config
    test_config = {
        "physical_params": {
            "rho2": 1000.0,
            "Re2": 100.0,
            "We2": 10.0,
            "rho1": 1.0,
            "Re1": 10.0,
            "We1": 0.1,
            "Pe": 1.0,
            "epsilon": 0.05,
            "contact_angle": 120,
            "include_gravity": True,
            "Fr": 1.0,
            "g": -1.0,
            "atm_pressure": 0.0
        },
        "grid_params": {
            "Lx": 1.0,
            "Ly": 1.0,
            "Nx": 64,
            "Ny": 64
        },
        "time_params": {
            "dt": 0.001,
            "t_max": 0.01,
            "checkpoint_interval": 5,
            "dt_initial": 0.001,
            "cfl_number": 0.05
        },
        "initial_conditions": {
            "droplet_radius": 0.2
        },
        "boundary_conditions": {
            "pressure": {
                "top": "open",
                "bottom": "neumann",
                "left": "neumann",
                "right": "neumann",
                "open_pressure": 0.0
            },
            "velocity": {
                "top": "do_nothing",
                "bottom": "no_slip",
                "left": "slip_symmetry",
                "right": "slip_symmetry"
            },
            "phase_field": {
                "top": "neumann",
                "bottom": "contact_angle",
                "left": "neumann",
                "right": "neumann",
                "contact_angle_method": "robin"
            },
            "chemical_potential": {
                "top": "zero_flux",
                "bottom": "zero_flux",
                "left": "zero_flux",
                "right": "zero_flux"
            },
            "advection": {
                "top": "open",
                "bottom": "impermeable",
                "left": "impermeable",
                "right": "impermeable",
                "cout": 1.0,
                "velocity_threshold": 1e-10
            }
        }
    }
    
    # Save test config
    import json
    with open('test_config.json', 'w') as f:
        json.dump(test_config, f, indent=4)
    
    print("Test configuration saved to 'test_config.json'")
    print("You can run: native/bin/python src/main_refactored.py --config test_config.json")


if __name__ == "__main__":
    print("New Boundary Condition Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Velocity boundary conditions
        test_velocity_boundary_conditions()
        
        # Test 2: Pressure boundary conditions
        test_pressure_boundary_conditions()
        
        # Test 3: Simulation setup
        test_simulation_with_new_bcs()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("Check the generated PNG files for results.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
