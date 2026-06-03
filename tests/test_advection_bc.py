#!/usr/bin/env python3
"""
Test script for advection boundary conditions.

This script demonstrates the new advection boundary conditions:
- Bottom: Impermeable (un = 0)
- Top/Left/Right: Open with radiation (∂tφ + cout ∂nφ = 0)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from boundary_conditions.advection_bc import AdvectionBoundaryConditions, apply_advection_boundary_conditions
from boundary_conditions.phase_field_bc import PhaseFieldBoundaryConditions


def create_test_phase_field(Nx=64, Ny=64, radius=0.2):
    """Create a test phase field with a circular droplet."""
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Center of droplet
    cx, cy = 0.5, 0.3
    
    # Distance from center
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    # Create phase field: +1 inside droplet, -1 outside
    phi = np.tanh((radius - r) / 0.05)
    
    return phi, x, y


def create_test_velocity_field(Nx=64, Ny=64):
    """Create a test velocity field."""
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create a simple velocity field
    U = np.zeros((Nx, Ny, 2))
    
    # Horizontal velocity (left to right)
    U[:, :, 0] = 0.1 * np.sin(np.pi * Y)  # Zero at boundaries
    
    # Vertical velocity (upward)
    U[:, :, 1] = 0.05 * np.sin(np.pi * X)  # Zero at boundaries
    
    return U


def test_advection_boundary_conditions():
    """Test advection boundary conditions."""
    print("Testing Advection Boundary Conditions")
    print("=" * 50)
    
    # Create test data
    phi, x, y = create_test_phase_field()
    U = create_test_velocity_field()
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = 0.001
    
    # Test configuration
    config = {
        "boundary_conditions": {
            "advection": {
                "top": "open",
                "bottom": "impermeable",
                "left": "open",
                "right": "open",
                "cout": 1.0,
                "velocity_threshold": 1e-10
            }
        }
    }
    
    # Create advection BC manager
    advection_bc = AdvectionBoundaryConditions(config)
    
    # Apply boundary conditions
    phi_with_bc = advection_bc.apply_boundary_conditions(phi, U, dt, dx, dy, use_jax=False)
    
    # Test different cout values
    cout_values = [0.5, 1.0, 2.0]
    
    fig, axes = plt.subplots(2, len(cout_values) + 1, figsize=(4*(len(cout_values) + 1), 8))
    
    # Original phase field
    im0 = axes[0, 0].imshow(phi.T, extent=[0, 1, 0, 1], origin='lower', 
                           cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 0].set_title('Original Phase Field')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)
    
    # Original velocity field
    im0_v = axes[1, 0].imshow(np.sqrt(U[:, :, 0]**2 + U[:, :, 1]**2).T, 
                             extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    axes[1, 0].set_title('Velocity Magnitude')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im0_v, ax=axes[1, 0], shrink=0.8)
    
    # Test different cout values
    for i, cout in enumerate(cout_values):
        config_cout = config.copy()
        config_cout["boundary_conditions"]["advection"]["cout"] = cout
        
        advection_bc_cout = AdvectionBoundaryConditions(config_cout)
        phi_cout = advection_bc_cout.apply_boundary_conditions(phi, U, dt, dx, dy, use_jax=False)
        
        # Plot phase field
        im = axes[0, i+1].imshow(phi_cout.T, extent=[0, 1, 0, 1], origin='lower', 
                                cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, i+1].set_title(f'Phase Field (cout={cout})')
        axes[0, i+1].set_xlabel('x')
        axes[0, i+1].set_ylabel('y')
        plt.colorbar(im, ax=axes[0, i+1], shrink=0.8)
        
        # Plot difference from original
        diff = phi_cout - phi
        im_diff = axes[1, i+1].imshow(diff.T, extent=[0, 1, 0, 1], origin='lower', 
                                     cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        axes[1, i+1].set_title(f'Difference (cout={cout})')
        axes[1, i+1].set_xlabel('x')
        axes[1, i+1].set_ylabel('y')
        plt.colorbar(im_diff, ax=axes[1, i+1], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('advection_bc_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Advection BC test completed!")
    print("Results saved to 'advection_bc_test.png'")


def test_impermeable_vs_open():
    """Test impermeable vs open boundary conditions."""
    print("\nTesting Impermeable vs Open Boundary Conditions")
    print("=" * 50)
    
    phi, x, y = create_test_phase_field()
    U = create_test_velocity_field()
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = 0.001
    
    # Test impermeable bottom
    config_impermeable = {
        "boundary_conditions": {
            "advection": {
                "top": "open",
                "bottom": "impermeable",
                "left": "open",
                "right": "open",
                "cout": 1.0
            }
        }
    }
    
    # Test open bottom
    config_open = {
        "boundary_conditions": {
            "advection": {
                "top": "open",
                "bottom": "open",
                "left": "open",
                "right": "open",
                "cout": 1.0
            }
        }
    }
    
    advection_impermeable = AdvectionBoundaryConditions(config_impermeable)
    advection_open = AdvectionBoundaryConditions(config_open)
    
    phi_impermeable = advection_impermeable.apply_boundary_conditions(phi, U, dt, dx, dy, use_jax=False)
    phi_open = advection_open.apply_boundary_conditions(phi, U, dt, dx, dy, use_jax=False)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    im0 = axes[0].imshow(phi.T, extent=[0, 1, 0, 1], origin='lower', 
                        cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('Original Phase Field')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    
    # Impermeable bottom
    im1 = axes[1].imshow(phi_impermeable.T, extent=[0, 1, 0, 1], origin='lower', 
                        cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Impermeable Bottom')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    
    # Open bottom
    im2 = axes[2].imshow(phi_open.T, extent=[0, 1, 0, 1], origin='lower', 
                        cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_title('Open Bottom')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('impermeable_vs_open_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Impermeable vs Open test completed!")
    print("Results saved to 'impermeable_vs_open_test.png'")


def test_integration_with_phase_field_bc():
    """Test integration with existing phase field boundary conditions."""
    print("\nTesting Integration with Phase Field BCs")
    print("=" * 50)
    
    phi, x, y = create_test_phase_field()
    U = create_test_velocity_field()
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = 0.001
    
    # Configuration with both advection and phase field BCs
    config = {
        "physical_params": {
            "contact_angle": 60,
            "epsilon": 0.02
        },
        "boundary_conditions": {
            "phase_field": {
                "top": "neumann",
                "bottom": "contact_angle",
                "left": "neumann",
                "right": "neumann",
                "contact_angle_method": "robin"
            },
            "advection": {
                "top": "open",
                "bottom": "impermeable",
                "left": "open",
                "right": "open",
                "cout": 1.0
            }
        }
    }
    
    # Test standalone advection BC
    advection_bc = AdvectionBoundaryConditions(config)
    phi_advection = advection_bc.apply_boundary_conditions(phi, U, dt, dx, dy, use_jax=False)
    
    # Test standalone phase field BC
    phase_field_bc = PhaseFieldBoundaryConditions(config)
    phi_phase_field = phase_field_bc.apply_boundary_conditions(phi, dx, dy, use_jax=False)
    
    # Test combined (advection first, then phase field)
    phi_combined = advection_bc.apply_boundary_conditions(phi, U, dt, dx, dy, use_jax=False)
    phi_combined = phase_field_bc.apply_boundary_conditions(phi_combined, dx, dy, use_jax=False)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original
    im0 = axes[0, 0].imshow(phi.T, extent=[0, 1, 0, 1], origin='lower', 
                           cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 0].set_title('Original')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)
    
    # Advection BC only
    im1 = axes[0, 1].imshow(phi_advection.T, extent=[0, 1, 0, 1], origin='lower', 
                           cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 1].set_title('Advection BC Only')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    
    # Phase field BC only
    im2 = axes[1, 0].imshow(phi_phase_field.T, extent=[0, 1, 0, 1], origin='lower', 
                           cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 0].set_title('Phase Field BC Only')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)
    
    # Combined
    im3 = axes[1, 1].imshow(phi_combined.T, extent=[0, 1, 0, 1], origin='lower', 
                           cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 1].set_title('Combined (Advection + Phase Field)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 1], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('integration_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Integration test completed!")
    print("Results saved to 'integration_test.png'")


if __name__ == "__main__":
    print("Advection Boundary Condition Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Basic advection boundary conditions
        test_advection_boundary_conditions()
        
        # Test 2: Impermeable vs open comparison
        test_impermeable_vs_open()
        
        # Test 3: Integration with phase field BCs
        test_integration_with_phase_field_bc()
        
        print("\n" + "=" * 60)
        print("All advection BC tests completed successfully!")
        print("Check the generated PNG files for results.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
