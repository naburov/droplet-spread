#!/usr/bin/env python3
"""
Test script for configurable contact angle boundary conditions.

This script demonstrates how to use different contact angle methods
selected from configuration files.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from boundary_conditions.contact_angle_bc import ContactAngleBoundaryCondition
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


def test_contact_angle_methods():
    """Test different contact angle methods."""
    print("Testing Contact Angle Boundary Conditions")
    print("=" * 50)
    
    # Create test phase field
    phi, x, y = create_test_phase_field()
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Test different contact angle methods
    methods = ["simple", "robin", "young_laplace"]
    contact_angles = [30, 60, 90, 120]
    
    fig, axes = plt.subplots(len(methods), len(contact_angles), 
                            figsize=(4*len(contact_angles), 3*len(methods)))
    
    for i, method in enumerate(methods):
        for j, angle in enumerate(contact_angles):
            # Create contact angle BC
            if method == "robin":
                bc = ContactAngleBoundaryCondition(
                    contact_angle=angle, 
                    method=method, 
                    epsilon=0.02
                )
            else:
                bc = ContactAngleBoundaryCondition(
                    contact_angle=angle, 
                    method=method
                )
            
            # Apply boundary condition
            phi_bc = bc.apply(phi, dx, dy, use_jax=False)
            
            # Plot
            ax = axes[i, j]
            im = ax.imshow(phi_bc.T, extent=[0, 1, 0, 1], origin='lower', 
                          cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title(f'{method}\nθ = {angle}°')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('contact_angle_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Contact angle test completed!")
    print("Results saved to 'contact_angle_test.png'")


def test_configurable_bcs():
    """Test configurable boundary conditions from config."""
    print("\nTesting Configurable Boundary Conditions")
    print("=" * 50)
    
    # Test configuration
    config = {
        "physical_params": {
            "contact_angle": 75,
            "epsilon": 0.02
        },
        "boundary_conditions": {
            "phase_field": {
                "top": "neumann",
                "bottom": "contact_angle",
                "left": "periodic",
                "right": "periodic",
                "contact_angle_method": "robin"
            }
        }
    }
    
    # Create phase field BC manager
    bc_manager = PhaseFieldBoundaryConditions(config)
    
    # Print boundary condition info
    info = bc_manager.get_boundary_info()
    print("Boundary Condition Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with different contact angles
    phi, x, y = create_test_phase_field()
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    angles = [30, 60, 90, 120]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for i, angle in enumerate(angles):
        # Update contact angle
        bc_manager.update_contact_angle(angle)
        
        # Apply boundary conditions
        phi_bc = bc_manager.apply_boundary_conditions(phi, dx, dy, use_jax=False)
        
        # Plot
        ax = axes[i]
        im = ax.imshow(phi_bc.T, extent=[0, 1, 0, 1], origin='lower', 
                      cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'Configurable BC\nθ = {angle}° ({bc_manager.contact_angle_bc.method})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('configurable_bc_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Configurable BC test completed!")
    print("Results saved to 'configurable_bc_test.png'")


def test_robin_vs_simple():
    """Compare Robin vs Simple contact angle methods."""
    print("\nComparing Robin vs Simple Contact Angle Methods")
    print("=" * 50)
    
    phi, x, y = create_test_phase_field()
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Test both methods
    simple_bc = ContactAngleBoundaryCondition(contact_angle=60, method="simple")
    robin_bc = ContactAngleBoundaryCondition(contact_angle=60, method="robin", epsilon=0.02)
    
    phi_simple = simple_bc.apply(phi, dx, dy, use_jax=False)
    phi_robin = robin_bc.apply(phi, dx, dy, use_jax=False)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    im0 = axes[0].imshow(phi.T, extent=[0, 1, 0, 1], origin='lower', 
                        cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('Original Phase Field')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    
    # Simple method
    im1 = axes[1].imshow(phi_simple.T, extent=[0, 1, 0, 1], origin='lower', 
                        cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Simple Contact Angle (θ=60°)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    
    # Robin method
    im2 = axes[2].imshow(phi_robin.T, extent=[0, 1, 0, 1], origin='lower', 
                        cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_title('Robin Contact Angle (θ=60°)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('robin_vs_simple_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Robin vs Simple comparison completed!")
    print("Results saved to 'robin_vs_simple_comparison.png'")


if __name__ == "__main__":
    print("Contact Angle Boundary Condition Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Different contact angle methods
        test_contact_angle_methods()
        
        # Test 2: Configurable boundary conditions
        test_configurable_bcs()
        
        # Test 3: Robin vs Simple comparison
        test_robin_vs_simple()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("Check the generated PNG files for results.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
