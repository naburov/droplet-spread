#!/usr/bin/env python3
"""
Test with different radius values to find the correct size.
"""

import os
import sys
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simulation.initial_conditions import initialize_phase


def test_radius_values():
    """Test different radius values."""
    print("=" * 60)
    print("RADIUS VALUE TEST")
    print("=" * 60)
    
    Nx, Ny = 32, 32
    
    # Test different radius values
    radius_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    for radius in radius_values:
        phi = initialize_phase(Nx, Ny, radius)
        
        droplet_pixels = np.sum(phi < 0)
        air_pixels = np.sum(phi > 0)
        total_pixels = Nx * Ny
        
        print(f"Radius {radius:4.2f}: Droplet {droplet_pixels:3d} pixels ({droplet_pixels/total_pixels*100:5.1f}%), Air {air_pixels:3d} pixels ({air_pixels/total_pixels*100:5.1f}%)")
    
    # Test with a very small radius
    print(f"\nTesting with very small radius:")
    radius = 0.05
    phi = initialize_phase(Nx, Ny, radius)
    
    print(f"Radius {radius}: phi range {phi.min():.6f} / {phi.max():.6f}")
    
    # Check specific locations
    print(f"  Center (16,16): phi={phi[16, 16]:.6f}")
    print(f"  Corner (0,31): phi={phi[0, 31]:.6f}")
    print(f"  Bottom center (16,0): phi={phi[16, 0]:.6f}")
    
    # Calculate expected coverage
    expected_coverage = np.pi * radius**2 / 4  # Semicircle area
    actual_coverage = np.sum(phi < 0) / (Nx * Ny)
    
    print(f"  Expected coverage: {expected_coverage:.4f} ({expected_coverage*100:.1f}%)")
    print(f"  Actual coverage: {actual_coverage:.4f} ({actual_coverage*100:.1f}%)")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Phase field
    im1 = ax1.imshow(phi.T, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r')
    ax1.set_title(f'Phase Field (radius={radius})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1)
    
    # Distance field
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    center_x, center_y = 0.5, 0
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    im2 = ax2.imshow(distance.T, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    ax2.set_title('Distance from Center')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2)
    
    # Add circle overlay
    circle = plt.Circle((0.5, 0), radius, fill=False, color='red', linewidth=2)
    ax2.add_patch(circle)
    
    plt.tight_layout()
    plt.savefig('radius_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved as 'radius_test.png'")


if __name__ == "__main__":
    test_radius_values()
