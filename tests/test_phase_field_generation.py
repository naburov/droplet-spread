#!/usr/bin/env python3
"""
Test phase field generation to understand the droplet positioning.
"""

import os
import sys
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simulation.initial_conditions import initialize_phase


def test_phase_field_generation():
    """Test phase field generation."""
    print("=" * 60)
    print("PHASE FIELD GENERATION TEST")
    print("=" * 60)
    
    # Create phase field
    Nx, Ny = 32, 32
    radius = 0.2
    phi = initialize_phase(Nx, Ny, radius)
    
    print(f"Grid: {Nx}x{Ny}")
    print(f"Radius: {radius}")
    print(f"Phase field range: {phi.min():.6f} / {phi.max():.6f}")
    
    # Check specific locations
    print(f"\nSpecific locations:")
    print(f"  Center (16,16): phi={phi[16, 16]:.6f}")
    print(f"  Corner (0,31): phi={phi[0, 31]:.6f}")
    print(f"  Corner (31,31): phi={phi[31, 31]:.6f}")
    print(f"  Bottom center (16,0): phi={phi[16, 0]:.6f}")
    print(f"  Top center (16,31): phi={phi[16, 31]:.6f}")
    
    # Check how many pixels are droplet vs air
    droplet_pixels = np.sum(phi < 0)
    air_pixels = np.sum(phi > 0)
    total_pixels = Nx * Ny
    
    print(f"\nPixel counts:")
    print(f"  Droplet pixels (phi < 0): {droplet_pixels} ({droplet_pixels/total_pixels*100:.1f}%)")
    print(f"  Air pixels (phi > 0): {air_pixels} ({air_pixels/total_pixels*100:.1f}%)")
    
    # Check the distance calculation
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    center_x = 0.5
    center_y = 0
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    print(f"\nDistance analysis:")
    print(f"  Distance range: {distance.min():.6f} / {distance.max():.6f}")
    print(f"  Distance at center: {distance[16, 16]:.6f}")
    print(f"  Distance at corner: {distance[0, 31]:.6f}")
    print(f"  Distance at top center: {distance[16, 31]:.6f}")
    
    # Check the tanh function
    scale_factor = 5.0  # Use the same scale factor as in the function
    tanh_input = (distance - radius) * scale_factor
    tanh_output = -np.tanh(tanh_input)
    
    print(f"\nTanh analysis:")
    print(f"  Scale factor: {scale_factor}")
    print(f"  Tanh input range: {tanh_input.min():.6f} / {tanh_input.max():.6f}")
    print(f"  Tanh output range: {tanh_output.min():.6f} / {tanh_output.max():.6f}")
    print(f"  Tanh at center: {tanh_output[16, 16]:.6f}")
    print(f"  Tanh at corner: {tanh_output[0, 31]:.6f}")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Distance field
    im1 = ax1.imshow(distance.T, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    ax1.set_title('Distance from Center')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1)
    
    # Phase field
    im2 = ax2.imshow(phi.T, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r')
    ax2.set_title('Phase Field (phi)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2)
    
    # Tanh input
    im3 = ax3.imshow(tanh_input.T, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    ax3.set_title('Tanh Input')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(im3, ax=ax3)
    
    # Tanh output
    im4 = ax4.imshow(tanh_output.T, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r')
    ax4.set_title('Tanh Output')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('phase_field_generation_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved as 'phase_field_generation_test.png'")


if __name__ == "__main__":
    test_phase_field_generation()
