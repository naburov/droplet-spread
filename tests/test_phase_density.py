#!/usr/bin/env python3
"""
Test phase field and density calculation to debug the inversion issue.
"""

import os
import sys
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simulation.initial_conditions import initialize_phase
from physics.properties import calculate_density
from config.config_loader import load_config


def test_phase_density():
    """Test phase field and density calculation."""
    print("=" * 60)
    print("PHASE FIELD AND DENSITY TEST")
    print("=" * 60)
    
    # Load config
    config = load_config("configs/config_water_droplet.json")
    rho1 = config["physical_params"]["rho1"]  # Air
    rho2 = config["physical_params"]["rho2"]  # Water
    
    print(f"Config: rho1={rho1} (air), rho2={rho2} (water)")
    
    # Create phase field
    Nx, Ny = 32, 32
    radius = 0.2
    phi = initialize_phase(Nx, Ny, radius)
    phi = jnp.array(phi)
    
    print(f"\nPhase field analysis:")
    print(f"  Shape: {phi.shape}")
    print(f"  Range: {phi.min():.6f} / {phi.max():.6f}")
    print(f"  Phi < 0 (droplet) pixels: {jnp.sum(phi < 0)}")
    print(f"  Phi > 0 (air) pixels: {jnp.sum(phi > 0)}")
    
    # Calculate density
    rho = calculate_density(phi, rho1, rho2)
    
    print(f"\nDensity analysis:")
    print(f"  Range: {rho.min():.6f} / {rho.max():.6f}")
    print(f"  Mean in droplet (phi < 0): {jnp.mean(rho[phi < 0]):.6f}")
    print(f"  Mean in air (phi > 0): {jnp.mean(rho[phi > 0]):.6f}")
    
    # Check specific values
    print(f"\nSpecific values:")
    print(f"  Center of domain (should be droplet): phi={phi[16, 16]:.6f}, rho={rho[16, 16]:.6f}")
    print(f"  Corner of domain (should be air): phi={phi[0, 31]:.6f}, rho={rho[0, 31]:.6f}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Phase field
    im1 = ax1.imshow(phi.T, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r')
    ax1.set_title('Phase Field (phi)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1)
    
    # Density field
    im2 = ax2.imshow(rho.T, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    ax2.set_title('Density (rho)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('phase_density_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved as 'phase_density_test.png'")
    
    # Check if the mapping is correct
    print(f"\nVerification:")
    if jnp.mean(rho[phi < 0]) > jnp.mean(rho[phi > 0]):
        print("✓ Droplet is denser than air (correct)")
    else:
        print("✗ Air is denser than droplet (wrong!)")
    
    if jnp.mean(rho[phi < 0]) > 500:  # Should be close to 1000
        print("✓ Droplet density is high (correct)")
    else:
        print("✗ Droplet density is low (wrong!)")
    
    if jnp.mean(rho[phi > 0]) < 100:  # Should be close to 1
        print("✓ Air density is low (correct)")
    else:
        print("✗ Air density is high (wrong!)")


if __name__ == "__main__":
    test_phase_density()
