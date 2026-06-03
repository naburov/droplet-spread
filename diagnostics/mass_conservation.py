#!/usr/bin/env python3
"""
Mass and phase conservation analysis for droplet spreading simulation.
Tracks ∫φ dΩ and droplet volume to detect advection or BC leakage.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def analyze_mass_conservation(experiment_dir):
    """Analyze mass and phase conservation.
    
    Args:
        experiment_dir (str): Path to experiment directory
    """
    print(f"Analyzing mass conservation in: {experiment_dir}")
    
    # Load configuration
    config_path = os.path.join(experiment_dir, 'simulation_parameters.json')
    if not os.path.exists(config_path):
        print("No simulation_parameters.json found!")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dx = config['grid_params']['Lx'] / config['grid_params']['Nx']
    dy = config['grid_params']['Ly'] / config['grid_params']['Ny']
    rho1 = config['physical_params']['rho1']
    rho2 = config['physical_params']['rho2']
    
    print(f"Grid spacing: dx={dx:.6f}, dy={dy:.6f}")
    print(f"Densities: rho1={rho1}, rho2={rho2}")
    
    # Find checkpoint files
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        print("No checkpoints directory found!")
        return
    
    checkpoint_files = []
    for file in os.listdir(checkpoints_dir):
        if file.startswith('checkpoint_') and file.endswith('.npz'):
            step_num = int(file.split('_')[1].split('.')[0])
            checkpoint_files.append((step_num, os.path.join(checkpoints_dir, file)))
    
    checkpoint_files.sort(key=lambda x: x[0])
    
    if not checkpoint_files:
        print("No checkpoint files found!")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Analyze mass conservation
    mass_data = []
    
    for step_num, file_path in checkpoint_files:
        try:
            data = np.load(file_path)
            phi = data['phi']  # Phase field
            U = data['U']      # Velocity field (Nx, Ny, 2)
            
            # Calculate phase field integral (mass)
            phase_integral = np.sum(phi) * dx * dy
            
            # Calculate droplet volume (where phi < 0)
            droplet_mask = phi < 0
            droplet_volume = np.sum(droplet_mask) * dx * dy
            
            # Calculate droplet area fraction
            total_area = phi.shape[0] * phi.shape[1] * dx * dy
            droplet_area_fraction = droplet_volume / total_area
            
            # Calculate phase field statistics
            phi_min = np.min(phi)
            phi_max = np.max(phi)
            phi_mean = np.mean(phi)
            phi_std = np.std(phi)
            
            # Calculate mass density
            rho = rho2 * (1 - (phi + 1) / 2) + rho1 * (phi + 1) / 2
            mass_density = rho * dx * dy
            total_mass = np.sum(mass_density)
            
            # Calculate droplet mass (where phi < 0)
            droplet_mass = np.sum(mass_density[droplet_mask])
            
            # Calculate interface length (approximate)
            from numerics.finite_differences import gradient
            grad_phi = gradient(phi, dx, dy)
            grad_magnitude = np.sqrt(np.sum(grad_phi**2, axis=-1))
            interface_length = np.sum(grad_magnitude) * dx * dy
            
            mass_data.append({
                'step': step_num,
                'phase_integral': phase_integral,
                'droplet_volume': droplet_volume,
                'droplet_area_fraction': droplet_area_fraction,
                'total_mass': total_mass,
                'droplet_mass': droplet_mass,
                'phi_min': phi_min,
                'phi_max': phi_max,
                'phi_mean': phi_mean,
                'phi_std': phi_std,
                'interface_length': interface_length
            })
            
            if step_num % 100 == 0:
                print(f"Step {step_num}: phase_int={phase_integral:.6f}, droplet_vol={droplet_volume:.6f}, mass={total_mass:.6f}")
        
        except Exception as e:
            print(f"Error loading step {step_num}: {e}")
            continue
    
    if not mass_data:
        print("No valid mass data found!")
        return
    
    # Convert to arrays
    steps = np.array([d['step'] for d in mass_data])
    phase_integrals = np.array([d['phase_integral'] for d in mass_data])
    droplet_volumes = np.array([d['droplet_volume'] for d in mass_data])
    droplet_area_fractions = np.array([d['droplet_area_fraction'] for d in mass_data])
    total_masses = np.array([d['total_mass'] for d in mass_data])
    droplet_masses = np.array([d['droplet_mass'] for d in mass_data])
    phi_mins = np.array([d['phi_min'] for d in mass_data])
    phi_maxs = np.array([d['phi_max'] for d in mass_data])
    phi_means = np.array([d['phi_mean'] for d in mass_data])
    phi_stds = np.array([d['phi_std'] for d in mass_data])
    interface_lengths = np.array([d['interface_length'] for d in mass_data])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Phase field integral (mass conservation)
    axes[0, 0].plot(steps, phase_integrals, 'b-', linewidth=2, label='∫φ dΩ')
    axes[0, 0].axhline(y=phase_integrals[0], color='r', linestyle='--', alpha=0.7, label='Initial value')
    
    # Add tolerance bands
    initial_value = phase_integrals[0]
    axes[0, 0].axhspan(initial_value * 0.99, initial_value * 1.01, alpha=0.2, color='g', label='±1% tolerance')
    axes[0, 0].axhspan(initial_value * 0.95, initial_value * 1.05, alpha=0.1, color='orange', label='±5% tolerance')
    
    axes[0, 0].set_title('Phase Field Integral (Mass Conservation)\n(Should be constant - perfect conservation)')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('∫φ dΩ')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Droplet volume evolution
    axes[0, 1].plot(steps, droplet_volumes, 'g-', linewidth=2, label='Droplet Volume')
    axes[0, 1].plot(steps, droplet_area_fractions, 'm-', linewidth=2, label='Area Fraction')
    
    # Add initial value reference
    axes[0, 1].axhline(y=droplet_volumes[0], color='g', linestyle='--', alpha=0.7, label='Initial volume')
    
    # Add expected behavior for spreading (volume should be conserved)
    axes[0, 1].axhspan(droplet_volumes[0] * 0.95, droplet_volumes[0] * 1.05, alpha=0.2, color='g', label='±5% volume conservation')
    
    axes[0, 1].set_title('Droplet Volume Evolution\n(Should be constant - volume conservation)')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Volume / Area Fraction')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mass evolution
    axes[1, 0].plot(steps, total_masses, 'b-', linewidth=2, label='Total Mass')
    axes[1, 0].plot(steps, droplet_masses, 'r-', linewidth=2, label='Droplet Mass')
    axes[1, 0].set_title('Mass Evolution')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Mass')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Phase field statistics
    axes[1, 1].plot(steps, phi_mins, 'b-', linewidth=2, label='φ_min')
    axes[1, 1].plot(steps, phi_maxs, 'r-', linewidth=2, label='φ_max')
    axes[1, 1].plot(steps, phi_means, 'g-', linewidth=2, label='φ_mean')
    axes[1, 1].plot(steps, phi_stds, 'm-', linewidth=2, label='φ_std')
    axes[1, 1].set_title('Phase Field Statistics')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Phase Field Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create diagnostics directory
    diagnostics_dir = os.path.join(experiment_dir, 'diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    # Save plot
    output_path = os.path.join(diagnostics_dir, 'mass_conservation_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Mass conservation analysis plot saved to: {output_path}")
    
    # Save data
    data_path = os.path.join(diagnostics_dir, 'mass_conservation_data.json')
    with open(data_path, 'w') as f:
        json_data = []
        for d in mass_data:
            json_d = {k: v for k, v in d.items()}
            json_d['step'] = int(json_d['step'])
            json_data.append(json_d)
        json.dump(json_data, f, indent=2)
    print(f"Mass conservation data saved to: {data_path}")
    
    # Analysis summary
    print("\n=== MASS CONSERVATION ANALYSIS SUMMARY ===")
    
    # Phase field conservation
    phase_change = (phase_integrals[-1] - phase_integrals[0]) / phase_integrals[0] if phase_integrals[0] != 0 else 0
    print(f"Phase field change: {phase_change:.6f} ({phase_change*100:.2f}%)")
    if abs(phase_change) > 0.01:
        print("⚠️  WARNING: Significant phase field drift detected!")
    else:
        print("✅ Phase field is well conserved.")
    
    # Droplet volume evolution
    volume_change = (droplet_volumes[-1] - droplet_volumes[0]) / droplet_volumes[0] if droplet_volumes[0] != 0 else 0
    print(f"Droplet volume change: {volume_change:.6f} ({volume_change*100:.2f}%)")
    if abs(volume_change) > 0.05:
        print("⚠️  WARNING: Significant droplet volume change detected!")
    else:
        print("✅ Droplet volume is stable.")
    
    # Mass conservation
    mass_change = (total_masses[-1] - total_masses[0]) / total_masses[0] if total_masses[0] != 0 else 0
    print(f"Total mass change: {mass_change:.6f} ({mass_change*100:.2f}%)")
    if abs(mass_change) > 0.01:
        print("⚠️  WARNING: Mass conservation violated!")
    else:
        print("✅ Mass is well conserved.")
    
    # Phase field bounds
    if phi_mins[-1] < -1.1 or phi_maxs[-1] > 1.1:
        print("⚠️  WARNING: Phase field values outside [-1, 1] bounds!")
    else:
        print("✅ Phase field values are within bounds.")


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python mass_conservation.py <experiment_directory>")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    
    if not os.path.exists(experiment_dir):
        print(f"Error: Directory {experiment_dir} does not exist!")
        sys.exit(1)
    
    analyze_mass_conservation(experiment_dir)


if __name__ == "__main__":
    main()
