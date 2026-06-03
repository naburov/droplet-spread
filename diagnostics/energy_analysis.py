#!/usr/bin/env python3
"""
Kinetic energy and velocity analysis for droplet spreading simulation.
Tracks ∫½ρ|u|² dΩ and max |u| to detect solver/BC issues.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from physics.properties import calculate_density


def analyze_energy(experiment_dir):
    """Analyze kinetic energy and velocity statistics.
    
    Args:
        experiment_dir (str): Path to experiment directory
    """
    print(f"Analyzing energy in: {experiment_dir}")
    
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
    
    # Analyze energy
    energy_data = []
    
    for step_num, file_path in checkpoint_files:
        try:
            data = np.load(file_path)
            phi = data['phi']  # Phase field
            U = data['U']      # Velocity field (Nx, Ny, 2)
            
            # Calculate density
            rho = calculate_density(phi, rho1, rho2)
            
            # Calculate velocity magnitude
            u_magnitude = np.sqrt(np.sum(U**2, axis=-1))
            max_velocity = np.max(u_magnitude)
            mean_velocity = np.mean(u_magnitude)
            
            # Calculate kinetic energy density
            kinetic_energy_density = 0.5 * rho * u_magnitude**2
            
            # Integrate kinetic energy over domain
            kinetic_energy = np.sum(kinetic_energy_density) * dx * dy
            
            # Calculate velocity statistics
            u_rms = np.sqrt(np.mean(u_magnitude**2))
            u_max = np.max(u_magnitude)
            
            # Calculate velocity components
            u_x = U[..., 0]
            u_y = U[..., 1]
            u_x_max = np.max(np.abs(u_x))
            u_y_max = np.max(np.abs(u_y))
            
            energy_data.append({
                'step': step_num,
                'kinetic_energy': kinetic_energy,
                'max_velocity': max_velocity,
                'mean_velocity': mean_velocity,
                'u_rms': u_rms,
                'u_x_max': u_x_max,
                'u_y_max': u_y_max,
                'kinetic_energy_density': kinetic_energy_density.copy()
            })
            
            if step_num % 100 == 0:
                print(f"Step {step_num}: KE={kinetic_energy:.2e}, max|u|={max_velocity:.2e}, u_rms={u_rms:.2e}")
        
        except Exception as e:
            print(f"Error loading step {step_num}: {e}")
            continue
    
    if not energy_data:
        print("No valid energy data found!")
        return
    
    # Convert to arrays
    steps = np.array([d['step'] for d in energy_data])
    kinetic_energies = np.array([d['kinetic_energy'] for d in energy_data])
    max_velocities = np.array([d['max_velocity'] for d in energy_data])
    mean_velocities = np.array([d['mean_velocity'] for d in energy_data])
    u_rms = np.array([d['u_rms'] for d in energy_data])
    u_x_max = np.array([d['u_x_max'] for d in energy_data])
    u_y_max = np.array([d['u_y_max'] for d in energy_data])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Kinetic energy evolution
    axes[0, 0].semilogy(steps, kinetic_energies, 'b-', linewidth=2, label='Kinetic Energy')
    
    # Add expected behavior lines
    # For a droplet spreading, KE should initially increase then decrease
    if len(steps) > 10:
        # Add trend line (polynomial fit)
        z = np.polyfit(steps, np.log10(kinetic_energies), 1)
        p = np.poly1d(z)
        trend_line = 10**p(steps)
        axes[0, 0].plot(steps, trend_line, 'r--', alpha=0.7, label=f'Trend (slope={z[0]:.3f})')
    
    axes[0, 0].set_title('Kinetic Energy vs Time\n(Should peak then decay for spreading)')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Kinetic Energy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Velocity statistics
    axes[0, 1].semilogy(steps, max_velocities, 'r-', label='Max |u|', linewidth=2)
    axes[0, 1].semilogy(steps, u_rms, 'g-', label='RMS |u|', linewidth=2)
    axes[0, 1].semilogy(steps, mean_velocities, 'm-', label='Mean |u|', linewidth=2)
    
    # Add reference lines for velocity scales
    axes[0, 1].axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='High speed (1.0)')
    axes[0, 1].axhline(y=0.1, color='g', linestyle='--', alpha=0.7, label='Moderate (0.1)')
    axes[0, 1].axhline(y=0.01, color='b', linestyle='--', alpha=0.7, label='Low (0.01)')
    
    axes[0, 1].set_title('Velocity Statistics vs Time\n(Should decrease for spreading)')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Velocity Magnitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Velocity components
    axes[1, 0].semilogy(steps, u_x_max, 'b-', label='Max |u_x|', linewidth=2)
    axes[1, 0].semilogy(steps, u_y_max, 'r-', label='Max |u_y|', linewidth=2)
    axes[1, 0].set_title('Velocity Components vs Time')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Max Velocity Component')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Kinetic energy density at different times
    n_plots = min(4, len(energy_data))
    plot_indices = np.linspace(0, len(energy_data)-1, n_plots, dtype=int)
    
    for i, idx in enumerate(plot_indices):
        row = 1 + i // 2
        col = i % 2
        
        if row < 2 and col < 2:
            ke_density = energy_data[idx]['kinetic_energy_density']
            im = axes[row, col].imshow(ke_density, cmap='viridis', aspect='equal')
            axes[row, col].set_title(f'KE Density at step {energy_data[idx]["step"]}')
            axes[row, col].set_xlabel('x')
            axes[row, col].set_ylabel('y')
            plt.colorbar(im, ax=axes[row, col])
    
    plt.tight_layout()
    
    # Create diagnostics directory
    diagnostics_dir = os.path.join(experiment_dir, 'diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    # Save plot
    output_path = os.path.join(diagnostics_dir, 'energy_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Energy analysis plot saved to: {output_path}")
    
    # Save data
    data_path = os.path.join(diagnostics_dir, 'energy_data.json')
    with open(data_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_data = []
        for d in energy_data:
            json_d = {}
            for k, v in d.items():
                if k == 'kinetic_energy_density':
                    continue  # Skip large arrays
                elif isinstance(v, np.ndarray):
                    json_d[k] = v.tolist()
                elif isinstance(v, (np.int64, np.int32)):
                    json_d[k] = int(v)
                elif isinstance(v, (np.float64, np.float32)):
                    json_d[k] = float(v)
                else:
                    json_d[k] = v
            json_data.append(json_d)
        json.dump(json_data, f, indent=2)
    print(f"Energy data saved to: {data_path}")
    
    # Analysis summary
    print("\n=== ENERGY ANALYSIS SUMMARY ===")
    print(f"Final kinetic energy: {kinetic_energies[-1]:.2e}")
    print(f"Final max velocity: {max_velocities[-1]:.2e}")
    print(f"Final RMS velocity: {u_rms[-1]:.2e}")
    print(f"Final mean velocity: {mean_velocities[-1]:.2e}")
    
    # Check for spikes
    ke_change = np.diff(kinetic_energies)
    max_ke_spike = np.max(np.abs(ke_change))
    if max_ke_spike > 0.1 * np.mean(kinetic_energies):
        print(f"⚠️  WARNING: Large kinetic energy spike detected: {max_ke_spike:.2e}")
    else:
        print("✅ Kinetic energy evolution is smooth.")
    
    # Check velocity bounds
    if max_velocities[-1] > 10.0:
        print("⚠️  WARNING: Very high velocities detected! Check for numerical instability.")
    else:
        print("✅ Velocities are within reasonable bounds.")


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python energy_analysis.py <experiment_directory>")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    
    if not os.path.exists(experiment_dir):
        print(f"Error: Directory {experiment_dir} does not exist!")
        sys.exit(1)
    
    analyze_energy(experiment_dir)


if __name__ == "__main__":
    main()
