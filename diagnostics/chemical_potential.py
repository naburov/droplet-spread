#!/usr/bin/env python3
"""
Chemical potential diagnostics for droplet spreading simulation.
Analyzes μ, Δφ, and |∇φ| near the interface to detect stiffness/penalty artifacts.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from numerics.finite_differences import gradient, laplacian
from physics.properties import df_2


def analyze_chemical_potential(experiment_dir):
    """Analyze chemical potential and related quantities.
    
    Args:
        experiment_dir (str): Path to experiment directory
    """
    print(f"Analyzing chemical potential in: {experiment_dir}")
    
    # Load configuration
    config_path = os.path.join(experiment_dir, 'simulation_parameters.json')
    if not os.path.exists(config_path):
        print("No simulation_parameters.json found!")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dx = config['grid_params']['Lx'] / config['grid_params']['Nx']
    dy = config['grid_params']['Ly'] / config['grid_params']['Ny']
    epsilon = config['physical_params']['epsilon']
    
    print(f"Grid spacing: dx={dx:.6f}, dy={dy:.6f}")
    print(f"Interface thickness: epsilon={epsilon}")
    
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
    
    # Analyze chemical potential
    chem_pot_data = []
    
    for step_num, file_path in checkpoint_files:
        try:
            data = np.load(file_path)
            phi = data['phi']  # Phase field
            
            # Calculate gradient and Laplacian
            grad_phi = gradient(phi, dx, dy)
            grad_magnitude = np.sqrt(np.sum(grad_phi**2, axis=-1))
            lap_phi = laplacian(phi, dx, dy)
            
            # Calculate chemical potential
            mu = df_2(phi) - epsilon**2 * lap_phi
            
            # Calculate interface region (where |phi| < 0.5)
            interface_mask = np.abs(phi) < 0.5
            
            # Statistics for interface region
            if np.any(interface_mask):
                mu_interface = mu[interface_mask]
                grad_mag_interface = grad_magnitude[interface_mask]
                lap_phi_interface = lap_phi[interface_mask]
                
                mu_mean = np.mean(mu_interface)
                mu_std = np.std(mu_interface)
                mu_min = np.min(mu_interface)
                mu_max = np.max(mu_interface)
                
                grad_mean = np.mean(grad_mag_interface)
                grad_std = np.std(grad_mag_interface)
                grad_max = np.max(grad_mag_interface)
                
                lap_mean = np.mean(lap_phi_interface)
                lap_std = np.std(lap_phi_interface)
                lap_min = np.min(lap_phi_interface)
                lap_max = np.max(lap_phi_interface)
            else:
                mu_mean = mu_std = mu_min = mu_max = 0.0
                grad_mean = grad_std = grad_max = 0.0
                lap_mean = lap_std = lap_min = lap_max = 0.0
            
            # Global statistics
            mu_global_mean = np.mean(mu)
            mu_global_std = np.std(mu)
            mu_global_min = np.min(mu)
            mu_global_max = np.max(mu)
            
            chem_pot_data.append({
                'step': step_num,
                'mu_interface_mean': mu_mean,
                'mu_interface_std': mu_std,
                'mu_interface_min': mu_min,
                'mu_interface_max': mu_max,
                'mu_global_mean': mu_global_mean,
                'mu_global_std': mu_global_std,
                'mu_global_min': mu_global_min,
                'mu_global_max': mu_global_max,
                'grad_mean': grad_mean,
                'grad_std': grad_std,
                'grad_max': grad_max,
                'lap_mean': lap_mean,
                'lap_std': lap_std,
                'lap_min': lap_min,
                'lap_max': lap_max,
                'mu_field': mu.copy(),
                'grad_magnitude': grad_magnitude.copy(),
                'laplacian': lap_phi.copy()
            })
            
            if step_num % 100 == 0:
                print(f"Step {step_num}: μ_mean={mu_mean:.2e}, μ_std={mu_std:.2e}, grad_max={grad_max:.2e}")
        
        except Exception as e:
            print(f"Error loading step {step_num}: {e}")
            continue
    
    if not chem_pot_data:
        print("No valid chemical potential data found!")
        return
    
    # Convert to arrays
    steps = np.array([d['step'] for d in chem_pot_data])
    mu_interface_means = np.array([d['mu_interface_mean'] for d in chem_pot_data])
    mu_interface_stds = np.array([d['mu_interface_std'] for d in chem_pot_data])
    mu_interface_mins = np.array([d['mu_interface_min'] for d in chem_pot_data])
    mu_interface_maxs = np.array([d['mu_interface_max'] for d in chem_pot_data])
    grad_means = np.array([d['grad_mean'] for d in chem_pot_data])
    grad_stds = np.array([d['grad_std'] for d in chem_pot_data])
    grad_maxs = np.array([d['grad_max'] for d in chem_pot_data])
    lap_means = np.array([d['lap_mean'] for d in chem_pot_data])
    lap_stds = np.array([d['lap_std'] for d in chem_pot_data])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Chemical potential statistics
    axes[0, 0].plot(steps, mu_interface_means, 'b-', linewidth=2, label='Mean μ')
    axes[0, 0].fill_between(steps, mu_interface_means - mu_interface_stds, 
                           mu_interface_means + mu_interface_stds, alpha=0.3)
    axes[0, 0].plot(steps, mu_interface_mins, 'r--', linewidth=1, label='Min μ')
    axes[0, 0].plot(steps, mu_interface_maxs, 'g--', linewidth=1, label='Max μ')
    
    # Add reference lines for chemical potential
    axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.5, label='Equilibrium (μ=0)')
    axes[0, 0].axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='High (μ=1)')
    axes[0, 0].axhline(y=-1.0, color='orange', linestyle='--', alpha=0.7, label='Low (μ=-1)')
    
    axes[0, 0].set_title('Chemical Potential Statistics (Interface Region)\n(Should be near 0 for equilibrium)')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Chemical Potential μ')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gradient magnitude statistics
    axes[0, 1].semilogy(steps, grad_means, 'b-', linewidth=2, label='Mean |∇φ|')
    axes[0, 1].fill_between(steps, grad_means - grad_stds, 
                           grad_means + grad_stds, alpha=0.3)
    axes[0, 1].semilogy(steps, grad_maxs, 'r-', linewidth=2, label='Max |∇φ|')
    axes[0, 1].set_title('Gradient Magnitude Statistics')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('|∇φ|')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Laplacian statistics
    axes[1, 0].plot(steps, lap_means, 'b-', linewidth=2, label='Mean Δφ')
    axes[1, 0].fill_between(steps, lap_means - lap_stds, 
                           lap_means + lap_stds, alpha=0.3)
    axes[1, 0].set_title('Laplacian Statistics')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Δφ')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Chemical potential field at different times
    n_plots = min(4, len(chem_pot_data))
    plot_indices = np.linspace(0, len(chem_pot_data)-1, n_plots, dtype=int)
    
    for i, idx in enumerate(plot_indices):
        row = 1 + i // 2
        col = i % 2
        
        if row < 2 and col < 2:
            mu_field = chem_pot_data[idx]['mu_field']
            im = axes[row, col].imshow(mu_field, cmap='RdBu_r', aspect='equal')
            axes[row, col].set_title(f'μ at step {chem_pot_data[idx]["step"]}')
            axes[row, col].set_xlabel('x')
            axes[row, col].set_ylabel('y')
            plt.colorbar(im, ax=axes[row, col])
    
    plt.tight_layout()
    
    # Create diagnostics directory
    diagnostics_dir = os.path.join(experiment_dir, 'diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    # Save plot
    output_path = os.path.join(diagnostics_dir, 'chemical_potential_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Chemical potential analysis plot saved to: {output_path}")
    
    # Save data
    data_path = os.path.join(diagnostics_dir, 'chemical_potential_data.json')
    with open(data_path, 'w') as f:
        json_data = []
        for d in chem_pot_data:
            json_d = {}
            for k, v in d.items():
                if k in ['mu_field', 'grad_magnitude', 'laplacian']:
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
    print(f"Chemical potential data saved to: {data_path}")
    
    # Analysis summary
    print("\n=== CHEMICAL POTENTIAL ANALYSIS SUMMARY ===")
    print(f"Final mean μ (interface): {mu_interface_means[-1]:.2e}")
    print(f"Final std μ (interface): {mu_interface_stds[-1]:.2e}")
    print(f"Final max |∇φ|: {grad_maxs[-1]:.2e}")
    print(f"Final mean |∇φ|: {grad_means[-1]:.2e}")
    
    # Check for stiffness issues
    if mu_interface_stds[-1] > 1.0:
        print("⚠️  WARNING: High chemical potential variance detected! Possible stiffness issues.")
    else:
        print("✅ Chemical potential is well-behaved.")
    
    # Check for over-sharpening
    if grad_maxs[-1] > 100.0:
        print("⚠️  WARNING: Very high gradients detected! Interface may be over-sharpened.")
    else:
        print("✅ Gradients are within reasonable bounds.")


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python chemical_potential.py <experiment_directory>")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    
    if not os.path.exists(experiment_dir):
        print(f"Error: Directory {experiment_dir} does not exist!")
        sys.exit(1)
    
    analyze_chemical_potential(experiment_dir)


if __name__ == "__main__":
    main()
