#!/usr/bin/env python3
"""
Divergence analysis for droplet spreading simulation.
Analyzes ∇·u field and validates projection efficacy.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from numerics.finite_differences import divergence


def analyze_divergence(experiment_dir):
    """Analyze divergence field and projection efficacy.
    
    Args:
        experiment_dir (str): Path to experiment directory
    """
    print(f"Analyzing divergence in: {experiment_dir}")
    
    # Load configuration
    config_path = os.path.join(experiment_dir, 'simulation_parameters.json')
    if not os.path.exists(config_path):
        print("No simulation_parameters.json found!")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dx = config['grid_params']['Lx'] / config['grid_params']['Nx']
    dy = config['grid_params']['Ly'] / config['grid_params']['Ny']
    
    print(f"Grid spacing: dx={dx:.6f}, dy={dy:.6f}")
    
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
    
    # Analyze divergence
    divergence_data = []
    
    for step_num, file_path in checkpoint_files:
        try:
            data = np.load(file_path)
            U = data['U']  # Velocity field (Nx, Ny, 2)
            
            # Calculate divergence
            div_u = divergence(U, dx, dy)
            
            # Calculate norms
            l2_norm = np.sqrt(np.mean(div_u**2))
            linf_norm = np.max(np.abs(div_u))
            
            # Calculate divergence magnitude
            div_magnitude = np.abs(div_u)
            max_div = np.max(div_magnitude)
            mean_div = np.mean(div_magnitude)
            
            divergence_data.append({
                'step': step_num,
                'l2_norm': l2_norm,
                'linf_norm': linf_norm,
                'max_div': max_div,
                'mean_div': mean_div,
                'div_field': div_u.copy()
            })
            
            if step_num % 100 == 0:
                print(f"Step {step_num}: L2={l2_norm:.2e}, L∞={linf_norm:.2e}, max={max_div:.2e}")
        
        except Exception as e:
            print(f"Error loading step {step_num}: {e}")
            continue
    
    if not divergence_data:
        print("No valid divergence data found!")
        return
    
    # Convert to arrays
    steps = np.array([d['step'] for d in divergence_data])
    l2_norms = np.array([d['l2_norm'] for d in divergence_data])
    linf_norms = np.array([d['linf_norm'] for d in divergence_data])
    max_divs = np.array([d['max_div'] for d in divergence_data])
    mean_divs = np.array([d['mean_div'] for d in divergence_data])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # L2 and L∞ norms
    axes[0, 0].semilogy(steps, l2_norms, 'b-', label='L₂ norm', linewidth=2)
    axes[0, 0].semilogy(steps, linf_norms, 'r-', label='L∞ norm', linewidth=2)
    
    # Add expected/reference lines
    axes[0, 0].axhline(y=1e-3, color='g', linestyle='--', alpha=0.7, label='Good (1e-3)')
    axes[0, 0].axhline(y=1e-2, color='orange', linestyle='--', alpha=0.7, label='Acceptable (1e-2)')
    axes[0, 0].axhline(y=1e-1, color='red', linestyle='--', alpha=0.7, label='Poor (1e-1)')
    
    axes[0, 0].set_title('Divergence Norms vs Time\n(Should decrease and stay < 1e-2)')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Divergence Norm')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Maximum and mean divergence
    axes[0, 1].semilogy(steps, max_divs, 'g-', label='Max |∇·u|', linewidth=2)
    axes[0, 1].semilogy(steps, mean_divs, 'm-', label='Mean |∇·u|', linewidth=2)
    
    # Add expected/reference lines
    axes[0, 1].axhline(y=1e-4, color='g', linestyle='--', alpha=0.7, label='Excellent (1e-4)')
    axes[0, 1].axhline(y=1e-3, color='orange', linestyle='--', alpha=0.7, label='Good (1e-3)')
    axes[0, 1].axhline(y=1e-2, color='red', linestyle='--', alpha=0.7, label='Poor (1e-2)')
    
    axes[0, 1].set_title('Divergence Magnitude vs Time\n(Should be < 1e-3 for good projection)')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Divergence Magnitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Divergence field at different times
    n_plots = min(4, len(divergence_data))
    plot_indices = np.linspace(0, len(divergence_data)-1, n_plots, dtype=int)
    
    for i, idx in enumerate(plot_indices):
        row = 1 + i // 2
        col = i % 2
        
        if row < 2 and col < 2:
            div_field = divergence_data[idx]['div_field']
            im = axes[row, col].imshow(div_field, cmap='RdBu_r', aspect='equal')
            axes[row, col].set_title(f'∇·u at step {divergence_data[idx]["step"]}')
            axes[row, col].set_xlabel('x')
            axes[row, col].set_ylabel('y')
            plt.colorbar(im, ax=axes[row, col])
    
    plt.tight_layout()
    
    # Create diagnostics directory
    diagnostics_dir = os.path.join(experiment_dir, 'diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    # Save plot
    output_path = os.path.join(diagnostics_dir, 'divergence_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Divergence analysis plot saved to: {output_path}")
    
    # Save data
    data_path = os.path.join(diagnostics_dir, 'divergence_data.json')
    with open(data_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_data = []
        for d in divergence_data:
            json_d = {}
            for k, v in d.items():
                if k == 'div_field':
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
    print(f"Divergence data saved to: {data_path}")
    
    # Analysis summary
    print("\n=== DIVERGENCE ANALYSIS SUMMARY ===")
    print(f"Final L₂ norm: {l2_norms[-1]:.2e}")
    print(f"Final L∞ norm: {linf_norms[-1]:.2e}")
    print(f"Final max divergence: {max_divs[-1]:.2e}")
    print(f"Final mean divergence: {mean_divs[-1]:.2e}")
    
    # Check if divergence is under control
    if linf_norms[-1] > 1e-3:
        print("⚠️  WARNING: High divergence detected! Projection may be failing.")
    else:
        print("✅ Divergence is under control.")


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python divergence_analysis.py <experiment_directory>")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    
    if not os.path.exists(experiment_dir):
        print(f"Error: Directory {experiment_dir} does not exist!")
        sys.exit(1)
    
    analyze_divergence(experiment_dir)


if __name__ == "__main__":
    main()
