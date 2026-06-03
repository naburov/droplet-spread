#!/usr/bin/env python3
"""
Force budget analysis for droplet spreading simulation.
Analyzes |−∇p|, |μ∇φ|, viscous term magnitude, and gravity along a vertical line through the droplet apex.
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
from physics.properties import calculate_density, calculate_reynolds_number


def analyze_force_budget(experiment_dir):
    """Analyze force budget terms.
    
    Args:
        experiment_dir (str): Path to experiment directory
    """
    print(f"Analyzing force budget in: {experiment_dir}")
    
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
    Re1 = config['physical_params']['Re1']
    Re2 = config['physical_params']['Re2']
    We1 = config['physical_params']['We1']
    We2 = config['physical_params']['We2']
    g = config['physical_params']['g']
    Fr = config['physical_params']['Fr']
    epsilon = config['physical_params']['epsilon']
    
    print(f"Grid spacing: dx={dx:.6f}, dy={dy:.6f}")
    print(f"Physical parameters: g={g}, Fr={Fr}, epsilon={epsilon}")
    
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
    
    # Analyze force budget
    force_data = []
    
    for step_num, file_path in checkpoint_files:
        try:
            data = np.load(file_path)
            phi = data['phi']      # Phase field
            U = data['U']          # Velocity field
            pressure = data['P']   # Pressure field
            
            # Calculate density and Reynolds number
            rho = calculate_density(phi, rho1, rho2)
            Re = calculate_reynolds_number(phi, Re1, Re2)
            
            # Calculate pressure gradient
            grad_p = gradient(pressure, dx, dy)
            grad_p_magnitude = np.sqrt(np.sum(grad_p**2, axis=-1))
            
            # Calculate surface tension force
            grad_phi = gradient(phi, dx, dy)
            grad_phi_magnitude = np.sqrt(np.sum(grad_phi**2, axis=-1))
            
            # Calculate curvature
            lap_phi = laplacian(phi, dx, dy)
            curvature = lap_phi / (grad_phi_magnitude + 1e-10)
            
            # Surface tension force magnitude
            surface_tension_force = 3 * np.sqrt(2) * epsilon / (4 * We1) * curvature * grad_phi_magnitude
            
            # Calculate viscous force
            viscous_force = np.zeros_like(U)
            viscous_force[..., 0] = laplacian(U[..., 0], dx, dy) / Re
            viscous_force[..., 1] = laplacian(U[..., 1], dx, dy) / Re
            viscous_force_magnitude = np.sqrt(np.sum(viscous_force**2, axis=-1))
            
            # Gravity force
            gravity_force = rho * g / Fr
            gravity_magnitude = np.abs(gravity_force)
            
            # Find droplet center (where phi is most negative)
            droplet_center_y, droplet_center_x = np.unravel_index(np.argmin(phi), phi.shape)
            
            # Sample forces along vertical line through droplet center
            x_line = droplet_center_x
            y_coords = np.arange(phi.shape[0])
            
            grad_p_line = grad_p_magnitude[y_coords, x_line]
            surface_tension_line = surface_tension_force[y_coords, x_line]
            viscous_line = viscous_force_magnitude[y_coords, x_line]
            gravity_line = gravity_magnitude[y_coords, x_line]
            phi_line = phi[y_coords, x_line]
            rho_line = rho[y_coords, x_line]
            
            # Calculate force ratios
            total_force = grad_p_line + surface_tension_line + viscous_line + gravity_line
            grad_p_ratio = grad_p_line / (total_force + 1e-10)
            surface_tension_ratio = surface_tension_line / (total_force + 1e-10)
            viscous_ratio = viscous_line / (total_force + 1e-10)
            gravity_ratio = gravity_line / (total_force + 1e-10)
            
            force_data.append({
                'step': step_num,
                'droplet_center_x': droplet_center_x,
                'droplet_center_y': droplet_center_y,
                'y_coords': y_coords,
                'grad_p_line': grad_p_line,
                'surface_tension_line': surface_tension_line,
                'viscous_line': viscous_line,
                'gravity_line': gravity_line,
                'phi_line': phi_line,
                'rho_line': rho_line,
                'grad_p_ratio': grad_p_ratio,
                'surface_tension_ratio': surface_tension_ratio,
                'viscous_ratio': viscous_ratio,
                'gravity_ratio': gravity_ratio,
                'total_force': total_force
            })
            
            if step_num % 100 == 0:
                max_grad_p = np.max(grad_p_line)
                max_surface_tension = np.max(surface_tension_line)
                max_viscous = np.max(viscous_line)
                print(f"Step {step_num}: max|∇p|={max_grad_p:.2e}, max|F_s|={max_surface_tension:.2e}, max|F_v|={max_viscous:.2e}")
        
        except Exception as e:
            print(f"Error loading step {step_num}: {e}")
            continue
    
    if not force_data:
        print("No valid force data found!")
        return
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Force magnitudes at different times
    n_plots = min(4, len(force_data))
    plot_indices = np.linspace(0, len(force_data)-1, n_plots, dtype=int)
    
    for i, idx in enumerate(plot_indices):
        row = i // 2
        col = i % 2
        
        d = force_data[idx]
        y_coords = d['y_coords']
        
        axes[row, col].plot(d['grad_p_line'], y_coords, 'b-', linewidth=2, label='|∇p|')
        axes[row, col].plot(d['surface_tension_line'], y_coords, 'r-', linewidth=2, label='|F_s|')
        axes[row, col].plot(d['viscous_line'], y_coords, 'g-', linewidth=2, label='|F_v|')
        axes[row, col].plot(d['gravity_line'], y_coords, 'm-', linewidth=2, label='|F_g|')
        axes[row, col].set_title(f'Force Budget at step {d["step"]}')
        axes[row, col].set_xlabel('Force Magnitude')
        axes[row, col].set_ylabel('y coordinate')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_yscale('log')
    
    plt.tight_layout()
    
    # Create diagnostics directory
    diagnostics_dir = os.path.join(experiment_dir, 'diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    # Save plot
    output_path = os.path.join(diagnostics_dir, 'force_budget_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Force budget analysis plot saved to: {output_path}")
    
    # Create force ratio plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Force ratios over time
    steps = np.array([d['step'] for d in force_data])
    grad_p_ratios = np.array([np.mean(d['grad_p_ratio']) for d in force_data])
    surface_tension_ratios = np.array([np.mean(d['surface_tension_ratio']) for d in force_data])
    viscous_ratios = np.array([np.mean(d['viscous_ratio']) for d in force_data])
    gravity_ratios = np.array([np.mean(d['gravity_ratio']) for d in force_data])
    
    axes[0].plot(steps, grad_p_ratios, 'b-', linewidth=2, label='Pressure gradient')
    axes[0].plot(steps, surface_tension_ratios, 'r-', linewidth=2, label='Surface tension')
    axes[0].plot(steps, viscous_ratios, 'g-', linewidth=2, label='Viscous')
    axes[0].plot(steps, gravity_ratios, 'm-', linewidth=2, label='Gravity')
    axes[0].set_title('Force Ratios vs Time')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Force Ratio')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Force balance at final time
    final_data = force_data[-1]
    y_coords = final_data['y_coords']
    
    axes[1].plot(final_data['grad_p_ratio'], y_coords, 'b-', linewidth=2, label='Pressure gradient')
    axes[1].plot(final_data['surface_tension_ratio'], y_coords, 'r-', linewidth=2, label='Surface tension')
    axes[1].plot(final_data['viscous_ratio'], y_coords, 'g-', linewidth=2, label='Viscous')
    axes[1].plot(final_data['gravity_ratio'], y_coords, 'm-', linewidth=2, label='Gravity')
    axes[1].set_title('Force Ratios at Final Time')
    axes[1].set_xlabel('Force Ratio')
    axes[1].set_ylabel('y coordinate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save force ratio plot
    ratio_output_path = os.path.join(diagnostics_dir, 'force_ratios_analysis.png')
    plt.savefig(ratio_output_path, dpi=150, bbox_inches='tight')
    print(f"Force ratios analysis plot saved to: {ratio_output_path}")
    
    # Save data
    data_path = os.path.join(diagnostics_dir, 'force_budget_data.json')
    with open(data_path, 'w') as f:
        json_data = []
        for d in force_data:
            json_d = {}
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    json_d[k] = v.tolist()
                elif isinstance(v, (np.int64, np.int32)):
                    json_d[k] = int(v)
                elif isinstance(v, (np.float64, np.float32)):
                    json_d[k] = float(v)
                else:
                    json_d[k] = v
            json_data.append(json_d)
        json.dump(json_data, f, indent=2)
    print(f"Force budget data saved to: {data_path}")
    
    # Analysis summary
    print("\n=== FORCE BUDGET ANALYSIS SUMMARY ===")
    print(f"Final pressure gradient ratio: {grad_p_ratios[-1]:.3f}")
    print(f"Final surface tension ratio: {surface_tension_ratios[-1]:.3f}")
    print(f"Final viscous ratio: {viscous_ratios[-1]:.3f}")
    print(f"Final gravity ratio: {gravity_ratios[-1]:.3f}")
    
    # Check force balance
    total_ratio = grad_p_ratios[-1] + surface_tension_ratios[-1] + viscous_ratios[-1] + gravity_ratios[-1]
    if abs(total_ratio - 1.0) < 0.1:
        print("✅ Force balance is well maintained.")
    else:
        print("⚠️  WARNING: Force balance may be violated!")


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python force_budget.py <experiment_directory>")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    
    if not os.path.exists(experiment_dir):
        print(f"Error: Directory {experiment_dir} does not exist!")
        sys.exit(1)
    
    analyze_force_budget(experiment_dir)


if __name__ == "__main__":
    main()
