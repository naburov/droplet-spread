#!/usr/bin/env python3
"""
CFL and capillary time-scale analysis for droplet spreading simulation.
Tracks advective CFL, diffusive CFL, and capillary time scale to help choose dt robustly.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from physics.properties import calculate_density, calculate_reynolds_number


def analyze_cfl_timescales(experiment_dir):
    """Analyze CFL numbers and time scales.
    
    Args:
        experiment_dir (str): Path to experiment directory
    """
    print(f"Analyzing CFL and time scales in: {experiment_dir}")
    
    # Load configuration
    config_path = os.path.join(experiment_dir, 'simulation_parameters.json')
    if not os.path.exists(config_path):
        print("No simulation_parameters.json found!")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dx = config['grid_params']['Lx'] / config['grid_params']['Nx']
    dy = config['grid_params']['Ly'] / config['grid_params']['Ny']
    dt = config['time_params']['dt']
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
    print(f"Time step: dt={dt:.6f}")
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
    
    # Analyze CFL and time scales
    cfl_data = []
    
    for step_num, file_path in checkpoint_files:
        try:
            data = np.load(file_path)
            phi = data['phi']  # Phase field
            U = data['U']      # Velocity field
            
            # Calculate density and Reynolds number
            rho = calculate_density(phi, rho1, rho2)
            Re = calculate_reynolds_number(phi, Re1, Re2)
            
            # Calculate velocity magnitude
            u_magnitude = np.sqrt(np.sum(U**2, axis=-1))
            max_velocity = np.max(u_magnitude)
            mean_velocity = np.mean(u_magnitude)
            
            # Advective CFL number
            cfl_advective = max_velocity * dt / min(dx, dy)
            
            # Diffusive CFL number (viscous)
            # CFL_diff = ν * dt / Δx², where ν = 1/Re
            nu = 1.0 / Re
            cfl_diffusive = np.max(nu) * dt / (min(dx, dy)**2)
            
            # Capillary time scale
            # τ_cap = √(ρ * Δx³ / σ), where σ = 1/We
            sigma = 1.0 / We1  # Reference surface tension
            rho_avg = np.mean(rho)
            tau_cap = np.sqrt(rho_avg * (min(dx, dy)**3) / sigma)
            cfl_capillary = dt / tau_cap
            
            # Gravity time scale
            # τ_grav = √(Δx / g)
            tau_grav = np.sqrt(min(dx, dy) / abs(g))
            cfl_gravity = dt / tau_grav
            
            # Interface time scale
            # τ_interface = ε² / (Pe * Δx²)
            Pe = config['physical_params']['Pe']
            tau_interface = (epsilon**2) / (Pe * (min(dx, dy)**2))
            cfl_interface = dt / tau_interface
            
            # Calculate time step recommendations
            dt_max_advective = 0.5 * min(dx, dy) / max_velocity
            dt_max_diffusive = 0.5 * (min(dx, dy)**2) / np.max(nu)
            dt_max_capillary = 0.5 * tau_cap
            dt_max_gravity = 0.5 * tau_grav
            dt_max_interface = 0.5 * tau_interface
            
            dt_recommended = min(dt_max_advective, dt_max_diffusive, dt_max_capillary, 
                               dt_max_gravity, dt_max_interface)
            
            cfl_data.append({
                'step': step_num,
                'cfl_advective': cfl_advective,
                'cfl_diffusive': cfl_diffusive,
                'cfl_capillary': cfl_capillary,
                'cfl_gravity': cfl_gravity,
                'cfl_interface': cfl_interface,
                'max_velocity': max_velocity,
                'mean_velocity': mean_velocity,
                'tau_cap': tau_cap,
                'tau_grav': tau_grav,
                'tau_interface': tau_interface,
                'dt_recommended': dt_recommended,
                'dt_actual': dt
            })
            
            if step_num % 100 == 0:
                print(f"Step {step_num}: CFL_adv={cfl_advective:.3f}, CFL_diff={cfl_diffusive:.3f}, CFL_cap={cfl_capillary:.3f}")
        
        except Exception as e:
            print(f"Error loading step {step_num}: {e}")
            continue
    
    if not cfl_data:
        print("No valid CFL data found!")
        return
    
    # Convert to arrays
    steps = np.array([d['step'] for d in cfl_data])
    cfl_advective = np.array([d['cfl_advective'] for d in cfl_data])
    cfl_diffusive = np.array([d['cfl_diffusive'] for d in cfl_data])
    cfl_capillary = np.array([d['cfl_capillary'] for d in cfl_data])
    cfl_gravity = np.array([d['cfl_gravity'] for d in cfl_data])
    cfl_interface = np.array([d['cfl_interface'] for d in cfl_data])
    max_velocities = np.array([d['max_velocity'] for d in cfl_data])
    dt_recommended = np.array([d['dt_recommended'] for d in cfl_data])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # CFL numbers
    axes[0, 0].semilogy(steps, cfl_advective, 'b-', linewidth=2, label='Advective CFL')
    axes[0, 0].semilogy(steps, cfl_diffusive, 'r-', linewidth=2, label='Diffusive CFL')
    axes[0, 0].semilogy(steps, cfl_capillary, 'g-', linewidth=2, label='Capillary CFL')
    axes[0, 0].semilogy(steps, cfl_gravity, 'm-', linewidth=2, label='Gravity CFL')
    axes[0, 0].semilogy(steps, cfl_interface, 'c-', linewidth=2, label='Interface CFL')
    
    # Add stability limits
    axes[0, 0].axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='Stability limit (0.5)')
    axes[0, 0].axhline(y=0.1, color='g', linestyle='--', alpha=0.7, label='Conservative (0.1)')
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Unstable (>1.0)')
    
    axes[0, 0].set_title('CFL Numbers vs Time\n(All should be < 0.5 for stability)')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('CFL Number')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Velocity evolution
    axes[0, 1].semilogy(steps, max_velocities, 'b-', linewidth=2, label='Max velocity')
    axes[0, 1].set_title('Maximum Velocity vs Time')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Time step recommendations
    axes[1, 0].semilogy(steps, dt_recommended, 'g-', linewidth=2, label='Recommended dt')
    axes[1, 0].axhline(y=dt, color='r', linestyle='-', linewidth=2, label='Actual dt')
    axes[1, 0].set_title('Time Step Recommendations')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Time Step')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # CFL stability check
    max_cfl = np.maximum.reduce([cfl_advective, cfl_diffusive, cfl_capillary, 
                                cfl_gravity, cfl_interface])
    axes[1, 1].semilogy(steps, max_cfl, 'r-', linewidth=2, label='Max CFL')
    axes[1, 1].axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='Stability limit')
    axes[1, 1].set_title('Maximum CFL Number')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Max CFL Number')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create diagnostics directory
    diagnostics_dir = os.path.join(experiment_dir, 'diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    # Save plot
    output_path = os.path.join(diagnostics_dir, 'cfl_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"CFL analysis plot saved to: {output_path}")
    
    # Save data
    data_path = os.path.join(diagnostics_dir, 'cfl_data.json')
    with open(data_path, 'w') as f:
        json_data = []
        for d in cfl_data:
            json_d = {k: v for k, v in d.items()}
            json_d['step'] = int(json_d['step'])
            json_data.append(json_d)
        json.dump(json_data, f, indent=2)
    print(f"CFL data saved to: {data_path}")
    
    # Analysis summary
    print("\n=== CFL ANALYSIS SUMMARY ===")
    print(f"Final advective CFL: {cfl_advective[-1]:.3f}")
    print(f"Final diffusive CFL: {cfl_diffusive[-1]:.3f}")
    print(f"Final capillary CFL: {cfl_capillary[-1]:.3f}")
    print(f"Final gravity CFL: {cfl_gravity[-1]:.3f}")
    print(f"Final interface CFL: {cfl_interface[-1]:.3f}")
    print(f"Final max velocity: {max_velocities[-1]:.2e}")
    print(f"Recommended dt: {dt_recommended[-1]:.2e}")
    print(f"Actual dt: {dt:.2e}")
    
    # Check stability
    max_cfl_final = max(cfl_advective[-1], cfl_diffusive[-1], cfl_capillary[-1], 
                       cfl_gravity[-1], cfl_interface[-1])
    
    if max_cfl_final < 0.5:
        print("✅ CFL numbers are within stability limits.")
    else:
        print("⚠️  WARNING: CFL numbers exceed stability limits!")
    
    # Check time step
    if dt_recommended[-1] > dt:
        print("✅ Time step is conservative.")
    else:
        print("⚠️  WARNING: Time step may be too large for stability!")


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python cfl_analysis.py <experiment_directory>")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    
    if not os.path.exists(experiment_dir):
        print(f"Error: Directory {experiment_dir} does not exist!")
        sys.exit(1)
    
    analyze_cfl_timescales(experiment_dir)


if __name__ == "__main__":
    main()
