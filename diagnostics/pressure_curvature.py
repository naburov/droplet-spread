#!/usr/bin/env python3
"""
Pressure jump vs curvature analysis (Young-Laplace test) for droplet spreading simulation.
Tests the relationship between pressure jump and surface tension.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path
from scipy import ndimage
from scipy.interpolate import griddata

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from numerics.finite_differences import gradient, laplacian


def find_interface_contour(phi, level=0.0):
    """Find the interface contour where phi = level.
    
    Args:
        phi (np.ndarray): Phase field
        level (float): Level to find contour at
    
    Returns:
        tuple: (x_coords, y_coords) of contour points
    """
    # Simple contour finding without scikit-image
    # Find points where phi crosses the level
    phi_shifted = phi - level
    
    # Find zero crossings
    zero_crossings = []
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1] - 1):
            if phi_shifted[i, j] * phi_shifted[i, j+1] < 0:  # Sign change
                # Linear interpolation
                x_interp = j + phi_shifted[i, j] / (phi_shifted[i, j] - phi_shifted[i, j+1])
                zero_crossings.append((x_interp, i))
    
    if len(zero_crossings) == 0:
        return np.array([]), np.array([])
    
    # Convert to arrays
    x_coords = np.array([p[0] for p in zero_crossings])
    y_coords = np.array([p[1] for p in zero_crossings])
    
    return x_coords, y_coords


def calculate_curvature(x_coords, y_coords, dx, dy):
    """Calculate curvature along a contour in physical coordinates.
    
    Args:
        x_coords (np.ndarray): x coordinates of contour (grid units)
        y_coords (np.ndarray): y coordinates of contour (grid units)
        dx (float): grid spacing in x direction
        dy (float): grid spacing in y direction
    
    Returns:
        np.ndarray: Curvature values in physical units
    """
    if len(x_coords) < 3:
        return np.array([])
    
    # Convert to physical coordinates
    x_physical = x_coords * dx
    y_physical = y_coords * dy
    
    # Calculate first and second derivatives
    dx_ds = np.gradient(x_physical)
    dy_ds = np.gradient(y_physical)
    d2x_ds2 = np.gradient(dx_ds)
    d2y_ds2 = np.gradient(dy_ds)
    
    # Calculate curvature
    curvature = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / (dx_ds**2 + dy_ds**2)**(3/2)
    
    return curvature


def sample_pressure_near_interface(phi, pressure, x_coords, y_coords, dx, dy, radius=3):
    """Sample pressure properly from inside and outside the droplet.
    
    Args:
        phi (np.ndarray): Phase field
        pressure (np.ndarray): Pressure field
        x_coords (np.ndarray): x coordinates of interface
        y_coords (np.ndarray): y coordinates of interface
        radius (int): Radius for sampling region around interface points
    
    Returns:
        tuple: (pressure_inside, pressure_outside, curvature)
    """
    if len(x_coords) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Calculate curvature in physical coordinates
    curvature = calculate_curvature(x_coords, y_coords, dx, dy)
    
    # Sample pressure properly from inside and outside regions
    pressure_inside = []
    pressure_outside = []
    valid_curvatures = []
    
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        # Convert to grid indices
        ix = int(np.round(x))
        iy = int(np.round(y))
        
        # Check bounds
        if (ix < radius or ix >= phi.shape[1] - radius or 
            iy < radius or iy >= phi.shape[0] - radius):
            continue
        
        # Sample pressure in a small region around the interface point
        region_phi = phi[iy-radius:iy+radius+1, ix-radius:ix+radius+1]
        region_p = pressure[iy-radius:iy+radius+1, ix-radius:ix+radius+1]
        
        # Find inside and outside points
        inside_mask = region_phi > 0.5   # Clearly inside droplet
        outside_mask = region_phi < -0.5  # Clearly outside droplet
        
        if np.sum(inside_mask) > 0 and np.sum(outside_mask) > 0:
            # Sample pressure from inside and outside
            p_inside = np.mean(region_p[inside_mask])
            p_outside = np.mean(region_p[outside_mask])
            
            pressure_inside.append(p_inside)
            pressure_outside.append(p_outside)
            valid_curvatures.append(curvature[i])
    
    return (np.array(pressure_inside), np.array(pressure_outside), 
            np.array(valid_curvatures))


def analyze_pressure_curvature(experiment_dir):
    """Analyze pressure jump vs curvature relationship.
    
    Args:
        experiment_dir (str): Path to experiment directory
    """
    print(f"Analyzing pressure-curvature relationship in: {experiment_dir}")
    
    # Load configuration
    config_path = os.path.join(experiment_dir, 'simulation_parameters.json')
    if not os.path.exists(config_path):
        print("No simulation_parameters.json found!")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dx = config['grid_params']['Lx'] / config['grid_params']['Nx']
    dy = config['grid_params']['Ly'] / config['grid_params']['Ny']
    
    # Get surface tension parameters
    We1 = config['physical_params']['We1']
    We2 = config['physical_params']['We2']
    rho1 = config['physical_params']['rho1']
    rho2 = config['physical_params']['rho2']
    
    # Calculate effective surface tension
    sigma = 1.0 / We1  # Assuming We1 is the reference Weber number
    
    print(f"Grid spacing: dx={dx:.6f}, dy={dy:.6f}")
    print(f"Surface tension: σ={sigma:.6f}")
    
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
    
    # Analyze pressure-curvature relationship
    pressure_curvature_data = []
    
    for step_num, file_path in checkpoint_files:
        try:
            data = np.load(file_path)
            phi = data['phi']      # Phase field
            pressure = data['P']   # Pressure field
            
            # Find interface contour
            x_coords, y_coords = find_interface_contour(phi, level=0.0)
            
            if len(x_coords) == 0:
                print(f"Step {step_num}: No interface found")
                continue
            
            # Sample pressure near interface
            p_inside, p_outside, curvature = sample_pressure_near_interface(
                phi, pressure, x_coords, y_coords, dx, dy)
            
            if len(p_inside) == 0:
                print(f"Step {step_num}: No valid pressure samples")
                continue
            
            # Calculate pressure jump
            pressure_jump = p_inside - p_outside
            
            # Calculate theoretical pressure jump (Young-Laplace)
            theoretical_jump = sigma * curvature
            
            # Calculate error
            error = pressure_jump - theoretical_jump
            relative_error = error / theoretical_jump if np.any(theoretical_jump != 0) else np.zeros_like(error)
            
            pressure_curvature_data.append({
                'step': step_num,
                'pressure_jump': pressure_jump,
                'curvature': curvature,
                'theoretical_jump': theoretical_jump,
                'error': error,
                'relative_error': relative_error,
                'x_coords': x_coords,
                'y_coords': y_coords
            })
            
            if step_num % 100 == 0:
                mean_error = np.mean(np.abs(error))
                print(f"Step {step_num}: Mean error={mean_error:.2e}, samples={len(pressure_jump)}")
        
        except Exception as e:
            print(f"Error loading step {step_num}: {e}")
            continue
    
    if not pressure_curvature_data:
        print("No valid pressure-curvature data found!")
        return
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Pressure jump vs curvature scatter plot
    all_pressure_jumps = np.concatenate([d['pressure_jump'] for d in pressure_curvature_data])
    all_curvatures = np.concatenate([d['curvature'] for d in pressure_curvature_data])
    all_theoretical = np.concatenate([d['theoretical_jump'] for d in pressure_curvature_data])
    
    axes[0, 0].scatter(all_curvatures, all_pressure_jumps, alpha=0.6, s=20, label='Numerical')
    axes[0, 0].scatter(all_curvatures, all_theoretical, alpha=0.6, s=20, label='Theoretical')
    
    # Add Young-Laplace line (perfect relationship)
    if len(all_curvatures) > 0:
        curvature_range = np.linspace(np.min(all_curvatures), np.max(all_curvatures), 100)
        young_laplace_line = sigma * curvature_range
        axes[0, 0].plot(curvature_range, young_laplace_line, 'k--', linewidth=2, 
                       label=f'Young-Laplace: Δp = {sigma}κ')
    
    axes[0, 0].set_xlabel('Curvature κ')
    axes[0, 0].set_ylabel('Pressure Jump Δp')
    axes[0, 0].set_title('Pressure Jump vs Curvature (Young-Laplace Test)\n(Should follow linear relationship)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error analysis
    all_errors = np.concatenate([d['error'] for d in pressure_curvature_data])
    all_relative_errors = np.concatenate([d['relative_error'] for d in pressure_curvature_data])
    
    axes[0, 1].scatter(all_curvatures, all_errors, alpha=0.6, s=20)
    axes[0, 1].set_xlabel('Curvature κ')
    axes[0, 1].set_ylabel('Error (Δp - σκ)')
    axes[0, 1].set_title('Pressure Jump Error vs Curvature')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Relative error
    axes[1, 0].scatter(all_curvatures, all_relative_errors, alpha=0.6, s=20)
    axes[1, 0].set_xlabel('Curvature κ')
    axes[1, 0].set_ylabel('Relative Error')
    axes[1, 0].set_title('Relative Error vs Curvature')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error histogram
    axes[1, 1].hist(all_errors, bins=50, alpha=0.7, density=True)
    axes[1, 1].set_xlabel('Error (Δp - σκ)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create diagnostics directory
    diagnostics_dir = os.path.join(experiment_dir, 'diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    # Save plot
    output_path = os.path.join(diagnostics_dir, 'pressure_curvature_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Pressure-curvature analysis plot saved to: {output_path}")
    
    # Save data
    data_path = os.path.join(diagnostics_dir, 'pressure_curvature_data.json')
    with open(data_path, 'w') as f:
        json_data = []
        for d in pressure_curvature_data:
            json_d = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                     for k, v in d.items() if k not in ['x_coords', 'y_coords']}
            json_d['step'] = int(json_d['step'])
            json_data.append(json_d)
        json.dump(json_data, f, indent=2)
    print(f"Pressure-curvature data saved to: {data_path}")
    
    # Analysis summary
    print("\n=== PRESSURE-CURVATURE ANALYSIS SUMMARY ===")
    print(f"Total samples: {len(all_pressure_jumps)}")
    print(f"Mean absolute error: {np.mean(np.abs(all_errors)):.2e}")
    print(f"RMS error: {np.sqrt(np.mean(all_errors**2)):.2e}")
    print(f"Mean relative error: {np.mean(np.abs(all_relative_errors)):.2e}")
    
    # Check Young-Laplace law compliance
    if np.mean(np.abs(all_relative_errors)) < 0.1:
        print("✅ Young-Laplace law is well satisfied.")
    else:
        print("⚠️  WARNING: Significant deviation from Young-Laplace law!")


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python pressure_curvature.py <experiment_directory>")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    
    if not os.path.exists(experiment_dir):
        print(f"Error: Directory {experiment_dir} does not exist!")
        sys.exit(1)
    
    analyze_pressure_curvature(experiment_dir)


if __name__ == "__main__":
    main()
