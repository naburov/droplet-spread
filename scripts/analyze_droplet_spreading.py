#!/usr/bin/env python3
"""
Analyze droplet spreading from checkpoint files.

This script loads checkpoints and analyzes:
- Droplet width (left-right extent)
- Droplet height
- Contact line positions
- Phase field statistics
- Surface tension forces
"""

try:
    import numpy as np
except ImportError:
    try:
        import jax.numpy as np
        print("Using JAX numpy instead of numpy")
    except ImportError:
        print("Error: Neither numpy nor jax.numpy available")
        sys.exit(1)

import json
import sys
import os
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")


def load_checkpoint(checkpoint_path):
    """Load a checkpoint file."""
    # Try numpy first, then fall back to manual loading
    try:
        data = np.load(checkpoint_path, allow_pickle=True)
        return {
            'phi': np.array(data['phi']),
            'U': np.array(data['U']),
            'P': np.array(data['P']),
            'surface_tension': np.array(data['surface_tension']) if 'surface_tension' in data else None,
            'step': int(data.get('step', 0)),
            'time': float(data.get('time', 0.0))
        }
    except Exception as e:
        # Fallback: use pickle or manual loading
        import pickle
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        return {
            'phi': np.array(data['phi']),
            'U': np.array(data['U']),
            'P': np.array(data['P']),
            'surface_tension': np.array(data.get('surface_tension', None)),
            'step': int(data.get('step', 0)),
            'time': float(data.get('time', 0.0))
        }


def find_droplet_extent(phi, threshold=0.0):
    """Find droplet extent in x and y directions.
    
    Args:
        phi: Phase field (Nx, Ny), phi=-1 is liquid, phi=+1 is air
        threshold: Phase field threshold for interface (default: 0.0)
    
    Returns:
        dict with x_min, x_max, y_min, y_max, width, height, center_x, center_y
    """
    # Find liquid region (phi < threshold)
    liquid_mask = phi < threshold
    
    if not np.any(liquid_mask):
        return None
    
    # Get indices where liquid exists
    y_indices, x_indices = np.where(liquid_mask)
    
    x_min = np.min(x_indices)
    x_max = np.max(x_indices)
    y_min = np.min(y_indices)
    y_max = np.max(y_indices)
    
    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'width': x_max - x_min + 1,
        'height': y_max - y_min + 1,
        'center_x': (x_min + x_max) / 2.0,
        'center_y': (y_min + y_max) / 2.0
    }


def find_contact_line(phi, bottom_row_idx=0, threshold=0.0, num_rows=3):
    """Find contact line positions at the bottom.
    
    Args:
        phi: Phase field (Nx, Ny)
        bottom_row_idx: Index of bottom row (default: 0)
        threshold: Phase field threshold for interface
        num_rows: Number of bottom rows to check (default: 3)
    
    Returns:
        dict with left_contact, right_contact, contact_width
    """
    # Check multiple bottom rows to find contact line
    bottom_rows = phi[bottom_row_idx:bottom_row_idx+num_rows, :]
    # Liquid is where phi < threshold
    liquid_mask = np.any(bottom_rows < threshold, axis=0)
    
    if not np.any(liquid_mask):
        return None
    
    x_indices = np.where(liquid_mask)[0]
    left_contact = np.min(x_indices)
    right_contact = np.max(x_indices)
    
    return {
        'left_contact': left_contact,
        'right_contact': right_contact,
        'contact_width': right_contact - left_contact + 1
    }


def analyze_checkpoint(checkpoint_path, dx=1.0, dy=1.0):
    """Analyze a single checkpoint."""
    try:
        data = load_checkpoint(checkpoint_path)
        phi = data['phi']
        U = data['U']
        P = data['P']
        step = data['step']
        time = data['time']
        
        # Find droplet extent
        extent = find_droplet_extent(phi)
        if extent is None:
            return None
        
        # Find contact line
        contact = find_contact_line(phi)
        
        # Calculate phase field statistics
        phi_min = np.min(phi)
        phi_max = np.max(phi)
        phi_mean = np.mean(phi)
        
        # Calculate velocity statistics in liquid region
        liquid_mask = phi < 0.0
        if np.any(liquid_mask):
            U_liquid = U[liquid_mask]
            U_mag_liquid = np.sqrt(U_liquid[:, 0]**2 + U_liquid[:, 1]**2)
            U_max_liquid = np.max(U_mag_liquid) if len(U_mag_liquid) > 0 else 0.0
            U_mean_liquid = np.mean(U_mag_liquid) if len(U_mag_liquid) > 0 else 0.0
        else:
            U_max_liquid = 0.0
            U_mean_liquid = 0.0
        
        # Surface tension magnitude
        if data['surface_tension'] is not None:
            st = data['surface_tension']
            st_mag = np.sqrt(st[:, :, 0]**2 + st[:, :, 1]**2)
            st_max = np.max(st_mag)
            st_mean = np.mean(st_mag)
        else:
            st_max = 0.0
            st_mean = 0.0
        
        result = {
            'step': step,
            'time': time,
            'width': extent['width'] * dx,
            'height': extent['height'] * dy,
            'center_x': extent['center_x'] * dx,
            'center_y': extent['center_y'] * dy,
            'phi_min': phi_min,
            'phi_max': phi_max,
            'phi_mean': phi_mean,
            'U_max_liquid': U_max_liquid,
            'U_mean_liquid': U_mean_liquid,
            'st_max': st_max,
            'st_mean': st_mean,
        }
        
        if contact is not None:
            result['contact_left'] = contact['left_contact'] * dx
            result['contact_right'] = contact['right_contact'] * dx
            result['contact_width'] = contact['contact_width'] * dx
        else:
            result['contact_left'] = None
            result['contact_right'] = None
            result['contact_width'] = None
        
        return result
        
    except Exception as e:
        print(f"Error analyzing {checkpoint_path}: {e}", file=sys.stderr)
        return None


def analyze_experiment(experiment_dir):
    """Analyze all checkpoints in an experiment directory."""
    experiment_path = Path(experiment_dir)
    checkpoint_dir = experiment_path / 'checkpoints'
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}", file=sys.stderr)
        return None
    
    # Load simulation parameters
    params_file = experiment_path / 'simulation_parameters.json'
    if params_file.exists():
        with open(params_file, 'r') as f:
            params = json.load(f)
        dx = params['grid_params']['Lx'] / params['grid_params']['Nx']
        dy = params['grid_params']['Ly'] / params['grid_params']['Ny']
    else:
        dx = 1.0 / 128  # Default
        dy = 1.0 / 128
    
    # Find all checkpoints
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.npz'))
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}", file=sys.stderr)
        return None
    
    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Grid spacing: dx={dx:.6f}, dy={dy:.6f}")
    print()
    
    # Analyze each checkpoint
    results = []
    for checkpoint_path in checkpoints:
        result = analyze_checkpoint(checkpoint_path, dx, dy)
        if result is not None:
            results.append(result)
            step = result.get('step', 0)
            time = result.get('time', 0.0)
            width = result.get('width', 0.0)
            height = result.get('height', 0.0)
            contact_width = result.get('contact_width')
            contact_str = f"{contact_width:.4f}" if contact_width is not None else "N/A"
            u_max = result.get('U_max_liquid', 0.0)
            print(f"Step {step:6d} | "
                  f"Time {time:.6f} | "
                  f"Width {width:.4f} | "
                  f"Height {height:.4f} | "
                  f"Contact width {contact_str:>8} | "
                  f"U_max {u_max:.4f}")
    
    return results


def plot_spreading(results, output_file=None):
    """Plot spreading analysis."""
    if not results:
        return
    
    steps = [r['step'] for r in results]
    times = [r['time'] for r in results]
    widths = [r['width'] for r in results]
    heights = [r['height'] for r in results]
    contact_widths = [r['contact_width'] if r['contact_width'] is not None else 0.0 for r in results]
    U_max = [r['U_max_liquid'] for r in results]
    st_max = [r['st_max'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Width and height over time
    ax = axes[0, 0]
    ax.plot(times, widths, 'b-', label='Width', linewidth=2)
    ax.plot(times, heights, 'r-', label='Height', linewidth=2)
    ax.plot(times, contact_widths, 'g--', label='Contact width', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Length')
    ax.set_title('Droplet Dimensions vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Width and height vs step
    ax = axes[0, 1]
    ax.plot(steps, widths, 'b-', label='Width', linewidth=2)
    ax.plot(steps, heights, 'r-', label='Height', linewidth=2)
    ax.plot(steps, contact_widths, 'g--', label='Contact width', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Length')
    ax.set_title('Droplet Dimensions vs Step')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Velocity in liquid
    ax = axes[1, 0]
    ax.plot(times, U_max, 'b-', label='Max velocity in liquid', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Velocity')
    ax.set_title('Velocity in Liquid Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Surface tension
    ax = axes[1, 1]
    ax.plot(times, st_max, 'r-', label='Max surface tension', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Surface Tension Force')
    ax.set_title('Surface Tension Force')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_droplet_spreading.py <experiment_dir> [output_plot.png]")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    output_plot = sys.argv[2] if len(sys.argv) > 2 else None
    
    results = analyze_experiment(experiment_dir)
    
    if results:
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total checkpoints analyzed: {len(results)}")
        if results:
            initial = results[0]
            final = results[-1]
            print(f"Initial width: {initial['width']:.4f}, height: {initial['height']:.4f}")
            print(f"Final width:   {final['width']:.4f}, height: {final['height']:.4f}")
            print(f"Width change:  {final['width'] - initial['width']:.4f} ({100*(final['width']/initial['width'] - 1):.1f}%)")
            print(f"Height change: {final['height'] - initial['height']:.4f} ({100*(final['height']/initial['height'] - 1):.1f}%)")
            if initial['contact_width'] and final['contact_width']:
                print(f"Contact width: {initial['contact_width']:.4f} -> {final['contact_width']:.4f}")
        
        if output_plot or True:  # Always create plot
            plot_file = output_plot or os.path.join(experiment_dir, 'spreading_analysis.png')
            plot_spreading(results, plot_file)


if __name__ == '__main__':
    main()
