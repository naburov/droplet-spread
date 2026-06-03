#!/usr/bin/env python3
"""
Contact angle measurement for droplet spreading simulation.
Extracts φ=0 near the wall and computes θ from the local gradient.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from numerics.finite_differences import gradient


def find_contact_line(phi, y_wall=0, num_rows=3):
    """Find left/right contact points from near-wall liquid footprint.
    
    Args:
        phi (np.ndarray): Phase field
        y_wall (int): y coordinate of the wall
        tolerance (float): Tolerance for finding phi ≈ 0
    
    Returns:
        tuple: (x_coords, y_coords) of contact line points
    """
    # phi layout is (Nx, Ny): first index is x, second is y.
    j0 = max(int(y_wall), 0)
    j1 = min(phi.shape[1], j0 + max(int(num_rows), 1))
    band = phi[:, j0:j1]
    # Liquid phase is phi < 0.
    liquid_near_wall = np.any(band < 0.0, axis=1)
    xs = np.where(liquid_near_wall)[0]
    if xs.size == 0:
        return np.array([]), np.array([])
    left = float(xs.min())
    right = float(xs.max())
    if abs(right - left) < 1e-12:
        return np.array([left]), np.array([float(y_wall)])
    return np.array([left, right]), np.array([float(y_wall), float(y_wall)])


def calculate_contact_angle(phi, x_contact, y_contact, dx, dy):
    """Calculate physical liquid-side contact angle from phase-field gradient.
    
    Args:
        phi (np.ndarray): Phase field
        x_contact (float): x coordinate of contact point
        y_contact (float): y coordinate of contact point
        dx (float): Grid spacing in x
        dy (float): Grid spacing in y
    
    Returns:
        float: Contact angle in degrees
    """
    # Convert to grid indices (phi shape: Nx, Ny)
    ix = int(np.round(x_contact))
    iy = int(np.round(y_contact))
    
    # Ensure indices are within bounds
    ix = max(0, min(ix, phi.shape[0] - 1))
    iy = max(0, min(iy, phi.shape[1] - 1))
    # Use first interior row to avoid using boundary-overwritten gradient at wall.
    iy = max(1, iy)
    
    # Calculate gradient at contact point
    grad_phi = gradient(phi, dx, dy)
    
    # Get gradient components
    dphi_dx = grad_phi[ix, iy, 0]
    dphi_dy = grad_phi[ix, iy, 1]
    
    grad_norm = np.sqrt(dphi_dx * dphi_dx + dphi_dy * dphi_dy)
    if grad_norm <= 1e-12:
        return 90.0

    # Runtime config uses liquid-side contact angle convention.
    cos_theta = dphi_dy / grad_norm
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def analyze_contact_angle(experiment_dir):
    """Analyze contact angle evolution.
    
    Args:
        experiment_dir (str): Path to experiment directory
    """
    print(f"Analyzing contact angle in: {experiment_dir}")
    
    # Load configuration
    config_path = os.path.join(experiment_dir, 'simulation_parameters.json')
    if not os.path.exists(config_path):
        print("No simulation_parameters.json found!")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dx = config['grid_params']['Lx'] / config['grid_params']['Nx']
    dy = config['grid_params']['Ly'] / config['grid_params']['Ny']
    target_angle = config['physical_params']['contact_angle']
    
    print(f"Grid spacing: dx={dx:.6f}, dy={dy:.6f}")
    print(f"Target contact angle: {target_angle}°")
    
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
    
    # Analyze contact angle
    contact_angle_data = []
    
    for step_num, file_path in checkpoint_files:
        try:
            data = np.load(file_path)
            phi = data['phi']  # Phase field
            
            # Find contact line
            x_contact, y_contact = find_contact_line(phi, y_wall=0)
            
            if len(x_contact) == 0:
                print(f"Step {step_num}: No contact line found")
                continue
            
            # Calculate contact angles for both sides (if present)
            angles = [
                calculate_contact_angle(phi, xc, yc, dx, dy)
                for xc, yc in zip(x_contact, y_contact)
            ]
            contact_angle = float(np.mean(angles))
            
            # Contact line positions (left/right and midpoint)
            contact_left = float(np.min(x_contact) * dx)
            contact_right = float(np.max(x_contact) * dx)
            contact_line_position = 0.5 * (contact_left + contact_right)
            
            # Calculate droplet height (distance from wall to top of droplet)
            droplet_mask = phi < 0
            if np.any(droplet_mask):
                droplet_top = np.max(np.where(droplet_mask)[1])
                droplet_height = droplet_top * dy
            else:
                droplet_height = 0.0
            
            # Calculate droplet width (horizontal extent)
            if np.any(droplet_mask):
                droplet_left = np.min(np.where(droplet_mask)[0])
                droplet_right = np.max(np.where(droplet_mask)[0])
                droplet_width = (droplet_right - droplet_left) * dx
            else:
                droplet_width = 0.0
            
            contact_angle_data.append({
                'step': step_num,
                'contact_angle': contact_angle,
                'contact_angle_left': float(angles[0]),
                'contact_angle_right': float(angles[-1]),
                'contact_line_position': contact_line_position,
                'contact_line_left': contact_left,
                'contact_line_right': contact_right,
                'droplet_height': droplet_height,
                'droplet_width': droplet_width,
                'x_contact': float(contact_line_position / dx),
                'y_contact': 0.0
            })
            
            if step_num % 100 == 0:
                print(f"Step {step_num}: θ={contact_angle:.1f}°, pos={contact_line_position:.3f}, height={droplet_height:.3f}")
        
        except Exception as e:
            print(f"Error loading step {step_num}: {e}")
            continue
    
    if not contact_angle_data:
        print("No valid contact angle data found!")
        return
    
    # Convert to arrays
    steps = np.array([d['step'] for d in contact_angle_data])
    contact_angles = np.array([d['contact_angle'] for d in contact_angle_data])
    contact_line_positions = np.array([d['contact_line_position'] for d in contact_angle_data])
    droplet_heights = np.array([d['droplet_height'] for d in contact_angle_data])
    droplet_widths = np.array([d['droplet_width'] for d in contact_angle_data])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Contact angle evolution
    contact_angles_left = np.array([d.get('contact_angle_left', d['contact_angle']) for d in contact_angle_data])
    contact_angles_right = np.array([d.get('contact_angle_right', d['contact_angle']) for d in contact_angle_data])
    contact_line_left = np.array([d.get('contact_line_left', d['contact_line_position']) for d in contact_angle_data])
    contact_line_right = np.array([d.get('contact_line_right', d['contact_line_position']) for d in contact_angle_data])

    axes[0, 0].plot(steps, contact_angles, 'b-', linewidth=2, label='Mean angle')
    axes[0, 0].plot(steps, contact_angles_left, color='teal', linewidth=1.4, alpha=0.7, label='Left')
    axes[0, 0].plot(steps, contact_angles_right, color='navy', linewidth=1.4, alpha=0.7, label='Right')
    axes[0, 0].axhline(y=target_angle, color='r', linestyle='--', linewidth=2, label=f'Target angle ({target_angle}°)')
    axes[0, 0].set_title('Contact Angle Evolution')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Contact Angle (degrees)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Contact line position
    axes[0, 1].plot(steps, contact_line_positions, 'g-', linewidth=2, label='Midpoint')
    axes[0, 1].plot(steps, contact_line_left, color='limegreen', linewidth=1.4, alpha=0.7, label='Left')
    axes[0, 1].plot(steps, contact_line_right, color='darkgreen', linewidth=1.4, alpha=0.7, label='Right')
    axes[0, 1].set_title('Contact Line Position')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Position (m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Droplet dimensions
    axes[1, 0].plot(steps, droplet_heights, 'b-', linewidth=2, label='Height')
    axes[1, 0].plot(steps, droplet_widths, 'r-', linewidth=2, label='Width')
    axes[1, 0].set_title('Droplet Dimensions')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Dimension (m)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Contact angle error
    angle_error = contact_angles - target_angle
    axes[1, 1].plot(steps, angle_error, 'm-', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.5)
    axes[1, 1].set_title('Contact Angle Error')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Error (degrees)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create diagnostics directory
    diagnostics_dir = os.path.join(experiment_dir, 'diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    # Save plot
    output_path = os.path.join(diagnostics_dir, 'contact_angle_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Contact angle analysis plot saved to: {output_path}")
    
    # Save data
    data_path = os.path.join(diagnostics_dir, 'contact_angle_data.json')
    with open(data_path, 'w') as f:
        json_data = []
        for d in contact_angle_data:
            json_d = {k: v for k, v in d.items()}
            json_d['step'] = int(json_d['step'])
            json_data.append(json_d)
        json.dump(json_data, f, indent=2)
    print(f"Contact angle data saved to: {data_path}")
    
    # Analysis summary
    print("\n=== CONTACT ANGLE ANALYSIS SUMMARY ===")
    print(f"Initial contact angle: {contact_angles[0]:.1f}°")
    print(f"Final contact angle: {contact_angles[-1]:.1f}°")
    print(f"Target contact angle: {target_angle}°")
    print(f"Final error: {angle_error[-1]:.1f}°")
    print(f"RMS error: {np.sqrt(np.mean(angle_error**2)):.1f}°")
    
    # Check contact angle accuracy
    if abs(angle_error[-1]) < 5.0:
        print("✅ Contact angle is close to target.")
    else:
        print("⚠️  WARNING: Contact angle deviates significantly from target!")
    
    # Check contact angle stability
    angle_std = np.std(contact_angles)
    if angle_std < 2.0:
        print("✅ Contact angle is stable.")
    else:
        print("⚠️  WARNING: Contact angle is unstable!")


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python contact_angle.py <experiment_directory>")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    
    if not os.path.exists(experiment_dir):
        print(f"Error: Directory {experiment_dir} does not exist!")
        sys.exit(1)
    
    analyze_contact_angle(experiment_dir)


if __name__ == "__main__":
    main()
