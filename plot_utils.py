import os
import sys
import json
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from utils import numerical_derivative, calculate_density


def plot_tension_force_vector(tension_force, save_path=None, title=None):
    """Plot the surface tension force as vectors."""
    
    # Calculate the magnitude of the force
    force_magnitude = np.sqrt(tension_force[:, :, 0]**2 + tension_force[:, :, 1]**2)
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Plot the magnitude as a heatmap
    im = plt.imshow(force_magnitude.T, origin='lower', cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Force Magnitude')
    
    # Create correct meshgrid
    X, Y = np.meshgrid(np.arange(tension_force.shape[0]), 
                       np.arange(tension_force.shape[1]), 
                       indexing='ij')
    
    skip = 5  # Skip every 5 points for clearer visualization
    plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
               tension_force[::skip, ::skip, 0], tension_force[::skip, ::skip, 1],
               color='white', scale=100)
    
    # Add title
    if title:
        plt.title(title)
    else:
        plt.title('Surface Tension Force Vector Field')
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def create_joint_plot(phi, U, P, surface_tension, dt, step, dx, dy, mass, rho1, rho2, save_path=None):
    """Create a joint plot with multiple subplots."""
    # Calculate derived fields
    U_magnitude = np.sqrt(U[..., 0]**2 + U[..., 1]**2)
    divergence = numerical_derivative(U[..., 0], axis=0, h=dx) + numerical_derivative(U[..., 1], axis=1, h=dy)
    ST_magnitude = np.sqrt(surface_tension[..., 0]**2 + surface_tension[..., 1]**2)
    rho = calculate_density(phi, rho1, rho2)
    
    # Setup the figure with more space at bottom for text
    fig = plt.figure(figsize=(18, 14))  # Increased height
    
    # Create grid for subplots with extra bottom space
    gs = GridSpec(2, 3, figure=fig, bottom=0.20)  # Leave 20% space at bottom
    
    # Create meshgrid for plotting
    x = np.linspace(0, 1, phi.shape[0])
    y = np.linspace(0, 1, phi.shape[1])
    X, Y = np.meshgrid(x, y)
    
    # Plot 1: Phase Field
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(phi.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    ax1.contour(X, Y, phi.T, levels=[0], colors='r', linewidths=2)
    fig.colorbar(im1, ax=ax1, label='Phase Value')
    ax1.set_title('Phase Field')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    
    # Plot 2: Surface Tension Force
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(ST_magnitude.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    ax2.contour(X, Y, phi.T, levels=[0], colors='r', linewidths=2)
    fig.colorbar(im2, ax=ax2, label='Force Magnitude')
    ax2.set_title('Surface Tension Force')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    
    # Plot 3: Velocity Streamlines
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(U_magnitude.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    ax3.contour(X, Y, phi.T, levels=[0], colors='r', linewidths=2)
    # Add streamlines
    ax3.streamplot(x, y, U[..., 0].T, U[..., 1].T, density=1.5, color='white')
    fig.colorbar(im3, ax=ax3, label='Speed')
    ax3.set_title('Velocity Field Streamlines')
    ax3.set_xlabel('X-axis')
    ax3.set_ylabel('Y-axis')
    
    # Plot 4: Velocity Magnitude
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(U_magnitude.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    ax4.contour(X, Y, phi.T, levels=[0], colors='r', linewidths=2)
    fig.colorbar(im4, ax=ax4, label='Speed')
    ax4.set_title('Velocity Magnitude')
    ax4.set_xlabel('X-axis')
    ax4.set_ylabel('Y-axis')
    
    # Plot 5: Pressure Field
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(-P.T, origin='lower', extent=[0, 1, 0, 1], cmap='coolwarm')
    ax5.contour(X, Y, phi.T, levels=[0], colors='k', linewidths=2)
    fig.colorbar(im5, ax=ax5, label='Pressure')
    ax5.set_title('Pressure Field')
    ax5.set_xlabel('X-axis')
    ax5.set_ylabel('Y-axis')
    
    # Plot 6: Pressure Gradient Magnitude
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(rho.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    ax6.contour(X, Y, phi.T, levels=[0], colors='k', linewidths=2)
    fig.colorbar(im6, ax=ax6, label='Density')
    ax6.set_title('Density')
    ax6.set_xlabel('X-axis')
    ax6.set_ylabel('Y-axis')
    
    # Add simulation information
    plt.figtext(0.5, 0.12, f'Time Step: {step}\nSimulation Time: {step * dt:.5f}', ha='center')
    
    # Add phase field statistics - moved higher to avoid overlap
    stats_text = f"Phase Field (phi):\n  Min: {phi.min():.5f}\n  Max: {phi.max():.5f}\n  Mean: {phi.mean():.5f}\n  Mass: {phi.sum():.5f}\n"
    stats_text += f"Velocity (U):\n  Min x: {U[..., 0].min():.5f}\n  Max x: {U[..., 0].max():.5f}\n  Min y: {U[..., 1].min():.5f}\n  Max y: {U[..., 1].max():.5f}\n  Max Speed: {U_magnitude.max():.5f}\n"
    stats_text += f"Pressure (P):\n  Min: {P.min():.5f}\n  Max: {P.max():.5f}\n  Mean: {P.mean():.5f}\n"
    stats_text += f"Surface Tension:\n  Max Magnitude: {ST_magnitude.max():.5f}\n"
    stats_text += f"Droplet mass: {mass:.5f}"
    
    plt.figtext(0.01, 0.5, stats_text, fontsize=10, verticalalignment='bottom')
    
    fig.tight_layout(rect=[0, 0.20, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_checkpoint(step, phi, U, P, directory="checkpoints"):
    """Save current simulation state to checkpoint files."""
    os.makedirs(directory, exist_ok=True)
    
    # Create checkpoint filename with step number
    checkpoint_path = os.path.join(directory, f"checkpoint_{step:06d}")
    
    # Save arrays
    np.savez_compressed(
        checkpoint_path, 
        step=step,
        phi=phi, 
        U=U, 
        P=P,
        timestamp=datetime.now().isoformat()
    )
    
    print(f"Checkpoint saved: {checkpoint_path}.npz")
    return checkpoint_path + ".npz"

def load_checkpoint(checkpoint_path):
    """Load simulation state from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load data
    data = np.load(checkpoint_path, allow_pickle=True)
    
    # Return as dictionary
    return {
        'step': int(data['step']),
        'phi': data['phi'],
        'U': data['U'],
        'P': data['P'],
        'timestamp': data['timestamp']
    }

def list_checkpoints(directory="checkpoints"):
    """List available checkpoint files."""
    if not os.path.exists(directory):
        return []
    
    checkpoints = []
    for filename in os.listdir(directory):
        if filename.startswith("checkpoint_") and filename.endswith(".npz"):
            checkpoints.append(os.path.join(directory, filename))
    
    return sorted(checkpoints)

def load_config(config_path=None):
    """
    Load configuration from a JSON file or use defaults.
    
    Args:
        config_path (str, optional): Path to the JSON configuration file.
        
    Returns:
        dict: Configuration parameters.
    """
    # Default configuration
    config = {
        "physical_params": {
            "rho": 1.0,
            "Re1": 1000.0,
            "Re2": 10.0,
            "We": 10.0,
            "Pe": 1.0,
            "epsilon": 0.05,
            "alpha": 1.0,
            "phase_penalty": 1000.0,
            "contact_angle": 120
        },
        "grid_params": {
            "Lx": 1.0,
            "Ly": 1.0,
            "Nx": 100,
            "Ny": 100
        },
        "time_params": {
            "dt": 0.001,
            "t_max": 1.0,
            "checkpoint_interval": 50,
            "dt_initial": 0.0005
        },
        "initial_conditions": {
            "droplet_radius": 0.2
        },
        "restart": {
            "restart_from": None
        }
    }
    
    # Load configuration from file if provided
    if config_path:
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                
            # Update default config with loaded values (recursive update)
            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        update_dict(d[k], v)
                    else:
                        d[k] = v
            
            update_dict(config, loaded_config)
            sys.stdout.write(f"Configuration loaded from {config_path}")
        except Exception as e:
            sys.stdout.write(f"Error loading config file: {e}")
            sys.stdout.write("Using default configuration")
    else:
        sys.stdout.write("No config file provided. Using default configuration.")
    
    return config

