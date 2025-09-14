import os
import numpy as np
import matplotlib.pyplot as plt
import json
from glob import glob
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

def load_checkpoint(checkpoint_path):
    """Load data from a checkpoint file."""
    data = np.load(checkpoint_path)
    return {
        'phi': data['phi'],
        # 'time': data['time'].item(),  # Convert 0-d array to scalar
        'step': data['step'].item()
    }

def find_interface(phi, level=0.0):
    """Find the interface points where phi crosses zero."""
    # Use marching squares to find contours
    contours = plt.contour(phi.T, levels=[level])
    plt.clf()  # Clear the figure
    # Get the contour vertices
    vertices = contours.allsegs[0]
    return vertices

def plot_interfaces(checkpoint_dir, output_dir, config_path=None, num_frames=None, ymin=None, ymax=None):
    """Plot the evolution of interfaces from checkpoints."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration if available
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Get all checkpoint files sorted by step number
    checkpoint_files = sorted(
        glob(os.path.join(checkpoint_dir, 'checkpoint_*.npz')),
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    
    # If num_frames is specified, select evenly spaced frames
    if num_frames is not None and num_frames < len(checkpoint_files):
        indices = np.linspace(0, len(checkpoint_files)-1, num_frames, dtype=int)
        checkpoint_files = [checkpoint_files[i] for i in indices]
    
    # Load first checkpoint to get dimensions
    first_cp = load_checkpoint(checkpoint_files[0])
    phi = first_cp['phi']
    Nx, Ny = phi.shape
    
    # Get domain size from config or default to unit square
    Lx = config.get('Lx', 1.0)
    Ly = config.get('Ly', 1.0)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot interfaces for each checkpoint
    colors = plt.cm.viridis(np.linspace(0, 1, len(checkpoint_files)))
    
    # Add surface line at y=0
    plt.axhline(y=0, color='black', linewidth=2, label='Surface')
    
    for i, (checkpoint_file, color) in enumerate(zip(checkpoint_files, colors)):
        data = load_checkpoint(checkpoint_file)
        phi = data['phi']
        step = data['step']
        
        # Plot filled contour for the phase field
        plt.contour(X, Y, phi, levels=[0], colors=[color], linewidths=0.5,
                    linestyles='--',
                   label=f'step = {step}')
        plt.text(0.02, 0.9 - i*0.075, f'step = {step}', 
                transform=plt.gca().transAxes, color=color)
    
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interface Evolution')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Set y-axis limits to start exactly at 0
    ymax = plt.gca().get_ylim()[1]
    # plt.ylim(0, 0.3)
    # plt.xlim(0.2, 0.8)
    plt.ylim(0, 0.6)
    plt.xlim(0.0, 1.0)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'interface_evolution.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Droplet Spreading Simulation')
    
    def animate(frame):
        data = load_checkpoint(checkpoint_files[frame])
        phi = data['phi']
        step = data['step']
        
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        
        # Plot phase field
        im1 = ax1.imshow(phi.T, origin='lower', extent=[0, Lx, 0, Ly],
                        cmap='RdBu', vmin=-1, vmax=1)
        ax1.set_title(f'Phase Field (step = {step})')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        # Add surface line
        ax1.axhline(y=0, color='black', linewidth=2)
        
        # Set y-axis limits to start at 0
        ax1.set_ylim(0, Ly)
        
        # Plot interface
        ax2.contour(X, Y, phi, levels=[0], colors=['k'], linewidths=2)
        ax2.set_title(f'Interface (step = {step})')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # Add surface line
        ax2.axhline(y=0, color='black', linewidth=2)
        
        # Set y-axis limits to start at 0
        ax2.set_ylim(0, Ly)
        
        return im1,
    
    anim = FuncAnimation(fig, animate, frames=len(checkpoint_files),
                        interval=200, blit=False)
    
    # Save animation
    anim.save(os.path.join(output_dir, 'interface_evolution.gif'),
              writer='pillow', fps=5)
    plt.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot interface evolution from checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoint files')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Directory to save plots')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to simulation configuration file')
    parser.add_argument('--num_frames', type=int, default=None,
                       help='Number of frames to plot (default: all)')
    parser.add_argument('--ymin', type=float, default=None,
                       help='Minimum y-value for plotting')
    parser.add_argument('--ymax', type=float, default=None,
                       help='Maximum y-value for plotting')
    
    args = parser.parse_args()
    
    plot_interfaces(args.checkpoint_dir, args.output_dir, 
                   args.config, args.num_frames,
                   args.ymin, args.ymax) 