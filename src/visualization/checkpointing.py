"""
Checkpointing functionality for droplet spreading simulation.

This module contains functions for saving and loading simulation state.

Curvilinear (x, eta): no solid-cell masking. Velocity and advection BCs apply at boundaries
only; checkpoints contain full grid (phi, U, P) without overwriting "solid" regions.

Origin in checkpoint: phi, U, P use shape (Nx, Ny) with index [i, j]; i=0 is x=0 (left),
j=0 is eta=0 (bottom). So origin (0, 0) is bottom-left: phi[0,0], U[0,0,:] = (x=0, eta=0).

Plotting from a checkpoint: grid (Nx, Ny, Lx, Ly, dx, dy) and geometry are NOT in the file;
they must come from config (e.g. simulation_parameters.json). Always use the config from
the same experiment as the checkpoint, and validate phi.shape == (config Nx, config Ny).
"""

import os
import numpy as np
from datetime import datetime


def save_checkpoint(step, phi, U, P, directory="checkpoints", psi=None, T=None, u_face=None, v_face=None, t=None, dt=None):
    """Save current simulation state to checkpoint files.

    Saves the full grid (phi, U, P) in curvilinear (x, eta); no solid-cell masking.

    Args:
        step (int): Current simulation step.
        phi (np.ndarray): Phase field.
        U (np.ndarray): Velocity field.
        P (np.ndarray): Pressure field.
        directory (str): Directory to save checkpoints.
        psi (np.ndarray, optional): Ice phase field.
        T (np.ndarray, optional): Temperature field.
        u_face (np.ndarray, optional): Staggered u on x-faces (when velocity_layout == 'staggered').
        v_face (np.ndarray, optional): Staggered v on y-faces (when velocity_layout == 'staggered').

    Returns:
        str: Path to saved checkpoint file.
    """
    os.makedirs(directory, exist_ok=True)
    
    # Create checkpoint filename with step number
    checkpoint_path = os.path.join(directory, f"checkpoint_{step:06d}")
    
    # Prepare data dictionary. coords: 'x_eta' = terrain-following (x, eta); plot uses rectangular mesh.
    data_dict = {
        'step': step,
        'phi': phi,
        'U': U,
        'P': P,
        'timestamp': datetime.now().isoformat(),
        'coords': np.array('x_eta', dtype=object),
    }
    if t is not None:
        data_dict['t'] = float(t)
    if dt is not None:
        data_dict['dt'] = float(dt)
    
    # Add ice-water fields if provided
    if psi is not None:
        data_dict['psi'] = psi
    if T is not None:
        data_dict['T'] = T
    # Staggered velocity faces (for velocity_layout == 'staggered')
    if u_face is not None:
        data_dict['u_face'] = np.asarray(u_face)
    if v_face is not None:
        data_dict['v_face'] = np.asarray(v_face)
    
    # Save arrays
    np.savez_compressed(checkpoint_path, **data_dict)
    
    print(f"Checkpoint saved: {checkpoint_path}.npz")
    return checkpoint_path + ".npz"


def load_checkpoint(checkpoint_path):
    """Load simulation state from checkpoint file.
    
    Args:
        checkpoint_path (str): Path to checkpoint file.
    
    Returns:
        dict: Loaded simulation state.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load data
    data = np.load(checkpoint_path, allow_pickle=True)
    
    # Return as dictionary
    checkpoint = {
        'step': int(data['step']),
        'phi': data['phi'],
        'U': data['U'],
        'P': data['P'],
        'timestamp': data['timestamp'],
        'coords': str(data['coords'].item()) if 'coords' in data else 'x_y',
    }
    if 't' in data:
        checkpoint['t'] = float(data['t'])
    if 'dt' in data:
        checkpoint['dt'] = float(data['dt'])

    # Load ice-water fields if present
    if 'psi' in data:
        checkpoint['psi'] = data['psi']
    if 'T' in data:
        checkpoint['T'] = data['T']
    # Staggered velocity faces (for velocity_layout == 'staggered')
    if 'u_face' in data:
        checkpoint['u_face'] = data['u_face']
    if 'v_face' in data:
        checkpoint['v_face'] = data['v_face']
    
    return checkpoint

