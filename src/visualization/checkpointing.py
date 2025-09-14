"""
Checkpointing functionality for droplet spreading simulation.

This module contains functions for saving and loading simulation state.
"""

import os
import numpy as np
from datetime import datetime


def save_checkpoint(step, phi, U, P, directory="checkpoints"):
    """Save current simulation state to checkpoint files.
    
    Args:
        step (int): Current simulation step.
        phi (np.ndarray): Phase field.
        U (np.ndarray): Velocity field.
        P (np.ndarray): Pressure field.
        directory (str): Directory to save checkpoints.
    
    Returns:
        str: Path to saved checkpoint file.
    """
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
    return {
        'step': int(data['step']),
        'phi': data['phi'],
        'U': data['U'],
        'P': data['P'],
        'timestamp': data['timestamp']
    }


def list_checkpoints(directory="checkpoints"):
    """List available checkpoint files.
    
    Args:
        directory (str): Directory containing checkpoints.
    
    Returns:
        list: List of checkpoint file paths.
    """
    if not os.path.exists(directory):
        return []
    
    checkpoints = []
    for filename in os.listdir(directory):
        if filename.startswith("checkpoint_") and filename.endswith(".npz"):
            checkpoints.append(os.path.join(directory, filename))
    
    return sorted(checkpoints)
