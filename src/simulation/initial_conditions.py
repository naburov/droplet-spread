"""
Initial condition setup for droplet spreading simulation.

This module contains functions for setting up initial conditions
for the droplet spreading simulation.
"""

import numpy as np
import jax.numpy as jnp


def initialize_phase(Nx, Ny, radius):
    """Initialize the phase field with a semicircle droplet resting on the bottom boundary.
    
    Args:
        Nx (int): Number of grid points in x-direction.
        Ny (int): Number of grid points in y-direction.
        radius (float): Droplet radius.
    
    Returns:
        np.ndarray: Initial phase field.
    """
    # Create a grid of coordinates
    x = np.linspace(0, 1, Nx)  # X coordinates from 0 to 1
    y = np.linspace(0, 1, Ny)   # Y coordinates from 0 to 1
    X, Y = np.meshgrid(x, y, indexing='ij')  # Create a meshgrid with correct indexing
    
    # Define the center of the semicircle at the bottom center of the domain
    center_x = 0.5
    center_y = 0  # Bottom of the domain
    
    # Calculate distance from the center
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Use tanh to create a smooth transition between phases
    # Positive inside the semicircle (phi = 1), negative outside (phi = -1)
    phi = -np.tanh((distance - radius) * (10/radius))  # Scale factor for sharpness
    
    # Apply boundary condition to ensure semicircle sits exactly on the bottom boundary
    phi[:,0] = phi[:,1]  # Copy the first interior row to the boundary
    
    return phi


def get_borders_of_droplet(phi):
    """Get the borders of the droplet.
    
    Args:
        phi (np.ndarray): Phase field.
    
    Returns:
        tuple: (start_of_droplet, end_of_droplet) indices.
    """
    # Find the first and last non-zero elements in each row
    start_of_droplet = 0
    end_of_droplet = phi.shape[0] - 1
    for i in range(0, phi.shape[0]):
        if phi[i, 0] > 0:
            start_of_droplet = i
            break
    for i in range(phi.shape[0] - 1, 0, -1):
        if phi[i, 0] > 0:
            end_of_droplet = i
            break
    return start_of_droplet, end_of_droplet
