"""
Helper functions for droplet spreading simulation.

This module contains general utility functions.
"""

import os
from datetime import datetime
import numpy as np


def calculate_non_dimensional_params(params_dict):
    """Calculate non-dimensional parameters from physical parameters.
    
    Args:
        params_dict (dict): Dictionary containing physical parameters.
    
    Returns:
        dict: Non-dimensional parameters.
    """
    rho = params_dict["rho"]
    mu = params_dict["mu"]
    sigma = params_dict["sigma"]
    g = params_dict["g"]
    velocity = params_dict["velocity"]
    length = params_dict["length"]
    diffusivity = params_dict["diffusivity"]

    Re = rho * velocity * length / mu
    We = rho * velocity**2 * length / sigma
    Pe = velocity * length / diffusivity
    Fr = velocity / np.sqrt(g * length)
    
    print(f"Re: {Re:.8f}, We: {We:.8f}, Pe: {Pe:.8f}, Fr: {Fr:.8f}")
    return {
        "Re": Re,
        "We": We,
        "Pe": Pe,
        "Fr": Fr
    }


def create_experiment_directory(output_dir=None):
    """Create a directory for the experiment.
    
    Args:
        output_dir (str, optional): Specific output directory.
    
    Returns:
        str: Path to the experiment directory.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = f"experiment_{timestamp}"
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir
