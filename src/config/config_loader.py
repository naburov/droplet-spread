"""
Configuration loader for droplet spreading simulation.

This module contains functions for loading and managing simulation configuration.
"""

import json
import sys


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
            "rho1": 0.001204,
            "rho2": 1.0,
            "Re1": 1000.0,
            "Re2": 10.0,
            "We1": 10.0,
            "We2": 10.0,
            "Pe": 1.0,
            "epsilon": 0.05,
            "alpha": 1.0,
            "phase_penalty": 1000.0,
            "contact_angle": 120,
            "include_gravity": False,
            "g": -1.0,
            "atm_pressure": 0.0,
            "Fr": 1.0
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
            _normalize_flat_bcs(config)
            sys.stdout.write(f"Configuration loaded from {config_path}\n")
        except Exception as e:
            sys.stdout.write(f"Error loading config file: {e}\n")
            sys.stdout.write("Using default configuration\n")
    else:
        sys.stdout.write("No config file provided. Using default configuration.\n")
    
    return config


def _normalize_flat_bcs(config):
    """Build nested boundary_conditions from flat keys (top_bc, bottom_bc, outflow_right, etc.)."""
    if config.get("boundary_conditions") is not None:
        return
    top_bc = config.get("top_bc")
    bottom_bc = config.get("bottom_bc")
    left_bc = config.get("left_bc")
    right_bc = config.get("right_bc")
    outflow_right = config.get("outflow_right", True)
    inlet_profile = config.get("inlet_profile")
    if top_bc is None and bottom_bc is None and left_bc is None and right_bc is None and not inlet_profile:
        return
    # Left: inlet if inlet_profile set, else no_slip for channel
    left_vel = "dirichlet" if inlet_profile else (left_bc or "no_slip")
    right_vel = "do_nothing" if outflow_right else (right_bc or "no_slip")
    config["boundary_conditions"] = {
        "velocity": {
            "top": top_bc or "no_slip",
            "bottom": bottom_bc or "no_slip",
            "left": left_vel,
            "right": right_vel,
        },
        "pressure": {
            "top": "neumann",
            "bottom": "neumann",
            "left": "neumann",
            "right": "open" if outflow_right else "neumann",
            "open_pressure": 0.0,
        },
    }