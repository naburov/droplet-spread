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
            "rho1": 1000.0,
            "rho2": 1.204,
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
            "g": 9.81,
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
            sys.stdout.write(f"Configuration loaded from {config_path}\n")
        except Exception as e:
            sys.stdout.write(f"Error loading config file: {e}\n")
            sys.stdout.write("Using default configuration\n")
    else:
        sys.stdout.write("No config file provided. Using default configuration.\n")
    
    return config


def save_config(config, config_path):
    """
    Save configuration to a JSON file.
    
    Args:
        config (dict): Configuration parameters.
        config_path (str): Path to save the configuration file.
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to {config_path}")
