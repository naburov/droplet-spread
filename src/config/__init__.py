"""
Configuration module for droplet spreading simulation.

Contains configuration loading and management functionality.
"""

from .config_loader import load_config
from .bc_compatibility import check_bc_compatibility

__all__ = ['load_config', 'check_bc_compatibility']
