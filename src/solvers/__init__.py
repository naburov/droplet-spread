"""
Solver modules for droplet spreading simulation.

Contains linear system solvers and projection methods.
"""

from .sparse_solver import SparseSolverWrapper
from .ppe import ppe_solve
from .ppe_utils import correction_step, check_divergence, apply_pressure_correction

# Backward compatibility aliases
ppe = ppe_solve
ppe_global = ppe_solve

__all__ = [
    'SparseSolverWrapper',
    'ppe_solve',
    'ppe',
    'ppe_global',
    'correction_step',
    'check_divergence',
    'apply_pressure_correction'
]
