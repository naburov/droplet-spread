"""
Solver modules for droplet spreading simulation.

Contains linear system solvers and projection methods.
"""

from .sparse_solver import SparseSolverWrapper
from .projection_methods import (
    correction_step,
    ppe,
    damp_divergence,
    jax_correction_step,
    jax_ppe,
    jax_damp_divergence
)

__all__ = [
    'SparseSolverWrapper',
    'correction_step',
    'ppe', 
    'damp_divergence',
    'jax_correction_step',
    'jax_ppe',
    'jax_damp_divergence'
]
