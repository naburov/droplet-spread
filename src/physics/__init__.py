"""
Physics module for droplet spreading simulation.
"""

from physics.phase_field import (
    BasePhaseFieldSolver,
    PhaseFieldSolver,
    PhaseFieldSolverSimple,
    PhaseFieldSolverGhostCell,
)
from physics.fluid_dynamics import FluidDynamicsSolver, jax_update_velocity, jax_check_continuity
from physics.surface_tension import SurfaceTensionSolver, jax_surface_tension_force, jax_curvature, jax_curvature_stats
from physics.properties import (
    calculate_density, calculate_reynolds_number, calculate_weber_number,
    jax_calculate_density, jax_calculate_reynolds_number, jax_calculate_weber_number, 
    jax_df_2, jax_advection_function
)
from physics.pressure import PressureSolver
from physics.ice_phase_field import IcePhaseFieldSolver
from physics.temperature import TemperatureSolver

__all__ = [
    # Solvers
    'BasePhaseFieldSolver',
    'PhaseFieldSolver',
    'PhaseFieldSolverSimple',
    'PhaseFieldSolverGhostCell',
    'FluidDynamicsSolver',
    'SurfaceTensionSolver',
    'PressureSolver',
    'IcePhaseFieldSolver',
    'TemperatureSolver',
    # JAX functions
    'jax_update_velocity',
    'jax_check_continuity',
    'jax_surface_tension_force',
    'jax_curvature',
    'jax_curvature_stats',
    'jax_calculate_density',
    'jax_calculate_reynolds_number',
    'jax_calculate_weber_number',
    'jax_df_2',
    'jax_advection_function',
    # NumPy functions (for compatibility)
    'calculate_density',
    'calculate_reynolds_number',
    'calculate_weber_number',
]
