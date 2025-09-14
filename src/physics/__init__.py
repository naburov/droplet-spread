"""
Physics module for droplet spreading simulation.

Contains all physics-related functionality including:
- Phase field equations
- Fluid dynamics
- Surface tension
- Material properties
"""

from physics.phase_field import (
    PhaseFieldSolver,
    apply_contact_angle_boundary_conditions,
    jax_apply_contact_angle_boundary_conditions
)

from physics.fluid_dynamics import (
    FluidDynamicsSolver,
    update_velocity,
    apply_velocity_boundary_conditions,
    check_continuity
)

from physics.surface_tension import (
    SurfaceTensionSolver,
    surface_tension_force,
    jax_surface_tension_force,
    apply_surface_tension_boundary_conditions,
    jax_apply_surface_tension_boundary_conditions
)

from physics.properties import (
    calculate_density,
    calculate_reynolds_number,
    calculate_weber_number,
    jax_calculate_density,
    jax_calculate_reynolds_number,
    jax_calculate_weber_number
)

__all__ = [
    # Phase field
    'PhaseFieldSolver',
    'apply_contact_angle_boundary_conditions',
    'jax_apply_contact_angle_boundary_conditions',
    
    # Fluid dynamics
    'FluidDynamicsSolver',
    'update_velocity',
    'apply_velocity_boundary_conditions',
    'check_continuity',
    
    # Surface tension
    'SurfaceTensionSolver',
    'surface_tension_force',
    'jax_surface_tension_force',
    'apply_surface_tension_boundary_conditions',
    'jax_apply_surface_tension_boundary_conditions',
    
    # Properties
    'calculate_density',
    'calculate_reynolds_number',
    'calculate_weber_number',
    'jax_calculate_density',
    'jax_calculate_reynolds_number',
    'jax_calculate_weber_number'
]
