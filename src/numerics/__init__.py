"""
Numerical methods module for droplet spreading simulation.

Contains all numerical methods including:
- Finite difference schemes
- Poisson equation solvers
- Time integration methods
"""

from .finite_differences import (
    numerical_derivative,
    gradient,
    divergence,
    laplacian,
    norm,
    jax_dx,
    jax_dy,
    jax_gradient,
    jax_divergence,
    jax_laplacian,
    jax_norm
)

from .poisson_solvers import (
    solve_poisson,
    solve_poisson_with_better_bc,
    solve_poisson_pyamg,
    solve_poisson_pyro,
    build_2d_laplacian_matrix_with_variable_steps,
    jax_build_2d_laplacian_matrix_with_variable_steps
)

from .time_integration import (
    cfl_dt,
    explicit_euler_step
)

__all__ = [
    # Finite differences
    'numerical_derivative',
    'gradient',
    'divergence', 
    'laplacian',
    'norm',
    'jax_dx',
    'jax_dy',
    'jax_gradient',
    'jax_divergence',
    'jax_laplacian',
    'jax_norm',
    
    # Poisson solvers
    'solve_poisson',
    'solve_poisson_with_better_bc',
    'solve_poisson_pyamg',
    'solve_poisson_pyro',
    'build_2d_laplacian_matrix_with_variable_steps',
    'jax_build_2d_laplacian_matrix_with_variable_steps',
    
    # Time integration
    'cfl_dt',
    'explicit_euler_step'
]
