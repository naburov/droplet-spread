"""
Time integration methods for numerical simulation.

This module contains time stepping methods and stability criteria
for the droplet spreading simulation.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit


def cfl_dt(u_max, v_max, dx, dy, C=0.4):
    """Return Δt that gives desired CFL=C.
    
    Args:
        u_max (float): Maximum velocity in x-direction.
        v_max (float): Maximum velocity in y-direction.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        C (float): CFL number (default: 0.4).
    
    Returns:
        float: Time step satisfying CFL condition.
    """
    return C / (abs(u_max)/dx + abs(v_max)/dy)


@jit
def jax_cfl_dt(u_max, v_max, dx, dy, C=0.4):
    """JAX-compiled version of CFL time step calculation.
    
    Args:
        u_max (float): Maximum velocity in x-direction.
        v_max (float): Maximum velocity in y-direction.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        C (float): CFL number (default: 0.4).
    
    Returns:
        float: Time step satisfying CFL condition.
    """
    return C / (jnp.abs(u_max)/dx + jnp.abs(v_max)/dy)


def explicit_euler_step(field, rhs, dt):
    """Perform one step of explicit Euler time integration.
    
    Args:
        field (np.ndarray): Current field value.
        rhs (np.ndarray): Right-hand side (time derivative).
        dt (float): Time step.
    
    Returns:
        np.ndarray: Updated field value.
    """
    return field + dt * rhs


@jit
def jax_explicit_euler_step(field, rhs, dt):
    """JAX-compiled version of explicit Euler time integration.
    
    Args:
        field (jnp.ndarray): Current field value.
        rhs (jnp.ndarray): Right-hand side (time derivative).
        dt (float): Time step.
    
    Returns:
        jnp.ndarray: Updated field value.
    """
    return field + dt * rhs


def adaptive_time_step(field, rhs, dt, tolerance=1e-6):
    """Adaptive time step based on field changes.
    
    Args:
        field (np.ndarray): Current field value.
        rhs (np.ndarray): Right-hand side (time derivative).
        dt (float): Current time step.
        tolerance (float): Maximum allowed relative change.
    
    Returns:
        float: Adjusted time step.
    """
    max_change = np.max(np.abs(rhs * dt))
    max_field = np.max(np.abs(field))
    
    if max_field > 0:
        relative_change = max_change / max_field
        if relative_change > tolerance:
            return dt * tolerance / relative_change
    
    return dt
