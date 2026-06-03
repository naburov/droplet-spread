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
        float: Time step satisfying CFL condition, or inf if velocities are zero.
    """
    denominator = abs(u_max)/dx + abs(v_max)/dy
    if denominator < 1e-10:  # Avoid division by zero
        return np.inf
    return C / denominator


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
        float: Time step satisfying CFL condition, or inf if velocities are zero.
    """
    denominator = jnp.abs(u_max)/dx + jnp.abs(v_max)/dy
    # Avoid division by zero - return large value if velocities are zero
    return jnp.where(denominator < 1e-10, jnp.inf, C / denominator)


@jit
def jax_capillary_cfl_dt(surface_tension_max, rho, epsilon, dx, dy, C=0.4):
    """JAX-compiled version of capillary CFL time step calculation.
    
    Capillary waves have a characteristic speed: c_cap = sqrt(σ * κ / ρ)
    where σ is surface tension, κ is curvature, ρ is density.
    
    For phase field method, the capillary time scale is approximately:
    τ_cap ~ sqrt(ρ * ε^3 / σ) where ε is interface thickness.
    
    More directly, we can use: dt < C * min(dx, dy) / c_cap
    
    Args:
        surface_tension_max (float): Maximum surface tension force magnitude.
        rho (float): Characteristic density.
        epsilon (float): Interface thickness parameter.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        C (float): CFL number (default: 0.4).
    
    Returns:
        float: Time step satisfying capillary CFL condition.
    """
    # Capillary wave speed estimate: c_cap ~ sqrt(surface_tension / (rho * epsilon))
    # More conservative: use surface_tension_max directly
    if surface_tension_max > 0:
        # Characteristic capillary speed
        c_cap = jnp.sqrt(surface_tension_max / (rho * epsilon + 1e-10))
        # Use minimum grid spacing
        h_min = jnp.minimum(dx, dy)
        # Capillary CFL condition
        dt_cap = C * h_min / (c_cap + 1e-10)
        return dt_cap
    else:
        return jnp.inf


def capillary_cfl_dt(surface_tension_max, rho, epsilon, dx, dy, C=0.4):
    """Capillary CFL time step calculation.
    
    Args:
        surface_tension_max (float): Maximum surface tension force magnitude.
        rho (float): Characteristic density.
        epsilon (float): Interface thickness parameter.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        C (float): CFL number (default: 0.4).
    
    Returns:
        float: Time step satisfying capillary CFL condition.
    """
    if surface_tension_max > 0:
        c_cap = np.sqrt(surface_tension_max / (rho * epsilon + 1e-10))
        h_min = min(dx, dy)
        dt_cap = C * h_min / (c_cap + 1e-10)
        return dt_cap
    else:
        return np.inf


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


def curvature_cfl_dt(curvature_max, dx, dy, C=0.1):
    """Curvature-based CFL time step.
    
    High curvature leads to high surface tension forces and can cause instability.
    This CFL condition limits dt based on maximum curvature to prevent
    force singularities from destabilizing the simulation.
    
    The idea: surface tension force ~ σ * κ, acceleration ~ σ * κ / ρ
    Velocity change per step: Δu ~ (σ * κ / ρ) * dt
    For stability: Δu * dt / dx < C, so dt < C * dx * ρ / (σ * κ)
    
    Simplified form (ignoring material properties that are constant):
    dt < C * min(dx, dy) / κ_max
    
    Args:
        curvature_max (float): Maximum curvature magnitude.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        C (float): CFL number (default: 0.1, conservative).
    
    Returns:
        float: Time step satisfying curvature CFL condition.
    """
    if curvature_max > 1e-10:
        h_min = min(dx, dy)
        return C * h_min / curvature_max
    else:
        return np.inf


@jit
def jax_curvature_cfl_dt(curvature_max, dx, dy, C=0.1):
    """JAX-compiled curvature-based CFL time step.
    
    Args:
        curvature_max (float): Maximum curvature magnitude.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        C (float): CFL number (default: 0.1, conservative).
    
    Returns:
        float: Time step satisfying curvature CFL condition.
    """
    h_min = jnp.minimum(dx, dy)
    return jnp.where(curvature_max > 1e-10, C * h_min / curvature_max, jnp.inf)
