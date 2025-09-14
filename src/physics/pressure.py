"""
Pressure field calculations for droplet spreading simulation.

This module contains functions for updating the pressure field
based on surface tension and other forces.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from numerics.finite_differences import divergence, jax_divergence
from physics.properties import calculate_density, jax_calculate_density


def update_pressure(surface_tension, Nx, Ny, dx, dy, rho1, rho2, phi, g, atm_pressure, pressure_solver):
    """Update the pressure field P based on the velocity field U and phase field phi.
    
    Args:
        surface_tension (np.ndarray): Surface tension force (shape: (Nx, Ny, 2)).
        Nx (int): Number of grid points in x-direction.
        Ny (int): Number of grid points in y-direction.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        rho1 (float): Density of phase 1.
        rho2 (float): Density of phase 2.
        phi (np.ndarray): Phase field.
        g (float): Gravitational acceleration.
        atm_pressure (float): Atmospheric pressure.
        pressure_solver: Solver for pressure equation.
    
    Returns:
        np.ndarray: Updated pressure field.
    """
    # Calculate divergence of surface tension force
    sf_grad = divergence(surface_tension, dx, dy)
    rho = calculate_density(phi, rho1, rho2)

    # Apply boundary conditions
    sf_grad[:, 0] = np.sum(rho * g * dy, axis=1) + atm_pressure
    sf_grad[:, -1] = atm_pressure

    # Solve for pressure
    pressure_solver.set_rhs(sf_grad)
    pressure_solver.solve()
    P = pressure_solver.get_solution()

    return P


@jit
def jax_update_pressure(surface_tension, Nx, Ny, dx, dy, rho1, rho2, phi, g, atm_pressure):
    """JAX-compiled version of pressure update.
    
    Args:
        surface_tension (jnp.ndarray): Surface tension force (shape: (Nx, Ny, 2)).
        Nx (int): Number of grid points in x-direction.
        Ny (int): Number of grid points in y-direction.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        rho1 (float): Density of phase 1.
        rho2 (float): Density of phase 2.
        phi (jnp.ndarray): Phase field.
        g (float): Gravitational acceleration.
        atm_pressure (float): Atmospheric pressure.
    
    Returns:
        jnp.ndarray: Updated pressure field.
    """
    # Calculate divergence of surface tension force
    sf_grad = jax_divergence(surface_tension, dx, dy)
    rho = jax_calculate_density(phi, rho1, rho2)

    # Apply boundary conditions
    sf_grad = sf_grad.at[:, 0].set(jnp.sum(rho * g * dy, axis=1) + atm_pressure)
    sf_grad = sf_grad.at[:, -1].set(atm_pressure)

    # For JAX version, we would need a JAX-compatible solver
    # This is a placeholder - would need proper implementation
    P = jnp.zeros_like(sf_grad)

    return P


def update_pressure_jax(surface_tension, Nx, Ny, dx, dy, rho1, rho2, phi, g, atm_pressure, pressure_solver):
    """JAX version of pressure update that works with the existing solver.
    
    Args:
        surface_tension (jnp.ndarray): Surface tension force (shape: (Nx, Ny, 2)).
        Nx (int): Number of grid points in x-direction.
        Ny (int): Number of grid points in y-direction.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        rho1 (float): Density of phase 1.
        rho2 (float): Density of phase 2.
        phi (jnp.ndarray): Phase field.
        g (float): Gravitational acceleration.
        atm_pressure (float): Atmospheric pressure.
        pressure_solver: Solver for pressure equation.
    
    Returns:
        jnp.ndarray: Updated pressure field.
    """
    # Calculate divergence of surface tension force using JAX
    sf_grad = jax_divergence(surface_tension, dx, dy)
    rho = jax_calculate_density(phi, rho1, rho2)

    # Apply boundary conditions using JAX syntax
    sf_grad = sf_grad.at[:, 0].set(jnp.sum(rho * g * dy, axis=1) + atm_pressure)
    sf_grad = sf_grad.at[:, -1].set(atm_pressure)

    # Convert to NumPy for solver, then back to JAX
    sf_grad_np = np.array(sf_grad)
    pressure_solver.set_rhs(sf_grad_np)
    pressure_solver.solve()
    P_np = pressure_solver.get_solution()
    P = jnp.array(P_np)

    return P


class PressureSolver:
    """Pressure solver class for droplet spreading simulation."""
    
    def __init__(self, rho1, rho2, g, atm_pressure):
        """Initialize the pressure solver.
        
        Args:
            rho1 (float): Density of phase 1.
            rho2 (float): Density of phase 2.
            g (float): Gravitational acceleration.
            atm_pressure (float): Atmospheric pressure.
        """
        self.rho1 = rho1
        self.rho2 = rho2
        self.g = g
        self.atm_pressure = atm_pressure
    
    def update_pressure(self, surface_tension, Nx, Ny, dx, dy, phi, pressure_solver, use_jax=False):
        """Update the pressure field.
        
        Args:
            surface_tension: Surface tension force array.
            Nx: Number of grid points in x-direction.
            Ny: Number of grid points in y-direction.
            dx: Grid spacing in x-direction.
            dy: Grid spacing in y-direction.
            phi: Phase field array.
            pressure_solver: Solver for pressure equation.
            use_jax: Whether to use JAX-compiled functions.
        
        Returns:
            Updated pressure field.
        """
        if use_jax:
            return update_pressure_jax(surface_tension, Nx, Ny, dx, dy,
                                     self.rho1, self.rho2, phi, self.g, self.atm_pressure, pressure_solver)
        else:
            return update_pressure(surface_tension, Nx, Ny, dx, dy,
                                 self.rho1, self.rho2, phi, self.g, self.atm_pressure, pressure_solver)
