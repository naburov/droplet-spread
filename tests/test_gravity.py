#!/usr/bin/env python3
"""
Test script to verify gravity implementation.

Updated to current src/ APIs:
- jax_update_velocity no longer takes a solid_mask argument; the geometry is
  described by f_1_grid / f_2_grid (zeros for a flat surface).
- The gravity body force now uses the classical Froude convention
  Fr = U / sqrt(g L), so the expected acceleration is g / Fr^2 (it was g / Fr
  in the old implementation this test was written against).
- Matplotlib visualization was dropped; the checks are numerical assertions.
"""

import numpy as np
import jax.numpy as jnp
import jax
import sys
import os

jax.config.update('jax_enable_x64', True)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from physics.fluid_dynamics import jax_update_velocity
from simulation.initial_conditions import initialize_phase


def test_gravity():
    """Gravity adds a uniform downward acceleration g/Fr^2 to a fluid at rest."""

    # Simple test parameters
    Nx, Ny = 64, 64
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / Nx, Ly / Ny

    # Physical parameters
    rho1 = 1.0      # Air density
    rho2 = 1000.0   # Water density
    Re1 = 10.0
    Re2 = 100.0
    Fr = 0.1
    g = -100.0      # Strong downward gravity
    include_gravity = True

    # Create a simple droplet in the center
    phi = jnp.array(initialize_phase(Nx, Ny, 0.2))

    # Initialize velocity field (initially at rest)
    U = jnp.zeros((Nx, Ny, 2))

    # Initialize pressure field
    P = jnp.zeros((Nx, Ny))

    # Surface tension force (zero for this test)
    surface_tension = jnp.zeros((Nx, Ny, 2))

    # Time step
    dt = 0.001

    # Flat geometry for tests
    f_1_grid = jnp.zeros((Nx, Ny))
    f_2_grid = jnp.zeros((Nx, Ny))

    # Update velocity with gravity
    U_new = jax_update_velocity(
        U, P, surface_tension, dt, dx, dy,
        rho1, rho2, Re1, Re2, Fr, g, phi, f_1_grid, f_2_grid,
        include_gravity=include_gravity
    )

    delta_U = U_new - U

    # Expected gravity-induced velocity change per time step (g / Fr^2 convention).
    expected_gravity_force = (1 / Fr**2) * g * dt

    # With U=0, P=0 and zero surface tension, the only contribution is gravity:
    # the y-velocity change must be exactly uniform and the x-velocity unchanged.
    np.testing.assert_allclose(
        np.asarray(delta_U[..., 1]), expected_gravity_force, rtol=1e-12,
        err_msg="gravity must add a uniform g/Fr^2 * dt to the y-velocity"
    )
    np.testing.assert_allclose(
        np.asarray(delta_U[..., 0]), 0.0, atol=1e-15,
        err_msg="gravity must not change the x-velocity"
    )
    assert np.all(np.isfinite(np.asarray(U_new)))

    # Test without gravity for comparison: nothing must change for a fluid at rest.
    U_no_gravity = jax_update_velocity(
        U, P, surface_tension, dt, dx, dy,
        rho1, rho2, Re1, Re2, Fr, g, phi, f_1_grid, f_2_grid,
        include_gravity=False
    )
    delta_U_no_gravity = U_no_gravity - U
    np.testing.assert_allclose(
        np.asarray(delta_U_no_gravity), 0.0, atol=1e-15,
        err_msg="without gravity, a fluid at rest must stay at rest"
    )


if __name__ == "__main__":
    test_gravity()
    print("Gravity test passed.")
