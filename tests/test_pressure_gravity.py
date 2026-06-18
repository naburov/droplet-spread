#!/usr/bin/env python3
"""
Test script to check if pressure is counteracting gravity.

Updated to current src/ APIs:
- jax_update_velocity has no solid_mask; the flat surface is described by
  f_1_grid / f_2_grid (zeros).
- SurfaceTensionSolver.calculate_force / apply_boundary_conditions and
  PressureSolver.update_pressure now take a geometry object
  (simulation.geometry.Geometry.flat).
- PressureSolver only includes the hydrostatic contribution when constructed
  with include_gravity=True and the Froude number.
- The gravity body force uses the g/Fr^2 (classical Froude) convention.
- numerics.finite_differences.jax_gradient requires f_1_grid.
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
from physics.surface_tension import SurfaceTensionSolver
from physics.pressure import PressureSolver
from physics.properties import jax_calculate_density
from simulation.initial_conditions import initialize_phase
from simulation.geometry import Geometry
from solvers.sparse_solver import SparseSolverWrapper
from numerics.finite_differences import jax_gradient


def test_pressure_gravity_interaction():
    """Hydrostatic pressure must (partially) counteract the gravity body force."""

    # Simple test parameters
    Nx, Ny = 64, 64
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / Nx, Ly / Ny

    # Physical parameters
    rho1 = 1.0      # Air density
    rho2 = 1000.0   # Water density
    Re1 = 10.0
    Re2 = 100.0
    We1 = 0.1
    We2 = 10.0
    Fr = 0.1
    g = -10.0
    include_gravity = True
    epsilon = 0.05
    contact_angle = 120
    atm_pressure = 0.0

    # Create a simple droplet in the center
    phi = jnp.array(initialize_phase(Nx, Ny, 0.2))

    # Initialize velocity field (initially at rest)
    U = jnp.zeros((Nx, Ny, 2))

    # Initialize pressure field
    P = jnp.zeros((Nx, Ny))

    # Time step
    dt = 0.001

    # Flat geometry for tests
    geometry = Geometry.flat(Nx, Ny)
    f_1_grid = geometry.f_1_grid
    f_2_grid = geometry.f_2_grid

    # Test 1: Velocity update with gravity only (no pressure)
    U_gravity_only = jax_update_velocity(
        U, P, jnp.zeros((Nx, Ny, 2)), dt, dx, dy,
        rho1, rho2, Re1, Re2, Fr, g, phi, f_1_grid, f_2_grid,
        include_gravity=include_gravity
    )

    gravity_only_mean = float(jnp.mean(U_gravity_only[..., 1]))
    expected_gravity_change = (1 / Fr**2) * g * dt
    np.testing.assert_allclose(
        np.asarray(U_gravity_only[..., 1]), expected_gravity_change, rtol=1e-12,
        err_msg="gravity-only update must be the uniform body force g/Fr^2 * dt"
    )

    # Test 2: Calculate surface tension and (hydrostatic) pressure
    surface_tension_solver = SurfaceTensionSolver(epsilon, We1, We2, contact_angle)
    surface_tension = surface_tension_solver.calculate_force(phi, dx, dy, geometry)
    surface_tension = surface_tension_solver.apply_boundary_conditions(
        surface_tension, phi, geometry=geometry
    )
    assert np.all(np.isfinite(np.asarray(surface_tension)))

    pressure_solver = PressureSolver(
        rho1, rho2, g, atm_pressure, Fr=Fr, include_gravity=include_gravity
    )

    pressure_linear_solver = SparseSolverWrapper(Nx, Ny, dx, dy, "pyamg")
    pressure_linear_solver.set_top_boundary_condition("neumann")
    pressure_linear_solver.set_bottom_boundary_condition("neumann")
    pressure_linear_solver.set_left_boundary_condition("neumann")
    pressure_linear_solver.set_right_boundary_condition("neumann")
    pressure_linear_solver.create_sparse_matrix()

    P_new = pressure_solver.update_pressure(
        surface_tension, dx, dy, geometry, phi, pressure_linear_solver
    )

    assert np.all(np.isfinite(np.asarray(P_new)))
    assert float(jnp.max(P_new) - jnp.min(P_new)) > 0.0, \
        "pressure with hydrostatic contribution must not be constant"

    # Test 3: Velocity update with pressure
    U_with_pressure = jax_update_velocity(
        U, P_new, surface_tension, dt, dx, dy,
        rho1, rho2, Re1, Re2, Fr, g, phi, f_1_grid, f_2_grid,
        include_gravity=include_gravity
    )
    with_pressure_mean = float(jnp.mean(U_with_pressure[..., 1]))

    # The hydrostatic pressure gradient must oppose gravity: the mean downward
    # acceleration with pressure must be smaller in magnitude than gravity alone.
    assert abs(with_pressure_mean) < abs(gravity_only_mean), (
        f"pressure should counteract gravity: |{with_pressure_mean:.6f}| !< "
        f"|{gravity_only_mean:.6f}|"
    )

    # Test 4: Check pressure gradient (terrain-aware gradient requires f_1_grid).
    p_grad = jax_gradient(P_new, dx, dy, f_1_grid)
    assert np.all(np.isfinite(np.asarray(p_grad)))
    # Hydrostatic balance: on average dP/dy must have the sign of rho*g (negative
    # for downward gravity), i.e. pressure decreases with height.
    assert float(jnp.mean(p_grad[..., 1])) < 0.0, \
        "mean dP/dy must be negative for downward gravity (hydrostatic pressure)"

    # Test 5: Check density mapping
    rho = jax_calculate_density(phi, rho1, rho2)
    rho_np = np.asarray(rho)
    assert rho_np.min() >= min(rho1, rho2) - 1e-6
    assert rho_np.max() <= max(rho1, rho2) + 1e-6


if __name__ == "__main__":
    test_pressure_gravity_interaction()
    print("Pressure-gravity interaction test passed.")
