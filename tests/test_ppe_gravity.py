#!/usr/bin/env python3
"""
Test script to check if PPE is counteracting gravity.

Updated to current src/ APIs:
- jax_update_velocity has no solid_mask; the flat surface is described by
  f_1_grid / f_2_grid (zeros). jax_check_continuity requires f_1_grid.
- SurfaceTensionSolver / PressureSolver methods take a geometry object
  (simulation.geometry.Geometry.flat).
- solvers.projection_methods.ppe requires a geometry argument and returns
  (U_corrected, info_dict). A velocity BC manager is passed explicitly because
  the manager-less fallback inside ppe_solve references a removed helper.
- The gravity body force uses the g/Fr^2 (classical Froude) convention.
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

from physics.fluid_dynamics import jax_update_velocity, jax_check_continuity
from physics.surface_tension import SurfaceTensionSolver
from physics.pressure import PressureSolver
from simulation.initial_conditions import initialize_phase
from simulation.geometry import Geometry
from solvers.sparse_solver import SparseSolverWrapper
from solvers.projection_methods import ppe
from boundary_conditions.velocity_bc import VelocityBoundaryConditions


def _make_neumann_solver(Nx, Ny, dx, dy):
    solver = SparseSolverWrapper(Nx, Ny, dx, dy, "pyamg")
    solver.set_top_boundary_condition("neumann")
    solver.set_bottom_boundary_condition("neumann")
    solver.set_left_boundary_condition("neumann")
    solver.set_right_boundary_condition("neumann")
    solver.create_sparse_matrix()
    return solver


def test_ppe_gravity_interaction():
    """PPE removes the divergent part of the velocity but not the uniform gravity sink."""

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

    # Initialize fields (fluid at rest)
    U = jnp.zeros((Nx, Ny, 2))

    # Time step
    dt = 0.001

    # Flat geometry for tests
    geometry = Geometry.flat(Nx, Ny)
    f_1_grid = geometry.f_1_grid
    f_2_grid = geometry.f_2_grid

    # Calculate surface tension
    surface_tension_solver = SurfaceTensionSolver(epsilon, We1, We2, contact_angle)
    surface_tension = surface_tension_solver.calculate_force(phi, dx, dy, geometry)
    surface_tension = surface_tension_solver.apply_boundary_conditions(
        surface_tension, phi, geometry=geometry
    )

    # Calculate pressure (capillary + hydrostatic)
    pressure_solver = PressureSolver(
        rho1, rho2, g, atm_pressure, Fr=Fr, include_gravity=include_gravity
    )
    P_new = pressure_solver.update_pressure(
        surface_tension, dx, dy, geometry, phi,
        _make_neumann_solver(Nx, Ny, dx, dy),
    )
    assert np.all(np.isfinite(np.asarray(P_new)))

    # Test 1: Velocity update with gravity
    U_after_velocity = jax_update_velocity(
        U, P_new, surface_tension, dt, dx, dy,
        rho1, rho2, Re1, Re2, Fr, g, phi, f_1_grid, f_2_grid,
        include_gravity=include_gravity
    )
    assert np.all(np.isfinite(np.asarray(U_after_velocity)))

    # Check continuity before PPE
    _, max_div, mean_div = jax_check_continuity(U_after_velocity, dx, dy, f_1_grid)

    # Test 2: Apply PPE correction
    velocity_bc_manager = VelocityBoundaryConditions({
        "boundary_conditions": {
            "velocity": {
                "top": "slip_symmetry",
                "bottom": "slip_symmetry",
                "left": "slip_symmetry",
                "right": "slip_symmetry",
            }
        }
    })

    U_after_ppe, info = ppe(
        U_after_velocity, dx, dy, dt, geometry,
        correction_solver=_make_neumann_solver(Nx, Ny, dx, dy),
        velocity_bc_manager=velocity_bc_manager,
        div_threshold=0.01, max_div_threshold=1.0, mean_div_threshold=0.1,
        max_iterations=50,
    )

    assert np.all(np.isfinite(np.asarray(U_after_ppe))), "PPE produced non-finite velocities"
    assert not info.get('diverged', False), "PPE must not blow up"

    # Check continuity after PPE
    _, max_div_after, mean_div_after = jax_check_continuity(U_after_ppe, dx, dy, f_1_grid)
    if info['applied']:
        assert float(mean_div_after) <= float(mean_div) + 1e-12, (
            f"PPE must not increase mean divergence "
            f"({float(mean_div):.3e} -> {float(mean_div_after):.3e})"
        )

    # Calculate changes
    delta_U_velocity = U_after_velocity - U
    delta_U_ppe = U_after_ppe - U_after_velocity
    delta_U_total = U_after_ppe - U

    predictor_effect = float(jnp.mean(delta_U_velocity[..., 1]))
    ppe_effect = float(jnp.mean(delta_U_ppe[..., 1]))
    total_effect = float(jnp.mean(delta_U_total[..., 1]))

    # Magnitude of the raw gravity body-force change per step (g/Fr^2 convention).
    body_force_change = abs((1 / Fr**2) * g * dt)

    # The hydrostatic pressure gradient must counteract most of the gravity body
    # force in the predictor step: the residual mean acceleration is far below
    # the raw body force.
    assert abs(predictor_effect) < 0.2 * body_force_change, (
        f"hydrostatic pressure should nearly balance gravity: mean predictor "
        f"change {predictor_effect:.6f} vs body force {body_force_change:.6f}"
    )

    # Test 3: PPE must not act as a spurious body force: its mean y-momentum
    # injection must stay far below the gravity body-force scale.
    assert abs(ppe_effect) < 0.2 * body_force_change, (
        f"PPE should not inject body-force-scale momentum: PPE effect "
        f"{ppe_effect:.6f} vs body force {body_force_change:.6f}"
    )
    assert abs(total_effect) < 0.2 * body_force_change, (
        f"net mean y-velocity change should remain small: {total_effect:.6f}"
    )


if __name__ == "__main__":
    test_ppe_gravity_interaction()
    print("PPE-gravity interaction test passed.")
