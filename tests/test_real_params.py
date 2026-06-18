#!/usr/bin/env python3
"""
Test with real simulation parameters.

Updated to current src/ APIs:
- configs/config_water_droplet.json no longer exists; the test uses
  configs/config_template.json, which is the current reference ("real")
  simulation configuration and includes solver_params.
- The predictor step uses jax_update_velocity directly with
  geometry.f_1_grid / f_2_grid (zeros for a flat surface). The
  FluidDynamicsSolver.update_velocity wrapper cannot currently be invoked: it
  forwards mu_convention (a string) as a non-static argument to the jitted
  kernel, which TypeErrors under the installed JAX. The jitted kernel is the
  same physics the wrapper would run.
- SurfaceTensionSolver.calculate_force / apply_boundary_conditions take a
  geometry object.
- check_continuity was replaced by jax_check_continuity(U, dx, dy, f_1_grid).
- ppe requires a geometry argument and returns (U_corrected, info_dict); a
  velocity BC manager is passed explicitly because the manager-less fallback
  inside ppe_solve references a removed helper.
"""

import os
import sys
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update('jax_enable_x64', True)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from physics.fluid_dynamics import jax_update_velocity, jax_check_continuity
from solvers.projection_methods import ppe
from boundary_conditions.velocity_bc import VelocityBoundaryConditions
from config.config_loader import load_config
from solvers.sparse_solver import SparseSolverWrapper
from simulation.initial_conditions import initialize_phase
from simulation.geometry import Geometry
from physics.surface_tension import SurfaceTensionSolver

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "configs", "config_template.json")


def test_real_params():
    """A few full predictor/BC/PPE steps with the reference config parameters stay sane."""
    # Load config
    config = load_config(CONFIG_PATH)
    assert "solver_params" in config

    # Real parameters from config
    Nx = config["grid_params"]["Nx"]
    Ny = config["grid_params"]["Ny"]
    Lx = config["grid_params"]["Lx"]
    Ly = config["grid_params"]["Ly"]
    dx = Lx / Nx
    dy = Ly / Ny
    dt = config["time_params"]["dt"]
    geometry = Geometry.flat(Nx, Ny)

    # Physical parameters
    rho1 = config["physical_params"]["rho1"]
    rho2 = config["physical_params"]["rho2"]
    Re1 = config["physical_params"]["Re1"]
    Re2 = config["physical_params"]["Re2"]
    We1 = config["physical_params"]["We1"]
    We2 = config["physical_params"]["We2"]
    epsilon = config["physical_params"]["epsilon"]
    contact_angle = config["physical_params"]["contact_angle"]
    Fr = config["physical_params"]["Fr"]
    g = config["physical_params"]["g"]
    include_gravity = config["physical_params"]["include_gravity"]

    # Create droplet phase field
    radius = config["initial_conditions"]["droplet_radius"]
    phi = jnp.array(initialize_phase(Nx, Ny, radius, epsilon=epsilon, config=config))
    phi_np = np.asarray(phi)
    assert phi_np.min() < -0.9 and phi_np.max() > 0.9, "phase field must contain both phases"

    # Initialize fields
    U = jnp.zeros((Nx, Ny, 2))
    P = jnp.zeros((Nx, Ny))

    # Create solvers
    surface_tension_solver = SurfaceTensionSolver(epsilon, We1, We2, contact_angle)
    velocity_bc_manager = VelocityBoundaryConditions(config)

    correction_solver = SparseSolverWrapper(Nx, Ny, dx, dy, "pyamg")
    correction_solver.set_top_boundary_condition("neumann")
    correction_solver.set_bottom_boundary_condition("neumann")
    correction_solver.set_left_boundary_condition("neumann")
    correction_solver.set_right_boundary_condition("neumann")
    correction_solver.create_sparse_matrix()

    # PPE parameters
    ppe_params = config["solver_params"]["ppe"]
    max_div_threshold = ppe_params["max_div_threshold"]
    mean_div_threshold = ppe_params["mean_div_threshold"]
    div_threshold = ppe_params.get("div_threshold", 0.05)

    # Run a few steps
    for step in range(3):
        # Calculate surface tension force
        surface_tension = surface_tension_solver.calculate_force(phi, dx, dy, geometry)
        surface_tension = surface_tension_solver.apply_boundary_conditions(
            surface_tension, phi, geometry=geometry
        )
        assert np.all(np.isfinite(np.asarray(surface_tension))), \
            f"step {step}: non-finite surface tension"

        # Step 1: Predictor step
        U_pred = jax_update_velocity(
            U, P, surface_tension, dt, dx, dy,
            rho1, rho2, Re1, Re2, Fr, g, phi,
            geometry.f_1_grid, geometry.f_2_grid,
            include_gravity=include_gravity
        )
        assert np.all(np.isfinite(np.asarray(U_pred))), \
            f"step {step}: predictor produced non-finite velocities"

        # Step 2: Apply velocity boundary conditions
        U_bc = velocity_bc_manager.apply_boundary_conditions(U_pred, dx, dy, use_jax=True)

        # Step 3: Check continuity
        divergence, max_div, mean_div = jax_check_continuity(U_bc, dx, dy, geometry.f_1_grid)
        assert float(max_div) < 1000.0, f"step {step}: divergence exploding"

        # Step 4: Apply PPE if needed
        if max_div > max_div_threshold or mean_div > mean_div_threshold:
            U, info = ppe(
                U_bc, dx, dy, dt, geometry, correction_solver,
                velocity_bc_manager=velocity_bc_manager,
                div_threshold=div_threshold, max_div_threshold=max_div_threshold,
                mean_div_threshold=mean_div_threshold, max_iterations=50,
            )
            assert not info.get('diverged', False), f"step {step}: PPE blew up"

            _, max_div_final, mean_div_final = jax_check_continuity(
                U, dx, dy, geometry.f_1_grid
            )
            if info['applied']:
                assert float(mean_div_final) <= float(mean_div) + 1e-12, (
                    f"step {step}: PPE increased mean divergence "
                    f"({float(mean_div):.3e} -> {float(mean_div_final):.3e})"
                )
        else:
            U = U_bc

        assert not (jnp.any(jnp.isnan(U)) or jnp.any(jnp.isinf(U))), \
            f"step {step}: exploding values detected"


if __name__ == "__main__":
    test_real_params()
    print("Real parameters test passed.")
