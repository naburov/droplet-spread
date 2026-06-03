#!/usr/bin/env python3
"""
Test a single simulation step to see what's happening.
"""

import os
import sys
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update('jax_enable_x64', True)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from physics.fluid_dynamics import FluidDynamicsSolver
from physics.properties import calculate_density
from solvers.projection_methods import ppe
from boundary_conditions.pressure_bc import PressureBoundaryConditions
from boundary_conditions.velocity_bc import VelocityBoundaryConditions
from config.config_loader import load_config
from solvers.sparse_solver import SparseSolverWrapper


def test_single_step():
    """Test a single simulation step."""
    print("=" * 60)
    print("SINGLE SIMULATION STEP TEST")
    print("=" * 60)
    
    # Load config
    config = load_config("configs/config_water_droplet.json")
    
    # Parameters
    Nx, Ny = 16, 16
    dx = dy = 0.1
    dt = 0.01
    
    # Physical parameters
    rho1 = config["physical_params"]["rho1"]
    rho2 = config["physical_params"]["rho2"]
    Re1 = config["physical_params"]["Re1"]
    Re2 = config["physical_params"]["Re2"]
    Fr = config["physical_params"]["Fr"]
    g = config["physical_params"]["g"]
    include_gravity = config["physical_params"]["include_gravity"]
    
    print(f"Physical parameters:")
    print(f"  rho1={rho1}, rho2={rho2}")
    print(f"  Re1={Re1}, Re2={Re2}")
    print(f"  Fr={Fr}, g={g}")
    print(f"  include_gravity={include_gravity}")
    
    # Initialize fields
    U = jnp.zeros((Nx, Ny, 2))
    P = jnp.zeros((Nx, Ny))
    phi = jnp.ones((Nx, Ny))  # All phase 1 (air)
    surface_tension = jnp.zeros((Nx, Ny, 2))
    
    print(f"\nInitial state:")
    print(f"  U range: {U.min():.6f} / {U.max():.6f}")
    print(f"  P range: {P.min():.6f} / {P.max():.6f}")
    print(f"  phi range: {phi.min():.6f} / {phi.max():.6f}")
    
    # Create solvers
    fluid_solver = FluidDynamicsSolver(rho1, rho2, Re1, Re2, Fr, g)
    
    # Create correction solver
    correction_solver = SparseSolverWrapper(Nx, Ny, dx, dy, "pyamg")
    correction_solver.set_top_boundary_condition("neumann")
    correction_solver.set_bottom_boundary_condition("neumann")
    correction_solver.set_left_boundary_condition("neumann")
    correction_solver.set_right_boundary_condition("neumann")
    correction_solver.create_sparse_matrix()
    
    print(f"\nSolver created successfully")
    
    # Step 1: Predictor step
    print(f"\n1. PREDICTOR STEP:")
    U_pred = fluid_solver.update_velocity(
        U, P, surface_tension, dt, dx, dy, phi, 
        include_gravity=include_gravity, use_jax=True
    )
    print(f"  U_pred range: {U_pred.min():.6f} / {U_pred.max():.6f}")
    print(f"  Mean y-velocity: {jnp.mean(U_pred[..., 1]):.6f}")
    
    # Step 2: Apply velocity boundary conditions
    print(f"\n2. VELOCITY BOUNDARY CONDITIONS:")
    velocity_bc_manager = VelocityBoundaryConditions(config)
    U_bc = velocity_bc_manager.apply_boundary_conditions(U_pred, dx, dy, use_jax=True)
    print(f"  U_bc range: {U_bc.min():.6f} / {U_bc.max():.6f}")
    print(f"  Mean y-velocity: {jnp.mean(U_bc[..., 1]):.6f}")
    
    # Step 3: Check continuity
    print(f"\n3. CONTINUITY CHECK:")
    from physics.fluid_dynamics import check_continuity
    divergence, max_div, mean_div = check_continuity(U_bc, dx, dy)
    print(f"  Divergence range: {divergence.min():.6f} / {divergence.max():.6f}")
    print(f"  Max div: {max_div:.6f}, Mean div: {mean_div:.6f}")
    
    # Step 4: Apply PPE
    print(f"\n4. APPLYING PPE:")
    max_div_threshold = config["solver_params"]["ppe"]["max_div_threshold"]
    mean_div_threshold = config["solver_params"]["ppe"]["mean_div_threshold"]
    div_threshold = config["solver_params"]["ppe"]["div_threshold"]
    
    print(f"  Thresholds: max={max_div_threshold}, mean={mean_div_threshold}, div={div_threshold}")
    print(f"  Should apply PPE: {max_div > max_div_threshold or mean_div > mean_div_threshold}")
    
    if max_div > max_div_threshold or mean_div > mean_div_threshold:
        print(f"  Applying PPE...")
        try:
            U_corrected = ppe(U_bc, dx, dy, dt, correction_solver, 
                             div_threshold=div_threshold, max_div_threshold=max_div_threshold, 
                             mean_div_threshold=mean_div_threshold)
            print(f"  ✓ PPE succeeded")
            print(f"  U_corrected range: {U_corrected.min():.6f} / {U_corrected.max():.6f}")
            print(f"  Mean y-velocity: {jnp.mean(U_corrected[..., 1]):.6f}")
            
            # Check final continuity
            div_final, max_div_final, mean_div_final = check_continuity(U_corrected, dx, dy)
            print(f"  Final divergence: Max={max_div_final:.6f}, Mean={mean_div_final:.6f}")
            
        except Exception as e:
            print(f"  ✗ PPE failed: {e}")
    else:
        print(f"  PPE not needed")


def main():
    """Run single step test."""
    test_single_step()
    
    print(f"\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print("This test shows exactly what happens in one simulation step.")
    print("If the divergence is zero after the predictor step, then")
    print("the issue is not with gravity but with something else.")


if __name__ == "__main__":
    main()
