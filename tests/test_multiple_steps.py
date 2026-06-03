#!/usr/bin/env python3
"""
Test multiple time steps to see if divergence accumulates.
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
from simulation.initial_conditions import initialize_phase


def test_multiple_steps():
    """Test multiple time steps."""
    print("=" * 60)
    print("MULTIPLE TIME STEPS TEST")
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
    
    # Create droplet phase field
    radius = config["initial_conditions"]["droplet_radius"]
    phi = initialize_phase(Nx, Ny, radius)
    phi = jnp.array(phi)
    
    # Initialize fields
    U = jnp.zeros((Nx, Ny, 2))
    P = jnp.zeros((Nx, Ny))
    surface_tension = jnp.zeros((Nx, Ny, 2))
    
    # Create solvers
    fluid_solver = FluidDynamicsSolver(rho1, rho2, Re1, Re2, Fr, g)
    
    # Create correction solver
    correction_solver = SparseSolverWrapper(Nx, Ny, dx, dy, "pyamg")
    correction_solver.set_top_boundary_condition("neumann")
    correction_solver.set_bottom_boundary_condition("neumann")
    correction_solver.set_left_boundary_condition("neumann")
    correction_solver.set_right_boundary_condition("neumann")
    correction_solver.create_sparse_matrix()
    
    # PPE parameters
    max_div_threshold = config["solver_params"]["ppe"]["max_div_threshold"]
    mean_div_threshold = config["solver_params"]["ppe"]["mean_div_threshold"]
    div_threshold = config["solver_params"]["ppe"]["div_threshold"]
    
    # Run multiple steps
    for step in range(10):
        print(f"\n--- STEP {step} ---")
        
        # Step 1: Predictor step
        U_pred = fluid_solver.update_velocity(
            U, P, surface_tension, dt, dx, dy, phi, 
            include_gravity=include_gravity, use_jax=True
        )
        
        # Step 2: Apply velocity boundary conditions
        velocity_bc_manager = VelocityBoundaryConditions(config)
        U_bc = velocity_bc_manager.apply_boundary_conditions(U_pred, dx, dy, use_jax=True)
        
        # Step 3: Check continuity
        from physics.fluid_dynamics import check_continuity
        divergence, max_div, mean_div = check_continuity(U_bc, dx, dy)
        
        print(f"  Max div: {max_div:.6f}, Mean div: {mean_div:.6f}")
        print(f"  U range: {U_bc.min():.6f} / {U_bc.max():.6f}")
        print(f"  Mean y-velocity: {jnp.mean(U_bc[..., 1]):.6f}")
        
        # Step 4: Apply PPE if needed
        if max_div > max_div_threshold or mean_div > mean_div_threshold:
            print(f"  Applying PPE...")
            try:
                U = ppe(U_bc, dx, dy, dt, correction_solver, 
                       div_threshold=div_threshold, max_div_threshold=max_div_threshold, 
                       mean_div_threshold=mean_div_threshold)
                print(f"  ✓ PPE succeeded")
                
                # Check final continuity
                div_final, max_div_final, mean_div_final = check_continuity(U, dx, dy)
                print(f"  Final div: Max={max_div_final:.6f}, Mean={mean_div_final:.6f}")
                
            except Exception as e:
                print(f"  ✗ PPE failed: {e}")
                break
        else:
            print(f"  PPE not needed")
            U = U_bc
        
        # Check for exploding values
        if jnp.any(jnp.isnan(U)) or jnp.any(jnp.isinf(U)):
            print(f"  ✗ Exploding values detected!")
            break
        
        if max_div > 1000:  # Very high divergence
            print(f"  ✗ Divergence exploding!")
            break


def main():
    """Run multiple steps test."""
    test_multiple_steps()
    
    print(f"\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("This test shows if divergence accumulates over time.")
    print("If it does, then the issue is in the iterative process.")


if __name__ == "__main__":
    main()
