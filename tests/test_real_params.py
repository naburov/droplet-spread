#!/usr/bin/env python3
"""
Test with real simulation parameters.
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
from physics.surface_tension import SurfaceTensionSolver


def test_real_params():
    """Test with real simulation parameters."""
    print("=" * 60)
    print("TEST WITH REAL SIMULATION PARAMETERS")
    print("=" * 60)
    
    # Load config
    config = load_config("configs/config_water_droplet.json")
    
    # Real parameters from config
    Nx = config["grid_params"]["Nx"]
    Ny = config["grid_params"]["Ny"]
    Lx = config["grid_params"]["Lx"]
    Ly = config["grid_params"]["Ly"]
    dx = Lx / Nx
    dy = Ly / Ny
    dt = config["time_params"]["dt"]
    
    print(f"Grid: {Nx}x{Ny}, dx={dx:.6f}, dy={dy:.6f}")
    print(f"Time step: {dt:.6f}")
    
    # Physical parameters
    rho1 = config["physical_params"]["rho1"]
    rho2 = config["physical_params"]["rho2"]
    Re1 = config["physical_params"]["Re1"]
    Re2 = config["physical_params"]["Re2"]
    We1 = config["physical_params"]["We1"]
    We2 = config["physical_params"]["We2"]
    Pe = config["physical_params"]["Pe"]
    epsilon = config["physical_params"]["epsilon"]
    contact_angle = config["physical_params"]["contact_angle"]
    Fr = config["physical_params"]["Fr"]
    g = config["physical_params"]["g"]
    include_gravity = config["physical_params"]["include_gravity"]
    
    print(f"Physical: rho1={rho1}, rho2={rho2}, Re1={Re1}, Re2={Re2}")
    print(f"Surface tension: We1={We1}, We2={We2}, epsilon={epsilon}")
    print(f"Gravity: Fr={Fr}, g={g}, include={include_gravity}")
    
    # Create droplet phase field
    radius = config["initial_conditions"]["droplet_radius"]
    phi = initialize_phase(Nx, Ny, radius)
    phi = jnp.array(phi)
    
    print(f"Phase field: {phi.min():.6f} / {phi.max():.6f}")
    
    # Initialize fields
    U = jnp.zeros((Nx, Ny, 2))
    P = jnp.zeros((Nx, Ny))
    
    # Create solvers
    fluid_solver = FluidDynamicsSolver(rho1, rho2, Re1, Re2, Fr, g)
    surface_tension_solver = SurfaceTensionSolver(epsilon, We1, We2, contact_angle)
    
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
    
    print(f"PPE thresholds: max={max_div_threshold}, mean={mean_div_threshold}, div={div_threshold}")
    
    # Run a few steps
    for step in range(3):
        print(f"\n--- STEP {step} ---")
        
        # Calculate surface tension force
        surface_tension = surface_tension_solver.calculate_force(phi, dx, dy, use_jax=False)
        surface_tension = surface_tension_solver.apply_boundary_conditions(surface_tension, phi, use_jax=False)
        
        print(f"  Surface tension range: {surface_tension.min():.6f} / {surface_tension.max():.6f}")
        
        # Step 1: Predictor step
        U_pred = fluid_solver.update_velocity(
            U, P, surface_tension, dt, dx, dy, phi, 
            include_gravity=include_gravity, use_jax=True
        )
        
        print(f"  U_pred range: {U_pred.min():.6f} / {U_pred.max():.6f}")
        print(f"  Mean y-velocity: {jnp.mean(U_pred[..., 1]):.6f}")
        
        # Step 2: Apply velocity boundary conditions
        velocity_bc_manager = VelocityBoundaryConditions(config)
        U_bc = velocity_bc_manager.apply_boundary_conditions(U_pred, dx, dy, use_jax=True)
        
        # Step 3: Check continuity
        from physics.fluid_dynamics import check_continuity
        divergence, max_div, mean_div = check_continuity(U_bc, dx, dy)
        
        print(f"  Max div: {max_div:.6f}, Mean div: {mean_div:.6f}")
        print(f"  U_bc range: {U_bc.min():.6f} / {U_bc.max():.6f}")
        
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
    """Run real parameters test."""
    test_real_params()
    
    print(f"\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("This test uses the exact same parameters as the simulation.")
    print("If it works here but fails in the simulation, then")
    print("the issue is in the simulation loop or some other factor.")


if __name__ == "__main__":
    main()
