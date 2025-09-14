"""
Refactored main simulation file for droplet spreading.

This file demonstrates how to use the new modular structure
for running droplet spreading simulations.
"""

import os
import argparse
import json
from datetime import datetime
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update('jax_enable_x64', True)  # Enable 64-bit precision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import time

# Import from the new modular structure
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import individual modules
from physics.phase_field import PhaseFieldSolver, apply_contact_angle_boundary_conditions
from physics.fluid_dynamics import FluidDynamicsSolver, apply_velocity_boundary_conditions, check_continuity
from physics.surface_tension import SurfaceTensionSolver
from physics.pressure import PressureSolver
from numerics.time_integration import cfl_dt
from solvers.sparse_solver import SparseSolverWrapper
from solvers.projection_methods import ppe
from visualization.plotting import create_joint_plot
from visualization.checkpointing import save_checkpoint, load_checkpoint
from config.config_loader import load_config
from simulation.initial_conditions import initialize_phase, get_borders_of_droplet


def main():
    """Main simulation function using the refactored modular structure."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Droplet spreading simulation (refactored)')
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON format)')
    parser.add_argument('--output', type=str, help='Path to output directory')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract parameters from config
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
    include_gravity = config["physical_params"]["include_gravity"]
    g = config["physical_params"]["g"]
    atm_pressure = config["physical_params"]["atm_pressure"]
    Fr = config["physical_params"]["Fr"]
    
    # Grid setup
    Lx, Ly = config["grid_params"]["Lx"], config["grid_params"]["Ly"]
    Nx, Ny = config["grid_params"]["Nx"], config["grid_params"]["Ny"]
    dx, dy = Lx / Nx, Ly / Ny
    
    # Time setup
    dt = config["time_params"]["dt"]
    t_max = config["time_params"]["t_max"]
    checkpoint_interval = config["time_params"]["checkpoint_interval"]
    dt_initial = config["time_params"]["dt_initial"]
    
    # Initial conditions
    radius = config["initial_conditions"]["droplet_radius"]
    
    # Restart information
    restart_from = config["restart"]["restart_from"]
    if restart_from == "None":
        restart_from = None
    
    # Create output directory
    if args.output:
        login_dir = args.output
        os.makedirs(login_dir, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        login_dir = f"experiment_{timestamp}"
        os.makedirs(login_dir, exist_ok=True)

    # Initialize solvers
    phase_solver = PhaseFieldSolver(Pe, epsilon, contact_angle)
    fluid_solver = FluidDynamicsSolver(rho1, rho2, Re1, Re2, Fr, g)
    surface_tension_solver = SurfaceTensionSolver(epsilon, We1, We2, contact_angle)
    pressure_solver = PressureSolver(rho1, rho2, g, atm_pressure)
    
    # Initialize the phase field
    phi = initialize_phase(Nx, Ny, radius)
    phi = jnp.array(phi)
    
    # Initialize fields
    U = jnp.zeros((Nx, Ny, 2))  # Velocity field
    P = jnp.zeros((Nx, Ny))     # Pressure field

    # Visualization of the initial phase field
    plt.imshow(phi.T, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
    plt.colorbar(label='Phase Field (phi)')
    plt.title('Initial Phase Field')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(f'{login_dir}/initial_phase_field.png', bbox_inches='tight')
    plt.clf()

    # Create checkpoint directory
    checkpoint_dir = os.path.join(login_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save configuration
    params_file = os.path.join(login_dir, "simulation_parameters.json")
    with open(params_file, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Simulation parameters saved to {params_file}")

    # Optional: restart from checkpoint
    if restart_from is not None:
        print(f"Restarting from checkpoint: {restart_from}")
        checkpoint_data = load_checkpoint(restart_from)
        start_step = checkpoint_data['step']
        phi = checkpoint_data['phi']
        U = checkpoint_data['U']
        P = checkpoint_data['P']
        print(f"Loaded state from step {start_step}, continuing simulation...")
    else:
        start_step = 0

    # Create sparse solvers
    correction_solver = SparseSolverWrapper(Nx, Ny, dx, dy, "pyamg")
    pressure_linear_solver = SparseSolverWrapper(Nx, Ny, dx, dy, "pyamg")
    
    # Set boundary conditions
    correction_solver.set_top_boundary_condition("neumann")
    correction_solver.set_bottom_boundary_condition("neumann")
    correction_solver.set_left_boundary_condition("neumann")
    correction_solver.set_right_boundary_condition("neumann")
    
    pressure_linear_solver.set_top_boundary_condition("dirichlet")
    pressure_linear_solver.set_bottom_boundary_condition("dirichlet")
    pressure_linear_solver.set_left_boundary_condition("neumann")
    pressure_linear_solver.set_right_boundary_condition("neumann")
    
    # Create sparse matrices with boundary conditions
    correction_solver.create_sparse_matrix()
    pressure_linear_solver.create_sparse_matrix()

    times = []
    cur_t = 0
    step = start_step
    
    print("Starting simulation...")
    
    # Main simulation loop
    while cur_t < t_max:
        start_time = time.time()
        current_dt = dt_initial if step < 500 else dt

        # CFL condition check
        cfl_computed_dt = cfl_dt(U[..., 0].max(), U[..., 1].max(), dx, dy, C=0.01)
        if cfl_computed_dt != np.inf:
            current_dt = cfl_computed_dt
            print(f"Updated dt: {current_dt}")
        
        cur_t += current_dt

        # Calculate surface tension force
        surface_tension = surface_tension_solver.calculate_force(phi, dx, dy, use_jax=False)
        surface_tension = surface_tension_solver.apply_boundary_conditions(surface_tension, phi, use_jax=False)
        
        # Update velocity field
        U = fluid_solver.update_velocity(U, P, surface_tension, current_dt, dx, dy, 
                                        phi, include_gravity=include_gravity, use_jax=True)

        # Apply velocity boundary conditions
        beta = 0.01
        U = apply_velocity_boundary_conditions(U, beta, dy)

        # Check continuity and apply PPE if needed
        divergence, max_div, mean_div = fluid_solver.check_continuity(U, dx, dy, use_jax=True)
        
        # Apply PPE to enforce incompressibility (now with exact original implementation)
        if mean_div > 0.1:  # Only apply PPE if mean divergence is significant
            U = ppe(U, dx, dy, current_dt, correction_solver, div_threshold=0.01)

        # Update phase field
        phi = phase_solver.update(phi, U, current_dt, dx, dy, use_jax=True)
        
        # Recompute surface tension
        surface_tension = surface_tension_solver.calculate_force(phi, dx, dy, use_jax=True)
        surface_tension = surface_tension_solver.apply_boundary_conditions(surface_tension, phi, use_jax=True)

        # Update pressure
        P = pressure_solver.update_pressure(surface_tension, Nx, Ny, dx, dy, phi, pressure_linear_solver, use_jax=True)
        
        end_time = time.time()
        times.append(end_time - start_time)
        
        # Print progress
        if step % 10 == 0:
            print(f"Step {step}, Time {cur_t:.2f}")
            print(f"Min/Max of U: {U.min():.4f} / {U.max():.4f}")
            print(f"Min/Max of P: {P.min():.4f} / {P.max():.4f}")
            print(f"Min/Max of phi: {phi.min():.4f} / {phi.max():.4f}")
            print(f"Continuity check - Max |div(U)|: {max_div:.6f}, Mean |div(U)|: {mean_div:.6f}")
            print(f"Time per step: {np.mean(times):.6f}")
            start_of_droplet, end_of_droplet = get_borders_of_droplet(phi)
            mass = np.sum(phi[phi > 0])
            print(f"Droplet mass: {mass}")
            print(f"Start/end of droplet: {start_of_droplet} - {end_of_droplet}")

        # Create plots
        if step % checkpoint_interval == 0:
            mass = np.sum(phi[phi > 0])
            create_joint_plot(
                phi=phi,
                U=U,
                P=P,
                surface_tension=surface_tension,
                dt=current_dt,
                step=step,
                dx=dx,
                dy=dy,
                mass=mass,
                rho1=rho1,
                rho2=rho2,
                save_path=f'{login_dir}/joint_plot_step_{step}.png'
            )

        # Save checkpoint
        if step % checkpoint_interval == 0:
            save_checkpoint(step, phi, U, P, directory=f"{login_dir}/checkpoints")
        
        step += 1

    print("Simulation completed!")


if __name__ == "__main__":
    main()
