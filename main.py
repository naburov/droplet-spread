import os
import argparse
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import utils  # Assuming utils.py is in the same directory
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time 
import sys

# Global variables used by multiple functions
phi = None
Re1 = None
Re2 = None
Pe = None
epsilon = None
phase_penalty = None
contact_angle = None
Fr = None

def load_config(config_path=None):
    """
    Load configuration from a JSON file or use defaults.
    
    Args:
        config_path (str, optional): Path to the JSON configuration file.
        
    Returns:
        dict: Configuration parameters.
    """
    # Default configuration
    config = {
        "physical_params": {
            "rho": 1.0,
            "Re1": 1000.0,
            "Re2": 10.0,
            "We": 10.0,
            "Pe": 1.0,
            "epsilon": 0.05,
            "alpha": 1.0,
            "phase_penalty": 1000.0,
            "contact_angle": 120
        },
        "grid_params": {
            "Lx": 1.0,
            "Ly": 1.0,
            "Nx": 100,
            "Ny": 100
        },
        "time_params": {
            "dt": 0.001,
            "t_max": 1.0,
            "checkpoint_interval": 50,
            "dt_initial": 0.0005
        },
        "initial_conditions": {
            "droplet_radius": 0.2
        },
        "restart": {
            "restart_from": None
        }
    }
    
    # Load configuration from file if provided
    if config_path:
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                
            # Update default config with loaded values (recursive update)
            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        update_dict(d[k], v)
                    else:
                        d[k] = v
            
            update_dict(config, loaded_config)
            sys.stdout.write(f"Configuration loaded from {config_path}")
        except Exception as e:
            sys.stdout.write(f"Error loading config file: {e}")
            sys.stdout.write("Using default configuration")
    else:
        sys.stdout.write("No config file provided. Using default configuration.")
    
    return config

def solve_poisson(rhs, dx, dy):
    """Solve the Poisson equation ∇²φ = f with a custom RHS."""
    Nx, Ny = rhs.shape
    phi = np.zeros((Nx, Ny))

    # Create the coefficient matrix for the finite difference method
    diagonals = [
        -2 * (1/dx**2 + 1/dy**2) * np.ones(Nx * Ny),  # Main diagonal
        (1/dx**2) * np.ones(Nx * Ny - 1),            # Upper diagonal
        (1/dx**2) * np.ones(Nx * Ny - 1),            # Lower diagonal
        (1/dy**2) * np.ones(Nx * (Ny-1)),           # Far upper diagonal
        (1/dy**2) * np.ones(Nx * (Ny-1))            # Far lower diagonal
    ]

    offsets = [0, 1, -1, Nx, -Nx]

    # Adjust for boundary conditions
    for i in range(1, Ny):
        diagonals[1][i * Nx - 1] = 0  # No connection between rows

    # Create the sparse matrix
    A = diags(diagonals, offsets).tocsc()

    # Apply boundary conditions
    # Apply Neumann boundary conditions (zero gradient) at all boundaries
    # For the left and right boundaries
    for i in range(Ny):
        # Left boundary: phi[0, i] = phi[1, i]
        diagonals[0][i * Nx] += diagonals[1][i * Nx]
        diagonals[1][i * Nx] = 0
        rhs[0, i] = 0
        
        # Right boundary: phi[Nx-1, i] = phi[Nx-2, i]
        diagonals[0][(i + 1) * Nx - 1] += diagonals[2][(i + 1) * Nx - 2]
        diagonals[2][(i + 1) * Nx - 2] = 0
        rhs[Nx-1, i] = 0
    
    # For the top and bottom boundaries
    for j in range(Nx):
        # Top boundary: phi[j, 0] = phi[j, 1]
        diagonals[0][j] += diagonals[3][j]
        diagonals[3][j] = 0
        rhs[j, 0] = 0
        
        # Bottom boundary: phi[j, Ny-1] = phi[j, Ny-2]
        diagonals[0][j + (Ny-1) * Nx] += diagonals[4][j + (Ny-2) * Nx]
        diagonals[4][j + (Ny-2) * Nx] = 0
        rhs[j, Ny-1] = 0

    # Ensure at least one Dirichlet condition
    # Top boundary: fixed pressure
    for j in range(Nx):
        diagonals[0][j] = 1.0
        diagonals[3][j] = 0.0
        rhs[j, 0] = 0.0

    # Solve the linear system
    rhs_flat = rhs.flatten()
    phi_flat = spsolve(A, rhs_flat)
    phi = phi_flat.reshape((Nx, Ny))

    return phi

def compute_viscous_term(U, dx, dy, Re):
    """Simplified viscous term for constant viscosity: (1/Re) * ∇²U"""
    viscous_term = np.zeros_like(U)
    viscous_term[..., 0] = utils.laplacian(U[..., 0], dx, dy) / Re
    viscous_term[..., 1] = utils.laplacian(U[..., 1], dx, dy) / Re
    return viscous_term

def check_continuity(U, dx, dy):
    """
    Check continuity equation condition (∇·U = 0)
    Returns the divergence field and maximum absolute divergence
    """
    # Calculate divergence: du/dx + dv/dy
    u_x = utils.numerical_derivative(U[..., 0], axis=0, h=dx)
    v_y = utils.numerical_derivative(U[..., 1], axis=1, h=dy)
    
    divergence = u_x + v_y
    max_div = np.max(np.abs(divergence))
    mean_div = np.mean(np.abs(divergence))
    
    return divergence, max_div, mean_div

def update_velocity(U, p, surface_tension, current_dt, dx, dy, include_gravity=False):
    """Update the velocity field U based on the phase field phi."""
    grad_U = utils.gradient(U, dx, dy)  # Shape: (M, N, 2)
    lap_U = utils.laplacian(U, dx, dy)  # Shape: (M, N, 2)

    # Calculate the Reynolds number
    Re = utils.calculate_reynolds_number(phi, Re1, Re2)

    # Calculate pressure gradient
    p_grad = utils.gradient(p, dx, dy) # Shape: (M, N, 2)

    # Calculate the viscous term
    # viscous_term = (1 / Re)[..., np.newaxis] * (lap_U + np.transpose(lap_U, (1, 0, 2)))  # Shape: (M, N, 2)
    viscous_term = compute_viscous_term(U, dx, dy, Re)

    # Calculate the convective term
    convective_term = np.zeros_like(U)  # Shape: (M, N, 2)

    # Corrected convective term calculations
    convective_term[..., 0] = U[..., 0] * grad_U[..., 0, 0] + U[..., 1] * grad_U[..., 0, 1]  # u * ∂u/∂x + v * ∂u/∂y
    convective_term[..., 1] = U[..., 0] * grad_U[..., 1, 0] + U[..., 1] * grad_U[..., 1, 1]  # u * ∂v/∂x + v * ∂v/∂y

    # Right-hand side of the Navier-Stokes equation
    rhs_U = -p_grad + viscous_term - surface_tension + convective_term  # Shape: (M, N, 2)
    if include_gravity:
        rhs_U += (1 / Fr**2) * np.stack([np.zeros_like(U[..., 0]), -np.ones_like(U[..., 1])], axis=-1)

    # Update velocity field using explicit Euler
    U = U + current_dt * rhs_U  # Shape: (M, N, 2)

    return U

def apply_velocity_boundary_conditions(U):
    """Apply physically appropriate boundary conditions to velocity field.
    
    Args:
        U (np.ndarray): Velocity field (shape: (Nx, Ny, 2)).
    
    Returns:
        np.ndarray: Velocity with boundary conditions applied.
    """
    # Make a copy to avoid modifying the original array
    U_new = U.copy()
    
    # Bottom boundary (solid wall): No-slip condition
    U_new[:, 0, :] = 0.0
    
    # Top boundary (open atmosphere): Zero-gradient condition
    U_new[:, -1, :] = U_new[:, -2, :]
    
    # Left and right boundaries: Zero-gradient condition
    U_new[0, :, :] = U_new[1, :, :]
    U_new[-1, :, :] = U_new[-2, :, :]
    
    return U_new 

def update_pressure(surface_tension, dx, dy):
    """Update the pressure field P based on the velocity field U and phase field phi."""
    sf_grad = utils.numerical_derivative(surface_tension[..., 0], axis=0, h=dx) +  \
              utils.numerical_derivative(surface_tension[..., 1], axis=1, h=dy)
    P = solve_poisson(sf_grad, dx, dy) 
    return P

def penalization(phi, alpha):
    # Apply penalization force if phi exceeds [-1, 1]
    mask_pos = phi > 1.0
    mask_neg = phi < -1.0
    
    penalty = np.zeros_like(phi)
    penalty[mask_pos] = (phi[mask_pos] - 1.0)
    penalty[mask_neg] = (phi[mask_neg] + 1.0)
    
    return alpha * penalty

def update_phase(phi, U, current_dt, dx, dy):
    """Update the phase field using the explicit Euler method.
    
    Args:
        phi (np.ndarray): Current phase field (shape: (Nx, Ny)).
        U (np.ndarray): Velocity field (shape: (Nx, Ny, 2)).
        Pe (float): Peclet number.
        epsilon (float): Small parameter for the equations.

    Returns:
        np.ndarray: Updated phase field (shape: (Nx, Ny)).
    """
    # Step 1: Calculate the gradient of phi
    grad_phi = utils.gradient(phi, dx, dy)  # Shape: (Nx, Ny, 2)

    # Step 2: Calculate the convective term
    convective_term = np.zeros_like(phi)  # Shape: (Nx, Ny)
    convective_term += U[..., 0] * grad_phi[..., 0] + U[..., 1] * grad_phi[..., 1]  # u * (∂phi/∂x + ∂phi/∂y)

    # Step 3: Calculate the Laplacian of phi
    lap_phi = utils.laplacian(phi, dx, dy)  # Shape: (Nx, Ny)

    # Step 4: Calculate the source term
    source_term = (-utils.f_2(phi) + epsilon**2 * lap_phi) / Pe  # Shape: (Nx, Ny)

    # Step 5: Right-hand side of the phase equation
    rhs_phi = -convective_term + source_term  # Shape: (Nx, Ny)

    # Step 6: Create Lagrangian multiplier
    # lagrange_multiplier = (-utils.f(phi) + epsilon**2 * lap_phi) 

    # Step 7: Update phase field using explicit Euler
    phi_new = phi + current_dt * (rhs_phi - penalization(phi, phase_penalty))  # Shape: (Nx, Ny)

    return phi_new  # Return the updated phase field 

def project_velocity(U, dx, dy, tolerance=1e-5, max_iterations=10):
    """Project velocity field with iterative refinement until divergence is sufficiently reduced."""
    U_projected = U.copy()
    
    for iteration in range(max_iterations):
        # Calculate divergence
        u_x = utils.numerical_derivative(U_projected[..., 0], axis=0, h=dx)
        v_y = utils.numerical_derivative(U_projected[..., 1], axis=1, h=dy)
        divergence = u_x + v_y
        
        max_div = np.max(np.abs(divergence))
        if max_div < tolerance:
            break  # Stop if divergence is small enough
        
        # Solve Poisson equation for pressure correction
        p_correction = solve_poisson(divergence, dx, dy)
        
        # Calculate gradient of pressure correction
        grad_p = utils.gradient(p_correction, dx, dy)
        
        # Project velocity field
        U_projected[..., 0] -= grad_p[..., 0]
        U_projected[..., 1] -= grad_p[..., 1]
        
        # Apply boundary conditions to projected velocity
        U_projected = apply_velocity_boundary_conditions(U_projected)
    
    return U_projected

def enforce_incompressibility(U, dx, dy, max_iterations=20, tolerance=1e-4):
    """More robust projection to enforce incompressibility"""
    for iter in range(max_iterations):
        # Calculate divergence
        u_x = utils.numerical_derivative(U[..., 0], axis=0, h=dx)
        v_y = utils.numerical_derivative(U[..., 1], axis=1, h=dy)
        div = u_x + v_y

        if iter > 100 and iter % 10 == 0:
            sys.stdout.write(f"Iteration {iter}, Max divergence: {np.max(np.abs(div))}\n")

        max_div = np.max(np.abs(div))
        if max_div < tolerance:
            break
            
        # Solve Poisson equation with stronger relaxation
        p_correction = utils.solve_poisson_with_better_bc(div, dx, dy)
        
        # Apply correction with relaxation factor
        relax = 1.0  # Relaxation factor for stability
        grad_p = utils.gradient(p_correction, dx, dy)
        U[..., 0] -= relax * grad_p[..., 0]
        U[..., 1] -= relax * grad_p[..., 1]
        
        # Apply boundary conditions after each iteration
        U = apply_velocity_boundary_conditions(U)
    return U
    
def initialize_phase(Nx, Ny, radius):
    """Initialize the phase field with a semicircle droplet resting on the bottom boundary."""
    # Create a grid of coordinates
    x = np.linspace(0, 1, Nx)  # X coordinates from 0 to 1
    y = np.linspace(0, 1, Ny)   # Y coordinates from 0 to 1
    X, Y = np.meshgrid(x, y, indexing='ij')  # Create a meshgrid with correct indexing
    
    # Define the center of the semicircle at the bottom center of the domain
    center_x = 0.5
    center_y = 0  # Bottom of the domain
    
    # Calculate distance from the center
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Use tanh to create a smooth transition between phases
    # Positive inside the semicircle (phi = 1), negative outside (phi = -1)
    phi = -np.tanh((distance - radius) * (10/radius))  # Scale factor for sharpness
    
    # Apply boundary condition to ensure semicircle sits exactly on the bottom boundary
    phi[:,0] = phi[:,1]  # Copy the first interior row to the boundary
    
    return phi

def get_borders_of_droplet(phi):
    """Get the  borders of the droplet."""
    # Find the first and last non-zero elements in each row=
    start_of_droplet = 0
    end_of_droplet = phi.shape[0] - 1
    for i in range(0, phi.shape[0]):
        if phi[i, 0] > 0:
            start_of_droplet = i
            break
    for i in range(phi.shape[0] - 1, 0, -1):
        if phi[i, 0] > 0:
            end_of_droplet = i
            break
    return start_of_droplet, end_of_droplet

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Droplet spreading simulation')
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON format)')
    parser.add_argument('--output', type=str, help='Path to output directory')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Access global variables
    global phi, Re1, Re2, Pe, epsilon, phase_penalty, contact_angle, Fr
    
    # Extract parameters from config
    # Physical parameters
    rho = config["physical_params"]["rho"]
    Re1 = config["physical_params"]["Re1"]
    Re2 = config["physical_params"]["Re2"]
    We = config["physical_params"]["We"]
    Pe = config["physical_params"]["Pe"]
    epsilon = config["physical_params"]["epsilon"]
    alpha = config["physical_params"]["alpha"]
    phase_penalty = config["physical_params"]["phase_penalty"]
    contact_angle = config["physical_params"]["contact_angle"]
    include_gravity = config["physical_params"]["include_gravity"]
    Fr = config["physical_params"]["Fr"]
    
    # Grid setup
    Lx, Ly = config["grid_params"]["Lx"], config["grid_params"]["Ly"]
    Nx, Ny = config["grid_params"]["Nx"], config["grid_params"]["Ny"]
    dx, dy = Lx / Nx, Ly / Ny
    
    # Time setup
    dt = config["time_params"]["dt"]
    t_max = config["time_params"]["t_max"]
    num_steps = int(t_max / dt)
    checkpoint_interval = config["time_params"]["checkpoint_interval"]
    dt_initial = config["time_params"]["dt_initial"]
    
    # Initial conditions
    radius = config["initial_conditions"]["droplet_radius"]
    
    # Restart information
    restart_from = config["restart"]["restart_from"]
    if restart_from == "None":
        restart_from = None
    
    # Create a directory for the experiment
    if args.output:
        # Use provided output directory
        login_dir = args.output
        os.makedirs(login_dir, exist_ok=True)
    else:
        # Use timestamped directory (original behavior)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        login_dir = f"experiment_{timestamp}"
        os.makedirs(login_dir, exist_ok=True)

    # Initialize the phase field with a semicircle droplet
    phi = initialize_phase(Nx, Ny, radius)  # Initialize the phase field
    phi = utils.apply_contact_angle_boundary_conditions(phi, dx, dy, contact_angle=contact_angle)

    # Initialize fields
    U = np.zeros((Nx, Ny, 2))  # Velocity field (2D vector field)
    P = np.zeros((Nx, Ny))  # Pressure field

    # Visualization of the phase field
    plt.imshow(phi.T, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
    plt.colorbar(label='Phase Field (phi)')
    plt.title('Phase Field Visualization')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(f'{login_dir}/initial_phase_field.png', bbox_inches='tight')
    plt.clf()

    # Create a directory for checkpoints
    checkpoint_dir = os.path.join(login_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the configuration to a JSON file
    params_file = os.path.join(login_dir, "simulation_parameters.json")
    with open(params_file, 'w') as f:
        json.dump(config, f, indent=4)
    sys.stdout.write(f"Simulation parameters saved to {params_file}\n")

    # Optional: restart from checkpoint
    if restart_from is not None:
        sys.stdout.write(f"Restarting from checkpoint: {restart_from} \n")
        checkpoint_data = utils.load_checkpoint(restart_from)
        
        # Load simulation state
        start_step = checkpoint_data['step']
        phi = checkpoint_data['phi']
        U = checkpoint_data['U']
        P = checkpoint_data['P']
        
        sys.stdout.write(f"Loaded state from step {start_step}, continuing simulation... \n")
    else:
        # Initialize from scratch
        start_step = 0

    times = []
    # Main simulation loop
    for step in range(start_step, num_steps):
        # Use dt_initial for the first 100 steps
        # Compute derivatives and other terms
        start_time = time.time()
        current_dt = dt_initial if step < 500 else dt
        surface_tension = utils.surface_tension_force(phi, epsilon, We, dx, dy)
        surface_tension = utils.apply_surface_tension_boundary_conditions(surface_tension, phi, contact_angle=contact_angle)
        
        U = update_velocity(U, P, surface_tension, current_dt, dx, dy, include_gravity=include_gravity)

        # Project velocity to ensure continuity is satisfied
        max_iterations = 1000 if step % 10 == 0 else 100
        U = enforce_incompressibility(U, dx, dy, tolerance=1e-4, max_iterations=max_iterations)

        # Add special handling for extreme divergence points
        divergence, max_div, mean_div = check_continuity(U, dx, dy)
        if max_div > 5.0:
            # Find locations of extreme divergence
            extreme_mask = np.abs(divergence) > 5.0
            if np.any(extreme_mask):
                # Apply local smoothing to problematic areas
                for i in range(2):
                    smooth_field = U[..., i].copy()
                    smooth_field[1:-1, 1:-1][extreme_mask[1:-1, 1:-1]] = (
                        U[0:-2, 1:-1, i][extreme_mask[1:-1, 1:-1]] + 
                        U[2:, 1:-1, i][extreme_mask[1:-1, 1:-1]] + 
                        U[1:-1, 0:-2, i][extreme_mask[1:-1, 1:-1]] + 
                        U[1:-1, 2:, i][extreme_mask[1:-1, 1:-1]]
                    ) / 4.0
                    U[..., i] = smooth_field

        # Apply no-slip boundary conditions
        U = apply_velocity_boundary_conditions(U)

        P = update_pressure(surface_tension, dx, dy)
        P = utils.apply_pressure_boundary_conditions(P)
        phi = update_phase(phi, U, current_dt, dx, dy)

        # Apply contact angle boundary conditions
        phi = utils.apply_contact_angle_boundary_conditions(phi, dx, dy, contact_angle=contact_angle)
        end_time = time.time()
        times.append(end_time - start_time)
        
        # Print or log results for analysis
        if step % 10 == 0:  # Print every 10 steps
            sys.stdout.write(f"Step {step}, Time {step * current_dt:.2f} \n")
            sys.stdout.write(f"Min/Max of U: {U.min():.4f} / {U.max():.4f} \n")
            sys.stdout.write(f"Min/Max of P: {P.min():.4f} / {P.max():.4f} \n")
            sys.stdout.write(f"Min/Max of phi: {phi.min():.4f} / {phi.max():.4f} \n")
            divergence, max_div, mean_div = check_continuity(U, dx, dy)
            sys.stdout.write(f"Continuity check - Max |div(U)|: {max_div:.6f}, Mean |div(U)|: {mean_div:.6f} \n")
            sys.stdout.write(f"Time: {np.mean(times):.6f} \n")
            start_of_droplet, end_of_droplet = get_borders_of_droplet(phi)
            mass = np.sum(phi[phi > 0])
            sys.stdout.write(f"Droplet mass {mass} \n")
            sys.stdout.write(f"Start/end of the droplet on the surface: {start_of_droplet} - {end_of_droplet} \n")

        if step % 25 == 0:  # Plot every 25 steps
            mass = np.sum(phi[phi > 0])
            utils.create_joint_plot(
                phi, U, P, surface_tension, current_dt, step, dx, dy, mass,
                save_path=f'{login_dir}/joint_plot_step_{step}.png'
            )

        # Save checkpoint at specified intervals
        if step % checkpoint_interval == 0:
            utils.save_checkpoint(step, phi, U, P, directory=f"{login_dir}/checkpoints")

if __name__ == "__main__":
    main()