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
from scipy.sparse import kron, identity, linalg

# Global variables used by multiple functions
phi = None
Re1 = None
Re2 = None
Pe = None
epsilon = None
phase_penalty = None
contact_angle = None
Fr = None
g = None
atm_pressure = None

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


def build_2d_laplacian_matrix(N, h=1.0, bc_type='dirichlet'):
    """
    Constructs the 2D Laplacian matrix using Kronecker product and scipy.diags,
    with optional Dirichlet or Neumann boundary conditions.

    Parameters:
        N (int): Number of interior grid points per dimension
        h (float): Grid spacing (default 1.0)
        bc_type (str): 'dirichlet' or 'neumann'

    Returns:
        scipy.sparse.csr_matrix: Sparse Laplacian matrix of shape (N^2, N^2)
    """
    # 1D Laplacian
    main_diag = -2 * np.ones(N)
    off_diag = np.ones(N - 1)
    T = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N))

    if bc_type == 'neumann':
        T = T.tolil()
        T[0, 1] = 2  # Left boundary mirror
        T[-1, -2] = 2  # Right boundary mirror
        T = T.tocsr()

    I = identity(N)
    A = (kron(I, T) + kron(T, I)) / h**2
    return A

def solve_poisson(rhs, dx, dy):
    """Solve the Poisson equation ∇²φ = f with Neumann boundary conditions."""
    A = utils.build_2d_laplacian_matrix_with_variable_steps(rhs.shape[0], rhs.shape[1], dx, dy, 'neumann')
    
    # Reshape right-hand side to match matrix equation
    rhs_flat = np.transpose(rhs).flatten(order='C')  # Use C-style ordering (row-major)
    
    # Solve the linear system
    phi_flat = linalg.spsolve(A, rhs_flat)
    
    # Reshape solution to 2D - use proper ordering
    phi = phi_flat.reshape((rhs.shape[1], rhs.shape[0]), order='C')
    phi = np.transpose(phi)
    
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

def update_velocity(U, p, surface_tension, current_dt, dx, dy, rho1, rho2, include_gravity=False):
    """Update the velocity field U based on the phase field phi."""
    # Calculate the Reynolds number and density
    Re = utils.calculate_reynolds_number(phi, Re1, Re2)
    rho = utils.calculate_density(phi, rho1, rho2)
    rho_stacked = np.stack([rho, rho], axis=-1) + 1e-6

    # Calculate gradients and terms
    grad_U = utils.gradient(U, dx, dy)
    p_grad = utils.gradient(p, dx, dy)
    
    # Calculate viscous term with proper scaling
    viscous_term = compute_viscous_term(U, dx, dy, Re)
    
    # Calculate convective term (in conservative form)
    convective_term = np.zeros_like(U)
    convective_term[..., 0] = (U[..., 0] * grad_U[..., 0, 0] + U[..., 1] * grad_U[..., 0, 1])
    convective_term[..., 1] = (U[..., 0] * grad_U[..., 1, 0] + U[..., 1] * grad_U[..., 1, 1])

    # Combine terms with proper density scaling
    rhs_U = (
        -p_grad / rho_stacked +  # Pressure term
        viscous_term / rho_stacked +  # Viscous term
        -surface_tension / rho_stacked +  # Surface tension
        -convective_term  # Convective term (already includes velocity)
    )

    # Add gravity if included
    if include_gravity:
        rhs_U += (1 / Fr) * np.stack([np.zeros_like(U[..., 0]), -np.ones_like(U[..., 1])], axis=-1)

    # Update velocity field using explicit Euler
    U = U + current_dt * rhs_U
    
    return U

def apply_velocity_boundary_conditions(U, beta, dy):
    """Apply physically appropriate boundary conditions to velocity field.
    
    Args:
        U (np.ndarray): Velocity field (shape: (Nx, Ny, 2)).
    
    Returns:
        np.ndarray: Velocity with boundary conditions applied.
    """
    # Make a copy to avoid modifying the original array
    U_new = U.copy()
    
    # Bottom boundary (solid wall): Slip condition
    U_new[:, 0, 1] = 0.0 # y
    U_new[:, 0, 0] = U_new[:, 1, 0] - dy * 1/beta * U_new[:, 1, 0]  # x

    # Top boundary (open atmosphere): Zero-gradient condition
    U_new[:, -1, :] = U_new[:, -2, :]
    
    # Left and right boundaries: Zero-gradient condition
    # U_new[0, :, :] = U_new[1, :, :]
    # U_new[-1, :, :] = U_new[-2, :, :]
    U_new[0, :, :] = 0.0
    U_new[-1, :, :] = 0.0
    
    return U_new 

def update_pressure(surface_tension, dx, dy, rho1, rho2):
    """Update the pressure field P based on the velocity field U and phase field phi."""
    sf_grad = utils.numerical_derivative(surface_tension[..., 0], axis=0, h=dx) +  \
              utils.numerical_derivative(surface_tension[..., 1], axis=1, h=dy)
    rho = utils.calculate_density(phi, rho1, rho2)
    sf_grad = sf_grad

    # Apply boundary conditions
    sf_grad[:, 0] = np.sum(rho * g * dy, axis=-1)
    sf_grad[:, 0] += atm_pressure

    sf_grad[:, -1] = atm_pressure

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

def update_phase(phi, U, current_dt, dx, dy, contact_angle):
    """Update the phase field using the explicit Euler method with interface thickness control.
    
    Args:
        phi (np.ndarray): Current phase field (shape: (Nx, Ny)).
        U (np.ndarray): Velocity field (shape: (Nx, Ny, 2)).
        Pe (float): Peclet number.
        epsilon (float): Interface thickness parameter.
    """
    # Step 1: Calculate the gradient of phi
    grad_phi = utils.gradient(phi, dx, dy)  # Shape: (Nx, Ny, 2)
    # grad_phi_mag = np.sqrt(grad_phi[..., 0]**2 + grad_phi[..., 1]**2)

    # Step 2: Calculate the convective term
    convective_term = np.zeros_like(phi)  # Shape: (Nx, Ny)
    convective_term += U[..., 0] * grad_phi[..., 0] + U[..., 1] * grad_phi[..., 1]

    # Step 3: Calculate the Laplacian of phi
    lap_phi = utils.laplacian(phi, dx, dy)

    # Step 4: Calculate stabilized chemical potential with interface thickness control
    # This formulation helps maintain the interface thickness
    chemical_potential = utils.df_2(phi) - epsilon**2 * lap_phi
    lagrange_multiplier = np.mean(chemical_potential)
    source_term = -1/Pe * (chemical_potential - lagrange_multiplier)

    # Add stabilization term to maintain interface thickness
    # stabilization = epsilon * (grad_phi_mag - 1/epsilon) * (grad_phi_mag > 1/epsilon)
    # chemical_potential += stabilization

    # Step 5: Calculate the source term using the stabilized chemical potential
    # chem_grad = utils.gradient(chemical_potential, dx, dy)
    # div_chem = utils.numerical_derivative(chem_grad[..., 0], axis=0, h=dx) + \
    #            utils.numerical_derivative(chem_grad[..., 1], axis=1, h=dy)
    # source_term = div_chem / Pe

    # Step 6: Right-hand side of the phase equation
    rhs_phi = -convective_term + source_term

    # Step 7: Update phase field
    phi = phi + current_dt * rhs_phi

    # Step 8: Apply boundary conditions and maintain phase field bounds
    phi = utils.apply_contact_angle_boundary_conditions(phi, dx, dy, contact_angle=contact_angle)
    
    # Ensure phi stays within physical bounds [-1, 1]
    phi = np.clip(phi, -1.0, 1.0)

    return phi

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

def correction_step(U, dx, dy, dt):
    """PPE method for incompressible flow."""
    u_x = utils.numerical_derivative(U[..., 0], axis=0, h=dx)
    v_y = utils.numerical_derivative(U[..., 1], axis=1, h=dy)
    div = u_x + v_y
    p_correction = utils.solve_poisson_with_better_bc(div/dt, dx, dy)
    U[..., 0] -= dt * utils.numerical_derivative(p_correction, axis=0, h=dx)
    U[..., 1] -= dt * utils.numerical_derivative(p_correction, axis=1, h=dy)
    return U

def ppe(U, dx, dy, dt):
    # global correction_step    
    U = correction_step(U, dx, dy, dt)
    U = apply_velocity_boundary_conditions(U, 0.01, dy)

    # local corrections
    divergence, max_div, mean_div = check_continuity(U, dx, dy)
    if max_div > 100:
        half = int(U.shape[1] / 2)
        to_replace = int(U.shape[1] * 0.4)
        count = 0
        while max_div > 100:
            U = np.clip(U, -100, 100)
            U = correction_step(U, dx, dy, dt)
            U = apply_velocity_boundary_conditions(U, 0.01, dy)
            divergence, max_div, mean_div = check_continuity(U, dx, dy)
            if max_div < 100:
                break
            count += 1
        sys.stdout.write(f"Corrected in {count} iterations \n")
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
    global phi, Re1, Re2, Pe, epsilon, phase_penalty, contact_angle, Fr, g, atm_pressure
    
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
    alpha = config["physical_params"]["alpha"]
    phase_penalty = config["physical_params"]["phase_penalty"]
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
        surface_tension = utils.surface_tension_force(phi, epsilon, We1, We2, dx, dy)
        surface_tension = utils.apply_surface_tension_boundary_conditions(surface_tension, phi, contact_angle=contact_angle)
        
        U = update_velocity(U, P, surface_tension, current_dt, dx, dy, rho1, rho2, include_gravity=include_gravity)

        # Project velocity to ensure continuity is satisfied
        beta = 0.01
        U = apply_velocity_boundary_conditions(U, beta, dy)

        # Add special handling for extreme divergence points
        divergence, max_div, mean_div = check_continuity(U, dx, dy)
        
        # Apply PPE
        U = ppe(U, dx, dy, current_dt)

        # beta = 0.01
        # Apply no-slip boundary conditions
        # U = apply_velocity_boundary_conditions(U, beta, dy)

        # Apply contact angle boundary conditions
        phi = update_phase(phi, U, current_dt, dx, dy, contact_angle)
        
        # Recompute surface tension
        surface_tension = utils.surface_tension_force(phi, epsilon, We1, We2, dx, dy)
        surface_tension = utils.apply_surface_tension_boundary_conditions(surface_tension, phi, contact_angle=contact_angle)

        # Update pressure
        P = update_pressure(surface_tension, dx, dy, rho1, rho2)
        
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

        if step % checkpoint_interval == 0:  # Plot every 25 steps
            mass = np.sum(phi[phi > 0])
            utils.create_joint_plot(
                phi, U, P, surface_tension, current_dt, step, dx, dy, mass, rho1, rho2,
                save_path=f'{login_dir}/joint_plot_step_{step}.png'
            )

        # Save checkpoint at specified intervals
        if step % checkpoint_interval == 0:
            utils.save_checkpoint(step, phi, U, P, directory=f"{login_dir}/checkpoints")

if __name__ == "__main__":
    main()