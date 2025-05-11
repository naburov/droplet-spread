import os
import argparse
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import utils
 # Assuming utils.py is in the same directory
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time 
import sys
import jax.numpy as jnp
import jax
from jax import jit
from scipy.sparse import kron, identity, linalg
from jax_utils import (
    jax_calculate_density,
    jax_calculate_reynolds_number,
    jax_calculate_weber_number,
    jax_dx,
    jax_dy,
    jax_apply_contact_angle_boundary_conditions,
    jax_build_2d_laplacian_matrix_with_variable_steps,
    jax_laplacian,
    jax_gradient,
    jax_divergence,
    jax_df_2,
    jax_f_2,
    jax_surface_tension_force,
    solve_poisson_pyamg,
    jax_apply_surface_tension_boundary_conditions,
)
from plot_utils import (
    create_joint_plot, 
    save_checkpoint, 
    list_checkpoints, 
    load_checkpoint, 
    load_config
)

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

def solve_poisson(rhs, Nx, Ny, dx, dy):
    """Solve the Poisson equation ∇²φ = f with Neumann boundary conditions."""
    A = jax_build_2d_laplacian_matrix_with_variable_steps(Nx, Ny, dx, dy)
    
    # Reshape right-hand side to match matrix equation
    rhs_flat = np.transpose(rhs).flatten(order='C')  # Use C-style ordering (row-major)
    
    # Solve the linear system
    phi_flat = spsolve(A, rhs_flat)
    
    # Reshape solution to 2D - use proper ordering
    phi = phi_flat.reshape((Nx, Ny), order='C')
    phi = np.transpose(phi)
    
    return phi

@jit
def compute_viscous_term(U, dx, dy, Re):
    """Simplified viscous term for constant viscosity: (1/Re) * ∇²U"""
    return jnp.stack([jax_laplacian(U[..., 0], dx, dy) / Re, 
                      jax_laplacian(U[..., 1], dx, dy) / Re], axis=-1)

@jit
def check_continuity(U, dx, dy):
    """
    Check continuity equation condition (∇·U = 0)
    Returns the divergence field and maximum absolute divergence
    """
    # Calculate divergence: du/dx + dv/dy
    u_x = jax_dx(U[..., 0], h=dx)
    v_y = jax_dy(U[..., 1], h=dy)
    
    divergence = u_x + v_y
    max_div = jnp.max(jnp.abs(divergence))
    mean_div = jnp.mean(jnp.abs(divergence))
    
    return divergence, max_div, mean_div

def jax_update_velocity(U, p, surface_tension, current_dt, dx, dy, rho1, rho2, include_gravity=False):
    """Update the velocity field U based on the phase field phi."""
    # Calculate the Reynolds number and density
    Re = jax_calculate_reynolds_number(phi, Re1, Re2)
    rho = jax_calculate_density(phi, rho1, rho2)
    rho_stacked = jnp.stack([rho, rho], axis=-1) + 1e-6

    # Calculate gradients and terms
    grad_U = jax_gradient(U, dx, dy)
    p_grad = jax_gradient(p, dx, dy)
    
    # Calculate viscous term with proper scaling
    viscous_term = compute_viscous_term(U, dx, dy, Re)
    
    # Calculate convective term (in conservative form)
    convective_term = jnp.stack(
        [
            U[..., 0] * grad_U[..., 0, 0] + U[..., 1] * grad_U[..., 0, 1],
            U[..., 0] * grad_U[..., 1, 0] + U[..., 1] * grad_U[..., 1, 1]
        ],
        axis=-1
    )

    # Combine terms with proper density scaling
    rhs_U = (
        -p_grad / rho_stacked +  # Pressure term
        viscous_term / rho_stacked +  # Viscous term
        -surface_tension / rho_stacked +  # Surface tension
        -convective_term  # Convective term (already includes velocity)
    )

    # Add gravity if included
    if include_gravity:
        rhs_U = rhs_U + (1 / Fr) * jnp.stack([jnp.zeros_like(U[..., 0]), -jnp.ones_like(U[..., 1])], axis=-1)

    # Update velocity field using explicit Euler
    U = U + current_dt * rhs_U
    
    return U

update_velocity = jit(jax_update_velocity, static_argnames='include_gravity')

@jit
def apply_velocity_boundary_conditions(U, beta, dy):
    """Apply physically appropriate boundary conditions to velocity field.
    
    Args:
        U (np.ndarray): Velocity field (shape: (Nx, Ny, 2)).
    
    Returns:
        np.ndarray: Velocity with boundary conditions applied.
    """
    # Make a copy to avoid modifying the original array
    U = U.at[:, 0, 1].set(0.0)
    U = U.at[:, 0, 0].set(U[:, 1, 0] - dy * 1/beta * U[:, 1, 0])

    # Top boundary (open atmosphere): Zero-gradient condition
    U = U.at[:, -1, :].set(U[:, -2, :])
    
    # Left and right boundaries: Zero-gradient condition
    U = U.at[0, :, :].set(0.0)
    U = U.at[-1, :, :].set(0.0)
    
    return U

def update_pressure(surface_tension, Nx, Ny, dx, dy, rho1, rho2):
    """Update the pressure field P based on the velocity field U and phase field phi."""
    sf_grad = jax_divergence(surface_tension, dx, dy)
    rho = jax_calculate_density(phi, rho1, rho2)

    sf_grad = sf_grad.at[:, 0].set(jnp.sum(rho * g * dy, axis=1) + atm_pressure)
    sf_grad = sf_grad.at[:, -1].set(atm_pressure)

    P = solve_poisson(sf_grad, Nx, Ny, dx, dy) 
    return P

def penalization(phi, alpha):
    # Apply penalization force if phi exceeds [-1, 1]
    mask_pos = phi > 1.0
    mask_neg = phi < -1.0
    
    penalty = np.zeros_like(phi)
    penalty[mask_pos] = (phi[mask_pos] - 1.0)
    penalty[mask_neg] = (phi[mask_neg] + 1.0)
    
    return alpha * penalty

@jit
def update_phase(phi, U, current_dt, dx, dy, contact_angle):
    """Update the phase field using the explicit Euler method with interface thickness control.
    
    Args:
        phi (np.ndarray): Current phase field (shape: (Nx, Ny)).
        U (np.ndarray): Velocity field (shape: (Nx, Ny, 2)).
        Pe (float): Peclet number.
        epsilon (float): Interface thickness parameter.
    """
    # Step 1: Calculate the gradient of phi
    grad_phi = jax_gradient(phi, dx, dy)  # Shape: (Nx, Ny, 2)
    # grad_phi_mag = np.sqrt(grad_phi[..., 0]**2 + grad_phi[..., 1]**2)

    # Step 2: Calculate the convective term
    convective_term = U[..., 0] * grad_phi[..., 0] + U[..., 1] * grad_phi[..., 1]

    # Step 3: Calculate the Laplacian of phi
    lap_phi = jax_laplacian(phi, dx, dy)

    # Step 4: Calculate stabilized chemical potential with interface thickness control
    # This formulation helps maintain the interface thickness
    chemical_potential = jax_df_2(phi) - epsilon**2 * lap_phi
    lagrange_multiplier = jnp.mean(chemical_potential)
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
    phi = jax_apply_contact_angle_boundary_conditions(phi, dx, dy, contact_angle=contact_angle)
    
    # Ensure phi stays within physical bounds [-1, 1]
    phi = jnp.clip(phi, -1.0, 1.0)

    return phi

def correction_step(U, dx, dy, dt, solution=None):
    """PPE method for incompressible flow."""
    u_x = jax_dx(U[..., 0], h=dx)
    v_y = jax_dy(U[..., 1], h=dy)
    div = u_x + v_y

    # p_correction = utils.solve_poisson_with_better_bc(div/dt, dx, dy)
    # p_correction = utils.solve_poisson_pyro(div/dt, dx, dy, solution=solution)
    p_correction = utils.solve_poisson_pyamg(np.array(div/dt), dx, dy, solution=solution)
    U = U.at[..., 0].set(U[..., 0] - dt * jax_dx(p_correction, h=dx))
    U = U.at[..., 1].set(U[..., 1] - dt * jax_dy(p_correction, h=dy))
    return U, p_correction

def damp_divergence(U, dx, dy, xi, dt):
    u_x = jax_dx(U[..., 0], h=dx)
    v_y = jax_dy(U[..., 1], h=dy)
    div = u_x + v_y
    div_grad = jax_gradient(div, dx, dy)
    U = U.at[..., 0].set(U[..., 0] - xi * dt * div_grad[..., 0])
    U = U.at[..., 1].set(U[..., 1] - xi * dt * div_grad[..., 1])
    return U

def ppe(U, dx, dy, dt, solution=None):
    max_div_threshold = 5
    # global correction_step    
    U = jnp.clip(U, -1000, 1000)
    U, solution = correction_step(U, dx, dy, dt, solution=solution)
    U = apply_velocity_boundary_conditions(U, 0.01, dy)

    # local corrections
    divergence, max_div, mean_div = check_continuity(U, dx, dy)
    if max_div > max_div_threshold:
        # U = U.astype(np.half)
        half = int(U.shape[1] / 2)
        to_replace = int(U.shape[1] * 0.4)
        count = 0
        while max_div > max_div_threshold:
            U = jnp.clip(U, -1000, 1000)
            U, solution = correction_step(U, dx, dy, dt, solution=solution)
            U = apply_velocity_boundary_conditions(U, 0.01, dy)
            divergence, max_div, mean_div = check_continuity(U, dx, dy)
            if count % 5 == 0:
                sys.stdout.write(f"\rMax|mean div: {max_div:.6f}  | {mean_div:.6f}")
            if max_div < max_div_threshold:
                break
            count += 1
        sys.stdout.write(f"\nCorrected in {count} iterations \n")
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

@jit
def cfl_dt(u_max, v_max, dx, dy, C=0.4):
    """Return Δt that gives desired CFL=C."""
    return C / (jnp.abs(u_max)/dx + jnp.abs(v_max)/dy)

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
        checkpoint_data = load_checkpoint(restart_from)
        
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

        cfl_computed_dt = cfl_dt(U[..., 0].max(), U[..., 1].max(), dx, dy, C=0.1)
        print(f"Current dt: {current_dt}")
        if cfl_computed_dt != np.inf:
            current_dt = cfl_computed_dt
            print(f"Updated dt: {current_dt}")

        surface_tension = jax_surface_tension_force(phi, epsilon, We1, We2, dx, dy)
        surface_tension = jax_apply_surface_tension_boundary_conditions(surface_tension, phi, contact_angle=contact_angle)
        
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
        surface_tension = jax_surface_tension_force(phi, epsilon, We1, We2, dx, dy)
        surface_tension = jax_apply_surface_tension_boundary_conditions(surface_tension, phi, contact_angle=contact_angle)

        # Update pressure
        P = update_pressure(surface_tension, Nx, Ny, dx, dy, rho1, rho2)
        
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
            create_joint_plot(
                phi, U, P, surface_tension, current_dt, step, dx, dy, mass, rho1, rho2,
                save_path=f'{login_dir}/joint_plot_step_{step}.png'
            )

        # Save checkpoint at specified intervals
        if step % checkpoint_interval == 0:
            save_checkpoint(step, phi, U, P, directory=f"{login_dir}/checkpoints")

if __name__ == "__main__":
    main()