import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import utils  # Assuming utils.py is in the same directory
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
# import jax.numpy as jnp
import time 

# Constants
rho = 1.0  # Density
Re1 = 1000.0  # Reynolds number
Re2 = 10.0  # Reynolds number
eta = lambda phi: 1.0  # Viscosity as a function of phi
We = 10.0  # Weber number
Pe = 1.0  # Peclet number
epsilon = 0.05  # Small parameter for the equations
alpha = 1.0  # Coefficient for curvature term
phase_penalty = 1000.0  # Coefficient for phase penalty
contact_angle = 120  # Contact angle
# Grid setup
Lx, Ly = 1.0, 1.0  # Domain size
Nx, Ny = 100, 100  # Number of grid points
dx, dy = Lx / Nx, Ly / Ny  # Grid spacing

# Time setup
dt = 0.001  # Time step
t_max = 1.0  # Maximum time
num_steps = int(t_max / dt)  # Number of time steps

# Initialize fields
U = np.zeros((Nx, Ny, 2))  # Velocity field (2D vector field)
# Initialize pressure field
P = np.zeros((Nx, Ny))  # Pressure field

# Create a directory for the experiment with a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
login_dir = f"experiment_{timestamp}"

# Create the directory if it doesn't exist
os.makedirs(login_dir, exist_ok=True)

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

def update_velocity(U, p, surface_tension, current_dt, dx, dy):
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

    # print(p_grad[0,:].max(), viscous_term[0,:].max(), surface_tension[0,:].max(), convective_term[0,:].max())
    # Right-hand side of the Navier-Stokes equation
    rhs_U = -p_grad + viscous_term - surface_tension + convective_term  # Shape: (M, N, 2)

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
    source_term = (-utils.f(phi) + epsilon**2 * lap_phi) / Pe  # Shape: (Nx, Ny)

    # Step 5: Right-hand side of the phase equation
    rhs_phi = -convective_term + source_term  # Shape: (Nx, Ny)

    # Step 6: Update phase field using explicit Euler
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
            print(f"Iteration {iter}, Max divergence: {np.max(np.abs(div))}")

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


# Initialize the phase field with a semicircle droplet
radius = 0.2  # Radius of the semicircle
checkpoint_interval = 50
phi = initialize_phase(Nx, Ny, radius)  # Initialize the phase field
phi = utils.apply_contact_angle_boundary_conditions(phi, dx, dy, contact_angle=contact_angle)


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

# Dump simulation parameters to a JSON file
import json

# Collect all simulation parameters
simulation_params = {
    "physical_params": {
        "rho": rho,
        "Re1": Re1,
        "Re2": Re2,
        "We": We,
        "Pe": Pe,
        "epsilon": epsilon,
        "alpha": alpha,
        "phase_penalty": phase_penalty,
        "contact_angle": contact_angle
    },
    "grid_params": {
        "Lx": Lx,
        "Ly": Ly,
        "Nx": Nx,
        "Ny": Ny,
        "dx": dx,
        "dy": dy
    },
    "time_params": {
        "dt": dt,
        "t_max": t_max,
        "num_steps": num_steps,
        "checkpoint_interval": checkpoint_interval
    },
    "initial_conditions": {
        "droplet_radius": radius
    }
}

# Save parameters to JSON file
params_file = os.path.join(login_dir, "simulation_parameters.json")
with open(params_file, 'w') as f:
    json.dump(simulation_params, f, indent=4)

print(f"Simulation parameters saved to {params_file}")


# Optional: restart from checkpoint
# restart_from = '/Users/burovnikita/Desktop/Study/PhD/droplet/droplet_spreading_modeling/experiment_20250320_225309/checkpoints/checkpoint_000500.npz'  # Set to checkpoint path to restart, e.g. "experiment_20231215_123456/checkpoints/checkpoint_000500.npz"
# restart_from = None
# restart_from = '/Users/burovnikita/Desktop/Study/PhD/droplet/droplet_spreading_modeling/experiment_20250321_215442/checkpoints/checkpoint_000100.npz'
restart_from = None
# restart_from = "/Users/burovnikita/Desktop/Study/PhD/droplet/droplet_spreading_modeling/experiment_20250321_231311/checkpoints/checkpoint_000500.npz"

if restart_from is not None:
    print(f"Restarting from checkpoint: {restart_from}")
    checkpoint_data = utils.load_checkpoint(restart_from)
    
    # Load simulation state
    start_step = checkpoint_data['step']
    phi = checkpoint_data['phi']
    U = checkpoint_data['U']
    P = checkpoint_data['P']
    
    print(f"Loaded state from step {start_step}, continuing simulation...")
else:
    # Initialize from scratch (your existing initialization code)
    start_step = 0
    # Initialize phi, U, P as you already do

dt_initial = 0.0005
times = []
# Then modify your main loop:
for step in range(start_step, num_steps):
    # Use dt_initial for the first 100 steps
    # Compute derivatives and other terms
    start_time = time.time()
    current_dt = dt_initial if step < 100 else dt_initial
    surface_tension = utils.surface_tension_force(phi, epsilon, We, dx, dy)
    surface_tension = utils.apply_surface_tension_boundary_conditions(surface_tension, phi, contact_angle=contact_angle)
    
    U = update_velocity(U, P, surface_tension, current_dt, dx, dy)

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
        print(f"Step {step}, Time {step * current_dt:.2f}")
        print(f"Min/Max of U: {U.min():.4f} / {U.max():.4f}")
        print(f"Min/Max of P: {P.min():.4f} / {P.max():.4f}")
        print(f"Min/Max of phi: {phi.min():.4f} / {phi.max():.4f}")
        divergence, max_div, mean_div = check_continuity(U, dx, dy)
        print(f"Continuity check - Max |∇·U|: {max_div:.6f}, Mean |∇·U|: {mean_div:.6f}")
        print(f"Time: {np.mean(times):.6f}")

    if step % 25 == 0:  # Plot every 25 steps
        utils.create_joint_plot(
            phi, U, P, surface_tension, current_dt, step, dx, dy,
            save_path=f'{login_dir}/joint_plot_step_{step}.png'
        )

    # Save checkpoint at specified intervals
    # Save every 50 steps
    if step % checkpoint_interval == 0:
        utils.save_checkpoint(step, phi, U, P, directory=f"{login_dir}/checkpoints")
