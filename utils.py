import numpy as np
from scipy.sparse import diags
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter
import os
from datetime import datetime
from scipy.sparse.linalg import cg

# import jax.numpy as jnp
# from jax import jit

# @jit
# def numerical_derivative(f, axis=0, h=1e-5):
#     """Calculate the numerical derivative of f along a specified axis using central difference with JIT compilation."""
#     return ((jnp.roll(f, -1, axis=axis) - jnp.roll(f, 1, axis=axis)) / (2 * h)).astype('float32')

def numerical_derivative(f, axis=0, h=1e-5):
    """Calculate the numerical derivative of f along a specified axis using central difference."""
    return ((np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * h)).astype('float32')

# numerical_derivative = jit(numerical_derivative)

def laplacian(f, dx=1e-5, dy=1e-5):
    """Calculate the Laplacian of f at all points in a 2D field using finite differences."""
    lap = np.zeros_like(f)
    
    # Interior points using finite difference
    lap[1:-1, 1:-1] = (
        (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / (dx**2) +  # x-direction
        (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / (dy**2)    # y-direction
    )
    
    # Handle boundaries (Neumann condition: zero-gradient)
    lap[:, 0] = lap[:, 1]      # Bottom boundary (y=0) 
    lap[:, -1] = lap[:, -2]    # Top boundary (y=max)
    lap[0, :] = lap[1, :]      # Left boundary (x=0)
    lap[-1, :] = lap[-2, :]    # Right boundary (x=max)
    
    return lap

def divergence(f, dx, dy):
    """Calculate the divergence of a 2D vector field f at all points."""
    div = np.zeros(*f.shape)  # Assuming f is a 2D vector field with shape (M, N, 2)
    div[1:-1, 1:-1] = (np.roll(f[..., 0], -1, axis=1)[1:-1, 1:-1] - np.roll(f[..., 0], 1, axis=1)[1:-1, 1:-1]) / (2 * dy) + \
                      (np.roll(f[..., 1], -1, axis=0)[1:-1, 1:-1] - np.roll(f[..., 1], 1, axis=0)[1:-1, 1:-1]) / (2 * dx)
    
    # Handle boundaries (Neumann condition: zero-gradient)
    div[:, 0] = div[:, 1]  # Bottom boundary
    div[:, -1] = div[:, -2]  # Top boundary
    div[0, :] = div[1, :]  # Left boundary
    div[-1, :] = div[-2, :]  # Right boundary
    
    return div

def gradient(f, dx, dy):
    """Calculate the gradient of a scalar field f at all points in a 2D field."""
    grad = np.zeros((*f.shape, 2))
    grad[:, :, 0] = numerical_derivative(f, axis=0, h=dx)  # Gradient in x-direction
    grad[:, :, 1] = numerical_derivative(f, axis=1, h=dy)  # Gradient in y-direction
    return grad

def norm(f):
    """Calculate the norm of a 2D vector field f at all points."""
    return np.sqrt(f[:, :, 0]**2 + f[:, :, 1]**2)

def surface_tension_force(phi, epsilon, We, dx, dy):
    """Calculate the surface tension force based on the phase field.
    
    The surface tension force is given by:
    
    F_{\text{tension}} = \frac{3\sqrt{2}\epsilon}{4We} \nabla \cdot \left( \frac{\nabla \phi}{|\nabla \phi|} \right) |\nabla \phi| \nabla \phi
    """
    # Step 1: Calculate the curvature using the curvature function
    curvature_value = improved_curvature(phi, dx, dy)  # Shape: (M, N)

    # Step 2: Calculate the gradient of phi
    grad_phi = gradient(phi, dx, dy)  # Shape: (M, N, 2)

    # Step 3: Calculate the norm of the gradient
    norm_grad_phi = norm(grad_phi)  # Shape: (M, N) 

    # Step 4: Calculate the surface tension force
    tension_force = (3 * np.sqrt(2) * epsilon / (4 * We)) * curvature_value[..., np.newaxis] * norm_grad_phi[..., np.newaxis] * grad_phi  # Shape: (M, N, 2)

    return tension_force  # Shape: (M, N, 2)

def curvature(phi, dx, dy):
    """Calculate the curvature of the phase field phi.
    
    The curvature is defined as:
    K = \nabla \cdot \left( \frac{\nabla \phi}{|\nabla \phi|} \right)
    """
    # Step 1: Calculate the gradient of phi
    grad_phi = np.zeros((*phi.shape, 2))  # Gradient of phi, shape: (M, N, 2)
    grad_phi[:, :, 0] = numerical_derivative(phi, axis=0, h=dx)  # Gradient in x-direction, shape: (M, N)
    grad_phi[:, :, 1] = numerical_derivative(phi, axis=1, h=dy)  # Gradient in y-direction, shape: (M, N)

    # Step 2: Calculate the norm of the gradient
    norm_grad_phi = np.sqrt(grad_phi[:, :, 0]**2 + grad_phi[:, :, 1]**2)  # Norm of the gradient, shape: (M, N)

    # Step 3: Avoid division by zero
    norm_grad_phi[norm_grad_phi == 0] = 1e-10

    # Step 4: Calculate the normalized gradient
    normalized_grad_phi = grad_phi / norm_grad_phi[..., np.newaxis]  # Shape: (M, N, 2)

    # Step 5: Calculate the divergence of the normalized gradient
    curvature_value = numerical_derivative(normalized_grad_phi[..., 0], axis=0, h=dx) +  \
                      numerical_derivative(normalized_grad_phi[..., 1], axis=1, h=dy)

    return curvature_value  # Shape: (M, N)

def improved_curvature(phi, dx, dy):
    """Calculate curvature with regularization to reduce numerical artifacts."""
    grad_phi = gradient(phi, dx, dy)
    grad_phi_magnitude = np.sqrt(grad_phi[..., 0]**2 + grad_phi[..., 1]**2)
    
    # Add regularization to avoid division by zero
    reg = 1e-6
    grad_phi_magnitude = np.maximum(grad_phi_magnitude, reg)
    
    # Normalize gradient
    n_x = grad_phi[..., 0] / grad_phi_magnitude
    n_y = grad_phi[..., 1] / grad_phi_magnitude
    
    # Calculate divergence with higher-order scheme
    div_n = numerical_derivative(n_x, axis=0, h=dx) + numerical_derivative(n_y, axis=1, h=dy)
    
    # Apply smoothing to curvature field
    div_n = gaussian_filter(div_n, sigma=1.0)
    
    return div_n

def solve_poisson(rhs):
    """Solve the Poisson equation ∇²φ = f with a custom RHS.
    
    Args:
        rhs (np.ndarray): Right-hand side of the Poisson equation (shape: (Nx, Ny)).
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
    
    Returns:
        np.ndarray: Solution to the Poisson equation (shape: (Nx, Ny)).
    """
    Nx, Ny = rhs.shape[0], rhs.shape[1]
    phi = np.zeros((Nx, Ny))  # Initialize the solution array

    # Create the coefficient matrix for the finite difference method
    diagonals = [
        -2 * np.ones(Nx * Ny),  # Main diagonal
        np.ones(Nx * Ny - 1),   # Upper diagonal
        np.ones(Nx * Ny - 1),   # Lower diagonal
    ]

    # Adjust for the boundary conditions (assuming Dirichlet conditions)
    for i in range(1, Ny):
        diagonals[1][i * Nx - 1] = 0  # No connection between rows

    # Create the sparse matrix
    A = diags(diagonals, [0, 1, -1]).tocsc()

    # Reshape the RHS to a 1D array
    rhs_flat = rhs.flatten()

    # Solve the linear system
    # phi_flat = spsolve(A, rhs_flat, use_umfpack=True)
    phi_flat = cg(A, rhs_flat, tol=1e-10, maxiter=1000)[0]
    # Reshape the solution back to 2D
    phi = phi_flat.reshape((Nx, Ny))

    return phi

def solve_poisson_with_better_bc(rhs, dx, dy):
    """Improved Poisson solver with properly enforced boundary conditions"""
    Nx, Ny = rhs.shape[0], rhs.shape[1]
    
    # Create A matrix with proper coefficients based on grid spacing
    diagonals = [
        -2 * (1/dx**2 + 1/dy**2) * np.ones(Nx * Ny),  # Main diagonal
        (1/dx**2) * np.ones(Nx * Ny - 1),            # Upper diagonal
        (1/dx**2) * np.ones(Nx * Ny - 1),            # Lower diagonal
        (1/dy**2) * np.ones(Nx * (Ny-1)),           # Far upper diagonal
        (1/dy**2) * np.ones(Nx * (Ny-1))            # Far lower diagonal
    ]
    
    offsets = [0, 1, -1, Nx, -Nx]
    A = diags(diagonals, offsets).tocsc()
    
    # Use a direct solver with more precision
    phi_flat = spsolve(A, rhs.flatten(), use_umfpack=True)
    
    return phi_flat.reshape((Nx, Ny))

def f_1(phi):
    """Double-well potential function with minimas at phi = 0 and phi = 1.
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
    
    Returns:
        np.ndarray: Value of the double-well potential (shape: (Nx, Ny)).
    """
    return phi**2 * (1 - phi)**2  # Shape: (Nx, Ny)

def f_2(phi):
    """Double-well potential function with minimas at phi = -1 and phi = 1.
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
    
    Returns:
        np.ndarray: Value of the double-well potential (shape: (Nx, Ny)).
    """
    return 1./4 * (phi**2 - 1)**2

def df_1(phi):
    """Derivative of the double-well potential function.
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
    
    Returns:
        np.ndarray: Derivative of the double-well potential (shape: (Nx, Ny)).
    """
    return 2 * phi * (1 - phi) * (1 - 2 * phi)  # Shape: (Nx, Ny)

def df_2(phi):
    """Derivative of the double-well potential function.
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
    
    Returns:
        np.ndarray: Derivative of the double-well potential (shape: (Nx, Ny)).
    """
    return phi * (phi**2 - 1)  # Shape: (Nx, Ny)

def calculate_reynolds_number(phi, Re1, Re2):
    """Calculate the Reynolds number based on the phase field.
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
        Re1 (float): Reynolds number for phase 1.
        Re2 (float): Reynolds number for phase 2.
    
    Returns:
        np.ndarray: Calculated Reynolds number (shape: (Nx, Ny)).
    """
    # Calculate the Reynolds number using the provided formula
    phi_mapped = (phi + 1) / 2.0
    Re = 1 / ((1 + phi_mapped) / (2 * Re1) + (1 - phi_mapped) / (2 * Re2))  # Shape: (Nx, Ny)

    return Re  # Return the calculated Reynolds number 

def calculate_density(phi, rho1, rho2):
    """Calculate the density based on the phase field.
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
        rho1 (float): Density for phase 1.
        rho2 (float): Density for phase 2.
    
    Returns:
        np.ndarray: Calculated density (shape: (Nx, Ny)).
    """
        # Calculate the Reynolds number using the provided formula
    phi_mapped = (phi + 1) / 2.0
    rho = 1 / ((1 + phi_mapped) / (2 * rho1) + (1 - phi_mapped) / (2 * rho2))  # Shape: (Nx, Ny)

    return rho  # Return the calculated density 

def apply_contact_angle_boundary_conditions(phi, dx, dy, contact_angle=90):
    """Apply contact angle boundary conditions to the phase field.
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        contact_angle (float): Contact angle in degrees (default: 90).
    
    Returns:
        np.ndarray: Phase field with contact angle boundary conditions applied.
    """
    # Create a copy of the phase field to avoid modifying the original
    phi_new = phi.copy()
    
    # Convert contact angle to radians
    theta = (180 - contact_angle) * np.pi / 180
    
    # Calculate the gradient of phi at the bottom boundary
    grad_phi_x = numerical_derivative(phi, axis=0, h=dx)[:, 1]  # x-component of gradient at y=1
    grad_phi_y = numerical_derivative(phi, axis=1, h=dy)[:, 1]  # y-component of gradient at y=1
    
    # Calculate the norm of the gradient
    norm_grad_phi = np.sqrt(grad_phi_x**2 + grad_phi_y**2)
    
    # Avoid division by zero
    norm_grad_phi[norm_grad_phi < 1e-10] = 1e-10
    
    # Calculate the normal derivative based on the contact angle
    normal_derivative = -np.cos(theta) * norm_grad_phi
    
    # Apply the boundary condition at the bottom (y=0)
    phi_new[:, 0] = phi_new[:, 1] - normal_derivative * dy
    
    # Apply Neumann boundary conditions (zero gradient) at other boundaries
    phi_new[:, -1] = phi_new[:, -2]  # Top boundary
    phi_new[0, :] = phi_new[1, :]    # Left boundary
    phi_new[-1, :] = phi_new[-2, :]  # Right boundary
    
    return phi_new 

def apply_pressure_boundary_conditions(P):
    """Apply boundary conditions to the pressure field.
    
    Args:
        P (np.ndarray): Pressure field (shape: (Nx, Ny)).
    
    Returns:
        np.ndarray: Pressure field with boundary conditions applied.
    """
    # Create a copy of the pressure field to avoid modifying the original
    P_new = P.copy()
    
    # Bottom boundary (wall): Zero gradient (Neumann condition)
    P_new[:, 0] = P_new[:, 1]
    
    # Top boundary (open): Fixed value (Dirichlet condition)
    P_new[:, -1] = 0.0
    
    # Left and right boundaries: Zero gradient (Neumann condition)
    P_new[0, :] = P_new[1, :]
    P_new[-1, :] = P_new[-2, :]
    
    return P_new 

def plot_tension_force_vector(tension_force, save_path=None, title=None):
    """Plot the surface tension force as vectors."""
    
    # Calculate the magnitude of the force
    force_magnitude = np.sqrt(tension_force[:, :, 0]**2 + tension_force[:, :, 1]**2)
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Plot the magnitude as a heatmap
    im = plt.imshow(force_magnitude.T, origin='lower', cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Force Magnitude')
    
    # Create correct meshgrid
    X, Y = np.meshgrid(np.arange(tension_force.shape[0]), 
                       np.arange(tension_force.shape[1]), 
                       indexing='ij')
    
    skip = 5  # Skip every 5 points for clearer visualization
    plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
               tension_force[::skip, ::skip, 0], tension_force[::skip, ::skip, 1],
               color='white', scale=100)
    
    # Add title
    if title:
        plt.title(title)
    else:
        plt.title('Surface Tension Force Vector Field')
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def apply_surface_tension_boundary_conditions(surface_tension, phi, contact_angle=60):
    """Apply boundary conditions to the surface tension force.
    
    Args:
        surface_tension (np.ndarray): Surface tension force (shape: (Nx, Ny, 2)).
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
        contact_angle (float): Contact angle in degrees.
    
    Returns:
        np.ndarray: Surface tension force with boundary conditions applied.
    """
    # Make a copy to avoid modifying the original
    sf = surface_tension.copy()
    
    # Bottom boundary (wall): Adjust normal component based on contact angle
    # and preserve tangential component
    theta = (180 - contact_angle) * np.pi / 180
    sf[:, 0, 1] = sf[:, 1, 1] * np.cos(theta)  # Normal component (y)
    sf[:, 0, 0] = sf[:, 1, 0]                  # Tangential component (x)
    
    # Top boundary (open): Zero gradient
    sf[:, -1, :] = sf[:, -2, :]
    
    # Left and right boundaries: Zero gradient
    sf[0, :, :] = sf[1, :, :]
    sf[-1, :, :] = sf[-2, :, :]
    
    return sf

def create_joint_plot(phi, U, P, surface_tension, dt, step, dx, dy, mass, save_path=None):
    """Create a joint plot with multiple subplots."""
    # Calculate derived fields
    U_magnitude = np.sqrt(U[..., 0]**2 + U[..., 1]**2)
    P_grad = np.sqrt(numerical_derivative(P, axis=0, h=dx)**2 + 
                     numerical_derivative(P, axis=1, h=dy)**2)
    ST_magnitude = np.sqrt(surface_tension[..., 0]**2 + surface_tension[..., 1]**2)
    
    # Setup the figure with more space at bottom for text
    fig = plt.figure(figsize=(18, 14))  # Increased height
    
    # Create grid for subplots with extra bottom space
    gs = GridSpec(2, 3, figure=fig, bottom=0.20)  # Leave 20% space at bottom
    
    # Create meshgrid for plotting
    x = np.linspace(0, 1, phi.shape[0])
    y = np.linspace(0, 1, phi.shape[1])
    X, Y = np.meshgrid(x, y)
    
    # Plot 1: Phase Field
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(phi.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    ax1.contour(X, Y, phi.T, levels=[0], colors='r', linewidths=2)
    fig.colorbar(im1, ax=ax1, label='Phase Value')
    ax1.set_title('Phase Field')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    
    # Plot 2: Surface Tension Force
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(ST_magnitude.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    ax2.contour(X, Y, phi.T, levels=[0], colors='r', linewidths=2)
    fig.colorbar(im2, ax=ax2, label='Force Magnitude')
    ax2.set_title('Surface Tension Force')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    
    # Plot 3: Velocity Streamlines
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(U_magnitude.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    ax3.contour(X, Y, phi.T, levels=[0], colors='r', linewidths=2)
    # Add streamlines
    ax3.streamplot(x, y, U[..., 0].T, U[..., 1].T, density=1.5, color='white')
    fig.colorbar(im3, ax=ax3, label='Speed')
    ax3.set_title('Velocity Field Streamlines')
    ax3.set_xlabel('X-axis')
    ax3.set_ylabel('Y-axis')
    
    # Plot 4: Velocity Magnitude
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(U_magnitude.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    ax4.contour(X, Y, phi.T, levels=[0], colors='r', linewidths=2)
    fig.colorbar(im4, ax=ax4, label='Speed')
    ax4.set_title('Velocity Magnitude')
    ax4.set_xlabel('X-axis')
    ax4.set_ylabel('Y-axis')
    
    # Plot 5: Pressure Field
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(P.T, origin='lower', extent=[0, 1, 0, 1], cmap='coolwarm')
    ax5.contour(X, Y, phi.T, levels=[0], colors='k', linewidths=2)
    fig.colorbar(im5, ax=ax5, label='Pressure')
    ax5.set_title('Pressure Field')
    ax5.set_xlabel('X-axis')
    ax5.set_ylabel('Y-axis')
    
    # Plot 6: Pressure Gradient Magnitude
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(P_grad.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    ax6.contour(X, Y, phi.T, levels=[0], colors='r', linewidths=2)
    fig.colorbar(im6, ax=ax6, label='Gradient Magnitude')
    ax6.set_title('Pressure Gradient Magnitude')
    ax6.set_xlabel('X-axis')
    ax6.set_ylabel('Y-axis')
    
    # Add simulation information
    plt.figtext(0.5, 0.12, f'Time Step: {step}\nSimulation Time: {step * dt:.5f}', ha='center')
    
    # Add phase field statistics - moved higher to avoid overlap
    stats_text = f"Phase Field (phi):\n  Min: {phi.min():.5f}\n  Max: {phi.max():.5f}\n  Mean: {phi.mean():.5f}\n  Mass: {phi.sum():.5f}\n"
    stats_text += f"Velocity (U):\n  Min x: {U[..., 0].min():.5f}\n  Max x: {U[..., 0].max():.5f}\n  Min y: {U[..., 1].min():.5f}\n  Max y: {U[..., 1].max():.5f}\n  Max Speed: {U_magnitude.max():.5f}\n"
    stats_text += f"Pressure (P):\n  Min: {P.min():.5f}\n  Max: {P.max():.5f}\n  Mean: {P.mean():.5f}\n"
    stats_text += f"Surface Tension:\n  Max Magnitude: {ST_magnitude.max():.5f}\n"
    stats_text += f"Droplet mass: {mass:.5f}"
    
    plt.figtext(0.01, 0.5, stats_text, fontsize=10, verticalalignment='bottom')
    
    fig.tight_layout(rect=[0, 0.20, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def apply_phi_boundary_conditions(phi, dx, dy, contact_angle=60):
    """Apply proper boundary conditions to the phase field."""
    # Create a copy
    phi_new = phi.copy()
    
    # 1. Bottom boundary (solid wall): Contact angle condition
    theta = contact_angle * np.pi / 180
    
    # Get phi values at first interior node
    phi_interior = phi_new[:, 1]
    
    # Apply contact angle condition: normal derivative = -cos(theta)
    phi_new[:, 0] = phi_interior - dy * (-np.cos(theta))
    
    # 2. Top boundary (open atmosphere): Zero gradient
    phi_new[:, -1] = phi_new[:, -2]
    
    # 3. Left and right boundaries: Zero gradient
    phi_new[0, :] = phi_new[1, :]
    phi_new[-1, :] = phi_new[-2, :]
    
    return phi_new

def save_checkpoint(step, phi, U, P, directory="checkpoints"):
    """Save current simulation state to checkpoint files."""
    os.makedirs(directory, exist_ok=True)
    
    # Create checkpoint filename with step number
    checkpoint_path = os.path.join(directory, f"checkpoint_{step:06d}")
    
    # Save arrays
    np.savez_compressed(
        checkpoint_path, 
        step=step,
        phi=phi, 
        U=U, 
        P=P,
        timestamp=datetime.now().isoformat()
    )
    
    print(f"Checkpoint saved: {checkpoint_path}.npz")
    return checkpoint_path + ".npz"

def load_checkpoint(checkpoint_path):
    """Load simulation state from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load data
    data = np.load(checkpoint_path, allow_pickle=True)
    
    # Return as dictionary
    return {
        'step': int(data['step']),
        'phi': data['phi'],
        'U': data['U'],
        'P': data['P'],
        'timestamp': data['timestamp']
    }

def list_checkpoints(directory="checkpoints"):
    """List available checkpoint files."""
    if not os.path.exists(directory):
        return []
    
    checkpoints = []
    for filename in os.listdir(directory):
        if filename.startswith("checkpoint_") and filename.endswith(".npz"):
            checkpoints.append(os.path.join(directory, filename))
    
    return sorted(checkpoints)

