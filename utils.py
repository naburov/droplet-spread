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
from scipy.sparse import kron, identity
from pyro.multigrid import MG
import pyamg

def numerical_derivative(f, axis=0, h=1e-5, dtype='float32'):
    """Calculate the numerical derivative of f along a specified axis using central difference."""
    return ((np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * h)).astype(dtype)

# numerical_derivative = jit(numerical_derivative)

def laplacian(f, dx=1e-5, dy=1e-5):
    """Calculate the Laplacian of f at all points in a 2D field using finite differences."""
    
    # Interior points using finite difference
    f_padded = np.pad(f.copy(), ((1, 1), (1, 1)), mode="edge")
    d2x = (f_padded[1:-1, 2:] - 2*f_padded[1:-1, 1:-1] + f_padded[1:-1, :-2]) / dx**2
    d2y = (f_padded[2:, 1:-1] - 2*f_padded[1:-1, 1:-1] + f_padded[:-2, 1:-1]) / dy**2

    return d2x + d2y

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

def surface_tension_force(phi, epsilon, We1, We2, dx, dy):
    """Calculate the surface tension force based on the phase field.
    
    The surface tension force is given by:
    
    F_{\text{tension}} = \frac{3\sqrt{2}\epsilon}{4We} \nabla \cdot \left( \frac{\nabla \phi}{|\nabla \phi|} \right) |\nabla \phi| \nabla \phi
    """
    # Step 1: Calculate the curvature using the curvature function
    curvature_value = improved_curvature(phi, dx, dy)  # Shape: (M, N)
    curvature_value = np.stack([curvature_value, curvature_value], axis=-1)

    # Step 2: Calculate the gradient of phi
    grad_phi = gradient(phi, dx, dy)  # Shape: (M, N, 2)

    # Step 3: Calculate the norm of the gradient
    norm_grad_phi = norm(grad_phi)  # Shape: (M, N) 
    norm_grad_phi = np.stack([norm_grad_phi, norm_grad_phi], axis=-1)
    We = calculate_weber_number(phi, We1, We2)
    We = np.stack([We, We], axis=-1)

    # Step 4: Calculate the surface tension force
    tension_force = (3 * np.sqrt(2) * epsilon / (4 * We)) * curvature_value * norm_grad_phi * grad_phi  # Shape: (M, N, 2)

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
    # div_n = gaussian_filter(div_n, sigma=1.0)
    
    return div_n

def build_2d_laplacian_matrix_with_variable_steps(Nx, Ny, dx, dy, bc_type='dirichlet'):
    """
    Constructs the 2D Laplacian matrix using Kronecker product with variable spatial steps
    in x and y directions, supporting different boundary conditions.
    
    Parameters:
        Nx (int): Number of interior grid points in x-direction
        Ny (int): Number of interior grid points in y-direction
        dx (float): Grid spacing in x-direction
        dy (float): Grid spacing in y-direction
        bc_type (str): 'dirichlet' or 'neumann'
    
    Returns:
        scipy.sparse.csr_matrix: Sparse Laplacian matrix of shape (Nx*Ny, Nx*Ny)
    """
    # 1D Laplacian for x-direction
    main_diag_x = -2 * np.ones(Nx) / (dx**2)
    off_diag_x = np.ones(Nx - 1) / (dx**2)
    Tx = diags([off_diag_x, main_diag_x, off_diag_x], [-1, 0, 1], shape=(Nx, Nx))
    
    # 1D Laplacian for y-direction
    main_diag_y = -2 * np.ones(Ny) / (dy**2)
    off_diag_y = np.ones(Ny - 1) / (dy**2)
    Ty = diags([off_diag_y, main_diag_y, off_diag_y], [-1, 0, 1], shape=(Ny, Ny))
    
    # Apply boundary conditions
    Tx = Tx.tolil()
    Tx[0, 1] = 2 / (dx**2)  # Left boundary mirror
    Tx[-1, -2] = 2 / (dx**2)  # Right boundary mirror
    Tx = Tx.tocsr()
        
    Ty = Ty.tolil()
    Ty[0, 0] = 1.0
    Ty[0, 1] = 0.0  
    
    Ty[-1, -1] = 1.0
    Ty[-1, -2] = 0.0
    Ty = Ty.tocsr()
    
    # Create identity matrices
    Ix = identity(Nx)
    Iy = identity(Ny)
    
    # Combine using Kronecker products to create 2D Laplacian
    A = kron(Iy, Tx) + kron(Ty, Ix)
    
    return A


def solve_poisson_with_better_bc(rhs, dx, dy):
    """Improved Poisson solver with properly enforced boundary conditions"""
    Nx, Ny = rhs.shape[0], rhs.shape[1]
    
    # Create A matrix with proper coefficients based on grid spacing
        # 1D Laplacian for x-direction
    main_diag_x = -2 * np.ones(Nx) / (dx**2)
    off_diag_x = np.ones(Nx - 1) / (dx**2)
    Tx = diags([off_diag_x, main_diag_x, off_diag_x], [-1, 0, 1], shape=(Nx, Nx))
    
    # 1D Laplacian for y-direction
    main_diag_y = -2 * np.ones(Ny) / (dy**2)
    off_diag_y = np.ones(Ny - 1) / (dy**2)
    Ty = diags([off_diag_y, main_diag_y, off_diag_y], [-1, 0, 1], shape=(Ny, Ny))

    # Apply boundary conditions
    Tx = Tx.tolil()
    Tx[0, 1] = 2 / (dx**2)  # Left boundary mirror
    Tx[-1, -2] = 2 / (dx**2)  # Right boundary mirror
    Tx = Tx.tocsr()
        
    Ty = Ty.tolil()
    Ty[0, 1] = 2 / (dy**2)
    Ty[-1, -2] = 2 / (dy**2)
    Ty = Ty.tocsr()
    
    # Create identity matrices
    Ix = identity(Nx)
    Iy = identity(Ny)
    
    # Combine using Kronecker products to create 2D Laplacian
    A = kron(Iy, Tx) + kron(Ty, Ix)
    
    # Use a direct solver with more precision
    phi_flat = spsolve(A, rhs.flatten(), use_umfpack=True)
    
    return phi_flat.reshape((Nx, Ny))

def create_correction_matrix(Nx, Ny, dx, dy):
    main_diag_x = -2 * np.ones(Nx) / (dx**2)
    off_diag_x = np.ones(Nx - 1) / (dx**2)
    Tx = diags([off_diag_x, main_diag_x, off_diag_x], [-1, 0, 1], shape=(Nx, Nx))
    
    # 1D Laplacian for y-direction
    main_diag_y = -2 * np.ones(Ny) / (dy**2)
    off_diag_y = np.ones(Ny - 1) / (dy**2)
    Ty = diags([off_diag_y, main_diag_y, off_diag_y], [-1, 0, 1], shape=(Ny, Ny))

    # Apply boundary conditions
    Tx = Tx.tolil()
    Tx[0, 1] = 2 / (dx**2)  # Left boundary mirror
    Tx[-1, -2] = 2 / (dx**2)  # Right boundary mirror
    Tx = Tx.tocsr()
        
    Ty = Ty.tolil()
    Ty[0, 1] = 2 / (dy**2)
    Ty[-1, -2] = 2 / (dy**2)
    Ty = Ty.tocsr()
    
    # Create identity matrices
    Ix = identity(Nx)
    Iy = identity(Ny)
    
    # Combine using Kronecker products to create 2D Laplacian
    A = kron(Iy, Tx) + kron(Ty, Ix)
    return A

def solve_poisson_pyamg(rhs, dx, dy, solution=None):
    Nx, Ny = rhs.shape[0], rhs.shape[1]
    
    A = create_correction_matrix(Nx, Ny, dx, dy)
    ml = pyamg.ruge_stuben_solver(A)                    # construct the multigrid hierarchy
    x = ml.solve(rhs.flatten(), tol=1e-1)                          # solve Ax=b to a tolerance of 1e-10
    
    return x.reshape((Nx, Ny))

def solve_poisson_pyro(rhs, dx, dy, solution=None):
    """Solve Poisson equation using Pyro."""

    nx = int(1.0 / dx)
    ny = nx

    # create the multigrid object
    a = MG.CellCenterMG2d(nx, ny,
                          xl_BC_type="neumann", yl_BC_type="neumann",
                          xr_BC_type="neumann", yr_BC_type="neumann",
                          verbose=False)

    # initialize the solution to 0

    if solution is not None:
        solution = np.pad(solution, ((1, 1), (1, 1)), mode="edge")
        solution[0, :] = solution[1, :]
        solution[-1, :] = solution[-2, :]
        solution[:, 0] = solution[:, 1]
        solution[:, -1] = solution[:, -2]
        a.init_solution(solution)
    else:
        a.init_zeros()

    rhs = np.pad(rhs, ((1, 1), (1, 1)), mode="edge")
    # initialize the RHS using the function f
    a.init_RHS(rhs)

    # solve to a relative tolerance of 1.e-11
    a.solve(rtol=1.e-1)

    # alternately, we can just use smoothing by uncommenting the following
    # a.smooth(a.nlevels-1,50000)

    # get the solution
    v = a.get_solution()
    return v[1:-1, 1:-1]

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
    Re = 1 / ((1 + phi_mapped) / (2 * Re2) + (1 - phi_mapped) / (2 * Re1))  # Shape: (Nx, Ny)

    return Re  # Return the calculated Reynolds number 

def calculate_weber_number(phi, We1, We2):
    """Calculate the Weber number based on the phase field.
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
        We1 (float): Weber number for phase 1.
        We2 (float): Weber number for phase 2.
    
    Returns:
        np.ndarray: Calculated Weber number (shape: (Nx, Ny)).
    """
    phi_mapped = (phi + 1) / 2.0
    We = 1 / ((1 + phi_mapped) / (2 * We2) + (1 - phi_mapped) / (2 * We1))  # Shape: (Nx, Ny)

    return We  # Return the calculated Weber number 

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
    rho = 1 / ((1 + phi_mapped) / (2 * rho2) + (1 - phi_mapped) / (2 * rho1))  # Shape: (Nx, Ny)

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

def apply_pressure_boundary_conditions(P, g, phi, rho1, rho2, dy, atm_pressure=0.0):
    """Apply boundary conditions to the pressure field.
    
    Args:
        P (np.ndarray): Pressure field (shape: (Nx, Ny)).
    
    Returns:
        np.ndarray: Pressure field with boundary conditions applied.
    """
    # Create a copy of the pressure field to avoid modifying the original
    P_new = P.copy()
    rho = calculate_density(phi, rho1, rho2)

    # Bottom boundary (wall): Zero gradient (Neumann condition)
    P_new[:, 0] = np.sum(rho * g * dy, axis=-1) + atm_pressure
    P_new[:, 0] *= np.where(phi[:, 0] < 0, 1, -1)
    
    # Top boundary (open): Fixed value (Dirichlet condition)
    P_new[:, -1] = atm_pressure
    
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



