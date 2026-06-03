"""
Poisson equation solvers for pressure and correction steps.

This module contains various implementations for solving Poisson equations
arising in incompressible flow simulations.
"""

import numpy as np
import jax.numpy as jnp
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import spsolve
import pyamg

try:
    from pyro.multigrid import MG
except ImportError:  # Optional dependency used only by solve_poisson_pyro().
    MG = None


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
    if bc_type == "neumann":
        # For Neumann BCs, use the corrected implementation
        return build_2d_laplacian_neumann_fixed(Nx, Ny, dx, dy)
    else:
        # For Dirichlet BCs, use interior points only
        if Nx > 2 and Ny > 2:
            return build_2d_laplacian_interior_fixed(Nx-2, Ny-2, dx, dy)
        else:
            # Fallback for very small grids
            return build_2d_laplacian_neumann_fixed(Nx, Ny, dx, dy)

def build_2d_laplacian_interior_fixed(Nx, Ny, dx, dy):
    """
    Create 2D Laplacian matrix for INTERIOR points only.
    
    For ∇²p = f, we solve: 
    (p[i-1,j] - 2*p[i,j] + p[i+1,j])/dx² + (p[i,j-1] - 2*p[i,j] + p[i,j+1])/dy² = f[i,j]
    
    Args:
        Nx (int): Number of interior points in x-direction
        Ny (int): Number of interior points in y-direction
        dx (float): Grid spacing in x-direction
        dy (float): Grid spacing in y-direction
    
    Returns:
        scipy.sparse matrix: 2D Laplacian matrix
    """
    # Create 1D Laplacian matrices
    main_diag_x = -2 * np.ones(Nx) / (dx**2)
    off_diag_x = np.ones(Nx - 1) / (dx**2)
    Tx = diags([off_diag_x, main_diag_x, off_diag_x], [-1, 0, 1], shape=(Nx, Nx))
    
    main_diag_y = -2 * np.ones(Ny) / (dy**2)
    off_diag_y = np.ones(Ny - 1) / (dy**2)
    Ty = diags([off_diag_y, main_diag_y, off_diag_y], [-1, 0, 1], shape=(Ny, Ny))
    
    # Create 2D matrix using Kronecker product
    Ix = identity(Nx)
    Iy = identity(Ny)
    A = kron(Iy, Tx) + kron(Ty, Ix)
    
    return A.tocsr()

def build_2d_laplacian_neumann_fixed(Nx, Ny, dx, dy):
    """
    Create 2D Laplacian matrix for Neumann boundary conditions.
    
    For Neumann BCs: ∂p/∂n = 0 on all boundaries
    This requires special handling because the system is singular.
    We fix one point to make it well-posed.
    
    Args:
        Nx (int): Number of points in x-direction (including boundaries)
        Ny (int): Number of points in y-direction (including boundaries)
        dx (float): Grid spacing in x-direction
        dy (float): Grid spacing in y-direction
    
    Returns:
        scipy.sparse matrix: 2D Laplacian matrix with Neumann BCs
    """
    # Create 1D Laplacian matrices
    main_diag_x = -2 * np.ones(Nx) / (dx**2)
    off_diag_x = np.ones(Nx - 1) / (dx**2)
    Tx = diags([off_diag_x, main_diag_x, off_diag_x], [-1, 0, 1], shape=(Nx, Nx))
    
    main_diag_y = -2 * np.ones(Ny) / (dy**2)
    off_diag_y = np.ones(Ny - 1) / (dy**2)
    Ty = diags([off_diag_y, main_diag_y, off_diag_y], [-1, 0, 1], shape=(Ny, Ny))
    
    # Apply Neumann boundary conditions
    Tx = Tx.tolil()
    Ty = Ty.tolil()
    
    # Neumann BCs: ∂p/∂x = 0 at x=0 and x=Lx
    Tx[0, 0] = -2 / (dx**2)  # Left boundary: p[0] = p[1]
    Tx[0, 1] = 2 / (dx**2)
    Tx[-1, -1] = -2 / (dx**2)  # Right boundary: p[N-1] = p[N-2]
    Tx[-1, -2] = 2 / (dx**2)
    
    # Neumann BCs: ∂p/∂y = 0 at y=0 and y=Ly
    Ty[0, 0] = -2 / (dy**2)  # Bottom boundary: p[0] = p[1]
    Ty[0, 1] = 2 / (dy**2)
    Ty[-1, -1] = -2 / (dy**2)  # Top boundary: p[N-1] = p[N-2]
    Ty[-1, -2] = 2 / (dy**2)
    
    Tx = Tx.tocsr()
    Ty = Ty.tocsr()
    
    # Create 2D matrix
    Ix = identity(Nx)
    Iy = identity(Ny)
    A = kron(Iy, Tx) + kron(Ty, Ix)
    
    # Fix one point to make the system well-posed
    center_idx = (Nx // 2) * Ny + (Ny // 2)
    A = A.tolil()
    A[center_idx, :] = 0.0
    A[center_idx, center_idx] = 1.0
    A = A.tocsr()
    
    return A


def jax_build_2d_laplacian_matrix_with_variable_steps(Nx, Ny, dx, dy):
    """
    JAX version of 2D Laplacian matrix construction.
    
    Parameters:
        Nx (int): Number of interior grid points in x-direction
        Ny (int): Number of interior grid points in y-direction
        dx (float): Grid spacing in x-direction
        dy (float): Grid spacing in y-direction
    
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
    Ty[0, :] = 0.0
    Ty[0, 0] = 1.0     # Dirichlet row at bottom
    Ty[-1, :] = 0.0
    Ty[-1, -1] = 1.0   # Dirichlet row at top
    Ty = Ty.tocsr()
    
    # Create identity matrices
    Ix = identity(Nx)
    Iy = identity(Ny)
    
    # Combine using Kronecker products to create 2D Laplacian
    A = kron(Iy, Tx) + kron(Ty, Ix)
    
    return A


def solve_poisson(rhs, Nx, Ny, dx, dy):
    """Solve the Poisson equation ∇²φ = f with Neumann boundary conditions.
    
    Args:
        rhs (np.ndarray): Right-hand side of the Poisson equation.
        Nx (int): Number of grid points in x-direction.
        Ny (int): Number of grid points in y-direction.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        np.ndarray: Solution of the Poisson equation.
    """
    A = build_2d_laplacian_matrix_with_variable_steps(Nx, Ny, dx, dy)
    
    # Reshape right-hand side to match matrix equation
    rhs_flat = np.transpose(rhs).flatten(order='C')  # Use C-style ordering (row-major)
    
    # Solve the linear system
    phi_flat = spsolve(A, rhs_flat)
    
    # Reshape solution to 2D - use proper ordering
    phi = phi_flat.reshape((Nx, Ny), order='C')
    phi = np.transpose(phi)
    
    return phi


def solve_poisson_with_better_bc(rhs, dx, dy):
    """Improved Poisson solver with properly enforced boundary conditions.
    
    Args:
        rhs (np.ndarray): Right-hand side of the Poisson equation.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        np.ndarray: Solution of the Poisson equation.
    """
    Nx, Ny = rhs.shape[0], rhs.shape[1]
    
    # Create A matrix with proper coefficients based on grid spacing
    main_diag_x = -2 * np.ones(Nx) / (dx**2)
    off_diag_x = np.ones(Nx - 1) / (dx**2)
    Tx = diags([off_diag_x, main_diag_x, off_diag_x], [-1, 0, 1], shape=(Nx, Nx))
    
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


def solve_poisson_pyamg(rhs, dx, dy, solution=None, ppe_bcs=None):
    """Solve Poisson equation using PyAMG multigrid solver.
    
    This function now supports mixed boundary conditions through the ppe_bcs parameter.
    
    Args:
        rhs (np.ndarray): Right-hand side of the Poisson equation.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        solution (np.ndarray, optional): Initial guess for the solution.
        ppe_bcs (dict, optional): PPE boundary conditions from config.
                                Format: {'top': 'dirichlet'/'neumann', 'bottom': 'dirichlet'/'neumann',
                                        'left': 'dirichlet'/'neumann', 'right': 'dirichlet'/'neumann'}
    
    Returns:
        np.ndarray: Solution of the Poisson equation.
    """
    Nx, Ny = rhs.shape[0], rhs.shape[1]
    
    # If ppe_bcs is provided, use the new mixed BC solver
    if ppe_bcs is not None:
        return solve_poisson_mixed_bcs(rhs, dx, dy, ppe_bcs, solution)
    
    # Fallback to original behavior for backward compatibility
    # Determine boundary condition type based on config
    if ppe_bcs and all(bc == "neumann" for bc in ppe_bcs.values()):
        bc_type = "neumann"
        # For Neumann BCs, use the corrected implementation
        A = build_2d_laplacian_neumann_fixed(Nx, Ny, dx, dy)
        # For Neumann BCs, we need to ensure the RHS has zero mean
        rhs_mean_sub = rhs - np.mean(rhs)
        ml = pyamg.ruge_stuben_solver(A)
        x = ml.solve(rhs_mean_sub.flatten(), tol=1e-6)
    else:
        bc_type = "dirichlet"  # Default fallback
        # For Dirichlet BCs, use interior points only
        if Nx > 2 and Ny > 2:
            # Extract interior points
            rhs_interior = rhs[1:-1, 1:-1]
            A = build_2d_laplacian_interior_fixed(Nx-2, Ny-2, dx, dy)
            ml = pyamg.ruge_stuben_solver(A)
            x = ml.solve(rhs_interior.flatten(), tol=1e-6)
            # Reshape and pad with zeros for boundaries
            x = x.reshape((Nx-2, Ny-2))
            x_padded = np.zeros((Nx, Ny))
            x_padded[1:-1, 1:-1] = x
            x = x_padded.flatten()
        else:
            # Fallback for very small grids
            A = build_2d_laplacian_neumann_fixed(Nx, Ny, dx, dy)
            rhs_mean_sub = rhs - np.mean(rhs)
            ml = pyamg.ruge_stuben_solver(A)
            x = ml.solve(rhs_mean_sub.flatten(), tol=1e-6)
    
    return x.reshape((Nx, Ny))


def solve_poisson_pyro(rhs, dx, dy, solution=None):
    """Solve Poisson equation using Pyro multigrid solver.
    
    Args:
        rhs (np.ndarray): Right-hand side of the Poisson equation.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        solution (np.ndarray, optional): Initial guess for the solution.
    
    Returns:
        np.ndarray: Solution of the Poisson equation.
    """
    if MG is None:
        raise ImportError(
            "solve_poisson_pyro requires the optional 'pyro' package with "
            "'pyro.multigrid.MG', but it is not installed."
        )

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

    # get the solution
    v = a.get_solution()
    return v[1:-1, 1:-1]


# =============================================================================
# MIXED BOUNDARY CONDITIONS IMPLEMENTATION
# =============================================================================

def build_2d_laplacian_mixed_bcs(Nx, Ny, dx, dy, bcs):
    """
    Build 2D Laplacian matrix with mixed boundary conditions.
    
    This implementation supports arbitrary BCs on each side:
    - top: 'dirichlet' or 'neumann'
    - bottom: 'dirichlet' or 'neumann'  
    - left: 'dirichlet' or 'neumann'
    - right: 'dirichlet' or 'neumann'
    
    Args:
        Nx (int): Number of grid points in x-direction
        Ny (int): Number of grid points in y-direction
        dx (float): Grid spacing in x-direction
        dy (float): Grid spacing in y-direction
        bcs (dict): Boundary conditions for each side
    
    Returns:
        scipy.sparse matrix: 2D Laplacian matrix with mixed BCs
    """
    # Validate input
    valid_bcs = {'dirichlet', 'neumann'}
    for side, bc in bcs.items():
        if bc not in valid_bcs:
            raise ValueError(f"Invalid boundary condition '{bc}' for {side}. Must be 'dirichlet' or 'neumann'")
    
    # Create 1D Laplacian matrices
    main_diag_x = -2 * np.ones(Nx) / (dx**2)
    off_diag_x = np.ones(Nx - 1) / (dx**2)
    Tx = diags([off_diag_x, main_diag_x, off_diag_x], [-1, 0, 1], shape=(Nx, Nx))
    
    main_diag_y = -2 * np.ones(Ny) / (dy**2)
    off_diag_y = np.ones(Ny - 1) / (dy**2)
    Ty = diags([off_diag_y, main_diag_y, off_diag_y], [-1, 0, 1], shape=(Ny, Ny))
    
    # Apply boundary conditions to x-direction
    Tx = apply_bc_to_1d_matrix_mixed(Tx, bcs['left'], bcs['right'], dx, 'x')
    
    # Apply boundary conditions to y-direction
    Ty = apply_bc_to_1d_matrix_mixed(Ty, bcs['bottom'], bcs['top'], dy, 'y')
    
    # Create 2D matrix using Kronecker product
    Ix = identity(Nx)
    Iy = identity(Ny)
    A = kron(Iy, Tx) + kron(Ty, Ix)
    
    # Handle singularity if needed
    if is_singular_case_mixed(bcs):
        A = fix_singularity_mixed(A, Nx, Ny)
    
    return A.tocsr()

def apply_bc_to_1d_matrix_mixed(matrix, bc_start, bc_end, step, direction):
    """
    Apply boundary conditions to a 1D Laplacian matrix for mixed BCs.
    
    Args:
        matrix: 1D Laplacian matrix
        bc_start: Boundary condition at start (left/bottom)
        bc_end: Boundary condition at end (right/top)
        step: Grid spacing
        direction: 'x' or 'y' for error messages
    
    Returns:
        Modified matrix with boundary conditions applied
    """
    matrix = matrix.tolil()
    
    # Apply boundary condition at start (left/bottom)
    if bc_start == 'dirichlet':
        # Dirichlet BC: p = 0
        matrix[0, :] = 0.0
        matrix[0, 0] = 1.0
    elif bc_start == 'neumann':
        # Neumann BC: ∂p/∂n = 0 using ghost points
        # For Neumann: p[ghost] = p[interior]
        # The stencil (p[i-1] - 2*p[i] + p[i+1])/dx² becomes (2*p[0] - 2*p[1])/dx²
        matrix[0, 0] = -2 / (step**2)
        matrix[0, 1] = 2 / (step**2)
    
    # Apply boundary condition at end (right/top)
    if bc_end == 'dirichlet':
        # Dirichlet BC: p = 0
        matrix[-1, :] = 0.0
        matrix[-1, -1] = 1.0
    elif bc_end == 'neumann':
        # Neumann BC: ∂p/∂n = 0 using ghost points
        # For Neumann: p[ghost] = p[interior]
        # The stencil (p[i-1] - 2*p[i] + p[i+1])/dx² becomes (2*p[N-1] - 2*p[N-2])/dx²
        matrix[-1, -1] = -2 / (step**2)
        matrix[-1, -2] = 2 / (step**2)
    
    return matrix.tocsr()

def needs_compatibility_mixed(bcs):
    """Check if compatibility condition is needed for mixed BCs."""
    return any(bc == 'neumann' for bc in bcs.values())

def is_singular_case_mixed(bcs):
    """Check if the matrix will be singular for mixed BCs."""
    # Matrix is singular if all sides have Neumann BCs
    return all(bc == 'neumann' for bc in bcs.values())

def fix_singularity_mixed(matrix, Nx, Ny):
    """
    Fix matrix singularity by constraining one point for mixed BCs.
    
    Args:
        matrix: Laplacian matrix
        Nx (int): Number of grid points in x-direction
        Ny (int): Number of grid points in y-direction
    
    Returns:
        Modified matrix with singularity fixed
    """
    matrix = matrix.tolil()
    
    # Fix center point to zero
    center_idx = (Nx // 2) * Ny + (Ny // 2)
    matrix[center_idx, :] = 0.0
    matrix[center_idx, center_idx] = 1.0
    
    return matrix.tocsr()

def solve_poisson_mixed_bcs(rhs, dx, dy, bcs, solution=None):
    """
    Solve Poisson equation with mixed boundary conditions.
    
    Args:
        rhs (np.ndarray): Right-hand side of the Poisson equation
        dx (float): Grid spacing in x-direction
        dy (float): Grid spacing in y-direction
        bcs (dict): Boundary conditions for each side
        solution (np.ndarray, optional): Initial guess for the solution
    
    Returns:
        np.ndarray: Solution of the Poisson equation
    """
    Nx, Ny = rhs.shape[0], rhs.shape[1]
    
    # Build matrix with mixed BCs
    A = build_2d_laplacian_mixed_bcs(Nx, Ny, dx, dy, bcs)
    
    # Prepare RHS
    rhs_modified = rhs.copy()
    
    # Apply compatibility condition if needed
    if needs_compatibility_mixed(bcs):
        rhs_modified = rhs_modified - np.mean(rhs_modified)
    
    # Solve using PyAMG
    try:
        ml = pyamg.ruge_stuben_solver(A)
        x = ml.solve(rhs_modified.flatten(), tol=1e-6)
    except:
        # Fallback to direct solver if PyAMG fails
        x = spsolve(A, rhs_modified.flatten())
    
    return x.reshape((Nx, Ny))
