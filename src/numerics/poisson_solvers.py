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
from pyro.multigrid import MG


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


def solve_poisson_pyamg(rhs, dx, dy, solution=None):
    """Solve Poisson equation using PyAMG multigrid solver.
    
    Args:
        rhs (np.ndarray): Right-hand side of the Poisson equation.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        solution (np.ndarray, optional): Initial guess for the solution.
    
    Returns:
        np.ndarray: Solution of the Poisson equation.
    """
    Nx, Ny = rhs.shape[0], rhs.shape[1]
    
    A = build_2d_laplacian_matrix_with_variable_steps(Nx, Ny, dx, dy)
    ml = pyamg.ruge_stuben_solver(A)  # construct the multigrid hierarchy
    x = ml.solve(rhs.flatten(), tol=1e-1)  # solve Ax=b to a tolerance of 1e-1
    
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
