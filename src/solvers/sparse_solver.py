"""
Sparse linear system solver wrapper.

This module contains the SparseSolverWrapper class for solving
sparse linear systems arising in the simulation.
"""

import numpy as np
import scipy.sparse
import pyamg
from scipy.sparse import diags
from scipy.sparse import identity
from scipy.sparse import kron
from functools import partial
import jax.numpy as jnp
import sys


class SparseSolverWrapper:
    """Wrapper class for sparse linear system solvers."""
    
    def __init__(self, Nx, Ny, dx, dy, backend="scipy"):
        """Initialize the sparse solver wrapper.
        
        Args:
            Nx (int): Number of grid points in x-direction.
            Ny (int): Number of grid points in y-direction.
            dx (float): Grid spacing in x-direction.
            dy (float): Grid spacing in y-direction.
            backend (str): Solver backend ("scipy" or "pyamg").
        """
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.backend = backend
        self.bottom_boundary_condition = 'dirichlet'
        self.top_boundary_condition = 'dirichlet'
        self.left_boundary_condition = 'dirichlet'
        self.right_boundary_condition = 'dirichlet'
        self.create_sparse_matrix()
        
        if backend == "scipy":
            self.solve_func = partial(scipy.sparse.linalg.spsolve, use_umfpack=True)
        elif backend == "pyamg":
            self.solver = pyamg.ruge_stuben_solver(self.A)
        else:
            raise ValueError(f"Invalid backend: {backend}")

    def set_bottom_boundary_condition(self, bottom_boundary_condition):
        """Set bottom boundary condition."""
        self.bottom_boundary_condition = bottom_boundary_condition

    def set_top_boundary_condition(self, top_boundary_condition):
        """Set top boundary condition."""
        self.top_boundary_condition = top_boundary_condition

    def set_left_boundary_condition(self, left_boundary_condition):
        """Set left boundary condition."""
        self.left_boundary_condition = left_boundary_condition

    def set_right_boundary_condition(self, right_boundary_condition):
        """Set right boundary condition."""
        self.right_boundary_condition = right_boundary_condition

    def create_sparse_matrix(self):
        """Create the sparse matrix for the linear system."""
        Tx = self.create_diag(self.Nx, self.dx)
        Ty = self.create_diag(self.Ny, self.dy)

        if self.bottom_boundary_condition == 'dirichlet':
            Ty = Ty.at[0, :].set(0.0)
            Ty = Ty.at[0, 0].set(1.0)
        elif self.bottom_boundary_condition == 'neumann':
            Ty = Ty.at[0, 1].set(2 / (self.dy**2))

        if self.top_boundary_condition == 'dirichlet':
            Ty = Ty.at[-1, :].set(0.0)
            Ty = Ty.at[-1, -1].set(1.0)
        elif self.top_boundary_condition == 'neumann':
            Ty = Ty.at[-1, -2].set(2 / (self.dy**2))

        if self.left_boundary_condition == 'dirichlet':
            Tx = Tx.at[0, :].set(0.0)
            Tx = Tx.at[0, 0].set(1.0)
        elif self.left_boundary_condition == 'neumann':
            Tx = Tx.at[0, 1].set(2 / (self.dx**2))

        if self.right_boundary_condition == 'dirichlet':
            Tx = Tx.at[-1, :].set(0.0)
            Tx = Tx.at[-1, -1].set(1.0)
        elif self.right_boundary_condition == 'neumann':
            Tx = Tx.at[-1, -2].set(2 / (self.dx**2))
        
        Ix = jnp.identity(self.Nx)
        Iy = jnp.identity(self.Ny)

        self.A = jnp.kron(Iy, Tx) + jnp.kron(Ty, Ix)
        
        self.A = scipy.sparse.csr_matrix(np.array(self.A).astype(np.float64))
        if self.backend == "pyamg":
            self.solver = pyamg.ruge_stuben_solver(self.A, 
                                                   max_coarse=16,
                                                   max_levels=48)
    
    def set_rhs(self, rhs):
        """Set the right-hand side of the linear system."""
        self.rhs = rhs.transpose().astype(np.float64)

    def create_diag(self, N, step):
        """Create diagonal matrix for 1D Laplacian."""
        main_diag = -2 * jnp.ones(N) / (step**2)
        off_diag = jnp.ones(N - 1) / (step**2)
        return jnp.diag(off_diag, k=-1) + jnp.diag(main_diag) + jnp.diag(off_diag, k=1)
    
    def solve(self, x0=None):
        """Solve the linear system."""
        if self.backend == "scipy":
            self.solution = self.solve_func(self.A, self.rhs.flatten()).reshape(self.rhs.shape)
        elif self.backend == "pyamg":
            residuals = []
            arg_dict = {'accel': 'bicgstab', 'tol': 0.1, 'residuals': residuals}
            if x0 is not None:
                arg_dict['x0'] = x0.flatten().astype(np.float64)
            self.solution = self.solver.solve(self.rhs.flatten().astype(np.float64), 
                                              **arg_dict).reshape(self.rhs.shape)
        self.solution = self.solution.transpose()

    def get_solution(self):
        """Get the solution of the linear system."""
        return self.solution
