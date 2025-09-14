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
    def __init__(self, Nx, Ny, dx, dy, backend="scipy", solver_args=None):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.backend = backend
        self.bottom_boundary_condition = 'dirichlet' # 'dirichlet' or 'neumann'
        self.top_boundary_condition = 'dirichlet' # 'dirichlet' or 'neumann'
        self.left_boundary_condition = 'dirichlet' # 'dirichlet' or 'neumann'
        self.right_boundary_condition = 'dirichlet' # 'dirichlet' or 'neumann'
        self.solver_args = solver_args
        self.create_sparse_matrix()
        
        if backend == "scipy":
            self.solve_func = partial(scipy.sparse.linalg.spsolve, use_umfpack=True)
        elif backend == "pyamg":
            self.solver = pyamg.ruge_stuben_solver(self.A)
        else:
            raise ValueError(f"Invalid backend: {backend}")

    def set_bottom_boundary_condition(self, bottom_boundary_condition):
        self.bottom_boundary_condition = bottom_boundary_condition

    def set_top_boundary_condition(self, top_boundary_condition):
        self.top_boundary_condition = top_boundary_condition

    def set_left_boundary_condition(self, left_boundary_condition):
        self.left_boundary_condition = left_boundary_condition

    def set_right_boundary_condition(self, right_boundary_condition):
        self.right_boundary_condition = right_boundary_condition

    def create_sparse_matrix(self):
        """Create 2D Laplacian matrix using hybrid approach:
        - Kronecker product for interior points
        - Manual setting for boundary conditions
        """
        N = self.Nx * self.Ny
        A = np.zeros((N, N))
        
        # Convert 2D indices (i,j) to 1D index: idx = i * Ny + j
        def idx(i, j):
            return i * self.Ny + j
        
        # Step 1: Build interior matrix using Kronecker product
        # Create 1D Laplacian matrices for interior points
        Tx_interior = self.create_diag(self.Nx, self.dx)
        Ty_interior = self.create_diag(self.Ny, self.dy)
        
        # Create identity matrices
        Ix = jnp.identity(self.Nx)
        Iy = jnp.identity(self.Ny)
        
        # Build full matrix using Kronecker product
        A_kronecker = jnp.kron(Iy, Tx_interior) + jnp.kron(Ty_interior, Ix)
        
        # Convert to numpy and copy to our matrix
        # For Poisson equation -∇²u = f, we need to negate the Laplacian
        A[:] = -np.array(A_kronecker)
        
        # Step 2: Manually set boundary conditions (only for Dirichlet)
        self._set_boundary_conditions(A)
        
        # Step 3: Check if we have pure Neumann BCs and apply constraint
        pure_neumann = (self.top_boundary_condition == 'neumann' and 
                       self.bottom_boundary_condition == 'neumann' and
                       self.left_boundary_condition == 'neumann' and
                       self.right_boundary_condition == 'neumann')
        
        if pure_neumann:
            # For pure Neumann BCs, add a constraint to make the system solvable
            # Use a single point constraint: u[center] = 0
            # This is the standard approach for pure Neumann problems
            center_i = self.Nx // 2
            center_j = self.Ny // 2
            constraint_idx = idx(center_i, center_j)
            A[constraint_idx, :] = 0.0
            A[constraint_idx, constraint_idx] = 1.0
        
        self.A = scipy.sparse.csr_matrix(A)
        if self.backend == "pyamg":
            self.solver = pyamg.ruge_stuben_solver(self.A, 
                                                   max_coarse=16,
                                                   max_levels=48)
    
    def _set_boundary_conditions(self, A):
        """Manually set boundary conditions with full control."""
        def idx(i, j):
            return i * self.Ny + j
        
        # Bottom boundary (i=0, all j)
        for j in range(self.Ny):
            center = idx(0, j)
            if self.bottom_boundary_condition == 'dirichlet':
                A[center, :] = 0.0
                A[center, center] = 1.0
            elif self.bottom_boundary_condition == 'neumann':
                A[center, :] = 0.0
                A[center, center] = 1.0
                A[center, idx(1, j)] = -1.0  # ∂u/∂y = 0: (u[1,j] - u[0,j])/dy = 0
        
        # Top boundary (i=Nx-1, all j)
        for j in range(self.Ny):
            center = idx(self.Nx-1, j)
            if self.top_boundary_condition == 'dirichlet':
                A[center, :] = 0.0
                A[center, center] = 1.0
            elif self.top_boundary_condition == 'neumann':
                A[center, :] = 0.0
                A[center, center] = 1.0
                A[center, idx(self.Nx-2, j)] = -1.0  # ∂u/∂y = 0: (u[Nx-1,j] - u[Nx-2,j])/dy = 0
        
        # Left boundary (all i, j=0)
        for i in range(self.Nx):
            center = idx(i, 0)
            if self.left_boundary_condition == 'dirichlet':
                A[center, :] = 0.0
                A[center, center] = 1.0
            elif self.left_boundary_condition == 'neumann':
                A[center, :] = 0.0
                A[center, center] = 1.0
                A[center, idx(i, 1)] = -1.0  # ∂u/∂x = 0: (u[i,1] - u[i,0])/dx = 0
        
        # Right boundary (all i, j=Ny-1)
        for i in range(self.Nx):
            center = idx(i, self.Ny-1)
            if self.right_boundary_condition == 'dirichlet':
                A[center, :] = 0.0
                A[center, center] = 1.0
            elif self.right_boundary_condition == 'neumann':
                A[center, :] = 0.0
                A[center, center] = 1.0
                A[center, idx(i, self.Ny-2)] = -1.0  # ∂u/∂x = 0: (u[i,Ny-1] - u[i,Ny-2])/dx = 0
    
    def set_rhs(self, rhs):
        # For -∇²u = f, we solve ∇²u = -f
        self.rhs = -rhs.transpose()
        print(f"DEBUG sparse_solver: RHS after transpose and negate min/max: {self.rhs.min():.6f} / {self.rhs.max():.6f}")
        print(f"DEBUG sparse_solver: RHS shape: {self.rhs.shape}")
        print(f"DEBUG sparse_solver: Matrix condition number estimate: {np.linalg.cond(self.A.todense()[:10, :10])}")

    def create_diag(self, N, step):
        """Create 1D Laplacian matrix for interior points only."""
        main_diag = 2 * jnp.ones(N) / (step**2)
        off_diag = -jnp.ones(N - 1) / (step**2)
        return jnp.diag(off_diag, k=-1) + jnp.diag(main_diag) + jnp.diag(off_diag, k=1)
    
    
    def solve(self, x0=None):
        # Check if we have pure Neumann BCs and adjust RHS accordingly
        pure_neumann = (self.top_boundary_condition == 'neumann' and 
                       self.bottom_boundary_condition == 'neumann' and
                       self.left_boundary_condition == 'neumann' and
                       self.right_boundary_condition == 'neumann')
        
        if pure_neumann:
            # For pure Neumann BCs, set the constraint RHS to 0
            center_i = self.Nx // 2
            center_j = self.Ny // 2
            center_idx = center_i * self.Ny + center_j
            rhs_flat = self.rhs.flatten()
            rhs_flat = rhs_flat.at[center_idx].set(0.0)  # u[center] = 0
            
            if self.backend == "scipy":
                self.solution = self.solve_func(self.A, rhs_flat).reshape(self.rhs.shape)
            elif self.backend == "pyamg":
                residuals = []
                arg_dict = {'accel': 'bicgstab', 'tol': 0.1, 'residuals': residuals}
                if self.solver_args is not None:
                    arg_dict.update(self.solver_args)
                if x0 is not None:
                    arg_dict['x0'] = x0.flatten()
                self.solution = self.solver.solve(rhs_flat, **arg_dict).reshape(self.rhs.shape)
        else:
            # Regular solve for non-pure Neumann BCs
            print(f"DEBUG sparse_solver: Solving with {self.backend} backend")
            if self.backend == "scipy":
                self.solution = self.solve_func(self.A, self.rhs.flatten()).reshape(self.rhs.shape)
            elif self.backend == "pyamg":
                residuals = []
                arg_dict = {'accel': 'bicgstab', 'tol': 0.1, 'residuals': residuals}
                if self.solver_args is not None:
                    arg_dict.update(self.solver_args)
                if x0 is not None:
                    arg_dict['x0'] = x0.flatten()
                self.solution = self.solver.solve(self.rhs.flatten(), 
                                                  **arg_dict).reshape(self.rhs.shape)
                print(f"DEBUG sparse_solver: PyAMG residuals: {residuals[-5:] if residuals else 'No residuals'}")
        
        print(f"DEBUG sparse_solver: Solution min/max: {self.solution.min():.6f} / {self.solution.max():.6f}")
        print(f"DEBUG sparse_solver: Solution contains NaN: {np.any(np.isnan(self.solution))}")
        print(f"DEBUG sparse_solver: Solution contains Inf: {np.any(np.isinf(self.solution))}")
        
        self.solution = self.solution.transpose()

    def get_solution(self):
        return self.solution
    
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    
    def test_poisson_solver():
        """Test the SparseSolver with different boundary conditions."""
        print("Testing SparseSolver with different boundary conditions...")
        
        # Grid parameters
        Nx, Ny = 5, 5
        Lx, Ly = 0.5, 0.5
        dx, dy = Lx/Nx, Ly/Ny
        
        # Create test cases with different boundary conditions
        test_cases = [
            {
                "name": "All Dirichlet",
                "boundary": {
                    "top": "dirichlet",
                    "bottom": "dirichlet",
                    "left": "dirichlet",
                    "right": "dirichlet"
                }
            },
            {
                "name": "All Neumann",
                "boundary": {
                    "top": "neumann",
                    "bottom": "neumann",
                    "left": "neumann",
                    "right": "neumann"
                }
            },
            {
                "name": "Mixed Boundary",
                "boundary": {
                    "top": "dirichlet",
                    "bottom": "neumann",
                    "left": "dirichlet",
                    "right": "neumann"
                }
            }
        ]
        
        for case in test_cases:
            print(f"\nTesting {case['name']} boundary conditions")
            
            # Create solver
            solver = SparseSolverWrapper(
                Nx, Ny, dx, dy,
                backend="pyamg"
            )
            solver.set_top_boundary_condition(case["boundary"]["top"])
            solver.set_bottom_boundary_condition(case["boundary"]["bottom"])
            solver.set_left_boundary_condition(case["boundary"]["left"])
            solver.set_right_boundary_condition(case["boundary"]["right"])
            
            # Create a simple RHS (e.g., a point source in the middle)
            rhs = np.zeros((Nx, Ny), dtype=jnp.float32)
            rhs[Nx//2, Ny//2] = 1.0
            
            # Set boundary values based on boundary conditions
            if case["boundary"]["top"] == "dirichlet":
                rhs[-1, :] = 0.0
            if case["boundary"]["bottom"] == "dirichlet":
                rhs[0, :] = 0.0
            if case["boundary"]["left"] == "dirichlet":
                rhs[:, 0] = 0.0
            if case["boundary"]["right"] == "dirichlet":
                rhs[:, -1] = 0.0
            
            # Solve
            print(solver.A.todense()[:5, :5])
            solver.set_rhs(rhs)
            solver.create_sparse_matrix()
            solver.solve()
            solution = solver.get_solution()

            
            # Print some statistics
            print(f"  Solution min: {solution.min():.6f}, max: {solution.max():.6f}")
            
            # Plot the solution
            plt.figure(figsize=(10, 8))
            plt.imshow(solution.T, origin='lower', cmap='viridis')
            plt.colorbar(label='Solution')
            plt.title(f"Poisson Solution - {case['name']} Boundary Conditions")
            plt.savefig(f"poisson_solution_{case['name'].replace(' ', '_').lower()}.png")
            plt.close()
    
    # Run the test
    test_poisson_solver()
    