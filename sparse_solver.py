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
    def __init__(self, Nx, Ny, dx, dy, backend="scipy"):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.backend = backend
        self.bottom_boundary_condition = 'dirichlet' # 'dirichlet' or 'neumann'
        self.top_boundary_condition = 'dirichlet' # 'dirichlet' or 'neumann'
        self.left_boundary_condition = 'dirichlet' # 'dirichlet' or 'neumann'
        self.right_boundary_condition = 'dirichlet' # 'dirichlet' or 'neumann'
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
        # Override Dirichlet boundary conditions in the matrix
        # This ensures proper handling of boundary values by setting
        # diagonal elements to 1 and off-diagonal elements to 0
        
        # Find indices corresponding to boundary points
        # bottom_indices = np.arange(self.Nx) if self.bottom_boundary_condition == 'dirichlet' else []
        # top_indices = np.arange(self.Nx * (self.Ny - 1), self.Nx * self.Ny) if self.top_boundary_condition == 'dirichlet' else []
        
        # left_indices = np.arange(0, self.Nx * self.Ny, self.Nx) if self.left_boundary_condition == 'dirichlet' else []
        # right_indices = np.arange(self.Nx - 1, self.Nx * self.Ny, self.Nx) if self.right_boundary_condition == 'dirichlet' else []
        
        # # Combine all boundary indices
        # boundary_indices = np.concatenate([bottom_indices, top_indices, left_indices, right_indices])
        # boundary_indices = np.unique(boundary_indices)
        # boundary_indices = jnp.array(boundary_indices)

        # # Zero out the entire row
        # self.A = self.A.at[boundary_indices.astype(int), :].set(0.0)
        # # Set diagonal element to 1
        # self.A = self.A.at[boundary_indices.astype(int), boundary_indices.astype(int)].set(1.0)
        
        self.A = scipy.sparse.csr_matrix(np.array(self.A))
        if self.backend == "pyamg":
            self.solver = pyamg.ruge_stuben_solver(self.A, 
                                                   max_coarse=16,
                                                   max_levels=48)
    
    def set_rhs(self, rhs):
        self.rhs = rhs.transpose()

    def create_diag(self, N, step):
        main_diag = -2 * jnp.ones(N) / (step**2)
        off_diag = jnp.ones(N - 1) / (step**2)
        return jnp.diag(off_diag, k=-1) + jnp.diag(main_diag) + jnp.diag(off_diag, k=1)
    
    
    def solve(self, x0=None):
        if self.backend == "scipy":
            self.solution = self.solve_func(self.A, self.rhs.flatten()).reshape(self.rhs.shape)
        elif self.backend == "pyamg":
            residuals = []
            arg_dict = {'accel': 'bicgstab', 'tol': 0.1, 'residuals': residuals}
            if x0 is not None:
                arg_dict['x0'] = x0.flatten()
            self.solution = self.solver.solve(self.rhs.flatten(), 
                                              **arg_dict).reshape(self.rhs.shape)
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
    