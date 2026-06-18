"""Sparse linear system solver for Poisson equation.

Supports Cartesian Laplacian ∇²φ = f and terrain Laplacian
L φ = φ_xx - 2 f' φ_xη - f'' φ_η + (1+f'²) φ_ηη = f (same stencil as jax_laplacian).
For flat geometry (f_1=0, f_2=0) the terrain stencil reduces to Cartesian.
"""

import numpy as np
import scipy.sparse
import pyamg
from functools import partial
class SparseSolverWrapper:
    """Sparse solver for 2D Poisson equation: ∇²φ = f or terrain Laplacian L φ = f."""
    
    def __init__(self, Nx, Ny, dx, dy, backend="scipy", solver_params=None, f_1_grid=None, f_2_grid=None):
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, dy
        self.backend = backend
        
        # Boundary conditions (default: Dirichlet everywhere)
        self.bcs = {"top": "dirichlet", "bottom": "dirichlet", "left": "dirichlet", "right": "dirichlet"}
        
        # Solver params
        params = solver_params or {}
        self.accel = params.get('accel', 'bicgstab')
        self.tol = params.get('tol', 0.1)
        self.maxiter = params.get('maxiter', 1000)
        self.terrain_laplacian_mode = params.get('terrain_laplacian_mode', 'legacy_stencil')
        
        self._rhs = None
        self._solution = None
        self._f_1 = np.array(f_1_grid, dtype=np.float64) if f_1_grid is not None else None
        self._f_2 = np.array(f_2_grid, dtype=np.float64) if f_2_grid is not None else None
        self._build_matrix()
    
    def set_bcs(self, **kwargs):
        """Set boundary conditions. Example: set_bcs(top='dirichlet', bottom='neumann')"""
        for key, val in kwargs.items():
            if key in self.bcs:
                self.bcs[key] = val
        self._build_matrix()
    
    # Legacy methods for compatibility with existing code
    def set_top_boundary_condition(self, bc): self.bcs["top"] = bc; self._build_matrix()
    def set_bottom_boundary_condition(self, bc): self.bcs["bottom"] = bc; self._build_matrix()
    def set_left_boundary_condition(self, bc): self.bcs["left"] = bc; self._build_matrix()
    def set_right_boundary_condition(self, bc): self.bcs["right"] = bc; self._build_matrix()
    
    def set_bcs_from_manager(self, bc_manager):
        """Set BCs from a BC manager object."""
        if hasattr(bc_manager, 'get_implicit_bc_types'):
            types = bc_manager.get_implicit_bc_types()
        elif hasattr(bc_manager, 'bc_raw'):
            types = {b: "dirichlet" if t in ['dirichlet', 'open'] else "neumann" 
                     for b, t in bc_manager.bc_raw.items()}
        else:
            raise ValueError("BC manager must have get_implicit_bc_types() or bc_raw")
        self.bcs.update(types)
        self._build_matrix()
    
    def set_terrain(self, f_1_grid, f_2_grid=None):
        """Set terrain coefficients for terrain Laplacian L φ = φ_xx - 2 f' φ_xη - f'' φ_η + (1+f'²) φ_ηη.
        f_1_grid, f_2_grid: (Nx, Ny). For flat (f=0) pass zeros; then stencil = Cartesian."""
        self._f_1 = np.asarray(f_1_grid, dtype=np.float64)
        if f_2_grid is not None:
            self._f_2 = np.asarray(f_2_grid, dtype=np.float64)
        else:
            self._f_2 = np.zeros_like(self._f_1)
        if self._f_1.shape != (self.Nx, self.Ny):
            self._f_1 = np.broadcast_to(np.reshape(self._f_1, (-1, 1)), (self.Nx, self.Ny))
        if self._f_2.shape != (self.Nx, self.Ny):
            self._f_2 = np.broadcast_to(np.reshape(self._f_2, (-1, 1)), (self.Nx, self.Ny))
        self._build_matrix()
    
    def _use_terrain(self):
        """True if terrain Laplacian should be used (f_1/f_2 set and non-zero)."""
        if self._f_1 is None:
            return False
        return np.any(np.abs(self._f_1) > 1e-14) or (self._f_2 is not None and np.any(np.abs(self._f_2) > 1e-14))
    
    def _build_matrix(self):
        """Build Laplacian matrix: Cartesian or terrain-aware depending on set_terrain."""
        if self._use_terrain():
            self._build_terrain_laplacian()
            return
        Tx = self._create_1d_laplacian(self.Nx, self.dx)
        Ty = self._create_1d_laplacian(self.Ny, self.dy)
        
        # Apply BCs to 1D matrices
        Ty = self._apply_bc_1d(Ty, "bottom", self.dy, is_start=True)
        Ty = self._apply_bc_1d(Ty, "top", self.dy, is_start=False)
        Tx = self._apply_bc_1d(Tx, "left", self.dx, is_start=True)
        Tx = self._apply_bc_1d(Tx, "right", self.dx, is_start=False)
        
        # Build 2D matrix via Kronecker products using SciPy only.
        Ix = scipy.sparse.identity(self.Nx, format="csr", dtype=np.float64)
        Iy = scipy.sparse.identity(self.Ny, format="csr", dtype=np.float64)
        A = scipy.sparse.kron(Iy, Tx, format="csr") + scipy.sparse.kron(Ty, Ix, format="csr")

        self.A = A.tocsr()
        self._apply_corner_dirichlet()
        # Setup solver
        self._setup_solver()
    
    def _build_terrain_laplacian(self):
        """Build terrain Laplacian L φ = φ_xx - 2 f' φ_xη - f'' φ_η + (1+f'²) φ_ηη.
        Same stencil as jax_laplacian. Flat (f_1=0, f_2=0) gives Cartesian 5-point."""
        if self.terrain_laplacian_mode == "jax_derivative_composition":
            self._build_terrain_laplacian_jax_composition()
            return
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy
        f1 = self._f_1  # (Nx, Ny)
        f2 = self._f_2 if self._f_2 is not None else np.zeros_like(f1)
        n = Nx * Ny
        row_ind = []
        col_ind = []
        data = []
        for j in range(Ny):
            for i in range(Nx):
                k = j * Nx + i
                if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
                    continue
                c_xx = 1.0 / (dx * dx)
                c_xy = 1.0 / (2.0 * dx * dy)
                c_yy = 1.0 / (dy * dy)
                f_ij = f1[i, j]
                f2_ij = f2[i, j]
                a_eta = (1.0 + f_ij * f_ij) * c_yy
                a_eta1 = f2_ij / (2.0 * dy)
                row_ind.extend([k] * 9)
                col_ind.extend([
                    (j - 1) * Nx + (i - 1), (j - 1) * Nx + i, (j - 1) * Nx + (i + 1),
                    j * Nx + (i - 1), j * Nx + i, j * Nx + (i + 1),
                    (j + 1) * Nx + (i - 1), (j + 1) * Nx + i, (j + 1) * Nx + (i + 1),
                ])
                # L φ = φ_xx - 2 f' φ_xη - f'' φ_η + (1 + f'^2) φ_ηη.
                # Row order is eta-south, center eta, eta-north; within each row x-west, x, x-east.
                data.extend([
                    -f_ij * c_xy, a_eta + a_eta1, f_ij * c_xy,
                    c_xx, -2.0 * c_xx - 2.0 * a_eta, c_xx,
                    f_ij * c_xy, a_eta - a_eta1, -f_ij * c_xy,
                ])
        A = scipy.sparse.csr_matrix(
            (data, (row_ind, col_ind)), shape=(n, n), dtype=np.float64
        )
        self._apply_bc_2d_terrain(A)
        self._apply_corner_dirichlet()
        self._setup_solver()

    def _build_terrain_laplacian_jax_composition(self):
        """Build the exact sparse analogue of numerics.finite_differences.jax_laplacian.

        This mode is used by the semi-implicit phase-field solver: the explicit
        JAX path computes φ_xx, φ_xη, φ_η and φ_ηη by composing first-derivative
        operators with one-sided boundary rows.  A sparse implicit operator with
        a different 3-point terrain stencil injects a residual at every step on
        grooved substrates, which shows up as column-wise chainsaw artifacts.
        """
        Nx, Ny = self.Nx, self.Ny
        f1 = self._f_1
        f2 = self._f_2 if self._f_2 is not None else np.zeros_like(f1)

        ix = scipy.sparse.identity(Nx, format="csr", dtype=np.float64)
        iy = scipy.sparse.identity(Ny, format="csr", dtype=np.float64)
        dx_cell = scipy.sparse.kron(
            iy, self._create_first_derivative_matrix(Nx, self.dx), format="csr"
        )
        dy_cell = scipy.sparse.kron(
            self._create_first_derivative_matrix(Ny, self.dy), ix, format="csr"
        )

        diag_f1 = scipy.sparse.diags(f1.T.reshape(-1), format="csr")
        diag_f2 = scipy.sparse.diags(f2.T.reshape(-1), format="csr")
        diag_metric = scipy.sparse.diags((1.0 + f1**2).T.reshape(-1), format="csr")

        self.A = (
            dx_cell @ dx_cell
            - 2.0 * diag_f1 @ (dx_cell @ dy_cell)
            - diag_f2 @ dy_cell
            + diag_metric @ (dy_cell @ dy_cell)
        ).tocsr()
        self._setup_solver()
    
    def _apply_bc_2d_terrain(self, A):
        """Apply boundary conditions to terrain Laplacian matrix (modify boundary rows)."""
        Nx, Ny = self.Nx, self.Ny
        n = Nx * Ny
        A_lil = A.tolil()
        dx, dy = self.dx, self.dy
        for j in range(Ny):
            for i in range(Nx):
                if i > 0 and i < Nx - 1 and j > 0 and j < Ny - 1:
                    continue
                k = j * Nx + i
                A_lil[k, :] = 0.0
                A_lil[k, k] = 1.0
                if self.bcs["bottom"] == "neumann" and j == 0:
                    if Ny > 1:
                        A_lil[k, k] = -2.0 / (dy * dy)
                        A_lil[k, (j + 1) * Nx + i] = 2.0 / (dy * dy)
                elif self.bcs["top"] == "neumann" and j == Ny - 1:
                    if Ny > 1:
                        A_lil[k, k] = -2.0 / (dy * dy)
                        A_lil[k, (j - 1) * Nx + i] = 2.0 / (dy * dy)
                elif self.bcs["left"] == "neumann" and i == 0:
                    if Nx > 1:
                        A_lil[k, k] = -2.0 / (dx * dx)
                        A_lil[k, j * Nx + (i + 1)] = 2.0 / (dx * dx)
                elif self.bcs["right"] == "neumann" and i == Nx - 1:
                    if Nx > 1:
                        A_lil[k, k] = -2.0 / (dx * dx)
                        A_lil[k, j * Nx + (i - 1)] = 2.0 / (dx * dx)
        self.A = A_lil.tocsr()

    def _apply_corner_dirichlet(self):
        """At corners, enforce Dirichlet if any of the two meeting sides is Dirichlet.
        Removes mixed Dirichlet+Neumann rows that cause spurious pressure gradients."""
        if not hasattr(self, 'A') or self.A is None:
            return
        Nx, Ny = self.Nx, self.Ny
        corners = [(0, 0), (Nx - 1, 0), (0, Ny - 1), (Nx - 1, Ny - 1)]
        A_lil = self.A.tolil()
        for i, j in corners:
            k = j * Nx + i
            use_dirichlet = (
                (i == 0 and self.bcs["left"] == "dirichlet")
                or (i == Nx - 1 and self.bcs["right"] == "dirichlet")
                or (j == 0 and self.bcs["bottom"] == "dirichlet")
                or (j == Ny - 1 and self.bcs["top"] == "dirichlet")
            )
            if use_dirichlet:
                A_lil[k, :] = 0.0
                A_lil[k, k] = 1.0
        self.A = A_lil.tocsr()

    def _setup_solver(self):
        """Setup solve function or AMG solver from current self.A."""
        if self.backend == "scipy":
            self.solve_func = partial(scipy.sparse.linalg.spsolve, use_umfpack=True)
        else:
            self.solver = pyamg.ruge_stuben_solver(self.A, max_coarse=16, max_levels=48)
    
    def _create_1d_laplacian(self, N, h):
        """Create 1D second derivative matrix."""
        main = np.full(N, -2.0 / h**2, dtype=np.float64)
        off = np.full(max(N - 1, 0), 1.0 / h**2, dtype=np.float64)
        return scipy.sparse.diags((off, main, off), offsets=(-1, 0, 1), shape=(N, N), format="lil")

    def _create_first_derivative_matrix(self, N, h):
        """Create the matrix analogue of jax_dx/jax_dy first derivatives."""
        rows, cols, data = [], [], []
        for i in range(N):
            if i == 0:
                rows.extend([i, i])
                cols.extend([0, 1])
                data.extend([-1.0 / h, 1.0 / h])
            elif i == N - 1:
                rows.extend([i, i])
                cols.extend([N - 2, N - 1])
                data.extend([-1.0 / h, 1.0 / h])
            else:
                rows.extend([i, i])
                cols.extend([i - 1, i + 1])
                data.extend([-0.5 / h, 0.5 / h])
        return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    
    def _apply_bc_1d(self, T, boundary, h, is_start):
        """Apply BC to 1D matrix row."""
        idx = 0 if is_start else -1
        neighbor = 1 if is_start else -2

        if self.bcs[boundary] == "dirichlet":
            T[idx, :] = 0.0
            T[idx, idx] = 1.0
        else:  # neumann
            T[idx, :] = 0.0
            T[idx, idx] = -2.0 / h**2
            T[idx, neighbor] = 2.0 / h**2
        return T
    
    def solve(self, rhs=None, x0=None):
        """Solve Ax = rhs. Input/output shape: (Nx, Ny)."""
        # Use stored RHS if not provided (legacy API)
        if rhs is None:
            rhs = self._rhs
        
        # Transpose RHS to match matrix layout
        rhs_t = np.asarray(rhs.T, dtype=np.float64)
        rhs_flat = rhs_t.flatten()
        
        if self.backend == "scipy":
            sol = self.solve_func(self.A, rhs_flat)
        else:
            args = {'accel': self.accel, 'tol': self.tol, 'residuals': []}
            if x0 is not None:
                args['x0'] = np.asarray(x0.T, dtype=np.float64).flatten()
            sol = self.solver.solve(rhs_flat, **args)

        # If iterative solver broke down (e.g. BiCGSTAB divide-by-zero), return zero
        # so callers never see NaN and can continue with "no correction"
        if not np.all(np.isfinite(sol)):
            sol = np.zeros_like(rhs_flat)

        self._solution = sol.reshape(rhs_t.shape).T
        return self._solution
    
    def clear_workspace(self):
        """Release references to RHS and solution arrays (and PyAMG residual lists if any).
        Call after each step in a simulation loop to reduce memory retention."""
        self._rhs = None
        self._solution = None

    # Legacy API for compatibility
    def set_rhs(self, rhs):
        """Set RHS for solve() (legacy API)."""
        self._rhs = np.asarray(rhs)
    
    def get_solution(self):
        """Get solution (legacy API)."""
        return self._solution
    
    def create_sparse_matrix(self):
        """Rebuild matrix (legacy API). Curvilinear: fluid-only, no solid mask."""
        self._build_matrix()
