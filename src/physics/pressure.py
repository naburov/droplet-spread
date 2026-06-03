"""
Pressure field calculations for droplet spreading simulation.

Pressure is split into two components:
- p_dynamic: from surface tension (capillary pressure)
- p_hydrostatic: from gravity (ρgh)

Total pressure: p = p_dynamic + p_hydrostatic
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import scipy.sparse
import scipy.sparse.linalg
from numerics.finite_differences import jax_divergence
from physics.properties import jax_calculate_density


@jit
def compute_hydrostatic_pressure(rho, g, dy, Fr, atm_pressure=0.0):
    """Compute hydrostatic pressure by integrating ρg from top.
    
    Hydrostatic equilibrium: dp/dy = ρ * g_y
    Integrating from top (p=atm) downward: p(y) = atm + ∫_{y}^{top} ρ|g|dy'
    
    Args:
        rho: Density field (Nx, Ny).
        g: Gravitational acceleration (negative for downward).
        dy: Grid spacing in y.
        Fr: Classical Froude number U/sqrt(gL).
        atm_pressure: Pressure at top boundary.
    
    Returns:
        Hydrostatic pressure field (Nx, Ny).
    """
    Nx, Ny = rho.shape
    
    # Initialize with atmospheric pressure at top
    p_hydro = jnp.zeros_like(rho)
    p_hydro = p_hydro.at[:, -1].set(atm_pressure)
    
    # Integrate from top to bottom: p[j] = p[j+1] - ρ[j] * g * dy
    # Note: g is negative (pointing down), so -g*dy is positive (pressure increases going down)
    # Classical Froude convention in runtime: gravity/hydrostatic scaling uses 1/Fr^2.
    fr2 = jnp.maximum(Fr * Fr, 1e-12)
    def integrate_step(carry, j):
        p_prev = carry
        # Use density at current level for integration
        p_curr = p_prev - rho[:, j] * g * dy / fr2
        return p_curr, p_curr
    
    # Scan from Ny-2 down to 0
    indices = jnp.arange(Ny - 2, -1, -1)
    _, p_columns = jax.lax.scan(integrate_step, p_hydro[:, -1], indices)
    
    # Reconstruct full pressure field
    # p_columns[k] corresponds to index Ny-2-k
    p_hydro = p_hydro.at[:, :-1].set(p_columns[::-1].T)
    
    return p_hydro


def _build_variable_coefficient_pressure_matrix(inv_rho_u_face, inv_rho_v_face, dx, dy, bcs):
    """Build Cartesian matrix for div((1/rho) grad p)=rhs with Neumann/Dirichlet BCs."""
    Nx = inv_rho_v_face.shape[0]
    Ny = inv_rho_u_face.shape[1]
    dx2 = dx * dx
    dy2 = dy * dy

    left_bc = bcs.get("left", "neumann")
    right_bc = bcs.get("right", "neumann")
    bottom_bc = bcs.get("bottom", "neumann")
    top_bc = bcs.get("top", "neumann")
    all_neumann = all(bc == "neumann" for bc in (left_bc, right_bc, bottom_bc, top_bc))

    A = scipy.sparse.lil_matrix((Nx * Ny, Nx * Ny), dtype=np.float64)

    def idx(i, j):
        return j * Nx + i

    for i in range(Nx):
        for j in range(Ny):
            on_left = i == 0
            on_right = i == Nx - 1
            on_bottom = j == 0
            on_top = j == Ny - 1
            is_dirichlet = (
                (on_left and left_bc == "dirichlet")
                or (on_right and right_bc == "dirichlet")
                or (on_bottom and bottom_bc == "dirichlet")
                or (on_top and top_bc == "dirichlet")
            )
            k = idx(i, j)
            if is_dirichlet:
                A[k, k] = 1.0
                continue

            beta_w = 0.0 if on_left else float(inv_rho_u_face[i, j])
            beta_e = 0.0 if on_right else float(inv_rho_u_face[i + 1, j])
            beta_s = 0.0 if on_bottom else float(inv_rho_v_face[i, j])
            beta_n = 0.0 if on_top else float(inv_rho_v_face[i, j + 1])

            diag = 0.0
            if i > 0:
                A[k, idx(i - 1, j)] = beta_w / dx2
                diag -= beta_w / dx2
            if i < Nx - 1:
                A[k, idx(i + 1, j)] = beta_e / dx2
                diag -= beta_e / dx2
            if j > 0:
                A[k, idx(i, j - 1)] = beta_s / dy2
                diag -= beta_s / dy2
            if j < Ny - 1:
                A[k, idx(i, j + 1)] = beta_n / dy2
                diag -= beta_n / dy2

            A[k, k] = diag

    gauge_ij = None
    if all_neumann:
        gauge_ij = (Nx // 2, Ny // 2)
        k0 = idx(gauge_ij[0], gauge_ij[1])
        A[k0, :] = 0.0
        A[k0, k0] = 1.0

    return A.tocsr(), all_neumann, gauge_ij


def _solve_variable_coefficient_pressure(A, rhs, x0=None, tol=1e-8, maxiter=2000):
    """Solve variable-coefficient pressure system robustly."""
    rhs_t = np.asarray(rhs.T, dtype=np.float64)
    rhs_flat = rhs_t.flatten()
    x0_flat = None if x0 is None else np.asarray(x0.T, dtype=np.float64).flatten()

    try:
        sol, info = scipy.sparse.linalg.bicgstab(
            A, rhs_flat, x0=x0_flat, rtol=tol, atol=0.0, maxiter=maxiter
        )
    except TypeError:
        sol, info = scipy.sparse.linalg.bicgstab(
            A, rhs_flat, x0=x0_flat, tol=tol, maxiter=maxiter
        )

    if info != 0 or not np.all(np.isfinite(sol)):
        try:
            sol = scipy.sparse.linalg.spsolve(A, rhs_flat)
        except Exception:
            sol = np.zeros_like(rhs_flat)

    if not np.all(np.isfinite(sol)):
        sol = np.zeros_like(rhs_flat)

    return sol.reshape(rhs_t.shape).T


def _smooth_3x3_edge(field):
    """3x3 box smoothing with edge padding (non-periodic)."""
    p = jnp.pad(field, ((1, 1), (1, 1)), mode="edge")
    return (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
        + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
        + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    ) / 9.0


def update_pressure_jax(surface_tension, dx, dy, geometry, rho1, rho2, phi, g, Fr, atm_pressure,
                        pressure_solver, include_gravity=False, has_dirichlet_bc=False,
                        dirichlet_rhs=None, capillary_rhs_smoothing_radius=1):
    """Update pressure. Terrain divergence of surface_tension. p_total = p_dynamic + p_hydrostatic.
    dirichlet_rhs: optional dict e.g. {"top": 0, "left": 0, "right": 0} so RHS at Dirichlet
    boundaries is set to the prescribed value (avoids spurious pressure gradients)."""
    sf_grad = jax_divergence(surface_tension, dx, dy, geometry.f_1_grid)
    # Optional smoothing of capillary RHS; now enabled by default (radius=1).
    # Set radius=0 in config to disable.
    smooth_r = int(capillary_rhs_smoothing_radius)
    if smooth_r > 0:
        for _ in range(smooth_r):
            sf_grad = _smooth_3x3_edge(sf_grad)
    # Set RHS at all Dirichlet boundaries to prescribed value so solution satisfies P = value there
    if dirichlet_rhs is not None:
        for side, value in dirichlet_rhs.items():
            if side == "top":
                sf_grad = sf_grad.at[:, -1].set(float(value))
            elif side == "bottom":
                sf_grad = sf_grad.at[:, 0].set(float(value))
            elif side == "left":
                sf_grad = sf_grad.at[0, :].set(float(value))
            elif side == "right":
                sf_grad = sf_grad.at[-1, :].set(float(value))
    else:
        sf_grad = sf_grad.at[:, -1].set(atm_pressure)
    sf_grad_np = np.array(sf_grad, dtype=np.float64)
    use_variable_density_pressure = phi is not None

    if use_variable_density_pressure:
        rho = np.array(jax_calculate_density(phi, rho1, rho2))
        rho = np.maximum(rho, 1e-6)
        inv_rho = 1.0 / rho
        Nx, Ny = inv_rho.shape

        inv_rho_u_face = np.zeros((Nx + 1, Ny), dtype=np.float64)
        inv_rho_v_face = np.zeros((Nx, Ny + 1), dtype=np.float64)
        inv_rho_u_face[1:Nx, :] = 0.5 * (inv_rho[1:, :] + inv_rho[:-1, :])
        inv_rho_u_face[0, :] = inv_rho[0, :]
        inv_rho_u_face[Nx, :] = inv_rho[-1, :]
        inv_rho_v_face[:, 1:Ny] = 0.5 * (inv_rho[:, 1:] + inv_rho[:, :-1])
        inv_rho_v_face[:, 0] = inv_rho[:, 0]
        inv_rho_v_face[:, Ny] = inv_rho[:, -1]

        p_bcs = getattr(pressure_solver, "bcs", None) or {
            "left": "neumann",
            "right": "neumann",
            "bottom": "neumann",
            "top": "neumann",
        }

        A, all_neumann, gauge_ij = _build_variable_coefficient_pressure_matrix(
            inv_rho_u_face, inv_rho_v_face, dx, dy, p_bcs
        )
        rhs_var = sf_grad_np.copy()
        if all_neumann and gauge_ij is not None:
            gi, gj = gauge_ij
            rhs_var[gi, gj] = 0.0

        P_dynamic = jnp.array(_solve_variable_coefficient_pressure(A, rhs_var))
    else:
        raise ValueError("Pressure update requires phi for staggered-only pressure path.")

    if not has_dirichlet_bc:
        P_top_mean = float(jnp.mean(P_dynamic[:, -1]))
        if abs(P_top_mean - atm_pressure) > 1e-6:
            P_dynamic = P_dynamic - P_top_mean + atm_pressure

    if include_gravity:
        rho = jax_calculate_density(phi, rho1, rho2)
        P_hydrostatic = compute_hydrostatic_pressure(rho, g, dy, Fr, atm_pressure)
        return P_dynamic + P_hydrostatic
    return P_dynamic


class PressureSolver:
    """Pressure solver for droplet spreading simulation."""
    
    def __init__(
        self,
        rho1,
        rho2,
        g,
        atm_pressure,
        Fr=1.0,
        include_gravity=False,
        capillary_rhs_smoothing_radius=1,
        pressure_bcs=None,
    ):
        self.rho1 = rho1
        self.rho2 = rho2
        self.g = g
        self.atm_pressure = atm_pressure
        self.Fr = Fr
        self.include_gravity = include_gravity
        self.capillary_rhs_smoothing_radius = int(capillary_rhs_smoothing_radius)
        self.pressure_bcs = pressure_bcs
    
    def update_pressure(self, surface_tension, dx, dy, geometry, phi, pressure_solver, has_dirichlet_bc=False,
                        dirichlet_rhs=None):
        """Update pressure. geometry: from state (terrain divergence of ST).
        dirichlet_rhs: optional dict {side: value} so RHS at Dirichlet boundaries matches BC (avoids spurious gradients)."""
        solver_for_pressure = pressure_solver
        effective_has_dirichlet = has_dirichlet_bc
        if self.pressure_bcs is not None:
            class _PressureBcs:
                pass

            solver_for_pressure = _PressureBcs()
            solver_for_pressure.bcs = dict(self.pressure_bcs)
            effective_has_dirichlet = any(v == "dirichlet" for v in solver_for_pressure.bcs.values())

        return update_pressure_jax(
            surface_tension, dx, dy, geometry, self.rho1, self.rho2, phi,
            self.g, self.Fr, self.atm_pressure, solver_for_pressure,
            include_gravity=self.include_gravity,
            has_dirichlet_bc=effective_has_dirichlet,
            dirichlet_rhs=dirichlet_rhs,
            capillary_rhs_smoothing_radius=self.capillary_rhs_smoothing_radius,
        )
