"""
Phase field equations and boundary conditions.

This module implements the Cahn-Hilliard equation with optional Willmore regularization.
The Willmore term is derived from the Willmore energy functional via variational principles.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from numerics.finite_differences import jax_gradient, jax_laplacian, jax_norm, jax_dx, jax_dy
from physics.properties import jax_df_2
from physics.surface_tension import jax_curvature
from boundary_conditions.chemical_potential_bc import (
    BC_DIRICHLET,
    BC_NEUMANN,
    jax_apply_chemical_potential_bc,
)


@jit
def jax_willmore_chemical_potential(phi, dx, dy, f_1_grid, f_2_grid, epsilon_w):
    """Willmore term μ_W = ε_W Δ²φ. f_1_grid, f_2_grid: (Nx, Ny)."""
    lap_phi = jax_laplacian(phi, dx, dy, f_1_grid, f_2_grid)
    return epsilon_w * jax_laplacian(lap_phi, dx, dy, f_1_grid, f_2_grid)


@jit
def jax_laplacian_flat_bottom_ghost(f, dx, dy, bottom_ghost):
    """Cartesian Laplacian with a custom ghost row below the bottom boundary."""
    d2x = jax_dx(jax_dx(f, h=dx), h=dx)
    d2y = (jnp.roll(f, -1, axis=1) - 2.0 * f + jnp.roll(f, 1, axis=1)) / (dy**2)
    d2y = d2y.at[:, 0].set((f[:, 1] - 2.0 * f[:, 0] + bottom_ghost) / (dy**2))
    d2y = d2y.at[:, -1].set((f[:, -2] - f[:, -1]) / (dy**2))
    return d2x + d2y


@jit
def jax_laplacian_terrain_bottom_ghost(f, dx, dy, f_1_grid, f_2_grid, bottom_ghost):
    """Terrain-following Laplacian with a custom ghost row below eta=0.

    Coordinates are x_physical = x, y_physical = eta + h(x).  Therefore

        ∂X = ∂x - h' ∂η,  ∂Y = ∂η
        Δφ = φ_xx - 2 h' φ_xη - h'' φ_η + (1 + h'^2) φ_ηη.

    At the bottom, φ_η and φ_ηη are evaluated with the supplied ghost row.
    Flat terrain reduces exactly to the previous Cartesian ghost operator.
    """
    f_1 = f_1_grid
    f_2 = f_2_grid
    phi_xx = jax_dx(jax_dx(f, h=dx), h=dx)

    phi_eta = jax_dy(f, h=dy)
    phi_eta = phi_eta.at[:, 0].set((f[:, 1] - bottom_ghost) / (2.0 * dy))

    phi_x_eta = jax_dx(phi_eta, h=dx)

    phi_eta_eta = jax_dy(jax_dy(f, h=dy), h=dy)
    phi_eta_eta = phi_eta_eta.at[:, 0].set((f[:, 1] - 2.0 * f[:, 0] + bottom_ghost) / (dy**2))
    phi_eta_eta = phi_eta_eta.at[:, -1].set((f[:, -2] - f[:, -1]) / (dy**2))

    return phi_xx - 2.0 * f_1 * phi_x_eta - f_2 * phi_eta + (1.0 + f_1**2) * phi_eta_eta


@jit
def jax_phase_convective_term(phi, U, dx, dy, f_1_grid):
    grad_phi = jax_gradient(phi, dx, dy, f_1_grid)
    return U[..., 0] * grad_phi[..., 0] + U[..., 1] * grad_phi[..., 1]


@jit
def jax_degenerate_mobility(phi, mobility_clip=0.0, mobility_blend=0.0):
    """Quasi-degenerate CH mobility with optional constant-mobility blending.

    M(phi) = (1-b) * (1-phi^2) + b, optionally clipped from below by `mobility_clip`.
    `mobility_blend` in [0,1] controls how close we are to constant mobility.
    """
    phi_limited = jnp.clip(phi, -1.0, 1.0)
    base = jnp.clip(1.0 - phi_limited**2, 0.0, 1.0)
    blend = jnp.clip(mobility_blend, 0.0, 1.0)
    mobility = (1.0 - blend) * base + blend
    return jnp.maximum(mobility, mobility_clip)


@jit
def jax_mobility_diffusion(mu, mobility, dx, dy, f_1_grid):
    """Compute div(M grad(mu)) in conservative face-flux form.

    This discretization enforces zero normal mobility flux at the domain edges
    and preserves the global phi integral up to advection/BC terms.
    """
    del f_1_grid  # Current degenerate-mobility runs are flat; keep signature stable.

    # Face-centered mobility (arithmetic average).
    mob_x_face = 0.5 * (mobility[1:, :] + mobility[:-1, :])
    mob_y_face = 0.5 * (mobility[:, 1:] + mobility[:, :-1])

    # Interior face fluxes for +div(M grad(mu)).
    flux_x_face = mob_x_face * (mu[1:, :] - mu[:-1, :]) / dx
    flux_y_face = mob_y_face * (mu[:, 1:] - mu[:, :-1]) / dy

    # Cell-centered divergence with zero-flux boundaries.
    div_x = jnp.zeros_like(mu)
    div_x = div_x.at[1:-1, :].set((flux_x_face[1:, :] - flux_x_face[:-1, :]) / dx)
    div_x = div_x.at[0, :].set(flux_x_face[0, :] / dx)
    div_x = div_x.at[-1, :].set(-flux_x_face[-1, :] / dx)

    div_y = jnp.zeros_like(mu)
    div_y = div_y.at[:, 1:-1].set((flux_y_face[:, 1:] - flux_y_face[:, :-1]) / dy)
    div_y = div_y.at[:, 0].set(flux_y_face[:, 0] / dy)
    div_y = div_y.at[:, -1].set(-flux_y_face[:, -1] / dy)

    return div_x + div_y


@jit
def jax_terrain_conservative_diffusion(mu, mobility, dx, dy, f_1_grid):
    """Conservative terrain-following div(M grad(mu)) with zero normal flux.

    For y = eta + h(x), the physical Laplacian can be written in divergence
    form in computational coordinates because the Jacobian is one:

        div(grad mu) = d_x(mu_x - h' mu_eta)
                     + d_eta(-h' mu_x + (1 + h'^2) mu_eta).

    This face-flux discretization makes the telescoping sum exactly zero when
    boundary fluxes are zero, preventing phase leakage on tilted/grooved grids.
    """
    f_1 = f_1_grid
    mobility = jnp.asarray(mobility)

    # x-faces, shape (Nx-1, Ny)
    mu_x_face = (mu[1:, :] - mu[:-1, :]) / dx
    mu_eta_cell = jax_dy(mu, h=dy)
    mu_eta_xface = 0.5 * (mu_eta_cell[1:, :] + mu_eta_cell[:-1, :])
    f_1_xface = 0.5 * (f_1[1:, :] + f_1[:-1, :])
    mobility_xface = 0.5 * (mobility[1:, :] + mobility[:-1, :])
    flux_x = mobility_xface * (mu_x_face - f_1_xface * mu_eta_xface)

    # eta-faces, shape (Nx, Ny-1)
    mu_eta_face = (mu[:, 1:] - mu[:, :-1]) / dy
    mu_x_cell = jax_dx(mu, h=dx)
    mu_x_yface = 0.5 * (mu_x_cell[:, 1:] + mu_x_cell[:, :-1])
    f_1_yface = 0.5 * (f_1[:, 1:] + f_1[:, :-1])
    mobility_yface = 0.5 * (mobility[:, 1:] + mobility[:, :-1])
    flux_eta = mobility_yface * (-f_1_yface * mu_x_yface + (1.0 + f_1_yface**2) * mu_eta_face)

    div_x = jnp.zeros_like(mu)
    div_x = div_x.at[1:-1, :].set((flux_x[1:, :] - flux_x[:-1, :]) / dx)
    div_x = div_x.at[0, :].set(flux_x[0, :] / dx)
    div_x = div_x.at[-1, :].set(-flux_x[-1, :] / dx)

    div_eta = jnp.zeros_like(mu)
    div_eta = div_eta.at[:, 1:-1].set((flux_eta[:, 1:] - flux_eta[:, :-1]) / dy)
    div_eta = div_eta.at[:, 0].set(flux_eta[:, 0] / dy)
    div_eta = div_eta.at[:, -1].set(-flux_eta[:, -1] / dy)

    return div_x + div_eta


@jit
def jax_phase_diffusive_term_simple(
    phi, dx, dy, Pe, epsilon, f_1_grid, f_2_grid,
    lambda_willmore=0.0, epsilon_willmore=0.0,
    mu_top_bc=BC_NEUMANN, mu_bottom_bc=BC_NEUMANN,
    mu_left_bc=BC_NEUMANN, mu_right_bc=BC_NEUMANN,
    mu_top_value=0.0, mu_bottom_value=0.0,
    mu_left_value=0.0, mu_right_value=0.0,
    use_degenerate_mobility=False, degenerate_mobility_clip=0.0, degenerate_mobility_blend=0.0,
):
    lap_phi = jax_laplacian(phi, dx, dy, f_1_grid, f_2_grid)
    mu_ch = jax_df_2(phi) - epsilon**2 * lap_phi
    mu_ch = jax_apply_chemical_potential_bc(
        mu_ch, dx, dy,
        top_bc=mu_top_bc, bottom_bc=mu_bottom_bc,
        left_bc=mu_left_bc, right_bc=mu_right_bc,
        top_value=mu_top_value, bottom_value=mu_bottom_value,
        left_value=mu_left_value, right_value=mu_right_value,
    )
    willmore_active = (lambda_willmore > 0) & (epsilon_willmore > 0)
    mu_willmore = jax_willmore_chemical_potential(phi, dx, dy, f_1_grid, f_2_grid, epsilon_willmore)
    mu_willmore = jax_apply_chemical_potential_bc(
        mu_willmore, dx, dy,
        top_bc=mu_top_bc, bottom_bc=mu_bottom_bc,
        left_bc=mu_left_bc, right_bc=mu_right_bc,
        top_value=mu_top_value, bottom_value=mu_bottom_value,
        left_value=mu_left_value, right_value=mu_right_value,
    )
    mu_total = mu_ch + jnp.where(willmore_active, lambda_willmore * mu_willmore, 0.0)
    mu_centered = mu_total - jnp.mean(mu_total)
    unit_mobility = jnp.ones_like(phi)
    diff_const = (1.0 / Pe) * jax_terrain_conservative_diffusion(
        mu_centered, unit_mobility, dx, dy, f_1_grid
    )
    mobility = jax_degenerate_mobility(
        phi, mobility_clip=degenerate_mobility_clip, mobility_blend=degenerate_mobility_blend
    )
    diff_deg = (1.0 / Pe) * jax_terrain_conservative_diffusion(
        mu_centered, mobility, dx, dy, f_1_grid
    )
    flag = jnp.asarray(use_degenerate_mobility, dtype=bool)
    return jnp.where(flag, diff_deg, diff_const)


@jit
def jax_phase_diffusive_term_ghost(
    phi, dx, dy, Pe, epsilon, bottom_ghost_phi, f_1_grid=None, f_2_grid=None,
    lambda_willmore=0.0, epsilon_willmore=0.0,
    mu_top_bc=BC_NEUMANN, mu_bottom_bc=BC_NEUMANN,
    mu_left_bc=BC_NEUMANN, mu_right_bc=BC_NEUMANN,
    mu_top_value=0.0, mu_bottom_value=0.0,
    mu_left_value=0.0, mu_right_value=0.0,
    use_degenerate_mobility=False, degenerate_mobility_clip=0.0, degenerate_mobility_blend=0.0,
):
    if f_1_grid is None:
        f_1_grid = jnp.zeros_like(phi)
    if f_2_grid is None:
        f_2_grid = jnp.zeros_like(phi)
    lap_phi = jax_laplacian_terrain_bottom_ghost(phi, dx, dy, f_1_grid, f_2_grid, bottom_ghost_phi)
    mu_ch = jax_df_2(phi) - epsilon**2 * lap_phi
    mu_ch = jax_apply_chemical_potential_bc_skip_bottom(
        mu_ch, dx, dy,
        top_bc=mu_top_bc, bottom_bc=mu_bottom_bc,
        left_bc=mu_left_bc, right_bc=mu_right_bc,
        top_value=mu_top_value, bottom_value=mu_bottom_value,
        left_value=mu_left_value, right_value=mu_right_value,
    )
    willmore_active = (lambda_willmore > 0) & (epsilon_willmore > 0)
    mu_willmore = epsilon_willmore * jax_laplacian_terrain_bottom_ghost(
        lap_phi, dx, dy, f_1_grid, f_2_grid, bottom_ghost_phi
    )
    mu_willmore = jax_apply_chemical_potential_bc_skip_bottom(
        mu_willmore, dx, dy,
        top_bc=mu_top_bc, bottom_bc=mu_bottom_bc,
        left_bc=mu_left_bc, right_bc=mu_right_bc,
        top_value=mu_top_value, bottom_value=mu_bottom_value,
        left_value=mu_left_value, right_value=mu_right_value,
    )
    mu_total = mu_ch + jnp.where(willmore_active, lambda_willmore * mu_willmore, 0.0)
    mu_centered = mu_total - jnp.mean(mu_total)
    unit_mobility = jnp.ones_like(phi)
    diff_const = (1.0 / Pe) * jax_terrain_conservative_diffusion(
        mu_centered, unit_mobility, dx, dy, f_1_grid
    )
    mobility = jax_degenerate_mobility(
        phi, mobility_clip=degenerate_mobility_clip, mobility_blend=degenerate_mobility_blend
    )
    diff_deg = (1.0 / Pe) * jax_terrain_conservative_diffusion(
        mu_centered, mobility, dx, dy, f_1_grid
    )
    flag = jnp.asarray(use_degenerate_mobility, dtype=bool)
    return jnp.where(flag, diff_deg, diff_const)


@jit
def jax_apply_chemical_potential_bc_skip_bottom(
    mu_c,
    dx,
    dy,
    top_bc=BC_NEUMANN,
    bottom_bc=BC_NEUMANN,
    left_bc=BC_NEUMANN,
    right_bc=BC_NEUMANN,
    top_value=0.0,
    bottom_value=0.0,
    left_value=0.0,
    right_value=0.0,
):
    """Apply chemical-potential BCs but keep the bottom row as a physical row.

    This is needed by the ghost-cell phase-field solver, where the bottom wall
    is handled through the discrete operator rather than by overwriting `mu[:, 0]`.
    """
    del dx, dy, bottom_bc, bottom_value
    bottom_row = mu_c[:, 0]
    mu_c = jax_apply_chemical_potential_bc(
        mu_c,
        0.0,
        0.0,
        top_bc=top_bc,
        bottom_bc=BC_DIRICHLET,
        left_bc=left_bc,
        right_bc=right_bc,
        top_value=top_value,
        bottom_value=0.0,
        left_value=left_value,
        right_value=right_value,
    )
    return mu_c.at[:, 0].set(bottom_row)


@jit
def jax_update_phase(phi, U, current_dt, dx, dy, Pe, epsilon, contact_angle, f_1_grid, f_2_grid,
                     lambda_willmore=0.0, epsilon_willmore=0.0,
                     mu_top_bc=BC_NEUMANN, mu_bottom_bc=BC_NEUMANN,
                     mu_left_bc=BC_NEUMANN, mu_right_bc=BC_NEUMANN,
                     mu_top_value=0.0, mu_bottom_value=0.0,
                     mu_left_value=0.0, mu_right_value=0.0,
                     use_degenerate_mobility=False, degenerate_mobility_clip=0.0, degenerate_mobility_blend=0.0):
    """JAX-compiled phase field update with optional Willmore regularization.
    
    Implements the Cahn-Hilliard-Willmore equation:
    ∂φ/∂t + u·∇φ = (1/Pe) Δ(μ_CH + λ_W μ_W)
    
    where:
    - μ_CH = f'(φ) - ε²Δφ is the Cahn-Hilliard chemical potential
    - μ_W = ε_W Δ²φ is the Willmore chemical potential (4th-order smoothing)
    - λ_W is the Willmore regularization strength
    - ε_W is the Willmore regularization parameter (typically << ε)
    
    The Willmore term provides additional smoothing to prevent high-curvature
    singularities while maintaining energy stability.
    
    Args:
        phi: Phase field (Nx, Ny).
        U: Velocity field (Nx, Ny, 2).
        current_dt: Time step.
        dx, dy: Grid spacing.
        Pe: Peclet number.
        epsilon: Interface thickness.
        contact_angle: Contact angle in degrees (unused, kept for compatibility).
        lambda_willmore: Willmore regularization strength (0 = disabled).
        epsilon_willmore: Willmore regularization parameter (typically 0.001 * epsilon).
    
    Returns:
        Updated phase field.
    """
    convective_term = jax_phase_convective_term(phi, U, dx, dy, f_1_grid)
    diffusive_term = jax_phase_diffusive_term_simple(
        phi, dx, dy, Pe, epsilon, f_1_grid, f_2_grid,
        lambda_willmore=lambda_willmore,
        epsilon_willmore=epsilon_willmore,
        mu_top_bc=mu_top_bc, mu_bottom_bc=mu_bottom_bc,
        mu_left_bc=mu_left_bc, mu_right_bc=mu_right_bc,
        mu_top_value=mu_top_value, mu_bottom_value=mu_bottom_value,
        mu_left_value=mu_left_value, mu_right_value=mu_right_value,
        use_degenerate_mobility=use_degenerate_mobility,
        degenerate_mobility_clip=degenerate_mobility_clip,
        degenerate_mobility_blend=degenerate_mobility_blend,
    )
    rhs_phi = -convective_term + diffusive_term

    phi = phi + current_dt * rhs_phi
    
    return phi


@jit
def jax_update_phase_ghost(phi, U, current_dt, dx, dy, Pe, epsilon, contact_angle, bottom_ghost_phi,
                           f_1_grid=None, f_2_grid=None,
                           lambda_willmore=0.0, epsilon_willmore=0.0,
                           mu_top_bc=BC_NEUMANN, mu_bottom_bc=BC_NEUMANN,
                           mu_left_bc=BC_NEUMANN, mu_right_bc=BC_NEUMANN,
                           mu_top_value=0.0, mu_bottom_value=0.0,
                           mu_left_value=0.0, mu_right_value=0.0,
                           use_degenerate_mobility=False, degenerate_mobility_clip=0.0, degenerate_mobility_blend=0.0):
    """Phase update with a bottom ghost cell for the wetting BC."""
    if f_1_grid is None:
        f_1_grid = jnp.zeros_like(phi)
    if f_2_grid is None:
        f_2_grid = jnp.zeros_like(phi)
    convective_term = jax_phase_convective_term(phi, U, dx, dy, f_1_grid)
    diffusive_term = jax_phase_diffusive_term_ghost(
        phi, dx, dy, Pe, epsilon, bottom_ghost_phi, f_1_grid, f_2_grid,
        lambda_willmore=lambda_willmore,
        epsilon_willmore=epsilon_willmore,
        mu_top_bc=mu_top_bc, mu_bottom_bc=mu_bottom_bc,
        mu_left_bc=mu_left_bc, mu_right_bc=mu_right_bc,
        mu_top_value=mu_top_value, mu_bottom_value=mu_bottom_value,
        mu_left_value=mu_left_value, mu_right_value=mu_right_value,
        use_degenerate_mobility=use_degenerate_mobility,
        degenerate_mobility_clip=degenerate_mobility_clip,
        degenerate_mobility_blend=degenerate_mobility_blend,
    )
    rhs_phi = -convective_term + diffusive_term

    return phi + current_dt * rhs_phi


class BasePhaseFieldSolver:
    """Common phase-field solver setup."""
    
    def __init__(self, Pe, epsilon, contact_angle, config=None):
        """Initialize solver.
        
        Args:
            Pe: Peclet number.
            epsilon: Interface thickness.
            contact_angle: Contact angle in degrees.
            config: Optional config for boundary conditions and smoothing params.
        """
        self.Pe = Pe
        self.epsilon = epsilon
        self.contact_angle = contact_angle
        self.config = config
        
        # Willmore regularization parameters
        # lambda_willmore: Willmore regularization strength (0 = disabled)
        # epsilon_willmore: Willmore regularization parameter (typically 0.001 * epsilon)
        if config is not None:
            phys = config.get("physical_params", {})
            self.lambda_willmore = phys.get("lambda_willmore", 0.0)
            # Default epsilon_willmore to 0.001 * epsilon if not specified
            self.epsilon_willmore = phys.get("epsilon_willmore", 0.001 * self.epsilon)
            # Backward compatibility: check for old parameter names
            if "lambda_smooth" in phys and self.lambda_willmore == 0.0:
                self.lambda_willmore = phys.get("lambda_smooth", 0.0)
                self.epsilon_willmore = 0.001 * self.epsilon  # Use default
        else:
            self.lambda_willmore = 0.0
            self.epsilon_willmore = 0.001 * self.epsilon
        
        if config is not None:
            from boundary_conditions.phase_field_bc import PhaseFieldBoundaryConditions
            from boundary_conditions.chemical_potential_bc import ChemicalPotentialBoundaryConditions
            from boundary_conditions.advection_bc import AdvectionBoundaryConditions
            solver_cfg = config.get("solver_params", {})
            self.bc_manager = PhaseFieldBoundaryConditions(config)
            self.chemical_potential_bc_manager = ChemicalPotentialBoundaryConditions(config)
            self.advection_bc_manager = AdvectionBoundaryConditions(config)
            self.mu_bc_codes, self.mu_bc_values = self.chemical_potential_bc_manager.jax_metadata()
            self.phase_update_mode = solver_cfg.get("phase_update_mode", "monolithic")
            self.phase_diffusion_solver_backend = solver_cfg.get("phase_diffusion_solver_backend", "pyamg")
            self.phase_diffusion_solver_tol = solver_cfg.get("phase_diffusion_solver_tol", 1e-8)
            self.phase_diffusion_solver_maxiter = solver_cfg.get("phase_diffusion_solver_maxiter", 500)
            self.use_degenerate_mobility = bool(solver_cfg.get("use_degenerate_mobility", False))
            self.degenerate_mobility_clip = float(solver_cfg.get("degenerate_mobility_clip", 0.0))
            self.degenerate_mobility_blend = float(solver_cfg.get("degenerate_mobility_blend", 0.0))
            self.degenerate_mobility_imex = bool(solver_cfg.get("degenerate_mobility_imex", True))
            self.degenerate_mobility_imex_mref = float(solver_cfg.get("degenerate_mobility_imex_mref", 1.0))
        else:
            self.bc_manager = None
            self.chemical_potential_bc_manager = None
            self.advection_bc_manager = None
            self.mu_bc_codes = (BC_NEUMANN, BC_NEUMANN, BC_NEUMANN, BC_NEUMANN)
            self.mu_bc_values = (0.0, 0.0, 0.0, 0.0)
            self.phase_update_mode = "monolithic"
            self.phase_diffusion_solver_backend = "pyamg"
            self.phase_diffusion_solver_tol = 1e-8
            self.phase_diffusion_solver_maxiter = 500
            self.use_degenerate_mobility = False
            self.degenerate_mobility_clip = 0.0
            self.degenerate_mobility_blend = 0.0
            self.degenerate_mobility_imex = True
            self.degenerate_mobility_imex_mref = 1.0

        self._phase_laplacian_cache = None
        self._phase_conservative_outer_cache = None
        self._phase_helmholtz_cache = {}
    
    def _apply_post_update_bcs(self, phi_new, U, current_dt, dx, dy, geometry, psi=None):
        skip_bottom_adv = self._should_skip_bottom_advection_bc()
        if self.advection_bc_manager is not None:
            phi_new = self.advection_bc_manager.apply_boundary_conditions(
                phi_new, U, current_dt, dx, dy, use_jax=True, geometry=geometry, skip_bottom=skip_bottom_adv)

        if self.bc_manager is not None:
            phi_new = self.bc_manager.apply_boundary_conditions(
                phi_new, dx, dy, use_jax=True, psi=psi, U=U, geometry=geometry)

        return phi_new

    def _should_skip_bottom_advection_bc(self) -> bool:
        """Bottom advection BC must not override contact-angle phase BC at the wall."""
        if self.bc_manager is None:
            return False
        bc_raw = getattr(self.bc_manager, "bc_raw", {}) or {}
        return bc_raw.get("bottom") == "contact_angle"

    def _apply_advection_bcs(self, phi_new, U, current_dt, dx, dy, geometry, skip_bottom=False):
        if self.advection_bc_manager is not None:
            phi_new = self.advection_bc_manager.apply_boundary_conditions(
                phi_new, U, current_dt, dx, dy, use_jax=True, geometry=geometry, skip_bottom=skip_bottom
            )
        return phi_new

    def _apply_phase_bcs_only(self, phi_new, dx, dy, geometry, psi=None, U=None):
        if self.bc_manager is not None:
            phi_new = self.bc_manager.apply_boundary_conditions(
                phi_new, dx, dy, use_jax=True, psi=psi, U=U, geometry=geometry
            )
        return phi_new

    def _preserve_phase_sum_if_requested(self, phi_old, phi_new):
        if self.bc_manager is None:
            return phi_new
        contact_angle_bc = getattr(self.bc_manager, "contact_angle_bc", None)
        if contact_angle_bc is None or not getattr(contact_angle_bc, "conserve_phi_sum", False):
            return phi_new
        return phi_new + (jnp.mean(phi_old) - jnp.mean(phi_new))

    def _terrain_cache_signature(self, geometry):
        f_1 = np.asarray(geometry.f_1_grid, dtype=np.float64)
        f_2 = np.asarray(geometry.f_2_grid, dtype=np.float64)
        has_terrain = bool(np.any(np.abs(f_1) > 1e-14) or np.any(np.abs(f_2) > 1e-14))
        if not has_terrain:
            return (False, 0.0, 0.0, 0.0, 0.0)
        return (
            True,
            float(np.min(f_1)),
            float(np.max(f_1)),
            float(np.min(f_2)),
            float(np.max(f_2)),
        )

    def _get_phase_laplacian_matrix(self, nx, ny, dx, dy, geometry=None):
        terrain_sig = self._terrain_cache_signature(geometry) if geometry is not None else (False, 0.0, 0.0, 0.0, 0.0)
        cache_key = (nx, ny, float(dx), float(dy), terrain_sig)
        if self._phase_laplacian_cache is not None and self._phase_laplacian_cache[0] == cache_key:
            return self._phase_laplacian_cache[1]

        from solvers.sparse_solver import SparseSolverWrapper
        f_1_grid = geometry.f_1_grid if terrain_sig[0] and geometry is not None else None
        f_2_grid = geometry.f_2_grid if terrain_sig[0] and geometry is not None else None

        solver = SparseSolverWrapper(
            nx, ny, dx, dy,
            backend=self.phase_diffusion_solver_backend,
            f_1_grid=f_1_grid,
            f_2_grid=f_2_grid,
            solver_params={
                "accel": "bicgstab",
                "tol": self.phase_diffusion_solver_tol,
                "maxiter": self.phase_diffusion_solver_maxiter,
            },
        )
        if self.chemical_potential_bc_manager is not None:
            solver.set_bcs_from_manager(self.chemical_potential_bc_manager)
        A = solver.A.tocsr()
        self._phase_laplacian_cache = (cache_key, A)
        return A

    def _apply_sparse_operator(self, A, field_np):
        rhs_t = np.asarray(field_np.T, dtype=np.float64)
        out = A.dot(rhs_t.flatten()).reshape(rhs_t.shape)
        return out.T

    def _create_first_derivative_matrix(self, n, h):
        import scipy.sparse
        rows, cols, data = [], [], []
        for i in range(n):
            if i == 0:
                rows.extend([i, i])
                cols.extend([0, 1])
                data.extend([-1.0 / h, 1.0 / h])
            elif i == n - 1:
                rows.extend([i, i])
                cols.extend([n - 2, n - 1])
                data.extend([-1.0 / h, 1.0 / h])
            else:
                rows.extend([i, i])
                cols.extend([i - 1, i + 1])
                data.extend([-0.5 / h, 0.5 / h])
        return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    def _create_face_gradient_matrix(self, n, h):
        import scipy.sparse
        rows, cols, data = [], [], []
        for i in range(n - 1):
            rows.extend([i, i])
            cols.extend([i, i + 1])
            data.extend([-1.0 / h, 1.0 / h])
        return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n - 1, n))

    def _create_face_average_matrix(self, n):
        import scipy.sparse
        rows, cols, data = [], [], []
        for i in range(n - 1):
            rows.extend([i, i])
            cols.extend([i, i + 1])
            data.extend([0.5, 0.5])
        return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n - 1, n))

    def _create_face_divergence_matrix(self, n, h):
        import scipy.sparse
        rows, cols, data = [], [], []
        for i in range(n):
            if i == 0:
                rows.append(i)
                cols.append(0)
                data.append(1.0 / h)
            elif i == n - 1:
                rows.append(i)
                cols.append(n - 2)
                data.append(-1.0 / h)
            else:
                rows.extend([i, i])
                cols.extend([i - 1, i])
                data.extend([-1.0 / h, 1.0 / h])
        return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n - 1))

    def _get_conservative_outer_matrix(self, nx, ny, dx, dy, geometry):
        terrain_sig = self._terrain_cache_signature(geometry)
        cache_key = (nx, ny, float(dx), float(dy), terrain_sig)
        if (
            self._phase_conservative_outer_cache is not None
            and self._phase_conservative_outer_cache[0] == cache_key
        ):
            return self._phase_conservative_outer_cache[1]

        import scipy.sparse

        ix = scipy.sparse.identity(nx, format="csr", dtype=np.float64)
        iy = scipy.sparse.identity(ny, format="csr", dtype=np.float64)

        dx_cell = scipy.sparse.kron(iy, self._create_first_derivative_matrix(nx, dx), format="csr")
        dy_cell = scipy.sparse.kron(self._create_first_derivative_matrix(ny, dy), ix, format="csr")

        gx = scipy.sparse.kron(iy, self._create_face_gradient_matrix(nx, dx), format="csr")
        gy = scipy.sparse.kron(self._create_face_gradient_matrix(ny, dy), ix, format="csr")
        ax = scipy.sparse.kron(iy, self._create_face_average_matrix(nx), format="csr")
        ay = scipy.sparse.kron(self._create_face_average_matrix(ny), ix, format="csr")
        divx = scipy.sparse.kron(iy, self._create_face_divergence_matrix(nx, dx), format="csr")
        divy = scipy.sparse.kron(self._create_face_divergence_matrix(ny, dy), ix, format="csr")

        f_1 = np.asarray(geometry.f_1_grid, dtype=np.float64)
        f_1_xface = 0.5 * (f_1[1:, :] + f_1[:-1, :])
        f_1_yface = 0.5 * (f_1[:, 1:] + f_1[:, :-1])
        diag_x = scipy.sparse.diags(f_1_xface.T.reshape(-1), format="csr")
        diag_y = scipy.sparse.diags(f_1_yface.T.reshape(-1), format="csr")
        diag_metric_y = scipy.sparse.diags((1.0 + f_1_yface**2).T.reshape(-1), format="csr")

        flux_x_op = gx - diag_x @ (ax @ dy_cell)
        flux_y_op = -diag_y @ (ay @ dx_cell) + diag_metric_y @ gy
        conservative_outer = (divx @ flux_x_op + divy @ flux_y_op).tocsr()
        self._phase_conservative_outer_cache = (cache_key, conservative_outer)
        return conservative_outer

    def _apply_conservative_outer_operator(self, field_np, dx, dy, geometry):
        matrix = self._get_conservative_outer_matrix(
            field_np.shape[0], field_np.shape[1], dx, dy, geometry
        )
        return self._apply_sparse_operator(matrix, field_np)

    def _phase_laplacian_with_wall_energy_bottom(self, A_mu, bottom_q_prime, dy):
        """Linearized phase Laplacian with full-wall wetting at the bottom.

        The explicit ghost residual uses
            phi_g = phi_1 - 2 dy q(phi_0)
        where q is the wall-free-energy normal derivative.  For the implicit
        stabilizer we freeze dq/dphi at the old state.  The affine part cancels
        in the IMEX replacement, so only the bottom diagonal correction is
        needed for the linear matrix.
        """
        q_prime = np.asarray(bottom_q_prime, dtype=np.float64)
        A_phi = A_mu.tolil(copy=True)
        nx = q_prime.shape[0]
        for i in range(nx):
            # Matrix flattening is field.T.flatten(), so bottom row j=0 has k=i.
            A_phi[i, i] = A_phi[i, i] - 2.0 * q_prime[i] / dy
        return A_phi.tocsr()

    def _solve_helmholtz_biharmonic(self, M, rhs_np, x0=None):
        import scipy.sparse.linalg
        rhs_t = np.asarray(rhs_np.T, dtype=np.float64)
        rhs_flat = rhs_t.flatten()
        x0_flat = None if x0 is None else np.asarray(x0.T, dtype=np.float64).flatten()
        sol, info = scipy.sparse.linalg.bicgstab(
            M, rhs_flat, x0=x0_flat,
            rtol=self.phase_diffusion_solver_tol,
            maxiter=self.phase_diffusion_solver_maxiter,
        )
        if info != 0 or not np.all(np.isfinite(sol)):
            sol = np.nan_to_num(rhs_flat, nan=0.0, posinf=0.0, neginf=0.0)
        return sol.reshape(rhs_t.shape).T

    def _semi_implicit_ch_step(self, phi, U, current_dt, dx, dy, geometry, psi=None, skip_bottom=False):
        # Keep a compatibility fallback to the legacy explicit variable-mobility path.
        if self.use_degenerate_mobility and not self.degenerate_mobility_imex:
            if skip_bottom:
                bottom_velocity_bc = self.config.get("boundary_conditions", {}).get("velocity", {}).get("bottom", "no_slip")
                bottom_ghost_phi = self.bc_manager.contact_angle_bc.build_bottom_ghost_row_jax(
                    phi, dx, dy, geometry, psi=psi, U=U, bottom_velocity_bc=bottom_velocity_bc
                )
                phi_new = jax_update_phase_ghost(
                    phi, U, current_dt, dx, dy,
                    self.Pe, self.epsilon, self.contact_angle,
                    bottom_ghost_phi,
                    geometry.f_1_grid, geometry.f_2_grid,
                    lambda_willmore=self.lambda_willmore,
                    epsilon_willmore=self.epsilon_willmore,
                    mu_top_bc=self.mu_bc_codes[0],
                    mu_bottom_bc=self.mu_bc_codes[1],
                    mu_left_bc=self.mu_bc_codes[2],
                    mu_right_bc=self.mu_bc_codes[3],
                    mu_top_value=self.mu_bc_values[0],
                    mu_bottom_value=self.mu_bc_values[1],
                    mu_left_value=self.mu_bc_values[2],
                    mu_right_value=self.mu_bc_values[3],
                    use_degenerate_mobility=True,
                    degenerate_mobility_clip=self.degenerate_mobility_clip,
                    degenerate_mobility_blend=self.degenerate_mobility_blend,
                )
            else:
                phi_new = jax_update_phase(
                    phi, U, current_dt, dx, dy,
                    self.Pe, self.epsilon, self.contact_angle,
                    geometry.f_1_grid, geometry.f_2_grid,
                    lambda_willmore=self.lambda_willmore,
                    epsilon_willmore=self.epsilon_willmore,
                    mu_top_bc=self.mu_bc_codes[0],
                    mu_bottom_bc=self.mu_bc_codes[1],
                    mu_left_bc=self.mu_bc_codes[2],
                    mu_right_bc=self.mu_bc_codes[3],
                    mu_top_value=self.mu_bc_values[0],
                    mu_bottom_value=self.mu_bc_values[1],
                    mu_left_value=self.mu_bc_values[2],
                    mu_right_value=self.mu_bc_values[3],
                    use_degenerate_mobility=True,
                    degenerate_mobility_clip=self.degenerate_mobility_clip,
                    degenerate_mobility_blend=self.degenerate_mobility_blend,
                )
            skip_bottom = bool(skip_bottom or self._should_skip_bottom_advection_bc())
            phi_new = self._apply_advection_bcs(phi_new, U, current_dt, dx, dy, geometry, skip_bottom=skip_bottom)
            phi_new = self._apply_phase_bcs_only(phi_new, dx, dy, geometry, psi=psi, U=U)
            phi_new = self._preserve_phase_sum_if_requested(phi, phi_new)
            return phi_new

        phi_np = np.asarray(phi)
        convective_term = np.asarray(
            jax_phase_convective_term(phi, U, dx, dy, geometry.f_1_grid)
        )
        nx, ny = phi_np.shape
        A = self._get_phase_laplacian_matrix(nx, ny, dx, dy, geometry=geometry)
        C = self._get_conservative_outer_matrix(nx, ny, dx, dy, geometry)
        A_phi = A
        bottom_velocity_bc = self.config.get("boundary_conditions", {}).get("velocity", {}).get("bottom", "no_slip")
        bottom_ghost_phi = None
        contact_laplacian_old = None
        if skip_bottom:
            bottom_ghost_phi = self.bc_manager.contact_angle_bc.build_bottom_ghost_row_jax(
                phi, dx, dy, geometry, psi=psi, U=U, bottom_velocity_bc=bottom_velocity_bc
            )
            if getattr(self.bc_manager.contact_angle_bc, "contact_angle_ghost_law_code", 0) == 1:
                bottom_q_prime = np.asarray(
                    self.bc_manager.contact_angle_bc.build_bottom_wall_energy_q_prime_jax(
                        phi, dx, dy, geometry, psi=psi, U=U, bottom_velocity_bc=bottom_velocity_bc
                    )
                )
                A_phi = self._phase_laplacian_with_wall_energy_bottom(A, bottom_q_prime, dy)
            else:
                contact_laplacian_old = np.asarray(
                    jax_laplacian_terrain_bottom_ghost(
                        phi, dx, dy, geometry.f_1_grid, geometry.f_2_grid, bottom_ghost_phi
                    )
                )
        phase_linear_operator = C @ A_phi
        mobility_ref = 1.0
        if self.use_degenerate_mobility:
            mobility_ref = max(float(self.degenerate_mobility_imex_mref), 0.0)
        alpha = current_dt * mobility_ref * (self.epsilon ** 2) / self.Pe
        M = None
        if alpha > 0.0:
            cache_key = (nx, ny, float(dx), float(dy), self._terrain_cache_signature(geometry), float(alpha))
            can_cache_helmholtz = A_phi is A
            if can_cache_helmholtz and cache_key in self._phase_helmholtz_cache:
                M = self._phase_helmholtz_cache[cache_key]
            else:
                import scipy.sparse
                M = scipy.sparse.identity(A.shape[0], format="csr") + alpha * phase_linear_operator
                if can_cache_helmholtz:
                    self._phase_helmholtz_cache[cache_key] = M
        if self.use_degenerate_mobility:
            if skip_bottom:
                diffusive_deg_old = np.asarray(
                    jax_phase_diffusive_term_ghost(
                        phi, dx, dy, self.Pe, self.epsilon, bottom_ghost_phi,
                        geometry.f_1_grid, geometry.f_2_grid,
                        lambda_willmore=self.lambda_willmore,
                        epsilon_willmore=self.epsilon_willmore,
                        mu_top_bc=self.mu_bc_codes[0],
                        mu_bottom_bc=self.mu_bc_codes[1],
                        mu_left_bc=self.mu_bc_codes[2],
                        mu_right_bc=self.mu_bc_codes[3],
                        mu_top_value=self.mu_bc_values[0],
                        mu_bottom_value=self.mu_bc_values[1],
                        mu_left_value=self.mu_bc_values[2],
                        mu_right_value=self.mu_bc_values[3],
                        use_degenerate_mobility=True,
                        degenerate_mobility_clip=self.degenerate_mobility_clip,
                        degenerate_mobility_blend=self.degenerate_mobility_blend,
                    )
                )
            else:
                diffusive_deg_old = np.asarray(
                    jax_phase_diffusive_term_simple(
                        phi, dx, dy, self.Pe, self.epsilon, geometry.f_1_grid, geometry.f_2_grid,
                        lambda_willmore=self.lambda_willmore,
                        epsilon_willmore=self.epsilon_willmore,
                        mu_top_bc=self.mu_bc_codes[0],
                        mu_bottom_bc=self.mu_bc_codes[1],
                        mu_left_bc=self.mu_bc_codes[2],
                        mu_right_bc=self.mu_bc_codes[3],
                        mu_top_value=self.mu_bc_values[0],
                        mu_bottom_value=self.mu_bc_values[1],
                        mu_left_value=self.mu_bc_values[2],
                        mu_right_value=self.mu_bc_values[3],
                        use_degenerate_mobility=True,
                        degenerate_mobility_clip=self.degenerate_mobility_clip,
                        degenerate_mobility_blend=self.degenerate_mobility_blend,
                    )
                )
            # IMEX split: keep full degenerate mobility diffusion explicit and
            # stabilize only the linear CH biharmonic part implicitly.
            rhs = phi_np - current_dt * convective_term + current_dt * diffusive_deg_old
            if mobility_ref > 0.0:
                lap_phi_old = self._apply_sparse_operator(A_phi, phi_np)
                lap2_phi_old = self._apply_sparse_operator(A, lap_phi_old)
                rhs = rhs + current_dt * mobility_ref * (self.epsilon ** 2 / self.Pe) * lap2_phi_old
        else:
            nonlinear_term = self._apply_conservative_outer_operator(
                np.asarray(jax_df_2(phi)), dx, dy, geometry
            )
            rhs = phi_np - current_dt * convective_term + (current_dt / self.Pe) * nonlinear_term
            if contact_laplacian_old is not None:
                base_laplacian_old = self._apply_sparse_operator(A_phi, phi_np)
                contact_delta = contact_laplacian_old - base_laplacian_old
                contact_delta_term = self._apply_conservative_outer_operator(
                    contact_delta, dx, dy, geometry
                )
                rhs = rhs - current_dt * (self.epsilon ** 2 / self.Pe) * contact_delta_term

        phi_new_np = rhs if M is None else self._solve_helmholtz_biharmonic(M, rhs, x0=phi_np)
        phi_new = jnp.asarray(phi_new_np, dtype=phi.dtype)
        skip_bottom = bool(skip_bottom or self._should_skip_bottom_advection_bc())
        phi_new = self._apply_advection_bcs(phi_new, U, current_dt, dx, dy, geometry, skip_bottom=skip_bottom)
        phi_new = self._apply_phase_bcs_only(phi_new, dx, dy, geometry, psi=psi, U=U)
        phi_new = self._preserve_phase_sum_if_requested(phi, phi_new)
        return phi_new


class PhaseFieldSolverSimple(BasePhaseFieldSolver):
    """Legacy phase-field solver with post-update boundary enforcement."""

    def update(self, phi, U, current_dt, dx, dy, geometry, use_jax=True, psi=None):
        """Update phase field using the legacy operator path."""
        if self.phase_update_mode == "semi_implicit_ch":
            return self._semi_implicit_ch_step(phi, U, current_dt, dx, dy, geometry, psi=psi, skip_bottom=False)

        phi_new = jax_update_phase(phi, U, current_dt, dx, dy,
                                   self.Pe, self.epsilon, self.contact_angle,
                                   geometry.f_1_grid, geometry.f_2_grid,
                                   lambda_willmore=self.lambda_willmore,
                                   epsilon_willmore=self.epsilon_willmore,
                                   mu_top_bc=self.mu_bc_codes[0],
                                   mu_bottom_bc=self.mu_bc_codes[1],
                                   mu_left_bc=self.mu_bc_codes[2],
                                   mu_right_bc=self.mu_bc_codes[3],
                                   mu_top_value=self.mu_bc_values[0],
                                   mu_bottom_value=self.mu_bc_values[1],
                                   mu_left_value=self.mu_bc_values[2],
                                   mu_right_value=self.mu_bc_values[3],
                                   use_degenerate_mobility=self.use_degenerate_mobility,
                                   degenerate_mobility_clip=self.degenerate_mobility_clip,
                                   degenerate_mobility_blend=self.degenerate_mobility_blend)
        return self._apply_post_update_bcs(phi_new, U, current_dt, dx, dy, geometry, psi=psi)


class PhaseFieldSolverGhostCell(BasePhaseFieldSolver):
    """Experimental ghost-cell phase-field solver for bottom wetting BCs."""

    def update(self, phi, U, current_dt, dx, dy, geometry, use_jax=True, psi=None):
        """Update phase field using the ghost-cell bottom-wall path."""
        if self.phase_update_mode == "semi_implicit_ch":
            return self._semi_implicit_ch_step(phi, U, current_dt, dx, dy, geometry, psi=psi, skip_bottom=True)

        bottom_velocity_bc = self.config.get("boundary_conditions", {}).get("velocity", {}).get("bottom", "no_slip")
        bottom_ghost_phi = self.bc_manager.contact_angle_bc.build_bottom_ghost_row_jax(
            phi, dx, dy, geometry, psi=psi, U=U, bottom_velocity_bc=bottom_velocity_bc
        )

        phi_new = jax_update_phase_ghost(
            phi, U, current_dt, dx, dy,
            self.Pe, self.epsilon, self.contact_angle,
            bottom_ghost_phi,
            geometry.f_1_grid, geometry.f_2_grid,
            lambda_willmore=self.lambda_willmore,
            epsilon_willmore=self.epsilon_willmore,
            mu_top_bc=self.mu_bc_codes[0],
            mu_bottom_bc=self.mu_bc_codes[1],
            mu_left_bc=self.mu_bc_codes[2],
            mu_right_bc=self.mu_bc_codes[3],
            mu_top_value=self.mu_bc_values[0],
            mu_bottom_value=self.mu_bc_values[1],
            mu_left_value=self.mu_bc_values[2],
            mu_right_value=self.mu_bc_values[3],
            use_degenerate_mobility=self.use_degenerate_mobility,
            degenerate_mobility_clip=self.degenerate_mobility_clip,
            degenerate_mobility_blend=self.degenerate_mobility_blend,
        )
        if self.advection_bc_manager is not None:
            phi_new = self.advection_bc_manager.apply_boundary_conditions(
                phi_new, U, current_dt, dx, dy, use_jax=True, geometry=geometry, skip_bottom=True
            )
        if self.bc_manager is not None:
            phi_new = self.bc_manager.apply_boundary_conditions(
                phi_new, dx, dy, use_jax=True, psi=psi, U=U, geometry=geometry
            )
        return phi_new


# Backward-compatible alias for code paths not yet migrated.
PhaseFieldSolver = PhaseFieldSolverSimple
