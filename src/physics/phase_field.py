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
from physics.free_energy import jax_free_energy_derivative
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
def jax_phase_conservative_advection(phi, u_face, v_face, dx, dy, f_1_grid, flux_shift=0.0):
    """Finite-volume MAC advection div((phi - flux_shift) * u_contravariant).

    In terrain-following coordinates y = eta + h(x), J=1 and the conservative
    scalar transport fluxes are
        F_x   = (phi - flux_shift) * u
        F_eta = (phi - flux_shift) * (v - h'(x) u)
    where u,v are physical velocity components.  For flat grids this reduces to
    the standard MAC divergence of upwinded scalar fluxes.

    `flux_shift=0` is the legacy conservative phi flux.  `flux_shift=1` is the
    phi-equation form of conservative liquid-fraction transport c=(1-phi)/2,
    which keeps pure gas invariant even when a boundary row has nonzero
    discrete MAC divergence.
    """
    nx, ny = phi.shape
    q = phi - flux_shift

    phi_w = jnp.concatenate([q[0:1, :], q], axis=0)
    phi_e = jnp.concatenate([q, q[-1:, :]], axis=0)
    phi_x_face = jnp.where(u_face >= 0.0, phi_w, phi_e)
    flux_x = u_face * phi_x_face

    u_center = 0.5 * (u_face[1:, :] + u_face[:-1, :])
    u_y_face = jnp.zeros((nx, ny + 1), dtype=phi.dtype)
    u_y_face = u_y_face.at[:, 1:ny].set(0.5 * (u_center[:, 1:] + u_center[:, :-1]))
    u_y_face = u_y_face.at[:, 0].set(u_center[:, 0])
    u_y_face = u_y_face.at[:, ny].set(u_center[:, -1])

    f1_y_face = jnp.zeros((nx, ny + 1), dtype=phi.dtype)
    f1_y_face = f1_y_face.at[:, 1:ny].set(0.5 * (f_1_grid[:, 1:] + f_1_grid[:, :-1]))
    f1_y_face = f1_y_face.at[:, 0].set(f_1_grid[:, 0])
    f1_y_face = f1_y_face.at[:, ny].set(f_1_grid[:, -1])

    w_face = v_face - f1_y_face * u_y_face
    phi_s = jnp.concatenate([q[:, 0:1], q], axis=1)
    phi_n = jnp.concatenate([q, q[:, -1:]], axis=1)
    phi_y_face = jnp.where(w_face >= 0.0, phi_s, phi_n)
    flux_y = w_face * phi_y_face

    return (flux_x[1:, :] - flux_x[:-1, :]) / dx + (flux_y[:, 1:] - flux_y[:, :-1]) / dy


@jit
def jax_checkerboard_filter(phi, strength):
    """Conservative 4th-difference filter damping wavelength-2h grid modes.

    Flux form: q_{i+1/2} = s/16 * (phi_{i+2} - 3 phi_{i+1} + 3 phi_i
    - phi_{i-1}), phi_i <- phi_i - (q_{i+1/2} - q_{i-1/2}), applied along both
    axes with zero flux through domain faces (global phi sum is preserved
    exactly).  A pure checkerboard is damped by factor (1 - s) per
    application; resolved modes are touched at O(s (kh)^4 / 16).

    Rationale: the semi-implicit CH stepping amplifies the wavelength-2h
    interface mode by ~e^1e-3 per step regardless of dt (the "chainsaw");
    runs whose dt is pushed down by the capillary CFL take many more steps
    per unit time and the mode outruns the scheme's own damping.  The filter
    adds per-step grid-mode damping that dominates the per-step growth while
    converging away (4th order) on resolved scales.
    """
    s = jnp.asarray(strength, dtype=phi.dtype) / 16.0

    # n+1 face fluxes per axis from a 2-ghost edge-replicated pad; boundary
    # faces carry zero flux so the global phi sum is exactly preserved.
    pad_x = jnp.concatenate(
        [phi[0:1, :], phi[0:1, :], phi, phi[-1:, :], phi[-1:, :]], axis=0
    )
    qx = s * (pad_x[3:, :] - 3.0 * pad_x[2:-1, :] + 3.0 * pad_x[1:-2, :] - pad_x[:-3, :])
    qx = qx.at[0, :].set(0.0).at[-1, :].set(0.0)
    out = phi - (qx[1:, :] - qx[:-1, :])

    pad_y = jnp.concatenate(
        [out[:, 0:1], out[:, 0:1], out, out[:, -1:], out[:, -1:]], axis=1
    )
    qy = s * (pad_y[:, 3:] - 3.0 * pad_y[:, 2:-1] + 3.0 * pad_y[:, 1:-2] - pad_y[:, :-3])
    qy = qy.at[:, 0].set(0.0).at[:, -1].set(0.0)
    return out - (qy[:, 1:] - qy[:, :-1])


@jit
def jax_degenerate_mobility(phi, mobility_blend=0.0, mobility_power=1.0):
    """Degenerate CH mobility with optional constant-mobility blending.

    M(phi) = (1-b) * max(1-phi^2, 0)^p + b.

    The max(.,0) is the degenerate extension of the mobility outside the
    physical range: mobility vanishes at and beyond the pure phases.  ``b``
    (`mobility_blend` in [0,1]) is a modeling choice that adds a residual
    constant mobility; b=0 is the fully degenerate model.
    """
    power = jnp.maximum(jnp.asarray(mobility_power, dtype=phi.dtype), 1.0)
    base = jnp.maximum(1.0 - phi**2, 0.0) ** power
    blend = jnp.asarray(mobility_blend, dtype=phi.dtype)
    return (1.0 - blend) * base + blend


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
    use_degenerate_mobility=False, degenerate_mobility_blend=0.0,
    degenerate_mobility_power=1.0,
    phase_potential_code=0,
    phase_log_theta=0.25, phase_log_theta_c=1.0, phase_log_delta=1e-6,
):
    lap_phi = jax_laplacian(phi, dx, dy, f_1_grid, f_2_grid)
    mu_ch = (
        jax_free_energy_derivative(
            phi,
            phase_potential_code,
            phase_log_theta,
            phase_log_theta_c,
            phase_log_delta,
        )
        - epsilon**2 * lap_phi
    )
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
        phi,
        mobility_blend=degenerate_mobility_blend,
        mobility_power=degenerate_mobility_power,
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
    use_degenerate_mobility=False, degenerate_mobility_blend=0.0,
    degenerate_mobility_power=1.0,
    phase_potential_code=0,
    phase_log_theta=0.25, phase_log_theta_c=1.0, phase_log_delta=1e-6,
):
    if f_1_grid is None:
        f_1_grid = jnp.zeros_like(phi)
    if f_2_grid is None:
        f_2_grid = jnp.zeros_like(phi)
    lap_phi = jax_laplacian_terrain_bottom_ghost(phi, dx, dy, f_1_grid, f_2_grid, bottom_ghost_phi)
    mu_ch = (
        jax_free_energy_derivative(
            phi,
            phase_potential_code,
            phase_log_theta,
            phase_log_theta_c,
            phase_log_delta,
        )
        - epsilon**2 * lap_phi
    )
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
        phi,
        mobility_blend=degenerate_mobility_blend,
        mobility_power=degenerate_mobility_power,
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
                     use_degenerate_mobility=False, degenerate_mobility_blend=0.0,
                     degenerate_mobility_power=1.0,
                     phase_potential_code=0,
                     phase_log_theta=0.25, phase_log_theta_c=1.0, phase_log_delta=1e-6,
                     u_face=None, v_face=None, phase_flux_shift=0.0):
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
    if u_face is not None and v_face is not None:
        convective_term = jax_phase_conservative_advection(
            phi, u_face, v_face, dx, dy, f_1_grid, phase_flux_shift
        )
    diffusive_term = jax_phase_diffusive_term_simple(
        phi, dx, dy, Pe, epsilon, f_1_grid, f_2_grid,
        lambda_willmore=lambda_willmore,
        epsilon_willmore=epsilon_willmore,
        mu_top_bc=mu_top_bc, mu_bottom_bc=mu_bottom_bc,
        mu_left_bc=mu_left_bc, mu_right_bc=mu_right_bc,
        mu_top_value=mu_top_value, mu_bottom_value=mu_bottom_value,
        mu_left_value=mu_left_value, mu_right_value=mu_right_value,
        use_degenerate_mobility=use_degenerate_mobility,
        degenerate_mobility_blend=degenerate_mobility_blend,
        degenerate_mobility_power=degenerate_mobility_power,
        phase_potential_code=phase_potential_code,
        phase_log_theta=phase_log_theta,
        phase_log_theta_c=phase_log_theta_c,
        phase_log_delta=phase_log_delta,
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
                           use_degenerate_mobility=False, degenerate_mobility_blend=0.0,
                           degenerate_mobility_power=1.0,
                           phase_potential_code=0,
                           phase_log_theta=0.25, phase_log_theta_c=1.0, phase_log_delta=1e-6,
                           u_face=None, v_face=None, phase_flux_shift=0.0):
    """Phase update with a bottom ghost cell for the wetting BC."""
    if f_1_grid is None:
        f_1_grid = jnp.zeros_like(phi)
    if f_2_grid is None:
        f_2_grid = jnp.zeros_like(phi)
    convective_term = jax_phase_convective_term(phi, U, dx, dy, f_1_grid)
    if u_face is not None and v_face is not None:
        convective_term = jax_phase_conservative_advection(
            phi, u_face, v_face, dx, dy, f_1_grid, phase_flux_shift
        )
    diffusive_term = jax_phase_diffusive_term_ghost(
        phi, dx, dy, Pe, epsilon, bottom_ghost_phi, f_1_grid, f_2_grid,
        lambda_willmore=lambda_willmore,
        epsilon_willmore=epsilon_willmore,
        mu_top_bc=mu_top_bc, mu_bottom_bc=mu_bottom_bc,
        mu_left_bc=mu_left_bc, mu_right_bc=mu_right_bc,
        mu_top_value=mu_top_value, mu_bottom_value=mu_bottom_value,
        mu_left_value=mu_left_value, mu_right_value=mu_right_value,
        use_degenerate_mobility=use_degenerate_mobility,
        degenerate_mobility_blend=degenerate_mobility_blend,
        degenerate_mobility_power=degenerate_mobility_power,
        phase_potential_code=phase_potential_code,
        phase_log_theta=phase_log_theta,
        phase_log_theta_c=phase_log_theta_c,
        phase_log_delta=phase_log_delta,
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
            if "degenerate_mobility_clip" in solver_cfg or "phase_potential_clip" in solver_cfg:
                raise ValueError(
                    "degenerate_mobility_clip and phase_potential_clip were removed: "
                    "clip-based bound enforcement is non-physical. Use the "
                    "flory_huggins potential with the log_entropy convex split "
                    "for guaranteed phase bounds."
                )
            self.degenerate_mobility_blend = float(solver_cfg.get("degenerate_mobility_blend", 0.0))
            self.degenerate_mobility_power = float(solver_cfg.get("degenerate_mobility_power", 1.0))
            self.degenerate_mobility_imex = bool(solver_cfg.get("degenerate_mobility_imex", True))
            self.degenerate_mobility_imex_mref = float(solver_cfg.get("degenerate_mobility_imex_mref", 1.0))
            potential_name = str(solver_cfg.get("phase_potential", "polynomial")).lower()
            if potential_name not in ("polynomial", "flory_huggins", "log", "logarithmic"):
                raise ValueError(
                    f"Unknown phase_potential={potential_name!r}. "
                    "Supported: polynomial, flory_huggins."
                )
            self.phase_potential = potential_name
            self.phase_potential_code = 1 if potential_name in ("flory_huggins", "log", "logarithmic") else 0
            self.phase_log_theta = float(solver_cfg.get("phase_log_theta", 0.25))
            self.phase_log_theta_c = float(solver_cfg.get("phase_log_theta_c", 1.0))
            self.phase_log_delta = float(solver_cfg.get("phase_log_delta", 1e-6))
            self.phase_convex_stabilization = float(
                solver_cfg.get("phase_convex_stabilization", 0.0)
            )
            self.phase_convex_split = str(
                solver_cfg.get("phase_convex_split", "none")
            ).lower()
            if self.phase_convex_split not in ("none", "log_entropy", "polynomial_degenerate"):
                raise ValueError(
                    f"Unknown phase_convex_split={self.phase_convex_split!r}. "
                    "Supported: none, log_entropy, polynomial_degenerate."
                )
            self.phase_convex_split_maxiter = int(
                solver_cfg.get("phase_convex_split_maxiter", 8)
            )
            self.phase_convex_split_tol = float(
                solver_cfg.get("phase_convex_split_tol", 1e-8)
            )
            self.phase_convex_split_damping = float(
                solver_cfg.get("phase_convex_split_damping", 0.8)
            )
            # Treat the concave -theta_c*phi part of the FH potential
            # implicitly inside the Newton solve instead of explicitly on the
            # RHS.  The explicit (Eyre) treatment amplifies the wavelength-2h
            # interface mode each step by (1 + dt*theta_c*M*k^2/Pe); at
            # capillary-CFL-limited dt this per-step injection outruns the
            # per-step damping and chainsaws the interface.  Implicit
            # treatment moves the term into the denominator of the growth
            # factor, making the grid mode strictly damped.
            self.phase_log_implicit_concave = bool(
                solver_cfg.get("phase_log_implicit_concave", False)
            )
            # Per-step conservative damping of wavelength-2h grid modes
            # (see jax_checkerboard_filter).  0 disables.
            self.phase_checkerboard_filter = float(
                solver_cfg.get("phase_checkerboard_filter", 0.0)
            )
            if not 0.0 <= self.phase_checkerboard_filter <= 1.0:
                raise ValueError(
                    "phase_checkerboard_filter must be in [0, 1], got "
                    f"{self.phase_checkerboard_filter}"
                )
            # BDF2 + Newton stepper (phase_update_mode == "bdf2_ch").
            self.phase_bdf2_newton_maxiter = int(
                solver_cfg.get("phase_bdf2_newton_maxiter", 12)
            )
            self.phase_bdf2_newton_tol = float(
                solver_cfg.get("phase_bdf2_newton_tol", 1e-10)
            )
            phase_advection_variable = str(solver_cfg.get("phase_advection_variable", "phi"))
            if phase_advection_variable not in ("phi", "liquid_fraction"):
                raise ValueError(
                    "Unknown phase_advection_variable="
                    f"{phase_advection_variable!r}. Allowed: 'phi', 'liquid_fraction'."
                )
            self.phase_advection_variable = phase_advection_variable
            self.phase_flux_shift = 1.0 if phase_advection_variable == "liquid_fraction" else 0.0
            self.record_ghost_row_instep = bool(
                solver_cfg.get(
                    "ghost_row_diagnostics",
                    solver_cfg.get("chainsaw_diagnostics", False),
                )
            )
            self.record_phase_stage_diagnostics = bool(
                solver_cfg.get(
                    "phase_stage_diagnostics",
                    solver_cfg.get("chainsaw_diagnostics", False),
                )
            )
            phase_debug = solver_cfg.get("phase_debug", {}) or {}
            self._phase_debug_zero_advection = bool(phase_debug.get("zero_phase_advection", False))
            self._phase_debug_skip_ch_diffusion = bool(phase_debug.get("skip_ch_diffusion", False))
            self._phase_debug_freeze_wall_after_solve = bool(
                phase_debug.get("freeze_wall_after_solve", False)
            )
            self._phase_debug_freeze_wall_after_advection = bool(
                phase_debug.get("freeze_wall_after_advection", False)
            )
            self._phase_debug_freeze_wall_after_phase_bcs = bool(
                phase_debug.get("freeze_wall_after_phase_bcs", False)
            )
            self._phase_debug_freeze_wall_after_preserve = bool(
                phase_debug.get("freeze_wall_after_preserve", False)
            )
            self._phase_debug_copy_wall_from_row1_after_solve = bool(
                phase_debug.get("copy_wall_from_row1_after_solve", False)
            )
            from runtime_diagnostics.semi_implicit_contact_split import normalize_split_mode

            split_cfg = solver_cfg.get("semi_implicit_contact_split")
            if split_cfg is None:
                pf_bc = config.get("boundary_conditions", {}).get("phase_field", {})
                ghost_law = str(pf_bc.get("contact_angle_ghost_law", "analytic_gradient"))
                if (
                    solver_cfg.get("phase_update_mode") == "semi_implicit_ch"
                    and str(solver_cfg.get("phase_field_solver", "")).lower() == "ghost_cell"
                    and ghost_law == "analytic_gradient"
                ):
                    # IMEX split: sparse A_phi does not match explicit contact_delta_term.
                    split_cfg = "no_delta"
                else:
                    split_cfg = "explicit_delta"
            self.semi_implicit_contact_split = normalize_split_mode(split_cfg)
            self.semi_implicit_contact_delta_beta = float(
                solver_cfg.get("semi_implicit_contact_delta_beta", 0.5)
            )
            self.semi_implicit_contact_filter_passes = int(
                solver_cfg.get("semi_implicit_contact_filter_passes", 1)
            )
            self.semi_implicit_contact_filter_strip_rows = int(
                solver_cfg.get("semi_implicit_contact_filter_strip_rows", 3)
            )
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
            self.degenerate_mobility_blend = 0.0
            self.degenerate_mobility_power = 1.0
            self.degenerate_mobility_imex = True
            self.degenerate_mobility_imex_mref = 1.0
            self.phase_potential = "polynomial"
            self.phase_potential_code = 0
            self.phase_log_theta = 0.25
            self.phase_log_theta_c = 1.0
            self.phase_log_delta = 1e-6
            self.phase_convex_stabilization = 0.0
            self.phase_convex_split = "none"
            self.phase_convex_split_maxiter = 8
            self.phase_convex_split_tol = 1e-8
            self.phase_convex_split_damping = 0.8
            self.phase_log_implicit_concave = False
            self.phase_checkerboard_filter = 0.0
            self.phase_bdf2_newton_maxiter = 12
            self.phase_bdf2_newton_tol = 1e-10
            self.phase_advection_variable = "phi"
            self.phase_flux_shift = 0.0
            self.record_ghost_row_instep = False
            self.record_phase_stage_diagnostics = False
            self._phase_debug_zero_advection = False
            self._phase_debug_skip_ch_diffusion = False
            self._phase_debug_freeze_wall_after_solve = False
            self._phase_debug_freeze_wall_after_advection = False
            self._phase_debug_freeze_wall_after_phase_bcs = False
            self._phase_debug_freeze_wall_after_preserve = False
            self._phase_debug_copy_wall_from_row1_after_solve = False
            self.semi_implicit_contact_split = "explicit_delta"
            self.semi_implicit_contact_delta_beta = 0.5
            self.semi_implicit_contact_filter_passes = 1
            self.semi_implicit_contact_filter_strip_rows = 3

        self._phase_laplacian_cache = None
        # BDF2 history (previous phi, dt and advection term); None until the
        # first completed bdf2_ch step, which therefore runs backward Euler.
        self._bdf2_phi_prev = None
        self._bdf2_dt_prev = None
        self._bdf2_adv_prev = None
        self._last_ghost_row_instep = None
        self._phase_stage_rows: list[dict] = []
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

    def _record_bottom_ghost_instep(self, bottom_ghost_phi, phi):
        if not getattr(self, "record_ghost_row_instep", False):
            return
        from runtime_diagnostics.ghost_row_diagnostics import record_bottom_ghost_instep

        record_bottom_ghost_instep(self, bottom_ghost_phi, phi)

    def _phase_stage_reset(self) -> None:
        if getattr(self, "record_phase_stage_diagnostics", False):
            self._phase_stage_rows = []

    def _phase_stage(self, name: str, phi, extra=None) -> None:
        if not getattr(self, "record_phase_stage_diagnostics", False):
            return
        from runtime_diagnostics.phase_update_stage_diagnostics import (
            wall_alt_contact_stats,
            wall_alt_stats,
        )

        row = wall_alt_stats(name, phi, extra=extra)
        row.update(wall_alt_contact_stats(name, phi))
        self._phase_stage_rows.append(row)

    def _phase_stage_ghost(self, name: str, bottom_ghost_phi, phi) -> None:
        if not getattr(self, "record_phase_stage_diagnostics", False):
            return
        from runtime_diagnostics.phase_update_stage_diagnostics import diag_ghost_stats

        self._phase_stage_rows.append(diag_ghost_stats(name, bottom_ghost_phi, phi))

    def _freeze_wall_row_numpy(self, phi_new_np, phi_np) -> None:
        phi_new_np[:, 0] = phi_np[:, 0]

    def _explicit_ghost_phase_step(
        self, phi, U, current_dt, dx, dy, geometry, psi=None, *, skip_bottom_advection: bool,
        u_face=None, v_face=None
    ):
        """Fully explicit ghost-cell CH update (bypasses semi-implicit Helmholtz)."""
        bottom_velocity_bc = self.config.get("boundary_conditions", {}).get("velocity", {}).get(
            "bottom", "no_slip"
        )
        U_phase = jnp.zeros_like(U) if self._phase_debug_zero_advection else U
        bottom_ghost_phi = self.bc_manager.contact_angle_bc.build_bottom_ghost_row_jax(
            phi, dx, dy, geometry, psi=psi, U=U, bottom_velocity_bc=bottom_velocity_bc
        )
        self._record_bottom_ghost_instep(bottom_ghost_phi, phi)
        self._phase_stage("01_after_build_bottom_ghost_input_phi", phi)
        self._phase_stage_ghost("01_bottom_ghost", bottom_ghost_phi, phi)
        phi_new = jax_update_phase_ghost(
            phi,
            U_phase,
            current_dt,
            dx,
            dy,
            self.Pe,
            self.epsilon,
            self.contact_angle,
            bottom_ghost_phi,
            geometry.f_1_grid,
            geometry.f_2_grid,
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
            degenerate_mobility_blend=self.degenerate_mobility_blend,
            degenerate_mobility_power=self.degenerate_mobility_power,
            phase_potential_code=self.phase_potential_code,
            phase_log_theta=self.phase_log_theta,
            phase_log_theta_c=self.phase_log_theta_c,
            phase_log_delta=self.phase_log_delta,
            u_face=u_face,
            v_face=v_face,
            phase_flux_shift=self.phase_flux_shift,
        )
        self._phase_stage("03_after_explicit_update", phi_new)
        phi_new = self._apply_advection_bcs(
            phi_new, U, current_dt, dx, dy, geometry, skip_bottom=skip_bottom_advection
        )
        self._phase_stage("05_after_advection_bcs", phi_new)
        phi_new = self._apply_phase_bcs_only(phi_new, dx, dy, geometry, psi=psi, U=U)
        self._phase_stage("06_after_phase_bcs", phi_new)
        phi_new = self._preserve_phase_sum_if_requested(phi, phi_new)
        self._phase_stage("07_after_preserve_phase_sum", phi_new)
        return phi_new

    def _build_semi_implicit_bottom_operators(
        self, phi, dx, dy, geometry, psi, U, bottom_velocity_bc, A
    ):
        """Return (A_phi, contact_laplacian_old, bottom_ghost_phi) for bottom ghost-cell CH."""
        split = self.semi_implicit_contact_split
        ca_bc = self.bc_manager.contact_angle_bc
        use_wall_energy = (
            split == "implicit_wall_energy" or ca_bc.contact_angle_ghost_law_code == 1
        )

        bottom_ghost_phi = ca_bc.build_bottom_ghost_row_jax(
            phi, dx, dy, geometry, psi=psi, U=U, bottom_velocity_bc=bottom_velocity_bc
        )
        self._record_bottom_ghost_instep(bottom_ghost_phi, phi)
        self._phase_stage("01_after_build_bottom_ghost_input_phi", phi)
        self._phase_stage_ghost("01_bottom_ghost", bottom_ghost_phi, phi)

        contact_laplacian_old = None
        if use_wall_energy:
            bottom_q_prime = np.asarray(
                ca_bc.build_bottom_wall_energy_q_prime_jax(
                    phi, dx, dy, geometry, psi=psi, U=U, bottom_velocity_bc=bottom_velocity_bc
                )
            )
            bottom_q_linear = self._bottom_wall_energy_q_linear_matrix(
                bottom_q_prime,
                dx,
                geometry,
                getattr(ca_bc, "contact_angle_wall_tangent_regularization", 0.0),
            )
            A_phi = self._phase_laplacian_with_wall_energy_bottom(
                A, bottom_q_linear, dx, dy, geometry
            )
            contact_laplacian_old = np.asarray(
                jax_laplacian_terrain_bottom_ghost(
                    phi, dx, dy, geometry.f_1_grid, geometry.f_2_grid, bottom_ghost_phi
                )
            )
        elif split == "no_delta":
            A_phi = A
        else:
            contact_laplacian_old = np.asarray(
                jax_laplacian_terrain_bottom_ghost(
                    phi, dx, dy, geometry.f_1_grid, geometry.f_2_grid, bottom_ghost_phi
                )
            )
            A_phi = A
        return A_phi, contact_laplacian_old, bottom_ghost_phi

    def _apply_contact_delta_to_rhs(
        self, phi_np, rhs, A_phi, contact_laplacian_old, current_dt, dx, dy, geometry,
        outer_matrix=None,
    ):
        """Add the affine part of the wetting-ghost relation to the RHS.

        The linearized ghost Laplacian A_phi only carries q'(phi_old)*phi_new;
        the remaining affine residual r = L_ghost(phi_old) - A_phi phi_old
        (which contains q(phi_old) - q'(phi_old) phi_old) must be forced
        explicitly, otherwise the implicit wall flux has the wrong sign and
        magnitude in near-pure phases.  ``outer_matrix`` selects the mobility
        operator the residual diffuses through (frozen degenerate mobility for
        the convex-split paths, constant mobility otherwise).
        """
        if contact_laplacian_old is None:
            return rhs, None, None

        base_laplacian_old = self._apply_sparse_operator(A_phi, phi_np)
        contact_delta = contact_laplacian_old - base_laplacian_old
        split = self.semi_implicit_contact_split

        if split == "filtered_delta":
            from runtime_diagnostics.semi_implicit_contact_split import lowpass_x_bottom_strip

            contact_delta = lowpass_x_bottom_strip(
                contact_delta,
                strip_rows=self.semi_implicit_contact_filter_strip_rows,
                passes=self.semi_implicit_contact_filter_passes,
            )

        if outer_matrix is not None:
            contact_delta_term = self._apply_sparse_operator(outer_matrix, contact_delta)
        else:
            contact_delta_term = self._apply_conservative_outer_operator(
                contact_delta, dx, dy, geometry
            )
        if split == "damped_delta":
            from runtime_diagnostics.semi_implicit_contact_split import damp_bottom_rows

            contact_delta_term = damp_bottom_rows(
                contact_delta_term, self.semi_implicit_contact_delta_beta, n_rows=2
            )

        ch_scale = current_dt * (self.epsilon ** 2 / self.Pe)
        rhs = rhs - ch_scale * contact_delta_term
        return rhs, contact_delta, contact_delta_term

    def _preserve_phase_sum_if_requested(self, phi_old, phi_new):
        """Restore the global phi sum changed by wall/BC corrections.

        The compensation is distributed over the interface band, weighted by
        w = (1 - phi^2)_+ , instead of a uniform shift.  A uniform shift dumps
        the missing mass into the bulk phases, where degenerate mobility
        (M ~ (1-phi^2)^p ~ 0) cannot relax it; over thousands of steps the gas
        bulk creeps from the binodal to the log-potential clamp, mu develops a
        spurious O(1) excursion across the interface tails, and the potential
        capillary force F = lambda*mu*grad(phi) turns that into grid-scale
        forcing (the "chainsawed interface").  Weighting by (1-phi^2) puts the
        correction where mobility is finite so CH dynamics absorb it.

        If there is no interface (sum of weights ~ 0; never the case in
        droplet runs) the weights would blow up; we then fall back to the
        legacy uniform shift, which is exact for a pure single-phase field.
        """
        if self.bc_manager is None:
            return phi_new
        contact_angle_bc = getattr(self.bc_manager, "contact_angle_bc", None)
        if contact_angle_bc is None or not getattr(contact_angle_bc, "conserve_phi_sum", False):
            return phi_new
        delta_mean = jnp.mean(phi_old) - jnp.mean(phi_new)
        weights = jnp.maximum(1.0 - phi_new**2, 0.0)
        weights_mean = jnp.mean(weights)
        return jnp.where(
            weights_mean > 1e-12,
            phi_new + delta_mean * weights / jnp.maximum(weights_mean, 1e-12),
            phi_new + delta_mean,
        )

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
            backend="scipy",
            f_1_grid=f_1_grid,
            f_2_grid=f_2_grid,
            solver_params={
                "accel": "bicgstab",
                "tol": self.phase_diffusion_solver_tol,
                "maxiter": self.phase_diffusion_solver_maxiter,
                "terrain_laplacian_mode": "jax_derivative_composition",
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

    def _get_frozen_mobility_outer_matrix(self, mobility_np, dx, dy, geometry):
        """Conservative sparse div(M^n grad(.)) using face-averaged mobility."""
        import scipy.sparse

        mobility = np.asarray(mobility_np, dtype=np.float64)
        nx, ny = mobility.shape
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
        mobility_xface = 0.5 * (mobility[1:, :] + mobility[:-1, :])
        mobility_yface = 0.5 * (mobility[:, 1:] + mobility[:, :-1])

        diag_x = scipy.sparse.diags(f_1_xface.T.reshape(-1), format="csr")
        diag_y = scipy.sparse.diags(f_1_yface.T.reshape(-1), format="csr")
        diag_metric_y = scipy.sparse.diags((1.0 + f_1_yface**2).T.reshape(-1), format="csr")
        diag_mx = scipy.sparse.diags(mobility_xface.T.reshape(-1), format="csr")
        diag_my = scipy.sparse.diags(mobility_yface.T.reshape(-1), format="csr")

        flux_x_op = diag_mx @ (gx - diag_x @ (ax @ dy_cell))
        flux_y_op = diag_my @ (-diag_y @ (ay @ dx_cell) + diag_metric_y @ gy)
        return (divx @ flux_x_op + divy @ flux_y_op).tocsr()

    def _apply_conservative_outer_operator(self, field_np, dx, dy, geometry):
        matrix = self._get_conservative_outer_matrix(
            field_np.shape[0], field_np.shape[1], dx, dy, geometry
        )
        return self._apply_sparse_operator(matrix, field_np)

    def _bottom_wall_energy_q_linear_matrix(self, bottom_q_prime, dx, geometry, tangent_beta):
        import scipy.sparse

        q_prime = np.asarray(bottom_q_prime, dtype=np.float64)
        nx = q_prime.shape[0]
        q_linear = scipy.sparse.diags(q_prime, format="csr")
        beta = float(tangent_beta)
        if beta == 0.0:
            return q_linear

        if geometry is not None:
            metric = 1.0 + np.asarray(geometry.f_1_grid[:, 0], dtype=np.float64) ** 2
        else:
            metric = np.ones(nx, dtype=np.float64)
        metric_face = 0.5 * (metric[1:] + metric[:-1])
        surface_laplacian = (
            scipy.sparse.diags(1.0 / np.sqrt(np.maximum(metric, 1e-12)), format="csr")
            @
            self._create_face_divergence_matrix(nx, dx)
            @ scipy.sparse.diags(
                1.0 / np.sqrt(np.maximum(metric_face, 1e-12)), format="csr"
            )
            @ self._create_face_gradient_matrix(nx, dx)
        ).tocsr()
        return (q_linear + beta * surface_laplacian).tocsr()

    def _phase_laplacian_with_wall_energy_bottom(self, A_mu, bottom_q_linear, dx, dy, geometry=None):
        """Linearized phase Laplacian with full-wall wetting at the bottom.

        The explicit ghost residual uses
            phi_g = phi_1 - 2 dy q(phi_0)
        where q is the wall-free-energy normal derivative.  For the implicit
        stabilizer we freeze dq/dphi at the old state.  The affine part is
        supplied separately as ``L_ghost(phi_old) - A_phi phi_old`` on the RHS.

        On terrain grids q is a physical wall-normal derivative, while the
        ghost row is placed in computational eta.  The linearized relation is

            phi_eta = (sqrt(1 + h_x^2) q'(phi_0) phi_0 + h_x phi_x) / (1 + h_x^2)

        and this must enter both phi_eta_eta and phi_x_eta in the bottom
        Laplacian row.  A diagonal-only correction is only valid for flat walls.
        """
        q_linear = bottom_q_linear.tocsr()
        terrain_sig = (
            self._terrain_cache_signature(geometry)
            if geometry is not None
            else (False, 0.0, 0.0, 0.0, 0.0)
        )
        if terrain_sig[0]:
            return self._phase_laplacian_with_terrain_wall_energy_bottom(
                A_mu, q_linear, dx, dy, geometry
            )

        A_phi = A_mu.tolil(copy=True)
        nx = q_linear.shape[0]
        bottom_correction = (-2.0 / dy) * q_linear
        for i in range(nx):
            row = bottom_correction.getrow(i)
            if row.nnz:
                for col, value in zip(row.indices, row.data):
                    A_phi[i, col] = A_phi[i, col] + value
        return A_phi.tocsr()

    def _phase_laplacian_with_terrain_wall_energy_bottom(
        self, A_mu, bottom_q_linear, dx, dy, geometry
    ):
        """Replace bottom rows by the terrain-aware linearized ghost Laplacian.

        This is the sparse analogue of ``jax_laplacian_terrain_bottom_ghost``
        with the wall-energy ghost relation linearized at the previous phase
        field.  It avoids the unstable explicit ``C(contact_delta)`` forcing and
        keeps the terrain metric terms in the implicit CH operator.
        """
        import scipy.sparse

        q_linear = bottom_q_linear.tocsr()
        f_1_surface = np.asarray(geometry.f_1_grid[:, 0], dtype=np.float64)
        f_2_surface = np.asarray(geometry.f_2_grid[:, 0], dtype=np.float64)
        nx = q_linear.shape[0]
        ny = A_mu.shape[0] // nx
        metric = 1.0 + f_1_surface**2

        dx_matrix = self._create_first_derivative_matrix(nx, dx).tocsr()
        dxx_matrix = dx_matrix @ dx_matrix

        eta_derivative_matrix = (
            scipy.sparse.diags(1.0 / np.sqrt(metric), format="csr") @ q_linear
            + scipy.sparse.diags(f_1_surface / metric, format="csr") @ dx_matrix
        ).tocsr()
        eta_eta_row0 = (
            scipy.sparse.diags(np.full(nx, -2.0 / (dy * dy)), format="csr")
            - (2.0 / dy) * eta_derivative_matrix
        ).tocsr()
        eta_eta_row1 = scipy.sparse.diags(np.full(nx, 2.0 / (dy * dy)), format="csr")

        bottom_row0 = (
            dxx_matrix
            - 2.0 * scipy.sparse.diags(f_1_surface, format="csr")
            @ (dx_matrix @ eta_derivative_matrix)
            - scipy.sparse.diags(f_2_surface, format="csr") @ eta_derivative_matrix
            + scipy.sparse.diags(metric, format="csr") @ eta_eta_row0
        ).tocsr()
        bottom_row1 = (scipy.sparse.diags(metric, format="csr") @ eta_eta_row1).tocsr()

        A_phi = A_mu.tolil(copy=True)
        for i in range(nx):
            A_phi[i, :] = 0.0
            row0 = bottom_row0.getrow(i)
            row1 = bottom_row1.getrow(i)
            if row0.nnz:
                A_phi.rows[i].extend(row0.indices.tolist())
                A_phi.data[i].extend(row0.data.tolist())
            if row1.nnz and ny > 1:
                A_phi.rows[i].extend((row1.indices + nx).tolist())
                A_phi.data[i].extend(row1.data.tolist())
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
            # Escalate to a direct solve; if that also fails, fail loudly
            # instead of silently corrupting the phase field.
            sol = scipy.sparse.linalg.spsolve(M.tocsc(), rhs_flat)
        if not np.all(np.isfinite(sol)):
            raise RuntimeError(
                "Phase Helmholtz/biharmonic solve produced non-finite values "
                f"(bicgstab info={info}); aborting step instead of falling back."
            )
        return sol.reshape(rhs_t.shape).T

    def _solve_log_entropy_convex_split(
        self,
        rhs_np,
        x0_np,
        C,
        A_phi,
        alpha,
        gamma,
    ):
        """Solve one constant-mobility logarithmic convex-splitting CH step.

        The split is

            f'(phi) = f_c'(phi) - theta_c phi,
            f_c'(phi) = theta/2 log((1 + phi) / (1 - phi)).

        The convex entropy derivative and interfacial Laplacian are implicit.
        With ``phase_log_implicit_concave`` disabled (legacy Eyre split) the
        concave ``-theta_c phi`` part is already included in ``rhs_np``; with
        it enabled the term is solved implicitly here and must NOT be in the
        RHS (see __init__ for why the explicit treatment chainsaws the
        interface at small dt).
        """
        import scipy.sparse
        import scipy.sparse.linalg

        nx, ny = rhs_np.shape
        identity = scipy.sparse.identity(nx * ny, format="csr")
        linear_operator = C @ A_phi
        delta = max(float(self.phase_log_delta), 1e-12)
        theta = float(self.phase_log_theta)
        theta_c_implicit = (
            float(self.phase_log_theta_c) if self.phase_log_implicit_concave else 0.0
        )
        damping = min(max(float(self.phase_convex_split_damping), 1e-3), 1.0)
        x = np.clip(np.asarray(x0_np, dtype=np.float64), -1.0 + delta, 1.0 - delta)
        rhs_flat = np.asarray(rhs_np.T, dtype=np.float64).flatten()

        def flatten(field):
            return np.asarray(field.T, dtype=np.float64).flatten()

        def unflatten(vector):
            return np.asarray(vector).reshape((ny, nx)).T

        for _ in range(max(int(self.phase_convex_split_maxiter), 1)):
            x = np.clip(x, -1.0 + delta, 1.0 - delta)
            fc_prime = (
                0.5 * theta * np.log((1.0 + x) / (1.0 - x)) - theta_c_implicit * x
            )
            residual = (
                flatten(x)
                + alpha * linear_operator.dot(flatten(x))
                - gamma * C.dot(flatten(fc_prime))
                - rhs_flat
            )
            residual_norm = np.linalg.norm(residual) / max(np.sqrt(residual.size), 1.0)
            if residual_norm < self.phase_convex_split_tol:
                break

            fc_second = theta / np.maximum(1.0 - x * x, delta) - theta_c_implicit
            jacobian = (
                identity
                + alpha * linear_operator
                - gamma * (C @ scipy.sparse.diags(flatten(fc_second), format="csr"))
            ).tocsr()
            update, info = scipy.sparse.linalg.bicgstab(
                jacobian,
                -residual,
                rtol=min(self.phase_diffusion_solver_tol, 1e-8),
                maxiter=self.phase_diffusion_solver_maxiter,
            )
            if info != 0 or not np.all(np.isfinite(update)):
                update = scipy.sparse.linalg.spsolve(jacobian.tocsc(), -residual)
            if not np.all(np.isfinite(update)):
                raise RuntimeError(
                    "log-entropy convex split Newton update is non-finite "
                    f"(bicgstab info={info}); aborting step."
                )
            x = x + damping * unflatten(update)

        # Interior projection: part of the projected-Newton algorithm for the
        # singular entropy term, not a physics clip. Converged solutions are
        # interior because mu_log diverges at +-1, so the projection is
        # inactive at convergence.
        return np.clip(x, -1.0 + delta, 1.0 - delta)

    def _bdf2_ch_step(
        self, phi, U, current_dt, dx, dy, geometry, psi=None,
        skip_bottom=False, u_face=None, v_face=None,
    ):
        """Variable-step BDF2 + Newton step for the Cahn-Hilliard update.

        Fully implicit in the CH diffusion: mobility, chemical potential and
        the interfacial Laplacian are all evaluated at phi^{n+1} (no Eyre
        split, no frozen mobility in the residual).  Advection is explicit
        with second-order extrapolation (SBDF2), matching how the phase
        solver couples to the flow elsewhere.  The first step (no history)
        is backward Euler.

        Rationale: the first-order semi-implicit stepper ratchets a
        wavelength-2h staircase into slowly drifting FH interfaces (the
        "chainsaw"); a second-order integrator tracks the drifting stiff
        front with O(dt^2) per-step error instead of O(dt).

        Newton uses the exact residual; the Jacobian refreshes mobility and
        f'' at every iterate but omits the dM/dphi rank term (chord-type
        approximation affecting convergence speed only, not the solution).
        """
        import scipy.sparse
        import scipy.sparse.linalg

        phi_np = np.asarray(phi, dtype=np.float64)
        nx, ny = phi_np.shape
        if self._phase_debug_zero_advection:
            adv_n = np.zeros_like(phi_np)
        elif u_face is not None and v_face is not None:
            adv_n = np.asarray(jax_phase_conservative_advection(
                phi, u_face, v_face, dx, dy, geometry.f_1_grid, self.phase_flux_shift
            ))
        else:
            adv_n = np.asarray(jax_phase_convective_term(
                phi, U, dx, dy, geometry.f_1_grid
            ))

        have_history = (
            self._bdf2_phi_prev is not None
            and self._bdf2_phi_prev.shape == phi_np.shape
            and self._bdf2_dt_prev is not None
        )
        if have_history:
            rho = float(current_dt) / float(self._bdf2_dt_prev)
            a0 = (1.0 + 2.0 * rho) / (1.0 + rho)
            a1 = -(1.0 + rho)
            a2 = rho**2 / (1.0 + rho)
            adv_star = (1.0 + rho) * adv_n - rho * self._bdf2_adv_prev
            history_term = a1 * phi_np + a2 * self._bdf2_phi_prev
        else:
            a0, a1, a2 = 1.0, -1.0, 0.0
            adv_star = adv_n
            history_term = a1 * phi_np

        A = self._get_phase_laplacian_matrix(nx, ny, dx, dy, geometry=geometry)
        identity = scipy.sparse.identity(nx * ny, format="csr")
        delta = max(float(self.phase_log_delta), 1e-12)
        theta = float(self.phase_log_theta)
        theta_c = float(self.phase_log_theta_c)
        kappa = float(current_dt) / self.Pe
        eps2 = self.epsilon**2

        def flatten(field):
            return np.asarray(field.T, dtype=np.float64).flatten()

        def unflatten(vector):
            return np.asarray(vector).reshape((ny, nx)).T

        # FH potential with C^1 linear extension beyond the log clamp: the
        # state itself is unconstrained (advection/BCs can push bulk cells
        # past |phi| = 1-delta, as in the semi-implicit path); a saturated
        # (C^0) clamp there makes residual and Jacobian inconsistent and
        # stalls Newton at the saturated cells.
        def f_prime(x):
            if self.phase_potential_code == 1:
                xc = np.clip(x, -1.0 + delta, 1.0 - delta)
                fp = 0.5 * theta * np.log((1.0 + xc) / (1.0 - xc)) - theta_c * xc
                return fp + (theta / (1.0 - xc * xc) - theta_c) * (x - xc)
            return x**3 - x

        def f_second(x):
            if self.phase_potential_code == 1:
                xc = np.clip(x, -1.0 + delta, 1.0 - delta)
                return theta / (1.0 - xc * xc) - theta_c
            return 3.0 * x**2 - 1.0

        def mobility_of(x):
            if not self.use_degenerate_mobility:
                return np.ones_like(x)
            return np.asarray(jax_degenerate_mobility(
                jnp.asarray(x),
                mobility_blend=self.degenerate_mobility_blend,
                mobility_power=self.degenerate_mobility_power,
            ))

        rhs_const_flat = flatten(history_term + current_dt * adv_star)
        x = phi_np.copy()
        converged = False
        residual_history = []
        for _ in range(max(int(self.phase_bdf2_newton_maxiter), 1)):
            C = self._get_frozen_mobility_outer_matrix(mobility_of(x), dx, dy, geometry)
            mu_flat = flatten(f_prime(x)) - eps2 * A.dot(flatten(x))
            residual = a0 * flatten(x) + rhs_const_flat - kappa * C.dot(mu_flat)
            residual_norm = np.linalg.norm(residual) / np.sqrt(residual.size)
            residual_history.append(residual_norm)
            if residual_norm < self.phase_bdf2_newton_tol:
                converged = True
                break
            jacobian = (
                a0 * identity
                - kappa * (C @ scipy.sparse.diags(flatten(f_second(x)), format="csr"))
                + kappa * eps2 * (C @ A)
            ).tocsr()
            update, info = scipy.sparse.linalg.bicgstab(
                jacobian, -residual,
                rtol=min(self.phase_diffusion_solver_tol, 1e-10),
                maxiter=self.phase_diffusion_solver_maxiter,
            )
            if info != 0 or not np.all(np.isfinite(update)):
                update = scipy.sparse.linalg.spsolve(jacobian.tocsc(), -residual)
            if not np.all(np.isfinite(update)):
                raise RuntimeError(
                    f"bdf2_ch Newton update is non-finite (bicgstab info={info})"
                )
            residual_history[-1] = (
                residual_norm, float(np.max(np.abs(update))),
                int(np.argmax(np.abs(residual)))
            )
            x = x + unflatten(update)
        if not converged:
            history = " | ".join(
                f"R={r[0]:.2e} dmax={r[1]:.2e} argmax={r[2]}"
                if isinstance(r, tuple) else f"R={r:.2e}"
                for r in residual_history
            )
            raise RuntimeError(
                "bdf2_ch Newton did not converge: [{}] > tol {:.1e} "
                "after {} iterations".format(
                    history, self.phase_bdf2_newton_tol,
                    self.phase_bdf2_newton_maxiter,
                )
            )

        # this call's input becomes phi^{n-1} of the next call
        self._bdf2_phi_prev = phi_np
        self._bdf2_dt_prev = float(current_dt)
        self._bdf2_adv_prev = adv_n

        phi_new = jnp.asarray(x, dtype=phi.dtype)
        if self.phase_checkerboard_filter > 0.0:
            phi_new = jax_checkerboard_filter(phi_new, self.phase_checkerboard_filter)
        skip_bottom = bool(skip_bottom or self._should_skip_bottom_advection_bc())
        phi_new = self._apply_advection_bcs(
            phi_new, U, current_dt, dx, dy, geometry, skip_bottom=skip_bottom
        )
        phi_new = self._apply_phase_bcs_only(phi_new, dx, dy, geometry, psi=psi, U=U)
        phi_new = self._preserve_phase_sum_if_requested(phi, phi_new)
        return phi_new

    def _solve_polynomial_degenerate_convex_split(
        self,
        rhs_np,
        x0_np,
        C_mob,
        A_phi,
        gamma,
    ):
        """Solve polynomial double-well convex split with frozen degenerate mobility.

        Uses the standard split

            f(phi) = 1/4 phi^4 - 1/2 phi^2 + const
            f_c'(phi^{n+1}) = (phi^{n+1})^3
            f_e'(phi^n) = phi^n.

        The mobility matrix is frozen at ``phi^n`` and is conservative, so the
        update preserves total phase to linear-solver tolerance.
        """
        import scipy.sparse
        import scipy.sparse.linalg

        nx, ny = rhs_np.shape
        identity = scipy.sparse.identity(nx * ny, format="csr")
        linear_operator = C_mob @ A_phi
        alpha = gamma * (self.epsilon ** 2)
        damping = min(max(float(self.phase_convex_split_damping), 1e-3), 1.0)
        x = np.asarray(x0_np, dtype=np.float64).copy()
        rhs_flat = np.asarray(rhs_np.T, dtype=np.float64).flatten()

        def flatten(field):
            return np.asarray(field.T, dtype=np.float64).flatten()

        def unflatten(vector):
            return np.asarray(vector).reshape((ny, nx)).T

        for _ in range(max(int(self.phase_convex_split_maxiter), 1)):
            x_flat = flatten(x)
            cubic = x * x * x
            residual = (
                x_flat
                + alpha * linear_operator.dot(x_flat)
                - gamma * C_mob.dot(flatten(cubic))
                - rhs_flat
            )
            residual_norm = np.linalg.norm(residual) / max(np.sqrt(residual.size), 1.0)
            if residual_norm < self.phase_convex_split_tol:
                break
            cubic_prime = 3.0 * x * x
            jacobian = (
                identity
                + alpha * linear_operator
                - gamma * (C_mob @ scipy.sparse.diags(flatten(cubic_prime), format="csr"))
            ).tocsr()
            update, info = scipy.sparse.linalg.bicgstab(
                jacobian,
                -residual,
                rtol=min(self.phase_diffusion_solver_tol, 1e-8),
                maxiter=self.phase_diffusion_solver_maxiter,
            )
            if info != 0 or not np.all(np.isfinite(update)):
                update = scipy.sparse.linalg.spsolve(jacobian.tocsc(), -residual)
            if not np.all(np.isfinite(update)):
                raise RuntimeError(
                    "polynomial-degenerate convex split Newton update is "
                    f"non-finite (bicgstab info={info}); aborting step."
                )
            x = x + damping * unflatten(update)

        return x

    def _semi_implicit_ch_step(
        self, phi, U, current_dt, dx, dy, geometry, psi=None, skip_bottom=False,
        u_face=None, v_face=None
    ):
        self._phase_stage_reset()
        self._phase_stage("00_input_phi", phi)

        if skip_bottom and self.semi_implicit_contact_split == "explicit_ghost":
            skip_bottom_adv = bool(skip_bottom or self._should_skip_bottom_advection_bc())
            return self._explicit_ghost_phase_step(
                phi, U, current_dt, dx, dy, geometry, psi=psi,
                skip_bottom_advection=skip_bottom_adv, u_face=u_face, v_face=v_face
            )

        U_phase = jnp.zeros_like(U) if self._phase_debug_zero_advection else U

        # Keep a compatibility fallback to the legacy explicit variable-mobility path.
        if self.use_degenerate_mobility and not self.degenerate_mobility_imex:
            if skip_bottom:
                bottom_velocity_bc = self.config.get("boundary_conditions", {}).get("velocity", {}).get("bottom", "no_slip")
                bottom_ghost_phi = self.bc_manager.contact_angle_bc.build_bottom_ghost_row_jax(
                    phi, dx, dy, geometry, psi=psi, U=U, bottom_velocity_bc=bottom_velocity_bc
                )
                self._record_bottom_ghost_instep(bottom_ghost_phi, phi)
                phi_new = jax_update_phase_ghost(
                    phi, U_phase, current_dt, dx, dy,
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
                    degenerate_mobility_blend=self.degenerate_mobility_blend,
                    degenerate_mobility_power=self.degenerate_mobility_power,
                    phase_potential_code=self.phase_potential_code,
                    phase_log_theta=self.phase_log_theta,
                    phase_log_theta_c=self.phase_log_theta_c,
                    phase_log_delta=self.phase_log_delta,
                    u_face=u_face,
                    v_face=v_face,
                    phase_flux_shift=self.phase_flux_shift,
                )
            else:
                phi_new = jax_update_phase(
                    phi, U_phase, current_dt, dx, dy,
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
                    degenerate_mobility_blend=self.degenerate_mobility_blend,
                    degenerate_mobility_power=self.degenerate_mobility_power,
                    phase_potential_code=self.phase_potential_code,
                    phase_log_theta=self.phase_log_theta,
                    phase_log_theta_c=self.phase_log_theta_c,
                    phase_log_delta=self.phase_log_delta,
                    u_face=u_face,
                    v_face=v_face,
                    phase_flux_shift=self.phase_flux_shift,
                )
            skip_bottom = bool(skip_bottom or self._should_skip_bottom_advection_bc())
            phi_new = self._apply_advection_bcs(phi_new, U, current_dt, dx, dy, geometry, skip_bottom=skip_bottom)
            phi_new = self._apply_phase_bcs_only(phi_new, dx, dy, geometry, psi=psi, U=U)
            phi_new = self._preserve_phase_sum_if_requested(phi, phi_new)
            return phi_new

        phi_np = np.asarray(phi)
        if self._phase_debug_zero_advection:
            convective_term = np.zeros_like(phi_np)
        else:
            if u_face is not None and v_face is not None:
                convective_term = np.asarray(
                    jax_phase_conservative_advection(
                        phi, u_face, v_face, dx, dy, geometry.f_1_grid, self.phase_flux_shift
                    )
                )
            else:
                convective_term = np.asarray(
                    jax_phase_convective_term(phi, U_phase, dx, dy, geometry.f_1_grid)
                )
        self._phase_stage("op_convective_term", convective_term)
        nx, ny = phi_np.shape
        A = self._get_phase_laplacian_matrix(nx, ny, dx, dy, geometry=geometry)
        C = self._get_conservative_outer_matrix(nx, ny, dx, dy, geometry)
        A_phi = A
        bottom_velocity_bc = self.config.get("boundary_conditions", {}).get("velocity", {}).get("bottom", "no_slip")
        bottom_ghost_phi = None
        contact_laplacian_old = None
        if skip_bottom:
            A_phi, contact_laplacian_old, bottom_ghost_phi = self._build_semi_implicit_bottom_operators(
                phi, dx, dy, geometry, psi, U, bottom_velocity_bc, A
            )
        phase_linear_operator = C @ A_phi
        mobility_ref = 1.0
        if self.use_degenerate_mobility:
            mobility_ref = max(float(self.degenerate_mobility_imex_mref), 0.0)
        alpha = current_dt * mobility_ref * (self.epsilon ** 2) / self.Pe
        gamma = current_dt / self.Pe
        beta_stabilization = 0.0
        use_true_convex_split = (
            not self.use_degenerate_mobility
            and self.phase_convex_split == "log_entropy"
            and self.phase_potential_code == 1
        )
        use_log_degenerate_convex_split = (
            self.use_degenerate_mobility
            and self.phase_convex_split == "log_entropy"
            and self.phase_potential_code == 1
        )
        use_poly_degenerate_convex_split = (
            self.use_degenerate_mobility
            and self.phase_convex_split == "polynomial_degenerate"
            and self.phase_potential_code == 0
        )
        C_mob = None
        if use_poly_degenerate_convex_split or use_log_degenerate_convex_split:
            mobility_old = np.asarray(
                jax_degenerate_mobility(
                    phi,
                    mobility_blend=self.degenerate_mobility_blend,
                    mobility_power=self.degenerate_mobility_power,
                )
            )
            C_mob = self._get_frozen_mobility_outer_matrix(mobility_old, dx, dy, geometry)
        if not self.use_degenerate_mobility and not use_true_convex_split:
            beta_stabilization = current_dt * max(self.phase_convex_stabilization, 0.0) / self.Pe
        M = None
        if alpha > 0.0 or beta_stabilization > 0.0:
            cache_key = (
                nx,
                ny,
                float(dx),
                float(dy),
                self._terrain_cache_signature(geometry),
                float(alpha),
                float(beta_stabilization),
            )
            can_cache_helmholtz = A_phi is A
            if can_cache_helmholtz and cache_key in self._phase_helmholtz_cache:
                M = self._phase_helmholtz_cache[cache_key]
            else:
                import scipy.sparse
                M = (
                    scipy.sparse.identity(A.shape[0], format="csr")
                    + alpha * phase_linear_operator
                    - beta_stabilization * C
                )
                if can_cache_helmholtz:
                    self._phase_helmholtz_cache[cache_key] = M
        if self.use_degenerate_mobility:
            if use_poly_degenerate_convex_split:
                nonlinear_term = -self._apply_sparse_operator(C_mob, phi_np)
                rhs = phi_np - current_dt * convective_term + gamma * nonlinear_term
                rhs, contact_delta, contact_delta_term = self._apply_contact_delta_to_rhs(
                    phi_np, rhs, A_phi, contact_laplacian_old, current_dt, dx, dy,
                    geometry, outer_matrix=C_mob,
                )
            elif use_log_degenerate_convex_split:
                # Concave part f_e'(phi) = -theta_c phi: explicit (Eyre) through
                # the frozen-mobility operator, or implicit inside the Newton
                # solve when phase_log_implicit_concave is enabled.
                if self.phase_log_implicit_concave:
                    rhs = phi_np - current_dt * convective_term
                else:
                    nonlinear_term = -self.phase_log_theta_c * self._apply_sparse_operator(
                        C_mob, phi_np
                    )
                    rhs = phi_np - current_dt * convective_term + gamma * nonlinear_term
                rhs, contact_delta, contact_delta_term = self._apply_contact_delta_to_rhs(
                    phi_np, rhs, A_phi, contact_laplacian_old, current_dt, dx, dy,
                    geometry, outer_matrix=C_mob,
                )
            elif skip_bottom:
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
                        degenerate_mobility_blend=self.degenerate_mobility_blend,
                        degenerate_mobility_power=self.degenerate_mobility_power,
                        phase_potential_code=self.phase_potential_code,
                        phase_log_theta=self.phase_log_theta,
                        phase_log_theta_c=self.phase_log_theta_c,
                        phase_log_delta=self.phase_log_delta,
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
                        degenerate_mobility_blend=self.degenerate_mobility_blend,
                        degenerate_mobility_power=self.degenerate_mobility_power,
                        phase_potential_code=self.phase_potential_code,
                        phase_log_theta=self.phase_log_theta,
                        phase_log_theta_c=self.phase_log_theta_c,
                        phase_log_delta=self.phase_log_delta,
                    )
                )
            # IMEX split: keep full degenerate mobility diffusion explicit and
            # stabilize only the linear CH biharmonic part implicitly.
            if not use_poly_degenerate_convex_split and not use_log_degenerate_convex_split:
                rhs = phi_np - current_dt * convective_term + current_dt * diffusive_deg_old
                if mobility_ref > 0.0:
                    lap_phi_old = self._apply_sparse_operator(A_phi, phi_np)
                    lap2_phi_old = self._apply_sparse_operator(A, lap_phi_old)
                    rhs = rhs + current_dt * mobility_ref * (self.epsilon ** 2 / self.Pe) * lap2_phi_old
        else:
            if self._phase_debug_skip_ch_diffusion:
                nonlinear_term = np.zeros_like(phi_np)
                contact_delta = None
                contact_delta_term = None
            elif use_true_convex_split:
                if self.phase_log_implicit_concave:
                    nonlinear_term = np.zeros_like(phi_np)
                else:
                    nonlinear_term = -self.phase_log_theta_c * self._apply_sparse_operator(
                        C, phi_np
                    )
                contact_delta = None
                contact_delta_term = None
            else:
                nonlinear_term = self._apply_conservative_outer_operator(
                    np.asarray(jax_free_energy_derivative(
                        phi,
                        self.phase_potential_code,
                        self.phase_log_theta,
                        self.phase_log_theta_c,
                        self.phase_log_delta,
                    )), dx, dy, geometry
                )
                contact_delta = None
                contact_delta_term = None
            self._phase_stage("op_nonlinear_term", nonlinear_term)
            lap_phi_old = self._apply_sparse_operator(A_phi, phi_np)
            self._phase_stage("op_lap_phi_old", lap_phi_old)
            rhs = phi_np - current_dt * convective_term + gamma * nonlinear_term
            if beta_stabilization > 0.0:
                rhs = rhs - beta_stabilization * self._apply_sparse_operator(C, phi_np)
            if (
                not self._phase_debug_skip_ch_diffusion
                and contact_laplacian_old is not None
                and self.semi_implicit_contact_split != "no_delta"
            ):
                rhs, contact_delta, contact_delta_term = self._apply_contact_delta_to_rhs(
                    phi_np, rhs, A_phi, contact_laplacian_old, current_dt, dx, dy, geometry
                )
            if contact_delta is not None:
                self._phase_stage("op_contact_delta", contact_delta)
            if contact_delta_term is not None:
                self._phase_stage("op_contact_delta_term", contact_delta_term)

        self._phase_stage("02_rhs_as_field", rhs)
        self._phase_stage("op_rhs_increment", rhs - phi_np)

        if use_true_convex_split and not self._phase_debug_skip_ch_diffusion:
            phi_new_np = self._solve_log_entropy_convex_split(
                rhs,
                phi_np,
                C,
                A_phi,
                alpha,
                gamma,
            )
        elif use_log_degenerate_convex_split and not self._phase_debug_skip_ch_diffusion:
            phi_new_np = self._solve_log_entropy_convex_split(
                rhs,
                phi_np,
                C_mob,
                A_phi,
                gamma * (self.epsilon ** 2),
                gamma,
            )
        elif use_poly_degenerate_convex_split and not self._phase_debug_skip_ch_diffusion:
            phi_new_np = self._solve_polynomial_degenerate_convex_split(
                rhs,
                phi_np,
                C_mob,
                A_phi,
                gamma,
            )
        else:
            phi_new_np = np.asarray(
                rhs if M is None else self._solve_helmholtz_biharmonic(M, rhs, x0=phi_np)
            )
        self._phase_stage("03_after_linear_solve", phi_new_np)

        if self._phase_debug_copy_wall_from_row1_after_solve:
            phi_new_np[:, 0] = phi_new_np[:, 1]
        elif self._phase_debug_freeze_wall_after_solve:
            self._freeze_wall_row_numpy(phi_new_np, phi_np)

        phi_new = jnp.asarray(phi_new_np, dtype=phi.dtype)
        self._phase_stage("04_after_jnp_cast", phi_new)

        if self.phase_checkerboard_filter > 0.0:
            phi_new = jax_checkerboard_filter(phi_new, self.phase_checkerboard_filter)
            self._phase_stage("04b_after_checkerboard_filter", phi_new)

        skip_bottom = bool(skip_bottom or self._should_skip_bottom_advection_bc())
        phi_new = self._apply_advection_bcs(phi_new, U, current_dt, dx, dy, geometry, skip_bottom=skip_bottom)
        self._phase_stage("05_after_advection_bcs", phi_new)
        if self._phase_debug_freeze_wall_after_advection:
            phi_new = phi_new.at[:, 0].set(phi[:, 0])

        phi_new = self._apply_phase_bcs_only(phi_new, dx, dy, geometry, psi=psi, U=U)
        self._phase_stage("06_after_phase_bcs", phi_new)
        if self._phase_debug_freeze_wall_after_phase_bcs:
            phi_new = phi_new.at[:, 0].set(phi[:, 0])

        phi_new = self._preserve_phase_sum_if_requested(phi, phi_new)
        self._phase_stage("07_after_preserve_phase_sum", phi_new)
        if self._phase_debug_freeze_wall_after_preserve:
            phi_new = phi_new.at[:, 0].set(phi[:, 0])

        return phi_new


class PhaseFieldSolverSimple(BasePhaseFieldSolver):
    """Legacy phase-field solver with post-update boundary enforcement."""

    def update(self, phi, U, current_dt, dx, dy, geometry, use_jax=True, psi=None, u_face=None, v_face=None):
        """Update phase field using the legacy operator path."""
        if self.phase_update_mode == "semi_implicit_ch":
            return self._semi_implicit_ch_step(
                phi, U, current_dt, dx, dy, geometry, psi=psi,
                skip_bottom=False, u_face=u_face, v_face=v_face
            )
        if self.phase_update_mode == "bdf2_ch":
            return self._bdf2_ch_step(
                phi, U, current_dt, dx, dy, geometry, psi=psi,
                skip_bottom=False, u_face=u_face, v_face=v_face
            )

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
                                   degenerate_mobility_blend=self.degenerate_mobility_blend,
                                   degenerate_mobility_power=self.degenerate_mobility_power,
                                   phase_potential_code=self.phase_potential_code,
                                   phase_log_theta=self.phase_log_theta,
                                   phase_log_theta_c=self.phase_log_theta_c,
                                   phase_log_delta=self.phase_log_delta,
                                   u_face=u_face,
                                   v_face=v_face)
        return self._apply_post_update_bcs(phi_new, U, current_dt, dx, dy, geometry, psi=psi)


class PhaseFieldSolverGhostCell(BasePhaseFieldSolver):
    """Experimental ghost-cell phase-field solver for bottom wetting BCs."""

    def update(self, phi, U, current_dt, dx, dy, geometry, use_jax=True, psi=None, u_face=None, v_face=None):
        """Update phase field using the ghost-cell bottom-wall path."""
        if self.phase_update_mode == "semi_implicit_ch":
            return self._semi_implicit_ch_step(
                phi, U, current_dt, dx, dy, geometry, psi=psi,
                skip_bottom=True, u_face=u_face, v_face=v_face
            )
        if self.phase_update_mode == "bdf2_ch":
            return self._bdf2_ch_step(
                phi, U, current_dt, dx, dy, geometry, psi=psi,
                skip_bottom=True, u_face=u_face, v_face=v_face
            )

        bottom_velocity_bc = self.config.get("boundary_conditions", {}).get("velocity", {}).get("bottom", "no_slip")
        self._phase_stage_reset()
        self._phase_stage("00_input_phi", phi)

        bottom_ghost_phi = self.bc_manager.contact_angle_bc.build_bottom_ghost_row_jax(
            phi, dx, dy, geometry, psi=psi, U=U, bottom_velocity_bc=bottom_velocity_bc
        )
        self._record_bottom_ghost_instep(bottom_ghost_phi, phi)
        self._phase_stage("01_after_build_bottom_ghost_input_phi", phi)
        self._phase_stage_ghost("01_bottom_ghost", bottom_ghost_phi, phi)

        U_phase = jnp.zeros_like(U) if self._phase_debug_zero_advection else U
        phi_new = jax_update_phase_ghost(
            phi, U_phase, current_dt, dx, dy,
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
            degenerate_mobility_blend=self.degenerate_mobility_blend,
            degenerate_mobility_power=self.degenerate_mobility_power,
            phase_potential_code=self.phase_potential_code,
            phase_log_theta=self.phase_log_theta,
            phase_log_theta_c=self.phase_log_theta_c,
            phase_log_delta=self.phase_log_delta,
            u_face=u_face,
            v_face=v_face,
            phase_flux_shift=self.phase_flux_shift,
        )
        self._phase_stage("03_after_explicit_update", phi_new)
        if self.advection_bc_manager is not None:
            phi_new = self.advection_bc_manager.apply_boundary_conditions(
                phi_new, U, current_dt, dx, dy, use_jax=True, geometry=geometry, skip_bottom=True
            )
        self._phase_stage("05_after_advection_bcs", phi_new)
        if self.bc_manager is not None:
            phi_new = self.bc_manager.apply_boundary_conditions(
                phi_new, dx, dy, use_jax=True, psi=psi, U=U, geometry=geometry
            )
        self._phase_stage("06_after_phase_bcs", phi_new)
        phi_new = self._preserve_phase_sum_if_requested(phi, phi_new)
        self._phase_stage("07_after_preserve_phase_sum", phi_new)
        return phi_new


# Backward-compatible alias for code paths not yet migrated.
PhaseFieldSolver = PhaseFieldSolverSimple
