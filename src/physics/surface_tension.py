"""
Surface tension calculations for droplet spreading simulation.
"""

import jax.numpy as jnp
from jax import jit
from numerics.finite_differences import jax_gradient, jax_divergence, jax_norm, jax_laplacian
from physics.free_energy import (
    jax_free_energy_derivative,
    potential_code_from_name,
    diffuse_interface_sigma,
)


@jit
def jax_curvature(phi, dx, dy, f_1_grid):
    """Curvature κ = ∇·(∇φ/|∇φ|). Terrain gradient and divergence. f_1_grid: (Nx, Ny)."""
    grad_phi = jax_gradient(phi, dx, dy, f_1_grid)
    grad_phi_magnitude = jnp.maximum(jax_norm(grad_phi), 1e-6)
    n_x = grad_phi[..., 0] / grad_phi_magnitude
    n_y = grad_phi[..., 1] / grad_phi_magnitude
    n_field = jnp.stack([n_x, n_y], axis=-1)
    return jax_divergence(n_field, dx, dy, f_1_grid)


def jax_curvature_smooth(phi, dx, dy, f_1_grid, smoothing_radius=1):
    """Calculate curvature with smoothing to reduce grid artifacts.
    
    Args:
        phi: Phase field (Nx, Ny).
        dx, dy: Grid spacing.
        f_1_grid: Terrain gradient f'(x) on grid (Nx, Ny).
        smoothing_radius: Radius of smoothing stencil (1 = 3x3).
    
    Returns:
        Smoothed curvature field (Nx, Ny).
    """
    curvature = jax_curvature(phi, dx, dy, f_1_grid)
    return _smooth_curvature(curvature, smoothing_radius)


@jit
def _smooth_curvature_3x3(curvature):
    """Smooth curvature using 3x3 averaging filter.
    
    Applies a 3x3 averaging stencil to reduce grid artifacts,
    especially at corners where rectangular grid creates spurious high curvature.
    
    Args:
        curvature: Curvature field (Nx, Ny).
    
    Returns:
        Smoothed curvature field (Nx, Ny).
    """
    padded = jnp.pad(curvature, ((1, 1), (1, 1)), mode="edge")
    return (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.0


def _smooth_curvature(curvature, radius=1):
    """Smooth curvature using averaging filter (wrapper for JIT compatibility).
    
    Args:
        curvature: Curvature field (Nx, Ny).
        radius: Smoothing radius (1 = 3x3, 2 = 5x5 via multiple passes).
    
    Returns:
        Smoothed curvature field (Nx, Ny).
    """
    if radius == 0:
        return curvature
    
    # Apply 3x3 smoothing iteratively for radius > 1
    smoothed = curvature
    for _ in range(radius):
        smoothed = _smooth_curvature_3x3(smoothed)
    return smoothed


@jit
def jax_curvature_stats(phi, dx, dy, f_1_grid):
    """Curvature statistics (max, mean) at the interface. f_1_grid: (Nx, Ny)."""
    curvature = jax_curvature(phi, dx, dy, f_1_grid)
    
    # Only consider curvature at the interface (where |phi| < 0.9)
    interface_mask = jnp.abs(phi) < 0.9
    curvature_at_interface = jnp.where(interface_mask, jnp.abs(curvature), 0.0)
    
    curvature_max = jnp.max(curvature_at_interface)
    
    # Mean only over interface cells
    interface_count = jnp.sum(interface_mask.astype(jnp.float32))
    curvature_sum = jnp.sum(curvature_at_interface)
    curvature_mean = jnp.where(interface_count > 0, curvature_sum / interface_count, 0.0)
    
    return curvature_max, curvature_mean


def _weber_field(phi, We1, We2, weber_interpolation):
    """Per-cell Weber number from the configured interpolation."""
    phi_mapped = 0.5 * (phi + 1.0)
    if weber_interpolation == "harmonic":
        return 1.0 / ((1.0 - phi_mapped) / We2 + phi_mapped / We1)
    if weber_interpolation == "arithmetic":
        return (1.0 - phi_mapped) * We2 + phi_mapped * We1
    if weber_interpolation == "constant_liquid":
        return jnp.full_like(phi, float(We2))
    raise ValueError(
        f"Unsupported weber_interpolation='{weber_interpolation}'. "
        "Use one of: 'constant_liquid', 'harmonic', 'arithmetic'."
    )


def jax_surface_tension_force(
    phi,
    epsilon,
    We1,
    We2,
    dx,
    dy,
    f_1_grid,
    interface_mask=None,
    smooth_curvature=True,
    smoothing_radius=1,
    use_composition_field=True,
    composition_force_scale=1.0,
    weber_interpolation="constant_liquid",
    force_form="csf",
):
    """Surface tension force. f_1_grid: terrain gradient (Nx, Ny).

    By default, compute curvature and gradients from composition c=(phi+1)/2
    following common diffuse-interface notation in literature.
    """
    phase_field = 0.5 * (phi + 1.0) if use_composition_field else phi
    if smooth_curvature:
        curvature = jax_curvature_smooth(phase_field, dx, dy, f_1_grid, smoothing_radius=smoothing_radius)
    else:
        curvature = jax_curvature(phase_field, dx, dy, f_1_grid)
    curvature = jnp.stack([curvature, curvature], axis=-1)
    grad_phase = jax_gradient(phase_field, dx, dy, f_1_grid)
    norm_grad_phase = jax_norm(grad_phase)
    norm_grad_phase = jnp.stack([norm_grad_phase, norm_grad_phase], axis=-1)
    
    We = _weber_field(phi, We1, We2, weber_interpolation)
    We = jnp.stack([We, We], axis=-1)
    
    prefactor = composition_force_scale * (3 * jnp.sqrt(2) * epsilon / (4 * We))
    if force_form == "legacy_norm_grad":
        tension_force = prefactor * curvature * norm_grad_phase * grad_phase
    else:
        tension_force = prefactor * curvature * grad_phase
    
    return tension_force


def jax_potential_surface_tension_force(
    phi,
    epsilon,
    We1,
    We2,
    dx,
    dy,
    f_1_grid,
    f_2_grid,
    sigma_ch,
    potential_code=0,
    log_theta=0.25,
    log_theta_c=1.0,
    log_delta=1e-6,
    weber_interpolation="constant_liquid",
    force_scale=1.0,
):
    """Energy-consistent capillary force F = lambda * mu * grad(phi).

    mu = f'(phi) - epsilon^2 lap(phi) is the same chemical potential as in the
    Cahn-Hilliard solve, so the force vanishes identically in bulk equilibrium
    (mu = const, grad(phi) = 0) and needs no curvature reconstruction,
    normalization, smoothing, or wall-row overwrites.

    Calibration: the planar-interface pressure jump of this force is
    lambda * sigma_ch * kappa, where sigma_ch = diffuse_interface_sigma(...).
    lambda is chosen so the effective surface tension equals the one the CSF
    form produced, sigma_eff = 3 sqrt(2) epsilon / (4 We); existing config
    Weber numbers therefore keep their calibrated meaning.
    """
    mu = jax_free_energy_derivative(
        phi, potential_code, log_theta, log_theta_c, log_delta
    ) - epsilon**2 * jax_laplacian(phi, dx, dy, f_1_grid, f_2_grid)
    grad_phi = jax_gradient(phi, dx, dy, f_1_grid)

    We = _weber_field(phi, We1, We2, weber_interpolation)
    sigma_eff = 3.0 * jnp.sqrt(2.0) * epsilon / (4.0 * We)
    lam = force_scale * sigma_eff / sigma_ch

    return jnp.stack([lam, lam], axis=-1) * mu[..., None] * grad_phi


@jit
def jax_apply_surface_tension_boundary_conditions(surface_tension, phi, contact_angle=60, f_1_surface=None):
    """Apply boundary conditions to surface tension force. Surface at j=0.
    
    f_1_surface: slope f'(x) at surface row (Nx,). If None, flat surface (normal vertical).
    """
    # Config uses liquid-side contact angle; convert to internal side so the
    # wall-force projection stays consistent with phase-field BC convention.
    theta = (180.0 - contact_angle) * jnp.pi / 180.0

    if f_1_surface is not None:
        norm_factor = jnp.sqrt(1 + f_1_surface**2)
        n_surface_x = -f_1_surface / norm_factor
        n_surface_y = 1.0 / norm_factor
        sf_surf_x = surface_tension[:, 0, 0]
        sf_surf_y = surface_tension[:, 0, 1]
        sf_normal = sf_surf_x * n_surface_x + sf_surf_y * n_surface_y
        sf_tangential = sf_surf_x * (-n_surface_y) + sf_surf_y * n_surface_x
        sf_normal_adj = sf_normal * jnp.cos(theta)
        sf_x_new = sf_normal_adj * n_surface_x - sf_tangential * n_surface_y
        sf_y_new = sf_normal_adj * n_surface_y + sf_tangential * n_surface_x
        sf = surface_tension.at[:, 0, 0].set(sf_x_new)
        sf = sf.at[:, 0, 1].set(sf_y_new)
    else:
        sf = surface_tension.at[:, 0, 1].set(surface_tension[:, 1, 1] * jnp.cos(theta))
        sf = sf.at[:, 0, 0].set(surface_tension[:, 1, 0])
    
    # Top, left, right: zero-gradient copy without interface gating.
    sf = sf.at[:, -1, 0].set(sf[:, -2, 0])
    sf = sf.at[:, -1, 1].set(sf[:, -2, 1])
    sf = sf.at[0, :, 0].set(sf[1, :, 0])
    sf = sf.at[0, :, 1].set(sf[1, :, 1])
    sf = sf.at[-1, :, 0].set(sf[-2, :, 0])
    sf = sf.at[-1, :, 1].set(sf[-2, :, 1])
    
    return sf


class SurfaceTensionSolver:
    """Surface tension solver for droplet spreading simulation."""
    
    def __init__(
        self,
        epsilon,
        We1,
        We2,
        contact_angle,
        smooth_curvature=True,
        smoothing_radius=1,
        use_composition_field=True,
        composition_force_scale=1.0,
        weber_interpolation="constant_liquid",
        apply_boundary_overwrite=True,
        force_form="csf",
        potential_params=None,
    ):
        """Initialize surface tension solver.
        
        Args:
            epsilon: Interface thickness.
            We1, We2: Weber numbers for phase 1 (air) and phase 2 (liquid).
            contact_angle: Contact angle in degrees.
            smooth_curvature: Whether to smooth curvature to reduce grid artifacts.
            smoothing_radius: Radius of smoothing stencil (1 = 3x3).
            force_form: 'csf', 'legacy_norm_grad', or 'potential' (mu grad phi).
            potential_params: required for force_form='potential'; dict with the
                CH free energy used by the phase solver: {'phase_potential',
                'phase_log_theta', 'phase_log_theta_c', 'phase_log_delta'}.
        """
        self.epsilon = epsilon
        self.We1 = We1
        self.We2 = We2
        self.contact_angle = contact_angle
        self.smooth_curvature = smooth_curvature
        self.smoothing_radius = smoothing_radius
        self.use_composition_field = bool(use_composition_field)
        self.composition_force_scale = float(composition_force_scale)
        self.weber_interpolation = str(weber_interpolation)
        self.apply_boundary_overwrite = bool(apply_boundary_overwrite)
        self.force_form = str(force_form)
        if self.force_form == "potential":
            if potential_params is None:
                raise ValueError(
                    "force_form='potential' requires potential_params (the CH "
                    "free-energy settings) so the force uses the same "
                    "thermodynamic potential as the phase solver."
                )
            self.potential_code = potential_code_from_name(
                potential_params.get("phase_potential", "polynomial")
            )
            self.log_theta = float(potential_params.get("phase_log_theta", 0.25))
            self.log_theta_c = float(potential_params.get("phase_log_theta_c", 1.0))
            self.log_delta = float(potential_params.get("phase_log_delta", 1e-6))
            self.sigma_ch = diffuse_interface_sigma(
                epsilon, self.potential_code, self.log_theta, self.log_theta_c
            )
    
    def calculate_force(self, phi, dx, dy, geometry, use_jax=True, interface_mask=None):
        """Calculate surface tension force. geometry: from state."""
        if self.force_form == "potential":
            return jax_potential_surface_tension_force(
                phi,
                self.epsilon,
                self.We1,
                self.We2,
                dx,
                dy,
                geometry.f_1_grid,
                geometry.f_2_grid,
                self.sigma_ch,
                potential_code=self.potential_code,
                log_theta=self.log_theta,
                log_theta_c=self.log_theta_c,
                log_delta=self.log_delta,
                weber_interpolation=self.weber_interpolation,
                force_scale=self.composition_force_scale,
            )
        return jax_surface_tension_force(
            phi,
            self.epsilon,
            self.We1,
            self.We2,
            dx,
            dy,
            geometry.f_1_grid,
            interface_mask=interface_mask,
            smooth_curvature=self.smooth_curvature,
            smoothing_radius=self.smoothing_radius,
            use_composition_field=self.use_composition_field,
            composition_force_scale=self.composition_force_scale,
            weber_interpolation=self.weber_interpolation,
            force_form=self.force_form,
        )
    
    def apply_boundary_conditions(self, surface_tension, phi, use_jax=True, geometry=None, **kwargs):
        """Apply boundary conditions to surface tension. Surface at j=0. Normal from geometry.f_1_grid if non-flat."""
        if self.force_form == "potential":
            # The potential force already vanishes in bulk phases and is
            # variationally consistent with the wetting BC; the wall-row
            # overwrite below is a CSF-only regularization.
            return surface_tension
        if not self.apply_boundary_overwrite:
            return surface_tension
        f_1_surface = None
        if geometry is not None and getattr(geometry, "has_geometry", False):
            f_1_surface = geometry.f_1_grid[:, 0]
        return jax_apply_surface_tension_boundary_conditions(
            surface_tension, phi, self.contact_angle, f_1_surface=f_1_surface)
