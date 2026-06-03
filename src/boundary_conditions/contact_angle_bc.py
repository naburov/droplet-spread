"""
Contact angle boundary conditions for phase field.

This module implements various contact angle boundary condition formulations
that can be selected from configuration files.

BOUNDARY CONDITIONS FOR dphi/dx AND dphi/dy:
============================================

This module enforces boundary conditions on the phase field phi by directly
modifying phi values at boundaries. The effective BCs on derivatives are:

BOTTOM BOUNDARY (y = 0 or y = h_bottom(x)):
--------------------------------------------
At contact points (where interface touches surface):
  - Normal derivative: ∂φ/∂n_surface = -cos(θ) |∇φ|
    where n_surface is the surface normal (computed from h_bottom or psi)
    and θ is the contact angle
  
  - For flat surface (h_bottom ≈ 0):
    * n_surface = (0, 1)  [vertical normal]
    * ∂φ/∂n = ∂φ/∂y = -cos(θ) |∇φ|
    * This implies: ∂φ/∂y = -cos(θ) √[(∂φ/∂x)² + (∂φ/∂y)²]
    * 
    * ∂φ/∂x at bottom boundary:
    *   - NOT explicitly constrained by boundary condition
    *   - Determined by interior solution and continuity
    *   - The contact angle condition relates ∂φ/∂y to |∇φ| (which includes ∂φ/∂x),
    *     but ∂φ/∂x itself is free to vary along the bottom boundary
    *   - In practice, ∂φ/∂x is computed from interior phi values using finite differences
    *   
    *   MATHEMATICAL NOTE: For Cahn-Hilliard (4th order PDE), we need 2 BCs per boundary.
    *   Currently we have:
    *     1. Contact angle: ∂nφ = -cos(θ)|∇φ| (constrains normal derivative)
    *        - Implemented in this module (contact_angle_bc.py)
    *     2. Zero flux on μ: ∂nμ = 0 (constrains chemical potential)
    *        - Implemented separately in phase_field.py (jax_update_phase)
    *        - Applied via jax_apply_chemical_potential_zero_flux_bc() from chemical_potential_bc.py
    *        - Applied to all boundaries: ∂μ/∂y = 0 (top/bottom), ∂μ/∂x = 0 (left/right)
    *   These are sufficient. The contact angle condition couples ∂φ/∂x and ∂φ/∂y through |∇φ|,
    *   so no explicit constraint on ∂φ/∂x is needed. However, for numerical consistency,
    *   we could optionally enforce continuity of ∂φ/∂x at the boundary if issues arise.
  
  - For curved surface (h_bottom(x) varies):
    * n_surface = (-dh/dx, 1) / √(1 + (dh/dx)²)
    * ∂φ/∂n = n_x * ∂φ/∂x + n_y * ∂φ/∂y = -cos(θ) |∇φ|
    * This couples ∂φ/∂x and ∂φ/∂y through the surface normal
  
  - At non-contact points (no interface):
    * Neumann BC: ∂φ/∂y = 0  (zero normal derivative)
    * For flat surface: ∂φ/∂y = 0
    * For curved surface: ∂φ/∂n = 0

Implementation note: The code enforces these BCs by:
  1. Computing gradient at surface cells
  2. Calculating desired normal derivative
  3. Applying correction to phi values at boundary
  4. Enforcing the contact-angle target directly in contact-line cells

TOP BOUNDARY (y = Ny-1):
------------------------
  - Neumann BC: ∂φ/∂y = 0
  - Implemented as: phi[:, -1] = phi[:, -2]

LEFT BOUNDARY (x = 0):
----------------------
  - Neumann BC: ∂φ/∂x = 0
  - Implemented as: phi[0, :] = phi[1, :]

RIGHT BOUNDARY (x = Nx-1):
--------------------------
  - Neumann BC: ∂φ/∂x = 0
  - Implemented as: phi[-1, :] = phi[-2, :]

GEOMETRY-AWARE MODE:
--------------------
When use_geometry_aware=True:
  - Surface normal computed from:
    1. Height function h_bottom(x) if available
    2. Ice phase field psi gradient if h_bottom not available
    3. Vertical normal (0, 1) otherwise
  
  - Contact angle enforced relative to surface normal, not horizontal
  - Surface cells identified at y = h_bottom(x), not y = 0
  - BCs applied at actual surface location, not computational boundary

SIMPLE MODE:
------------
When method="simple":
  - Assumes flat surface (vertical normal)
  - ∂φ/∂y = -cos(θ_effective) |∇φ| at contact points; θ_effective comes from _get_effective_contact_angle_jax
  - Cox-Voinov: if use_cox_voinov=True, θ_effective = θ₀ + C·sign(U_cl)·|U_cl|^n; U_cl is U[:,0,0] (slip)
    or U[:,1,0] (no_slip). So "simple" and Cox-Voinov are compatible for flat surfaces.
  - For tilted/non-flat surfaces use method="geometry_aware" so the surface normal and contact-line
    velocity (tangential along slope) are correct; "simple" would use wrong normal and wrong U_cl.
  - ∂φ/∂y = 0 at non-contact points
  - Applied at y = 0 (computational boundary)

"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from numerics.finite_differences import jax_dx, jax_dy, jax_gradient


@jit
def jax_build_contact_angle_ghost_bottom(
    phi,
    dx,
    dy,
    f_1_grid,
    theta_effective,
    contact_mask_soft_band=0.0,
    contact_mask_grad_scale=0.0,
    contact_angle_ghost_law=1,
    epsilon=1.0,
    contact_angle_full_wall=False,
    contact_angle_wall_energy_scale=1.0,
):
    """Build a ghost row below the bottom wall for contact-angle enforcement.

    The ghost value is chosen so the centered wall-normal derivative satisfies the
    target contact-angle relation near the contact line, while regions away from the
    contact line fall back to zero-normal-gradient (ghost == interior wall value).

    For terrain-following geometry y = eta + h(x), physical derivatives are
    grad(phi) = (phi_x - h' phi_eta, phi_eta).  The wall normal derivative is

        q = grad(phi)·n = (-h' phi_x + (1 + h'^2) phi_eta) / sqrt(1 + h'^2).

    Solving this for phi_eta gives the ghost value below eta=0.
    """
    f_1_surface = f_1_grid[:, 0]
    metric = 1.0 + f_1_surface**2
    metric_sqrt = jnp.sqrt(metric)

    phi_x_comp = jax_dx(phi, h=dx)[:, 0]
    phi_eta_old = jax_dy(phi, h=dy)[:, 0]
    grad_phi_x = phi_x_comp - f_1_surface * phi_eta_old
    grad_phi_y = phi_eta_old
    tangential_derivative = (grad_phi_x + f_1_surface * grad_phi_y) / metric_sqrt

    norm_grad_phi = jnp.sqrt(grad_phi_x**2 + grad_phi_y**2)
    norm_grad_phi = jnp.where(norm_grad_phi < 1e-10, 1e-10, norm_grad_phi)

    # Solve the wetting relation q = -cos(theta) * sqrt(phi_x^2 + q^2)
    # for the wall-normal derivative q. This keeps the ghost value consistent
    # with the centered ghost-cell derivative instead of reusing the old
    # one-sided wall gradient inside |grad(phi)|.
    cos_theta = jnp.cos(theta_effective)
    sin_theta = jnp.sqrt(jnp.maximum(1.0 - cos_theta**2, 1e-12))
    analytic_derivative = -(cos_theta / sin_theta) * jnp.abs(tangential_derivative)

    # Variational wall-free-energy form for the standard tanh profile:
    # |grad(phi)| = (1 - phi^2)/(sqrt(2) * epsilon). It avoids feeding noisy
    # wall gradients back into the ghost row and vanishes smoothly in bulk.
    phi_wall = jnp.clip(phi[:, 0], -1.0, 1.0)
    wall_energy_derivative = (
        -contact_angle_wall_energy_scale
        * cos_theta
        * (1.0 - phi_wall**2)
        / (jnp.sqrt(2.0) * jnp.maximum(epsilon, 1e-12))
    )
    use_wall_energy = contact_angle_ghost_law == 1
    normal_derivative = jnp.where(use_wall_energy, wall_energy_derivative, analytic_derivative)

    phi_bottom = phi[:, 0]
    phi_above = phi[:, 1]

    phi_crosses_zero = (phi_bottom * phi_above) < 0.0
    phi_near_zero = (jnp.abs(phi_bottom) < 0.5) | (jnp.abs(phi_above) < 0.5)
    has_interface = norm_grad_phi > 1e-3
    contact_mask = (phi_crosses_zero | phi_near_zero) & has_interface
    hard_weight = contact_mask.astype(phi.dtype)

    # Optional soft contact-line mask (diffuse-interface-consistent blend).
    # This avoids binary mask flicker near |phi|≈0 that can leave a sawtooth
    # imprint on the wall profile. Defaults keep legacy hard-mask behavior.
    interface_strength = jnp.minimum(jnp.abs(phi_bottom), jnp.abs(phi_above))
    band = jnp.maximum(contact_mask_soft_band, 1e-12)
    grad_scale = jnp.maximum(contact_mask_grad_scale, 1e-12)
    w_phi = jnp.clip((band - interface_strength) / band, 0.0, 1.0)
    w_grad = jnp.clip((norm_grad_phi - 1e-3) / grad_scale, 0.0, 1.0)
    soft_weight = w_phi * w_grad

    # The ghost row is below eta=0. Convert the requested physical normal
    # derivative q to a computational eta derivative, then center it between
    # phi_above and the ghost value.
    eta_derivative = (normal_derivative * metric_sqrt + f_1_surface * phi_x_comp) / metric
    ghost_contact = phi_above - 2.0 * dy * eta_derivative
    ghost_neumann = phi_above
    use_soft = contact_mask_soft_band > 0.0
    contact_weight = jnp.where(use_soft, soft_weight, hard_weight)
    contact_weight = jnp.where(contact_angle_full_wall | use_wall_energy, 1.0, contact_weight)
    return ghost_neumann + contact_weight * (ghost_contact - ghost_neumann)


def _geometry_aware_contact_angle_impl(phi, dx, dy, f_1_grid, psi, theta_effective, use_geometry_aware, f_1_surface):
    """Implementation of geometry-aware contact angle BC. Surface at j=0 (eta=0).
    
    f_1_surface: slope f'(x) at surface row (Nx,). Pass zeros if flat. Normal from f_1 or psi or vertical.
    """
    from jax import lax
    has_f_1 = jnp.any(jnp.abs(f_1_surface) > 1e-10)

    def compute_normal_from_f_1():
        norm_factor = jnp.sqrt(1 + f_1_surface**2)
        n_surface_x = -f_1_surface / norm_factor
        n_surface_y = 1.0 / norm_factor
        return n_surface_x, n_surface_y

    def compute_normal_from_psi():
        # Compute gradient of psi
        grad_psi = jax_gradient(psi, dx, dy, f_1_grid)  # Shape: (Nx, Ny, 2)
        
        # Get gradient at bottom boundary (where contact happens)
        grad_psi_bottom = grad_psi[:, 0, :]  # Shape: (Nx, 2)
        
        # Check if ice is present at bottom
        psi_bottom = psi[:, 0]  # Shape: (Nx,)
        ice_present = psi_bottom > 0.0  # Ice where psi > 0
        
        # Compute norm of gradient
        norm_grad_psi = jnp.sqrt(grad_psi_bottom[:, 0]**2 + grad_psi_bottom[:, 1]**2)
        norm_grad_psi = jnp.where(norm_grad_psi < 1e-10, 1.0, norm_grad_psi)
        
        # Ice surface normal (pointing from ice to water)
        # Normal is opposite to gradient of psi (since psi increases in ice)
        n_surface_x = -grad_psi_bottom[:, 0] / norm_grad_psi
        n_surface_y = -grad_psi_bottom[:, 1] / norm_grad_psi
        
        # If no ice present, use vertical normal (substrate)
        n_surface_x = jnp.where(ice_present, n_surface_x, 0.0)
        n_surface_y = jnp.where(ice_present, n_surface_y, 1.0)
        return n_surface_x, n_surface_y
    
    def compute_vertical_normal():
        # No geometry or geometry-aware disabled: use vertical normal (substrate)
        n_surface_x = jnp.zeros(phi.shape[0])
        n_surface_y = jnp.ones(phi.shape[0])
        return n_surface_x, n_surface_y
    
    if use_geometry_aware:
        has_psi = jnp.any(jnp.abs(psi) > 1e-10)
        n_surface_x, n_surface_y = lax.cond(
            has_f_1,
            lambda _: compute_normal_from_f_1(),
            lambda _: lax.cond(
                has_psi,
                lambda _: compute_normal_from_psi(),
                lambda _: compute_vertical_normal(),
                None
            ),
            None
        )
    else:
        n_surface_x, n_surface_y = compute_vertical_normal()
    
    # Step 2: Surface at j=0 (curvilinear eta=0)
    Nx, Ny = phi.shape
    grad_phi = jax_gradient(phi, dx, dy, f_1_grid)  # Shape: (Nx, Ny, 2)
    grad_phi_x_at_surf = grad_phi[:, 0, 0]  # Shape: (Nx,)
    grad_phi_y_at_surf = grad_phi[:, 0, 1]  # Shape: (Nx,)
    
    # Step 4: Compute normal derivative relative to surface
    # Current normal derivative: ∇φ · n_surface (at actual surface)
    grad_phi_dot_n_surface = grad_phi_x_at_surf * n_surface_x + grad_phi_y_at_surf * n_surface_y
    
    # Magnitude of gradient at actual surface
    norm_grad_phi = jnp.sqrt(grad_phi_x_at_surf**2 + grad_phi_y_at_surf**2)
    norm_grad_phi = jnp.where(norm_grad_phi < 1e-10, 1e-10, norm_grad_phi)
    
    # Desired normal derivative: ∂φ/∂n_surface = -cos(θ) |∇φ|
    # Contact angle θ is measured from the surface into the liquid (standard convention)
    # - θ < 90°: hydrophilic (wetting)
    # - θ > 90°: hydrophobic (non-wetting)
    # theta_effective can be scalar or array
    cos_theta = jnp.cos(theta_effective)
    
    # Ensure cos_theta has the right shape for broadcasting
    if cos_theta.ndim == 0:
        # Scalar: broadcast to match norm_grad_phi
        cos_theta = jnp.broadcast_to(cos_theta, norm_grad_phi.shape)
    elif cos_theta.shape != norm_grad_phi.shape:
        # Array: ensure shapes match
        cos_theta = jnp.broadcast_to(cos_theta, norm_grad_phi.shape)
    
    desired_normal_derivative = -cos_theta * norm_grad_phi
    
    # Correction needed to achieve desired normal derivative
    correction = desired_normal_derivative - grad_phi_dot_n_surface
    
    # Step 5: Contact points at surface (curvilinear: j=0)
    phi_bottom = phi[:, 0]
    phi_above = phi[:, 1]

    # Step 6: Apply correction at surface (j=0)

    # Project correction onto surface normal direction
    # Account for surface slope: correction_normal = correction (already computed)
    # To apply at surface, we need to project onto the direction from surface to first fluid cell above
    # For simplicity, use the vertical component scaled by surface normal
    n_surface_y_safe = jnp.where(jnp.abs(n_surface_y) < 1e-10, 1.0, n_surface_y)
    correction_y = correction / n_surface_y_safe
    
    # Get first cell above the surface (j=1)
    phi_above = phi[:, 1]  # Shape: (Nx,)
    
    # Compute what phi_surf should be for contact angle BC
    phi_contact_target = phi_above - correction_y * dy  # Target value for contact angle
    # No contact mask: enforce contact-angle target everywhere along the bottom wall.
    phi_surf_new = phi_contact_target
    phi_new = phi.at[:, 0].set(phi_surf_new)

    # Apply other boundaries (Neumann)
    phi_new = phi_new.at[:, -1].set(phi_new[:, -2])  # Top
    phi_new = phi_new.at[0, :].set(phi_new[1, :])    # Left
    phi_new = phi_new.at[-1, :].set(phi_new[-2, :])  # Right
    
    return phi_new


class ContactAngleBoundaryCondition:
    """Contact angle boundary condition implementation."""
    
    def __init__(self, contact_angle=90, method="simple", epsilon=None, 
                 contact_angle_ice=None, use_ice_aware=False, use_geometry_aware=False,
                 use_cox_voinov=False, cox_voinov_coefficient=1.0, cox_voinov_exponent=1.0/3.0,
                 contact_angle_relaxation=1.0, conserve_phi_sum=True,
                 contact_mask_soft_band=0.0, contact_mask_grad_scale=0.0,
                 cox_voinov_velocity_mode="side_aware",
                 contact_angle_ghost_law="wall_energy", contact_angle_full_wall=False,
                 contact_angle_wall_energy_scale=1.0):
        """Initialize contact angle boundary condition.
        
        Args:
            contact_angle (float): Contact angle in degrees (for water).
            method (str): Method for applying contact angle BC.
                         Options: "simple", "geometry_aware", "ghost_cell"
            epsilon (float): Interface thickness parameter (for Robin method).
            contact_angle_ice (float, optional): Contact angle in degrees for ice.
                                                If None, uses same as water.
            use_ice_aware (bool): Whether to use ice-aware contact angle (requires psi).
            use_geometry_aware (bool): Whether to account for non-flat ice geometry.
                                     If True, contact angle is applied relative to ice surface normal.
            use_cox_voinov (bool): Whether to use Cox-Voinov dynamic contact angle model.
            cox_voinov_coefficient (float): Coefficient for Cox-Voinov law (C in θ = θ₀ ± C|U|^n).
            cox_voinov_exponent (float): Exponent for Cox-Voinov law (n in θ = θ₀ ± C|U|^n).
            contact_angle_relaxation (float): Deprecated compatibility parameter.
                The solver now always applies contact-angle BC without relaxation/blending.
            conserve_phi_sum (bool): Preserve global phi sum after simple contact-angle correction.
            contact_mask_soft_band (float): If > 0, use soft blending width in |phi| units
                for ghost-cell contact-line mask (0 keeps legacy hard mask).
            contact_mask_grad_scale (float): Gradient scaling for soft-mask activation.
            cox_voinov_velocity_mode (str): Dynamic-angle velocity sign mapping.
                "side_aware" keeps legacy side-based sign assignment;
                "local_wall" uses local wall tangential velocity directly.
            contact_angle_ghost_law (str): Ghost-row normal derivative law.
                "analytic_gradient" solves q=-cos(theta)*sqrt(phi_x^2+q^2);
                "wall_energy" uses the tanh-profile wall-free-energy derivative.
            contact_angle_full_wall (bool): Apply ghost contact law along the full
                bottom wall. Useful with "wall_energy" because its derivative
                vanishes smoothly in bulk.
            contact_angle_wall_energy_scale (float): Multiplier for the
                wall-free-energy derivative. 1.0 is the standard tanh-profile
                coefficient.
        """
        self.contact_angle = contact_angle
        self.contact_angle_ice = contact_angle_ice if contact_angle_ice is not None else contact_angle
        self.method = method
        self.epsilon = epsilon
        self.use_ice_aware = use_ice_aware
        self.use_geometry_aware = use_geometry_aware
        self.use_cox_voinov = use_cox_voinov
        self.cox_voinov_coefficient = cox_voinov_coefficient
        self.cox_voinov_exponent = cox_voinov_exponent
        # Compatibility only: keep config/API stable, but ignore this value.
        self.contact_angle_relaxation = 1.0
        self.conserve_phi_sum = bool(conserve_phi_sum)
        self.contact_mask_soft_band = float(contact_mask_soft_band)
        self.contact_mask_grad_scale = float(contact_mask_grad_scale)
        self.cox_voinov_velocity_mode = str(cox_voinov_velocity_mode)
        ghost_law = str(contact_angle_ghost_law)
        if ghost_law not in ("analytic_gradient", "wall_energy"):
            raise ValueError(
                f"Unknown contact_angle_ghost_law: {ghost_law}. "
                "Supported values are: analytic_gradient, wall_energy."
            )
        self.contact_angle_ghost_law = ghost_law
        self.contact_angle_ghost_law_code = 1 if ghost_law == "wall_energy" else 0
        self.contact_angle_full_wall = bool(contact_angle_full_wall)
        self.contact_angle_wall_energy_scale = float(contact_angle_wall_energy_scale)
        
        # Cache compiled function for geometry-aware method
        self._compiled_geometry_aware_fn = None
        
    def apply(self, phi, dx, dy, geometry=None, use_jax=True, psi=None, U=None, bottom_velocity_bc="no_slip"):
        """Apply contact angle BC. geometry: from state (flat if None).
        bottom_velocity_bc: velocity BC at bottom ('no_slip', 'slip_symmetry', 'navier_slip', ...).
        With slip, Cox-Voinov uses U at y=0 (wall); with no_slip, U just above wall (j=1)."""
        if geometry is None:
            from simulation.geometry import Geometry
            geometry = Geometry.flat(phi.shape[0], phi.shape[1])
        return self._apply_jax(phi, dx, dy, geometry, psi=psi, U=U, bottom_velocity_bc=bottom_velocity_bc)

    def _apply_jax(self, phi, dx, dy, geometry, psi=None, U=None, bottom_velocity_bc="no_slip"):
        if self.method == "geometry_aware" or self.use_geometry_aware:
            return self._geometry_aware_contact_angle_jax(phi, dx, dy, geometry, psi=psi, U=U, bottom_velocity_bc=bottom_velocity_bc)
        elif self.method == "simple":
            return self._simple_contact_angle_jax(phi, dx, dy, geometry, psi=psi, U=U, bottom_velocity_bc=bottom_velocity_bc)
        elif self.method == "ghost_cell":
            return phi
        else:
            raise ValueError(
                f"Unknown contact angle method: {self.method}. "
                "Supported methods are: simple, geometry_aware, ghost_cell."
            )

    def _wall_tangential_velocity(self, U, bottom_velocity_bc):
        """Tangential velocity proxy at the wall used by Cox-Voinov."""
        if U is None:
            return None
        if bottom_velocity_bc in ("slip_symmetry", "navier_slip"):
            return U[:, 0, 0]
        return U[:, 1, 0]

    def _contact_side_sign(self, phi, dx, dy, f_1_grid):
        """Side marker from interface orientation at the wall: left=-1, right=+1."""
        grad_phi = jax_gradient(phi, dx, dy, f_1_grid)
        grad_phi_x_wall = grad_phi[:, 0, 0]
        side = jnp.sign(grad_phi_x_wall)
        return jnp.where(jnp.abs(side) < 1e-10, 1.0, side)

    def _signed_contact_line_velocity(self, phi, U, dx, dy, f_1_grid, bottom_velocity_bc):
        """Signed CL speed according to configured Cox-Voinov velocity mode."""
        U_wall = self._wall_tangential_velocity(U, bottom_velocity_bc)
        if U_wall is None:
            return None
        if self.cox_voinov_velocity_mode == "local_wall":
            # Use local wall tangential velocity sign directly.
            return U_wall
        side = self._contact_side_sign(phi, dx, dy, f_1_grid)
        phi_bottom = phi[:, 0]
        phi_above = phi[:, 1]
        interface_mask = ((phi_bottom * phi_above) < 0.0) | (jnp.abs(phi_bottom) < 0.6) | (jnp.abs(phi_above) < 0.6)
        interface_mask_f = interface_mask.astype(U_wall.dtype)
        mask_count = jnp.sum(interface_mask_f)
        mean_u = jnp.where(mask_count > 0.0, jnp.sum(U_wall * interface_mask_f) / mask_count, jnp.mean(U_wall))
        flow_sign = jnp.sign(mean_u)
        flow_sign = jnp.where(jnp.abs(flow_sign) < 1e-10, 1.0, flow_sign)
        return flow_sign * side * jnp.abs(U_wall)

    def build_bottom_ghost_row_jax(self, phi, dx, dy, geometry, psi=None, U=None, bottom_velocity_bc="no_slip"):
        contact_line_velocity = None
        if self.use_cox_voinov and U is not None:
            contact_line_velocity = self._signed_contact_line_velocity(
                phi, U, dx, dy, geometry.f_1_grid, bottom_velocity_bc
            )
        theta_effective = self._get_effective_contact_angle_jax(
            psi, U=U, contact_line_velocity=contact_line_velocity, bottom_velocity_bc=bottom_velocity_bc
        )
        return jax_build_contact_angle_ghost_bottom(
            phi,
            dx,
            dy,
            geometry.f_1_grid,
            theta_effective,
            self.contact_mask_soft_band,
            self.contact_mask_grad_scale,
            self.contact_angle_ghost_law_code,
            self.epsilon if self.epsilon is not None else dy,
            self.contact_angle_full_wall,
            self.contact_angle_wall_energy_scale,
        )

    def build_bottom_wall_energy_q_prime_jax(self, phi, dx, dy, geometry, psi=None, U=None, bottom_velocity_bc="no_slip"):
        """Derivative dq/dphi_wall for the wall-free-energy ghost relation.

        The semi-implicit CH stabilizer freezes this derivative at the old step
        so its bottom-wall Laplacian uses the same physical wetting relation as
        the explicit residual while remaining linear.
        """
        contact_line_velocity = None
        if self.use_cox_voinov and U is not None:
            contact_line_velocity = self._signed_contact_line_velocity(
                phi, U, dx, dy, geometry.f_1_grid, bottom_velocity_bc
            )
        theta_effective = self._get_effective_contact_angle_jax(
            psi, U=U, contact_line_velocity=contact_line_velocity, bottom_velocity_bc=bottom_velocity_bc
        )
        epsilon = self.epsilon if self.epsilon is not None else dy
        phi_wall = jnp.clip(phi[:, 0], -1.0, 1.0)
        return (
            self.contact_angle_wall_energy_scale
            * 2.0
            * jnp.cos(theta_effective)
            * phi_wall
            / (jnp.sqrt(2.0) * jnp.maximum(epsilon, 1e-12))
        )
    
    def _simple_contact_angle_jax(self, phi, dx, dy, geometry, psi=None, U=None, bottom_velocity_bc="no_slip"):
        contact_line_velocity = None
        if self.use_cox_voinov and U is not None:
            contact_line_velocity = self._signed_contact_line_velocity(
                phi, U, dx, dy, geometry.f_1_grid, bottom_velocity_bc
            )
        theta_effective = self._get_effective_contact_angle_jax(
            psi, U=U, contact_line_velocity=contact_line_velocity, bottom_velocity_bc=bottom_velocity_bc
        )
        grad_phi = jax_gradient(phi, dx, dy, geometry.f_1_grid)
        grad_phi_x = grad_phi[:, 1, 0]
        grad_phi_y = grad_phi[:, 1, 1]
        
        norm_grad_phi = jnp.sqrt(grad_phi_x**2 + grad_phi_y**2)
        norm_grad_phi = jnp.where(norm_grad_phi < 1e-10, 1e-10, norm_grad_phi)
        normal_derivative = -jnp.cos(theta_effective) * norm_grad_phi
        
        # Preserve the advected wall state away from the actual contact-line zone.
        phi_bottom_advected = phi[:, 0]
        phi_above = phi[:, 1]
        phi_contact_target = phi_above - normal_derivative * dy

        phi_crosses_zero = (phi_bottom_advected * phi_above) < 0.0
        phi_near_zero = (jnp.abs(phi_bottom_advected) < 0.5) | (jnp.abs(phi_above) < 0.5)
        has_interface = norm_grad_phi > 1e-3
        contact_mask = (phi_crosses_zero | phi_near_zero) & has_interface

        phi_contact_new = phi_contact_target
        phi_bottom_new = jnp.where(contact_mask, phi_contact_new, phi_bottom_advected)
        phi_new = phi.at[:, 0].set(phi_bottom_new)
        phi_new = phi_new.at[:, -1].set(phi_new[:, -2])
        phi_new = phi_new.at[0, :].set(phi_new[1, :])
        phi_new = phi_new.at[-1, :].set(phi_new[-2, :])

        # Simple local wall correction can introduce a small global mass drift.
        # Preserve CH invariance by restoring global phi sum via a local correction
        # on the same contact-line mask used by the boundary update.
        if self.conserve_phi_sum:
            mask = contact_mask.astype(phi_new.dtype)
            mask_count = jnp.sum(mask)
            has_mask = mask_count > 0.0
            delta_sum = jnp.sum(phi) - jnp.sum(phi_new)
            correction = jnp.where(has_mask, delta_sum / jnp.maximum(mask_count, 1.0), 0.0)
            phi_bottom_corrected = jnp.where(contact_mask, phi_new[:, 0] + correction, phi_new[:, 0])
            phi_new = phi_new.at[:, 0].set(phi_bottom_corrected)
        
        return phi_new
    
    def _geometry_aware_contact_angle_jax(self, phi, dx, dy, geometry, psi=None, U=None, bottom_velocity_bc="no_slip"):
        """Geometry-aware contact angle BC. Surface at j=0. Normal from geometry.f_1_grid[:, 0] if non-flat."""
        # Contact line velocity for Cox-Voinov: tangential along the surface (curvilinear (x, eta))
        if self.use_cox_voinov and U is not None:
            u_surf = U[:, 0, 0] if bottom_velocity_bc in ("slip_symmetry", "navier_slip") else U[:, 1, 0]
            v_surf = U[:, 0, 1] if bottom_velocity_bc in ("slip_symmetry", "navier_slip") else U[:, 1, 1]
            f_1_surf = geometry.f_1_grid[:, 0] if geometry is not None else jnp.zeros(phi.shape[0])
            # Tangent along surface in +x: (1, f')/sqrt(1+f'^2) -> u_tang = (u + v*f')/sqrt(1+f'^2)
            denom = jnp.sqrt(1.0 + f_1_surf**2)
            contact_line_velocity = (u_surf + v_surf * f_1_surf) / jnp.where(denom > 1e-10, denom, 1.0)
            side = self._contact_side_sign(phi, dx, dy, geometry.f_1_grid)
            contact_line_velocity = contact_line_velocity * side
        else:
            contact_line_velocity = jnp.zeros(phi.shape[0])
        theta_effective = self._get_effective_contact_angle_jax(psi, U=U, contact_line_velocity=contact_line_velocity, bottom_velocity_bc=bottom_velocity_bc)
        if psi is None:
            psi = jnp.zeros_like(phi)
        f_1_surface = geometry.f_1_grid[:, 0] if geometry is not None else jnp.zeros(phi.shape[0])
        if self._compiled_geometry_aware_fn is None:
            self._compiled_geometry_aware_fn = jit(_geometry_aware_contact_angle_impl, static_argnums=(6,))
        return self._compiled_geometry_aware_fn(
            phi, dx, dy, geometry.f_1_grid, psi, theta_effective,
            self.use_geometry_aware, f_1_surface
        )
    
    def _get_effective_contact_angle_jax(self, psi, U=None, contact_line_velocity=None, bottom_velocity_bc="no_slip"):
        """Get effective contact angle based on ice phase field and optionally Cox-Voinov dynamics.
        
        Args:
            psi: Ice phase field at substrate (psi[:, 0] for bottom boundary).
                 psi = -1 (water), psi = +1 (ice).
            U: Velocity field (Nx, Ny, 2) (optional, for Cox-Voinov).
            contact_line_velocity: Contact line velocity array (Nx,) (optional, precomputed).
            bottom_velocity_bc: Velocity BC at bottom. With slip_symmetry or navier_slip use U at y=0 (wall);
                with no_slip use U just above wall (j=1) since U[:,0,0]=0 at wall.
        
        Returns:
            Effective contact angle in radians (array if psi/U provided, scalar otherwise).
        """
        # User-facing config is physical liquid-side contact angle.
        # Internal phase-BC relation in this code path is expressed for the
        # opposite-side angle, so convert once here and use consistently.
        theta_water_liq = self.contact_angle * jnp.pi / 180.0
        theta_ice_liq = self.contact_angle_ice * jnp.pi / 180.0
        
        # Base contact angle (static or ice-aware)
        if not self.use_ice_aware or psi is None:
            # Use water contact angle if ice-aware mode is disabled or psi not provided
            theta_base_liq = theta_water_liq
        else:
            # Interpolate contact angle based on ice fraction at substrate
            psi_bottom = psi[:, 0] if psi.ndim == 2 else psi
            ice_fraction = (psi_bottom + 1.0) / 2.0  # Maps [-1, 1] to [0, 1]
            ice_fraction = jnp.clip(ice_fraction, 0.0, 1.0)
            # Linear interpolation between water and ice contact angles
            theta_base_liq = theta_water_liq * (1.0 - ice_fraction) + theta_ice_liq * ice_fraction
        
        # Apply Cox-Voinov dynamic contact angle correction if enabled
        if self.use_cox_voinov:
            if contact_line_velocity is not None:
                # Use precomputed contact line velocity (e.g. from geometry-aware path)
                U_cl = contact_line_velocity
            elif U is not None:
                # Contact line velocity: with slip at bottom use U at wall (y=0, j=0);
                # with no-slip U[:,0,0]=0 so use U just above wall (j=1)
                if bottom_velocity_bc in ("slip_symmetry", "navier_slip"):
                    U_cl = U[:, 0, 0]  # (Nx,) - tangential velocity at wall
                else:
                    U_cl = U[:, 1, 0]  # (Nx,) - velocity just above boundary
            else:
                # No velocity available, use static angle
                U_cl = jnp.zeros_like(theta_base_liq) if jnp.ndim(theta_base_liq) > 0 else 0.0
            
            # Cox-Voinov law (power-law form): θ_app = θ₀ + C * sign(U_cl) * |U_cl|^n
            # Advancing (U_cl > 0): angle increases; receding (U_cl < 0): angle decreases.
            # Coefficient C: config value in degrees per (velocity unit)^n; convert to radians.
            U_cl_safe = jnp.where(jnp.abs(U_cl) < 1e-10, 0.0, U_cl)
            U_magnitude = jnp.abs(U_cl_safe)
            U_power = jnp.sign(U_cl_safe) * (U_magnitude ** self.cox_voinov_exponent)
            C_rad = self.cox_voinov_coefficient * jnp.pi / 180.0
            
            # Apply correction: advancing increases angle, receding decreases
            theta_correction = C_rad * U_power
            
            # Ensure theta_base_liq is array if U_cl is array
            if jnp.ndim(theta_base_liq) == 0 and jnp.ndim(U_cl) > 0:
                theta_base_liq = jnp.broadcast_to(theta_base_liq, U_cl.shape)
            
            theta_liq = theta_base_liq + theta_correction
            # Clamp to physical bounds (0..pi) in liquid-side convention.
            theta_liq = jnp.clip(theta_liq, 0.0, jnp.pi)
        else:
            theta_liq = theta_base_liq

        # Convert liquid-side angle from config to internal angle used by
        # the discrete boundary update relation.
        theta_effective = jnp.pi - theta_liq
        return theta_effective
