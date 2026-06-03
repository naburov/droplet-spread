"""Phase field boundary conditions with contact angle support."""

import jax.numpy as jnp
from boundary_conditions.base_bc import BaseBoundaryCondition, BCType
from boundary_conditions.contact_angle_bc import ContactAngleBoundaryCondition


class PhaseFieldBoundaryConditions(BaseBoundaryCondition):
    """Phase field BC with contact angle support."""
    
    BC_ALIASES = {
        **BaseBoundaryCondition.BC_ALIASES,
        "contact_angle": BCType.SPECIAL,
    }
    
    def __init__(self, config=None):
        super().__init__(config, "phase_field")
        # Set phase field defaults
        if not self.config.get("boundary_conditions", {}).get("phase_field"):
            self.bc_raw = {"top": "neumann", "bottom": "contact_angle", "left": "neumann", "right": "neumann"}
            self.bc_types = {b: self._resolve(t) for b, t in self.bc_raw.items()}
        
        # Initialize contact angle BC
        phys = self.config.get("physical_params", {})
        bc_cfg = self.config.get("boundary_conditions", {}).get("phase_field", {})
        ice_params = self.config.get("ice_water_params", {})
        
        # Cox-Voinov parameters
        use_cox_voinov = bc_cfg.get("use_cox_voinov", False)
        cox_voinov_coeff = bc_cfg.get("cox_voinov_coefficient", 1.0)
        cox_voinov_exp = bc_cfg.get("cox_voinov_exponent", 1.0/3.0)
        cox_voinov_velocity_mode = bc_cfg.get("cox_voinov_velocity_mode", "side_aware")
        contact_mask_soft_band = bc_cfg.get("contact_mask_soft_band", 0.0)
        contact_mask_grad_scale = bc_cfg.get("contact_mask_grad_scale", 0.0)
        contact_angle_ghost_law = bc_cfg.get("contact_angle_ghost_law", "wall_energy")
        contact_angle_full_wall = bc_cfg.get("contact_angle_full_wall", False)
        contact_angle_wall_energy_scale = bc_cfg.get("contact_angle_wall_energy_scale", 1.0)
        
        self.contact_angle_bc = ContactAngleBoundaryCondition(
            contact_angle=phys.get("contact_angle", 90),
            method=bc_cfg.get("contact_angle_method", "simple"),
            epsilon=phys.get("epsilon", 0.02),
            contact_angle_ice=ice_params.get("contact_angle_ice"),
            use_ice_aware=phys.get("include_ice_water_transition", False),
            use_geometry_aware=bc_cfg.get("use_geometry_aware", False) or bc_cfg.get("contact_angle_method") == "geometry_aware",
            use_cox_voinov=use_cox_voinov,
            cox_voinov_coefficient=cox_voinov_coeff,
            cox_voinov_exponent=cox_voinov_exp,
            cox_voinov_velocity_mode=cox_voinov_velocity_mode,
            conserve_phi_sum=bc_cfg.get("conserve_phi_sum", True),
            contact_mask_soft_band=contact_mask_soft_band,
            contact_mask_grad_scale=contact_mask_grad_scale,
            contact_angle_ghost_law=contact_angle_ghost_law,
            contact_angle_full_wall=contact_angle_full_wall,
            contact_angle_wall_energy_scale=contact_angle_wall_energy_scale,
        )
    
    def apply(self, phi, dx: float, dy: float, **kwargs):
        """Apply phase field BCs. phi shape: (Nx, Ny). geometry: from state (optional)."""
        psi = kwargs.get('psi')
        U = kwargs.get('U')
        geometry = kwargs.get('geometry')
        bottom_velocity_bc = self.config.get("boundary_conditions", {}).get("velocity", {}).get("bottom", "no_slip")
        if self.bc_raw["bottom"] == "contact_angle":
            phi = self.contact_angle_bc.apply(phi, dx, dy, geometry=geometry, psi=psi, U=U, bottom_velocity_bc=bottom_velocity_bc)
        else:
            phi = self._apply_standard_boundary(phi, "bottom")
        
        # Other boundaries - use standard BCs
        for b in ["top", "left", "right"]:
            phi = self._apply_standard_boundary(phi, b)
        
        return phi
    
    def _apply_standard_boundary(self, phi, b):
        bc = self.bc_types[b]
        if bc == BCType.NEUMANN:
            return self._neumann(phi, b)
        elif bc == BCType.DIRICHLET:
            val = self.dirichlet_values.get(b, 0.0)
            return self._dirichlet(phi, b, float(val) if val else 0.0)
        elif bc == BCType.PERIODIC:
            return self._periodic(phi, b)
        return phi
