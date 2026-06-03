"""Ice phase field boundary conditions."""

import jax.numpy as jnp
from boundary_conditions.base_bc import BaseBoundaryCondition, BCType


class IcePhaseFieldBoundaryConditions(BaseBoundaryCondition):
    """Ice phase field (ψ) boundary conditions."""
    
    def __init__(self, config=None):
        super().__init__(config, "ice_phase_field")
        # Default to Neumann at all boundaries
        if not self.config.get("boundary_conditions", {}).get("ice_phase_field"):
            self.bc_raw = {"top": "neumann", "bottom": "neumann", "left": "neumann", "right": "neumann"}
            self.bc_types = {b: BCType.NEUMANN for b in self.bc_raw}
        
        ice_params = self.config.get("ice_water_params", {})
        self.T_melt = ice_params.get("T_melt", 273.15)
        
        bc_cfg = self.config.get("boundary_conditions", {}).get("ice_phase_field", {})
        self.fixed_temperature_bc = bc_cfg.get("fixed_temperature_bc", False)
    
    def apply(self, psi, dx: float, dy: float, **kwargs):
        """Apply ice phase field BCs. psi shape: (Nx, Ny)."""
        T = kwargs.get('T')
        
        for b in ["top", "bottom", "left", "right"]:
            psi = self._apply_boundary(psi, b, T)
        return psi
    
    def _apply_boundary(self, psi, b, T):
        bc = self.bc_types[b]
        
        if bc == BCType.NEUMANN:
            return self._neumann(psi, b)
        elif bc == BCType.DIRICHLET:
            if self.fixed_temperature_bc and T is not None:
                # Use temperature to determine ice/water at boundary
                T_boundary = self._get_boundary_slice(T, b)
                psi_value = jnp.where(T_boundary < self.T_melt, 1.0, -1.0)
                return self._set_boundary(psi, b, psi_value)
            else:
                val = self.dirichlet_values.get(b, -1.0)
                return self._dirichlet(psi, b, float(val))
        return psi
    
    def _get_boundary_slice(self, f, b):
        if b == "top":    return f[:, -1]
        if b == "bottom": return f[:, 0]
        if b == "left":   return f[0, :]
        if b == "right":  return f[-1, :]
    
    def _set_boundary(self, f, b, values):
        if b == "top":    return f.at[:, -1].set(values)
        if b == "bottom": return f.at[:, 0].set(values)
        if b == "left":   return f.at[0, :].set(values)
        if b == "right":  return f.at[-1, :].set(values)
        return f
