"""Temperature boundary conditions."""

import jax.numpy as jnp
from boundary_conditions.base_bc import BaseBoundaryCondition, BCType


class TemperatureBoundaryConditions(BaseBoundaryCondition):
    """Temperature BC with adiabatic and robin support."""
    
    BC_ALIASES = {
        **BaseBoundaryCondition.BC_ALIASES,
        "adiabatic": BCType.NEUMANN,
    }
    
    def __init__(self, config=None):
        super().__init__(config, "temperature")
        # Set temperature defaults
        if not self.config.get("boundary_conditions", {}).get("temperature"):
            self.bc_raw = {"top": "dirichlet", "bottom": "dirichlet", "left": "neumann", "right": "neumann"}
            self.bc_types = {b: self._resolve(t) for b, t in self.bc_raw.items()}
        
        bc_cfg = self.config.get("boundary_conditions", {}).get("temperature", {})
        self.robin_coefficients = bc_cfg.get("robin_coefficients", {})
    
    def apply(self, T, dx: float, dy: float, **kwargs):
        """Apply temperature BCs. T shape: (Nx, Ny)."""
        for b in ["top", "bottom", "left", "right"]:
            T = self._apply_boundary(T, b, dx, dy)
        return T
    
    def _apply_boundary(self, T, b, dx, dy):
        bc = self.bc_types[b]
        raw = self.bc_raw[b]
        
        if bc == BCType.DIRICHLET:
            val = self._get_temp_value(b)
            return self._dirichlet(T, b, val)
        elif bc == BCType.NEUMANN or raw == "adiabatic":
            return self._neumann(T, b)
        elif bc == BCType.ROBIN or raw == "robin":
            dn = dy if b in ["top", "bottom"] else dx
            return self._robin(T, b, dn)
        return T
    
    def _get_temp_value(self, b):
        val = self.dirichlet_values.get(b)
        if val is None:
            return 273.15 if b == "bottom" else 293.15
        if hasattr(val, '__len__') and not isinstance(val, str):
            return float(val[0]) if len(val) > 0 else 293.15
        return float(val)
    
    def _robin(self, T, b, dn):
        """Robin BC: h*(T - T_ambient) = -k*dT/dn."""
        robin = self.robin_coefficients.get(b, {})
        h = robin.get("h", 10.0)
        T_amb = robin.get("T_ambient", 293.15)
        coeff = h * dn / (1 + h * dn) if (1 + h * dn) != 0 else 0.0
        
        if b == "top":    return T.at[:, -1].set(T[:, -2] + (T_amb - T[:, -2]) * coeff)
        if b == "bottom": return T.at[:, 0].set(T[:, 1] + (T_amb - T[:, 1]) * coeff)
        if b == "left":   return T.at[0, :].set(T[1, :] + (T_amb - T[1, :]) * coeff)
        if b == "right":  return T.at[-1, :].set(T[-2, :] + (T_amb - T[-2, :]) * coeff)
        return T
