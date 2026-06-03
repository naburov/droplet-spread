"""Pressure boundary conditions. Curvilinear (x, eta): surface at eta=0 (j=0)."""

import jax.numpy as jnp
from boundary_conditions.base_bc import BaseBoundaryCondition, BCType


class PressureBoundaryConditions(BaseBoundaryCondition):
    """Pressure BC with open boundary support."""
    
    BC_ALIASES = {
        **BaseBoundaryCondition.BC_ALIASES,
        "open": BCType.DIRICHLET,
    }
    
    def __init__(self, config=None):
        super().__init__(config, "pressure")
        # Set pressure defaults
        if not self.config.get("boundary_conditions", {}).get("pressure"):
            self.bc_raw = {"top": "open", "bottom": "neumann", "left": "open", "right": "open"}
            self.bc_types = {b: self._resolve(t) for b, t in self.bc_raw.items()}
        
        bc_cfg = self.config.get("boundary_conditions", {}).get("pressure", {})
        self.open_pressure = bc_cfg.get("open_pressure", 0.0)
        self.use_geometry = bc_cfg.get("use_geometry", False)
    
    def apply(self, p, dx: float, dy: float, **kwargs):
        """Apply pressure BCs. p shape: (Nx, Ny). Surface at eta=0 (j=0)."""
        # Bottom (surface at j=0)
        if self.bc_raw["bottom"] in ["neumann", "zero_flux"]:
            p = self._neumann(p, "bottom")
        elif self.bc_raw["bottom"] == "open":
            p = self._dirichlet(p, "bottom", self.open_pressure)
        
        # Other boundaries
        for b in ["top", "left", "right"]:
            if self.bc_raw[b] == "open":
                p = self._dirichlet(p, b, self.open_pressure)
            elif self.bc_types[b] == BCType.NEUMANN:
                p = self._neumann(p, b)
            elif self.bc_types[b] == BCType.DIRICHLET:
                val = self.dirichlet_values.get(b, self.open_pressure)
                p = self._dirichlet(p, b, float(val) if val else self.open_pressure)
        
        return p

    def get_implicit_bc_types(self):
        """Get BC types for matrix solver."""
        return {b: "dirichlet" if self.bc_raw[b] in ["dirichlet", "open"] else "neumann"
                for b in ["top", "bottom", "left", "right"]}
