"""Base boundary condition class with common BC logic."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional
import jax.numpy as jnp


class BCType(Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    PERIODIC = "periodic"
    ROBIN = "robin"
    SPECIAL = "special"


class BaseBoundaryCondition(ABC):
    """Base class for all boundary conditions."""
    
    # Subclasses extend this mapping
    BC_ALIASES: Dict[str, BCType] = {
        "dirichlet": BCType.DIRICHLET,
        "neumann": BCType.NEUMANN,
        "zero_flux": BCType.NEUMANN,
        "periodic": BCType.PERIODIC,
        "robin": BCType.ROBIN,
    }
    
    def __init__(self, config: Optional[dict] = None, field_name: str = ""):
        self.config = config or {}
        self.field_name = field_name
        bc_config = self.config.get("boundary_conditions", {}).get(field_name, {})
        
        # Parse boundary types
        self.bc_raw = {
            "top": bc_config.get("top", "neumann"),
            "bottom": bc_config.get("bottom", "neumann"),
            "left": bc_config.get("left", "neumann"),
            "right": bc_config.get("right", "neumann"),
        }
        self.bc_types = {b: self._resolve(t) for b, t in self.bc_raw.items()}
        self.dirichlet_values = bc_config.get("dirichlet_values", {})
    
    def _resolve(self, bc_type: str) -> BCType:
        """Resolve config string to BCType."""
        return self.BC_ALIASES.get(bc_type, BCType.SPECIAL)
    
    @abstractmethod
    def apply(self, field, dx: float, dy: float, **kwargs):
        """Apply BCs to field. Must be implemented by subclasses."""
        pass
    
    def apply_boundary_conditions(self, field, *args, **kwargs):
        """Alias for apply() - for backward compatibility."""
        return self.apply(field, *args, **kwargs)
    
    # Standard BC helpers for scalar fields
    def _dirichlet(self, f, boundary: str, value: float):
        if boundary == "top":    return f.at[:, -1].set(value)
        if boundary == "bottom": return f.at[:, 0].set(value)
        if boundary == "left":   return f.at[0, :].set(value)
        if boundary == "right":  return f.at[-1, :].set(value)
        return f
    
    def _neumann(self, f, boundary: str):
        if boundary == "top":    return f.at[:, -1].set(f[:, -2])
        if boundary == "bottom": return f.at[:, 0].set(f[:, 1])
        if boundary == "left":   return f.at[0, :].set(f[1, :])
        if boundary == "right":  return f.at[-1, :].set(f[-2, :])
        return f
    
    def _periodic(self, f, boundary: str):
        if boundary == "left":   return f.at[0, :].set(f[-2, :])
        if boundary == "right":  return f.at[-1, :].set(f[1, :])
        if boundary == "bottom": return f.at[:, 0].set(f[:, -2])
        if boundary == "top":    return f.at[:, -1].set(f[:, 1])
        return f
    
    def apply_standard_scalar(self, f, dx: float, dy: float):
        """Apply standard BCs (Dirichlet/Neumann/Periodic) to scalar field."""
        for b in ["top", "bottom", "left", "right"]:
            bc = self.bc_types[b]
            if bc == BCType.DIRICHLET:
                val = self.dirichlet_values.get(b, 0.0)
                f = self._dirichlet(f, b, float(val) if val is not None else 0.0)
            elif bc == BCType.NEUMANN:
                f = self._neumann(f, b)
            elif bc == BCType.PERIODIC:
                f = self._periodic(f, b)
        return f
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.bc_raw})"
