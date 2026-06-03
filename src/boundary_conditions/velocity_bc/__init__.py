"""
High-level velocity boundary-condition wrapper.

The project now uses staggered (MAC) velocity layout only.
This wrapper keeps the old construction API and always returns
the staggered implementation.
"""

from __future__ import annotations

from typing import Any, Dict

from .staggered import StaggeredVelocityBoundaryConditions


class VelocityBoundaryConditions:
    """Factory-style wrapper that returns a concrete BC implementation.

    Usage is identical to the old monolithic class:

        bc = VelocityBoundaryConditions(config)
        U_bc = bc.apply(U, dx, dy, psi=..., geometry=...)
    """

    def __new__(cls, config: Dict[str, Any] | None = None):
        cfg = config or {}
        return StaggeredVelocityBoundaryConditions(cfg)

