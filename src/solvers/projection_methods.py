"""
Projection methods for incompressible flow.

Re-exports from the unified PPE module for backward compatibility.
"""

from solvers.ppe import ppe_solve

__all__ = ['ppe', 'ppe_solve']

# Alias for backward compatibility
ppe = ppe_solve
