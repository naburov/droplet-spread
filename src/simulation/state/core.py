"""
Simulation state management.

This module provides the SimulationState class that contains all fields,
parameters, and solvers for the simulation.
"""

import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional
from simulation.geometry import Geometry
from simulation.state.specs import SimulationContext
from simulation.state.bundles import SolverBundle, BCBundle
from physics.phase_field import BasePhaseFieldSolver
from physics.fluid_dynamics import FluidDynamicsSolver
from physics.surface_tension import SurfaceTensionSolver
from physics.pressure import PressureSolver
from solvers.sparse_solver import SparseSolverWrapper
from boundary_conditions.velocity_bc import VelocityBoundaryConditions
from boundary_conditions.pressure_bc import PressureBoundaryConditions
from boundary_conditions.phase_field_bc import PhaseFieldBoundaryConditions


@dataclass
class SimulationState:
    """Unified simulation state - contains all fields, parameters, and solvers.

    This is the single source of truth for the simulation.
    All update functions operate on this state (in-place modifications).
    """
    # Fields
    phi: jnp.ndarray  # Phase field (Nx, Ny)
    U: jnp.ndarray  # Velocity field (Nx, Ny, 2) - collocated view
    P: jnp.ndarray  # Pressure field (Nx, Ny)
    psi: jnp.ndarray  # Ice phase field (Nx, Ny) - zeros if not used
    T: jnp.ndarray  # Temperature field (Nx, Ny) - zeros if not used

    # Geometry (always present)
    geometry: Geometry

    # Grid parameters
    Nx: int
    Ny: int
    dx: float
    dy: float
    Lx: float
    Ly: float

    # Physical parameters
    rho1: float
    rho2: float
    Re1: float
    Re2: float
    We1: float
    We2: float
    Pe: float
    epsilon: float
    contact_angle: float
    g: float
    atm_pressure: float
    Fr: float

    # Time parameters
    t: float
    dt: float
    step: int

    # Flags
    include_ice_water: bool
    include_gravity: bool

    # Solvers (stored in state)
    phase_solver: BasePhaseFieldSolver
    fluid_solver: FluidDynamicsSolver
    surface_tension_solver: SurfaceTensionSolver
    pressure_solver: PressureSolver
    correction_solver: SparseSolverWrapper
    pressure_linear_solver: SparseSolverWrapper

    # Boundary condition managers
    velocity_bc: VelocityBoundaryConditions
    pressure_bc: PressureBoundaryConditions
    phase_field_bc: PhaseFieldBoundaryConditions

    # Optional grouped metadata/dependencies for cleaner APIs.
    context: Optional[SimulationContext] = None
    solver_bundle: Optional[SolverBundle] = None
    bc_bundle: Optional[BCBundle] = None

    # Cached quantities (computed on demand)
    _surface_tension: Optional[jnp.ndarray] = None
    _density: Optional[jnp.ndarray] = None

    # Staggered (MAC) face velocities.
    # (u_face, v_face) hold the canonical predictor/corrector velocities on faces,
    # and U is treated as a derived, cell-centred view (phase field, diagnostics).
    u_face: Optional[jnp.ndarray] = None  # (Nx+1, Ny)
    v_face: Optional[jnp.ndarray] = None  # (Nx, Ny+1)

    def __post_init__(self):
        """Ensure staggered faces are always initialized from collocated velocity."""
        self.ensure_face_velocities()

    def ensure_face_velocities(self):
        """Ensure staggered face velocities exist and are shape-consistent."""
        from numerics.staggered_utils import to_staggered

        if self.u_face is None or self.v_face is None:
            self.u_face, self.v_face = to_staggered(self.U)
            return

        expected_u = (self.Nx + 1, self.Ny)
        expected_v = (self.Nx, self.Ny + 1)
        if tuple(self.u_face.shape) != expected_u or tuple(self.v_face.shape) != expected_v:
            self.u_face, self.v_face = to_staggered(self.U)

    def sync_collocated_from_faces(self):
        """Update collocated velocity view from canonical staggered faces."""
        from numerics.staggered_utils import to_collocated

        self.ensure_face_velocities()
        self.U = to_collocated(self.u_face, self.v_face)

    def set_face_velocities(self, u_face, v_face, sync_collocated: bool = True):
        """Set canonical face velocities with optional collocated-view sync."""
        self.u_face = u_face
        self.v_face = v_face
        if sync_collocated:
            self.sync_collocated_from_faces()

    def compute_surface_tension(self):
        """Compute and cache surface tension force."""
        if self._surface_tension is None:
            self._surface_tension = self.surface_tension_solver.calculate_force(
                self.phi, self.dx, self.dy, self.geometry, use_jax=True, interface_mask=None
            )
            self._surface_tension = self.surface_tension_solver.apply_boundary_conditions(
                self._surface_tension, self.phi, use_jax=True,
                geometry=self.geometry, dx=self.dx, dy=self.dy
            )
        return self._surface_tension

    def compute_density(self):
        """Compute and cache density field."""
        if self._density is None:
            from physics.properties import jax_calculate_density
            self._density = jax_calculate_density(self.phi, self.rho1, self.rho2)
        return self._density

    def invalidate_cache(self):
        """Invalidate cached quantities after field updates."""
        self._surface_tension = None
        self._density = None

    @staticmethod
    def from_config(config, restart_from=None):
        """Create initial state from configuration."""
        from simulation.state.factory import create_state_from_config

        return create_state_from_config(config=config, restart_from=restart_from)
