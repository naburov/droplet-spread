"""Grouped solver/BC dependencies for simulation state."""

from dataclasses import dataclass

from physics.phase_field import BasePhaseFieldSolver
from physics.fluid_dynamics import FluidDynamicsSolver
from physics.surface_tension import SurfaceTensionSolver
from physics.pressure import PressureSolver
from solvers.sparse_solver import SparseSolverWrapper
from boundary_conditions.velocity_bc import VelocityBoundaryConditions
from boundary_conditions.pressure_bc import PressureBoundaryConditions
from boundary_conditions.phase_field_bc import PhaseFieldBoundaryConditions


@dataclass(frozen=True)
class SolverBundle:
    phase_solver: BasePhaseFieldSolver
    fluid_solver: FluidDynamicsSolver
    surface_tension_solver: SurfaceTensionSolver
    pressure_solver: PressureSolver
    correction_solver: SparseSolverWrapper
    pressure_linear_solver: SparseSolverWrapper


@dataclass(frozen=True)
class BCBundle:
    velocity_bc: VelocityBoundaryConditions
    pressure_bc: PressureBoundaryConditions
    phase_field_bc: PhaseFieldBoundaryConditions
