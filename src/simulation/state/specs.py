"""Typed simulation metadata specs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GridSpec:
    Nx: int
    Ny: int
    dx: float
    dy: float
    Lx: float
    Ly: float


@dataclass(frozen=True)
class PhysicalParams:
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


@dataclass(frozen=True)
class FeatureFlags:
    include_ice_water: bool
    include_gravity: bool


@dataclass(frozen=True)
class SimulationContext:
    grid: GridSpec
    physical: PhysicalParams
    flags: FeatureFlags
    geometry_type: str

