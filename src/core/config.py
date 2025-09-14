"""
Simple configuration management for droplet simulation.
"""
import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class PhysicalParams:
    """Physical parameters for the simulation."""
    rho1: float = 1.0
    rho2: float = 1000.0
    Re1: float = 10.0
    Re2: float = 150.0
    We1: float = 1.0
    We2: float = 10000.0
    Pe: float = 1.0
    epsilon: float = 0.02
    alpha: float = 1.0
    phase_penalty: float = 1000.0
    contact_angle: float = 120.0
    include_gravity: bool = True
    g: float = -10.0
    atm_pressure: float = -10000.0
    Fr: float = 1.0

@dataclass
class GridParams:
    """Grid parameters."""
    Lx: float = 1.0
    Ly: float = 1.0
    Nx: int = 128
    Ny: int = 128

@dataclass
class TimeParams:
    """Time integration parameters."""
    dt: float = 0.00025
    t_max: float = 1.0
    checkpoint_interval: int = 5
    dt_initial: float = 0.00025
    cfl_number: float = 0.01

@dataclass
class BoundaryConditions:
    """Boundary condition settings."""
    pressure: dict = None
    velocity: dict = None
    phase_field: dict = None
    
    def __post_init__(self):
        if self.pressure is None:
            self.pressure = {"top": "dirichlet", "bottom": "dirichlet", "left": "dirichlet", "right": "dirichlet"}
        if self.velocity is None:
            self.velocity = {"top": "neumann", "bottom": "no_slip", "left": "periodic", "right": "periodic"}
        if self.phase_field is None:
            self.phase_field = {"top": "neumann", "bottom": "contact_angle", "left": "periodic", "right": "periodic"}

@dataclass
class SolverParams:
    """Solver configuration parameters."""
    pressure_solver: dict = None
    correction_solver: dict = None
    divergence_threshold: float = 0.05
    max_correction_iterations: int = 100
    
    def __post_init__(self):
        if self.pressure_solver is None:
            self.pressure_solver = {"backend": "pyamg", "accel": "bicgstab", "tol": 0.05, "maxiter": 10000}
        if self.correction_solver is None:
            self.correction_solver = {"backend": "pyamg", "accel": "bicgstab", "tol": 0.05, "maxiter": 10000}

@dataclass
class PlottingParams:
    """Plotting and visualization parameters."""
    figure_size: list = None
    dpi: int = 100
    colormap: str = "viridis"
    save_format: str = "png"
    show_velocity_vectors: bool = True
    vector_density: int = 8
    vector_scale: int = 50
    show_contours: bool = True
    contour_levels: int = 20
    show_colorbar: bool = True
    title_fontsize: int = 16
    label_fontsize: int = 14
    tick_fontsize: int = 12
    
    def __post_init__(self):
        if self.figure_size is None:
            self.figure_size = [18, 14]

@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    physical: PhysicalParams
    grid: GridParams
    time: TimeParams
    boundary_conditions: BoundaryConditions
    solver_params: SolverParams
    plotting_params: PlottingParams
    droplet_radius: float = 0.2
    restart_from: Optional[str] = None

def load_config(config_path: Optional[str] = None) -> SimulationConfig:
    """Load configuration from JSON file or use defaults."""
    if config_path:
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        physical = PhysicalParams(**data.get('physical_params', {}))
        grid = GridParams(**data.get('grid_params', {}))
        time = TimeParams(**data.get('time_params', {}))
        
        # Load new parameter sections
        boundary_conditions = BoundaryConditions(**data.get('boundary_conditions', {}))
        solver_params = SolverParams(**data.get('solver_params', {}))
        plotting_params = PlottingParams(**data.get('plotting_params', {}))
        
        return SimulationConfig(
            physical=physical,
            grid=grid,
            time=time,
            boundary_conditions=boundary_conditions,
            solver_params=solver_params,
            plotting_params=plotting_params,
            droplet_radius=data.get('initial_conditions', {}).get('droplet_radius', 0.2),
            restart_from=data.get('restart', {}).get('restart_from', None)
        )
    else:
        return SimulationConfig(
            physical=PhysicalParams(),
            grid=GridParams(),
            time=TimeParams(),
            boundary_conditions=BoundaryConditions(),
            solver_params=SolverParams(),
            plotting_params=PlottingParams()
        )
