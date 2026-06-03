"""
Simulation module for droplet spreading simulation.

Contains simulation classes and the run_simulation factory function.
"""

from simulation.initial_conditions import (
    initialize_phase,
    initialize_phase_two_droplets_touching,
    initialize_phase_rectangle,
    get_borders_of_droplet,
    get_y_borders_of_droplet,
)
from simulation.state import SimulationState
from simulation.geometry import Geometry
from simulation.base import BaseSimulation
from simulation.two_phase import TwoPhaseSimulation
from simulation.ice_water import IceWaterSimulation


def run_simulation(config, output_dir=None):
    """Factory function that selects and runs the appropriate simulation.
    
    Args:
        config: Configuration dictionary.
        output_dir: Optional output directory path.
    
    Returns:
        None (simulation runs to completion).
    """
    include_ice_water = config.get("physical_params", {}).get("include_ice_water_transition", False)
    
    if include_ice_water:
        sim = IceWaterSimulation(config, output_dir)
    else:
        sim = TwoPhaseSimulation(config, output_dir)
    
    sim.run()


__all__ = [
    # Factory function
    'run_simulation',
    # Simulation classes
    'BaseSimulation',
    'TwoPhaseSimulation', 
    'IceWaterSimulation',
    # State and geometry
    'SimulationState',
    'Geometry',
    # Initial conditions
    'initialize_phase',
    'initialize_phase_two_droplets_touching',
    'initialize_phase_rectangle',
    'get_borders_of_droplet',
    'get_y_borders_of_droplet'
]
