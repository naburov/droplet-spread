"""
Base simulation class with infrastructure methods.

This abstract class handles common simulation infrastructure:
- Output directory management
- Telemetry logging
- Checkpointing
- Visualization
"""

import gc
import os
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation.state import SimulationState
from visualization.plotting import create_joint_plot
from visualization.checkpointing import save_checkpoint
from visualization.pyvista_utils import PYVISTA_AVAILABLE, create_joint_plot_pyvista_full
from visualization.telemetry import TelemetryLogger
from visualization.plot_telemetry import (
    plot_statistics,
    plot_boundary_statistics,
    plot_ppe_updates,
    plot_contact_line_dynamics,
)
from simulation.initial_conditions import get_borders_of_droplet, get_y_borders_of_droplet


class BaseSimulation(ABC):
    """Abstract base class for simulations.
    
    Handles infrastructure: directories, telemetry, checkpointing.
    Subclasses implement the physics in step().
    """
    
    def __init__(self, config, output_dir=None):
        """Initialize simulation infrastructure.
        
        Args:
            config: Configuration dictionary.
            output_dir: Optional output directory path.
        """
        self.config = config
        
        # Time parameters
        self.t_max = config["time_params"]["t_max"]
        self.checkpoint_interval = config["time_params"]["checkpoint_interval"]
        self.dt_initial = config["time_params"]["dt_initial"]
        self.dt_normal = config["time_params"]["dt"]
        self.cfl_number = config["time_params"].get("cfl_number", 0.01)
        self.capillary_cfl_number = config["time_params"].get("capillary_cfl_number", self.cfl_number)
        
        # PPE parameters
        ppe_params = config.get("solver_params", {}).get("ppe", {})
        self.mean_div_threshold = ppe_params.get("mean_div_threshold", 0.1)
        self.max_div_threshold = ppe_params.get("max_div_threshold", 0.05)
        self.div_threshold = ppe_params.get("div_threshold", 0.05)
        
        # Restart handling
        restart_from = config["restart"]["restart_from"]
        self.restart_from = None if restart_from == "None" else restart_from
        
        # Create output directories
        if output_dir:
            self.output_dir = output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"experiment_{timestamp}"
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.visualization_dir = os.path.join(self.output_dir, "visualization")
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Save configuration
        params_file = os.path.join(self.output_dir, "simulation_parameters.json")
        with open(params_file, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Simulation parameters saved to {params_file}")
        
        # Initialize state
        self.state = SimulationState.from_config(config, restart_from=self.restart_from)
        
        # Initialize telemetry (subclasses may override include_ice_water)
        self.include_ice_water = self.state.include_ice_water
        self.telemetry_logger = TelemetryLogger(
            self.output_dir, include_ice_water=self.include_ice_water, config=config
        )
        print(f"Telemetry logging initialized in {self.output_dir}")
        
        # Timing
        self.step_times = []
    
    def setup(self):
        """Setup before main loop: visualizations, diagnostics."""
        print("Initialized flat geometry")
        if self.restart_from is None:
            print("Applying contact angle boundary conditions to initial phase field...")
            self.state.phi = self.state.phase_field_bc.apply_boundary_conditions(
                self.state.phi, self.state.dx, self.state.dy, use_jax=True,
                geometry=self.state.geometry
            )
            print("Contact angle BC applied")
        
        # Initial phase field visualization
        plt.imshow(self.state.phi.T, extent=[0, self.state.Lx, 0, self.state.Ly], origin='lower', cmap='viridis')
        plt.colorbar(label='Phase Field (phi)')
        plt.title('Initial Phase Field')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig(f'{self.output_dir}/initial_phase_field.png', bbox_inches='tight')
        plt.clf()
        
        # Create initial joint plot (PyVista when available and enabled, else matplotlib)
        use_pyvista = (
            PYVISTA_AVAILABLE
            and self.config.get("visualization", {}).get("use_pyvista", True)
            and not os.environ.get("DISABLE_PYVISTA")
        )
        viz_backend = "PyVista" if use_pyvista else "matplotlib"
        print(f"Creating initial state visualization ({viz_backend})...")
        self._create_joint_plot(step=0)
        print(f"Initial state plot saved to {self.visualization_dir}/joint_plot_step_0.png")
        
        # First-step diagnostics
        self._create_first_step_diagnostics()
        
        print("Starting simulation...")
    
    def run(self):
        """Main simulation loop."""
        self.setup()
        
        while self.state.t < self.t_max:
            start_time = time.time()
            
            # Set time step
            self.state.dt = self.dt_initial if self.state.step < 500 else self.dt_normal
            
            # Physics step (implemented by subclasses)
            self.step()
            
            # Release solver workspace and intermediate refs to reduce memory retention
            self._after_step_cleanup()
            if self.state.step % 50 == 0:
                gc.collect()
            
            # Timing
            end_time = time.time()
            self.step_times.append(end_time - start_time)
            
            # Telemetry
            self._log_telemetry()
            
            # Progress output
            if self.state.step % 10 == 0:
                self._print_progress()
            
            # Checkpointing and visualization
            if self.state.step % self.checkpoint_interval == 0:
                self._create_joint_plot(step=self.state.step)
                self._generate_telemetry_plots()
                self._save_checkpoint()
                print(f"  Telemetry plots updated at step {self.state.step}")
            
            self.state.step += 1
        
        self.finalize()
    
    @abstractmethod
    def step(self):
        """Perform one physics step. Implemented by subclasses."""
        pass

    def _after_step_cleanup(self):
        """Release solver workspace and intermediate tensors after each step. Override in subclasses."""
        pass

    def finalize(self):
        """Cleanup after simulation: final plots."""
        print("Simulation completed!")
        print("\nGenerating final telemetry plots...")
        self._generate_telemetry_plots()
        print(f"Final telemetry plots saved to {self.visualization_dir}/")
    
    # ==================== Helper Methods ====================
    
    def _create_joint_plot(self, step):
        """Create joint visualization plot (PyVista when available, else matplotlib)."""
        surface_tension = self.state.compute_surface_tension()
        rho = self.state.compute_density()
        mass = float(np.sum(np.array(rho)[np.array(self.state.phi) < 0]) * self.state.dx * self.state.dy)
        save_path = f'{self.visualization_dir}/joint_plot_step_{step}.png'
        kwargs = dict(
            phi=self.state.phi,
            U=self.state.U,
            P=self.state.P,
            surface_tension=surface_tension,
            dt=self.state.dt,
            step=step,
            dx=self.state.dx,
            dy=self.state.dy,
            mass=mass,
            rho1=self.state.rho1,
            rho2=self.state.rho2,
            save_path=save_path,
            psi=self.state.psi if self.include_ice_water else None,
            T=self.state.T if self.include_ice_water else None,
            include_ice_water=self.include_ice_water,
            Lx=self.state.Lx,
            Ly=self.state.Ly,
            geometry=self.state.geometry,
        )
        use_pyvista = (
            PYVISTA_AVAILABLE
            and self.config.get("visualization", {}).get("use_pyvista", True)
            and not os.environ.get("DISABLE_PYVISTA")
        )
        if use_pyvista:
            create_joint_plot_pyvista_full(**kwargs)
        else:
            create_joint_plot(**kwargs)
    
    def _create_first_step_diagnostics(self):
        """Create first-step diagnostics."""
        print("Creating first-step diagnostics...")
        try:
            from diagnostics.first_step_visualization import create_first_step_visualizations
            diagnostics_dir = os.path.join(self.output_dir, 'diagnostics')
            os.makedirs(diagnostics_dir, exist_ok=True)
            # Need to pass config path - construct from output_dir
            config_path = os.path.join(self.output_dir, "simulation_parameters.json")
            stats = create_first_step_visualizations(config_path, diagnostics_dir)
            print(f"First-step diagnostics saved to {diagnostics_dir}/first_step_comprehensive.png")
            print(f"  Max divergence: {stats['max_div_initial']:.2e} → {stats['max_div_final']:.2e}")
        except Exception as e:
            print(f"Warning: Could not create first-step diagnostics: {e}")
    
    def _log_telemetry(self):
        """Log telemetry data. Can be extended by subclasses."""
        start_of_droplet, end_of_droplet = get_borders_of_droplet(self.state.phi)
        bottom_of_droplet, top_of_droplet = get_y_borders_of_droplet(self.state.phi)
        rho = self.state.compute_density()
        mass = np.sum(rho[self.state.phi < 0]) * self.state.dx * self.state.dy
        
        # Store for subclasses and progress output
        self._last_mass = mass
        self._last_droplet_bounds = (start_of_droplet, end_of_droplet, bottom_of_droplet, top_of_droplet)
        
        surface_tension = self.state.compute_surface_tension()
        
        # Get curvature stats if available (computed by subclass)
        curvature_max = getattr(self, '_last_curvature_max', None)
        curvature_mean = getattr(self, '_last_curvature_mean', None)
        
        self.telemetry_logger.log_statistics(
            step=self.state.step,
            time=self.state.t,
            dt=self.state.dt,
            phi=self.state.phi,
            U=self.state.U,
            P=self.state.P,
            surface_tension=surface_tension,
            mass=mass,
            droplet_start=start_of_droplet,
            droplet_end=end_of_droplet,
            droplet_bottom=bottom_of_droplet,
            droplet_top=top_of_droplet,
            max_div=self._last_max_div,
            mean_div=self._last_mean_div,
            psi=self.state.psi if self.include_ice_water else None,
            T=self.state.T if self.include_ice_water else None,
            geometry=self.state.geometry,
            dx=self.state.dx,
            curvature_max=curvature_max,
            curvature_mean=curvature_mean
        )
        
        self.telemetry_logger.log_boundary_statistics(
            step=self.state.step,
            time=self.state.t,
            phi=self.state.phi,
            U=self.state.U,
            P=self.state.P,
            surface_tension=surface_tension,
            psi=self.state.psi if self.include_ice_water else None,
            T=self.state.T if self.include_ice_water else None
        )
        
        self.telemetry_logger.log_ppe_update(
            step=self.state.step,
            time=self.state.t,
            ppe_applied=self._last_ppe_info['applied'],
            ppe_iterations=self._last_ppe_info['iterations'],
            div_before_max=self._last_ppe_info['div_before_max'],
            div_before_mean=self._last_ppe_info['div_before_mean'],
            div_after_max=self._last_ppe_info['div_after_max'],
            div_after_mean=self._last_ppe_info['div_after_mean'],
            div_threshold=self.div_threshold,
            max_div_threshold=self.max_div_threshold,
            mean_div_threshold=self.mean_div_threshold
        )
    
    def _print_progress(self):
        """Print progress information."""
        start, end, bottom, top = self._last_droplet_bounds
        print(f"Step {self.state.step}, Time {self.state.t:.2f}")
        print(f"Min/Max of U: {self.state.U.min():.4f} / {self.state.U.max():.4f}")
        print(f"Min/Max of P: {self.state.P.min():.4f} / {self.state.P.max():.4f}")
        print(f"Min/Max of phi: {self.state.phi.min():.4f} / {self.state.phi.max():.4f}")
        print(f"Continuity check - Max |div(U)|: {self._last_max_div:.6f}, Mean |div(U)|: {self._last_mean_div:.6f}")
        print(f"Time per step: {np.mean(self.step_times):.6f}")
        print(f"Droplet mass: {self._last_mass}")
        print(f"Start/end of droplet: {start} - {end}")
    
    def _generate_telemetry_plots(self):
        """Generate telemetry plots with optional baseline comparison."""
        baseline_dir = self.config.get('baseline_experiment', None)
        
        def load_baseline(filename):
            if not baseline_dir or not os.path.exists(baseline_dir):
                return None
            try:
                import pandas as pd
                path = os.path.join(baseline_dir, filename)
                return pd.read_csv(path) if os.path.exists(path) else None
            except ImportError:
                return None
        
        plot_statistics(
            os.path.join(self.output_dir, 'statistics.csv'), self.visualization_dir,
            baseline_df=load_baseline('statistics.csv'))
        plot_boundary_statistics(
            os.path.join(self.output_dir, 'boundary_statistics.csv'), self.visualization_dir,
            baseline_df=load_baseline('boundary_statistics.csv'))
        plot_ppe_updates(
            os.path.join(self.output_dir, 'ppe_updates.csv'), self.visualization_dir,
            baseline_df=load_baseline('ppe_updates.csv'))
        plot_contact_line_dynamics(self.output_dir, self.output_dir)

    def _save_checkpoint(self):
        """Save checkpoint."""
        save_checkpoint(
            self.state.step, self.state.phi, self.state.U, self.state.P,
            directory=self.checkpoint_dir,
            psi=self.state.psi if self.include_ice_water else None,
            T=self.state.T if self.include_ice_water else None,
            u_face=getattr(self.state, "u_face", None),
            v_face=getattr(self.state, "v_face", None),
        )
