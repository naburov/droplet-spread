"""
Simple droplet simulation class - minimal refactoring of existing code.
"""
import numpy as np
import jax.numpy as jnp
from .config import SimulationConfig
from .jax_utils import (
    jax_surface_tension_force, jax_apply_surface_tension_boundary_conditions,
    jax_dx, jax_dy, jax_laplacian, jax_gradient, jax_divergence,
    jax_calculate_reynolds_number, jax_calculate_density, jax_df_2,
    jax_apply_contact_angle_boundary_conditions
)
# Import proven working functions from jax_main
import sys
import os
# Add parent directory to path to import the working jax_main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import jax_main
from .sparse_solver import SparseSolverWrapper
from .plot_utils import create_joint_plot, save_checkpoint, load_checkpoint

class DropletSimulator:
    """Simple droplet spreading simulator."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.setup_grid()
        self.setup_solvers()
        self.initialize_fields()
    
    def setup_grid(self):
        """Setup grid parameters."""
        self.dx = self.config.grid.Lx / self.config.grid.Nx
        self.dy = self.config.grid.Ly / self.config.grid.Ny
        self.Nx = self.config.grid.Nx
        self.Ny = self.config.grid.Ny
    
    def setup_solvers(self):
        """Setup numerical solvers."""
        # Get solver parameters from config
        pressure_config = self.config.solver_params.pressure_solver
        correction_config = self.config.solver_params.correction_solver
        bc = self.config.boundary_conditions
        
        # Pressure solver
        self.pressure_solver = SparseSolverWrapper(
            self.Nx, self.Ny, self.dx, self.dy, pressure_config["backend"],
            {
                'accel': pressure_config["accel"], 
                'tol': pressure_config["tol"],
                'maxiter': pressure_config["maxiter"]
            }
        )
        self.pressure_solver.set_top_boundary_condition(bc.pressure["top"])
        self.pressure_solver.set_bottom_boundary_condition(bc.pressure["bottom"])
        self.pressure_solver.set_left_boundary_condition(bc.pressure["left"])
        self.pressure_solver.set_right_boundary_condition(bc.pressure["right"])
        self.pressure_solver.create_sparse_matrix()
        
        # Velocity correction solver
        self.correction_solver = SparseSolverWrapper(
            self.Nx, self.Ny, self.dx, self.dy, correction_config["backend"],
            {
                'accel': correction_config["accel"], 
                'tol': correction_config["tol"],
                'maxiter': correction_config["maxiter"]
            }
        )
        self.correction_solver.set_top_boundary_condition("neumann")
        self.correction_solver.set_bottom_boundary_condition("neumann")
        self.correction_solver.set_left_boundary_condition("neumann")
        self.correction_solver.set_right_boundary_condition("neumann")
        self.correction_solver.create_sparse_matrix()
    
    def initialize_fields(self):
        """Initialize simulation fields."""
        # Phase field
        self.phi = self.initialize_phase_field()
        
        # Velocity field
        self.U = np.zeros((self.Nx, self.Ny, 2))
        
        # Pressure field
        self.P = np.zeros((self.Nx, self.Ny))
        
        # Time
        self.t = 0.0
        self.step = 0
    
    def initialize_phase_field(self):
        """Initialize phase field with semicircle droplet."""
        x = np.linspace(0, 1, self.Nx)
        y = np.linspace(0, 1, self.Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        center_x, center_y = 0.5, 0
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        phi = -np.tanh((distance - self.config.droplet_radius) * (10/self.config.droplet_radius))
        phi[:, 0] = phi[:, 1]  # Bottom boundary
        
        return jnp.array(phi)
    
    def step_simulation(self):
        """Single simulation step."""
        # Get current parameters
        p = self.config.physical
        
        # Set up global variables for jax_main functions
        jax_main.phi = self.phi
        jax_main.Re1 = p.Re1
        jax_main.Re2 = p.Re2
        jax_main.Pe = p.Pe
        jax_main.epsilon = p.epsilon
        jax_main.contact_angle = p.contact_angle
        jax_main.Fr = p.Fr
        jax_main.g = p.g
        jax_main.atm_pressure = p.atm_pressure
        
        # Time step
        current_dt = self.config.time.dt_initial if self.step < 500 else self.config.time.dt
        
        # CFL condition using proven function
        cfl_computed_dt = jax_main.cfl_dt(self.U[..., 0].max(), self.U[..., 1].max(), self.dx, self.dy, C=self.config.time.cfl_number)
        if cfl_computed_dt != np.inf:
            current_dt = min(cfl_computed_dt, current_dt)
        
        # Surface tension
        surface_tension = jax_surface_tension_force(
            self.phi, p.epsilon, p.We1, p.We2, self.dx, self.dy
        )
        surface_tension = jax_apply_surface_tension_boundary_conditions(
            surface_tension, self.phi, contact_angle=p.contact_angle
        )
        
        # Update velocity using proven function
        self.U = jax_main.update_velocity(
            self.U, self.P, surface_tension, current_dt, 
            self.dx, self.dy, p.rho1, p.rho2, include_gravity=p.include_gravity
        )
        
        # Apply boundary conditions using proven function
        self.U = jax_main.apply_velocity_boundary_conditions(self.U, 0.01, self.dy)
        
        # Pressure projection using proven function
        self.U = jax_main.ppe(self.U, self.dx, self.dy, current_dt, self.correction_solver, div_threshold=self.config.solver_params.divergence_threshold)
        
        # Update phase field using proven function
        self.phi = jax_main.update_phase(
            self.phi, self.U, current_dt, self.dx, self.dy, p.contact_angle
        )
        
        # Update pressure using proven function
        self.P = jax_main.update_pressure(
            surface_tension, self.Nx, self.Ny, self.dx, self.dy, p.rho1, p.rho2, self.pressure_solver
        )
        
        # Update time
        self.t += current_dt
        self.step += 1
    
    
    
    def run(self, output_dir: str = None):
        """Run complete simulation."""
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
        
        while self.t < self.config.time.t_max:
            self.step_simulation()
            
            # Checkpoint
            if self.step % self.config.time.checkpoint_interval == 0:
                if output_dir:
                    self.save_checkpoint(output_dir)
                    self.plot_state(output_dir)
    
    def save_checkpoint(self, output_dir: str):
        """Save simulation state."""
        save_checkpoint(self.step, self.phi, self.U, self.P, directory=output_dir)
    
    def plot_state(self, output_dir: str):
        """Plot current state."""
        p = self.config.physical
        surface_tension = jax_surface_tension_force(
            self.phi, p.epsilon, p.We1, p.We2, self.dx, self.dy
        )
        mass = np.sum(self.phi[self.phi > 0])
        
        # Use plotting parameters from config
        plot_params = self.config.plotting_params
        
        create_joint_plot(
            self.phi, self.U, self.P, surface_tension, 
            self.config.time.dt, self.step, self.dx, self.dy, 
            mass, p.rho1, p.rho2, self.t,
            save_path=f'{output_dir}/joint_plot_step_{self.step}.png',
            plotting_params=plot_params
        )
    
    def get_continuity_info(self):
        """Get continuity information using proven function."""
        divergence, max_div, mean_div = jax_main.check_continuity(self.U, self.dx, self.dy)
        return divergence, max_div, mean_div
