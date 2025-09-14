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
        
        # Time step
        current_dt = self.config.time.dt_initial if self.step < 500 else self.config.time.dt
        
        # CFL condition
        cfl_dt = self.cfl_condition()
        if cfl_dt != np.inf:
            current_dt = min(cfl_dt, current_dt)
        
        # Surface tension
        surface_tension = jax_surface_tension_force(
            self.phi, p.epsilon, p.We1, p.We2, self.dx, self.dy
        )
        surface_tension = jax_apply_surface_tension_boundary_conditions(
            surface_tension, self.phi, contact_angle=p.contact_angle
        )
        
        # Update velocity
        self.U = self._update_velocity(
            self.U, self.P, surface_tension, current_dt, 
            self.dx, self.dy, p.rho1, p.rho2, p.include_gravity
        )
        
        # Apply boundary conditions
        self.U = self._apply_velocity_boundary_conditions(self.U, 0.01, self.dy)
        
        # Pressure projection
        self.U = self.ppe(self.U, self.dx, self.dy, current_dt)
        
        # Update phase field
        self.phi = self._update_phase_field(
            self.phi, self.U, current_dt, self.dx, self.dy,
            p.contact_angle, p.rho1, p.rho2, p.Pe, p.epsilon
        )
        
        # Update pressure
        self.P = self._update_pressure(
            surface_tension, self.Nx, self.Ny, self.dx, self.dy, p.rho1, p.rho2
        )
        
        # Update time
        self.t += current_dt
        self.step += 1
    
    def cfl_condition(self, C=None):
        """Calculate CFL-limited time step."""
        if C is None:
            C = self.config.time.cfl_number
        u_max = np.max(np.abs(self.U[..., 0]))
        v_max = np.max(np.abs(self.U[..., 1]))
        return C / (u_max/self.dx + v_max/self.dy) if (u_max + v_max) > 0 else np.inf
    
    def _update_velocity(self, U, P, surface_tension, dt, dx, dy, rho1, rho2, include_gravity):
        """Update velocity field."""
        # Get physical parameters
        p = self.config.physical
        
        # Calculate the Reynolds number and density
        Re = jax_calculate_reynolds_number(self.phi, p.Re1, p.Re2)
        rho = jax_calculate_density(self.phi, rho1, rho2)
        rho_stacked = jnp.stack([rho, rho], axis=-1) + 1e-6

        # Calculate gradients and terms
        grad_U = jax_gradient(U, dx, dy)
        p_grad = jax_gradient(P, dx, dy)
        
        # Calculate viscous term with proper scaling
        viscous_term = self._compute_viscous_term(U, dx, dy, Re)
        
        # Calculate convective term (in conservative form)
        convective_term = jnp.stack(
            [
                U[..., 0] * grad_U[..., 0, 0] + U[..., 1] * grad_U[..., 0, 1],
                U[..., 0] * grad_U[..., 1, 0] + U[..., 1] * grad_U[..., 1, 1]
            ],
            axis=-1
        )

        # Combine terms with proper density scaling
        rhs_U = (
            -p_grad / rho_stacked +  # Pressure term
            viscous_term / rho_stacked +  # Viscous term
            -surface_tension / rho_stacked +  # Surface tension
            -convective_term  # Convective term (already includes velocity)
        )

        # Add gravity if included
        if include_gravity:
            rhs_U = rhs_U + (1 / p.Fr) * jnp.stack([jnp.zeros_like(U[..., 0]), -jnp.ones_like(U[..., 1])], axis=-1)

        # Update velocity field using explicit Euler
        U = U + dt * rhs_U
        
        return U
    
    def _update_phase_field(self, phi, U, dt, dx, dy, contact_angle, rho1, rho2, Pe, epsilon):
        """Update phase field."""
        # Step 1: Calculate the gradient of phi
        grad_phi = jax_gradient(phi, dx, dy)  # Shape: (Nx, Ny, 2)

        # Step 2: Calculate the convective term
        convective_term = U[..., 0] * grad_phi[..., 0] + U[..., 1] * grad_phi[..., 1]

        # Step 3: Calculate the Laplacian of phi
        lap_phi = jax_laplacian(phi, dx, dy)

        # Step 4: Calculate stabilized chemical potential with interface thickness control
        chemical_potential = jax_df_2(phi) - epsilon**2 * lap_phi
        lagrange_multiplier = jnp.mean(chemical_potential)
        source_term = -1/Pe * (chemical_potential - lagrange_multiplier)

        # Step 5: Right-hand side of the phase equation
        rhs_phi = -convective_term + source_term

        # Step 6: Update phase field
        phi = phi + dt * rhs_phi

        # Step 7: Apply boundary conditions and maintain phase field bounds
        phi = jax_apply_contact_angle_boundary_conditions(phi, dx, dy, contact_angle=contact_angle)
        
        # Ensure phi stays within physical bounds [-1, 1]
        phi = jnp.clip(phi, -1.0, 1.0)

        return phi
    
    def _update_pressure(self, surface_tension, Nx, Ny, dx, dy, rho1, rho2):
        """Update pressure field."""
        sf_grad = jax_divergence(surface_tension, dx, dy)
        rho = jax_calculate_density(self.phi, rho1, rho2)

        sf_grad = sf_grad.at[:, 0].set(jnp.sum(rho * self.config.physical.g * dy, axis=1) + self.config.physical.atm_pressure)
        sf_grad = sf_grad.at[:, -1].set(self.config.physical.atm_pressure)

        self.pressure_solver.set_rhs(sf_grad)
        self.pressure_solver.solve()
        P = self.pressure_solver.get_solution()

        return P
    
    def ppe(self, U, dx, dy, dt):
        """Pressure projection - using existing function."""
        return self._ppe(U, dx, dy, dt, self.correction_solver)
    
    def _compute_viscous_term(self, U, dx, dy, Re):
        """Simplified viscous term for constant viscosity: (1/Re) * ∇²U"""
        return jnp.stack([jax_laplacian(U[..., 0], dx, dy) / Re, 
                          jax_laplacian(U[..., 1], dx, dy) / Re], axis=-1)
    
    def _check_continuity(self, U, dx, dy):
        """Check continuity equation condition (∇·U = 0)"""
        u_x = jax_dx(U[..., 0], h=dx)
        v_y = jax_dy(U[..., 1], h=dy)
        
        divergence = u_x + v_y
        max_div = jnp.max(jnp.abs(divergence))
        mean_div = jnp.mean(jnp.abs(divergence))
        
        return divergence, max_div, mean_div
    
    def _apply_velocity_boundary_conditions(self, U, beta, dy):
        """Apply physically appropriate boundary conditions to velocity field."""
        U = U.at[:, 0, 1].set(0.0)
        U = U.at[:, 0, 0].set(U[:, 1, 0] - dy * 1/beta * U[:, 1, 0])
        U = U.at[:, -1, :].set(U[:, -2, :])
        U = U.at[0, :, :].set(U[1, :, :])
        U = U.at[-1, :, :].set(U[-2, :, :])
        return U
    
    def _correction_step(self, U, dx, dy, dt, correction_solver=None, div=None):
        """Single correction step for pressure projection."""
        if div is None:
            div = jax_divergence(U, dx, dy) / dt
        
        div = div - jnp.mean(div)
        correction_solver.set_rhs(div)
        correction_solver.solve()
        p_correction = correction_solver.get_solution()
        
        U = U.at[..., 0].set(U[..., 0] - dt * jax_dx(p_correction, h=dx))
        U = U.at[..., 1].set(U[..., 1] - dt * jax_dy(p_correction, h=dy))
        return U, p_correction
    
    def _ppe(self, U, dx, dy, dt, correction_solver=None, div_threshold=None):
        """Pressure projection with divergence correction."""
        if div_threshold is None:
            div_threshold = self.config.solver_params.divergence_threshold
        max_div_threshold = div_threshold
        max_iterations = self.config.solver_params.max_correction_iterations
        
        U, solution = self._correction_step(U, dx, dy, dt, correction_solver=correction_solver)
        U = self._apply_velocity_boundary_conditions(U, 0.01, dy)
        
        divergence, max_div, mean_div = self._check_continuity(U, dx, dy)
        if mean_div > div_threshold:
            count = 0
            while mean_div > div_threshold and count < max_iterations:
                U, solution = self._correction_step(U, dx, dy, dt, correction_solver=correction_solver, div=divergence/dt)
                U = self._apply_velocity_boundary_conditions(U, 0.01, dy)
                divergence, max_div, mean_div = self._check_continuity(U, dx, dy)
                if count % 20 == 0:
                    print(f"\rMax|mean div: {max_div:.6f}  | {mean_div:.6f}")
                if max_div < max_div_threshold:
                    break
                count += 1
            if count >= max_iterations:
                print(f"\nMax iterations ({max_iterations}) reached")
            else:
                print(f"\nCorrected in {count} iterations \n")
        return U
    
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
