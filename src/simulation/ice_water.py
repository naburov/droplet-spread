"""
Ice-water phase transition simulation.

Extends TwoPhaseSimulation with ice-water phase transition physics.
"""

from simulation.two_phase import TwoPhaseSimulation
from visualization.plot_telemetry import plot_ice_phase_transition, plot_temperature_evolution


class IceWaterSimulation(TwoPhaseSimulation):
    """Ice-water phase transition simulation.
    
    Extends two-phase flow with:
    - Ice phase field evolution
    - Temperature evolution with latent heat
    """
    
    def __init__(self, config, output_dir=None):
        """Initialize ice-water simulation with additional solvers."""
        super().__init__(config, output_dir)
        
        # Verify ice-water is enabled
        if not self.include_ice_water:
            raise ValueError("IceWaterSimulation requires include_ice_water_transition=True in config")
        
        # Get ice parameters
        self.ice_params = config.get("ice_water_params", {})
        if not self.ice_params:
            raise ValueError("ice_water_params not found in config")
        
        # Initialize ice and temperature solvers
        from physics.ice_phase_field import IcePhaseFieldSolver
        from physics.temperature import TemperatureSolver
        from boundary_conditions.ice_phase_field_bc import IcePhaseFieldBoundaryConditions
        from boundary_conditions.temperature_bc import TemperatureBoundaryConditions
        
        M_psi = self.ice_params.get("M_psi", 1.0)
        epsilon_psi = self.ice_params.get("epsilon_psi", 0.02)
        self.T_melt = self.ice_params.get("T_melt", 273.15)
        
        self.ice_phase_solver = IcePhaseFieldSolver(M_psi, epsilon_psi, self.T_melt, config)
        
        self.alpha_water = self.ice_params.get("alpha_water", 1.4e-7)
        self.alpha_ice = self.ice_params.get("alpha_ice", 1.1e-6)
        self.L = self.ice_params.get("L", 334000.0)
        self.c_p_water = self.ice_params.get("c_p_water", 4186.0)
        self.c_p_ice = self.ice_params.get("c_p_ice", 2100.0)
        
        self.temperature_solver = TemperatureSolver(
            self.alpha_water, self.alpha_ice, self.L, 
            self.c_p_water, self.c_p_ice, self.T_melt, config
        )
        
        self.ice_phase_bc_manager = IcePhaseFieldBoundaryConditions(config)
        self.temperature_bc_manager = TemperatureBoundaryConditions(config)
        
        # For latent heat calculation
        self.psi_old = None
        
        print("Ice-water phase transition enabled")
    
    def step(self):
        """Perform one step with ice-water physics."""
        # Store old psi for latent heat calculation
        self.psi_old = self.state.psi.copy()
        
        # Two-phase physics (predictor, corrector, phase, pressure)
        super().step()
        
        # Ice-water specific updates
        self._ice_update()
        self._temperature_update()
    
    def _ice_update(self):
        """Update ice phase field."""
        self.state.psi = self.ice_phase_solver.update(
            self.state.psi, self.state.T, self.state.U,
            self.state.dt, self.state.dx, self.state.dy,
            self.state.geometry, use_jax=True, phi=self.state.phi
        )
        
        # Re-apply phase field BC with updated psi for ice-aware contact angle
        if self.state.phase_solver.bc_manager is not None:
            self.state.phi = self.state.phase_solver.bc_manager.apply_boundary_conditions(
                self.state.phi, self.state.dx, self.state.dy, use_jax=True,
                psi=self.state.psi, geometry=self.state.geometry
            )
    
    def _temperature_update(self):
        """Update temperature field with latent heat."""
        self.state.T = self.temperature_solver.update(
            self.state.T, self.state.psi, self.state.U,
            self.state.dt, self.state.dx, self.state.dy,
            self.state.geometry, psi_old=self.psi_old, use_jax=True
        )
        self.state.T = self.temperature_bc_manager.apply_boundary_conditions(
            self.state.T, self.state.dx, self.state.dy, use_jax=True
        )
    
    # ==================== Override Hook Methods ====================
    
    def _get_psi_for_physics(self):
        """Return ice phase field for physics calculations."""
        return self.state.psi
    
    def _apply_velocity_bc(self):
        """Apply velocity BC with ice phase field."""
        return self.state.velocity_bc.apply_boundary_conditions(
            self.state.U, self.state.dx, self.state.dy, use_jax=True,
            psi=self.state.psi, geometry=self.state.geometry
        )
    
    # ==================== Extended Telemetry ====================
    
    def _log_telemetry(self):
        """Log telemetry including ice-water specific data."""
        super()._log_telemetry()
        
        # Ice-specific telemetry
        rho_water = self.state.rho2
        rho_ice = self.ice_params.get("rho_ice", 917.0)
        
        # Phase change rate
        if self.psi_old is not None and self.state.step > 0:
            dpsi_dt = (self.state.psi - self.psi_old) / self.state.dt
        else:
            dpsi_dt = None
        
        self.telemetry_logger.log_ice_phase_statistics(
            self.state.step, self.state.t, self.state.psi, self.state.T,
            self.T_melt, rho_water, rho_ice,
            self.state.dx, self.state.dy, psi_old=self.psi_old
        )
        
        self.telemetry_logger.log_temperature_evolution(
            self.state.step, self.state.t, self.state.T, self.T_melt,
            psi=self.state.psi, dpsi_dt=dpsi_dt,
            alpha_water=self.alpha_water, alpha_ice=self.alpha_ice, L=self.L,
            c_p_water=self.c_p_water, c_p_ice=self.c_p_ice,
            U=self.state.U, dx=self.state.dx, dy=self.state.dy
        )
    
    def _generate_telemetry_plots(self):
        """Generate telemetry plots including ice-water plots."""
        super()._generate_telemetry_plots()
        
        # Ice-water specific plots
        baseline_dir = self.config.get('baseline_experiment', None)
        
        def load_baseline(filename):
            if not baseline_dir:
                return None
            try:
                import os
                import pandas as pd
                path = os.path.join(baseline_dir, filename)
                return pd.read_csv(path) if os.path.exists(path) else None
            except ImportError:
                return None
        
        import os
        plot_ice_phase_transition(
            os.path.join(self.output_dir, 'ice_phase_transition.csv'),
            self.visualization_dir,
            baseline_df=load_baseline('ice_phase_transition.csv')
        )
        plot_temperature_evolution(
            os.path.join(self.output_dir, 'temperature_evolution.csv'),
            self.visualization_dir,
            baseline_df=load_baseline('temperature_evolution.csv')
        )
