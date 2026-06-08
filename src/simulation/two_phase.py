"""
Two-phase flow simulation.

Implements the physics for two-phase (air-water) droplet spreading.
"""

import numpy as np
import jax.numpy as jnp

from simulation.base import BaseSimulation
from numerics.time_integration import cfl_dt, capillary_cfl_dt, curvature_cfl_dt
from physics.surface_tension import jax_curvature_stats


class TwoPhaseSimulation(BaseSimulation):
    """Two-phase flow simulation (air-water droplet spreading)."""
    
    def __init__(self, config, output_dir=None):
        """Initialize two-phase simulation."""
        super().__init__(config, output_dir)
        print("Two-phase simulation initialized (standard two-phase flow)")
        
        # Initialize step tracking variables
        self._last_max_div = 0.0
        self._last_mean_div = 0.0
        self._last_max_div_interior = 0.0
        self._last_mean_div_interior = 0.0
        self._last_curvature_max = 0.0
        self._last_curvature_mean = 0.0
        self._last_ppe_info = {
            'applied': False, 'iterations': 0,
            'div_before_max': 0.0, 'div_before_mean': 0.0,
            'div_after_max': 0.0, 'div_after_mean': 0.0
        }
    
    def step(self):
        """Perform one two-phase physics step."""
        # Compute CFL-limited time step
        self._compute_cfl_dt()
        
        # Advance time
        self.state.t += self.state.dt
        
        # Predictor step: velocity update
        self._predictor_step()
        
        # Corrector step: PPE for incompressibility
        self._corrector_step()
        
        # Phase field update
        self._phase_update()
        
        # Pressure update
        self._pressure_update()
    
    def _compute_cfl_dt(self):
        """Compute CFL-limited time step including curvature-based CFL."""
        surface_tension = self.state.compute_surface_tension()
        
        # Compute curvature statistics
        curvature_max, curvature_mean = jax_curvature_stats(
            self.state.phi, self.state.dx, self.state.dy, self.state.geometry.f_1_grid
        )
        self._last_curvature_max = float(curvature_max)
        self._last_curvature_mean = float(curvature_mean)
        
        # Velocity-based CFL
        cfl_velocity_dt = cfl_dt(
            float(self.state.U[..., 0].max()), 
            float(self.state.U[..., 1].max()),
            self.state.dx, self.state.dy, C=self.cfl_number
        )
        
        # Capillary CFL
        rho_mean = float(np.mean(self.state.compute_density()))
        st_max = float(np.sqrt(surface_tension[..., 0]**2 + surface_tension[..., 1]**2).max())
        cfl_capillary_dt = capillary_cfl_dt(
            st_max, rho_mean, self.state.epsilon, 
            self.state.dx, self.state.dy, C=self.capillary_cfl_number
        )
        
        # Curvature-based CFL (prevents singularities from high curvature regions)
        curvature_cfl_number = self.config.get('time_params', {}).get('curvature_cfl_number', 0.1)
        cfl_curvature_dt = curvature_cfl_dt(
            self._last_curvature_max, self.state.dx, self.state.dy, C=curvature_cfl_number
        )
        
        # Apply most restrictive CFL
        cfl_dt_min = min(cfl_velocity_dt, cfl_capillary_dt, cfl_curvature_dt)
        if cfl_dt_min < self.state.dt and cfl_dt_min != np.inf:
            self.state.dt = cfl_dt_min
            if self.state.step % 10 == 0:
                limiting = "velocity" if cfl_dt_min == cfl_velocity_dt else \
                          ("capillary" if cfl_dt_min == cfl_capillary_dt else "curvature")
                print(f"CFL-limited dt: {self.state.dt:.6e} ({limiting})")
    
    def _predictor_step(self):
        """Predictor step on staggered MAC faces."""
        surface_tension = self.state.compute_surface_tension()
        from solvers.staggered_velocity import staggered_predictor_step

        self.state.ensure_face_velocities()

        u_face, v_face = staggered_predictor_step(
            self.state.u_face,
            self.state.v_face,
            surface_tension,
            self.state.dt,
            self.state.dx,
            self.state.dy,
            self.state.fluid_solver.Re2,
            self.state.fluid_solver.Fr,
            self.state.fluid_solver.g,
            include_gravity=self.state.include_gravity,
            include_advection=True,
            P=self.state.P,
            phi=self.state.phi,
            rho1=self.state.rho1,
            rho2=self.state.rho2,
            geometry=self.state.geometry,
        )

        # Keep BC application in face space for MAC consistency.
        u_face, v_face = self.state.velocity_bc.apply_to_faces(
            u_face, v_face,
            self.state.dx, self.state.dy,
            psi=self._get_psi_for_physics(), geometry=self.state.geometry, phi=self.state.phi
        )
        self.state.set_face_velocities(u_face, v_face, sync_collocated=True)
    
    def _corrector_step(self):
        """Corrector step: staggered PPE to enforce incompressibility."""
        from solvers.ppe import _mac_divergence_geometry

        self.state.ensure_face_velocities()
        u_face, v_face = self.state.u_face, self.state.v_face
        div_face = _mac_divergence_geometry(
            u_face, v_face, self.state.dx, self.state.dy, self.state.geometry
        )
        abs_div = jnp.abs(div_face)
        max_div = float(jnp.max(abs_div))
        mean_div = float(jnp.mean(abs_div))

        # Initialize PPE info
        self._last_ppe_info = {
            'applied': False, 'iterations': 0,
            'div_before_max': float(max_div),
            'div_before_mean': float(mean_div),
            'div_after_max': float(max_div),
            'div_after_mean': float(mean_div)
        }
        
        # Apply PPE if needed
        if max_div > self.max_div_threshold or mean_div > self.mean_div_threshold:
            solver_params = self.config.get('solver_params', {})
            ppe_settings = solver_params.get('ppe', {})
            
            # Derive PPE BCs automatically from velocity BCs (if not explicitly set)
            from solvers.ppe_bc_derivation import derive_ppe_bcs_from_config
            ppe_bcs_explicit = ppe_settings.get('boundary_conditions')
            if ppe_bcs_explicit is None:
                # Automatically derive from velocity BCs
                ppe_bcs = derive_ppe_bcs_from_config(self.config)
            else:
                # Use explicit BCs (for backward compatibility)
                ppe_bcs = ppe_bcs_explicit
            
            max_iterations = ppe_settings.get('max_iterations', 1000)
            min_iterations = ppe_settings.get('min_iterations', 1)
            under_relaxation = ppe_settings.get('under_relaxation', 1.0)
            default_convergence_mode = (
                'both' if getattr(self.state.geometry, "has_geometry", False) else 'interior'
            )
            convergence_mode = ppe_settings.get('convergence_mode', default_convergence_mode)
            # No crutches: checkerboard filter, Rhie–Chow disabled; under_relaxation optional
            # Pass output_dir and step for diagnostics ONLY at checkpoint steps
            if hasattr(self, "checkpoint_interval") and self.state.step % self.checkpoint_interval == 0:
                debug_output_dir = self.output_dir if hasattr(self, 'output_dir') else None
                debug_step = self.state.step if hasattr(self.state, 'step') else None
            else:
                debug_output_dir = None
                debug_step = None
            from solvers.ppe import ppe_solve_staggered
            self.state.U, self._last_ppe_info = ppe_solve_staggered(
                self.state.U,
                self.state.dx,
                self.state.dy,
                self.state.dt,
                self.state.geometry,
                self.state.correction_solver,
                velocity_bc_manager=self.state.velocity_bc,
                ppe_bcs=ppe_bcs,
                psi=self._get_psi_for_physics(),
                div_threshold=self.div_threshold,
                max_div_threshold=self.max_div_threshold,
                mean_div_threshold=self.mean_div_threshold,
                min_iterations=min_iterations,
                max_iterations=max_iterations,
                debug_output_dir=debug_output_dir,
                debug_step=debug_step,
                under_relaxation=under_relaxation,
                phi=self.state.phi,
                rho1=self.state.rho1,
                rho2=self.state.rho2,
                convergence_mode=convergence_mode,
                u_face_in=self.state.u_face,
                v_face_in=self.state.v_face,
            )
            # Sync converged faces from staggered PPE output.
            if (
                isinstance(self._last_ppe_info, dict)
                and "u_face_out" in self._last_ppe_info
                and "v_face_out" in self._last_ppe_info
            ):
                self.state.set_face_velocities(
                    self._last_ppe_info["u_face_out"],
                    self._last_ppe_info["v_face_out"],
                    sync_collocated=True,
                )

            div_face = _mac_divergence_geometry(
                self.state.u_face,
                self.state.v_face,
                self.state.dx,
                self.state.dy,
                self.state.geometry,
            )
            abs_div = jnp.abs(div_face)
            max_div = float(jnp.max(abs_div))
            mean_div = float(jnp.mean(abs_div))
            if abs_div.shape[0] > 2 and abs_div.shape[1] > 2:
                interior = abs_div[1:-1, 1:-1]
                max_div_interior = float(jnp.max(interior))
                mean_div_interior = float(jnp.mean(interior))
            else:
                max_div_interior = max_div
                mean_div_interior = mean_div
            self._last_ppe_info['div_after_max'] = float(max_div)
            self._last_ppe_info['div_after_mean'] = float(mean_div)
            self._last_ppe_info['div_after_max_interior'] = float(max_div_interior)
            self._last_ppe_info['div_after_mean_interior'] = float(mean_div_interior)
        
        # Store for telemetry
        self._last_max_div = float(max_div)
        self._last_mean_div = float(mean_div)
        self._last_max_div_interior = float(
            self._last_ppe_info.get('div_after_max_interior', max_div)
        )
        self._last_mean_div_interior = float(
            self._last_ppe_info.get('div_after_mean_interior', mean_div)
        )
    
    def _phase_update(self):
        """Update phase field."""
        self.state.phi = self.state.phase_solver.update(
            self.state.phi, self.state.U, self.state.dt, self.state.dx, self.state.dy,
            self.state.geometry, use_jax=True, psi=self._get_psi_for_physics(),
            u_face=self.state.u_face, v_face=self.state.v_face
        )
        self.state.invalidate_cache()
    
    def _pressure_update(self):
        """Update pressure field."""
        from boundary_conditions.base_bc import BCType
        import jax.numpy as jnp
        
        surface_tension = self.state.compute_surface_tension()
        
        # Check if any Dirichlet BC is set (to prevent normalization from breaking it)
        has_dirichlet = any(
            self.state.pressure_bc.bc_types.get(b) == BCType.DIRICHLET
            for b in ["top", "bottom", "left", "right"]
        )
        
        self.state.P = self.state.pressure_solver.update_pressure(
            surface_tension, self.state.dx, self.state.dy, self.state.geometry, self.state.phi,
            self.state.pressure_linear_solver,
            has_dirichlet_bc=has_dirichlet
        )
        
        # CRITICAL: Apply pressure offset to enforce Dirichlet BCs
        # Surface tension creates negative capillary pressure, which can make
        # the entire field negative. We need to add an offset to preserve
        # Dirichlet BC values while maintaining pressure gradients.
        if has_dirichlet:
            # Find Dirichlet boundaries and their target values
            dirichlet_offsets = []
            for boundary in ["top", "bottom", "left", "right"]:
                if self.state.pressure_bc.bc_types.get(boundary) == BCType.DIRICHLET:
                    target_value = self.state.pressure_bc.dirichlet_values.get(
                        boundary, self.state.pressure_bc.open_pressure
                    )
                    # Get current value at boundary
                    if boundary == "left":
                        current_value = float(jnp.mean(self.state.P[0, :]))
                    elif boundary == "right":
                        current_value = float(jnp.mean(self.state.P[-1, :]))
                    elif boundary == "top":
                        current_value = float(jnp.mean(self.state.P[:, -1]))
                    elif boundary == "bottom":
                        current_value = float(jnp.mean(self.state.P[:, 0]))
                    else:
                        continue
                    
                    offset = target_value - current_value
                    dirichlet_offsets.append(offset)
            
            # Use the maximum offset to ensure all Dirichlet BCs are satisfied
            # (or average if they conflict, but typically they should be consistent)
            if dirichlet_offsets:
                pressure_offset = max(dirichlet_offsets)  # Use max to ensure all BCs satisfied
                self.state.P = self.state.P + pressure_offset
        
        # Re-apply only Dirichlet pressure boundaries.
        # Do NOT re-apply Neumann copy on total pressure here:
        # that can corrupt dynamic-pressure consistency near the wall
        # after the variable-coefficient capillary solve.
        for boundary in ["top", "bottom", "left", "right"]:
            if self.state.pressure_bc.bc_types.get(boundary) != BCType.DIRICHLET:
                continue
            target_value = self.state.pressure_bc.dirichlet_values.get(
                boundary, self.state.pressure_bc.open_pressure
            )
            self.state.P = self.state.pressure_bc._dirichlet(
                self.state.P, boundary, float(target_value)
            )
        self.state.invalidate_cache()
    
    # ==================== Hook Methods for Subclasses ====================
    
    def _get_psi_for_physics(self):
        """Get ice phase field for physics calculations. Override in ice-water sim."""
        return None
    
    def _apply_velocity_bc(self):
        """Apply velocity boundary conditions. Can be overridden."""
        return self.state.velocity_bc.apply_boundary_conditions(
            self.state.U, self.state.dx, self.state.dy, use_jax=True,
            geometry=self.state.geometry, psi=self._get_psi_for_physics()
        )
