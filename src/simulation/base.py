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
import jax.numpy as jnp
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
from numerics.finite_differences import jax_gradient
from physics.pressure import compute_hydrostatic_pressure


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
            self.output_dir,
            include_ice_water=self.include_ice_water,
            config=config,
            append=self.restart_from is not None,
        )
        print(f"Telemetry logging initialized in {self.output_dir}")
        
        # Timing
        self.step_times = []
        self._last_curvature_max = None
        self._last_curvature_mean = None
        self._pyvista_disabled_due_to_error = False

    def _should_use_pyvista(self):
        return (
            PYVISTA_AVAILABLE
            and self.config.get("visualization", {}).get("use_pyvista", True)
            and not os.environ.get("DISABLE_PYVISTA")
            and not self._pyvista_disabled_due_to_error
        )
    
    def setup(self):
        """Setup before main loop: visualizations, diagnostics."""
        if getattr(self.state.geometry, "has_geometry", False):
            geometry_type = (
                getattr(getattr(self.state, "context", None), "geometry_type", None)
                or self.config.get("initial_conditions", {}).get("geometry", {}).get("type", "non-flat")
            )
            print(f"Initialized {geometry_type} geometry")
        else:
            print("Initialized flat geometry")
        if self.restart_from is None:
            print("Applying contact angle boundary conditions to initial phase field...")
            self.state.phi = self.state.phase_field_bc.apply_boundary_conditions(
                self.state.phi, self.state.dx, self.state.dy, use_jax=True,
                geometry=self.state.geometry
            )
            print("Contact angle BC applied")
        
        if self.restart_from is None:
            plt.imshow(self.state.phi.T, extent=[0, self.state.Lx, 0, self.state.Ly], origin='lower', cmap='viridis')
            plt.colorbar(label='Phase Field (phi)')
            plt.title('Initial Phase Field')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.savefig(f'{self.output_dir}/initial_phase_field.png', bbox_inches='tight')
            plt.clf()
        
        # Create initial joint plot (PyVista when available and enabled, else matplotlib)
        use_pyvista = self._should_use_pyvista()
        viz_backend = "PyVista" if use_pyvista else "matplotlib"
        if self.restart_from is None:
            print(f"Creating initial state visualization ({viz_backend})...")
            self._create_joint_plot(step=0)
            print(f"Initial state plot saved to {self.visualization_dir}/joint_plot_step_0.png")
        else:
            print(f"Resuming from {self.restart_from} at step {self.state.step}, t={self.state.t:.6f}")
            print("Re-applying contact angle BC to resumed phase field...")
            import jax.numpy as jnp
            self.state.phi = jnp.asarray(
                self.state.phase_field_bc.apply_boundary_conditions(
                    jnp.asarray(self.state.phi), self.state.dx, self.state.dy, use_jax=True,
                    geometry=self.state.geometry, U=self.state.U,
                )
            )
        
        # First-step diagnostics
        if self.restart_from is None:
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
            if self.state.step % 500 == 0:
                self._log_memory_usage()
            
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
        use_pyvista = self._should_use_pyvista()
        if use_pyvista:
            try:
                create_joint_plot_pyvista_full(**kwargs)
                return
            except Exception as exc:
                self._pyvista_disabled_due_to_error = True
                print(f"Warning: PyVista plotting failed, falling back to matplotlib for the rest of the run: {exc}")
                create_joint_plot(**kwargs)
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
        liquid_fraction = np.clip(0.5 * (1.0 - np.asarray(self.state.phi)), 0.0, 1.0)
        mass = float(np.sum(liquid_fraction) * self.state.dx * self.state.dy)
        
        # Store for subclasses and progress output
        self._last_mass = mass
        self._last_droplet_bounds = (start_of_droplet, end_of_droplet, bottom_of_droplet, top_of_droplet)
        self._last_phase_health = self._compute_phase_health_metrics(self.state.phi)
        
        surface_tension = self.state.compute_surface_tension()
        contact_line_forces = self._compute_contact_line_force_metrics(
            phi=self.state.phi,
            P=self.state.P,
            surface_tension=surface_tension,
            rho=rho,
            dx=self.state.dx,
            dy=self.state.dy,
        )
        
        # Get curvature stats if available (computed by subclass)
        curvature_max = self._last_curvature_max
        curvature_mean = self._last_curvature_mean
        
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
            max_div_interior=getattr(self, "_last_max_div_interior", self._last_max_div),
            mean_div_interior=getattr(self, "_last_mean_div_interior", self._last_mean_div),
            psi=self.state.psi if self.include_ice_water else None,
            T=self.state.T if self.include_ice_water else None,
            geometry=self.state.geometry,
            dx=self.state.dx,
            curvature_max=curvature_max,
            curvature_mean=curvature_mean,
            contact_line_forces=contact_line_forces,
            phase_health=self._last_phase_health,
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

    @staticmethod
    def _max_contiguous_true_run(mask, axis):
        arr = mask if axis == 1 else mask.T
        best = 0
        for line in arr:
            padded = np.concatenate(([False], np.asarray(line, dtype=bool), [False]))
            edges = np.flatnonzero(padded[1:] != padded[:-1])
            if edges.size:
                best = max(best, int((edges[1::2] - edges[::2]).max()))
        return best

    def _compute_phase_health_metrics(self, phi):
        phi_np = np.asarray(phi)
        finite = np.isfinite(phi_np)
        band = finite & (phi_np > -0.9) & (phi_np < 0.9)
        liquid_unclipped = 0.5 * (1.0 - phi_np)
        return {
            "interface_cells_abs_lt_0p9": int(np.sum(band)),
            "interface_max_width_cells": int(
                max(
                    self._max_contiguous_true_run(band, axis=0),
                    self._max_contiguous_true_run(band, axis=1),
                )
            ),
            "liquid_mass_unclipped": float(np.sum(liquid_unclipped) * self.state.dx * self.state.dy),
            "liquid_area_phi_lt_0": int(np.sum(finite & (phi_np < 0.0))),
            "vapor_area_phi_gt_0": int(np.sum(finite & (phi_np > 0.0))),
        }

    def _check_phase_health(self):
        settings = self.config.get("solver_params", {}).get("phase_health", {}) or {}
        if not settings.get("enabled", False):
            return
        metrics = self._compute_phase_health_metrics(self.state.phi)
        if not hasattr(self, "_phase_health_initial"):
            self._phase_health_initial = dict(metrics)
        phi_np = np.asarray(self.state.phi)
        failures = []
        min_phi_floor = settings.get("min_phi_floor")
        if min_phi_floor is not None and float(np.nanmin(phi_np)) < float(min_phi_floor):
            failures.append(f"phi_min={float(np.nanmin(phi_np)):.6g} < {float(min_phi_floor):.6g}")
        min_liquid_phi = settings.get("min_liquid_phi")
        if min_liquid_phi is not None and float(np.nanmin(phi_np)) > float(min_liquid_phi):
            failures.append(f"phi_min={float(np.nanmin(phi_np)):.6g} > {float(min_liquid_phi):.6g}")
        max_phi_ceiling = settings.get("max_phi_ceiling")
        if max_phi_ceiling is not None and float(np.nanmax(phi_np)) > float(max_phi_ceiling):
            failures.append(f"phi_max={float(np.nanmax(phi_np)):.6g} > {float(max_phi_ceiling):.6g}")
        max_interface_cells_ratio = settings.get("max_interface_cells_ratio")
        if max_interface_cells_ratio is not None:
            base = max(int(self._phase_health_initial["interface_cells_abs_lt_0p9"]), 1)
            ratio = metrics["interface_cells_abs_lt_0p9"] / base
            if ratio > float(max_interface_cells_ratio):
                failures.append(
                    f"interface_cells_ratio={ratio:.6g} > {float(max_interface_cells_ratio):.6g}"
                )
        max_interface_width_cells = settings.get("max_interface_width_cells")
        if max_interface_width_cells is not None and metrics["interface_max_width_cells"] > int(max_interface_width_cells):
            failures.append(
                f"interface_max_width_cells={metrics['interface_max_width_cells']} > {int(max_interface_width_cells)}"
            )
        if failures:
            raise RuntimeError(
                "Phase health guard failed at step "
                f"{self.state.step}: " + "; ".join(failures)
            )
    
    def _log_memory_usage(self):
        """Log process RSS and JAX device memory (when available)."""
        import os
        rss_mb = None
        try:
            import psutil
            rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except ImportError:
            pass
        jax_mb = None
        try:
            from jax.lib import xla_bridge
            stats = xla_bridge.get_backend().memory_stats()
            if stats and "bytes_in_use" in stats:
                jax_mb = float(stats["bytes_in_use"]) / (1024 * 1024)
        except Exception:
            pass
        parts = [f"step={self.state.step}"]
        if rss_mb is not None:
            parts.append(f"rss_mb={rss_mb:.1f}")
        if jax_mb is not None:
            parts.append(f"jax_mb={jax_mb:.1f}")
        if len(parts) > 1:
            print("Memory: " + ", ".join(parts))

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

    def _compute_contact_line_force_metrics(self, phi, P, surface_tension, rho, dx, dy):
        """Compute force-balance diagnostics in a narrow contact-line strip near the wall."""
        phi_np = np.array(phi)
        P_np = np.array(P)
        sf_np = np.array(surface_tension)
        rho_np = np.array(rho)
        Nx, Ny = phi_np.shape
        force_diag_cfg = self.config.get("solver_params", {}).get("force_diagnostics", {})
        use_ppe_corr_pressure = bool(force_diag_cfg.get("use_ppe_correction_pressure", False))
        pressure_smoothing_passes = int(force_diag_cfg.get("pressure_smoothing_passes", 1))
        contact_strip_height = int(force_diag_cfg.get("contact_strip_height", 4))
        robust_trim_quantile = float(force_diag_cfg.get("robust_trim_quantile", 0.1))

        def _smooth_2d(arr, passes=1):
            out = np.array(arr, dtype=np.float64, copy=True)
            if out.ndim != 2 or passes <= 0:
                return out
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
            for _ in range(passes):
                # x-pass
                px = np.pad(out, ((1, 1), (0, 0)), mode="edge")
                out = (
                    kernel[0] * px[:-2, :]
                    + kernel[1] * px[1:-1, :]
                    + kernel[2] * px[2:, :]
                )
                # y-pass
                py = np.pad(out, ((0, 0), (1, 1)), mode="edge")
                out = (
                    kernel[0] * py[:, :-2]
                    + kernel[1] * py[:, 1:-1]
                    + kernel[2] * py[:, 2:]
                )
            return out

        def _robust_mean_1d(values):
            if values.size == 0:
                return 0.0
            if values.size < 10:
                return float(np.mean(values))
            q = min(max(robust_trim_quantile, 0.0), 0.49)
            lo = np.quantile(values, q)
            hi = np.quantile(values, 1.0 - q)
            core = values[(values >= lo) & (values <= hi)]
            if core.size == 0:
                return float(np.mean(values))
            return float(np.mean(core))

        def _robust_mean_dot(vec, direction, mask):
            if not np.any(mask):
                return 0.0
            proj = np.sum(vec[mask] * direction[mask], axis=1)
            return _robust_mean_1d(proj)

        if Ny < 2:
            return {
                "left_index": -1.0, "right_index": -1.0,
                "sf_norm_mean": 0.0, "pg_norm_mean": 0.0, "pg_dyn_norm_mean": 0.0, "pg_hydro_norm_mean": 0.0,
                "g_norm": 0.0, "sf_to_g_ratio": 0.0, "sf_to_pg_dyn_ratio": 0.0,
                "sf_ax_mean_abs": 0.0, "pg_dyn_ax_mean_abs": 0.0,
                "sf_to_pg_dyn_ratio_xabs": 0.0,
                "sf_n_mean": 0.0, "pg_dyn_n_mean": 0.0, "pg_h_n_mean": 0.0, "g_n_mean": 0.0, "res_n_mean": 0.0,
                "sf_t_mean": 0.0, "pg_dyn_t_mean": 0.0, "pg_h_t_mean": 0.0, "g_t_mean": 0.0, "res_t_mean": 0.0,
                "sf_norm_mean_liquid": 0.0, "sf_norm_mean_gas": 0.0,
                "pg_dyn_norm_mean_liquid": 0.0, "pg_dyn_norm_mean_gas": 0.0,
            }

        phi_bottom = phi_np[:, 0]
        phi_above = phi_np[:, 1]
        contact_mask = ((phi_bottom * phi_above) < 0.0) | (np.abs(phi_bottom) < 0.5)
        idx = np.where(contact_mask)[0]
        if idx.size == 0:
            return {
                "left_index": -1.0, "right_index": -1.0,
                "sf_norm_mean": 0.0, "pg_norm_mean": 0.0, "pg_dyn_norm_mean": 0.0, "pg_hydro_norm_mean": 0.0,
                "g_norm": abs(float(self.state.g) / max(float(self.state.Fr) ** 2, 1e-12)),
                "sf_to_g_ratio": 0.0, "sf_to_pg_dyn_ratio": 0.0,
                "sf_ax_mean_abs": 0.0, "pg_dyn_ax_mean_abs": 0.0,
                "sf_to_pg_dyn_ratio_xabs": 0.0,
                "sf_n_mean": 0.0, "pg_dyn_n_mean": 0.0, "pg_h_n_mean": 0.0, "g_n_mean": 0.0, "res_n_mean": 0.0,
                "sf_t_mean": 0.0, "pg_dyn_t_mean": 0.0, "pg_h_t_mean": 0.0, "g_t_mean": 0.0, "res_t_mean": 0.0,
                "sf_norm_mean_liquid": 0.0, "sf_norm_mean_gas": 0.0,
                "pg_dyn_norm_mean_liquid": 0.0, "pg_dyn_norm_mean_gas": 0.0,
            }

        left_idx = int(idx.min())
        right_idx = int(idx.max())
        pad = 2
        i0 = max(0, left_idx - pad)
        i1 = min(Nx - 1, right_idx + pad)
        j1 = min(Ny, max(2, contact_strip_height))
        region = np.zeros((Nx, Ny), dtype=bool)
        region[i0:i1 + 1, 0:j1] = True

        # Optional pressure correction contribution from the latest PPE step.
        # Disabled by default as it can inject high-frequency checker noise.
        P_eff = P_np
        if use_ppe_corr_pressure and isinstance(self._last_ppe_info, dict):
            p_corr_out = self._last_ppe_info.get("p_corr_out")
            if p_corr_out is not None:
                p_corr_np = np.array(p_corr_out)
                if p_corr_np.shape == P_np.shape:
                    P_eff = P_np + p_corr_np
        P_eff = _smooth_2d(P_eff, passes=pressure_smoothing_passes)

        inv_rho = 1.0 / np.maximum(rho_np, 1e-12)
        grad_p = np.array(jax_gradient(jnp.array(P_eff), float(dx), float(dy), self.state.geometry.f_1_grid))
        a_pg = -grad_p * inv_rho[..., None]

        p_hydro = np.array(
            compute_hydrostatic_pressure(
                jnp.array(rho_np), float(self.state.g), float(dy),
                float(self.state.Fr), float(self.state.atm_pressure),
            )
        )
        p_dyn = P_eff - p_hydro
        grad_p_h = np.array(jax_gradient(jnp.array(p_hydro), float(dx), float(dy), self.state.geometry.f_1_grid))
        grad_p_dyn = np.array(jax_gradient(jnp.array(p_dyn), float(dx), float(dy), self.state.geometry.f_1_grid))
        a_pg_h = -grad_p_h * inv_rho[..., None]
        a_pg_dyn = -grad_p_dyn * inv_rho[..., None]
        a_sf = -sf_np * inv_rho[..., None]

        g_norm = abs(float(self.state.g) / max(float(self.state.Fr) ** 2, 1e-12))
        # Focus diagnostics on interface-adjacent cells to avoid bulk/gas dilution.
        interface_band = np.abs(phi_np) < 0.75
        m = region & interface_band
        if not np.any(m):
            m = region
        sf_norm = np.linalg.norm(a_sf[m], axis=1) if np.any(m) else np.array([0.0])
        pg_norm = np.linalg.norm(a_pg[m], axis=1) if np.any(m) else np.array([0.0])
        pg_dyn_norm = np.linalg.norm(a_pg_dyn[m], axis=1) if np.any(m) else np.array([0.0])
        pg_h_norm = np.linalg.norm(a_pg_h[m], axis=1) if np.any(m) else np.array([0.0])

        sf_norm_mean = _robust_mean_1d(sf_norm)
        pg_norm_mean = _robust_mean_1d(pg_norm)
        pg_dyn_norm_mean = _robust_mean_1d(pg_dyn_norm)
        pg_h_norm_mean = _robust_mean_1d(pg_h_norm)
        sf_ax_mean_abs = _robust_mean_1d(np.abs(a_sf[m, 0])) if np.any(m) else 0.0
        pg_dyn_ax_mean_abs = _robust_mean_1d(np.abs(a_pg_dyn[m, 0])) if np.any(m) else 0.0
        sf_to_pg_dyn_ratio_xabs = float(sf_ax_mean_abs / max(pg_dyn_ax_mean_abs, 1e-12))

        # Signed decomposition along local interface normal/tangent.
        grad_phi = np.array(jax_gradient(jnp.array(phi_np), float(dx), float(dy), self.state.geometry.f_1_grid))
        nrm = np.linalg.norm(grad_phi, axis=-1, keepdims=True)
        n_hat = grad_phi / np.maximum(nrm, 1e-12)
        t_hat = np.concatenate([-n_hat[..., 1:2], n_hat[..., 0:1]], axis=-1)
        g_vec = np.zeros_like(a_sf)
        g_vec[..., 1] = float(self.state.g) / max(float(self.state.Fr) ** 2, 1e-12)
        a_res = a_sf + a_pg_dyn + a_pg_h + g_vec

        # Phase split in the same region for liquid/gas sensitivity checks.
        liquid_mask = m & (phi_np < 0.0)
        gas_mask = m & (phi_np >= 0.0)
        sf_norm_mean_liquid = _robust_mean_1d(np.linalg.norm(a_sf[liquid_mask], axis=1)) if np.any(liquid_mask) else 0.0
        sf_norm_mean_gas = _robust_mean_1d(np.linalg.norm(a_sf[gas_mask], axis=1)) if np.any(gas_mask) else 0.0
        pg_dyn_norm_mean_liquid = _robust_mean_1d(np.linalg.norm(a_pg_dyn[liquid_mask], axis=1)) if np.any(liquid_mask) else 0.0
        pg_dyn_norm_mean_gas = _robust_mean_1d(np.linalg.norm(a_pg_dyn[gas_mask], axis=1)) if np.any(gas_mask) else 0.0

        return {
            "left_index": float(left_idx),
            "right_index": float(right_idx),
            "sf_norm_mean": sf_norm_mean,
            "pg_norm_mean": pg_norm_mean,
            "pg_dyn_norm_mean": pg_dyn_norm_mean,
            "pg_hydro_norm_mean": pg_h_norm_mean,
            "g_norm": float(g_norm),
            "sf_to_g_ratio": float(sf_norm_mean / max(g_norm, 1e-12)),
            "sf_to_pg_dyn_ratio": float(sf_norm_mean / max(pg_dyn_norm_mean, 1e-12)),
            "sf_ax_mean_abs": sf_ax_mean_abs,
            "pg_dyn_ax_mean_abs": pg_dyn_ax_mean_abs,
            "sf_to_pg_dyn_ratio_xabs": sf_to_pg_dyn_ratio_xabs,
            "sf_n_mean": _robust_mean_dot(a_sf, n_hat, m),
            "pg_dyn_n_mean": _robust_mean_dot(a_pg_dyn, n_hat, m),
            "pg_h_n_mean": _robust_mean_dot(a_pg_h, n_hat, m),
            "g_n_mean": _robust_mean_dot(g_vec, n_hat, m),
            "res_n_mean": _robust_mean_dot(a_res, n_hat, m),
            "sf_t_mean": _robust_mean_dot(a_sf, t_hat, m),
            "pg_dyn_t_mean": _robust_mean_dot(a_pg_dyn, t_hat, m),
            "pg_h_t_mean": _robust_mean_dot(a_pg_h, t_hat, m),
            "g_t_mean": _robust_mean_dot(g_vec, t_hat, m),
            "res_t_mean": _robust_mean_dot(a_res, t_hat, m),
            "sf_norm_mean_liquid": sf_norm_mean_liquid,
            "sf_norm_mean_gas": sf_norm_mean_gas,
            "pg_dyn_norm_mean_liquid": pg_dyn_norm_mean_liquid,
            "pg_dyn_norm_mean_gas": pg_dyn_norm_mean_gas,
        }
    
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
        solver_params = self.config.get("solver_params", {})
        if solver_params.get("chainsaw_diagnostics", False):
            from runtime_diagnostics.chainsaw_step_diagnostics import append_chainsaw_diagnostics_csv
            from runtime_diagnostics.ghost_row_diagnostics import append_ghost_row_instep_csv
            from runtime_diagnostics.phase_update_stage_diagnostics import append_phase_stage_rows

            append_ghost_row_instep_csv(
                self.state.phase_solver,
                self.state.step,
                self.state.t,
                self.output_dir,
            )
            if solver_params.get(
                "phase_stage_diagnostics",
                solver_params.get("chainsaw_diagnostics", False),
            ):
                append_phase_stage_rows(
                    self.state.phase_solver,
                    self.state.step,
                    self.state.t,
                    self.output_dir,
                )
            append_chainsaw_diagnostics_csv(self, self.output_dir)
        self.state.ensure_face_velocities()
        save_checkpoint(
            self.state.step, self.state.phi, self.state.U, self.state.P,
            directory=self.checkpoint_dir,
            psi=self.state.psi if self.include_ice_water else None,
            T=self.state.T if self.include_ice_water else None,
            u_face=self.state.u_face,
            v_face=self.state.v_face,
            t=self.state.t,
            dt=self.state.dt,
        )
