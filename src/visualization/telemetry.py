"""
Telemetry and CSV logging for droplet spreading simulation.

This module handles logging simulation statistics to CSV files for later analysis.
"""

import csv
import os
import numpy as np
import jax.numpy as jnp
from pathlib import Path


class TelemetryLogger:
    """Logger for simulation telemetry data."""
    
    def __init__(self, output_dir, include_ice_water=False, config=None, append=False):
        """Initialize telemetry logger.
        
        Args:
            output_dir (str): Directory to save CSV files.
            include_ice_water (bool): Whether ice-water phase transition is enabled.
            config (dict, optional): Configuration dictionary for accessing parameters.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_ice_water = include_ice_water
        self.config = config
        self.append = append
        
        # CSV file paths
        self.stats_file = self.output_dir / "statistics.csv"
        self.boundary_stats_file = self.output_dir / "boundary_statistics.csv"
        self.ppe_file = self.output_dir / "ppe_updates.csv"
        
        # Initialize CSV files with headers
        self._init_stats_csv()
        self._init_boundary_stats_csv()
        self._init_ppe_csv()
        
        # Initialize ice-water specific CSV files if enabled
        if include_ice_water:
            self.ice_transition_file = self.output_dir / "ice_phase_transition.csv"
            self.temperature_file = self.output_dir / "temperature_evolution.csv"
            self._init_ice_transition_csv()
            self._init_temperature_evolution_csv()
    
    def _init_stats_csv(self):
        """Initialize statistics CSV file."""
        headers = [
            'step', 'time', 'dt',
            'phi_min', 'phi_max', 'phi_mean', 'phi_sum',
            'u_x_min', 'u_x_max', 'u_x_mean',
            'u_y_min', 'u_y_max', 'u_y_mean',
            'u_magnitude_max',
            'p_min', 'p_max', 'p_mean',
            'surface_tension_max',
            'curvature_max', 'curvature_mean',
            'droplet_mass',
            'droplet_start', 'droplet_end', 'droplet_bottom', 'droplet_top',
            'divergence_max', 'divergence_mean',
            'divergence_max_interior', 'divergence_mean_interior',
            'contact_left_index', 'contact_right_index',
            'cl_sf_norm_mean', 'cl_pg_norm_mean', 'cl_pg_dyn_norm_mean', 'cl_pg_hydro_norm_mean',
            'cl_g_norm', 'cl_sf_to_g_ratio', 'cl_sf_to_pg_dyn_ratio',
            'cl_sf_ax_mean_abs', 'cl_pg_dyn_ax_mean_abs',
            'cl_sf_to_pg_dyn_ratio_xabs',
            'cl_sf_n_mean', 'cl_pg_dyn_n_mean', 'cl_pg_h_n_mean', 'cl_g_n_mean', 'cl_res_n_mean',
            'cl_sf_t_mean', 'cl_pg_dyn_t_mean', 'cl_pg_h_t_mean', 'cl_g_t_mean', 'cl_res_t_mean',
            'cl_sf_norm_mean_liquid', 'cl_sf_norm_mean_gas',
            'cl_pg_dyn_norm_mean_liquid', 'cl_pg_dyn_norm_mean_gas'
        ]
        
        # Add geometry statistics
        headers.extend([
            'hump_height_max', 'hump_height_mean', 'hump_height_min',
            'surface_slope_max', 'surface_slope_mean'
        ])
        
        # Add ice-water statistics if enabled
        if self.include_ice_water:
            headers.extend([
                'psi_min', 'psi_max', 'psi_mean', 'psi_sum',
                'ice_fraction',
                'T_min', 'T_max', 'T_mean',
                'T_below_melt', 'T_above_melt'
            ])
        
        if self.append and self.stats_file.exists() and self.stats_file.stat().st_size > 0:
            return
        with open(self.stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def _init_boundary_stats_csv(self):
        """Initialize boundary statistics CSV file."""
        headers = ['step', 'time']
        # Add headers for each field at each boundary
        fields = ['phi', 'u_x', 'u_y', 'p', 'surface_tension_x', 'surface_tension_y']
        
        # Add ice-water fields if enabled
        if self.include_ice_water:
            fields.extend(['psi', 'T'])
        
        boundaries = ['left', 'right', 'top', 'bottom']
        stats = ['min', 'max', 'mean']
        
        for field in fields:
            for boundary in boundaries:
                for stat in stats:
                    headers.append(f'{field}_{boundary}_{stat}')
        
        if self.append and self.boundary_stats_file.exists() and self.boundary_stats_file.stat().st_size > 0:
            return
        with open(self.boundary_stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def _init_ppe_csv(self):
        """Initialize PPE updates CSV file."""
        headers = [
            'step', 'time',
            'ppe_applied', 'ppe_iterations',
            'divergence_before_max', 'divergence_before_mean',
            'divergence_after_max', 'divergence_after_mean',
            'divergence_threshold', 'max_div_threshold', 'mean_div_threshold'
        ]
        if self.append and self.ppe_file.exists() and self.ppe_file.stat().st_size > 0:
            return
        with open(self.ppe_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def _convert_to_numpy(self, arr):
        """Convert JAX array to NumPy if needed."""
        if isinstance(arr, jnp.ndarray):
            return np.array(arr)
        return arr
    
    def _init_ice_transition_csv(self):
        """Initialize ice phase transition CSV file."""
        headers = [
            'step', 'time',
            'ice_fraction', 'freezing_rate', 'melting_rate',
            'ice_mass', 'water_mass', 'ice_volume',
            'latent_heat_total', 'max_phase_change_rate',
            'T_mean_ice', 'T_min_ice', 'T_max_ice',
            'T_mean_water', 'T_min_water', 'T_max_water',
            'T_mean_interface', 'T_min_interface', 'T_max_interface',
            'freezing_front_y', 'interface_length',
            'max_temperature_coupling', 'mean_temperature_coupling',
            'subcooled_water_fraction', 'superheated_ice_fraction'
        ]
        if self.append and self.ice_transition_file.exists() and self.ice_transition_file.stat().st_size > 0:
            return
        with open(self.ice_transition_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def _init_temperature_evolution_csv(self):
        """Initialize temperature evolution CSV file."""
        headers = [
            'step', 'time',
            'T_min', 'T_max', 'T_mean', 'T_std',
            'T_gradient_max', 'T_gradient_mean',
            'T_bottom_mean', 'T_bottom_min', 'T_top_mean', 'T_top_min',
            'T_diff_bottom_melt', 'T_diff_top_melt',
            'latent_heat_rate',
            'subcooled_fraction', 'superheated_fraction',
            # Boundary diagnostics
            'T_bottom_cell0', 'T_bottom_cell1', 'T_bottom_cell2',  # First 3 cells at bottom
            'T_top_cell0', 'T_top_cell1', 'T_top_cell2',  # Last 3 cells at top
            'T_gradient_bottom', 'T_gradient_top',  # Gradient at boundaries
            # Diffusion diagnostics
            'laplacian_max', 'laplacian_mean', 'laplacian_bottom_cell1',
            'diffusion_term_max', 'diffusion_term_mean', 'diffusion_term_bottom_cell1',
            'alpha_bottom_mean', 'alpha_bottom_cell1',  # Thermal diffusivity
            # Latent heat diagnostics
            'latent_heat_max', 'latent_heat_mean', 'latent_heat_bottom_cell1',
            'dpsi_dt_bottom_cell1', 'dpsi_dt_max', 'dpsi_dt_mean',
            # Advection diagnostics
            'advection_term_max', 'advection_term_mean', 'advection_term_bottom_cell1',
            'A_bottom_mean', 'A_bottom_cell1',  # Advection function
            'u_magnitude_bottom_mean', 'u_magnitude_bottom_cell1',
            # Energy balance
            'energy_change_rate', 'diffusion_flux_bottom', 'latent_heat_flux_total'
        ]
        if self.append and self.temperature_file.exists() and self.temperature_file.stat().st_size > 0:
            return
        with open(self.temperature_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_statistics(self, step, time, dt, phi, U, P, surface_tension, 
                      mass, droplet_start, droplet_end, max_div, mean_div,
                      psi=None, T=None, geometry=None, dx=None,
                      droplet_bottom=None, droplet_top=None,
                      curvature_max=None, curvature_mean=None,
                      contact_line_forces=None,
                      max_div_interior=None, mean_div_interior=None):
        """Log general statistics. geometry: from state (optional, for terrain stats)."""
        # Convert to NumPy if needed
        phi = self._convert_to_numpy(phi)
        U = self._convert_to_numpy(U)
        P = self._convert_to_numpy(P)
        surface_tension = self._convert_to_numpy(surface_tension)
        
        # Calculate derived quantities
        U_magnitude = np.sqrt(U[..., 0]**2 + U[..., 1]**2)
        ST_magnitude = np.sqrt(surface_tension[..., 0]**2 + surface_tension[..., 1]**2)
        
        # Calculate geometry statistics (terrain f(x) from geometry)
        h_bottom = geometry.h_bottom if (geometry is not None and getattr(geometry, "has_geometry", False)) else None
        if h_bottom is not None:
            h_bottom_np = self._convert_to_numpy(h_bottom)
            hump_height_max = float(h_bottom_np.max())
            hump_height_mean = float(h_bottom_np.mean())
            hump_height_min = float(h_bottom_np.min())
            
            # Calculate surface slope (dh/dx)
            if dx is not None and len(h_bottom_np) > 1:
                dh_dx = np.gradient(h_bottom_np, dx)
                surface_slope_max = float(np.abs(dh_dx).max())
                surface_slope_mean = float(np.abs(dh_dx).mean())
            else:
                surface_slope_max = 0.0
                surface_slope_mean = 0.0
        else:
            hump_height_max = 0.0
            hump_height_mean = 0.0
            hump_height_min = 0.0
            surface_slope_max = 0.0
            surface_slope_mean = 0.0
        
        row = [
            step, time, dt,
            float(phi.min()), float(phi.max()), float(phi.mean()), float(phi.sum()),
            float(U[..., 0].min()), float(U[..., 0].max()), float(U[..., 0].mean()),
            float(U[..., 1].min()), float(U[..., 1].max()), float(U[..., 1].mean()),
            float(U_magnitude.max()),
            float(P.min()), float(P.max()), float(P.mean()),
            float(ST_magnitude.max()),
            float(curvature_max) if curvature_max is not None else 0.0,
            float(curvature_mean) if curvature_mean is not None else 0.0,
            float(mass),
            float(droplet_start), float(droplet_end),
            float(droplet_bottom) if droplet_bottom is not None else 0.0,
            float(droplet_top) if droplet_top is not None else 0.0,
            float(max_div), float(mean_div),
            float(max_div_interior) if max_div_interior is not None else float(max_div),
            float(mean_div_interior) if mean_div_interior is not None else float(mean_div),
            float((contact_line_forces or {}).get('left_index', -1.0)),
            float((contact_line_forces or {}).get('right_index', -1.0)),
            float((contact_line_forces or {}).get('sf_norm_mean', 0.0)),
            float((contact_line_forces or {}).get('pg_norm_mean', 0.0)),
            float((contact_line_forces or {}).get('pg_dyn_norm_mean', 0.0)),
            float((contact_line_forces or {}).get('pg_hydro_norm_mean', 0.0)),
            float((contact_line_forces or {}).get('g_norm', 0.0)),
            float((contact_line_forces or {}).get('sf_to_g_ratio', 0.0)),
            float((contact_line_forces or {}).get('sf_to_pg_dyn_ratio', 0.0)),
            float((contact_line_forces or {}).get('sf_ax_mean_abs', 0.0)),
            float((contact_line_forces or {}).get('pg_dyn_ax_mean_abs', 0.0)),
            float((contact_line_forces or {}).get('sf_to_pg_dyn_ratio_xabs', 0.0)),
            float((contact_line_forces or {}).get('sf_n_mean', 0.0)),
            float((contact_line_forces or {}).get('pg_dyn_n_mean', 0.0)),
            float((contact_line_forces or {}).get('pg_h_n_mean', 0.0)),
            float((contact_line_forces or {}).get('g_n_mean', 0.0)),
            float((contact_line_forces or {}).get('res_n_mean', 0.0)),
            float((contact_line_forces or {}).get('sf_t_mean', 0.0)),
            float((contact_line_forces or {}).get('pg_dyn_t_mean', 0.0)),
            float((contact_line_forces or {}).get('pg_h_t_mean', 0.0)),
            float((contact_line_forces or {}).get('g_t_mean', 0.0)),
            float((contact_line_forces or {}).get('res_t_mean', 0.0)),
            float((contact_line_forces or {}).get('sf_norm_mean_liquid', 0.0)),
            float((contact_line_forces or {}).get('sf_norm_mean_gas', 0.0)),
            float((contact_line_forces or {}).get('pg_dyn_norm_mean_liquid', 0.0)),
            float((contact_line_forces or {}).get('pg_dyn_norm_mean_gas', 0.0)),
            # Geometry statistics
            hump_height_max, hump_height_mean, hump_height_min,
            surface_slope_max, surface_slope_mean
        ]
        
        # Add ice-water statistics if enabled
        if self.include_ice_water and psi is not None and T is not None:
            psi = self._convert_to_numpy(psi)
            T = self._convert_to_numpy(T)
            ice_fraction = float(np.sum(psi > 0) / psi.size)
            T_below_melt = float(np.sum(T < 273.15) / T.size)  # Assuming T_melt = 273.15
            T_above_melt = float(np.sum(T > 273.15) / T.size)
            
            row.extend([
                float(psi.min()), float(psi.max()), float(psi.mean()), float(psi.sum()),
                ice_fraction,
                float(T.min()), float(T.max()), float(T.mean()),
                T_below_melt, T_above_melt
            ])
        
        with open(self.stats_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def log_boundary_statistics(self, step, time, phi, U, P, surface_tension, psi=None, T=None):
        """Log statistics at each boundary.
        
        Args:
            step (int): Current simulation step.
            time (float): Current simulation time.
            phi: Phase field.
            U: Velocity field.
            P: Pressure field.
            surface_tension: Surface tension force.
        """
        # Convert to NumPy if needed
        phi = self._convert_to_numpy(phi)
        U = self._convert_to_numpy(U)
        P = self._convert_to_numpy(P)
        surface_tension = self._convert_to_numpy(surface_tension)
        
        row = [step, time]
        
        # Extract boundary values
        boundaries = {
            'left': {'phi': phi[0, :], 'u_x': U[0, :, 0], 'u_y': U[0, :, 1],
                    'p': P[0, :], 'st_x': surface_tension[0, :, 0], 'st_y': surface_tension[0, :, 1]},
            'right': {'phi': phi[-1, :], 'u_x': U[-1, :, 0], 'u_y': U[-1, :, 1],
                     'p': P[-1, :], 'st_x': surface_tension[-1, :, 0], 'st_y': surface_tension[-1, :, 1]},
            'top': {'phi': phi[:, -1], 'u_x': U[:, -1, 0], 'u_y': U[:, -1, 1],
                   'p': P[:, -1], 'st_x': surface_tension[:, -1, 0], 'st_y': surface_tension[:, -1, 1]},
            'bottom': {'phi': phi[:, 0], 'u_x': U[:, 0, 0], 'u_y': U[:, 0, 1],
                      'p': P[:, 0], 'st_x': surface_tension[:, 0, 0], 'st_y': surface_tension[:, 0, 1]}
        }
        
        # Add ice-water fields if enabled
        if self.include_ice_water and psi is not None and T is not None:
            psi = self._convert_to_numpy(psi)
            T = self._convert_to_numpy(T)
            boundaries['left']['psi'] = psi[0, :]
            boundaries['left']['T'] = T[0, :]
            boundaries['right']['psi'] = psi[-1, :]
            boundaries['right']['T'] = T[-1, :]
            boundaries['top']['psi'] = psi[:, -1]
            boundaries['top']['T'] = T[:, -1]
            boundaries['bottom']['psi'] = psi[:, 0]
            boundaries['bottom']['T'] = T[:, 0]
        
        # Log statistics for each field at each boundary
        field_keys = {
            'phi': 'phi',
            'u_x': 'u_x',
            'u_y': 'u_y',
            'p': 'p',
            'surface_tension_x': 'st_x',
            'surface_tension_y': 'st_y'
        }
        
        fields_to_log = ['phi', 'u_x', 'u_y', 'p', 'surface_tension_x', 'surface_tension_y']
        if self.include_ice_water:
            fields_to_log.extend(['psi', 'T'])
            field_keys['psi'] = 'psi'
            field_keys['T'] = 'T'
        
        for field in fields_to_log:
            for boundary in ['left', 'right', 'top', 'bottom']:
                key = field_keys[field]
                if key in boundaries[boundary]:
                    values = boundaries[boundary][key]
                    row.extend([float(np.min(values)), float(np.max(values)), float(np.mean(values))])
                else:
                    row.extend([0.0, 0.0, 0.0])
        
        with open(self.boundary_stats_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def log_ppe_update(self, step, time, ppe_applied, ppe_iterations,
                      div_before_max, div_before_mean,
                      div_after_max, div_after_mean,
                      div_threshold, max_div_threshold, mean_div_threshold):
        """Log PPE correction step.
        
        Args:
            step (int): Current simulation step.
            time (float): Current simulation time.
            ppe_applied (bool): Whether PPE was applied.
            ppe_iterations (int): Number of PPE iterations.
            div_before_max (float): Max divergence before PPE.
            div_before_mean (float): Mean divergence before PPE.
            div_after_max (float): Max divergence after PPE.
            div_after_mean (float): Mean divergence after PPE.
            div_threshold (float): Divergence threshold.
            max_div_threshold (float): Max divergence threshold.
            mean_div_threshold (float): Mean divergence threshold.
        """
        row = [
            step, time,
            1 if ppe_applied else 0, ppe_iterations,
            float(div_before_max), float(div_before_mean),
            float(div_after_max), float(div_after_mean),
            float(div_threshold), float(max_div_threshold), float(mean_div_threshold)
        ]
        
        with open(self.ppe_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def log_ice_phase_statistics(self, step, time, psi, T, T_melt, rho_water, rho_ice, dx, dy, psi_old=None):
        """Log ice phase field and temperature statistics.
        
        Args:
            step, time: Standard parameters
            psi: Ice phase field (ψ)
            T: Temperature field
            T_melt: Melting temperature
            rho_water, rho_ice: Densities
            dx, dy: Grid spacing
            psi_old: Previous ice phase field (for rate calculation)
        """
        if not self.include_ice_water:
            return
        
        psi = self._convert_to_numpy(psi)
        T = self._convert_to_numpy(T)
        
        # Calculate statistics
        ice_fraction = float(np.sum(psi > 0) / psi.size)
        ice_mass = float(np.sum(np.where(psi > 0, rho_ice, 0)) * dx * dy)
        water_mass = float(np.sum(np.where(psi < 0, rho_water, 0)) * dx * dy)
        ice_volume = float(np.sum(psi > 0) * dx * dy)
        
        # Phase change rates
        if psi_old is not None:
            psi_old = self._convert_to_numpy(psi_old)
            dpsi_dt = (psi - psi_old)  # Change per step (not normalized by dt yet)
            freezing_rate = float(np.sum(np.where(dpsi_dt > 0, dpsi_dt, 0)) / psi.size)
            melting_rate = float(np.sum(np.where(dpsi_dt < 0, -dpsi_dt, 0)) / psi.size)
            max_dpsi_dt = float(np.max(np.abs(dpsi_dt)))
            mean_dpsi_dt = float(np.mean(np.abs(dpsi_dt)))
        else:
            freezing_rate = 0.0
            melting_rate = 0.0
            max_dpsi_dt = 0.0
            mean_dpsi_dt = 0.0
        
        # Temperature statistics
        T_mean_ice = float(np.mean(T[psi > 0])) if np.any(psi > 0) else T_melt
        T_mean_water = float(np.mean(T[psi < 0])) if np.any(psi < 0) else T_melt
        T_min_ice = float(np.min(T[psi > 0])) if np.any(psi > 0) else T_melt
        T_max_ice = float(np.max(T[psi > 0])) if np.any(psi > 0) else T_melt
        T_min_water = float(np.min(T[psi < 0])) if np.any(psi < 0) else T_melt
        T_max_water = float(np.max(T[psi < 0])) if np.any(psi < 0) else T_melt
        
        # Interface statistics
        interface_mask = np.abs(psi) < 0.5
        T_mean_interface = float(np.mean(T[interface_mask])) if np.any(interface_mask) else T_melt
        T_min_interface = float(np.min(T[interface_mask])) if np.any(interface_mask) else T_melt
        T_max_interface = float(np.max(T[interface_mask])) if np.any(interface_mask) else T_melt
        
        # Interface position (find where psi crosses zero, typically at bottom)
        # Find the highest y-position where psi > 0 (freezing front height)
        freezing_front_y = 0.0
        if np.any(psi > 0):
            # Find the maximum y-index where ice exists
            ice_mask = psi > 0
            y_indices = np.where(ice_mask.any(axis=0))[0]
            if len(y_indices) > 0:
                freezing_front_y = float(np.max(y_indices) / psi.shape[1])  # Normalized to [0, 1]
        
        # Interface length (perimeter of ice region)
        interface_length = float(np.sum(np.abs(np.diff((psi > 0).astype(float), axis=0))) + 
                                  np.sum(np.abs(np.diff((psi > 0).astype(float), axis=1))))
        
        # Driving force statistics (temperature coupling strength)
        # Match the actual implementation: linear coupling -lambda * (T_melt - T)
        # Get transition_width from config if available, otherwise use default
        ice_params = self.config.get("ice_water_params", {}) if hasattr(self, 'config') else {}
        transition_width = ice_params.get("transition_width", 0.5)
        epsilon_psi = ice_params.get("epsilon_psi", 0.02)
        lambda_coupling = 1.0 / (epsilon_psi * transition_width)
        
        # Calculate T_diff = T_melt - T (positive when T < T_melt, negative when T > T_melt)
        T_diff = T_melt - T
        # Temperature coupling = -lambda * T_diff (matches implementation)
        temperature_coupling_strength = -lambda_coupling * T_diff
        max_coupling = float(np.max(np.abs(temperature_coupling_strength)))
        mean_coupling = float(np.mean(np.abs(temperature_coupling_strength)))
        
        # Subcooling/superheating
        subcooled_water = np.sum((psi < 0) & (T < T_melt))  # Water below melting point
        superheated_ice = np.sum((psi > 0) & (T > T_melt))  # Ice above melting point
        subcooled_fraction = float(subcooled_water / psi.size)
        superheated_fraction = float(superheated_ice / psi.size)
        
        row = [
            step, time,
            ice_fraction, freezing_rate, melting_rate,
            ice_mass, water_mass, ice_volume,
            0.0, max_dpsi_dt,  # latent_heat_total, max_phase_change_rate
            T_mean_ice, T_min_ice, T_max_ice,
            T_mean_water, T_min_water, T_max_water,
            T_mean_interface, T_min_interface, T_max_interface,
            freezing_front_y, interface_length,
            max_coupling, mean_coupling,
            subcooled_fraction, superheated_fraction
        ]
        
        with open(self.ice_transition_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def log_temperature_evolution(self, step, time, T, T_melt, psi=None, dpsi_dt=None,
                                  alpha_water=None, alpha_ice=None, L=None, c_p_water=None, c_p_ice=None,
                                  U=None, dx=None, dy=None):
        """Log temperature evolution with detailed diagnostics.
        
        Args:
            step, time: Standard parameters
            T: Temperature field
            T_melt: Melting temperature
            psi: Ice phase field (for latent heat calculation)
            dpsi_dt: Phase change rate (for latent heat rate)
            alpha_water, alpha_ice: Thermal diffusivities
            L: Latent heat
            c_p_water, c_p_ice: Specific heats
            U: Velocity field (for advection diagnostics)
            dx, dy: Grid spacing (for gradient/Laplacian calculations)
        """
        if not self.include_ice_water:
            return
        
        T = self._convert_to_numpy(T)
        Nx, Ny = T.shape
        
        # Calculate statistics
        T_gradient = np.gradient(T)
        T_gradient_magnitude = np.sqrt(T_gradient[0]**2 + T_gradient[1]**2)
        
        subcooled_fraction = float(np.sum(T < T_melt) / T.size)
        superheated_fraction = float(np.sum(T > T_melt) / T.size)
        
        # Temperature at boundaries (important for freezing)
        T_bottom_mean = float(np.mean(T[:, 0]))
        T_bottom_min = float(np.min(T[:, 0]))
        T_top_mean = float(np.mean(T[:, -1]))
        T_top_min = float(np.min(T[:, -1]))
        
        # Boundary cell values (first 3 cells)
        T_bottom_cell0 = float(np.mean(T[:, 0])) if Ny > 0 else 0.0
        T_bottom_cell1 = float(np.mean(T[:, 1])) if Ny > 1 else 0.0
        T_bottom_cell2 = float(np.mean(T[:, 2])) if Ny > 2 else 0.0
        T_top_cell0 = float(np.mean(T[:, -1])) if Ny > 0 else 0.0
        T_top_cell1 = float(np.mean(T[:, -2])) if Ny > 1 else 0.0
        T_top_cell2 = float(np.mean(T[:, -3])) if Ny > 2 else 0.0
        
        # Temperature gradient at boundaries
        T_gradient_bottom = float(np.mean(np.abs(T[:, 1] - T[:, 0]))) if Ny > 1 and dy else 0.0
        T_gradient_top = float(np.mean(np.abs(T[:, -1] - T[:, -2]))) if Ny > 1 and dy else 0.0
        
        # Temperature difference (driving force)
        T_diff_bottom_melt = T_bottom_mean - T_melt  # Should be negative for freezing
        T_diff_top_melt = T_top_mean - T_melt
        
        # Diffusion diagnostics
        laplacian_max = 0.0
        laplacian_mean = 0.0
        laplacian_bottom_cell1 = 0.0
        diffusion_term_max = 0.0
        diffusion_term_mean = 0.0
        diffusion_term_bottom_cell1 = 0.0
        alpha_bottom_mean = 0.0
        alpha_bottom_cell1 = 0.0
        
        if dx and dy:
            # Calculate Laplacian (simplified 2D)
            lap_T = np.zeros_like(T)
            for i in range(1, Nx-1):
                for j in range(1, Ny-1):
                    lap_T[i, j] = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2 + \
                                  (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2
            
            laplacian_max = float(np.max(np.abs(lap_T)))
            laplacian_mean = float(np.mean(np.abs(lap_T)))
            laplacian_bottom_cell1 = float(np.mean(np.abs(lap_T[:, 1]))) if Ny > 1 else 0.0
            
            # Calculate thermal diffusivity if psi is available
            if psi is not None and alpha_water is not None and alpha_ice is not None:
                psi_np = self._convert_to_numpy(psi)
                psi_mapped = (psi_np + 1) / 2.0
                alpha = alpha_water * (1 - psi_mapped) + alpha_ice * psi_mapped
                
                # Diffusion term: alpha * laplacian
                diffusion_term = alpha * lap_T
                diffusion_term_max = float(np.max(np.abs(diffusion_term)))
                diffusion_term_mean = float(np.mean(np.abs(diffusion_term)))
                diffusion_term_bottom_cell1 = float(np.mean(np.abs(diffusion_term[:, 1]))) if Ny > 1 else 0.0
                
                alpha_bottom_mean = float(np.mean(alpha[:, 0]))
                alpha_bottom_cell1 = float(np.mean(alpha[:, 1])) if Ny > 1 else 0.0
        
        # Latent heat diagnostics
        latent_heat_rate = 0.0
        latent_heat_max = 0.0
        latent_heat_mean = 0.0
        latent_heat_bottom_cell1 = 0.0
        dpsi_dt_bottom_cell1 = 0.0
        dpsi_dt_max = 0.0
        dpsi_dt_mean = 0.0
        
        if psi is not None and dpsi_dt is not None:
            psi_np = self._convert_to_numpy(psi)
            dpsi_dt_np = self._convert_to_numpy(dpsi_dt)
            dpsi_dt_max = float(np.max(np.abs(dpsi_dt_np)))
            dpsi_dt_mean = float(np.mean(np.abs(dpsi_dt_np)))
            dpsi_dt_bottom_cell1 = float(np.mean(np.abs(dpsi_dt_np[:, 1]))) if Ny > 1 else 0.0
            
            # Estimate latent heat release rate
            latent_heat_rate = float(np.mean(np.abs(dpsi_dt_np)))  # Simplified metric
            
            if L is not None and c_p_water is not None and c_p_ice is not None:
                # Calculate phase-dependent c_p
                psi_mapped = (psi_np + 1) / 2.0
                c_p = c_p_water * (1 - psi_mapped) + c_p_ice * psi_mapped
                
                # Latent heat term: (L/c_p) * (1/2) * dpsi_dt
                latent_heat_term = 0.5 * (L / c_p) * dpsi_dt_np
                latent_heat_max = float(np.max(np.abs(latent_heat_term)))
                latent_heat_mean = float(np.mean(np.abs(latent_heat_term)))
                latent_heat_bottom_cell1 = float(np.mean(np.abs(latent_heat_term[:, 1]))) if Ny > 1 else 0.0
        
        # Advection diagnostics
        advection_term_max = 0.0
        advection_term_mean = 0.0
        advection_term_bottom_cell1 = 0.0
        A_bottom_mean = 0.0
        A_bottom_cell1 = 0.0
        u_magnitude_bottom_mean = 0.0
        u_magnitude_bottom_cell1 = 0.0
        
        if U is not None and dx and dy and psi is not None:
            U_np = self._convert_to_numpy(U)
            psi_np = self._convert_to_numpy(psi)
            
            # Advection function: A = 0.5 * (1 - tanh(psi / 0.1))
            A = 0.5 * (1.0 - np.tanh(psi_np / 0.1))
            A_bottom_mean = float(np.mean(A[:, 0]))
            A_bottom_cell1 = float(np.mean(A[:, 1])) if Ny > 1 else 0.0
            
            # Velocity magnitude
            u_magnitude = np.sqrt(U_np[..., 0]**2 + U_np[..., 1]**2)
            u_magnitude_bottom_mean = float(np.mean(u_magnitude[:, 0]))
            u_magnitude_bottom_cell1 = float(np.mean(u_magnitude[:, 1])) if Ny > 1 else 0.0
            
            # Advection term: -A * u·∇T
            T_grad_x = np.gradient(T, axis=0) / dx if dx else np.zeros_like(T)
            T_grad_y = np.gradient(T, axis=1) / dy if dy else np.zeros_like(T)
            advection_term = -A[..., np.newaxis] * (U_np[..., 0:1] * T_grad_x[..., np.newaxis] + 
                                                     U_np[..., 1:2] * T_grad_y[..., np.newaxis])
            advection_term = advection_term.squeeze()
            advection_term_max = float(np.max(np.abs(advection_term)))
            advection_term_mean = float(np.mean(np.abs(advection_term)))
            advection_term_bottom_cell1 = float(np.mean(np.abs(advection_term[:, 1]))) if Ny > 1 else 0.0
        
        # Energy balance (simplified)
        energy_change_rate = 0.0
        diffusion_flux_bottom = 0.0
        latent_heat_flux_total = 0.0
        
        if dx and dy and alpha_water is not None:
            # Diffusion flux at bottom: -alpha * dT/dy
            if Ny > 1:
                dT_dy_bottom = (T[:, 1] - T[:, 0]) / dy
                diffusion_flux_bottom = float(-np.mean(alpha_bottom_mean * dT_dy_bottom)) if alpha_bottom_mean > 0 else 0.0
            
            # Total latent heat flux (simplified)
            if latent_heat_mean > 0:
                latent_heat_flux_total = latent_heat_mean * T.size * dx * dy  # Approximate
        
        row = [
            step, time,
            float(T.min()), float(T.max()), float(T.mean()), float(np.std(T)),
            float(T_gradient_magnitude.max()), float(T_gradient_magnitude.mean()),
            T_bottom_mean, T_bottom_min, T_top_mean, T_top_min,
            T_diff_bottom_melt, T_diff_top_melt,
            latent_heat_rate,
            subcooled_fraction, superheated_fraction,
            # Boundary diagnostics
            T_bottom_cell0, T_bottom_cell1, T_bottom_cell2,
            T_top_cell0, T_top_cell1, T_top_cell2,
            T_gradient_bottom, T_gradient_top,
            # Diffusion diagnostics
            laplacian_max, laplacian_mean, laplacian_bottom_cell1,
            diffusion_term_max, diffusion_term_mean, diffusion_term_bottom_cell1,
            alpha_bottom_mean, alpha_bottom_cell1,
            # Latent heat diagnostics
            latent_heat_max, latent_heat_mean, latent_heat_bottom_cell1,
            dpsi_dt_bottom_cell1, dpsi_dt_max, dpsi_dt_mean,
            # Advection diagnostics
            advection_term_max, advection_term_mean, advection_term_bottom_cell1,
            A_bottom_mean, A_bottom_cell1,
            u_magnitude_bottom_mean, u_magnitude_bottom_cell1,
            # Energy balance
            energy_change_rate, diffusion_flux_bottom, latent_heat_flux_total
        ]
        
        with open(self.temperature_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
