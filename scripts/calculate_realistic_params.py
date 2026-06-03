#!/usr/bin/env python3
"""
Calculate realistic non-dimensional parameters for droplet simulation.

This script computes Re, We, Fr, Pe, etc. based on:
- Droplet size (radius in mm)
- Real physical properties (water, air)
- Characteristic velocity and length scales
"""

import math
import json
import sys
import argparse


# Physical properties at 20°C (standard conditions)
PHYSICAL_PROPERTIES = {
    # Air properties
    "rho_air": 1.225,  # kg/m³
    "mu_air": 1.82e-5,  # Pa·s (dynamic viscosity)
    "nu_air": 1.48e-5,  # m²/s (kinematic viscosity)
    
    # Water properties
    "rho_water": 998.2,  # kg/m³ (at 20°C)
    "mu_water": 1.002e-3,  # Pa·s (dynamic viscosity)
    "nu_water": 1.004e-6,  # m²/s (kinematic viscosity)
    
    # Surface tension (water-air interface)
    "sigma": 0.0728,  # N/m (at 20°C)
    
    # Gravity
    "g": 9.81,  # m/s²
    
    # Phase field properties
    "mobility": 1.0e-6,  # m³·s/kg (typical for Cahn-Hilliard, adjusted for realistic Pe)
    "interface_thickness": 1e-5,  # m (10 microns, typical)
}


def calculate_capillary_velocity(sigma, rho, R):
    """Calculate capillary velocity scale: U_cap = sqrt(sigma / (rho * R))
    
    This is the characteristic velocity for capillary-driven flows.
    """
    return math.sqrt(sigma / (rho * R))


def calculate_characteristic_scales(droplet_radius_mm, contact_angle_deg=60):
    """Calculate characteristic length, velocity, and time scales.
    
    Args:
        droplet_radius_mm: Droplet radius in millimeters
        contact_angle_deg: Contact angle in degrees
    
    Returns:
        dict with scales and non-dimensional numbers
    """
    # Convert to meters
    R = droplet_radius_mm * 1e-3  # m
    
    # Characteristic length: droplet radius
    L_char = R
    
    # Characteristic velocity: capillary velocity (based on water properties)
    U_char = calculate_capillary_velocity(
        PHYSICAL_PROPERTIES["sigma"],
        PHYSICAL_PROPERTIES["rho_water"],
        R
    )
    
    # Characteristic time: capillary time
    t_char = R / U_char
    
    # Domain size: should be 4-5× droplet size to avoid boundary effects
    domain_size = 5.0 * R
    
    # Calculate non-dimensional numbers
    # Reynolds number: Re = ρUL/μ
    Re_air = (PHYSICAL_PROPERTIES["rho_air"] * U_char * L_char / 
              PHYSICAL_PROPERTIES["mu_air"])
    Re_water = (PHYSICAL_PROPERTIES["rho_water"] * U_char * L_char / 
                PHYSICAL_PROPERTIES["mu_water"])
    
    # Weber number: We = ρU²L/σ
    We_air = (PHYSICAL_PROPERTIES["rho_air"] * U_char**2 * L_char / 
              PHYSICAL_PROPERTIES["sigma"])
    We_water = (PHYSICAL_PROPERTIES["rho_water"] * U_char**2 * L_char / 
                PHYSICAL_PROPERTIES["sigma"])
    
    # Froude number: Fr = U/√(gL)
    Fr = U_char / math.sqrt(PHYSICAL_PROPERTIES["g"] * L_char)
    
    # Bond number: Bo = ρgL²/σ (gravity vs surface tension)
    # Bo < 1: Surface tension dominates (small droplets)
    # Bo > 1: Gravity dominates (large droplets)
    Bo = (PHYSICAL_PROPERTIES["rho_water"] * PHYSICAL_PROPERTIES["g"] * L_char**2 / 
          PHYSICAL_PROPERTIES["sigma"])
    
    # Capillary number: Ca = μU/σ (viscous vs surface tension)
    # Ca < 1: Surface tension dominates
    Ca_air = PHYSICAL_PROPERTIES["mu_air"] * U_char / PHYSICAL_PROPERTIES["sigma"]
    Ca_water = PHYSICAL_PROPERTIES["mu_water"] * U_char / PHYSICAL_PROPERTIES["sigma"]
    
    # Peclet number: Pe = UL/D (advection vs diffusion)
    # For phase field, use a more realistic diffusion coefficient
    # D ~ interface_thickness² / characteristic_time gives reasonable Pe
    # Or use: Pe = U * L / (M * sigma * interface_thickness)
    # For realistic values, Pe should be 10-100 for good interface resolution
    # We'll calculate based on interface thickness and adjust mobility accordingly
    target_pe = 10.0  # Target Peclet number for good interface resolution
    D_effective = U_char * L_char / target_pe
    Pe = target_pe  # Use target value
    
    # Interface thickness (non-dimensional): epsilon = δ/L
    epsilon = PHYSICAL_PROPERTIES["interface_thickness"] / L_char
    
    return {
        "droplet_radius_mm": droplet_radius_mm,
        "droplet_radius_m": R,
        "domain_size_m": domain_size,
        "characteristic_length": L_char,
        "characteristic_velocity": U_char,
        "characteristic_time": t_char,
        "Re1": Re_air,
        "Re2": Re_water,
        "We1": We_air,
        "We2": We_water,
        "Fr": Fr,
        "Bo": Bo,
        "Ca1": Ca_air,
        "Ca2": Ca_water,
        "Pe": Pe,
        "epsilon": epsilon,
        "contact_angle": contact_angle_deg,
    }


def generate_config(params, output_file=None):
    """Generate a config JSON from calculated parameters.
    
    Args:
        params: Dictionary with calculated parameters
        output_file: Optional file to write config to
    """
    # Non-dimensionalize domain size (Lx = Ly = 1.0 in simulation units)
    # So droplet_radius in simulation units = R / domain_size
    droplet_radius_nd = params["droplet_radius_m"] / params["domain_size_m"]
    
    # Ensure reasonable values (clip if needed)
    Re1 = max(0.1, min(params["Re1"], 100.0))
    Re2 = max(1.0, min(params["Re2"], 1000.0))
    We1 = max(1e-6, min(params["We1"], 10.0))
    We2 = max(0.001, min(params["We2"], 100.0))
    Fr = max(0.01, min(params["Fr"], 10.0))
    Pe = max(1.0, min(params["Pe"], 100.0))
    # Epsilon: ensure at least 2-3 grid points across interface
    # For 128x128 grid, dx ≈ 1/128 ≈ 0.008, so epsilon should be ≥ 0.015-0.02
    epsilon = max(0.015, min(params["epsilon"], 0.05))
    
    config = {
        "description": f"Realistic {params['droplet_radius_mm']:.1f}mm water droplet (auto-generated from physical properties)",
        "physical_params": {
            "rho1": PHYSICAL_PROPERTIES["rho_air"] / PHYSICAL_PROPERTIES["rho_water"],
            "rho2": 1.0,
            "Re1": float(Re1),
            "Re2": float(Re2),
            "We1": float(We1),
            "We2": float(We2),
            "Pe": float(Pe),
            "epsilon": float(epsilon),
            "contact_angle": params["contact_angle"],
            "include_gravity": True,
            "Fr": float(Fr),
            "g": -1.0,
            "atm_pressure": 0.0,
            "lambda_willmore": 0.001,
            "epsilon_willmore": float(epsilon * 0.001)
        },
        "grid_params": {
            "Lx": 1.0,
            "Ly": 1.0,
            "Nx": 128,
            "Ny": 128
        },
        "time_params": {
            "dt": float(params["characteristic_time"] * 0.01),  # Small fraction of characteristic time
            "dt_initial": float(params["characteristic_time"] * 0.005),
            "t_max": float(params["characteristic_time"] * 10),  # 10 characteristic times
            "checkpoint_interval": 250,
            "cfl_number": 0.2,
            "capillary_cfl_number": 0.2,
            "curvature_cfl_number": 0.2
        },
        "initial_conditions": {
            "droplet_radius": float(droplet_radius_nd),
            "droplet_center_x": 0.5,
            "droplet_center_y": 0.0
        },
        "boundary_conditions": {
            "pressure": {
                "top": "open",
                "bottom": "neumann",
                "left": "neumann",
                "right": "neumann",
                "open_pressure": 0.0
            },
            "velocity": {
                "top": "do_nothing",
                "bottom": "no_slip",
                "left": "slip_symmetry",
                "right": "slip_symmetry"
            },
            "phase_field": {
                "top": "neumann",
                "bottom": "contact_angle",
                "left": "neumann",
                "right": "neumann",
                "contact_angle_method": "robin"
            },
            "chemical_potential": {
                "top": "zero_flux",
                "bottom": "zero_flux",
                "left": "zero_flux",
                "right": "zero_flux"
            },
            "advection": {
                "top": "open",
                "bottom": "impermeable",
                "left": "impermeable",
                "right": "impermeable",
                "cout": 1.0
            }
        },
        "solver_params": {
            "pressure_solver": {
                "backend": "pyamg",
                "accel": "bicgstab",
                "tol": 0.05,
                "maxiter": 15000
            },
            "correction_solver": {
                "backend": "pyamg",
                "accel": "bicgstab",
                "tol": 0.05,
                "maxiter": 15000
            },
            "ppe": {
                "mean_div_threshold": 0.05,
                "max_div_threshold": 0.1,
                "use_local_ppe": True,
                "local_threshold_factor": 0.1,
                "buffer_size": 5,
                "boundary_conditions": {
                    "top": "neumann",
                    "bottom": "neumann",
                    "left": "neumann",
                    "right": "neumann"
                }
            }
        },
        "restart": {
            "restart_from": None
        }
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Config written to {output_file}")
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Calculate realistic non-dimensional parameters for droplet simulation"
    )
    parser.add_argument(
        "--radius", type=float, default=3.0,
        help="Droplet radius in mm (default: 3.0)"
    )
    parser.add_argument(
        "--contact-angle", type=float, default=60.0,
        help="Contact angle in degrees (default: 60)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output config file (default: print to stdout)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed parameter information"
    )
    
    args = parser.parse_args()
    
    # Calculate parameters
    params = calculate_characteristic_scales(args.radius, args.contact_angle)
    
    if args.verbose:
        print("=" * 60)
        print("PHYSICAL PARAMETERS")
        print("=" * 60)
        print(f"Droplet radius: {params['droplet_radius_mm']:.2f} mm = {params['droplet_radius_m']*1e3:.3f} mm")
        print(f"Domain size: {params['domain_size_m']*1e3:.2f} mm")
        print(f"Characteristic length: {params['characteristic_length']*1e3:.3f} mm")
        print(f"Characteristic velocity: {params['characteristic_velocity']:.4f} m/s")
        print(f"Characteristic time: {params['characteristic_time']:.6f} s")
        print()
        print("=" * 60)
        print("NON-DIMENSIONAL NUMBERS")
        print("=" * 60)
        print(f"Re1 (air):     {params['Re1']:.4f}")
        print(f"Re2 (water):   {params['Re2']:.4f}")
        print(f"We1 (air):     {params['We1']:.6f}")
        print(f"We2 (water):   {params['We2']:.4f}")
        print(f"Fr (Froude):   {params['Fr']:.4f}")
        print(f"Bo (Bond):     {params['Bo']:.4f}")
        print(f"Ca1 (air):     {params['Ca1']:.6f}")
        print(f"Ca2 (water):   {params['Ca2']:.6f}")
        print(f"Pe (Peclet):   {params['Pe']:.2f}")
        print(f"epsilon:       {params['epsilon']:.6f}")
        print()
        print("=" * 60)
        print("PHYSICAL INTERPRETATION")
        print("=" * 60)
        if params['Bo'] < 1:
            print("✓ Surface tension dominates gravity (Bo < 1) - droplet shape controlled by surface tension")
        else:
            print("⚠ Gravity dominates (Bo > 1) - droplet may flatten significantly")
        if params['Ca2'] < 1:
            print("✓ Surface tension dominates viscosity (Ca < 1) - interface shape preserved")
        else:
            print("⚠ Viscous forces significant (Ca > 1) - interface may deform")
        print()
        print("PARAMETER GUIDE:")
        print("  - Lower We = Stronger surface tension (more resistance to spreading)")
        print("  - Lower Bo = Surface tension > gravity (smaller droplets)")
        print("  - Lower Ca = Surface tension > viscosity (preserves interface)")
        print("  - Higher Re = Less viscous damping (more flow)")
        print()
    
    # Generate config
    config = generate_config(params, args.output)
    
    if not args.output:
        print(json.dumps(config, indent=4))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
