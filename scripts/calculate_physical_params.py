#!/usr/bin/env python3
"""
Calculate non-dimensional parameters (Re, We, Pe, Fr) from physical properties.

This script can:
1. Calculate parameters from physical properties
2. Update existing config files with recalculated parameters
3. Validate parameter consistency
"""

import json
import math
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def calculate_reynolds_number(rho, velocity, length, mu):
    """
    Calculate the Reynolds number: Re = ρ U L / μ
    
    Parameters:
    -----------
    rho : float
        Density (kg/m³)
    velocity : float
        Characteristic velocity (m/s)
    length : float
        Characteristic length (m)
    mu : float
        Dynamic viscosity (Pa·s)
        
    Returns:
    --------
    float
        Reynolds number
    """
    return rho * velocity * length / mu


def calculate_weber_number(rho, velocity, length, sigma):
    """
    Calculate the Weber number: We = ρ U² L / σ
    
    Parameters:
    -----------
    rho : float
        Density (kg/m³)
    velocity : float
        Characteristic velocity (m/s)
    length : float
        Characteristic length (m)
    sigma : float
        Surface tension (N/m)
        
    Returns:
    --------
    float
        Weber number
    """
    return rho * velocity**2 * length / sigma


def calculate_peclet_number(velocity, length, mobility, epsilon):
    """
    Calculate the Peclet number: Pe = U L / (M ε²)
    
    Parameters:
    -----------
    velocity : float
        Characteristic velocity (m/s)
    length : float
        Characteristic length (m)
    mobility : float
        Mobility coefficient (m³·s/kg)
    epsilon : float
        Interface thickness parameter (m)
        
    Returns:
    --------
    float
        Peclet number
    """
    diffusivity = mobility * epsilon**2
    return velocity * length / diffusivity


def calculate_froude_number(velocity, length, g):
    """
    Calculate the Froude number: Fr = U / √(g L)
    
    Parameters:
    -----------
    velocity : float
        Characteristic velocity (m/s)
    length : float
        Characteristic length (m)
    g : float
        Gravitational acceleration (m/s²)
        
    Returns:
    --------
    float
        Froude number
    """
    return velocity / math.sqrt(abs(g) * length)


def calculate_from_physical_properties(physical_params, L0=None):
    """
    Calculate non-dimensional parameters from physical properties.
    
    Parameters:
    -----------
    physical_params : dict
        Dictionary with physical properties:
        - rho1, rho2: densities (kg/m³)
        - mu1, mu2: viscosities (Pa·s)
        - sigma: surface tension (N/m)
        - U0: characteristic velocity (m/s)
        - L0: characteristic length (m) [optional, uses droplet_radius if not provided]
        - M: mobility (m³·s/kg)
        - epsilon: interface thickness (m)
        - g: gravity (m/s²)
        - droplet_radius: initial droplet radius (m)
    
    Returns:
    --------
    dict
        Dictionary with calculated non-dimensional parameters
    """
    # Extract parameters
    rho1 = physical_params.get('rho1', 1.225)  # Air at STP
    rho2 = physical_params.get('rho2', 1000.0)  # Water
    mu1 = physical_params.get('mu1', 1.82e-5)    # Air viscosity
    mu2 = physical_params.get('mu2', 0.001)     # Water viscosity
    sigma = physical_params.get('sigma', 0.0728) # Water-air surface tension
    U0 = physical_params.get('U0', 0.01)        # Characteristic velocity
    M = physical_params.get('M', 1e-3)           # Mobility
    epsilon = physical_params.get('epsilon', 0.05)
    g = physical_params.get('g', 9.81)
    
    # Use droplet radius as characteristic length if L0 not provided
    if L0 is None:
        L0 = physical_params.get('droplet_radius', 0.2)
        if L0 is None:
            L0 = physical_params.get('L0', 0.01)  # Default 1 cm
    
    # Calculate parameters
    Re1 = calculate_reynolds_number(rho1, U0, L0, mu1)
    Re2 = calculate_reynolds_number(rho2, U0, L0, mu2)
    We1 = calculate_weber_number(rho1, U0, L0, sigma)
    We2 = calculate_weber_number(rho2, U0, L0, sigma)
    Pe = calculate_peclet_number(U0, L0, M, epsilon)
    Fr = calculate_froude_number(U0, L0, g)
    
    return {
        'Re1': Re1,
        'Re2': Re2,
        'We1': We1,
        'We2': We2,
        'Pe': Pe,
        'Fr': Fr
    }


def update_config_file(config_path, physical_params=None, recalculate=True):
    """
    Update a config file with recalculated parameters.
    
    Parameters:
    -----------
    config_path : str
        Path to config file
    physical_params : dict, optional
        Physical properties to use. If None, uses values from config.
    recalculate : bool
        Whether to recalculate parameters
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if not recalculate:
        print("Config file loaded (no recalculation)")
        return config
    
    # Get physical properties
    if physical_params is None:
        phys = config.get('physical_params', {})
        # Add defaults for missing physical properties
        phys.setdefault('rho1', 1.225)      # Air
        phys.setdefault('rho2', 1000.0)     # Water
        phys.setdefault('mu1', 1.82e-5)     # Air viscosity
        phys.setdefault('mu2', 0.001)       # Water viscosity
        phys.setdefault('sigma', 0.0728)    # Surface tension
        phys.setdefault('U0', 0.01)         # Characteristic velocity
        phys.setdefault('M', 1e-3)          # Mobility
        phys.setdefault('epsilon', phys.get('epsilon', 0.05))
        phys.setdefault('g', phys.get('g', -9.81))
        phys.setdefault('droplet_radius', config.get('initial_conditions', {}).get('droplet_radius', 0.2))
    else:
        phys = physical_params
    
    # Calculate parameters
    L0 = config.get('initial_conditions', {}).get('droplet_radius', 0.2)
    params = calculate_from_physical_properties(phys, L0=L0)
    
    # Update config
    if 'physical_params' not in config:
        config['physical_params'] = {}
    
    config['physical_params'].update({
        'Re1': params['Re1'],
        'Re2': params['Re2'],
        'We1': params['We1'],
        'We2': params['We2'],
        'Pe': params['Pe'],
        'Fr': params['Fr']
    })
    
    # Keep existing physical properties
    for key in ['rho1', 'rho2', 'epsilon', 'contact_angle', 'include_gravity', 'g', 'atm_pressure']:
        if key in phys:
            config['physical_params'][key] = phys[key]
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Calculate and update non-dimensional parameters')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--output', type=str, help='Output config file (default: overwrite input)')
    parser.add_argument('--dry-run', action='store_true', help='Print calculated values without updating file')
    parser.add_argument('--U0', type=float, help='Characteristic velocity (m/s)')
    parser.add_argument('--mu1', type=float, help='Air viscosity (Pa·s)')
    parser.add_argument('--mu2', type=float, help='Water viscosity (Pa·s)')
    parser.add_argument('--sigma', type=float, help='Surface tension (N/m)')
    parser.add_argument('--M', type=float, help='Mobility (m³·s/kg)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Prepare physical parameters
    phys = config.get('physical_params', {})
    if args.U0:
        phys['U0'] = args.U0
    if args.mu1:
        phys['mu1'] = args.mu1
    if args.mu2:
        phys['mu2'] = args.mu2
    if args.sigma:
        phys['sigma'] = args.sigma
    if args.M:
        phys['M'] = args.M
    
    # Set defaults
    phys.setdefault('rho1', 1.225)
    phys.setdefault('rho2', 1000.0)
    phys.setdefault('mu1', 1.82e-5)
    phys.setdefault('mu2', 0.001)
    phys.setdefault('sigma', 0.0728)
    phys.setdefault('U0', 0.01)
    phys.setdefault('M', 1e-3)
    phys.setdefault('epsilon', phys.get('epsilon', 0.05))
    phys.setdefault('g', phys.get('g', -9.81))
    phys.setdefault('droplet_radius', config.get('initial_conditions', {}).get('droplet_radius', 0.2))
    
    # Calculate parameters
    L0 = config.get('initial_conditions', {}).get('droplet_radius', 0.2)
    params = calculate_from_physical_properties(phys, L0=L0)
    
    # Print results
    print("=" * 60)
    print("Physical Properties:")
    print(f"  rho1 (air):     {phys['rho1']:.6f} kg/m³")
    print(f"  rho2 (water):   {phys['rho2']:.6f} kg/m³")
    print(f"  mu1 (air):      {phys['mu1']:.6e} Pa·s")
    print(f"  mu2 (water):    {phys['mu2']:.6e} Pa·s")
    print(f"  sigma:          {phys['sigma']:.6f} N/m")
    print(f"  U0:             {phys['U0']:.6f} m/s")
    print(f"  L0 (radius):    {L0:.6f} m")
    print(f"  M:              {phys['M']:.6e} m³·s/kg")
    print(f"  epsilon:        {phys['epsilon']:.6f}")
    print(f"  g:              {phys['g']:.6f} m/s²")
    print()
    print("Calculated Non-Dimensional Parameters:")
    print(f"  Re1 (air):      {params['Re1']:.6f}")
    print(f"  Re2 (water):    {params['Re2']:.6f}")
    print(f"  We1 (air):      {params['We1']:.6e}")
    print(f"  We2 (water):    {params['We2']:.6f}")
    print(f"  Pe:             {params['Pe']:.6f}")
    print(f"  Fr:             {params['Fr']:.6f}")
    print("=" * 60)
    
    # Update config if not dry-run
    if not args.dry_run:
        config['physical_params'].update({
            'Re1': params['Re1'],
            'Re2': params['Re2'],
            'We1': params['We1'],
            'We2': params['We2'],
            'Pe': params['Pe'],
            'Fr': params['Fr']
        })
        
        output_path = args.output or args.config
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"\n✓ Updated config file: {output_path}")
    else:
        print("\n(Dry run - no file updated)")


if __name__ == '__main__':
    main()

