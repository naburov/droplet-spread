import numpy as np

def calculate_reynolds_number(rho, velocity, length, mu):
    """
    Calculate the Reynolds number.
    
    Parameters:
    -----------
    rho : float
        Density of the fluid (kg/m^3)
    velocity : float
        Characteristic velocity (m/s)
    length : float
        Characteristic length (m)
    mu : float
        Dynamic viscosity (Pa·s)
        
    Returns:
    --------
    float
        Reynolds number (dimensionless)
    """
    return rho * velocity * length / mu


def calculate_weber_number(rho, velocity, length, sigma):
    """
    Calculate the Weber number.
    
    Parameters:
    -----------
    rho : float
        Density of the fluid (kg/m^3)
    velocity : float
        Characteristic velocity (m/s)
    length : float
        Characteristic length (m)
    sigma : float
        Surface tension (N/m)
        
    Returns:
    --------
    float
        Weber number (dimensionless)
    """
    return rho * velocity**2 * length / sigma


def calculate_peclet_number(velocity, length, diffusivity):
    """
    Calculate the Peclet number.
    
    Parameters:
    -----------
    velocity : float
        Characteristic velocity (m/s)
    length : float
        Characteristic length (m)
    diffusivity : float
        Mass diffusivity (m^2/s)
        
    Returns:
    --------
    float
        Peclet number (dimensionless)
    """
    return velocity * length / diffusivity


def calculate_froude_number(velocity, length, g):
    """
    Calculate the Froude number.
    
    Parameters:
    -----------
    velocity : float
        Characteristic velocity (m/s)
    length : float
        Characteristic length (m)
    g : float
        Gravitational acceleration (m/s^2)
        
    Returns:
    --------
    float
        Froude number (dimensionless)
    """
    return velocity / np.sqrt(g * length)

def calculate_params_from_dict(params_dict):
    rho = params_dict["rho"]
    mu = params_dict["mu"]
    sigma = params_dict["sigma"]
    g = params_dict["g"]
    velocity = params_dict["velocity"]
    length = params_dict["length"]
    diffusivity = params_dict["diffusivity"]

    Re = calculate_reynolds_number(rho, velocity, length, mu)
    We = calculate_weber_number(rho, velocity, length, sigma)
    Pe = calculate_peclet_number(velocity, length, diffusivity)
    Fr = calculate_froude_number(velocity, length, g)
    print(f"Re: {Re:.8f}, We: {We:.8f}, Pe: {Pe:.8f}, Fr: {Fr:.8f}")
    return {
        "Re": Re,
        "We": We,
        "Pe": Pe,
        "Fr": Fr
    }


if __name__ == "__main__":
    water_params = {
            "rho": 1000,      # kg/m³
            "mu": 0.001,      # Pa·s
            "sigma": 0.0728,  # N/m
            "g": 9.81,        # m/s²
            "length": 0.002,  # 2mm droplet diameter
            "velocity": 0.05, # Slower velocity for stable formation
            "diffusivity": 1.43e-7
    }
    
    air_params = {
        "rho": 1.204,      # kg/m³ (correct density of air)
        "mu": 1.82e-5,     # Pa·s (correct viscosity of air)
        "sigma": 0.0728,   # N/m (same as water-air interface)
        "g": 9.81,         # m/s² (same gravity)
        "length": 0.002,   # 2mm - MUST match water length scale
        "velocity": 0.05,  # MUST match water velocity scale
        "diffusivity": 2e-5  # m²/s (correct diffusivity)
    }

    print("\nWater params:")
    calculate_params_from_dict(water_params)
    print("\nAir params:")
    calculate_params_from_dict(air_params)


    
