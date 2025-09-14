import numpy as np

"""
Non-dimensional numbers are calculated as follows:

Reynolds number (Re): Ratio of inertial forces to viscous forces
Re = (rho * U * L) / mu 
where rho is density, U is velocity, L is length scale, mu is dynamic viscosity

Weber number (We): Ratio of inertial forces to surface tension forces
We = (rho * U^2 * L) / sigma
where rho is density, U is velocity, L is length scale, sigma is surface tension

Froude number (Fr): Ratio of inertial forces to gravitational forces  
Fr = U / sqrt(g*L)
where U is velocity, g is gravitational acceleration, L is length scale

Peclet number (Pe): Ratio of advective to diffusive transport
Pe = (U * L) / D
where U is velocity, L is length scale, D is diffusion coefficient
"""

def calculate_sigma(We, rho, length):
    return We * rho * length**2

def calculate_nu(Re, rho, length):
    return Re * rho * length

def calculate_mu(nu, rho):
    return nu * rho

def calculate_params_from_dict(params_dict):
    rho = params_dict["rho"]
    Re = params_dict["Re"]
    length = params_dict["length"]
    g = params_dict["g"]
    We = params_dict["We"]


    nu = calculate_nu(Re, rho, length)
    sigma = calculate_sigma(We, rho, length)
    print(f"nu: {nu:.8f} sigma: {sigma:.8f}")
    return {
        "nu": nu,
        "sigma": sigma,
    }


if __name__ == "__main__":
    water_params = {
            "rho": 1000,      # kg/m³
            "Re": 10,
            "We": 10.0,
            "g": 1.0,        # m/s²
            "rho":1000,
            "length": 0.002,  #
    }
    
    air_params = {
            "rho": 1.0,      # kg/m³
            "Re": 150,
            "We": 1.0,
            "g": 1.0,        # m/s²
            "rho": 1.0,
            "length": 0.002,  #
    }

    print("\nWater params:")
    calculate_params_from_dict(water_params)
    print("\nAir params:")
    calculate_params_from_dict(air_params)


    
