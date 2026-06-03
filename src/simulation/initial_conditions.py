"""
Initial condition setup for droplet spreading simulation.

This module contains functions for setting up initial conditions
for the droplet spreading simulation.
"""

import numpy as np
import jax.numpy as jnp


def initialize_phase(Nx, Ny, radius, epsilon=0.03, config=None):
    """Initialize the phase field with a semicircle droplet on the surface.

    Uses only computational (x, eta) coordinates. The grid is terrain-following:
    eta=0 (Y=0) is the fluid surface; there is no h_bottom or physical height.
    Droplet is a circle in (x, eta) centered at (center_x, center_y), giving
    a semicircle when center_y=0 (circle intersects surface at eta=0).

    Args:
        Nx, Ny: grid size.
        radius: droplet radius (same units as normalized [0,1] coords).
        epsilon: interface thickness for tanh.
        config: optional dict; initial_conditions.droplet_center_x/y, is_bubble.

    Returns:
        np.ndarray: phase field (phi < 0 liquid, phi > 0 air).
    """
    # Computational grid: x, eta in [0, 1]. Surface is at eta=0 (bottom row).
    x = np.linspace(0, 1, Nx)
    eta = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, eta, indexing="ij")

    ic = config.get("initial_conditions", {}) if config else {}
    center_x = float(ic.get("droplet_center_x", 0.5))
    center_y = float(ic.get("droplet_center_y", 0.0))
    is_bubble = ic.get("is_bubble", False)

    # Circle in (x, eta): semicircle on surface when center_y = 0
    distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    phi = np.tanh((distance - radius) / (epsilon * np.sqrt(2)))

    # Semicircle: air below surface (eta < 0). Grid has eta >= 0 only, so the
    # circle naturally yields only the upper half when center_y = 0.
    # Explicit mask for any point below surface (no grid points there, but keeps logic clear).
    phi = np.where(Y < 0.0, 1.0, phi)

    if is_bubble:
        phi = -phi

    # Boundary at eta=0: copy from interior so BCs are consistent
    phi[:, 0] = phi[:, 1]

    return phi


def initialize_phase_two_droplets_touching(Nx, Ny, radius, epsilon=0.03, config=None):
    """Initialize two droplets whose sharp interfaces touch without overlap.

    The phase field is built from the signed distance to the union of two circles.
    By default, both droplets use the same radius and lie on the substrate
    (`droplet_center_y = 0`), so the configuration represents two neighboring
    sessile droplets. The droplets are tangent when `inter_droplet_gap = 0`.

    Config keys under ``initial_conditions``:
      - ``droplet_center_x``: midpoint between droplet centers.
      - ``droplet_center_y``: common vertical center for both droplets.
      - ``droplet_radius``: default radius for both droplets.
      - ``left_droplet_radius`` / ``right_droplet_radius``: optional overrides.
      - ``inter_droplet_gap``: sharp-interface gap; `0` means just touching.
      - ``left_droplet_center_x`` / ``right_droplet_center_x``: optional explicit
        center positions. When provided, they override automatic placement.
      - ``is_bubble``: invert phase sign.

    Args:
        Nx, Ny: grid size.
        radius: default droplet radius.
        epsilon: interface thickness for tanh.
        config: optional configuration dictionary.

    Returns:
        np.ndarray: phase field (phi < 0 liquid, phi > 0 air).
    """
    x = np.linspace(0, 1, Nx)
    eta = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, eta, indexing="ij")

    ic = config.get("initial_conditions", {}) if config else {}
    center_y = float(ic.get("droplet_center_y", 0.0))
    pair_center_x = float(ic.get("droplet_center_x", 0.5))
    left_radius = float(ic.get("left_droplet_radius", radius))
    right_radius = float(ic.get("right_droplet_radius", radius))
    inter_droplet_gap = float(ic.get("inter_droplet_gap", 0.0))
    is_bubble = ic.get("is_bubble", False)

    if inter_droplet_gap < 0.0:
        raise ValueError(
            f"inter_droplet_gap must be non-negative to avoid overlap, got {inter_droplet_gap}"
        )

    left_center_x = ic.get("left_droplet_center_x")
    right_center_x = ic.get("right_droplet_center_x")
    if left_center_x is None or right_center_x is None:
        center_distance = left_radius + right_radius + inter_droplet_gap
        left_center_x = pair_center_x - 0.5 * center_distance
        right_center_x = pair_center_x + 0.5 * center_distance
    else:
        left_center_x = float(left_center_x)
        right_center_x = float(right_center_x)
        center_distance = abs(right_center_x - left_center_x)
        min_distance = left_radius + right_radius
        if center_distance + 1e-14 < min_distance:
            raise ValueError(
                "Two droplets overlap: "
                f"center_distance={center_distance}, required >= {min_distance}"
            )

    left_distance = np.sqrt((X - left_center_x) ** 2 + (Y - center_y) ** 2) - left_radius
    right_distance = np.sqrt((X - right_center_x) ** 2 + (Y - center_y) ** 2) - right_radius
    signed_distance = np.minimum(left_distance, right_distance)
    phi = np.tanh(signed_distance / (epsilon * np.sqrt(2)))

    phi = np.where(Y < 0.0, 1.0, phi)

    if is_bubble:
        phi = -phi

    phi[:, 0] = phi[:, 1]
    return phi


def initialize_phase_rectangle(Nx, Ny, epsilon=0.03, config=None):
    """Initialize the phase field with a rectangular droplet on the surface.

    Liquid (phi < 0) in [x_min, x_max] x [0, y_max]. Uses signed-distance and tanh
    for smooth edges so the droplet can relax into a circular cap under surface tension.

    Args:
        Nx, Ny: grid size.
        epsilon: interface thickness for tanh.
        config: optional dict; initial_conditions.rectangle_x_min, rectangle_x_max, rectangle_y_max.

    Returns:
        np.ndarray: phase field (phi < 0 liquid, phi > 0 air).
    """
    x = np.linspace(0, 1, Nx)
    eta = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, eta, indexing="ij")

    ic = config.get("initial_conditions", {}) if config else {}
    x_min = float(ic.get("rectangle_x_min", 0.35))
    x_max = float(ic.get("rectangle_x_max", 0.65))
    y_max = float(ic.get("rectangle_y_max", 0.2))
    is_bubble = ic.get("is_bubble", False)

    # Signed distance from rectangle [x_min, x_max] x [0, y_max]: negative inside
    signed = np.maximum(np.maximum(np.maximum(x_min - X, X - x_max), -Y), Y - y_max)
    phi = np.tanh(-signed / (epsilon * np.sqrt(2)))

    phi = np.where(Y < 0.0, 1.0, phi)

    if is_bubble:
        phi = -phi

    phi[:, 0] = phi[:, 1]
    return phi


def get_borders_of_droplet(phi):
    """Get the borders of the droplet.
    
    Args:
        phi (np.ndarray): Phase field.
    
    Returns:
        tuple: (start_of_droplet, end_of_droplet) indices in x-direction.
    """
    # Find the first and last non-zero elements in each row
    # Droplet has negative phi values (phi < 0)
    start_of_droplet = 0
    end_of_droplet = phi.shape[0] - 1
    for i in range(0, phi.shape[0]):
        if phi[i, 0] < 0:
            start_of_droplet = i
            break
    for i in range(phi.shape[0] - 1, 0, -1):
        if phi[i, 0] < 0:
            end_of_droplet = i
            break
    return start_of_droplet, end_of_droplet


def get_y_borders_of_droplet(phi):
    """Get the y-borders (top and bottom) of the droplet.
    
    Args:
        phi (np.ndarray): Phase field, shape (Nx, Ny).
    
    Returns:
        tuple: (bottom_of_droplet, top_of_droplet) indices in y-direction.
    """
    # Find the first and last rows (y-indices) where droplet exists
    # Droplet has negative phi values (phi < 0)
    # Check each column (x-direction) and find min/max y where phi < 0
    bottom_of_droplet = phi.shape[1] - 1  # Start from top, find first droplet
    top_of_droplet = 0  # Start from bottom, find last droplet
    
    # Find minimum y (bottom) where droplet exists
    for j in range(phi.shape[1]):
        if np.any(phi[:, j] < 0):
            bottom_of_droplet = j
            break
    
    # Find maximum y (top) where droplet exists
    for j in range(phi.shape[1] - 1, -1, -1):
        if np.any(phi[:, j] < 0):
            top_of_droplet = j
            break
    
    return bottom_of_droplet, top_of_droplet


def initialize_ice_phase(Nx, Ny, config, phi=None):
    """Initialize the ice phase field (ψ).
    
    Args:
        Nx (int): Number of grid points in x-direction.
        Ny (int): Number of grid points in y-direction.
        config (dict): Configuration dictionary.
        phi (np.ndarray, optional): Liquid-gas phase field to identify droplet region.
    
    Returns:
        np.ndarray: Initial ice phase field (ψ = -1 for water, ψ = +1 for ice).
    """
    ice_ic = config.get("initial_conditions", {}).get("ice_phase", {})
    initial_state = ice_ic.get("initial_state", "all_water")
    
    if initial_state == "all_water":
        # All water initially: ψ = -1 everywhere
        psi = np.full((Nx, Ny), -1.0)
        
        # Add ice cells inside the droplet
        add_ice_cells_in_droplet = ice_ic.get("add_ice_cells_in_droplet", False)
        if add_ice_cells_in_droplet and phi is not None:
            num_ice_cells = ice_ic.get("num_ice_cells", 5)
            # Find droplet region (phi < 0 is water)
            droplet_mask = phi < 0
            droplet_indices = np.where(droplet_mask)
            
            if len(droplet_indices[0]) > 0:
                # Randomly select cells inside droplet to be ice
                num_droplet_cells = len(droplet_indices[0])
                num_to_convert = min(num_ice_cells, num_droplet_cells)
                selected_indices = np.random.choice(num_droplet_cells, size=num_to_convert, replace=False)
                
                # Set selected cells to ice (psi = +1)
                for idx in selected_indices:
                    i, j = droplet_indices[0][idx], droplet_indices[1][idx]
                    psi[i, j] = 1.0
    
    elif initial_state == "all_ice":
        # All ice initially: ψ = +1 everywhere
        psi = np.full((Nx, Ny), 1.0)
    
    elif initial_state == "partial_ice":
        # Partial ice: specified fraction
        ice_fraction = ice_ic.get("ice_fraction", 0.0)
        # Create a smooth transition region
        psi = np.full((Nx, Ny), -1.0 + 2.0 * ice_fraction)
    
    elif initial_state == "custom":
        # Custom region defined in config
        region = ice_ic.get("ice_region", {})
        region_type = region.get("type", "circle")
        
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        if region_type == "circle":
            center_x = region.get("center_x", 0.5)
            center_y = region.get("center_y", 0.5)
            radius = region.get("radius", 0.1)
            distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            # Ice inside circle, water outside
            psi = np.tanh((radius - distance) * 10)  # Smooth transition
        
        elif region_type == "rectangle":
            x_min = region.get("x_min", 0.4)
            x_max = region.get("x_max", 0.6)
            y_min = region.get("y_min", 0.0)
            y_max = region.get("y_max", 0.2)
            # Ice inside rectangle, water outside
            mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
            psi = np.where(mask, 1.0, -1.0)
        
        else:
            # Default: all water
            psi = np.full((Nx, Ny), -1.0)
    else:
        # Default: all water
        psi = np.full((Nx, Ny), -1.0)
    
    return psi


def initialize_temperature(Nx, Ny, config):
    """Initialize the temperature field.
    
    Args:
        Nx (int): Number of grid points in x-direction.
        Ny (int): Number of grid points in y-direction.
        config (dict): Configuration dictionary.
    
    Returns:
        np.ndarray: Initial temperature field (in Kelvin).
    """
    temp_ic = config.get("initial_conditions", {}).get("temperature", {})
    initial_distribution = temp_ic.get("initial_distribution", "uniform")
    
    ice_params = config.get("ice_water_params", {})
    T_melt = ice_params.get("T_melt", 273.15)
    T_ambient = ice_params.get("T_ambient", 293.15)
    
    if initial_distribution == "uniform":
        T_initial = temp_ic.get("T_initial", T_ambient)
        T = np.full((Nx, Ny), T_initial)
    
    elif initial_distribution == "linear":
        # Linear gradient from bottom to top
        T_gradient = temp_ic.get("T_gradient", {})
        T_bottom = T_gradient.get("T_bottom", T_melt - 10.0)
        T_top = T_gradient.get("T_top", T_ambient)
        
        y = np.linspace(0, 1, Ny)
        Y = np.meshgrid(np.linspace(0, 1, Nx), y, indexing='ij')[1]
        T = T_bottom + (T_top - T_bottom) * Y
    
    elif initial_distribution == "gaussian":
        # Gaussian cold spot
        cold_spot = temp_ic.get("T_cold_spot", {})
        center_x = cold_spot.get("center_x", 0.5)
        center_y = cold_spot.get("center_y", 0.0)
        sigma = cold_spot.get("sigma", 0.1)
        T_min = cold_spot.get("T_min", T_melt - 20.0)
        
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        distance_sq = (X - center_x)**2 + (Y - center_y)**2
        T = T_ambient + (T_min - T_ambient) * np.exp(-distance_sq / (2 * sigma**2))
    
    else:
        # Default: uniform at ambient
        T = np.full((Nx, Ny), T_ambient)
    
    return T
