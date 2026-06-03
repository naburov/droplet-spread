"""Shared layout and data for joint plots (matplotlib and PyVista)."""

import numpy as np
from physics.properties import calculate_density

# Subplot kinds — same for both backends
KIND_PHASE, KIND_SURFACE_TENSION, KIND_VELOCITY_STREAMLINES = "phase", "surface_tension", "velocity_streamlines"
KIND_VELOCITY_MAGNITUDE, KIND_PRESSURE, KIND_DENSITY = "velocity_magnitude", "pressure", "density"
KIND_ICE_PHASE, KIND_TEMPERATURE, KIND_COMBINED_PHASE, KIND_TEMPERATURE_INTERFACES = "ice_phase", "temperature", "combined_phase", "temperature_interfaces"

_BASE_SPECS = [
    (0, 0, KIND_PHASE, "Phase Field (Liquid-Gas)", "Phase", "viridis", -1, 1),
    (0, 1, KIND_SURFACE_TENSION, "Surface Tension Force", "Force Magnitude", "viridis", 0, None),
    (0, 2, KIND_VELOCITY_STREAMLINES, "Velocity (streamlines)", "Speed", "viridis", None, None),
    (1, 0, KIND_VELOCITY_MAGNITUDE, "Velocity Magnitude", "Speed", "viridis", None, None),
    (1, 1, KIND_PRESSURE, "Pressure Field", "Pressure", "coolwarm", None, None),
    (1, 2, KIND_DENSITY, "Density", "Density", "viridis", None, None),
]
_ICE_SPECS = [
    (2, 0, KIND_ICE_PHASE, "Ice Phase (ψ)", "Ice Phase (ψ)", "coolwarm", -1, 1),
    (2, 1, KIND_TEMPERATURE, "Temperature Field", "Temperature (K)", "hot", None, None),
    (2, 2, KIND_COMBINED_PHASE, "Combined Phase Fields", "Phase", "RdYlBu", -1, 1),
    (2, 3, KIND_TEMPERATURE_INTERFACES, "Temperature with Interfaces", "Temperature (K)", "hot", None, None),
]


def get_joint_plot_layout(include_ice_water):
    """Return (nrows, ncols, specs). Each spec is dict with row, col, kind, title, cbar_label, cmap, vmin, vmax."""
    specs = [{"row": r, "col": c, "kind": k, "title": t, "cbar_label": cl, "cmap": cm, "vmin": vmin, "vmax": vmax}
            for r, c, k, t, cl, cm, vmin, vmax in _BASE_SPECS]
    if include_ice_water:
        specs += [{"row": r, "col": c, "kind": k, "title": t, "cbar_label": cl, "cmap": cm, "vmin": vmin, "vmax": vmax}
                  for r, c, k, t, cl, cm, vmin, vmax in _ICE_SPECS]
        return 3, 4, specs
    return 2, 3, specs


def prepare_joint_plot_data(phi, U, P, surface_tension, rho1, rho2, Lx, Ly, psi=None, T=None, include_ice_water=False):
    """Build grid and derived arrays for joint plots. Returns dict with extent, x, eta, X, Eta, phi, U, rho, etc."""
    phi = np.asarray(phi)
    U = np.asarray(U)
    P = np.asarray(P)
    surface_tension = np.asarray(surface_tension)
    Nx, Ny = phi.shape[0], phi.shape[1]
    x = np.linspace(0, Lx, Nx)
    eta = np.linspace(0, Ly, Ny)
    X, Eta = np.meshgrid(x, eta)

    rho = calculate_density(phi, rho1, rho2)
    U_mag = np.sqrt(U[..., 0] ** 2 + U[..., 1] ** 2)
    ST_mag = np.sqrt(surface_tension[..., 0] ** 2 + surface_tension[..., 1] ** 2)
    st_vmax = float(np.max(ST_mag[ST_mag > 1e-10])) if np.any(ST_mag > 1e-10) else 1.0

    if include_ice_water and psi is not None:
        psi = np.asarray(psi)
        ice = psi > 0.0
        U_mag_masked = np.where(ice, 0.0, U_mag)
        U_masked = U.copy()
        U_masked[ice] = 0.0
    else:
        U_mag_masked, U_masked = U_mag, U

    out = {
        "Nx": Nx, "Ny": Ny, "Lx": Lx, "Ly": Ly, "extent": [0.0, float(Lx), 0.0, float(Ly)],
        "x": x, "eta": eta, "X": X, "Eta": Eta,
        "phi": phi, "U": U, "P": P, "surface_tension": surface_tension, "rho": rho,
        "U_magnitude": U_mag, "U_magnitude_masked": U_mag_masked, "U_masked": U_masked,
        "ST_magnitude": ST_mag, "st_vmax": st_vmax, "P_vis": np.asarray(P),
        "include_ice_water": include_ice_water, "psi": psi, "T": T, "rho1": rho1, "rho2": rho2,
        "T_min": 263.15, "T_max": 293.15,
    }
    if include_ice_water and T is not None and T.size > 0:
        tmin, tmax = float(np.nanmin(T)), float(np.nanmax(T))
        if tmax - tmin < 10.0:
            tc = (tmin + tmax) / 2.0
            tmin, tmax = tc - 5.0, tc + 5.0
        out["T_min"], out["T_max"] = tmin, tmax
    if include_ice_water and psi is not None:
        c = np.zeros_like(phi)
        c[phi < 0], c[phi > 0] = -1, 1
        c[(phi < 0) & (psi > 0)] = 0
        out["combined"] = c
    return out
