# Config Cleanup Summary

## Removed Parameters

The following **unused** physical parameters have been removed from all config files:

1. ✅ `mu1` - Dynamic viscosity of phase 1 (air) - **REMOVED**
   - Not used by code (uses `Re1` instead)
   
2. ✅ `mu2` - Dynamic viscosity of phase 2 (liquid) - **REMOVED**
   - Not used by code (uses `Re2` instead)

3. ✅ `sigma` - Surface tension (N/m) - **REMOVED**
   - Not used by code (uses `We1`, `We2` instead)
   - Note: Different `sigma` in temperature IC (Gaussian width) is unrelated

4. ✅ `U0` - Characteristic velocity (m/s) - **REMOVED**
   - Not used by code (only written by parameter generation script)

5. ✅ `M` - Mobility (m³·s/kg) - **REMOVED**
   - Not used by code (uses `Pe` instead)

## Remaining Parameters

### Non-Dimensional (Primary):
- `Re1`, `Re2`: Reynolds numbers
- `We1`, `We2`: Weber numbers
- `Pe`: Peclet number
- `Fr`: Froude number
- `epsilon`: Interface thickness (non-dimensional)
- `contact_angle`: Contact angle (degrees)
- `lambda_willmore`, `epsilon_willmore`: Willmore regularization

### Physical (Necessary):
- `rho1`, `rho2`: Phase densities (kg/m³) - **USED** in density interpolation
- `g`: Gravitational acceleration (m/s²) - **USED** in gravity term and hydrostatic pressure
- `atm_pressure`: Atmospheric pressure - **USED** in pressure BC

## Files Updated

All config files in `configs/` have been cleaned:
- ✅ `config_droplet_realistic_auto.json`
- ✅ `config_droplet_realistic.json`
- ✅ `config_droplet_simple.json`
- ✅ `config_droplet_ice.json`
- ✅ `config_rising_bubble.json`
- ✅ `config_falling_droplet.json`
- ✅ `config_template.json` (already clean)
- ✅ `config_droplet_geometry.json` (already clean)
- ✅ `config_ice_template.json` (already clean)
- ✅ `config_template_restart.json` (already clean)

## Script Updated

- ✅ `scripts/calculate_realistic_params.py` - No longer writes unused parameters

## Result

Configs now contain **only parameters that are actually used** by the simulation code:
- Non-dimensional numbers (Re, We, Pe, Fr) for fluid dynamics
- Physical densities (rho1, rho2) for density interpolation
- Physical gravity (g) for gravity term
- Interface and regularization parameters

This eliminates:
- Redundancy (no duplicate information)
- Confusion (clear which parameters matter)
- Potential inconsistencies (can't have mismatched values)
