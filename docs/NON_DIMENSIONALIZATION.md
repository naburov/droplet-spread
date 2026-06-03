# Non-Dimensionalization of Physical Parameters

## Changes Made

All config files have been updated to use **non-dimensional** values for `rho1`, `rho2`, and `g`.

## Non-Dimensionalization Approach

### Density (rho1, rho2)

**Method**: Use `rho2` (liquid density) as reference density
- `rho1* = rho1 / rho2` (non-dimensional air density)
- `rho2* = 1.0` (reference density)

**Example**:
- Physical: `rho1 = 1.225 kg/m³`, `rho2 = 998.2 kg/m³`
- Non-dimensional: `rho1 = 0.001227`, `rho2 = 1.0`

**Verification**:
- Density ratio preserved: `rho1/rho2 = 0.001227` (same in both systems)
- Density interpolation: `rho = (1 - w) * rho2 + w * rho1` works correctly
- To get physical density: multiply by reference density `rho2_phys`

### Gravity (g)

**Method**: Use `g* = -1.0` (non-dimensional, preserves downward direction)
- Physical: `g = -9.81 m/s²` (negative = downward)
- Non-dimensional: `g = -1.0`

**Code Usage**:
- Gravity term: `(1/Fr²) * g` → with `g = -1.0`: `-1/Fr²` (downward) ✓
- Hydrostatic pressure: `p = p_prev - rho * g * dy / Fr²` → with `g = -1.0`: `p = p_prev + rho * dy / Fr²` ✓

**Note**: The sign is preserved (`-1.0`) to maintain downward direction. The code already scales by `Fr`, so `g = -1.0` gives the correct non-dimensional behavior.

## Updated Files

### Config Files (all updated):
- ✅ `config_droplet_realistic_auto.json`
- ✅ `config_template.json`
- ✅ `config_droplet_realistic.json`
- ✅ `config_droplet_simple.json`
- ✅ `config_droplet_ice.json`
- ✅ `config_rising_bubble.json`
- ✅ `config_falling_droplet.json`
- ✅ `config_droplet_geometry.json`
- ✅ `config_ice_template.json`
- ✅ `config_template_restart.json`

### Code Files (updated):
- ✅ `src/config/config_loader.py` - Default values
- ✅ `scripts/calculate_realistic_params.py` - Parameter generation

## Physical Interpretation

### Density
- **Non-dimensional density** represents density relative to liquid density
- `rho1 = 0.001227` means air is 0.12% as dense as water (correct!)
- `rho2 = 1.0` is the reference (water density)

### Gravity
- **Non-dimensional gravity** `g = -1.0` represents unit downward acceleration
- The actual magnitude is controlled by `Fr` (classical Froude number)
- Formula: `gravity_term = (1/Fr²) * g = -1/Fr²` (downward)

## Benefits

1. **Consistency**: All parameters are now non-dimensional
2. **Clarity**: No mixing of physical and non-dimensional units
3. **Flexibility**: Easy to scale to different physical systems
4. **Standard Practice**: Matches CFD conventions

## Conversion Back to Physical

If needed to convert back to physical units:
- **Density**: Multiply by reference density `rho2_phys`
- **Gravity**: The non-dimensional `g = -1.0` with `Fr` scaling gives correct physical behavior

## Verification

The density ratio and gravity direction are preserved:
- Density ratio: `rho1/rho2 = 0.001227` (same in both systems)
- Gravity direction: Downward (negative) preserved
- Physical behavior: Unchanged (scaling handled by non-dimensional numbers)
