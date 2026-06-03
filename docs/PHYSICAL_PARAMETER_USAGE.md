# Physical Parameter Usage Analysis

## Summary

| Parameter | Used? | Where Used | Can Remove? | Notes |
|-----------|-------|------------|-------------|-------|
| `rho1` | ✅ YES | Density interpolation | ❌ NO | Used in `jax_calculate_density()` |
| `rho2` | ✅ YES | Density interpolation | ❌ NO | Used in `jax_calculate_density()` |
| `g` | ✅ YES | Gravity term, hydrostatic pressure | ❌ NO | Used with Fr² scaling |
| `mu1` | ❌ NO | - | ✅ YES | Not used anywhere |
| `mu2` | ❌ NO | - | ✅ YES | Not used anywhere |
| `sigma` | ❌ NO | - | ✅ YES | Not used (different `sigma` in temp IC) |
| `U0` | ❌ NO | - | ✅ YES | Not used anywhere |
| `M` | ❌ NO | - | ✅ YES | Not used (only in comment) |

## Detailed Analysis

### 1. `rho1`, `rho2` - **MUST KEEP** (but could be non-dimensionalized)

**Usage**:
- `src/physics/properties.py`: `jax_calculate_density(phi, rho1, rho2)`
- `src/physics/fluid_dynamics.py`: Density interpolation for momentum equation
- `src/physics/pressure.py`: Density for hydrostatic pressure
- `src/visualization/plotting.py`: Density for visualization

**Current**: Physical densities (kg/m³)
- `rho1 = 1.225` (air)
- `rho2 = 998.2` (water)

**Non-dimensionalization option**:
- Could use density ratio: `rho_ratio = rho1/rho2`
- Then: `rho = rho2 * (1 + (rho_ratio - 1) * phi_mapped)`
- But would need to refactor code

**Recommendation**: Keep for now (used extensively), but could be refactored later

### 2. `g` - **MUST KEEP** (but already partially non-dimensionalized)

**Usage**:
- `src/physics/fluid_dynamics.py`: Gravity term `(1/Fr²) * g`
- `src/physics/pressure.py`: Hydrostatic pressure `p = p_prev - rho * g * dy / Fr²`

**Current**: Physical acceleration (m/s²)
- `g = -9.81` (downward)

**Non-dimensionalization**:
- Already scaled by `Fr²` (classical Froude number convention) in the code
- Formula: `Fr = U/√(gL)` and gravity contribution is `g/Fr²`
- Could compute `g` from `Fr` if we had `U` and `L`, but we don't store those

**Recommendation**: Keep (used with Fr scaling)

### 3. `mu1`, `mu2` - **CAN REMOVE**

**Usage**: ❌ Not used anywhere in code

**Current**: Physical viscosities (Pa·s)
- `mu1 = 1.82e-05` (air)
- `mu2 = 0.001002` (water)

**Why not needed**: Code uses `Re1`, `Re2` (Reynolds numbers) instead
- `Re = ρUL/μ`, so `μ = ρUL/Re`
- Viscosity is embedded in Reynolds number

**Recommendation**: ✅ **REMOVE** - Redundant with Re1, Re2

### 4. `sigma` - **CAN REMOVE**

**Usage**: ❌ Not used for surface tension

**Current**: Physical surface tension (N/m)
- `sigma = 0.0728` (water-air interface)

**Note**: There's a different `sigma` in `initial_conditions.py` for temperature cold spot (Gaussian width), but that's a different parameter.

**Why not needed**: Code uses `We1`, `We2` (Weber numbers) instead
- `We = ρU²L/σ`, so `σ = ρU²L/We`
- Surface tension is embedded in Weber number

**Recommendation**: ✅ **REMOVE** - Redundant with We1, We2

### 5. `U0` - **CAN REMOVE**

**Usage**: ❌ Not used anywhere in code

**Current**: Characteristic velocity (m/s)
- `U0 = 0.270` (example value)

**Why not needed**: 
- Only written by `calculate_realistic_params.py` for reference
- Not read by simulation code
- Velocity is non-dimensionalized in the simulation

**Recommendation**: ✅ **REMOVE** - Reference only, not used

### 6. `M` - **CAN REMOVE**

**Usage**: ❌ Not used anywhere in code

**Current**: Mobility (m³·s/kg)
- `M = 1e-06` (example value)

**Why not needed**: Code uses `Pe` (Peclet number) instead
- `Pe = UL/D`, where `D = M * sigma * epsilon` (effective diffusion)
- Mobility is embedded in Peclet number

**Recommendation**: ✅ **REMOVE** - Redundant with Pe

## Non-Dimensionalization Status

### Already Non-Dimensional:
- ✅ `Re1`, `Re2`: Reynolds numbers
- ✅ `We1`, `We2`: Weber numbers
- ✅ `Pe`: Peclet number
- ✅ `Fr`: Froude number
- ✅ `epsilon`: Interface thickness (non-dimensional)
- ✅ `contact_angle`: Degrees (but used as-is, not dimensional)

### Physical (but needed):
- ⚠️ `rho1`, `rho2`: Physical densities - used directly in code
- ⚠️ `g`: Physical acceleration - used with Fr scaling

### Physical (unused - can remove):
- ❌ `mu1`, `mu2`: Not used
- ❌ `sigma`: Not used
- ❌ `U0`: Not used
- ❌ `M`: Not used

## Action Plan

1. ✅ **Remove unused parameters**: `mu1`, `mu2`, `sigma`, `U0`, `M`
2. ⚠️ **Keep `rho1`, `rho2`**: Used extensively, would require refactoring to remove
3. ⚠️ **Keep `g`**: Used with Fr, already partially non-dimensionalized
4. 📝 **Document**: Explain that `rho1`, `rho2`, `g` are physical but necessary

## Future Refactoring (Optional)

To make everything fully non-dimensional:
1. Replace `rho1`, `rho2` with density ratio: `rho_ratio = rho1/rho2`
2. Compute `g` from `Fr` if we store characteristic scales
3. This would require significant refactoring

For now, keeping `rho1`, `rho2`, `g` is acceptable as they're necessary for the current implementation.
