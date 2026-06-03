# Configuration Parameter Analysis

## Problem: Mix of Physical and Non-Dimensional Parameters

### Current State

Configs contain **both** physical and non-dimensional parameters:

#### Non-Dimensional Parameters (Used in Code):
- `Re1`, `Re2`: Reynolds numbers
- `We1`, `We2`: Weber numbers  
- `Pe`: Peclet number
- `Fr`: Froude number
- `epsilon`: Interface thickness (non-dimensional)
- `contact_angle`: Contact angle (degrees, but used as-is)

#### Physical Parameters (Present in Configs but Usage Unclear):
- `rho1`, `rho2`: Densities (kg/m³)
- `mu1`, `mu2`: Dynamic viscosities (Pa·s)
- `sigma`: Surface tension (N/m)
- `g`: Gravitational acceleration (m/s²)
- `U0`: Characteristic velocity (m/s)
- `M`: Mobility (m³·s/kg)

### Issues Identified

#### 1. **Redundancy and Potential Inconsistency**

**Problem**: We have both:
- Non-dimensional: `Re1`, `Re2`, `We1`, `We2`
- Physical: `rho1`, `rho2`, `mu1`, `mu2`, `sigma`

**Issue**: 
- If code uses non-dimensional parameters, physical ones are **redundant**
- If physical parameters are used, they might be **inconsistent** with non-dimensional ones
- Example: `Re = ρUL/μ` - if we specify both Re and (ρ, μ, U, L), they must be consistent

**Current Usage**:
- Code uses `Re1`, `Re2` directly (non-dimensional)
- Code uses `rho1`, `rho2` directly (physical)
- Code uses `g` directly (physical)
- `mu1`, `mu2`, `sigma`, `U0`, `M` appear in configs but **may not be used**

#### 2. **Dimensional Inconsistency**

**Problem**: Mixing physical units with non-dimensional numbers

**Example from `config_droplet_realistic_auto.json`**:
```json
{
  "rho1": 1.225,           // Physical: kg/m³
  "rho2": 998.2,           // Physical: kg/m³
  "Re1": 18.18,            // Non-dimensional
  "Re2": 269.03,           // Non-dimensional
  "We1": 0.0012,           // Non-dimensional
  "We2": 0.02,             // Non-dimensional
  "g": -9.81,              // Physical: m/s²
  "mu1": 1.82e-05,         // Physical: Pa·s (NOT USED?)
  "mu2": 0.001002,         // Physical: Pa·s (NOT USED?)
  "sigma": 0.0728,         // Physical: N/m (NOT USED?)
  "U0": 0.270,             // Physical: m/s (NOT USED?)
  "M": 1e-06               // Physical: m³·s/kg (NOT USED?)
}
```

**Issue**: 
- `Re = ρUL/μ` - if we have Re, we don't need individual ρ, μ, U, L
- `We = ρU²L/σ` - if we have We, we don't need individual ρ, σ, U, L
- Having both creates confusion about which is authoritative

#### 3. **Unused Parameters**

**Problem**: Some physical parameters may be stored but not used

**Suspected Unused**:
- `mu1`, `mu2`: Viscosities - code uses `Re1`, `Re2` instead
- `sigma`: Surface tension - code uses `We1`, `We2` instead
- `U0`: Characteristic velocity - may be for reference only
- `M`: Mobility - code uses `Pe` instead

**Need to verify**: Check if these are actually used anywhere in the codebase

#### 4. **Inconsistent Scaling**

**Problem**: Different configs use different conventions

**Examples**:
- `config_template.json`: Only non-dimensional params (Re, We, Pe, Fr)
- `config_droplet_realistic_auto.json`: Mix of both
- `config_droplet_simple.json`: Mix of both with simplified values

**Issue**: 
- Hard to know which parameters are actually used
- Hard to convert between physical and non-dimensional
- Risk of inconsistency when editing configs

### Recommendations

#### Option 1: Pure Non-Dimensional (Recommended)
**Keep only non-dimensional parameters**:
- `Re1`, `Re2`, `We1`, `We2`, `Pe`, `Fr`, `epsilon`, `contact_angle`
- Remove: `mu1`, `mu2`, `sigma`, `U0`, `M`
- Keep: `rho1`, `rho2`, `g` (if used for density interpolation and gravity)

**Pros**: 
- Clean, consistent
- Standard practice for CFD
- No redundancy

**Cons**: 
- Less intuitive for users who think in physical units
- Need to calculate non-dimensional numbers from physical properties

#### Option 2: Physical Parameters Only
**Use only physical parameters**, compute non-dimensional numbers internally:
- `rho1`, `rho2`, `mu1`, `mu2`, `sigma`, `g`, `U0`, `M`
- Compute: `Re = ρUL/μ`, `We = ρU²L/σ`, `Pe = UL/D`, `Fr = U/√(gL)`

**Pros**: 
- More intuitive
- Direct physical meaning

**Cons**: 
- Need to define characteristic scales (L, U)
- More complex code
- Less flexible (can't easily tune non-dimensional numbers)

#### Option 3: Hybrid with Clear Separation
**Keep both but clearly separate**:
- **Primary**: Non-dimensional parameters (used by code)
- **Reference**: Physical parameters (for documentation/reference only)
- Add comments/documentation explaining which are used

**Pros**: 
- Best of both worlds
- Can verify consistency
- Useful for documentation

**Cons**: 
- Still have redundancy
- Need to maintain consistency

### Specific Issues to Fix

1. **Verify parameter usage**:
   - Check if `mu1`, `mu2`, `sigma`, `U0`, `M` are actually used
   - If not used, remove them or mark as "reference only"

2. **Consistency check**:
   - If both physical and non-dimensional are present, verify they're consistent
   - Example: `Re1 = ρ1 * U0 * L_char / μ1` should hold

3. **Documentation**:
   - Clearly document which parameters are used
   - Explain the relationship between physical and non-dimensional
   - Provide conversion formulas

4. **Config validation**:
   - Add validation to ensure consistency
   - Warn if redundant parameters are inconsistent

### Current Parameter Usage (From Code Analysis)

**Definitely Used**:
- `rho1`, `rho2`: Used in density interpolation
- `Re1`, `Re2`: Used in Reynolds number calculation
- `We1`, `We2`: Used in Weber number calculation
- `Pe`: Used in phase field evolution
- `Fr`: Used in Froude number calculation
- `g`: Used in gravity term
- `epsilon`: Used in interface thickness
- `contact_angle`: Used in contact angle BC

**Unclear/Unused**:
- `mu1`, `mu2`: May be for reference only
- `sigma`: May be for reference only
- `U0`: May be for reference only
- `M`: May be for reference only

### Next Steps

1. **Audit codebase**: Check if `mu1`, `mu2`, `sigma`, `U0`, `M` are actually used
2. **Decide on approach**: Choose Option 1, 2, or 3
3. **Clean up configs**: Remove unused parameters or clearly mark them
4. **Add validation**: Ensure consistency between physical and non-dimensional params
5. **Document**: Explain parameter conventions clearly
