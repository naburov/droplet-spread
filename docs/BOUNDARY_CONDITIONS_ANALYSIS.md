# Boundary Conditions Analysis

## Current Implementation

### 1. Velocity Boundary Conditions (No-Slip at Bottom)

**Location**: `src/boundary_conditions/velocity_bc.py`

**Implementation**:
- **No-slip at bottom**: `U[:, 0, :] = 0.0` (both u and v components set to zero)
- Applied in `_no_slip()` method: `U.at[:, 0, :].set(0.0)`
- This enforces: **u(x, 0) = 0, v(x, 0) = 0**

**Connection to Chemical Potential**:
- No direct connection. Velocity BC controls fluid motion.
- No-slip prevents fluid from moving at the boundary, which should prevent spreading.

**Potential Issues**:
1. Velocity BC is applied **after** velocity update, but might be overwritten during PPE
2. Need to verify velocity remains zero at bottom throughout simulation

### 2. Chemical Potential Boundary Conditions (Zero Flux)

**Location**: `src/boundary_conditions/chemical_potential_bc.py`

**Implementation**:
- **Zero flux**: `∂μ/∂n = 0` at all boundaries
- Applied via: `mu_c[:, 0] = mu_c[:, 1]` (bottom), etc.
- This enforces: **∂μ/∂y = 0** at bottom, **∂μ/∂x = 0** at left/right

**Connection to Velocity**:
- **Indirect**: Chemical potential μ controls phase field evolution
- Zero flux BC ensures no mass flux through boundaries: **n·∇μ = 0**
- This is correct for Cahn-Hilliard equation (conserves mass)

**Mathematical Relationship**:
- Phase field evolution: `∂φ/∂t = -u·∇φ + (1/Pe) ∇·(M(φ)∇μ)`
- Zero flux on μ: `n·∇μ = 0` → no diffusive flux through boundaries
- Combined with no-slip velocity: prevents both advective and diffusive mass flux

### 3. Contact Angle Boundary Conditions

**Location**: `src/boundary_conditions/contact_angle_bc.py`

**Implementation**:
- **Robin BC**: `∂φ/∂n = -cos(θ)|∇φ|` at contact points
- Applied at bottom boundary where interface touches surface
- Method: "robin" or "simple"

**Connection to Velocity**:
- **Indirect**: Contact angle controls interface shape
- No-slip velocity at boundary + contact angle BC → determines contact line dynamics
- Contact line can move even with no-slip (interface moves, not fluid at boundary)

**Potential Issues**:
1. Contact angle might not be enforced strongly enough
2. Contact line detection might miss some contact points
3. Interface might be spreading due to weak surface tension (We2 too high)

## Summary of Connections

```
Velocity BC (No-Slip)
    ↓
    Prevents fluid motion at boundary
    ↓
    Should prevent spreading
    BUT: Contact line can still move (interface motion ≠ fluid motion)

Chemical Potential BC (Zero Flux)
    ↓
    Prevents diffusive mass flux through boundaries
    ↓
    Ensures mass conservation
    Works with velocity BC to prevent mass loss

Contact Angle BC
    ↓
    Controls interface shape at boundary
    ↓
    Determines contact line position
    Combined with surface tension (We) → controls spreading
```

## Issues Identified

1. **We2 = 0.1 might still be too high**
   - Lower We = stronger surface tension
   - Changed to We2 = 0.01 in config

2. **Velocity BC enforcement**
   - Need to verify velocity remains zero at bottom
   - Check if PPE or other operations overwrite velocity BC

3. **Contact angle enforcement**
   - Verify contact angle is being applied correctly
   - Check if contact line detection is working

## Recommendations

1. ✅ Lower We2 to 0.01 (done)
2. Add diagnostic to check velocity at bottom boundary
3. Add diagnostic to check contact angle enforcement
4. Verify no-slip is maintained throughout simulation
