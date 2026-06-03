# Geometry Accounting in Computation Loop - Analysis

This document provides a comprehensive analysis of where geometry (hump) is and **is NOT** accounted for in the computation loop.

## Computation Loop Overview

The main computation loop in `main.py` follows this sequence:

1. **Velocity Update (Predictor Step)**
2. **Velocity Boundary Conditions**
3. **Continuity Check**
4. **PPE (Pressure Projection Equation)**
5. **Phase Field Update**
6. **Ice Phase Field Update** (if enabled)
7. **Temperature Update** (if enabled)
8. **Surface Tension Calculation**
9. **Pressure Update**

---

## ✅ Where Geometry IS Accounted For

### 1. **Velocity Boundary Conditions** ✅

**Location**: `src/boundary_conditions/velocity_bc.py`

**Status**: ✅ **FULLY ACCOUNTED**

```python
# Line 356-357 in main.py
if velocity_bc_manager.use_geometry and h_bottom is not None:
    velocity_bc_manager.geometry_mask.set_height_function(h_bottom)

# Line 360-362
U = velocity_bc_manager.apply_boundary_conditions(U, dx, dy, use_jax=True, psi=psi)
```

**What it does**:
- Computes solid mask from `h_bottom`: `solid_mask[i, j] = True` where `y[j] < h_bottom[i]`
- Applies no-slip condition: `U = 0` in all solid cells
- Applied **before** and **after** other boundary conditions

**Implementation**:
- Uses `apply_no_slip_at_surface(U, solid_mask)` from `geometry_mask.py`
- Zeroes velocity in solid regions and at interface cells

---

### 2. **Pressure Boundary Conditions** ✅

**Location**: `src/boundary_conditions/pressure_bc.py`

**Status**: ✅ **PARTIALLY ACCOUNTED** (simplified approach)

```python
# Line 454-456 in main.py
if pressure_bc_manager.use_geometry and h_bottom is not None:
    pressure_bc_manager.geometry_mask.set_height_function(h_bottom)
P = pressure_bc_manager.apply_boundary_conditions(P, dy=dy)
```

**What it does**:
- For Neumann BC: applies `∂p/∂n = 0` at solid surface
- Uses simple extrapolation: `p_bottom = p_above` for solid cells
- **Limitation**: Uses simplified approach, not full geometry-aware Neumann BC

**Implementation**:
```python
# Line 94 in pressure_bc.py
p_bottom = jnp.where(solid_mask[:, 0], p_above, p_bottom)
```

---

### 3. **Phase Field Contact Angle** ✅

**Location**: `src/boundary_conditions/contact_angle_bc.py`

**Status**: ✅ **FULLY ACCOUNTED**

```python
# Line 420-422 in main.py
phi = phase_solver.update(phi, U, current_dt, dx, dy, use_jax=True, 
                         psi=psi if include_ice_water else None,
                         h_bottom=h_bottom)
```

**What it does**:
- Computes surface normal from `h_bottom(x)`: `n = (-dh/dx, 1) / √(1 + (dh/dx)²)`
- Applies contact angle **relative to surface normal**, not vertical
- Ensures `∂φ/∂n = -cos(θ) |∇φ|` at the surface

**Implementation**:
- Uses `_geometry_aware_contact_angle_impl()` function
- Surface normal computed from height function using finite differences
- Contact angle correction accounts for surface slope

---

## ❌ Where Geometry is NOT Accounted For

### 1. **Velocity Update (Predictor Step)** ❌

**Location**: `src/physics/fluid_dynamics.py` → `jax_update_velocity()`

**Status**: ❌ **NOT ACCOUNTED**

```python
# Line 344-348 in main.py
U = fluid_solver.update_velocity(U, P, surface_tension, current_dt, dx, dy, 
                                phi, include_gravity=include_gravity, use_jax=True, psi=psi)
```

**What's missing**:
- Velocity update calculates: `ũ = uⁿ + Δt [−(u·∇)u + (1/Re)∇²u − (1/(Re·Ca)) φ ∇μ + g]`
- **No masking of solid regions** during calculation
- Velocity terms (convective, viscous, pressure gradient) are calculated **even in solid cells**
- Only masked **after** the update (if `psi` is provided for ice)

**Impact**:
- Velocity terms are computed incorrectly in solid regions
- This could lead to numerical artifacts near the hump
- The velocity field may have non-zero values in solid cells during intermediate calculations

**What should be done**:
```python
# Before calculating velocity terms:
if h_bottom is not None:
    solid_mask = compute_solid_mask_from_height(h_bottom, Ny, dy)
    U = jnp.where(solid_mask[..., jnp.newaxis], 0.0, U)

# Calculate velocity terms...

# After update, re-apply mask:
if h_bottom is not None:
    U = jnp.where(solid_mask[..., jnp.newaxis], 0.0, U)
```

---

### 2. **Continuity Check (Divergence Calculation)** ❌

**Location**: `src/physics/fluid_dynamics.py` → `jax_check_continuity()`

**Status**: ❌ **NOT ACCOUNTED**

```python
# Line 365 in main.py
divergence, max_div, mean_div = fluid_solver.check_continuity(U, dx, dy, use_jax=True)
```

**What's missing**:
- Divergence is calculated over **entire domain**: `∇·U = du/dx + dv/dy`
- **No exclusion of solid regions** from divergence calculation
- Solid cells contribute to divergence even though they should be excluded

**Impact**:
- Divergence may be non-zero in solid regions (where velocity should be zero)
- This could trigger unnecessary PPE corrections
- The divergence check doesn't account for the fact that solid regions don't need to satisfy continuity

**What should be done**:
```python
# Mask out solid regions from divergence calculation:
if h_bottom is not None:
    solid_mask = compute_solid_mask_from_height(h_bottom, Ny, dy)
    # Only calculate divergence in fluid regions
    divergence_field = jnp.where(solid_mask, 0.0, jax_divergence(U, dx, dy))
```

---

### 3. **PPE (Pressure Projection Equation)** ❌

**Location**: `src/solvers/ppe_global.py` → `ppe_global()`

**Status**: ❌ **NOT ACCOUNTED**

```python
# Line 404-409 in main.py
U, ppe_info = ppe(U, dx, dy, current_dt, correction_solver, 
                 div_threshold=div_threshold, 
                 max_div_threshold=max_div_threshold, 
                 mean_div_threshold=mean_div_threshold, 
                 ppe_bcs=ppe_bcs,
                 velocity_bc_manager=velocity_bc_manager)
```

**What's missing**:
- PPE solves: `∇²p = (1/Δt) ∇·ũ` over **entire domain**
- **No exclusion of solid regions** from the Poisson equation
- Pressure correction is calculated even in solid cells
- The solver matrix doesn't account for geometry

**Impact**:
- Pressure correction may be non-zero in solid regions
- The PPE solver treats all cells equally, regardless of geometry
- This could lead to incorrect pressure gradients near the hump

**What should be done**:
- Modify the Poisson solver to exclude solid cells from the system
- Set pressure correction to zero in solid regions
- Apply proper boundary conditions at the solid-fluid interface

---

### 4. **Surface Tension Calculation** ❌

**Location**: `src/physics/surface_tension.py` → `jax_surface_tension_force()`

**Status**: ❌ **NOT ACCOUNTED**

```python
# Line 443-444 in main.py
surface_tension = surface_tension_solver.calculate_force(phi, dx, dy, use_jax=True)
surface_tension = surface_tension_solver.apply_boundary_conditions(surface_tension, phi, use_jax=True)
```

**What's missing**:
- Surface tension force calculated from phase field: `F = (3√2 ε / (4We)) κ |∇φ| ∇φ`
- **No geometry awareness** in the force calculation
- Boundary conditions use **flat surface assumption** (contact angle at y=0)
- The `jax_apply_surface_tension_boundary_conditions()` function assumes a flat bottom

**Impact**:
- Surface tension force doesn't account for the hump geometry
- Boundary conditions are applied at y=0, not at the actual surface
- This could lead to incorrect surface tension forces near the hump

**What should be done**:
- Modify surface tension BC to use geometry-aware contact angle
- Apply BC at the actual solid surface, not at y=0
- Account for surface normal in force calculation

**Current implementation**:
```python
# Line 131 in surface_tension.py - assumes flat surface
sf = surface_tension.at[:, 0, 1].set(surface_tension[:, 1, 1] * jnp.cos(theta))
```

---

### 5. **Pressure Update (Poisson Solver)** ❌

**Location**: `src/physics/pressure.py` → `update_pressure_jax()`

**Status**: ❌ **NOT ACCOUNTED**

```python
# Line 449 in main.py
P = pressure_solver.update_pressure(surface_tension, Nx, Ny, dx, dy, phi, pressure_linear_solver, use_jax=True)
```

**What's missing**:
- Pressure is solved from: `∇²p = ∇·(surface_tension / ρ)`
- The Poisson solver (`pressure_linear_solver`) uses a **flat bottom boundary**
- **No geometry awareness** in the solver matrix construction
- Boundary conditions are set at y=0, not at the actual surface

**Impact**:
- Pressure field doesn't account for the hump geometry
- The solver treats the bottom boundary as flat
- This could lead to incorrect pressure gradients near the hump

**What should be done**:
- Modify the Poisson solver to account for non-uniform geometry
- Set boundary conditions at the actual solid surface
- Exclude solid cells from the pressure solve

**Current implementation**:
```python
# Line 122 in pressure.py - assumes flat top boundary
sf_grad = sf_grad.at[:, -1].set(atm_pressure)  # Top: Dirichlet
# No geometry-aware bottom boundary handling
```

---

### 6. **Finite Difference Operations** ❌

**Location**: `src/numerics/finite_differences.py`

**Status**: ❌ **NOT ACCOUNTED**

**What's missing**:
- All finite difference operations (gradient, divergence, Laplacian) use **standard stencils**
- **No special treatment** for cells near the solid surface
- Operations cross solid-fluid boundaries without accounting for geometry

**Impact**:
- Derivatives may be calculated incorrectly near the hump
- Stencils may include solid cells in calculations
- This could lead to numerical errors near the interface

**What should be done**:
- Use one-sided differences at the solid surface
- Exclude solid cells from stencil calculations
- Account for surface normal in derivative calculations

---

## Summary Table

| Component | Geometry Accounted? | Status | Impact |
|-----------|-------------------|--------|--------|
| **Velocity BC** | ✅ Yes | Fully implemented | Correct |
| **Pressure BC** | ⚠️ Partial | Simplified approach | Acceptable |
| **Contact Angle** | ✅ Yes | Fully implemented | Correct |
| **Velocity Update** | ❌ No | Missing | **High** - Terms calculated in solid |
| **Continuity Check** | ❌ No | Missing | **Medium** - Divergence includes solid |
| **PPE** | ❌ No | Missing | **High** - Solver doesn't exclude solid |
| **Surface Tension** | ❌ No | Missing | **Medium** - BC assumes flat surface |
| **Pressure Update** | ❌ No | Missing | **High** - Solver assumes flat boundary |
| **Finite Differences** | ❌ No | Missing | **Low** - May cause numerical errors |

---

## Recommendations

### High Priority Fixes

1. **Velocity Update**: Mask solid regions before and after velocity calculation
2. **PPE**: Exclude solid cells from the Poisson equation solve
3. **Pressure Update**: Modify Poisson solver to account for geometry

### Medium Priority Fixes

4. **Continuity Check**: Exclude solid regions from divergence calculation
5. **Surface Tension**: Use geometry-aware boundary conditions

### Low Priority Fixes

6. **Finite Differences**: Use one-sided differences at solid surface

---

## Implementation Notes

### Current Workaround

The current implementation relies on:
- **Velocity BC** to enforce no-slip after calculations
- **Pressure BC** to apply Neumann conditions after solving
- **Contact Angle BC** to account for geometry in phase field

However, this is a **post-processing approach** - the calculations themselves don't account for geometry.

### Proper Solution

A proper implementation would:
1. **Exclude solid cells** from all calculations
2. **Modify solver matrices** to account for geometry
3. **Use geometry-aware stencils** for finite differences
4. **Apply BCs at actual surface**, not at y=0

This would require significant refactoring of the solvers and finite difference operations.

---

## References

- **Main Loop**: `main.py` lines 336-456
- **Velocity Update**: `src/physics/fluid_dynamics.py`
- **PPE**: `src/solvers/ppe_global.py`
- **Pressure**: `src/physics/pressure.py`
- **Surface Tension**: `src/physics/surface_tension.py`
- **Finite Differences**: `src/numerics/finite_differences.py`


