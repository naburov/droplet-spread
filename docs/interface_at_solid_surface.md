# Handling Air-Water Interface Near Solid Surface

## Current Implementation

### 1. Interface Detection

The interface is identified using two criteria:
- **Phase field value**: `|phi| < 0.5` (interface region)
- **Gradient magnitude**: `|∇φ| > epsilon/10` (steep gradient at interface)

```python
# In state.py:compute_surface_tension()
interface_threshold = epsilon / 10.0
interface_mask = norm_grad_phi > interface_threshold
phi_near_zero = jnp.abs(phi) < 0.3  # For contact line detection
```

### 2. Surface Tension at Contact Line

**Key Principle**: Surface tension should exist at the contact line (where interface meets solid), but be zero in bulk solid regions.

**Current Approach**:
1. Compute surface tension only at interface (`interface_mask`)
2. Apply contact angle BC at surface cells where `|phi| < 0.5`
3. Re-apply interface mask to zero bulk regions
4. **Special handling**: Allow surface tension in solid cells IF interface exists there (contact line)

```python
# In state.py:compute_surface_tension()
interface_exists = (jnp.abs(phi) < 0.3) | (grad_phi_mag > interface_threshold)
zero_mask = solid_mask & ~interface_exists  # Zero in solid, except at contact line
surface_tension = jnp.where(zero_mask, 0.0, surface_tension)
```

### 3. Phase Field Advection Near Surface

**Advection BC**: Impermeable at bottom
- Prevents phase field from advecting into solid
- Uses velocity threshold: `|v| > 1e-10` to detect flow

```python
# In advection_bc.py:_apply_impermeable_bottom_jax()
v_bottom = U[:, 0, 1]
mask = jnp.abs(v_bottom) > velocity_threshold
phi = jnp.where(mask[:, None], phi.at[:, 0].set(phi[:, 1]), phi)
```

**Issue**: This only handles the computational boundary (y=0), not the actual surface `h_bottom(x)`.

### 4. Contact Angle Boundary Condition

**Geometry-Aware Contact Angle**:
- Computes surface normal from `h_bottom(x)` or ice phase field
- Applies contact angle relative to surface normal
- **Blending**: Uses relaxation parameter (α=0.5) to blend advected value with constraint

```python
# In contact_angle_bc.py
phi_new = alpha * phi_advected + (1 - alpha) * phi_constraint
```

This allows the contact line to move while maintaining the contact angle.

### 5. Phase Field Update Sequence

1. **Advection**: `∂φ/∂t + U·∇φ = 0`
2. **Advection BC**: Impermeable at bottom (prevents flow into solid)
3. **Contact Angle BC**: Enforces angle at surface cells where interface exists
4. **Chemical Potential**: Diffusion term (Cahn-Hilliard)

## Current Issues and Challenges

### Issue 1: Interface Very Close to Solid

**Problem**: When `phi ≈ 0` at cells just above `h_bottom(x)`, the interface is very close to the solid surface. This can cause:
- Numerical instabilities in surface tension computation
- Incorrect contact angle enforcement
- Phase field "leaking" into solid

**Current Mitigation**:
- Interface mask based on `|∇φ|` helps identify true interface
- Contact angle BC only applied where `|phi| < 0.5`
- Phase field forced to `phi = 1.0` (air) in solid regions during initialization

### Issue 2: Contact Line Movement

**Problem**: The contact line needs to move along the solid surface as the droplet spreads, but the contact angle must be maintained.

**Current Solution**:
- Blending in contact angle BC (α=0.5) allows movement
- Advection BC is impermeable, preventing phase field from going into solid
- Contact angle computed at actual surface cells (`j_surf`)

**Potential Issue**: If blending is too strong, contact angle may not be maintained. If too weak, contact line may be pinned.

### Issue 3: Surface Tension Direction at Contact Line

**Problem**: Surface tension force direction must respect the contact angle relative to the surface normal.

**Current Solution**:
- Contact angle BC applied to surface tension at surface cells
- Direction computed from surface normal and contact angle
- Applied only where interface exists (`|phi| < 0.5`)

### Issue 4: Advection BC for Non-Flat Surfaces

**Problem**: ~~Current impermeable BC only handles `y=0`, not the actual surface `h_bottom(x)`.~~ ✅ **FIXED**

**Solution Implemented**:
- Geometry-aware advection BC now handles `h_bottom(x)` explicitly
- Prevents phase field from advecting into solid at actual surface
- Forces `phi = 1.0` (air) in solid regions as safety measure

## Recommendations for Improvement

### 1. Geometry-Aware Advection BC ✅ **IMPLEMENTED**

**Implementation**: Extended impermeable BC to handle `h_bottom(x)`:

- **Location**: `src/boundary_conditions/advection_bc.py`
- **Method**: `_apply_impermeable_bottom_geometry_aware_jax()`
- **Features**:
  1. Finds surface cells for each x (first cell where `y[j] >= h_bottom[i]`)
  2. Prevents advection when velocity points into solid (`v < 0` at surface)
  3. Forces `phi = 1.0` (air) in solid regions as safety measure
  4. Automatically used when `h_bottom` is provided to `apply_boundary_conditions()`

**Usage**:
```python
bc_manager = AdvectionBoundaryConditions(config)
phi_new = bc_manager.apply_boundary_conditions(
    phi, U, dt, dx, dy, h_bottom=h_bottom  # Pass h_bottom for geometry-aware BC
)
```

### 2. Improved Interface Detection Near Surface

**Proposal**: Use distance from surface in addition to phase field value:

```python
def detect_interface_near_surface(phi, h_bottom, dx, dy, epsilon):
    """Detect interface with special handling near solid surface."""
    # Standard interface detection
    grad_phi = jax_gradient(phi, dx, dy)
    norm_grad_phi = jax_norm(grad_phi)
    interface_mask = norm_grad_phi > epsilon / 10.0
    
    # Near surface: also check distance from h_bottom(x)
    Nx, Ny = phi.shape
    y_coords = jnp.linspace(0, (Ny-1)*dy, Ny)
    
    for i in range(Nx):
        h_surf = h_bottom[i]
        for j in range(Ny):
            y_j = y_coords[j]
            distance_from_surface = y_j - h_surf
            
            # If very close to surface (within 2*epsilon) and |phi| < 0.5, it's interface
            if distance_from_surface < 2*epsilon and jnp.abs(phi[i, j]) < 0.5:
                interface_mask = interface_mask.at[i, j].set(True)
    
    return interface_mask
```

### 3. Smoother Contact Line Treatment

**Proposal**: Use a smooth transition zone near the contact line:

```python
def smooth_contact_line_transition(phi, h_bottom, dx, dy, transition_width=2*epsilon):
    """Apply smooth transition for phase field near contact line."""
    # Create transition mask: cells within transition_width of surface
    # In this zone, blend phase field smoothly
    # This prevents sharp jumps that can cause numerical issues
    pass
```

### 4. Enhanced Surface Tension at Contact Line

**Proposal**: Ensure surface tension direction is correct at contact line:

```python
def enforce_contact_angle_in_surface_tension(surface_tension, phi, h_bottom, contact_angle, dx, dy):
    """Enforce contact angle in surface tension force direction at contact line."""
    # At contact line cells (interface at surface):
    # 1. Compute surface normal from h_bottom
    # 2. Compute desired surface tension direction from contact angle
    # 3. Project surface tension onto this direction
    # 4. Ensure magnitude is correct
    pass
```

### 5. Phase Field Constraint Near Surface

**Proposal**: Explicitly prevent phase field from going into solid:

```python
def enforce_phase_field_at_surface(phi, h_bottom, dx, dy):
    """Ensure phase field respects solid boundary."""
    Nx, Ny = phi.shape
    y_coords = jnp.linspace(0, (Ny-1)*dy, Ny)
    j_surf = find_surface_cells(h_bottom, Ny, dy)
    
    for i in range(Nx):
        j_surf_i = j_surf[i]
        h_surf = h_bottom[i]
        
        # Force phi = 1.0 (air) in solid cells (y < h_surf)
        for j in range(Ny):
            y_j = y_coords[j]
            if y_j < h_surf:
                phi = phi.at[i, j].set(1.0)  # Air in solid
        
        # Ensure smooth transition at surface cell
        if j_surf_i < Ny:
            # Surface cell should have phi consistent with contact angle
            # This is handled by contact angle BC, but we can add safety check
            pass
    
    return phi
```

## Summary

**Current Strengths**:
- ✅ Interface detection using gradient magnitude
- ✅ Surface tension only at interface
- ✅ Contact angle BC with blending (allows movement)
- ✅ Special handling for contact line in surface tension

**Current Weaknesses**:
- ✅ ~~Advection BC doesn't account for geometry~~ **FIXED**
- ⚠️ No explicit distance-based interface detection near surface
- ⚠️ Phase field might not be perfectly constrained at surface (partially addressed)
- ⚠️ No smooth transition zone near contact line

**Recommended Next Steps**:
1. ✅ ~~Implement geometry-aware advection BC~~ **DONE**
2. Add distance-based interface detection near surface
3. Add explicit phase field constraint at surface (partially done - solid cells forced to air)
4. Test with interface very close to solid surface

