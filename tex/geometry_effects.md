# Effects of Non-Uniform Geometry on Simulation Fields

This document describes how the non-uniform bottom geometry (hump) affects the three main fields in the droplet spreading simulation.

## 1. Velocity Field

### Physical Effects

The hump geometry creates a **non-uniform solid boundary** that affects fluid flow in several ways:

1. **No-Slip Condition at Solid Surface**: 
   - Velocity is **zero** at all points where the fluid contacts the solid hump
   - This is enforced via `apply_no_slip_at_surface()` which sets `U = 0` in solid regions
   - The solid mask is computed from the height function: `solid_mask = (y < h(x))`

2. **Flow Obstruction**:
   - The hump acts as an obstacle, forcing fluid to flow around it
   - Flow is redirected upward and around the sides of the hump
   - Creates recirculation zones and flow separation on the leeward side

3. **Velocity Gradients**:
   - Strong velocity gradients develop near the hump surface due to the no-slip condition
   - The viscous boundary layer forms along the hump surface
   - Flow acceleration occurs as fluid is squeezed between the hump and the droplet interface

### Implementation Details

**Location**: `src/boundary_conditions/velocity_bc.py`

```python
# Solid mask computed from height function
solid_mask = geometry_mask.compute_solid_mask(Nx, Ny, dy)
# Where: solid_mask[i, j] = True if y[j] < h_bottom[i]

# No-slip applied in solid regions
U = apply_no_slip_at_surface(U, solid_mask)
# Sets U = 0 where solid_mask is True
```

**Key Points**:
- The velocity field is **masked to zero** in all solid cells (below the hump surface)
- This happens **before** and **after** applying other boundary conditions
- The solid mask is updated dynamically if using phase-field geometry (ice)

### Visual Effects

- **Velocity streamlines** will curve around the hump
- **Velocity magnitude** will be reduced near the hump surface
- **Flow patterns** will show upward deflection over the hump
- **Recirculation zones** may form behind the hump (depending on flow speed)

---

## 2. Pressure Field

### Physical Effects

The hump geometry affects pressure through:

1. **Neumann Boundary Condition at Solid Surface**:
   - Pressure gradient normal to the surface is zero: `∂p/∂n = 0`
   - This is applied at the **actual solid surface**, not just at y=0
   - For a hump, this means the pressure gradient follows the surface contour

2. **Pressure Buildup**:
   - Fluid flowing over the hump experiences pressure variations
   - **High pressure** on the windward (upstream) side
   - **Low pressure** on the leeward (downstream) side
   - This creates a pressure gradient that drives flow around the hump

3. **Hydrostatic Effects**:
   - Pressure increases with depth (hydrostatic pressure)
   - The hump creates local variations in the hydrostatic pressure field
   - Fluid above the hump has less depth, so lower hydrostatic pressure

### Implementation Details

**Location**: `src/boundary_conditions/pressure_bc.py`

```python
# For Neumann BC at bottom with geometry:
if self.use_geometry and dy is not None:
    solid_mask = geometry_mask.compute_solid_mask(Nx, Ny, dy)
    
    # Apply Neumann at solid surface
    # For cells with solid at bottom, use value from first fluid cell above
    p_bottom = jnp.where(solid_mask[:, 0], p[:, 1], p[:, 0])
    p = p.at[:, 0].set(p_bottom)
```

**Key Points**:
- Pressure BC is applied at the **first fluid cell** above the solid surface
- The boundary condition follows the **contour of the hump**, not a flat plane
- This ensures pressure gradients are consistent with the geometry

### Visual Effects

- **Pressure contours** will follow the hump shape
- **High pressure** regions on the upstream side of the hump
- **Low pressure** regions on the downstream side
- **Pressure gradients** drive flow around the obstacle

---

## 3. Air-Water Phase Field (φ)

### Physical Effects

The hump geometry affects the phase field through:

1. **Geometry-Aware Contact Angle**:
   - The contact angle is applied **relative to the local surface normal**, not the horizontal
   - For a hump with slope `dh/dx`, the surface normal is: `n = (-dh/dx, 1) / √(1 + (dh/dx)²)`
   - The contact angle `θ` is measured from this **local normal**, not from vertical

2. **Droplet Shape Modification**:
   - The droplet interface must satisfy the contact angle condition at the hump surface
   - On the **windward slope** (positive slope), the effective contact angle changes
   - On the **leeward slope** (negative slope), the effective contact angle changes in the opposite direction
   - This causes the droplet to **wrap around** the hump, following its contour

3. **Interface Curvature**:
   - The non-uniform geometry creates **localized curvature** in the interface
   - Surface tension forces adjust to maintain the contact angle condition
   - The interface curvature is higher where the hump slope is steeper

### Implementation Details

**Location**: `src/boundary_conditions/contact_angle_bc.py`

```python
# Compute surface normal from height function
dh_dx = jax_dx(h_bottom, h=dx)  # Surface slope
n_surface_x = -dh_dx / sqrt(1 + dh_dx²)  # Normal x-component
n_surface_y = 1.0 / sqrt(1 + dh_dx²)      # Normal y-component

# Apply contact angle relative to surface normal
grad_phi_dot_n_surface = grad_phi_x * n_surface_x + grad_phi_y * n_surface_y
desired_normal_derivative = -cos(θ) * |∇φ|

# Correction to achieve desired contact angle
correction = desired_normal_derivative - grad_phi_dot_n_surface
phi_new = phi.at[:, 0].set(phi[:, 1] - correction_y * dy)
```

**Key Points**:
- The contact angle BC is applied at the **actual contact line** on the hump surface
- The surface normal is computed from `h_bottom(x)` using finite differences
- The phase field gradient is projected onto the surface normal
- The correction ensures `∂φ/∂n = -cos(θ) |∇φ|` at the surface

### Visual Effects

- **Droplet interface** follows the hump contour
- **Contact line** is not horizontal but follows the hump shape
- **Interface curvature** varies along the hump surface
- **Droplet shape** is asymmetric, with different contact angles on each side of the hump

---

## Summary of Interactions

### Coupling Between Fields

1. **Velocity ↔ Geometry**:
   - Geometry defines solid regions → velocity is zero in solid
   - Velocity gradients develop near the hump surface
   - Flow patterns are modified by the obstacle

2. **Pressure ↔ Geometry**:
   - Geometry defines boundary location → pressure BC applied at surface
   - Pressure gradients drive flow around the hump
   - Hydrostatic pressure varies with local depth

3. **Phase Field ↔ Geometry**:
   - Geometry defines surface normal → contact angle relative to normal
   - Phase field interface follows the hump contour
   - Surface tension forces adjust to maintain contact angle

4. **All Fields Together**:
   - Velocity affects phase field through advection
   - Pressure affects velocity through pressure gradient
   - Phase field affects velocity through surface tension
   - Geometry affects all three through boundary conditions

### Key Implementation Features

1. **Solid Mask**: `solid_mask[i, j] = True` where `y[j] < h_bottom[i]`
2. **Surface Normal**: Computed from `h_bottom(x)` using `n = (-dh/dx, 1) / √(1 + (dh/dx)²)`
3. **Boundary Conditions**: Applied at the **actual solid surface**, not at y=0
4. **Dynamic Updates**: Geometry mask can be updated from phase field (for ice) or height function

### Limitations and Future Improvements

1. **Current Implementation**:
   - Simplified no-slip at interface cells (sets velocity to zero)
   - Neumann pressure BC uses simple extrapolation
   - Contact angle BC assumes contact at y=0 (first grid row)

2. **Potential Improvements**:
   - More sophisticated interface treatment (project velocity onto surface normal)
   - Better pressure BC at curved surfaces (account for surface curvature)
   - Sub-grid contact line tracking (for better contact angle application)
   - Adaptive mesh refinement near the hump surface

---

## References

- **Velocity BC**: `src/boundary_conditions/velocity_bc.py`
- **Pressure BC**: `src/boundary_conditions/pressure_bc.py`
- **Contact Angle BC**: `src/boundary_conditions/contact_angle_bc.py`
- **Geometry Mask**: `src/boundary_conditions/geometry_mask.py`
- **General Geometry Documentation**: `tex/non_uniform_geometry_general.md`


