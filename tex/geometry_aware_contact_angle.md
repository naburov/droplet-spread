# Geometry-Aware Contact Angle Boundary Condition

## Overview

The geometry-aware contact angle boundary condition accounts for non-flat ice surfaces when applying the contact angle at the substrate. Instead of assuming a flat surface at y=0, it computes the local ice surface normal and applies the contact angle relative to that normal.

## Mathematical Formulation

### Traditional Approach (Flat Surface)
The traditional contact angle BC assumes a flat substrate:
```
∂φ/∂n = -cos(θ) |∇φ|
```
where the normal `n` is vertical (pointing upward).

### Geometry-Aware Approach (Non-Flat Ice Surface)
When ice forms, the surface may be non-flat. The geometry-aware BC:

1. **Computes ice surface normal** from the gradient of the ice phase field:
   ```
   n_ice = -∇ψ / |∇ψ|
   ```
   where `n_ice` points from ice (ψ > 0) to water (ψ < 0).

2. **Applies contact angle relative to ice surface normal**:
   ```
   ∂φ/∂n_ice = -cos(θ) |∇φ|
   ```
   where the contact angle is now measured from the ice surface normal, not the horizontal.

3. **Projects correction** onto the vertical direction for application at the boundary.

## Implementation

### Configuration

To enable geometry-aware contact angle BC, add to your config file:

```json
{
  "boundary_conditions": {
    "phase_field": {
      "contact_angle_method": "geometry_aware",
      "use_geometry_aware": true
    }
  }
}
```

Or set `contact_angle_method` to `"geometry_aware"` (this automatically enables geometry-aware mode).

### Code Usage

The geometry-aware BC is automatically used when:
- `use_geometry_aware=True` is set in `ContactAngleBoundaryCondition`
- OR `method="geometry_aware"` is specified

The BC requires the ice phase field `psi` to be passed:

```python
phi_new = contact_angle_bc.apply(phi, dx, dy, psi=psi)
```

## Key Features

1. **Automatic Detection**: Detects ice presence at the substrate (where `ψ > 0`)
2. **Surface Normal Computation**: Computes local ice surface normal from `∇ψ`
3. **Fallback to Substrate**: If no ice is present, uses vertical normal (substrate)
4. **Ice-Aware Contact Angle**: Can use different contact angles for ice vs. water (if `use_ice_aware=True`)

## Physical Interpretation

When ice forms a non-flat surface:
- The liquid-gas interface contacts ice at various heights
- The contact angle should be measured relative to the local ice surface normal
- This accounts for ice surface roughness, protrusions, or complex geometries

## Limitations

1. **BC Applied at y=0**: The BC is still applied at the flat substrate (y=0), but uses the ice surface normal. For very thick ice layers, this may not capture the full geometry.

2. **Gradient at Boundary**: The ice surface normal is computed from the gradient at the boundary. For very sharp ice features, this may need refinement.

3. **Mixed Regions**: When both ice and substrate are present, the normal is computed from ice gradient. The transition between ice and substrate regions is handled smoothly.

## Example

```python
# Initialize with geometry-aware mode
contact_angle_bc = ContactAngleBoundaryCondition(
    contact_angle=60,  # degrees
    contact_angle_ice=80,  # different angle for ice
    method="geometry_aware",
    use_ice_aware=True,
    use_geometry_aware=True
)

# Apply BC (requires psi for geometry-aware mode)
phi_new = contact_angle_bc.apply(phi, dx, dy, psi=psi)
```

## Comparison with Traditional BC

| Feature | Traditional BC | Geometry-Aware BC |
|---------|---------------|-------------------|
| Surface assumption | Flat at y=0 | Non-flat (ice surface) |
| Normal direction | Vertical (0, 1) | From ice gradient |
| Ice geometry | Ignored | Accounted for |
| Contact angle reference | Horizontal | Ice surface normal |
| Computational cost | Low | Slightly higher (gradient computation) |

## Future Improvements

Potential enhancements:
1. **Height-dependent BC**: Apply BC at actual ice surface location (not just y=0)
2. **Surface tracking**: Explicitly track ice surface height h_ice(x)
3. **Multi-scale**: Handle ice features at different length scales
4. **Curvature correction**: Account for ice surface curvature in contact angle


