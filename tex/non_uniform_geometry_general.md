# Non-Uniform Bottom Geometry Implementation

## Overview

This implementation provides a **general framework** for handling non-uniform bottom boundaries in both:
- **Two-phase flow**: Arbitrary bottom geometry (height functions, obstacles, etc.)
- **Three-phase flow**: Ice surface geometry (from ice phase field)

The geometry can be specified via multiple representations:
1. **Height function** `h(x)`: Bottom surface height at each x position
2. **Phase field** (ice): Ice phase field `ψ` for three-phase flow
3. **Solid mask**: Boolean mask of solid regions

## Architecture

### Geometry Mask Module (`geometry_mask.py`)

The core module provides:

#### Functions:
- `compute_solid_mask_from_height()`: Create solid mask from height function
- `compute_solid_mask_from_phase_field()`: Create solid mask from phase field
- `compute_surface_height()`: Extract surface height from geometry
- `compute_surface_normal()`: Compute surface normal vectors
- `apply_no_slip_at_surface()`: Apply no-slip BC at solid surface

#### Class: `GeometryMask`
Manages geometry representation and provides unified interface:
```python
geometry_mask = GeometryMask(
    use_geometry=True,
    geometry_type="height"  # or "phase_field" or "mask"
)

# Set geometry
geometry_mask.set_height_function(h_bottom)  # For two-phase
# OR
geometry_mask.set_phase_field(psi)  # For three-phase (ice)
# OR
geometry_mask.set_solid_mask(solid_mask)  # Direct mask

# Use geometry
solid_mask = geometry_mask.compute_solid_mask(Nx, Ny, dy)
surface_normal = geometry_mask.compute_surface_normal(dx, dy)
```

## Usage

### Configuration

#### For Two-Phase Flow with Height Function:
```json
{
  "boundary_conditions": {
    "velocity": {
      "use_geometry": true,
      "geometry_type": "height"
    },
    "pressure": {
      "use_geometry": true,
      "geometry_type": "height"
    }
  }
}
```

Then in code, set the height function:
```python
# Define bottom surface height h(x)
h_bottom = ...  # Array of shape (Nx,)

# Set in geometry mask
velocity_bc.geometry_mask.set_height_function(h_bottom)
pressure_bc.geometry_mask.set_height_function(h_bottom)
```

#### For Three-Phase Flow with Ice:
```json
{
  "boundary_conditions": {
    "velocity": {
      "use_geometry": true,
      "geometry_type": "phase_field"
    },
    "pressure": {
      "use_geometry": true,
      "geometry_type": "phase_field"
    }
  }
}
```

The ice phase field `ψ` is automatically used when passed to BCs:
```python
# Ice phase field is automatically detected
U = velocity_bc.apply_boundary_conditions(U, dx, dy, psi=psi)
p = pressure_bc.apply_boundary_conditions(p, psi=psi, dy=dy)
```

### Example: Two-Phase Flow with Sinusoidal Bottom

```python
import numpy as np
import jax.numpy as jnp

# Create sinusoidal bottom surface
Nx = 128
Lx = 1.0
x = np.linspace(0, Lx, Nx)
h_bottom = 0.1 * np.sin(2 * np.pi * x / Lx)  # Height function

# Initialize BCs with geometry
velocity_bc = VelocityBoundaryConditions(config)
velocity_bc.geometry_mask.set_height_function(jnp.array(h_bottom))

# Apply BCs (geometry is automatically used)
U = velocity_bc.apply_boundary_conditions(U, dx, dy)
```

### Example: Three-Phase Flow with Ice

```python
# Ice phase field is automatically used
U = velocity_bc.apply_boundary_conditions(U, dx, dy, psi=psi)
p = pressure_bc.apply_boundary_conditions(p, psi=psi, dy=dy)
```

## Boundary Conditions

### Velocity BCs

**No-slip at solid surface:**
- Velocity is zero in solid regions
- Velocity is zero at interface cells (fluid cells adjacent to solid)
- Applied before and after other boundary conditions

**Geometry-aware:**
- Uses solid mask to identify solid regions
- Applies BCs at actual surface location, not just y=0
- Works with height functions, phase fields, or direct masks

### Pressure BCs

**Neumann at solid surface:**
- `∂p/∂n = 0` at solid surface
- Extrapolates from fluid cells when solid is at boundary
- Accounts for non-uniform geometry

## Mathematical Formulation

### Height Function Approach

Given bottom surface height `h(x)`:
- **Solid mask**: `solid(x,y) = True` if `y < h(x)`
- **Surface normal**: `n = (-dh/dx, 1) / sqrt(1 + (dh/dx)²)`
- **Surface height**: `h_surface(x) = h(x)`

### Phase Field Approach

Given ice phase field `ψ`:
- **Solid mask**: `solid(x,y) = True` if `ψ(x,y) > threshold`
- **Surface normal**: `n = -∇ψ / |∇ψ|` at surface
- **Surface height**: `h_surface(x) = y` where `ψ(x,y) ≈ 0`

## Implementation Details

### Solid Mask Computation

The solid mask identifies where flow is **not allowed**:
- `solid_mask[i,j] = True` → cell `(i,j)` is solid
- `solid_mask[i,j] = False` → cell `(i,j)` is fluid

### Interface Cells

Interface cells are fluid cells adjacent to solid cells:
- Used for applying boundary conditions
- Ensures smooth transition between solid and fluid

### Surface Normal

Surface normal is computed from:
- **Height function**: `n = (-dh/dx, 1) / sqrt(1 + (dh/dx)²)`
- **Phase field**: `n = -∇ψ / |∇ψ|`

Used for:
- Contact angle boundary conditions
- Pressure boundary conditions
- Surface tension calculations

## Advantages

1. **General**: Works for any geometry representation
2. **Flexible**: Can switch between height function, phase field, or mask
3. **Unified**: Same interface for two-phase and three-phase flows
4. **Backward Compatible**: Falls back to simple masking if geometry disabled

## Future Enhancements

1. **Level Set Method**: Add support for level set functions (signed distance)
2. **Moving Boundaries**: Handle time-dependent geometry
3. **Complex Shapes**: Support for obstacles, pillars, etc.
4. **Solver Integration**: Modify solvers to exclude solid regions from calculations

## Solver Modifications Needed

For complete implementation, solvers need to:

1. **PPE Solver**: Exclude solid regions from divergence calculations
2. **Pressure Solver**: Modify sparse matrix to handle non-uniform boundaries
3. **Velocity Update**: Account for geometry in velocity calculations

These modifications are the next step in the implementation.


