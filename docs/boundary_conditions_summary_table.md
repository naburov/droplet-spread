# Boundary Conditions at h_bottom(x) - Summary Table

## Quick Reference

| Variable | BC Type | Applied At | Implementation | Key Function |
|----------|---------|------------|----------------|--------------|
| **Velocity (U)** | No-slip | All cells at/below surface | `jax_apply_velocity_boundary_conditions` | `src/physics/fluid_dynamics.py:189` |
| **Phase Field (φ)** | Contact angle | Surface cells (contact line) | `_geometry_aware_contact_angle_impl` | `src/boundary_conditions/contact_angle_bc.py:14` |
| **Pressure (P)** | Dirichlet (solid) | Solid cells (below surface) | `_apply_solid_mask_to_matrix` | `src/solvers/sparse_solver.py:146` |
| **Surface Tension (F_sf)** | Contact angle | Surface cells (contact line) | `jax_apply_surface_tension_boundary_conditions` | `src/physics/surface_tension.py:132` |

## Detailed Implementation

### 1. Velocity (U)

**Function**: `jax_apply_velocity_boundary_conditions(U, beta, dy, h_bottom, dx)`

**Steps**:
1. Find surface cells: `surface_cell_indices = find_surface_cells(h_bottom, Ny, dy)`
2. Create mask: `cells_below_surface = j <= j_surf[i]` for all x
3. Apply no-slip: `U = 0` for all cells at or below surface
4. Clamp cells above: `v >= 0` for cells at `j = j_surf[i] + 1`

**Code Location**: `src/physics/fluid_dynamics.py:189-260`

### 2. Phase Field (φ)

**Function**: `_geometry_aware_contact_angle_impl(phi, dx, dy, psi, theta_effective, use_geometry_aware, h_bottom)`

**Steps**:
1. Find surface cells: `surface_cell_indices = find_surface_cells(h_bottom, Ny, dy)`
2. Compute surface normal: `n = (-dh/dx, 1) / sqrt(1 + (dh/dx)²)`
3. Compute gradient at first fluid cell above surface
4. Compute desired normal derivative: `∂φ/∂n = -cos(θ) |∇φ|`
5. Apply correction at surface cells: `φ_surf = φ_above - correction * dy / n_y`

**Code Location**: `src/boundary_conditions/contact_angle_bc.py:14-227`

### 3. Pressure (P)

**Function**: `_apply_solid_mask_to_matrix(A, solid_mask)`

**Steps**:
1. Compute solid mask: `solid_mask = y < h_bottom(x)`
2. Set in solver: `correction_solver.set_solid_mask(solid_mask, solid_value=0.0)`
3. Modify matrix: Solid cells → Identity row (Dirichlet: p = 0)
4. Interface cells → Boundary-aware stencils

**Code Location**: `src/solvers/sparse_solver.py:146-200`

### 4. Surface Tension (F_sf)

**Function**: `jax_apply_surface_tension_boundary_conditions(surface_tension, phi, contact_angle, h_bottom, dx, dy)`

**Steps**:
1. Find surface cells: `surface_cell_indices = find_surface_cells(h_bottom, Ny, dy)`
2. Compute surface normal: `n = (-dh/dx, 1) / sqrt(1 + (dh/dx)²)`
3. Project surface tension: `sf_normal = sf · n`, `sf_tangential = sf · t`
4. Apply contact angle: `sf_normal_adj = sf_normal * cos(θ)`
5. Convert back: `sf_new = sf_normal_adj * n + sf_tangential * t`
6. Apply ONLY where interface exists: `|phi| < 0.5`

**Code Location**: `src/physics/surface_tension.py:132-225`

## Common Pattern

All BCs follow this pattern:

```python
# 1. Find surface cells
from boundary_conditions.geometry_mask import find_surface_cells
surface_cell_indices, _ = find_surface_cells(h_bottom, Ny, dy)
j_surf = surface_cell_indices  # Shape: (Nx,)

# 2. Compute surface normal (if needed)
from numerics.finite_differences import jax_dx
dh_dx = jax_dx(h_bottom, h=dx)
n_surface_x = -dh_dx / sqrt(1 + dh_dx²)
n_surface_y = 1.0 / sqrt(1 + dh_dx²)

# 3. Apply BC at surface cells
x_indices = jnp.arange(Nx)
# Modify field at surface cells: field[x_indices, j_surf]
```

## Important Notes

1. **Surface cells are fluid cells**: They are part of the fluid domain, not solid
2. **Cells below surface are solid**: `y < h_bottom(x)` → solid mask = True
3. **BCs applied at surface, not y=0**: For hump geometry, BCs are at actual surface
4. **Surface normal matters**: For contact angle BCs, normal is computed from `dh/dx`
5. **Interface detection**: Some BCs only apply where interface exists (`|phi| < 0.5`)

