# Boundary Conditions at h_bottom(x)

This document explains how boundary conditions are applied at the actual surface `y = h_bottom(x)` for each variable in the droplet spreading simulation.

## Overview

For a non-flat surface (hump geometry), boundary conditions are applied at the actual surface position `y = h_bottom(x)` rather than at the computational boundary `y = 0`. This requires:

1. **Finding surface cells**: For each x, find the cell index j where `y[j] ≈ h_bottom(x)`
2. **Applying BCs at surface cells**: Apply appropriate BCs at those cells
3. **Handling cells below surface**: These are solid (no flow, no phase field evolution)

## Finding Surface Cells

The function `find_surface_cells(h_bottom, Ny, dy)` is used to find surface cells:

```python
from boundary_conditions.geometry_mask import find_surface_cells

# For each x, find cell index j closest to h_bottom(x)
surface_cell_indices, near_surface_mask = find_surface_cells(h_bottom, Ny, dy)
# surface_cell_indices: (Nx,) array of j indices
# near_surface_mask: (Nx, Ny) boolean mask
```

**Algorithm**:
- For each x position, compute distance from each y cell to `h_bottom(x)`
- Find cell with minimum distance: `j_surf[i] = argmin_j |y[j] - h_bottom[i]|`
- Create mask for cells at or near surface (within tolerance)

**Key Point**: `surface_cell_indices[i]` gives the cell index `j` for each `x[i]` where the surface is located.

## Boundary Conditions by Variable

### 1. Velocity (U)

**Location**: `src/physics/fluid_dynamics.py::jax_apply_velocity_boundary_conditions`

**BC Type**: No-slip at surface

**Implementation**:
```python
# Find surface cells
surface_cell_indices, _ = find_surface_cells(h_bottom, Ny, dy)

# Apply no-slip: U = 0 at all cells at or below surface
j_surf = surface_cell_indices  # Shape: (Nx,)
cells_below_surface = j_indices <= j_surf[:, None]  # Shape: (Nx, Ny)

U = U.at[..., 0].set(jnp.where(cells_below_surface, 0.0, U[..., 0]))
U = U.at[..., 1].set(jnp.where(cells_below_surface, 0.0, U[..., 1]))

# Also clamp cells just above surface: v >= 0 (not pointing into solid)
j_above_surf = j_surf + 1
j_above_mask = j_indices == j_above_surf[:, None]
v_clamped = jnp.where(v_current < 0.0, 0.0, v_current)
U = U.at[..., 1].set(jnp.where(j_above_mask, v_clamped, U[..., 1]))
```

**Key Points**:
- All cells at or below surface: `U = 0` (no-slip)
- Cells just above surface: `v >= 0` (no flow into solid)
- Surface normal is computed from `dh/dx` for geometry-aware contact angle

### 2. Phase Field (φ)

**Location**: `src/boundary_conditions/contact_angle_bc.py::_geometry_aware_contact_angle_impl`

**BC Type**: Contact angle at surface

**Implementation**:
```python
# Step 1: Find surface cells
surface_cell_indices, _ = find_surface_cells(h_bottom, Ny, dy)
j_surf = surface_cell_indices  # Shape: (Nx,)

# Step 2: Compute surface normal from h_bottom
dh_dx = jax_dx(h_bottom, h=dx)  # Shape: (Nx,)
norm_factor = sqrt(1 + dh_dx²)
n_surface_x = -dh_dx / norm_factor  # Shape: (Nx,)
n_surface_y = 1.0 / norm_factor      # Shape: (Nx,)

# Step 3: Compute gradient of φ at first fluid cell above surface
grad_phi = jax_gradient(phi, dx, dy)
grad_phi_x = grad_phi[:, 1, 0]  # At y=1 (first interior)
grad_phi_y = grad_phi[:, 1, 1]

# Step 4: Compute normal derivative relative to surface
grad_phi_dot_n = grad_phi_x * n_surface_x + grad_phi_y * n_surface_y

# Step 5: Desired normal derivative for contact angle θ
# ∂φ/∂n = -cos(θ) |∇φ|
desired_normal_derivative = -cos(θ) * |∇φ|

# Step 6: Correction needed
correction = desired_normal_derivative - grad_phi_dot_n

# Step 7: Apply at surface cells
x_indices = jnp.arange(Nx)
phi_surf = phi[x_indices, j_surf]
phi_above = phi[x_indices, j_surf + 1]
phi_surf_new = phi_above - correction * dy / n_surface_y
phi = phi.at[x_indices, j_surf].set(phi_surf_new)
```

**Key Points**:
- Contact angle BC applied at actual surface `y = h_bottom(x)`
- Surface normal computed from `dh/dx`
- Correction applied to match desired contact angle
- Only applied where interface exists (contact line)

### 3. Pressure (P)

**Location**: `src/solvers/sparse_solver.py::_apply_solid_mask_to_matrix`

**BC Type**: Dirichlet in solid, Neumann at domain boundaries

**Implementation**:
```python
# Solid mask identifies cells below surface
solid_mask = y < h_bottom(x)  # Shape: (Nx, Ny)

# Set solid mask in solver
correction_solver.set_solid_mask(solid_mask, solid_value=0.0)

# The matrix is modified:
# - Solid cells: Identity row (p = 0, Dirichlet BC)
# - Interface cells (fluid adjacent to solid): Use boundary-aware stencils
# - Domain boundaries: Neumann BC (zero normal gradient)
```

**Key Points**:
- Solid cells (below surface) are treated as Dirichlet: `p = 0` (or `p = p_atm`)
- Surface cells are part of the fluid domain (not solid)
- Matrix Laplacian uses boundary-aware stencils for interface cells
- Domain boundaries (y=0, y=Ly, x=0, x=Lx) use Neumann BC

### 4. Surface Tension (F_sf)

**Location**: `src/physics/surface_tension.py::jax_apply_surface_tension_boundary_conditions`

**BC Type**: Contact angle at contact line

**Implementation**:
```python
# Step 1: Find surface cells
surface_cell_indices, _ = find_surface_cells(h_bottom, Ny, dy)
j_surf = surface_cell_indices  # Shape: (Nx,)

# Step 2: Compute surface normal
dh_dx = jax_dx(h_bottom, h=dx)  # Shape: (Nx,)
norm_factor = sqrt(1 + dh_dx²)
n_surface_x = -dh_dx / norm_factor  # Shape: (Nx,)
n_surface_y = 1.0 / norm_factor      # Shape: (Nx,)

# Step 3: Get surface tension at surface cells
x_indices = jnp.arange(Nx)
sf_surf_x = surface_tension[x_indices, j_surf, 0]  # Shape: (Nx,)
sf_surf_y = surface_tension[x_indices, j_surf, 1]  # Shape: (Nx,)

# Step 4: Project onto surface normal and tangential
sf_normal = sf_surf_x * n_surface_x + sf_surf_y * n_surface_y
sf_tangential = sf_surf_x * (-n_surface_y) + sf_surf_y * n_surface_x

# Step 5: Apply contact angle: normal component scaled by cos(θ)
theta = (180 - contact_angle) * π / 180
sf_normal_adj = sf_normal * cos(theta)

# Step 6: Convert back to x, y components
sf_x_new = sf_normal_adj * n_surface_x - sf_tangential * n_surface_y
sf_y_new = sf_normal_adj * n_surface_y + sf_tangential * n_surface_x

# Step 7: Apply ONLY where interface exists (contact line)
phi_at_surface = phi[x_indices, j_surf]
interface_at_surface = abs(phi_at_surface) < 0.5
sf = sf.at[x_indices, j_surf, 0].set(jnp.where(interface_at_surface, sf_x_new, 0.0))
sf = sf.at[x_indices, j_surf, 1].set(jnp.where(interface_at_surface, sf_y_new, 0.0))
```

**Key Points**:
- Contact angle BC applied at surface cells where interface exists (contact line)
- Surface tension is zeroed in solid regions, EXCEPT at contact line
- Surface normal computed from `dh/dx`
- Only applied where `|phi| < 0.5` (interface exists)

## Solid Mask

The solid mask identifies cells that are solid (below surface):

```python
solid_mask = y < h_bottom(x)  # Shape: (Nx, Ny)
```

**Usage**:
- Velocity: `U = 0` in solid cells
- Phase field: No evolution in solid cells
- Pressure: Dirichlet BC in solid cells
- Surface tension: Zero in solid cells (except contact line)

## Surface Cell Identification

For each x position:
- **Surface cell**: `j_surf[i]` = cell index closest to `h_bottom[i]`
- **Cells below surface**: `j < j_surf[i]` (solid)
- **Cells at surface**: `j = j_surf[i]` (boundary)
- **Cells above surface**: `j > j_surf[i]` (fluid)

## Key Functions

1. **`find_surface_cells(h_bottom, Ny, dy)`**: Find surface cell indices
2. **`compute_solid_mask_from_height(h_bottom, Ny, dy)`**: Compute solid mask
3. **`jax_apply_velocity_boundary_conditions(U, beta, dy, h_bottom, dx)`**: Apply velocity BCs
4. **`jax_apply_surface_tension_boundary_conditions(sf, phi, contact_angle, h_bottom, dx, dy)`**: Apply surface tension BCs

## Example: Applying BC at Surface

```python
from boundary_conditions.geometry_mask import find_surface_cells
import jax.numpy as jnp

# Find surface cells
surface_cell_indices, _ = find_surface_cells(h_bottom, Ny, dy)
j_surf = surface_cell_indices  # Shape: (Nx,)

# Apply BC at surface cells
x_indices = jnp.arange(Nx)
# Example: Set some value at surface
field = field.at[x_indices, j_surf].set(surface_value)
```

## Important Notes

1. **Surface cells are fluid cells**: They are part of the fluid domain, not solid
2. **Cells below surface are solid**: They have `U = 0`, no phase field evolution
3. **BCs are applied at surface cells**: Not at computational boundary `y = 0`
4. **Surface normal matters**: For contact angle BCs, surface normal is computed from `dh/dx`
5. **Interface at surface**: When interface touches surface, special handling is needed for contact line

