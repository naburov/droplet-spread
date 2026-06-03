# Implementation Plan: Curved / Terrain-Following Grid for Non-Flat Surfaces

## Goal

Solve the two-phase (air–water) system on **non-flat surfaces** by introducing a **coordinate transform** that aligns the grid with the surface. The transform is

- **Physical vertical coordinate**: \( y \) (height above a reference)
- **Surface height**: \( f(x) \) (given)
- **Computational vertical coordinate**: \( \eta = y - f(x) \)

So **\( \eta = 0 \)** is the surface; the bottom boundary of the **computational** domain is the physical surface. All PDEs and BCs are expressed in \( (x, \eta) \) with metric terms from the transform.

---

## 1. Current Code (What Exists)

### 1.1 Geometry and Surface

- **`Geometry`** (`simulation/geometry.py`): holds `h_bottom(x)` (surface height), `solid_mask`, `surface_normal`, `fluid_mask`, `has_geometry`.
- **`Geometry.from_height(h_bottom, Ny, dy, dx)`**: builds solid mask where **\( y < h_{\mathrm{bottom}}(x) \)** (physical y at cell centers: `y_coords[j] = j*dy`).
- **`geometry_mask.py`**: `compute_solid_mask_from_height`, `find_surface_cells(h_bottom, Ny, dy)` (first fluid cell per x), `compute_surface_normal_from_height` → **n** = (−dh/dx, 1) / √(1+(dh/dx)²).

So today: **Cartesian grid** \( (x_i, y_j) = (i\,dx,\, j\,dy) \), uniform `dx`, `dy`. Surface is **not** a grid line; it cuts through cells. Solid/fluid is decided by `y < h_bottom(x)`.

### 1.2 Where Geometry Is Used

| Component | Use of geometry / h_bottom |
|-----------|-----------------------------|
| **State** | `geometry` in `SimulationState`; `h_bottom` passed to surface tension BC, phase BC, velocity BC, pressure, PPE. |
| **Velocity BC** | `jax_apply_geometry_aware_velocity_bc(U, beta, dy, h_bottom, dx)` — no-slip along surface, surface cell index \( j_{\mathrm{surf}}(x) \), tangential slip option. |
| **Contact angle** | `_geometry_aware_contact_angle_jax`: surface normal from `h_bottom`, contact at cells near \( y = h_{\mathrm{bottom}}(x) \). |
| **Phase / advection BC** | `h_bottom` for impermeable / geometry-aware advection. |
| **Pressure** | `solid_mask` to zero out solid, mask RHS; `fluid_mask` for solver. |
| **Finite differences** | `jax_dx`, `jax_dy`, `jax_laplacian`, `jax_divergence` use **constant** `dx`, `dy`; `jax_divergence` can take `solid_mask` for one-sided stencils near solid. |
| **Poisson solver** | `SparseSolverWrapper(Nx, Ny, dx, dy)`: builds 1D Laplacians in x and y, Kronecker product; `solid_mask` applied to rows. **No metric** — assumes Cartesian. |
| **Initial conditions** | `initialize_phase` with `geometry.type` (flat / hump / tilted): droplet placed relative to \( h_{\mathrm{bottom}}(x) \), solid masked out. |

### 1.3 Summary of Current Approach

- **Single uniform Cartesian grid** in physical \( (x, y) \).
- Surface = level set \( y = f(x) \); **cut-cell style**: solid mask, surface cell indices, surface normals.
- All derivatives are **physical** \( \partial_x, \partial_y \) with constant spacing; no transform.

---

## 2. Target: Terrain-Following Coordinate \( \eta = y - f(x) \)

### 2.1 Transform

- **Physical**: \( (x, y) \), surface \( y = f(x) \).
- **Computational**: \( (x, \eta) \) with
  \[
  \eta = y - f(x),\qquad y = \eta + f(x).
  \]
- **Grid**: uniform in \( (x, \eta) \): \( x_i = i\,\Delta x \), \( \eta_j = j\,\Delta\eta \) with \( \eta_0 = 0 \) (or first cell center at \( \Delta\eta/2 \)) so that **\( \eta = 0 \) is the surface** (one grid line = surface).

### 2.2 Chain Rule (Metrics)

Let \( \xi = x \) (unchanged). Then \( \partial_x|_y = \partial_x|_\eta - f'(x)\,\partial_\eta \), \( \partial_y = \partial_\eta \). So:

- **Gradient** (scalar \( \phi \)):
  \[
  \nabla\phi = \bigl( \partial_x\phi - f'\,\partial_\eta\phi,\; \partial_\eta\phi \bigr).
  \]
- **Divergence** (vector **u** = (u, v) in physical (x,y)):
  \[
  \nabla\cdot\mathbf{u} = \partial_x u + \partial_y v = (\partial_x u - f'\,\partial_\eta u) + \partial_\eta v.
  \]
- **Laplacian** (scalar \( \phi \)): more involved; expands to terms like \( \partial_{xx}\phi \), \( \partial_{x\eta}\phi \), \( \partial_{\eta\eta}\phi \) with coefficients depending on \( f'(x) \), \( f''(x) \). Standard in terrain-following formulations (e.g. \( (1+f'^2)\partial_{\eta\eta}\phi + \ldots \)).

Jacobian of \( (x,y)\mapsto (x,\eta) \): \( J = 1 \). Contravariant velocity components in \( (x,\eta) \) can be taken as \( u \) (x-component) and \( v \) (y-component), with derivatives interpreted in \( (x,\eta) \) using the chain rule above.

### 2.3 Benefits

- **Surface = one grid line** (\( \eta = 0 \)): no cut cells, no \( j_{\mathrm{surf}}(x) \), no “first fluid cell” logic.
- **BCs**: no-slip and contact angle at **\( \eta = 0 \)** (single index \( j=0 \)), same as flat case in computational space.
- **Solid mask**: unnecessary in the fluid domain (all computational cells are fluid above the surface); or trivial “below \( \eta=0 \)” if we keep a ghost row.

---

## 3. Implementation Plan (Phased)

### Phase 1: Grid and metric representation

**1.1 Grid in computational space**

- Add a **grid/coordinates** module (or extend `Geometry`) that stores:
  - \( N_x, N_\eta \), \( \Delta x, \Delta\eta \)
  - \( f(x) \) at \( x_i \) (and optionally \( f'(x) \), \( f''(x) \) for Laplacian)
  - Optional: physical coordinates \( y_{ij} = \eta_j + f(x_i) \) for visualization / output only.
- **Convention**: \( \eta_0 = 0 \) (surface), \( \eta_j = j\,\Delta\eta \), so \( j=0 \) is the surface row; or \( \eta_{1/2} = \Delta\eta/2 \) as first cell center (then surface is at face between ghost and first cell). Choose one and stick to it (recommend: cell-centered \( \eta_j = (j+1/2)\Delta\eta \) for \( j=0,\ldots,N_\eta-1 \) with \( \eta=0 \) as boundary, so surface is at \( j=-1/2 \) in index, i.e. bottom boundary of domain).

Simplest: **cell-centered** \( \eta_j = j\,\Delta\eta \) for \( j = 0, \ldots, N_\eta-1 \) with \( j=0 \) the row **on** the surface (so \( \eta_0 = 0 \) or \( \eta_0 = \Delta\eta/2 \), to be fixed). Then “bottom” BC is at \( j=0 \).

**1.2 Metric fields**

- Precompute and store (per \( x_i \) or per \( (i,j) \)): \( f'(x_i) \), and if needed \( f''(x_i) \), as 1D arrays.
- Use these in all transformed derivative stencils.

**1.3 Config**

- In `initial_conditions.geometry` (or new `grid` section): allow `"type": "terrain_following"` with `"surface_function": "h_bottom"` (reuse existing \( h_{\mathrm{bottom}}(x) \)) or a new symbol (e.g. `f`). Keep existing `hump` / `tilted` as ways to **define** \( f(x) \); the new path is “once \( f(x) \) is defined, build a terrain-following grid”.

---

### Phase 2: Transformed finite differences

**2.1 Gradient (scalar)**

- Implement \( \nabla\phi \) in \( (x,\eta) \):
  - \( (\partial_x\phi)_\eta - f'(x)\,\partial_\eta\phi \) in x-component (use `jax_dx` for \( \partial_x\phi \), `jax_dy` for \( \partial_\eta\phi \), and 1D array \( f'(x) \) broadcast to grid).
  - \( \partial_\eta\phi \) in y-component.
- New routine e.g. `jax_gradient_terrain(phi, dx, d_eta, f_x)` where `f_x` is \( f'(x) \) at cell centers.

**2.2 Divergence (vector)**

- \( \nabla\cdot\mathbf{u} = (\partial_x u - f'\,\partial_\eta u) + \partial_\eta v \).
- New routine `jax_divergence_terrain(u, v, dx, d_eta, f_x)`.

**2.3 Laplacian (scalar)**

- Expand \( \nabla^2\phi \) in \( (x,\eta) \): write \( \partial_{xx}\phi + \partial_{yy}\phi \) in terms of \( \partial_x|_\eta, \partial_\eta \) and \( f', f'' \). Implement `jax_laplacian_terrain(phi, dx, d_eta, f_x, f_xx)`.
- Option: start with **constant** \( f' \) (e.g. tilted plane) to avoid mixed derivatives; then add full 2D form.

**2.4 Velocity Laplacian (for N–S)**

- Same metric as Laplacian: \( \nabla^2\mathbf{u} \) component-wise in \( (x,\eta) \) with same coefficients (and optionally different BCs for u/v).

**2.5 Integration**

- Replace calls to `jax_gradient`, `jax_divergence`, `jax_laplacian` in the **terrain-following** code path with the new routines when `grid.type == "terrain_following"`. Keep Cartesian path for flat / current hump (if we retain both).

---

### Phase 3: Boundary conditions in \( (x,\eta) \)

**3.1 Bottom (\( \eta = 0 \))**

- **Velocity**: no-slip \( u = 0, v = 0 \) at \( j=0 \) (same as current flat bottom).
- **Phase**: contact angle at \( j=0 \); normal in **physical** space is still \( \mathbf{n} = (-f', 1)/\sqrt{1+f'^2} \). In computational space the “normal derivative” to the boundary is \( \partial_\eta \); the relation \( \partial_n = \mathbf{n}\cdot\nabla \) gives the same constraint (already implemented in terms of gradient); we only need to evaluate the gradient with the terrain metric and apply at \( j=0 \).
- **Pressure**: Neumann \( \partial p/\partial n \) (or compatibility) at \( \eta=0 \); in terms of \( \partial_\eta p \) and \( f' \): \( \partial_n p = (\mathbf{n}\cdot\nabla)p \) with terrain gradient.

**3.2 Top, left, right**

- Unchanged in type (Neumann / Dirichlet / open); only derivative routines change to terrain versions when active.

**3.3 Removal of “surface cell” logic**

- With terrain-following grid, we no longer need `find_surface_cells`, “first fluid cell” per x, or solid mask for the **fluid** domain (the domain is only fluid). We can delete or bypass that branch when using terrain grid.

---

### Phase 4: Poisson solver (pressure) in \( (x,\eta) \)

**4.1 Transformed Laplacian matrix**

- Current: `A = kron(Iy, Tx) + kron(Tx, Iy)` with 1D Laplacians in x and y and **constant** dx, dy.
- Terrain: Laplacian has cross-term \( \partial_{x\eta} \) and variable coefficients. Options:
  - **Matrix**: build 2D 5- or 9-point stencil with spatially varying coefficients (from \( f'(x) \), \( f''(x) \)), then construct sparse matrix row by row. BCs at \( \eta=0 \) (e.g. Neumann) as now.
  - **Iterative**: keep using a solver (e.g. pyamg) but with a new matrix builder that encodes the terrain Laplacian.

**4.2 RHS and compatibility**

- RHS remains \( \nabla\cdot\mathbf{u}^*/\Delta t \) with **terrain** divergence.
- Compatibility (inlet Dirichlet velocity) unchanged in concept; implementation uses terrain divergence and gradient.

---

### Phase 5: Physics in \( (x,\eta) \)

**5.1 Momentum (N–S)**

- Convective term \( \mathbf{u}\cdot\nabla\mathbf{u} \): use terrain gradient for \( \nabla\mathbf{u} \).
- Viscous term \( (1/Re)\nabla^2\mathbf{u} \): use terrain Laplacian.
- Pressure gradient: terrain gradient of \( p \).
- Gravity: unchanged (vector in physical space); same \( (0, g) \) in y-component.
- Surface tension: \( \mathbf{F}_{\mathrm{ST}} \) already defined in physical space; compute curvature and \( \nabla\phi \) with terrain gradient; then add \( \mathbf{F}_{\mathrm{ST}}/\rho \) to momentum.

**5.2 Cahn–Hilliard**

- \( \partial_t\phi + \mathbf{u}\cdot\nabla\phi = (1/Pe)\Delta\mu \): advection with terrain gradient; Laplacian of \( \mu \) with terrain Laplacian.
- Chemical potential \( \mu = f'(\phi) - \varepsilon^2\Delta\phi \): Laplacian of \( \phi \) with terrain Laplacian.

**5.3 Curvature / surface tension**

- Curvature \( \kappa = \nabla\cdot(\nabla\phi/|\nabla\phi|) \): use terrain gradient and terrain divergence.

---

### Phase 6: Initial conditions and geometry

**6.1 Droplet on surface**

- In physical space the droplet sits on \( y = f(x) \). In \( (x,\eta) \), that is \( \eta = 0 \). So initial \( \phi \) in computational space: same semicircle/circle logic but with **vertical** coordinate \( \eta \) (distance above surface). E.g. distance from droplet center \( (x_c, \eta_c) \) with \( \eta_c = \eta_{\mathrm{center}} > 0 \), radius \( R \); \( \phi = \tanh((d - R)/(\varepsilon\sqrt{2})) \) with \( d = \sqrt{(x-x_c)^2 + (\eta-\eta_c)^2} \). No solid mask needed in the fluid domain.

**6.2 Geometry object**

- For terrain-following runs: `Geometry` can hold \( f(x) \), \( f'(x) \), and a flag `grid_aligned = True` so that BCs and numerics use the terrain path. `solid_mask` can be “all false” (no solid cells in domain) or “true only for \( j<0 \)” if we keep a ghost row.

---

### Phase 7: Solver and state wiring

**7.1 State**

- Add optional `grid_transform: "terrain_following"` and store \( f \), \( f' \) (and grid step \( d\eta \)) in state or in `Geometry`. All solvers that take `geometry` can branch on `grid_aligned` and call terrain FD routines and terrain Poisson.

**7.2 Two-phase loop**

- In `two_phase.py`: when `state.geometry.grid_aligned` (or config `grid.type == "terrain_following"`), use terrain gradient/divergence/Laplacian and terrain Poisson; pass terrain metric to BCs. Leave existing Cartesian + height-function path for backward compatibility.

**7.3 Config**

- Example new block:
  ```json
  "grid": {
    "type": "terrain_following",
    "surface": "from_geometry",
    "eta_min": 0,
    "eta_max": 1,
    "N_eta": 128
  }
  ```
  with `initial_conditions.geometry` still defining \( f(x) \) (e.g. hump or tilted). So \( L_\eta = \eta_{\max} - \eta_{\min} \), \( d\eta = L_\eta / N_\eta \).

---

## 4. File / Module Changes (Checklist)

| Area | Files to add or change |
|------|-------------------------|
| Grid / metrics | New `numerics/terrain_metrics.py` or `geometry/terrain_grid.py`: \( f(x) \), \( f'(x) \), grid lengths. |
| Finite differences | `numerics/finite_differences.py`: add `jax_gradient_terrain`, `jax_divergence_terrain`, `jax_laplacian_terrain` (or new `numerics/terrain_finite_differences.py`). |
| Poisson | `solvers/sparse_solver.py` or new `solvers/terrain_poisson.py`: build Laplacian matrix with terrain stencil. |
| Geometry | `simulation/geometry.py`: add `TerrainGeometry` or extend `Geometry` with `grid_aligned`, \( f \), \( f' \). |
| BCs | `velocity_bc`, `contact_angle_bc`, `phase_field_bc`: when grid is terrain, apply at \( j=0 \) with terrain normal / gradient. |
| State | `simulation/state.py`: create geometry with terrain when config says so; pass terrain flag and metrics to solvers. |
| Physics | `fluid_dynamics`, `phase_field`, `surface_tension`: branch on terrain and call terrain FD. |
| Initial conditions | `simulation/initial_conditions.py`: terrain branch for droplet on \( \eta=0 \). |
| Config | `config_loader`, `config_template`: optional `grid.type`, `grid.terrain_following`. |

---

## 5. Testing Strategy

1. **Flat \( f(x) = 0 \)**: terrain formulation with \( f' = 0 \), \( f'' = 0 \) must reproduce current Cartesian results (same equations).
2. **Tilted plane \( f(x) = \alpha x \)**: constant \( f' \); compare with current “tilted” geometry (same physics, different grid).
3. **Smooth hump**: compare droplet shape and contact line vs current height-function + cut-cell run (similar config, same \( f(x) \)).

---

## 6. Order of Work (Suggested)

1. **Phase 1** – Grid and \( f(x) \), \( f'(x) \) in config/geometry.  
2. **Phase 2** – Terrain gradient and divergence only; run with **flat** \( f=0 \) and compare to current (sanity check).  
3. **Phase 3** – BCs at \( \eta=0 \) (no-slip, contact angle) using terrain gradient.  
4. **Phase 2 (Laplacian)** – Terrain Laplacian; then **Phase 4** – Poisson with terrain Laplacian.  
5. **Phase 5** – Full N–S and Cahn–Hilliard in terrain form.  
6. **Phase 6–7** – Initial conditions, state wiring, config, and non-flat tests.

This keeps the change incremental and verifiable at each step.
