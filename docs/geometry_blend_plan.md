# Plan: Blending Geometry into Equations via Terrain-Following Coordinates

**Goal:** Use `Geometry.f(x)`, `f_1(x)`, `f_2(x)` in the governing equations by expressing derivatives in terrain-following coordinates \( (x, \eta) \) with \( \eta = y - f(x) \).

**Design choices:**
1. **Single code path:** Diff operators always use the terrain form and take geometry (or f_1, f_2). For the flat case, geometry has \( f = f' = f'' = 0 \), so the extra terms cancel and we recover the Cartesian form. No branching on “flat vs terrain” inside the operators.
2. **No recomputation:** Geometry stores f_1 and f_2 after computation. Use `geometry.f_1_grid()` and `geometry.f_2_grid()` (shape (Nx,)) in diff operators so we never recompute them.

---

## 1. Coordinate Transform and Operator Forms

### 1.1 Transform

- **Physical:** \( (x, y) \), surface \( y = f(x) \).
- **Computational:** \( (x, \eta) \) with \( \eta = y - f(x) \), so \( y = \eta + f(x) \).
- **Chain rule:** \( \partial/\partial x\big|_y = \partial/\partial x\big|_\eta - f'(x)\,\partial/\partial\eta \), \( \partial/\partial y = \partial/\partial\eta \).

We use \( f' = \texttt{geometry.f\_1\_grid()} \), \( f'' = \texttt{geometry.f\_2\_grid()} \) (stored, shape (Nx,)) so they are not recomputed. For flat geometry, these are zero arrays and the terrain terms cancel.

### 1.2 Gradient (scalar \( \phi \))

In physical \( (x,y) \): \( \nabla\phi = (\partial_x\phi, \partial_y\phi) \).

In \( (x,\eta) \):
\[
\nabla\phi = \bigl( (\partial_x\phi)_\eta - f'\,(\partial_\eta\phi),\;\; (\partial_\eta\phi) \bigr).
\]

- **Where mixed terms appear:** The x-component mixes \( \partial_x\phi \) (standard x-difference) and \( \partial_\eta\phi \) (standard y-difference) via \( -f'\,\partial_\eta\phi \). No second derivatives; no \( \partial_{x\eta} \) here.
- **Discretization:** \( \partial_x\phi \to \texttt{jax\_dx(phi, dx)} \), \( \partial_\eta\phi \to \texttt{jax\_dy(phi, dy)} \). Use \( f' = \texttt{geometry.f\_1\_grid()} \) (shape (Nx,), broadcast to grid) and subtract \( f' * \partial_\eta\phi \) from the x-component. Flat: f_1_grid is zeros, so the term cancels.

### 1.3 Divergence (vector \( \mathbf{u} = (u, v) \))

In physical: \( \nabla\cdot\mathbf{u} = \partial_x u + \partial_y v \).

In \( (x,\eta) \):
\[
\nabla\cdot\mathbf{u} = (\partial_x u)_\eta - f'\,(\partial_\eta u) + \partial_\eta v.
\]

- **Where mixed terms appear:** Again only first derivatives; the “mixing” is \( -f'\,\partial_\eta u \). No second or mixed second derivatives.
- **Discretization:** \( \partial_x u \to \texttt{jax\_dx(u, dx)} \), \( \partial_\eta u \to \texttt{jax\_dy(u, dy)} \), \( \partial_\eta v \to \texttt{jax\_dy(v, dy)} \). Use \( f' = \texttt{geometry.f\_1\_grid()} \), broadcast; flat: f_1 is zeros, term cancels.

### 1.4 Laplacian (scalar \( \phi \))

In physical: \( \nabla^2\phi = \partial_{xx}\phi + \partial_{yy}\phi \).

Applying the chain rule to \( \partial/\partial x\big|_y = \partial_x - f'\,\partial_\eta \) and \( \partial/\partial y = \partial_\eta \):
\[
\nabla^2\phi = \partial_{xx}\phi - 2f'\,\partial_{x\eta}\phi - f''\,\partial_\eta\phi + \bigl(1 + (f')^2\bigr)\,\partial_{\eta\eta}\phi.
\]

- **Where mixed derivatives appear:** Explicitly in the **cross term** \( -2f'\,\partial_{x\eta}\phi \). The term \( -f''\,\partial_\eta\phi \) is first order in \( \eta \); \( (1+(f')^2)\partial_{\eta\eta}\phi \) is pure \( \eta \)-Laplacian.
- **Mixed derivative \( \partial_{x\eta}\phi \):** Discretize as \( \partial/\partial x\,(\partial_\eta\phi) \) or \( \partial/\partial\eta\,(\partial_x\phi) \), i.e. apply `jax_dx` to `jax_dy(phi)` or `jax_dy` to `jax_dx(phi)` (both give the same cross-derivative stencil on the grid).
- **Coefficients:** All depend on \( x \) only. Use \( f' = \texttt{geometry.f\_1\_grid()} \), \( f'' = \texttt{geometry.f\_2\_grid()} \) (stored, no recomputation). Broadcast to \( (N_x, N_y) \) for pointwise multiplication. Flat: f_1 and f_2 are zeros, so cross term and \(-f''\partial_\eta\phi\) vanish and \(1+(f')^2 = 1\).

### 1.5 Summary of Mixed/Second-Derivative Usage

| Operator    | Mixed derivative?        | Uses \( f \), \( f' \), \( f'' \)? | Notes |
|------------|---------------------------|-------------------------------------|--------|
| Gradient   | No                        | \( f' \) only (x-component)         | \( -f'\partial_\eta\phi \) |
| Divergence | No                        | \( f' \) only                       | \( -f'\partial_\eta u \) |
| Laplacian  | **Yes:** \( \partial_{x\eta}\phi \) | \( f', f'' \), and \( 1+(f')^2 \) | Single place where \( \partial_{x\eta} \) appears |

So the **only** operator that introduces a true mixed second derivative (\( \partial_{x\eta} \)) is the **Laplacian**. Gradient and divergence only mix first derivatives with coefficients \( f' \).

---

## 2. Where Each Operator Is Used (Codebase Map)

### 2.1 Gradient

| Module / file | Function / use | Purpose |
|---------------|----------------|---------|
| `numerics/finite_differences.py` | `jax_gradient(f, dx, dy)` | Core implementation (Cartesian). |
| `physics/fluid_dynamics.py` | `jax_gradient(U, dx, dy)` | Convective term \( \mathbf{u}\cdot\nabla\mathbf{u} \); also pressure gradient \( \nabla p \). |
| `physics/phase_field.py` | `jax_gradient(phi, dx, dy)` | Cahn–Hilliard advection \( \mathbf{u}\cdot\nabla\phi \), and in \( \mu \). |
| `physics/surface_tension.py` | `jax_gradient(phi, dx, dy)` | Curvature \( \kappa = \nabla\cdot(\nabla\phi/|\nabla\phi|) \); also ST force. |
| `physics/temperature.py` | `jax_gradient(T, dx, dy)`, `jax_gradient(alpha, ...)` | Temp and phase-dependent coeffs. |
| `physics/ice_phase_field.py` | `jax_gradient(psi, dx, dy)` | Ice phase. |
| `boundary_conditions/contact_angle_bc.py` | `jax_gradient(phi, dx, dy)` | Contact angle \( \partial_n\phi \propto \nabla\phi\cdot\mathbf{n} \). |
| `boundary_conditions/geometry_mask.py` | `jax_gradient(psi, dx, dy)` | Surface normal from \( \psi \). |
| `simulation/state.py` | `jax_gradient(self.phi, ...)` | Interface mask / surface tension. |
| `solvers/ppe_utils.py` | `jax_gradient_solid_aware(p_correction, ...)` | Pressure correction gradient; also `jax_dx`, `jax_dy` for \( \nabla p' \). |

**Introduction point:** Single gradient routine that always takes geometry (or f_1_grid). Use \( f_1 = \texttt{geometry.f\_1\_grid()} \) (stored); flat geometry gives zeros so the terrain term cancels. No separate “Cartesian” path.

### 2.2 Divergence

| Module / file | Function / use | Purpose |
|---------------|----------------|---------|
| `numerics/finite_differences.py` | `jax_divergence(f, dx, dy, solid_mask=None)` | Core (Cartesian). |
| `physics/fluid_dynamics.py` | `jax_divergence(U, dx, dy, solid_mask=...)` | Continuity check; predictor momentum not used directly here but div of U used in PPE. |
| `physics/pressure.py` | `jax_divergence(surface_tension, dx, dy)` | Divergence of ST field for pressure. |
| `solvers/ppe_utils.py` | `jax_divergence(U, dx, dy)` | RHS of pressure Poisson: \( \nabla\cdot\mathbf{u}^*/\Delta t \). |

**Introduction point:** Single divergence routine that always takes geometry; use \( f_1 = \texttt{geometry.f\_1\_grid()} \). Flat: f_1 is zeros, term cancels. No separate Cartesian path.

### 2.3 Laplacian

| Module / file | Function / use | Purpose |
|---------------|----------------|---------|
| `numerics/finite_differences.py` | `jax_laplacian(f, dx, dy)` | Core (Cartesian). |
| `physics/fluid_dynamics.py` | `jax_laplacian(U[...,0], ...)`, `jax_laplacian(U[...,1], ...)` | Viscous term \( (1/Re)\nabla^2\mathbf{u} \). |
| `physics/phase_field.py` | `jax_laplacian(phi, dx, dy)` | Cahn–Hilliard \( \mu = f'(\phi)-\varepsilon^2\nabla^2\phi \); also \( \Delta\mu \) in \( \partial_t\phi \); Willmore \( \Delta^2\phi \) (Laplacian of Laplacian). |
| `physics/surface_tension.py` | Curvature: `jax_dx(n_x, dx) + jax_dy(n_y, dy)` | \( \kappa = \nabla\cdot\mathbf{n} \); this is a **divergence** of a vector that depends on gradient of \( \phi \), so terrain gradient + terrain divergence. |
| `physics/temperature.py` | `jax_laplacian(T, ...)`, `jax_laplacian(alpha, ...)` | Diffusion. |
| `physics/ice_phase_field.py` | `jax_laplacian(psi, dx, dy)` | Allen–Cahn / Cahn–Hilliard ice. |
| `boundary_conditions/contact_angle_bc.py` | `jax_dx(jax_dx(phi,...), ...)` and `jax_dy(jax_dy(phi,...), ...)` | Some paths use explicit \( \phi_{xx} \), \( \phi_{yy} \) (e.g. Young–Laplace). |
| `numerics/rhie_chow.py` | `jax_laplacian(p, dx, dy)` | Rhie–Chow pressure Laplacian. |

**Introduction point:** Single Laplacian routine that always takes geometry; use \( f_1 = \texttt{geometry.f\_1\_grid()} \), \( f_2 = \texttt{geometry.f\_2\_grid()} \) (stored). Implementing
\[
\nabla^2\phi \to \partial_{xx}\phi - 2f'\,\partial_{x\eta}\phi - f''\,\partial_\eta\phi + (1+(f')^2)\,\partial_{\eta\eta}\phi,
\]
with \( \partial_{x\eta}\phi \) discretized as `jax_dx(jax_dy(phi, dy), dx)` or `jax_dy(jax_dx(phi, dx), dy)`. This is the **only** place where the mixed derivative \( \partial_{x\eta} \) is introduced. All momentum, phase-field, and (if used) pressure Laplacians should call this when geometry is non-flat.

### 2.4 Direct use of `jax_dx` / `jax_dy`

Used in: `ppe_utils.py` (pressure gradient components), `surface_tension.py` (curvature: \( \partial_x n_x + \partial_y n_y \)), `contact_angle_bc.py`, `geometry_mask.py`. Under terrain, any **physical** \( \partial_x \) or \( \partial_y \) that appears in a PDE (not just in BC helpers) should go through the terrain gradient/divergence/Laplacian; standalone `jax_dx`/`jax_dy` in those PDE paths then become “along computational directions” and get combined with \( f' \) in the formulas above. So the main blend points are the three operators; `jax_dx`/`jax_dy` stay as the building blocks for \( \partial_x|_\eta \) and \( \partial_\eta \).

---

## 3. Where and How to Introduce Mixed Derivatives (Concrete)

### 3.1 Single point of mixed derivative: Laplacian

- **Where:** In the **Laplacian** only. No mixed derivatives in gradient or divergence.
- **How:**
  - One Laplacian routine that takes \( (phi, dx, dy, geometry) \). Read \( f_1 = \texttt{geometry.f\_1\_grid()} \), \( f_2 = \texttt{geometry.f\_2\_grid()} \) (stored; flat gives zeros).
  - Compute on the grid:
    - \( \phi_{xx} \): existing `jax_dx(jax_dx(phi, dx), dx)` (or Laplacian-x part).
    - \( \phi_{x\eta} \): **mixed** — `jax_dx(jax_dy(phi, dy), dx)` (or equivalently `jax_dy(jax_dx(phi, dx), dy)`).
    - \( \phi_\eta \): `jax_dy(phi, dy)`.
    - \( \phi_{\eta\eta} \): `jax_dy(jax_dy(phi, dy), dy)`.
  - Combine: \( \phi_{xx} - 2\,f'\,\phi_{x\eta} - f''\,\phi_\eta + (1+(f')^2)\,\phi_{\eta\eta} \), with \( f' \), \( f'' \) broadcast to \( (N_x, N_y) \).
  - **File:** In `numerics/finite_differences.py` (or a small `numerics/terrain_operators.py`) using `jax_dx`, `jax_dy`. Same routine serves flat and terrain; flat uses stored zero f_1, f_2.

### 3.2 Gradient and divergence (no mixed derivatives)

- **Gradient:** Single routine: comp_x = `jax_dx(phi, dx) - f_1 * jax_dy(phi, dy)`, comp_y = `jax_dy(phi, dy)`, with `f_1 = geometry.f_1_grid()` broadcast to grid. Flat: f_1 is zeros → Cartesian.
- **Divergence:** Single routine: `jax_dx(u, dx) - f_1 * jax_dy(u, dy) + jax_dy(v, dy)` with `f_1 = geometry.f_1_grid()`. Flat: f_1 is zeros → Cartesian.

No \( \partial_{x\eta} \) here; only first derivatives and the coefficient \( f' \).

### 3.3 Wiring geometry into the grid

- **Grid:** Data live on \( (x_i, \eta_j) \) with \( x_i = i\,\Delta x \), \( \eta_j = j\,\Delta\eta \). So `dy` in the code is \( \Delta\eta \).
- **Metric arrays:** Geometry always stores f_1 and f_2 after computation. Use `geometry.f_1_grid()` and `geometry.f_2_grid()` (shape (Nx,)); broadcast to (Nx, Ny) inside the operator. No recomputation.
- **Flat case:** `Geometry.flat(Nx, Ny)` stores zero arrays for _f, _f_1, _f_2. Operators always use the same formula; the f_1 and f_2 terms cancel when they are zero.

### 3.4 Order of introduction (recommended)

1. **Gradient (terrain):** Implement terrain gradient; use it in one place (e.g. phase field or surface tension) and compare with flat for \( f \equiv 0 \).
2. **Divergence (terrain):** Implement terrain divergence; plug into continuity check and PPE RHS; again compare with flat.
3. **Laplacian (terrain), including mixed derivative:** Implement \( \nabla^2 \) with \( \partial_{x\eta} \) term; use in momentum (viscous), phase field (Cahn–Hilliard), and any pressure Poisson that uses a Laplacian matrix. This is the step that actually introduces the mixed-derivative stencil.
4. **Curvature / ST:** Curvature is \( \nabla\cdot(\nabla\phi/|\nabla\phi|) \); use terrain gradient for \( \nabla\phi \) and terrain divergence for the outer divergence.
5. **Pressure Poisson matrix:** If the solver is matrix-based, build the 2D Laplacian matrix with the terrain stencil (variable coefficients and cross-term \( \partial_{x\eta} \)); this is the most invasive change.

---

## 4. File-Level Checklist (Where to Touch)

| Location | Change |
|----------|--------|
| `numerics/finite_differences.py` (or new `numerics/terrain_operators.py`) | Single gradient/divergence/Laplacian that take geometry; use `geometry.f_1_grid()`, `geometry.f_2_grid()`. Flat geometry (zeros) recovers Cartesian. Laplacian is the only one with \( \partial_{x\eta} \). |
| `simulation/geometry.py` | Always stores _f, _f_1, _f_2 (flat = zeros). Exposes `f_1_grid()`, `f_2_grid()` so diff operators use stored arrays and never recompute. |
| `physics/fluid_dynamics.py` | When geometry is non-flat: use terrain gradient for convection and pressure gradient; use terrain Laplacian for viscous term; use terrain divergence for continuity. |
| `physics/phase_field.py` | When non-flat: use terrain gradient in advection and in \( \mu \); use terrain Laplacian in \( \mu \) and in \( \Delta\mu \). |
| `physics/surface_tension.py` | When non-flat: curvature = terrain divergence of (terrain gradient of \( \phi \) normalized); ST force uses that gradient. |
| `physics/pressure.py` | When non-flat: divergence of ST uses terrain divergence. |
| `solvers/ppe_utils.py` | When non-flat: RHS uses terrain divergence of \( \mathbf{u}^* \); gradient of \( p' \) uses terrain gradient. |
| `solvers/sparse_solver.py` (or new terrain Poisson builder) | When non-flat: build Poisson matrix from terrain Laplacian stencil (coefficients and \( \partial_{x\eta} \) term). |
| `boundary_conditions/contact_angle_bc.py` | When non-flat: use terrain gradient for \( \nabla\phi \) in contact angle condition. |
| Config / state | Add a way to select “terrain” mode (e.g. `geometry.has_geometry` from `Geometry.from_height(...)`) and pass geometry (or `f_1`, `f_2` on grid) into the above. |

---

## 5. Concrete operator changes (code)

All operators take `geometry` (e.g. `Geometry` from `simulation.geometry`). Grid arrays `f_1`, `f_2` are `geometry.f_1_grid`, `geometry.f_2_grid` — shape `(Nx, Ny)`, already stored. **Leave `jax_dx` and `jax_dy` unchanged**; they implement \( \partial_x|_\eta \) and \( \partial_\eta \).

### 5.1 Gradient

**Current (Cartesian):**
```python
grad_x = jax_dx(f, dx)
grad_y = jax_dy(f, dy)
return jnp.stack([grad_x, grad_y], axis=-1)
```

**Terrain (single code path; flat = zeros):**
```python
f_1 = geometry.f_1_grid   # (Nx, Ny)
phi_x = jax_dx(f, dx)
phi_eta = jax_dy(f, dy)
grad_x = phi_x - f_1 * phi_eta
grad_y = phi_eta
return jnp.stack([grad_x, grad_y], axis=-1)
```

**Signature:** `jax_gradient(f, dx, dy, geometry)` — always pass geometry (use `Geometry.flat(Nx, Ny)` when flat).

---

### 5.2 Divergence

**Current (Cartesian):**  
`div = jax_dx(u, dx) + jax_dy(v, dy)` (with optional solid_mask logic).

**Terrain:**
```python
f_1 = geometry.f_1_grid   # (Nx, Ny)
u_x = jax_dx(u, dx)
u_eta = jax_dy(u, dy)
v_eta = jax_dy(v, dy)
div = u_x - f_1 * u_eta + v_eta
```

**Signature:** `jax_divergence(f, dx, dy, geometry, solid_mask=None)` — add `geometry`; keep solid_mask for BCs if needed.

---

### 5.3 Laplacian

**Current (Cartesian):**  
`lap = phi_xx + phi_yy` with second derivatives via pad/roll or `jax_dx(jax_dx(...))`, `jax_dy(jax_dy(...))`.

**Terrain:**
\[
\nabla^2\phi = \phi_{xx} - 2f'\,\phi_{x\eta} - f''\,\phi_\eta + (1+(f')^2)\,\phi_{\eta\eta}.
\]

```python
f_1 = geometry.f_1_grid   # (Nx, Ny)
f_2 = geometry.f_2_grid   # (Nx, Ny)

phi_xx = jax_dx(jax_dx(phi, dx), dx)
phi_x_eta = jax_dx(jax_dy(phi, dy), dx)   # or jax_dy(jax_dx(phi, dx), dy)
phi_eta = jax_dy(phi, dy)
phi_eta_eta = jax_dy(jax_dy(phi, dy), dy)

lap = phi_xx - 2.0 * f_1 * phi_x_eta - f_2 * phi_eta + (1.0 + f_1**2) * phi_eta_eta
```

**Signature:** `jax_laplacian(f, dx, dy, geometry)`.

---

### 5.4 Call sites

- **Physics / solvers:** Replace `jax_gradient(phi, dx, dy)` with `jax_gradient(phi, dx, dy, geometry)` (and similarly for divergence and Laplacian). Pass `geometry` from state or config; use `Geometry.flat(Nx, Ny)` when the domain is flat so no API branch by “flat vs terrain” — same call, geometry carries the zeros.
- **Curvature / ST:** Curvature = divergence of (gradient of φ normalized). Use terrain gradient then terrain divergence, both with the same `geometry`.

---

## 6. Summary

- **Mixed derivatives** appear only in the **Laplacian**, as the term \( -2f'\,\partial_{x\eta}\phi \) (plus \( -f''\,\partial_\eta\phi \) and \( (1+(f')^2)\partial_{\eta\eta}\phi \)).
- **Gradient** and **divergence** use only first derivatives and the coefficient \( f' \); no \( \partial_{x\eta} \) there.
- **Where:** Implement terrain gradient, divergence, and Laplacian (with mixed derivative) in numerics; then switch physics and solvers to use them when geometry is non-flat. The Laplacian is the only operator that needs an explicit mixed-derivative stencil \( \partial_{x\eta} \).
- **How:** Use `geometry.f_1_grid` and `geometry.f_2_grid` (properties, shape `(Nx, Ny)`). Pass geometry into gradient/divergence/Laplacian; combine `jax_dx` and `jax_dy` with these coefficients; for Laplacian, add the cross-term with coefficient \( -2f' \). Flat: f_1 and f_2 are zero arrays, so terrain terms cancel and we get the usual Cartesian operators.
