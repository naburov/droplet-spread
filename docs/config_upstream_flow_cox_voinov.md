# System and Boundary Conditions: `config_upstream_flow_cox_voinov.json`

This document outlines the physical system and boundary conditions for the upstream-flow Cox–Voinov configuration. The setup models **natural flow development**: a boundary-layer velocity profile at the left inlet, outlet pressure at the right, a stationary plate at the bottom, and a droplet on the plate experiencing cross-flow with a **Cox–Voinov** dynamic contact angle.

---

## Governing equations

All variables are non-dimensional. **u** = (u, v) is velocity, p pressure, φ phase field (φ = −1 liquid, φ = +1 gas), ρ density, Re and We phase-dependent (interpolated from Re₁, Re₂ and We₁, We₂).

### Navier–Stokes (momentum)

\[
\rho \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right)
= -\nabla p + \frac{1}{\mathrm{Re}} \nabla^2 \mathbf{u}
- \mathbf{F}_{\mathrm{ST}} + \frac{\rho}{Fr}\,\mathbf{g}.
\]

- **ρ**: interpolated from phase field (ρ₁ in gas, ρ₂ in liquid).
- **Re**: harmonic interpolation in φ (Re₁, Re₂).
- **F_ST**: surface tension force (diffuse-interface form below).
- **g**: gravity vector (e.g. g = (0, −1) for downward).

### Incompressibility (continuity)

\[
\nabla \cdot \mathbf{u} = 0.
\]

### Cahn–Hilliard (phase field)

\[
\frac{\partial \phi}{\partial t} + \mathbf{u} \cdot \nabla \phi
= \frac{1}{\mathrm{Pe}} \Delta \mu,
\qquad
\mu = f'(\phi) - \varepsilon^2 \Delta \phi.
\]

- **f(φ)**: double-well \( f(\phi) = \tfrac{1}{4}(\phi^2 - 1)^2 \), so \( f'(\phi) = \phi(\phi^2 - 1) \).
- **Pe**: Peclet number (mobility).
- **ε**: interface thickness.

### Chemical potential

\[
\mu = f'(\phi) - \varepsilon^2 \Delta \phi
= \phi(\phi^2 - 1) - \varepsilon^2 \Delta \phi.
\]

### Curvature

\[
\kappa = \nabla \cdot \mathbf{n},
\qquad
\mathbf{n} = \frac{\nabla \phi}{|\nabla \phi|}.
\]

### Surface tension force

\[
\mathbf{F}_{\mathrm{ST}} = \frac{3\sqrt{2}\,\varepsilon}{4\,\mathrm{We}}\;
\kappa\, |\nabla \phi|\, \nabla \phi.
\]

- **We**: harmonic interpolation in φ (We₁, We₂).

### Pressure Poisson (correction step)

Predictor **u*** from momentum; then solve for pressure correction p′ and project:

\[
\Delta p' = \frac{\nabla \cdot \mathbf{u}^*}{\Delta t},
\qquad
\mathbf{u}^{n+1} = \mathbf{u}^* - \Delta t\,\frac{\nabla p'}{\rho}.
\]

(BCs for p′ are Neumann derived from velocity BCs; absolute pressure is then fixed by imposing outlet pressure.)

---

## Boundary conditions (equations)

### Pressure

| Boundary | BC | Equation |
|----------|----|----------|
| Top, bottom, left | Neumann | \( \partial p / \partial n = 0 \) |
| Right | Dirichlet | \( p = 0 \) (outlet reference) |

### Velocity

| Boundary | BC | Equation |
|----------|----|----------|
| Top | Neumann | \( \partial u / \partial n = 0,\ \partial v / \partial n = 0 \) |
| Bottom | No-slip | \( u = 0,\ v = 0 \) |
| Left | Dirichlet (Blasius-type) | \( u = u_{\mathrm{target}}\, \tanh(2.5\,\eta),\ v = 0 \), with \( \eta = y/\delta \), \( \delta = L_y / \mathrm{Re}_2^{0.9} \), and \( u(0) = 0 \) |
| Right | Do-nothing | Open outflow (no normal stress imposed) |

### Phase field φ

| Boundary | BC | Equation |
|----------|----|----------|
| Top, left, right | Neumann | \( \partial \phi / \partial n = 0 \) |
| Bottom | Contact angle + Cox–Voinov | \( \partial \phi / \partial n = -\cos\theta_{\mathrm{eff}}\, |\nabla \phi| \) with \( \theta_{\mathrm{eff}} = \theta_0 + C\,\mathrm{sign}(U_{\mathrm{cl}})\, |U_{\mathrm{cl}}|^n \) (radians), \( n = 1/3 \), U_cl = u just above wall |

### Chemical potential μ

| Boundary | BC | Equation |
|----------|----|----------|
| All | Zero flux | \( \partial \mu / \partial n = 0 \) |

### Advection (for φ update)

| Boundary | BC | Equation |
|----------|----|----------|
| Top | Open | \( \partial \phi / \partial t + c_{\mathrm{out}}\, \partial \phi / \partial n = 0 \) (radiation) |
| Bottom, left, right | Impermeable | No flux through boundary (φ from interior / contact angle at wall) |

---

## 1. Physical System

### 1.1 Domain and Grid

| Parameter | Value | Description |
|-----------|--------|-------------|
| **Lx, Ly** | 1.0, 1.0 | Domain size (non-dimensional) |
| **Nx, Ny** | 128, 128 | Grid resolution |

### 1.2 Physical Parameters

| Parameter | Value | Role |
|-----------|--------|------|
| **rho1** | 0.001225 | Air density (phase 1, φ = +1) |
| **rho2** | 1.0 | Water density (phase 2, φ = -1) |
| **Re1** | 35.0 | Reynolds number (air) |
| **Re2** | 500.0 | Reynolds number (water); used for Blasius boundary layer scale |
| **We1, We2** | 0.01, 2.0 | Weber numbers |
| **Pe** | 20.0 | Peclet number (phase field mobility) |
| **epsilon** | 0.04 | Interface thickness |
| **contact_angle** | 60° | Equilibrium contact angle (degrees) |
| **include_gravity** | true | Gravity enabled |
| **Fr** | 1.0 | Froude number |
| **g** | -1.0 | Gravitational acceleration (y) |
| **atm_pressure** | 0.0 | Reference pressure |

### 1.3 Initial Conditions

- **Type**: droplet on flat surface.
- **Droplet**: semicircle (circle cut by bottom wall), radius **0.15**, center **(0.5, 0.0)** (mid-x, on the wall).
- **Phase field**: φ = tanh((r − R)/(ε√2)) with r distance from droplet center; φ &lt; 0 liquid, φ &gt; 0 air.
- **Velocity**: zero initially; inlet profile is applied by boundary conditions.
- **Pressure**: hydrostatic (if gravity) plus a **linear dynamic gradient** from inlet to outlet:
  - p(x) = p_outlet + **pressure_drop** × (1 − x/Lx), with **pressure_drop** = 0.01 and **p_outlet** = 0 (right Dirichlet value).

Code: `src/simulation/initial_conditions.py` (flat case with `droplet_center_y = 0`), `src/simulation/state.py` (`initialize_pressure_field` and flow-simulation branch).

---

## 2. Boundary Conditions Summary

| Boundary | Pressure | Velocity | Phase field | Chemical potential | Advection |
|----------|----------|----------|-------------|---------------------|-----------|
| **Top** | Neumann | Neumann | Neumann | Zero flux | Open |
| **Bottom** | Neumann | No-slip | Contact angle (+ Cox–Voinov) | Zero flux | Impermeable |
| **Left** | Neumann | Dirichlet (Blasius profile) | Neumann | Zero flux | Impermeable |
| **Right** | Dirichlet (0) | Do-nothing | Neumann | Zero flux | Impermeable |

Details below.

---

## 3. Pressure Boundary Conditions

- **Top, bottom, left**: **Neumann** (∂p/∂n = 0). Natural for channel flow; no imposed pressure gradient on these sides.
- **Right**: **Dirichlet** p = **0.0** (outlet pressure). Provides a reference and drives the mean pressure gradient from left to right.

Pressure BCs are applied in `src/boundary_conditions/pressure_bc.py`. The PPE (pressure Poisson equation) for the **correction** p′ uses BCs **derived from velocity BCs** (see `src/solvers/ppe_bc_derivation.py`): all sides end up with Neumann for p′; the Dirichlet pressure is used for the **hydrodynamic pressure field** P (e.g. after solve, offset so that the right boundary equals the configured Dirichlet value).

---

## 4. Velocity Boundary Conditions

- **Top**: **Neumann** — ∂u/∂n = 0, ∂v/∂n = 0 (free development).
- **Bottom**: **No-slip** — u = 0, v = 0 at the wall (stationary plate).
- **Left**: **Dirichlet** with a **Blasius-type boundary layer profile**:
  - Base scale: u_target = 1.0, v = 0.0 from `dirichlet_values.left`.
  - Profile type: `boundary_layer`, subtype **blasius**.
  - Boundary layer thickness: δ = L_char / Re2^bl_exponent with **characteristic_length** = Ly, **reynolds_number** = Re2, **bl_exponent** = 0.9 → δ ∝ Ly/Re2^0.9 (Blasius-like scaling).
  - Velocity: u(y) = u_target × tanh(2.5 η) for η = y/δ &lt; 3, else u_target; u(0) = 0.
- **Right**: **Do-nothing** — open outflow; no normal stress constraint so the flow can leave naturally.

Implementation: `src/boundary_conditions/velocity_bc/collocated.py` (`_dirichlet_vel_boundary_layer`, Blasius branch, and inlet profile for PPE compatibility). The **PPE** uses Neumann for p′ on all sides; for the left Dirichlet velocity, a **compatibility term** is applied so the pressure correction is consistent with the prescribed inlet velocity (`src/solvers/ppe_utils.py`, “Fix B”).

---

## 5. Phase Field Boundary Conditions

- **Top, left, right**: **Neumann** — ∂φ/∂n = 0 (no flux of order parameter).
- **Bottom**: **Contact angle** with **Cox–Voinov**:
  - **contact_angle_method**: `"simple"`.
  - **use_cox_voinov**: true.
  - **cox_voinov_coefficient**: 1.0 (C in θ = θ₀ ± C|U|^n).
  - **cox_voinov_exponent**: 0.333 (n ≈ 1/3).

**Simple contact angle**: normal derivative at the wall set from the effective angle, ∂nφ = −cos(θ_eff)|∇φ|, with blending so that advection and contact-line motion are preserved.

**Cox–Voinov**: effective angle depends on **contact-line velocity** U_cl (x-component of velocity **just above** the wall, U[:, 1, 0], since no-slip makes U[:, 0, 0] = 0):
- θ_eff = θ₀ + C × sign(U_cl) × |U_cl|^n (radians; C converted from degrees in code).
- Advancing (U_cl &gt; 0): angle increases; receding (U_cl &lt; 0): angle decreases.

Implementation: `src/boundary_conditions/contact_angle_bc.py` (`_simple_contact_angle_jax`, `_get_effective_contact_angle_jax` with Cox–Voinov branch); phase BC wrapper in `src/boundary_conditions/phase_field_bc.py` passes `use_cox_voinov` and parameters.

---

## 6. Chemical Potential Boundary Conditions

- **All sides**: **Zero flux** — ∂μ/∂n = 0. Consistent with Cahn–Hilliard (no diffusive flux of φ through boundaries). Implemented in `src/boundary_conditions/chemical_potential_bc.py`.

---

## 7. Advection Boundary Conditions (Phase Field)

Used when updating φ so that inflow/outflow and the solid wall are treated correctly:

- **Top**: **Open** — radiation-style BC ∂φ/∂t + c_out ∂nφ = 0 so the interface can leave the domain.
- **Bottom**: **Impermeable** — no flux through the wall; at y = 0, φ is set from interior or from wall treatment (e.g. contact angle).
- **Left, right**: **Impermeable** — no normal advective flux through vertical sides (flow is along x; inlet/outlet are handled by velocity and pressure, not by advective “open” φ here).

Implementation: `src/boundary_conditions/advection_bc.py` (`_impermeable`, `_open_radiation`).

---

## 8. PPE (Pressure Poisson Equation) and Uniqueness

- PPE BCs for the **correction** p′ are **derived from velocity BCs** (`src/solvers/ppe_bc_derivation.py`):
  - Dirichlet velocity (left) → Neumann for p′ (with compatibility term in RHS/application).
  - No-slip (bottom) → Neumann for p′.
  - Do-nothing (right) → Neumann for p′.
  - Neumann (top) → Neumann for p′.
- So the discrete Laplacian has a null space (constant). Uniqueness is handled by filtering/removing the checkerboard mode and/or pinning (see `src/solvers/ppe_utils.py`, `ppe_filtering` / `ppe_uniqueness`). The **absolute** pressure level is then set so that the right boundary matches the Dirichlet pressure (0.0).

---

## 9. Solver and Numerical Settings (Relevant to BCs)

- **Pressure/Correction solver**: pyamg, BiCGSTAB, tol 0.05, maxiter 1000.
- **use_rhie_chow**: false (collocated grid).
- **ppe**: mean_div_threshold 0.15, max_div_threshold 0.25, use_local_ppe false.

---

## 10. Flow Summary

1. **Left**: Blasius-like inlet profile (u(y), v=0), no-slip at y=0; drives channel flow.
2. **Right**: Do-nothing velocity (open outflow), Dirichlet pressure p = 0.
3. **Bottom**: No-slip wall; droplet sits here with 60° equilibrium contact angle and **Cox–Voinov** dynamic angle (contact line can move with flow).
4. **Top**: Neumann velocity and pressure (stress-free, natural development).

The droplet at (0.5, 0) experiences cross-flow from the left, with contact angle responding to contact-line motion via the Cox–Voinov law (θ = θ₀ ± C|U_cl|^0.333).
