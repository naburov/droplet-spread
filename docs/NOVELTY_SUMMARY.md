# Novelty Summary: droplet_spreading_modeling Codebase

Based on exploration of the code and docs, the following **novel or distinctive** elements can be identified for the article/thesis.

---

## 1. **Cox–Voinov dynamic contact angle with phase field**

- **What:** Effective contact angle depends on contact-line velocity:  
  **θ_eff = θ₀ + C · sign(U_cl) · |U_cl|^n** (n = 1/3), with U_cl taken as the tangential velocity just above the wall (no-slip) or at the wall (slip).
- **Where:** `contact_angle_bc.py` (`_get_effective_contact_angle_jax`, Cox–Voinov branch), config `use_cox_voinov`, `cox_voinov_coefficient`, `cox_voinov_exponent`.
- **Novelty:** Coupling of a Cox–Voinov–type dynamic contact angle law to a **phase-field** (Cahn–Hilliard) model with contact-angle BCs on φ. Many phase-field codes use only a static contact angle; here the moving contact line is explicitly linked to the effective angle via U_cl.
- **Ref for article:** Standard Cox–Voinov scaling; your implementation is the coupling to the diffuse-interface formulation and the way U_cl is taken from the velocity field (j=1 for no-slip, j=0 for slip).

---

## 2. **Contact-angle BC with advection-compatible blending**

- **What:** Contact angle is enforced by **blending** the advected φ at the wall with the value that would satisfy ∂φ/∂n = −cos(θ)|∇φ| (α = 0.5), instead of overwriting the boundary.
- **Where:** `contact_angle_bc.py`: “CRITICAL: Preserve advection while enforcing contact angle” — `phi_contact_blended = (1−α)*phi_advected + α*phi_contact_target`.
- **Novelty:** Avoids pinning the contact line: the interface can move along the wall while the effective angle is still enforced. Many implementations either pin the line or do not couple advection and contact angle in this way.
- **Docs:** `interface_at_solid_surface.md` describes the design.

---

## 3. **Curvilinear (terrain-following) geometry in phase-field + NS**

- **What:** Bottom surface is **y = f(x)**. Gradient, divergence, and Laplacian are implemented in terrain-following coordinates (e.g. gradient: (φ_x − f' φ_η, φ_η); Laplacian: full expression with f', f'').
- **Where:** `geometry.py` (Geometry, FlatGeometry, etc.), `finite_differences.py` (`jax_gradient`, `jax_divergence`, `jax_laplacian` with `f_1_grid`, `f_2_grid`), velocity BCs in `staggered.py` (top normal n = (−f', 1)).
- **Novelty:** Phase-field droplet/NS solver with **non-flat substrate** and consistent differential operators and BCs in one formulation. Contact angle and normal are defined relative to the surface normal from f(x) (and optionally from an ice phase field).
- **Use case:** Tilted/sloped walls, later ice geometry (y = h from ice phase ψ).

---

## 4. **Geometry-aware contact angle and contact-line velocity**

- **What:** For non-flat surfaces, contact angle is applied relative to the **surface normal** (from f'(x) or from ∇ψ for ice). Contact-line velocity for Cox–Voinov is the **tangential** component along the surface: (u + v f')/√(1+f'²).
- **Where:** `contact_angle_bc.py`: `_geometry_aware_contact_angle_jax`, `_get_effective_contact_angle_jax` with `contact_line_velocity` for curved surface.
- **Novelty:** Correct geometric definition of angle and U_cl on slopes; consistent with curvilinear operators.

---

## 5. **Cahn–Hilliard–Willmore (optional 4th-order regularization)**

- **What:** Optional Willmore-type term **μ_W = ε_W Δ²φ** in the chemical potential, so the phase-field equation includes a 4th-order smoothing term derived from a Willmore energy (simplified from the full 6th-order term).
- **Where:** `physics/phase_field.py` (`jax_willmore_chemical_potential`, `jax_update_phase` with `lambda_willmore`, `epsilon_willmore`), `CAHN_HILLIARD_WILLMORE.md`.
- **Novelty:** Energy-based regularization in the phase-field equation to control high curvature while keeping mass conservation and compatibility with contact-angle BCs. Configurable (λ_W = 0 turns it off).

---

## 6. **PPE boundary conditions derived from velocity BCs**

- **What:** Boundary conditions for the **pressure correction** (PPE) are **derived from velocity BCs**, not set independently: Dirichlet inlet → Neumann for p'; outlet opposite to inlet → Dirichlet p' = 0 to fix the pressure level; no-slip/Neumann velocity → Neumann for p'.
- **Where:** `solvers/ppe_bc_derivation.py`, `ppe_utils.py` (compatibility “Fix B” for inlet).
- **Novelty:** Clear, consistent link between velocity BCs and PPE, with documentation; avoids ad hoc pressure BC choices.

---

## 7. **Upstream flow: Blasius-type inlet + droplet with Cox–Voinov**

- **What:** Full configuration with **left inlet** (Blasius-type boundary-layer profile u(y) ∝ tanh(2.5 η), η = y/δ, δ ∝ Ly/Re^0.9), **right** do-nothing velocity + Dirichlet pressure, and a **droplet on the plate** with **Cox–Voinov** dynamic contact angle.
- **Where:** `config_upstream_flow_cox_voinov.md`, configs `config_upstream_flow_cox_voinov*.json`, `velocity_bc/collocated.py` (`_dirichlet_vel_boundary_layer`, blasius).
- **Novelty:** Combined setup: boundary-layer inflow + moving contact line with dynamic angle in a phase-field solver. Good candidate for “droplet in cross-flow” or “shear-driven contact line” studies.

---

## 8. **Dual formulation: collocated (main) + staggered (MAC)**

- **What:** Two implementations: (1) **Collocated** (state.py, two_phase.py): all unknowns at cell centers, projection with optional Rhie–Chow (currently disabled), used by main.py and ice-water. (2) **Staggered MAC** (two_phase_staggered.py, staggered_mac.py): p and φ at centers, u/v on faces; predictor–corrector with a single PPE solve (pyamg).
- **Where:** `simulation/state.py`, `two_phase.py`, `two_phase_staggered.py`, `numerics/staggered_mac.py`.
- **Novelty:** Same physics (phase field + NS + contact angle) available in both formulations; staggered branch is the one aligned with the article’s MAC description and “one projection per step.”

---

## 9. **Ice–water phase transition extension**

- **What:** Optional second phase field (ice ψ) and temperature with latent heat; ice surface provides geometry (h from ψ) for contact angle and advection; ice-aware contact angle (θ interpolated between water and ice).
- **Where:** `simulation/ice_water.py`, `physics/ice_phase_field.py`, `temperature.py`, `contact_angle_bc.py` (use_ice_aware, geometry from ψ).
- **Novelty:** Phase-field droplet code extended to **freezing/melting** on a substrate with consistent geometry and contact angle.

---

## 10. **Semicircle-on-wall initial condition**

- **What:** Droplet initialized as a **semicircle** on the wall (circle centered at (x_c, 0) in (x, η), so only the upper half is in the domain). Standard for “droplet on plate” without artificial cut.
- **Where:** `initial_conditions.py` (`initialize_phase` with `droplet_center_y = 0`).
- **Novelty:** Minor but correct: initial condition matches the intended geometry (droplet sitting on the surface).

---

## Suggested “novelty” wording for the article

- **Core (for drop.tex):**  
  (1) **Cox–Voinov dynamic contact angle** in a phase-field setting (θ_eff = θ₀ + C|U_cl|^n with U_cl from the velocity field).  
  (2) **Advection-compatible contact-angle BC** (blending so the contact line can move while enforcing the angle).  
  (3) **Optional Cahn–Hilliard–Willmore** 4th-order regularization.  
  (4) **Staggered (MAC) + single projection per step** for the droplet-without-upstream case.

- **Extended (for thesis / future work):**  
  Curvilinear geometry y = f(x), geometry-aware contact angle and U_cl, PPE BCs derived from velocity BCs, Blasius-type inlet + droplet with Cox–Voinov, ice–water phase transition.

If you tell me which of these you want to stress in the article (e.g. only droplet without upstream), I can suggest a single short “novelty” paragraph for drop.tex.
