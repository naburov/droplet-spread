# Boundary conditions audit

Where each BC is applied and how they must be consistent.

## 1. Velocity BC

**Config key:** `boundary_conditions.velocity`  
**Applied in:** `velocity_bc.apply_boundary_conditions()`  
**When:** After predictor step, after PPE correction, and in PPE when using staggered layout.

**Per-boundary types:**
- `no_slip`: U = 0
- `do_nothing`: copy from interior (outflow)
- `dirichlet`: prescribed (u, v) or profile (e.g. inlet)
- `slip_symmetry`: zero normal velocity, zero tangential gradient

**Poiseuille config:** top/bottom = no_slip, left = dirichlet (u=1, profile=linear), right = do_nothing.

**Consistency:** PPE BCs are derived from velocity BCs (Neumann for p′ at no_slip/dirichlet/do_nothing). Pressure solver uses `boundary_conditions.pressure` (separate from PPE).

---

## 2. Pressure BC

**Config key:** `boundary_conditions.pressure`  
**Used by:** Pressure linear solver (for full pressure P), not for pressure correction p′.

**Poiseuille config:** top/bottom/left = neumann, right = dirichlet (0).

---

## 3. PPE (pressure correction) BCs

**Source:** Derived from velocity BCs in `derive_ppe_bcs_from_config()` (all Neumann for p′ in standard setup), or set explicitly in `solver_params.ppe.boundary_conditions`.

**Used by:** Correction solver (Poisson for p′).  
**Consistency:** Must match velocity BCs (Neumann at no_slip/dirichlet/do_nothing). Inlet compatibility term in `solve_pressure_correction` uses U_star and inlet profile when left = dirichlet.

---

## 4. Phase field BC

**Config key:** `boundary_conditions.phase_field`  
**Applied in:** `phase_field_bc.apply_boundary_conditions()` after phase update.

**Per-boundary:** neumann, contact_angle (bottom only), etc.  
**Poiseuille config:** top/left/right = neumann, bottom = contact_angle.

**Contact angle:** Only modifies φ where interface is present (phi near 0, |∇φ| > threshold). For uniform liquid (φ = −1), contact_mask is false → effective Neumann at bottom. No conflict.

---

## 5. Advection BC (phase field φ)

**Config key:** `boundary_conditions.advection`  
**Applied in:** `advection_bc_manager.apply_boundary_conditions(phi_new, U, dt, dx, dy)` inside `PhaseFieldSolver.update()`, **before** phase_field_bc.

**Implemented behaviour:**
- **bottom = "impermeable"**: `_impermeable(phi, U)` — sets φ at j=0 from interior or copy when velocity into wall (no-slip → keep/copy from interior).
- **"open"** (top/left/right): `_open_radiation(phi, b, dt, dx, dy)` — radiation BC ∂φ/∂n type.
- **"impermeable" at left/right/top:** Not implemented in the loop; only "open" is applied for non-bottom. So left/right = impermeable ⇒ **no advection BC applied** at those faces (φ left as from interior).

**Poiseuille config (before fix):** top = open, bottom = impermeable, left = impermeable, right = impermeable.  
**Issue:** For channel flow, left = inlet and right = outflow. Advection should allow φ to enter/leave: left and right should be **open**, not impermeable. Using impermeable at left/right means we never set φ there from an “inflow/outflow” perspective; the code doesn’t apply any advection BC there, so φ is whatever the interior update wrote. That can be acceptable but is clearer if aligned with velocity: **left = open (inflow), right = open (outflow)**.

**Recommendation:** For channel (inlet left, outflow right), set advection to:
- bottom = impermeable  
- top = open  
- left = open  
- right = open  

---

## 6. Chemical potential BC

**Config key:** `boundary_conditions.chemical_potential`  
**Applied in:** `jax_apply_chemical_potential_zero_flux_bc(mu_ch, dx, dy)` inside `jax_update_phase`.  
**Poiseuille config:** all zero_flux. No conflict.

---

## Order of application (one step)

1. Predictor: U* from momentum (velocity update).  
2. Velocity BC on U*.  
3. PPE: solve for p′, U ← U* − dt ∇p′.  
4. Velocity BC on U again (critical so inlet/no_slip/outflow are correct).  
5. Phase update: φ_new from Cahn–Hilliard; then **advection BC** on φ_new; then **phase_field BC** (contact_angle / neumann) on φ_new.  
6. Pressure update: P ← P + p′ (or full solve if used).

Any mismatch (e.g. velocity inlet but advection impermeable at inlet, or forgetting to re-apply velocity BC after PPE) can cause divergence blow-up or wrong streamlines.

---

## Tracing divergence explosion

If the solver converges for a while and then divergence explodes, check in order:

1. **After predictor:** div(U*) — should be O(1) or moderate before PPE.
2. **After PPE (before velocity BC re-apply):** div(U) — should be small if PPE converged.
3. **After velocity BC re-apply:** div(U) — must stay small; re-apply must not destroy solenoidality (e.g. do_nothing/copy is safe; Dirichlet at inlet is prescribed so div is then recomputed next step).
4. **After phase update:** φ and then next step’s predictor use this φ for ρ, ST, etc. Large ∇φ or bad φ at boundaries can drive large forces and then large div next step.
5. **Advection BC:** If advection BC writes φ at a boundary in a way that creates a large gradient (e.g. discontinuity), the next step’s chemical potential and surface tension can spike and cause instability.
6. **Staggered vs collocated:** With `velocity_layout: "staggered"`, velocity is stored on faces; conversion to/from collocated U for phase and BCs must be consistent. Check that velocity BC is applied to the correct representation and that div is computed from the same representation used in the PPE.
