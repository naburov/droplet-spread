# Config audit: `contact_angle_60.json`

Section-by-section check that each part of the config is read and correctly implemented in the codebase.

---

## 1. `physical_params`

| Key | Config value | Where used | Status |
|-----|--------------|------------|--------|
| `rho1`, `rho2` | 0.001225, 1 | `state.py` → state, `FluidDynamicsSolver`, `PressureSolver`, density | OK |
| `Re1`, `Re2` | 500, 100 | `state.py` → state, `FluidDynamicsSolver` | OK |
| `We1`, `We2` | 100, 0.01 | `state.py` → state, `SurfaceTensionSolver` | OK |
| `Pe` | 500 | `state.py` → state, `PhaseFieldSolver` | OK |
| `epsilon` | 0.02 | `state.py` → state, phase + ST solvers, initial_conditions | OK |
| `contact_angle` | 60 | `state.py` → state, `PhaseFieldSolver`, `SurfaceTensionSolver`, `PhaseFieldBoundaryConditions` → `ContactAngleBoundaryCondition` | OK |
| `include_gravity` | true | `state.py` → state, `PressureSolver` | OK |
| `Fr` | 0.8 | `state.py` → state, `PressureSolver`, `FluidDynamicsSolver` | OK |
| `g` | -1 | `state.py` → state, pressure/fluid | OK |
| `atm_pressure` | 0 | `state.py` → state, `PressureSolver`, pressure BC | OK |
| `lambda_willmore` | 0 | `physics/phase_field.py`: `PhaseFieldSolver(..., config)` reads `physical_params.lambda_willmore` (0 = disabled) | OK |
| `epsilon_willmore` | 0 | Same; `physical_params.epsilon_willmore` | OK |

**Verdict:** All keys implemented. Willmore is disabled (0, 0) as intended.

---

## 2. `grid_params`

| Key | Config value | Where used | Status |
|-----|--------------|------------|--------|
| `Lx`, `Ly` | 1, 1 | `state.py` → geometry, state | OK |
| `Nx`, `Ny` | 64, 64 | `state.py` → grid, solvers, geometry | OK |

**Verdict:** Implemented.

---

## 3. `time_params`

| Key | Config value | Where used | Status |
|-----|--------------|------------|--------|
| `dt` | 0.0001 | `state.py` (dt), `base.py` as `dt_normal` | OK |
| `dt_initial` | 0.0005 | `state.py`, `base.py` | OK |
| `t_max` | 2 | `base.py` | OK |
| `checkpoint_interval` | 10 | `base.py` (checkpoints, viz), `two_phase.py` (PPE debug) | OK |
| `cfl_number` | 0.1 | `base.py` (CFL dt limit) | OK |
| `capillary_cfl_number` | 10 | `base.py` | OK |
| `curvature_cfl_number` | 10 | `two_phase.py` → `curvature_cfl_dt()` for dt limiting | OK |

**Verdict:** All used.

---

## 4. `initial_conditions`

| Key | Config value | Where used | Status |
|-----|--------------|------------|--------|
| `type` | "droplet" | `state.py`: `ic.get("type", "droplet")` → branch droplet / rectangle / uniform_air / uniform_liquid | OK |
| `droplet_radius` | 0.15 | `state.py` → `initialize_phase(Nx, Ny, radius, ...)` via `ic.get("droplet_radius", 0.2)` | OK |
| `droplet_center_x` | 0.5 | `initial_conditions.py`: `initialize_phase` → `ic.get("droplet_center_x", 0.5)` | OK |
| `droplet_center_y` | 0 | `initial_conditions.py`: `ic.get("droplet_center_y", 0.0)` (semicircle on surface) | OK |

**Verdict:** Implemented. Semicircle droplet at (0.5, 0) with radius 0.15.

---

## 5. `boundary_conditions`

### 5.1 `pressure`

| Key | Config | Where used | Status |
|-----|--------|------------|--------|
| `top` | "open" | `pressure_bc.py`, `state.py` (pressure_linear_solver BC), sparse_solver | OK |
| `bottom`, `left`, `right` | "neumann" | Same | OK |
| `open_pressure` | 0 | `pressure_bc.py`: `bc_cfg.get("open_pressure", 0.0)` for Dirichlet value at "open" sides | OK |

**Verdict:** Correct. Pressure pinned at top (atmosphere) with value 0.

### 5.2 `velocity`

| Key | Config | Where used | Status |
|-----|--------|------------|--------|
| `top` | "dirichlet" | `velocity_bc` (staggered/collocated): top u,v from dirichlet_values | OK |
| `bottom` | "navier_slip" | Staggered: v=0, u = u_int * λ/(λ+dy); `slip_length` used | OK |
| `left`, `right` | "neumann" | Treated as do_nothing in staggered (extrapolate); fixed in codebase | OK |
| `slip_length` | 1000 | `velocity_bc` (collocated/staggered): `bc_cfg.get("slip_length", 0.01)` | OK |
| `dirichlet_values` | top/left/right u=0,v=0 | Used for top (left/right unused when BC is neumann) | OK |

**Verdict:** Implemented. Left/right Neumann implemented as zero-stress (do_nothing) in velocity BCs.

### 5.3 `phase_field`

| Key | Config | Where used | Status |
|-----|--------|------------|--------|
| `top`, `left`, `right` | "neumann" | `phase_field_bc.py`: standard Neumann | OK |
| `bottom` | "contact_angle" | `phase_field_bc.py` → `ContactAngleBoundaryCondition.apply` | OK |
| `contact_angle_method` | "simple" | `phase_field_bc.py` → `ContactAngleBoundaryCondition(method=...)` | OK |
| `use_cox_voinov` | true | Same → `use_cox_voinov=True` | OK |
| `cox_voinov_coefficient` | 1 | Same | OK |
| `cox_voinov_exponent` | 0.333 | Same | OK |

**Verdict:** All used. Contact angle 60° + Cox–Voinov with C=1, n≈1/3.

### 5.4 `chemical_potential`

| Key | Config | Where used | Status |
|-----|--------|------------|--------|
| `top`, `bottom`, `left`, `right` | "zero_flux" | `chemical_potential_bc.py`: base maps "zero_flux" → NEUMANN; `apply_standard_scalar` applies Neumann | OK |

**Verdict:** Zero flux on all sides.

### 5.5 `advection`

| Key | Config | Where used | Status |
|-----|--------|------------|--------|
| `top`, `bottom`, `left`, `right` | "impermeable" | `advection_bc.py`: `_impermeable_*` for each side (no flux through boundary) | OK |
| `cout` | 1 | `advection_bc.py`: `bc_cfg.get("cout", 1.0)` (used only for "open" radiation) | OK (unused when all impermeable) |

**Verdict:** All sides impermeable; `cout` only matters if a side is "open".

---

## 6. `solver_params`

### 6.1 `pressure_solver` & `correction_solver`

| Key | Config | Where used | Status |
|-----|--------|------------|--------|
| `backend` | "pyamg" | `state.py` → `SparseSolverWrapper(..., backend=...)` | OK |
| `accel` | "bicgstab" | `sparse_solver.py`: `params.get('accel', 'bicgstab')` | OK |
| `tol` | 0.01 | `params.get('tol', 0.1)` | OK |
| `maxiter` | 1000 | `params.get('maxiter', 1000)` | OK |

**Verdict:** Used for both pressure and correction (PPE) solvers.

### 6.2 `velocity_layout`

| Key | Config | Where used | Status |
|-----|--------|------------|--------|
| `velocity_layout` | "staggered" | `state.py`, `velocity_bc/__init__.py` → `StaggeredVelocityBoundaryConditions`, PPE path in `two_phase.py` | OK |

**Verdict:** Staggered layout and staggered PPE path used.

### 6.3 `ppe`

| Key | Config | Where used | Status |
|-----|--------|------------|--------|
| `mean_div_threshold` | 0.01 | `base.py` → `ppe_params.get("mean_div_threshold", 0.1)`; passed to `ppe_solve` / `ppe_solve_staggered` | OK |
| `max_div_threshold` | 0.15 | `base.py`; same | OK |
| `div_threshold` | 0.01 | `base.py`; same | OK |
| `max_iterations` | 5000 | `two_phase.py`: `ppe_settings.get('max_iterations', 1000)` → PPE loop | OK |
| `under_relaxation` | 1 | `two_phase.py`: `ppe_settings.get('under_relaxation', 1.0)` → correction step | OK |
| `boundary_conditions` | top=dirichlet, rest neumann | `state.py` (explicit PPE BCs), `correction_solver.set_*_boundary_condition` | OK |

**Verdict:** All used. Pressure correction pinned at top (Dirichlet), others Neumann.

---

## 7. `visualization`

| Key | Config | Where used | Status |
|-----|--------|------------|--------|
| `use_pyvista` | false | `base.py`: `config.get("visualization", {}).get("use_pyvista", True)` → backend choice (matplotlib vs PyVista) | OK |

**Verdict:** Matplotlib used when false (default in code is True; config overrides to false).

---

## 8. `restart`

| Key | Config | Where used | Status |
|-----|--------|------------|--------|
| `restart_from` | null | `base.py`: `config["restart"]["restart_from"]`; `None if restart_from == "None" else restart_from`; `SimulationState.from_config(..., restart_from=...)` | OK |

**Verdict:** No restart; fresh run.

---

## 9. Top-level `description`

Not read by the code; for humans only. No change needed.

---

## Summary

| Section | Status | Notes |
|---------|--------|--------|
| physical_params | OK | All keys used; Willmore off |
| grid_params | OK | 64×64, Lx=Ly=1 |
| time_params | OK | dt, CFL, curvature CFL used |
| initial_conditions | OK | droplet, r=0.15, center (0.5,0) |
| boundary_conditions | OK | Pressure open at top; velocity top Dirichlet, bottom navier_slip, left/right Neumann; phase bottom contact_angle + Cox–Voinov; chemical zero_flux; advection impermeable |
| solver_params | OK | Staggered, pyamg/bicgstab, PPE thresholds and explicit BCs |
| visualization | OK | use_pyvista=false |
| restart | OK | null → no restart |

No missing or unimplemented keys found. One code fix was applied earlier: velocity BC "neumann" on left/right is now implemented (as do_nothing) in staggered and collocated velocity BCs; and phase_field `bottom` was added explicitly as `"contact_angle"` in the config.
