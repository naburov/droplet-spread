# Config audit: used vs dead parameters

This document lists parameters that the codebase actually uses vs those that are **dead** (never read). Configs that use dead params are noted so you can clean them or keep them for future use.

---

## Parameters that are USED (main.py two-phase path)

### physical_params
- `rho1`, `rho2`, `Re1`, `Re2`, `We1`, `We2`, `Pe`, `epsilon`, `contact_angle`, `include_gravity`, `g`, `atm_pressure`, `Fr` — used in state/physics
- `include_ice_water_transition` — used for ice-water
- `lambda_willmore`, `epsilon_willmore` — used in phase_field.py (Willmore regularization)
- `surface_tension.smooth_curvature`, `surface_tension.smoothing_radius` — used in state

### grid_params
- `Lx`, `Ly`, `Nx`, `Ny` — used everywhere

### time_params
- `dt`, `t_max`, `checkpoint_interval`, `dt_initial` — used in base
- `cfl_number`, `capillary_cfl_number`, `curvature_cfl_number` — used in two_phase

### geometry (top-level, or under initial_conditions)
- `type` ("flat" | "tilted" | "hump"), `degree`, `origin`, `amplitude`, `sigma`, `center_x`, `center_y` — used in state for geometry

### initial_conditions
- `type`, `droplet_radius`, `droplet_center_x`, `droplet_center_y`, `is_bubble`, `phi_value`, `pressure_drop`, `initial_velocity.u/v` — used in state/initial_conditions
- `ice_phase.*`, `temperature.*` — used for ice/temp ICs

### boundary_conditions
- **pressure**: `top/bottom/left/right`, `open_pressure`, `dirichlet_values.{side}` (numeric or `{value}`), `use_geometry`
- **velocity**: `top/bottom/left/right`, `slip_length`, `slip_parameter`, `use_geometry`, `dirichlet_values.{side}` (`u`, `v`), `dirichlet_profiles.{side}.type` (inlet profile)
  - **Inlet profile** (`dirichlet_profiles.left.type`): Omit or no `dirichlet_profiles` → **constant** (u = target everywhere, slip at wall): strong flow near bottom. `"linear"` → ramp from 0 at y=0 to target at y=Ly/2 then flat: **near-bottom almost zero**. `"boundary_layer"` → BL shape (subtype: `blasius`, `power_law`, `exponential`; optional `bl_exponent`, `transition_scale`, `characteristic_length`): low u near wall. Use constant (omit profile) for more flow near the bottom.
- **phase_field**: `top/bottom/left/right`, `contact_angle_method`, `use_cox_voinov`, `cox_voinov_coefficient`, `cox_voinov_exponent`, `use_geometry_aware`
  - `contact_angle_method`: "simple" = flat surface (vertical normal); Cox-Voinov is still applied when `use_cox_voinov` is true (θ_effective from velocity). Use "geometry_aware" for tilted/non-flat surfaces so normal and contact-line velocity (tangential along slope) are correct.
- **advection**: `top/bottom/left/right`, `cout`, `velocity_threshold`
- **chemical_potential**, **ice_phase_field**, **temperature**: per-side BCs and `dirichlet_values`

### solver_params
- **pressure_solver** / **correction_solver**: `backend`, `accel`, `tol`, `maxiter`
- **ppe**: `boundary_conditions.{top,bottom,left,right}`, `mean_div_threshold`, `max_div_threshold`, `div_threshold`, `max_iterations`, `under_relaxation`
- **velocity_layout**: `"staggered"` | `"collocated"` (default)

### restart
- `restart_from` — path or null

### Optional (telemetry / comparison)
- `baseline_experiment` — used in base and plot_telemetry
- `ice_water_params` — used for ice simulations

---

## Parameters that are DEAD (not read by code)

These appear in defaults or in some configs but are **never used** at runtime. Safe to remove from configs or keep as documentation.

| Location | Key | Note |
|----------|-----|------|
| **physical_params** | `alpha` | In config_loader default only; no physics reads it. |
| **physical_params** | `phase_penalty` | In config_loader default only; no physics reads it. |
| **solver_params.ppe** | `use_local_ppe` | Only in state default; no PPE logic uses it (local PPE not implemented). |
| **solver_params.ppe** | `local_threshold_factor` | Only in state default; unused. |
| **solver_params.ppe** | `buffer_size` | Only in state default; unused. |

Configs that set `use_local_ppe: true` (e.g. `config_droplet_simple.json`) have **no effect**; the code always runs global PPE.

---

## Other config keys

- **description** — Not read; documentation only. OK to keep.
- **_bc_corner_compatibility** — Not read; comment/documentation. OK to keep.
- **accel** in pressure_solver / correction_solver — Used by sparse_solver (e.g. `bicgstab`). Valid.

---

## Per-config notes (two-phase configs)

| Config | Notes |
|--------|------|
| **config_template.json** | ppe.boundary_conditions has `top: dirichlet` — unusual; often outlet is left/right. Otherwise fine. |
| **config_droplet_simple.json** | Sets `use_local_ppe: true`, `local_threshold_factor`, `buffer_size` — all dead; no functional effect. |
| **config_upstream_flow_cox_voinov_staggered_64_faster.json** | pressure `right: dirichlet` + velocity `left: dirichlet` (inlet left, outlet right). ppe has no explicit boundary_conditions — derived from velocity (outlet right → PPE right dirichlet). OK. |
| **sliding_droplet_tilted.json** | Explicit ppe.boundary_conditions left=dirichlet (outlet). velocity_layout staggered. All used. |
| **config_droplet_ice.json** | If present, check `ice_water_params` and `include_ice_water_transition`; temperature/ice_phase_field BCs used. |
| **staggered_poiseuille.json** | Different schema for `run_staggered_flow.py` (single-phase). Uses `top_bc`, `bottom_bc`, `inlet_profile`, `outflow_right`, `nu`, `steps`, etc. Not for main.py. |

---

## Recommendations

1. **Remove or ignore dead params** in configs: `use_local_ppe`, `local_threshold_factor`, `buffer_size` in ppe; `phase_penalty`, `alpha` in physical_params if you copy from config_loader defaults.
2. **Collocated PPE** now reads `under_relaxation` from `solver_params.ppe` (same as staggered).
3. **dirichlet_values** for pressure: use a numeric value (e.g. `"left": 0.0`) or the code accepts it; base_bc and pressure_bc use `dirichlet_values.get(side, open_pressure)`.
4. **velocity dirichlet_profiles**: `type` can be `"linear"` (boundary layer) or similar; velocity_bc/collocated.py reads it for inlet profile.
