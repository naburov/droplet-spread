# Sliding Droplet Experiment Grid

## Inlet Model

The previous `boundary_layer` / `blasius` profile was a tanh ramp with a fitted thickness. It is useful as a numerical forcing baseline, but it is not a Blasius solution.

For the paper runs, use `subtype: "slip_blasius"`. It solves

```text
f''' + 0.5 f f'' = 0
f(0) = 0
f'(0) = K f''(0)
f'(inf) = 1
```

with `K = slip_length / delta` by default. This is the Navier-slip analogue of the slip-flow Blasius condition in Martin and Boyd, where the wall tangential velocity is proportional to the wall shear.

The code can also compute the Blasius wall-normal velocity via `include_normal_velocity: true`, but production runs currently keep it off because the top boundary still enforces `v = 0`. Turning it on requires a mixed far-field top boundary: `u = U_inf`, `dv/dy = 0` or an open normal-velocity treatment.

## Paper Figures

Goal 1: show the algorithm works.

Run a stable baseline with `ghost_cell + semi_implicit_ch + slip_blasius` at `128^2` and `256^2`, contact angles `60, 90, 120`, and slip lengths `0.005, 0.02, 0.05`. Report mass drift, center-of-mass trajectory, contact-line positions, velocity divergence, and snapshots.

Goal 2: show how numerical artifacts diverge.

Run controlled ablations against the baseline:

```text
tanh_blasius + simple CH
tanh_blasius + ghost explicit CH
slip_blasius + simple CH
slip_blasius + ghost semi-implicit CH
```

For each, report time to instability, mass drift, phi overshoot, interface width growth, and max boundary divergence. These are the artifacts we already observed locally.

Goal 3: show novelty.

Use paired runs:

```text
simple wall overwrite vs ghost-cell wetting
explicit CH vs semi-implicit CH
tanh inlet vs slip-Blasius inlet
flat substrate vs non-flat substrate
```

The novelty claim should be algorithmic: stable moving contact-line phase-field simulation with Navier slip, Cox-Voinov dynamics, ghost-cell wetting, semi-implicit CH diffusion, and geometry-aware MAC boundary handling.

Goal 4: compare against real experiments.

Use `configs/config_upstream_flow_cox_voinov_staggered_64_faster_real.json` as the dimensional/realistic seed. Match reported observables rather than raw fields:

```text
droplet centroid speed
advancing/receding contact-line speed
dynamic contact angles
steady or terminal sliding speed
shape aspect ratio over time
```

Fit only two model parameters for comparison runs: effective slip length and Cox-Voinov coefficient. Keep grid, Pe, epsilon, and inlet model fixed once selected.

### Experimental Targets

Primary air-shear target:

```text
Chahine, Sebilleau, Mathis, Legendre, "Sliding droplets in a laminar or turbulent boundary layer",
Phys. Rev. Fluids 7, 113605, 2022.
https://doi.org/10.1103/PhysRevFluids.7.113605
```

This is the closest match to the present setup: a sessile droplet on a horizontal substrate driven by laminar or turbulent airflow. The paper reports onset via a critical Weber number, shape regimes (`oval`, `corner`, `rivulet`), and phase diagrams using capillary and Bond numbers. It also frames the force competition as aerodynamic drag versus contact-angle-hysteresis capillary retention. For this target, compare:

```text
critical airflow / critical Weber number
COM speed after onset
wetting length and aspect ratio
advancing/receding apparent contact angles
shape regime transition: compact/oval -> corner -> rivulet
```

Secondary high-deformation target:

```text
Chahine, Sebilleau, Mathis, Legendre, "Caterpillar like motion of droplet in a shear flow",
Phys. Rev. Fluids 8, 093601, 2023.
https://doi.org/10.1103/PhysRevFluids.8.093601
```

This is useful only after the baseline is stable and sufficiently 3D-capable. It focuses on glycerin droplets under horizontal shear flow, elongation into a rivulet, surface waves, stretching/contracting motion, breakup, and coalescence. In 2D, we can only mimic the onset of elongation and wave-like interface motion, not full transverse rivulet physics.

Secondary wind-tunnel wettability target:

```text
Wang, Chang, Zhao, Yang, "Dynamic behaviors of water droplet moving on surfaces with different wettability driven by airflow",
International Journal of Multiphase Flow 154, 104127, 2022.
https://doi.org/10.1016/j.ijmultiphaseflow.2022.104127
```

This paper gives a pragmatic parameter grid: airflow ramp up to 22.2 m/s, droplet volumes 20-80 μL, multiple substrate wettabilities, high-speed imaging, wetting length, COM velocity, and dynamic contact angles. It is a good validation template because it reports motion stages: pinned oscillation, slight wetting-line motion, sharp elongation/acceleration, and possible breakup.

Surface-topography target:

```text
"Displacement of liquid droplets on micro-grooved surfaces with air flow",
Experimental Thermal and Fluid Science 49, 86-93, 2013.
https://doi.org/10.1016/j.expthermflusci.2013.04.005
```

This target is useful for the novelty section, not the first validation. It requires anisotropic substrate structure because the experiment varies groove direction and dimensions, and reports that initiation velocity depends on groove geometry.

Inclined-plane target:

```text
Kawahara et al., "Relationship between Onset of Sliding Behavior and Size of Droplet on Inclined Solid Substrate",
open PMC record, 2022.
https://pmc.ncbi.nlm.nih.gov/articles/PMC9695122/
```

This is the best gravity-driven control case. It studies onset versus droplet volume on an inclined substrate and emphasizes advancing/receding contact angles and hysteresis. Use it to validate the gravity/inclined-plane branch once ghost-cell wetting supports non-flat or terrain-following geometry.

## Required Code Changes Before Full Grid

Implemented:

```text
slip_blasius inlet subtype
optional Blasius normal inlet velocity
MAC inlet support for optional normal profile
```

Still needed for stronger paper claims:

```text
mixed top far-field velocity BC: u = U_inf, dv/dy = 0
scripted metric extraction: mass, COM, contact lines, dynamic angles
config generator for the full grid
remote launcher that maps jobs across two A100s
non-flat substrate grid cases using existing geometry configs
contact-angle hysteresis model: separate advancing/receding angles
airflow ramp schedule: U_inf(t), not only instantaneous steady inlet
experimental nondimensionalizer: Vd, H, U_inf, Bo, Ca, We, Re -> config
wetting-length / aspect-ratio metric extraction
shape-regime classifier: oval, corner, rivulet/tail, breakup
```

Blocking limitation for geometry runs:

```text
PhaseFieldSolverGhostCell currently supports only flat geometry.
```

This is enforced in `src/physics/phase_field.py`. The code already has `flat`, `tilted`, `hump`, and `from_height` geometry objects, and state creation can build `tilted` and `hump` substrates. However, the robust ghost-cell wetting path is flat-only. For real inclined-plane or grooved-substrate experiments, the next solver task is extending ghost-cell wetting to a local wall-normal ghost construction:

```text
build ghost values along local normal n = (-h'(x), 1) / sqrt(1 + h'(x)^2)
apply wall tangential Navier slip on the same surface normal
use terrain/cut-cell divergence and PPE operators consistently near the wall
extract contact-line velocity projected onto the local tangent
```

Without this, non-flat cases should be treated as exploratory only, because they would fall back to the legacy/simple contact-angle path that caused mass and boundedness artifacts.

## Curved-Substrate Ghost-Cell Equations

For the existing terrain-following representation,

```text
X = x
Y = eta + h(x)
```

the physical derivatives are

```text
∂/∂X = ∂/∂x - h'(x) ∂/∂eta
∂/∂Y = ∂/∂eta
grad(phi) = (phi_x - h' phi_eta, phi_eta)
```

The terrain Laplacian used by the phase solver is

```text
Δphi = phi_xx - 2 h' phi_xeta - h'' phi_eta + (1 + h'^2) phi_etaeta
```

At the bottom wall, the unit tangent and normal are

```text
t = (1, h') / sqrt(1 + h'^2)
n = (-h', 1) / sqrt(1 + h'^2)
```

The contact-angle ghost row enforces the physical normal derivative

```text
q = grad(phi) · n
```

with the same law as the flat ghost-cell solver:

```text
analytic_gradient: q = -cot(theta) |grad(phi) · t|
wall_energy:       q = -cos(theta) (1 - phi_wall^2) / (sqrt(2) epsilon)
```

The computational eta derivative required at the wall is therefore

```text
phi_eta_wall = (q sqrt(1 + h'^2) + h' phi_x) / (1 + h'^2)
```

and the ghost row below `eta = 0` is

```text
phi_ghost = phi_1 - 2 deta phi_eta_wall
```

The semi-implicit CH update now uses the terrain Laplacian matrix for curved geometry and adds an explicit correction for the nonlinear analytic ghost-row contribution. This matters because the analytic ghost law is state-dependent and cannot be represented as a fixed sparse matrix without linearization.

Remaining full-simulation blocker:

```text
PPE currently rejects non-flat geometry in the staggered path.
```

So curved ghost-cell support is locally testable in the phase solver, but end-to-end curved sliding still needs the staggered PPE/projection path to accept terrain/cut-cell geometry.

## Initial Grid

Recommended first batch:

```text
ca: 60, 90, 120
grid: 128, 256
slip_length: 0.005, 0.02, 0.05
phase solver: ghost_cell
phase update: semi_implicit_ch
inlet: slip_blasius
cox_voinov: on
```

Recommended artifact batch:

```text
ca: 120
grid: 128
slip_length: 0.02
inlet: tanh_blasius, slip_blasius
phase solver/update: simple_explicit, ghost_explicit, ghost_semi_implicit
```

Recommended geometry batch:

```text
substrate: flat, shallow sinusoid, single hump, tilted line
ca: 120
grid: 256
slip_length: 0.02
solver/update: ghost_cell + semi_implicit_ch
inlet: slip_blasius
```

## Launched Long Grid

Remote run folder prefix:

```text
/home/jovyan/shares/SR003.nfs2/naburov/drop/experiment_longreal_*
```

Config folder:

```text
configs/generated_sliding_long_realpaper
```

Long baseline flat grid:

```text
cases: 9
t_max: 0.20
checkpoint_interval: 5000
ca: 60, 90, 120
slip_length: 0.005, 0.02, 0.05
U_inf: 1.0
grid: 128^2
```

Chahine PRF 2022 air-shear onset-inspired grid:

```text
cases: 4
t_max: 0.10
checkpoint_interval: 2500
ca: 90
radius: 0.15
U_inf: 0.5, 1.0, 1.5, 2.0
Re ∝ U_inf
We ∝ U_inf^2
```

Wang IJMF 2022 wettability/volume-inspired grid:

```text
cases: 6
t_max: 0.10
checkpoint_interval: 2500
ca: 50, 90, 120
radius: 0.12, 0.18
U_inf: 1.5
Re, We fixed to the U_inf=1.5 air-shear scaling
```

These are still flat-substrate approximations. Inclined-plane and grooved-substrate experiment matching remains blocked until the staggered PPE/projection path accepts terrain/cut-cell geometry.
