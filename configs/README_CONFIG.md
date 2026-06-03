# Configuration Files

## Available Configs

| Config | Description | Use Case |
|--------|-------------|----------|
| `staggered_poiseuille.json` | **Single-phase** channel flow (fluid only, no phase field) | Use `run_staggered_flow.py`, not `main.py` |
| `config_template.json` | Base template with default parameters | Starting point for new simulations |
| `config_droplet_simple.json` | Relaxed parameters, fast execution | Debugging and quick tests |
| `config_droplet_realistic.json` | Realistic air-water parameters | Production simulations |
| `config_droplet_geometry.json` | Non-flat surface (hump) | Geometry-aware BC testing |
| `config_ice_template.json` | Ice-water transition template | Starting point for freezing simulations |
| `config_droplet_ice.json` | Realistic freezing simulation | Production ice simulations |

## Quick Start

```bash
# Single-phase Poiseuille (channel flow, no droplet/phase field)
python3 run_staggered_flow.py --config configs/staggered_poiseuille.json

# Two-phase (droplet) simulations — use main.py:
# Simple test
PYTHONPATH=src python main.py --config configs/config_droplet_simple.json

# Realistic simulation  
PYTHONPATH=src python main.py --config configs/config_droplet_realistic.json

# Geometry test
PYTHONPATH=src python main.py --config configs/config_droplet_geometry.json

# Ice simulation
PYTHONPATH=src python main.py --config configs/config_droplet_ice.json
```

## Config Structure

### physical_params
- `rho1`, `rho2`: Density of phase 1 (air) and phase 2 (water)
- `Re1`, `Re2`: Reynolds numbers
- `We1`, `We2`: Weber numbers  
- `Pe`: Peclet number (phase field mobility)
- `epsilon`: Interface thickness
- `contact_angle`: Contact angle in degrees
- `Fr`: Froude number
- `g`: Gravitational acceleration
- `include_ice_water_transition`: Enable ice physics

### grid_params
- `Lx`, `Ly`: Domain size
- `Nx`, `Ny`: Grid resolution

### time_params
- `dt`: Maximum time step
- `dt_initial`: Initial time step (for ramp-up)
- `t_max`: Simulation end time
- `cfl_number`: CFL limit for advection
- `capillary_cfl_number`: CFL limit for surface tension

### boundary_conditions
- `pressure`: Pressure BC (open, neumann, dirichlet)
- `velocity`: Velocity BC (no_slip, do_nothing, slip_symmetry)
- `phase_field`: Phase field BC (neumann, contact_angle)
- `advection`: Advection BC (open, impermeable)

For ice simulations:
- `temperature`: Temperature BC
- `ice_phase_field`: Ice phase field BC

### solver_params
- `pressure_solver`, `correction_solver`: Linear solver settings
- `ppe`: Pressure projection settings

## Phase Convention

- Phase 1 (φ = +1): Air/gas
- Phase 2 (φ = -1): Water/liquid
- Ice phase (ψ = +1): Ice
- Water phase (ψ = -1): Liquid water
