# Droplet Spreading Simulation Diagnostics

This directory contains comprehensive diagnostic scripts for analyzing droplet spreading simulations. The diagnostics are designed to quickly localize problems and validate the numerical implementation.

## Quick Start

Run all diagnostics on an experiment:
```bash
python run_all_diagnostics.py experiment_20250915_165314
```

Run individual diagnostics:
```bash
python divergence_analysis.py experiment_20250915_165314
python energy_analysis.py experiment_20250915_165314
# ... etc
```

## Diagnostic Scripts

### 1. **Divergence Analysis** (`divergence_analysis.py`)
- **Purpose**: Validates projection efficacy by analyzing ∇·u field
- **Outputs**: 
  - L₂ and L∞ norms of divergence vs time
  - Divergence field maps at different time steps
  - Divergence magnitude statistics
- **Key Metrics**: 
  - Divergence norms should be small (< 1e-3)
  - L∞ norm should be < 1e-2 for good projection

### 2. **Energy Analysis** (`energy_analysis.py`)
- **Purpose**: Tracks kinetic energy and velocity statistics to detect solver/BC issues
- **Outputs**:
  - Kinetic energy evolution
  - Velocity statistics (max, RMS, mean)
  - Velocity component analysis
  - Kinetic energy density maps
- **Key Metrics**:
  - Kinetic energy should decrease over time
  - No sudden spikes in velocity
  - Velocities should be within reasonable bounds

### 3. **Mass Conservation** (`mass_conservation.py`)
- **Purpose**: Validates mass and phase conservation
- **Outputs**:
  - Phase field integral evolution
  - Droplet volume and area fraction
  - Mass evolution (total and droplet)
  - Phase field statistics
- **Key Metrics**:
  - Phase field integral should be constant
  - Mass should be conserved
  - Phase field values should stay within [-1, 1]

### 4. **Pressure-Curvature Analysis** (`pressure_curvature.py`)
- **Purpose**: Tests Young-Laplace law compliance
- **Outputs**:
  - Pressure jump vs curvature scatter plot
  - Error analysis (numerical vs theoretical)
  - Error distribution histograms
- **Key Metrics**:
  - Pressure jump should follow Δp = σκ
  - Relative error should be < 10%
  - Good correlation between numerical and theoretical values

### 5. **Chemical Potential Diagnostics** (`chemical_potential.py`)
- **Purpose**: Analyzes μ, Δφ, and |∇φ| near interface
- **Outputs**:
  - Chemical potential statistics
  - Gradient magnitude evolution
  - Laplacian statistics
  - Chemical potential field maps
- **Key Metrics**:
  - Low variance in chemical potential
  - Reasonable gradient magnitudes
  - No stiffness artifacts

### 6. **Force Budget Analysis** (`force_budget.py`)
- **Purpose**: Analyzes force balance along vertical line through droplet
- **Outputs**:
  - Force magnitude profiles (pressure, surface tension, viscous, gravity)
  - Force ratio evolution
  - Force balance validation
- **Key Metrics**:
  - Force ratios should sum to ~1
  - Balanced force distribution
  - No dominant force terms

### 7. **Contact Angle Analysis** (`contact_angle.py`)
- **Purpose**: Measures contact angle evolution and accuracy
- **Outputs**:
  - Contact angle vs time
  - Contact line position
  - Droplet dimensions
  - Contact angle error analysis
- **Key Metrics**:
  - Contact angle should match target
  - Stable contact angle evolution
  - Reasonable droplet dimensions

### 8. **CFL Analysis** (`cfl_analysis.py`)
- **Purpose**: Tracks CFL numbers and time scales for stability
- **Outputs**:
  - CFL numbers (advective, diffusive, capillary, gravity, interface)
  - Velocity evolution
  - Time step recommendations
  - Stability analysis
- **Key Metrics**:
  - All CFL numbers should be < 0.5
  - Recommended dt should be > actual dt
  - Stable velocity evolution

### 9. **First Step Visualization** (`first_step_visualization.py`)
- **Purpose**: Comprehensive visualization of the first simulation step for debugging
- **Outputs**:
  - Phase field evolution (initial vs final)
  - Surface tension distribution and divergence
  - Velocity field and divergence at each stage
  - Pressure correction and distribution
  - Divergence evolution through the step
  - Cross-sections and surface analysis
  - Statistics summary
- **Key Metrics**:
  - Divergence should decrease through the step
  - Surface tension should be properly distributed
  - Pressure correction should be reasonable
  - All fields should be physically consistent
- **Usage**: 
  - Automatically run at step 0 during simulation
  - Can be run manually: `python first_step_visualization.py <config_path> [output_dir]`

## Output Files

Each diagnostic generates:
- **Plot files**: `.png` files with analysis plots
- **Data files**: `.json` files with numerical data
- **Console output**: Summary statistics and warnings

## Common Issues and Solutions

### High Divergence
- **Symptom**: L∞ norm > 1e-2
- **Causes**: Poor projection, incorrect boundary conditions
- **Solutions**: Check PPE solver, verify BCs, reduce time step

### Mass Loss
- **Symptom**: Phase field integral decreasing
- **Causes**: Advection errors, boundary condition leakage
- **Solutions**: Check advection scheme, verify BCs, improve mass conservation

### Interface Diffusion
- **Symptom**: Interface becoming too thick
- **Causes**: Low Peclet number, large time step
- **Solutions**: Increase Pe, decrease dt, improve interface treatment

### Contact Angle Errors
- **Symptom**: Measured angle ≠ target angle
- **Causes**: Incorrect Robin parameter, poor gradient calculation
- **Solutions**: Tune Robin parameter, improve gradient scheme

### CFL Violations
- **Symptom**: CFL > 0.5
- **Causes**: Time step too large, high velocities
- **Solutions**: Reduce dt, improve velocity limiting, check stability

## Usage Tips

1. **Run all diagnostics first** to get a complete picture
2. **Check critical diagnostics** (divergence, mass, CFL) for stability
3. **Look for patterns** across multiple diagnostics
4. **Compare with target values** from literature or theory
5. **Use plots** to identify spatial and temporal patterns

## Dependencies

- NumPy
- Matplotlib
- SciPy
- Scikit-image (for contour finding)
- Custom modules from `../src/`

## Troubleshooting

If a diagnostic fails:
1. Check that the experiment directory exists
2. Verify checkpoint files are present
3. Check Python path includes `../src/`
4. Look at error messages for specific issues
5. Try running individual diagnostics to isolate problems




