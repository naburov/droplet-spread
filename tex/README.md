# LaTeX Documentation for Droplet Spreading Model

This directory contains comprehensive LaTeX documentation of the mathematical models implemented in the droplet spreading simulation code.

## Files

### 1. `two_phase_model.tex`
Complete mathematical description of the two-phase droplet spreading model **without** ice-water transition.

**Contents:**
- Governing equations (Cahn-Hilliard, Navier-Stokes)
- All boundary conditions for each variable
- Material property definitions
- Time integration scheme
- Initial conditions
- Summary tables

**Variables covered:**
- Phase field ($\phi$): liquid-gas interface
- Velocity ($\mathbf{u}$)
- Pressure ($p$)
- Chemical potential ($\mu_c$)
- Surface tension force ($\mathbf{F}_{st}$)

### 2. `three_phase_model_with_ice.tex`
Complete mathematical description of the three-phase droplet spreading model **with** ice-water phase transition.

**Contents:**
- All equations from two-phase model
- Additional ice phase field equation ($\psi$)
- Temperature equation with latent heat
- Modified Navier-Stokes with ice constraints
- Ice-water coupling terms
- All boundary conditions including ice-specific ones
- Three-phase material properties

**Additional variables:**
- Ice phase field ($\psi$): water-ice interface
- Temperature ($T$)

## Compilation

To compile the LaTeX documents:

```bash
cd tex
pdflatex two_phase_model.tex
pdflatex three_phase_model_with_ice.tex
```

Or use your preferred LaTeX compiler (e.g., `xelatex`, `lualatex`).

## Notes

- All formulas are directly extracted from the code implementation
- Boundary conditions match exactly what is implemented in the code
- The documents include both continuous (mathematical) and discrete (numerical) formulations
- All dimensionless parameters are explained

## Structure

Each document follows this structure:
1. Introduction
2. Problem domain
3. Governing equations (with all terms explained)
4. Boundary conditions (for every variable and every boundary)
5. Time integration
6. Initial conditions
7. Summary tables

The boundary conditions section is particularly detailed, as requested, covering every boundary condition for each variable.


