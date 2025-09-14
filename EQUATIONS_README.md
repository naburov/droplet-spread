# Two-Phase Droplet Spreading: System of Equations and Numerical Scheme

## Overview

This document provides the mathematical formulation and numerical implementation of the two-phase droplet spreading simulation. The system solves coupled partial differential equations for interface tracking and fluid flow.

## Core Governing Equations

### 1. Phase Field Equation (Cahn-Hilliard)

The interface between two phases is tracked using the Cahn-Hilliard equation:

```
∂φ/∂t + ∇·(φu) = ∇·(M∇μ)
```

where:
- `φ(x,t)` is the phase field variable (φ = 1 for phase 1, φ = -1 for phase 2)
- `u(x,t)` is the velocity field
- `M` is the mobility parameter
- `μ` is the chemical potential: `μ = f'(φ) - ε²∇²φ`
- `f(φ) = (1/4)(φ² - 1)²` is the double-well potential

### 2. Navier-Stokes Equations

The fluid flow is governed by the incompressible Navier-Stokes equations:

**Mass Conservation:**
```
∂ρ/∂t + ∇·(ρu) = 0
```

**Momentum Conservation:**
```
ρ ∂u/∂t + ρ(u·∇)u = -∇p + ∇·τ + F_surface + ρg
```

where:
- `ρ(φ)` is the density field
- `p(x,t)` is the pressure field
- `τ` is the viscous stress tensor
- `F_surface` is the surface tension force
- `g` is the gravitational acceleration

### 3. Surface Tension Force

```
F_surface = (3√2ε)/(4We) ∇·(∇φ/|∇φ|) |∇φ| ∇φ
```

## Material Properties

The material properties vary smoothly across the interface:

```
ρ(φ) = (ρ₁ + ρ₂)/2 + (ρ₂ - ρ₁)/2 φ
1/Re(φ) = 1/Re₁ + 1/Re₂ - 1/Re₁ (1 + φ)/2
1/We(φ) = 1/We₁ + 1/We₂ - 1/We₁ (1 + φ)/2
```

## Numerical Scheme

### Time Discretization

The system uses explicit time-stepping with adaptive time step control:

```
Δt = min(Δt_CFL, Δt_CH, Δt_user)
```

where:
- `Δt_CFL = C_CFL / (|u_max|/Δx + |v_max|/Δy)` (CFL condition)
- `Δt_CH = 0.01 min(Δx, Δy)⁴ / (M ε²)` (Cahn-Hilliard stability)
- `Δt_user` is the user-specified time step

### Solution Algorithm

**Step 1: Update Phase Field**
```
φ^(n+1) = φ^n + Δt [-∇·(φ^n u^n) + ∇·(M∇μ^n)]
```

**Step 2: Calculate Surface Tension**
```
F_surface^(n+1) = (3√2ε)/(4We(φ^(n+1))) ∇·(∇φ^(n+1)/|∇φ^(n+1)|) |∇φ^(n+1)| ∇φ^(n+1)
```

**Step 3: Update Velocity Field**
```
u^(n+1) = u^n + Δt [-∇p^n/ρ^(n+1) + ∇·τ^n/ρ^(n+1) - F_surface^(n+1)/ρ^(n+1) - (u^n·∇)u^n + g]
```

**Step 4: Pressure Projection (PPE)**
```
∇²p_correction = ∇·u^(n+1)/Δt
u^(n+1) = u^(n+1) - Δt ∇p_correction
```

## Boundary Conditions

### Phase Field
- **Contact Angle:** `∂φ/∂n = -cos(θ_c)/ε √(2f(φ))` at solid surface

### Velocity
- **No-Slip:** `u = 0` on solid boundaries

### Pressure
- **Top:** `∂p/∂n = 0` (Neumann)
- **Bottom:** `p = p_atm + ∫₀ʸ ρg dy` (Dirichlet)

## Spatial Discretization

Finite differences on uniform Cartesian grid:

```
∂f/∂x ≈ (f_{i+1,j} - f_{i-1,j})/(2Δx)
∂f/∂y ≈ (f_{i,j+1} - f_{i,j-1})/(2Δy)
∇²f ≈ (f_{i+1,j} - 2f_{i,j} + f_{i-1,j})/Δx² + (f_{i,j+1} - 2f_{i,j} + f_{i,j-1})/Δy²
```

## Dimensionless Numbers

```
Re_i = ρ_i U₀ L₀ / μ_i     (Reynolds number)
We_i = ρ_i U₀² L₀ / σ      (Weber number)
Pe = U₀ L₀ / (M ε²)        (Peclet number)
Fr = U₀² / (g L₀)          (Froude number)
```

## Implementation Details

- **Language:** Python with JAX for automatic differentiation and JIT compilation
- **Grid:** Uniform Cartesian with finite differences
- **Time Integration:** Explicit Euler with adaptive time stepping
- **Interface Tracking:** Cahn-Hilliard phase field method
- **Incompressibility:** Pressure projection method (PPE)
- **Linear Solvers:** PyAMG with BiCGSTAB acceleration
- **Boundary Conditions:** Contact angle, no-slip, and pressure conditions

## Generated Documents

1. **`system_equations.pdf`** - Complete detailed mathematical formulation
2. **`equations_summary.pdf`** - Concise summary of key equations
3. **`system_equations.tex`** - LaTeX source for detailed document
4. **`equations_summary.tex`** - LaTeX source for summary document

## Files Created

- `system_equations.tex` - Complete LaTeX document with full mathematical formulation
- `system_equations.pdf` - Compiled PDF (5 pages)
- `equations_summary.tex` - Concise LaTeX summary
- `equations_summary.pdf` - Compiled PDF summary (3 pages)
- `EQUATIONS_README.md` - This markdown summary

