#!/usr/bin/env python3
"""
Diagnostic Guide: What Each Plot Should Look Like

This script provides a comprehensive guide for interpreting diagnostic plots
and what the expected behavior should be for a well-functioning simulation.
"""

import matplotlib.pyplot as plt
import numpy as np

def create_diagnostic_guide():
    """Create a comprehensive guide showing what each diagnostic should look like."""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Diagnostic Guide: Expected Behavior for Well-Functioning Simulation', fontsize=16)
    
    # 1. Divergence Analysis
    ax = axes[0, 0]
    steps = np.linspace(0, 1000, 100)
    
    # Good divergence: starts high, quickly drops and stays low
    good_div = 1e-1 * np.exp(-steps/50) + 1e-4
    bad_div = 1e-1 * np.ones_like(steps)  # Constant high divergence
    unstable_div = 1e-1 * np.exp(steps/200)  # Growing divergence
    
    ax.semilogy(steps, good_div, 'g-', linewidth=2, label='Good (decreasing)')
    ax.semilogy(steps, bad_div, 'r-', linewidth=2, label='Bad (constant high)')
    ax.semilogy(steps, unstable_div, 'orange', linewidth=2, label='Unstable (growing)')
    ax.axhline(y=1e-3, color='g', linestyle='--', alpha=0.7, label='Good threshold')
    ax.axhline(y=1e-2, color='orange', linestyle='--', alpha=0.7, label='Acceptable')
    ax.axhline(y=1e-1, color='r', linestyle='--', alpha=0.7, label='Poor')
    ax.set_title('Divergence Norms\n(Should decrease and stay < 1e-2)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Divergence Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Energy Analysis
    ax = axes[0, 1]
    
    # Good energy: peaks then decays
    good_energy = 1e-2 * np.exp(-(steps-200)**2/10000) + 1e-4
    bad_energy = 1e-2 * np.ones_like(steps)  # Constant high energy
    growing_energy = 1e-2 * np.exp(steps/500)  # Growing energy
    
    ax.semilogy(steps, good_energy, 'g-', linewidth=2, label='Good (peaks then decays)')
    ax.semilogy(steps, bad_energy, 'r-', linewidth=2, label='Bad (constant high)')
    ax.semilogy(steps, growing_energy, 'orange', linewidth=2, label='Unstable (growing)')
    ax.set_title('Kinetic Energy\n(Should peak then decay for spreading)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Kinetic Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Mass Conservation
    ax = axes[0, 2]
    
    # Good mass: constant
    good_mass = 1.0 * np.ones_like(steps)
    bad_mass = 1.0 - 0.1 * steps/1000  # Decreasing mass
    oscillating_mass = 1.0 + 0.05 * np.sin(steps/50)  # Oscillating mass
    
    ax.plot(steps, good_mass, 'g-', linewidth=2, label='Good (constant)')
    ax.plot(steps, bad_mass, 'r-', linewidth=2, label='Bad (decreasing)')
    ax.plot(steps, oscillating_mass, 'orange', linewidth=2, label='Oscillating')
    ax.axhspan(0.99, 1.01, alpha=0.2, color='g', label='±1% tolerance')
    ax.axhspan(0.95, 1.05, alpha=0.1, color='orange', label='±5% tolerance')
    ax.set_title('Mass Conservation\n(Should be constant - perfect conservation)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Mass')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. CFL Analysis
    ax = axes[1, 0]
    
    # Good CFL: all below 0.5
    good_cfl = 0.1 * np.ones_like(steps)
    bad_cfl = 0.8 * np.ones_like(steps)  # Above stability limit
    growing_cfl = 0.1 * np.exp(steps/500)  # Growing CFL
    
    ax.semilogy(steps, good_cfl, 'g-', linewidth=2, label='Good (< 0.5)')
    ax.semilogy(steps, bad_cfl, 'r-', linewidth=2, label='Bad (> 0.5)')
    ax.semilogy(steps, growing_cfl, 'orange', linewidth=2, label='Growing')
    ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='Stability limit')
    ax.axhline(y=0.1, color='g', linestyle='--', alpha=0.7, label='Conservative')
    ax.set_title('CFL Numbers\n(All should be < 0.5 for stability)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('CFL Number')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Pressure-Curvature Relationship
    ax = axes[1, 1]
    
    # Good relationship: linear with slope = surface tension
    curvature = np.linspace(-2, 2, 50)
    sigma = 10.0  # Surface tension
    good_pressure = sigma * curvature + 0.1 * np.random.normal(0, 0.1, len(curvature))
    bad_pressure = 5.0 * curvature + 0.5 * np.random.normal(0, 0.5, len(curvature))  # Wrong slope
    scattered_pressure = sigma * curvature + 2.0 * np.random.normal(0, 1.0, len(curvature))  # High scatter
    
    ax.scatter(curvature, good_pressure, alpha=0.6, s=20, label='Good (linear, low scatter)')
    ax.scatter(curvature, bad_pressure, alpha=0.6, s=20, label='Bad (wrong slope)')
    ax.scatter(curvature, scattered_pressure, alpha=0.6, s=20, label='Poor (high scatter)')
    ax.plot(curvature, sigma * curvature, 'k--', linewidth=2, label=f'Young-Laplace: Δp = {sigma}κ')
    ax.set_title('Pressure-Curvature\n(Should follow linear relationship)')
    ax.set_xlabel('Curvature κ')
    ax.set_ylabel('Pressure Jump Δp')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Chemical Potential
    ax = axes[1, 2]
    
    # Good chemical potential: near zero
    good_mu = 0.1 * np.sin(steps/100) + 0.05 * np.random.normal(0, 0.01, len(steps))
    bad_mu = 1.0 * np.ones_like(steps)  # High constant
    oscillating_mu = 0.5 * np.sin(steps/20)  # High amplitude oscillations
    
    ax.plot(steps, good_mu, 'g-', linewidth=2, label='Good (near 0)')
    ax.plot(steps, bad_mu, 'r-', linewidth=2, label='Bad (high constant)')
    ax.plot(steps, oscillating_mu, 'orange', linewidth=2, label='Oscillating')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5, label='Equilibrium (μ=0)')
    ax.axhline(y=0.1, color='g', linestyle='--', alpha=0.7, label='Good threshold')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Poor threshold')
    ax.set_title('Chemical Potential\n(Should be near 0 for equilibrium)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Chemical Potential μ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Contact Angle
    ax = axes[2, 0]
    
    # Good contact angle: approaches target value
    target_angle = 60
    good_angle = target_angle + 10 * np.exp(-steps/100) + 2 * np.random.normal(0, 1, len(steps))
    bad_angle = 30 * np.ones_like(steps)  # Wrong angle
    oscillating_angle = target_angle + 20 * np.sin(steps/50)  # Oscillating
    
    ax.plot(steps, good_angle, 'g-', linewidth=2, label='Good (approaches target)')
    ax.plot(steps, bad_angle, 'r-', linewidth=2, label='Bad (wrong angle)')
    ax.plot(steps, oscillating_angle, 'orange', linewidth=2, label='Oscillating')
    ax.axhline(y=target_angle, color='k', linestyle='--', alpha=0.7, label=f'Target ({target_angle}°)')
    ax.axhspan(target_angle-5, target_angle+5, alpha=0.2, color='g', label='±5° tolerance')
    ax.set_title('Contact Angle\n(Should approach target value)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Contact Angle (degrees)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 8. Force Budget
    ax = axes[2, 1]
    
    # Good force balance: forces are balanced
    steps_short = steps[:50]
    pressure_force = 1.0 * np.ones_like(steps_short)
    surface_force = 0.8 * np.ones_like(steps_short)
    viscous_force = 0.2 * np.ones_like(steps_short)
    gravity_force = 0.1 * np.ones_like(steps_short)
    
    ax.plot(steps_short, pressure_force, 'b-', linewidth=2, label='Pressure force')
    ax.plot(steps_short, surface_force, 'g-', linewidth=2, label='Surface tension')
    ax.plot(steps_short, viscous_force, 'r-', linewidth=2, label='Viscous force')
    ax.plot(steps_short, gravity_force, 'm-', linewidth=2, label='Gravity')
    ax.set_title('Force Budget\n(Forces should be balanced)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Force Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 9. Summary Table
    ax = axes[2, 2]
    ax.axis('off')
    
    summary_text = """
    DIAGNOSTIC INTERPRETATION GUIDE
    
    ✅ GOOD SIMULATION:
    • Divergence: < 1e-2, decreasing
    • Energy: peaks then decays
    • Mass: constant (±1%)
    • CFL: all < 0.5
    • Pressure-curvature: linear, low scatter
    • Chemical potential: near 0
    • Contact angle: approaches target
    • Forces: balanced
    
    ❌ PROBLEMS TO WATCH:
    • High/constant divergence → PPE issues
    • Growing energy → Instability
    • Mass loss → Conservation issues
    • CFL > 0.5 → Time step too large
    • Non-linear pressure-curvature → Interface issues
    • High chemical potential → Non-equilibrium
    • Wrong contact angle → BC issues
    • Unbalanced forces → Physics errors
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the guide
    plt.savefig('diagnostic_guide.png', dpi=300, bbox_inches='tight')
    print("Diagnostic guide saved as 'diagnostic_guide.png'")
    
    return fig

def main():
    """Create and display the diagnostic guide."""
    print("Creating diagnostic guide...")
    fig = create_diagnostic_guide()
    
    # Don't show the plot when called from run_all_diagnostics
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--no-display":
        plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    main()
