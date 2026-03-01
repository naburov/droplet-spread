"""
Plot telemetry data from CSV files.

This script reads statistics, boundary statistics, and PPE updates CSV files
and creates comprehensive visualizations. Supports baseline experiment overlay.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path


def align_by_step(current_df, baseline_df):
    """Align baseline data to current data by step.
    
    Args:
        current_df (pd.DataFrame): Current experiment data
        baseline_df (pd.DataFrame): Baseline experiment data
    
    Returns:
        pd.DataFrame: Baseline data aligned to current steps
    """
    if baseline_df is None or len(baseline_df) == 0:
        return None
    
    # Merge on step column, keeping only steps that exist in current
    aligned = pd.merge(
        current_df[['step']],
        baseline_df,
        on='step',
        how='left',
        suffixes=('', '_baseline')
    )
    
    return aligned


def plot_statistics(statistics_file, output_dir, baseline_df=None):
    """Plot general statistics from statistics.csv.
    
    Args:
        statistics_file (str): Path to statistics.csv
        output_dir (str): Directory to save plots
        baseline_df (pd.DataFrame, optional): Baseline statistics for overlay
    """
    if not os.path.exists(statistics_file):
        print(f"Warning: {statistics_file} not found, skipping statistics plots")
        return
    
    df = pd.read_csv(statistics_file)
    
    # Align baseline data by step
    baseline_aligned = align_by_step(df, baseline_df) if baseline_df is not None else None
    
    # Create figure with subplots (4 rows to include time step plot)
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Simulation Statistics Over Time', fontsize=16, fontweight='bold')
    
    # Plot 1: Phase field statistics
    ax = axes[0, 0]
    ax.plot(df['time'], df['phi_min'], label='Min', alpha=0.7, linewidth=1.5)
    ax.plot(df['time'], df['phi_max'], label='Max', alpha=0.7, linewidth=1.5)
    ax.plot(df['time'], df['phi_mean'], label='Mean', alpha=0.7, linewidth=1.5)
    if baseline_aligned is not None and 'phi_mean_baseline' in baseline_aligned.columns:
        ax.plot(df['time'], baseline_aligned['phi_mean_baseline'], 
               label='Baseline (mean)', linestyle='--', alpha=0.6, linewidth=1.5, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Phase Field Value')
    ax.set_title('Phase Field Statistics')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Velocity X component
    ax = axes[0, 1]
    ax.plot(df['time'], df['u_x_min'], label='Min', alpha=0.7, linewidth=1.5)
    ax.plot(df['time'], df['u_x_max'], label='Max', alpha=0.7, linewidth=1.5)
    ax.plot(df['time'], df['u_x_mean'], label='Mean', alpha=0.7, linewidth=1.5)
    if baseline_aligned is not None and 'u_x_mean_baseline' in baseline_aligned.columns:
        ax.plot(df['time'], baseline_aligned['u_x_mean_baseline'], 
               label='Baseline (mean)', linestyle='--', alpha=0.6, linewidth=1.5, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Velocity X')
    ax.set_title('Velocity X Component')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Velocity Y component
    ax = axes[0, 2]
    ax.plot(df['time'], df['u_y_min'], label='Min', alpha=0.7, linewidth=1.5)
    ax.plot(df['time'], df['u_y_max'], label='Max', alpha=0.7, linewidth=1.5)
    ax.plot(df['time'], df['u_y_mean'], label='Mean', alpha=0.7, linewidth=1.5)
    if baseline_aligned is not None and 'u_y_mean_baseline' in baseline_aligned.columns:
        ax.plot(df['time'], baseline_aligned['u_y_mean_baseline'], 
               label='Baseline (mean)', linestyle='--', alpha=0.6, linewidth=1.5, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Velocity Y')
    ax.set_title('Velocity Y Component')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Velocity magnitude
    ax = axes[1, 0]
    ax.plot(df['time'], df['u_magnitude_max'], label='Max Speed', color='red', linewidth=2)
    if baseline_aligned is not None and 'u_magnitude_max_baseline' in baseline_aligned.columns:
        ax.plot(df['time'], baseline_aligned['u_magnitude_max_baseline'], 
               label='Baseline', linestyle='--', alpha=0.6, linewidth=2, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Max Velocity Magnitude')
    ax.set_title('Maximum Velocity Magnitude')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Pressure statistics
    ax = axes[1, 1]
    ax.plot(df['time'], df['p_min'], label='Min', alpha=0.7, linewidth=1.5)
    ax.plot(df['time'], df['p_max'], label='Max', alpha=0.7, linewidth=1.5)
    ax.plot(df['time'], df['p_mean'], label='Mean', alpha=0.7, linewidth=1.5)
    if baseline_aligned is not None and 'p_mean_baseline' in baseline_aligned.columns:
        ax.plot(df['time'], baseline_aligned['p_mean_baseline'], 
               label='Baseline (mean)', linestyle='--', alpha=0.6, linewidth=1.5, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Pressure')
    ax.set_title('Pressure Statistics')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Surface tension
    ax = axes[1, 2]
    ax.plot(df['time'], df['surface_tension_max'], label='Max', color='purple', linewidth=2)
    if baseline_aligned is not None and 'surface_tension_max_baseline' in baseline_aligned.columns:
        ax.plot(df['time'], baseline_aligned['surface_tension_max_baseline'], 
               label='Baseline', linestyle='--', alpha=0.6, linewidth=2, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Surface Tension')
    ax.set_title('Maximum Surface Tension')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 7: Droplet mass
    ax = axes[2, 0]
    ax.plot(df['time'], df['droplet_mass'], label='Mass', color='green', linewidth=2)
    if baseline_aligned is not None and 'droplet_mass_baseline' in baseline_aligned.columns:
        ax.plot(df['time'], baseline_aligned['droplet_mass_baseline'], 
               label='Baseline', linestyle='--', alpha=0.6, linewidth=2, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Droplet Mass')
    ax.set_title('Droplet Mass Conservation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Droplet position (X and Y)
    ax = axes[2, 1]
    ax.plot(df['time'], df['droplet_start'], label='Start (X)', alpha=0.7, linewidth=1.5, color='blue')
    ax.plot(df['time'], df['droplet_end'], label='End (X)', alpha=0.7, linewidth=1.5, color='cyan')
    has_y_position = 'droplet_bottom' in df.columns and 'droplet_top' in df.columns
    if has_y_position:
        ax2 = ax.twinx()  # Create second y-axis for Y position
        ax2.plot(df['time'], df['droplet_bottom'], label='Bottom (Y)', alpha=0.7, linewidth=1.5, color='red', linestyle='--')
        ax2.plot(df['time'], df['droplet_top'], label='Top (Y)', alpha=0.7, linewidth=1.5, color='orange', linestyle='--')
        ax2.set_ylabel('Y Position (grid points)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper right', fontsize=8)
    if baseline_aligned is not None and 'droplet_start_baseline' in baseline_aligned.columns:
        ax.plot(df['time'], baseline_aligned['droplet_start_baseline'], 
               label='Baseline (start)', linestyle=':', alpha=0.4, linewidth=1.5, color='gray')
        ax.plot(df['time'], baseline_aligned['droplet_end_baseline'], 
               label='Baseline (end)', linestyle=':', alpha=0.4, linewidth=1.5, color='darkgray')
        if has_y_position and 'droplet_bottom_baseline' in baseline_aligned.columns:
            ax2.plot(df['time'], baseline_aligned['droplet_bottom_baseline'], 
                   label='Baseline (bottom)', linestyle=':', alpha=0.4, linewidth=1.5, color='lightcoral')
            ax2.plot(df['time'], baseline_aligned['droplet_top_baseline'], 
                   label='Baseline (top)', linestyle=':', alpha=0.4, linewidth=1.5, color='peachpuff')
    ax.set_xlabel('Time')
    if has_y_position:
        ax.set_ylabel('X Position (grid points)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_title('Droplet Position (X and Y)')
    else:
        ax.set_ylabel('Position (grid points)')
        ax.set_title('Droplet Position (X only)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Divergence
    ax = axes[2, 2]
    ax.semilogy(df['time'], df['divergence_max'], label='Max', alpha=0.7, color='red', linewidth=1.5)
    ax.semilogy(df['time'], df['divergence_mean'], label='Mean', alpha=0.7, color='blue', linewidth=1.5)
    if baseline_aligned is not None and 'divergence_mean_baseline' in baseline_aligned.columns:
        ax.semilogy(df['time'], baseline_aligned['divergence_mean_baseline'], 
                   label='Baseline (mean)', linestyle='--', alpha=0.6, linewidth=1.5, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Divergence (log scale)')
    ax.set_title('Velocity Divergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 10: Time Step (dt)
    ax = axes[3, 0]
    ax.semilogy(df['time'], df['dt'], label='Time Step (dt)', color='orange', linewidth=2, marker='o', markersize=2)
    if baseline_aligned is not None and 'dt_baseline' in baseline_aligned.columns:
        ax.semilogy(df['time'], baseline_aligned['dt_baseline'], 
                   label='Baseline', linestyle='--', alpha=0.6, linewidth=1.5, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Time Step (log scale)')
    ax.set_title('Time Step Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 11: Time Step vs Step Number
    ax = axes[3, 1]
    ax.semilogy(df['step'], df['dt'], label='Time Step (dt)', color='orange', linewidth=2, marker='o', markersize=2)
    if baseline_aligned is not None and 'dt_baseline' in baseline_aligned.columns:
        ax.semilogy(df['step'], baseline_aligned['dt_baseline'], 
                   label='Baseline', linestyle='--', alpha=0.6, linewidth=1.5, color='gray')
    ax.set_xlabel('Step Number')
    ax.set_ylabel('Time Step (log scale)')
    ax.set_title('Time Step vs Step Number')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 12: Cumulative Time
    ax = axes[3, 2]
    ax.plot(df['step'], df['time'], label='Cumulative Time', color='blue', linewidth=2)
    if baseline_aligned is not None and 'time_baseline' in baseline_aligned.columns:
        ax.plot(df['step'], baseline_aligned['time_baseline'], 
               label='Baseline', linestyle='--', alpha=0.6, linewidth=1.5, color='gray')
    ax.set_xlabel('Step Number')
    ax.set_ylabel('Cumulative Time')
    ax.set_title('Cumulative Simulation Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'statistics_plots.png')
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved statistics plots to {output_path}")
    finally:
        plt.close(fig)


def plot_boundary_statistics(boundary_file, output_dir, baseline_df=None):
    """Plot boundary statistics from boundary_statistics.csv.
    
    Args:
        boundary_file (str): Path to boundary_statistics.csv
        output_dir (str): Directory to save plots
        baseline_df (pd.DataFrame, optional): Baseline boundary statistics for overlay
    """
    if not os.path.exists(boundary_file):
        print(f"Warning: {boundary_file} not found, skipping boundary statistics plots")
        return
    
    df = pd.read_csv(boundary_file)
    
    # Align baseline data by step
    baseline_aligned = align_by_step(df, baseline_df) if baseline_df is not None else None
    
    # Extract boundary columns
    boundaries = ['left', 'right', 'top', 'bottom']
    fields = ['phi', 'u_x', 'u_y', 'p']
    
    # Create figure with subplots for each field
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Boundary Statistics Over Time', fontsize=16, fontweight='bold')
    
    for idx, field in enumerate(fields):
        ax = axes[idx // 2, idx % 2]
        
        for boundary in boundaries:
            mean_col = f'{field}_{boundary}_mean'
            
            if mean_col in df.columns:
                ax.plot(df['time'], df[mean_col], 
                       label=f'{boundary.capitalize()}', alpha=0.7, linewidth=2)
                
                # Add baseline if available
                baseline_col = f'{mean_col}_baseline'
                if baseline_aligned is not None and baseline_col in baseline_aligned.columns:
                    ax.plot(df['time'], baseline_aligned[baseline_col], 
                           label=f'{boundary.capitalize()} (baseline)', 
                           linestyle='--', alpha=0.5, linewidth=1.5, color='gray')
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{field.upper()} Value')
        ax.set_title(f'{field.upper()} at Boundaries')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'boundary_statistics_plots.png')
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved boundary statistics plots to {output_path}")
    finally:
        plt.close(fig)


def plot_ppe_updates(ppe_file, output_dir, baseline_df=None):
    """Plot PPE updates from ppe_updates.csv.
    
    Args:
        ppe_file (str): Path to ppe_updates.csv
        output_dir (str): Directory to save plots
        baseline_df (pd.DataFrame, optional): Baseline PPE updates for overlay
    """
    if not os.path.exists(ppe_file):
        print(f"Warning: {ppe_file} not found, skipping PPE plots")
        return
    
    df = pd.read_csv(ppe_file)
    
    # Align baseline data by step
    baseline_aligned = align_by_step(df, baseline_df) if baseline_df is not None else None
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('PPE Correction Statistics', fontsize=16, fontweight='bold')
    
    # Plot 1: PPE iterations
    ax = axes[0, 0]
    ax.plot(df['time'], df['ppe_iterations'], marker='o', markersize=3, alpha=0.7, linewidth=1.5)
    if baseline_aligned is not None and 'ppe_iterations_baseline' in baseline_aligned.columns:
        ax.plot(df['time'], baseline_aligned['ppe_iterations_baseline'], 
               marker='s', markersize=2, alpha=0.5, linestyle='--', 
               linewidth=1.5, color='gray', label='Baseline')
        ax.legend(fontsize=8)
    ax.set_xlabel('Time')
    ax.set_ylabel('PPE Iterations')
    ax.set_title('PPE Correction Iterations')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Divergence before and after
    ax = axes[0, 1]
    ax.semilogy(df['time'], df['divergence_before_max'], 
               label='Before (max)', alpha=0.7, color='red', linewidth=2)
    ax.semilogy(df['time'], df['divergence_after_max'], 
               label='After (max)', alpha=0.7, color='green', linewidth=2)
    if baseline_aligned is not None and 'divergence_after_max_baseline' in baseline_aligned.columns:
        ax.semilogy(df['time'], baseline_aligned['divergence_after_max_baseline'], 
                   label='Baseline (after)', linestyle='--', alpha=0.5, 
                   linewidth=1.5, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Divergence (log scale)')
    ax.set_title('Divergence Before/After PPE')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Mean divergence before and after
    ax = axes[1, 0]
    ax.semilogy(df['time'], df['divergence_before_mean'], 
               label='Before (mean)', alpha=0.7, color='red', linewidth=2)
    ax.semilogy(df['time'], df['divergence_after_mean'], 
               label='After (mean)', alpha=0.7, color='green', linewidth=2)
    if baseline_aligned is not None and 'divergence_after_mean_baseline' in baseline_aligned.columns:
        ax.semilogy(df['time'], baseline_aligned['divergence_after_mean_baseline'], 
                   label='Baseline (after)', linestyle='--', alpha=0.5, 
                   linewidth=1.5, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Divergence (log scale)')
    ax.set_title('Mean Divergence Before/After PPE')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: PPE applied indicator
    ax = axes[1, 1]
    ax.plot(df['time'], df['ppe_applied'], marker='o', markersize=2, alpha=0.7, linewidth=1.5)
    if baseline_aligned is not None and 'ppe_applied_baseline' in baseline_aligned.columns:
        ax.plot(df['time'], baseline_aligned['ppe_applied_baseline'], 
               marker='s', markersize=2, alpha=0.5, linestyle='--', 
               linewidth=1.5, color='gray', label='Baseline')
        ax.legend(fontsize=8)
    ax.set_xlabel('Time')
    ax.set_ylabel('PPE Applied (0/1)')
    ax.set_title('PPE Application Indicator')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ppe_updates_plots.png')
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved PPE plots to {output_path}")
    finally:
        plt.close(fig)


def _grid_like_plot_utils(Nx, Ny, Lx, Ly):
    """Same grid as plot_layout.prepare_joint_plot_data: x, eta, X, Eta."""
    x = np.linspace(0, Lx, Nx)
    eta = np.linspace(0, Ly, Ny)
    X, Eta = np.meshgrid(x, eta)
    return x, eta, X, Eta


def _phi0_contour_surface(phi, X, Eta):
    """Extract phi=0 contour (same as plot utils). Return free-surface part (y > 0) as (x_phys, y_phys)."""
    phi = np.asarray(phi)
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(X, Eta, phi.T, levels=[0.0])
    plt.close(fig_tmp)
    segs = cs.allsegs[0] if len(cs.levels) > 0 and cs.allsegs else []
    if not segs:
        return np.array([]), np.array([])
    best = max(segs, key=lambda s: np.max(s[:, 1]) - np.min(s[:, 1]) if len(s) > 0 else 0)
    best = np.asarray(best)
    if len(best) == 0:
        return np.array([]), np.array([])
    y_eps = 1e-6 * (np.nanmax(best[:, 1]) - np.nanmin(best[:, 1]) + 1e-12)
    mask = best[:, 1] >= y_eps
    if not np.any(mask):
        mask = best[:, 1] >= 0
    x_phys = best[mask, 0]
    y_phys = best[mask, 1]
    order = np.argsort(x_phys)
    x_phys = x_phys[order]
    y_phys = y_phys[order]
    # Extend to substrate (y=0) so contour meets the surface line at both contact points
    if len(x_phys) >= 2:
        x_phys = np.concatenate([[x_phys[0]], x_phys, [x_phys[-1]]])
        y_phys = np.concatenate([[0.0], y_phys, [0.0]])
    return x_phys, y_phys


def plot_contact_line_dynamics(experiment_dir, output_dir):
    """Plot interface height vs x at beginning, middle, end (striped lines). Saves to output_dir/contact_line_dynamics.png (use experiment root for output_dir)."""
    experiment_dir = Path(experiment_dir) if not isinstance(experiment_dir, Path) else experiment_dir
    output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
    params_path = experiment_dir / "simulation_parameters.json"
    stats_path = experiment_dir / "statistics.csv"
    checkpoint_dir = experiment_dir / "checkpoints"
    if not params_path.exists() or not stats_path.exists():
        return
    try:
        from .checkpointing import load_checkpoint
    except ImportError:
        from visualization.checkpointing import load_checkpoint
    import json
    with open(params_path) as f:
        params = json.load(f)
    grid = params.get("grid_params", {})
    Nx = grid.get("Nx", 128)
    Ly = grid.get("Ly", 1.0)
    Lx = grid.get("Lx", 1.0)
    dx = Lx / Nx
    dy = Ly / grid.get("Ny", 128)
    df = pd.read_csv(stats_path)
    steps = df["step"].values
    times = df["time"].values
    cands = sorted(Path(checkpoint_dir).glob("checkpoint_*.npz"))
    if len(cands) < 1:
        return
    step_vals = [int(p.stem.split("_")[1]) for p in cands]
    n_c = len(step_vals)
    step_initial = step_vals[0]
    step_beginning = step_vals[n_c // 4] if n_c >= 4 else step_vals[0]
    step_mid = step_vals[n_c // 2]
    step_end = step_vals[-1]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.axhline(0.0, color="gray", linestyle="-", linewidth=1.5, zorder=0)
    ax.fill_between([0, Lx], 0, -0.02 * Ly, color="gray", alpha=0.3, zorder=0)
    segments = [
        (step_initial, "Initial", "solid", "#4a4a4a"),
        (step_beginning, "Beginning", "dashed", "#1f77b4"),
        (step_mid, "Middle", "dotted", "#ff7f0e"),
        (step_end, "End", (0, (3, 1, 1, 1)), "#2ca02c"),
    ]
    lw = 1.0
    x_gr, eta_gr, X_gr, Eta_gr = _grid_like_plot_utils(
        grid.get("Nx", 128), grid.get("Ny", 128), Lx, Ly)
    for step, label, linestyle, color in segments:
        p = checkpoint_dir / f"checkpoint_{step:06d}.npz"
        if not p.exists():
            continue
        ck = load_checkpoint(str(p))
        phi = np.asarray(ck["phi"])
        x_phys, y_phys = _phi0_contour_surface(phi, X_gr, Eta_gr)
        if len(x_phys) == 0:
            continue
        t_val = times[np.argmin(np.abs(steps - step))]
        ax.plot(x_phys, y_phys, linestyle=linestyle, color=color, linewidth=lw, label=f"{label} (step {step}, t={t_val:.2e})", zorder=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Contact line dynamics")
    ax.set_xlim(0.3, 0.7)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02 * Ly, None)
    ax.set_aspect("equal")
    plt.tight_layout()
    out_path = output_dir / "contact_line_dynamics.png"
    try:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved contact line dynamics to {out_path}")
    finally:
        plt.close(fig)


def plot_ice_phase_transition(ice_file, output_dir, baseline_df=None):
    """Plot ice phase transition statistics from ice_phase_transition.csv.
    
    Args:
        ice_file (str): Path to ice_phase_transition.csv
        output_dir (str): Directory to save plots
        baseline_df (pd.DataFrame, optional): Baseline ice phase data for overlay
    """
    if not os.path.exists(ice_file):
        print(f"Warning: {ice_file} not found, skipping ice phase plots")
        return
    
    df = pd.read_csv(ice_file)
    
    # Align baseline data by step
    baseline_aligned = align_by_step(df, baseline_df) if baseline_df is not None else None
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Ice Phase Transition Statistics', fontsize=16, fontweight='bold')
    
    # Plot 1: Ice fraction
    ax = axes[0, 0]
    ax.plot(df['time'], df['ice_fraction'], label='Ice Fraction', color='blue', linewidth=2)
    if baseline_aligned is not None and 'ice_fraction_baseline' in baseline_aligned.columns:
        ax.plot(df['time'], baseline_aligned['ice_fraction_baseline'], 
               label='Baseline', linestyle='--', alpha=0.6, linewidth=1.5, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Ice Fraction')
    ax.set_title('Ice Fraction Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Freezing and melting rates
    ax = axes[0, 1]
    ax.plot(df['time'], df['freezing_rate'], label='Freezing Rate', color='cyan', linewidth=2)
    ax.plot(df['time'], df['melting_rate'], label='Melting Rate', color='orange', linewidth=2)
    if baseline_aligned is not None and 'freezing_rate_baseline' in baseline_aligned.columns:
        ax.plot(df['time'], baseline_aligned['freezing_rate_baseline'], 
               label='Baseline (freezing)', linestyle='--', alpha=0.5, linewidth=1.5, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Phase Change Rate')
    ax.set_title('Freezing and Melting Rates')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Freezing front position
    ax = axes[0, 2]
    ax.plot(df['time'], df['freezing_front_y'], label='Freezing Front Y', color='darkblue', linewidth=2)
    if baseline_aligned is not None and 'freezing_front_y_baseline' in baseline_aligned.columns:
        ax.plot(df['time'], baseline_aligned['freezing_front_y_baseline'], 
               label='Baseline', linestyle='--', alpha=0.6, linewidth=1.5, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Freezing Front Position (normalized)')
    ax.set_title('Freezing Front Position')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Temperature in ice region
    ax = axes[1, 0]
    ax.plot(df['time'], df['T_mean_ice'], label='Mean', color='blue', linewidth=2)
    ax.plot(df['time'], df['T_min_ice'], label='Min', color='lightblue', linewidth=1.5, alpha=0.7)
    ax.plot(df['time'], df['T_max_ice'], label='Max', color='darkblue', linewidth=1.5, alpha=0.7)
    ax.axhline(y=273.15, color='red', linestyle='--', alpha=0.5, label='T_melt')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Temperature in Ice Region')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Temperature in water region
    ax = axes[1, 1]
    ax.plot(df['time'], df['T_mean_water'], label='Mean', color='green', linewidth=2)
    ax.plot(df['time'], df['T_min_water'], label='Min', color='lightgreen', linewidth=1.5, alpha=0.7)
    ax.plot(df['time'], df['T_max_water'], label='Max', color='darkgreen', linewidth=1.5, alpha=0.7)
    ax.axhline(y=273.15, color='red', linestyle='--', alpha=0.5, label='T_melt')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Temperature in Water Region')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Temperature at interface
    ax = axes[1, 2]
    ax.plot(df['time'], df['T_mean_interface'], label='Mean', color='purple', linewidth=2)
    ax.plot(df['time'], df['T_min_interface'], label='Min', color='lavender', linewidth=1.5, alpha=0.7)
    ax.plot(df['time'], df['T_max_interface'], label='Max', color='darkviolet', linewidth=1.5, alpha=0.7)
    ax.axhline(y=273.15, color='red', linestyle='--', alpha=0.5, label='T_melt')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Temperature at Ice-Water Interface')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 7: Temperature coupling strength
    ax = axes[2, 0]
    ax.plot(df['time'], df['max_temperature_coupling'], label='Max', color='red', linewidth=2)
    ax.plot(df['time'], df['mean_temperature_coupling'], label='Mean', color='orange', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Coupling Strength')
    ax.set_title('Temperature Coupling Strength')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Subcooling and superheating
    ax = axes[2, 1]
    ax.plot(df['time'], df['subcooled_water_fraction'], label='Subcooled Water', color='cyan', linewidth=2)
    ax.plot(df['time'], df['superheated_ice_fraction'], label='Superheated Ice', color='magenta', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction')
    ax.set_title('Subcooling and Superheating')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Interface length
    ax = axes[2, 2]
    ax.plot(df['time'], df['interface_length'], label='Interface Length', color='brown', linewidth=2)
    if baseline_aligned is not None and 'interface_length_baseline' in baseline_aligned.columns:
        ax.plot(df['time'], baseline_aligned['interface_length_baseline'], 
               label='Baseline', linestyle='--', alpha=0.6, linewidth=1.5, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Interface Length')
    ax.set_title('Ice-Water Interface Length')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ice_phase_transition_plots.png')
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved ice phase transition plots to {output_path}")
    finally:
        plt.close(fig)


def plot_temperature_evolution(temp_file, output_dir, baseline_df=None):
    """Plot temperature evolution statistics from temperature_evolution.csv.
    
    Args:
        temp_file (str): Path to temperature_evolution.csv
        output_dir (str): Directory to save plots
        baseline_df (pd.DataFrame, optional): Baseline temperature data for overlay
    """
    if not os.path.exists(temp_file):
        print(f"Warning: {temp_file} not found, skipping temperature plots")
        return
    
    df = pd.read_csv(temp_file)
    
    # Align baseline data by step
    baseline_aligned = align_by_step(df, baseline_df) if baseline_df is not None else None
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Temperature Evolution Statistics', fontsize=16, fontweight='bold')
    
    # Plot 1: Temperature statistics
    ax = axes[0, 0]
    ax.plot(df['time'], df['T_min'], label='Min', color='blue', linewidth=1.5, alpha=0.7)
    ax.plot(df['time'], df['T_max'], label='Max', color='red', linewidth=1.5, alpha=0.7)
    ax.plot(df['time'], df['T_mean'], label='Mean', color='green', linewidth=2)
    ax.axhline(y=273.15, color='black', linestyle='--', alpha=0.5, label='T_melt')
    if baseline_aligned is not None and 'T_mean_baseline' in baseline_aligned.columns:
        ax.plot(df['time'], baseline_aligned['T_mean_baseline'], 
               label='Baseline (mean)', linestyle='--', alpha=0.6, linewidth=1.5, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Temperature Statistics')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Temperature gradient
    ax = axes[0, 1]
    ax.plot(df['time'], df['T_gradient_max'], label='Max Gradient', color='purple', linewidth=2)
    ax.plot(df['time'], df['T_gradient_mean'], label='Mean Gradient', color='orange', linewidth=2)
    if baseline_aligned is not None and 'T_gradient_mean_baseline' in baseline_aligned.columns:
        ax.plot(df['time'], baseline_aligned['T_gradient_mean_baseline'], 
               label='Baseline (mean)', linestyle='--', alpha=0.6, linewidth=1.5, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature Gradient')
    ax.set_title('Temperature Gradient')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Boundary temperatures
    ax = axes[0, 2]
    ax.plot(df['time'], df['T_bottom_mean'], label='Bottom (mean)', color='blue', linewidth=2)
    ax.plot(df['time'], df['T_bottom_min'], label='Bottom (min)', color='lightblue', linewidth=1.5, alpha=0.7)
    ax.plot(df['time'], df['T_top_mean'], label='Top (mean)', color='red', linewidth=2)
    ax.plot(df['time'], df['T_top_min'], label='Top (min)', color='pink', linewidth=1.5, alpha=0.7)
    ax.axhline(y=273.15, color='black', linestyle='--', alpha=0.5, label='T_melt')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Boundary Temperatures')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Temperature difference from melting point
    ax = axes[1, 0]
    ax.plot(df['time'], df['T_diff_bottom_melt'], label='Bottom - T_melt', color='blue', linewidth=2)
    ax.plot(df['time'], df['T_diff_top_melt'], label='Top - T_melt', color='red', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature Difference (K)')
    ax.set_title('Temperature Difference from Melting Point')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Latent heat rate
    ax = axes[1, 1]
    ax.plot(df['time'], df['latent_heat_rate'], label='Latent Heat Rate', color='orange', linewidth=2)
    if baseline_aligned is not None and 'latent_heat_rate_baseline' in baseline_aligned.columns:
        ax.plot(df['time'], baseline_aligned['latent_heat_rate_baseline'], 
               label='Baseline', linestyle='--', alpha=0.6, linewidth=1.5, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Latent Heat Rate')
    ax.set_title('Latent Heat Release/Absorption Rate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Subcooled and superheated regions
    ax = axes[1, 2]
    if 'subcooled_fraction' in df.columns:
        ax.plot(df['time'], df['subcooled_fraction'], label='Subcooled Region', color='cyan', linewidth=2)
        ax.plot(df['time'], df['superheated_fraction'], label='Superheated Region', color='magenta', linewidth=2)
    elif 'subcooled_region_fraction' in df.columns:
        ax.plot(df['time'], df['subcooled_region_fraction'], label='Subcooled Region', color='cyan', linewidth=2)
        ax.plot(df['time'], df['superheated_region_fraction'], label='Superheated Region', color='magenta', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction')
    ax.set_title('Subcooled and Superheated Regions')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'temperature_evolution_plots.png')
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved temperature evolution plots to {output_path}")
    finally:
        plt.close(fig)
    
    # Create additional diagnostic plots if columns exist
    if 'T_bottom_cell0' in df.columns:
        plot_temperature_diagnostics(temp_file, output_dir, baseline_df)


def plot_temperature_diagnostics(temp_file, output_dir, baseline_df=None):
    """Plot detailed temperature diagnostics for boundary jump analysis.
    
    Args:
        temp_file (str): Path to temperature_evolution.csv
        output_dir (str): Directory to save plots
        baseline_df (pd.DataFrame, optional): Baseline temperature data for overlay
    """
    if not os.path.exists(temp_file):
        return
    
    df = pd.read_csv(temp_file)
    
    # Align baseline data by step
    baseline_aligned = align_by_step(df, baseline_df) if baseline_df is not None else None
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Temperature Diagnostics: Boundary Jump Analysis', fontsize=16, fontweight='bold')
    
    # Row 1: Boundary Cell Temperatures
    # Plot 1: Bottom boundary cells (first 3)
    ax = axes[0, 0]
    if 'T_bottom_cell0' in df.columns:
        ax.plot(df['time'], df['T_bottom_cell0'], label='Cell 0 (boundary)', color='blue', linewidth=2)
        ax.plot(df['time'], df['T_bottom_cell1'], label='Cell 1', color='cyan', linewidth=2)
        ax.plot(df['time'], df['T_bottom_cell2'], label='Cell 2', color='lightblue', linewidth=2)
    ax.axhline(y=273.15, color='black', linestyle='--', alpha=0.5, label='T_melt')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Bottom Boundary Cells (First 3)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Top boundary cells (last 3)
    ax = axes[0, 1]
    if 'T_top_cell0' in df.columns:
        ax.plot(df['time'], df['T_top_cell0'], label='Cell 0 (boundary)', color='red', linewidth=2)
        ax.plot(df['time'], df['T_top_cell1'], label='Cell -1', color='orange', linewidth=2)
        ax.plot(df['time'], df['T_top_cell2'], label='Cell -2', color='pink', linewidth=2)
    ax.axhline(y=273.15, color='black', linestyle='--', alpha=0.5, label='T_melt')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Top Boundary Cells (Last 3)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Temperature gradient at boundaries
    ax = axes[0, 2]
    if 'T_gradient_bottom' in df.columns:
        ax.plot(df['time'], df['T_gradient_bottom'], label='Bottom Gradient', color='blue', linewidth=2)
        ax.plot(df['time'], df['T_gradient_top'], label='Top Gradient', color='red', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature Gradient (K/m)')
    ax.set_title('Temperature Gradient at Boundaries')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Row 2: Diffusion Diagnostics
    # Plot 4: Laplacian
    ax = axes[1, 0]
    if 'laplacian_max' in df.columns:
        ax.plot(df['time'], df['laplacian_max'], label='Max Laplacian', color='purple', linewidth=2)
        ax.plot(df['time'], df['laplacian_mean'], label='Mean Laplacian', color='violet', linewidth=2)
        if 'laplacian_bottom_cell1' in df.columns:
            ax.plot(df['time'], df['laplacian_bottom_cell1'], label='Bottom Cell 1', 
                   color='blue', linewidth=2, linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Laplacian (K/m²)')
    ax.set_title('Temperature Laplacian (Diffusion Driving Force)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Diffusion term
    ax = axes[1, 1]
    if 'diffusion_term_max' in df.columns:
        ax.plot(df['time'], df['diffusion_term_max'], label='Max Diffusion', color='green', linewidth=2)
        ax.plot(df['time'], df['diffusion_term_mean'], label='Mean Diffusion', color='lightgreen', linewidth=2)
        if 'diffusion_term_bottom_cell1' in df.columns:
            ax.plot(df['time'], df['diffusion_term_bottom_cell1'], label='Bottom Cell 1', 
                   color='blue', linewidth=2, linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Diffusion Term (K/s)')
    ax.set_title('Diffusion Term: α∇²T')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Thermal diffusivity
    ax = axes[1, 2]
    if 'alpha_bottom_mean' in df.columns:
        ax.plot(df['time'], df['alpha_bottom_mean'], label='Bottom Mean α', color='blue', linewidth=2)
        if 'alpha_bottom_cell1' in df.columns:
            ax.plot(df['time'], df['alpha_bottom_cell1'], label='Bottom Cell 1 α', 
                   color='cyan', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Thermal Diffusivity (m²/s)')
    ax.set_title('Thermal Diffusivity at Bottom')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Row 3: Latent Heat and Advection
    # Plot 7: Latent heat term
    ax = axes[2, 0]
    if 'latent_heat_max' in df.columns:
        ax.plot(df['time'], df['latent_heat_max'], label='Max Latent Heat', color='orange', linewidth=2)
        ax.plot(df['time'], df['latent_heat_mean'], label='Mean Latent Heat', color='yellow', linewidth=2)
        if 'latent_heat_bottom_cell1' in df.columns:
            ax.plot(df['time'], df['latent_heat_bottom_cell1'], label='Bottom Cell 1', 
                   color='red', linewidth=2, linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Latent Heat Term (K/s)')
    ax.set_title('Latent Heat: (L/c_p)(1/2)dψ/dt')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Phase change rate
    ax = axes[2, 1]
    if 'dpsi_dt_max' in df.columns:
        ax.plot(df['time'], df['dpsi_dt_max'], label='Max dψ/dt', color='purple', linewidth=2)
        ax.plot(df['time'], df['dpsi_dt_mean'], label='Mean dψ/dt', color='violet', linewidth=2)
        if 'dpsi_dt_bottom_cell1' in df.columns:
            ax.plot(df['time'], df['dpsi_dt_bottom_cell1'], label='Bottom Cell 1', 
                   color='blue', linewidth=2, linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Phase Change Rate (1/s)')
    ax.set_title('Phase Change Rate: dψ/dt')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Advection term
    ax = axes[2, 2]
    if 'advection_term_max' in df.columns:
        ax.plot(df['time'], df['advection_term_max'], label='Max Advection', color='brown', linewidth=2)
        ax.plot(df['time'], df['advection_term_mean'], label='Mean Advection', color='tan', linewidth=2)
        if 'advection_term_bottom_cell1' in df.columns:
            ax.plot(df['time'], df['advection_term_bottom_cell1'], label='Bottom Cell 1', 
                   color='blue', linewidth=2, linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Advection Term (K/s)')
    ax.set_title('Advection: -A(ψ)u·∇T')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'temperature_diagnostics_plots.png')
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved temperature diagnostics plots to {output_path}")
    finally:
        plt.close(fig)


def main():
    """Main function to plot all telemetry data."""
    parser = argparse.ArgumentParser(description='Plot telemetry data from CSV files')
    parser.add_argument('experiment_dir', type=str, 
                       help='Path to experiment directory containing CSV files')
    parser.add_argument('--baseline', type=str, default=None,
                       help='Path to baseline experiment directory for overlay')
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    
    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return
    
    # Create visualization directory
    visualization_dir = experiment_dir / 'visualization'
    visualization_dir.mkdir(exist_ok=True)
    
    print(f"Plotting telemetry data from: {experiment_dir}")
    
    # Load baseline data if provided
    baseline_stats = None
    baseline_boundary = None
    baseline_ppe = None
    
    baseline_dir = args.baseline
    if baseline_dir is None:
        # Try to load from config
        config_file = experiment_dir / 'simulation_parameters.json'
        if config_file.exists():
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
                baseline_dir = config.get('baseline_experiment', None)
    
    if baseline_dir and os.path.exists(baseline_dir):
        baseline_stats_path = os.path.join(baseline_dir, 'statistics.csv')
        baseline_boundary_path = os.path.join(baseline_dir, 'boundary_statistics.csv')
        baseline_ppe_path = os.path.join(baseline_dir, 'ppe_updates.csv')
        
        if os.path.exists(baseline_stats_path):
            baseline_stats = pd.read_csv(baseline_stats_path)
            print(f"Loaded baseline statistics from {baseline_dir}")
        if os.path.exists(baseline_boundary_path):
            baseline_boundary = pd.read_csv(baseline_boundary_path)
        if os.path.exists(baseline_ppe_path):
            baseline_ppe = pd.read_csv(baseline_ppe_path)
    
    # Plot statistics
    statistics_file = experiment_dir / 'statistics.csv'
    plot_statistics(str(statistics_file), str(visualization_dir), baseline_df=baseline_stats)
    
    # Plot boundary statistics
    boundary_file = experiment_dir / 'boundary_statistics.csv'
    plot_boundary_statistics(str(boundary_file), str(visualization_dir), baseline_df=baseline_boundary)
    
    # Plot PPE updates
    ppe_file = experiment_dir / 'ppe_updates.csv'
    plot_ppe_updates(str(ppe_file), str(visualization_dir), baseline_df=baseline_ppe)
    
    # Plot ice phase transition (if available)
    ice_file = experiment_dir / 'ice_phase_transition.csv'
    baseline_ice = None
    if baseline_dir and os.path.exists(baseline_dir):
        baseline_ice_path = os.path.join(baseline_dir, 'ice_phase_transition.csv')
        if os.path.exists(baseline_ice_path):
            baseline_ice = pd.read_csv(baseline_ice_path)
    plot_ice_phase_transition(str(ice_file), str(visualization_dir), baseline_df=baseline_ice)
    
    # Plot temperature evolution (if available)
    temp_file = experiment_dir / 'temperature_evolution.csv'
    baseline_temp = None
    if baseline_dir and os.path.exists(baseline_dir):
        baseline_temp_path = os.path.join(baseline_dir, 'temperature_evolution.csv')
        if os.path.exists(baseline_temp_path):
            baseline_temp = pd.read_csv(baseline_temp_path)
    plot_temperature_evolution(str(temp_file), str(visualization_dir), baseline_df=baseline_temp)
    
    # Plot diagnostic data if available
    if os.path.exists(temp_file):
        df_temp = pd.read_csv(temp_file)
        if 'T_bottom_cell0' in df_temp.columns:
            plot_temperature_diagnostics(str(temp_file), str(visualization_dir), baseline_df=baseline_temp)
    
    print(f"\nAll telemetry plots generated successfully in {visualization_dir}/")


if __name__ == "__main__":
    main()
