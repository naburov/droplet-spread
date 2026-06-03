#!/usr/bin/env python3
"""
Master script to run all diagnostic analyses for droplet spreading simulation.
"""

import os
import sys
import subprocess
import time
import shutil
from pathlib import Path

def run_diagnostic(script_name, experiment_dir, config_path=None):
    """Run a single diagnostic script.
    
    Args:
        script_name (str): Name of the diagnostic script
        experiment_dir (str): Path to experiment directory
        config_path (str, optional): Path to config file (for first_step_visualization)
    
    Returns:
        bool: True if successful, False otherwise
    """
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ Script {script_name} not found!")
        return False
    
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Special handling for first_step_visualization which needs config
        if script_name == "first_step_visualization.py":
            # Try to find config in experiment directory
            if config_path is None:
                config_path = os.path.join(experiment_dir, 'simulation_parameters.json')
            if not os.path.exists(config_path):
                # Try to find config in configs directory
                configs_dir = os.path.join(os.path.dirname(os.path.dirname(experiment_dir)), 'configs')
                if os.path.exists(configs_dir):
                    config_files = [f for f in os.listdir(configs_dir) if f.endswith('.json')]
                    if config_files:
                        config_path = os.path.join(configs_dir, config_files[0])
            
            if not os.path.exists(config_path):
                print(f"⚠️  Skipping {script_name}: config file not found")
                return False
            
            output_dir = os.path.join(experiment_dir, 'diagnostics')
            result = subprocess.run([sys.executable, script_path, config_path, output_dir], 
                                  capture_output=True, text=True, timeout=300)
        else:
            result = subprocess.run([sys.executable, script_path, experiment_dir], 
                                  capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully in {end_time - start_time:.1f}s")
            if result.stdout:
                print("Output:")
                print(result.stdout)
            return True
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            if result.stderr:
                print("Error:")
                print(result.stderr)
            return False
    
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    """Main function to run all diagnostics."""
    if len(sys.argv) != 2:
        print("Usage: python run_all_diagnostics.py <experiment_directory>")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    
    if not os.path.exists(experiment_dir):
        print(f"Error: Directory {experiment_dir} does not exist!")
        sys.exit(1)
    
    print(f"Running all diagnostics for: {experiment_dir}")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List of diagnostic scripts to run
    diagnostics = [
        "first_step_visualization.py",  # First step diagnostics (if config available)
        "divergence_analysis.py",
        "energy_analysis.py", 
        "mass_conservation.py",
        "pressure_curvature.py",
        "chemical_potential.py",
        "force_budget.py",
        "contact_angle.py",
        "cfl_analysis.py"
    ]
    
    # Generate diagnostic guide for this experiment
    print(f"\n{'='*60}")
    print("Generating diagnostic reference guide...")
    print(f"{'='*60}")
    
    try:
        # Create diagnostics directory if it doesn't exist
        diagnostics_dir = os.path.join(experiment_dir, 'diagnostics')
        os.makedirs(diagnostics_dir, exist_ok=True)
        
        # Generate the diagnostic guide
        guide_script = os.path.join(os.path.dirname(__file__), "diagnostic_guide.py")
        if os.path.exists(guide_script):
            result = subprocess.run([sys.executable, guide_script, "--no-display"], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Copy the generated guide to the experiment's diagnostics folder
                guide_source = "diagnostic_guide.png"
                guide_dest = os.path.join(diagnostics_dir, "diagnostic_reference_guide.png")
                
                if os.path.exists(guide_source):
                    shutil.copy2(guide_source, guide_dest)
                    print(f"✅ Diagnostic reference guide saved to: {guide_dest}")
                else:
                    print("⚠️  Diagnostic guide generated but file not found")
            else:
                print(f"❌ Failed to generate diagnostic guide: {result.stderr}")
        else:
            print("⚠️  Diagnostic guide script not found")
            
    except Exception as e:
        print(f"❌ Error generating diagnostic guide: {e}")
    
    # Run diagnostics
    results = {}
    successful = 0
    failed = 0
    
    # Try to find config for first_step_visualization
    config_path = os.path.join(experiment_dir, 'simulation_parameters.json')
    if not os.path.exists(config_path):
        config_path = None
    
    for diagnostic in diagnostics:
        success = run_diagnostic(diagnostic, experiment_dir, config_path=config_path)
        results[diagnostic] = success
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    print(f"Total diagnostics: {len(diagnostics)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(diagnostics)*100:.1f}%")
    
    print(f"\nDetailed results:")
    for diagnostic, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {diagnostic}: {status}")
    
    # Check for critical issues
    print(f"\n{'='*60}")
    print("CRITICAL ISSUES CHECK")
    print(f"{'='*60}")
    
    # Check if any critical diagnostics failed
    critical_diagnostics = ["divergence_analysis.py", "mass_conservation.py", "cfl_analysis.py"]
    critical_failures = [d for d in critical_diagnostics if not results.get(d, False)]
    
    if critical_failures:
        print("⚠️  WARNING: Critical diagnostics failed:")
        for d in critical_failures:
            print(f"  - {d}")
    else:
        print("✅ All critical diagnostics passed")
    
    # Check output files
    print(f"\n{'='*60}")
    print("OUTPUT FILES CHECK")
    print(f"{'='*60}")
    
    expected_files = [
        "first_step_comprehensive.png",
        "divergence_analysis.png",
        "energy_analysis.png",
        "mass_conservation_analysis.png", 
        "pressure_curvature_analysis.png",
        "force_ratios_analysis.png",
        "chemical_potential_analysis.png",
        "force_budget_analysis.png",
        "contact_angle_analysis.png",
        "cfl_analysis.png",
        "diagnostic_reference_guide.png"
    ]
    
    existing_files = []
    missing_files = []
    
    # Check in diagnostics folder
    diagnostics_dir = os.path.join(experiment_dir, 'diagnostics')
    
    for file in expected_files:
        file_path = os.path.join(diagnostics_dir, file)
        if os.path.exists(file_path):
            existing_files.append(file)
        else:
            missing_files.append(file)
    
    print(f"Generated plots: {len(existing_files)}/{len(expected_files)}")
    for file in existing_files:
        print(f"  ✅ {file}")
    
    if missing_files:
        print("Missing plots:")
        for file in missing_files:
            print(f"  ❌ {file}")
    
    print(f"\nCompleted at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return appropriate exit code
    if failed == 0:
        print("\n🎉 All diagnostics completed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {failed} diagnostic(s) failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
