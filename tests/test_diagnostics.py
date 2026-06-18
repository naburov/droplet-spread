#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced diagnostic system.

Updated for pytest: the function used to return True/False (which only emitted
a PytestReturnNotNoneWarning and never failed). It now skips when the fixture
experiment directory / interpreter are not present on this machine, and
asserts on the diagnostics run result when they are.
"""

import subprocess
import os

import pytest

# Fixture data for this test: an experiment directory produced by a simulation
# run, analyzed with the bundled diagnostics runner.
EXPERIMENT_DIR = "experiment_20250915_214517"
DIAGNOSTICS_PYTHON = os.path.join("native", "bin", "python")
DIAGNOSTICS_RUNNER = os.path.join("diagnostics", "run_all_diagnostics.py")


def test_diagnostics():
    """Run the diagnostics pipeline on a stored experiment and check its outputs."""
    if not os.path.exists(EXPERIMENT_DIR):
        pytest.skip(f"experiment directory {EXPERIMENT_DIR} not available")
    if not os.path.exists(DIAGNOSTICS_PYTHON):
        pytest.skip(f"diagnostics interpreter {DIAGNOSTICS_PYTHON} not available")
    if not os.path.exists(DIAGNOSTICS_RUNNER):
        pytest.skip(f"diagnostics runner {DIAGNOSTICS_RUNNER} not available")

    result = subprocess.run(
        [DIAGNOSTICS_PYTHON, DIAGNOSTICS_RUNNER, EXPERIMENT_DIR],
        capture_output=True, text=True, timeout=300,
    )

    assert result.returncode == 0, \
        f"diagnostic system run failed:\n{result.stderr}"

    # Check what files were created
    diagnostics_dir = os.path.join(EXPERIMENT_DIR, "diagnostics")
    assert os.path.exists(diagnostics_dir), "diagnostics output directory missing"

    files = os.listdir(diagnostics_dir)
    png_files = [f for f in files if f.endswith('.png')]
    json_files = [f for f in files if f.endswith('.json')]

    assert png_files, "diagnostics produced no plots"
    assert json_files, "diagnostics produced no data files"

    # The reference guide must be included in the generated plots.
    assert "diagnostic_reference_guide.png" in png_files, \
        "diagnostic reference guide missing from output"


if __name__ == "__main__":
    test_diagnostics()
    print("Diagnostics test passed (or skipped).")
