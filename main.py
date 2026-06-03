"""
Main entry point for the two-phase (droplet) simulation framework.

Runs incompressible two-phase flow with phase field, surface tension,
contact angle, and optional ice-water transition. For single-phase
channel flow (e.g. Poiseuille, no droplet), use run_staggered_flow.py instead.
"""

import argparse
import csv
import glob
import os
from pathlib import Path
import jax


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


jax.config.update('jax_enable_x64', _env_flag("JAX_ENABLE_X64", True))

from config.config_loader import load_config
from config.bc_compatibility import check_bc_compatibility
from simulation import run_simulation


def _find_latest_checkpoint(experiment_dir):
    checkpoint_pattern = os.path.join(experiment_dir, 'checkpoints', 'checkpoint_*.npz')
    checkpoints = sorted(glob.glob(checkpoint_pattern))
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found under {checkpoint_pattern}')
    return checkpoints[-1]


def _read_resume_time(experiment_dir):
    stats_path = Path(experiment_dir) / 'statistics.csv'
    if not stats_path.exists():
        return None, None

    cleaned_lines = []
    with stats_path.open('rb') as handle:
        for raw_line in handle:
            line = raw_line.replace(b'\x00', b'').decode('utf-8', errors='ignore').strip()
            if line:
                cleaned_lines.append(line)

    if len(cleaned_lines) < 2:
        return None, None

    last_row = None
    reader = csv.DictReader(cleaned_lines)
    for row in reader:
        time_value = row.get('time')
        dt_value = row.get('dt')
        if time_value in (None, '') or dt_value in (None, ''):
            continue
        last_row = row

    if not last_row:
        return None, None

    time_value = last_row.get('time')
    dt_value = last_row.get('dt')
    return (
        float(time_value) if time_value not in (None, '') else None,
        float(dt_value) if dt_value not in (None, '') else None,
    )


def _prepare_resume(resume_dir):
    experiment_dir = Path(resume_dir).resolve()
    config_path = experiment_dir / 'simulation_parameters.json'
    if not config_path.exists():
        raise FileNotFoundError(f'Missing config for resume: {config_path}')

    latest_checkpoint = _find_latest_checkpoint(str(experiment_dir))
    config = load_config(str(config_path))
    resume_time, resume_dt = _read_resume_time(str(experiment_dir))
    config['restart']['restart_from'] = latest_checkpoint
    config['_resume'] = {
        'experiment_dir': str(experiment_dir),
        'time': resume_time,
        'dt': resume_dt,
    }
    return config, str(experiment_dir), latest_checkpoint


def main():
    """Run two-phase droplet spreading simulation."""
    parser = argparse.ArgumentParser(
        description='Two-phase droplet spreading simulation (phase field, surface tension, contact angle).',
        epilog='Example: PYTHONPATH=src python main.py --config configs/config_template.json',
    )
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--resume', type=str, help='Resume from an experiment directory using its latest checkpoint')
    parser.add_argument('--output', type=str, help='Output directory (default: experiment_<timestamp>)')
    args = parser.parse_args()

    if bool(args.config) == bool(args.resume):
        parser.error('Provide exactly one of --config or --resume')

    if args.resume:
        if args.output:
            parser.error('--output cannot be used with --resume; resumed runs always write into the same directory')
        config, output_dir, latest_checkpoint = _prepare_resume(args.resume)
        print(f'Resuming experiment in {output_dir}')
        print(f'Latest checkpoint: {latest_checkpoint}')
    else:
        config = load_config(args.config)
        output_dir = args.output

    check_bc_compatibility(config, verbose=True)
    run_simulation(config, output_dir=output_dir)


if __name__ == "__main__":
    main()
