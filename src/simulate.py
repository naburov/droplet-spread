#!/usr/bin/env python3
"""
Droplet spreading simulation script - clean and working version.
"""
import argparse
import sys
import os
from datetime import datetime

# Add current directory to path to import jax_main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='Droplet spreading simulation')
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON format)')
    parser.add_argument('--output', type=str, help='Path to output directory')
    args = parser.parse_args()
    
    # Generate automatic output directory with timestamp if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Determine prefix based on config file
        config_name = args.config or 'config_template.json'
        if 'water_droplet' in config_name:
            prefix = 'water_droplet'
        elif 'air_bubble' in config_name:
            prefix = 'air_bubble'
        else:
            prefix = 'experiment'
        args.output = f'experiments/{prefix}_{timestamp}'
    
    # Ensure experiments directory exists
    os.makedirs('experiments', exist_ok=True)
    
    # Import and run the existing main function
    from jax_main import main as jax_main
    sys.argv = ['jax_main.py', '--config', args.config or 'configs/config_template.json']
    sys.argv.extend(['--output', args.output])
    
    jax_main()

if __name__ == "__main__":
    main()
