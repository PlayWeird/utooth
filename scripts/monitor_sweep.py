#!/usr/bin/env python3
"""
Monitor and visualize hyperparameter sweep progress.

Usage:
  python scripts/monitor_sweep.py --sweep_dir outputs/sweeps/sweep_name --study_name utooth_sweep
  python scripts/monitor_sweep.py --sweep_dir outputs/sweeps/sweep_name --auto-detect
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.sweep.utils.monitoring import monitor_sweep_progress, SweepMonitor
import yaml
import json


def find_study_name(sweep_dir: Path) -> str:
    """Auto-detect study name from sweep configuration."""
    
    # Try to find from sweep config
    config_file = sweep_dir / "sweep_config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
            if 'study' in config and 'name' in config['study']:
                return config['study']['name']
    
    # Try to find from statistics file
    stats_file = sweep_dir / "reports" / "sweep_statistics.json"
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
            if 'study_name' in stats:
                return stats['study_name']
    
    # Default fallback
    return sweep_dir.name


def main():
    parser = argparse.ArgumentParser(description='Monitor hyperparameter sweep progress')
    parser.add_argument('--sweep_dir', type=str, required=True,
                       help='Path to sweep directory')
    parser.add_argument('--study_name', type=str, default=None,
                       help='Name of Optuna study')
    parser.add_argument('--auto-detect', action='store_true',
                       help='Auto-detect study name from sweep configuration')
    parser.add_argument('--export-format', choices=['csv', 'json', 'excel'], 
                       default='csv', help='Export format for results')
    
    args = parser.parse_args()
    
    sweep_dir = Path(args.sweep_dir)
    
    if not sweep_dir.exists():
        print(f"Error: Sweep directory does not exist: {sweep_dir}")
        return 1
    
    # Determine study name
    study_name = args.study_name
    if args.auto_detect or study_name is None:
        study_name = find_study_name(sweep_dir)
        print(f"Auto-detected study name: {study_name}")
    
    if not study_name:
        print("Error: Could not determine study name. Please specify with --study_name")
        return 1
    
    try:
        monitor_sweep_progress(sweep_dir, study_name)
        print(f"\nMonitoring completed successfully!")
        print(f"Check {sweep_dir}/plots/ for visualizations")
        print(f"Check {sweep_dir}/reports/ for reports")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())