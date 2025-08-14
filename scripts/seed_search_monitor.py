#!/usr/bin/env python3
"""
Real-time monitoring script for seed search progress
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
import argparse


def get_latest_seed_search_dir(base_dir: str = "outputs/seed_search") -> Path:
    """Find the most recent seed search directory"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    
    search_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("seed_search_")])
    return search_dirs[-1] if search_dirs else None


def monitor_progress(output_dir: Path, refresh_interval: int = 5):
    """Monitor seed search progress in real-time"""
    
    print(f"Monitoring seed search in: {output_dir}")
    print("Press Ctrl+C to stop monitoring\n")
    print("="*80)
    
    try:
        while True:
            # Clear screen (works on Unix-like systems)
            print("\033[2J\033[H")
            
            # Print header
            print(f"Seed Search Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            # Load config
            config_file = output_dir / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print(f"Total seeds to evaluate: {config['n_seeds']}")
                print(f"GPUs in use: {config['n_gpus']}")
                print("-"*80)
            
            # Check seed directories
            seeds_dir = output_dir / "seeds"
            if seeds_dir.exists():
                seed_dirs = list(seeds_dir.glob("seed_*"))
                
                completed_seeds = []
                in_progress_seeds = []
                
                for seed_dir in seed_dirs:
                    seed_num = int(seed_dir.name.split('_')[1])
                    results_file = seed_dir / "results.json"
                    
                    if results_file.exists():
                        with open(results_file, 'r') as f:
                            results = json.load(f)
                        if 'mean_dice' in results:
                            completed_seeds.append({
                                'seed': seed_num,
                                'mean_dice': results['mean_dice'],
                                'std_dice': results['std_dice'],
                                'n_folds': results.get('n_valid_folds', 0)
                            })
                    else:
                        # Check fold progress
                        fold_dirs = list(seed_dir.glob("fold_*"))
                        n_folds = len(fold_dirs)
                        in_progress_seeds.append({
                            'seed': seed_num,
                            'folds_completed': n_folds
                        })
                
                print(f"\nProgress: {len(completed_seeds)}/{config['n_seeds']} seeds completed")
                print(f"In progress: {len(in_progress_seeds)} seeds")
                
                # Show in-progress seeds
                if in_progress_seeds:
                    print("\n--- Seeds in Progress ---")
                    for seed_info in sorted(in_progress_seeds, key=lambda x: x['seed']):
                        print(f"  Seed {seed_info['seed']}: {seed_info['folds_completed']}/10 folds")
                
                # Show completed seeds (top 10)
                if completed_seeds:
                    print("\n--- Top Performing Seeds ---")
                    completed_seeds.sort(key=lambda x: x['mean_dice'], reverse=True)
                    
                    for i, seed_info in enumerate(completed_seeds[:10], 1):
                        print(f"  {i}. Seed {seed_info['seed']}: "
                              f"Dice = {seed_info['mean_dice']:.4f} ± {seed_info['std_dice']:.4f} "
                              f"({seed_info['n_folds']} folds)")
                    
                    # Overall statistics
                    all_dice = [s['mean_dice'] for s in completed_seeds]
                    print(f"\n--- Overall Statistics ---")
                    print(f"  Best Dice: {max(all_dice):.4f}")
                    print(f"  Mean Dice: {sum(all_dice)/len(all_dice):.4f}")
                    print(f"  Worst Dice: {min(all_dice):.4f}")
                
                # Check for errors
                error_count = 0
                for seed_dir in seed_dirs:
                    log_file = seed_dir / "seed.log"
                    if log_file.exists():
                        with open(log_file, 'r') as f:
                            if 'ERROR' in f.read():
                                error_count += 1
                
                if error_count > 0:
                    print(f"\n⚠️  Errors detected in {error_count} seed(s)")
                
            else:
                print("Waiting for seed search to start...")
            
            print("\n" + "="*80)
            print(f"Refreshing in {refresh_interval} seconds...")
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        
        # Final summary
        reports_dir = output_dir / "reports"
        if reports_dir.exists():
            final_results = reports_dir / "final_results.json"
            if final_results.exists():
                print("\n--- Final Results Available ---")
                print(f"View results at: {final_results}")


def main():
    parser = argparse.ArgumentParser(description="Monitor seed search progress")
    parser.add_argument('--dir', type=str, help='Specific seed search directory to monitor')
    parser.add_argument('--interval', type=int, default=5, help='Refresh interval in seconds')
    
    args = parser.parse_args()
    
    if args.dir:
        output_dir = Path(args.dir)
    else:
        output_dir = get_latest_seed_search_dir()
    
    if not output_dir or not output_dir.exists():
        print("Error: No seed search directory found!")
        sys.exit(1)
    
    monitor_progress(output_dir, args.interval)


if __name__ == "__main__":
    main()