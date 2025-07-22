#!/usr/bin/env python3
"""
Monitor training progress for uTooth experiments
"""

import os
import sys
import json
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
import time

def get_experiment_status(run_dir):
    """Get the status of a training experiment"""
    config_path = os.path.join(run_dir, 'config.json')
    results_path = os.path.join(run_dir, 'results_summary.json')
    state_path = os.path.join(run_dir, 'experiment_state.json')
    
    if not os.path.exists(config_path):
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    status = {
        'experiment_name': config.get('experiment_name', 'unknown'),
        'start_time': config.get('start_time', 'unknown'),
        'max_epochs': config.get('max_epochs', 50),
        'n_folds': config.get('n_folds', 5),
        'status': 'running',
        'completed_folds': 0,
        'failed_folds': 0,
        'current_fold': None,
        'resume_count': config.get('resume_count', 0),
        'progress': 0.0,
        'can_resume': False
    }
    
    # Check experiment state
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            state = json.load(f)
            status['completed_folds'] = len(state.get('completed_folds', []))
            status['failed_folds'] = len(state.get('failed_folds', []))
            status['current_fold'] = state.get('current_fold')
            status['resume_count'] = state.get('resume_count', 0)
            status['status'] = state.get('status', 'running')
            
            # Check if experiment can be resumed
            if status['status'] in ['running', 'interrupted'] and status['completed_folds'] < status['n_folds']:
                status['can_resume'] = True
    
    # Calculate progress
    if status['completed_folds'] > 0:
        status['progress'] = status['completed_folds'] / status['n_folds'] * 100
    
    # Check for checkpoint availability
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    if os.path.exists(checkpoints_dir):
        for fold_idx in range(status['n_folds']):
            fold_ckpt_dir = os.path.join(checkpoints_dir, f'fold_{fold_idx}')
            if os.path.exists(fold_ckpt_dir) and os.listdir(fold_ckpt_dir):
                status['can_resume'] = True
                break
    
    # Check if fully completed
    if os.path.exists(results_path):
        status['status'] = 'completed'
        status['progress'] = 100.0
        
        with open(results_path, 'r') as f:
            results = json.load(f)
            status['mean_val_loss'] = results['summary']['mean_val_loss']
            status['completed_at'] = results['summary']['completed_at']
    
    return status

def list_experiments():
    """List all experiments"""
    runs_dir = Path('outputs/runs')
    if not runs_dir.exists():
        print("No experiments found. The outputs/runs directory doesn't exist.")
        return
    
    experiments = []
    for exp_dir in runs_dir.iterdir():
        if exp_dir.is_dir():
            status = get_experiment_status(str(exp_dir))
            if status:
                experiments.append(status)
    
    if not experiments:
        print("No experiments found.")
        return
    
    # Sort by start time (newest first)
    experiments.sort(key=lambda x: x.get('start_time', ''), reverse=True)
    
    print(f"{'Experiment Name':<30} {'Status':<12} {'Progress':<10} {'Resume':<8} {'Mean Val Loss':<15} {'Start Time':<20}")
    print("-" * 108)
    
    for exp in experiments:
        mean_loss = f"{exp.get('mean_val_loss', 'N/A'):.4f}" if exp.get('mean_val_loss') else 'N/A'
        start_time = exp.get('start_time', 'Unknown')[:16].replace('T', ' ')
        progress = f"{exp['progress']:.1f}%"
        
        can_resume = '✅' if exp.get('can_resume', False) else '❌'
        print(f"{exp['experiment_name']:<30} {exp['status']:<12} {progress:<10} {can_resume:<8} {mean_loss:<15} {start_time:<20}")

def monitor_experiment(experiment_name, refresh_interval=10):
    """Monitor a specific experiment"""
    run_dir = f"outputs/runs/{experiment_name}"
    
    if not os.path.exists(run_dir):
        print(f"Experiment '{experiment_name}' not found.")
        return
    
    print(f"Monitoring experiment: {experiment_name}")
    print(f"Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            status = get_experiment_status(run_dir)
            if not status:
                print("Could not read experiment status.")
                break
            
            os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
            
            print(f"Experiment: {experiment_name}")
            print(f"Status: {status['status']}")
            print(f"Progress: {status['progress']:.1f}% ({status['completed_folds']}/{status['n_folds']} folds)")
            print(f"Start Time: {status['start_time']}")
            
            if status['status'] == 'completed':
                print(f"Completed At: {status.get('completed_at', 'Unknown')}")
                print(f"Mean Validation Loss: {status.get('mean_val_loss', 'N/A'):.4f}")
                break
            
            # Show current fold progress if available
            current_fold_dir = None
            for i in range(status['n_folds']):
                fold_dir = os.path.join(run_dir, 'metrics', f'fold_{i}')
                if os.path.exists(fold_dir):
                    metrics_file = os.path.join(fold_dir, 'metrics.csv')
                    if os.path.exists(metrics_file):
                        try:
                            df = pd.read_csv(metrics_file)
                            if not df.empty:
                                current_epoch = df['epoch'].max() + 1
                                latest_val_loss = df[df['epoch'] == df['epoch'].max()]['val_loss'].iloc[0]
                                print(f"Fold {i}: Epoch {current_epoch}/{status['max_epochs']}, Val Loss: {latest_val_loss:.4f}")
                        except:
                            pass
            
            print(f"\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("Press Ctrl+C to stop monitoring")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

def show_experiment_details(experiment_name):
    """Show detailed information about an experiment"""
    run_dir = f"outputs/runs/{experiment_name}"
    
    if not os.path.exists(run_dir):
        print(f"Experiment '{experiment_name}' not found.")
        return
    
    status = get_experiment_status(run_dir)
    if not status:
        print("Could not read experiment status.")
        return
    
    print(f"Experiment Details: {experiment_name}")
    print("=" * 50)
    print(f"Status: {status['status']}")
    print(f"Progress: {status['progress']:.1f}%")
    print(f"Completed Folds: {status['completed_folds']}/{status['n_folds']}")
    print(f"Max Epochs: {status['max_epochs']}")
    print(f"Start Time: {status['start_time']}")
    
    if status['status'] == 'completed':
        print(f"Completed At: {status.get('completed_at', 'Unknown')}")
        print(f"Mean Validation Loss: {status.get('mean_val_loss', 'N/A'):.4f}")
        
        # Show fold details if available
        results_path = os.path.join(run_dir, 'results_summary.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
                
            print("\nFold Results:")
            print("-" * 50)
            for i, fold_result in enumerate(results['fold_results']):
                print(f"Fold {i+1}: Val Loss = {fold_result['best_val_loss']:.4f}, "
                      f"Best Epoch = {fold_result['best_epoch']}, "
                      f"Time = {fold_result['training_time_seconds']/60:.1f} min")

def resume_experiment(experiment_name):
    """Show how to resume an experiment"""
    run_dir = f"outputs/runs/{experiment_name}"
    
    if not os.path.exists(run_dir):
        print(f"Experiment '{experiment_name}' not found.")
        return
    
    status = get_experiment_status(run_dir)
    if not status:
        print("Could not read experiment status.")
        return
    
    if not status['can_resume']:
        print(f"Experiment '{experiment_name}' cannot be resumed.")
        print(f"Status: {status['status']}")
        if status['status'] == 'completed':
            print("Experiment is already completed.")
        return
    
    print(f"Experiment '{experiment_name}' can be resumed:")
    print(f"  Status: {status['status']}")
    print(f"  Progress: {status['progress']:.1f}% ({status['completed_folds']}/{status['n_folds']} folds)")
    if status['current_fold'] is not None:
        print(f"  Current fold: {status['current_fold'] + 1}")
    if status['failed_folds'] > 0:
        print(f"  Failed folds: {status['failed_folds']}")
    if status['resume_count'] > 0:
        print(f"  Previous resumes: {status['resume_count']}")
    
    print(f"\nTo resume this experiment, run:")
    print(f"  python scripts/train.py --resume --experiment_name {experiment_name}")
    print(f"\nOr to auto-resume without confirmation:")
    print(f"  python scripts/train.py --auto_resume --experiment_name {experiment_name}")

def main():
    parser = argparse.ArgumentParser(description='Monitor uTooth training experiments')
    parser.add_argument('command', choices=['list', 'monitor', 'details', 'resume'], 
                        help='Command to execute')
    parser.add_argument('--experiment', '-e', type=str, 
                        help='Experiment name (for monitor/details/resume commands)')
    parser.add_argument('--refresh', '-r', type=int, default=10,
                        help='Refresh interval in seconds (for monitor command)')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_experiments()
    elif args.command == 'monitor':
        if not args.experiment:
            print("Error: --experiment is required for monitor command")
            return
        monitor_experiment(args.experiment, args.refresh)
    elif args.command == 'details':
        if not args.experiment:
            print("Error: --experiment is required for details command")
            return
        show_experiment_details(args.experiment)
    elif args.command == 'resume':
        if not args.experiment:
            print("Error: --experiment is required for resume command")
            return
        resume_experiment(args.experiment)

if __name__ == "__main__":
    main()