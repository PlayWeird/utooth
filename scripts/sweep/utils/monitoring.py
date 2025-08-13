"""
Monitoring and visualization tools for hyperparameter sweeps.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import sqlite3


class SweepMonitor:
    """Monitor and visualize hyperparameter sweep progress."""
    
    def __init__(self, sweep_dir: Path):
        self.sweep_dir = Path(sweep_dir)
        self.db_path = self.sweep_dir / "optuna_study.db"
        self.plots_dir = self.sweep_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
    def load_study(self, study_name: str) -> optuna.Study:
        """Load Optuna study from database."""
        storage_url = f"sqlite:///{self.db_path}"
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_url
        )
        return study
    
    def get_trials_dataframe(self, study: optuna.Study) -> pd.DataFrame:
        """Convert study trials to pandas DataFrame."""
        trials_data = []
        
        for trial in study.trials:
            trial_data = {
                'trial_number': trial.number,
                'state': trial.state.name,
                'value': trial.value,
                'datetime_start': trial.datetime_start,
                'datetime_complete': trial.datetime_complete,
            }
            
            # Add parameters
            trial_data.update({f'param_{k}': v for k, v in trial.params.items()})
            
            # Add user attributes
            trial_data.update(trial.user_attrs)
            
            # Calculate duration
            if trial.datetime_start and trial.datetime_complete:
                duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
                trial_data['duration_seconds'] = duration
                trial_data['duration_minutes'] = duration / 60
            
            trials_data.append(trial_data)
        
        return pd.DataFrame(trials_data)
    
    def plot_optimization_history(self, study: optuna.Study, save: bool = True) -> plt.Figure:
        """Plot optimization history."""
        fig = optuna.visualization.plot_optimization_history(study)
        
        if save:
            fig.write_html(self.plots_dir / "optimization_history.html")
        
        return fig
    
    def plot_param_importances(self, study: optuna.Study, save: bool = True) -> plt.Figure:
        """Plot parameter importances."""
        try:
            fig = optuna.visualization.plot_param_importances(study)
            
            if save:
                fig.write_html(self.plots_dir / "param_importances.html")
            
            return fig
        except Exception as e:
            print(f"Could not generate parameter importances plot: {e}")
            return None
    
    def plot_parallel_coordinate(self, study: optuna.Study, save: bool = True) -> plt.Figure:
        """Plot parallel coordinate."""
        try:
            fig = optuna.visualization.plot_parallel_coordinate(study)
            
            if save:
                fig.write_html(self.plots_dir / "parallel_coordinate.html")
            
            return fig
        except Exception as e:
            print(f"Could not generate parallel coordinate plot: {e}")
            return None
    
    def create_custom_plots(self, study: optuna.Study):
        """Create custom matplotlib plots."""
        df = self.get_trials_dataframe(study)
        completed_df = df[df['state'] == 'COMPLETE']
        
        if len(completed_df) < 2:
            print("Not enough completed trials for custom plots")
            return
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Validation loss over trials
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(completed_df['trial_number'], completed_df['value'], 'o-', alpha=0.7)
        ax.axhline(y=completed_df['value'].min(), color='r', linestyle='--', 
                  label=f'Best: {completed_df["value"].min():.4f}')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss Over Trials')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "loss_over_trials.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Parameter correlation heatmap
        param_cols = [col for col in completed_df.columns if col.startswith('param_')]
        if len(param_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = completed_df[param_cols + ['value']].corr()
            
            # Rename columns for better display
            display_names = {col: col.replace('param_', '') for col in param_cols}
            display_names['value'] = 'val_loss'
            corr_matrix = corr_matrix.rename(columns=display_names, index=display_names)
            
            sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                       square=True, ax=ax, fmt='.2f')
            ax.set_title('Parameter Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.plots_dir / "param_correlation.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Training time analysis
        if 'duration_minutes' in completed_df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Duration over trials
            ax1.plot(completed_df['trial_number'], completed_df['duration_minutes'], 'o-', alpha=0.7)
            ax1.set_xlabel('Trial Number')
            ax1.set_ylabel('Duration (minutes)')
            ax1.set_title('Training Time Per Trial')
            ax1.grid(True, alpha=0.3)
            
            # Duration vs performance
            ax2.scatter(completed_df['duration_minutes'], completed_df['value'], alpha=0.7)
            ax2.set_xlabel('Duration (minutes)')
            ax2.set_ylabel('Validation Loss')
            ax2.set_title('Training Time vs Performance')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / "training_time_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Best parameters evolution
        if len(completed_df) >= 5:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Calculate cumulative best
            completed_df_sorted = completed_df.sort_values('trial_number')
            completed_df_sorted['cumulative_best'] = completed_df_sorted['value'].cummin()
            
            ax.plot(completed_df_sorted['trial_number'], completed_df_sorted['value'], 
                   'o', alpha=0.5, label='Trial Results')
            ax.plot(completed_df_sorted['trial_number'], completed_df_sorted['cumulative_best'], 
                   'r-', linewidth=2, label='Best So Far')
            
            ax.set_xlabel('Trial Number')
            ax.set_ylabel('Validation Loss')
            ax.set_title('Convergence Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Custom plots saved to: {self.plots_dir}")
    
    def generate_progress_report(self, study: optuna.Study) -> str:
        """Generate a progress report."""
        df = self.get_trials_dataframe(study)
        
        report = []
        report.append("# Hyperparameter Sweep Progress Report\n")
        report.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall statistics
        total_trials = len(df)
        completed_trials = len(df[df['state'] == 'COMPLETE'])
        pruned_trials = len(df[df['state'] == 'PRUNED'])
        failed_trials = len(df[df['state'] == 'FAIL'])
        
        report.append("## Overview\n")
        report.append(f"- **Total Trials**: {total_trials}")
        report.append(f"- **Completed**: {completed_trials} ({completed_trials/total_trials*100:.1f}%)")
        report.append(f"- **Pruned**: {pruned_trials} ({pruned_trials/total_trials*100:.1f}%)")
        report.append(f"- **Failed**: {failed_trials} ({failed_trials/total_trials*100:.1f}%)\n")
        
        if completed_trials > 0:
            completed_df = df[df['state'] == 'COMPLETE']
            
            # Best trial
            best_trial = completed_df.loc[completed_df['value'].idxmin()]
            report.append("## Best Trial So Far\n")
            report.append(f"- **Trial Number**: {int(best_trial['trial_number'])}")
            report.append(f"- **Validation Loss**: {best_trial['value']:.4f}")
            if 'avg_val_accu' in best_trial:
                report.append(f"- **Validation Accuracy**: {best_trial['avg_val_accu']:.4f}")
            report.append("")
            
            # Top parameters
            param_cols = [col for col in completed_df.columns if col.startswith('param_')]
            if param_cols:
                report.append("### Best Parameters\n")
                report.append("| Parameter | Value |")
                report.append("| --- | --- |")
                for col in param_cols:
                    param_name = col.replace('param_', '')
                    value = best_trial[col]
                    report.append(f"| {param_name} | {value} |")
                report.append("")
            
            # Statistics
            report.append("## Performance Statistics\n")
            report.append("| Metric | Value |")
            report.append("| --- | --- |")
            report.append(f"| Best Loss | {completed_df['value'].min():.4f} |")
            report.append(f"| Mean Loss | {completed_df['value'].mean():.4f} |")
            report.append(f"| Std Loss | {completed_df['value'].std():.4f} |")
            if 'duration_minutes' in completed_df.columns:
                avg_duration = completed_df['duration_minutes'].mean()
                report.append(f"| Avg Duration | {avg_duration:.1f} min |")
            report.append("")
        
        return "\n".join(report)
    
    def save_progress_report(self, study: optuna.Study):
        """Save progress report to file."""
        report = self.generate_progress_report(study)
        report_file = self.sweep_dir / "reports" / "progress_report.md"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Progress report saved to: {report_file}")
    
    def export_results(self, study: optuna.Study, format: str = 'csv'):
        """Export results to various formats."""
        df = self.get_trials_dataframe(study)
        
        if format.lower() == 'csv':
            output_file = self.sweep_dir / "trials_data.csv"
            df.to_csv(output_file, index=False)
        elif format.lower() == 'json':
            output_file = self.sweep_dir / "trials_data.json"
            df.to_json(output_file, orient='records', indent=2)
        elif format.lower() == 'excel':
            output_file = self.sweep_dir / "trials_data.xlsx"
            df.to_excel(output_file, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results exported to: {output_file}")
        return output_file


def monitor_sweep_progress(sweep_dir: Path, study_name: str):
    """Monitor and generate all visualizations for a sweep."""
    monitor = SweepMonitor(sweep_dir)
    
    try:
        study = monitor.load_study(study_name)
        
        # Generate all plots
        print("Generating optimization history...")
        monitor.plot_optimization_history(study)
        
        print("Generating parameter importance...")
        monitor.plot_param_importances(study)
        
        print("Generating parallel coordinate plot...")
        monitor.plot_parallel_coordinate(study)
        
        print("Creating custom plots...")
        monitor.create_custom_plots(study)
        
        print("Generating progress report...")
        monitor.save_progress_report(study)
        
        print("Exporting results...")
        monitor.export_results(study, 'csv')
        monitor.export_results(study, 'json')
        
        print(f"Monitoring complete. Results in {sweep_dir}")
        
    except Exception as e:
        print(f"Error monitoring sweep: {e}")
        import traceback
        traceback.print_exc()