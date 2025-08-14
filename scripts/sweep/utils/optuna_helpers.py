"""
Optuna helper functions for hyperparameter sweeps.
Handles study creation, parameter suggestion, and results processing.
"""

import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler, RandomSampler
from typing import Dict, Any, Union
import logging
from pathlib import Path


def create_optuna_study(config, storage_path: Path):
    """Create an Optuna study with the specified configuration."""
    
    # Configure pruner
    pruner = None
    if config.pruner_type == "median":
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    elif config.pruner_type == "successive_halving":
        pruner = SuccessiveHalvingPruner()
    
    # Configure sampler
    sampler = None
    if config.sampler_type == "tpe":
        sampler = TPESampler(seed=config.seed)
    elif config.sampler_type == "random":
        sampler = RandomSampler(seed=config.seed)
    
    # Create storage URL
    storage_url = f"sqlite:///{storage_path / 'optuna_study.db'}"
    
    # Create study
    study = optuna.create_study(
        study_name=config.study_name,
        storage=storage_url,
        direction=config.direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    return study


def suggest_hyperparameters(trial: optuna.Trial, hyperparams_config: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest hyperparameters based on configuration."""
    
    suggested_params = {}
    
    for param_name, param_config in hyperparams_config.items():
        param_type = param_config['type']
        
        if param_type == 'float':
            # Ensure low and high are floats (handle scientific notation from YAML)
            low = float(param_config['low'])
            high = float(param_config['high'])
            
            if param_config.get('log', False):
                value = trial.suggest_float(
                    param_name, 
                    low, 
                    high, 
                    log=True
                )
            elif 'step' in param_config:
                step = float(param_config['step'])
                value = trial.suggest_float(
                    param_name,
                    low,
                    high,
                    step=step
                )
            else:
                value = trial.suggest_float(
                    param_name,
                    low,
                    high
                )
                
        elif param_type == 'int':
            # Ensure low and high are integers
            low = int(param_config['low'])
            high = int(param_config['high'])
            
            if 'step' in param_config:
                step = int(param_config['step'])
                value = trial.suggest_int(
                    param_name,
                    low,
                    high,
                    step=step
                )
            else:
                value = trial.suggest_int(
                    param_name,
                    low,
                    high
                )
                
        elif param_type == 'categorical':
            value = trial.suggest_categorical(
                param_name,
                param_config['choices']
            )
            
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
        
        suggested_params[param_name] = value
    
    return suggested_params


def enqueue_baseline_trial(study: optuna.study.Study, baseline_params: Dict[str, Any]):
    """Enqueue the baseline parameters as the first trial."""
    
    if baseline_params:
        # Convert string values to proper types
        converted_params = {}
        for key, value in baseline_params.items():
            if key == 'learning_rate' and isinstance(value, str):
                # Convert scientific notation strings to floats
                converted_params[key] = float(value)
            else:
                converted_params[key] = value
        
        study.enqueue_trial(converted_params)
        print(f"Enqueued baseline trial with parameters: {converted_params}")


def calculate_trial_metrics(fold_results: list) -> Dict[str, float]:
    """Calculate aggregated metrics from fold results."""
    
    import numpy as np
    
    val_losses = [result['val_loss'] for result in fold_results]
    val_accus = [result['val_accu'] for result in fold_results]
    
    metrics = {
        'mean_val_loss': float(np.mean(val_losses)),
        'std_val_loss': float(np.std(val_losses)),
        'min_val_loss': float(np.min(val_losses)),
        'max_val_loss': float(np.max(val_losses)),
        'mean_val_accu': float(np.mean(val_accus)),
        'std_val_accu': float(np.std(val_accus)),
        'min_val_accu': float(np.min(val_accus)),
        'max_val_accu': float(np.max(val_accus)),
        'fold_val_losses': val_losses,
        'fold_val_accus': val_accus
    }
    
    return metrics


def log_trial_results(trial: optuna.Trial, metrics: Dict[str, float], 
                     suggested_params: Dict[str, Any]):
    """Log trial results and set user attributes."""
    
    # Set user attributes for detailed tracking
    for key, value in metrics.items():
        if not key.startswith('fold_'):  # Don't log fold-level arrays as attributes
            trial.set_user_attr(key, value)
    
    # Log suggested parameters
    for key, value in suggested_params.items():
        trial.set_user_attr(f"param_{key}", value)
    
    print(f"Trial {trial.number} completed:")
    print(f"  Mean validation loss: {metrics['mean_val_loss']:.4f} ± {metrics['std_val_loss']:.4f}")
    print(f"  Mean validation accuracy: {metrics['mean_val_accu']:.4f} ± {metrics['std_val_accu']:.4f}")


def get_study_statistics(study: optuna.study.Study) -> Dict[str, Any]:
    """Get comprehensive statistics from completed study."""
    
    completed_trials = [t for t in study.trials 
                       if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials 
                    if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials 
                    if t.state == optuna.trial.TrialState.FAIL]
    
    stats = {
        'total_trials': len(study.trials),
        'completed_trials': len(completed_trials),
        'pruned_trials': len(pruned_trials),
        'failed_trials': len(failed_trials),
        'success_rate': len(completed_trials) / len(study.trials) if study.trials else 0,
        'best_trial_number': study.best_trial.number if completed_trials else None,
        'best_value': study.best_value if completed_trials else None,
        'best_params': study.best_params if completed_trials else None,
    }
    
    # Add parameter importance if enough trials
    if len(completed_trials) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            stats['param_importance'] = importance
        except Exception as e:
            print(f"Could not calculate parameter importance: {e}")
            stats['param_importance'] = {}
    else:
        stats['param_importance'] = {}
    
    return stats


def setup_logging(sweep_dir: Path, verbose: bool = True):
    """Setup logging for the sweep."""
    
    log_file = sweep_dir / "logs" / "sweep.log"
    log_file.parent.mkdir(exist_ok=True)
    
    # Configure logging
    log_level = logging.INFO if verbose else logging.WARNING
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Reduce optuna logging verbosity
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    return logging.getLogger(__name__)