"""
Configuration loader for hyperparameter sweeps.
Handles YAML configuration parsing and validation.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SweepConfig:
    """Configuration class for hyperparameter sweeps."""
    
    # Study configuration
    study_name: str = "utooth_sweep"
    direction: str = "minimize"
    storage_type: str = "sqlite"
    pruner_type: str = "median"
    sampler_type: str = "tpe"
    seed: int = 42
    
    # Hardware configuration
    n_gpus: int = 3
    memory_per_gpu: str = "24GB"
    
    # Training configuration
    max_epochs: int = 30
    k_folds: int = 3
    early_stopping_patience: int = 5
    trials_per_gpu: int = 10
    
    # Paths
    data_path: str = "DATA"  # Relative to project root
    output_base: str = "outputs/sweeps"
    
    # Hyperparameters space
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    baseline: Dict[str, Any] = field(default_factory=dict)
    
    # Output settings
    save_checkpoints: bool = False
    generate_plots: bool = True
    create_report: bool = True


def load_sweep_config(config_path: Optional[str] = None) -> SweepConfig:
    """Load sweep configuration from YAML file."""
    
    if config_path is None:
        # Use default config
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / "configs" / "default_sweep_config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create SweepConfig object
    config = SweepConfig()
    
    # Parse study configuration
    if 'study' in config_dict:
        study_config = config_dict['study']
        config.study_name = study_config.get('name', config.study_name)
        config.direction = study_config.get('direction', config.direction)
        config.storage_type = study_config.get('storage', config.storage_type)
        
        if 'pruner' in study_config:
            config.pruner_type = study_config['pruner'].get('type', config.pruner_type)
        
        if 'sampler' in study_config:
            config.sampler_type = study_config['sampler'].get('type', config.sampler_type)
            config.seed = study_config['sampler'].get('seed', config.seed)
    
    # Parse hardware configuration
    if 'hardware' in config_dict:
        hw_config = config_dict['hardware']
        config.n_gpus = hw_config.get('n_gpus', config.n_gpus)
        config.memory_per_gpu = hw_config.get('memory_per_gpu', config.memory_per_gpu)
    
    # Parse training configuration
    if 'training' in config_dict:
        train_config = config_dict['training']
        config.max_epochs = train_config.get('max_epochs', config.max_epochs)
        config.k_folds = train_config.get('k_folds', config.k_folds)
        config.early_stopping_patience = train_config.get('early_stopping_patience', 
                                                         config.early_stopping_patience)
        config.trials_per_gpu = train_config.get('trials_per_gpu', config.trials_per_gpu)
    
    # Parse hyperparameters
    config.hyperparameters = config_dict.get('hyperparameters', {})
    config.baseline = config_dict.get('baseline', {})
    
    # Parse output configuration
    if 'output' in config_dict:
        output_config = config_dict['output']
        config.output_base = output_config.get('base_directory', config.output_base)
        config.save_checkpoints = output_config.get('save_checkpoints', config.save_checkpoints)
        config.generate_plots = output_config.get('generate_plots', config.generate_plots)
        config.create_report = output_config.get('create_report', config.create_report)
    
    return config


def create_sweep_directory(config: SweepConfig) -> Path:
    """Create organized directory structure for sweep."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_name = f"{config.study_name}_{timestamp}"
    sweep_dir = Path(config.output_base) / sweep_name
    
    # Create directory structure
    dirs_to_create = [
        sweep_dir,
        sweep_dir / "trials",
        sweep_dir / "checkpoints",
        sweep_dir / "logs",
        sweep_dir / "plots",
        sweep_dir / "reports"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration to sweep directory
    config_save_path = sweep_dir / "sweep_config.yaml"
    config_dict = {
        'study': {
            'name': config.study_name,
            'direction': config.direction,
            'storage': config.storage_type,
            'pruner': {'type': config.pruner_type},
            'sampler': {'type': config.sampler_type, 'seed': config.seed}
        },
        'hardware': {
            'n_gpus': config.n_gpus,
            'memory_per_gpu': config.memory_per_gpu
        },
        'training': {
            'max_epochs': config.max_epochs,
            'k_folds': config.k_folds,
            'early_stopping_patience': config.early_stopping_patience,
            'trials_per_gpu': config.trials_per_gpu
        },
        'hyperparameters': config.hyperparameters,
        'baseline': config.baseline,
        'output': {
            'base_directory': config.output_base,
            'save_checkpoints': config.save_checkpoints,
            'generate_plots': config.generate_plots,
            'create_report': config.create_report
        },
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'sweep_directory': str(sweep_dir)
        }
    }
    
    with open(config_save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    return sweep_dir


def validate_config(config: SweepConfig) -> bool:
    """Validate configuration parameters."""
    
    errors = []
    
    # Check required fields
    if not config.study_name:
        errors.append("Study name is required")
    
    if config.n_gpus <= 0:
        errors.append("Number of GPUs must be positive")
    
    if config.max_epochs <= 0:
        errors.append("Max epochs must be positive")
    
    if config.k_folds < 2:
        errors.append("K-folds must be at least 2")
    
    # Check hyperparameters format
    if not config.hyperparameters:
        errors.append("Hyperparameters configuration is empty")
    
    for param_name, param_config in config.hyperparameters.items():
        if 'type' not in param_config:
            errors.append(f"Parameter {param_name} missing type specification")
    
    # Check if baseline parameters are valid
    if config.baseline:
        for param_name in config.baseline:
            if param_name not in config.hyperparameters:
                errors.append(f"Baseline parameter {param_name} not in hyperparameters space")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True