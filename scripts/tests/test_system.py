#!/usr/bin/env python3
"""
Comprehensive system tests for uTooth hyperparameter sweep system.
Tests GPU detection, configuration loading, and basic functionality.
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from scripts.sweep.utils.config_loader import load_sweep_config, create_sweep_directory, validate_config
from scripts.sweep.utils.optuna_helpers import create_optuna_study, suggest_hyperparameters
from scripts.sweep.utils.monitoring import SweepMonitor


class SystemTester:
    """Comprehensive system testing suite."""
    
    def __init__(self):
        self.results = {}
        self.temp_dir = None
        
    def run_all_tests(self):
        """Run all system tests."""
        print("🧪 Starting uTooth Sweep System Tests")
        print("=" * 50)
        
        # Run tests in order
        test_methods = [
            self.test_gpu_detection,
            self.test_pytorch_cuda_setup,
            self.test_config_loading,
            self.test_directory_creation,
            self.test_optuna_setup,
            self.test_hyperparameter_suggestion,
            self.test_monitoring_setup,
            self.test_mock_sweep,
        ]
        
        for test_method in test_methods:
            try:
                test_name = test_method.__name__
                print(f"\n🔍 Running {test_name}...")
                result = test_method()
                self.results[test_name] = {"status": "PASS", "result": result}
                print(f"✅ {test_name}: PASSED")
            except Exception as e:
                self.results[test_name] = {"status": "FAIL", "error": str(e)}
                print(f"❌ {test_name}: FAILED - {str(e)}")
        
        # Print summary
        self.print_test_summary()
        
        return self.results
    
    def test_gpu_detection(self):
        """Test GPU detection and CUDA availability."""
        
        # Basic CUDA test
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            raise Exception("CUDA not available")
        
        # GPU count test
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            raise Exception("No GPUs detected")
        
        # Get GPU info
        gpu_info = []
        for i in range(gpu_count):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                'device_id': i,
                'name': gpu_props.name,
                'memory_total_gb': gpu_props.total_memory / (1024**3),
                'memory_available_gb': (gpu_props.total_memory - torch.cuda.memory_allocated(i)) / (1024**3)
            })
        
        return {
            'cuda_available': cuda_available,
            'gpu_count': gpu_count,
            'expected_gpus': 3,
            'gpu_info': gpu_info
        }
    
    def test_pytorch_cuda_setup(self):
        """Test PyTorch CUDA operations on each GPU."""
        
        results = []
        
        for gpu_id in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{gpu_id}')
            
            # Test basic operations
            try:
                # Create tensors on GPU
                a = torch.randn(1000, 1000, device=device)
                b = torch.randn(1000, 1000, device=device)
                
                # Matrix multiplication
                start_time = time.time()
                c = torch.mm(a, b)
                operation_time = time.time() - start_time
                
                # Memory test
                memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)
                
                results.append({
                    'gpu_id': gpu_id,
                    'operation_successful': True,
                    'operation_time_seconds': operation_time,
                    'memory_allocated_gb': memory_allocated,
                    'memory_reserved_gb': memory_reserved
                })
                
                # Clean up
                del a, b, c
                torch.cuda.empty_cache()
                
            except Exception as e:
                results.append({
                    'gpu_id': gpu_id,
                    'operation_successful': False,
                    'error': str(e)
                })
        
        return results
    
    def test_config_loading(self):
        """Test configuration loading and validation."""
        
        # Test default config loading
        config = load_sweep_config()
        
        # Validate config
        is_valid = validate_config(config)
        if not is_valid:
            raise Exception("Default configuration validation failed")
        
        # Test config parameters
        expected_params = ['learning_rate', 'loss_alpha', 'batch_size']
        for param in expected_params:
            if param not in config.hyperparameters:
                raise Exception(f"Missing expected parameter: {param}")
        
        return {
            'config_loaded': True,
            'validation_passed': is_valid,
            'hyperparameter_count': len(config.hyperparameters),
            'gpu_count_config': config.n_gpus,
            'max_epochs': config.max_epochs
        }
    
    def test_directory_creation(self):
        """Test sweep directory creation and organization."""
        
        config = load_sweep_config()
        
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_base = temp_dir
            sweep_dir = create_sweep_directory(config)
            
            # Check if directories were created
            expected_dirs = ['trials', 'logs', 'plots', 'reports']
            created_dirs = []
            
            for expected_dir in expected_dirs:
                dir_path = sweep_dir / expected_dir
                if dir_path.exists():
                    created_dirs.append(expected_dir)
            
            # Check config file
            config_file = sweep_dir / 'sweep_config.yaml'
            config_file_exists = config_file.exists()
            
            return {
                'sweep_dir_created': sweep_dir.exists(),
                'expected_dirs': expected_dirs,
                'created_dirs': created_dirs,
                'config_file_exists': config_file_exists,
                'sweep_dir_path': str(sweep_dir)
            }
    
    def test_optuna_setup(self):
        """Test Optuna study creation."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_sweep_config()
            config.output_base = temp_dir
            sweep_dir = create_sweep_directory(config)
            
            # Create Optuna study
            study = create_optuna_study(config, sweep_dir)
            
            return {
                'study_created': study is not None,
                'study_name': study.study_name,
                'direction': study.direction.name,
                'sampler_type': type(study.sampler).__name__,
                'pruner_type': type(study.pruner).__name__
            }
    
    def test_hyperparameter_suggestion(self):
        """Test hyperparameter suggestion mechanism."""
        
        config = load_sweep_config()
        
        # Create mock trial
        class MockTrial:
            def __init__(self):
                self.params = {}
            
            def suggest_float(self, name, low, high, **kwargs):
                # Return middle value for testing
                return float(low + high) / 2
            
            def suggest_int(self, name, low, high, **kwargs):
                return int((int(low) + int(high)) / 2)
            
            def suggest_categorical(self, name, choices):
                return choices[0]  # Return first choice
        
        mock_trial = MockTrial()
        suggested_params = suggest_hyperparameters(mock_trial, config.hyperparameters)
        
        return {
            'suggestion_successful': len(suggested_params) > 0,
            'suggested_params': suggested_params,
            'param_count': len(suggested_params),
            'expected_params': list(config.hyperparameters.keys())
        }
    
    def test_monitoring_setup(self):
        """Test monitoring system setup."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock sweep directory structure
            plots_dir = temp_path / "plots"
            plots_dir.mkdir()
            
            monitor = SweepMonitor(temp_path)
            
            return {
                'monitor_created': monitor is not None,
                'plots_dir_exists': monitor.plots_dir.exists(),
                'plots_dir_path': str(monitor.plots_dir)
            }
    
    def test_mock_sweep(self):
        """Test a very short mock sweep with fake data."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_sweep_config()
            config.output_base = temp_dir
            config.max_epochs = 1  # Very short for testing
            config.k_folds = 2     # Fewer folds
            
            sweep_dir = create_sweep_directory(config)
            
            # Create Optuna study
            study = create_optuna_study(config, sweep_dir)
            
            # Test parameter suggestion
            def mock_objective(trial):
                # Suggest parameters
                params = suggest_hyperparameters(trial, config.hyperparameters)
                
                # Simulate training result
                mock_loss = np.random.uniform(0.1, 1.0)
                return mock_loss
            
            # Run a few mock trials
            study.optimize(mock_objective, n_trials=3)
            
            return {
                'mock_sweep_completed': True,
                'trials_completed': len(study.trials),
                'best_value': study.best_value,
                'best_params': study.best_params
            }
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        
        print("\n" + "=" * 70)
        print("🧪 SYSTEM TEST SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        # GPU Information
        if 'test_gpu_detection' in self.results and self.results['test_gpu_detection']['status'] == 'PASS':
            gpu_info = self.results['test_gpu_detection']['result']
            print(f"\n🖥️  GPU Information:")
            print(f"   CUDA Available: {gpu_info['cuda_available']}")
            print(f"   GPUs Detected: {gpu_info['gpu_count']}")
            for gpu in gpu_info['gpu_info']:
                print(f"   GPU {gpu['device_id']}: {gpu['name']} ({gpu['memory_total_gb']:.1f}GB)")
        
        # Failed tests details
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for test_name, result in self.results.items():
                if result['status'] == 'FAIL':
                    print(f"   {test_name}: {result['error']}")
        
        print("=" * 70)


def main():
    """Run all system tests."""
    tester = SystemTester()
    results = tester.run_all_tests()
    
    # Save test results
    results_file = Path(__file__).parent / "test_results.json"
    results['timestamp'] = datetime.now().isoformat()
    results['python_version'] = sys.version
    results['pytorch_version'] = torch.__version__
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Test results saved to: {results_file}")
    
    # Return exit code based on results
    failed_count = sum(1 for r in results.values() if isinstance(r, dict) and r.get('status') == 'FAIL')
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())