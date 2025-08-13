#!/usr/bin/env python3
"""
Comprehensive test runner for the uTooth hyperparameter sweep system.
Runs all tests and generates a comprehensive report.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch


class TestRunner:
    """Comprehensive test runner."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def run_test_script(self, script_name, description):
        """Run a test script and capture results."""
        
        print(f"\n{'='*60}")
        print(f"🧪 {description}")
        print(f"{'='*60}")
        
        script_path = Path(__file__).parent / script_name
        
        try:
            # Run test with current activated environment
            env = os.environ.copy()
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, timeout=300, 
            cwd=Path(__file__).parent.parent.parent, env=env)
            
            success = result.returncode == 0
            
            self.results[script_name] = {
                'description': description,
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': time.time() - self.start_time
            }
            
            if success:
                print(f"✅ {description}: PASSED")
            else:
                print(f"❌ {description}: FAILED")
                if result.stderr:
                    print(f"Error: {result.stderr[:500]}...")
                    
        except subprocess.TimeoutExpired:
            print(f"⏰ {description}: TIMEOUT")
            self.results[script_name] = {
                'description': description,
                'success': False,
                'error': 'timeout'
            }
        except Exception as e:
            print(f"💥 {description}: EXCEPTION - {str(e)}")
            self.results[script_name] = {
                'description': description,
                'success': False,
                'error': str(e)
            }
    
    def run_all_tests(self):
        """Run all test suites."""
        
        print("🚀 uTooth Hyperparameter Sweep - Comprehensive Test Suite")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.cuda.is_available()}")
        print(f"GPUs: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f}GB)")
        print("=" * 80)
        
        # Test suite
        tests = [
            ('test_system.py', 'System Tests (GPU Detection, Config Loading, Basic Functionality)'),
            ('test_mini_sweep.py', 'Mini Sweep Integration Test (End-to-End Components)'),
            ('test_gpu_utilization.py', 'GPU Utilization Test (Memory and Compute)'),
        ]
        
        for script, description in tests:
            self.run_test_script(script, description)
        
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        
        total_time = time.time() - self.start_time
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get('success', False))
        failed_tests = total_tests - passed_tests
        
        print(f"\n{'='*80}")
        print("🧪 COMPREHENSIVE TEST SUITE RESULTS")
        print(f"{'='*80}")
        print(f"Total Test Suites: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        print(f"Total Time: {total_time:.1f} seconds")
        
        # Detailed results
        print(f"\n📊 Detailed Results:")
        for script, result in self.results.items():
            status = "✅ PASS" if result.get('success') else "❌ FAIL"
            print(f"  {script}: {status}")
            if not result.get('success') and 'error' in result:
                print(f"    Error: {result['error']}")
        
        # System readiness assessment
        print(f"\n🔍 System Readiness Assessment:")
        
        system_test_passed = self.results.get('test_system.py', {}).get('success', False)
        mini_sweep_passed = self.results.get('test_mini_sweep.py', {}).get('success', False)
        
        if system_test_passed and mini_sweep_passed:
            print("✅ SYSTEM READY FOR HYPERPARAMETER SWEEPS!")
            print("✅ All core components working")
            print("✅ GPUs detected and functional") 
            print("✅ Configuration system operational")
            print("✅ Optuna integration working")
            print("✅ Monitoring system ready")
            
            print(f"\n🚀 Next Steps:")
            print("1. Run default sweep: python scripts/sweep_runner.py")
            print("2. Monitor progress: python scripts/monitor_sweep.py --sweep_dir outputs/sweeps/latest --auto-detect")
            print("3. Customize config: cp scripts/sweep/configs/default_sweep_config.yaml my_config.yaml")
            
        else:
            print("❌ SYSTEM NOT READY")
            print("❌ Fix the failed tests before running sweeps")
            
            if not system_test_passed:
                print("   - System tests failed (check GPU setup, dependencies)")
            if not mini_sweep_passed:
                print("   - Integration tests failed (check component integration)")
        
        # GPU Assessment
        gpu_test_passed = self.results.get('test_gpu_utilization.py', {}).get('success', False)
        print(f"\n🖥️  GPU Status:")
        print(f"   GPUs Available: {torch.cuda.device_count()}")
        
        if torch.cuda.device_count() >= 3:
            print("   ✅ Optimal GPU count (3+ RTX 3090s)")
        elif torch.cuda.device_count() >= 1:
            print("   ⚠️  Limited GPU count (parallel training will be slower)")
        else:
            print("   ❌ No GPUs available (sweeps will fail)")
        
        print(f"{'='*80}")
        
        # Save detailed results
        detailed_results = {
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': total_time,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests/total_tests,
            'system_ready': system_test_passed and mini_sweep_passed,
            'gpu_count': torch.cuda.device_count(),
            'pytorch_version': torch.__version__,
            'test_results': self.results
        }
        
        results_file = Path(__file__).parent / 'comprehensive_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"📁 Detailed results saved to: {results_file}")
        
        return passed_tests == total_tests


def main():
    """Run comprehensive test suite."""
    
    runner = TestRunner()
    success = runner.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())