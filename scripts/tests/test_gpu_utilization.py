#!/usr/bin/env python3
"""
GPU utilization test for multi-GPU parallel training.
Tests that all GPUs can be used simultaneously.
"""

import os
import sys
import time
import multiprocessing
import concurrent.futures
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import numpy as np


class SimpleModel(nn.Module):
    """Simple model for testing GPU utilization."""
    
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1)
        )
    
    def forward(self, x):
        return self.layers(x)


def gpu_worker(gpu_id, duration=5):
    """Worker function to test GPU utilization."""
    
    print(f"🔥 Starting GPU {gpu_id} worker...")
    
    # Set GPU
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    
    # Create model and data on GPU
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Track metrics
    start_time = time.time()
    iterations = 0
    total_loss = 0
    
    # Run training loop for specified duration
    while time.time() - start_time < duration:
        # Generate random data
        batch_size = 256
        x = torch.randn(batch_size, 1000, device=device)
        y = torch.randn(batch_size, 1, device=device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        iterations += 1
        
        # Small delay to prevent overwhelming
        time.sleep(0.01)
    
    end_time = time.time()
    actual_duration = end_time - start_time
    
    # Get GPU memory info
    memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
    memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)   # GB
    
    # Clean up
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    
    print(f"✅ GPU {gpu_id} worker completed")
    
    return {
        'gpu_id': gpu_id,
        'duration': actual_duration,
        'iterations': iterations,
        'avg_loss': total_loss / iterations if iterations > 0 else 0,
        'iterations_per_second': iterations / actual_duration,
        'memory_allocated_gb': memory_allocated,
        'memory_reserved_gb': memory_reserved
    }


def test_single_gpu():
    """Test single GPU functionality."""
    print("🧪 Testing single GPU...")
    
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    result = gpu_worker(0, duration=2)
    return {'single_gpu_test': result}


def test_multi_gpu_sequential():
    """Test GPUs one by one (sequential)."""
    print("🧪 Testing multi-GPU sequential...")
    
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return {'error': 'No GPUs available'}
    
    results = []
    for gpu_id in range(min(3, gpu_count)):  # Test up to 3 GPUs
        result = gpu_worker(gpu_id, duration=2)
        results.append(result)
    
    return {'sequential_gpu_test': results}


def test_multi_gpu_parallel():
    """Test all GPUs in parallel (the real test)."""
    print("🧪 Testing multi-GPU parallel...")
    
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return {'error': 'No GPUs available'}
    
    # Test up to 3 GPUs in parallel
    gpus_to_test = list(range(min(3, gpu_count)))
    
    print(f"   Testing GPUs: {gpus_to_test}")
    
    # Use ProcessPoolExecutor for true parallelism
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(gpus_to_test)) as executor:
        # Submit all GPU workers
        futures = {
            executor.submit(gpu_worker, gpu_id, 3): gpu_id 
            for gpu_id in gpus_to_test
        }
        
        results = []
        for future in concurrent.futures.as_completed(futures):
            gpu_id = futures[future]
            try:
                result = future.result(timeout=10)
                results.append(result)
            except Exception as e:
                results.append({
                    'gpu_id': gpu_id,
                    'error': str(e)
                })
    
    return {'parallel_gpu_test': results}


def test_gpu_memory_stress():
    """Test GPU memory allocation across all GPUs."""
    print("🧪 Testing GPU memory stress...")
    
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return {'error': 'No GPUs available'}
    
    results = []
    
    for gpu_id in range(min(3, gpu_count)):
        device = torch.device(f'cuda:{gpu_id}')
        
        try:
            # Allocate progressively larger tensors until we hit limits
            tensors = []
            allocated_gb = 0
            
            # Start with 1GB tensors
            tensor_size_gb = 1
            max_tensors = 20
            
            for i in range(max_tensors):
                try:
                    # Calculate tensor size for ~1GB
                    elements = int(tensor_size_gb * (1024**3) / 4)  # 4 bytes per float32
                    tensor = torch.randn(elements, device=device)
                    tensors.append(tensor)
                    allocated_gb += tensor_size_gb
                    
                except torch.cuda.OutOfMemoryError:
                    break
            
            memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)
            
            # Clean up
            del tensors
            torch.cuda.empty_cache()
            
            results.append({
                'gpu_id': gpu_id,
                'tensors_allocated': len(tensors) if 'tensors' in locals() else 0,
                'estimated_allocated_gb': allocated_gb,
                'actual_memory_allocated_gb': memory_allocated,
                'memory_reserved_gb': memory_reserved
            })
            
        except Exception as e:
            results.append({
                'gpu_id': gpu_id,
                'error': str(e)
            })
    
    return {'memory_stress_test': results}


def main():
    """Run all GPU utilization tests."""
    
    print("🚀 GPU Utilization Test Suite")
    print("=" * 50)
    
    # Check basic GPU info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f}GB)")
    
    print()
    
    all_results = {}
    
    # Run tests
    tests = [
        ('Single GPU Test', test_single_gpu),
        ('Multi-GPU Sequential Test', test_multi_gpu_sequential),  
        ('Multi-GPU Parallel Test', test_multi_gpu_parallel),
        ('GPU Memory Stress Test', test_gpu_memory_stress)
    ]
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            all_results[test_name] = result
            print(f"✅ {test_name} completed")
        except Exception as e:
            all_results[test_name] = {'error': str(e)}
            print(f"❌ {test_name} failed: {e}")
        print()
    
    # Print summary
    print("=" * 70)
    print("🧪 GPU UTILIZATION TEST SUMMARY")
    print("=" * 70)
    
    # Parallel GPU test results
    if 'Multi-GPU Parallel Test' in all_results:
        parallel_results = all_results['Multi-GPU Parallel Test'].get('parallel_gpu_test', [])
        
        if parallel_results and all('error' not in r for r in parallel_results):
            print("✅ Multi-GPU Parallel Test: SUCCESS")
            print("   All GPUs working simultaneously:")
            
            for result in parallel_results:
                gpu_id = result['gpu_id']
                its_per_sec = result.get('iterations_per_second', 0)
                memory_gb = result.get('memory_allocated_gb', 0)
                print(f"     GPU {gpu_id}: {its_per_sec:.1f} iter/sec, {memory_gb:.2f}GB used")
        else:
            print("❌ Multi-GPU Parallel Test: FAILED")
            for result in parallel_results:
                if 'error' in result:
                    print(f"     GPU {result['gpu_id']}: {result['error']}")
    
    # Memory test results
    if 'GPU Memory Stress Test' in all_results:
        memory_results = all_results['GPU Memory Stress Test'].get('memory_stress_test', [])
        print(f"\n📊 Memory Test Results:")
        
        for result in memory_results:
            if 'error' not in result:
                gpu_id = result['gpu_id']
                allocated = result.get('actual_memory_allocated_gb', 0)
                print(f"     GPU {gpu_id}: {allocated:.1f}GB max allocation")
    
    print("=" * 70)
    
    # Save results
    import json
    results_file = Path(__file__).parent / "gpu_utilization_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"📁 Results saved to: {results_file}")
    
    # Return success if parallel test passed
    if 'Multi-GPU Parallel Test' in all_results:
        parallel_results = all_results['Multi-GPU Parallel Test'].get('parallel_gpu_test', [])
        success = parallel_results and all('error' not in r for r in parallel_results)
        return 0 if success else 1
    
    return 1


if __name__ == "__main__":
    sys.exit(main())