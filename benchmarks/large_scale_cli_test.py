#!/usr/bin/env python3
"""
Large-Scale CLI Testing for ProteinMD
Tests CLI functionality with substantial simulations and larger proteins
"""

import os
import sys
import time
import subprocess
import psutil
import threading
from pathlib import Path

def monitor_resources(process, results):
    """Monitor CPU and memory usage during simulation"""
    max_memory = 0
    max_cpu = 0
    
    while process.poll() is None:
        try:
            p = psutil.Process(process.pid)
            cpu = p.cpu_percent()
            memory = p.memory_info().rss / 1024 / 1024  # MB
            
            max_memory = max(max_memory, memory)
            max_cpu = max(max_cpu, cpu)
            
            time.sleep(0.5)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
    
    results['max_memory'] = max_memory
    results['max_cpu'] = max_cpu

def run_large_scale_test():
    """Run comprehensive large-scale CLI tests"""
    print("🧪 LARGE-SCALE CLI TESTING FOR PROTEINMD")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        {
            'name': 'Medium Protein (1000 steps)',
            'protein': 'data/proteins/1ubq.pdb',
            'steps': 1000,
            'timeout': 300  # 5 minutes
        },
        {
            'name': 'Extended Simulation (2000 steps)',
            'protein': 'data/proteins/1ubq.pdb', 
            'steps': 2000,
            'timeout': 600  # 10 minutes
        }
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n🔬 Running Test: {config['name']}")
        print(f"   Protein: {config['protein']}")
        print(f"   Steps: {config['steps']}")
        print(f"   Timeout: {config['timeout']}s")
        
        # Create test command
        cmd = [
            sys.executable, '-c', f'''
import sys
sys.path.append(".")
from proteinMD.cli import ProteinMDCLI
import tempfile
import os

# Create CLI instance
cli = ProteinMDCLI()

# Create temporary output directory
output_dir = tempfile.mkdtemp(prefix="proteinmd_test_")
print(f"Output directory: {{output_dir}}")

# Load protein
print("Loading protein...")
cli.load_protein("{config['protein']}")

# Set parameters
cli.set_temperature(300.0)
cli.set_timestep(0.002)
cli.set_simulation_length({config['steps']})

# Run simulation
print("Starting simulation...")
start_time = time.time()
cli.run_simulation(output_dir=output_dir)
end_time = time.time()

print(f"Simulation completed in {{end_time - start_time:.2f}} seconds")
print(f"Output saved to: {{output_dir}}")

# Cleanup
import shutil
shutil.rmtree(output_dir, ignore_errors=True)
'''
        ]
        
        # Start process with resource monitoring
        start_time = time.time()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='/home/emilio/Documents/ai/md'
        )
        
        # Monitor resources in background thread
        resource_results = {}
        monitor_thread = threading.Thread(
            target=monitor_resources, 
            args=(process, resource_results)
        )
        monitor_thread.start()
        
        try:
            # Wait for process with timeout
            stdout, stderr = process.communicate(timeout=config['timeout'])
            end_time = time.time()
            
            # Wait for monitor thread
            monitor_thread.join()
            
            # Record results
            test_result = {
                'success': process.returncode == 0,
                'duration': end_time - start_time,
                'stdout_lines': len(stdout.split('\n')),
                'stderr_lines': len(stderr.split('\n')),
                'max_memory_mb': resource_results.get('max_memory', 0),
                'max_cpu_percent': resource_results.get('max_cpu', 0),
                'return_code': process.returncode
            }
            
            results[config['name']] = test_result
            
            # Print results
            if test_result['success']:
                print(f"   ✅ SUCCESS")
                print(f"   ⏱️  Duration: {test_result['duration']:.2f}s")
                print(f"   💾 Max Memory: {test_result['max_memory_mb']:.1f}MB")
                print(f"   🔥 Max CPU: {test_result['max_cpu_percent']:.1f}%")
                print(f"   📄 Output Lines: {test_result['stdout_lines']}")
            else:
                print(f"   ❌ FAILED (return code: {test_result['return_code']})")
                print(f"   ⏱️  Duration: {test_result['duration']:.2f}s")
                if stderr:
                    print(f"   🚨 Error Output: {stderr[:200]}...")
                    
        except subprocess.TimeoutExpired:
            print(f"   ⏰ TIMEOUT after {config['timeout']}s")
            process.kill()
            results[config['name']] = {
                'success': False,
                'duration': config['timeout'],
                'timeout': True
            }
            monitor_thread.join()
            
        except Exception as e:
            print(f"   🚨 EXCEPTION: {e}")
            results[config['name']] = {
                'success': False,
                'error': str(e)
            }
    
    # Summary
    print(f"\n📊 LARGE-SCALE TEST SUMMARY")
    print("=" * 40)
    
    total_tests = len(test_configs)
    successful_tests = sum(1 for r in results.values() if r.get('success', False))
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
        print(f"{status} {test_name}")
        if 'duration' in result:
            print(f"     Duration: {result['duration']:.2f}s")
        if 'max_memory_mb' in result:
            print(f"     Peak Memory: {result['max_memory_mb']:.1f}MB")
    
    return successful_tests == total_tests

if __name__ == "__main__":
    import time
    success = run_large_scale_test()
    if success:
        print("\n🎉 ALL LARGE-SCALE TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n💥 SOME TESTS FAILED!")
        sys.exit(1)
