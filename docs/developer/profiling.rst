Profiling Guide
===============

This comprehensive guide covers profiling techniques and tools for optimizing ProteinMD performance, from basic Python profiling to advanced GPU analysis.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

Profiling is essential for identifying performance bottlenecks and optimizing molecular dynamics simulations. This guide provides systematic approaches to profiling at different levels of the ProteinMD stack.

Profiling Philosophy
~~~~~~~~~~~~~~~~~~~

- **Profile before optimizing:** Measure to identify real bottlenecks
- **Use appropriate tools:** Different tools for different types of analysis
- **Profile realistic workloads:** Use representative simulation conditions
- **Iterative optimization:** Profile, optimize, measure, repeat
- **Document findings:** Keep records of optimization results

Python Profiling
----------------

Built-in Profilers
~~~~~~~~~~~~~~~~~

**cProfile - Function-Level Profiling:**

.. code-block:: python

   import cProfile
   import pstats
   import io
   from pstats import SortKey
   
   def profile_simulation():
       """Profile a complete simulation."""
       pr = cProfile.Profile()
       
       # Start profiling
       pr.enable()
       
       # Run simulation code
       simulation = setup_simulation()
       simulation.run(steps=1000)
       
       # Stop profiling
       pr.disable()
       
       # Analyze results
       s = io.StringIO()
       sortby = SortKey.CUMULATIVE
       ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
       ps.print_stats()
       
       print(s.getvalue())
       
       # Save detailed report
       ps.dump_stats('simulation_profile.prof')
   
   # Analyze saved profile
   def analyze_profile(filename='simulation_profile.prof'):
       """Analyze saved profile data."""
       stats = pstats.Stats(filename)
       
       # Top 20 functions by cumulative time
       print("Top 20 functions by cumulative time:")
       stats.sort_stats('cumulative').print_stats(20)
       
       # Functions taking most self-time
       print("\nTop 20 functions by self time:")
       stats.sort_stats('tottime').print_stats(20)
       
       # Specific function details
       print("\nForce calculation details:")
       stats.print_stats('force')

**Using cProfile from Command Line:**

.. code-block:: bash

   # Profile script execution
   python -m cProfile -o profile_output.prof simulation_script.py
   
   # Analyze with snakeviz (visual profiler)
   pip install snakeviz
   snakeviz profile_output.prof

Line-by-Line Profiling
~~~~~~~~~~~~~~~~~~~~~~

**line_profiler for Detailed Analysis:**

.. code-block:: python

   # Install: pip install line_profiler
   
   @profile  # Add this decorator to functions you want to profile
   def calculate_forces(positions, charges):
       """Calculate electrostatic forces between particles."""
       n_atoms = len(positions)
       forces = np.zeros_like(positions)
       
       for i in range(n_atoms):
           for j in range(i + 1, n_atoms):
               r = positions[i] - positions[j]
               r_mag = np.linalg.norm(r)
               
               if r_mag > 0:
                   force_mag = charges[i] * charges[j] / (r_mag**3)
                   force = force_mag * r
                   
                   forces[i] += force
                   forces[j] -= force
       
       return forces
   
   # Run with: kernprof -l -v script.py

**Memory Profiling with memory_profiler:**

.. code-block:: python

   # Install: pip install memory_profiler
   
   from memory_profiler import profile
   
   @profile
   def memory_intensive_function():
       """Function to profile memory usage."""
       import numpy as np
       
       # Large array allocation
       positions = np.random.random((100000, 3))
       velocities = np.random.random((100000, 3))
       forces = np.zeros((100000, 3))
       
       # Some computation
       for i in range(100):
           forces += np.random.random((100000, 3))
           
       return positions, velocities, forces
   
   # Run with: python -m memory_profiler script.py

Advanced Python Profiling
~~~~~~~~~~~~~~~~~~~~~~~~~

**Custom Profiling Context Manager:**

.. code-block:: python

   import time
   import functools
   from contextlib import contextmanager
   
   @contextmanager
   def profile_block(name):
       """Context manager for profiling code blocks."""
       start_time = time.perf_counter()
       print(f"Starting {name}...")
       
       try:
           yield
       finally:
           end_time = time.perf_counter()
           print(f"Completed {name} in {end_time - start_time:.4f} seconds")
   
   def timing_decorator(func):
       """Decorator for timing function calls."""
       @functools.wraps(func)
       def wrapper(*args, **kwargs):
           start_time = time.perf_counter()
           result = func(*args, **kwargs)
           end_time = time.perf_counter()
           
           print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
           return result
       return wrapper
   
   # Usage examples
   def example_profiling():
       """Example of using profiling tools."""
       
       with profile_block("Force calculation"):
           forces = calculate_forces(positions, charges)
       
       @timing_decorator
       def expensive_operation():
           return np.linalg.eigvals(large_matrix)
       
       eigenvalues = expensive_operation()

Statistical Profiling
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   import statistics
   from typing import List, Callable, Any
   
   class StatisticalProfiler:
       """Statistical profiler for benchmarking functions."""
       
       def __init__(self, n_runs=10, warmup_runs=2):
           self.n_runs = n_runs
           self.warmup_runs = warmup_runs
           
       def profile_function(self, func: Callable, *args, **kwargs) -> dict:
           """Profile function with statistical analysis."""
           times = []
           
           # Warmup runs
           for _ in range(self.warmup_runs):
               func(*args, **kwargs)
           
           # Timing runs
           for _ in range(self.n_runs):
               start_time = time.perf_counter()
               result = func(*args, **kwargs)
               end_time = time.perf_counter()
               times.append(end_time - start_time)
           
           return {
               'mean_time': statistics.mean(times),
               'median_time': statistics.median(times),
               'std_time': statistics.stdev(times) if len(times) > 1 else 0,
               'min_time': min(times),
               'max_time': max(times),
               'n_runs': self.n_runs,
               'all_times': times
           }
       
       def compare_functions(self, functions: List[Callable], *args, **kwargs):
           """Compare performance of multiple functions."""
           results = {}
           
           for func in functions:
               results[func.__name__] = self.profile_function(func, *args, **kwargs)
           
           # Print comparison
           print("Function Performance Comparison:")
           print("-" * 50)
           for name, stats in results.items():
               print(f"{name:20} {stats['mean_time']:.4f}s ± {stats['std_time']:.4f}s")
           
           return results

NumPy and Scientific Computing Profiling
----------------------------------------

Vectorization Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import time
   
   def compare_vectorization():
       """Compare vectorized vs. loop-based implementations."""
       n = 1000000
       a = np.random.random(n)
       b = np.random.random(n)
       
       # Pure Python loop (slow)
       def python_loop():
           result = []
           for i in range(len(a)):
               result.append(a[i] * b[i] + a[i]**2)
           return result
       
       # NumPy loop (medium)
       def numpy_loop():
           result = np.zeros(len(a))
           for i in range(len(a)):
               result[i] = a[i] * b[i] + a[i]**2
           return result
       
       # Vectorized (fast)
       def vectorized():
           return a * b + a**2
       
       # Benchmark
       profiler = StatisticalProfiler(n_runs=5)
       
       print("Vectorization Comparison:")
       results_python = profiler.profile_function(python_loop)
       results_numpy = profiler.profile_function(numpy_loop)
       results_vectorized = profiler.profile_function(vectorized)
       
       print(f"Python loop:    {results_python['mean_time']:.4f}s")
       print(f"NumPy loop:     {results_numpy['mean_time']:.4f}s")
       print(f"Vectorized:     {results_vectorized['mean_time']:.4f}s")
       
       speedup = results_python['mean_time'] / results_vectorized['mean_time']
       print(f"Vectorization speedup: {speedup:.1f}x")

Memory Access Pattern Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   def analyze_memory_patterns():
       """Analyze different memory access patterns."""
       n = 1000
       matrix = np.random.random((n, n))
       
       def row_major_access():
           """Cache-friendly row-major access."""
           total = 0.0
           for i in range(n):
               for j in range(n):
                   total += matrix[i, j]
           return total
       
       def column_major_access():
           """Cache-unfriendly column-major access."""
           total = 0.0
           for j in range(n):
               for i in range(n):
                   total += matrix[i, j]
           return total
       
       def strided_access():
           """Strided access pattern."""
           total = 0.0
           for i in range(0, n, 2):
               for j in range(0, n, 2):
                   total += matrix[i, j]
           return total
       
       profiler = StatisticalProfiler()
       
       print("Memory Access Pattern Analysis:")
       results = profiler.compare_functions([
           row_major_access, 
           column_major_access, 
           strided_access
       ])

GPU Profiling
------------

CUDA Profiling with CuPy
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cupy as cp
   import time
   
   class GPUProfiler:
       """Profiler for GPU operations using CuPy."""
       
       def __init__(self):
           self.device = cp.cuda.Device()
           
       def profile_kernel(self, kernel_func, *args, n_runs=10):
           """Profile GPU kernel execution."""
           # Warmup
           for _ in range(2):
               kernel_func(*args)
               cp.cuda.Stream.null.synchronize()
           
           # Profile
           times = []
           for _ in range(n_runs):
               start_event = cp.cuda.Event()
               end_event = cp.cuda.Event()
               
               start_event.record()
               kernel_func(*args)
               end_event.record()
               end_event.synchronize()
               
               elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)
               times.append(elapsed_time)
           
           return {
               'mean_time_ms': np.mean(times),
               'std_time_ms': np.std(times),
               'min_time_ms': np.min(times),
               'max_time_ms': np.max(times)
           }
       
       def profile_memory_transfer(self, data_size_mb):
           """Profile CPU-GPU memory transfer rates."""
           # Create test data
           size = int(data_size_mb * 1024 * 1024 / 4)  # float32
           cpu_data = np.random.random(size).astype(np.float32)
           
           # Host to Device
           start_time = time.perf_counter()
           gpu_data = cp.asarray(cpu_data)
           cp.cuda.Stream.null.synchronize()
           h2d_time = time.perf_counter() - start_time
           
           # Device to Host
           start_time = time.perf_counter()
           result = cp.asnumpy(gpu_data)
           d2h_time = time.perf_counter() - start_time
           
           h2d_bandwidth = data_size_mb / h2d_time
           d2h_bandwidth = data_size_mb / d2h_time
           
           return {
               'h2d_time_s': h2d_time,
               'd2h_time_s': d2h_time,
               'h2d_bandwidth_mb_s': h2d_bandwidth,
               'd2h_bandwidth_mb_s': d2h_bandwidth
           }

NVIDIA Profiling Tools
~~~~~~~~~~~~~~~~~~~~~

**Using nvprof:**

.. code-block:: bash

   # Profile Python script with CUDA
   nvprof python simulation_gpu.py
   
   # Detailed profiling with metrics
   nvprof --metrics gld_efficiency,gst_efficiency python simulation_gpu.py
   
   # Timeline profiling
   nvprof --output-profile timeline.nvvp python simulation_gpu.py

**Using Nsight Systems:**

.. code-block:: bash

   # Modern replacement for nvprof
   nsys profile --trace=cuda,nvtx python simulation_gpu.py
   
   # With detailed options
   nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
        --output=simulation_profile python simulation_gpu.py

GPU Memory Profiling
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cupy as cp
   
   def profile_gpu_memory():
       """Profile GPU memory usage patterns."""
       mempool = cp.get_default_memory_pool()
       
       def get_memory_info():
           used_bytes = mempool.used_bytes()
           total_bytes = mempool.total_bytes()
           free_bytes, device_total = cp.cuda.runtime.memGetInfo()
           
           return {
               'pool_used_mb': used_bytes / 1024**2,
               'pool_total_mb': total_bytes / 1024**2,
               'device_free_mb': free_bytes / 1024**2,
               'device_total_mb': device_total / 1024**2
           }
       
       print("Initial GPU memory:")
       initial_memory = get_memory_info()
       print(f"Pool used: {initial_memory['pool_used_mb']:.1f} MB")
       print(f"Device free: {initial_memory['device_free_mb']:.1f} MB")
       
       # Allocate test arrays
       arrays = []
       for i in range(10):
           size_mb = 100
           array = cp.zeros(size_mb * 1024 * 1024 // 4, dtype=cp.float32)
           arrays.append(array)
           
           current_memory = get_memory_info()
           print(f"After allocation {i+1}: "
                 f"Pool used: {current_memory['pool_used_mb']:.1f} MB, "
                 f"Device free: {current_memory['device_free_mb']:.1f} MB")
       
       # Clean up
       del arrays
       mempool.free_all_blocks()
       
       final_memory = get_memory_info()
       print(f"After cleanup: Pool used: {final_memory['pool_used_mb']:.1f} MB")

Molecular Dynamics Specific Profiling
-------------------------------------

Force Calculation Profiling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MDProfiler:
       """Specialized profiler for molecular dynamics simulations."""
       
       def __init__(self):
           self.force_timings = []
           self.integration_timings = []
           self.neighbor_timings = []
           
       def profile_force_calculation(self, simulation, n_steps=100):
           """Profile force calculation performance."""
           timings = {
               'lennard_jones': [],
               'electrostatic': [],
               'bonded': [],
               'total': []
           }
           
           for step in range(n_steps):
               # Total force calculation time
               start_total = time.perf_counter()
               
               # Individual force components
               start_lj = time.perf_counter()
               simulation.calculate_lennard_jones_forces()
               timings['lennard_jones'].append(time.perf_counter() - start_lj)
               
               start_elec = time.perf_counter()
               simulation.calculate_electrostatic_forces()
               timings['electrostatic'].append(time.perf_counter() - start_elec)
               
               start_bonded = time.perf_counter()
               simulation.calculate_bonded_forces()
               timings['bonded'].append(time.perf_counter() - start_bonded)
               
               timings['total'].append(time.perf_counter() - start_total)
           
           # Analyze timings
           results = {}
           for force_type, times in timings.items():
               results[force_type] = {
                   'mean_time': np.mean(times),
                   'std_time': np.std(times),
                   'percentage': np.mean(times) / np.mean(timings['total']) * 100
               }
           
           return results
       
       def profile_neighbor_list(self, simulation, n_updates=20):
           """Profile neighbor list construction."""
           times = []
           
           for _ in range(n_updates):
               start_time = time.perf_counter()
               simulation.update_neighbor_list()
               times.append(time.perf_counter() - start_time)
           
           return {
               'mean_time': np.mean(times),
               'std_time': np.std(times),
               'atoms_per_second': simulation.n_atoms / np.mean(times)
           }

Scalability Profiling
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def profile_scalability():
       """Profile simulation performance vs. system size."""
       system_sizes = [1000, 2000, 5000, 10000, 20000]
       results = []
       
       for n_atoms in system_sizes:
           print(f"Profiling system with {n_atoms} atoms...")
           
           # Create simulation
           simulation = create_simulation(n_atoms)
           
           # Profile single step
           start_time = time.perf_counter()
           simulation.step()
           step_time = time.perf_counter() - start_time
           
           # Calculate performance metrics
           atoms_per_second = n_atoms / step_time
           time_per_atom = step_time / n_atoms * 1e6  # microseconds
           
           results.append({
               'n_atoms': n_atoms,
               'step_time': step_time,
               'atoms_per_second': atoms_per_second,
               'time_per_atom_us': time_per_atom
           })
       
       # Analyze scaling
       import matplotlib.pyplot as plt
       
       n_atoms_list = [r['n_atoms'] for r in results]
       step_times = [r['step_time'] for r in results]
       
       plt.figure(figsize=(10, 6))
       plt.loglog(n_atoms_list, step_times, 'o-', label='Actual')
       
       # Theoretical O(N) scaling
       theoretical = [step_times[0] * n / n_atoms_list[0] for n in n_atoms_list]
       plt.loglog(n_atoms_list, theoretical, '--', label='O(N) theoretical')
       
       # Theoretical O(N²) scaling
       theoretical_n2 = [step_times[0] * (n / n_atoms_list[0])**2 for n in n_atoms_list]
       plt.loglog(n_atoms_list, theoretical_n2, '--', label='O(N²) theoretical')
       
       plt.xlabel('Number of Atoms')
       plt.ylabel('Step Time (s)')
       plt.title('Simulation Performance Scaling')
       plt.legend()
       plt.grid(True)
       plt.savefig('scaling_analysis.png')
       
       return results

Performance Bottleneck Analysis
------------------------------

Hotspot Identification
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class BottleneckAnalyzer:
       """Identify and analyze performance bottlenecks."""
       
       def __init__(self):
           self.function_times = {}
           self.call_counts = {}
           
       def time_function(self, func_name):
           """Decorator to time function calls."""
           def decorator(func):
               @functools.wraps(func)
               def wrapper(*args, **kwargs):
                   start_time = time.perf_counter()
                   result = func(*args, **kwargs)
                   end_time = time.perf_counter()
                   
                   if func_name not in self.function_times:
                       self.function_times[func_name] = []
                       self.call_counts[func_name] = 0
                   
                   self.function_times[func_name].append(end_time - start_time)
                   self.call_counts[func_name] += 1
                   
                   return result
               return wrapper
           return decorator
       
       def analyze_bottlenecks(self):
           """Analyze collected timing data."""
           analysis = {}
           
           for func_name, times in self.function_times.items():
               total_time = sum(times)
               mean_time = np.mean(times)
               call_count = self.call_counts[func_name]
               
               analysis[func_name] = {
                   'total_time': total_time,
                   'mean_time': mean_time,
                   'call_count': call_count,
                   'percentage': 0  # Will be calculated below
               }
           
           # Calculate percentages
           total_all_time = sum(data['total_time'] for data in analysis.values())
           for data in analysis.values():
               data['percentage'] = data['total_time'] / total_all_time * 100
           
           # Sort by total time
           sorted_analysis = sorted(analysis.items(), 
                                  key=lambda x: x[1]['total_time'], 
                                  reverse=True)
           
           print("Performance Bottleneck Analysis:")
           print("-" * 60)
           print(f"{'Function':<25} {'Total(s)':<10} {'Mean(s)':<10} {'Calls':<8} {'%':<6}")
           print("-" * 60)
           
           for func_name, data in sorted_analysis:
               print(f"{func_name:<25} {data['total_time']:<10.4f} "
                     f"{data['mean_time']:<10.6f} {data['call_count']:<8} "
                     f"{data['percentage']:<6.1f}")
           
           return analysis

I/O Profiling
~~~~~~~~~~~~

.. code-block:: python

   def profile_io_operations():
       """Profile file I/O operations."""
       import tempfile
       import os
       
       def profile_trajectory_writing():
           """Profile trajectory file writing."""
           n_atoms = 10000
           n_frames = 100
           
           # Generate test data
           trajectory_data = np.random.random((n_frames, n_atoms, 3))
           
           # Profile different file formats
           formats = {
               'numpy_binary': lambda: np.save('test.npy', trajectory_data),
               'numpy_compressed': lambda: np.savez_compressed('test.npz', 
                                                             trajectory=trajectory_data),
               'hdf5': lambda: save_hdf5('test.h5', trajectory_data),
               'pickle': lambda: pickle.dump(trajectory_data, 
                                           open('test.pkl', 'wb'))
           }
           
           results = {}
           for format_name, save_func in formats.items():
               # Time the save operation
               start_time = time.perf_counter()
               save_func()
               save_time = time.perf_counter() - start_time
               
               # Get file size
               filename = f'test.{format_name.split("_")[0]}'
               if format_name == 'numpy_compressed':
                   filename = 'test.npz'
               elif format_name == 'hdf5':
                   filename = 'test.h5'
               elif format_name == 'pickle':
                   filename = 'test.pkl'
               
               file_size = os.path.getsize(filename) / 1024**2  # MB
               
               results[format_name] = {
                   'save_time': save_time,
                   'file_size_mb': file_size,
                   'throughput_mb_s': file_size / save_time
               }
               
               # Clean up
               os.remove(filename)
           
           return results
       
       def save_hdf5(filename, data):
           """Helper function to save HDF5."""
           import h5py
           with h5py.File(filename, 'w') as f:
               f.create_dataset('trajectory', data=data, compression='gzip')

Automated Performance Monitoring
-------------------------------

Continuous Performance Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import json
   import datetime
   
   class PerformanceTracker:
       """Track performance over time for regression detection."""
       
       def __init__(self, log_file='performance_log.json'):
           self.log_file = log_file
           self.load_history()
           
       def load_history(self):
           """Load historical performance data."""
           try:
               with open(self.log_file, 'r') as f:
                   self.history = json.load(f)
           except FileNotFoundError:
               self.history = []
           
       def save_history(self):
           """Save performance history."""
           with open(self.log_file, 'w') as f:
               json.dump(self.history, f, indent=2)
           
       def benchmark_simulation(self, simulation_config):
           """Benchmark simulation and record results."""
           timestamp = datetime.datetime.now().isoformat()
           
           # Run benchmark
           simulation = create_simulation(**simulation_config)
           
           start_time = time.perf_counter()
           simulation.run(steps=100)
           total_time = time.perf_counter() - start_time
           
           # Collect system info
           import platform
           import psutil
           
           result = {
               'timestamp': timestamp,
               'config': simulation_config,
               'performance': {
                   'total_time': total_time,
                   'steps_per_second': 100 / total_time,
                   'atoms_per_step_per_second': simulation.n_atoms * 100 / total_time
               },
               'system_info': {
                   'python_version': platform.python_version(),
                   'cpu_count': psutil.cpu_count(),
                   'memory_gb': psutil.virtual_memory().total / 1024**3,
                   'platform': platform.platform()
               }
           }
           
           self.history.append(result)
           self.save_history()
           
           return result
       
       def detect_regressions(self, threshold=0.1):
           """Detect performance regressions."""
           if len(self.history) < 2:
               return []
           
           recent = self.history[-1]
           baseline = self.history[-2]
           
           regressions = []
           
           for metric in ['total_time', 'steps_per_second']:
               recent_value = recent['performance'][metric]
               baseline_value = baseline['performance'][metric]
               
               if metric == 'total_time':
                   # Lower is better for time
                   change = (recent_value - baseline_value) / baseline_value
               else:
                   # Higher is better for throughput
                   change = (baseline_value - recent_value) / baseline_value
               
               if change > threshold:
                   regressions.append({
                       'metric': metric,
                       'change_percent': change * 100,
                       'recent_value': recent_value,
                       'baseline_value': baseline_value
                   })
           
           return regressions

Visualization and Reporting
--------------------------

Performance Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns
   
   def create_performance_dashboard(profiling_results):
       """Create comprehensive performance dashboard."""
       fig, axes = plt.subplots(2, 2, figsize=(15, 10))
       
       # Force calculation breakdown
       force_data = profiling_results['force_breakdown']
       force_types = list(force_data.keys())
       force_times = [force_data[ft]['mean_time'] for ft in force_types]
       
       axes[0, 0].pie(force_times, labels=force_types, autopct='%1.1f%%')
       axes[0, 0].set_title('Force Calculation Breakdown')
       
       # Scaling analysis
       scaling_data = profiling_results['scaling']
       n_atoms = [d['n_atoms'] for d in scaling_data]
       step_times = [d['step_time'] for d in scaling_data]
       
       axes[0, 1].loglog(n_atoms, step_times, 'o-')
       axes[0, 1].set_xlabel('Number of Atoms')
       axes[0, 1].set_ylabel('Step Time (s)')
       axes[0, 1].set_title('Performance Scaling')
       axes[0, 1].grid(True)
       
       # Memory usage over time
       memory_data = profiling_results['memory_usage']
       times = range(len(memory_data))
       memory_mb = [m / 1024**2 for m in memory_data]
       
       axes[1, 0].plot(times, memory_mb)
       axes[1, 0].set_xlabel('Time Step')
       axes[1, 0].set_ylabel('Memory Usage (MB)')
       axes[1, 0].set_title('Memory Usage Over Time')
       
       # Performance timeline
       timeline_data = profiling_results['timeline']
       timestamps = [datetime.datetime.fromisoformat(t['timestamp']) for t in timeline_data]
       throughput = [t['performance']['steps_per_second'] for t in timeline_data]
       
       axes[1, 1].plot(timestamps, throughput, 'o-')
       axes[1, 1].set_xlabel('Date')
       axes[1, 1].set_ylabel('Steps per Second')
       axes[1, 1].set_title('Performance Over Time')
       axes[1, 1].tick_params(axis='x', rotation=45)
       
       plt.tight_layout()
       plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight')
       
   def generate_performance_report(profiling_results, output_file='performance_report.html'):
       """Generate comprehensive HTML performance report."""
       html_template = """
       <!DOCTYPE html>
       <html>
       <head>
           <title>ProteinMD Performance Report</title>
           <style>
               body { font-family: Arial, sans-serif; margin: 40px; }
               .metric { background: #f0f0f0; padding: 10px; margin: 10px 0; }
               .warning { background: #ffeeaa; border-left: 4px solid #ff9900; }
               .good { background: #eeffee; border-left: 4px solid #00aa00; }
               table { border-collapse: collapse; width: 100%; }
               th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
               th { background-color: #f2f2f2; }
           </style>
       </head>
       <body>
           <h1>ProteinMD Performance Report</h1>
           <h2>Summary</h2>
           <div class="metric">
               <strong>Total Simulation Time:</strong> {total_time:.4f} seconds<br>
               <strong>Steps per Second:</strong> {steps_per_second:.2f}<br>
               <strong>Atoms per Step per Second:</strong> {atoms_per_step_per_second:.2e}
           </div>
           
           <h2>Force Calculation Breakdown</h2>
           <table>
               <tr><th>Force Type</th><th>Mean Time (s)</th><th>Percentage</th></tr>
               {force_table_rows}
           </table>
           
           <h2>Performance Recommendations</h2>
           {recommendations}
           
           <h2>Detailed Metrics</h2>
           <img src="performance_dashboard.png" alt="Performance Dashboard">
       </body>
       </html>
       """
       
       # Generate force table rows
       force_breakdown = profiling_results['force_breakdown']
       force_rows = ""
       for force_type, data in force_breakdown.items():
           force_rows += f"<tr><td>{force_type}</td><td>{data['mean_time']:.4f}</td><td>{data['percentage']:.1f}%</td></tr>"
       
       # Generate recommendations
       recommendations = generate_recommendations(profiling_results)
       
       # Fill template
       html_content = html_template.format(
           total_time=profiling_results['summary']['total_time'],
           steps_per_second=profiling_results['summary']['steps_per_second'],
           atoms_per_step_per_second=profiling_results['summary']['atoms_per_step_per_second'],
           force_table_rows=force_rows,
           recommendations=recommendations
       )
       
       with open(output_file, 'w') as f:
           f.write(html_content)

Best Practices
--------------

Profiling Guidelines
~~~~~~~~~~~~~~~~~~~

1. **Profile Realistic Workloads**
   - Use representative system sizes
   - Include typical simulation conditions
   - Profile complete workflows, not just individual functions

2. **Profile at Multiple Levels**
   - Function-level profiling for hotspots
   - Line-level profiling for detailed optimization
   - System-level profiling for resource utilization

3. **Establish Baselines**
   - Record performance before optimization
   - Use version control for performance tracking
   - Document hardware and software configurations

4. **Focus on Bottlenecks**
   - Optimize the slowest components first
   - Consider Amdahl's law for parallel optimizations
   - Balance development effort with potential gains

Common Profiling Pitfalls
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Profiling Debug Builds**
   - Always profile optimized/release builds
   - Disable debugging flags and assertions
   - Use appropriate compiler optimizations

2. **Insufficient Sample Sizes**
   - Run multiple iterations for statistical significance
   - Account for system noise and variability
   - Use warmup runs to eliminate cold start effects

3. **Micro-benchmarking Errors**
   - Avoid optimizing unrealistic code patterns
   - Consider whole-system performance impact
   - Validate optimizations with real workloads

Tools Summary
------------

**Python Profiling:**
- ``cProfile``: Built-in function profiler
- ``line_profiler``: Line-by-line profiling
- ``memory_profiler``: Memory usage profiling
- ``py-spy``: Sampling profiler for production

**GPU Profiling:**
- ``nvprof``: Legacy CUDA profiler
- ``nsys``: Nsight Systems for modern profiling
- ``ncu``: Nsight Compute for kernel analysis

**Visualization:**
- ``snakeviz``: Interactive cProfile visualization
- ``gprof2dot``: Call graph visualization
- ``matplotlib/seaborn``: Custom performance plots

**System Profiling:**
- ``htop/top``: System resource monitoring
- ``perf``: Linux performance analysis
- ``Intel VTune``: Comprehensive profiler

This comprehensive profiling guide provides the tools and techniques needed to identify and eliminate performance bottlenecks in ProteinMD simulations, ensuring optimal performance across different system configurations and scales.
