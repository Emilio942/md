Benchmarking Guide
==================

This guide provides comprehensive benchmarking strategies for ProteinMD, covering performance measurement, regression testing, and comparative analysis across different configurations and platforms.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

Benchmarking is essential for quantifying performance improvements, detecting regressions, and comparing different implementations. This guide establishes standardized benchmarking practices for ProteinMD development.

Benchmarking Philosophy
~~~~~~~~~~~~~~~~~~~~~~

- **Reproducible results:** Consistent methodology and environments
- **Representative workloads:** Realistic simulation scenarios
- **Statistical rigor:** Multiple runs and statistical analysis
- **Comprehensive coverage:** Different system sizes and configurations
- **Automated tracking:** Continuous performance monitoring

Benchmark Suite Design
---------------------

Core Benchmark Categories
~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Micro-benchmarks: Individual Components**

.. code-block:: python

   import numpy as np
   import time
   import statistics
   from typing import Dict, List, Callable
   
   class MicroBenchmark:
       """Benchmark individual functions and operations."""
       
       def __init__(self, n_runs=10, warmup_runs=3):
           self.n_runs = n_runs
           self.warmup_runs = warmup_runs
           
       def benchmark_force_calculation(self):
           """Benchmark different force calculation methods."""
           n_atoms = 1000
           positions = np.random.random((n_atoms, 3)).astype(np.float32)
           charges = np.random.random(n_atoms).astype(np.float32)
           
           def naive_coulomb():
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
           
           def vectorized_coulomb():
               # Vectorized implementation
               forces = np.zeros_like(positions)
               for i in range(n_atoms):
                   r_vectors = positions[i] - positions[i+1:]
                   r_magnitudes = np.linalg.norm(r_vectors, axis=1)
                   valid = r_magnitudes > 0
                   
                   force_mags = (charges[i] * charges[i+1:][valid] / 
                                r_magnitudes[valid]**3)
                   force_vectors = force_mags[:, np.newaxis] * r_vectors[valid]
                   
                   forces[i] += np.sum(force_vectors, axis=0)
                   forces[i+1:][valid] -= force_vectors
               return forces
           
           results = {}
           for name, func in [('naive', naive_coulomb), ('vectorized', vectorized_coulomb)]:
               times = self._time_function(func)
               results[name] = {
                   'mean_time': np.mean(times),
                   'std_time': np.std(times),
                   'speedup': 1.0  # Will calculate relative to baseline
               }
           
           # Calculate speedups relative to naive implementation
           baseline_time = results['naive']['mean_time']
           for result in results.values():
               result['speedup'] = baseline_time / result['mean_time']
           
           return results
       
       def _time_function(self, func: Callable) -> List[float]:
           """Time function execution with warmup."""
           # Warmup runs
           for _ in range(self.warmup_runs):
               func()
           
           # Actual timing runs
           times = []
           for _ in range(self.n_runs):
               start_time = time.perf_counter()
               func()
               end_time = time.perf_counter()
               times.append(end_time - start_time)
           
           return times

**2. System-level Benchmarks: Complete Simulations**

.. code-block:: python

   class SystemBenchmark:
       """Benchmark complete simulation systems."""
       
       def __init__(self):
           self.benchmark_configs = {
               'small_protein': {
                   'n_atoms': 1000,
                   'box_size': [5.0, 5.0, 5.0],
                   'cutoff': 1.2,
                   'steps': 1000
               },
               'medium_protein': {
                   'n_atoms': 10000,
                   'box_size': [10.0, 10.0, 10.0],
                   'cutoff': 1.2,
                   'steps': 1000
               },
               'large_protein': {
                   'n_atoms': 100000,
                   'box_size': [20.0, 20.0, 20.0],
                   'cutoff': 1.2,
                   'steps': 500
               }
           }
       
       def run_benchmark_suite(self) -> Dict:
           """Run complete benchmark suite."""
           results = {}
           
           for config_name, config in self.benchmark_configs.items():
               print(f"Running benchmark: {config_name}")
               result = self._benchmark_configuration(config)
               results[config_name] = result
               
           return results
       
       def _benchmark_configuration(self, config: Dict) -> Dict:
           """Benchmark specific configuration."""
           # Create simulation
           simulation = self._create_simulation(config)
           
           # Measure different aspects
           results = {
               'config': config,
               'initialization_time': self._measure_initialization(simulation),
               'step_performance': self._measure_step_performance(simulation, config['steps']),
               'memory_usage': self._measure_memory_usage(simulation),
               'force_breakdown': self._measure_force_breakdown(simulation)
           }
           
           return results
       
       def _measure_step_performance(self, simulation, n_steps: int) -> Dict:
           """Measure simulation step performance."""
           # Warmup
           for _ in range(10):
               simulation.step()
           
           # Measure performance
           start_time = time.perf_counter()
           for _ in range(n_steps):
               simulation.step()
           total_time = time.perf_counter() - start_time
           
           return {
               'total_time': total_time,
               'time_per_step': total_time / n_steps,
               'steps_per_second': n_steps / total_time,
               'ns_per_day': self._calculate_ns_per_day(simulation.timestep, total_time, n_steps)
           }
       
       def _calculate_ns_per_day(self, timestep: float, total_time: float, n_steps: int) -> float:
           """Calculate nanoseconds of simulation per day of real time."""
           simulated_time_ns = timestep * n_steps * 1e-3  # Convert fs to ns
           seconds_per_day = 24 * 60 * 60
           time_per_step = total_time / n_steps
           steps_per_day = seconds_per_day / time_per_step
           return timestep * steps_per_day * 1e-3

**3. Scaling Benchmarks: Performance vs. System Size**

.. code-block:: python

   class ScalingBenchmark:
       """Benchmark performance scaling with system size."""
       
       def __init__(self):
           self.system_sizes = [500, 1000, 2000, 5000, 10000, 20000, 50000]
           
       def run_scaling_analysis(self) -> Dict:
           """Run comprehensive scaling analysis."""
           results = {
               'strong_scaling': self._strong_scaling_benchmark(),
               'weak_scaling': self._weak_scaling_benchmark(),
               'memory_scaling': self._memory_scaling_benchmark()
           }
           
           return results
       
       def _strong_scaling_benchmark(self) -> List[Dict]:
           """Fixed problem size, varying computational resources."""
           results = []
           
           for n_atoms in self.system_sizes:
               config = {
                   'n_atoms': n_atoms,
                   'density': 1.0,  # Fixed density
                   'steps': 100
               }
               
               simulation = self._create_simulation(config)
               
               start_time = time.perf_counter()
               simulation.run(config['steps'])
               total_time = time.perf_counter() - start_time
               
               results.append({
                   'n_atoms': n_atoms,
                   'total_time': total_time,
                   'time_per_step': total_time / config['steps'],
                   'efficiency': self._calculate_efficiency(n_atoms, total_time)
               })
               
           return results
       
       def _weak_scaling_benchmark(self) -> List[Dict]:
           """Fixed problem size per processor."""
           # For single-threaded analysis, this is similar to strong scaling
           # In multi-threaded scenarios, this would fix atoms per thread
           return self._strong_scaling_benchmark()
       
       def _memory_scaling_benchmark(self) -> List[Dict]:
           """Analyze memory usage vs. system size."""
           import psutil
           import os
           
           results = []
           process = psutil.Process(os.getpid())
           
           for n_atoms in self.system_sizes:
               # Measure baseline memory
               baseline_memory = process.memory_info().rss
               
               # Create simulation
               config = {'n_atoms': n_atoms, 'density': 1.0}
               simulation = self._create_simulation(config)
               
               # Measure memory after creation
               simulation_memory = process.memory_info().rss
               memory_usage = simulation_memory - baseline_memory
               
               results.append({
                   'n_atoms': n_atoms,
                   'memory_bytes': memory_usage,
                   'memory_per_atom': memory_usage / n_atoms,
                   'memory_mb': memory_usage / 1024**2
               })
               
               # Clean up
               del simulation
               
           return results

Platform-Specific Benchmarks
---------------------------

CPU Benchmarks
~~~~~~~~~~~~~

.. code-block:: python

   class CPUBenchmark:
       """CPU-specific benchmarking."""
       
       def __init__(self):
           self.cpu_info = self._get_cpu_info()
           
       def _get_cpu_info(self) -> Dict:
           """Get CPU information."""
           import platform
           import psutil
           
           return {
               'processor': platform.processor(),
               'cpu_count': psutil.cpu_count(logical=False),
               'cpu_count_logical': psutil.cpu_count(logical=True),
               'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else None
           }
       
       def benchmark_vectorization(self) -> Dict:
           """Benchmark vectorization performance."""
           n = 1000000
           a = np.random.random(n).astype(np.float32)
           b = np.random.random(n).astype(np.float32)
           
           # Scalar operations
           def scalar_operations():
               result = np.zeros(n, dtype=np.float32)
               for i in range(n):
                   result[i] = a[i] * b[i] + np.sin(a[i])
               return result
           
           # Vectorized operations
           def vectorized_operations():
               return a * b + np.sin(a)
           
           # NumPy optimized operations
           def numpy_optimized():
               return np.fma(a, b, np.sin(a))  # Fused multiply-add
           
           results = {}
           for name, func in [('scalar', scalar_operations), 
                            ('vectorized', vectorized_operations),
                            ('numpy_optimized', numpy_optimized)]:
               times = []
               for _ in range(5):
                   start_time = time.perf_counter()
                   result = func()
                   times.append(time.perf_counter() - start_time)
               
               results[name] = {
                   'mean_time': np.mean(times),
                   'throughput_gflops': (n * 2) / np.mean(times) / 1e9  # 2 ops per element
               }
           
           return results
       
       def benchmark_memory_bandwidth(self) -> Dict:
           """Benchmark memory bandwidth."""
           sizes_mb = [1, 10, 100, 1000]  # Different sizes to test cache hierarchy
           results = {}
           
           for size_mb in sizes_mb:
               n_elements = size_mb * 1024 * 1024 // 4  # float32
               data = np.random.random(n_elements).astype(np.float32)
               
               # Sequential read
               def sequential_read():
                   return np.sum(data)
               
               # Sequential write
               def sequential_write():
                   data[:] = 1.0
               
               # Random access
               indices = np.random.randint(0, n_elements, size=n_elements//10)
               def random_access():
                   return np.sum(data[indices])
               
               size_results = {}
               for name, func in [('sequential_read', sequential_read),
                                ('sequential_write', sequential_write),
                                ('random_access', random_access)]:
                   times = []
                   for _ in range(3):
                       start_time = time.perf_counter()
                       func()
                       times.append(time.perf_counter() - start_time)
                   
                   bandwidth_gb_s = (size_mb / 1024) / np.mean(times)
                   size_results[name] = {
                       'mean_time': np.mean(times),
                       'bandwidth_gb_s': bandwidth_gb_s
                   }
               
               results[f'{size_mb}MB'] = size_results
           
           return results

GPU Benchmarks
~~~~~~~~~~~~~

.. code-block:: python

   import cupy as cp
   
   class GPUBenchmark:
       """GPU-specific benchmarking."""
       
       def __init__(self):
           self.gpu_info = self._get_gpu_info()
           
       def _get_gpu_info(self) -> Dict:
           """Get GPU information."""
           if not cp.cuda.is_available():
               return {'available': False}
           
           device = cp.cuda.Device()
           return {
               'available': True,
               'name': device.attributes['Name'],
               'compute_capability': device.compute_capability,
               'memory_gb': device.mem_info[1] / 1024**3,
               'multiprocessor_count': device.attributes['MultiProcessorCount']
           }
       
       def benchmark_memory_transfer(self) -> Dict:
           """Benchmark CPU-GPU memory transfer rates."""
           if not self.gpu_info['available']:
               return {'error': 'GPU not available'}
           
           sizes_mb = [1, 10, 100, 1000]
           results = {}
           
           for size_mb in sizes_mb:
               n_elements = size_mb * 1024 * 1024 // 4  # float32
               cpu_data = np.random.random(n_elements).astype(np.float32)
               
               # Host to Device transfer
               times_h2d = []
               for _ in range(5):
                   start_time = time.perf_counter()
                   gpu_data = cp.asarray(cpu_data)
                   cp.cuda.Stream.null.synchronize()
                   times_h2d.append(time.perf_counter() - start_time)
               
               # Device to Host transfer
               times_d2h = []
               for _ in range(5):
                   start_time = time.perf_counter()
                   result = cp.asnumpy(gpu_data)
                   times_d2h.append(time.perf_counter() - start_time)
               
               results[f'{size_mb}MB'] = {
                   'h2d_time': np.mean(times_h2d),
                   'd2h_time': np.mean(times_d2h),
                   'h2d_bandwidth_gb_s': size_mb / 1024 / np.mean(times_h2d),
                   'd2h_bandwidth_gb_s': size_mb / 1024 / np.mean(times_d2h)
               }
           
           return results
       
       def benchmark_kernel_performance(self) -> Dict:
           """Benchmark GPU kernel performance."""
           if not self.gpu_info['available']:
               return {'error': 'GPU not available'}
           
           n = 10000000  # 10M elements
           a = cp.random.random(n, dtype=cp.float32)
           b = cp.random.random(n, dtype=cp.float32)
           
           # Simple arithmetic
           def gpu_arithmetic():
               return a * b + cp.sin(a)
           
           # Matrix operations
           def gpu_matrix_ops():
               matrix_size = int(np.sqrt(n))
               a_matrix = a[:matrix_size**2].reshape(matrix_size, matrix_size)
               b_matrix = b[:matrix_size**2].reshape(matrix_size, matrix_size)
               return cp.dot(a_matrix, b_matrix)
           
           # Reduction operations
           def gpu_reduction():
               return cp.sum(a) + cp.max(b)
           
           results = {}
           for name, func in [('arithmetic', gpu_arithmetic),
                            ('matrix_ops', gpu_matrix_ops),
                            ('reduction', gpu_reduction)]:
               # Warmup
               for _ in range(2):
                   func()
                   cp.cuda.Stream.null.synchronize()
               
               # Timing
               times = []
               for _ in range(10):
                   start_event = cp.cuda.Event()
                   end_event = cp.cuda.Event()
                   
                   start_event.record()
                   result = func()
                   end_event.record()
                   end_event.synchronize()
                   
                   elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event) / 1000  # Convert to seconds
                   times.append(elapsed_time)
               
               results[name] = {
                   'mean_time': np.mean(times),
                   'throughput_gflops': (n * 2) / np.mean(times) / 1e9  # Approximate FLOPS
               }
           
           return results

Automated Benchmark Infrastructure
---------------------------------

Continuous Benchmarking
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import json
   import datetime
   import subprocess
   import os
   from typing import Optional
   
   class BenchmarkRunner:
       """Automated benchmark execution and tracking."""
       
       def __init__(self, config_file='benchmark_config.json'):
           self.config = self._load_config(config_file)
           self.results_dir = self.config.get('results_dir', 'benchmark_results')
           os.makedirs(self.results_dir, exist_ok=True)
           
       def _load_config(self, config_file: str) -> Dict:
           """Load benchmark configuration."""
           default_config = {
               'benchmarks': ['micro', 'system', 'scaling'],
               'git_tracking': True,
               'system_info': True,
               'notification': {
                   'enabled': False,
                   'threshold_regression': 0.1
               }
           }
           
           try:
               with open(config_file, 'r') as f:
                   config = json.load(f)
                   default_config.update(config)
           except FileNotFoundError:
               pass
           
           return default_config
       
       def run_benchmarks(self) -> Dict:
           """Run complete benchmark suite."""
           timestamp = datetime.datetime.now()
           
           # Collect system information
           system_info = self._collect_system_info() if self.config['system_info'] else {}
           
           # Collect git information
           git_info = self._collect_git_info() if self.config['git_tracking'] else {}
           
           # Run benchmarks
           benchmark_results = {}
           
           if 'micro' in self.config['benchmarks']:
               benchmark_results['micro'] = MicroBenchmark().benchmark_force_calculation()
           
           if 'system' in self.config['benchmarks']:
               benchmark_results['system'] = SystemBenchmark().run_benchmark_suite()
           
           if 'scaling' in self.config['benchmarks']:
               benchmark_results['scaling'] = ScalingBenchmark().run_scaling_analysis()
           
           if 'cpu' in self.config['benchmarks']:
               benchmark_results['cpu'] = CPUBenchmark().benchmark_vectorization()
           
           if 'gpu' in self.config['benchmarks']:
               benchmark_results['gpu'] = GPUBenchmark().benchmark_kernel_performance()
           
           # Compile full results
           full_results = {
               'timestamp': timestamp.isoformat(),
               'system_info': system_info,
               'git_info': git_info,
               'benchmark_results': benchmark_results,
               'config': self.config
           }
           
           # Save results
           self._save_results(full_results, timestamp)
           
           # Check for regressions
           if self.config.get('regression_detection', True):
               regressions = self._detect_regressions(full_results)
               if regressions:
                   self._handle_regressions(regressions)
           
           return full_results
       
       def _collect_system_info(self) -> Dict:
           """Collect comprehensive system information."""
           import platform
           import psutil
           
           return {
               'python_version': platform.python_version(),
               'platform': platform.platform(),
               'processor': platform.processor(),
               'cpu_count': psutil.cpu_count(),
               'memory_gb': psutil.virtual_memory().total / 1024**3,
               'disk_usage_gb': psutil.disk_usage('.').total / 1024**3
           }
       
       def _collect_git_info(self) -> Dict:
           """Collect git repository information."""
           try:
               commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
               branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
               dirty = bool(subprocess.check_output(['git', 'diff', '--shortstat']).decode().strip())
               
               return {
                   'commit_hash': commit_hash,
                   'branch': branch,
                   'dirty': dirty
               }
           except subprocess.CalledProcessError:
               return {'error': 'Git information not available'}
       
       def _save_results(self, results: Dict, timestamp: datetime.datetime):
           """Save benchmark results."""
           filename = f"benchmark_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
           filepath = os.path.join(self.results_dir, filename)
           
           with open(filepath, 'w') as f:
               json.dump(results, f, indent=2, default=str)
       
       def _detect_regressions(self, current_results: Dict) -> List[Dict]:
           """Detect performance regressions."""
           # Load recent historical results
           history = self._load_recent_results(n_recent=5)
           if len(history) < 2:
               return []
           
           regressions = []
           threshold = self.config.get('notification', {}).get('threshold_regression', 0.1)
           
           # Compare with baseline (median of recent results)
           baseline_metrics = self._calculate_baseline_metrics(history)
           current_metrics = self._extract_key_metrics(current_results)
           
           for metric_name, current_value in current_metrics.items():
               if metric_name in baseline_metrics:
                   baseline_value = baseline_metrics[metric_name]
                   
                   # Calculate regression (positive means performance degraded)
                   if 'time' in metric_name.lower():
                       # Lower is better for time metrics
                       regression = (current_value - baseline_value) / baseline_value
                   else:
                       # Higher is better for throughput metrics
                       regression = (baseline_value - current_value) / baseline_value
                   
                   if regression > threshold:
                       regressions.append({
                           'metric': metric_name,
                           'regression_percent': regression * 100,
                           'current_value': current_value,
                           'baseline_value': baseline_value
                       })
           
           return regressions

Regression Testing
-----------------

Performance Regression Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class RegressionTester:
       """Detect and analyze performance regressions."""
       
       def __init__(self, history_file='benchmark_history.json'):
           self.history_file = history_file
           self.load_history()
           
       def load_history(self):
           """Load benchmark history."""
           try:
               with open(self.history_file, 'r') as f:
                   self.history = json.load(f)
           except FileNotFoundError:
               self.history = []
       
       def add_result(self, result: Dict):
           """Add new benchmark result to history."""
           self.history.append(result)
           
           # Keep only recent results (e.g., last 100)
           if len(self.history) > 100:
               self.history = self.history[-100:]
           
           # Save updated history
           with open(self.history_file, 'w') as f:
               json.dump(self.history, f, indent=2, default=str)
       
       def detect_regressions(self, significance_level=0.05) -> List[Dict]:
           """Detect statistically significant regressions."""
           if len(self.history) < 10:
               return []
           
           from scipy import stats
           
           # Get recent results for comparison
           recent_results = self.history[-5:]  # Last 5 runs
           historical_results = self.history[-20:-5]  # Previous 15 runs
           
           regressions = []
           
           # Extract key metrics
           metrics = self._extract_common_metrics(self.history)
           
           for metric_name in metrics:
               recent_values = [self._get_metric_value(r, metric_name) for r in recent_results]
               historical_values = [self._get_metric_value(r, metric_name) for r in historical_results]
               
               # Remove None values
               recent_values = [v for v in recent_values if v is not None]
               historical_values = [v for v in historical_values if v is not None]
               
               if len(recent_values) >= 3 and len(historical_values) >= 5:
                   # Perform t-test
                   statistic, p_value = stats.ttest_ind(recent_values, historical_values)
                   
                   # Check if regression is significant and in the wrong direction
                   mean_recent = np.mean(recent_values)
                   mean_historical = np.mean(historical_values)
                   
                   is_regression = False
                   if 'time' in metric_name.lower():
                       # For time metrics, recent > historical is bad
                       is_regression = mean_recent > mean_historical
                   else:
                       # For throughput metrics, recent < historical is bad
                       is_regression = mean_recent < mean_historical
                   
                   if p_value < significance_level and is_regression:
                       regression_magnitude = abs(mean_recent - mean_historical) / mean_historical
                       
                       regressions.append({
                           'metric': metric_name,
                           'p_value': p_value,
                           'regression_percent': regression_magnitude * 100,
                           'recent_mean': mean_recent,
                           'historical_mean': mean_historical,
                           'significance': 'high' if p_value < 0.01 else 'moderate'
                       })
           
           return regressions

Benchmark Reporting
------------------

Automated Reports
~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns
   
   class BenchmarkReporter:
       """Generate comprehensive benchmark reports."""
       
       def __init__(self, results_dir='benchmark_results'):
           self.results_dir = results_dir
           
       def generate_report(self, results: Dict, output_dir='reports') -> str:
           """Generate comprehensive benchmark report."""
           os.makedirs(output_dir, exist_ok=True)
           
           timestamp = results['timestamp']
           report_name = f"benchmark_report_{timestamp.replace(':', '-').split('.')[0]}"
           
           # Generate plots
           plot_files = self._generate_plots(results, output_dir, report_name)
           
           # Generate HTML report
           html_file = self._generate_html_report(results, plot_files, output_dir, report_name)
           
           # Generate summary text
           summary_file = self._generate_summary(results, output_dir, report_name)
           
           return html_file
       
       def _generate_plots(self, results: Dict, output_dir: str, report_name: str) -> List[str]:
           """Generate benchmark visualization plots."""
           plot_files = []
           
           # Performance over time (if historical data available)
           if self._has_historical_data():
               plot_file = self._plot_performance_timeline(output_dir, report_name)
               plot_files.append(plot_file)
           
           # Scaling analysis
           if 'scaling' in results['benchmark_results']:
               plot_file = self._plot_scaling_analysis(results['benchmark_results']['scaling'], 
                                                     output_dir, report_name)
               plot_files.append(plot_file)
           
           # Force breakdown
           if 'micro' in results['benchmark_results']:
               plot_file = self._plot_force_breakdown(results['benchmark_results']['micro'], 
                                                    output_dir, report_name)
               plot_files.append(plot_file)
           
           # Memory usage
           if 'system' in results['benchmark_results']:
               plot_file = self._plot_memory_usage(results['benchmark_results']['system'], 
                                                 output_dir, report_name)
               plot_files.append(plot_file)
           
           return plot_files
       
       def _plot_scaling_analysis(self, scaling_results: Dict, output_dir: str, report_name: str) -> str:
           """Plot scaling analysis results."""
           fig, axes = plt.subplots(2, 2, figsize=(15, 10))
           
           # Strong scaling - execution time
           strong_scaling = scaling_results['strong_scaling']
           n_atoms = [r['n_atoms'] for r in strong_scaling]
           times = [r['total_time'] for r in strong_scaling]
           
           axes[0, 0].loglog(n_atoms, times, 'o-', label='Actual')
           
           # Theoretical O(N) and O(N²) lines
           theoretical_n = [times[0] * n / n_atoms[0] for n in n_atoms]
           theoretical_n2 = [times[0] * (n / n_atoms[0])**2 for n in n_atoms]
           
           axes[0, 0].loglog(n_atoms, theoretical_n, '--', label='O(N)')
           axes[0, 0].loglog(n_atoms, theoretical_n2, '--', label='O(N²)')
           axes[0, 0].set_xlabel('Number of Atoms')
           axes[0, 0].set_ylabel('Execution Time (s)')
           axes[0, 0].set_title('Strong Scaling - Execution Time')
           axes[0, 0].legend()
           axes[0, 0].grid(True)
           
           # Performance efficiency
           efficiencies = [r['efficiency'] for r in strong_scaling]
           axes[0, 1].plot(n_atoms, efficiencies, 'o-')
           axes[0, 1].set_xlabel('Number of Atoms')
           axes[0, 1].set_ylabel('Efficiency')
           axes[0, 1].set_title('Scaling Efficiency')
           axes[0, 1].grid(True)
           
           # Memory scaling
           memory_scaling = scaling_results['memory_scaling']
           memory_mb = [r['memory_mb'] for r in memory_scaling]
           memory_atoms = [r['n_atoms'] for r in memory_scaling]
           
           axes[1, 0].plot(memory_atoms, memory_mb, 'o-')
           axes[1, 0].set_xlabel('Number of Atoms')
           axes[1, 0].set_ylabel('Memory Usage (MB)')
           axes[1, 0].set_title('Memory Scaling')
           axes[1, 0].grid(True)
           
           # Memory per atom
           memory_per_atom = [r['memory_per_atom'] for r in memory_scaling]
           axes[1, 1].plot(memory_atoms, memory_per_atom, 'o-')
           axes[1, 1].set_xlabel('Number of Atoms')
           axes[1, 1].set_ylabel('Memory per Atom (bytes)')
           axes[1, 1].set_title('Memory Efficiency')
           axes[1, 1].grid(True)
           
           plt.tight_layout()
           plot_file = os.path.join(output_dir, f'{report_name}_scaling.png')
           plt.savefig(plot_file, dpi=300, bbox_inches='tight')
           plt.close()
           
           return plot_file
       
       def _generate_html_report(self, results: Dict, plot_files: List[str], 
                               output_dir: str, report_name: str) -> str:
           """Generate HTML benchmark report."""
           html_template = """
           <!DOCTYPE html>
           <html>
           <head>
               <title>ProteinMD Benchmark Report</title>
               <style>
                   body {{ font-family: Arial, sans-serif; margin: 40px; }}
                   .header {{ background: #f0f0f0; padding: 20px; margin-bottom: 20px; }}
                   .metric {{ background: #f9f9f9; padding: 10px; margin: 10px 0; }}
                   .regression {{ background: #ffeeaa; border-left: 4px solid #ff9900; }}
                   .improvement {{ background: #eeffee; border-left: 4px solid #00aa00; }}
                   table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                   th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                   th {{ background-color: #f2f2f2; }}
                   .plot {{ text-align: center; margin: 20px 0; }}
                   .plot img {{ max-width: 100%; height: auto; }}
               </style>
           </head>
           <body>
               <div class="header">
                   <h1>ProteinMD Benchmark Report</h1>
                   <p><strong>Generated:</strong> {timestamp}</p>
                   <p><strong>Git Commit:</strong> {git_commit}</p>
                   <p><strong>Platform:</strong> {platform}</p>
               </div>
               
               <h2>Executive Summary</h2>
               {executive_summary}
               
               <h2>Performance Metrics</h2>
               {performance_metrics}
               
               <h2>Scaling Analysis</h2>
               {scaling_plots}
               
               <h2>Detailed Results</h2>
               {detailed_results}
               
               <h2>System Information</h2>
               {system_info}
           </body>
           </html>
           """
           
           # Fill template sections
           executive_summary = self._generate_executive_summary(results)
           performance_metrics = self._generate_performance_metrics_table(results)
           scaling_plots = self._generate_plot_section(plot_files)
           detailed_results = self._generate_detailed_results(results)
           system_info = self._generate_system_info_table(results)
           
           html_content = html_template.format(
               timestamp=results['timestamp'],
               git_commit=results.get('git_info', {}).get('commit_hash', 'Unknown'),
               platform=results.get('system_info', {}).get('platform', 'Unknown'),
               executive_summary=executive_summary,
               performance_metrics=performance_metrics,
               scaling_plots=scaling_plots,
               detailed_results=detailed_results,
               system_info=system_info
           )
           
           html_file = os.path.join(output_dir, f'{report_name}.html')
           with open(html_file, 'w') as f:
               f.write(html_content)
           
           return html_file

Best Practices
--------------

Benchmarking Guidelines
~~~~~~~~~~~~~~~~~~~~~~

1. **Environment Control**
   - Use dedicated benchmark machines
   - Disable power management features
   - Control system load during benchmarking
   - Use consistent compiler flags and optimizations

2. **Statistical Rigor**
   - Run multiple iterations for statistical significance
   - Use warmup runs to eliminate cold start effects
   - Report confidence intervals and standard deviations
   - Apply appropriate statistical tests for regression detection

3. **Reproducibility**
   - Document exact hardware and software configurations
   - Use version control for benchmark code
   - Save complete benchmark parameters
   - Provide scripts for benchmark reproduction

4. **Comprehensive Coverage**
   - Test multiple system sizes and configurations
   - Include both synthetic and realistic workloads
   - Cover different algorithms and implementations
   - Test edge cases and boundary conditions

Common Pitfalls
~~~~~~~~~~~~~~

1. **Measurement Errors**
   - Timer resolution limitations
   - System noise and interference
   - Insufficient measurement duration
   - Cache and memory effects

2. **Comparison Issues**
   - Different optimization levels
   - Unfair algorithm comparisons
   - Platform-specific optimizations
   - Version differences

3. **Statistical Problems**
   - Insufficient sample sizes
   - Ignoring measurement variance
   - Cherry-picking favorable results
   - Misinterpreting statistical significance

Tools and Infrastructure
-----------------------

Recommended Tools
~~~~~~~~~~~~~~~

**Performance Measurement:**
- ``time.perf_counter()``: High-resolution timing
- ``cProfile``: Function-level profiling
- ``pytest-benchmark``: Automated microbenchmarking
- ``asv`` (airspeed velocity): Continuous benchmarking

**Statistical Analysis:**
- ``scipy.stats``: Statistical testing
- ``numpy``: Numerical computations
- ``pandas``: Data manipulation
- ``matplotlib/seaborn``: Visualization

**System Monitoring:**
- ``psutil``: System resource monitoring
- ``nvidia-ml-py``: GPU monitoring
- ``perf``: Hardware performance counters

**Automation:**
- ``pytest``: Test framework integration
- ``Jenkins/GitHub Actions``: CI/CD integration
- ``Docker``: Consistent environments

This comprehensive benchmarking guide provides the foundation for systematic performance evaluation in ProteinMD development, enabling data-driven optimization decisions and continuous performance monitoring.
