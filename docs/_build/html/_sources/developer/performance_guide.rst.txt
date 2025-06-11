Performance Guide
================

This guide covers performance optimization strategies, profiling techniques, and best practices for high-performance molecular dynamics simulations in ProteinMD.

.. contents:: Contents
   :local:
   :depth: 2

Performance Philosophy
---------------------

Design Principles
~~~~~~~~~~~~~~~~

**Correct First, Fast Second**
  Always prioritize correctness over performance. Optimize only after ensuring the implementation is scientifically accurate.

**Profile Before Optimizing**
  Use data-driven optimization. Measure performance bottlenecks before making changes.

**Consider the Algorithm**
  Often, algorithmic improvements provide greater benefits than micro-optimizations.

**Readable Optimizations**
  Maintain code readability even when optimizing. Document performance-critical sections clearly.

Performance Targets
~~~~~~~~~~~~~~~~~~

**Target Performance Metrics**

.. code-block:: text

   System Size Performance Targets
   ===============================
   
   Small Systems (< 1,000 atoms):
   - Initialization: < 100 ms
   - MD Step: < 1 ms
   - Analysis: < 10 ms
   
   Medium Systems (1,000 - 10,000 atoms):
   - Initialization: < 1 s
   - MD Step: < 10 ms
   - Analysis: < 100 ms
   
   Large Systems (10,000 - 100,000 atoms):
   - Initialization: < 10 s
   - MD Step: < 100 ms
   - Analysis: < 1 s
   
   Very Large Systems (> 100,000 atoms):
   - Should utilize GPU acceleration
   - MD Step: < 1 s
   - Memory usage: < 50% of available RAM

**Scalability Requirements**

- **Linear scaling** for most operations with system size
- **Efficient memory usage** - O(N) memory complexity
- **GPU acceleration** available for compute-intensive operations
- **Parallel processing** support for independent calculations

Profiling and Benchmarking
--------------------------

Profiling Tools
~~~~~~~~~~~~~~

**Python Profilers**

.. code-block:: python

   import cProfile
   import pstats
   from proteinMD.core import MDSystem
   
   def profile_simulation():
       """Profile a complete MD simulation."""
       system = create_test_system(n_atoms=10000)
       
       # Profile with cProfile
       pr = cProfile.Profile()
       pr.enable()
       
       system.run_simulation(steps=1000)
       
       pr.disable()
       
       # Analyze results
       stats = pstats.Stats(pr)
       stats.sort_stats('cumulative')
       stats.print_stats(20)  # Top 20 functions
   
   # Usage
   profile_simulation()

**Line Profiler**

.. code-block:: python

   # Install: pip install line_profiler
   
   @profile  # Add this decorator
   def calculate_forces(positions, parameters):
       """Calculate forces - line by line profiling."""
       forces = np.zeros_like(positions)
       
       # This line will be profiled
       for i in range(len(positions)):
           for j in range(i+1, len(positions)):
               # Profile each line
               r = np.linalg.norm(positions[i] - positions[j])
               force = lennard_jones_force(r, parameters)
               forces[i] += force
               forces[j] -= force
       
       return forces
   
   # Run with: kernprof -l -v script.py

**Memory Profiler**

.. code-block:: python

   from memory_profiler import profile
   
   @profile
   def memory_intensive_function():
       """Monitor memory usage line by line."""
       # Large array allocation
       positions = np.random.random((100000, 3))
       
       # Force calculation
       forces = calculate_all_forces(positions)
       
       # Trajectory storage
       trajectory = np.zeros((1000, 100000, 3))
       
       return trajectory
   
   # Run with: python -m memory_profiler script.py

**CPU and GPU Profiling**

.. code-block:: python

   import time
   import psutil
   import GPUtil
   
   class PerformanceMonitor:
       """Monitor CPU and GPU performance during simulation."""
       
       def __init__(self):
           self.start_time = None
           self.cpu_percent = []
           self.memory_usage = []
           self.gpu_usage = []
       
       def start_monitoring(self):
           """Start performance monitoring."""
           self.start_time = time.time()
           
       def record_metrics(self):
           """Record current performance metrics."""
           # CPU usage
           cpu = psutil.cpu_percent(interval=None)
           self.cpu_percent.append(cpu)
           
           # Memory usage
           memory = psutil.virtual_memory()
           self.memory_usage.append(memory.percent)
           
           # GPU usage (if available)
           try:
               gpus = GPUtil.getGPUs()
               if gpus:
                   gpu_load = gpus[0].load * 100
                   self.gpu_usage.append(gpu_load)
           except:
               pass
       
       def generate_report(self):
           """Generate performance report."""
           total_time = time.time() - self.start_time
           
           report = f"""
           Performance Report
           ==================
           Total Time: {total_time:.2f} seconds
           Average CPU Usage: {np.mean(self.cpu_percent):.1f}%
           Peak Memory Usage: {max(self.memory_usage):.1f}%
           Average GPU Usage: {np.mean(self.gpu_usage):.1f}%
           """
           
           return report

Benchmarking Framework
~~~~~~~~~~~~~~~~~~~~~

**Automated Benchmarks**

.. code-block:: python

   import time
   import numpy as np
   from typing import Dict, List, Callable
   
   class BenchmarkSuite:
       """Comprehensive benchmarking suite for ProteinMD."""
       
       def __init__(self):
           self.results: Dict[str, List[float]] = {}
       
       def benchmark_function(
           self,
           func: Callable,
           name: str,
           *args,
           **kwargs
       ) -> float:
           """Benchmark a single function."""
           # Warm up
           for _ in range(3):
               func(*args, **kwargs)
           
           # Actual benchmark
           times = []
           for _ in range(10):
               start = time.perf_counter()
               result = func(*args, **kwargs)
               end = time.perf_counter()
               times.append(end - start)
           
           avg_time = np.mean(times)
           std_time = np.std(times)
           
           self.results[name] = times
           
           print(f"{name}: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
           return avg_time
       
       def benchmark_scaling(
           self,
           func: Callable,
           name: str,
           sizes: List[int],
           create_args: Callable[[int], tuple]
       ) -> Dict[int, float]:
           """Benchmark function scaling with system size."""
           scaling_results = {}
           
           for size in sizes:
               args = create_args(size)
               time_taken = self.benchmark_function(
                   func, f"{name}_{size}", *args
               )
               scaling_results[size] = time_taken
               
           return scaling_results
       
       def generate_report(self) -> str:
           """Generate comprehensive benchmark report."""
           report = "ProteinMD Performance Benchmark\n"
           report += "=" * 40 + "\n\n"
           
           for name, times in self.results.items():
               avg = np.mean(times) * 1000
               std = np.std(times) * 1000
               report += f"{name:30s}: {avg:8.2f} ± {std:6.2f} ms\n"
           
           return report

**Example Benchmark Suite**

.. code-block:: python

   def run_performance_benchmarks():
       """Run comprehensive performance benchmarks."""
       suite = BenchmarkSuite()
       
       # Force calculation benchmarks
       def create_force_test(n_atoms):
           positions = np.random.random((n_atoms, 3)) * 10
           return (positions,)
       
       scaling_results = suite.benchmark_scaling(
           func=calculate_nonbonded_forces,
           name="nonbonded_forces",
           sizes=[100, 500, 1000, 5000, 10000],
           create_args=create_force_test
       )
       
       # Integration benchmarks
       def create_integration_test(n_atoms):
           positions = np.random.random((n_atoms, 3))
           velocities = np.random.random((n_atoms, 3))
           forces = np.random.random((n_atoms, 3))
           masses = np.ones(n_atoms)
           return positions, velocities, forces, masses, 0.002
       
       suite.benchmark_scaling(
           func=velocity_verlet_step,
           name="integration",
           sizes=[1000, 5000, 10000, 50000],
           create_args=create_integration_test
       )
       
       # Print results
       print(suite.generate_report())
       
       return scaling_results

Optimization Strategies
----------------------

Algorithmic Optimization
~~~~~~~~~~~~~~~~~~~~~~~

**Neighbor Lists**

Optimize force calculations with neighbor lists:

.. code-block:: python

   class NeighborList:
       """Efficient neighbor list for force calculations."""
       
       def __init__(self, cutoff: float, skin: float = 1.0):
           self.cutoff = cutoff
           self.skin = skin
           self.list_cutoff = cutoff + skin
           self.neighbors = {}
           self.last_positions = None
           self.update_frequency = 20  # Update every N steps
           
       def needs_update(self, positions: np.ndarray) -> bool:
           """Check if neighbor list needs updating."""
           if self.last_positions is None:
               return True
           
           # Check maximum displacement
           max_displacement = np.max(
               np.linalg.norm(positions - self.last_positions, axis=1)
           )
           
           return max_displacement > self.skin / 2
       
       def update(self, positions: np.ndarray) -> None:
           """Update neighbor list using spatial decomposition."""
           n_atoms = len(positions)
           
           # Clear existing neighbors
           self.neighbors = {i: [] for i in range(n_atoms)}
           
           # Use spatial binning for O(N) scaling
           bins = self._create_spatial_bins(positions)
           
           for i in range(n_atoms):
               # Only check nearby bins
               nearby_atoms = self._get_nearby_atoms(i, positions, bins)
               
               for j in nearby_atoms:
                   if i < j:  # Avoid double counting
                       distance = np.linalg.norm(positions[i] - positions[j])
                       if distance < self.list_cutoff:
                           self.neighbors[i].append(j)
                           self.neighbors[j].append(i)
           
           self.last_positions = positions.copy()
       
       def get_neighbors(self, atom_idx: int) -> List[int]:
           """Get neighbors for specific atom."""
           return self.neighbors.get(atom_idx, [])

**Spatial Data Structures**

.. code-block:: python

   class SpatialGrid:
       """3D spatial grid for efficient neighbor searching."""
       
       def __init__(self, box_size: np.ndarray, cell_size: float):
           self.box_size = box_size
           self.cell_size = cell_size
           self.grid_dims = np.ceil(box_size / cell_size).astype(int)
           self.grid = {}
       
       def _get_cell_index(self, position: np.ndarray) -> tuple:
           """Get grid cell index for position."""
           indices = np.floor(position / self.cell_size).astype(int)
           # Handle periodic boundary conditions
           indices = indices % self.grid_dims
           return tuple(indices)
       
       def add_particle(self, particle_id: int, position: np.ndarray):
           """Add particle to spatial grid."""
           cell = self._get_cell_index(position)
           if cell not in self.grid:
               self.grid[cell] = []
           self.grid[cell].append(particle_id)
       
       def get_nearby_particles(
           self, 
           position: np.ndarray, 
           radius: float
       ) -> List[int]:
           """Get all particles within radius of position."""
           nearby = []
           
           # Calculate which cells to check
           cell_radius = int(np.ceil(radius / self.cell_size))
           center_cell = self._get_cell_index(position)
           
           for dx in range(-cell_radius, cell_radius + 1):
               for dy in range(-cell_radius, cell_radius + 1):
                   for dz in range(-cell_radius, cell_radius + 1):
                       cell = (
                           (center_cell[0] + dx) % self.grid_dims[0],
                           (center_cell[1] + dy) % self.grid_dims[1],
                           (center_cell[2] + dz) % self.grid_dims[2]
                       )
                       
                       if cell in self.grid:
                           nearby.extend(self.grid[cell])
           
           return nearby

NumPy Optimization
~~~~~~~~~~~~~~~~~

**Vectorization**

Replace loops with NumPy operations:

.. code-block:: python

   # Slow: Python loops
   def calculate_distances_slow(positions):
       n = len(positions)
       distances = np.zeros((n, n))
       
       for i in range(n):
           for j in range(n):
               if i != j:
                   diff = positions[i] - positions[j]
                   distances[i, j] = np.sqrt(np.sum(diff**2))
       
       return distances

   # Fast: Vectorized operations
   def calculate_distances_fast(positions):
       # Broadcasting: (n, 1, 3) - (1, n, 3) = (n, n, 3)
       diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
       distances = np.linalg.norm(diff, axis=2)
       return distances

   # Even faster: Optimized for specific use cases
   def calculate_distances_optimized(positions, cutoff=None):
       from scipy.spatial.distance import pdist, squareform
       
       if cutoff is None:
           # Use scipy for all distances
           distances = squareform(pdist(positions))
       else:
           # Use KDTree for cutoff-based calculations
           from scipy.spatial import KDTree
           tree = KDTree(positions)
           pairs = tree.query_pairs(cutoff)
           # Build sparse distance matrix
           n = len(positions)
           distances = np.zeros((n, n))
           for i, j in pairs:
               dist = np.linalg.norm(positions[i] - positions[j])
               distances[i, j] = distances[j, i] = dist
       
       return distances

**Memory Layout Optimization**

.. code-block:: python

   # Optimize memory access patterns
   class OptimizedArrays:
       """Memory-optimized array layouts for MD simulations."""
       
       def __init__(self, n_atoms: int):
           # Structure of Arrays (SoA) layout for better vectorization
           self.x = np.zeros(n_atoms, dtype=np.float32)
           self.y = np.zeros(n_atoms, dtype=np.float32)  
           self.z = np.zeros(n_atoms, dtype=np.float32)
           
           # Forces
           self.fx = np.zeros(n_atoms, dtype=np.float32)
           self.fy = np.zeros(n_atoms, dtype=np.float32)
           self.fz = np.zeros(n_atoms, dtype=np.float32)
       
       def get_positions(self) -> np.ndarray:
           """Get positions in standard (N, 3) format."""
           return np.column_stack([self.x, self.y, self.z])
       
       def set_positions(self, positions: np.ndarray) -> None:
           """Set positions from standard (N, 3) format."""
           self.x[:] = positions[:, 0]
           self.y[:] = positions[:, 1] 
           self.z[:] = positions[:, 2]
       
       def calculate_distance_squared_vectorized(self, i: int, j_array: np.ndarray):
           """Vectorized distance calculation using SoA layout."""
           dx = self.x[i] - self.x[j_array]
           dy = self.y[i] - self.y[j_array]
           dz = self.z[i] - self.z[j_array]
           return dx*dx + dy*dy + dz*dz

CUDA Acceleration
~~~~~~~~~~~~~~~

**GPU Memory Management**

.. code-block:: python

   import cupy as cp  # CUDA Python
   
   class CudaForceCalculator:
       """GPU-accelerated force calculations."""
       
       def __init__(self, n_atoms: int):
           self.n_atoms = n_atoms
           
           # Allocate GPU memory
           self.positions_gpu = cp.zeros((n_atoms, 3), dtype=cp.float32)
           self.forces_gpu = cp.zeros((n_atoms, 3), dtype=cp.float32)
           self.parameters_gpu = None
           
           # Compile CUDA kernels
           self._compile_kernels()
       
       def _compile_kernels(self):
           """Compile CUDA kernels for force calculations."""
           self.lennard_jones_kernel = cp.RawKernel(r'''
           extern "C" __global__
           void lennard_jones_forces(
               const float* positions,
               float* forces,
               const float* parameters,
               int n_atoms,
               float cutoff
           ) {
               int i = blockIdx.x * blockDim.x + threadIdx.x;
               if (i >= n_atoms) return;
               
               float fx = 0.0f, fy = 0.0f, fz = 0.0f;
               float xi = positions[3*i], yi = positions[3*i+1], zi = positions[3*i+2];
               
               for (int j = 0; j < n_atoms; j++) {
                   if (i == j) continue;
                   
                   float dx = xi - positions[3*j];
                   float dy = yi - positions[3*j+1];
                   float dz = zi - positions[3*j+2];
                   
                   float r2 = dx*dx + dy*dy + dz*dz;
                   if (r2 < cutoff*cutoff) {
                       // Lennard-Jones calculation
                       float sigma = parameters[0];
                       float epsilon = parameters[1];
                       
                       float sigma2 = sigma*sigma;
                       float sigma6 = sigma2*sigma2*sigma2;
                       float sigma12 = sigma6*sigma6;
                       
                       float r6 = r2*r2*r2;
                       float r12 = r6*r6;
                       
                       float force_magnitude = 24.0f * epsilon * (2.0f*sigma12/r12 - sigma6/r6) / r2;
                       
                       fx += force_magnitude * dx;
                       fy += force_magnitude * dy;
                       fz += force_magnitude * dz;
                   }
               }
               
               forces[3*i] = fx;
               forces[3*i+1] = fy;
               forces[3*i+2] = fz;
           }
           ''', 'lennard_jones_forces')
       
       def calculate_forces(
           self, 
           positions: np.ndarray, 
           parameters: np.ndarray,
           cutoff: float = 10.0
       ) -> np.ndarray:
           """Calculate forces on GPU."""
           # Transfer data to GPU
           self.positions_gpu[:] = cp.asarray(positions)
           if self.parameters_gpu is None:
               self.parameters_gpu = cp.asarray(parameters)
           
           # Launch kernel
           block_size = 256
           grid_size = (self.n_atoms + block_size - 1) // block_size
           
           self.lennard_jones_kernel(
               (grid_size,), (block_size,),
               (self.positions_gpu.ravel(),
                self.forces_gpu.ravel(),
                self.parameters_gpu,
                self.n_atoms,
                cutoff)
           )
           
           # Transfer result back to CPU
           return cp.asnumpy(self.forces_gpu)

**Multi-GPU Support**

.. code-block:: python

   class MultiGPUSimulation:
       """Multi-GPU molecular dynamics simulation."""
       
       def __init__(self, n_gpus: int = None):
           self.n_gpus = n_gpus or cp.cuda.runtime.getDeviceCount()
           self.gpu_contexts = []
           
           # Initialize GPU contexts
           for gpu_id in range(self.n_gpus):
               with cp.cuda.Device(gpu_id):
                   self.gpu_contexts.append({
                       'device_id': gpu_id,
                       'positions': None,
                       'forces': None,
                       'stream': cp.cuda.Stream()
                   })
       
       def distribute_atoms(self, positions: np.ndarray):
           """Distribute atoms across GPUs."""
           n_atoms = len(positions)
           atoms_per_gpu = n_atoms // self.n_gpus
           
           for i, context in enumerate(self.gpu_contexts):
               start_idx = i * atoms_per_gpu
               end_idx = start_idx + atoms_per_gpu
               if i == self.n_gpus - 1:  # Last GPU gets remaining atoms
                   end_idx = n_atoms
               
               with cp.cuda.Device(context['device_id']):
                   context['positions'] = cp.asarray(
                       positions[start_idx:end_idx]
                   )
                   context['forces'] = cp.zeros_like(context['positions'])
       
       def calculate_forces_parallel(self):
           """Calculate forces in parallel across GPUs."""
           # Launch kernels on all GPUs simultaneously
           for context in self.gpu_contexts:
               with cp.cuda.Device(context['device_id']):
                   with context['stream']:
                       # Launch force calculation kernel
                       self._launch_force_kernel(context)
           
           # Synchronize all streams
           for context in self.gpu_contexts:
               context['stream'].synchronize()

Memory Optimization
~~~~~~~~~~~~~~~~~

**Memory Pool Management**

.. code-block:: python

   class MemoryPool:
       """Memory pool for efficient array allocation."""
       
       def __init__(self, max_size: int = 1024**3):  # 1 GB default
           self.max_size = max_size
           self.pools = {}  # {dtype: {shape: [arrays]}}
           self.allocated_size = 0
       
       def get_array(self, shape: tuple, dtype=np.float64) -> np.ndarray:
           """Get array from pool or allocate new one."""
           key = (tuple(shape), dtype)
           
           if dtype not in self.pools:
               self.pools[dtype] = {}
           if shape not in self.pools[dtype]:
               self.pools[dtype][shape] = []
           
           pool = self.pools[dtype][shape]
           
           if pool:
               return pool.pop()
           else:
               array_size = np.prod(shape) * np.dtype(dtype).itemsize
               if self.allocated_size + array_size > self.max_size:
                   self._cleanup_pools()
               
               array = np.empty(shape, dtype=dtype)
               self.allocated_size += array_size
               return array
       
       def return_array(self, array: np.ndarray) -> None:
           """Return array to pool for reuse."""
           shape = array.shape
           dtype = array.dtype
           
           if dtype in self.pools and shape in self.pools[dtype]:
               # Clear array and return to pool
               array.fill(0)
               self.pools[dtype][shape].append(array)
       
       def _cleanup_pools(self):
           """Remove least recently used arrays."""
           # Simple cleanup - remove half of each pool
           for dtype_pools in self.pools.values():
               for shape_pool in dtype_pools.values():
                   removed = len(shape_pool) // 2
                   for _ in range(removed):
                       if shape_pool:
                           array = shape_pool.pop(0)
                           self.allocated_size -= array.nbytes

**In-Place Operations**

.. code-block:: python

   def update_positions_inplace(
       positions: np.ndarray,
       velocities: np.ndarray,
       dt: float
   ) -> None:
       """Update positions in-place to avoid memory allocation."""
       # Instead of: positions = positions + velocities * dt
       positions += velocities * dt  # In-place operation
   
   def normalize_vectors_inplace(vectors: np.ndarray) -> None:
       """Normalize vectors in-place."""
       norms = np.linalg.norm(vectors, axis=1, keepdims=True)
       # Avoid division by zero
       norms[norms == 0] = 1.0
       vectors /= norms  # In-place division
   
   def apply_periodic_boundary_conditions_inplace(
       positions: np.ndarray,
       box_vectors: np.ndarray
   ) -> None:
       """Apply PBC in-place using modulo operations."""
       # Extract box dimensions (assuming orthogonal box)
       box_lengths = np.diag(box_vectors)
       
       # Apply PBC in-place
       positions %= box_lengths  # Modulo operation in-place

Parallel Processing
~~~~~~~~~~~~~~~~~

**Thread-Based Parallelism**

.. code-block:: python

   import concurrent.futures
   import multiprocessing
   
   class ParallelForceCalculator:
       """Parallel force calculation using thread pools."""
       
       def __init__(self, n_workers: int = None):
           self.n_workers = n_workers or multiprocessing.cpu_count()
           self.executor = concurrent.futures.ThreadPoolExecutor(
               max_workers=self.n_workers
           )
       
       def calculate_forces_parallel(
           self,
           positions: np.ndarray,
           parameters: dict
       ) -> np.ndarray:
           """Calculate forces using parallel workers."""
           n_atoms = len(positions)
           chunk_size = max(1, n_atoms // self.n_workers)
           
           # Divide work among workers
           futures = []
           for i in range(0, n_atoms, chunk_size):
               end_idx = min(i + chunk_size, n_atoms)
               future = self.executor.submit(
                   self._calculate_forces_chunk,
                   positions, parameters, i, end_idx
               )
               futures.append(future)
           
           # Collect results
           all_forces = np.zeros_like(positions)
           for i, future in enumerate(futures):
               start_idx = i * chunk_size
               end_idx = min(start_idx + chunk_size, n_atoms)
               all_forces[start_idx:end_idx] = future.result()
           
           return all_forces
       
       def _calculate_forces_chunk(
           self,
           positions: np.ndarray,
           parameters: dict,
           start_idx: int,
           end_idx: int
       ) -> np.ndarray:
           """Calculate forces for a chunk of atoms."""
           chunk_forces = np.zeros((end_idx - start_idx, 3))
           
           for i in range(start_idx, end_idx):
               force = self._calculate_single_atom_force(
                   i, positions, parameters
               )
               chunk_forces[i - start_idx] = force
           
           return chunk_forces

**Process-Based Parallelism**

.. code-block:: python

   import multiprocessing as mp
   from multiprocessing import shared_memory
   
   class SharedMemorySimulation:
       """Simulation using shared memory for parallel processing."""
       
       def __init__(self, n_atoms: int, n_processes: int = None):
           self.n_atoms = n_atoms
           self.n_processes = n_processes or mp.cpu_count()
           
           # Create shared memory arrays
           self.positions_shm = shared_memory.SharedMemory(
               create=True, size=n_atoms * 3 * 8  # float64
           )
           self.forces_shm = shared_memory.SharedMemory(
               create=True, size=n_atoms * 3 * 8
           )
           
           # Create numpy arrays backed by shared memory
           self.positions = np.ndarray(
               (n_atoms, 3), dtype=np.float64, buffer=self.positions_shm.buf
           )
           self.forces = np.ndarray(
               (n_atoms, 3), dtype=np.float64, buffer=self.forces_shm.buf
           )
       
       def calculate_forces_multiprocess(self):
           """Calculate forces using multiple processes."""
           processes = []
           chunk_size = self.n_atoms // self.n_processes
           
           for i in range(self.n_processes):
               start_idx = i * chunk_size
               end_idx = start_idx + chunk_size
               if i == self.n_processes - 1:
                   end_idx = self.n_atoms
               
               p = mp.Process(
                   target=self._force_calculation_worker,
                   args=(start_idx, end_idx)
               )
               p.start()
               processes.append(p)
           
           # Wait for all processes to complete
           for p in processes:
               p.join()
       
       def _force_calculation_worker(self, start_idx: int, end_idx: int):
           """Worker process for force calculation."""
           # Attach to shared memory
           positions_array = np.ndarray(
               (self.n_atoms, 3), dtype=np.float64, 
               buffer=self.positions_shm.buf
           )
           forces_array = np.ndarray(
               (self.n_atoms, 3), dtype=np.float64,
               buffer=self.forces_shm.buf
           )
           
           # Calculate forces for assigned atoms
           for i in range(start_idx, end_idx):
               force = self._calculate_atom_force(i, positions_array)
               forces_array[i] = force

I/O Optimization
~~~~~~~~~~~~~~~

**Efficient File I/O**

.. code-block:: python

   import h5py
   import zarr
   
   class OptimizedTrajectoryWriter:
       """High-performance trajectory file writer."""
       
       def __init__(self, filename: str, n_atoms: int, compression='lzf'):
           self.filename = filename
           self.n_atoms = n_atoms
           self.compression = compression
           self.buffer_size = 100  # Frames to buffer
           
           # Initialize HDF5 file
           self.file = h5py.File(filename, 'w')
           
           # Create datasets with chunking and compression
           self.positions_ds = self.file.create_dataset(
               'positions',
               shape=(0, n_atoms, 3),
               maxshape=(None, n_atoms, 3),
               chunks=(self.buffer_size, n_atoms, 3),
               compression=compression,
               dtype=np.float32
           )
           
           self.times_ds = self.file.create_dataset(
               'times',
               shape=(0,),
               maxshape=(None,),
               chunks=(self.buffer_size,),
               compression=compression,
               dtype=np.float64
           )
           
           # Buffer for batch writing
           self.position_buffer = []
           self.time_buffer = []
       
       def write_frame(self, positions: np.ndarray, time: float):
           """Write trajectory frame with buffering."""
           self.position_buffer.append(positions.astype(np.float32))
           self.time_buffer.append(time)
           
           if len(self.position_buffer) >= self.buffer_size:
               self._flush_buffer()
       
       def _flush_buffer(self):
           """Flush buffered frames to disk."""
           if not self.position_buffer:
               return
           
           # Current dataset size
           current_size = self.positions_ds.shape[0]
           new_size = current_size + len(self.position_buffer)
           
           # Resize datasets
           self.positions_ds.resize((new_size, self.n_atoms, 3))
           self.times_ds.resize((new_size,))
           
           # Write buffered data
           positions_array = np.array(self.position_buffer)
           times_array = np.array(self.time_buffer)
           
           self.positions_ds[current_size:new_size] = positions_array
           self.times_ds[current_size:new_size] = times_array
           
           # Clear buffers
           self.position_buffer.clear()
           self.time_buffer.clear()
       
       def close(self):
           """Close file and flush remaining data."""
           self._flush_buffer()
           self.file.close()

Performance Testing
------------------

Regression Testing
~~~~~~~~~~~~~~~~~

**Performance Regression Tests**

.. code-block:: python

   import json
   import pytest
   
   class PerformanceRegressionTest:
       """Test for performance regressions."""
       
       def __init__(self, baseline_file: str = "performance_baseline.json"):
           self.baseline_file = baseline_file
           self.baseline_data = self._load_baseline()
       
       def _load_baseline(self) -> dict:
           """Load baseline performance data."""
           try:
               with open(self.baseline_file, 'r') as f:
                   return json.load(f)
           except FileNotFoundError:
               return {}
       
       def test_force_calculation_performance(self):
           """Test force calculation performance."""
           n_atoms = 1000
           positions = np.random.random((n_atoms, 3))
           
           # Benchmark current implementation
           start_time = time.perf_counter()
           forces = calculate_forces(positions)
           end_time = time.perf_counter()
           
           current_time = end_time - start_time
           baseline_time = self.baseline_data.get('force_calculation_1000', 0.1)
           
           # Allow 10% regression
           max_allowed_time = baseline_time * 1.1
           
           assert current_time <= max_allowed_time, (
               f"Performance regression detected: "
               f"current={current_time:.3f}s, "
               f"baseline={baseline_time:.3f}s, "
               f"max_allowed={max_allowed_time:.3f}s"
           )
       
       def update_baseline(self, test_name: str, performance_time: float):
           """Update baseline performance data."""
           self.baseline_data[test_name] = performance_time
           
           with open(self.baseline_file, 'w') as f:
               json.dump(self.baseline_data, f, indent=2)

**Continuous Performance Monitoring**

.. code-block:: python

   class ContinuousPerformanceMonitor:
       """Monitor performance across development."""
       
       def __init__(self, results_file: str = "performance_history.json"):
           self.results_file = results_file
           self.history = self._load_history()
       
       def _load_history(self) -> dict:
           """Load performance history."""
           try:
               with open(self.results_file, 'r') as f:
                   return json.load(f)
           except FileNotFoundError:
               return {'measurements': []}
       
       def record_benchmark(self, commit_hash: str, benchmarks: dict):
           """Record benchmark results for a commit."""
           measurement = {
               'commit': commit_hash,
               'timestamp': time.time(),
               'benchmarks': benchmarks
           }
           
           self.history['measurements'].append(measurement)
           
           # Keep only last 100 measurements
           if len(self.history['measurements']) > 100:
               self.history['measurements'] = self.history['measurements'][-100:]
           
           self._save_history()
       
       def _save_history(self):
           """Save performance history."""
           with open(self.results_file, 'w') as f:
               json.dump(self.history, f, indent=2)
       
       def detect_regressions(self, threshold: float = 0.1) -> List[str]:
           """Detect performance regressions."""
           if len(self.history['measurements']) < 2:
               return []
           
           recent = self.history['measurements'][-1]['benchmarks']
           previous = self.history['measurements'][-2]['benchmarks']
           
           regressions = []
           for test_name, current_time in recent.items():
               if test_name in previous:
                   previous_time = previous[test_name]
                   regression = (current_time - previous_time) / previous_time
                   
                   if regression > threshold:
                       regressions.append(
                           f"{test_name}: {regression*100:.1f}% slower"
                       )
           
           return regressions

Best Practices Summary
---------------------

Development Guidelines
~~~~~~~~~~~~~~~~~~~~~

**Performance-First Development**

1. **Design for Performance**: Consider performance implications during design
2. **Profile Early**: Identify bottlenecks before optimizing
3. **Optimize Algorithms**: Focus on algorithmic improvements first
4. **Use Appropriate Data Structures**: Choose the right tool for the job
5. **Leverage NumPy**: Use vectorized operations whenever possible
6. **Consider Memory Layout**: Structure data for efficient access patterns

**Optimization Workflow**

1. **Measure Current Performance**: Establish baseline measurements
2. **Identify Bottlenecks**: Use profiling tools to find slow code
3. **Choose Optimization Strategy**: Algorithm, vectorization, or parallelization
4. **Implement Changes**: Make targeted optimizations
5. **Verify Correctness**: Ensure optimizations don't break functionality
6. **Measure Improvement**: Quantify performance gains
7. **Document Changes**: Explain optimization rationale and trade-offs

**Common Performance Pitfalls**

- **Premature Optimization**: Optimizing before identifying bottlenecks
- **Micro-Optimizations**: Focusing on minor improvements while ignoring major issues
- **Memory Leaks**: Not properly managing memory in long-running simulations
- **Inefficient Algorithms**: Using O(n²) algorithms when O(n log n) is possible
- **Python Loops**: Using Python loops for intensive numerical computation
- **Frequent Array Allocation**: Creating new arrays instead of reusing existing ones

Hardware Considerations
~~~~~~~~~~~~~~~~~~~~~~

**CPU Optimization**

- **Cache Efficiency**: Organize data for good cache locality
- **SIMD Instructions**: Use operations that can leverage CPU vector units
- **Thread Parallelism**: Utilize multiple CPU cores effectively
- **Memory Bandwidth**: Avoid memory-bound operations when possible

**GPU Optimization**

- **Memory Coalescing**: Ensure efficient GPU memory access patterns
- **Occupancy**: Maximize GPU utilization with appropriate block sizes
- **Memory Hierarchy**: Use shared memory and registers effectively
- **Asynchronous Operations**: Overlap computation and memory transfers

**Memory Optimization**

- **Memory Pools**: Reuse allocated memory to avoid fragmentation
- **Data Types**: Use appropriate precision (float32 vs float64)
- **Memory Layout**: Consider structure-of-arrays vs array-of-structures
- **Garbage Collection**: Minimize Python object creation in hot paths

This performance guide provides the foundation for developing high-performance molecular dynamics simulations in ProteinMD while maintaining code quality and scientific accuracy.
