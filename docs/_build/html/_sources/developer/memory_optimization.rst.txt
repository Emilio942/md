Memory Optimization Guide
========================

This guide provides comprehensive strategies for optimizing memory usage in ProteinMD, covering both CPU and GPU memory management, data structures, and algorithms.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

Memory optimization is crucial for molecular dynamics simulations, which often involve millions of atoms and require efficient data access patterns. This guide covers memory-related optimizations across all levels of the ProteinMD stack.

Key Principles
~~~~~~~~~~~~~

- **Minimize memory footprint:** Use appropriate data types and structures
- **Optimize access patterns:** Ensure cache-friendly memory layouts
- **Reduce allocations:** Reuse memory when possible
- **Profile memory usage:** Identify bottlenecks and leaks
- **Balance speed vs. memory:** Trade-offs between performance and memory usage

Memory Profiling
---------------

Python Memory Profiling
~~~~~~~~~~~~~~~~~~~~~~~

**Memory Profiler Usage:**

.. code-block:: python

   from memory_profiler import profile
   import psutil
   import os
   
   @profile
   def memory_intensive_function():
       """Example function for memory profiling."""
       # Large array allocation
       import numpy as np
       data = np.zeros((10000, 10000), dtype=np.float64)
       
       # Process data
       result = np.sum(data, axis=0)
       return result
   
   def track_memory_usage():
       """Track overall memory usage."""
       process = psutil.Process(os.getpid())
       memory_info = process.memory_info()
       
       print(f"RSS: {memory_info.rss / 1024**2:.1f} MB")
       print(f"VMS: {memory_info.vms / 1024**2:.1f} MB")
       
       # Memory percentage
       memory_percent = process.memory_percent()
       print(f"Memory usage: {memory_percent:.1f}%")

**Line-by-Line Profiling:**

.. code-block:: python

   # Run with: kernprof -l -v script.py
   @profile
   def optimize_memory_access(positions, forces):
       """Profile memory access patterns."""
       n_atoms = positions.shape[0]
       
       # Line-by-line memory tracking
       energies = np.zeros(n_atoms)  # Memory allocation
       
       for i in range(n_atoms):     # Memory access pattern
           pos = positions[i]        # Cache-friendly access
           energies[i] = np.sum(pos * forces[i])
       
       return energies

Advanced Memory Tracking
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import tracemalloc
   import gc
   
   class MemoryTracker:
       """Advanced memory tracking for ProteinMD."""
       
       def __init__(self):
           self.snapshots = []
           
       def start_tracking(self):
           """Start memory tracking."""
           tracemalloc.start()
           
       def take_snapshot(self, name):
           """Take a memory snapshot."""
           snapshot = tracemalloc.take_snapshot()
           self.snapshots.append((name, snapshot))
           
       def compare_snapshots(self, name1, name2):
           """Compare two memory snapshots."""
           snap1 = next(s for n, s in self.snapshots if n == name1)
           snap2 = next(s for n, s in self.snapshots if n == name2)
           
           top_stats = snap2.compare_to(snap1, 'lineno')
           
           print(f"Memory comparison: {name1} -> {name2}")
           for stat in top_stats[:10]:
               print(stat)
               
       def get_memory_leaks(self):
           """Identify potential memory leaks."""
           gc.collect()  # Force garbage collection
           
           if len(self.snapshots) >= 2:
               recent = self.snapshots[-1][1]
               baseline = self.snapshots[0][1]
               
               top_stats = recent.compare_to(baseline, 'lineno')
               leaks = [stat for stat in top_stats if stat.size_diff > 1024*1024]  # > 1MB
               
               return leaks
           return []

Data Structure Optimization
---------------------------

Efficient Array Layouts
~~~~~~~~~~~~~~~~~~~~~~~

**Array of Structures vs Structure of Arrays:**

.. code-block:: python

   import numpy as np
   
   # Array of Structures (AoS) - Less cache-friendly
   class ParticleAoS:
       def __init__(self, n_particles):
           # Each particle is a structure containing position, velocity, force
           dtype = np.dtype([
               ('position', np.float32, 3),
               ('velocity', np.float32, 3),
               ('force', np.float32, 3),
               ('mass', np.float32),
               ('charge', np.float32)
           ])
           self.particles = np.zeros(n_particles, dtype=dtype)
   
   # Structure of Arrays (SoA) - More cache-friendly
   class ParticleSoA:
       def __init__(self, n_particles):
           # Separate arrays for each property
           self.positions = np.zeros((n_particles, 3), dtype=np.float32)
           self.velocities = np.zeros((n_particles, 3), dtype=np.float32)
           self.forces = np.zeros((n_particles, 3), dtype=np.float32)
           self.masses = np.zeros(n_particles, dtype=np.float32)
           self.charges = np.zeros(n_particles, dtype=np.float32)
   
   def benchmark_memory_access():
       """Benchmark memory access patterns."""
       n_particles = 1000000
       
       # AoS access
       aos = ParticleAoS(n_particles)
       positions_aos = aos.particles['position']
       
       # SoA access
       soa = ParticleSoA(n_particles)
       positions_soa = soa.positions
       
       # Benchmark shows SoA is typically faster for vectorized operations
       import time
       
       start = time.time()
       result_aos = np.sum(positions_aos, axis=0)
       aos_time = time.time() - start
       
       start = time.time()
       result_soa = np.sum(positions_soa, axis=0)
       soa_time = time.time() - start
       
       print(f"AoS time: {aos_time:.4f}s")
       print(f"SoA time: {soa_time:.4f}s")

Memory-Efficient Data Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   class OptimizedDataTypes:
       """Demonstrate memory-efficient data type choices."""
       
       @staticmethod
       def compare_precision():
           """Compare memory usage of different precisions."""
           n = 1000000
           
           # Double precision (64-bit)
           data_f64 = np.zeros(n, dtype=np.float64)
           
           # Single precision (32-bit) - 50% memory savings
           data_f32 = np.zeros(n, dtype=np.float32)
           
           # Half precision (16-bit) - 75% memory savings (limited range)
           data_f16 = np.zeros(n, dtype=np.float16)
           
           print(f"Float64 memory: {data_f64.nbytes / 1024**2:.1f} MB")
           print(f"Float32 memory: {data_f32.nbytes / 1024**2:.1f} MB")
           print(f"Float16 memory: {data_f16.nbytes / 1024**2:.1f} MB")
           
       @staticmethod
       def optimize_integer_types():
           """Use appropriate integer types."""
           n_atoms = 100000
           
           # Over-sized integer (wastes memory)
           atom_ids_large = np.arange(n_atoms, dtype=np.int64)
           
           # Right-sized integer
           if n_atoms < 65536:
               atom_ids_small = np.arange(n_atoms, dtype=np.uint16)
           elif n_atoms < 4294967296:
               atom_ids_small = np.arange(n_atoms, dtype=np.uint32)
           else:
               atom_ids_small = np.arange(n_atoms, dtype=np.uint64)
               
           print(f"Large int memory: {atom_ids_large.nbytes / 1024:.1f} KB")
           print(f"Small int memory: {atom_ids_small.nbytes / 1024:.1f} KB")

Memory Pool Management
---------------------

CPU Memory Pools
~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from typing import Dict, List, Tuple
   
   class MemoryPool:
       """Efficient memory pool for array allocations."""
       
       def __init__(self, initial_size_mb=100):
           self.pools: Dict[Tuple[int, np.dtype], List[np.ndarray]] = {}
           self.allocated_arrays = set()
           self.initial_size = initial_size_mb * 1024 * 1024
           
       def allocate(self, shape, dtype=np.float32):
           """Allocate array from pool."""
           key = (np.prod(shape), dtype)
           
           if key in self.pools and self.pools[key]:
               # Reuse existing array
               array = self.pools[key].pop()
               array = array.reshape(shape)
           else:
               # Create new array
               array = np.zeros(shape, dtype=dtype)
               
           self.allocated_arrays.add(id(array))
           return array
           
       def deallocate(self, array):
           """Return array to pool."""
           array_id = id(array)
           if array_id in self.allocated_arrays:
               self.allocated_arrays.remove(array_id)
               
               # Flatten array for reuse
               flat_array = array.flatten()
               key = (flat_array.size, flat_array.dtype)
               
               if key not in self.pools:
                   self.pools[key] = []
               
               self.pools[key].append(flat_array)
               
       def clear(self):
           """Clear all pools."""
           self.pools.clear()
           self.allocated_arrays.clear()
           
       def get_stats(self):
           """Get memory pool statistics."""
           total_pooled = 0
           for arrays in self.pools.values():
               total_pooled += sum(arr.nbytes for arr in arrays)
               
           return {
               'pooled_memory_mb': total_pooled / 1024**2,
               'n_pool_types': len(self.pools),
               'n_allocated': len(self.allocated_arrays)
           }

Smart Pointers for Python
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import weakref
   from typing import Optional, Any
   
   class SmartArray:
       """Smart pointer-like behavior for NumPy arrays."""
       
       def __init__(self, shape, dtype=np.float32, pool: Optional[MemoryPool] = None):
           self.pool = pool
           self.shape = shape
           self.dtype = dtype
           self._array: Optional[np.ndarray] = None
           self._ref_count = 0
           
       def acquire(self):
           """Acquire reference to array."""
           if self._array is None:
               if self.pool:
                   self._array = self.pool.allocate(self.shape, self.dtype)
               else:
                   self._array = np.zeros(self.shape, dtype=self.dtype)
                   
           self._ref_count += 1
           return self._array
           
       def release(self):
           """Release reference to array."""
           self._ref_count -= 1
           if self._ref_count <= 0 and self._array is not None:
               if self.pool:
                   self.pool.deallocate(self._array)
               self._array = None
               
       def __enter__(self):
           return self.acquire()
           
       def __exit__(self, exc_type, exc_val, exc_tb):
           self.release()

Cache Optimization
-----------------

CPU Cache Optimization
~~~~~~~~~~~~~~~~~~~~~

**Cache-Friendly Data Layout:**

.. code-block:: python

   import numpy as np
   
   def optimize_cache_access():
       """Demonstrate cache-friendly access patterns."""
       n = 1000
       data = np.random.random((n, n))
       
       # Cache-friendly: row-major access
       def row_major_sum():
           total = 0.0
           for i in range(n):
               for j in range(n):
                   total += data[i, j]  # Sequential memory access
           return total
           
       # Cache-unfriendly: column-major access
       def column_major_sum():
           total = 0.0
           for j in range(n):
               for i in range(n):
                   total += data[i, j]  # Strided memory access
           return total
           
       # Vectorized (most efficient)
       def vectorized_sum():
           return np.sum(data)
       
       import time
       
       # Benchmark
       start = time.time()
       result1 = row_major_sum()
       row_time = time.time() - start
       
       start = time.time()
       result2 = column_major_sum()
       col_time = time.time() - start
       
       start = time.time()
       result3 = vectorized_sum()
       vec_time = time.time() - start
       
       print(f"Row-major: {row_time:.4f}s")
       print(f"Column-major: {col_time:.4f}s") 
       print(f"Vectorized: {vec_time:.4f}s")

**Blocking for Cache Efficiency:**

.. code-block:: python

   def blocked_matrix_multiply(A, B, block_size=64):
       """Cache-efficient blocked matrix multiplication."""
       n, m, p = A.shape[0], A.shape[1], B.shape[1]
       C = np.zeros((n, p))
       
       for i in range(0, n, block_size):
           for j in range(0, p, block_size):
               for k in range(0, m, block_size):
                   # Define block boundaries
                   i_end = min(i + block_size, n)
                   j_end = min(j + block_size, p)
                   k_end = min(k + block_size, m)
                   
                   # Multiply blocks
                   C[i:i_end, j:j_end] += A[i:i_end, k:k_end] @ B[k:k_end, j:j_end]
                   
       return C

Memory Access Patterns
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   class OptimizedForceCalculation:
       """Memory-optimized force calculation."""
       
       def __init__(self, n_atoms):
           self.n_atoms = n_atoms
           # Use contiguous arrays for better cache performance
           self.positions = np.zeros((n_atoms, 3), dtype=np.float32, order='C')
           self.forces = np.zeros((n_atoms, 3), dtype=np.float32, order='C')
           
       def calculate_forces_optimized(self):
           """Cache-optimized force calculation."""
           # Process atoms in blocks for better cache utilization
           block_size = 256
           
           for block_start in range(0, self.n_atoms, block_size):
               block_end = min(block_start + block_size, self.n_atoms)
               
               # Load block data into cache
               pos_block = self.positions[block_start:block_end]
               force_block = np.zeros_like(pos_block)
               
               # Calculate forces within block
               for i in range(len(pos_block)):
                   for j in range(i + 1, len(pos_block)):
                       r = pos_block[i] - pos_block[j]
                       r_norm = np.linalg.norm(r)
                       
                       if r_norm > 0:
                           force = r / (r_norm**3)  # Simplified force
                           force_block[i] += force
                           force_block[j] -= force
               
               # Write back to main array
               self.forces[block_start:block_end] = force_block

GPU Memory Optimization
----------------------

CUDA Memory Management
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cupy as cp
   import numpy as np
   
   class GPUMemoryOptimizer:
       """Optimize GPU memory usage for ProteinMD."""
       
       def __init__(self):
           self.memory_pool = cp.get_default_memory_pool()
           self.pinned_memory_pool = cp.get_default_pinned_memory_pool()
           
       def optimize_transfers(self, cpu_data):
           """Optimize CPU-GPU transfers using pinned memory."""
           # Use pinned memory for faster transfers
           pinned_array = cp.cuda.alloc_pinned_memory(cpu_data.nbytes)
           pinned_view = np.frombuffer(pinned_array, dtype=cpu_data.dtype)
           pinned_view[:] = cpu_data.flatten()
           
           # Transfer to GPU
           gpu_data = cp.asarray(pinned_view.reshape(cpu_data.shape))
           
           return gpu_data
           
       def minimize_allocations(self, shapes_and_types):
           """Pre-allocate GPU arrays to minimize allocations."""
           arrays = {}
           
           for name, (shape, dtype) in shapes_and_types.items():
               arrays[name] = cp.zeros(shape, dtype=dtype)
               
           return arrays
           
       def use_unified_memory(self, shape, dtype=cp.float32):
           """Use CUDA Unified Memory for automatic management."""
           # Note: This is conceptual - CuPy doesn't directly expose unified memory
           # In practice, use memory pools and careful transfer management
           return cp.zeros(shape, dtype=dtype)
           
       def get_memory_stats(self):
           """Get detailed GPU memory statistics."""
           used_bytes = self.memory_pool.used_bytes()
           total_bytes = self.memory_pool.total_bytes()
           
           free_bytes, device_total = cp.cuda.runtime.memGetInfo()
           
           return {
               'pool_used_mb': used_bytes / 1024**2,
               'pool_total_mb': total_bytes / 1024**2,
               'device_free_mb': free_bytes / 1024**2,
               'device_total_mb': device_total / 1024**2,
               'utilization': (device_total - free_bytes) / device_total
           }

Memory Coalescing
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Good: Coalesced memory access
   def coalesced_kernel_example():
       """Example of memory coalescing optimization."""
       kernel_code = '''
       extern "C" __global__
       void coalesced_add(float* a, float* b, float* c, int n) {
           int idx = blockIdx.x * blockDim.x + threadIdx.x;
           if (idx < n) {
               c[idx] = a[idx] + b[idx];  // Coalesced access
           }
       }
       '''
       
       # Compile and use kernel
       kernel = cp.RawKernel(kernel_code, 'coalesced_add')
       return kernel
   
   # Bad: Non-coalesced memory access
   def non_coalesced_kernel_example():
       """Example of poor memory access pattern."""
       kernel_code = '''
       extern "C" __global__
       void non_coalesced_add(float* a, float* b, float* c, int n, int stride) {
           int idx = blockIdx.x * blockDim.x + threadIdx.x;
           if (idx < n) {
               int real_idx = idx * stride;  // Non-coalesced access
               c[real_idx] = a[real_idx] + b[real_idx];
           }
       }
       '''
       
       kernel = cp.RawKernel(kernel_code, 'non_coalesced_add')
       return kernel

Memory Layout Optimization
--------------------------

Data Alignment
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   def ensure_alignment(array, alignment=32):
       """Ensure array is aligned for SIMD operations."""
       if array.ctypes.data % alignment != 0:
           # Create aligned copy
           aligned_array = np.empty_like(array, order='C')
           aligned_array[:] = array
           return aligned_array
       return array
   
   def create_aligned_array(shape, dtype=np.float32, alignment=32):
       """Create aligned array from scratch."""
       # Calculate total size with padding
       size = np.prod(shape)
       itemsize = np.dtype(dtype).itemsize
       total_bytes = size * itemsize
       
       # Allocate with extra space for alignment
       raw_data = np.empty(total_bytes + alignment, dtype=np.uint8)
       
       # Find aligned offset
       offset = alignment - (raw_data.ctypes.data % alignment)
       if offset == alignment:
           offset = 0
           
       # Create view of aligned data
       aligned_data = raw_data[offset:offset + total_bytes]
       return aligned_data.view(dtype).reshape(shape)

Structure Padding
~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   # Inefficient: poor padding
   particle_dtype_bad = np.dtype([
       ('id', np.int32),      # 4 bytes
       ('mass', np.float64),  # 8 bytes (will be padded to 8-byte boundary)
       ('charge', np.float32), # 4 bytes
       ('position', np.float32, 3), # 12 bytes
   ])  # Total: likely 32 bytes due to padding
   
   # Efficient: manual padding optimization
   particle_dtype_good = np.dtype([
       ('id', np.int32),      # 4 bytes
       ('charge', np.float32), # 4 bytes (fills gap)
       ('mass', np.float64),  # 8 bytes (naturally aligned)
       ('position', np.float32, 3), # 12 bytes
       ('_padding', np.uint8, 4),   # 4 bytes padding for alignment
   ])  # Total: 32 bytes with explicit control

Algorithm-Level Optimizations
----------------------------

Spatial Data Structures
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from typing import List, Tuple
   
   class SpatialGrid:
       """Memory-efficient spatial grid for neighbor finding."""
       
       def __init__(self, box_size, cell_size):
           self.box_size = np.array(box_size)
           self.cell_size = cell_size
           self.n_cells = (self.box_size / cell_size).astype(int)
           
           # Use flat array instead of nested lists for better memory locality
           total_cells = np.prod(self.n_cells)
           self.cell_contents = [[] for _ in range(total_cells)]
           
       def _get_cell_index(self, position):
           """Get flat cell index from position."""
           cell_coords = (position / self.cell_size).astype(int)
           cell_coords = np.clip(cell_coords, 0, self.n_cells - 1)
           
           # Convert 3D coordinates to flat index
           return (cell_coords[0] * self.n_cells[1] * self.n_cells[2] + 
                   cell_coords[1] * self.n_cells[2] + 
                   cell_coords[2])
       
       def update_grid(self, positions):
           """Update grid with new positions."""
           # Clear existing contents
           for cell in self.cell_contents:
               cell.clear()
               
           # Add particles to cells
           for i, pos in enumerate(positions):
               cell_idx = self._get_cell_index(pos)
               self.cell_contents[cell_idx].append(i)
               
       def get_neighbors(self, position, radius):
           """Get neighbors within radius."""
           neighbors = []
           
           # Calculate cell range to check
           cell_radius = int(np.ceil(radius / self.cell_size))
           center_cell = (position / self.cell_size).astype(int)
           
           for dx in range(-cell_radius, cell_radius + 1):
               for dy in range(-cell_radius, cell_radius + 1):
                   for dz in range(-cell_radius, cell_radius + 1):
                       cell_coords = center_cell + np.array([dx, dy, dz])
                       
                       # Check bounds
                       if np.any(cell_coords < 0) or np.any(cell_coords >= self.n_cells):
                           continue
                           
                       cell_idx = (cell_coords[0] * self.n_cells[1] * self.n_cells[2] + 
                                  cell_coords[1] * self.n_cells[2] + 
                                  cell_coords[2])
                       
                       neighbors.extend(self.cell_contents[cell_idx])
                       
           return neighbors

Lazy Evaluation
~~~~~~~~~~~~~~

.. code-block:: python

   class LazyArray:
       """Lazy evaluation for memory-intensive computations."""
       
       def __init__(self, compute_func, *args, **kwargs):
           self.compute_func = compute_func
           self.args = args
           self.kwargs = kwargs
           self._cached_result = None
           self._is_computed = False
           
       def __array__(self):
           """Convert to numpy array when needed."""
           if not self._is_computed:
               self._cached_result = self.compute_func(*self.args, **self.kwargs)
               self._is_computed = True
           return self._cached_result
           
       def invalidate(self):
           """Invalidate cache to save memory."""
           self._cached_result = None
           self._is_computed = False
           
       @property
       def shape(self):
           """Get shape without computing full array."""
           if self._is_computed:
               return self._cached_result.shape
           else:
               # Return shape without full computation if possible
               return self.compute_func(*self.args, compute_shape_only=True, **self.kwargs).shape

Memory-Aware Algorithms
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def memory_efficient_matrix_ops(A, B, chunk_size=1000):
       """Memory-efficient large matrix operations."""
       n, m = A.shape
       result = np.zeros((n, B.shape[1]))
       
       # Process in chunks to control memory usage
       for i in range(0, n, chunk_size):
           end_i = min(i + chunk_size, n)
           chunk_A = A[i:end_i]
           
           # Compute chunk result
           chunk_result = chunk_A @ B
           result[i:end_i] = chunk_result
           
           # Explicitly delete chunk to free memory
           del chunk_A, chunk_result
           
       return result

Performance Monitoring
---------------------

Memory Usage Tracking
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import psutil
   import time
   from contextlib import contextmanager
   
   @contextmanager
   def memory_monitor(name="operation"):
       """Context manager for monitoring memory usage."""
       process = psutil.Process()
       
       # Initial memory
       initial_memory = process.memory_info().rss / 1024**2
       initial_time = time.time()
       
       print(f"Starting {name} - Memory: {initial_memory:.1f} MB")
       
       try:
           yield
       finally:
           # Final memory
           final_memory = process.memory_info().rss / 1024**2
           final_time = time.time()
           
           memory_delta = final_memory - initial_memory
           time_delta = final_time - initial_time
           
           print(f"Completed {name} - Memory: {final_memory:.1f} MB "
                 f"(Δ{memory_delta:+.1f} MB) in {time_delta:.2f}s")

Automated Memory Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class AutoMemoryOptimizer:
       """Automatically optimize memory usage based on monitoring."""
       
       def __init__(self, memory_limit_mb=8000):
           self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
           self.optimizations = []
           
       def check_memory_usage(self):
           """Check current memory usage."""
           process = psutil.Process()
           current_memory = process.memory_info().rss
           return current_memory / 1024**2, current_memory > self.memory_limit
           
       def register_optimization(self, condition_func, optimization_func, name):
           """Register an optimization strategy."""
           self.optimizations.append({
               'condition': condition_func,
               'optimization': optimization_func,
               'name': name
           })
           
       def optimize_if_needed(self):
           """Apply optimizations if memory usage is high."""
           memory_mb, needs_optimization = self.check_memory_usage()
           
           if needs_optimization:
               print(f"Memory usage high ({memory_mb:.1f} MB), applying optimizations...")
               
               for opt in self.optimizations:
                   if opt['condition']():
                       print(f"Applying optimization: {opt['name']}")
                       opt['optimization']()
                       
                       # Check if optimization helped
                       new_memory_mb, still_high = self.check_memory_usage()
                       if not still_high:
                           print(f"Optimization successful: {new_memory_mb:.1f} MB")
                           break
                           
   # Example usage
   def setup_auto_optimizer():
       """Set up automatic memory optimization."""
       optimizer = AutoMemoryOptimizer(memory_limit_mb=6000)
       
       # Register optimizations
       optimizer.register_optimization(
           condition_func=lambda: len(cache) > 100,
           optimization_func=lambda: cache.clear(),
           name="Clear computation cache"
       )
       
       optimizer.register_optimization(
           condition_func=lambda: hasattr(memory_pool, 'used_bytes'),
           optimization_func=lambda: memory_pool.free_all_blocks(),
           name="Clear memory pool"
       )
       
       return optimizer

Best Practices Summary
---------------------

General Guidelines
~~~~~~~~~~~~~~~~~

1. **Profile First:** Always profile before optimizing
2. **Use Appropriate Data Types:** Don't over-allocate precision
3. **Minimize Allocations:** Reuse memory when possible
4. **Optimize Access Patterns:** Consider cache locality
5. **Monitor Memory Usage:** Track memory consumption in production

Memory Optimization Checklist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Data Structures:**
- [ ] Use Structure of Arrays (SoA) layout
- [ ] Choose appropriate data types
- [ ] Align data for SIMD operations
- [ ] Consider memory pooling

**Algorithms:**
- [ ] Implement cache-friendly access patterns
- [ ] Use spatial data structures for neighbor searches
- [ ] Consider lazy evaluation for expensive computations
- [ ] Implement chunking for large operations

**GPU Memory:**
- [ ] Minimize CPU-GPU transfers
- [ ] Use memory coalescing
- [ ] Implement proper memory pools
- [ ] Monitor GPU memory usage

**Monitoring:**
- [ ] Profile memory usage regularly
- [ ] Track memory leaks
- [ ] Monitor cache hit rates
- [ ] Benchmark memory access patterns

Tools and Resources
------------------

**Profiling Tools:**
- ``memory_profiler``: Line-by-line memory profiling
- ``tracemalloc``: Built-in memory tracking
- ``pympler``: Advanced memory analysis
- ``objgraph``: Object reference tracking

**GPU Tools:**
- ``nvidia-smi``: GPU memory monitoring
- ``nvprof``: CUDA profiling
- ``compute-sanitizer``: Memory error detection

**System Tools:**
- ``htop/top``: System memory monitoring
- ``valgrind``: Memory error detection (C/C++)
- ``perf``: Linux performance analysis

By following this comprehensive memory optimization guide, you can significantly improve the memory efficiency of ProteinMD simulations, enabling larger systems and better performance.
