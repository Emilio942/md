CUDA Development Guide
=====================

This guide covers CUDA development practices for ProteinMD, including GPU kernel optimization, memory management, and integration with the main codebase.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

ProteinMD leverages CUDA for high-performance molecular dynamics simulations. This guide provides comprehensive information for developing and optimizing CUDA code within the ProteinMD framework.

CUDA Architecture in ProteinMD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Design Principles:**

- **Hybrid CPU/GPU approach:** Critical path computations on GPU, coordination on CPU
- **Memory efficiency:** Minimize host-device transfers
- **Kernel specialization:** Separate kernels for different force types
- **Scalability:** Support for multi-GPU systems

**Key Components:**

- Force calculation kernels
- Integration kernels
- Energy computation kernels
- Neighbor list construction
- Boundary condition handling

Development Environment Setup
----------------------------

Prerequisites
~~~~~~~~~~~~

.. code-block:: bash

   # CUDA Toolkit (11.0 or later)
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda

Development Tools
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install CuPy for Python-CUDA integration
   pip install cupy-cuda11x
   
   # Install additional tools
   pip install pycuda
   pip install numba[cuda]
   
   # Profiling tools
   pip install py-nvml-monitor

Project Structure
~~~~~~~~~~~~~~~~

.. code-block:: text

   proteinmd/
   ├── cuda/
   │   ├── __init__.py
   │   ├── kernels/
   │   │   ├── forces.cu
   │   │   ├── integration.cu
   │   │   ├── energy.cu
   │   │   └── neighbor_list.cu
   │   ├── device/
   │   │   ├── memory_pool.py
   │   │   ├── context_manager.py
   │   │   └── device_info.py
   │   ├── utils/
   │   │   ├── kernel_launcher.py
   │   │   ├── error_handling.py
   │   │   └── profiling.py
   │   └── tests/
   │       ├── test_kernels.py
   │       └── test_performance.py

CUDA Kernel Development
----------------------

Force Calculation Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~

**Lennard-Jones Forces:**

.. code-block:: cuda

   __global__ void lennard_jones_forces(
       const float* __restrict__ positions,
       const float* __restrict__ charges,
       const int* __restrict__ neighbor_list,
       const int* __restrict__ neighbor_count,
       float* __restrict__ forces,
       float* __restrict__ energy,
       const int n_atoms,
       const float cutoff_sq,
       const float epsilon,
       const float sigma
   ) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       
       if (idx >= n_atoms) return;
       
       float fx = 0.0f, fy = 0.0f, fz = 0.0f;
       float pe = 0.0f;
       
       float xi = positions[3 * idx];
       float yi = positions[3 * idx + 1];
       float zi = positions[3 * idx + 2];
       
       int start = idx * MAX_NEIGHBORS;
       int count = neighbor_count[idx];
       
       for (int i = 0; i < count; i++) {
           int j = neighbor_list[start + i];
           
           float dx = xi - positions[3 * j];
           float dy = yi - positions[3 * j + 1];
           float dz = zi - positions[3 * j + 2];
           
           float r2 = dx * dx + dy * dy + dz * dz;
           
           if (r2 < cutoff_sq && r2 > 0.0f) {
               float inv_r2 = 1.0f / r2;
               float inv_r6 = inv_r2 * inv_r2 * inv_r2;
               float inv_r12 = inv_r6 * inv_r6;
               
               // Lennard-Jones potential
               float sigma_r6 = sigma * sigma * sigma * sigma * sigma * sigma * inv_r6;
               float sigma_r12 = sigma_r6 * sigma_r6;
               
               pe += 4.0f * epsilon * (sigma_r12 - sigma_r6);
               
               // Force calculation
               float force_magnitude = 24.0f * epsilon * inv_r2 * (2.0f * sigma_r12 - sigma_r6);
               
               fx += force_magnitude * dx;
               fy += force_magnitude * dy;
               fz += force_magnitude * dz;
           }
       }
       
       forces[3 * idx] = fx;
       forces[3 * idx + 1] = fy;
       forces[3 * idx + 2] = fz;
       
       energy[idx] = pe * 0.5f; // Avoid double counting
   }

**Bonded Forces:**

.. code-block:: cuda

   __global__ void bond_forces(
       const float* __restrict__ positions,
       const int2* __restrict__ bonds,
       const float* __restrict__ bond_params, // k, r0
       float* __restrict__ forces,
       float* __restrict__ energy,
       const int n_bonds
   ) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       
       if (idx >= n_bonds) return;
       
       int2 bond = bonds[idx];
       int i = bond.x;
       int j = bond.y;
       
       float k = bond_params[2 * idx];
       float r0 = bond_params[2 * idx + 1];
       
       float dx = positions[3 * i] - positions[3 * j];
       float dy = positions[3 * i + 1] - positions[3 * j + 1];
       float dz = positions[3 * i + 2] - positions[3 * j + 2];
       
       float r = sqrtf(dx * dx + dy * dy + dz * dz);
       float dr = r - r0;
       
       // Bond energy: 0.5 * k * (r - r0)^2
       energy[idx] = 0.5f * k * dr * dr;
       
       // Force magnitude: -k * (r - r0) / r
       float force_mag = -k * dr / r;
       
       float fx = force_mag * dx;
       float fy = force_mag * dy;
       float fz = force_mag * dz;
       
       // Apply forces atomically
       atomicAdd(&forces[3 * i], fx);
       atomicAdd(&forces[3 * i + 1], fy);
       atomicAdd(&forces[3 * i + 2], fz);
       
       atomicAdd(&forces[3 * j], -fx);
       atomicAdd(&forces[3 * j + 1], -fy);
       atomicAdd(&forces[3 * j + 2], -fz);
   }

Integration Kernels
~~~~~~~~~~~~~~~~~~

**Velocity Verlet Integration:**

.. code-block:: cuda

   __global__ void velocity_verlet_step1(
       float* __restrict__ positions,
       float* __restrict__ velocities,
       const float* __restrict__ forces,
       const float* __restrict__ masses,
       const float dt,
       const int n_atoms
   ) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       
       if (idx >= n_atoms) return;
       
       float inv_mass = 1.0f / masses[idx];
       float dt_half = 0.5f * dt;
       
       // Update positions: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
       float ax = forces[3 * idx] * inv_mass;
       float ay = forces[3 * idx + 1] * inv_mass;
       float az = forces[3 * idx + 2] * inv_mass;
       
       positions[3 * idx] += velocities[3 * idx] * dt + ax * dt_half * dt;
       positions[3 * idx + 1] += velocities[3 * idx + 1] * dt + ay * dt_half * dt;
       positions[3 * idx + 2] += velocities[3 * idx + 2] * dt + az * dt_half * dt;
       
       // Update velocities: v(t+dt/2) = v(t) + 0.5*a(t)*dt
       velocities[3 * idx] += ax * dt_half;
       velocities[3 * idx + 1] += ay * dt_half;
       velocities[3 * idx + 2] += az * dt_half;
   }
   
   __global__ void velocity_verlet_step2(
       float* __restrict__ velocities,
       const float* __restrict__ forces,
       const float* __restrict__ masses,
       const float dt,
       const int n_atoms
   ) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       
       if (idx >= n_atoms) return;
       
       float inv_mass = 1.0f / masses[idx];
       float dt_half = 0.5f * dt;
       
       // Complete velocity update: v(t+dt) = v(t+dt/2) + 0.5*a(t+dt)*dt
       velocities[3 * idx] += forces[3 * idx] * inv_mass * dt_half;
       velocities[3 * idx + 1] += forces[3 * idx + 1] * inv_mass * dt_half;
       velocities[3 * idx + 2] += forces[3 * idx + 2] * inv_mass * dt_half;
   }

Memory Management
----------------

Device Memory Pool
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cupy as cp
   
   class CudaMemoryPool:
       """Efficient CUDA memory management for ProteinMD."""
       
       def __init__(self, initial_size_mb=1024):
           self.mempool = cp.get_default_memory_pool()
           self.initial_size = initial_size_mb * 1024 * 1024
           self._allocate_initial_pool()
           
       def _allocate_initial_pool(self):
           """Pre-allocate memory pool."""
           # Allocate and immediately free to establish pool
           temp = cp.zeros(self.initial_size // 4, dtype=cp.float32)
           del temp
           
       def allocate(self, shape, dtype=cp.float32):
           """Allocate device memory."""
           return cp.zeros(shape, dtype=dtype)
           
       def get_memory_info(self):
           """Get current memory usage."""
           return {
               'used_bytes': self.mempool.used_bytes(),
               'total_bytes': self.mempool.total_bytes(),
               'n_free_blocks': self.mempool.n_free_blocks()
           }
           
       def clear_cache(self):
           """Clear memory cache."""
           self.mempool.free_all_blocks()

Unified Memory Management
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cupy as cp
   import numpy as np
   
   class UnifiedArray:
       """Wrapper for unified memory arrays."""
       
       def __init__(self, shape, dtype=np.float32):
           self.shape = shape
           self.dtype = dtype
           self._host_data = None
           self._device_data = None
           self._location = 'cpu'  # 'cpu' or 'gpu'
           
       def to_gpu(self):
           """Transfer data to GPU."""
           if self._location == 'cpu' and self._host_data is not None:
               self._device_data = cp.asarray(self._host_data)
               self._location = 'gpu'
           elif self._device_data is None:
               self._device_data = cp.zeros(self.shape, dtype=self.dtype)
               self._location = 'gpu'
           return self._device_data
           
       def to_cpu(self):
           """Transfer data to CPU."""
           if self._location == 'gpu' and self._device_data is not None:
               self._host_data = cp.asnumpy(self._device_data)
               self._location = 'cpu'
           elif self._host_data is None:
               self._host_data = np.zeros(self.shape, dtype=self.dtype)
               self._location = 'cpu'
           return self._host_data
           
       @property
       def data(self):
           """Get data in current location."""
           if self._location == 'gpu':
               return self._device_data
           else:
               return self._host_data

Kernel Launch Optimization
--------------------------

Block and Grid Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def calculate_launch_config(n_elements, max_threads_per_block=1024):
       """Calculate optimal CUDA launch configuration."""
       if n_elements <= max_threads_per_block:
           return (1, n_elements)
       
       # Calculate grid and block dimensions
       threads_per_block = max_threads_per_block
       blocks_per_grid = (n_elements + threads_per_block - 1) // threads_per_block
       
       return (blocks_per_grid, threads_per_block)

   class KernelLauncher:
       """Manages CUDA kernel launches with optimized configurations."""
       
       def __init__(self, device_id=0):
           self.device = cp.cuda.Device(device_id)
           self.stream = cp.cuda.Stream()
           
       def launch_force_kernel(self, kernel, positions, forces, n_atoms, **kwargs):
           """Launch force calculation kernel with optimal configuration."""
           blocks, threads = calculate_launch_config(n_atoms)
           
           with self.device:
               kernel(
                   (blocks,), (threads,),
                   (positions, forces, n_atoms, **kwargs),
                   stream=self.stream
               )
               
       def synchronize(self):
           """Synchronize stream."""
           self.stream.synchronize()

Performance Optimization
-----------------------

Memory Coalescing
~~~~~~~~~~~~~~~~

.. code-block:: cuda

   // Good: Coalesced memory access
   __global__ void coalesced_copy(float* input, float* output, int n) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < n) {
           output[idx] = input[idx];  // Sequential access pattern
       }
   }
   
   // Bad: Non-coalesced memory access
   __global__ void non_coalesced_copy(float* input, float* output, int n, int stride) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < n) {
           output[idx] = input[idx * stride];  // Strided access pattern
       }
   }

Shared Memory Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: cuda

   __global__ void force_reduction_shared(
       const float* forces,
       float* total_force,
       const int n_atoms
   ) {
       extern __shared__ float shared_forces[];
       
       int tid = threadIdx.x;
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       
       // Load data into shared memory
       shared_forces[tid] = (idx < n_atoms) ? forces[idx] : 0.0f;
       __syncthreads();
       
       // Reduction in shared memory
       for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
           if (tid < stride) {
               shared_forces[tid] += shared_forces[tid + stride];
           }
           __syncthreads();
       }
       
       // Write result
       if (tid == 0) {
           atomicAdd(total_force, shared_forces[0]);
       }
   }

Warp-Level Optimizations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cuda

   __device__ float warp_reduce_sum(float val) {
       for (int offset = warpSize/2; offset > 0; offset /= 2) {
           val += __shfl_down_sync(0xFFFFFFFF, val, offset);
       }
       return val;
   }
   
   __global__ void fast_energy_reduction(
       const float* energies,
       float* total_energy,
       const int n_atoms
   ) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       
       float energy = (idx < n_atoms) ? energies[idx] : 0.0f;
       
       // Warp-level reduction
       energy = warp_reduce_sum(energy);
       
       // Block-level reduction using shared memory
       __shared__ float warp_sums[32];
       int warp_id = threadIdx.x / warpSize;
       int lane_id = threadIdx.x % warpSize;
       
       if (lane_id == 0) {
           warp_sums[warp_id] = energy;
       }
       __syncthreads();
       
       // Final reduction
       if (warp_id == 0) {
           energy = (lane_id < blockDim.x / warpSize) ? warp_sums[lane_id] : 0.0f;
           energy = warp_reduce_sum(energy);
           
           if (lane_id == 0) {
               atomicAdd(total_energy, energy);
           }
       }
   }

Multi-GPU Support
----------------

Device Management
~~~~~~~~~~~~~~~~

.. code-block:: python

   import cupy as cp
   
   class MultiGPUManager:
       """Manages multiple GPU devices for ProteinMD."""
       
       def __init__(self):
           self.n_devices = cp.cuda.runtime.getDeviceCount()
           self.devices = [cp.cuda.Device(i) for i in range(self.n_devices)]
           self.streams = [cp.cuda.Stream() for _ in range(self.n_devices)]
           
       def distribute_atoms(self, n_atoms):
           """Distribute atoms across GPUs."""
           atoms_per_gpu = n_atoms // self.n_devices
           distribution = []
           
           for i in range(self.n_devices):
               start = i * atoms_per_gpu
               if i == self.n_devices - 1:
                   end = n_atoms  # Last GPU takes remaining atoms
               else:
                   end = (i + 1) * atoms_per_gpu
               distribution.append((start, end))
               
           return distribution
           
       def synchronize_all(self):
           """Synchronize all GPU streams."""
           for stream in self.streams:
               stream.synchronize()

Domain Decomposition
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class DomainDecomposition:
       """Implements spatial domain decomposition for multi-GPU."""
       
       def __init__(self, box_size, n_gpus):
           self.box_size = box_size
           self.n_gpus = n_gpus
           self.domains = self._create_domains()
           
       def _create_domains(self):
           """Create spatial domains for each GPU."""
           # Simple 1D decomposition along z-axis
           z_size = self.box_size[2] / self.n_gpus
           domains = []
           
           for i in range(self.n_gpus):
               z_min = i * z_size
               z_max = (i + 1) * z_size
               domains.append({
                   'x_range': (0, self.box_size[0]),
                   'y_range': (0, self.box_size[1]),
                   'z_range': (z_min, z_max),
                   'device_id': i
               })
               
           return domains
           
       def assign_atoms(self, positions):
           """Assign atoms to domains based on position."""
           assignments = []
           
           for pos in positions:
               for i, domain in enumerate(self.domains):
                   if (domain['z_range'][0] <= pos[2] < domain['z_range'][1]):
                       assignments.append(i)
                       break
                       
           return assignments

Error Handling and Debugging
----------------------------

CUDA Error Checking
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cupy as cp
   
   class CudaErrorHandler:
       """Comprehensive CUDA error handling."""
       
       @staticmethod
       def check_cuda_error(operation_name="CUDA operation"):
           """Check for CUDA errors and provide detailed information."""
           try:
               cp.cuda.runtime.deviceSynchronize()
           except cp.cuda.runtime.CUDARuntimeError as e:
               error_info = {
                   'operation': operation_name,
                   'error_code': e.code,
                   'error_message': str(e),
                   'device_id': cp.cuda.Device().id
               }
               
               # Get device properties
               device = cp.cuda.Device()
               error_info['device_name'] = device.attributes['Name']
               error_info['compute_capability'] = device.compute_capability
               
               # Get memory information
               free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
               error_info['free_memory_mb'] = free_bytes / 1024**2
               error_info['total_memory_mb'] = total_bytes / 1024**2
               
               raise RuntimeError(f"CUDA error in {operation_name}: {error_info}")

Memory Debugging
~~~~~~~~~~~~~~~

.. code-block:: python

   def debug_gpu_memory_usage():
       """Debug GPU memory usage patterns."""
       mempool = cp.get_default_memory_pool()
       
       print("GPU Memory Debug Information:")
       print(f"Used bytes: {mempool.used_bytes() / 1024**2:.1f} MB")
       print(f"Total bytes: {mempool.total_bytes() / 1024**2:.1f} MB")
       print(f"Free blocks: {mempool.n_free_blocks()}")
       
       # Device memory info
       free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
       used_bytes = total_bytes - free_bytes
       print(f"Device memory used: {used_bytes / 1024**2:.1f} MB")
       print(f"Device memory free: {free_bytes / 1024**2:.1f} MB")
       print(f"Device memory total: {total_bytes / 1024**2:.1f} MB")

Testing and Validation
----------------------

Kernel Unit Tests
~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   import cupy as cp
   import numpy as np
   
   class TestCudaKernels:
       """Unit tests for CUDA kernels."""
       
       def setup_method(self):
           """Set up test environment."""
           self.device = cp.cuda.Device(0)
           self.stream = cp.cuda.Stream()
           
       def test_lennard_jones_forces(self):
           """Test Lennard-Jones force calculation."""
           # Create test data
           n_atoms = 100
           positions = cp.random.random((n_atoms, 3), dtype=cp.float32)
           forces = cp.zeros((n_atoms, 3), dtype=cp.float32)
           
           # Launch kernel
           with self.device:
               lennard_jones_kernel = cp.RawKernel(
                   lennard_jones_code, 'lennard_jones_forces'
               )
               
               blocks, threads = calculate_launch_config(n_atoms)
               lennard_jones_kernel(
                   (blocks,), (threads,),
                   (positions, forces, n_atoms, 2.5**2, 1.0, 1.0),
                   stream=self.stream
               )
               
           self.stream.synchronize()
           
           # Validate results
           forces_cpu = cp.asnumpy(forces)
           assert not np.any(np.isnan(forces_cpu))
           assert not np.any(np.isinf(forces_cpu))
           
       def test_force_conservation(self):
           """Test Newton's third law in force calculations."""
           # Implementation of conservation test
           pass

Performance Benchmarks
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CudaPerformanceBenchmark:
       """Benchmark CUDA kernel performance."""
       
       def __init__(self):
           self.device = cp.cuda.Device(0)
           
       def benchmark_kernel(self, kernel_func, *args, n_runs=10):
           """Benchmark kernel execution time."""
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

Best Practices
--------------

Development Guidelines
~~~~~~~~~~~~~~~~~~~~~

1. **Memory Management**
   - Use memory pools for frequent allocations
   - Minimize host-device transfers
   - Prefer unified memory for development

2. **Kernel Design**
   - Keep kernels simple and focused
   - Optimize for memory coalescing
   - Use shared memory effectively

3. **Error Handling**
   - Always check CUDA errors
   - Provide informative error messages
   - Use defensive programming

4. **Testing**
   - Unit test individual kernels
   - Validate against CPU implementations
   - Performance regression testing

Common Pitfalls
~~~~~~~~~~~~~~

1. **Memory Issues**
   - Memory leaks from improper cleanup
   - Out-of-bounds memory access
   - Unaligned memory access

2. **Synchronization Problems**
   - Race conditions in shared memory
   - Missing __syncthreads() calls
   - Incorrect atomic operations

3. **Performance Issues**
   - Poor memory access patterns
   - Inefficient thread divergence
   - Suboptimal block sizes

Integration with ProteinMD
-------------------------

Python-CUDA Interface
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CudaAccelerator:
       """Main interface for CUDA acceleration in ProteinMD."""
       
       def __init__(self, device_id=0):
           self.device = cp.cuda.Device(device_id)
           self.memory_pool = CudaMemoryPool()
           self.kernel_launcher = KernelLauncher(device_id)
           
       def calculate_forces(self, simulation_state):
           """Calculate forces using CUDA acceleration."""
           positions = self.memory_pool.allocate(simulation_state.positions.shape)
           forces = self.memory_pool.allocate(simulation_state.positions.shape)
           
           # Transfer data to GPU
           positions[:] = simulation_state.positions
           
           # Launch force kernels
           self.kernel_launcher.launch_force_kernel(
               lennard_jones_kernel, positions, forces,
               simulation_state.n_atoms
           )
           
           # Transfer results back
           simulation_state.forces[:] = cp.asnumpy(forces)

Build System Integration
~~~~~~~~~~~~~~~~~~~~~~

Add to `setup.py`:

.. code-block:: python

   from setuptools import setup, Extension
   from pybind11.setup_helpers import Pybind11Extension
   import pybind11
   
   cuda_extension = Pybind11Extension(
       "proteinmd._cuda",
       ["src/cuda/python_bindings.cpp"],
       include_dirs=[
           pybind11.get_cmake_dir(),
           "/usr/local/cuda/include",
       ],
       libraries=["cudart", "cublas"],
       library_dirs=["/usr/local/cuda/lib64"],
       language="c++",
   )
   
   setup(
       ext_modules=[cuda_extension],
       # ... other setup options
   )

This comprehensive CUDA development guide provides the foundation for high-performance GPU acceleration in ProteinMD. Follow these practices to develop efficient, maintainable CUDA code that integrates seamlessly with the main codebase.
