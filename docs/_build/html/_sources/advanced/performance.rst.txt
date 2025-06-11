Performance Optimization
======================

This guide covers performance optimization techniques for ProteinMD simulations.

.. contents:: Performance Topics
   :local:
   :depth: 2

Hardware Considerations
----------------------

CPU Optimization
~~~~~~~~~~~~~~~

**Multi-threading Configuration**

.. code-block:: python

   from proteinmd.core import PerformanceSettings
   
   # Configure threading
   perf_settings = PerformanceSettings()
   perf_settings.set_num_threads(8)  # Use 8 CPU cores
   perf_settings.enable_thread_affinity()  # Pin threads to cores
   
   # Apply to simulation
   simulation.apply_performance_settings(perf_settings)

**SIMD Optimizations**

.. code-block:: python

   # Enable SIMD instructions (AVX2, AVX-512)
   perf_settings.enable_simd()
   perf_settings.set_simd_level("avx2")  # or "avx512" if available
   
   # Verify SIMD support
   if perf_settings.check_simd_support("avx2"):
       print("AVX2 instructions available")

GPU Acceleration
~~~~~~~~~~~~~~~

**CUDA Setup**

.. code-block:: python

   from proteinmd.core import CudaSettings
   
   # Check GPU availability
   cuda_settings = CudaSettings()
   if cuda_settings.is_available():
       print(f"Found {cuda_settings.get_device_count()} CUDA devices")
       
       # Select GPU
       cuda_settings.set_device(0)  # Use first GPU
       
       # Configure GPU settings
       cuda_settings.set_precision("mixed")  # mixed precision for speed
       cuda_settings.enable_tensor_cores()   # for modern GPUs
       
       # Apply to simulation
       simulation.use_gpu(cuda_settings)
   else:
       print("CUDA not available, using CPU")

**Multiple GPU Usage**

.. code-block:: python

   # For very large systems or replica exchange
   multi_gpu = CudaSettings()
   multi_gpu.enable_multi_gpu([0, 1, 2, 3])  # Use 4 GPUs
   multi_gpu.set_domain_decomposition("particle")  # or "spatial"
   
   simulation.use_multi_gpu(multi_gpu)

Memory Optimization
~~~~~~~~~~~~~~~~~~

**Memory Pool Management**

.. code-block:: python

   from proteinmd.utils import MemoryManager
   
   # Configure memory pools
   memory_manager = MemoryManager()
   memory_manager.set_pool_size_gb(4.0)  # 4 GB memory pool
   memory_manager.enable_memory_prefetching()
   
   # Apply to system
   system.set_memory_manager(memory_manager)

Algorithm Optimization
---------------------

Force Calculation Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Neighbor Lists**

.. code-block:: python

   from proteinmd.core import NeighborList
   
   # Optimize neighbor list settings
   neighbor_list = NeighborList()
   neighbor_list.set_cutoff(1.2)      # 1.2 nm cutoff
   neighbor_list.set_buffer(0.2)       # 0.2 nm buffer
   neighbor_list.set_update_frequency(10)  # Update every 10 steps
   
   # Use Verlet lists for better performance
   neighbor_list.set_algorithm("verlet")
   
   system.set_neighbor_list(neighbor_list)

**PME Optimization**

.. code-block:: python

   from proteinmd.core import PMESettings
   
   # Optimize PME parameters
   pme = PMESettings()
   pme.set_grid_spacing(0.12)  # 0.12 nm grid spacing
   pme.set_alpha(0.31)         # Ewald parameter
   pme.enable_pme_gpu()        # Use GPU for PME if available
   
   # Auto-tune PME parameters
   pme.auto_tune(system)
   
   system.set_pme_settings(pme)

**Constraint Algorithms**

.. code-block:: python

   from proteinmd.core import ConstraintSettings
   
   # Use LINCS for bond constraints
   constraints = ConstraintSettings()
   constraints.set_algorithm("lincs")
   constraints.set_lincs_order(4)      # LINCS expansion order
   constraints.set_lincs_iterations(1) # Usually 1 is sufficient
   
   # Apply constraints to hydrogen bonds only
   constraints.constrain_bonds("h-bonds")
   
   system.set_constraints(constraints)

Integration Optimization
~~~~~~~~~~~~~~~~~~~~~~~

**Timestep Selection**

.. code-block:: python

   # Choose optimal timestep based on system
   if system.has_constraints("h-bonds"):
       timestep = 0.002  # 2 fs with H-bond constraints
   else:
       timestep = 0.001  # 1 fs without constraints
   
   integrator = VelocityVerletIntegrator(timestep=timestep)

**Multiple Time-Stepping**

.. code-block:: python

   from proteinmd.core import MultipleTimestepIntegrator
   
   # Use different timesteps for different interactions
   mts_integrator = MultipleTimestepIntegrator()
   mts_integrator.set_short_timestep(0.001)  # 1 fs for bonds/angles
   mts_integrator.set_medium_timestep(0.002) # 2 fs for short-range NB
   mts_integrator.set_long_timestep(0.004)   # 4 fs for long-range
   
   simulation.set_integrator(mts_integrator)

System-Specific Optimizations
----------------------------

Large System Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

**Domain Decomposition**

.. code-block:: python

   from proteinmd.core import DomainDecomposition
   
   # For systems > 100k atoms
   if system.get_atom_count() > 100000:
       domain_decomp = DomainDecomposition()
       domain_decomp.set_method("spatial")
       domain_decomp.set_cutoff_scheme("group")
       
       system.enable_domain_decomposition(domain_decomp)

**Load Balancing**

.. code-block:: python

   # Dynamic load balancing
   load_balancer = simulation.get_load_balancer()
   load_balancer.enable_dynamic_balancing()
   load_balancer.set_balance_frequency(1000)  # Rebalance every 1000 steps

Water Model Optimization
~~~~~~~~~~~~~~~~~~~~~~~

**Rigid Water Models**

.. code-block:: python

   # Use rigid water models for better performance
   water_model = WaterModel("TIP3P")
   water_model.set_rigid(True)  # Enables SETTLE algorithm
   
   # For even better performance with large systems
   water_model = WaterModel("TIP4P-Ew")  # Often faster than TIP3P

**Water Optimization Settings**

.. code-block:: python

   # Optimize water-specific interactions
   system.set_water_optimization(True)
   system.enable_water_constraint_algorithms()
   system.set_water_cutoff_scheme("group")

I/O Optimization
~~~~~~~~~~~~~~~

**Trajectory Output**

.. code-block:: python

   from proteinmd.io import TrajectoryWriter
   
   # Optimize trajectory writing
   writer = TrajectoryWriter()
   writer.set_compression("gzip")     # Compress trajectory files
   writer.set_precision("single")     # Use single precision
   writer.enable_buffering(size_mb=64) # 64 MB buffer
   
   simulation.set_trajectory_writer(writer)

**Parallel I/O**

.. code-block:: python

   # For large-scale simulations
   parallel_io = simulation.get_parallel_io()
   parallel_io.enable_collective_io()
   parallel_io.set_stripe_count(4)  # For Lustre filesystems

Performance Monitoring
---------------------

Real-time Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinmd.utils import PerformanceMonitor
   
   # Set up performance monitoring
   monitor = PerformanceMonitor()
   monitor.enable_timing()
   monitor.enable_memory_tracking()
   monitor.set_report_frequency(10000)  # Report every 10k steps
   
   simulation.add_monitor(monitor)
   
   # Run simulation with monitoring
   simulation.run(steps=1000000)
   
   # Get performance report
   report = monitor.get_report()
   print(f"Average performance: {report['ns_per_day']:.1f} ns/day")
   print(f"Memory usage: {report['peak_memory_gb']:.1f} GB")

Performance Profiling
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinmd.utils import Profiler
   
   # Profile simulation performance
   profiler = Profiler()
   profiler.enable_detailed_timing()
   
   with profiler:
       simulation.run(steps=10000)  # Short run for profiling
   
   # Analyze results
   timing_data = profiler.get_timing_data()
   print("Time breakdown:")
   for component, time_ms in timing_data.items():
       percentage = (time_ms / sum(timing_data.values())) * 100
       print(f"  {component}: {time_ms:.1f} ms ({percentage:.1f}%)")

Benchmarking Tools
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinmd.utils import Benchmark
   
   # Run standardized benchmarks
   benchmark = Benchmark()
   
   # Test different system sizes
   for n_atoms in [10000, 50000, 100000]:
       system = benchmark.create_test_system(n_atoms=n_atoms)
       
       # Run benchmark
       result = benchmark.run_performance_test(
           system=system,
           steps=10000,
           measurements=5
       )
       
       print(f"System size: {n_atoms} atoms")
       print(f"  Performance: {result['ns_per_day']:.1f} ± {result['std']:.1f} ns/day")
       print(f"  Memory: {result['memory_gb']:.1f} GB")

Platform-Specific Optimizations
------------------------------

HPC Cluster Optimization
~~~~~~~~~~~~~~~~~~~~~~~

**SLURM Integration**

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=proteinmd
   #SBATCH --nodes=4
   #SBATCH --ntasks-per-node=28
   #SBATCH --gres=gpu:4
   #SBATCH --time=24:00:00
   
   module load proteinmd/1.0
   module load cuda/11.8
   
   # Set optimal environment variables
   export OMP_NUM_THREADS=7  # 28 tasks / 4 GPUs
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   
   # Run simulation
   mpirun -np 112 proteinmd-cli run simulation.yaml

**MPI Configuration**

.. code-block:: python

   from proteinmd.parallel import MPISettings
   
   # Configure MPI for large-scale simulations
   mpi_settings = MPISettings()
   mpi_settings.set_communication_mode("cuda_aware")
   mpi_settings.enable_overlap()  # Overlap computation and communication
   mpi_settings.set_buffer_size_mb(256)  # Large communication buffers
   
   simulation.use_mpi(mpi_settings)

Cloud Optimization
~~~~~~~~~~~~~~~~~

**AWS/GCP/Azure Setup**

.. code-block:: python

   from proteinmd.cloud import CloudOptimizer
   
   # Optimize for cloud instances
   cloud_opt = CloudOptimizer()
   cloud_opt.detect_instance_type()  # Auto-detect cloud instance
   
   if cloud_opt.instance_type == "aws_p3.8xlarge":
       # Optimize for 4x V100 GPUs
       settings = cloud_opt.get_optimal_settings()
       simulation.apply_settings(settings)

Performance Tuning Guidelines
----------------------------

System Size Guidelines
~~~~~~~~~~~~~~~~~~~~~

**Small Systems (< 50k atoms)**

.. code-block:: python

   # Optimal settings for small systems
   perf_config = {
       'timestep': 0.002,
       'neighbor_list_update': 10,
       'output_frequency': 1000,
       'pme_grid_spacing': 0.12,
       'use_gpu': True,
       'precision': 'mixed'
   }

**Medium Systems (50k - 500k atoms)**

.. code-block:: python

   # Optimal settings for medium systems
   perf_config = {
       'timestep': 0.002,
       'neighbor_list_update': 20,
       'output_frequency': 5000,
       'pme_grid_spacing': 0.15,
       'use_multi_gpu': True,
       'domain_decomposition': True
   }

**Large Systems (> 500k atoms)**

.. code-block:: python

   # Optimal settings for large systems
   perf_config = {
       'timestep': 0.002,
       'neighbor_list_update': 25,
       'output_frequency': 10000,
       'pme_grid_spacing': 0.16,
       'use_mpi': True,
       'load_balancing': True
   }

Troubleshooting Performance Issues
---------------------------------

Common Performance Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Low ns/day Performance**

.. code-block:: python

   # Diagnostic steps
   diagnostic = PerformanceDiagnostic()
   
   # Check bottlenecks
   bottlenecks = diagnostic.identify_bottlenecks(simulation)
   
   for bottleneck in bottlenecks:
       print(f"Bottleneck: {bottleneck.component}")
       print(f"  Impact: {bottleneck.impact_percent:.1f}%")
       print(f"  Suggestion: {bottleneck.suggestion}")

**Memory Issues**

.. code-block:: python

   # Memory optimization
   memory_optimizer = MemoryOptimizer()
   
   # Analyze memory usage
   memory_analysis = memory_optimizer.analyze(system)
   
   if memory_analysis.usage > 0.8:  # > 80% memory usage
       # Apply memory reduction strategies
       recommendations = memory_optimizer.get_recommendations(system)
       for rec in recommendations:
           print(f"Memory optimization: {rec}")

Performance Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Always profile before optimizing**: Use the built-in profiling tools
2. **Start with hardware optimization**: Ensure proper CPU/GPU utilization
3. **Optimize algorithms before scaling**: Improve single-node performance first
4. **Monitor continuously**: Keep track of performance throughout simulation
5. **Test different configurations**: What works for one system may not work for another

See Also
--------

* :doc:`../api/core` - Core simulation API
* :doc:`../user_guide/quick_start` - Getting started
* :doc:`troubleshooting` - Troubleshooting guide
* :doc:`../developer/testing` - Performance testing
