Troubleshooting Guide
====================

This comprehensive troubleshooting guide helps developers diagnose and resolve common issues encountered during ProteinMD development, testing, and deployment.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

This guide provides systematic approaches to identifying and resolving problems in ProteinMD development. Issues are organized by category with step-by-step diagnostic procedures and solutions.

General Troubleshooting Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Reproduce the issue:** Create minimal test cases that consistently trigger the problem
2. **Gather information:** Collect error messages, logs, system information, and environment details
3. **Isolate the problem:** Narrow down the scope to specific modules or functions
4. **Check documentation:** Review relevant API documentation and known issues
5. **Search for solutions:** Check issue tracker, forums, and similar reported problems
6. **Apply systematic fixes:** Try solutions in order of likelihood and impact

Build and Installation Issues
-----------------------------

Dependency Problems
~~~~~~~~~~~~~~~~~~

**Issue: Missing Dependencies**

.. code-block:: text

   ModuleNotFoundError: No module named 'numpy'
   ImportError: cannot import name 'cupy' from 'cupy'

**Diagnosis:**

.. code-block:: bash

   # Check Python environment
   python --version
   pip list
   
   # Check for specific packages
   python -c "import numpy; print(numpy.__version__)"
   python -c "import cupy; print(cupy.__version__)"

**Solutions:**

.. code-block:: bash

   # Install missing dependencies
   pip install numpy scipy matplotlib
   
   # For CUDA support
   pip install cupy-cuda11x  # Adjust CUDA version as needed
   
   # Install all ProteinMD dependencies
   pip install -r requirements.txt
   
   # Development dependencies
   pip install -r requirements-dev.txt

**Issue: Version Conflicts**

.. code-block:: text

   ERROR: pip's dependency resolver does not currently consider all the ways
   that a package can depend on another

**Solutions:**

.. code-block:: bash

   # Create clean environment
   conda create -n proteinmd python=3.9
   conda activate proteinmd
   
   # Install with conda for better dependency resolution
   conda install numpy scipy matplotlib
   pip install -e .
   
   # Check for conflicts
   pip check

Compilation Errors
~~~~~~~~~~~~~~~~~

**Issue: C++ Compilation Failures**

.. code-block:: text

   error: Microsoft Visual C++ 14.0 is required
   gcc: error: unrecognized command line option '-std=c++17'

**Solutions:**

.. code-block:: bash

   # Windows - Install Visual Studio Build Tools
   # Download from https://visualstudio.microsoft.com/downloads/
   
   # Linux - Update GCC
   sudo apt update
   sudo apt install gcc-9 g++-9
   
   # macOS - Update Xcode Command Line Tools
   xcode-select --install
   
   # Force compiler version
   export CC=gcc-9
   export CXX=g++-9
   pip install -e .

**Issue: CUDA Compilation Problems**

.. code-block:: text

   nvcc fatal : Unsupported gpu architecture 'compute_86'
   error: identifier "atomicAdd" is undefined

**Solutions:**

.. code-block:: bash

   # Check CUDA version compatibility
   nvcc --version
   nvidia-smi
   
   # Update CUDA toolkit
   # Download from https://developer.nvidia.com/cuda-downloads
   
   # Set CUDA environment variables
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   
   # Specify compute capability
   export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

Runtime Errors
--------------

Simulation Failures
~~~~~~~~~~~~~~~~~~

**Issue: Numerical Instabilities**

.. code-block:: text

   RuntimeError: NaN or infinite values detected in forces
   ValueError: Simulation exploded - particles moved too far

**Diagnosis:**

.. code-block:: python

   def diagnose_instability(simulation):
       """Diagnose numerical instability causes."""
       state = simulation.state
       
       # Check positions
       pos_stats = {
           'min': np.min(state.positions),
           'max': np.max(state.positions),
           'nan_count': np.sum(np.isnan(state.positions)),
           'inf_count': np.sum(np.isinf(state.positions))
       }
       print(f"Position stats: {pos_stats}")
       
       # Check velocities
       velocities = state.velocities
       vel_magnitude = np.linalg.norm(velocities, axis=1)
       vel_stats = {
           'max_velocity': np.max(vel_magnitude),
           'mean_velocity': np.mean(vel_magnitude),
           'nan_count': np.sum(np.isnan(velocities)),
           'inf_count': np.sum(np.isinf(velocities))
       }
       print(f"Velocity stats: {vel_stats}")
       
       # Check forces
       forces = state.forces
       force_magnitude = np.linalg.norm(forces, axis=1)
       force_stats = {
           'max_force': np.max(force_magnitude),
           'mean_force': np.mean(force_magnitude),
           'nan_count': np.sum(np.isnan(forces)),
           'inf_count': np.sum(np.isinf(forces))
       }
       print(f"Force stats: {force_stats}")
       
       # Check energy
       energy = simulation.get_total_energy()
       print(f"Total energy: {energy}")

**Solutions:**

1. **Reduce timestep:**

.. code-block:: python

   # Reduce integration timestep
   simulation.integrator.timestep *= 0.5
   print(f"Reduced timestep to {simulation.integrator.timestep}")

2. **Minimize energy first:**

.. code-block:: python

   # Perform energy minimization
   converged = simulation.minimize_energy(max_iterations=1000, tolerance=1e-6)
   if not converged:
       print("Warning: Energy minimization did not converge")

3. **Check initial conditions:**

.. code-block:: python

   def validate_initial_conditions(system):
       """Validate system initial conditions."""
       positions = system.get_positions()
       
       # Check for overlapping atoms
       from scipy.spatial.distance import pdist
       distances = pdist(positions)
       min_distance = np.min(distances)
       
       if min_distance < 0.1:  # 1 Angstrom
           print(f"Warning: Atoms too close - minimum distance: {min_distance:.3f} nm")
           return False
           
       # Check box size
       if system.periodic:
           box_vectors = system.box_vectors
           box_size = np.diag(box_vectors)
           max_coord = np.max(np.abs(positions))
           
           if max_coord > np.min(box_size) / 2:
               print("Warning: Atoms outside periodic box")
               return False
               
       return True

**Issue: Force Calculation Errors**

.. code-block:: text

   RuntimeError: CUDA out of memory
   ValueError: Incompatible array shapes in force calculation

**Diagnosis:**

.. code-block:: python

   def diagnose_force_errors(simulation):
       """Diagnose force calculation problems."""
       # Check system size
       n_atoms = simulation.system.n_atoms
       print(f"System size: {n_atoms} atoms")
       
       # Check memory usage
       import psutil
       memory_gb = psutil.virtual_memory().total / 1024**3
       print(f"System memory: {memory_gb:.1f} GB")
       
       # Check GPU memory if using CUDA
       if simulation.platform == 'cuda':
           import cupy as cp
           free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
           print(f"GPU memory: {free_bytes/1024**3:.1f}/{total_bytes/1024**3:.1f} GB")
       
       # Check force parameters
       for force in simulation.forces:
           print(f"Force: {type(force).__name__}")
           if hasattr(force, 'cutoff'):
               print(f"  Cutoff: {force.cutoff}")
           if hasattr(force, 'parameters'):
               print(f"  Parameters: {len(force.parameters)} entries")

**Solutions:**

1. **Memory optimization:**

.. code-block:: python

   # Reduce cutoff distances
   for force in simulation.forces:
       if hasattr(force, 'cutoff'):
           force.cutoff = min(force.cutoff, 1.0)  # 10 Angstrom max
   
   # Use neighbor lists
   simulation.use_neighbor_list = True
   simulation.neighbor_list_cutoff = 1.2  # 12 Angstrom

2. **GPU memory management:**

.. code-block:: python

   import cupy as cp
   
   # Clear GPU memory
   mempool = cp.get_default_memory_pool()
   mempool.free_all_blocks()
   
   # Reduce batch size for large systems
   simulation.force_calculation_batch_size = 1000

I/O and File Format Issues
-------------------------

File Reading Errors
~~~~~~~~~~~~~~~~~~

**Issue: Corrupted or Invalid Files**

.. code-block:: text

   ValueError: could not convert string to float: 'NaN'
   FileNotFoundError: [Errno 2] No such file or directory: 'input.pdb'

**Diagnosis:**

.. code-block:: python

   def validate_input_file(filename):
       """Validate input file integrity."""
       import os
       
       # Check file existence
       if not os.path.exists(filename):
           print(f"File not found: {filename}")
           return False
       
       # Check file size
       file_size = os.path.getsize(filename)
       if file_size == 0:
           print(f"Empty file: {filename}")
           return False
       
       # Check file format
       _, ext = os.path.splitext(filename)
       print(f"File extension: {ext}")
       
       # Try to read first few lines
       try:
           with open(filename, 'r') as f:
               lines = [f.readline().strip() for _ in range(5)]
               print("First 5 lines:")
               for i, line in enumerate(lines, 1):
                   print(f"  {i}: {line}")
       except UnicodeDecodeError:
           print("File appears to be binary")
           
       return True

**Solutions:**

.. code-block:: python

   # Robust file reading with error handling
   def safe_read_pdb(filename):
       """Safely read PDB file with error recovery."""
       try:
           from proteinmd.io import PDBReader
           reader = PDBReader()
           system = reader.read(filename)
           return system
           
       except FileNotFoundError:
           raise FileNotFoundError(f"Input file not found: {filename}")
           
       except ValueError as e:
           if "could not convert" in str(e):
               print("Warning: Found invalid numeric values, attempting repair...")
               return repair_and_read_pdb(filename)
           raise
           
       except Exception as e:
           print(f"Unexpected error reading {filename}: {e}")
           raise

   def repair_and_read_pdb(filename):
       """Attempt to repair and read corrupted PDB file."""
       import tempfile
       import re
       
       with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
           with open(filename, 'r') as f:
               for line_num, line in enumerate(f, 1):
                   if line.startswith(('ATOM', 'HETATM')):
                       # Validate and fix coordinate fields
                       try:
                           # Extract coordinate fields
                           x = float(line[30:38].strip())
                           y = float(line[38:46].strip())
                           z = float(line[46:54].strip())
                           tmp.write(line)
                       except ValueError:
                           print(f"Skipping invalid line {line_num}: {line.strip()}")
                   else:
                       tmp.write(line)
           
           # Read repaired file
           from proteinmd.io import PDBReader
           reader = PDBReader()
           return reader.read(tmp.name)

Memory and Performance Issues
----------------------------

Memory Leaks
~~~~~~~~~~~

**Issue: Increasing Memory Usage**

.. code-block:: text

   MemoryError: Unable to allocate array
   Warning: Memory usage increasing over time

**Diagnosis:**

.. code-block:: python

   import psutil
   import gc
   from memory_profiler import profile
   
   def monitor_memory_usage(simulation, steps=1000):
       """Monitor memory usage during simulation."""
       process = psutil.Process()
       
       initial_memory = process.memory_info().rss / 1024**2
       print(f"Initial memory: {initial_memory:.1f} MB")
       
       for step in range(0, steps, 100):
           simulation.run(100)
           
           current_memory = process.memory_info().rss / 1024**2
           memory_increase = current_memory - initial_memory
           
           print(f"Step {step + 100}: {current_memory:.1f} MB "
                 f"(+{memory_increase:.1f} MB)")
           
           if memory_increase > 1000:  # 1 GB increase
               print("Warning: Significant memory increase detected")
               break

**Solutions:**

.. code-block:: python

   # Explicit memory management
   def run_simulation_with_cleanup(simulation, total_steps, checkpoint_interval=10000):
       """Run simulation with periodic memory cleanup."""
       for chunk_start in range(0, total_steps, checkpoint_interval):
           chunk_steps = min(checkpoint_interval, total_steps - chunk_start)
           
           # Run chunk
           simulation.run(chunk_steps)
           
           # Periodic cleanup
           gc.collect()
           
           # Clear GPU memory if using CUDA
           if simulation.platform == 'cuda':
               import cupy as cp
               mempool = cp.get_default_memory_pool()
               mempool.free_all_blocks()
           
           # Save checkpoint
           simulation.save_checkpoint(f"checkpoint_{chunk_start + chunk_steps}.pkl")
           
           print(f"Completed {chunk_start + chunk_steps}/{total_steps} steps")

Performance Bottlenecks
~~~~~~~~~~~~~~~~~~~~~~

**Issue: Slow Simulation Performance**

**Diagnosis:**

.. code-block:: python

   import cProfile
   import time
   
   def profile_simulation_performance(simulation, steps=1000):
       """Profile simulation performance."""
       # Time individual components
       timings = {}
       
       # Profile force calculation
       start_time = time.time()
       simulation._update_forces()
       timings['force_calculation'] = time.time() - start_time
       
       # Profile integration
       start_time = time.time()
       simulation.integrator.step(simulation.state)
       timings['integration'] = time.time() - start_time
       
       # Profile neighbor list update
       if hasattr(simulation, 'neighbor_list'):
           start_time = time.time()
           simulation.neighbor_list.update(simulation.state.positions)
           timings['neighbor_list'] = time.time() - start_time
       
       # Full step timing
       start_time = time.time()
       simulation.step()
       timings['full_step'] = time.time() - start_time
       
       for component, timing in timings.items():
           percentage = timing / timings['full_step'] * 100
           print(f"{component}: {timing:.4f}s ({percentage:.1f}%)")

**Solutions:**

1. **Optimize force calculations:**

.. code-block:: python

   # Use appropriate cutoffs
   def optimize_force_cutoffs(simulation):
       """Optimize force calculation cutoffs."""
       for force in simulation.forces:
           if hasattr(force, 'cutoff'):
               # Reduce cutoff if too large
               if force.cutoff > 1.2:  # 12 Angstrom
                   print(f"Reducing {type(force).__name__} cutoff from "
                         f"{force.cutoff} to 1.2 nm")
                   force.cutoff = 1.2

2. **Enable neighbor lists:**

.. code-block:: python

   # Use neighbor lists for better scaling
   simulation.use_neighbor_list = True
   simulation.neighbor_list_skin = 0.2  # 2 Angstrom skin
   simulation.neighbor_list_update_frequency = 10  # Update every 10 steps

3. **Platform optimization:**

.. code-block:: python

   # Switch to GPU if available
   if simulation.platform == 'cpu':
       try:
           import cupy as cp
           if cp.cuda.is_available():
               print("Switching to CUDA platform for better performance")
               simulation.set_platform('cuda')
       except ImportError:
           print("CuPy not available, staying on CPU")

Testing and Debugging Issues
----------------------------

Test Failures
~~~~~~~~~~~~~

**Issue: Intermittent Test Failures**

.. code-block:: text

   AssertionError: Arrays are not almost equal to 6 decimals
   FAILED test_energy_conservation - assert abs(energy_drift) < 0.001

**Solutions:**

.. code-block:: python

   # More robust numerical comparisons
   import numpy.testing as npt
   
   def test_energy_conservation_robust(simulation):
       """More robust energy conservation test."""
       initial_energy = simulation.get_total_energy()
       
       # Run simulation
       energies = []
       for step in range(1000):
           simulation.step()
           if step % 100 == 0:
               energies.append(simulation.get_total_energy())
       
       # Calculate energy drift with statistical analysis
       energy_drift = np.array(energies) - initial_energy
       max_drift = np.max(np.abs(energy_drift))
       mean_drift = np.mean(np.abs(energy_drift))
       std_drift = np.std(energy_drift)
       
       # More lenient criteria for noisy simulations
       assert max_drift < 0.01, f"Maximum energy drift: {max_drift}"
       assert mean_drift < 0.005, f"Mean energy drift: {mean_drift}"
       
   # Platform-specific tolerances
   def get_tolerance_for_platform(platform):
       """Get appropriate numerical tolerance for platform."""
       tolerances = {
           'cpu': 1e-6,
           'cuda': 1e-5,  # GPU calculations less precise
           'mixed': 1e-5
       }
       return tolerances.get(platform, 1e-6)

Environment and System Issues
----------------------------

Platform Compatibility
~~~~~~~~~~~~~~~~~~~~~

**Issue: Platform-Specific Failures**

.. code-block:: text

   RuntimeError: CUDA driver version is insufficient for CUDA runtime version
   ImportError: DLL load failed while importing: The specified module could not be found

**Diagnosis:**

.. code-block:: python

   def diagnose_platform_issues():
       """Diagnose platform-specific issues."""
       import platform
       import sys
       
       print(f"Python: {sys.version}")
       print(f"Platform: {platform.platform()}")
       print(f"Architecture: {platform.architecture()}")
       
       # Check CUDA availability
       try:
           import cupy as cp
           print(f"CuPy version: {cp.__version__}")
           
           if cp.cuda.is_available():
               print(f"CUDA devices: {cp.cuda.runtime.getDeviceCount()}")
               for i in range(cp.cuda.runtime.getDeviceCount()):
                   device = cp.cuda.Device(i)
                   print(f"  Device {i}: {device.attributes['Name']}")
           else:
               print("CUDA not available")
               
       except ImportError:
           print("CuPy not installed")
       
       # Check OpenMP
       try:
           import os
           omp_threads = os.environ.get('OMP_NUM_THREADS', 'not set')
           print(f"OMP_NUM_THREADS: {omp_threads}")
       except:
           pass

**Solutions:**

.. code-block:: bash

   # CUDA driver issues
   # Update NVIDIA drivers
   sudo apt update
   sudo apt install nvidia-driver-470  # or latest version
   
   # Check CUDA installation
   nvidia-smi
   nvcc --version
   
   # Environment variables
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

Documentation and API Issues
---------------------------

Missing Documentation
~~~~~~~~~~~~~~~~~~~~

**Issue: Unclear API Usage**

**Solutions:**

.. code-block:: python

   # Generate API documentation
   def document_class_methods(cls):
       """Generate documentation for class methods."""
       print(f"Class: {cls.__name__}")
       print(f"Docstring: {cls.__doc__}")
       print("\nMethods:")
       
       for name, method in cls.__dict__.items():
           if callable(method) and not name.startswith('_'):
               print(f"  {name}: {method.__doc__}")
   
   # Usage
   from proteinmd.core import Simulation
   document_class_methods(Simulation)

Error Reporting and Logging
--------------------------

Enhanced Error Reporting
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import logging
   import traceback
   import sys
   
   def setup_enhanced_logging():
       """Set up comprehensive logging for debugging."""
       # Create logger
       logger = logging.getLogger('proteinmd')
       logger.setLevel(logging.DEBUG)
       
       # Create handlers
       console_handler = logging.StreamHandler(sys.stdout)
       file_handler = logging.FileHandler('proteinmd_debug.log')
       
       # Create formatters
       detailed_formatter = logging.Formatter(
           '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
       )
       
       console_handler.setFormatter(detailed_formatter)
       file_handler.setFormatter(detailed_formatter)
       
       # Add handlers
       logger.addHandler(console_handler)
       logger.addHandler(file_handler)
       
       return logger
   
   def create_bug_report(error, context=None):
       """Create comprehensive bug report."""
       import platform
       import datetime
       
       report = {
           'timestamp': datetime.datetime.now().isoformat(),
           'error_type': type(error).__name__,
           'error_message': str(error),
           'traceback': traceback.format_exc(),
           'python_version': platform.python_version(),
           'platform': platform.platform(),
           'context': context or {}
       }
       
       # Add environment information
       try:
           import numpy as np
           report['numpy_version'] = np.__version__
       except ImportError:
           pass
       
       try:
           import cupy as cp
           report['cupy_version'] = cp.__version__
           report['cuda_available'] = cp.cuda.is_available()
       except ImportError:
           pass
       
       return report

Getting Help
-----------

When to Seek Help
~~~~~~~~~~~~~~~~

Seek help when:

1. **Issue persists after trying documented solutions**
2. **Error messages are unclear or undocumented**
3. **Performance problems cannot be resolved with standard optimization**
4. **System requirements or compatibility questions**
5. **Contributing new features or major changes**

How to Report Issues
~~~~~~~~~~~~~~~~~~

When reporting issues, include:

1. **Minimal reproducible example**
2. **Complete error messages and stack traces**
3. **System information (OS, Python version, dependencies)**
4. **Steps to reproduce the issue**
5. **Expected vs. actual behavior**
6. **Any attempted solutions**

Example Issue Report Template:

.. code-block:: text

   **Issue Description:**
   Brief description of the problem
   
   **Environment:**
   - OS: Ubuntu 20.04
   - Python: 3.9.7
   - ProteinMD version: 1.0.0
   - NumPy: 1.21.0
   - CuPy: 9.6.0 (if applicable)
   
   **Minimal Reproducible Example:**
   ```python
   # Code that reproduces the issue
   ```
   
   **Error Message:**
   ```
   Complete error message and stack trace
   ```
   
   **Expected Behavior:**
   What you expected to happen
   
   **Actual Behavior:**
   What actually happened
   
   **Attempted Solutions:**
   - Tried solution A: result
   - Tried solution B: result

Resources
--------

- **Documentation:** https://proteinmd.readthedocs.io/
- **Issue Tracker:** https://github.com/proteinmd/proteinmd/issues
- **Discussions:** https://github.com/proteinmd/proteinmd/discussions
- **Stack Overflow:** Use tag `proteinmd`
- **Community Forum:** [Link to community forum]

This troubleshooting guide provides systematic approaches to resolving common development issues in ProteinMD. Keep it updated as new issues are discovered and resolved.
