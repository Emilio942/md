Debugging Guide
===============

This guide provides comprehensive debugging strategies and tools for ProteinMD development. Effective debugging is crucial for maintaining code quality and resolving complex simulation issues.

.. contents:: Contents
   :local:
   :depth: 2

General Debugging Principles
----------------------------

Systematic Approach
~~~~~~~~~~~~~~~~~~

1. **Reproduce the Issue**
   - Create minimal test cases
   - Document exact conditions
   - Verify consistency across environments

2. **Isolate the Problem**
   - Use binary search to narrow down code sections
   - Test individual components separately
   - Check input data validity

3. **Gather Information**
   - Collect relevant logs and error messages
   - Document system configuration
   - Note timing and sequence of events

4. **Form Hypotheses**
   - Based on symptoms and code knowledge
   - Test one hypothesis at a time
   - Document findings

Common Debugging Scenarios
--------------------------

Simulation Accuracy Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**Energy Conservation Problems**

.. code-block:: python

   # Debug energy drift
   def debug_energy_conservation(simulation):
       """Debug energy conservation in MD simulation."""
       energies = []
       times = []
       
       for step in range(1000):
           simulation.step()
           if step % 10 == 0:
               energy = simulation.get_total_energy()
               energies.append(energy)
               times.append(simulation.time)
               
               # Check for sudden energy jumps
               if len(energies) > 1:
                   delta = abs(energies[-1] - energies[-2])
                   if delta > 0.1:  # Threshold
                       print(f"Large energy change at step {step}: {delta}")
                       simulation.save_state(f"debug_step_{step}.pdb")
       
       # Plot energy vs time
       import matplotlib.pyplot as plt
       plt.plot(times, energies)
       plt.xlabel('Time (ps)')
       plt.ylabel('Total Energy (kcal/mol)')
       plt.title('Energy Conservation Check')
       plt.savefig('energy_debug.png')

**Force Calculation Validation**

.. code-block:: python

   def validate_forces(system, tolerance=1e-6):
       """Validate forces using numerical differentiation."""
       import numpy as np
       
       positions = system.get_positions()
       analytical_forces = system.get_forces()
       
       numerical_forces = np.zeros_like(positions)
       h = 1e-8  # Small displacement
       
       for i in range(positions.shape[0]):
           for j in range(3):
               # Forward difference
               pos_plus = positions.copy()
               pos_plus[i, j] += h
               system.set_positions(pos_plus)
               energy_plus = system.get_potential_energy()
               
               # Backward difference
               pos_minus = positions.copy()
               pos_minus[i, j] -= h
               system.set_positions(pos_minus)
               energy_minus = system.get_potential_energy()
               
               # Numerical force
               numerical_forces[i, j] = -(energy_plus - energy_minus) / (2 * h)
       
       # Restore original positions
       system.set_positions(positions)
       
       # Compare forces
       diff = np.abs(analytical_forces - numerical_forces)
       max_diff = np.max(diff)
       
       if max_diff > tolerance:
           print(f"Force validation failed: max difference = {max_diff}")
           return False
       return True

Performance Debugging
~~~~~~~~~~~~~~~~~~~~~

**Profiling Slow Code**

.. code-block:: python

   import cProfile
   import pstats
   import io
   from pstats import SortKey
   
   def profile_simulation(simulation_func, *args, **kwargs):
       """Profile a simulation function."""
       pr = cProfile.Profile()
       pr.enable()
       
       result = simulation_func(*args, **kwargs)
       
       pr.disable()
       s = io.StringIO()
       sortby = SortKey.CUMULATIVE
       ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
       ps.print_stats()
       
       print(s.getvalue())
       return result

**Memory Usage Tracking**

.. code-block:: python

   import tracemalloc
   import psutil
   import os
   
   def monitor_memory_usage(func, *args, **kwargs):
       """Monitor memory usage during function execution."""
       # Start tracing
       tracemalloc.start()
       process = psutil.Process(os.getpid())
       
       initial_memory = process.memory_info().rss / 1024 / 1024  # MB
       print(f"Initial memory: {initial_memory:.1f} MB")
       
       try:
           result = func(*args, **kwargs)
       finally:
           # Get memory statistics
           current, peak = tracemalloc.get_traced_memory()
           final_memory = process.memory_info().rss / 1024 / 1024  # MB
           
           print(f"Final memory: {final_memory:.1f} MB")
           print(f"Memory increase: {final_memory - initial_memory:.1f} MB")
           print(f"Peak traced memory: {peak / 1024 / 1024:.1f} MB")
           
           tracemalloc.stop()
       
       return result

Debugging Tools
---------------

Python Debugger (pdb)
~~~~~~~~~~~~~~~~~~~~~

**Basic Usage**

.. code-block:: python

   import pdb
   
   def problematic_function(data):
       # Set breakpoint
       pdb.set_trace()
       
       # Your code here
       result = process_data(data)
       return result

**Advanced pdb Commands**

.. code-block:: text

   # Navigation
   n          # Next line
   s          # Step into function
   c          # Continue execution
   l          # List current code
   
   # Inspection
   p variable # Print variable
   pp variable # Pretty print
   args       # Show function arguments
   
   # Stack
   w          # Show stack trace
   u          # Move up stack frame
   d          # Move down stack frame
   
   # Execution
   !statement # Execute Python statement
   exit       # Exit debugger

IDE Debugging
~~~~~~~~~~~~~

**VS Code Configuration**

Create `.vscode/launch.json`:

.. code-block:: json

   {
       "version": "0.2.0",
       "configurations": [
           {
               "name": "Debug Tests",
               "type": "python",
               "request": "launch",
               "module": "pytest",
               "args": ["tests/", "-v"],
               "console": "integratedTerminal",
               "cwd": "${workspaceFolder}"
           },
           {
               "name": "Debug Simulation",
               "type": "python",
               "request": "launch",
               "program": "${workspaceFolder}/examples/run_simulation.py",
               "console": "integratedTerminal",
               "args": ["--debug"]
           }
       ]
   }

Logging for Debugging
~~~~~~~~~~~~~~~~~~~~~

**Structured Logging**

.. code-block:: python

   import logging
   import sys
   
   def setup_debug_logging():
       """Set up comprehensive debug logging."""
       # Create logger
       logger = logging.getLogger('proteinmd')
       logger.setLevel(logging.DEBUG)
       
       # Create handlers
       console_handler = logging.StreamHandler(sys.stdout)
       file_handler = logging.FileHandler('debug.log')
       
       # Create formatters
       detailed_formatter = logging.Formatter(
           '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
       )
       console_formatter = logging.Formatter(
           '%(levelname)s - %(message)s'
       )
       
       # Set formatters
       console_handler.setFormatter(console_formatter)
       file_handler.setFormatter(detailed_formatter)
       
       # Add handlers
       logger.addHandler(console_handler)
       logger.addHandler(file_handler)
       
       return logger

**Context-Aware Logging**

.. code-block:: python

   import contextlib
   import logging
   
   @contextlib.contextmanager
   def debug_context(name, **kwargs):
       """Provide debugging context for operations."""
       logger = logging.getLogger('proteinmd')
       logger.debug(f"Entering {name} with {kwargs}")
       
       try:
           yield
           logger.debug(f"Successfully completed {name}")
       except Exception as e:
           logger.error(f"Error in {name}: {e}", exc_info=True)
           raise

GPU Debugging
-------------

CUDA Debugging
~~~~~~~~~~~~~~

**Memory Debugging**

.. code-block:: python

   import cupy as cp
   
   def debug_gpu_memory():
       """Debug GPU memory usage."""
       mempool = cp.get_default_memory_pool()
       
       print(f"Used bytes: {mempool.used_bytes()}")
       print(f"Total bytes: {mempool.total_bytes()}")
       
       # Get detailed memory info
       with cp.cuda.Device(0):
           free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
           print(f"GPU free memory: {free_bytes / 1024**2:.1f} MB")
           print(f"GPU total memory: {total_bytes / 1024**2:.1f} MB")

**Kernel Debugging**

.. code-block:: python

   # Enable CUDA debugging
   import os
   os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
   
   def debug_cuda_kernel(kernel_func, *args):
       """Debug CUDA kernel execution."""
       try:
           result = kernel_func(*args)
           cp.cuda.Stream.null.synchronize()  # Ensure completion
           return result
       except cp.cuda.runtime.CUDARuntimeError as e:
           print(f"CUDA error: {e}")
           # Print device information
           device = cp.cuda.Device()
           print(f"Device: {device}")
           print(f"Compute capability: {device.compute_capability}")
           raise

Testing and Validation
----------------------

Assertion-Based Debugging
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def validate_simulation_state(simulation):
       """Validate simulation state with assertions."""
       # Check basic invariants
       assert simulation.n_atoms > 0, "No atoms in simulation"
       assert simulation.time >= 0, f"Negative time: {simulation.time}"
       
       # Check physical constraints
       positions = simulation.get_positions()
       assert not np.any(np.isnan(positions)), "NaN positions detected"
       assert not np.any(np.isinf(positions)), "Infinite positions detected"
       
       velocities = simulation.get_velocities()
       max_velocity = np.max(np.linalg.norm(velocities, axis=1))
       assert max_velocity < 1000, f"Unrealistic velocity: {max_velocity}"
       
       # Check energy reasonableness
       energy = simulation.get_total_energy()
       assert not np.isnan(energy), "NaN energy detected"
       assert energy < 1e6, f"Unreasonably high energy: {energy}"

Unit Test Debugging
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   
   def test_with_debug_info():
       """Example of debugging-friendly test."""
       simulation = create_test_simulation()
       
       # Capture initial state
       initial_energy = simulation.get_total_energy()
       initial_positions = simulation.get_positions().copy()
       
       # Run simulation
       simulation.run(steps=100)
       
       # Debug information on failure
       if not np.allclose(simulation.get_positions(), expected_positions, rtol=1e-3):
           print(f"Initial energy: {initial_energy}")
           print(f"Final energy: {simulation.get_total_energy()}")
           print(f"Position drift: {np.max(np.abs(simulation.get_positions() - expected_positions))}")
           
           # Save debug files
           simulation.save_trajectory('debug_trajectory.dcd')
           
       assert np.allclose(simulation.get_positions(), expected_positions, rtol=1e-3)

Common Issues and Solutions
--------------------------

Numerical Instabilities
~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- NaN or infinite values
- Energy explosions
- Unrealistic atomic positions

**Solutions:**

.. code-block:: python

   def handle_numerical_instability(simulation):
       """Handle and recover from numerical instabilities."""
       # Check for problems
       positions = simulation.get_positions()
       if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
           print("Numerical instability detected!")
           
           # Try recovery strategies
           if simulation.has_checkpoint():
               print("Restoring from checkpoint...")
               simulation.restore_checkpoint()
               simulation.timestep *= 0.5  # Reduce timestep
               print(f"Reduced timestep to {simulation.timestep}")
           else:
               print("No checkpoint available, minimizing energy...")
               simulation.minimize_energy()

Memory Leaks
~~~~~~~~~~~~

**Detection:**

.. code-block:: python

   def detect_memory_leak(test_function, iterations=10):
       """Detect memory leaks in repeated function calls."""
       import gc
       import psutil
       import os
       
       process = psutil.Process(os.getpid())
       initial_memory = process.memory_info().rss
       
       for i in range(iterations):
           test_function()
           gc.collect()  # Force garbage collection
           
           current_memory = process.memory_info().rss
           memory_increase = (current_memory - initial_memory) / 1024 / 1024  # MB
           
           print(f"Iteration {i+1}: Memory increase = {memory_increase:.1f} MB")
           
           if memory_increase > 100:  # 100 MB threshold
               print("Potential memory leak detected!")
               return True
       
       return False

Performance Regressions
~~~~~~~~~~~~~~~~~~~~~~

**Benchmarking for Debugging:**

.. code-block:: python

   import time
   import statistics
   
   def benchmark_function(func, *args, runs=5, **kwargs):
       """Benchmark function performance for debugging."""
       times = []
       
       for _ in range(runs):
           start_time = time.perf_counter()
           result = func(*args, **kwargs)
           end_time = time.perf_counter()
           times.append(end_time - start_time)
       
       mean_time = statistics.mean(times)
       std_time = statistics.stdev(times) if len(times) > 1 else 0
       
       print(f"Function: {func.__name__}")
       print(f"Mean time: {mean_time:.4f} ± {std_time:.4f} seconds")
       print(f"Min time: {min(times):.4f} seconds")
       print(f"Max time: {max(times):.4f} seconds")
       
       return result

Best Practices
--------------

Debugging Workflow
~~~~~~~~~~~~~~~~~

1. **Start with Simple Cases**
   - Test with minimal examples
   - Use known-good reference data
   - Isolate specific components

2. **Use Version Control**
   - Create debug branches
   - Commit frequently during debugging
   - Tag working versions

3. **Document Findings**
   - Keep debugging notes
   - Share solutions with team
   - Update documentation

4. **Automate Validation**
   - Create regression tests
   - Add assertions for invariants
   - Use continuous integration

Debugging Checklist
~~~~~~~~~~~~~~~~~~

Before reporting a bug:

- [ ] Can you reproduce the issue consistently?
- [ ] Have you tested with different input data?
- [ ] Is the issue present in the latest version?
- [ ] Have you checked for similar issues in the bug tracker?
- [ ] Do you have a minimal example that demonstrates the problem?
- [ ] Have you gathered all relevant error messages and logs?
- [ ] Have you tested on different platforms/environments?

Tools and Resources
------------------

Essential Tools
~~~~~~~~~~~~~~

1. **Python Debugger (pdb/ipdb)**
   - Built-in debugging capabilities
   - Interactive debugging session

2. **Memory Profilers**
   - ``memory_profiler`` for line-by-line memory usage
   - ``tracemalloc`` for memory allocation tracking

3. **Performance Profilers**
   - ``cProfile`` for function-level profiling
   - ``line_profiler`` for line-by-line timing

4. **GPU Tools**
   - ``nvidia-smi`` for GPU monitoring
   - ``nvprof`` for CUDA profiling
   - ``compute-sanitizer`` for memory checking

External Resources
~~~~~~~~~~~~~~~~~

- **Documentation:** Official Python debugging guide
- **Books:** "Effective Python" debugging chapters
- **Tools:** PyCharm, VS Code debugging features
- **Community:** StackOverflow, Python debugging forums

Remember: Good debugging skills develop with practice. Start with simple techniques and gradually incorporate more advanced tools as needed.
