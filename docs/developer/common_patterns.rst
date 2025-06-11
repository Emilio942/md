Common Patterns
===============

This guide documents common design patterns, coding idioms, and best practices used throughout the ProteinMD codebase to maintain consistency and code quality.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

ProteinMD follows established design patterns and coding conventions to ensure maintainable, extensible, and performant code. This guide serves as a reference for developers to understand and apply these patterns consistently.

Design Patterns
--------------

Factory Pattern
~~~~~~~~~~~~~~

Used extensively for creating objects with complex initialization logic.

**Force Factory:**

.. code-block:: python

   from abc import ABC, abstractmethod
   from typing import Dict, Any
   
   class ForceFactory:
       """Factory for creating force calculation objects.
       
       Centralizes force creation logic and ensures proper
       configuration based on simulation parameters.
       """
       
       _force_types = {}
       
       @classmethod
       def register_force_type(cls, name: str, force_class):
           """Register a new force type.
           
           Args:
               name (str): Force type identifier
               force_class: Force class constructor
           """
           cls._force_types[name] = force_class
           
       @classmethod
       def create_force(cls, force_type: str, **kwargs):
           """Create force instance.
           
           Args:
               force_type (str): Type of force to create
               **kwargs: Force-specific parameters
               
           Returns:
               Force: Configured force instance
           """
           if force_type not in cls._force_types:
               raise ValueError(f"Unknown force type: {force_type}")
               
           force_class = cls._force_types[force_type]
           return force_class(**kwargs)
       
       @classmethod
       def list_available_forces(cls):
           """List all registered force types."""
           return list(cls._force_types.keys())
   
   # Usage example
   from proteinmd.forces import LennardJonesForce, ElectrostaticForce
   
   # Register force types
   ForceFactory.register_force_type('lennard_jones', LennardJonesForce)
   ForceFactory.register_force_type('electrostatic', ElectrostaticForce)
   
   # Create forces
   lj_force = ForceFactory.create_force('lennard_jones', cutoff=1.2)
   elec_force = ForceFactory.create_force('electrostatic', method='pme')

**System Builder Factory:**

.. code-block:: python

   class SystemBuilderFactory:
       """Factory for creating system builders for different input types."""
       
       _builders = {
           '.pdb': 'PDBSystemBuilder',
           '.gro': 'GromacsSystemBuilder',
           '.mol2': 'Mol2SystemBuilder',
           '.xyz': 'XYZSystemBuilder'
       }
       
       @classmethod
       def create_builder(cls, filename: str):
           """Create appropriate system builder based on file extension.
           
           Args:
               filename (str): Input file path
               
           Returns:
               SystemBuilder: Appropriate builder instance
           """
           import os
           _, ext = os.path.splitext(filename)
           
           if ext not in cls._builders:
               raise ValueError(f"Unsupported file format: {ext}")
               
           builder_name = cls._builders[ext]
           builder_module = __import__(f'proteinmd.io.builders', fromlist=[builder_name])
           builder_class = getattr(builder_module, builder_name)
           
           return builder_class(filename)

Strategy Pattern
~~~~~~~~~~~~~~~

Used for algorithms that can be interchanged at runtime.

**Integration Strategy:**

.. code-block:: python

   from abc import ABC, abstractmethod
   
   class IntegrationStrategy(ABC):
       """Abstract base class for integration algorithms."""
       
       @abstractmethod
       def step(self, state, forces, timestep):
           """Perform one integration step.
           
           Args:
               state (SimulationState): Current simulation state
               forces (array): Forces on all particles
               timestep (float): Integration timestep
           """
           pass
   
   class VelocityVerletStrategy(IntegrationStrategy):
       """Velocity Verlet integration algorithm."""
       
       def step(self, state, forces, timestep):
           """Velocity Verlet integration step."""
           # Store old forces for second velocity update
           old_forces = forces.copy()
           
           # Update positions: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
           accelerations = forces / state.masses[:, np.newaxis]
           state.positions += state.velocities * timestep + 0.5 * accelerations * timestep**2
           
           # Update velocities (first half): v(t+dt/2) = v(t) + 0.5*a(t)*dt
           state.velocities += 0.5 * accelerations * timestep
           
           # Calculate new forces at new positions
           # (This would be done by the simulation engine)
           
           # Update velocities (second half): v(t+dt) = v(t+dt/2) + 0.5*a(t+dt)*dt
           new_accelerations = forces / state.masses[:, np.newaxis]
           state.velocities += 0.5 * new_accelerations * timestep
   
   class LeapfrogStrategy(IntegrationStrategy):
       """Leapfrog integration algorithm."""
       
       def step(self, state, forces, timestep):
           """Leapfrog integration step."""
           accelerations = forces / state.masses[:, np.newaxis]
           
           # Update velocities: v(t+dt/2) = v(t-dt/2) + a(t)*dt
           state.velocities += accelerations * timestep
           
           # Update positions: x(t+dt) = x(t) + v(t+dt/2)*dt
           state.positions += state.velocities * timestep
   
   class Integrator:
       """Context class that uses integration strategies."""
       
       def __init__(self, strategy: IntegrationStrategy, timestep: float):
           self.strategy = strategy
           self.timestep = timestep
           
       def set_strategy(self, strategy: IntegrationStrategy):
           """Change integration strategy at runtime."""
           self.strategy = strategy
           
       def step(self, state, forces):
           """Perform integration step using current strategy."""
           self.strategy.step(state, forces, self.timestep)

Observer Pattern
~~~~~~~~~~~~~~~

Used for event handling and data collection during simulations.

**Simulation Observer:**

.. code-block:: python

   from abc import ABC, abstractmethod
   from typing import List
   
   class SimulationObserver(ABC):
       """Abstract base class for simulation observers."""
       
       @abstractmethod
       def notify(self, event_type: str, data: Any):
           """Handle simulation event.
           
           Args:
               event_type (str): Type of event ('step', 'energy', 'temperature', etc.)
               data: Event-specific data
           """
           pass
   
   class EnergyObserver(SimulationObserver):
       """Observer that tracks energy during simulation."""
       
       def __init__(self):
           self.energies = []
           self.times = []
           
       def notify(self, event_type: str, data: Any):
           """Handle energy-related events."""
           if event_type == 'energy_calculated':
               self.energies.append(data['total_energy'])
               self.times.append(data['time'])
               
           elif event_type == 'step_completed':
               # Could also trigger energy calculation here
               pass
   
   class TrajectoryObserver(SimulationObserver):
       """Observer that saves trajectory data."""
       
       def __init__(self, filename: str, save_interval: int = 1):
           self.filename = filename
           self.save_interval = save_interval
           self.step_count = 0
           self.writer = None
           
       def notify(self, event_type: str, data: Any):
           """Handle trajectory events."""
           if event_type == 'step_completed':
               self.step_count += 1
               
               if self.step_count % self.save_interval == 0:
                   if self.writer is None:
                       from proteinmd.io import TrajectoryWriter
                       self.writer = TrajectoryWriter(self.filename)
                   
                   self.writer.write_frame(data['state'])
   
   class SimulationSubject:
       """Subject that notifies observers of simulation events."""
       
       def __init__(self):
           self._observers: List[SimulationObserver] = []
           
       def attach(self, observer: SimulationObserver):
           """Attach observer to receive notifications."""
           self._observers.append(observer)
           
       def detach(self, observer: SimulationObserver):
           """Detach observer from notifications."""
           self._observers.remove(observer)
           
       def notify_observers(self, event_type: str, data: Any):
           """Notify all observers of event."""
           for observer in self._observers:
               observer.notify(event_type, data)

Command Pattern
~~~~~~~~~~~~~~

Used for operations that can be undone or queued.

**Simulation Commands:**

.. code-block:: python

   from abc import ABC, abstractmethod
   from typing import Any
   
   class Command(ABC):
       """Abstract base class for simulation commands."""
       
       @abstractmethod
       def execute(self):
           """Execute the command."""
           pass
           
       @abstractmethod
       def undo(self):
           """Undo the command."""
           pass
   
   class MinimizeEnergyCommand(Command):
       """Command to minimize system energy."""
       
       def __init__(self, simulation, max_iterations: int = 1000):
           self.simulation = simulation
           self.max_iterations = max_iterations
           self.initial_positions = None
           
       def execute(self):
           """Execute energy minimization."""
           # Store initial state for undo
           self.initial_positions = self.simulation.state.positions.copy()
           
           # Perform minimization
           converged = self.simulation.minimize_energy(
               max_iterations=self.max_iterations
           )
           
           return converged
           
       def undo(self):
           """Restore initial positions."""
           if self.initial_positions is not None:
               self.simulation.state.positions = self.initial_positions.copy()
   
   class RunSimulationCommand(Command):
       """Command to run simulation for specified steps."""
       
       def __init__(self, simulation, steps: int):
           self.simulation = simulation
           self.steps = steps
           self.initial_state = None
           
       def execute(self):
           """Execute simulation run."""
           # Store initial state
           self.initial_state = self.simulation.state.copy()
           
           # Run simulation
           self.simulation.run(self.steps)
           
       def undo(self):
           """Restore initial state."""
           if self.initial_state is not None:
               self.simulation.state = self.initial_state.copy()
   
   class CommandInvoker:
       """Manages command execution and undo/redo functionality."""
       
       def __init__(self):
           self.command_history = []
           self.current_position = -1
           
       def execute_command(self, command: Command):
           """Execute command and add to history."""
           # Remove any commands after current position (for redo functionality)
           self.command_history = self.command_history[:self.current_position + 1]
           
           # Execute and store command
           result = command.execute()
           self.command_history.append(command)
           self.current_position += 1
           
           return result
           
       def undo(self):
           """Undo last command."""
           if self.current_position >= 0:
               command = self.command_history[self.current_position]
               command.undo()
               self.current_position -= 1
               return True
           return False
           
       def redo(self):
           """Redo next command."""
           if self.current_position < len(self.command_history) - 1:
               self.current_position += 1
               command = self.command_history[self.current_position]
               command.execute()
               return True
           return False

Data Access Patterns
-------------------

Repository Pattern
~~~~~~~~~~~~~~~~~

Used for data persistence and retrieval operations.

**Trajectory Repository:**

.. code-block:: python

   from abc import ABC, abstractmethod
   from typing import List, Optional
   import numpy as np
   
   class TrajectoryRepository(ABC):
       """Abstract repository for trajectory data access."""
       
       @abstractmethod
       def save_frame(self, frame_data: dict):
           """Save trajectory frame."""
           pass
           
       @abstractmethod
       def load_frame(self, frame_index: int) -> dict:
           """Load specific trajectory frame."""
           pass
           
       @abstractmethod
       def load_frames(self, start: int, end: int) -> List[dict]:
           """Load range of trajectory frames."""
           pass
           
       @abstractmethod
       def get_frame_count(self) -> int:
           """Get total number of frames."""
           pass
   
   class HDF5TrajectoryRepository(TrajectoryRepository):
       """HDF5-based trajectory repository."""
       
       def __init__(self, filename: str):
           self.filename = filename
           self.file_handle = None
           self._open_file()
           
       def _open_file(self):
           """Open HDF5 file for reading/writing."""
           import h5py
           self.file_handle = h5py.File(self.filename, 'a')
           
           # Create groups if they don't exist
           if 'trajectory' not in self.file_handle:
               self.file_handle.create_group('trajectory')
               
       def save_frame(self, frame_data: dict):
           """Save trajectory frame to HDF5."""
           traj_group = self.file_handle['trajectory']
           frame_index = len(traj_group)
           
           frame_group = traj_group.create_group(f'frame_{frame_index}')
           
           # Save positions
           frame_group.create_dataset('positions', data=frame_data['positions'])
           
           # Save optional data
           if 'velocities' in frame_data:
               frame_group.create_dataset('velocities', data=frame_data['velocities'])
           if 'forces' in frame_data:
               frame_group.create_dataset('forces', data=frame_data['forces'])
               
           # Save metadata
           frame_group.attrs['time'] = frame_data.get('time', 0.0)
           frame_group.attrs['step'] = frame_data.get('step', 0)
           
       def load_frame(self, frame_index: int) -> dict:
           """Load specific frame from HDF5."""
           traj_group = self.file_handle['trajectory']
           frame_group = traj_group[f'frame_{frame_index}']
           
           frame_data = {
               'positions': frame_group['positions'][...],
               'time': frame_group.attrs['time'],
               'step': frame_group.attrs['step']
           }
           
           # Load optional data
           if 'velocities' in frame_group:
               frame_data['velocities'] = frame_group['velocities'][...]
           if 'forces' in frame_group:
               frame_data['forces'] = frame_group['forces'][...]
               
           return frame_data
           
       def get_frame_count(self) -> int:
           """Get number of frames in trajectory."""
           return len(self.file_handle['trajectory'])

Unit of Work Pattern
~~~~~~~~~~~~~~~~~~~

Used for managing complex operations that involve multiple steps.

**Simulation Unit of Work:**

.. code-block:: python

   class SimulationUnitOfWork:
       """Manages a complete simulation workflow as a unit of work."""
       
       def __init__(self, simulation):
           self.simulation = simulation
           self.operations = []
           self.committed = False
           
       def add_operation(self, operation):
           """Add operation to unit of work."""
           self.operations.append(operation)
           
       def commit(self):
           """Execute all operations in order."""
           try:
               for operation in self.operations:
                   operation.execute()
               self.committed = True
           except Exception as e:
               self.rollback()
               raise e
               
       def rollback(self):
           """Undo all executed operations."""
           for operation in reversed(self.operations):
               if hasattr(operation, 'undo'):
                   operation.undo()
           self.committed = False
   
   # Usage example
   def run_complete_simulation_workflow(system, output_dir):
       """Example of using unit of work for complex simulation."""
       from proteinmd.core import Simulation, VelocityVerletIntegrator
       
       # Create simulation
       integrator = VelocityVerletIntegrator(timestep=0.002)
       simulation = Simulation(system, integrator)
       
       # Create unit of work
       uow = SimulationUnitOfWork(simulation)
       
       # Add operations
       uow.add_operation(MinimizeEnergyCommand(simulation))
       uow.add_operation(EquilibrationCommand(simulation, steps=10000))
       uow.add_operation(ProductionCommand(simulation, steps=100000, output_dir=output_dir))
       
       # Execute all operations
       try:
           uow.commit()
           print("Simulation workflow completed successfully")
       except Exception as e:
           print(f"Simulation workflow failed: {e}")
           # Cleanup is handled by rollback

Error Handling Patterns
----------------------

Exception Hierarchy
~~~~~~~~~~~~~~~~~~

ProteinMD uses a structured exception hierarchy for better error handling.

.. code-block:: python

   class ProteinMDError(Exception):
       """Base exception for all ProteinMD errors."""
       pass
   
   class SimulationError(ProteinMDError):
       """Errors related to simulation execution."""
       pass
   
   class ForceCalculationError(SimulationError):
       """Errors in force calculations."""
       pass
   
   class IntegrationError(SimulationError):
       """Errors in time integration."""
       pass
   
   class IOError(ProteinMDError):
       """Errors in input/output operations."""
       pass
   
   class FileFormatError(IOError):
       """Errors related to file format parsing."""
       pass
   
   class SystemError(ProteinMDError):
       """Errors in system setup or configuration."""
       pass
   
   class TopologyError(SystemError):
       """Errors in molecular topology."""
       pass
   
   # Usage with context information
   class DetailedProteinMDError(ProteinMDError):
       """Enhanced error with context information."""
       
       def __init__(self, message, context=None, suggestions=None):
           super().__init__(message)
           self.context = context or {}
           self.suggestions = suggestions or []
           
       def __str__(self):
           msg = super().__str__()
           
           if self.context:
               msg += f"\nContext: {self.context}"
               
           if self.suggestions:
               msg += f"\nSuggestions: {', '.join(self.suggestions)}"
               
           return msg

Error Recovery Pattern
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from typing import Callable, Any, Optional
   import logging
   
   class ErrorRecoveryManager:
       """Manages error recovery strategies for robust simulations."""
       
       def __init__(self):
           self.recovery_strategies = {}
           self.logger = logging.getLogger(__name__)
           
       def register_strategy(self, error_type: type, strategy: Callable):
           """Register recovery strategy for specific error type.
           
           Args:
               error_type: Exception type to handle
               strategy: Recovery function
           """
           self.recovery_strategies[error_type] = strategy
           
       def execute_with_recovery(self, operation: Callable, *args, **kwargs) -> Any:
           """Execute operation with automatic error recovery.
           
           Args:
               operation: Function to execute
               *args, **kwargs: Arguments for operation
               
           Returns:
               Result of operation or recovery
           """
           max_retries = 3
           retry_count = 0
           
           while retry_count < max_retries:
               try:
                   return operation(*args, **kwargs)
                   
               except Exception as e:
                   self.logger.warning(f"Operation failed (attempt {retry_count + 1}): {e}")
                   
                   # Try recovery
                   if type(e) in self.recovery_strategies:
                       try:
                           recovery_strategy = self.recovery_strategies[type(e)]
                           recovery_strategy(e, *args, **kwargs)
                           retry_count += 1
                           continue
                       except Exception as recovery_error:
                           self.logger.error(f"Recovery failed: {recovery_error}")
                   
                   # If no recovery or recovery failed, re-raise
                   if retry_count == max_retries - 1:
                       raise
                   
                   retry_count += 1
           
           raise RuntimeError("Maximum retry attempts exceeded")
   
   # Example recovery strategies
   def numerical_instability_recovery(error, simulation, *args, **kwargs):
       """Recovery strategy for numerical instabilities."""
       # Reduce timestep
       original_timestep = simulation.integrator.timestep
       simulation.integrator.timestep *= 0.5
       
       # Restore from checkpoint if available
       if hasattr(simulation, 'restore_checkpoint'):
           simulation.restore_checkpoint()
           
       logging.info(f"Reduced timestep from {original_timestep} to {simulation.integrator.timestep}")
   
   def memory_error_recovery(error, *args, **kwargs):
       """Recovery strategy for memory errors."""
       import gc
       
       # Force garbage collection
       gc.collect()
       
       # Try to free GPU memory if using CUDA
       try:
           import cupy as cp
           mempool = cp.get_default_memory_pool()
           mempool.free_all_blocks()
       except ImportError:
           pass
       
       logging.info("Attempted memory cleanup")

Performance Patterns
-------------------

Lazy Loading Pattern
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class LazyTrajectory:
       """Lazy-loading trajectory that loads frames on demand."""
       
       def __init__(self, filename):
           self.filename = filename
           self._frame_cache = {}
           self._metadata = None
           
       @property
       def metadata(self):
           """Lazy-load trajectory metadata."""
           if self._metadata is None:
               self._metadata = self._load_metadata()
           return self._metadata
           
       def __getitem__(self, frame_index):
           """Get frame with caching."""
           if frame_index not in self._frame_cache:
               self._frame_cache[frame_index] = self._load_frame(frame_index)
               
               # Limit cache size
               if len(self._frame_cache) > 100:
                   # Remove oldest entry
                   oldest_key = min(self._frame_cache.keys())
                   del self._frame_cache[oldest_key]
                   
           return self._frame_cache[frame_index]
           
       def _load_frame(self, frame_index):
           """Load specific frame from file."""
           # Implementation depends on file format
           pass
           
       def _load_metadata(self):
           """Load trajectory metadata."""
           # Implementation depends on file format
           pass

Object Pool Pattern
~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ArrayPool:
       """Pool of reusable NumPy arrays to reduce allocation overhead."""
       
       def __init__(self):
           self.pools = {}  # (shape, dtype) -> list of arrays
           self.in_use = set()  # Track arrays currently in use
           
       def get_array(self, shape, dtype=np.float64):
           """Get array from pool or create new one.
           
           Args:
               shape: Array shape
               dtype: Array data type
               
           Returns:
               ndarray: Reusable array
           """
           key = (tuple(shape), dtype)
           
           if key in self.pools and self.pools[key]:
               array = self.pools[key].pop()
           else:
               array = np.zeros(shape, dtype=dtype)
               
           # Mark as in use
           array_id = id(array)
           self.in_use.add(array_id)
           
           # Zero out array for clean state
           array.fill(0)
           
           return array
           
       def return_array(self, array):
           """Return array to pool for reuse.
           
           Args:
               array: Array to return to pool
           """
           array_id = id(array)
           
           if array_id in self.in_use:
               self.in_use.remove(array_id)
               
               key = (array.shape, array.dtype)
               if key not in self.pools:
                   self.pools[key] = []
                   
               self.pools[key].append(array)
               
       def clear(self):
           """Clear all pools."""
           self.pools.clear()
           self.in_use.clear()
   
   # Usage with context manager
   from contextlib import contextmanager
   
   @contextmanager
   def temporary_array(pool, shape, dtype=np.float64):
       """Context manager for temporary array usage."""
       array = pool.get_array(shape, dtype)
       try:
           yield array
       finally:
           pool.return_array(array)

Functional Programming Patterns
------------------------------

Functional Composition
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from functools import reduce, partial
   from typing import Callable, List, Any
   
   def compose(*functions):
       """Compose multiple functions into a single function.
       
       Args:
           *functions: Functions to compose (right to left)
           
       Returns:
           Callable: Composed function
       """
       return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
   
   def pipe(value, *functions):
       """Apply functions to value in sequence (left to right).
       
       Args:
           value: Initial value
           *functions: Functions to apply
           
       Returns:
           Final transformed value
       """
       return reduce(lambda v, f: f(v), functions, value)
   
   # Example: Analysis pipeline
   def center_coordinates(positions):
       """Center coordinates at origin."""
       center = np.mean(positions, axis=0)
       return positions - center
   
   def align_to_reference(positions, reference):
       """Align positions to reference structure."""
       # Simplified alignment
       return positions  # Would implement Kabsch algorithm
   
   def calculate_rmsd(positions, reference):
       """Calculate RMSD."""
       diff = positions - reference
       return np.sqrt(np.mean(np.sum(diff**2, axis=1)))
   
   # Functional pipeline
   def analyze_structure(positions, reference):
       """Analyze structure using functional composition."""
       align_func = partial(align_to_reference, reference=reference)
       rmsd_func = partial(calculate_rmsd, reference=reference)
       
       return pipe(
           positions,
           center_coordinates,
           align_func,
           rmsd_func
       )

Immutable Data Patterns
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dataclasses import dataclass, replace
   from typing import Tuple
   import numpy as np
   
   @dataclass(frozen=True)
   class ImmutableSimulationState:
       """Immutable simulation state for functional programming."""
       
       positions: np.ndarray
       velocities: np.ndarray
       forces: np.ndarray
       time: float
       step: int
       
       def with_positions(self, new_positions):
           """Create new state with updated positions."""
           return replace(self, positions=new_positions)
           
       def with_velocities(self, new_velocities):
           """Create new state with updated velocities."""
           return replace(self, velocities=new_velocities)
           
       def with_time(self, new_time, new_step=None):
           """Create new state with updated time."""
           if new_step is None:
               new_step = self.step + 1
           return replace(self, time=new_time, step=new_step)
   
   def integration_step(state: ImmutableSimulationState, 
                       forces: np.ndarray, 
                       timestep: float) -> ImmutableSimulationState:
       """Pure function for integration step."""
       # Velocity Verlet integration
       masses = np.ones(len(state.positions))  # Simplified
       
       accelerations = forces / masses[:, np.newaxis]
       
       new_positions = (state.positions + 
                       state.velocities * timestep + 
                       0.5 * accelerations * timestep**2)
       
       new_velocities = state.velocities + accelerations * timestep
       
       new_time = state.time + timestep
       new_step = state.step + 1
       
       return ImmutableSimulationState(
           positions=new_positions,
           velocities=new_velocities,
           forces=forces,
           time=new_time,
           step=new_step
       )

Configuration Patterns
---------------------

Configuration Builder
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dataclasses import dataclass, field
   from typing import Dict, List, Optional, Any
   
   @dataclass
   class SimulationConfig:
       """Configuration for simulation parameters."""
       
       # System parameters
       system_file: str = ""
       topology_file: Optional[str] = None
       force_field: str = "amber99sb"
       
       # Simulation parameters
       timestep: float = 0.002  # ps
       temperature: float = 300.0  # K
       pressure: Optional[float] = None  # bar (None = NVT, value = NPT)
       
       # Integration parameters
       integrator: str = "velocity_verlet"
       constraints: List[str] = field(default_factory=list)
       
       # Output parameters
       output_dir: str = "output"
       trajectory_interval: int = 1000
       energy_interval: int = 100
       
       # Performance parameters
       platform: str = "cpu"
       gpu_device: int = 0
       threads: Optional[int] = None
   
   class ConfigurationBuilder:
       """Builder for creating simulation configurations."""
       
       def __init__(self):
           self.config = SimulationConfig()
           
       def with_system(self, system_file: str, topology_file: str = None):
           """Set system files."""
           self.config.system_file = system_file
           self.config.topology_file = topology_file
           return self
           
       def with_dynamics(self, timestep: float, temperature: float, pressure: float = None):
           """Set dynamics parameters."""
           self.config.timestep = timestep
           self.config.temperature = temperature
           self.config.pressure = pressure
           return self
           
       def with_output(self, output_dir: str, traj_interval: int = 1000):
           """Set output parameters."""
           self.config.output_dir = output_dir
           self.config.trajectory_interval = traj_interval
           return self
           
       def on_platform(self, platform: str, device: int = 0):
           """Set computational platform."""
           self.config.platform = platform
           self.config.gpu_device = device
           return self
           
       def build(self) -> SimulationConfig:
           """Build final configuration."""
           self._validate_config()
           return self.config
           
       def _validate_config(self):
           """Validate configuration parameters."""
           if not self.config.system_file:
               raise ValueError("System file must be specified")
               
           if self.config.timestep <= 0:
               raise ValueError("Timestep must be positive")
               
           if self.config.temperature <= 0:
               raise ValueError("Temperature must be positive")

Validation Patterns
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from abc import ABC, abstractmethod
   from typing import List, Any
   
   class ValidationRule(ABC):
       """Abstract base class for validation rules."""
       
       @abstractmethod
       def validate(self, value: Any) -> bool:
           """Validate value according to rule."""
           pass
           
       @abstractmethod
       def get_error_message(self, value: Any) -> str:
           """Get error message for invalid value."""
           pass
   
   class RangeValidationRule(ValidationRule):
       """Validation rule for numeric ranges."""
       
       def __init__(self, min_value: float, max_value: float):
           self.min_value = min_value
           self.max_value = max_value
           
       def validate(self, value: Any) -> bool:
           """Check if value is in range."""
           return self.min_value <= value <= self.max_value
           
       def get_error_message(self, value: Any) -> str:
           """Get range error message."""
           return f"Value {value} not in range [{self.min_value}, {self.max_value}]"
   
   class Validator:
       """Validates objects using multiple rules."""
       
       def __init__(self):
           self.rules: Dict[str, List[ValidationRule]] = {}
           
       def add_rule(self, field_name: str, rule: ValidationRule):
           """Add validation rule for field."""
           if field_name not in self.rules:
               self.rules[field_name] = []
           self.rules[field_name].append(rule)
           
       def validate(self, obj: Any) -> List[str]:
           """Validate object and return list of errors."""
           errors = []
           
           for field_name, rules in self.rules.items():
               if hasattr(obj, field_name):
                   value = getattr(obj, field_name)
                   
                   for rule in rules:
                       if not rule.validate(value):
                           errors.append(rule.get_error_message(value))
                           
           return errors

This comprehensive guide to common patterns provides developers with established solutions for common problems in molecular dynamics software development, promoting code consistency and maintainability across the ProteinMD codebase.
