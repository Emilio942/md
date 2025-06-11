API Design Principles
====================

This document outlines the design principles and guidelines for creating consistent, intuitive, and maintainable APIs in ProteinMD.

.. contents:: Contents
   :local:
   :depth: 2

Core Design Philosophy
---------------------

User-Centered Design
~~~~~~~~~~~~~~~~~~~

ProteinMD APIs are designed with the end user in mind - computational scientists who need to perform molecular dynamics simulations and analysis. Our design prioritizes:

**Simplicity**
  Simple tasks should be simple to accomplish. Common workflows should require minimal code.

**Clarity**
  Function and parameter names should be self-explanatory. The API should guide users toward correct usage.

**Consistency**
  Similar concepts should be represented similarly across the API. Patterns learned in one area should apply elsewhere.

**Flexibility**
  While simple tasks are easy, complex or specialized use cases should still be possible through advanced parameters or extension points.

**Performance**
  The API should enable efficient implementations without sacrificing usability.

Design Principles
~~~~~~~~~~~~~~~~

**1. Principle of Least Surprise**

Users should be able to predict API behavior based on names and conventions:

.. code-block:: python

   # Good: Predictable behavior
   trajectory.save("output.xtc")  # Saves to file
   trajectory.load("input.xtc")   # Loads from file
   
   # Bad: Surprising behavior  
   trajectory.write("output.xtc", mode="read")  # Confusing

**2. Fail Fast and Clearly**

Problems should be detected as early as possible with clear error messages:

.. code-block:: python

   # Good: Early validation with clear message
   def set_temperature(self, temperature: float) -> None:
       if temperature <= 0:
           raise ValueError(
               f"Temperature must be positive, got {temperature}"
           )
       self._temperature = temperature
   
   # Bad: Late failure with unclear message
   def set_temperature(self, temperature: float) -> None:
       self._temperature = temperature  # Validation happens later
       # ... simulation crashes with "division by zero"

**3. Sensible Defaults**

Provide reasonable defaults for optional parameters:

.. code-block:: python

   # Good: Sensible defaults for common use case
   def run_simulation(
       system: MDSystem,
       steps: int,
       temperature: float = 300.0,    # Room temperature
       time_step: float = 0.002,      # 2 fs - stable for most systems
       output_frequency: int = 1000   # Save every 1000 steps
   ) -> None:
       pass

**4. Progressive Disclosure**

Simple APIs for common cases, with advanced options available when needed:

.. code-block:: python

   # Simple case - good defaults
   minimizer = EnergyMinimizer()
   minimizer.minimize(system)
   
   # Advanced case - full control
   minimizer = EnergyMinimizer(
       algorithm="L-BFGS",
       max_iterations=5000,
       tolerance=1e-8,
       line_search="strong_wolfe"
   )
   minimizer.minimize(system, constraints=constraints)

API Patterns
-----------

Naming Conventions
~~~~~~~~~~~~~~~~~

**Functions and Methods**

Use clear, descriptive verb phrases:

.. code-block:: python

   # Good: Clear action verbs
   system.add_solvent()
   system.remove_hydrogen()
   trajectory.calculate_rmsd()
   forcefield.load_parameters()
   
   # Bad: Unclear or missing verbs
   system.solvent()        # Unclear what happens
   system.hydrogen(False)  # Unclear meaning
   trajectory.rmsd()       # Missing verb

**Properties vs Methods**

Use properties for simple attribute access, methods for computations:

.. code-block:: python

   class MDSystem:
       @property
       def n_atoms(self) -> int:
           """Number of atoms in the system."""
           return self._positions.shape[0]
       
       @property 
       def temperature(self) -> float:
           """Current system temperature."""
           return self._temperature
       
       def calculate_kinetic_energy(self) -> float:
           """Calculate total kinetic energy (requires computation)."""
           return 0.5 * np.sum(self._masses * self._velocities**2)

**Boolean Parameters**

Use positive, descriptive names for boolean parameters:

.. code-block:: python

   # Good: Positive, clear meaning
   def minimize_energy(
       self,
       include_solvent: bool = True,
       use_constraints: bool = False,
       verbose: bool = False
   ) -> None:
       pass
   
   # Bad: Negative or unclear
   def minimize_energy(
       self,
       no_solvent: bool = False,  # Double negative
       constrained: bool = False,  # Ambiguous
       quiet: bool = True         # Negative logic
   ) -> None:
       pass

Constructor Patterns
~~~~~~~~~~~~~~~~~~~

**Primary Constructor**

The main constructor should handle the most common initialization:

.. code-block:: python

   class MDSystem:
       def __init__(
           self,
           name: str = "MD_System",
           temperature: float = 300.0,
           pressure: float = 1.0
       ):
           """Initialize MD system with basic parameters."""
           self.name = name
           self._temperature = temperature
           self._pressure = pressure
           self._positions = None
           self._velocities = None

**Factory Methods**

Provide class methods for alternative construction patterns:

.. code-block:: python

   class MDSystem:
       @classmethod
       def from_pdb(
           cls,
           pdb_file: str,
           temperature: float = 300.0,
           add_solvent: bool = True
       ) -> "MDSystem":
           """Create system from PDB file."""
           system = cls(temperature=temperature)
           system.load_structure(pdb_file)
           if add_solvent:
               system.add_solvent()
           return system
       
       @classmethod
       def from_topology(
           cls,
           topology_file: str,
           coordinate_file: str,
           **kwargs
       ) -> "MDSystem":
           """Create system from topology and coordinate files."""
           system = cls(**kwargs)
           system.load_topology(topology_file)
           system.load_coordinates(coordinate_file)
           return system

Configuration and Options
~~~~~~~~~~~~~~~~~~~~~~~~~

**Configuration Objects**

Use dedicated configuration classes for complex option sets:

.. code-block:: python

   @dataclass
   class SimulationConfig:
       """Configuration for MD simulation."""
       temperature: float = 300.0
       pressure: float = 1.0
       time_step: float = 0.002
       integrator: str = "verlet"
       thermostat: str = "berendsen"
       barostat: str = "parrinello_rahman"
       output_frequency: int = 1000
       
       def validate(self) -> None:
           """Validate configuration parameters."""
           if self.temperature <= 0:
               raise ValueError("Temperature must be positive")
           if self.time_step <= 0:
               raise ValueError("Time step must be positive")
   
   # Usage
   config = SimulationConfig(temperature=310.0, time_step=0.001)
   simulation = MDSimulation(system, config)

**Builder Pattern**

For complex object construction:

.. code-block:: python

   class SimulationBuilder:
       """Builder for MD simulations."""
       
       def __init__(self, system: MDSystem):
           self._system = system
           self._config = SimulationConfig()
       
       def temperature(self, temp: float) -> "SimulationBuilder":
           """Set simulation temperature."""
           self._config.temperature = temp
           return self
       
       def time_step(self, dt: float) -> "SimulationBuilder":
           """Set integration time step."""
           self._config.time_step = dt
           return self
       
       def integrator(self, name: str) -> "SimulationBuilder":
           """Set integration algorithm."""
           self._config.integrator = name
           return self
       
       def build(self) -> MDSimulation:
           """Build the simulation."""
           self._config.validate()
           return MDSimulation(self._system, self._config)
   
   # Usage
   simulation = (
       SimulationBuilder(system)
       .temperature(310.0)
       .time_step(0.001)
       .integrator("velocity_verlet")
       .build()
   )

Context Managers
~~~~~~~~~~~~~~~

Use context managers for resource management:

.. code-block:: python

   class TrajectoryWriter:
       """Context manager for writing trajectory files."""
       
       def __init__(self, filename: str, format: str = "xtc"):
           self._filename = filename
           self._format = format
           self._file = None
       
       def __enter__(self) -> "TrajectoryWriter":
           self._file = open_trajectory_file(self._filename, self._format)
           return self
       
       def __exit__(self, exc_type, exc_val, exc_tb):
           if self._file:
               self._file.close()
       
       def write_frame(self, positions: np.ndarray, time: float) -> None:
           """Write a single trajectory frame."""
           if not self._file:
               raise RuntimeError("Writer not opened")
           self._file.write_frame(positions, time)
   
   # Usage
   with TrajectoryWriter("output.xtc") as writer:
       for step in range(1000):
           # ... run simulation step ...
           writer.write_frame(positions, time)

Error Handling Patterns
-----------------------

Exception Hierarchy
~~~~~~~~~~~~~~~~~~

Define a clear exception hierarchy for different error types:

.. code-block:: python

   class ProteinMDError(Exception):
       """Base exception for ProteinMD."""
       pass

   class SimulationError(ProteinMDError):
       """Base class for simulation-related errors."""
       pass

   class ConvergenceError(SimulationError):
       """Simulation failed to converge."""
       pass

   class StabilityError(SimulationError):
       """Simulation became unstable."""
       pass

   class ForceFieldError(ProteinMDError):
       """Force field related errors."""
       pass

   class ParameterError(ForceFieldError):
       """Invalid or missing force field parameters."""
       pass

   class FileFormatError(ProteinMDError):
       """File format or I/O related errors."""
       pass

Error Messages
~~~~~~~~~~~~~

Provide actionable error messages with context:

.. code-block:: python

   def load_structure(self, filename: str) -> None:
       """Load molecular structure from file."""
       if not os.path.exists(filename):
           raise FileNotFoundError(
               f"Structure file not found: {filename}. "
               f"Please check the file path and ensure the file exists."
           )
       
       try:
           # Load structure
           structure = parse_structure_file(filename)
       except Exception as e:
           raise FileFormatError(
               f"Failed to parse structure file {filename}. "
               f"Supported formats: PDB, MOL2, XYZ. "
               f"Original error: {e}"
           ) from e

Validation Patterns
~~~~~~~~~~~~~~~~~~

**Input Validation**

Validate inputs early and provide clear feedback:

.. code-block:: python

   def set_box_vectors(self, vectors: np.ndarray) -> None:
       """Set periodic boundary condition box vectors.
       
       Args:
           vectors: Box vectors as 3x3 matrix where each row is a vector
           
       Raises:
           ValueError: If vectors array has wrong shape or invalid values
       """
       vectors = np.asarray(vectors)
       
       if vectors.shape != (3, 3):
           raise ValueError(
               f"Box vectors must be 3x3 array, got shape {vectors.shape}"
           )
       
       if np.any(np.diag(vectors) <= 0):
           raise ValueError(
               "Diagonal elements of box vectors must be positive"
           )
       
       # Check for reasonable box dimensions
       if np.any(np.diag(vectors) > 1000.0):  # 1000 nm
           warnings.warn(
               "Very large box dimensions detected. "
               "This may cause performance issues."
           )
       
       self._box_vectors = vectors.copy()

**State Validation**

Check object state before operations:

.. code-block:: python

   def run_simulation(self, steps: int) -> None:
       """Run molecular dynamics simulation."""
       # Check system is properly initialized
       if self._positions is None:
           raise RuntimeError(
               "System positions not set. Load structure first."
           )
       
       if self._velocities is None:
           raise RuntimeError(
               "System velocities not set. "
               "Call initialize_velocities() or load from file."
           )
       
       if self._force_field is None:
           raise RuntimeError(
               "Force field not set. Set force field before simulation."
           )
       
       # Proceed with simulation
       self._run_simulation_steps(steps)

Data Access Patterns
-------------------

Properties and Accessors
~~~~~~~~~~~~~~~~~~~~~~~

**Read-Only Properties**

Use properties for computed or derived values:

.. code-block:: python

   class MDSystem:
       @property
       def kinetic_energy(self) -> float:
           """Current kinetic energy of the system."""
           if self._velocities is None:
               return 0.0
           return 0.5 * np.sum(self._masses * np.sum(self._velocities**2, axis=1))
       
       @property
       def total_mass(self) -> float:
           """Total mass of all atoms."""
           return np.sum(self._masses) if self._masses is not None else 0.0
       
       @property
       def center_of_mass(self) -> np.ndarray:
           """Center of mass coordinates."""
           if self._positions is None or self._masses is None:
               return np.zeros(3)
           return np.average(self._positions, weights=self._masses, axis=0)

**Mutable Properties**

Provide setters with validation for mutable properties:

.. code-block:: python

   class MDSystem:
       @property
       def temperature(self) -> float:
           """Target temperature for the simulation."""
           return self._temperature
       
       @temperature.setter
       def temperature(self, value: float) -> None:
           """Set target temperature."""
           if value <= 0:
               raise ValueError("Temperature must be positive")
           self._temperature = value
           # Update thermostat if present
           if hasattr(self, '_thermostat') and self._thermostat:
               self._thermostat.set_temperature(value)

Array Access
~~~~~~~~~~~

**Safe Array Access**

Provide safe access to numpy arrays with appropriate copying:

.. code-block:: python

   class MDSystem:
       def get_positions(self, copy: bool = True) -> np.ndarray:
           """Get atomic positions.
           
           Args:
               copy: If True, return a copy. If False, return view.
                    Be careful with views as modifications affect the original.
           
           Returns:
               Atomic positions, shape (n_atoms, 3)
           """
           if self._positions is None:
               return np.empty((0, 3))
           return self._positions.copy() if copy else self._positions
       
       def set_positions(self, positions: np.ndarray) -> None:
           """Set atomic positions.
           
           Args:
               positions: New positions, shape (n_atoms, 3)
           """
           positions = np.asarray(positions)
           if positions.ndim != 2 or positions.shape[1] != 3:
               raise ValueError("Positions must have shape (n_atoms, 3)")
           
           self._positions = positions.copy()

Collection Access
~~~~~~~~~~~~~~~~

**Iteration Support**

Support iteration where it makes sense:

.. code-block:: python

   class Trajectory:
       """Container for MD trajectory data."""
       
       def __len__(self) -> int:
           """Number of frames in trajectory."""
           return self._n_frames
       
       def __getitem__(self, index: Union[int, slice]) -> Union[Frame, List[Frame]]:
           """Get frame(s) by index."""
           if isinstance(index, int):
               if index < 0:
                   index += self._n_frames
               if not 0 <= index < self._n_frames:
                   raise IndexError(f"Frame index {index} out of range")
               return self._get_frame(index)
           
           elif isinstance(index, slice):
               indices = range(*index.indices(self._n_frames))
               return [self._get_frame(i) for i in indices]
           
           else:
               raise TypeError("Index must be int or slice")
       
       def __iter__(self):
           """Iterate over frames."""
           for i in range(self._n_frames):
               yield self._get_frame(i)

Extensibility Patterns
---------------------

Plugin Architecture
~~~~~~~~~~~~~~~~~~

**Abstract Base Classes**

Define clear interfaces for extensible components:

.. code-block:: python

   from abc import ABC, abstractmethod

   class ForceField(ABC):
       """Abstract base class for force field implementations."""
       
       @abstractmethod
       def calculate_energy(
           self,
           positions: np.ndarray,
           box_vectors: Optional[np.ndarray] = None
       ) -> float:
           """Calculate potential energy."""
           pass
       
       @abstractmethod
       def calculate_forces(
           self,
           positions: np.ndarray, 
           box_vectors: Optional[np.ndarray] = None
       ) -> np.ndarray:
           """Calculate forces on all atoms."""
           pass
       
       @abstractmethod
       def get_parameter_names(self) -> List[str]:
           """Get list of parameter names."""
           pass

**Registration System**

Allow registration of new implementations:

.. code-block:: python

   class ForceFieldRegistry:
       """Registry for force field implementations."""
       
       _force_fields: Dict[str, Type[ForceField]] = {}
       
       @classmethod
       def register(cls, name: str, force_field_class: Type[ForceField]) -> None:
           """Register a force field implementation."""
           if not issubclass(force_field_class, ForceField):
               raise TypeError("Must be subclass of ForceField")
           cls._force_fields[name] = force_field_class
       
       @classmethod
       def create(cls, name: str, **kwargs) -> ForceField:
           """Create force field instance by name."""
           if name not in cls._force_fields:
               available = ", ".join(cls._force_fields.keys())
               raise ValueError(f"Unknown force field: {name}. Available: {available}")
           return cls._force_fields[name](**kwargs)
       
       @classmethod
       def list_available(cls) -> List[str]:
           """List available force field names."""
           return list(cls._force_fields.keys())

   # Register built-in force fields
   ForceFieldRegistry.register("charmm36", CHARMM36ForceField)
   ForceFieldRegistry.register("amber99sb", Amber99SBForceField)
   
   # Usage
   ff = ForceFieldRegistry.create("charmm36", parameter_file="charmm36.prm")

Callback Systems
~~~~~~~~~~~~~~~

**Event Hooks**

Allow users to hook into simulation events:

.. code-block:: python

   from typing import Callable, Dict, Any

   class MDSimulation:
       """MD simulation with event system."""
       
       def __init__(self):
           self._callbacks: Dict[str, List[Callable]] = {}
       
       def register_callback(
           self,
           event: str,
           callback: Callable[[Dict[str, Any]], None]
       ) -> None:
           """Register callback for simulation event.
           
           Args:
               event: Event name (e.g., 'step', 'frame_saved', 'energy_calculated')
               callback: Function to call when event occurs
           """
           if event not in self._callbacks:
               self._callbacks[event] = []
           self._callbacks[event].append(callback)
       
       def _trigger_event(self, event: str, data: Dict[str, Any]) -> None:
           """Trigger all callbacks for an event."""
           for callback in self._callbacks.get(event, []):
               try:
                   callback(data)
               except Exception as e:
                   # Log error but don't stop simulation
                   logger.warning(f"Callback error for event {event}: {e}")
       
       def run_step(self, step: int) -> None:
           """Run single simulation step."""
           # ... simulation logic ...
           
           # Trigger step event
           self._trigger_event('step', {
               'step': step,
               'time': step * self._time_step,
               'energy': self._energy,
               'temperature': self._temperature
           })

Version Compatibility
--------------------

Deprecation Handling
~~~~~~~~~~~~~~~~~~~

**Graceful Deprecation**

Provide clear deprecation warnings with migration paths:

.. code-block:: python

   import warnings
   from typing import Optional

   def old_function(parameter: float) -> float:
       """Old function - deprecated.
       
       .. deprecated:: 0.2.0
           Use :func:`new_function` instead. Will be removed in version 0.4.0.
       """
       warnings.warn(
           "old_function is deprecated and will be removed in version 0.4.0. "
           "Use new_function instead.",
           DeprecationWarning,
           stacklevel=2
       )
       return new_function(parameter)

**Version Compatibility**

Check version compatibility for data files:

.. code-block:: python

   class DataFile:
       """Handle versioned data files."""
       
       CURRENT_VERSION = "1.2"
       SUPPORTED_VERSIONS = ["1.0", "1.1", "1.2"]
       
       def load(self, filename: str) -> None:
           """Load data file with version checking."""
           with open(filename, 'r') as f:
               header = json.loads(f.readline())
           
           version = header.get('version', '1.0')
           
           if version not in self.SUPPORTED_VERSIONS:
               raise ValueError(
                   f"Unsupported file version: {version}. "
                   f"Supported versions: {self.SUPPORTED_VERSIONS}"
               )
           
           if version != self.CURRENT_VERSION:
               warnings.warn(
                   f"Loading older file format (v{version}). "
                   f"Consider updating to v{self.CURRENT_VERSION}."
               )
           
           # Load with appropriate version handler
           self._load_version(filename, version)

Documentation in Code
--------------------

Docstring Requirements
~~~~~~~~~~~~~~~~~~~~~

**Complete API Documentation**

All public functions must have comprehensive docstrings:

.. code-block:: python

   def calculate_rmsd(
       reference: np.ndarray,
       trajectory: np.ndarray,
       align: bool = True,
       mass_weighted: bool = False,
       atom_indices: Optional[np.ndarray] = None
   ) -> np.ndarray:
       """Calculate root-mean-square deviation between structures.
       
       Computes RMSD between a reference structure and each frame
       in a trajectory. Optionally performs structural alignment
       and mass weighting.
       
       Args:
           reference: Reference structure coordinates, shape (n_atoms, 3)
           trajectory: Trajectory coordinates, shape (n_frames, n_atoms, 3)
           align: Whether to perform structural alignment before RMSD calculation
           mass_weighted: Whether to use mass-weighted RMSD
           atom_indices: Subset of atoms to include, shape (n_subset,).
                        If None, use all atoms.
       
       Returns:
           RMSD values for each frame, shape (n_frames,)
       
       Raises:
           ValueError: If reference and trajectory have incompatible shapes
           TypeError: If inputs are not numpy arrays
       
       Examples:
           Basic RMSD calculation:
           
           >>> ref = np.random.random((100, 3))
           >>> traj = np.random.random((1000, 100, 3))
           >>> rmsd = calculate_rmsd(ref, traj)
           >>> print(f"Average RMSD: {rmsd.mean():.2f} Å")
           
           Mass-weighted RMSD with alignment:
           
           >>> rmsd_mw = calculate_rmsd(ref, traj, mass_weighted=True)
           >>> print(f"Mass-weighted RMSD: {rmsd_mw.mean():.2f} Å")
           
           RMSD for backbone atoms only:
           
           >>> backbone_indices = np.array([0, 1, 2, 4])  # N, CA, C, O
           >>> rmsd_bb = calculate_rmsd(ref, traj, atom_indices=backbone_indices)
       
       Note:
           - RMSD is calculated in Ångströms
           - Alignment uses Kabsch algorithm when align=True
           - Mass weighting requires masses to be set in the system
           - For large trajectories, consider using chunked processing
       
       See Also:
           calculate_rmsf: Calculate root-mean-square fluctuation
           align_structures: Perform structural alignment only
           
       References:
           .. [1] Kabsch, W. (1976). A solution for the best rotation to relate
                  two sets of vectors. Acta Crystallographica A32, 922-923.
           .. [2] Theobald, D.L. (2005). Rapid calculation of RMSDs using a
                  quaternion-based characteristic polynomial. Acta Crystallographica
                  A61, 478-480.
       """
       pass

Type Annotations
~~~~~~~~~~~~~~~

Use comprehensive type annotations for better API clarity:

.. code-block:: python

   from typing import Dict, List, Optional, Union, Tuple, Any, Callable
   from pathlib import Path
   import numpy as np
   from numpy.typing import NDArray

   # Type aliases for clarity
   Float1D = NDArray[np.float64]  # 1D float array
   Float2D = NDArray[np.float64]  # 2D float array  
   Float3D = NDArray[np.float64]  # 3D float array
   
   def run_analysis(
       trajectory_file: Union[str, Path],
       analysis_functions: List[Callable[[Float2D], Dict[str, Any]]],
       output_format: str = "json",
       chunk_size: Optional[int] = None,
       parallel: bool = False,
       progress_callback: Optional[Callable[[float], None]] = None
   ) -> Dict[str, List[Dict[str, Any]]]:
       """Run multiple analysis functions on trajectory data."""
       pass

Testing API Design
-----------------

API Testing Strategies
~~~~~~~~~~~~~~~~~~~~~

**Public API Tests**

Test the public API thoroughly:

.. code-block:: python

   def test_api_basic_usage():
       """Test basic API usage patterns."""
       # Test simple case
       system = MDSystem("test_system")
       system.load_structure("test.pdb")
       system.add_solvent()
       
       simulation = MDSimulation(system)
       simulation.run(steps=1000)
       
       assert system.n_atoms > 0
       assert simulation.current_step == 1000

   def test_api_error_handling():
       """Test API error handling."""
       system = MDSystem()
       
       # Should raise clear error for uninitialized system
       with pytest.raises(RuntimeError, match="positions not set"):
           simulation = MDSimulation(system)
           simulation.run(steps=100)

   def test_api_configuration():
       """Test API configuration options."""
       config = SimulationConfig(
           temperature=310.0,
           time_step=0.001,
           integrator="velocity_verlet"
       )
       
       system = MDSystem.from_pdb("test.pdb")
       simulation = MDSimulation(system, config)
       
       assert simulation.temperature == 310.0
       assert simulation.time_step == 0.001

**Backwards Compatibility Tests**

Ensure API changes don't break existing code:

.. code-block:: python

   def test_backwards_compatibility():
       """Test that old API still works."""
       # Old way should still work
       with warnings.catch_warnings(record=True) as w:
           result = old_function(parameter=5.0)
           assert len(w) == 1
           assert issubclass(w[0].category, DeprecationWarning)
       
       # New way should give same result
       new_result = new_function(parameter=5.0)
       assert result == new_result

Summary
-------

Good API design in ProteinMD follows these key principles:

1. **User-Centered**: Design for the scientist user, not the implementation
2. **Consistent**: Use patterns consistently across the codebase  
3. **Clear**: Names and behavior should be predictable
4. **Robust**: Handle errors gracefully with helpful messages
5. **Extensible**: Allow customization and extension through well-defined interfaces
6. **Well-Documented**: Comprehensive documentation with examples
7. **Tested**: Thorough testing of public APIs and error conditions

These principles ensure that ProteinMD remains easy to use while being powerful enough for advanced scientific computing needs.
