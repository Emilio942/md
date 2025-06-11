Coding Standards and Style Guide
================================

This document outlines the coding standards and style guidelines for ProteinMD development. Following these standards ensures code consistency, maintainability, and readability across the project.

.. contents:: Contents
   :local:
   :depth: 2

Python Code Style
-----------------

Base Standards
~~~~~~~~~~~~~

ProteinMD follows **PEP 8** as the base Python style guide, with specific extensions and clarifications outlined below.

**Automatic Formatting**

All Python code must be formatted with **Black** using the project configuration:

.. code-block:: bash

   # Format all Python files
   black proteinMD/ tests/ examples/
   
   # Check formatting
   black --check proteinMD/

**Line Length**

- **Maximum line length: 88 characters** (Black default)
- **Docstring line length: 79 characters** for better readability
- Long expressions should be broken using parentheses for natural grouping

.. code-block:: python

   # Good: Natural grouping with parentheses
   result = (
       some_long_function_name(parameter_one, parameter_two) +
       another_function(parameter_three, parameter_four)
   )
   
   # Good: Function calls with many parameters
   md_system = MDSystem(
       name="protein_simulation",
       temperature=300.0,
       pressure=1.0,
       time_step=0.002,
       integrator="verlet"
   )

**Imports**

Follow the standard import order with blank lines between groups:

.. code-block:: python

   # Standard library imports
   import os
   import sys
   from typing import Dict, List, Optional, Union, Any
   
   # Third-party imports
   import numpy as np
   import scipy.optimize
   import matplotlib.pyplot as plt
   
   # Local imports
   from proteinMD.core import MDSystem
   from proteinMD.forcefield import ForceField
   from .utils import calculate_distance

**Import Guidelines:**

- Use absolute imports for proteinMD modules
- Use relative imports only within the same package
- Avoid wildcard imports (``from module import *``)
- Group imports by type and alphabetize within groups

Naming Conventions
~~~~~~~~~~~~~~~~~

**Variables and Functions**
  Use ``snake_case`` for variables, functions, and module names:

.. code-block:: python

   # Variables
   atom_count = 1000
   simulation_time = 100.0  # ps
   
   # Functions
   def calculate_kinetic_energy(velocities: np.ndarray) -> float:
       """Calculate total kinetic energy."""
       return 0.5 * np.sum(velocities**2)
   
   def update_positions(positions, velocities, dt):
       """Update atomic positions using Verlet integration."""
       return positions + velocities * dt

**Classes**
  Use ``PascalCase`` for class names:

.. code-block:: python

   class MDSystem:
       """Molecular dynamics system container."""
       pass
   
   class ForceFieldParameter:
       """Individual force field parameter."""
       pass
   
   class TrajectoryAnalyzer:
       """Analyzer for MD trajectory data."""
       pass

**Constants**
  Use ``UPPER_CASE`` for module-level constants:

.. code-block:: python

   # Physical constants
   BOLTZMANN_CONSTANT = 8.314462618e-3  # kJ/(mol·K)
   AVOGADRO_NUMBER = 6.02214076e23
   
   # Default simulation parameters
   DEFAULT_TEMPERATURE = 300.0  # K
   DEFAULT_PRESSURE = 1.0       # atm
   DEFAULT_TIME_STEP = 0.002    # ps

**Private Attributes**
  Use single leading underscore for internal use:

.. code-block:: python

   class MDSystem:
       def __init__(self):
           self.public_attribute = "visible"
           self._internal_cache = {}
           self._atom_types = []

Type Hints and Annotations
~~~~~~~~~~~~~~~~~~~~~~~~~

**Required for Public APIs**

All public functions and methods must include type hints:

.. code-block:: python

   from typing import Dict, List, Optional, Union, Any, Tuple
   import numpy as np
   
   def run_simulation(
       system: MDSystem,
       steps: int,
       temperature: float = 300.0,
       output_file: Optional[str] = None
   ) -> Dict[str, Any]:
       """Run molecular dynamics simulation.
       
       Args:
           system: MD system to simulate
           steps: Number of simulation steps
           temperature: Simulation temperature in K
           output_file: Optional output trajectory file
           
       Returns:
           Dictionary containing simulation results
       """
       pass

**NumPy Array Types**

Use specific numpy array type hints where possible:

.. code-block:: python

   def calculate_distances(
       positions: np.ndarray,  # Shape: (n_atoms, 3)
       box_vectors: np.ndarray  # Shape: (3, 3)
   ) -> np.ndarray:  # Shape: (n_atoms, n_atoms)
       """Calculate pairwise distances with PBC."""
       pass

**Generic Types**

Use appropriate generic types for containers:

.. code-block:: python

   from typing import Dict, List, Optional, Union, Any, Callable
   
   def register_callback(
       event_type: str,
       callback: Callable[[Dict[str, Any]], None]
   ) -> None:
       """Register event callback function."""
       pass

Documentation Standards
-----------------------

Docstring Format
~~~~~~~~~~~~~~~

Use **Google-style docstrings** for all public functions, classes, and modules:

.. code-block:: python

   def calculate_rmsd(
       reference: np.ndarray,
       trajectory: np.ndarray,
       align: bool = True
   ) -> np.ndarray:
       """Calculate root-mean-square deviation between structures.
       
       This function computes the RMSD between a reference structure
       and each frame in a trajectory. Optional structural alignment
       can be performed to minimize rigid-body differences.
       
       Args:
           reference: Reference structure coordinates, shape (n_atoms, 3)
           trajectory: Trajectory coordinates, shape (n_frames, n_atoms, 3)
           align: Whether to perform structural alignment before RMSD
           
       Returns:
           RMSD values for each frame, shape (n_frames,)
           
       Raises:
           ValueError: If reference and trajectory have incompatible shapes
           
       Example:
           >>> ref = np.random.random((100, 3))
           >>> traj = np.random.random((1000, 100, 3))
           >>> rmsd = calculate_rmsd(ref, traj)
           >>> print(f"Average RMSD: {rmsd.mean():.2f} Å")
           Average RMSD: 1.23 Å
           
       Note:
           RMSD calculation assumes all atoms have equal weight.
           For mass-weighted RMSD, use calculate_rmsd_weighted().
       """
       pass

**Module Docstrings**

Every module should have a comprehensive docstring:

.. code-block:: python

   """Molecular dynamics force field implementations.
   
   This module provides classes and functions for working with molecular
   mechanics force fields, including parameter loading, force calculations,
   and energy evaluations.
   
   Classes:
       ForceField: Main force field container
       BondForce: Harmonic bond force implementation
       AngleForce: Harmonic angle force implementation
       DihedralForce: Torsional angle force implementation
       NonbondedForce: Van der Waals and electrostatic forces
       
   Functions:
       load_forcefield: Load force field from parameter files
       validate_parameters: Check force field parameter consistency
       
   Example:
       >>> from proteinMD.forcefield import ForceField
       >>> ff = ForceField.from_file("charmm36.prm")
       >>> energy = ff.calculate_energy(positions)
       
   References:
       [1] MacKerell et al. J. Phys. Chem. B 102, 3586-3616 (1998)
       [2] Huang & MacKerell. J. Comput. Chem. 34, 2135-2145 (2013)
   """

**Class Docstrings**

Classes should document their purpose, attributes, and usage:

.. code-block:: python

   class MDSystem:
       """Container for molecular dynamics simulation system.
       
       This class manages all components of an MD simulation including
       atomic positions, velocities, force field parameters, and
       simulation conditions.
       
       Attributes:
           name: Human-readable system name
           n_atoms: Number of atoms in the system
           positions: Current atomic positions, shape (n_atoms, 3)
           velocities: Current atomic velocities, shape (n_atoms, 3)
           forces: Current atomic forces, shape (n_atoms, 3)
           box_vectors: Simulation box vectors, shape (3, 3)
           
       Example:
           >>> system = MDSystem("protein_in_water")
           >>> system.load_structure("protein.pdb")
           >>> system.add_solvent(padding=1.0)
           >>> system.minimize_energy()
       """
       pass

Code Organization
-----------------

File and Module Structure
~~~~~~~~~~~~~~~~~~~~~~~~

**Package Layout**

.. code-block:: text

   proteinMD/
   ├── __init__.py          # Package initialization and version
   ├── core/                # Core MD simulation engine
   │   ├── __init__.py
   │   ├── simulation.py    # Main simulation class
   │   ├── integrators.py   # Numerical integrators
   │   └── thermostats.py   # Temperature coupling
   ├── forcefield/          # Force field implementations
   │   ├── __init__.py
   │   ├── base.py         # Base force field classes
   │   ├── charmm.py       # CHARMM force fields
   │   └── amber.py        # AMBER force fields
   ├── analysis/            # Analysis tools
   │   ├── __init__.py
   │   ├── structural.py   # Structural analysis
   │   └── thermodynamic.py # Thermodynamic properties
   ├── utils/               # Utility functions
   │   ├── __init__.py
   │   ├── math.py         # Mathematical utilities
   │   └── io.py           # File I/O utilities
   └── visualization/       # Plotting and visualization
       ├── __init__.py
       └── plotting.py

**Import Guidelines**

Each ``__init__.py`` should expose the public API:

.. code-block:: python

   # proteinMD/core/__init__.py
   """Core molecular dynamics simulation components."""
   
   from .simulation import MDSimulation
   from .integrators import VerletIntegrator, LeapFrogIntegrator
   from .thermostats import BerendsenThermostat, NoseHooverThermostat
   
   __all__ = [
       "MDSimulation",
       "VerletIntegrator", 
       "LeapFrogIntegrator",
       "BerendsenThermostat",
       "NoseHooverThermostat"
   ]

Function and Class Design
~~~~~~~~~~~~~~~~~~~~~~~~

**Function Design Principles**

1. **Single Responsibility**: Each function should do one thing well
2. **Pure Functions**: Prefer functions without side effects when possible
3. **Clear Interfaces**: Use type hints and descriptive names
4. **Error Handling**: Validate inputs and provide informative error messages

.. code-block:: python

   def calculate_potential_energy(
       positions: np.ndarray,
       force_field: ForceField,
       box_vectors: Optional[np.ndarray] = None
   ) -> float:
       """Calculate total potential energy of the system.
       
       Args:
           positions: Atomic coordinates, shape (n_atoms, 3)
           force_field: Force field for energy calculation
           box_vectors: Periodic box vectors for PBC, shape (3, 3)
           
       Returns:
           Total potential energy in kJ/mol
           
       Raises:
           ValueError: If positions array has wrong shape
           TypeError: If force_field is not a ForceField instance
       """
       # Input validation
       if positions.ndim != 2 or positions.shape[1] != 3:
           raise ValueError("Positions must have shape (n_atoms, 3)")
       
       if not isinstance(force_field, ForceField):
           raise TypeError("force_field must be ForceField instance")
       
       # Calculation
       return force_field.calculate_energy(positions, box_vectors)

**Class Design Principles**

1. **Composition over Inheritance**: Prefer composition when possible
2. **Clear Interfaces**: Define abstract base classes for pluggable components
3. **Immutability**: Make objects immutable when possible
4. **Resource Management**: Use context managers for resource cleanup

.. code-block:: python

   from abc import ABC, abstractmethod
   
   class Integrator(ABC):
       """Abstract base class for MD integrators."""
       
       @abstractmethod
       def step(
           self,
           positions: np.ndarray,
           velocities: np.ndarray,
           forces: np.ndarray,
           dt: float
       ) -> Tuple[np.ndarray, np.ndarray]:
           """Perform one integration step.
           
           Args:
               positions: Current positions
               velocities: Current velocities  
               forces: Current forces
               dt: Time step
               
           Returns:
               Updated (positions, velocities)
           """
           pass

Error Handling
~~~~~~~~~~~~~

**Exception Guidelines**

1. **Use specific exception types** rather than generic Exception
2. **Provide informative error messages** with context
3. **Validate inputs early** in public functions
4. **Document expected exceptions** in docstrings

.. code-block:: python

   class SimulationError(Exception):
       """Base exception for simulation-related errors."""
       pass
   
   class ConvergenceError(SimulationError):
       """Raised when simulation fails to converge."""
       pass
   
   class InvalidParameterError(SimulationError):
       """Raised when invalid parameters are provided."""
       pass
   
   def run_minimization(
       system: MDSystem,
       max_iterations: int = 1000,
       tolerance: float = 1e-6
   ) -> None:
       """Run energy minimization.
       
       Raises:
           InvalidParameterError: If max_iterations <= 0 or tolerance <= 0
           ConvergenceError: If minimization fails to converge
       """
       if max_iterations <= 0:
           raise InvalidParameterError("max_iterations must be positive")
       
       if tolerance <= 0:
           raise InvalidParameterError("tolerance must be positive")
       
       # Run minimization
       for i in range(max_iterations):
           gradient = system.calculate_gradient()
           if np.linalg.norm(gradient) < tolerance:
               return  # Converged
       
       raise ConvergenceError(
           f"Minimization failed to converge after {max_iterations} iterations"
       )

Performance Guidelines
---------------------

NumPy Best Practices
~~~~~~~~~~~~~~~~~~~

**Vectorization**

Prefer NumPy vectorized operations over Python loops:

.. code-block:: python

   # Bad: Python loops
   def calculate_distances_slow(pos1, pos2):
       distances = []
       for i in range(len(pos1)):
           dx = pos1[i, 0] - pos2[i, 0]
           dy = pos1[i, 1] - pos2[i, 1] 
           dz = pos1[i, 2] - pos2[i, 2]
           distances.append(np.sqrt(dx**2 + dy**2 + dz**2))
       return np.array(distances)
   
   # Good: Vectorized operations
   def calculate_distances_fast(pos1, pos2):
       diff = pos1 - pos2
       return np.linalg.norm(diff, axis=1)

**Memory Efficiency**

- Use in-place operations when possible
- Avoid unnecessary array copies
- Use appropriate data types (float32 vs float64)

.. code-block:: python

   # Good: In-place operations
   def update_velocities_inplace(
       velocities: np.ndarray,
       forces: np.ndarray,
       masses: np.ndarray,
       dt: float
   ) -> None:
       """Update velocities in-place using F = ma."""
       velocities += (forces / masses[:, np.newaxis]) * dt
   
   # Good: Specify data type for memory efficiency
   positions = np.zeros((n_atoms, 3), dtype=np.float32)

Algorithm Complexity
~~~~~~~~~~~~~~~~~~~

**Document Complexity**

Include time and space complexity in docstrings for non-trivial algorithms:

.. code-block:: python

   def calculate_pairwise_distances(positions: np.ndarray) -> np.ndarray:
       """Calculate all pairwise distances.
       
       Time complexity: O(n²) where n is the number of atoms
       Space complexity: O(n²) for the distance matrix
       
       For large systems (>10,000 atoms), consider using
       spatial data structures or cutoff schemes.
       """
       n_atoms = len(positions)
       distances = np.zeros((n_atoms, n_atoms))
       
       for i in range(n_atoms):
           for j in range(i+1, n_atoms):
               dist = np.linalg.norm(positions[i] - positions[j])
               distances[i, j] = dist
               distances[j, i] = dist
       
       return distances

Testing Guidelines
-----------------

Test Structure
~~~~~~~~~~~~~

**Test File Organization**

.. code-block:: text

   tests/
   ├── test_core/
   │   ├── test_simulation.py
   │   ├── test_integrators.py
   │   └── test_thermostats.py
   ├── test_forcefield/
   │   ├── test_charmm.py
   │   └── test_amber.py
   ├── test_analysis/
   │   └── test_structural.py
   └── conftest.py  # Shared fixtures

**Test Naming**

Use descriptive test names that explain what is being tested:

.. code-block:: python

   def test_verlet_integrator_conserves_energy_for_harmonic_oscillator():
       """Test that Verlet integration conserves energy for harmonic motion."""
       pass
   
   def test_rmsd_calculation_returns_zero_for_identical_structures():
       """Test RMSD returns 0.0 when comparing identical structures."""
       pass
   
   def test_force_field_raises_error_for_invalid_atom_types():
       """Test that invalid atom types raise appropriate exceptions."""
       pass

**Test Coverage**

- Aim for >90% test coverage
- Test both success and failure cases
- Include edge cases and boundary conditions
- Use property-based testing for numerical functions

Git and Version Control
----------------------

Commit Guidelines
~~~~~~~~~~~~~~~~

**Commit Message Format**

Use conventional commit format:

.. code-block:: text

   <type>[optional scope]: <description>
   
   [optional body]
   
   [optional footer(s)]

**Types:**
- ``feat``: New features
- ``fix``: Bug fixes  
- ``docs``: Documentation updates
- ``style``: Code style changes
- ``refactor``: Code refactoring
- ``test``: Test additions/modifications
- ``perf``: Performance improvements
- ``chore``: Maintenance tasks

**Examples:**

.. code-block:: text

   feat(core): add velocity Verlet integrator
   
   Implement velocity Verlet integration algorithm for improved
   energy conservation in MD simulations.
   
   Closes #123

.. code-block:: text

   fix(forcefield): correct bond energy calculation
   
   Fixed sign error in harmonic bond energy calculation that was
   causing incorrect energies for stretched bonds.
   
   Breaking change: Energy values will be different for systems
   with significantly stretched bonds.

**Branch Naming**

Use descriptive branch names:

.. code-block:: text

   feature/velocity-verlet-integrator
   bugfix/bond-energy-calculation
   docs/api-documentation-update
   refactor/forcefield-module-structure

Code Review Standards
--------------------

Review Checklist
~~~~~~~~~~~~~~~

**Before Submitting PR:**

- [ ] Code follows style guidelines (Black formatting)
- [ ] All tests pass locally
- [ ] New code has appropriate test coverage
- [ ] Documentation is updated for public APIs
- [ ] Commit messages follow conventional format
- [ ] No debugging code or commented-out sections
- [ ] Performance impact considered and documented

**Reviewer Guidelines:**

- [ ] Code is readable and well-documented
- [ ] Logic is correct and efficient
- [ ] Error handling is appropriate
- [ ] Tests adequately cover new functionality
- [ ] API design follows project conventions
- [ ] Breaking changes are clearly documented
- [ ] Security implications considered

Quality Assurance Tools
----------------------

Automated Checks
~~~~~~~~~~~~~~~

**Pre-commit Hooks**

Install and configure pre-commit hooks:

.. code-block:: bash

   # Install pre-commit
   pip install pre-commit
   
   # Install hooks
   pre-commit install
   
   # Run manually
   pre-commit run --all-files

**Continuous Integration**

All code must pass CI checks:

- **Linting**: Flake8 compliance
- **Formatting**: Black compliance
- **Type Checking**: MyPy validation
- **Testing**: Pytest with coverage
- **Documentation**: Sphinx build success

**Local Development Tools**

.. code-block:: bash

   # Format code
   black proteinMD/ tests/
   
   # Check linting
   flake8 proteinMD/ tests/
   
   # Type checking
   mypy proteinMD/
   
   # Run tests with coverage
   pytest --cov=proteinMD --cov-report=html

Summary
-------

Following these coding standards ensures:

- **Consistency**: All code looks and behaves similarly
- **Maintainability**: Code is easy to understand and modify
- **Quality**: Fewer bugs and better performance
- **Collaboration**: Easier code reviews and contributions

For questions about these standards or suggestions for improvements, please open an issue or discussion on GitHub.
