Extending ProteinMD
==================

This guide covers how to extend ProteinMD with custom force fields, analysis methods, and integrators.

.. contents:: Extension Topics
   :local:
   :depth: 2

Custom Force Fields
------------------

Creating Force Field Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Basic Force Field Structure**

.. code-block:: python

   from proteinmd.forcefield.base import ForceField
   from proteinmd.forcefield.parameters import BondParameters, AngleParameters
   
   class CustomForceField(ForceField):
       """Custom force field implementation."""
       
       def __init__(self):
           super().__init__(name="CustomFF")
           self.version = "1.0"
           self.citations = ["Custom et al. (2024)"]
           
           # Load parameters
           self._load_parameters()
       
       def _load_parameters(self):
           """Load force field parameters."""
           # Bond parameters
           self.bond_params = BondParameters()
           self.bond_params.add_parameter(
               atom_types=["CT", "HC"],
               k=2845.12,  # kJ/mol/nm²
               r0=0.1090   # nm
           )
           
           # Angle parameters
           self.angle_params = AngleParameters()
           self.angle_params.add_parameter(
               atom_types=["HC", "CT", "HC"],
               k=276.144,  # kJ/mol/rad²
               theta0=109.5  # degrees
           )
           
           # Add more parameter types as needed
       
       def assign_parameters(self, system):
           """Assign parameters to system atoms."""
           for atom in system.atoms:
               # Assign atom types based on chemical environment
               atom_type = self._determine_atom_type(atom)
               atom.set_type(atom_type)
               
               # Assign charges
               charge = self._calculate_charge(atom)
               atom.set_charge(charge)
           
           # Assign bonded parameters
           self._assign_bonds(system)
           self._assign_angles(system)
           self._assign_dihedrals(system)
       
       def _determine_atom_type(self, atom):
           """Determine atom type based on chemical environment."""
           # Custom logic for atom typing
           if atom.element == "C":
               if len(atom.bonds) == 4:
                   return "CT"  # sp3 carbon
               elif len(atom.bonds) == 3:
                   return "CA"  # aromatic carbon
           elif atom.element == "H":
               if atom.bonded_to.element == "C":
                   return "HC"  # hydrogen on carbon
           
           # Default fallback
           return f"{atom.element}X"

**Advanced Parameter Assignment**

.. code-block:: python

   def _assign_bonds(self, system):
       """Assign bond parameters."""
       for bond in system.bonds:
           atom1_type = bond.atom1.type
           atom2_type = bond.atom2.type
           
           # Look up parameters
           params = self.bond_params.get_parameters(atom1_type, atom2_type)
           
           if params:
               bond.set_parameters(k=params.k, r0=params.r0)
           else:
               # Handle missing parameters
               self._handle_missing_bond_params(bond)
   
   def _handle_missing_bond_params(self, bond):
       """Handle missing bond parameters."""
       # Option 1: Use default parameters
       default_k = 2000.0  # kJ/mol/nm²
       default_r0 = 0.150  # nm
       
       # Option 2: Estimate from similar bonds
       similar_bonds = self._find_similar_bonds(bond)
       if similar_bonds:
           avg_k = sum(b.k for b in similar_bonds) / len(similar_bonds)
           avg_r0 = sum(b.r0 for b in similar_bonds) / len(similar_bonds)
           bond.set_parameters(k=avg_k, r0=avg_r0)
       else:
           bond.set_parameters(k=default_k, r0=default_r0)
       
       # Log warning
       self.logger.warning(f"Missing bond parameters for {bond.atom1.type}-{bond.atom2.type}")

Custom Integrators
-----------------

Implementing New Integration Schemes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Base Integrator Structure**

.. code-block:: python

   from proteinmd.core.integrators.base import Integrator
   import numpy as np
   
   class CustomIntegrator(Integrator):
       """Custom molecular dynamics integrator."""
       
       def __init__(self, timestep=0.001, **kwargs):
           super().__init__(timestep=timestep)
           self.name = "CustomIntegrator"
           
           # Custom parameters
           self.parameter1 = kwargs.get('parameter1', 1.0)
           self.parameter2 = kwargs.get('parameter2', 0.5)
       
       def step(self, system, forces):
           """Perform one integration step."""
           dt = self.timestep
           
           # Get current state
           positions = system.get_positions()
           velocities = system.get_velocities()
           masses = system.get_masses()
           
           # Custom integration algorithm
           new_positions, new_velocities = self._integrate_step(
               positions, velocities, forces, masses, dt
           )
           
           # Update system state
           system.set_positions(new_positions)
           system.set_velocities(new_velocities)
           
           # Update time
           self.current_time += dt
       
       def _integrate_step(self, pos, vel, forces, masses, dt):
           """Custom integration algorithm implementation."""
           # Example: Modified velocity-Verlet
           accelerations = forces / masses[:, np.newaxis]
           
           # Position update with custom modification
           new_pos = pos + vel * dt + 0.5 * accelerations * dt**2 * self.parameter1
           
           # Calculate forces at new positions (would be done by simulation)
           # new_forces = system.calculate_forces(new_pos)
           # new_accelerations = new_forces / masses[:, np.newaxis]
           
           # Velocity update with custom modification
           # new_vel = vel + 0.5 * (accelerations + new_accelerations) * dt * self.parameter2
           
           # For this example, use simple update
           new_vel = vel + accelerations * dt * self.parameter2
           
           return new_pos, new_vel

**Specialized Integrators**

.. code-block:: python

   class LangevinMiddleIntegrator(Integrator):
       """Langevin integrator with middle scheme."""
       
       def __init__(self, timestep, temperature, friction):
           super().__init__(timestep)
           self.temperature = temperature
           self.friction = friction
           self.kB = 0.008314462618  # kJ/mol/K
           
           # Pre-calculate constants
           self.gamma = friction
           self.sigma = np.sqrt(2 * self.gamma * self.kB * temperature)
       
       def step(self, system, forces):
           """Langevin middle scheme integration."""
           dt = self.timestep
           positions = system.get_positions()
           velocities = system.get_velocities()
           masses = system.get_masses()
           
           # Random forces for each particle
           random_forces = np.random.normal(
               0, self.sigma * np.sqrt(dt), velocities.shape
           ) * np.sqrt(masses[:, np.newaxis])
           
           # Langevin integration
           accelerations = forces / masses[:, np.newaxis]
           
           # Update velocities (half step)
           velocities += 0.5 * accelerations * dt
           
           # Apply friction and random forces
           c1 = np.exp(-self.gamma * dt)
           c2 = np.sqrt(1 - c1**2)
           
           velocities = c1 * velocities + c2 * random_forces / masses[:, np.newaxis]
           
           # Update positions
           positions += velocities * dt
           
           # Update velocities (second half step) 
           # Note: would need new forces here
           velocities += 0.5 * accelerations * dt
           
           system.set_positions(positions)
           system.set_velocities(velocities)

Custom Analysis Methods
----------------------

Creating Analysis Classes
~~~~~~~~~~~~~~~~~~~~~~~~~

**Base Analysis Framework**

.. code-block:: python

   from proteinmd.analysis.base import AnalysisMethod
   import numpy as np
   
   class CustomAnalysis(AnalysisMethod):
       """Custom analysis method for protein dynamics."""
       
       def __init__(self, **kwargs):
           super().__init__(name="CustomAnalysis")
           
           # Analysis parameters
           self.parameter1 = kwargs.get('parameter1', 1.0)
           self.cutoff = kwargs.get('cutoff', 0.5)
           
           # Results storage
           self.results = {}
       
       def calculate(self, trajectory, **kwargs):
           """Main calculation method."""
           # Initialize results
           self.results = {
               'values': [],
               'metadata': {
                   'n_frames': len(trajectory),
                   'parameters': {
                       'parameter1': self.parameter1,
                       'cutoff': self.cutoff
                   }
               }
           }
           
           # Process each frame
           for i, frame in enumerate(trajectory):
               value = self._analyze_frame(frame)
               self.results['values'].append(value)
               
               # Progress reporting
               if i % 100 == 0:
                   progress = (i + 1) / len(trajectory) * 100
                   print(f"Analysis progress: {progress:.1f}%")
           
           # Post-process results
           self._post_process()
           
           return self.results
       
       def _analyze_frame(self, frame):
           """Analyze individual frame."""
           # Example: Calculate custom geometric property
           coordinates = frame.get_coordinates()
           
           # Custom calculation
           custom_value = self._custom_calculation(coordinates)
           
           return custom_value
       
       def _custom_calculation(self, coordinates):
           """Implement custom calculation."""
           # Example: Average distance from center
           center = np.mean(coordinates, axis=0)
           distances = np.linalg.norm(coordinates - center, axis=1)
           
           # Apply custom logic
           filtered_distances = distances[distances < self.cutoff]
           
           if len(filtered_distances) > 0:
               return np.mean(filtered_distances) * self.parameter1
           else:
               return 0.0
       
       def _post_process(self):
           """Post-process analysis results."""
           values = np.array(self.results['values'])
           
           # Calculate statistics
           self.results['statistics'] = {
               'mean': np.mean(values),
               'std': np.std(values),
               'min': np.min(values),
               'max': np.max(values)
           }
           
           # Calculate time series properties
           self.results['time_series'] = {
               'trend': self._calculate_trend(values),
               'autocorrelation': self._calculate_autocorrelation(values)
           }

**Advanced Analysis Example**

.. code-block:: python

   class ProteinFlexibilityAnalysis(AnalysisMethod):
       """Analyze protein flexibility using multiple metrics."""
       
       def __init__(self, selection="protein", window_size=100):
           super().__init__(name="ProteinFlexibilityAnalysis")
           self.selection = selection
           self.window_size = window_size
       
       def calculate(self, trajectory, **kwargs):
           """Calculate multiple flexibility metrics."""
           # Get selected atoms
           selected_atoms = self._get_selected_atoms(trajectory[0])
           
           results = {
               'rmsf': self._calculate_rmsf(trajectory, selected_atoms),
               'dynamic_correlation': self._calculate_correlation(trajectory, selected_atoms),
               'flexibility_profile': self._calculate_flexibility_profile(trajectory, selected_atoms),
               'hinge_regions': self._identify_hinge_regions(trajectory, selected_atoms)
           }
           
           return results
       
       def _calculate_rmsf(self, trajectory, atoms):
           """Calculate root mean square fluctuations."""
           n_atoms = len(atoms)
           n_frames = len(trajectory)
           
           # Extract coordinates for selected atoms
           coords = np.zeros((n_frames, n_atoms, 3))
           for i, frame in enumerate(trajectory):
               coords[i] = frame.get_coordinates()[atoms]
           
           # Align structures to remove translation/rotation
           aligned_coords = self._align_coordinates(coords)
           
           # Calculate average structure
           avg_coords = np.mean(aligned_coords, axis=0)
           
           # Calculate RMSF
           rmsf = np.sqrt(np.mean(
               np.sum((aligned_coords - avg_coords)**2, axis=2), axis=0
           ))
           
           return rmsf
       
       def _calculate_correlation(self, trajectory, atoms):
           """Calculate dynamic cross-correlation matrix."""
           coords = self._extract_coordinates(trajectory, atoms)
           aligned_coords = self._align_coordinates(coords)
           
           # Calculate fluctuations
           avg_coords = np.mean(aligned_coords, axis=0)
           fluctuations = aligned_coords - avg_coords
           
           # Flatten to (n_frames, n_atoms * 3)
           flat_fluct = fluctuations.reshape(len(trajectory), -1)
           
           # Calculate correlation matrix
           correlation_matrix = np.corrcoef(flat_fluct.T)
           
           return correlation_matrix

Custom Sampling Methods
----------------------

Enhanced Sampling Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Custom Biasing Potential**

.. code-block:: python

   from proteinmd.sampling.base import BiasingMethod
   
   class CustomBias(BiasingMethod):
       """Custom biasing potential for enhanced sampling."""
       
       def __init__(self, collective_variables, bias_strength=1.0):
           super().__init__(name="CustomBias")
           self.collective_variables = collective_variables
           self.bias_strength = bias_strength
           
           # Bias history
           self.bias_history = []
       
       def calculate_bias(self, system):
           """Calculate biasing force and energy."""
           # Evaluate collective variables
           cv_values = []
           cv_derivatives = []
           
           for cv in self.collective_variables:
               value, derivative = cv.evaluate(system)
               cv_values.append(value)
               cv_derivatives.append(derivative)
           
           # Calculate bias potential
           bias_energy = self._calculate_bias_energy(cv_values)
           bias_forces = self._calculate_bias_forces(cv_values, cv_derivatives)
           
           # Store history
           self.bias_history.append({
               'cv_values': cv_values,
               'bias_energy': bias_energy,
               'time': system.get_time()
           })
           
           return bias_energy, bias_forces
       
       def _calculate_bias_energy(self, cv_values):
           """Calculate bias potential energy."""
           # Example: Harmonic bias towards target values
           target_values = [0.0, 1.0]  # Target CV values
           
           bias_energy = 0.0
           for i, (cv_val, target) in enumerate(zip(cv_values, target_values)):
               bias_energy += 0.5 * self.bias_strength * (cv_val - target)**2
           
           return bias_energy
       
       def _calculate_bias_forces(self, cv_values, cv_derivatives):
           """Calculate bias forces."""
           target_values = [0.0, 1.0]
           
           bias_forces = np.zeros_like(cv_derivatives[0])
           
           for i, (cv_val, target, cv_deriv) in enumerate(
               zip(cv_values, target_values, cv_derivatives)
           ):
               force_magnitude = -self.bias_strength * (cv_val - target)
               bias_forces += force_magnitude * cv_deriv
           
           return bias_forces

**Adaptive Sampling Method**

.. code-block:: python

   class AdaptiveSampling(BiasingMethod):
       """Adaptive sampling method that adjusts bias based on sampling history."""
       
       def __init__(self, collective_variables, adaptation_rate=0.1):
           super().__init__(name="AdaptiveSampling")
           self.collective_variables = collective_variables
           self.adaptation_rate = adaptation_rate
           
           # Sampling statistics
           self.cv_histogram = {}
           self.bias_potential = {}
           self.update_frequency = 1000  # Update every 1000 steps
           self.step_count = 0
       
       def calculate_bias(self, system):
           """Calculate adaptive bias."""
           self.step_count += 1
           
           # Evaluate collective variables
           cv_values = tuple(cv.evaluate(system)[0] for cv in self.collective_variables)
           
           # Update sampling histogram
           self._update_histogram(cv_values)
           
           # Update bias potential periodically
           if self.step_count % self.update_frequency == 0:
               self._update_bias_potential()
           
           # Calculate current bias
           bias_energy = self._get_bias_energy(cv_values)
           bias_forces = self._get_bias_forces(system, cv_values)
           
           return bias_energy, bias_forces
       
       def _update_histogram(self, cv_values):
           """Update sampling histogram."""
           # Discretize CV values
           discretized = tuple(round(val, 2) for val in cv_values)
           
           if discretized in self.cv_histogram:
               self.cv_histogram[discretized] += 1
           else:
               self.cv_histogram[discretized] = 1
       
       def _update_bias_potential(self):
           """Update bias potential based on sampling frequency."""
           if not self.cv_histogram:
               return
           
           # Calculate sampling probabilities
           total_counts = sum(self.cv_histogram.values())
           
           for cv_point, count in self.cv_histogram.items():
               probability = count / total_counts
               
               # Apply bias to under-sampled regions
               if probability > 0:
                   bias_value = -self.adaptation_rate * np.log(probability)
                   self.bias_potential[cv_point] = bias_value

Plugin System
------------

Creating Plugin Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Plugin Base Class**

.. code-block:: python

   from abc import ABC, abstractmethod
   
   class ProteinMDPlugin(ABC):
       """Base class for ProteinMD plugins."""
       
       def __init__(self, name, version="1.0"):
           self.name = name
           self.version = version
           self.description = ""
           self.author = ""
           self.dependencies = []
       
       @abstractmethod
       def initialize(self, proteinmd_instance):
           """Initialize plugin with ProteinMD instance."""
           pass
       
       @abstractmethod
       def register_components(self, registry):
           """Register plugin components with ProteinMD."""
           pass
       
       def check_dependencies(self):
           """Check if plugin dependencies are satisfied."""
           missing_deps = []
           for dep in self.dependencies:
               try:
                   __import__(dep)
               except ImportError:
                   missing_deps.append(dep)
           
           return missing_deps

**Example Plugin Implementation**

.. code-block:: python

   class CustomAnalysisPlugin(ProteinMDPlugin):
       """Plugin providing custom analysis methods."""
       
       def __init__(self):
           super().__init__(name="CustomAnalysisPlugin", version="1.0")
           self.description = "Custom analysis methods for specialized simulations"
           self.author = "Your Name"
           self.dependencies = ["scipy", "sklearn"]
       
       def initialize(self, proteinmd_instance):
           """Initialize plugin."""
           # Check dependencies
           missing = self.check_dependencies()
           if missing:
               raise ImportError(f"Missing dependencies: {missing}")
           
           # Initialize plugin-specific resources
           self.proteinmd = proteinmd_instance
           self._setup_analysis_methods()
       
       def register_components(self, registry):
           """Register analysis methods."""
           # Register custom analysis classes
           registry.register_analysis("custom_flexibility", CustomFlexibilityAnalysis)
           registry.register_analysis("advanced_correlation", AdvancedCorrelationAnalysis)
           registry.register_analysis("machine_learning_clustering", MLClusteringAnalysis)
           
           # Register custom force fields
           registry.register_forcefield("custom_ff", CustomForceField)
           
           # Register custom integrators
           registry.register_integrator("custom_langevin", CustomLangevinIntegrator)
       
       def _setup_analysis_methods(self):
           """Set up plugin-specific analysis methods."""
           # Initialize any plugin-specific resources
           pass

**Plugin Manager**

.. code-block:: python

   class PluginManager:
       """Manages ProteinMD plugins."""
       
       def __init__(self):
           self.plugins = {}
           self.registry = ComponentRegistry()
       
       def load_plugin(self, plugin_class):
           """Load and initialize a plugin."""
           plugin = plugin_class()
           
           # Check dependencies
           missing_deps = plugin.check_dependencies()
           if missing_deps:
               raise ImportError(f"Plugin {plugin.name} missing dependencies: {missing_deps}")
           
           # Initialize plugin
           plugin.initialize(self)
           
           # Register components
           plugin.register_components(self.registry)
           
           # Store plugin
           self.plugins[plugin.name] = plugin
           
           print(f"Loaded plugin: {plugin.name} v{plugin.version}")
       
       def get_available_components(self, component_type):
           """Get available components of specified type."""
           return self.registry.get_components(component_type)
       
       def create_component(self, component_type, component_name, **kwargs):
           """Create component instance."""
           return self.registry.create_component(component_type, component_name, **kwargs)

Testing Extensions
-----------------

Unit Testing for Custom Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Test Framework**

.. code-block:: python

   import unittest
   from proteinmd.testing import TestCase, create_test_system
   
   class TestCustomForceField(TestCase):
       """Test custom force field implementation."""
       
       def setUp(self):
           """Set up test fixtures."""
           self.forcefield = CustomForceField()
           self.test_system = create_test_system("small_protein")
       
       def test_parameter_assignment(self):
           """Test parameter assignment."""
           # Apply force field
           self.forcefield.assign_parameters(self.test_system)
           
           # Check that all atoms have types assigned
           for atom in self.test_system.atoms:
               self.assertIsNotNone(atom.type)
               self.assertIsNotNone(atom.charge)
       
       def test_energy_calculation(self):
           """Test energy calculation."""
           self.forcefield.assign_parameters(self.test_system)
           
           # Calculate energy
           energy = self.test_system.calculate_potential_energy()
           
           # Check reasonable energy range
           energy_per_atom = energy / len(self.test_system.atoms)
           self.assertLess(abs(energy_per_atom), 1000.0)  # kJ/mol
       
       def test_force_calculation(self):
           """Test force calculation."""
           self.forcefield.assign_parameters(self.test_system)
           
           # Calculate forces
           forces = self.test_system.calculate_forces()
           
           # Check force dimensions
           self.assertEqual(forces.shape, (len(self.test_system.atoms), 3))
           
           # Check that forces are finite
           self.assertTrue(np.all(np.isfinite(forces)))

**Integration Testing**

.. code-block:: python

   class TestCustomIntegrator(TestCase):
       """Test custom integrator implementation."""
       
       def setUp(self):
           self.integrator = CustomIntegrator(timestep=0.001)
           self.system = create_test_system("water_box")
           self.simulation = MDSimulation(
               system=self.system,
               integrator=self.integrator
           )
       
       def test_energy_conservation(self):
           """Test energy conservation in NVE simulation."""
           # Run short NVE simulation
           initial_energy = self.system.get_total_energy()
           
           self.simulation.run(steps=1000)
           
           final_energy = self.system.get_total_energy()
           energy_drift = abs(final_energy - initial_energy)
           
           # Check energy conservation (allow small drift)
           self.assertLess(energy_drift, 10.0)  # kJ/mol
       
       def test_temperature_control(self):
           """Test temperature control with thermostat."""
           from proteinmd.core import LangevinThermostat
           
           thermostat = LangevinThermostat(temperature=300.0, friction=1.0)
           self.simulation.add_thermostat(thermostat)
           
           # Equilibrate
           self.simulation.run(steps=5000)
           
           # Check temperature
           temperature = self.system.get_temperature()
           self.assertAlmostEqual(temperature, 300.0, delta=10.0)

Documentation for Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Documenting Custom Components**

.. code-block:: python

   class CustomAnalysisMethod(AnalysisMethod):
       """
       Custom analysis method for protein dynamics.
       
       This analysis method calculates a custom property that combines
       structural and dynamical information to characterize protein behavior.
       
       Parameters
       ----------
       parameter1 : float, default=1.0
           Scaling parameter for the calculation.
       cutoff : float, default=0.5
           Distance cutoff in nanometers.
       selection : str, default="protein"
           Atom selection for analysis.
       
       Attributes
       ----------
       results : dict
           Dictionary containing analysis results.
       
       Examples
       --------
       >>> from proteinmd.analysis.custom import CustomAnalysisMethod
       >>> analyzer = CustomAnalysisMethod(parameter1=2.0, cutoff=0.8)
       >>> results = analyzer.calculate(trajectory)
       >>> print(f"Average value: {results['statistics']['mean']:.3f}")
       
       Notes
       -----
       This method implements the algorithm described in [1]_.
       
       References
       ----------
       .. [1] Author et al. "Custom Analysis Method." Journal (2024).
       """

Distribution and Packaging
-------------------------

Creating Installable Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Setup Script**

.. code-block:: python

   # setup.py
   from setuptools import setup, find_packages
   
   setup(
       name="proteinmd-custom-extensions",
       version="1.0.0",
       description="Custom extensions for ProteinMD",
       author="Your Name",
       author_email="your.email@example.com",
       packages=find_packages(),
       install_requires=[
           "proteinmd>=1.0.0",
           "numpy>=1.20.0",
           "scipy>=1.7.0"
       ],
       entry_points={
           'proteinmd.plugins': [
               'custom_analysis = proteinmd_custom.plugins:CustomAnalysisPlugin',
               'custom_forcefield = proteinmd_custom.plugins:CustomForceFieldPlugin'
           ]
       },
       classifiers=[
           "Development Status :: 4 - Beta",
           "Intended Audience :: Science/Research",
           "License :: OSI Approved :: MIT License",
           "Programming Language :: Python :: 3.8",
           "Programming Language :: Python :: 3.9",
           "Programming Language :: Python :: 3.10"
       ]
   )

See Also
--------

* :doc:`../api/index` - Core API reference
* :doc:`../developer/contributing` - Contributing guidelines
* :doc:`../developer/testing` - Testing framework
* :doc:`../user_guide/examples` - Usage examples
