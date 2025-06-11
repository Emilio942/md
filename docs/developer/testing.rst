Testing Framework
=================

This document describes the testing framework and best practices for ProteinMD.

.. contents:: Testing Topics
   :local:
   :depth: 2

Test Organization
----------------

Test Structure
~~~~~~~~~~~~~

**Directory Layout**

.. code-block:: text

   tests/
   ├── conftest.py              # pytest configuration and fixtures
   ├── unit/                    # Unit tests
   │   ├── __init__.py
   │   ├── test_core/
   │   │   ├── test_integrators.py
   │   │   ├── test_thermostats.py
   │   │   └── test_simulation.py
   │   ├── test_structure/
   │   │   ├── test_protein.py
   │   │   ├── test_validation.py
   │   │   └── test_io.py
   │   ├── test_forcefield/
   │   │   ├── test_amber.py
   │   │   ├── test_charmm.py
   │   │   └── test_parameters.py
   │   ├── test_analysis/
   │   │   ├── test_rmsd.py
   │   │   ├── test_ramachandran.py
   │   │   └── test_clustering.py
   │   └── test_utils/
   │       ├── test_math.py
   │       └── test_io.py
   ├── integration/             # Integration tests
   │   ├── test_workflows/
   │   │   ├── test_basic_md.py
   │   │   ├── test_free_energy.py
   │   │   └── test_analysis_pipeline.py
   │   └── test_backends/
   │       ├── test_openmm.py
   │       └── test_gromacs.py
   ├── regression/              # Regression tests
   │   ├── test_numerical/
   │   │   ├── test_energy_conservation.py
   │   │   └── test_trajectory_comparison.py
   │   └── reference_data/
   │       ├── energy_values.json
   │       └── trajectory_checksums.json
   ├── performance/             # Performance benchmarks
   │   ├── test_benchmarks.py
   │   └── profile_analysis.py
   └── data/                    # Test data
       ├── structures/
       │   ├── small_protein.pdb
       │   ├── water_box.pdb
       │   └── ligand_complex.pdb
       ├── trajectories/
       │   └── test_trajectory.dcd
       └── parameters/
           ├── amber_params.dat
           └── custom_ff.xml

Test Categories
~~~~~~~~~~~~~~

**1. Unit Tests**

Test individual components in isolation:

.. code-block:: python

   import pytest
   import numpy as np
   from proteinmd.core.integrators import VelocityVerletIntegrator
   from proteinmd.testing import create_test_system
   
   
   class TestVelocityVerletIntegrator:
       """Test VelocityVerlet integrator implementation."""
       
       def test_integration_step(self):
           """Test single integration step."""
           integrator = VelocityVerletIntegrator(timestep=0.001)
           system = create_test_system("harmonic_oscillator")
           
           # Get initial state
           initial_pos = system.get_positions().copy()
           initial_vel = system.get_velocities().copy()
           
           # Perform integration step
           forces = system.calculate_forces()
           integrator.step(system, forces)
           
           # Check positions and velocities changed
           final_pos = system.get_positions()
           final_vel = system.get_velocities()
           
           assert not np.allclose(initial_pos, final_pos)
           assert not np.allclose(initial_vel, final_vel)
       
       def test_energy_conservation(self):
           """Test energy conservation for harmonic oscillator."""
           integrator = VelocityVerletIntegrator(timestep=0.001)
           system = create_test_system("harmonic_oscillator")
           
           initial_energy = system.get_total_energy()
           
           # Run 1000 steps
           for _ in range(1000):
               forces = system.calculate_forces()
               integrator.step(system, forces)
           
           final_energy = system.get_total_energy()
           energy_drift = abs(final_energy - initial_energy)
           
           # Energy should be conserved within tolerance
           assert energy_drift < 0.01  # 0.01 kJ/mol tolerance

**2. Integration Tests**

Test component interactions:

.. code-block:: python

   import pytest
   from proteinmd.workflows import BasicMDWorkflow
   from proteinmd.testing import TestCase, temporary_directory
   
   
   class TestBasicMDWorkflow(TestCase):
       """Test complete MD workflow."""
       
       @pytest.mark.slow
       def test_protein_in_water_simulation(self):
           """Test complete protein simulation workflow."""
           with temporary_directory() as tmpdir:
               # Set up workflow
               workflow = BasicMDWorkflow(
                   input_pdb="tests/data/structures/small_protein.pdb",
                   output_dir=tmpdir,
                   simulation_time=0.01,  # 10 ps
                   temperature=300.0,
                   pressure=1.0
               )
               
               # Run workflow
               results = workflow.run()
               
               # Check outputs
               assert results.final_energy is not None
               assert results.trajectory_file.exists()
               assert results.log_file.exists()
               
               # Check trajectory length
               trajectory = results.load_trajectory()
               expected_frames = int(0.01 / 0.002 / 100)  # 10 ps / 2 fs / 100 (output freq)
               assert abs(len(trajectory) - expected_frames) <= 1

**3. Regression Tests**

Ensure numerical stability:

.. code-block:: python

   import pytest
   import numpy as np
   import json
   from proteinmd.testing import TestCase, load_reference_data
   
   
   class TestNumericalRegression(TestCase):
       """Test numerical regression."""
       
       @pytest.mark.regression
       def test_energy_values_regression(self):
           """Test that energy calculations remain consistent."""
           # Load reference system
           system = self.load_reference_system("1ubq_solvated")
           
           # Calculate energies
           potential_energy = system.calculate_potential_energy()
           kinetic_energy = system.calculate_kinetic_energy()
           
           # Load reference values
           reference = load_reference_data("energy_values.json")
           
           # Compare within tolerance
           np.testing.assert_allclose(
               potential_energy,
               reference["potential_energy"],
               rtol=1e-10
           )
           
           np.testing.assert_allclose(
               kinetic_energy,
               reference["kinetic_energy"],
               rtol=1e-10
           )
       
       def test_trajectory_reproducibility(self):
           """Test that simulations are reproducible."""
           # Run same simulation twice with same seed
           results1 = self.run_reference_simulation(seed=12345)
           results2 = self.run_reference_simulation(seed=12345)
           
           # Trajectories should be identical
           traj1 = results1.load_trajectory()
           traj2 = results2.load_trajectory()
           
           for frame1, frame2 in zip(traj1, traj2):
               np.testing.assert_array_equal(
                   frame1.coordinates,
                   frame2.coordinates
               )

**4. Performance Tests**

Monitor performance characteristics:

.. code-block:: python

   import pytest
   import time
   from proteinmd.testing import BenchmarkCase
   from proteinmd.utils import PerformanceProfiler
   
   
   class TestPerformanceBenchmarks(BenchmarkCase):
       """Performance benchmark tests."""
       
       @pytest.mark.performance
       @pytest.mark.parametrize("system_size", [1000, 10000, 50000])
       def test_simulation_performance(self, system_size):
           """Benchmark simulation performance for different system sizes."""
           # Create test system
           system = self.create_benchmark_system(n_atoms=system_size)
           simulation = self.create_benchmark_simulation(system)
           
           # Warm up
           simulation.run(steps=100)
           
           # Benchmark
           profiler = PerformanceProfiler()
           
           with profiler:
               simulation.run(steps=1000)
           
           # Check performance metrics
           ns_per_day = profiler.get_ns_per_day()
           
           # Store benchmark result
           self.record_benchmark(
               test_name="simulation_performance",
               system_size=system_size,
               ns_per_day=ns_per_day
           )
           
           # Performance should meet minimum requirements
           min_performance = self.get_minimum_performance(system_size)
           assert ns_per_day >= min_performance

Test Utilities
--------------

Test Framework Classes
~~~~~~~~~~~~~~~~~~~~~

**Base Test Case**

.. code-block:: python

   import unittest
   import tempfile
   import shutil
   from pathlib import Path
   import numpy as np
   
   
   class TestCase(unittest.TestCase):
       """Enhanced test case for ProteinMD."""
       
       @classmethod
       def setUpClass(cls):
           """Set up test class."""
           cls.test_data_dir = Path(__file__).parent.parent / "data"
           cls.temp_dir = None
       
       def setUp(self):
           """Set up individual test."""
           self.temp_dir = Path(tempfile.mkdtemp())
           self.addCleanup(self._cleanup_temp_dir)
       
       def _cleanup_temp_dir(self):
           """Clean up temporary directory."""
           if self.temp_dir and self.temp_dir.exists():
               shutil.rmtree(self.temp_dir)
       
       def assertArrayAlmostEqual(
           self,
           array1: np.ndarray,
           array2: np.ndarray,
           decimal: int = 7,
           msg: str = None
       ):
           """Assert two arrays are almost equal."""
           np.testing.assert_array_almost_equal(
               array1, array2, decimal=decimal, err_msg=msg
           )
       
       def assertSystemEqual(
           self,
           system1,
           system2,
           tolerance: float = 1e-10
       ):
           """Assert two systems are equal within tolerance."""
           # Compare coordinates
           coords1 = system1.get_coordinates()
           coords2 = system2.get_coordinates()
           
           np.testing.assert_allclose(
               coords1, coords2, rtol=tolerance
           )
           
           # Compare velocities
           vel1 = system1.get_velocities()
           vel2 = system2.get_velocities()
           
           np.testing.assert_allclose(
               vel1, vel2, rtol=tolerance
           )
       
       def create_temp_file(self, suffix: str = ".tmp") -> Path:
           """Create temporary file in test temp directory."""
           import tempfile
           fd, path = tempfile.mkstemp(suffix=suffix, dir=self.temp_dir)
           os.close(fd)
           return Path(path)

**Test System Factory**

.. code-block:: python

   from typing import Dict, Any
   import numpy as np
   from proteinmd.structure import ProteinStructure
   from proteinmd.environment import WaterModel
   from proteinmd.forcefield import AmberFF14SB
   
   
   class TestSystemFactory:
       """Factory for creating standardized test systems."""
       
       def __init__(self):
           self.systems = {
               "small_protein": self._create_small_protein,
               "water_box": self._create_water_box,
               "harmonic_oscillator": self._create_harmonic_oscillator,
               "lennard_jones": self._create_lennard_jones,
               "protein_ligand": self._create_protein_ligand_complex
           }
       
       def create_system(self, system_type: str, **kwargs) -> Any:
           """Create test system of specified type."""
           if system_type not in self.systems:
               available = ", ".join(self.systems.keys())
               raise ValueError(f"Unknown system type '{system_type}'. "
                              f"Available: {available}")
           
           return self.systems[system_type](**kwargs)
       
       def _create_small_protein(self, **kwargs):
           """Create small test protein system."""
           # Create minimal protein structure (e.g., polyalanine)
           n_residues = kwargs.get("n_residues", 10)
           
           protein = ProteinStructure()
           for i in range(n_residues):
               residue = self._create_alanine_residue(i + 1)
               protein.add_residue(residue)
           
           # Apply force field
           forcefield = AmberFF14SB()
           system = forcefield.create_system(protein)
           
           return system
       
       def _create_water_box(self, **kwargs):
           """Create water box system."""
           box_size = kwargs.get("box_size", 2.0)  # nm
           water_model = WaterModel("TIP3P")
           
           return water_model.create_box(box_size)
       
       def _create_harmonic_oscillator(self, **kwargs):
           """Create harmonic oscillator for testing integrators."""
           n_particles = kwargs.get("n_particles", 1)
           k = kwargs.get("spring_constant", 100.0)  # kJ/mol/nm²
           
           # Create simple harmonic system
           system = HarmonicSystem(n_particles=n_particles, k=k)
           
           # Set initial conditions
           positions = np.random.rand(n_particles, 3) - 0.5
           velocities = np.random.rand(n_particles, 3) - 0.5
           
           system.set_positions(positions)
           system.set_velocities(velocities)
           
           return system

**Custom Assertions**

.. code-block:: python

   class MDAssertions:
       """Custom assertions for MD testing."""
       
       @staticmethod
       def assert_energy_conserved(
           initial_energy: float,
           final_energy: float,
           tolerance: float = 0.01
       ):
           """Assert energy is conserved within tolerance."""
           energy_drift = abs(final_energy - initial_energy)
           if energy_drift > tolerance:
               raise AssertionError(
                   f"Energy not conserved: drift = {energy_drift:.6f} kJ/mol "
                   f"(tolerance = {tolerance} kJ/mol)"
               )
       
       @staticmethod
       def assert_temperature_stable(
           temperatures: np.ndarray,
           target_temperature: float,
           tolerance: float = 10.0
       ):
           """Assert temperature remains stable around target."""
           mean_temp = np.mean(temperatures)
           temp_deviation = abs(mean_temp - target_temperature)
           
           if temp_deviation > tolerance:
               raise AssertionError(
                   f"Temperature not stable: mean = {mean_temp:.1f} K, "
                   f"target = {target_temperature:.1f} K, "
                   f"deviation = {temp_deviation:.1f} K"
               )
       
       @staticmethod
       def assert_trajectory_reasonable(trajectory):
           """Assert trajectory contains reasonable structures."""
           # Check for NaN or infinite coordinates
           for i, frame in enumerate(trajectory):
               coords = frame.get_coordinates()
               
               if np.any(np.isnan(coords)):
                   raise AssertionError(f"NaN coordinates in frame {i}")
               
               if np.any(np.isinf(coords)):
                   raise AssertionError(f"Infinite coordinates in frame {i}")
               
               # Check for unreasonably large displacements
               if i > 0:
                   prev_coords = trajectory[i-1].get_coordinates()
                   displacement = np.max(np.linalg.norm(coords - prev_coords, axis=1))
                   
                   if displacement > 1.0:  # 1 nm max displacement per frame
                       raise AssertionError(
                           f"Large displacement in frame {i}: {displacement:.3f} nm"
                       )

Test Fixtures
------------

Pytest Fixtures
~~~~~~~~~~~~~~~

**Common Fixtures**

.. code-block:: python

   # conftest.py
   import pytest
   import tempfile
   import shutil
   from pathlib import Path
   
   
   @pytest.fixture(scope="session")
   def test_data_dir():
       """Provide path to test data directory."""
       return Path(__file__).parent / "data"
   
   
   @pytest.fixture
   def temp_dir():
       """Provide temporary directory for test files."""
       tmpdir = Path(tempfile.mkdtemp())
       yield tmpdir
       shutil.rmtree(tmpdir)
   
   
   @pytest.fixture
   def small_protein(test_data_dir):
       """Load small test protein."""
       from proteinmd.structure import ProteinStructure
       
       pdb_file = test_data_dir / "structures" / "small_protein.pdb"
       return ProteinStructure.from_pdb(str(pdb_file))
   
   
   @pytest.fixture
   def water_system():
       """Create water system for testing."""
       from proteinmd.testing import TestSystemFactory
       
       factory = TestSystemFactory()
       return factory.create_system("water_box", box_size=1.0)
   
   
   @pytest.fixture(params=["amber", "charmm"])
   def forcefield(request):
       """Parametrized fixture for different force fields."""
       if request.param == "amber":
           from proteinmd.forcefield import AmberFF14SB
           return AmberFF14SB()
       elif request.param == "charmm":
           from proteinmd.forcefield import CHARMM36
           return CHARMM36()

**Performance Fixtures**

.. code-block:: python

   @pytest.fixture
   def performance_monitor():
       """Fixture for performance monitoring."""
       from proteinmd.utils import PerformanceMonitor
       
       monitor = PerformanceMonitor()
       monitor.start()
       
       yield monitor
       
       monitor.stop()
       report = monitor.get_report()
       
       # Store performance data
       with open("performance_data.json", "a") as f:
           import json
           json.dump(report.to_dict(), f)
           f.write("\n")

**Mock Fixtures**

.. code-block:: python

   @pytest.fixture
   def mock_gpu():
       """Mock GPU for testing GPU code without hardware."""
       from unittest.mock import MagicMock
       
       mock_gpu = MagicMock()
       mock_gpu.is_available.return_value = True
       mock_gpu.get_device_count.return_value = 1
       mock_gpu.allocate_memory.return_value = MagicMock()
       
       return mock_gpu

Test Configuration
-----------------

Pytest Configuration
~~~~~~~~~~~~~~~~~~~

**pytest.ini**

.. code-block:: ini

   [tool:pytest]
   minversion = 6.0
   addopts = 
       -ra
       --strict-markers
       --strict-config
       --cov=proteinmd
       --cov-report=term-missing:skip-covered
       --cov-report=html:htmlcov
       --cov-report=xml
   testpaths = tests
   markers =
       slow: marks tests as slow (deselect with '-m "not slow"')
       performance: marks tests as performance benchmarks
       regression: marks tests as regression tests
       gpu: marks tests that require GPU
       integration: marks tests as integration tests
       unit: marks tests as unit tests

**Test Markers**

.. code-block:: python

   import pytest
   
   
   # Mark slow tests
   @pytest.mark.slow
   def test_long_simulation():
       """Test that takes a long time to run."""
       pass
   
   
   # Mark GPU tests
   @pytest.mark.gpu
   def test_gpu_acceleration():
       """Test that requires GPU hardware."""
       pass
   
   
   # Skip tests based on conditions
   @pytest.mark.skipif(
       not gpu_available(),
       reason="GPU not available"
   )
   def test_cuda_kernels():
       """Test CUDA kernels."""
       pass
   
   
   # Parametrize tests
   @pytest.mark.parametrize("backend", ["openmm", "gromacs"])
   def test_backend_compatibility(backend):
       """Test compatibility with different backends."""
       pass

Continuous Integration
---------------------

GitHub Actions Workflow
~~~~~~~~~~~~~~~~~~~~~~~

**.github/workflows/tests.yml**

.. code-block:: yaml

   name: Tests
   
   on:
     push:
       branches: [ main, develop ]
     pull_request:
       branches: [ main ]
   
   jobs:
     test:
       runs-on: ${{ matrix.os }}
       strategy:
         matrix:
           os: [ubuntu-latest, windows-latest, macos-latest]
           python-version: [3.8, 3.9, "3.10", "3.11"]
           exclude:
             # Exclude some combinations to reduce CI time
             - os: windows-latest
               python-version: 3.8
             - os: macos-latest
               python-version: 3.8
       
       steps:
       - uses: actions/checkout@v3
       
       - name: Set up Python ${{ matrix.python-version }}
         uses: actions/setup-python@v4
         with:
           python-version: ${{ matrix.python-version }}
       
       - name: Install dependencies
         run: |
           python -m pip install --upgrade pip
           pip install -r requirements-dev.txt
           pip install -e .
       
       - name: Run unit tests
         run: |
           pytest tests/unit/ -v --cov=proteinmd
       
       - name: Run integration tests
         run: |
           pytest tests/integration/ -v -m "not slow"
       
       - name: Upload coverage to Codecov
         uses: codecov/codecov-action@v3
         with:
           file: ./coverage.xml
           flags: unittests
           name: codecov-umbrella

**Performance Testing Workflow**

.. code-block:: yaml

   name: Performance Tests
   
   on:
     schedule:
       - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM
     workflow_dispatch:
   
   jobs:
     performance:
       runs-on: ubuntu-latest
       
       steps:
       - uses: actions/checkout@v3
       
       - name: Set up Python
         uses: actions/setup-python@v4
         with:
           python-version: "3.10"
       
       - name: Install dependencies
         run: |
           pip install -r requirements-dev.txt
           pip install -e .
       
       - name: Run performance benchmarks
         run: |
           pytest tests/performance/ -v --benchmark-json=benchmark.json
       
       - name: Store benchmark results
         uses: benchmark-action/github-action-benchmark@v1
         with:
           tool: 'pytest'
           output-file-path: benchmark.json
           github-token: ${{ secrets.GITHUB_TOKEN }}
           auto-push: true

Test Data Management
-------------------

Test Data Organization
~~~~~~~~~~~~~~~~~~~~~

**Version Control Strategy**

.. code-block:: bash

   # Small test files (< 1 MB) - store in git
   tests/data/structures/small_protein.pdb
   tests/data/parameters/simple_forcefield.xml
   
   # Large test files - use Git LFS
   tests/data/trajectories/long_simulation.dcd
   tests/data/structures/large_complex.pdb
   
   # Very large files - download on demand
   tests/data/download_large_files.py

**Test Data Generation**

.. code-block:: python

   # generate_test_data.py
   import numpy as np
   from proteinmd.structure import ProteinStructure
   from proteinmd.testing import TestSystemFactory
   
   
   def generate_small_protein():
       """Generate small protein for testing."""
       factory = TestSystemFactory()
       protein = factory.create_system("small_protein", n_residues=20)
       protein.to_pdb("tests/data/structures/small_protein.pdb")
   
   
   def generate_test_trajectory():
       """Generate test trajectory."""
       system = TestSystemFactory().create_system("water_box")
       simulation = create_test_simulation(system)
       
       simulation.run(steps=1000)
       simulation.save_trajectory("tests/data/trajectories/test_traj.dcd")
   
   
   if __name__ == "__main__":
       generate_small_protein()
       generate_test_trajectory()

Test Reporting
-------------

Coverage Reports
~~~~~~~~~~~~~~~

**Coverage Configuration**

.. code-block:: ini

   # .coveragerc
   [run]
   source = proteinmd
   omit = 
       */tests/*
       */setup.py
       */venv/*
       */__pycache__/*
   
   [report]
   exclude_lines =
       pragma: no cover
       def __repr__
       if self.debug:
       if settings.DEBUG
       raise AssertionError
       raise NotImplementedError
       if 0:
       if __name__ == .__main__.:
   
   [html]
   directory = htmlcov

**Performance Reporting**

.. code-block:: python

   # performance_report.py
   import json
   import matplotlib.pyplot as plt
   from pathlib import Path
   
   
   class PerformanceReporter:
       """Generate performance reports from benchmark data."""
       
       def __init__(self, data_file="performance_data.json"):
           self.data_file = Path(data_file)
           self.data = self._load_data()
       
       def _load_data(self):
           """Load performance data."""
           data = []
           if self.data_file.exists():
               with open(self.data_file) as f:
                   for line in f:
                       data.append(json.loads(line))
           return data
       
       def generate_report(self, output_dir="performance_reports"):
           """Generate performance report."""
           output_dir = Path(output_dir)
           output_dir.mkdir(exist_ok=True)
           
           # Plot performance trends
           self._plot_performance_trends(output_dir / "trends.png")
           
           # Generate HTML report
           self._generate_html_report(output_dir / "report.html")
       
       def _plot_performance_trends(self, output_file):
           """Plot performance trends over time."""
           # Implementation for plotting
           pass

Best Practices
-------------

Test Writing Guidelines
~~~~~~~~~~~~~~~~~~~~~~

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Test Organization**: Group related tests in classes
3. **Setup/Teardown**: Use appropriate setup and cleanup methods
4. **Assertions**: Use specific assertions with clear error messages
5. **Test Data**: Use minimal, focused test data
6. **Mocking**: Mock external dependencies appropriately

**Example Best Practices**

.. code-block:: python

   class TestRMSDCalculation:
       """Test RMSD calculation functionality."""
       
       def test_rmsd_identical_structures_returns_zero(self):
           """RMSD of identical structures should be zero."""
           coords = np.random.rand(10, 3)
           rmsd = calculate_rmsd(coords, coords)
           assert rmsd == 0.0, "RMSD of identical structures should be exactly zero"
       
       def test_rmsd_with_known_values(self):
           """Test RMSD calculation with known expected values."""
           # Use simple, manually calculable case
           coords1 = np.array([[0, 0, 0], [1, 0, 0]])
           coords2 = np.array([[0, 0, 1], [1, 0, 1]])
           
           expected_rmsd = 1.0  # All atoms displaced by 1 unit in z
           actual_rmsd = calculate_rmsd(coords1, coords2)
           
           assert abs(actual_rmsd - expected_rmsd) < 1e-10

See Also
--------

* :doc:`contributing` - Contributing guidelines
* :doc:`architecture` - Software architecture
* :doc:`../api/index` - API reference for testing utilities
