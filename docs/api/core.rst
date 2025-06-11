Core Simulation Engine
======================

The :mod:`proteinMD.core` module provides the fundamental molecular dynamics simulation engine and algorithms that power ProteinMD.

.. currentmodule:: proteinMD.core

Overview
--------

The core module implements the heart of the molecular dynamics simulation, including:

- **Simulation Engine**: Main MD simulation loop with integrators
- **Force Calculations**: Efficient force computation algorithms  
- **Energy Management**: Kinetic and potential energy tracking
- **Thermostats and Barostats**: Temperature and pressure control
- **Trajectory Management**: Coordinate evolution and storage

Quick Example
-------------

Basic simulation setup and execution:

.. code-block:: python

   from proteinMD.core.simulation import MolecularDynamicsSimulation
   from proteinMD.structure.protein import Protein
   from proteinMD.forcefield.amber_ff14sb import AmberFF14SB

   # Initialize simulation
   protein = Protein.from_pdb("example.pdb")
   forcefield = AmberFF14SB()
   
   simulation = MolecularDynamicsSimulation(
       system=protein,
       force_field=forcefield,
       timestep=0.002,  # 2 fs
       temperature=300.0  # 300 K
   )
   
   # Run simulation
   simulation.run(50000)  # 50k steps = 100 ps
   
   # Access results
   final_energy = simulation.get_potential_energy()
   trajectory = simulation.get_trajectory()

Simulation Module
-----------------

.. automodule:: proteinMD.core.simulation
   :members:
   :undoc-members:
   :show-inheritance:

Main Simulation Class
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.core.simulation.MolecularDynamicsSimulation
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   
   **Examples**
   
   Basic simulation:
   
   .. code-block:: python
   
      # Simple protein simulation
      sim = MolecularDynamicsSimulation(protein, forcefield)
      sim.run(10000)
   
   Advanced simulation with custom parameters:
   
   .. code-block:: python
   
      sim = MolecularDynamicsSimulation(
          system=protein,
          force_field=forcefield,
          timestep=0.001,      # Smaller timestep for stability
          temperature=310.0,    # Physiological temperature
          pressure=1.0,        # 1 bar pressure
          integrator='verlet'   # Velocity Verlet integrator
      )
      
      # Set up trajectory output
      sim.set_trajectory_output("trajectory.npz", save_interval=100)
      
      # Run with progress monitoring
      sim.run(100000, progress_callback=lambda step: print(f"Step {step}"))

Integrators Module
------------------

.. automodule:: proteinMD.core.integrators
   :members:
   :undoc-members:
   :show-inheritance:

Available Integrators
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.core.integrators.VelocityVerletIntegrator
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.core.integrators import VelocityVerletIntegrator
      
      integrator = VelocityVerletIntegrator(timestep=0.002)
      simulation.set_integrator(integrator)

.. autoclass:: proteinMD.core.integrators.LeapfrogIntegrator
   :members:
   :undoc-members:
   :show-inheritance:

Thermostats Module
------------------

.. automodule:: proteinMD.core.thermostats
   :members:
   :undoc-members:
   :show-inheritance:

Temperature Control
~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.core.thermostats.NoseHooverThermostat
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.core.thermostats import NoseHooverThermostat
      
      thermostat = NoseHooverThermostat(
          target_temperature=300.0,
          coupling_time=0.1  # 0.1 ps coupling time
      )
      simulation.set_thermostat(thermostat)

.. autoclass:: proteinMD.core.thermostats.BerendsenThermostat
   :members:
   :undoc-members:
   :show-inheritance:

Barostats Module
----------------

.. automodule:: proteinMD.core.barostats
   :members:
   :undoc-members:
   :show-inheritance:

Pressure Control
~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.core.barostats.BerendsenBarostat
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.core.barostats import BerendsenBarostat
      
      barostat = BerendsenBarostat(
          target_pressure=1.0,  # 1 bar
          coupling_time=1.0,    # 1 ps coupling time
          compressibility=4.5e-5  # Water compressibility
      )
      simulation.set_barostat(barostat)

Forces Module
-------------

.. automodule:: proteinMD.core.forces
   :members:
   :undoc-members:
   :show-inheritance:

Force Calculation Engine
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.core.forces.ForceCalculator
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.core.forces import ForceCalculator
      
      force_calc = ForceCalculator(
          cutoff=1.2,           # 1.2 nm cutoff
          switch_distance=1.0,  # 1.0 nm switch distance
          use_pme=True,         # Use PME for electrostatics
          pme_grid_spacing=0.1  # 0.1 nm PME grid spacing
      )
      
      # Calculate forces and energy
      forces, potential_energy = force_calc.calculate(
          positions=protein.get_positions(),
          forcefield=amber_ff14sb
      )

Energy Module
-------------

.. automodule:: proteinMD.core.energy
   :members:
   :undoc-members:
   :show-inheritance:

Energy Tracking
~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.core.energy.EnergyMonitor
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.core.energy import EnergyMonitor
      
      energy_monitor = EnergyMonitor()
      simulation.add_observer(energy_monitor)
      
      # After simulation
      kinetic_energies = energy_monitor.get_kinetic_energy_history()
      potential_energies = energy_monitor.get_potential_energy_history()
      total_energies = energy_monitor.get_total_energy_history()

Trajectory Module
-----------------

.. automodule:: proteinMD.core.trajectory
   :members:
   :undoc-members:
   :show-inheritance:

Trajectory Management
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.core.trajectory.TrajectoryWriter
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.core.trajectory import TrajectoryWriter
      
      writer = TrajectoryWriter("simulation.npz")
      simulation.add_observer(writer)
      
      # Writer automatically saves trajectory during simulation

.. autoclass:: proteinMD.core.trajectory.TrajectoryReader
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.core.trajectory import TrajectoryReader
      
      reader = TrajectoryReader("simulation.npz")
      
      # Read trajectory data
      frames = reader.read_all_frames()
      frame_10 = reader.read_frame(10)
      
      # Iterate through trajectory
      for frame in reader:
          positions = frame.positions
          time = frame.time

Common Usage Patterns
---------------------

Complete Simulation Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinMD.core.simulation import MolecularDynamicsSimulation
   from proteinMD.core.thermostats import NoseHooverThermostat
   from proteinMD.core.barostats import BerendsenBarostat
   from proteinMD.structure.pdb_parser import PDBParser
   from proteinMD.forcefield.amber_ff14sb import AmberFF14SB

   # Load system
   parser = PDBParser()
   protein = parser.parse("system.pdb")
   
   # Setup force field
   forcefield = AmberFF14SB()
   
   # Create simulation with advanced options
   simulation = MolecularDynamicsSimulation(
       system=protein,
       force_field=forcefield,
       timestep=0.002,
       temperature=300.0,
       pressure=1.0
   )
   
   # Configure thermostats and barostats
   thermostat = NoseHooverThermostat(300.0, coupling_time=0.1)
   barostat = BerendsenBarostat(1.0, coupling_time=1.0)
   
   simulation.set_thermostat(thermostat)
   simulation.set_barostat(barostat)
   
   # Setup trajectory output
   simulation.set_trajectory_output("trajectory.npz", save_interval=100)
   
   # Run simulation
   simulation.run(100000)  # 100k steps = 200 ps

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimize for large systems
   simulation = MolecularDynamicsSimulation(
       system=large_protein,
       force_field=forcefield,
       timestep=0.002,
       temperature=300.0
   )
   
   # Enable parallel force calculation
   simulation.set_parallel_forces(n_threads=4)
   
   # Use optimized neighbor lists
   simulation.set_neighbor_list_cutoff(1.2)
   simulation.set_neighbor_list_update_frequency(20)
   
   # Memory-efficient trajectory storage
   simulation.set_trajectory_output(
       "trajectory.npz", 
       save_interval=1000,  # Save every 1000 steps
       compression=True     # Compress trajectory data
   )

See Also
--------

- :doc:`structure` - Protein structure handling
- :doc:`forcefield` - Force field implementations  
- :doc:`environment` - Environment setup (water, boundaries)
- :doc:`../user_guide/tutorials` - Step-by-step tutorials
- :doc:`../advanced/performance` - Performance optimization guide
