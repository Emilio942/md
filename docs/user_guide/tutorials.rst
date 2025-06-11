Tutorials
=========

This section provides step-by-step tutorials for common molecular dynamics tasks using ProteinMD.

.. contents:: Tutorial Topics
   :local:
   :depth: 2

Basic Protein Simulation
-----------------------

Setting Up Your First Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial walks through setting up a basic MD simulation of a protein in water.

**Step 1: Prepare the Protein Structure**

.. code-block:: python

   from proteinmd.structure import ProteinStructure
   from proteinmd.structure.validation import StructureValidator
   
   # Load protein structure
   protein = ProteinStructure.from_pdb("1ubq.pdb")
   
   # Validate structure
   validator = StructureValidator()
   issues = validator.validate(protein)
   
   if issues:
       print("Structure issues found:")
       for issue in issues:
           print(f"  - {issue}")
   
   # Add missing hydrogens
   protein.add_hydrogens(ph=7.0)
   
   # Save prepared structure
   protein.to_pdb("1ubq_prepared.pdb")

**Step 2: Set Up Simulation Environment**

.. code-block:: python

   from proteinmd.environment import WaterModel, PeriodicBoundary
   from proteinmd.forcefield import AmberFF14SB
   
   # Create water box
   water_model = WaterModel("TIP3P")
   box_size = protein.get_bounding_box() + 2.0  # 2 nm padding
   
   # Set up periodic boundaries
   boundary = PeriodicBoundary(box_size)
   
   # Solvate protein
   solvated_system = water_model.solvate(protein, boundary)
   
   # Add ions for neutralization
   solvated_system.add_ions(concentration=0.15)  # 150 mM NaCl

**Step 3: Configure Force Field**

.. code-block:: python

   # Initialize force field
   forcefield = AmberFF14SB()
   
   # Apply force field parameters
   system = forcefield.create_system(solvated_system)
   
   # Set up non-bonded interactions
   system.set_nonbonded_cutoff(1.0)  # 1 nm cutoff
   system.set_pme_parameters(alpha=0.31)  # PME electrostatics

**Step 4: Run Simulation**

.. code-block:: python

   from proteinmd.core import MDSimulation, VelocityVerletIntegrator
   from proteinmd.core import LangevinThermostat, BerendsenBarostat
   
   # Set up integrator and thermostats
   integrator = VelocityVerletIntegrator(timestep=0.002)  # 2 fs
   thermostat = LangevinThermostat(temperature=300.0, friction=1.0)
   barostat = BerendsenBarostat(pressure=1.0, tau=1.0)
   
   # Create simulation
   simulation = MDSimulation(
       system=system,
       integrator=integrator,
       thermostat=thermostat,
       barostat=barostat
   )
   
   # Energy minimization
   simulation.minimize_energy(max_iterations=1000)
   
   # Equilibration (NVT)
   simulation.run_nvt(steps=50000, temperature=300.0)  # 100 ps
   
   # Production run (NPT)
   simulation.run_npt(
       steps=2500000,  # 5 ns
       temperature=300.0,
       pressure=1.0,
       output_frequency=1000
   )

Advanced Simulation Techniques
-----------------------------

Enhanced Sampling with Umbrella Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Learn how to use umbrella sampling to calculate free energy profiles.

**Setting Up Umbrella Sampling**

.. code-block:: python

   from proteinmd.sampling import UmbrellaSampling
   from proteinmd.analysis import ReactionCoordinate
   
   # Define reaction coordinate (e.g., distance between two residues)
   rc = ReactionCoordinate.distance(
       atom1="CA:LEU10",
       atom2="CA:VAL50"
   )
   
   # Set up umbrella sampling windows
   umbrella = UmbrellaSampling(
       reaction_coordinate=rc,
       windows=20,
       window_range=(0.5, 3.0),  # 0.5 to 3.0 nm
       force_constant=1000.0  # kJ/mol/nm²
   )
   
   # Run sampling
   for window_id, restraint_value in umbrella.get_windows():
       print(f"Running window {window_id}: restraint at {restraint_value:.2f} nm")
       
       # Set up restraint
       system.add_harmonic_restraint(rc, restraint_value, 1000.0)
       
       # Run simulation for this window
       simulation.run(steps=1000000)  # 2 ns per window
       
       # Save trajectory
       simulation.save_trajectory(f"umbrella_window_{window_id}.dcd")

**Analyzing Results with WHAM**

.. code-block:: python

   from proteinmd.analysis import WHAM
   
   # Collect data from all windows
   wham = WHAM(temperature=300.0)
   
   for window_id in range(20):
       trajectory = f"umbrella_window_{window_id}.dcd"
       restraint_center = umbrella.get_restraint_center(window_id)
       
       wham.add_data(
           trajectory=trajectory,
           restraint_center=restraint_center,
           force_constant=1000.0
       )
   
   # Calculate free energy profile
   bins, free_energy, error = wham.calculate_pmf(nbins=100)
   
   # Plot results
   import matplotlib.pyplot as plt
   plt.figure(figsize=(10, 6))
   plt.errorbar(bins, free_energy, yerr=error, fmt='o-')
   plt.xlabel('Distance (nm)')
   plt.ylabel('Free Energy (kJ/mol)')
   plt.title('Free Energy Profile from Umbrella Sampling')
   plt.show()

Replica Exchange Molecular Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set up temperature replica exchange simulations for enhanced conformational sampling.

.. code-block:: python

   from proteinmd.sampling import ReplicaExchange
   
   # Define temperature range
   temperatures = [300, 310, 320, 330, 345, 360, 375, 390, 410, 430]
   
   # Set up replica exchange
   remd = ReplicaExchange(
       system=system,
       temperatures=temperatures,
       exchange_frequency=1000,  # Attempt exchanges every 1000 steps
       output_frequency=5000
   )
   
   # Run REMD simulation
   remd.run(
       steps_per_replica=5000000,  # 10 ns per replica
       equilibration_steps=500000  # 1 ns equilibration
   )
   
   # Analyze exchange statistics
   exchange_stats = remd.get_exchange_statistics()
   print(f"Average exchange probability: {exchange_stats['avg_probability']:.3f}")
   
   # Extract conformations at different temperatures
   for temp in [300, 350, 400]:
       conformations = remd.get_conformations(temperature=temp)
       print(f"Sampled {len(conformations)} conformations at {temp} K")

Analysis Workflows
-----------------

Structural Analysis
~~~~~~~~~~~~~~~~~~

Comprehensive analysis of protein structure and dynamics.

**RMSD and RMSF Analysis**

.. code-block:: python

   from proteinmd.analysis import RMSD, RMSF
   import numpy as np
   
   # Load trajectory
   trajectory = simulation.get_trajectory()
   
   # Calculate RMSD relative to initial structure
   rmsd_calc = RMSD(reference=trajectory[0])
   rmsd_values = []
   
   for frame in trajectory:
       rmsd = rmsd_calc.calculate(frame, selection="protein")
       rmsd_values.append(rmsd)
   
   # Calculate RMSF for each residue
   rmsf_calc = RMSF()
   rmsf_values = rmsf_calc.calculate(trajectory, selection="CA")
   
   # Plot results
   import matplotlib.pyplot as plt
   
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
   
   # RMSD plot
   time = np.arange(len(rmsd_values)) * 0.002  # 2 fs timestep
   ax1.plot(time, rmsd_values)
   ax1.set_xlabel('Time (ns)')
   ax1.set_ylabel('RMSD (nm)')
   ax1.set_title('Protein RMSD vs Time')
   
   # RMSF plot
   residue_ids = range(1, len(rmsf_values) + 1)
   ax2.plot(residue_ids, rmsf_values)
   ax2.set_xlabel('Residue Number')
   ax2.set_ylabel('RMSF (nm)')
   ax2.set_title('Per-Residue Flexibility')
   
   plt.tight_layout()
   plt.show()

**Secondary Structure Analysis**

.. code-block:: python

   from proteinmd.analysis import SecondaryStructure
   
   # Calculate secondary structure for each frame
   ss_calc = SecondaryStructure(method="dssp")
   ss_timeline = []
   
   for frame in trajectory:
       ss = ss_calc.calculate(frame)
       ss_timeline.append(ss)
   
   # Analyze secondary structure content
   ss_content = ss_calc.analyze_content(ss_timeline)
   
   print("Average Secondary Structure Content:")
   print(f"  α-helix: {ss_content['helix']:.1f}%")
   print(f"  β-sheet: {ss_content['sheet']:.1f}%")
   print(f"  Turn:    {ss_content['turn']:.1f}%")
   print(f"  Coil:    {ss_content['coil']:.1f}%")
   
   # Plot secondary structure timeline
   ss_calc.plot_timeline(ss_timeline, save_path="ss_timeline.png")

**Hydrogen Bond Analysis**

.. code-block:: python

   from proteinmd.analysis import HydrogenBonds
   
   # Analyze hydrogen bonds
   hb_calc = HydrogenBonds(
       donor_cutoff=0.35,    # 3.5 Å
       angle_cutoff=30.0     # 30 degrees
   )
   
   # Calculate hydrogen bonds for each frame
   hb_data = []
   for frame in trajectory:
       hbonds = hb_calc.calculate(frame)
       hb_data.append(hbonds)
   
   # Analyze persistent hydrogen bonds
   persistent_hbonds = hb_calc.find_persistent(
       hb_data, 
       occupancy_threshold=0.5  # Present in 50% of frames
   )
   
   print(f"Found {len(persistent_hbonds)} persistent hydrogen bonds:")
   for hbond in persistent_hbonds[:10]:  # Show first 10
       print(f"  {hbond['donor']} → {hbond['acceptor']} "
             f"(occupancy: {hbond['occupancy']:.2f})")

Energy Analysis
~~~~~~~~~~~~~~

Analyze energy components and thermodynamic properties.

.. code-block:: python

   from proteinmd.analysis import EnergyAnalysis
   
   # Load energy data
   energy_data = simulation.get_energy_data()
   
   # Create energy analyzer
   energy_analyzer = EnergyAnalysis(energy_data)
   
   # Calculate averages and fluctuations
   stats = energy_analyzer.calculate_statistics()
   
   print("Energy Statistics (last 1 ns):")
   print(f"  Potential Energy: {stats['potential']['mean']:.1f} ± "
         f"{stats['potential']['std']:.1f} kJ/mol")
   print(f"  Kinetic Energy:   {stats['kinetic']['mean']:.1f} ± "
         f"{stats['kinetic']['std']:.1f} kJ/mol")
   print(f"  Temperature:      {stats['temperature']['mean']:.1f} ± "
         f"{stats['temperature']['std']:.1f} K")
   
   # Check energy conservation
   drift = energy_analyzer.calculate_energy_drift()
   print(f"  Energy Drift:     {drift:.3f} kJ/mol/ns")
   
   # Plot energy components
   energy_analyzer.plot_components(save_path="energy_analysis.png")

Advanced Analysis Techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principal Component Analysis**

.. code-block:: python

   from proteinmd.analysis import PCA
   
   # Perform PCA on Cα coordinates
   pca = PCA()
   pca.fit(trajectory, selection="CA")
   
   # Get principal components
   eigenvalues = pca.get_eigenvalues()
   eigenvectors = pca.get_eigenvectors()
   
   # Project trajectory onto PC space
   projections = pca.transform(trajectory)
   
   print(f"First 5 eigenvalues: {eigenvalues[:5]}")
   print(f"Cumulative variance explained by first 10 PCs: "
         f"{pca.get_explained_variance(n_components=10):.1f}%")
   
   # Plot first two principal components
   import matplotlib.pyplot as plt
   plt.figure(figsize=(10, 8))
   plt.scatter(projections[:, 0], projections[:, 1], 
              c=range(len(projections)), cmap='viridis', alpha=0.6)
   plt.xlabel(f'PC1 ({pca.get_explained_variance(1):.1f}% variance)')
   plt.ylabel(f'PC2 ({pca.get_explained_variance(2):.1f}% variance)')
   plt.title('Principal Component Analysis')
   plt.colorbar(label='Time (frames)')
   plt.show()

**Dynamic Cross-Correlation**

.. code-block:: python

   from proteinmd.analysis import DynamicCrossCorrelation
   
   # Calculate cross-correlation matrix
   dcc = DynamicCrossCorrelation()
   correlation_matrix = dcc.calculate(trajectory, selection="CA")
   
   # Plot correlation matrix
   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(10, 8))
   plt.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
   plt.colorbar(label='Correlation Coefficient')
   plt.xlabel('Residue Number')
   plt.ylabel('Residue Number')
   plt.title('Dynamic Cross-Correlation Matrix')
   plt.show()
   
   # Find highly correlated motions
   high_correlations = dcc.find_correlations(
       correlation_matrix, 
       threshold=0.7
   )
   
   print(f"Found {len(high_correlations)} highly correlated residue pairs:")
   for pair in high_correlations[:5]:
       print(f"  Residues {pair[0]} - {pair[1]}: "
             f"correlation = {pair[2]:.3f}")

Troubleshooting Common Issues
----------------------------

Simulation Stability Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue: Simulation Crashes with "LINCS Warning"**

.. code-block:: python

   # Solution 1: Reduce timestep
   integrator = VelocityVerletIntegrator(timestep=0.001)  # 1 fs instead of 2 fs
   
   # Solution 2: Use constraints
   system.add_constraints("h-bonds")  # Constrain hydrogen bonds
   
   # Solution 3: More thorough energy minimization
   simulation.minimize_energy(
       max_iterations=5000,
       convergence_tolerance=10.0  # kJ/mol/nm
   )

**Issue: Temperature or Pressure Fluctuations**

.. code-block:: python

   # Solution 1: Adjust thermostat parameters
   thermostat = LangevinThermostat(
       temperature=300.0,
       friction=2.0  # Increase friction for better coupling
   )
   
   # Solution 2: Use different barostat
   from proteinmd.core import ParrinelloRahmanBarostat
   barostat = ParrinelloRahmanBarostat(
       pressure=1.0,
       tau=2.0,  # Longer coupling time
       compressibility=4.5e-5
   )

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

**Improving Simulation Speed**

.. code-block:: python

   # Use optimized settings
   system.set_nonbonded_cutoff(1.0)  # Reasonable cutoff
   system.set_neighbor_list_update(10)  # Update every 10 steps
   system.enable_fast_kernels()  # Use optimized algorithms
   
   # Adjust output frequency
   simulation.set_output_frequency(
       trajectory=5000,    # Save coordinates every 10 ps
       energy=1000,        # Save energy every 2 ps
       checkpoint=50000    # Checkpoint every 100 ps
   )

**Memory Usage Optimization**

.. code-block:: python

   # Process trajectory in chunks
   from proteinmd.utils import TrajectoryChunker
   
   chunker = TrajectoryChunker(chunk_size=1000)  # 1000 frames per chunk
   
   for chunk in chunker.process_trajectory("trajectory.dcd"):
       # Analyze each chunk separately
       rmsd_values = rmsd_calc.calculate_chunk(chunk)
       # Process results...

See Also
--------

* :doc:`quick_start` - Getting started with ProteinMD
* :doc:`cli_reference` - Command-line interface reference
* :doc:`examples` - More examples and use cases
* :doc:`../api/analysis` - Analysis module API reference
* :doc:`../api/sampling` - Enhanced sampling API reference
