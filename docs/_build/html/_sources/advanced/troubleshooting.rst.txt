Troubleshooting Guide
===================

This guide helps diagnose and resolve common issues in ProteinMD simulations.

.. contents:: Troubleshooting Topics
   :local:
   :depth: 2

Installation Issues
------------------

Package Installation Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue: Installation fails with dependency conflicts**

.. code-block:: bash

   ERROR: pip's dependency resolver does not currently consider all packages...

*Solution:*

.. code-block:: bash

   # Create clean environment
   conda create -n proteinmd-clean python=3.9
   conda activate proteinmd-clean
   
   # Install with conda first
   conda install numpy scipy matplotlib
   
   # Then install ProteinMD
   pip install proteinmd

**Issue: CUDA/GPU detection problems**

.. code-block:: python

   # Check CUDA availability
   from proteinmd.core import CudaSettings
   
   cuda = CudaSettings()
   if not cuda.is_available():
       print("CUDA not available")
       print(f"CUDA path: {cuda.get_cuda_path()}")
       print(f"Driver version: {cuda.get_driver_version()}")

*Solution:*

.. code-block:: bash

   # Verify CUDA installation
   nvidia-smi
   nvcc --version
   
   # Reinstall with CUDA support
   pip uninstall proteinmd
   pip install proteinmd[cuda]

**Issue: OpenMM/GROMACS backend not found**

.. code-block:: python

   # Check available backends
   from proteinmd.core import list_available_backends
   
   backends = list_available_backends()
   print(f"Available backends: {backends}")

*Solution:*

.. code-block:: bash

   # Install missing backend
   conda install -c conda-forge openmm
   # or
   conda install -c conda-forge gromacs

Structure Preparation Issues
---------------------------

PDB Loading Problems
~~~~~~~~~~~~~~~~~~~

**Issue: PDB file parsing errors**

.. code-block:: python

   from proteinmd.structure import ProteinStructure
   from proteinmd.structure.validation import StructureValidator
   
   try:
       protein = ProteinStructure.from_pdb("problematic.pdb")
   except Exception as e:
       print(f"PDB loading error: {e}")
       
       # Try with error recovery
       protein = ProteinStructure.from_pdb(
           "problematic.pdb",
           ignore_errors=True,
           fix_residue_names=True
       )

*Common issues and fixes:*

1. **Non-standard residue names**
   
   .. code-block:: python
   
      # Fix residue names
      from proteinmd.structure.utils import fix_residue_names
      
      protein = fix_residue_names(protein, mapping={
          'HIE': 'HIS',
          'HID': 'HIS',
          'CYX': 'CYS'
      })

2. **Missing hydrogen atoms**
   
   .. code-block:: python
   
      # Add hydrogens
      protein.add_hydrogens(ph=7.0, ionic_strength=0.15)
      
      # Verify hydrogens were added
      h_count = sum(1 for atom in protein.atoms if atom.element == 'H')
      print(f"Added {h_count} hydrogen atoms")

3. **Chain breaks**
   
   .. code-block:: python
   
      # Detect chain breaks
      from proteinmd.structure.validation import ChainBreakDetector
      
      detector = ChainBreakDetector()
      breaks = detector.find_breaks(protein)
      
      for break_info in breaks:
          print(f"Chain break between {break_info.residue1} and {break_info.residue2}")

**Issue: Structure validation failures**

.. code-block:: python

   validator = StructureValidator()
   issues = validator.validate(protein)
   
   # Categorize issues by severity
   critical_issues = [i for i in issues if i.severity == "CRITICAL"]
   warnings = [i for i in issues if i.severity == "WARNING"]
   
   print(f"Critical issues: {len(critical_issues)}")
   print(f"Warnings: {len(warnings)}")
   
   # Auto-fix common issues
   if critical_issues:
       fixed_protein = validator.auto_fix(protein, fix_critical=True)

Force Field Issues
-----------------

Parameter Assignment Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue: Missing force field parameters**

.. code-block:: python

   from proteinmd.forcefield import AmberFF14SB
   from proteinmd.forcefield.validation import ParameterCoverage
   
   forcefield = AmberFF14SB()
   coverage_checker = ParameterCoverage()
   
   # Check parameter coverage
   coverage = coverage_checker.check(protein, forcefield)
   
   if coverage.missing_parameters:
       print("Missing parameters:")
       for param in coverage.missing_parameters:
           print(f"  {param.type}: {param.atoms}")
           
           # Try to find similar parameters
           similar = forcefield.find_similar_parameters(param)
           if similar:
               print(f"    Similar parameter: {similar}")

*Solutions for missing parameters:*

1. **Use GAFF for small molecules**
   
   .. code-block:: python
   
      from proteinmd.forcefield import GAFF2
      
      # For ligands/small molecules
      gaff = GAFF2()
      if not forcefield.has_parameters_for(ligand):
          gaff_params = gaff.generate_parameters(ligand)
          forcefield.add_parameters(gaff_params)

2. **Generate custom parameters**
   
   .. code-block:: python
   
      from proteinmd.forcefield.parameterization import ParameterGenerator
      
      param_gen = ParameterGenerator(method="am1bcc")
      custom_params = param_gen.generate(missing_molecule)
      forcefield.add_parameters(custom_params)

**Issue: Unreasonable initial energies**

.. code-block:: python

   # Check initial energy
   system = forcefield.create_system(protein)
   initial_energy = system.get_potential_energy()
   energy_per_atom = initial_energy / system.get_num_particles()
   
   print(f"Initial energy: {initial_energy:.1f} kJ/mol")
   print(f"Energy per atom: {energy_per_atom:.1f} kJ/mol")
   
   if energy_per_atom > 1000:  # Very high energy
       print("WARNING: Unreasonably high initial energy")
       
       # Check for close contacts
       from proteinmd.structure.validation import ContactChecker
       
       contact_checker = ContactChecker()
       close_contacts = contact_checker.find_close_contacts(protein, threshold=0.2)
       
       if close_contacts:
           print(f"Found {len(close_contacts)} close contacts")
           for contact in close_contacts[:5]:  # Show first 5
               print(f"  {contact.atom1} - {contact.atom2}: {contact.distance:.3f} nm")

Simulation Setup Issues
----------------------

System Preparation Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue: Solvation box too small**

.. code-block:: python

   from proteinmd.environment import WaterModel
   
   # Check minimum box size
   protein_bbox = protein.get_bounding_box()
   min_box_size = protein_bbox + 2.0  # 2 nm minimum padding
   
   print(f"Protein bounding box: {protein_bbox:.2f} nm")
   print(f"Minimum box size: {min_box_size:.2f} nm")
   
   # Create appropriately sized box
   water_model = WaterModel("TIP3P")
   box_size = max(min_box_size, 5.0)  # At least 5 nm
   
   solvated_system = water_model.solvate(protein, box_size=box_size)

**Issue: Ion concentration problems**

.. code-block:: python

   # Check system charge
   total_charge = sum(atom.charge for atom in protein.atoms)
   print(f"System charge: {total_charge:.2f} e")
   
   # Add neutralizing ions
   if abs(total_charge) > 0.1:
       n_counterions = int(abs(total_charge))
       if total_charge > 0:
           solvated_system.add_ions(negative_ions=n_counterions)
       else:
           solvated_system.add_ions(positive_ions=n_counterions)
   
   # Add physiological salt concentration
   solvated_system.add_salt(concentration=0.15)  # 150 mM NaCl

**Issue: Periodic boundary condition problems**

.. code-block:: python

   # Check for protein parts crossing PBC
   from proteinmd.environment import PBCChecker
   
   pbc_checker = PBCChecker()
   pbc_issues = pbc_checker.check_system(solvated_system)
   
   if pbc_issues:
       print("PBC issues detected:")
       for issue in pbc_issues:
           print(f"  {issue.description}")
       
       # Fix PBC issues
       fixed_system = pbc_checker.fix_pbc_issues(solvated_system)

Runtime Simulation Issues
------------------------

Stability Problems
~~~~~~~~~~~~~~~~~

**Issue: Simulation crashes with LINCS errors**

.. code-block:: python

   # Diagnostic approach
   from proteinmd.core.diagnostics import SimulationDiagnostic
   
   diagnostic = SimulationDiagnostic()
   
   # Check timestep
   if integrator.timestep > 0.002:
       print("WARNING: Timestep may be too large")
       integrator.timestep = 0.001  # Reduce to 1 fs
   
   # Check constraints
   if not system.has_constraints():
       system.add_constraints("h-bonds")  # Constrain hydrogen bonds
   
   # More aggressive energy minimization
   simulation.minimize_energy(
       max_iterations=10000,
       convergence_tolerance=1.0
   )
   
   # Gradual heating
   temperatures = [50, 100, 150, 200, 250, 300]
   for temp in temperatures:
       simulation.set_target_temperature(temp)
       simulation.run(steps=1000)  # 1 ps at each temperature

**Issue: Exploding coordinates**

.. code-block:: python

   # Monitor coordinate stability
   def check_coordinate_stability(simulation, max_displacement=1.0):
       """Check for exploding coordinates."""
       prev_coords = simulation.get_coordinates()
       
       for step in range(1000):
           simulation.step()
           current_coords = simulation.get_coordinates()
           
           # Check maximum displacement
           displacement = np.max(np.linalg.norm(current_coords - prev_coords, axis=1))
           
           if displacement > max_displacement:
               print(f"WARNING: Large displacement at step {step}: {displacement:.3f} nm")
               return False
           
           prev_coords = current_coords
       
       return True
   
   # Use stability check
   if not check_coordinate_stability(simulation):
       # Apply fixes
       simulation.add_position_restraints(selection="protein", force_constant=1000.0)
       simulation.minimize_energy(max_iterations=5000)

**Issue: Temperature control problems**

.. code-block:: python

   # Diagnose thermostat issues
   temp_data = []
   for step in range(5000):
       simulation.step()
       if step % 100 == 0:
           temp_data.append(simulation.get_temperature())
   
   avg_temp = np.mean(temp_data)
   temp_std = np.std(temp_data)
   
   print(f"Average temperature: {avg_temp:.1f} ± {temp_std:.1f} K")
   
   if abs(avg_temp - target_temperature) > 10.0:
       print("Temperature control problem detected")
       
       # Adjust thermostat parameters
       if isinstance(thermostat, LangevinThermostat):
           thermostat.friction = 2.0  # Increase coupling
       
       # Re-equilibrate
       simulation.run_equilibration(steps=10000)

Performance Issues
-----------------

Slow Simulation Speed
~~~~~~~~~~~~~~~~~~~

**Issue: Low ns/day performance**

.. code-block:: python

   from proteinmd.utils import PerformanceProfiler
   
   # Profile simulation performance
   profiler = PerformanceProfiler()
   
   with profiler:
       simulation.run(steps=1000)  # Short test run
   
   # Analyze bottlenecks
   report = profiler.get_report()
   print(f"Performance: {report.ns_per_day:.2f} ns/day")
   
   bottlenecks = report.identify_bottlenecks()
   for bottleneck in bottlenecks:
       print(f"Bottleneck: {bottleneck.component} ({bottleneck.percentage:.1f}%)")

*Common performance fixes:*

1. **Optimize neighbor lists**
   
   .. code-block:: python
   
      # Adjust neighbor list settings
      system.set_nonbonded_cutoff(1.0)  # Don't make too large
      system.set_neighbor_list_update_frequency(20)  # Reduce frequency

2. **Use GPU acceleration**
   
   .. code-block:: python
   
      from proteinmd.core import CudaSettings
      
      if CudaSettings.is_available():
           cuda_settings = CudaSettings()
           cuda_settings.set_precision("mixed")  # Faster than double
           simulation.use_gpu(cuda_settings)

3. **Optimize PME parameters**
   
   .. code-block:: python
   
      # Auto-tune PME
      system.auto_tune_pme()
      
      # Or set manually
      system.set_pme_grid_spacing(0.12)  # nm

**Issue: Memory usage problems**

.. code-block:: python

   import psutil
   
   # Monitor memory usage
   def monitor_memory(simulation, max_steps=10000):
       process = psutil.Process()
       
       for step in range(max_steps):
           simulation.step()
           
           if step % 1000 == 0:
               memory_mb = process.memory_info().rss / 1024 / 1024
               print(f"Step {step}: Memory usage {memory_mb:.1f} MB")
               
               if memory_mb > 8000:  # 8 GB limit
                   print("WARNING: High memory usage")
                   return False
       
       return True

*Memory optimization:*

.. code-block:: python

   # Reduce trajectory storage frequency
   simulation.set_output_frequency(
       coordinates=10000,  # Every 20 ps instead of every step
       velocities=0,       # Don't save velocities
       energies=1000       # Every 2 ps
   )
   
   # Use compressed trajectory format
   simulation.set_trajectory_format("dcd", compression=True)

Analysis Issues
--------------

Trajectory Analysis Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue: Trajectory loading failures**

.. code-block:: python

   from proteinmd.io import TrajectoryReader
   
   try:
       reader = TrajectoryReader("trajectory.dcd")
       trajectory = reader.load()
   except Exception as e:
       print(f"Trajectory loading error: {e}")
       
       # Try with error recovery
       reader = TrajectoryReader("trajectory.dcd", ignore_errors=True)
       trajectory = reader.load_partial(max_frames=1000)

**Issue: Memory issues with large trajectories**

.. code-block:: python

   # Process trajectory in chunks
   from proteinmd.utils import TrajectoryChunker
   
   chunker = TrajectoryChunker(chunk_size=500)  # 500 frames per chunk
   
   results = []
   for chunk in chunker.iterate_trajectory("large_trajectory.dcd"):
       # Analyze each chunk
       chunk_result = analyze_chunk(chunk)
       results.append(chunk_result)
   
   # Combine results
   final_result = combine_chunk_results(results)

**Issue: Analysis convergence problems**

.. code-block:: python

   from proteinmd.analysis import ConvergenceChecker
   
   # Check analysis convergence
   convergence_checker = ConvergenceChecker()
   
   # Analyze RMSD convergence
   rmsd_values = calculate_rmsd_trajectory(trajectory)
   is_converged = convergence_checker.check_rmsd_convergence(rmsd_values)
   
   if not is_converged:
       print("RMSD analysis not converged")
       print(f"Recommended additional simulation time: "
             f"{convergence_checker.get_additional_time_needed()} ns")

Common Error Messages
--------------------

Error Message Decoder
~~~~~~~~~~~~~~~~~~~~~

**"Fatal error: The system has become unstable"**

*Cause:* Usually due to timestep too large or bad initial structure.

*Solution:*

.. code-block:: python

   # Reduce timestep
   integrator.timestep = 0.0005  # 0.5 fs
   
   # More thorough minimization
   simulation.minimize_energy(max_iterations=20000)
   
   # Gradual equilibration
   simulation.gradual_equilibration(
       initial_temp=50.0,
       final_temp=300.0,
       steps_per_temp=5000
   )

**"Error: Cannot find force field parameters for atom type X"**

*Cause:* Missing force field parameters for unusual residues or ligands.

*Solution:*

.. code-block:: python

   # Generate missing parameters
   from proteinmd.forcefield.parameterization import AutoParameterizer
   
   auto_param = AutoParameterizer()
   missing_params = auto_param.generate_missing_parameters(system, forcefield)
   forcefield.add_parameters(missing_params)

**"Warning: Large force detected on atom X"**

*Cause:* Close contacts or unreasonable geometry.

*Solution:*

.. code-block:: python

   # Find problematic atoms
   forces = system.calculate_forces()
   large_forces = np.where(np.linalg.norm(forces, axis=1) > 1000.0)[0]
   
   for atom_idx in large_forces:
       atom = system.atoms[atom_idx]
       print(f"Large force on {atom.name} in residue {atom.residue}")
   
   # Apply position restraints and minimize
   system.add_position_restraints(
       selection="large_force_atoms",
       force_constant=10000.0
   )
   simulation.minimize_energy(max_iterations=10000)

Log File Analysis
~~~~~~~~~~~~~~~~

**Analyzing simulation logs**

.. code-block:: python

   from proteinmd.utils import LogAnalyzer
   
   # Analyze log file for issues
   log_analyzer = LogAnalyzer()
   log_data = log_analyzer.parse_log("simulation.log")
   
   # Check for warnings and errors
   warnings = log_data.get_warnings()
   errors = log_data.get_errors()
   
   print(f"Found {len(warnings)} warnings and {len(errors)} errors")
   
   # Analyze energy drift
   energy_drift = log_analyzer.calculate_energy_drift(log_data)
   if abs(energy_drift) > 1.0:  # kJ/mol/ns
       print(f"WARNING: Energy drift detected: {energy_drift:.3f} kJ/mol/ns")

Getting Help
-----------

Diagnostic Tools
~~~~~~~~~~~~~~~

**Automated problem detection**

.. code-block:: python

   from proteinmd.diagnostics import SystemDiagnostic
   
   # Run comprehensive system diagnostic
   diagnostic = SystemDiagnostic()
   report = diagnostic.run_full_diagnostic(
       system=system,
       simulation=simulation,
       trajectory="trajectory.dcd"  # Optional
   )
   
   # Print diagnostic report
   print(report.summary())
   
   # Get specific recommendations
   recommendations = report.get_recommendations()
   for rec in recommendations:
       print(f"Recommendation: {rec.description}")
       print(f"  Priority: {rec.priority}")
       print(f"  Action: {rec.suggested_action}")

**Debug mode**

.. code-block:: python

   # Enable debug mode for detailed logging
   import proteinmd
   proteinmd.set_debug_mode(True)
   
   # Set verbose logging
   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   # Run simulation with detailed output
   simulation.run(steps=1000)

Community Resources
~~~~~~~~~~~~~~~~~~

1. **GitHub Issues**: Report bugs and ask questions
2. **Documentation**: Check the latest documentation
3. **Tutorials**: Step-by-step guides for common tasks
4. **Examples**: Working examples for reference

**Preparing bug reports**

.. code-block:: python

   from proteinmd.utils import BugReporter
   
   # Generate bug report
   bug_reporter = BugReporter()
   report = bug_reporter.generate_report(
       error_description="Simulation crashes after 1000 steps",
       system_info=True,
       log_files=["simulation.log"],
       input_files=["system.pdb", "simulation.yaml"]
   )
   
   # Save report
   bug_reporter.save_report(report, "bug_report.zip")

See Also
--------

* :doc:`../api/index` - Complete API reference
* :doc:`performance` - Performance optimization
* :doc:`validation` - Validation protocols
* :doc:`../user_guide/tutorials` - Step-by-step tutorials
