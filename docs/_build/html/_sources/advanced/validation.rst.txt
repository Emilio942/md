Validation and Quality Assurance
=================================

This guide covers validation protocols and quality assurance procedures for ProteinMD simulations.

.. contents:: Validation Topics
   :local:
   :depth: 2

Pre-Simulation Validation
-------------------------

Structure Validation
~~~~~~~~~~~~~~~~~~~

**Comprehensive Structure Checking**

.. code-block:: python

   from proteinmd.structure.validation import StructureValidator
   from proteinmd.structure.validation import GeometryValidator, ChemistryValidator
   
   # Load structure
   protein = ProteinStructure.from_pdb("protein.pdb")
   
   # Comprehensive validation
   validator = StructureValidator()
   
   # Basic structure checks
   basic_issues = validator.validate_basic(protein)
   print(f"Basic validation issues: {len(basic_issues)}")
   
   for issue in basic_issues:
       print(f"  {issue.severity}: {issue.description}")
       if issue.severity == "CRITICAL":
           print(f"    Suggested fix: {issue.suggested_fix}")

**Geometry Validation**

.. code-block:: python

   # Detailed geometry checking
   geom_validator = GeometryValidator()
   
   # Check bond lengths
   bond_issues = geom_validator.validate_bonds(protein)
   print(f"\nBond length issues: {len(bond_issues)}")
   
   for issue in bond_issues:
       if issue.deviation > 0.1:  # > 0.1 Å deviation
           print(f"  {issue.atom1}-{issue.atom2}: "
                 f"{issue.length:.3f} Å (expected: {issue.expected:.3f} Å)")
   
   # Check angles
   angle_issues = geom_validator.validate_angles(protein)
   severe_angle_issues = [issue for issue in angle_issues if issue.deviation > 10.0]
   print(f"Severe angle issues (>10°): {len(severe_angle_issues)}")
   
   # Check dihedrals
   dihedral_issues = geom_validator.validate_dihedrals(protein)
   print(f"Dihedral issues: {len(dihedral_issues)}")

**Chemistry Validation**

.. code-block:: python

   # Chemical consistency checks
   chem_validator = ChemistryValidator()
   
   # Validate protonation states
   protonation_issues = chem_validator.validate_protonation(protein, ph=7.0)
   print(f"\nProtonation issues: {len(protonation_issues)}")
   
   # Check for unusual residues
   unusual_residues = chem_validator.find_unusual_residues(protein)
   if unusual_residues:
       print("Unusual residues found:")
       for residue in unusual_residues:
           print(f"  {residue.name} at position {residue.number}")
   
   # Validate disulfide bonds
   disulfide_issues = chem_validator.validate_disulfide_bonds(protein)
   print(f"Disulfide bond issues: {len(disulfide_issues)}")

Force Field Validation
~~~~~~~~~~~~~~~~~~~~~

**Parameter Coverage Check**

.. code-block:: python

   from proteinmd.forcefield.validation import ForceFieldValidator
   
   # Check force field parameter coverage
   ff_validator = ForceFieldValidator()
   forcefield = AmberFF14SB()
   
   coverage_report = ff_validator.check_coverage(protein, forcefield)
   
   print(f"Parameter coverage: {coverage_report.coverage_percentage:.1f}%")
   
   if coverage_report.missing_parameters:
       print("Missing parameters:")
       for param in coverage_report.missing_parameters:
           print(f"  {param.type}: {param.description}")

**Energy Validation**

.. code-block:: python

   from proteinmd.validation import EnergyValidator
   
   # Create system for energy validation
   system = forcefield.create_system(protein)
   
   # Validate initial energies
   energy_validator = EnergyValidator()
   energy_report = energy_validator.validate_initial_energy(system)
   
   print(f"Initial potential energy: {energy_report.potential_energy:.1f} kJ/mol")
   print(f"Energy per atom: {energy_report.energy_per_atom:.1f} kJ/mol")
   
   # Check for unreasonable energies
   if energy_report.has_extreme_energies:
       print("Warning: Extreme energies detected")
       for warning in energy_report.warnings:
           print(f"  {warning}")

Simulation Validation
--------------------

Equilibration Validation
~~~~~~~~~~~~~~~~~~~~~~~

**Energy Convergence**

.. code-block:: python

   from proteinmd.validation import EquilibrationValidator
   
   # Monitor equilibration
   equilibration_validator = EquilibrationValidator()
   
   def validate_equilibration(simulation, steps):
       """Run equilibration with validation"""
       
       # Collect data during equilibration
       energy_data = []
       temperature_data = []
       pressure_data = []
       
       for step in range(steps):
           simulation.step()
           
           if step % 100 == 0:  # Collect every 100 steps
               state = simulation.get_current_state()
               energy_data.append(state.potential_energy)
               temperature_data.append(state.temperature)
               pressure_data.append(state.pressure)
       
       # Validate convergence
       convergence_report = equilibration_validator.check_convergence({
           'energy': energy_data,
           'temperature': temperature_data,
           'pressure': pressure_data
       })
       
       return convergence_report
   
   # Run validation
   report = validate_equilibration(simulation, 50000)  # 50k steps
   
   print("Equilibration validation:")
   print(f"  Energy converged: {report.energy_converged}")
   print(f"  Temperature stable: {report.temperature_stable}")
   print(f"  Pressure stable: {report.pressure_stable}")
   
   if not report.fully_equilibrated:
       print(f"  Recommended additional steps: {report.additional_steps}")

**Volume Equilibration**

.. code-block:: python

   # Specific validation for NPT equilibration
   volume_data = []
   
   for step in range(steps):
       simulation.step()
       if step % 100 == 0:
           box_vectors = simulation.get_box_vectors()
           volume = np.linalg.det(box_vectors)
           volume_data.append(volume)
   
   # Check volume convergence
   volume_converged = equilibration_validator.check_volume_convergence(
       volume_data, 
       tolerance=0.02  # 2% tolerance
   )
   
   print(f"Volume equilibration: {'Converged' if volume_converged else 'Not converged'}")

Production Run Validation
~~~~~~~~~~~~~~~~~~~~~~~~

**Trajectory Quality Checks**

.. code-block:: python

   from proteinmd.validation import TrajectoryValidator
   
   # Validate trajectory quality
   traj_validator = TrajectoryValidator()
   trajectory = simulation.get_trajectory()
   
   # Check for structural integrity
   integrity_report = traj_validator.check_structural_integrity(trajectory)
   
   print("Trajectory validation:")
   print(f"  Frames analyzed: {len(trajectory)}")
   print(f"  Structural integrity: {integrity_report.integrity_score:.3f}")
   
   if integrity_report.issues:
       print("  Issues detected:")
       for issue in integrity_report.issues:
           print(f"    Frame {issue.frame}: {issue.description}")

**Thermodynamic Validation**

.. code-block:: python

   from proteinmd.validation import ThermodynamicValidator
   
   # Validate thermodynamic properties
   thermo_validator = ThermodynamicValidator()
   
   # Load energy data
   energy_data = simulation.get_energy_data()
   
   # Validate energy conservation (for NVE)
   if simulation.ensemble == "NVE":
       conservation_report = thermo_validator.check_energy_conservation(energy_data)
       print(f"Energy drift: {conservation_report.drift_per_ns:.3f} kJ/mol/ns")
       
       if abs(conservation_report.drift_per_ns) > 1.0:
           print("Warning: Significant energy drift detected")
   
   # Validate temperature distribution
   temp_validation = thermo_validator.validate_temperature_distribution(
       energy_data['temperature'],
       target_temperature=300.0
   )
   
   print(f"Temperature validation:")
   print(f"  Mean: {temp_validation.mean:.1f} K")
   print(f"  Std: {temp_validation.std:.1f} K")
   print(f"  Distribution normal: {temp_validation.is_normal}")

Sampling Validation
~~~~~~~~~~~~~~~~~~

**Conformational Sampling**

.. code-block:: python

   from proteinmd.validation import SamplingValidator
   
   # Validate conformational sampling
   sampling_validator = SamplingValidator()
   
   # Check RMSD distribution
   rmsd_values = []
   reference_structure = trajectory[0]
   
   for frame in trajectory:
       rmsd = calculate_rmsd(reference_structure, frame)
       rmsd_values.append(rmsd)
   
   sampling_report = sampling_validator.analyze_rmsd_distribution(rmsd_values)
   
   print("Sampling validation:")
   print(f"  RMSD range: {sampling_report.rmsd_range:.2f} nm")
   print(f"  Sampling efficiency: {sampling_report.efficiency:.2f}")
   print(f"  Convergence achieved: {sampling_report.converged}")

**Phase Space Coverage**

.. code-block:: python

   # Analyze phase space coverage using dihedral angles
   from proteinmd.analysis import DihedralAnalysis
   
   dihedral_analyzer = DihedralAnalysis()
   phi_psi_data = dihedral_analyzer.extract_phi_psi(trajectory)
   
   # Calculate coverage of Ramachandran space
   coverage_report = sampling_validator.analyze_ramachandran_coverage(phi_psi_data)
   
   print(f"Ramachandran coverage: {coverage_report.coverage_percentage:.1f}%")
   print(f"Outliers: {coverage_report.outlier_percentage:.1f}%")

Statistical Validation
---------------------

Convergence Analysis
~~~~~~~~~~~~~~~~~~

**Block Averaging**

.. code-block:: python

   from proteinmd.validation import ConvergenceAnalyzer
   
   # Perform block averaging analysis
   convergence_analyzer = ConvergenceAnalyzer()
   
   # Analyze property convergence (e.g., RMSD)
   property_data = rmsd_values  # From previous example
   
   block_analysis = convergence_analyzer.block_averaging(
       data=property_data,
       max_block_size=len(property_data) // 10
   )
   
   print("Block averaging analysis:")
   print(f"  Converged value: {block_analysis.converged_value:.3f}")
   print(f"  Statistical uncertainty: {block_analysis.uncertainty:.3f}")
   print(f"  Correlation time: {block_analysis.correlation_time:.1f} frames")

**Autocorrelation Analysis**

.. code-block:: python

   # Calculate autocorrelation functions
   autocorr_analysis = convergence_analyzer.autocorrelation_analysis(property_data)
   
   print(f"Autocorrelation time: {autocorr_analysis.correlation_time:.1f} frames")
   print(f"Effective sample size: {autocorr_analysis.effective_samples}")
   
   # Plot autocorrelation function
   autocorr_analysis.plot(save_path="autocorrelation.png")

Error Analysis
~~~~~~~~~~~~~

**Bootstrap Error Estimation**

.. code-block:: python

   from proteinmd.validation import ErrorAnalyzer
   
   # Bootstrap error analysis
   error_analyzer = ErrorAnalyzer()
   
   bootstrap_results = error_analyzer.bootstrap_analysis(
       data=property_data,
       n_bootstrap=1000,
       confidence_level=0.95
   )
   
   print("Bootstrap error analysis:")
   print(f"  Mean: {bootstrap_results.mean:.3f}")
   print(f"  95% CI: [{bootstrap_results.ci_lower:.3f}, {bootstrap_results.ci_upper:.3f}]")

Comparative Validation
--------------------

Literature Comparison
~~~~~~~~~~~~~~~~~~~

**Experimental Data Comparison**

.. code-block:: python

   from proteinmd.validation import ExperimentalComparator
   
   # Compare with experimental data
   exp_comparator = ExperimentalComparator()
   
   # Load experimental reference data
   exp_data = exp_comparator.load_experimental_data("1ubq_experimental.json")
   
   # Compare structural properties
   structural_comparison = exp_comparator.compare_structure(
       simulation_structure=protein,
       experimental_data=exp_data
   )
   
   print("Experimental comparison:")
   print(f"  RMSD to X-ray: {structural_comparison.rmsd_xray:.2f} nm")
   print(f"  B-factor correlation: {structural_comparison.bfactor_correlation:.3f}")

**Cross-Validation with Other Software**

.. code-block:: python

   from proteinmd.validation import CrossValidator
   
   # Cross-validate with GROMACS/AMBER results
   cross_validator = CrossValidator()
   
   # Load reference trajectory
   reference_trajectory = cross_validator.load_reference("gromacs_traj.xtc")
   
   # Compare trajectories
   comparison_report = cross_validator.compare_trajectories(
       trajectory1=trajectory,
       trajectory2=reference_trajectory
   )
   
   print("Cross-validation with GROMACS:")
   print(f"  RMSD difference: {comparison_report.rmsd_difference:.3f} nm")
   print(f"  Energy difference: {comparison_report.energy_difference:.1f} kJ/mol")

Force Field Validation
~~~~~~~~~~~~~~~~~~~~~

**QM/MM Validation**

.. code-block:: python

   from proteinmd.validation import QMMMValidator
   
   # Validate against quantum mechanical calculations
   qmmm_validator = QMMMValidator()
   
   # Select representative conformations
   representative_frames = sampling_validator.select_representative_frames(
       trajectory, n_frames=10
   )
   
   for i, frame in enumerate(representative_frames):
       # Calculate QM energy for small region
       qm_energy = qmmm_validator.calculate_qm_energy(
           structure=frame,
           qm_region="active_site",  # Define QM region
           method="B3LYP",
           basis_set="6-31G*"
       )
       
       # Calculate MM energy for same region
       mm_energy = qmmm_validator.calculate_mm_energy(
           structure=frame,
           region="active_site",
           forcefield=forcefield
       )
       
       energy_diff = qm_energy - mm_energy
       print(f"Frame {i}: QM-MM difference = {energy_diff:.1f} kJ/mol")

Validation Reporting
-------------------

Automated Validation Reports
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinmd.validation import ValidationReporter
   
   # Generate comprehensive validation report
   reporter = ValidationReporter()
   
   # Compile all validation results
   validation_data = {
       'structure_validation': basic_issues,
       'equilibration_report': report,
       'trajectory_validation': integrity_report,
       'sampling_analysis': sampling_report,
       'convergence_analysis': block_analysis,
       'experimental_comparison': structural_comparison
   }
   
   # Generate report
   html_report = reporter.generate_html_report(
       validation_data=validation_data,
       output_path="validation_report.html"
   )
   
   print("Validation report generated: validation_report.html")

**Quality Metrics Dashboard**

.. code-block:: python

   from proteinmd.validation import QualityDashboard
   
   # Create quality metrics dashboard
   dashboard = QualityDashboard()
   
   # Add metrics
   dashboard.add_metric("Structure Quality", structural_comparison.quality_score)
   dashboard.add_metric("Sampling Efficiency", sampling_report.efficiency)
   dashboard.add_metric("Energy Conservation", abs(conservation_report.drift_per_ns))
   dashboard.add_metric("Equilibration Status", report.equilibration_score)
   
   # Generate dashboard
   dashboard.create_dashboard("quality_dashboard.html")

Validation Best Practices
-------------------------

Pre-Simulation Checklist
~~~~~~~~~~~~~~~~~~~~~~~

1. **Structure Preparation**
   
   - Validate input structure geometry
   - Check for missing atoms/residues
   - Verify protonation states
   - Validate force field parameters

2. **System Setup**
   
   - Check box size and solvation
   - Validate ion concentrations
   - Verify periodic boundary conditions
   - Test initial energy values

3. **Simulation Parameters**
   
   - Validate timestep choice
   - Check constraint settings
   - Verify thermostat/barostat parameters
   - Test neighbor list settings

During Simulation Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinmd.validation import RealTimeValidator
   
   # Set up real-time validation
   rt_validator = RealTimeValidator()
   rt_validator.add_check("energy_drift", threshold=1.0)  # kJ/mol/ns
   rt_validator.add_check("temperature_drift", threshold=5.0)  # K
   rt_validator.add_check("pressure_drift", threshold=10.0)  # bar
   
   # Add to simulation
   simulation.add_validator(rt_validator)
   
   # Validator will automatically stop simulation if issues detected

Post-Simulation Analysis
~~~~~~~~~~~~~~~~~~~~~~

1. **Trajectory Quality**
   
   - Check for structural anomalies
   - Validate sampling adequacy
   - Analyze convergence properties

2. **Statistical Analysis**
   
   - Perform error analysis
   - Check correlation times
   - Validate statistical significance

3. **Physical Validation**
   
   - Compare with experimental data
   - Cross-validate with literature
   - Check thermodynamic consistency

Common Validation Issues
-----------------------

Structure Problems
~~~~~~~~~~~~~~~~~

**Issue: High Initial Energy**

.. code-block:: python

   # Diagnostic and fix
   if energy_report.energy_per_atom > 100:  # kJ/mol
       print("High initial energy detected")
       
       # Perform more aggressive minimization
       simulation.minimize_energy(
           max_iterations=10000,
           convergence_tolerance=1.0
       )
       
       # Check for close contacts
       close_contacts = validator.find_close_contacts(protein, threshold=0.2)
       if close_contacts:
           print(f"Found {len(close_contacts)} close contacts")

**Issue: Simulation Instability**

.. code-block:: python

   # Stabilization protocol
   stabilizer = SimulationStabilizer()
   
   if simulation.has_instability():
       # Apply stabilization measures
       stabilizer.reduce_timestep(factor=0.5)
       stabilizer.increase_constraints()
       stabilizer.add_position_restraints(selection="protein", force=100.0)
       
       # Re-equilibrate
       simulation.run_stabilization_protocol()

See Also
--------

* :doc:`../api/structure` - Structure handling API
* :doc:`../api/core` - Core simulation API
* :doc:`troubleshooting` - Troubleshooting guide
* :doc:`../user_guide/tutorials` - Simulation tutorials
