Examples
========

This page provides practical examples for common molecular dynamics tasks using ProteinMD.

.. contents:: Example Categories
   :local:
   :depth: 2

Basic Examples
--------------

Protein Structure Loading and Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load a protein structure from a PDB file and perform basic validation:

.. code-block:: python

   from proteinmd.structure import ProteinStructure
   from proteinmd.structure.validation import StructureValidator
   
   # Load protein structure
   protein = ProteinStructure.from_pdb("examples/1ubq.pdb")
   
   print(f"Protein: {protein.name}")
   print(f"Chains: {len(protein.chains)}")
   print(f"Residues: {len(protein.residues)}")
   print(f"Atoms: {len(protein.atoms)}")
   
   # Basic structure validation
   validator = StructureValidator()
   issues = validator.validate(protein)
   
   if issues:
       print("\nStructure validation issues:")
       for issue in issues:
           print(f"  - {issue.type}: {issue.description}")
   else:
       print("\nStructure validation passed!")
   
   # Get basic properties
   print(f"\nBounding box: {protein.get_bounding_box()}")
   print(f"Center of mass: {protein.get_center_of_mass()}")
   print(f"Radius of gyration: {protein.get_radius_of_gyration():.2f} nm")

Simple Energy Minimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform energy minimization on a protein structure:

.. code-block:: python

   from proteinmd.core import MDSimulation, VelocityVerletIntegrator
   from proteinmd.forcefield import AmberFF14SB
   from proteinmd.environment import ImplicitSolvent
   
   # Set up system
   forcefield = AmberFF14SB()
   solvent = ImplicitSolvent(model="gb")
   
   system = forcefield.create_system(protein)
   system.add_solvent_model(solvent)
   
   # Create simulation
   integrator = VelocityVerletIntegrator(timestep=0.001)
   simulation = MDSimulation(system=system, integrator=integrator)
   
   # Energy minimization
   print("Initial energy:", simulation.get_potential_energy())
   
   simulation.minimize_energy(
       max_iterations=1000,
       convergence_tolerance=10.0  # kJ/mol/nm
   )
   
   print("Final energy:", simulation.get_potential_energy())
   
   # Save minimized structure
   minimized_coords = simulation.get_coordinates()
   protein.set_coordinates(minimized_coords)
   protein.to_pdb("minimized_structure.pdb")

Quick MD Simulation
~~~~~~~~~~~~~~~~~~

Run a short molecular dynamics simulation:

.. code-block:: python

   from proteinmd.core import LangevinThermostat
   
   # Add thermostat for dynamics
   thermostat = LangevinThermostat(temperature=300.0, friction=1.0)
   simulation.add_thermostat(thermostat)
   
   # Heat up gradually
   for temp in [50, 100, 150, 200, 250, 300]:
       simulation.set_temperature(temp)
       simulation.run(steps=1000)  # 1 ps at each temperature
       print(f"Temperature: {temp} K, Current T: {simulation.get_temperature():.1f} K")
   
   # Production run
   simulation.run(
       steps=100000,  # 100 ps
       output_frequency=1000  # Save every 1 ps
   )
   
   # Save trajectory
   simulation.save_trajectory("short_md.dcd")
   print("Simulation completed!")

Intermediate Examples
--------------------

Protein in Explicit Water
~~~~~~~~~~~~~~~~~~~~~~~~~

Set up and run a protein simulation in explicit water:

.. code-block:: python

   from proteinmd.environment import WaterModel, PeriodicBoundary
   from proteinmd.core import BerendsenBarostat
   import numpy as np
   
   # Create water box
   water_model = WaterModel("TIP3P")
   
   # Determine box size (protein + 1.5 nm padding)
   protein_box = protein.get_bounding_box()
   box_size = protein_box + 3.0  # 3 nm total padding
   
   print(f"Protein bounding box: {protein_box}")
   print(f"Water box size: {box_size}")
   
   # Set up periodic boundaries
   boundary = PeriodicBoundary(box_size)
   
   # Solvate the protein
   solvated_system = water_model.solvate(protein, boundary)
   print(f"Added {solvated_system.n_water_molecules} water molecules")
   
   # Add counter ions
   solvated_system.add_ions(
       positive_ion="Na+",
       negative_ion="Cl-",
       concentration=0.15  # 150 mM
   )
   print(f"Added {solvated_system.n_positive_ions} Na+ and "
         f"{solvated_system.n_negative_ions} Cl- ions")
   
   # Create system with force field
   forcefield = AmberFF14SB()
   system = forcefield.create_system(solvated_system)
   
   # Set up non-bonded interactions
   system.set_nonbonded_cutoff(1.0)  # 1 nm
   system.set_pme_parameters(
       alpha=0.31,
       grid_spacing=0.12
   )
   
   # Set up simulation
   integrator = VelocityVerletIntegrator(timestep=0.002)  # 2 fs
   thermostat = LangevinThermostat(temperature=300.0, friction=1.0)
   barostat = BerendsenBarostat(pressure=1.0, tau=1.0)
   
   simulation = MDSimulation(
       system=system,
       integrator=integrator,
       thermostat=thermostat,
       barostat=barostat
   )
   
   # Equilibration protocol
   print("Starting equilibration...")
   
   # 1. Energy minimization
   simulation.minimize_energy(max_iterations=2000)
   print("Energy minimization completed")
   
   # 2. NVT equilibration (constant volume)
   simulation.run_nvt(
       steps=25000,      # 50 ps
       temperature=300.0
   )
   print("NVT equilibration completed")
   
   # 3. NPT equilibration (constant pressure)
   simulation.run_npt(
       steps=50000,      # 100 ps
       temperature=300.0,
       pressure=1.0
   )
   print("NPT equilibration completed")
   
   # 4. Production run
   print("Starting production run...")
   simulation.run_npt(
       steps=1000000,    # 2 ns
       temperature=300.0,
       pressure=1.0,
       output_frequency=1000
   )
   
   # Save final state
   simulation.save_trajectory("production.dcd")
   simulation.save_checkpoint("final_state.chk")
   print("Production run completed!")

Multiple Chain Analysis
~~~~~~~~~~~~~~~~~~~~~~

Analyze a multi-chain protein complex:

.. code-block:: python

   from proteinmd.structure import ProteinComplex
   from proteinmd.analysis import InterfaceAnalysis, ContactMap
   
   # Load protein complex
   complex_structure = ProteinComplex.from_pdb("complex.pdb")
   
   print(f"Complex contains {len(complex_structure.chains)} chains:")
   for chain in complex_structure.chains:
       print(f"  Chain {chain.id}: {len(chain.residues)} residues")
   
   # Analyze interfaces between chains
   interface_analyzer = InterfaceAnalysis()
   
   for i, chain_a in enumerate(complex_structure.chains):
       for chain_b in complex_structure.chains[i+1:]:
           interface = interface_analyzer.calculate_interface(chain_a, chain_b)
           
           if interface.area > 500:  # Significant interface (Ų)
               print(f"\nInterface {chain_a.id}-{chain_b.id}:")
               print(f"  Contact area: {interface.area:.1f} Ų")
               print(f"  Contact residues: {len(interface.residues)}")
               print(f"  Hydrogen bonds: {len(interface.hydrogen_bonds)}")
               print(f"  Salt bridges: {len(interface.salt_bridges)}")
   
   # Create contact map for the complex
   contact_map = ContactMap(cutoff=0.8)  # 8 Å cutoff
   contacts = contact_map.calculate(complex_structure)
   
   # Visualize contact map
   contact_map.plot(contacts, save_path="complex_contacts.png")

Advanced Examples
----------------

Free Energy Calculations
~~~~~~~~~~~~~~~~~~~~~~~~

Calculate binding free energy using thermodynamic integration:

.. code-block:: python

   from proteinmd.sampling import ThermodynamicIntegration
   from proteinmd.analysis import FreeEnergyAnalysis
   
   # Set up protein-ligand system
   complex_system = setup_protein_ligand_complex("protein.pdb", "ligand.mol2")
   
   # Define λ schedule for TI
   lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
   
   # Set up thermodynamic integration
   ti_calc = ThermodynamicIntegration(
       complex_system=complex_system,
       lambda_schedule=lambda_values,
       perturbation_type="electrostatic_then_vdw"
   )
   
   # Run TI calculations
   print("Running thermodynamic integration...")
   ti_results = []
   
   for i, lambda_val in enumerate(lambda_values):
       print(f"λ = {lambda_val:.1f}")
       
       # Set up simulation for this λ value
       system = ti_calc.create_lambda_system(lambda_val)
       simulation = MDSimulation(system=system, integrator=integrator)
       
       # Equilibration
       simulation.run(steps=100000)  # 200 ps
       
       # Production with energy collection
       dhdl_values = []
       for step in range(500000):  # 1 ns
           simulation.step()
           if step % 100 == 0:  # Collect every 200 fs
               dhdl = ti_calc.calculate_dhdl(simulation.get_state())
               dhdl_values.append(dhdl)
       
       ti_results.append({
           'lambda': lambda_val,
           'dhdl_mean': np.mean(dhdl_values),
           'dhdl_std': np.std(dhdl_values),
           'dhdl_values': dhdl_values
       })
   
   # Analyze results
   fe_analyzer = FreeEnergyAnalysis()
   binding_energy = fe_analyzer.integrate_ti(ti_results)
   
   print(f"\nBinding free energy: {binding_energy:.2f} ± {fe_analyzer.error:.2f} kJ/mol")
   
   # Plot TI integrand
   fe_analyzer.plot_ti_integrand(ti_results, save_path="ti_integrand.png")

Enhanced Sampling with Metadynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use metadynamics to enhance sampling of conformational transitions:

.. code-block:: python

   from proteinmd.sampling import Metadynamics
   from proteinmd.analysis import ReactionCoordinate
   
   # Define collective variables (CVs)
   cv1 = ReactionCoordinate.dihedral(
       atoms=["PHE8:CA", "PHE8:CB", "PHE8:CG", "PHE8:CD1"],
       name="phe8_chi1"
   )
   
   cv2 = ReactionCoordinate.distance(
       atom1="GLU15:CG",
       atom2="LYS27:NZ",
       name="salt_bridge"
   )
   
   # Set up metadynamics
   metad = Metadynamics(
       collective_variables=[cv1, cv2],
       hill_height=1.2,      # kJ/mol
       hill_width=[0.1, 0.05],  # Gaussian widths
       deposition_frequency=500  # Add hill every 500 steps (1 ps)
   )
   
   # Add metadynamics bias to simulation
   simulation.add_bias(metad)
   
   # Run metadynamics simulation
   print("Running metadynamics simulation...")
   simulation.run(
       steps=5000000,  # 10 ns
       output_frequency=1000
   )
   
   # Analyze results
   cv_trajectory = metad.get_cv_trajectory()
   bias_surface = metad.get_bias_surface()
   
   # Calculate free energy surface
   fes = metad.calculate_fes(
       cv_range=[(−π, π), (0.2, 1.5)],  # CV ranges
       grid_size=[100, 75]
   )
   
   # Plot free energy surface
   import matplotlib.pyplot as plt
   
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
   
   # CV trajectory
   ax1.plot(cv_trajectory[:, 0], cv_trajectory[:, 1], alpha=0.6)
   ax1.set_xlabel('φ (PHE8 χ1)')
   ax1.set_ylabel('Distance (GLU15-LYS27)')
   ax1.set_title('CV Trajectory')
   
   # Free energy surface
   im = ax2.contourf(fes.cv1_grid, fes.cv2_grid, fes.values, levels=20)
   ax2.set_xlabel('φ (PHE8 χ1)')
   ax2.set_ylabel('Distance (GLU15-LYS27)')
   ax2.set_title('Free Energy Surface')
   plt.colorbar(im, ax=ax2, label='Free Energy (kJ/mol)')
   
   plt.tight_layout()
   plt.savefig('metadynamics_analysis.png', dpi=300)
   plt.show()

Machine Learning Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

Use machine learning to analyze conformational states:

.. code-block:: python

   from proteinmd.analysis import ConformationalClustering, FeatureExtraction
   from sklearn.cluster import KMeans
   from sklearn.decomposition import PCA
   import numpy as np
   
   # Load trajectory
   trajectory = simulation.get_trajectory()
   
   # Extract features for clustering
   feature_extractor = FeatureExtraction()
   
   # Combine multiple feature types
   features = []
   
   # 1. Backbone dihedrals
   phi_psi = feature_extractor.extract_backbone_dihedrals(trajectory)
   features.append(phi_psi)
   
   # 2. Distance features
   ca_distances = feature_extractor.extract_ca_distances(trajectory)
   features.append(ca_distances)
   
   # 3. Secondary structure content
   ss_content = feature_extractor.extract_ss_content(trajectory)
   features.append(ss_content)
   
   # Combine and normalize features
   X = np.hstack(features)
   X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
   
   # Dimensionality reduction with PCA
   pca = PCA(n_components=10)
   X_pca = pca.fit_transform(X_normalized)
   
   print(f"Explained variance ratio (first 10 PCs): "
         f"{pca.explained_variance_ratio_.sum():.3f}")
   
   # Clustering analysis
   clustering = ConformationalClustering()
   
   # Determine optimal number of clusters
   n_clusters_range = range(2, 11)
   silhouette_scores = []
   
   for n_clusters in n_clusters_range:
       kmeans = KMeans(n_clusters=n_clusters, random_state=42)
       cluster_labels = kmeans.fit_predict(X_pca)
       score = clustering.silhouette_score(X_pca, cluster_labels)
       silhouette_scores.append(score)
   
   optimal_clusters = n_clusters_range[np.argmax(silhouette_scores)]
   print(f"Optimal number of clusters: {optimal_clusters}")
   
   # Final clustering
   kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
   cluster_labels = kmeans.fit_predict(X_pca)
   
   # Analyze clusters
   for cluster_id in range(optimal_clusters):
       cluster_frames = np.where(cluster_labels == cluster_id)[0]
       cluster_size = len(cluster_frames)
       cluster_percentage = (cluster_size / len(trajectory)) * 100
       
       print(f"Cluster {cluster_id + 1}: {cluster_size} frames ({cluster_percentage:.1f}%)")
       
       # Get representative structure (centroid)
       cluster_data = X_pca[cluster_frames]
       centroid_idx = cluster_frames[
           np.argmin(np.sum((cluster_data - kmeans.cluster_centers_[cluster_id])**2, axis=1))
       ]
       
       # Save representative structure
       representative_frame = trajectory[centroid_idx]
       protein.set_coordinates(representative_frame.coordinates)
       protein.to_pdb(f"cluster_{cluster_id + 1}_representative.pdb")
   
   # Visualize clustering results
   import matplotlib.pyplot as plt
   
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
   
   # PCA projection with clusters
   scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                        cmap='tab10', alpha=0.6)
   ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
   ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
   ax1.set_title('Conformational Clusters')
   plt.colorbar(scatter, ax=ax1, label='Cluster')
   
   # Cluster population over time
   time_axis = np.arange(len(cluster_labels)) * 0.002  # 2 fs timestep
   ax2.plot(time_axis, cluster_labels, alpha=0.7)
   ax2.set_xlabel('Time (ns)')
   ax2.set_ylabel('Cluster ID')
   ax2.set_title('Cluster Evolution')
   
   plt.tight_layout()
   plt.savefig('clustering_analysis.png', dpi=300)
   plt.show()

Specialized Applications
-----------------------

Membrane Protein Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulate a membrane protein in a lipid bilayer:

.. code-block:: python

   from proteinmd.environment import LipidBilayer, MembraneSystem
   
   # Load membrane protein structure
   membrane_protein = ProteinStructure.from_pdb("membrane_protein.pdb")
   
   # Set up lipid bilayer
   bilayer = LipidBilayer(
       lipid_type="POPC",          # Phospholipid type
       area_per_lipid=0.68,        # nm²
       protein=membrane_protein,
       water_thickness=1.5         # nm on each side
   )
   
   # Create membrane system
   membrane_system = MembraneSystem(
       protein=membrane_protein,
       bilayer=bilayer
   )
   
   # Add ions and water
   membrane_system.add_water("TIP3P")
   membrane_system.add_ions(concentration=0.15)
   
   print(f"System composition:")
   print(f"  Protein atoms: {membrane_system.n_protein_atoms}")
   print(f"  Lipid molecules: {membrane_system.n_lipids}")
   print(f"  Water molecules: {membrane_system.n_water}")
   print(f"  Ions: {membrane_system.n_ions}")
   
   # Set up simulation with membrane-specific settings
   forcefield = AmberFF14SB()
   system = forcefield.create_system(membrane_system)
   
   # Use anisotropic pressure coupling for membranes
   from proteinmd.core import AnisotropicBarostat
   barostat = AnisotropicBarostat(
       pressure_xy=1.0,    # Bar, lateral pressure
       pressure_z=1.0,     # Bar, normal pressure
       compressibility_xy=4.5e-5,
       compressibility_z=4.5e-5
   )
   
   simulation = MDSimulation(
       system=system,
       integrator=VelocityVerletIntegrator(timestep=0.002),
       thermostat=LangevinThermostat(temperature=310.0, friction=1.0),
       barostat=barostat
   )
   
   # Run membrane equilibration protocol
   print("Running membrane equilibration...")
   
   # 1. Minimize with positional restraints on protein
   simulation.add_restraints(selection="protein", force_constant=1000.0)
   simulation.minimize_energy(max_iterations=5000)
   
   # 2. Equilibrate lipids with protein restrained
   simulation.run(steps=250000)  # 500 ps
   
   # 3. Gradually release protein restraints
   for force_constant in [500.0, 100.0, 10.0, 0.0]:
       simulation.update_restraints(force_constant=force_constant)
       simulation.run(steps=50000)  # 100 ps each
   
   # 4. Production run
   simulation.run(
       steps=10000000,  # 20 ns
       output_frequency=5000
   )

Drug Design Workflow
~~~~~~~~~~~~~~~~~~~

Complete workflow for structure-based drug design:

.. code-block:: python

   from proteinmd.structure import BindingSite
   from proteinmd.docking import AutoDock, DrugLikeFilter
   from proteinmd.analysis import BindingAffinityCalculation
   
   # Load target protein
   target_protein = ProteinStructure.from_pdb("target.pdb")
   
   # Identify and prepare binding site
   binding_site = BindingSite.from_coordinates(
       protein=target_protein,
       center=[25.0, 30.0, 15.0],  # Å
       radius=10.0
   )
   
   # Prepare protein for docking
   binding_site.add_hydrogens()
   binding_site.optimize_sidechains()
   
   # Load compound library
   compound_library = "chembl_fragments.sdf"
   
   # Filter compounds for drug-likeness
   drug_filter = DrugLikeFilter()
   filtered_compounds = drug_filter.filter_library(
       compound_library,
       rules=["lipinski", "veber", "pains"]
   )
   
   print(f"Filtered library: {len(filtered_compounds)} compounds")
   
   # Virtual screening with docking
   docker = AutoDock(
       receptor=binding_site,
       search_space=binding_site.get_search_space(),
       exhaustiveness=16
   )
   
   docking_results = []
   for i, compound in enumerate(filtered_compounds):
       if i % 100 == 0:
           print(f"Docking compound {i+1}/{len(filtered_compounds)}")
       
       result = docker.dock(compound)
       if result.score < -6.0:  # Good binding score threshold
           docking_results.append(result)
   
   # Sort by docking score
   docking_results.sort(key=lambda x: x.score)
   top_hits = docking_results[:50]  # Top 50 compounds
   
   print(f"Found {len(top_hits)} promising compounds")
   
   # Validate top hits with MD simulations
   binding_calculator = BindingAffinityCalculation()
   
   for i, hit in enumerate(top_hits[:10]):  # Validate top 10
       print(f"Validating compound {i+1}: {hit.compound_id}")
       
       # Create protein-ligand complex
       complex_structure = binding_site.create_complex(hit.pose)
       
       # Set up MD simulation
       complex_system = forcefield.create_system(complex_structure)
       simulation = MDSimulation(system=complex_system, integrator=integrator)
       
       # Short equilibration and production
       simulation.minimize_energy(max_iterations=1000)
       simulation.run(steps=1000000)  # 2 ns
       
       # Calculate binding affinity
       binding_energy = binding_calculator.calculate_mm_pbsa(
           trajectory=simulation.get_trajectory(),
           complex_structure=complex_structure
       )
       
       print(f"  Binding energy: {binding_energy:.2f} kJ/mol")
       
       # Update results
       hit.md_binding_energy = binding_energy
       hit.md_trajectory = simulation.get_trajectory()

See Also
--------

* :doc:`tutorials` - Step-by-step tutorials
* :doc:`quick_start` - Getting started guide
* :doc:`../api/index` - Complete API reference
* :doc:`../advanced/performance` - Performance optimization
* :doc:`../developer/contributing` - Contributing guidelines
