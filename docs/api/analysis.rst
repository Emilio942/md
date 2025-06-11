Analysis Tools
==============

The :mod:`proteinMD.analysis` module provides comprehensive analysis tools for molecular dynamics trajectory data, including structural analysis, energetic analysis, and conformational analysis.

.. currentmodule:: proteinMD.analysis

Overview
--------

The analysis module includes:

- **RMSD Analysis**: Root Mean Square Deviation calculations and trajectory comparison
- **Ramachandran Plots**: Phi-Psi angle analysis for protein backbone conformations
- **Radius of Gyration**: Protein compactness analysis over time
- **Secondary Structure**: α-helix and β-sheet content tracking
- **Hydrogen Bond Analysis**: H-bond network identification and dynamics
- **Distance Analysis**: Pairwise distances and contact maps
- **Principal Component Analysis**: Conformational clustering and dynamics
- **Free Energy Landscapes**: Energy surface construction and analysis

Quick Example
-------------

Basic trajectory analysis:

.. code-block:: python

   from proteinMD.analysis.rmsd import RMSDAnalyzer
   from proteinMD.analysis.radius_of_gyration import RadiusOfGyrationAnalyzer
   from proteinMD.structure.pdb_parser import PDBParser
   import numpy as np
   
   # Load reference structure
   parser = PDBParser()
   protein = parser.parse("protein.pdb")
   
   # Load trajectory
   trajectory = np.load("trajectory.npz")
   
   # RMSD analysis
   rmsd_analyzer = RMSDAnalyzer(protein)
   rmsd_values = rmsd_analyzer.calculate_trajectory(trajectory)
   
   # Radius of gyration analysis
   rg_analyzer = RadiusOfGyrationAnalyzer(protein)
   rg_values = rg_analyzer.calculate_trajectory(trajectory)
   
   print(f"Average RMSD: {np.mean(rmsd_values):.3f} nm")
   print(f"Average Rg: {np.mean(rg_values):.3f} nm")

RMSD Analysis
-------------

.. automodule:: proteinMD.analysis.rmsd
   :members:
   :undoc-members:
   :show-inheritance:

RMSD Calculator
~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.analysis.rmsd.RMSDAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Examples**
   
   Basic RMSD calculation:
   
   .. code-block:: python
   
      from proteinMD.analysis.rmsd import RMSDAnalyzer
      
      # Initialize RMSD analyzer
      rmsd_analyzer = RMSDAnalyzer(
          reference_structure=protein,
          selection="backbone",      # Analyze backbone atoms only
          align=True                # Align structures before RMSD
      )
      
      # Calculate RMSD for single frame
      rmsd = rmsd_analyzer.calculate_frame(trajectory_frame)
      
      # Calculate RMSD for entire trajectory
      rmsd_trajectory = rmsd_analyzer.calculate_trajectory(trajectory)
      
      # Save results
      rmsd_analyzer.save_results("rmsd_analysis.json")
      rmsd_analyzer.save_plot("rmsd_plot.png")
   
   Advanced RMSD analysis:
   
   .. code-block:: python
   
      # Custom atom selection
      rmsd_analyzer = RMSDAnalyzer(
          reference_structure=protein,
          selection="name CA",       # Only Cα atoms
          align_selection="backbone", # Align using backbone
          mass_weighted=True         # Mass-weighted RMSD
      )
      
      # Per-residue RMSD
      per_residue_rmsd = rmsd_analyzer.calculate_per_residue_rmsd(trajectory)
      
      # RMSD matrix (all vs all frames)
      rmsd_matrix = rmsd_analyzer.calculate_rmsd_matrix(trajectory)
      
      # Clustering based on RMSD
      clusters = rmsd_analyzer.cluster_conformations(
          trajectory, 
          rmsd_cutoff=0.2  # 0.2 nm cutoff
      )

.. autoclass:: proteinMD.analysis.rmsd.RMSDCalculator
   :members:
   :undoc-members:
   :show-inheritance:

Radius of Gyration Analysis
---------------------------

.. automodule:: proteinMD.analysis.radius_of_gyration
   :members:
   :undoc-members:
   :show-inheritance:

Radius of Gyration Calculator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.analysis.radius_of_gyration.RadiusOfGyrationAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Examples**
   
   Basic Rg calculation:
   
   .. code-block:: python
   
      from proteinMD.analysis.radius_of_gyration import RadiusOfGyrationAnalysis
      
      # Initialize Rg analyzer
      rg_analyzer = RadiusOfGyrationAnalysis(
          protein=protein,
          selection="protein",       # All protein atoms
          mass_weighted=True         # Use atomic masses
      )
      
      # Calculate Rg for trajectory
      rg_values = rg_analyzer.calculate_trajectory(trajectory)
      
      # Calculate components (Rgx, Rgy, Rgz)
      rg_components = rg_analyzer.calculate_components(trajectory)
      
      # Save results
      rg_analyzer.save_results("rg_analysis.json")
      rg_analyzer.plot_time_series("rg_timeseries.png")
   
   Advanced Rg analysis:
   
   .. code-block:: python
   
      # Per-chain analysis
      rg_analyzer = RadiusOfGyrationAnalysis(protein)
      
      for chain in protein.chains:
          chain_rg = rg_analyzer.calculate_chain_rg(trajectory, chain.id)
          print(f"Chain {chain.id} Rg: {np.mean(chain_rg):.3f} nm")
      
      # Secondary structure-specific Rg
      helix_rg = rg_analyzer.calculate_selection_rg(
          trajectory, 
          selection="resname ALA VAL LEU ILE and name CA"
      )

Ramachandran Analysis
---------------------

.. automodule:: proteinMD.analysis.ramachandran
   :members:
   :undoc-members:
   :show-inheritance:

Ramachandran Plot Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.analysis.ramachandran.RamachandranAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Examples**
   
   Basic Ramachandran analysis:
   
   .. code-block:: python
   
      from proteinMD.analysis.ramachandran import RamachandranAnalysis
      
      # Initialize Ramachandran analyzer
      rama_analyzer = RamachandranAnalysis(
          protein=protein,
          exclude_proline=True,      # Exclude proline residues
          exclude_glycine=False      # Include glycine
      )
      
      # Calculate phi-psi angles for trajectory
      phi_psi_angles = rama_analyzer.calculate_trajectory(trajectory)
      
      # Generate Ramachandran plot
      rama_analyzer.plot_ramachandran("ramachandran.png")
      
      # Color by residue type
      rama_analyzer.plot_ramachandran(
          "ramachandran_colored.png",
          color_by="residue_type"
      )
      
      # Calculate outliers
      outliers = rama_analyzer.find_outliers(phi_psi_angles)
      print(f"Found {len(outliers)} Ramachandran outliers")
   
   Advanced Ramachandran analysis:
   
   .. code-block:: python
   
      # Time-resolved analysis
      rama_analyzer = RamachandranAnalysis(protein)
      
      # Analyze conformational transitions
      transitions = rama_analyzer.analyze_transitions(trajectory)
      
      # Calculate residence times in different regions
      residence_times = rama_analyzer.calculate_residence_times(
          trajectory,
          regions=["alpha", "beta", "left_handed"]
      )
      
      # Generate animation of phi-psi evolution
      rama_analyzer.animate_trajectory("rama_evolution.gif", trajectory)

Secondary Structure Analysis
----------------------------

.. automodule:: proteinMD.analysis.secondary_structure
   :members:
   :undoc-members:
   :show-inheritance:

Secondary Structure Analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.analysis.secondary_structure.SecondaryStructureAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Examples**
   
   Basic secondary structure analysis:
   
   .. code-block:: python
   
      from proteinMD.analysis.secondary_structure import SecondaryStructureAnalysis
      
      # Initialize SS analyzer
      ss_analyzer = SecondaryStructureAnalysis(
          protein=protein,
          method="dssp",             # DSSP-like algorithm
          hydrogen_bond_cutoff=0.35  # H-bond distance cutoff (nm)
      )
      
      # Analyze trajectory
      ss_trajectory = ss_analyzer.analyze_trajectory(trajectory)
      
      # Calculate secondary structure content
      ss_content = ss_analyzer.calculate_content(ss_trajectory)
      print(f"Helix content: {ss_content['helix']:.1%}")
      print(f"Sheet content: {ss_content['sheet']:.1%}")
      print(f"Coil content: {ss_content['coil']:.1%}")
      
      # Plot secondary structure evolution
      ss_analyzer.plot_evolution("ss_evolution.png", ss_trajectory)
   
   Advanced secondary structure analysis:
   
   .. code-block:: python
   
      # Per-residue analysis
      ss_analyzer = SecondaryStructureAnalysis(protein)
      
      # Calculate secondary structure propensities
      propensities = ss_analyzer.calculate_propensities(trajectory)
      
      # Identify stable secondary structure elements
      stable_elements = ss_analyzer.find_stable_elements(
          trajectory,
          min_length=4,              # Minimum 4 residues
          stability_threshold=0.8    # 80% stable
      )
      
      # Analyze helix-coil transitions
      transitions = ss_analyzer.analyze_helix_coil_transitions(trajectory)

Hydrogen Bond Analysis
----------------------

.. automodule:: proteinMD.analysis.hydrogen_bonds
   :members:
   :undoc-members:
   :show-inheritance:

Hydrogen Bond Analyzer
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.analysis.hydrogen_bonds.HydrogenBondAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Examples**
   
   Basic hydrogen bond analysis:
   
   .. code-block:: python
   
      from proteinMD.analysis.hydrogen_bonds import HydrogenBondAnalysis
      
      # Initialize H-bond analyzer
      hb_analyzer = HydrogenBondAnalysis(
          protein=protein,
          distance_cutoff=0.35,      # 3.5 Å distance cutoff
          angle_cutoff=30.0,         # 30° angle cutoff
          include_water=True         # Include protein-water H-bonds
      )
      
      # Analyze hydrogen bonds in trajectory
      hbonds = hb_analyzer.analyze_trajectory(trajectory)
      
      # Calculate H-bond statistics
      stats = hb_analyzer.calculate_statistics(hbonds)
      print(f"Average H-bonds: {stats['average']:.1f}")
      print(f"Max H-bonds: {stats['maximum']}")
      print(f"Min H-bonds: {stats['minimum']}")
      
      # Plot H-bond time series
      hb_analyzer.plot_time_series("hbonds_timeseries.png", hbonds)
   
   Advanced hydrogen bond analysis:
   
   .. code-block:: python
   
      # Specific H-bond analysis
      hb_analyzer = HydrogenBondAnalysis(protein)
      
      # Analyze intramolecular H-bonds
      intra_hbonds = hb_analyzer.analyze_intramolecular_hbonds(trajectory)
      
      # Calculate H-bond lifetimes
      lifetimes = hb_analyzer.calculate_lifetimes(trajectory)
      
      # Network analysis
      network = hb_analyzer.build_hbond_network(trajectory)
      centrality = hb_analyzer.calculate_network_centrality(network)
      
      # Visualize H-bond network
      hb_analyzer.visualize_network("hbond_network.png", network)

Distance Analysis
-----------------

.. automodule:: proteinMD.analysis.distances
   :members:
   :undoc-members:
   :show-inheritance:

Distance Calculator
~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.analysis.distances.DistanceAnalysis
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Examples**
   
   Basic distance analysis:
   
   .. code-block:: python
   
      from proteinMD.analysis.distances import DistanceAnalysis
      
      # Initialize distance analyzer
      dist_analyzer = DistanceAnalysis(protein)
      
      # Calculate specific distance over trajectory
      distance = dist_analyzer.calculate_distance(
          trajectory,
          atom1_id=10,   # First atom index
          atom2_id=50    # Second atom index
      )
      
      # Calculate contact map
      contact_map = dist_analyzer.calculate_contact_map(
          trajectory,
          cutoff=0.8     # 8 Å contact cutoff
      )
      
      # Plot distance evolution
      dist_analyzer.plot_distance("distance_evolution.png", distance)
   
   Advanced distance analysis:
   
   .. code-block:: python
   
      # Multiple distance analysis
      dist_analyzer = DistanceAnalysis(protein)
      
      # Define distance pairs
      distance_pairs = [
          (10, 50),   # Distance 1
          (20, 60),   # Distance 2
          (30, 70)    # Distance 3
      ]
      
      # Calculate all distances
      distances = dist_analyzer.calculate_multiple_distances(
          trajectory, distance_pairs
      )
      
      # Cross-correlation analysis
      correlations = dist_analyzer.calculate_distance_correlations(distances)
      
      # Dynamic contact analysis
      dynamic_contacts = dist_analyzer.analyze_dynamic_contacts(
          trajectory,
          cutoff=0.8,
          min_lifetime=10  # Contact must exist for 10 frames
      )

Principal Component Analysis
----------------------------

.. automodule:: proteinMD.analysis.pca
   :members:
   :undoc-members:
   :show-inheritance:

PCA Analyzer
~~~~~~~~~~~~

.. autoclass:: proteinMD.analysis.pca.PCAAnalysis
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Examples**
   
   Basic PCA analysis:
   
   .. code-block:: python
   
      from proteinMD.analysis.pca import PCAAnalysis
      
      # Initialize PCA analyzer
      pca_analyzer = PCAAnalysis(
          protein=protein,
          selection="name CA",       # Use Cα atoms
          align=True                 # Align structures first
      )
      
      # Perform PCA on trajectory
      pca_results = pca_analyzer.fit_transform(trajectory)
      
      # Access results
      eigenvalues = pca_results.eigenvalues
      eigenvectors = pca_results.eigenvectors
      projections = pca_results.projections
      
      # Plot PC projections
      pca_analyzer.plot_projections("pca_projections.png", pca_results)
      
      # Calculate explained variance
      explained_variance = pca_analyzer.calculate_explained_variance(eigenvalues)
      print(f"PC1 explains {explained_variance[0]:.1%} of variance")
   
   Advanced PCA analysis:
   
   .. code-block:: python
   
      # Advanced PCA with clustering
      pca_analyzer = PCAAnalysis(protein)
      pca_results = pca_analyzer.fit_transform(trajectory)
      
      # Cluster conformations in PC space
      clusters = pca_analyzer.cluster_pc_space(
          pca_results.projections,
          n_clusters=3
      )
      
      # Generate PC mode animations
      pca_analyzer.animate_pc_mode(
          "pc1_mode.pdb",
          pca_results,
          mode=0,        # First principal component
          amplitude=3.0  # Animation amplitude
      )
      
      # Essential dynamics analysis
      essential_space = pca_analyzer.calculate_essential_subspace(
          pca_results,
          n_components=10  # First 10 PCs
      )

Free Energy Analysis
--------------------

.. automodule:: proteinMD.analysis.free_energy
   :members:
   :undoc-members:
   :show-inheritance:

Free Energy Calculator
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.analysis.free_energy.FreeEnergyAnalysis
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Examples**
   
   Basic free energy analysis:
   
   .. code-block:: python
   
      from proteinMD.analysis.free_energy import FreeEnergyAnalysis
      
      # Initialize free energy analyzer
      fe_analyzer = FreeEnergyAnalysis(
          temperature=300.0,         # Temperature in Kelvin
          kT=2.494                   # kT in kJ/mol at 300K
      )
      
      # Calculate 1D free energy profile
      coordinate = radius_of_gyration_values  # Some reaction coordinate
      free_energy = fe_analyzer.calculate_1d_profile(
          coordinate,
          n_bins=50,
          range=(0.8, 1.2)  # Rg range in nm
      )
      
      # Plot free energy profile
      fe_analyzer.plot_1d_profile("free_energy_1d.png", free_energy)
   
   Advanced free energy analysis:
   
   .. code-block:: python
   
      # 2D free energy landscape
      fe_analyzer = FreeEnergyAnalysis()
      
      # Define two reaction coordinates
      coord1 = rmsd_values
      coord2 = rg_values
      
      # Calculate 2D free energy surface
      free_energy_2d = fe_analyzer.calculate_2d_profile(
          coord1, coord2,
          n_bins=[50, 50],
          ranges=[(0.0, 0.5), (0.8, 1.2)]
      )
      
      # Plot 2D landscape
      fe_analyzer.plot_2d_landscape("free_energy_2d.png", free_energy_2d)
      
      # Find minima and transition states
      minima = fe_analyzer.find_minima(free_energy_2d)
      transition_states = fe_analyzer.find_saddle_points(free_energy_2d)

Analysis Utilities
------------------

.. automodule:: proteinMD.analysis.utils
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: proteinMD.analysis.utils.smooth_data
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.analysis.utils import smooth_data
      
      # Smooth noisy trajectory data
      smoothed_rmsd = smooth_data(
          rmsd_values,
          window_size=10,
          method="savgol"  # Savitzky-Golay filter
      )

.. autofunction:: proteinMD.analysis.utils.calculate_autocorrelation
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.analysis.utils import calculate_autocorrelation
      
      # Calculate autocorrelation function
      autocorr = calculate_autocorrelation(time_series)
      
      # Find correlation time
      correlation_time = find_correlation_time(autocorr)

.. autofunction:: proteinMD.analysis.utils.bootstrap_error
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.analysis.utils import bootstrap_error
      
      # Calculate bootstrap error estimates
      mean_value, error = bootstrap_error(
          data=rmsd_values,
          statistic="mean",
          n_bootstrap=1000
      )

Common Usage Patterns
---------------------

Complete Trajectory Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinMD.analysis import *
   from proteinMD.structure.pdb_parser import PDBParser
   import numpy as np
   
   # Load data
   parser = PDBParser()
   protein = parser.parse("protein.pdb")
   trajectory = np.load("trajectory.npz")
   
   # Initialize analyzers
   rmsd_analyzer = RMSDAnalysis(protein, selection="backbone")
   rg_analyzer = RadiusOfGyrationAnalysis(protein)
   rama_analyzer = RamachandranAnalysis(protein)
   ss_analyzer = SecondaryStructureAnalysis(protein)
   hb_analyzer = HydrogenBondAnalysis(protein)
   
   # Perform analysis
   print("Running trajectory analysis...")
   
   rmsd_values = rmsd_analyzer.calculate_trajectory(trajectory)
   rg_values = rg_analyzer.calculate_trajectory(trajectory)
   phi_psi_angles = rama_analyzer.calculate_trajectory(trajectory)
   ss_data = ss_analyzer.analyze_trajectory(trajectory)
   hbonds = hb_analyzer.analyze_trajectory(trajectory)
   
   # Generate reports
   print(f"Average RMSD: {np.mean(rmsd_values):.3f} ± {np.std(rmsd_values):.3f} nm")
   print(f"Average Rg: {np.mean(rg_values):.3f} ± {np.std(rg_values):.3f} nm")
   
   ss_content = ss_analyzer.calculate_content(ss_data)
   print(f"Secondary structure: {ss_content['helix']:.1%} helix, "
         f"{ss_content['sheet']:.1%} sheet, {ss_content['coil']:.1%} coil")
   
   hb_stats = hb_analyzer.calculate_statistics(hbonds)
   print(f"Average H-bonds: {hb_stats['average']:.1f}")
   
   # Save plots
   rmsd_analyzer.save_plot("rmsd_analysis.png")
   rg_analyzer.plot_time_series("rg_analysis.png")
   rama_analyzer.plot_ramachandran("ramachandran.png")
   ss_analyzer.plot_evolution("ss_evolution.png", ss_data)
   hb_analyzer.plot_time_series("hbonds.png", hbonds)

Conformational Clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinMD.analysis.pca import PCAAnalysis
   from proteinMD.analysis.rmsd import RMSDAnalysis
   
   # PCA-based clustering
   pca_analyzer = PCAAnalysis(protein, selection="name CA")
   pca_results = pca_analyzer.fit_transform(trajectory)
   
   # Cluster in PC space
   clusters = pca_analyzer.cluster_pc_space(
       pca_results.projections,
       n_clusters=5
   )
   
   # RMSD-based clustering
   rmsd_analyzer = RMSDAnalysis(protein)
   rmsd_clusters = rmsd_analyzer.cluster_conformations(
       trajectory,
       rmsd_cutoff=0.2
   )
   
   # Extract representative structures
   representatives = []
   for cluster_id in np.unique(clusters):
       cluster_frames = np.where(clusters == cluster_id)[0]
       centroid_frame = cluster_frames[0]  # Simplified selection
       representatives.append(trajectory[centroid_frame])

Free Energy Landscape Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinMD.analysis.free_energy import FreeEnergyAnalysis
   
   # Calculate reaction coordinates
   rmsd_values = rmsd_analyzer.calculate_trajectory(trajectory)
   rg_values = rg_analyzer.calculate_trajectory(trajectory)
   
   # 2D free energy landscape
   fe_analyzer = FreeEnergyAnalysis(temperature=300.0)
   landscape = fe_analyzer.calculate_2d_profile(
       rmsd_values, rg_values,
       n_bins=[50, 50],
       ranges=[(0.0, 0.5), (0.8, 1.2)]
   )
   
   # Find stable states
   minima = fe_analyzer.find_minima(landscape)
   print(f"Found {len(minima)} stable states")
   
   # Calculate transition barriers
   barriers = fe_analyzer.calculate_barriers(landscape, minima)
   
   # Plot landscape with minima
   fe_analyzer.plot_2d_landscape(
       "free_energy_landscape.png",
       landscape,
       minima=minima
   )

Time-resolved Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze trajectory in time windows
   window_size = 1000  # frames
   n_windows = len(trajectory) // window_size
   
   time_resolved_data = {
       'rmsd': [],
       'rg': [],
       'ss_content': []
   }
   
   for i in range(n_windows):
       start = i * window_size
       end = (i + 1) * window_size
       window_trajectory = trajectory[start:end]
       
       # Calculate properties for this window
       window_rmsd = np.mean(rmsd_analyzer.calculate_trajectory(window_trajectory))
       window_rg = np.mean(rg_analyzer.calculate_trajectory(window_trajectory))
       window_ss = ss_analyzer.analyze_trajectory(window_trajectory)
       window_ss_content = ss_analyzer.calculate_content(window_ss)
       
       time_resolved_data['rmsd'].append(window_rmsd)
       time_resolved_data['rg'].append(window_rg)
       time_resolved_data['ss_content'].append(window_ss_content['helix'])
   
   # Plot time evolution
   import matplotlib.pyplot as plt
   
   fig, axes = plt.subplots(3, 1, figsize=(10, 8))
   
   axes[0].plot(time_resolved_data['rmsd'])
   axes[0].set_ylabel('RMSD (nm)')
   
   axes[1].plot(time_resolved_data['rg'])
   axes[1].set_ylabel('Rg (nm)')
   
   axes[2].plot(time_resolved_data['ss_content'])
   axes[2].set_ylabel('Helix Content')
   axes[2].set_xlabel('Time Window')
   
   plt.tight_layout()
   plt.savefig('time_resolved_analysis.png', dpi=300)

See Also
--------

- :doc:`core` - Core simulation engine that generates trajectory data
- :doc:`structure` - Protein structures used in analysis
- :doc:`visualization` - Visualization of analysis results
- :doc:`../user_guide/tutorials` - Analysis tutorials and examples
- :doc:`../advanced/extending` - Implementing custom analysis tools
