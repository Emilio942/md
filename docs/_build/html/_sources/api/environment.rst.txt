Environment Models
==================

The :mod:`proteinMD.environment` module provides comprehensive environment models for molecular dynamics simulations, including explicit and implicit solvent models, periodic boundary conditions, and other environmental effects.

.. currentmodule:: proteinMD.environment

Overview
--------

The environment module includes:

- **Water Models**: TIP3P explicit water model for realistic solvation
- **Implicit Solvent**: Generalized Born (GB) model for fast approximate solvation
- **Periodic Boundary Conditions**: PBC for bulk properties and realistic boundary effects
- **Ion Models**: Support for common ions (Na⁺, Cl⁻, K⁺, Mg²⁺)
- **Membrane Models**: Lipid bilayer models for membrane protein simulations
- **Vacuum Simulations**: Gas-phase simulations without solvent

Quick Example
-------------

Basic solvation setup:

.. code-block:: python

   from proteinMD.environment.water import TIP3PWaterModel
   from proteinMD.environment.periodic_boundary import PeriodicBoundaryConditions
   from proteinMD.structure.pdb_parser import PDBParser
   
   # Load protein
   parser = PDBParser()
   protein = parser.parse("protein.pdb")
   
   # Add explicit water
   water_model = TIP3PWaterModel()
   solvated_system = water_model.solvate_protein(
       protein=protein,
       padding=1.0,        # 1.0 nm padding
       ion_concentration=0.15,  # 150 mM NaCl
       neutralize=True     # Neutralize system
   )
   
   # Setup periodic boundaries
   pbc = PeriodicBoundaryConditions()
   pbc.setup_cubic_box(solvated_system, padding=0.5)
   
   print(f"Solvated system: {len(solvated_system.atoms)} atoms")
   print(f"Water molecules: {water_model.count_water_molecules()}")

Explicit Water Models
---------------------

.. automodule:: proteinMD.environment.water
   :members:
   :undoc-members:
   :show-inheritance:

TIP3P Water Model
~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.environment.water.TIP3PWaterModel
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   
   **Examples**
   
   Basic solvation:
   
   .. code-block:: python
   
      water_model = TIP3PWaterModel()
      
      # Solvate protein with default settings
      solvated_system = water_model.solvate_protein(protein, padding=1.0)
   
   Advanced solvation with ions:
   
   .. code-block:: python
   
      water_model = TIP3PWaterModel(
          density=1000.0,        # Water density (kg/m³)
          temperature=300.0      # Temperature for density calculation
      )
      
      # Solvate with specific ion concentration
      solvated_system = water_model.solvate_protein(
          protein=protein,
          padding=1.5,           # 1.5 nm padding around protein
          box_type="cubic",      # Cubic simulation box
          ion_concentration=0.15, # 150 mM salt concentration
          positive_ion="Na+",    # Sodium ions
          negative_ion="Cl-",    # Chloride ions
          neutralize=True,       # Neutralize system charge
          min_distance=0.23      # Minimum distance between atoms (nm)
      )
      
      print(f"Added {water_model.n_water_molecules} water molecules")
      print(f"Added {water_model.n_positive_ions} Na+ ions")
      print(f"Added {water_model.n_negative_ions} Cl- ions")

.. autoclass:: proteinMD.environment.water.WaterMolecule
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      # Create individual water molecule
      water = WaterMolecule(
          oxygen_position=[0.0, 0.0, 0.0],
          orientation="random"
      )
      
      # Access atom positions
      o_pos = water.oxygen_position
      h1_pos, h2_pos = water.hydrogen_positions
      
      # Get molecule properties
      dipole_moment = water.dipole_moment()
      com = water.center_of_mass()

TIP4P Water Model
~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.environment.water.TIP3PWaterModel
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      # TIP4P model with virtual sites
      tip4p_model = TIP4PWaterModel()
      
      # Solvate with TIP4P water
      solvated_system = tip4p_model.solvate_protein(
          protein=protein,
          padding=1.0
      )
      
      # TIP4P includes virtual sites for better electrostatics
      print(f"Virtual sites: {tip4p_model.n_virtual_sites}")

Implicit Solvent Models
-----------------------

.. automodule:: proteinMD.environment.implicit_solvent
   :members:
   :undoc-members:
   :show-inheritance:

Generalized Born Model
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.environment.implicit_solvent.ImplicitSolventModel
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   
   **Examples**
   
   Basic implicit solvent:
   
   .. code-block:: python
   
      from proteinMD.environment.implicit_solvent import ImplicitSolventModel
      
      # Setup implicit solvent
      implicit_solvent = ImplicitSolventModel(
          solvent_dielectric=80.0,    # Water dielectric constant
          solute_dielectric=1.0,      # Protein dielectric constant
          salt_concentration=0.15,    # 150 mM ionic strength
          temperature=300.0           # Temperature in Kelvin
      )
      
      # Apply to protein
      implicit_solvent.apply_to_protein(protein)
      
      # Calculate solvation free energy
      delta_g_solv = implicit_solvent.calculate_solvation_energy(
          protein.get_positions()
      )
      print(f"Solvation free energy: {delta_g_solv:.2f} kJ/mol")
   
   Advanced GB model configuration:
   
   .. code-block:: python
   
      # Configure GB model parameters
      implicit_solvent = ImplicitSolventModel(
          gb_model="HCT",            # HCT, OBC1, or OBC2
          solvent_dielectric=80.0,
          solute_dielectric=1.0,
          salt_concentration=0.15,
          surface_tension=0.0072,    # kJ/mol/nm² (for SA term)
          probe_radius=0.14,         # Solvent probe radius (nm)
          use_cutoff=True,
          cutoff=2.0                 # GB cutoff distance (nm)
      )
      
      # Calculate Born radii
      born_radii = implicit_solvent.calculate_born_radii(protein.get_positions())
      
      # Calculate GB energy components
      energy_components = implicit_solvent.calculate_energy_components(
          protein.get_positions()
      )
      print(f"GB energy: {energy_components['gb']:.2f} kJ/mol")
      print(f"SA energy: {energy_components['sa']:.2f} kJ/mol")

.. autoclass:: proteinMD.environment.implicit_solvent.GeneralizedBornCalculator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: proteinMD.environment.implicit_solvent.SurfaceAreaCalculator
   :members:
   :undoc-members:
   :show-inheritance:

Periodic Boundary Conditions
-----------------------------

.. automodule:: proteinMD.environment.periodic_boundary
   :members:
   :undoc-members:
   :show-inheritance:

PBC Implementation
~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.environment.periodic_boundary.PeriodicBoundaryConditions
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   
   **Examples**
   
   Cubic box setup:
   
   .. code-block:: python
   
      from proteinMD.environment.periodic_boundary import PeriodicBoundaryConditions
      
      pbc = PeriodicBoundaryConditions()
      
      # Setup cubic box
      box_size = pbc.setup_cubic_box(
          system=solvated_system,
          padding=0.5  # Additional padding beyond solvation
      )
      print(f"Box size: {box_size:.2f} nm")
   
   Custom box dimensions:
   
   .. code-block:: python
   
      # Setup rectangular box
      pbc.setup_rectangular_box(
          system=solvated_system,
          dimensions=[5.0, 6.0, 7.0]  # nm
      )
      
      # Setup from box vectors
      box_vectors = np.array([
          [5.0, 0.0, 0.0],
          [0.0, 6.0, 0.0],
          [0.0, 0.0, 7.0]
      ])
      pbc.set_box_vectors(box_vectors)
      
      # Apply minimum image convention
      distance = pbc.minimum_image_distance(pos1, pos2)
      
      # Wrap coordinates into box
      wrapped_coords = pbc.wrap_coordinates(coordinates)

.. autoclass:: proteinMD.environment.periodic_boundary.BoxType
   :members:
   :undoc-members:
   :show-inheritance:

Ion Models
----------

.. automodule:: proteinMD.environment.ions
   :members:
   :undoc-members:
   :show-inheritance:

Ion Management
~~~~~~~~~~~~~~

.. autoclass:: proteinMD.environment.ions.IonModel
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.environment.ions import IonModel
      
      ion_model = IonModel()
      
      # Add ions to neutralize system
      neutralized_system = ion_model.neutralize_system(
          system=solvated_system,
          positive_ion="Na+",
          negative_ion="Cl-"
      )
      
      # Add specific ion concentration
      ionized_system = ion_model.add_ions(
          system=solvated_system,
          ion_concentration=0.15,  # 150 mM
          positive_ion="K+",       # Potassium
          negative_ion="Cl-",      # Chloride
          replace_solvent=True     # Replace water molecules with ions
      )

.. autoclass:: proteinMD.environment.ions.Ion
   :members:
   :undoc-members:
   :show-inheritance:

Membrane Models
---------------

.. automodule:: proteinMD.environment.membrane
   :members:
   :undoc-members:
   :show-inheritance:

Lipid Bilayer Models
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.environment.membrane.LipidBilayer
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.environment.membrane import LipidBilayer
      
      # Create POPC bilayer
      bilayer = LipidBilayer(
          lipid_type="POPC",       # Phospholipid type
          dimensions=[10.0, 10.0], # Membrane dimensions (nm)
          n_lipids=128,            # Number of lipids per leaflet
          hydration=40             # Water molecules per lipid
      )
      
      # Insert membrane protein
      membrane_system = bilayer.insert_protein(
          protein=membrane_protein,
          orientation="transmembrane",
          z_position=0.0  # Center in membrane
      )
      
      # Equilibrate membrane
      bilayer.equilibrate(n_steps=50000)

Environment Utilities
---------------------

.. automodule:: proteinMD.environment.utils
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: proteinMD.environment.utils.calculate_box_volume
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.environment.utils import calculate_box_volume
      
      volume = calculate_box_volume(box_vectors)
      print(f"Box volume: {volume:.2f} nm³")

.. autofunction:: proteinMD.environment.utils.minimum_image_distance
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.environment.utils import minimum_image_distance
      
      # Calculate minimum image distance
      distance = minimum_image_distance(
          pos1=atom1.position,
          pos2=atom2.position,
          box_vectors=box_vectors
      )

.. autofunction:: proteinMD.environment.utils.solvation_shell_analysis
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.environment.utils import solvation_shell_analysis
      
      # Analyze hydration shell
      shell_waters = solvation_shell_analysis(
          protein_positions=protein.get_positions(),
          water_positions=water_coords,
          cutoff=0.35  # First solvation shell cutoff (nm)
      )
      
      print(f"Waters in first shell: {len(shell_waters)}")

Common Usage Patterns
---------------------

Complete Solvation Setup
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinMD.environment.water import TIP3PWaterModel
   from proteinMD.environment.periodic_boundary import PeriodicBoundaryConditions
   from proteinMD.environment.ions import IonModel
   
   # Load protein
   protein = parser.parse("protein.pdb")
   
   # Center protein at origin
   protein.translate(-protein.center_of_mass())
   
   # Solvate protein
   water_model = TIP3PWaterModel()
   solvated_system = water_model.solvate_protein(
       protein=protein,
       padding=1.2,              # 1.2 nm water padding
       min_distance=0.23         # Minimum atom-water distance
   )
   
   # Add ions for physiological conditions
   ion_model = IonModel()
   ionized_system = ion_model.add_ions(
       system=solvated_system,
       ion_concentration=0.15,   # 150 mM NaCl
       positive_ion="Na+",
       negative_ion="Cl-",
       neutralize=True
   )
   
   # Setup periodic boundaries
   pbc = PeriodicBoundaryConditions()
   box_size = pbc.setup_cubic_box(ionized_system, padding=0.5)
   
   print(f"Final system: {len(ionized_system.atoms)} atoms")
   print(f"Box size: {box_size:.2f} nm")
   print(f"System density: {ionized_system.density():.1f} kg/m³")

Implicit vs Explicit Solvent Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinMD.environment.implicit_solvent import ImplicitSolventModel
   from proteinMD.environment.water import TIP3PWaterModel
   
   # Setup explicit solvation
   water_model = TIP3PWaterModel()
   explicit_system = water_model.solvate_protein(protein, padding=1.0)
   
   # Setup implicit solvation
   implicit_solvent = ImplicitSolventModel()
   implicit_system = protein.copy()
   implicit_solvent.apply_to_protein(implicit_system)
   
   print("System comparison:")
   print(f"Explicit solvent: {len(explicit_system.atoms)} atoms")
   print(f"Implicit solvent: {len(implicit_system.atoms)} atoms")
   print(f"Speedup factor: {len(explicit_system.atoms) / len(implicit_system.atoms):.1f}x")

Membrane Protein Setup
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinMD.environment.membrane import LipidBilayer
   from proteinMD.environment.water import TIP3PWaterModel
   
   # Create lipid bilayer
   bilayer = LipidBilayer(
       lipid_type="POPC",
       dimensions=[12.0, 12.0],  # Large membrane for protein
       thickness=4.0,            # Membrane thickness
       area_per_lipid=0.68       # nm² per lipid
   )
   
   # Insert transmembrane protein
   membrane_system = bilayer.insert_protein(
       protein=transmembrane_protein,
       orientation="transmembrane",
       insertion_method="CHARMM-GUI"
   )
   
   # Add water layers above and below membrane
   water_model = TIP3PWaterModel()
   hydrated_membrane = water_model.hydrate_membrane(
       membrane_system=membrane_system,
       water_thickness=3.0  # 3 nm water layer on each side
   )
   
   # Setup periodic boundaries for membrane simulation
   pbc = PeriodicBoundaryConditions()
   pbc.setup_membrane_box(hydrated_membrane)

Custom Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinMD.environment.custom import CustomEnvironment
   
   # Create custom environment
   custom_env = CustomEnvironment()
   
   # Add custom solvent model
   custom_env.add_solvent(
       name="methanol",
       dielectric=32.7,
       density=791.3,  # kg/m³
       molecule_structure="methanol.mol2"
   )
   
   # Add custom boundary conditions
   custom_env.set_boundary_conditions(
       type="spherical",
       radius=5.0,     # 5 nm sphere
       center="protein_com"
   )
   
   # Apply environment to system
   custom_system = custom_env.apply_to_system(protein)

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For large systems, optimize environment setup
   water_model = TIP3PWaterModel(
       density_optimization=True,     # Optimize water density
       parallel_solvation=True,       # Use parallel solvation
       n_threads=4                    # Number of threads
   )
   
   # Use reduced precision for speed
   implicit_solvent = ImplicitSolventModel(
       precision="single",            # Single vs double precision
       use_lookup_tables=True,        # Faster Born radii calculation
       cutoff=2.0                     # Smaller cutoff for speed
   )
   
   # Efficient PBC implementation
   pbc = PeriodicBoundaryConditions(
       use_cell_lists=True,           # Cell list optimization
       cell_list_cutoff=1.2,          # Cell list cutoff
       update_frequency=20            # Update every 20 steps
   )

See Also
--------

- :doc:`core` - Core simulation engine that uses environment models
- :doc:`forcefield` - Force field parameters for environment interactions
- :doc:`analysis` - Analysis tools for solvation and environment properties
- :doc:`../user_guide/tutorials` - Environment setup tutorials
- :doc:`../advanced/performance` - Performance optimization for large systems
