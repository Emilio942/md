Force Fields
============

The :mod:`proteinMD.forcefield` module provides comprehensive force field implementations for molecular dynamics simulations, including popular force fields like AMBER ff14SB and CHARMM36.

.. currentmodule:: proteinMD.forcefield

Overview
--------

The forcefield module includes:

- **AMBER ff14SB**: Complete implementation of the AMBER ff14SB protein force field
- **CHARMM36**: CHARMM36 all-atom force field for proteins
- **Custom Force Fields**: Framework for implementing custom force field parameters
- **Parameter Management**: Automatic parameter assignment and validation
- **Non-bonded Interactions**: Lennard-Jones and Coulomb force calculations
- **Bonded Interactions**: Bond, angle, and dihedral potential energy terms

Quick Example
-------------

Basic force field setup:

.. code-block:: python

   from proteinMD.forcefield.amber_ff14sb import AmberFF14SB
   from proteinMD.structure.pdb_parser import PDBParser
   
   # Load protein structure
   parser = PDBParser()
   protein = parser.parse("protein.pdb")
   
   # Initialize force field
   forcefield = AmberFF14SB()
   
   # Assign parameters to protein
   forcefield.assign_parameters(protein)
   
   # Calculate energy and forces
   energy = forcefield.calculate_energy(protein.get_positions())
   forces = forcefield.calculate_forces(protein.get_positions())
   
   print(f"Potential energy: {energy:.2f} kJ/mol")

AMBER ff14SB Force Field
------------------------

.. automodule:: proteinMD.forcefield.amber_ff14sb
   :members:
   :undoc-members:
   :show-inheritance:

Main Force Field Class
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.forcefield.amber_ff14sb.AmberFF14SB
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   
   **Examples**
   
   Basic usage:
   
   .. code-block:: python
   
      # Initialize AMBER ff14SB force field
      ff = AmberFF14SB()
      
      # Assign parameters to protein
      ff.assign_parameters(protein)
      
      # Calculate total energy
      total_energy = ff.calculate_energy(positions)
   
   Advanced configuration:
   
   .. code-block:: python
   
      # Configure force field options
      ff = AmberFF14SB(
          cutoff=1.2,                    # 1.2 nm cutoff
          switch_distance=1.0,           # 1.0 nm switch distance
          use_pme=True,                  # Use PME for electrostatics
          pme_alpha=0.31,                # PME alpha parameter
          pme_grid_spacing=0.1,          # PME grid spacing
          constraint_tolerance=1e-6,     # Constraint tolerance
          hydrogen_mass=1.5              # Hydrogen mass for mass repartitioning
      )
      
      # Load custom parameters
      ff.load_custom_parameters("custom_residues.frcmod")
      
      # Assign parameters with validation
      ff.assign_parameters(protein, validate=True)

Parameter Classes
~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.forcefield.amber_ff14sb.AmberParameters
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      # Access force field parameters
      params = ff.parameters
      
      # Get atom type parameters
      atom_type = params.get_atom_type("CT")  # sp3 carbon
      print(f"Sigma: {atom_type.sigma}, Epsilon: {atom_type.epsilon}")
      
      # Get bond parameters
      bond_param = params.get_bond_parameter("CT", "HC")  # C-H bond
      print(f"Equilibrium length: {bond_param.r0}, Force constant: {bond_param.k}")
      
      # Get angle parameters
      angle_param = params.get_angle_parameter("CT", "CT", "HC")
      print(f"Equilibrium angle: {angle_param.theta0}, Force constant: {angle_param.k}")

.. autoclass:: proteinMD.forcefield.amber_ff14sb.AtomType
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: proteinMD.forcefield.amber_ff14sb.BondParameter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: proteinMD.forcefield.amber_ff14sb.AngleParameter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: proteinMD.forcefield.amber_ff14sb.DihedralParameter
   :members:
   :undoc-members:
   :show-inheritance:

CHARMM36 Force Field
--------------------

.. automodule:: proteinMD.forcefield.charmm36
   :members:
   :undoc-members:
   :show-inheritance:

CHARMM Implementation
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.forcefield.charmm36.CHARMM36
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.forcefield.charmm36 import CHARMM36
      
      # Initialize CHARMM36 force field
      ff = CHARMM36()
      
      # Load protein structure file (PSF format)
      ff.load_psf("protein.psf")
      
      # Assign parameters
      ff.assign_parameters(protein)
      
      # Calculate energy components
      energy_components = ff.calculate_energy_components(positions)
      print(f"Bond energy: {energy_components['bond']:.2f} kJ/mol")
      print(f"Angle energy: {energy_components['angle']:.2f} kJ/mol")
      print(f"Dihedral energy: {energy_components['dihedral']:.2f} kJ/mol")
      print(f"Nonbonded energy: {energy_components['nonbonded']:.2f} kJ/mol")

Custom Force Fields
-------------------

.. automodule:: proteinMD.forcefield.custom
   :members:
   :undoc-members:
   :show-inheritance:

Custom Force Field Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.forcefield.custom.CustomForceField
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.forcefield.custom import CustomForceField
      
      # Create custom force field
      custom_ff = CustomForceField()
      
      # Load parameters from file
      custom_ff.load_parameters("my_forcefield.json")
      
      # Or define parameters programmatically
      custom_ff.add_atom_type("MY", sigma=0.35, epsilon=0.5, charge=0.0)
      custom_ff.add_bond_parameter("MY", "CT", k=1000.0, r0=0.15)
      
      # Use like any other force field
      custom_ff.assign_parameters(protein)
      energy = custom_ff.calculate_energy(positions)

Parameter File Formats
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.forcefield.custom.ParameterLoader
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      loader = ParameterLoader()
      
      # Load different parameter file formats
      params = loader.load_amber_frcmod("custom.frcmod")
      params = loader.load_charmm_prm("custom.prm")
      params = loader.load_json("custom.json")
      
      # Apply to force field
      custom_ff.set_parameters(params)

Non-bonded Interactions
-----------------------

.. automodule:: proteinMD.forcefield.nonbonded
   :members:
   :undoc-members:
   :show-inheritance:

Lennard-Jones Interactions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.forcefield.nonbonded.LennardJonesCalculator
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.forcefield.nonbonded import LennardJonesCalculator
      
      lj_calc = LennardJonesCalculator(
          cutoff=1.2,                # 1.2 nm cutoff
          switch_distance=1.0,       # Start switching at 1.0 nm
          use_switching=True         # Use switching function
      )
      
      # Calculate LJ energy and forces
      lj_energy, lj_forces = lj_calc.calculate(
          positions=positions,
          atom_types=atom_types,
          lj_parameters=lj_params
      )

Electrostatic Interactions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.forcefield.nonbonded.CoulombCalculator
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.forcefield.nonbonded import CoulombCalculator
      
      coulomb_calc = CoulombCalculator(
          method="pme",              # Use PME method
          cutoff=1.2,                # Real-space cutoff
          alpha=0.31,                # PME alpha parameter
          grid_spacing=0.1           # PME grid spacing
      )
      
      # Calculate electrostatic energy and forces
      elec_energy, elec_forces = coulomb_calc.calculate(
          positions=positions,
          charges=charges
      )

Bonded Interactions
-------------------

.. automodule:: proteinMD.forcefield.bonded
   :members:
   :undoc-members:
   :show-inheritance:

Bond Potentials
~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.forcefield.bonded.BondCalculator
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.forcefield.bonded import BondCalculator
      
      bond_calc = BondCalculator()
      
      # Calculate bond energy and forces
      bond_energy, bond_forces = bond_calc.calculate(
          positions=positions,
          bond_list=bonds,
          bond_parameters=bond_params
      )

Angle Potentials
~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.forcefield.bonded.AngleCalculator
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.forcefield.bonded import AngleCalculator
      
      angle_calc = AngleCalculator()
      
      # Calculate angle energy and forces
      angle_energy, angle_forces = angle_calc.calculate(
          positions=positions,
          angle_list=angles,
          angle_parameters=angle_params
      )

Dihedral Potentials
~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.forcefield.bonded.DihedralCalculator
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.forcefield.bonded import DihedralCalculator
      
      dihedral_calc = DihedralCalculator()
      
      # Calculate dihedral energy and forces
      dihedral_energy, dihedral_forces = dihedral_calc.calculate(
          positions=positions,
          dihedral_list=dihedrals,
          dihedral_parameters=dihedral_params
      )

Parameter Validation
--------------------

.. automodule:: proteinMD.forcefield.validation
   :members:
   :undoc-members:
   :show-inheritance:

Force Field Validation
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.forcefield.validation.ForceFieldValidator
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.forcefield.validation import ForceFieldValidator
      
      validator = ForceFieldValidator()
      
      # Validate force field assignment
      is_valid, issues = validator.validate_parameters(protein, forcefield)
      
      if not is_valid:
          for issue in issues:
              print(f"Parameter issue: {issue}")
      
      # Check for missing parameters
      missing_params = validator.find_missing_parameters(protein, forcefield)
      if missing_params:
          print(f"Missing parameters: {missing_params}")

Common Usage Patterns
---------------------

Complete Force Field Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinMD.forcefield.amber_ff14sb import AmberFF14SB
   from proteinMD.forcefield.validation import ForceFieldValidator
   from proteinMD.structure.pdb_parser import PDBParser
   
   # Load protein
   parser = PDBParser()
   protein = parser.parse("protein.pdb")
   
   # Setup force field
   forcefield = AmberFF14SB(
       cutoff=1.2,
       use_pme=True,
       pme_grid_spacing=0.1
   )
   
   # Assign and validate parameters
   forcefield.assign_parameters(protein)
   
   validator = ForceFieldValidator()
   is_valid, issues = validator.validate_parameters(protein, forcefield)
   
   if is_valid:
       print("Force field parameters successfully assigned!")
   else:
       print("Parameter issues found:")
       for issue in issues:
           print(f"  - {issue}")
   
   # Calculate initial energy
   positions = protein.get_positions()
   energy = forcefield.calculate_energy(positions)
   print(f"Initial potential energy: {energy:.2f} kJ/mol")

Energy Component Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate detailed energy breakdown
   energy_components = forcefield.calculate_energy_components(positions)
   
   print("Energy Components:")
   print(f"  Bond energy:      {energy_components['bond']:.2f} kJ/mol")
   print(f"  Angle energy:     {energy_components['angle']:.2f} kJ/mol")
   print(f"  Dihedral energy:  {energy_components['dihedral']:.2f} kJ/mol")
   print(f"  Lennard-Jones:    {energy_components['lj']:.2f} kJ/mol")
   print(f"  Electrostatic:    {energy_components['electrostatic']:.2f} kJ/mol")
   print(f"  Total energy:     {sum(energy_components.values()):.2f} kJ/mol")

Custom Parameter Modification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Modify specific parameters
   params = forcefield.parameters
   
   # Change LJ parameters for a specific atom type
   ct_type = params.get_atom_type("CT")  # sp3 carbon
   ct_type.epsilon *= 1.1  # Increase well depth by 10%
   
   # Add custom bond parameter
   params.add_bond_parameter(
       atom_type1="CT",
       atom_type2="MY",  # Custom atom type
       k=1000.0,         # Force constant (kJ/mol/nm²)
       r0=0.15           # Equilibrium length (nm)
   )
   
   # Update force field with modified parameters
   forcefield.update_parameters(params)

Force Field Comparison
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinMD.forcefield.amber_ff14sb import AmberFF14SB
   from proteinMD.forcefield.charmm36 import CHARMM36
   
   # Compare different force fields
   amber_ff = AmberFF14SB()
   charmm_ff = CHARMM36()
   
   # Assign parameters
   amber_ff.assign_parameters(protein)
   charmm_ff.assign_parameters(protein)
   
   # Calculate energies
   positions = protein.get_positions()
   amber_energy = amber_ff.calculate_energy(positions)
   charmm_energy = charmm_ff.calculate_energy(positions)
   
   print(f"AMBER ff14SB energy: {amber_energy:.2f} kJ/mol")
   print(f"CHARMM36 energy:     {charmm_energy:.2f} kJ/mol")
   print(f"Difference:          {abs(amber_energy - charmm_energy):.2f} kJ/mol")

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimize force calculations for performance
   forcefield = AmberFF14SB(
       cutoff=1.0,                    # Smaller cutoff for speed
       use_pme=False,                 # Use simple cutoff instead of PME
       neighbor_list_cutoff=1.2,      # Neighbor list buffer
       neighbor_list_update_freq=20   # Update every 20 steps
   )
   
   # Enable parallel force calculation
   forcefield.set_parallel_calculation(n_threads=4)
   
   # Use single precision for speed (if accuracy allows)
   forcefield.set_precision("single")

See Also
--------

- :doc:`core` - Core simulation engine that uses force fields
- :doc:`structure` - Protein structures that force fields operate on
- :doc:`environment` - Environment models (water, boundaries)
- :doc:`../user_guide/tutorials` - Force field setup tutorials
- :doc:`../advanced/extending` - Implementing custom force fields
