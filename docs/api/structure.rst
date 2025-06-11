Structure Handling
==================

The :mod:`proteinMD.structure` module provides comprehensive tools for handling protein structures, parsing molecular data files, and managing molecular representations.

.. currentmodule:: proteinMD.structure

Overview
--------

The structure module includes:

- **PDB Parser**: Robust PDB file parsing with error handling
- **Protein Representation**: Hierarchical protein data structures (Protein → Chain → Residue → Atom)
- **Structure Manipulation**: Superposition, RMSD calculation, center of mass
- **File I/O**: Support for multiple structure formats (PDB, PDBx/mmCIF, MOL2)
- **Validation**: Structure integrity checking and error reporting

Quick Example
-------------

Basic structure loading and manipulation:

.. code-block:: python

   from proteinMD.structure.pdb_parser import PDBParser
   from proteinMD.structure.protein import Protein
   
   # Parse PDB file
   parser = PDBParser()
   protein = parser.parse("1ubq.pdb")
   
   # Access structure information
   print(f"Protein has {len(protein.atoms)} atoms")
   print(f"Protein has {len(protein.residues)} residues")
   print(f"Protein has {len(protein.chains)} chains")
   
   # Calculate properties
   center_of_mass = protein.center_of_mass()
   radius_of_gyration = protein.radius_of_gyration()
   
   # Structure manipulation
   protein.translate(-center_of_mass)  # Center at origin
   protein.rotate(rotation_matrix)     # Apply rotation

PDB Parser Module
-----------------

.. automodule:: proteinMD.structure.pdb_parser
   :members:
   :undoc-members:
   :show-inheritance:

Main Parser Classes
~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.structure.pdb_parser.PDBParser
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   
   **Examples**
   
   Basic PDB parsing:
   
   .. code-block:: python
   
      parser = PDBParser()
      protein = parser.parse("protein.pdb")
   
   Advanced parsing with options:
   
   .. code-block:: python
   
      parser = PDBParser(
          ignore_hetatm=False,    # Include HETATM records
          ignore_hydrogens=False, # Include hydrogen atoms
          validate_structure=True # Enable structure validation
      )
      
      try:
          protein = parser.parse("complex_structure.pdb")
      except PDBParseError as e:
          print(f"Parsing failed: {e}")

.. autoclass:: proteinMD.structure.pdb_parser.PDBParser
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      reader = PDBReader("multi_model.pdb")
      
      # Read all models
      models = reader.read_all_models()
      
      # Read specific model
      first_model = reader.read_model(0)
      
      # Iterate through models
      for model_id, protein in reader:
          print(f"Model {model_id}: {len(protein.atoms)} atoms")

.. autoclass:: proteinMD.structure.pdb_parser.PDBWriter
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      writer = PDBWriter("output.pdb")
      
      # Write single structure
      writer.write_structure(protein)
      
      # Write trajectory frames
      for frame in trajectory:
          protein.set_positions(frame.positions)
          writer.write_structure(protein, model_id=frame.step)
      
      writer.close()

Protein Data Structures
-----------------------

.. automodule:: proteinMD.structure.protein
   :members:
   :undoc-members:
   :show-inheritance:

Core Classes
~~~~~~~~~~~~

.. autoclass:: proteinMD.structure.protein.Protein
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   
   **Examples**
   
   Create protein from scratch:
   
   .. code-block:: python
   
      from proteinMD.structure.protein import Protein, Chain, Residue, Atom
      
      # Create atoms
      atoms = [
          Atom("N", [0.0, 0.0, 0.0], "N", "ALA", 1, "A"),
          Atom("CA", [1.0, 0.0, 0.0], "C", "ALA", 1, "A"),
          Atom("C", [2.0, 0.0, 0.0], "C", "ALA", 1, "A")
      ]
      
      # Create residue
      residue = Residue("ALA", 1, "A", atoms)
      
      # Create chain
      chain = Chain("A", [residue])
      
      # Create protein
      protein = Protein([chain])
   
   Protein analysis:
   
   .. code-block:: python
   
      # Structural properties
      com = protein.center_of_mass()
      rg = protein.radius_of_gyration()
      bbox = protein.bounding_box()
      
      # Atom selection
      ca_atoms = protein.select_atoms(name="CA")
      backbone_atoms = protein.select_atoms(name=["N", "CA", "C"])
      chain_a = protein.select_atoms(chain_id="A")
      
      # Distance calculations
      distance = protein.distance(atom1_id, atom2_id)
      distances = protein.distance_matrix()

.. autoclass:: proteinMD.structure.protein.Chain
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   
   **Example Usage**
   
   .. code-block:: python
   
      chain = protein.chains[0]  # First chain
      
      # Chain properties
      sequence = chain.sequence()
      n_residues = len(chain.residues)
      
      # Residue access
      first_residue = chain.residues[0]
      residue_10 = chain.get_residue(10)

.. autoclass:: proteinMD.structure.protein.Residue
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   
   **Example Usage**
   
   .. code-block:: python
   
      residue = chain.residues[0]
      
      # Residue information
      print(f"Residue: {residue.name} {residue.number}")
      print(f"Chain: {residue.chain_id}")
      
      # Atom access
      ca_atom = residue.get_atom("CA")
      backbone = residue.get_backbone_atoms()
      sidechain = residue.get_sidechain_atoms()
      
      # Geometric properties
      center = residue.center_of_mass()
      
      # Phi/Psi angles (for amino acids)
      phi, psi = residue.phi_psi_angles()

.. autoclass:: proteinMD.structure.protein.Atom
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   
   **Example Usage**
   
   .. code-block:: python
   
      atom = residue.atoms[0]
      
      # Atom properties
      print(f"Atom: {atom.name} ({atom.element})")
      print(f"Position: {atom.position}")
      print(f"Residue: {atom.residue_name} {atom.residue_number}")
      
      # Update position
      atom.position = np.array([1.0, 2.0, 3.0])
      
      # Calculate distance to another atom
      distance = atom.distance_to(other_atom)

Structure Validation
--------------------

.. automodule:: proteinMD.structure.validation
   :members:
   :undoc-members:
   :show-inheritance:

Validation Tools
~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.structure.validation.StructureValidator
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.structure.validation import StructureValidator
      
      validator = StructureValidator()
      
      # Validate structure
      is_valid, issues = validator.validate(protein)
      
      if not is_valid:
          for issue in issues:
              print(f"Warning: {issue}")
      
      # Specific checks
      missing_atoms = validator.check_missing_atoms(protein)
      clashes = validator.check_atomic_clashes(protein)
      bond_issues = validator.check_bond_lengths(protein)

Structure Utilities
-------------------

.. automodule:: proteinMD.structure.utils
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: proteinMD.structure.utils.superpose_structures
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.structure.utils import superpose_structures
      
      # Superpose two structures
      rmsd, rotation, translation = superpose_structures(
          reference_coords=ref_protein.get_positions(),
          mobile_coords=mobile_protein.get_positions()
      )
      
      # Apply transformation
      mobile_protein.rotate(rotation)
      mobile_protein.translate(translation)

.. autofunction:: proteinMD.structure.utils.calculate_rmsd
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.structure.utils import calculate_rmsd
      
      # Calculate RMSD between structures
      rmsd = calculate_rmsd(
          coords1=protein1.get_positions(),
          coords2=protein2.get_positions()
      )
      
      # Calculate CA-only RMSD
      ca_coords1 = protein1.select_atoms(name="CA").get_positions()
      ca_coords2 = protein2.select_atoms(name="CA").get_positions()
      ca_rmsd = calculate_rmsd(ca_coords1, ca_coords2)

.. autofunction:: proteinMD.structure.utils.center_of_mass
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.structure.utils import center_of_mass
      
      # Calculate center of mass
      com = center_of_mass(
          positions=protein.get_positions(),
          masses=protein.get_masses()
      )
      
      # Center protein at origin
      protein.translate(-com)

File Format Support
-------------------

.. automodule:: proteinMD.structure.formats
   :members:
   :undoc-members:
   :show-inheritance:

Multiple Format Support
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: proteinMD.structure.formats.StructureIO
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Example Usage**
   
   .. code-block:: python
   
      from proteinMD.structure.formats import StructureIO
      
      # Auto-detect format and load
      protein = StructureIO.load("structure.pdb")
      protein = StructureIO.load("structure.cif")
      protein = StructureIO.load("structure.mol2")
      
      # Save in different formats
      StructureIO.save(protein, "output.pdb")
      StructureIO.save(protein, "output.cif")
      StructureIO.save(protein, "output.xyz")

Common Usage Patterns
---------------------

Loading and Analyzing Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinMD.structure.pdb_parser import PDBParser
   from proteinMD.structure.validation import StructureValidator
   
   # Load and validate structure
   parser = PDBParser(validate_structure=True)
   protein = parser.parse("protein.pdb")
   
   # Structure analysis
   print(f"Protein: {len(protein.atoms)} atoms, {len(protein.residues)} residues")
   print(f"Chains: {[chain.id for chain in protein.chains]}")
   print(f"Center of mass: {protein.center_of_mass()}")
   print(f"Radius of gyration: {protein.radius_of_gyration():.2f} nm")
   
   # Atom selection examples
   ca_atoms = protein.select_atoms(name="CA")
   hydrophobic_residues = protein.select_atoms(
       residue_name=["ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "TYR"]
   )
   
   # Distance analysis
   distances = protein.distance_matrix()
   contacts = protein.find_contacts(cutoff=0.5)  # 5 Å contacts

Structure Comparison
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinMD.structure.utils import superpose_structures, calculate_rmsd
   
   # Load two structures
   protein1 = parser.parse("structure1.pdb")
   protein2 = parser.parse("structure2.pdb")
   
   # Align structures
   rmsd, rotation, translation = superpose_structures(
       reference_coords=protein1.get_positions(),
       mobile_coords=protein2.get_positions()
   )
   
   # Apply alignment
   protein2.rotate(rotation)
   protein2.translate(translation)
   
   print(f"RMSD after alignment: {rmsd:.3f} nm")
   
   # Calculate CA-only RMSD
   ca1 = protein1.select_atoms(name="CA").get_positions()
   ca2 = protein2.select_atoms(name="CA").get_positions()
   ca_rmsd = calculate_rmsd(ca1, ca2)
   print(f"CA RMSD: {ca_rmsd:.3f} nm")

Trajectory Processing
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinMD.structure.pdb_parser import PDBWriter
   
   # Process trajectory frames
   writer = PDBWriter("trajectory.pdb")
   
   for frame_id, frame in enumerate(trajectory):
       # Update protein coordinates
       protein.set_positions(frame.positions)
       
       # Calculate properties for this frame
       rg = protein.radius_of_gyration()
       
       # Write frame to PDB
       writer.write_structure(protein, model_id=frame_id)
       
       print(f"Frame {frame_id}: Rg = {rg:.3f} nm")
   
   writer.close()

Custom Structure Creation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from proteinMD.structure.protein import Protein, Chain, Residue, Atom
   
   # Create a simple dipeptide (ALA-VAL)
   atoms = []
   
   # Alanine residue
   ala_atoms = [
       Atom("N", [0.0, 0.0, 0.0], "N", "ALA", 1, "A"),
       Atom("CA", [1.0, 0.0, 0.0], "C", "ALA", 1, "A"),
       Atom("C", [2.0, 0.0, 0.0], "C", "ALA", 1, "A"),
       Atom("O", [2.5, 1.0, 0.0], "O", "ALA", 1, "A"),
       Atom("CB", [1.0, 0.0, 1.5], "C", "ALA", 1, "A")
   ]
   
   # Valine residue  
   val_atoms = [
       Atom("N", [3.0, 0.0, 0.0], "N", "VAL", 2, "A"),
       Atom("CA", [4.0, 0.0, 0.0], "C", "VAL", 2, "A"),
       Atom("C", [5.0, 0.0, 0.0], "C", "VAL", 2, "A"),
       Atom("O", [5.5, 1.0, 0.0], "O", "VAL", 2, "A"),
       Atom("CB", [4.0, 0.0, 1.5], "C", "VAL", 2, "A"),
       Atom("CG1", [3.0, 0.0, 2.5], "C", "VAL", 2, "A"),
       Atom("CG2", [5.0, 0.0, 2.5], "C", "VAL", 2, "A")
   ]
   
   # Create residues and chain
   ala_residue = Residue("ALA", 1, "A", ala_atoms)
   val_residue = Residue("VAL", 2, "A", val_atoms)
   chain = Chain("A", [ala_residue, val_residue])
   
   # Create protein
   dipeptide = Protein([chain])
   
   # Save structure
   writer = PDBWriter("dipeptide.pdb")
   writer.write_structure(dipeptide)

See Also
--------

- :doc:`core` - Core simulation engine
- :doc:`forcefield` - Force field parameters for structural data
- :doc:`analysis` - Structure analysis tools
- :doc:`../user_guide/tutorials` - Structure handling tutorials
- :doc:`../advanced/extending` - Adding new file format support
