API Reference
=============

Complete API documentation for all ProteinMD modules.

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   core
   structure
   forcefield
   environment
   analysis
   sampling
   visualization
   utils
   cli

Overview
--------

The ProteinMD API is organized into several key modules:

* :doc:`core` - Core simulation engine and molecular dynamics functionality
* :doc:`structure` - Protein structure handling, parsing, and manipulation
* :doc:`forcefield` - Force field implementations (AMBER, CHARMM, etc.)
* :doc:`environment` - Simulation environment setup (solvation, boundaries)
* :doc:`analysis` - Analysis tools for trajectories and structures
* :doc:`sampling` - Enhanced sampling methods and techniques
* :doc:`visualization` - Molecular visualization and plotting utilities
* :doc:`utils` - Utility functions and helper classes
* :doc:`cli` - Command-line interface

Quick Reference
---------------

Core Classes
~~~~~~~~~~~~~

.. currentmodule:: proteinMD.core.simulation

.. autosummary::
   :nosignatures:

   MolecularDynamicsSimulation

.. currentmodule:: proteinMD.structure.protein

.. autosummary::
   :nosignatures:

   Protein
   Chain
   Residue
   Atom

.. currentmodule:: proteinMD.structure.pdb_parser

.. autosummary::
   :nosignatures:

   PDBParser

Analysis Tools
~~~~~~~~~~~~~~

.. currentmodule:: proteinMD.analysis.rmsd

.. autosummary::
   :nosignatures:

   RMSDAnalyzer

.. currentmodule:: proteinMD.analysis.radius_of_gyration

.. autosummary::
   :nosignatures:

   RadiusOfGyrationAnalyzer

.. currentmodule:: proteinMD.analysis.secondary_structure

.. autosummary::
   :nosignatures:

   SecondaryStructureAnalyzer

Force Fields
~~~~~~~~~~~~~

.. currentmodule:: proteinMD.forcefield.amber_ff14sb

.. autosummary::
   :nosignatures:

   AmberFF14SB

.. currentmodule:: proteinMD.forcefield.charmm36

.. autosummary::
   :nosignatures:

   CHARMM36

Environment
~~~~~~~~~~~

.. currentmodule:: proteinMD.environment.water

.. autosummary::
   :nosignatures:

   TIP3PWaterModel

.. currentmodule:: proteinMD.environment.implicit_solvent

.. autosummary::
   :nosignatures:

   ImplicitSolventModel

.. currentmodule:: proteinMD.environment.periodic_boundary

.. autosummary::
   :nosignatures:

   PeriodicBoundaryConditions

Sampling Methods
~~~~~~~~~~~~~~~~

.. currentmodule:: proteinMD.sampling.umbrella_sampling

.. autosummary::
   :nosignatures:

   UmbrellaSampling

.. currentmodule:: proteinMD.sampling.steered_md

.. autosummary::
   :nosignatures:

   SteeredMD
   ReplicaExchangeMD

Visualization
~~~~~~~~~~~~~

.. currentmodule:: proteinMD.visualization.plotting

.. autosummary::
   :nosignatures:

   TrajectoryAnimator
   EnergyPlotter
   StructurePlotter

See Also
--------

* :doc:`../user_guide/quick_start` - User guide and tutorials
* :doc:`../developer/contributing` - Developer documentation
* :doc:`../advanced/performance` - Advanced topics and optimization
