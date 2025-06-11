ProteinMD Documentation
========================

.. image:: https://img.shields.io/badge/version-1.0.0-blue.svg
   :target: #
   :alt: Version

.. image:: https://img.shields.io/badge/python-3.8+-green.svg
   :target: #
   :alt: Python Version

.. image:: https://img.shields.io/badge/docs-sphinx-brightgreen.svg
   :target: #
   :alt: Documentation

**ProteinMD** is a comprehensive molecular dynamics simulation package designed for protein structure analysis, simulation, and visualization. This documentation provides complete API reference, tutorials, and examples for all functionality.

.. note::
   This documentation was generated automatically from code docstrings and covers all public APIs, complete with examples and cross-references.

Quick Start
-----------

Install and run your first simulation:

.. code-block:: python

   from proteinMD.structure.pdb_parser import PDBParser
   from proteinMD.core.simulation import MolecularDynamicsSimulation
   from proteinMD.forcefield.amber_ff14sb import AmberFF14SB
   
   # Load protein structure
   parser = PDBParser()
   protein = parser.parse("protein.pdb")
   
   # Setup force field
   forcefield = AmberFF14SB()
   
   # Create and run simulation
   simulation = MolecularDynamicsSimulation(
       system=protein,
       force_field=forcefield,
       timestep=0.002,
       temperature=300.0
   )
   simulation.run(50000)

Key Features
------------

✅ **Complete MD Simulation Stack**
   - Advanced force fields (AMBER ff14SB, CHARMM36)
   - Multiple environment models (explicit/implicit solvent)
   - Enhanced sampling methods (umbrella sampling, REMD, steered MD)

✅ **Comprehensive Analysis Tools**
   - RMSD, radius of gyration, Ramachandran plots
   - Secondary structure analysis
   - Hydrogen bond networks
   - Free energy calculations

✅ **Advanced Visualization**
   - 3D protein visualization with multiple rendering modes
   - Real-time simulation monitoring
   - Trajectory animations
   - Energy dashboards and analysis plots

✅ **Production-Ready CLI**
   - Complete command-line interface
   - Batch processing capabilities
   - Workflow templates
   - Integration with computational pipelines

✅ **Scientific Validation**
   - Benchmarked against GROMACS, AMBER, and NAMD
   - Comprehensive test suite (>90% coverage goal)
   - Peer-reviewed validation studies
   - Performance optimization

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/user_manual
   user_guide/installation
   user_guide/quick_start
   user_guide/tutorials
   user_guide/cli_reference
   user_guide/examples

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/index
   api/core
   api/structure
   api/forcefield
   api/environment
   api/analysis
   api/sampling
   api/visualization
   api/cli
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/scientific_background
   advanced/performance
   advanced/validation
   advanced/extending
   advanced/troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer/developer_guide
   developer/architecture
   developer/contributing
   developer/coding_standards
   developer/api_design
   developer/testing
   developer/documentation
   developer/git_workflow
   developer/pull_request_guide
   developer/review_process
   developer/release_process
   developer/performance_guide

.. toctree::
   :maxdepth: 1
   :caption: About

   about/license
   about/changelog
   about/citation

Module Overview
---------------

Core Modules
~~~~~~~~~~~~

:mod:`proteinMD.core`
   Core simulation engine and molecular dynamics algorithms

:mod:`proteinMD.structure`
   Protein structure handling, PDB parsing, and molecular representations

:mod:`proteinMD.forcefield`
   Force field implementations (AMBER ff14SB, CHARMM36, custom)

:mod:`proteinMD.environment`
   Simulation environment setup (water models, periodic boundaries, implicit solvent)

Analysis & Sampling
~~~~~~~~~~~~~~~~~~~

:mod:`proteinMD.analysis`
   Analysis tools for trajectory data (RMSD, Rg, secondary structure, H-bonds)

:mod:`proteinMD.sampling`
   Enhanced sampling methods (umbrella sampling, replica exchange, steered MD)

Visualization & Interface
~~~~~~~~~~~~~~~~~~~~~~~~~

:mod:`proteinMD.visualization`
   3D visualization, trajectory animation, energy plots, real-time monitoring

:mod:`proteinMD.cli`
   Command-line interface for automated workflows and batch processing

Utilities & Validation
~~~~~~~~~~~~~~~~~~~~~~

:mod:`proteinMD.utils`
   Utility functions, benchmarking tools, and helper classes

:mod:`proteinMD.validation`
   Scientific validation framework and performance benchmarking

Quick Links
-----------

* :doc:`user_guide/installation` - Get started with installation
* :doc:`user_guide/quick_start` - Your first simulation in 5 minutes
* :doc:`api/core` - Core simulation engine API
* :doc:`user_guide/cli_reference` - Complete CLI documentation
* :doc:`advanced/performance` - Performance optimization guide
* :doc:`developer/contributing` - Contributing to ProteinMD

Search and Index
----------------

* :ref:`genindex` - General index of all functions and classes
* :ref:`modindex` - Module index with all packages
* :ref:`search` - Full-text search across documentation

Recent Updates
--------------

.. versionadded:: 1.0.0
   Initial release with complete MD simulation stack, comprehensive analysis tools, 
   and production-ready CLI interface. Full validation against established MD software.

Support and Community
---------------------

- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join the community discussions
- **Citation**: See :doc:`about/citation` for how to cite ProteinMD in publications

.. note::
   ProteinMD is actively developed and maintained. This documentation is automatically 
   updated with each release to reflect the latest API changes and new features.
