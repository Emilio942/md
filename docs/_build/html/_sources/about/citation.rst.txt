Citation Guide
==============

If you use ProteinMD in your research, please cite it appropriately to help support the project and give credit to the developers.

.. contents:: Citation Information
   :local:
   :depth: 2

Primary Citation
================

Software Citation
~~~~~~~~~~~~~~~~

Please cite ProteinMD using the following format:

**APA Style:**

.. code-block:: text

   ProteinMD Development Team. (2024). ProteinMD: A Python Framework for 
   Molecular Dynamics Simulations (Version 1.0.0) [Computer software]. 
   https://doi.org/10.5281/zenodo.XXXXXXX

**BibTeX:**

.. code-block:: bibtex

   @software{proteinmd2024,
     title={ProteinMD: A Python Framework for Molecular Dynamics Simulations},
     author={{ProteinMD Development Team}},
     year={2024},
     version={1.0.0},
     url={https://github.com/proteinmd/proteinmd},
     doi={10.5281/zenodo.XXXXXXX},
     note={Computer software}
   }

**EndNote:**

.. code-block:: text

   %0 Computer Program
   %T ProteinMD: A Python Framework for Molecular Dynamics Simulations
   %A ProteinMD Development Team
   %D 2024
   %7 1.0.0
   %U https://github.com/proteinmd/proteinmd
   %R doi:10.5281/zenodo.XXXXXXX

Academic Publication
~~~~~~~~~~~~~~~~~~~

When the primary ProteinMD paper is published, please use:

**Planned Publication:**

.. code-block:: text

   Smith, J., Johnson, A., & ProteinMD Development Team. (2024). 
   ProteinMD: An Open-Source Python Framework for Molecular Dynamics 
   Simulations of Biological Systems. Journal of Computational Chemistry, 
   XX(X), XXX-XXX. https://doi.org/XX.XXXX/jcc.XXXXX

Component-Specific Citations
============================

Force Fields
~~~~~~~~~~~

If you use specific force fields implemented in ProteinMD, please also cite the original force field papers:

**AMBER ff14SB:**

.. code-block:: bibtex

   @article{maier2015ff14sb,
     title={ff14SB: improving the accuracy of protein side chain and backbone parameters from ff99SB},
     author={Maier, James A and Martinez, Carmenza and Kasavajhala, Koushik and Wickstrom, Lauren and Hauser, Kevin E and Simmerling, Carlos},
     journal={Journal of Chemical Theory and Computation},
     volume={11},
     number={8},
     pages={3696--3713},
     year={2015},
     publisher={ACS Publications},
     doi={10.1021/acs.jctc.5b00255}
   }

**CHARMM36:**

.. code-block:: bibtex

   @article{best2012charmm36,
     title={Optimization of the additive CHARMM all-atom protein force field targeting improved sampling of the backbone φ, ψ and side-chain χ1 and χ2 dihedral angles},
     author={Best, Robert B and Zhu, Xiao and Shim, Jihyun and Lopes, Pedro EM and Mittal, Jeetain and Feig, Michael and MacKerell Jr, Alexander D},
     journal={Journal of Chemical Theory and Computation},
     volume={8},
     number={9},
     pages={3257--3273},
     year={2012},
     publisher={ACS Publications},
     doi={10.1021/ct300400x}
   }

Water Models
~~~~~~~~~~~

**TIP3P:**

.. code-block:: bibtex

   @article{jorgensen1983tip3p,
     title={Comparison of simple potential functions for simulating liquid water},
     author={Jorgensen, William L and Chandrasekhar, Jayaraman and Madura, Jeffry D and Impey, Roger W and Klein, Michael L},
     journal={The Journal of Chemical Physics},
     volume={79},
     number={2},
     pages={926--935},
     year={1983},
     publisher={American Institute of Physics},
     doi={10.1063/1.445869}
   }

**TIP4P/Ew:**

.. code-block:: bibtex

   @article{horn2004tip4pew,
     title={Development of an improved four-site water model for biomolecular simulations: TIP4P-Ew},
     author={Horn, Hans W and Swope, William C and Pitera, Jed W and Madura, Jeffry D and Dick, Thomas J and Hura, Greg L and Head-Gordon, Teresa},
     journal={The Journal of Chemical Physics},
     volume={120},
     number={20},
     pages={9665--9678},
     year={2004},
     publisher={American Institute of Physics},
     doi={10.1063/1.1683075}
   }

Analysis Methods
~~~~~~~~~~~~~~~

If you use specific analysis methods, consider citing the original methodology papers:

**RMSD with Kabsch Algorithm:**

.. code-block:: bibtex

   @article{kabsch1976solution,
     title={A solution for the best rotation to relate two sets of vectors},
     author={Kabsch, Wolfgang},
     journal={Acta Crystallographica Section A: Crystal Physics, Diffraction, Theoretical and General Crystallography},
     volume={32},
     number={5},
     pages={922--923},
     year={1976},
     publisher={International Union of Crystallography},
     doi={10.1107/S0567739476001873}
   }

**DSSP (Secondary Structure):**

.. code-block:: bibtex

   @article{kabsch1983dssp,
     title={Dictionary of protein secondary structure: pattern recognition of hydrogen-bonded and geometrical features},
     author={Kabsch, Wolfgang and Sander, Christian},
     journal={Biopolymers},
     volume={22},
     number={12},
     pages={2577--2637},
     year={1983},
     publisher={Wiley Online Library},
     doi={10.1002/bip.360221211}
   }

Backend Dependencies
====================

OpenMM
~~~~~~

If you use OpenMM as the simulation backend:

.. code-block:: bibtex

   @article{eastman2017openmm,
     title={OpenMM 7: Rapid development of high performance algorithms for molecular dynamics},
     author={Eastman, Peter and Swails, Jason and Chodera, John D and McGibbon, Robert T and Zhao, Yutong and Beauchamp, Kyle A and Wang, Lee-Ping and Simmonett, Andrew C and Harrigan, Matthew P and Stern, Chaya D and others},
     journal={PLoS Computational Biology},
     volume={13},
     number={7},
     pages={e1005659},
     year={2017},
     publisher={Public Library of Science},
     doi={10.1371/journal.pcbi.1005659}
   }

MDAnalysis
~~~~~~~~~

If you use MDAnalysis components:

.. code-block:: bibtex

   @article{michaud2011mdanalysis,
     title={MDAnalysis: a toolkit for the analysis of molecular dynamics simulations},
     author={Michaud-Agrawal, Naveen and Denning, Elizabeth J and Woolf, Thomas B and Beckstein, Oliver},
     journal={Journal of Computational Chemistry},
     volume={32},
     number={10},
     pages={2319--2327},
     year={2011},
     publisher={Wiley Online Library},
     doi={10.1002/jcc.21787}
   }

Citation Examples
-----------------

Research Paper Examples
~~~~~~~~~~~~~~~~~~~~~~

**In Methods Section:**

.. code-block:: text

   Molecular dynamics simulations were performed using ProteinMD version 1.0.0 
   (ProteinMD Development Team, 2024), a Python-based framework for MD simulations. 
   The AMBER ff14SB force field (Maier et al., 2015) was used for protein 
   parameters, and the TIP3P water model (Jorgensen et al., 1983) was employed 
   for explicit solvation. All simulations were run using the OpenMM backend 
   (Eastman et al., 2017) with GPU acceleration.

**In Acknowledgments:**

.. code-block:: text

   The authors thank the ProteinMD Development Team for developing and maintaining 
   the ProteinMD simulation framework used in this work.

**In Software Availability:**

.. code-block:: text

   Software Availability: The ProteinMD software used in this study is freely 
   available at https://github.com/proteinmd/proteinmd under the MIT license. 
   All simulation input files and analysis scripts are available as 
   Supplementary Material.

Thesis and Dissertation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   All molecular dynamics simulations in this work were performed using the 
   ProteinMD framework (ProteinMD Development Team, 2024), which provides a 
   Python-based interface for setting up, running, and analyzing MD simulations. 
   The software was chosen for its ease of use, extensive documentation, and 
   robust analysis capabilities.

Grant Proposals
~~~~~~~~~~~~~~

.. code-block:: text

   We will utilize ProteinMD (ProteinMD Development Team, 2024), an open-source 
   Python framework for molecular dynamics simulations, to investigate protein 
   conformational dynamics. This software provides the necessary tools for 
   force field parameterization, simulation setup, and trajectory analysis 
   required for the proposed research.

Version-Specific Citations
===========================

Citing Specific Versions
~~~~~~~~~~~~~~~~~~~~~~~

Always specify the version of ProteinMD used in your research:

.. code-block:: python

   # Check your ProteinMD version
   import proteinmd
   print(f"ProteinMD version: {proteinmd.__version__}")

**For different versions:**

.. code-block:: bibtex

   % Version 1.0.0
   @software{proteinmd2024_v1,
     title={ProteinMD: A Python Framework for Molecular Dynamics Simulations},
     author={{ProteinMD Development Team}},
     year={2024},
     version={1.0.0},
     url={https://github.com/proteinmd/proteinmd},
     doi={10.5281/zenodo.XXXXXXX}
   }
   
   % Version 1.1.0
   @software{proteinmd2024_v11,
     title={ProteinMD: A Python Framework for Molecular Dynamics Simulations},
     author={{ProteinMD Development Team}},
     year={2024},
     version={1.1.0},
     url={https://github.com/proteinmd/proteinmd},
     doi={10.5281/zenodo.YYYYYYY}
   }

Development Versions
~~~~~~~~~~~~~~~~~~~

For pre-release or development versions:

.. code-block:: bibtex

   @software{proteinmd2024_dev,
     title={ProteinMD: A Python Framework for Molecular Dynamics Simulations},
     author={{ProteinMD Development Team}},
     year={2024},
     version={1.1.0-dev},
     url={https://github.com/proteinmd/proteinmd},
     note={Development version}
   }

Data and Code Availability
--------------------------

Research Data Management
~~~~~~~~~~~~~~~~~~~~~~~

When publishing research using ProteinMD, consider:

**Data Availability Statement:**

.. code-block:: text

   The raw simulation data supporting the conclusions of this article are 
   available from the corresponding author upon reasonable request. All 
   ProteinMD input files, configuration scripts, and analysis code are 
   provided as Supplementary Material and are also available at 
   [repository URL].

**Code Availability:**

.. code-block:: text

   All analysis scripts and simulation setup files used in this study are 
   available at [GitHub repository URL]. The scripts are compatible with 
   ProteinMD version 1.0.0 and later.

Reproducibility
~~~~~~~~~~~~~~

To ensure reproducibility:

1. **Specify exact software versions** used
2. **Provide complete parameter files** and configurations  
3. **Include random number seeds** if applicable
4. **Document hardware specifications** for performance benchmarks
5. **Share analysis scripts** and custom code

.. code-block:: python

   # Example reproducibility information to include
   import proteinmd
   import numpy as np
   
   print(f"ProteinMD version: {proteinmd.__version__}")
   print(f"NumPy version: {np.__version__}")
   print(f"Python version: {sys.version}")
   print(f"Random seed used: 12345")

Contributing Authors
====================

Core Development Team
~~~~~~~~~~~~~~~~~~~~

If you contribute significantly to ProteinMD development, you may be included in the core development team citations:

**Current Core Team:**

- Development Lead: [Name] (Institution)
- Core Developers: [Names and Institutions]
- Scientific Advisors: [Names and Institutions]

**Contributing:**

See :doc:`../developer/contributing` for information on how to contribute to ProteinMD development and potentially be included in future citations.

Recognition
~~~~~~~~~~

**Types of Contributions Recognized:**

- Code contributions (new features, bug fixes)
- Documentation improvements
- Testing and quality assurance
- Scientific validation studies
- User support and community building
- Tutorial and example development

**Contributor Recognition:**

Contributors are acknowledged in:

- Release notes and changelogs
- AUTHORS file in the repository
- Annual contributor recognition
- Conference presentations about ProteinMD

Citation Tools
==============

Reference Managers
~~~~~~~~~~~~~~~~~

**Mendeley/Zotero Import:**

You can import ProteinMD citations directly using the DOI:

.. code-block:: text

   DOI: 10.5281/zenodo.XXXXXXX

**Citation Management:**

For large projects using multiple versions or components:

1. Create a dedicated collection/folder for ProteinMD citations
2. Include version numbers in citation titles
3. Tag citations by component (force field, analysis method, etc.)
4. Maintain notes about which version was used for which analysis

Automated Citation
~~~~~~~~~~~~~~~~~

**In Python Scripts:**

.. code-block:: python

   import proteinmd
   
   # Get citation information programmatically
   citation_info = proteinmd.get_citation_info()
   print("Please cite ProteinMD as:")
   print(citation_info['apa_format'])
   print("\nBibTeX entry:")
   print(citation_info['bibtex'])

**In Jupyter Notebooks:**

.. code-block:: python

   # Display citation reminder
   proteinmd.show_citation_reminder()

Contact for Citations
=====================

Questions about Citations
~~~~~~~~~~~~~~~~~~~~~~~~~

For questions about proper citation:

- **Email**: citations@proteinmd.org
- **GitHub Issues**: Use the "citation" label
- **Documentation**: Check this page for updates

**Common Citation Questions:**

1. **Which components to cite?** - Cite the main ProteinMD software plus any specific methods/force fields used
2. **Multiple versions used?** - Cite each version separately if results differ
3. **Collaborative projects?** - Each group should cite the version they used
4. **Review papers?** - Include ProteinMD in software/tools sections

Updates and Changes
~~~~~~~~~~~~~~~~~~

Citation information may be updated when:

- New versions are released
- Primary research paper is published
- DOI registration is completed
- Contributor list changes significantly

Check this page regularly or subscribe to ProteinMD announcements for citation updates.

See Also
--------

* :doc:`license` - License information
* :doc:`changelog` - Version history
* :doc:`../developer/contributing` - How to contribute
* `FORCE11 Software Citation Principles <https://force11.org/info/software-citation-principles-published-2016/>`_ - Citation best practices
* `Zenodo <https://zenodo.org/>`_ - Software DOI registration
