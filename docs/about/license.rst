License
=======

ProteinMD is released under the MIT License.

MIT License
-----------

Copyright (c) 2024 ProteinMD Development Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Third-Party Licenses
--------------------

ProteinMD includes or depends on several third-party libraries with their own licenses:

Dependencies
~~~~~~~~~~~

**Required Dependencies**

- **NumPy** - BSD 3-Clause License
  
  - Copyright (c) 2005-2024, NumPy Developers
  - Used for numerical computations and array operations

- **SciPy** - BSD 3-Clause License
  
  - Copyright (c) 2001-2024, SciPy Developers  
  - Used for scientific computing functions

- **Matplotlib** - PSF License (Python Software Foundation License)
  
  - Copyright (c) 2012-2024, Matplotlib Development Team
  - Used for plotting and visualization

**Optional Dependencies**

- **OpenMM** - MIT License
  
  - Copyright (c) 2008-2024, Stanford University
  - Used as simulation backend for GPU acceleration

- **MDAnalysis** - GPL v2 License
  
  - Copyright (c) 2006-2024, MDAnalysis Development Team
  - Used for trajectory analysis and file I/O

- **PyMOL** - Custom License (Commercial/Educational)
  
  - Copyright (c) Schrödinger, Inc.
  - Used for molecular visualization (optional)

- **VMD** - Custom License (Free for non-commercial use)
  
  - Copyright (c) University of Illinois
  - Used for molecular visualization (optional)

Force Field Parameters
~~~~~~~~~~~~~~~~~~~~~

**AMBER Force Fields**

- **AMBER ff14SB** - AMBER License
  
  - Copyright (c) University of California
  - Freely available for academic and commercial use
  - Reference: Maier et al. (2015) J. Chem. Theory Comput. 11, 3696-3713

- **AMBER GAFF/GAFF2** - AMBER License
  
  - Copyright (c) University of California
  - General AMBER Force Field for small molecules

**CHARMM Force Fields**

- **CHARMM36** - CHARMM License
  
  - Copyright (c) Harvard University and University of Maryland
  - Freely available for academic use
  - Reference: Best et al. (2012) J. Chem. Theory Comput. 8, 3257-3273

**Water Models**

- **TIP3P** - Public Domain
  
  - Reference: Jorgensen et al. (1983) J. Chem. Phys. 79, 926

- **TIP4P/Ew** - Public Domain
  
  - Reference: Horn et al. (2004) J. Chem. Phys. 120, 9665

Data and Examples
~~~~~~~~~~~~~~~~

**Test Structures**

- **1UBQ (Ubiquitin)** - Protein Data Bank (PDB)
  
  - PDB ID: 1UBQ
  - Copyright: Public Domain (wwPDB)
  - Reference: Vijay-Kumar et al. (1987) J. Mol. Biol. 194, 531-544

- **Example Protein Structures** - Various PDB Sources
  
  - All structures from the Protein Data Bank are in the public domain
  - Individual structures may have associated publications

Documentation Assets
~~~~~~~~~~~~~~~~~~~

**Icons and Graphics**

- **Font Awesome Icons** - Font Awesome Free License (SIL OFL 1.1 + MIT)
  
  - Copyright (c) Fonticons, Inc.
  - Used for documentation icons

- **Custom Graphics** - MIT License
  
  - Created specifically for ProteinMD documentation
  - Copyright (c) 2024 ProteinMD Development Team

License Compatibility
--------------------

Permissive Licenses
~~~~~~~~~~~~~~~~~~

The MIT License used by ProteinMD is compatible with most other open-source licenses, including:

- **BSD Licenses** (2-clause, 3-clause, 4-clause)
- **Apache License 2.0**
- **ISC License**
- **X11 License**

Copyleft Licenses
~~~~~~~~~~~~~~~~

Some dependencies use copyleft licenses that may affect distribution:

- **GPL v2/v3**: MDAnalysis uses GPL v2, which means any derivative work that includes MDAnalysis must also be GPL-compatible
- **LGPL**: Some numerical libraries use LGPL, which allows linking but requires source availability for the LGPL components

Commercial Use
~~~~~~~~~~~~~

The MIT License allows commercial use of ProteinMD. However, users should be aware of:

1. **Third-party license obligations**: Some dependencies may have different licensing terms
2. **Patent considerations**: While MIT doesn't grant patent rights, some dependencies might
3. **Trademark restrictions**: The ProteinMD name and logo may be subject to trademark protection

Attribution Requirements
-----------------------

Required Attribution
~~~~~~~~~~~~~~~~~~~

When using ProteinMD in academic work, please cite:

.. code-block:: text

   ProteinMD Development Team. (2024). ProteinMD: A Python Framework for 
   Molecular Dynamics Simulations. Version X.Y.Z. 
   Available at: https://github.com/proteinmd/proteinmd

Optional Attribution
~~~~~~~~~~~~~~~~~~~

For software that incorporates ProteinMD, consider including:

.. code-block:: text

   This software uses ProteinMD (https://proteinmd.org), 
   licensed under the MIT License.

Disclaimer
----------

Warranty Disclaimer
~~~~~~~~~~~~~~~~~~

ProteinMD is provided "as is" without warranty of any kind. The developers make no representations or warranties regarding:

- **Accuracy**: Results produced by the software
- **Reliability**: Continuous operation without interruption
- **Fitness**: Suitability for any particular purpose
- **Non-infringement**: Freedom from third-party intellectual property claims

Limitation of Liability
~~~~~~~~~~~~~~~~~~~~~~

In no event shall the ProteinMD developers be liable for any damages including:

- Direct, indirect, incidental, or consequential damages
- Loss of profits, data, or business interruption
- Any damages arising from use or inability to use the software

Scientific Use Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using ProteinMD for scientific research:

1. **Validation**: Always validate results against experimental data or established methods
2. **Method verification**: Understand the limitations and assumptions of the methods used
3. **Publication standards**: Follow appropriate scientific standards for reporting computational results
4. **Reproducibility**: Provide sufficient detail for others to reproduce your work

License Updates
--------------

License Changes
~~~~~~~~~~~~~~

The ProteinMD development team reserves the right to change the license for future versions. However:

- Existing versions will retain their current license
- Any license change will be clearly communicated
- We aim to maintain permissive licensing to support the scientific community

Version-Specific Licensing
~~~~~~~~~~~~~~~~~~~~~~~~~

Different versions of ProteinMD may have different licensing terms:

- **Version 1.x**: MIT License (current)
- **Pre-release versions**: May have different or additional restrictions
- **Development branches**: Follow the same license as the main branch

Contact Information
------------------

License Questions
~~~~~~~~~~~~~~~~

For questions about licensing, contact:

- **Email**: legal@proteinmd.org
- **GitHub**: Open an issue with the "license" label
- **Documentation**: See the LICENSE file in the repository

Compliance Issues
~~~~~~~~~~~~~~~~

If you believe there is a license compliance issue:

1. Contact us immediately via legal@proteinmd.org
2. Provide detailed information about the concern
3. We will investigate and respond promptly

Contributing and Licensing
~~~~~~~~~~~~~~~~~~~~~~~~~

When contributing to ProteinMD:

- All contributions are submitted under the MIT License
- Contributors retain copyright to their contributions
- By submitting a pull request, you agree to license your contribution under MIT
- See :doc:`../developer/contributing` for details

See Also
--------

* :doc:`citation` - How to cite ProteinMD
* :doc:`changelog` - Version history and changes
* :doc:`../developer/contributing` - Contributing guidelines
* `MIT License <https://opensource.org/licenses/MIT>`_ - Full license text
* `Open Source Initiative <https://opensource.org/>`_ - Open source licensing information
