Documentation Guidelines
=======================

This guide covers how to write, build, and maintain documentation for ProteinMD.

.. contents:: Documentation Topics
   :local:
   :depth: 2

Documentation Structure
----------------------

Overview
~~~~~~~~

ProteinMD documentation is built using Sphinx and follows a structured approach:

.. code-block:: text

   docs/
   ├── conf.py                 # Sphinx configuration
   ├── index.rst              # Main documentation index
   ├── _static/               # Static files (CSS, images)
   │   ├── custom.css
   │   └── logo.png
   ├── _templates/            # Custom templates
   ├── api/                   # API reference documentation
   │   ├── index.rst
   │   ├── core.rst
   │   ├── structure.rst
   │   ├── forcefield.rst
   │   ├── analysis.rst
   │   └── ...
   ├── user_guide/           # User guides and tutorials
   │   ├── installation.rst
   │   ├── quick_start.rst
   │   ├── tutorials.rst
   │   └── examples.rst
   ├── advanced/             # Advanced topics
   │   ├── performance.rst
   │   ├── validation.rst
   │   ├── extending.rst
   │   └── troubleshooting.rst
   ├── developer/            # Developer documentation
   │   ├── architecture.rst
   │   ├── contributing.rst
   │   ├── testing.rst
   │   └── documentation.rst
   └── about/               # Project information
       ├── license.rst
       ├── changelog.rst
       └── citation.rst

Documentation Types
~~~~~~~~~~~~~~~~~

**1. API Reference**

Auto-generated from docstrings using Sphinx autodoc:

.. code-block:: rst

   .. automodule:: proteinmd.analysis.rmsd
      :members:
      :undoc-members:
      :show-inheritance:

**2. User Guides**

Step-by-step instructions for common tasks:

.. code-block:: rst

   Quick Start Tutorial
   ===================
   
   This tutorial walks you through your first MD simulation...
   
   .. code-block:: python
   
      from proteinmd import *
      
      # Your first simulation
      simulation = MDSimulation(...)

**3. Examples**

Practical code examples with explanations:

.. code-block:: rst

   Examples
   ========
   
   Basic Protein Analysis
   ~~~~~~~~~~~~~~~~~~~~
   
   This example shows how to analyze protein dynamics:
   
   .. literalinclude:: ../examples/basic_analysis.py
      :language: python
      :lines: 1-20

**4. Advanced Topics**

In-depth technical documentation:

.. code-block:: rst

   Performance Optimization
   =======================
   
   This section covers advanced performance tuning...

Writing Documentation
--------------------

RestructuredText Syntax
~~~~~~~~~~~~~~~~~~~~~~~

**Basic Formatting**

.. code-block:: rst

   Section Headers
   ===============
   
   Subsection Headers
   ~~~~~~~~~~~~~~~~~~
   
   **Bold text**
   *Italic text*
   ``Inline code``
   
   - Bullet list item
   - Another item
   
   1. Numbered list
   2. Second item
   
   .. note::
      This is a note box.
   
   .. warning::
      This is a warning box.

**Code Blocks**

.. code-block:: rst

   .. code-block:: python
      :linenos:
      :emphasize-lines: 2,3
   
      from proteinmd import MDSimulation
      simulation = MDSimulation()  # emphasized
      simulation.run(steps=1000)   # emphasized
   
   .. literalinclude:: example.py
      :language: python
      :lines: 10-20
      :dedent: 4

**Cross-References**

.. code-block:: rst

   # Reference other documents
   See :doc:`../user_guide/installation` for setup instructions.
   
   # Reference API elements
   The :class:`proteinmd.core.MDSimulation` class...
   Use the :func:`proteinmd.analysis.calculate_rmsd` function...
   
   # External links
   See the `OpenMM documentation <http://openmm.org>`_ for details.

**Math and Equations**

.. code-block:: rst

   The RMSD is calculated as:
   
   .. math::
      RMSD = \sqrt{\frac{1}{N} \sum_{i=1}^{N} |r_i - r_i'|^2}
   
   Inline math: :math:`E = mc^2`

Docstring Standards
~~~~~~~~~~~~~~~~~~

**NumPy Style Docstrings**

.. code-block:: python

   def calculate_rmsd(
       coords1: np.ndarray,
       coords2: np.ndarray,
       align: bool = True
   ) -> float:
       """
       Calculate root mean square deviation between coordinate sets.
       
       This function computes the RMSD between two sets of coordinates,
       optionally after optimal structural alignment using the Kabsch
       algorithm.
       
       Parameters
       ----------
       coords1 : np.ndarray, shape (n_atoms, 3)
           First set of coordinates in nanometers.
       coords2 : np.ndarray, shape (n_atoms, 3)
           Second set of coordinates in nanometers.
       align : bool, default=True
           Whether to optimally align structures before RMSD calculation.
           If False, structures are assumed to be pre-aligned.
       
       Returns
       -------
       float
           RMSD value in nanometers.
       
       Raises
       ------
       ValueError
           If coordinate arrays have different shapes or contain invalid values.
       TypeError
           If input coordinates are not numpy arrays.
       
       See Also
       --------
       align_structures : Align two structures optimally.
       calculate_rmsf : Calculate root mean square fluctuations.
       
       Notes
       -----
       The RMSD calculation follows the standard formula:
       
       .. math::
           RMSD = \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} |r_i - r_i'|^2}
       
       When `align=True`, the Kabsch algorithm [1]_ is used to find the
       optimal rotation matrix that minimizes the RMSD.
       
       Examples
       --------
       Calculate RMSD between two random coordinate sets:
       
       >>> import numpy as np
       >>> coords1 = np.random.rand(100, 3)
       >>> coords2 = np.random.rand(100, 3) 
       >>> rmsd = calculate_rmsd(coords1, coords2)
       >>> print(f"RMSD: {rmsd:.3f} nm")
       RMSD: 0.456 nm
       
       Calculate RMSD without alignment:
       
       >>> rmsd_no_align = calculate_rmsd(coords1, coords2, align=False)
       >>> print(f"RMSD (no alignment): {rmsd_no_align:.3f} nm")
       RMSD (no alignment): 0.523 nm
       
       References
       ----------
       .. [1] Kabsch, W. (1976). A solution for the best rotation to relate
              two sets of vectors. Acta Crystallographica Section A, 32(5), 922-923.
       """

**Class Docstrings**

.. code-block:: python

   class MDSimulation:
       """
       Main class for molecular dynamics simulations.
       
       This class provides a high-level interface for setting up and running
       molecular dynamics simulations with various integrators, thermostats,
       and barostats.
       
       Parameters
       ----------
       system : System
           The molecular system to simulate.
       integrator : Integrator
           Integration algorithm for time evolution.
       thermostat : Thermostat, optional
           Temperature control method.
       barostat : Barostat, optional
           Pressure control method.
       
       Attributes
       ----------
       current_step : int
           Current simulation step number.
       current_time : float
           Current simulation time in picoseconds.
       reporters : list
           List of output reporters.
       
       Examples
       --------
       Set up a basic NVT simulation:
       
       >>> from proteinmd import *
       >>> system = create_system_from_pdb("protein.pdb")
       >>> integrator = VelocityVerletIntegrator(timestep=0.002)
       >>> thermostat = LangevinThermostat(temperature=300.0)
       >>> simulation = MDSimulation(system, integrator, thermostat)
       >>> simulation.run(steps=100000)
       
       Set up an NPT simulation:
       
       >>> barostat = BerendsenBarostat(pressure=1.0)
       >>> simulation = MDSimulation(system, integrator, thermostat, barostat)
       >>> simulation.minimize_energy()
       >>> simulation.run(steps=500000)
       """

API Documentation
----------------

Automatic Generation
~~~~~~~~~~~~~~~~~~~

**Sphinx Autodoc Configuration**

.. code-block:: python

   # conf.py
   extensions = [
       'sphinx.ext.autodoc',
       'sphinx.ext.autosummary',
       'sphinx.ext.napoleon',
       'sphinx.ext.viewcode',
       'sphinx.ext.intersphinx'
   ]
   
   # Autodoc settings
   autodoc_default_options = {
       'members': True,
       'undoc-members': True,
       'show-inheritance': True,
       'member-order': 'bysource'
   }
   
   # Napoleon settings for numpy-style docstrings
   napoleon_google_docstring = False
   napoleon_numpy_docstring = True
   napoleon_include_init_with_doc = False
   napoleon_include_private_with_doc = False
   napoleon_use_param = True
   napoleon_use_rtype = True

**Module Documentation Template**

.. code-block:: rst

   Core Module (:mod:`proteinmd.core`)
   ===================================
   
   The core module provides the fundamental simulation engine components.
   
   .. currentmodule:: proteinmd.core
   
   Simulation Classes
   ------------------
   
   .. autosummary::
      :toctree: generated/
      
      MDSimulation
      System
      State
   
   Integrators
   -----------
   
   .. autosummary::
      :toctree: generated/
      
      Integrator
      VelocityVerletIntegrator
      LeapFrogIntegrator
      LangevinIntegrator
   
   Detailed API
   ------------
   
   .. automodule:: proteinmd.core
      :members:
      :undoc-members:
      :show-inheritance:

**Custom Directives**

.. code-block:: python

   # Custom Sphinx directive for examples
   from docutils import nodes
   from docutils.parsers.rst import directives
   from sphinx.util.docutils import SphinxDirective
   
   
   class ExampleDirective(SphinxDirective):
       """Custom directive for code examples."""
       
       has_content = True
       required_arguments = 0
       optional_arguments = 1
       option_spec = {
           'language': directives.unchanged,
           'linenos': directives.flag,
       }
       
       def run(self):
           language = self.options.get('language', 'python')
           code = '\n'.join(self.content)
           
           # Create code block node
           literal_node = nodes.literal_block(code, code)
           literal_node['language'] = language
           
           if 'linenos' in self.options:
               literal_node['linenos'] = True
           
           return [literal_node]

Building Documentation
---------------------

Local Development
~~~~~~~~~~~~~~~~

**Setup Documentation Environment**

.. code-block:: bash

   # Create docs environment
   conda create -n proteinmd-docs python=3.9
   conda activate proteinmd-docs
   
   # Install documentation dependencies
   pip install -r docs/requirements.txt
   pip install -e .  # Install ProteinMD in development mode

**Build Documentation**

.. code-block:: bash

   # Navigate to docs directory
   cd docs/
   
   # Clean previous builds
   make clean
   
   # Build HTML documentation
   make html
   
   # Open documentation
   open _build/html/index.html

**Live Preview**

.. code-block:: bash

   # Install sphinx-autobuild for live preview
   pip install sphinx-autobuild
   
   # Start live preview server
   sphinx-autobuild . _build/html --host 0.0.0.0 --port 8000
   
   # Documentation will auto-rebuild on changes
   # View at http://localhost:8000

**Check for Issues**

.. code-block:: bash

   # Check for broken links
   make linkcheck
   
   # Check for warnings
   make html 2>&1 | grep WARNING
   
   # Spell check (if aspell installed)
   make spelling

Continuous Integration
~~~~~~~~~~~~~~~~~~~~~

**GitHub Actions for Documentation**

.. code-block:: yaml

   # .github/workflows/docs.yml
   name: Documentation
   
   on:
     push:
       branches: [ main ]
     pull_request:
       branches: [ main ]
   
   jobs:
     docs:
       runs-on: ubuntu-latest
       
       steps:
       - uses: actions/checkout@v3
       
       - name: Set up Python
         uses: actions/setup-python@v4
         with:
           python-version: "3.10"
       
       - name: Install dependencies
         run: |
           pip install -r docs/requirements.txt
           pip install -e .
       
       - name: Build documentation
         run: |
           cd docs/
           make html SPHINXOPTS="-W --keep-going"
       
       - name: Check links
         run: |
           cd docs/
           make linkcheck
       
       - name: Deploy to GitHub Pages
         if: github.ref == 'refs/heads/main'
         uses: peaceiris/actions-gh-pages@v3
         with:
           github_token: ${{ secrets.GITHUB_TOKEN }}
           publish_dir: ./docs/_build/html

Documentation Quality
--------------------

Content Guidelines
~~~~~~~~~~~~~~~~~

**Writing Style**

1. **Clear and Concise**: Use simple, direct language
2. **Active Voice**: Prefer active voice over passive
3. **Consistent Terminology**: Use consistent terms throughout
4. **User-Focused**: Write from the user's perspective

**Example Style Guide**

.. code-block:: rst

   # Good
   Calculate the RMSD between two structures:
   
   # Bad
   The RMSD between two structures can be calculated:
   
   # Good
   This function returns the RMSD value.
   
   # Bad
   The RMSD value is returned by this function.

**Code Examples**

1. **Complete Examples**: Provide runnable code
2. **Realistic Use Cases**: Use practical scenarios
3. **Error Handling**: Show how to handle common errors
4. **Comments**: Explain non-obvious parts

.. code-block:: python

   # Good example - complete and realistic
   from proteinmd import MDSimulation, VelocityVerletIntegrator
   from proteinmd.analysis import RMSD
   
   # Load protein structure
   protein = ProteinStructure.from_pdb("protein.pdb")
   
   # Set up simulation
   integrator = VelocityVerletIntegrator(timestep=0.002)
   simulation = MDSimulation(protein, integrator)
   
   # Run simulation
   try:
       simulation.run(steps=10000)
   except SimulationError as e:
       print(f"Simulation failed: {e}")
       # Handle error appropriately
   
   # Analyze results
   trajectory = simulation.get_trajectory()
   rmsd_calc = RMSD(reference=trajectory[0])
   rmsd_values = [rmsd_calc.calculate(frame) for frame in trajectory]

Review Process
~~~~~~~~~~~~~

**Documentation Review Checklist**

.. code-block:: text

   Content Review:
   - [ ] Accurate and up-to-date information
   - [ ] Clear and understandable language
   - [ ] Complete examples that work
   - [ ] Appropriate level of detail
   - [ ] Consistent terminology
   
   Technical Review:
   - [ ] Code examples run without errors
   - [ ] API documentation matches implementation
   - [ ] Cross-references work correctly
   - [ ] No broken links
   - [ ] Proper formatting and markup
   
   Style Review:
   - [ ] Follows project style guide
   - [ ] Consistent formatting
   - [ ] Good organization and flow
   - [ ] Appropriate use of directives and markup

**Automated Quality Checks**

.. code-block:: python

   # check_docs.py - Documentation quality checker
   import re
   import sys
   from pathlib import Path
   
   
   class DocumentationChecker:
       """Check documentation quality."""
       
       def __init__(self, docs_dir="docs"):
           self.docs_dir = Path(docs_dir)
           self.issues = []
       
       def check_all(self):
           """Run all documentation checks."""
           self.check_broken_references()
           self.check_code_blocks()
           self.check_style_consistency()
           
           return len(self.issues) == 0
       
       def check_broken_references(self):
           """Check for broken cross-references."""
           for rst_file in self.docs_dir.rglob("*.rst"):
               content = rst_file.read_text()
               
               # Check for broken :doc: references
               doc_refs = re.findall(r':doc:`([^`]+)`', content)
               for ref in doc_refs:
                   if not self._reference_exists(ref, rst_file):
                       self.issues.append(
                           f"Broken reference in {rst_file}: {ref}"
                       )
       
       def check_code_blocks(self):
           """Check code blocks for syntax errors."""
           for rst_file in self.docs_dir.rglob("*.rst"):
               content = rst_file.read_text()
               
               # Extract Python code blocks
               code_blocks = re.findall(
                   r'.. code-block:: python\n\n((?:   .*\n)*)',
                   content
               )
               
               for i, code_block in enumerate(code_blocks):
                   # Remove indentation
                   code = '\n'.join(line[3:] for line in code_block.split('\n') if line.strip())
                   
                   try:
                       compile(code, f"{rst_file}:block_{i}", "exec")
                   except SyntaxError as e:
                       self.issues.append(
                           f"Syntax error in {rst_file}, block {i}: {e}"
                       )

Accessibility
------------

Making Documentation Accessible
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Screen Reader Support**

.. code-block:: rst

   # Use descriptive alt text for images
   .. image:: molecular_structure.png
      :alt: 3D structure of ubiquitin protein showing alpha helices in red and beta sheets in blue
   
   # Provide text descriptions for complex figures
   .. figure:: energy_plot.png
      :alt: Energy vs time plot
      
      Energy conservation during MD simulation. The plot shows potential energy 
      (blue line) and kinetic energy (red line) over 10 nanoseconds. Both energies 
      fluctuate around constant values, indicating good energy conservation.

**Color Accessibility**

.. code-block:: css

   /* Use colorblind-friendly palettes */
   .code-block {
       background-color: #f8f8f8;
       border-left: 4px solid #e74c3c;  /* Use patterns in addition to color */
   }
   
   .warning {
       background-color: #fff3cd;
       border: 1px solid #ffeaa7;
       /* Use icons in addition to color coding */
   }

**Keyboard Navigation**

.. code-block:: html

   <!-- Ensure all interactive elements are keyboard accessible -->
   <a href="#section1" class="reference internal">
       <span class="doc">Installation Guide</span>
   </a>

Translation Support
~~~~~~~~~~~~~~~~~~

**Internationalization Setup**

.. code-block:: python

   # conf.py - Translation configuration
   language = 'en'
   locale_dirs = ['locale/']
   gettext_compact = False
   
   # Generate translation templates
   extensions.append('sphinx.ext.intersphinx')

**Translation Workflow**

.. code-block:: bash

   # Extract translatable strings
   make gettext
   
   # Create language-specific directories
   sphinx-intl update -p _build/gettext -l es -l fr -l de
   
   # Build translated documentation
   make -e SPHINXOPTS="-D language='es'" html

Maintenance
----------

Regular Updates
~~~~~~~~~~~~~~

**Documentation Maintenance Checklist**

.. code-block:: text

   Monthly:
   - [ ] Check for broken links
   - [ ] Update version numbers
   - [ ] Review recent API changes
   - [ ] Test all code examples
   
   Quarterly:
   - [ ] Review user feedback
   - [ ] Update screenshots and figures
   - [ ] Check for outdated information
   - [ ] Performance documentation updates
   
   Per Release:
   - [ ] Update changelog
   - [ ] Review API documentation
   - [ ] Update installation instructions
   - [ ] Check example compatibility

**Automated Maintenance**

.. code-block:: python

   # maintenance_bot.py - Automated documentation maintenance
   import subprocess
   import json
   from pathlib import Path
   
   
   class DocMaintenanceBot:
       """Automated documentation maintenance."""
       
       def __init__(self):
           self.issues = []
       
       def run_checks(self):
           """Run automated maintenance checks."""
           self.check_links()
           self.validate_examples()
           self.update_api_docs()
           
           return self.generate_report()
       
       def check_links(self):
           """Check for broken links."""
           result = subprocess.run(
               ["sphinx-build", "-b", "linkcheck", ".", "_build/linkcheck"],
               capture_output=True,
               text=True
           )
           
           if result.returncode != 0:
               self.issues.append({
                   "type": "broken_links",
                   "details": result.stdout
               })
       
       def validate_examples(self):
           """Validate code examples."""
           # Implementation for validating examples
           pass

Analytics and Feedback
~~~~~~~~~~~~~~~~~~~~~

**Documentation Analytics**

.. code-block:: html

   <!-- Add Google Analytics or similar -->
   <script>
     // Track popular pages and user behavior
     gtag('config', 'GA_MEASUREMENT_ID');
   </script>

**User Feedback Collection**

.. code-block:: html

   <!-- Add feedback widget -->
   <div class="feedback-widget">
     <p>Was this page helpful?</p>
     <button onclick="submitFeedback('yes')">Yes</button>
     <button onclick="submitFeedback('no')">No</button>
   </div>

See Also
--------

* :doc:`contributing` - Contributing guidelines
* :doc:`../api/index` - API reference
* `Sphinx Documentation <https://www.sphinx-doc.org/>`_ - Sphinx user guide
* `reStructuredText Primer <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ - RST syntax guide
