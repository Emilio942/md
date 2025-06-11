Contributing to ProteinMD
========================

Thank you for your interest in contributing to ProteinMD! This guide will help you get started.

.. contents:: Contributing Topics
   :local:
   :depth: 2

Getting Started
--------------

Development Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Fork and Clone Repository**

.. code-block:: bash

   # Fork repository on GitHub
   # Then clone your fork
   git clone https://github.com/yourusername/proteinmd.git
   cd proteinmd
   
   # Add upstream remote
   git remote add upstream https://github.com/proteinmd/proteinmd.git

**2. Create Development Environment**

.. code-block:: bash

   # Create conda environment
   conda create -n proteinmd-dev python=3.9
   conda activate proteinmd-dev
   
   # Install dependencies
   conda install numpy scipy matplotlib pytest
   conda install -c conda-forge openmm mdanalysis
   
   # Install in development mode
   pip install -e .
   
   # Install development dependencies
   pip install -r requirements-dev.txt

**3. Verify Installation**

.. code-block:: bash

   # Run tests to verify installation
   pytest tests/ -v
   
   # Check import
   python -c "import proteinmd; print(proteinmd.__version__)"

Development Workflow
~~~~~~~~~~~~~~~~~~~

**Branch Strategy**

.. code-block:: bash

   # Create feature branch
   git checkout -b feature/new-analysis-method
   
   # Make changes and commit
   git add .
   git commit -m "Add new analysis method for protein flexibility"
   
   # Push to your fork
   git push origin feature/new-analysis-method
   
   # Create pull request on GitHub

**Keeping Fork Updated**

.. code-block:: bash

   # Fetch upstream changes
   git fetch upstream
   
   # Update main branch
   git checkout main
   git merge upstream/main
   
   # Update your feature branch
   git checkout feature/new-analysis-method
   git rebase main

Code Guidelines
--------------

Coding Standards
~~~~~~~~~~~~~~~

**Python Style Guide**

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black formatter default)
- Use type hints for all public functions
- Docstrings in NumPy format
- Import organization: standard library, third-party, local imports

**Example Code Style**

.. code-block:: python

   from typing import List, Optional, Union
   import numpy as np
   from proteinmd.core.base import BaseClass
   
   
   class AnalysisMethod(BaseClass):
       """
       Base class for analysis methods.
       
       Parameters
       ----------
       name : str
           Name of the analysis method.
       parameters : dict, optional
           Analysis parameters.
       
       Attributes
       ----------
       results : dict
           Analysis results storage.
       
       Examples
       --------
       >>> method = AnalysisMethod("rmsd")
       >>> results = method.calculate(trajectory)
       >>> print(results["mean"])
       """
       
       def __init__(
           self,
           name: str,
           parameters: Optional[dict] = None
       ) -> None:
           super().__init__()
           self.name = name
           self.parameters = parameters or {}
           self.results: dict = {}
       
       def calculate(
           self,
           trajectory: "Trajectory",
           **kwargs
       ) -> dict:
           """
           Calculate analysis for trajectory.
           
           Parameters
           ----------
           trajectory : Trajectory
               Input trajectory for analysis.
           **kwargs
               Additional parameters.
           
           Returns
           -------
           dict
               Analysis results.
           
           Raises
           ------
           ValueError
               If trajectory is empty.
           """
           if len(trajectory) == 0:
               raise ValueError("Trajectory cannot be empty")
           
           # Implementation here
           self.results = self._perform_calculation(trajectory, **kwargs)
           return self.results
       
       def _perform_calculation(
           self,
           trajectory: "Trajectory",
           **kwargs
       ) -> dict:
           """Perform the actual calculation (to be implemented by subclasses)."""
           raise NotImplementedError("Subclasses must implement this method")

**Formatting Tools**

.. code-block:: bash

   # Format code with Black
   black proteinmd/ tests/
   
   # Sort imports with isort
   isort proteinmd/ tests/
   
   # Check with flake8
   flake8 proteinmd/ tests/
   
   # Type checking with mypy
   mypy proteinmd/

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~

**Docstring Format**

Use NumPy-style docstrings:

.. code-block:: python

   def calculate_rmsd(
       coords1: np.ndarray,
       coords2: np.ndarray,
       align: bool = True
   ) -> float:
       """
       Calculate root mean square deviation between two coordinate sets.
       
       This function computes the RMSD between two sets of coordinates,
       optionally after optimal alignment.
       
       Parameters
       ----------
       coords1 : np.ndarray, shape (n_atoms, 3)
           First coordinate set.
       coords2 : np.ndarray, shape (n_atoms, 3)
           Second coordinate set.
       align : bool, default=True
           Whether to optimally align structures before RMSD calculation.
       
       Returns
       -------
       float
           RMSD value in nanometers.
       
       Raises
       ------
       ValueError
           If coordinate arrays have different shapes.
       
       Examples
       --------
       >>> coords1 = np.random.rand(100, 3)
       >>> coords2 = np.random.rand(100, 3)
       >>> rmsd = calculate_rmsd(coords1, coords2)
       >>> print(f"RMSD: {rmsd:.3f} nm")
       
       Notes
       -----
       The RMSD is calculated as:
       
       .. math::
           RMSD = \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} |r_i - r_i'|^2}
       
       where N is the number of atoms.
       
       References
       ----------
       .. [1] Kabsch, W. (1976). A solution for the best rotation to relate
              two sets of vectors. Acta Crystallographica, 32(5), 922-923.
       """

Testing Guidelines
-----------------

Test Structure
~~~~~~~~~~~~~

**Test Organization**

.. code-block:: text

   tests/
   ├── unit/                 # Unit tests
   │   ├── test_core/
   │   ├── test_structure/
   │   ├── test_forcefield/
   │   └── test_analysis/
   ├── integration/          # Integration tests
   │   ├── test_workflows/
   │   └── test_pipelines/
   ├── regression/           # Regression tests
   │   └── test_numerical/
   ├── performance/          # Performance tests
   │   └── test_benchmarks/
   └── data/                 # Test data
       ├── structures/
       ├── trajectories/
       └── parameters/

**Writing Tests**

.. code-block:: python

   import pytest
   import numpy as np
   from proteinmd.testing import TestCase, create_test_system
   from proteinmd.analysis import RMSD
   
   
   class TestRMSD(TestCase):
       """Test RMSD calculation functionality."""
       
       def setUp(self):
           """Set up test fixtures."""
           self.coords1 = np.random.rand(100, 3)
           self.coords2 = self.coords1 + 0.1 * np.random.rand(100, 3)
           self.rmsd_calc = RMSD()
       
       def test_rmsd_calculation_basic(self):
           """Test basic RMSD calculation."""
           rmsd = self.rmsd_calc.calculate(self.coords1, self.coords2)
           
           # RMSD should be positive
           self.assertGreater(rmsd, 0.0)
           
           # RMSD of identical structures should be zero
           rmsd_identical = self.rmsd_calc.calculate(self.coords1, self.coords1)
           self.assertAlmostEqual(rmsd_identical, 0.0, places=10)
       
       def test_rmsd_with_alignment(self):
           """Test RMSD calculation with alignment."""
           # Rotate second structure
           rotation_matrix = self._create_rotation_matrix(np.pi/4, axis='z')
           coords2_rotated = self.coords2 @ rotation_matrix.T
           
           # RMSD without alignment should be large
           rmsd_no_align = self.rmsd_calc.calculate(
               self.coords1, coords2_rotated, align=False
           )
           
           # RMSD with alignment should be smaller
           rmsd_aligned = self.rmsd_calc.calculate(
               self.coords1, coords2_rotated, align=True
           )
           
           self.assertLess(rmsd_aligned, rmsd_no_align)
       
       def test_rmsd_input_validation(self):
           """Test input validation."""
           # Different shapes should raise ValueError
           coords_wrong_shape = np.random.rand(50, 3)
           
           with self.assertRaises(ValueError):
               self.rmsd_calc.calculate(self.coords1, coords_wrong_shape)
       
       def test_rmsd_edge_cases(self):
           """Test edge cases."""
           # Single atom
           single_atom1 = np.array([[0.0, 0.0, 0.0]])
           single_atom2 = np.array([[1.0, 0.0, 0.0]])
           
           rmsd_single = self.rmsd_calc.calculate(single_atom1, single_atom2)
           self.assertAlmostEqual(rmsd_single, 1.0, places=10)
       
       @pytest.mark.parametrize("n_atoms", [10, 100, 1000])
       def test_rmsd_performance(self, n_atoms):
           """Test RMSD calculation performance for different sizes."""
           coords1 = np.random.rand(n_atoms, 3)
           coords2 = np.random.rand(n_atoms, 3)
           
           import time
           start_time = time.time()
           rmsd = self.rmsd_calc.calculate(coords1, coords2)
           elapsed_time = time.time() - start_time
           
           # Should complete in reasonable time
           self.assertLess(elapsed_time, 1.0)  # 1 second max
           self.assertIsInstance(rmsd, float)

**Test Fixtures and Data**

.. code-block:: python

   # conftest.py - pytest fixtures
   import pytest
   import numpy as np
   from proteinmd.structure import ProteinStructure
   
   
   @pytest.fixture
   def small_protein():
       """Fixture providing small test protein."""
       return ProteinStructure.from_pdb("tests/data/structures/1ubq.pdb")
   
   
   @pytest.fixture
   def water_box():
       """Fixture providing water box system."""
       from proteinmd.environment import WaterModel
       
       water_model = WaterModel("TIP3P")
       return water_model.create_box(size=2.0, n_molecules=1000)
   
   
   @pytest.fixture
   def test_trajectory():
       """Fixture providing test trajectory."""
       # Generate synthetic trajectory
       n_frames = 100
       n_atoms = 50
       
       coordinates = []
       for i in range(n_frames):
           # Add some realistic motion
           coords = np.random.rand(n_atoms, 3) + 0.01 * i
           coordinates.append(coords)
       
       return coordinates

**Running Tests**

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run specific test file
   pytest tests/unit/test_analysis/test_rmsd.py
   
   # Run with coverage
   pytest --cov=proteinmd --cov-report=html
   
   # Run performance tests
   pytest tests/performance/ -m performance
   
   # Run only fast tests
   pytest -m "not slow"

Pull Request Process
-------------------

Before Submitting
~~~~~~~~~~~~~~~~

**Pre-submission Checklist**

.. code-block:: bash

   # 1. Update from upstream
   git fetch upstream
   git rebase upstream/main
   
   # 2. Run tests
   pytest tests/ -x  # Stop on first failure
   
   # 3. Check code quality
   black --check proteinmd/ tests/
   flake8 proteinmd/ tests/
   mypy proteinmd/
   
   # 4. Update documentation
   cd docs/
   make html
   
   # 5. Add changelog entry
   # Edit CHANGELOG.md

**Commit Message Format**

Use conventional commit format:

.. code-block:: text

   type(scope): brief description
   
   Longer description explaining the change in more detail.
   
   - Key changes
   - Breaking changes (if any)
   
   Fixes #123
   Closes #456

Example:

.. code-block:: text

   feat(analysis): add dynamic cross-correlation analysis
   
   Implement dynamic cross-correlation matrix calculation for analyzing
   correlated motions between residues in protein trajectories.
   
   - Add DynamicCrossCorrelation class with calculate() method
   - Support for different distance metrics
   - Include visualization utilities
   - Add comprehensive tests and documentation
   
   Closes #78

Pull Request Template
~~~~~~~~~~~~~~~~~~~~

**PR Description Template**

.. code-block:: markdown

   ## Description
   Brief description of the changes in this PR.
   
   ## Type of Change
   - [ ] Bug fix (non-breaking change which fixes an issue)
   - [ ] New feature (non-breaking change which adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
   - [ ] Documentation update
   
   ## Changes Made
   - List key changes
   - Include any breaking changes
   
   ## Testing
   - [ ] Added tests for new functionality
   - [ ] All tests pass
   - [ ] Tested on different platforms (if applicable)
   
   ## Documentation
   - [ ] Updated docstrings
   - [ ] Updated user documentation
   - [ ] Updated API documentation
   
   ## Performance Impact
   Describe any performance implications of the changes.
   
   ## Related Issues
   Fixes #issue_number
   
   ## Screenshots (if applicable)
   Add screenshots to help explain your changes.

Review Process
~~~~~~~~~~~~~

**Code Review Guidelines**

1. **Automated Checks**: All CI checks must pass
2. **Code Quality**: Follow coding standards and best practices
3. **Tests**: Adequate test coverage for new functionality
4. **Documentation**: Clear documentation for user-facing changes
5. **Performance**: No significant performance regressions

**Review Criteria**

- **Correctness**: Does the code do what it's supposed to do?
- **Clarity**: Is the code easy to understand and maintain?
- **Completeness**: Are edge cases handled appropriately?
- **Consistency**: Does it follow project conventions?
- **Performance**: Are there any performance concerns?

Types of Contributions
---------------------

Bug Reports
~~~~~~~~~~

**Bug Report Template**

.. code-block:: markdown

   ## Bug Description
   A clear and concise description of what the bug is.
   
   ## To Reproduce
   Steps to reproduce the behavior:
   1. Go to '...'
   2. Click on '....'
   3. Scroll down to '....'
   4. See error
   
   ## Expected Behavior
   A clear and concise description of what you expected to happen.
   
   ## Actual Behavior
   What actually happened.
   
   ## Environment
   - OS: [e.g. Ubuntu 20.04]
   - Python version: [e.g. 3.9.7]
   - ProteinMD version: [e.g. 1.0.0]
   - Backend: [e.g. OpenMM 7.7.0]
   
   ## Error Messages
   ```
   Full error message and stack trace
   ```
   
   ## Additional Context
   Add any other context about the problem here.

Feature Requests
~~~~~~~~~~~~~~~

**Feature Request Template**

.. code-block:: markdown

   ## Feature Summary
   Brief description of the feature you'd like to see.
   
   ## Motivation
   Why would this feature be useful? What problem does it solve?
   
   ## Proposed Implementation
   How do you envision this feature working?
   
   ## Alternatives Considered
   What alternatives have you considered?
   
   ## Additional Context
   Any other context or examples that would help.

Documentation Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

**Documentation Contributions**

- Fix typos and grammatical errors
- Improve clarity of explanations
- Add examples and tutorials
- Update API documentation
- Translate documentation (future)

.. code-block:: bash

   # Build documentation locally
   cd docs/
   pip install -r requirements.txt
   make html
   
   # View documentation
   open _build/html/index.html

New Features
~~~~~~~~~~~

**Feature Development Process**

1. **Discuss**: Open an issue to discuss the feature
2. **Design**: Create design document for complex features
3. **Implement**: Develop the feature following guidelines
4. **Test**: Add comprehensive tests
5. **Document**: Update documentation
6. **Review**: Submit PR for review

**Example: Adding New Analysis Method**

.. code-block:: python

   # 1. Create base structure
   class NewAnalysisMethod(AnalysisMethod):
       """New analysis method for [purpose]."""
       
       def __init__(self, parameter1=default_value):
           super().__init__(name="new_analysis")
           self.parameter1 = parameter1
       
       def calculate(self, trajectory, **kwargs):
           """Calculate analysis."""
           # Implementation
           pass
   
   # 2. Add to module __init__.py
   from .new_analysis import NewAnalysisMethod
   __all__.append("NewAnalysisMethod")
   
   # 3. Add tests
   class TestNewAnalysisMethod(TestCase):
       # Test implementation
       pass
   
   # 4. Add documentation
   # Update docs/api/analysis.rst

Community Guidelines
-------------------

Code of Conduct
~~~~~~~~~~~~~~

We are committed to providing a welcoming and inclusive environment for all contributors.

**Our Standards**

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable Behavior**

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing private information without permission
- Other conduct which could reasonably be considered inappropriate

Communication Channels
~~~~~~~~~~~~~~~~~~~~~

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code contributions
- **Documentation**: User guides and API reference

Recognition
~~~~~~~~~~

**Contributors**

All contributors are recognized in:

- AUTHORS file
- Release notes
- Documentation credits

**Types of Recognition**

- Code contributions
- Documentation improvements
- Bug reports and testing
- Community support and mentoring

Getting Help
-----------

**For New Contributors**

- Read this contributing guide
- Look for "good first issue" labels
- Ask questions in GitHub Discussions
- Join community meetings (if available)

**For Development Issues**

- Check existing issues and discussions
- Provide minimal reproducible examples
- Include environment details
- Be patient and respectful

See Also
--------

* :doc:`testing` - Testing framework details
* :doc:`documentation` - Documentation guidelines
* :doc:`architecture` - Software architecture
* :doc:`../user_guide/installation` - Installation guide
