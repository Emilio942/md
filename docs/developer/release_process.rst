Release Process
===============

This document describes the complete release process for ProteinMD, including version management, testing procedures, and distribution.

.. contents:: Contents
   :local:
   :depth: 2

Release Strategy
---------------

Versioning Scheme
~~~~~~~~~~~~~~~~~

ProteinMD follows **Semantic Versioning (SemVer)** with the format ``MAJOR.MINOR.PATCH``:

**MAJOR (X.0.0)**
  - Incompatible API changes
  - Major architectural changes
  - Breaking changes to data formats
  - Significant performance regressions

**MINOR (0.X.0)**
  - New features in backwards-compatible manner
  - New force field implementations
  - New analysis methods
  - Performance improvements
  - Deprecation of existing features

**PATCH (0.0.X)**
  - Backwards-compatible bug fixes
  - Documentation corrections
  - Security patches
  - Minor performance fixes

**Pre-release Versions**
  - Alpha: ``1.2.0-alpha.1`` (early development)
  - Beta: ``1.2.0-beta.1`` (feature complete, testing)
  - Release Candidate: ``1.2.0-rc.1`` (potentially final)

Release Types
~~~~~~~~~~~~

**Regular Releases**
  - **Schedule**: Every 3-4 months
  - **Content**: New features, improvements, bug fixes
  - **Testing**: Full test suite, performance benchmarks

**Patch Releases**
  - **Schedule**: As needed for bug fixes
  - **Content**: Critical bug fixes, security patches
  - **Testing**: Focused testing on changed areas

**Hotfix Releases**
  - **Schedule**: Emergency releases for critical issues
  - **Content**: Minimal changes to fix specific problems
  - **Testing**: Targeted testing and validation

**Long-Term Support (LTS)**
  - **Schedule**: Annual LTS releases
  - **Support**: 2 years of bug fixes and security updates
  - **Compatibility**: API stability guarantees

Release Planning
---------------

Release Schedule
~~~~~~~~~~~~~~~

**Quarterly Release Cycle**

.. code-block:: text

   Quarter 1 (Jan-Mar): v1.1.0
   ├── Month 1: Development
   ├── Month 2: Feature freeze, testing
   └── Month 3: Release preparation, RC
   
   Quarter 2 (Apr-Jun): v1.2.0
   ├── Month 1: Development
   ├── Month 2: Feature freeze, testing  
   └── Month 3: Release preparation, RC
   
   Quarter 3 (Jul-Sep): v1.3.0
   ├── Month 1: Development
   ├── Month 2: Feature freeze, testing
   └── Month 3: Release preparation, RC
   
   Quarter 4 (Oct-Dec): v1.4.0 LTS
   ├── Month 1: Development
   ├── Month 2: Feature freeze, testing
   └── Month 3: Release preparation, RC

**Milestone Planning**

Create GitHub milestones for each release:

.. code-block:: markdown

   ## v1.2.0 Milestone
   
   **Target Date**: June 30, 2025
   **Theme**: Enhanced Sampling Methods
   
   ### Major Features
   - [ ] Umbrella Sampling Implementation (#123)
   - [ ] Replica Exchange Molecular Dynamics (#145)
   - [ ] Steered Molecular Dynamics (#167)
   
   ### Improvements
   - [ ] CUDA Performance Optimization (#134)
   - [ ] Trajectory Compression (#156)
   - [ ] Enhanced Documentation (#178)
   
   ### Bug Fixes
   - [ ] Memory Leak in Trajectory Writer (#142)
   - [ ] Force Field Parameter Loading (#159)

Feature Planning
~~~~~~~~~~~~~~~

**Feature Proposal Process**

1. **Create RFC (Request for Comments)** for major features
2. **Community discussion** in GitHub Discussions
3. **Design review** with maintainers
4. **Implementation planning** and task breakdown
5. **Assignment to milestone**

**Feature Freeze Timeline**

.. code-block:: text

   Week -8: Feature proposals due
   Week -6: Design reviews completed
   Week -4: Implementation deadline
   Week -2: Feature freeze begins
   Week -1: Release candidate
   Week 0: Final release

Release Process Workflow
-----------------------

Pre-Release Phase
~~~~~~~~~~~~~~~~

**1. Feature Freeze (2 weeks before release)**

.. code-block:: bash

   # Create release branch
   git checkout main
   git pull upstream main
   git checkout -b release/v1.2.0
   
   # Update version numbers
   # - setup.py
   # - proteinMD/__init__.py
   # - docs/conf.py
   
   # Commit version update
   git add .
   git commit -m "chore: bump version to 1.2.0"

**2. Release Candidate Testing**

.. code-block:: bash

   # Tag release candidate
   git tag -a v1.2.0-rc.1 -m "Release candidate 1.2.0-rc.1"
   git push upstream v1.2.0-rc.1
   
   # Trigger automated testing
   # - Full test suite on multiple platforms
   # - Performance benchmarks
   # - Integration tests
   # - Documentation build

**3. CHANGELOG Preparation**

Update CHANGELOG.md with release notes:

.. code-block:: markdown

   # Changelog
   
   ## [1.2.0] - 2025-06-30
   
   ### Added
   - Umbrella sampling implementation for enhanced conformational sampling
   - Replica exchange molecular dynamics (REMD) support
   - Steered molecular dynamics for force probe simulations
   - CUDA acceleration for non-bonded force calculations
   - Trajectory compression using LZMA algorithm
   
   ### Changed
   - Improved memory efficiency in trajectory handling
   - Enhanced error messages for force field loading
   - Updated documentation with more examples
   
   ### Fixed
   - Memory leak in TrajectoryWriter when exceptions occur
   - Incorrect bond angle calculations in custom force fields
   - Race condition in parallel tempering simulations
   
   ### Deprecated
   - `old_analysis_function()` - use `new_analysis_function()` instead
   
   ### Breaking Changes
   - Force field parameter format updated to version 2.0
   - Trajectory file format changed for better compression
   
   ### Migration Guide
   #### Force Field Parameters
   ```python
   # Old format
   forcefield = ForceField.from_file("params.dat")
   
   # New format  
   forcefield = ForceField.from_file("params.json")
   ```

Release Testing
~~~~~~~~~~~~~~

**Testing Matrix**

.. code-block:: text

   Operating Systems:
   ├── Ubuntu 20.04 LTS
   ├── Ubuntu 22.04 LTS  
   ├── CentOS 8
   ├── macOS 11 (Intel)
   ├── macOS 12 (Apple Silicon)
   └── Windows 10

   Python Versions:
   ├── Python 3.8
   ├── Python 3.9
   ├── Python 3.10
   └── Python 3.11

   Dependencies:
   ├── NumPy 1.19-1.24
   ├── SciPy 1.7-1.10
   ├── OpenMM 7.7-8.0
   └── CUDA 11.0-12.0

**Automated Testing Pipeline**

.. code-block:: yaml

   # .github/workflows/release-testing.yml
   name: Release Testing
   
   on:
     push:
       tags:
         - 'v*-rc.*'
   
   jobs:
     test-matrix:
       strategy:
         matrix:
           os: [ubuntu-20.04, ubuntu-22.04, macos-11, macos-12, windows-2019]
           python: [3.8, 3.9, '3.10', 3.11]
       
       runs-on: ${{ matrix.os }}
       
       steps:
         - uses: actions/checkout@v3
         
         - name: Set up Python ${{ matrix.python }}
           uses: actions/setup-python@v4
           with:
             python-version: ${{ matrix.python }}
         
         - name: Install dependencies
           run: |
             pip install -e .
             pip install -r requirements-test.txt
         
         - name: Run full test suite
           run: pytest tests/ --tb=short -v
         
         - name: Run performance benchmarks
           run: python benchmarks/run_benchmarks.py
         
         - name: Test installation from source
           run: |
             pip uninstall proteinmd -y
             pip install .
             python -c "import proteinMD; print(proteinMD.__version__)"

**Manual Testing Checklist**

- [ ] **Installation**: Install from source on clean system
- [ ] **Examples**: All example scripts run successfully
- [ ] **Documentation**: Sphinx docs build without errors
- [ ] **Performance**: No significant regressions in benchmarks
- [ ] **Backwards Compatibility**: Existing user code works
- [ ] **New Features**: All new features work as documented

Release Execution
~~~~~~~~~~~~~~~~

**1. Final Release Preparation**

.. code-block:: bash

   # Ensure all tests pass
   pytest tests/ --tb=short
   
   # Update documentation
   cd docs/
   make clean
   make html
   
   # Final version check
   python -c "import proteinMD; print(proteinMD.__version__)"
   
   # Create final commit
   git add .
   git commit -m "chore: prepare v1.2.0 release"

**2. Release Tag Creation**

.. code-block:: bash

   # Create annotated release tag
   git tag -a v1.2.0 -m "Release v1.2.0: Enhanced Sampling Methods
   
   Major Features:
   - Umbrella Sampling implementation
   - Replica Exchange Molecular Dynamics
   - Steered Molecular Dynamics
   - CUDA performance optimizations
   
   Bug Fixes:
   - Memory leak in trajectory writer
   - Force field parameter loading issues
   
   Breaking Changes:
   - Updated force field parameter format
   - Changed trajectory file format
   
   See CHANGELOG.md for complete details."
   
   # Push tag to trigger release automation
   git push upstream v1.2.0

**3. Automated Release Pipeline**

.. code-block:: yaml

   # .github/workflows/release.yml
   name: Release
   
   on:
     push:
       tags:
         - 'v*'
   
   jobs:
     release:
       runs-on: ubuntu-latest
       
       steps:
         - uses: actions/checkout@v3
         
         - name: Set up Python
           uses: actions/setup-python@v4
           with:
             python-version: 3.9
         
         - name: Build distribution packages
           run: |
             pip install build
             python -m build
         
         - name: Upload to PyPI
           uses: pypa/gh-action-pypi-publish@v1
           with:
             password: ${{ secrets.PYPI_API_TOKEN }}
         
         - name: Create GitHub Release
           uses: actions/create-release@v1
           with:
             tag_name: ${{ github.ref }}
             release_name: Release ${{ github.ref }}
             body_path: RELEASE_NOTES.md
             draft: false
             prerelease: false

Distribution
-----------

Package Building
~~~~~~~~~~~~~~~

**Source Distribution**

.. code-block:: bash

   # Build source distribution
   python -m build --sdist
   
   # Verify contents
   tar -tzf dist/proteinMD-1.2.0.tar.gz

**Binary Wheels**

.. code-block:: bash

   # Build wheels for multiple platforms
   pip install cibuildwheel
   cibuildwheel --platform linux
   cibuildwheel --platform macos
   cibuildwheel --platform windows

**Conda Package**

.. code-block:: yaml

   # conda-recipe/meta.yaml
   package:
     name: proteinmd
     version: {{ environ.get('GIT_DESCRIBE_TAG', '')[1:] }}
   
   source:
     git_url: https://github.com/proteinmd/proteinmd.git
     git_tag: v{{ environ.get('GIT_DESCRIBE_TAG', '')[1:] }}
   
   build:
     number: 0
     script: pip install . -vv
   
   requirements:
     build:
       - python
       - pip
       - setuptools
     run:
       - python >=3.8
       - numpy >=1.19
       - scipy >=1.7
       - openmm >=7.7

PyPI Release
~~~~~~~~~~~

**Package Upload**

.. code-block:: bash

   # Upload to TestPyPI first
   twine upload --repository testpypi dist/*
   
   # Test installation from TestPyPI
   pip install --index-url https://test.pypi.org/simple/ proteinMD
   
   # Upload to PyPI
   twine upload dist/*

**Release Verification**

.. code-block:: bash

   # Verify PyPI release
   pip install proteinMD==1.2.0
   python -c "import proteinMD; print(proteinMD.__version__)"
   
   # Test key functionality
   python -c "
   import proteinMD
   from proteinMD.core import MDSystem
   system = MDSystem('test')
   print('Release verification successful')
   "

Conda-Forge Release
~~~~~~~~~~~~~~~~~

**Update Feedstock**

.. code-block:: bash

   # Fork conda-forge/proteinmd-feedstock
   git clone https://github.com/yourusername/proteinmd-feedstock.git
   cd proteinmd-feedstock
   
   # Update recipe/meta.yaml
   # - Update version number
   # - Update source URL and SHA256
   # - Update dependencies if needed
   
   # Create PR to conda-forge

Documentation Release
~~~~~~~~~~~~~~~~~~~

**Update Documentation Site**

.. code-block:: bash

   # Build documentation
   cd docs/
   make clean
   make html
   
   # Deploy to GitHub Pages
   # (automated via GitHub Actions)

**Version-Specific Documentation**

.. code-block:: bash

   # Create versioned documentation
   sphinx-multiversion docs/ docs/_build/html/
   
   # Deploy versioned docs
   # - Latest stable: /stable/
   # - Development: /latest/
   # - Specific versions: /v1.2.0/

Post-Release Activities
---------------------

Communication
~~~~~~~~~~~~~

**Release Announcement**

.. code-block:: markdown

   # ProteinMD v1.2.0 Released: Enhanced Sampling Methods
   
   We're excited to announce the release of ProteinMD v1.2.0, featuring
   powerful new enhanced sampling methods for molecular dynamics simulations.
   
   ## 🎉 What's New
   
   ### Enhanced Sampling Methods
   - **Umbrella Sampling**: Explore rare events and calculate free energy profiles
   - **Replica Exchange MD**: Enhanced conformational sampling with temperature exchange
   - **Steered MD**: Apply external forces to study protein unfolding and binding
   
   ### Performance Improvements
   - **CUDA Acceleration**: Up to 10x faster force calculations on GPU
   - **Memory Optimization**: 30% reduction in memory usage for large systems
   - **Trajectory Compression**: 50% smaller trajectory files with LZMA
   
   ### User Experience
   - **Enhanced Documentation**: More examples and tutorials
   - **Better Error Messages**: Clearer guidance when things go wrong
   - **API Improvements**: More intuitive interfaces
   
   ## 📋 Breaking Changes
   
   This release includes some breaking changes. See our [migration guide]
   for details on updating your code.
   
   ## 🚀 Get Started
   
   ```bash
   pip install --upgrade proteinMD
   ```
   
   Check out our [getting started guide] and [new examples].
   
   ## 🙏 Acknowledgments
   
   Thanks to all contributors who made this release possible!

**Communication Channels**

- **GitHub Release**: Detailed release notes
- **Project Website**: Release announcement blog post
- **Mailing Lists**: Scientific computing communities
- **Social Media**: Twitter, LinkedIn announcements
- **Conferences**: Present at relevant scientific meetings

User Migration Support
~~~~~~~~~~~~~~~~~~~~~

**Migration Documentation**

.. code-block:: rst

   Migration Guide: v1.1.x to v1.2.0
   ==================================
   
   Force Field Parameter Format
   ---------------------------
   
   The force field parameter format has been updated for better performance
   and extensibility.
   
   **Before (v1.1.x):**
   
   .. code-block:: python
   
       forcefield = ForceField.from_file("charmm36.dat")
   
   **After (v1.2.0):**
   
   .. code-block:: python
   
       forcefield = ForceField.from_file("charmm36.json")
   
   **Migration Script:**
   
   We provide a migration script to convert old parameter files:
   
   .. code-block:: bash
   
       python -m proteinMD.tools.migrate_forcefield charmm36.dat charmm36.json

**Support Channels**

- **GitHub Issues**: Technical problems and questions
- **Discussions**: General questions and community support
- **Documentation**: Updated examples and guides
- **Chat/Forum**: Real-time community support

Monitoring and Feedback
~~~~~~~~~~~~~~~~~~~~~~

**Release Metrics**

Track release success metrics:

.. code-block:: text

   Release v1.2.0 Metrics (First 30 Days)
   ======================================
   
   Downloads:
   - PyPI: 15,234 downloads
   - Conda: 8,567 downloads
   - GitHub: 2,341 clones
   
   Issues:
   - Bug reports: 3 (all fixed in v1.2.1)
   - Feature requests: 12
   - Questions: 28
   
   Community:
   - New contributors: 5
   - Documentation improvements: 8 PRs
   - Third-party packages using v1.2.0: 3

**Feedback Collection**

- **User surveys**: Collect feedback on new features
- **Issue tracking**: Monitor bug reports and feature requests
- **Usage analytics**: Track feature adoption (if users opt-in)
- **Performance monitoring**: Watch for regression reports

Hotfix Process
-------------

Emergency Releases
~~~~~~~~~~~~~~~~~

For critical issues requiring immediate fixes:

**1. Hotfix Branch Creation**

.. code-block:: bash

   # Create hotfix branch from latest release tag
   git checkout v1.2.0
   git checkout -b hotfix/v1.2.1

**2. Rapid Fix and Testing**

.. code-block:: bash

   # Make minimal fix
   git add .
   git commit -m "fix: resolve critical simulation crash
   
   Fixed null pointer dereference in force calculation
   when using custom force fields with missing parameters.
   
   Fixes #789"
   
   # Test fix thoroughly
   pytest tests/test_forcefield.py -v
   
   # Run specific regression test
   python tests/test_issue_789.py

**3. Emergency Release**

.. code-block:: bash

   # Update version to 1.2.1
   # Create release tag
   git tag -a v1.2.1 -m "Hotfix v1.2.1: Critical simulation crash fix"
   
   # Fast-track release process
   git push upstream v1.2.1

**4. Communication**

.. code-block:: markdown

   # URGENT: ProteinMD v1.2.1 Hotfix Released
   
   We've released an urgent hotfix (v1.2.1) to address a critical issue
   that could cause simulation crashes when using custom force fields.
   
   **What's Fixed:**
   - Critical crash in force calculation with custom force fields
   
   **Who Should Update:**
   - All users using custom force field parameters
   - Users experiencing unexpected simulation crashes
   
   **How to Update:**
   ```bash
   pip install --upgrade proteinMD
   ```
   
   **Need Help?**
   If you encountered this issue, please see our [troubleshooting guide].

Quality Assurance
----------------

Release Validation
~~~~~~~~~~~~~~~~~

**Automated Validation Suite**

.. code-block:: python

   # scripts/validate_release.py
   """Comprehensive release validation suite."""
   
   import subprocess
   import tempfile
   import os
   
   def test_installation():
       """Test clean installation from PyPI."""
       with tempfile.TemporaryDirectory() as tmpdir:
           # Create fresh virtual environment
           subprocess.run([
               "python", "-m", "venv", f"{tmpdir}/test_env"
           ])
           
           # Install package
           subprocess.run([
               f"{tmpdir}/test_env/bin/pip", "install", "proteinMD"
           ])
           
           # Test import
           result = subprocess.run([
               f"{tmpdir}/test_env/bin/python", 
               "-c", "import proteinMD; print(proteinMD.__version__)"
           ], capture_output=True, text=True)
           
           assert result.returncode == 0
           print(f"Installation test passed: {result.stdout.strip()}")
   
   def test_examples():
       """Test all example scripts."""
       examples_dir = "examples/"
       for example in os.listdir(examples_dir):
           if example.endswith('.py'):
               result = subprocess.run([
                   "python", f"{examples_dir}/{example}"
               ], capture_output=True)
               assert result.returncode == 0
               print(f"Example {example} passed")
   
   def test_backwards_compatibility():
       """Test backwards compatibility with previous version."""
       # Test that v1.1.x code still works
       test_scripts = [
           "tests/compatibility/test_v1_1_api.py",
           "tests/compatibility/test_v1_1_data_formats.py"
       ]
       
       for script in test_scripts:
           result = subprocess.run(["python", script])
           assert result.returncode == 0
           print(f"Compatibility test {script} passed")

**Performance Benchmarks**

.. code-block:: python

   # benchmarks/release_benchmarks.py
   """Performance benchmarks for release validation."""
   
   import time
   import numpy as np
   from proteinMD.core import MDSystem
   from proteinMD.examples import create_test_system
   
   def benchmark_simulation_performance():
       """Benchmark core simulation performance."""
       system = create_test_system(n_atoms=10000)
       
       start_time = time.time()
       system.run_simulation(steps=1000)
       end_time = time.time()
       
       performance = 1000 / (end_time - start_time)  # steps/second
       
       # Compare with baseline
       baseline_performance = 850  # steps/second from v1.1.0
       assert performance >= baseline_performance * 0.95  # Allow 5% regression
       
       print(f"Simulation performance: {performance:.1f} steps/s")
       return performance

Release Documentation
~~~~~~~~~~~~~~~~~~~~

**Release Notes Template**

.. code-block:: markdown

   # ProteinMD v1.2.0 Release Notes
   
   **Release Date**: June 30, 2025
   **Release Type**: Minor Release
   **Python Support**: 3.8, 3.9, 3.10, 3.11
   
   ## Overview
   
   ProteinMD v1.2.0 introduces powerful enhanced sampling methods...
   
   ## Installation
   
   ```bash
   pip install proteinMD==1.2.0
   conda install -c conda-forge proteinMD=1.2.0
   ```
   
   ## New Features
   
   ### Enhanced Sampling Methods
   
   #### Umbrella Sampling
   ```python
   from proteinMD.sampling import UmbrellaSampling
   
   sampler = UmbrellaSampling(
       reaction_coordinate=distance_coordinate,
       windows=np.linspace(2.0, 10.0, 20),
       force_constant=1000.0  # kJ/(mol·nm²)
   )
   ```
   
   ## Breaking Changes
   
   ### Force Field Parameter Format
   
   The force field parameter format has been updated...
   
   ## Migration Guide
   
   ### Updating Force Field Files
   
   Use the migration script to convert old parameter files...
   
   ## Performance Improvements
   
   - CUDA acceleration: 10x speedup for large systems
   - Memory optimization: 30% reduction in memory usage
   - Trajectory compression: 50% smaller files
   
   ## Bug Fixes
   
   - Fixed memory leak in trajectory writer (#456)
   - Corrected force field parameter loading (#789)
   - Resolved race condition in parallel simulations (#123)
   
   ## Acknowledgments
   
   Thanks to all contributors...

Best Practices
--------------

Release Management
~~~~~~~~~~~~~~~~~

**Release Readiness Checklist**

- [ ] **All milestone issues closed** or moved to next release
- [ ] **Test suite passes** on all supported platforms
- [ ] **Documentation updated** and builds successfully
- [ ] **Performance benchmarks** show no significant regressions
- [ ] **Breaking changes documented** with migration guide
- [ ] **Release notes prepared** with clear descriptions
- [ ] **Security review completed** for security-sensitive changes

**Quality Gates**

.. code-block:: text

   Release Quality Gates
   ====================
   
   ✅ Automated Tests Pass (100%)
   ✅ Code Coverage >= 85%
   ✅ Performance >= 95% of baseline
   ✅ Documentation Builds Successfully
   ✅ Security Scan Clean
   ✅ License Compliance Check
   ✅ Backwards Compatibility Verified
   ✅ Manual Testing Complete

**Release Coordination**

- **Release manager**: Single person responsible for coordination
- **Cross-team communication**: Involve all stakeholders
- **Timeline management**: Buffer time for unexpected issues
- **Risk assessment**: Identify and mitigate release risks

Summary
-------

The ProteinMD release process ensures:

**Quality Assurance**
- Comprehensive testing across platforms and Python versions
- Performance validation and regression testing
- Documentation accuracy and completeness

**User Experience**
- Clear communication about changes and improvements
- Migration guides for breaking changes
- Responsive support during release adoption

**Project Health**
- Regular, predictable release schedule
- Community involvement and feedback incorporation
- Long-term support for stable releases

**Technical Excellence**
- Automated release pipeline with quality gates
- Proper version management and compatibility tracking
- Efficient hotfix process for critical issues

Following this release process maintains ProteinMD's reputation as a reliable, high-quality scientific software package while enabling continuous improvement and innovation.
