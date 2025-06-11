Pull Request Guide
=================

This guide covers the complete process for submitting, reviewing, and merging pull requests in ProteinMD.

.. contents:: Contents
   :local:
   :depth: 2

Before You Start
---------------

Prerequisites
~~~~~~~~~~~~

Before submitting a pull request, ensure you have:

- [ ] **Development environment set up** following the :doc:`contributing` guide
- [ ] **Issue created or assigned** for the feature/bug you're working on
- [ ] **Understanding of the codebase** by reading :doc:`architecture` documentation
- [ ] **Familiarity with coding standards** outlined in :doc:`coding_standards`

Choosing What to Work On
~~~~~~~~~~~~~~~~~~~~~~~

**Good First Contributions**

Look for issues labeled:
- ``good-first-issue``: Well-defined tasks for new contributors
- ``documentation``: Documentation improvements
- ``bug``: Small, well-defined bug fixes
- ``enhancement``: Minor feature improvements

**Larger Contributions**

For significant features:
1. **Create an issue** first to discuss the approach
2. **Get feedback** from maintainers before starting implementation
3. **Consider breaking down** large features into smaller, reviewable pieces

Pre-Development Checklist
-------------------------

Planning Phase
~~~~~~~~~~~~~

- [ ] **Issue exists** and is assigned to you
- [ ] **Scope is clear** - what exactly will be implemented/fixed
- [ ] **Acceptance criteria** defined in the issue
- [ ] **Breaking changes** identified and documented
- [ ] **Tests approach** planned
- [ ] **Documentation updates** identified

Environment Setup
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Ensure your fork is up to date
   git checkout main
   git pull upstream main
   git push origin main
   
   # Create feature branch
   git checkout -b feature/your-feature-name
   
   # Verify environment
   pytest tests/ --tb=short
   black --check proteinMD/
   flake8 proteinMD/

Development Workflow
-------------------

Branch Naming
~~~~~~~~~~~~

Use descriptive branch names following this pattern:

.. code-block:: text

   <type>/<short-description>

**Types:**
- ``feature/``: New features
- ``bugfix/``: Bug fixes
- ``docs/``: Documentation updates
- ``refactor/``: Code refactoring
- ``test/``: Test improvements
- ``chore/``: Maintenance tasks

**Examples:**

.. code-block:: text

   feature/velocity-verlet-integrator
   bugfix/bond-energy-calculation-error
   docs/api-documentation-improvements
   refactor/forcefield-module-structure
   test/integration-test-coverage
   chore/update-dependencies

Commit Guidelines
~~~~~~~~~~~~~~~~

**Commit Message Format**

Use conventional commits format:

.. code-block:: text

   <type>[optional scope]: <description>
   
   [optional body]
   
   [optional footer(s)]

**Examples:**

.. code-block:: text

   feat(core): implement velocity Verlet integrator
   
   Add velocity Verlet integration algorithm with improved energy
   conservation properties for molecular dynamics simulations.
   
   - Implements velocity Verlet with configurable time step
   - Includes energy conservation tests
   - Updates documentation with usage examples
   
   Closes #123

.. code-block:: text

   fix(forcefield): correct harmonic bond energy calculation
   
   Fixed sign error in bond energy calculation that caused incorrect
   energies for stretched bonds.
   
   Breaking change: Energy values will differ for systems with
   significantly stretched bonds.
   
   Fixes #456

**Commit Best Practices:**

- **One logical change per commit**
- **Clear, descriptive commit messages**
- **Atomic commits** - each commit should build and pass tests
- **Logical history** - use ``git rebase`` to clean up history before PR

Development Process
~~~~~~~~~~~~~~~~~~

**1. Write Tests First (TDD)**

.. code-block:: python

   def test_velocity_verlet_integration():
       """Test velocity Verlet integrator for harmonic oscillator."""
       # Set up harmonic oscillator system
       system = create_harmonic_oscillator()
       integrator = VelocityVerletIntegrator(time_step=0.001)
       
       # Run integration
       initial_energy = system.total_energy()
       for step in range(1000):
           integrator.step(system)
       
       # Energy should be conserved
       final_energy = system.total_energy()
       assert abs(final_energy - initial_energy) < 1e-6

**2. Implement Feature**

.. code-block:: python

   class VelocityVerletIntegrator:
       """Velocity Verlet integration algorithm."""
       
       def __init__(self, time_step: float = 0.002):
           """Initialize integrator.
           
           Args:
               time_step: Integration time step in ps
           """
           self.time_step = time_step
       
       def step(self, system: MDSystem) -> None:
           """Perform one integration step."""
           # Implementation here
           pass

**3. Update Documentation**

.. code-block:: python

   def velocity_verlet_step(positions, velocities, forces, masses, dt):
       """Perform velocity Verlet integration step.
       
       The velocity Verlet algorithm is a symplectic integrator that
       provides better energy conservation than standard Verlet.
       
       Args:
           positions: Current positions, shape (n_atoms, 3)
           velocities: Current velocities, shape (n_atoms, 3)
           forces: Current forces, shape (n_atoms, 3)
           masses: Atomic masses, shape (n_atoms,)
           dt: Time step in ps
       
       Returns:
           Tuple of (new_positions, new_velocities)
       
       Example:
           >>> pos, vel = velocity_verlet_step(pos, vel, forces, masses, 0.002)
       """
       pass

**4. Run Tests Locally**

.. code-block:: bash

   # Run all tests
   pytest tests/ -v
   
   # Run specific test file
   pytest tests/test_integrators.py -v
   
   # Run with coverage
   pytest --cov=proteinMD --cov-report=html
   
   # Check formatting and linting
   black --check proteinMD/
   flake8 proteinMD/
   mypy proteinMD/

Code Quality Checklist
----------------------

Before Submitting PR
~~~~~~~~~~~~~~~~~~~

**Code Quality**

- [ ] **Code follows style guidelines** (Black formatted, flake8 clean)
- [ ] **Type hints added** for all public functions
- [ ] **Docstrings written** following Google style
- [ ] **Comments added** for complex logic
- [ ] **No debugging code** or commented-out sections
- [ ] **Error handling** appropriate and tested

**Testing**

- [ ] **Unit tests written** for new functionality
- [ ] **Integration tests** for workflow changes
- [ ] **Edge cases tested** (empty inputs, boundary conditions)
- [ ] **Error cases tested** (invalid inputs, exceptions)
- [ ] **All tests pass** locally
- [ ] **Test coverage** maintained or improved

**Documentation**

- [ ] **API documentation** updated for public functions
- [ ] **User guide** updated if user-facing changes
- [ ] **CHANGELOG** entry added
- [ ] **Examples** provided for new features
- [ ] **Breaking changes** documented

**Git History**

- [ ] **Commits are logical** and atomic
- [ ] **Commit messages** follow conventional format
- [ ] **History is clean** (rebased if necessary)
- [ ] **No merge conflicts** with main branch

Submitting the Pull Request
--------------------------

PR Template
~~~~~~~~~~

Use the following template for your pull request description:

.. code-block:: markdown

   ## Description
   
   Brief description of what this PR does.
   
   Fixes #(issue number)
   
   ## Type of Change
   
   - [ ] Bug fix (non-breaking change which fixes an issue)
   - [ ] New feature (non-breaking change which adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
   - [ ] Documentation update
   - [ ] Refactoring (no functional changes)
   - [ ] Performance improvement
   
   ## Changes Made
   
   - List specific changes made
   - Include any important implementation details
   - Mention any design decisions
   
   ## Testing
   
   - [ ] Unit tests added/updated
   - [ ] Integration tests added/updated
   - [ ] Manual testing performed
   - [ ] All tests pass
   
   Describe the tests you ran and how to reproduce them.
   
   ## Documentation
   
   - [ ] Code comments added/updated
   - [ ] API documentation updated
   - [ ] User guide updated
   - [ ] CHANGELOG updated
   
   ## Breaking Changes
   
   List any breaking changes and migration path for users.
   
   ## Additional Notes
   
   Any additional information, concerns, or questions for reviewers.

PR Best Practices
~~~~~~~~~~~~~~~~

**Title and Description**

- **Clear, descriptive title** summarizing the change
- **Reference the issue number** (e.g., "Fixes #123")
- **Explain the problem** being solved
- **Describe the solution** implemented
- **Include testing information**

**Size and Scope**

- **Keep PRs focused** - one logical change per PR
- **Limit PR size** - aim for <500 lines of changes when possible
- **Break down large features** into smaller, reviewable chunks
- **Separate refactoring** from functional changes

**Communication**

- **Be responsive** to review feedback
- **Ask questions** if review comments are unclear
- **Explain design decisions** in PR description or comments
- **Update PR description** if scope changes during development

Review Process
-------------

Automated Checks
~~~~~~~~~~~~~~~

All PRs must pass automated checks:

**Continuous Integration**

- **Tests**: All unit and integration tests must pass
- **Linting**: Code must pass flake8 checks
- **Formatting**: Code must be Black formatted
- **Type Checking**: MyPy type checking must pass
- **Coverage**: Test coverage should not decrease
- **Documentation**: Sphinx docs must build successfully

**Security Checks**

- **Dependency scanning**: No known vulnerabilities in dependencies
- **Code scanning**: No obvious security issues detected

Human Review
~~~~~~~~~~~

**Review Timeline**

- **Initial response**: Within 2 business days
- **Full review**: Within 1 week for most PRs
- **Complex features**: May require additional review time

**Review Criteria**

Reviewers will check:

- **Correctness**: Does the code do what it claims?
- **Design**: Is the approach sound and consistent?
- **Testing**: Are tests comprehensive and correct?
- **Documentation**: Is documentation clear and complete?
- **Performance**: Are there any performance implications?
- **Security**: Are there any security concerns?
- **Maintainability**: Is the code easy to understand and modify?

**Addressing Review Feedback**

**Best Practices for Feedback**

- **Respond promptly** to review comments
- **Ask for clarification** if feedback is unclear
- **Implement suggestions** or explain why you disagree
- **Push new commits** for changes (don't force-push during review)
- **Mark conversations resolved** when addressed

**Common Review Issues**

1. **Missing tests** for new functionality
2. **Insufficient documentation** for public APIs
3. **Code style violations** or inconsistencies
4. **Performance concerns** for critical paths
5. **API design issues** or inconsistencies
6. **Missing error handling** or validation

**Handling Disagreements**

If you disagree with reviewer feedback:

1. **Explain your reasoning** clearly
2. **Provide evidence** or references if applicable
3. **Suggest alternatives** if the concern is valid
4. **Escalate to maintainers** if needed

Merging Process
--------------

Merge Requirements
~~~~~~~~~~~~~~~~~

Before merging, ensure:

- [ ] **All CI checks pass**
- [ ] **At least one approval** from a maintainer
- [ ] **All conversations resolved**
- [ ] **Branch is up to date** with main
- [ ] **No merge conflicts**

Merge Strategies
~~~~~~~~~~~~~~~

**Squash and Merge** (Preferred)

- Creates single commit on main branch
- Keeps main branch history clean
- Use for most feature PRs

**Merge Commit**

- Preserves PR branch history
- Use for complex features with logical commit history

**Rebase and Merge**

- Replays commits on main branch
- Use when PR has clean, logical commit history

Post-Merge Tasks
~~~~~~~~~~~~~~~

After your PR is merged:

- [ ] **Delete feature branch** (GitHub will prompt)
- [ ] **Update local repository**:

.. code-block:: bash

   git checkout main
   git pull upstream main
   git push origin main
   git branch -d feature/your-feature-name

- [ ] **Close related issues** if not automatically closed
- [ ] **Update documentation** if needed
- [ ] **Announce breaking changes** to users

Common Issues and Solutions
--------------------------

CI Failures
~~~~~~~~~~~

**Test Failures**

.. code-block:: bash

   # Run failing tests locally
   pytest tests/test_specific.py::test_failing_function -v
   
   # Debug with print statements or debugger
   pytest tests/test_specific.py::test_failing_function -v -s

**Formatting Issues**

.. code-block:: bash

   # Fix formatting
   black proteinMD/ tests/
   
   # Check what changed
   git diff

**Linting Issues**

.. code-block:: bash

   # Check specific issues
   flake8 proteinMD/
   
   # Fix common issues
   # - Remove unused imports
   # - Break long lines
   # - Add missing docstrings

Merge Conflicts
~~~~~~~~~~~~~~

.. code-block:: bash

   # Update your branch with latest main
   git checkout main
   git pull upstream main
   git checkout feature/your-branch
   git rebase main
   
   # Resolve conflicts in your editor
   # Then continue rebase
   git add .
   git rebase --continue
   
   # Force push updated branch
   git push --force-with-lease origin feature/your-branch

Review Delays
~~~~~~~~~~~~

If your PR isn't getting reviewed:

1. **Check if all CI passes** - reviewers often wait for green builds
2. **Ping reviewers** politely in PR comments after 1 week
3. **Ask in community channels** if you need urgent review
4. **Consider breaking down** large PRs into smaller ones

Quality Gates
-----------

Automated Quality Checks
~~~~~~~~~~~~~~~~~~~~~~~

All PRs must pass:

.. code-block:: yaml

   # .github/workflows/ci.yml example
   quality_checks:
     - name: Run tests
       run: pytest tests/ --cov=proteinMD
     
     - name: Check formatting  
       run: black --check proteinMD/
       
     - name: Run linting
       run: flake8 proteinMD/
       
     - name: Type checking
       run: mypy proteinMD/
       
     - name: Security scan
       run: bandit -r proteinMD/

Manual Review Checklist
~~~~~~~~~~~~~~~~~~~~~~

**Code Review Checklist**

Reviewers use this checklist:

- [ ] **Functionality**: Code works as intended
- [ ] **Tests**: Adequate test coverage for changes
- [ ] **Documentation**: Public APIs documented
- [ ] **Design**: Follows project architecture patterns
- [ ] **Performance**: No obvious performance regressions
- [ ] **Security**: No security vulnerabilities introduced
- [ ] **Compatibility**: Backwards compatibility maintained
- [ ] **Style**: Follows coding standards

**Documentation Review**

- [ ] **API docs**: All public functions documented
- [ ] **Examples**: Working examples provided
- [ ] **User guide**: Updated for user-facing changes
- [ ] **Migration guide**: Breaking changes documented

Getting Help
-----------

When to Ask for Help
~~~~~~~~~~~~~~~~~~~

- **Unclear requirements** in the issue
- **Complex architectural decisions** needed
- **Unsure about testing approach**
- **Performance optimization** needed
- **Breaking changes** required

Where to Get Help
~~~~~~~~~~~~~~~~

1. **Issue comments**: Ask questions in the related issue
2. **PR comments**: Ask for guidance in your PR
3. **Discussions**: Use GitHub Discussions for broader questions
4. **Community channels**: Developer chat or forums

Best Practices Summary
---------------------

**Before Starting**

- Read the issue carefully and ask questions
- Set up development environment properly  
- Plan your approach and get feedback

**During Development**

- Write tests first (TDD approach)
- Follow coding standards consistently
- Keep commits atomic and well-described
- Test thoroughly before submitting

**PR Submission**

- Use clear title and description
- Include testing information
- Reference related issues
- Keep PR focused and reasonably sized

**Review Process**

- Respond promptly to feedback
- Be open to suggestions and improvements
- Ask questions when unclear
- Update documentation as needed

**After Merge**

- Clean up your local repository
- Monitor for any issues introduced
- Help with follow-up questions or bugs

Contributing to ProteinMD is a collaborative process, and good pull requests make the review process smooth and efficient for everyone involved!
