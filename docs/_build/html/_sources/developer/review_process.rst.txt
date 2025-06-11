Review Process
==============

This document outlines the code review process for ProteinMD, including guidelines for both reviewers and contributors.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

Purpose of Code Review
~~~~~~~~~~~~~~~~~~~~~

Code review in ProteinMD serves multiple critical purposes:

**Quality Assurance**
  - Catch bugs and logic errors before they reach production
  - Ensure code follows project standards and best practices
  - Verify that tests adequately cover new functionality

**Knowledge Sharing**
  - Spread knowledge about codebase across team members
  - Share domain expertise and scientific knowledge
  - Educate contributors about project conventions

**Design Validation**
  - Ensure architectural consistency across the project
  - Validate API design decisions
  - Identify potential performance or maintainability issues

**Documentation**
  - Verify that code is properly documented
  - Ensure API documentation is complete and accurate
  - Check that user-facing changes include appropriate guides

Review Principles
~~~~~~~~~~~~~~~~

**Constructive and Respectful**
  Reviews should be helpful and educational, not critical or dismissive.

**Focus on Code, Not Person**
  Comments should address the code and approach, not the individual.

**Scientific Accuracy**
  Special attention to correctness of algorithms and physical implementations.

**Thorough but Timely**
  Reviews should be comprehensive but completed in reasonable time.

**Collaborative**
  Reviews are a conversation between reviewer and contributor.

Review Process Workflow
----------------------

Automated Checks
~~~~~~~~~~~~~~~

Before human review, all PRs must pass automated checks:

**Continuous Integration Pipeline**

.. code-block:: yaml

   # Example CI checks
   - Code formatting (Black)
   - Linting (Flake8)
   - Type checking (MyPy)
   - Unit tests (pytest)
   - Integration tests
   - Documentation build (Sphinx)
   - Security scan (Bandit)
   - Dependency scan

**Quality Gates**

- **Test Coverage**: Minimum 85% coverage for new code
- **Performance**: No significant performance regressions
- **Documentation**: All public APIs documented
- **Security**: No known vulnerabilities introduced

Review Assignment
~~~~~~~~~~~~~~~~

**Automatic Assignment**

GitHub automatically assigns reviewers based on:

- **CODEOWNERS file**: Specific maintainers for different modules
- **Round-robin**: Distribution among available reviewers
- **Expertise matching**: Assign experts for specialized areas

**Manual Assignment**

Contributors can request specific reviewers for:

- **Domain expertise**: Complex algorithms or scientific methods
- **Architectural input**: Significant design changes
- **Learning opportunities**: New contributors learning the codebase

**Review Team Structure**

.. code-block:: text

   Core Maintainers
   ├── Lead Maintainer (required for breaking changes)
   ├── Algorithm Experts (MD methods, force fields)
   ├── Performance Experts (optimization, CUDA)
   └── Documentation Specialists

   Module Maintainers
   ├── Core Engine (@core-team)
   ├── Force Fields (@forcefield-team)
   ├── Analysis Tools (@analysis-team)
   ├── I/O Systems (@io-team)
   └── Visualization (@viz-team)

Review Timeline
~~~~~~~~~~~~~~

**Expected Response Times**

- **Initial acknowledgment**: Within 24 hours
- **First review**: Within 3 business days for most PRs
- **Complex features**: Up to 1 week for thorough review
- **Hotfixes**: Within 4 hours for critical issues

**Escalation Process**

If reviews are delayed:

1. **Ping reviewers** politely after expected timeframe
2. **Ask in team channels** for alternative reviewers
3. **Contact maintainers** if blocking critical work
4. **Emergency review** for production issues

Types of Reviews
---------------

Feature Reviews
~~~~~~~~~~~~~~

**New Feature Implementation**

Focus areas for feature reviews:

- **Correctness**: Does the implementation match requirements?
- **Design**: Is the approach consistent with project architecture?
- **Testing**: Are all code paths and edge cases tested?
- **Documentation**: Is the feature properly documented?
- **Performance**: Are there any performance implications?
- **API Design**: Is the public interface well-designed?

**Example Feature Review Checklist:**

.. code-block:: markdown

   ## Feature Review: Velocity Verlet Integrator
   
   ### Correctness
   - [ ] Algorithm implementation matches literature
   - [ ] Energy conservation properties verified
   - [ ] Handles edge cases (zero forces, single atoms)
   - [ ] Proper error handling for invalid inputs
   
   ### Design
   - [ ] Follows existing integrator interface
   - [ ] Integrates cleanly with simulation loop
   - [ ] Configuration options are appropriate
   - [ ] Memory usage is reasonable
   
   ### Testing
   - [ ] Unit tests for algorithm correctness
   - [ ] Integration tests with full simulation
   - [ ] Performance benchmarks included
   - [ ] Comparison tests with existing integrators
   
   ### Documentation
   - [ ] API documentation complete
   - [ ] Examples provided
   - [ ] Scientific background explained
   - [ ] Performance characteristics documented

Bug Fix Reviews
~~~~~~~~~~~~~~

**Bug Fix Validation**

Focus areas for bug fix reviews:

- **Root Cause**: Does the fix address the actual problem?
- **Scope**: Are there other places with the same issue?
- **Testing**: Does the test reproduce the original bug?
- **Regression**: Could this fix introduce new problems?

**Example Bug Fix Review:**

.. code-block:: python

   # Original buggy code
   def calculate_bond_energy(r, k, r0):
       return k * (r - r0)**2  # Missing factor of 0.5!
   
   # Fixed code
   def calculate_bond_energy(r, k, r0):
       """Calculate harmonic bond energy.
       
       E = (1/2) * k * (r - r0)^2
       """
       return 0.5 * k * (r - r0)**2
   
   # Review focus:
   # 1. Is the physics correct now?
   # 2. Are there other energy functions with same bug?
   # 3. Does the test verify the correct energy value?
   # 4. Should we add integration tests for energy conservation?

Documentation Reviews
~~~~~~~~~~~~~~~~~~~

**Documentation Standards**

- **Completeness**: All public APIs documented
- **Accuracy**: Documentation matches implementation
- **Examples**: Working code examples provided
- **Clarity**: Clear for target audience (scientists)
- **Formatting**: Proper reStructuredText/Sphinx formatting

**Scientific Documentation Review**

Special attention for scientific accuracy:

.. code-block:: rst

   .. math::
      
      E_{bond} = \frac{1}{2} k (r - r_0)^2
   
   where :math:`k` is the force constant in kJ/(mol·Å²)
   and :math:`r_0` is the equilibrium bond length in Å.

Review checklist:
- Mathematical notation is correct
- Units are specified and consistent
- Physical interpretation is accurate
- References to literature are appropriate

Performance Reviews
~~~~~~~~~~~~~~~~~

**Performance-Critical Changes**

For changes affecting computational performance:

- **Benchmarking**: Before/after performance measurements
- **Profiling**: Identification of bottlenecks
- **Scaling**: Performance with different system sizes
- **Memory**: Memory usage and potential leaks

**Example Performance Review:**

.. code-block:: python

   # Performance review for optimized distance calculation
   
   # Before: O(n²) nested loops
   def calculate_distances_old(positions):
       n = len(positions)
       distances = np.zeros((n, n))
       for i in range(n):
           for j in range(i+1, n):
               dist = np.linalg.norm(positions[i] - positions[j])
               distances[i, j] = distances[j, i] = dist
       return distances
   
   # After: Vectorized calculation
   def calculate_distances_new(positions):
       diff = positions[:, np.newaxis] - positions[np.newaxis, :]
       return np.linalg.norm(diff, axis=2)
   
   # Review requirements:
   # 1. Benchmark showing performance improvement
   # 2. Memory usage comparison
   # 3. Numerical accuracy verification
   # 4. Test with different system sizes

Review Guidelines
----------------

For Reviewers
~~~~~~~~~~~~~

**Preparation**

.. code-block:: bash

   # Checkout PR branch for testing
   git fetch origin pull/123/head:pr-123
   git checkout pr-123
   
   # Install and test locally
   pip install -e .
   pytest tests/ -v
   
   # Run specific tests for changed areas
   pytest tests/test_integrators.py -v

**Review Checklist**

**Code Quality**

- [ ] **Readability**: Code is clear and well-organized
- [ ] **Style**: Follows project coding standards
- [ ] **Complexity**: Functions are not overly complex
- [ ] **Naming**: Variables and functions have descriptive names
- [ ] **Comments**: Complex logic is explained
- [ ] **Error Handling**: Appropriate exception handling

**Functionality**

- [ ] **Correctness**: Code does what it claims to do
- [ ] **Edge Cases**: Handles boundary conditions properly
- [ ] **Input Validation**: Validates inputs appropriately
- [ ] **Output Format**: Returns expected types and formats
- [ ] **Side Effects**: No unintended side effects

**Testing**

- [ ] **Coverage**: New code is adequately tested
- [ ] **Test Quality**: Tests are meaningful and thorough
- [ ] **Test Organization**: Tests are well-structured
- [ ] **Performance Tests**: Benchmarks for performance-critical code
- [ ] **Integration**: Tests work with existing codebase

**Documentation**

- [ ] **API Docs**: Public functions have proper docstrings
- [ ] **Examples**: Code examples are provided and work
- [ ] **User Guide**: User-facing changes documented
- [ ] **Comments**: Complex algorithms explained
- [ ] **Type Hints**: Proper type annotations

**Design**

- [ ] **Architecture**: Fits well with existing design
- [ ] **API Design**: Public interface is intuitive
- [ ] **Performance**: No obvious performance issues
- [ ] **Security**: No security vulnerabilities
- [ ] **Maintainability**: Code will be easy to maintain

**Scientific Accuracy**

- [ ] **Algorithms**: Implementations match literature
- [ ] **Units**: Physical units are correct and consistent
- [ ] **Mathematics**: Mathematical formulations are accurate
- [ ] **References**: Appropriate citations provided
- [ ] **Validation**: Results validated against known benchmarks

**Providing Feedback**

**Types of Comments**

.. code-box:: markdown

   # Blocking Issues (must be fixed)
   **Issue**: This will cause incorrect energies for stretched bonds.
   
   ```python
   # Current code
   energy = k * (r - r0)**2  # Missing factor of 0.5
   
   # Should be
   energy = 0.5 * k * (r - r0)**2
   ```
   
   **References**: See CHARMM force field documentation, Eq. 3.1
   
   # Suggestions (nice to have)
   **Suggestion**: Consider using np.linalg.norm for clarity:
   
   ```python
   # Instead of
   distance = np.sqrt(dx**2 + dy**2 + dz**2)
   
   # Consider
   distance = np.linalg.norm([dx, dy, dz])
   ```
   
   # Questions (seeking clarification)
   **Question**: Is this algorithm numerically stable for very small time steps?
   
   # Praise (positive feedback)
   **Nice**: Great use of vectorization here - this should be much faster!

**Comment Guidelines**

- **Be Specific**: Point to exact lines and provide concrete suggestions
- **Explain Why**: Don't just say what's wrong, explain the impact
- **Provide Solutions**: Suggest improvements when possible
- **Ask Questions**: When unsure, ask for clarification
- **Be Positive**: Acknowledge good code and clever solutions

For Contributors
~~~~~~~~~~~~~~~

**Responding to Reviews**

**Addressing Feedback**

.. code-block:: bash

   # Make requested changes
   git add .
   git commit -m "address review feedback: fix bond energy calculation"
   
   # Push updated branch
   git push origin feature/velocity-verlet-integrator

**Communication Guidelines**

- **Acknowledge All Comments**: Respond to each review comment
- **Ask for Clarification**: If feedback is unclear, ask questions
- **Explain Decisions**: If you disagree, explain your reasoning
- **Be Gracious**: Thank reviewers for their time and feedback
- **Mark Resolved**: Mark conversations as resolved when addressed

**Example Responses**

.. code-block:: markdown

   # Accepting feedback
   > The bond energy calculation is missing the 1/2 factor.
   
   Good catch! Fixed in commit abc123. I also added a test to verify 
   the energy values match CHARMM reference data.
   
   # Asking for clarification
   > This algorithm might not be numerically stable.
   
   Could you elaborate on what specific numerical issues you're concerned 
   about? I tested with time steps from 0.1 to 0.001 fs and didn't see 
   stability problems.
   
   # Disagreeing respectfully
   > Consider using a different data structure here.
   
   I initially considered that approach, but the current implementation 
   provides O(1) lookup which is critical for this hot path. The memory 
   overhead is acceptable given the performance benefit. What specific 
   concerns do you have about the current approach?

**Managing Multiple Reviewers**

When multiple reviewers provide conflicting feedback:

1. **Acknowledge the conflict** in PR comments
2. **Tag all relevant reviewers** for discussion
3. **Escalate to maintainers** if needed
4. **Document the final decision** and reasoning

Advanced Review Scenarios
-------------------------

Large Feature Reviews
~~~~~~~~~~~~~~~~~~~~

For large features that span multiple PRs:

**Design Review First**

.. code-block:: markdown

   ## Design Review: Enhanced Sampling Module
   
   ### Architecture Overview
   - New `proteinMD.sampling` module
   - Plugin architecture for sampling methods
   - Integration with existing simulation loop
   
   ### Public API
   ```python
   from proteinMD.sampling import UmbrellaSampling, REMD
   
   # Umbrella sampling
   sampler = UmbrellaSampling(reaction_coordinate, windows)
   simulation.add_sampler(sampler)
   
   # Replica exchange
   remd = REMD(temperatures=[300, 310, 320])
   remd.run(systems, steps=10000)
   ```
   
   ### Questions for Review
   1. Is the plugin architecture appropriate?
   2. Should sampling methods be tightly coupled to simulation?
   3. Are there performance concerns with the proposed approach?

**Phased Implementation Review**

Break large features into reviewable phases:

1. **Phase 1**: Core infrastructure and interfaces
2. **Phase 2**: Basic sampling method implementations  
3. **Phase 3**: Advanced features and optimizations
4. **Phase 4**: Documentation and examples

Cross-Repository Reviews
~~~~~~~~~~~~~~~~~~~~~~

For changes affecting multiple repositories:

**Coordinated Reviews**

- **Link related PRs** in descriptions
- **Test integration** across repositories
- **Coordinate merge timing**
- **Update documentation** in all affected repos

Security Reviews
~~~~~~~~~~~~~~~

For security-sensitive changes:

**Security Review Checklist**

- [ ] **Input validation**: All inputs properly validated
- [ ] **File access**: No arbitrary file access vulnerabilities
- [ ] **Code injection**: No dynamic code execution vulnerabilities
- [ ] **Dependency security**: New dependencies are secure
- [ ] **Data handling**: Sensitive data handled appropriately

**Example Security Review**

.. code-block:: python

   # Security review for file loading function
   
   def load_trajectory(filename: str) -> Trajectory:
       """Load trajectory from file."""
       # Security concern: Path traversal vulnerability
       
       # Bad: Allows access to any file
       with open(filename, 'rb') as f:
           return parse_trajectory(f.read())
       
       # Better: Validate filename
       if '..' in filename or filename.startswith('/'):
           raise ValueError("Invalid filename")
       
       # Best: Use safe path handling
       safe_path = Path(filename).resolve()
       if not safe_path.is_relative_to(Path.cwd()):
           raise ValueError("File must be in current directory")

Review Tools and Automation
---------------------------

GitHub Integration
~~~~~~~~~~~~~~~~~

**Review Request Automation**

.. code-block:: yaml

   # .github/CODEOWNERS
   # Global owners
   * @proteinmd/core-maintainers
   
   # Module-specific owners
   proteinMD/core/ @proteinmd/core-team
   proteinMD/forcefield/ @proteinmd/forcefield-team
   proteinMD/analysis/ @proteinmd/analysis-team
   docs/ @proteinmd/docs-team

**PR Templates**

.. code-block:: markdown

   ## Pull Request Template
   
   ### Description
   Brief description of changes
   
   ### Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ### Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests pass
   - [ ] Manual testing performed
   
   ### Review Focus Areas
   Please pay special attention to:
   - Algorithm correctness in `calculate_forces()`
   - Performance impact of vectorization changes
   - API design for new `Integrator` interface

**Automated Review Checks**

.. code-block:: yaml

   # .github/workflows/review-checks.yml
   name: Review Checks
   on: pull_request
   
   jobs:
     review_checks:
       runs-on: ubuntu-latest
       steps:
         - name: Check PR size
           if: github.event.pull_request.additions > 1000
           run: echo "::warning::Large PR - consider breaking into smaller pieces"
         
         - name: Check test coverage
           run: |
             coverage run -m pytest
             coverage report --fail-under=85

Code Review Metrics
~~~~~~~~~~~~~~~~~~

**Tracking Review Quality**

- **Review thoroughness**: Comments per line of code changed
- **Review turnaround time**: Time from PR submission to merge
- **Bug catch rate**: Issues found in review vs. production
- **Knowledge sharing**: Number of educational comments

**Review Dashboard**

Track key metrics:

.. code-block:: text

   ProteinMD Review Metrics (Last 30 Days)
   =====================================
   
   Review Turnaround Time:
   - Average: 2.3 days
   - 95th percentile: 5.1 days
   
   Review Quality:
   - Average comments per PR: 4.2
   - PRs requiring multiple rounds: 23%
   
   Bug Detection:
   - Issues found in review: 15
   - Issues found in production: 2
   - Review effectiveness: 88%

Best Practices
--------------

For Effective Reviews
~~~~~~~~~~~~~~~~~~~~

**Reviewer Best Practices**

1. **Review promptly**: Respect contributor's time
2. **Be thorough**: Don't rush through reviews
3. **Focus on important issues**: Distinguish between critical and style issues
4. **Provide examples**: Show better alternatives when possible
5. **Learn from code**: Use reviews as learning opportunities
6. **Communicate clearly**: Be specific and actionable in feedback

**Contributor Best Practices**

1. **Self-review first**: Review your own code before submitting
2. **Keep PRs focused**: One logical change per PR
3. **Write clear descriptions**: Explain what and why
4. **Respond promptly**: Address feedback quickly
5. **Test thoroughly**: Ensure code works before review
6. **Be open to feedback**: View reviews as collaborative improvement

Review Culture
~~~~~~~~~~~~~

**Building Positive Review Culture**

- **Celebrate good code**: Acknowledge clever solutions and improvements
- **Share knowledge**: Use reviews to teach and learn
- **Be patient**: Everyone is learning and improving
- **Focus on code quality**: Reviews improve the entire codebase
- **Maintain high standards**: Consistent quality benefits everyone

**Handling Difficult Situations**

**Persistent Disagreements**

1. **Discuss in person/video call** if possible
2. **Involve neutral third party** (another maintainer)
3. **Document the decision** and reasoning
4. **Move forward** - don't block indefinitely

**Review Conflicts**

1. **Stay professional** and focus on technical merits
2. **Assume good intentions** from all parties
3. **Escalate to maintainers** if needed
4. **Learn from conflicts** to improve process

Summary
-------

Effective code review is essential for ProteinMD's quality and success:

**Key Principles**

- **Quality Focus**: Reviews ensure high code quality and correctness
- **Scientific Rigor**: Special attention to physical accuracy and algorithms
- **Collaborative Learning**: Reviews are opportunities for knowledge sharing
- **Constructive Feedback**: Reviews should be helpful and educational
- **Timely Process**: Reviews should be thorough but not block progress

**Review Checklist**

- Functionality and correctness
- Code quality and style
- Testing and documentation
- Performance and security
- Scientific accuracy
- API design and usability

By following these review guidelines, we maintain ProteinMD as a high-quality, scientifically accurate, and maintainable molecular dynamics simulation library.
