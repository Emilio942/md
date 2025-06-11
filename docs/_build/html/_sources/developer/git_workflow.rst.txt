Git Workflow
============

This document describes the Git workflow and branching strategy used in ProteinMD development.

.. contents:: Contents
   :local:
   :depth: 2

Branching Strategy
-----------------

Overview
~~~~~~~~

ProteinMD uses a **GitHub Flow** based branching strategy with some modifications for scientific software development:

.. code-block:: text

   main ─────●─────●─────●─────●─────●─────●─────→
             │     │     │     │     │     │
             │     │   feature/  │     │     │
             │     │   analysis  │     │     │
             │     │      ↓      │     │     │
             │     └─────●───────┘     │     │
             │                         │     │
           hotfix/                  release/  │
           critical-bug             v1.2.0    │
                ↓                      ↓      │
               ●──────────────────────●───────┘
               │                      │
               └──────────────────────┘

**Key Principles:**

- **main branch** is always deployable and stable
- **Feature branches** for all development work
- **Release branches** for preparing releases
- **Hotfix branches** for critical production fixes
- **No direct commits** to main (except hotfixes)

Branch Types
~~~~~~~~~~~

**Main Branch**

- **Purpose**: Stable, deployable code
- **Protection**: Protected branch with required reviews
- **Merging**: Only via pull requests with passing CI
- **Naming**: ``main``

.. code-block:: bash

   # Keep main up to date
   git checkout main
   git pull upstream main

**Feature Branches**

- **Purpose**: Development of new features or improvements
- **Source**: Branched from main
- **Naming**: ``feature/<description>``
- **Lifecycle**: Created → Developed → PR → Merged → Deleted

.. code-block:: bash

   # Create feature branch
   git checkout main
   git pull upstream main
   git checkout -b feature/velocity-verlet-integrator
   
   # Work on feature
   git add .
   git commit -m "feat(core): implement velocity verlet integrator"
   
   # Push to your fork
   git push origin feature/velocity-verlet-integrator

**Bug Fix Branches**

- **Purpose**: Fix specific bugs or issues
- **Source**: Branched from main
- **Naming**: ``bugfix/<description>``
- **Priority**: Higher priority than features

.. code-block:: bash

   # Create bugfix branch
   git checkout -b bugfix/memory-leak-in-trajectory-writer
   
   # Fix the bug
   git add .
   git commit -m "fix(io): resolve memory leak in trajectory writer"

**Documentation Branches**

- **Purpose**: Documentation improvements
- **Source**: Branched from main  
- **Naming**: ``docs/<description>``

.. code-block:: bash

   # Create docs branch
   git checkout -b docs/improve-api-documentation

**Refactoring Branches**

- **Purpose**: Code restructuring without functional changes
- **Source**: Branched from main
- **Naming**: ``refactor/<description>``

.. code-block:: bash

   # Create refactor branch
   git checkout -b refactor/reorganize-forcefield-modules

**Release Branches**

- **Purpose**: Prepare releases, fix release-specific issues
- **Source**: Branched from main
- **Naming**: ``release/v<version>``
- **Merging**: Into main and develop (if used)

.. code-block:: bash

   # Create release branch (maintainers only)
   git checkout -b release/v1.2.0
   
   # Update version numbers, documentation
   git add .
   git commit -m "chore: prepare v1.2.0 release"

**Hotfix Branches**

- **Purpose**: Critical fixes for production issues
- **Source**: Branched from main
- **Naming**: ``hotfix/<description>``
- **Priority**: Highest priority, fast-tracked review

.. code-block:: bash

   # Create hotfix branch (for critical issues)
   git checkout -b hotfix/critical-simulation-crash

Workflow for Contributors
------------------------

Initial Setup
~~~~~~~~~~~~

**1. Fork Repository**

.. code-block:: bash

   # Fork on GitHub, then clone your fork
   git clone https://github.com/yourusername/proteinmd.git
   cd proteinmd
   
   # Add upstream remote
   git remote add upstream https://github.com/proteinmd/proteinmd.git
   
   # Verify remotes
   git remote -v

**2. Configure Git**

.. code-block:: bash

   # Set up your identity
   git config user.name "Your Name"
   git config user.email "your.email@example.com"
   
   # Optional: Set up GPG signing
   git config user.signingkey YOUR_GPG_KEY
   git config commit.gpgsign true

**3. Set Up Development Environment**

.. code-block:: bash

   # Create development environment
   conda create -n proteinmd-dev python=3.9
   conda activate proteinmd-dev
   
   # Install dependencies
   pip install -e .
   pip install -r requirements-dev.txt
   
   # Set up pre-commit hooks
   pre-commit install

Standard Development Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Start New Work**

.. code-block:: bash

   # Sync with upstream
   git checkout main
   git pull upstream main
   git push origin main
   
   # Create feature branch
   git checkout -b feature/your-feature-name

**2. Development Cycle**

.. code-block:: bash

   # Make changes
   # Edit files, write tests, update docs
   
   # Stage changes
   git add .
   
   # Commit with descriptive message
   git commit -m "feat(module): add new functionality
   
   Detailed description of what was added and why.
   
   Closes #123"
   
   # Push to your fork
   git push origin feature/your-feature-name

**3. Keep Branch Updated**

.. code-block:: bash

   # Regularly sync with upstream main
   git fetch upstream
   git rebase upstream/main
   
   # Force push if rebase changed history
   git push --force-with-lease origin feature/your-feature-name

**4. Submit Pull Request**

- Create PR on GitHub
- Fill out PR template
- Respond to review feedback
- Update branch as needed

**5. After Merge**

.. code-block:: bash

   # Clean up after merge
   git checkout main
   git pull upstream main
   git push origin main
   
   # Delete feature branch
   git branch -d feature/your-feature-name
   git push origin --delete feature/your-feature-name

Commit Guidelines
----------------

Commit Message Format
~~~~~~~~~~~~~~~~~~~~

Use **Conventional Commits** specification:

.. code-block:: text

   <type>[optional scope]: <description>
   
   [optional body]
   
   [optional footer(s)]

**Types:**

- ``feat``: New features
- ``fix``: Bug fixes
- ``docs``: Documentation only changes
- ``style``: Changes that don't affect meaning (formatting, etc.)
- ``refactor``: Code changes that neither fix bugs nor add features
- ``perf``: Performance improvements
- ``test``: Adding missing tests or correcting existing tests
- ``chore``: Changes to build process or auxiliary tools

**Scopes (optional):**

- ``core``: Core simulation engine
- ``forcefield``: Force field implementations  
- ``analysis``: Analysis tools
- ``io``: Input/output functionality
- ``vis``: Visualization
- ``cli``: Command line interface
- ``docs``: Documentation

**Examples:**

.. code-block:: text

   feat(core): implement velocity Verlet integrator
   
   Add velocity Verlet integration algorithm with improved energy
   conservation for molecular dynamics simulations.
   
   - Configurable time step
   - Energy conservation tests included
   - Documentation updated
   
   Closes #123

.. code-block:: text

   fix(io): resolve memory leak in trajectory writer
   
   Fixed memory leak caused by unclosed file handles in
   TrajectoryWriter class when exceptions occurred.
   
   Fixes #456

.. code-block:: text

   docs(api): update MDSystem class documentation
   
   - Add examples for all public methods
   - Clarify parameter types and units
   - Fix formatting issues

Commit Best Practices
~~~~~~~~~~~~~~~~~~~~

**1. Atomic Commits**

Each commit should represent one logical change:

.. code-block:: bash

   # Good: Separate commits for separate concerns
   git add tests/test_integrator.py
   git commit -m "test(core): add tests for velocity Verlet integrator"
   
   git add proteinMD/core/integrators.py
   git commit -m "feat(core): implement velocity Verlet integrator"
   
   git add docs/api/integrators.rst
   git commit -m "docs(api): document velocity Verlet integrator"

**2. Commit Often**

Make frequent, small commits during development:

.. code-block:: bash

   # Work incrementally
   git add -p  # Stage partial changes
   git commit -m "feat(core): add basic integrator interface"
   
   # Continue development
   git add tests/test_verlet.py
   git commit -m "test(core): add unit tests for Verlet integrator"

**3. Good Commit Messages**

Write clear, descriptive commit messages:

.. code-block:: text

   # Good: Clear and descriptive
   fix(forcefield): correct bond angle calculation in CHARMM36
   
   The angle calculation was using radians instead of degrees,
   causing incorrect force calculations for angle terms.
   
   # Bad: Vague and unhelpful
   fix stuff
   more changes
   working on bug

Branch Management
----------------

Keeping Branches Updated
~~~~~~~~~~~~~~~~~~~~~~~

**Rebase vs Merge**

Prefer **rebase** for feature branches to maintain clean history:

.. code-block:: bash

   # Rebase feature branch onto latest main
   git fetch upstream
   git rebase upstream/main
   
   # If conflicts occur, resolve them
   git add .
   git rebase --continue
   
   # Force push (use --force-with-lease for safety)
   git push --force-with-lease origin feature/your-branch

**When to Use Merge**

Use merge for:
- Bringing release branches back to main
- Preserving parallel development history
- When rebase would be too complex

.. code-block:: bash

   # Merge release branch
   git checkout main
   git merge --no-ff release/v1.2.0

Handling Conflicts
~~~~~~~~~~~~~~~~~

**During Rebase**

.. code-block:: bash

   # Start rebase
   git rebase upstream/main
   
   # If conflicts occur
   # 1. Edit conflicted files
   # 2. Stage resolved files
   git add resolved_file.py
   
   # 3. Continue rebase
   git rebase --continue
   
   # If you need to abort
   git rebase --abort

**During Merge**

.. code-block:: bash

   # If merge has conflicts
   git merge upstream/main
   
   # Edit conflicted files, then
   git add resolved_file.py
   git commit  # Complete the merge

Branch Cleanup
~~~~~~~~~~~~~

**Local Cleanup**

.. code-block:: bash

   # List all branches
   git branch -a
   
   # Delete merged local branches
   git branch -d feature/completed-feature
   
   # Force delete unmerged branches (be careful!)
   git branch -D feature/abandoned-feature
   
   # Clean up remote tracking branches
   git remote prune origin

**Remote Cleanup**

.. code-block:: bash

   # Delete remote branch
   git push origin --delete feature/completed-feature
   
   # Clean up your fork
   git push origin --prune

Advanced Git Workflows
---------------------

Interactive Rebase
~~~~~~~~~~~~~~~~~

Use interactive rebase to clean up commit history:

.. code-block:: bash

   # Rebase last 3 commits interactively
   git rebase -i HEAD~3
   
   # In the editor, you can:
   # - pick: use commit as-is
   # - reword: change commit message
   # - edit: modify commit
   # - squash: combine with previous commit
   # - drop: remove commit

**Example interactive rebase:**

.. code-block:: text

   pick f7f3f6d feat(core): add integrator base class
   squash 310154e fix(core): fix typo in integrator
   reword a5f4a0d feat(core): implement velocity Verlet
   
   # This will:
   # 1. Keep first commit
   # 2. Squash second commit into first
   # 3. Allow editing third commit message

Cherry-picking
~~~~~~~~~~~~~

Apply specific commits to another branch:

.. code-block:: bash

   # Apply specific commit to current branch
   git cherry-pick <commit-hash>
   
   # Cherry-pick a range of commits
   git cherry-pick A^..B
   
   # Cherry-pick without committing (for modifications)
   git cherry-pick --no-commit <commit-hash>

Stashing
~~~~~~~

Temporarily save uncommitted changes:

.. code-block:: bash

   # Stash current changes
   git stash
   
   # Stash with message
   git stash save "work in progress on feature X"
   
   # List stashes
   git stash list
   
   # Apply latest stash
   git stash pop
   
   # Apply specific stash
   git stash apply stash@{0}

Release Workflow
---------------

Version Tagging
~~~~~~~~~~~~~~

**Semantic Versioning**

ProteinMD follows semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes

**Creating Releases**

.. code-block:: bash

   # Create annotated tag for release
   git tag -a v1.2.0 -m "Release version 1.2.0
   
   Features:
   - New velocity Verlet integrator
   - Improved energy conservation
   - Enhanced trajectory analysis
   
   Bug fixes:
   - Fixed memory leak in trajectory writer
   - Corrected force field parameter loading"
   
   # Push tag
   git push upstream v1.2.0

**Release Branch Workflow**

.. code-block:: bash

   # Create release branch
   git checkout -b release/v1.2.0
   
   # Update version numbers
   # - setup.py
   # - proteinMD/__init__.py
   # - docs/conf.py
   
   # Update CHANGELOG
   git add .
   git commit -m "chore: prepare v1.2.0 release"
   
   # Create PR for release branch
   # After review and merge, tag the release

Collaboration Patterns
---------------------

Code Reviews
~~~~~~~~~~~

**Preparing for Review**

.. code-block:: bash

   # Clean up commit history
   git rebase -i upstream/main
   
   # Ensure tests pass
   pytest tests/
   
   # Check code quality
   black proteinMD/
   flake8 proteinMD/
   
   # Update documentation
   # Write clear PR description

**Responding to Reviews**

.. code-block:: bash

   # Make requested changes
   git add .
   git commit -m "address review feedback: improve error handling"
   
   # If major changes needed, consider squashing
   git rebase -i HEAD~2  # Squash the fix commit

Pair Programming
~~~~~~~~~~~~~~~

**Shared Branch Development**

.. code-block:: bash

   # Create shared feature branch
   git checkout -b feature/shared-development
   git push origin feature/shared-development
   
   # Both developers work on same branch
   # Coordinate pushes and pulls
   git pull origin feature/shared-development
   # ... make changes ...
   git push origin feature/shared-development

**Co-authored Commits**

.. code-block:: bash

   git commit -m "feat(core): implement new algorithm
   
   Co-authored-by: Jane Developer <jane@example.com>"

Emergency Procedures
-------------------

Hotfixes
~~~~~~~

For critical production issues:

.. code-block:: bash

   # Create hotfix branch from main
   git checkout main
   git pull upstream main
   git checkout -b hotfix/critical-crash-fix
   
   # Make minimal fix
   git add .
   git commit -m "fix: resolve critical simulation crash
   
   Emergency fix for crash when loading large PDB files.
   Properly handles memory allocation for systems > 100k atoms."
   
   # Fast-track review and merge
   # Tag emergency release if needed

Reverting Changes
~~~~~~~~~~~~~~~~

**Revert Commits**

.. code-block:: bash

   # Revert specific commit
   git revert <commit-hash>
   
   # Revert merge commit
   git revert -m 1 <merge-commit-hash>
   
   # Revert range of commits
   git revert <start-commit>^..<end-commit>

**Reset Branch**

.. code-block:: bash

   # Reset to previous commit (dangerous!)
   git reset --hard <commit-hash>
   
   # Reset and keep changes in working directory
   git reset --soft <commit-hash>

Tools and Aliases
-----------------

Useful Git Aliases
~~~~~~~~~~~~~~~~~

Add these to your ``~/.gitconfig``:

.. code-block:: ini

   [alias]
       # Short status
       st = status -s
       
       # Pretty log
       lg = log --oneline --graph --decorate --all
       
       # Commit with message
       cm = commit -m
       
       # Checkout branch
       co = checkout
       
       # Create and checkout branch
       cob = checkout -b
       
       # Pull with rebase
       pr = pull --rebase
       
       # Push current branch
       pc = push origin HEAD
       
       # Undo last commit but keep changes
       undo = reset HEAD~1 --mixed
       
       # Show files in last commit
       dl = show --name-only
       
       # Clean up merged branches
       cleanup = "!git branch --merged | grep -v '\\*\\|main\\|develop' | xargs -n 1 git branch -d"

Git Hooks
~~~~~~~~

Set up useful git hooks:

.. code-block:: bash

   # Pre-commit hook to run tests
   cat > .git/hooks/pre-commit << 'EOF'
   #!/bin/bash
   black --check proteinMD/ tests/
   flake8 proteinMD/ tests/
   pytest tests/ --tb=short
   EOF
   
   chmod +x .git/hooks/pre-commit

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

**"Your branch is ahead of origin/main"**

.. code-block:: bash

   # You committed to main instead of feature branch
   # Create feature branch from current state
   git checkout -b feature/accidental-commits
   
   # Reset main to upstream
   git checkout main
   git reset --hard upstream/main

**"Cannot rebase: You have unstaged changes"**

.. code-block:: bash

   # Stash changes before rebase
   git stash
   git rebase upstream/main
   git stash pop

**"Merge conflict in file.py"**

.. code-block:: bash

   # Edit file.py to resolve conflicts
   # Remove conflict markers: <<<<<<<, =======, >>>>>>>
   git add file.py
   git rebase --continue  # or git commit for merge

**"Remote rejected (non-fast-forward)"**

.. code-block:: bash

   # Someone else pushed to the same branch
   git pull --rebase origin feature-branch
   # Resolve any conflicts
   git push origin feature-branch

Recovery Procedures
~~~~~~~~~~~~~~~~~~

**Lost Commits**

.. code-block:: bash

   # Find lost commits in reflog
   git reflog
   
   # Recover specific commit
   git checkout <commit-hash>
   git checkout -b recovery-branch

**Accidentally Deleted Branch**

.. code-block:: bash

   # Find branch in reflog
   git reflog | grep branch-name
   
   # Recover branch
   git checkout -b branch-name <commit-hash>

Best Practices Summary
---------------------

**Daily Workflow**

1. **Start with updated main**
2. **Create focused feature branches**
3. **Commit frequently with good messages**
4. **Keep branches updated with rebase**
5. **Clean up history before PR**

**Code Quality**

1. **Run tests before committing**
2. **Use pre-commit hooks**
3. **Follow conventional commit format**
4. **Write descriptive commit messages**
5. **Keep commits atomic and logical**

**Collaboration**

1. **Communicate in PRs and issues**
2. **Respond to reviews promptly**
3. **Keep PRs focused and small**
4. **Document breaking changes**
5. **Help others with reviews**

This Git workflow ensures clean history, easy collaboration, and maintainable code for the ProteinMD project.
