=============
CLI Reference
=============

ProteinMD provides a comprehensive command-line interface for running simulations, 
analyzing trajectories, and managing workflows without writing Python code.

Overview
========

The CLI is designed for:

* **Quick simulations** - Run standard MD with minimal setup
* **Batch processing** - Process multiple structures automatically  
* **Pipeline integration** - Easy integration with HPC job schedulers
* **Reproducible workflows** - Template-based simulation setup

Installation and Setup
======================

The CLI is installed automatically with ProteinMD::

    pip install proteinMD

Verify installation::

    proteinmd --version
    proteinmd --help

Global Options
==============

All commands support these global options::

    proteinmd [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]

**Global Options:**

``--verbose, -v``
    Increase verbosity (use -vv for debug output)

``--quiet, -q``  
    Suppress output except errors

``--config FILE``
    Use custom configuration file

``--output-dir DIR``
    Set output directory (default: current directory)

``--temp-dir DIR``
    Set temporary directory (default: system temp)

``--log-file FILE``
    Write logs to file

``--no-color``
    Disable colored output

Commands Overview
=================

Core Commands
-------------

``simulate``
    Run molecular dynamics simulations

``analyze``  
    Analyze trajectory data

``convert``
    Convert between file formats

``validate``
    Validate structures and parameters

``benchmark``
    Performance benchmarking

``workflow``
    Run predefined workflows

Utility Commands  
----------------

``examples``
    Download example files and tutorials

``config``
    Manage configuration settings

``info``
    System information and diagnostics

``clean``
    Clean temporary and cache files

simulate - Run MD Simulations
=============================

Run molecular dynamics simulations with automatic setup.

Basic Usage
-----------

::

    proteinmd simulate INPUT [OPTIONS]

**Arguments:**

``INPUT``
    Input structure file (PDB, MOL2, etc.)

**Basic Options:**

``--steps INTEGER``
    Number of simulation steps (default: 100000)

``--timestep FLOAT``
    Integration timestep in fs (default: 2.0)

``--temperature FLOAT``
    Simulation temperature in K (default: 300.0)

``--pressure FLOAT``  
    Simulation pressure in atm (default: 1.0)

``--output DIR``
    Output directory (default: ./simulation_output)

Force Field Options
-------------------

``--forcefield NAME``
    Force field to use (default: amber14sb)
    
    Choices: amber14sb, amber99sb-ildn, charmm36, gromos54a7

``--water-model NAME``
    Water model (default: tip3p)
    
    Choices: tip3p, tip4p, tip5p, spc, spce

``--ff-params FILE``
    Custom force field parameter file

Environment Options
-------------------

``--solvent-type TYPE``
    Solvent environment (default: explicit)
    
    Choices: explicit, implicit, vacuum

``--box-size FLOAT``
    Cubic box size in nm (default: auto)

``--box-shape SHAPE``
    Box shape (default: cubic)
    
    Choices: cubic, rectangular, dodecahedron, octahedron

``--padding FLOAT``
    Minimum distance to box edge in nm (default: 1.0)

``--ion-concentration FLOAT``
    Salt concentration in M (default: 0.15)

``--neutralize``
    Add counterions to neutralize system

Simulation Control
------------------

``--minimize-steps INTEGER``
    Energy minimization steps (default: 1000)

``--equilibration-steps INTEGER``
    Equilibration steps (default: 100000)

``--production-steps INTEGER``
    Production MD steps (default: 500000)

``--ensemble ENSEMBLE``
    Simulation ensemble (default: npt)
    
    Choices: nve, nvt, npt, nph

``--thermostat TYPE``
    Temperature control method (default: langevin)
    
    Choices: langevin, nose-hoover, andersen, berendsen

``--barostat TYPE``
    Pressure control method (default: monte-carlo)
    
    Choices: monte-carlo, parrinello-rahman, berendsen

Output Options
--------------

``--save-frequency INTEGER``
    Save trajectory every N steps (default: 1000)

``--energy-frequency INTEGER``
    Save energy every N steps (default: 100)

``--checkpoint-frequency INTEGER``
    Save checkpoint every N steps (default: 10000)

``--trajectory-format FORMAT``
    Trajectory file format (default: dcd)
    
    Choices: dcd, xtc, trr, netcdf, pdb

``--precision PRECISION``
    Coordinate precision (default: mixed)
    
    Choices: single, double, mixed

``--compress``
    Compress trajectory files

Examples
--------

**Basic protein simulation:**

::

    proteinmd simulate protein.pdb --steps 1000000

**High-temperature simulation:**

::

    proteinmd simulate protein.pdb \
        --temperature 350 \
        --steps 2000000 \
        --output high_temp_sim

**Implicit solvent simulation:**

::

    proteinmd simulate protein.pdb \
        --solvent-type implicit \
        --steps 5000000

**Custom force field:**

::

    proteinmd simulate protein.pdb \
        --forcefield charmm36 \
        --water-model tip4p \
        --ion-concentration 0.1

**Membrane protein simulation:**

::

    proteinmd simulate membrane_protein.pdb \
        --environment membrane \
        --lipid-type popc \
        --steps 10000000

analyze - Trajectory Analysis
=============================

Analyze molecular dynamics trajectories with built-in tools.

Basic Usage
-----------

::

    proteinmd analyze TRAJECTORY TOPOLOGY [OPTIONS]

**Arguments:**

``TRAJECTORY``
    Trajectory file (DCD, XTC, TRR, etc.)

``TOPOLOGY``
    Topology/structure file (PDB, GRO, PSF, etc.)

**Analysis Options:**

``--rmsd``
    Calculate root mean square deviation

``--rmsf``  
    Calculate root mean square fluctuation

``--radius-of-gyration``
    Calculate radius of gyration

``--secondary-structure``
    Analyze secondary structure

``--hydrogen-bonds``
    Analyze hydrogen bonding

``--contacts``
    Calculate contact maps

``--distances``
    Calculate specific distances

``--angles``
    Calculate specific angles

``--dihedrals``
    Calculate dihedral angles

Selection Options
-----------------

``--selection TEXT``
    Atom selection for analysis (default: protein)

``--reference-frame INTEGER``
    Reference frame for RMSD (default: 0)

``--start-frame INTEGER``
    First frame to analyze (default: 0)

``--end-frame INTEGER``
    Last frame to analyze (default: -1)

``--stride INTEGER``
    Frame stride for analysis (default: 1)

Output Options
--------------

``--output-prefix TEXT``
    Prefix for output files (default: analysis)

``--plot``
    Generate plots for analyses

``--format FORMAT``
    Output data format (default: txt)
    
    Choices: txt, csv, json, hdf5

``--interactive``
    Launch interactive analysis session

Examples
--------

**Basic RMSD analysis:**

::

    proteinmd analyze trajectory.dcd topology.pdb --rmsd --plot

**Comprehensive analysis:**

::

    proteinmd analyze trajectory.dcd topology.pdb \
        --rmsd --rmsf --secondary-structure \
        --hydrogen-bonds --plot

**Backbone analysis:**

::

    proteinmd analyze trajectory.dcd topology.pdb \
        --rmsd --selection "backbone" \
        --reference-frame 100

**Time window analysis:**

::

    proteinmd analyze trajectory.dcd topology.pdb \
        --rmsd --start-frame 1000 --end-frame 5000 \
        --stride 10

**Custom distance analysis:**

::

    proteinmd analyze trajectory.dcd topology.pdb \
        --distances \
        --distance-pairs "resid 10 and name CA" "resid 50 and name CA"

convert - File Format Conversion
================================

Convert between different molecular file formats.

Basic Usage
-----------

::

    proteinmd convert INPUT OUTPUT [OPTIONS]

**Arguments:**

``INPUT``
    Input file

``OUTPUT``
    Output file (format detected from extension)

**Options:**

``--input-format FORMAT``
    Explicitly specify input format

``--output-format FORMAT``
    Explicitly specify output format

``--selection TEXT``
    Convert only selected atoms

``--frame INTEGER``
    Convert specific frame (trajectories)

``--start-frame INTEGER``
    First frame to convert

``--end-frame INTEGER``  
    Last frame to convert

``--stride INTEGER``
    Frame stride

``--remove-water``
    Remove water molecules

``--remove-ions``
    Remove ions

``--remove-hydrogens``
    Remove hydrogen atoms

``--center``
    Center coordinates

``--align-to FILE``
    Align structure to reference

Examples
--------

**Structure format conversion:**

::

    proteinmd convert protein.pdb protein.mol2

**Trajectory format conversion:**

::

    proteinmd convert trajectory.dcd trajectory.xtc \
        --start-frame 100 --stride 10

**Extract protein only:**

::

    proteinmd convert complex.pdb protein_only.pdb \
        --selection "protein" --remove-water

**Trajectory to PDB frames:**

::

    proteinmd convert trajectory.dcd frames.pdb \
        --stride 100

workflow - Predefined Workflows
===============================

Run complete simulation and analysis workflows from templates.

Basic Usage
-----------

::

    proteinmd workflow TEMPLATE INPUT [OPTIONS]

**Arguments:**

``TEMPLATE``
    Workflow template name or file

``INPUT``
    Input structure file

**Built-in Templates:**

``protein_folding``
    Complete protein folding study

``ligand_binding``
    Protein-ligand binding analysis

``membrane_protein``
    Membrane protein simulation

``free_energy``
    Free energy perturbation calculation

``enhanced_sampling``
    Enhanced sampling with REMD

``stability_analysis``
    Protein stability assessment

``conformational_sampling``
    Conformational space exploration

Custom Workflows
----------------

``--workflow-file FILE``
    Use custom workflow YAML file

``--parameters FILE``
    Override workflow parameters

``--steps SECTION:VALUE``
    Override specific workflow steps

``--dry-run``
    Show workflow steps without running

Examples
--------

**Protein folding study:**

::

    proteinmd workflow protein_folding unfolded.pdb \
        --output folding_study

**Custom workflow:**

::

    proteinmd workflow custom_workflow.yaml protein.pdb \
        --parameters my_params.yaml

**Free energy calculation:**

::

    proteinmd workflow free_energy complex.pdb \
        --steps "lambda_windows:21" \
        --steps "simulation_time:5ns"

batch - Batch Processing
========================

Process multiple structures or trajectories in parallel.

Basic Usage
-----------

::

    proteinmd batch COMMAND INPUT_DIR [OPTIONS]

**Arguments:**

``COMMAND``
    Command to run in batch (simulate, analyze, convert)

``INPUT_DIR``
    Directory containing input files

**Options:**

``--pattern PATTERN``
    File pattern to match (default: *.pdb)

``--output-dir DIR``
    Output directory (default: batch_output)

``--parallel INTEGER``
    Number of parallel processes (default: auto)

``--resume``
    Resume interrupted batch job

``--force``
    Overwrite existing results

Examples
--------

**Batch simulation:**

::

    proteinmd batch simulate structures/ \
        --pattern "*.pdb" \
        --parallel 8 \
        --output-dir batch_simulations

**Batch analysis:**

::

    proteinmd batch analyze trajectories/ \
        --pattern "*.dcd" \
        --topology reference.pdb

validate - Structure Validation
===============================

Validate protein structures and simulation parameters.

Basic Usage
-----------

::

    proteinmd validate INPUT [OPTIONS]

**Validation Checks:**

``--structure``
    Check structure integrity

``--stereochemistry``
    Validate bond lengths, angles

``--clashes``
    Detect atomic clashes

``--missing-atoms``
    Find missing heavy atoms

``--hydrogens``
    Check hydrogen placement

``--force-field``
    Validate force field assignment

``--waters``
    Check water molecules

``--fix``
    Attempt to fix issues automatically

Examples
--------

**Complete structure validation:**

::

    proteinmd validate protein.pdb \
        --structure --stereochemistry --clashes

**Fix structure issues:**

::

    proteinmd validate protein.pdb --fix \
        --output protein_fixed.pdb

Configuration Management
========================

config - Manage Settings
-------------------------

::

    proteinmd config COMMAND [OPTIONS]

**Commands:**

``show``
    Display current configuration

``set KEY VALUE``
    Set configuration value

``get KEY``
    Get configuration value

``reset``
    Reset to default configuration

``path``
    Show configuration file path

Examples
--------

::

    # Show all settings
    proteinmd config show
    
    # Set default force field
    proteinmd config set default.forcefield amber14sb
    
    # Set number of threads
    proteinmd config set performance.threads 8
    
    # Reset configuration
    proteinmd config reset

Workflow Templates
==================

Creating Custom Workflows
--------------------------

Create a YAML workflow file:

.. code-block:: yaml

    name: "Custom Protein Analysis"
    description: "Custom workflow for protein stability analysis"
    
    parameters:
      temperature: 300.0
      simulation_time: "10ns"
      analysis_stride: 10
    
    steps:
      - name: "energy_minimization"
        command: "minimize"
        parameters:
          steps: 1000
          
      - name: "equilibration"
        command: "equilibrate"
        parameters:
          ensemble: "nvt"
          steps: 100000
          
      - name: "production"
        command: "simulate"
        parameters:
          steps: "{{ simulation_time | to_steps }}"
          
      - name: "analysis"
        command: "analyze"
        parameters:
          analyses: ["rmsd", "rmsf", "secondary_structure"]
          stride: "{{ analysis_stride }}"

Run custom workflow::

    proteinmd workflow my_workflow.yaml protein.pdb

HPC Integration
===============

Job Scheduler Integration
-------------------------

**SLURM example:**

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=proteinmd
    #SBATCH --nodes=1
    #SBATCH --ntasks=8
    #SBATCH --gres=gpu:1
    #SBATCH --time=24:00:00
    
    module load proteinMD
    
    proteinmd simulate protein.pdb \
        --steps 10000000 \
        --output $SLURM_JOB_ID

**PBS example:**

.. code-block:: bash

    #!/bin/bash
    #PBS -N proteinmd
    #PBS -l nodes=1:ppn=8:gpus=1
    #PBS -l walltime=24:00:00
    
    cd $PBS_O_WORKDIR
    proteinmd simulate protein.pdb --steps 10000000

Array Jobs
----------

Run multiple simulations as array job:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --array=1-100
    #SBATCH --job-name=proteinmd_array
    
    INPUT_FILE="structures/protein_${SLURM_ARRAY_TASK_ID}.pdb"
    OUTPUT_DIR="simulation_${SLURM_ARRAY_TASK_ID}"
    
    proteinmd simulate $INPUT_FILE --output $OUTPUT_DIR

Environment Variables
=====================

ProteinMD CLI respects these environment variables:

``PROTEINMD_DATA_DIR``
    Default data directory

``PROTEINMD_CONFIG_FILE``
    Configuration file path

``PROTEINMD_THREADS``
    Number of CPU threads

``PROTEINMD_GPU_DEVICE``
    GPU device ID

``PROTEINMD_LOG_LEVEL``
    Logging level (DEBUG, INFO, WARNING, ERROR)

``PROTEINMD_TEMP_DIR``
    Temporary file directory

Troubleshooting
===============

Common Issues
-------------

**Command not found:**

::

    # Check installation
    pip show proteinMD
    
    # Reinstall if needed
    pip install --force-reinstall proteinMD

**Permission errors:**

::

    # Check file permissions
    ls -la input_file.pdb
    
    # Set output directory
    proteinmd simulate protein.pdb --output ~/simulations/

**Memory issues:**

::

    # Reduce precision for large systems
    proteinmd simulate large_system.pdb \
        --precision single \
        --save-frequency 10000

**GPU errors:**

::

    # Check GPU availability
    proteinmd info --gpu
    
    # Force CPU platform
    proteinmd simulate protein.pdb --platform CPU

Getting Help
============

**Command help:**

::

    proteinmd --help
    proteinmd simulate --help
    proteinmd analyze --help

**System information:**

::

    proteinmd info

**Example files:**

::

    proteinmd examples --list
    proteinmd examples --download tutorial_1

**Documentation:**

- Full API documentation: :doc:`../api/core`
- Tutorials: :doc:`tutorials`
- Troubleshooting: :doc:`../advanced/troubleshooting`
