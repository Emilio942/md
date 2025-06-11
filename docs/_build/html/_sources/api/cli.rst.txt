Command Line Interface
======================

The :mod:`proteinMD.cli` module provides a comprehensive command-line interface for ProteinMD, enabling automated workflows, batch processing, and integration with computational pipelines.

.. currentmodule:: proteinMD.cli

Overview
--------

The CLI module includes:

- **Complete Simulation Workflows**: End-to-end MD simulation pipelines
- **Analysis Pipeline Automation**: Automated post-simulation analysis
- **Batch Processing**: Process multiple structures efficiently
- **Template-based Configuration**: Predefined workflows for common tasks
- **Progress Monitoring**: Real-time simulation progress and reporting
- **Error Handling**: Robust error handling and logging
- **Script Integration**: Return codes for automated workflows

Quick Example
-------------

Basic CLI usage:

.. code-block:: bash

   # Run basic simulation
   proteinmd simulate protein.pdb
   
   # Use workflow template
   proteinmd simulate protein.pdb --template protein_folding
   
   # Custom configuration
   proteinmd simulate protein.pdb --config my_config.json --output-dir results/
   
   # Batch processing
   proteinmd batch-process ./structures/ --template equilibration

Main CLI Class
--------------

.. autoclass:: proteinMD.cli.ProteinMDCLI
   :members:
   :undoc-members:
   :show-inheritance:
   
   **Examples**
   
   Programmatic CLI usage:
   
   .. code-block:: python
   
      from proteinMD.cli import ProteinMDCLI
      
      # Initialize CLI
      cli = ProteinMDCLI(workspace="/path/to/workspace")
      
      # Run simulation
      exit_code = cli.run_simulation(
          input_file="protein.pdb",
          config_file="config.json",
          output_dir="results/"
      )
      
      if exit_code == 0:
          print("Simulation completed successfully!")
   
   Advanced CLI configuration:
   
   .. code-block:: python
   
      # Configure CLI with custom settings
      cli = ProteinMDCLI()
      cli.config.update({
          'simulation': {
              'n_steps': 100000,
              'temperature': 310.0,
              'timestep': 0.002
          },
          'analysis': {
              'rmsd': True,
              'ramachandran': True,
              'secondary_structure': True
          },
          'visualization': {
              'enabled': True,
              'realtime': False
          }
      })
      
      # Run with custom configuration
      cli.run_simulation("protein.pdb", output_dir="custom_results/")

Simulation Commands
-------------------

Run Simulation
~~~~~~~~~~~~~~

.. automethod:: proteinMD.cli.ProteinMDCLI.run_simulation
   
   **Command Line Usage**
   
   .. code-block:: bash
   
      # Basic simulation
      proteinmd simulate protein.pdb
      
      # With configuration file
      proteinmd simulate protein.pdb --config simulation.json
      
      # With workflow template
      proteinmd simulate protein.pdb --template protein_folding
      
      # Specify output directory
      proteinmd simulate protein.pdb --output-dir results/
      
      # Combine options
      proteinmd simulate protein.pdb \
          --template equilibration \
          --config custom.json \
          --output-dir equilibration_results/ \
          --verbose
   
   **Python Usage**
   
   .. code-block:: python
   
      # Run simulation programmatically
      cli = ProteinMDCLI()
      
      result = cli.run_simulation(
          input_file="1ubq.pdb",
          template="protein_folding",
          output_dir="1ubq_results/"
      )
      
      if result == 0:
          print("Simulation successful!")
      else:
          print("Simulation failed!")

Analysis Commands
-----------------

Run Analysis
~~~~~~~~~~~~

.. automethod:: proteinMD.cli.ProteinMDCLI.run_analysis
   
   **Command Line Usage**
   
   .. code-block:: bash
   
      # Analyze trajectory
      proteinmd analyze trajectory.npz protein.pdb
      
      # With custom analysis configuration
      proteinmd analyze trajectory.npz protein.pdb --config analysis.json
      
      # Specify output directory
      proteinmd analyze trajectory.npz protein.pdb --output-dir analysis_results/
   
   **Python Usage**
   
   .. code-block:: python
   
      # Run analysis only
      result = cli.run_analysis(
          trajectory_file="simulation.npz",
          structure_file="protein.pdb",
          output_dir="analysis/"
      )

Batch Processing
----------------

Batch Process
~~~~~~~~~~~~~

.. automethod:: proteinMD.cli.ProteinMDCLI.batch_process
   
   **Command Line Usage**
   
   .. code-block:: bash
   
      # Process all PDB files in directory
      proteinmd batch-process ./structures/
      
      # Use specific pattern
      proteinmd batch-process ./structures/ --pattern "*.pdb"
      
      # Use template and configuration
      proteinmd batch-process ./structures/ \
          --template protein_folding \
          --config batch_config.json \
          --output-dir batch_results/
      
      # Enable parallel processing (future feature)
      proteinmd batch-process ./structures/ --parallel
   
   **Python Usage**
   
   .. code-block:: python
   
      # Batch process multiple structures
      result = cli.batch_process(
          input_dir="protein_family/",
          pattern="*.pdb",
          template="free_energy",
          output_dir="family_analysis/",
          parallel=False
      )
      
      print(f"Batch processing exit code: {result}")

Template Management
-------------------

Create Template
~~~~~~~~~~~~~~~

.. automethod:: proteinMD.cli.ProteinMDCLI.create_template
   
   **Command Line Usage**
   
   .. code-block:: bash
   
      # Create custom template
      proteinmd create-template high_temp \
          "High temperature simulation" \
          config_high_temp.json
   
   **Python Usage**
   
   .. code-block:: python
   
      # Create template programmatically
      cli.create_template(
          name="custom_workflow",
          description="Custom workflow for my research",
          config_file="my_config.json"
      )

List Templates
~~~~~~~~~~~~~~

.. automethod:: proteinMD.cli.ProteinMDCLI.list_templates
   
   **Command Line Usage**
   
   .. code-block:: bash
   
      # List all available templates
      proteinmd list-templates
   
   **Python Usage**
   
   .. code-block:: python
   
      # List templates programmatically
      cli.list_templates()

Utility Commands
----------------

Validate Setup
~~~~~~~~~~~~~~

.. automethod:: proteinMD.cli.ProteinMDCLI.validate_setup
   
   **Command Line Usage**
   
   .. code-block:: bash
   
      # Validate ProteinMD installation
      proteinmd validate-setup
   
   **Example Output**
   
   .. code-block:: text
   
      🔍 ProteinMD Setup Validation
      ========================================
      
      📦 Module Availability:
        ✅ Core modules: Available
        ✅ Force fields: Available
        ✅ Environment: Available
        ✅ Analysis: Available
        ✅ Sampling: Available
        ✅ Visualization: Available
      
      📚 Dependencies:
        ✅ NumPy: 1.21.0
        ✅ Matplotlib: 3.5.0
        ✅ SciPy: 1.8.0
      
      📁 File System:
        📍 Current directory: /home/user/simulations
        📍 Home directory: /home/user
        📍 Config directory: /home/user/.proteinmd
      
      ✅ ProteinMD setup validation completed

Configuration Management
------------------------

Configuration Utilities
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: proteinMD.cli.create_sample_config
   
   **Command Line Usage**
   
   .. code-block:: bash
   
      # Create sample configuration file
      proteinmd sample-config --output my_config.json
   
   **Generated Configuration Structure**
   
   .. code-block:: json
   
      {
        "simulation": {
          "timestep": 0.002,
          "temperature": 300.0,
          "n_steps": 50000,
          "output_frequency": 100,
          "trajectory_output": "trajectory.npz"
        },
        "forcefield": {
          "type": "amber_ff14sb",
          "water_model": "tip3p",
          "cutoff": 1.2
        },
        "environment": {
          "solvent": "explicit",
          "box_padding": 1.0,
          "periodic_boundary": true
        },
        "analysis": {
          "rmsd": true,
          "ramachandran": true,
          "radius_of_gyration": true,
          "secondary_structure": true,
          "hydrogen_bonds": true,
          "output_dir": "analysis_results"
        },
        "visualization": {
          "enabled": true,
          "realtime": false,
          "animation_output": "animation.gif",
          "plots_output": "plots"
        }
      }

Workflow Templates
------------------

Built-in Templates
~~~~~~~~~~~~~~~~~~

ProteinMD provides several built-in workflow templates:

**protein_folding**
   Standard protein folding simulation with comprehensive analysis

   .. code-block:: json
   
      {
        "description": "Standard protein folding simulation",
        "config": {
          "simulation": {
            "n_steps": 50000,
            "temperature": 300.0,
            "timestep": 0.002
          },
          "environment": {"solvent": "explicit"},
          "analysis": {
            "rmsd": true,
            "radius_of_gyration": true,
            "secondary_structure": true
          }
        }
      }

**equilibration**
   System equilibration workflow with shorter timestep

   .. code-block:: json
   
      {
        "description": "System equilibration workflow",
        "config": {
          "simulation": {
            "n_steps": 25000,
            "temperature": 300.0,
            "timestep": 0.001
          },
          "environment": {"solvent": "explicit"},
          "analysis": {
            "rmsd": true,
            "hydrogen_bonds": true
          }
        }
      }

**free_energy**
   Free energy calculation using umbrella sampling

   .. code-block:: json
   
      {
        "description": "Free energy calculation workflow",
        "config": {
          "simulation": {
            "n_steps": 100000,
            "temperature": 300.0,
            "timestep": 0.002
          },
          "sampling": {
            "method": "umbrella_sampling",
            "windows": 20,
            "force_constant": 1000.0
          },
          "analysis": {
            "pmf_calculation": true
          }
        }
      }

**steered_md**
   Steered molecular dynamics for protein unfolding

   .. code-block:: json
   
      {
        "description": "Steered molecular dynamics simulation",
        "config": {
          "simulation": {
            "n_steps": 50000,
            "temperature": 300.0,
            "timestep": 0.002
          },
          "sampling": {
            "method": "steered_md",
            "pulling_velocity": 0.005,
            "spring_constant": 1000.0,
            "coordinate_type": "distance"
          },
          "analysis": {
            "force_curves": true,
            "work_calculation": true
          }
        }
      }

Configuration Hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~

ProteinMD uses a hierarchical configuration system:

1. **Command-line arguments** (highest priority)
2. **Configuration file** specified with ``--config``
3. **Template configuration** specified with ``--template``
4. **Default configuration** (lowest priority)

.. code-block:: python

   # Configuration merging example
   def merge_configs(default, template, config_file, cli_args):
       """Merge configurations with proper priority."""
       result = default.copy()
       
       if template:
           result = merge_dicts(result, template)
       
       if config_file:
           result = merge_dicts(result, config_file)
       
       if cli_args:
           result = merge_dicts(result, cli_args)
       
       return result

Command Line Reference
----------------------

Complete Command Syntax
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   proteinmd [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS] [ARGUMENTS]

**Global Options:**
   - ``-v, --verbose``: Enable verbose output
   - ``-q, --quiet``: Suppress non-error output
   - ``--version``: Show version information
   - ``--help``: Show help message

**Commands:**

``simulate``
   Run molecular dynamics simulation

   .. code-block:: bash
   
      proteinmd simulate INPUT_FILE [OPTIONS]
   
   **Options:**
      - ``--config FILE``: Configuration file (JSON/YAML)
      - ``--template NAME``: Workflow template name
      - ``--output-dir DIR``: Output directory
      - ``--force``: Overwrite existing output

``analyze``
   Analyze trajectory data

   .. code-block:: bash
   
      proteinmd analyze TRAJECTORY_FILE STRUCTURE_FILE [OPTIONS]
   
   **Options:**
      - ``--config FILE``: Analysis configuration file
      - ``--output-dir DIR``: Output directory

``batch-process``
   Process multiple structures

   .. code-block:: bash
   
      proteinmd batch-process INPUT_DIR [OPTIONS]
   
   **Options:**
      - ``--pattern PATTERN``: File pattern (default: ``"*.pdb"``)
      - ``--config FILE``: Configuration file
      - ``--template NAME``: Workflow template
      - ``--output-dir DIR``: Output directory
      - ``--parallel``: Enable parallel processing

``create-template``
   Create custom workflow template

   .. code-block:: bash
   
      proteinmd create-template NAME DESCRIPTION CONFIG_FILE

``list-templates``
   List available workflow templates

   .. code-block:: bash
   
      proteinmd list-templates

``sample-config``
   Create sample configuration file

   .. code-block:: bash
   
      proteinmd sample-config [--output FILE]

``validate-setup``
   Validate ProteinMD installation

   .. code-block:: bash
   
      proteinmd validate-setup

Error Handling and Logging
---------------------------

Return Codes
~~~~~~~~~~~~

The CLI uses standard return codes for script integration:

- **0**: Success
- **1**: General error
- **130**: Interrupted by user (Ctrl+C)

.. code-block:: bash

   # Check exit status in shell scripts
   proteinmd simulate protein.pdb
   if [ $? -eq 0 ]; then
       echo "Simulation successful"
   else
       echo "Simulation failed"
   fi

Logging Configuration
~~~~~~~~~~~~~~~~~~~~~

ProteinMD provides comprehensive logging:

.. code-block:: python

   import logging
   
   # Configure logging levels
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('proteinmd.log'),
           logging.StreamHandler()
       ]
   )

**Log Files:**
   - ``proteinmd_cli.log``: CLI-specific logs
   - ``simulation.log``: Simulation progress and errors
   - ``analysis.log``: Analysis pipeline logs

Common Usage Patterns
---------------------

Complete Workflow Example
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Validate installation
   proteinmd validate-setup
   
   # 2. Create sample configuration
   proteinmd sample-config --output my_config.json
   
   # 3. Edit configuration as needed
   nano my_config.json
   
   # 4. Run simulation
   proteinmd simulate protein.pdb \
       --config my_config.json \
       --output-dir results/ \
       --verbose
   
   # 5. Additional analysis
   proteinmd analyze results/trajectory.npz protein.pdb \
       --output-dir extended_analysis/

High-Throughput Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Process entire protein family
   proteinmd batch-process ./protein_family/ \
       --template free_energy \
       --output-dir family_analysis/ \
       --verbose
   
   # Check batch results
   cat family_analysis/batch_summary.txt

Script Integration
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   #!/bin/bash
   # Automated processing script
   
   for pdb in *.pdb; do
       echo "Processing $pdb..."
       
       proteinmd simulate "$pdb" \
           --template protein_folding \
           --output-dir "${pdb%.pdb}_results/"
       
       if [ $? -eq 0 ]; then
           echo "Successfully processed $pdb"
       else
           echo "Failed to process $pdb" >&2
       fi
   done

Python Integration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import subprocess
   import sys
   from pathlib import Path
   
   def run_proteinmd_simulation(pdb_file, template="protein_folding"):
       """Run ProteinMD simulation with error handling."""
       cmd = [
           "proteinmd", "simulate", str(pdb_file),
           "--template", template,
           "--output-dir", f"{pdb_file.stem}_results/"
       ]
       
       try:
           result = subprocess.run(
               cmd, 
               capture_output=True, 
               text=True, 
               check=True
           )
           print(f"Success: {pdb_file}")
           return True
           
       except subprocess.CalledProcessError as e:
           print(f"Error processing {pdb_file}: {e.stderr}")
           return False
   
   # Process multiple files
   pdb_files = Path(".").glob("*.pdb")
   success_count = 0
   
   for pdb_file in pdb_files:
       if run_proteinmd_simulation(pdb_file):
           success_count += 1
   
   print(f"Successfully processed {success_count} files")

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # For large systems, optimize CLI performance
   proteinmd simulate large_protein.pdb \
       --config optimized.json \
       --output-dir large_system_results/
   
   # Where optimized.json contains:
   {
     "simulation": {
       "output_frequency": 1000,
       "trajectory_compression": true
     },
     "environment": {
       "solvent": "implicit"
     },
     "visualization": {
       "enabled": false,
       "realtime": false
     }
   }

See Also
--------

- :doc:`../user_guide/cli_reference` - Complete CLI reference guide
- :doc:`../user_guide/tutorials` - Step-by-step CLI tutorials
- :doc:`../advanced/troubleshooting` - Troubleshooting CLI issues
- :doc:`core` - Core simulation engine used by CLI
- :doc:`../user_guide/examples` - Real-world CLI usage examples
