============
Installation
============

This guide covers installation of ProteinMD for different use cases and operating systems.

Quick Installation
==================

For most users, install via pip::

    pip install proteinMD

Or from conda-forge::

    conda install -c conda-forge proteinMD

Requirements
============

System Requirements
-------------------

**Operating Systems**
  - Linux (Ubuntu 18.04+, CentOS 7+, RHEL 7+)
  - macOS (10.14+)
  - Windows 10/11 (with WSL2 recommended)

**Hardware Requirements**
  - CPU: Multi-core processor (8+ cores recommended)
  - RAM: 8GB minimum, 16GB+ recommended
  - GPU: CUDA-compatible GPU optional but recommended
  - Storage: 10GB+ free space

Python Requirements
-------------------

**Python Version**
  - Python 3.8 or higher
  - Python 3.9-3.11 recommended

**Core Dependencies**
  - NumPy >= 1.19.0
  - SciPy >= 1.6.0
  - MDAnalysis >= 2.0.0
  - OpenMM >= 7.6.0 (optional, for GPU acceleration)
  - matplotlib >= 3.3.0
  - pandas >= 1.2.0

Installation Methods
====================

Method 1: pip (Recommended)
----------------------------

Install the latest stable release::

    pip install proteinMD

Install with optional dependencies::

    # For GPU acceleration
    pip install proteinMD[gpu]
    
    # For visualization
    pip install proteinMD[viz]
    
    # For development
    pip install proteinMD[dev]
    
    # Install everything
    pip install proteinMD[all]

Method 2: conda
---------------

Using conda-forge::

    conda install -c conda-forge proteinMD

Create a dedicated environment::

    conda create -n proteinmd python=3.9
    conda activate proteinmd
    conda install -c conda-forge proteinMD

Method 3: Development Installation
----------------------------------

For development or the latest features::

    git clone https://github.com/proteinmd/proteinMD.git
    cd proteinMD
    pip install -e .

Install development dependencies::

    pip install -e .[dev]

Method 4: Docker
----------------

Use the official Docker image::

    docker pull proteinmd/proteinmd:latest
    docker run -it proteinmd/proteinmd:latest

Create a container with volume mounting::

    docker run -it -v $(pwd):/workspace proteinmd/proteinmd:latest

Optional Dependencies
=====================

GPU Acceleration
----------------

For NVIDIA GPUs with CUDA support::

    # Install CUDA toolkit (system-wide)
    # Follow NVIDIA CUDA installation guide
    
    # Install OpenMM with CUDA
    conda install -c conda-forge openmm cudatoolkit

For AMD GPUs with OpenCL::

    conda install -c conda-forge openmm ocl-icd-system

Visualization Tools
-------------------

For 3D molecular visualization::

    # PyMOL (recommended)
    conda install -c conda-forge pymol-open-source
    
    # VMD (manual installation)
    # Download from https://www.ks.uiuc.edu/Research/vmd/
    
    # NGL viewer for Jupyter
    pip install nglview
    jupyter-nbextension enable nglview --py --sys-prefix

Additional Analysis Tools
-------------------------

For enhanced analysis capabilities::

    # MDTraj for trajectory analysis
    conda install -c conda-forge mdtraj
    
    # ProDy for protein dynamics
    pip install ProDy
    
    # NetworkX for graph analysis
    pip install networkx

Verification
============

Test Installation
-----------------

Verify your installation::

    python -c "import proteinMD; print(proteinMD.__version__)"

Run basic tests::

    python -c "
    from proteinMD.structure import PDBParser
    from proteinMD.core import Simulation
    print('ProteinMD installation successful!')
    "

Performance Test
----------------

Test computational performance::

    python -m proteinMD.utils.benchmark

This will run basic benchmarks and report performance metrics.

GPU Test
--------

If you installed GPU support::

    python -c "
    from proteinMD.core import check_gpu_support
    print(f'GPU support: {check_gpu_support()}')
    "

Troubleshooting
===============

Common Issues
-------------

**Import Error: No module named 'proteinMD'**

Solution::

    # Ensure pip installed to correct Python
    python -m pip install proteinMD
    
    # Or check Python path
    python -c "import sys; print(sys.path)"

**CUDA/GPU Issues**

Solution::

    # Check CUDA installation
    nvidia-smi
    
    # Check OpenMM CUDA platforms
    python -c "
    import openmm
    print([platform.getName() for platform in 
           [openmm.Platform.getPlatform(i) 
            for i in range(openmm.Platform.getNumPlatforms())]])
    "

**Memory Issues During Installation**

Solution::

    # Install without cache
    pip install --no-cache-dir proteinMD
    
    # Or increase swap space (Linux)
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile

**Conflicting Dependencies**

Solution::

    # Create clean environment
    conda create -n proteinmd-clean python=3.9
    conda activate proteinmd-clean
    pip install proteinMD

Platform-Specific Notes
=======================

Linux
-----

**Ubuntu/Debian**::

    # Install system dependencies
    sudo apt-get update
    sudo apt-get install build-essential python3-dev
    
    # For visualization
    sudo apt-get install pymol

**CentOS/RHEL**::

    # Install development tools
    sudo yum groupinstall "Development Tools"
    sudo yum install python3-devel
    
    # Enable EPEL for additional packages
    sudo yum install epel-release

macOS
-----

**Using Homebrew**::

    # Install Python and dependencies
    brew install python
    brew install openmm
    
    # Install ProteinMD
    pip3 install proteinMD

**Using MacPorts**::

    sudo port install py39-numpy py39-scipy
    pip install proteinMD

Windows
-------

**WSL2 (Recommended)**::

    # Install WSL2 with Ubuntu
    wsl --install -d Ubuntu
    
    # Follow Linux installation in WSL

**Native Windows**::

    # Install Visual Studio Build Tools
    # Install Python from python.org
    pip install proteinMD

HPC Clusters
============

Module Systems
--------------

Many HPC systems use environment modules::

    # Load required modules
    module load python/3.9
    module load cuda/11.2
    module load openmpi/4.1
    
    # Install in user space
    pip install --user proteinMD

Singularity/Apptainer
---------------------

Use containers on HPC systems::

    # Pull container
    singularity pull proteinmd.sif docker://proteinmd/proteinmd:latest
    
    # Run with container
    singularity exec proteinmd.sif python your_script.py

Job Submission Examples
-----------------------

**SLURM example**::

    #!/bin/bash
    #SBATCH --job-name=proteinmd
    #SBATCH --nodes=1
    #SBATCH --ntasks=8
    #SBATCH --gres=gpu:1
    #SBATCH --time=24:00:00
    
    module load python/3.9 cuda/11.2
    python md_simulation.py

**PBS example**::

    #!/bin/bash
    #PBS -N proteinmd
    #PBS -l nodes=1:ppn=8:gpus=1
    #PBS -l walltime=24:00:00
    
    cd $PBS_O_WORKDIR
    module load python cuda
    python md_simulation.py

Virtual Environments
====================

Best Practices
--------------

Always use virtual environments::

    # Create environment
    python -m venv proteinmd-env
    
    # Activate (Linux/macOS)
    source proteinmd-env/bin/activate
    
    # Activate (Windows)
    proteinmd-env\Scripts\activate
    
    # Install ProteinMD
    pip install proteinMD

Conda Environments
------------------

Recommended for scientific computing::

    # Create environment with dependencies
    conda create -n proteinmd python=3.9 numpy scipy matplotlib
    conda activate proteinmd
    pip install proteinMD

Environment Export/Import
-------------------------

Share environments::

    # Export environment
    conda env export > proteinmd-environment.yml
    
    # Import environment
    conda env create -f proteinmd-environment.yml

Configuration
=============

Environment Variables
--------------------

Set ProteinMD configuration::

    export PROTEINMD_DATA_DIR=/path/to/data
    export PROTEINMD_TEMP_DIR=/tmp/proteinmd
    export PROTEINMD_LOG_LEVEL=INFO

Configuration File
------------------

Create ``~/.proteinmd/config.yaml``::

    general:
      data_directory: ~/proteinmd_data
      temp_directory: /tmp/proteinmd
      log_level: INFO
      
    performance:
      num_threads: 8
      use_gpu: true
      
    visualization:
      default_backend: pymol
      image_quality: high

Next Steps
==========

After successful installation:

1. :doc:`quick_start` - Run your first simulation
2. :doc:`tutorials` - Follow step-by-step tutorials  
3. :doc:`../api/core` - Explore the API reference
4. :doc:`cli_reference` - Learn the command-line interface

Getting Help
============

If you encounter issues:

1. Check the :doc:`../advanced/troubleshooting` guide
2. Search existing issues on GitHub
3. Post questions in the community forum
4. Report bugs with detailed error messages

Version History
===============

See :doc:`../about/changelog` for detailed version history and upgrade notes.
