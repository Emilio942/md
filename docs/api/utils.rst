==========================
Utilities API Reference
==========================

.. currentmodule:: proteinMD.utils

The utilities module provides essential helper functions, data structures, and tools 
used throughout the ProteinMD package for common computational tasks.

Overview
========

The utilities module includes:

* **Mathematical Utilities** - Linear algebra, statistics, and numerical methods
* **File I/O Operations** - Reading/writing various molecular file formats
* **Data Structures** - Efficient containers for molecular data
* **Performance Tools** - Profiling, timing, and optimization utilities
* **Configuration Management** - Settings and parameter handling

Quick Start
===========

Basic utility usage::

    from proteinMD.utils import (
        Timer, FileManager, VectorMath, 
        ConfigManager, ProgressBar
    )
    
    # Time expensive operations
    with Timer() as t:
        # Perform calculation
        result = expensive_calculation()
    print(f"Calculation took {t.elapsed:.2f} seconds")
    
    # Manage file operations
    fm = FileManager()
    data = fm.load_trajectory('trajectory.dcd')
    fm.backup_file('important_data.txt')
    
    # Vector operations
    distance = VectorMath.distance(atom1_coords, atom2_coords)
    angle = VectorMath.angle(v1, v2, v3)

Mathematical Utilities
======================

VectorMath
----------

.. autoclass:: VectorMath
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Geometric calculations**

.. code-block:: python

    from proteinMD.utils import VectorMath
    import numpy as np
    
    # Define atomic coordinates
    atom1 = np.array([0.0, 0.0, 0.0])
    atom2 = np.array([1.5, 0.0, 0.0])
    atom3 = np.array([1.5, 1.2, 0.0])
    atom4 = np.array([0.0, 1.2, 1.0])
    
    # Calculate distances
    bond_length = VectorMath.distance(atom1, atom2)
    print(f"Bond length: {bond_length:.3f} Å")
    
    # Calculate angles
    bond_angle = VectorMath.angle(atom1, atom2, atom3)
    print(f"Bond angle: {np.degrees(bond_angle):.1f}°")
    
    # Calculate dihedral angles
    dihedral = VectorMath.dihedral(atom1, atom2, atom3, atom4)
    print(f"Dihedral angle: {np.degrees(dihedral):.1f}°")
    
    # Vector operations
    v1 = atom2 - atom1
    v2 = atom3 - atom2
    
    cross_product = VectorMath.cross(v1, v2)
    dot_product = VectorMath.dot(v1, v2)
    
    # Rotation matrices
    rotation_matrix = VectorMath.rotation_matrix(
        axis=[0, 0, 1], 
        angle=np.pi/4
    )
    
    rotated_coords = VectorMath.rotate_coordinates(
        coords=atom1, 
        matrix=rotation_matrix
    )

StatisticalAnalysis
-------------------

.. autoclass:: StatisticalAnalysis
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Statistical analysis of simulation data**

.. code-block:: python

    from proteinMD.utils import StatisticalAnalysis
    import numpy as np
    
    # Generate sample data (e.g., RMSD time series)
    time_series = np.random.normal(2.5, 0.5, 10000)
    
    # Basic statistics
    stats = StatisticalAnalysis()
    
    mean_val = stats.mean(time_series)
    std_val = stats.std(time_series)
    median_val = stats.median(time_series)
    
    print(f"Mean: {mean_val:.3f}")
    print(f"Std Dev: {std_val:.3f}")
    print(f"Median: {median_val:.3f}")
    
    # Statistical tests
    is_normal = stats.test_normality(time_series)
    autocorr_time = stats.autocorrelation_time(time_series)
    
    # Block averaging for error estimation
    block_averages, block_errors = stats.block_average(
        time_series, 
        max_block_size=1000
    )
    
    # Convergence analysis
    convergence_data = stats.analyze_convergence(
        time_series,
        window_size=500,
        stride=100
    )
    
    print(f"Autocorrelation time: {autocorr_time:.1f}")
    print(f"Is normally distributed: {is_normal}")

NumericalMethods
----------------

.. autoclass:: NumericalMethods
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Numerical integration and differentiation**

.. code-block:: python

    from proteinMD.utils import NumericalMethods
    import numpy as np
    
    # Create numerical methods instance
    num = NumericalMethods()
    
    # Example: integrate potential energy curve
    x = np.linspace(0, 10, 1000)
    y = x**2 * np.exp(-x)  # Sample function
    
    # Numerical integration
    integral = num.integrate(x, y, method='simpson')
    print(f"Integral value: {integral:.6f}")
    
    # Numerical differentiation
    derivative = num.differentiate(x, y, method='central')
    
    # Find extrema
    minima, maxima = num.find_extrema(x, y)
    print(f"Found {len(minima)} minima and {len(maxima)} maxima")
    
    # Curve fitting
    def gaussian(x, a, b, c):
        return a * np.exp(-(x - b)**2 / (2 * c**2))
    
    fitted_params, covariance = num.curve_fit(
        gaussian, x, y,
        initial_guess=[1.0, 5.0, 1.0]
    )
    
    # Interpolation
    x_new = np.linspace(0, 10, 2000)
    y_interpolated = num.interpolate(x, y, x_new, method='cubic')

LinearAlgebra
-------------

.. autoclass:: LinearAlgebra
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Matrix operations for molecular analysis**

.. code-block:: python

    from proteinMD.utils import LinearAlgebra
    import numpy as np
    
    # Create linear algebra utilities
    linalg = LinearAlgebra()
    
    # Example: Principal Component Analysis
    # Generate sample coordinate data (N_frames x N_atoms x 3)
    coordinates = np.random.randn(1000, 100, 3)
    
    # Flatten coordinates for PCA
    coord_matrix = coordinates.reshape(1000, -1)
    
    # Perform PCA
    eigenvalues, eigenvectors = linalg.pca(coord_matrix)
    
    # Project onto principal components
    pc_scores = linalg.project_pca(coord_matrix, eigenvectors)
    
    print(f"First 5 eigenvalues: {eigenvalues[:5]}")
    
    # Structural alignment using SVD
    coords1 = np.random.randn(100, 3)
    coords2 = np.random.randn(100, 3)
    
    rotation_matrix, translation = linalg.kabsch_alignment(coords1, coords2)
    
    # Apply alignment
    aligned_coords2 = linalg.apply_transformation(
        coords2, rotation_matrix, translation
    )
    
    # Calculate RMSD after alignment
    rmsd = linalg.rmsd(coords1, aligned_coords2)
    print(f"RMSD after alignment: {rmsd:.3f} Å")

File I/O Operations
===================

FileManager
-----------

.. autoclass:: FileManager
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Advanced file management**

.. code-block:: python

    from proteinMD.utils import FileManager
    import os
    
    # Create file manager
    fm = FileManager(base_directory='/path/to/simulation')
    
    # Setup directory structure
    fm.create_directory_structure({
        'input': ['structures', 'parameters'],
        'output': ['trajectories', 'analysis', 'logs'],
        'backup': []
    })
    
    # File operations with automatic backup
    fm.backup_file('important_results.txt')
    fm.copy_file('template.pdb', 'input/structures/protein.pdb')
    
    # Check file integrity
    checksum = fm.calculate_checksum('trajectory.dcd')
    fm.store_checksum('trajectory.dcd', checksum)
    
    is_valid = fm.verify_checksum('trajectory.dcd')
    print(f"File integrity check: {'PASSED' if is_valid else 'FAILED'}")
    
    # Compress large files
    fm.compress_file('large_trajectory.dcd', method='gzip')
    
    # Find files by pattern
    pdb_files = fm.find_files('*.pdb', recursive=True)
    recent_files = fm.find_recent_files(days=7)
    
    # Clean up temporary files
    fm.cleanup_temp_files()

FormatConverter
---------------

.. autoclass:: FormatConverter
    :members:
    :undoc-members:
    :show-inheritance:

**Example: File format conversions**

.. code-block:: python

    from proteinMD.utils import FormatConverter
    
    # Create format converter
    converter = FormatConverter()
    
    # Convert between structure formats
    converter.convert_structure(
        input_file='protein.pdb',
        output_file='protein.mol2',
        input_format='pdb',
        output_format='mol2'
    )
    
    # Convert trajectory formats
    converter.convert_trajectory(
        input_trajectory='trajectory.dcd',
        input_topology='topology.pdb',
        output_trajectory='trajectory.xtc',
        output_format='xtc',
        frame_range=(100, 500),
        stride=2
    )
    
    # Batch conversion
    converter.batch_convert(
        input_directory='input_pdbs/',
        output_directory='output_mol2/',
        input_format='pdb',
        output_format='mol2',
        pattern='*.pdb'
    )
    
    # Custom conversion with filtering
    converter.convert_with_filter(
        input_file='complex.pdb',
        output_file='protein_only.pdb',
        filter_selection='protein',
        remove_hydrogens=True,
        remove_water=True
    )

DataReader
----------

.. autoclass:: DataReader
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Reading various data formats**

.. code-block:: python

    from proteinMD.utils import DataReader
    
    # Create data reader
    reader = DataReader()
    
    # Read energy files
    energy_data = reader.read_energy_file(
        'energy.txt',
        columns=['time', 'potential', 'kinetic', 'total'],
        skip_header=1
    )
    
    # Read XVG files (GROMACS format)
    xvg_data = reader.read_xvg('rmsd.xvg')
    
    # Read CSV data with automatic type detection
    csv_data = reader.read_csv(
        'analysis_results.csv',
        auto_detect_types=True,
        parse_dates=['timestamp']
    )
    
    # Read binary data
    binary_data = reader.read_binary(
        'coordinates.bin',
        data_type='float32',
        shape=(1000, 300, 3)
    )
    
    # Read HDF5 files
    hdf5_data = reader.read_hdf5(
        'simulation_data.h5',
        datasets=['coordinates', 'velocities', 'forces']
    )

Data Structures
===============

AtomicData
----------

.. autoclass:: AtomicData
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Efficient atomic data storage**

.. code-block:: python

    from proteinMD.utils import AtomicData
    import numpy as np
    
    # Create atomic data container
    atoms = AtomicData()
    
    # Add atomic information
    atoms.add_atom(
        element='C',
        coordinates=[0.0, 0.0, 0.0],
        charge=0.0,
        mass=12.01,
        residue_name='ALA',
        residue_id=1,
        atom_name='CA'
    )
    
    # Bulk operations
    coordinates = np.random.randn(1000, 3)
    elements = ['C'] * 500 + ['N'] * 300 + ['O'] * 200
    
    atoms.add_atoms_bulk(
        coordinates=coordinates,
        elements=elements,
        masses=atoms.get_standard_masses(elements)
    )
    
    # Query operations
    carbon_atoms = atoms.select_by_element('C')
    backbone_atoms = atoms.select_by_name(['N', 'CA', 'C', 'O'])
    residue_atoms = atoms.select_by_residue(1)
    
    # Spatial queries
    nearby_atoms = atoms.find_within_distance(
        center=[0.0, 0.0, 0.0],
        radius=5.0
    )
    
    # Update coordinates
    atoms.update_coordinates(new_coordinates)
    
    # Export data
    atoms.to_dict()
    atoms.to_dataframe()

MolecularGraph
--------------

.. autoclass:: MolecularGraph
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Graph-based molecular representation**

.. code-block:: python

    from proteinMD.utils import MolecularGraph
    from proteinMD.structure import Protein
    
    # Create molecular graph
    protein = Protein('protein.pdb')
    graph = MolecularGraph.from_structure(protein)
    
    # Add connectivity
    graph.add_bonds_from_distance(cutoff=1.8)
    graph.add_bonds_from_topology()
    
    # Graph analysis
    shortest_path = graph.shortest_path(atom1=100, atom2=200)
    connected_components = graph.connected_components()
    
    # Topological analysis
    degree_centrality = graph.degree_centrality()
    betweenness_centrality = graph.betweenness_centrality()
    
    # Find structural motifs
    rings = graph.find_rings()
    bridges = graph.find_bridges()
    
    # Subgraph extraction
    active_site_atoms = [45, 67, 89, 123, 156]
    active_site_graph = graph.subgraph(active_site_atoms)
    
    # Export to standard formats
    graph.to_networkx()
    graph.to_adjacency_matrix()

ParameterSet
------------

.. autoclass:: ParameterSet
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Parameter management**

.. code-block:: python

    from proteinMD.utils import ParameterSet
    
    # Create parameter set
    params = ParameterSet()
    
    # Add parameters with validation
    params.add_parameter(
        'temperature',
        value=300.0,
        units='K',
        valid_range=(0, 1000),
        description='Simulation temperature'
    )
    
    params.add_parameter(
        'timestep',
        value=2.0,
        units='fs',
        valid_range=(0.1, 10.0),
        description='Integration timestep'
    )
    
    # Parameter groups
    params.create_group('simulation', ['temperature', 'timestep'])
    params.create_group('output', ['save_frequency', 'output_format'])
    
    # Validation
    is_valid, errors = params.validate()
    if not is_valid:
        print("Parameter errors:", errors)
    
    # Load/save parameters
    params.save_to_file('parameters.yaml')
    params.load_from_file('parameters.yaml')
    
    # Parameter sweeps
    sweep_params = params.create_sweep({
        'temperature': [300, 310, 320],
        'timestep': [1.0, 2.0]
    })

Performance Tools
=================

Timer
-----

.. autoclass:: Timer
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Performance timing and profiling**

.. code-block:: python

    from proteinMD.utils import Timer
    import time
    
    # Basic timing
    with Timer() as t:
        time.sleep(1.0)  # Simulate work
    print(f"Operation took {t.elapsed:.3f} seconds")
    
    # Named timers
    timer = Timer()
    
    timer.start('initialization')
    time.sleep(0.5)
    timer.stop('initialization')
    
    timer.start('calculation')
    time.sleep(1.5)
    timer.stop('calculation')
    
    # Print timing report
    timer.report()
    
    # Decorator usage
    @timer.time_function
    def expensive_function():
        time.sleep(2.0)
        return "result"
    
    result = expensive_function()
    timer.report()

Profiler
--------

.. autoclass:: Profiler
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Code profiling and optimization**

.. code-block:: python

    from proteinMD.utils import Profiler
    import numpy as np
    
    # Create profiler
    profiler = Profiler()
    
    # Profile function execution
    @profiler.profile_function
    def matrix_multiplication(n=1000):
        a = np.random.randn(n, n)
        b = np.random.randn(n, n)
        return np.dot(a, b)
    
    # Run profiled function
    result = matrix_multiplication(500)
    
    # Memory profiling
    @profiler.profile_memory
    def memory_intensive_function():
        large_array = np.zeros((10000, 10000))
        return large_array.sum()
    
    memory_intensive_function()
    
    # Generate profiling report
    profiler.generate_report('profiling_report.html')
    
    # Line-by-line profiling
    profiler.line_profile(matrix_multiplication)

MemoryManager
-------------

.. autoclass:: MemoryManager
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Memory usage optimization**

.. code-block:: python

    from proteinMD.utils import MemoryManager
    import numpy as np
    
    # Create memory manager
    mem = MemoryManager()
    
    # Monitor memory usage
    mem.start_monitoring()
    
    # Allocate memory with tracking
    large_array = mem.allocate_array(
        shape=(10000, 10000),
        dtype=np.float64,
        name='trajectory_data'
    )
    
    # Memory pool for frequent allocations
    mem.create_pool('temp_arrays', size_mb=1024)
    
    temp_array = mem.allocate_from_pool(
        'temp_arrays',
        shape=(1000, 1000),
        dtype=np.float32
    )
    
    # Memory usage report
    usage_report = mem.get_usage_report()
    print(f"Total memory used: {usage_report['total_mb']:.1f} MB")
    
    # Cleanup
    mem.free_array('trajectory_data')
    mem.clear_pool('temp_arrays')
    mem.stop_monitoring()

Configuration Management
========================

ConfigManager
-------------

.. autoclass:: ConfigManager
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Configuration management**

.. code-block:: python

    from proteinMD.utils import ConfigManager
    
    # Create configuration manager
    config = ConfigManager()
    
    # Load configuration from file
    config.load_config('simulation_config.yaml')
    
    # Access configuration values
    temperature = config.get('simulation.temperature', default=300.0)
    output_dir = config.get('output.directory', default='./output')
    
    # Set configuration values
    config.set('simulation.steps', 1000000)
    config.set('forcefield.type', 'amber14sb')
    
    # Nested configuration
    config.set_nested('analysis.rmsd.reference_frame', 0)
    config.set_nested('analysis.rmsd.selection', 'backbone')
    
    # Configuration validation
    schema = {
        'simulation': {
            'temperature': {'type': 'float', 'min': 0, 'max': 1000},
            'steps': {'type': 'int', 'min': 1}
        }
    }
    
    config.set_schema(schema)
    is_valid = config.validate()
    
    # Save configuration
    config.save_config('updated_config.yaml')
    
    # Environment variable support
    config.load_from_env(prefix='PROTEINMD_')

ProgressBar
-----------

.. autoclass:: ProgressBar
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Progress tracking**

.. code-block:: python

    from proteinMD.utils import ProgressBar
    import time
    
    # Simple progress bar
    with ProgressBar(total=100, description="Processing") as pbar:
        for i in range(100):
            time.sleep(0.01)  # Simulate work
            pbar.update(1)
    
    # Multi-level progress tracking
    main_pbar = ProgressBar(total=5, description="Main task")
    
    for i in range(5):
        main_pbar.set_description(f"Subtask {i+1}")
        
        with ProgressBar(total=20, description="  Subtask") as sub_pbar:
            for j in range(20):
                time.sleep(0.01)
                sub_pbar.update(1)
        
        main_pbar.update(1)
    
    # Custom progress callback
    def progress_callback(current, total, message=""):
        percent = 100 * current / total
        print(f"\r{message} {percent:.1f}% complete", end="")
    
    pbar = ProgressBar(callback=progress_callback)
    pbar.track_range(range(100), description="Custom tracking")

Constants and Utilities
=======================

.. autodata:: ATOMIC_MASSES
.. autodata:: ATOMIC_RADII
.. autodata:: PHYSICAL_CONSTANTS
.. autodata:: CONVERSION_FACTORS

Logging Utilities
=================

Logger
------

.. autoclass:: Logger
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Comprehensive logging system**

.. code-block:: python

    from proteinMD.utils import Logger
    
    # Create logger
    logger = Logger('ProteinMD', level='INFO')
    
    # Add file handler
    logger.add_file_handler('simulation.log')
    logger.add_file_handler('errors.log', level='ERROR')
    
    # Different log levels
    logger.debug("Debug information")
    logger.info("Simulation started")
    logger.warning("High temperature detected")
    logger.error("Force field parameter missing")
    logger.critical("Simulation failed")
    
    # Context logging
    with logger.context("Energy minimization"):
        logger.info("Starting minimization")
        logger.info("Convergence reached")
    
    # Performance logging
    logger.log_performance("calculation", execution_time=1.5)
    
    # Structured logging
    logger.log_structured({
        'event': 'simulation_step',
        'step': 1000,
        'energy': -12345.67,
        'temperature': 298.15
    })

See Also
========

* :doc:`core` - Core simulation classes that use these utilities
* :doc:`../user_guide/tutorials` - Tutorials showing utility usage
* :doc:`../developer/testing` - Testing utilities and frameworks
* :doc:`../advanced/performance` - Performance optimization guides

References
==========

1. Oliphant, T.E. A guide to NumPy (Trelgol Publishing, 2006)
2. McKinney, W. pandas: a foundational Python library. Python for High Performance Scientific Computing 14, 1-9 (2011)
3. Virtanen, P. et al. SciPy 1.0: fundamental algorithms for scientific computing. Nat. Methods 17, 261-272 (2020)
