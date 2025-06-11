"""
Test configuration and shared fixtures for proteinMD tests.

This module provides common fixtures, utilities, and configuration
for all tests in the proteinMD test suite.

Task 10.1: Umfassende Unit Tests 🚀 - Comprehensive Test Infrastructure
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import logging
import sys
import warnings
import time
import psutil
import os

# Add proteinMD to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ============================================================================
# PERFORMANCE THRESHOLDS
# ============================================================================

# Performance thresholds for regression testing
PERFORMANCE_THRESHOLDS = {
    'force_calculation_per_particle_us': 100.0,  # microseconds per particle
    'trajectory_step_ms': 50.0,  # milliseconds per simulation step
    'memory_per_particle_mb': 1.0,  # MB per particle
    'neighbor_list_update_ms': 10.0,  # milliseconds
    'energy_calculation_ms': 20.0,  # milliseconds
}

# ============================================================================
# COMMON TEST DATA
# ============================================================================

@pytest.fixture
def small_protein_coordinates():
    """Small protein coordinates for testing (10 atoms)."""
    return np.array([
        [0.0, 0.0, 0.0],    # N
        [0.1, 0.0, 0.0],    # CA
        [0.2, 0.1, 0.0],    # C
        [0.3, 0.1, 0.1],    # O
        [0.05, -0.1, 0.05], # H
        [0.15, -0.05, 0.0], # HA
        [0.25, 0.05, -0.05],# CB
        [0.35, 0.0, 0.0],   # CG
        [0.4, 0.1, 0.1],    # CD
        [0.5, 0.05, 0.05]   # CE
    ])

@pytest.fixture
def small_protein_masses():
    """Masses for small protein (10 atoms)."""
    return np.array([14.01, 12.01, 12.01, 16.00, 1.008, 
                     1.008, 12.01, 12.01, 12.01, 12.01])

@pytest.fixture
def small_protein_charges():
    """Charges for small protein (10 atoms)."""
    return np.array([-0.3, 0.1, 0.7, -0.6, 0.3, 
                     0.1, -0.1, 0.0, 0.0, -0.1])

@pytest.fixture
def medium_protein_coordinates():
    """Medium protein coordinates for testing (100 atoms)."""
    np.random.seed(42)  # For reproducible test data
    return np.random.uniform(-2.0, 2.0, (100, 3))

@pytest.fixture
def medium_protein_masses():
    """Masses for medium protein (100 atoms)."""
    np.random.seed(42)
    # Realistic mass distribution for protein atoms
    masses = np.random.choice([1.008, 12.01, 14.01, 16.00, 32.06], 100, 
                             p=[0.5, 0.3, 0.1, 0.09, 0.01])
    return masses

@pytest.fixture
def medium_protein_charges():
    """Charges for medium protein (100 atoms)."""
    np.random.seed(42)
    charges = np.random.uniform(-0.8, 0.8, 100)
    # Ensure overall neutrality
    charges[-1] = -np.sum(charges[:-1])
    return charges

@pytest.fixture
def large_protein_coordinates():
    """Large protein coordinates for performance testing (1000 atoms)."""
    np.random.seed(42)
    return np.random.uniform(-5.0, 5.0, (1000, 3))

@pytest.fixture
def large_protein_masses():
    """Masses for large protein (1000 atoms)."""
    np.random.seed(42)
    masses = np.random.choice([1.008, 12.01, 14.01, 16.00, 32.06], 1000,
                             p=[0.5, 0.3, 0.1, 0.09, 0.01])
    return masses

@pytest.fixture
def large_protein_charges():
    """Charges for large protein (1000 atoms)."""
    np.random.seed(42)
    charges = np.random.uniform(-0.8, 0.8, 1000)
    charges[-1] = -np.sum(charges[:-1])
    return charges

# ============================================================================
# TRAJECTORY DATA
# ============================================================================

@pytest.fixture
def sample_trajectory():
    """Sample trajectory data for testing."""
    np.random.seed(42)
    n_frames = 100
    n_atoms = 50
    trajectory = []
    
    for i in range(n_frames):
        # Simulate small random movements
        positions = np.random.uniform(-1.0, 1.0, (n_atoms, 3))
        positions += np.random.normal(0, 0.01, (n_atoms, 3))  # Small perturbations
        trajectory.append(positions.copy())
    
    return trajectory

@pytest.fixture
def trajectory_times():
    """Time points for trajectory data."""
    return np.linspace(0, 10.0, 100)  # 100 frames over 10 time units

# ============================================================================
# FILE SYSTEM FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_pdb_content():
    """Sample PDB file content for testing."""
    return '''HEADER    TRANSFERASE/DNA                         01-JUL-93   1ABC              
ATOM      1  N   ALA A   1      20.154  16.967  12.784  1.00 25.00           N  
ATOM      2  CA  ALA A   1      21.618  17.090  12.703  1.00 25.00           C  
ATOM      3  C   ALA A   1      22.219  15.718  12.897  1.00 25.00           C  
ATOM      4  O   ALA A   1      21.622  14.692  13.201  1.00 25.00           O  
ATOM      5  CB  ALA A   1      22.067  17.771  11.415  1.00 25.00           C  
ATOM      6  N   VAL A   2      23.519  15.690  12.730  1.00 25.00           N  
ATOM      7  CA  VAL A   2      24.234  14.449  12.895  1.00 25.00           C  
ATOM      8  C   VAL A   2      25.667  14.719  13.326  1.00 25.00           C  
ATOM      9  O   VAL A   2      26.324  15.643  12.912  1.00 25.00           O  
ATOM     10  CB  VAL A   2      24.268  13.617  11.604  1.00 25.00           C  
END
'''

@pytest.fixture
def sample_pdb_file(temp_dir, sample_pdb_content):
    """Create a sample PDB file for testing."""
    pdb_file = temp_dir / "sample.pdb"
    pdb_file.write_text(sample_pdb_content)
    return pdb_file

# ============================================================================
# MOCK OBJECTS
# ============================================================================

@pytest.fixture
def mock_simulation():
    """Mock simulation object for testing."""
    sim = Mock()
    sim.positions = np.random.uniform(-1.0, 1.0, (50, 3))
    sim.masses = np.ones(50) * 12.0
    sim.charges = np.random.uniform(-0.5, 0.5, 50)
    sim.box_dimensions = np.array([10.0, 10.0, 10.0])
    sim.trajectory = []
    sim.current_time = 0.0
    sim.time_step = 0.002
    return sim

@pytest.fixture
def mock_force_field():
    """Mock force field for testing."""
    ff = Mock()
    ff.calculate_forces = MagicMock(return_value=np.zeros((50, 3)))
    ff.calculate_energy = MagicMock(return_value=100.0)
    ff.validate_parameters = MagicMock(return_value=True)
    return ff

# ============================================================================
# PERFORMANCE TESTING UTILITIES
# ============================================================================

class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time

@pytest.fixture
def performance_timer():
    """Performance timer fixture."""
    return PerformanceTimer

# ============================================================================
# TESTING UTILITIES
# ============================================================================

def assert_valid_coordinates(coordinates, min_val=-100.0, max_val=100.0):
    """Assert that coordinates are valid (finite and in reasonable range)."""
    assert np.all(np.isfinite(coordinates)), "Coordinates contain non-finite values"
    assert np.all(coordinates >= min_val), f"Coordinates below minimum {min_val}"
    assert np.all(coordinates <= max_val), f"Coordinates above maximum {max_val}"

def assert_valid_forces(forces, max_force=1000.0):
    """Assert that forces are valid (finite and not too large)."""
    assert np.all(np.isfinite(forces)), "Forces contain non-finite values"
    force_magnitudes = np.linalg.norm(forces, axis=1)
    assert np.all(force_magnitudes <= max_force), f"Forces exceed maximum {max_force}"

def assert_energy_conservation(energies, tolerance=0.05):
    """Assert that energy is approximately conserved (within tolerance)."""
    if len(energies) < 2:
        return
    
    initial_energy = energies[0]
    energy_drift = np.abs(energies - initial_energy) / np.abs(initial_energy)
    max_drift = np.max(energy_drift)
    
    assert max_drift <= tolerance, f"Energy drift {max_drift:.3f} exceeds tolerance {tolerance}"

def assert_performance_threshold(duration_ms, threshold_key):
    """Assert that performance meets threshold requirements."""
    threshold = PERFORMANCE_THRESHOLDS.get(threshold_key)
    if threshold is not None:
        assert duration_ms <= threshold, \
            f"Performance {duration_ms:.2f}ms exceeds threshold {threshold}ms for {threshold_key}"

# ============================================================================
# PARAMETRIZED TEST DATA
# ============================================================================

# Common test parameters for different system sizes
SYSTEM_SIZES = [
    pytest.param(10, id="small_10_atoms"),
    pytest.param(50, id="medium_50_atoms"),
    pytest.param(100, id="large_100_atoms"),
]

PERFORMANCE_SYSTEM_SIZES = [
    pytest.param(100, id="performance_100_atoms"),
    pytest.param(500, id="performance_500_atoms"),
    pytest.param(1000, id="performance_1000_atoms", marks=pytest.mark.slow),
]

# Force field types for testing
FORCE_FIELD_TYPES = [
    pytest.param("amber", id="amber_ff14sb"),
    pytest.param("charmm", id="charmm36"),
    pytest.param("custom", id="custom_ff"),
]

# Integration algorithms for testing
INTEGRATION_ALGORITHMS = [
    pytest.param("verlet", id="velocity_verlet"),
    pytest.param("leapfrog", id="leapfrog"),
    pytest.param("rk4", id="runge_kutta_4th"),
]

# Analysis methods for testing
ANALYSIS_METHODS = [
    pytest.param("rmsd", id="rmsd_analysis"),
    pytest.param("rg", id="radius_of_gyration"),
    pytest.param("ramachandran", id="ramachandran_plot"),
    pytest.param("secondary_structure", id="secondary_structure"),
    pytest.param("hydrogen_bonds", id="hydrogen_bonds"),
]

# ============================================================================
# TEST MARKERS AND CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure custom test markers."""
    config.addinivalue_line("markers", "slow: mark test as slow (requires longer execution time)")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance benchmark")
    config.addinivalue_line("markers", "memory: mark test as memory usage test")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "coverage: mark test for coverage analysis")

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark slow tests
        if any(keyword in item.name.lower() for keyword in ["slow", "benchmark", "performance", "large"]):
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if any(keyword in item.name.lower() for keyword in ["integration", "end_to_end", "workflow"]):
            item.add_marker(pytest.mark.integration)
        
        # Mark memory tests
        if any(keyword in item.name.lower() for keyword in ["memory", "leak", "allocation"]):
            item.add_marker(pytest.mark.memory)
        
        # Mark unit tests by default
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)

# ============================================================================
# PERFORMANCE AND COVERAGE MONITORING
# ============================================================================

@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.metrics = {}
        
        def start(self):
            self.start_time = time.perf_counter()
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        def stop(self):
            if self.start_time is None:
                self.start()
            
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            self.metrics = {
                'duration_ms': (end_time - self.start_time) * 1000,
                'memory_delta_mb': end_memory - self.start_memory,
                'peak_memory_mb': end_memory
            }
            return self.metrics
        
        def __enter__(self):
            self.start()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stop()
    
    return PerformanceMonitor()

@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    yield initial_memory
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Log warning if memory usage increased significantly
    if memory_increase > 50:  # 50MB increase
        logger.warning(f"Test caused significant memory increase: {memory_increase:.1f} MB")

# ============================================================================
# ADVANCED MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_protein_structure():
    """Create a comprehensive mock protein structure."""
    protein = Mock()
    protein.name = "test_protein"
    protein.num_atoms = 100
    protein.num_residues = 10
    
    # Mock residues with realistic structure
    residues = []
    aa_types = ["ALA", "VAL", "GLY", "SER", "LEU", "ILE", "PRO", "PHE", "TYR", "TRP"]
    
    for i in range(10):
        residue = Mock()
        residue.name = aa_types[i]
        residue.index = i
        residue.atoms = []
        
        # Define realistic atom sets for different amino acids
        atom_sets = {
            "ALA": ["N", "CA", "C", "O", "CB", "H", "HA", "HB1", "HB2", "HB3"],
            "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "H", "HA", "HB"],
            "GLY": ["N", "CA", "C", "O", "H", "HA2", "HA3"],
            "SER": ["N", "CA", "C", "O", "CB", "OG", "H", "HA", "HB2", "HB3", "HG"],
            "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "H", "HA"],
            "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "H", "HA"],
            "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "HA", "HB2", "HB3"],
            "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
            "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
            "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2"]
        }
        
        atom_names = atom_sets.get(residue.name, ["N", "CA", "C", "O", "CB"])
        
        for j, atom_name in enumerate(atom_names):
            atom = Mock()
            atom.name = atom_name
            atom.index = len(residue.atoms) + i * 15
            atom.element = atom_name[0] if atom_name[0] in ['C', 'N', 'O', 'S'] else 'H'
            atom.position = np.random.randn(3) + np.array([i*2.0, 0, 0])  # Spread along x-axis
            atom.mass = {'C': 12.01, 'N': 14.01, 'O': 16.00, 'S': 32.06, 'H': 1.008}[atom.element]
            atom.charge = np.random.uniform(-0.5, 0.5)
            residue.atoms.append(atom)
        
        residues.append(residue)
    
    protein.residues = residues
    
    # Add positions and masses arrays for analysis compatibility
    all_positions = []
    all_masses = []
    for residue in residues:
        for atom in residue.atoms:
            all_positions.append(atom.position)
            all_masses.append(atom.mass)
    
    protein.positions = np.array(all_positions)
    protein.masses = np.array(all_masses)
    
    return protein

@pytest.fixture  
def mock_trajectory():
    """Create a realistic mock trajectory with correlated motion."""
    n_frames = 100
    n_atoms = 100
    
    # Generate realistic protein-like trajectory with breathing motion
    base_positions = np.random.randn(n_atoms, 3) * 2.0 + 5.0
    positions = np.zeros((n_frames, n_atoms, 3))
    velocities = np.zeros((n_frames, n_atoms, 3))
    forces = np.zeros((n_frames, n_atoms, 3))
    
    # Add breathing and random motion
    for frame in range(n_frames):
        # Breathing motion (expansion/contraction)
        breathing_factor = 1.0 + 0.1 * np.sin(2 * np.pi * frame / 20)
        positions[frame] = base_positions * breathing_factor
        
        # Add random thermal motion
        positions[frame] += np.random.randn(n_atoms, 3) * 0.05
        
        # Calculate velocities (if not first frame)
        if frame > 0:
            velocities[frame] = (positions[frame] - positions[frame-1]) / 0.001
        
        # Mock forces with distance-based interactions
        forces[frame] = np.random.randn(n_atoms, 3) * 5.0
    
    trajectory = Mock()
    trajectory.positions = positions
    trajectory.velocities = velocities  
    trajectory.forces = forces
    trajectory.n_frames = n_frames
    trajectory.n_atoms = n_atoms
    trajectory.timestep = 0.001  # 1 fs
    trajectory.total_time = n_frames * trajectory.timestep
    trajectory.box_vectors = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    
    # Add frames attribute for compatibility with trajectory analysis
    frames = []
    for frame_idx in range(n_frames):
        frame = Mock()
        frame.positions = positions[frame_idx]
        frame.velocities = velocities[frame_idx]
        frame.forces = forces[frame_idx]
        frames.append(frame)
    trajectory.frames = frames
    
    return trajectory

@pytest.fixture
def mock_force_field():
    """Create a comprehensive mock force field."""
    ff = Mock()
    ff.name = "test_force_field"
    ff.cutoff = 1.0
    ff.switch_distance = 0.8
    
    # Mock parameter methods with realistic values
    ff.get_bond_parameters = Mock(return_value=Mock(k=1000.0, r0=0.15, is_valid=Mock(return_value=True)))
    ff.get_angle_parameters = Mock(return_value=Mock(k=100.0, theta0=109.5, is_valid=Mock(return_value=True)))
    ff.get_dihedral_parameters = Mock(return_value=Mock(k=10.0, n=2, phase=0.0, is_valid=Mock(return_value=True)))
    
    # Mock atom type parameters
    ff.get_atom_parameters = Mock(return_value=Mock(
        sigma=0.35, epsilon=0.1, mass=12.0, charge=0.0, is_valid=Mock(return_value=True)
    ))
    
    # Mock validation methods
    ff.validate_protein_parameters = Mock(return_value={
        'total_atoms': 100,
        'parametrized_atoms': 95,
        'missing_parameters': ['UNK:X'],
        'coverage_percentage': 95.0
    })
    
    # Mock system creation
    ff.create_simulation_system = Mock(return_value=Mock(
        name="test_system",
        n_atoms=100,
        force_terms=[],
        add_force_term=Mock()
    ))
    
    return ff

@pytest.fixture
def mock_simulation_system():
    """Create a comprehensive mock simulation system."""
    system = Mock()
    system.name = "test_system"
    system.n_atoms = 100
    system.box_vectors = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    
    # Mock force terms
    system.force_terms = []
    system.add_force_term = Mock()
    
    # Mock calculation methods
    def mock_calculate_forces(positions):
        n_atoms = len(positions)
        forces = np.random.randn(n_atoms, 3) * 10.0
        potential_energy = np.random.uniform(100, 1000)
        return forces, potential_energy
    
    system.calculate_forces = Mock(side_effect=mock_calculate_forces)
    system.calculate_energy = Mock(return_value=np.random.uniform(100, 1000))
    
    # Mock integration methods
    system.integrate_step = Mock()
    system.apply_constraints = Mock()
    
    return system

# ============================================================================
# ANALYSIS AND VALIDATION FIXTURES
# ============================================================================

@pytest.fixture
def mock_analysis_results():
    """Create comprehensive mock analysis results."""
    n_frames = 100
    
    results = {
        'rmsd': {
            'values': np.random.uniform(0.5, 3.0, n_frames),
            'mean': 1.5,
            'std': 0.5,
            'reference_structure': 'frame_0',
            'aligned_structures': np.random.randn(n_frames, 100, 3)
        },
        'radius_of_gyration': {
            'values': np.random.uniform(1.0, 2.5, n_frames),
            'mean': 1.8,
            'std': 0.3,
            'segmental_rg': {
                'backbone': np.random.uniform(0.8, 2.0, n_frames),
                'sidechain': np.random.uniform(0.5, 1.5, n_frames)
            }
        },
        'hydrogen_bonds': {
            'count_per_frame': np.random.randint(5, 15, n_frames),
            'mean_count': 10.2,
            'lifetimes': np.random.exponential(2.0, 500),
            'donors': ['N-H', 'O-H', 'S-H'],
            'acceptors': ['O', 'N', 'S'],
            'distance_cutoff': 0.35,
            'angle_cutoff': 30.0
        },
        'secondary_structure': {
            'helix_content': np.random.uniform(0.2, 0.6, n_frames),
            'sheet_content': np.random.uniform(0.1, 0.3, n_frames),
            'coil_content': np.random.uniform(0.3, 0.5, n_frames),
            'turn_content': np.random.uniform(0.05, 0.15, n_frames),
            'dssp_assignment': np.random.choice(['H', 'E', 'C', 'T'], (n_frames, 50))
        },
        'ramachandran': {
            'phi_angles': np.random.uniform(-180, 180, (n_frames, 49)),
            'psi_angles': np.random.uniform(-180, 180, (n_frames, 49)),
            'ramachandran_plot_data': np.random.rand(36, 36),
            'outliers': np.random.choice(49, 5, replace=False)
        },
        'energy': {
            'total': np.random.uniform(1000, 2000, n_frames),
            'kinetic': np.random.uniform(200, 400, n_frames),
            'potential': np.random.uniform(600, 1600, n_frames),
            'bond': np.random.uniform(50, 150, n_frames),
            'angle': np.random.uniform(30, 100, n_frames),
            'dihedral': np.random.uniform(20, 80, n_frames),
            'nonbonded': np.random.uniform(400, 1200, n_frames)
        }
    }
    return results

@pytest.fixture
def validation_thresholds():
    """Define validation thresholds for testing."""
    return {
        'rmsd_max': 5.0,  # nm
        'rg_range': (0.5, 5.0),  # nm
        'energy_conservation': 0.05,  # 5% drift allowed
        'force_magnitude_max': 1000.0,  # kJ/mol/nm
        'temperature_tolerance': 20.0,  # K
        'pressure_tolerance': 50.0,  # bar
        'bond_length_tolerance': 0.02,  # nm
        'angle_tolerance': 5.0,  # degrees
        'performance_factor': 2.0,  # 2x slowdown acceptable
    }

# ============================================================================
# SPECIALIZED FIXTURES FOR DIFFERENT MODULES
# ============================================================================

@pytest.fixture
def water_system_fixture():
    """Create a mock TIP3P water system."""
    water = Mock()
    water.n_molecules = 216  # Standard 6x6x6 water box
    water.n_atoms = 648  # 3 atoms per water
    
    # Generate realistic water box
    box_size = 1.86  # nm for ~1000 kg/m³ density
    positions = []
    
    for i in range(216):
        # Place water molecules in grid with some disorder
        grid_pos = np.array([i % 6, (i // 6) % 6, i // 36]) * (box_size / 6)
        grid_pos += np.random.normal(0, 0.02, 3)  # Small random displacement
        
        # O-H bond length: 0.0957 nm, H-O-H angle: 104.5°
        o_pos = grid_pos
        h1_pos = o_pos + np.array([0.0757, 0.0587, 0.0])
        h2_pos = o_pos + np.array([-0.0757, 0.0587, 0.0])
        
        positions.extend([o_pos, h1_pos, h2_pos])
    
    water.positions = np.array(positions)
    water.box_size = np.array([box_size, box_size, box_size])
    water.atom_types = ['O', 'H', 'H'] * 216
    water.charges = [-0.834, 0.417, 0.417] * 216
    water.masses = [15.999, 1.008, 1.008] * 216
    
    return water

@pytest.fixture
def pbc_system_fixture():
    """Create a periodic boundary conditions test system."""
    pbc = Mock()
    pbc.box_vectors = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    pbc.box_type = "cubic"
    pbc.volume = 125.0
    
    # Mock PBC methods
    def mock_wrap_positions(positions):
        box_size = 5.0
        return positions - np.floor(positions / box_size) * box_size
    
    def mock_minimum_image_distance(pos1, pos2):
        diff = pos1 - pos2
        box_size = 5.0
        diff = diff - np.round(diff / box_size) * box_size
        return np.linalg.norm(diff)
    
    pbc.wrap_positions = Mock(side_effect=mock_wrap_positions)
    pbc.minimum_image_distance = Mock(side_effect=mock_minimum_image_distance)
    
    return pbc

@pytest.fixture
def force_field_benchmarks():
    """Provide benchmark data for force field validation."""
    return {
        'amber_ff14sb': {
            'alanine_dipeptide': {
                'bond_energy': 245.3,  # kJ/mol
                'angle_energy': 134.7,
                'dihedral_energy': 15.2,
                'nonbonded_energy': -89.4,
                'total_energy': 305.8
            },
            'ubiquitin': {
                'rmsd_stability': 0.15,  # nm after 10 ns
                'rg_mean': 1.21,  # nm
                'secondary_structure_stability': 0.85
            }
        }
    }

# Additional fixtures for comprehensive testing

@pytest.fixture
def mock_trajectory_data():
    """Mock trajectory data for animation testing."""
    n_frames = 20
    n_atoms = 50
    trajectory = np.random.randn(n_frames, n_atoms, 3) * 2.0
    # Add some correlated motion
    for frame in range(1, n_frames):
        trajectory[frame] = trajectory[frame-1] * 0.9 + trajectory[frame] * 0.1
    return trajectory

@pytest.fixture
def mock_large_trajectory_data():
    """Mock large trajectory data for performance testing."""
    n_frames = 100
    n_atoms = 500
    return np.random.randn(n_frames, n_atoms, 3) * 3.0

@pytest.fixture
def mock_energy_data():
    """Mock energy data for dashboard testing."""
    n_points = 100
    time_points = np.linspace(0, 10, n_points)
    
    # Generate realistic energy data
    kinetic = 300 + 50 * np.sin(time_points) + np.random.normal(0, 10, n_points)
    potential = -1000 + 100 * np.cos(time_points * 0.5) + np.random.normal(0, 20, n_points)
    total = kinetic + potential
    
    return {
        'time': time_points,
        'kinetic': kinetic,
        'potential': potential,
        'total': total
    }

@pytest.fixture
def mock_large_protein():
    """Mock large protein for performance testing."""
    protein = Mock()
    
    # Create 1000 atoms
    atoms = []
    for i in range(1000):
        atom = Mock()
        atom.name = 'CA' if i % 4 == 1 else 'C'
        atom.position = np.random.randn(3) * 10
        atom.element = 'C'
        atom.residue_name = 'ALA'
        atom.residue_number = i // 4 + 1
        atom.chain_id = 'A'
        atom.occupancy = 1.0
        atom.b_factor = 20.0
        atom.distance_to = lambda other: np.linalg.norm(atom.position - other.position)
        atoms.append(atom)
    
    protein.atoms = atoms
    protein.n_atoms = len(atoms)
    
    # Create residues
    residues = []
    for i in range(0, len(atoms), 4):
        residue = Mock()
        residue.name = 'ALA'
        residue.number = i // 4 + 1
        residue.chain_id = 'A'
        residue.atoms = atoms[i:i+4]
        residues.append(residue)
    
    protein.residues = residues
    
    # Create chains
    chain_a = Mock()
    chain_a.id = 'A'
    chain_a.residues = residues
    protein.chains = {'A': chain_a}
    
    # Add methods
    protein.center_of_mass = lambda: np.mean([atom.position for atom in atoms], axis=0)
    protein.bounding_box = lambda: {
        'min': np.min([atom.position for atom in atoms], axis=0),
        'max': np.max([atom.position for atom in atoms], axis=0)
    }
    protein.select_atoms = lambda **kwargs: [
        atom for atom in atoms 
        if all(getattr(atom, k, None) == v for k, v in kwargs.items())
    ]
    protein.validate = lambda: {'valid': True, 'errors': []}
    protein.align_to = lambda other: np.random.uniform(0, 5)
    protein.translate = lambda vec: None
    protein.rotate = lambda mat: None
    protein.superpose_on = lambda other: np.random.uniform(0, 3)
    
    return protein

# Fix fixture name consistency and add missing fixtures

@pytest.fixture
def mock_protein(mock_protein_structure):
    """Alias for mock_protein_structure for consistency."""
    return mock_protein_structure

@pytest.fixture
def mock_large_trajectory():
    """Mock large trajectory for performance testing."""
    return mock_large_trajectory_data()

@pytest.fixture
def benchmark():
    """Mock benchmark fixture for performance tests."""
    def mock_benchmark(func):
        import time
        start = time.time()
        result = func()
        end = time.time()
        return result
    return mock_benchmark
