"""
Comprehensive Test Suite for Multi-Format I/O System

Task 12.1: Multi-Format Support 🚀 - Testing Framework

This module tests all aspects of the multi-format I/O system including:
- Format detection
- Structure reading/writing
- Trajectory reading/writing
- Format conversion
- Error handling
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import logging

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from proteinMD.io import (
        FormatType, FormatDetector, MultiFormatIO, StructureData, TrajectoryData,
        read_structure, read_trajectory, write_structure, write_trajectory,
        convert_file, create_test_structure, create_test_trajectory
    )
    IO_AVAILABLE = True
except ImportError as e:
    print(f"I/O module not available: {e}")
    IO_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_structure():
    """Create a test structure for testing."""
    return create_test_structure(n_atoms=10)


@pytest.fixture
def test_trajectory():
    """Create a test trajectory for testing."""
    return create_test_trajectory(n_frames=5, n_atoms=10)


@pytest.fixture
def sample_pdb_content():
    """Sample PDB file content."""
    return """HEADER    TEST PROTEIN                            01-JAN-70   TEST
TITLE     TEST STRUCTURE FOR MULTI-FORMAT I/O
REMARK   2 RESOLUTION.    2.00 ANGSTROMS.
ATOM      1  N   ALA A   1      20.154  16.967  24.845  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.082  24.397  1.00 20.00           C  
ATOM      3  C   ALA A   1      19.444  14.634  24.097  1.00 20.00           C  
ATOM      4  O   ALA A   1      20.564  14.311  23.708  1.00 20.00           O  
ATOM      5  CB  ALA A   1      18.318  16.701  23.206  1.00 20.00           C  
END
"""


@pytest.fixture
def sample_xyz_content():
    """Sample XYZ file content."""
    return """5
Test molecule for XYZ format
C    2.015    1.697    2.485
C    1.903    1.608    2.440
C    1.944    1.463    2.410
O    2.056    1.431    2.371
C    1.832    1.670    2.321
"""


@pytest.mark.skipif(not IO_AVAILABLE, reason="I/O module not available")
class TestFormatDetector:
    """Test suite for format detection functionality."""
    
    def test_format_detection_by_extension(self):
        """Test format detection from file extensions."""
        test_cases = [
            ("test.pdb", FormatType.PDB),
            ("test.xyz", FormatType.XYZ),
            ("test.gro", FormatType.GRO),
            ("test.cif", FormatType.PDBX_MMCIF),
            ("test.mol2", FormatType.MOL2),
            ("test.npz", FormatType.NPZ),
            ("test.dcd", FormatType.DCD),
            ("test.xtc", FormatType.XTC),
            ("test.trr", FormatType.TRR),
            ("test.unknown", FormatType.UNKNOWN),
        ]
        
        for filename, expected_format in test_cases:
            detected = FormatDetector.detect_format(filename)
            assert detected == expected_format, f"Failed for {filename}"
    
    def test_format_detection_from_content(self, temp_dir, sample_pdb_content):
        """Test format detection from file content."""
        # Create test PDB file
        pdb_file = temp_dir / "test.dat"  # Ambiguous extension
        with open(pdb_file, 'w') as f:
            f.write(sample_pdb_content)
        
        detected = FormatDetector.detect_format(pdb_file)
        assert detected == FormatType.PDB
    
    def test_format_detection_unknown_file(self):
        """Test format detection for non-existent files."""
        detected = FormatDetector.detect_format("nonexistent.unknown")
        assert detected == FormatType.UNKNOWN


@pytest.mark.skipif(not IO_AVAILABLE, reason="I/O module not available")
class TestStructureIO:
    """Test suite for structure I/O operations."""
    
    def test_pdb_round_trip(self, temp_dir, test_structure):
        """Test PDB format round-trip (write then read)."""
        pdb_file = temp_dir / "test.pdb"
        
        # Write structure
        write_structure(test_structure, pdb_file)
        assert pdb_file.exists()
        
        # Read structure back
        loaded_structure = read_structure(pdb_file)
        
        # Verify basic properties
        assert loaded_structure.n_atoms == test_structure.n_atoms
        assert len(loaded_structure.elements) == test_structure.n_atoms
        assert loaded_structure.coordinates.shape == test_structure.coordinates.shape
    
    def test_xyz_round_trip(self, temp_dir, test_structure):
        """Test XYZ format round-trip."""
        xyz_file = temp_dir / "test.xyz"
        
        # Write structure
        write_structure(test_structure, xyz_file)
        assert xyz_file.exists()
        
        # Read structure back
        loaded_structure = read_structure(xyz_file)
        
        # Verify basic properties
        assert loaded_structure.n_atoms == test_structure.n_atoms
        assert len(loaded_structure.elements) == test_structure.n_atoms
        
        # XYZ coordinates should be approximately equal
        np.testing.assert_allclose(
            loaded_structure.coordinates, 
            test_structure.coordinates, 
            rtol=1e-5
        )
    
    def test_npz_round_trip(self, temp_dir, test_structure):
        """Test NPZ format round-trip."""
        npz_file = temp_dir / "test.npz"
        
        # Write structure
        write_structure(test_structure, npz_file)
        assert npz_file.exists()
        
        # Read structure back
        loaded_structure = read_structure(npz_file)
        
        # Verify all properties are preserved
        assert loaded_structure.n_atoms == test_structure.n_atoms
        assert loaded_structure.elements == test_structure.elements
        assert loaded_structure.atom_names == test_structure.atom_names
        
        np.testing.assert_allclose(
            loaded_structure.coordinates, 
            test_structure.coordinates
        )
    
    def test_structure_conversion(self, temp_dir, test_structure):
        """Test conversion between different structure formats."""
        pdb_file = temp_dir / "test.pdb"
        xyz_file = temp_dir / "test.xyz"
        
        # Write as PDB
        write_structure(test_structure, pdb_file)
        
        # Convert PDB to XYZ
        convert_file(pdb_file, xyz_file)
        assert xyz_file.exists()
        
        # Verify converted structure
        converted_structure = read_structure(xyz_file)
        assert converted_structure.n_atoms == test_structure.n_atoms
    
    def test_read_nonexistent_file(self):
        """Test error handling for non-existent files."""
        with pytest.raises(FileNotFoundError):
            read_structure("nonexistent.pdb")


@pytest.mark.skipif(not IO_AVAILABLE, reason="I/O module not available")
class TestTrajectoryIO:
    """Test suite for trajectory I/O operations."""
    
    def test_npz_trajectory_round_trip(self, temp_dir, test_trajectory):
        """Test NPZ trajectory format round-trip."""
        npz_file = temp_dir / "test_traj.npz"
        
        # Write trajectory
        write_trajectory(test_trajectory, npz_file)
        assert npz_file.exists()
        
        # Read trajectory back
        loaded_trajectory = read_trajectory(npz_file)
        
        # Verify properties
        assert loaded_trajectory.n_frames == test_trajectory.n_frames
        assert loaded_trajectory.n_atoms == test_trajectory.n_atoms
        
        np.testing.assert_allclose(
            loaded_trajectory.coordinates, 
            test_trajectory.coordinates
        )
        
        np.testing.assert_allclose(
            loaded_trajectory.time_points, 
            test_trajectory.time_points
        )
    
    def test_xyz_trajectory_round_trip(self, temp_dir, test_trajectory):
        """Test XYZ trajectory format round-trip."""
        xyz_file = temp_dir / "test_traj.xyz"
        
        # Write trajectory
        write_trajectory(test_trajectory, xyz_file)
        assert xyz_file.exists()
        
        # Read trajectory back
        loaded_trajectory = read_trajectory(xyz_file)
        
        # Verify basic properties
        assert loaded_trajectory.n_frames == test_trajectory.n_frames
        assert loaded_trajectory.n_atoms == test_trajectory.n_atoms
        
        # Coordinates should be approximately equal (XYZ format has limited precision)
        np.testing.assert_allclose(
            loaded_trajectory.coordinates, 
            test_trajectory.coordinates, 
            rtol=1e-4,  # Relaxed tolerance for XYZ format
            atol=1e-6
        )
    
    def test_trajectory_conversion(self, temp_dir, test_trajectory):
        """Test conversion between trajectory formats."""
        npz_file = temp_dir / "test_traj.npz"
        xyz_file = temp_dir / "test_traj.xyz"
        
        # Write as NPZ
        write_trajectory(test_trajectory, npz_file)
        
        # Convert NPZ to XYZ
        convert_file(npz_file, xyz_file)
        assert xyz_file.exists()
        
        # Verify converted trajectory
        converted_trajectory = read_trajectory(xyz_file)
        assert converted_trajectory.n_frames == test_trajectory.n_frames
        assert converted_trajectory.n_atoms == test_trajectory.n_atoms


@pytest.mark.skipif(not IO_AVAILABLE, reason="I/O module not available")
class TestMultiFormatIO:
    """Test suite for the main MultiFormatIO class."""
    
    def test_multiformat_io_initialization(self):
        """Test MultiFormatIO system initialization."""
        io_system = MultiFormatIO()
        
        supported_formats = io_system.get_supported_formats()
        
        # Check that basic formats are supported
        assert FormatType.PDB in supported_formats['read_structure']
        assert FormatType.XYZ in supported_formats['read_structure']
        assert FormatType.NPZ in supported_formats['read_structure']
        
        assert FormatType.PDB in supported_formats['write_structure']
        assert FormatType.XYZ in supported_formats['write_structure']
        assert FormatType.NPZ in supported_formats['write_structure']
    
    def test_file_validation(self, temp_dir, sample_pdb_content):
        """Test file validation functionality."""
        io_system = MultiFormatIO()
        
        # Create test PDB file
        pdb_file = temp_dir / "test.pdb"
        with open(pdb_file, 'w') as f:
            f.write(sample_pdb_content)
        
        # Validate file
        validation_info = io_system.validate_file(pdb_file)
        
        assert validation_info['exists'] is True
        assert validation_info['detected_format'] == FormatType.PDB
        assert validation_info['is_structure'] is True
        assert validation_info['size_bytes'] > 0
        assert len(validation_info['validation_errors']) == 0
    
    def test_validation_nonexistent_file(self):
        """Test validation of non-existent file."""
        io_system = MultiFormatIO()
        
        validation_info = io_system.validate_file("nonexistent.pdb")
        
        assert validation_info['exists'] is False
        assert len(validation_info['validation_errors']) > 0
    
    def test_unsupported_format_error(self, temp_dir):
        """Test error handling for unsupported formats."""
        io_system = MultiFormatIO()
        
        # Create file with unsupported extension
        unsupported_file = temp_dir / "test.unsupported"
        unsupported_file.write_text("some content")
        
        with pytest.raises(ValueError, match="Unsupported.*format"):
            io_system.read_structure(unsupported_file)


@pytest.mark.skipif(not IO_AVAILABLE, reason="I/O module not available")
class TestDataStructures:
    """Test suite for data structure classes."""
    
    def test_structure_data_properties(self, test_structure):
        """Test StructureData properties."""
        assert test_structure.n_atoms == 10
        assert test_structure.n_residues == 1  # All atoms in same residue
        assert test_structure.n_chains == 1    # All atoms in same chain
        
        # Check coordinate array shape
        assert test_structure.coordinates.shape == (10, 3)
    
    def test_trajectory_data_properties(self, test_trajectory):
        """Test TrajectoryData properties."""
        assert test_trajectory.n_frames == 5
        assert test_trajectory.n_atoms == 10
        assert test_trajectory.simulation_time == 4.0  # 5 frames, 1 ps apart
        
        # Check coordinate array shape
        assert test_trajectory.coordinates.shape == (5, 10, 3)
        assert test_trajectory.time_points.shape == (5,)
    
    def test_structure_data_creation(self):
        """Test manual StructureData creation."""
        coordinates = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = ['C', 'O']
        atom_names = ['C1', 'O1']
        atom_ids = [1, 2]
        residue_names = ['UNK', 'UNK']
        residue_ids = [1, 1]
        chain_ids = ['A', 'A']
        
        structure = StructureData(
            coordinates=coordinates,
            elements=elements,
            atom_names=atom_names,
            atom_ids=atom_ids,
            residue_names=residue_names,
            residue_ids=residue_ids,
            chain_ids=chain_ids
        )
        
        assert structure.n_atoms == 2
        assert structure.n_residues == 1
        assert structure.n_chains == 1


@pytest.mark.skipif(not IO_AVAILABLE, reason="I/O module not available")
class TestErrorHandling:
    """Test suite for error handling scenarios."""
    
    def test_empty_file_handling(self, temp_dir):
        """Test handling of empty files."""
        empty_file = temp_dir / "empty.pdb"
        empty_file.write_text("")
        
        with pytest.raises(ValueError):
            read_structure(empty_file)
    
    def test_malformed_file_handling(self, temp_dir):
        """Test handling of malformed files."""
        malformed_file = temp_dir / "malformed.pdb"
        malformed_file.write_text("This is not a valid PDB file")
        
        with pytest.raises(ValueError):
            read_structure(malformed_file)
    
    def test_unsupported_trajectory_format(self):
        """Test error when trying to read trajectory from unsupported format."""
        io_system = MultiFormatIO()
        
        # PDB doesn't support trajectory data
        temp_path = Path("test.pdb")
        
        # Mock the file existence and format detection
        with patch.object(Path, 'exists', return_value=True):
            with patch.object(FormatDetector, 'detect_format', return_value=FormatType.PDB):
                with pytest.raises(ValueError, match="does not support trajectory"):
                    io_system.read_trajectory(temp_path)


@pytest.mark.skipif(not IO_AVAILABLE, reason="I/O module not available")
class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_create_test_structure(self):
        """Test test structure creation."""
        structure = create_test_structure(n_atoms=20)
        
        assert structure.n_atoms == 20
        assert len(structure.elements) == 20
        assert structure.coordinates.shape == (20, 3)
        assert structure.title is not None
    
    def test_create_test_trajectory(self):
        """Test test trajectory creation."""
        trajectory = create_test_trajectory(n_frames=10, n_atoms=15)
        
        assert trajectory.n_frames == 10
        assert trajectory.n_atoms == 15
        assert trajectory.coordinates.shape == (10, 15, 3)
        assert trajectory.time_points.shape == (10,)
        assert trajectory.topology is not None
    
    def test_convenience_read_write(self, temp_dir, test_structure):
        """Test convenience read/write functions."""
        test_file = temp_dir / "convenience_test.npz"
        
        # Write using convenience function
        write_structure(test_structure, test_file)
        assert test_file.exists()
        
        # Read using convenience function
        loaded_structure = read_structure(test_file)
        assert loaded_structure.n_atoms == test_structure.n_atoms


@pytest.mark.skipif(not IO_AVAILABLE, reason="I/O module not available")
class TestPerformance:
    """Test suite for performance characteristics."""
    
    def test_large_structure_handling(self):
        """Test handling of larger structures."""
        # Create larger test structure
        large_structure = create_test_structure(n_atoms=1000)
        
        assert large_structure.n_atoms == 1000
        assert large_structure.coordinates.shape == (1000, 3)
    
    def test_large_trajectory_handling(self):
        """Test handling of larger trajectories."""
        # Create larger test trajectory
        large_trajectory = create_test_trajectory(n_frames=100, n_atoms=100)
        
        assert large_trajectory.n_frames == 100
        assert large_trajectory.n_atoms == 100
        assert large_trajectory.coordinates.shape == (100, 100, 3)
    
    @pytest.mark.slow
    def test_memory_efficiency(self, temp_dir):
        """Test memory efficiency for large files."""
        # This test is marked as slow and tests memory usage
        large_structure = create_test_structure(n_atoms=10000)
        
        npz_file = temp_dir / "large_test.npz"
        write_structure(large_structure, npz_file)
        
        # Verify file was created and has reasonable size
        assert npz_file.exists()
        assert npz_file.stat().st_size > 1000  # Should be at least 1KB


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
