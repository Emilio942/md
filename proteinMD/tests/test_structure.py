"""
Comprehensive Unit Tests for Structure Module

Task 10.1: Umfassende Unit Tests 🚀

Tests the structure modules including:
- PDB parser and reader
- Protein data structures
- Structure validation and manipulation
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
import io

# Try to import structure modules
try:
    from structure.pdb_parser import PDBParser, PDBReader, PDBWriter
    PDB_PARSER_AVAILABLE = True
except ImportError:
    PDB_PARSER_AVAILABLE = False

try:
    from structure.protein import Protein, Residue, Atom, Chain
    PROTEIN_AVAILABLE = True
except ImportError:
    PROTEIN_AVAILABLE = False


@pytest.mark.skipif(not PDB_PARSER_AVAILABLE, reason="PDB parser module not available")
class TestPDBParser:
    """Test suite for PDB parser functionality."""
    
    @pytest.fixture
    def sample_pdb_content(self):
        """Sample PDB file content for testing."""
        return """HEADER    PROTEIN                                 01-JAN-00   1ABC              
ATOM      1  N   ALA A   1      20.154  16.967  23.421  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  17.860  23.632  1.00 20.00           C  
ATOM      3  C   ALA A   1      18.042  17.319  24.650  1.00 20.00           C  
ATOM      4  O   ALA A   1      18.005  16.140  24.969  1.00 20.00           O  
ATOM      5  CB  ALA A   1      18.357  18.100  22.289  1.00 20.00           C  
ATOM      6  N   VAL A   2      17.221  18.132  25.172  1.00 20.00           N  
ATOM      7  CA  VAL A   2      16.167  17.738  26.125  1.00 20.00           C  
ATOM      8  C   VAL A   2      14.805  18.336  25.780  1.00 20.00           C  
ATOM      9  O   VAL A   2      14.691  19.521  25.464  1.00 20.00           O  
ATOM     10  CB  VAL A   2      16.055  16.214  26.287  1.00 20.00           C  
END"""
    
    def test_pdb_parser_initialization(self):
        """Test PDB parser initialization."""
        parser = PDBParser()
        assert parser is not None
    
    def test_pdb_parsing_from_string(self, sample_pdb_content):
        """Test PDB parsing from string content."""
        parser = PDBParser()
        protein = parser.parse_string(sample_pdb_content)
        
        assert protein is not None
        assert len(protein.atoms) == 10
        assert len(protein.residues) == 2
    
    def test_pdb_parsing_from_file(self, sample_pdb_content):
        """Test PDB parsing from file."""
        parser = PDBParser()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            tmp.write(sample_pdb_content)
            tmp.flush()
            
            protein = parser.parse_file(tmp.name)
            
        assert protein is not None
        assert len(protein.atoms) == 10
    
    def test_atom_parsing(self, sample_pdb_content):
        """Test individual atom parsing."""
        parser = PDBParser()
        protein = parser.parse_string(sample_pdb_content)
        
        first_atom = protein.atoms[0]
        assert first_atom.name == 'N'
        assert first_atom.residue_name == 'ALA'
        assert first_atom.chain_id == 'A'
        assert first_atom.residue_number == 1
        assert np.allclose(first_atom.position, [20.154, 16.967, 23.421])
        assert first_atom.occupancy == 1.00
        assert first_atom.b_factor == 20.00
    
    def test_residue_parsing(self, sample_pdb_content):
        """Test residue parsing and organization."""
        parser = PDBParser()
        protein = parser.parse_string(sample_pdb_content)
        
        residues = protein.residues
        assert len(residues) == 2
        
        ala_residue = residues[0]
        assert ala_residue.name == 'ALA'
        assert ala_residue.number == 1
        assert ala_residue.chain_id == 'A'
        assert len(ala_residue.atoms) == 5
    
    def test_chain_parsing(self, sample_pdb_content):
        """Test chain parsing and organization."""
        parser = PDBParser()
        protein = parser.parse_string(sample_pdb_content)
        
        chains = protein.chains
        assert len(chains) == 1
        
        chain_a = chains['A']
        assert chain_a.id == 'A'
        assert len(chain_a.residues) == 2
    
    def test_malformed_pdb_handling(self):
        """Test handling of malformed PDB content."""
        parser = PDBParser()
        malformed_content = """INVALID LINE FORMAT
ATOM      1  N   ALA A   1      NOT_A_NUMBER  16.967  23.421  1.00 20.00           N  
"""
        
        with pytest.raises((ValueError, TypeError)):
            parser.parse_string(malformed_content)
    
    def test_empty_pdb_handling(self):
        """Test handling of empty PDB content."""
        parser = PDBParser()
        empty_content = """HEADER    EMPTY STRUCTURE
END"""
        
        protein = parser.parse_string(empty_content)
        assert len(protein.atoms) == 0
        assert len(protein.residues) == 0


@pytest.mark.skipif(not PDB_PARSER_AVAILABLE, reason="PDB parser module not available")
class TestPDBWriter:
    """Test suite for PDB writer functionality."""
    
    def test_pdb_writer_initialization(self):
        """Test PDB writer initialization."""
        writer = PDBWriter()
        assert writer is not None
    
    def test_protein_to_pdb_string(self, mock_protein):
        """Test converting protein to PDB string."""
        writer = PDBWriter()
        pdb_string = writer.write_string(mock_protein)
        
        assert isinstance(pdb_string, str)
        assert 'ATOM' in pdb_string
        assert 'END' in pdb_string
    
    def test_protein_to_pdb_file(self, mock_protein):
        """Test writing protein to PDB file."""
        writer = PDBWriter()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            writer.write_file(mock_protein, tmp.name)
            
            # Verify file was created and has content
            assert Path(tmp.name).exists()
            with open(tmp.name, 'r') as f:
                content = f.read()
                assert 'ATOM' in content
    
    def test_atom_formatting(self, mock_protein):
        """Test proper atom formatting in PDB output."""
        writer = PDBWriter()
        pdb_string = writer.write_string(mock_protein)
        
        lines = pdb_string.split('\n')
        atom_lines = [line for line in lines if line.startswith('ATOM')]
        
        assert len(atom_lines) > 0
        
        # Check format of first atom line
        first_line = atom_lines[0]
        assert len(first_line) >= 66  # Minimum PDB line length
        assert first_line[12:16].strip()  # Atom name should be present
    
    def test_coordinate_precision(self, mock_protein):
        """Test coordinate precision in PDB output."""
        writer = PDBWriter()
        pdb_string = writer.write_string(mock_protein)
        
        lines = pdb_string.split('\n')
        atom_lines = [line for line in lines if line.startswith('ATOM')]
        
        for line in atom_lines:
            # Extract coordinates
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            
            # Should be reasonable protein coordinates
            assert -1000 < x < 1000
            assert -1000 < y < 1000
            assert -1000 < z < 1000


@pytest.mark.skipif(not PROTEIN_AVAILABLE, reason="Protein module not available")
class TestProteinStructure:
    """Test suite for protein data structure functionality."""
    
    def test_atom_creation(self):
        """Test atom object creation."""
        atom = Atom(
            name='CA',
            position=np.array([1.0, 2.0, 3.0]),
            element='C',
            residue_name='ALA',
            residue_number=1,
            chain_id='A'
        )
        
        assert atom.name == 'CA'
        assert np.allclose(atom.position, [1.0, 2.0, 3.0])
        assert atom.element == 'C'
    
    def test_residue_creation(self):
        """Test residue object creation."""
        atoms = [
            Atom('N', np.array([0.0, 0.0, 0.0]), 'N', 'ALA', 1, 'A'),
            Atom('CA', np.array([1.0, 1.0, 1.0]), 'C', 'ALA', 1, 'A'),
            Atom('C', np.array([2.0, 2.0, 2.0]), 'C', 'ALA', 1, 'A')
        ]
        
        residue = Residue(name='ALA', number=1, chain_id='A', atoms=atoms)
        
        assert residue.name == 'ALA'
        assert residue.number == 1
        assert len(residue.atoms) == 3
    
    def test_chain_creation(self):
        """Test chain object creation."""
        residues = [
            Mock(name='ALA', number=1, chain_id='A'),
            Mock(name='VAL', number=2, chain_id='A')
        ]
        
        chain = Chain(id='A', residues=residues)
        
        assert chain.id == 'A'
        assert len(chain.residues) == 2
    
    def test_protein_creation(self):
        """Test protein object creation."""
        atoms = [Mock() for _ in range(10)]
        residues = [Mock() for _ in range(3)]
        chains = {'A': Mock()}
        
        protein = Protein(atoms=atoms, residues=residues, chains=chains)
        
        assert len(protein.atoms) == 10
        assert len(protein.residues) == 3
        assert len(protein.chains) == 1
    
    def test_protein_center_of_mass(self, mock_protein):
        """Test protein center of mass calculation."""
        com = mock_protein.center_of_mass()
        
        assert isinstance(com, np.ndarray)
        assert len(com) == 3
    
    def test_protein_bounding_box(self, mock_protein):
        """Test protein bounding box calculation."""
        bbox = mock_protein.bounding_box()
        
        assert 'min' in bbox
        assert 'max' in bbox
        assert len(bbox['min']) == 3
        assert len(bbox['max']) == 3
    
    def test_atom_selection(self, mock_protein):
        """Test atom selection functionality."""
        # Select all CA atoms
        ca_atoms = mock_protein.select_atoms(name='CA')
        assert all(atom.name == 'CA' for atom in ca_atoms)
        
        # Select atoms by residue
        residue_atoms = mock_protein.select_atoms(residue_name='ALA')
        assert all(atom.residue_name == 'ALA' for atom in residue_atoms)
    
    def test_distance_calculation(self, mock_protein):
        """Test distance calculation between atoms."""
        atoms = mock_protein.atoms
        if len(atoms) >= 2:
            distance = atoms[0].distance_to(atoms[1])
            assert distance >= 0.0
    
    def test_protein_validation(self, mock_protein):
        """Test protein structure validation."""
        validation = mock_protein.validate()
        
        assert 'valid' in validation
        assert 'errors' in validation
        assert isinstance(validation['valid'], bool)
        assert isinstance(validation['errors'], list)


class TestStructureManipulation:
    """Test suite for structure manipulation operations."""
    
    def test_structure_alignment(self, mock_protein):
        """Test structure alignment functionality."""
        if not PROTEIN_AVAILABLE:
            pytest.skip("Protein module not available")
        
        # Create a slightly modified version
        modified_protein = Mock()
        modified_protein.atoms = [Mock() for _ in range(len(mock_protein.atoms))]
        
        # Test alignment
        rmsd = mock_protein.align_to(modified_protein)
        assert rmsd >= 0.0
    
    def test_structure_transformation(self, mock_protein):
        """Test structure transformation operations."""
        if not PROTEIN_AVAILABLE:
            pytest.skip("Protein module not available")
        
        # Test translation
        translation = np.array([1.0, 2.0, 3.0])
        mock_protein.translate(translation)
        
        # Test rotation
        rotation_matrix = np.eye(3)
        mock_protein.rotate(rotation_matrix)
    
    def test_structure_superposition(self, mock_protein):
        """Test structure superposition."""
        if not PROTEIN_AVAILABLE:
            pytest.skip("Protein module not available")
        
        # Create reference structure
        reference = Mock()
        reference.atoms = [Mock() for _ in range(len(mock_protein.atoms))]
        
        # Test superposition
        rmsd = mock_protein.superpose_on(reference)
        assert rmsd >= 0.0


class TestStructureIntegration:
    """Integration tests for structure modules."""
    
    def test_pdb_round_trip(self, sample_pdb_content=None):
        """Test PDB parsing and writing round trip."""
        if not (PDB_PARSER_AVAILABLE and PROTEIN_AVAILABLE):
            pytest.skip("Required modules not available")
        
        if sample_pdb_content is None:
            sample_pdb_content = """ATOM      1  N   ALA A   1      20.154  16.967  23.421  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  17.860  23.632  1.00 20.00           C  
END"""
        
        # Parse PDB
        parser = PDBParser()
        protein = parser.parse_string(sample_pdb_content)
        
        # Write back to PDB
        writer = PDBWriter()
        output_pdb = writer.write_string(protein)
        
        # Parse again
        protein2 = parser.parse_string(output_pdb)
        
        # Should have same number of atoms
        assert len(protein.atoms) == len(protein2.atoms)
    
    def test_structure_analysis_workflow(self, mock_protein):
        """Test complete structure analysis workflow."""
        if not PROTEIN_AVAILABLE:
            pytest.skip("Protein module not available")
        
        # Calculate basic properties
        com = mock_protein.center_of_mass()
        bbox = mock_protein.bounding_box()
        
        # Perform validation
        validation = mock_protein.validate()
        
        # Select specific atoms
        ca_atoms = mock_protein.select_atoms(name='CA')
        
        # All operations should complete successfully
        assert com is not None
        assert bbox is not None
        assert validation is not None
        assert ca_atoms is not None
    
    def test_multi_chain_protein_handling(self):
        """Test handling of multi-chain proteins."""
        if not (PDB_PARSER_AVAILABLE and PROTEIN_AVAILABLE):
            pytest.skip("Required modules not available")
        
        multi_chain_pdb = """ATOM      1  N   ALA A   1      20.154  16.967  23.421  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  17.860  23.632  1.00 20.00           C  
ATOM      3  N   VAL B   1      18.154  15.967  22.421  1.00 20.00           N  
ATOM      4  CA  VAL B   1      17.030  16.860  22.632  1.00 20.00           C  
END"""
        
        parser = PDBParser()
        protein = parser.parse_string(multi_chain_pdb)
        
        assert len(protein.chains) == 2
        assert 'A' in protein.chains
        assert 'B' in protein.chains


# Performance regression tests
class TestStructurePerformanceRegression:
    """Performance regression tests for structure modules."""
    
    @pytest.mark.performance
    def test_pdb_parsing_performance(self, benchmark):
        """Benchmark PDB parsing performance."""
        if not PDB_PARSER_AVAILABLE:
            pytest.skip("PDB parser not available")
        
        # Create large PDB content
        large_pdb = ""
        for i in range(1000):
            large_pdb += f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}      {i:8.3f}  0.000  0.000  1.00 20.00           C  \n"
        large_pdb += "END"
        
        parser = PDBParser()
        
        def parse_large_pdb():
            return parser.parse_string(large_pdb)
        
        protein = benchmark(parse_large_pdb)
        assert len(protein.atoms) == 1000
    
    @pytest.mark.performance
    def test_protein_analysis_performance(self, mock_large_protein, benchmark):
        """Benchmark protein analysis performance."""
        if not PROTEIN_AVAILABLE:
            pytest.skip("Protein module not available")
        
        def analyze_protein():
            com = mock_large_protein.center_of_mass()
            bbox = mock_large_protein.bounding_box()
            ca_atoms = mock_large_protein.select_atoms(name='CA')
            return com, bbox, ca_atoms
        
        com, bbox, ca_atoms = benchmark(analyze_protein)
        assert com is not None
        assert bbox is not None
        assert ca_atoms is not None
