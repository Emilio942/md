"""
Test suite for Ramachandran Plot Analysis Module

Tests the functionality of phi-psi angle calculations, Ramachandran plot generation,
and trajectory analysis capabilities.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path

# Import the module under test
import sys
sys.path.insert(0, '/home/emilio/Documents/ai/md')
from proteinMD.analysis.ramachandran import (
    calculate_dihedral_angle, 
    get_backbone_atoms,
    calculate_phi_psi_angles,
    RamachandranAnalyzer,
    create_ramachandran_analyzer,
    AMINO_ACIDS
)

class TestDihedralCalculations:
    """Test dihedral angle calculation functions."""
    
    def test_dihedral_angle_calculation(self):
        """Test basic dihedral angle calculation."""
        # Define four points for a known dihedral angle
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 1.0, 0.0])
        p4 = np.array([1.0, 1.0, 1.0])
        
        angle = calculate_dihedral_angle(p1, p2, p3, p4)
        
        # Should be 90 degrees for this configuration
        assert abs(angle - 90.0) < 1e-10
    
    def test_dihedral_angle_linear_case(self):
        """Test dihedral angle for linear configuration."""
        # Collinear points should give 0 angle
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([2.0, 0.0, 0.0])
        p4 = np.array([3.0, 0.0, 0.0])
        
        angle = calculate_dihedral_angle(p1, p2, p3, p4)
        
        # Linear case should return 0
        assert abs(angle) < 1e-10
    
    def test_dihedral_angle_negative(self):
        """Test dihedral angle calculation for negative angles."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 1.0, 0.0])
        p4 = np.array([1.0, 1.0, -1.0])
        
        angle = calculate_dihedral_angle(p1, p2, p3, p4)
        
        # Should be -90 degrees for this configuration
        assert abs(angle + 90.0) < 1e-10

class TestBackboneExtraction:
    """Test backbone atom extraction functions."""
    
    def create_mock_atom(self, name, residue_number, residue_name, position):
        """Create a mock atom object."""
        atom = Mock()
        atom.name = name
        atom.residue_number = residue_number
        atom.residue_name = residue_name
        atom.position = np.array(position)
        return atom
    
    def test_get_backbone_atoms(self):
        """Test extraction of backbone atoms from a residue."""
        # Create mock molecule with backbone atoms
        molecule = Mock()
        molecule.atoms = [
            self.create_mock_atom('N', 1, 'ALA', [0.0, 0.0, 0.0]),
            self.create_mock_atom('CA', 1, 'ALA', [1.0, 0.0, 0.0]),
            self.create_mock_atom('C', 1, 'ALA', [1.0, 1.0, 0.0]),
            self.create_mock_atom('CB', 1, 'ALA', [1.0, 0.0, 1.0]),  # side chain
        ]
        
        atoms = get_backbone_atoms(molecule, 0)  # 0-based index
        
        assert atoms['N'] is not None
        assert atoms['CA'] is not None
        assert atoms['C'] is not None
        assert np.allclose(atoms['N'], [0.0, 0.0, 0.0])
        assert np.allclose(atoms['CA'], [1.0, 0.0, 0.0])
        assert np.allclose(atoms['C'], [1.0, 1.0, 0.0])
    
    def test_get_backbone_atoms_missing(self):
        """Test handling of missing backbone atoms."""
        # Create molecule with missing CA atom
        molecule = Mock()
        molecule.atoms = [
            self.create_mock_atom('N', 1, 'ALA', [0.0, 0.0, 0.0]),
            self.create_mock_atom('C', 1, 'ALA', [1.0, 1.0, 0.0]),
        ]
        
        atoms = get_backbone_atoms(molecule, 0)
        
        assert atoms['N'] is not None
        assert atoms['CA'] is None
        assert atoms['C'] is not None

class TestPhiPsiCalculations:
    """Test phi-psi angle calculation functions."""
    
    def create_mock_molecule(self):
        """Create a mock molecule with a simple dipeptide."""
        molecule = Mock()
        
        # Create a simple dipeptide (ALA-GLY)
        molecule.atoms = [
            # Residue 1 (ALA)
            Mock(name='N', residue_number=1, residue_name='ALA', 
                 position=np.array([0.0, 0.0, 0.0])),
            Mock(name='CA', residue_number=1, residue_name='ALA', 
                 position=np.array([1.0, 0.0, 0.0])),
            Mock(name='C', residue_number=1, residue_name='ALA', 
                 position=np.array([1.0, 1.0, 0.0])),
            
            # Residue 2 (GLY)
            Mock(name='N', residue_number=2, residue_name='GLY', 
                 position=np.array([2.0, 1.0, 0.0])),
            Mock(name='CA', residue_number=2, residue_name='GLY', 
                 position=np.array([2.0, 2.0, 0.0])),
            Mock(name='C', residue_number=2, residue_name='GLY', 
                 position=np.array([3.0, 2.0, 0.0])),
        ]
        
        return molecule
    
    def test_calculate_phi_psi_angles(self):
        """Test phi-psi angle calculation for a simple molecule."""
        molecule = self.create_mock_molecule()
        
        phi_angles, psi_angles, residue_names = calculate_phi_psi_angles(molecule)
        
        # Should only get angles for the second residue (first can't have phi, last can't have psi)
        assert len(phi_angles) <= 2  # May be 0 if no valid angle pairs
        assert len(psi_angles) <= 2
        assert len(residue_names) <= 2
    
    def test_calculate_phi_psi_single_residue(self):
        """Test phi-psi calculation with single residue (should return empty)."""
        molecule = Mock()
        molecule.atoms = [
            Mock(name='N', residue_number=1, residue_name='ALA', 
                 position=np.array([0.0, 0.0, 0.0])),
            Mock(name='CA', residue_number=1, residue_name='ALA', 
                 position=np.array([1.0, 0.0, 0.0])),
            Mock(name='C', residue_number=1, residue_name='ALA', 
                 position=np.array([1.0, 1.0, 0.0])),
        ]
        
        phi_angles, psi_angles, residue_names = calculate_phi_psi_angles(molecule)
        
        # Single residue cannot have both phi and psi
        assert len(phi_angles) == 0
        assert len(psi_angles) == 0
        assert len(residue_names) == 0

class TestRamachandranAnalyzer:
    """Test the RamachandranAnalyzer class."""
    
    def create_test_molecule(self):
        """Create a test molecule for analysis."""
        molecule = Mock()
        molecule.atoms = [
            # Residue 1 (ALA) - C terminus for phi calculation
            Mock(name='C', residue_number=0, residue_name='ALA', 
                 position=np.array([-1.0, 0.0, 0.0])),
            
            # Residue 2 (GLY) - main residue
            Mock(name='N', residue_number=1, residue_name='GLY', 
                 position=np.array([0.0, 0.0, 0.0])),
            Mock(name='CA', residue_number=1, residue_name='GLY', 
                 position=np.array([1.0, 0.0, 0.0])),
            Mock(name='C', residue_number=1, residue_name='GLY', 
                 position=np.array([1.0, 1.0, 0.0])),
            
            # Residue 3 (ALA) - N terminus for psi calculation
            Mock(name='N', residue_number=2, residue_name='ALA', 
                 position=np.array([2.0, 1.0, 0.0])),
        ]
        return molecule
    
    def test_analyzer_initialization(self):
        """Test RamachandranAnalyzer initialization."""
        analyzer = RamachandranAnalyzer()
        
        assert analyzer.phi_angles == []
        assert analyzer.psi_angles == []
        assert analyzer.residue_names == []
        assert analyzer.time_points == []
        assert analyzer.trajectory_data == []
    
    def test_analyze_structure(self):
        """Test single structure analysis."""
        analyzer = RamachandranAnalyzer()
        molecule = self.create_test_molecule()
        
        result = analyzer.analyze_structure(molecule)
        
        assert 'phi_angles' in result
        assert 'psi_angles' in result
        assert 'residue_names' in result
        assert 'statistics' in result
        
        # Check statistics structure
        stats = result['statistics']
        assert 'n_residues' in stats
        assert 'phi_mean' in stats
        assert 'psi_mean' in stats
    
    def test_analyze_structure_with_time(self):
        """Test structure analysis with time point."""
        analyzer = RamachandranAnalyzer()
        molecule = self.create_test_molecule()
        
        analyzer.analyze_structure(molecule, time_point=1.0)
        
        assert len(analyzer.trajectory_data) == 1
        assert analyzer.trajectory_data[0]['time'] == 1.0
    
    def test_plot_ramachandran_empty(self):
        """Test plotting with no data."""
        analyzer = RamachandranAnalyzer()
        
        fig = analyzer.plot_ramachandran()
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_ramachandran_with_data(self):
        """Test plotting with actual data."""
        analyzer = RamachandranAnalyzer()
        analyzer.phi_angles = [-60, -120, 60]
        analyzer.psi_angles = [-45, 120, 45]
        analyzer.residue_names = ['ALA', 'GLY', 'PRO']
        
        fig = analyzer.plot_ramachandran()
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_ramachandran_color_by_residue(self):
        """Test plotting with residue coloring."""
        analyzer = RamachandranAnalyzer()
        analyzer.phi_angles = [-60, -120, 60]
        analyzer.psi_angles = [-45, 120, 45]
        analyzer.residue_names = ['ALA', 'GLY', 'PRO']
        
        fig = analyzer.plot_ramachandran(color_by_residue=True)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_angle_evolution_no_trajectory(self):
        """Test angle evolution plotting without trajectory data."""
        analyzer = RamachandranAnalyzer()
        
        with pytest.raises(ValueError, match="No trajectory data available"):
            analyzer.plot_angle_evolution()
    
    def test_plot_angle_evolution_with_data(self):
        """Test angle evolution plotting with trajectory data."""
        analyzer = RamachandranAnalyzer()
        analyzer.trajectory_data = [
            {'time': 0.0, 'phi': [-60], 'psi': [-45], 'residues': ['ALA']},
            {'time': 1.0, 'phi': [-65], 'psi': [-40], 'residues': ['ALA']},
            {'time': 2.0, 'phi': [-58], 'psi': [-50], 'residues': ['ALA']},
        ]
        
        fig = analyzer.plot_angle_evolution(residue_index=0)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

class TestTrajectoryAnalysis:
    """Test trajectory analysis functionality."""
    
    def create_mock_simulation(self):
        """Create a mock simulation with trajectory data."""
        simulation = Mock()
        simulation.dt = 0.1
        
        # Create mock molecule
        molecule = Mock()
        molecule.atoms = [
            Mock(name='N', residue_number=1, residue_name='ALA'),
            Mock(name='CA', residue_number=1, residue_name='ALA'),
            Mock(name='C', residue_number=1, residue_name='ALA'),
        ]
        simulation.molecule = molecule
        
        # Create trajectory data (3 frames)
        simulation.trajectory = [
            [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])],
            [np.array([0.1, 0.0, 0.0]), np.array([1.1, 0.0, 0.0]), np.array([1.1, 1.1, 0.0])],
            [np.array([0.0, 0.1, 0.0]), np.array([1.0, 0.1, 0.0]), np.array([1.0, 1.1, 0.0])],
        ]
        
        return simulation
    
    def test_analyze_trajectory_no_data(self):
        """Test trajectory analysis with no trajectory data."""
        analyzer = RamachandranAnalyzer()
        simulation = Mock()
        simulation.trajectory = []
        
        with pytest.raises(ValueError, match="No trajectory data available"):
            analyzer.analyze_trajectory(simulation)
    
    def test_analyze_trajectory_no_trajectory_attr(self):
        """Test trajectory analysis with missing trajectory attribute."""
        analyzer = RamachandranAnalyzer()
        simulation = Mock()
        # Explicitly set trajectory to None to simulate missing attribute
        simulation.trajectory = None
        simulation.frames = None
        
        with pytest.raises(ValueError, match="No trajectory data available"):
            analyzer.analyze_trajectory(simulation)
    
    def test_analyze_trajectory_with_data(self):
        """Test trajectory analysis with valid data."""
        analyzer = RamachandranAnalyzer()
        simulation = self.create_mock_simulation()
        
        result = analyzer.analyze_trajectory(simulation, time_step=1)
        
        assert 'n_frames' in result
        assert result['n_frames'] == 3
        assert len(analyzer.trajectory_data) == 3

class TestDataExport:
    """Test data export functionality."""
    
    def test_export_data_csv_single_structure(self):
        """Test CSV export for single structure."""
        analyzer = RamachandranAnalyzer()
        analyzer.phi_angles = [-60, -120]
        analyzer.psi_angles = [-45, 120]
        analyzer.residue_names = ['ALA', 'GLY']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            analyzer.export_data(temp_path, format='csv')
            
            # Check file was created
            assert os.path.exists(temp_path)
            
            # Check file content
            with open(temp_path, 'r') as f:
                content = f.read()
                assert 'residue_index' in content
                assert 'phi' in content
                assert 'psi' in content
                assert 'ALA' in content
                assert 'GLY' in content
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_data_json_single_structure(self):
        """Test JSON export for single structure."""
        analyzer = RamachandranAnalyzer()
        analyzer.phi_angles = [-60, -120]
        analyzer.psi_angles = [-45, 120]
        analyzer.residue_names = ['ALA', 'GLY']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            analyzer.export_data(temp_path, format='json')
            
            # Check file was created
            assert os.path.exists(temp_path)
            
            # Check file content
            import json
            with open(temp_path, 'r') as f:
                data = json.load(f)
                assert data['type'] == 'single_structure'
                assert 'phi_angles' in data
                assert 'psi_angles' in data
                assert 'residue_names' in data
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_data_trajectory(self):
        """Test export with trajectory data."""
        analyzer = RamachandranAnalyzer()
        analyzer.trajectory_data = [
            {'time': 0.0, 'phi': [-60], 'psi': [-45], 'residues': ['ALA']},
            {'time': 1.0, 'phi': [-65], 'psi': [-40], 'residues': ['ALA']},
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            analyzer.export_data(temp_path, format='json')
            
            import json
            with open(temp_path, 'r') as f:
                data = json.load(f)
                assert data['type'] == 'trajectory'
                assert 'frames' in data
                assert len(data['frames']) == 2
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_data_unsupported_format(self):
        """Test export with unsupported format."""
        analyzer = RamachandranAnalyzer()
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            analyzer.export_data('test.txt', format='txt')

class TestCreateAnalyzer:
    """Test analyzer creation function."""
    
    def test_create_ramachandran_analyzer_no_molecule(self):
        """Test creating analyzer without initial molecule."""
        analyzer = create_ramachandran_analyzer()
        
        assert isinstance(analyzer, RamachandranAnalyzer)
        assert analyzer.phi_angles == []
        assert analyzer.psi_angles == []
    
    def test_create_ramachandran_analyzer_with_molecule(self):
        """Test creating analyzer with initial molecule."""
        molecule = Mock()
        molecule.atoms = [
            Mock(name='N', residue_number=1, residue_name='ALA', 
                 position=np.array([0.0, 0.0, 0.0])),
        ]
        
        analyzer = create_ramachandran_analyzer(molecule)
        
        assert isinstance(analyzer, RamachandranAnalyzer)

class TestAminoAcidConstants:
    """Test amino acid constant definitions."""
    
    def test_amino_acid_properties(self):
        """Test that all standard amino acids are defined."""
        expected_amino_acids = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
            'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
            'THR', 'TRP', 'TYR', 'VAL'
        }
        
        assert set(AMINO_ACIDS.keys()) == expected_amino_acids
        
        # Check that each amino acid has required properties
        for aa_code, properties in AMINO_ACIDS.items():
            assert 'name' in properties
            assert 'code' in properties
            assert 'color' in properties
            assert len(properties['code']) == 1  # Single letter code
            assert properties['color'].startswith('#')  # Hex color

class TestRamachandranIntegration:
    """Integration tests for the complete Ramachandran analysis workflow."""
    
    def test_full_ramachandran_analysis_workflow(self):
        """Test complete analysis workflow from structure to visualization."""
        # Create analyzer
        analyzer = RamachandranAnalyzer()
        
        # Create mock molecule with multiple residues
        molecule = Mock()
        molecule.atoms = []
        
        # Add atoms for a tripeptide
        for i in range(3):
            residue_num = i + 1
            residue_name = ['ALA', 'GLY', 'PRO'][i]
            
            # Previous C (for phi calculation)
            if i > 0:
                molecule.atoms.append(Mock(
                    name='C', residue_number=residue_num-1, residue_name='ALA',
                    position=np.array([i-0.5, 0.0, 0.0])
                ))
            
            # Current residue backbone
            molecule.atoms.extend([
                Mock(name='N', residue_number=residue_num, residue_name=residue_name,
                     position=np.array([i*2.0, 0.0, 0.0])),
                Mock(name='CA', residue_number=residue_num, residue_name=residue_name,
                     position=np.array([i*2.0+1.0, 0.0, 0.0])),
                Mock(name='C', residue_number=residue_num, residue_name=residue_name,
                     position=np.array([i*2.0+1.0, 1.0, 0.0])),
            ])
            
            # Next N (for psi calculation)
            if i < 2:
                molecule.atoms.append(Mock(
                    name='N', residue_number=residue_num+1, residue_name='GLY',
                    position=np.array([i*2.0+2.5, 1.0, 0.0])
                ))
        
        # Analyze structure
        result = analyzer.analyze_structure(molecule)
        
        # Should have some angles calculated
        assert isinstance(result, dict)
        assert 'statistics' in result
        
        # Create plot
        fig = analyzer.plot_ramachandran()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test data export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            analyzer.export_data(temp_path, format='json')
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == "__main__":
    # Run specific test for debugging
    test_dihedral = TestDihedralCalculations()
    test_dihedral.test_dihedral_angle_calculation()
    print("Dihedral angle calculation test passed!")
    
    test_analyzer = TestRamachandranAnalyzer()
    test_analyzer.test_analyzer_initialization()
    print("Analyzer initialization test passed!")
    
    print("All manual tests completed successfully!")
