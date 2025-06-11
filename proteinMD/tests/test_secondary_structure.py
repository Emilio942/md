"""
Unit tests for Secondary Structure Analysis module.

This module contains comprehensive tests for the secondary structure analysis
functionality including DSSP-like algorithm implementation, hydrogen bond
detection, and trajectory analysis capabilities.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, MagicMock
import sys
import pytest

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from proteinMD.analysis.secondary_structure import (
    calculate_dihedral_angle,
    get_backbone_atoms,
    calculate_hydrogen_bond_energy,
    identify_hydrogen_bonds,
    assign_secondary_structure_dssp,
    SecondaryStructureAnalyzer,
    create_secondary_structure_analyzer,
    SS_TYPES
)


class TestDihedralCalculations(unittest.TestCase):
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
        self.assertAlmostEqual(abs(angle), 90.0, places=1)
    
    def test_dihedral_angle_linear_case(self):
        """Test dihedral angle for linear configuration."""
        # Collinear points should give 0 angle
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([2.0, 0.0, 0.0])
        p4 = np.array([3.0, 0.0, 0.0])
        
        angle = calculate_dihedral_angle(p1, p2, p3, p4)
        
        # Linear case should return 0
        self.assertAlmostEqual(abs(angle), 0.0, places=8)
    
    def test_dihedral_angle_negative(self):
        """Test dihedral angle calculation for negative angles."""
        # Configuration that should give negative angle
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 1.0, 0.0])
        p4 = np.array([1.0, 1.0, -1.0])  # Opposite direction
        
        angle = calculate_dihedral_angle(p1, p2, p3, p4)
        
        # Should be -90 degrees for this configuration
        self.assertAlmostEqual(angle, -90.0, places=1)


class TestBackboneExtraction(unittest.TestCase):
    """Test backbone atom extraction functions."""
    
    def create_mock_atom(self, name, res_num, res_name, position):
        """Create a mock atom for testing."""
        atom = Mock()
        atom.atom_name = name
        atom.residue_number = res_num
        atom.residue_name = res_name
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
            self.create_mock_atom('O', 1, 'ALA', [1.0, 1.0, 1.0]),
            self.create_mock_atom('CB', 1, 'ALA', [1.0, 0.0, 1.0]),  # side chain
        ]
        
        atoms = get_backbone_atoms(molecule, 0)  # 0-based index
        
        self.assertIsNotNone(atoms['N'])
        self.assertIsNotNone(atoms['CA'])
        self.assertIsNotNone(atoms['C'])
        self.assertIsNotNone(atoms['O'])
        self.assertTrue(np.allclose(atoms['N'], [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(atoms['CA'], [1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(atoms['C'], [1.0, 1.0, 0.0]))
        self.assertTrue(np.allclose(atoms['O'], [1.0, 1.0, 1.0]))
    
    def test_get_backbone_atoms_missing(self):
        """Test backbone extraction with missing atoms."""
        # Create molecule with incomplete backbone
        molecule = Mock()
        molecule.atoms = [
            self.create_mock_atom('N', 1, 'GLY', [0.0, 0.0, 0.0]),
            self.create_mock_atom('CA', 1, 'GLY', [1.0, 0.0, 0.0]),
            # Missing C and O atoms
        ]
        
        atoms = get_backbone_atoms(molecule, 0)
        
        self.assertIsNotNone(atoms['N'])
        self.assertIsNotNone(atoms['CA'])
        self.assertIsNone(atoms['C'])
        self.assertIsNone(atoms['O'])


class TestHydrogenBondAnalysis(unittest.TestCase):
    """Test hydrogen bond detection and energy calculation."""
    
    def test_hydrogen_bond_energy_calculation(self):
        """Test hydrogen bond energy calculation."""
        # Define positions for a typical hydrogen bond
        donor_pos = np.array([0.0, 0.0, 0.0])  # N
        hydrogen_pos = np.array([-0.1, 0.0, 0.0])  # H
        acceptor_pos = np.array([0.3, 0.0, 0.0])  # O
        antecedent_pos = np.array([0.4, 0.1, 0.0])  # C
        
        energy = calculate_hydrogen_bond_energy(
            donor_pos, hydrogen_pos, acceptor_pos, antecedent_pos
        )
        
        # Should be a negative value for favorable hydrogen bond
        self.assertLess(energy, 0.0)
        self.assertGreater(energy, -15.0)  # Reasonable range for hydrogen bonds (-1 to -15 kcal/mol)
    
    def test_hydrogen_bond_energy_too_close(self):
        """Test hydrogen bond energy for atoms that are too close."""
        # Atoms too close together
        donor_pos = np.array([0.0, 0.0, 0.0])
        hydrogen_pos = np.array([0.0, 0.0, 0.0])  # Same position
        acceptor_pos = np.array([0.0, 0.0, 0.0])
        antecedent_pos = np.array([0.0, 0.0, 0.0])
        
        energy = calculate_hydrogen_bond_energy(
            donor_pos, hydrogen_pos, acceptor_pos, antecedent_pos
        )
        
        # Should return 0 for unrealistic geometry
        self.assertEqual(energy, 0.0)
    
    def test_identify_hydrogen_bonds_simple(self):
        """Test hydrogen bond identification in a simple structure."""
        # Create a simple mock molecule with potential hydrogen bonds
        molecule = Mock()
        
        # Create atoms for two residues that could form a hydrogen bond
        atoms = []
        
        # Residue 1 (donor)
        atoms.extend([
            Mock(atom_name='N', residue_number=1, position=np.array([0.0, 0.0, 0.0])),
            Mock(atom_name='CA', residue_number=1, position=np.array([0.14, 0.0, 0.0])),
            Mock(atom_name='C', residue_number=1, position=np.array([0.21, 0.14, 0.0])),
            Mock(atom_name='O', residue_number=1, position=np.array([0.18, 0.25, 0.0])),
        ])
        
        # Residue 4 (acceptor) - far enough for hydrogen bond
        atoms.extend([
            Mock(atom_name='N', residue_number=4, position=np.array([0.0, 0.0, 0.6])),
            Mock(atom_name='CA', residue_number=4, position=np.array([0.14, 0.0, 0.6])),
            Mock(atom_name='C', residue_number=4, position=np.array([0.21, 0.14, 0.6])),
            Mock(atom_name='O', residue_number=4, position=np.array([0.25, 0.15, 0.6])),
        ])
        
        molecule.atoms = atoms
        
        hydrogen_bonds = identify_hydrogen_bonds(molecule)
        
        # Should be a list (may be empty for this simple test)
        self.assertIsInstance(hydrogen_bonds, list)


class TestSecondaryStructureAssignment(unittest.TestCase):
    """Test secondary structure assignment algorithms."""
    
    def create_test_molecule(self, structure_type='helix'):
        """Create a test molecule with known secondary structure."""
        molecule = Mock()
        atoms = []
        
        n_residues = 10
        
        for i in range(n_residues):
            res_num = i + 1
            
            if structure_type == 'helix':
                # Alpha-helix geometry
                z = i * 0.15  # 1.5 Å rise per residue
                theta = i * 100.0 * np.pi / 180.0  # 100° rotation
                radius = 0.23  # 2.3 Å radius
                
                x_offset = radius * np.cos(theta)
                y_offset = radius * np.sin(theta)
                
                n_pos = [x_offset - 0.05, y_offset, z]
                ca_pos = [x_offset, y_offset, z]
                c_pos = [x_offset + 0.05, y_offset + 0.03, z + 0.02]
                o_pos = [x_offset + 0.08, y_offset + 0.05, z + 0.01]
                
            elif structure_type == 'strand':
                # Beta-strand geometry (extended)
                n_pos = [i * 0.35, 0.0, 0.0]
                ca_pos = [i * 0.35 + 0.05, 0.0, 0.0]
                c_pos = [i * 0.35 + 0.10, 0.0, 0.0]
                o_pos = [i * 0.35 + 0.13, 0.05, 0.0]
                
            else:  # random coil
                # Random positions
                np.random.seed(42 + i)
                base_pos = np.random.normal(0, 0.1, 3)
                n_pos = base_pos + [-0.05, 0, 0]
                ca_pos = base_pos
                c_pos = base_pos + [0.05, 0.03, 0.02]
                o_pos = base_pos + [0.08, 0.05, 0.01]
            
            # Add atoms
            atoms.extend([
                Mock(atom_name='N', residue_number=res_num, residue_name='ALA', position=np.array(n_pos)),
                Mock(atom_name='CA', residue_number=res_num, residue_name='ALA', position=np.array(ca_pos)),
                Mock(atom_name='C', residue_number=res_num, residue_name='ALA', position=np.array(c_pos)),
                Mock(atom_name='O', residue_number=res_num, residue_name='ALA', position=np.array(o_pos)),
            ])
        
        molecule.atoms = atoms
        return molecule
    
    def test_assign_secondary_structure_helix(self):
        """Test secondary structure assignment for helical structure."""
        molecule = self.create_test_molecule('helix')
        assignments = assign_secondary_structure_dssp(molecule)
        
        # Should return assignments for all residues
        self.assertEqual(len(assignments), 10)
        
        # Should contain mostly helix or coil assignments
        unique_assignments = set(assignments)
        self.assertTrue(unique_assignments.issubset(set(SS_TYPES.keys())))
    
    def test_assign_secondary_structure_strand(self):
        """Test secondary structure assignment for strand structure."""
        molecule = self.create_test_molecule('strand')
        assignments = assign_secondary_structure_dssp(molecule)
        
        # Should return assignments for all residues
        self.assertEqual(len(assignments), 10)
        
        # All assignments should be valid SS types
        for assignment in assignments:
            self.assertIn(assignment, SS_TYPES.keys())


class TestSecondaryStructureAnalyzer(unittest.TestCase):
    """Test the SecondaryStructureAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SecondaryStructureAnalyzer()
    
    def create_mock_molecule(self, n_residues=5):
        """Create a mock molecule for testing."""
        molecule = Mock()
        atoms = []
        
        for i in range(n_residues):
            res_num = i + 1
            
            # Simple linear arrangement
            base_x = i * 0.35
            
            atoms.extend([
                Mock(atom_name='N', residue_number=res_num, residue_name='ALA', 
                     position=np.array([base_x, 0.0, 0.0])),
                Mock(atom_name='CA', residue_number=res_num, residue_name='ALA', 
                     position=np.array([base_x + 0.05, 0.0, 0.0])),
                Mock(atom_name='C', residue_number=res_num, residue_name='ALA', 
                     position=np.array([base_x + 0.10, 0.0, 0.0])),
                Mock(atom_name='O', residue_number=res_num, residue_name='ALA', 
                     position=np.array([base_x + 0.13, 0.05, 0.0])),
            ])
        
        molecule.atoms = atoms
        return molecule
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer, SecondaryStructureAnalyzer)
        self.assertEqual(len(self.analyzer.trajectory_data), 0)
        self.assertIsInstance(self.analyzer.structure_cutoffs, dict)
    
    def test_analyze_structure(self):
        """Test single structure analysis."""
        molecule = self.create_mock_molecule(n_residues=5)
        
        result = self.analyzer.analyze_structure(molecule, time_point=0.0)
        
        # Check result structure
        self.assertIn('time_point', result)
        self.assertIn('n_residues', result)
        self.assertIn('assignments', result)
        self.assertIn('residue_names', result)
        self.assertIn('counts', result)
        self.assertIn('percentages', result)
        self.assertIn('hydrogen_bonds', result)
        
        # Check values
        self.assertEqual(result['time_point'], 0.0)
        self.assertEqual(result['n_residues'], 5)
        self.assertEqual(len(result['assignments']), 5)
        self.assertEqual(len(result['residue_names']), 5)
        
        # Check that all assignments are valid
        for assignment in result['assignments']:
            self.assertIn(assignment, SS_TYPES.keys())
        
        # Check that percentages sum to 100
        total_percentage = sum(result['percentages'].values())
        self.assertAlmostEqual(total_percentage, 100.0, places=1)
    
    def test_analyze_trajectory_mock(self):
        """Test trajectory analysis with mock simulation."""
        # Create mock simulation
        simulation = Mock()
        simulation.molecule = self.create_mock_molecule(n_residues=3)
        simulation.dt = 0.1
        
        # Create mock trajectory
        n_frames = 10
        n_atoms = len(simulation.molecule.atoms)
        trajectory = []
        
        for frame in range(n_frames):
            frame_data = []
            for atom in simulation.molecule.atoms:
                # Add small random displacement
                new_pos = atom.position + np.random.normal(0, 0.01, 3)
                frame_data.append(new_pos)
            trajectory.append(frame_data)
        
        simulation.trajectory = trajectory
        
        # Analyze trajectory
        result = self.analyzer.analyze_trajectory(simulation, time_step=2)
        
        # Check that data was stored
        self.assertGreater(len(self.analyzer.trajectory_data), 0)
        self.assertIn('trajectory_data', result)
        self.assertIn('statistics', result)
    
    def test_export_functionality(self):
        """Test data export functionality."""
        # First analyze a structure
        molecule = self.create_mock_molecule(n_residues=3)
        self.analyzer.analyze_structure(molecule, time_point=0.0)
        
        # Add to trajectory data manually for testing
        result = self.analyzer.analyze_structure(molecule, time_point=1.0)
        self.analyzer.trajectory_data = [result]
        self.analyzer._calculate_trajectory_statistics()
        
        # Test CSV export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_csv:
            try:
                self.analyzer.export_timeline_data(tmp_csv.name, format='csv')
                self.assertTrue(os.path.exists(tmp_csv.name))
                self.assertGreater(os.path.getsize(tmp_csv.name), 0)
            finally:
                os.unlink(tmp_csv.name)
        
        # Test JSON export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_json:
            try:
                self.analyzer.export_timeline_data(tmp_json.name, format='json')
                self.assertTrue(os.path.exists(tmp_json.name))
                self.assertGreater(os.path.getsize(tmp_json.name), 0)
            finally:
                os.unlink(tmp_json.name)
    
    def test_statistics_calculation(self):
        """Test statistics calculation."""
        # Create and analyze multiple structures
        molecule = self.create_mock_molecule(n_residues=3)
        
        # Add multiple time points
        for i in range(5):
            result = self.analyzer.analyze_structure(molecule, time_point=i * 0.1)
            self.analyzer.trajectory_data.append(result)
        
        # Calculate statistics
        self.analyzer._calculate_trajectory_statistics()
        
        # Check statistics
        stats = self.analyzer.get_statistics()
        
        self.assertIn('trajectory_stats', stats)
        self.assertIn('residue_stats', stats)
        self.assertIn('structure_counts', stats)
        
        # Check trajectory stats
        traj_stats = stats['trajectory_stats']
        self.assertIn('n_frames', traj_stats)
        self.assertIn('n_residues', traj_stats)
        self.assertIn('avg_percentages', traj_stats)
        self.assertIn('std_percentages', traj_stats)
        
        self.assertEqual(traj_stats['n_frames'], 5)
        self.assertEqual(traj_stats['n_residues'], 3)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test plotting without trajectory data
        with self.assertRaises(ValueError):
            self.analyzer.plot_time_evolution()
        
        with self.assertRaises(ValueError):
            self.analyzer.plot_residue_timeline()
        
        with self.assertRaises(ValueError):
            self.analyzer.plot_structure_distribution()
        
        # Test export without trajectory data
        with self.assertRaises(ValueError):
            self.analyzer.export_timeline_data('test.csv')
        
        # Test invalid export format
        molecule = self.create_mock_molecule(n_residues=2)
        self.analyzer.trajectory_data = [self.analyzer.analyze_structure(molecule)]
        
        with self.assertRaises(ValueError):
            self.analyzer.export_timeline_data('test.xyz', format='invalid')


class TestFactoryFunction(unittest.TestCase):
    """Test the factory function for creating analyzers."""
    
    def test_create_secondary_structure_analyzer(self):
        """Test the factory function."""
        analyzer = create_secondary_structure_analyzer()
        
        self.assertIsInstance(analyzer, SecondaryStructureAnalyzer)
        
        # Test with custom parameters
        custom_cutoffs = {'hb_distance': 0.4, 'hb_energy': -0.3}
        analyzer_custom = create_secondary_structure_analyzer(
            structure_cutoffs=custom_cutoffs
        )
        
        self.assertIsInstance(analyzer_custom, SecondaryStructureAnalyzer)
        self.assertEqual(analyzer_custom.structure_cutoffs['hb_distance'], 0.4)


class TestSSTypes(unittest.TestCase):
    """Test secondary structure type definitions."""
    
    def test_ss_types_structure(self):
        """Test that SS_TYPES contains required information."""
        required_keys = ['name', 'color', 'priority']
        
        for ss_type, info in SS_TYPES.items():
            self.assertIsInstance(ss_type, str)
            self.assertEqual(len(ss_type), 1)  # Single character
            
            for key in required_keys:
                self.assertIn(key, info)
            
            self.assertIsInstance(info['name'], str)
            self.assertIsInstance(info['color'], str)
            self.assertIsInstance(info['priority'], int)
    
    def test_ss_types_completeness(self):
        """Test that all major secondary structure types are defined."""
        expected_types = ['H', 'E', 'C', 'T', 'G', 'B', 'I', 'S', '-']
        
        for ss_type in expected_types:
            self.assertIn(ss_type, SS_TYPES)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
