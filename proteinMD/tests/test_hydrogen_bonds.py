"""
Tests for hydrogen bond analysis module.

This module contains comprehensive tests for hydrogen bond detection,
trajectory analysis, and statistical calculations.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
import json

# Import the module being tested
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from analysis.hydrogen_bonds import (
    HydrogenBondDetector, HydrogenBondAnalyzer, HydrogenBond,
    analyze_hydrogen_bonds, quick_hydrogen_bond_summary
)


class MockAtom:
    """Mock atom class for testing."""
    
    def __init__(self, atom_id, atom_name, element, residue_id=0):
        self.atom_id = atom_id
        self.atom_name = atom_name
        self.element = element
        self.residue_id = residue_id


class TestHydrogenBondDetector(unittest.TestCase):
    """Test hydrogen bond detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = HydrogenBondDetector()
        
        # Create mock atoms for testing
        self.atoms = [
            MockAtom(0, 'N', 'N', 0),    # Donor (backbone nitrogen)
            MockAtom(1, 'H', 'H', 0),    # Hydrogen
            MockAtom(2, 'CA', 'C', 0),   # Carbon (not involved)
            MockAtom(3, 'C', 'C', 0),    # Carbon (carbonyl)
            MockAtom(4, 'O', 'O', 0),    # Acceptor (carbonyl oxygen)
            MockAtom(5, 'N', 'N', 1),    # Donor (next residue)
            MockAtom(6, 'H', 'H', 1),    # Hydrogen
        ]
    
    def test_detector_initialization(self):
        """Test detector initialization with custom parameters."""
        custom_detector = HydrogenBondDetector(
            max_distance=3.0,
            min_angle=130.0,
            max_h_distance=2.0
        )
        
        self.assertEqual(custom_detector.max_distance, 3.0)
        self.assertEqual(custom_detector.min_angle, 130.0)
        self.assertEqual(custom_detector.max_h_distance, 2.0)
    
    def test_angle_calculation(self):
        """Test angle calculation between three points."""
        # Test perfect linear arrangement (180°)
        donor = np.array([0.0, 0.0, 0.0])
        hydrogen = np.array([1.0, 0.0, 0.0])
        acceptor = np.array([2.0, 0.0, 0.0])
        
        angle = self.detector.calculate_angle(donor, hydrogen, acceptor)
        self.assertAlmostEqual(angle, 180.0, places=1)
        
        # Test right angle (90°)
        acceptor = np.array([1.0, 1.0, 0.0])
        angle = self.detector.calculate_angle(donor, hydrogen, acceptor)
        self.assertAlmostEqual(angle, 90.0, places=1)
    
    def test_bond_strength_classification(self):
        """Test hydrogen bond strength classification."""
        # Very strong bond
        strength = self.detector.classify_bond_strength(2.3, 175.0)
        self.assertEqual(strength, 'very_strong')
        
        # Strong bond
        strength = self.detector.classify_bond_strength(2.7, 155.0)
        self.assertEqual(strength, 'strong')
        
        # Moderate bond
        strength = self.detector.classify_bond_strength(3.0, 135.0)
        self.assertEqual(strength, 'moderate')
        
        # Weak bond
        strength = self.detector.classify_bond_strength(3.4, 125.0)
        self.assertEqual(strength, 'weak')
        
        # Very weak bond
        strength = self.detector.classify_bond_strength(3.6, 115.0)
        self.assertEqual(strength, 'very_weak')
    
    def test_bond_type_determination(self):
        """Test hydrogen bond type classification."""
        # Intra-residue
        bond_type = self.detector.determine_bond_type(5, 5)
        self.assertEqual(bond_type, 'intra_residue')
        
        # Adjacent residues
        bond_type = self.detector.determine_bond_type(5, 6)
        self.assertEqual(bond_type, 'adjacent')
        
        # Short range
        bond_type = self.detector.determine_bond_type(5, 8)
        self.assertEqual(bond_type, 'short_range')
        
        # Long range
        bond_type = self.detector.determine_bond_type(5, 15)
        self.assertEqual(bond_type, 'long_range')
    
    def test_hydrogen_bond_detection_ideal_case(self):
        """Test detection of ideal hydrogen bond."""
        # Ideal hydrogen bond geometry: N-H...O
        positions = np.array([
            [0.0, 0.0, 0.0],  # N (donor)
            [1.0, 0.0, 0.0],  # H
            [1.5, 0.5, 0.0],  # CA (not involved)
            [2.0, 0.0, 0.0],  # C
            [3.0, 0.0, 0.0],  # O (acceptor) - perfect geometry
            [4.0, 0.0, 0.0],  # N (next residue)
            [5.0, 0.0, 0.0],  # H
        ])
        
        bonds = self.detector.detect_hydrogen_bonds(self.atoms, positions)
        
        # Should detect one hydrogen bond
        self.assertEqual(len(bonds), 1)
        
        bond = bonds[0]
        self.assertEqual(bond.donor_atom_idx, 0)
        self.assertEqual(bond.hydrogen_idx, 1)
        self.assertEqual(bond.acceptor_atom_idx, 4)
        self.assertAlmostEqual(bond.distance, 3.0, places=1)
        self.assertAlmostEqual(bond.angle, 180.0, places=0)
    
    def test_no_hydrogen_bond_detection(self):
        """Test case where no hydrogen bonds should be detected."""
        # Too far apart
        positions = np.array([
            [0.0, 0.0, 0.0],  # N
            [1.0, 0.0, 0.0],  # H
            [1.5, 0.5, 0.0],  # CA
            [2.0, 0.0, 0.0],  # C
            [6.0, 0.0, 0.0],  # O (too far)
            [7.0, 0.0, 0.0],  # N
            [8.0, 0.0, 0.0],  # H
        ])
        
        bonds = self.detector.detect_hydrogen_bonds(self.atoms, positions)
        self.assertEqual(len(bonds), 0)
    
    def test_hydrogen_bond_detection_poor_angle(self):
        """Test case with poor hydrogen bond angle."""
        # Poor angle (perpendicular)
        positions = np.array([
            [0.0, 0.0, 0.0],  # N
            [1.0, 0.0, 0.0],  # H
            [1.5, 0.5, 0.0],  # CA
            [2.0, 0.0, 0.0],  # C
            [1.0, 2.0, 0.0],  # O (perpendicular, poor angle)
            [4.0, 0.0, 0.0],  # N
            [5.0, 0.0, 0.0],  # H
        ])
        
        bonds = self.detector.detect_hydrogen_bonds(self.atoms, positions)
        # Should not detect bond due to poor angle
        self.assertEqual(len(bonds), 0)


class TestHydrogenBondAnalyzer(unittest.TestCase):
    """Test hydrogen bond trajectory analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = HydrogenBondAnalyzer()
        
        # Create mock atoms
        self.atoms = [
            MockAtom(0, 'N', 'N', 0),
            MockAtom(1, 'H', 'H', 0),
            MockAtom(2, 'O', 'O', 1),
            MockAtom(3, 'N', 'N', 2),
            MockAtom(4, 'H', 'H', 2),
            MockAtom(5, 'O', 'O', 3),
        ]
        
        # Create test trajectory
        self.trajectory = self._create_test_trajectory()
    
    def _create_test_trajectory(self, n_frames=10):
        """Create a test trajectory with hydrogen bonds."""
        trajectory = []
        
        for frame in range(n_frames):
            # Oscillating hydrogen bond
            oscillation = np.sin(frame * np.pi / 5) * 0.5
            
            positions = np.array([
                [0.0, 0.0, 0.0],                    # N
                [1.0, 0.0, 0.0],                    # H
                [2.5 + oscillation, 0.0, 0.0],     # O (oscillating distance)
                [4.0, 0.0, 0.0],                    # N
                [5.0, 0.0, 0.0],                    # H
                [6.5, 0.0, 0.0],                    # O (stable)
            ])
            
            trajectory.append(positions)
        
        return np.array(trajectory)
    
    def test_trajectory_analysis(self):
        """Test basic trajectory analysis."""
        self.analyzer.analyze_trajectory(self.atoms, self.trajectory)
        
        # Check that analysis was performed
        self.assertGreater(len(self.analyzer.trajectory_bonds), 0)
        self.assertTrue(bool(self.analyzer.bond_statistics))
        self.assertTrue(bool(self.analyzer.lifetime_data))
    
    def test_statistics_calculation(self):
        """Test statistics calculation."""
        self.analyzer.analyze_trajectory(self.atoms, self.trajectory)
        stats = self.analyzer.get_summary_statistics()
        
        # Check required statistics are present
        self.assertIn('mean_bonds_per_frame', stats)
        self.assertIn('bond_type_distribution', stats)
        self.assertIn('bond_strength_distribution', stats)
        self.assertIn('lifetime_statistics', stats)
        
        # Check values are reasonable
        self.assertGreaterEqual(stats['mean_bonds_per_frame'], 0)
        self.assertGreaterEqual(stats['max_bonds_per_frame'], 0)
    
    def test_empty_trajectory_handling(self):
        """Test handling of empty trajectory."""
        empty_trajectory = np.array([]).reshape(0, len(self.atoms), 3)
        
        self.analyzer.analyze_trajectory(self.atoms, empty_trajectory)
        
        # Should handle gracefully
        self.assertEqual(len(self.analyzer.trajectory_bonds), 0)
    
    def test_statistics_without_analysis(self):
        """Test that requesting statistics before analysis raises error."""
        with self.assertRaises(ValueError):
            self.analyzer.get_summary_statistics()
    
    def test_plotting_without_analysis(self):
        """Test that plotting before analysis raises error."""
        with self.assertRaises(ValueError):
            self.analyzer.plot_bond_evolution()
        
        with self.assertRaises(ValueError):
            self.analyzer.plot_residue_network()


class TestDataExport(unittest.TestCase):
    """Test data export functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = HydrogenBondAnalyzer()
        
        # Create simple test data
        self.atoms = [
            MockAtom(0, 'N', 'N', 0),
            MockAtom(1, 'H', 'H', 0),
            MockAtom(2, 'O', 'O', 1),
        ]
        
        # Simple trajectory with consistent hydrogen bond
        self.trajectory = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        ])
        
        self.analyzer.analyze_trajectory(self.atoms, self.trajectory)
    
    def test_csv_export(self):
        """Test CSV export functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            self.analyzer.export_statistics_csv(csv_path)
            
            # Check file was created and has content
            self.assertTrue(os.path.exists(csv_path))
            self.assertGreater(os.path.getsize(csv_path), 0)
            
            # Check CSV structure
            with open(csv_path, 'r') as f:
                header = f.readline().strip()
                self.assertIn('frame', header)
                self.assertIn('donor_atom', header)
                self.assertIn('acceptor_atom', header)
        
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_json_export(self):
        """Test JSON export functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            self.analyzer.export_lifetime_analysis(json_path)
            
            # Check file was created and has content
            self.assertTrue(os.path.exists(json_path))
            self.assertGreater(os.path.getsize(json_path), 0)
            
            # Check JSON structure
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.assertIn('summary', data)
                
                # Should have at least one bond entry
                bond_keys = [k for k in data.keys() if k.startswith('bond_')]
                self.assertGreater(len(bond_keys), 0)
        
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)
    
    def test_export_without_analysis(self):
        """Test that export without analysis raises error."""
        empty_analyzer = HydrogenBondAnalyzer()
        
        with tempfile.NamedTemporaryFile(suffix='.csv') as f:
            with self.assertRaises(ValueError):
                empty_analyzer.export_statistics_csv(f.name)
        
        with tempfile.NamedTemporaryFile(suffix='.json') as f:
            with self.assertRaises(ValueError):
                empty_analyzer.export_lifetime_analysis(f.name)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for convenience."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.atoms = [
            MockAtom(0, 'N', 'N', 0),
            MockAtom(1, 'H', 'H', 0),
            MockAtom(2, 'O', 'O', 1),
        ]
        
        self.trajectory = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        ])
    
    def test_analyze_hydrogen_bonds_function(self):
        """Test convenience analysis function."""
        analyzer = analyze_hydrogen_bonds(self.atoms, self.trajectory)
        
        self.assertIsInstance(analyzer, HydrogenBondAnalyzer)
        self.assertGreater(len(analyzer.trajectory_bonds), 0)
    
    def test_analyze_with_custom_parameters(self):
        """Test analysis with custom detector parameters."""
        custom_params = {
            'max_distance': 4.0,
            'min_angle': 110.0,
            'max_h_distance': 3.0
        }
        
        analyzer = analyze_hydrogen_bonds(
            self.atoms, self.trajectory, detector_params=custom_params
        )
        
        self.assertEqual(analyzer.detector.max_distance, 4.0)
        self.assertEqual(analyzer.detector.min_angle, 110.0)
        self.assertEqual(analyzer.detector.max_h_distance, 3.0)
    
    def test_quick_summary_function(self):
        """Test quick summary function."""
        summary = quick_hydrogen_bond_summary(self.atoms, self.trajectory)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('mean_bonds_per_frame', summary)
        self.assertIn('bond_type_distribution', summary)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_single_atom_system(self):
        """Test system with only one atom."""
        detector = HydrogenBondDetector()
        atoms = [MockAtom(0, 'N', 'N', 0)]
        positions = np.array([[0.0, 0.0, 0.0]])
        
        bonds = detector.detect_hydrogen_bonds(atoms, positions)
        self.assertEqual(len(bonds), 0)
    
    def test_no_hydrogen_atoms(self):
        """Test system without hydrogen atoms."""
        detector = HydrogenBondDetector()
        atoms = [
            MockAtom(0, 'N', 'N', 0),
            MockAtom(1, 'O', 'O', 0),
            MockAtom(2, 'C', 'C', 0),
        ]
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        
        bonds = detector.detect_hydrogen_bonds(atoms, positions)
        self.assertEqual(len(bonds), 0)
    
    def test_invalid_geometry(self):
        """Test with invalid geometric arrangements."""
        detector = HydrogenBondDetector()
        
        # Test with identical positions (should not crash)
        donor = np.array([0.0, 0.0, 0.0])
        hydrogen = np.array([0.0, 0.0, 0.0])  # Same as donor
        acceptor = np.array([1.0, 0.0, 0.0])
        
        # Should handle gracefully (may return NaN or 0)
        angle = detector.calculate_angle(donor, hydrogen, acceptor)
        self.assertTrue(np.isfinite(angle) or np.isnan(angle))
    
    def test_large_system_performance(self):
        """Test performance with larger system."""
        # Create larger system for performance test
        n_atoms = 100
        atoms = []
        for i in range(n_atoms):
            element = 'N' if i % 3 == 0 else ('H' if i % 3 == 1 else 'O')
            atoms.append(MockAtom(i, element, element, i // 10))
        
        # Random positions
        np.random.seed(42)  # For reproducible tests
        positions = np.random.random((n_atoms, 3)) * 10.0
        
        detector = HydrogenBondDetector()
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        bonds = detector.detect_hydrogen_bonds(atoms, positions)
        end_time = time.time()
        
        # Should complete within 5 seconds
        self.assertLess(end_time - start_time, 5.0)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)
