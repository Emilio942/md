"""
Unit tests for Radius of Gyration analysis module.

This test suite covers:
- Basic Rg calculations
- Center of mass calculations
- Segmental analysis
- Trajectory analysis
- Statistical calculations
- Data export functionality

Author: ProteinMD Development Team
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.radius_of_gyration import (
    calculate_center_of_mass,
    calculate_radius_of_gyration,
    calculate_segmental_rg,
    RadiusOfGyrationAnalyzer,
    create_rg_analyzer
)


class TestBasicCalculations(unittest.TestCase):
    """Test basic radius of gyration calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Simple 3-atom system with known properties
        self.positions_simple = np.array([
            [0.0, 0.0, 0.0],  # Origin
            [1.0, 0.0, 0.0],  # 1 unit along x
            [0.0, 1.0, 0.0]   # 1 unit along y
        ])
        self.masses_simple = np.array([1.0, 1.0, 1.0])  # Equal masses
        
        # Compact spherical system
        np.random.seed(42)
        self.positions_sphere = np.random.normal(0, 0.5, (20, 3))
        self.masses_sphere = np.ones(20)
        
        # Linear system
        self.positions_linear = np.array([[i, 0, 0] for i in range(10)], dtype=float)
        self.masses_linear = np.ones(10)
    
    def test_center_of_mass_calculation(self):
        """Test center of mass calculation."""
        # Simple 3-atom system
        com = calculate_center_of_mass(self.positions_simple, self.masses_simple)
        expected_com = np.array([1/3, 1/3, 0.0])
        np.testing.assert_array_almost_equal(com, expected_com, decimal=6)
        
        # Single atom
        single_pos = np.array([[1.0, 2.0, 3.0]])
        single_mass = np.array([5.0])
        com_single = calculate_center_of_mass(single_pos, single_mass)
        np.testing.assert_array_almost_equal(com_single, [1.0, 2.0, 3.0])
        
        # Different masses
        masses_weighted = np.array([1.0, 2.0, 3.0])
        com_weighted = calculate_center_of_mass(self.positions_simple, masses_weighted)
        expected_weighted = (1.0 * np.array([0, 0, 0]) + 
                           2.0 * np.array([1, 0, 0]) + 
                           3.0 * np.array([0, 1, 0])) / 6.0
        np.testing.assert_array_almost_equal(com_weighted, expected_weighted)
    
    def test_center_of_mass_edge_cases(self):
        """Test center of mass edge cases."""
        # Empty arrays
        com_empty = calculate_center_of_mass(np.array([]).reshape(0, 3), np.array([]))
        np.testing.assert_array_equal(com_empty, np.zeros(3))
        
        # Zero masses
        zero_masses = np.zeros(3)
        com_zero = calculate_center_of_mass(self.positions_simple, zero_masses)
        expected_geom_center = np.mean(self.positions_simple, axis=0)
        np.testing.assert_array_almost_equal(com_zero, expected_geom_center)
    
    def test_radius_of_gyration_calculation(self):
        """Test radius of gyration calculation."""
        # For a 3-atom system with equal masses at the corners of a triangle
        rg = calculate_radius_of_gyration(self.positions_simple, self.masses_simple)
        
        # Calculate expected Rg manually
        com = calculate_center_of_mass(self.positions_simple, self.masses_simple)
        distances_sq = np.sum((self.positions_simple - com)**2, axis=1)
        expected_rg = np.sqrt(np.mean(distances_sq))
        
        self.assertAlmostEqual(rg, expected_rg, places=6)
        
        # Single atom should have Rg = 0
        single_pos = np.array([[0.0, 0.0, 0.0]])
        single_mass = np.array([1.0])
        rg_single = calculate_radius_of_gyration(single_pos, single_mass)
        self.assertAlmostEqual(rg_single, 0.0, places=6)
    
    def test_radius_of_gyration_linear_system(self):
        """Test Rg for a linear system."""
        # Linear chain along x-axis
        rg_linear = calculate_radius_of_gyration(self.positions_linear, self.masses_linear)
        
        # For a uniform discrete chain of n points (0,1,2,...,n-1), Rg = sqrt((n^2-1)/12)
        # Chain has 10 points: 0, 1, 2, ..., 9
        n = 10
        expected_rg = np.sqrt((n**2 - 1) / 12)
        self.assertAlmostEqual(rg_linear, expected_rg, places=5)
    
    def test_segmental_analysis(self):
        """Test segmental radius of gyration calculation."""
        # Define segments
        segments = {
            'first_half': np.array([0, 1, 2, 3, 4]),
            'second_half': np.array([5, 6, 7, 8, 9]),
            'center': np.array([4, 5])
        }
        
        segmental_rg = calculate_segmental_rg(self.positions_linear, self.masses_linear, segments)
        
        # Check that all segments are present
        self.assertEqual(set(segmental_rg.keys()), set(segments.keys()))
        
        # Check that all values are positive
        for rg_value in segmental_rg.values():
            self.assertGreaterEqual(rg_value, 0.0)
        
        # Test empty segment
        empty_segments = {'empty': np.array([], dtype=int)}
        empty_rg = calculate_segmental_rg(self.positions_linear, self.masses_linear, empty_segments)
        self.assertEqual(empty_rg['empty'], 0.0)
    
    def test_mismatched_arrays(self):
        """Test error handling for mismatched position and mass arrays."""
        positions = np.array([[0, 0, 0], [1, 1, 1]])
        masses = np.array([1.0])  # Wrong length
        
        with self.assertRaises(ValueError):
            calculate_center_of_mass(positions, masses)
        
        with self.assertRaises(ValueError):
            calculate_radius_of_gyration(positions, masses)


class TestRadiusOfGyrationAnalyzer(unittest.TestCase):
    """Test the RadiusOfGyrationAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = RadiusOfGyrationAnalyzer()
        
        # Create test data
        np.random.seed(42)
        self.n_atoms = 50
        self.n_frames = 20
        
        # Base structure
        self.base_positions = np.random.normal(0, 1.0, (self.n_atoms, 3))
        self.masses = np.random.uniform(12.0, 16.0, self.n_atoms)
        
        # Create trajectory
        self.trajectory_positions = []
        for i in range(self.n_frames):
            positions = self.base_positions + np.random.normal(0, 0.1, (self.n_atoms, 3))
            self.trajectory_positions.append(positions)
        self.trajectory_positions = np.array(self.trajectory_positions)
        
        self.time_points = np.arange(self.n_frames) * 0.1
        
        # Define segments
        self.segments = {
            'N_terminal': np.arange(0, 15),
            'Core': np.arange(15, 35),
            'C_terminal': np.arange(35, 50)
        }
    
    def test_single_structure_analysis(self):
        """Test analysis of a single structure."""
        result = self.analyzer.analyze_structure(self.base_positions, self.masses)
        
        # Check required keys
        required_keys = ['overall_rg', 'center_of_mass', 'segmental_rg', 'n_atoms', 'total_mass']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check data types and values
        self.assertIsInstance(result['overall_rg'], float)
        self.assertGreater(result['overall_rg'], 0.0)
        self.assertEqual(result['n_atoms'], self.n_atoms)
        self.assertAlmostEqual(result['total_mass'], np.sum(self.masses))
        self.assertEqual(len(result['center_of_mass']), 3)
        
        # Should have no segmental data without segments defined
        self.assertEqual(len(result['segmental_rg']), 0)
    
    def test_segment_definition(self):
        """Test segment definition and analysis."""
        self.analyzer.define_segments(self.segments)
        
        result = self.analyzer.analyze_structure(self.base_positions, self.masses)
        
        # Check segmental results
        self.assertEqual(len(result['segmental_rg']), len(self.segments))
        for segment_name in self.segments.keys():
            self.assertIn(segment_name, result['segmental_rg'])
            self.assertGreater(result['segmental_rg'][segment_name], 0.0)
    
    def test_trajectory_analysis(self):
        """Test trajectory analysis."""
        self.analyzer.define_segments(self.segments)
        
        results = self.analyzer.analyze_trajectory(
            self.trajectory_positions, self.masses, self.time_points
        )
        
        # Check trajectory data
        self.assertEqual(len(self.analyzer.trajectory_data), self.n_frames)
        
        # Check statistics
        self.assertIn('overall_statistics', results)
        self.assertIn('segmental_statistics', results)
        
        stats = results['overall_statistics']
        required_stats = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
        for stat in required_stats:
            self.assertIn(stat, stats)
            self.assertIsInstance(stats[stat], float)
        
        # Check segmental statistics
        seg_stats = results['segmental_statistics']
        self.assertEqual(len(seg_stats), len(self.segments))
        for segment_name in self.segments.keys():
            self.assertIn(segment_name, seg_stats)
    
    def test_trajectory_time_mismatch(self):
        """Test error handling for mismatched trajectory and time arrays."""
        wrong_time_points = np.arange(self.n_frames - 5) * 0.1
        
        with self.assertRaises(ValueError):
            self.analyzer.analyze_trajectory(
                self.trajectory_positions, self.masses, wrong_time_points
            )
    
    def test_empty_trajectory_statistics(self):
        """Test statistics calculation with no trajectory data."""
        stats = self.analyzer.get_trajectory_statistics()
        self.assertEqual(stats, {})


class TestDataExport(unittest.TestCase):
    """Test data export functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = RadiusOfGyrationAnalyzer()
        
        # Create minimal test data
        np.random.seed(42)
        positions = np.random.normal(0, 1.0, (10, 3))
        masses = np.ones(10)
        
        segments = {'test_segment': np.array([0, 1, 2, 3, 4])}
        self.analyzer.define_segments(segments)
        
        # Create trajectory data
        trajectory_positions = np.array([positions + np.random.normal(0, 0.1, (10, 3)) 
                                       for _ in range(5)])
        time_points = np.arange(5) * 0.1
        
        self.analyzer.analyze_trajectory(trajectory_positions, masses, time_points)
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_csv_export(self):
        """Test CSV export functionality."""
        csv_path = os.path.join(self.temp_dir, 'test_rg.csv')
        
        # Test CSV export
        self.analyzer.export_data(csv_path, format='csv')
        self.assertTrue(os.path.exists(csv_path))
        
        # Read and verify content
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        # Should have header + data lines
        self.assertGreater(len(lines), 1)
        
        # Check header
        header = lines[0].strip().split(',')
        expected_cols = ['time', 'overall_rg', 'center_of_mass_x', 'center_of_mass_y', 
                        'center_of_mass_z', 'n_atoms', 'total_mass', 'rg_test_segment']
        for col in expected_cols:
            self.assertIn(col, header)
    
    def test_json_export(self):
        """Test JSON export functionality."""
        json_path = os.path.join(self.temp_dir, 'test_rg.json')
        
        # Test JSON export
        self.analyzer.export_data(json_path, format='json')
        self.assertTrue(os.path.exists(json_path))
        
        # Read and verify content
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Check structure
        required_keys = ['trajectory_data', 'single_frame_data', 'segment_definitions', 'statistics']
        for key in required_keys:
            self.assertIn(key, data)
        
        # Check trajectory data
        self.assertEqual(len(data['trajectory_data']), 5)
    
    def test_unsupported_format(self):
        """Test error handling for unsupported export formats."""
        with self.assertRaises(ValueError):
            self.analyzer.export_data('test.xyz', format='xyz')


class TestAnalyzerCreation(unittest.TestCase):
    """Test analyzer creation utilities."""
    
    def test_create_rg_analyzer(self):
        """Test create_rg_analyzer function."""
        analyzer = create_rg_analyzer()
        self.assertIsInstance(analyzer, RadiusOfGyrationAnalyzer)
        self.assertEqual(len(analyzer.segment_definitions), 0)
        
        # Test with segments
        segments = {'test': np.array([0, 1, 2])}
        analyzer_with_segments = create_rg_analyzer(segments)
        self.assertEqual(len(analyzer_with_segments.segment_definitions), 1)
        self.assertIn('test', analyzer_with_segments.segment_definitions)


class TestPhysicalRealism(unittest.TestCase):
    """Test that the calculations produce physically realistic results."""
    
    def test_protein_like_rg_values(self):
        """Test that Rg values are in realistic range for protein-like structures."""
        # Create a protein-like structure (roughly spherical, ~2 nm diameter)
        np.random.seed(42)
        n_atoms = 100
        positions = np.random.normal(0, 0.8, (n_atoms, 3))  # ~1.6 nm diameter
        masses = np.random.uniform(12.0, 16.0, n_atoms)
        
        rg = calculate_radius_of_gyration(positions, masses)
        
        # Typical protein Rg values are 0.5-5 nm
        self.assertGreater(rg, 0.3)
        self.assertLess(rg, 3.0)
    
    def test_rg_scaling_with_size(self):
        """Test that Rg scales appropriately with structure size."""
        # Create two structures of different sizes
        base_positions = np.random.normal(0, 1.0, (50, 3))
        masses = np.ones(50)
        
        rg1 = calculate_radius_of_gyration(base_positions, masses)
        rg2 = calculate_radius_of_gyration(base_positions * 2, masses)  # Scaled up 2x
        
        # Rg should scale linearly with size
        self.assertAlmostEqual(rg2 / rg1, 2.0, places=1)
    
    def test_rg_mass_independence(self):
        """Test that Rg is independent of total mass (only mass distribution matters)."""
        positions = np.random.normal(0, 1.0, (20, 3))
        masses1 = np.ones(20)
        masses2 = np.ones(20) * 5.0  # 5x heavier
        
        rg1 = calculate_radius_of_gyration(positions, masses1)
        rg2 = calculate_radius_of_gyration(positions, masses2)
        
        # Rg should be the same regardless of absolute mass scale
        self.assertAlmostEqual(rg1, rg2, places=6)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
