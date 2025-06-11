"""
Test suite for the RMSD analysis module.

This module tests all functionality of the RMSD analysis including
core calculations, trajectory analysis, and visualization.
"""

import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.rmsd import (
    RMSDAnalyzer, calculate_rmsd, align_structures, kabsch_algorithm,
    create_rmsd_analyzer
)

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRMSDCalculations(unittest.TestCase):
    """Test core RMSD calculation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test coordinate sets
        np.random.seed(42)  # For reproducible tests
        
        # Simple test case: identity transformation
        self.coords_identical = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Translated coordinates
        self.coords_translated = self.coords_identical + np.array([2.0, 3.0, 1.0])
        
        # Rotated coordinates (90 degrees around z-axis)
        rotation_z_90 = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ])
        self.coords_rotated = self.coords_identical @ rotation_z_90.T
        
        # Random coordinates for noise testing
        self.coords_random = np.random.random((10, 3)) * 5.0
        self.coords_noisy = self.coords_random + np.random.normal(0, 0.1, (10, 3))
    
    def test_rmsd_identical_structures(self):
        """Test RMSD between identical structures."""
        rmsd = calculate_rmsd(self.coords_identical, self.coords_identical)
        self.assertAlmostEqual(rmsd, 0.0, places=10)
    
    def test_rmsd_translated_structures(self):
        """Test RMSD between translated structures (should be 0 with alignment)."""
        rmsd_with_align = calculate_rmsd(self.coords_identical, self.coords_translated, align=True)
        rmsd_without_align = calculate_rmsd(self.coords_identical, self.coords_translated, align=False)
        
        # With alignment, RMSD should be ~0
        self.assertAlmostEqual(rmsd_with_align, 0.0, places=10)
        
        # Without alignment, RMSD should be the translation distance
        expected_rmsd = np.sqrt(np.sum([2.0**2, 3.0**2, 1.0**2]))
        self.assertAlmostEqual(rmsd_without_align, expected_rmsd, places=10)
    
    def test_rmsd_rotated_structures(self):
        """Test RMSD between rotated structures (should be 0 with alignment)."""
        rmsd = calculate_rmsd(self.coords_identical, self.coords_rotated, align=True)
        self.assertAlmostEqual(rmsd, 0.0, places=8)  # Allow for numerical precision
    
    def test_kabsch_algorithm(self):
        """Test Kabsch algorithm for optimal alignment."""
        rotation_matrix, aligned_coords = kabsch_algorithm(
            self.coords_translated, self.coords_identical
        )
        
        # Check that rotation matrix is orthogonal
        should_be_identity = rotation_matrix @ rotation_matrix.T
        np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-10)
        
        # Check that determinant is 1 (proper rotation)
        self.assertAlmostEqual(np.linalg.det(rotation_matrix), 1.0, places=10)
        
        # Check that aligned coordinates match the target
        np.testing.assert_allclose(aligned_coords, self.coords_identical, atol=1e-10)
    
    def test_align_structures(self):
        """Test structure alignment function."""
        aligned = align_structures(self.coords_translated, self.coords_identical)
        np.testing.assert_allclose(aligned, self.coords_identical, atol=1e-10)
    
    def test_rmsd_input_validation(self):
        """Test input validation for RMSD calculations."""
        # Test mismatched shapes
        coords1 = np.random.random((5, 3))
        coords2 = np.random.random((3, 3))
        
        with self.assertRaises(ValueError):
            calculate_rmsd(coords1, coords2)
        
        # Test wrong dimensions
        coords_wrong_dim = np.random.random((5, 2))
        
        with self.assertRaises(ValueError):
            calculate_rmsd(coords_wrong_dim, coords1)
    
    def test_rmsd_empty_coordinates(self):
        """Test RMSD with empty coordinate arrays."""
        empty_coords = np.array([]).reshape(0, 3)
        rmsd = calculate_rmsd(empty_coords, empty_coords)
        self.assertEqual(rmsd, 0.0)


class TestRMSDAnalyzer(unittest.TestCase):
    """Test the RMSDAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create reference structure
        self.reference = np.random.random((20, 3)) * 10.0
        
        # Create test trajectory with gradual drift
        self.trajectory = []
        self.n_frames = 25
        
        for i in range(self.n_frames):
            noise = np.random.normal(0, 0.05, (20, 3))
            drift = np.array([0.01 * i, 0.005 * i, 0.008 * i])
            frame = self.reference + noise + drift
            self.trajectory.append(frame)
        
        self.times = np.arange(self.n_frames) * 0.1  # 0.1 ps timesteps
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_analyzer_initialization(self):
        """Test RMSD analyzer initialization."""
        analyzer = RMSDAnalyzer(reference_structure=self.reference)
        
        self.assertEqual(analyzer.atom_selection, 'backbone')
        np.testing.assert_array_equal(analyzer.reference_structure, self.reference)
        self.assertTrue(analyzer.align_structures)
        self.assertEqual(len(analyzer.trajectory_names), 0)
    
    def test_set_reference(self):
        """Test setting reference structure."""
        analyzer = RMSDAnalyzer()
        self.assertIsNone(analyzer.reference_structure)
        
        analyzer.set_reference(self.reference)
        np.testing.assert_array_equal(analyzer.reference_structure, self.reference)
    
    def test_trajectory_rmsd_calculation(self):
        """Test RMSD calculation for trajectory."""
        analyzer = RMSDAnalyzer(reference_structure=self.reference)
        
        times, rmsd_values = analyzer.calculate_trajectory_rmsd(
            self.trajectory, self.times, "test_traj"
        )
        
        # Check data was stored
        self.assertIn("test_traj", analyzer.rmsd_data)
        self.assertIn("test_traj", analyzer.time_data)
        self.assertIn("test_traj", analyzer.trajectory_names)
        
        # Check data shapes
        self.assertEqual(len(rmsd_values), self.n_frames)
        self.assertEqual(len(times), self.n_frames)
        
        # Check RMSD values are reasonable
        self.assertTrue(all(rmsd >= 0 for rmsd in rmsd_values))
        self.assertLess(rmsd_values[0], rmsd_values[-1])  # Should increase due to drift
    
    def test_trajectory_rmsd_without_reference(self):
        """Test trajectory RMSD when no reference is set (uses first frame)."""
        analyzer = RMSDAnalyzer()
        
        times, rmsd_values = analyzer.calculate_trajectory_rmsd(
            self.trajectory, self.times, "auto_ref"
        )
        
        # First frame should have RMSD ≈ 0
        self.assertAlmostEqual(rmsd_values[0], 0.0, places=10)
        
        # Reference should now be set
        self.assertIsNotNone(analyzer.reference_structure)
        np.testing.assert_array_equal(analyzer.reference_structure, self.trajectory[0])
    
    def test_pairwise_rmsd_calculation(self):
        """Test pairwise RMSD matrix calculation."""
        analyzer = RMSDAnalyzer()
        
        # Use a subset of trajectory frames
        structures = [self.trajectory[0], self.trajectory[5], self.trajectory[10], self.trajectory[-1]]
        names = ["Frame_0", "Frame_5", "Frame_10", "Frame_24"]
        
        rmsd_matrix = analyzer.calculate_pairwise_rmsd(structures, names)
        
        # Check matrix properties
        self.assertEqual(rmsd_matrix.shape, (4, 4))
        
        # Diagonal should be zero
        np.testing.assert_allclose(np.diag(rmsd_matrix), 0.0, atol=1e-10)
        
        # Matrix should be symmetric
        np.testing.assert_allclose(rmsd_matrix, rmsd_matrix.T, atol=1e-10)
        
        # RMSD values should increase with frame distance
        self.assertLess(rmsd_matrix[0, 1], rmsd_matrix[0, 3])
    
    def test_running_average(self):
        """Test running average calculation."""
        analyzer = RMSDAnalyzer(reference_structure=self.reference)
        analyzer.calculate_trajectory_rmsd(self.trajectory, self.times, "test")
        
        window_size = 5
        avg_times, avg_rmsd = analyzer.calculate_running_average(
            window_size=window_size, trajectory_name="test"
        )
        
        expected_length = self.n_frames - window_size + 1
        self.assertEqual(len(avg_times), expected_length)
        self.assertEqual(len(avg_rmsd), expected_length)
        
        # Averaged values should be smoother (less variation)
        original_std = np.std(analyzer.rmsd_data["test"])
        averaged_std = np.std(avg_rmsd)
        self.assertLess(averaged_std, original_std)
    
    def test_statistics_calculation(self):
        """Test statistics calculation."""
        analyzer = RMSDAnalyzer(reference_structure=self.reference)
        analyzer.calculate_trajectory_rmsd(self.trajectory, self.times, "test")
        
        stats = analyzer.get_statistics("test")
        
        # Check required statistics are present
        required_stats = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'n_frames']
        for stat in required_stats:
            self.assertIn(stat, stats)
        
        # Check logical relationships
        self.assertLessEqual(stats['min'], stats['mean'])
        self.assertLessEqual(stats['mean'], stats['max'])
        self.assertLessEqual(stats['q25'], stats['median'])
        self.assertLessEqual(stats['median'], stats['q75'])
        self.assertEqual(stats['n_frames'], self.n_frames)
    
    def test_plot_time_series(self):
        """Test RMSD time series plotting."""
        analyzer = RMSDAnalyzer(reference_structure=self.reference)
        analyzer.calculate_trajectory_rmsd(self.trajectory, self.times, "test")
        
        # Test plotting
        fig = analyzer.plot_rmsd_time_series()
        
        # Check figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check axes were created
        self.assertEqual(len(fig.axes), 2)
    
    def test_plot_pairwise_matrix(self):
        """Test pairwise RMSD matrix plotting."""
        analyzer = RMSDAnalyzer()
        
        structures = [self.trajectory[0], self.trajectory[10], self.trajectory[-1]]
        names = ["Start", "Middle", "End"]
        
        analyzer.calculate_pairwise_rmsd(structures, names)
        fig = analyzer.plot_pairwise_rmsd_matrix()
        
        # Check figure was created
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 2)  # Main plot + colorbar
    
    def test_export_data(self):
        """Test data export functionality."""
        analyzer = RMSDAnalyzer(reference_structure=self.reference)
        analyzer.calculate_trajectory_rmsd(self.trajectory, self.times, "test")
        
        output_file = os.path.join(self.temp_dir, "test_rmsd_export.csv")
        analyzer.export_data(output_file)
        
        # Check file was created
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)
        
        # Check file content
        with open(output_file, 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 1)  # Header + data
            self.assertIn('test_time', lines[0])  # Check header
            self.assertIn('test_rmsd', lines[0])


class TestCreateRMSDAnalyzer(unittest.TestCase):
    """Test the convenience function for creating RMSD analyzers."""
    
    def test_create_rmsd_analyzer_default(self):
        """Test creating analyzer with default settings."""
        analyzer = create_rmsd_analyzer()
        
        self.assertIsInstance(analyzer, RMSDAnalyzer)
        self.assertIsNone(analyzer.reference_structure)
        self.assertEqual(analyzer.atom_selection, 'backbone')
    
    def test_create_rmsd_analyzer_with_reference(self):
        """Test creating analyzer with reference structure."""
        reference = np.random.random((10, 3))
        analyzer = create_rmsd_analyzer(reference_structure=reference)
        
        self.assertIsInstance(analyzer, RMSDAnalyzer)
        np.testing.assert_array_equal(analyzer.reference_structure, reference)
    
    def test_create_rmsd_analyzer_custom_selection(self):
        """Test creating analyzer with custom atom selection."""
        analyzer = create_rmsd_analyzer(atom_selection='heavy')
        
        self.assertIsInstance(analyzer, RMSDAnalyzer)
        self.assertEqual(analyzer.atom_selection, 'heavy')


class TestRMSDIntegration(unittest.TestCase):
    """Integration tests for RMSD analysis."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        np.random.seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after integration tests."""
        plt.close('all')
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_rmsd_analysis_workflow(self):
        """Test complete RMSD analysis workflow."""
        # Create synthetic protein trajectory
        n_atoms = 50
        n_frames = 30
        
        # Reference structure (folded state)
        reference = np.random.random((n_atoms, 3)) * 8.0
        
        # Create trajectory with unfolding simulation
        trajectory = []
        for i in range(n_frames):
            # Gradual unfolding: increase disorder over time
            disorder_factor = 1.0 + 0.1 * i
            noise_level = 0.05 + 0.02 * i
            
            noise = np.random.normal(0, noise_level, (n_atoms, 3))
            expansion = reference * disorder_factor + noise
            trajectory.append(expansion)
        
        times = np.arange(n_frames) * 0.5  # 0.5 ps timesteps
        
        # Initialize analyzer
        analyzer = create_rmsd_analyzer(reference_structure=reference)
        
        # Calculate trajectory RMSD
        calc_times, rmsd_values = analyzer.calculate_trajectory_rmsd(
            trajectory, times, "unfolding_simulation"
        )
        
        # Verify increasing RMSD trend (unfolding)
        self.assertLess(rmsd_values[0], rmsd_values[-1])
        self.assertGreater(np.corrcoef(times, rmsd_values)[0, 1], 0.8)  # Strong correlation
        
        # Create plots
        fig1 = analyzer.plot_rmsd_time_series(
            save_path=os.path.join(self.temp_dir, "unfolding_rmsd.png")
        )
        
        # Test pairwise comparison
        key_frames = [trajectory[0], trajectory[10], trajectory[20], trajectory[-1]]
        frame_names = ["Native", "Early", "Middle", "Unfolded"]
        
        rmsd_matrix = analyzer.calculate_pairwise_rmsd(key_frames, frame_names)
        fig2 = analyzer.plot_pairwise_rmsd_matrix(
            save_path=os.path.join(self.temp_dir, "pairwise_rmsd.png")
        )
        
        # Export data
        analyzer.export_data(os.path.join(self.temp_dir, "rmsd_data.csv"))
        
        # Get statistics
        stats = analyzer.get_statistics("unfolding_simulation")
        
        # Verify results
        self.assertGreater(stats['mean'], 0)
        self.assertGreater(stats['max'], stats['min'])
        self.assertEqual(stats['n_frames'], n_frames)
        
        # Check files were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "unfolding_rmsd.png")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "pairwise_rmsd.png")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "rmsd_data.csv")))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
