#!/usr/bin/env python3
"""
Comprehensive Test Suite for Task 3.3: Radius of Gyration Analysis

This test script verifies all requirements for Task 3.3:
- Gyrationsradius wird für jeden Zeitschritt berechnet
- Zeitverlauf wird als Graph dargestellt
- Getrennte Analyse für verschiedene Proteinsegmente möglich
- Statistische Auswertung (Mittelwert, Standardabweichung) verfügbar

Author: ProteinMD Test Suite
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

# Add the project directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
proteinmd_path = os.path.join(project_root, 'proteinMD')
if proteinmd_path not in sys.path:
    sys.path.insert(0, proteinmd_path)

try:
    from analysis.radius_of_gyration import (
        RadiusOfGyrationAnalyzer, 
        create_rg_analyzer,
        calculate_radius_of_gyration,
        calculate_center_of_mass,
        calculate_segmental_rg
    )
    print("✓ Successfully imported radius of gyration analysis modules")
except ImportError as e:
    print(f"✗ Failed to import radius of gyration modules: {e}")
    sys.exit(1)


class MockProtein:
    """Mock protein structure for testing."""
    
    def __init__(self, n_atoms=120):
        self.n_atoms = n_atoms
        np.random.seed(42)
        
        # Create realistic protein structure (compact globular protein)
        # Use a combination of secondary structure motifs
        self.base_positions = self._create_protein_structure()
        self.masses = self._create_atomic_masses()
        
    def _create_protein_structure(self):
        """Create a realistic protein structure with secondary structure elements."""
        positions = []
        
        # N-terminal loop (residues 1-20)
        for i in range(20):
            # Extended conformation
            x = i * 0.38  # Ca-Ca distance ~3.8 Å
            y = np.sin(i * 0.5) * 2.0  # Some flexibility
            z = np.cos(i * 0.5) * 1.0
            positions.append([x, y, z])
        
        # Alpha helix (residues 21-50)
        for i in range(30):
            # Helical parameters: rise per residue = 1.5 Å, radius = 2.3 Å
            angle = i * 100 * np.pi / 180  # 100° per residue
            x = 20 * 0.38 + i * 0.15  # Continue from previous
            y = 2.3 * np.cos(angle)
            z = 2.3 * np.sin(angle) + i * 0.15
            positions.append([x, y, z])
        
        # Beta sheet (residues 51-80)
        for i in range(30):
            # Extended beta strand
            x = 50 * 0.38 + (i % 10) * 0.35  # 3.5 Å between residues
            y = 4.0 if i < 15 else -4.0  # Two strands
            z = 50 * 0.15 + (i // 10) * 0.48  # 4.8 Å between strands
            positions.append([x, y, z])
        
        # C-terminal loop (residues 81-120)
        for i in range(40):
            # Random coil returning to compact state
            base_x = 50 * 0.38 + 10 * 0.35
            x = base_x + np.random.normal(0, 3.0)
            y = np.random.normal(0, 4.0)
            z = 50 * 0.15 + np.random.normal(0, 3.0)
            positions.append([x, y, z])
        
        return np.array(positions)
    
    def _create_atomic_masses(self):
        """Create realistic atomic masses (approximating Ca atoms)."""
        # Most atoms are carbon (12.01), some nitrogen (14.01), oxygen (16.00)
        masses = np.random.choice([12.01, 14.01, 16.00], 
                                size=self.n_atoms, 
                                p=[0.6, 0.2, 0.2])
        return masses
    
    def create_trajectory(self, n_frames=100, dynamics_type='breathing'):
        """Create a molecular dynamics trajectory."""
        trajectory = []
        
        for frame in range(n_frames):
            if dynamics_type == 'breathing':
                # Breathing motion - periodic expansion/contraction
                factor = 1.0 + 0.1 * np.sin(2 * np.pi * frame / 20)
                positions = self.base_positions * factor
                
            elif dynamics_type == 'unfolding':
                # Gradual unfolding
                factor = 1.0 + 0.005 * frame
                positions = self.base_positions * factor
                
            elif dynamics_type == 'random_walk':
                # Random fluctuations
                factor = 1.0 + np.random.normal(0, 0.05)
                positions = self.base_positions * factor
                
            else:  # stable
                factor = 1.0
                positions = self.base_positions.copy()
            
            # Add thermal noise
            noise = np.random.normal(0, 0.1, positions.shape)
            positions += noise
            
            trajectory.append(positions)
        
        return np.array(trajectory)


def test_basic_rg_calculation():
    """Test basic radius of gyration calculation."""
    print("\n=== Testing Basic Rg Calculation ===")
    
    # Simple test case
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    masses = np.array([1.0, 1.0, 1.0, 1.0])
    
    # Calculate center of mass
    com = calculate_center_of_mass(positions, masses)
    print(f"Center of mass: {com}")
    
    # Calculate Rg
    rg = calculate_radius_of_gyration(positions, masses, com)
    print(f"Radius of gyration: {rg:.4f}")
    
    # Expected Rg for this symmetric configuration
    expected_rg = np.sqrt(3/4)  # ~0.866
    assert abs(rg - expected_rg) < 1e-3, f"Expected Rg ~{expected_rg:.3f}, got {rg:.3f}"
    
    print("✓ Basic Rg calculation test passed")
    return True


def test_segmental_analysis():
    """Test segmental radius of gyration analysis."""
    print("\n=== Testing Segmental Analysis ===")
    
    protein = MockProtein(n_atoms=120)
    
    # Define protein segments
    segments = {
        'N_terminal': np.arange(0, 20),      # Residues 1-20
        'Helix': np.arange(20, 50),          # Residues 21-50  
        'Beta_sheet': np.arange(50, 80),     # Residues 51-80
        'C_terminal': np.arange(80, 120),    # Residues 81-120
        'Core': np.arange(20, 80),           # Helix + Beta sheet
        'Termini': np.concatenate([np.arange(0, 20), np.arange(80, 120)])  # Both termini
    }
    
    # Calculate segmental Rg
    segmental_rg = calculate_segmental_rg(protein.base_positions, protein.masses, segments)
    
    print("Segmental Rg values:")
    for segment_name, rg_value in segmental_rg.items():
        print(f"  {segment_name}: {rg_value:.3f} nm")
    
    # Verify all segments were calculated
    assert len(segmental_rg) == len(segments), "Not all segments were calculated"
    
    # Verify all values are positive
    for segment_name, rg_value in segmental_rg.items():
        assert rg_value > 0, f"Segment {segment_name} has non-positive Rg: {rg_value}"
    
    # Core should be more compact than full protein
    overall_rg = calculate_radius_of_gyration(protein.base_positions, protein.masses)
    assert segmental_rg['Core'] < overall_rg, "Core should be more compact than full protein"
    
    print("✓ Segmental analysis test passed")
    return True


def test_trajectory_analysis():
    """Test trajectory analysis with time evolution."""
    print("\n=== Testing Trajectory Analysis ===")
    
    protein = MockProtein(n_atoms=100)
    
    # Create different types of trajectories
    test_cases = [
        ('breathing', 'Breathing Motion'),
        ('unfolding', 'Unfolding Process'),
        ('stable', 'Stable Structure')
    ]
    
    all_results = {}
    
    for dynamics_type, description in test_cases:
        print(f"\nTesting {description}...")
        
        # Create trajectory
        n_frames = 50
        trajectory = protein.create_trajectory(n_frames, dynamics_type)
        time_points = np.arange(n_frames) * 0.1  # 0.1 ns timesteps
        
        # Define segments
        segments = {
            'N_half': np.arange(0, 50),
            'C_half': np.arange(50, 100)
        }
        
        # Create analyzer
        analyzer = create_rg_analyzer(segments)
        
        # Analyze trajectory
        results = analyzer.analyze_trajectory(trajectory, protein.masses, time_points)
        all_results[dynamics_type] = results
        
        # Verify results structure
        assert 'n_frames' in results, "Missing n_frames in results"
        assert 'time_range' in results, "Missing time_range in results"
        assert 'overall_statistics' in results, "Missing overall_statistics in results"
        assert 'segmental_statistics' in results, "Missing segmental_statistics in results"
        
        # Verify frame count
        assert results['n_frames'] == n_frames, f"Expected {n_frames} frames, got {results['n_frames']}"
        
        # Verify time range
        expected_time_range = [0.0, (n_frames-1) * 0.1]
        assert abs(results['time_range'][0] - expected_time_range[0]) < 1e-6, "Incorrect start time"
        assert abs(results['time_range'][1] - expected_time_range[1]) < 1e-6, "Incorrect end time"
        
        # Verify statistical measures exist
        stats = results['overall_statistics']
        required_stats = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
        for stat in required_stats:
            assert stat in stats, f"Missing {stat} in overall statistics"
            assert isinstance(stats[stat], (float, int)), f"{stat} should be numeric"
        
        # Verify segmental statistics
        seg_stats = results['segmental_statistics']
        assert len(seg_stats) == len(segments), "Segmental statistics missing"
        
        print(f"  ✓ {description}: Mean Rg = {stats['mean']:.3f} ± {stats['std']:.3f} nm")
        print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}] nm")
    
    # Compare dynamics types
    breathing_std = all_results['breathing']['overall_statistics']['std']
    stable_std = all_results['stable']['overall_statistics']['std']
    
    # Breathing should show more variation than stable
    assert breathing_std > stable_std, "Breathing motion should show more Rg variation than stable"
    
    print("✓ Trajectory analysis test passed")
    return True


def test_visualization():
    """Test visualization capabilities."""
    print("\n=== Testing Visualization ===")
    
    protein = MockProtein(n_atoms=80)
    
    # Create trajectory with interesting dynamics
    trajectory = protein.create_trajectory(n_frames=60, dynamics_type='breathing')
    time_points = np.arange(60) * 0.05  # 0.05 ns timesteps
    
    # Define segments
    segments = {
        'Domain_A': np.arange(0, 40),
        'Domain_B': np.arange(40, 80)
    }
    
    # Create and run analysis
    analyzer = create_rg_analyzer(segments)
    analyzer.analyze_trajectory(trajectory, protein.masses, time_points)
    
    # Test time series plot
    print("Creating time series plot...")
    fig1 = analyzer.plot_rg_time_series(
        figsize=(12, 8),
        include_segments=True,
        title="Radius of Gyration Evolution - Test Case"
    )
    
    # Save plot
    output_file1 = "test_rg_timeseries.png"
    fig1.savefig(output_file1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    assert os.path.exists(output_file1), f"Time series plot not saved: {output_file1}"
    print(f"  ✓ Time series plot saved: {output_file1}")
    
    # Test distribution plot
    print("Creating distribution plot...")
    fig2 = analyzer.plot_rg_distribution(
        figsize=(10, 6),
        bins=25
    )
    
    # Save plot
    output_file2 = "test_rg_distribution.png"
    fig2.savefig(output_file2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    assert os.path.exists(output_file2), f"Distribution plot not saved: {output_file2}"
    print(f"  ✓ Distribution plot saved: {output_file2}")
    
    # Test plot without segments
    print("Creating plot without segmental analysis...")
    analyzer_simple = RadiusOfGyrationAnalyzer()
    analyzer_simple.analyze_trajectory(trajectory, protein.masses, time_points)
    
    fig3 = analyzer_simple.plot_rg_time_series(
        include_segments=False,
        title="Simple Rg Analysis"
    )
    
    output_file3 = "test_rg_simple.png"
    fig3.savefig(output_file3, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    assert os.path.exists(output_file3), f"Simple plot not saved: {output_file3}"
    print(f"  ✓ Simple plot saved: {output_file3}")
    
    print("✓ Visualization test passed")
    return True


def test_data_export():
    """Test data export functionality."""
    print("\n=== Testing Data Export ===")
    
    protein = MockProtein(n_atoms=60)
    
    # Create trajectory
    trajectory = protein.create_trajectory(n_frames=30, dynamics_type='unfolding')
    time_points = np.arange(30) * 0.1
    
    # Define segments
    segments = {
        'N_term': np.arange(0, 20),
        'C_term': np.arange(40, 60)
    }
    
    # Create and run analysis
    analyzer = create_rg_analyzer(segments)
    analyzer.analyze_trajectory(trajectory, protein.masses, time_points)
    
    # Test CSV export
    csv_file = "test_rg_data.csv"
    analyzer.export_data(csv_file, format='csv')
    
    assert os.path.exists(csv_file), f"CSV file not created: {csv_file}"
    
    # Verify CSV content
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 1, "CSV file should have header and data"
        header = lines[0].strip().split(',')
        assert 'time' in header, "CSV should contain time column"
        assert 'overall_rg' in header, "CSV should contain overall_rg column"
        assert any('rg_N_term' in col for col in header), "CSV should contain segmental data"
    
    print(f"  ✓ CSV export successful: {csv_file}")
    
    # Test JSON export
    json_file = "test_rg_data.json"
    analyzer.export_data(json_file, format='json')
    
    assert os.path.exists(json_file), f"JSON file not created: {json_file}"
    
    # Verify JSON content
    import json
    with open(json_file, 'r') as f:
        data = json.load(f)
        assert 'trajectory_data' in data, "JSON should contain trajectory_data"
        assert 'statistics' in data, "JSON should contain statistics"
        assert len(data['trajectory_data']) == 30, "JSON should contain all frames"
    
    print(f"  ✓ JSON export successful: {json_file}")
    
    print("✓ Data export test passed")
    return True


def test_statistical_analysis():
    """Test comprehensive statistical analysis."""
    print("\n=== Testing Statistical Analysis ===")
    
    protein = MockProtein(n_atoms=100)
    
    # Create trajectory with known properties
    n_frames = 100
    trajectory = protein.create_trajectory(n_frames, dynamics_type='breathing')
    time_points = np.arange(n_frames) * 0.02  # 0.02 ns timesteps
    
    # Define segments
    segments = {
        'Segment_1': np.arange(0, 25),
        'Segment_2': np.arange(25, 50),
        'Segment_3': np.arange(50, 75),
        'Segment_4': np.arange(75, 100)
    }
    
    # Run analysis
    analyzer = create_rg_analyzer(segments)
    results = analyzer.analyze_trajectory(trajectory, protein.masses, time_points)
    
    # Test overall statistics
    overall_stats = results['overall_statistics']
    
    # Verify statistical consistency
    assert overall_stats['min'] <= overall_stats['q25'], "Min should be <= Q25"
    assert overall_stats['q25'] <= overall_stats['median'], "Q25 should be <= median"
    assert overall_stats['median'] <= overall_stats['q75'], "Median should be <= Q75"
    assert overall_stats['q75'] <= overall_stats['max'], "Q75 should be <= max"
    
    # Standard deviation should be positive for breathing motion
    assert overall_stats['std'] > 0, "Standard deviation should be positive for dynamic system"
    
    # Mean should be between min and max
    assert overall_stats['min'] <= overall_stats['mean'] <= overall_stats['max'], "Mean should be within range"
    
    print(f"Overall Statistics:")
    print(f"  Mean: {overall_stats['mean']:.4f} nm")
    print(f"  Std:  {overall_stats['std']:.4f} nm")
    print(f"  Range: [{overall_stats['min']:.4f}, {overall_stats['max']:.4f}] nm")
    print(f"  Quartiles: Q25={overall_stats['q25']:.4f}, Q75={overall_stats['q75']:.4f}")
    
    # Test segmental statistics
    seg_stats = results['segmental_statistics']
    assert len(seg_stats) == len(segments), "Should have statistics for all segments"
    
    print(f"\nSegmental Statistics:")
    for segment_name, stats in seg_stats.items():
        print(f"  {segment_name}: {stats['mean']:.4f} ± {stats['std']:.4f} nm")
        
        # Verify statistical consistency for each segment
        assert stats['min'] <= stats['mean'] <= stats['max'], f"Mean out of range for {segment_name}"
        assert stats['std'] >= 0, f"Negative std for {segment_name}"
    
    # Verify segments are smaller than overall protein
    for segment_name, stats in seg_stats.items():
        assert stats['mean'] < overall_stats['mean'], f"Segment {segment_name} should be more compact than whole protein"
    
    print("✓ Statistical analysis test passed")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")
    
    # Test empty trajectory
    analyzer = RadiusOfGyrationAnalyzer()
    stats = analyzer.get_trajectory_statistics()
    assert stats == {}, "Empty trajectory should return empty statistics"
    print("  ✓ Empty trajectory handled correctly")
    
    # Test single atom
    single_position = np.array([[0, 0, 0]])
    single_mass = np.array([1.0])
    rg = calculate_radius_of_gyration(single_position, single_mass)
    assert rg == 0.0, "Single atom should have Rg = 0"
    print("  ✓ Single atom case handled correctly")
    
    # Test identical positions
    identical_positions = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    identical_masses = np.array([1.0, 1.0, 1.0])
    rg = calculate_radius_of_gyration(identical_positions, identical_masses)
    assert rg == 0.0, "Identical positions should have Rg = 0"
    print("  ✓ Identical positions handled correctly")
    
    # Test mismatched array lengths
    try:
        positions = np.array([[0, 0, 0], [1, 1, 1]])
        masses = np.array([1.0])  # Wrong length
        rg = calculate_radius_of_gyration(positions, masses)
        assert False, "Should have raised ValueError for mismatched arrays"
    except ValueError:
        print("  ✓ Mismatched array lengths properly detected")
    
    # Test zero masses
    zero_masses = np.array([0.0, 0.0, 0.0])
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    rg = calculate_radius_of_gyration(positions, zero_masses)
    # Should handle gracefully (returns 0 or uses geometric center)
    assert rg >= 0, "Zero masses should not cause negative Rg"
    print("  ✓ Zero masses handled correctly")
    
    print("✓ Edge cases test passed")
    return True


def run_comprehensive_test():
    """Run all tests for Task 3.3 requirements."""
    print("="*60)
    print("COMPREHENSIVE TEST SUITE FOR TASK 3.3: RADIUS OF GYRATION")
    print("="*60)
    
    print("\nTask 3.3 Requirements:")
    print("✓ Gyrationsradius wird für jeden Zeitschritt berechnet")
    print("✓ Zeitverlauf wird als Graph dargestellt")
    print("✓ Getrennte Analyse für verschiedene Proteinsegmente möglich")
    print("✓ Statistische Auswertung (Mittelwert, Standardabweichung) verfügbar")
    
    tests = [
        ("Basic Rg Calculation", test_basic_rg_calculation),
        ("Segmental Analysis", test_segmental_analysis),
        ("Trajectory Analysis", test_trajectory_analysis),
        ("Visualization", test_visualization),
        ("Data Export", test_data_export),
        ("Statistical Analysis", test_statistical_analysis),
        ("Edge Cases", test_edge_cases)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*50}")
            print(f"Running: {test_name}")
            print('='*50)
            
            success = test_func()
            if success:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Success rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Task 3.3 is COMPLETE! 🎉")
        print("\nTask 3.3 Requirements Verification:")
        print("✅ Gyrationsradius wird für jeden Zeitschritt berechnet - VERIFIED")
        print("✅ Zeitverlauf wird als Graph dargestellt - VERIFIED")
        print("✅ Getrennte Analyse für verschiedene Proteinsegmente möglich - VERIFIED")
        print("✅ Statistische Auswertung (Mittelwert, Standardabweichung) verfügbar - VERIFIED")
        
        print("\nGenerated Test Files:")
        print("📊 test_rg_timeseries.png - Time evolution plot with segmental analysis")
        print("📊 test_rg_distribution.png - Rg distribution histogram")
        print("📊 test_rg_simple.png - Simple time series plot")
        print("📄 test_rg_data.csv - Exported trajectory data")
        print("📄 test_rg_data.json - Complete analysis results")
        
        return True
    else:
        print(f"\n❌ {failed} tests failed. Task 3.3 needs additional work.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
