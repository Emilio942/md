#!/usr/bin/env python3
"""
Task 3.3 Radius of Gyration - Quick Verification Test
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from proteinMD.analysis.radius_of_gyration import (
    RadiusOfGyrationAnalyzer, 
    create_rg_analyzer,
    calculate_radius_of_gyration,
    calculate_center_of_mass
)

def test_task_3_3_requirements():
    """Test all Task 3.3 requirements."""
    print("="*60)
    print("TASK 3.3 RADIUS OF GYRATION - REQUIREMENT VERIFICATION")
    print("="*60)
    
    # Create test protein structure (100 atoms)
    np.random.seed(42)
    n_atoms = 100
    n_frames = 50
    
    # Create a realistic protein structure
    base_positions = np.random.normal(0, 2.0, (n_atoms, 3))
    masses = np.random.uniform(12.0, 16.0, n_atoms)
    
    # Create trajectory with dynamics
    trajectory = []
    for i in range(n_frames):
        # Add breathing motion
        factor = 1.0 + 0.1 * np.sin(2 * np.pi * i / 20)
        positions = base_positions * factor
        # Add noise
        positions += np.random.normal(0, 0.1, (n_atoms, 3))
        trajectory.append(positions)
    
    trajectory = np.array(trajectory)
    time_points = np.arange(n_frames) * 0.1  # 0.1 ns timesteps
    
    # Define protein segments for testing
    segments = {
        'N_terminal': np.arange(0, 25),
        'Core': np.arange(25, 75),
        'C_terminal': np.arange(75, 100)
    }
    
    print("\n1️⃣ Testing: Gyrationsradius wird für jeden Zeitschritt berechnet")
    print("-" * 55)
    
    # Create analyzer with segments
    analyzer = create_rg_analyzer(segments)
    
    # Analyze trajectory
    results = analyzer.analyze_trajectory(trajectory, masses, time_points)
    
    # Verify we have Rg for each timestep
    assert results['n_frames'] == n_frames, "Should have Rg for each frame"
    assert len(analyzer.trajectory_data) == n_frames, "Trajectory data incomplete"
    
    # Verify each frame has Rg calculated
    for i, frame_data in enumerate(analyzer.trajectory_data):
        assert 'overall_rg' in frame_data, f"Frame {i} missing overall_rg"
        assert frame_data['overall_rg'] > 0, f"Frame {i} has invalid Rg"
        assert 'time' in frame_data, f"Frame {i} missing time"
        assert frame_data['time'] == time_points[i], f"Frame {i} time mismatch"
    
    print(f"✅ Calculated Rg for all {n_frames} timesteps")
    print(f"   Time range: {results['time_range'][0]:.2f} - {results['time_range'][1]:.2f} ns")
    
    print("\n2️⃣ Testing: Zeitverlauf wird als Graph dargestellt")
    print("-" * 48)
    
    # Create time series plot
    fig = analyzer.plot_rg_time_series(
        figsize=(12, 8),
        include_segments=True,
        title="Task 3.3 Test: Rg Time Evolution"
    )
    
    # Save plot to temporary directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "test_task_3_3_timeseries.png")
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        assert os.path.exists(output_file), "Time series plot not created"
        print(f"✅ Generated time series plot: {output_file}")
    
    # Also create distribution plot
    fig2 = analyzer.plot_rg_distribution(figsize=(10, 6), bins=20)
    with tempfile.TemporaryDirectory() as tmpdir:
        dist_file = os.path.join(tmpdir, "test_task_3_3_distribution.png")
        fig2.savefig(dist_file, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        assert os.path.exists(dist_file), "Distribution plot not created"
        print(f"✅ Generated distribution plot: {dist_file}")
    
    print("\n3️⃣ Testing: Getrennte Analyse für verschiedene Proteinsegmente möglich")
    print("-" * 68)
    
    # Verify segmental analysis was performed
    assert 'segmental_statistics' in results, "Segmental statistics missing"
    seg_stats = results['segmental_statistics']
    
    # Verify all segments were analyzed
    for segment_name in segments.keys():
        assert segment_name in seg_stats, f"Segment {segment_name} not analyzed"
        assert 'mean' in seg_stats[segment_name], f"Segment {segment_name} missing mean"
        assert seg_stats[segment_name]['mean'] > 0, f"Segment {segment_name} invalid mean"
    
    print(f"✅ Analyzed {len(segments)} protein segments:")
    for segment_name, stats in seg_stats.items():
        print(f"   {segment_name}: {stats['mean']:.3f} ± {stats['std']:.3f} nm")
    
    # Verify segments have reasonable Rg values
    overall_mean = results['overall_statistics']['mean']
    for segment_name, stats in seg_stats.items():
        # Segments should have positive, finite Rg values
        assert 0 < stats['mean'] < 50, f"Segment {segment_name} has unrealistic Rg: {stats['mean']}"
    
    print("✅ Segmental analysis shows realistic Rg values for all segments")
    
    print("\n4️⃣ Testing: Statistische Auswertung verfügbar")
    print("-" * 45)
    
    # Verify comprehensive statistics
    overall_stats = results['overall_statistics']
    required_stats = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
    
    for stat in required_stats:
        assert stat in overall_stats, f"Missing {stat} in statistics"
        assert isinstance(overall_stats[stat], (float, int)), f"{stat} not numeric"
    
    print("✅ Overall statistics available:")
    print(f"   Mean: {overall_stats['mean']:.4f} nm")
    print(f"   Std:  {overall_stats['std']:.4f} nm") 
    print(f"   Range: [{overall_stats['min']:.4f}, {overall_stats['max']:.4f}] nm")
    print(f"   Quartiles: Q25={overall_stats['q25']:.4f}, Q75={overall_stats['q75']:.4f}")
    
    # Verify statistical consistency
    assert overall_stats['min'] <= overall_stats['q25'], "Min > Q25"
    assert overall_stats['q25'] <= overall_stats['median'], "Q25 > median"
    assert overall_stats['median'] <= overall_stats['q75'], "Median > Q75"
    assert overall_stats['q75'] <= overall_stats['max'], "Q75 > max"
    assert overall_stats['std'] >= 0, "Negative standard deviation"
    
    print("✅ Statistical consistency verified")
    
    # Test data export capabilities
    print("\n📊 Testing Data Export...")
    print("-" * 25)
    
    import tempfile
    # CSV export
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        csv_file = tmp.name
    analyzer.export_data(csv_file, format='csv')
    assert os.path.exists(csv_file), "CSV export failed"
    print(f"✅ CSV export successful: {csv_file}")
    
    # JSON export  
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        json_file = tmp.name
    analyzer.export_data(json_file, format='json')
    assert os.path.exists(json_file), "JSON export failed"
    print(f"✅ JSON export successful: {json_file}")
    
    # Verify CSV content
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 1, "CSV should have header and data"
        header = lines[0].strip().split(',')
        assert 'time' in header, "CSV missing time column"
        assert 'overall_rg' in header, "CSV missing overall_rg column"
    
    # Clean up temp files
    os.unlink(csv_file)
    os.unlink(json_file)
    
    print("\n" + "="*60)
    print("🎉 TASK 3.3 VERIFICATION COMPLETE 🎉")
    print("="*60)
    
    print("\n✅ ALL REQUIREMENTS VERIFIED:")
    print("✅ Gyrationsradius wird für jeden Zeitschritt berechnet")
    print("✅ Zeitverlauf wird als Graph dargestellt")
    print("✅ Getrennte Analyse für verschiedene Proteinsegmente möglich")
    print("✅ Statistische Auswertung (Mittelwert, Standardabweichung) verfügbar")
    
    print("\n📁 Generated Files:")
    print(f"📊 {output_file} - Time evolution with segmental analysis")
    print(f"📊 {dist_file} - Rg distribution histogram")
    print(f"📄 {csv_file} - Exported trajectory data")
    print(f"📄 {json_file} - Complete analysis results")
    
    print(f"\n📈 Analysis Summary:")
    print(f"Frames analyzed: {results['n_frames']}")
    print(f"Time coverage: {results['time_range'][1] - results['time_range'][0]:.1f} ns")
    print(f"Mean Rg: {overall_stats['mean']:.3f} ± {overall_stats['std']:.3f} nm")
    print(f"Segments analyzed: {len(segments)}")
    
    assert True  # Test completed successfully

if __name__ == "__main__":
    try:
        success = test_task_3_3_requirements()
        if success:
            print("\n🟢 TASK 3.3 IS COMPLETE AND FUNCTIONAL! 🟢")
        else:
            print("\n🔴 TASK 3.3 VERIFICATION FAILED! 🔴")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
