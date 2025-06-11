"""
Radius of Gyration Analysis Demo

This demo shows how to use the radius of gyration analysis module for:
- Single structure analysis
- Trajectory analysis with time evolution
- Segmental analysis for different protein regions
- Statistical analysis and visualization
- Data export capabilities

Author: ProteinMD Development Team
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.radius_of_gyration import create_rg_analyzer


def create_demo_protein_trajectory():
    """Create a demo protein trajectory showing folding/unfolding."""
    print("Creating demo protein trajectory with folding dynamics...")
    
    np.random.seed(42)
    n_atoms = 150  # Medium-sized protein
    n_frames = 100
    
    # Create initial extended structure
    initial_positions = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        # Create an extended chain
        initial_positions[i] = [i * 0.4, 0, 0]  # 0.4 nm spacing
        # Add some random displacement
        initial_positions[i] += np.random.normal(0, 0.1, 3)
    
    # Create masses (roughly carbon/nitrogen/oxygen weights)
    masses = np.random.choice([12.0, 14.0, 16.0], size=n_atoms, p=[0.5, 0.3, 0.2])
    
    # Simulate folding process: transition from extended to compact
    trajectory_positions = []
    time_points = np.linspace(0, 10.0, n_frames)  # 10 ns simulation
    
    for i, t in enumerate(time_points):
        # Folding parameter: 0 (extended) to 1 (compact)
        folding_factor = 1.0 / (1.0 + np.exp(-0.5 * (t - 5.0)))  # Sigmoid folding
        
        # Target compact structure (sphere)
        compact_positions = np.random.normal(0, 1.5, (n_atoms, 3))
        
        # Interpolate between extended and compact
        current_positions = (1 - folding_factor) * initial_positions + folding_factor * compact_positions
        
        # Add thermal motion
        thermal_noise = np.random.normal(0, 0.05, (n_atoms, 3))
        current_positions += thermal_noise
        
        trajectory_positions.append(current_positions)
    
    trajectory_positions = np.array(trajectory_positions)
    
    print(f"Created trajectory with {n_frames} frames and {n_atoms} atoms")
    print(f"Simulation time: {time_points[0]:.1f} - {time_points[-1]:.1f} ns")
    
    return trajectory_positions, masses, time_points


def define_protein_segments(n_atoms):
    """Define realistic protein segments."""
    # Define segments based on typical protein structure
    segments = {
        'N_terminal': np.arange(0, n_atoms//6),
        'Domain1': np.arange(n_atoms//6, n_atoms//3),
        'Linker': np.arange(n_atoms//3, n_atoms//2),
        'Domain2': np.arange(n_atoms//2, 5*n_atoms//6),
        'C_terminal': np.arange(5*n_atoms//6, n_atoms)
    }
    
    print(f"Defined protein segments:")
    for name, indices in segments.items():
        print(f"  - {name}: residues {indices[0]+1}-{indices[-1]+1} ({len(indices)} atoms)")
    
    return segments


def demo_single_structure_analysis():
    """Demonstrate single structure analysis."""
    print("\n" + "="*60)
    print("SINGLE STRUCTURE ANALYSIS DEMO")
    print("="*60)
    
    # Create a simple protein-like structure
    np.random.seed(42)
    n_atoms = 100
    positions = np.random.normal(0, 1.2, (n_atoms, 3))  # Roughly spherical
    masses = np.random.uniform(12.0, 16.0, n_atoms)
    
    # Define segments
    segments = define_protein_segments(n_atoms)
    
    # Create analyzer
    analyzer = create_rg_analyzer(segments)
    
    # Analyze structure
    result = analyzer.analyze_structure(positions, masses)
    
    print(f"\nSingle Structure Analysis Results:")
    print(f"- Total atoms: {result['n_atoms']}")
    print(f"- Total mass: {result['total_mass']:.1f} amu")
    print(f"- Center of mass: ({result['center_of_mass'][0]:.3f}, {result['center_of_mass'][1]:.3f}, {result['center_of_mass'][2]:.3f}) nm")
    print(f"- Overall radius of gyration: {result['overall_rg']:.3f} nm")
    
    print(f"\nSegmental Analysis:")
    for segment_name, rg_value in result['segmental_rg'].items():
        print(f"  - {segment_name}: {rg_value:.3f} nm")
    
    return analyzer


def demo_trajectory_analysis():
    """Demonstrate trajectory analysis with folding dynamics."""
    print("\n" + "="*60)
    print("TRAJECTORY ANALYSIS DEMO")
    print("="*60)
    
    # Create demo trajectory
    trajectory_positions, masses, time_points = create_demo_protein_trajectory()
    n_atoms = len(masses)
    
    # Define segments
    segments = define_protein_segments(n_atoms)
    
    # Create analyzer
    analyzer = create_rg_analyzer(segments)
    
    # Analyze trajectory
    print("\nAnalyzing trajectory...")
    results = analyzer.analyze_trajectory(trajectory_positions, masses, time_points)
    
    print(f"\nTrajectory Analysis Results:")
    print(f"- Frames analyzed: {results['n_frames']}")
    print(f"- Time range: {results['time_range'][0]:.1f} - {results['time_range'][1]:.1f} ns")
    
    # Overall statistics
    overall_stats = results['overall_statistics']
    print(f"\nOverall Radius of Gyration Statistics:")
    print(f"  - Mean: {overall_stats['mean']:.3f} ± {overall_stats['std']:.3f} nm")
    print(f"  - Range: [{overall_stats['min']:.3f}, {overall_stats['max']:.3f}] nm")
    print(f"  - Median: {overall_stats['median']:.3f} nm")
    print(f"  - Quartiles: Q25={overall_stats['q25']:.3f}, Q75={overall_stats['q75']:.3f} nm")
    
    # Segmental statistics
    print(f"\nSegmental Analysis:")
    segmental_stats = results['segmental_statistics']
    for segment_name, stats in segmental_stats.items():
        print(f"  - {segment_name}: {stats['mean']:.3f} ± {stats['std']:.3f} nm "
              f"[{stats['min']:.3f}, {stats['max']:.3f}]")
    
    return analyzer


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n" + "="*60)
    print("VISUALIZATION DEMO")
    print("="*60)
    
    # Create demo trajectory
    trajectory_positions, masses, time_points = create_demo_protein_trajectory()
    n_atoms = len(masses)
    
    # Define segments
    segments = define_protein_segments(n_atoms)
    
    # Create analyzer and analyze
    analyzer = create_rg_analyzer(segments)
    analyzer.analyze_trajectory(trajectory_positions, masses, time_points)
    
    print("Generating visualizations...")
    
    # Time series plot
    try:
        fig1 = analyzer.plot_rg_time_series(figsize=(14, 10), include_segments=True,
                                           title="Protein Folding: Radius of Gyration vs Time")
        fig1.savefig('rg_time_series_demo.png', dpi=300, bbox_inches='tight')
        print("✓ Time series plot saved as 'rg_time_series_demo.png'")
    except Exception as e:
        print(f"✗ Time series plot failed: {e}")
    
    # Distribution plot
    try:
        fig2 = analyzer.plot_rg_distribution(figsize=(10, 6), bins=25)
        fig2.savefig('rg_distribution_demo.png', dpi=300, bbox_inches='tight')
        print("✓ Distribution plot saved as 'rg_distribution_demo.png'")
    except Exception as e:
        print(f"✗ Distribution plot failed: {e}")
    
    plt.close('all')  # Clean up plots
    
    return analyzer


def demo_data_export():
    """Demonstrate data export capabilities."""
    print("\n" + "="*60)
    print("DATA EXPORT DEMO")
    print("="*60)
    
    # Create simple trajectory for export
    np.random.seed(42)
    n_atoms = 50
    n_frames = 10
    
    positions = np.random.normal(0, 1, (n_atoms, 3))
    masses = np.ones(n_atoms)
    
    trajectory_positions = np.array([positions + np.random.normal(0, 0.1, (n_atoms, 3)) 
                                   for _ in range(n_frames)])
    time_points = np.arange(n_frames) * 0.5
    
    # Define segments
    segments = {
        'N_term': np.arange(0, 15),
        'Core': np.arange(15, 35),
        'C_term': np.arange(35, 50)
    }
    
    # Analyze
    analyzer = create_rg_analyzer(segments)
    analyzer.analyze_trajectory(trajectory_positions, masses, time_points)
    
    # Export data
    print("Exporting data...")
    
    try:
        analyzer.export_data('rg_data_demo.csv', format='csv')
        print("✓ CSV data exported to 'rg_data_demo.csv'")
        
        # Show first few lines
        with open('rg_data_demo.csv', 'r') as f:
            lines = f.readlines()[:6]  # Header + 5 data lines
        
        print("First few lines of CSV:")
        for line in lines:
            print(f"  {line.strip()}")
            
    except Exception as e:
        print(f"✗ CSV export failed: {e}")
    
    try:
        analyzer.export_data('rg_data_demo.json', format='json')
        print("✓ JSON data exported to 'rg_data_demo.json'")
    except Exception as e:
        print(f"✗ JSON export failed: {e}")


def demo_physical_interpretation():
    """Demonstrate physical interpretation of Rg values."""
    print("\n" + "="*60)
    print("PHYSICAL INTERPRETATION DEMO")
    print("="*60)
    
    print("Comparing different protein conformations...")
    
    np.random.seed(42)
    n_atoms = 100
    masses = np.ones(n_atoms)
    
    # Extended structure (denatured protein)
    extended_positions = np.array([[i * 0.4, 0, 0] for i in range(n_atoms)], dtype=float)
    extended_positions += np.random.normal(0, 0.1, (n_atoms, 3))
    
    # Compact structure (folded protein)
    compact_positions = np.random.normal(0, 1.0, (n_atoms, 3))
    
    # Intermediate structure
    intermediate_positions = 0.5 * extended_positions + 0.5 * compact_positions
    intermediate_positions += np.random.normal(0, 0.2, (n_atoms, 3))
    
    # Calculate Rg for each
    rg_extended = calculate_radius_of_gyration(extended_positions, masses)
    rg_compact = calculate_radius_of_gyration(compact_positions, masses)
    rg_intermediate = calculate_radius_of_gyration(intermediate_positions, masses)
    
    print(f"\nRadius of Gyration Comparison:")
    print(f"  - Extended (denatured): {rg_extended:.3f} nm")
    print(f"  - Intermediate:         {rg_intermediate:.3f} nm")
    print(f"  - Compact (folded):     {rg_compact:.3f} nm")
    
    print(f"\nPhysical Interpretation:")
    print(f"  - Compactness ratio (folded/denatured): {rg_compact/rg_extended:.2f}")
    
    if rg_compact < 2.0:
        print(f"  - The folded structure is very compact (typical for small proteins)")
    elif rg_compact < 3.0:
        print(f"  - The folded structure has moderate compactness (typical for medium proteins)")
    else:
        print(f"  - The folded structure is relatively extended (typical for large proteins)")
    
    # Estimate protein size from Rg
    # For globular proteins: Rg ≈ 0.2 * N^0.6 nm (N = number of residues)
    estimated_residues = (rg_compact / 0.2) ** (1/0.6)
    print(f"  - Estimated protein size: ~{estimated_residues:.0f} residues")


def benchmark_performance():
    """Benchmark the performance of radius of gyration analysis."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    import time
    
    sizes = [100, 500, 1000, 2000]
    frame_counts = [10, 50, 100]
    
    print(f"{'System Size':<12} {'Frames':<8} {'Analysis Time':<15} {'Per Frame':<12}")
    print("-" * 55)
    
    for n_atoms in sizes:
        for n_frames in frame_counts:
            # Create test data
            np.random.seed(42)
            positions = np.random.normal(0, 1, (n_atoms, 3))
            masses = np.ones(n_atoms)
            
            trajectory_positions = np.array([positions + np.random.normal(0, 0.1, (n_atoms, 3)) 
                                           for _ in range(n_frames)])
            time_points = np.arange(n_frames) * 0.1
            
            # Benchmark analysis
            analyzer = create_rg_analyzer()
            
            start_time = time.time()
            analyzer.analyze_trajectory(trajectory_positions, masses, time_points)
            analysis_time = time.time() - start_time
            
            per_frame_time = analysis_time / n_frames * 1000  # ms per frame
            
            print(f"{n_atoms:<12} {n_frames:<8} {analysis_time*1000:10.2f} ms    {per_frame_time:8.2f} ms")


def main():
    """Run all demos."""
    print("="*60)
    print("RADIUS OF GYRATION ANALYSIS - COMPREHENSIVE DEMO")
    print("="*60)
    print("This demo showcases the capabilities of the radius of gyration analysis module.")
    
    try:
        # Run all demo functions
        demo_single_structure_analysis()
        demo_trajectory_analysis()
        demo_visualization()
        demo_data_export()
        demo_physical_interpretation()
        benchmark_performance()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Check the generated files:")
        print("  - rg_time_series_demo.png")
        print("  - rg_distribution_demo.png")
        print("  - rg_data_demo.csv")
        print("  - rg_data_demo.json")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
