"""
Hydrogen Bond Analysis Demo

This script demonstrates the capabilities of the hydrogen bond analysis module
including detection, trajectory analysis, visualization, and data export.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from analysis.hydrogen_bonds import (
    HydrogenBondDetector, HydrogenBondAnalyzer,
    analyze_hydrogen_bonds, quick_hydrogen_bond_summary
)


class DemoAtom:
    """Simple atom class for demonstration."""
    
    def __init__(self, atom_id, atom_name, element, residue_id=0):
        self.atom_id = atom_id
        self.atom_name = atom_name
        self.element = element
        self.residue_id = residue_id


def create_demo_protein():
    """Create a simple protein structure for demonstration."""
    # Create a small peptide: N-CA-C-O-N-CA-C-O (2 residues)
    atoms = [
        # Residue 1
        DemoAtom(0, 'N', 'N', 0),
        DemoAtom(1, 'H', 'H', 0),
        DemoAtom(2, 'CA', 'C', 0),
        DemoAtom(3, 'C', 'C', 0),
        DemoAtom(4, 'O', 'O', 0),
        
        # Residue 2
        DemoAtom(5, 'N', 'N', 1),
        DemoAtom(6, 'H', 'H', 1),
        DemoAtom(7, 'CA', 'C', 1),
        DemoAtom(8, 'C', 'C', 1),
        DemoAtom(9, 'O', 'O', 1),
        
        # Residue 3
        DemoAtom(10, 'N', 'N', 2),
        DemoAtom(11, 'H', 'H', 2),
        DemoAtom(12, 'CA', 'C', 2),
        DemoAtom(13, 'C', 'C', 2),
        DemoAtom(14, 'O', 'O', 2),
        
        # Side chain atoms (simplified)
        DemoAtom(15, 'OH', 'O', 1),  # Serine hydroxyl
        DemoAtom(16, 'HO', 'H', 1),  # Hydroxyl hydrogen
        
        DemoAtom(17, 'ND', 'N', 2),  # Asparagine amide
        DemoAtom(18, 'HD1', 'H', 2), # Amide hydrogen 1
        DemoAtom(19, 'HD2', 'H', 2), # Amide hydrogen 2
    ]
    
    return atoms


def create_single_frame_positions():
    """Create positions for a single frame with hydrogen bonds."""
    # Idealized backbone geometry with hydrogen bonds
    positions = np.array([
        # Residue 1
        [0.0, 0.0, 0.0],    # N
        [0.8, 0.6, 0.0],    # H (bonded to N)
        [1.5, 0.0, 0.0],    # CA
        [2.5, 0.5, 0.0],    # C
        [3.5, 0.0, 0.0],    # O
        
        # Residue 2 
        [2.8, 1.5, 0.0],    # N
        [2.5, 2.3, 0.0],    # H (forms H-bond with O of res 1)
        [3.8, 2.0, 0.0],    # CA
        [4.8, 2.5, 0.0],    # C
        [5.8, 2.0, 0.0],    # O
        
        # Residue 3
        [5.1, 3.5, 0.0],    # N  
        [4.8, 4.3, 0.0],    # H (forms H-bond with O of res 2)
        [6.1, 4.0, 0.0],    # CA
        [7.1, 4.5, 0.0],    # C
        [8.1, 4.0, 0.0],    # O
        
        # Side chains
        [3.5, 2.8, 1.0],    # Serine OH
        [3.2, 3.6, 1.0],    # Serine HO
        
        [6.5, 4.8, 1.5],    # Asparagine ND
        [6.2, 5.6, 1.5],    # Asparagine HD1
        [7.3, 4.9, 1.8],    # Asparagine HD2
    ])
    
    return positions


def create_dynamic_trajectory(atoms, n_frames=50):
    """Create a trajectory showing hydrogen bond dynamics."""
    trajectory = []
    base_positions = create_single_frame_positions()
    
    for frame in range(n_frames):
        # Create dynamic motion
        t = frame / n_frames * 2 * np.pi
        
        # Base positions with small oscillations
        positions = base_positions.copy()
        
        # Add thermal motion (small random displacements)
        thermal_motion = np.random.normal(0, 0.1, positions.shape)
        positions += thermal_motion
        
        # Add systematic motion to break/form hydrogen bonds
        # Oscillate backbone to create H-bond dynamics
        oscillation = 0.3 * np.sin(t)
        
        # Move residue 2 to modulate H-bond with residue 1
        positions[5:10, 1] += oscillation  # Y-coordinate of residue 2
        
        # Move side chain to create intermittent H-bonds
        side_oscillation = 0.5 * np.sin(t * 1.5)
        positions[15:17, 0] += side_oscillation  # X-coordinate of serine
        
        trajectory.append(positions)
    
    return np.array(trajectory)


def demo_single_structure_analysis():
    """Demonstrate hydrogen bond detection in a single structure."""
    print("=" * 60)
    print("DEMO 1: Single Structure Hydrogen Bond Detection")
    print("=" * 60)
    
    atoms = create_demo_protein()
    positions = create_single_frame_positions()
    
    # Create detector
    detector = HydrogenBondDetector()
    
    # Detect hydrogen bonds
    bonds = detector.detect_hydrogen_bonds(atoms, positions)
    
    print(f"Found {len(bonds)} hydrogen bonds:")
    print()
    
    for i, bond in enumerate(bonds):
        donor_atom = atoms[bond.donor_atom_idx]
        acceptor_atom = atoms[bond.acceptor_atom_idx]
        
        print(f"H-bond {i+1}:")
        print(f"  Donor: {donor_atom.atom_name} (residue {donor_atom.residue_id})")
        print(f"  Acceptor: {acceptor_atom.atom_name} (residue {acceptor_atom.residue_id})")
        print(f"  Distance: {bond.distance:.2f} Å")
        print(f"  Angle: {bond.angle:.1f}°")
        print(f"  Type: {bond.bond_type}")
        print(f"  Strength: {bond.strength}")
        print()
    
    return bonds


def demo_trajectory_analysis():
    """Demonstrate hydrogen bond analysis over a trajectory."""
    print("=" * 60)
    print("DEMO 2: Trajectory Hydrogen Bond Analysis")
    print("=" * 60)
    
    atoms = create_demo_protein()
    trajectory = create_dynamic_trajectory(atoms, n_frames=30)
    
    print(f"Analyzing trajectory with {len(trajectory)} frames...")
    
    # Perform analysis
    analyzer = analyze_hydrogen_bonds(atoms, trajectory)
    
    # Get summary statistics
    stats = analyzer.get_summary_statistics()
    
    print("\nSummary Statistics:")
    print(f"  Mean H-bonds per frame: {stats['mean_bonds_per_frame']:.2f}")
    print(f"  Std H-bonds per frame: {stats['std_bonds_per_frame']:.2f}")
    print(f"  Max H-bonds in any frame: {stats['max_bonds_per_frame']}")
    print(f"  Min H-bonds in any frame: {stats['min_bonds_per_frame']}")
    
    # Bond type distribution
    print("\nBond Type Distribution:")
    for bond_type, type_stats in stats['bond_type_distribution'].items():
        print(f"  {bond_type}: {type_stats['count']} occurrences")
        print(f"    Mean distance: {type_stats['mean_distance']:.2f} Å")
    
    # Bond strength distribution
    print("\nBond Strength Distribution:")
    for strength, strength_stats in stats['bond_strength_distribution'].items():
        print(f"  {strength}: {strength_stats['count']} occurrences")
        print(f"    Mean distance: {strength_stats['mean_distance']:.2f} Å")
    
    # Lifetime statistics
    if 'lifetime_statistics' in stats:
        lifetime_stats = stats['lifetime_statistics']
        print("\nLifetime Statistics:")
        print(f"  Mean lifetime: {lifetime_stats['mean_lifetime']:.2f} frames")
        print(f"  Std lifetime: {lifetime_stats['std_lifetime']:.2f} frames")
        print(f"  Max lifetime: {lifetime_stats['max_lifetime']} frames")
        print(f"  Mean occupancy: {lifetime_stats['mean_occupancy']:.3f}")
    
    return analyzer


def demo_visualization(analyzer):
    """Demonstrate visualization capabilities."""
    print("=" * 60)
    print("DEMO 3: Hydrogen Bond Visualization")
    print("=" * 60)
    
    # Create plots
    print("Creating hydrogen bond evolution plot...")
    fig1 = analyzer.plot_bond_evolution()
    plt.show()
    
    print("Creating residue network plot...")
    fig2 = analyzer.plot_residue_network()
    plt.show()
    
    return fig1, fig2


def demo_data_export(analyzer):
    """Demonstrate data export capabilities."""
    print("=" * 60)
    print("DEMO 4: Data Export")
    print("=" * 60)
    
    # Create temporary files for export
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "hydrogen_bonds.csv")
        json_path = os.path.join(temp_dir, "lifetime_analysis.json")
        
        # Export data
        print("Exporting H-bond statistics to CSV...")
        analyzer.export_statistics_csv(csv_path)
        
        print("Exporting lifetime analysis to JSON...")
        analyzer.export_lifetime_analysis(json_path)
        
        # Show file sizes
        csv_size = os.path.getsize(csv_path)
        json_size = os.path.getsize(json_path)
        
        print(f"  CSV file size: {csv_size} bytes")
        print(f"  JSON file size: {json_size} bytes")
        
        # Show sample CSV content
        print("\nSample CSV content (first 5 lines):")
        with open(csv_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"  {line.strip()}")
        
        # Show sample JSON structure
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"\nJSON structure contains {len(data)} entries:")
        for key in list(data.keys())[:3]:
            print(f"  {key}")
        if len(data) > 3:
            print(f"  ... and {len(data) - 3} more")


def demo_custom_parameters():
    """Demonstrate analysis with custom parameters."""
    print("=" * 60)
    print("DEMO 5: Custom Detection Parameters")
    print("=" * 60)
    
    atoms = create_demo_protein()
    positions = create_single_frame_positions()
    
    # Standard parameters
    standard_detector = HydrogenBondDetector()
    standard_bonds = standard_detector.detect_hydrogen_bonds(atoms, positions)
    
    # Strict parameters
    strict_detector = HydrogenBondDetector(
        max_distance=3.0,   # Stricter distance
        min_angle=140.0,    # Stricter angle
        max_h_distance=2.0  # Stricter H-A distance
    )
    strict_bonds = strict_detector.detect_hydrogen_bonds(atoms, positions)
    
    # Loose parameters
    loose_detector = HydrogenBondDetector(
        max_distance=4.0,   # More permissive distance
        min_angle=100.0,    # More permissive angle
        max_h_distance=3.0  # More permissive H-A distance
    )
    loose_bonds = loose_detector.detect_hydrogen_bonds(atoms, positions)
    
    print("Detection with different parameters:")
    print(f"  Standard parameters: {len(standard_bonds)} H-bonds")
    print(f"  Strict parameters: {len(strict_bonds)} H-bonds")
    print(f"  Loose parameters: {len(loose_bonds)} H-bonds")
    
    # Show parameter effects
    print("\nParameter effects on bond classification:")
    
    for i, bond in enumerate(standard_bonds):
        strict_strength = strict_detector.classify_bond_strength(
            bond.distance, bond.angle
        )
        loose_strength = loose_detector.classify_bond_strength(
            bond.distance, bond.angle
        )
        
        print(f"  Bond {i+1} (d={bond.distance:.2f}Å, α={bond.angle:.1f}°):")
        print(f"    Standard: {bond.strength}")
        print(f"    Strict: {strict_strength}")
        print(f"    Loose: {loose_strength}")


def demo_performance_test():
    """Demonstrate performance with different system sizes."""
    print("=" * 60)
    print("DEMO 6: Performance Testing")
    print("=" * 60)
    
    import time
    
    system_sizes = [20, 50, 100, 200]
    
    print("Performance test with different system sizes:")
    print("(Note: Using simplified random structures)")
    
    for n_atoms in system_sizes:
        # Create random system
        atoms = []
        for i in range(n_atoms):
            element = 'N' if i % 4 == 0 else ('H' if i % 4 == 1 else 'O')
            atoms.append(DemoAtom(i, element, element, i // 10))
        
        # Random positions
        np.random.seed(42)  # Reproducible
        positions = np.random.random((n_atoms, 3)) * 20.0  # 20Å box
        
        # Time the detection
        detector = HydrogenBondDetector()
        start_time = time.time()
        bonds = detector.detect_hydrogen_bonds(atoms, positions)
        end_time = time.time()
        
        elapsed = end_time - start_time
        bonds_per_second = len(bonds) / elapsed if elapsed > 0 else float('inf')
        
        print(f"  {n_atoms:3d} atoms: {len(bonds):3d} H-bonds in {elapsed:.3f}s "
              f"({bonds_per_second:.1f} bonds/s)")


def demo_quick_analysis():
    """Demonstrate quick analysis function."""
    print("=" * 60)
    print("DEMO 7: Quick Analysis Function")
    print("=" * 60)
    
    atoms = create_demo_protein()
    trajectory = create_dynamic_trajectory(atoms, n_frames=10)
    
    print("Using quick_hydrogen_bond_summary function...")
    
    # Quick analysis
    summary = quick_hydrogen_bond_summary(atoms, trajectory)
    
    print("Quick summary results:")
    print(f"  Mean H-bonds per frame: {summary['mean_bonds_per_frame']:.2f}")
    print(f"  Bond types found: {list(summary['bond_type_distribution'].keys())}")
    print(f"  Bond strengths found: {list(summary['bond_strength_distribution'].keys())}")
    
    if 'lifetime_statistics' in summary:
        print(f"  Mean H-bond lifetime: {summary['lifetime_statistics']['mean_lifetime']:.2f} frames")


def main():
    """Run all demonstration examples."""
    print("HYDROGEN BOND ANALYSIS MODULE DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the capabilities of the hydrogen bond analysis module")
    print("including detection, trajectory analysis, visualization, and export.")
    print()
    
    try:
        # Demo 1: Single structure analysis
        bonds = demo_single_structure_analysis()
        
        # Demo 2: Trajectory analysis
        analyzer = demo_trajectory_analysis()
        
        # Demo 3: Visualization (optional - requires display)
        try:
            demo_visualization(analyzer)
        except Exception as e:
            print(f"Visualization demo skipped: {e}")
        
        # Demo 4: Data export
        demo_data_export(analyzer)
        
        # Demo 5: Custom parameters
        demo_custom_parameters()
        
        # Demo 6: Performance testing
        demo_performance_test()
        
        # Demo 7: Quick analysis
        demo_quick_analysis()
        
        print("\n" + "=" * 60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe hydrogen bond analysis module provides:")
        print("✓ Accurate geometric H-bond detection")
        print("✓ Comprehensive trajectory analysis")
        print("✓ Statistical lifetime calculations")
        print("✓ Multiple visualization options")
        print("✓ Flexible data export (CSV/JSON)")
        print("✓ Customizable detection parameters")
        print("✓ Good performance for typical MD systems")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
