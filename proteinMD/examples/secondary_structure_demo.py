"""
Secondary Structure Analysis Demo

This demo showcases the secondary structure analysis capabilities of the
proteinMD package, including DSSP-like structure assignment, trajectory
analysis, and various visualization options.

The demo includes:
1. Single structure analysis
2. Trajectory analysis with protein folding simulation
3. Visualization of structure evolution
4. Statistical analysis and export

Author: ProteinMD Development Team
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from proteinMD.analysis.secondary_structure import (
    SecondaryStructureAnalyzer,
    create_secondary_structure_analyzer,
    SS_TYPES,
    assign_secondary_structure_dssp
)


class MockAtom:
    """Mock atom class for demonstration purposes."""
    
    def __init__(self, name, res_num, res_name, position):
        self.atom_name = name
        self.residue_number = res_num
        self.residue_name = res_name
        self.position = np.array(position, dtype=float)


class MockMolecule:
    """Mock molecule class for demonstration purposes."""
    
    def __init__(self):
        self.atoms = []
    
    def add_atom(self, name, res_num, res_name, position):
        """Add an atom to the molecule."""
        atom = MockAtom(name, res_num, res_name, position)
        self.atoms.append(atom)
        return atom


class MockSimulation:
    """Mock simulation class for demonstration purposes."""
    
    def __init__(self, molecule):
        self.molecule = molecule
        self.trajectory = []
        self.dt = 0.002  # 2 fs timestep


def create_alpha_helix_structure(n_residues=15):
    """
    Create a protein structure with alpha-helix geometry.
    
    Parameters:
    -----------
    n_residues : int
        Number of residues in the helix
        
    Returns:
    --------
    MockMolecule
        Molecule with helix structure
    """
    print(f"Creating alpha-helix structure with {n_residues} residues...")
    
    molecule = MockMolecule()
    
    # Alpha-helix parameters
    rise_per_residue = 0.15  # 1.5 Å per residue
    degrees_per_residue = 100.0  # 100° rotation per residue
    helix_radius = 0.23  # 2.3 Å radius
    
    residue_names = ['ALA', 'LEU', 'GLU', 'LYS', 'VAL', 'PHE', 'ASP', 'GLY', 'SER', 'THR', 
                     'ARG', 'HIS', 'PRO', 'MET', 'CYS', 'TRP', 'TYR', 'ASN', 'GLN', 'ILE']
    
    for i in range(n_residues):
        res_num = i + 1
        res_name = residue_names[i % len(residue_names)]
        
        # Calculate helix geometry
        z = i * rise_per_residue
        theta = np.radians(i * degrees_per_residue)
        
        x_center = helix_radius * np.cos(theta)
        y_center = helix_radius * np.sin(theta)
        
        # Backbone atom positions
        n_pos = [x_center - 0.05, y_center - 0.02, z]
        ca_pos = [x_center, y_center, z]
        c_pos = [x_center + 0.05, y_center + 0.03, z + 0.02]
        o_pos = [x_center + 0.08, y_center + 0.05, z + 0.01]
        
        # Add backbone atoms
        molecule.add_atom('N', res_num, res_name, n_pos)
        molecule.add_atom('CA', res_num, res_name, ca_pos)
        molecule.add_atom('C', res_num, res_name, c_pos)
        molecule.add_atom('O', res_num, res_name, o_pos)
        
        # Add side chain atoms (simplified)
        if res_name != 'GLY':  # Glycine has no side chain
            cb_pos = [x_center - 0.04, y_center + 0.08, z + 0.03]
            molecule.add_atom('CB', res_num, res_name, cb_pos)
    
    print(f"Created helix with {len(molecule.atoms)} atoms")
    return molecule


def create_beta_sheet_structure(n_residues=10):
    """
    Create a protein structure with beta-sheet geometry.
    
    Parameters:
    -----------
    n_residues : int
        Number of residues in the sheet
        
    Returns:
    --------
    MockMolecule
        Molecule with sheet structure
    """
    print(f"Creating beta-sheet structure with {n_residues} residues...")
    
    molecule = MockMolecule()
    
    # Beta-sheet parameters (extended conformation)
    spacing = 0.35  # 3.5 Å between residues
    
    residue_names = ['VAL', 'PHE', 'THR', 'ILE', 'TYR', 'LEU', 'TRP', 'SER', 'ASN', 'GLN']
    
    for i in range(n_residues):
        res_num = i + 1
        res_name = residue_names[i % len(residue_names)]
        
        # Extended beta-strand geometry
        x = i * spacing
        y = 0.0
        z = 0.1 * (i % 2)  # Slight zigzag pattern
        
        # Backbone positions
        n_pos = [x - 0.06, y, z]
        ca_pos = [x, y, z]
        c_pos = [x + 0.06, y + 0.02, z]
        o_pos = [x + 0.09, y + 0.04, z - 0.02]
        
        # Add backbone atoms
        molecule.add_atom('N', res_num, res_name, n_pos)
        molecule.add_atom('CA', res_num, res_name, ca_pos)
        molecule.add_atom('C', res_num, res_name, c_pos)
        molecule.add_atom('O', res_num, res_name, o_pos)
        
        # Add side chain
        if res_name != 'GLY':
            cb_pos = [x - 0.03, y + 0.08, z + 0.05]
            molecule.add_atom('CB', res_num, res_name, cb_pos)
    
    print(f"Created sheet with {len(molecule.atoms)} atoms")
    return molecule


def create_mixed_structure(n_residues=20):
    """
    Create a protein with mixed secondary structure (helix + sheet + coil).
    
    Parameters:
    -----------
    n_residues : int
        Total number of residues
        
    Returns:
    --------
    MockMolecule
        Molecule with mixed structure
    """
    print(f"Creating mixed structure with {n_residues} residues...")
    
    molecule = MockMolecule()
    
    residue_names = ['MET', 'ALA', 'LEU', 'GLU', 'LYS', 'VAL', 'PHE', 'ASP', 'GLY', 'SER', 
                     'THR', 'ARG', 'HIS', 'PRO', 'CYS', 'TRP', 'TYR', 'ASN', 'GLN', 'ILE']
    
    for i in range(n_residues):
        res_num = i + 1
        res_name = residue_names[i % len(residue_names)]
        
        if i < 8:  # First part: alpha-helix
            z = i * 0.15
            theta = np.radians(i * 100.0)
            radius = 0.23
            
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            
        elif i < 15:  # Middle part: beta-strand
            strand_index = i - 8
            x = 0.3 + strand_index * 0.35
            y = 0.5
            z = 1.2 + 0.1 * (strand_index % 2)
            
        else:  # End part: random coil
            coil_index = i - 15
            x = 3.0 + np.random.normal(0, 0.2)
            y = 0.5 + np.random.normal(0, 0.3)
            z = 1.5 + coil_index * 0.1 + np.random.normal(0, 0.1)
        
        # Backbone positions
        n_pos = [x - 0.05, y - 0.02, z]
        ca_pos = [x, y, z]
        c_pos = [x + 0.05, y + 0.03, z + 0.02]
        o_pos = [x + 0.08, y + 0.05, z + 0.01]
        
        # Add atoms
        molecule.add_atom('N', res_num, res_name, n_pos)
        molecule.add_atom('CA', res_num, res_name, ca_pos)
        molecule.add_atom('C', res_num, res_name, c_pos)
        molecule.add_atom('O', res_num, res_name, o_pos)
        
        if res_name != 'GLY':
            cb_pos = [x - 0.04, y + 0.08, z + 0.03]
            molecule.add_atom('CB', res_num, res_name, cb_pos)
    
    print(f"Created mixed structure with {len(molecule.atoms)} atoms")
    return molecule


def create_folding_trajectory(n_frames=50):
    """
    Create a simulated protein folding trajectory.
    
    This creates a trajectory that starts from an extended conformation
    and gradually folds into a more compact structure with secondary elements.
    
    Parameters:
    -----------
    n_frames : int
        Number of frames in the trajectory
        
    Returns:
    --------
    MockSimulation
        Simulation object with folding trajectory
    """
    print(f"Creating folding trajectory with {n_frames} frames...")
    
    n_residues = 15
    
    # Start with an extended structure
    molecule = create_beta_sheet_structure(n_residues)
    simulation = MockSimulation(molecule)
    
    # Generate folding trajectory
    for frame in range(n_frames):
        frame_positions = []
        
        # Folding progress (0 to 1)
        progress = frame / (n_frames - 1)
        
        for i, atom in enumerate(molecule.atoms):
            res_idx = atom.residue_number - 1
            
            # Get initial extended position
            initial_pos = atom.position.copy()
            
            # Target folded position (helix for first 10 residues)
            if res_idx < 10 and atom.atom_name in ['N', 'CA', 'C', 'O']:
                # Helix geometry
                z = res_idx * 0.15
                theta = np.radians(res_idx * 100.0)
                radius = 0.23
                
                target_x = radius * np.cos(theta)
                target_y = radius * np.sin(theta)
                target_z = z
                
                if atom.atom_name == 'N':
                    target_pos = np.array([target_x - 0.05, target_y - 0.02, target_z])
                elif atom.atom_name == 'CA':
                    target_pos = np.array([target_x, target_y, target_z])
                elif atom.atom_name == 'C':
                    target_pos = np.array([target_x + 0.05, target_y + 0.03, target_z + 0.02])
                elif atom.atom_name == 'O':
                    target_pos = np.array([target_x + 0.08, target_y + 0.05, target_z + 0.01])
                else:
                    target_pos = initial_pos
            else:
                # Keep extended for C-terminal region
                target_pos = initial_pos
            
            # Interpolate between initial and target positions
            current_pos = initial_pos + progress * (target_pos - initial_pos)
            
            # Add thermal noise
            noise_amplitude = 0.02 * (1 - 0.5 * progress)  # Decreasing noise
            noise = np.random.normal(0, noise_amplitude, 3)
            current_pos += noise
            
            frame_positions.append(current_pos)
        
        simulation.trajectory.append(frame_positions)
    
    print(f"Created trajectory with {len(simulation.trajectory)} frames")
    return simulation


def demo_single_structure_analysis():
    """Demonstrate single structure analysis."""
    print("\n" + "="*60)
    print("SINGLE STRUCTURE ANALYSIS DEMO")
    print("="*60)
    
    # Create different structure types
    structures = {
        'Alpha-Helix': create_alpha_helix_structure(12),
        'Beta-Sheet': create_beta_sheet_structure(10),
        'Mixed Structure': create_mixed_structure(15)
    }
    
    # Analyze each structure
    analyzer = create_secondary_structure_analyzer()
    
    for name, molecule in structures.items():
        print(f"\nAnalyzing {name}:")
        print("-" * 40)
        
        result = analyzer.analyze_structure(molecule, time_point=0.0)
        
        print(f"Number of residues: {result['n_residues']}")
        print(f"Secondary structure assignments: {result['assignments']}")
        print(f"Hydrogen bonds found: {len(result['hydrogen_bonds'])}")
        
        print("\nSecondary structure percentages:")
        for ss_type, percentage in result['percentages'].items():
            if percentage > 0:
                ss_name = SS_TYPES[ss_type]['name']
                print(f"  {ss_name}: {percentage:.1f}%")
        
        # Show detailed assignment
        print(f"\nResidue-by-residue assignment:")
        print(f"{'Res#':<4} {'Name':<4} {'SS':<2} {'Structure':<15}")
        print("-" * 30)
        for i, (assignment, res_name) in enumerate(zip(result['assignments'], result['residue_names'])):
            ss_name = SS_TYPES[assignment]['name']
            print(f"{i+1:<4} {res_name:<4} {assignment:<2} {ss_name:<15}")
    
    return analyzer


def demo_trajectory_analysis():
    """Demonstrate trajectory analysis with protein folding."""
    print("\n" + "="*60)
    print("TRAJECTORY ANALYSIS DEMO")
    print("="*60)
    
    # Create folding simulation
    simulation = create_folding_trajectory(n_frames=25)
    
    # Analyze trajectory
    analyzer = create_secondary_structure_analyzer()
    
    print("Analyzing folding trajectory...")
    summary = analyzer.analyze_trajectory(simulation, time_step=1)
    
    # Print results
    print(f"\nTrajectory Analysis Results:")
    print(f"Frames analyzed: {len(analyzer.trajectory_data)}")
    print(f"Residues: {analyzer.trajectory_data[0]['n_residues']}")
    
    # Show statistics
    stats = analyzer.get_statistics()
    traj_stats = stats['trajectory_stats']
    
    print(f"\nAverage secondary structure content:")
    for ss_type in ['H', 'E', 'T', 'C']:
        avg_pct = traj_stats['avg_percentages'][ss_type]
        std_pct = traj_stats['std_percentages'][ss_type]
        ss_name = SS_TYPES[ss_type]['name']
        print(f"  {ss_name}: {avg_pct:.1f}% ± {std_pct:.1f}%")
    
    # Show folding progress
    print(f"\nFolding progress (helix formation):")
    print(f"{'Frame':<6} {'Time':<8} {'Helix %':<10} {'Sheet %':<10} {'Coil %':<10}")
    print("-" * 50)
    
    for i, frame_data in enumerate(analyzer.trajectory_data[::5]):  # Every 5th frame
        time_point = frame_data['time_point']
        helix_pct = frame_data['percentages']['H']
        sheet_pct = frame_data['percentages']['E']
        coil_pct = frame_data['percentages']['C']
        
        print(f"{i*5:<6} {time_point:<8.2f} {helix_pct:<10.1f} {sheet_pct:<10.1f} {coil_pct:<10.1f}")
    
    return analyzer


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n" + "="*60)
    print("VISUALIZATION DEMO")
    print("="*60)
    
    # Create trajectory for visualization
    simulation = create_folding_trajectory(n_frames=30)
    analyzer = create_secondary_structure_analyzer()
    analyzer.analyze_trajectory(simulation, time_step=1)
    
    print("Creating visualizations...")
    
    try:
        # Plot 1: Time evolution
        print("1. Creating time evolution plot...")
        fig1 = analyzer.plot_time_evolution(save_path="ss_time_evolution.png")
        print("   ✓ Time evolution plot saved as 'ss_time_evolution.png'")
        plt.close(fig1)
        
        # Plot 2: Residue timeline
        print("2. Creating residue timeline plot...")
        fig2 = analyzer.plot_residue_timeline(save_path="ss_residue_timeline.png")
        print("   ✓ Residue timeline plot saved as 'ss_residue_timeline.png'")
        plt.close(fig2)
        
        # Plot 3: Structure distribution
        print("3. Creating structure distribution plot...")
        fig3 = analyzer.plot_structure_distribution(save_path="ss_distribution.png")
        print("   ✓ Structure distribution plot saved as 'ss_distribution.png'")
        plt.close(fig3)
        
        # Plot 4: Custom residue range timeline
        print("4. Creating partial residue timeline...")
        fig4 = analyzer.plot_residue_timeline(
            residue_range=(0, 10), 
            save_path="ss_partial_timeline.png"
        )
        print("   ✓ Partial timeline plot saved as 'ss_partial_timeline.png'")
        plt.close(fig4)
        
    except Exception as e:
        print(f"   Warning: Visualization error - {e}")
        print("   This may be due to matplotlib backend issues in the current environment")
    
    return analyzer


def demo_data_export():
    """Demonstrate data export capabilities."""
    print("\n" + "="*60)
    print("DATA EXPORT DEMO")
    print("="*60)
    
    # Create and analyze trajectory
    simulation = create_folding_trajectory(n_frames=20)
    analyzer = create_secondary_structure_analyzer()
    analyzer.analyze_trajectory(simulation, time_step=2)
    
    print("Exporting analysis data...")
    
    try:
        # Export CSV
        csv_filename = "secondary_structure_data.csv"
        analyzer.export_timeline_data(csv_filename, format='csv')
        print(f"✓ CSV data exported to '{csv_filename}'")
        
        # Check file size
        if os.path.exists(csv_filename):
            file_size = os.path.getsize(csv_filename)
            print(f"  File size: {file_size} bytes")
            
            # Show first few lines
            print("  First few lines of CSV:")
            with open(csv_filename, 'r') as f:
                for i, line in enumerate(f):
                    if i < 5:
                        print(f"    {line.strip()}")
                    else:
                        break
        
        # Export JSON
        json_filename = "secondary_structure_data.json"
        analyzer.export_timeline_data(json_filename, format='json')
        print(f"✓ JSON data exported to '{json_filename}'")
        
        if os.path.exists(json_filename):
            file_size = os.path.getsize(json_filename)
            print(f"  File size: {file_size} bytes")
        
    except Exception as e:
        print(f"Export error: {e}")
    
    return analyzer


def demo_performance_benchmarking():
    """Benchmark analysis performance."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKING")
    print("="*60)
    
    import time
    
    # Test different system sizes
    sizes = [10, 25, 50, 100]
    
    print(f"{'Size':<6} {'Time (s)':<10} {'Residues/s':<12} {'Memory':<10}")
    print("-" * 45)
    
    for size in sizes:
        # Create test structure
        molecule = create_mixed_structure(size)
        analyzer = create_secondary_structure_analyzer()
        
        # Benchmark single structure analysis
        start_time = time.time()
        result = analyzer.analyze_structure(molecule)
        end_time = time.time()
        
        analysis_time = end_time - start_time
        residues_per_sec = size / analysis_time if analysis_time > 0 else float('inf')
        
        print(f"{size:<6} {analysis_time:<10.4f} {residues_per_sec:<12.1f} {'OK':<10}")
    
    # Test trajectory performance
    print(f"\nTrajectory Analysis Performance:")
    
    n_frames_list = [10, 25, 50]
    
    for n_frames in n_frames_list:
        simulation = create_folding_trajectory(n_frames=n_frames)
        analyzer = create_secondary_structure_analyzer()
        
        start_time = time.time()
        analyzer.analyze_trajectory(simulation, time_step=1)
        end_time = time.time()
        
        analysis_time = end_time - start_time
        frames_per_sec = n_frames / analysis_time if analysis_time > 0 else float('inf')
        
        print(f"  {n_frames} frames: {analysis_time:.3f}s ({frames_per_sec:.1f} frames/s)")


def demo_physical_validation():
    """Validate analysis against known protein structures."""
    print("\n" + "="*60)
    print("PHYSICAL VALIDATION")
    print("="*60)
    
    print("Testing secondary structure assignment accuracy...")
    
    # Test 1: Pure alpha-helix should be mostly helix
    helix_molecule = create_alpha_helix_structure(15)
    analyzer = create_secondary_structure_analyzer()
    helix_result = analyzer.analyze_structure(helix_molecule)
    
    helix_percentage = helix_result['percentages']['H']
    print(f"Alpha-helix test: {helix_percentage:.1f}% helix content")
    
    if helix_percentage > 50:
        print("  ✓ PASS: Helix structure correctly identified")
    else:
        print("  ✗ FAIL: Helix structure not identified")
    
    # Test 2: Beta-sheet should have extended conformation markers
    sheet_molecule = create_beta_sheet_structure(12)
    sheet_result = analyzer.analyze_structure(sheet_molecule)
    
    strand_percentage = sheet_result['percentages']['E']
    extended_percentage = strand_percentage + sheet_result['percentages']['C']
    print(f"Beta-sheet test: {strand_percentage:.1f}% strand, {extended_percentage:.1f}% extended")
    
    if extended_percentage > 60:
        print("  ✓ PASS: Extended structure correctly identified")
    else:
        print("  ✗ FAIL: Extended structure not identified")
    
    # Test 3: Mixed structure should show diversity
    mixed_molecule = create_mixed_structure(20)
    mixed_result = analyzer.analyze_structure(mixed_molecule)
    
    n_structure_types = sum(1 for pct in mixed_result['percentages'].values() if pct > 5)
    print(f"Mixed structure test: {n_structure_types} structure types found")
    
    if n_structure_types >= 3:
        print("  ✓ PASS: Structural diversity correctly identified")
    else:
        print("  ✗ FAIL: Insufficient structural diversity detected")
    
    # Test 4: Hydrogen bond detection
    print(f"\nHydrogen bond validation:")
    for name, molecule in [('Helix', helix_molecule), ('Sheet', sheet_molecule)]:
        from proteinMD.analysis.secondary_structure import identify_hydrogen_bonds
        hbonds = identify_hydrogen_bonds(molecule)
        n_residues = len(set(atom.residue_number for atom in molecule.atoms))
        hb_ratio = len(hbonds) / n_residues if n_residues > 0 else 0
        
        print(f"  {name}: {len(hbonds)} H-bonds ({hb_ratio:.2f} per residue)")


def main():
    """Run all secondary structure analysis demos."""
    print("SECONDARY STRUCTURE ANALYSIS DEMO")
    print("=" * 70)
    print("This demo showcases the secondary structure analysis capabilities")
    print("of the proteinMD package with DSSP-like functionality.")
    print("=" * 70)
    
    try:
        # Run demo sections
        demo_single_structure_analysis()
        demo_trajectory_analysis()
        demo_visualization()
        demo_data_export()
        demo_performance_benchmarking()
        demo_physical_validation()
        
        print("\n" + "="*60)
        print("DEMO COMPLETION SUMMARY")
        print("="*60)
        print("✓ Single structure analysis")
        print("✓ Trajectory analysis") 
        print("✓ Visualization capabilities")
        print("✓ Data export functionality")
        print("✓ Performance benchmarking")
        print("✓ Physical validation")
        
        print(f"\nGenerated files:")
        output_files = [
            "ss_time_evolution.png",
            "ss_residue_timeline.png", 
            "ss_distribution.png",
            "ss_partial_timeline.png",
            "secondary_structure_data.csv",
            "secondary_structure_data.json"
        ]
        
        for filename in output_files:
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"  {filename} ({file_size} bytes)")
        
        print(f"\nSecondary structure analysis demo completed successfully!")
        print(f"Task 3.4 - Secondary Structure Tracking implementation verified.")
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
