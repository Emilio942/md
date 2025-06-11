"""
Ramachandran Plot Analysis Demo

This script demonstrates the Ramachandran plot analysis capabilities,
including phi-psi angle calculation, visualization, and trajectory analysis.

Author: ProteinMD Development Team
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add proteinMD to path
sys.path.insert(0, '/home/emilio/Documents/ai/md')

from proteinMD.analysis.ramachandran import (
    RamachandranAnalyzer, 
    create_ramachandran_analyzer,
    calculate_dihedral_angle,
    AMINO_ACIDS
)

class MockAtom:
    """Mock atom class for demonstration."""
    def __init__(self, name, residue_number, residue_name, position):
        self.name = name
        self.residue_number = residue_number
        self.residue_name = residue_name
        self.position = np.array(position)

class MockMolecule:
    """Mock molecule class for demonstration."""
    def __init__(self):
        self.atoms = []
    
    def add_atom(self, name, residue_number, residue_name, position):
        """Add an atom to the molecule."""
        atom = MockAtom(name, residue_number, residue_name, position)
        self.atoms.append(atom)

def create_demo_protein():
    """Create a demo protein structure for analysis."""
    print("Creating demo protein structure...")
    
    molecule = MockMolecule()
    
    # Create a small peptide with various conformations
    # Simulate alpha-helix, beta-sheet, and random coil regions
    
    residues = [
        ('ALA', 'alpha'),    # Alpha helix region
        ('GLU', 'alpha'),
        ('LEU', 'alpha'),
        ('LYS', 'alpha'),
        ('VAL', 'beta'),     # Beta sheet region
        ('PHE', 'beta'),
        ('THR', 'beta'),
        ('GLY', 'random'),   # Random coil region
        ('PRO', 'random'),
        ('SER', 'random')
    ]
    
    for i, (residue_name, conformation) in enumerate(residues):
        residue_num = i + 1
        
        # Base positions along a chain
        base_x = i * 3.8  # ~3.8 Å spacing between residues
        
        # Adjust positions based on secondary structure
        if conformation == 'alpha':
            # Alpha helix: phi ~ -60°, psi ~ -45°
            ca_y = 0.5 * np.sin(i * 0.6)
            ca_z = 1.5 * i * np.sin(i * 0.3)
            phi_offset = -60 + np.random.normal(0, 10)
            psi_offset = -45 + np.random.normal(0, 10)
        elif conformation == 'beta':
            # Beta sheet: phi ~ -120°, psi ~ 120°
            ca_y = 0.2 * (i % 2)  # Slight zigzag
            ca_z = 0.1 * i
            phi_offset = -120 + np.random.normal(0, 15)
            psi_offset = 120 + np.random.normal(0, 15)
        else:  # random
            # Random coil: varied angles
            ca_y = np.random.normal(0, 0.5)
            ca_z = np.random.normal(0, 0.5)
            phi_offset = np.random.uniform(-180, 180)
            psi_offset = np.random.uniform(-180, 180)
        
        # Calculate backbone positions
        # These are approximate positions to give realistic phi/psi angles
        n_pos = [base_x - 1.2, ca_y - 0.3, ca_z]
        ca_pos = [base_x, ca_y, ca_z]
        c_pos = [base_x + 1.2, ca_y + 0.5, ca_z + 0.2]
        
        molecule.add_atom('N', residue_num, residue_name, n_pos)
        molecule.add_atom('CA', residue_num, residue_name, ca_pos)
        molecule.add_atom('C', residue_num, residue_name, c_pos)
        
        # Add some side chain atoms (simplified)
        if residue_name != 'GLY':  # Glycine has no side chain
            cb_pos = [base_x + 0.5, ca_y - 0.8, ca_z + 0.5]
            molecule.add_atom('CB', residue_num, residue_name, cb_pos)
    
    print(f"Created protein with {len(residues)} residues and {len(molecule.atoms)} atoms")
    return molecule

def demo_basic_analysis():
    """Demonstrate basic Ramachandran analysis."""
    print("\n" + "="*60)
    print("BASIC RAMACHANDRAN ANALYSIS DEMO")
    print("="*60)
    
    # Create demo protein
    molecule = create_demo_protein()
    
    # Create analyzer and analyze structure
    analyzer = create_ramachandran_analyzer(molecule)
    
    result = analyzer.analyze_structure(molecule)
    
    print(f"\nAnalysis Results:")
    print(f"- Number of residues analyzed: {result['statistics']['n_residues']}")
    print(f"- Phi angle range: {result['statistics']['phi_range'][0]:.1f}° to {result['statistics']['phi_range'][1]:.1f}°")
    print(f"- Psi angle range: {result['statistics']['psi_range'][0]:.1f}° to {result['statistics']['psi_range'][1]:.1f}°")
    print(f"- Average phi angle: {result['statistics']['phi_mean']:.1f}° ± {result['statistics']['phi_std']:.1f}°")
    print(f"- Average psi angle: {result['statistics']['psi_mean']:.1f}° ± {result['statistics']['psi_std']:.1f}°")
    
    # Print individual residue angles
    print(f"\nIndividual Residue Angles:")
    print(f"{'Residue':<8} {'Phi':<8} {'Psi':<8}")
    print("-" * 24)
    for i, (phi, psi, res) in enumerate(zip(result['phi_angles'], result['psi_angles'], result['residue_names'])):
        print(f"{res:<8} {phi:7.1f}° {psi:7.1f}°")
    
    return analyzer

def demo_ramachandran_plots(analyzer):
    """Demonstrate Ramachandran plot generation."""
    print("\n" + "="*60)
    print("RAMACHANDRAN PLOT GENERATION DEMO")
    print("="*60)
    
    # Create basic Ramachandran plot
    print("\nGenerating basic Ramachandran plot...")
    fig1 = analyzer.plot_ramachandran(
        color_by_residue=False,
        show_regions=True,
        title="Basic Ramachandran Plot"
    )
    
    # Create colored Ramachandran plot
    print("Generating color-coded Ramachandran plot...")
    fig2 = analyzer.plot_ramachandran(
        color_by_residue=True,
        show_regions=True,
        title="Ramachandran Plot - Colored by Amino Acid"
    )
    
    # Show plots
    plt.show()
    
    # Save plots
    output_dir = Path("/home/emilio/Documents/ai/md/proteinMD/examples/output")
    output_dir.mkdir(exist_ok=True)
    
    fig1.savefig(output_dir / "ramachandran_basic.png", dpi=300, bbox_inches='tight')
    fig2.savefig(output_dir / "ramachandran_colored.png", dpi=300, bbox_inches='tight')
    
    print(f"Plots saved to {output_dir}/")
    
    plt.close('all')

def demo_trajectory_analysis():
    """Demonstrate trajectory analysis with time evolution."""
    print("\n" + "="*60)
    print("TRAJECTORY ANALYSIS DEMO")
    print("="*60)
    
    analyzer = RamachandranAnalyzer()
    
    # Simulate a trajectory with conformational changes
    print("Simulating molecular dynamics trajectory...")
    
    base_molecule = create_demo_protein()
    n_frames = 20
    
    for frame in range(n_frames):
        # Create a copy of the molecule with slightly perturbed positions
        molecule = MockMolecule()
        
        for atom in base_molecule.atoms:
            # Add small random perturbations to simulate thermal motion
            noise = np.random.normal(0, 0.1, 3)  # 0.1 Å standard deviation
            
            # Add systematic changes to simulate conformational transitions
            systematic_change = np.array([0.0, 0.0, 0.0])
            if atom.name == 'CA':
                # Gradual conformational change
                systematic_change[1] = 0.05 * frame * np.sin(frame * 0.3)
                systematic_change[2] = 0.03 * frame * np.cos(frame * 0.2)
            
            new_position = atom.position + noise + systematic_change
            molecule.add_atom(atom.name, atom.residue_number, 
                            atom.residue_name, new_position)
        
        # Analyze this frame
        time_point = frame * 0.1  # 0.1 ps per frame
        analyzer.analyze_structure(molecule, time_point)
    
    # Calculate trajectory statistics
    traj_stats = analyzer._calculate_trajectory_statistics()
    
    print(f"\nTrajectory Analysis Results:")
    print(f"- Total frames analyzed: {traj_stats['n_frames']}")
    print(f"- Total angle measurements: {traj_stats['total_angles']}")
    print(f"- Phi distribution: {traj_stats['phi_distribution']['mean']:.1f}° ± {traj_stats['phi_distribution']['std']:.1f}°")
    print(f"- Psi distribution: {traj_stats['psi_distribution']['mean']:.1f}° ± {traj_stats['psi_distribution']['std']:.1f}°")
    
    # Plot trajectory evolution
    print("\nGenerating angle evolution plots...")
    fig_evolution = analyzer.plot_angle_evolution(residue_index=2)  # Middle residue
    
    # Plot final Ramachandran plot from trajectory
    print("Generating trajectory Ramachandran plot...")
    fig_traj = analyzer.plot_ramachandran(
        color_by_residue=True,
        title="Ramachandran Plot - MD Trajectory"
    )
    
    plt.show()
    
    # Save plots
    output_dir = Path("/home/emilio/Documents/ai/md/proteinMD/examples/output")
    fig_evolution.savefig(output_dir / "angle_evolution.png", dpi=300, bbox_inches='tight')
    fig_traj.savefig(output_dir / "ramachandran_trajectory.png", dpi=300, bbox_inches='tight')
    
    print(f"Trajectory plots saved to {output_dir}/")
    plt.close('all')
    
    return analyzer

def demo_data_export(analyzer):
    """Demonstrate data export capabilities."""
    print("\n" + "="*60)
    print("DATA EXPORT DEMO")
    print("="*60)
    
    output_dir = Path("/home/emilio/Documents/ai/md/proteinMD/examples/output")
    output_dir.mkdir(exist_ok=True)
    
    # Export CSV data
    csv_path = output_dir / "ramachandran_data.csv"
    analyzer.export_data(csv_path, format='csv')
    print(f"CSV data exported to: {csv_path}")
    
    # Export JSON data
    json_path = output_dir / "ramachandran_data.json"
    analyzer.export_data(json_path, format='json')
    print(f"JSON data exported to: {json_path}")
    
    # Show file contents preview
    print(f"\nCSV file preview:")
    with open(csv_path, 'r') as f:
        lines = f.readlines()[:6]  # First 6 lines
        for line in lines:
            print(f"  {line.strip()}")
        if len(lines) == 6:
            print("  ...")

def demo_amino_acid_properties():
    """Demonstrate amino acid property analysis."""
    print("\n" + "="*60)
    print("AMINO ACID PROPERTIES DEMO")
    print("="*60)
    
    print("Standard amino acid properties:")
    print(f"{'Code':<4} {'Name':<15} {'Single':<6} {'Color'}")
    print("-" * 40)
    
    for code, props in AMINO_ACIDS.items():
        print(f"{code:<4} {props['name']:<15} {props['code']:<6} {props['color']}")
    
    print(f"\nTotal amino acids defined: {len(AMINO_ACIDS)}")

def benchmark_performance():
    """Benchmark the performance of Ramachandran analysis."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    import time
    
    sizes = [10, 50, 100, 200]
    
    print(f"{'Protein Size':<12} {'Analysis Time':<15} {'Plot Time':<12}")
    print("-" * 40)
    
    for size in sizes:
        # Create larger protein
        molecule = MockMolecule()
        
        for i in range(size):
            residue_num = i + 1
            residue_name = list(AMINO_ACIDS.keys())[i % 20]  # Cycle through amino acids
            
            # Simple linear arrangement
            base_x = i * 3.8
            n_pos = [base_x - 1.2, 0, 0]
            ca_pos = [base_x, 0, 0]
            c_pos = [base_x + 1.2, 0, 0]
            
            molecule.add_atom('N', residue_num, residue_name, n_pos)
            molecule.add_atom('CA', residue_num, residue_name, ca_pos)
            molecule.add_atom('C', residue_num, residue_name, c_pos)
        
        # Benchmark analysis
        analyzer = RamachandranAnalyzer()
        
        start_time = time.time()
        analyzer.analyze_structure(molecule)
        analysis_time = time.time() - start_time
        
        # Benchmark plotting
        start_time = time.time()
        fig = analyzer.plot_ramachandran()
        plt.close(fig)
        plot_time = time.time() - start_time
        
        print(f"{size:<12} {analysis_time*1000:10.2f} ms    {plot_time*1000:8.2f} ms")

def main():
    """Run all demonstration functions."""
    print("RAMACHANDRAN PLOT ANALYSIS - COMPREHENSIVE DEMO")
    print("=" * 80)
    print("This demo showcases the Ramachandran analysis capabilities of ProteinMD")
    
    # Create output directory
    output_dir = Path("/home/emilio/Documents/ai/md/proteinMD/examples/output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Basic analysis
        analyzer = demo_basic_analysis()
        
        # Plot generation
        demo_ramachandran_plots(analyzer)
        
        # Trajectory analysis
        traj_analyzer = demo_trajectory_analysis()
        
        # Data export
        demo_data_export(traj_analyzer)
        
        # Amino acid properties
        demo_amino_acid_properties()
        
        # Performance benchmark
        benchmark_performance()
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"All output files saved to: {output_dir}")
        print("\nKey features demonstrated:")
        print("✓ Phi-psi angle calculation")
        print("✓ Ramachandran plot generation")
        print("✓ Color coding by amino acid type")
        print("✓ Trajectory analysis and time evolution")
        print("✓ Data export (CSV/JSON)")
        print("✓ Performance benchmarking")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
