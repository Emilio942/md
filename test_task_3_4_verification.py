#!/usr/bin/env python3
"""
Task 3.4 Verification Script: Secondary Structure Tracking

This script verifies that Task 3.4 (Secondary Structure Tracking) meets all requirements:
✅ DSSP-ähnlicher Algorithmus implementiert
✅ Sekundärstrukturänderungen werden farbkodiert visualisiert  
✅ Zeitanteil verschiedener Strukturen wird berechnet
✅ Export der Sekundärstruktur-Timeline möglich

Author: ProteinMD Development Team
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the proteinMD directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'proteinMD'))

try:
    from analysis.secondary_structure import (
        SecondaryStructureAnalyzer, 
        assign_secondary_structure_dssp,
        SS_TYPES,
        calculate_dihedral_angle,
        identify_hydrogen_bonds
    )
    print("✅ All secondary structure modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Simple mock classes for testing
class MockAtom:
    def __init__(self, atom_type, position, residue_number, residue_name, atom_name):
        self.atom_type = atom_type
        self.position = np.array(position)
        self.residue_number = residue_number
        self.residue_name = residue_name
        self.atom_name = atom_name

class MockMolecule:
    def __init__(self):
        self.atoms = []
    
    def add_atom(self, atom):
        self.atoms.append(atom)


class MockSimulation:
    """Mock simulation for testing with realistic trajectory data."""
    
    def __init__(self, n_residues=20, n_frames=50):
        self.n_residues = n_residues
        self.n_frames = n_frames
        self.dt = 2.0  # ps per frame
        self.molecule = self._create_test_molecule()
        self.trajectory = self._create_test_trajectory()
    
    def _create_test_molecule(self):
        """Create a test molecule with realistic backbone geometry."""
        molecule = MockMolecule()
        
        # Create residues with backbone atoms
        for i in range(self.n_residues):
            res_num = i + 1
            res_name = 'ALA'
            
            # Calculate positions for alpha-helix-like structure
            phi_angle = -60.0 + np.random.normal(0, 10)  # degrees with noise
            psi_angle = -45.0 + np.random.normal(0, 10)
            
            z = i * 1.5  # 1.5 Å rise per residue
            theta = i * 100.0 * np.pi / 180.0  # 100° rotation per residue
            radius = 2.3 + np.random.normal(0, 0.1)  # helix radius with noise
            
            x_offset = radius * np.cos(theta)
            y_offset = radius * np.sin(theta)
            
            # Backbone atoms with realistic positions
            atoms_data = [
                ('N', [x_offset - 0.5, y_offset, z]),
                ('CA', [x_offset, y_offset, z]),
                ('C', [x_offset + 0.5, y_offset + 0.3, z + 0.2]),
                ('O', [x_offset + 0.8, y_offset + 0.5, z + 0.1])
            ]
            
            for atom_name, pos in atoms_data:
                atom = MockAtom(
                    atom_type=atom_name,
                    position=np.array(pos) / 10.0,  # Convert to nm
                    residue_number=res_num,
                    residue_name=res_name,
                    atom_name=atom_name
                )
                molecule.add_atom(atom)
        
        return molecule
    
    def _create_test_trajectory(self):
        """Create a realistic test trajectory with structural changes."""
        trajectory = []
        
        for frame in range(self.n_frames):
            frame_positions = []
            
            # Add some dynamics to the structure
            time_factor = frame / self.n_frames
            
            for atom in self.molecule.atoms:
                # Base position
                pos = atom.position.copy()
                
                # Add thermal motion
                thermal_noise = np.random.normal(0, 0.01, 3)  # 0.1 Å noise
                
                # Add some systematic changes (e.g., partial unfolding)
                if time_factor > 0.5:  # Halfway through, start unfolding
                    unfold_factor = (time_factor - 0.5) * 2.0
                    unfold_displacement = np.random.normal(0, 0.05 * unfold_factor, 3)
                    pos += unfold_displacement
                
                # Add breathing motion
                breathing = 0.005 * np.sin(2 * np.pi * frame / 20)  # 20-frame period
                pos *= (1 + breathing)
                
                frame_positions.append(pos + thermal_noise)
            
            trajectory.append(frame_positions)
        
        return trajectory


def test_dssp_algorithm():
    """Test requirement 1: DSSP-ähnlicher Algorithmus implementiert"""
    print("\n🧪 Testing DSSP-like Algorithm Implementation...")
    
    # Create test simulation
    sim = MockSimulation(n_residues=15, n_frames=1)
    
    # Test single structure assignment
    assignments = assign_secondary_structure_dssp(sim.molecule)
    
    # Verify assignments
    assert len(assignments) == sim.n_residues, f"Expected {sim.n_residues} assignments, got {len(assignments)}"
    
    valid_ss_types = set(SS_TYPES.keys())
    for ss in assignments:
        assert ss in valid_ss_types, f"Invalid secondary structure type: {ss}"
    
    # Test hydrogen bond identification
    hbonds = identify_hydrogen_bonds(sim.molecule)
    assert isinstance(hbonds, list), "Hydrogen bonds should be returned as a list"
    
    print(f"✅ DSSP algorithm working correctly")
    print(f"   - Assigned secondary structures: {assignments}")
    print(f"   - Found {len(hbonds)} hydrogen bonds")
    print(f"   - Structure distribution: {dict(zip(*np.unique(assignments, return_counts=True)))}")
    
    return True


def test_color_coded_visualization():
    """Test requirement 2: Sekundärstrukturänderungen werden farbkodiert visualisiert"""
    print("\n🎨 Testing Color-Coded Visualization...")
    
    # Create analyzer with trajectory
    sim = MockSimulation(n_residues=20, n_frames=30)
    analyzer = SecondaryStructureAnalyzer()
    
    # Analyze trajectory
    results = analyzer.analyze_trajectory(sim, time_step=1)
    
    # Test time evolution plot (color-coded)
    fig1 = analyzer.plot_time_evolution(
        figsize=(12, 8),
        save_path='/home/emilio/Documents/ai/md/test_task_3_4_time_evolution.png'
    )
    assert fig1 is not None, "Time evolution plot should be created"
    
    # Test residue timeline plot (color-coded visualization)
    fig2 = analyzer.plot_residue_timeline(
        residue_range=(1, 15),
        figsize=(14, 6),
        save_path='/home/emilio/Documents/ai/md/test_task_3_4_residue_timeline.png'
    )
    assert fig2 is not None, "Residue timeline plot should be created"
    
    # Test structure distribution plot
    fig3 = analyzer.plot_structure_distribution(
        figsize=(12, 8),
        save_path='/home/emilio/Documents/ai/md/test_task_3_4_distribution.png'
    )
    assert fig3 is not None, "Structure distribution plot should be created"
    
    plt.close('all')  # Clean up
    
    print(f"✅ Color-coded visualization working correctly")
    print(f"   - Time evolution plot: test_task_3_4_time_evolution.png")
    print(f"   - Residue timeline plot: test_task_3_4_residue_timeline.png")
    print(f"   - Distribution plot: test_task_3_4_distribution.png")
    
    return True


def test_structure_time_analysis():
    """Test requirement 3: Zeitanteil verschiedener Strukturen wird berechnet"""
    print("\n📊 Testing Structure Time Analysis...")
    
    # Create analyzer with trajectory
    sim = MockSimulation(n_residues=25, n_frames=40)
    analyzer = SecondaryStructureAnalyzer()
    
    # Analyze trajectory
    results = analyzer.analyze_trajectory(sim, time_step=1)
    
    # Verify time evolution data
    assert 'time_evolution' in results, "Time evolution data should be available"
    time_evolution = results['time_evolution']
    
    assert 'times' in time_evolution, "Time points should be recorded"
    assert 'percentages' in time_evolution, "Structure percentages should be calculated"
    
    # Check that percentages are calculated for each structure type
    for ss_type in SS_TYPES.keys():
        assert ss_type in time_evolution['percentages'], f"Missing percentages for {ss_type}"
        percentages = time_evolution['percentages'][ss_type]
        assert len(percentages) == len(time_evolution['times']), "Percentage count mismatch"
        
        # Verify percentage validity (0-100%)
        for pct in percentages:
            assert 0 <= pct <= 100, f"Invalid percentage: {pct}"
    
    # Check statistics
    stats = analyzer.get_statistics()
    assert 'trajectory_stats' in stats, "Trajectory statistics should be available"
    
    traj_stats = stats['trajectory_stats']
    assert 'avg_percentages' in traj_stats, "Average percentages should be calculated"
    assert 'std_percentages' in traj_stats, "Standard deviations should be calculated"
    assert 'stability_scores' in traj_stats, "Stability scores should be calculated"
    
    print(f"✅ Structure time analysis working correctly")
    print(f"   - Analyzed {len(time_evolution['times'])} time points")
    print(f"   - Average structure content:")
    for ss_type, avg_pct in traj_stats['avg_percentages'].items():
        std_pct = traj_stats['std_percentages'][ss_type]
        if avg_pct > 0.1:  # Only show significant structures
            print(f"     {SS_TYPES[ss_type]['name']}: {avg_pct:.1f}% ± {std_pct:.1f}%")
    
    return True


def test_timeline_export():
    """Test requirement 4: Export der Sekundärstruktur-Timeline möglich"""
    print("\n💾 Testing Timeline Export Functionality...")
    
    # Create analyzer with trajectory
    sim = MockSimulation(n_residues=15, n_frames=25)
    analyzer = SecondaryStructureAnalyzer()
    
    # Analyze trajectory
    results = analyzer.analyze_trajectory(sim, time_step=2)
    
    # Test CSV export
    csv_file = '/home/emilio/Documents/ai/md/test_task_3_4_timeline.csv'
    analyzer.export_timeline_data(csv_file, format='csv')
    
    # Verify CSV file was created and has content
    assert os.path.exists(csv_file), "CSV file should be created"
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 1, "CSV should have header and data"
        assert 'Time,Frame,Residue,ResName,SecondaryStructure' in lines[0], "CSV header incorrect"
    
    # Test JSON export
    json_file = '/home/emilio/Documents/ai/md/test_task_3_4_timeline.json'
    analyzer.export_timeline_data(json_file, format='json')
    
    # Verify JSON file was created
    assert os.path.exists(json_file), "JSON file should be created"
    
    # Load and verify JSON content
    import json
    with open(json_file, 'r') as f:
        data = json.load(f)
        
        assert 'metadata' in data, "JSON should contain metadata"
        assert 'trajectory_data' in data, "JSON should contain trajectory data"
        assert 'statistics' in data, "JSON should contain statistics"
        assert 'time_evolution' in data, "JSON should contain time evolution"
        
        # Verify metadata
        metadata = data['metadata']
        assert 'n_frames' in metadata, "Metadata should include frame count"
        assert 'n_residues' in metadata, "Metadata should include residue count"
        assert 'ss_types' in metadata, "Metadata should include structure types"
    
    print(f"✅ Timeline export working correctly")
    print(f"   - CSV export: {csv_file} ({os.path.getsize(csv_file)} bytes)")
    print(f"   - JSON export: {json_file} ({os.path.getsize(json_file)} bytes)")
    
    return True


def comprehensive_analysis_demo():
    """Demonstrate comprehensive secondary structure analysis capabilities."""
    print("\n🔬 Running Comprehensive Analysis Demo...")
    
    # Create larger simulation for demo
    sim = MockSimulation(n_residues=30, n_frames=50)
    analyzer = SecondaryStructureAnalyzer()
    
    print(f"   - Created simulation: {sim.n_residues} residues, {sim.n_frames} frames")
    
    # Full trajectory analysis
    results = analyzer.analyze_trajectory(sim, time_step=1)
    
    # Get comprehensive statistics
    stats = analyzer.get_statistics()
    
    print(f"   - Analysis completed successfully")
    print(f"   - Trajectory statistics:")
    traj_stats = stats['trajectory_stats']
    for ss_type in ['H', 'E', 'T', 'C']:  # Major structure types
        if ss_type in traj_stats['avg_percentages']:
            avg = traj_stats['avg_percentages'][ss_type]
            std = traj_stats['std_percentages'][ss_type]
            stability = traj_stats['stability_scores'][ss_type]
            print(f"     {SS_TYPES[ss_type]['name']}: {avg:.1f}±{std:.1f}% (stability: {stability:.3f})")
    
    # Create all visualizations
    fig1 = analyzer.plot_time_evolution(
        save_path='/home/emilio/Documents/ai/md/demo_ss_evolution.png'
    )
    fig2 = analyzer.plot_residue_timeline(
        save_path='/home/emilio/Documents/ai/md/demo_ss_timeline.png'
    )
    fig3 = analyzer.plot_structure_distribution(
        save_path='/home/emilio/Documents/ai/md/demo_ss_distribution.png'
    )
    
    plt.close('all')
    
    # Export comprehensive data
    analyzer.export_timeline_data(
        '/home/emilio/Documents/ai/md/demo_ss_data.csv', format='csv'
    )
    analyzer.export_timeline_data(
        '/home/emilio/Documents/ai/md/demo_ss_data.json', format='json'
    )
    
    print(f"   - Generated comprehensive analysis outputs")
    return True


def main():
    """Main verification function."""
    print("🧬 TASK 3.4 VERIFICATION: Secondary Structure Tracking")
    print("="*60)
    
    # Test all requirements
    tests = [
        ("DSSP-ähnlicher Algorithmus implementiert", test_dssp_algorithm),
        ("Sekundärstrukturänderungen werden farbkodiert visualisiert", test_color_coded_visualization),
        ("Zeitanteil verschiedener Strukturen wird berechnet", test_structure_time_analysis),
        ("Export der Sekundärstruktur-Timeline möglich", test_timeline_export)
    ]
    
    results = []
    for requirement, test_func in tests:
        try:
            success = test_func()
            results.append((requirement, success))
            print(f"✅ PASSED: {requirement}")
        except Exception as e:
            results.append((requirement, False))
            print(f"❌ FAILED: {requirement}")
            print(f"   Error: {str(e)}")
    
    # Run comprehensive demo
    try:
        comprehensive_analysis_demo()
        print(f"✅ PASSED: Comprehensive analysis demo")
    except Exception as e:
        print(f"❌ FAILED: Comprehensive analysis demo - {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    print("📋 TASK 3.4 VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for requirement, success in results:
        status = "✅ COMPLETE" if success else "❌ INCOMPLETE"
        print(f"{status}: {requirement}")
    
    print(f"\n🏆 OVERALL RESULT: {passed}/{total} requirements verified")
    
    if passed == total:
        print("🎉 Task 3.4 (Secondary Structure Tracking) is FULLY COMPLETE!")
        print("\n📁 Generated Files:")
        generated_files = [
            'test_task_3_4_time_evolution.png',
            'test_task_3_4_residue_timeline.png', 
            'test_task_3_4_distribution.png',
            'test_task_3_4_timeline.csv',
            'test_task_3_4_timeline.json',
            'demo_ss_evolution.png',
            'demo_ss_timeline.png',
            'demo_ss_distribution.png',
            'demo_ss_data.csv',
            'demo_ss_data.json'
        ]
        
        for filename in generated_files:
            filepath = f'/home/emilio/Documents/ai/md/{filename}'
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"   📄 {filename} ({size/1024:.1f} KB)")
        
        return True
    else:
        print("⚠️  Some requirements are not yet fully implemented")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
