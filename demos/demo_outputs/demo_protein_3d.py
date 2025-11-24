#!/usr/bin/env python3
"""
Demo script for 3D Protein Visualization - Task 2.1

This script demonstrates the comprehensive 3D protein visualization capabilities
including Ball-and-Stick, Cartoon, and Surface modes with interactive controls
and export functionality.

Usage:
    python demo_protein_3d.py [pdb_file]
"""

import sys
import numpy as np
from pathlib import Path

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

from proteinMD.structure.protein import Protein, Atom, Residue
from proteinMD.visualization.protein_3d import (
    Protein3DVisualizer, 
    InteractiveProteinViewer,
    quick_visualize,
    create_comparison_view
)

def create_sample_protein():
    """
    Create a sample protein structure for demonstration.
    
    This creates a small helical peptide with realistic atom positions
    and bonding patterns.
    """
    protein = Protein("Demo_Helix")
    
    # Create a simple alpha helix with 10 residues
    # Approximate coordinates for a right-handed alpha helix
    
    residues = []
    atoms = []
    atom_id = 1
    
    # Helix parameters
    radius = 2.3  # Angstroms
    rise_per_residue = 1.5  # Angstroms
    turn_per_residue = 100  # degrees
    
    for i in range(10):  # 10 residues
        # Calculate backbone positions
        angle = np.radians(i * turn_per_residue)
        z = i * rise_per_residue
        
        # Create residue
        residue = Residue(
            residue_id=i+1,
            residue_name='ALA',  # Alanine
            chain_id='A'
        )
        
        # N atom
        n_pos = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            z
        ])
        n_atom = Atom(
            atom_id=atom_id,
            atom_name='N',
            element='N',
            mass=14.007,
            charge=-0.4,
            position=n_pos,
            residue_id=i+1,
            chain_id='A'
        )
        atoms.append(n_atom)
        atom_id += 1
        
        # CA atom (alpha carbon)
        ca_angle = angle + np.radians(20)
        ca_pos = np.array([
            radius * np.cos(ca_angle),
            radius * np.sin(ca_angle),
            z + 0.5
        ])
        ca_atom = Atom(
            atom_id=atom_id,
            atom_name='CA',
            element='C',
            mass=12.011,
            charge=0.1,
            position=ca_pos,
            residue_id=i+1,
            chain_id='A'
        )
        atoms.append(ca_atom)
        atom_id += 1
        
        # C atom (carbonyl carbon)
        c_angle = angle + np.radians(40)
        c_pos = np.array([
            radius * np.cos(c_angle),
            radius * np.sin(c_angle),
            z + 1.0
        ])
        c_atom = Atom(
            atom_id=atom_id,
            atom_name='C',
            element='C',
            mass=12.011,
            charge=0.6,
            position=c_pos,
            residue_id=i+1,
            chain_id='A'
        )
        atoms.append(c_atom)
        atom_id += 1
        
        # O atom (carbonyl oxygen)
        o_pos = c_pos + np.array([0.5, 0.5, 0.2])
        o_atom = Atom(
            atom_id=atom_id,
            atom_name='O',
            element='O',
            mass=15.999,
            charge=-0.6,
            position=o_pos,
            residue_id=i+1,
            chain_id='A'
        )
        atoms.append(o_atom)
        atom_id += 1
        
        # CB atom (beta carbon for alanine)
        cb_pos = ca_pos + np.array([1.0, 0.0, 0.0])
        cb_atom = Atom(
            atom_id=atom_id,
            atom_name='CB',
            element='C',
            mass=12.011,
            charge=-0.1,
            position=cb_pos,
            residue_id=i+1,
            chain_id='A'
        )
        atoms.append(cb_atom)
        atom_id += 1
        
        # Add some hydrogen atoms
        for j, h_name in enumerate(['H1', 'H2', 'H3']):
            h_pos = cb_pos + np.array([
                0.5 * np.cos(j * np.pi/3),
                0.5 * np.sin(j * np.pi/3),
                0.3
            ])
            h_atom = Atom(
                atom_id=atom_id,
                atom_name=h_name,
                element='H',
                mass=1.008,
                charge=0.1,
                position=h_pos,
                residue_id=i+1,
                chain_id='A'
            )
            atoms.append(h_atom)
            atom_id += 1
        
        # Add bonds within residue
        n_atom.add_bond(ca_atom.atom_id)
        ca_atom.add_bond(n_atom.atom_id)
        ca_atom.add_bond(c_atom.atom_id)
        c_atom.add_bond(ca_atom.atom_id)
        c_atom.add_bond(o_atom.atom_id)
        o_atom.add_bond(c_atom.atom_id)
        ca_atom.add_bond(cb_atom.atom_id)
        cb_atom.add_bond(ca_atom.atom_id)
        
        # Bond CB to hydrogens
        for h_atom in atoms[-3:]:  # Last 3 atoms are hydrogens
            cb_atom.add_bond(h_atom.atom_id)
            h_atom.add_bond(cb_atom.atom_id)
        
        residues.append(residue)
    
    # Add peptide bonds between residues
    for i in range(len(residues) - 1):
        # Find C atom of residue i and N atom of residue i+1
        c_atoms = [a for a in atoms if a.residue_id == i+1 and a.atom_name == 'C']
        n_atoms = [a for a in atoms if a.residue_id == i+2 and a.atom_name == 'N']
        
        if c_atoms and n_atoms:
            c_atoms[0].add_bond(n_atoms[0].atom_id)
            n_atoms[0].add_bond(c_atoms[0].atom_id)
    
    # Add atoms and residues to protein
    for atom in atoms:
        protein.add_atom(atom)
    
    for residue in residues:
        protein.add_residue(residue)
    
    return protein


def demo_basic_visualization():
    """Demonstrate basic 3D visualization modes."""
    print("=== Demo: Basic 3D Protein Visualization ===")
    
    # Create sample protein
    protein = create_sample_protein()
    print(f"Created sample protein with {len(protein.atoms)} atoms")
    
    # 1. Ball-and-Stick mode
    print("\n1. Ball-and-Stick mode")
    visualizer = Protein3DVisualizer(protein)
    fig1 = visualizer.ball_and_stick(show_hydrogens=True)
    visualizer.export_png("demo_ball_stick.png")
    print("   - Ball-and-stick visualization created")
    print("   - Exported as demo_ball_stick.png")
    
    # 2. Cartoon mode
    print("\n2. Cartoon mode")
    fig2 = visualizer.cartoon()
    visualizer.export_png("demo_cartoon.png")
    print("   - Cartoon visualization created")
    print("   - Exported as demo_cartoon.png")
    
    # 3. Surface mode
    print("\n3. Surface mode")
    fig3 = visualizer.surface()
    visualizer.export_png("demo_surface.png")
    visualizer.export_svg("demo_surface.svg")
    print("   - Surface visualization created")
    print("   - Exported as demo_surface.png and demo_surface.svg")
    
    return visualizer


def demo_interactive_controls():
    """Demonstrate interactive rotation and zoom controls."""
    print("\n=== Demo: Interactive Controls ===")
    
    protein = create_sample_protein()
    visualizer = Protein3DVisualizer(protein)
    
    # Create initial visualization
    visualizer.ball_and_stick()
    
    # Demonstrate different viewing angles
    views = [
        ("Front view", 0, 0),
        ("Side view", 0, 90),
        ("Top view", 90, 0),
        ("Angled view", 30, 45)
    ]
    
    for view_name, elev, azim in views:
        print(f"   - Setting {view_name}: elevation={elev}°, azimuth={azim}°")
        visualizer.set_view(elevation=elev, azimuth=azim)
        visualizer.export_png(f"demo_{view_name.lower().replace(' ', '_')}.png")
    
    # Demonstrate zoom
    print("   - Zooming in (2x)")
    visualizer.zoom(2.0)
    visualizer.export_png("demo_zoomed_in.png")
    
    print("   - Zooming out (0.5x)")
    visualizer.zoom(0.5)
    visualizer.export_png("demo_zoomed_out.png")
    
    return visualizer


def demo_trajectory_animation():
    """Demonstrate trajectory animation capabilities."""
    print("\n=== Demo: Trajectory Animation ===")
    
    protein = create_sample_protein()
    
    # Create simulated trajectory data (breathing motion)
    n_frames = 50
    trajectory_data = []
    
    base_positions = np.array([atom.position for atom in protein.atoms.values()])
    
    for frame in range(n_frames):
        # Create breathing motion
        scale_factor = 1.0 + 0.2 * np.sin(2 * np.pi * frame / n_frames)
        
        # Calculate center of mass
        com = np.mean(base_positions, axis=0)
        
        # Scale positions around center of mass
        scaled_positions = com + scale_factor * (base_positions - com)
        
        # Add some random thermal motion
        thermal_motion = np.random.normal(0, 0.1, scaled_positions.shape)
        frame_positions = scaled_positions + thermal_motion
        
        trajectory_data.append(frame_positions)
    
    # Create animation
    visualizer = Protein3DVisualizer(protein)
    animation = visualizer.animate_trajectory(
        trajectory_data, 
        interval=100,  # 100ms between frames
        save_as="demo_trajectory.gif"
    )
    
    print(f"   - Created animation with {n_frames} frames")
    print("   - Saved as demo_trajectory.gif")
    
    return visualizer, animation


def demo_comparison_view():
    """Demonstrate comparison view with multiple modes."""
    print("\n=== Demo: Comparison View ===")
    
    protein = create_sample_protein()
    
    # Create comparison figure
    fig = create_comparison_view(
        protein, 
        modes=['ball_stick', 'cartoon', 'surface']
    )
    
    fig.savefig("demo_comparison.png", dpi=300, bbox_inches='tight')
    print("   - Created comparison view with all three modes")
    print("   - Saved as demo_comparison.png")
    
    return fig


def demo_quick_functions():
    """Demonstrate quick visualization functions."""
    print("\n=== Demo: Quick Visualization Functions ===")
    
    protein = create_sample_protein()
    
    # Quick ball-and-stick
    print("   - Quick ball-and-stick visualization")
    vis1 = quick_visualize(protein, mode='ball_stick', show=False)
    vis1.export_png("demo_quick_ball_stick.png")
    
    # Quick cartoon
    print("   - Quick cartoon visualization")
    vis2 = quick_visualize(protein, mode='cartoon', show=False)
    vis2.export_png("demo_quick_cartoon.png")
    
    # Quick surface
    print("   - Quick surface visualization")
    vis3 = quick_visualize(protein, mode='surface', show=False)
    vis3.export_png("demo_quick_surface.png")
    
    return vis1, vis2, vis3


def main():
    """Run the complete demonstration."""
    print("3D Protein Visualization Demo - Task 2.1")
    print("=" * 50)
    
    try:
        # Run all demonstrations
        vis1 = demo_basic_visualization()
        vis2 = demo_interactive_controls()
        vis3, anim = demo_trajectory_animation()
        fig = demo_comparison_view()
        quick_vis = demo_quick_functions()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nGenerated files:")
        
        generated_files = [
            "demo_ball_stick.png",
            "demo_cartoon.png", 
            "demo_surface.png",
            "demo_surface.svg",
            "demo_front_view.png",
            "demo_side_view.png",
            "demo_top_view.png",
            "demo_angled_view.png",
            "demo_zoomed_in.png",
            "demo_zoomed_out.png",
            "demo_trajectory.gif",
            "demo_comparison.png",
            "demo_quick_ball_stick.png",
            "demo_quick_cartoon.png",
            "demo_quick_surface.png"
        ]
        
        for i, filename in enumerate(generated_files, 1):
            print(f"{i:2d}. {filename}")
        
        print("\nTask 2.1 Requirements Status:")
        print("✓ Protein displayed as 3D model with atoms and bonds")
        print("✓ Multiple display modes (Ball-and-Stick, Cartoon, Surface)")
        print("✓ Interactive rotation and zoom functionality")
        print("✓ Export capabilities (PNG/SVG)")
        print("✓ Trajectory animation support")
        
        # Clean up
        vis1.close()
        vis2.close()
        vis3.close()
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
