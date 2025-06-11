#!/usr/bin/env python3
"""
Demo script for Task 2.2 - Trajectory Animation

This script demonstrates the trajectory animation capabilities including:
- Interactive Play/Pause/Step controls
- Adjustable animation speed
- Export to MP4/GIF formats
- Different display modes
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging

# Add the proteinMD package to path
sys.path.append(str(Path(__file__).parent))

from proteinMD.visualization.protein_3d import animate_trajectory, TrajectoryAnimator
from proteinMD.structure.protein import Protein, Atom, Residue
from proteinMD.structure.pdb_parser import PDBParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_trajectory(n_frames=50, n_atoms=20):
    """
    Generate a sample protein trajectory for demonstration.
    
    Parameters
    ----------
    n_frames : int
        Number of frames in trajectory
    n_atoms : int
        Number of atoms
        
    Returns
    -------
    dict
        Trajectory data with positions, elements, and bonds
    """
    logger.info(f"Generating sample trajectory with {n_frames} frames and {n_atoms} atoms")
    
    # Generate initial positions in a helical pattern
    t = np.linspace(0, 4*np.pi, n_atoms)
    radius = 3.0
    
    initial_positions = np.column_stack([
        radius * np.cos(t),
        radius * np.sin(t),
        t * 0.5  # Height increases along helix
    ])
    
    # Generate trajectory with breathing motion and rotation
    positions = []
    for frame in range(n_frames):
        time_factor = frame / n_frames
        
        # Breathing motion (radius oscillation)
        breath_factor = 1.0 + 0.3 * np.sin(2 * np.pi * time_factor * 3)
        
        # Rotation
        rotation_angle = 2 * np.pi * time_factor
        
        # Apply transformations
        frame_positions = initial_positions.copy()
        frame_positions[:, :2] *= breath_factor  # Breathing
        
        # Rotation around z-axis
        cos_rot = np.cos(rotation_angle)
        sin_rot = np.sin(rotation_angle)
        
        x_new = frame_positions[:, 0] * cos_rot - frame_positions[:, 1] * sin_rot
        y_new = frame_positions[:, 0] * sin_rot + frame_positions[:, 1] * cos_rot
        
        frame_positions[:, 0] = x_new
        frame_positions[:, 1] = y_new
        
        # Add some random noise for realism
        noise = np.random.normal(0, 0.1, frame_positions.shape)
        frame_positions += noise
        
        positions.append(frame_positions)
    
    # Generate elements (mix of common protein atoms)
    elements = ['C', 'N', 'O', 'S'] * (n_atoms // 4)
    elements.extend(['C'] * (n_atoms - len(elements)))  # Fill remaining with carbon
    
    # Generate bonds (connect sequential atoms)
    bonds = [(i, i+1) for i in range(n_atoms-1)]
    # Add some cross-links for helical structure
    bonds.extend([(i, i+4) for i in range(n_atoms-4)])
    
    return {
        'positions': positions,
        'elements': elements,
        'bonds': bonds
    }

def demo_basic_animation():
    """Demonstrate basic trajectory animation."""
    logger.info("=== Demo 1: Basic Trajectory Animation ===")
    
    # Generate sample data
    trajectory_data = generate_sample_trajectory(n_frames=30, n_atoms=15)
    
    # Create animation without controls for simplicity
    animator = animate_trajectory(
        positions=trajectory_data['positions'],
        elements=trajectory_data['elements'],
        bonds=trajectory_data['bonds'],
        interval=200,  # Slower for demo
        display_mode='ball_stick',
        controls=False,
        show=False  # Don't show yet
    )
    
    # Save as GIF
    output_file = "demo_basic_animation.gif"
    try:
        animator.save_animation(output_file, fps=5)
        logger.info(f"Saved basic animation as {output_file}")
    except Exception as e:
        logger.warning(f"Could not save GIF: {e}")
    
    plt.close(animator.fig)
    return animator

def demo_interactive_controls():
    """Demonstrate interactive controls."""
    logger.info("=== Demo 2: Interactive Controls ===")
    
    # Generate sample data
    trajectory_data = generate_sample_trajectory(n_frames=60, n_atoms=25)
    
    # Create animation with full controls
    animator = TrajectoryAnimator(
        positions=trajectory_data['positions'],
        elements=trajectory_data['elements'],
        bonds=trajectory_data['bonds'],
        interval=100,
        display_mode='ball_stick',
        controls=True
    )
    
    animator.setup_animation()
    
    # Demonstrate programmatic control
    logger.info("Demonstrating programmatic controls:")
    logger.info("- Setting speed to 2x")
    animator.set_speed(2.0)
    
    logger.info("- Going to frame 20")
    animator.goto_frame(20)
    
    logger.info("- Animation ready with Play/Pause/Step controls")
    logger.info("  Use Play button to start/stop animation")
    logger.info("  Use Step button to advance one frame")
    logger.info("  Use Speed slider to adjust playback speed")
    logger.info("  Use Frame slider to jump to specific frame")
    
    # Save as MP4 (if possible)
    output_file = "demo_interactive_controls.mp4"
    try:
        animator.save_animation(output_file, fps=10)
        logger.info(f"Saved interactive demo as {output_file}")
    except Exception as e:
        logger.warning(f"Could not save MP4: {e}")
    
    return animator

def demo_different_modes():
    """Demonstrate different visualization modes."""
    logger.info("=== Demo 3: Different Display Modes ===")
    
    # Generate sample data
    trajectory_data = generate_sample_trajectory(n_frames=40, n_atoms=30)
    
    modes = ['ball_stick', 'cartoon']  # Surface mode simplified for trajectory
    
    animators = []
    for mode in modes:
        logger.info(f"Creating animation in {mode} mode")
        
        animator = TrajectoryAnimator(
            positions=trajectory_data['positions'],
            elements=trajectory_data['elements'],
            bonds=trajectory_data['bonds'],
            interval=150,
            display_mode=mode,
            controls=False
        )
        
        animator.setup_animation()
        
        # Save each mode
        output_file = f"demo_{mode}_mode.gif"
        try:
            animator.save_animation(output_file, fps=6)
            logger.info(f"Saved {mode} mode animation as {output_file}")
        except Exception as e:
            logger.warning(f"Could not save {mode} animation: {e}")
        
        animators.append(animator)
        plt.close(animator.fig)
    
    return animators

def demo_real_protein_trajectory():
    """Demonstrate with real protein structure (if available)."""
    logger.info("=== Demo 4: Real Protein Trajectory ===")
    
    # Try to load a real protein structure
    pdb_file = Path("1ubq.pdb")
    
    if pdb_file.exists():
        logger.info(f"Loading real protein structure from {pdb_file}")
        
        try:
            parser = PDBParser()
            protein = parser.parse(str(pdb_file))
            
            # Extract atom positions
            atoms = list(protein.atoms.values())
            real_positions = np.array([[atom.x, atom.y, atom.z] for atom in atoms])
            elements = [atom.element for atom in atoms]
            
            # Generate a simple trajectory by adding motion to real structure
            n_frames = 25
            trajectory_positions = []
            
            for frame in range(n_frames):
                # Add gentle oscillation
                time_factor = frame / n_frames
                displacement = np.sin(2 * np.pi * time_factor) * 0.5
                
                frame_pos = real_positions.copy()
                frame_pos[:, 2] += displacement  # Vertical oscillation
                
                # Add some random thermal motion
                thermal_motion = np.random.normal(0, 0.1, frame_pos.shape)
                frame_pos += thermal_motion
                
                trajectory_positions.append(frame_pos)
            
            # Create animation
            animator = TrajectoryAnimator(
                positions=trajectory_positions,
                elements=elements,
                bonds=[],  # No bonds for simplicity with real protein
                interval=200,
                display_mode='ball_stick',
                controls=True
            )
            
            animator.setup_animation()
            
            # Save animation
            output_file = "demo_real_protein_trajectory.gif"
            try:
                animator.save_animation(output_file, fps=4)
                logger.info(f"Saved real protein trajectory as {output_file}")
            except Exception as e:
                logger.warning(f"Could not save real protein animation: {e}")
            
            logger.info(f"Real protein trajectory: {len(atoms)} atoms, {n_frames} frames")
            return animator
            
        except Exception as e:
            logger.warning(f"Could not process real protein file: {e}")
            
    else:
        logger.info("No real protein file found, generating synthetic protein-like trajectory")
        
        # Generate a more protein-like synthetic trajectory
        trajectory_data = generate_protein_like_trajectory()
        
        animator = TrajectoryAnimator(
            positions=trajectory_data['positions'],
            elements=trajectory_data['elements'],
            bonds=trajectory_data['bonds'],
            interval=150,
            display_mode='ball_stick',
            controls=True
        )
        
        animator.setup_animation()
        
        output_file = "demo_protein_like_trajectory.gif"
        try:
            animator.save_animation(output_file, fps=5)
            logger.info(f"Saved protein-like trajectory as {output_file}")
        except Exception as e:
            logger.warning(f"Could not save protein-like animation: {e}")
        
        return animator

def generate_protein_like_trajectory(n_frames=35, n_residues=8):
    """Generate a more realistic protein-like trajectory."""
    n_atoms_per_residue = 10
    n_atoms = n_residues * n_atoms_per_residue
    
    # Create a more structured initial configuration
    positions_init = []
    elements = []
    bonds = []
    
    for res_idx in range(n_residues):
        # Position residues along a backbone
        backbone_x = res_idx * 3.8  # Typical residue spacing
        backbone_y = np.sin(res_idx * 0.5) * 2.0  # Slight curve
        backbone_z = 0.0
        
        # Add atoms for this residue
        for atom_idx in range(n_atoms_per_residue):
            # Spread atoms around residue center
            angle = atom_idx * 2 * np.pi / n_atoms_per_residue
            radius = 1.5 + np.random.uniform(-0.3, 0.3)
            
            x = backbone_x + radius * np.cos(angle)
            y = backbone_y + radius * np.sin(angle)
            z = backbone_z + np.random.uniform(-0.5, 0.5)
            
            positions_init.append([x, y, z])
            
            # Assign elements (simplified protein composition)
            if atom_idx == 0:
                elements.append('N')  # Backbone nitrogen
            elif atom_idx == 1:
                elements.append('C')  # Alpha carbon
            elif atom_idx == 2:
                elements.append('C')  # Carbonyl carbon
            elif atom_idx == 3:
                elements.append('O')  # Carbonyl oxygen
            else:
                elements.append(['C', 'N', 'O', 'S'][atom_idx % 4])  # Side chain
            
            # Add bonds within residue
            global_atom_idx = res_idx * n_atoms_per_residue + atom_idx
            if atom_idx > 0:
                bonds.append((global_atom_idx - 1, global_atom_idx))
        
        # Add bonds between residues (peptide bonds)
        if res_idx > 0:
            prev_c = (res_idx - 1) * n_atoms_per_residue + 2  # Previous carbonyl C
            curr_n = res_idx * n_atoms_per_residue + 0        # Current N
            bonds.append((prev_c, curr_n))
    
    positions_init = np.array(positions_init)
    
    # Generate trajectory with protein-like motions
    trajectory_positions = []
    
    for frame in range(n_frames):
        time_factor = frame / n_frames
        
        # Overall protein motion (tumbling)
        tumble_angle = 2 * np.pi * time_factor * 0.5
        cos_t, sin_t = np.cos(tumble_angle), np.sin(tumble_angle)
        
        frame_pos = positions_init.copy()
        
        # Apply rotation
        x_rot = frame_pos[:, 0] * cos_t - frame_pos[:, 1] * sin_t
        y_rot = frame_pos[:, 0] * sin_t + frame_pos[:, 1] * cos_t
        frame_pos[:, 0] = x_rot
        frame_pos[:, 1] = y_rot
        
        # Add breathing motion (domain movement)
        breathing = 1.0 + 0.1 * np.sin(2 * np.pi * time_factor * 2)
        frame_pos *= breathing
        
        # Add thermal fluctuations
        thermal = np.random.normal(0, 0.08, frame_pos.shape)
        frame_pos += thermal
        
        trajectory_positions.append(frame_pos)
    
    return {
        'positions': trajectory_positions,
        'elements': elements,
        'bonds': bonds
    }

def demo_export_formats():
    """Demonstrate different export formats."""
    logger.info("=== Demo 5: Export Formats ===")
    
    # Generate compact data for faster export
    trajectory_data = generate_sample_trajectory(n_frames=20, n_atoms=12)
    
    animator = TrajectoryAnimator(
        positions=trajectory_data['positions'],
        elements=trajectory_data['elements'],
        bonds=trajectory_data['bonds'],
        interval=200,
        display_mode='ball_stick',
        controls=False
    )
    
    animator.setup_animation()
    
    # Test different export formats
    formats = [
        ('demo_export.gif', {'fps': 8, 'dpi': 80}),
        ('demo_export.mp4', {'fps': 10, 'dpi': 100}),
    ]
    
    for filename, kwargs in formats:
        try:
            logger.info(f"Exporting {filename}...")
            animator.save_animation(filename, **kwargs)
            logger.info(f"Successfully exported {filename}")
        except Exception as e:
            logger.warning(f"Could not export {filename}: {e}")
    
    plt.close(animator.fig)
    return animator

def main():
    """Run all trajectory animation demos."""
    logger.info("Starting Task 2.2 - Trajectory Animation Demo")
    logger.info("=" * 50)
    
    # List of demo functions
    demos = [
        demo_basic_animation,
        demo_interactive_controls,
        demo_different_modes,
        demo_real_protein_trajectory,
        demo_export_formats,
    ]
    
    results = []
    
    for demo_func in demos:
        try:
            result = demo_func()
            results.append(result)
            logger.info("Demo completed successfully")
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info("-" * 30)
    
    # Summary
    logger.info("=== Demo Summary ===")
    logger.info("Task 2.2 - Trajectory Animation features demonstrated:")
    logger.info("✅ Basic trajectory animation")
    logger.info("✅ Interactive Play/Pause/Step controls")
    logger.info("✅ Adjustable animation speed")
    logger.info("✅ Frame navigation")
    logger.info("✅ Multiple display modes")
    logger.info("✅ Export to GIF and MP4 formats")
    logger.info("✅ Real protein trajectory support")
    
    logger.info("\nGenerated animation files:")
    output_files = [
        "demo_basic_animation.gif",
        "demo_interactive_controls.mp4",
        "demo_ball_stick_mode.gif",
        "demo_cartoon_mode.gif",
        "demo_real_protein_trajectory.gif",
        "demo_protein_like_trajectory.gif",
        "demo_export.gif",
        "demo_export.mp4"
    ]
    
    for filename in output_files:
        if Path(filename).exists():
            logger.info(f"  ✅ {filename}")
        else:
            logger.info(f"  ❌ {filename} (not created)")
    
    logger.info("\nTask 2.2 - Trajectory Animation implementation complete!")
    
    return results

if __name__ == "__main__":
    main()
