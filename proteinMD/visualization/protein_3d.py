"""
3D Protein Visualization Module - Task 2.1

This module provides comprehensive 3D visualization capabilities for protein structures,
including multiple display modes, interactive controls, and export functionality.

Requirements fulfilled:
- Protein displayed as 3D model with atoms and bonds
- Multiple display modes (Ball-and-Stick, Cartoon, Surface)
- Interactive rotation and zoom functionality
- Export capabilities (PNG/SVG)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from matplotlib.patches import Circle
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import warnings

# Configure logging
logger = logging.getLogger(__name__)

# Color schemes for different visualization modes
ELEMENT_COLORS = {
    'H': '#FFFFFF',    # White
    'C': '#000000',    # Black
    'N': '#0000FF',    # Blue
    'O': '#FF0000',    # Red
    'S': '#FFFF00',    # Yellow
    'P': '#FFA500',    # Orange
    'CA': '#00FF00',   # Green (Calcium)
    'MG': '#228B22',   # Forest Green (Magnesium)
    'FE': '#8B4513',   # Saddle Brown (Iron)
    'ZN': '#A0A0A0',   # Gray (Zinc)
}

# Van der Waals radii for elements (in Angstroms)
VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
    'P': 1.80, 'CA': 2.31, 'MG': 1.73, 'FE': 2.00, 'ZN': 1.39,
}

# Secondary structure colors
SS_COLORS = {
    'helix': '#FF0000',      # Red for alpha helices
    'sheet': '#0000FF',      # Blue for beta sheets
    'loop': '#00FF00',       # Green for loops/coils
    'turn': '#FFFF00',       # Yellow for turns
}


class Protein3DVisualizer:
    """
    Advanced 3D protein visualization with multiple display modes.
    
    This class provides comprehensive 3D visualization capabilities for protein
    structures including Ball-and-Stick, Cartoon, and Surface representations.
    """
    
    def __init__(self, protein_structure, figsize=(12, 10)):
        """
        Initialize the 3D protein visualizer.
        
        Parameters
        ----------
        protein_structure : Protein
            Protein structure object from proteinMD.structure
        figsize : tuple, optional
            Figure size for the visualization
        """
        self.protein = protein_structure
        self.figsize = figsize
        
        # Visualization state
        self.fig = None
        self.ax = None
        self.display_mode = 'ball_stick'
        self.current_frame = 0
        
        # Atom and bond data
        self.atom_positions = None
        self.atom_colors = None
        self.atom_sizes = None
        self.bonds = None
        
        # Animation data
        self.trajectory_data = None
        self.animation = None
        
        # Interactive controls
        self.rotation_angle = 0
        self.elevation_angle = 20
        self.azimuth_angle = 45
        
        logger.info(f"Initialized 3D protein visualizer for {len(self.protein.atoms)} atoms")
        
    def _prepare_atom_data(self):
        """Prepare atom positions, colors, and sizes for visualization."""
        atoms = list(self.protein.atoms.values())
        n_atoms = len(atoms)
        
        # Extract positions
        self.atom_positions = np.array([atom.position for atom in atoms])
        
        # Assign colors based on element
        self.atom_colors = []
        self.atom_sizes = []
        
        for atom in atoms:
            # Get element color
            element = atom.element.upper()
            color = ELEMENT_COLORS.get(element, '#808080')  # Gray for unknown elements
            self.atom_colors.append(color)
            
            # Get atom size based on Van der Waals radius
            radius = VDW_RADII.get(element, 1.5)
            self.atom_sizes.append(radius * 50)  # Scale for visualization
            
        self.atom_colors = np.array(self.atom_colors)
        self.atom_sizes = np.array(self.atom_sizes)
        
        logger.debug(f"Prepared data for {n_atoms} atoms")
    
    def _prepare_bond_data(self):
        """Prepare bond data for visualization."""
        bonds = []
        atoms = list(self.protein.atoms.values())
        atom_id_to_index = {atom.atom_id: i for i, atom in enumerate(atoms)}
        
        for atom in atoms:
            for bonded_id in atom.bonded_atoms:
                if bonded_id in atom_id_to_index:
                    i = atom_id_to_index[atom.atom_id]
                    j = atom_id_to_index[bonded_id]
                    if i < j:  # Avoid duplicate bonds
                        bonds.append((i, j))
        
        self.bonds = bonds
        logger.debug(f"Prepared {len(bonds)} bonds")
    
    def _create_figure(self):
        """Create the matplotlib figure and 3D axis."""
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set up interactive navigation
        self.ax.mouse_init()
        
        # Set initial view
        self.ax.view_init(elev=self.elevation_angle, azim=self.azimuth_angle)
        
        # Set labels
        self.ax.set_xlabel('X (Å)')
        self.ax.set_ylabel('Y (Å)')
        self.ax.set_zlabel('Z (Å)')
        
        return self.fig, self.ax
    
    def ball_and_stick(self, show_hydrogens=True, bond_width=2.0):
        """
        Display protein in Ball-and-Stick mode.
        
        Parameters
        ----------
        show_hydrogens : bool, optional
            Whether to show hydrogen atoms
        bond_width : float, optional
            Width of bond lines
        """
        self.display_mode = 'ball_stick'
        
        # Prepare data
        self._prepare_atom_data()
        self._prepare_bond_data()
        
        # Create figure
        self._create_figure()
        
        # Filter atoms if hiding hydrogens
        if not show_hydrogens:
            atoms = list(self.protein.atoms.values())
            non_h_indices = [i for i, atom in enumerate(atoms) if atom.element != 'H']
            positions = self.atom_positions[non_h_indices]
            colors = self.atom_colors[non_h_indices]
            sizes = self.atom_sizes[non_h_indices]
        else:
            positions = self.atom_positions
            colors = self.atom_colors
            sizes = self.atom_sizes
        
        # Draw atoms as spheres
        scatter = self.ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c=colors, s=sizes, alpha=0.8, edgecolors='black', linewidth=0.5
        )
        
        # Draw bonds as lines
        for bond in self.bonds:
            i, j = bond
            if not show_hydrogens:
                # Skip bonds involving hydrogens
                atoms = list(self.protein.atoms.values())
                if atoms[i].element == 'H' or atoms[j].element == 'H':
                    continue
                # Map to non-hydrogen indices
                try:
                    i_mapped = non_h_indices.index(i)
                    j_mapped = non_h_indices.index(j)
                    pos1 = positions[i_mapped]
                    pos2 = positions[j_mapped]
                except ValueError:
                    continue
            else:
                pos1 = self.atom_positions[i]
                pos2 = self.atom_positions[j]
            
            self.ax.plot3D(
                [pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                'k-', linewidth=bond_width, alpha=0.6
            )
        
        # Set title
        self.ax.set_title(f'Ball-and-Stick: {self.protein.name}', fontsize=14, fontweight='bold')
        
        # Adjust axes
        self._adjust_axes()
        
        logger.info("Ball-and-stick visualization created")
        return self.fig
    
    def cartoon(self, color_by_structure=True):
        """
        Display protein in Cartoon mode.
        
        Parameters
        ----------
        color_by_structure : bool, optional
            Whether to color by secondary structure
        """
        self.display_mode = 'cartoon'
        
        # Prepare data
        self._prepare_atom_data()
        
        # Create figure
        self._create_figure()
        
        # Get backbone atoms (CA atoms for cartoon representation)
        ca_atoms = []
        atoms = list(self.protein.atoms.values())
        
        for atom in atoms:
            if atom.atom_name == 'CA':  # Alpha carbon
                ca_atoms.append(atom)
        
        if not ca_atoms:
            logger.warning("No CA atoms found for cartoon representation")
            # Fallback to all atoms with reduced size
            positions = self.atom_positions
            colors = ['#808080'] * len(positions)  # Gray color
        else:
            positions = np.array([atom.position for atom in ca_atoms])
            
            # Color by secondary structure if available
            if color_by_structure and hasattr(self.protein, 'secondary_structure'):
                colors = []
                for atom in ca_atoms:
                    residue = self.protein.get_residue_by_id(atom.residue_id)
                    if residue and hasattr(residue, 'secondary_structure'):
                        ss_type = getattr(residue, 'secondary_structure', 'loop')
                        colors.append(SS_COLORS.get(ss_type, SS_COLORS['loop']))
                    else:
                        colors.append(SS_COLORS['loop'])
            else:
                colors = ['#00AA00'] * len(positions)  # Default green
        
        # Draw cartoon as connected line with varying width
        if len(positions) > 1:
            # Create smooth curve through CA positions
            for i in range(len(positions) - 1):
                color = colors[i] if i < len(colors) else '#00AA00'
                self.ax.plot3D(
                    [positions[i, 0], positions[i+1, 0]],
                    [positions[i, 1], positions[i+1, 1]],
                    [positions[i, 2], positions[i+1, 2]],
                    color=color, linewidth=6, alpha=0.8
                )
        
        # Add CA atoms as small spheres
        scatter = self.ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c=colors, s=30, alpha=0.9, edgecolors='black', linewidth=0.3
        )
        
        # Set title
        self.ax.set_title(f'Cartoon: {self.protein.name}', fontsize=14, fontweight='bold')
        
        # Adjust axes
        self._adjust_axes()
        
        logger.info("Cartoon visualization created")
        return self.fig
    
    def surface(self, resolution=20, alpha=0.6):
        """
        Display protein in Surface mode.
        
        Parameters
        ----------
        resolution : int, optional
            Resolution of the surface mesh
        alpha : float, optional
            Transparency of the surface
        """
        self.display_mode = 'surface'
        
        # Prepare data
        self._prepare_atom_data()
        
        # Create figure
        self._create_figure()
        
        # Calculate molecular surface using alpha shapes approximation
        positions = self.atom_positions
        
        # Find the convex hull points for surface approximation
        from scipy.spatial import ConvexHull
        
        try:
            hull = ConvexHull(positions)
            
            # Create surface using triangulation
            for simplex in hull.simplices:
                # Get triangle vertices
                triangle = positions[simplex]
                
                # Draw triangle faces
                self.ax.plot_trisurf(
                    triangle[:, 0], triangle[:, 1], triangle[:, 2],
                    alpha=alpha, color='lightblue', edgecolor='none'
                )
                
        except Exception as e:
            logger.warning(f"Could not create surface: {e}")
            # Fallback: show atoms with large spheres
            scatter = self.ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c=self.atom_colors, s=self.atom_sizes*2, alpha=alpha,
                edgecolors='black', linewidth=0.1
            )
        
        # Set title
        self.ax.set_title(f'Surface: {self.protein.name}', fontsize=14, fontweight='bold')
        
        # Adjust axes
        self._adjust_axes()
        
        logger.info("Surface visualization created")
        return self.fig
    
    def _adjust_axes(self):
        """Adjust axes to fit the protein structure."""
        if self.atom_positions is not None:
            # Calculate bounds
            margin = 5.0  # Angstrom margin
            
            x_min, x_max = self.atom_positions[:, 0].min() - margin, self.atom_positions[:, 0].max() + margin
            y_min, y_max = self.atom_positions[:, 1].min() - margin, self.atom_positions[:, 1].max() + margin
            z_min, z_max = self.atom_positions[:, 2].min() - margin, self.atom_positions[:, 2].max() + margin
            
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            self.ax.set_zlim(z_min, z_max)
            
            # Make axes equal
            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
            mid_x = (x_max + x_min) * 0.5
            mid_y = (y_max + y_min) * 0.5
            mid_z = (z_max + z_min) * 0.5
            
            self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    def set_view(self, elevation=None, azimuth=None):
        """
        Set the viewing angle for interactive rotation.
        
        Parameters
        ----------
        elevation : float, optional
            Elevation angle in degrees
        azimuth : float, optional
            Azimuth angle in degrees
        """
        if elevation is not None:
            self.elevation_angle = elevation
        if azimuth is not None:
            self.azimuth_angle = azimuth
            
        if self.ax is not None:
            self.ax.view_init(elev=self.elevation_angle, azim=self.azimuth_angle)
            if hasattr(self.fig, 'canvas'):
                self.fig.canvas.draw()
    
    def zoom(self, factor):
        """
        Zoom in or out of the visualization.
        
        Parameters
        ----------
        factor : float
            Zoom factor (>1 zooms in, <1 zooms out)
        """
        if self.ax is not None:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            zlim = self.ax.get_zlim()
            
            # Calculate centers
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            z_center = (zlim[0] + zlim[1]) / 2
            
            # Calculate new ranges
            x_range = (xlim[1] - xlim[0]) / factor / 2
            y_range = (ylim[1] - ylim[0]) / factor / 2
            z_range = (zlim[1] - zlim[0]) / factor / 2
            
            # Set new limits
            self.ax.set_xlim(x_center - x_range, x_center + x_range)
            self.ax.set_ylim(y_center - y_range, y_center + y_range)
            self.ax.set_zlim(z_center - z_range, z_center + z_range)
            
            if hasattr(self.fig, 'canvas'):
                self.fig.canvas.draw()
    
    def export_png(self, filename, dpi=300):
        """
        Export the current visualization as PNG.
        
        Parameters
        ----------
        filename : str
            Output filename
        dpi : int, optional
            Resolution in dots per inch
        """
        if self.fig is not None:
            filepath = Path(filename)
            if not filepath.suffix:
                filepath = filepath.with_suffix('.png')
            
            self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
            logger.info(f"Exported PNG to {filepath}")
        else:
            logger.error("No figure to export")
    
    def export_svg(self, filename):
        """
        Export the current visualization as SVG.
        
        Parameters
        ----------
        filename : str
            Output filename
        """
        if self.fig is not None:
            filepath = Path(filename)
            if not filepath.suffix:
                filepath = filepath.with_suffix('.svg')
            
            self.fig.savefig(filepath, format='svg', bbox_inches='tight',
                           facecolor='white', edgecolor='none')
            logger.info(f"Exported SVG to {filepath}")
        else:
            logger.error("No figure to export")
    
    def animate_trajectory(self, trajectory_data, interval=100, save_as=None, 
                          display_mode='ball_stick', controls=True):
        """
        Create an animation of protein trajectory with interactive controls.
        
        Parameters
        ----------
        trajectory_data : list of np.ndarray or dict
            List of position arrays for each frame, or dict with 'positions', 
            'elements', and optional 'bonds' keys
        interval : int, optional
            Interval between frames in milliseconds (default: 100)
        save_as : str, optional
            Filename to save animation (supports .mp4, .gif, .avi)
        display_mode : str, optional
            Visualization mode ('ball_stick', 'cartoon', 'surface')
        controls : bool, optional
            Whether to add interactive controls (default: True)
            
        Returns
        -------
        TrajectoryAnimator
            Animation object with playback controls
        """
        if not trajectory_data:
            raise ValueError("Trajectory data cannot be empty")
            
        # Parse trajectory data
        if isinstance(trajectory_data, dict):
            positions = trajectory_data['positions']
            elements = trajectory_data.get('elements', ['C'] * positions[0].shape[0])
            bonds = trajectory_data.get('bonds', [])
        else:
            positions = trajectory_data
            elements = ['C'] * positions[0].shape[0]  # Default to carbon
            bonds = []
        
        # Create trajectory animator
        animator = TrajectoryAnimator(
            positions=positions,
            elements=elements,
            bonds=bonds,
            interval=interval,
            display_mode=display_mode,
            controls=controls
        )
        
        # Set up the animation
        animator.setup_animation()
        
        # Save if requested
        if save_as:
            animator.save_animation(save_as)
            
        logger.info(f"Created trajectory animation with {len(positions)} frames")
        return animator
    
    def show(self):
        """Display the current visualization."""
        if self.fig is not None:
            plt.show()
        else:
            logger.error("No visualization to show. Create one first.")
    
    def close(self):
        """Close the current visualization and clean up."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        
        # Clean up animation
        if self.animation is not None:
            self.animation = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.debug("Visualization closed and cleaned up")


class InteractiveProteinViewer:
    """
    Interactive protein viewer with GUI controls.
    
    This class provides an enhanced viewer with interactive controls for
    display modes, viewing angles, and other visualization parameters.
    """
    
    def __init__(self, protein_structure):
        """
        Initialize the interactive viewer.
        
        Parameters
        ----------
        protein_structure : Protein
            Protein structure to visualize
        """
        self.protein = protein_structure
        self.visualizer = Protein3DVisualizer(protein_structure)
        
        # GUI state
        self.current_mode = 'ball_stick'
        self.show_hydrogens = True
        
        logger.info("Interactive protein viewer initialized")
    
    def create_controls(self):
        """Create interactive control widgets."""
        try:
            from matplotlib.widgets import Button, RadioButtons, CheckButtons
            
            # Create control panel
            fig = self.visualizer._create_figure()
            
            # Add control axes
            ax_mode = plt.axes([0.02, 0.7, 0.15, 0.15])
            ax_view = plt.axes([0.02, 0.5, 0.15, 0.15])
            ax_export = plt.axes([0.02, 0.3, 0.15, 0.1])
            
            # Display mode controls
            radio_mode = RadioButtons(ax_mode, ('Ball-Stick', 'Cartoon', 'Surface'))
            radio_mode.on_clicked(self._change_mode)
            
            # View controls
            radio_view = RadioButtons(ax_view, ('Front', 'Side', 'Top', 'Custom'))
            radio_view.on_clicked(self._change_view)
            
            # Export button
            button_export = Button(ax_export, 'Export PNG')
            button_export.on_clicked(self._export_png)
            
            return fig
            
        except ImportError:
            logger.warning("Interactive widgets not available")
            return self.visualizer._create_figure()
    
    def _change_mode(self, mode):
        """Change display mode."""
        mode_map = {
            'Ball-Stick': 'ball_stick',
            'Cartoon': 'cartoon', 
            'Surface': 'surface'
        }
        
        self.current_mode = mode_map.get(mode, 'ball_stick')
        self._update_display()
    
    def _change_view(self, view):
        """Change viewing angle."""
        view_angles = {
            'Front': (0, 0),
            'Side': (0, 90),
            'Top': (90, 0),
            'Custom': (20, 45)
        }
        
        elev, azim = view_angles.get(view, (20, 45))
        self.visualizer.set_view(elevation=elev, azimuth=azim)
    
    def _export_png(self, event):
        """Export current view as PNG."""
        filename = f"{self.protein.name}_{self.current_mode}.png"
        self.visualizer.export_png(filename)
    
    def _update_display(self):
        """Update the display based on current settings."""
        if self.current_mode == 'ball_stick':
            self.visualizer.ball_and_stick(show_hydrogens=self.show_hydrogens)
        elif self.current_mode == 'cartoon':
            self.visualizer.cartoon()
        elif self.current_mode == 'surface':
            self.visualizer.surface()
    
    def launch(self):
        """Launch the interactive viewer."""
        fig = self.create_controls()
        self._update_display()
        self.visualizer.show()
        return fig


# Convenience functions for quick visualization
def quick_visualize(protein, mode='ball_stick', show=True, **kwargs):
    """
    Quick visualization function for protein structures.
    
    Parameters
    ----------
    protein : Protein
        Protein structure to visualize
    mode : str, optional
        Display mode ('ball_stick', 'cartoon', 'surface')
    show : bool, optional
        Whether to show the visualization immediately
    **kwargs
        Additional arguments for the visualization mode
    
    Returns
    -------
    Protein3DVisualizer
        The visualizer instance
    """
    visualizer = Protein3DVisualizer(protein)
    
    if mode == 'ball_stick':
        visualizer.ball_and_stick(**kwargs)
    elif mode == 'cartoon':
        visualizer.cartoon(**kwargs)
    elif mode == 'surface':
        visualizer.surface(**kwargs)
    else:
        logger.error(f"Unknown mode: {mode}")
        return None
    
    if show:
        visualizer.show()
    
    return visualizer


def create_comparison_view(protein, modes=['ball_stick', 'cartoon', 'surface']):
    """
    Create a comparison view showing multiple display modes.
    
    Parameters
    ----------
    protein : Protein
        Protein structure to visualize
    modes : list, optional
        List of display modes to show
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure with comparison views
    """
    n_modes = len(modes)
    fig = plt.figure(figsize=(6*n_modes, 6))
    
    for i, mode in enumerate(modes):
        ax = fig.add_subplot(1, n_modes, i+1, projection='3d')
        
        # Create individual visualizer for this subplot
        visualizer = Protein3DVisualizer(protein)
        visualizer.fig = fig
        visualizer.ax = ax
        
        if mode == 'ball_stick':
            visualizer.ball_and_stick()
        elif mode == 'cartoon':
            visualizer.cartoon()
        elif mode == 'surface':
            visualizer.surface()
    
    plt.tight_layout()
    return fig


class TrajectoryAnimator:
    """
    Advanced trajectory animation system with interactive controls.
    
    This class provides comprehensive trajectory animation capabilities including:
    - Play/Pause/Step controls
    - Adjustable animation speed
    - Multiple export formats (MP4, GIF, AVI)
    - Different display modes
    """
    
    def __init__(self, positions, elements=None, bonds=None, interval=100, 
                 display_mode='ball_stick', controls=True):
        """
        Initialize trajectory animator.
        
        Parameters
        ----------
        positions : list of np.ndarray
            List of position arrays for each frame [n_frames, n_atoms, 3]
        elements : list of str, optional
            Element names for each atom
        bonds : list of tuples, optional
            Bond connectivity as list of (atom1_idx, atom2_idx) tuples
        interval : int
            Animation interval in milliseconds
        display_mode : str
            Visualization mode ('ball_stick', 'cartoon', 'surface')
        controls : bool
            Whether to add interactive controls
        """
        self.positions = np.array(positions)
        self.n_frames, self.n_atoms = self.positions.shape[:2]
        self.elements = elements or ['C'] * self.n_atoms
        self.bonds = bonds or []
        self.interval = interval
        self.display_mode = display_mode
        self.controls = controls
        
        # Animation state
        self.current_frame = 0
        self.is_playing = False
        self.animation_speed = 1.0
        self._updating_slider = False  # Flag to prevent circular slider updates
        
        # Matplotlib objects
        self.fig = None
        self.ax = None
        self.animation = None
        self.scatter = None
        self.bond_lines = []
        
        # UI elements
        self.play_button = None
        self.step_button = None
        self.speed_slider = None
        self.frame_slider = None
        
    def setup_animation(self):
        """Set up the animation figure and controls."""
        # Create figure with space for controls
        if self.controls:
            self.fig = plt.figure(figsize=(12, 10))
            # Main plot takes most space
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.fig.subplots_adjust(bottom=0.2)
        else:
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set up initial visualization
        self._setup_initial_plot()
        
        # Add controls if requested
        if self.controls:
            self._setup_controls()
        
        # Create animation
        def animation_func(frame_num):
            """Animation function that respects manual frame changes."""
            # Only update if the animation is actually running
            if self.is_playing:
                return self._update_frame(frame_num)
            return []
            
        self.animation = FuncAnimation(
            self.fig, animation_func, frames=self.n_frames,
            interval=self.interval, blit=False, repeat=True
        )
        
        # Start paused
        self.animation.pause()
        
        logger.info(f"Set up trajectory animation with {self.n_frames} frames")
    
    def _setup_initial_plot(self):
        """Set up the initial 3D plot."""
        # Get colors and sizes for atoms
        colors = [ELEMENT_COLORS.get(elem, '#808080') for elem in self.elements]
        sizes = [VDW_RADII.get(elem, 1.5) * 100 for elem in self.elements]
        
        # Initial positions (frame 0)
        pos = self.positions[0]
        
        if self.display_mode == 'ball_stick':
            # Create scatter plot for atoms
            self.scatter = self.ax.scatter(
                pos[:, 0], pos[:, 1], pos[:, 2],
                c=colors, s=sizes, alpha=0.8, edgecolors='black'
            )
            
            # Draw bonds
            self._update_bonds(pos)
            
        elif self.display_mode == 'cartoon':
            # Simplified cartoon representation
            self.scatter = self.ax.plot(
                pos[:, 0], pos[:, 1], pos[:, 2],
                'o-', color='blue', alpha=0.7, markersize=5
            )[0]
            
        else:  # surface mode - simplified as wireframe
            self.scatter = self.ax.plot_wireframe(
                pos[:, 0].reshape(-1, 1), 
                pos[:, 1].reshape(-1, 1), 
                pos[:, 2].reshape(-1, 1),
                alpha=0.5
            )
        
        # Set labels and limits
        self.ax.set_xlabel('X (Å)')
        self.ax.set_ylabel('Y (Å)')
        self.ax.set_zlabel('Z (Å)')
        
        # Set axis limits based on trajectory extent
        all_pos = self.positions.reshape(-1, 3)
        margin = 2.0
        self.ax.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
        self.ax.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
        self.ax.set_zlim(all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)
        
        # Set title
        self.ax.set_title(f'Trajectory Animation - Frame 0/{self.n_frames-1}')
    
    def _update_bonds(self, positions):
        """Update bond lines for current positions."""
        # Clear existing bond lines
        for line in self.bond_lines:
            line.remove()
        self.bond_lines.clear()
        
        # Draw new bonds
        for atom1, atom2 in self.bonds:
            if atom1 < len(positions) and atom2 < len(positions):
                pos1, pos2 = positions[atom1], positions[atom2]
                line = self.ax.plot(
                    [pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                    'k-', linewidth=1, alpha=0.6
                )[0]
                self.bond_lines.append(line)
    
    def _setup_controls(self):
        """Set up interactive controls."""
        from matplotlib.widgets import Button, Slider
        
        # Control panel area
        control_height = 0.15
        button_width = 0.08
        button_height = 0.04
        slider_height = 0.03
        
        # Play/Pause button
        play_ax = self.fig.add_axes([0.1, 0.05, button_width, button_height])
        self.play_button = Button(play_ax, 'Play')
        self.play_button.on_clicked(self._toggle_play)
        
        # Step forward button
        step_ax = self.fig.add_axes([0.2, 0.05, button_width, button_height])
        self.step_button = Button(step_ax, 'Step')
        self.step_button.on_clicked(self._step_frame)
        
        # Speed slider
        speed_ax = self.fig.add_axes([0.35, 0.05, 0.2, slider_height])
        self.speed_slider = Slider(
            speed_ax, 'Speed', 0.1, 3.0, valinit=1.0, valfmt='%.1fx'
        )
        self.speed_slider.on_changed(self._update_speed)
        
        # Frame slider
        frame_ax = self.fig.add_axes([0.65, 0.05, 0.25, slider_height])
        self.frame_slider = Slider(
            frame_ax, 'Frame', 0, self.n_frames-1, 
            valinit=0, valfmt='%d', valstep=1
        )
        self.frame_slider.on_changed(self._goto_frame)
    
    def _update_frame(self, frame_num):
        """Update animation frame."""
        self.current_frame = frame_num
        pos = self.positions[frame_num]
        
        if self.display_mode == 'ball_stick':
            # Update scatter plot
            self.scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
            # Update bonds
            self._update_bonds(pos)
            
        elif self.display_mode == 'cartoon':
            # Update line plot
            self.scatter.set_data_3d(pos[:, 0], pos[:, 1], pos[:, 2])
            
        # Update title
        self.ax.set_title(f'Trajectory Animation - Frame {frame_num}/{self.n_frames-1}')
        
        # Update frame slider if it exists
        if self.frame_slider and not self.frame_slider.ax.contains_point(self.fig.canvas.get_tk_widget().winfo_pointerxy() if hasattr(self.fig.canvas, 'get_tk_widget') else (0, 0)):
            self.frame_slider.set_val(frame_num)
        
        return [self.scatter] + self.bond_lines
    
    def _toggle_play(self, event):
        """Toggle play/pause state."""
        if self.is_playing:
            self.animation.pause()
            if self.play_button:
                self.play_button.label.set_text('Play')
            self.is_playing = False
        else:
            self.animation.resume()
            if self.play_button:
                self.play_button.label.set_text('Pause')
            self.is_playing = True
    
    def _step_frame(self, event):
        """Step to next frame."""
        if not self.is_playing:
            next_frame = (self.current_frame + 1) % self.n_frames
            self.current_frame = next_frame  # Update current frame
            self._update_frame(next_frame)
            self.fig.canvas.draw()
    
    def _update_speed(self, val):
        """Update animation speed."""
        self.animation_speed = val
        # Update animation interval
        new_interval = self.interval / val
        self.animation.event_source.interval = new_interval
    
    def _goto_frame(self, val):
        """Go to specific frame."""
        if self._updating_slider:
            return  # Prevent circular updates
        frame_num = int(val)
        if frame_num != self.current_frame:
            self.goto_frame(frame_num)
    
    def play(self):
        """Start animation playback."""
        if not self.is_playing and self.animation:
            if self.controls and self.play_button:
                self._toggle_play(None)
            else:
                # For animations without controls, just resume
                self.animation.resume()
                self.is_playing = True
    
    def pause(self):
        """Pause animation playback."""
        if self.is_playing and self.animation:
            if self.controls and self.play_button:
                self._toggle_play(None)
            else:
                # For animations without controls, just pause
                self.animation.pause()
                self.is_playing = False
    
    def step(self):
        """Step to next frame."""
        if not self.is_playing:
            next_frame = (self.current_frame + 1) % self.n_frames
            self.goto_frame(next_frame)
    
    def goto_frame(self, frame_num):
        """Go to specific frame."""
        if 0 <= frame_num < self.n_frames:
            self.current_frame = frame_num
            # Directly update the visuals
            pos = self.positions[frame_num]
            
            if self.display_mode == 'ball_stick':
                self.scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
                self._update_bonds(pos)
            elif self.display_mode == 'cartoon':
                self.scatter.set_data_3d(pos[:, 0], pos[:, 1], pos[:, 2])
            
            # Update title
            self.ax.set_title(f'Trajectory Animation - Frame {frame_num}/{self.n_frames-1}')
            
            # Update frame slider if it exists (avoid circular calls)
            if hasattr(self, 'frame_slider') and self.frame_slider and not self._updating_slider:
                self._updating_slider = True
                self.frame_slider.set_val(frame_num)
                self._updating_slider = False
            
            if self.fig:
                self.fig.canvas.draw()
    
    def set_speed(self, speed):
        """Set animation speed multiplier."""
        if hasattr(self, 'speed_slider') and self.speed_slider:
            self.speed_slider.set_val(speed)
        else:
            self._update_speed(speed)
    
    def save_animation(self, filename, fps=10, dpi=100, bitrate=1800):
        """
        Save animation to file.
        
        Parameters
        ----------
        filename : str
            Output filename (extension determines format: .mp4, .gif, .avi)
        fps : int
            Frames per second
        dpi : int
            Resolution in dots per inch
        bitrate : int
            Bitrate for video formats
        """
        try:
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == '.gif':
                # Save as GIF
                writer = 'pillow'
                self.animation.save(filename, writer=writer, fps=fps, dpi=dpi)
                
            elif file_ext in ['.mp4', '.avi']:
                # Save as video
                writer = 'ffmpeg'
                self.animation.save(
                    filename, writer=writer, fps=fps, dpi=dpi, 
                    bitrate=bitrate, extra_args=['-vcodec', 'libx264']
                )
                
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
            logger.info(f"Saved animation to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save animation: {e}")
            # Fallback: try without extra args
            try:
                self.animation.save(filename, fps=fps, dpi=dpi)
                logger.info(f"Saved animation to {filename} (fallback method)")
            except Exception as e2:
                logger.error(f"Fallback save also failed: {e2}")
                raise
    
    def show(self):
        """Display the animation."""
        if self.fig:
            plt.show()
        else:
            logger.error("Animation not set up. Call setup_animation() first.")
    
    def close(self):
        """Close animation and clean up."""
        if self.animation:
            self.animation.pause()
        if self.fig:
            plt.close(self.fig)
        logger.info("Closed trajectory animation")


# Utility function for quick trajectory visualization
def animate_trajectory(positions, elements=None, bonds=None, interval=100, 
                      display_mode='ball_stick', controls=True, save_as=None, show=True):
    """
    Quick function to create and display trajectory animation.
    
    Parameters
    ----------
    positions : array_like
        Position data [n_frames, n_atoms, 3]
    elements : list of str, optional
        Element names for each atom
    bonds : list of tuples, optional
        Bond connectivity
    interval : int
        Animation interval in milliseconds
    display_mode : str
        Visualization mode ('ball_stick', 'cartoon', 'surface')
    controls : bool
        Whether to add interactive controls
    save_as : str, optional
        Filename to save animation
    show : bool
        Whether to display the animation
        
    Returns
    -------
    TrajectoryAnimator
        The animation object
    """
    animator = TrajectoryAnimator(
        positions=positions,
        elements=elements,
        bonds=bonds,
        interval=interval,
        display_mode=display_mode,
        controls=controls
    )
    
    animator.setup_animation()
    
    if save_as:
        animator.save_animation(save_as)
    
    if show:
        animator.show()
    
    return animator
