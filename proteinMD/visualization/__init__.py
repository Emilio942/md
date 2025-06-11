"""
Visualization module for molecular dynamics simulations.

This module provides classes and functions for visualizing
the results of MD simulations using various plotting libraries.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Import 3D protein visualization
try:
    from .protein_3d import (
        Protein3DVisualizer, 
        InteractiveProteinViewer,
        quick_visualize,
        create_comparison_view,
        ELEMENT_COLORS,
        VDW_RADII,
        SS_COLORS
    )
except ImportError:
    logger.warning("3D protein visualization not available")

# Import energy dashboard
try:
    from .energy_dashboard import (
        EnergyPlotDashboard,
        create_energy_dashboard
    )
except ImportError:
    logger.warning("Energy dashboard not available")

# Import trajectory animation - Task 2.2
try:
    from .trajectory_animation import (
        TrajectoryAnimator,
        create_trajectory_animator, 
        animate_trajectory,
        export_trajectory_video,
        load_trajectory_from_file
    )
except ImportError:
    logger.warning("Trajectory animation not available")

# Import real-time viewer if available - commented out to avoid circular import
# Imports should be done directly from proteinMD.visualization.realtime_viewer
# try:
#     from .realtime_viewer import RealTimeViewer
# except ImportError:
#     logger.warning("RealTimeViewer not available")

class Visualization:
    """
    Base class for visualization methods.
    
    This provides common functionality for creating visualizations.
    """
    
    def __init__(self, name: str):
        """
        Initialize a visualization method.
        
        Parameters
        ----------
        name : str
            Name of the visualization method
        """
        self.name = name
        self.fig = None
        self.ax = None
    
    def show(self):
        """Show the current visualization."""
        if self.fig is not None:
            plt.show()
        # Cleanup after animation
        plt.close('all')
        import gc
        gc.collect()
    
    def save(self, filename: str, dpi: int = 300):
        """
        Save the current visualization to a file.
        
        Parameters
        ----------
        filename : str
            Name of the output file
        dpi : int
            Resolution in dots per inch
        """
        if self.fig is not None:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved visualization to {filename}")


class StructureVisualization(Visualization):
    """
    Visualization of molecular structures.
    
    This class provides methods for visualizing protein structures
    and other molecular configurations.
    """
    
    def __init__(self):
        """Initialize structure visualization."""
        super().__init__("Structure Visualization")
    
    def plot_structure(self, positions: np.ndarray, elements: List[str] = None,
                      box_size: np.ndarray = None, view_angle: Tuple[float, float] = (30, 30)):
        """
        Plot a molecular structure.
        
        Parameters
        ----------
        positions : ndarray
            Atom positions [N, 3]
        elements : list of str, optional
            Element names for each atom
        box_size : ndarray, optional
            Size of the simulation box [x, y, z]
        view_angle : tuple of float
            View angle as (elevation, azimuth) in degrees
        """
        # Create figure
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set up element colors
        if elements is None:
            elements = ['C'] * len(positions)
        
        element_colors = {
            'H': 'white',
            'C': 'black',
            'N': 'blue',
            'O': 'red',
            'P': 'orange',
            'S': 'yellow',
            'Na': 'purple',
            'Cl': 'green',
            'Fe': 'brown',
            'Zn': 'gray'
        }
        
        element_sizes = {
            'H': 20,
            'C': 40,
            'N': 40,
            'O': 40,
            'P': 50,
            'S': 50,
            'Na': 30,
            'Cl': 30,
            'Fe': 60,
            'Zn': 60
        }
        
        # Get colors and sizes for each atom
        colors = [element_colors.get(e, 'gray') for e in elements]
        sizes = [element_sizes.get(e, 40) for e in elements]
        
        # Plot atoms
        self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c=colors, s=sizes, alpha=0.8, edgecolors='black')
        
        # Draw simulation box if provided
        if box_size is not None:
            self._draw_box(box_size)
        
        # Set axis labels
        self.ax.set_xlabel('X (nm)')
        self.ax.set_ylabel('Y (nm)')
        self.ax.set_zlabel('Z (nm)')
        
        # Set view angle
        self.ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Set axis limits
        if box_size is not None:
            self.ax.set_xlim(0, box_size[0])
            self.ax.set_ylim(0, box_size[1])
            self.ax.set_zlim(0, box_size[2])
        else:
            # Set limits based on positions
            margin = 0.1
            min_pos = np.min(positions, axis=0) - margin
            max_pos = np.max(positions, axis=0) + margin
            self.ax.set_xlim(min_pos[0], max_pos[0])
            self.ax.set_ylim(min_pos[1], max_pos[1])
            self.ax.set_zlim(min_pos[2], max_pos[2])
        
        # Set title
        self.ax.set_title('Molecular Structure')
        
        return self.fig, self.ax
    
    def _draw_box(self, box_size: np.ndarray):
        """
        Draw a simulation box.
        
        Parameters
        ----------
        box_size : ndarray
            Size of the simulation box [x, y, z]
        """
        # Define box corners
        corners = np.array([
            [0, 0, 0],
            [box_size[0], 0, 0],
            [box_size[0], box_size[1], 0],
            [0, box_size[1], 0],
            [0, 0, box_size[2]],
            [box_size[0], 0, box_size[2]],
            [box_size[0], box_size[1], box_size[2]],
            [0, box_size[1], box_size[2]]
        ])
        
        # Define box edges as pairs of corner indices
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
        
        # Draw each edge
        for i, j in edges:
            self.ax.plot(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                color='gray', linestyle='--', linewidth=1
            )


class TrajectoryVisualization(Visualization):
    """
    Visualization of MD trajectories.
    
    This class provides methods for creating animations of
    molecular dynamics simulations.
    """
    
    def __init__(self):
        """Initialize trajectory visualization."""
        super().__init__("Trajectory Visualization")
        self.animation = None
    
    def animate_trajectory(self, trajectory, elements: List[str] = None,
                          interval: int = 50, stride: int = 1,
                          start_frame: int = 0, end_frame: int = None):
        """
        Create an animation of a trajectory.
        
        Parameters
        ----------
        trajectory : Trajectory
            The trajectory to animate
        elements : list of str, optional
            Element names for each atom
        interval : int
            Time between frames in milliseconds
        stride : int
            Stride for selecting frames
        start_frame : int
            First frame to include
        end_frame : int, optional
            Last frame to include
        
        Returns
        -------
        FuncAnimation
            The animation object
        """
        # Get positions and box sizes
        positions = trajectory.positions
        box_sizes = trajectory.box_sizes
        
        # Apply frame selection
        if end_frame is None:
            end_frame = len(positions)
        
        selected_frames = np.arange(start_frame, end_frame, stride)
        positions = positions[selected_frames]
        box_sizes = box_sizes[selected_frames]
        
        num_frames = len(selected_frames)
        
        # Set up element colors
        if elements is None:
            elements = ['C'] * positions.shape[1]
        
        element_colors = {
            'H': 'white',
            'C': 'black',
            'N': 'blue',
            'O': 'red',
            'P': 'orange',
            'S': 'yellow',
            'Na': 'purple',
            'Cl': 'green',
            'Fe': 'brown',
            'Zn': 'gray'
        }
        
        element_sizes = {
            'H': 20,
            'C': 40,
            'N': 40,
            'O': 40,
            'P': 50,
            'S': 50,
            'Na': 30,
            'Cl': 30,
            'Fe': 60,
            'Zn': 60
        }
        
        # Get colors and sizes for each atom
        colors = [element_colors.get(e, 'gray') for e in elements]
        sizes = [element_sizes.get(e, 40) for e in elements]
        
        # Create figure
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initial plot
        frame_idx = selected_frames[0]
        box_size = box_sizes[0]
        
        # Plot atoms
        scatter = self.ax.scatter(
            positions[0, :, 0],
            positions[0, :, 1],
            positions[0, :, 2],
            c=colors, s=sizes, alpha=0.8, edgecolors='black'
        )
        
        # Draw simulation box
        box_artists = self._draw_box(box_size)
        
        # Set axis labels
        self.ax.set_xlabel('X (nm)')
        self.ax.set_ylabel('Y (nm)')
        self.ax.set_zlabel('Z (nm)')
        
        # Set axis limits
        self.ax.set_xlim(0, box_size[0])
        self.ax.set_ylim(0, box_size[1])
        self.ax.set_zlim(0, box_size[2])
        
        # Set title with time information
        title = self.ax.set_title(f'Frame {frame_idx} (Time: {trajectory.time[frame_idx]:.1f} ps)')
        
        # Function to update the animation
        def update(frame_idx):
            # Update positions
            scatter._offsets3d = (
                positions[frame_idx, :, 0],
                positions[frame_idx, :, 1],
                positions[frame_idx, :, 2]
            )
            
            # Update box if it changes
            box_size = box_sizes[frame_idx]
            self._update_box(box_artists, box_size)
            
            # Update title
            title.set_text(f'Frame {frame_idx} (Time: {trajectory.time[frame_idx]:.1f} ps)')
            
            return scatter, *box_artists, title
        
        # Create animation
        self.animation = FuncAnimation(
            self.fig, update, frames=selected_frames,
            interval=interval, blit=True
        )
        
        return self.animation
    
    def save_animation(self, filename: str, fps: int = 20, dpi: int = 100):
        """
        Save an animation to a file.
        
        Parameters
        ----------
        filename : str
            Name of the output file
        fps : int
            Frames per second
        dpi : int
            Resolution in dots per inch
        """
        if self.animation is not None:
            self.animation.save(filename, fps=fps, dpi=dpi)
            logger.info(f"Saved animation to {filename}")
    
    def _draw_box(self, box_size: np.ndarray):
        """
        Draw a simulation box.
        
        Parameters
        ----------
        box_size : ndarray
            Size of the simulation box [x, y, z]
        
        Returns
        -------
        list
            List of artists (line objects)
        """
        # Define box corners
        corners = np.array([
            [0, 0, 0],
            [box_size[0], 0, 0],
            [box_size[0], box_size[1], 0],
            [0, box_size[1], 0],
            [0, 0, box_size[2]],
            [box_size[0], 0, box_size[2]],
            [box_size[0], box_size[1], box_size[2]],
            [0, box_size[1], box_size[2]]
        ])
        
        # Define box edges as pairs of corner indices
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
        
        # Draw each edge and store the artists
        artists = []
        for i, j in edges:
            line, = self.ax.plot(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                color='gray', linestyle='--', linewidth=1
            )
            artists.append(line)
        
        return artists
    
    def _update_box(self, artists, box_size: np.ndarray):
        """
        Update the simulation box.
        
        Parameters
        ----------
        artists : list
            List of line artists
        box_size : ndarray
            Size of the simulation box [x, y, z]
        """
        # Define box corners
        corners = np.array([
            [0, 0, 0],
            [box_size[0], 0, 0],
            [box_size[0], box_size[1], 0],
            [0, box_size[1], 0],
            [0, 0, box_size[2]],
            [box_size[0], 0, box_size[2]],
            [box_size[0], box_size[1], box_size[2]],
            [0, box_size[1], box_size[2]]
        ])
        
        # Define box edges as pairs of corner indices
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
        
        # Update each edge
        for k, (i, j) in enumerate(edges):
            artists[k].set_data_3d(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]]
            )


class AnalysisVisualization(Visualization):
    """
    Visualization of analysis results.
    
    This class provides methods for creating plots of
    analysis results from MD simulations.
    """
    
    def __init__(self):
        """Initialize analysis visualization."""
        super().__init__("Analysis Visualization")
    
    def plot_rmsd(self, rmsd_data: Dict):
        """
        Plot RMSD data.
        
        Parameters
        ----------
        rmsd_data : dict
            RMSD data from RMSD analysis
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Plot RMSD vs. time
        self.ax.plot(rmsd_data['time'], rmsd_data['rmsd'])
        
        # Set labels
        self.ax.set_xlabel('Time (ps)')
        self.ax.set_ylabel('RMSD (nm)')
        self.ax.set_title('Root Mean Square Deviation')
        
        # Add grid
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        return self.fig, self.ax
    
    def plot_rmsf(self, rmsf_data: Dict):
        """
        Plot RMSF data.
        
        Parameters
        ----------
        rmsf_data : dict
            RMSF data from RMSF analysis
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Plot RMSF vs. atom ID
        self.ax.plot(rmsf_data['atom_ids'], rmsf_data['rmsf'])
        
        # Set labels
        self.ax.set_xlabel('Atom ID')
        self.ax.set_ylabel('RMSF (nm)')
        self.ax.set_title('Root Mean Square Fluctuation')
        
        # Add grid
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        return self.fig, self.ax
    
    def plot_rdf(self, rdf_data: Dict, label: str = None):
        """
        Plot Radial Distribution Function data.
        
        Parameters
        ----------
        rdf_data : dict
            RDF data from RadialDistributionFunction analysis
        label : str, optional
            Label for the RDF curve
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Plot RDF vs. distance
        if label:
            self.ax.plot(rdf_data['r'], rdf_data['rdf'], label=label)
            self.ax.legend()
        else:
            self.ax.plot(rdf_data['r'], rdf_data['rdf'])
        
        # Set labels
        self.ax.set_xlabel('Distance (nm)')
        self.ax.set_ylabel('g(r)')
        self.ax.set_title('Radial Distribution Function')
        
        # Add grid
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        return self.fig, self.ax
    
    def plot_hbonds(self, hbond_data: Dict):
        """
        Plot hydrogen bond data.
        
        Parameters
        ----------
        hbond_data : dict
            Hydrogen bond data from HydrogenBondAnalysis
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Plot number of hydrogen bonds vs. time
        self.ax.plot(hbond_data['time'], hbond_data['num_hbonds'])
        
        # Set labels
        self.ax.set_xlabel('Time (ps)')
        self.ax.set_ylabel('Number of Hydrogen Bonds')
        self.ax.set_title('Hydrogen Bond Analysis')
        
        # Add grid
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        return self.fig, self.ax
    
    def plot_secondary_structure(self, ss_data: Dict):
        """
        Plot secondary structure data.
        
        Parameters
        ----------
        ss_data : dict
            Secondary structure data from SecondaryStructureAnalysis
        """
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Plot percentage of each secondary structure type vs. time
        for i, ss_type in enumerate(ss_data['ss_types']):
            self.ax.plot(ss_data['time'], ss_data['ss_percent'][:, i], label=ss_type)
        
        # Set labels
        self.ax.set_xlabel('Time (ps)')
        self.ax.set_ylabel('Percentage')
        self.ax.set_title('Secondary Structure Analysis')
        
        # Add legend
        self.ax.legend()
        
        # Add grid
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        return self.fig, self.ax
