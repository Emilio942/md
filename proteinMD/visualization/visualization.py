"""
Visualization module for molecular dynamics simulations.

This module provides classes and functions for visualizing
the results of MD simulations using various plotting libraries.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import os
from pathlib import Path
import io
import base64
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches

# Configure logging
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import nglview as nv
    HAS_NGLVIEW = True
except ImportError:
    HAS_NGLVIEW = False
    logger.warning("nglview not found, 3D interactive visualization will not be available")

try:
    import py3Dmol
    HAS_PY3DMOL = True
except ImportError:
    HAS_PY3DMOL = False
    logger.warning("py3Dmol not found, some 3D visualization features will not be available")


class TrajectoryVisualizer:
    """
    Base class for visualizing MD trajectories.
    """
    
    def __init__(self, trajectory_data: List[np.ndarray], topology=None, box_vectors=None):
        """
        Initialize a trajectory visualizer.
        
        Parameters
        ----------
        trajectory_data : list of np.ndarray
            List of position arrays, each with shape (n_particles, 3)
        topology : object, optional
            Topology information for the system
        box_vectors : np.ndarray, optional
            Periodic box vectors
        """
        self.trajectory_data = trajectory_data
        self.topology = topology
        self.box_vectors = box_vectors
        
        # Calculate derived properties
        self.n_frames = len(trajectory_data)
        if self.n_frames > 0:
            self.n_particles = trajectory_data[0].shape[0]
        else:
            self.n_particles = 0
            
        logger.info(f"Initialized trajectory visualizer with {self.n_frames} frames, {self.n_particles} particles")
    
    def animate(self, output_file: Optional[str] = None, start_frame: int = 0, 
               end_frame: Optional[int] = None, stride: int = 1,
               fps: int = 10):
        """
        Create an animation of the trajectory.
        
        Parameters
        ----------
        output_file : str, optional
            Path to save the animation to
        start_frame : int, optional
            First frame to include
        end_frame : int, optional
            Last frame to include (exclusive)
        stride : int, optional
            Stride between frames
        fps : int, optional
            Frames per second in the animation
            
        Returns
        -------
        Animation
            Animation object
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def render_frame(self, frame_index: int = 0):
        """
        Render a single frame from the trajectory.
        
        Parameters
        ----------
        frame_index : int, optional
            Index of the frame to render
            
        Returns
        -------
        object
            Rendering of the frame
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def create_interactive_view(self):
        """
        Create an interactive view of the trajectory.
        
        Returns
        -------
        object
            Interactive view
        """
        raise NotImplementedError("Subclasses must implement this method")


class MatplotlibVisualizer(TrajectoryVisualizer):
    """
    Trajectory visualizer using Matplotlib.
    """
    
    def __init__(self, trajectory_data: List[np.ndarray], topology=None, box_vectors=None):
        """
        Initialize a Matplotlib trajectory visualizer.
        
        Parameters
        ----------
        trajectory_data : list of np.ndarray
            List of position arrays, each with shape (n_particles, 3)
        topology : object, optional
            Topology information for the system
        box_vectors : np.ndarray, optional
            Periodic box vectors
        """
        super().__init__(trajectory_data, topology, box_vectors)
        
        # Default colors and sizes
        self.default_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        self.default_particle_size = 50
        
        # Particle properties
        self.particle_colors = None
        self.particle_sizes = None
        self.particle_labels = None
        
        # Set default particle properties if topology is provided
        if topology is not None:
            self._set_particle_properties_from_topology()
    
    def _set_particle_properties_from_topology(self):
        """
        Set particle properties from topology information.
        """
        # This would be implemented to extract colors, sizes, etc.
        # from the topology
        pass
    
    def set_particle_colors(self, colors):
        """
        Set particle colors.
        
        Parameters
        ----------
        colors : array-like
            Colors for each particle
        """
        self.particle_colors = colors
    
    def set_particle_sizes(self, sizes):
        """
        Set particle sizes.
        
        Parameters
        ----------
        sizes : array-like
            Sizes for each particle
        """
        self.particle_sizes = sizes
    
    def set_particle_labels(self, labels):
        """
        Set particle labels.
        
        Parameters
        ----------
        labels : array-like
            Labels for each particle
        """
        self.particle_labels = labels
    
    def render_frame(self, frame_index: int = 0, ax=None, show_box: bool = True, 
                   show_labels: bool = False, view_angle: Optional[Tuple[float, float]] = None):
        """
        Render a single frame from the trajectory using Matplotlib.
        
        Parameters
        ----------
        frame_index : int, optional
            Index of the frame to render
        ax : Axes3D, optional
            Matplotlib 3D axis to render on
        show_box : bool, optional
            Whether to show the simulation box
        show_labels : bool, optional
            Whether to show particle labels
        view_angle : tuple, optional
            (elevation, azimuth) view angles
            
        Returns
        -------
        tuple
            Tuple of (fig, ax)
        """
        if frame_index < 0 or frame_index >= self.n_frames:
            raise ValueError(f"Frame index {frame_index} out of range [0, {self.n_frames-1}]")
            
        # Get frame data
        positions = self.trajectory_data[frame_index]
        
        # Create figure if needed
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
            
        # Clear existing plot
        ax.clear()
        
        # Get particle colors and sizes
        colors = self.particle_colors if self.particle_colors is not None else self.default_colors[0]
        sizes = self.particle_sizes if self.particle_sizes is not None else self.default_particle_size
        
        # Plot particles
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        ax.scatter(x, y, z, c=colors, s=sizes, alpha=0.8)
        
        # Add labels if requested
        if show_labels and self.particle_labels is not None:
            for i, (xi, yi, zi, label) in enumerate(zip(x, y, z, self.particle_labels)):
                ax.text(xi, yi, zi, label, size=8, zorder=1, color='k')
        
        # Show simulation box if requested
        if show_box and self.box_vectors is not None:
            self._draw_box(ax)
            
        # Set view angle if provided
        if view_angle is not None:
            ax.view_init(elev=view_angle[0], azim=view_angle[1])
            
        # Set labels and title
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Z (nm)')
        ax.set_title(f'Frame {frame_index}')
        
        # Make axes equal
        self._set_axes_equal(ax)
        
        return fig, ax
    
    def _draw_box(self, ax):
        """
        Draw the simulation box.
        
        Parameters
        ----------
        ax : Axes3D
            Matplotlib 3D axis
        """
        if self.box_vectors is None:
            return
            
        # Extract box vectors
        if self.box_vectors.ndim == 1:
            # Simple box dimensions
            box_dims = self.box_vectors
            a, b, c = box_dims[0], box_dims[1], box_dims[2]
            box = np.array([
                [0, 0, 0],
                [a, 0, 0],
                [a, b, 0],
                [0, b, 0],
                [0, 0, c],
                [a, 0, c],
                [a, b, c],
                [0, b, c]
            ])
        else:
            # General triclinic box
            a, b, c = self.box_vectors
            origin = np.zeros(3)
            box = np.array([
                origin,
                a,
                a + b,
                b,
                c,
                a + c,
                a + b + c,
                b + c
            ])
        
        # Plot box edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
        
        for i, j in edges:
            ax.plot([box[i, 0], box[j, 0]], 
                   [box[i, 1], box[j, 1]], 
                   [box[i, 2], box[j, 2]], 'k-', alpha=0.3)
    
    def _set_axes_equal(self, ax):
        """
        Make axes of 3D plot have equal scale.
        
        Parameters
        ----------
        ax : Axes3D
            Matplotlib 3D axis
        """
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        
        # Find the greatest range for equal aspect ratio
        max_range = max(x_range, y_range, z_range)
        
        # Set limits based on the center and max_range
        x_mid = np.mean(x_limits)
        y_mid = np.mean(y_limits)
        z_mid = np.mean(z_limits)
        
        ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
        ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
        ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])
    
    def animate(self, output_file: Optional[str] = None, start_frame: int = 0, 
               end_frame: Optional[int] = None, stride: int = 1,
               fps: int = 10, view_angle: Optional[Tuple[float, float]] = None):
        """
        Create an animation of the trajectory using Matplotlib.
        
        Parameters
        ----------
        output_file : str, optional
            Path to save the animation to
        start_frame : int, optional
            First frame to include
        end_frame : int, optional
            Last frame to include (exclusive)
        stride : int, optional
            Stride between frames
        fps : int, optional
            Frames per second in the animation
        view_angle : tuple, optional
            (elevation, azimuth) view angles
            
        Returns
        -------
        Animation
            Matplotlib animation object
        """
        # Validate frame range
        if start_frame < 0:
            start_frame = 0
            
        if end_frame is None or end_frame > self.n_frames:
            end_frame = self.n_frames
            
        # Create figure and initial plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if view_angle is not None:
            ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Function to update the plot for each frame
        def update(frame_idx):
            # Calculate actual frame index
            actual_idx = start_frame + frame_idx * stride
            if actual_idx >= end_frame:
                actual_idx = end_frame - 1
                
            # Render the frame
            self.render_frame(actual_idx, ax=ax)
            ax.set_title(f'Frame {actual_idx}')
            
            return ax,
        
        # Create animation
        n_animation_frames = (end_frame - start_frame + stride - 1) // stride
        ani = FuncAnimation(
            fig, update, frames=n_animation_frames, 
            interval=1000/fps, blit=False
        )
        
        # Save if requested
        if output_file is not None:
            ani.save(output_file, writer='pillow', fps=fps)
            logger.info(f"Saved animation to {output_file}")
        
        return ani
    
    def create_interactive_view(self):
        """
        Create an interactive view of the trajectory.
        
        For MatplotlibVisualizer, this just returns the animation.
        """
        return self.animate()


class NGLViewVisualizer(TrajectoryVisualizer):
    """
    Trajectory visualizer using NGLView for interactive 3D visualization.
    
    Requires the nglview package to be installed.
    """
    
    def __init__(self, trajectory_data: List[np.ndarray], topology=None, box_vectors=None):
        """
        Initialize an NGLView trajectory visualizer.
        
        Parameters
        ----------
        trajectory_data : list of np.ndarray
            List of position arrays, each with shape (n_particles, 3)
        topology : object, optional
            Topology information for the system
        box_vectors : np.ndarray, optional
            Periodic box vectors
        """
        super().__init__(trajectory_data, topology, box_vectors)
        
        if not HAS_NGLVIEW:
            raise ImportError("nglview package is required for NGLViewVisualizer")
            
        # Create a structure suitable for nglview
        self._prepare_structure()
    
    def _prepare_structure(self):
        """
        Prepare structure for NGLView.
        """
        # This would convert the trajectory data into a format
        # that NGLView can understand, typically a structure + trajectory
        pass
    
    def render_frame(self, frame_index: int = 0):
        """
        Render a single frame from the trajectory using NGLView.
        
        Parameters
        ----------
        frame_index : int, optional
            Index of the frame to render
            
        Returns
        -------
        nglview.NGLWidget
            NGLView widget showing the frame
        """
        # Create a view focused on a single frame
        pass
    
    def create_interactive_view(self):
        """
        Create an interactive view of the trajectory using NGLView.
        
        Returns
        -------
        nglview.NGLWidget
            Interactive NGLView widget
        """
        # Create an interactive view of the trajectory
        pass


class EnergyPlotter:
    """
    Class for plotting energy data from simulations.
    """
    
    def __init__(self, energy_data: Dict[str, List[float]], time_data: Optional[List[float]] = None):
        """
        Initialize an energy plotter.
        
        Parameters
        ----------
        energy_data : dict
            Dictionary mapping energy component names to lists of values
        time_data : list, optional
            Time points for the energy data
        """
        self.energy_data = energy_data
        
        # Use frame indices if time_data not provided
        if time_data is None:
            if len(next(iter(energy_data.values()))) > 0:
                self.time_data = np.arange(len(next(iter(energy_data.values()))))
            else:
                self.time_data = []
        else:
            self.time_data = time_data
            
        logger.info(f"Initialized energy plotter with {len(self.time_data)} data points")
    
    def plot_energies(self, components: Optional[List[str]] = None, 
                     start_time: Optional[float] = None, end_time: Optional[float] = None,
                     rolling_avg: Optional[int] = None, ax=None):
        """
        Plot energy components.
        
        Parameters
        ----------
        components : list, optional
            List of energy components to plot. If None, plot all.
        start_time : float, optional
            Start time for the plot
        end_time : float, optional
            End time for the plot
        rolling_avg : int, optional
            Window size for rolling average
        ax : Axes, optional
            Matplotlib axis to plot on
            
        Returns
        -------
        tuple
            Tuple of (fig, ax)
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
            
        # Determine components to plot
        if components is None:
            components = list(self.energy_data.keys())
            
        # Filter by time range
        mask = np.ones_like(self.time_data, dtype=bool)
        if start_time is not None:
            mask &= self.time_data >= start_time
        if end_time is not None:
            mask &= self.time_data <= end_time
            
        time = np.array(self.time_data)[mask]
        
        # Plot each component
        for component in components:
            if component not in self.energy_data:
                logger.warning(f"Energy component '{component}' not found")
                continue
                
            values = np.array(self.energy_data[component])[mask]
            
            # Apply rolling average if requested
            if rolling_avg is not None and rolling_avg > 1:
                values = self._rolling_average(values, rolling_avg)
                # Trim time to match averaged values
                time_avg = time[:len(values)]
                ax.plot(time_avg, values, label=component)
            else:
                ax.plot(time, values, label=component)
            
        # Add labels and legend
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Energy (kJ/mol)')
        ax.set_title('Energy Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def plot_energy_distribution(self, component: str, bins: int = 50, ax=None):
        """
        Plot the distribution of an energy component.
        
        Parameters
        ----------
        component : str
            Energy component to plot
        bins : int, optional
            Number of bins for the histogram
        ax : Axes, optional
            Matplotlib axis to plot on
            
        Returns
        -------
        tuple
            Tuple of (fig, ax)
        """
        if component not in self.energy_data:
            raise ValueError(f"Energy component '{component}' not found")
            
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
            
        # Get data
        values = np.array(self.energy_data[component])
        
        # Plot histogram
        ax.hist(values, bins=bins, density=True, alpha=0.7)
        
        # Add a kernel density estimate
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(values)
            x = np.linspace(min(values), max(values), 1000)
            ax.plot(x, kde(x), 'r-', lw=2)
        except ImportError:
            pass
        
        # Add mean and std lines
        mean = np.mean(values)
        std = np.std(values)
        
        ax.axvline(mean, color='k', linestyle='--', 
                  label=f'Mean: {mean:.2f} kJ/mol')
        ax.axvline(mean + std, color='k', linestyle=':', 
                  label=f'Std: {std:.2f} kJ/mol')
        ax.axvline(mean - std, color='k', linestyle=':')
        
        # Add labels and legend
        ax.set_xlabel(f'{component} (kJ/mol)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Distribution of {component}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def plot_energy_breakdown(self, time_point: Optional[float] = None, ax=None):
        """
        Plot a breakdown of energy components at a specific time point.
        
        Parameters
        ----------
        time_point : float, optional
            Time point to use. If None, use the last point.
        ax : Axes, optional
            Matplotlib axis to plot on
            
        Returns
        -------
        tuple
            Tuple of (fig, ax)
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
            
        # Determine index to use
        if time_point is None:
            idx = -1
        else:
            idx = np.argmin(np.abs(np.array(self.time_data) - time_point))
            
        # Get values at the selected time point
        labels = []
        values = []
        
        for component, data in self.energy_data.items():
            if len(data) > idx:
                labels.append(component)
                values.append(data[idx])
        
        # Plot bar chart
        bars = ax.bar(labels, values)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}',
                   ha='center', va='bottom', rotation=0)
        
        # Add labels
        time_label = f'at {self.time_data[idx]:.1f} ps' if time_point is not None else '(final)'
        ax.set_xlabel('Energy Component')
        ax.set_ylabel('Energy (kJ/mol)')
        ax.set_title(f'Energy Breakdown {time_label}')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Make room for labels
        fig.tight_layout()
        
        return fig, ax
    
    def _rolling_average(self, values, window_size):
        """
        Calculate a rolling average.
        
        Parameters
        ----------
        values : array-like
            Values to average
        window_size : int
            Size of the averaging window
            
        Returns
        -------
        np.ndarray
            Averaged values
        """
        cumsum = np.cumsum(np.insert(values, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


class StructureVisualizer:
    """
    Class for visualizing static molecular structures.
    """
    
    def __init__(self, positions: np.ndarray, elements=None, bonds=None, box_vectors=None):
        """
        Initialize a structure visualizer.
        
        Parameters
        ----------
        positions : np.ndarray
            Particle positions with shape (n_particles, 3)
        elements : list, optional
            Element symbols for each particle
        bonds : list, optional
            List of bond indices as (i, j) tuples
        box_vectors : np.ndarray, optional
            Periodic box vectors
        """
        self.positions = positions
        self.n_particles = positions.shape[0]
        self.elements = elements
        self.bonds = bonds
        self.box_vectors = box_vectors
        
        # Set up default colors and sizes
        self._set_default_properties()
        
        logger.info(f"Initialized structure visualizer with {self.n_particles} particles")
    
    def _set_default_properties(self):
        """
        Set default visualization properties.
        """
        # Default element properties
        self.element_colors = {
            'C': 'black',
            'N': 'blue',
            'O': 'red',
            'H': 'white',
            'S': 'yellow',
            'P': 'orange',
            'F': 'green',
            'Cl': 'green',
            'Br': 'brown',
            'I': 'purple',
            'Na': 'purple',
            'K': 'purple',
            'Ca': 'gray',
            'Mg': 'gray',
            'Fe': 'darkred',
            'Zn': 'gray'
        }
        
        self.element_sizes = {
            'C': 70,
            'N': 65,
            'O': 60,
            'H': 30,
            'S': 100,
            'P': 100,
            'F': 50,
            'Cl': 100,
            'Br': 115,
            'I': 140,
            'Na': 180,
            'K': 220,
            'Ca': 180,
            'Mg': 150,
            'Fe': 140,
            'Zn': 139
        }
        
        # Set colors and sizes based on elements
        self.particle_colors = []
        self.particle_sizes = []
        
        if self.elements is not None:
            for element in self.elements:
                self.particle_colors.append(
                    self.element_colors.get(element, 'gray'))
                self.particle_sizes.append(
                    self.element_sizes.get(element, 50))
        else:
            self.particle_colors = ['gray'] * self.n_particles
            self.particle_sizes = [50] * self.n_particles
    
    def render_structure(self, ax=None, style='ball_and_stick', 
                       show_box=True, show_legend=True, view_angle=None):
        """
        Render the structure using Matplotlib.
        
        Parameters
        ----------
        ax : Axes3D, optional
            Matplotlib 3D axis to render on
        style : str, optional
            Rendering style: 'ball_and_stick', 'space_filling', or 'wireframe'
        show_box : bool, optional
            Whether to show the simulation box
        show_legend : bool, optional
            Whether to show a legend for element colors
        view_angle : tuple, optional
            (elevation, azimuth) view angles
            
        Returns
        -------
        tuple
            Tuple of (fig, ax)
        """
        # Create figure if needed
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
            
        # Clear existing plot
        ax.clear()
        
        # Apply style settings
        if style == 'ball_and_stick':
            # Default settings
            pass
        elif style == 'space_filling':
            # Increase particle sizes
            self.particle_sizes = [s * 2 for s in self.particle_sizes]
        elif style == 'wireframe':
            # Decrease particle sizes, focus on bonds
            self.particle_sizes = [s * 0.5 for s in self.particle_sizes]
        else:
            logger.warning(f"Unknown style '{style}', using default")
            
        # Plot particles
        x, y, z = self.positions[:, 0], self.positions[:, 1], self.positions[:, 2]
        ax.scatter(x, y, z, c=self.particle_colors, s=self.particle_sizes, alpha=0.8)
        
        # Plot bonds if available
        if self.bonds is not None and len(self.bonds) > 0:
            for i, j in self.bonds:
                if i < self.n_particles and j < self.n_particles:
                    xi, yi, zi = self.positions[i]
                    xj, yj, zj = self.positions[j]
                    ax.plot([xi, xj], [yi, yj], [zi, zj], 'k-', alpha=0.6, lw=1)
        
        # Show box if requested
        if show_box and self.box_vectors is not None:
            self._draw_box(ax)
            
        # Set view angle if provided
        if view_angle is not None:
            ax.view_init(elev=view_angle[0], azim=view_angle[1])
            
        # Set labels
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Z (nm)')
        
        # Add legend if requested
        if show_legend and self.elements is not None:
            # Get unique elements
            unique_elements = set(self.elements)
            legend_elements = []
            
            for element in unique_elements:
                color = self.element_colors.get(element, 'gray')
                legend_elements.append(
                    mpatches.Patch(color=color, label=element))
                    
            ax.legend(handles=legend_elements, loc='upper right')
        
        # Make axes equal
        self._set_axes_equal(ax)
        
        return fig, ax
    
    def _draw_box(self, ax):
        """
        Draw the simulation box.
        
        Parameters
        ----------
        ax : Axes3D
            Matplotlib 3D axis
        """
        if self.box_vectors is None:
            return
            
        # Extract box vectors
        if self.box_vectors.ndim == 1:
            # Simple box dimensions
            box_dims = self.box_vectors
            a, b, c = box_dims[0], box_dims[1], box_dims[2]
            box = np.array([
                [0, 0, 0],
                [a, 0, 0],
                [a, b, 0],
                [0, b, 0],
                [0, 0, c],
                [a, 0, c],
                [a, b, c],
                [0, b, c]
            ])
        else:
            # General triclinic box
            a, b, c = self.box_vectors
            origin = np.zeros(3)
            box = np.array([
                origin,
                a,
                a + b,
                b,
                c,
                a + c,
                a + b + c,
                b + c
            ])
        
        # Plot box edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
        
        for i, j in edges:
            ax.plot([box[i, 0], box[j, 0]], 
                   [box[i, 1], box[j, 1]], 
                   [box[i, 2], box[j, 2]], 'k-', alpha=0.3)
    
    def _set_axes_equal(self, ax):
        """
        Make axes of 3D plot have equal scale.
        
        Parameters
        ----------
        ax : Axes3D
            Matplotlib 3D axis
        """
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        
        # Find the greatest range for equal aspect ratio
        max_range = max(x_range, y_range, z_range)
        
        # Set limits based on the center and max_range
        x_mid = np.mean(x_limits)
        y_mid = np.mean(y_limits)
        z_mid = np.mean(z_limits)
        
        ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
        ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
        ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])
    
    def render_py3dmol(self, width=500, height=500, style='stick'):
        """
        Render the structure using py3Dmol.
        
        Parameters
        ----------
        width : int, optional
            Width of the viewer
        height : int, optional
            Height of the viewer
        style : str, optional
            Rendering style: 'stick', 'sphere', 'line', or 'cross'
            
        Returns
        -------
        py3Dmol.view
            Interactive 3D viewer
        """
        if not HAS_PY3DMOL:
            raise ImportError("py3Dmol package is required for interactive 3D visualization")
            
        # Create viewer
        view = py3Dmol.view(width=width, height=height)
        
        # Create a model from positions and elements
        model = {"elem": [], "x": [], "y": [], "z": []}
        
        for i in range(self.n_particles):
            # Set element (default to carbon if not specified)
            elem = self.elements[i] if self.elements is not None else "C"
            model["elem"].append(elem)
            
            # Set coordinates (convert to Angstroms)
            model["x"].append(float(self.positions[i, 0] * 10))
            model["y"].append(float(self.positions[i, 1] * 10))
            model["z"].append(float(self.positions[i, 2] * 10))
        
        # Add model to viewer
        view.addModel(model, "json")
        
        # Add bonds if available
        if self.bonds is not None and len(self.bonds) > 0:
            for i, j in self.bonds:
                if i < self.n_particles and j < self.n_particles:
                    view.addBond(i, j)
        
        # Set style
        if style == 'stick':
            view.setStyle({'stick': {}})
        elif style == 'sphere':
            view.setStyle({'sphere': {'radius': 0.5}})
        elif style == 'line':
            view.setStyle({'line': {}})
        elif style == 'cross':
            view.setStyle({'cross': {'linewidth': 1}})
        else:
            view.setStyle({'stick': {}})
        
        # Set background color
        view.setBackgroundColor('white')
        
        # Center the view
        view.zoomTo()
        
        return view


class ProteinVisualizer:
    """
    Specialized visualizer for protein structures.
    """
    
    def __init__(self, protein):
        """
        Initialize a protein visualizer.
        
        Parameters
        ----------
        protein : Protein
            Protein object to visualize
        """
        self.protein = protein
        
        # Extract basic information
        positions, elements, bonds = self._extract_protein_data()
        
        # Create structure visualizer
        self.structure_viz = StructureVisualizer(positions, elements, bonds)
        
        logger.info(f"Initialized protein visualizer for {protein.name}")
    
    def _extract_protein_data(self):
        """
        Extract visualization data from the protein.
        
        Returns
        -------
        tuple
            Tuple of (positions, elements, bonds)
        """
        # This would extract the necessary information from the protein object
        # For now, return placeholders (should be implemented based on actual Protein class)
        positions = np.zeros((1, 3))
        elements = ["C"]
        bonds = []
        
        return positions, elements, bonds
    
    def render_protein(self, ax=None, style='cartoon', color_by='chain',
                     show_sidechains=False, show_labels=False, view_angle=None):
        """
        Render the protein structure.
        
        Parameters
        ----------
        ax : Axes3D, optional
            Matplotlib 3D axis to render on
        style : str, optional
            Rendering style: 'cartoon', 'backbone', 'all_atom', 'surface'
        color_by : str, optional
            Coloring scheme: 'chain', 'residue_type', 'secondary_structure', 'b_factor'
        show_sidechains : bool, optional
            Whether to show side chains
        show_labels : bool, optional
            Whether to show residue labels
        view_angle : tuple, optional
            (elevation, azimuth) view angles
            
        Returns
        -------
        tuple
            Tuple of (fig, ax)
        """
        # This would implement specialized protein rendering
        # For now, delegate to the structure visualizer
        return self.structure_viz.render_structure(ax, style='ball_and_stick', view_angle=view_angle)
    
    def render_interactive(self, width=800, height=600, style='cartoon'):
        """
        Create an interactive 3D visualization of the protein.
        
        Parameters
        ----------
        width : int, optional
            Width of the viewer
        height : int, optional
            Height of the viewer
        style : str, optional
            Visualization style
            
        Returns
        -------
        object
            Interactive 3D viewer
        """
        if HAS_PY3DMOL:
            return self.structure_viz.render_py3dmol(width, height, style='stick')
        else:
            logger.warning("py3Dmol not available, falling back to static rendering")
            fig, ax = self.render_protein(style=style)
            return fig


# Utility functions

def plot_multiple_energies(energy_data_list, labels=None, components=None, 
                          start_time=None, end_time=None, rolling_avg=None):
    """
    Plot multiple energy datasets together for comparison.
    
    Parameters
    ----------
    energy_data_list : list
        List of energy data dictionaries
    labels : list, optional
        Labels for each dataset
    components : list, optional
        Energy components to plot
    start_time : float, optional
        Start time for the plot
    end_time : float, optional
        End time for the plot
    rolling_avg : int, optional
        Window size for rolling average
        
    Returns
    -------
    tuple
        Tuple of (fig, ax)
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up labels
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(energy_data_list))]
    
    # Determine components to plot
    if components is None:
        # Use components from the first dataset
        if len(energy_data_list) > 0:
            components = list(energy_data_list[0].keys())
        else:
            components = []
    
    # Use different line styles for each dataset
    line_styles = ['-', '--', ':', '-.']
    
    # Plot each dataset
    for i, (energy_data, label) in enumerate(zip(energy_data_list, labels)):
        plotter = EnergyPlotter(energy_data)
        ls = line_styles[i % len(line_styles)]
        
        # Plot each component
        for component in components:
            if component in energy_data:
                time = np.array(plotter.time_data)
                values = np.array(energy_data[component])
                
                # Apply filters
                mask = np.ones_like(time, dtype=bool)
                if start_time is not None:
                    mask &= time >= start_time
                if end_time is not None:
                    mask &= time <= end_time
                
                time = time[mask]
                values = values[mask]
                
                # Apply rolling average if requested
                if rolling_avg is not None and rolling_avg > 1:
                    values = plotter._rolling_average(values, rolling_avg)
                    time = time[:len(values)]
                
                ax.plot(time, values, linestyle=ls, 
                       label=f"{label} - {component}")
    
    # Add labels and legend
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Energy (kJ/mol)')
    ax.set_title('Energy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def create_rmsd_plot(rmsd_data, reference_labels=None, target_labels=None, 
                   time_data=None, ax=None):
    """
    Create a plot of RMSD (Root Mean Square Deviation) data.
    
    Parameters
    ----------
    rmsd_data : array-like
        RMSD data with shape (n_references, n_targets, n_frames)
        or (n_targets, n_frames) if there's only one reference
    reference_labels : list, optional
        Labels for reference structures
    target_labels : list, optional
        Labels for target structures
    time_data : array-like, optional
        Time points for each frame
    ax : Axes, optional
        Matplotlib axis to plot on
        
    Returns
    -------
    tuple
        Tuple of (fig, ax)
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Handle different input shapes
    rmsd_data = np.atleast_3d(rmsd_data)
    n_references, n_targets, n_frames = rmsd_data.shape
    
    # Create time data if not provided
    if time_data is None:
        time_data = np.arange(n_frames)
    
    # Create labels if not provided
    if reference_labels is None:
        reference_labels = [f"Ref {i+1}" for i in range(n_references)]
    
    if target_labels is None:
        target_labels = [f"Target {i+1}" for i in range(n_targets)]
    
    # Use different line styles for references and colors for targets
    line_styles = ['-', '--', ':', '-.']
    
    # Plot RMSD data
    for i in range(n_references):
        for j in range(n_targets):
            label = f"{reference_labels[i]} vs {target_labels[j]}"
            ls = line_styles[i % len(line_styles)]
            
            ax.plot(time_data, rmsd_data[i, j], linestyle=ls, label=label)
    
    # Add labels and legend
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('RMSD (nm)')
    ax.set_title('Root Mean Square Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def create_rmsf_plot(rmsf_data, residue_indices=None, chain_labels=None, ax=None):
    """
    Create a plot of RMSF (Root Mean Square Fluctuation) data.
    
    Parameters
    ----------
    rmsf_data : array-like
        RMSF data with shape (n_chains, n_residues)
        or (n_residues,) if there's only one chain
    residue_indices : array-like, optional
        Residue indices for the x-axis
    chain_labels : list, optional
        Labels for each chain
    ax : Axes, optional
        Matplotlib axis to plot on
        
    Returns
    -------
    tuple
        Tuple of (fig, ax)
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure
    
    # Handle different input shapes
    rmsf_data = np.atleast_2d(rmsf_data)
    n_chains, n_residues = rmsf_data.shape
    
    # Create residue indices if not provided
    if residue_indices is None:
        residue_indices = np.arange(1, n_residues + 1)
    
    # Create chain labels if not provided
    if chain_labels is None:
        chain_labels = [f"Chain {i+1}" for i in range(n_chains)]
    
    # Plot RMSF data
    for i in range(n_chains):
        ax.plot(residue_indices, rmsf_data[i], label=chain_labels[i])
        
        # Add filled area under the curve
        ax.fill_between(residue_indices, 0, rmsf_data[i], alpha=0.2)
    
    # Add labels and legend
    ax.set_xlabel('Residue Number')
    ax.set_ylabel('RMSF (nm)')
    ax.set_title('Root Mean Square Fluctuation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def create_contact_map(contact_map, residue_labels=None, colormap='viridis', 
                      threshold=None, ax=None):
    """
    Create a contact map visualization.
    
    Parameters
    ----------
    contact_map : array-like
        Contact map matrix with shape (n_residues, n_residues)
    residue_labels : list, optional
        Labels for each residue
    colormap : str, optional
        Matplotlib colormap to use
    threshold : float, optional
        Threshold for contacts (values below will be white)
    ax : Axes, optional
        Matplotlib axis to plot on
        
    Returns
    -------
    tuple
        Tuple of (fig, ax)
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    # Apply threshold if specified
    if threshold is not None:
        masked_map = np.ma.masked_where(contact_map < threshold, contact_map)
        im = ax.imshow(masked_map, cmap=colormap, interpolation='nearest')
    else:
        im = ax.imshow(contact_map, cmap=colormap, interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Contact Probability')
    
    # Set tick labels if provided
    if residue_labels is not None:
        # If too many labels, thin them out
        if len(residue_labels) > 40:
            stride = len(residue_labels) // 20
            ax.set_xticks(np.arange(0, len(residue_labels), stride))
            ax.set_yticks(np.arange(0, len(residue_labels), stride))
            ax.set_xticklabels([residue_labels[i] for i in range(0, len(residue_labels), stride)])
            ax.set_yticklabels([residue_labels[i] for i in range(0, len(residue_labels), stride)])
        else:
            ax.set_xticks(np.arange(len(residue_labels)))
            ax.set_yticks(np.arange(len(residue_labels)))
            ax.set_xticklabels(residue_labels)
            ax.set_yticklabels(residue_labels)
    
    # Rotate x tick labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add title and labels
    ax.set_title('Residue Contact Map')
    ax.set_xlabel('Residue')
    ax.set_ylabel('Residue')
    
    return fig, ax


def create_distance_plot(distances, labels=None, time_data=None, ax=None):
    """
    Create a plot of distances between pairs of residues or atoms.
    
    Parameters
    ----------
    distances : array-like
        Distances with shape (n_pairs, n_frames)
    labels : list, optional
        Labels for each pair
    time_data : array-like, optional
        Time points for each frame
    ax : Axes, optional
        Matplotlib axis to plot on
        
    Returns
    -------
    tuple
        Tuple of (fig, ax)
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Handle different input shapes
    distances = np.atleast_2d(distances)
    n_pairs, n_frames = distances.shape
    
    # Create time data if not provided
    if time_data is None:
        time_data = np.arange(n_frames)
    
    # Create labels if not provided
    if labels is None:
        labels = [f"Pair {i+1}" for i in range(n_pairs)]
    
    # Plot distance data
    for i in range(n_pairs):
        ax.plot(time_data, distances[i], label=labels[i])
    
    # Add labels and legend
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Distance (nm)')
    ax.set_title('Distances')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax

class MatplotlibTrajectoryVisualizer(TrajectoryVisualizer):
    """
    Visualize MD trajectories using Matplotlib.
    """
    
    def __init__(self, trajectory_data: List[np.ndarray], topology=None, box_vectors=None):
        """
        Initialize a trajectory visualizer using Matplotlib.
        
        Parameters
        ----------
        trajectory_data : list of np.ndarray
            List of position arrays, each with shape (n_particles, 3)
        topology : object, optional
            Topology information for the system
        box_vectors : np.ndarray, optional
            Periodic box vectors
        """
        super().__init__(trajectory_data, topology, box_vectors)
        
        # Default atom colors based on element type or index
        self.atom_colors = self._generate_default_colors()
        
    def _generate_default_colors(self):
        """Generate default colors for atoms based on index or element type if available."""
        colors = []
        
        # Default color palette
        default_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Element-based colors (if topology with element information is available)
        element_colors = {
            'H': '#FFFFFF',   # White
            'C': '#808080',   # Gray
            'N': '#0000FF',   # Blue
            'O': '#FF0000',   # Red
            'S': '#FFFF00',   # Yellow
            'P': '#FFA500',   # Orange
            'F': '#90EE90',   # Light green
            'Cl': '#00FF00',  # Green
            'Br': '#A52A2A',  # Brown
            'I': '#800080',   # Purple
            'He': '#D3D3D3',  # Light gray
            'Ne': '#D3D3D3',  # Light gray
            'Ar': '#D3D3D3',  # Light gray
            'Xe': '#D3D3D3',  # Light gray
            'K': '#8F40D4',   # Purple
            'Na': '#0000FF',  # Blue
            'Ca': '#808090',  # Gray
            'Mg': '#228B22',  # Green
            'Fe': '#FFA500',  # Orange
            'Zn': '#A9A9A9',  # Dark gray
        }
        
        # If we have element information in the topology, use it
        if hasattr(self, 'topology') and self.topology is not None and hasattr(self.topology, 'elements'):
            for element in self.topology.elements:
                if element in element_colors:
                    colors.append(element_colors[element])
                else:
                    # If element not in our dictionary, choose a color from the palette
                    idx = len(colors) % len(default_palette)
                    colors.append(default_palette[idx])
        else:
            # If no topology with elements, use a palette based on atom index
            for i in range(self.n_particles):
                idx = i % len(default_palette)
                colors.append(default_palette[idx])
                
        return colors
    
    def render_frame(self, frame_index: int = 0, figsize=(10, 8), elev=30, azim=45,
                    show_box=True, show_bonds=True, atom_scale=50):
        """
        Render a single frame from the trajectory in 3D.
        
        Parameters
        ----------
        frame_index : int, optional
            Index of the frame to render
        figsize : tuple, optional
            Figure size
        elev : float, optional
            Elevation angle for 3D view
        azim : float, optional
            Azimuth angle for 3D view
        show_box : bool, optional
            Whether to show the simulation box
        show_bonds : bool, optional
            Whether to show bonds
        atom_scale : float, optional
            Scaling factor for atom sizes
            
        Returns
        -------
        fig, ax
            Matplotlib figure and axis objects
        """
        # Bounds checking
        if frame_index < 0 or frame_index >= self.n_frames:
            raise ValueError(f"Frame index {frame_index} out of range (0-{self.n_frames-1})")
            
        # Get the positions for the selected frame
        positions = self.trajectory_data[frame_index]
        
        # Create a 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot atoms as spheres (using scatter in matplotlib)
        # Size is proportional to atomic radius (if available) or a default value
        sizes = np.ones(self.n_particles) * atom_scale
        
        # Plot atoms
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c=self.atom_colors, s=sizes, alpha=0.8, edgecolors='black')
        
        # Plot bonds if requested and if we have bond information
        if show_bonds and hasattr(self, 'topology') and self.topology is not None and hasattr(self.topology, 'bonds'):
            for bond in self.topology.bonds:
                i, j = bond[0], bond[1]
                ax.plot([positions[i, 0], positions[j, 0]],
                       [positions[i, 1], positions[j, 1]],
                       [positions[i, 2], positions[j, 2]], 'k-', alpha=0.6, linewidth=1)
        
        # Show the simulation box if requested and available
        if show_box and self.box_vectors is not None:
            box = self.box_vectors
            
            # Define the vertices of the box
            r = [
                [0, 0, 0],
                [box[0], 0, 0],
                [0, box[1], 0],
                [0, 0, box[2]],
                [box[0], box[1], 0],
                [box[0], 0, box[2]],
                [0, box[1], box[2]],
                [box[0], box[1], box[2]]
            ]
            
            # Define the edges of the box
            edges = [
                [0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6],
                [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]
            ]
            
            # Plot the edges
            for edge in edges:
                ax.plot([r[edge[0]][0], r[edge[1]][0]],
                       [r[edge[0]][1], r[edge[1]][1]],
                       [r[edge[0]][2], r[edge[1]][2]], 'k-', alpha=0.3, linewidth=1)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Set a nicer viewpoint
        ax.view_init(elev=elev, azim=azim)
        
        # Set labels
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Z (nm)')
        ax.set_title(f'Frame {frame_index} of Molecular Dynamics Trajectory')
        
        plt.tight_layout()
        return fig, ax
        
    def animate(self, output_file: Optional[str] = None, start_frame: int = 0, 
               end_frame: Optional[int] = None, stride: int = 1,
               fps: int = 10, figsize=(10, 8), elev=30, azim=45,
               show_box=True, show_bonds=True, atom_scale=50):
        """
        Create an animation of the trajectory using Matplotlib.
        
        Parameters
        ----------
        output_file : str, optional
            Path to save the animation to (mp4, gif, etc.)
        start_frame : int, optional
            First frame to include
        end_frame : int, optional
            Last frame to include (exclusive)
        stride : int, optional
            Stride between frames
        fps : int, optional
            Frames per second in the animation
        figsize : tuple, optional
            Figure size
        elev : float, optional
            Elevation angle for 3D view
        azim : float, optional
            Azimuth angle for 3D view
        show_box : bool, optional
            Whether to show the simulation box
        show_bonds : bool, optional
            Whether to show bonds
        atom_scale : float, optional
            Scaling factor for atom sizes
            
        Returns
        -------
        Animation
            Animation object
        """
        # Handle default for end_frame
        if end_frame is None:
            end_frame = self.n_frames
            
        # Bounds checking
        start_frame = max(0, min(start_frame, self.n_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, self.n_frames))
        
        # Initialize the figure and axis
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set the viewpoint
        ax.view_init(elev=elev, azim=azim)
        
        # Function to update the plot for each frame
        def update(frame_idx):
            ax.clear()
            actual_frame = start_frame + frame_idx * stride
            positions = self.trajectory_data[actual_frame]
            
            # Plot atoms
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                      c=self.atom_colors, s=np.ones(self.n_particles) * atom_scale, 
                      alpha=0.8, edgecolors='black')
            
            # Plot bonds if requested
            if show_bonds and hasattr(self, 'topology') and self.topology is not None and hasattr(self.topology, 'bonds'):
                for bond in self.topology.bonds:
                    i, j = bond[0], bond[1]
                    ax.plot([positions[i, 0], positions[j, 0]],
                           [positions[i, 1], positions[j, 1]],
                           [positions[i, 2], positions[j, 2]], 'k-', alpha=0.6, linewidth=1)
            
            # Show box if requested
            if show_box and self.box_vectors is not None:
                box = self.box_vectors
                r = [
                    [0, 0, 0],
                    [box[0], 0, 0],
                    [0, box[1], 0],
                    [0, 0, box[2]],
                    [box[0], box[1], 0],
                    [box[0], 0, box[2]],
                    [0, box[1], box[2]],
                    [box[0], box[1], box[2]]
                ]
                edges = [
                    [0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6],
                    [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]
                ]
                for edge in edges:
                    ax.plot([r[edge[0]][0], r[edge[1]][0]],
                           [r[edge[0]][1], r[edge[1]][1]],
                           [r[edge[0]][2], r[edge[1]][2]], 'k-', alpha=0.3, linewidth=1)
            
            # Set equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
            
            # Set labels
            ax.set_xlabel('X (nm)')
            ax.set_ylabel('Y (nm)')
            ax.set_zlabel('Z (nm)')
            ax.set_title(f'Frame {actual_frame} of Molecular Dynamics Trajectory')
            
            # Set viewpoint (needs to be reset for each frame)
            ax.view_init(elev=elev, azim=azim)
            
            return ax
        
        # Create the animation
        frames = (end_frame - start_frame) // stride
        anim = FuncAnimation(fig, update, frames=frames, blit=False)
        
        # Save if output_file is provided
        if output_file is not None:
            writer = 'ffmpeg' if output_file.endswith('.mp4') else 'imagemagick'
            anim.save(output_file, writer=writer, fps=fps)
            logger.info(f"Animation saved to {output_file}")
        
        plt.close(fig)
        return anim
    
    def plot_distance_over_time(self, atom_i: int, atom_j: int, figsize=(10, 6)):
        """
        Plot the distance between two atoms over time.
        
        Parameters
        ----------
        atom_i : int
            Index of the first atom
        atom_j : int
            Index of the second atom
        figsize : tuple, optional
            Figure size
            
        Returns
        -------
        fig, ax
            Matplotlib figure and axis objects
        """
        # Bounds checking
        if atom_i < 0 or atom_i >= self.n_particles:
            raise ValueError(f"Atom index {atom_i} out of range (0-{self.n_particles-1})")
        if atom_j < 0 or atom_j >= self.n_particles:
            raise ValueError(f"Atom index {atom_j} out of range (0-{self.n_particles-1})")
            
        # Calculate distances over time
        distances = []
        for frame in self.trajectory_data:
            r_i = frame[atom_i]
            r_j = frame[atom_j]
            
            # Calculate vector from i to j
            rij = r_j - r_i
            
            # Apply minimum image convention if we have box vectors
            if self.box_vectors is not None:
                for dim in range(3):
                    if rij[dim] > self.box_vectors[dim] / 2:
                        rij[dim] -= self.box_vectors[dim]
                    elif rij[dim] < -self.box_vectors[dim] / 2:
                        rij[dim] += self.box_vectors[dim]
            
            # Calculate distance
            distance = np.linalg.norm(rij)
            distances.append(distance)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        time_points = np.arange(self.n_frames)
        ax.plot(time_points, distances, 'b-', lw=2)
        
        # Add labels and title
        ax.set_xlabel('Frame')
        ax.set_ylabel('Distance (nm)')
        ax.set_title(f'Distance between atoms {atom_i} and {atom_j} over time')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig, ax

def visualize_trajectory(trajectory_file, topology=None, renderer='matplotlib',
                    start_frame=0, end_frame=None, stride=1, output_file=None):
    """
    High-level function to visualize a molecular dynamics trajectory.
    
    Parameters
    ----------
    trajectory_file : str
        Path to the trajectory file (.npz, .dcd, etc.)
    topology : object, optional
        Topology information for the system
    renderer : str, optional
        Visualization backend to use ('matplotlib', 'nglview', 'py3dmol')
    start_frame : int, optional
        First frame to include
    end_frame : int, optional
        Last frame to include (exclusive)
    stride : int, optional
        Stride between frames
    output_file : str, optional
        Path to save visualization to
        
    Returns
    -------
    visualizer
        The visualizer object used
    """
    # Load trajectory data
    trajectory_data = load_trajectory(trajectory_file)
    
    # Get box vectors if available
    box_vectors = get_box_vectors(trajectory_file)
    
    # Create appropriate visualizer
    if renderer.lower() == 'matplotlib':
        visualizer = MatplotlibTrajectoryVisualizer(trajectory_data, topology, box_vectors)
    elif renderer.lower() == 'nglview' and HAS_NGLVIEW:
        raise NotImplementedError("NGLView support not yet implemented")
    elif renderer.lower() == 'py3dmol' and HAS_PY3DMOL:
        raise NotImplementedError("py3Dmol support not yet implemented")
    else:
        logger.warning(f"Requested renderer '{renderer}' not available, falling back to Matplotlib")
        visualizer = MatplotlibTrajectoryVisualizer(trajectory_data, topology, box_vectors)
    
    # If output file is specified, generate and save visualization
    if output_file is not None:
        if output_file.endswith(('.mp4', '.gif')):
            visualizer.animate(output_file=output_file, start_frame=start_frame,
                              end_frame=end_frame, stride=stride)
        elif output_file.endswith(('.png', '.jpg', '.pdf')):
            fig, ax = visualizer.render_frame(frame_index=start_frame)
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    return visualizer

def load_trajectory(trajectory_file):
    """
    Load trajectory data from a file.
    
    Parameters
    ----------
    trajectory_file : str
        Path to the trajectory file
        
    Returns
    -------
    list
        List of position arrays, each with shape (n_particles, 3)
    """
    if trajectory_file.endswith('.npz'):
        # Load from NumPy compressed archive
        data = np.load(trajectory_file)
        
        # Check if the file contains a 'trajectory' array
        if 'trajectory' in data:
            return data['trajectory']
        
        # Otherwise, find the first array with the right shape
        for key in data:
            if len(data[key].shape) == 3 and data[key].shape[2] == 3:
                return data[key]
                
        raise ValueError(f"Could not find trajectory data in {trajectory_file}")
        
    else:
        # For other formats, we could implement more loaders
        # For example, for .dcd, .xtc, etc.
        raise ValueError(f"Unsupported trajectory format: {trajectory_file}")

def get_box_vectors(trajectory_file):
    """
    Extract box vectors from a trajectory file if available.
    
    Parameters
    ----------
    trajectory_file : str
        Path to the trajectory file
        
    Returns
    -------
    np.ndarray or None
        Box vectors if available, otherwise None
    """
    if trajectory_file.endswith('.npz'):
        data = np.load(trajectory_file)
        
        # Check for box vectors in the file
        if 'box_vectors' in data:
            return data['box_vectors']
        elif 'box' in data:
            return data['box']
        elif 'box_dimensions' in data:
            return data['box_dimensions']
    
    return None

def plot_energy(energy_data, figsize=(12, 8), output_file=None):
    """
    Plot energy components over time.
    
    Parameters
    ----------
    energy_data : dict
        Dictionary containing energy components (kinetic, potential, total)
    figsize : tuple, optional
        Figure size
    output_file : str, optional
        Path to save the plot to
        
    Returns
    -------
    fig, ax
        Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot energy components
    time_points = np.arange(len(energy_data['total']))
    
    if 'kinetic' in energy_data:
        ax.plot(time_points, energy_data['kinetic'], 'r-', label='Kinetic Energy')
    
    if 'potential' in energy_data:
        ax.plot(time_points, energy_data['potential'], 'b-', label='Potential Energy')
    
    if 'total' in energy_data:
        ax.plot(time_points, energy_data['total'], 'g-', label='Total Energy', linewidth=2)
    
    # Add legend, labels, title
    ax.legend(fontsize=12)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Energy (kJ/mol)', fontsize=12)
    ax.set_title('Energy Components over Time', fontsize=14)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save if output_file is provided
    if output_file is not None:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Energy plot saved to {output_file}")
    
    return fig, ax

def plot_temperature(temperature_data, target_temperature=None, figsize=(12, 6), output_file=None):
    """
    Plot temperature over time.
    
    Parameters
    ----------
    temperature_data : array-like
        Temperature values over time
    target_temperature : float, optional
        Target temperature (for reference line)
    figsize : tuple, optional
        Figure size
    output_file : str, optional
        Path to save the plot to
        
    Returns
    -------
    fig, ax
        Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot temperature
    time_points = np.arange(len(temperature_data))
    ax.plot(time_points, temperature_data, 'b-', label='System Temperature')
    
    # Add target temperature line if provided
    if target_temperature is not None:
        ax.axhline(y=target_temperature, color='r', linestyle='--', 
                  label=f'Target Temperature ({target_temperature} K)')
    
    # Add legend, labels, title
    ax.legend(fontsize=12)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title('System Temperature over Time', fontsize=14)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save if output_file is provided
    if output_file is not None:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Temperature plot saved to {output_file}")
    
    return fig, ax
