"""
Real-time simulation viewer for molecular dynamics.

This module provides live visualization capabilities during simulation execution,
allowing users to watch the simulation unfold in real-time with performance optimization.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
import time
import logging
from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path

# Set up logger first
logger = logging.getLogger(__name__)

# Import from proteinMD
try:
    from ..core.simulation import MolecularDynamicsSimulation
except ImportError:
    from proteinMD.core.simulation import MolecularDynamicsSimulation

# Import visualization classes - avoid circular import
try:
    from .protein_3d import Protein3DVisualizer
except ImportError:
    try:
        from proteinMD.visualization.protein_3d import Protein3DVisualizer
    except ImportError:
        logger.warning("Could not import Protein3DVisualizer, using fallback")
        Protein3DVisualizer = None

# Define TrajectoryVisualization directly here to avoid circular import
class TrajectoryVisualization:
    """Simplified trajectory visualization for real-time viewer."""
    def __init__(self):
        self.name = "Trajectory Visualization"
    
    def animate_trajectory(self, *args, **kwargs):
        logger.info("TrajectoryVisualization.animate_trajectory called")
        return None


class RealTimeViewer:
    """
    Real-time simulation viewer for molecular dynamics simulations.
    
    This class provides live visualization during simulation execution with:
    - Live visualization every Nth step (configurable)
    - Performance optimization to maintain >80% simulation speed
    - Toggle functionality for live display on/off
    - Multi-panel dashboard with 3D view, energy plots, and statistics
    - Performance monitoring with FPS tracking
    
    Features:
    - 3D molecular visualization with multiple display modes
    - Real-time energy and temperature monitoring
    - Performance statistics display
    - Configurable visualization frequency
    - Non-blocking simulation execution
    """
    
    def __init__(self, 
                 simulation: MolecularDynamicsSimulation,
                 visualization_frequency: int = 10,
                 max_history_length: int = 1000,
                 display_mode: str = 'ball_stick',
                 figure_size: Tuple[float, float] = (16, 10)):
        """
        Initialize the real-time viewer.
        
        Parameters
        ----------
        simulation : MolecularDynamicsSimulation
            The simulation to visualize
        visualization_frequency : int
            Visualize every Nth simulation step (default: 10)
        max_history_length : int
            Maximum number of data points to keep in plots
        display_mode : str
            Initial display mode ('ball_stick', 'cartoon', 'surface')
        figure_size : tuple
            Figure size in inches (width, height)
        """
        self.simulation = simulation
        self.visualization_frequency = visualization_frequency
        self.max_history_length = max_history_length
        self.display_mode = display_mode
        self.figure_size = figure_size
        
        # Visualization state
        self.live_visualization_enabled = True
        self.visualization_paused = False
        self.step_counter = 0
        
        # Performance tracking
        self.performance_data = {
            'simulation_fps': [],
            'visualization_fps': [],
            'total_fps': [],
            'time_stamps': [],
            'efficiency': []  # Ratio of simulation speed with/without visualization
        }
        self.baseline_fps = None  # FPS without visualization
        self.last_fps_update = time.time()
        self.fps_update_interval = 1.0  # Update FPS every second
        
        # Data storage for plots
        self.plot_data = {
            'time': [],
            'kinetic_energy': [],
            'potential_energy': [],
            'total_energy': [],
            'temperature': [],
            'step_numbers': []
        }
        
        # Visualization components
        self.fig = None
        self.axes = {}
        self.plot_elements = {}
        self.animation = None
        self.protein_visualizer = None
        
        # Create trajectory visualization for fallback
        self.trajectory_viz = TrajectoryVisualization()
        
        logger.info(f"Initialized RealTimeViewer with frequency={visualization_frequency}")
    
    def setup_visualization(self) -> Dict:
        """
        Set up the visualization dashboard with multiple panels.
        
        Returns
        -------
        dict
            Dictionary containing all visualization elements
        """
        # Create figure with grid layout
        self.fig = plt.figure(figsize=self.figure_size)
        gs = gridspec.GridSpec(3, 3, 
                              height_ratios=[2, 1, 1], 
                              width_ratios=[2, 1, 1],
                              hspace=0.3, wspace=0.3)
        
        # 3D molecular view (main panel)
        self.axes['3d'] = self.fig.add_subplot(gs[0, :2], projection='3d')
        self._setup_3d_view()
        
        # Control panel
        self.axes['controls'] = self.fig.add_subplot(gs[0, 2])
        self._setup_control_panel()
        
        # Energy plot
        self.axes['energy'] = self.fig.add_subplot(gs[1, :2])
        self._setup_energy_plot()
        
        # Temperature plot
        self.axes['temperature'] = self.fig.add_subplot(gs[1, 2])
        self._setup_temperature_plot()
        
        # Performance monitor
        self.axes['performance'] = self.fig.add_subplot(gs[2, :2])
        self._setup_performance_plot()
        
        # Statistics panel
        self.axes['stats'] = self.fig.add_subplot(gs[2, 2])
        self._setup_statistics_panel()
        
        # Set up protein visualizer if positions are available (optional)
        # Note: Protein3DVisualizer requires protein structure, so we skip it for basic MD visualization
        # self.protein_visualizer = Protein3DVisualizer() if hasattr(self.simulation, 'positions') and self.simulation.positions is not None else None
            
        plt.tight_layout()
        
        return {
            'fig': self.fig,
            'axes': self.axes,
            'plot_elements': self.plot_elements
        }
    
    def _setup_3d_view(self):
        """Set up the 3D molecular visualization panel."""
        ax = self.axes['3d']
        
        # Set box dimensions if available
        if hasattr(self.simulation, 'box_dimensions'):
            box = self.simulation.box_dimensions
            ax.set_xlim(0, box[0])
            ax.set_ylim(0, box[1])
            ax.set_zlim(0, box[2])
        else:
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_zlim(0, 10)
        
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Z (nm)')
        ax.set_title('Molecular Dynamics Simulation - Real Time')
        
        # Initialize empty scatter plot
        if hasattr(self.simulation, 'positions') and self.simulation.positions is not None:
            positions = self.simulation.positions
            colors = self._get_particle_colors()
            sizes = self._get_particle_sizes()
            
            self.plot_elements['scatter'] = ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c=colors, s=sizes, alpha=0.8
            )
            
            # Add velocity vectors if available
            if hasattr(self.simulation, 'velocities') and self.simulation.velocities is not None:
                self.plot_elements['quiver'] = ax.quiver(
                    positions[:, 0], positions[:, 1], positions[:, 2],
                    self.simulation.velocities[:, 0], 
                    self.simulation.velocities[:, 1], 
                    self.simulation.velocities[:, 2],
                    color='gray', alpha=0.6, length=0.1
                )
        else:
            # Empty scatter plot
            self.plot_elements['scatter'] = ax.scatter([], [], [], c=[], s=[], alpha=0.8)
        
        # Add bonds if available
        if hasattr(self.simulation, 'bonds') and self.simulation.bonds:
            self._draw_bonds()
    
    def _setup_control_panel(self):
        """Set up the control panel with buttons and status."""
        ax = self.axes['controls']
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Controls & Status')
        
        # Status text
        self.plot_elements['status_text'] = ax.text(
            0.05, 0.9, "Simulation: Ready", 
            fontsize=10, weight='bold',
            transform=ax.transAxes
        )
        
        # Performance text
        self.plot_elements['performance_text'] = ax.text(
            0.05, 0.7, "FPS: --\nEfficiency: --%", 
            fontsize=9,
            transform=ax.transAxes
        )
        
        # Control instructions
        control_text = (
            "Controls:\n"
            "• Press 'p' to pause/resume\n" 
            "• Press 'v' to toggle visualization\n"
            "• Press 's' to save frame\n"
            "• Press 'r' to reset view"
        )
        ax.text(0.05, 0.4, control_text, fontsize=8, transform=ax.transAxes)
        
        # Visualization frequency display
        freq_text = f"Viz Frequency: Every {self.visualization_frequency} steps"
        ax.text(0.05, 0.1, freq_text, fontsize=8, transform=ax.transAxes)
    
    def _setup_energy_plot(self):
        """Set up the energy monitoring plot."""
        ax = self.axes['energy']
        ax.set_title('System Energy')
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Energy (kJ/mol)')
        
        # Initialize empty lines
        self.plot_elements['energy_total'], = ax.plot([], [], 'g-', label='Total', linewidth=2)
        self.plot_elements['energy_kinetic'], = ax.plot([], [], 'r-', label='Kinetic', linewidth=1)
        self.plot_elements['energy_potential'], = ax.plot([], [], 'b-', label='Potential', linewidth=1)
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _setup_temperature_plot(self):
        """Set up the temperature monitoring plot."""
        ax = self.axes['temperature']
        ax.set_title('Temperature')
        ax.set_xlabel('Step')
        ax.set_ylabel('T (K)')
        
        # Initialize empty line
        self.plot_elements['temperature_line'], = ax.plot([], [], 'r-', linewidth=2)
        
        # Add target temperature line if available
        if hasattr(self.simulation, 'temperature'):
            ax.axhline(y=self.simulation.temperature, color='k', 
                      linestyle='--', alpha=0.5, label='Target')
            ax.legend()
        
        ax.grid(True, alpha=0.3)
    
    def _setup_performance_plot(self):
        """Set up the performance monitoring plot."""
        ax = self.axes['performance']
        ax.set_title('Performance Monitor')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('FPS')
        
        # Initialize empty lines
        self.plot_elements['sim_fps'], = ax.plot([], [], 'g-', label='Simulation FPS', linewidth=2)
        self.plot_elements['viz_fps'], = ax.plot([], [], 'b-', label='Visualization FPS', linewidth=1)
        self.plot_elements['total_fps'], = ax.plot([], [], 'r-', label='Total FPS', linewidth=1)
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _setup_statistics_panel(self):
        """Set up the statistics display panel."""
        ax = self.axes['stats']
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Statistics')
        
        # Statistics text
        self.plot_elements['stats_text'] = ax.text(
            0.05, 0.9, "Statistics:\n", 
            fontsize=9,
            transform=ax.transAxes,
            verticalalignment='top'
        )
    
    def _get_particle_colors(self):
        """Get colors for particles based on type/charge."""
        if hasattr(self.simulation, 'charges') and self.simulation.charges is not None and len(self.simulation.charges) > 0:
            # Color by charge: red for negative, blue for positive, gray for neutral
            colors = []
            for charge in self.simulation.charges:
                if charge > 0.1:
                    colors.append('blue')
                elif charge < -0.1:
                    colors.append('red')
                else:
                    colors.append('gray')
            return colors
        else:
            # Default colors - get number of particles
            if hasattr(self.simulation, 'num_particles') and self.simulation.num_particles > 0:
                num_particles = self.simulation.num_particles
            elif hasattr(self.simulation, 'positions') and self.simulation.positions is not None:
                num_particles = len(self.simulation.positions)
            else:
                num_particles = 1  # Default fallback
                
            return ['blue'] * num_particles
    
    def _get_particle_sizes(self):
        """Get sizes for particles based on mass/type."""
        if hasattr(self.simulation, 'masses') and self.simulation.masses is not None and len(self.simulation.masses) > 0:
            # Size proportional to mass
            base_size = 50
            masses = np.array(self.simulation.masses)
            
            if len(masses) > 0:
                max_mass = np.max(masses)
                min_mass = np.min(masses)
                
                if max_mass > min_mass:
                    normalized_masses = (masses - min_mass) / (max_mass - min_mass)
                    sizes = base_size + 50 * normalized_masses
                else:
                    sizes = [base_size] * len(masses)
                
                return sizes.tolist()
        
        # Default size - get number of particles
        if hasattr(self.simulation, 'num_particles') and self.simulation.num_particles > 0:
            num_particles = self.simulation.num_particles
        elif hasattr(self.simulation, 'positions') and self.simulation.positions is not None:
            num_particles = len(self.simulation.positions)
        else:
            num_particles = 1  # Default fallback
            
        return [50] * num_particles
    
    def _draw_bonds(self):
        """Draw bonds between particles."""
        if not hasattr(self.simulation, 'bonds') or not self.simulation.bonds:
            return
            
        ax = self.axes['3d']
        positions = self.simulation.positions
        
        for bond in self.simulation.bonds:
            i, j = bond[0], bond[1]
            if i < len(positions) and j < len(positions):
                line_data = [
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    [positions[i, 2], positions[j, 2]]
                ]
                ax.plot3D(*line_data, 'k-', alpha=0.6, linewidth=1)
    
    def update_visualization(self, frame_data: Dict):
        """
        Update all visualization elements with new simulation data.
        
        Parameters
        ----------
        frame_data : dict
            Dictionary containing current simulation state
        """
        if not self.live_visualization_enabled or self.visualization_paused:
            return
        
        # Store time data for tracking
        if 'time' in frame_data:
            self.plot_data['time'].append(frame_data['time'])
        elif 'step' in frame_data:
            # Fallback: use step as time
            self.plot_data['time'].append(frame_data['step'])
        
        # Update 3D visualization
        if 'positions' in frame_data:
            self._update_3d_positions(frame_data['positions'])
        
        if 'velocities' in frame_data:
            self._update_velocity_vectors(frame_data['velocities'])
        
        # Update energy plots
        if 'energies' in frame_data:
            self._update_energy_plots(frame_data)
        
        # Update temperature plot
        if 'temperature' in frame_data:
            self._update_temperature_plot(frame_data)
        
        # Update performance plot
        self._update_performance_plots()
        
        # Update status and statistics
        self._update_status_display(frame_data)
        
        # Update statistics panel
        self._update_statistics_panel(frame_data)
        
        # Redraw if not using animation
        if self.animation is None:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    
    def _update_3d_positions(self, positions: np.ndarray):
        """Update 3D particle positions."""
        if 'scatter' in self.plot_elements and positions is not None:
            # Update scatter plot positions
            self.plot_elements['scatter']._offsets3d = (
                positions[:, 0], 
                positions[:, 1], 
                positions[:, 2]
            )
            
            # Redraw bonds if they exist
            if hasattr(self.simulation, 'bonds') and self.simulation.bonds:
                # Clear existing bond lines
                ax = self.axes['3d']
                for line in ax.lines:
                    line.remove()
                    
                # Redraw bonds with new positions
                for bond in self.simulation.bonds:
                    i, j = bond[0], bond[1]
                    if i < len(positions) and j < len(positions):
                        line_data = [
                            [positions[i, 0], positions[j, 0]],
                            [positions[i, 1], positions[j, 1]],
                            [positions[i, 2], positions[j, 2]]
                        ]
                        ax.plot3D(*line_data, 'k-', alpha=0.6, linewidth=1)
    
    def _update_velocity_vectors(self, velocities: np.ndarray):
        """Update velocity vectors display."""
        if 'quiver' in self.plot_elements and velocities is not None:
            # Remove old quiver plot
            self.plot_elements['quiver'].remove()
            
            # Create new quiver plot
            ax = self.axes['3d']
            positions = self.simulation.positions
            
            # Scale velocities for visualization
            scale_factor = 0.1  # Adjust as needed
            
            self.plot_elements['quiver'] = ax.quiver(
                positions[:, 0], positions[:, 1], positions[:, 2],
                velocities[:, 0] * scale_factor, 
                velocities[:, 1] * scale_factor, 
                velocities[:, 2] * scale_factor,
                color='gray', alpha=0.6
            )
    
    def _update_energy_plots(self, frame_data: Dict):
        """Update energy monitoring plots."""
        # Add new data point
        step = frame_data.get('step', len(self.plot_data['step_numbers']))
        energies = frame_data['energies']
        
        self.plot_data['step_numbers'].append(step)
        self.plot_data['kinetic_energy'].append(energies.get('kinetic', 0))
        self.plot_data['potential_energy'].append(energies.get('potential', 0))
        self.plot_data['total_energy'].append(energies.get('total', 0))
        
        # Limit history length
        if len(self.plot_data['step_numbers']) > self.max_history_length:
            for key in ['step_numbers', 'kinetic_energy', 'potential_energy', 'total_energy']:
                self.plot_data[key] = self.plot_data[key][-self.max_history_length:]
        
        # Update plot lines
        steps = self.plot_data['step_numbers']
        self.plot_elements['energy_total'].set_data(steps, self.plot_data['total_energy'])
        self.plot_elements['energy_kinetic'].set_data(steps, self.plot_data['kinetic_energy'])
        self.plot_elements['energy_potential'].set_data(steps, self.plot_data['potential_energy'])
        
        # Auto-scale axes
        ax = self.axes['energy']
        ax.relim()
        ax.autoscale_view()
    
    def _update_temperature_plot(self, frame_data: Dict):
        """Update temperature monitoring plot."""
        # Add new data point
        step = frame_data.get('step', len(self.plot_data['step_numbers']))
        temperature = frame_data['temperature']
        
        if 'temperature' not in self.plot_data:
            self.plot_data['temperature'] = []
            
        self.plot_data['temperature'].append(temperature)
        
        # Limit history length
        if len(self.plot_data['temperature']) > self.max_history_length:
            self.plot_data['temperature'] = self.plot_data['temperature'][-self.max_history_length:]
        
        # Update plot line
        steps = self.plot_data['step_numbers'][-len(self.plot_data['temperature']):]
        self.plot_elements['temperature_line'].set_data(steps, self.plot_data['temperature'])
        
        # Auto-scale axes
        ax = self.axes['temperature']
        ax.relim()
        ax.autoscale_view()
    
    def _update_performance_plots(self):
        """Update performance monitoring plots."""
        current_time = time.time()
        
        # Update performance data
        if hasattr(self.simulation, 'performance_stats'):
            sim_fps = self.simulation.performance_stats.get('fps', 0)
            self.performance_data['simulation_fps'].append(sim_fps)
            self.performance_data['time_stamps'].append(current_time)
            
            # Calculate visualization FPS
            viz_fps = 1.0 / max(0.001, current_time - self.last_fps_update)
            self.performance_data['visualization_fps'].append(viz_fps)
            
            # Calculate total FPS (effective simulation rate with visualization)
            total_fps = sim_fps * (self.visualization_frequency / 100.0)  # Approximate
            self.performance_data['total_fps'].append(total_fps)
            
            # Calculate efficiency
            if self.baseline_fps is not None:
                efficiency = (sim_fps / self.baseline_fps) * 100
            else:
                efficiency = 100
            self.performance_data['efficiency'].append(efficiency)
            
            # Limit history
            max_perf_history = 100
            for key in self.performance_data:
                if len(self.performance_data[key]) > max_perf_history:
                    self.performance_data[key] = self.performance_data[key][-max_perf_history:]
            
            # Update plot lines
            times = [t - self.performance_data['time_stamps'][0] for t in self.performance_data['time_stamps']]
            
            self.plot_elements['sim_fps'].set_data(times, self.performance_data['simulation_fps'])
            self.plot_elements['viz_fps'].set_data(times, self.performance_data['visualization_fps'])
            self.plot_elements['total_fps'].set_data(times, self.performance_data['total_fps'])
            
            # Auto-scale axes
            ax = self.axes['performance']
            ax.relim()
            ax.autoscale_view()
        
        self.last_fps_update = current_time
    
    def _update_status_display(self, frame_data: Dict):
        """Update status text display."""
        step = frame_data.get('step', 0)
        time_ps = frame_data.get('time', 0)
        
        status = f"Step: {step}\nTime: {time_ps:.2f} ps"
        if self.visualization_paused:
            status += "\nPAUSED"
        elif not self.live_visualization_enabled:
            status += "\nVIZ OFF"
        else:
            status += "\nRUNNING"
        
        self.plot_elements['status_text'].set_text(status)
        
        # Update performance text
        if hasattr(self.simulation, 'performance_stats'):
            fps = self.simulation.performance_stats.get('fps', 0)
            if self.performance_data['efficiency']:
                efficiency = self.performance_data['efficiency'][-1]
            else:
                efficiency = 100
            
            perf_text = f"FPS: {fps:.1f}\nEfficiency: {efficiency:.1f}%"
            self.plot_elements['performance_text'].set_text(perf_text)
    
    def _update_statistics_panel(self, frame_data: Dict):
        """Update statistics display panel."""
        stats_lines = ["Statistics:"]
        
        # Basic simulation info
        if 'step' in frame_data:
            stats_lines.append(f"Step: {frame_data['step']}")
        if 'time' in frame_data:
            stats_lines.append(f"Time: {frame_data['time']:.2f} ps")
        if 'temperature' in frame_data:
            stats_lines.append(f"Temp: {frame_data['temperature']:.1f} K")
            
        # Energy info
        if 'energies' in frame_data:
            energies = frame_data['energies']
            if 'total' in energies:
                stats_lines.append(f"E_total: {energies['total']:.2f} kJ/mol")
            if 'kinetic' in energies:
                stats_lines.append(f"E_kinetic: {energies['kinetic']:.2f} kJ/mol")
            if 'potential' in energies:
                stats_lines.append(f"E_potential: {energies['potential']:.2f} kJ/mol")
        
        # Performance info
        if self.performance_data['efficiency']:
            efficiency = self.performance_data['efficiency'][-1]
            stats_lines.append(f"Efficiency: {efficiency:.1f}%")
            
        # Visualization info
        stats_lines.append(f"Viz Freq: 1/{self.visualization_frequency}")
        stats_lines.append(f"Live Viz: {'ON' if self.live_visualization_enabled else 'OFF'}")
        
        stats_text = "\n".join(stats_lines)
        self.plot_elements['stats_text'].set_text(stats_text)
    
    def toggle_visualization(self):
        """Toggle live visualization on/off."""
        self.live_visualization_enabled = not self.live_visualization_enabled
        logger.info(f"Live visualization {'enabled' if self.live_visualization_enabled else 'disabled'}")
    
    def toggle_pause(self):
        """Toggle visualization pause."""
        self.visualization_paused = not self.visualization_paused
        logger.info(f"Visualization {'paused' if self.visualization_paused else 'resumed'}")
    
    def set_visualization_frequency(self, frequency: int):
        """
        Set how often to update the visualization.
        
        Parameters
        ----------
        frequency : int
            Update visualization every N simulation steps
        """
        if frequency > 0:
            self.visualization_frequency = frequency
            logger.info(f"Visualization frequency set to every {frequency} steps")
    
    def run_with_visualization(self, steps: int, callback: Optional[Callable] = None):
        """
        Run simulation with real-time visualization.
        
        Parameters
        ----------
        steps : int
            Number of simulation steps to run
        callback : callable, optional
            Additional callback function to call after each step
        """
        # Set up visualization if not already done
        if self.fig is None:
            self.setup_visualization()
        
        # Measure baseline FPS without visualization if not done yet
        if self.baseline_fps is None:
            self._measure_baseline_performance()
        
        logger.info(f"Starting real-time simulation for {steps} steps")
        start_time = time.time()
        
        # Custom callback that includes visualization
        def visualization_callback(simulation, state, step_idx):
            self.step_counter += 1
            
            # Update visualization every N steps
            if self.step_counter % self.visualization_frequency == 0:
                self.update_visualization(state)
            
            # Call user callback if provided
            if callback is not None:
                callback(simulation, state, step_idx)
        
        # Run simulation with visualization callback
        try:
            result = self.simulation.run(steps, callback=visualization_callback)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Calculate and log final performance
            final_fps = steps / elapsed if elapsed > 0 else 0
            if self.baseline_fps is not None:
                efficiency = (final_fps / self.baseline_fps) * 100
            else:
                efficiency = 100
                
            logger.info(f"Real-time simulation completed: {steps} steps in {elapsed:.2f}s")
            logger.info(f"Final performance: {final_fps:.2f} steps/s ({efficiency:.1f}% efficiency)")
            
            return result
            
        except KeyboardInterrupt:
            logger.info("Real-time simulation interrupted by user")
            return None
        except Exception as e:
            logger.error(f"Error in real-time simulation: {str(e)}")
            raise
    
    def _measure_baseline_performance(self):
        """Measure simulation performance without visualization."""
        logger.info("Measuring baseline performance...")
        
        # Temporarily disable visualization
        original_state = self.live_visualization_enabled
        self.live_visualization_enabled = False
        
        # Run a short simulation to measure baseline
        baseline_steps = min(100, self.simulation.step_count + 100)
        start_time = time.time()
        
        try:
            self.simulation.run(baseline_steps)
            elapsed = time.time() - start_time
            self.baseline_fps = baseline_steps / elapsed if elapsed > 0 else 0
            logger.info(f"Baseline performance: {self.baseline_fps:.2f} steps/s")
        except Exception as e:
            logger.warning(f"Could not measure baseline performance: {str(e)}")
            self.baseline_fps = 100  # Default assumption
        
        # Restore visualization state
        self.live_visualization_enabled = original_state
    
    def save_visualization_frame(self, filename: Optional[str] = None):
        """
        Save current visualization as an image.
        
        Parameters
        ----------
        filename : str, optional
            Output filename. If None, auto-generate based on current step.
        """
        if self.fig is None:
            logger.warning("No visualization to save")
            return
            
        if filename is None:
            step = getattr(self.simulation, 'step_count', 0)
            filename = f"realtime_viz_step_{step:06d}.png"
        
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization frame to {filepath}")
    
    def setup_keyboard_controls(self):
        """Set up keyboard controls for interactive use."""
        def on_key_press(event):
            if event.key == 'p':
                self.toggle_pause()
            elif event.key == 'v':
                self.toggle_visualization()
            elif event.key == 's':
                self.save_visualization_frame()
            elif event.key == 'r':
                # Reset 3D view
                ax = self.axes.get('3d')
                if ax is not None:
                    ax.view_init(elev=20, azim=45)
                    self.fig.canvas.draw()
            elif event.key == '1':
                self.set_visualization_frequency(1)
            elif event.key == '5':
                self.set_visualization_frequency(5) 
            elif event.key == 'escape':
                plt.close(self.fig)
        
        if self.fig is not None:
            self.fig.canvas.mpl_connect('key_press_event', on_key_press)
            logger.info("Keyboard controls enabled: p(pause), v(viz toggle), s(save), r(reset view)")
    
    def get_performance_summary(self) -> Dict:
        """
        Get performance summary statistics.
        
        Returns
        -------
        dict
            Performance summary with FPS, efficiency, etc.
        """
        if not self.performance_data['simulation_fps']:
            return {}
        
        summary = {
            'baseline_fps': self.baseline_fps,
            'avg_simulation_fps': np.mean(self.performance_data['simulation_fps']),
            'avg_visualization_fps': np.mean(self.performance_data['visualization_fps']),
            'avg_efficiency': np.mean(self.performance_data['efficiency']) if self.performance_data['efficiency'] else 100,
            'visualization_frequency': self.visualization_frequency,
            'total_frames_visualized': len(self.performance_data['simulation_fps']),
            'live_visualization_enabled': self.live_visualization_enabled
        }
        
        return summary
    
    def enable_visualization(self):
        """Enable live visualization."""
        self.live_visualization_enabled = True
        logger.info("Live visualization enabled")
    
    def disable_visualization(self):
        """Disable live visualization."""
        self.live_visualization_enabled = False
        logger.info("Live visualization disabled")
    
    def is_visualization_enabled(self) -> bool:
        """Check if visualization is enabled."""
        return self.live_visualization_enabled
    
    def update_display(self, frame_data: Dict = None):
        """Update the display with current simulation data."""
        if frame_data is None:
            frame_data = {}
        self.update_visualization(frame_data)
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics."""
        return self.get_performance_summary()
    
    def get_energy_history(self) -> Dict:
        """Get energy history data."""
        return {
            'time': self.plot_data['time'].copy(),
            'kinetic_energy': self.plot_data['kinetic_energy'].copy(),
            'potential_energy': self.plot_data['potential_energy'].copy(),
            'total_energy': self.plot_data['total_energy'].copy(),
            'temperature': self.plot_data['temperature'].copy()
        }
    
    def cleanup(self):
        """Clean up visualization resources."""
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None
        
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        
        logger.info("Real-time viewer cleaned up")


# Example usage and testing functions
def demo_realtime_viewer():
    """
    Demonstration of the real-time viewer with a simple simulation.
    """
    # Create a simple simulation for demonstration
    from ..core.simulation import MolecularDynamicsSimulation
    
    # Initialize simulation with a few particles
    sim = MolecularDynamicsSimulation(
        num_particles=10,
        box_dimensions=np.array([5.0, 5.0, 5.0]),
        temperature=300.0,
        time_step=0.002,
        cutoff_distance=2.0
    )
    
    # Add some particles
    positions = np.random.uniform(0, 5, (10, 3))
    masses = np.ones(10) * 16.0  # Oxygen-like mass
    charges = np.random.choice([-1, 0, 1], 10) * 0.1
    
    sim.add_particles(positions, masses=masses, charges=charges)
    sim.initialize_velocities(temperature=300)
    
    # Add some bonds for visualization
    bonds = [(0, 1, 1000, 0.15), (1, 2, 1000, 0.15), (2, 3, 1000, 0.15)]
    sim.add_bonds(bonds)
    
    # Create real-time viewer
    viewer = RealTimeViewer(
        simulation=sim,
        visualization_frequency=5,  # Visualize every 5 steps
        display_mode='ball_stick'
    )
    
    # Set up visualization
    viewer.setup_visualization()
    viewer.setup_keyboard_controls()
    
    # Run simulation with real-time visualization
    try:
        viewer.run_with_visualization(steps=1000)
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    
    # Show performance summary
    summary = viewer.get_performance_summary()
    print("\nPerformance Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    viewer.cleanup()


if __name__ == "__main__":
    # Run demonstration if executed directly
    demo_realtime_viewer()
