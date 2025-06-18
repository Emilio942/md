"""
Energy Plot Dashboard for real-time monitoring of MD simulation energetics.

This module provides a dedicated dashboard for visualizing energy, temperature,
and pressure evolution during molecular dynamics simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg
import threading
import time
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from pathlib import Path
from collections import deque
import io

# Configure logging
logger = logging.getLogger(__name__)


class EnergyPlotDashboard:
    """
    Real-time energy plot dashboard for MD simulations.
    
    Features:
    - Real-time plotting of kinetic, potential, and total energy
    - Temperature and pressure monitoring
    - Automatic updates during simulation
    - High-resolution export capabilities
    - Configurable plot appearance and layout
    """
    
    def __init__(self, max_points: int = 1000, update_interval: int = 100):
        """
        Initialize the Energy Plot Dashboard.
        
        Parameters
        ----------
        max_points : int, optional
            Maximum number of data points to display (default: 1000)
        update_interval : int, optional
            Update interval in milliseconds (default: 100)
        """
        self.max_points = max_points
        self.update_interval = update_interval
        
        # Data storage using deques for efficient append/pop operations
        self.time_data = deque(maxlen=max_points)
        self.kinetic_energy = deque(maxlen=max_points)
        self.potential_energy = deque(maxlen=max_points)
        self.total_energy = deque(maxlen=max_points)
        self.temperature = deque(maxlen=max_points)
        self.pressure = deque(maxlen=max_points)
        
        # Simulation interface
        self.simulation = None
        self.is_running = False
        self.update_thread = None
        
        # Matplotlib figure and axes
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.animation = None
        
        # Export settings
        self.export_dpi = 300
        self.export_format = 'png'
        
        logger.info(f"Initialized EnergyPlotDashboard with max_points={max_points}, update_interval={update_interval}ms")
    
    @property
    def energy_history(self):
        """Provide energy history for test compatibility."""
        return list(zip(self.time_data, self.kinetic_energy, self.potential_energy, self.total_energy))
    
    def setup_plots(self, figsize: Tuple[float, float] = (15, 10)) -> None:
        """
        Set up the matplotlib figure and subplots.
        
        Parameters
        ----------
        figsize : tuple of float, optional
            Figure size in inches (width, height) (default: (15, 10))
        """
        # Create figure with subplots
        self.fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
            3, 2, figsize=figsize, tight_layout=True
        )
        
        # Store axes references
        self.axes = {
            'energy_total': ax1,
            'energy_components': ax2,
            'temperature': ax3,
            'pressure': ax4,
            'energy_conservation': ax5,
            'statistics': ax6
        }
        
        # Configure energy plots
        self._setup_energy_plots()
        self._setup_thermodynamic_plots()
        self._setup_analysis_plots()
        
        # Set up interactive features
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        logger.info("Set up energy dashboard plots")
    
    def _setup_energy_plots(self) -> None:
        """Set up the energy-related subplots."""
        # Total energy plot
        ax = self.axes['energy_total']
        ax.set_title('Total Energy vs Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Total Energy (kJ/mol)')
        ax.grid(True, alpha=0.3)
        
        line_total, = ax.plot([], [], 'b-', linewidth=2, label='Total Energy')
        self.lines['total_energy'] = line_total
        ax.legend()
        
        # Energy components plot
        ax = self.axes['energy_components']
        ax.set_title('Energy Components vs Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Energy (kJ/mol)')
        ax.grid(True, alpha=0.3)
        
        line_kinetic, = ax.plot([], [], 'r-', linewidth=2, label='Kinetic Energy')
        line_potential, = ax.plot([], [], 'g-', linewidth=2, label='Potential Energy')
        
        self.lines['kinetic_energy'] = line_kinetic
        self.lines['potential_energy'] = line_potential
        ax.legend()
    
    def _setup_thermodynamic_plots(self) -> None:
        """Set up temperature and pressure plots."""
        # Temperature plot
        ax = self.axes['temperature']
        ax.set_title('Temperature vs Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Temperature (K)')
        ax.grid(True, alpha=0.3)
        
        line_temp, = ax.plot([], [], 'orange', linewidth=2, label='Temperature')
        self.lines['temperature'] = line_temp
        ax.legend()
        
        # Pressure plot
        ax = self.axes['pressure']
        ax.set_title('Pressure vs Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Pressure (bar)')
        ax.grid(True, alpha=0.3)
        
        line_pressure, = ax.plot([], [], 'purple', linewidth=2, label='Pressure')
        self.lines['pressure'] = line_pressure
        ax.legend()
    
    def _setup_analysis_plots(self) -> None:
        """Set up analysis and statistics plots."""
        # Energy conservation plot (drift analysis)
        ax = self.axes['energy_conservation']
        ax.set_title('Energy Conservation Analysis', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Energy Drift (kJ/mol)')
        ax.grid(True, alpha=0.3)
        
        line_drift, = ax.plot([], [], 'red', linewidth=1, alpha=0.7, label='Energy Drift')
        self.lines['energy_drift'] = line_drift
        ax.legend()
        
        # Statistics summary
        ax = self.axes['statistics']
        ax.set_title('Current Statistics', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Text for displaying statistics
        self.stats_text = ax.text(0.1, 0.9, '', transform=ax.transAxes, 
                                 fontsize=10, verticalalignment='top',
                                 fontfamily='monospace')
    
    def connect_simulation(self, simulation) -> None:
        """
        Connect the dashboard to a simulation object.
        
        Parameters
        ----------
        simulation : MolecularDynamicsSimulation
            The simulation object to monitor
        """
        self.simulation = simulation
        logger.info(f"Connected to simulation: {type(simulation).__name__}")
    
    def add_data_point(self, time_ps: float, kinetic: float, potential: float, 
                      temperature: float, pressure: float = None) -> None:
        """
        Add a new data point to the dashboard.
        
        Parameters
        ----------
        time_ps : float
            Simulation time in picoseconds
        kinetic : float
            Kinetic energy in kJ/mol
        potential : float
            Potential energy in kJ/mol
        temperature : float
            Temperature in Kelvin
        pressure : float, optional
            Pressure in bar
        """
        # Add data to deques
        self.time_data.append(time_ps)
        self.kinetic_energy.append(kinetic)
        self.potential_energy.append(potential)
        self.total_energy.append(kinetic + potential)
        self.temperature.append(temperature)
        
        if pressure is not None:
            self.pressure.append(pressure)
        else:
            self.pressure.append(0.0)  # Default for NVT simulations
    
    def update_plots(self) -> None:
        """Update all plots with current data."""
        if len(self.time_data) == 0:
            return
        
        # Convert deques to numpy arrays for plotting
        times = np.array(self.time_data)
        kinetic = np.array(self.kinetic_energy)
        potential = np.array(self.potential_energy)
        total = np.array(self.total_energy)
        temp = np.array(self.temperature)
        press = np.array(self.pressure)
        
        # Update energy plots
        self.lines['total_energy'].set_data(times, total)
        self.lines['kinetic_energy'].set_data(times, kinetic)
        self.lines['potential_energy'].set_data(times, potential)
        self.lines['temperature'].set_data(times, temp)
        self.lines['pressure'].set_data(times, press)
        
        # Calculate and plot energy drift
        if len(total) > 1:
            energy_drift = total - total[0]
            self.lines['energy_drift'].set_data(times, energy_drift)
        
        # Update axis limits
        self._update_axis_limits(times, kinetic, potential, total, temp, press)
        
        # Update statistics
        self._update_statistics(kinetic, potential, total, temp, press)
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _update_axis_limits(self, times: np.ndarray, kinetic: np.ndarray, 
                           potential: np.ndarray, total: np.ndarray,
                           temp: np.ndarray, press: np.ndarray) -> None:
        """Update axis limits based on current data."""
        if len(times) == 0:
            return
        
        time_margin = (times[-1] - times[0]) * 0.05 if len(times) > 1 else 1.0
        
        # Energy plots
        self.axes['energy_total'].set_xlim(times[0] - time_margin, times[-1] + time_margin)
        self.axes['energy_total'].set_ylim(np.min(total) * 1.01, np.max(total) * 1.01)
        
        energy_min = min(np.min(kinetic), np.min(potential))
        energy_max = max(np.max(kinetic), np.max(potential))
        energy_range = energy_max - energy_min
        
        self.axes['energy_components'].set_xlim(times[0] - time_margin, times[-1] + time_margin)
        self.axes['energy_components'].set_ylim(
            energy_min - energy_range * 0.05, 
            energy_max + energy_range * 0.05
        )
        
        # Temperature plot
        self.axes['temperature'].set_xlim(times[0] - time_margin, times[-1] + time_margin)
        temp_range = np.max(temp) - np.min(temp)
        self.axes['temperature'].set_ylim(
            np.min(temp) - temp_range * 0.05, 
            np.max(temp) + temp_range * 0.05
        )
        
        # Pressure plot
        self.axes['pressure'].set_xlim(times[0] - time_margin, times[-1] + time_margin)
        if np.max(press) > 0:  # Only if pressure data is meaningful
            press_range = np.max(press) - np.min(press)
            self.axes['pressure'].set_ylim(
                np.min(press) - press_range * 0.05, 
                np.max(press) + press_range * 0.05
            )
        
        # Energy drift plot
        self.axes['energy_conservation'].set_xlim(times[0] - time_margin, times[-1] + time_margin)
        if len(total) > 1:
            drift = total - total[0]
            drift_range = np.max(np.abs(drift))
            self.axes['energy_conservation'].set_ylim(-drift_range * 1.1, drift_range * 1.1)
    
    def _update_statistics(self, kinetic: np.ndarray, potential: np.ndarray, 
                          total: np.ndarray, temp: np.ndarray, press: np.ndarray) -> None:
        """Update the statistics display."""
        if len(kinetic) == 0:
            return
        
        # Calculate statistics
        stats_text = "Current Statistics:\n\n"
        stats_text += f"Kinetic Energy:\n"
        stats_text += f"  Mean: {np.mean(kinetic):.2f} kJ/mol\n"
        stats_text += f"  Std:  {np.std(kinetic):.2f} kJ/mol\n\n"
        
        stats_text += f"Potential Energy:\n"
        stats_text += f"  Mean: {np.mean(potential):.2f} kJ/mol\n"
        stats_text += f"  Std:  {np.std(potential):.2f} kJ/mol\n\n"
        
        stats_text += f"Total Energy:\n"
        stats_text += f"  Mean: {np.mean(total):.2f} kJ/mol\n"
        stats_text += f"  Std:  {np.std(total):.2f} kJ/mol\n\n"
        
        stats_text += f"Temperature:\n"
        stats_text += f"  Mean: {np.mean(temp):.1f} K\n"
        stats_text += f"  Std:  {np.std(temp):.1f} K\n\n"
        
        if np.max(press) > 0:
            stats_text += f"Pressure:\n"
            stats_text += f"  Mean: {np.mean(press):.1f} bar\n"
            stats_text += f"  Std:  {np.std(press):.1f} bar\n\n"
        
        # Energy conservation
        if len(total) > 1:
            energy_drift = total[-1] - total[0]
            drift_percent = (energy_drift / total[0]) * 100
            stats_text += f"Energy Conservation:\n"
            stats_text += f"  Drift: {energy_drift:.3f} kJ/mol\n"
            stats_text += f"  Drift: {drift_percent:.3f}%"
        
        self.stats_text.set_text(stats_text)
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring of the connected simulation."""
        if self.simulation is None:
            raise ValueError("No simulation connected. Use connect_simulation() first.")
        
        if self.fig is None:
            self.setup_plots()
        
        self.is_running = True
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("Started real-time energy monitoring")
        
        # Show the plot
        plt.show()
    
    def _update_loop(self) -> None:
        """Main update loop running in separate thread."""
        while self.is_running:
            try:
                # Get current simulation state
                if hasattr(self.simulation, 'current_step') and self.simulation.current_step > 0:
                    # Extract energy data from simulation
                    time_ps = self.simulation.current_time if hasattr(self.simulation, 'current_time') else 0.0
                    
                    # Calculate energies
                    kinetic = self.simulation.calculate_kinetic_energy()
                    
                    # Get potential energy from last force calculation
                    potential = getattr(self.simulation, 'potential_energy', 0.0)
                    
                    # Calculate temperature
                    temperature = self.simulation.calculate_temperature(kinetic)
                    
                    # Get pressure if available
                    pressure = getattr(self.simulation, 'pressure', None)
                    
                    # Add data point
                    self.add_data_point(time_ps, kinetic, potential, temperature, pressure)
                    
                    # Update plots
                    self.update_plots()
                
                time.sleep(self.update_interval / 1000.0)
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(0.1)
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.is_running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join()
        logger.info("Stopped real-time energy monitoring")
    
    def export_plot(self, filename: str, dpi: Optional[int] = None, 
                   format: Optional[str] = None) -> None:
        """
        Export the current plot as a high-resolution image.
        
        Parameters
        ----------
        filename : str
            Output filename
        dpi : int, optional
            Resolution in dots per inch (default: 300)
        format : str, optional
            Output format ('png', 'pdf', 'svg', etc.) (default: 'png')
        """
        if self.fig is None:
            raise ValueError("No plot to export. Call setup_plots() first.")
        
        dpi = dpi or self.export_dpi
        format = format or self.export_format
        
        # Ensure filename has correct extension
        if not filename.endswith(f'.{format}'):
            filename = f"{filename}.{format}"
        
        # Save with high quality
        self.fig.savefig(filename, dpi=dpi, format=format, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
        
        logger.info(f"Exported energy plot to {filename} (DPI: {dpi}, Format: {format})")
    
    def export_data(self, filename: str) -> None:
        """
        Export the current data to a CSV file.
        
        Parameters
        ----------
        filename : str
            Output CSV filename
        """
        if len(self.time_data) == 0:
            logger.warning("No data to export")
            return
        
        try:
            import pandas as pd
            
            # Create DataFrame
            data = {
                'Time_ps': list(self.time_data),
                'Kinetic_Energy_kJ_mol': list(self.kinetic_energy),
                'Potential_Energy_kJ_mol': list(self.potential_energy),
                'Total_Energy_kJ_mol': list(self.total_energy),
                'Temperature_K': list(self.temperature),
                'Pressure_bar': list(self.pressure)
            }
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
        except ImportError:
            # Fallback to basic CSV writing
            import csv
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['Time_ps', 'Kinetic_Energy_kJ_mol', 'Potential_Energy_kJ_mol', 
                               'Total_Energy_kJ_mol', 'Temperature_K', 'Pressure_bar'])
                # Write data
                for i in range(len(self.time_data)):
                    writer.writerow([
                        self.time_data[i], self.kinetic_energy[i], self.potential_energy[i],
                        self.total_energy[i], self.temperature[i], self.pressure[i]
                    ])
        
        logger.info(f"Exported energy data to {filename}")
    
    def _on_key_press(self, event) -> None:
        """Handle key press events for interactive features."""
        if event.key == 's':
            # Save current plot
            timestamp = int(time.time())
            filename = f"energy_plot_{timestamp}"
            self.export_plot(filename)
        elif event.key == 'd':
            # Export data
            timestamp = int(time.time())
            filename = f"energy_data_{timestamp}.csv"
            self.export_data(filename)
        elif event.key == 'r':
            # Reset zoom
            for ax in self.axes.values():
                ax.relim()
                ax.autoscale()
            self.fig.canvas.draw()
    
    def clear_data(self) -> None:
        """Clear all stored data."""
        self.time_data.clear()
        self.kinetic_energy.clear()
        self.potential_energy.clear()
        self.total_energy.clear()
        self.temperature.clear()
        self.pressure.clear()
        
        logger.info("Cleared all energy data")
    
    def get_current_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get current statistics for all monitored quantities.
        
        Returns
        -------
        dict
            Statistics dictionary with mean, std, min, max for each quantity
        """
        if len(self.time_data) == 0:
            return {}
        
        def calc_stats(data):
            arr = np.array(data)
            return {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr))
            }
        
        return {
            'kinetic_energy': calc_stats(self.kinetic_energy),
            'potential_energy': calc_stats(self.potential_energy),
            'total_energy': calc_stats(self.total_energy),
            'temperature': calc_stats(self.temperature),
            'pressure': calc_stats(self.pressure) if len(self.pressure) > 0 else {}
        }
    
    def create_energy_plot(self, energy_data: Dict, output_file: str = None):
        """Create an energy plot from energy data."""
        if self.fig is None:
            self.setup_plots()
        
        # Extract data from energy_data dict
        times = energy_data.get('time', [])
        kinetic = energy_data.get('kinetic', [])
        potential = energy_data.get('potential', [])
        temperature = energy_data.get('temperature', [300.0] * len(times))  # Default temperature if not provided
        
        # Add data points
        for t, k, p, temp in zip(times, kinetic, potential, temperature):
            self.add_data_point(t, k, p, temp)
        
        # Update plots
        self.update_plots()
        
        # Export if filename provided
        if output_file:
            self.export_plot(output_file)
    
    def update_energy_data(self, step: int, energies: Dict):
        """Update energy data from simulation step."""
        time_ps = step * 0.001  # Assume 1 fs timestep
        kinetic = energies.get('kinetic', 0.0)
        potential = energies.get('potential', 0.0)
        temperature = energies.get('temperature', 300.0)
        self.add_data_point(time_ps, kinetic, potential, temperature)
    
    def calculate_statistics(self, energy_data: Dict) -> Dict:
        """Calculate energy statistics."""
        # If no data in dashboard, calculate from provided energy_data
        if len(self.time_data) == 0 and energy_data:
            # Calculate statistics from provided data
            total_energy = energy_data.get('total', [])
            if len(total_energy) == 0:
                # Calculate total from kinetic and potential if total not provided
                kinetic = energy_data.get('kinetic', [])
                potential = energy_data.get('potential', [])
                if len(kinetic) > 0 and len(potential) > 0:
                    total_energy = np.array(kinetic) + np.array(potential)
            
            if len(total_energy) > 0:
                arr = np.array(total_energy)
                return {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr))
                }
        
        # Use current dashboard data
        current_stats = self.get_current_statistics()
        if 'total_energy' in current_stats:
            return current_stats['total_energy']
        
        return {}
    
    def analyze_conservation(self, energy_data: Dict) -> Dict:
        """Analyze energy conservation."""
        total_energy = energy_data.get('total', [])
        if len(total_energy) == 0:
            # Calculate total from kinetic and potential if total not provided
            kinetic = energy_data.get('kinetic', [])
            potential = energy_data.get('potential', [])
            if len(kinetic) > 0 and len(potential) > 0:
                total_energy = np.array(kinetic) + np.array(potential)
        
        if len(total_energy) > 1:
            arr = np.array(total_energy)
            drift = float((total_energy[-1] - total_energy[0]) / total_energy[0] * 100)
            fluctuation = float(np.std(arr) / np.mean(arr) * 100)  # Coefficient of variation
            return {
                'drift': drift,
                'fluctuation': fluctuation
            }
        return {
            'drift': 0.0,
            'fluctuation': 0.0
        }


def create_energy_dashboard(simulation=None, max_points: int = 1000, 
                          update_interval: int = 100) -> EnergyPlotDashboard:
    """
    Convenience function to create and set up an energy dashboard.
    
    Parameters
    ----------
    simulation : MolecularDynamicsSimulation, optional
        Simulation to connect to
    max_points : int, optional
        Maximum number of data points to display
    update_interval : int, optional
        Update interval in milliseconds
    
    Returns
    -------
    EnergyPlotDashboard
        Configured energy dashboard
    """
    dashboard = EnergyPlotDashboard(max_points=max_points, update_interval=update_interval)
    
    if simulation is not None:
        dashboard.connect_simulation(simulation)
    
    return dashboard


if __name__ == "__main__":
    # Example usage and testing
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    
    # Create test dashboard
    dashboard = create_energy_dashboard(max_points=100)
    dashboard.setup_plots()
    
    # Add some test data
    import numpy as np
    times = np.linspace(0, 10, 50)
    for t in times:
        kinetic = 1000 + 50 * np.sin(0.5 * t) + np.random.normal(0, 10)
        potential = -2000 + 100 * np.cos(0.3 * t) + np.random.normal(0, 20)
        temperature = 300 + 20 * np.sin(0.2 * t) + np.random.normal(0, 5)
        pressure = 1.0 + 0.1 * np.sin(0.4 * t) + np.random.normal(0, 0.05)
        
        dashboard.add_data_point(t, kinetic, potential, temperature, pressure)
    
    # Update plots and export
    dashboard.update_plots()
    dashboard.export_plot("test_energy_dashboard.png")
    
    print("Energy dashboard test completed successfully!")
    print(f"Statistics: {dashboard.get_current_statistics()}")
