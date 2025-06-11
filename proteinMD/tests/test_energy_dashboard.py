"""
Test suite for the EnergyPlotDashboard class.

This module tests all functionality of the energy dashboard including
real-time monitoring, plotting, and export capabilities.
"""

import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import tempfile
import os
import time
from unittest.mock import Mock, patch
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from visualization.energy_dashboard import EnergyPlotDashboard, create_energy_dashboard

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockSimulation:
    """Mock simulation class for testing the energy dashboard."""
    
    def __init__(self):
        self.current_step = 0
        self.current_time = 0.0
        self.potential_energy = -2000.0
        self.pressure = 1.0
        self.positions = np.random.random((100, 3))
        self.velocities = np.random.normal(0, 1, (100, 3))
        
    def calculate_kinetic_energy(self):
        """Calculate kinetic energy from velocities."""
        # KE = 0.5 * m * v^2, assume unit mass
        return 0.5 * np.sum(self.velocities**2) * 10  # Scale for realistic values
    
    def calculate_temperature(self, kinetic_energy=None):
        """Calculate temperature from kinetic energy."""
        if kinetic_energy is None:
            kinetic_energy = self.calculate_kinetic_energy()
        # T = 2*KE / (3*N*k_B), simplified calculation
        return 300 + (kinetic_energy - 1000) * 0.1
    
    def step(self):
        """Simulate one MD step."""
        self.current_step += 1
        self.current_time += 0.001  # 1 fs timestep
        
        # Add some realistic variation
        self.velocities += np.random.normal(0, 0.01, self.velocities.shape)
        self.potential_energy += np.random.normal(0, 10)
        self.pressure += np.random.normal(0, 0.05)


class TestEnergyPlotDashboard(unittest.TestCase):
    """Test cases for the EnergyPlotDashboard class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dashboard = EnergyPlotDashboard(max_points=100, update_interval=50)
        self.mock_simulation = MockSimulation()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop monitoring if running
        if self.dashboard.is_running:
            self.dashboard.stop_monitoring()
        
        # Close any open figures
        plt.close('all')
        
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test dashboard initialization."""
        self.assertEqual(self.dashboard.max_points, 100)
        self.assertEqual(self.dashboard.update_interval, 50)
        self.assertFalse(self.dashboard.is_running)
        self.assertIsNone(self.dashboard.simulation)
        self.assertEqual(len(self.dashboard.time_data), 0)
    
    def test_connect_simulation(self):
        """Test connecting a simulation to the dashboard."""
        self.dashboard.connect_simulation(self.mock_simulation)
        self.assertEqual(self.dashboard.simulation, self.mock_simulation)
    
    def test_add_data_point(self):
        """Test adding data points to the dashboard."""
        # Add single data point
        self.dashboard.add_data_point(
            time_ps=1.0, 
            kinetic=1000.0, 
            potential=-2000.0, 
            temperature=300.0, 
            pressure=1.0
        )
        
        # Check data was stored
        self.assertEqual(len(self.dashboard.time_data), 1)
        self.assertEqual(self.dashboard.time_data[0], 1.0)
        self.assertEqual(self.dashboard.kinetic_energy[0], 1000.0)
        self.assertEqual(self.dashboard.potential_energy[0], -2000.0)
        self.assertEqual(self.dashboard.total_energy[0], -1000.0)
        self.assertEqual(self.dashboard.temperature[0], 300.0)
        self.assertEqual(self.dashboard.pressure[0], 1.0)
    
    def test_add_multiple_data_points(self):
        """Test adding multiple data points."""
        # Add multiple data points
        for i in range(10):
            self.dashboard.add_data_point(
                time_ps=i * 0.1,
                kinetic=1000 + i * 10,
                potential=-2000 - i * 5,
                temperature=300 + i,
                pressure=1.0 + i * 0.1
            )
        
        # Check all data was stored
        self.assertEqual(len(self.dashboard.time_data), 10)
        self.assertEqual(len(self.dashboard.kinetic_energy), 10)
        self.assertEqual(len(self.dashboard.potential_energy), 10)
        self.assertEqual(len(self.dashboard.total_energy), 10)
        self.assertEqual(len(self.dashboard.temperature), 10)
        self.assertEqual(len(self.dashboard.pressure), 10)
    
    def test_setup_plots(self):
        """Test plot setup."""
        self.dashboard.setup_plots()
        
        # Check figure was created
        self.assertIsNotNone(self.dashboard.fig)
        
        # Check axes were created
        expected_axes = ['energy_total', 'energy_components', 'temperature', 
                        'pressure', 'energy_conservation', 'statistics']
        for ax_name in expected_axes:
            self.assertIn(ax_name, self.dashboard.axes)
        
        # Check lines were created
        expected_lines = ['total_energy', 'kinetic_energy', 'potential_energy', 
                         'temperature', 'pressure', 'energy_drift']
        for line_name in expected_lines:
            self.assertIn(line_name, self.dashboard.lines)
    
    def test_update_plots(self):
        """Test plot updating with data."""
        # Set up plots and add data
        self.dashboard.setup_plots()
        
        # Add test data
        times = np.linspace(0, 5, 20)
        for t in times:
            kinetic = 1000 + 50 * np.sin(0.5 * t)
            potential = -2000 + 100 * np.cos(0.3 * t)
            temperature = 300 + 20 * np.sin(0.2 * t)
            pressure = 1.0 + 0.1 * np.sin(0.4 * t)
            
            self.dashboard.add_data_point(t, kinetic, potential, temperature, pressure)
        
        # Update plots (should not raise any exceptions)
        self.dashboard.update_plots()
        
        # Check that lines have data
        total_line = self.dashboard.lines['total_energy']
        x_data, y_data = total_line.get_data()
        self.assertEqual(len(x_data), len(times))
        self.assertEqual(len(y_data), len(times))
    
    def test_export_plot(self):
        """Test plot export functionality."""
        # Set up plots and add data
        self.dashboard.setup_plots()
        
        # Add some test data
        for i in range(5):
            self.dashboard.add_data_point(
                time_ps=i * 0.1,
                kinetic=1000 + i * 10,
                potential=-2000 - i * 5,
                temperature=300 + i,
                pressure=1.0 + i * 0.1
            )
        
        self.dashboard.update_plots()
        
        # Test export
        output_file = os.path.join(self.temp_dir, "test_energy_plot.png")
        self.dashboard.export_plot(output_file)
        
        # Check file was created
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)
    
    def test_export_data(self):
        """Test data export functionality."""
        # Add some test data
        for i in range(5):
            self.dashboard.add_data_point(
                time_ps=i * 0.1,
                kinetic=1000 + i * 10,
                potential=-2000 - i * 5,
                temperature=300 + i,
                pressure=1.0 + i * 0.1
            )
        
        # Test export
        output_file = os.path.join(self.temp_dir, "test_energy_data.csv")
        self.dashboard.export_data(output_file)
        
        # Check file was created
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)
        
        # Check file content
        with open(output_file, 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 1)  # Header + data
            self.assertIn('Time_ps', lines[0])  # Check header
    
    def test_get_current_statistics(self):
        """Test statistics calculation."""
        # Add test data with known values
        test_data = [
            (0.0, 1000.0, -2000.0, 300.0, 1.0),
            (0.1, 1010.0, -2010.0, 301.0, 1.1),
            (0.2, 1020.0, -2020.0, 302.0, 1.2),
            (0.3, 1030.0, -2030.0, 303.0, 1.3),
            (0.4, 1040.0, -2040.0, 304.0, 1.4),
        ]
        
        for time_ps, kinetic, potential, temperature, pressure in test_data:
            self.dashboard.add_data_point(time_ps, kinetic, potential, temperature, pressure)
        
        # Get statistics
        stats = self.dashboard.get_current_statistics()
        
        # Check structure
        self.assertIn('kinetic_energy', stats)
        self.assertIn('potential_energy', stats)
        self.assertIn('total_energy', stats)
        self.assertIn('temperature', stats)
        self.assertIn('pressure', stats)
        
        # Check kinetic energy statistics
        ke_stats = stats['kinetic_energy']
        self.assertAlmostEqual(ke_stats['mean'], 1020.0, places=1)
        self.assertAlmostEqual(ke_stats['min'], 1000.0, places=1)
        self.assertAlmostEqual(ke_stats['max'], 1040.0, places=1)
    
    def test_clear_data(self):
        """Test data clearing functionality."""
        # Add some data
        for i in range(5):
            self.dashboard.add_data_point(
                time_ps=i * 0.1,
                kinetic=1000.0,
                potential=-2000.0,
                temperature=300.0,
                pressure=1.0
            )
        
        # Check data exists
        self.assertEqual(len(self.dashboard.time_data), 5)
        
        # Clear data
        self.dashboard.clear_data()
        
        # Check data was cleared
        self.assertEqual(len(self.dashboard.time_data), 0)
        self.assertEqual(len(self.dashboard.kinetic_energy), 0)
        self.assertEqual(len(self.dashboard.potential_energy), 0)
        self.assertEqual(len(self.dashboard.total_energy), 0)
        self.assertEqual(len(self.dashboard.temperature), 0)
        self.assertEqual(len(self.dashboard.pressure), 0)
    
    def test_max_points_limit(self):
        """Test that max_points limit is respected."""
        # Set small max_points for testing
        dashboard = EnergyPlotDashboard(max_points=5)
        
        # Add more data points than the limit
        for i in range(10):
            dashboard.add_data_point(
                time_ps=i * 0.1,
                kinetic=1000.0,
                potential=-2000.0,
                temperature=300.0,
                pressure=1.0
            )
        
        # Check that only max_points data is stored
        self.assertEqual(len(dashboard.time_data), 5)
        self.assertEqual(len(dashboard.kinetic_energy), 5)
        
        # Check that the latest data is kept (data from i=5 to i=9)
        self.assertEqual(dashboard.time_data[0], 0.5)  # i=5
        self.assertEqual(dashboard.time_data[-1], 0.9)  # i=9
    
    def test_simulation_integration(self):
        """Test integration with mock simulation."""
        # Connect simulation
        self.dashboard.connect_simulation(self.mock_simulation)
        
        # Simulate some steps and collect data manually
        for _ in range(5):
            self.mock_simulation.step()
            
            # Manually add data (simulating what the update loop would do)
            kinetic = self.mock_simulation.calculate_kinetic_energy()
            potential = self.mock_simulation.potential_energy
            temperature = self.mock_simulation.calculate_temperature(kinetic)
            pressure = self.mock_simulation.pressure
            
            self.dashboard.add_data_point(
                self.mock_simulation.current_time,
                kinetic, potential, temperature, pressure
            )
        
        # Check data was collected
        self.assertEqual(len(self.dashboard.time_data), 5)
        self.assertGreater(self.dashboard.time_data[-1], 0)  # Time should advance


class TestCreateEnergyDashboard(unittest.TestCase):
    """Test the convenience function for creating energy dashboards."""
    
    def test_create_energy_dashboard_no_simulation(self):
        """Test creating dashboard without simulation."""
        dashboard = create_energy_dashboard(max_points=200, update_interval=50)
        
        self.assertIsInstance(dashboard, EnergyPlotDashboard)
        self.assertEqual(dashboard.max_points, 200)
        self.assertEqual(dashboard.update_interval, 50)
        self.assertIsNone(dashboard.simulation)
    
    def test_create_energy_dashboard_with_simulation(self):
        """Test creating dashboard with simulation."""
        mock_sim = MockSimulation()
        dashboard = create_energy_dashboard(simulation=mock_sim)
        
        self.assertIsInstance(dashboard, EnergyPlotDashboard)
        self.assertEqual(dashboard.simulation, mock_sim)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
