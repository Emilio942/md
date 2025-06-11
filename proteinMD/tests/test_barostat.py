"""
Test module for barostat functionality in the simulation module.

This module contains tests to ensure that pressure control
algorithms function correctly in molecular dynamics simulations.
"""
import unittest
import numpy as np
from core.simulation import MolecularDynamicsSimulation

class TestBarostat(unittest.TestCase):
    """Test class for barostat functionality in MolecularDynamicsSimulation."""
    
    def setUp(self):
        """Set up a simple simulation for testing."""
        # Create a simulation with a small box and several particles
        self.sim = MolecularDynamicsSimulation(
            num_particles=0,
            box_dimensions=np.array([3.0, 3.0, 3.0]),  # 3x3x3 nm box
            barostat='berendsen'  # Use Berendsen barostat
        )
        
        # Add some particles to create pressure
        num_particles = 50
        positions = np.random.uniform(0, 3.0, (num_particles, 3))
        masses = np.ones(num_particles) * 12.0  # atomic mass units
        charges = np.random.uniform(-0.5, 0.5, num_particles)  # random charges
        
        self.sim.add_particles(positions, masses, charges)
        
        # Initialize velocities to create kinetic energy
        self.sim.initialize_velocities(temperature=300.0)
        
        # Store original state
        self.original_box = self.sim.box_dimensions.copy()
        self.original_positions = self.sim.positions.copy()
        
    def test_pressure_calculation(self):
        """Test that pressure calculation returns reasonable values."""
        # Calculate pressure
        pressure = self.sim.calculate_pressure()
        
        # Pressure should be finite
        self.assertTrue(np.isfinite(pressure))
        
        # Pressure should be positive for this simulation setup
        self.assertGreater(pressure, 0.0)
        
    def test_berendsen_barostat(self):
        """Test that Berendsen barostat adjusts the box size."""
        # Set target pressure
        self.sim.target_pressure = 1.0  # 1 bar
        
        # Apply barostat
        self.sim.apply_barostat()
        
        # Box dimensions should change
        self.assertFalse(np.allclose(self.sim.box_dimensions, self.original_box))
        
        # Positions should scale with the box
        scaling_factor = np.mean(self.sim.box_dimensions / self.original_box)
        for i in range(self.sim.num_particles):
            expected_position = self.original_positions[i] * scaling_factor
            np.testing.assert_allclose(self.sim.positions[i], expected_position, rtol=1e-5)
        
    def test_parrinello_rahman_barostat(self):
        """Test that Parrinello-Rahman barostat works."""
        # Change to Parrinello-Rahman barostat
        self.sim.barostat = 'parrinello-rahman'
        
        # Set target pressure
        self.sim.target_pressure = 1.0  # 1 bar
        
        # Apply barostat
        self.sim.apply_barostat()
        
        # Box dimensions should change
        self.assertFalse(np.allclose(self.sim.box_dimensions, self.original_box))
        
        # Positions should scale with the box
        scaling_factor = np.mean(self.sim.box_dimensions / self.original_box)
        for i in range(self.sim.num_particles):
            expected_position = self.original_positions[i] * scaling_factor
            np.testing.assert_allclose(self.sim.positions[i], expected_position, rtol=1e-5)
        
    def test_no_barostat(self):
        """Test that no changes occur when barostat is disabled."""
        # Disable barostat
        self.sim.barostat = None
        
        # Apply barostat (should do nothing)
        pressure = self.sim.apply_barostat()
        
        # Box dimensions should remain unchanged
        np.testing.assert_allclose(self.sim.box_dimensions, self.original_box)
        
        # Positions should remain unchanged
        np.testing.assert_allclose(self.sim.positions, self.original_positions)
        
        # But pressure should still be calculated correctly
        self.assertTrue(np.isfinite(pressure))
        
    def test_extreme_pressure(self):
        """Test barostat behavior under extreme pressure conditions."""
        # Create a simulation with particles very close together to generate high pressure
        sim = MolecularDynamicsSimulation(
            num_particles=0,
            box_dimensions=np.array([1.0, 1.0, 1.0]),  # Small box
            barostat='berendsen'
        )
        
        # Add many particles in a small volume
        num_particles = 100
        positions = np.random.uniform(0, 1.0, (num_particles, 3))
        masses = np.ones(num_particles) * 12.0
        charges = np.random.uniform(-0.5, 0.5, num_particles)
        
        sim.add_particles(positions, masses, charges)
        
        # Calculate initial pressure (should be high)
        initial_pressure = sim.calculate_pressure()
        
        # Set a low target pressure
        sim.target_pressure = 1.0  # 1 bar
        
        # Apply barostat
        new_pressure = sim.apply_barostat()
        
        # Box should expand to reduce pressure
        self.assertTrue(np.all(sim.box_dimensions > np.array([1.0, 1.0, 1.0])))
        
        # New pressure should be lower than initial pressure
        self.assertLess(new_pressure, initial_pressure)
        
if __name__ == '__main__':
    unittest.main()
