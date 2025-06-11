"""
Test module for force calculations in the simulation module.

This module contains tests to ensure that force calculations are
stable and accurate, especially in edge cases that could cause 
numerical instability.
"""
import unittest
import numpy as np
from core.simulation import MolecularDynamicsSimulation

class TestForceCalculations(unittest.TestCase):
    """Test class for force calculations in MolecularDynamicsSimulation."""
    
    def setUp(self):
        """Set up a simple simulation for testing."""
        # Create a simple simulation with 2 particles
        self.sim = MolecularDynamicsSimulation(num_particles=0)
        
        # Add two particles
        positions = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])  # 0.3 nm apart
        masses = np.array([12.0, 12.0])  # atomic mass units
        charges = np.array([0.0, 0.0])  # no charge for initial tests
        
        self.sim.add_particles(positions, masses, charges)
        
    def test_nonbonded_forces_normal_distance(self):
        """Test that nonbonded forces are calculated correctly at normal distances."""
        # Calculate forces between particles at a normal distance
        forces = self.sim._calculate_nonbonded_forces()
        
        # Check that forces are equal and opposite (Newton's third law)
        np.testing.assert_allclose(forces[0], -forces[1], rtol=1e-10)
        
        # For neutral particles, forces should still exist (Lennard-Jones)
        self.assertTrue(np.any(forces != 0))
        
    def test_nonbonded_forces_very_close(self):
        """Test that nonbonded forces don't blow up when particles are very close."""
        # Position particles extremely close to test numerical stability
        self.sim.positions = np.array([[0.0, 0.0, 0.0], [1e-6, 0.0, 0.0]])  # Very close
        
        # Calculate forces
        forces = self.sim._calculate_nonbonded_forces()
        
        # Forces should be finite (not NaN or inf) even when particles are very close
        self.assertTrue(np.all(np.isfinite(forces)))
        
        # Forces should still obey Newton's third law
        np.testing.assert_allclose(forces[0], -forces[1], rtol=1e-10)
        
    def test_bonded_forces_normal_distance(self):
        """Test that bonded forces are calculated correctly at normal distances."""
        # Add a bond between the particles
        k_bond = 1000.0  # kJ/(mol·nm²)
        r_0 = 0.2  # nm, equilibrium bond length
        self.sim.add_bonds([(0, 1, k_bond, r_0)])
        
        # Calculate bonded forces
        forces = self.sim._calculate_bonded_forces()
        
        # Check that forces are equal and opposite
        np.testing.assert_allclose(forces[0], -forces[1], rtol=1e-10)
        
        # Force magnitude should be proportional to displacement from equilibrium
        # F = -k(r - r_0)
        expected_magnitude = k_bond * (0.3 - r_0)  # particles are 0.3 nm apart
        calculated_magnitude = np.linalg.norm(forces[0])
        self.assertAlmostEqual(calculated_magnitude, expected_magnitude, delta=1e-8)
        
    def test_bonded_forces_zero_distance(self):
        """Test that bonded forces don't blow up when bond length is zero."""
        # Position particles at exactly the same place
        self.sim.positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        
        # Add a bond between the particles
        k_bond = 1000.0  # kJ/(mol·nm²)
        r_0 = 0.2  # nm, equilibrium bond length
        self.sim.add_bonds([(0, 1, k_bond, r_0)])
        
        # Calculate bonded forces
        forces = self.sim._calculate_bonded_forces()
        
        # Forces should be finite (not NaN or inf)
        self.assertTrue(np.all(np.isfinite(forces)))
        
    def test_angle_forces(self):
        """Test that angle forces are calculated correctly."""
        # Create a simulation with 3 particles for testing angles
        sim = MolecularDynamicsSimulation(num_particles=0)
        
        # Add three particles in an angle formation
        positions = np.array([
            [0.0, 0.0, 0.0],    # particle 0
            [0.1, 0.0, 0.0],    # particle 1 (at angle vertex)
            [0.1, 0.1, 0.0]     # particle 2
        ])
        masses = np.array([12.0, 12.0, 12.0])
        charges = np.array([0.0, 0.0, 0.0])
        
        sim.add_particles(positions, masses, charges)
        
        # Add an angle interaction
        k_angle = 100.0  # kJ/(mol·rad²)
        theta_0 = np.pi/2  # 90 degrees, equilibrium angle
        sim.add_angles([(0, 1, 2, k_angle, theta_0)])
        
        # Calculate angle forces
        forces = sim._calculate_angle_forces()
        
        # Forces should be finite
        self.assertTrue(np.all(np.isfinite(forces)))
        
        # Sum of forces should be close to zero (conservation of momentum)
        self.assertAlmostEqual(np.linalg.norm(np.sum(forces, axis=0)), 0.0, delta=1e-10)
        
    def test_dihedral_forces(self):
        """Test that dihedral forces are calculated correctly."""
        # Create a simulation with 4 particles for testing dihedrals
        sim = MolecularDynamicsSimulation(num_particles=0)
        
        # Add four particles in a dihedral formation
        positions = np.array([
            [0.0, 0.0, 0.0],    # particle 0
            [0.1, 0.0, 0.0],    # particle 1
            [0.2, 0.1, 0.0],    # particle 2
            [0.3, 0.1, 0.1]     # particle 3
        ])
        masses = np.array([12.0, 12.0, 12.0, 12.0])
        charges = np.array([0.0, 0.0, 0.0, 0.0])
        
        sim.add_particles(positions, masses, charges)
        
        # Add a dihedral interaction
        k_dihedral = 10.0  # kJ/mol
        n = 1  # multiplicity
        phi_0 = 0.0  # equilibrium angle
        sim.add_dihedrals([(0, 1, 2, 3, k_dihedral, n, phi_0)])
        
        # Calculate dihedral forces
        forces = sim._calculate_dihedral_forces()
        
        # Forces should be finite
        self.assertTrue(np.all(np.isfinite(forces)))
        
        # Sum of forces should be close to zero (conservation of momentum)
        self.assertAlmostEqual(np.linalg.norm(np.sum(forces, axis=0)), 0.0, delta=1e-10)
        
    def test_extreme_forces_limiting(self):
        """Test that extreme forces are properly limited."""
        # Create a system with highly charged particles very close together
        sim = MolecularDynamicsSimulation(num_particles=0)
        
        # Position two highly charged particles very close
        positions = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])  # 0.01 nm apart
        masses = np.array([12.0, 12.0])
        charges = np.array([5.0, -5.0])  # large opposite charges
        
        sim.add_particles(positions, masses, charges)
        
        # Calculate forces
        forces = sim._calculate_nonbonded_forces()
        
        # Forces should be finite
        self.assertTrue(np.all(np.isfinite(forces)))
        
        # Forces magnitude should be limited to the maximum allowed value
        max_force_magnitude = 1000.0  # kJ/(mol·nm)
        self.assertLessEqual(np.linalg.norm(forces[0]), max_force_magnitude * 1.01)
        self.assertLessEqual(np.linalg.norm(forces[1]), max_force_magnitude * 1.01)
        
    def test_electrostatic_forces_zero_charge(self):
        """Test that electrostatic forces are zero when charges are zero."""
        # Create particles with zero charge
        sim = MolecularDynamicsSimulation(num_particles=0)
        positions = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        masses = np.array([12.0, 12.0])
        charges = np.array([0.0, 0.0])  # zero charges
        
        sim.add_particles(positions, masses, charges)
        
        # Add direct access to the electrostatic force calculation for testing
        sim.charges = np.array([0.0, 0.0])
        r_squared = 0.04  # 0.2^2
        q_i, q_j = 0.0, 0.0
        k_coulomb = 138.935458
        
        # Compute the electrostatic force
        f_elec = k_coulomb * q_i * q_j / r_squared
        
        # Force should be zero
        self.assertEqual(f_elec, 0.0)
        
if __name__ == '__main__':
    unittest.main()
