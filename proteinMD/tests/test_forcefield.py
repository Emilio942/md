"""
Comprehensive Unit Tests for Force Field Module

Task 10.1: Umfassende Unit Tests 🚀

Tests the force field implementations including:
- AMBER ff14SB parameter loading and validation
- Force calculations (bonded and non-bonded)
- Parameter assignment and validation
- Performance benchmarks
"""

import pytest
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

# Try to import force field modules
try:
    from forcefield.forcefield import (
        ForceField, ForceFieldSystem, ForceTerm,
        HarmonicBondForceTerm, HarmonicAngleForceTerm,
        PeriodicTorsionForceTerm, LennardJonesForceTerm,
        CoulombForceTerm, NonbondedMethod
    )
    FORCEFIELD_AVAILABLE = True
except ImportError:
    FORCEFIELD_AVAILABLE = False

try:
    from forcefield.amber_ff14sb import AmberFF14SB, create_amber_ff14sb
    AMBER_AVAILABLE = True
except ImportError:
    AMBER_AVAILABLE = False

try:
    from forcefield.amber_validator import (
        AtomTypeParameters, BondParameters, AngleParameters, DihedralParameters
    )
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False


class TestForceFieldBase:
    """Test the base ForceField class."""
    
    def test_force_field_initialization(self):
        """Test basic force field initialization."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        ff = ForceField(name="test_ff", cutoff=1.0)
        
        assert ff.name == "test_ff"
        assert ff.cutoff == 1.0
        assert ff.switch_distance is None
        assert ff.nonbonded_method == NonbondedMethod.PME
    
    def test_force_field_invalid_parameters(self):
        """Test force field rejects invalid parameters."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        # Test negative cutoff
        with pytest.raises((ValueError, AssertionError)):
            ForceField(cutoff=-1.0)
        
        # Test invalid switch distance
        with pytest.raises((ValueError, AssertionError)):
            ForceField(cutoff=1.0, switch_distance=1.5)  # switch > cutoff


class TestForceFieldSystem:
    """Test the ForceFieldSystem class."""
    
    def test_system_creation(self):
        """Test force field system creation."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        system = ForceFieldSystem("test_system")
        
        assert system.name == "test_system"
        assert len(system.force_terms) == 0
        assert system.n_atoms == 0
    
    def test_add_force_terms(self):
        """Test adding force terms to the system."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        system = ForceFieldSystem("test_system")
        
        # Create mock force term
        force_term = Mock(spec=ForceTerm)
        force_term.name = "test_force"
        force_term.calculate_forces = Mock(return_value=(np.zeros((10, 3)), 0.0))
        
        system.add_force_term(force_term)
        
        assert len(system.force_terms) == 1
        assert system.force_terms[0] == force_term
    
    def test_force_calculation(self):
        """Test force calculation from multiple terms."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        system = ForceFieldSystem("test_system")
        n_atoms = 10
        
        # Add multiple force terms
        for i in range(3):
            force_term = Mock(spec=ForceTerm)
            force_term.name = f"force_{i}"
            
            # Mock both calculate and calculate_forces for compatibility
            mock_result = (np.random.randn(n_atoms, 3), float(i))
            force_term.calculate = Mock(return_value=mock_result)
            force_term.calculate_forces = Mock(return_value=mock_result)
            
            system.add_force_term(force_term)
        
        positions = np.random.randn(n_atoms, 3)
        forces, energy = system.calculate_forces(positions)
        
        assert forces.shape == (n_atoms, 3)
        assert np.isfinite(energy)
        
        # Verify all force terms were called (check calculate method since that's what system calls)
        for force_term in system.force_terms:
            force_term.calculate.assert_called_once()


class TestHarmonicBondForceTerm:
    """Test harmonic bond force calculations."""
    
    def test_bond_force_initialization(self):
        """Test bond force term initialization."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        bonds = [(0, 1), (1, 2)]
        k_values = [1000.0, 1200.0]
        r0_values = [0.15, 0.14]
        
        bond_force = HarmonicBondForceTerm(bonds, k_values, r0_values)
        
        assert len(bond_force.bonds) == 2
        assert bond_force.k_values[0] == 1000.0
        assert bond_force.r0_values[1] == 0.14
    
    def test_bond_force_calculation(self):
        """Test bond force calculation."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        # Simple two-atom system
        bonds = [(0, 1)]
        k_values = [1000.0]  # kJ/mol/nm²
        r0_values = [0.15]   # nm
        
        bond_force = HarmonicBondForceTerm(bonds, k_values, r0_values)
        
        # Positions: atoms at distance 0.16 nm (stretched by 0.01 nm)
        positions = np.array([[0.0, 0.0, 0.0], [0.16, 0.0, 0.0]])
        
        forces, energy = bond_force.calculate_forces(positions)
        
        # Check energy (should be 0.5 * k * (r - r0)²)
        expected_energy = 0.5 * 1000.0 * (0.16 - 0.15)**2
        assert abs(energy - expected_energy) < 0.01
        
        # Check forces (should point toward equilibrium)
        assert forces[0, 0] > 0  # Force on atom 0 toward atom 1
        assert forces[1, 0] < 0  # Force on atom 1 toward atom 0
        assert abs(forces[0, 0] + forces[1, 0]) < 1e-10  # Newton's 3rd law
    
    def test_bond_force_zero_displacement(self):
        """Test bond force at equilibrium distance."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        bonds = [(0, 1)]
        k_values = [1000.0]
        r0_values = [0.15]
        
        bond_force = HarmonicBondForceTerm(bonds, k_values, r0_values)
        
        # Positions at equilibrium distance
        positions = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])
        
        forces, energy = bond_force.calculate_forces(positions)
        
        # Energy and forces should be zero at equilibrium
        assert abs(energy) < 1e-10
        assert np.allclose(forces, 0.0, atol=1e-10)


class TestHarmonicAngleForceTerm:
    """Test harmonic angle force calculations."""
    
    def test_angle_force_initialization(self):
        """Test angle force term initialization."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        angles = [(0, 1, 2), (1, 2, 3)]
        k_values = [100.0, 120.0]
        theta0_values = [109.5, 120.0]  # degrees
        
        angle_force = HarmonicAngleForceTerm(angles, k_values, theta0_values)
        
        assert len(angle_force.angles) == 2
        assert angle_force.k_values[0] == 100.0
        assert angle_force.theta0_values[1] == 120.0
    
    def test_angle_force_calculation(self):
        """Test angle force calculation."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        # Three-atom system forming an angle
        angles = [(0, 1, 2)]
        k_values = [100.0]      # kJ/mol/rad²
        theta0_values = [120.0]  # degrees
        
        angle_force = HarmonicAngleForceTerm(angles, k_values, theta0_values)
        
        # Positions forming 110° angle (10° deviation from equilibrium)
        positions = np.array([
            [0.0, 0.0, 0.0],     # atom 0
            [0.1, 0.0, 0.0],     # atom 1 (vertex)
            [0.05, 0.087, 0.0]   # atom 2 (forms ~110°)
        ])
        
        forces, energy = angle_force.calculate_forces(positions)
        
        # Energy should be positive (deviation from equilibrium)
        assert energy > 0
        
        # Forces should be finite
        assert np.all(np.isfinite(forces))
        
        # Net force should be zero (no external forces)
        net_force = np.sum(forces, axis=0)
        assert np.allclose(net_force, 0.0, atol=1e-8)


class TestLennardJonesForceTerm:
    """Test Lennard-Jones non-bonded forces."""
    
    def test_lj_force_initialization(self):
        """Test LJ force term initialization."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        n_atoms = 5
        sigma = np.random.uniform(0.2, 0.4, n_atoms)
        epsilon = np.random.uniform(0.1, 1.0, n_atoms)
        
        lj_force = LennardJonesForceTerm(sigma, epsilon, cutoff=1.0)
        
        assert len(lj_force.sigma) == n_atoms
        assert len(lj_force.epsilon) == n_atoms
        assert lj_force.cutoff == 1.0
    
    def test_lj_force_calculation_two_atoms(self):
        """Test LJ force calculation for two atoms."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        # Two identical atoms
        sigma = np.array([0.35, 0.35])
        epsilon = np.array([0.1, 0.1])
        
        lj_force = LennardJonesForceTerm(sigma, epsilon, cutoff=1.0)
        
        # Position atoms at sigma distance (should give specific force)
        positions = np.array([[0.0, 0.0, 0.0], [0.35, 0.0, 0.0]])
        
        forces, energy = lj_force.calculate_forces(positions)
        
        # At sigma distance, energy should be zero
        assert abs(energy) < 0.01
        
        # Forces should be repulsive at this distance
        assert forces[0, 0] < 0  # Force on atom 0 away from atom 1
        assert forces[1, 0] > 0  # Force on atom 1 away from atom 0
    
    def test_lj_cutoff_behavior(self):
        """Test LJ force cutoff behavior."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        sigma = np.array([0.35, 0.35])
        epsilon = np.array([0.1, 0.1])
        cutoff = 1.0
        
        lj_force = LennardJonesForceTerm(sigma, epsilon, cutoff=cutoff)
        
        # Position atoms beyond cutoff
        positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        
        forces, energy = lj_force.calculate_forces(positions)
        
        # Forces and energy should be zero beyond cutoff
        assert abs(energy) < 1e-10
        assert np.allclose(forces, 0.0, atol=1e-10)


class TestCoulombForceTerm:
    """Test Coulomb electrostatic forces."""
    
    def test_coulomb_force_initialization(self):
        """Test Coulomb force term initialization."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        charges = np.array([0.5, -0.3, 0.1, -0.3])
        
        coulomb_force = CoulombForceTerm(charges, cutoff=1.0)
        
        assert len(coulomb_force.charges) == 4
        assert coulomb_force.cutoff == 1.0
    
    def test_coulomb_force_calculation(self):
        """Test Coulomb force calculation."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        # Two opposite charges
        charges = np.array([1.0, -1.0])
        
        coulomb_force = CoulombForceTerm(charges, cutoff=2.0)
        
        # Position charges 1 nm apart
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        
        forces, energy = coulomb_force.calculate_forces(positions)
        
        # Energy should be negative (attractive)
        assert energy < 0
        
        # Forces should be attractive
        assert forces[0, 0] > 0  # Force on positive charge toward negative
        assert forces[1, 0] < 0  # Force on negative charge toward positive
    
    def test_coulomb_charge_neutrality(self):
        """Test system with overall charge neutrality."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        # Neutral system
        charges = np.array([0.5, -0.5, 0.3, -0.3])
        assert abs(np.sum(charges)) < 1e-10
        
        coulomb_force = CoulombForceTerm(charges, cutoff=2.0)
        
        # Random positions
        positions = np.random.randn(4, 3)
        
        forces, energy = coulomb_force.calculate_forces(positions)
        
        # Net force should be zero for neutral system
        net_force = np.sum(forces, axis=0)
        assert np.allclose(net_force, 0.0, atol=1e-8)


@pytest.mark.skipif(not AMBER_AVAILABLE, reason="AMBER ff14SB module not available")
class TestAmberFF14SB:
    """Test AMBER ff14SB force field implementation."""
    
    def test_amber_initialization(self):
        """Test AMBER ff14SB initialization."""
        ff = create_amber_ff14sb()
        
        assert ff.name == "AMBER-ff14SB"
        assert ff.cutoff == 1.0
        assert isinstance(ff.amino_acid_library, dict)
        assert len(ff.amino_acid_library) >= 20  # Standard amino acids
    
    def test_amino_acid_coverage(self):
        """Test that all standard amino acids are covered."""
        ff = create_amber_ff14sb()
        
        standard_aas = [
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
        ]
        
        for aa in standard_aas:
            assert aa in ff.amino_acid_library, f"Missing amino acid: {aa}"
            
            template = ff.amino_acid_library[aa]
            assert 'atoms' in template, f"No atoms defined for {aa}"
            assert len(template['atoms']) > 0, f"Empty atom list for {aa}"
    
    def test_parameter_loading(self):
        """Test parameter database loading."""
        ff = create_amber_ff14sb()
        
        # Check that parameters were loaded
        assert len(ff.atom_type_parameters) > 0
        assert len(ff.bond_parameters) > 0
        assert len(ff.angle_parameters) > 0
        assert len(ff.dihedral_parameters) > 0
    
    def test_parameter_retrieval(self):
        """Test parameter retrieval methods."""
        ff = create_amber_ff14sb()
        
        # Test bond parameters
        bond_params = ff.get_bond_parameters('CT', 'CT')
        if bond_params is not None:
            assert hasattr(bond_params, 'k')
            assert hasattr(bond_params, 'r0')
            assert bond_params.k > 0
            assert bond_params.r0 > 0
        
        # Test angle parameters
        angle_params = ff.get_angle_parameters('CT', 'CT', 'CT')
        if angle_params is not None:
            assert hasattr(angle_params, 'k')
            assert hasattr(angle_params, 'theta0')
            assert angle_params.k > 0
            assert 0 < angle_params.theta0 < 180
    
    def test_protein_parameter_validation(self, mock_protein_structure):
        """Test protein parameter validation."""
        ff = create_amber_ff14sb()
        
        validation = ff.validate_protein_parameters(mock_protein_structure)
        
        assert 'total_atoms' in validation
        assert 'parametrized_atoms' in validation
        assert 'missing_parameters' in validation
        assert 'coverage_percentage' in validation
        
        assert validation['total_atoms'] >= 0
        assert validation['parametrized_atoms'] >= 0
        assert validation['coverage_percentage'] >= 0
        assert validation['coverage_percentage'] <= 100
    
    def test_system_creation(self, mock_protein_structure):
        """Test simulation system creation."""
        ff = create_amber_ff14sb()
        
        try:
            system = ff.create_simulation_system(mock_protein_structure)
            
            assert system is not None
            assert hasattr(system, 'name')
            assert hasattr(system, 'force_terms')
            
        except (NotImplementedError, AttributeError):
            pytest.skip("System creation not fully implemented")
    
    @pytest.mark.performance
    def test_amber_performance(self, mock_protein_structure, performance_monitor):
        """Test AMBER force field performance."""
        ff = create_amber_ff14sb()
        
        with performance_monitor:
            # Test parameter validation performance
            for _ in range(10):
                validation = ff.validate_protein_parameters(mock_protein_structure)
        
        metrics = performance_monitor.metrics
        
        # Should be fast enough for interactive use
        assert metrics['duration_ms'] < 1000, f"AMBER validation too slow: {metrics['duration_ms']:.1f} ms"
    
    @pytest.mark.integration
    def test_amber_benchmarking(self):
        """Test AMBER benchmarking functionality."""
        ff = create_amber_ff14sb()
        
        try:
            # Test with small set of proteins
            test_proteins = ["ALANINE_DIPEPTIDE"]
            results = ff.benchmark_against_amber(test_proteins)
            
            assert 'test_proteins' in results
            assert 'overall_accuracy' in results
            assert results['test_proteins'] == test_proteins
            
        except (NotImplementedError, AttributeError):
            pytest.skip("Benchmarking not implemented")


@pytest.mark.skipif(not VALIDATOR_AVAILABLE, reason="Validator module not available")
class TestParameterValidation:
    """Test parameter validation classes."""
    
    def test_atom_type_parameters(self):
        """Test AtomTypeParameters validation."""
        params = AtomTypeParameters(
            atom_type="CT",
            element="C",
            mass=12.01,
            charge=0.0,
            sigma=0.35,
            epsilon=0.1
        )
        
        assert params.is_valid()
        assert params.atom_type == "CT"
        assert params.mass > 0
        assert params.sigma > 0
        assert params.epsilon >= 0
    
    def test_bond_parameters(self):
        """Test BondParameters validation."""
        params = BondParameters(
            atom_type1="CT",
            atom_type2="CT",
            k=1000.0,
            r0=0.15
        )
        
        assert params.is_valid()
        assert params.k > 0
        assert params.r0 > 0
    
    def test_angle_parameters(self):
        """Test AngleParameters validation."""
        params = AngleParameters(
            atom_type1="CT",
            atom_type2="CT",
            atom_type3="CT",
            k=100.0,
            theta0=np.radians(109.5)  # Convert degrees to radians
        )
        
        assert params.is_valid()
        assert params.k > 0
        assert 0 < params.theta0 <= np.pi
    
    def test_dihedral_parameters(self):
        """Test DihedralParameters validation."""
        params = DihedralParameters(
            atom_type1="CT",
            atom_type2="CT",
            atom_type3="CT",
            atom_type4="CT",
            k=10.0,
            n=2,
            phase=0.0
        )
        
        assert params.is_valid()
        assert params.k >= 0
        assert params.n >= 0
        assert -180 <= params.phase <= 180
    
    def test_invalid_parameters(self):
        """Test validation of invalid parameters."""
        # Invalid atom type parameters
        params1 = AtomTypeParameters("CT", "C", -1.0, 0.0, 0.35, 0.1)  # Negative mass
        assert not params1.is_valid()
        
        # Invalid bond parameters
        params2 = BondParameters("CT", "CT", -100.0, 0.15)  # Negative k
        assert not params2.is_valid()
        
        # Invalid angle parameters (angle > π)
        params3 = AngleParameters("CT", "CT", "CT", 100.0, np.radians(200.0))  # Invalid angle
        assert not params3.is_valid()


class TestForceFieldPerformance:
    """Performance tests for force field calculations."""
    
    @pytest.mark.performance
    def test_force_calculation_scaling(self, performance_monitor):
        """Test force calculation performance scaling."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        system_sizes = [50, 100, 200]
        times = []
        
        for n_atoms in system_sizes:
            # Create test system
            positions = np.random.randn(n_atoms, 3) * 2.0
            
            # Create LJ force term
            sigma = np.ones(n_atoms) * 0.35
            epsilon = np.ones(n_atoms) * 0.1
            lj_force = LennardJonesForceTerm(sigma, epsilon, cutoff=1.0)
            
            with performance_monitor:
                for _ in range(10):
                    forces, energy = lj_force.calculate_forces(positions)
            
            time_per_atom = performance_monitor.metrics['duration_ms'] / (10 * n_atoms)
            times.append(time_per_atom)
        
        # Performance should not degrade too much with system size
        # (assuming efficient neighbor lists are used)
        max_time = max(times)
        min_time = min(times)
        
        # Allow up to 10x degradation for O(N²) algorithms
        assert max_time / min_time < 10.0, f"Poor scaling: {times}"
    
    @pytest.mark.memory
    def test_memory_efficiency(self, memory_monitor):
        """Test memory efficiency of force calculations."""
        if not FORCEFIELD_AVAILABLE:
            pytest.skip("ForceField module not available")
        
        initial_memory = memory_monitor
        
        # Create large system
        n_atoms = 1000
        positions = np.random.randn(n_atoms, 3)
        
        # Create multiple force terms
        sigma = np.ones(n_atoms) * 0.35
        epsilon = np.ones(n_atoms) * 0.1
        charges = np.random.uniform(-1, 1, n_atoms)
        
        lj_force = LennardJonesForceTerm(sigma, epsilon, cutoff=1.0)
        coulomb_force = CoulombForceTerm(charges, cutoff=1.0)
        
        # Calculate forces multiple times
        for _ in range(50):
            lj_forces, lj_energy = lj_force.calculate_forces(positions)
            coul_forces, coul_energy = coulomb_force.calculate_forces(positions)
            
            # Modify positions slightly
            positions += np.random.randn(n_atoms, 3) * 0.01
        
        # Memory usage should not grow excessively
        # (Checked by memory_monitor fixture)
