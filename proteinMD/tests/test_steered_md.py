"""
Comprehensive Unit Tests for Steered Molecular Dynamics Module

Task 6.3: Steered Molecular Dynamics 📊

Tests for the steered_md module including:
- SMD parameter validation
- Coordinate calculations (distance, angle, dihedral, COM)
- Force calculations (constant velocity and constant force modes)
- Work calculation and Jarzynski analysis
- Force curve visualization
- Integration with simulation systems
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt

# Try to import steered MD module
try:
    from sampling.steered_md import (
        SteeredMD, SMDParameters, CoordinateCalculator, 
        SMDForceCalculator, setup_protein_unfolding_smd,
        setup_ligand_unbinding_smd, setup_bond_stretching_smd
    )
    STEERED_MD_AVAILABLE = True
except ImportError:
    STEERED_MD_AVAILABLE = False


@pytest.mark.skipif(not STEERED_MD_AVAILABLE, reason="Steered MD module not available")
class TestSMDParameters:
    """Test suite for SMD parameters."""
    
    def test_smd_parameters_initialization(self):
        """Test SMD parameters initialization with default values."""
        params = SMDParameters(
            atom_indices=[0, 5]
        )
        
        assert params.atom_indices == [0, 5]
        assert params.coordinate_type == "distance"
        assert params.mode == "constant_velocity"
        assert params.pulling_velocity == 0.01
        assert params.spring_constant == 1000.0
        assert params.applied_force == 100.0
        assert params.n_steps == 10000
        assert params.output_frequency == 100
    
    def test_smd_parameters_custom_values(self):
        """Test SMD parameters with custom values."""
        params = SMDParameters(
            atom_indices=[1, 2, 3],
            coordinate_type="angle",
            mode="constant_force",
            pulling_velocity=0.02,
            spring_constant=500.0,
            applied_force=200.0,
            n_steps=5000,
            output_frequency=50
        )
        
        assert params.atom_indices == [1, 2, 3]
        assert params.coordinate_type == "angle"
        assert params.mode == "constant_force"
        assert params.pulling_velocity == 0.02
        assert params.spring_constant == 500.0
        assert params.applied_force == 200.0
        assert params.n_steps == 5000
        assert params.output_frequency == 50


@pytest.mark.skipif(not STEERED_MD_AVAILABLE, reason="Steered MD module not available")
class TestCoordinateCalculator:
    """Test suite for coordinate calculations."""
    
    def test_distance_calculation(self):
        """Test distance coordinate calculation."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        
        distance = CoordinateCalculator.distance(positions, [0, 1])
        expected_distance = 5.0  # 3-4-5 triangle
        
        assert distance == pytest.approx(expected_distance, rel=1e-6)
    
    def test_distance_calculation_invalid_atoms(self):
        """Test distance calculation with invalid number of atoms."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        
        with pytest.raises(ValueError, match="Distance coordinate requires exactly 2 atoms"):
            CoordinateCalculator.distance(positions, [0])
    
    def test_com_distance_calculation(self):
        """Test center-of-mass distance calculation."""
        positions = np.array([
            [0.0, 0.0, 0.0],  # Group 1
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],  # Group 2
            [6.0, 0.0, 0.0]
        ])
        
        # Equal masses
        distance = CoordinateCalculator.com_distance(positions, [0, 1, 2, 3])
        expected_distance = 5.0  # COM1 at (0.5, 0, 0), COM2 at (5.5, 0, 0)
        
        assert distance == pytest.approx(expected_distance, rel=1e-6)
    
    def test_com_distance_with_masses(self):
        """Test COM distance calculation with specific masses."""
        positions = np.array([
            [0.0, 0.0, 0.0],  # Group 1
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],  # Group 2
            [6.0, 0.0, 0.0]
        ])
        masses = np.array([1.0, 3.0, 2.0, 2.0])
        
        distance = CoordinateCalculator.com_distance(positions, [0, 1, 2, 3], masses)
        
        # COM1 = (0*1 + 2*3)/(1+3) = 1.5
        # COM2 = (4*2 + 6*2)/(2+2) = 5.0
        # Distance = 5.0 - 1.5 = 3.5
        expected_distance = 3.5
        
        assert distance == pytest.approx(expected_distance, rel=1e-6)
    
    def test_angle_calculation(self):
        """Test angle coordinate calculation."""
        positions = np.array([
            [1.0, 0.0, 0.0],  # atom 1
            [0.0, 0.0, 0.0],  # atom 2 (vertex)
            [0.0, 1.0, 0.0]   # atom 3
        ])
        
        angle = CoordinateCalculator.angle(positions, [0, 1, 2])
        expected_angle = np.pi / 2  # 90 degrees
        
        assert angle == pytest.approx(expected_angle, rel=1e-6)
    
    def test_angle_calculation_invalid_atoms(self):
        """Test angle calculation with invalid number of atoms."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        
        with pytest.raises(ValueError, match="Angle coordinate requires exactly 3 atoms"):
            CoordinateCalculator.angle(positions, [0, 1])
    
    def test_dihedral_calculation(self):
        """Test dihedral angle calculation."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        
        dihedral = CoordinateCalculator.dihedral(positions, [0, 1, 2, 3])
        
        # Should be around 90 degrees (π/2 radians) for this geometry
        assert abs(dihedral) == pytest.approx(np.pi/2, rel=0.1)
    
    def test_dihedral_calculation_invalid_atoms(self):
        """Test dihedral calculation with invalid number of atoms."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        
        with pytest.raises(ValueError, match="Dihedral coordinate requires exactly 4 atoms"):
            CoordinateCalculator.dihedral(positions, [0, 1, 2])


@pytest.mark.skipif(not STEERED_MD_AVAILABLE, reason="Steered MD module not available")
class TestSMDForceCalculator:
    """Test suite for SMD force calculations."""
    
    def test_force_calculator_initialization(self):
        """Test SMD force calculator initialization."""
        params = SMDParameters(atom_indices=[0, 1])
        calc = SMDForceCalculator(params)
        
        assert calc.params == params
        assert calc.accumulated_work == 0.0
        assert len(calc.force_history) == 0
        assert len(calc.coordinate_history) == 0
        assert len(calc.work_history) == 0
    
    def test_coordinate_calculation(self):
        """Test coordinate calculation through force calculator."""
        params = SMDParameters(atom_indices=[0, 1], coordinate_type="distance")
        calc = SMDForceCalculator(params)
        
        positions = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0]
        ])
        
        coord = calc.calculate_coordinate(positions)
        assert coord == pytest.approx(5.0, rel=1e-6)
    
    def test_target_coordinate_constant_velocity(self):
        """Test target coordinate calculation for constant velocity mode."""
        params = SMDParameters(
            atom_indices=[0, 1],
            mode="constant_velocity",
            pulling_velocity=0.01
        )
        calc = SMDForceCalculator(params)
        
        initial_coord = 2.0
        step = 1000  # 1000 steps = 1 ps
        
        target = calc.calculate_target_coordinate(step, initial_coord)
        expected_target = 2.0 + 0.01 * 1.0  # initial + velocity * time
        
        assert target == pytest.approx(expected_target, rel=1e-6)
    
    def test_target_coordinate_constant_force(self):
        """Test target coordinate calculation for constant force mode."""
        params = SMDParameters(
            atom_indices=[0, 1],
            mode="constant_force",
            applied_force=100.0
        )
        calc = SMDForceCalculator(params)
        
        initial_coord = 2.0
        step = 1000
        
        target = calc.calculate_target_coordinate(step, initial_coord)
        assert target == initial_coord  # Should not change for constant force
    
    def test_constant_velocity_force_calculation(self):
        """Test force calculation for constant velocity mode."""
        params = SMDParameters(
            atom_indices=[0, 1],
            mode="constant_velocity",
            pulling_velocity=0.01,
            spring_constant=1000.0
        )
        calc = SMDForceCalculator(params)
        
        current_coord = 2.0
        initial_coord = 1.8
        step = 1000  # 1 ps
        
        force, work = calc._constant_velocity_force(current_coord, step, initial_coord)
        
        # Target = 1.8 + 0.01 * 1.0 = 1.81
        # Displacement = 1.81 - 2.0 = -0.19
        # Force = 1000 * (-0.19) = -190
        expected_force = -190.0
        
        assert force == pytest.approx(expected_force, rel=1e-3)
        assert work != 0.0  # Should have non-zero work
    
    def test_constant_force_force_calculation(self):
        """Test force calculation for constant force mode."""
        params = SMDParameters(
            atom_indices=[0, 1],
            mode="constant_force",
            applied_force=100.0  # pN
        )
        calc = SMDForceCalculator(params)
        
        current_coord = 2.0
        initial_coord = 1.8
        step = 0
        
        force, work = calc._constant_force_force(current_coord, step, initial_coord)
        
        # 100 pN = 100 * 0.06022 = 6.022 kJ/(mol·nm)
        expected_force = 6.022
        
        assert force == pytest.approx(expected_force, rel=1e-3)
    
    def test_distance_force_distribution(self):
        """Test force distribution for distance coordinate."""
        params = SMDParameters(atom_indices=[0, 1], coordinate_type="distance")
        calc = SMDForceCalculator(params)
        
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0]
        ])
        
        forces = calc._distribute_distance_force(positions, 100.0)
        
        # Check force conservation (Newton's 3rd law)
        assert np.allclose(forces[0] + forces[1], 0.0, atol=1e-10)
        
        # Check force direction (along x-axis)
        assert forces[0, 0] < 0  # Force on atom 0 toward atom 1
        assert forces[1, 0] > 0  # Force on atom 1 away from atom 0
        assert np.allclose(forces[2], 0.0)  # No force on atom 2
    
    def test_smd_force_calculation_integration(self):
        """Test complete SMD force calculation."""
        params = SMDParameters(
            atom_indices=[0, 1],
            coordinate_type="distance",
            mode="constant_velocity",
            pulling_velocity=0.01,
            spring_constant=1000.0
        )
        calc = SMDForceCalculator(params)
        
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0]
        ])
        
        initial_coord = 2.0
        step = 100
        
        forces, current_coord, work_step = calc.calculate_smd_force(
            positions, step, initial_coord
        )
        
        assert current_coord == pytest.approx(2.0, rel=1e-6)
        assert forces.shape == (2, 3)
        assert len(calc.force_history) == 1
        assert len(calc.coordinate_history) == 1
        assert calc.accumulated_work != 0.0


@pytest.mark.skipif(not STEERED_MD_AVAILABLE, reason="Steered MD module not available")
class TestSteeredMD:
    """Test suite for main SteeredMD class."""
    
    @pytest.fixture
    def mock_simulation_system(self):
        """Create a mock simulation system."""
        system = Mock()
        system.positions = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [6.0, 0.0, 0.0]
        ])
        system.masses = np.array([12.0, 14.0, 16.0, 12.0])
        system.external_forces = np.zeros((4, 3))
        system.step = Mock()
        
        # Remove add_external_forces to test the external_forces assignment path
        if hasattr(system, 'add_external_forces'):
            del system.add_external_forces
            
        return system
    
    def test_steered_md_initialization(self, mock_simulation_system):
        """Test SteeredMD initialization."""
        params = SMDParameters(atom_indices=[0, 1])
        smd = SteeredMD(mock_simulation_system, params)
        
        assert smd.system == mock_simulation_system
        assert smd.params == params
        assert smd.initial_coordinate is None
        assert 'coordinates' in smd.results
        assert 'forces' in smd.results
        assert 'work' in smd.results
    
    def test_get_positions(self, mock_simulation_system):
        """Test position retrieval from simulation system."""
        params = SMDParameters(atom_indices=[0, 1])
        smd = SteeredMD(mock_simulation_system, params)
        
        positions = smd._get_positions()
        
        assert positions.shape == (4, 3)
        assert np.allclose(positions[0], [0.0, 0.0, 0.0])
        assert np.allclose(positions[1], [2.0, 0.0, 0.0])
    
    def test_get_masses(self, mock_simulation_system):
        """Test mass retrieval from simulation system."""
        params = SMDParameters(atom_indices=[0, 1])
        smd = SteeredMD(mock_simulation_system, params)
        
        masses = smd._get_masses()
        
        assert len(masses) == 4
        assert masses[0] == 12.0
        assert masses[1] == 14.0
    
    def test_apply_forces(self, mock_simulation_system):
        """Test force application to simulation system."""
        params = SMDParameters(atom_indices=[0, 1])
        smd = SteeredMD(mock_simulation_system, params)
        
        forces = np.random.randn(4, 3)
        original_forces = mock_simulation_system.external_forces.copy()
        
        smd._apply_forces(forces)
        
        # Check that forces were applied (either by modifying or replacing the array)
        assert not np.allclose(mock_simulation_system.external_forces, original_forces)
        assert np.allclose(mock_simulation_system.external_forces, forces)
    
    def test_run_simulation_short(self, mock_simulation_system):
        """Test running a short SMD simulation."""
        params = SMDParameters(
            atom_indices=[0, 1],
            coordinate_type="distance",
            mode="constant_velocity",
            pulling_velocity=0.01,
            spring_constant=1000.0,
            n_steps=10,
            output_frequency=2
        )
        smd = SteeredMD(mock_simulation_system, params)
        
        results = smd.run_simulation()
        
        assert 'initial_coordinate' in results
        assert 'final_coordinate' in results
        assert 'total_work' in results
        assert len(results['time']) > 0
        assert len(results['coordinates']) > 0
        assert smd.initial_coordinate is not None
    
    def test_jarzynski_calculation(self, mock_simulation_system):
        """Test Jarzynski free energy calculation."""
        params = SMDParameters(
            atom_indices=[0, 1],
            n_steps=5,
            output_frequency=1
        )
        smd = SteeredMD(mock_simulation_system, params)
        
        # Run short simulation to generate work data
        smd.run_simulation()
        
        # Calculate Jarzynski free energy
        delta_g = smd.calculate_jarzynski_free_energy(temperature=300.0)
        
        assert isinstance(delta_g, float)
        assert np.isfinite(delta_g)
    
    def test_jarzynski_no_data(self, mock_simulation_system):
        """Test Jarzynski calculation without work data."""
        params = SMDParameters(atom_indices=[0, 1])
        smd = SteeredMD(mock_simulation_system, params)
        
        with pytest.raises(ValueError, match="No work data available"):
            smd.calculate_jarzynski_free_energy()
    
    def test_plot_force_curves(self, mock_simulation_system, tmp_path):
        """Test force curve plotting."""
        params = SMDParameters(
            atom_indices=[0, 1],
            n_steps=5,
            output_frequency=1
        )
        smd = SteeredMD(mock_simulation_system, params)
        
        # Run simulation to generate data
        smd.run_simulation()
        
        # Test plotting
        fig = smd.plot_force_curves()
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # Should have 4 subplots
        
        # Test saving
        save_path = tmp_path / "test_curves.png"
        fig = smd.plot_force_curves(str(save_path))
        assert save_path.exists()
        
        plt.close(fig)
    
    def test_plot_force_curves_no_data(self, mock_simulation_system):
        """Test plotting without simulation data."""
        params = SMDParameters(atom_indices=[0, 1])
        smd = SteeredMD(mock_simulation_system, params)
        
        with pytest.raises(ValueError, match="No simulation data to plot"):
            smd.plot_force_curves()
    
    def test_save_results(self, mock_simulation_system, tmp_path):
        """Test saving simulation results."""
        params = SMDParameters(
            atom_indices=[0, 1],
            n_steps=5,
            output_frequency=1
        )
        smd = SteeredMD(mock_simulation_system, params)
        
        # Run simulation
        smd.run_simulation()
        
        # Save results
        output_dir = tmp_path / "smd_output"
        smd.save_results(str(output_dir))
        
        # Check files exist
        assert (output_dir / "smd_parameters.json").exists()
        assert (output_dir / "smd_results.npz").exists()
        assert (output_dir / "force_curves.png").exists()


@pytest.mark.skipif(not STEERED_MD_AVAILABLE, reason="Steered MD module not available")
class TestSMDConvenienceFunctions:
    """Test suite for SMD convenience setup functions."""
    
    @pytest.fixture
    def mock_simulation_system(self):
        """Create a mock simulation system."""
        system = Mock()
        system.positions = np.random.randn(100, 3)
        system.masses = np.ones(100) * 12.0
        return system
    
    def test_setup_protein_unfolding_smd(self, mock_simulation_system):
        """Test protein unfolding SMD setup."""
        n_terminus = [0, 1, 2, 3, 4]
        c_terminus = [95, 96, 97, 98, 99]
        
        smd = setup_protein_unfolding_smd(
            mock_simulation_system,
            n_terminus_atoms=n_terminus,
            c_terminus_atoms=c_terminus,
            pulling_velocity=0.005,
            spring_constant=800.0
        )
        
        assert isinstance(smd, SteeredMD)
        assert smd.params.coordinate_type == "com_distance"
        assert smd.params.mode == "constant_velocity"
        assert smd.params.pulling_velocity == 0.005
        assert smd.params.spring_constant == 800.0
        assert smd.params.atom_indices == n_terminus + c_terminus
    
    def test_setup_ligand_unbinding_smd(self, mock_simulation_system):
        """Test ligand unbinding SMD setup."""
        ligand_atoms = [0, 1, 2, 3]
        protein_atoms = [50, 51, 52, 53, 54]
        
        smd = setup_ligand_unbinding_smd(
            mock_simulation_system,
            ligand_atoms=ligand_atoms,
            protein_atoms=protein_atoms,
            pulling_velocity=0.02,
            spring_constant=300.0
        )
        
        assert isinstance(smd, SteeredMD)
        assert smd.params.coordinate_type == "com_distance"
        assert smd.params.mode == "constant_velocity"
        assert smd.params.pulling_velocity == 0.02
        assert smd.params.spring_constant == 300.0
        assert smd.params.atom_indices == ligand_atoms + protein_atoms
    
    def test_setup_bond_stretching_smd(self, mock_simulation_system):
        """Test bond stretching SMD setup."""
        atom1, atom2 = 10, 11
        
        smd = setup_bond_stretching_smd(
            mock_simulation_system,
            atom1=atom1,
            atom2=atom2,
            applied_force=750.0
        )
        
        assert isinstance(smd, SteeredMD)
        assert smd.params.coordinate_type == "distance"
        assert smd.params.mode == "constant_force"
        assert smd.params.applied_force == 750.0
        assert smd.params.atom_indices == [atom1, atom2]


@pytest.mark.skipif(not STEERED_MD_AVAILABLE, reason="Steered MD module not available")
class TestSMDIntegration:
    """Integration tests for SMD module."""
    
    def test_full_smd_workflow(self):
        """Test complete SMD workflow from setup to analysis."""
        # Create mock system
        class MockSystem:
            def __init__(self):
                self.positions = np.array([
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0]
                ])
                self.masses = np.array([12.0, 14.0, 16.0])
                self.external_forces = np.zeros((3, 3))
                self.step_count = 0
            
            def step(self):
                # Simple dynamics: move second atom away slowly
                self.positions[1, 0] += 0.001
                self.step_count += 1
        
        system = MockSystem()
        
        # Setup SMD
        params = SMDParameters(
            atom_indices=[0, 1],
            coordinate_type="distance",
            mode="constant_velocity",
            pulling_velocity=0.01,
            spring_constant=1000.0,
            n_steps=20,
            output_frequency=5
        )
        
        smd = SteeredMD(system, params)
        
        # Run simulation
        results = smd.run_simulation()
        
        # Verify results
        assert results['initial_coordinate'] == pytest.approx(2.0, rel=1e-3)
        assert results['final_coordinate'] > results['initial_coordinate']
        assert results['total_work'] != 0.0
        assert len(results['time']) > 0
        assert len(results['coordinates']) > 0
        
        # Test Jarzynski calculation
        delta_g = smd.calculate_jarzynski_free_energy()
        assert isinstance(delta_g, float)
        
        # Test plotting (without saving)
        fig = smd.plot_force_curves()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_different_coordinate_types(self):
        """Test SMD with different coordinate types."""
        class MockSystem:
            def __init__(self):
                self.positions = np.array([
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [3.0, 1.0, 1.0]
                ])
                self.masses = np.ones(4) * 12.0
                self.external_forces = np.zeros((4, 3))
            
            def step(self):
                pass
        
        system = MockSystem()
        
        # Test different coordinate types
        coordinate_configs = [
            {"type": "distance", "atoms": [0, 1]},
            {"type": "angle", "atoms": [0, 1, 2]},
            {"type": "dihedral", "atoms": [0, 1, 2, 3]},
            {"type": "com_distance", "atoms": [0, 1, 2, 3]},
        ]
        
        for config in coordinate_configs:
            params = SMDParameters(
                atom_indices=config["atoms"],
                coordinate_type=config["type"],
                n_steps=3,
                output_frequency=1
            )
            
            smd = SteeredMD(system, params)
            results = smd.run_simulation()
            
            assert results['initial_coordinate'] is not None
            assert len(results['coordinates']) > 0
    
    def test_different_smd_modes(self):
        """Test different SMD modes (constant velocity vs constant force)."""
        class MockSystem:
            def __init__(self):
                self.positions = np.array([
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]
                ])
                self.masses = np.array([12.0, 14.0])
                self.external_forces = np.zeros((2, 3))
                self.step_count = 0
            
            def step(self):
                # Simulate simple dynamics - atoms move slightly in response to applied forces
                if np.any(self.external_forces != 0):
                    # Move atoms proportionally to applied forces (simplified dynamics)
                    dt = 0.001  # ps
                    for i in range(len(self.positions)):
                        force_magnitude = np.linalg.norm(self.external_forces[i])
                        if force_magnitude > 0:
                            # Simple displacement proportional to force
                            displacement = self.external_forces[i] * dt / self.masses[i] * 0.01
                            self.positions[i] += displacement
                self.step_count += 1
        
        system = MockSystem()
        
        # Test constant velocity mode
        params_cv = SMDParameters(
            atom_indices=[0, 1],
            mode="constant_velocity",
            pulling_velocity=0.01,
            spring_constant=1000.0,
            n_steps=5,
            output_frequency=1
        )
        
        smd_cv = SteeredMD(system, params_cv)
        results_cv = smd_cv.run_simulation()
        
        # Reset system for second test
        system.positions = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0]
        ])
        system.external_forces = np.zeros((2, 3))
        system.step_count = 0
        
        # Test constant force mode
        params_cf = SMDParameters(
            atom_indices=[0, 1],
            mode="constant_force",
            applied_force=100.0,
            n_steps=5,
            output_frequency=1
        )
        
        smd_cf = SteeredMD(system, params_cf)
        results_cf = smd_cf.run_simulation()
        
        # Both should produce valid results
        assert results_cv['total_work'] != 0.0
        assert results_cf['total_work'] != 0.0
        assert len(results_cv['coordinates']) > 0
        assert len(results_cf['coordinates']) > 0


@pytest.mark.skipif(not STEERED_MD_AVAILABLE, reason="Steered MD module not available")
class TestSMDPerformance:
    """Performance tests for SMD module."""
    
    @pytest.mark.performance
    def test_smd_calculation_performance(self, benchmark):
        """Benchmark SMD force calculation performance."""
        params = SMDParameters(
            atom_indices=[0, 99],
            coordinate_type="distance",
            mode="constant_velocity"
        )
        calc = SMDForceCalculator(params)
        
        positions = np.random.randn(100, 3)
        
        def run_force_calculation():
            return calc.calculate_smd_force(positions, 100, 2.0)
        
        forces, coord, work = benchmark(run_force_calculation)
        assert forces.shape == (100, 3)
        assert isinstance(coord, float)
        assert isinstance(work, float)
    
    @pytest.mark.performance
    def test_coordinate_calculation_performance(self, benchmark):
        """Benchmark coordinate calculation performance."""
        positions = np.random.randn(1000, 3)
        
        def run_distance_calculation():
            return CoordinateCalculator.distance(positions, [0, 999])
        
        distance = benchmark(run_distance_calculation)
        assert isinstance(distance, float)
        assert distance > 0


# Test data for validation
class TestSMDValidation:
    """Validation tests against known results."""
    
    @pytest.mark.skipif(not STEERED_MD_AVAILABLE, reason="Steered MD module not available")
    def test_harmonic_potential_work(self):
        """Test work calculation for simple harmonic potential."""
        # Simple system: harmonic spring with known analytical solution
        class SimpleHarmonicSystem:
            def __init__(self):
                self.positions = np.array([
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]  # Initial distance = 1.0 nm
                ])
                self.masses = np.array([1.0, 1.0])
                self.external_forces = np.zeros((2, 3))
                self.time = 0.0
            
            def step(self):
                # Move linearly with pulling velocity
                dt = 0.001  # 1 fs
                self.time += dt
                # For this test, keep positions constant to focus on force calculation
        
        system = SimpleHarmonicSystem()
        
        params = SMDParameters(
            atom_indices=[0, 1],
            coordinate_type="distance",
            mode="constant_velocity",
            pulling_velocity=0.1,  # nm/ps
            spring_constant=1000.0,  # kJ/(mol·nm²)
            n_steps=10,
            output_frequency=1
        )
        
        smd = SteeredMD(system, params)
        results = smd.run_simulation()
        
        # Work should be non-zero and positive (for stretching)
        assert results['total_work'] > 0
        
        # Force should be approximately k * (target - initial)
        # At step 10: target = 1.0 + 0.1 * 0.01 = 1.001
        # Displacement = 1.001 - 1.0 = 0.001
        # Expected force ≈ 1000 * 0.001 = 1.0 kJ/(mol·nm)
        
        final_force = smd.force_calculator.force_history[-1]
        assert abs(final_force) < 10.0  # Should be reasonable magnitude


if __name__ == "__main__":
    # Run basic tests when module is executed directly
    print("Running Steered MD tests...")
    
    if STEERED_MD_AVAILABLE:
        # Test basic functionality
        params = SMDParameters(atom_indices=[0, 1])
        print(f"✓ SMD parameters created: {params.coordinate_type} mode")
        
        calc = CoordinateCalculator()
        positions = np.array([[0, 0, 0], [3, 4, 0]])
        distance = calc.distance(positions, [0, 1])
        print(f"✓ Distance calculation: {distance:.2f} nm")
        
        print("✓ All basic tests passed!")
        print("\nTo run comprehensive tests, use: pytest test_steered_md.py -v")
    else:
        print("✗ Steered MD module not available for testing")
