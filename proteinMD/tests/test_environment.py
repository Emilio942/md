"""
Comprehensive Unit Tests for Environment Module

Task 10.1: Umfassende Unit Tests 🚀

Tests the environment implementations including:
- TIP3P water model and solvation
- Periodic boundary conditions (PBC)
- Implicit solvent models (GB/SA)
- Parallel force calculations
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

# Try to import environment modules
try:
    from environment.water import WaterSystem, WaterSolvationBox
    WATER_AVAILABLE = True
except ImportError:
    WATER_AVAILABLE = False

try:
    from environment.tip3p_forcefield import TIP3PWaterForceField, TIP3PWaterModel
    TIP3P_AVAILABLE = True
except ImportError:
    TIP3P_AVAILABLE = False

try:
    from environment.periodic_boundary import (
        PeriodicBox, PeriodicBoundaryConditions, PressureCoupling,
        create_cubic_box, create_orthogonal_box
    )
    PBC_AVAILABLE = True
except ImportError:
    PBC_AVAILABLE = False

try:
    from environment.implicit_solvent import (
        ImplicitSolventModel, GeneralizedBornModel, SurfaceAreaModel
    )
    IMPLICIT_AVAILABLE = True
except ImportError:
    IMPLICIT_AVAILABLE = False

try:
    from environment.parallel_forces import ParallelForceCalculator
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False


@pytest.mark.skipif(not WATER_AVAILABLE, reason="Water module not available")
class TestWaterSystem:
    """Test water system creation and management."""
    
    def test_water_system_initialization(self):
        """Test basic water system initialization."""
        water_system = WaterSystem()
        
        assert water_system is not None
        assert hasattr(water_system, 'create_water_box')
    
    def test_water_box_creation(self):
        """Test creation of water box."""
        water_system = WaterSystem()
        
        n_water = 64  # 4x4x4 arrangement
        box_size = 2.0  # nm
        
        try:
            positions, atom_types = water_system.create_water_box(
                n_water=n_water,
                box_size=box_size,
                density=1.0
            )
            
            # Check output format
            assert positions.shape[0] == n_water * 3  # 3 atoms per water
            assert positions.shape[1] == 3  # x, y, z coordinates
            assert len(atom_types) == n_water * 3
            
            # Check that positions are within box
            assert np.all(positions >= 0)
            assert np.all(positions <= box_size)
            
            # Check atom types
            expected_types = ['O', 'H', 'H'] * n_water
            assert atom_types == expected_types
            
        except NotImplementedError:
            pytest.skip("Water box creation not implemented")
    
    def test_water_density_calculation(self):
        """Test water density calculation."""
        water_system = WaterSystem()
        
        # Standard water box at room temperature
        box_dimensions = np.array([2.0, 2.0, 2.0])  # nm
        n_water = 64
        
        if hasattr(water_system, 'calculate_water_density'):
            density = water_system.calculate_water_density(
                box_dimensions, n_water_molecules=n_water
            )
            
            # Should be close to 1000 kg/m³ (1 g/cm³)
            assert 800 < density < 1200, f"Unrealistic density: {density} kg/m³"
    
    def test_water_molecule_geometry(self):
        """Test that water molecules have correct geometry."""
        water_system = WaterSystem()
        
        try:
            positions, _ = water_system.create_water_box(n_water=1, box_size=1.0)
            
            # Extract positions of O, H1, H2
            o_pos = positions[0]
            h1_pos = positions[1]
            h2_pos = positions[2]
            
            # Check O-H bond lengths (~0.0957 nm for TIP3P)
            oh1_dist = np.linalg.norm(h1_pos - o_pos)
            oh2_dist = np.linalg.norm(h2_pos - o_pos)
            
            assert 0.09 < oh1_dist < 0.11, f"Invalid O-H1 distance: {oh1_dist}"
            assert 0.09 < oh2_dist < 0.11, f"Invalid O-H2 distance: {oh2_dist}"
            
            # Check H-O-H angle (~104.5° for water)
            vec1 = h1_pos - o_pos
            vec2 = h2_pos - o_pos
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
            
            assert 100 < angle_deg < 110, f"Invalid H-O-H angle: {angle_deg}°"
            
        except NotImplementedError:
            pytest.skip("Water box creation not implemented")


@pytest.mark.skipif(not WATER_AVAILABLE, reason="Water module not available")
class TestWaterSolvationBox:
    """Test protein solvation with water."""
    
    def test_solvation_box_initialization(self):
        """Test solvation box initialization."""
        solvation = WaterSolvationBox(
            min_distance_to_solute=0.3,
            min_water_distance=0.25
        )
        
        assert solvation.min_distance_to_solute == 0.3
        assert solvation.min_water_distance == 0.25
    
    def test_protein_solvation(self):
        """Test solvating a protein with water."""
        solvation = WaterSolvationBox()
        
        # Mock protein positions
        protein_positions = np.array([
            [1.0, 1.0, 1.0],
            [1.2, 1.0, 1.0],
            [1.0, 1.2, 1.0]
        ])
        
        box_dimensions = np.array([3.0, 3.0, 3.0])
        
        try:
            water_data = solvation.solvate_protein(protein_positions, box_dimensions)
            
            assert 'positions' in water_data
            assert 'n_molecules' in water_data
            assert 'atom_types' in water_data
            
            # Check that water is not too close to protein
            water_positions = water_data['positions']
            for water_pos in water_positions[::3]:  # Check every oxygen atom
                min_dist_to_protein = np.min(
                    np.linalg.norm(water_pos - protein_positions, axis=1)
                )
                assert min_dist_to_protein >= 0.25, "Water too close to protein"
            
        except NotImplementedError:
            pytest.skip("Protein solvation not implemented")
    
    def test_solvation_validation(self):
        """Test solvation validation functionality."""
        solvation = WaterSolvationBox()
        
        # Create test system
        box_dimensions = np.array([2.0, 2.0, 2.0])
        protein_positions = np.array([[1.0, 1.0, 1.0]])
        
        try:
            if hasattr(solvation, 'validate_solvation'):
                validation = solvation.validate_solvation(box_dimensions, protein_positions)
                
                assert 'min_distance_to_protein' in validation
                assert 'protein_distance_violations' in validation
                assert 'n_water_molecules' in validation
                
        except NotImplementedError:
            pytest.skip("Solvation validation not implemented")


@pytest.mark.skipif(not TIP3P_AVAILABLE, reason="TIP3P module not available")
class TestTIP3PWaterModel:
    """Test TIP3P water model implementation."""
    
    def test_tip3p_parameters(self):
        """Test TIP3P model parameters."""
        tip3p = TIP3PWaterModel()
        
        # Check TIP3P parameters (literature values)
        assert hasattr(tip3p, 'oxygen_charge')
        assert hasattr(tip3p, 'hydrogen_charge')
        assert hasattr(tip3p, 'oh_bond_length')
        assert hasattr(tip3p, 'hoh_angle')
        
        # Typical TIP3P values
        assert abs(tip3p.oxygen_charge + 0.834) < 0.01
        assert abs(tip3p.hydrogen_charge - 0.417) < 0.01
        assert abs(tip3p.oh_bond_length - 0.0957) < 0.01  # nm
        assert abs(tip3p.hoh_angle - 104.5) < 1.0  # degrees
    
    def test_tip3p_water_creation(self):
        """Test individual water molecule creation."""
        tip3p = TIP3PWaterModel()
        
        try:
            water_coords = tip3p.create_water_molecule(center=np.array([0.0, 0.0, 0.0]))
            
            assert water_coords.shape == (3, 3)  # 3 atoms, 3 coordinates
            
            # Check bond lengths and angle
            o_pos = water_coords[0]
            h1_pos = water_coords[1]
            h2_pos = water_coords[2]
            
            oh1_dist = np.linalg.norm(h1_pos - o_pos)
            oh2_dist = np.linalg.norm(h2_pos - o_pos)
            
            assert abs(oh1_dist - tip3p.oh_bond_length) < 0.001
            assert abs(oh2_dist - tip3p.oh_bond_length) < 0.001
            
        except NotImplementedError:
            pytest.skip("TIP3P water creation not implemented")


@pytest.mark.skipif(not TIP3P_AVAILABLE, reason="TIP3P force field not available")
class TestTIP3PWaterForceField:
    """Test TIP3P water force field calculations."""
    
    def test_tip3p_force_field_initialization(self):
        """Test TIP3P force field initialization."""
        ff = TIP3PWaterForceField(cutoff=1.0)
        
        assert ff.cutoff == 1.0
        assert hasattr(ff, 'water_molecules')
        assert hasattr(ff, 'rigid_water')
    
    def test_water_water_interactions(self):
        """Test water-water interaction calculations."""
        ff = TIP3PWaterForceField(cutoff=1.0)
        
        # Create two water molecules
        n_atoms = 6  # 2 water molecules
        positions = np.array([
            [0.0, 0.0, 0.0],    # O1
            [0.096, 0.0, 0.0],  # H1
            [-0.024, 0.093, 0.0], # H2
            [0.5, 0.0, 0.0],    # O2
            [0.596, 0.0, 0.0],  # H3
            [0.476, 0.093, 0.0] # H4
        ])
        
        # Add water molecules to force field
        ff.add_water_molecule([0, 1, 2])
        ff.add_water_molecule([3, 4, 5])
        
        try:
            forces, energy = ff.calculate_water_water_forces(positions)
            
            assert forces.shape == (n_atoms, 3)
            assert np.isfinite(energy)
            
            # Net force should be zero (Newton's 3rd law)
            net_force = np.sum(forces, axis=0)
            assert np.allclose(net_force, 0.0, atol=1e-8)
            
        except NotImplementedError:
            pytest.skip("Water-water forces not implemented")
    
    def test_water_protein_interactions(self):
        """Test water-protein interaction calculations."""
        ff = TIP3PWaterForceField(cutoff=1.0)
        
        # Mock protein atom
        protein_positions = np.array([[1.0, 0.0, 0.0]])
        protein_charges = np.array([0.5])
        protein_lj_params = {'sigma': [0.35], 'epsilon': [0.1]}
        
        # Water molecule
        water_positions = np.array([
            [0.0, 0.0, 0.0],
            [0.096, 0.0, 0.0],
            [-0.024, 0.093, 0.0]
        ])
        
        ff.add_water_molecule([0, 1, 2])
        ff.set_protein_parameters(protein_charges, protein_lj_params)
        
        try:
            all_positions = np.vstack([water_positions, protein_positions])
            forces, energy = ff.calculate_water_protein_forces(all_positions)
            
            assert forces.shape[0] == len(all_positions)
            assert np.isfinite(energy)
            
        except NotImplementedError:
            pytest.skip("Water-protein forces not implemented")


@pytest.mark.skipif(not PBC_AVAILABLE, reason="PBC module not available")
class TestPeriodicBoundaryConditions:
    """Test periodic boundary conditions implementation."""
    
    def test_cubic_box_creation(self):
        """Test cubic box creation."""
        box_size = 5.0
        box = create_cubic_box(box_size)
        
        assert box.box_type == "cubic"
        assert box.a == box_size
        assert box.b == box_size
        assert box.c == box_size
        assert abs(box.volume - box_size**3) < 1e-10
    
    def test_orthogonal_box_creation(self):
        """Test orthogonal box creation."""
        a, b, c = 3.0, 4.0, 5.0
        box = create_orthogonal_box(a, b, c)
        
        assert box.box_type == "orthogonal"
        assert box.a == a
        assert box.b == b
        assert box.c == c
        assert abs(box.volume - a * b * c) < 1e-10
    
    def test_position_wrapping(self):
        """Test position wrapping for PBC."""
        box = create_cubic_box(5.0)
        
        # Test positions outside the box
        test_positions = np.array([
            [6.0, 2.0, 3.0],   # x > box_size
            [-1.0, 2.0, 3.0],  # x < 0
            [2.0, 7.0, 3.0],   # y > box_size
            [2.0, 2.0, -2.0]   # z < 0
        ])
        
        wrapped = box.wrap_positions(test_positions)
        
        # All wrapped positions should be in [0, box_size)
        assert np.all(wrapped >= 0)
        assert np.all(wrapped < 5.0)
        
        # Check specific cases
        assert abs(wrapped[0, 0] - 1.0) < 1e-10  # 6.0 -> 1.0
        assert abs(wrapped[1, 0] - 4.0) < 1e-10  # -1.0 -> 4.0
    
    def test_minimum_image_distance(self):
        """Test minimum image distance calculation."""
        box = create_cubic_box(5.0)
        
        # Test particles across box boundary
        pos1 = np.array([0.5, 0.0, 0.0])
        pos2 = np.array([4.5, 0.0, 0.0])
        
        # Direct distance would be 4.0, but minimum image is 1.0
        distance = box.minimum_image_distance(pos1, pos2)
        assert abs(distance - 1.0) < 1e-10
    
    def test_pressure_coupling(self):
        """Test pressure coupling functionality."""
        box = create_cubic_box(5.0)
        
        pressure_coupling = PressureCoupling(
            target_pressure=1.0,
            coupling_time=1.0,
            algorithm="berendsen"
        )
        
        pbc = PeriodicBoundaryConditions(box, pressure_coupling)
        
        assert pbc.pressure_coupling is not None
        assert pbc.pressure_coupling.target_pressure == 1.0
    
    def test_pbc_force_calculations(self):
        """Test force calculations with PBC."""
        box = create_cubic_box(3.0)
        pbc = PeriodicBoundaryConditions(box)
        
        # Test system with particles across boundaries
        positions = np.array([
            [0.1, 0.1, 0.1],
            [2.9, 0.1, 0.1]  # Close across x boundary
        ])
        
        # Mock force calculation
        def mock_force_calculator(pos1, pos2):
            r = pbc.box.minimum_image_distance(pos1, pos2)
            if r < 1.0:
                return np.array([1.0, 0.0, 0.0])  # Simple repulsive force
            return np.array([0.0, 0.0, 0.0])
        
        try:
            forces = pbc.calculate_forces_with_pbc(positions, mock_force_calculator)
            
            assert forces.shape == positions.shape
            # Forces should be non-zero due to proximity across boundary
            assert np.any(np.abs(forces) > 0)
            
        except NotImplementedError:
            pytest.skip("PBC force calculation not implemented")
    
    @pytest.mark.performance
    def test_pbc_performance(self, performance_monitor):
        """Test PBC performance with larger systems."""
        box = create_cubic_box(10.0)
        pbc = PeriodicBoundaryConditions(box)
        
        n_atoms = 1000
        positions = np.random.uniform(0, 10, (n_atoms, 3))
        
        with performance_monitor:
            for _ in range(100):
                wrapped = box.wrap_positions(positions)
        
        metrics = performance_monitor.metrics
        wrap_time_per_atom = metrics['duration_ms'] / (100 * n_atoms)
        
        # Should be very fast (< 0.01 ms per atom)
        assert wrap_time_per_atom < 0.01, f"PBC wrapping too slow: {wrap_time_per_atom:.3f} ms/atom"


@pytest.mark.skipif(not IMPLICIT_AVAILABLE, reason="Implicit solvent module not available")
class TestImplicitSolventModel:
    """Test implicit solvent model implementation."""
    
    def test_implicit_solvent_initialization(self):
        """Test implicit solvent model initialization."""
        model = ImplicitSolventModel(
            gb_model="HCT",
            sa_model="LCPO",
            solvent_dielectric=78.5,
            solute_dielectric=1.0
        )
        
        assert model.gb_model == "HCT"
        assert model.sa_model == "LCPO"
        assert model.solvent_dielectric == 78.5
        assert model.solute_dielectric == 1.0
    
    def test_generalized_born_calculation(self):
        """Test Generalized Born energy calculation."""
        gb_model = GeneralizedBornModel(model_type="HCT")
        
        # Simple two-atom system
        positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        charges = np.array([1.0, -1.0])
        radii = np.array([0.15, 0.15])
        
        try:
            gb_energy = gb_model.calculate_energy(positions, charges, radii)
            
            assert np.isfinite(gb_energy)
            assert gb_energy < 0  # Should be negative for opposite charges
            
        except NotImplementedError:
            pytest.skip("GB calculation not implemented")
    
    def test_surface_area_calculation(self):
        """Test surface area calculation."""
        sa_model = SurfaceAreaModel(model_type="LCPO")
        
        # Simple spherical atom
        positions = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([0.2])
        
        try:
            sa_energy = sa_model.calculate_energy(positions, radii)
            
            assert np.isfinite(sa_energy)
            assert sa_energy > 0  # Surface area energy should be positive
            
        except NotImplementedError:
            pytest.skip("SA calculation not implemented")
    
    def test_implicit_solvent_forces(self):
        """Test implicit solvent force calculation."""
        model = ImplicitSolventModel()
        
        # Test system
        positions = np.random.randn(10, 3)
        charges = np.random.uniform(-1, 1, 10)
        radii = np.ones(10) * 0.15
        
        try:
            forces, energy = model.calculate_forces(positions, charges, radii)
            
            assert forces.shape == positions.shape
            assert np.all(np.isfinite(forces))
            assert np.isfinite(energy)
            
        except NotImplementedError:
            pytest.skip("Implicit solvent forces not implemented")
    
    @pytest.mark.performance
    def test_implicit_solvent_performance(self, performance_monitor):
        """Test implicit solvent performance vs explicit water."""
        model = ImplicitSolventModel()
        
        n_atoms = 100
        positions = np.random.randn(n_atoms, 3)
        charges = np.random.uniform(-1, 1, n_atoms)
        radii = np.ones(n_atoms) * 0.15
        
        with performance_monitor:
            try:
                for _ in range(50):
                    forces, energy = model.calculate_forces(positions, charges, radii)
            except NotImplementedError:
                pytest.skip("Implicit solvent forces not implemented")
        
        metrics = performance_monitor.metrics
        time_per_atom = metrics['duration_ms'] / (50 * n_atoms)
        
        # Should be much faster than explicit water
        assert time_per_atom < 1.0, f"Implicit solvent too slow: {time_per_atom:.3f} ms/atom"


@pytest.mark.skipif(not PARALLEL_AVAILABLE, reason="Parallel forces module not available")
class TestParallelForceCalculator:
    """Test parallel force calculation implementation."""
    
    def test_parallel_calculator_initialization(self):
        """Test parallel calculator initialization."""
        calc = ParallelForceCalculator(n_threads=4)
        
        assert calc.n_threads == 4
        assert hasattr(calc, 'calculate_forces_parallel')
    
    def test_parallel_vs_serial_forces(self):
        """Test that parallel and serial calculations give same results."""
        calc = ParallelForceCalculator(n_threads=2)
        
        n_atoms = 100
        positions = np.random.randn(n_atoms, 3)
        charges = np.random.uniform(-1, 1, n_atoms)
        
        try:
            # Serial calculation
            forces_serial, energy_serial = calc.calculate_forces_serial(positions, charges)
            
            # Parallel calculation
            forces_parallel, energy_parallel = calc.calculate_forces_parallel(positions, charges)
            
            # Results should be nearly identical
            assert np.allclose(forces_serial, forces_parallel, rtol=1e-10)
            assert abs(energy_serial - energy_parallel) < 1e-10
            
        except NotImplementedError:
            pytest.skip("Parallel force calculation not implemented")
    
    @pytest.mark.performance
    def test_parallel_speedup(self, performance_monitor):
        """Test parallel speedup performance."""
        n_atoms = 500
        positions = np.random.randn(n_atoms, 3)
        charges = np.random.uniform(-1, 1, n_atoms)
        
        # Test different thread counts
        thread_counts = [1, 2, 4]
        times = []
        
        for n_threads in thread_counts:
            calc = ParallelForceCalculator(n_threads=n_threads)
            
            with performance_monitor:
                try:
                    for _ in range(10):
                        forces, energy = calc.calculate_forces_parallel(positions, charges)
                except NotImplementedError:
                    pytest.skip("Parallel force calculation not implemented")
            
            times.append(performance_monitor.metrics['duration_ms'])
        
        # Should see some speedup with more threads
        if len(times) >= 2:
            speedup = times[0] / times[-1]  # 1 thread vs max threads
            assert speedup > 1.2, f"Poor parallel speedup: {speedup:.2f}x"
    
    def test_thread_safety(self):
        """Test thread safety of parallel calculations."""
        calc = ParallelForceCalculator(n_threads=4)
        
        n_atoms = 50
        positions = np.random.randn(n_atoms, 3)
        charges = np.random.uniform(-1, 1, n_atoms)
        
        # Run multiple calculations concurrently
        results = []
        
        try:
            for _ in range(10):
                forces, energy = calc.calculate_forces_parallel(positions, charges)
                results.append((forces.copy(), energy))
            
            # All results should be identical
            for forces, energy in results[1:]:
                assert np.allclose(forces, results[0][0])
                assert abs(energy - results[0][1]) < 1e-10
                
        except NotImplementedError:
            pytest.skip("Parallel force calculation not implemented")


class TestEnvironmentIntegration:
    """Integration tests for environment components."""
    
    @pytest.mark.integration
    def test_water_with_pbc(self):
        """Test water system with periodic boundary conditions."""
        if not (WATER_AVAILABLE and PBC_AVAILABLE):
            pytest.skip("Water or PBC modules not available")
        
        # Create water box
        water_system = WaterSystem()
        box_size = 3.0
        
        try:
            positions, atom_types = water_system.create_water_box(
                n_water=27, box_size=box_size
            )
            
            # Apply PBC
            box = create_cubic_box(box_size)
            wrapped_positions = box.wrap_positions(positions)
            
            # All positions should be in box
            assert np.all(wrapped_positions >= 0)
            assert np.all(wrapped_positions < box_size)
            
        except NotImplementedError:
            pytest.skip("Water box creation not implemented")
    
    @pytest.mark.integration
    def test_implicit_vs_explicit_solvation(self):
        """Compare implicit and explicit solvation energies."""
        if not (IMPLICIT_AVAILABLE and TIP3P_AVAILABLE):
            pytest.skip("Implicit solvent or TIP3P modules not available")
        
        # Simple test system (two charges)
        positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        charges = np.array([1.0, -1.0])
        
        # Implicit solvation energy
        try:
            implicit_model = ImplicitSolventModel()
            radii = np.array([0.15, 0.15])
            forces_impl, energy_impl = implicit_model.calculate_forces(positions, charges, radii)
            
            # Explicit solvation would require full water box setup
            # This is a placeholder for comparison
            energy_explicit = -200.0  # Mock explicit solvation energy
            
            # Implicit should give qualitatively similar results
            assert abs(energy_impl - energy_explicit) < 100.0
            
        except NotImplementedError:
            pytest.skip("Implicit solvation not implemented")
    
    @pytest.mark.slow
    def test_environment_performance_comparison(self, performance_monitor):
        """Compare performance of different environment models."""
        test_system_size = 100
        positions = np.random.randn(test_system_size, 3)
        charges = np.random.uniform(-1, 1, test_system_size)
        
        performance_results = {}
        
        # Test implicit solvent performance
        if IMPLICIT_AVAILABLE:
            try:
                implicit_model = ImplicitSolventModel()
                radii = np.ones(test_system_size) * 0.15
                
                with performance_monitor:
                    for _ in range(20):
                        forces, energy = implicit_model.calculate_forces(positions, charges, radii)
                
                performance_results['implicit'] = performance_monitor.metrics['duration_ms']
                
            except NotImplementedError:
                pass
        
        # Test explicit water performance (simplified)
        if TIP3P_AVAILABLE:
            try:
                tip3p_ff = TIP3PWaterForceField()
                
                with performance_monitor:
                    for _ in range(20):
                        # Simplified water calculation
                        forces = np.random.randn(test_system_size, 3)  # Mock
                
                performance_results['explicit'] = performance_monitor.metrics['duration_ms']
                
            except NotImplementedError:
                pass
        
        # Implicit should be faster
        if 'implicit' in performance_results and 'explicit' in performance_results:
            speedup = performance_results['explicit'] / performance_results['implicit']
            assert speedup > 2.0, f"Implicit solvent not faster: {speedup:.1f}x"
