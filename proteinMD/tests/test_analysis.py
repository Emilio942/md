"""
Comprehensive Unit Tests for Analysis Module

Task 10.1: Umfassende Unit Tests 🚀

Tests the analysis modules including:
- RMSD calculations
- Ramachandran plot analysis
- Hydrogen bond analysis
- Radius of gyration calculations
- Secondary structure analysis
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

# Try to import analysis modules
try:
    from analysis.rmsd import RMSDCalculator, calculate_rmsd, align_structures
    RMSD_AVAILABLE = True
except ImportError:
    RMSD_AVAILABLE = False

try:
    from analysis.ramachandran import RamachandranAnalyzer, calculate_dihedral
    RAMACHANDRAN_AVAILABLE = True
except ImportError:
    RAMACHANDRAN_AVAILABLE = False

try:
    from analysis.hydrogen_bonds import HydrogenBondAnalyzer, find_hydrogen_bonds
    HYDROGEN_BONDS_AVAILABLE = True
except ImportError:
    HYDROGEN_BONDS_AVAILABLE = False

try:
    from analysis.radius_of_gyration import RadiusOfGyrationAnalyzer, calculate_radius_of_gyration
    RG_AVAILABLE = True
except ImportError:
    RG_AVAILABLE = False

try:
    from analysis.secondary_structure import SecondaryStructureAnalyzer, assign_secondary_structure
    SS_AVAILABLE = True
except ImportError:
    SS_AVAILABLE = False


@pytest.mark.skipif(not RMSD_AVAILABLE, reason="RMSD module not available")
class TestRMSDAnalysis:
    """Test suite for RMSD analysis functionality."""
    
    def test_rmsd_calculator_initialization(self, mock_protein):
        """Test RMSD calculator initialization."""
        calc = RMSDCalculator(reference_structure=mock_protein)
        assert calc is not None
        assert hasattr(calc, 'reference_structure')
    
    def test_rmsd_calculation_identical_structures(self, mock_protein):
        """Test RMSD calculation for identical structures."""
        calc = RMSDCalculator(reference_structure=mock_protein)
        rmsd = calc.calculate_rmsd(mock_protein)
        assert rmsd == pytest.approx(0.0, abs=1e-6)
    
    def test_rmsd_calculation_different_structures(self, mock_protein):
        """Test RMSD calculation for different structures."""
        calc = RMSDCalculator(reference_structure=mock_protein)
        
        # Create a modified structure
        modified_protein = Mock()
        modified_protein.atoms = [Mock() for _ in range(len(mock_protein.atoms))]
        for i, atom in enumerate(modified_protein.atoms):
            atom.position = np.array([i, i, i], dtype=float)
        
        rmsd = calc.calculate_rmsd(modified_protein)
        assert rmsd > 0.0
    
    def test_structural_alignment(self, mock_protein):
        """Test structural alignment functionality."""
        calc = RMSDCalculator(reference_structure=mock_protein)
        aligned = calc.align_structure(mock_protein)
        assert aligned is not None
    
    def test_rmsd_trajectory_analysis(self, mock_trajectory):
        """Test RMSD analysis over trajectory."""
        calc = RMSDCalculator(reference_structure=mock_trajectory.frames[0])
        rmsd_values = calc.analyze_trajectory(mock_trajectory)
        
        assert len(rmsd_values) == len(mock_trajectory.frames)
        assert all(rmsd >= 0 for rmsd in rmsd_values)


@pytest.mark.skipif(not RAMACHANDRAN_AVAILABLE, reason="Ramachandran module not available")
class TestRamachandranAnalysis:
    """Test suite for Ramachandran plot analysis."""
    
    def test_ramachandran_analyzer_initialization(self):
        """Test Ramachandran analyzer initialization."""
        analyzer = RamachandranAnalyzer()
        assert analyzer is not None
    
    def test_dihedral_angle_calculation(self, mock_protein):
        """Test phi/psi dihedral angle calculation."""
        analyzer = RamachandranAnalyzer()
        phi_angles, psi_angles = analyzer.calculate_phi_psi_angles(mock_protein)
        
        assert len(phi_angles) > 0
        assert len(psi_angles) > 0
        assert len(phi_angles) == len(psi_angles)
    
    def test_ramachandran_plot_generation(self, mock_protein):
        """Test Ramachandran plot generation."""
        analyzer = RamachandranAnalyzer()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            analyzer.create_plot(mock_protein, output_file=tmp.name)
            assert Path(tmp.name).exists()
    
    def test_ramachandran_trajectory_analysis(self, mock_trajectory):
        """Test Ramachandran analysis over trajectory."""
        analyzer = RamachandranAnalyzer()
        results = analyzer.analyze_trajectory(mock_trajectory)
        
        assert 'phi_angles' in results
        assert 'psi_angles' in results
        assert len(results['phi_angles']) == len(mock_trajectory.frames)


@pytest.mark.skipif(not HYDROGEN_BONDS_AVAILABLE, reason="Hydrogen bonds module not available")
class TestHydrogenBondAnalysis:
    """Test suite for hydrogen bond analysis."""
    
    def test_hydrogen_bond_analyzer_initialization(self):
        """Test hydrogen bond analyzer initialization."""
        analyzer = HydrogenBondAnalyzer()
        assert analyzer is not None
    
    def test_hydrogen_bond_detection(self, mock_protein):
        """Test hydrogen bond detection."""
        analyzer = HydrogenBondAnalyzer()
        bonds = analyzer.find_hydrogen_bonds(mock_protein)
        
        assert isinstance(bonds, list)
        # Check bond structure
        if bonds:
            bond = bonds[0]
            assert hasattr(bond, 'donor')
            assert hasattr(bond, 'acceptor')
            assert hasattr(bond, 'distance')
    
    def test_hydrogen_bond_criteria(self, mock_protein):
        """Test hydrogen bond geometric criteria."""
        analyzer = HydrogenBondAnalyzer(
            distance_cutoff=3.5,  # Angstroms
            angle_cutoff=30.0     # degrees
        )
        bonds = analyzer.find_hydrogen_bonds(mock_protein)
        
        for bond in bonds:
            assert bond.distance <= 3.5
            assert bond.angle <= 30.0
    
    def test_hydrogen_bond_lifetime_analysis(self, mock_trajectory):
        """Test hydrogen bond lifetime analysis."""
        analyzer = HydrogenBondAnalyzer()
        lifetime_stats = analyzer.analyze_bond_lifetimes(mock_trajectory)
        
        assert 'mean_lifetime' in lifetime_stats
        assert 'bond_occupancy' in lifetime_stats
        assert lifetime_stats['mean_lifetime'] >= 0


@pytest.mark.skipif(not RG_AVAILABLE, reason="Radius of gyration module not available")
class TestRadiusOfGyrationAnalysis:
    """Test suite for radius of gyration analysis."""
    
    def test_rg_analyzer_initialization(self):
        """Test radius of gyration analyzer initialization."""
        analyzer = RadiusOfGyrationAnalyzer()
        assert analyzer is not None
    
    def test_rg_calculation(self, mock_protein):
        """Test radius of gyration calculation."""
        analyzer = RadiusOfGyrationAnalyzer()
        rg = analyzer.calculate_radius_of_gyration(mock_protein)
        
        assert rg > 0.0
        assert isinstance(rg, float)
    
    def test_rg_trajectory_analysis(self, mock_trajectory):
        """Test radius of gyration analysis over trajectory."""
        analyzer = RadiusOfGyrationAnalyzer()
        rg_values = analyzer.analyze_trajectory(mock_trajectory)
        
        assert len(rg_values) == len(mock_trajectory.frames)
        assert all(rg > 0 for rg in rg_values)
    
    def test_rg_segmental_analysis(self, mock_protein):
        """Test segmental radius of gyration analysis."""
        analyzer = RadiusOfGyrationAnalyzer()
        
        # Define segments
        segments = {
            'N-terminal': list(range(0, 10)),
            'C-terminal': list(range(10, 20))
        }
        
        rg_segments = analyzer.calculate_segmental_rg(mock_protein, segments)
        
        assert 'N-terminal' in rg_segments
        assert 'C-terminal' in rg_segments
        assert all(rg > 0 for rg in rg_segments.values())


@pytest.mark.skipif(not SS_AVAILABLE, reason="Secondary structure module not available")
class TestSecondaryStructureAnalysis:
    """Test suite for secondary structure analysis."""
    
    def test_ss_analyzer_initialization(self):
        """Test secondary structure analyzer initialization."""
        analyzer = SecondaryStructureAnalyzer()
        assert analyzer is not None
    
    def test_ss_assignment(self, mock_protein):
        """Test secondary structure assignment."""
        analyzer = SecondaryStructureAnalyzer()
        ss_assignment = analyzer.assign_secondary_structure(mock_protein)
        
        assert len(ss_assignment) == len(mock_protein.residues)
        valid_ss_types = ['H', 'G', 'I', 'E', 'B', 'T', 'S', 'C']
        assert all(ss in valid_ss_types for ss in ss_assignment)
    
    def test_ss_trajectory_analysis(self, mock_trajectory):
        """Test secondary structure analysis over trajectory."""
        analyzer = SecondaryStructureAnalyzer()
        ss_evolution = analyzer.analyze_trajectory(mock_trajectory)
        
        assert 'timeline' in ss_evolution
        assert 'statistics' in ss_evolution
        assert len(ss_evolution['timeline']) == len(mock_trajectory.frames)
    
    def test_ss_statistics_calculation(self, mock_trajectory):
        """Test secondary structure statistics calculation."""
        analyzer = SecondaryStructureAnalyzer()
        stats = analyzer.calculate_statistics(mock_trajectory)
        
        assert 'percentages' in stats
        assert 'stability' in stats
        assert sum(stats['percentages'].values()) == pytest.approx(100.0, abs=1.0)


class TestAnalysisIntegration:
    """Integration tests for analysis modules."""
    
    def test_multi_analysis_workflow(self, mock_trajectory):
        """Test combined analysis workflow."""
        results = {}
        
        # RMSD analysis
        if RMSD_AVAILABLE:
            rmsd_calc = RMSDCalculator(reference_structure=mock_trajectory.frames[0])
            results['rmsd'] = rmsd_calc.analyze_trajectory(mock_trajectory)
        
        # Radius of gyration analysis
        if RG_AVAILABLE:
            rg_analyzer = RadiusOfGyrationAnalyzer()
            results['rg'] = rg_analyzer.analyze_trajectory(mock_trajectory)
        
        # Secondary structure analysis
        if SS_AVAILABLE:
            ss_analyzer = SecondaryStructureAnalyzer()
            results['ss'] = ss_analyzer.analyze_trajectory(mock_trajectory)
        
        # Validate consistency
        if 'rmsd' in results and 'rg' in results:
            assert len(results['rmsd']) == len(results['rg'])
    
    def test_analysis_performance_benchmarking(self, mock_large_trajectory):
        """Test analysis performance on large trajectories."""
        import time
        
        if not RG_AVAILABLE:
            pytest.skip("RG module not available")
        
        analyzer = RadiusOfGyrationAnalyzer()
        
        start_time = time.time()
        rg_values = analyzer.analyze_trajectory(mock_large_trajectory)
        analysis_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert analysis_time < 10.0  # 10 seconds for large trajectory
        assert len(rg_values) == len(mock_large_trajectory.frames)
    
    def test_analysis_memory_efficiency(self, mock_large_trajectory):
        """Test memory efficiency of analysis modules."""
        import tracemalloc
        
        if not RMSD_AVAILABLE:
            pytest.skip("RMSD module not available")
        
        tracemalloc.start()
        
        rmsd_calc = RMSDCalculator(reference_structure=mock_large_trajectory.frames[0])
        rmsd_values = rmsd_calc.analyze_trajectory(mock_large_trajectory)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory usage should be reasonable
        assert peak < 100 * 1024 * 1024  # Less than 100 MB
        assert len(rmsd_values) == len(mock_large_trajectory.frames)


# Performance regression tests
class TestAnalysisPerformanceRegression:
    """Performance regression tests for analysis modules."""
    
    @pytest.mark.performance
    def test_rmsd_calculation_performance(self, mock_trajectory, benchmark):
        """Benchmark RMSD calculation performance."""
        if not RMSD_AVAILABLE:
            pytest.skip("RMSD module not available")
        
        calc = RMSDCalculator(reference_structure=mock_trajectory.frames[0])
        
        def run_rmsd_analysis():
            return calc.analyze_trajectory(mock_trajectory)
        
        result = benchmark(run_rmsd_analysis)
        assert len(result) == len(mock_trajectory.frames)
    
    @pytest.mark.performance
    def test_hydrogen_bond_detection_performance(self, mock_trajectory, benchmark):
        """Benchmark hydrogen bond detection performance."""
        if not HYDROGEN_BONDS_AVAILABLE:
            pytest.skip("Hydrogen bonds module not available")
        
        analyzer = HydrogenBondAnalyzer()
        
        def run_hb_analysis():
            return analyzer.analyze_trajectory(mock_trajectory)
        
        result = benchmark(run_hb_analysis)
        assert isinstance(result, dict)
