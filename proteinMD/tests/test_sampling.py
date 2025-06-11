"""
Comprehensive Unit Tests for Sampling Module

Task 10.1: Umfassende Unit Tests 🚀

Tests the sampling modules including:
- Umbrella sampling
- Replica exchange molecular dynamics
- Enhanced sampling methods
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

# Try to import sampling modules
try:
    from sampling.umbrella_sampling import UmbrellaSampling, WHAMAnalysis
    UMBRELLA_SAMPLING_AVAILABLE = True
except ImportError:
    UMBRELLA_SAMPLING_AVAILABLE = False

try:
    from sampling.replica_exchange import ReplicaExchangeMD, TemperatureLadder
    REPLICA_EXCHANGE_AVAILABLE = True
except ImportError:
    REPLICA_EXCHANGE_AVAILABLE = False


@pytest.mark.skipif(not UMBRELLA_SAMPLING_AVAILABLE, reason="Umbrella sampling module not available")
class TestUmbrellaSampling:
    """Test suite for umbrella sampling functionality."""
    
    def test_umbrella_sampling_initialization(self, mock_simulation_system):
        """Test umbrella sampling initialization."""
        # Define umbrella windows
        windows = [
            {'coordinate': 'distance', 'center': 2.0, 'k': 1000.0},
            {'coordinate': 'distance', 'center': 2.5, 'k': 1000.0},
            {'coordinate': 'distance', 'center': 3.0, 'k': 1000.0}
        ]
        
        umbrella = UmbrellaSampling(mock_simulation_system, windows)
        assert umbrella is not None
        assert len(umbrella.windows) == 3
    
    def test_harmonic_restraints(self, mock_simulation_system):
        """Test harmonic restraint implementation."""
        windows = [{'coordinate': 'distance', 'center': 2.0, 'k': 1000.0}]
        umbrella = UmbrellaSampling(mock_simulation_system, windows)
        
        # Test restraint force calculation
        coordinate_value = 2.5
        force = umbrella.calculate_restraint_force(0, coordinate_value)
        
        # Force should point toward the restraint center
        expected_force = -1000.0 * (coordinate_value - 2.0)
        assert force == pytest.approx(expected_force, rel=1e-3)
    
    def test_multiple_windows_simulation(self, mock_simulation_system):
        """Test simulation with multiple umbrella windows."""
        windows = [
            {'coordinate': 'distance', 'center': i * 0.5 + 1.5, 'k': 1000.0}
            for i in range(10)
        ]
        
        umbrella = UmbrellaSampling(mock_simulation_system, windows)
        
        # Run simulation for each window
        results = umbrella.run_all_windows(steps_per_window=100)
        
        assert len(results) == 10
        for i, result in enumerate(results):
            assert 'trajectory' in result
            assert 'coordinates' in result
    
    def test_coordinate_calculation(self, mock_simulation_system):
        """Test reaction coordinate calculation."""
        windows = [{'coordinate': 'distance', 'center': 2.0, 'k': 1000.0}]
        umbrella = UmbrellaSampling(mock_simulation_system, windows)
        
        # Mock atom positions for distance calculation
        positions = np.array([
            [0.0, 0.0, 0.0],  # Atom 1
            [2.0, 0.0, 0.0]   # Atom 2
        ])
        
        distance = umbrella.calculate_coordinate(positions, atom_indices=[0, 1])
        assert distance == pytest.approx(2.0, rel=1e-3)
    
    def test_convergence_checking(self, mock_simulation_system):
        """Test convergence checking for umbrella windows."""
        windows = [{'coordinate': 'distance', 'center': 2.0, 'k': 1000.0}]
        umbrella = UmbrellaSampling(mock_simulation_system, windows)
        
        # Generate mock coordinate data
        coordinates = np.random.normal(2.0, 0.1, 1000)
        
        is_converged = umbrella.check_convergence(coordinates, window_index=0)
        assert isinstance(is_converged, bool)


@pytest.mark.skipif(not UMBRELLA_SAMPLING_AVAILABLE, reason="Umbrella sampling module not available")
class TestWHAMAnalysis:
    """Test suite for WHAM analysis functionality."""
    
    def test_wham_initialization(self):
        """Test WHAM analysis initialization."""
        wham = WHAMAnalysis()
        assert wham is not None
    
    def test_pmf_calculation(self):
        """Test potential of mean force calculation."""
        wham = WHAMAnalysis()
        
        # Mock umbrella sampling data
        window_data = []
        for i in range(5):
            center = 2.0 + i * 0.5
            coordinates = np.random.normal(center, 0.2, 500)
            window_data.append({
                'coordinates': coordinates,
                'center': center,
                'k': 1000.0
            })
        
        pmf, bins = wham.calculate_pmf(window_data)
        
        assert len(pmf) == len(bins) - 1
        assert not np.any(np.isnan(pmf))
    
    def test_error_estimation(self):
        """Test PMF error estimation using bootstrap."""
        wham = WHAMAnalysis()
        
        # Mock umbrella sampling data
        window_data = []
        for i in range(5):
            center = 2.0 + i * 0.5
            coordinates = np.random.normal(center, 0.2, 500)
            window_data.append({
                'coordinates': coordinates,
                'center': center,
                'k': 1000.0
            })
        
        pmf, errors = wham.calculate_pmf_with_errors(window_data, n_bootstrap=10)
        
        assert len(pmf) == len(errors)
        assert all(error >= 0 for error in errors)
    
    def test_histogram_reweighting(self):
        """Test histogram reweighting algorithm."""
        wham = WHAMAnalysis()
        
        # Mock data with known distribution
        coordinates = np.random.exponential(2.0, 1000)
        weights = wham.calculate_weights(coordinates, bias_potential=lambda x: 0.5 * x**2)
        
        assert len(weights) == len(coordinates)
        assert all(weight > 0 for weight in weights)


@pytest.mark.skipif(not REPLICA_EXCHANGE_AVAILABLE, reason="Replica exchange module not available")
class TestReplicaExchangeMD:
    """Test suite for replica exchange molecular dynamics."""
    
    def test_replica_exchange_initialization(self, mock_simulation_system):
        """Test replica exchange MD initialization."""
        temperatures = [300, 320, 340, 360]
        remd = ReplicaExchangeMD(mock_simulation_system, temperatures)
        
        assert remd is not None
        assert len(remd.replicas) == 4
        assert remd.temperatures == temperatures
    
    def test_temperature_ladder_generation(self):
        """Test temperature ladder generation."""
        ladder = TemperatureLadder.exponential(min_temp=300, max_temp=400, n_replicas=6)
        
        assert len(ladder) == 6
        assert ladder[0] == 300
        assert ladder[-1] == 400
        assert all(ladder[i] < ladder[i+1] for i in range(len(ladder)-1))
    
    def test_replica_exchange_attempt(self, mock_simulation_system):
        """Test replica exchange attempt between neighboring replicas."""
        temperatures = [300, 320, 340, 360]
        remd = ReplicaExchangeMD(mock_simulation_system, temperatures)
        
        # Mock energies for replicas
        energies = [-1000, -950, -900, -850]
        
        # Attempt exchange between replicas 0 and 1
        exchange_accepted = remd.attempt_exchange(0, 1, energies)
        
        assert isinstance(exchange_accepted, bool)
    
    def test_metropolis_criterion(self, mock_simulation_system):
        """Test Metropolis criterion for replica exchange."""
        temperatures = [300, 320]
        remd = ReplicaExchangeMD(mock_simulation_system, temperatures)
        
        # Test exchange probability calculation
        energy1, energy2 = -1000, -950
        temp1, temp2 = 300, 320
        
        accept_prob = remd.calculate_exchange_probability(
            energy1, energy2, temp1, temp2
        )
        
        assert 0 <= accept_prob <= 1
    
    def test_acceptance_rate_tracking(self, mock_simulation_system):
        """Test acceptance rate tracking."""
        temperatures = [300, 320, 340, 360]
        remd = ReplicaExchangeMD(mock_simulation_system, temperatures)
        
        # Run several exchange attempts
        for _ in range(100):
            energies = np.random.uniform(-1100, -800, 4)
            for i in range(len(temperatures) - 1):
                remd.attempt_exchange(i, i+1, energies)
        
        acceptance_rates = remd.get_acceptance_rates()
        
        assert len(acceptance_rates) == len(temperatures) - 1
        assert all(0 <= rate <= 1 for rate in acceptance_rates)
    
    def test_parallel_execution(self, mock_simulation_system):
        """Test parallel execution of replicas."""
        temperatures = [300, 320, 340, 360]
        remd = ReplicaExchangeMD(mock_simulation_system, temperatures)
        
        # Test parallel simulation step
        results = remd.run_parallel_step(n_steps=10)
        
        assert len(results) == len(temperatures)
        for result in results:
            assert 'energy' in result
            assert 'trajectory' in result
    
    def test_replica_mixing(self, mock_simulation_system):
        """Test replica mixing analysis."""
        temperatures = [300, 320, 340, 360]
        remd = ReplicaExchangeMD(mock_simulation_system, temperatures)
        
        # Run simulation to generate exchange history
        remd.run_simulation(n_steps=100, exchange_frequency=10)
        
        mixing_analysis = remd.analyze_mixing()
        
        assert 'replica_paths' in mixing_analysis
        assert 'mixing_efficiency' in mixing_analysis


class TestSamplingIntegration:
    """Integration tests for sampling modules."""
    
    def test_umbrella_sampling_workflow(self, mock_simulation_system):
        """Test complete umbrella sampling workflow."""
        if not UMBRELLA_SAMPLING_AVAILABLE:
            pytest.skip("Umbrella sampling not available")
        
        # Setup umbrella windows
        windows = [
            {'coordinate': 'distance', 'center': 2.0 + i * 0.2, 'k': 1000.0}
            for i in range(8)
        ]
        
        umbrella = UmbrellaSampling(mock_simulation_system, windows)
        
        # Run sampling
        sampling_results = umbrella.run_all_windows(steps_per_window=50)
        
        # Perform WHAM analysis
        wham = WHAMAnalysis()
        pmf, bins = wham.calculate_pmf(sampling_results)
        
        assert len(pmf) > 0
        assert len(bins) == len(pmf) + 1
    
    def test_replica_exchange_workflow(self, mock_simulation_system):
        """Test complete replica exchange workflow."""
        if not REPLICA_EXCHANGE_AVAILABLE:
            pytest.skip("Replica exchange not available")
        
        temperatures = [300, 320, 340, 360]
        remd = ReplicaExchangeMD(mock_simulation_system, temperatures)
        
        # Run REMD simulation
        remd.run_simulation(n_steps=50, exchange_frequency=5)
        
        # Analyze results
        mixing = remd.analyze_mixing()
        acceptance_rates = remd.get_acceptance_rates()
        
        assert len(acceptance_rates) == len(temperatures) - 1
        assert 'mixing_efficiency' in mixing
    
    def test_sampling_convergence_analysis(self, mock_simulation_system):
        """Test convergence analysis for enhanced sampling methods."""
        if not UMBRELLA_SAMPLING_AVAILABLE:
            pytest.skip("Umbrella sampling not available")
        
        windows = [{'coordinate': 'distance', 'center': 2.5, 'k': 1000.0}]
        umbrella = UmbrellaSampling(mock_simulation_system, windows)
        
        # Generate time series data
        coordinates = []
        for step in range(1000):
            coord = 2.5 + 0.1 * np.sin(step * 0.01) + np.random.normal(0, 0.05)
            coordinates.append(coord)
        
        # Test convergence
        convergence = umbrella.analyze_convergence(coordinates)
        
        assert 'autocorrelation_time' in convergence
        assert 'effective_samples' in convergence


# Performance regression tests
class TestSamplingPerformanceRegression:
    """Performance regression tests for sampling modules."""
    
    @pytest.mark.performance
    def test_umbrella_sampling_performance(self, mock_simulation_system, benchmark):
        """Benchmark umbrella sampling performance."""
        if not UMBRELLA_SAMPLING_AVAILABLE:
            pytest.skip("Umbrella sampling not available")
        
        windows = [{'coordinate': 'distance', 'center': 2.5, 'k': 1000.0}]
        umbrella = UmbrellaSampling(mock_simulation_system, windows)
        
        def run_umbrella_window():
            return umbrella.run_window(0, n_steps=100)
        
        result = benchmark(run_umbrella_window)
        assert 'trajectory' in result
    
    @pytest.mark.performance
    def test_replica_exchange_performance(self, mock_simulation_system, benchmark):
        """Benchmark replica exchange performance."""
        if not REPLICA_EXCHANGE_AVAILABLE:
            pytest.skip("Replica exchange not available")
        
        temperatures = [300, 320, 340, 360]
        remd = ReplicaExchangeMD(mock_simulation_system, temperatures)
        
        def run_remd_step():
            return remd.run_parallel_step(n_steps=10)
        
        result = benchmark(run_remd_step)
        assert len(result) == len(temperatures)
    
    @pytest.mark.performance
    def test_wham_analysis_performance(self, benchmark):
        """Benchmark WHAM analysis performance."""
        if not UMBRELLA_SAMPLING_AVAILABLE:
            pytest.skip("Umbrella sampling not available")
        
        # Generate large dataset
        window_data = []
        for i in range(20):
            center = 2.0 + i * 0.1
            coordinates = np.random.normal(center, 0.1, 2000)
            window_data.append({
                'coordinates': coordinates,
                'center': center,
                'k': 1000.0
            })
        
        wham = WHAMAnalysis()
        
        def run_wham_analysis():
            return wham.calculate_pmf(window_data)
        
        pmf, bins = benchmark(run_wham_analysis)
        assert len(pmf) > 0
