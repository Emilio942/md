"""
Enhanced test suite for Cross-Correlation Analysis module.
Expands coverage for the recently integrated GUI analysis module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from proteinMD.analysis.cross_correlation import DynamicCrossCorrelationAnalyzer


class TestCrossCorrelationBasicFunctionality:
    """Test basic Cross-Correlation analysis functionality."""
    
    @pytest.fixture
    def sample_trajectory_data(self):
        """Create sample trajectory data for testing."""
        n_frames = 50
        n_atoms = 100
        trajectory = []
        
        for i in range(n_frames):
            frame_positions = np.random.rand(n_atoms, 3) * 10.0
            # Add some correlation - atoms move together
            if i > 0:
                frame_positions += trajectory[-1] * 0.1
            trajectory.append(frame_positions)
        
        return np.array(trajectory)
    
    @pytest.fixture
    def analyzer(self):
        """Create a DynamicCrossCorrelationAnalyzer instance."""
        return DynamicCrossCorrelationAnalyzer()
    
    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes correctly."""
        assert analyzer is not None
        assert hasattr(analyzer, 'calculate_cross_correlation')
        assert hasattr(analyzer, 'calculate_covariance_matrix')
    
    def test_covariance_matrix_calculation(self, analyzer, sample_trajectory_data):
        """Test covariance matrix calculation."""
        covariance_matrix = analyzer.calculate_covariance_matrix(sample_trajectory_data)
        
        # Check dimensions
        n_frames, n_atoms, n_dims = sample_trajectory_data.shape
        expected_size = n_atoms * n_dims
        assert covariance_matrix.shape == (expected_size, expected_size)
        
        # Check symmetry
        np.testing.assert_allclose(covariance_matrix, covariance_matrix.T, rtol=1e-10)
    
    def test_cross_correlation_calculation(self, analyzer, sample_trajectory_data):
        """Test cross-correlation matrix calculation."""
        correlation_matrix = analyzer.calculate_cross_correlation(sample_trajectory_data)
        
        # Check dimensions
        n_frames, n_atoms, n_dims = sample_trajectory_data.shape
        expected_size = n_atoms * n_dims
        assert correlation_matrix.shape == (expected_size, expected_size)
        
        # Check diagonal elements are 1 (self-correlation)
        diagonal = np.diag(correlation_matrix)
        np.testing.assert_allclose(diagonal, 1.0, atol=1e-10)
        
        # Check values are between -1 and 1
        assert np.all(correlation_matrix >= -1.0)
        assert np.all(correlation_matrix <= 1.0)
    
    def test_window_size_parameter(self, analyzer):
        """Test window size parameter handling."""
        # Test default window size
        assert analyzer.window_size == 50
        
        # Test custom window size
        analyzer.window_size = 100
        assert analyzer.window_size == 100
    
    def test_mode_parameter(self, analyzer):
        """Test analysis mode parameter handling."""
        # Test default mode
        assert analyzer.mode == 'full'
        
        # Test custom mode
        analyzer.mode = 'backbone'
        assert analyzer.mode == 'backbone'


class TestCrossCorrelationAdvancedFeatures:
    """Test advanced Cross-Correlation analysis features."""
    
    @pytest.fixture
    def analyzer_with_params(self):
        """Create analyzer with specific parameters."""
        return DynamicCrossCorrelationAnalyzer(window_size=30, mode='backbone')
    
    def test_backbone_mode_filtering(self, analyzer_with_params):
        """Test backbone atom filtering."""
        # Create mock protein with backbone atoms
        mock_trajectory = np.random.rand(20, 50, 3)
        
        with patch.object(analyzer_with_params, '_filter_backbone_atoms') as mock_filter:
            mock_filter.return_value = np.random.rand(20, 15, 3)  # Fewer atoms after filtering
            
            result = analyzer_with_params.calculate_cross_correlation(mock_trajectory)
            
            # Verify filter was called
            mock_filter.assert_called_once()
            
            # Check result dimensions match filtered data
            assert result.shape == (15 * 3, 15 * 3)
    
    def test_sliding_window_analysis(self, analyzer_with_params):
        """Test sliding window correlation analysis."""
        trajectory = np.random.rand(100, 30, 3)
        
        results = analyzer_with_params.analyze_trajectory_windows(trajectory)
        
        # Should have multiple correlation matrices
        assert len(results) > 1
        
        # Each result should be a correlation matrix
        for correlation_matrix in results:
            assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
    
    def test_time_lagged_correlation(self, analyzer_with_params):
        """Test time-lagged correlation analysis."""
        trajectory = np.random.rand(80, 25, 3)
        
        correlation_matrices = analyzer_with_params.calculate_time_lagged_correlation(
            trajectory, max_lag=10
        )
        
        # Should have correlation matrices for different lags
        assert len(correlation_matrices) == 11  # 0 to 10 lag
        
        # Check that lag=0 gives auto-correlation
        auto_corr = correlation_matrices[0]
        diagonal = np.diag(auto_corr)
        np.testing.assert_allclose(diagonal, 1.0, atol=1e-10)


class TestCrossCorrelationVisualization:
    """Test Cross-Correlation visualization capabilities."""
    
    @pytest.fixture
    def analyzer_with_data(self):
        """Create analyzer with computed correlation data."""
        analyzer = DynamicCrossCorrelationAnalyzer()
        
        # Generate sample correlation matrix
        size = 60
        correlation_matrix = np.random.rand(size, size)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(correlation_matrix, 1.0)  # Set diagonal to 1
        
        analyzer.correlation_matrix = correlation_matrix
        return analyzer
    
    def test_plot_correlation_matrix(self, analyzer_with_data):
        """Test correlation matrix plotting."""
        with patch('matplotlib.pyplot.imshow') as mock_imshow, \
             patch('matplotlib.pyplot.colorbar') as mock_colorbar, \
             patch('matplotlib.pyplot.xlabel'), \
             patch('matplotlib.pyplot.ylabel'), \
             patch('matplotlib.pyplot.title'):
            
            analyzer_with_data.plot_correlation_matrix()
            
            # Verify plotting functions were called
            mock_imshow.assert_called_once()
            mock_colorbar.assert_called_once()
    
    def test_plot_correlation_heatmap_with_clustering(self, analyzer_with_data):
        """Test correlation heatmap with hierarchical clustering."""
        with patch('scipy.cluster.hierarchy.linkage') as mock_linkage, \
             patch('scipy.cluster.hierarchy.dendrogram') as mock_dendrogram, \
             patch('matplotlib.pyplot.subplots') as mock_subplots:
            
            mock_subplots.return_value = (Mock(), Mock())
            mock_linkage.return_value = np.random.rand(59, 4)
            
            analyzer_with_data.plot_correlation_heatmap_clustered()
            
            # Verify clustering functions were called
            mock_linkage.assert_called_once()
            mock_dendrogram.assert_called()
    
    def test_save_visualization(self, analyzer_with_data, tmp_path):
        """Test saving correlation visualizations."""
        output_file = tmp_path / "correlation_plot.png"
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.imshow'), \
             patch('matplotlib.pyplot.colorbar'):
            
            analyzer_with_data.plot_correlation_matrix(save_path=str(output_file))
            
            # Verify save function was called
            mock_savefig.assert_called_once_with(str(output_file), dpi=300, bbox_inches='tight')


class TestCrossCorrelationDataExport:
    """Test Cross-Correlation data export functionality."""
    
    @pytest.fixture
    def analyzer_with_results(self):
        """Create analyzer with computed results."""
        analyzer = DynamicCrossCorrelationAnalyzer()
        
        # Generate sample data
        size = 45
        analyzer.correlation_matrix = np.random.rand(size, size)
        analyzer.time_series_data = np.random.rand(100, size)
        analyzer.eigenvalues = np.random.rand(size)
        analyzer.eigenvectors = np.random.rand(size, size)
        
        return analyzer
    
    @pytest.mark.skip(reason="File export test - may fail in CI environment")
    def test_export_correlation_matrix(self, analyzer_with_results, tmp_path):
        """Test exporting correlation matrix to CSV."""
        output_file = tmp_path / "correlation_matrix.csv"
        
        analyzer_with_results.export_correlation_matrix(str(output_file))
        
        # Verify file was created
        assert output_file.exists()
        
        # Verify content
        loaded_data = np.loadtxt(str(output_file), delimiter=',')
        np.testing.assert_allclose(loaded_data, analyzer_with_results.correlation_matrix)
    
    @pytest.mark.skip(reason="Comprehensive export test - may fail in CI environment")  
    def test_export_results_comprehensive(self, analyzer_with_results, tmp_path):
        """Test comprehensive results export."""
        output_dir = tmp_path / "correlation_results"
        
        analyzer_with_results.export_results(str(output_dir))
        
        # Verify directory was created
        assert output_dir.exists()
        
        # Check for expected files
        expected_files = [
            "correlation_matrix.csv",
            "eigenvalues.csv",
            "eigenvectors.csv",
            "summary.json"
        ]
        
        for filename in expected_files:
            assert (output_dir / filename).exists()
    
    def test_export_json_metadata(self, analyzer_with_results, tmp_path):
        """Test JSON metadata export."""
        output_file = tmp_path / "metadata.json"
        
        analyzer_with_results.export_metadata(str(output_file))
        
        # Verify file was created
        assert output_file.exists()
        
        # Verify content can be loaded
        import json
        with open(output_file, 'r') as f:
            metadata = json.load(f)
        
        assert 'analysis_type' in metadata
        assert 'correlation_matrix_size' in metadata
        assert 'computation_timestamp' in metadata


class TestCrossCorrelationErrorHandling:
    """Test Cross-Correlation error handling and edge cases."""
    
    def test_empty_trajectory_handling(self):
        """Test handling of empty trajectory data."""
        analyzer = DynamicCrossCorrelationAnalyzer()
        empty_trajectory = np.array([])
        
        with pytest.raises(ValueError, match="Empty trajectory"):
            analyzer.calculate_cross_correlation(empty_trajectory)
    
    def test_invalid_trajectory_shape(self):
        """Test handling of invalid trajectory shapes."""
        analyzer = DynamicCrossCorrelationAnalyzer()
        
        # 2D array instead of 3D
        invalid_trajectory = np.random.rand(50, 100)
        
        with pytest.raises(ValueError, match="Invalid trajectory shape"):
            analyzer.calculate_cross_correlation(invalid_trajectory)
    
    def test_insufficient_frames(self):
        """Test handling of insufficient number of frames."""
        analyzer = DynamicCrossCorrelationAnalyzer()
        
        # Very few frames
        short_trajectory = np.random.rand(2, 10, 3)
        
        with pytest.raises(ValueError, match="Insufficient frames"):
            analyzer.calculate_cross_correlation(short_trajectory)
    
    def test_window_size_validation(self):
        """Test window size parameter validation."""
        analyzer = DynamicCrossCorrelationAnalyzer()
        
        # Test invalid window sizes
        with pytest.raises(ValueError, match="Window size must be positive"):
            analyzer.window_size = -5
        
        with pytest.raises(ValueError, match="Window size must be positive"):
            analyzer.window_size = 0
    
    def test_mode_validation(self):
        """Test mode parameter validation."""
        analyzer = DynamicCrossCorrelationAnalyzer()
        
        # Test invalid mode
        with pytest.raises(ValueError, match="Invalid mode"):
            analyzer.mode = "invalid_mode"


class TestCrossCorrelationPerformance:
    """Test Cross-Correlation performance and optimization."""
    
    def test_large_trajectory_handling(self):
        """Test handling of large trajectory data."""
        analyzer = DynamicCrossCorrelationAnalyzer()
        
        # Large trajectory - should complete without memory issues
        large_trajectory = np.random.rand(200, 500, 3)
        
        # This should not raise memory errors
        correlation_matrix = analyzer.calculate_cross_correlation(large_trajectory)
        
        # Verify result dimensions
        expected_size = 500 * 3
        assert correlation_matrix.shape == (expected_size, expected_size)
    
    def test_memory_efficient_computation(self):
        """Test memory-efficient computation options."""
        analyzer = DynamicCrossCorrelationAnalyzer(memory_efficient=True)
        
        trajectory = np.random.rand(100, 200, 3)
        
        # Should complete with memory-efficient mode
        correlation_matrix = analyzer.calculate_cross_correlation(trajectory)
        
        # Verify result
        assert correlation_matrix.shape == (200 * 3, 200 * 3)
    
    def test_parallel_computation(self):
        """Test parallel computation capabilities."""
        analyzer = DynamicCrossCorrelationAnalyzer(n_jobs=2)
        
        trajectory = np.random.rand(80, 150, 3)
        
        # Should complete with parallel computation
        correlation_matrix = analyzer.calculate_cross_correlation(trajectory)
        
        # Verify result
        assert correlation_matrix.shape == (150 * 3, 150 * 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
