"""
Dynamic Cross-Correlation Analysis for Protein MD Trajectories

This module implements Task 13.2: Dynamic Cross-Correlation Analysis for analyzing
correlated motions between protein regions in molecular dynamics trajectories.

Requirements:
1. Cross-Correlation-Matrix berechnet und visualisiert
2. Statistische Signifikanz der Korrelationen
3. Netzwerk-Darstellung stark korrelierter Regionen
4. Zeit-abhängige Korrelations-Analyse

Author: GitHub Copilot
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
import scipy.stats as stats
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import networkx as nx
from scipy.cluster.hierarchy import linkage, dendrogram
import json
import os
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CrossCorrelationResults:
    """Results container for cross-correlation analysis."""
    
    # Core correlation data
    correlation_matrix: np.ndarray
    n_frames: int
    n_atoms: int
    atom_selection: str
    
    # Statistical significance
    p_values: Optional[np.ndarray] = None
    confidence_intervals: Optional[np.ndarray] = None
    significant_pairs: Optional[List[Tuple[int, int, float]]] = None
    
    # Network analysis
    network_graph: Optional[nx.Graph] = None
    communities: Optional[List[List[int]]] = None
    network_metrics: Optional[Dict[str, Any]] = None
    
    # Time-dependent analysis
    time_windows: Optional[List[Tuple[int, int]]] = None
    windowed_correlations: Optional[List[np.ndarray]] = None
    correlation_evolution: Optional[np.ndarray] = None
    
    # Metadata
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class NetworkAnalysisResults:
    """Results container for network analysis."""
    
    graph: nx.Graph
    communities: List[List[int]]
    modularity: float
    centrality_measures: Dict[str, Dict[int, float]]
    highly_connected_nodes: List[int]
    network_statistics: Dict[str, float]


@dataclass
class StatisticalSignificanceResults:
    """Results container for statistical significance analysis."""
    
    p_values: np.ndarray
    adjusted_p_values: np.ndarray
    confidence_intervals: np.ndarray
    significant_correlations: List[Tuple[int, int, float, float]]
    correction_method: str
    alpha_level: float


class DynamicCrossCorrelationAnalyzer:
    """
    Analyzer for dynamic cross-correlation analysis of protein trajectories.
    
    This class implements comprehensive cross-correlation analysis including:
    - Correlation matrix calculation and visualization
    - Statistical significance testing
    - Network representation of correlated regions
    - Time-dependent correlation analysis
    """
    
    def __init__(self, atom_selection: str = 'CA', **kwargs):
        """
        Initialize the cross-correlation analyzer.
        
        Parameters
        ----------
        atom_selection : str
            Atom selection for analysis ('CA', 'backbone', 'all')
        **kwargs : additional parameters for backward compatibility
            window_size, memory_efficient, n_jobs, mode, etc.
        """
        self.atom_selection = atom_selection
        self.trajectory_data = []
        
        # Analysis parameters
        self.correlation_threshold = 0.5
        self.significance_level = 0.05
        self._window_size = kwargs.get('window_size', 50)
        self.step_size = 10
        
        # Add mode for backward compatibility
        self._mode = kwargs.get('mode', 'full')
        
        # Add attributes for test compatibility
        self.correlation_matrix = None
        self.time_series_data = None
        self.eigenvalues = None
        self.eigenvectors = None
        
        logger.info(f"Initialized CrossCorrelationAnalyzer with selection: {atom_selection}")
    
    @property
    def mode(self):
        """Get analysis mode."""
        return self._mode
    
    @mode.setter 
    def mode(self, value):
        """Set analysis mode with validation."""
        valid_modes = ['full', 'backbone', 'CA', 'all']
        if value not in valid_modes:
            raise ValueError(f"Invalid mode '{value}'. Must be one of {valid_modes}")
        self._mode = value
    
    @property
    def window_size(self):
        """Get window size for analysis."""
        return self._window_size
    
    @window_size.setter
    def window_size(self, value):
        """Set window size with validation."""
        if value <= 0:
            raise ValueError("Window size must be positive")
        self._window_size = value
    
    def calculate_correlation_matrix(self, 
                                   trajectory: np.ndarray,
                                   align_trajectory: bool = True,
                                   correlation_type: str = 'pearson') -> CrossCorrelationResults:
        """
        Calculate dynamic cross-correlation matrix.
        
        Parameters
        ----------
        trajectory : np.ndarray
            Trajectory array of shape (n_frames, n_atoms, 3)
        align_trajectory : bool
            Whether to align trajectory frames
        correlation_type : str
            Type of correlation ('pearson', 'spearman', 'kendall')
            
        Returns
        -------
        CrossCorrelationResults
            Results containing correlation matrix and metadata
        """
        logger.info("Calculating cross-correlation matrix...")
        
        n_frames, n_atoms, n_dims = trajectory.shape
        
        # Align trajectory if requested
        if align_trajectory:
            from .pca import TrajectoryAligner
            aligner = TrajectoryAligner()
            trajectory = aligner.align_trajectory(trajectory)
            logger.info("Trajectory aligned using Kabsch algorithm")
        
        # Extract coordinates based on atom selection
        coords = self._extract_coordinates(trajectory)
        n_selected_atoms = coords.shape[1] // 3
        
        # Calculate correlation matrix
        correlation_matrix = self._compute_correlation_matrix(coords, correlation_type)
        
        # Create results object
        results = CrossCorrelationResults(
            correlation_matrix=correlation_matrix,
            n_frames=n_frames,
            n_atoms=n_selected_atoms,
            atom_selection=self.atom_selection,
            parameters={
                'correlation_type': correlation_type,
                'aligned': align_trajectory,
                'n_original_atoms': n_atoms
            }
        )
        
        # Store trajectory data for further analysis
        self.trajectory_data = coords
        
        # Store correlation matrix for backward compatibility
        self.correlation_matrix = correlation_matrix
        
        logger.info(f"Correlation matrix calculated: {correlation_matrix.shape}")
        return results
    
    def calculate_significance(self, 
                             results: CrossCorrelationResults,
                             method: str = 'bootstrap',
                             n_bootstrap: int = 1000,
                             correction: str = 'bonferroni') -> StatisticalSignificanceResults:
        """
        Calculate statistical significance of correlations.
        
        Parameters
        ----------
        results : CrossCorrelationResults
            Correlation analysis results
        method : str
            Significance testing method ('bootstrap', 'ttest', 'permutation')
        n_bootstrap : int
            Number of bootstrap samples
        correction : str
            Multiple testing correction ('bonferroni', 'fdr', 'none')
            
        Returns
        -------
        StatisticalSignificanceResults
            Statistical significance results
        """
        logger.info(f"Calculating statistical significance using {method} method...")
        
        correlation_matrix = results.correlation_matrix
        n_atoms = correlation_matrix.shape[0]
        
        # Initialize p-value matrix
        p_values = np.ones_like(correlation_matrix)
        confidence_intervals = np.zeros((n_atoms, n_atoms, 2))
        
        # Calculate p-values for off-diagonal elements
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if method == 'bootstrap':
                    p_val, ci = self._bootstrap_correlation_test(i, j, n_bootstrap)
                elif method == 'ttest':
                    p_val, ci = self._ttest_correlation(correlation_matrix[i, j], results.n_frames)
                elif method == 'permutation':
                    p_val, ci = self._permutation_test(i, j, n_bootstrap)
                else:
                    raise ValueError(f"Unknown significance test method: {method}")
                
                p_values[i, j] = p_val
                p_values[j, i] = p_val
                confidence_intervals[i, j] = ci
                confidence_intervals[j, i] = ci
        
        # Apply multiple testing correction
        adjusted_p_values = self._apply_correction(p_values, correction)
        
        # Find significant correlations
        significant_correlations = self._extract_significant_correlations(
            correlation_matrix, adjusted_p_values, self.significance_level
        )
        
        significance_results = StatisticalSignificanceResults(
            p_values=p_values,
            adjusted_p_values=adjusted_p_values,
            confidence_intervals=confidence_intervals,
            significant_correlations=significant_correlations,
            correction_method=correction,
            alpha_level=self.significance_level
        )
        
        # Update results object
        results.p_values = p_values
        results.confidence_intervals = confidence_intervals
        results.significant_pairs = significant_correlations
        
        logger.info(f"Found {len(significant_correlations)} significant correlations")
        return significance_results
    
    def analyze_network(self, 
                       results: CrossCorrelationResults,
                       threshold: Optional[float] = None,
                       use_significance: bool = True) -> NetworkAnalysisResults:
        """
        Analyze correlation network and identify communities.
        
        Parameters
        ----------
        results : CrossCorrelationResults
            Correlation analysis results
        threshold : float, optional
            Correlation threshold for edge creation
        use_significance : bool
            Whether to use statistical significance for filtering
            
        Returns
        -------
        NetworkAnalysisResults
            Network analysis results
        """
        logger.info("Analyzing correlation network...")
        
        if threshold is None:
            threshold = self.correlation_threshold
        
        correlation_matrix = results.correlation_matrix
        n_atoms = correlation_matrix.shape[0]
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for i in range(n_atoms):
            G.add_node(i, residue_id=i)
        
        # Add edges based on correlation threshold
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                correlation = abs(correlation_matrix[i, j])
                
                # Check significance if requested
                include_edge = correlation >= threshold
                if use_significance and results.p_values is not None:
                    p_val = results.p_values[i, j]
                    include_edge = include_edge and (p_val < self.significance_level)
                
                if include_edge:
                    G.add_edge(i, j, 
                             weight=correlation,
                             correlation=correlation_matrix[i, j])
        
        # Community detection
        communities = self._detect_communities(G)
        
        # Calculate network metrics
        centrality_measures = self._calculate_centrality_measures(G)
        
        # Calculate modularity
        if len(communities) > 1:
            modularity = nx.community.modularity(G, communities)
        else:
            modularity = 0.0
        
        # Identify highly connected nodes
        degree_centrality = centrality_measures['degree']
        threshold_centrality = np.percentile(list(degree_centrality.values()), 80)
        highly_connected_nodes = [
            node for node, centrality in degree_centrality.items() 
            if centrality >= threshold_centrality
        ]
        
        # Network statistics
        network_statistics = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G),
            'n_communities': len(communities),
            'modularity': modularity
        }
        
        network_results = NetworkAnalysisResults(
            graph=G,
            communities=communities,
            modularity=modularity,
            centrality_measures=centrality_measures,
            highly_connected_nodes=highly_connected_nodes,
            network_statistics=network_statistics
        )
        
        # Update results object
        results.network_graph = G
        results.communities = communities
        results.network_metrics = network_statistics
        
        logger.info(f"Network analysis complete: {network_statistics}")
        return network_results
    
    def time_dependent_analysis(self, 
                              trajectory: np.ndarray,
                              window_size: Optional[int] = None,
                              step_size: Optional[int] = None,
                              max_lag: int = 20) -> CrossCorrelationResults:
        """
        Perform time-dependent correlation analysis.
        
        Parameters
        ----------
        trajectory : np.ndarray
            Trajectory array of shape (n_frames, n_atoms, 3)
        window_size : int, optional
            Size of sliding window
        step_size : int, optional
            Step size for sliding window
        max_lag : int
            Maximum time lag for correlation analysis
            
        Returns
        -------
        CrossCorrelationResults
            Results with time-dependent analysis
        """
        logger.info("Performing time-dependent correlation analysis...")
        
        if window_size is None:
            window_size = self.window_size
        if step_size is None:
            step_size = self.step_size
        
        n_frames = trajectory.shape[0]
        coords = self._extract_coordinates(trajectory)
        
        # Sliding window analysis
        time_windows = []
        windowed_correlations = []
        
        for start in range(0, n_frames - window_size + 1, step_size):
            end = start + window_size
            window_coords = coords[start:end]
            
            # Calculate correlation matrix for this window
            correlation_matrix = self._compute_correlation_matrix(window_coords, 'pearson')
            
            time_windows.append((start, end))
            windowed_correlations.append(correlation_matrix)
        
        # Analyze correlation evolution
        correlation_evolution = self._analyze_correlation_evolution(windowed_correlations)
        
        # Time-lagged correlation analysis
        lagged_correlations = self._calculate_lagged_correlations(coords, max_lag)
        
        # Create comprehensive results
        results = CrossCorrelationResults(
            correlation_matrix=windowed_correlations[-1] if windowed_correlations else np.array([]),
            n_frames=n_frames,
            n_atoms=coords.shape[1] // 3,
            atom_selection=self.atom_selection,
            time_windows=time_windows,
            windowed_correlations=windowed_correlations,
            correlation_evolution=correlation_evolution,
            parameters={
                'window_size': window_size,
                'step_size': step_size,
                'max_lag': max_lag,
                'n_windows': len(time_windows),
                'lagged_correlations': lagged_correlations
            }
        )
        
        logger.info(f"Time-dependent analysis complete: {len(time_windows)} windows")
        return results
    
    def visualize_matrix(self, 
                        results: CrossCorrelationResults,
                        output_file: Optional[str] = None,
                        figsize: Tuple[float, float] = (12, 10),
                        cmap: str = 'RdBu_r',
                        show_significance: bool = True) -> plt.Figure:
        """
        Visualize correlation matrix as heatmap.
        
        Parameters
        ----------
        results : CrossCorrelationResults
            Correlation analysis results
        output_file : str, optional
            Output file path
        figsize : tuple
            Figure size
        cmap : str
            Colormap for heatmap
        show_significance : bool
            Whether to overlay significance markers
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        correlation_matrix = results.correlation_matrix
        
        fig, axes = plt.subplots(1, 2 if results.p_values is not None else 1, 
                                figsize=figsize)
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Main correlation heatmap
        im1 = axes[0].imshow(correlation_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='equal')
        axes[0].set_title('Dynamic Cross-Correlation Matrix')
        axes[0].set_xlabel('Residue Index')
        axes[0].set_ylabel('Residue Index')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Correlation Coefficient')
        
        # Overlay significance markers if available
        if show_significance and results.p_values is not None:
            significant_mask = results.p_values < self.significance_level
            y_indices, x_indices = np.where(significant_mask)
            axes[0].scatter(x_indices, y_indices, marker='.', s=1, color='black', alpha=0.3)
        
        # P-values heatmap if available
        if results.p_values is not None and len(axes) > 1:
            im2 = axes[1].imshow(results.p_values, cmap='viridis', vmin=0, vmax=0.1, aspect='equal')
            axes[1].set_title('Statistical Significance (p-values)')
            axes[1].set_xlabel('Residue Index')
            axes[1].set_ylabel('Residue Index')
            
            cbar2 = plt.colorbar(im2, ax=axes[1])
            cbar2.set_label('p-value')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Matrix visualization saved to {output_file}")
        
        return fig
    
    def visualize_network(self, 
                         network_results: NetworkAnalysisResults,
                         output_file: Optional[str] = None,
                         figsize: Tuple[float, float] = (12, 8),
                         layout: str = 'spring') -> plt.Figure:
        """
        Visualize correlation network graph.
        
        Parameters
        ----------
        network_results : NetworkAnalysisResults
            Network analysis results
        output_file : str, optional
            Output file path
        figsize : tuple
            Figure size
        layout : str
            Network layout algorithm ('spring', 'circular', 'kamada_kawai')
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        G = network_results.graph
        communities = network_results.communities
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Color nodes by community
        colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        node_colors = ['lightgray'] * G.number_of_nodes()
        
        for i, community in enumerate(communities):
            for node in community:
                if node < len(node_colors):
                    node_colors[node] = colors[i]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=100, alpha=0.8, ax=ax)
        
        # Draw edges with thickness proportional to correlation strength
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], 
                              alpha=0.5, edge_color='gray', ax=ax)
        
        # Add labels for highly connected nodes
        highly_connected = network_results.highly_connected_nodes
        labels = {node: str(node) for node in highly_connected}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title('Correlation Network\n'
                    f'Communities: {len(communities)}, '
                    f'Modularity: {network_results.modularity:.3f}')
        ax.axis('off')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Network visualization saved to {output_file}")
        
        return fig
    
    def visualize_time_evolution(self, 
                               results: CrossCorrelationResults,
                               output_file: Optional[str] = None,
                               figsize: Tuple[float, float] = (15, 8)) -> plt.Figure:
        """
        Visualize time evolution of correlations.
        
        Parameters
        ----------
        results : CrossCorrelationResults
            Results with time-dependent analysis
        output_file : str, optional
            Output file path
        figsize : tuple
            Figure size
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        if results.correlation_evolution is None:
            raise ValueError("No time evolution data available")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # 1. Correlation evolution heatmap
        im1 = axes[0].imshow(results.correlation_evolution.T, aspect='auto', 
                            cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0].set_title('Correlation Evolution Over Time')
        axes[0].set_xlabel('Time Window')
        axes[0].set_ylabel('Residue Pair')
        plt.colorbar(im1, ax=axes[0])
        
        # 2. Average correlation per time window
        avg_correlations = [np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])) 
                           for corr_matrix in results.windowed_correlations]
        axes[1].plot(avg_correlations, 'b-', linewidth=2)
        axes[1].set_title('Average Absolute Correlation')
        axes[1].set_xlabel('Time Window')
        axes[1].set_ylabel('Mean |Correlation|')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Correlation variance per time window
        var_correlations = [np.var(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]) 
                           for corr_matrix in results.windowed_correlations]
        axes[2].plot(var_correlations, 'r-', linewidth=2)
        axes[2].set_title('Correlation Variance')
        axes[2].set_xlabel('Time Window')
        axes[2].set_ylabel('Correlation Variance')
        axes[2].grid(True, alpha=0.3)
        
        # 4. Lagged correlation analysis
        if 'lagged_correlations' in results.parameters:
            lagged_corr = results.parameters['lagged_correlations']
            for i, (pair, lags) in enumerate(lagged_corr.items()):
                if i < 5:  # Show only first 5 pairs
                    axes[3].plot(lags, alpha=0.7, label=f'Pair {pair}')
            axes[3].set_title('Lagged Correlations (Selected Pairs)')
            axes[3].set_xlabel('Time Lag')
            axes[3].set_ylabel('Correlation')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Time evolution visualization saved to {output_file}")
        
        return fig
    
    def export_results_original(self, 
                      results: CrossCorrelationResults,
                      network_results: Optional[NetworkAnalysisResults] = None,
                      output_dir: str = "cross_correlation_export") -> str:
        """
        Export analysis results to files.
        
        Parameters
        ----------
        results : CrossCorrelationResults
            Correlation analysis results
        network_results : NetworkAnalysisResults, optional
            Network analysis results
        output_dir : str
            Output directory path
            
        Returns
        -------
        str
            Path to export directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Export correlation matrix
        np.savetxt(os.path.join(output_dir, 'correlation_matrix.txt'), 
                  results.correlation_matrix)
        np.save(os.path.join(output_dir, 'correlation_matrix.npy'), 
               results.correlation_matrix)
        
        # Export significance data if available
        if results.p_values is not None:
            np.savetxt(os.path.join(output_dir, 'p_values.txt'), results.p_values)
            np.save(os.path.join(output_dir, 'p_values.npy'), results.p_values)
        
        if results.confidence_intervals is not None:
            np.save(os.path.join(output_dir, 'confidence_intervals.npy'), 
                   results.confidence_intervals)
        
        # Export significant pairs
        if results.significant_pairs:
            with open(os.path.join(output_dir, 'significant_pairs.txt'), 'w') as f:
                f.write("# Residue_i Residue_j Correlation p_value\n")
                for i, j, corr, p_val in results.significant_pairs:
                    f.write(f"{i} {j} {corr:.6f} {p_val:.6e}\n")
        
        # Export network data if available
        if network_results:
            nx.write_gml(network_results.graph, 
                        os.path.join(output_dir, 'correlation_network.gml'))
            
            # Export community data
            with open(os.path.join(output_dir, 'communities.json'), 'w') as f:
                json.dump({
                    'communities': network_results.communities,
                    'modularity': network_results.modularity,
                    'network_statistics': network_results.network_statistics
                }, f, indent=2)
        
        # Export time evolution data if available
        if results.windowed_correlations:
            for i, corr_matrix in enumerate(results.windowed_correlations):
                np.save(os.path.join(output_dir, f'window_{i:03d}_correlation.npy'), 
                       corr_matrix)
        
        # Export metadata
        metadata = {
            'n_frames': int(results.n_frames),
            'n_atoms': int(results.n_atoms),
            'atom_selection': results.atom_selection,
            'parameters': results.parameters or {},
            'analysis_type': 'dynamic_cross_correlation'
        }
        
        with open(os.path.join(output_dir, 'analysis_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=self._json_serialize)
        
        logger.info(f"Results exported to {output_dir}")
        return output_dir
    
    def export_results(self, output_dir: str, **kwargs):
        """
        Export analysis results to files (overridden for backward compatibility).
        
        Parameters
        ----------
        output_dir : str
            Output directory path
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export correlation matrix if available
        if self.correlation_matrix is not None:
            np.savetxt(os.path.join(output_dir, 'correlation_matrix.csv'), 
                      self.correlation_matrix, delimiter=',')
        
        # Export eigenvalues and eigenvectors if available
        if self.eigenvalues is not None:
            np.savetxt(os.path.join(output_dir, 'eigenvalues.csv'), 
                      self.eigenvalues, delimiter=',')
        
        if self.eigenvectors is not None:
            np.savetxt(os.path.join(output_dir, 'eigenvectors.csv'), 
                      self.eigenvectors, delimiter=',')
        
        # Export summary
        summary = {
            'atom_selection': self.atom_selection,
            'correlation_threshold': self.correlation_threshold,
            'significance_level': self.significance_level,
            'matrix_shape': list(self.correlation_matrix.shape) if self.correlation_matrix is not None else None,
            'n_eigenvalues': len(self.eigenvalues) if self.eigenvalues is not None else 0
        }
        
        summary_file = os.path.join(output_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return output_dir
    
    # Alias methods for backward compatibility with tests
    def calculate_cross_correlation(self, trajectory: np.ndarray, **kwargs) -> np.ndarray:
        """Alias for calculate_correlation_matrix for backward compatibility."""
        # Input validation
        if trajectory.size == 0:
            raise ValueError("Empty trajectory provided")
        
        if len(trajectory.shape) != 3:
            raise ValueError("Invalid trajectory shape. Expected (n_frames, n_atoms, 3)")
        
        n_frames, n_atoms, n_dims = trajectory.shape
        
        if n_frames < 3:
            raise ValueError("Insufficient frames for correlation analysis. Need at least 3 frames")
        
        # Apply filtering based on mode
        if hasattr(self, 'mode') and self.mode == 'backbone':
            trajectory = self._filter_backbone_atoms(trajectory)
            n_frames, n_atoms, n_dims = trajectory.shape
        
        # For backward compatibility, calculate coordinate-level correlations
        # Flatten coordinates to (n_frames, n_atoms * 3)
        coords_flat = trajectory.reshape(n_frames, n_atoms * n_dims)
        
        # Calculate correlation matrix between all coordinates
        n_coords = coords_flat.shape[1]
        correlation_matrix = np.corrcoef(coords_flat.T)
        
        # Store for other methods
        self.correlation_matrix = correlation_matrix
        
        return correlation_matrix
    
    def calculate_covariance_matrix(self, trajectory: np.ndarray) -> np.ndarray:
        """Calculate covariance matrix for backward compatibility."""
        # For covariance matrix, work with the original trajectory shape
        n_frames, n_atoms, n_dims = trajectory.shape
        # Flatten coordinates to (n_frames, n_atoms * 3)
        coords_flat = trajectory.reshape(n_frames, n_atoms * n_dims)
        # Center the coordinates
        coords_centered = coords_flat - np.mean(coords_flat, axis=0)
        # Calculate covariance matrix
        return np.cov(coords_centered.T)
    
    def plot_correlation_matrix(self, save_path=None, **kwargs):
        """Alias for visualize_matrix for backward compatibility."""
        if self.correlation_matrix is not None:
            import matplotlib.pyplot as plt
            
            # Use plt.imshow to match test expectations
            plt.figure(figsize=kwargs.get('figsize', (10, 8)))
            plt.imshow(self.correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.title('Cross-Correlation Matrix')
            plt.xlabel('Coordinate Index')
            plt.ylabel('Coordinate Index')
            plt.colorbar(label='Correlation')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return plt.gcf()
        else:
            raise ValueError("No correlation matrix available. Run calculate_cross_correlation first.")
    
    def plot_correlation_heatmap_clustered(self, **kwargs):
        """Create a clustered heatmap visualization."""
        if self.correlation_matrix is not None:
            import matplotlib.pyplot as plt
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform
            
            # Create distance matrix for clustering
            distance_matrix = 1 - np.abs(self.correlation_matrix)
            condensed_dist = squareform(distance_matrix, checks=False)
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(condensed_dist, method='average')
            
            fig, axes = plt.subplots(1, 2, figsize=kwargs.get('figsize', (15, 6)))
            ax1, ax2 = axes if hasattr(axes, '__iter__') and len(axes) == 2 else (axes, axes)
            
            # Plot dendrogram
            dendrogram(linkage_matrix, ax=ax1)
            
            # Plot clustered heatmap  
            ax2.imshow(self.correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            
            return fig
        else:
            raise ValueError("No correlation matrix available. Run calculate_cross_correlation first.")
    
    def export_correlation_matrix(self, filepath: str, **kwargs):
        """Export correlation matrix to file."""
        if self.correlation_matrix is not None:
            np.savetxt(filepath, self.correlation_matrix, delimiter=',')
        else:
            raise ValueError("No correlation matrix available. Run calculate_cross_correlation first.")
    
    def export_metadata(self, filepath: str, **kwargs):
        """Export analysis metadata to JSON file."""
        from datetime import datetime
        
        metadata = {
            'analysis_type': 'cross_correlation',
            'atom_selection': self.atom_selection,
            'window_size': self.window_size,
            'mode': self.mode,
            'correlation_threshold': self.correlation_threshold,
            'significance_level': self.significance_level,
            'correlation_matrix_size': list(self.correlation_matrix.shape) if self.correlation_matrix is not None else None,
            'computation_timestamp': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Private helper methods
    
    def _extract_coordinates(self, trajectory: np.ndarray) -> np.ndarray:
        """Extract coordinates based on atom selection."""
        n_frames, n_atoms, n_dims = trajectory.shape
        
        if self.atom_selection == 'all':
            coords = trajectory.reshape(n_frames, -1)
        elif self.atom_selection == 'CA':
            # For simplicity, assume every 4th atom is CA (needs protein structure info)
            ca_indices = np.arange(0, n_atoms, 4)
            coords = trajectory[:, ca_indices, :].reshape(n_frames, -1)
        elif self.atom_selection == 'backbone':
            # Assume backbone atoms are first 3 of every 4 (N, CA, C)
            backbone_indices = []
            for i in range(0, n_atoms, 4):
                backbone_indices.extend([i, i+1, i+2])
            backbone_indices = np.array(backbone_indices)
            backbone_indices = backbone_indices[backbone_indices < n_atoms]
            coords = trajectory[:, backbone_indices, :].reshape(n_frames, -1)
        else:
            raise ValueError(f"Unknown atom selection: {self.atom_selection}")
        
        return coords
    
    def _compute_correlation_matrix(self, coords: np.ndarray, correlation_type: str) -> np.ndarray:
        """Compute correlation matrix between atomic coordinates."""
        n_frames, n_coords = coords.shape
        n_atoms = n_coords // 3
        
        # Reshape to separate x, y, z coordinates for each atom
        coords_reshaped = coords.reshape(n_frames, n_atoms, 3)
        
        # Calculate correlation matrix between atoms
        correlation_matrix = np.zeros((n_atoms, n_atoms))
        
        for i in range(n_atoms):
            for j in range(i, n_atoms):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Calculate correlation between atomic displacement vectors
                    vec_i = coords_reshaped[:, i, :].flatten()
                    vec_j = coords_reshaped[:, j, :].flatten()
                    
                    if correlation_type == 'pearson':
                        corr, _ = pearsonr(vec_i, vec_j)
                    elif correlation_type == 'spearman':
                        corr, _ = stats.spearmanr(vec_i, vec_j)
                    elif correlation_type == 'kendall':
                        corr, _ = stats.kendalltau(vec_i, vec_j)
                    else:
                        raise ValueError(f"Unknown correlation type: {correlation_type}")
                    
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr
        
        return correlation_matrix
    
    def _bootstrap_correlation_test(self, atom_i: int, atom_j: int, n_bootstrap: int) -> Tuple[float, Tuple[float, float]]:
        """Perform bootstrap test for correlation significance."""
        n_frames = self.trajectory_data.shape[0]
        n_atoms = self.trajectory_data.shape[1] // 3
        
        # Extract data for the two atoms
        coords_reshaped = self.trajectory_data.reshape(n_frames, n_atoms, 3)
        vec_i = coords_reshaped[:, atom_i, :].flatten()
        vec_j = coords_reshaped[:, atom_j, :].flatten()
        
        # Original correlation
        original_corr, _ = pearsonr(vec_i, vec_j)
        
        # Bootstrap sampling
        bootstrap_corrs = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(vec_i), len(vec_i), replace=True)
            boot_corr, _ = pearsonr(vec_i[indices], vec_j[indices])
            bootstrap_corrs.append(boot_corr)
        
        bootstrap_corrs = np.array(bootstrap_corrs)
        
        # Calculate p-value (two-tailed test)
        p_value = np.sum(np.abs(bootstrap_corrs) >= np.abs(original_corr)) / n_bootstrap
        
        # Confidence interval
        ci_lower = np.percentile(bootstrap_corrs, 2.5)
        ci_upper = np.percentile(bootstrap_corrs, 97.5)
        
        return p_value, (ci_lower, ci_upper)
    
    def _ttest_correlation(self, correlation: float, n_frames: int) -> Tuple[float, Tuple[float, float]]:
        """Perform t-test for correlation significance."""
        # Convert correlation to t-statistic
        if abs(correlation) >= 1.0:
            return 0.0, (correlation, correlation)
        
        df = n_frames - 2
        t_stat = correlation * np.sqrt(df / (1 - correlation**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # Confidence interval using Fisher z-transformation
        z = 0.5 * np.log((1 + correlation) / (1 - correlation))
        se_z = 1 / np.sqrt(n_frames - 3)
        z_critical = stats.norm.ppf(0.975)
        
        z_lower = z - z_critical * se_z
        z_upper = z + z_critical * se_z
        
        ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return p_value, (ci_lower, ci_upper)
    
    def _permutation_test(self, atom_i: int, atom_j: int, n_permutations: int) -> Tuple[float, Tuple[float, float]]:
        """Perform permutation test for correlation significance."""
        n_frames = self.trajectory_data.shape[0]
        n_atoms = self.trajectory_data.shape[1] // 3
        
        # Extract data for the two atoms
        coords_reshaped = self.trajectory_data.reshape(n_frames, n_atoms, 3)
        vec_i = coords_reshaped[:, atom_i, :].flatten()
        vec_j = coords_reshaped[:, atom_j, :].flatten()
        
        # Original correlation
        original_corr, _ = pearsonr(vec_i, vec_j)
        
        # Permutation test
        perm_corrs = []
        for _ in range(n_permutations):
            shuffled_j = np.random.permutation(vec_j)
            perm_corr, _ = pearsonr(vec_i, shuffled_j)
            perm_corrs.append(perm_corr)
        
        perm_corrs = np.array(perm_corrs)
        
        # Calculate p-value
        p_value = np.sum(np.abs(perm_corrs) >= np.abs(original_corr)) / n_permutations
        
        # Confidence interval from permutation distribution
        ci_lower = np.percentile(perm_corrs, 2.5)
        ci_upper = np.percentile(perm_corrs, 97.5)
        
        return p_value, (ci_lower, ci_upper)
    
    def _apply_correction(self, p_values: np.ndarray, correction: str) -> np.ndarray:
        """Apply multiple testing correction."""
        mask = np.triu(np.ones_like(p_values, dtype=bool), k=1)
        flat_p_values = p_values[mask]
        
        if correction == 'bonferroni':
            adjusted_p_values = flat_p_values * len(flat_p_values)
        elif correction == 'fdr':
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(flat_p_values)
            sorted_p_values = flat_p_values[sorted_indices]
            
            m = len(sorted_p_values)
            adjusted = np.zeros_like(sorted_p_values)
            
            for i in range(m-1, -1, -1):
                if i == m-1:
                    adjusted[i] = sorted_p_values[i]
                else:
                    adjusted[i] = min(adjusted[i+1], 
                                    sorted_p_values[i] * m / (i + 1))
            
            # Restore original order
            adjusted_p_values = np.zeros_like(flat_p_values)
            adjusted_p_values[sorted_indices] = adjusted
        elif correction == 'none':
            adjusted_p_values = flat_p_values
        else:
            raise ValueError(f"Unknown correction method: {correction}")
        
        # Reconstruct full matrix
        result = np.ones_like(p_values)
        result[mask] = np.clip(adjusted_p_values, 0, 1)
        result = result + result.T - np.diag(np.diag(result))
        
        return result
    
    def _extract_significant_correlations(self, 
                                        correlation_matrix: np.ndarray,
                                        p_values: np.ndarray,
                                        alpha: float) -> List[Tuple[int, int, float, float]]:
        """Extract significant correlation pairs."""
        significant_pairs = []
        n_atoms = correlation_matrix.shape[0]
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if p_values[i, j] < alpha:
                    significant_pairs.append((i, j, correlation_matrix[i, j], p_values[i, j]))
        
        # Sort by correlation strength
        significant_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return significant_pairs
    
    def _detect_communities(self, graph: nx.Graph) -> List[List[int]]:
        """Detect communities in correlation network."""
        try:
            # Use greedy modularity optimization
            communities = nx.community.greedy_modularity_communities(graph)
            return [list(community) for community in communities]
        except:
            # Fallback to simple connected components
            components = nx.connected_components(graph)
            return [list(component) for component in components]
    
    def _calculate_centrality_measures(self, graph: nx.Graph) -> Dict[str, Dict[int, float]]:
        """Calculate various centrality measures."""
        centrality_measures = {}
        
        if graph.number_of_nodes() > 0:
            centrality_measures['degree'] = nx.degree_centrality(graph)
            centrality_measures['betweenness'] = nx.betweenness_centrality(graph)
            centrality_measures['closeness'] = nx.closeness_centrality(graph)
            centrality_measures['eigenvector'] = nx.eigenvector_centrality(graph, max_iter=1000)
        else:
            # Empty graph
            centrality_measures = {
                'degree': {},
                'betweenness': {},
                'closeness': {},
                'eigenvector': {}
            }
        
        return centrality_measures
    
    def _analyze_correlation_evolution(self, windowed_correlations: List[np.ndarray]) -> np.ndarray:
        """Analyze evolution of correlations over time."""
        n_windows = len(windowed_correlations)
        n_atoms = windowed_correlations[0].shape[0]
        
        # Extract upper triangular correlations for each window
        evolution_matrix = []
        
        for corr_matrix in windowed_correlations:
            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            evolution_matrix.append(upper_tri)
        
        return np.array(evolution_matrix)
    
    def _calculate_lagged_correlations(self, coords: np.ndarray, max_lag: int) -> Dict[Tuple[int, int], np.ndarray]:
        """Calculate time-lagged correlations."""
        n_frames, n_coords = coords.shape
        n_atoms = n_coords // 3
        
        # Select a subset of atom pairs for lagged correlation analysis
        n_pairs = min(10, n_atoms * (n_atoms - 1) // 2)
        atom_pairs = [(i, j) for i in range(n_atoms) for j in range(i+1, n_atoms)][:n_pairs]
        
        lagged_correlations = {}
        coords_reshaped = coords.reshape(n_frames, n_atoms, 3)
        
        for atom_i, atom_j in atom_pairs:
            vec_i = coords_reshaped[:, atom_i, :].flatten()
            vec_j = coords_reshaped[:, atom_j, :].flatten()
            
            correlations = []
            for lag in range(max_lag + 1):
                if lag == 0:
                    corr, _ = pearsonr(vec_i, vec_j)
                else:
                    if len(vec_i) > lag:
                        corr, _ = pearsonr(vec_i[:-lag], vec_j[lag:])
                    else:
                        corr = 0.0
                correlations.append(corr)
            
            lagged_correlations[(atom_i, atom_j)] = np.array(correlations)
        
        return lagged_correlations
    
    def _json_serialize(self, obj):
        """JSON serialization helper for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def _filter_backbone_atoms(self, trajectory: np.ndarray) -> np.ndarray:
        """Filter backbone atoms from trajectory."""
        n_frames, n_atoms, n_dims = trajectory.shape
        # Assume backbone atoms are first 3 of every 4 (N, CA, C)
        backbone_indices = []
        for i in range(0, n_atoms, 4):
            backbone_indices.extend([i, i+1, i+2])
        backbone_indices = np.array(backbone_indices)
        backbone_indices = backbone_indices[backbone_indices < n_atoms]
        
        return trajectory[:, backbone_indices, :]

    def analyze_trajectory_windows(self, trajectory: np.ndarray, **kwargs):
        """Analyze trajectory using sliding windows."""
        n_frames = trajectory.shape[0]
        results = []
        
        for start in range(0, n_frames - self.window_size + 1, self.step_size):
            end = start + self.window_size
            window_traj = trajectory[start:end]
            corr_matrix = self.calculate_cross_correlation(window_traj)
            results.append(corr_matrix)
        
        return results
    
    def calculate_time_lagged_correlation(self, trajectory: np.ndarray, max_lag: int = 10, **kwargs):
        """Calculate time-lagged correlation matrices."""
        results = {}
        
        for lag in range(max_lag + 1):
            if lag == 0:
                corr_matrix = self.calculate_cross_correlation(trajectory)
            else:
                # Create lagged trajectory
                traj1 = trajectory[:-lag]
                traj2 = trajectory[lag:]
                
                # Calculate cross-correlation between lagged trajectories
                n_frames, n_atoms, n_dims = traj1.shape
                coords1 = traj1.reshape(n_frames, n_atoms * n_dims)
                coords2 = traj2.reshape(n_frames, n_atoms * n_dims)
                
                # Calculate correlation
                corr_matrix = np.corrcoef(coords1.T, coords2.T)[:coords1.shape[1], coords1.shape[1]:]
            
            results[lag] = corr_matrix
        
        return results
def create_test_trajectory(n_frames: int = 100, n_atoms: int = 50, 
                         motion_type: str = 'correlated') -> np.ndarray:
    """
    Create a test trajectory for cross-correlation analysis.
    
    Parameters
    ----------
    n_frames : int
        Number of trajectory frames
    n_atoms : int
        Number of atoms
    motion_type : str
        Type of motion ('correlated', 'anticorrelated', 'random')
        
    Returns
    -------
    np.ndarray
        Test trajectory of shape (n_frames, n_atoms, 3)
    """
    trajectory = np.zeros((n_frames, n_atoms, 3))
    
    if motion_type == 'correlated':
        # Create groups of correlated atoms
        n_groups = 3
        group_size = n_atoms // n_groups
        
        for frame in range(n_frames):
            for group in range(n_groups):
                start_idx = group * group_size
                end_idx = min(start_idx + group_size, n_atoms)
                
                # Generate correlated motion for this group
                base_motion = np.random.normal(0, 0.1, 3)
                noise_scale = 0.05
                
                for atom in range(start_idx, end_idx):
                    trajectory[frame, atom, :] = (
                        base_motion + np.random.normal(0, noise_scale, 3)
                    )
    
    elif motion_type == 'anticorrelated':
        # Create anticorrelated pairs
        for frame in range(n_frames):
            for atom in range(0, n_atoms-1, 2):
                base_motion = np.random.normal(0, 0.1, 3)
                trajectory[frame, atom, :] = base_motion
                if atom + 1 < n_atoms:
                    trajectory[frame, atom + 1, :] = -base_motion
    
    else:  # random
        trajectory = np.random.normal(0, 0.1, (n_frames, n_atoms, 3))
    
    # Add cumulative displacement to simulate actual MD trajectory
    trajectory = np.cumsum(trajectory, axis=0)
    
    return trajectory


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Dynamic Cross-Correlation Analysis...")
    
    # Create test trajectory
    trajectory = create_test_trajectory(n_frames=100, n_atoms=20, motion_type='correlated')
    print(f"Created test trajectory: {trajectory.shape}")
    
    # Initialize analyzer
    analyzer = DynamicCrossCorrelationAnalyzer(atom_selection='all')
    
    # Calculate correlation matrix
    results = analyzer.calculate_correlation_matrix(trajectory)
    print(f"Correlation matrix shape: {results.correlation_matrix.shape}")
    
    # Calculate significance
    significance_results = analyzer.calculate_significance(results, method='bootstrap', n_bootstrap=100)
    print(f"Found {len(significance_results.significant_correlations)} significant correlations")
    
    # Network analysis
    network_results = analyzer.analyze_network(results)
    print(f"Network statistics: {network_results.network_statistics}")
    
    # Time-dependent analysis
    time_results = analyzer.time_dependent_analysis(trajectory, window_size=30, step_size=10)
    print(f"Time-dependent analysis: {len(time_results.time_windows)} windows")
    
    # Create visualizations
    with tempfile.TemporaryDirectory() as temp_dir:
        # Matrix visualization
        fig1 = analyzer.visualize_matrix(results, 
                                       output_file=os.path.join(temp_dir, 'correlation_matrix.png'))
        
        # Network visualization
        fig2 = analyzer.visualize_network(network_results,
                                        output_file=os.path.join(temp_dir, 'correlation_network.png'))
        
        # Time evolution visualization
        fig3 = analyzer.visualize_time_evolution(time_results,
                                               output_file=os.path.join(temp_dir, 'time_evolution.png'))
        
        # Export results
        export_dir = analyzer.export_results(results, network_results, 
                                           output_dir=os.path.join(temp_dir, 'results'))
        
        print(f"Test completed successfully. Results would be saved to: {export_dir}")
    
    print("✅ Dynamic Cross-Correlation Analysis implementation complete!")
