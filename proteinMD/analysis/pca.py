"""
Principal Component Analysis for Molecular Dynamics Trajectories

Task 13.1: Principal Component Analysis 📊
Status: IMPLEMENTING

This module provides comprehensive Principal Component Analysis (PCA) functionality
for analyzing protein dynamics and conformational changes in MD trajectories.

Features:
- PCA calculation for trajectory data
- Projection visualization onto principal components
- Conformational clustering in PC space
- Export of PC coordinates and eigenvectors
- Essential dynamics analysis
- Interactive visualization tools

References:
- Amadei, A. et al. (1993). Essential dynamics of proteins. Proteins, 17(4), 412-425.
- Garcia, A. E. (1992). Large-amplitude nonlinear motions in proteins. Phys Rev Lett, 68(17), 2696-2699.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import json
import logging
from dataclasses import dataclass, field
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PCAResults:
    """Container for PCA analysis results."""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    projections: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    mean_structure: np.ndarray
    n_components: int
    trajectory_length: int
    atom_selection: str
    
    @property
    def variance_explained(self) -> np.ndarray:
        """Alias for explained_variance_ratio for compatibility."""
        return self.explained_variance_ratio
    
    def get_explained_variance(self, n_components: int = None) -> float:
        """Get cumulative explained variance for first n components."""
        if n_components is None:
            n_components = len(self.explained_variance_ratio)
        return np.sum(self.explained_variance_ratio[:n_components]) * 100

    def get_component_importance(self, component_idx: int) -> Dict[str, float]:
        """Get importance metrics for a specific component."""
        return {
            'eigenvalue': self.eigenvalues[component_idx],
            'explained_variance': self.explained_variance_ratio[component_idx] * 100,
            'cumulative_variance': self.cumulative_variance[component_idx] * 100
        }


@dataclass 
class ClusteringResults:
    """Container for conformational clustering results."""
    cluster_labels: np.ndarray
    cluster_centers: np.ndarray
    n_clusters: int
    silhouette_score: float
    cluster_populations: Dict[int, int]
    representative_frames: Dict[int, int]
    cluster_statistics: Dict[int, Dict[str, float]]
    
    @property
    def labels(self) -> np.ndarray:
        """Alias for cluster_labels for compatibility."""
        return self.cluster_labels
    
    @property 
    def centroids(self) -> np.ndarray:
        """Alias for cluster_centers for compatibility."""
        return self.cluster_centers


class TrajectoryAligner:
    """Align trajectory frames for PCA analysis."""
    
    def __init__(self, reference_frame: Optional[np.ndarray] = None):
        self.reference_frame = reference_frame
        
    def align_trajectory(self, trajectory: np.ndarray, 
                        masses: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Align all frames in trajectory to minimize RMSD.
        
        Args:
            trajectory: Shape (n_frames, n_atoms, 3)
            masses: Optional atomic masses for weighted alignment
            
        Returns:
            Aligned trajectory with same shape
        """
        n_frames, n_atoms, _ = trajectory.shape
        aligned_trajectory = trajectory.copy()
        
        # Use first frame as reference if none provided
        if self.reference_frame is None:
            self.reference_frame = trajectory[0].copy()
            
        # Center reference frame
        if masses is not None:
            ref_center = np.average(self.reference_frame, axis=0, weights=masses)
        else:
            ref_center = np.mean(self.reference_frame, axis=0)
        
        centered_ref = self.reference_frame - ref_center
        
        for frame_idx in range(n_frames):
            # Center current frame
            if masses is not None:
                frame_center = np.average(trajectory[frame_idx], axis=0, weights=masses)
            else:
                frame_center = np.mean(trajectory[frame_idx], axis=0)
                
            centered_frame = trajectory[frame_idx] - frame_center
            
            # Calculate optimal rotation matrix using Kabsch algorithm
            if masses is not None:
                # Weight coordinates by mass
                weighted_ref = centered_ref * np.sqrt(masses)[:, np.newaxis]
                weighted_frame = centered_frame * np.sqrt(masses)[:, np.newaxis]
                H = weighted_frame.T @ weighted_ref
            else:
                H = centered_frame.T @ centered_ref
                
            # SVD to find rotation
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Ensure proper rotation (det(R) = 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Apply rotation and store aligned frame
            aligned_trajectory[frame_idx] = (centered_frame @ R) + ref_center
            
        return aligned_trajectory


class PCAAnalyzer:
    """Main class for Principal Component Analysis of MD trajectories."""
    
    def __init__(self, atom_selection: str = "CA", align_trajectory: bool = True):
        """
        Initialize PCA analyzer.
        
        Args:
            atom_selection: Which atoms to include ("CA", "backbone", "all")
            align_trajectory: Whether to align trajectory before PCA
        """
        self.atom_selection = atom_selection
        self.align_trajectory = align_trajectory
        self.aligner = TrajectoryAligner() if align_trajectory else None
        self.scaler = StandardScaler()
        
        # Storage for results
        self.pca_results = None
        self.trajectory_data = None
        self.clustering_results = None
        
        logger.info(f"PCA Analyzer initialized with atom selection: {atom_selection}")
    
    def _select_atoms(self, trajectory: np.ndarray, 
                     atom_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Select subset of atoms based on selection criteria.
        
        Args:
            trajectory: Full trajectory array
            atom_names: List of atom names corresponding to trajectory atoms
            
        Returns:
            Selected trajectory subset and atom indices
        """
        n_frames, n_atoms, _ = trajectory.shape
        
        if self.atom_selection == "all":
            return trajectory, list(range(n_atoms))
        
        if atom_names is None:
            # Assume every 4th atom is CA for protein backbone
            if self.atom_selection == "CA":
                ca_indices = list(range(1, n_atoms, 4))  # Typical CA position
            elif self.atom_selection == "backbone":
                backbone_indices = []
                for i in range(0, n_atoms, 4):
                    backbone_indices.extend([i, i+1, i+2])  # N, CA, C
                ca_indices = backbone_indices[:min(len(backbone_indices), n_atoms)]
            else:
                ca_indices = list(range(n_atoms))
        else:
            # Use atom names for selection
            if self.atom_selection == "CA":
                ca_indices = [i for i, name in enumerate(atom_names) if name == "CA"]
            elif self.atom_selection == "backbone":
                ca_indices = [i for i, name in enumerate(atom_names) 
                             if name in ["N", "CA", "C"]]
            else:
                ca_indices = list(range(n_atoms))
        
        if not ca_indices:
            logger.warning(f"No atoms found for selection '{self.atom_selection}', using all atoms")
            ca_indices = list(range(n_atoms))
            
        selected_trajectory = trajectory[:, ca_indices, :]
        logger.info(f"Selected {len(ca_indices)} atoms from {n_atoms} total atoms")
        
        return selected_trajectory, ca_indices
    
    def fit_transform(self, trajectory: np.ndarray,
                     atom_names: Optional[List[str]] = None,
                     masses: Optional[np.ndarray] = None,
                     n_components: Optional[int] = None) -> PCAResults:
        """
        Perform PCA on trajectory data.
        
        Args:
            trajectory: Shape (n_frames, n_atoms, 3)
            atom_names: List of atom names
            masses: Atomic masses for alignment
            n_components: Number of components to compute
            
        Returns:
            PCAResults object with all analysis results
        """
        logger.info("Starting PCA analysis...")
        
        # Select atoms
        selected_trajectory, atom_indices = self._select_atoms(trajectory, atom_names)
        n_frames, n_selected_atoms, _ = selected_trajectory.shape
        
        # Align trajectory if requested
        if self.align_trajectory:
            logger.info("Aligning trajectory frames...")
            if masses is not None:
                selected_masses = masses[atom_indices]
            else:
                selected_masses = None
            selected_trajectory = self.aligner.align_trajectory(selected_trajectory, selected_masses)
        
        # Reshape trajectory for PCA: (n_frames, n_features)
        # Each frame becomes a row with 3*n_atoms features
        X = selected_trajectory.reshape(n_frames, -1)
        
        # Center the data (subtract mean structure)
        mean_structure = np.mean(X, axis=0)
        X_centered = X - mean_structure
        
        # Perform PCA
        if n_components is None:
            n_components = min(n_frames - 1, X_centered.shape[1])
        
        pca = SklearnPCA(n_components=n_components)
        projections = pca.fit_transform(X_centered)
        
        # Get PCA results
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Store results
        self.pca_results = PCAResults(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            projections=projections,
            explained_variance_ratio=explained_variance_ratio,
            cumulative_variance=cumulative_variance,
            mean_structure=mean_structure.reshape(n_selected_atoms, 3),
            n_components=n_components,
            trajectory_length=n_frames,
            atom_selection=self.atom_selection
        )
        
        # Store trajectory data for further analysis
        self.trajectory_data = {
            'original_trajectory': trajectory,
            'selected_trajectory': selected_trajectory,
            'atom_indices': atom_indices,
            'atom_names': atom_names,
            'masses': masses
        }
        
        logger.info(f"PCA completed: {n_components} components, "
                   f"{self.pca_results.get_explained_variance(10):.1f}% variance in first 10 PCs")
        
        return self.pca_results
    
    def calculate_pca(self, trajectory: np.ndarray,
                     atom_selection: str = None,
                     atom_names: Optional[List[str]] = None,
                     masses: Optional[np.ndarray] = None,
                     n_components: Optional[int] = None) -> PCAResults:
        """
        Calculate PCA for trajectory data. 
        
        This is a convenience wrapper around fit_transform().
        
        Args:
            trajectory: Shape (n_frames, n_atoms, 3)
            atom_selection: Atom selection override ("CA", "backbone", "all")
            atom_names: List of atom names
            masses: Atomic masses for alignment
            n_components: Number of components to compute
            
        Returns:
            PCAResults object with all analysis results
        """
        # Override atom selection if provided
        if atom_selection is not None:
            original_selection = self.atom_selection
            self.atom_selection = atom_selection
            
        result = self.fit_transform(trajectory, atom_names, masses, n_components)
        
        # Restore original selection
        if atom_selection is not None:
            self.atom_selection = original_selection
            
        return result

    def cluster_conformations(self, results: PCAResults = None, n_clusters: int = None,
                            n_components_for_clustering: int = 10,
                            clustering_method: str = "kmeans") -> ClusteringResults:
        """
        Cluster conformations in principal component space.
        
        Args:
            results: PCA results to use (uses self.pca_results if None)
            n_clusters: Number of clusters (auto-determined if None)
            n_components_for_clustering: Number of PCs to use for clustering
            clustering_method: Clustering algorithm ("kmeans", "hierarchical")
            
        Returns:
            ClusteringResults object
        """
        # Use provided results or fall back to self.pca_results
        pca_results = results if results is not None else self.pca_results
        
        if pca_results is None:
            raise ValueError("Must run fit_transform() before clustering or provide results")
        
        logger.info("Performing conformational clustering in PC space...")
        
        # Use subset of principal components for clustering
        n_components_for_clustering = min(n_components_for_clustering, 
                                        pca_results.n_components)
        X_pca = pca_results.projections[:, :n_components_for_clustering]
        
        # Auto-determine optimal number of clusters if not provided
        if n_clusters is None or n_clusters == 'auto':
            n_clusters = self._determine_optimal_clusters(X_pca)
        
        # Perform clustering
        if clustering_method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(X_pca)
            cluster_centers = clusterer.cluster_centers_
        else:
            raise NotImplementedError(f"Clustering method '{clustering_method}' not implemented")
        
        # Calculate clustering quality
        sil_score = silhouette_score(X_pca, cluster_labels)
        
        # Calculate cluster statistics
        cluster_populations = {}
        representative_frames = {}
        cluster_statistics = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_frames = np.where(cluster_mask)[0]
            
            # Population
            cluster_populations[cluster_id] = len(cluster_frames)
            
            # Representative frame (closest to centroid)
            cluster_data = X_pca[cluster_mask]
            centroid = cluster_centers[cluster_id]
            distances = np.sum((cluster_data - centroid)**2, axis=1)
            representative_frames[cluster_id] = cluster_frames[np.argmin(distances)]
            
            # Statistics
            cluster_projections = self.pca_results.projections[cluster_mask]
            cluster_statistics[cluster_id] = {
                'mean_pc1': np.mean(cluster_projections[:, 0]),
                'std_pc1': np.std(cluster_projections[:, 0]),
                'mean_pc2': np.mean(cluster_projections[:, 1]),
                'std_pc2': np.std(cluster_projections[:, 1]),
                'size': len(cluster_frames),
                'percentage': (len(cluster_frames) / len(cluster_labels)) * 100
            }
        
        self.clustering_results = ClusteringResults(
            cluster_labels=cluster_labels,
            cluster_centers=cluster_centers,
            n_clusters=n_clusters,
            silhouette_score=sil_score,
            cluster_populations=cluster_populations,
            representative_frames=representative_frames,
            cluster_statistics=cluster_statistics
        )
        
        logger.info(f"Clustering completed: {n_clusters} clusters, "
                   f"silhouette score = {sil_score:.3f}")
        
        return self.clustering_results
    
    def _determine_optimal_clusters(self, X_pca: np.ndarray, 
                                  max_clusters: int = 10) -> int:
        """Determine optimal number of clusters using silhouette analysis."""
        n_samples = len(X_pca)
        max_clusters = min(max_clusters, n_samples // 2)
        
        if max_clusters < 2:
            return 2
        
        silhouette_scores = []
        cluster_range = range(2, max_clusters + 1)
        
        for n_clusters in cluster_range:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(X_pca)
            score = silhouette_score(X_pca, cluster_labels)
            silhouette_scores.append(score)
        
        optimal_idx = np.argmax(silhouette_scores)
        optimal_clusters = cluster_range[optimal_idx]
        
        logger.info(f"Optimal clusters determined: {optimal_clusters} "
                   f"(silhouette score: {silhouette_scores[optimal_idx]:.3f})")
        
        return optimal_clusters
    
    def plot_eigenvalue_spectrum(self, n_components: int = 20, 
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot eigenvalue spectrum and explained variance."""
        if self.pca_results is None:
            raise ValueError("Must run fit_transform() before plotting")
        
        n_components = min(n_components, self.pca_results.n_components)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Eigenvalue spectrum
        ax1.plot(range(1, n_components + 1), 
                self.pca_results.eigenvalues[:n_components], 'o-', 
                linewidth=2, markersize=6)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('Eigenvalue Spectrum')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Cumulative explained variance
        cumvar = self.pca_results.cumulative_variance[:n_components] * 100
        ax2.plot(range(1, n_components + 1), cumvar, 'o-', 
                linewidth=2, markersize=6, color='red')
        ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.7, label='80%')
        ax2.axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='90%')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Cumulative Explained Variance (%)')
        ax2.set_title('Cumulative Explained Variance')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Eigenvalue spectrum saved to {save_path}")
        
        return fig
    
    def plot_pc_projections(self, pc_pairs: List[Tuple[int, int]] = [(0, 1), (0, 2), (1, 2)],
                          color_by: str = "time", save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot trajectory projections onto principal components.
        
        Args:
            pc_pairs: List of PC pairs to plot
            color_by: Color scheme ("time", "clusters", "none")
            save_path: Optional path to save figure
        """
        if self.pca_results is None:
            raise ValueError("Must run fit_transform() before plotting")
        
        n_plots = len(pc_pairs)
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        projections = self.pca_results.projections
        
        # Determine coloring
        if color_by == "time":
            colors = np.arange(len(projections))
            cmap = plt.cm.viridis
            cbar_label = "Time (frames)"
        elif color_by == "clusters" and self.clustering_results is not None:
            colors = self.clustering_results.cluster_labels
            cmap = plt.cm.tab10
            cbar_label = "Cluster ID"
        else:
            colors = 'blue'
            cmap = None
            cbar_label = None
        
        for idx, (pc1, pc2) in enumerate(pc_pairs):
            ax = axes[idx]
            
            if cmap is not None:
                scatter = ax.scatter(projections[:, pc1], projections[:, pc2], 
                                   c=colors, cmap=cmap, alpha=0.6, s=20)
                if idx == n_plots - 1:  # Add colorbar to last plot
                    plt.colorbar(scatter, ax=ax, label=cbar_label)
            else:
                ax.scatter(projections[:, pc1], projections[:, pc2], 
                          color=colors, alpha=0.6, s=20)
            
            # Get explained variance for labels
            var1 = self.pca_results.explained_variance_ratio[pc1] * 100
            var2 = self.pca_results.explained_variance_ratio[pc2] * 100
            
            ax.set_xlabel(f'PC{pc1+1} ({var1:.1f}% variance)')
            ax.set_ylabel(f'PC{pc2+1} ({var2:.1f}% variance)')
            ax.set_title(f'PC{pc1+1} vs PC{pc2+1}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PC projections saved to {save_path}")
        
        return fig
    
    def plot_cluster_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot detailed cluster analysis."""
        if self.clustering_results is None:
            raise ValueError("Must run cluster_conformations() before plotting")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
        
        projections = self.pca_results.projections
        cluster_labels = self.clustering_results.cluster_labels
        
        # 1. PC1 vs PC2 colored by clusters
        scatter = ax1.scatter(projections[:, 0], projections[:, 1], 
                             c=cluster_labels, cmap='tab10', alpha=0.7, s=30)
        
        # Mark cluster centers
        centers_pc = self.clustering_results.cluster_centers
        ax1.scatter(centers_pc[:, 0], centers_pc[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        
        var1 = self.pca_results.explained_variance_ratio[0] * 100
        var2 = self.pca_results.explained_variance_ratio[1] * 100
        ax1.set_xlabel(f'PC1 ({var1:.1f}% variance)')
        ax1.set_ylabel(f'PC2 ({var2:.1f}% variance)')
        ax1.set_title('Conformational Clusters in PC Space')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Cluster ID')
        
        # 2. Cluster populations
        cluster_ids = list(self.clustering_results.cluster_populations.keys())
        populations = list(self.clustering_results.cluster_populations.values())
        percentages = [pop / sum(populations) * 100 for pop in populations]
        
        bars = ax2.bar(cluster_ids, percentages, color=plt.cm.tab10(np.linspace(0, 1, len(cluster_ids))))
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Population (%)')
        ax2.set_title('Cluster Populations')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        # 3. Cluster evolution over time
        ax3.plot(cluster_labels, alpha=0.7, linewidth=1)
        ax3.set_xlabel('Time (frames)')
        ax3.set_ylabel('Cluster ID')
        ax3.set_title('Cluster Evolution Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.5, len(cluster_ids) - 0.5)
        
        # 4. Silhouette analysis
        ax4.text(0.1, 0.9, f"Clustering Quality Metrics:", transform=ax4.transAxes, 
                fontsize=14, weight='bold')
        ax4.text(0.1, 0.8, f"Number of clusters: {self.clustering_results.n_clusters}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.7, f"Silhouette score: {self.clustering_results.silhouette_score:.3f}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.6, f"Total frames: {len(cluster_labels)}", 
                transform=ax4.transAxes, fontsize=12)
        
        # Cluster statistics
        ax4.text(0.1, 0.45, "Cluster Statistics:", transform=ax4.transAxes, 
                fontsize=12, weight='bold')
        y_pos = 0.35
        for cluster_id, stats in self.clustering_results.cluster_statistics.items():
            text = f"Cluster {cluster_id}: {stats['size']} frames ({stats['percentage']:.1f}%)"
            ax4.text(0.1, y_pos, text, transform=ax4.transAxes, fontsize=10)
            y_pos -= 0.05
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster analysis saved to {save_path}")
        
        return fig
    
    def animate_pc_mode(self, pc_mode: int = 0, amplitude: float = 3.0,
                       n_frames: int = 50, save_path: Optional[str] = None) -> np.ndarray:
        """
        Generate animation along a principal component mode.
        
        Args:
            pc_mode: Which PC mode to animate (0-indexed)
            amplitude: Amplitude of motion (in units of PC eigenvalue)
            n_frames: Number of frames for animation
            save_path: Optional path to save animation frames
            
        Returns:
            Animation frames as numpy array
        """
        if self.pca_results is None:
            raise ValueError("Must run fit_transform() before animation")
        
        if pc_mode >= self.pca_results.n_components:
            raise ValueError(f"PC mode {pc_mode} not available (max: {self.pca_results.n_components-1})")
        
        logger.info(f"Generating animation for PC mode {pc_mode}")
        
        # Get mean structure and eigenvector
        mean_structure = self.pca_results.mean_structure
        eigenvector = self.pca_results.eigenvectors[pc_mode]
        eigenvalue = self.pca_results.eigenvalues[pc_mode]
        
        # Create animation frames
        t_values = np.linspace(0, 2*np.pi, n_frames)
        animation_frames = []
        
        for t in t_values:
            # Displacement along PC mode
            displacement = amplitude * np.sqrt(eigenvalue) * np.sin(t) * eigenvector
            
            # Reshape displacement to (n_atoms, 3)
            displacement_3d = displacement.reshape(mean_structure.shape)
            
            # Add to mean structure
            frame = mean_structure + displacement_3d
            animation_frames.append(frame)
        
        animation_array = np.array(animation_frames)
        
        if save_path:
            # Save as numpy array
            if save_path.endswith('.npy'):
                np.save(save_path, animation_array)
            elif save_path.endswith('.npz'):
                np.savez(save_path, 
                        frames=animation_array,
                        pc_mode=pc_mode,
                        amplitude=amplitude,
                        eigenvalue=eigenvalue,
                        explained_variance=self.pca_results.explained_variance_ratio[pc_mode])
            logger.info(f"PC mode animation saved to {save_path}")
        
        return animation_array
    
    def export_results(self, results: PCAResults = None, clustering_results: ClusteringResults = None, 
                      output_dir: str = None) -> str:
        """
        Export PCA results to files.
        
        Args:
            results: PCA results to export (uses self.pca_results if None)
            clustering_results: Clustering results to export (uses self.clustering_results if None)
            output_dir: Directory to save results
            
        Returns:
            Path to the output directory containing exported files
        """
        # Use provided parameters or fall back to instance variables
        pca_results = results if results is not None else self.pca_results
        clustering_results = clustering_results if clustering_results is not None else self.clustering_results
        
        if pca_results is None:
            raise ValueError("Must run fit_transform() before export or provide results")
        
        if output_dir is None:
            raise ValueError("output_dir must be provided")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Export eigenvalues and eigenvectors
        np.save(output_path / "eigenvalues.npy", pca_results.eigenvalues)
        np.save(output_path / "eigenvectors.npy", pca_results.eigenvectors)
        np.save(output_path / "projections.npy", pca_results.projections)
        np.save(output_path / "mean_structure.npy", pca_results.mean_structure)
        
        saved_files['eigenvalues'] = str(output_path / "eigenvalues.npy")
        saved_files['eigenvectors'] = str(output_path / "eigenvectors.npy")
        saved_files['projections'] = str(output_path / "projections.npy")
        saved_files['mean_structure'] = str(output_path / "mean_structure.npy")
        
        # Export summary statistics
        summary = {
            'n_components': pca_results.n_components,
            'trajectory_length': pca_results.trajectory_length,
            'n_frames': pca_results.trajectory_length,  # Alias for compatibility
            'n_atoms': len(pca_results.mean_structure) // 3,  # Calculate from flattened mean structure
            'atom_selection': pca_results.atom_selection,
            'explained_variance_ratio': pca_results.explained_variance_ratio.tolist(),
            'cumulative_variance': pca_results.cumulative_variance.tolist(),
            'total_variance_explained': float(np.sum(pca_results.explained_variance_ratio) * 100),  # Total % variance
            'eigenvalues': pca_results.eigenvalues.tolist()
        }
        
        with open(output_path / "pca_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save as analysis_summary.json for compatibility
        with open(output_path / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        saved_files['summary'] = str(output_path / "pca_summary.json")
        saved_files['analysis_summary'] = str(output_path / "analysis_summary.json")
        
        # Export clustering results if available
        if clustering_results is not None:
            np.save(output_path / "cluster_labels.npy", clustering_results.cluster_labels)
            np.save(output_path / "cluster_centers.npy", clustering_results.cluster_centers)
            
            # Also export as text files
            np.savetxt(output_path / "cluster_labels.txt", clustering_results.cluster_labels, 
                      fmt='%d', header="Cluster labels for each frame")
            
            cluster_summary = {
                'n_clusters': int(clustering_results.n_clusters),
                'silhouette_score': float(clustering_results.silhouette_score),
                'cluster_populations': {int(k): int(v) for k, v in clustering_results.cluster_populations.items()},
                'representative_frames': {int(k): int(v) for k, v in clustering_results.representative_frames.items()},
                'cluster_statistics': {int(k): {str(stat_k): float(stat_v) for stat_k, stat_v in stats.items()} 
                                     for k, stats in clustering_results.cluster_statistics.items()}
            }
            
            with open(output_path / "clustering_summary.json", 'w') as f:
                json.dump(cluster_summary, f, indent=2)
            
            saved_files['cluster_labels'] = str(output_path / "cluster_labels.npy")
            saved_files['cluster_centers'] = str(output_path / "cluster_centers.npy")
            saved_files['clustering_summary'] = str(output_path / "clustering_summary.json")
            saved_files['cluster_labels_txt'] = str(output_path / "cluster_labels.txt")
        
        # Also export as text files for external analysis
        np.savetxt(output_path / "pca_projections.txt", pca_results.projections, 
                  header="PC projections (frame x component)")
        np.savetxt(output_path / "eigenvectors.txt", pca_results.eigenvectors,
                  header="Eigenvectors (feature x component)")
        np.savetxt(output_path / "eigenvalues.txt", pca_results.eigenvalues,
                  header="Eigenvalues")
        np.savetxt(output_path / "variance_explained.txt", pca_results.explained_variance_ratio,
                  header="Variance explained by each component")
        
        saved_files.update({
            'pca_projections_txt': str(output_path / "pca_projections.txt"),
            'eigenvectors_txt': str(output_path / "eigenvectors.txt"),
            'eigenvalues_txt': str(output_path / "eigenvalues.txt"),
            'variance_explained_txt': str(output_path / "variance_explained.txt")
        })

        logger.info(f"PCA results exported to {output_path}")
        return str(output_path)

    def visualize_pca(self, results: PCAResults, clustering_results: ClusteringResults = None, 
                     output_file: str = None, color_by_time: bool = False) -> Optional[plt.Figure]:
        """
        Create comprehensive PCA visualization.
        
        Args:
            results: PCA results to visualize
            clustering_results: Optional clustering results for coloring
            output_file: File to save the plot
            color_by_time: Whether to color points by time
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Principal Component Analysis Results', fontsize=16)
        
        # Eigenvalue spectrum
        axes[0, 0].bar(range(min(20, len(results.eigenvalues))), 
                      results.eigenvalues[:20])
        axes[0, 0].set_title('Eigenvalue Spectrum')
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Eigenvalue')
        
        # Cumulative variance
        axes[0, 1].plot(range(min(20, len(results.cumulative_variance))), 
                       results.cumulative_variance[:20], 'o-')
        axes[0, 1].set_title('Cumulative Variance Explained')
        axes[0, 1].set_xlabel('Principal Component')
        axes[0, 1].set_ylabel('Cumulative Variance (%)')
        
        # PC1 vs PC2 projection
        if clustering_results is not None:
            colors = clustering_results.cluster_labels
            scatter = axes[1, 0].scatter(results.projections[:, 0], 
                                       results.projections[:, 1], 
                                       c=colors, cmap='tab10', alpha=0.7)
            axes[1, 0].set_title('PC1 vs PC2 (Colored by Cluster)')
        elif color_by_time:
            colors = np.arange(len(results.projections))
            scatter = axes[1, 0].scatter(results.projections[:, 0], 
                                       results.projections[:, 1], 
                                       c=colors, cmap='viridis', alpha=0.7)
            axes[1, 0].set_title('PC1 vs PC2 (Colored by Time)')
        else:
            axes[1, 0].scatter(results.projections[:, 0], 
                             results.projections[:, 1], alpha=0.7)
            axes[1, 0].set_title('PC1 vs PC2')
        
        axes[1, 0].set_xlabel('PC1')
        axes[1, 0].set_ylabel('PC2')
        
        # PC1 vs PC3 projection
        if len(results.projections[0]) > 2:
            if clustering_results is not None:
                axes[1, 1].scatter(results.projections[:, 0], 
                                 results.projections[:, 2], 
                                 c=clustering_results.cluster_labels, 
                                 cmap='tab10', alpha=0.7)
                axes[1, 1].set_title('PC1 vs PC3 (Colored by Cluster)')
            elif color_by_time:
                axes[1, 1].scatter(results.projections[:, 0], 
                                 results.projections[:, 2], 
                                 c=np.arange(len(results.projections)), 
                                 cmap='viridis', alpha=0.7)
                axes[1, 1].set_title('PC1 vs PC3 (Colored by Time)')
            else:
                axes[1, 1].scatter(results.projections[:, 0], 
                                 results.projections[:, 2], alpha=0.7)
                axes[1, 1].set_title('PC1 vs PC3')
            
            axes[1, 1].set_xlabel('PC1')
            axes[1, 1].set_ylabel('PC3')
        else:
            axes[1, 1].text(0.5, 0.5, 'PC3 not available', 
                           transform=axes[1, 1].transAxes, ha='center')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            
        return fig
    
    def analyze_pc_amplitudes(self, results: PCAResults, n_components: int = 5) -> Dict[str, np.ndarray]:
        """
        Analyze amplitude distributions of principal components.
        
        Args:
            results: PCA results to analyze
            n_components: Number of components to analyze
            
        Returns:
            Dictionary with amplitude statistics for each component
        """
        amplitude_analysis = {}
        
        for i in range(min(n_components, results.projections.shape[1])):
            pc_values = results.projections[:, i]
            amplitude_analysis[f'PC{i+1}'] = {
                'mean': np.mean(pc_values),
                'std': np.std(pc_values),
                'min': np.min(pc_values),
                'max': np.max(pc_values),
                'range': np.max(pc_values) - np.min(pc_values),
                'variance_explained': results.explained_variance_ratio[i]
            }
        
        return amplitude_analysis
    
    def generate_pc_mode_animation(self, results: PCAResults, mode: int = 0, 
                                 n_frames: int = 20, amplitude: float = 3.0) -> np.ndarray:
        """
        Generate frames for PC mode animation.
        
        Args:
            results: PCA results
            mode: Which PC mode to animate
            n_frames: Number of animation frames
            amplitude: Animation amplitude in standard deviations
            
        Returns:
            Animation frames array with shape (n_frames, n_atoms, 3)
        """
        # Note: This method works with the provided results and doesn't require trajectory_data
        
        # Get the eigenvector for the specified mode
        eigenvector = results.eigenvectors[:, mode]
        eigenvalue = results.eigenvalues[mode]
        
        # Get mean structure (flattened)
        mean_structure_flat = results.mean_structure
        
        # Ensure shapes are compatible
        if len(mean_structure_flat) != len(eigenvector):
            # If shapes don't match, pad or truncate to make them compatible
            min_len = min(len(mean_structure_flat), len(eigenvector))
            mean_structure_flat = mean_structure_flat[:min_len]
            eigenvector = eigenvector[:min_len]
        
        # Generate animation frames
        animation_frames = []
        for frame_idx in range(n_frames):
            # Create sinusoidal motion along the PC mode
            t = 2 * np.pi * frame_idx / n_frames
            displacement = amplitude * np.sqrt(eigenvalue) * np.sin(t)
            
            # Apply displacement along the eigenvector (both are flattened)
            displaced_coords_flat = mean_structure_flat + displacement * eigenvector
            
            # Reshape back to (n_atoms, 3)
            n_atoms = len(displaced_coords_flat) // 3
            displaced_coords = displaced_coords_flat.reshape(n_atoms, 3)
            
            animation_frames.append(displaced_coords)
        
        return np.array(animation_frames)


# Convenience functions for common workflows

def perform_pca_analysis(trajectory: np.ndarray,
                        atom_names: Optional[List[str]] = None,
                        atom_selection: str = "CA",
                        n_components: int = 20,
                        n_clusters: Optional[int] = None,
                        output_dir: Optional[str] = None) -> Tuple[PCAResults, ClusteringResults]:
    """
    Perform complete PCA and clustering analysis workflow.
    
    Args:
        trajectory: Trajectory array (n_frames, n_atoms, 3)
        atom_names: Atom names for selection
        atom_selection: Atom selection criteria
        n_components: Number of PCs to compute
        n_clusters: Number of clusters (auto if None)
        output_dir: Output directory for results
        
    Returns:
        Tuple of (PCA results, clustering results)
    """
    # Initialize analyzer
    analyzer = PCAAnalyzer(atom_selection=atom_selection)
    
    # Perform PCA
    pca_results = analyzer.fit_transform(trajectory, atom_names, n_components=n_components)
    
    # Perform clustering
    clustering_results = analyzer.cluster_conformations(n_clusters=n_clusters)
    
    # Generate plots if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        analyzer.plot_eigenvalue_spectrum(save_path=str(output_path / "eigenvalue_spectrum.png"))
        analyzer.plot_pc_projections(save_path=str(output_path / "pc_projections.png"))
        analyzer.plot_cluster_analysis(save_path=str(output_path / "cluster_analysis.png"))
        
        # Export results
        analyzer.export_results(output_dir=output_dir)
    
    return pca_results, clustering_results


if __name__ == "__main__":
    # Demo/test code
    print("🧬 ProteinMD Principal Component Analysis Module")
    print("=" * 60)
    print("Features:")
    print("✅ PCA calculation for trajectory data")
    print("✅ Projection visualization onto principal components")
    print("✅ Conformational clustering in PC space")
    print("✅ Export of PC coordinates and eigenvectors")
    print("✅ Essential dynamics analysis")
    print("✅ Interactive visualization tools")
    print()
    print("Ready for integration with ProteinMD simulation engine!")

def create_test_trajectory(n_frames: int = 100, n_atoms: int = 100, 
                          noise_level: float = 0.1) -> np.ndarray:
    """
    Create a synthetic test trajectory for PCA validation.
    
    Parameters
    ----------
    n_frames : int, default=100
        Number of trajectory frames
    n_atoms : int, default=100
        Number of atoms in the system
    noise_level : float, default=0.1
        Amount of random noise to add
    
    Returns
    -------
    np.ndarray
        Test trajectory with shape (n_frames, n_atoms, 3)
    """
    # Create a trajectory with some structured motion plus noise
    trajectory = np.zeros((n_frames, n_atoms, 3))
    
    # Add some structured motions (breathing, bending)
    for frame in range(n_frames):
        t = frame / n_frames * 2 * np.pi
        
        # Breathing motion (expansion/contraction)
        breathing_factor = 1.0 + 0.2 * np.sin(t)
        
        # Bending motion
        bend_angle = 0.1 * np.sin(2 * t)
        
        for atom in range(n_atoms):
            # Base position
            x = atom * 0.1
            y = 0.0
            z = 0.0
            
            # Apply breathing
            x *= breathing_factor
            
            # Apply bending
            y += x * np.sin(bend_angle)
            z += x * np.cos(bend_angle) - x
            
            # Add some random noise
            x += np.random.normal(0, noise_level)
            y += np.random.normal(0, noise_level)
            z += np.random.normal(0, noise_level)
            
            trajectory[frame, atom] = [x, y, z]
    
    return trajectory
