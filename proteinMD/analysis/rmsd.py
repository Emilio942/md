"""
Root Mean Square Deviation (RMSD) analysis for molecular dynamics simulations.

This module provides functions and classes for calculating RMSD between protein
structures, analyzing structural evolution over time, and comparing different
conformations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import warnings

# Configure logging
logger = logging.getLogger(__name__)


def kabsch_algorithm(coords1: np.ndarray, coords2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kabsch algorithm for optimal superposition of two coordinate sets.
    
    This algorithm finds the optimal rotation matrix to align coords1 onto coords2
    by minimizing the RMSD between them.
    
    Parameters
    ----------
    coords1 : np.ndarray
        First coordinate set with shape (N, 3)
    coords2 : np.ndarray  
        Second coordinate set with shape (N, 3)
    
    Returns
    -------
    rotation_matrix : np.ndarray
        Optimal rotation matrix (3, 3)
    aligned_coords1 : np.ndarray
        coords1 after optimal rotation and translation
    """
    # Center both coordinate sets
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    
    centered1 = coords1 - centroid1
    centered2 = coords2 - centroid2
    
    # Calculate the covariance matrix
    H = centered1.T @ centered2
    
    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Calculate rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (determinant should be 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation and translation
    aligned_coords1 = (centered1 @ R.T) + centroid2
    
    return R, aligned_coords1


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray, 
                  align: bool = True) -> float:
    """
    Calculate Root Mean Square Deviation between two coordinate sets.
    
    RMSD = sqrt(sum((r_i - r_ref_i)^2) / N)
    
    Parameters
    ----------
    coords1 : np.ndarray
        First coordinate set with shape (N, 3)
    coords2 : np.ndarray
        Second coordinate set with shape (N, 3)
    align : bool, optional
        Whether to perform optimal alignment before RMSD calculation (default: True)
    
    Returns
    -------
    float
        RMSD value in the same units as input coordinates
    """
    if coords1.shape != coords2.shape:
        raise ValueError(f"Coordinate shapes must match: {coords1.shape} vs {coords2.shape}")
    
    if coords1.shape[1] != 3:
        raise ValueError(f"Coordinates must be Nx3 arrays, got shape {coords1.shape}")
    
    if len(coords1) == 0:
        return 0.0
    
    if align:
        _, aligned_coords1 = kabsch_algorithm(coords1, coords2)
        diff = aligned_coords1 - coords2
    else:
        diff = coords1 - coords2
    
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd


def align_structures(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Align mobile structure to reference using optimal superposition.
    
    Parameters
    ----------
    mobile : np.ndarray
        Mobile coordinate set to be aligned, shape (N, 3)
    reference : np.ndarray
        Reference coordinate set, shape (N, 3)
    
    Returns
    -------
    np.ndarray
        Aligned mobile coordinates
    """
    _, aligned_mobile = kabsch_algorithm(mobile, reference)
    return aligned_mobile


class RMSDAnalyzer:
    """
    RMSD analysis class for MD trajectories.
    
    This class provides comprehensive RMSD analysis capabilities including:
    - Time series RMSD calculation
    - Multiple reference structure comparison
    - Different atom selection options
    - Statistical analysis and visualization
    """
    
    def __init__(self, reference_structure: Optional[np.ndarray] = None,
                 atom_selection: str = 'backbone'):
        """
        Initialize RMSD analyzer.
        
        Parameters
        ----------
        reference_structure : np.ndarray, optional
            Reference structure coordinates, shape (N, 3)
        atom_selection : str, optional
            Type of atoms to include in RMSD calculation:
            - 'backbone': backbone atoms (N, CA, C, O)
            - 'heavy': all heavy atoms
            - 'ca': alpha carbon atoms only
            - 'all': all atoms (default: 'backbone')
        """
        # Handle reference structure setup
        if reference_structure is not None:
            try:
                if hasattr(reference_structure, 'atoms') and hasattr(reference_structure.atoms, '__iter__'):
                    self.reference_structure = np.array([atom.position for atom in reference_structure.atoms])
                elif hasattr(reference_structure, 'positions'):
                    self.reference_structure = reference_structure.positions.copy()
                elif isinstance(reference_structure, np.ndarray):
                    self.reference_structure = reference_structure.copy()
                else:
                    self.reference_structure = reference_structure
            except (TypeError, AttributeError):
                # Fall back to positions if atoms iteration fails
                if hasattr(reference_structure, 'positions'):
                    self.reference_structure = reference_structure.positions.copy()
                else:
                    self.reference_structure = reference_structure
        else:
            self.reference_structure = None
            
        self.atom_selection = atom_selection
        
        # Storage for results
        self.rmsd_data = {}
        self.time_data = {}
        self.trajectory_names = []
        
        # Analysis parameters
        self.align_structures = True
        self.center_structures = True
        
        logger.info(f"Initialized RMSD analyzer with atom selection: {atom_selection}")
    
    def set_reference(self, reference_structure: np.ndarray) -> None:
        """
        Set the reference structure for RMSD calculations.
        
        Parameters
        ----------
        reference_structure : np.ndarray
            Reference coordinates, shape (N, 3)
        """
        self.reference_structure = reference_structure.copy()
        logger.info(f"Set reference structure with {len(reference_structure)} atoms")
    
    def calculate_trajectory_rmsd(self, trajectory: List[np.ndarray], 
                                times: Optional[List[float]] = None,
                                trajectory_name: str = "trajectory") -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate RMSD for each frame in a trajectory.
        
        Parameters
        ----------
        trajectory : list of np.ndarray
            List of coordinate arrays, each with shape (N, 3)
        times : list of float, optional
            Time points for each frame
        trajectory_name : str, optional
            Name identifier for this trajectory
            
        Returns
        -------
        times : np.ndarray
            Time points
        rmsd_values : np.ndarray
            RMSD values for each frame
        """
        if self.reference_structure is None:
            # Use first frame as reference
            self.reference_structure = trajectory[0].copy()
            logger.info("Using first frame as reference structure")
        
        if times is None:
            times = np.arange(len(trajectory), dtype=float)
        
        rmsd_values = []
        
        for i, frame in enumerate(trajectory):
            try:
                rmsd = calculate_rmsd(frame, self.reference_structure, 
                                    align=self.align_structures)
                rmsd_values.append(rmsd)
            except Exception as e:
                logger.warning(f"Error calculating RMSD for frame {i}: {e}")
                rmsd_values.append(np.nan)
        
        rmsd_values = np.array(rmsd_values)
        times = np.array(times)
        
        # Store results
        self.rmsd_data[trajectory_name] = rmsd_values
        self.time_data[trajectory_name] = times
        if trajectory_name not in self.trajectory_names:
            self.trajectory_names.append(trajectory_name)
        
        logger.info(f"Calculated RMSD for {len(trajectory)} frames, "
                   f"mean RMSD: {np.nanmean(rmsd_values):.3f}")
        
        return times, rmsd_values
    
    def calculate_pairwise_rmsd(self, structures: List[np.ndarray],
                               structure_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Calculate pairwise RMSD matrix between multiple structures.
        
        Parameters
        ----------
        structures : list of np.ndarray
            List of coordinate arrays to compare
        structure_names : list of str, optional
            Names for each structure
            
        Returns
        -------
        np.ndarray
            Symmetric RMSD matrix, shape (N, N)
        """
        n_structures = len(structures)
        rmsd_matrix = np.zeros((n_structures, n_structures))
        
        if structure_names is None:
            structure_names = [f"Structure_{i}" for i in range(n_structures)]
        
        for i in range(n_structures):
            for j in range(i + 1, n_structures):
                rmsd = calculate_rmsd(structures[i], structures[j], 
                                    align=self.align_structures)
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd  # Symmetric matrix
        
        self.pairwise_rmsd = rmsd_matrix
        self.structure_names = structure_names
        
        logger.info(f"Calculated pairwise RMSD for {n_structures} structures")
        return rmsd_matrix
    
    def plot_rmsd_time_series(self, trajectory_names: Optional[List[str]] = None,
                             figsize: Tuple[float, float] = (12, 8),
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot RMSD vs time for trajectories.
        
        Parameters
        ----------
        trajectory_names : list of str, optional
            Names of trajectories to plot (default: all)
        figsize : tuple of float, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        if not self.rmsd_data:
            raise ValueError("No RMSD data available. Run calculate_trajectory_rmsd first.")
        
        if trajectory_names is None:
            trajectory_names = self.trajectory_names
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # Main RMSD plot
        ax_main = axes[0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectory_names)))
        
        stats_data = {}
        
        for i, traj_name in enumerate(trajectory_names):
            if traj_name not in self.rmsd_data:
                logger.warning(f"No data for trajectory: {traj_name}")
                continue
                
            times = self.time_data[traj_name]
            rmsd_vals = self.rmsd_data[traj_name]
            
            # Remove NaN values for plotting
            valid_mask = ~np.isnan(rmsd_vals)
            times_clean = times[valid_mask]
            rmsd_clean = rmsd_vals[valid_mask]
            
            ax_main.plot(times_clean, rmsd_clean, 
                        color=colors[i], linewidth=2, label=traj_name, alpha=0.8)
            
            # Calculate statistics
            stats_data[traj_name] = {
                'mean': np.nanmean(rmsd_vals),
                'std': np.nanstd(rmsd_vals),
                'min': np.nanmin(rmsd_vals),
                'max': np.nanmax(rmsd_vals)
            }
        
        ax_main.set_xlabel('Time (ps)')
        ax_main.set_ylabel('RMSD (Å)')
        ax_main.set_title('RMSD vs Time', fontsize=14, fontweight='bold')
        ax_main.grid(True, alpha=0.3)
        ax_main.legend()
        
        # Statistics subplot
        ax_stats = axes[1]
        ax_stats.axis('off')
        
        stats_text = "Statistics:\n"
        for traj_name, stats in stats_data.items():
            stats_text += f"{traj_name}: "
            stats_text += f"μ={stats['mean']:.2f}±{stats['std']:.2f} Å, "
            stats_text += f"range=[{stats['min']:.2f}, {stats['max']:.2f}] Å\n"
        
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved RMSD plot to {save_path}")
        
        return fig
    
    def plot_pairwise_rmsd_matrix(self, figsize: Tuple[float, float] = (10, 8),
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot pairwise RMSD matrix as a heatmap.
        
        Parameters
        ----------
        figsize : tuple of float, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        if not hasattr(self, 'pairwise_rmsd'):
            raise ValueError("No pairwise RMSD data. Run calculate_pairwise_rmsd first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(self.pairwise_rmsd, cmap='viridis', aspect='equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('RMSD (Å)', rotation=270, labelpad=20)
        
        # Set ticks and labels
        n_structures = len(self.structure_names)
        ax.set_xticks(range(n_structures))
        ax.set_yticks(range(n_structures))
        ax.set_xticklabels(self.structure_names, rotation=45, ha='right')
        ax.set_yticklabels(self.structure_names)
        
        # Add text annotations
        for i in range(n_structures):
            for j in range(n_structures):
                text = ax.text(j, i, f'{self.pairwise_rmsd[i, j]:.2f}',
                             ha="center", va="center", color="white", fontsize=8)
        
        ax.set_title('Pairwise RMSD Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved pairwise RMSD matrix to {save_path}")
        
        return fig
    
    def calculate_running_average(self, window_size: int = 10,
                                trajectory_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate running average of RMSD values.
        
        Parameters
        ----------
        window_size : int, optional
            Size of the averaging window (default: 10)
        trajectory_name : str, optional
            Name of trajectory to analyze (default: first available)
            
        Returns
        -------
        times : np.ndarray
            Time points for averaged data
        averaged_rmsd : np.ndarray
            Running average RMSD values
        """
        if trajectory_name is None:
            trajectory_name = self.trajectory_names[0] if self.trajectory_names else None
        
        if trajectory_name not in self.rmsd_data:
            raise ValueError(f"No RMSD data for trajectory: {trajectory_name}")
        
        rmsd_vals = self.rmsd_data[trajectory_name]
        times = self.time_data[trajectory_name]
        
        # Calculate running average
        averaged_rmsd = []
        averaged_times = []
        
        for i in range(len(rmsd_vals) - window_size + 1):
            window = rmsd_vals[i:i + window_size]
            if not np.any(np.isnan(window)):
                averaged_rmsd.append(np.mean(window))
                averaged_times.append(np.mean(times[i:i + window_size]))
        
        return np.array(averaged_times), np.array(averaged_rmsd)
    
    def export_data(self, filename: str, trajectory_names: Optional[List[str]] = None) -> None:
        """
        Export RMSD data to CSV file.
        
        Parameters
        ----------
        filename : str
            Output CSV filename
        trajectory_names : list of str, optional
            Names of trajectories to export (default: all)
        """
        if not self.rmsd_data:
            logger.warning("No RMSD data to export")
            return
        
        if trajectory_names is None:
            trajectory_names = self.trajectory_names
        
        try:
            import pandas as pd
            
            # Create DataFrame
            data = {}
            max_length = 0
            
            for traj_name in trajectory_names:
                if traj_name in self.rmsd_data:
                    times = self.time_data[traj_name]
                    rmsd_vals = self.rmsd_data[traj_name]
                    max_length = max(max_length, len(times))
                    
                    data[f'{traj_name}_time'] = times
                    data[f'{traj_name}_rmsd'] = rmsd_vals
            
            # Pad shorter arrays with NaN
            for key, values in data.items():
                if len(values) < max_length:
                    padded = np.full(max_length, np.nan)
                    padded[:len(values)] = values
                    data[key] = padded
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
        except ImportError:
            # Fallback to basic CSV writing
            import csv
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                header = []
                for traj_name in trajectory_names:
                    if traj_name in self.rmsd_data:
                        header.extend([f'{traj_name}_time', f'{traj_name}_rmsd'])
                writer.writerow(header)
                
                # Write data
                max_length = max(len(self.rmsd_data[traj]) 
                               for traj in trajectory_names 
                               if traj in self.rmsd_data)
                
                for i in range(max_length):
                    row = []
                    for traj_name in trajectory_names:
                        if traj_name in self.rmsd_data:
                            if i < len(self.time_data[traj_name]):
                                row.extend([self.time_data[traj_name][i], 
                                          self.rmsd_data[traj_name][i]])
                            else:
                                row.extend(['', ''])
                    writer.writerow(row)
        
        logger.info(f"Exported RMSD data to {filename}")
    
    def get_statistics(self, trajectory_name: Optional[str] = None) -> Dict[str, float]:
        """
        Get statistical summary of RMSD data.
        
        Parameters
        ----------
        trajectory_name : str, optional
            Name of trajectory to analyze (default: all trajectories)
            
        Returns
        -------
        dict
            Statistics dictionary with mean, std, min, max, etc.
        """
        if trajectory_name is None:
            # Combine all trajectories
            all_rmsd = []
            for traj_name in self.trajectory_names:
                rmsd_vals = self.rmsd_data[traj_name]
                all_rmsd.extend(rmsd_vals[~np.isnan(rmsd_vals)])
            rmsd_data = np.array(all_rmsd)
        else:
            if trajectory_name not in self.rmsd_data:
                raise ValueError(f"No data for trajectory: {trajectory_name}")
            rmsd_data = self.rmsd_data[trajectory_name]
            rmsd_data = rmsd_data[~np.isnan(rmsd_data)]
        
        if len(rmsd_data) == 0:
            return {}
        
        return {
            'mean': float(np.mean(rmsd_data)),
            'std': float(np.std(rmsd_data)),
            'min': float(np.min(rmsd_data)),
            'max': float(np.max(rmsd_data)),
            'median': float(np.median(rmsd_data)),
            'q25': float(np.percentile(rmsd_data, 25)),
            'q75': float(np.percentile(rmsd_data, 75)),
            'n_frames': len(rmsd_data)
        }
    
    def calculate_rmsd(self, structure) -> float:
        """
        Calculate RMSD between a structure and the reference structure.
        
        Compatibility method for tests.
        
        Parameters
        ----------
        structure : object with atoms attribute or np.ndarray
            Structure to compare with reference
            
        Returns
        -------
        float
            RMSD value
        """
        # Handle reference structure setup
        if self.reference_structure is None:
            raise ValueError("Reference structure not set")
            
        # Extract coordinates from reference structure
        if hasattr(self.reference_structure, 'atoms'):
            ref_coords = np.array([atom.position for atom in self.reference_structure.atoms])
        elif hasattr(self.reference_structure, 'positions'):
            ref_coords = self.reference_structure.positions
        elif isinstance(self.reference_structure, np.ndarray):
            ref_coords = self.reference_structure
        else:
            raise ValueError("Invalid reference structure format")
        
        # Extract coordinates from input structure
        if hasattr(structure, 'atoms'):
            coords = np.array([atom.position for atom in structure.atoms])
        elif hasattr(structure, 'positions'):
            coords = structure.positions
        elif isinstance(structure, np.ndarray):
            coords = structure
        else:
            raise ValueError("Invalid structure format")
            
        return calculate_rmsd(coords, ref_coords, align=self.align_structures)
    
    def align_structure(self, structure):
        """
        Align a structure to the reference structure.
        
        Compatibility method for tests.
        
        Parameters
        ----------
        structure : object with atoms attribute or np.ndarray
            Structure to align
            
        Returns
        -------
        aligned structure (same type as input)
        """
        # Handle reference structure setup
        if self.reference_structure is None:
            raise ValueError("Reference structure not set")
            
        # Extract coordinates from reference structure
        if hasattr(self.reference_structure, 'atoms'):
            ref_coords = np.array([atom.position for atom in self.reference_structure.atoms])
        elif hasattr(self.reference_structure, 'positions'):
            ref_coords = self.reference_structure.positions
        elif isinstance(self.reference_structure, np.ndarray):
            ref_coords = self.reference_structure
        else:
            raise ValueError("Invalid reference structure format")
        
        # Extract coordinates from structure and align
        if hasattr(structure, 'atoms'):
            coords = np.array([atom.position for atom in structure.atoms])
            aligned_coords = align_structures(coords, ref_coords)
            # Update atom positions
            for i, atom in enumerate(structure.atoms):
                atom.position = aligned_coords[i]
            return structure
        elif hasattr(structure, 'positions'):
            coords = structure.positions
            aligned_coords = align_structures(coords, ref_coords)
            structure.positions = aligned_coords
            return structure
        elif isinstance(structure, np.ndarray):
            return align_structures(structure, ref_coords)
        else:
            raise ValueError("Invalid structure format")
    
    def analyze_trajectory(self, trajectory):
        """
        Analyze trajectory and return RMSD values.
        
        Compatibility method for tests.
        
        Parameters
        ----------
        trajectory : object with frames attribute
            Trajectory to analyze
            
        Returns
        -------
        list
            RMSD values for each frame
        """
        if hasattr(trajectory, 'frames'):
            # Convert frames to coordinate arrays
            coord_trajectory = []
            for frame in trajectory.frames:
                try:
                    # Try to access atoms if they exist and are iterable
                    if hasattr(frame, 'atoms') and hasattr(frame.atoms, '__iter__'):
                        coords = np.array([atom.position for atom in frame.atoms])
                    elif hasattr(frame, 'positions'):
                        coords = frame.positions
                    else:
                        coords = frame
                except (TypeError, AttributeError):
                    # Fall back to positions if atoms iteration fails
                    if hasattr(frame, 'positions'):
                        coords = frame.positions
                    else:
                        coords = frame
                coord_trajectory.append(coords)
            
            times, rmsd_values = self.calculate_trajectory_rmsd(coord_trajectory)
            return rmsd_values.tolist()
        else:
            raise ValueError("Invalid trajectory format")


def create_rmsd_analyzer(reference_structure: Optional[np.ndarray] = None,
                        atom_selection: str = 'backbone') -> RMSDAnalyzer:
    """
    Convenience function to create an RMSD analyzer.
    
    Parameters
    ----------
    reference_structure : np.ndarray, optional
        Reference structure coordinates
    atom_selection : str, optional
        Type of atoms to include in analysis
        
    Returns
    -------
    RMSDAnalyzer
        Configured RMSD analyzer
    """
    return RMSDAnalyzer(reference_structure=reference_structure,
                       atom_selection=atom_selection)


if __name__ == "__main__":
    # Example usage and testing
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    
    # Create test data
    print("Testing RMSD analysis...")
    
    # Generate synthetic protein trajectory
    n_atoms = 100
    n_frames = 50
    
    # Reference structure
    ref_structure = np.random.random((n_atoms, 3)) * 10.0
    
    # Create trajectory with gradual structural change
    trajectory = []
    for i in range(n_frames):
        # Add noise and gradual drift
        noise = np.random.normal(0, 0.1, (n_atoms, 3))
        drift = np.random.normal(0, 0.02 * i, (n_atoms, 3))
        frame = ref_structure + noise + drift
        trajectory.append(frame)
    
    # Test RMSD analysis
    analyzer = create_rmsd_analyzer(reference_structure=ref_structure)
    
    # Calculate trajectory RMSD
    times = np.arange(n_frames) * 0.1  # 0.1 ps timesteps
    times, rmsd_values = analyzer.calculate_trajectory_rmsd(
        trajectory, times, "test_trajectory"
    )
    
    # Plot results
    fig = analyzer.plot_rmsd_time_series(save_path="test_rmsd_analysis.png")
    
    # Test pairwise RMSD
    test_structures = [trajectory[0], trajectory[10], trajectory[25], trajectory[-1]]
    structure_names = ["Frame_0", "Frame_10", "Frame_25", "Frame_49"]
    rmsd_matrix = analyzer.calculate_pairwise_rmsd(test_structures, structure_names)
    
    fig2 = analyzer.plot_pairwise_rmsd_matrix(save_path="test_pairwise_rmsd.png")
    
    # Export data
    analyzer.export_data("test_rmsd_data.csv")
    
    # Print statistics
    stats = analyzer.get_statistics()
    print(f"RMSD Statistics: {stats}")
    
    print("RMSD analysis test completed successfully!")
    print(f"Mean RMSD: {stats['mean']:.3f} ± {stats['std']:.3f} Å")
    print(f"RMSD range: [{stats['min']:.3f}, {stats['max']:.3f}] Å")
