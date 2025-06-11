"""
Hydrogen Bond Analysis for molecular dynamics simulations.

This module provides functions and classes for detecting hydrogen bonds,
tracking their evolution over time, analyzing their statistics, and
visualizing hydrogen bond networks in protein structures.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import logging
from pathlib import Path
import warnings
from collections import defaultdict, namedtuple
import csv
import json
from scipy.spatial.distance import cdist

# Configure logging
logger = logging.getLogger(__name__)

# Define hydrogen bond data structure
HydrogenBond = namedtuple('HydrogenBond', [
    'donor_atom_idx', 'hydrogen_idx', 'acceptor_atom_idx',
    'donor_residue', 'acceptor_residue', 'distance', 'angle',
    'bond_type', 'strength'
])

class HydrogenBondDetector:
    """
    Hydrogen bond detection using geometric criteria.
    
    Standard criteria:
    - D-H...A distance: < 3.5 Å (default)
    - D-H...A angle: > 120° (default)
    - H...A distance: < 2.5 Å (default)
    """
    
    def __init__(self, 
                 max_distance: float = 3.5,
                 min_angle: float = 120.0,
                 max_h_distance: float = 2.5):
        """
        Initialize hydrogen bond detector.
        
        Parameters
        ----------
        max_distance : float
            Maximum D-A distance in Angstroms
        min_angle : float
            Minimum D-H...A angle in degrees
        max_h_distance : float
            Maximum H...A distance in Angstroms
        """
        self.max_distance = max_distance
        self.min_angle = min_angle
        self.max_h_distance = max_h_distance
        
        # Define donor and acceptor atoms
        self.donor_atoms = {
            'N', 'O', 'S'  # Nitrogen, Oxygen, Sulfur can be donors
        }
        self.acceptor_atoms = {
            'N', 'O', 'S', 'F'  # Nitrogen, Oxygen, Sulfur, Fluorine can be acceptors
        }
        
        # Hydrogen bond strength classification
        self.strength_thresholds = {
            'very_strong': (2.5, 175),  # (max_dist, min_angle)
            'strong': (2.8, 150),
            'moderate': (3.1, 130),
            'weak': (3.5, 120)
        }
    
    def calculate_angle(self, donor_pos: np.ndarray, 
                       hydrogen_pos: np.ndarray,
                       acceptor_pos: np.ndarray) -> float:
        """
        Calculate D-H...A angle in degrees.
        
        The D-H...A angle is measured as the angle between the vectors H->D and H->A.
        A linear hydrogen bond (D-H-A) has an angle of 180°.
        
        Parameters
        ----------
        donor_pos : np.ndarray
            Donor atom position (3,)
        hydrogen_pos : np.ndarray
            Hydrogen atom position (3,)
        acceptor_pos : np.ndarray
            Acceptor atom position (3,)
            
        Returns
        -------
        float
            Angle in degrees (180° = perfect linear hydrogen bond)
        """
        # Vectors: H->D and H->A (both starting from hydrogen)
        hd_vector = donor_pos - hydrogen_pos
        ha_vector = acceptor_pos - hydrogen_pos
        
        # Calculate norms
        hd_norm = np.linalg.norm(hd_vector)
        ha_norm = np.linalg.norm(ha_vector)
        
        # Handle zero-length vectors
        if hd_norm == 0 or ha_norm == 0:
            return 0.0
        
        # Calculate angle between H->D and H->A vectors
        cos_angle = np.dot(hd_vector, ha_vector) / (hd_norm * ha_norm)
        
        # Clamp to valid range for arccos
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180.0 / np.pi
        
        return angle
    
    def classify_bond_strength(self, distance: float, angle: float) -> str:
        """
        Classify hydrogen bond strength based on geometric criteria.
        
        Parameters
        ----------
        distance : float
            D-A distance in Angstroms
        angle : float
            D-H...A angle in degrees
            
        Returns
        -------
        str
            Bond strength classification
        """
        for strength, (max_dist, min_ang) in self.strength_thresholds.items():
            if distance <= max_dist and angle >= min_ang:
                return strength
        return 'very_weak'
    
    def determine_bond_type(self, donor_residue: int, acceptor_residue: int) -> str:
        """
        Determine hydrogen bond type based on residue positions.
        
        Parameters
        ----------
        donor_residue : int
            Donor residue index
        acceptor_residue : int
            Acceptor residue index
            
        Returns
        -------
        str
            Bond type classification
        """
        res_diff = abs(donor_residue - acceptor_residue)
        
        if res_diff == 0:
            return 'intra_residue'
        elif res_diff == 1:
            return 'adjacent'
        elif res_diff <= 4:
            return 'short_range'
        else:
            return 'long_range'
    
    def detect_hydrogen_bonds(self, atoms: List, positions: np.ndarray) -> List[HydrogenBond]:
        """
        Detect hydrogen bonds in a single structure.
        
        Parameters
        ----------
        atoms : List
            List of atom objects with attributes: element, name, residue_id
        positions : np.ndarray
            Atomic coordinates with shape (N, 3) in Angstroms
            
        Returns
        -------
        List[HydrogenBond]
            List of detected hydrogen bonds
        """
        hydrogen_bonds = []
        
        # Find potential donors and acceptors
        donors = []
        acceptors = []
        hydrogens = []
        
        for i, atom in enumerate(atoms):
            element = getattr(atom, 'element', atom.atom_name[0])
            
            if element in self.donor_atoms:
                donors.append(i)
            if element in self.acceptor_atoms:
                acceptors.append(i)
            if element == 'H':
                hydrogens.append(i)
        
        # Find donor-hydrogen pairs
        donor_hydrogen_pairs = []
        for donor_idx in donors:
            donor_pos = positions[donor_idx]
            
            # Find hydrogens bonded to this donor (within 1.2 Å)
            for h_idx in hydrogens:
                h_pos = positions[h_idx]
                dist = np.linalg.norm(donor_pos - h_pos)
                
                if dist <= 1.2:  # Typical covalent bond length
                    donor_hydrogen_pairs.append((donor_idx, h_idx))
        
        # Check for hydrogen bonds
        for donor_idx, h_idx in donor_hydrogen_pairs:
            donor_pos = positions[donor_idx]
            h_pos = positions[h_idx]
            donor_residue = getattr(atoms[donor_idx], 'residue_id', 0)
            
            for acceptor_idx in acceptors:
                if acceptor_idx == donor_idx:  # Skip self
                    continue
                
                acceptor_pos = positions[acceptor_idx]
                acceptor_residue = getattr(atoms[acceptor_idx], 'residue_id', 0)
                
                # Calculate distances
                da_distance = np.linalg.norm(donor_pos - acceptor_pos)
                ha_distance = np.linalg.norm(h_pos - acceptor_pos)
                
                # Check distance criteria
                if (da_distance <= self.max_distance and 
                    ha_distance <= self.max_h_distance):
                    
                    # Calculate angle
                    angle = self.calculate_angle(donor_pos, h_pos, acceptor_pos)
                    
                    # Check angle criteria
                    if angle >= self.min_angle:
                        # Classify bond
                        strength = self.classify_bond_strength(da_distance, angle)
                        bond_type = self.determine_bond_type(donor_residue, acceptor_residue)
                        
                        # Create hydrogen bond
                        hbond = HydrogenBond(
                            donor_atom_idx=donor_idx,
                            hydrogen_idx=h_idx,
                            acceptor_atom_idx=acceptor_idx,
                            donor_residue=donor_residue,
                            acceptor_residue=acceptor_residue,
                            distance=da_distance,
                            angle=angle,
                            bond_type=bond_type,
                            strength=strength
                        )
                        hydrogen_bonds.append(hbond)
        
        return hydrogen_bonds


class HydrogenBondAnalyzer:
    """
    Comprehensive hydrogen bond analysis for molecular dynamics trajectories.
    """
    
    def __init__(self, detector: Optional[HydrogenBondDetector] = None):
        """
        Initialize hydrogen bond analyzer.
        
        Parameters
        ----------
        detector : HydrogenBondDetector, optional
            Custom hydrogen bond detector. If None, uses default settings.
        """
        self.detector = detector or HydrogenBondDetector()
        self.trajectory_bonds = []  # List of hydrogen bonds for each frame
        self.bond_statistics = {}
        self.lifetime_data = {}
        
    def analyze_trajectory(self, atoms: List, trajectory: np.ndarray) -> None:
        """
        Analyze hydrogen bonds throughout a trajectory.
        
        Parameters
        ----------
        atoms : List
            List of atom objects
        trajectory : np.ndarray
            Trajectory coordinates with shape (n_frames, n_atoms, 3) in Angstroms
        """
        logger.info(f"Analyzing hydrogen bonds for {len(trajectory)} frames")
        
        self.trajectory_bonds = []
        
        for frame_idx, positions in enumerate(trajectory):
            hbonds = self.detector.detect_hydrogen_bonds(atoms, positions)
            self.trajectory_bonds.append(hbonds)
            
            if frame_idx % 100 == 0:
                logger.debug(f"Processed frame {frame_idx}, found {len(hbonds)} H-bonds")
        
        # Calculate statistics
        self._calculate_statistics()
        self._calculate_lifetimes()
        
        logger.info("Hydrogen bond analysis completed")
    
    def _calculate_statistics(self) -> None:
        """Calculate hydrogen bond statistics."""
        self.bond_statistics = {
            'total_bonds_per_frame': [],
            'bond_types': defaultdict(list),
            'bond_strengths': defaultdict(list),
            'residue_pairs': defaultdict(list),
            'occupancy': defaultdict(list)
        }
        
        # Track unique bonds
        unique_bonds = set()
        bond_occurrences = defaultdict(list)
        
        for frame_idx, frame_bonds in enumerate(self.trajectory_bonds):
            self.bond_statistics['total_bonds_per_frame'].append(len(frame_bonds))
            
            for bond in frame_bonds:
                # Create unique bond identifier
                bond_id = (bond.donor_atom_idx, bond.acceptor_atom_idx)
                unique_bonds.add(bond_id)
                bond_occurrences[bond_id].append(frame_idx)
                
                # Collect statistics
                self.bond_statistics['bond_types'][bond.bond_type].append(bond.distance)
                self.bond_statistics['bond_strengths'][bond.strength].append(bond.distance)
                
                res_pair = (bond.donor_residue, bond.acceptor_residue)
                self.bond_statistics['residue_pairs'][res_pair].append(bond.distance)
        
        # Calculate occupancy for each unique bond
        n_frames = len(self.trajectory_bonds)
        for bond_id, frames in bond_occurrences.items():
            occupancy = len(frames) / n_frames
            self.bond_statistics['occupancy'][bond_id] = occupancy
    
    def _calculate_lifetimes(self) -> None:
        """Calculate hydrogen bond lifetimes."""
        # Track consecutive occurrences of each bond
        bond_states = defaultdict(list)  # bond_id -> list of frame indices
        
        # Collect all bond occurrences
        for frame_idx, frame_bonds in enumerate(self.trajectory_bonds):
            frame_bond_ids = set()
            for bond in frame_bonds:
                bond_id = (bond.donor_atom_idx, bond.acceptor_atom_idx)
                frame_bond_ids.add(bond_id)
            
            # Update bond states
            for bond_id in frame_bond_ids:
                bond_states[bond_id].append(frame_idx)
        
        # Calculate lifetimes from consecutive sequences
        self.lifetime_data = {}
        
        for bond_id, frames in bond_states.items():
            if not frames:
                continue
                
            # Find consecutive sequences
            lifetimes = []
            current_lifetime = 1
            
            for i in range(1, len(frames)):
                if frames[i] - frames[i-1] == 1:  # Consecutive frames
                    current_lifetime += 1
                else:
                    lifetimes.append(current_lifetime)
                    current_lifetime = 1
            
            # Add the last sequence
            lifetimes.append(current_lifetime)
            
            # Calculate statistics
            self.lifetime_data[bond_id] = {
                'lifetimes': lifetimes,
                'mean_lifetime': np.mean(lifetimes),
                'max_lifetime': max(lifetimes),
                'total_occurrences': len(frames),
                'formation_events': len(lifetimes)
            }
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for hydrogen bonds.
        
        Returns
        -------
        Dict[str, Any]
            Summary statistics
        """
        if not self.bond_statistics:
            raise ValueError("No analysis data available. Run analyze_trajectory first.")
        
        stats = {}
        
        # Overall statistics
        total_bonds = self.bond_statistics['total_bonds_per_frame']
        stats['mean_bonds_per_frame'] = np.mean(total_bonds)
        stats['std_bonds_per_frame'] = np.std(total_bonds)
        stats['max_bonds_per_frame'] = np.max(total_bonds)
        stats['min_bonds_per_frame'] = np.min(total_bonds)
        
        # Bond type distribution
        stats['bond_type_distribution'] = {}
        for bond_type, distances in self.bond_statistics['bond_types'].items():
            stats['bond_type_distribution'][bond_type] = {
                'count': len(distances),
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances)
            }
        
        # Bond strength distribution
        stats['bond_strength_distribution'] = {}
        for strength, distances in self.bond_statistics['bond_strengths'].items():
            stats['bond_strength_distribution'][strength] = {
                'count': len(distances),
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances)
            }
        
        # Lifetime statistics
        if self.lifetime_data:
            all_lifetimes = []
            all_occupancies = []
            
            for bond_data in self.lifetime_data.values():
                all_lifetimes.extend(bond_data['lifetimes'])
            
            for occupancy in self.bond_statistics['occupancy'].values():
                all_occupancies.append(occupancy)
            
            stats['lifetime_statistics'] = {
                'mean_lifetime': np.mean(all_lifetimes),
                'std_lifetime': np.std(all_lifetimes),
                'max_lifetime': np.max(all_lifetimes),
                'mean_occupancy': np.mean(all_occupancies),
                'std_occupancy': np.std(all_occupancies)
            }
        
        return stats
    
    def plot_bond_evolution(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot hydrogen bond evolution over time.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        if not self.bond_statistics:
            raise ValueError("No analysis data available. Run analyze_trajectory first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total bonds per frame
        ax1 = axes[0, 0]
        frames = range(len(self.bond_statistics['total_bonds_per_frame']))
        ax1.plot(frames, self.bond_statistics['total_bonds_per_frame'], 'b-', alpha=0.7)
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Number of H-bonds')
        ax1.set_title('H-bond Count Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Bond type distribution
        ax2 = axes[0, 1]
        bond_types = list(self.bond_statistics['bond_types'].keys())
        type_counts = [len(self.bond_statistics['bond_types'][bt]) for bt in bond_types]
        ax2.bar(bond_types, type_counts, alpha=0.7)
        ax2.set_xlabel('Bond Type')
        ax2.set_ylabel('Total Occurrences')
        ax2.set_title('H-bond Type Distribution')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Bond strength distribution
        ax3 = axes[1, 0]
        strengths = list(self.bond_statistics['bond_strengths'].keys())
        strength_counts = [len(self.bond_statistics['bond_strengths'][s]) for s in strengths]
        colors = ['red', 'orange', 'yellow', 'lightblue', 'gray']
        ax3.bar(strengths, strength_counts, color=colors[:len(strengths)], alpha=0.7)
        ax3.set_xlabel('Bond Strength')
        ax3.set_ylabel('Total Occurrences')
        ax3.set_title('H-bond Strength Distribution')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Lifetime distribution
        ax4 = axes[1, 1]
        if self.lifetime_data:
            all_lifetimes = []
            for bond_data in self.lifetime_data.values():
                all_lifetimes.extend(bond_data['lifetimes'])
            
            ax4.hist(all_lifetimes, bins=20, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Lifetime (frames)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('H-bond Lifetime Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"H-bond evolution plot saved to {save_path}")
        
        return fig
    
    def plot_residue_network(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot hydrogen bond network between residues.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        if not self.bond_statistics:
            raise ValueError("No analysis data available. Run analyze_trajectory first.")
        
        # Create adjacency matrix for residue pairs
        residue_pairs = self.bond_statistics['residue_pairs']
        
        if not residue_pairs:
            logger.warning("No residue pair data available for network plot")
            return plt.figure()
        
        # Get all unique residues
        all_residues = set()
        for (res1, res2), _ in residue_pairs.items():
            all_residues.add(res1)
            all_residues.add(res2)
        
        residue_list = sorted(all_residues)
        n_residues = len(residue_list)
        
        # Create adjacency matrix
        adjacency = np.zeros((n_residues, n_residues))
        
        for (res1, res2), distances in residue_pairs.items():
            i = residue_list.index(res1)
            j = residue_list.index(res2)
            # Use count as weight
            adjacency[i, j] = len(distances)
            adjacency[j, i] = len(distances)  # Symmetric
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(adjacency, cmap='Blues', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(n_residues))
        ax.set_yticks(range(n_residues))
        ax.set_xticklabels(residue_list)
        ax.set_yticklabels(residue_list)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('H-bond Count')
        
        ax.set_xlabel('Residue Index')
        ax.set_ylabel('Residue Index')
        ax.set_title('Hydrogen Bond Network Between Residues')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"H-bond network plot saved to {save_path}")
        
        return fig
    
    def export_statistics_csv(self, filepath: str) -> None:
        """
        Export hydrogen bond statistics to CSV file.
        
        Parameters
        ----------
        filepath : str
            Path to save CSV file
        """
        if not self.bond_statistics:
            raise ValueError("No analysis data available. Run analyze_trajectory first.")
        
        # Prepare data for CSV
        rows = []
        
        # Add individual bond data
        for frame_idx, frame_bonds in enumerate(self.trajectory_bonds):
            for bond in frame_bonds:
                rows.append({
                    'frame': frame_idx,
                    'donor_atom': bond.donor_atom_idx,
                    'hydrogen_atom': bond.hydrogen_idx,
                    'acceptor_atom': bond.acceptor_atom_idx,
                    'donor_residue': bond.donor_residue,
                    'acceptor_residue': bond.acceptor_residue,
                    'distance': bond.distance,
                    'angle': bond.angle,
                    'bond_type': bond.bond_type,
                    'strength': bond.strength
                })
        
        # Write CSV
        if rows:
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['frame', 'donor_atom', 'hydrogen_atom', 'acceptor_atom', 
                             'donor_residue', 'acceptor_residue', 'distance', 'angle', 
                             'bond_type', 'strength']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            # Write empty CSV with headers
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['frame', 'donor_atom', 'hydrogen_atom', 'acceptor_atom', 
                             'donor_residue', 'acceptor_residue', 'distance', 'angle', 
                             'bond_type', 'strength']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
        
        logger.info(f"H-bond statistics exported to {filepath}")
    
    def export_lifetime_analysis(self, filepath: str) -> None:
        """
        Export detailed lifetime analysis to JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to save JSON file
        """
        if not self.lifetime_data:
            raise ValueError("No lifetime data available. Run analyze_trajectory first.")
        
        # Convert bond IDs to strings for JSON serialization
        export_data = {}
        
        for (donor_idx, acceptor_idx), bond_data in self.lifetime_data.items():
            bond_key = f"bond_{donor_idx}_{acceptor_idx}"
            export_data[bond_key] = {
                'donor_atom_idx': int(donor_idx),
                'acceptor_atom_idx': int(acceptor_idx),
                'lifetimes': [int(x) for x in bond_data['lifetimes']],
                'mean_lifetime': float(bond_data['mean_lifetime']),
                'max_lifetime': int(bond_data['max_lifetime']),
                'total_occurrences': int(bond_data['total_occurrences']),
                'formation_events': int(bond_data['formation_events']),
                'occupancy': float(self.bond_statistics['occupancy'].get((donor_idx, acceptor_idx), 0.0))
            }
        
        # Add summary statistics (convert numpy types to native Python types)
        summary = self.get_summary_statistics()
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(x) for x in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        export_data['summary'] = convert_numpy_types(summary)
        
        with open(filepath, 'w') as jsonfile:
            json.dump(export_data, jsonfile, indent=2)
        
        logger.info(f"H-bond lifetime analysis exported to {filepath}")


# Factory functions for common analyses
def analyze_hydrogen_bonds(atoms: List, trajectory: np.ndarray, 
                          detector_params: Optional[Dict] = None) -> HydrogenBondAnalyzer:
    """
    Convenience function for hydrogen bond analysis.
    
    Parameters
    ----------
    atoms : List
        List of atom objects
    trajectory : np.ndarray
        Trajectory coordinates (n_frames, n_atoms, 3) in Angstroms
    detector_params : Dict, optional
        Parameters for hydrogen bond detector
    
    Returns
    -------
    HydrogenBondAnalyzer
        Analyzer with completed analysis
    """
    # Create detector with custom parameters if provided
    detector = HydrogenBondDetector(**(detector_params or {}))
    
    # Create analyzer and run analysis
    analyzer = HydrogenBondAnalyzer(detector)
    analyzer.analyze_trajectory(atoms, trajectory)
    
    return analyzer


def quick_hydrogen_bond_summary(atoms: List, trajectory: np.ndarray) -> Dict[str, Any]:
    """
    Quick hydrogen bond analysis with summary statistics.
    
    Parameters
    ----------
    atoms : List
        List of atom objects
    trajectory : np.ndarray
        Trajectory coordinates (n_frames, n_atoms, 3) in Angstroms
    
    Returns
    -------
    Dict[str, Any]
        Summary statistics
    """
    analyzer = analyze_hydrogen_bonds(atoms, trajectory)
    return analyzer.get_summary_statistics()
