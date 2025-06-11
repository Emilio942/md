"""
Secondary Structure Analysis Module

This module provides DSSP-like functionality to analyze and track secondary
structure elements (α-helices, β-sheets, turns, coils) in protein structures
over molecular dynamics trajectories.

The implementation includes:
- DSSP-like algorithm for secondary structure assignment
- Tracking of structure changes over time
- Statistical analysis of structure populations
- Color-coded visualization of structure evolution
- Export capabilities for timeline data

Author: ProteinMD Development Team
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, Normalize
from typing import List, Tuple, Dict, Optional, Union, Any
import logging
from pathlib import Path
import json
import csv
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Secondary structure definitions
SS_TYPES = {
    'H': {'name': 'Alpha-Helix', 'color': '#FF6B6B', 'priority': 1},
    'G': {'name': '3-10 Helix', 'color': '#FF8E53', 'priority': 2},
    'I': {'name': 'Pi-Helix', 'color': '#FF6B9D', 'priority': 3},
    'E': {'name': 'Beta-Strand', 'color': '#4ECDC4', 'priority': 4},
    'B': {'name': 'Beta-Bridge', 'color': '#45B7D1', 'priority': 5},
    'T': {'name': 'Turn', 'color': '#96CEB4', 'priority': 6},
    'S': {'name': 'Bend', 'color': '#FFEAA7', 'priority': 7},
    'C': {'name': 'Coil', 'color': '#DDA0DD', 'priority': 8},
    '-': {'name': 'Unassigned', 'color': '#CCCCCC', 'priority': 9}
}

# Hydrogen bond parameters
HB_DISTANCE_CUTOFF = 0.35  # nm
HB_ANGLE_CUTOFF = 30.0     # degrees
HB_ENERGY_CUTOFF = -0.5    # kcal/mol


def calculate_dihedral_angle(p1: np.ndarray, p2: np.ndarray, 
                           p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Calculate dihedral angle between four points.
    
    Parameters:
    -----------
    p1, p2, p3, p4 : np.ndarray
        3D coordinates of the four points
        
    Returns:
    --------
    float
        Dihedral angle in degrees
    """
    # Calculate vectors
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3
    
    # Calculate normal vectors to planes
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    
    # Normalize
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    
    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0
    
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    
    # Calculate dihedral angle
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    # Determine sign using the scalar triple product
    if np.dot(np.cross(n1, n2), v2) < 0:
        angle = -angle
        
    return np.degrees(angle)


def get_backbone_atoms(molecule: Any, residue_idx: int) -> Dict[str, Optional[np.ndarray]]:
    """
    Extract backbone atoms (N, CA, C, O) from a specific residue.
    
    Parameters:
    -----------
    molecule : Molecule object
        The protein molecule
    residue_idx : int
        0-based residue index
        
    Returns:
    --------
    dict
        Dictionary with backbone atom positions (or None if not found)
    """
    backbone = {'N': None, 'CA': None, 'C': None, 'O': None}
    
    # Find atoms belonging to this residue
    for atom in molecule.atoms:
        if atom.residue_number == residue_idx + 1:  # Convert to 1-based
            if atom.atom_name in backbone:
                backbone[atom.atom_name] = np.array(atom.position)
    
    return backbone


def calculate_hydrogen_bond_energy(donor_pos: np.ndarray, hydrogen_pos: np.ndarray,
                                 acceptor_pos: np.ndarray, acceptor_antecedent_pos: np.ndarray) -> float:
    """
    Calculate hydrogen bond energy using a simple electrostatic model.
    
    This is a simplified version of the DSSP hydrogen bond energy calculation.
    
    Parameters:
    -----------
    donor_pos : np.ndarray
        Position of donor atom (typically N)
    hydrogen_pos : np.ndarray
        Position of hydrogen atom
    acceptor_pos : np.ndarray
        Position of acceptor atom (typically O)
    acceptor_antecedent_pos : np.ndarray
        Position of acceptor's antecedent atom (typically C)
        
    Returns:
    --------
    float
        Hydrogen bond energy in kcal/mol (negative values indicate favorable bonds)
    """
    # Convert to Angstroms for DSSP compatibility
    def nm_to_angstrom(pos):
        return pos * 10.0
    
    donor = nm_to_angstrom(donor_pos)
    hydrogen = nm_to_angstrom(hydrogen_pos)
    acceptor = nm_to_angstrom(acceptor_pos)
    antecedent = nm_to_angstrom(acceptor_antecedent_pos)
    
    # Calculate distances
    rho = np.linalg.norm(hydrogen - acceptor)
    rca = np.linalg.norm(acceptor - antecedent)
    rch = np.linalg.norm(hydrogen - antecedent)
    rna = np.linalg.norm(donor - acceptor)
    
    # DSSP energy formula (simplified)
    # E = q1*q2 * (1/rON + 1/rCH - 1/rOH - 1/rCN) * 332
    # Where q1 = 0.42, q2 = 0.20, and 332 is the electrostatic constant
    
    if rho < 0.5 or rca < 0.5 or rch < 0.5 or rna < 0.5:
        return 0.0  # Too close, probably not a real hydrogen bond
    
    q1q2 = 0.42 * 0.20
    f = 332.0
    
    energy = q1q2 * (1.0/rna + 1.0/rch - 1.0/rho - 1.0/rca) * f
    
    return energy


def identify_hydrogen_bonds(molecule: Any) -> List[Tuple[int, int, float]]:
    """
    Identify hydrogen bonds in the protein structure.
    
    Parameters:
    -----------
    molecule : Molecule object
        The protein molecule
        
    Returns:
    --------
    List[Tuple[int, int, float]]
        List of hydrogen bonds as (donor_residue, acceptor_residue, energy)
    """
    hydrogen_bonds = []
    
    # Get number of residues
    n_residues = len(set(atom.residue_number for atom in molecule.atoms))
    
    for i in range(n_residues):
        donor_atoms = get_backbone_atoms(molecule, i)
        
        if donor_atoms['N'] is None or donor_atoms['CA'] is None:
            continue
            
        # Look for hydrogen bonds with other residues (skip adjacent ones)
        for j in range(n_residues):
            if abs(i - j) < 2:  # Skip self and adjacent residues
                continue
                
            acceptor_atoms = get_backbone_atoms(molecule, j)
            
            if acceptor_atoms['C'] is None or acceptor_atoms['O'] is None:
                continue
            
            # Calculate hydrogen position (simplified - assume it's on the N-CA vector)
            n_to_ca = donor_atoms['CA'] - donor_atoms['N']
            n_to_ca_norm = n_to_ca / np.linalg.norm(n_to_ca)
            hydrogen_pos = donor_atoms['N'] - 0.1 * n_to_ca_norm  # 1 Angstrom back
            
            # Calculate distance
            h_to_o_dist = np.linalg.norm(hydrogen_pos - acceptor_atoms['O'])
            
            if h_to_o_dist > HB_DISTANCE_CUTOFF:
                continue
            
            # Calculate energy
            energy = calculate_hydrogen_bond_energy(
                donor_atoms['N'], hydrogen_pos, 
                acceptor_atoms['O'], acceptor_atoms['C']
            )
            
            if energy < HB_ENERGY_CUTOFF:  # Favorable hydrogen bond
                hydrogen_bonds.append((i, j, energy))
    
    return hydrogen_bonds


def assign_secondary_structure_dssp(molecule: Any) -> List[str]:
    """
    Assign secondary structure using a DSSP-like algorithm.
    
    This is a simplified version of the DSSP algorithm that focuses on
    the most common secondary structure elements.
    
    Parameters:
    -----------
    molecule : Molecule object
        The protein molecule
        
    Returns:
    --------
    List[str]
        Secondary structure assignment for each residue
    """
    # Get number of residues
    n_residues = len(set(atom.residue_number for atom in molecule.atoms))
    
    # Initialize with coil
    ss_assignment = ['C'] * n_residues
    
    # Identify hydrogen bonds
    hydrogen_bonds = identify_hydrogen_bonds(molecule)
    
    # Create hydrogen bond pattern matrix
    hb_matrix = np.zeros((n_residues, n_residues))
    for donor, acceptor, energy in hydrogen_bonds:
        hb_matrix[donor, acceptor] = 1
    
    # Identify alpha-helices (i to i+4 hydrogen bonds)
    for i in range(n_residues - 4):
        # Check for alpha-helix pattern (i->i+4)
        if hb_matrix[i+1, i] and hb_matrix[i+2, i+1]:  # Consecutive H-bonds
            # Mark as alpha-helix
            for j in range(i, min(i+4, n_residues)):
                if ss_assignment[j] == 'C':  # Don't override higher priority
                    ss_assignment[j] = 'H'
    
    # Identify 3-10 helices (i to i+3 hydrogen bonds)
    for i in range(n_residues - 3):
        if hb_matrix[i+1, i] and hb_matrix[i+2, i+1]:
            # Mark as 3-10 helix if not already alpha-helix
            for j in range(i, min(i+3, n_residues)):
                if ss_assignment[j] == 'C':
                    ss_assignment[j] = 'G'
    
    # Identify beta-strands (check phi/psi angles as well)
    phi_psi_angles = []
    for i in range(n_residues):
        prev_atoms = get_backbone_atoms(molecule, i-1) if i > 0 else None
        curr_atoms = get_backbone_atoms(molecule, i)
        next_atoms = get_backbone_atoms(molecule, i+1) if i < n_residues-1 else None
        
        phi = None
        if prev_atoms and curr_atoms:
            if (prev_atoms['C'] is not None and curr_atoms['N'] is not None and 
                curr_atoms['CA'] is not None and curr_atoms['C'] is not None):
                phi = calculate_dihedral_angle(
                    prev_atoms['C'], curr_atoms['N'], 
                    curr_atoms['CA'], curr_atoms['C']
                )
        
        psi = None
        if curr_atoms and next_atoms:
            if (curr_atoms['N'] is not None and curr_atoms['CA'] is not None and 
                curr_atoms['C'] is not None and next_atoms['N'] is not None):
                psi = calculate_dihedral_angle(
                    curr_atoms['N'], curr_atoms['CA'], 
                    curr_atoms['C'], next_atoms['N']
                )
        
        phi_psi_angles.append((phi, psi))
    
    # Assign beta-strands based on phi/psi angles
    for i, (phi, psi) in enumerate(phi_psi_angles):
        if phi is not None and psi is not None:
            # Beta-strand region: phi ~ -120°, psi ~ 120°
            if (-180 <= phi <= -90) and (90 <= psi <= 180):
                if ss_assignment[i] == 'C':  # Don't override helices
                    ss_assignment[i] = 'E'
    
    # Post-process to identify turns and refine assignments
    for i in range(1, n_residues - 1):
        if ss_assignment[i] == 'C':
            # Check if it's between different secondary structures (turn/bend)
            prev_ss = ss_assignment[i-1]
            next_ss = ss_assignment[i+1]
            
            if prev_ss != 'C' or next_ss != 'C':
                ss_assignment[i] = 'T'  # Turn
    
    return ss_assignment


class SecondaryStructureAnalyzer:
    """
    Class for analyzing secondary structure evolution in MD trajectories.
    
    This class provides comprehensive secondary structure analysis including:
    - DSSP-like structure assignment
    - Time evolution tracking
    - Statistical analysis
    - Visualization and export capabilities
    """
    
    def __init__(self, structure_cutoffs: Dict[str, float] = None):
        """
        Initialize the secondary structure analyzer.
        
        Parameters:
        -----------
        structure_cutoffs : dict, optional
            Custom cutoffs for structure assignment
        """
        self.structure_cutoffs = structure_cutoffs or {
            'hb_distance': HB_DISTANCE_CUTOFF,
            'hb_angle': HB_ANGLE_CUTOFF,
            'hb_energy': HB_ENERGY_CUTOFF
        }
        
        # Storage for analysis results
        self.trajectory_data = []
        self.residue_assignments = {}
        self.time_evolution = {}
        self.statistics = {}
        
        logger.info("SecondaryStructureAnalyzer initialized")
    
    def analyze_structure(self, molecule: Any, time_point: float = 0.0) -> Dict[str, Any]:
        """
        Analyze secondary structure for a single structure.
        
        Parameters:
        -----------
        molecule : Molecule object
            The protein molecule to analyze
        time_point : float, optional
            Time point for this structure (default: 0.0)
            
        Returns:
        --------
        dict
            Analysis results including assignments and statistics
        """
        logger.info(f"Analyzing secondary structure at time {time_point}")
        
        # Get secondary structure assignments
        ss_assignments = assign_secondary_structure_dssp(molecule)
        
        # Get residue information
        n_residues = len(ss_assignments)
        residue_names = []
        for i in range(n_residues):
            for atom in molecule.atoms:
                if atom.residue_number == i + 1:
                    residue_names.append(atom.residue_name)
                    break
            else:
                residue_names.append('UNK')
        
        # Calculate statistics
        ss_counts = {}
        for ss_type in SS_TYPES.keys():
            ss_counts[ss_type] = ss_assignments.count(ss_type)
        
        ss_percentages = {}
        for ss_type, count in ss_counts.items():
            ss_percentages[ss_type] = (count / n_residues) * 100.0 if n_residues > 0 else 0.0
        
        result = {
            'time_point': time_point,
            'n_residues': n_residues,
            'assignments': ss_assignments,
            'residue_names': residue_names,
            'counts': ss_counts,
            'percentages': ss_percentages,
            'hydrogen_bonds': identify_hydrogen_bonds(molecule)
        }
        
        return result
    
    def analyze_trajectory(self, simulation: Any, time_step: int = 10) -> Dict[str, Any]:
        """
        Analyze secondary structure evolution over a trajectory.
        
        Parameters:
        -----------
        simulation : Simulation object
            The MD simulation with trajectory data
        time_step : int, optional
            Analyze every nth frame (default: 10)
            
        Returns:
        --------
        dict
            Trajectory analysis results
        """
        logger.info(f"Analyzing secondary structure trajectory (every {time_step} frames)")
        
        self.trajectory_data = []
        trajectory = simulation.trajectory
        
        for i in range(0, len(trajectory), time_step):
            frame = trajectory[i]
            
            # Update molecule positions
            for j, atom in enumerate(simulation.molecule.atoms):
                if j < len(frame):
                    atom.position = frame[j].copy()
            
            # Analyze this frame
            time_point = i * time_step * simulation.dt if hasattr(simulation, 'dt') else i
            result = self.analyze_structure(simulation.molecule, time_point)
            self.trajectory_data.append(result)
        
        # Calculate trajectory statistics
        self._calculate_trajectory_statistics()
        
        logger.info(f"Trajectory analysis complete: {len(self.trajectory_data)} frames analyzed")
        return self.get_trajectory_summary()
    
    def _calculate_trajectory_statistics(self):
        """Calculate statistics over the entire trajectory."""
        if not self.trajectory_data:
            return
        
        n_frames = len(self.trajectory_data)
        n_residues = self.trajectory_data[0]['n_residues']
        
        # Time evolution of secondary structure percentages
        self.time_evolution = {
            'times': [frame['time_point'] for frame in self.trajectory_data],
            'percentages': {ss_type: [] for ss_type in SS_TYPES.keys()}
        }
        
        for frame in self.trajectory_data:
            for ss_type in SS_TYPES.keys():
                percentage = frame['percentages'].get(ss_type, 0.0)
                self.time_evolution['percentages'][ss_type].append(percentage)
        
        # Per-residue analysis
        self.residue_assignments = {}
        for res_idx in range(n_residues):
            assignments = []
            for frame in self.trajectory_data:
                if res_idx < len(frame['assignments']):
                    assignments.append(frame['assignments'][res_idx])
                else:
                    assignments.append('C')
            
            self.residue_assignments[res_idx] = {
                'assignments': assignments,
                'residue_name': self.trajectory_data[0]['residue_names'][res_idx],
                'percentages': {}
            }
            
            # Calculate percentages for this residue
            for ss_type in SS_TYPES.keys():
                count = assignments.count(ss_type)
                self.residue_assignments[res_idx]['percentages'][ss_type] = \
                    (count / n_frames) * 100.0 if n_frames > 0 else 0.0
        
        # Overall statistics
        self.statistics = {
            'n_frames': n_frames,
            'n_residues': n_residues,
            'avg_percentages': {},
            'std_percentages': {},
            'stability_scores': {}
        }
        
        for ss_type in SS_TYPES.keys():
            percentages = self.time_evolution['percentages'][ss_type]
            self.statistics['avg_percentages'][ss_type] = np.mean(percentages)
            self.statistics['std_percentages'][ss_type] = np.std(percentages)
            
            # Stability score: 1 - (std / (avg + 0.1))
            avg = self.statistics['avg_percentages'][ss_type]
            std = self.statistics['std_percentages'][ss_type]
            self.statistics['stability_scores'][ss_type] = 1.0 - (std / (avg + 0.1))
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get a summary of trajectory analysis results."""
        return {
            'trajectory_data': self.trajectory_data,
            'time_evolution': self.time_evolution,
            'residue_assignments': self.residue_assignments,
            'statistics': self.statistics
        }
    
    def plot_time_evolution(self, figsize: Tuple[int, int] = (14, 10),
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot secondary structure evolution over time.
        
        Parameters:
        -----------
        figsize : Tuple[int, int], optional
            Figure size (default: (14, 10))
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        if not self.time_evolution:
            raise ValueError("No trajectory data available. Run analyze_trajectory first.")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # Plot 1: Time evolution of secondary structure percentages
        times = self.time_evolution['times']
        
        # Plot major secondary structures
        major_types = ['H', 'E', 'T', 'C']
        for ss_type in major_types:
            percentages = self.time_evolution['percentages'][ss_type]
            color = SS_TYPES[ss_type]['color']
            label = SS_TYPES[ss_type]['name']
            ax1.plot(times, percentages, color=color, linewidth=2, label=label, alpha=0.8)
        
        ax1.set_xlabel('Time (ps)')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Secondary Structure Evolution Over Time', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Plot 2: Average percentages with error bars
        ss_names = [SS_TYPES[ss]['name'] for ss in major_types]
        avg_values = [self.statistics['avg_percentages'][ss] for ss in major_types]
        std_values = [self.statistics['std_percentages'][ss] for ss in major_types]
        colors = [SS_TYPES[ss]['color'] for ss in major_types]
        
        bars = ax2.bar(ss_names, avg_values, yerr=std_values, 
                      color=colors, alpha=0.7, capsize=5)
        
        ax2.set_ylabel('Average %')
        ax2.set_title('Average Secondary Structure Content', fontsize=12)
        ax2.set_ylim(0, max(avg_values) * 1.2 if avg_values else 100)
        
        # Add value labels on bars
        for bar, avg_val, std_val in zip(bars, avg_values, std_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std_val + 1,
                    f'{avg_val:.1f}±{std_val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Time evolution plot saved to {save_path}")
        
        return fig
    
    def plot_residue_timeline(self, residue_range: Optional[Tuple[int, int]] = None,
                            figsize: Tuple[int, int] = (16, 8),
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot secondary structure timeline for individual residues.
        
        Parameters:
        -----------
        residue_range : Tuple[int, int], optional
            Range of residues to plot (start, end). If None, plot all residues.
        figsize : Tuple[int, int], optional
            Figure size (default: (16, 8))
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        if not self.residue_assignments:
            raise ValueError("No trajectory data available. Run analyze_trajectory first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine residue range
        n_residues = len(self.residue_assignments)
        if residue_range is None:
            start_res, end_res = 0, n_residues
        else:
            start_res, end_res = residue_range
            start_res = max(0, start_res)
            end_res = min(n_residues, end_res)
        
        # Create color map
        ss_types_list = list(SS_TYPES.keys())
        colors = [SS_TYPES[ss]['color'] for ss in ss_types_list]
        cmap = ListedColormap(colors)
        
        # Create matrix for visualization
        n_frames = len(self.trajectory_data)
        display_residues = end_res - start_res
        
        ss_matrix = np.zeros((display_residues, n_frames))
        
        for i, res_idx in enumerate(range(start_res, end_res)):
            assignments = self.residue_assignments[res_idx]['assignments']
            for j, ss in enumerate(assignments):
                if ss in ss_types_list:
                    ss_matrix[i, j] = ss_types_list.index(ss)
        
        # Plot timeline
        im = ax.imshow(ss_matrix, cmap=cmap, aspect='auto', 
                      vmin=0, vmax=len(ss_types_list)-1,
                      extent=[0, n_frames, end_res, start_res])
        
        # Customize plot
        ax.set_xlabel('Frame')
        ax.set_ylabel('Residue Number')
        ax.set_title(f'Secondary Structure Timeline (Residues {start_res+1}-{end_res})', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_ticks(range(len(ss_types_list)))
        cbar.set_ticklabels([SS_TYPES[ss]['name'] for ss in ss_types_list])
        cbar.set_label('Secondary Structure', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residue timeline plot saved to {save_path}")
        
        return fig
    
    def plot_structure_distribution(self, figsize: Tuple[int, int] = (12, 8),
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of secondary structures per residue.
        
        Parameters:
        -----------
        figsize : Tuple[int, int], optional
            Figure size (default: (12, 8))
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        if not self.residue_assignments:
            raise ValueError("No trajectory data available. Run analyze_trajectory first.")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Helix content per residue
        residue_nums = list(range(1, len(self.residue_assignments) + 1))
        helix_percentages = [self.residue_assignments[i]['percentages']['H'] 
                           for i in range(len(self.residue_assignments))]
        
        ax1.bar(residue_nums, helix_percentages, color=SS_TYPES['H']['color'], alpha=0.7)
        ax1.set_title('α-Helix Content per Residue')
        ax1.set_xlabel('Residue Number')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_ylim(0, 100)
        
        # Plot 2: Beta-strand content per residue
        strand_percentages = [self.residue_assignments[i]['percentages']['E'] 
                            for i in range(len(self.residue_assignments))]
        
        ax2.bar(residue_nums, strand_percentages, color=SS_TYPES['E']['color'], alpha=0.7)
        ax2.set_title('β-Strand Content per Residue')
        ax2.set_xlabel('Residue Number')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_ylim(0, 100)
        
        # Plot 3: Turn content per residue
        turn_percentages = [self.residue_assignments[i]['percentages']['T'] 
                          for i in range(len(self.residue_assignments))]
        
        ax3.bar(residue_nums, turn_percentages, color=SS_TYPES['T']['color'], alpha=0.7)
        ax3.set_title('Turn Content per Residue')
        ax3.set_xlabel('Residue Number')
        ax3.set_ylabel('Percentage (%)')
        ax3.set_ylim(0, 100)
        
        # Plot 4: Coil content per residue
        coil_percentages = [self.residue_assignments[i]['percentages']['C'] 
                          for i in range(len(self.residue_assignments))]
        
        ax4.bar(residue_nums, coil_percentages, color=SS_TYPES['C']['color'], alpha=0.7)
        ax4.set_title('Coil Content per Residue')
        ax4.set_xlabel('Residue Number')
        ax4.set_ylabel('Percentage (%)')
        ax4.set_ylim(0, 100)
        
        plt.suptitle('Secondary Structure Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Structure distribution plot saved to {save_path}")
        
        return fig
    
    def export_timeline_data(self, filename: str, format: str = 'csv') -> None:
        """
        Export secondary structure timeline data.
        
        Parameters:
        -----------
        filename : str
            Output filename
        format : str, optional
            Export format ('csv' or 'json', default: 'csv')
        """
        if not self.trajectory_data:
            raise ValueError("No trajectory data available. Run analyze_trajectory first.")
        
        if format.lower() == 'csv':
            self._export_csv(filename)
        elif format.lower() == 'json':
            self._export_json(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Timeline data exported to {filename}")
    
    def _export_csv(self, filename: str) -> None:
        """Export data in CSV format."""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['Time', 'Frame', 'Residue', 'ResName', 'SecondaryStructure']
            writer.writerow(header)
            
            # Write data
            for frame_idx, frame_data in enumerate(self.trajectory_data):
                time_point = frame_data['time_point']
                assignments = frame_data['assignments']
                residue_names = frame_data['residue_names']
                
                for res_idx, (ss, res_name) in enumerate(zip(assignments, residue_names)):
                    writer.writerow([time_point, frame_idx, res_idx + 1, res_name, ss])
    
    def _export_json(self, filename: str) -> None:
        """Export data in JSON format."""
        export_data = {
            'metadata': {
                'n_frames': len(self.trajectory_data),
                'n_residues': self.trajectory_data[0]['n_residues'] if self.trajectory_data else 0,
                'ss_types': SS_TYPES
            },
            'trajectory_data': self.trajectory_data,
            'statistics': self.statistics,
            'time_evolution': self.time_evolution
        }
        
        with open(filename, 'w') as jsonfile:
            json.dump(export_data, jsonfile, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about secondary structure analysis.
        
        Returns:
        --------
        dict
            Dictionary with various statistics
        """
        return {
            'trajectory_stats': self.statistics,
            'residue_stats': self.residue_assignments,
            'structure_counts': {
                frame['time_point']: frame['counts'] 
                for frame in self.trajectory_data
            }
        }


def create_secondary_structure_analyzer(**kwargs) -> SecondaryStructureAnalyzer:
    """
    Create a secondary structure analyzer with optional custom parameters.
    
    Parameters
    ----------
    **kwargs
        Optional parameters for the analyzer
        
    Returns
    -------
    SecondaryStructureAnalyzer
        Configured analyzer instance
    """
    return SecondaryStructureAnalyzer(**kwargs)


if __name__ == "__main__":
    # Example usage and testing
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    
    print("Testing Secondary Structure Analysis...")
    
    # Create a mock molecule for testing
    class MockAtom:
        def __init__(self, name, res_num, res_name, position):
            self.atom_name = name
            self.residue_number = res_num
            self.residue_name = res_name
            self.position = np.array(position)
    
    class MockMolecule:
        def __init__(self):
            self.atoms = []
    
    # Create test protein structure (small helix)
    molecule = MockMolecule()
    n_residues = 10
    
    for i in range(n_residues):
        res_num = i + 1
        res_name = 'ALA'
        
        # Approximate alpha-helix geometry
        phi_angle = -60.0  # degrees
        psi_angle = -45.0
        
        # Calculate positions along helix
        z = i * 1.5  # 1.5 Å rise per residue
        theta = i * 100.0 * np.pi / 180.0  # 100° rotation per residue
        radius = 2.3  # helix radius
        
        x_offset = radius * np.cos(theta)
        y_offset = radius * np.sin(theta)
        
        # Backbone atoms
        n_pos = [x_offset - 0.5, y_offset, z]
        ca_pos = [x_offset, y_offset, z]
        c_pos = [x_offset + 0.5, y_offset + 0.3, z + 0.2]
        o_pos = [x_offset + 0.8, y_offset + 0.5, z + 0.1]
        
        molecule.atoms.extend([
            MockAtom('N', res_num, res_name, n_pos),
            MockAtom('CA', res_num, res_name, ca_pos),
            MockAtom('C', res_num, res_name, c_pos),
            MockAtom('O', res_num, res_name, o_pos)
        ])
    
    # Test secondary structure analysis
    analyzer = create_secondary_structure_analyzer()
    result = analyzer.analyze_structure(molecule, time_point=0.0)
    
    print(f"Test Results:")
    print(f"- Number of residues: {result['n_residues']}")
    print(f"- Secondary structure assignments: {result['assignments']}")
    print(f"- Structure percentages:")
    for ss_type, percentage in result['percentages'].items():
        if percentage > 0:
            print(f"  {SS_TYPES[ss_type]['name']}: {percentage:.1f}%")
    
    print("Secondary structure analysis test completed successfully!")
