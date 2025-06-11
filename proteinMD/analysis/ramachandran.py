"""
Ramachandran Plot Analysis Module

This module provides functionality to calculate and analyze Ramachandran plots
for protein structures, including phi-psi angle distributions and conformational
analysis over molecular dynamics trajectories.

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard amino acid names and properties
AMINO_ACIDS = {
    'ALA': {'name': 'Alanine', 'code': 'A', 'color': '#FF6B6B'},
    'ARG': {'name': 'Arginine', 'code': 'R', 'color': '#4ECDC4'},
    'ASN': {'name': 'Asparagine', 'code': 'N', 'color': '#45B7D1'},
    'ASP': {'name': 'Aspartic acid', 'code': 'D', 'color': '#96CEB4'},
    'CYS': {'name': 'Cysteine', 'code': 'C', 'color': '#FFEAA7'},
    'GLN': {'name': 'Glutamine', 'code': 'Q', 'color': '#DDA0DD'},
    'GLU': {'name': 'Glutamic acid', 'code': 'E', 'color': '#98D8C8'},
    'GLY': {'name': 'Glycine', 'code': 'G', 'color': '#F7DC6F'},
    'HIS': {'name': 'Histidine', 'code': 'H', 'color': '#BB8FCE'},
    'ILE': {'name': 'Isoleucine', 'code': 'I', 'color': '#85C1E9'},
    'LEU': {'name': 'Leucine', 'code': 'L', 'color': '#F8C471'},
    'LYS': {'name': 'Lysine', 'code': 'K', 'color': '#82E0AA'},
    'MET': {'name': 'Methionine', 'code': 'M', 'color': '#D7BDE2'},
    'PHE': {'name': 'Phenylalanine', 'code': 'F', 'color': '#A3E4D7'},
    'PRO': {'name': 'Proline', 'code': 'P', 'color': '#FAD7A0'},
    'SER': {'name': 'Serine', 'code': 'S', 'color': '#AED6F1'},
    'THR': {'name': 'Threonine', 'code': 'T', 'color': '#A9DFBF'},
    'TRP': {'name': 'Tryptophan', 'code': 'W', 'color': '#D5A6BD'},
    'TYR': {'name': 'Tyrosine', 'code': 'Y', 'color': '#F9E79F'},
    'VAL': {'name': 'Valine', 'code': 'V', 'color': '#FADBD8'}
}

def calculate_dihedral_angle(p1: np.ndarray, p2: np.ndarray, 
                           p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Calculate the dihedral angle between four points.
    
    Parameters:
    -----------
    p1, p2, p3, p4 : np.ndarray
        3D coordinates of the four points defining the dihedral angle
        
    Returns:
    --------
    float
        Dihedral angle in degrees (-180 to 180)
    """
    # Calculate vectors
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3
    
    # Calculate normal vectors to the planes
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    
    # Normalize normal vectors
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    
    if n1_norm == 0 or n2_norm == 0:
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
    Extract backbone atom coordinates for a specific residue.
    
    Parameters:
    -----------
    molecule : Molecule object
        The molecule containing the protein structure
    residue_idx : int
        Index of the residue
        
    Returns:
    --------
    dict
        Dictionary containing coordinates of backbone atoms (N, CA, C)
    """
    atoms = {'N': None, 'CA': None, 'C': None}
    
    # Find backbone atoms for this residue
    for atom in molecule.atoms:
        if atom.residue_number == residue_idx + 1:  # 1-based indexing
            if atom.name in ['N', 'CA', 'C']:
                atoms[atom.name] = atom.position.copy()
                
    return atoms

def calculate_phi_psi_angles(molecule: Any) -> Tuple[List[float], List[float], List[str]]:
    """
    Calculate phi and psi angles for all residues in a protein.
    
    Parameters:
    -----------
    molecule : Molecule object
        The protein molecule
        
    Returns:
    --------
    tuple
        (phi_angles, psi_angles, residue_names) lists
    """
    phi_angles = []
    psi_angles = []
    residue_names = []
    
    # Get number of residues
    n_residues = len(set(atom.residue_number for atom in molecule.atoms))
    
    for i in range(n_residues):
        # Get current residue name
        residue_name = None
        for atom in molecule.atoms:
            if atom.residue_number == i + 1:
                residue_name = atom.residue_name
                break
                
        if residue_name is None:
            continue
            
        # Get backbone atoms for current and adjacent residues
        prev_atoms = get_backbone_atoms(molecule, i - 1) if i > 0 else None
        curr_atoms = get_backbone_atoms(molecule, i)
        next_atoms = get_backbone_atoms(molecule, i + 1) if i < n_residues - 1 else None
        
        # Calculate phi angle (C-1, N, CA, C)
        phi = None
        if prev_atoms and curr_atoms:
            if (prev_atoms['C'] is not None and 
                curr_atoms['N'] is not None and 
                curr_atoms['CA'] is not None and 
                curr_atoms['C'] is not None):
                phi = calculate_dihedral_angle(
                    prev_atoms['C'], curr_atoms['N'], 
                    curr_atoms['CA'], curr_atoms['C']
                )
        
        # Calculate psi angle (N, CA, C, N+1)
        psi = None
        if curr_atoms and next_atoms:
            if (curr_atoms['N'] is not None and 
                curr_atoms['CA'] is not None and 
                curr_atoms['C'] is not None and 
                next_atoms['N'] is not None):
                psi = calculate_dihedral_angle(
                    curr_atoms['N'], curr_atoms['CA'], 
                    curr_atoms['C'], next_atoms['N']
                )
        
        # Only add if both angles are valid
        if phi is not None and psi is not None:
            phi_angles.append(phi)
            psi_angles.append(psi)
            residue_names.append(residue_name)
    
    return phi_angles, psi_angles, residue_names

class RamachandranAnalyzer:
    """
    Class for analyzing Ramachandran plots and phi-psi angle distributions.
    """
    
    def __init__(self):
        """Initialize the Ramachandran analyzer."""
        self.phi_angles = []
        self.psi_angles = []
        self.residue_names = []
        self.time_points = []
        self.trajectory_data = []
        
    def analyze_structure(self, molecule: Any, time_point: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze a single protein structure for phi-psi angles.
        
        Parameters:
        -----------
        molecule : Molecule object
            The protein structure to analyze
        time_point : float, optional
            Time point for trajectory analysis
            
        Returns:
        --------
        dict
            Analysis results including angles and statistics
        """
        phi_angles, psi_angles, residue_names = calculate_phi_psi_angles(molecule)
        
        # Store data
        if time_point is not None:
            self.trajectory_data.append({
                'time': time_point,
                'phi': phi_angles.copy(),
                'psi': psi_angles.copy(),
                'residues': residue_names.copy()
            })
        else:
            self.phi_angles = phi_angles
            self.psi_angles = psi_angles
            self.residue_names = residue_names
        
        # Calculate statistics
        stats = {
            'n_residues': len(phi_angles),
            'phi_mean': np.mean(phi_angles) if phi_angles else 0,
            'phi_std': np.std(phi_angles) if phi_angles else 0,
            'psi_mean': np.mean(psi_angles) if psi_angles else 0,
            'psi_std': np.std(psi_angles) if psi_angles else 0,
            'phi_range': (np.min(phi_angles), np.max(phi_angles)) if phi_angles else (0, 0),
            'psi_range': (np.min(psi_angles), np.max(psi_angles)) if psi_angles else (0, 0)
        }
        
        return {
            'phi_angles': phi_angles,
            'psi_angles': psi_angles,
            'residue_names': residue_names,
            'statistics': stats
        }
    
    def analyze_trajectory(self, simulation: Any, time_step: int = 10) -> Dict[str, Any]:
        """
        Analyze Ramachandran angles over a molecular dynamics trajectory.
        
        Parameters:
        -----------
        simulation : MolecularDynamicsSimulation
            The simulation object containing trajectory data
        time_step : int
            Analyze every nth frame to reduce computation
            
        Returns:
        --------
        dict
            Trajectory analysis results
        """
        if not hasattr(simulation, 'trajectory') or not simulation.trajectory:
            raise ValueError("No trajectory data available in simulation")
        
        # Check if trajectory has len method, otherwise check if it's empty
        try:
            n_frames = len(simulation.trajectory)
        except TypeError:
            # Handle case where trajectory is a Mock or doesn't support len()
            if hasattr(simulation.trajectory, '__iter__'):
                try:
                    n_frames = sum(1 for _ in simulation.trajectory)
                except:
                    raise ValueError("No trajectory data available in simulation")
            else:
                raise ValueError("No trajectory data available in simulation")
        
        logger.info(f"Analyzing Ramachandran angles for {n_frames} frames")
        
        self.trajectory_data = []
        
        for i, frame in enumerate(simulation.trajectory[::time_step]):
            # Update molecule coordinates
            for j, atom in enumerate(simulation.molecule.atoms):
                if j < len(frame):
                    atom.position = frame[j].copy()
            
            # Analyze this frame
            time_point = i * time_step * simulation.dt if hasattr(simulation, 'dt') else i
            self.analyze_structure(simulation.molecule, time_point)
        
        return self._calculate_trajectory_statistics()
    
    def _calculate_trajectory_statistics(self) -> Dict[str, Any]:
        """Calculate statistics over the entire trajectory."""
        if not self.trajectory_data:
            return {}
        
        all_phi = []
        all_psi = []
        
        for frame in self.trajectory_data:
            all_phi.extend(frame['phi'])
            all_psi.extend(frame['psi'])
        
        return {
            'n_frames': len(self.trajectory_data),
            'total_angles': len(all_phi),
            'phi_distribution': {
                'mean': np.mean(all_phi) if all_phi else 0,
                'std': np.std(all_phi) if all_phi else 0,
                'min': np.min(all_phi) if all_phi else 0,
                'max': np.max(all_phi) if all_phi else 0
            },
            'psi_distribution': {
                'mean': np.mean(all_psi) if all_psi else 0,
                'std': np.std(all_psi) if all_psi else 0,
                'min': np.min(all_psi) if all_psi else 0,
                'max': np.max(all_psi) if all_psi else 0
            }
        }
    
    def plot_ramachandran(self, 
                         color_by_residue: bool = True,
                         show_regions: bool = True,
                         figsize: Tuple[int, int] = (10, 8),
                         title: str = "Ramachandran Plot") -> plt.Figure:
        """
        Create a Ramachandran plot from the analyzed data.
        
        Parameters:
        -----------
        color_by_residue : bool
            Whether to color points by amino acid type
        show_regions : bool
            Whether to show allowed/favored regions
        figsize : tuple
            Figure size (width, height)
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use current data or trajectory average
        if self.trajectory_data:
            phi_data, psi_data, residue_data = self._get_trajectory_average()
        else:
            phi_data = self.phi_angles
            psi_data = self.psi_angles
            residue_data = self.residue_names
        
        if not phi_data or not psi_data:
            ax.text(0.5, 0.5, 'No data to plot', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Show allowed regions if requested
        if show_regions:
            self._add_ramachandran_regions(ax)
        
        # Plot points
        if color_by_residue and residue_data:
            unique_residues = list(set(residue_data))
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_residues)))
            
            for i, residue in enumerate(unique_residues):
                mask = [res == residue for res in residue_data]
                phi_res = [phi_data[j] for j in range(len(mask)) if mask[j]]
                psi_res = [psi_data[j] for j in range(len(mask)) if mask[j]]
                
                color = AMINO_ACIDS.get(residue, {}).get('color', colors[i])
                ax.scatter(phi_res, psi_res, c=color, label=residue, 
                          alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
        else:
            ax.scatter(phi_data, psi_data, alpha=0.6, s=30, 
                      edgecolors='black', linewidth=0.5)
        
        # Formatting
        ax.set_xlabel('Phi (degrees)', fontsize=12)
        ax.set_ylabel('Psi (degrees)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
        
        # Add legend if coloring by residue
        if color_by_residue and residue_data:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def _add_ramachandran_regions(self, ax: plt.Axes) -> None:
        """Add favored and allowed regions to Ramachandran plot."""
        # Alpha-helix region (approximate)
        alpha_helix = patches.Ellipse((-60, -45), 60, 60, 
                                     alpha=0.2, color='blue', label='α-helix')
        ax.add_patch(alpha_helix)
        
        # Beta-sheet region (approximate)
        beta_sheet = patches.Rectangle((-150, 100), 80, 60, 
                                      alpha=0.2, color='red', label='β-sheet')
        ax.add_patch(beta_sheet)
        
        # Left-handed alpha-helix (rare)
        left_alpha = patches.Ellipse((60, 45), 40, 40, 
                                    alpha=0.1, color='green', label='Left α-helix')
        ax.add_patch(left_alpha)
    
    def _get_trajectory_average(self) -> Tuple[List[float], List[float], List[str]]:
        """Get average phi-psi angles from trajectory data."""
        if not self.trajectory_data:
            return [], [], []
        
        # Use the last frame for now (could be averaged or all frames)
        last_frame = self.trajectory_data[-1]
        return last_frame['phi'], last_frame['psi'], last_frame['residues']
    
    def plot_angle_evolution(self, residue_index: int = 0, 
                           figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Plot the evolution of phi and psi angles over time for a specific residue.
        
        Parameters:
        -----------
        residue_index : int
            Index of the residue to plot
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        if not self.trajectory_data:
            raise ValueError("No trajectory data available")
        
        times = []
        phi_evolution = []
        psi_evolution = []
        
        for frame in self.trajectory_data:
            times.append(frame['time'])
            if residue_index < len(frame['phi']):
                phi_evolution.append(frame['phi'][residue_index])
                psi_evolution.append(frame['psi'][residue_index])
            else:
                phi_evolution.append(np.nan)
                psi_evolution.append(np.nan)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Phi angle evolution
        ax1.plot(times, phi_evolution, 'b-', linewidth=1, label='Phi')
        ax1.set_ylabel('Phi angle (degrees)')
        ax1.set_ylim(-180, 180)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
        
        # Psi angle evolution
        ax2.plot(times, psi_evolution, 'r-', linewidth=1, label='Psi')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Psi angle (degrees)')
        ax2.set_ylim(-180, 180)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
        
        residue_name = (self.trajectory_data[0]['residues'][residue_index] 
                       if residue_index < len(self.trajectory_data[0]['residues']) 
                       else f"Residue {residue_index}")
        
        fig.suptitle(f'Phi-Psi Angle Evolution - {residue_name}', fontsize=14)
        plt.tight_layout()
        return fig
    
    def export_data(self, filepath: str, format: str = 'csv') -> None:
        """
        Export Ramachandran data to file.
        
        Parameters:
        -----------
        filepath : str
            Output file path
        format : str
            Export format ('csv', 'json')
        """
        filepath = Path(filepath)
        
        if format.lower() == 'csv':
            # Try to use pandas for CSV export
            try:
                import pandas as pd
                
                if self.trajectory_data:
                    # Export trajectory data
                    data = []
                    for frame in self.trajectory_data:
                        for i, (phi, psi, res) in enumerate(zip(frame['phi'], frame['psi'], frame['residues'])):
                            data.append({
                                'time': frame['time'],
                                'residue_index': i,
                                'residue_name': res,
                                'phi': phi,
                                'psi': psi
                            })
                    df = pd.DataFrame(data)
                else:
                    # Export single structure data
                    data = {
                        'residue_index': list(range(len(self.phi_angles))),
                        'residue_name': self.residue_names,
                        'phi': self.phi_angles,
                        'psi': self.psi_angles
                    }
                    df = pd.DataFrame(data)
                
                df.to_csv(filepath, index=False)
                
            except ImportError:
                # Fallback to manual CSV writing
                with open(filepath, 'w') as f:
                    if self.trajectory_data:
                        f.write('time,residue_index,residue_name,phi,psi\n')
                        for frame in self.trajectory_data:
                            for i, (phi, psi, res) in enumerate(zip(frame['phi'], frame['psi'], frame['residues'])):
                                f.write(f"{frame['time']},{i},{res},{phi},{psi}\n")
                    else:
                        f.write('residue_index,residue_name,phi,psi\n')
                        for i, (phi, psi, res) in enumerate(zip(self.phi_angles, self.psi_angles, self.residue_names)):
                            f.write(f"{i},{res},{phi},{psi}\n")
        
        elif format.lower() == 'json':
            if self.trajectory_data:
                data = {
                    'type': 'trajectory',
                    'frames': self.trajectory_data
                }
            else:
                data = {
                    'type': 'single_structure',
                    'phi_angles': self.phi_angles,
                    'psi_angles': self.psi_angles,
                    'residue_names': self.residue_names
                }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Ramachandran data exported to {filepath}")

def create_ramachandran_analyzer(molecule: Any = None) -> RamachandranAnalyzer:
    """
    Create and initialize a Ramachandran analyzer.
    
    Parameters:
    -----------
    molecule : Molecule, optional
        Initial molecule to analyze
        
    Returns:
    --------
    RamachandranAnalyzer
        Initialized analyzer object
    """
    analyzer = RamachandranAnalyzer()
    
    if molecule is not None:
        analyzer.analyze_structure(molecule)
        logger.info(f"Ramachandran analyzer created and analyzed {len(analyzer.phi_angles)} residues")
    else:
        logger.info("Ramachandran analyzer created (no initial structure)")
    
    return analyzer
