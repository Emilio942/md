"""
Analysis module for molecular dynamics simulations.

This module provides classes and functions for analyzing the results
of MD simulations, including structural and dynamic properties.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)

# Import hydrogen bond analysis
try:
    from .hydrogen_bonds import (
        HydrogenBondDetector, HydrogenBondAnalyzer, HydrogenBond,
        analyze_hydrogen_bonds, quick_hydrogen_bond_summary
    )
    HAS_HYDROGEN_BONDS = True
except ImportError:
    HAS_HYDROGEN_BONDS = False
    logger.warning("Hydrogen bond analysis not available")

# Import RMSD analysis
try:
    from .rmsd import RMSDAnalyzer, calculate_rmsd, align_structures, create_rmsd_analyzer
    HAS_RMSD = True
except ImportError:
    HAS_RMSD = False
    logger.warning("RMSD analysis not available")

# Import Radius of Gyration analysis
try:
    from .radius_of_gyration import (RadiusOfGyrationAnalyzer, calculate_radius_of_gyration, 
                                   calculate_center_of_mass, create_rg_analyzer)
    HAS_RG = True
except ImportError:
    HAS_RG = False
    logger.warning("Radius of Gyration analysis not available")

# Import hydrogen bond analysis
try:
    from .hydrogen_bonds import (
        HydrogenBondDetector, HydrogenBondAnalyzer, HydrogenBond,
        analyze_hydrogen_bonds, quick_hydrogen_bond_summary
    )
    HAS_HYDROGEN_BONDS = True
except ImportError:
    HAS_HYDROGEN_BONDS = False
    logger.warning("Hydrogen bond analysis not available")

# Import Radius of Gyration analysis
try:
    from .radius_of_gyration import (RadiusOfGyrationAnalyzer, calculate_radius_of_gyration, 
                                   calculate_center_of_mass, create_rg_analyzer)
    HAS_RG = True
except ImportError:
    HAS_RG = False
    logger.warning("Radius of Gyration analysis not available")

class Analysis:
    """
    Base class for analysis methods.
    
    This provides common functionality for analyzing MD trajectories.
    """
    
    def __init__(self, name: str):
        """
        Initialize an analysis method.
        
        Parameters
        ----------
        name : str
            Name of the analysis method
        """
        self.name = name
    
    def analyze(self, trajectory, selection=None):
        """
        Analyze a trajectory.
        
        Parameters
        ----------
        trajectory : Trajectory
            The trajectory to analyze
        selection : Selection, optional
            A subset of atoms to analyze
        
        Returns
        -------
        dict
            Analysis results
        """
        raise NotImplementedError("This method should be implemented by derived classes")


class RMSD(Analysis):
    """
    Root Mean Square Deviation (RMSD) analysis.
    
    RMSD measures the structural deviation from a reference structure.
    """
    
    def __init__(self, reference=None):
        """
        Initialize RMSD analysis.
        
        Parameters
        ----------
        reference : ndarray, optional
            Reference structure to compare against (default: first frame)
        """
        super().__init__("RMSD")
        self.reference = reference
    
    def analyze(self, trajectory, selection=None):
        """
        Calculate RMSD for each frame in a trajectory.
        
        Parameters
        ----------
        trajectory : Trajectory
            The trajectory to analyze
        selection : Selection, optional
            A subset of atoms to include (default: all atoms)
        
        Returns
        -------
        dict
            Dictionary with RMSD values for each frame
        """
        # Get coordinates
        if selection is None:
            coords = trajectory.positions
        else:
            coords = trajectory.positions[:, selection, :]
        
        num_frames = len(coords)
        
        # Use first frame as reference if not provided
        if self.reference is None:
            self.reference = coords[0]
        
        # Initialize results
        rmsd_values = np.zeros(num_frames)
        
        # Calculate RMSD for each frame
        for i in range(num_frames):
            # Calculate RMSD without fitting
            diff = coords[i] - self.reference
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            rmsd_values[i] = rmsd
        
        return {
            "rmsd": rmsd_values,
            "frames": np.arange(num_frames),
            "time": trajectory.time
        }


class RMSF(Analysis):
    """
    Root Mean Square Fluctuation (RMSF) analysis.
    
    RMSF measures the structural flexibility of atoms over time.
    """
    
    def __init__(self):
        """Initialize RMSF analysis."""
        super().__init__("RMSF")
    
    def analyze(self, trajectory, selection=None):
        """
        Calculate RMSF for each atom in a trajectory.
        
        Parameters
        ----------
        trajectory : Trajectory
            The trajectory to analyze
        selection : Selection, optional
            A subset of atoms to include (default: all atoms)
        
        Returns
        -------
        dict
            Dictionary with RMSF values for each atom
        """
        # Get coordinates
        if selection is None:
            coords = trajectory.positions
            atom_ids = np.arange(coords.shape[1])
        else:
            coords = trajectory.positions[:, selection, :]
            atom_ids = selection
        
        # Calculate average positions
        avg_coords = np.mean(coords, axis=0)
        
        # Calculate RMSF for each atom
        rmsf_values = np.zeros(len(atom_ids))
        
        for i, atom_idx in enumerate(atom_ids):
            # Calculate displacements from average
            disp = coords[:, i, :] - avg_coords[i]
            
            # Calculate RMSF
            rmsf = np.sqrt(np.mean(np.sum(disp**2, axis=1)))
            rmsf_values[i] = rmsf
        
        return {
            "rmsf": rmsf_values,
            "atom_ids": atom_ids
        }


class RadialDistributionFunction(Analysis):
    """
    Radial Distribution Function (RDF) analysis.
    
    RDF measures the probability of finding a particle at a given distance
    from a reference particle.
    """
    
    def __init__(self, max_distance: float = 2.0, n_bins: int = 100):
        """
        Initialize RDF analysis.
        
        Parameters
        ----------
        max_distance : float
            Maximum distance to consider in nanometers
        n_bins : int
            Number of distance bins
        """
        super().__init__("RDF")
        self.max_distance = max_distance
        self.n_bins = n_bins
    
    def analyze(self, trajectory, selection1, selection2=None):
        """
        Calculate RDF between two selections of atoms.
        
        Parameters
        ----------
        trajectory : Trajectory
            The trajectory to analyze
        selection1 : ndarray
            First selection of atom indices
        selection2 : ndarray, optional
            Second selection of atom indices (default: same as selection1)
        
        Returns
        -------
        dict
            Dictionary with RDF data
        """
        # Get coordinates
        coords = trajectory.positions
        
        # Use the same selection for both if selection2 not provided
        if selection2 is None:
            selection2 = selection1
        
        # Get box sizes
        box_sizes = trajectory.box_sizes
        
        # Initialize histogram
        bins = np.linspace(0, self.max_distance, self.n_bins + 1)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        hist = np.zeros(self.n_bins)
        
        # Calculate RDF for each frame
        num_frames = len(coords)
        
        for frame in range(num_frames):
            # Extract coordinates and box size for current frame
            frame_coords1 = coords[frame, selection1, :]
            frame_coords2 = coords[frame, selection2, :]
            box_size = box_sizes[frame]
            
            # Calculate distances considering periodic boundaries
            for i, pos1 in enumerate(frame_coords1):
                for j, pos2 in enumerate(frame_coords2):
                    # Skip self-interactions
                    if np.array_equal(selection1, selection2) and i == j:
                        continue
                    
                    # Calculate distance with minimum image convention
                    dr = pos2 - pos1
                    dr = dr - np.round(dr / box_size) * box_size
                    dist = np.linalg.norm(dr)
                    
                    # Add to histogram if within range
                    if dist < self.max_distance:
                        bin_idx = int(dist / self.max_distance * self.n_bins)
                        if bin_idx < self.n_bins:
                            hist[bin_idx] += 1
        
        # Normalize histogram
        # Volume of each spherical shell
        bin_width = self.max_distance / self.n_bins
        shell_volumes = 4 * np.pi * bin_centers**2 * bin_width
        
        # Number of reference particles and number of target particles
        n_ref = len(selection1)
        n_target = len(selection2)
        
        # Average number density of target particles
        if selection1 is selection2:
            # Factor of 2 to account for double-counting
            n_pairs = n_ref * (n_ref - 1) / 2
        else:
            n_pairs = n_ref * n_target
        
        # Average box volume
        avg_box_volume = np.mean(np.prod(box_sizes, axis=1))
        
        # Average number density of target particles
        rho = n_target / avg_box_volume
        
        # Normalize RDF
        norm_factor = n_pairs * num_frames * shell_volumes * rho
        rdf = hist / norm_factor
        
        return {
            "r": bin_centers,
            "rdf": rdf
        }


class HydrogenBondAnalysis(Analysis):
    """
    Hydrogen Bond Analysis.
    
    This analysis identifies and counts hydrogen bonds in a protein structure.
    """
    
    def __init__(self, 
                 distance_cutoff: float = 0.35,  # nm
                 angle_cutoff: float = 30.0):    # degrees
        """
        Initialize Hydrogen Bond Analysis.
        
        Parameters
        ----------
        distance_cutoff : float
            Maximum hydrogen bond distance in nanometers
        angle_cutoff : float
            Maximum hydrogen bond angle deviation in degrees
        """
        super().__init__("Hydrogen Bond Analysis")
        self.distance_cutoff = distance_cutoff
        self.angle_cutoff = angle_cutoff
    
    def analyze(self, trajectory, selection=None):
        """
        Analyze hydrogen bonds in a trajectory.
        
        Parameters
        ----------
        trajectory : Trajectory
            The trajectory to analyze
        selection : Selection, optional
            A subset of atoms to include (default: all atoms)
        
        Returns
        -------
        dict
            Dictionary with hydrogen bond data
        """
        # In a real implementation, would identify hydrogen bond
        # donors and acceptors based on atom types and connectivity
        
        # Simplified hydrogen bond analysis
        
        # Placeholder result for demonstration
        result = {
            "num_hbonds": np.random.randint(10, 50, len(trajectory.positions)),
            "frames": np.arange(len(trajectory.positions)),
            "time": trajectory.time
        }
        
        return result


class SecondaryStructureAnalysis(Analysis):
    """
    Secondary Structure Analysis.
    
    This analysis identifies and quantifies secondary structure elements
    in a protein structure over time.
    """
    
    def __init__(self):
        """Initialize Secondary Structure Analysis."""
        super().__init__("Secondary Structure Analysis")
    
    def analyze(self, trajectory, selection=None):
        """
        Analyze secondary structure in a trajectory.
        
        Parameters
        ----------
        trajectory : Trajectory
            The trajectory to analyze
        selection : Selection, optional
            A subset of atoms to include (default: all atoms)
        
        Returns
        -------
        dict
            Dictionary with secondary structure data
        """
        # In a real implementation, would use the DSSP algorithm or similar
        # to identify secondary structure elements
        
        # Simplified secondary structure analysis
        
        # Placeholder result for demonstration
        num_frames = len(trajectory.positions)
        
        # Example secondary structure assignment
        # For each residue, assign a secondary structure type:
        # H: alpha-helix, E: beta-strand, C: coil, T: turn, G: 3-10 helix
        # B: beta-bridge, I: pi-helix, S: bend
        ss_types = ["H", "E", "C", "T", "G", "B", "I", "S"]
        
        # Assume 100 residues for this example
        num_residues = 100
        
        # Generate random secondary structure assignments
        ss_matrix = np.random.choice(ss_types, size=(num_frames, num_residues))
        
        # Count occurrences of each secondary structure type for each frame
        ss_counts = np.zeros((num_frames, len(ss_types)))
        
        for i, ss_type in enumerate(ss_types):
            ss_counts[:, i] = np.sum(ss_matrix == ss_type, axis=1)
        
        # Calculate percentage of each secondary structure type
        ss_percent = 100.0 * ss_counts / num_residues
        
        return {
            "frames": np.arange(num_frames),
            "time": trajectory.time,
            "ss_types": ss_types,
            "ss_matrix": ss_matrix,
            "ss_counts": ss_counts,
            "ss_percent": ss_percent
        }


class Trajectory:
    """
    Class for handling simulation trajectories.
    
    This class provides methods for loading, manipulating, and
    analyzing trajectory data.
    """
    
    def __init__(self, system=None, filename=None):
        """
        Initialize a trajectory.
        
        Parameters
        ----------
        system : MDSystem, optional
            The molecular dynamics system
        filename : str, optional
            Name of a trajectory file to load
        """
        self.system = system
        self.positions = []
        self.velocities = []
        self.forces = []
        self.box_sizes = []
        self.time = []
        
        if filename is not None:
            self.load(filename)
    
    def add_frame(self, positions, velocities=None, forces=None, box_size=None, time=None):
        """
        Add a frame to the trajectory.
        
        Parameters
        ----------
        positions : ndarray
            Atom positions for this frame
        velocities : ndarray, optional
            Atom velocities for this frame
        forces : ndarray, optional
            Forces on atoms for this frame
        box_size : ndarray, optional
            Simulation box size for this frame
        time : float, optional
            Time of this frame in picoseconds
        """
        self.positions.append(positions.copy())
        
        if velocities is not None:
            self.velocities.append(velocities.copy())
        
        if forces is not None:
            self.forces.append(forces.copy())
        
        if box_size is not None:
            self.box_sizes.append(box_size.copy())
        elif self.system is not None:
            self.box_sizes.append(self.system.box_size.copy())
        else:
            # Estimate box size from positions
            min_pos = np.min(positions, axis=0)
            max_pos = np.max(positions, axis=0)
            box_size = max_pos - min_pos
            self.box_sizes.append(box_size)
        
        if time is not None:
            self.time.append(time)
        elif self.system is not None and len(self.time) > 0:
            self.time.append(self.time[-1] + self.system.time_step)
        elif len(self.time) > 0:
            self.time.append(self.time[-1] + 1.0)  # Arbitrary time step
        else:
            self.time.append(0.0)
    
    def finalize(self):
        """Finalize the trajectory by converting lists to arrays."""
        self.positions = np.array(self.positions)
        
        if self.velocities:
            self.velocities = np.array(self.velocities)
        
        if self.forces:
            self.forces = np.array(self.forces)
        
        self.box_sizes = np.array(self.box_sizes)
        self.time = np.array(self.time)
        
        logger.info(f"Finalized trajectory with {len(self.positions)} frames")
    
    def load(self, filename):
        """
        Load a trajectory from a file.
        
        Parameters
        ----------
        filename : str
            Name of the trajectory file
        """
        # In a real implementation, would use a specialized format like
        # DCD, XTC, TRR, etc. using external libraries
        
        # Simplified implementation for demonstration
        logger.info(f"Loading trajectory from {filename}")
        
        # Here we'd actually parse the trajectory file
        # ...
        
        self.finalize()
    
    def save(self, filename, format="xtc"):
        """
        Save a trajectory to a file.
        
        Parameters
        ----------
        filename : str
            Name of the output file
        format : str
            Format of the output file (e.g., "xtc", "dcd", "trr")
        """
        # In a real implementation, would use a specialized format
        
        # Simplified implementation for demonstration
        logger.info(f"Saving trajectory to {filename} in {format} format")
        
        # Here we'd actually write the trajectory file
        # ...
    
    def slice(self, start=None, stop=None, step=None):
        """
        Create a new trajectory by slicing the current one.
        
        Parameters
        ----------
        start : int, optional
            Start frame (default: 0)
        stop : int, optional
            Stop frame (default: last frame)
        step : int, optional
            Step size (default: 1)
        
        Returns
        -------
        Trajectory
            A new trajectory with the selected frames
        """
        new_traj = Trajectory(self.system)
        
        # Convert to numpy arrays if not already
        if not isinstance(self.positions, np.ndarray):
            self.finalize()
        
        # Apply slice
        slice_obj = slice(start, stop, step)
        
        new_traj.positions = self.positions[slice_obj]
        new_traj.box_sizes = self.box_sizes[slice_obj]
        new_traj.time = self.time[slice_obj]
        
        if len(self.velocities) > 0:
            new_traj.velocities = self.velocities[slice_obj]
        
        if len(self.forces) > 0:
            new_traj.forces = self.forces[slice_obj]
        
        return new_traj
    
    def analyze(self, analysis, selection=None):
        """
        Analyze the trajectory.
        
        Parameters
        ----------
        analysis : Analysis
            The analysis method to use
        selection : ndarray, optional
            A subset of atoms to analyze
        
        Returns
        -------
        dict
            Analysis results
        """
        return analysis.analyze(self, selection)
