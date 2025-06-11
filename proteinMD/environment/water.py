"""
TIP3P Water Model for Explicit Solvation

This module implements the TIP3P (Transferable Intermolecular Potential 3-Point) water model
for explicit solvation in molecular dynamics simulations.

The TIP3P model represents water molecules with three interaction sites:
- Oxygen atom with Lennard-Jones and electrostatic interactions
- Two hydrogen atoms with only electrostatic interactions
- Rigid geometry with fixed bond lengths and angles

Reference: W. L. Jorgensen et al., J. Chem. Phys. 79, 926 (1983)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import logging
from scipy.spatial.distance import cdist
import random

logger = logging.getLogger(__name__)

class TIP3PWaterModel:
    """
    TIP3P water model implementation.
    
    This class provides the TIP3P water model parameters and geometry
    for explicit water solvation in molecular dynamics simulations.
    """
    
    # TIP3P Parameters (standard values)
    # Oxygen parameters
    OXYGEN_SIGMA = 0.31507  # nm (Lennard-Jones sigma)
    OXYGEN_EPSILON = 0.636  # kJ/mol (Lennard-Jones epsilon)
    OXYGEN_CHARGE = -0.834  # elementary charges
    OXYGEN_MASS = 15.9994  # atomic mass units (u)
    
    # Hydrogen parameters
    HYDROGEN_SIGMA = 0.0  # nm (no LJ interaction for H in TIP3P)
    HYDROGEN_EPSILON = 0.0  # kJ/mol
    HYDROGEN_CHARGE = 0.417  # elementary charges (2 * 0.417 + (-0.834) = 0)
    HYDROGEN_MASS = 1.008  # atomic mass units (u)
    
    # Geometry parameters
    OH_BOND_LENGTH = 0.09572  # nm
    HOH_ANGLE = 104.52  # degrees
    
    # Water density parameters
    WATER_DENSITY = 997.0  # kg/m³ at 25°C, 1 atm
    WATER_MOLAR_MASS = 18.01528  # g/mol
    AVOGADRO = 6.02214076e23  # molecules/mol
    
    def __init__(self):
        """Initialize the TIP3P water model."""
        logger.info("Initialized TIP3P water model")
        
        # Calculate derived properties
        self.water_molecule_mass = self.OXYGEN_MASS + 2 * self.HYDROGEN_MASS  # u
        self.expected_number_density = self._calculate_number_density()  # molecules/nm³
        
    def _calculate_number_density(self) -> float:
        """
        Calculate the expected number density of water molecules.
        
        Returns
        -------
        float
            Number density in molecules/nm³
        """
        # Convert density from kg/m³ to molecules/nm³
        # 1 m³ = (10⁹ nm)³ = 10²⁷ nm³, so 1 kg/m³ = 1e-27 kg/nm³
        # molecules/nm³ = (density in kg/nm³) * (Avogadro/molar_mass)
        
        density_kg_per_nm3 = self.WATER_DENSITY * 1e-27  # kg/nm³
        molar_mass_kg = self.WATER_MOLAR_MASS * 1e-3  # kg/mol
        
        number_density = density_kg_per_nm3 * self.AVOGADRO / molar_mass_kg
        
        logger.debug(f"Expected water number density: {number_density:.3f} molecules/nm³")
        return number_density
    
    def create_single_water_molecule(self, 
                                   center_position: np.ndarray, 
                                   orientation: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Create a single TIP3P water molecule at a specified position.
        
        Parameters
        ----------
        center_position : np.ndarray
            Position of the oxygen atom (center) in nm
        orientation : np.ndarray, optional
            Orientation vector for the molecule. If None, random orientation is used.
            
        Returns
        -------
        dict
            Dictionary containing positions, masses, charges, and atom types
        """
        if orientation is None:
            # Random orientation
            theta = random.uniform(0, 2 * np.pi)
            phi = random.uniform(0, np.pi)
            orientation = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
        
        # Normalize orientation vector
        orientation = orientation / np.linalg.norm(orientation)
        
        # Calculate positions of hydrogen atoms
        angle_rad = np.radians(self.HOH_ANGLE / 2)  # Half angle from O-H to O-H bisector
        
        # Create two H positions symmetrically around the orientation axis
        # First, create a perpendicular vector
        if abs(orientation[2]) < 0.9:
            perp = np.array([0, 0, 1])
        else:
            perp = np.array([1, 0, 0])
        
        # Make it perpendicular to orientation
        perp = perp - np.dot(perp, orientation) * orientation
        perp = perp / np.linalg.norm(perp)
        
        # Create the second perpendicular vector
        perp2 = np.cross(orientation, perp)
        
        # Calculate H positions
        h1_direction = (
            np.cos(angle_rad) * orientation +
            np.sin(angle_rad) * perp
        )
        h2_direction = (
            np.cos(angle_rad) * orientation +
            np.sin(angle_rad) * (-perp)  # Symmetric to h1
        )
        
        # Scale by bond length and add to oxygen position
        h1_position = center_position + self.OH_BOND_LENGTH * h1_direction
        h2_position = center_position + self.OH_BOND_LENGTH * h2_direction
        
        # Return molecule data
        return {
            'positions': np.array([center_position, h1_position, h2_position]),
            'masses': np.array([self.OXYGEN_MASS, self.HYDROGEN_MASS, self.HYDROGEN_MASS]),
            'charges': np.array([self.OXYGEN_CHARGE, self.HYDROGEN_CHARGE, self.HYDROGEN_CHARGE]),
            'atom_types': ['O', 'H', 'H'],
            'molecule_type': 'TIP3P_water'
        }
    
    def get_water_force_field_parameters(self) -> Dict[str, Dict]:
        """
        Get force field parameters for TIP3P water model.
        
        Returns
        -------
        dict
            Dictionary containing force field parameters
        """
        return {
            'nonbonded': {
                'O': {
                    'sigma': self.OXYGEN_SIGMA,
                    'epsilon': self.OXYGEN_EPSILON,
                    'charge': self.OXYGEN_CHARGE,
                    'mass': self.OXYGEN_MASS
                },
                'H': {
                    'sigma': self.HYDROGEN_SIGMA,
                    'epsilon': self.HYDROGEN_EPSILON,
                    'charge': self.HYDROGEN_CHARGE,
                    'mass': self.HYDROGEN_MASS
                }
            },
            'bonds': {
                ('O', 'H'): {
                    'length': self.OH_BOND_LENGTH,
                    'force_constant': 4518.72  # kJ/(mol·nm²) - very stiff for rigid water
                }
            },
            'angles': {
                ('H', 'O', 'H'): {
                    'angle': self.HOH_ANGLE,
                    'force_constant': 682.02  # kJ/(mol·rad²) - very stiff for rigid water
                }
            }
        }

class WaterSolvationBox:
    """
    Class for creating and managing water solvation around biomolecules.
    
    This class provides functionality to:
    - Place water molecules around proteins
    - Maintain minimum distances
    - Calculate densities
    - Validate solvation quality
    """
    
    def __init__(self, 
                 tip3p_model: Optional[TIP3PWaterModel] = None,
                 min_distance_to_solute: float = 0.23,  # nm
                 min_water_distance: float = 0.23,  # nm
                 target_density: Optional[float] = None):
        """
        Initialize the water solvation box.
        
        Parameters
        ----------
        tip3p_model : TIP3PWaterModel, optional
            TIP3P model instance. If None, creates a new one.
        min_distance_to_solute : float, optional
            Minimum distance from water to solute atoms in nm
        min_water_distance : float, optional
            Minimum distance between water molecules in nm
        target_density : float, optional
            Target density in kg/m³. If None, uses standard water density.
        """
        self.tip3p = tip3p_model if tip3p_model is not None else TIP3PWaterModel()
        self.min_distance_to_solute = min_distance_to_solute
        self.min_water_distance = min_water_distance
        self.target_density = target_density if target_density is not None else self.tip3p.WATER_DENSITY
        
        # Storage for placed water molecules
        self.water_molecules = []
        self.water_positions = []  # Only oxygen positions for distance checking
        
        logger.info(f"Initialized water solvation box with min distances: "
                   f"solute={min_distance_to_solute:.3f} nm, water={min_water_distance:.3f} nm")
    
    def solvate_protein(self, 
                       protein_positions: np.ndarray,
                       box_dimensions: np.ndarray,
                       padding: float = 1.0,
                       max_attempts: int = 100000) -> Dict[str, np.ndarray]:
        """
        Solvate a protein with TIP3P water molecules.
        
        Parameters
        ----------
        protein_positions : np.ndarray
            Positions of protein atoms with shape (n_atoms, 3) in nm
        box_dimensions : np.ndarray
            Box dimensions [x, y, z] in nm
        padding : float, optional
            Minimum padding around protein in nm
        max_attempts : int, optional
            Maximum attempts to place water molecules
            
        Returns
        -------
        dict
            Dictionary containing water molecule data
        """
        logger.info(f"Starting protein solvation in box {box_dimensions} nm")
        
        # Clear existing water
        self.water_molecules = []
        self.water_positions = []
        
        # Calculate available volume for water
        protein_bbox_min = np.min(protein_positions, axis=0) - padding
        protein_bbox_max = np.max(protein_positions, axis=0) + padding
        
        # Ensure protein fits in box
        if np.any(protein_bbox_min < 0) or np.any(protein_bbox_max > box_dimensions):
            logger.warning("Protein with padding extends beyond box boundaries")
            
        # Estimate number of water molecules needed
        total_volume = np.prod(box_dimensions)  # nm³
        protein_volume = self._estimate_protein_volume(protein_positions)
        available_volume = total_volume - protein_volume
        
        target_n_waters = int(available_volume * self.tip3p.expected_number_density * 0.9)  # 90% of max density
        logger.info(f"Target number of water molecules: {target_n_waters}")
        
        # Place water molecules
        placed_count = 0
        attempts = 0
        
        while placed_count < target_n_waters and attempts < max_attempts:
            # Generate random position in box
            position = np.random.uniform(0, box_dimensions)
            
            # Check distance to protein
            if self._check_distance_to_protein(position, protein_positions):
                # Check distance to existing water
                if self._check_distance_to_water(position):
                    # Place water molecule
                    water_mol = self.tip3p.create_single_water_molecule(position)
                    self.water_molecules.append(water_mol)
                    self.water_positions.append(position)
                    placed_count += 1
                    
                    if placed_count % 1000 == 0:
                        logger.debug(f"Placed {placed_count}/{target_n_waters} water molecules")
            
            attempts += 1
            
            if attempts % 10000 == 0:
                logger.debug(f"Attempt {attempts}/{max_attempts}, placed {placed_count} waters")
        
        logger.info(f"Placed {placed_count} water molecules in {attempts} attempts")
        
        # Combine all water molecule data
        return self._combine_water_data()
    
    def _estimate_protein_volume(self, protein_positions: np.ndarray) -> float:
        """
        Estimate the volume occupied by the protein.
        
        Parameters
        ----------
        protein_positions : np.ndarray
            Protein atom positions
            
        Returns
        -------
        float
            Estimated volume in nm³
        """
        # Simple bounding box estimate
        bbox_min = np.min(protein_positions, axis=0)
        bbox_max = np.max(protein_positions, axis=0)
        bbox_volume = np.prod(bbox_max - bbox_min)
        
        # Assume protein occupies ~60% of its bounding box
        return bbox_volume * 0.6
    
    def _check_distance_to_protein(self, position: np.ndarray, protein_positions: np.ndarray) -> bool:
        """
        Check if a water position is far enough from protein atoms.
        
        Parameters
        ----------
        position : np.ndarray
            Proposed water position
        protein_positions : np.ndarray
            Protein atom positions
            
        Returns
        -------
        bool
            True if position is acceptable
        """
        min_dist = np.min(np.linalg.norm(protein_positions - position, axis=1))
        return min_dist >= self.min_distance_to_solute
    
    def _check_distance_to_water(self, position: np.ndarray) -> bool:
        """
        Check if a water position is far enough from existing water molecules.
        
        Parameters
        ----------
        position : np.ndarray
            Proposed water position
            
        Returns
        -------
        bool
            True if position is acceptable
        """
        if len(self.water_positions) == 0:
            return True
            
        water_positions = np.array(self.water_positions)
        min_dist = np.min(np.linalg.norm(water_positions - position, axis=1))
        return min_dist >= self.min_water_distance
    
    def _combine_water_data(self) -> Dict[str, np.ndarray]:
        """
        Combine data from all water molecules into simulation arrays.
        
        Returns
        -------
        dict
            Combined water molecule data
        """
        if not self.water_molecules:
            return {
                'positions': np.array([]).reshape(0, 3),
                'masses': np.array([]),
                'charges': np.array([]),
                'atom_types': [],
                'molecule_indices': np.array([]),
                'bonds': [],
                'angles': []
            }
        
        # Combine positions, masses, charges
        all_positions = []
        all_masses = []
        all_charges = []
        all_atom_types = []
        molecule_indices = []
        bonds = []
        angles = []
        
        atom_index = 0
        for mol_idx, mol in enumerate(self.water_molecules):
            all_positions.append(mol['positions'])
            all_masses.append(mol['masses'])
            all_charges.append(mol['charges'])
            all_atom_types.extend(mol['atom_types'])
            
            # Track which atoms belong to which molecule
            molecule_indices.extend([mol_idx] * 3)  # 3 atoms per water
            
            # Add bonds (O-H1, O-H2)
            bonds.append([atom_index, atom_index + 1])  # O-H1
            bonds.append([atom_index, atom_index + 2])  # O-H2
            
            # Add angle (H1-O-H2)
            angles.append([atom_index + 1, atom_index, atom_index + 2])  # H1-O-H2
            
            atom_index += 3
        
        return {
            'positions': np.vstack(all_positions),
            'masses': np.concatenate(all_masses),
            'charges': np.concatenate(all_charges),
            'atom_types': all_atom_types,
            'molecule_indices': np.array(molecule_indices),
            'bonds': bonds,
            'angles': angles,
            'n_molecules': len(self.water_molecules)
        }
    
    def calculate_water_density(self, box_dimensions: np.ndarray) -> float:
        """
        Calculate the density of the water in the box.
        
        Parameters
        ----------
        box_dimensions : np.ndarray
            Box dimensions in nm
            
        Returns
        -------
        float
            Water density in kg/m³
        """
        if not self.water_molecules:
            return 0.0
        
        n_water = len(self.water_molecules)
        box_volume_nm3 = np.prod(box_dimensions)  # nm³
        box_volume_m3 = box_volume_nm3 * 1e-27  # m³
        
        # Mass of water molecules
        total_mass_u = n_water * self.tip3p.water_molecule_mass  # atomic mass units
        total_mass_kg = total_mass_u * 1.66054e-27  # kg
        
        density = total_mass_kg / box_volume_m3  # kg/m³
        
        logger.info(f"Water density: {density:.1f} kg/m³ (target: {self.target_density:.1f} kg/m³)")
        
        return density
    
    def validate_solvation(self, box_dimensions: np.ndarray, protein_positions: np.ndarray) -> Dict[str, float]:
        """
        Validate the quality of the solvation.
        
        Parameters
        ----------
        box_dimensions : np.ndarray
            Box dimensions in nm
        protein_positions : np.ndarray
            Protein atom positions
            
        Returns
        -------
        dict
            Validation metrics
        """
        metrics = {}
        
        # Calculate density
        density = self.calculate_water_density(box_dimensions)
        metrics['density_kg_m3'] = density
        metrics['density_relative_error'] = abs(density - self.target_density) / self.target_density
        
        # Check minimum distances
        water_data = self._combine_water_data()
        if water_data['positions'].shape[0] > 0:
            water_oxygens = water_data['positions'][::3]  # Every third atom is oxygen
            
            # Distance to protein
            if len(protein_positions) > 0:
                distances_to_protein = cdist(water_oxygens, protein_positions)
                min_dist_to_protein = np.min(distances_to_protein)
                metrics['min_distance_to_protein'] = min_dist_to_protein
                metrics['protein_distance_violations'] = np.sum(distances_to_protein < self.min_distance_to_solute)
            
            # Water-water distances
            if len(water_oxygens) > 1:
                water_distances = cdist(water_oxygens, water_oxygens)
                # Set diagonal to large value to ignore self-distances
                np.fill_diagonal(water_distances, 1000.0)
                min_water_dist = np.min(water_distances)
                metrics['min_water_distance'] = min_water_dist
                metrics['water_distance_violations'] = np.sum(water_distances < self.min_water_distance)
        
        # Number of water molecules
        metrics['n_water_molecules'] = len(self.water_molecules)
        
        # Volume fraction
        total_volume = np.prod(box_dimensions)
        water_volume = len(self.water_molecules) / self.tip3p.expected_number_density
        metrics['water_volume_fraction'] = water_volume / total_volume
        
        return metrics

def create_pure_water_box(box_dimensions: np.ndarray, 
                         target_density: Optional[float] = None,
                         min_water_distance: float = 0.23) -> Dict[str, np.ndarray]:
    """
    Create a box of pure TIP3P water for density validation.
    
    Parameters
    ----------
    box_dimensions : np.ndarray
        Box dimensions [x, y, z] in nm
    target_density : float, optional
        Target density in kg/m³
    min_water_distance : float, optional
        Minimum distance between water molecules in nm
        
    Returns
    -------
    dict
        Water system data including positions, masses, charges, etc.
    """
    logger.info(f"Creating pure water box with dimensions {box_dimensions} nm")
    
    # Create solvation box
    solvation = WaterSolvationBox(
        min_distance_to_solute=0.0,  # No solute
        min_water_distance=min_water_distance,
        target_density=target_density
    )
    
    # "Solvate" with no protein (empty positions array)
    empty_protein = np.array([]).reshape(0, 3)
    water_data = solvation.solvate_protein(empty_protein, box_dimensions)
    
    # Calculate and report density
    density = solvation.calculate_water_density(box_dimensions)
    logger.info(f"Created pure water box with {water_data['n_molecules']} molecules, "
               f"density = {density:.1f} kg/m³")
    
    return water_data

class WaterSystem:
    """
    High-level interface for water system creation and management.
    
    This class provides a simplified interface for creating water boxes
    and managing TIP3P water systems for molecular dynamics simulations.
    """
    
    def __init__(self):
        """Initialize water system."""
        self.tip3p_model = TIP3PWaterModel()
        self.solvation_box = WaterSolvationBox()
    
    def create_water_box(self, n_water: int, box_size: float, density: float = 1.0):
        """
        Create a box of water molecules.
        
        Args:
            n_water: Number of water molecules
            box_size: Size of the cubic box in nm
            density: Target density in g/cm³
            
        Returns:
            tuple: (positions, atom_types) where positions is (N, 3) array
                  and atom_types is list of atom type strings
        """
        # Use the existing pure water box creation
        box_dimensions = np.array([box_size, box_size, box_size])
        water_data = create_pure_water_box(box_dimensions)
        
        # Extract positions and atom types
        positions = water_data['positions']
        
        # Limit to requested number of water molecules
        if n_water * 3 < len(positions):
            positions = positions[:n_water * 3]
        
        atom_types = ['O', 'H', 'H'] * min(n_water, len(positions) // 3)
        
        return positions, atom_types
    
    def calculate_water_density(self, box_dimensions, n_water_molecules):
        """
        Calculate water density for given box and number of molecules.
        
        Args:
            box_dimensions: Box dimensions in nm
            n_water_molecules: Number of water molecules
            
        Returns:
            float: Density in kg/m³
        """
        return self.solvation_box.calculate_water_density(box_dimensions)
