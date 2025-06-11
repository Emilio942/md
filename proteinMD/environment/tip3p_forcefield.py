"""
TIP3P Force Field Integration

This module integrates the TIP3P water model with the force field system,
providing force field parameters and force calculation methods specifically
for TIP3P water molecules.

Task 7.1: Multi-Threading Support Integration ✓
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from ..forcefield.forcefield import ForceField, ForceTerm

# Import parallel force calculator for Task 7.1
try:
    from .parallel_forces import get_parallel_calculator
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

logger = logging.getLogger(__name__)

class TIP3PWaterForceTerm(ForceTerm):
    """
    Force term specifically for TIP3P water-water interactions.
    
    This implements the optimized force calculations for TIP3P water molecules,
    including Lennard-Jones interactions on oxygen atoms and Coulomb interactions
    on all atoms.
    
    Task 7.1: Multi-Threading Support
    - OpenMP-style parallelization for force loops ✓
    - Thread-safe force calculations ✓
    - Scalable performance on 4+ cores ✓
    """
    
    def __init__(self, 
                 cutoff: float = 1.0,
                 water_molecule_indices: Optional[List[List[int]]] = None,
                 use_parallel: bool = True,
                 n_threads: Optional[int] = None):
        """
        Initialize TIP3P water force term.
        
        Parameters
        ----------
        cutoff : float, optional
            Cutoff distance for interactions in nm
        water_molecule_indices : list of lists, optional
            List of [O_idx, H1_idx, H2_idx] for each water molecule
        use_parallel : bool, optional
            Enable multi-threading for force calculations (Task 7.1)
        n_threads : int, optional
            Number of threads to use. If None, uses all available cores
        """
        super().__init__()
        self.cutoff = cutoff
        self.water_molecules = water_molecule_indices if water_molecule_indices is not None else []
        self.use_parallel = use_parallel and PARALLEL_AVAILABLE
        
        # Initialize parallel calculator if requested
        if self.use_parallel:
            self.parallel_calculator = get_parallel_calculator(n_threads)
            logger.info(f"Initialized TIP3P water force term with parallel support ({self.parallel_calculator.n_threads} threads)")
        else:
            self.parallel_calculator = None
            if use_parallel and not PARALLEL_AVAILABLE:
                logger.warning("Parallel force calculation requested but not available - using serial calculation")
            logger.info("Initialized TIP3P water force term with serial calculation")
        
        # TIP3P parameters
        self.oxygen_sigma = 0.31507  # nm
        self.oxygen_epsilon = 0.636  # kJ/mol
        self.oxygen_charge = -0.834  # e
        self.hydrogen_charge = 0.417  # e
        
        logger.info(f"TIP3P force term: {len(self.water_molecules)} molecules, cutoff={cutoff} nm")
    
    def add_water_molecule(self, oxygen_idx: int, hydrogen1_idx: int, hydrogen2_idx: int):
        """
        Add a water molecule to the force term.
        
        Parameters
        ----------
        oxygen_idx : int
            Index of oxygen atom
        hydrogen1_idx : int
            Index of first hydrogen atom
        hydrogen2_idx : int
            Index of second hydrogen atom
        """
        self.water_molecules.append([oxygen_idx, hydrogen1_idx, hydrogen2_idx])
    
    def calculate(self, positions: np.ndarray, box_vectors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Calculate TIP3P water-water interaction forces and energy.
        
        Task 7.1: Multi-Threading Support
        Uses parallel force calculation for significant speedup on multi-core systems.
        
        Parameters
        ----------
        positions : np.ndarray
            Particle positions with shape (n_particles, 3)
        box_vectors : np.ndarray, optional
            Periodic box vectors
            
        Returns
        -------
        tuple
            Tuple of (forces, potential_energy)
        """
        n_particles = positions.shape[0]
        forces = np.zeros((n_particles, 3))
        potential_energy = 0.0
        
        if len(self.water_molecules) < 2:
            return forces, potential_energy
        
        # Use parallel force calculation if available (Task 7.1)
        if self.use_parallel and self.parallel_calculator is not None:
            try:
                potential_energy, forces = self.parallel_calculator.calculate_water_water_forces_parallel(
                    positions, self.water_molecules, forces, self.cutoff, box_vectors
                )
                return forces, potential_energy
            except Exception as e:
                logger.warning(f"Parallel calculation failed, falling back to serial: {e}")
                # Fall through to serial calculation
        
        # Serial calculation (original implementation)
        # Coulomb constant
        coulomb_factor = 138.935458  # kJ·nm·mol⁻¹·e⁻²
        
        # Calculate water-water interactions
        for i in range(len(self.water_molecules)):
            mol_i = self.water_molecules[i]
            
            for j in range(i + 1, len(self.water_molecules)):
                mol_j = self.water_molecules[j]
                
                # Calculate all pairwise interactions between atoms in the two molecules
                for atom_i_idx, atom_i in enumerate(mol_i):
                    for atom_j_idx, atom_j in enumerate(mol_j):
                        
                        # Get positions
                        pos_i = positions[atom_i]
                        pos_j = positions[atom_j]
                        
                        # Calculate vector from i to j
                        r_vec = pos_j - pos_i
                        
                        # Apply minimum image convention if using periodic boundaries
                        if box_vectors is not None:
                            for k in range(3):
                                if box_vectors[k, k] > 0:
                                    while r_vec[k] > 0.5 * box_vectors[k, k]:
                                        r_vec[k] -= box_vectors[k, k]
                                    while r_vec[k] < -0.5 * box_vectors[k, k]:
                                        r_vec[k] += box_vectors[k, k]
                        
                        # Calculate distance
                        r = np.linalg.norm(r_vec)
                        
                        # Skip pairs beyond cutoff
                        if r > self.cutoff:
                            continue
                        
                        # Avoid division by zero
                        if r < 1e-10:
                            continue
                        
                        # Get charges
                        q_i = self.oxygen_charge if atom_i_idx == 0 else self.hydrogen_charge
                        q_j = self.oxygen_charge if atom_j_idx == 0 else self.hydrogen_charge
                        
                        # Coulomb interaction (all atom pairs)
                        coulomb_energy = coulomb_factor * q_i * q_j / r
                        coulomb_force_mag = coulomb_factor * q_i * q_j / r**2
                        
                        potential_energy += coulomb_energy
                        
                        # Lennard-Jones interaction (only O-O pairs)
                        lj_energy = 0.0
                        lj_force_mag = 0.0
                        
                        if atom_i_idx == 0 and atom_j_idx == 0:  # Both are oxygen atoms
                            # Calculate LJ terms
                            inv_r = 1.0 / r
                            sigma_r = self.oxygen_sigma * inv_r
                            sigma_r6 = sigma_r**6
                            sigma_r12 = sigma_r6**2
                            
                            lj_energy = 4.0 * self.oxygen_epsilon * (sigma_r12 - sigma_r6)
                            lj_force_mag = 24.0 * self.oxygen_epsilon * inv_r * (2.0 * sigma_r12 - sigma_r6)
                            
                            potential_energy += lj_energy
                        
                        # Total force magnitude
                        total_force_mag = coulomb_force_mag + lj_force_mag
                        
                        # Calculate force vector
                        force_vec = total_force_mag * r_vec / r
                        
                        # Apply forces (Newton's third law)
                        forces[atom_i] += force_vec
                        forces[atom_j] -= force_vec
        
        return forces, potential_energy

class TIP3PWaterProteinForceTerm(ForceTerm):
    """
    Force term for TIP3P water-protein interactions.
    
    This implements interactions between TIP3P water molecules and protein atoms,
    using standard AMBER force field combining rules.
    
    Task 7.1: Multi-Threading Support
    - Parallel water-protein force calculations ✓
    """
    
    def __init__(self, 
                 cutoff: float = 1.0,
                 water_molecule_indices: Optional[List[List[int]]] = None,
                 protein_atom_indices: Optional[List[int]] = None,
                 protein_lj_params: Optional[Dict[int, Tuple[float, float]]] = None,
                 protein_charges: Optional[Dict[int, float]] = None,
                 use_parallel: bool = True,
                 n_threads: Optional[int] = None):
        """
        Initialize water-protein force term.
        
        Parameters
        ----------
        cutoff : float, optional
            Cutoff distance for interactions in nm
        water_molecule_indices : list of lists, optional
            List of [O_idx, H1_idx, H2_idx] for each water molecule
        protein_atom_indices : list, optional
            List of protein atom indices
        protein_lj_params : dict, optional
            Map from atom index to (sigma, epsilon) parameters
        protein_charges : dict, optional
            Map from atom index to charge
        use_parallel : bool, optional
            Enable multi-threading for force calculations (Task 7.1)
        n_threads : int, optional
            Number of threads to use. If None, uses all available cores
        """
        super().__init__()
        self.cutoff = cutoff
        self.water_molecules = water_molecule_indices if water_molecule_indices is not None else []
        self.protein_atoms = protein_atom_indices if protein_atom_indices is not None else []
        self.protein_lj_params = protein_lj_params if protein_lj_params is not None else {}
        self.protein_charges = protein_charges if protein_charges is not None else {}
        self.use_parallel = use_parallel and PARALLEL_AVAILABLE
        
        # Initialize parallel calculator if requested
        if self.use_parallel:
            self.parallel_calculator = get_parallel_calculator(n_threads)
            logger.info(f"Initialized water-protein force term with parallel support ({self.parallel_calculator.n_threads} threads)")
        else:
            self.parallel_calculator = None
            if use_parallel and not PARALLEL_AVAILABLE:
                logger.warning("Parallel force calculation requested but not available - using serial calculation")
        
        # TIP3P parameters
        self.oxygen_sigma = 0.31507  # nm
        self.oxygen_epsilon = 0.636  # kJ/mol
        self.oxygen_charge = -0.834  # e
        self.hydrogen_charge = 0.417  # e
        
        logger.info(f"Initialized water-protein force term: {len(self.water_molecules)} waters, "
                   f"{len(self.protein_atoms)} protein atoms")
    
    def calculate(self, positions: np.ndarray, box_vectors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Calculate water-protein interaction forces and energy.
        
        Task 7.1: Multi-Threading Support
        Uses parallel force calculation for water-protein interactions.
        
        Parameters
        ----------
        positions : np.ndarray
            Particle positions with shape (n_particles, 3)
        box_vectors : np.ndarray, optional
            Periodic box vectors
            
        Returns
        -------
        tuple
            Tuple of (forces, potential_energy)
        """
        n_particles = positions.shape[0]
        forces = np.zeros((n_particles, 3))
        potential_energy = 0.0
        
        if len(self.water_molecules) == 0 or len(self.protein_atoms) == 0:
            return forces, potential_energy
        
        # Use parallel force calculation if available (Task 7.1)
        if self.use_parallel and self.parallel_calculator is not None:
            try:
                potential_energy, forces = self.parallel_calculator.calculate_water_protein_forces_parallel(
                    positions, self.water_molecules, self.protein_atoms, 
                    self.protein_charges, self.protein_lj_params, forces, self.cutoff, box_vectors
                )
                return forces, potential_energy
            except Exception as e:
                logger.warning(f"Parallel water-protein calculation failed, falling back to serial: {e}")
                # Fall through to serial calculation
        
        # Serial calculation (original implementation)
        # Coulomb constant
        coulomb_factor = 138.935458  # kJ·nm·mol⁻¹·e⁻²
        
        # Calculate water-protein interactions
        for mol_idx, water_mol in enumerate(self.water_molecules):
            for atom_idx, water_atom in enumerate(water_mol):
                
                # Get water atom properties
                water_pos = positions[water_atom]
                water_charge = self.oxygen_charge if atom_idx == 0 else self.hydrogen_charge
                
                # Only oxygen has LJ interactions
                water_sigma = self.oxygen_sigma if atom_idx == 0 else 0.0
                water_epsilon = self.oxygen_epsilon if atom_idx == 0 else 0.0
                
                for protein_atom in self.protein_atoms:
                    protein_pos = positions[protein_atom]
                    
                    # Calculate vector from water to protein
                    r_vec = protein_pos - water_pos
                    
                    # Apply minimum image convention if using periodic boundaries
                    if box_vectors is not None:
                        for k in range(3):
                            if box_vectors[k, k] > 0:
                                while r_vec[k] > 0.5 * box_vectors[k, k]:
                                    r_vec[k] -= box_vectors[k, k]
                                while r_vec[k] < -0.5 * box_vectors[k, k]:
                                    r_vec[k] += box_vectors[k, k]
                    
                    # Calculate distance
                    r = np.linalg.norm(r_vec)
                    
                    # Skip pairs beyond cutoff
                    if r > self.cutoff:
                        continue
                    
                    # Avoid division by zero
                    if r < 1e-10:
                        continue
                    
                    # Get protein atom properties
                    protein_charge = self.protein_charges.get(protein_atom, 0.0)
                    
                    # Coulomb interaction
                    coulomb_energy = coulomb_factor * water_charge * protein_charge / r
                    coulomb_force_mag = coulomb_factor * water_charge * protein_charge / r**2
                    
                    potential_energy += coulomb_energy
                    
                    # Lennard-Jones interaction (only for oxygen-protein)
                    lj_energy = 0.0
                    lj_force_mag = 0.0
                    
                    if atom_idx == 0 and protein_atom in self.protein_lj_params:  # Water oxygen
                        protein_sigma, protein_epsilon = self.protein_lj_params[protein_atom]
                        
                        # Combining rules (Lorentz-Berthelot)
                        combined_sigma = 0.5 * (water_sigma + protein_sigma)
                        combined_epsilon = np.sqrt(water_epsilon * protein_epsilon)
                        
                        # Calculate LJ terms
                        inv_r = 1.0 / r
                        sigma_r = combined_sigma * inv_r
                        sigma_r6 = sigma_r**6
                        sigma_r12 = sigma_r6**2
                        
                        lj_energy = 4.0 * combined_epsilon * (sigma_r12 - sigma_r6)
                        lj_force_mag = 24.0 * combined_epsilon * inv_r * (2.0 * sigma_r12 - sigma_r6)
                        
                        potential_energy += lj_energy
                    
                    # Total force magnitude
                    total_force_mag = coulomb_force_mag + lj_force_mag
                    
                    # Calculate force vector
                    force_vec = total_force_mag * r_vec / r
                    
                    # Apply forces (Newton's third law)
                    forces[water_atom] += force_vec
                    forces[protein_atom] -= force_vec
        
        return forces, potential_energy

class TIP3PWaterForceField(ForceField):
    """
    Force field specifically designed for TIP3P water simulations.
    
    This force field handles TIP3P water molecules with rigid geometry
    and optimized force calculations.
    """
    
    def __init__(self, 
                 cutoff: float = 1.0,
                 switch_distance: Optional[float] = None,
                 rigid_water: bool = True):
        """
        Initialize TIP3P water force field.
        
        Parameters
        ----------
        cutoff : float, optional
            Cutoff distance for non-bonded interactions in nm
        switch_distance : float, optional
            Switching distance for non-bonded interactions in nm
        rigid_water : bool, optional
            Whether to treat water molecules as rigid
        """
        super().__init__(
            name="TIP3P-Water",
            cutoff=cutoff,
            switch_distance=switch_distance
        )
        
        self.rigid_water = rigid_water
        self.water_molecules = []  # List of [O_idx, H1_idx, H2_idx]
        self.protein_atoms = []
        self.protein_lj_params = {}
        self.protein_charges = {}
        
        logger.info(f"Initialized TIP3P water force field (rigid={rigid_water})")
    
    def add_water_molecules(self, water_molecule_indices: List[List[int]]):
        """
        Add water molecules to the force field.
        
        Parameters
        ----------
        water_molecule_indices : list of lists
            List of [O_idx, H1_idx, H2_idx] for each water molecule
        """
        self.water_molecules.extend(water_molecule_indices)
        logger.info(f"Added {len(water_molecule_indices)} water molecules to force field")
    
    def add_protein_atoms(self, 
                         atom_indices: List[int],
                         lj_params: Dict[int, Tuple[float, float]],
                         charges: Dict[int, float]):
        """
        Add protein atoms to the force field.
        
        Parameters
        ----------
        atom_indices : list
            List of protein atom indices
        lj_params : dict
            Map from atom index to (sigma, epsilon) parameters
        charges : dict
            Map from atom index to charge
        """
        self.protein_atoms.extend(atom_indices)
        self.protein_lj_params.update(lj_params)
        self.protein_charges.update(charges)
        logger.info(f"Added {len(atom_indices)} protein atoms to force field")
    
    def create_system(self, topology=None, box_vectors=None):
        """
        Create a force field system with TIP3P water interactions.
        
        Parameters
        ----------
        topology : optional
            Molecular topology (not used for TIP3P)
        box_vectors : np.ndarray, optional
            Periodic box vectors
            
        Returns
        -------
        ForceFieldSystem
            System with TIP3P force terms
        """
        from ..forcefield.forcefield import ForceFieldSystem
        
        system = ForceFieldSystem(name=self.name)
        
        # Add water-water interactions
        if len(self.water_molecules) > 1:
            water_term = TIP3PWaterForceTerm(
                cutoff=self.cutoff,
                water_molecule_indices=self.water_molecules
            )
            system.add_force_term(water_term)
        
        # Add water-protein interactions
        if len(self.water_molecules) > 0 and len(self.protein_atoms) > 0:
            water_protein_term = TIP3PWaterProteinForceTerm(
                cutoff=self.cutoff,
                water_molecule_indices=self.water_molecules,
                protein_atom_indices=self.protein_atoms,
                protein_lj_params=self.protein_lj_params,
                protein_charges=self.protein_charges
            )
            system.add_force_term(water_protein_term)
        
        # Add constraints for rigid water if enabled
        if self.rigid_water:
            self._add_water_constraints(system)
        
        return system
    
    def _add_water_constraints(self, system):
        """
        Add distance constraints for rigid TIP3P water.
        
        Parameters
        ----------
        system : ForceFieldSystem
            System to add constraints to
        """
        # This would implement SHAKE/RATTLE constraints for rigid water
        # For now, we'll use very stiff harmonic bonds and angles
        
        from ..forcefield.forcefield import HarmonicBondForceTerm, HarmonicAngleForceTerm
        
        # Very stiff bonds for O-H
        oh_bonds = []
        for water_mol in self.water_molecules:
            o_idx, h1_idx, h2_idx = water_mol
            oh_bonds.append((o_idx, h1_idx, 4518.72, 0.09572))  # Very stiff k, TIP3P length
            oh_bonds.append((o_idx, h2_idx, 4518.72, 0.09572))
        
        if oh_bonds:
            bond_term = HarmonicBondForceTerm()
            bond_term.bonds = oh_bonds
            system.add_force_term(bond_term)
        
        # Very stiff angles for H-O-H
        hoh_angles = []
        for water_mol in self.water_molecules:
            o_idx, h1_idx, h2_idx = water_mol
            hoh_angles.append((h1_idx, o_idx, h2_idx, 682.02, np.radians(104.52)))  # Very stiff k, TIP3P angle
        
        if hoh_angles:
            angle_term = HarmonicAngleForceTerm()
            angle_term.angles = hoh_angles
            system.add_force_term(angle_term)
