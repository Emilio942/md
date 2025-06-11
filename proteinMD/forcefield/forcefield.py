"""
Force field module for molecular dynamics simulations.

This module provides classes and functions for calculating forces
and energies using various biomolecular force fields.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from enum import Enum
import os
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Import AMBER validation
try:
    from .amber_validator import AMBERParameterValidator, ValidationResult
    AMBER_VALIDATION_AVAILABLE = True
except ImportError:
    try:
        from proteinMD.forcefield.amber_validator import AMBERParameterValidator, ValidationResult
        AMBER_VALIDATION_AVAILABLE = True
    except ImportError:
        logger.warning("AMBER validation not available")
        AMBER_VALIDATION_AVAILABLE = False

class ForceFieldType(Enum):
    """Enumeration of supported force field types."""
    AMBER = "amber"
    CHARMM = "charmm"
    OPLS = "opls"
    GROMOS = "gromos"
    MARTINI = "martini"  # Coarse-grained
    CUSTOM = "custom"

class NonbondedMethod(Enum):
    """Enumeration of methods for handling nonbonded interactions."""
    NoCutoff = "nocutoff"  # No cutoff (full N² calculation)
    CutoffNonPeriodic = "cutoffnonperiodic"  # Simple cutoff, non-periodic
    CutoffPeriodic = "cutoffperiodic"  # Simple cutoff with periodic boundaries
    PME = "pme"  # Particle Mesh Ewald for electrostatics
    LJPME = "ljpme"  # PME for both electrostatics and LJ

class ForceField:
    """
    Base class for molecular force fields.
    
    A force field defines the potential energy function and parameters
    used in molecular dynamics simulations to model the interactions
    between atoms and molecules.
    """
    
    def __init__(self, 
                 name: str = "ForceField",
                 cutoff: float = 1.0,  # nm
                 switch_distance: Optional[float] = None,  # nm
                 nonbonded_method: NonbondedMethod = NonbondedMethod.PME,
                 use_long_range_correction: bool = True,
                 ewaldErrorTolerance: float = 0.0001,
                 validate_parameters: bool = True):
        """
        Initialize a force field.
        
        Parameters
        ----------
        name : str, optional
            Name of the force field
        cutoff : float, optional
            Cutoff distance for nonbonded interactions in nm
        switch_distance : float, optional
            Distance at which to start switching the LJ potential
        nonbonded_method : NonbondedMethod, optional
            Method for handling nonbonded interactions
        use_long_range_correction : bool, optional
            Whether to use a long-range correction for LJ interactions
        ewaldErrorTolerance : float, optional
            Error tolerance for Ewald summation
        validate_parameters : bool, optional
            Whether to validate parameters automatically
        """
        # Validate cutoff
        if cutoff < 0:
            raise ValueError("Cutoff distance must be non-negative")
        if switch_distance is not None and switch_distance > cutoff:
            raise ValueError("Switch distance cannot be greater than cutoff")
            
        self.name = name
        self.cutoff = cutoff
        self.switch_distance = switch_distance
        self.nonbonded_method = nonbonded_method
        self.use_long_range_correction = use_long_range_correction
        self.ewaldErrorTolerance = ewaldErrorTolerance
        self.validate_parameters = validate_parameters
        
        # Data structures to store force field parameters
        self.atom_types = {}  # Map atom type name to parameters
        self.bond_types = {}  # Map bond type name to parameters
        self.angle_types = {}  # Map angle type name to parameters
        self.dihedral_types = {}  # Map dihedral type name to parameters
        self.improper_types = {}  # Map improper dihedral type name to parameters
        self.nonbonded_pairs = {}  # Map atom type pairs to LJ parameters
        
        # Initialize AMBER validator if available
        self.amber_validator = None
        if AMBER_VALIDATION_AVAILABLE and self.validate_parameters:
            self.amber_validator = AMBERParameterValidator()
            logger.info("AMBER parameter validation enabled")
        
        logger.info(f"Initialized {name} force field with {nonbonded_method.value} nonbonded method")
    
    def validate_system_parameters(self, protein_structure) -> Optional[ValidationResult]:
        """
        Validate force field parameters for a protein system.
        
        Parameters
        ----------
        protein_structure : Protein
            Protein structure to validate
            
        Returns
        -------
        ValidationResult or None
            Validation results if AMBER validation is available
        """
        if not self.amber_validator:
            logger.warning("AMBER validation not available - skipping parameter validation")
            return None
        
        logger.info("🔍 Validating AMBER force field parameters...")
        result = self.amber_validator.validate_protein_parameters(protein_structure)
        
        if result.is_valid:
            logger.info("✅ All force field parameters are valid!")
        else:
            logger.error("❌ Force field parameter validation failed!")
            logger.error(f"Missing atom types: {len(result.missing_atom_types)}")
            logger.error(f"Missing bonds: {len(result.missing_bonds)}")
            logger.error(f"Errors: {len(result.errors)}")
            
            # Log detailed errors
            for error in result.errors[:5]:  # Show first 5 errors
                logger.error(f"  - {error}")
        
        return result
    
    def check_parameter_coverage(self, atom_types: List[str]) -> Dict[str, bool]:
        """
        Check parameter coverage for a list of atom types.
        
        Parameters
        ----------
        atom_types : list
            List of atom types to check
            
        Returns
        -------
        dict
            Coverage status for each atom type
        """
        if not self.amber_validator:
            logger.warning("AMBER validation not available")
            return {atom_type: False for atom_type in atom_types}
        
        coverage = {}
        for atom_type in atom_types:
            is_valid, _ = self.amber_validator.validate_atom_type(atom_type)
            coverage[atom_type] = is_valid
        
        return coverage
    
    def get_missing_parameters_report(self, protein_structure) -> str:
        """
        Generate a detailed report of missing parameters.
        
        Parameters
        ----------
        protein_structure : Protein
            Protein structure to analyze
            
        Returns
        -------
        str
            Detailed missing parameters report
        """
        if not self.amber_validator:
            return "AMBER validation not available"
        
        result = self.validate_system_parameters(protein_structure)
        if result:
            return self.amber_validator.generate_validation_report([result], [protein_structure.name if hasattr(protein_structure, 'name') else 'Unknown'])
        else:
            return "Validation failed"
    
    def load_parameters(self, param_files: List[str]):
        """
        Load force field parameters from files.
        
        Parameters
        ----------
        param_files : list of str
            List of parameter file paths
        """
        for file_path in param_files:
            if not os.path.exists(file_path):
                logger.warning(f"Parameter file not found: {file_path}")
                continue
            
            logger.info(f"Loading parameters from {file_path}")
            
            # Read parameters (implementation depends on file format)
            self._read_parameter_file(file_path)
    
    def _read_parameter_file(self, file_path: str):
        """
        Read a force field parameter file.
        
        Parameters
        ----------
        file_path : str
            Path to the parameter file
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def assign_parameters(self, topology):
        """
        Assign force field parameters to a molecular topology.
        
        Parameters
        ----------
        topology : molecular topology
            Molecular topology to assign parameters to
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def create_system(self, topology, box_vectors=None):
        """
        Create a system object with all necessary force terms.
        
        Parameters
        ----------
        topology : molecular topology
            Molecular topology for the system
        box_vectors : np.ndarray, optional
            Periodic box vectors
            
        Returns
        -------
        ForceFieldSystem
            System object with force terms
        """
        system = ForceFieldSystem(name=self.name)
        self.assign_parameters(topology)
        
        # Create force terms
        self._add_bonded_forces(system, topology)
        self._add_nonbonded_forces(system, topology, box_vectors)
        
        return system
    
    def _add_bonded_forces(self, system, topology):
        """
        Add bonded force terms to the system.
        
        Parameters
        ----------
        system : ForceFieldSystem
            System to add forces to
        topology : molecular topology
            Molecular topology
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _add_nonbonded_forces(self, system, topology, box_vectors):
        """
        Add nonbonded force terms to the system.
        
        Parameters
        ----------
        system : ForceFieldSystem
            System to add forces to
        topology : molecular topology
            Molecular topology
        box_vectors : np.ndarray, optional
            Periodic box vectors
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def calculate_forces(self, positions, box_vectors=None):
        """
        Calculate forces for a set of particle positions.
        
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
        raise NotImplementedError("Subclasses must implement this method")


class ForceFieldSystem:
    """
    Represents a simulation system with force field terms.
    """
    
    def __init__(self, name: str = ""):
        """
        Initialize a force field system.
        
        Parameters
        ----------
        name : str, optional
            Name of the system
        """
        self.name = name
        self.force_terms = []  # List of force terms
        self.n_atoms = 0  # Number of atoms in the system
    
    def add_force_term(self, force_term):
        """
        Add a force term to the system.
        
        Parameters
        ----------
        force_term : ForceTerm
            Force term to add
        """
        self.force_terms.append(force_term)
    
    def calculate_forces(self, positions, box_vectors=None):
        """
        Calculate forces for all terms.
        
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
        
        # Calculate forces for each term
        for term in self.force_terms:
            result = term.calculate(positions, box_vectors)
            if isinstance(result, tuple) and len(result) == 2:
                term_forces, term_energy = result
                forces += term_forces
                potential_energy += term_energy
            else:
                # Handle mock objects or invalid returns gracefully
                logger.warning(f"Force term {term.name} returned invalid result: {result}")
                continue
        
        return forces, potential_energy


class ForceTerm:
    """
    Base class for individual force terms in a force field.
    """
    
    def __init__(self, name: str):
        """
        Initialize a force term.
        
        Parameters
        ----------
        name : str
            Name of the force term
        """
        self.name = name
    
    def calculate(self, positions, box_vectors=None):
        """
        Calculate forces for this term.
        
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
        raise NotImplementedError("Subclasses must implement this method")


class HarmonicBondForceTerm(ForceTerm):
    """
    Force term for harmonic bond potentials.
    
    The harmonic bond potential is defined as:
    V(r) = 0.5 * k * (r - r0)^2
    
    where k is the spring constant, r is the bond length,
    and r0 is the equilibrium bond length.
    """
    
    def __init__(self, bonds=None, k_values=None, r0_values=None, name: str = "HarmonicBond"):
        """
        Initialize a harmonic bond force term.
        
        Parameters
        ----------
        bonds : list, optional
            List of (particle1, particle2) tuples
        k_values : list, optional
            List of spring constants in kJ/(mol*nm^2)
        r0_values : list, optional
            List of equilibrium bond lengths in nm
        name : str, optional
            Name of the force term
        """
        super().__init__(name)
        self.bonds = []  # List of (particle1, particle2, k, r0) tuples
        
        # Support both old constructor style and new test style
        if bonds is not None and k_values is not None and r0_values is not None:
            if len(bonds) != len(k_values) or len(bonds) != len(r0_values):
                raise ValueError("bonds, k_values, and r0_values must have the same length")
            
            # Store for test compatibility
            self.k_values = k_values
            self.r0_values = r0_values
            
            # Convert to internal format
            for i, (p1, p2) in enumerate(bonds):
                self.add_bond(p1, p2, k_values[i], r0_values[i])
    
    def add_bond(self, particle1: int, particle2: int, k: float, r0: float):
        """
        Add a bond to the force term.
        
        Parameters
        ----------
        particle1 : int
            Index of the first particle
        particle2 : int
            Index of the second particle
        k : float
            Spring constant in kJ/(mol*nm^2)
        r0 : float
            Equilibrium bond length in nm
        """
        self.bonds.append((particle1, particle2, k, r0))
    
    def calculate(self, positions, box_vectors=None):
        """
        Calculate bond forces.
        
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
        
        for p1, p2, k, r0 in self.bonds:
            # Get positions
            pos1 = positions[p1]
            pos2 = positions[p2]
            
            # Calculate vector from p1 to p2
            r_vec = pos2 - pos1
            
            # Apply minimum image convention if using periodic boundaries
            if box_vectors is not None:
                for i in range(3):
                    if box_vectors[i, i] > 0:
                        while r_vec[i] > 0.5 * box_vectors[i, i]:
                            r_vec[i] -= box_vectors[i, i]
                        while r_vec[i] < -0.5 * box_vectors[i, i]:
                            r_vec[i] += box_vectors[i, i]
            
            # Calculate bond length
            r = np.linalg.norm(r_vec)
            
            # Avoid division by zero
            if r < 1e-10:
                continue
            
            # Calculate force magnitude: dV/dr = k * (r - r0)
            force_mag = k * (r - r0)
            
            # Calculate force vector direction
            force_dir = r_vec / r
            
            # Calculate force vectors
            force_vec = force_mag * force_dir
            
            # Apply forces (Newton's third law)
            forces[p1] += force_vec
            forces[p2] -= force_vec
            
            # Calculate potential energy: V = 0.5 * k * (r - r0)^2
            potential_energy += 0.5 * k * (r - r0)**2
        
        return forces, potential_energy
    
    def calculate_forces(self, positions, box_vectors=None):
        """
        Calculate bond forces (alias for calculate method for test compatibility).
        
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
        return self.calculate(positions, box_vectors)


class HarmonicAngleForceTerm(ForceTerm):
    """
    Force term for harmonic angle potentials.
    
    The harmonic angle potential is defined as:
    V(θ) = 0.5 * k * (θ - θ0)^2
    
    where k is the angle constant, θ is the angle between three particles,
    and θ0 is the equilibrium angle.
    """
    
    def __init__(self, angles=None, k_values=None, theta0_values=None, name: str = "HarmonicAngle"):
        """
        Initialize a harmonic angle force term.
        
        Parameters
        ----------
        angles : list, optional
            List of (particle1, particle2, particle3) tuples
        k_values : list, optional
            List of angle constants in kJ/(mol*rad^2)
        theta0_values : list, optional
            List of equilibrium angles in degrees
        name : str, optional
            Name of the force term
        """
        super().__init__(name)
        self.angles = []  # List of (particle1, particle2, particle3, k, theta0) tuples
        
        # Support both old constructor style and new test style
        if angles is not None and k_values is not None and theta0_values is not None:
            if len(angles) != len(k_values) or len(angles) != len(theta0_values):
                raise ValueError("angles, k_values, and theta0_values must have the same length")
            
            # Store for test compatibility
            self.k_values = k_values
            self.theta0_values = theta0_values
            
            # Convert to internal format (theta0 in radians)
            for i, (p1, p2, p3) in enumerate(angles):
                theta0_rad = np.radians(theta0_values[i])  # Convert degrees to radians
                self.add_angle(p1, p2, p3, k_values[i], theta0_rad)
    
    def add_angle(self, particle1: int, particle2: int, particle3: int, k: float, theta0: float):
        """
        Add an angle to the force term.
        
        Parameters
        ----------
        particle1 : int
            Index of the first particle
        particle2 : int
            Index of the second particle (central)
        particle3 : int
            Index of the third particle
        k : float
            Angle constant in kJ/(mol*rad^2)
        theta0 : float
            Equilibrium angle in radians
        """
        self.angles.append((particle1, particle2, particle3, k, theta0))
    
    def calculate(self, positions, box_vectors=None):
        """
        Calculate angle forces.
        
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
        
        for p1, p2, p3, k, theta0 in self.angles:
            # Get positions
            pos1 = positions[p1]
            pos2 = positions[p2]
            pos3 = positions[p3]
            
            # Calculate vectors from central particle to endpoints
            r21 = pos1 - pos2
            r23 = pos3 - pos2
            
            # Apply minimum image convention if using periodic boundaries
            if box_vectors is not None:
                for i in range(3):
                    if box_vectors[i, i] > 0:
                        while r21[i] > 0.5 * box_vectors[i, i]:
                            r21[i] -= box_vectors[i, i]
                        while r21[i] < -0.5 * box_vectors[i, i]:
                            r21[i] += box_vectors[i, i]
                        while r23[i] > 0.5 * box_vectors[i, i]:
                            r23[i] -= box_vectors[i, i]
                        while r23[i] < -0.5 * box_vectors[i, i]:
                            r23[i] += box_vectors[i, i]
            
            # Calculate vector magnitudes
            r21_mag = np.linalg.norm(r21)
            r23_mag = np.linalg.norm(r23)
            
            # Avoid division by zero
            if r21_mag < 1e-10 or r23_mag < 1e-10:
                continue
            
            # Calculate cosine of angle using dot product
            cos_theta = np.dot(r21, r23) / (r21_mag * r23_mag)
            
            # Clamp to valid range to avoid numerical issues
            cos_theta = max(-1.0, min(1.0, cos_theta))
            
            # Calculate angle
            theta = np.arccos(cos_theta)
            
            # Calculate force magnitude: dV/dθ = k * (θ - θ0)
            force_mag = k * (theta - theta0)
            
            # Calculate gradients and forces (using the chain rule)
            # This is a simplified implementation - a real-world one would
            # use a more numerically stable approach
            
            # Normalize vectors
            r21_norm = r21 / r21_mag
            r23_norm = r23 / r23_mag
            
            # Calculate gradients
            grad1 = (r23_norm - cos_theta * r21_norm) / (r21_mag * np.sqrt(1 - cos_theta**2))
            grad3 = (r21_norm - cos_theta * r23_norm) / (r23_mag * np.sqrt(1 - cos_theta**2))
            grad2 = -grad1 - grad3
            
            # Apply forces
            forces[p1] -= force_mag * grad1
            forces[p2] -= force_mag * grad2
            forces[p3] -= force_mag * grad3
            
            # Calculate potential energy: V = 0.5 * k * (θ - θ0)^2
            potential_energy += 0.5 * k * (theta - theta0)**2
        
        return forces, potential_energy
    
    def calculate_forces(self, positions, box_vectors=None):
        """
        Calculate angle forces (alias for calculate method for test compatibility).
        
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
        return self.calculate(positions, box_vectors)


class PeriodicTorsionForceTerm(ForceTerm):
    """
    Force term for periodic torsion (dihedral) potentials.
    
    The periodic torsion potential is defined as:
    V(φ) = k * (1 + cos(n * φ - δ))
    
    where k is the energy constant, n is the periodicity,
    φ is the dihedral angle, and δ is the phase shift.
    """
    
    def __init__(self, name: str = "PeriodicTorsion"):
        """
        Initialize a periodic torsion force term.
        
        Parameters
        ----------
        name : str, optional
            Name of the force term
        """
        super().__init__(name)
        self.torsions = []  # List of (p1, p2, p3, p4, k, n, delta) tuples
    
    def add_torsion(self, p1: int, p2: int, p3: int, p4: int, k: float, n: int, delta: float):
        """
        Add a torsion to the force term.
        
        Parameters
        ----------
        p1 : int
            Index of the first particle
        p2 : int
            Index of the second particle
        p3 : int
            Index of the third particle
        p4 : int
            Index of the fourth particle
        k : float
            Energy constant in kJ/mol
        n : int
            Periodicity (number of minima)
        delta : float
            Phase shift in radians
        """
        self.torsions.append((p1, p2, p3, p4, k, n, delta))
    
    def calculate(self, positions, box_vectors=None):
        """
        Calculate torsion forces.
        
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
        
        for p1, p2, p3, p4, k, n, delta in self.torsions:
            # Get positions
            pos1 = positions[p1]
            pos2 = positions[p2]
            pos3 = positions[p3]
            pos4 = positions[p4]
            
            # Calculate bond vectors
            r12 = pos2 - pos1
            r23 = pos3 - pos2
            r34 = pos4 - pos3
            
            # Apply minimum image convention if using periodic boundaries
            if box_vectors is not None:
                for i in range(3):
                    if box_vectors[i, i] > 0:
                        while r12[i] > 0.5 * box_vectors[i, i]:
                            r12[i] -= box_vectors[i, i]
                        while r12[i] < -0.5 * box_vectors[i, i]:
                            r12[i] += box_vectors[i, i]
                        # Repeat for r23 and r34
            
            # Calculate cross products to get the two planes
            cross1 = np.cross(r12, r23)
            cross2 = np.cross(r23, r34)
            
            # Normalize cross products
            norm1 = np.linalg.norm(cross1)
            norm2 = np.linalg.norm(cross2)
            
            # Avoid division by zero
            if norm1 < 1e-10 or norm2 < 1e-10:
                continue
            
            cross1 /= norm1
            cross2 /= norm2
            
            # Calculate cosine of dihedral angle
            cos_phi = np.dot(cross1, cross2)
            
            # Clamp to valid range to avoid numerical issues
            cos_phi = max(-1.0, min(1.0, cos_phi))
            
            # Calculate sine of dihedral angle to determine sign
            r23_norm = r23 / np.linalg.norm(r23)
            sin_phi = np.dot(np.cross(cross1, cross2), r23_norm)
            
            # Calculate dihedral angle in correct quadrant
            phi = np.arccos(cos_phi)
            if sin_phi < 0:
                phi = -phi
            
            # Calculate energy and forces (simplified implementation)
            # In a real implementation, we would compute gradients with respect to positions
            
            # Calculate potential energy: V = k * (1 + cos(n * φ - δ))
            potential_energy += k * (1.0 + np.cos(n * phi - delta))
            
            # Force magnitude: dV/dφ = -k * n * sin(n * φ - δ)
            force_mag = -k * n * np.sin(n * phi - delta)
            
            # In a real implementation, we would compute forces based on the gradients
            # of the dihedral angle with respect to the positions of the four particles
            # This is a complex calculation omitted here for brevity
        
        return forces, potential_energy


class LennardJonesForceTerm(ForceTerm):
    """
    Force term for Lennard-Jones potentials.
    
    The Lennard-Jones potential is defined as:
    V(r) = 4ε * [(σ/r)^12 - (σ/r)^6]
    
    where ε is the well depth, σ is the distance at which the potential is zero,
    and r is the distance between particles.
    """
    
    def __init__(self, sigma=None, epsilon=None, cutoff: float = 1.0, name: str = "LennardJones", switch_distance: Optional[float] = None):
        """
        Initialize a Lennard-Jones force term.
        
        Parameters
        ----------
        sigma : array-like, optional
            Distance parameters in nm for each particle
        epsilon : array-like, optional
            Energy parameters in kJ/mol for each particle
        cutoff : float, optional
            Cutoff distance in nm
        name : str, optional
            Name of the force term
        switch_distance : float, optional
            Distance at which to start switching the potential
        """
        super().__init__(name)
        self.cutoff = cutoff
        self.switch_distance = switch_distance if switch_distance is not None else 0.9 * cutoff
        self.particles = []  # List of (sigma, epsilon) tuples for each particle
        self.exclusions = set()  # Set of (i, j) tuples for excluded pairs
        self.scale_factors = {}  # Dict mapping (i, j) to scale factor
        
        # Support test-style constructor
        if sigma is not None and epsilon is not None:
            sigma = np.asarray(sigma)
            epsilon = np.asarray(epsilon)
            if sigma.shape != epsilon.shape:
                raise ValueError("sigma and epsilon must have the same shape")
            
            # Store for test compatibility
            self.sigma = sigma
            self.epsilon = epsilon
            
            # Add particles
            for s, e in zip(sigma, epsilon):
                self.add_particle(s, e)
    
    def add_particle(self, sigma: float, epsilon: float):
        """
        Add a particle to the force term.
        
        Parameters
        ----------
        sigma : float
            Distance parameter in nm
        epsilon : float
            Energy parameter in kJ/mol
        
        Returns
        -------
        int
            Index of the added particle
        """
        self.particles.append((sigma, epsilon))
        return len(self.particles) - 1
    
    def add_exclusion(self, particle1: int, particle2: int):
        """
        Add an exclusion between two particles.
        
        Parameters
        ----------
        particle1 : int
            Index of the first particle
        particle2 : int
            Index of the second particle
        """
        if particle1 == particle2:
            return
        i, j = min(particle1, particle2), max(particle1, particle2)
        self.exclusions.add((i, j))
    
    def set_scale_factor(self, particle1: int, particle2: int, scale_factor: float):
        """
        Set a scale factor for the interaction between two particles.
        
        Parameters
        ----------
        particle1 : int
            Index of the first particle
        particle2 : int
            Index of the second particle
        scale_factor : float
            Scale factor for the interaction (0 to 1)
        """
        if particle1 == particle2:
            return
        i, j = min(particle1, particle2), max(particle1, particle2)
        self.scale_factors[(i, j)] = scale_factor
    
    def calculate(self, positions, box_vectors=None):
        """
        Calculate Lennard-Jones forces.
        
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
        
        # Ensure we have parameters for all particles
        if len(self.particles) < n_particles:
            logger.warning(f"Not all particles have LJ parameters ({len(self.particles)} vs {n_particles})")
            return forces, potential_energy
        
        # Loop over all pairs of particles
        for i in range(n_particles):
            sigma_i, epsilon_i = self.particles[i]
            
            for j in range(i+1, n_particles):
                # Skip excluded pairs
                if (i, j) in self.exclusions:
                    continue
                
                # Get scale factor
                scale_factor = self.scale_factors.get((i, j), 1.0)
                if scale_factor == 0.0:
                    continue
                
                sigma_j, epsilon_j = self.particles[j]
                
                # Combine parameters using Lorentz-Berthelot mixing rules
                sigma = 0.5 * (sigma_i + sigma_j)
                epsilon = np.sqrt(epsilon_i * epsilon_j)
                
                # Scale parameters
                epsilon *= scale_factor
                
                # Calculate vector from i to j
                r_vec = positions[j] - positions[i]
                
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
                
                # Calculate Lennard-Jones terms
                sigma_over_r = sigma / r
                sigma_over_r6 = sigma_over_r**6
                sigma_over_r12 = sigma_over_r6**2
                
                # Calculate energy and force
                energy = 4.0 * epsilon * (sigma_over_r12 - sigma_over_r6)
                force_mag = 24.0 * epsilon / r * (2.0 * sigma_over_r12 - sigma_over_r6)
                
                # Apply switching function if needed
                if self.switch_distance is not None and r > self.switch_distance:
                    # Switching function: (r_cut^2 - r^2)^2 * (r_cut^2 + 2r^2 - 3r_switch^2) / (r_cut^2 - r_switch^2)^3
                    r2 = r**2
                    r_switch2 = self.switch_distance**2
                    r_cut2 = self.cutoff**2
                    
                    t = (r_cut2 - r2) / (r_cut2 - r_switch2)
                    t2 = t**2
                    switch = t2 * (3.0 - 2.0 * t)
                    switch_der = 6.0 * t * (1.0 - t) / (r_cut2 - r_switch2)
                    
                    energy *= switch
                    force_mag = switch * force_mag - energy * switch_der / r
                
                # Calculate force vector (force on particle i, pointing away from particle j for repulsion)
                force_vec = -force_mag * r_vec / r
                
                # Apply forces (Newton's third law)
                forces[i] += force_vec
                forces[j] -= force_vec
                
                # Add to potential energy
                potential_energy += energy
        
        return forces, potential_energy
    
    def calculate_forces(self, positions, box_vectors=None):
        """
        Calculate LJ forces (alias for calculate method for test compatibility).
        
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
        return self.calculate(positions, box_vectors)


class CoulombForceTerm(ForceTerm):
    """
    Force term for Coulomb electrostatic potentials.
    
    The Coulomb potential is defined as:
    V(r) = q_i * q_j / (4πε₀ * ε_r * r)
    
    where q_i and q_j are the charges of the particles,
    ε₀ is the vacuum permittivity, ε_r is the relative permittivity,
    and r is the distance between particles.
    """
    
    # Physical constants
    COULOMB_CONSTANT = 1389.35  # kJ/(mol*nm) * e^-2, where e is the elementary charge
    
    def __init__(self, charges=None, cutoff: float = 1.0, name: str = "Coulomb", relative_permittivity: float = 1.0):
        """
        Initialize a Coulomb force term.
        
        Parameters
        ----------
        charges : array-like, optional
            Particle charges in elementary charge units
        cutoff : float, optional
            Cutoff distance in nm
        name : str, optional
            Name of the force term
        relative_permittivity : float, optional
            Relative permittivity (dielectric constant)
        """
        super().__init__(name)
        self.cutoff = cutoff
        self.relative_permittivity = relative_permittivity
        self.particles = []  # List of charges for each particle
        self.exclusions = set()  # Set of (i, j) tuples for excluded pairs
        self.scale_factors = {}  # Dict mapping (i, j) to scale factor
        
        # Support test-style constructor
        if charges is not None:
            charges = np.asarray(charges)
            self.charges = charges  # Store for test compatibility
            
            # Add particles
            for charge in charges:
                self.add_particle(charge)
        else:
            self.charges = np.array([])  # Initialize empty array
    
    def add_particle(self, charge: float):
        """
        Add a particle to the force term.
        
        Parameters
        ----------
        charge : float
            Particle charge in elementary charge units
        
        Returns
        -------
        int
            Index of the added particle
        """
        self.particles.append(charge)
        return len(self.particles) - 1
    
    def add_exclusion(self, particle1: int, particle2: int):
        """
        Add an exclusion between two particles.
        
        Parameters
        ----------
        particle1 : int
            Index of the first particle
        particle2 : int
            Index of the second particle
        """
        if particle1 == particle2:
            return
        i, j = min(particle1, particle2), max(particle1, particle2)
        self.exclusions.add((i, j))
    
    def set_scale_factor(self, particle1: int, particle2: int, scale_factor: float):
        """
        Set a scale factor for the interaction between two particles.
        
        Parameters
        ----------
        particle1 : int
            Index of the first particle
        particle2 : int
            Index of the second particle
        scale_factor : float
            Scale factor for the interaction (0 to 1)
        """
        if particle1 == particle2:
            return
        i, j = min(particle1, particle2), max(particle1, particle2)
        self.scale_factors[(i, j)] = scale_factor
    
    def calculate(self, positions, box_vectors=None):
        """
        Calculate Coulomb forces.
        
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
        
        # Ensure we have charges for all particles
        if len(self.particles) < n_particles:
            logger.warning(f"Not all particles have charges ({len(self.particles)} vs {n_particles})")
            return forces, potential_energy
        
        # Calculate constant factor for Coulomb's law
        coulomb_factor = self.COULOMB_CONSTANT / self.relative_permittivity
        
        # Loop over all pairs of particles
        for i in range(n_particles):
            q_i = self.particles[i]
            
            # Skip particles with zero charge
            if abs(q_i) < 1e-10:
                continue
                
            for j in range(i+1, n_particles):
                # Skip excluded pairs
                if (i, j) in self.exclusions:
                    continue
                
                # Get scale factor
                scale_factor = self.scale_factors.get((i, j), 1.0)
                if scale_factor == 0.0:
                    continue
                
                q_j = self.particles[j]
                
                # Skip particles with zero charge
                if abs(q_j) < 1e-10:
                    continue
                
                # Calculate vector from i to j
                r_vec = positions[j] - positions[i]
                
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
                
                # Calculate Coulomb energy and force
                q_ij = q_i * q_j * scale_factor
                energy = coulomb_factor * q_ij / r
                force_mag = -coulomb_factor * q_ij / r**2  # Negative sign for attractive/repulsive direction
                
                # Calculate force vector (force on particle i due to particle j)
                force_vec = force_mag * r_vec / r
                
                # Apply forces (Newton's third law)
                forces[i] += force_vec
                forces[j] -= force_vec
                
                # Add to potential energy
                potential_energy += energy
        
        return forces, potential_energy
    
    def calculate_forces(self, positions, box_vectors=None):
        """
        Calculate Coulomb forces (alias for calculate method for test compatibility).
        
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
        return self.calculate(positions, box_vectors)


class PMEElectrostaticsForceTerm(ForceTerm):
    """
    Force term for Particle Mesh Ewald electrostatics.
    
    PME is an efficient method for calculating long-range electrostatic
    interactions in periodic systems by splitting the calculation into
    direct and reciprocal space components.
    """
    
    def __init__(self, name: str = "PME", cutoff: float = 1.0, relative_permittivity: float = 1.0, ewald_error_tolerance: float = 0.0001):
        """
        Initialize a PME electrostatics force term.
        
        Parameters
        ----------
        name : str, optional
            Name of the force term
        cutoff : float, optional
            Real-space cutoff distance in nm
        relative_permittivity : float, optional
            Relative permittivity (dielectric constant)
        ewald_error_tolerance : float, optional
            Error tolerance for Ewald summation
        """
        super().__init__(name)
        self.cutoff = cutoff
        self.relative_permittivity = relative_permittivity
        self.ewald_error_tolerance = ewald_error_tolerance
        
        # PME parameters
        self.alpha = 0.0  # Ewald spreading parameter
        self.grid_size = [0, 0, 0]  # Grid dimensions for reciprocal space
        
        # Particle data
        self.particles = []  # List of charges for each particle
        self.exclusions = set()  # Set of (i, j) tuples for excluded pairs
        self.scale_factors = {}  # Dict mapping (i, j) to scale factor
    
    def add_particle(self, charge: float):
        """
        Add a particle to the force term.
        
        Parameters
        ----------
        charge : float
            Particle charge in elementary charge units
        
        Returns
        -------
        int
            Index of the added particle
        """
        self.particles.append(charge)
        return len(self.particles) - 1
    
    def add_exclusion(self, particle1: int, particle2: int):
        """
        Add an exclusion between two particles.
        
        Parameters
        ----------
        particle1 : int
            Index of the first particle
        particle2 : int
            Index of the second particle
        """
        if particle1 == particle2:
            return
        i, j = min(particle1, particle2), max(particle1, particle2)
        self.exclusions.add((i, j))
    
    def set_scale_factor(self, particle1: int, particle2: int, scale_factor: float):
        """
        Set a scale factor for the interaction between two particles.
        
        Parameters
        ----------
        particle1 : int
            Index of the first particle
        particle2 : int
            Index of the second particle
        scale_factor : float
            Scale factor for the interaction (0 to 1)
        """
        if particle1 == particle2:
            return
        i, j = min(particle1, particle2), max(particle1, particle2)
        self.scale_factors[(i, j)] = scale_factor
    
    def initialize(self, box_vectors):
        """
        Initialize PME parameters based on the system.
        
        Parameters
        ----------
        box_vectors : np.ndarray
            Periodic box vectors
        """
        # Set alpha based on cutoff and error tolerance
        if box_vectors is None:
            raise ValueError("PME requires periodic boundary conditions")
            
        # Get box dimensions
        box_lengths = np.diagonal(box_vectors)
        
        # Calculate alpha based on error tolerance and cutoff
        self.alpha = np.sqrt(-np.log(2 * self.ewald_error_tolerance)) / self.cutoff
        
        # Calculate grid dimensions
        for i in range(3):
            if box_lengths[i] > 0:
                self.grid_size[i] = int(np.ceil(2 * self.alpha * box_lengths[i] / (3 * self.ewald_error_tolerance**(1/3))))
                # Ensure grid size is a multiple of 2 (makes FFT more efficient)
                self.grid_size[i] += self.grid_size[i] % 2
            else:
                self.grid_size[i] = 0
    
    def calculate(self, positions, box_vectors=None):
        """
        Calculate PME electrostatic forces.
        
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
        
        # Initialize PME parameters if needed
        if self.alpha == 0.0 and box_vectors is not None:
            self.initialize(box_vectors)
        
        # For a real PME implementation, we would need to:
        # 1. Calculate the direct space contributions
        # 2. Spread charges onto the reciprocal grid
        # 3. Solve Poisson's equation using FFT
        # 4. Differentiate the potential to get forces
        # 5. Apply forces to particles
        
        # This is a complex calculation that would require a full PME implementation
        # For now, just use a simple cutoff electrostatics as placeholder
        coulomb_term = CoulombForceTerm(
            name=f"{self.name}_Direct",
            cutoff=self.cutoff,
            relative_permittivity=self.relative_permittivity
        )
        
        # Copy particle data
        for charge in self.particles:
            coulomb_term.add_particle(charge)
            
        # Copy exclusions and scale factors
        coulomb_term.exclusions = self.exclusions.copy()
        coulomb_term.scale_factors = self.scale_factors.copy()
        
        # Calculate forces using cutoff method
        forces, potential_energy = coulomb_term.calculate(positions, box_vectors)
        
        return forces, potential_energy


class AmberForceField(ForceField):
    """
    Implementation of the AMBER force field.
    
    AMBER (Assisted Model Building with Energy Refinement) is a widely used
    force field for biomolecular simulations, particularly for proteins and nucleic acids.
    """
    
    def __init__(self, 
                 cutoff: float = 1.0,
                 switch_distance: Optional[float] = None,
                 nonbonded_method: NonbondedMethod = NonbondedMethod.PME,
                 use_long_range_correction: bool = True,
                 ewaldErrorTolerance: float = 0.0001,
                 variant: str = "ff14SB"):
        """
        Initialize the AMBER force field.
        
        Parameters
        ----------
        cutoff : float, optional
            Cutoff distance for nonbonded interactions in nm
        switch_distance : float, optional
            Distance at which to start switching the LJ potential
        nonbonded_method : NonbondedMethod, optional
            Method for handling nonbonded interactions
        use_long_range_correction : bool, optional
            Whether to use a long-range correction for LJ interactions
        ewaldErrorTolerance : float, optional
            Error tolerance for Ewald summation
        variant : str, optional
            AMBER force field variant to use
        """
        super().__init__(
            name=f"AMBER-{variant}",
            cutoff=cutoff,
            switch_distance=switch_distance,
            nonbonded_method=nonbonded_method,
            use_long_range_correction=use_long_range_correction,
            ewaldErrorTolerance=ewaldErrorTolerance
        )
        
        self.variant = variant
        self.residue_templates = {}  # Map residue name to template parameters
        
        # Load standard force field parameters
        amber_data_dir = os.path.join(os.path.dirname(__file__), "data", "amber")
        
        # In a real implementation, we would load parameters from files
        # Here we just set up the structure
        if os.path.exists(amber_data_dir):
            param_files = [
                os.path.join(amber_data_dir, f"{variant}.dat"),
                os.path.join(amber_data_dir, "amino12.lib"),
                os.path.join(amber_data_dir, "atomic_ions.lib")
            ]
            self.load_parameters(param_files)
        else:
            logger.warning(f"AMBER data directory not found: {amber_data_dir}")
    
    def _read_parameter_file(self, file_path: str):
        """
        Read an AMBER parameter file.
        
        Parameters
        ----------
        file_path : str
            Path to the parameter file
        """
        if not os.path.exists(file_path):
            logger.warning(f"Parameter file not found: {file_path}")
            return
            
        # In a real implementation, this would parse the file format
        # and populate the parameter data structures
        
        logger.info(f"Loaded parameters from {file_path}")
    
    def assign_parameters(self, topology):
        """
        Assign AMBER force field parameters to a molecular topology.
        
        Parameters
        ----------
        topology : molecular topology
            Molecular topology to assign parameters to
        """
        # In a real implementation, this would match atom types
        # and assign parameters based on the force field rules
        pass
    
    def _add_bonded_forces(self, system, topology):
        """
        Add bonded force terms to the system.
        
        Parameters
        ----------
        system : ForceFieldSystem
            System to add forces to
        topology : molecular topology
            Molecular topology
        """
        # Add harmonic bond force
        bond_force = HarmonicBondForceTerm()
        
        # Add bonds from topology with parameters from force field
        # Here we would load the actual bonds and parameters
        
        system.add_force_term(bond_force)
        
        # Add harmonic angle force
        angle_force = HarmonicAngleForceTerm()
        
        # Add angles from topology with parameters from force field
        # Here we would load the actual angles and parameters
        
        system.add_force_term(angle_force)
        
        # Add periodic torsion force
        torsion_force = PeriodicTorsionForceTerm()
        
        # Add dihedrals from topology with parameters from force field
        # Here we would load the actual dihedrals and parameters
        
        system.add_force_term(torsion_force)
    
    def _add_nonbonded_forces(self, system, topology, box_vectors):
        """
        Add nonbonded force terms to the system.
        
        Parameters
        ----------
        system : ForceFieldSystem
            System to add forces to
        topology : molecular topology
            Molecular topology
        box_vectors : np.ndarray, optional
            Periodic box vectors
        """
        # Add Lennard-Jones force
        lj_force = LennardJonesForceTerm(cutoff=self.cutoff, switch_distance=self.switch_distance)
        
        # Add particles with parameters from force field
        # Here we would load the actual particle parameters
        
        # Add exclusions and scale factors
        # In AMBER, 1-4 interactions are typically scaled by 0.5
        
        system.add_force_term(lj_force)
        
        # Add electrostatics force based on method
        if self.nonbonded_method == NonbondedMethod.PME:
            elec_force = PMEElectrostaticsForceTerm(
                cutoff=self.cutoff,
                ewald_error_tolerance=self.ewaldErrorTolerance
            )
        else:
            elec_force = CoulombForceTerm(cutoff=self.cutoff)
        
        # Add particles with charges
        # Here we would load the actual particle charges
        
        # Add exclusions and scale factors
        # In AMBER, 1-4 electrostatic interactions are typically scaled by 0.8333
        
        system.add_force_term(elec_force)


# More force field implementations can be added as needed: CHARMM, GROMOS, etc.
