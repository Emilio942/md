"""
Protein structure module for molecular dynamics simulations.

This module provides classes and functions for representing
protein structures, including residues, chains, and atoms.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set
import logging
from pathlib import Path
import os

# Configure logging
logger = logging.getLogger(__name__)

class Atom:
    """
    Represents an individual atom in a molecular system.
    """
    
    def __init__(self, 
                 atom_id: int,
                 atom_name: str,
                 element: str,
                 mass: float,
                 charge: float,
                 position: np.ndarray,
                 residue_id: int = -1,
                 chain_id: str = '',
                 b_factor: float = 0.0,
                 occupancy: float = 1.0):
        """
        Initialize an atom.
        
        Parameters
        ----------
        atom_id : int
            Atom identifier
        atom_name : str
            Atom name (e.g., 'CA', 'N', 'C')
        element : str
            Chemical element symbol
        mass : float
            Atomic mass in atomic mass units (u)
        charge : float
            Atomic charge in elementary charge units (e)
        position : np.ndarray
            3D coordinates in nanometers
        residue_id : int, optional
            Identifier of the residue this atom belongs to
        chain_id : str, optional
            Identifier of the chain this atom belongs to
        b_factor : float, optional
            Temperature factor (B-factor) from structure
        occupancy : float, optional
            Occupancy from structure (0 to 1)
        """
        self.atom_id = atom_id
        self.atom_name = atom_name
        self.element = element
        self.mass = mass
        self.charge = charge
        self.position = np.array(position, dtype=float)
        self.residue_id = residue_id
        self.chain_id = chain_id
        self.b_factor = b_factor
        self.occupancy = occupancy
        
        # Connectivity information
        self.bonded_atoms = set()  # Set of atom IDs this atom is bonded to
        
        # Runtime properties (added during simulation)
        self.velocity = np.zeros(3, dtype=float)
        self.force = np.zeros(3, dtype=float)
    
    def add_bond(self, atom_id: int):
        """
        Add a bond to another atom.
        
        Parameters
        ----------
        atom_id : int
            ID of atom to bond with
        """
        if atom_id != self.atom_id:            
            self.bonded_atoms.add(atom_id)
    
    def is_bonded_to(self, atom_id: int) -> bool:
        """
        Check if this atom is bonded to another atom.
        
        Parameters
        ----------
        atom_id : int
            ID of atom to check
            
        Returns
        -------
        bool
            True if atoms are bonded, False otherwise
        """
        return atom_id in self.bonded_atoms
    
    def distance_to(self, other_atom) -> float:
        """
        Calculate distance to another atom.
        
        Parameters
        ----------
        other_atom : Atom
            Other atom to measure distance to
            
        Returns
        -------
        float
            Distance in nanometers
        """
        return np.linalg.norm(self.position - other_atom.position)
    
    def __repr__(self) -> str:
        return f"Atom(id={self.atom_id}, name='{self.atom_name}', element='{self.element}', res_id={self.residue_id}, chain='{self.chain_id}')"


class Residue:
    """
    Represents an amino acid residue or other molecular residue.
    """
    
    # Standard amino acid 3-letter to 1-letter code mapping
    THREE_TO_ONE = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 
        'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 
        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 
        'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 
        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'HSD': 'H', 'HSE': 'H', 'HSP': 'H'  # Common histidine variants
    }
    
    def __init__(self, 
                 residue_id: Union[int, str],
                 residue_name: str,
                 chain_id: str = '',
                 insertion_code: str = '',
                 is_hetero: bool = False,
                 one_letter_code: Optional[str] = None):
        """
        Initialize a residue.
        
        Parameters
        ----------
        residue_id : int or str
            Residue number, optionally including insertion code
        residue_name : str
            Three-letter residue code (e.g., 'ALA', 'LYS')
        chain_id : str, optional
            Chain identifier
        insertion_code : str, optional
            PDB insertion code
        is_hetero : bool, optional
            Flag for non-standard residues
        one_letter_code : str, optional
            One-letter amino acid code, automatically determined if None
        """
        self.residue_id = residue_id
        self.residue_name = residue_name.upper()
        self.chain_id = chain_id
        self.insertion_code = insertion_code
        self.is_hetero = is_hetero
        
        # Store atoms in this residue
        self.atoms = {}  # Dict mapping atom_name to Atom object
        
        # Get one-letter code if it's a standard amino acid
        if one_letter_code is not None:
            self.one_letter_code = one_letter_code
        else:
            self.one_letter_code = self.THREE_TO_ONE.get(self.residue_name, 'X')
        
        # Secondary structure assignment (H=helix, E=sheet, C=coil)
        self.secondary_structure = 'C'  # Default to coil
        
        # Additional metadata
        self.metadata = {}
    
    def add_atom(self, atom: Atom):
        """
        Add an atom to this residue.
        
        Parameters
        ----------
        atom : Atom
            Atom to add
        """
        self.atoms[atom.atom_name] = atom
        atom.residue_id = self.residue_id
        atom.chain_id = self.chain_id
    
    def get_atom(self, atom_name: str) -> Optional[Atom]:
        """
        Get an atom by name.
        
        Parameters
        ----------
        atom_name : str
            Name of the atom to retrieve
            
        Returns
        -------
        Atom or None
            Atom object if found, otherwise None
        """
        return self.atoms.get(atom_name)
    
    def get_backbone_atoms(self) -> Dict[str, Atom]:
        """
        Get backbone atoms (N, CA, C, O) of this residue.
        
        Returns
        -------
        dict
            Dictionary mapping atom names to Atom objects
        """
        backbone = {}
        for name in ('N', 'CA', 'C', 'O'):
            atom = self.atoms.get(name)
            if atom:
                backbone[name] = atom
        return backbone
    
    def get_center_of_mass(self) -> np.ndarray:
        """
        Calculate center of mass of the residue.
        
        Returns
        -------
        np.ndarray
            Center of mass coordinates
        """
        if not self.atoms:
            return np.zeros(3)
            
        total_mass = 0.0
        weighted_sum = np.zeros(3)
        
        for atom in self.atoms.values():
            weighted_sum += atom.position * atom.mass
            total_mass += atom.mass
            
        return weighted_sum / total_mass if total_mass > 0 else np.zeros(3)
    
    def get_atoms_array(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get arrays of positions, masses, and charges for all atoms.
        
        Returns
        -------
        tuple
            Tuple of (positions, masses, charges) arrays
        """
        n_atoms = len(self.atoms)
        positions = np.zeros((n_atoms, 3))
        masses = np.zeros(n_atoms)
        charges = np.zeros(n_atoms)
        
        for i, atom in enumerate(self.atoms.values()):
            positions[i] = atom.position
            masses[i] = atom.mass
            charges[i] = atom.charge
            
        return positions, masses, charges
    
    def __repr__(self) -> str:
        return f"Residue(id={self.residue_id}, name='{self.residue_name}', chain='{self.chain_id}', atoms={len(self.atoms)})"


class Chain:
    """
    Represents a polypeptide chain or other molecular chain.
    """
    
    def __init__(self, chain_id: str = ''):
        """
        Initialize a chain.
        
        Parameters
        ----------
        chain_id : str, optional
            Chain identifier
        """
        self.chain_id = chain_id
        self.residues = {}  # Dict mapping residue_id to Residue object
    
    def add_residue(self, residue: Residue):
        """
        Add a residue to this chain.
        
        Parameters
        ----------
        residue : Residue
            Residue to add
        """
        self.residues[residue.residue_id] = residue
        residue.chain_id = self.chain_id
    
    def get_residue(self, residue_id: int) -> Optional[Residue]:
        """
        Get a residue by ID.
        
        Parameters
        ----------
        residue_id : int
            ID of the residue to retrieve
            
        Returns
        -------
        Residue or None
            Residue object if found, otherwise None
        """
        return self.residues.get(residue_id)
    
    def get_sequence(self) -> str:
        """
        Get the amino acid sequence of this chain.
        
        Returns
        -------
        str
            One-letter amino acid sequence
        """
        # Sort residues by ID to ensure correct order
        sorted_residues = sorted(self.residues.values(), key=lambda r: r.residue_id)
        return ''.join(r.one_letter_code for r in sorted_residues)
    
    def get_atoms(self) -> List[Atom]:
        """
        Get all atoms in this chain.
        
        Returns
        -------
        list
            List of all Atom objects in this chain
        """
        atoms = []
        for residue in self.residues.values():
            atoms.extend(residue.atoms.values())
        return atoms
    
    def __len__(self) -> int:
        """
        Get number of residues in chain.
        
        Returns
        -------
        int
            Number of residues
        """
        return len(self.residues)
    
    def __repr__(self) -> str:
        return f"Chain(id='{self.chain_id}', residues={len(self.residues)})"


class Protein:
    """
    Represents a protein structure with multiple chains.
    """
    
    def __init__(self, protein_id: str = '', name: str = ''):
        """
        Initialize a protein.
        
        Parameters
        ----------
        protein_id : str, optional
            Protein identifier (e.g., PDB ID)
        name : str, optional
            Protein name or description
        """
        self.protein_id = protein_id
        self.name = name if name else protein_id
        self.chains = {}  # Dict mapping chain_id to Chain object
        self.metadata = {}  # Dict storing metadata
        
        # Track atoms for easy access
        self._atoms = {}  # Dict mapping atom_id to Atom object
        self._atom_count = 0
        
        # Track sequence information
        self.sequence = {}  # Dict mapping chain_id to sequence
        
        # Properties
        self.center_of_mass = None
        self.bounding_box = None
    
    def add_chain(self, chain: Chain):
        """
        Add a chain to this protein.
        
        Parameters
        ----------
        chain : Chain
            Chain to add
        """
        self.chains[chain.chain_id] = chain
        
        # Add atoms to the atom list
        for atom in chain.get_atoms():
            self._atoms[atom.atom_id] = atom
            self._atom_count += 1
        
        # Update sequence dictionary
        self.sequence[chain.chain_id] = chain.get_sequence()
    
    def get_chain(self, chain_id: str) -> Optional[Chain]:
        """
        Get a chain by ID.
        
        Parameters
        ----------
        chain_id : str
            ID of the chain to retrieve
            
        Returns
        -------
        Chain or None
            Chain object if found, otherwise None
        """
        return self.chains.get(chain_id)
    
    def get_atom(self, atom_id: int) -> Optional[Atom]:
        """
        Get an atom by ID.
        
        Parameters
        ----------
        atom_id : int
            ID of the atom to retrieve
            
        Returns
        -------
        Atom or None
            Atom object if found, otherwise None
        """
        return self._atoms.get(atom_id)
    
    def get_positions(self) -> np.ndarray:
        """
        Get positions of all atoms.
        
        Returns
        -------
        np.ndarray
            Array of positions with shape (n_atoms, 3)
        """
        positions = np.zeros((len(self._atoms), 3))
        for i, atom in enumerate(self._atoms.values()):
            positions[i] = atom.position
        return positions
    
    def set_positions(self, positions: np.ndarray):
        """
        Set positions of all atoms.
        
        Parameters
        ----------
        positions : np.ndarray
            Array of positions with shape (n_atoms, 3)
        """
        if positions.shape[0] != len(self._atoms):
            raise ValueError(f"Position array has wrong shape: expected {len(self._atoms)} atoms, got {positions.shape[0]}")
            
        for i, atom_id in enumerate(self._atoms):
            self._atoms[atom_id].position = positions[i]
    
    def get_center_of_mass(self) -> np.ndarray:
        """
        Calculate center of mass of the protein.
        
        Returns
        -------
        np.ndarray
            Center of mass coordinates
        """
        if not self._atoms:
            return np.zeros(3)
            
        total_mass = 0.0
        weighted_sum = np.zeros(3)
        
        for atom in self._atoms.values():
            weighted_sum += atom.position * atom.mass
            total_mass += atom.mass
            
        return weighted_sum / total_mass if total_mass > 0 else np.zeros(3)
    
    def get_atoms_array(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
        """
        Get arrays of positions, masses, charges, and IDs for all atoms.
        
        Returns
        -------
        tuple
            Tuple of (positions, masses, charges, atom_ids, chain_ids)
        """
        n_atoms = len(self._atoms)
        positions = np.zeros((n_atoms, 3))
        masses = np.zeros(n_atoms)
        charges = np.zeros(n_atoms)
        atom_ids = []
        chain_ids = []
        
        for i, (atom_id, atom) in enumerate(self._atoms.items()):
            positions[i] = atom.position
            masses[i] = atom.mass
            charges[i] = atom.charge
            atom_ids.append(atom_id)
            chain_ids.append(atom.chain_id)
            
        return positions, masses, charges, atom_ids, chain_ids
    
    def get_bonds(self) -> List[Tuple[int, int]]:
        """
        Get all bonds in the protein.
        
        Returns
        -------
        list
            List of (atom_id1, atom_id2) tuples representing bonds
        """
        bonds = []
        for atom_id, atom in self._atoms.items():
            for bonded_id in atom.bonded_atoms:
                # Add bond only once (by ensuring atom_id < bonded_id)
                if atom_id < bonded_id:
                    bonds.append((atom_id, bonded_id))
        return bonds
    
    def translate(self, vector: np.ndarray):
        """
        Translate all atoms by a vector.
        
        Parameters
        ----------
        vector : np.ndarray
            3D translation vector
        """
        for atom in self._atoms.values():
            atom.position += vector
    
    def rotate(self, rotation_matrix: np.ndarray):
        """
        Rotate all atoms using a rotation matrix.
        
        Parameters
        ----------
        rotation_matrix : np.ndarray
            3x3 rotation matrix
        """
        for atom in self._atoms.values():
            atom.position = np.dot(rotation_matrix, atom.position)
    
    def calculate_properties(self):
        """
        Calculate various properties of the protein structure.
        
        This method computes various structural properties such as:
        - Secondary structure assignments
        - Solvent accessible surface area
        - Hydrogen bond networks
        - Center of mass
        - Radius of gyration
        
        This should be called after the protein structure is fully loaded.
        """
        logger.info(f"Calculating properties for protein {self.protein_id}")
        
        # Calculate center of mass
        self.center_of_mass = self.get_center_of_mass()
        
        # Calculate radius of gyration
        total_mass = 0.0
        squared_distances_sum = 0.0
        
        for atom in self._atoms.values():
            squared_distances_sum += atom.mass * np.sum((atom.position - self.center_of_mass)**2)
            total_mass += atom.mass
            
        self.radius_of_gyration = np.sqrt(squared_distances_sum / total_mass) if total_mass > 0 else 0.0
        
        # Analyze bond network
        self.analyze_bond_network()
        
        logger.info(f"Protein properties calculated: {len(self._atoms)} atoms, " 
                   f"COM: {self.center_of_mass}, Rg: {self.radius_of_gyration:.3f} nm")
    
    def analyze_bond_network(self):
        """
        Analyze and set up the bond network based on distance criteria.
        
        This is a simple distance-based approach to identify bonds.
        For more accurate bonding information, a proper force field
        or connectivity table should be used.
        """
        logger.info("Analyzing bond network...")
        
        # Simple distance-based approach to find bonds
        # This is a naive approach and should be replaced with proper bonding rules
        bond_distance_threshold = 0.2  # nm
        
        positions = {}
        for atom_id, atom in self._atoms.items():
            positions[atom_id] = atom.position
        
        # Find bonds based on distance
        bonds_found = 0
        
        for atom_id1, pos1 in positions.items():
            atom1 = self._atoms[atom_id1]
            element1 = atom1.element.upper() if atom1.element else ''
            
            for atom_id2, pos2 in positions.items():
                if atom_id1 >= atom_id2:
                    continue  # Avoid duplicate bonds and self-bonds
                
                atom2 = self._atoms[atom_id2]
                element2 = atom2.element.upper() if atom2.element else ''
                
                # Skip if both atoms are in different chains or residues that are far apart
                if (atom1.chain_id != atom2.chain_id or 
                    abs(atom1.residue_id - atom2.residue_id) > 1):
                    # Exception for disulfide bonds (S-S)
                    if not (element1 == 'S' and element2 == 'S'):
                        continue
                
                # Calculate distance
                distance = np.linalg.norm(pos1 - pos2)
                
                # Adjust bond threshold based on atoms involved
                adjusted_threshold = bond_distance_threshold
                
                # Hydrogen bonds are shorter
                if element1 == 'H' or element2 == 'H':
                    adjusted_threshold = 0.12
                
                # Carbon-Carbon bonds
                elif element1 == 'C' and element2 == 'C':
                    adjusted_threshold = 0.16
                
                # Carbon-Nitrogen bonds
                elif (element1 == 'C' and element2 == 'N') or (element1 == 'N' and element2 == 'C'):
                    adjusted_threshold = 0.15
                
                # Carbon-Oxygen bonds
                elif (element1 == 'C' and element2 == 'O') or (element1 == 'O' and element2 == 'C'):
                    adjusted_threshold = 0.14
                
                # Disulfide bonds
                elif element1 == 'S' and element2 == 'S':
                    adjusted_threshold = 0.21
                
                # Create bond if distance is within threshold
                if distance < adjusted_threshold:
                    atom1.add_bond(atom_id2)
                    atom2.add_bond(atom_id1)
                    bonds_found += 1
        
        logger.info(f"Bond network analysis complete: {bonds_found} bonds found")
    
    def __len__(self) -> int:
        """
        Get number of atoms in protein.
        
        Returns
        -------
        int
            Number of atoms
        """
        return len(self._atoms)
    
    @property
    def atoms(self) -> Dict[int, Atom]:
        """
        Get dictionary of all atoms in the protein.
        
        Returns
        -------
        dict
            Dictionary mapping atom_id to Atom object
        """
        return self._atoms
    
    def add_atom(self, atom: Atom):
        """
        Add an atom directly to the protein.
        
        This is a convenience method that automatically creates
        chains and residues as needed.
        
        Parameters
        ----------
        atom : Atom
            Atom to add
        """
        # Get or create chain
        if atom.chain_id not in self.chains:
            self.chains[atom.chain_id] = Chain(atom.chain_id)
        
        chain = self.chains[atom.chain_id]
        
        # Get or create residue
        if atom.residue_id not in chain.residues:
            # Create residue with placeholder name
            residue_name = getattr(atom, 'residue_name', 'UNK')
            chain.residues[atom.residue_id] = Residue(
                residue_id=atom.residue_id,
                residue_name=residue_name,
                chain_id=atom.chain_id
            )
        
        residue = chain.residues[atom.residue_id]
        
        # Add atom to residue and update internal tracking
        residue.add_atom(atom)
        self._atoms[atom.atom_id] = atom
        self._atom_count += 1
    
    def add_residue(self, residue: Residue):
        """
        Add a residue to the protein.
        
        This is a convenience method that automatically creates
        chains as needed.
        
        Parameters
        ----------
        residue : Residue
            Residue to add
        """
        # Get or create chain
        if residue.chain_id not in self.chains:
            self.chains[residue.chain_id] = Chain(residue.chain_id)
        
        chain = self.chains[residue.chain_id]
        chain.add_residue(residue)
        
        # Update internal atom tracking
        for atom in residue.atoms.values():
            self._atoms[atom.atom_id] = atom
            self._atom_count += 1
        
        # Update sequence
        self.sequence[residue.chain_id] = chain.get_sequence()
    
    def get_residue_by_id(self, residue_id: int) -> Optional[Residue]:
        """
        Get a residue by ID, searching all chains.
        
        Parameters
        ----------
        residue_id : int
            ID of the residue to retrieve
            
        Returns
        -------
        Residue or None
            Residue object if found, otherwise None
        """
        for chain in self.chains.values():
            if residue_id in chain.residues:
                return chain.residues[residue_id]
        return None

    def __repr__(self) -> str:
        """
        Get string representation of protein.
        
        Returns
        -------
        str
            String representation
        """
        return f"Protein(id='{self.protein_id}', chains={len(self.chains)}, atoms={len(self._atoms)})"


class ProteinReader:
    """
    Base class for protein structure readers.
    """
    
    @staticmethod
    def read_file(filename: str) -> Protein:
        """
        Read a protein structure file.
        
        Parameters
        ----------
        filename : str
            Path to the structure file
        
        Returns
        -------
        Protein
            Protein object constructed from the file
        """
        raise NotImplementedError("Subclasses must implement this method")


class PDBReader(ProteinReader):
    """
    Reader for Protein Data Bank (PDB) format files.
    """
    
    # Atomic mass lookup table (simplified)
    ELEMENT_MASSES = {
        'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'S': 32.06,
        'P': 30.974, 'FE': 55.845, 'CA': 40.078, 'ZN': 65.38, 'NA': 22.99,
        'K': 39.098, 'MG': 24.305, 'CL': 35.45, 'F': 18.998, 'BR': 79.904,
        'I': 126.90
    }
    
    @staticmethod
    def read_file(filename: str) -> Protein:
        """
        Read a PDB file and create a Protein object.
        
        Parameters
        ----------
        filename : str
            Path to the PDB file
        
        Returns
        -------
        Protein
            Protein object constructed from the PDB file
        """
        protein = Protein(name=Path(filename).stem)
        
        current_chain = None
        current_residue = None
        atom_index = 0
        
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    is_hetero = line.startswith('HETATM')
                    
                    # Parse PDB atom line
                    atom_id = int(line[6:11])
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    chain_id = line[21:22].strip()
                    residue_id = int(line[22:26])
                    insertion_code = line[26:27].strip()
                    
                    # 3D coordinates in Angstroms (convert to nm)
                    x = float(line[30:38]) / 10.0
                    y = float(line[38:46]) / 10.0
                    z = float(line[46:54]) / 10.0
                    position = np.array([x, y, z])
                    
                    # Optional fields
                    try:
                        occupancy = float(line[54:60])
                    except ValueError:
                        occupancy = 1.0
                        
                    try:
                        b_factor = float(line[60:66])
                    except ValueError:
                        b_factor = 0.0
                    
                    # Element symbol
                    if len(line) >= 78:
                        element = line[76:78].strip().upper()
                    else:
                        # Guess element from atom name
                        element = ''.join(c for c in atom_name if c.isalpha()).upper()
                        if not element:
                            element = 'C'  # Default to carbon
                    
                    # Get mass from element
                    mass = PDBReader.ELEMENT_MASSES.get(element, 12.0)
                    
                    # For now, all atoms get zero charge (would be determined by force field)
                    charge = 0.0
                    
                    # Create atom
                    atom = Atom(
                        atom_id=atom_index,
                        atom_name=atom_name,
                        element=element,
                        mass=mass,
                        charge=charge,
                        position=position,
                        residue_id=residue_id,
                        chain_id=chain_id,
                        b_factor=b_factor,
                        occupancy=occupancy
                    )
                    
                    # Add atom to residue/chain structure
                    if current_chain is None or current_chain.chain_id != chain_id:
                        current_chain = Chain(chain_id)
                        protein.add_chain(current_chain)
                    
                    if current_residue is None or current_residue.residue_id != residue_id or current_residue.chain_id != chain_id:
                        current_residue = Residue(
                            residue_id=residue_id,
                            name=residue_name,
                            chain_id=chain_id,
                            insertion_code=insertion_code,
                            is_hetero=is_hetero
                        )
                        current_chain.add_residue(current_residue)
                    
                    current_residue.add_atom(atom)
                    protein._atoms[atom_index] = atom
                    atom_index += 1
                
                elif line.startswith('CONECT'):
                    # Parse connectivity records
                    fields = line.strip().split()
                    if len(fields) > 2:
                        atom_id = int(fields[1]) - 1  # PDB atom IDs start at 1
                        
                        # Add bonds to other atoms
                        for i in range(2, len(fields)):
                            bonded_atom_id = int(fields[i]) - 1  # PDB atom IDs start at 1
                            if atom_id in protein._atoms and bonded_atom_id in protein._atoms:
                                protein._atoms[atom_id].add_bond(bonded_atom_id)
                                protein._atoms[bonded_atom_id].add_bond(atom_id)
                
                elif line.startswith('HEADER'):
                    # Extract structure title
                    if len(line) > 10:
                        protein.header['title'] = line[10:].strip()
                    # Extract structure information
                    protein.metadata['pdb_id'] = line[62:66].strip()
                    protein.metadata['classification'] = line[10:50].strip()
                    
                    # Try to extract date if available
                    if len(line) >= 50:
                        try:
                            protein.metadata['deposition_date'] = line[50:59].strip()
                        except:
                            pass
                
                elif line.startswith('TITLE'):
                    # Extract title
                    if 'title' not in protein.header:
                        protein.metadata['title'] = line[10:].strip()
                    else:
                        protein.metadata['title'] += ' ' + line[10:].strip()
        
        # Infer bonds if not specified in CONECT records
        if not protein.get_bonds():
            PDBReader._infer_bonds(protein)
        
        protein._atom_count = atom_index
        return protein
    
    @staticmethod
    def _infer_bonds(protein: Protein):
        """
        Infer covalent bonds based on distance between atoms.
        
        Parameters
        ----------
        protein : Protein
            Protein to infer bonds for
        """
        # Bond distance thresholds by element pair (in nm)
        bond_thresholds = {
            ('C', 'C'): 0.154,  # C-C bond
            ('C', 'N'): 0.147,  # C-N bond
            ('C', 'O'): 0.143,  # C-O bond
            ('C', 'S'): 0.182,  # C-S bond
            ('C', 'H'): 0.109,  # C-H bond
            ('N', 'N'): 0.145,  # N-N bond
            ('N', 'H'): 0.101,  # N-H bond
            ('O', 'H'): 0.096,  # O-H bond
            ('S', 'H'): 0.134,  # S-H bond
        }
        
        # Margin for bond detection
        margin = 0.035  # nm
        
        # Add common backbone bonds first
        for chain in protein.chains.values():
            residues = sorted(chain.residues.values(), key=lambda r: r.residue_id)
            
            for i, res in enumerate(residues):
                # Connect backbone atoms within residue
                backbone = res.get_backbone_atoms()
                
                if 'N' in backbone and 'CA' in backbone:
                    backbone['N'].add_bond(backbone['CA'].atom_id)
                    backbone['CA'].add_bond(backbone['N'].atom_id)
                
                if 'CA' in backbone and 'C' in backbone:
                    backbone['CA'].add_bond(backbone['C'].atom_id)
                    backbone['C'].add_bond(backbone['CA'].atom_id)
                
                if 'C' in backbone and 'O' in backbone:
                    backbone['C'].add_bond(backbone['O'].atom_id)
                    backbone['O'].add_bond(backbone['C'].atom_id)
                
                # Connect to next residue if it exists
                if i < len(residues) - 1:
                    next_res = residues[i+1]
                    next_backbone = next_res.get_backbone_atoms()
                    
                    # Peptide bond between C and N of next residue
                    if 'C' in backbone and 'N' in next_backbone:
                        backbone['C'].add_bond(next_backbone['N'].atom_id)
                        next_backbone['N'].add_bond(backbone['C'].atom_id)
        
        # For side chains and other atoms, use distance criteria
        atom_list = list(protein._atoms.values())
        n_atoms = len(atom_list)
        
        for i in range(n_atoms):
            atom_i = atom_list[i]
            element_i = atom_i.element.upper()
            
            for j in range(i+1, n_atoms):
                atom_j = atom_list[j]
                element_j = atom_j.element.upper()
                
                # Skip if atoms are in different chains or too far apart in sequence
                if atom_i.chain_id != atom_j.chain_id:
                    continue
                    
                if abs(atom_i.residue_id - atom_j.residue_id) > 1:
                    # Make an exception for disulfide bonds
                    if not (element_i == 'S' and element_j == 'S' and 
                           atom_i.atom_name == 'SG' and atom_j.atom_name == 'SG'):
                        continue
                
                # Get bond threshold for this pair
                element_pair = (element_i, element_j)
                reverse_pair = (element_j, element_i)
                
                # Find the appropriate threshold
                if element_pair in bond_thresholds:
                    threshold = bond_thresholds[element_pair] + margin
                elif reverse_pair in bond_thresholds:
                    threshold = bond_thresholds[reverse_pair] + margin
                else:
                    # Default threshold
                    threshold = 0.18  # nm, with margin
                
                # Calculate distance
                distance = np.linalg.norm(atom_i.position - atom_j.position)
                
                # Add bond if within threshold
                if distance <= threshold:
                    atom_i.add_bond(atom_j.atom_id)
                    atom_j.add_bond(atom_i.atom_id)


class Membrane:
    """
    Represents a lipid bilayer membrane for cellular simulations.
    """
    
    def __init__(self, 
                 x_dim: float,  # nm
                 y_dim: float,  # nm
                 lipid_type: str = 'POPC',
                 thickness: float = 4.0,  # nm
                 area_per_lipid: float = 0.65):  # nm²
        """
        Initialize a lipid bilayer membrane.
        
        Parameters
        ----------
        x_dim : float
            X dimension of membrane in nanometers
        y_dim : float
            Y dimension of membrane in nanometers
        lipid_type : str, optional
            Type of lipid molecule (e.g., 'POPC', 'DPPC')
        thickness : float, optional
            Membrane thickness in nanometers
        area_per_lipid : float, optional
            Area per lipid molecule in nm²
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.lipid_type = lipid_type
        self.thickness = thickness
        self.area_per_lipid = area_per_lipid
        
        # Membrane center
        self.center = np.array([0.0, 0.0, 0.0])
        
        # Lipid positions
        self.upper_leaflet = []  # List of lipid positions in upper leaflet
        self.lower_leaflet = []  # List of lipid positions in lower leaflet
        
        # Generate the membrane structure
        self.generate_membrane()
    
    def generate_membrane(self):
        """
        Generate a lipid bilayer membrane structure.
        
        This creates a simplified model of a membrane with lipids
        positioned in a grid pattern in both leaflets.
        """
        # Calculate number of lipids in each dimension
        num_x = int(self.x_dim / np.sqrt(self.area_per_lipid))
        num_y = int(self.y_dim / np.sqrt(self.area_per_lipid))
        
        # Total number of lipids in each leaflet
        num_lipids = num_x * num_y
        
        # Spacing between lipids
        dx = self.x_dim / num_x
        dy = self.y_dim / num_y
        
        # Z positions of the leaflets
        z_upper = self.thickness / 2
        z_lower = -self.thickness / 2
        
        # Generate grid positions for upper leaflet
        self.upper_leaflet = []
        for i in range(num_x):
            for j in range(num_y):
                # Add some random perturbation (10% of spacing)
                perturb_x = np.random.uniform(-0.1 * dx, 0.1 * dx)
                perturb_y = np.random.uniform(-0.1 * dy, 0.1 * dy)
                
                x = (i + 0.5) * dx + perturb_x - self.x_dim / 2
                y = (j + 0.5) * dy + perturb_y - self.y_dim / 2
                
                self.upper_leaflet.append(np.array([x, y, z_upper]))
        
        # Generate grid positions for lower leaflet
        self.lower_leaflet = []
        for i in range(num_x):
            for j in range(num_y):
                # Add some random perturbation (10% of spacing)
                perturb_x = np.random.uniform(-0.1 * dx, 0.1 * dx)
                perturb_y = np.random.uniform(-0.1 * dy, 0.1 * dy)
                
                x = (i + 0.5) * dx + perturb_x - self.x_dim / 2
                y = (j + 0.5) * dy + perturb_y - self.y_dim / 2
                
                self.lower_leaflet.append(np.array([x, y, z_lower]))
        
        logger.info(f"Generated membrane: {self.lipid_type}, {len(self.upper_leaflet)} lipids per leaflet, "
                   f"dimensions: {self.x_dim} x {self.y_dim} nm")
    
    def get_lipid_positions(self):
        """
        Get positions of all lipids in the membrane.
        
        Returns
        -------
        tuple
            Tuple of upper and lower leaflet positions arrays
        """
        return np.array(self.upper_leaflet), np.array(self.lower_leaflet)
    
    def translate(self, vector):
        """
        Translate the membrane by a vector.
        
        Parameters
        ----------
        vector : np.ndarray
            Translation vector
        """
        self.center += vector
        
        # Translate all lipids
        for i in range(len(self.upper_leaflet)):
            self.upper_leaflet[i] += vector
            
        for i in range(len(self.lower_leaflet)):
            self.lower_leaflet[i] += vector
    
    def contains_point(self, point):
        """
        Check if a point is inside the membrane.
        
        Parameters
        ----------
        point : np.ndarray
            3D point coordinates
            
        Returns
        -------
        bool
            True if point is inside membrane, False otherwise
        """
        # Extract x, y, z coordinates
        x, y, z = point
        
        # Check if point is within x-y dimensions
        if (abs(x - self.center[0]) > self.x_dim / 2 or 
            abs(y - self.center[1]) > self.y_dim / 2):
            return False
        
        # Check if point is within z thickness
        return abs(z - self.center[2]) < self.thickness / 2
    
    def get_normal_vector(self, point):
        """
        Get the membrane normal vector at a given point.
        
        Parameters
        ----------
        point : np.ndarray
            3D point coordinates
            
        Returns
        -------
        np.ndarray
            Unit normal vector (typically [0, 0, 1] for flat membranes)
        """
        # For a flat membrane, normal is always in z direction
        return np.array([0, 0, 1])
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the bounding box of the membrane.
        
        Returns
        -------
        tuple
            Tuple of (min_coords, max_coords)
        """
        half_thickness = self.thickness / 2
        return (
            np.array([0, 0, -half_thickness]),
            np.array([self.x_dim, self.y_dim, half_thickness])
        )
    
    def __repr__(self) -> str:
        return f"Membrane(type='{self.lipid_type}', size={self.x_dim}x{self.y_dim} nm, lipids={2*self.n_lipids_per_leaflet})"


# A class for representing cellular environments will be implemented in the environment module
