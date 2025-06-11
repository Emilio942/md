"""
Protein structure module for molecular dynamics simulations.

This module provides classes and functions for handling protein structures,
including loading from PDB files and generating topologies.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import os
from ..core import Particle

# Set up logging
logger = logging.getLogger(__name__)

class Atom:
    """
    Class representing an atom in a protein or other molecule.
    
    This extends the basic Particle class with additional properties
    specific to molecular structures.
    """
    
    def __init__(self, 
                 atom_id: int,
                 name: str,
                 element: str,
                 residue_name: str,
                 residue_id: int,
                 chain_id: str,
                 mass: float,
                 charge: float,
                 position: np.ndarray,
                 velocity: Optional[np.ndarray] = None):
        """
        Initialize an atom.
        
        Parameters
        ----------
        atom_id : int
            Unique identifier for the atom
        name : str
            Atom name (e.g., CA for alpha carbon)
        element : str
            Chemical element (e.g., C, N, O)
        residue_name : str
            Name of the residue (e.g., ALA, GLY)
        residue_id : int
            Identifier of the residue
        chain_id : str
            Identifier of the chain
        mass : float
            Mass in atomic mass units (u)
        charge : float
            Charge in elementary charge units (e)
        position : np.ndarray
            Position [x, y, z] in nanometers
        velocity : np.ndarray, optional
            Velocity [vx, vy, vz] in nm/ps
        """
        # Initialize the underlying particle
        self.particle = Particle(atom_id, name, element, mass, charge, position, velocity)
        
        # Add atom-specific properties
        self.residue_name = residue_name
        self.residue_id = residue_id
        self.chain_id = chain_id
        
        # List of bonded atoms
        self.bonded_atoms = []
    
    def __repr__(self):
        return (f"Atom(id={self.particle.id}, name='{self.particle.name}', "
                f"element='{self.particle.element}', "
                f"residue={self.residue_name}{self.residue_id})")
    
    @property
    def id(self):
        return self.particle.id
    
    @property
    def name(self):
        return self.particle.name
    
    @property
    def element(self):
        return self.particle.element
    
    @property
    def mass(self):
        return self.particle.mass
    
    @property
    def charge(self):
        return self.particle.charge
    
    @property
    def position(self):
        return self.particle.position
    
    @position.setter
    def position(self, value):
        self.particle.position = value
    
    @property
    def velocity(self):
        return self.particle.velocity
    
    @velocity.setter
    def velocity(self, value):
        self.particle.velocity = value
    
    @property
    def force(self):
        return self.particle.force
    
    @force.setter
    def force(self, value):
        self.particle.force = value


class Residue:
    """
    Class representing a residue in a protein.
    
    A residue is a group of atoms that form an amino acid or nucleotide.
    """
    
    def __init__(self, residue_id: int, name: str, chain_id: str):
        """
        Initialize a residue.
        
        Parameters
        ----------
        residue_id : int
            Identifier of the residue
        name : str
            Name of the residue (e.g., ALA, GLY)
        chain_id : str
            Identifier of the chain
        """
        self.id = residue_id
        self.name = name
        self.chain_id = chain_id
        self.atoms = []
    
    def add_atom(self, atom):
        """Add an atom to the residue."""
        if atom.residue_id != self.id or atom.residue_name != self.name:
            logger.warning(f"Atom {atom.id} doesn't match residue {self.name}{self.id}")
        
        self.atoms.append(atom)
    
    def __repr__(self):
        return f"Residue(id={self.id}, name='{self.name}', chain='{self.chain_id}')"


class Chain:
    """
    Class representing a chain in a protein.
    
    A chain is a sequence of residues connected by peptide bonds.
    """
    
    def __init__(self, chain_id: str):
        """
        Initialize a chain.
        
        Parameters
        ----------
        chain_id : str
            Identifier of the chain
        """
        self.id = chain_id
        self.residues = []
    
    def add_residue(self, residue):
        """Add a residue to the chain."""
        if residue.chain_id != self.id:
            logger.warning(f"Residue {residue.name}{residue.id} doesn't match chain {self.id}")
        
        self.residues.append(residue)
    
    def __repr__(self):
        return f"Chain(id='{self.id}', residues={len(self.residues)})"


class Protein:
    """
    Class representing a protein.
    
    A protein is a collection of chains, which in turn are
    collections of residues, which are collections of atoms.
    """
    
    def __init__(self, name: str):
        """
        Initialize a protein.
        
        Parameters
        ----------
        name : str
            Name of the protein
        """
        self.name = name
        self.chains = {}
        self.atoms = []
    
    def add_chain(self, chain):
        """Add a chain to the protein."""
        self.chains[chain.id] = chain
        
        # Add atoms from this chain
        for residue in chain.residues:
            self.atoms.extend(residue.atoms)
    
    def __repr__(self):
        return f"Protein(name='{self.name}', chains={len(self.chains)}, atoms={len(self.atoms)})"


class PDBParser:
    """
    Class for parsing PDB files.
    
    This parses PDB files and creates a Protein object with
    the appropriate chains, residues, and atoms.
    """
    
    def __init__(self):
        """Initialize a PDB parser."""
        pass
    
    def parse(self, pdb_file: str) -> Protein:
        """
        Parse a PDB file and return a Protein object.
        
        Parameters
        ----------
        pdb_file : str
            Path to the PDB file
        
        Returns
        -------
        Protein
            The protein object created from the PDB file
        """
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        # Extract protein name from file name
        protein_name = os.path.splitext(os.path.basename(pdb_file))[0]
        protein = Protein(protein_name)
        
        # Dictionary to store chains and residues
        chains = {}
        current_residue = None
        
        logger.info(f"Parsing PDB file: {pdb_file}")
        
        with open(pdb_file, 'r') as f:
            atom_counter = 0
            
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # Parse atom information from the line
                    atom_id = int(line[6:11].strip())
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    chain_id = line[21].strip()
                    residue_id = int(line[22:26].strip())
                    x = float(line[30:38].strip()) / 10.0  # Å to nm
                    y = float(line[38:46].strip()) / 10.0
                    z = float(line[46:54].strip()) / 10.0
                    
                    # Extract element from columns 76-78 or from atom name
                    if len(line) >= 78:
                        element = line[76:78].strip()
                    else:
                        # Infer element from atom name
                        if atom_name.startswith(('C', 'N', 'O', 'H', 'S', 'P')):
                            element = atom_name[0]
                        else:
                            element = "C"  # Default to carbon
                    
                    # Get mass and charge from element
                    mass = self._get_mass(element)
                    charge = self._get_charge(element, atom_name, residue_name)
                    
                    # Create atom
                    position = np.array([x, y, z])
                    atom = Atom(atom_id, atom_name, element, residue_name, residue_id,
                               chain_id, mass, charge, position)
                    
                    # Create or get chain
                    if chain_id not in chains:
                        chains[chain_id] = Chain(chain_id)
                    
                    chain = chains[chain_id]
                    
                    # Create or get residue
                    if (current_residue is None or
                        current_residue.id != residue_id or
                        current_residue.name != residue_name or
                        current_residue.chain_id != chain_id):
                        
                        # Create new residue
                        current_residue = Residue(residue_id, residue_name, chain_id)
                        chain.add_residue(current_residue)
                    
                    # Add atom to residue
                    current_residue.add_atom(atom)
                    atom_counter += 1
        
        # Add chains to protein
        for chain in chains.values():
            protein.add_chain(chain)
        
        logger.info(f"Parsed protein with {len(protein.atoms)} atoms in {len(protein.chains)} chains")
        
        return protein
    
    def _get_mass(self, element: str) -> float:
        """Get the mass of an element in atomic mass units (u)."""
        masses = {
            'H': 1.008,
            'C': 12.011,
            'N': 14.007,
            'O': 15.999,
            'P': 30.974,
            'S': 32.065,
            'Fe': 55.845,
            'Zn': 65.38
        }
        
        return masses.get(element, 12.0)  # Default to carbon mass
    
    def _get_charge(self, element: str, atom_name: str, residue_name: str) -> float:
        """
        Get the charge of an atom based on element, atom name, and residue.
        
        In a real implementation, this would use a force field's charge assignments.
        """
        # This is a highly simplified placeholder
        # Real implementations would use proper force field assignments
        if residue_name == "ASP" and atom_name in ["OD1", "OD2"]:
            return -0.5
        elif residue_name == "GLU" and atom_name in ["OE1", "OE2"]:
            return -0.5
        elif residue_name == "LYS" and atom_name == "NZ":
            return 1.0
        elif residue_name == "ARG" and atom_name in ["NH1", "NH2"]:
            return 0.5
        else:
            return 0.0  # Default to neutral


class TopologyBuilder:
    """
    Class for building molecular topologies.
    
    This identifies bonds, angles, dihedrals, and other topological
    features needed for force field calculations.
    """
    
    def __init__(self):
        """Initialize a topology builder."""
        pass
    
    def build_topology(self, protein: Protein) -> dict:
        """
        Build a topology for a protein.
        
        Parameters
        ----------
        protein : Protein
            The protein to build a topology for
        
        Returns
        -------
        dict
            A dictionary containing the topology information
        """
        topology = {
            'bonds': [],
            'angles': [],
            'dihedrals': [],
            'impropers': []
        }
        
        # Identify bonds based on distance criteria
        self._identify_bonds(protein, topology)
        
        # Identify angles based on bonds
        self._identify_angles(topology)
        
        # Identify dihedrals based on angles
        self._identify_dihedrals(topology)
        
        # Identify impropers for planar groups
        self._identify_impropers(protein, topology)
        
        logger.info(f"Built topology with {len(topology['bonds'])} bonds, {len(topology['angles'])} angles")
        
        return topology
    
    def _identify_bonds(self, protein: Protein, topology: dict):
        """Identify bonds in a protein based on distance criteria."""
        # Dictionary of covalent radii in nm
        covalent_radii = {
            'H': 0.031,
            'C': 0.076,
            'N': 0.071,
            'O': 0.066,
            'P': 0.107,
            'S': 0.105
        }
        
        # Bond distance tolerance factor
        tolerance = 1.3
        
        # Loop through residues and identify bonds
        for chain in protein.chains.values():
            for i, residue in enumerate(chain.residues):
                # Identify bonds within the residue
                self._identify_intra_residue_bonds(residue, topology, covalent_radii, tolerance)
                
                # Identify bonds to the next residue
                if i < len(chain.residues) - 1:
                    next_residue = chain.residues[i + 1]
                    self._identify_inter_residue_bonds(residue, next_residue, topology, covalent_radii, tolerance)
    
    def _identify_intra_residue_bonds(self, residue, topology, covalent_radii, tolerance):
        """Identify bonds within a residue."""
        atoms = residue.atoms
        
        for i, atom1 in enumerate(atoms):
            for j in range(i + 1, len(atoms)):
                atom2 = atoms[j]
                
                # Calculate distance
                distance = np.linalg.norm(atom1.position - atom2.position)
                
                # Calculate maximum bond distance
                radius1 = covalent_radii.get(atom1.element, 0.076)  # Default to carbon
                radius2 = covalent_radii.get(atom2.element, 0.076)
                max_bond = (radius1 + radius2) * tolerance
                
                # If within bonding distance, add bond
                if distance <= max_bond:
                    topology['bonds'].append((atom1.id, atom2.id))
                    atom1.bonded_atoms.append(atom2)
                    atom2.bonded_atoms.append(atom1)
    
    def _identify_inter_residue_bonds(self, residue1, residue2, topology, covalent_radii, tolerance):
        """Identify bonds between consecutive residues."""
        # For proteins, the C of residue i bonds to the N of residue i+1
        c_atom = None
        n_atom = None
        
        for atom in residue1.atoms:
            if atom.name == "C":
                c_atom = atom
                break
        
        for atom in residue2.atoms:
            if atom.name == "N":
                n_atom = atom
                break
        
        if c_atom is not None and n_atom is not None:
            # Calculate distance
            distance = np.linalg.norm(c_atom.position - n_atom.position)
            
            # Calculate maximum bond distance
            radius1 = covalent_radii.get(c_atom.element, 0.076)
            radius2 = covalent_radii.get(n_atom.element, 0.071)
            max_bond = (radius1 + radius2) * tolerance
            
            # If within bonding distance, add bond
            if distance <= max_bond:
                topology['bonds'].append((c_atom.id, n_atom.id))
                c_atom.bonded_atoms.append(n_atom)
                n_atom.bonded_atoms.append(c_atom)
    
    def _identify_angles(self, topology: dict):
        """Identify angles based on bonds."""
        # Create a dictionary of bonded atoms
        bonds = topology['bonds']
        bonded_to = {}
        
        for atom1, atom2 in bonds:
            if atom1 not in bonded_to:
                bonded_to[atom1] = []
            if atom2 not in bonded_to:
                bonded_to[atom2] = []
            
            bonded_to[atom1].append(atom2)
            bonded_to[atom2].append(atom1)
        
        # Identify angles
        for atom2 in bonded_to:
            bonded_atoms = bonded_to[atom2]
            
            for i, atom1 in enumerate(bonded_atoms):
                for atom3 in bonded_atoms[i+1:]:
                    topology['angles'].append((atom1, atom2, atom3))
    
    def _identify_dihedrals(self, topology: dict):
        """Identify dihedrals based on angles."""
        # Simplified placeholder implementation
        pass
    
    def _identify_impropers(self, protein: Protein, topology: dict):
        """Identify improper dihedrals for planar groups."""
        # Simplified placeholder implementation
        pass
