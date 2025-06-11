"""
PDB file parser for protein structure processing.

This module provides classes and functions for reading and parsing 
Protein Data Bank (PDB) files to create internal protein structure representations.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set
import logging
import re
from pathlib import Path
import gzip
from ..structure.protein import Atom, Residue, Chain, Protein

# Configure logging
logger = logging.getLogger(__name__)

# Standard atomic masses for common elements (in atomic mass units)
ATOMIC_MASSES = {
    'H': 1.008, 'HE': 4.003, 'LI': 6.941, 'BE': 9.012, 'B': 10.811,
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'NE': 20.180,
    'NA': 22.990, 'MG': 24.305, 'AL': 26.982, 'SI': 28.086, 'P': 30.974,
    'S': 32.066, 'CL': 35.453, 'AR': 39.948, 'K': 39.098, 'CA': 40.078,
    'SC': 44.956, 'TI': 47.867, 'V': 50.942, 'CR': 51.996, 'MN': 54.938,
    'FE': 55.845, 'CO': 58.933, 'NI': 58.693, 'CU': 63.546, 'ZN': 65.39,
    'GA': 69.723, 'GE': 72.61, 'AS': 74.922, 'SE': 78.96, 'BR': 79.904,
    'KR': 83.80, 'RB': 85.468, 'SR': 87.62, 'Y': 88.906, 'ZR': 91.224,
    'NB': 92.906, 'MO': 95.94, 'TC': 98.0, 'RU': 101.07, 'RH': 102.906,
    'PD': 106.42, 'AG': 107.868, 'CD': 112.411, 'IN': 114.818, 'SN': 118.711,
    'SB': 121.760, 'TE': 127.60, 'I': 126.904, 'XE': 131.29, 'CS': 132.905,
    'BA': 137.327, 'LA': 138.906, 'CE': 140.116, 'PR': 140.908, 'ND': 144.24,
    'PM': 145.0, 'SM': 150.36, 'EU': 151.964, 'GD': 157.25, 'TB': 158.925,
    'DY': 162.50, 'HO': 164.930, 'ER': 167.26, 'TM': 168.934, 'YB': 173.04,
    'LU': 174.967, 'HF': 178.49, 'TA': 180.948, 'W': 183.84, 'RE': 186.207,
    'OS': 190.23, 'IR': 192.217, 'PT': 195.078, 'AU': 196.967, 'HG': 200.59,
    'TL': 204.383, 'PB': 207.2, 'BI': 208.980, 'PO': 209.0, 'AT': 210.0,
    'RN': 222.0, 'FR': 223.0, 'RA': 226.0, 'AC': 227.0, 'TH': 232.038,
    'PA': 231.036, 'U': 238.029, 'NP': 237.0, 'PU': 244.0, 'AM': 243.0,
    'CM': 247.0, 'BK': 247.0, 'CF': 251.0, 'ES': 252.0, 'FM': 257.0,
    'MD': 258.0, 'NO': 259.0, 'LR': 262.0, 'RF': 261.0, 'DB': 262.0,
    'SG': 266.0, 'BH': 264.0, 'HS': 277.0, 'MT': 268.0, 'DS': 281.0,
    'RG': 272.0, 'CN': 285.0, 'NH': 286.0, 'FL': 289.0, 'MC': 289.0,
    'LV': 293.0, 'TS': 294.0, 'OG': 294.0
}

# Standard residue mappings
THREE_TO_ONE_LETTER = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    # Non-standard amino acids
    'ASX': 'B', 'GLX': 'Z', 'SEC': 'U', 'PYL': 'O', 'XLE': 'J',
    # Nucleic acids
    'DA': 'A', 'DT': 'T', 'DG': 'G', 'DC': 'C',
    'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C',
    # Unknown
    'UNK': 'X'
}

class PDBParser:
    """
    Parser for Protein Data Bank (PDB) files.
    
    This class provides methods for reading PDB files and creating
    internal protein structure representations.
    """
    
    def __init__(self, ignore_missing_atoms: bool = False, ignore_missing_residues: bool = False):
        """
        Initialize a PDB parser.
        
        Parameters
        ----------
        ignore_missing_atoms : bool, optional
            Whether to ignore missing atoms in residues
        ignore_missing_residues : bool, optional
            Whether to ignore missing residues
        """
        self.ignore_missing_atoms = ignore_missing_atoms
        self.ignore_missing_residues = ignore_missing_residues
    
    def parse_file(self, pdb_path: Union[str, Path]) -> Protein:
        """
        Parse a PDB file and return a Protein object.
        
        Parameters
        ----------
        pdb_path : str or Path
            Path to PDB file
            
        Returns
        -------
        Protein
            Protein object containing the parsed data
        """
        pdb_path = Path(pdb_path)
        
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        
        # Check if the file is gzipped
        is_gzipped = pdb_path.suffix.lower() == '.gz'
        
        logger.info(f"Parsing PDB file: {pdb_path}")
        
        try:
            if is_gzipped:
                with gzip.open(pdb_path, 'rt') as f:
                    lines = f.readlines()
            else:
                with open(pdb_path, 'r') as f:
                    lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading PDB file: {str(e)}")
            raise
            
        # Get the PDB ID from the filename if possible
        pdb_id = pdb_path.stem
        if pdb_id.endswith('.pdb'):
            pdb_id = pdb_id[:-4]
        
        return self.parse_lines(lines, pdb_id)
        
    def parse_lines(self, lines: List[str], name: str) -> Protein:
        """
        Parse PDB lines and return a Protein object.
        
        Parameters
        ----------
        lines : list of str
            Lines from PDB file
        name : str
            Name of the protein (usually the PDB ID)
            
        Returns
        -------
        Protein
            Protein object containing the parsed data
        """
        # Initialize protein object
        protein = Protein(protein_id=name, name=name)
        
        # Parse header information for metadata
        title = ''
        experiment = ''
        resolution = None
        authors = []
        
        chains = {}  # Map chain_id to Chain object
        residues = {}  # Map (chain_id, residue_id) to Residue object
        atoms = {}  # Map atom_id to Atom object
        
        # First pass: extract metadata
        for line in lines:
            if line.startswith('TITLE'):
                title += line[10:].strip()
            elif line.startswith('EXPDTA'):
                experiment = line[10:].strip()
            elif line.startswith('REMARK   2 RESOLUTION'):
                try:
                    res_part = line[22:].strip()
                    if res_part and res_part != 'NOT APPLICABLE':
                        resolution = float(res_part.split()[0])
                except (ValueError, IndexError):
                    pass
            elif line.startswith('AUTHOR'):
                authors.extend([auth.strip() for auth in line[10:].strip().split(',')])
        
        protein.metadata['title'] = title
        protein.metadata['experiment'] = experiment
        protein.metadata['resolution'] = resolution
        protein.metadata['authors'] = ', '.join(authors)
        
        # Second pass: extract atoms, residues, and chains
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    atom_id = int(line[6:11].strip())
                    atom_name = line[12:16].strip()
                    alt_loc = line[16].strip()  # Alternate location indicator
                    
                    # Skip alternate locations other than '' or 'A'
                    if alt_loc and alt_loc != 'A':
                        continue
                    
                    residue_name = line[17:20].strip()
                    chain_id = line[21].strip()
                    if not chain_id:  # Use space if no chain ID
                        chain_id = ' '
                        
                    residue_id = int(line[22:26].strip())
                    insertion_code = line[26].strip()  # Insertion code
                    
                    if insertion_code:
                        # Append insertion code to residue_id
                        residue_id = f"{residue_id}{insertion_code}"
                    
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    position = np.array([x, y, z]) / 10.0  # Convert from Angstroms to nm
                    
                    occupancy = float(line[54:60].strip()) if line[54:60].strip() else 1.0
                    b_factor = float(line[60:66].strip()) if line[60:66].strip() else 0.0
                    
                    # Extract element and charge
                    element = line[76:78].strip()
                    if not element and atom_name:
                        # Try to infer element from atom name
                        element = ''.join([c for c in atom_name if not c.isdigit()]).strip()
                        # For most common atom names, first character is the element
                        if len(element) > 1 and not element[0].isalpha():
                            element = element[1:]
                        if len(element) > 1 and element[0].isalpha():
                            element = element[0]
                    
                    # Get mass from element
                    mass = ATOMIC_MASSES.get(element.upper(), 0.0)
                    
                    # Determine charge (often blank in PDB files)
                    charge = 0.0  # Default to neutral
                    charge_str = line[78:80].strip()
                    if charge_str:
                        try:
                            if charge_str[-1] in ['+', '-']:
                                # Format like '1+', '2-'
                                charge_val = int(charge_str[:-1]) if len(charge_str) > 1 else 1
                                charge = charge_val if charge_str[-1] == '+' else -charge_val
                            else:
                                # Format like '1', '-2'
                                charge = float(charge_str)
                        except ValueError:
                            logger.warning(f"Invalid charge format: {charge_str}")
                    
                    # Create or get chain
                    if chain_id not in chains:
                        chain = Chain(chain_id=chain_id)
                        chains[chain_id] = chain
                        protein.add_chain(chain)
                    else:
                        chain = chains[chain_id]
                    
                    # Create or get residue
                    residue_key = (chain_id, residue_id)
                    if residue_key not in residues:
                        is_hetero = line.startswith('HETATM')
                        residue = Residue(
                            residue_id=residue_id,
                            residue_name=residue_name,
                            chain_id=chain_id,
                            insertion_code=insertion_code,
                            is_hetero=is_hetero
                        )
                        residues[residue_key] = residue
                        chain.add_residue(residue)
                    else:
                        residue = residues[residue_key]
                    
                    # Create atom
                    atom = Atom(
                        atom_id=atom_id,
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
                    
                    atoms[atom_id] = atom
                    residue.add_atom(atom)
                    protein._atoms[atom_id] = atom  # Add directly to protein's atom dictionary
                    protein._atom_count += 1
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing atom line: {line.strip()} - {str(e)}")
                    continue
            
            # Parse CONECT records to establish bonds
            elif line.startswith('CONECT'):
                try:
                    fields = line[6:].strip().split()
                    if len(fields) > 1:
                        atom_id = int(fields[0])
                        if atom_id in atoms:
                            atom = atoms[atom_id]
                            for i in range(1, len(fields)):
                                try:
                                    bonded_id = int(fields[i])
                                    if bonded_id in atoms:
                                        atom.add_bond(bonded_id)
                                except ValueError:
                                    continue
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing CONECT line: {line.strip()} - {str(e)}")
                    continue
                    
        # Process secondary structure
        self._process_secondary_structure(lines, residues)
        
        # Add disulfide bonds
        self._process_disulfide_bonds(lines, atoms, residues)
        
        # Check if any atoms or residues were found
        if not atoms:
            raise ValueError("No atoms found in PDB file")
        
        # Calculate center of mass and other global properties
        protein.calculate_properties()
        
        logger.info(f"Parsed {len(atoms)} atoms, {len(residues)} residues, {len(chains)} chains")
        
        return protein
    
    def _process_secondary_structure(self, lines: List[str], residues: Dict[Tuple[str, str], 'Residue']):
        """Process secondary structure information from HELIX and SHEET records."""
        for line in lines:
            if line.startswith('HELIX'):
                try:
                    # Extract helix information
                    start_chain = line[19].strip()
                    start_res = int(line[21:25].strip())
                    end_chain = line[31].strip()
                    end_res = int(line[33:37].strip())
                    
                    # Assign alpha helix to residues in this range
                    for chain_id, res_id in residues:
                        if chain_id == start_chain and chain_id == end_chain:
                            if isinstance(res_id, int) and start_res <= res_id <= end_res:
                                residues[(chain_id, res_id)].secondary_structure = 'H'  # H for helix
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing HELIX line: {line.strip()} - {str(e)}")
                    continue
                    
            elif line.startswith('SHEET'):
                try:
                    # Extract sheet information
                    start_chain = line[21].strip()
                    start_res = int(line[22:26].strip())
                    end_chain = line[32].strip()
                    end_res = int(line[33:37].strip())
                    
                    # Assign beta sheet to residues in this range
                    for chain_id, res_id in residues:
                        if chain_id == start_chain and chain_id == end_chain:
                            if isinstance(res_id, int) and start_res <= res_id <= end_res:
                                residues[(chain_id, res_id)].secondary_structure = 'E'  # E for extended (sheet)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing SHEET line: {line.strip()} - {str(e)}")
                    continue
    
    def _process_disulfide_bonds(self, lines: List[str], atoms: Dict[int, 'Atom'], 
                                residues: Dict[Tuple[str, str], 'Residue']):
        """Process disulfide bond information from SSBOND records."""
        for line in lines:
            if line.startswith('SSBOND'):
                try:
                    # Extract disulfide bond information
                    first_chain = line[15].strip()
                    first_res = int(line[17:21].strip())
                    second_chain = line[29].strip()
                    second_res = int(line[31:35].strip())
                    
                    # Find the SG atoms in these cysteine residues
                    sg1 = None
                    sg2 = None
                    
                    for atom_id, atom in atoms.items():
                        if atom.chain_id == first_chain and atom.residue_id == first_res and atom.atom_name == 'SG':
                            sg1 = atom
                        elif atom.chain_id == second_chain and atom.residue_id == second_res and atom.atom_name == 'SG':
                            sg2 = atom
                    
                    # Add bond between SG atoms
                    if sg1 and sg2:
                        sg1.add_bond(sg2.atom_id)
                        sg2.add_bond(sg1.atom_id)
                        
                        # Mark residues as forming disulfide bond
                        res1_key = (first_chain, first_res)
                        res2_key = (second_chain, second_res)
                        
                        if res1_key in residues and res2_key in residues:
                            residues[res1_key].metadata['disulfide'] = True
                            residues[res2_key].metadata['disulfide'] = True
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing SSBOND line: {line.strip()} - {str(e)}")
                    continue
