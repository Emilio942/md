"""
MOL2 Format Reader/Writer for ProteinMD

This module provides support for Tripos MOL2 format files,
commonly used for small molecules and ligands.
"""

import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Tuple
import logging

from . import StructureData, BaseFormatReader, BaseFormatWriter

logger = logging.getLogger(__name__)


class MOL2Reader(BaseFormatReader):
    """Reader for Tripos MOL2 format files."""
    
    def read_structure(self, file_path: Union[str, Path]) -> StructureData:
        """Read a MOL2 structure file."""
        logger.info(f"Reading MOL2 file: {file_path}")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse sections
        atoms_section = False
        bonds_section = False
        
        coordinates = []
        elements = []
        atom_names = []
        atom_ids = []
        residue_names = []
        residue_ids = []
        chain_ids = []
        charges = []
        bonds = []
        
        title = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('@<TRIPOS>MOLECULE'):
                atoms_section = False
                bonds_section = False
                continue
            elif line.startswith('@<TRIPOS>ATOM'):
                atoms_section = True
                bonds_section = False
                continue
            elif line.startswith('@<TRIPOS>BOND'):
                atoms_section = False
                bonds_section = True
                continue
            elif line.startswith('@<TRIPOS>'):
                atoms_section = False
                bonds_section = False
                continue
            
            if not line or line.startswith('#'):
                continue
            
            if not atoms_section and not bonds_section and title is None:
                # First non-section line is the molecule name
                title = line
                continue
            
            if atoms_section:
                # Parse atom line: atom_id atom_name x y z atom_type subst_id subst_name charge
                parts = line.split()
                if len(parts) >= 6:
                    atom_id = int(parts[0])
                    atom_name = parts[1]
                    x, y, z = map(float, parts[2:5])
                    atom_type = parts[5] if len(parts) > 5 else 'C'
                    
                    # Extract element from atom type
                    element = atom_type.split('.')[0] if '.' in atom_type else atom_type
                    element = element.strip('0123456789')  # Remove numbers
                    if not element:
                        element = 'C'
                    
                    # Residue information
                    residue_id = int(parts[6]) if len(parts) > 6 else 1
                    residue_name = parts[7] if len(parts) > 7 else 'UNK'
                    charge = float(parts[8]) if len(parts) > 8 else 0.0
                    
                    coordinates.append([x, y, z])
                    elements.append(element)
                    atom_names.append(atom_name)
                    atom_ids.append(atom_id)
                    residue_names.append(residue_name)
                    residue_ids.append(residue_id)
                    chain_ids.append('A')  # MOL2 doesn't have chain info
                    charges.append(charge)
            
            elif bonds_section:
                # Parse bond line: bond_id origin_atom_id target_atom_id bond_type
                parts = line.split()
                if len(parts) >= 3:
                    origin_atom = int(parts[1])
                    target_atom = int(parts[2])
                    bonds.append((origin_atom - 1, target_atom - 1))  # Convert to 0-based
        
        if not coordinates:
            raise ValueError(f"No atoms found in MOL2 file: {file_path}")
        
        return StructureData(
            coordinates=np.array(coordinates) / 10.0,  # Convert Å to nm
            elements=elements,
            atom_names=atom_names,
            atom_ids=atom_ids,
            residue_names=residue_names,
            residue_ids=residue_ids,
            chain_ids=chain_ids,
            charges=charges,
            bonds=bonds,
            title=title
        )
    
    def can_read_trajectory(self) -> bool:
        """MOL2 reader doesn't support trajectory data."""
        return False


class MOL2Writer(BaseFormatWriter):
    """Writer for Tripos MOL2 format files."""
    
    def write_structure(self, structure: StructureData, file_path: Union[str, Path]) -> None:
        """Write a structure to MOL2 format."""
        logger.info(f"Writing MOL2 file: {file_path}")
        
        with open(file_path, 'w') as f:
            # Write header
            f.write("@<TRIPOS>MOLECULE\n")
            title = structure.title or "Molecule"
            f.write(f"{title}\n")
            
            n_bonds = len(structure.bonds) if structure.bonds else 0
            f.write(f"{structure.n_atoms} {n_bonds} 1 0 0\n")
            f.write("SMALL\n")
            f.write("GASTEIGER\n")
            f.write("\n")
            
            # Write atoms
            f.write("@<TRIPOS>ATOM\n")
            for i in range(structure.n_atoms):
                atom_id = structure.atom_ids[i] if structure.atom_ids else i + 1
                atom_name = structure.atom_names[i] if structure.atom_names else f"{structure.elements[i]}{i+1}"
                
                # Convert nm to Å
                x, y, z = structure.coordinates[i] * 10.0
                
                element = structure.elements[i] if structure.elements else "C"
                atom_type = f"{element}.3"  # Default to sp3 hybridization
                
                residue_id = structure.residue_ids[i] if structure.residue_ids else 1
                residue_name = structure.residue_names[i] if structure.residue_names else "UNK"
                charge = structure.charges[i] if structure.charges else 0.0
                
                f.write(f"{atom_id:6d} {atom_name:<8s} {x:10.4f} {y:10.4f} {z:10.4f} "
                       f"{atom_type:<8s} {residue_id:3d} {residue_name:<8s} {charge:10.6f}\n")
            
            # Write bonds if available
            if structure.bonds:
                f.write("@<TRIPOS>BOND\n")
                for bond_id, (atom1, atom2) in enumerate(structure.bonds, 1):
                    f.write(f"{bond_id:6d} {atom1+1:6d} {atom2+1:6d} 1\n")  # Convert to 1-based
    
    def can_write_trajectory(self) -> bool:
        """MOL2 writer doesn't support trajectory data."""
        return False
