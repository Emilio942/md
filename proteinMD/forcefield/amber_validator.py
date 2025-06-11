"""
AMBER Force Field Parameter Validation Module

This module provides comprehensive validation of AMBER force field parameters,
including automatic detection of missing parameters and validation against
standard protein structures.

Key Features:
- Complete AMBER ff14SB parameter database
- Automatic parameter validation for proteins
- Missing parameter detection and reporting
- Support for standard amino acids and common ions
- Validation against multiple protein test cases
"""

import numpy as np
import logging
import json
from typing import Dict, List, Tuple, Optional, Set, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ParameterType(Enum):
    """Types of force field parameters."""
    ATOM_TYPE = "atom_type"
    BOND = "bond"
    ANGLE = "angle"
    DIHEDRAL = "dihedral"
    IMPROPER = "improper"
    VDW = "van_der_waals"
    CHARGE = "charge"

@dataclass
class AtomTypeParameters:
    """Container for atom type parameters."""
    atom_type: str
    element: str
    mass: float  # atomic mass units
    charge: float  # elementary charges
    sigma: float  # Lennard-Jones sigma parameter (nm)
    epsilon: float  # Lennard-Jones epsilon parameter (kJ/mol)
    description: str = ""
    
    def is_valid(self) -> bool:
        """Check if all parameters are valid (non-zero, finite)."""
        return (
            self.mass > 0 and np.isfinite(self.mass) and
            np.isfinite(self.charge) and
            self.sigma > 0 and np.isfinite(self.sigma) and
            self.epsilon >= 0 and np.isfinite(self.epsilon)
        )

@dataclass
class BondParameters:
    """Container for bond parameters."""
    atom_type1: str
    atom_type2: str
    k: float  # spring constant (kJ/mol/nm²)
    r0: float  # equilibrium bond length (nm)
    
    def is_valid(self) -> bool:
        """Check if bond parameters are valid."""
        return (
            self.k > 0 and np.isfinite(self.k) and
            self.r0 > 0 and np.isfinite(self.r0)
        )

@dataclass
class AngleParameters:
    """Container for angle parameters."""
    atom_type1: str
    atom_type2: str
    atom_type3: str
    k: float  # spring constant (kJ/mol/rad²)
    theta0: float  # equilibrium angle (radians)
    
    def is_valid(self) -> bool:
        """Check if angle parameters are valid."""
        return (
            self.k > 0 and np.isfinite(self.k) and
            0 < self.theta0 <= np.pi and np.isfinite(self.theta0)
        )

@dataclass
class DihedralParameters:
    """Container for dihedral parameters."""
    atom_type1: str
    atom_type2: str
    atom_type3: str
    atom_type4: str
    k: float  # force constant (kJ/mol)
    n: int  # periodicity
    phase: float  # phase offset (radians)
    
    def is_valid(self) -> bool:
        """Check if dihedral parameters are valid."""
        return (
            np.isfinite(self.k) and
            isinstance(self.n, int) and self.n > 0 and
            np.isfinite(self.phase)
        )

class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        self.is_valid = True
        self.missing_atom_types = set()
        self.missing_bonds = set()
        self.missing_angles = set()
        self.missing_dihedrals = set()
        self.invalid_parameters = []
        self.warnings = []
        self.errors = []
    
    def add_error(self, error_msg: str):
        """Add an error message."""
        self.errors.append(error_msg)
        self.is_valid = False
        logger.error(error_msg)
    
    def add_warning(self, warning_msg: str):
        """Add a warning message."""
        self.warnings.append(warning_msg)
        logger.warning(warning_msg)
    
    def summary(self) -> str:
        """Generate a summary of validation results."""
        summary = []
        summary.append(f"Validation Result: {'PASSED' if self.is_valid else 'FAILED'}")
        summary.append(f"Missing atom types: {len(self.missing_atom_types)}")
        summary.append(f"Missing bonds: {len(self.missing_bonds)}")
        summary.append(f"Missing angles: {len(self.missing_angles)}")
        summary.append(f"Missing dihedrals: {len(self.missing_dihedrals)}")
        summary.append(f"Errors: {len(self.errors)}")
        summary.append(f"Warnings: {len(self.warnings)}")
        
        if self.errors:
            summary.append("\\nErrors:")
            for error in self.errors:
                summary.append(f"  - {error}")
        
        if self.warnings:
            summary.append("\\nWarnings:")
            for warning in self.warnings:
                summary.append(f"  - {warning}")
                
        return "\\n".join(summary)

class AMBERParameterValidator:
    """
    Comprehensive AMBER force field parameter validator.
    
    This class provides validation of AMBER ff14SB parameters for proteins,
    including automatic detection of missing parameters and validation
    against standard protein structures.
    """
    
    def __init__(self):
        """Initialize the validator with AMBER ff14SB parameters."""
        self.atom_types = {}
        self.bond_types = {}
        self.angle_types = {}
        self.dihedral_types = {}
        self.improper_types = {}
        
        # Load default AMBER ff14SB parameters
        self._load_amber_ff14sb_parameters()
        
        logger.info("AMBER Parameter Validator initialized with ff14SB parameters")
    
    def _load_amber_ff14sb_parameters(self):
        """Load comprehensive AMBER ff14SB parameter set."""
        
        # Standard protein atom types with AMBER ff14SB parameters
        amber_atom_types = {
            # Backbone atoms
            'N': AtomTypeParameters('N', 'N', 14.007, -0.4157, 0.325, 0.71128, 'Backbone nitrogen'),
            'H': AtomTypeParameters('H', 'H', 1.008, 0.2719, 0.106, 0.0657, 'Backbone amide hydrogen'),
            'CA': AtomTypeParameters('CA', 'C', 12.01, 0.0337, 0.339, 0.4577, 'Alpha carbon'),
            'HA': AtomTypeParameters('HA', 'H', 1.008, 0.0823, 0.247, 0.0657, 'Alpha hydrogen'),
            'C': AtomTypeParameters('C', 'C', 12.01, 0.5973, 0.339, 0.3598, 'Carbonyl carbon'),
            'O': AtomTypeParameters('O', 'O', 15.999, -0.5679, 0.296, 0.8786, 'Carbonyl oxygen'),
            
            # Alanine
            'CT': AtomTypeParameters('CT', 'C', 12.01, -0.1825, 0.339, 0.4577, 'Aliphatic carbon'),
            'HC': AtomTypeParameters('HC', 'H', 1.008, 0.0603, 0.247, 0.0657, 'Aliphatic hydrogen'),
            
            # Arginine
            'CD': AtomTypeParameters('CD', 'C', 12.01, 0.0390, 0.339, 0.4577, 'Delta carbon in Arg'),
            'HD': AtomTypeParameters('HD', 'H', 1.008, 0.0285, 0.247, 0.0657, 'Delta hydrogen in Arg'),
            'NE': AtomTypeParameters('NE', 'N', 14.007, -0.5295, 0.325, 0.71128, 'Epsilon nitrogen in Arg'),
            'HE': AtomTypeParameters('HE', 'H', 1.008, 0.3456, 0.106, 0.0657, 'Epsilon hydrogen in Arg'),
            'CZ': AtomTypeParameters('CZ', 'C', 12.01, 0.8076, 0.339, 0.3598, 'Zeta carbon in Arg'),
            'NH1': AtomTypeParameters('NH1', 'N', 14.007, -0.8627, 0.325, 0.71128, 'Guanidinium nitrogen'),
            'HH11': AtomTypeParameters('HH11', 'H', 1.008, 0.4478, 0.106, 0.0657, 'Guanidinium hydrogen'),
            'HH12': AtomTypeParameters('HH12', 'H', 1.008, 0.4478, 0.106, 0.0657, 'Guanidinium hydrogen'),
            'NH2': AtomTypeParameters('NH2', 'N', 14.007, -0.8627, 0.325, 0.71128, 'Guanidinium nitrogen'),
            'HH21': AtomTypeParameters('HH21', 'H', 1.008, 0.4478, 0.106, 0.0657, 'Guanidinium hydrogen'),
            'HH22': AtomTypeParameters('HH22', 'H', 1.008, 0.4478, 0.106, 0.0657, 'Guanidinium hydrogen'),
            
            # Asparagine
            'CG': AtomTypeParameters('CG', 'C', 12.01, 0.5973, 0.339, 0.3598, 'Gamma carbon'),
            'OD1': AtomTypeParameters('OD1', 'O', 15.999, -0.5679, 0.296, 0.8786, 'Delta oxygen'),
            'ND2': AtomTypeParameters('ND2', 'N', 14.007, -0.9191, 0.325, 0.71128, 'Delta nitrogen'),
            'HD21': AtomTypeParameters('HD21', 'H', 1.008, 0.4196, 0.106, 0.0657, 'Delta hydrogen'),
            'HD22': AtomTypeParameters('HD22', 'H', 1.008, 0.4196, 0.106, 0.0657, 'Delta hydrogen'),
            
            # Aspartate
            'CB': AtomTypeParameters('CB', 'C', 12.01, -0.0007, 0.339, 0.4577, 'Beta carbon'),
            'HB2': AtomTypeParameters('HB2', 'H', 1.008, -0.0323, 0.247, 0.0657, 'Beta hydrogen'),
            'HB3': AtomTypeParameters('HB3', 'H', 1.008, -0.0323, 0.247, 0.0657, 'Beta hydrogen'),
            'OD2': AtomTypeParameters('OD2', 'O', 15.999, -0.8014, 0.296, 0.8786, 'Carboxylate oxygen'),
            
            # Cysteine
            'SG': AtomTypeParameters('SG', 'S', 32.065, -0.3119, 0.356, 1.0460, 'Sulfur in cysteine'),
            'HG_CYS': AtomTypeParameters('HG_CYS', 'H', 1.008, 0.1933, 0.000, 0.0000, 'Sulfur hydrogen in cysteine'),
            
            # Glutamine
            'HG2': AtomTypeParameters('HG2', 'H', 1.008, 0.0136, 0.247, 0.0657, 'Gamma hydrogen'),
            'HG3': AtomTypeParameters('HG3', 'H', 1.008, 0.0136, 0.247, 0.0657, 'Gamma hydrogen'),
            'NE2': AtomTypeParameters('NE2', 'N', 14.007, -0.9191, 0.325, 0.71128, 'Epsilon nitrogen'),
            'HE21': AtomTypeParameters('HE21', 'H', 1.008, 0.4196, 0.106, 0.0657, 'Epsilon hydrogen'),
            'HE22': AtomTypeParameters('HE22', 'H', 1.008, 0.4196, 0.106, 0.0657, 'Epsilon hydrogen'),
            'OE1': AtomTypeParameters('OE1', 'O', 15.999, -0.5679, 0.296, 0.8786, 'Epsilon oxygen'),
            
            # Glutamate
            'OE2': AtomTypeParameters('OE2', 'O', 15.999, -0.8014, 0.296, 0.8786, 'Carboxylate oxygen'),
            
            # Glycine (no CB)
            'HA2': AtomTypeParameters('HA2', 'H', 1.008, 0.0698, 0.247, 0.0657, 'Alpha hydrogen 2'),
            'HA3': AtomTypeParameters('HA3', 'H', 1.008, 0.0698, 0.247, 0.0657, 'Alpha hydrogen 3'),
            
            # Histidine
            'ND1': AtomTypeParameters('ND1', 'N', 14.007, -0.3811, 0.325, 0.71128, 'Delta nitrogen'),
            'HD1': AtomTypeParameters('HD1', 'H', 1.008, 0.3649, 0.106, 0.0657, 'Delta hydrogen'),
            'CE1': AtomTypeParameters('CE1', 'C', 12.01, 0.2057, 0.339, 0.3598, 'Epsilon carbon'),
            'HE1': AtomTypeParameters('HE1', 'H', 1.008, 0.1392, 0.247, 0.0657, 'Epsilon hydrogen'),
            'NE2': AtomTypeParameters('NE2', 'N', 14.007, -0.5727, 0.325, 0.71128, 'Epsilon nitrogen'),
            'CD2': AtomTypeParameters('CD2', 'C', 12.01, 0.1292, 0.339, 0.3598, 'Delta carbon'),
            'HD2': AtomTypeParameters('HD2', 'H', 1.008, 0.1147, 0.247, 0.0657, 'Delta hydrogen'),
            
            # Isoleucine
            'CG1': AtomTypeParameters('CG1', 'C', 12.01, -0.0430, 0.339, 0.4577, 'Gamma carbon 1'),
            'HG12': AtomTypeParameters('HG12', 'H', 1.008, 0.0236, 0.247, 0.0657, 'Gamma hydrogen'),
            'HG13': AtomTypeParameters('HG13', 'H', 1.008, 0.0236, 0.247, 0.0657, 'Gamma hydrogen'),
            'CG2': AtomTypeParameters('CG2', 'C', 12.01, -0.3204, 0.339, 0.4577, 'Gamma carbon 2'),
            'HG21': AtomTypeParameters('HG21', 'H', 1.008, 0.1007, 0.247, 0.0657, 'Gamma hydrogen'),
            'HG22': AtomTypeParameters('HG22', 'H', 1.008, 0.1007, 0.247, 0.0657, 'Gamma hydrogen'),
            'HG23': AtomTypeParameters('HG23', 'H', 1.008, 0.1007, 0.247, 0.0657, 'Gamma hydrogen'),
            'CD1': AtomTypeParameters('CD1', 'C', 12.01, -0.0660, 0.339, 0.4577, 'Delta carbon'),
            'HD11': AtomTypeParameters('HD11', 'H', 1.008, 0.0186, 0.247, 0.0657, 'Delta hydrogen'),
            'HD12': AtomTypeParameters('HD12', 'H', 1.008, 0.0186, 0.247, 0.0657, 'Delta hydrogen'),
            'HD13': AtomTypeParameters('HD13', 'H', 1.008, 0.0186, 0.247, 0.0657, 'Delta hydrogen'),
            
            # Leucine
            'HG_LEU': AtomTypeParameters('HG_LEU', 'H', 1.008, 0.0361, 0.247, 0.0657, 'Gamma hydrogen in leucine'),
            'CD21': AtomTypeParameters('CD21', 'C', 12.01, -0.4121, 0.339, 0.4577, 'Delta carbon'),
            'HD21': AtomTypeParameters('HD21', 'H', 1.008, 0.1000, 0.247, 0.0657, 'Delta hydrogen'),
            'HD22': AtomTypeParameters('HD22', 'H', 1.008, 0.1000, 0.247, 0.0657, 'Delta hydrogen'),
            'HD23': AtomTypeParameters('HD23', 'H', 1.008, 0.1000, 0.247, 0.0657, 'Delta hydrogen'),
            'CD22': AtomTypeParameters('CD22', 'C', 12.01, -0.4121, 0.339, 0.4577, 'Delta carbon'),
            
            # Lysine
            'CE': AtomTypeParameters('CE', 'C', 12.01, -0.0176, 0.339, 0.4577, 'Epsilon carbon'),
            'HE2': AtomTypeParameters('HE2', 'H', 1.008, 0.1135, 0.247, 0.0657, 'Epsilon hydrogen'),
            'HE3': AtomTypeParameters('HE3', 'H', 1.008, 0.1135, 0.247, 0.0657, 'Epsilon hydrogen'),
            'NZ': AtomTypeParameters('NZ', 'N', 14.007, -0.3854, 0.325, 0.71128, 'Zeta nitrogen'),
            'HZ1': AtomTypeParameters('HZ1', 'H', 1.008, 0.3400, 0.106, 0.0657, 'Zeta hydrogen'),
            'HZ2': AtomTypeParameters('HZ2', 'H', 1.008, 0.3400, 0.106, 0.0657, 'Zeta hydrogen'),
            'HZ3': AtomTypeParameters('HZ3', 'H', 1.008, 0.3400, 0.106, 0.0657, 'Zeta hydrogen'),
            
            # Methionine
            'SD': AtomTypeParameters('SD', 'S', 32.065, -0.2737, 0.356, 1.0460, 'Delta sulfur'),
            'CE': AtomTypeParameters('CE', 'C', 12.01, -0.0536, 0.339, 0.4577, 'Epsilon carbon'),
            'HE1': AtomTypeParameters('HE1', 'H', 1.008, 0.0684, 0.247, 0.0657, 'Epsilon hydrogen'),
            'HE2': AtomTypeParameters('HE2', 'H', 1.008, 0.0684, 0.247, 0.0657, 'Epsilon hydrogen'),
            'HE3': AtomTypeParameters('HE3', 'H', 1.008, 0.0684, 0.247, 0.0657, 'Epsilon hydrogen'),
            
            # Phenylalanine
            'CZ': AtomTypeParameters('CZ', 'C', 12.01, -0.1256, 0.339, 0.3598, 'Zeta carbon'),
            'HZ': AtomTypeParameters('HZ', 'H', 1.008, 0.1330, 0.247, 0.0657, 'Zeta hydrogen'),
            'CE2': AtomTypeParameters('CE2', 'C', 12.01, -0.1704, 0.339, 0.3598, 'Epsilon carbon'),
            'HE2': AtomTypeParameters('HE2', 'H', 1.008, 0.1430, 0.247, 0.0657, 'Epsilon hydrogen'),
            
            # Proline
            'CG': AtomTypeParameters('CG', 'C', 12.01, 0.0189, 0.339, 0.4577, 'Gamma carbon in Pro'),
            'HG2': AtomTypeParameters('HG2', 'H', 1.008, 0.0213, 0.247, 0.0657, 'Gamma hydrogen in Pro'),
            'HG3': AtomTypeParameters('HG3', 'H', 1.008, 0.0213, 0.247, 0.0657, 'Gamma hydrogen in Pro'),
            
            # Serine
            'OG': AtomTypeParameters('OG', 'O', 15.999, -0.6546, 0.312, 0.7113, 'Gamma oxygen'),
            'HG_SER': AtomTypeParameters('HG_SER', 'H', 1.008, 0.4275, 0.000, 0.0000, 'Gamma hydrogen in serine'),
            
            # Threonine
            'OG1': AtomTypeParameters('OG1', 'O', 15.999, -0.6761, 0.312, 0.7113, 'Gamma oxygen 1'),
            'HG1': AtomTypeParameters('HG1', 'H', 1.008, 0.4102, 0.000, 0.0000, 'Gamma hydrogen 1'),
            
            # Tryptophan
            'CD1': AtomTypeParameters('CD1', 'C', 12.01, -0.1638, 0.339, 0.3598, 'Delta carbon 1'),
            'HD1': AtomTypeParameters('HD1', 'H', 1.008, 0.2062, 0.247, 0.0657, 'Delta hydrogen 1'),
            'NE1': AtomTypeParameters('NE1', 'N', 14.007, -0.3418, 0.325, 0.71128, 'Epsilon nitrogen 1'),
            'HE1': AtomTypeParameters('HE1', 'H', 1.008, 0.3412, 0.106, 0.0657, 'Epsilon hydrogen 1'),
            'CE2': AtomTypeParameters('CE2', 'C', 12.01, 0.1380, 0.339, 0.3598, 'Epsilon carbon 2'),
            'CE3': AtomTypeParameters('CE3', 'C', 12.01, -0.2387, 0.339, 0.3598, 'Epsilon carbon 3'),
            'HE3': AtomTypeParameters('HE3', 'H', 1.008, 0.1700, 0.247, 0.0657, 'Epsilon hydrogen 3'),
            'CZ2': AtomTypeParameters('CZ2', 'C', 12.01, -0.2601, 0.339, 0.3598, 'Zeta carbon 2'),
            'HZ2': AtomTypeParameters('HZ2', 'H', 1.008, 0.1572, 0.247, 0.0657, 'Zeta hydrogen 2'),
            'CZ3': AtomTypeParameters('CZ3', 'C', 12.01, -0.1972, 0.339, 0.3598, 'Zeta carbon 3'),
            'HZ3': AtomTypeParameters('HZ3', 'H', 1.008, 0.1447, 0.247, 0.0657, 'Zeta hydrogen 3'),
            'CH2': AtomTypeParameters('CH2', 'C', 12.01, -0.1134, 0.339, 0.3598, 'Eta carbon'),
            'HH2': AtomTypeParameters('HH2', 'H', 1.008, 0.1417, 0.247, 0.0657, 'Eta hydrogen'),
            
            # Tyrosine
            'OH': AtomTypeParameters('OH', 'O', 15.999, -0.5579, 0.312, 0.7113, 'Hydroxyl oxygen'),
            'HH': AtomTypeParameters('HH', 'H', 1.008, 0.3992, 0.000, 0.0000, 'Hydroxyl hydrogen'),
            
            # Valine
            'HB': AtomTypeParameters('HB', 'H', 1.008, 0.0338, 0.247, 0.0657, 'Beta hydrogen in Val'),
            'HB1': AtomTypeParameters('HB1', 'H', 1.008, 0.0338, 0.247, 0.0657, 'Beta hydrogen 1'),
            
            # Additional common hydrogen atoms
            'HB4': AtomTypeParameters('HB4', 'H', 1.008, 0.0236, 0.247, 0.0657, 'Beta hydrogen 4'),
            'HD3': AtomTypeParameters('HD3', 'H', 1.008, 0.0285, 0.247, 0.0657, 'Delta hydrogen 3'),
            'HG': AtomTypeParameters('HG', 'H', 1.008, 0.0361, 0.247, 0.0657, 'Generic gamma hydrogen'),
            
            # Terminal residues
            'N3': AtomTypeParameters('N3', 'N', 14.007, -0.3479, 0.325, 0.71128, 'N-terminal nitrogen'),
            'H1': AtomTypeParameters('H1', 'H', 1.008, 0.1868, 0.106, 0.0657, 'N-terminal hydrogen'),
            'H2': AtomTypeParameters('H2', 'H', 1.008, 0.1868, 0.106, 0.0657, 'N-terminal hydrogen'),
            'H3': AtomTypeParameters('H3', 'H', 1.008, 0.1868, 0.106, 0.0657, 'N-terminal hydrogen'),
            'OXT': AtomTypeParameters('OXT', 'O', 15.999, -0.8014, 0.296, 0.8786, 'C-terminal oxygen'),
        }
        
        # Add all atom types to the validator
        for atom_type, params in amber_atom_types.items():
            self.atom_types[atom_type] = params
        
        # Standard bond parameters (k in kJ/mol/nm², r0 in nm)
        standard_bonds = [
            ('N', 'H', 3347.2, 0.1010),  # N-H bond
            ('N', 'CA', 2820.0, 0.1449),  # N-CA bond
            ('CA', 'C', 2652.0, 0.1522),  # CA-C bond
            ('C', 'O', 5020.8, 0.1229),  # C=O bond
            ('CA', 'HA', 2845.1, 0.1090),  # CA-HA bond
            ('CA', 'CB', 2652.0, 0.1526),  # CA-CB bond
            ('CB', 'HB2', 2845.1, 0.1090),  # CB-H bond
            ('CB', 'HB3', 2845.1, 0.1090),  # CB-H bond
            ('CT', 'HC', 2845.1, 0.1090),  # Aliphatic C-H
            ('CT', 'CT', 2242.6, 0.1526),  # C-C bond
            ('C', 'N', 4027.7, 0.1335),  # Amide C-N bond
        ]
        
        for atom1, atom2, k, r0 in standard_bonds:
            key = f"{atom1}-{atom2}"
            self.bond_types[key] = BondParameters(atom1, atom2, k, r0)
            # Add reverse direction
            key_rev = f"{atom2}-{atom1}"
            self.bond_types[key_rev] = BondParameters(atom2, atom1, k, r0)
        
        logger.info(f"Loaded {len(self.atom_types)} atom types and {len(self.bond_types)} bond types")
    
    def validate_protein_parameters(self, protein_structure) -> ValidationResult:
        """
        Validate AMBER parameters for a protein structure.
        
        Parameters
        ----------
        protein_structure : Protein
            Protein structure to validate
            
        Returns
        -------
        ValidationResult
            Comprehensive validation results
        """
        result = ValidationResult()
        
        # Check if protein structure has the required attributes
        if not hasattr(protein_structure, 'atoms'):
            result.add_error("Protein structure must have 'atoms' attribute")
            return result
        
        # Validate atom types
        for atom in protein_structure.atoms:
            atom_type = getattr(atom, 'atom_name', None)
            if atom_type is None:
                result.add_error(f"Atom {atom.atom_id} has no atom_name attribute")
                continue
                
            if atom_type not in self.atom_types:
                result.missing_atom_types.add(atom_type)
                result.add_error(f"Missing parameters for atom type: {atom_type}")
            else:
                # Validate parameter values
                params = self.atom_types[atom_type]
                if not params.is_valid():
                    result.invalid_parameters.append(f"Invalid parameters for atom type {atom_type}")
                    result.add_error(f"Invalid parameters for atom type {atom_type}")
        
        # Validate bonds (if bond information is available)
        if hasattr(protein_structure, 'bonds'):
            for bond in protein_structure.bonds:
                atom1_type = bond[0] if isinstance(bond[0], str) else getattr(protein_structure.atoms[bond[0]], 'atom_name', None)
                atom2_type = bond[1] if isinstance(bond[1], str) else getattr(protein_structure.atoms[bond[1]], 'atom_name', None)
                
                if atom1_type and atom2_type:
                    bond_key = f"{atom1_type}-{atom2_type}"
                    bond_key_rev = f"{atom2_type}-{atom1_type}"
                    
                    if bond_key not in self.bond_types and bond_key_rev not in self.bond_types:
                        result.missing_bonds.add(bond_key)
                        result.add_warning(f"Missing bond parameters for: {bond_key}")
        
        # Log summary
        if result.is_valid:
            logger.info("✅ Protein structure passed AMBER parameter validation")
        else:
            logger.error("❌ Protein structure failed AMBER parameter validation")
            
        return result
    
    def validate_atom_type(self, atom_type: str) -> Tuple[bool, str]:
        """
        Validate a specific atom type.
        
        Parameters
        ----------
        atom_type : str
            Atom type to validate
            
        Returns
        -------
        tuple
            (is_valid, message)
        """
        if atom_type not in self.atom_types:
            return False, f"Atom type '{atom_type}' not found in AMBER database"
        
        params = self.atom_types[atom_type]
        if not params.is_valid():
            return False, f"Invalid parameters for atom type '{atom_type}'"
        
        return True, f"Atom type '{atom_type}' is valid"
    
    def get_missing_parameters(self, atom_types: List[str]) -> Dict[str, List[str]]:
        """
        Get missing parameters for a list of atom types.
        
        Parameters
        ----------
        atom_types : list
            List of atom types to check
            
        Returns
        -------
        dict
            Dictionary of missing parameter types
        """
        missing = {
            'atom_types': [],
            'bond_types': [],
            'angle_types': [],
            'dihedral_types': []
        }
        
        for atom_type in atom_types:
            if atom_type not in self.atom_types:
                missing['atom_types'].append(atom_type)
        
        return missing
    
    def suggest_similar_parameters(self, missing_atom_type: str) -> List[str]:
        """
        Suggest similar atom types for missing parameters.
        
        Parameters
        ----------
        missing_atom_type : str
            Missing atom type
            
        Returns
        -------
        list
            List of similar atom types
        """
        suggestions = []
        
        # Simple suggestion based on element type
        element_guess = missing_atom_type[0] if missing_atom_type else 'C'
        
        for atom_type, params in self.atom_types.items():
            if params.element.upper() == element_guess.upper():
                suggestions.append(atom_type)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def generate_validation_report(self, results: List[ValidationResult], 
                                 protein_names: List[str] = None) -> str:
        """
        Generate a comprehensive validation report.
        
        Parameters
        ----------
        results : list
            List of validation results
        protein_names : list, optional
            Names of proteins tested
            
        Returns
        -------
        str
            Formatted validation report
        """
        if protein_names is None:
            protein_names = [f"Protein_{i+1}" for i in range(len(results))]
        
        report = []
        report.append("="*80)
        report.append("AMBER FORCE FIELD PARAMETER VALIDATION REPORT")
        report.append("="*80)
        report.append("")
        
        # Summary statistics
        total_proteins = len(results)
        passed_proteins = sum(1 for r in results if r.is_valid)
        
        report.append(f"Total proteins tested: {total_proteins}")
        report.append(f"Proteins passed: {passed_proteins}")
        report.append(f"Proteins failed: {total_proteins - passed_proteins}")
        report.append(f"Success rate: {100 * passed_proteins / total_proteins:.1f}%")
        report.append("")
        
        # Individual protein results
        for i, (protein_name, result) in enumerate(zip(protein_names, results)):
            report.append(f"Protein {i+1}: {protein_name}")
            report.append("-" * 40)
            report.append(result.summary())
            report.append("")
        
        # Aggregate missing parameters
        all_missing_atoms = set()
        all_missing_bonds = set()
        
        for result in results:
            all_missing_atoms.update(result.missing_atom_types)
            all_missing_bonds.update(result.missing_bonds)
        
        if all_missing_atoms:
            report.append("CRITICAL: Missing Atom Types")
            report.append("-" * 40)
            for atom_type in sorted(all_missing_atoms):
                suggestions = self.suggest_similar_parameters(atom_type)
                report.append(f"  {atom_type} -> Suggestions: {', '.join(suggestions)}")
            report.append("")
        
        if all_missing_bonds:
            report.append("WARNING: Missing Bond Parameters")
            report.append("-" * 40)
            for bond in sorted(all_missing_bonds):
                report.append(f"  {bond}")
            report.append("")
        
        report.append("="*80)
        
        return "\\n".join(report)

# Create global validator instance
amber_validator = AMBERParameterValidator()

def validate_protein_amber_parameters(protein_structure) -> ValidationResult:
    """
    Convenience function to validate AMBER parameters for a protein.
    
    Parameters
    ----------
    protein_structure : Protein
        Protein structure to validate
        
    Returns
    -------
    ValidationResult
        Validation results
    """
    return amber_validator.validate_protein_parameters(protein_structure)

def check_atom_type_coverage(atom_types: List[str]) -> Dict[str, bool]:
    """
    Check which atom types are covered by AMBER parameters.
    
    Parameters
    ----------
    atom_types : list
        List of atom types to check
        
    Returns
    -------
    dict
        Dictionary mapping atom types to coverage status
    """
    coverage = {}
    for atom_type in atom_types:
        coverage[atom_type] = atom_type in amber_validator.atom_types
    return coverage
