"""
Custom Force Field Import Module

This module provides functionality to import custom force field parameters
from JSON or XML files, with comprehensive validation and error handling.

Features:
- JSON and XML format support for force field parameters
- Comprehensive parameter validation
- Error handling for invalid parameters
- Documentation and examples for custom parameters
- Compatible with existing ForceField infrastructure
"""

import json
import xml.etree.ElementTree as ET
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Optional jsonschema import
try:
    import jsonschema
    from jsonschema import validate, ValidationError as JSONValidationError
    JSONSCHEMA_AVAILABLE = True
except (ImportError, SyntaxError) as e:
    JSONSCHEMA_AVAILABLE = False
    JSONValidationError = ValueError  # Fallback
    logger.warning(f"jsonschema not available: {e}")

from .forcefield import ForceField, NonbondedMethod, ForceFieldType

@dataclass
class CustomAtomTypeParameters:
    """Custom atom type parameters."""
    atom_type: str
    mass: float  # amu
    sigma: float  # nm (LJ parameter)
    epsilon: float  # kJ/mol (LJ parameter)
    charge: Optional[float] = None  # elementary charge
    description: Optional[str] = None
    source: Optional[str] = None

@dataclass
class CustomBondParameters:
    """Custom bond parameters."""
    atom_types: Tuple[str, str]
    k: float  # Force constant in kJ/mol/nm²
    r0: float  # Equilibrium bond length in nm
    description: Optional[str] = None
    source: Optional[str] = None

@dataclass
class CustomAngleParameters:
    """Custom angle parameters."""
    atom_types: Tuple[str, str, str]
    k: float  # Force constant in kJ/mol/rad²
    theta0: float  # Equilibrium angle in degrees
    description: Optional[str] = None
    source: Optional[str] = None

@dataclass
class CustomDihedralParameters:
    """Custom dihedral parameters."""
    atom_types: Tuple[str, str, str, str]
    k: float  # Force constant in kJ/mol
    n: int    # Periodicity
    phase: float  # Phase in degrees
    description: Optional[str] = None
    source: Optional[str] = None

# JSON Schema for custom force field validation
CUSTOM_FORCEFIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "metadata": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "description": {"type": "string"},
                "author": {"type": "string"},
                "date": {"type": "string"},
                "references": {"type": "array", "items": {"type": "string"}},
                "units": {
                    "type": "object",
                    "properties": {
                        "length": {"type": "string", "enum": ["nm", "angstrom"]},
                        "energy": {"type": "string", "enum": ["kJ/mol", "kcal/mol"]},
                        "mass": {"type": "string", "enum": ["amu", "g/mol"]},
                        "angle": {"type": "string", "enum": ["degrees", "radians"]}
                    },
                    "required": ["length", "energy", "mass", "angle"]
                }
            },
            "required": ["name", "units"]
        },
        "atom_types": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "atom_type": {"type": "string"},
                    "mass": {"type": "number", "minimum": 0},
                    "sigma": {"type": "number", "minimum": 0},
                    "epsilon": {"type": "number", "minimum": 0},
                    "charge": {"type": ["number", "null"]},
                    "description": {"type": ["string", "null"]},
                    "source": {"type": ["string", "null"]}
                },
                "required": ["atom_type", "mass", "sigma", "epsilon"]
            }
        },
        "bond_types": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "atom_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2
                    },
                    "k": {"type": "number", "minimum": 0},
                    "r0": {"type": "number", "minimum": 0},
                    "description": {"type": ["string", "null"]},
                    "source": {"type": ["string", "null"]}
                },
                "required": ["atom_types", "k", "r0"]
            }
        },
        "angle_types": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "atom_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3,
                        "maxItems": 3
                    },
                    "k": {"type": "number", "minimum": 0},
                    "theta0": {"type": "number", "minimum": 0, "maximum": 180},
                    "description": {"type": ["string", "null"]},
                    "source": {"type": ["string", "null"]}
                },
                "required": ["atom_types", "k", "theta0"]
            }
        },
        "dihedral_types": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "atom_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 4,
                        "maxItems": 4
                    },
                    "k": {"type": "number"},
                    "n": {"type": "integer", "minimum": 1},
                    "phase": {"type": "number", "minimum": -180, "maximum": 180},
                    "description": {"type": ["string", "null"]},
                    "source": {"type": ["string", "null"]}
                },
                "required": ["atom_types", "k", "n", "phase"]
            }
        }
    },
    "required": ["metadata", "atom_types"]
}

class CustomForceFieldImporter:
    """Importer for custom force field parameters from JSON or XML files."""
    
    def __init__(self, validate_schema: bool = True):
        """
        Initialize the custom force field importer.
        
        Parameters
        ----------
        validate_schema : bool, optional
            Whether to validate JSON schema on import (default: True)
        """
        self.validate_schema = validate_schema
        self.loaded_forcefields = {}
        
    def import_from_json(self, filename: Union[str, Path]) -> 'CustomForceField':
        """
        Import custom force field parameters from a JSON file.
        
        Parameters
        ----------
        filename : str or Path
            Path to the JSON file containing force field parameters
            
        Returns
        -------
        CustomForceField
            The imported custom force field
            
        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        JSONDecodeError
            If the file contains invalid JSON
        ValidationError
            If the JSON structure does not match the expected schema (when jsonschema is available)
        ValueError
            If parameter values are invalid
        """
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"Force field file not found: {filename}")
            
        logger.info(f"Importing custom force field from JSON: {filename}")
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in force field file {filename}: {e}")
            
        # Validate JSON schema
        if self.validate_schema and JSONSCHEMA_AVAILABLE:
            try:
                validate(instance=data, schema=CUSTOM_FORCEFIELD_SCHEMA)
            except JSONValidationError as e:
                raise ValueError(f"JSON schema validation failed for {filename}: {e.message}")
        elif self.validate_schema and not JSONSCHEMA_AVAILABLE:
            logger.warning("jsonschema not available, skipping schema validation")
        
        # Convert units if necessary
        data = self._convert_units(data)
        
        # Validate parameter values
        self._validate_parameters(data)
        
        # Create CustomForceField instance
        forcefield = CustomForceField()
        forcefield.load_from_data(data)
        
        # Cache the loaded force field
        self.loaded_forcefields[str(filename)] = forcefield
        
        logger.info(f"Successfully imported custom force field: {data['metadata']['name']}")
        return forcefield
        
    def import_from_xml(self, filename: Union[str, Path]) -> 'CustomForceField':
        """
        Import custom force field parameters from an XML file.
        
        Parameters
        ----------
        filename : str or Path
            Path to the XML file containing force field parameters
            
        Returns
        -------
        CustomForceField
            The imported custom force field
            
        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        ET.ParseError
            If the file contains invalid XML
        ValueError
            If parameter values are invalid
        """
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"Force field file not found: {filename}")
            
        logger.info(f"Importing custom force field from XML: {filename}")
        
        try:
            tree = ET.parse(filename)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ET.ParseError(f"Invalid XML in force field file {filename}: {e}")
            
        # Convert XML to dictionary format
        data = self._xml_to_dict(root)
        
        # Validate parameter values
        self._validate_parameters(data)
        
        # Create CustomForceField instance
        forcefield = CustomForceField()
        forcefield.load_from_data(data)
        
        # Cache the loaded force field
        self.loaded_forcefields[str(filename)] = forcefield
        
        logger.info(f"Successfully imported custom force field: {data['metadata']['name']}")
        return forcefield
    
    def _convert_units(self, data: Dict) -> Dict:
        """Convert units to standard ProteinMD units (nm, kJ/mol, amu, degrees)."""
        units = data['metadata']['units']
        
        # Length conversion factors to nm
        length_factors = {
            'nm': 1.0,
            'angstrom': 0.1
        }
        
        # Energy conversion factors to kJ/mol
        energy_factors = {
            'kJ/mol': 1.0,
            'kcal/mol': 4.184
        }
        
        # Mass conversion factors to amu
        mass_factors = {
            'amu': 1.0,
            'g/mol': 1.0  # Same as amu
        }
        
        # Angle conversion factors to degrees
        angle_factors = {
            'degrees': 1.0,
            'radians': 180.0 / np.pi
        }
        
        length_factor = length_factors.get(units['length'], 1.0)
        energy_factor = energy_factors.get(units['energy'], 1.0)
        mass_factor = mass_factors.get(units['mass'], 1.0)
        angle_factor = angle_factors.get(units['angle'], 1.0)
        
        # Convert atom types
        for atom_type in data.get('atom_types', []):
            atom_type['mass'] *= mass_factor
            atom_type['sigma'] *= length_factor
            atom_type['epsilon'] *= energy_factor
            
        # Convert bond types
        for bond_type in data.get('bond_types', []):
            bond_type['k'] *= energy_factor / (length_factor ** 2)
            bond_type['r0'] *= length_factor
            
        # Convert angle types
        for angle_type in data.get('angle_types', []):
            angle_type['k'] *= energy_factor / (angle_factor * np.pi / 180.0) ** 2
            angle_type['theta0'] *= angle_factor
            
        # Convert dihedral types
        for dihedral_type in data.get('dihedral_types', []):
            dihedral_type['k'] *= energy_factor
            dihedral_type['phase'] *= angle_factor
            
        return data
    
    def _validate_parameters(self, data: Dict):
        """Validate parameter values for physical reasonableness."""
        errors = []
        
        # Validate atom types
        for i, atom_type in enumerate(data.get('atom_types', [])):
            if atom_type['mass'] <= 0:
                errors.append(f"Atom type {i}: Mass must be positive")
            if atom_type['mass'] > 1000:  # Sanity check
                errors.append(f"Atom type {i}: Mass {atom_type['mass']} amu seems too large")
            if atom_type['sigma'] <= 0:
                errors.append(f"Atom type {i}: Sigma must be positive")
            if atom_type['sigma'] > 1.0:  # 1 nm is very large
                errors.append(f"Atom type {i}: Sigma {atom_type['sigma']} nm seems too large")
            if atom_type['epsilon'] < 0:
                errors.append(f"Atom type {i}: Epsilon must be non-negative")
                
        # Validate bond types
        for i, bond_type in enumerate(data.get('bond_types', [])):
            if bond_type['k'] <= 0:
                errors.append(f"Bond type {i}: Force constant must be positive")
            if bond_type['r0'] <= 0:
                errors.append(f"Bond type {i}: Equilibrium length must be positive")
            if bond_type['r0'] > 1.0:  # 1 nm is very long bond
                errors.append(f"Bond type {i}: Bond length {bond_type['r0']} nm seems too large")
                
        # Validate angle types
        for i, angle_type in enumerate(data.get('angle_types', [])):
            if angle_type['k'] <= 0:
                errors.append(f"Angle type {i}: Force constant must be positive")
            if not (0 <= angle_type['theta0'] <= 180):
                errors.append(f"Angle type {i}: Equilibrium angle must be between 0 and 180 degrees")
                
        # Validate dihedral types
        for i, dihedral_type in enumerate(data.get('dihedral_types', [])):
            if dihedral_type['n'] <= 0:
                errors.append(f"Dihedral type {i}: Periodicity must be positive")
            if not (-180 <= dihedral_type['phase'] <= 180):
                errors.append(f"Dihedral type {i}: Phase must be between -180 and 180 degrees")
                
        if errors:
            raise ValueError("Parameter validation failed:\\n" + "\\n".join(errors))
    
    def _xml_to_dict(self, root: ET.Element) -> Dict:
        """Convert XML tree to dictionary format."""
        data = {
            'metadata': {},
            'atom_types': [],
            'bond_types': [],
            'angle_types': [],
            'dihedral_types': []
        }
        
        # Parse metadata
        metadata_elem = root.find('metadata')
        if metadata_elem is not None:
            for child in metadata_elem:
                if child.tag == 'units':
                    units = {}
                    for unit_child in child:
                        units[unit_child.tag] = unit_child.text
                    data['metadata']['units'] = units
                else:
                    data['metadata'][child.tag] = child.text
        
        # Parse atom types
        atom_types_elem = root.find('atom_types')
        if atom_types_elem is not None:
            for atom_elem in atom_types_elem.findall('atom_type'):
                atom_data = {}
                for child in atom_elem:
                    if child.tag in ['mass', 'sigma', 'epsilon', 'charge']:
                        atom_data[child.tag] = float(child.text)
                    else:
                        atom_data[child.tag] = child.text
                data['atom_types'].append(atom_data)
        
        # Parse bond types
        bond_types_elem = root.find('bond_types')
        if bond_types_elem is not None:
            for bond_elem in bond_types_elem.findall('bond_type'):
                bond_data = {}
                for child in bond_elem:
                    if child.tag == 'atom_types':
                        bond_data['atom_types'] = [t.strip() for t in child.text.split(',')]
                    elif child.tag in ['k', 'r0']:
                        bond_data[child.tag] = float(child.text)
                    else:
                        bond_data[child.tag] = child.text
                data['bond_types'].append(bond_data)
        
        # Parse angle types
        angle_types_elem = root.find('angle_types')
        if angle_types_elem is not None:
            for angle_elem in angle_types_elem.findall('angle_type'):
                angle_data = {}
                for child in angle_elem:
                    if child.tag == 'atom_types':
                        angle_data['atom_types'] = [t.strip() for t in child.text.split(',')]
                    elif child.tag in ['k', 'theta0']:
                        angle_data[child.tag] = float(child.text)
                    else:
                        angle_data[child.tag] = child.text
                data['angle_types'].append(angle_data)
        
        # Parse dihedral types
        dihedral_types_elem = root.find('dihedral_types')
        if dihedral_types_elem is not None:
            for dihedral_elem in dihedral_types_elem.findall('dihedral_type'):
                dihedral_data = {}
                for child in dihedral_elem:
                    if child.tag == 'atom_types':
                        dihedral_data['atom_types'] = [t.strip() for t in child.text.split(',')]
                    elif child.tag in ['k', 'phase']:
                        dihedral_data[child.tag] = float(child.text)
                    elif child.tag == 'n':
                        dihedral_data[child.tag] = int(child.text)
                    else:
                        dihedral_data[child.tag] = child.text
                data['dihedral_types'].append(dihedral_data)
        
        return data

class CustomForceField(ForceField):
    """Custom force field implementation that supports user-defined parameters."""
    
    def __init__(self,
                 name: str = "CustomForceField",
                 cutoff: float = 1.0,
                 switch_distance: Optional[float] = None,
                 nonbonded_method: NonbondedMethod = NonbondedMethod.PME):
        """
        Initialize custom force field.
        
        Parameters
        ----------
        name : str, optional
            Name of the custom force field
        cutoff : float, optional
            Cutoff distance for nonbonded interactions in nm
        switch_distance : float, optional
            Distance at which to start switching the LJ potential
        nonbonded_method : NonbondedMethod, optional
            Method for handling nonbonded interactions
        """
        super().__init__(
            name=name,
            cutoff=cutoff,
            switch_distance=switch_distance,
            nonbonded_method=nonbonded_method
        )
        
        self.forcefield_type = ForceFieldType.CUSTOM
        self.metadata = {}
        self.custom_atom_types = {}
        self.custom_bond_types = {}
        self.custom_angle_types = {}
        self.custom_dihedral_types = {}
        
    def load_from_data(self, data: Dict):
        """Load force field parameters from dictionary data."""
        self.metadata = data.get('metadata', {})
        self.name = self.metadata.get('name', 'CustomForceField')
        
        # Load atom types
        for atom_data in data.get('atom_types', []):
            atom_type = CustomAtomTypeParameters(
                atom_type=atom_data['atom_type'],
                mass=atom_data['mass'],
                sigma=atom_data['sigma'],
                epsilon=atom_data['epsilon'],
                charge=atom_data.get('charge'),
                description=atom_data.get('description'),
                source=atom_data.get('source')
            )
            self.custom_atom_types[atom_type.atom_type] = atom_type
            
            # Add to base ForceField atom_types
            self.atom_types[atom_type.atom_type] = {
                'mass': atom_type.mass,
                'sigma': atom_type.sigma,
                'epsilon': atom_type.epsilon,
                'charge': atom_type.charge or 0.0
            }
        
        # Load bond types
        for bond_data in data.get('bond_types', []):
            bond_type = CustomBondParameters(
                atom_types=tuple(bond_data['atom_types']),
                k=bond_data['k'],
                r0=bond_data['r0'],
                description=bond_data.get('description'),
                source=bond_data.get('source')
            )
            key = tuple(sorted(bond_type.atom_types))
            self.custom_bond_types[key] = bond_type
            
            # Add to base ForceField bond_types
            self.bond_types[key] = {
                'k': bond_type.k,
                'r0': bond_type.r0
            }
        
        # Load angle types
        for angle_data in data.get('angle_types', []):
            angle_type = CustomAngleParameters(
                atom_types=tuple(angle_data['atom_types']),
                k=angle_data['k'],
                theta0=angle_data['theta0'],
                description=angle_data.get('description'),
                source=angle_data.get('source')
            )
            key = tuple(angle_type.atom_types)
            self.custom_angle_types[key] = angle_type
            
            # Add to base ForceField angle_types
            self.angle_types[key] = {
                'k': angle_type.k,
                'theta0': np.radians(angle_type.theta0)  # Convert to radians
            }
        
        # Load dihedral types
        for dihedral_data in data.get('dihedral_types', []):
            dihedral_type = CustomDihedralParameters(
                atom_types=tuple(dihedral_data['atom_types']),
                k=dihedral_data['k'],
                n=dihedral_data['n'],
                phase=dihedral_data['phase'],
                description=dihedral_data.get('description'),
                source=dihedral_data.get('source')
            )
            key = tuple(dihedral_type.atom_types)
            self.custom_dihedral_types[key] = dihedral_type
            
            # Add to base ForceField dihedral_types
            self.dihedral_types[key] = {
                'k': dihedral_type.k,
                'n': dihedral_type.n,
                'phase': np.radians(dihedral_type.phase)  # Convert to radians
            }
    
    def get_atom_parameters(self, atom_type: str) -> Optional[Dict]:
        """Get parameters for a specific atom type."""
        return self.atom_types.get(atom_type)
    
    def get_bond_parameters(self, atom_type1: str, atom_type2: str) -> Optional[Dict]:
        """Get bond parameters for a pair of atom types."""
        key1 = tuple(sorted([atom_type1, atom_type2]))
        return self.bond_types.get(key1)
    
    def get_angle_parameters(self, atom_type1: str, atom_type2: str, atom_type3: str) -> Optional[Dict]:
        """Get angle parameters for a triplet of atom types."""
        key = (atom_type1, atom_type2, atom_type3)
        return self.angle_types.get(key)
    
    def get_dihedral_parameters(self, atom_type1: str, atom_type2: str, 
                              atom_type3: str, atom_type4: str) -> Optional[Dict]:
        """Get dihedral parameters for a quartet of atom types."""
        key = (atom_type1, atom_type2, atom_type3, atom_type4)
        return self.dihedral_types.get(key)
    
    def get_metadata(self) -> Dict:
        """Get force field metadata."""
        return self.metadata.copy()
    
    def export_to_json(self, filename: Union[str, Path]):
        """Export the current force field to a JSON file."""
        filename = Path(filename)
        
        data = {
            'metadata': self.metadata,
            'atom_types': [asdict(atom) for atom in self.custom_atom_types.values()],
            'bond_types': [asdict(bond) for bond in self.custom_bond_types.values()],
            'angle_types': [asdict(angle) for angle in self.custom_angle_types.values()],
            'dihedral_types': [asdict(dihedral) for dihedral in self.custom_dihedral_types.values()]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Custom force field exported to: {filename}")

# Convenience function for easy import
def import_custom_forcefield(filename: Union[str, Path], 
                           validate_schema: bool = True) -> CustomForceField:
    """
    Import a custom force field from a JSON or XML file.
    
    Parameters
    ----------
    filename : str or Path
        Path to the force field file
    validate_schema : bool, optional
        Whether to validate JSON schema (default: True)
        
    Returns
    -------
    CustomForceField
        The imported custom force field
    """
    importer = CustomForceFieldImporter(validate_schema=validate_schema)
    
    filename = Path(filename)
    if filename.suffix.lower() == '.json':
        return importer.import_from_json(filename)
    elif filename.suffix.lower() == '.xml':
        return importer.import_from_xml(filename)
    else:
        raise ValueError(f"Unsupported file format: {filename.suffix}. Use .json or .xml")

logger.info("Custom force field module successfully imported")
