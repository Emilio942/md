"""
Custom Force Field Import Module

This module provides functionality to import custom force field parameters
from JSON or XML files with comprehensive validation and error handling.
"""

import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
from dataclasses import dataclass, field
import numpy as np
from enum import Enum

from .forcefield import ForceField, ForceFieldType, NonbondedMethod

logger = logging.getLogger(__name__)

class ParameterFormat(Enum):
    """Supported parameter file formats."""
    JSON = "json"
    XML = "xml"

@dataclass
class CustomAtomType:
    """Custom atom type parameters."""
    name: str
    mass: float  # atomic mass units (amu)
    charge: float = 0.0  # elementary charge units
    sigma: float = 0.0  # LJ sigma parameter (nm)
    epsilon: float = 0.0  # LJ epsilon parameter (kJ/mol)
    description: str = ""

@dataclass  
class CustomBondType:
    """Custom bond type parameters."""
    atom_types: tuple  # (atom_type1, atom_type2)
    length: float  # equilibrium bond length (nm)
    k: float  # force constant (kJ/mol/nm²)
    description: str = ""

@dataclass
class CustomAngleType:
    """Custom angle type parameters."""
    atom_types: tuple  # (atom_type1, atom_type2, atom_type3)
    angle: float  # equilibrium angle (radians)
    k: float  # force constant (kJ/mol/rad²)
    description: str = ""

@dataclass
class CustomDihedralType:
    """Custom dihedral type parameters."""
    atom_types: tuple  # (atom_type1, atom_type2, atom_type3, atom_type4)
    periodicity: int  # periodicity (n)
    phase: float  # phase angle (radians)
    k: float  # force constant (kJ/mol)
    description: str = ""

@dataclass
class CustomResidueTemplate:
    """Custom residue template."""
    name: str
    atoms: List[Dict[str, Any]] = field(default_factory=list)
    bonds: List[tuple] = field(default_factory=list)
    angles: List[tuple] = field(default_factory=list) 
    dihedrals: List[tuple] = field(default_factory=list)
    description: str = ""

class ValidationError(Exception):
    """Custom exception for parameter validation errors."""
    pass

class CustomForceField(ForceField):
    """
    Custom force field implementation supporting user-defined parameters.
    
    This class allows users to define their own force field parameters
    using JSON or XML configuration files.
    """
    
    def __init__(self, 
                 name: str = "Custom",
                 parameter_file: Optional[str] = None,
                 format: ParameterFormat = ParameterFormat.JSON,
                 **kwargs):
        """
        Initialize custom force field.
        
        Parameters
        ----------
        name : str
            Name of the custom force field
        parameter_file : str, optional
            Path to parameter file
        format : ParameterFormat
            Format of the parameter file (JSON or XML)
        **kwargs
            Additional arguments passed to ForceField base class
        """
        super().__init__(name, **kwargs)
        
        # Custom parameter storage
        self.custom_atom_types: Dict[str, CustomAtomType] = {}
        self.custom_bond_types: Dict[tuple, CustomBondType] = {}
        self.custom_angle_types: Dict[tuple, CustomAngleType] = {}
        self.custom_dihedral_types: Dict[tuple, CustomDihedralType] = {}
        self.custom_residue_templates: Dict[str, CustomResidueTemplate] = {}
        
        self.parameter_format = format
        self.parameter_file_path = parameter_file
        
        # Load parameters if file provided
        if parameter_file:
            self.load_parameters(parameter_file, format)
            
        logger.info(f"Initialized custom force field '{name}' with {format.value} format")
    
    def load_parameters(self, parameter_file: str, format: ParameterFormat = None) -> None:
        """
        Load custom parameters from file.
        
        Parameters
        ----------
        parameter_file : str
            Path to parameter file
        format : ParameterFormat, optional
            File format, auto-detected if not provided
        
        Raises
        ------
        ValidationError
            If parameter validation fails
        FileNotFoundError
            If parameter file not found
        """
        param_path = Path(parameter_file)
        if not param_path.exists():
            raise FileNotFoundError(f"Parameter file not found: {parameter_file}")
        
        # Auto-detect format if not provided
        if format is None:
            if param_path.suffix.lower() == '.json':
                format = ParameterFormat.JSON
            elif param_path.suffix.lower() in ['.xml', '.pff']:
                format = ParameterFormat.XML
            else:
                raise ValidationError(f"Unknown file format: {param_path.suffix}")
        
        self.parameter_format = format
        self.parameter_file_path = parameter_file
        
        try:
            if format == ParameterFormat.JSON:
                self._load_json_parameters(param_path)
            elif format == ParameterFormat.XML:
                self._load_xml_parameters(param_path)
            else:
                raise ValidationError(f"Unsupported format: {format}")
                
            self._validate_loaded_parameters()
            logger.info(f"Successfully loaded parameters from {parameter_file}")
            
        except Exception as e:
            logger.error(f"Failed to load parameters from {parameter_file}: {e}")
            raise ValidationError(f"Parameter loading failed: {e}")
    
    def _load_json_parameters(self, param_path: Path) -> None:
        """Load parameters from JSON file."""
        with open(param_path, 'r') as f:
            data = json.load(f)
        
        # Validate JSON structure
        self._validate_json_structure(data)
        
        # Load atom types
        if 'atom_types' in data:
            for atom_data in data['atom_types']:
                atom_type = CustomAtomType(
                    name=atom_data['name'],
                    mass=float(atom_data['mass']),
                    charge=float(atom_data.get('charge', 0.0)),
                    sigma=float(atom_data.get('sigma', 0.0)),
                    epsilon=float(atom_data.get('epsilon', 0.0)),
                    description=atom_data.get('description', '')
                )
                self.custom_atom_types[atom_type.name] = atom_type
        
        # Load bond types
        if 'bond_types' in data:
            for bond_data in data['bond_types']:
                atom_types = tuple(bond_data['atom_types'])
                bond_type = CustomBondType(
                    atom_types=atom_types,
                    length=float(bond_data['length']),
                    k=float(bond_data['k']),
                    description=bond_data.get('description', '')
                )
                self.custom_bond_types[atom_types] = bond_type
                # Also add reverse order for symmetric lookup
                self.custom_bond_types[atom_types[::-1]] = bond_type
        
        # Load angle types
        if 'angle_types' in data:
            for angle_data in data['angle_types']:
                atom_types = tuple(angle_data['atom_types'])
                angle_type = CustomAngleType(
                    atom_types=atom_types,
                    angle=float(angle_data['angle']),
                    k=float(angle_data['k']),
                    description=angle_data.get('description', '')
                )
                self.custom_angle_types[atom_types] = angle_type
                # Also add reverse order
                self.custom_angle_types[atom_types[::-1]] = angle_type
        
        # Load dihedral types
        if 'dihedral_types' in data:
            for dihedral_data in data['dihedral_types']:
                atom_types = tuple(dihedral_data['atom_types'])
                dihedral_type = CustomDihedralType(
                    atom_types=atom_types,
                    periodicity=int(dihedral_data['periodicity']),
                    phase=float(dihedral_data['phase']),
                    k=float(dihedral_data['k']),
                    description=dihedral_data.get('description', '')
                )
                self.custom_dihedral_types[atom_types] = dihedral_type
                # Also add reverse order
                self.custom_dihedral_types[atom_types[::-1]] = dihedral_type
        
        # Load residue templates
        if 'residue_templates' in data:
            for residue_data in data['residue_templates']:
                residue = CustomResidueTemplate(
                    name=residue_data['name'],
                    atoms=residue_data.get('atoms', []),
                    bonds=residue_data.get('bonds', []),
                    angles=residue_data.get('angles', []),
                    dihedrals=residue_data.get('dihedrals', []),
                    description=residue_data.get('description', '')
                )
                self.custom_residue_templates[residue.name] = residue
    
    def _load_xml_parameters(self, param_path: Path) -> None:
        """Load parameters from XML file."""
        tree = ET.parse(param_path)
        root = tree.getroot()
        
        # Load atom types
        for atom_elem in root.findall('.//AtomType'):
            atom_type = CustomAtomType(
                name=atom_elem.get('name'),
                mass=float(atom_elem.get('mass')),
                charge=float(atom_elem.get('charge', 0.0)),
                sigma=float(atom_elem.get('sigma', 0.0)),
                epsilon=float(atom_elem.get('epsilon', 0.0)),
                description=atom_elem.get('description', '')
            )
            self.custom_atom_types[atom_type.name] = atom_type
        
        # Load bond types
        for bond_elem in root.findall('.//BondType'):
            atom_types = tuple(bond_elem.get('class').split('-'))
            bond_type = CustomBondType(
                atom_types=atom_types,
                length=float(bond_elem.get('length')),
                k=float(bond_elem.get('k')),
                description=bond_elem.get('description', '')
            )
            self.custom_bond_types[atom_types] = bond_type
            self.custom_bond_types[atom_types[::-1]] = bond_type
        
        # Load angle types
        for angle_elem in root.findall('.//AngleType'):
            atom_types = tuple(angle_elem.get('class').split('-'))
            angle_type = CustomAngleType(
                atom_types=atom_types,
                angle=float(angle_elem.get('angle')),
                k=float(angle_elem.get('k')),
                description=angle_elem.get('description', '')
            )
            self.custom_angle_types[atom_types] = angle_type
            self.custom_angle_types[atom_types[::-1]] = angle_type
        
        # Load dihedral types
        for dihedral_elem in root.findall('.//DihedralType'):
            atom_types = tuple(dihedral_elem.get('class').split('-'))
            dihedral_type = CustomDihedralType(
                atom_types=atom_types,
                periodicity=int(dihedral_elem.get('periodicity')),
                phase=float(dihedral_elem.get('phase')),
                k=float(dihedral_elem.get('k')),
                description=dihedral_elem.get('description', '')
            )
            self.custom_dihedral_types[atom_types] = dihedral_type
            self.custom_dihedral_types[atom_types[::-1]] = dihedral_type
    
    def _validate_json_structure(self, data: Dict) -> None:
        """Validate JSON parameter file structure."""
        if not isinstance(data, dict):
            raise ValidationError("Root element must be a JSON object")
        
        # Check for required sections
        valid_sections = {'atom_types', 'bond_types', 'angle_types', 'dihedral_types', 'residue_templates', 'metadata'}
        if not any(section in data for section in valid_sections):
            raise ValidationError(f"At least one parameter section required: {valid_sections}")
        
        # Validate atom types structure
        if 'atom_types' in data:
            if not isinstance(data['atom_types'], list):
                raise ValidationError("atom_types must be a list")
            for i, atom in enumerate(data['atom_types']):
                if not isinstance(atom, dict):
                    raise ValidationError(f"atom_types[{i}] must be an object")
                if 'name' not in atom or 'mass' not in atom:
                    raise ValidationError(f"atom_types[{i}] missing required fields: name, mass")
        
        # Validate bond types structure
        if 'bond_types' in data:
            if not isinstance(data['bond_types'], list):
                raise ValidationError("bond_types must be a list")
            for i, bond in enumerate(data['bond_types']):
                if not isinstance(bond, dict):
                    raise ValidationError(f"bond_types[{i}] must be an object")
                if not all(field in bond for field in ['atom_types', 'length', 'k']):
                    raise ValidationError(f"bond_types[{i}] missing required fields")
                if len(bond['atom_types']) != 2:
                    raise ValidationError(f"bond_types[{i}] atom_types must have exactly 2 elements")
    
    def _validate_loaded_parameters(self) -> None:
        """Validate loaded parameters for consistency and physical reasonableness."""
        validation_errors = []
        
        # Validate atom types
        for name, atom_type in self.custom_atom_types.items():
            if atom_type.mass <= 0:
                validation_errors.append(f"Atom type {name}: mass must be positive")
            if atom_type.sigma < 0:
                validation_errors.append(f"Atom type {name}: sigma must be non-negative")
            if atom_type.epsilon < 0:
                validation_errors.append(f"Atom type {name}: epsilon must be non-negative")
        
        # Validate bond types
        for atom_types, bond_type in self.custom_bond_types.items():
            if bond_type.length <= 0:
                validation_errors.append(f"Bond {atom_types}: length must be positive")
            if bond_type.k < 0:
                validation_errors.append(f"Bond {atom_types}: force constant must be non-negative")
            
            # Check if atom types exist
            for atom_type in atom_types:
                if atom_type not in self.custom_atom_types:
                    validation_errors.append(f"Bond {atom_types}: unknown atom type {atom_type}")
        
        # Validate angle types
        for atom_types, angle_type in self.custom_angle_types.items():
            if not (0 <= angle_type.angle <= np.pi):
                validation_errors.append(f"Angle {atom_types}: angle must be between 0 and π")
            if angle_type.k < 0:
                validation_errors.append(f"Angle {atom_types}: force constant must be non-negative")
        
        # Validate dihedral types
        for atom_types, dihedral_type in self.custom_dihedral_types.items():
            if dihedral_type.periodicity <= 0:
                validation_errors.append(f"Dihedral {atom_types}: periodicity must be positive")
            if not (0 <= dihedral_type.phase <= 2*np.pi):
                validation_errors.append(f"Dihedral {atom_types}: phase must be between 0 and 2π")
        
        if validation_errors:
            raise ValidationError("Parameter validation failed:\n" + "\n".join(validation_errors))
    
    def validate_protein_parameters(self, protein_data: Dict) -> Dict:
        """
        Validate that all required parameters are available for a protein.
        
        Parameters
        ----------
        protein_data : Dict
            Protein structure data
            
        Returns
        -------
        Dict
            Validation results
        """
        validation_result = {
            "is_valid": True,
            "missing_atom_types": set(),
            "missing_bond_parameters": set(),
            "missing_angle_parameters": set(),
            "missing_dihedral_parameters": set(),
            "coverage_statistics": {},
            "validation_errors": []
        }
        
        # Check atom type coverage
        for atom in protein_data.get("atoms", []):
            atom_type = atom.get("atom_type", "")
            if atom_type not in self.custom_atom_types:
                validation_result["missing_atom_types"].add(atom_type)
                validation_result["is_valid"] = False
        
        # Check bond parameter coverage
        for bond in protein_data.get("bonds", []):
            if len(bond) >= 2:
                atom1_type = protein_data["atoms"][bond[0]].get("atom_type", "")
                atom2_type = protein_data["atoms"][bond[1]].get("atom_type", "")
                bond_key = (atom1_type, atom2_type)
                if bond_key not in self.custom_bond_types:
                    validation_result["missing_bond_parameters"].add(bond_key)
                    validation_result["is_valid"] = False
        
        # Calculate coverage statistics
        total_atom_types = len(set(atom.get("atom_type", "") for atom in protein_data.get("atoms", [])))
        covered_atom_types = total_atom_types - len(validation_result["missing_atom_types"])
        
        validation_result["coverage_statistics"] = {
            "atom_type_coverage": 100.0 * covered_atom_types / max(1, total_atom_types),
            "total_atom_types": total_atom_types,
            "covered_atom_types": covered_atom_types
        }
        
        return validation_result
    
    def assign_parameters_to_protein(self, protein_data: Dict) -> Dict:
        """
        Assign custom force field parameters to a protein.
        
        Parameters
        ----------
        protein_data : Dict
            Protein structure data
            
        Returns
        -------
        Dict
            Protein data with assigned parameters
        """
        assigned_protein = protein_data.copy()
        assigned_protein["force_field"] = self.name
        assigned_protein["custom_parameters"] = {}
        
        # Assign atom type parameters
        atom_params = []
        for atom in assigned_protein.get("atoms", []):
            atom_type = atom.get("atom_type", "")
            if atom_type in self.custom_atom_types:
                params = self.custom_atom_types[atom_type]
                atom_params.append({
                    "atom_id": atom.get("id"),
                    "atom_type": atom_type,
                    "mass": params.mass,
                    "charge": params.charge,
                    "sigma": params.sigma,
                    "epsilon": params.epsilon
                })
        
        assigned_protein["custom_parameters"]["atoms"] = atom_params
        
        # Assign bond parameters
        bond_params = []
        for bond in assigned_protein.get("bonds", []):
            if len(bond) >= 2:
                atom1_type = assigned_protein["atoms"][bond[0]].get("atom_type", "")
                atom2_type = assigned_protein["atoms"][bond[1]].get("atom_type", "")
                bond_key = (atom1_type, atom2_type)
                
                if bond_key in self.custom_bond_types:
                    params = self.custom_bond_types[bond_key]
                    bond_params.append({
                        "bond": bond,
                        "atom_types": bond_key,
                        "length": params.length,
                        "k": params.k
                    })
        
        assigned_protein["custom_parameters"]["bonds"] = bond_params
        
        return assigned_protein
    
    def export_parameters(self, output_file: str, format: ParameterFormat = ParameterFormat.JSON) -> None:
        """
        Export current parameters to file.
        
        Parameters
        ----------
        output_file : str
            Output file path
        format : ParameterFormat
            Output format
        """
        if format == ParameterFormat.JSON:
            self._export_json(output_file)
        elif format == ParameterFormat.XML:
            self._export_xml(output_file)
        else:
            raise ValidationError(f"Unsupported export format: {format}")
    
    def _export_json(self, output_file: str) -> None:
        """Export parameters as JSON."""
        data = {
            "metadata": {
                "force_field_name": self.name,
                "created_by": "proteinMD CustomForceField",
                "format_version": "1.0"
            },
            "atom_types": [
                {
                    "name": atom.name,
                    "mass": atom.mass,
                    "charge": atom.charge,
                    "sigma": atom.sigma,
                    "epsilon": atom.epsilon,
                    "description": atom.description
                }
                for atom in self.custom_atom_types.values()
            ],
            "bond_types": [
                {
                    "atom_types": list(bond.atom_types),
                    "length": bond.length,
                    "k": bond.k,
                    "description": bond.description
                }
                for bond in self.custom_bond_types.values()
            ],
            "angle_types": [
                {
                    "atom_types": list(angle.atom_types),
                    "angle": angle.angle,
                    "k": angle.k,
                    "description": angle.description
                }
                for angle in self.custom_angle_types.values()
            ],
            "dihedral_types": [
                {
                    "atom_types": list(dihedral.atom_types),
                    "periodicity": dihedral.periodicity,
                    "phase": dihedral.phase,
                    "k": dihedral.k,
                    "description": dihedral.description
                }
                for dihedral in self.custom_dihedral_types.values()
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported parameters to {output_file}")
    
    def _export_xml(self, output_file: str) -> None:
        """Export parameters as XML."""
        root = ET.Element("ForceField")
        root.set("name", self.name)
        
        # Export atom types
        atom_types_elem = ET.SubElement(root, "AtomTypes")
        for atom in self.custom_atom_types.values():
            atom_elem = ET.SubElement(atom_types_elem, "AtomType")
            atom_elem.set("name", atom.name)
            atom_elem.set("mass", str(atom.mass))
            atom_elem.set("charge", str(atom.charge))
            atom_elem.set("sigma", str(atom.sigma))
            atom_elem.set("epsilon", str(atom.epsilon))
            if atom.description:
                atom_elem.set("description", atom.description)
        
        # Export bond types
        bond_types_elem = ET.SubElement(root, "BondTypes")
        for bond in self.custom_bond_types.values():
            bond_elem = ET.SubElement(bond_types_elem, "BondType")
            bond_elem.set("class", "-".join(bond.atom_types))
            bond_elem.set("length", str(bond.length))
            bond_elem.set("k", str(bond.k))
            if bond.description:
                bond_elem.set("description", bond.description)
        
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        
        logger.info(f"Exported parameters to {output_file}")
    
    def get_supported_atom_types(self) -> List[str]:
        """Get list of supported atom types."""
        return list(self.custom_atom_types.keys())
    
    def get_parameter_summary(self) -> Dict:
        """Get summary of loaded parameters."""
        return {
            "force_field_name": self.name,
            "parameter_file": self.parameter_file_path,
            "format": self.parameter_format.value if self.parameter_format else None,
            "atom_types": len(self.custom_atom_types),
            "bond_types": len(self.custom_bond_types),
            "angle_types": len(self.custom_angle_types), 
            "dihedral_types": len(self.custom_dihedral_types),
            "residue_templates": len(self.custom_residue_templates)
        }

def create_parameter_template(output_file: str, format: ParameterFormat = ParameterFormat.JSON) -> None:
    """
    Create a template parameter file with example parameters.
    
    Parameters
    ----------
    output_file : str
        Output file path
    format : ParameterFormat
        Template format
    """
    if format == ParameterFormat.JSON:
        template_data = {
            "metadata": {
                "force_field_name": "Custom Example",
                "description": "Example custom force field parameters",
                "created_by": "proteinMD CustomForceField",
                "format_version": "1.0"
            },
            "atom_types": [
                {
                    "name": "CA",
                    "mass": 12.01,
                    "charge": 0.0,
                    "sigma": 0.339967,
                    "epsilon": 0.359824,
                    "description": "Carbon alpha"
                },
                {
                    "name": "CB",
                    "mass": 12.01,
                    "charge": 0.0,
                    "sigma": 0.339967,
                    "epsilon": 0.359824,
                    "description": "Carbon beta"
                },
                {
                    "name": "NH",
                    "mass": 14.007,
                    "charge": -0.4,
                    "sigma": 0.325,
                    "epsilon": 0.71128,
                    "description": "Nitrogen amide"
                }
            ],
            "bond_types": [
                {
                    "atom_types": ["CA", "CB"],
                    "length": 0.1529,
                    "k": 259408.0,
                    "description": "CA-CB bond"
                },
                {
                    "atom_types": ["CA", "NH"],
                    "length": 0.1449,
                    "k": 282001.6,
                    "description": "CA-NH bond"
                }
            ],
            "angle_types": [
                {
                    "atom_types": ["NH", "CA", "CB"],
                    "angle": 1.9146,
                    "k": 418.4,
                    "description": "NH-CA-CB angle"
                }
            ],
            "dihedral_types": [
                {
                    "atom_types": ["NH", "CA", "CB", "NH"],
                    "periodicity": 3,
                    "phase": 0.0,
                    "k": 4.6024,
                    "description": "Protein backbone dihedral"
                }
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(template_data, f, indent=2)
    
    elif format == ParameterFormat.XML:
        root = ET.Element("ForceField")
        root.set("name", "Custom Example")
        
        # Atom types
        atom_types_elem = ET.SubElement(root, "AtomTypes")
        
        ca_elem = ET.SubElement(atom_types_elem, "AtomType")
        ca_elem.set("name", "CA")
        ca_elem.set("mass", "12.01")
        ca_elem.set("charge", "0.0")
        ca_elem.set("sigma", "0.339967")
        ca_elem.set("epsilon", "0.359824")
        ca_elem.set("description", "Carbon alpha")
        
        # Bond types
        bond_types_elem = ET.SubElement(root, "BondTypes")
        
        bond_elem = ET.SubElement(bond_types_elem, "BondType")
        bond_elem.set("class", "CA-CB")
        bond_elem.set("length", "0.1529")
        bond_elem.set("k", "259408.0")
        bond_elem.set("description", "CA-CB bond")
        
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    logger.info(f"Created parameter template: {output_file}")
