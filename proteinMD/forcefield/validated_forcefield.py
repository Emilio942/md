"""
Enhanced Force Field module with AMBER parameter validation.

This module extends the base force field functionality with comprehensive
AMBER parameter validation and automatic parameter checking.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import sys

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Import AMBER validation
try:
    from proteinMD.forcefield.amber_validator import (
        AMBERParameterValidator, 
        ValidationResult,
        AtomTypeParameters,
        amber_validator
    )
    AMBER_VALIDATION_AVAILABLE = True
    logger.info("AMBER validation module loaded successfully")
except ImportError as e:
    logger.warning(f"AMBER validation not available: {e}")
    AMBER_VALIDATION_AVAILABLE = False

class ValidatedForceField:
    """
    Enhanced force field class with comprehensive parameter validation.
    
    This class extends the basic force field functionality with automatic
    parameter validation, missing parameter detection, and comprehensive
    reporting for AMBER force fields.
    """
    
    def __init__(self, 
                 name: str = "AMBER_ff14SB",
                 validate_on_init: bool = True,
                 auto_fix_missing: bool = False):
        """
        Initialize a validated force field.
        
        Parameters
        ----------
        name : str
            Name of the force field
        validate_on_init : bool
            Whether to validate parameters on initialization
        auto_fix_missing : bool
            Whether to attempt automatic fixing of missing parameters
        """
        self.name = name
        self.validate_on_init = validate_on_init
        self.auto_fix_missing = auto_fix_missing
        
        # Initialize AMBER validator
        if AMBER_VALIDATION_AVAILABLE:
            self.validator = AMBERParameterValidator()
            logger.info("✅ AMBER parameter validator initialized")
        else:
            self.validator = None
            logger.warning("❌ AMBER parameter validator not available")
        
        # Validation statistics
        self.validation_stats = {
            'proteins_tested': 0,
            'proteins_passed': 0,
            'total_atom_types_checked': 0,
            'missing_atom_types': set(),
            'validation_errors': []
        }
        
        logger.info(f"Initialized {name} force field with validation")
    
    def validate_protein(self, protein_structure, protein_name: str = "Unknown") -> ValidationResult:
        """
        Validate AMBER parameters for a protein structure.
        
        Parameters
        ----------
        protein_structure : object
            Protein structure with atoms attribute
        protein_name : str
            Name of the protein for reporting
            
        Returns
        -------
        ValidationResult
            Comprehensive validation results
        """
        if not self.validator:
            logger.error("❌ No validator available - cannot validate protein")
            result = ValidationResult()
            result.add_error("AMBER validation not available")
            return result
        
        logger.info(f"🔍 Validating protein: {protein_name}")
        
        # Perform validation
        result = self.validator.validate_protein_parameters(protein_structure)
        
        # Update statistics
        self.validation_stats['proteins_tested'] += 1
        if result.is_valid:
            self.validation_stats['proteins_passed'] += 1
            logger.info(f"✅ {protein_name}: All parameters valid")
        else:
            logger.warning(f"❌ {protein_name}: Validation failed")
            self.validation_stats['missing_atom_types'].update(result.missing_atom_types)
            self.validation_stats['validation_errors'].extend(result.errors)
        
        # Count unique atom types
        if hasattr(protein_structure, 'atoms'):
            atom_types = set()
            for atom in protein_structure.atoms:
                if hasattr(atom, 'atom_name'):
                    atom_types.add(atom.atom_name)
            self.validation_stats['total_atom_types_checked'] += len(atom_types)
        
        return result
    
    def batch_validate_proteins(self, proteins: List[Tuple[object, str]]) -> Dict[str, ValidationResult]:
        """
        Validate multiple proteins in batch.
        
        Parameters
        ----------
        proteins : list
            List of (protein_structure, protein_name) tuples
            
        Returns
        -------
        dict
            Dictionary mapping protein names to validation results
        """
        logger.info(f"🧬 Starting batch validation of {len(proteins)} proteins")
        
        results = {}
        
        for protein_structure, protein_name in proteins:
            try:
                result = self.validate_protein(protein_structure, protein_name)
                results[protein_name] = result
            except Exception as e:
                logger.error(f"❌ Error validating {protein_name}: {e}")
                error_result = ValidationResult()
                error_result.add_error(f"Validation error: {e}")
                results[protein_name] = error_result
        
        # Generate summary
        passed = sum(1 for r in results.values() if r.is_valid)
        total = len(results)
        
        logger.info(f"📊 Batch validation complete: {passed}/{total} proteins passed")
        
        return results
    
    def check_standard_proteins(self) -> bool:
        """
        Check if the force field can handle standard proteins.
        
        This method creates test structures for common amino acids
        and validates their parameters.
        
        Returns
        -------
        bool
            True if all standard proteins can be handled
        """
        logger.info("🧪 Testing standard protein compatibility...")
        
        # Create test structures for all 20 standard amino acids
        standard_proteins = self._create_standard_protein_tests()
        
        # Validate all test proteins
        results = self.batch_validate_proteins(standard_proteins)
        
        # Check if all passed
        all_passed = all(result.is_valid for result in results.values())
        
        if all_passed:
            logger.info("✅ All standard proteins can be simulated without errors")
        else:
            failed = [name for name, result in results.items() if not result.is_valid]
            logger.warning(f"❌ Standard protein validation failed for: {failed}")
        
        return all_passed
    
    def _create_standard_protein_tests(self) -> List[Tuple[object, str]]:
        """Create test protein structures for all standard amino acids."""
        
        class TestAtom:
            def __init__(self, atom_id, atom_name, element):
                self.atom_id = atom_id
                self.atom_name = atom_name
                self.element = element
        
        class TestProtein:
            def __init__(self, name):
                self.name = name
                self.atoms = []
        
        # Define standard amino acid atom types
        amino_acid_atoms = {
            'ALA': ['N', 'H', 'CA', 'HA', 'CB', 'HB1', 'HB2', 'HB3', 'C', 'O'],
            'ARG': ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 
                   'CD', 'HD2', 'HD3', 'NE', 'HE', 'CZ', 'NH1', 'HH11', 'HH12', 'NH2', 'HH21', 'HH22', 'C', 'O'],
            'ASN': ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'OD1', 'ND2', 'HD21', 'HD22', 'C', 'O'],
            'ASP': ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'OD1', 'OD2', 'C', 'O'],
            'CYS': ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'SG', 'HG', 'C', 'O'],
            'GLN': ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'OE1', 'NE2', 'HE21', 'HE22', 'C', 'O'],
            'GLU': ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'OE1', 'OE2', 'C', 'O'],
            'GLY': ['N', 'H', 'CA', 'HA2', 'HA3', 'C', 'O'],
            'HIS': ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'ND1', 'HD1', 'CE1', 'HE1', 'NE2', 'CD2', 'HD2', 'C', 'O'],
            'ILE': ['N', 'H', 'CA', 'HA', 'CB', 'HB', 'CG1', 'HG12', 'HG13', 'CG2', 'HG21', 'HG22', 'HG23', 'CD1', 'HD11', 'HD12', 'HD13', 'C', 'O'],
            'LEU': ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG', 'CD1', 'HD11', 'HD12', 'HD13', 'CD2', 'HD21', 'HD22', 'HD23', 'C', 'O'],
            'LYS': ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'HD2', 'HD3', 'CE', 'HE2', 'HE3', 'NZ', 'HZ1', 'HZ2', 'HZ3', 'C', 'O'],
            'MET': ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'SD', 'CE', 'HE1', 'HE2', 'HE3', 'C', 'O'],
            'PHE': ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'CD1', 'HD1', 'CE1', 'HE1', 'CZ', 'HZ', 'CE2', 'HE2', 'CD2', 'HD2', 'C', 'O'],
            'PRO': ['N', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'HD2', 'HD3', 'C', 'O'],
            'SER': ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'OG', 'HG', 'C', 'O'],
            'THR': ['N', 'H', 'CA', 'HA', 'CB', 'HB', 'OG1', 'HG1', 'CG2', 'HG21', 'HG22', 'HG23', 'C', 'O'],
            'TRP': ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'CD1', 'HD1', 'NE1', 'HE1', 'CE2', 'CE3', 'HE3', 'CZ2', 'HZ2', 'CZ3', 'HZ3', 'CH2', 'HH2', 'CD2', 'C', 'O'],
            'TYR': ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'CD1', 'HD1', 'CE1', 'HE1', 'CZ', 'OH', 'HH', 'CE2', 'HE2', 'CD2', 'HD2', 'C', 'O'],
            'VAL': ['N', 'H', 'CA', 'HA', 'CB', 'HB', 'CG1', 'HG11', 'HG12', 'HG13', 'CG2', 'HG21', 'HG22', 'HG23', 'C', 'O'],
        }
        
        proteins = []
        
        for residue_name, atom_names in amino_acid_atoms.items():
            protein = TestProtein(f"Test_{residue_name}")
            
            for i, atom_name in enumerate(atom_names):
                # Map atom name to element
                element = 'C'  # default
                if atom_name.startswith('N'):
                    element = 'N'
                elif atom_name.startswith('O'):
                    element = 'O'
                elif atom_name.startswith('H'):
                    element = 'H'
                elif atom_name.startswith('S'):
                    element = 'S'
                
                atom = TestAtom(i, atom_name, element)
                protein.atoms.append(atom)
            
            proteins.append((protein, f"Test_{residue_name}"))
        
        return proteins
    
    def generate_coverage_report(self) -> str:
        """
        Generate a comprehensive coverage report.
        
        Returns
        -------
        str
            Formatted coverage report
        """
        report = []
        report.append("="*80)
        report.append("AMBER FORCE FIELD PARAMETER COVERAGE REPORT")
        report.append("="*80)
        report.append("")
        
        # Validation statistics
        stats = self.validation_stats
        report.append(f"Proteins tested: {stats['proteins_tested']}")
        report.append(f"Proteins passed: {stats['proteins_passed']}")
        if stats['proteins_tested'] > 0:
            success_rate = 100 * stats['proteins_passed'] / stats['proteins_tested']
            report.append(f"Success rate: {success_rate:.1f}%")
        report.append(f"Total atom types checked: {stats['total_atom_types_checked']}")
        report.append("")
        
        # Missing parameters
        if stats['missing_atom_types']:
            report.append("MISSING ATOM TYPES:")
            report.append("-" * 40)
            for atom_type in sorted(stats['missing_atom_types']):
                if self.validator:
                    suggestions = self.validator.suggest_similar_parameters(atom_type)
                    report.append(f"  {atom_type} -> Suggestions: {', '.join(suggestions)}")
                else:
                    report.append(f"  {atom_type}")
            report.append("")
        
        # Validation errors
        if stats['validation_errors']:
            report.append("VALIDATION ERRORS:")
            report.append("-" * 40)
            for error in stats['validation_errors'][:10]:  # Show first 10
                report.append(f"  - {error}")
            if len(stats['validation_errors']) > 10:
                report.append(f"  ... and {len(stats['validation_errors']) - 10} more errors")
            report.append("")
        
        # AMBER parameter database info
        if self.validator:
            report.append("AMBER PARAMETER DATABASE:")
            report.append("-" * 40)
            report.append(f"  Atom types available: {len(self.validator.atom_types)}")
            report.append(f"  Bond types available: {len(self.validator.bond_types)}")
            report.append("")
        
        report.append("="*80)
        
        return "\\n".join(report)
    
    def get_validation_summary(self) -> Dict:
        """
        Get a summary of validation statistics.
        
        Returns
        -------
        dict
            Validation summary statistics
        """
        stats = self.validation_stats.copy()
        
        # Calculate derived statistics
        if stats['proteins_tested'] > 0:
            stats['success_rate'] = stats['proteins_passed'] / stats['proteins_tested']
        else:
            stats['success_rate'] = 0.0
        
        stats['missing_atom_types'] = list(stats['missing_atom_types'])
        
        return stats

# Create global validated force field instance
validated_forcefield = ValidatedForceField() if AMBER_VALIDATION_AVAILABLE else None

def validate_protein_amber_ff(protein_structure, protein_name: str = "Unknown") -> ValidationResult:
    """
    Convenience function to validate a protein with AMBER force field.
    
    Parameters
    ----------
    protein_structure : object
        Protein structure to validate
    protein_name : str
        Name of the protein
        
    Returns
    -------
    ValidationResult
        Validation results
    """
    if validated_forcefield is None:
        logger.error("Validated force field not available")
        result = ValidationResult()
        result.add_error("AMBER validation not available")
        return result
    
    return validated_forcefield.validate_protein(protein_structure, protein_name)

def check_ff_compatibility() -> bool:
    """
    Check if the force field is compatible with standard proteins.
    
    Returns
    -------
    bool
        True if compatible with standard proteins
    """
    if validated_forcefield is None:
        logger.error("Validated force field not available")
        return False
    
    return validated_forcefield.check_standard_proteins()
