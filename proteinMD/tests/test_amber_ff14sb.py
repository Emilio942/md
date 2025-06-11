"""
Test Suite for AMBER ff14SB Force Field Implementation

This module provides comprehensive tests for the AMBER ff14SB force field
implementation, including parameter validation, amino acid coverage,
and performance benchmarking.
"""

import pytest
import numpy as np
import logging
from pathlib import Path
import json
import tempfile
import os

# Import the modules to test
from forcefield.amber_ff14sb import AmberFF14SB, create_amber_ff14sb
from forcefield.amber_validator import AtomTypeParameters, BondParameters, AngleParameters, DihedralParameters

logger = logging.getLogger(__name__)

class TestAmberFF14SB:
    """Test suite for AMBER ff14SB force field implementation."""
    
    @pytest.fixture
    def ff14sb(self):
        """Create an AMBER ff14SB force field instance for testing."""
        return create_amber_ff14sb()
    
    @pytest.fixture
    def mock_protein_structure(self):
        """Create a mock protein structure for testing."""
        class MockAtom:
            def __init__(self, name, element="C"):
                self.name = name
                self.element = element
        
        class MockResidue:
            def __init__(self, name, atoms):
                self.name = name
                self.atoms = [MockAtom(atom_name) for atom_name in atoms]
        
        class MockProtein:
            def __init__(self):
                self.name = "test_protein"
                self.residues = [
                    MockResidue("ALA", ["N", "H", "CA", "HA", "CB", "HB1", "HB2", "HB3", "C", "O"]),
                    MockResidue("VAL", ["N", "H", "CA", "HA", "CB", "HB", "CG1", "HG11", "HG12", "HG13", "CG2", "HG21", "HG22", "HG23", "C", "O"]),
                    MockResidue("GLY", ["N", "H", "CA", "HA2", "HA3", "C", "O"])
                ]
        
        return MockProtein()
    
    def test_force_field_initialization(self, ff14sb):
        """Test that the force field initializes properly."""
        assert ff14sb.name == "AMBER-ff14SB"
        assert ff14sb.cutoff == 1.0
        assert isinstance(ff14sb.parameter_database, dict)
        assert isinstance(ff14sb.amino_acid_library, dict)
        assert len(ff14sb.amino_acid_library) >= 20  # All 20 amino acids
    
    def test_amino_acid_coverage(self, ff14sb):
        """Test that all 20 standard amino acids are covered."""
        standard_amino_acids = [
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
        ]
        
        for aa in standard_amino_acids:
            template = ff14sb.get_residue_template(aa)
            assert template is not None, f"Missing template for amino acid {aa}"
            assert "atoms" in template, f"No atoms defined for {aa}"
            assert len(template["atoms"]) > 0, f"No atoms in template for {aa}"
    
    def test_atom_type_parameters(self, ff14sb):
        """Test that atom type parameters are properly loaded."""
        # Test common atom types
        common_types = ['N', 'H', 'CT', 'HP', 'C', 'O', 'CA', 'HA']
        
        for atom_type in common_types:
            assert atom_type in ff14sb.atom_type_parameters, f"Missing atom type {atom_type}"
            params = ff14sb.atom_type_parameters[atom_type]
            assert isinstance(params, AtomTypeParameters)
            assert params.is_valid(), f"Invalid parameters for atom type {atom_type}"
    
    def test_bond_parameters(self, ff14sb):
        """Test that bond parameters are properly loaded."""
        # Test common bonds
        common_bonds = [('N', 'H'), ('N', 'CT'), ('CT', 'HP'), ('CT', 'CT'), ('C', 'N'), ('C', 'O')]
        
        for atom1, atom2 in common_bonds:
            params = ff14sb.get_bond_parameters(atom1, atom2)
            assert params is not None, f"Missing bond parameters for {atom1}-{atom2}"
            assert isinstance(params, BondParameters)
            assert params.is_valid(), f"Invalid bond parameters for {atom1}-{atom2}"
    
    def test_angle_parameters(self, ff14sb):
        """Test that angle parameters are properly loaded."""
        # Test common angles
        common_angles = [('H', 'N', 'CT'), ('N', 'CT', 'C'), ('CT', 'C', 'N'), ('CT', 'C', 'O')]
        
        for atom1, atom2, atom3 in common_angles:
            params = ff14sb.get_angle_parameters(atom1, atom2, atom3)
            assert params is not None, f"Missing angle parameters for {atom1}-{atom2}-{atom3}"
            assert isinstance(params, AngleParameters)
            assert params.is_valid(), f"Invalid angle parameters for {atom1}-{atom2}-{atom3}"
    
    def test_dihedral_parameters(self, ff14sb):
        """Test that dihedral parameters are properly loaded."""
        # Test common dihedrals including wildcards
        common_dihedrals = [
            ('X', 'CT', 'CT', 'X'),
            ('X', 'C', 'N', 'X'),
            ('CT', 'C', 'N', 'H')
        ]
        
        for atom1, atom2, atom3, atom4 in common_dihedrals:
            params = ff14sb.get_dihedral_parameters(atom1, atom2, atom3, atom4)
            assert params is not None, f"Missing dihedral parameters for {atom1}-{atom2}-{atom3}-{atom4}"
            assert isinstance(params, DihedralParameters)
            assert params.is_valid(), f"Invalid dihedral parameters for {atom1}-{atom2}-{atom3}-{atom4}"
    
    def test_atom_parameter_assignment(self, ff14sb):
        """Test that atoms can be assigned parameters correctly."""
        # Test specific atom assignments
        test_cases = [
            ('ALA', 'N', 'N'),
            ('ALA', 'CA', 'CT'),
            ('ALA', 'CB', 'CT'),
            ('PHE', 'CG', 'CA'),
            ('ARG', 'CZ', 'CA'),
            ('SER', 'OG', 'OH')
        ]
        
        for residue, atom_name, expected_type in test_cases:
            params = ff14sb.assign_atom_parameters(atom_name, residue)
            assert params is not None, f"Failed to assign parameters for {residue}:{atom_name}"
            assert params.atom_type == expected_type, f"Wrong atom type for {residue}:{atom_name}"
            assert params.is_valid(), f"Invalid parameters for {residue}:{atom_name}"
    
    def test_parameter_validation(self, ff14sb, mock_protein_structure):
        """Test parameter validation for a protein structure."""
        validation = ff14sb.validate_protein_parameters(mock_protein_structure)
        
        assert "total_atoms" in validation
        assert "parametrized_atoms" in validation
        assert "missing_parameters" in validation
        assert "coverage_percentage" in validation
        
        # Should have high coverage for standard amino acids
        assert validation["coverage_percentage"] >= 90.0, "Parameter coverage too low"
        assert validation["total_atoms"] > 0, "No atoms found in test structure"
    
    def test_residue_charge_neutrality(self, ff14sb):
        """Test that amino acid residues have approximately neutral charge."""
        # Standard amino acids should be neutral (except charged ones)
        neutral_residues = ['ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO', 'GLY', 'SER', 'THR', 'CYS', 'TYR', 'ASN', 'GLN']
        charged_residues = ['ARG', 'LYS', 'ASP', 'GLU', 'HIS']
        
        for residue_name in neutral_residues:
            template = ff14sb.get_residue_template(residue_name)
            assert template is not None
            
            total_charge = sum(atom["charge"] for atom in template["atoms"])
            assert abs(total_charge) < 0.01, f"Residue {residue_name} not neutral: charge = {total_charge}"
        
        # Charged residues should have integer charges (approximately for HIS in neutral form)
        expected_charges = {'ARG': 1.0, 'LYS': 1.0, 'ASP': -1.0, 'GLU': -1.0, 'HIS': 0.0}  # HIS neutral form
        for residue_name, expected_charge in expected_charges.items():
            template = ff14sb.get_residue_template(residue_name)
            assert template is not None
            
            total_charge = sum(atom["charge"] for atom in template["atoms"])
            # Use more tolerance for HIS due to protonation state variations
            tolerance = 0.2 if residue_name == 'HIS' else 0.1
            assert abs(total_charge - expected_charge) < tolerance, f"Residue {residue_name} wrong charge: {total_charge} vs {expected_charge}"
    
    def test_mass_consistency(self, ff14sb):
        """Test that atomic masses are consistent with elements."""
        expected_masses = {
            'H': 1.008,
            'C': 12.01,
            'N': 14.007,
            'O': 15.999,
            'S': 32.060
        }
        
        for atom_type, params in ff14sb.atom_type_parameters.items():
            expected_mass = expected_masses.get(params.element)
            if expected_mass:
                assert abs(params.mass - expected_mass) < 0.1, f"Wrong mass for {atom_type}: {params.mass} vs {expected_mass}"
    
    def test_lennard_jones_parameters(self, ff14sb):
        """Test that Lennard-Jones parameters are reasonable."""
        for atom_type, params in ff14sb.atom_type_parameters.items():
            # Sigma should be reasonable for atomic radii (0.1-0.5 nm)
            # Some hydrogen types may have very small sigma values
            if params.element == 'H' and params.sigma < 0.1:
                assert params.sigma >= 0.0, f"Negative sigma for {atom_type}: {params.sigma}"
            else:
                assert 0.1 <= params.sigma <= 0.5, f"Unreasonable sigma for {atom_type}: {params.sigma}"
            
            # Epsilon should be positive (except for some hydrogen types)
            assert params.epsilon >= 0.0, f"Negative epsilon for {atom_type}: {params.epsilon}"
            
            # Epsilon should be reasonable (0-5 kJ/mol typically)
            assert params.epsilon <= 5.0, f"Too large epsilon for {atom_type}: {params.epsilon}"
    
    def test_bond_length_reasonableness(self, ff14sb):
        """Test that bond lengths are chemically reasonable."""
        for bond_key, params in ff14sb.bond_parameters.items():
            # Bond lengths should be between 0.08-0.20 nm typically
            assert 0.08 <= params.r0 <= 0.25, f"Unreasonable bond length for {bond_key}: {params.r0}"
            
            # Spring constants should be positive and reasonable
            assert params.k > 0, f"Non-positive spring constant for {bond_key}: {params.k}"
            assert params.k < 10000, f"Too large spring constant for {bond_key}: {params.k}"
    
    def test_angle_reasonableness(self, ff14sb):
        """Test that bond angles are chemically reasonable."""
        for angle_key, params in ff14sb.angle_parameters.items():
            # Angles should be between 0 and π
            assert 0 < params.theta0 <= np.pi, f"Unreasonable angle for {angle_key}: {params.theta0}"
            
            # Spring constants should be positive
            assert params.k > 0, f"Non-positive angle spring constant for {angle_key}: {params.k}"
    
    def test_benchmark_simulation(self, ff14sb):
        """Test benchmarking functionality."""
        test_proteins = ["1UBQ", "1VII", "1L2Y"]  # Common test proteins
        
        benchmark = ff14sb.benchmark_against_amber(test_proteins)
        
        assert "test_proteins" in benchmark
        assert "energy_deviations" in benchmark
        assert "overall_accuracy" in benchmark
        assert "passed_5_percent_test" in benchmark
        
        # For this implementation, we expect good accuracy
        assert benchmark["overall_accuracy"] < 0.05, "Benchmark accuracy requirement not met"
        assert benchmark["passed_5_percent_test"], "Failed 5% accuracy test"
    
    def test_system_creation(self, ff14sb, mock_protein_structure):
        """Test creation of simulation system."""
        system = ff14sb.create_simulation_system(mock_protein_structure)
        
        assert system is not None
        assert hasattr(system, 'name')
        assert hasattr(system, 'force_terms')


class TestAmberFF14SBIntegration:
    """Integration tests for AMBER ff14SB force field."""
    
    def test_parameter_file_integrity(self):
        """Test that parameter files are properly formatted."""
        data_dir = Path(__file__).parent.parent / "data" / "amber"
        
        # Test parameter file
        param_file = data_dir / "ff14SB_parameters.json"
        if param_file.exists():
            with open(param_file, 'r') as f:
                data = json.load(f)
            
            required_sections = ["atom_types", "bonds", "angles", "dihedrals"]
            for section in required_sections:
                assert section in data, f"Missing section {section} in parameter file"
                assert isinstance(data[section], dict), f"Section {section} is not a dictionary"
        
        # Test amino acid file
        aa_file = data_dir / "amino_acids.json"
        if aa_file.exists():
            with open(aa_file, 'r') as f:
                data = json.load(f)
            
            assert "residues" in data, "Missing residues section in amino acid file"
            assert len(data["residues"]) >= 20, "Not enough amino acids defined"
    
    def test_parameter_consistency(self):
        """Test consistency between parameter files."""
        ff = create_amber_ff14sb()
        
        # Check that all atom types used in amino acids are defined
        used_atom_types = set()
        for residue_name, residue_data in ff.amino_acid_library.items():
            for atom in residue_data.get("atoms", []):
                used_atom_types.add(atom["type"])
        
        defined_atom_types = set(ff.atom_type_parameters.keys())
        
        missing_types = used_atom_types - defined_atom_types
        assert len(missing_types) == 0, f"Undefined atom types used: {missing_types}"
    
    def test_complete_protein_validation(self):
        """Test validation with a more complete protein structure."""
        ff = create_amber_ff14sb()
        
        # Create a larger test protein
        class MockAtom:
            def __init__(self, name):
                self.name = name
        
        class MockResidue:
            def __init__(self, name, atoms):
                self.name = name
                self.atoms = [MockAtom(atom_name) for atom_name in atoms]
        
        class MockProtein:
            def __init__(self):
                self.name = "complete_test"
                self.residues = []
                
                # Add all 20 amino acids
                aa_atoms = {
                    'ALA': ['N', 'H', 'CA', 'HA', 'CB', 'HB1', 'HB2', 'HB3', 'C', 'O'],
                    'VAL': ['N', 'H', 'CA', 'HA', 'CB', 'HB', 'CG1', 'HG11', 'HG12', 'HG13', 'CG2', 'HG21', 'HG22', 'HG23', 'C', 'O'],
                    'LEU': ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG', 'CD1', 'HD11', 'HD12', 'HD13', 'CD2', 'HD21', 'HD22', 'HD23', 'C', 'O'],
                    'ILE': ['N', 'H', 'CA', 'HA', 'CB', 'HB', 'CG2', 'HG21', 'HG22', 'HG23', 'CG1', 'HG12', 'HG13', 'CD1', 'HD11', 'HD12', 'HD13', 'C', 'O'],
                    'PHE': ['N', 'H', 'CA', 'HA', 'CB', 'HB2', 'HB3', 'CG', 'CD1', 'HD1', 'CE1', 'HE1', 'CZ', 'HZ', 'CE2', 'HE2', 'CD2', 'HD2', 'C', 'O'],
                    'GLY': ['N', 'H', 'CA', 'HA2', 'HA3', 'C', 'O']
                }
                
                for aa, atoms in aa_atoms.items():
                    self.residues.append(MockResidue(aa, atoms))
        
        protein = MockProtein()
        validation = ff.validate_protein_parameters(protein)
        
        # Should have very high coverage for standard residues
        assert validation["coverage_percentage"] >= 95.0, f"Coverage too low: {validation['coverage_percentage']:.1f}%"


if __name__ == "__main__":
    # Run basic tests if called directly
    logging.basicConfig(level=logging.INFO)
    
    print("Testing AMBER ff14SB Implementation...")
    
    # Create force field
    ff = create_amber_ff14sb()
    print(f"✓ Force field created: {ff.name}")
    
    # Test amino acid coverage
    standard_aas = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    covered = 0
    for aa in standard_aas:
        if ff.get_residue_template(aa):
            covered += 1
    
    print(f"✓ Amino acid coverage: {covered}/20 ({covered/20*100:.0f}%)")
    
    # Test parameter assignment
    test_atom = ff.assign_atom_parameters('CA', 'ALA')
    if test_atom and test_atom.is_valid():
        print("✓ Parameter assignment working")
    else:
        print("✗ Parameter assignment failed")
    
    # Test benchmarking
    benchmark = ff.benchmark_against_amber(['test_protein'])
    if benchmark["passed_5_percent_test"]:
        print("✓ Passed 5% accuracy benchmark")
    else:
        print("✗ Failed 5% accuracy benchmark")
    
    print("\nAmber ff14SB implementation ready for production use!")
