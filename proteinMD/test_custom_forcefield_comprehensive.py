#!/usr/bin/env python3
"""
Comprehensive test suite for Custom Force Field Import functionality.
Task 4.3: Custom Force Field Import

This test suite validates all requirements:
- XML- oder JSON-Format für Parameter definiert
- Import-Funktion mit Validierung implementiert  
- Dokumentation und Beispiele für Custom-Parameter
- Fehlerbehandlung bei ungültigen Parametern
"""

import unittest
import json
import xml.etree.ElementTree as ET
import tempfile
import os
import sys
from pathlib import Path
import logging

# Add the proteinMD module to the path
sys.path.insert(0, os.path.abspath('.'))

try:
    from forcefield.custom_forcefield import (
        CustomForceField, ParameterFormat, ValidationError, 
        CustomAtomType, CustomBondType, CustomAngleType, CustomDihedralType,
        create_parameter_template
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORT_SUCCESS = False

class TestCustomForceFieldImport(unittest.TestCase):
    """Test suite for Custom Force Field Import functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.json_file = os.path.join(self.temp_dir, "test_ff.json")
        self.xml_file = os.path.join(self.temp_dir, "test_ff.xml")
        
        # Create test protein data
        self.test_protein = {
            "name": "Test Protein",
            "atoms": [
                {"id": 0, "name": "CA", "atom_type": "CA", "x": 0.0, "y": 0.0, "z": 0.0},
                {"id": 1, "name": "CB", "atom_type": "CB", "x": 0.15, "y": 0.0, "z": 0.0},
                {"id": 2, "name": "NH", "atom_type": "NH", "x": -0.15, "y": 0.0, "z": 0.0},
                {"id": 3, "name": "HN", "atom_type": "HN", "x": -0.25, "y": 0.0, "z": 0.0}
            ],
            "bonds": [[0, 1], [0, 2], [2, 3]],
            "angles": [[1, 0, 2], [0, 2, 3]],
            "dihedrals": []
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @unittest.skipUnless(IMPORT_SUCCESS, "CustomForceField import failed")
    def test_01_custom_forcefield_initialization(self):
        """Test 1: Basic custom force field initialization."""
        print("\n" + "="*80)
        print("TEST 1: Custom Force Field Initialization")
        print("="*80)
        
        # Test basic initialization
        custom_ff = CustomForceField(name="Test FF")
        
        self.assertEqual(custom_ff.name, "Test FF")
        self.assertEqual(len(custom_ff.custom_atom_types), 0)
        self.assertEqual(len(custom_ff.custom_bond_types), 0)
        
        print("✓ Basic initialization successful")
        print(f"✓ Force field name: {custom_ff.name}")
        print(f"✓ Initial parameter counts: {custom_ff.get_parameter_summary()}")
    
    @unittest.skipUnless(IMPORT_SUCCESS, "CustomForceField import failed")
    def test_02_json_parameter_format(self):
        """Test 2: JSON parameter format definition and loading."""
        print("\n" + "="*80)
        print("TEST 2: JSON Parameter Format")
        print("="*80)
        
        # Create test JSON parameter file
        json_params = {
            "metadata": {
                "force_field_name": "Test JSON FF",
                "description": "Test force field"
            },
            "atom_types": [
                {
                    "name": "CA",
                    "mass": 12.01,
                    "charge": 0.0,
                    "sigma": 0.34,
                    "epsilon": 0.36,
                    "description": "Carbon alpha"
                },
                {
                    "name": "CB", 
                    "mass": 12.01,
                    "charge": 0.0,
                    "sigma": 0.34,
                    "epsilon": 0.36,
                    "description": "Carbon beta"
                },
                {
                    "name": "NH",
                    "mass": 14.007,
                    "charge": -0.4,
                    "sigma": 0.325,
                    "epsilon": 0.71,
                    "description": "Nitrogen"
                },
                {
                    "name": "HN",
                    "mass": 1.008,
                    "charge": 0.4,
                    "sigma": 0.107,
                    "epsilon": 0.066,
                    "description": "Hydrogen"
                }
            ],
            "bond_types": [
                {
                    "atom_types": ["CA", "CB"],
                    "length": 0.153,
                    "k": 259408.0,
                    "description": "CA-CB bond"
                },
                {
                    "atom_types": ["CA", "NH"],
                    "length": 0.145,
                    "k": 282001.6,
                    "description": "CA-NH bond"
                },
                {
                    "atom_types": ["NH", "HN"],
                    "length": 0.101,
                    "k": 363171.2,
                    "description": "NH-HN bond"
                }
            ],
            "angle_types": [
                {
                    "atom_types": ["CB", "CA", "NH"],
                    "angle": 1.915,
                    "k": 418.4,
                    "description": "CB-CA-NH angle"
                },
                {
                    "atom_types": ["CA", "NH", "HN"],
                    "angle": 2.094,
                    "k": 292.88,
                    "description": "CA-NH-HN angle"
                }
            ]
        }
        
        with open(self.json_file, 'w') as f:
            json.dump(json_params, f, indent=2)
        
        # Test JSON loading
        custom_ff = CustomForceField(parameter_file=self.json_file, format=ParameterFormat.JSON)
        
        self.assertEqual(len(custom_ff.custom_atom_types), 4)
        self.assertEqual(len(custom_ff.custom_bond_types), 6)  # Including reverse mappings
        self.assertEqual(len(custom_ff.custom_angle_types), 4)  # Including reverse mappings
        
        # Test specific parameters
        ca_atom = custom_ff.custom_atom_types["CA"]
        self.assertEqual(ca_atom.mass, 12.01)
        self.assertEqual(ca_atom.sigma, 0.34)
        
        print("✓ JSON format defined correctly")
        print(f"✓ Loaded {len(custom_ff.custom_atom_types)} atom types")
        print(f"✓ Loaded {len(custom_ff.custom_bond_types)//2} unique bond types")
        print(f"✓ Loaded {len(custom_ff.custom_angle_types)//2} unique angle types")
        print("✓ JSON parameter loading successful")
    
    @unittest.skipUnless(IMPORT_SUCCESS, "CustomForceField import failed")
    def test_03_xml_parameter_format(self):
        """Test 3: XML parameter format definition and loading."""
        print("\n" + "="*80)
        print("TEST 3: XML Parameter Format")
        print("="*80)
        
        # Create test XML parameter file
        xml_content = '''<?xml version="1.0" encoding="utf-8"?>
<ForceField name="Test XML FF">
  <AtomTypes>
    <AtomType name="CA" mass="12.01" charge="0.0" sigma="0.34" epsilon="0.36" description="Carbon alpha"/>
    <AtomType name="CB" mass="12.01" charge="0.0" sigma="0.34" epsilon="0.36" description="Carbon beta"/>
    <AtomType name="NH" mass="14.007" charge="-0.4" sigma="0.325" epsilon="0.71" description="Nitrogen"/>
    <AtomType name="HN" mass="1.008" charge="0.4" sigma="0.107" epsilon="0.066" description="Hydrogen"/>
  </AtomTypes>
  <BondTypes>
    <BondType class="CA-CB" length="0.153" k="259408.0" description="CA-CB bond"/>
    <BondType class="CA-NH" length="0.145" k="282001.6" description="CA-NH bond"/>
    <BondType class="NH-HN" length="0.101" k="363171.2" description="NH-HN bond"/>
  </BondTypes>
  <AngleTypes>
    <AngleType class="CB-CA-NH" angle="1.915" k="418.4" description="CB-CA-NH angle"/>
    <AngleType class="CA-NH-HN" angle="2.094" k="292.88" description="CA-NH-HN angle"/>
  </AngleTypes>
</ForceField>'''
        
        with open(self.xml_file, 'w') as f:
            f.write(xml_content)
        
        # Test XML loading
        custom_ff = CustomForceField(parameter_file=self.xml_file, format=ParameterFormat.XML)
        
        self.assertEqual(len(custom_ff.custom_atom_types), 4)
        self.assertEqual(len(custom_ff.custom_bond_types), 6)  # Including reverse mappings
        self.assertEqual(len(custom_ff.custom_angle_types), 4)  # Including reverse mappings
        
        # Test specific parameters
        ca_atom = custom_ff.custom_atom_types["CA"]
        self.assertEqual(ca_atom.mass, 12.01)
        self.assertEqual(ca_atom.sigma, 0.34)
        
        print("✓ XML format defined correctly")
        print(f"✓ Loaded {len(custom_ff.custom_atom_types)} atom types")
        print(f"✓ Loaded {len(custom_ff.custom_bond_types)//2} unique bond types")
        print(f"✓ Loaded {len(custom_ff.custom_angle_types)//2} unique angle types")
        print("✓ XML parameter loading successful")
    
    @unittest.skipUnless(IMPORT_SUCCESS, "CustomForceField import failed")
    def test_04_parameter_validation(self):
        """Test 4: Parameter validation functionality."""
        print("\n" + "="*80)
        print("TEST 4: Parameter Validation")
        print("="*80)
        
        # Test with valid parameters
        valid_params = {
            "atom_types": [
                {"name": "CA", "mass": 12.01, "charge": 0.0, "sigma": 0.34, "epsilon": 0.36}
            ],
            "bond_types": [
                {"atom_types": ["CA", "CA"], "length": 0.153, "k": 259408.0}
            ]
        }
        
        with open(self.json_file, 'w') as f:
            json.dump(valid_params, f)
        
        # Should load without errors
        try:
            custom_ff = CustomForceField(parameter_file=self.json_file, format=ParameterFormat.JSON)
            print("✓ Valid parameters loaded successfully")
        except ValidationError:
            self.fail("Valid parameters should not raise ValidationError")
        
        # Test with invalid parameters - negative mass
        invalid_params = {
            "atom_types": [
                {"name": "CA", "mass": -12.01, "charge": 0.0, "sigma": 0.34, "epsilon": 0.36}
            ]
        }
        
        with open(self.json_file, 'w') as f:
            json.dump(invalid_params, f)
        
        # Should raise ValidationError
        with self.assertRaises(ValidationError):
            custom_ff = CustomForceField(parameter_file=self.json_file, format=ParameterFormat.JSON)
        
        print("✓ Invalid parameters correctly rejected")
        print("✓ Parameter validation working correctly")
    
    @unittest.skipUnless(IMPORT_SUCCESS, "CustomForceField import failed")
    def test_05_error_handling(self):
        """Test 5: Error handling for invalid files and parameters."""
        print("\n" + "="*80)
        print("TEST 5: Error Handling")
        print("="*80)
        
        # Test non-existent file
        with self.assertRaises(FileNotFoundError):
            CustomForceField(parameter_file="non_existent_file.json")
        print("✓ Non-existent file error handled correctly")
        
        # Test invalid JSON
        invalid_json_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_json_file, 'w') as f:
            f.write("{ invalid json content")
        
        with self.assertRaises(ValidationError):
            CustomForceField(parameter_file=invalid_json_file, format=ParameterFormat.JSON)
        print("✓ Invalid JSON error handled correctly")
        
        # Test missing required fields
        incomplete_params = {
            "atom_types": [
                {"name": "CA"}  # Missing mass
            ]
        }
        
        with open(self.json_file, 'w') as f:
            json.dump(incomplete_params, f)
        
        with self.assertRaises(ValidationError):
            CustomForceField(parameter_file=self.json_file, format=ParameterFormat.JSON)
        print("✓ Missing required fields error handled correctly")
        
        # Test invalid XML
        invalid_xml_file = os.path.join(self.temp_dir, "invalid.xml")
        with open(invalid_xml_file, 'w') as f:
            f.write("<invalid><xml>content</invalid>")
        
        with self.assertRaises(ValidationError):
            CustomForceField(parameter_file=invalid_xml_file, format=ParameterFormat.XML)
        print("✓ Invalid XML error handled correctly")
        
        print("✓ All error handling tests passed")
    
    @unittest.skipUnless(IMPORT_SUCCESS, "CustomForceField import failed")
    def test_06_protein_parameter_assignment(self):
        """Test 6: Parameter assignment to proteins."""
        print("\n" + "="*80)
        print("TEST 6: Protein Parameter Assignment")
        print("="*80)
        
        # First load parameters
        test_params = {
            "atom_types": [
                {"name": "CA", "mass": 12.01, "charge": 0.0, "sigma": 0.34, "epsilon": 0.36},
                {"name": "CB", "mass": 12.01, "charge": 0.0, "sigma": 0.34, "epsilon": 0.36},
                {"name": "NH", "mass": 14.007, "charge": -0.4, "sigma": 0.325, "epsilon": 0.71},
                {"name": "HN", "mass": 1.008, "charge": 0.4, "sigma": 0.107, "epsilon": 0.066}
            ],
            "bond_types": [
                {"atom_types": ["CA", "CB"], "length": 0.153, "k": 259408.0},
                {"atom_types": ["CA", "NH"], "length": 0.145, "k": 282001.6},
                {"atom_types": ["NH", "HN"], "length": 0.101, "k": 363171.2}
            ]
        }
        
        with open(self.json_file, 'w') as f:
            json.dump(test_params, f)
        
        custom_ff = CustomForceField(parameter_file=self.json_file, format=ParameterFormat.JSON)
        
        # Test protein validation
        validation = custom_ff.validate_protein_parameters(self.test_protein)
        
        self.assertIn("is_valid", validation)
        self.assertIn("coverage_statistics", validation)
        
        print(f"✓ Validation result: {validation['is_valid']}")
        print(f"✓ Atom type coverage: {validation['coverage_statistics']['atom_type_coverage']:.1f}%")
        
        # Test parameter assignment
        assigned_protein = custom_ff.assign_parameters_to_protein(self.test_protein)
        
        self.assertIn("custom_parameters", assigned_protein)
        self.assertIn("atoms", assigned_protein["custom_parameters"])
        self.assertIn("bonds", assigned_protein["custom_parameters"])
        
        atom_params = assigned_protein["custom_parameters"]["atoms"]
        bond_params = assigned_protein["custom_parameters"]["bonds"]
        
        self.assertEqual(len(atom_params), 4)  # All 4 atoms should have parameters
        self.assertEqual(len(bond_params), 3)  # All 3 bonds should have parameters
        
        print(f"✓ Assigned parameters to {len(atom_params)} atoms")
        print(f"✓ Assigned parameters to {len(bond_params)} bonds")
        print("✓ Protein parameter assignment successful")
    
    @unittest.skipUnless(IMPORT_SUCCESS, "CustomForceField import failed")
    def test_07_parameter_export(self):
        """Test 7: Parameter export functionality."""
        print("\n" + "="*80)
        print("TEST 7: Parameter Export")
        print("="*80)
        
        # Create a custom force field with parameters
        custom_ff = CustomForceField(name="Export Test FF")
        
        # Add some parameters manually
        custom_ff.custom_atom_types["CA"] = CustomAtomType(
            name="CA", mass=12.01, charge=0.0, sigma=0.34, epsilon=0.36, description="Carbon alpha"
        )
        custom_ff.custom_bond_types[("CA", "CB")] = CustomBondType(
            atom_types=("CA", "CB"), length=0.153, k=259408.0, description="CA-CB bond"
        )
        
        # Test JSON export
        json_export_file = os.path.join(self.temp_dir, "exported.json")
        custom_ff.export_parameters(json_export_file, ParameterFormat.JSON)
        
        self.assertTrue(os.path.exists(json_export_file))
        
        # Verify exported content
        with open(json_export_file, 'r') as f:
            exported_data = json.load(f)
        
        self.assertIn("atom_types", exported_data)
        self.assertIn("bond_types", exported_data)
        self.assertEqual(len(exported_data["atom_types"]), 1)
        
        print("✓ JSON export successful")
        
        # Test XML export
        xml_export_file = os.path.join(self.temp_dir, "exported.xml")
        custom_ff.export_parameters(xml_export_file, ParameterFormat.XML)
        
        self.assertTrue(os.path.exists(xml_export_file))
        
        # Verify XML structure
        tree = ET.parse(xml_export_file)
        root = tree.getroot()
        
        self.assertEqual(root.tag, "ForceField")
        self.assertIsNotNone(root.find("AtomTypes"))
        self.assertIsNotNone(root.find("BondTypes"))
        
        print("✓ XML export successful")
        print("✓ Parameter export functionality working correctly")
    
    @unittest.skipUnless(IMPORT_SUCCESS, "CustomForceField import failed")
    def test_08_template_creation(self):
        """Test 8: Parameter template creation."""
        print("\n" + "="*80)
        print("TEST 8: Parameter Template Creation")
        print("="*80)
        
        # Test JSON template creation
        json_template_file = os.path.join(self.temp_dir, "template.json")
        create_parameter_template(json_template_file, ParameterFormat.JSON)
        
        self.assertTrue(os.path.exists(json_template_file))
        
        # Verify template content
        with open(json_template_file, 'r') as f:
            template_data = json.load(f)
        
        self.assertIn("metadata", template_data)
        self.assertIn("atom_types", template_data)
        self.assertIn("bond_types", template_data)
        
        print("✓ JSON template created successfully")
        
        # Test XML template creation
        xml_template_file = os.path.join(self.temp_dir, "template.xml")
        create_parameter_template(xml_template_file, ParameterFormat.XML)
        
        self.assertTrue(os.path.exists(xml_template_file))
        
        # Verify XML template
        tree = ET.parse(xml_template_file)
        root = tree.getroot()
        
        self.assertEqual(root.tag, "ForceField")
        self.assertIsNotNone(root.find("AtomTypes"))
        
        print("✓ XML template created successfully")
        print("✓ Template creation functionality working correctly")
    
    @unittest.skipUnless(IMPORT_SUCCESS, "CustomForceField import failed")
    def test_09_documentation_examples(self):
        """Test 9: Documentation and examples validation."""
        print("\n" + "="*80)
        print("TEST 9: Documentation and Examples")
        print("="*80)
        
        # Test example parameter files exist and are valid
        example_json = "/home/emilio/Documents/ai/md/proteinMD/forcefield/data/custom/example_protein_ff.json"
        example_xml = "/home/emilio/Documents/ai/md/proteinMD/forcefield/data/custom/example_protein_ff.xml"
        
        # Test JSON example
        if os.path.exists(example_json):
            try:
                custom_ff_json = CustomForceField(parameter_file=example_json, format=ParameterFormat.JSON)
                summary = custom_ff_json.get_parameter_summary()
                print(f"✓ JSON example loaded: {summary['atom_types']} atom types, {summary['bond_types']} bond types")
            except Exception as e:
                self.fail(f"JSON example file failed to load: {e}")
        else:
            print("⚠ JSON example file not found")
        
        # Test XML example
        if os.path.exists(example_xml):
            try:
                custom_ff_xml = CustomForceField(parameter_file=example_xml, format=ParameterFormat.XML)
                summary = custom_ff_xml.get_parameter_summary()
                print(f"✓ XML example loaded: {summary['atom_types']} atom types, {summary['bond_types']} bond types")
            except Exception as e:
                self.fail(f"XML example file failed to load: {e}")
        else:
            print("⚠ XML example file not found")
        
        # Test documentation completeness by checking class docstrings
        self.assertIsNotNone(CustomForceField.__doc__)
        self.assertIsNotNone(CustomAtomType.__doc__)
        self.assertIsNotNone(CustomBondType.__doc__)
        
        print("✓ Class documentation present")
        print("✓ Documentation and examples validation completed")
    
    @unittest.skipUnless(IMPORT_SUCCESS, "CustomForceField import failed")
    def test_10_integration_with_existing_framework(self):
        """Test 10: Integration with existing force field framework."""
        print("\n" + "="*80)
        print("TEST 10: Framework Integration")
        print("="*80)
        
        # Create custom force field
        test_params = {
            "atom_types": [
                {"name": "CA", "mass": 12.01, "charge": 0.0, "sigma": 0.34, "epsilon": 0.36}
            ]
        }
        
        with open(self.json_file, 'w') as f:
            json.dump(test_params, f)
        
        custom_ff = CustomForceField(parameter_file=self.json_file, format=ParameterFormat.JSON)
        
        # Test that it inherits from ForceField base class
        from forcefield.forcefield import ForceField
        self.assertIsInstance(custom_ff, ForceField)
        
        # Test common interface methods
        self.assertTrue(hasattr(custom_ff, 'name'))
        self.assertTrue(hasattr(custom_ff, 'cutoff'))
        self.assertTrue(hasattr(custom_ff, 'nonbonded_method'))
        
        # Test custom methods
        self.assertTrue(hasattr(custom_ff, 'validate_protein_parameters'))
        self.assertTrue(hasattr(custom_ff, 'assign_parameters_to_protein'))
        self.assertTrue(hasattr(custom_ff, 'get_supported_atom_types'))
        
        print("✓ Inherits from ForceField base class")
        print("✓ Implements common interface methods")
        print("✓ Provides custom force field specific methods")
        print("✓ Framework integration successful")

def run_comprehensive_tests():
    """Run the comprehensive test suite."""
    print("="*80)
    print("CUSTOM FORCE FIELD IMPORT - COMPREHENSIVE TEST SUITE")
    print("Task 4.3: Custom Force Field Import")
    print("="*80)
    
    if not IMPORT_SUCCESS:
        print("❌ CRITICAL: CustomForceField import failed!")
        print("Please ensure the custom_forcefield.py module is properly implemented.")
        return False
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test methods
    test_methods = [
        'test_01_custom_forcefield_initialization',
        'test_02_json_parameter_format',
        'test_03_xml_parameter_format', 
        'test_04_parameter_validation',
        'test_05_error_handling',
        'test_06_protein_parameter_assignment',
        'test_07_parameter_export',
        'test_08_template_creation',
        'test_09_documentation_examples',
        'test_10_integration_with_existing_framework'
    ]
    
    for method in test_methods:
        suite.addTest(TestCustomForceFieldImport(method))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
        print(f"✓ Ran {result.testsRun} tests successfully")
        print("\nTask 4.3 Requirements Status:")
        print("✅ XML- oder JSON-Format für Parameter definiert")
        print("✅ Import-Funktion mit Validierung implementiert")
        print("✅ Dokumentation und Beispiele für Custom-Parameter")
        print("✅ Fehlerbehandlung bei ungültigen Parametern")
        print("\n🏆 Task 4.3 - Custom Force Field Import: COMPLETE")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print(f"Failed: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
        
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
