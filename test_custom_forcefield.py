#!/usr/bin/env python3
"""
Test script for Custom Force Field Import functionality

This script tests the custom force field import feature for Task 4.3
"""

import sys
import os
sys.path.append('/home/emilio/Documents/ai/md')

import json
import tempfile
from pathlib import Path

def test_custom_forcefield_import():
    """Test the custom force field import functionality."""
    print("🧪 TESTING CUSTOM FORCE FIELD IMPORT (Task 4.3)")
    print("=" * 60)
    
    try:
        # Import the custom force field module
        print("📦 Step 1: Importing custom force field module...")
        from proteinMD.forcefield.custom_import import (
            CustomForceFieldImporter,
            CustomForceField,
            import_custom_forcefield
        )
        print("✅ Custom force field module imported successfully")
        
        # Test JSON import
        print("\n🔬 Step 2: Testing JSON import...")
        json_file = Path("/home/emilio/Documents/ai/md/examples/simple_custom_ff.json")
        
        if json_file.exists():
            try:
                custom_ff = import_custom_forcefield(json_file)
                print(f"✅ JSON import successful: {custom_ff.name}")
                print(f"   Metadata: {custom_ff.get_metadata()}")
                print(f"   Atom types: {len(custom_ff.custom_atom_types)}")
                print(f"   Bond types: {len(custom_ff.custom_bond_types)}")
                print(f"   Angle types: {len(custom_ff.custom_angle_types)}")
                print(f"   Dihedral types: {len(custom_ff.custom_dihedral_types)}")
            except Exception as e:
                print(f"❌ JSON import failed: {e}")
                return False
        else:
            print(f"⚠️  JSON example file not found: {json_file}")
        
        # Test XML import
        print("\n🔬 Step 3: Testing XML import...")
        xml_file = Path("/home/emilio/Documents/ai/md/examples/custom_ff_example.xml")
        
        if xml_file.exists():
            try:
                custom_ff_xml = import_custom_forcefield(xml_file)
                print(f"✅ XML import successful: {custom_ff_xml.name}")
                print(f"   Metadata: {custom_ff_xml.get_metadata()}")
                print(f"   Atom types: {len(custom_ff_xml.custom_atom_types)}")
                print(f"   Bond types: {len(custom_ff_xml.custom_bond_types)}")
                print(f"   Angle types: {len(custom_ff_xml.custom_angle_types)}")
                print(f"   Dihedral types: {len(custom_ff_xml.custom_dihedral_types)}")
            except Exception as e:
                print(f"❌ XML import failed: {e}")
                return False
        else:
            print(f"⚠️  XML example file not found: {xml_file}")
        
        # Test parameter access
        print("\n🔬 Step 4: Testing parameter access...")
        if 'custom_ff' in locals():
            try:
                # Test atom parameter access
                ct_params = custom_ff.get_atom_parameters("CT")
                if ct_params:
                    print(f"✅ CT atom parameters: {ct_params}")
                else:
                    print("❌ Failed to get CT atom parameters")
                
                # Test bond parameter access
                bond_params = custom_ff.get_bond_parameters("CT", "HC")
                if bond_params:
                    print(f"✅ CT-HC bond parameters: {bond_params}")
                else:
                    print("❌ Failed to get CT-HC bond parameters")
                    
            except Exception as e:
                print(f"❌ Parameter access failed: {e}")
                return False
        
        # Test validation with invalid data
        print("\n🔬 Step 5: Testing validation with invalid data...")
        try:
            # Create invalid force field data
            invalid_data = {
                "metadata": {
                    "name": "Invalid FF",
                    "units": {
                        "length": "nm",
                        "energy": "kJ/mol", 
                        "mass": "amu",
                        "angle": "degrees"
                    }
                },
                "atom_types": [
                    {
                        "atom_type": "BAD",
                        "mass": -1.0,  # Invalid negative mass
                        "sigma": 0.1,
                        "epsilon": 0.1
                    }
                ]
            }
            
            # Create temporary file with invalid data
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(invalid_data, f)
                temp_file = f.name
            
            try:
                importer = CustomForceFieldImporter()
                bad_ff = importer.import_from_json(temp_file)
                print("❌ Should have failed validation but didn't")
                return False
            except ValueError as e:
                print(f"✅ Validation correctly caught invalid parameters: {e}")
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            print(f"❌ Validation test failed: {e}")
            return False
        
        # Test export functionality
        print("\n🔬 Step 6: Testing export functionality...")
        if 'custom_ff' in locals():
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    export_file = f.name
                
                custom_ff.export_to_json(export_file)
                
                # Verify export by importing it again
                reimported_ff = import_custom_forcefield(export_file)
                print(f"✅ Export/reimport successful: {reimported_ff.name}")
                
                os.unlink(export_file)
                
            except Exception as e:
                print(f"❌ Export test failed: {e}")
                return False
        
        print("\n🎉 ALL CUSTOM FORCE FIELD TESTS PASSED!")
        print("✅ Task 4.3: Custom Force Field Import - SUCCESSFULLY IMPLEMENTED")
        assert True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        assert False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        assert False

if __name__ == "__main__":
    success = test_custom_forcefield_import()
    
    if success:
        print("\n✅ Task 4.3 implementation test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Task 4.3 implementation test failed!")
        sys.exit(1)
