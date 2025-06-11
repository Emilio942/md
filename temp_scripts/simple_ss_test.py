#!/usr/bin/env python3
"""
Simple test to check if secondary structure analysis implementation exists and works.
"""

import sys
import os
import numpy as np

# Add the proteinMD directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'proteinMD'))

print("Testing secondary structure analysis imports...")

# Test imports
try:
    from analysis.secondary_structure import (
        SecondaryStructureAnalyzer, 
        assign_secondary_structure_dssp,
        SS_TYPES
    )
    print("✅ Secondary structure analysis imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Create a simple mock structure for testing
class SimpleAtom:
    def __init__(self, name, pos, res_num, res_name):
        self.atom_name = name
        self.position = np.array(pos)
        self.residue_number = res_num
        self.residue_name = res_name

class SimpleMolecule:
    def __init__(self):
        self.atoms = []

# Create test molecule
molecule = SimpleMolecule()

# Add some atoms for a small helix
for i in range(5):
    res_num = i + 1
    z = i * 0.15  # 1.5 Å in nm
    
    # Simple helix positions
    x = 0.23 * np.cos(i * 1.75)  # ~100 degrees per residue
    y = 0.23 * np.sin(i * 1.75)
    
    atoms = [
        ('N', [x - 0.05, y, z]),
        ('CA', [x, y, z]),
        ('C', [x + 0.05, y + 0.03, z + 0.02]),
        ('O', [x + 0.08, y + 0.05, z + 0.01])
    ]
    
    for atom_name, pos in atoms:
        molecule.atoms.append(SimpleAtom(atom_name, pos, res_num, 'ALA'))

print(f"Created test molecule with {len(molecule.atoms)} atoms, {len(set(a.residue_number for a in molecule.atoms))} residues")

# Test DSSP algorithm
try:
    assignments = assign_secondary_structure_dssp(molecule)
    print(f"✅ DSSP assignment successful: {assignments}")
except Exception as e:
    print(f"❌ DSSP assignment failed: {e}")
    sys.exit(1)

# Test analyzer
try:
    analyzer = SecondaryStructureAnalyzer()
    result = analyzer.analyze_structure(molecule, time_point=0.0)
    print(f"✅ Structure analysis successful")
    print(f"   Structure percentages: {result['percentages']}")
except Exception as e:
    print(f"❌ Structure analysis failed: {e}")
    sys.exit(1)

print("\n✅ All basic tests passed! Secondary structure analysis is working.")
print("Task 3.4 appears to be implemented correctly.")
