#!/usr/bin/env python3
"""
Vereinfachter CLI Test - Schritt für Schritt aufbau
Maximal 100 Zeilen Output!
"""

import sys
import os
from pathlib import Path

# Add path
sys.path.insert(0, '.')

def test_step_by_step():
    print("🔍 Vereinfachter CLI Test")
    print("=" * 40)
    
    try:
        # Step 1: Basic imports
        print("Step 1: Basic imports...")
        from proteinMD.structure.pdb_parser import PDBParser
        from proteinMD.forcefield.amber_ff14sb import AmberFF14SB
        print("✅ Core imports OK")
        
        # Step 2: Load protein
        print("Step 2: Load protein...")
        parser = PDBParser()
        protein = parser.parse_file('data/proteins/1ubq.pdb')
        print(f"✅ Protein: {len(protein.atoms)} atoms")
        
        # Step 3: Force field
        print("Step 3: Create force field...")
        ff = AmberFF14SB()
        print(f"✅ Force field: {ff.name}")
        
        # Step 4: Simple config
        print("Step 4: Create simple config...")
        config = {
            'simulation': {'n_steps': 10, 'temperature': 300.0},
            'environment': {'solvent': 'implicit'}
        }
        print("✅ Config created")
        
        print("\n🎉 ALL STEPS SUCCESSFUL!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Failed at step: {e}")
        return 1

if __name__ == "__main__":
    exit_code = test_step_by_step()
    sys.exit(exit_code)
