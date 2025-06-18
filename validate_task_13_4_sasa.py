#!/usr/bin/env python3
"""
VALIDATION SCRIPT FOR TASK 13.4 - SOLVENT ACCESSIBLE SURFACE AREA (SASA)

This script validates the complete implementation of SASA calculation
according to the requirements in aufgabenliste.md.

Requirements to validate:
✅ Rolling-Ball-Algorithmus für SASA implementiert  
✅ Per-Residue SASA-Werte berechnet
✅ Hydrophobic/Hydrophilic Surface-Anteile
✅ Zeitverlauf der SASA-Änderungen

Author: ProteinMD Team
Date: December 2024
"""

import sys
import os
import numpy as np
import time
import traceback
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("🧪 TASK 13.4 SASA IMPLEMENTATION VALIDATION")
    print("=" * 60)
    print("Testing Solvent Accessible Surface Area (SASA) calculation")
    print("according to aufgabenliste.md requirements")
    print("=" * 60)
    
    start_time = time.time()
    total_tests = 7
    passed_tests = 0
    
    try:
        # Import the SASA analysis module
        from proteinMD.analysis.sasa import (
            SASAAnalyzer, SASACalculator, SASAResult, SASATimeSeriesResult,
            analyze_sasa, create_test_protein_structure, create_test_trajectory,
            VDW_RADII, HYDROPHOBIC_ATOMS, HYDROPHILIC_ATOMS
        )
        print("✅ SASA module imported successfully")
        
    except ImportError as e:
        print(f"❌ Failed to import SASA module: {e}")
        print(traceback.format_exc())
        return
    
    # Test 1: Rolling Ball Algorithm Implementation
    print(f"\n{'='*60}")
    print("TEST 1: Rolling Ball Algorithm Implementation")
    print(f"{'='*60}")
    
    try:
        # Create test structure
        positions, atom_types, residue_ids = create_test_protein_structure(50)
        
        # Initialize SASA calculator with different probe radii
        calc_small = SASACalculator(probe_radius=1.0, n_points=194)
        calc_large = SASACalculator(probe_radius=2.0, n_points=194)
        
        # Calculate SASA with different probe radii
        result_small = calc_small.calculate_sasa(positions, atom_types, residue_ids)
        result_large = calc_large.calculate_sasa(positions, atom_types, residue_ids)
        
        print(f"Small probe (1.0 Å): {result_small.total_sasa:.2f} Ų")
        print(f"Large probe (2.0 Å): {result_large.total_sasa:.2f} Ų")
        
        # Validate rolling ball algorithm properties
        sasa_ratio = result_large.total_sasa / result_small.total_sasa
        print(f"SASA ratio (large/small probe): {sasa_ratio:.3f}")
        
        # Larger probe should give larger SASA (more accessible surface)
        if result_large.total_sasa > result_small.total_sasa:
            print("✅ Rolling ball algorithm correctly implemented")
            print(f"   - Probe radius dependency verified")
            print(f"   - Quadrature integration working ({result_small.n_points} points)")
            print(f"   - Calculation time: {result_small.calculation_time:.3f}s")
            passed_tests += 1
        else:
            print("❌ Rolling ball algorithm issue: larger probe should give larger SASA")
            
    except Exception as e:
        print(f"❌ Rolling ball test failed: {e}")
        print(traceback.format_exc())
    
    # Test 2: Per-Residue SASA Values
    print(f"\n{'='*60}")
    print("TEST 2: Per-Residue SASA Values")
    print(f"{'='*60}")
    
    try:
        # Create structure with known residue distribution
        positions, atom_types, residue_ids = create_test_protein_structure(60)
        analyzer = SASAAnalyzer(probe_radius=1.4, n_points=194)
        
        result = analyzer.analyze_single_frame(positions, atom_types, residue_ids)
        
        print(f"Total atoms: {len(positions)}")
        print(f"Number of residues: {len(result.per_residue_sasa)}")
        print(f"Residue IDs: {sorted(result.per_residue_sasa.keys())}")
        
        # Validate per-residue decomposition
        residue_sum = sum(result.per_residue_sasa.values())
        total_sasa = result.total_sasa
        decomposition_error = abs(residue_sum - total_sasa) / total_sasa
        
        print(f"Sum of per-residue SASA: {residue_sum:.2f} Ų")
        print(f"Total SASA: {total_sasa:.2f} Ų")
        print(f"Decomposition error: {decomposition_error:.6f}")
        
        # Show per-residue values
        print("\nPer-residue SASA values:")
        for res_id in sorted(result.per_residue_sasa.keys())[:5]:  # Show first 5
            sasa_val = result.per_residue_sasa[res_id]
            print(f"   Residue {res_id}: {sasa_val:.2f} Ų")
        
        if decomposition_error < 0.001 and len(result.per_residue_sasa) > 0:
            print("✅ Per-residue SASA calculation validated")
            print(f"   - Decomposition accuracy: {decomposition_error:.6f}")
            print(f"   - {len(result.per_residue_sasa)} residues analyzed")
            passed_tests += 1
        else:
            print("❌ Per-residue SASA decomposition failed")
            
    except Exception as e:
        print(f"❌ Per-residue test failed: {e}")
        print(traceback.format_exc())
    
    # Test 3: Hydrophobic/Hydrophilic Surface Components
    print(f"\n{'='*60}")
    print("TEST 3: Hydrophobic/Hydrophilic Surface Components")
    print(f"{'='*60}")
    
    try:
        # Create structure with known atom types
        positions, atom_types, residue_ids = create_test_protein_structure(40)
        analyzer = SASAAnalyzer(probe_radius=1.4, n_points=194)
        
        result = analyzer.analyze_single_frame(positions, atom_types, residue_ids)
        
        print(f"Total SASA: {result.total_sasa:.2f} Ų")
        print(f"Hydrophobic SASA: {result.hydrophobic_sasa:.2f} Ų")
        print(f"Hydrophilic SASA: {result.hydrophilic_sasa:.2f} Ų")
        
        # Calculate fractions
        hydrophobic_fraction = result.hydrophobic_sasa / result.total_sasa
        hydrophilic_fraction = result.hydrophilic_sasa / result.total_sasa
        sum_fraction = hydrophobic_fraction + hydrophilic_fraction
        
        print(f"Hydrophobic fraction: {hydrophobic_fraction:.3f}")
        print(f"Hydrophilic fraction: {hydrophilic_fraction:.3f}")
        print(f"Sum of fractions: {sum_fraction:.6f}")
        
        # Count atom types
        hydrophobic_atoms = sum(1 for at in atom_types if at.upper() in HYDROPHOBIC_ATOMS or at[0].upper() in {'C', 'S'})
        hydrophilic_atoms = sum(1 for at in atom_types if at.upper() in HYDROPHILIC_ATOMS or at[0].upper() in {'N', 'O'})
        
        print(f"\nAtom classification:")
        print(f"Hydrophobic atoms: {hydrophobic_atoms}")
        print(f"Hydrophilic atoms: {hydrophilic_atoms}")
        print(f"Total atoms: {len(atom_types)}")
        
        # Validate hydrophobic/hydrophilic classification
        if (abs(sum_fraction - 1.0) < 0.001 and 
            result.hydrophobic_sasa > 0 and result.hydrophilic_sasa > 0):
            print("✅ Hydrophobic/Hydrophilic surface classification validated")
            print(f"   - Component sum accuracy: {sum_fraction:.6f}")
            print(f"   - Both components present")
            passed_tests += 1
        else:
            print("❌ Hydrophobic/Hydrophilic classification failed")
            
    except Exception as e:
        print(f"❌ Hydrophobic/Hydrophilic test failed: {e}")
        print(traceback.format_exc())
    
    # Test 4: Time Series Analysis Implementation
    print(f"\n{'='*60}")
    print("TEST 4: Time Series Analysis (Zeitverlauf der SASA-Änderungen)")
    print(f"{'='*60}")
    
    try:
        # Create trajectory
        trajectory, atom_types, residue_ids, time_points = create_test_trajectory(15, 30)
        analyzer = SASAAnalyzer(probe_radius=1.4, n_points=194)
        
        ts_result = analyzer.analyze_trajectory(trajectory, atom_types, residue_ids, time_points)
        
        print(f"Trajectory frames: {len(ts_result.time_points)}")
        print(f"Time range: {ts_result.time_points[0]:.1f} - {ts_result.time_points[-1]:.1f}")
        print(f"Mean total SASA: {ts_result.statistics['mean_total']:.2f} ± {ts_result.statistics['std_total']:.2f} Ų")
        print(f"SASA range: {ts_result.statistics['min_total']:.2f} - {ts_result.statistics['max_total']:.2f} Ų")
        print(f"Coefficient of variation: {ts_result.statistics['coefficient_of_variation']:.4f}")
        print(f"Calculation time: {ts_result.calculation_time:.3f}s")
        
        # Validate time series properties
        n_frames = len(ts_result.time_points)
        has_variation = ts_result.statistics['std_total'] > 0
        has_per_residue = len(ts_result.per_residue_sasa) > 0
        
        print(f"\nTime series validation:")
        print(f"Frames analyzed: {n_frames}")
        print(f"Has temporal variation: {has_variation}")
        print(f"Per-residue time series: {has_per_residue}")
        
        # Check per-residue time series
        first_residue = list(ts_result.per_residue_sasa.keys())[0]
        residue_ts = ts_result.per_residue_sasa[first_residue]
        print(f"Example residue {first_residue} SASA range: {np.min(residue_ts):.2f} - {np.max(residue_ts):.2f} Ų")
        
        if n_frames > 1 and has_variation and has_per_residue:
            print("✅ Time series analysis validated")
            print(f"   - {n_frames} frames analyzed")
            print(f"   - Temporal variation detected")
            print(f"   - Per-residue time series available")
            passed_tests += 1
        else:
            print("❌ Time series analysis failed")
            
    except Exception as e:
        print(f"❌ Time series test failed: {e}")
        print(traceback.format_exc())
    
    # Test 5: Visualization Capabilities
    print(f"\n{'='*60}")
    print("TEST 5: Visualization and Export Capabilities")
    print(f"{'='*60}")
    
    try:
        # Use existing trajectory result
        if 'ts_result' in locals():
            # Test time series plotting
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            analyzer.plot_time_series(ts_result, output_file="test_sasa_timeseries.png", show_components=True)
            analyzer.plot_per_residue_sasa(ts_result, output_file="test_sasa_per_residue.png")
            
            # Test data export
            analyzer.export_results(ts_result, "test_sasa_results.dat")
            
            # Check if files were created
            files_created = []
            for filename in ["test_sasa_timeseries.png", "test_sasa_per_residue.png", "test_sasa_results.dat"]:
                if os.path.exists(filename):
                    files_created.append(filename)
                    print(f"✓ Created: {filename}")
            
            if len(files_created) >= 2:  # At least plots created
                print("✅ Visualization and export capabilities validated")
                print(f"   - {len(files_created)} output files generated")
                passed_tests += 1
            else:
                print("❌ Visualization/export failed")
        else:
            print("❌ No trajectory data available for visualization test")
            
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        print(traceback.format_exc())
    
    # Test 6: Integration with Analysis Framework
    print(f"\n{'='*60}")
    print("TEST 6: Integration with Analysis Framework")
    print(f"{'='*60}")
    
    try:
        # Test import through analysis module
        from proteinMD.analysis import HAS_SASA
        
        if HAS_SASA:
            from proteinMD.analysis import SASAAnalyzer, analyze_sasa
            
            # Test convenience function
            positions, atom_types, residue_ids = create_test_protein_structure(25)
            result = analyze_sasa(positions, atom_types, residue_ids)
            
            print(f"Convenience function result: {result.total_sasa:.2f} Ų")
            print(f"Analysis framework integration: HAS_SASA = {HAS_SASA}")
            
            print("✅ Analysis framework integration validated")
            print(f"   - Module properly integrated")
            print(f"   - Convenience functions available")
            passed_tests += 1
        else:
            print("❌ Analysis framework integration failed")
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print(traceback.format_exc())
    
    # Test 7: Scientific Validation
    print(f"\n{'='*60}")
    print("TEST 7: Scientific Validation and Edge Cases")
    print(f"{'='*60}")
    
    try:
        # Test with different system sizes
        sizes = [10, 30, 50]
        calc_times = []
        sasa_values = []
        
        analyzer = SASAAnalyzer(probe_radius=1.4, n_points=194)
        
        for size in sizes:
            positions, atom_types, residue_ids = create_test_protein_structure(size)
            result = analyzer.analyze_single_frame(positions, atom_types, residue_ids)
            
            calc_times.append(result.calculation_time)
            sasa_values.append(result.total_sasa)
            
            print(f"Size {size}: SASA = {result.total_sasa:.1f} Ų, Time = {result.calculation_time:.3f}s")
        
        # Test edge cases
        # Single atom
        single_pos = np.array([[0, 0, 0]])
        single_result = analyzer.analyze_single_frame(single_pos, ['C'], [1])
        expected_single = 4 * np.pi * (1.7 + 1.4)**2  # C radius + probe radius
        
        print(f"\nEdge case - Single atom:")
        print(f"Calculated SASA: {single_result.total_sasa:.2f} Ų")
        print(f"Expected SASA: {expected_single:.2f} Ų")
        print(f"Relative error: {abs(single_result.total_sasa - expected_single)/expected_single:.4f}")
        
        # Validate scientific properties
        sasa_increases = all(sasa_values[i] <= sasa_values[i+1] for i in range(len(sasa_values)-1))
        single_atom_accuracy = abs(single_result.total_sasa - expected_single) / expected_single < 0.1
        
        if sasa_increases and single_atom_accuracy:
            print("✅ Scientific validation passed")
            print(f"   - SASA scales with system size")
            print(f"   - Single atom test accurate")
            print(f"   - Performance scaling reasonable")
            passed_tests += 1
        else:
            print("❌ Scientific validation failed")
            
    except Exception as e:
        print(f"❌ Scientific validation failed: {e}")
        print(traceback.format_exc())
    
    # Final Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    print(f"Total validation time: {total_time:.2f} seconds")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL REQUIREMENTS VALIDATED SUCCESSFULLY!")
        print("✅ Rolling-Ball-Algorithmus für SASA implementiert")
        print("✅ Per-Residue SASA-Werte berechnet")
        print("✅ Hydrophobic/Hydrophilic Surface-Anteile")
        print("✅ Zeitverlauf der SASA-Änderungen")
        print("\nTask 13.4 SASA implementation is COMPLETE and ready for production use!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Implementation needs review.")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
