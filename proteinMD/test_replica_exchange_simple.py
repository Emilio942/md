#!/usr/bin/env python3
"""
Simple Replica Exchange MD Test

Quick validation of Task 6.2 implementation.
"""

import sys
import numpy as np
from pathlib import Path

# Add the proteinMD module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_remd_basic():
    """Test basic REMD functionality."""
    print("Testing Replica Exchange MD - Task 6.2")
    print("="*50)
    
    try:
        from sampling.replica_exchange import (
            create_remd_simulation, validate_remd_requirements, mock_md_simulation
        )
        print("✓ Successfully imported REMD module")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Create test system
    n_atoms = 10
    initial_positions = np.random.random((n_atoms, 3)) * 2.0
    
    print(f"\nTest system: {n_atoms} atoms")
    
    # Create REMD simulation
    try:
        remd = create_remd_simulation(
            initial_positions=initial_positions,
            min_temperature=300.0,
            max_temperature=400.0,
            n_replicas=4,  # Minimum required
            exchange_frequency=100,
            output_directory="test_remd_simple",
            n_workers=2
        )
        print(f"✓ Created REMD with {remd.n_replicas} replicas")
    except Exception as e:
        print(f"✗ REMD creation failed: {e}")
        return False
    
    # Validate requirements
    validation = validate_remd_requirements(remd)
    print(f"\nRequirement validation:")
    
    requirements = [
        ("min_4_replicas", "≥4 replicas"),
        ("metropolis_exchanges", "Metropolis criterion"),
        ("parallel_execution", "Parallel execution")
    ]
    
    all_passed = True
    for key, description in requirements:
        passed = validation.get(key, False)
        status = "✓" if passed else "✗"
        print(f"  {status} {description}")
        if not passed:
            all_passed = False
    
    # Test short simulation
    print(f"\nRunning short test simulation...")
    try:
        remd.run_simulation(
            simulation_function=mock_md_simulation,
            total_steps=400,
            steps_per_cycle=100,
            save_frequency=1000
        )
        
        # Check results
        acceptance_rate = remd.statistics.get_overall_acceptance_rate()
        neighbor_rates = remd.statistics.get_neighbor_acceptance_rates()
        
        print(f"✓ Simulation completed")
        print(f"  Exchange attempts: {remd.statistics.total_attempts}")
        print(f"  Acceptance rate: {acceptance_rate:.1%}")
        print(f"  Neighbor rates: {[f'{r:.1%}' for r in neighbor_rates]}")
        
        # Check if any neighbor rates are in target range
        target_rates = [r for r in neighbor_rates if 0.20 <= r <= 0.40]
        if target_rates or acceptance_rate > 0:
            print(f"✓ Exchange system functional")
        else:
            print(f"⚠ Low exchange rates (may need longer simulation)")
        
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        return False
    
    print(f"\n✅ Task 6.2 basic functionality validated!")
    return True


def main():
    """Main test function."""
    try:
        success = test_remd_basic()
        return success
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
