#!/usr/bin/env python3
"""
Enhanced memory leak test that accounts for initial allocation period.
"""

import sys
import os
import time
import gc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_profiler import MemoryProfiler
from optimized_simulation import MemoryOptimizedMDSimulation


def test_steady_state_memory_usage():
    """
    Test memory usage after initial allocation period to get accurate leak detection.
    """
    print("ENHANCED MEMORY LEAK TEST - STEADY STATE ANALYSIS")
    print("="*70)
    
    # Create simulation
    sim = MemoryOptimizedMDSimulation(
        num_molecules=20,
        box_size=15.0,
        temperature=300,
        max_trajectory_length=1000,
        max_energy_history=5000
    )
    
    print("Phase 1: Initial allocation and warmup...")
    # Run initial phase to allow memory to stabilize
    warmup_steps = 2000
    for i in range(warmup_steps):
        sim.step_simulation()
        if i % 500 == 0:
            print(f"  Warmup step {i}")
    
    # Force garbage collection
    gc.collect()
    time.sleep(1)
    
    print("Phase 2: Memory monitoring during steady state...")
    # Start monitoring after warmup
    profiler = MemoryProfiler(max_samples=1000)
    profiler.start_monitoring(interval=1.0)
    
    try:
        # Run steady state simulation
        steady_state_steps = 10000
        for i in range(steady_state_steps):
            sim.step_simulation()
            
            if i % 1000 == 0:
                current_memory = profiler.get_current_memory()
                print(f"  Steady state step {i}, Memory: {current_memory:.2f} MB")
                
            # Periodic cleanup
            if i % 2000 == 0:
                gc.collect()
                
    finally:
        profiler.stop_monitoring()
    
    # Analyze steady state results
    stats = profiler.get_memory_stats()
    leak_check = profiler.check_memory_leak(threshold_mb_per_min=0.5)  # Stricter threshold
    
    print("\n" + "="*70)
    print("STEADY STATE MEMORY ANALYSIS RESULTS")
    print("="*70)
    
    if stats:
        print(f"Steady State Memory Range: {stats['min_mb']:.2f} - {stats['peak_mb']:.2f} MB")
        print(f"Memory Variance: {stats['std_mb']:.2f} MB")
        print(f"Growth Rate: {stats['growth_rate_mb_per_min']:.3f} MB/min")
        print(f"Monitoring Duration: {leak_check['monitoring_duration_min']:.1f} minutes")
    
    # Determine if memory usage is stable
    memory_stable = abs(stats['growth_rate_mb_per_min']) < 0.5
    variance_acceptable = stats['std_mb'] < 2.0
    
    print(f"\nMemory Stability Analysis:")
    print(f"✅ Growth rate acceptable: {memory_stable} ({stats['growth_rate_mb_per_min']:.3f} < 0.5 MB/min)")
    print(f"✅ Memory variance acceptable: {variance_acceptable} ({stats['std_mb']:.2f} < 2.0 MB)")
    
    overall_success = memory_stable and variance_acceptable
    
    if overall_success:
        print(f"\n🎉 MEMORY LEAK TEST PASSED!")
        print("✅ Memory usage is stable in steady state")
        print("✅ No significant memory growth detected")
    else:
        print(f"\n❌ MEMORY LEAK TEST FAILED!")
        print("⚠️  Memory usage shows concerning patterns")
    
    # Save detailed plot
    profiler.plot_memory_usage(
        save_path="/home/emilio/Documents/ai/md/steady_state_memory_test.png",
        show=False
    )
    
    return overall_success


def validate_memory_bounds():
    """
    Validate that memory-bounded data structures work correctly.
    """
    print("\nVALIDATING MEMORY-BOUNDED DATA STRUCTURES")
    print("="*50)
    
    sim = MemoryOptimizedMDSimulation(
        num_molecules=5,
        max_trajectory_length=100,  # Small limit for testing
        max_energy_history=200
    )
    
    print(f"Initial stats: {sim.get_memory_stats()}")
    
    # Run simulation to exceed bounds
    for i in range(300):  # More than the limits
        sim.step_simulation()
    
    final_stats = sim.get_memory_stats()
    print(f"Final stats: {final_stats}")
    
    # Check bounds are respected
    trajectory_bounded = final_stats['trajectory_length'] <= final_stats['max_trajectory_length']
    energy_bounded = final_stats['energy_history_length'] <= final_stats['max_energy_history']
    
    print(f"\nBounds validation:")
    print(f"✅ Trajectory bounded: {trajectory_bounded} ({final_stats['trajectory_length']} <= {final_stats['max_trajectory_length']})")
    print(f"✅ Energy history bounded: {energy_bounded} ({final_stats['energy_history_length']} <= {final_stats['max_energy_history']})")
    
    return trajectory_bounded and energy_bounded


if __name__ == "__main__":
    print("ENHANCED MEMORY LEAK VALIDATION")
    print("="*50)
    
    # Test memory bounds
    bounds_ok = validate_memory_bounds()
    
    # Test steady state memory usage
    steady_state_ok = test_steady_state_memory_usage()
    
    print(f"\n{'='*70}")
    print("FINAL VALIDATION RESULTS")
    print("="*70)
    
    if bounds_ok and steady_state_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Task 1.3 Memory Leak Behebung COMPLETED!")
        print("✅ Simulation of 10,000+ steps shows constant memory usage")
        print("✅ No continuous memory increase detected")
        print("✅ Memory profiling shows stable allocation patterns")
        print("✅ Memory-bounded data structures work correctly")
    else:
        print("❌ SOME TESTS FAILED!")
        if not bounds_ok:
            print("❌ Memory bounds validation failed")
        if not steady_state_ok:
            print("❌ Steady state memory test failed")
    
    print("="*70)
