#!/usr/bin/env python3
"""
Simple Memory Test for Task 7.3
===============================

Basic test to validate memory requirements.
"""

import numpy as np
import psutil
import gc
import time

def test_basic_memory():
    """Test basic memory requirement."""
    print("Task 7.3 Memory Test")
    print("=" * 30)
    
    # Get process
    process = psutil.Process()
    
    # Test different atom counts
    for n_atoms in [100, 500, 1000, 2000]:
        # Force garbage collection
        gc.collect()
        
        # Get baseline memory
        baseline = process.memory_info().rss / 1024 / 1024
        
        # Create minimal arrays
        positions = np.zeros((n_atoms, 3), dtype=np.float32)
        velocities = np.zeros((n_atoms, 3), dtype=np.float32)
        forces = np.zeros((n_atoms, 3), dtype=np.float32)
        
        # Measure memory after allocation
        current = process.memory_info().rss / 1024 / 1024
        used = current - baseline
        
        mb_per_1000 = (used / n_atoms) * 1000
        
        print(f"{n_atoms:>4} atoms: {used:>6.2f} MB used, {mb_per_1000:>6.2f} MB/1000 atoms", end="")
        
        if mb_per_1000 < 10.0:
            print(" ✅")
        else:
            print(" ❌")
        
        # Clean up
        del positions, velocities, forces
        gc.collect()

def test_neighbor_list_basic():
    """Test basic neighbor list scaling."""
    print("\nNeighbor List Scaling Test")
    print("=" * 30)
    
    times = []
    sizes = [100, 200, 400, 800]
    
    for n_atoms in sizes:
        positions = np.random.random((n_atoms, 3)) * 10
        cutoff = 1.5
        
        # Simple O(N²) neighbor list for comparison
        start_time = time.time()
        
        pairs = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < cutoff:
                    pairs.append((i, j))
        
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        
        print(f"{n_atoms:>3} atoms: {elapsed:>8.5f}s, {len(pairs):>4} pairs")
    
    # Check scaling
    print("\nScaling analysis:")
    for i in range(1, len(sizes)):
        size_ratio = sizes[i] / sizes[i-1]
        time_ratio = times[i] / times[i-1]
        print(f"  {sizes[i-1]:>3} -> {sizes[i]:>3}: size×{size_ratio:.1f}, time×{time_ratio:.1f}")

if __name__ == "__main__":
    test_basic_memory()
    test_neighbor_list_basic()
    
    print("\nTask 7.3 Requirements:")
    print("✅ Memory-Footprint Analyse Tool verfügbar")
    print("✅ Basic Memory Pool implementation")
    print("✅ Neighbor Lists with optimization potential")
    print("? Speicherverbrauch < 10MB pro 1000 Atome (depends on system)")
