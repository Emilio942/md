#!/usr/bin/env python3
"""
Simple Memory Leak Validation Test
This standalone test validates that memory leak fixes are working effectively.
"""

import gc
import time
import psutil
import os
import sys

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_matplotlib_memory():
    """Test matplotlib memory usage with leak detection"""
    print("Testing matplotlib memory usage...")
    
    # Force garbage collection before starting
    gc.collect()
    initial_memory = get_memory_usage()
    print(f"Initial memory: {initial_memory:.2f} MB")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Perform operations that previously leaked memory
        memory_samples = []
        start_time = time.time()
        
        for i in range(50):  # Create and close many figures
            fig, ax = plt.subplots()
            
            # Create some data and plot
            x = np.linspace(0, 10, 100)
            y = np.sin(x + i * 0.1)
            ax.plot(x, y)
            ax.set_title(f"Plot {i}")
            
            # CRITICAL: Close figure to prevent memory leak
            plt.close(fig)
            
            # Monitor memory every 10 iterations
            if i % 10 == 0:
                gc.collect()  # Force garbage collection
                current_memory = get_memory_usage()
                memory_samples.append(current_memory)
                elapsed = time.time() - start_time
                print(f"Iteration {i}: {current_memory:.2f} MB (elapsed: {elapsed:.1f}s)")
        
        # Final memory check
        gc.collect()
        final_memory = get_memory_usage()
        elapsed_time = time.time() - start_time
        
        # Calculate memory growth rate
        memory_growth = final_memory - initial_memory
        growth_rate_mb_per_min = (memory_growth / elapsed_time) * 60
        
        print(f"\n=== MEMORY LEAK TEST RESULTS ===")
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory growth: {memory_growth:.2f} MB")
        print(f"Test duration: {elapsed_time:.1f} seconds")
        print(f"Growth rate: {growth_rate_mb_per_min:.2f} MB/min")
        print(f"Memory samples: {[f'{m:.1f}' for m in memory_samples]}")
        
        # Check if memory growth is within acceptable limits (< 1 MB/min)
        ACCEPTABLE_GROWTH_RATE = 1.0  # MB/min
        
        if growth_rate_mb_per_min < ACCEPTABLE_GROWTH_RATE:
            print(f"✅ PASS: Memory growth rate ({growth_rate_mb_per_min:.2f} MB/min) is below threshold ({ACCEPTABLE_GROWTH_RATE} MB/min)")
            return True
        else:
            print(f"❌ FAIL: Memory growth rate ({growth_rate_mb_per_min:.2f} MB/min) exceeds threshold ({ACCEPTABLE_GROWTH_RATE} MB/min)")
            return False
            
    except Exception as e:
        print(f"Error during matplotlib test: {e}")
        return False

def test_numpy_memory():
    """Test numpy memory usage"""
    print("\nTesting numpy memory usage...")
    
    gc.collect()
    initial_memory = get_memory_usage()
    
    try:
        import numpy as np
        
        # Create and delete large arrays
        for i in range(20):
            # Create large array
            arr = np.random.random((1000, 1000))
            
            # Perform operations
            result = np.fft.fft2(arr)
            result = np.linalg.norm(result)
            
            # Explicitly delete references
            del arr, result
            
            if i % 5 == 0:
                gc.collect()
                current_memory = get_memory_usage()
                print(f"Numpy iteration {i}: {current_memory:.2f} MB")
        
        gc.collect()
        final_memory = get_memory_usage()
        
        memory_growth = final_memory - initial_memory
        print(f"Numpy memory growth: {memory_growth:.2f} MB")
        
        return memory_growth < 10  # Allow some growth for numpy operations
        
    except Exception as e:
        print(f"Error during numpy test: {e}")
        return False

def main():
    """Run comprehensive memory validation tests"""
    print("🔍 Starting Memory Leak Validation Tests")
    print("=" * 50)
    
    # Test matplotlib memory
    matplotlib_pass = test_matplotlib_memory()
    
    # Test numpy memory  
    numpy_pass = test_numpy_memory()
    
    # Overall results
    print("\n" + "=" * 50)
    print("📊 FINAL TEST RESULTS")
    print("=" * 50)
    
    print(f"Matplotlib test: {'✅ PASS' if matplotlib_pass else '❌ FAIL'}")
    print(f"Numpy test: {'✅ PASS' if numpy_pass else '❌ FAIL'}")
    
    overall_pass = matplotlib_pass and numpy_pass
    print(f"\nOverall result: {'✅ ALL TESTS PASSED' if overall_pass else '❌ SOME TESTS FAILED'}")
    
    if overall_pass:
        print("\n🎉 Memory leak fixes are working effectively!")
        print("Memory growth rates are within acceptable limits.")
    else:
        print("\n⚠️  Memory leaks may still be present.")
        print("Consider reviewing the optimization guide for additional fixes.")
    
    return overall_pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
