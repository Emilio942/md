#!/usr/bin/env python3
"""
Memory profiling tools for molecular dynamics simulations.
Monitors memory usage during long-running simulations to detect memory leaks.
"""

import psutil
import gc
import time
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Dict, Any
import numpy as np
import threading
import sys


class MemoryProfiler:
    """Real-time memory profiler for MD simulations."""
    
    def __init__(self, max_samples: int = 1000):
        """
        Initialize memory profiler.
        
        Args:
            max_samples: Maximum number of memory samples to store
        """
        self.max_samples = max_samples
        self.memory_samples = deque(maxlen=max_samples)
        self.time_samples = deque(maxlen=max_samples)
        self.gc_counts = deque(maxlen=max_samples)
        
        self.process = psutil.Process()
        self.start_time = time.time()
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval: float = 1.0):
        """
        Start continuous memory monitoring.
        
        Args:
            interval: Sampling interval in seconds
        """
        if self.monitoring:
            return
            
        self.monitoring = True
        self.start_time = time.time()
        
        def monitor_loop():
            while self.monitoring:
                self.sample_memory()
                time.sleep(interval)
                
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
    def sample_memory(self):
        """Take a memory sample."""
        try:
            # Get memory info
            memory_info = self.process.memory_info()
            rss_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            
            # Get garbage collection stats
            gc_stats = gc.get_stats()
            total_collections = sum(stat['collections'] for stat in gc_stats)
            
            # Store samples
            current_time = time.time() - self.start_time
            self.memory_samples.append(rss_mb)
            self.time_samples.append(current_time)
            self.gc_counts.append(total_collections)
            
        except Exception as e:
            print(f"Error sampling memory: {e}")
            
    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
            
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if not self.memory_samples:
            return {}
            
        samples = list(self.memory_samples)
        return {
            'current_mb': samples[-1] if samples else 0.0,
            'peak_mb': max(samples),
            'min_mb': min(samples),
            'mean_mb': np.mean(samples),
            'std_mb': np.std(samples),
            'growth_rate_mb_per_min': self._calculate_growth_rate(),
            'total_samples': len(samples)
        }
        
    def _calculate_growth_rate(self) -> float:
        """Calculate memory growth rate in MB per minute."""
        if len(self.memory_samples) < 2:
            return 0.0
            
        times = list(self.time_samples)
        memories = list(self.memory_samples)
        
        # Linear regression to find growth rate
        time_diff = times[-1] - times[0]
        if time_diff < 1.0:  # Less than 1 second
            return 0.0
            
        memory_diff = memories[-1] - memories[0]
        growth_rate_per_sec = memory_diff / time_diff
        growth_rate_per_min = growth_rate_per_sec * 60
        
        return growth_rate_per_min
        
    def plot_memory_usage(self, save_path: str = None, show: bool = True):
        """
        Plot memory usage over time.
        
        Args:
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if not self.memory_samples:
            print("No memory samples to plot")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        times = list(self.time_samples)
        memories = list(self.memory_samples)
        gc_counts = list(self.gc_counts)
        
        # Memory usage plot
        ax1.plot(times, memories, 'b-', linewidth=1, alpha=0.7)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_title('Memory Usage Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        if len(times) > 1:
            z = np.polyfit(times, memories, 1)
            p = np.poly1d(z)
            ax1.plot(times, p(times), 'r--', alpha=0.8, 
                    label=f'Trend: {z[0]:.3f} MB/s')
            ax1.legend()
        
        # GC collections plot
        ax2.plot(times, gc_counts, 'g-', linewidth=1, alpha=0.7)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Total GC Collections')
        ax2.set_title('Garbage Collection Activity')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
            
    def check_memory_leak(self, threshold_mb_per_min: float = 1.0) -> Dict[str, Any]:
        """
        Check for potential memory leaks.
        
        Args:
            threshold_mb_per_min: Growth rate threshold for leak detection
            
        Returns:
            Dictionary with leak detection results
        """
        stats = self.get_memory_stats()
        
        if not stats:
            return {'leak_detected': False, 'reason': 'No data available'}
            
        growth_rate = stats.get('growth_rate_mb_per_min', 0.0)
        
        leak_detected = abs(growth_rate) > threshold_mb_per_min
        
        return {
            'leak_detected': leak_detected,
            'growth_rate_mb_per_min': growth_rate,
            'threshold_mb_per_min': threshold_mb_per_min,
            'current_memory_mb': stats['current_mb'],
            'peak_memory_mb': stats['peak_mb'],
            'memory_variance_mb': stats['std_mb'],
            'total_samples': stats['total_samples'],
            'monitoring_duration_min': (time.time() - self.start_time) / 60
        }
        
    def force_garbage_collection(self):
        """Force garbage collection and return collected objects count."""
        collected = gc.collect()
        return collected
        
    def print_memory_report(self):
        """Print detailed memory report."""
        stats = self.get_memory_stats()
        leak_check = self.check_memory_leak()
        
        print("\n" + "="*60)
        print("MEMORY PROFILER REPORT")
        print("="*60)
        
        if stats:
            print(f"Current Memory:     {stats['current_mb']:.2f} MB")
            print(f"Peak Memory:        {stats['peak_mb']:.2f} MB")
            print(f"Memory Range:       {stats['min_mb']:.2f} - {stats['peak_mb']:.2f} MB")
            print(f"Memory Std Dev:     {stats['std_mb']:.2f} MB")
            print(f"Growth Rate:        {stats['growth_rate_mb_per_min']:.3f} MB/min")
            print(f"Total Samples:      {stats['total_samples']}")
        
        print(f"\nMonitoring Duration: {leak_check['monitoring_duration_min']:.1f} minutes")
        
        if leak_check['leak_detected']:
            print(f"\n⚠️  MEMORY LEAK DETECTED!")
            print(f"Growth rate ({leak_check['growth_rate_mb_per_min']:.3f} MB/min) exceeds threshold ({leak_check['threshold_mb_per_min']} MB/min)")
        else:
            print(f"\n✅ No memory leak detected")
            print(f"Growth rate ({leak_check['growth_rate_mb_per_min']:.3f} MB/min) is within acceptable limits")
            
        # Force GC and report
        collected = self.force_garbage_collection()
        print(f"\nGarbage Collection: {collected} objects collected")
        
        print("="*60)


def profile_simulation(simulation_func, *args, **kwargs):
    """
    Profile memory usage of a simulation function.
    
    Args:
        simulation_func: Function to profile
        *args, **kwargs: Arguments for the simulation function
        
    Returns:
        Tuple of (simulation_result, memory_profiler)
    """
    profiler = MemoryProfiler()
    
    try:
        profiler.start_monitoring(interval=0.5)
        print("Starting memory profiling...")
        
        # Run the simulation
        result = simulation_func(*args, **kwargs)
        
        return result, profiler
        
    finally:
        profiler.stop_monitoring()
        profiler.print_memory_report()


if __name__ == "__main__":
    # Example usage
    profiler = MemoryProfiler()
    profiler.start_monitoring(interval=1.0)
    
    try:
        # Simulate some memory usage
        data = []
        for i in range(100):
            data.append(np.random.rand(1000))
            time.sleep(0.1)
            
    finally:
        profiler.stop_monitoring()
        profiler.print_memory_report()
        profiler.plot_memory_usage()
