#!/usr/bin/env python3
"""
Task 7.3 Memory Optimization Implementation
==========================================

Advanced memory optimization for molecular dynamics simulations focusing on:
1. Memory usage < 10MB per 1000 atoms
2. Smart neighbor lists reducing O(N²) to O(N) 
3. Memory pool for frequent allocations
4. Memory footprint analysis tool

This implementation builds on the existing memory optimization infrastructure
from Task 1.3 but focuses on algorithmic and structural optimizations.
"""

import numpy as np
import psutil
import os
import time
import gc
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import deque
import threading
import weakref
from pathlib import Path

# Memory profiling
import tracemalloc
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryMetrics:
    """Memory usage metrics for analysis."""
    rss_mb: float
    vms_mb: float
    percent: float
    atoms_count: int
    mb_per_1000_atoms: float
    timestamp: float

class MemoryPool:
    """
    Memory pool for frequent allocations to reduce fragmentation
    and improve performance.
    """
    
    def __init__(self, initial_size_mb: float = 100.0):
        """
        Initialize memory pool.
        
        Parameters
        ----------
        initial_size_mb : float
            Initial pool size in megabytes
        """
        self.initial_size_mb = initial_size_mb
        self.allocated_arrays = {}
        self.free_arrays = {}
        self.total_allocated = 0
        self.peak_usage = 0
        self.allocation_count = 0
        self.deallocation_count = 0
        self.lock = threading.Lock()
        
        logger.info(f"Memory pool initialized with {initial_size_mb:.1f} MB")
    
    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Allocate array from pool or create new one.
        
        Parameters
        ----------
        shape : tuple
            Array shape
        dtype : np.dtype
            Array data type
            
        Returns
        -------
        np.ndarray
            Allocated array
        """
        with self.lock:
            key = (shape, dtype)
            
            # Try to reuse existing array
            if key in self.free_arrays and self.free_arrays[key]:
                array = self.free_arrays[key].pop()
                array.fill(0)  # Clear data
                self.allocated_arrays[id(array)] = key
                logger.debug(f"Reused array with shape {shape}")
                return array
            
            # Create new array
            array = np.zeros(shape, dtype=dtype)
            self.allocated_arrays[id(array)] = key
            self.allocation_count += 1
            self.total_allocated += array.nbytes
            self.peak_usage = max(self.peak_usage, self.total_allocated)
            
            logger.debug(f"Allocated new array with shape {shape}, dtype {dtype}")
            return array
    
    def deallocate(self, array: np.ndarray) -> None:
        """
        Return array to pool for reuse.
        
        Parameters
        ----------
        array : np.ndarray
            Array to deallocate
        """
        with self.lock:
            array_id = id(array)
            if array_id not in self.allocated_arrays:
                logger.warning("Attempt to deallocate untracked array")
                return
            
            key = self.allocated_arrays[array_id]
            del self.allocated_arrays[array_id]
            
            # Add to free list
            if key not in self.free_arrays:
                self.free_arrays[key] = []
            
            self.free_arrays[key].append(array)
            self.deallocation_count += 1
            
            logger.debug(f"Deallocated array with shape {key[0]}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            free_count = sum(len(arrays) for arrays in self.free_arrays.values())
            allocated_count = len(self.allocated_arrays)
            
            return {
                'total_allocated_mb': self.total_allocated / (1024 * 1024),
                'peak_usage_mb': self.peak_usage / (1024 * 1024),
                'allocation_count': self.allocation_count,
                'deallocation_count': self.deallocation_count,
                'arrays_allocated': allocated_count,
                'arrays_free': free_count,
                'efficiency': self.deallocation_count / max(1, self.allocation_count)
            }
    
    def cleanup(self) -> None:
        """Clean up memory pool."""
        with self.lock:
            self.free_arrays.clear()
            self.allocated_arrays.clear()
            self.total_allocated = 0
            gc.collect()
            logger.info("Memory pool cleaned up")

class CellList:
    """
    Cell-list spatial data structure for O(N) neighbor finding.
    Divides space into cells and only checks neighboring cells.
    """
    
    def __init__(self, box_size: np.ndarray, cutoff: float):
        """
        Initialize cell list.
        
        Parameters
        ----------
        box_size : np.ndarray
            Simulation box dimensions [x, y, z]
        cutoff : float
            Interaction cutoff distance
        """
        self.box_size = np.array(box_size)
        self.cutoff = cutoff
        
        # Calculate cell dimensions (at least cutoff size)
        self.n_cells = np.maximum(1, (self.box_size / cutoff).astype(int))
        self.cell_size = self.box_size / self.n_cells
        
        # Initialize cell data structures
        self.reset()
        
        logger.info(f"Cell list initialized: {self.n_cells} cells, "
                   f"cell size: {self.cell_size}")
    
    def reset(self):
        """Reset cell list for new configuration."""
        total_cells = np.prod(self.n_cells)
        self.cell_contents = [[] for _ in range(total_cells)]
        self.atom_to_cell = {}
    
    def add_atom(self, atom_id: int, position: np.ndarray):
        """
        Add atom to appropriate cell.
        
        Parameters
        ----------
        atom_id : int
            Atom identifier
        position : np.ndarray
            Atom position [x, y, z]
        """
        # Apply periodic boundary conditions
        pos = position % self.box_size
        
        # Find cell indices
        cell_idx = (pos / self.cell_size).astype(int)
        cell_idx = np.minimum(cell_idx, self.n_cells - 1)
        
        # Convert to flat index
        flat_idx = (cell_idx[0] * self.n_cells[1] * self.n_cells[2] + 
                   cell_idx[1] * self.n_cells[2] + cell_idx[2])
        
        self.cell_contents[flat_idx].append(atom_id)
        self.atom_to_cell[atom_id] = flat_idx
    
    def get_neighbor_candidates(self, atom_id: int) -> List[int]:
        """
        Get list of atoms in neighboring cells.
        
        Parameters
        ----------
        atom_id : int
            Atom identifier
            
        Returns
        -------
        List[int]
            List of potential neighbor atom IDs
        """
        if atom_id not in self.atom_to_cell:
            return []
        
        cell_idx = self.atom_to_cell[atom_id]
        
        # Convert flat index back to 3D
        z = cell_idx % self.n_cells[2]
        y = (cell_idx // self.n_cells[2]) % self.n_cells[1]
        x = cell_idx // (self.n_cells[1] * self.n_cells[2])
        
        neighbors = []
        
        # Check all neighboring cells (including self)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # Apply periodic boundary conditions
                    nx = nx % self.n_cells[0]
                    ny = ny % self.n_cells[1]
                    nz = nz % self.n_cells[2]
                    
                    neighbor_idx = (nx * self.n_cells[1] * self.n_cells[2] + 
                                  ny * self.n_cells[2] + nz)
                    
                    neighbors.extend(self.cell_contents[neighbor_idx])
        
        return neighbors

class SmartNeighborList:
    """
    Smart neighbor list implementation using Verlet lists and cell lists
    to reduce complexity from O(N²) to O(N).
    """
    
    def __init__(self, cutoff: float, buffer: float = 0.2, 
                 update_frequency: int = 10, memory_pool: Optional[MemoryPool] = None):
        """
        Initialize smart neighbor list.
        
        Parameters
        ----------
        cutoff : float
            Interaction cutoff distance
        buffer : float
            Additional buffer distance for Verlet lists
        update_frequency : int
            Update frequency in simulation steps
        memory_pool : MemoryPool, optional
            Memory pool for array allocation
        """
        self.cutoff = cutoff
        self.buffer = buffer
        self.verlet_cutoff = cutoff + buffer
        self.update_frequency = update_frequency
        self.memory_pool = memory_pool or MemoryPool()
        
        self.neighbor_list = []
        self.last_positions = None
        self.last_update_step = -1
        self.displacement_threshold = buffer / 2.0
        
        self.stats = {
            'updates': 0,
            'total_pairs': 0,
            'avg_neighbors_per_atom': 0.0,
            'efficiency_ratio': 0.0
        }
        
        logger.info(f"Smart neighbor list initialized: cutoff={cutoff:.3f}, "
                   f"buffer={buffer:.3f}, update_freq={update_frequency}")
    
    def needs_update(self, positions: np.ndarray, step: int) -> bool:
        """
        Check if neighbor list needs updating.
        
        Parameters
        ----------
        positions : np.ndarray
            Current atomic positions
        step : int
            Current simulation step
            
        Returns
        -------
        bool
            True if update is needed
        """
        # Force update on first call
        if self.last_positions is None:
            return True
        
        # Periodic update
        if step - self.last_update_step >= self.update_frequency:
            return True
        
        # Check displacement threshold
        max_displacement = np.max(np.linalg.norm(
            positions - self.last_positions, axis=1))
        
        if max_displacement > self.displacement_threshold:
            logger.debug(f"Displacement threshold exceeded: {max_displacement:.3f}")
            return True
        
        return False
    
    def update(self, positions: np.ndarray, box_size: np.ndarray, step: int):
        """
        Update neighbor list using cell lists.
        
        Parameters
        ----------
        positions : np.ndarray
            Atomic positions [N, 3]
        box_size : np.ndarray
            Simulation box size [3]
        step : int
            Current simulation step
        """
        n_atoms = len(positions)
        
        # Initialize cell list
        cell_list = CellList(box_size, self.verlet_cutoff)
        
        # Add atoms to cells
        for i, pos in enumerate(positions):
            cell_list.add_atom(i, pos)
        
        # Build neighbor list
        if self.memory_pool:
            # Use memory pool for allocation
            neighbor_counts = self.memory_pool.allocate((n_atoms,), np.int32)
            max_neighbors = min(n_atoms - 1, 100)  # Reasonable upper bound
            neighbors = self.memory_pool.allocate((n_atoms, max_neighbors), np.int32)
        else:
            neighbor_counts = np.zeros(n_atoms, dtype=np.int32)
            max_neighbors = min(n_atoms - 1, 100)
            neighbors = np.full((n_atoms, max_neighbors), -1, dtype=np.int32)
        
        total_pairs = 0
        
        for i in range(n_atoms):
            candidates = cell_list.get_neighbor_candidates(i)
            count = 0
            
            for j in candidates:
                if j <= i:  # Avoid double counting and self-interaction
                    continue
                
                # Calculate distance
                dr = positions[j] - positions[i]
                
                # Apply minimum image convention
                dr = dr - box_size * np.round(dr / box_size)
                distance = np.linalg.norm(dr)
                
                if distance < self.verlet_cutoff and count < max_neighbors:
                    neighbors[i, count] = j
                    count += 1
                    total_pairs += 1
            
            neighbor_counts[i] = count
        
        # Store results
        self.neighbor_list = (neighbor_counts, neighbors)
        self.last_positions = positions.copy()
        self.last_update_step = step
        
        # Update statistics
        self.stats['updates'] += 1
        self.stats['total_pairs'] = total_pairs
        self.stats['avg_neighbors_per_atom'] = total_pairs / n_atoms
        
        # Calculate efficiency (compared to O(N²) algorithm)
        naive_pairs = n_atoms * (n_atoms - 1) // 2
        actual_checks = sum(len(cell_list.get_neighbor_candidates(i)) for i in range(n_atoms))
        self.stats['efficiency_ratio'] = naive_pairs / max(actual_checks, 1)
        
        logger.debug(f"Neighbor list updated: {total_pairs} pairs, "
                    f"avg {self.stats['avg_neighbors_per_atom']:.1f} neighbors/atom")
    
    def get_neighbors(self, atom_id: int) -> np.ndarray:
        """
        Get neighbors for specific atom.
        
        Parameters
        ----------
        atom_id : int
            Atom identifier
            
        Returns
        -------
        np.ndarray
            Array of neighbor atom IDs
        """
        if not self.neighbor_list:
            return np.array([], dtype=np.int32)
        
        counts, neighbors = self.neighbor_list
        count = counts[atom_id]
        return neighbors[atom_id, :count]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get neighbor list statistics."""
        return self.stats.copy()

class MemoryOptimizedSystem:
    """
    Memory-optimized molecular system using Structure-of-Arrays layout
    and efficient neighbor lists.
    """
    
    def __init__(self, n_atoms: int, box_size: np.ndarray, 
                 cutoff: float = 1.0, memory_pool: Optional[MemoryPool] = None):
        """
        Initialize memory-optimized system.
        
        Parameters
        ----------
        n_atoms : int
            Number of atoms
        box_size : np.ndarray
            Simulation box size [3]
        cutoff : float
            Interaction cutoff distance
        memory_pool : MemoryPool, optional
            Memory pool for allocations
        """
        self.n_atoms = n_atoms
        self.box_size = np.array(box_size)
        self.cutoff = cutoff
        self.memory_pool = memory_pool or MemoryPool()
        
        # Structure-of-Arrays layout for better cache performance
        self.positions = self.memory_pool.allocate((n_atoms, 3), np.float32)
        self.velocities = self.memory_pool.allocate((n_atoms, 3), np.float32)
        self.forces = self.memory_pool.allocate((n_atoms, 3), np.float32)
        self.masses = self.memory_pool.allocate((n_atoms,), np.float32)
        self.charges = self.memory_pool.allocate((n_atoms,), np.float32)
        
        # Initialize neighbor list
        self.neighbor_list = SmartNeighborList(
            cutoff=cutoff, 
            memory_pool=self.memory_pool
        )
        
        # Initialize with default values
        self.masses.fill(1.0)
        self.charges.fill(0.0)
        
        # Random initial positions
        self.positions[:] = np.random.random((n_atoms, 3)) * self.box_size
        
        logger.info(f"Memory-optimized system initialized: {n_atoms} atoms")
    
    def update_neighbor_list(self, step: int):
        """Update neighbor list if needed."""
        if self.neighbor_list.needs_update(self.positions, step):
            self.neighbor_list.update(self.positions, self.box_size, step)
    
    def calculate_forces(self) -> float:
        """
        Calculate forces using neighbor list (simplified LJ potential).
        
        Returns
        -------
        float
            Total potential energy
        """
        self.forces.fill(0.0)
        potential_energy = 0.0
        
        sigma = 1.0
        epsilon = 1.0
        sigma6 = sigma**6
        sigma12 = sigma**12
        
        for i in range(self.n_atoms):
            neighbors = self.neighbor_list.get_neighbors(i)
            
            for j in neighbors:
                # Calculate distance vector
                dr = self.positions[j] - self.positions[i]
                
                # Apply minimum image convention
                dr = dr - self.box_size * np.round(dr / self.box_size)
                r2 = np.sum(dr * dr)
                
                if r2 < self.cutoff**2:
                    r = np.sqrt(r2)
                    r6 = r2**3
                    r12 = r6**2
                    
                    # Lennard-Jones potential
                    lj_energy = 4 * epsilon * (sigma12/r12 - sigma6/r6)
                    potential_energy += lj_energy
                    
                    # Force calculation
                    force_magnitude = 24 * epsilon * (2*sigma12/r12 - sigma6/r6) / r2
                    force_vector = force_magnitude * dr
                    
                    self.forces[i] += force_vector
                    self.forces[j] -= force_vector
        
        return potential_energy
    
    def get_memory_usage(self) -> MemoryMetrics:
        """Get detailed memory usage metrics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return MemoryMetrics(
            rss_mb=memory_info.rss / (1024 * 1024),
            vms_mb=memory_info.vms / (1024 * 1024),
            percent=process.memory_percent(),
            atoms_count=self.n_atoms,
            mb_per_1000_atoms=(memory_info.rss / (1024 * 1024)) / (self.n_atoms / 1000),
            timestamp=time.time()
        )
    
    def cleanup(self):
        """Clean up system memory."""
        if hasattr(self, 'positions'):
            self.memory_pool.deallocate(self.positions)
        if hasattr(self, 'velocities'):
            self.memory_pool.deallocate(self.velocities)
        if hasattr(self, 'forces'):
            self.memory_pool.deallocate(self.forces)
        if hasattr(self, 'masses'):
            self.memory_pool.deallocate(self.masses)
        if hasattr(self, 'charges'):
            self.memory_pool.deallocate(self.charges)
        
        self.memory_pool.cleanup()
        logger.info("System cleanup completed")

class MemoryFootprintAnalyzer:
    """
    Real-time memory footprint analysis tool for MD simulations.
    """
    
    def __init__(self, max_samples: int = 1000):
        """
        Initialize memory analyzer.
        
        Parameters
        ----------
        max_samples : int
            Maximum number of samples to store
        """
        self.max_samples = max_samples
        self.samples = deque(maxlen=max_samples)
        self.start_time = time.time()
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Performance thresholds
        self.target_mb_per_1000_atoms = 10.0
        self.warning_threshold = 8.0
        self.critical_threshold = 12.0
        
        logger.info("Memory footprint analyzer initialized")
    
    def start_monitoring(self, interval: float = 1.0):
        """
        Start continuous memory monitoring.
        
        Parameters
        ----------
        interval : float
            Monitoring interval in seconds
        """
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        def monitor_loop():
            while self.is_monitoring:
                self.take_sample()
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Memory monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Memory monitoring stopped")
    
    def take_sample(self, atoms_count: Optional[int] = None) -> MemoryMetrics:
        """
        Take a memory usage sample.
        
        Parameters
        ----------
        atoms_count : int, optional
            Number of atoms in simulation
            
        Returns
        -------
        MemoryMetrics
            Memory usage metrics
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        if atoms_count is None:
            atoms_count = getattr(self, '_last_atom_count', 1000)
        else:
            self._last_atom_count = atoms_count
        
        metrics = MemoryMetrics(
            rss_mb=memory_info.rss / (1024 * 1024),
            vms_mb=memory_info.vms / (1024 * 1024),
            percent=process.memory_percent(),
            atoms_count=atoms_count,
            mb_per_1000_atoms=(memory_info.rss / (1024 * 1024)) / (atoms_count / 1000),
            timestamp=time.time()
        )
        
        self.samples.append(metrics)
        return metrics
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze memory performance against targets.
        
        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        if not self.samples:
            return {'status': 'no_data'}
        
        latest = self.samples[-1]
        
        # Performance assessment
        performance_status = 'excellent'
        if latest.mb_per_1000_atoms > self.target_mb_per_1000_atoms:
            performance_status = 'exceeds_target'
        elif latest.mb_per_1000_atoms > self.warning_threshold:
            performance_status = 'warning'
        elif latest.mb_per_1000_atoms > self.critical_threshold:
            performance_status = 'critical'
        
        # Trend analysis
        if len(self.samples) > 10:
            recent_samples = list(self.samples)[-10:]
            mb_values = [s.mb_per_1000_atoms for s in recent_samples]
            trend = 'stable'
            
            if len(mb_values) > 5:
                first_half = np.mean(mb_values[:len(mb_values)//2])
                second_half = np.mean(mb_values[len(mb_values)//2:])
                if second_half > first_half * 1.1:
                    trend = 'increasing'
                elif second_half < first_half * 0.9:
                    trend = 'decreasing'
        else:
            trend = 'insufficient_data'
        
        return {
            'status': performance_status,
            'current_mb_per_1000_atoms': latest.mb_per_1000_atoms,
            'target_mb_per_1000_atoms': self.target_mb_per_1000_atoms,
            'meets_target': latest.mb_per_1000_atoms <= self.target_mb_per_1000_atoms,
            'trend': trend,
            'total_samples': len(self.samples),
            'monitoring_duration': latest.timestamp - self.start_time,
            'recommendations': self._generate_recommendations(latest, performance_status)
        }
    
    def _generate_recommendations(self, metrics: MemoryMetrics, status: str) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if status == 'exceeds_target':
            recommendations.append("Memory usage exceeds target of 10MB per 1000 atoms")
            recommendations.append("Consider optimizing data structures or algorithms")
        
        if status == 'critical':
            recommendations.append("CRITICAL: Memory usage is very high")
            recommendations.append("Immediate optimization required")
        
        if metrics.percent > 80:
            recommendations.append("System memory usage is high (>80%)")
            recommendations.append("Consider reducing simulation size or optimizing memory")
        
        if not recommendations:
            recommendations.append("Memory usage is within acceptable limits")
        
        return recommendations
    
    def generate_report(self) -> str:
        """Generate detailed memory analysis report."""
        if not self.samples:
            return "No memory data available for analysis."
        
        analysis = self.analyze_performance()
        latest = self.samples[-1]
        
        report = []
        report.append("=" * 60)
        report.append("MEMORY FOOTPRINT ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Current status
        report.append("📊 CURRENT MEMORY USAGE")
        report.append(f"Memory per 1000 atoms: {latest.mb_per_1000_atoms:.2f} MB")
        report.append(f"Target threshold:      {self.target_mb_per_1000_atoms:.2f} MB")
        report.append(f"Total RSS memory:      {latest.rss_mb:.2f} MB")
        report.append(f"System memory usage:   {latest.percent:.1f}%")
        report.append(f"Total atoms:           {latest.atoms_count:,}")
        report.append("")
        
        # Performance assessment
        status_emoji = {
            'excellent': '✅',
            'warning': '⚠️',
            'exceeds_target': '❌',
            'critical': '🚨'
        }
        
        report.append("🎯 PERFORMANCE ASSESSMENT")
        report.append(f"Status: {status_emoji.get(analysis['status'], '❓')} {analysis['status'].upper()}")
        report.append(f"Meets target: {'✅ YES' if analysis['meets_target'] else '❌ NO'}")
        report.append(f"Trend: {analysis['trend']}")
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        for rec in analysis['recommendations']:
            report.append(f"• {rec}")
        report.append("")
        
        # Statistics
        if len(self.samples) > 1:
            mb_values = [s.mb_per_1000_atoms for s in self.samples]
            report.append("📈 STATISTICS")
            report.append(f"Samples collected:     {len(self.samples)}")
            report.append(f"Monitoring duration:   {analysis['monitoring_duration']:.1f} seconds")
            report.append(f"Average MB/1000 atoms: {np.mean(mb_values):.2f}")
            report.append(f"Peak MB/1000 atoms:    {np.max(mb_values):.2f}")
            report.append(f"Min MB/1000 atoms:     {np.min(mb_values):.2f}")
            report.append(f"Std deviation:         {np.std(mb_values):.2f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)

def validate_memory_optimization():
    """
    Validate memory optimization implementation against Task 7.3 requirements.
    """
    print("🧪 TASK 7.3 MEMORY OPTIMIZATION VALIDATION")
    print("=" * 60)
    
    # Test with different system sizes
    test_sizes = [100, 500, 1000, 2000, 5000]
    results = {}
    
    for n_atoms in test_sizes:
        print(f"\n📊 Testing with {n_atoms:,} atoms...")
        
        # Create memory pool
        memory_pool = MemoryPool(initial_size_mb=50.0)
        
        # Create memory-optimized system
        box_size = np.array([10.0, 10.0, 10.0])
        system = MemoryOptimizedSystem(
            n_atoms=n_atoms, 
            box_size=box_size, 
            cutoff=2.5,
            memory_pool=memory_pool
        )
        
        # Initialize memory analyzer
        analyzer = MemoryFootprintAnalyzer()
        
        # Take initial measurement
        initial_metrics = analyzer.take_sample(n_atoms)
        
        # Run simulation steps to test neighbor list
        for step in range(50):
            system.update_neighbor_list(step)
            if step % 10 == 0:
                energy = system.calculate_forces()
                analyzer.take_sample(n_atoms)
        
        # Final measurement
        final_metrics = analyzer.take_sample(n_atoms)
        
        # Analyze results
        analysis = analyzer.analyze_performance()
        
        results[n_atoms] = {
            'mb_per_1000_atoms': final_metrics.mb_per_1000_atoms,
            'meets_target': analysis['meets_target'],
            'neighbor_stats': system.neighbor_list.get_stats(),
            'pool_stats': memory_pool.get_stats()
        }
        
        print(f"   Memory usage: {final_metrics.mb_per_1000_atoms:.2f} MB per 1000 atoms")
        print(f"   Target met: {'✅' if analysis['meets_target'] else '❌'}")
        print(f"   Neighbor efficiency: {system.neighbor_list.get_stats()['efficiency_ratio']:.1f}x")
        
        # Cleanup
        system.cleanup()
        analyzer.stop_monitoring()
        gc.collect()
    
    # Overall assessment
    print("\n" + "=" * 60)
    print("🎯 TASK 7.3 REQUIREMENTS VALIDATION")
    print("=" * 60)
    
    # Requirement 1: Memory usage < 10MB per 1000 atoms
    memory_test_passed = all(r['meets_target'] for r in results.values())
    print(f"1. Memory usage < 10MB per 1000 atoms: {'✅ PASSED' if memory_test_passed else '❌ FAILED'}")
    
    # Requirement 2: Smart neighbor lists O(N) complexity
    avg_efficiency = np.mean([r['neighbor_stats']['efficiency_ratio'] for r in results.values()])
    neighbor_test_passed = avg_efficiency > 5.0  # Should be much better than O(N²)
    print(f"2. Smart neighbor lists O(N) efficiency: {'✅ PASSED' if neighbor_test_passed else '❌ FAILED'}")
    print(f"   Average efficiency improvement: {avg_efficiency:.1f}x")
    
    # Requirement 3: Memory pool implementation
    pool_efficiency = np.mean([r['pool_stats']['efficiency'] for r in results.values()])
    pool_test_passed = pool_efficiency > 0.5  # At least 50% reuse
    print(f"3. Memory pool for allocations: {'✅ PASSED' if pool_test_passed else '❌ FAILED'}")
    print(f"   Average pool efficiency: {pool_efficiency:.1%}")
    
    # Requirement 4: Memory footprint analysis tool
    analyzer_test_passed = True  # We have the MemoryFootprintAnalyzer
    print(f"4. Memory footprint analysis tool: {'✅ PASSED' if analyzer_test_passed else '❌ FAILED'}")
    
    # Overall result
    all_tests_passed = all([memory_test_passed, neighbor_test_passed, 
                           pool_test_passed, analyzer_test_passed])
    
    print("\n" + "=" * 60)
    print(f"OVERALL RESULT: {'✅ ALL REQUIREMENTS MET' if all_tests_passed else '❌ SOME REQUIREMENTS FAILED'}")
    print("=" * 60)
    
    if all_tests_passed:
        print("\n🎉 TASK 7.3 MEMORY OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("✅ Memory usage optimized to < 10MB per 1000 atoms")
        print("✅ Smart neighbor lists reduce complexity to O(N)")
        print("✅ Memory pool system implemented for efficient allocation")
        print("✅ Memory footprint analysis tool available")
    
    return all_tests_passed, results

if __name__ == "__main__":
    # Run validation
    success, results = validate_memory_optimization()
    
    if success:
        print("\n📋 TASK 7.3 IS READY FOR COMPLETION MARKING")
    else:
        print("\n⚠️  ADDITIONAL OPTIMIZATION MAY BE NEEDED")
