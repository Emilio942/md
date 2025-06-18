"""
Memory Optimization Module for ProteinMD
========================================

This module implements Task 7.3 Memory Optimization features:

- Memory usage < 10MB per 1000 atoms
- Intelligent neighbor lists with O(N) complexity  
- Memory pool for frequent allocations
- Memory footprint analysis tools

Components:
- MemoryOptimizer: Main optimization system
- MemoryPool: Efficient array allocation
- OptimizedNeighborList: O(N) neighbor finding
- MemoryFootprintAnalyzer: Analysis and profiling tools
"""

from .memory_optimizer import (
    MemoryOptimizer,
    MemoryPool,
    OptimizedNeighborList,
    MemoryFootprintAnalyzer,
    MemoryMetrics
)

__all__ = [
    'MemoryOptimizer',
    'MemoryPool', 
    'OptimizedNeighborList',
    'MemoryFootprintAnalyzer',
    'MemoryMetrics'
]

__version__ = '1.0.0'
