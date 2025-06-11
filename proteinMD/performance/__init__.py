"""
Performance optimization module for proteinMD.

This module contains GPU acceleration and other performance optimizations
for molecular dynamics simulations, implementing Task 7.2 requirements.

Features:
- GPU kernels for Lennard-Jones and Coulomb force calculations
- Automatic CPU/GPU fallback mechanism
- Performance improvements >5x for large systems (>1000 atoms)
- Compatibility with CUDA and OpenCL backends
"""

import logging
logger = logging.getLogger(__name__)

# Initialize availability flags
GPU_AVAILABLE = False
GPU_FORCE_TERMS_AVAILABLE = False
GPU_TESTING_AVAILABLE = False

# Import GPU acceleration with fallback handling
try:
    from .gpu_acceleration import (
        GPUAccelerator,
        is_gpu_available,
        get_gpu_device_info,
        LennardJonesGPU,
        CoulombGPU,
        GPUBackend
    )
    GPU_AVAILABLE = True
    logger.info("GPU acceleration modules loaded successfully")
except ImportError as e:
    logger.warning(f"GPU acceleration not available: {e}")
    # Create dummy classes for when GPU dependencies are not available
    
    class DummyGPUAccelerator:
        def __init__(self, *args, **kwargs):
            pass
        def is_gpu_enabled(self):
            return False
    
    GPUAccelerator = DummyGPUAccelerator
    LennardJonesGPU = DummyGPUAccelerator
    CoulombGPU = DummyGPUAccelerator
    
    def is_gpu_available():
        return False
    
    def get_gpu_device_info():
        return []
    
    class GPUBackend:
        CPU = "cpu"

# Import GPU-accelerated force terms with fallback
try:
    from .gpu_force_terms import (
        GPULennardJonesForceTerm,
        GPUCoulombForceTerm,
        GPUAmberForceField
    )
    GPU_FORCE_TERMS_AVAILABLE = True
    logger.info("GPU force terms loaded successfully")
except ImportError as e:
    logger.warning(f"GPU force terms not available: {e}")
    GPU_FORCE_TERMS_AVAILABLE = False
    
    # Fallback to regular force terms if available
    try:
        from ..forcefield.forcefield import LennardJonesForceTerm, CoulombForceTerm
        GPULennardJonesForceTerm = LennardJonesForceTerm
        GPUCoulombForceTerm = CoulombForceTerm
        
        class GPUAmberForceField:
            @staticmethod
            def create_gpu_system(system, *args, **kwargs):
                return system  # Return original system unchanged
    except ImportError:
        # Create minimal dummy classes
        class DummyForceTerm:
            def __init__(self, *args, **kwargs):
                pass
        
        GPULennardJonesForceTerm = DummyForceTerm
        GPUCoulombForceTerm = DummyForceTerm
        
        class GPUAmberForceField:
            @staticmethod
            def create_gpu_system(system, *args, **kwargs):
                return system

# Import testing and examples with fallback
try:
    from .gpu_testing import run_gpu_validation, GPUPerformanceTester
    from .gpu_examples import main as run_gpu_examples
    GPU_TESTING_AVAILABLE = True
    logger.info("GPU testing modules loaded successfully")
except ImportError as e:
    logger.warning(f"GPU testing not available: {e}")
    GPU_TESTING_AVAILABLE = False
    
    def run_gpu_validation():
        print("GPU testing not available - missing dependencies")
        return {"error": "GPU dependencies not installed"}
    
    def run_gpu_examples():
        print("GPU examples not available - missing dependencies")
    
    class GPUPerformanceTester:
        def __init__(self, *args, **kwargs):
            pass

# Check for actual GPU availability if modules are loaded
if GPU_AVAILABLE and is_gpu_available():
    devices = get_gpu_device_info()
    logger.info(f"Found {len(devices)} GPU device(s)")
elif GPU_AVAILABLE:
    logger.info("GPU modules available but no devices detected")

__all__ = [
    # Core GPU acceleration
    'GPUAccelerator',
    'is_gpu_available', 
    'get_gpu_device_info',
    'LennardJonesGPU',
    'CoulombGPU',
    'GPUBackend',
    
    # GPU-accelerated force terms
    'GPULennardJonesForceTerm',
    'GPUCoulombForceTerm',
    'GPUAmberForceField',
    
    # Testing and validation
    'run_gpu_validation',
    'run_gpu_examples',
    'GPUPerformanceTester',
    
    # Availability flags
    'GPU_AVAILABLE',
    'GPU_FORCE_TERMS_AVAILABLE',
    'GPU_TESTING_AVAILABLE'
]
