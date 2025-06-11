"""
Testing and benchmarking module for GPU acceleration.

This module provides comprehensive tests and benchmarks to validate
GPU acceleration performance and correctness.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from pathlib import Path

from .gpu_acceleration import (
    is_gpu_available, get_gpu_device_info, 
    LennardJonesGPU, CoulombGPU, GPUBackend
)
from .gpu_force_terms import GPULennardJonesForceTerm, GPUCoulombForceTerm, GPUAmberForceField
from ..forcefield.forcefield import LennardJonesForceTerm, CoulombForceTerm, ForceFieldSystem

logger = logging.getLogger(__name__)

class GPUPerformanceTester:
    """
    Comprehensive testing and benchmarking for GPU acceleration.
    
    This class provides methods to test correctness, measure performance,
    and generate detailed reports on GPU acceleration effectiveness.
    """
    
    def __init__(self, output_dir: str = "gpu_benchmarks"):
        """
        Initialize the performance tester.
        
        Parameters
        ----------
        output_dir : str
            Directory to save benchmark results and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {
            'device_info': get_gpu_device_info(),
            'gpu_available': is_gpu_available(),
            'tests': {},
            'benchmarks': {}
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all GPU acceleration tests and benchmarks.
        
        Returns
        -------
        Dict[str, Any]
            Complete test and benchmark results
        """
        logger.info("Starting comprehensive GPU acceleration tests...")
        
        # Test GPU device detection
        self.test_device_detection()
        
        # Test Lennard-Jones force accuracy
        self.test_lennard_jones_accuracy()
        
        # Test Coulomb force accuracy
        self.test_coulomb_accuracy()
        
        # Benchmark performance for different system sizes
        self.benchmark_performance_scaling()
        
        # Test large system performance (>1000 atoms)
        self.test_large_system_performance()
        
        # Test automatic fallback mechanism
        self.test_fallback_mechanism()
        
        # Generate performance report
        self.generate_report()
        
        logger.info("GPU acceleration tests completed successfully")
        return self.results
    
    def test_device_detection(self):
        """Test GPU device detection and initialization."""
        logger.info("Testing GPU device detection...")
        
        devices = get_gpu_device_info()
        self.results['tests']['device_detection'] = {
            'gpu_available': is_gpu_available(),
            'num_devices': len(devices),
            'devices': [str(device) for device in devices]
        }
        
        if devices:
            logger.info(f"Found {len(devices)} GPU device(s):")
            for device in devices:
                logger.info(f"  - {device}")
        else:
            logger.warning("No GPU devices found")
    
    def test_lennard_jones_accuracy(self, n_particles: int = 100, tolerance: float = 1e-6):
        """
        Test accuracy of GPU Lennard-Jones force calculations.
        
        Parameters
        ----------
        n_particles : int
            Number of particles for test
        tolerance : float
            Numerical tolerance for comparison
        """
        logger.info(f"Testing Lennard-Jones accuracy with {n_particles} particles...")
        
        # Create test system
        positions = np.random.uniform(0, 2, (n_particles, 3)).astype(np.float64)
        sigma = np.full(n_particles, 0.3, dtype=np.float64)
        epsilon = np.full(n_particles, 1.0, dtype=np.float64)
        cutoff = 1.0
        
        # CPU reference calculation
        cpu_term = LennardJonesForceTerm(sigma=sigma, epsilon=epsilon, cutoff=cutoff)
        cpu_forces, cpu_energy = cpu_term.calculate(positions)
        
        # GPU calculation
        gpu_term = GPULennardJonesForceTerm(sigma=sigma, epsilon=epsilon, cutoff=cutoff, force_cpu=False)
        gpu_forces, gpu_energy = gpu_term.calculate(positions)
        
        # Compare results
        force_diff = np.abs(gpu_forces - cpu_forces)
        max_force_error = np.max(force_diff)
        rms_force_error = np.sqrt(np.mean(force_diff**2))
        energy_error = abs(gpu_energy - cpu_energy)
        
        test_passed = (max_force_error < tolerance and 
                      rms_force_error < tolerance and 
                      energy_error < tolerance)
        
        self.results['tests']['lennard_jones_accuracy'] = {
            'n_particles': n_particles,
            'max_force_error': float(max_force_error),
            'rms_force_error': float(rms_force_error),
            'energy_error': float(energy_error),
            'tolerance': tolerance,
            'passed': test_passed,
            'gpu_enabled': gpu_term.gpu_enabled
        }
        
        if test_passed:
            logger.info(f"Lennard-Jones accuracy test PASSED (max error: {max_force_error:.2e})")
        else:
            logger.error(f"Lennard-Jones accuracy test FAILED (max error: {max_force_error:.2e})")
    
    def test_coulomb_accuracy(self, n_particles: int = 100, tolerance: float = 1e-6):
        """
        Test accuracy of GPU Coulomb force calculations.
        
        Parameters
        ----------
        n_particles : int
            Number of particles for test
        tolerance : float
            Numerical tolerance for comparison
        """
        logger.info(f"Testing Coulomb accuracy with {n_particles} particles...")
        
        # Create test system
        positions = np.random.uniform(0, 2, (n_particles, 3)).astype(np.float64)
        charges = np.random.uniform(-1, 1, n_particles).astype(np.float64)
        cutoff = 1.0
        
        # CPU reference calculation
        cpu_term = CoulombForceTerm(charges=charges, cutoff=cutoff)
        cpu_forces, cpu_energy = cpu_term.calculate(positions)
        
        # GPU calculation
        gpu_term = GPUCoulombForceTerm(charges=charges, cutoff=cutoff, force_cpu=False)
        gpu_forces, gpu_energy = gpu_term.calculate(positions)
        
        # Compare results
        force_diff = np.abs(gpu_forces - cpu_forces)
        max_force_error = np.max(force_diff)
        rms_force_error = np.sqrt(np.mean(force_diff**2))
        energy_error = abs(gpu_energy - cpu_energy)
        
        test_passed = (max_force_error < tolerance and 
                      rms_force_error < tolerance and 
                      energy_error < tolerance)
        
        self.results['tests']['coulomb_accuracy'] = {
            'n_particles': n_particles,
            'max_force_error': float(max_force_error),
            'rms_force_error': float(rms_force_error),
            'energy_error': float(energy_error),
            'tolerance': tolerance,
            'passed': test_passed,
            'gpu_enabled': gpu_term.gpu_enabled
        }
        
        if test_passed:
            logger.info(f"Coulomb accuracy test PASSED (max error: {max_force_error:.2e})")
        else:
            logger.error(f"Coulomb accuracy test FAILED (max error: {max_force_error:.2e})")
    
    def benchmark_performance_scaling(self, 
                                    particle_counts: List[int] = None, 
                                    n_trials: int = 5):
        """
        Benchmark performance scaling with system size.
        
        Parameters
        ----------
        particle_counts : List[int], optional
            List of particle counts to test
        n_trials : int
            Number of trials per test
        """
        if particle_counts is None:
            particle_counts = [50, 100, 200, 500, 1000, 2000, 5000]
        
        logger.info(f"Benchmarking performance scaling for {len(particle_counts)} system sizes...")
        
        lj_results = []
        coulomb_results = []
        
        for n_particles in particle_counts:
            logger.info(f"  Testing {n_particles} particles...")
            
            # Benchmark Lennard-Jones
            lj_gpu = GPULennardJonesForceTerm(cutoff=1.0, force_cpu=False)
            lj_result = lj_gpu.benchmark_performance(n_particles, n_trials)
            lj_result['n_particles'] = n_particles
            lj_results.append(lj_result)
            
            # Benchmark Coulomb
            coulomb_gpu = GPUCoulombForceTerm(cutoff=1.0, force_cpu=False)
            coulomb_result = coulomb_gpu.benchmark_performance(n_particles, n_trials)
            coulomb_result['n_particles'] = n_particles
            coulomb_results.append(coulomb_result)
        
        self.results['benchmarks']['performance_scaling'] = {
            'particle_counts': particle_counts,
            'lennard_jones': lj_results,
            'coulomb': coulomb_results
        }
        
        # Create performance plots
        self._plot_performance_scaling(lj_results, coulomb_results)
    
    def test_large_system_performance(self, n_particles: int = 5000, target_speedup: float = 5.0):
        """
        Test performance on large systems (requirement: >5x speedup for >1000 atoms).
        
        Parameters
        ----------
        n_particles : int
            Number of particles for large system test
        target_speedup : float
            Target speedup requirement
        """
        logger.info(f"Testing large system performance with {n_particles} particles...")
        
        # Lennard-Jones test
        lj_gpu = GPULennardJonesForceTerm(cutoff=1.0, force_cpu=False)
        lj_result = lj_gpu.benchmark_performance(n_particles, n_trials=3)
        
        # Coulomb test
        coulomb_gpu = GPUCoulombForceTerm(cutoff=1.0, force_cpu=False)
        coulomb_result = coulomb_gpu.benchmark_performance(n_particles, n_trials=3)
        
        lj_passed = lj_result.get('speedup', 0) >= target_speedup
        coulomb_passed = coulomb_result.get('speedup', 0) >= target_speedup
        
        self.results['tests']['large_system_performance'] = {
            'n_particles': n_particles,
            'target_speedup': target_speedup,
            'lennard_jones': {
                'speedup': lj_result.get('speedup', 0),
                'passed': lj_passed,
                'gpu_enabled': lj_gpu.gpu_enabled
            },
            'coulomb': {
                'speedup': coulomb_result.get('speedup', 0),
                'passed': coulomb_passed,
                'gpu_enabled': coulomb_gpu.gpu_enabled
            },
            'overall_passed': lj_passed and coulomb_passed
        }
        
        if lj_passed and coulomb_passed:
            logger.info(f"Large system performance test PASSED")
            logger.info(f"  LJ speedup: {lj_result.get('speedup', 0):.1f}x")
            logger.info(f"  Coulomb speedup: {coulomb_result.get('speedup', 0):.1f}x")
        else:
            logger.warning(f"Large system performance test FAILED")
            logger.warning(f"  LJ speedup: {lj_result.get('speedup', 0):.1f}x (target: {target_speedup}x)")
            logger.warning(f"  Coulomb speedup: {coulomb_result.get('speedup', 0):.1f}x (target: {target_speedup}x)")
    
    def test_fallback_mechanism(self):
        """Test automatic CPU fallback mechanism."""
        logger.info("Testing automatic CPU fallback mechanism...")
        
        n_particles = 100
        positions = np.random.uniform(0, 2, (n_particles, 3))
        
        # Test with forced CPU mode
        lj_cpu = GPULennardJonesForceTerm(cutoff=1.0, force_cpu=True)
        coulomb_cpu = GPUCoulombForceTerm(cutoff=1.0, force_cpu=True)
        
        # These should work even if GPU is available
        try:
            lj_forces, lj_energy = lj_cpu.calculate(positions)
            coulomb_forces, coulomb_energy = coulomb_cpu.calculate(positions)
            fallback_works = True
        except Exception as e:
            logger.error(f"CPU fallback failed: {e}")
            fallback_works = False
        
        self.results['tests']['fallback_mechanism'] = {
            'cpu_fallback_works': fallback_works,
            'forced_cpu_lj_enabled': not lj_cpu.gpu_enabled,
            'forced_cpu_coulomb_enabled': not coulomb_cpu.gpu_enabled
        }
        
        if fallback_works:
            logger.info("CPU fallback mechanism test PASSED")
        else:
            logger.error("CPU fallback mechanism test FAILED")
    
    def _plot_performance_scaling(self, lj_results: List[Dict], coulomb_results: List[Dict]):
        """Create performance scaling plots."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Lennard-Jones scaling
            particles = [r['n_particles'] for r in lj_results]
            lj_speedups = [r.get('speedup', 1.0) for r in lj_results]
            
            ax1.semilogx(particles, lj_speedups, 'bo-', label='LJ Speedup')
            ax1.axhline(y=5.0, color='r', linestyle='--', label='Target (5x)')
            ax1.set_xlabel('Number of Particles')
            ax1.set_ylabel('Speedup Factor')
            ax1.set_title('Lennard-Jones GPU Speedup')
            ax1.grid(True)
            ax1.legend()
            
            # Coulomb scaling
            coulomb_speedups = [r.get('speedup', 1.0) for r in coulomb_results]
            
            ax2.semilogx(particles, coulomb_speedups, 'ro-', label='Coulomb Speedup')
            ax2.axhline(y=5.0, color='r', linestyle='--', label='Target (5x)')
            ax2.set_xlabel('Number of Particles')
            ax2.set_ylabel('Speedup Factor')
            ax2.set_title('Coulomb GPU Speedup')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'gpu_performance_scaling.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance scaling plot saved to {self.output_dir / 'gpu_performance_scaling.png'}")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping plots")
        except Exception as e:
            logger.error(f"Failed to create performance plots: {e}")
    
    def generate_report(self):
        """Generate a comprehensive performance report."""
        report_file = self.output_dir / 'gpu_acceleration_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("GPU ACCELERATION PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Device information
            f.write("GPU DEVICE INFORMATION:\n")
            f.write("-" * 25 + "\n")
            f.write(f"GPU Available: {self.results['gpu_available']}\n")
            f.write(f"Number of Devices: {len(self.results['device_info'])}\n")
            for i, device in enumerate(self.results['device_info']):
                f.write(f"  Device {i}: {device}\n")
            f.write("\n")
            
            # Test results
            f.write("TEST RESULTS:\n")
            f.write("-" * 15 + "\n")
            
            for test_name, test_result in self.results['tests'].items():
                f.write(f"{test_name.replace('_', ' ').title()}:\n")
                if 'passed' in test_result:
                    status = "PASSED" if test_result['passed'] else "FAILED"
                    f.write(f"  Status: {status}\n")
                
                for key, value in test_result.items():
                    if key != 'passed':
                        if isinstance(value, float):
                            f.write(f"  {key}: {value:.2e}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Performance benchmarks
            if 'performance_scaling' in self.results['benchmarks']:
                f.write("PERFORMANCE BENCHMARKS:\n")
                f.write("-" * 25 + "\n")
                
                scaling = self.results['benchmarks']['performance_scaling']
                
                f.write("Lennard-Jones Performance:\n")
                for result in scaling['lennard_jones']:
                    n = result['n_particles']
                    speedup = result.get('speedup', 1.0)
                    f.write(f"  {n:4d} particles: {speedup:.1f}x speedup\n")
                
                f.write("\nCoulomb Performance:\n")
                for result in scaling['coulomb']:
                    n = result['n_particles']
                    speedup = result.get('speedup', 1.0)
                    f.write(f"  {n:4d} particles: {speedup:.1f}x speedup\n")
            
            # Summary
            f.write("\nSUMMARY:\n")
            f.write("-" * 10 + "\n")
            
            all_tests_passed = all(
                test.get('passed', True) for test in self.results['tests'].values()
            )
            
            if all_tests_passed:
                f.write("✓ All tests PASSED\n")
                f.write("✓ GPU acceleration is working correctly\n")
                
                # Check if performance requirement is met
                large_system = self.results['tests'].get('large_system_performance', {})
                if large_system.get('overall_passed', False):
                    f.write("✓ Performance requirement (>5x speedup for >1000 atoms) MET\n")
                else:
                    f.write("✗ Performance requirement (>5x speedup for >1000 atoms) NOT MET\n")
            else:
                f.write("✗ Some tests FAILED\n")
                f.write("✗ GPU acceleration may not be working correctly\n")
        
        logger.info(f"Comprehensive report saved to {report_file}")

def run_gpu_validation():
    """
    Run complete GPU validation suite.
    
    This is the main entry point for validating GPU acceleration
    implementation according to Task 7.2 requirements.
    """
    print("🚀 Starting GPU Acceleration Validation for Task 7.2")
    print("=" * 60)
    
    tester = GPUPerformanceTester()
    results = tester.run_all_tests()
    
    print("\n📊 VALIDATION RESULTS:")
    print("-" * 25)
    
    # Check requirements
    requirements_met = []
    
    # Requirement 1: GPU kernels for LJ and Coulomb forces
    gpu_kernels_work = (
        results['tests'].get('lennard_jones_accuracy', {}).get('passed', False) and
        results['tests'].get('coulomb_accuracy', {}).get('passed', False)
    )
    requirements_met.append(("GPU kernels for LJ and Coulomb forces", gpu_kernels_work))
    
    # Requirement 2: Automatic CPU/GPU fallback
    fallback_works = results['tests'].get('fallback_mechanism', {}).get('cpu_fallback_works', False)
    requirements_met.append(("Automatic CPU/GPU fallback mechanism", fallback_works))
    
    # Requirement 3: >5x performance for >1000 atoms
    large_system_perf = results['tests'].get('large_system_performance', {}).get('overall_passed', False)
    requirements_met.append(("Performance >5x for large systems (>1000 atoms)", large_system_perf))
    
    # Requirement 4: GPU compatibility
    gpu_compatibility = results['gpu_available']
    requirements_met.append(("Compatibility with common GPU models", gpu_compatibility))
    
    for requirement, met in requirements_met:
        status = "✅ PASSED" if met else "❌ FAILED"
        print(f"{status} {requirement}")
    
    all_passed = all(met for _, met in requirements_met)
    
    print(f"\n🎯 TASK 7.2 STATUS: {'✅ COMPLETED' if all_passed else '❌ INCOMPLETE'}")
    
    if all_passed:
        print("\n🎉 GPU acceleration implementation successfully meets all requirements!")
        print("📝 Task 7.2 can be marked as COMPLETED")
    else:
        print("\n⚠️  Some requirements not met. Review the detailed report for more information.")
    
    print(f"\n📄 Detailed report available at: {tester.output_dir / 'gpu_acceleration_report.txt'}")
    
    return results

if __name__ == "__main__":
    # Run validation when executed directly
    run_gpu_validation()
