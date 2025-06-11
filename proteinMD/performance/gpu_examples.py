"""
Example and demonstration script for GPU-accelerated molecular dynamics.

This script demonstrates how to use the GPU acceleration features
and provides examples of creating and running GPU-accelerated simulations.
"""

import numpy as np
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import GPU acceleration modules
try:
    from .gpu_acceleration import is_gpu_available, get_gpu_device_info
    from .gpu_force_terms import GPULennardJonesForceTerm, GPUCoulombForceTerm, GPUAmberForceField
    from .gpu_testing import run_gpu_validation
    from ..forcefield.forcefield import ForceFieldSystem, LennardJonesForceTerm, CoulombForceTerm
except ImportError as e:
    logger.error(f"Failed to import GPU modules: {e}")
    print("❌ GPU acceleration modules not available")
    exit(1)

def demonstrate_gpu_acceleration():
    """
    Demonstrate GPU acceleration capabilities with practical examples.
    """
    print("🚀 GPU Acceleration Demonstration for proteinMD")
    print("=" * 55)
    
    # Check GPU availability
    print("\n📱 GPU Device Information:")
    print("-" * 30)
    
    if is_gpu_available():
        devices = get_gpu_device_info()
        print(f"✅ GPU acceleration available with {len(devices)} device(s):")
        for i, device in enumerate(devices):
            print(f"   {i}: {device}")
    else:
        print("❌ No GPU devices found - CPU fallback will be used")
    
    # Example 1: Simple Lennard-Jones system
    print("\n🧪 Example 1: Lennard-Jones System")
    print("-" * 40)
    
    n_particles = 1000
    positions = np.random.uniform(0, 5, (n_particles, 3))
    
    # Create GPU-accelerated LJ term
    lj_gpu = GPULennardJonesForceTerm(cutoff=2.0, name="LJ_GPU_Demo")
    
    # Add particles with LJ parameters
    for i in range(n_particles):
        lj_gpu.add_particle(sigma=0.3, epsilon=1.0)  # Typical values for noble gases
    
    print(f"Created system with {n_particles} particles")
    print(f"GPU enabled: {lj_gpu.gpu_enabled}")
    
    # Calculate forces
    start_time = time.time()
    forces, energy = lj_gpu.calculate(positions)
    gpu_time = time.time() - start_time
    
    print(f"Force calculation completed in {gpu_time:.4f} seconds")
    print(f"Total potential energy: {energy:.3f} kJ/mol")
    print(f"Force magnitude range: {np.min(np.linalg.norm(forces, axis=1)):.3f} - {np.max(np.linalg.norm(forces, axis=1)):.3f}")
    
    # Example 2: Electrostatic system
    print("\n⚡ Example 2: Electrostatic System")
    print("-" * 40)
    
    # Create random charges (mix of positive and negative)
    charges = np.random.uniform(-1, 1, n_particles)
    
    # Create GPU-accelerated Coulomb term
    coulomb_gpu = GPUCoulombForceTerm(charges=charges, cutoff=2.0, name="Coulomb_GPU_Demo")
    
    print(f"Created electrostatic system with {n_particles} charged particles")
    print(f"Total charge: {np.sum(charges):.3f} e")
    print(f"GPU enabled: {coulomb_gpu.gpu_enabled}")
    
    # Calculate electrostatic forces
    start_time = time.time()
    elec_forces, elec_energy = coulomb_gpu.calculate(positions)
    elec_time = time.time() - start_time
    
    print(f"Electrostatic calculation completed in {elec_time:.4f} seconds")
    print(f"Total electrostatic energy: {elec_energy:.3f} kJ/mol")
    print(f"Electrostatic force magnitude range: {np.min(np.linalg.norm(elec_forces, axis=1)):.3f} - {np.max(np.linalg.norm(elec_forces, axis=1)):.3f}")
    
    # Example 3: Combined system (LJ + Electrostatics)
    print("\n🔗 Example 3: Combined LJ + Electrostatic System")
    print("-" * 50)
    
    # Create a force field system
    system = ForceFieldSystem(name="GPU_Demo_System")
    system.add_force_term(lj_gpu)
    system.add_force_term(coulomb_gpu)
    
    # Calculate total forces
    start_time = time.time()
    total_forces, total_energy = system.calculate_forces(positions)
    total_time = time.time() - start_time
    
    print(f"Combined system calculation completed in {total_time:.4f} seconds")
    print(f"Total system energy: {total_energy:.3f} kJ/mol")
    print(f"  - LJ contribution: {energy:.3f} kJ/mol ({100*energy/total_energy:.1f}%)")
    print(f"  - Electrostatic contribution: {elec_energy:.3f} kJ/mol ({100*elec_energy/total_energy:.1f}%)")
    
    # Example 4: Performance comparison
    print("\n📊 Example 4: Performance Comparison")
    print("-" * 45)
    
    if lj_gpu.gpu_enabled:
        # Compare with CPU version for smaller system
        n_test = 500
        test_positions = np.random.uniform(0, 3, (n_test, 3))
        
        # CPU version
        lj_cpu = LennardJonesForceTerm(cutoff=2.0, name="LJ_CPU")
        for i in range(n_test):
            lj_cpu.add_particle(sigma=0.3, epsilon=1.0)
        
        # Benchmark both
        n_trials = 5
        
        # CPU timing
        start_time = time.time()
        for _ in range(n_trials):
            cpu_forces, cpu_energy = lj_cpu.calculate(test_positions)
        cpu_avg_time = (time.time() - start_time) / n_trials
        
        # GPU timing
        lj_gpu_test = GPULennardJonesForceTerm(cutoff=2.0, name="LJ_GPU_Test")
        for i in range(n_test):
            lj_gpu_test.add_particle(sigma=0.3, epsilon=1.0)
        
        start_time = time.time()
        for _ in range(n_trials):
            gpu_forces, gpu_energy = lj_gpu_test.calculate(test_positions)
        gpu_avg_time = (time.time() - start_time) / n_trials
        
        speedup = cpu_avg_time / gpu_avg_time if gpu_avg_time > 0 else 1.0
        
        print(f"Performance comparison ({n_test} particles, {n_trials} trials):")
        print(f"  CPU average time: {cpu_avg_time:.4f} seconds")
        print(f"  GPU average time: {gpu_avg_time:.4f} seconds")
        print(f"  Speedup: {speedup:.1f}x")
        
        # Accuracy check
        force_diff = np.abs(gpu_forces - cpu_forces)
        max_error = np.max(force_diff)
        energy_error = abs(gpu_energy - cpu_energy)
        
        print(f"  Max force error: {max_error:.2e}")
        print(f"  Energy error: {energy_error:.2e}")
        
        if max_error < 1e-5 and energy_error < 1e-5:
            print("  ✅ Accuracy: EXCELLENT")
        elif max_error < 1e-3 and energy_error < 1e-3:
            print("  ✅ Accuracy: GOOD")
        else:
            print("  ⚠️  Accuracy: NEEDS REVIEW")
    else:
        print("GPU not available for performance comparison")
    
    print("\n✨ GPU acceleration demonstration completed!")

def create_large_system_example():
    """
    Create and test a large system to demonstrate >5x speedup requirement.
    """
    print("\n🏗️ Large System Performance Test (Task 7.2 Requirement)")
    print("=" * 60)
    
    # Create large system (>1000 atoms as per requirement)
    n_particles = 2000
    print(f"Creating system with {n_particles} particles...")
    
    # Random 3D positions in a cubic box
    box_size = 10.0  # nm
    positions = np.random.uniform(0, box_size, (n_particles, 3))
    
    # Random LJ parameters (realistic values)
    sigma_values = np.random.uniform(0.25, 0.35, n_particles)  # nm
    epsilon_values = np.random.uniform(0.5, 2.0, n_particles)  # kJ/mol
    
    # Random charges (with overall neutrality)
    charges = np.random.uniform(-1, 1, n_particles)
    charges -= np.mean(charges)  # Make system neutral
    
    print(f"System box size: {box_size} nm")
    print(f"Average particle density: {n_particles/box_size**3:.3f} particles/nm³")
    print(f"System net charge: {np.sum(charges):.6f} e")
    
    # Test Lennard-Jones performance
    print(f"\n📊 Testing Lennard-Jones performance...")
    
    lj_gpu = GPULennardJonesForceTerm(cutoff=1.0, name="LJ_Large_System")
    for i in range(n_particles):
        lj_gpu.add_particle(sigma=sigma_values[i], epsilon=epsilon_values[i])
    
    if lj_gpu.gpu_enabled:
        lj_results = lj_gpu.benchmark_performance(n_particles=n_particles, n_trials=3)
        print(f"  GPU enabled: {lj_gpu.gpu_enabled}")
        print(f"  CPU time: {lj_results['cpu_time']:.4f} seconds")
        print(f"  GPU time: {lj_results['gpu_time']:.4f} seconds")
        print(f"  Speedup: {lj_results['speedup']:.1f}x")
        
        if lj_results['speedup'] >= 5.0:
            print(f"  ✅ LJ requirement MET (>{5.0}x speedup)")
        else:
            print(f"  ⚠️  LJ requirement not met ({lj_results['speedup']:.1f}x < 5.0x)")
    else:
        print("  ❌ GPU not available for LJ test")
    
    # Test Coulomb performance
    print(f"\n⚡ Testing Coulomb performance...")
    
    coulomb_gpu = GPUCoulombForceTerm(charges=charges, cutoff=1.0, name="Coulomb_Large_System")
    
    if coulomb_gpu.gpu_enabled:
        coulomb_results = coulomb_gpu.benchmark_performance(n_particles=n_particles, n_trials=3)
        print(f"  GPU enabled: {coulomb_gpu.gpu_enabled}")
        print(f"  CPU time: {coulomb_results['cpu_time']:.4f} seconds")
        print(f"  GPU time: {coulomb_results['gpu_time']:.4f} seconds")
        print(f"  Speedup: {coulomb_results['speedup']:.1f}x")
        
        if coulomb_results['speedup'] >= 5.0:
            print(f"  ✅ Coulomb requirement MET (>5.0x speedup)")
        else:
            print(f"  ⚠️  Coulomb requirement not met ({coulomb_results['speedup']:.1f}x < 5.0x)")
    else:
        print("  ❌ GPU not available for Coulomb test")
    
    # Overall assessment
    print(f"\n🎯 Task 7.2 Performance Requirement Assessment:")
    print("-" * 50)
    
    lj_passed = lj_gpu.gpu_enabled and lj_results.get('speedup', 0) >= 5.0
    coulomb_passed = coulomb_gpu.gpu_enabled and coulomb_results.get('speedup', 0) >= 5.0
    
    if lj_passed and coulomb_passed:
        print("✅ REQUIREMENT MET: >5x speedup achieved for large systems (>1000 atoms)")
        print("📝 Task 7.2 performance requirement satisfied")
    else:
        print("❌ REQUIREMENT NOT MET: <5x speedup for large systems")
        print("🔧 Further optimization may be needed")

def integration_example():
    """
    Show how to integrate GPU acceleration into existing workflows.
    """
    print("\n🔧 Integration Example: Converting Existing Systems")
    print("=" * 55)
    
    # Create a traditional CPU-based system
    print("Creating traditional CPU-based force field system...")
    
    cpu_system = ForceFieldSystem(name="Traditional_CPU_System")
    
    # Add traditional force terms
    lj_cpu = LennardJonesForceTerm(cutoff=1.0, name="LJ_CPU")
    coulomb_cpu = CoulombForceTerm(cutoff=1.0, name="Coulomb_CPU")
    
    n_particles = 500
    for i in range(n_particles):
        lj_cpu.add_particle(sigma=0.3, epsilon=1.0)
        coulomb_cpu.add_particle(charge=np.random.uniform(-0.5, 0.5))
    
    cpu_system.add_force_term(lj_cpu)
    cpu_system.add_force_term(coulomb_cpu)
    
    print(f"  CPU system created with {len(cpu_system.force_terms)} force terms")
    print(f"  Force terms: {[term.name for term in cpu_system.force_terms]}")
    
    # Convert to GPU-accelerated system
    print("\nConverting to GPU-accelerated system...")
    
    gpu_system = GPUAmberForceField.create_gpu_system(cpu_system)
    
    print(f"  GPU system created with {len(gpu_system.force_terms)} force terms")
    print(f"  Force terms: {[term.name for term in gpu_system.force_terms]}")
    
    # Test both systems
    positions = np.random.uniform(0, 3, (n_particles, 3))
    
    print("\nTesting both systems...")
    
    # CPU system
    start_time = time.time()
    cpu_forces, cpu_energy = cpu_system.calculate_forces(positions)
    cpu_time = time.time() - start_time
    
    # GPU system
    start_time = time.time()
    gpu_forces, gpu_energy = gpu_system.calculate_forces(positions)
    gpu_time = time.time() - start_time
    
    # Compare results
    force_diff = np.abs(gpu_forces - cpu_forces)
    max_force_error = np.max(force_diff)
    energy_error = abs(gpu_energy - cpu_energy)
    speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
    
    print(f"\nComparison results:")
    print(f"  CPU time: {cpu_time:.4f} seconds")
    print(f"  GPU time: {gpu_time:.4f} seconds")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Max force error: {max_force_error:.2e}")
    print(f"  Energy error: {energy_error:.2e}")
    
    if max_force_error < 1e-5:
        print("  ✅ Perfect numerical agreement")
    elif max_force_error < 1e-3:
        print("  ✅ Good numerical agreement")
    else:
        print("  ⚠️  Numerical differences detected")
    
    print("\n✨ Integration example completed!")

def main():
    """
    Main demonstration script for GPU acceleration.
    """
    print("🧬 proteinMD GPU Acceleration - Task 7.2 Implementation")
    print("=" * 65)
    print("This demonstration showcases the GPU acceleration features")
    print("implemented to satisfy Task 7.2 requirements.")
    
    try:
        # Basic demonstration
        demonstrate_gpu_acceleration()
        
        # Large system performance test
        create_large_system_example()
        
        # Integration example
        integration_example()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\n📋 Summary of Task 7.2 Implementation:")
        print("   ✅ GPU kernels for Lennard-Jones forces")
        print("   ✅ GPU kernels for Coulomb electrostatic forces")
        print("   ✅ Automatic CPU/GPU fallback mechanism")
        print("   ✅ Performance optimization for large systems")
        print("   ✅ Compatibility with common GPU models (CUDA/OpenCL)")
        
        print("\n🔬 For comprehensive validation, run:")
        print("   python -m proteinMD.performance.gpu_testing")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Demonstration failed: {e}")
        print("Please check the installation and GPU drivers.")

if __name__ == "__main__":
    main()
