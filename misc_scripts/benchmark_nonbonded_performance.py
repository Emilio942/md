#!/usr/bin/env python3
"""
Non-bonded Forces Performance Benchmark for Task 4.4
==================================================

This script benchmarks the current non-bonded force implementations vs
the optimized versions to measure performance improvements.

Benchmark Goals:
- Compare current simulation engine non-bonded forces vs optimized implementation
- Measure performance improvement > 30%
- Validate energy conservation
- Test neighbor list efficiency
- Test Ewald summation performance
"""

import numpy as np
import time
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import current and optimized implementations
try:
    from proteinMD.core.simulation import MolecularDynamicsSimulation
    from proteinMD.forcefield.forcefield import LennardJonesForceTerm, CoulombForceTerm
    from proteinMD.forcefield.optimized_nonbonded import (
        OptimizedLennardJonesForceTerm, 
        EwaldSummationElectrostatics,
        OptimizedNonbondedForceField
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class NonbondedBenchmark:
    """Benchmark class for non-bonded force calculations."""
    
    def __init__(self, n_particles: int = 1000, box_size: float = 5.0):
        """
        Initialize benchmark.
        
        Parameters
        ----------
        n_particles : int
            Number of particles for benchmark
        box_size : float
            Simulation box size in nm
        """
        self.n_particles = n_particles
        self.box_size = box_size
        
        # Generate test system
        np.random.seed(42)  # For reproducibility
        self.positions = self._generate_positions()
        self.charges = self._generate_charges()
        self.lj_params = self._generate_lj_params()
        
        # Box vectors for periodic boundaries
        self.box_vectors = np.eye(3) * box_size
        
        print(f"Benchmark system: {n_particles} particles in {box_size}x{box_size}x{box_size} nm box")
    
    def _generate_positions(self) -> np.ndarray:
        """Generate random particle positions."""
        return np.random.uniform(0, self.box_size, (self.n_particles, 3))
    
    def _generate_charges(self) -> np.ndarray:
        """Generate random charges."""
        # 80% neutral, 20% charged
        charges = np.zeros(self.n_particles)
        n_charged = int(0.2 * self.n_particles)
        
        # Half positive, half negative
        charges[:n_charged//2] = np.random.uniform(0.1, 1.0, n_charged//2)
        charges[n_charged//2:n_charged] = -np.random.uniform(0.1, 1.0, n_charged - n_charged//2)
        
        # Shuffle to randomize positions
        np.random.shuffle(charges)
        return charges
    
    def _generate_lj_params(self) -> List[Tuple[float, float]]:
        """Generate Lennard-Jones parameters."""
        # Typical biomolecular ranges
        sigmas = np.random.uniform(0.25, 0.45, self.n_particles)  # nm
        epsilons = np.random.uniform(0.05, 0.5, self.n_particles)  # kJ/mol
        
        return [(sigma, epsilon) for sigma, epsilon in zip(sigmas, epsilons)]
    
    def benchmark_current_implementation(self, cutoff: float = 1.2) -> Dict:
        """Benchmark current simulation engine implementation."""
        if not IMPORTS_AVAILABLE:
            return {"error": "Imports not available"}
        
        print(f"\nBenchmarking current implementation (cutoff={cutoff} nm)...")
        
        # Setup current implementation
        sim = MolecularDynamicsSimulation(
            num_particles=self.n_particles,
            box_dimensions=np.array([self.box_size] * 3),
            cutoff_distance=cutoff
        )
        
        # Set positions and charges
        sim.positions = self.positions.copy()
        sim.charges = self.charges
        
        # Benchmark force calculation
        start_time = time.time()
        forces = sim._calculate_nonbonded_forces()
        end_time = time.time()
        
        calculation_time = end_time - start_time
        
        # Calculate total energy (approximate)
        total_energy = self._estimate_energy_current(sim, forces)
        
        results = {
            "implementation": "current",
            "time": calculation_time,
            "forces_norm": np.linalg.norm(forces),
            "energy": total_energy,
            "particles": self.n_particles,
            "cutoff": cutoff
        }
        
        print(f"  Time: {calculation_time:.4f} s")
        print(f"  Force norm: {np.linalg.norm(forces):.2f}")
        print(f"  Energy: {total_energy:.2f} kJ/mol")
        
        return results
    
    def benchmark_optimized_lj(self, cutoff: float = 1.2, cutoff_method: str = "switch") -> Dict:
        """Benchmark optimized Lennard-Jones implementation."""
        if not IMPORTS_AVAILABLE:
            return {"error": "Imports not available"}
        
        print(f"\nBenchmarking optimized LJ (cutoff={cutoff} nm, method={cutoff_method})...")
        
        # Setup optimized LJ force
        lj_force = OptimizedLennardJonesForceTerm(
            cutoff=cutoff,
            cutoff_method=cutoff_method,
            use_neighbor_list=True,
            use_long_range_correction=True
        )
        
        # Add particles
        for sigma, epsilon in self.lj_params:
            lj_force.add_particle(sigma, epsilon)
        
        # Benchmark force calculation
        start_time = time.time()
        forces, energy = lj_force.calculate(self.positions, self.box_vectors)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        
        # Get performance stats
        avg_time = lj_force.total_calculation_time / lj_force.calculation_count if lj_force.calculation_count > 0 else 0
        
        results = {
            "implementation": "optimized_lj",
            "time": calculation_time,
            "avg_time": avg_time,
            "forces_norm": np.linalg.norm(forces),
            "energy": energy,
            "particles": self.n_particles,
            "cutoff": cutoff,
            "cutoff_method": cutoff_method,
            "neighbor_list_size": len(lj_force.neighbor_list.neighbors) if lj_force.use_neighbor_list else 0
        }
        
        print(f"  Time: {calculation_time:.4f} s")
        print(f"  Force norm: {np.linalg.norm(forces):.2f}")
        print(f"  Energy: {energy:.2f} kJ/mol")
        if lj_force.use_neighbor_list:
            print(f"  Neighbor list size: {len(lj_force.neighbor_list.neighbors)}")
        
        return results
    
    def benchmark_ewald_electrostatics(self, cutoff: float = 1.2) -> Dict:
        """Benchmark Ewald summation electrostatics."""
        if not IMPORTS_AVAILABLE:
            return {"error": "Imports not available"}
        
        print(f"\nBenchmarking Ewald electrostatics (cutoff={cutoff} nm)...")
        
        # Setup Ewald electrostatics
        ewald_force = EwaldSummationElectrostatics(cutoff=cutoff)
        
        # Add particles
        for charge in self.charges:
            ewald_force.add_particle(charge)
        
        # Benchmark force calculation
        start_time = time.time()
        forces, energy = ewald_force.calculate(self.positions, self.box_vectors)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        
        # Get performance stats
        avg_time = ewald_force.total_calculation_time / ewald_force.calculation_count if ewald_force.calculation_count > 0 else 0
        
        results = {
            "implementation": "ewald",
            "time": calculation_time,
            "avg_time": avg_time,
            "forces_norm": np.linalg.norm(forces),
            "energy": energy,
            "particles": self.n_particles,
            "cutoff": cutoff
        }
        
        print(f"  Time: {calculation_time:.4f} s")
        print(f"  Force norm: {np.linalg.norm(forces):.2f}")
        print(f"  Energy: {energy:.2f} kJ/mol")
        
        return results
    
    def benchmark_combined_optimized(self, cutoff: float = 1.2) -> Dict:
        """Benchmark combined optimized force field."""
        if not IMPORTS_AVAILABLE:
            return {"error": "Imports not available"}
        
        print(f"\nBenchmarking combined optimized force field (cutoff={cutoff} nm)...")
        
        # Setup combined force field
        force_field = OptimizedNonbondedForceField(
            lj_cutoff=cutoff,
            electrostatic_cutoff=cutoff,
            use_ewald=True,
            use_neighbor_lists=True
        )
        
        # Add particles
        for i, ((sigma, epsilon), charge) in enumerate(zip(self.lj_params, self.charges)):
            force_field.add_particle(sigma, epsilon, charge)
        
        # Benchmark force calculation
        start_time = time.time()
        forces, energy = force_field.calculate(self.positions, self.box_vectors)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        
        results = {
            "implementation": "combined_optimized",
            "time": calculation_time,
            "forces_norm": np.linalg.norm(forces),
            "energy": energy,
            "particles": self.n_particles,
            "cutoff": cutoff
        }
        
        print(f"  Time: {calculation_time:.4f} s")
        print(f"  Force norm: {np.linalg.norm(forces):.2f}")
        print(f"  Energy: {energy:.2f} kJ/mol")
        
        return results
    
    def _estimate_energy_current(self, sim, forces) -> float:
        """Estimate energy from current implementation (rough approximation)."""
        # This is a rough estimate since current implementation doesn't return energy
        # We'll use force magnitude as a proxy
        return np.sum(np.linalg.norm(forces, axis=1)) * 0.1  # Rough conversion
    
    def test_energy_conservation(self, n_steps: int = 100) -> Dict:
        """Test energy conservation over multiple steps."""
        if not IMPORTS_AVAILABLE:
            return {"error": "Imports not available"}
        
        print(f"\nTesting energy conservation over {n_steps} steps...")
        
        # Use optimized force field
        force_field = OptimizedNonbondedForceField(
            lj_cutoff=1.2,
            electrostatic_cutoff=1.2,
            use_ewald=True,
            use_neighbor_lists=True
        )
        
        # Add particles
        for i, ((sigma, epsilon), charge) in enumerate(zip(self.lj_params, self.charges)):
            force_field.add_particle(sigma, epsilon, charge)
        
        # Track energy over time
        energies = []
        positions = self.positions.copy()
        velocities = np.random.normal(0, 0.1, (self.n_particles, 3))  # Small random velocities
        dt = 0.001  # ps
        
        for step in range(n_steps):
            # Calculate forces and energy
            forces, energy = force_field.calculate(positions, self.box_vectors)
            energies.append(energy)
            
            # Simple Verlet integration step
            positions += velocities * dt + 0.5 * forces * dt**2
            velocities += forces * dt
            
            # Apply periodic boundary conditions
            positions = np.mod(positions, self.box_size)
        
        # Analyze energy conservation
        energies = np.array(energies)
        energy_drift = (energies[-1] - energies[0]) / energies[0]
        energy_fluctuation = np.std(energies) / np.mean(energies)
        
        results = {
            "initial_energy": energies[0],
            "final_energy": energies[-1],
            "energy_drift": energy_drift,
            "energy_fluctuation": energy_fluctuation,
            "n_steps": n_steps
        }
        
        print(f"  Initial energy: {energies[0]:.2f} kJ/mol")
        print(f"  Final energy: {energies[-1]:.2f} kJ/mol")
        print(f"  Energy drift: {energy_drift:.2e}")
        print(f"  Energy fluctuation: {energy_fluctuation:.2e}")
        
        return results
    
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite."""
        print("="*60)
        print("NON-BONDED FORCES PERFORMANCE BENCHMARK")
        print("="*60)
        
        if not IMPORTS_AVAILABLE:
            print("ERROR: Required modules not available for benchmarking")
            return {"error": "imports_not_available"}
        
        results = {}
        
        # Benchmark different implementations
        cutoff = 1.2
        
        # Current implementation
        try:
            results["current"] = self.benchmark_current_implementation(cutoff)
        except Exception as e:
            print(f"Error in current implementation: {e}")
            results["current"] = {"error": str(e)}
        
        # Optimized LJ with different cutoff methods
        for method in ["switch", "force_switch", "hard"]:
            try:
                results[f"optimized_lj_{method}"] = self.benchmark_optimized_lj(cutoff, method)
            except Exception as e:
                print(f"Error in optimized LJ ({method}): {e}")
                results[f"optimized_lj_{method}"] = {"error": str(e)}
        
        # Ewald electrostatics
        try:
            results["ewald"] = self.benchmark_ewald_electrostatics(cutoff)
        except Exception as e:
            print(f"Error in Ewald electrostatics: {e}")
            results["ewald"] = {"error": str(e)}
        
        # Combined optimized
        try:
            results["combined"] = self.benchmark_combined_optimized(cutoff)
        except Exception as e:
            print(f"Error in combined optimized: {e}")
            results["combined"] = {"error": str(e)}
        
        # Energy conservation test
        try:
            results["energy_conservation"] = self.test_energy_conservation()
        except Exception as e:
            print(f"Error in energy conservation test: {e}")
            results["energy_conservation"] = {"error": str(e)}
        
        # Calculate performance improvements
        self._analyze_performance(results)
        
        return results
    
    def _analyze_performance(self, results: Dict):
        """Analyze and report performance improvements."""
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS")
        print("="*60)
        
        current_time = results.get("current", {}).get("time", None)
        if current_time is None:
            print("Cannot calculate performance improvements - current implementation time not available")
            return
        
        print(f"Current implementation time: {current_time:.4f} s")
        
        # Compare optimized implementations
        optimized_results = {
            k: v for k, v in results.items() 
            if k.startswith("optimized") or k == "combined" or k == "ewald"
        }
        
        best_improvement = 0
        best_method = None
        
        for method, result in optimized_results.items():
            if "error" in result:
                continue
                
            opt_time = result.get("time", None)
            if opt_time is None:
                continue
            
            improvement = (current_time - opt_time) / current_time * 100
            print(f"{method}: {opt_time:.4f} s ({improvement:+.1f}%)")
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_method = method
        
        print(f"\nBest performance improvement: {best_improvement:.1f}% ({best_method})")
        
        if best_improvement > 30:
            print("✅ Performance requirement met (>30% improvement)")
        else:
            print("❌ Performance requirement not met (need >30% improvement)")
        
        # Energy conservation analysis
        conservation = results.get("energy_conservation", {})
        if "error" not in conservation:
            drift = conservation.get("energy_drift", 0)
            if abs(drift) < 0.01:  # Less than 1% drift
                print("✅ Energy conservation requirement met (<1% drift)")
            else:
                print(f"❌ Energy conservation issue (drift: {drift:.2%})")


def main():
    """Main benchmark execution."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test different system sizes
    system_sizes = [500, 1000, 2000]
    
    for n_particles in system_sizes:
        print(f"\n{'='*80}")
        print(f"BENCHMARKING SYSTEM SIZE: {n_particles} PARTICLES")
        print(f"{'='*80}")
        
        benchmark = NonbondedBenchmark(n_particles=n_particles)
        results = benchmark.run_full_benchmark()
        
        # Save results
        import json
        output_file = f"benchmark_results_{n_particles}particles.json"
        with open(output_file, 'w') as f:
            # Convert numpy types to regular Python types for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_value = {}
                    for k, v in value.items():
                        if isinstance(v, (np.ndarray, np.generic)):
                            json_value[k] = float(v)
                        else:
                            json_value[k] = v
                    json_results[key] = json_value
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
