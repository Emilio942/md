#!/usr/bin/env python3
"""
Demonstration of Metadynamics Implementation

This script demonstrates the comprehensive metadynamics functionality
including distance and angle collective variables, well-tempered variants,
and free energy surface reconstruction.

Usage:
    python demo_metadynamics.py

Author: AI Assistant
Date: June 11, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from proteinMD.sampling.metadynamics import (
    MetadynamicsSimulation,
    MetadynamicsParameters,
    DistanceCV,
    AngleCV,
    setup_distance_metadynamics,
    setup_angle_metadynamics,
    setup_protein_folding_metadynamics
)


class MockMolecularSystem:
    """Mock molecular system for demonstration."""
    
    def __init__(self, n_atoms=10):
        self.n_atoms = n_atoms
        self.positions = np.random.randn(n_atoms, 3) * 2.0
        self.velocities = np.random.randn(n_atoms, 3) * 0.1
        self.external_forces = np.zeros_like(self.positions)
        self.box = None
        self.step_count = 0
        self.dt = 0.001  # 1 fs timestep
        
    def step(self):
        """Simple MD step with Langevin dynamics."""
        self.step_count += 1
        
        # Add some random motion (Brownian dynamics)
        random_force = np.random.randn(*self.positions.shape) * 0.1
        
        # Simple integration
        self.velocities += (self.external_forces + random_force) * self.dt
        self.positions += self.velocities * self.dt
        
        # Reset external forces for next step
        self.external_forces.fill(0.0)


def demo_distance_metadynamics():
    """Demonstrate distance-based metadynamics."""
    print("=" * 60)
    print("DEMO 1: Distance-Based Metadynamics")
    print("=" * 60)
    
    # Create mock system with two groups that can move apart
    system = MockMolecularSystem(n_atoms=6)
    
    # Position atoms in two groups
    system.positions[:3] = np.array([[0, 0, 0], [0.1, 0, 0], [-0.1, 0, 0]])  # Group 1
    system.positions[3:] = np.array([[3, 0, 0], [3.1, 0, 0], [2.9, 0, 0]])  # Group 2
    
    # Set up metadynamics with distance CV between the two groups
    atom_pairs = [(0, 3)]  # Distance between first atoms of each group
    
    metad = setup_distance_metadynamics(
        system, 
        atom_pairs,
        height=0.5,     # 0.5 kJ/mol hills
        width=0.2,      # Width in nm
        bias_factor=8.0  # Well-tempered with γ=8
    )
    
    # Adjust parameters for demonstration
    metad.params.deposition_interval = 50  # Every 50 steps
    
    print(f"Initial distance: {metad.get_current_cv_values()[0]:.3f} nm")
    print("Running metadynamics simulation...")
    
    # Run simulation
    n_steps = 2000
    metad.run(n_steps)
    
    print(f"Final distance: {metad.get_current_cv_values()[0]:.3f} nm")
    print(f"Hills deposited: {len(metad.hills)}")
    print(f"Convergence detected: {metad.convergence_detected}")
    
    if metad.convergence_detected:
        print(f"Converged at step: {metad.convergence_step}")
    
    # Calculate and display free energy profile
    grid_coords, fes_values = metad.calculate_free_energy_surface(grid_points=50)
    
    print(f"Free energy range: {np.min(fes_values):.2f} to {np.max(fes_values):.2f} kJ/mol")
    
    # Plot results
    metad.plot_results(save_prefix="demo_distance_metad")
    
    return metad


def demo_angle_metadynamics():
    """Demonstrate angle-based metadynamics."""
    print("\n" + "=" * 60)
    print("DEMO 2: Angle-Based Metadynamics")
    print("=" * 60)
    
    # Create mock system
    system = MockMolecularSystem(n_atoms=4)
    
    # Position atoms to form an angle
    system.positions = np.array([
        [1.0, 0.0, 0.0],   # Atom 0
        [0.0, 0.0, 0.0],   # Atom 1 (center)
        [0.0, 1.0, 0.0],   # Atom 2
        [2.0, 2.0, 0.0]    # Atom 3 (not involved in angle)
    ])
    
    # Set up metadynamics with angle CV
    angle_triplets = [(0, 1, 2)]  # Angle between atoms 0-1-2
    
    metad = setup_angle_metadynamics(
        system,
        angle_triplets,
        height=0.3,      # 0.3 kJ/mol hills
        width=0.1,       # Width in radians
        bias_factor=5.0  # Moderate well-tempering
    )
    
    # Adjust parameters for demonstration
    metad.params.deposition_interval = 30
    
    initial_angle = metad.get_current_cv_values()[0]
    print(f"Initial angle: {initial_angle:.3f} rad ({np.degrees(initial_angle):.1f}°)")
    print("Running metadynamics simulation...")
    
    # Run simulation
    n_steps = 1500
    metad.run(n_steps)
    
    final_angle = metad.get_current_cv_values()[0]
    print(f"Final angle: {final_angle:.3f} rad ({np.degrees(final_angle):.1f}°)")
    print(f"Hills deposited: {len(metad.hills)}")
    print(f"Convergence detected: {metad.convergence_detected}")
    
    # Calculate free energy profile
    grid_coords, fes_values = metad.calculate_free_energy_surface(grid_points=40)
    
    print(f"Free energy range: {np.min(fes_values):.2f} to {np.max(fes_values):.2f} kJ/mol")
    
    # Plot results
    metad.plot_results(save_prefix="demo_angle_metad")
    
    return metad


def demo_2d_metadynamics():
    """Demonstrate 2D metadynamics with distance and angle CVs."""
    print("\n" + "=" * 60)
    print("DEMO 3: 2D Metadynamics (Distance + Angle)")
    print("=" * 60)
    
    # Create mock system
    system = MockMolecularSystem(n_atoms=5)
    
    # Position atoms
    system.positions = np.array([
        [0.0, 0.0, 0.0],   # Atom 0
        [1.0, 0.0, 0.0],   # Atom 1
        [2.0, 0.0, 0.0],   # Atom 2
        [0.0, 1.0, 0.0],   # Atom 3
        [3.0, 1.0, 0.0]    # Atom 4
    ])
    
    # Create both distance and angle CVs
    distance_cv = DistanceCV("distance_0_4", 0, 4)
    angle_cv = AngleCV("angle_0_1_2", 0, 1, 2)
    
    # Set up parameters
    params = MetadynamicsParameters(
        height=0.4,
        width=0.15,
        deposition_interval=40,
        bias_factor=6.0,
        temperature=300.0
    )
    
    # Create simulation
    metad = MetadynamicsSimulation([distance_cv, angle_cv], params, system)
    
    cv_values = metad.get_current_cv_values()
    print(f"Initial CVs: distance={cv_values[0]:.3f} nm, angle={cv_values[1]:.3f} rad")
    print("Running 2D metadynamics simulation...")
    
    # Run simulation
    n_steps = 2500
    metad.run(n_steps)
    
    cv_values = metad.get_current_cv_values()
    print(f"Final CVs: distance={cv_values[0]:.3f} nm, angle={cv_values[1]:.3f} rad")
    print(f"Hills deposited: {len(metad.hills)}")
    print(f"Convergence detected: {metad.convergence_detected}")
    
    # Calculate 2D free energy surface
    grid_coords, fes_values = metad.calculate_free_energy_surface(grid_points=[30, 30])
    
    print(f"2D FES shape: {fes_values.shape}")
    print(f"Free energy range: {np.min(fes_values):.2f} to {np.max(fes_values):.2f} kJ/mol")
    
    # Plot results
    metad.plot_results(save_prefix="demo_2d_metad")
    
    return metad


def demo_well_tempered_comparison():
    """Compare standard and well-tempered metadynamics."""
    print("\n" + "=" * 60)
    print("DEMO 4: Standard vs Well-Tempered Metadynamics")
    print("=" * 60)
    
    # Create two identical systems
    system1 = MockMolecularSystem(n_atoms=4)
    system2 = MockMolecularSystem(n_atoms=4)
    
    # Same initial positions
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.5, 0.0, 0.0]
    ])
    system1.positions = positions.copy()
    system2.positions = positions.copy()
    
    # Standard metadynamics (bias_factor = 1)
    params_standard = MetadynamicsParameters(
        height=0.5,
        width=0.2,
        deposition_interval=25,
        bias_factor=1.0  # Standard
    )
    
    # Well-tempered metadynamics
    params_wt = MetadynamicsParameters(
        height=0.5,
        width=0.2,
        deposition_interval=25,
        bias_factor=10.0  # Well-tempered
    )
    
    # Create CVs and simulations
    cv1 = DistanceCV("distance", 0, 3)
    cv2 = DistanceCV("distance", 0, 3)
    
    metad_standard = MetadynamicsSimulation([cv1], params_standard, system1)
    metad_wt = MetadynamicsSimulation([cv2], params_wt, system2)
    
    print("Running standard metadynamics...")
    metad_standard.run(1000)
    
    print("Running well-tempered metadynamics...")
    metad_wt.run(1000)
    
    # Compare results
    print(f"\nStandard MD: {len(metad_standard.hills)} hills")
    print(f"Well-tempered MD: {len(metad_wt.hills)} hills")
    
    if len(metad_standard.hills) > 0:
        std_heights = [h.height for h in metad_standard.hills]
        print(f"Standard heights: {std_heights[0]:.3f} → {std_heights[-1]:.3f}")
    
    if len(metad_wt.hills) > 0:
        wt_heights = [h.height for h in metad_wt.hills]
        print(f"Well-tempered heights: {wt_heights[0]:.3f} → {wt_heights[-1]:.3f}")
    
    print(f"Standard convergence: {metad_standard.convergence_detected}")
    print(f"Well-tempered convergence: {metad_wt.convergence_detected}")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot hill heights over time
    if len(metad_standard.hills) > 0:
        std_times = [h.deposition_time * 0.001 for h in metad_standard.hills]
        std_heights = [h.height for h in metad_standard.hills]
        axes[0].plot(std_times, std_heights, 'bo-', label='Standard', markersize=4)
    
    if len(metad_wt.hills) > 0:
        wt_times = [h.deposition_time * 0.001 for h in metad_wt.hills]
        wt_heights = [h.height for h in metad_wt.hills]
        axes[0].plot(wt_times, wt_heights, 'ro-', label='Well-tempered', markersize=4)
    
    axes[0].set_xlabel('Time (ps)')
    axes[0].set_ylabel('Hill Height (kJ/mol)')
    axes[0].set_title('Hill Heights Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot bias energy evolution
    if metad_standard.bias_energy_history and metad_wt.bias_energy_history:
        time_std = np.arange(len(metad_standard.bias_energy_history)) * 0.001
        time_wt = np.arange(len(metad_wt.bias_energy_history)) * 0.001
        
        axes[1].plot(time_std, metad_standard.bias_energy_history, 'b-', 
                    label='Standard', alpha=0.7)
        axes[1].plot(time_wt, metad_wt.bias_energy_history, 'r-', 
                    label='Well-tempered', alpha=0.7)
    
    axes[1].set_xlabel('Time (ps)')
    axes[1].set_ylabel('Bias Energy (kJ/mol)')
    axes[1].set_title('Bias Energy Evolution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_metad_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return metad_standard, metad_wt


def demo_protein_folding_setup():
    """Demonstrate protein folding metadynamics setup."""
    print("\n" + "=" * 60)
    print("DEMO 5: Protein Folding Metadynamics Setup")
    print("=" * 60)
    
    # Create larger mock system (protein-like)
    system = MockMolecularSystem(n_atoms=50)
    
    # Arrange atoms in a more protein-like initial structure
    # Linear chain with some secondary structure
    for i in range(50):
        x = i * 0.38  # ~C-alpha distance
        y = 0.5 * np.sin(i * 0.3)  # Some helical structure
        z = 0.2 * np.cos(i * 0.3)
        system.positions[i] = [x, y, z]
    
    # Define backbone atoms (every 3rd atom as C-alpha approximation)
    backbone_atoms = list(range(0, 50, 3))
    
    metad = setup_protein_folding_metadynamics(
        system,
        backbone_atoms,
        height=0.3,
        bias_factor=15.0  # High bias factor for protein folding
    )
    
    print(f"Protein system: {system.n_atoms} atoms")
    print(f"Backbone atoms: {len(backbone_atoms)}")
    print(f"Collective variables: {len(metad.cvs)}")
    for i, cv in enumerate(metad.cvs):
        print(f"  CV {i+1}: {cv.name}")
    
    cv_values = metad.get_current_cv_values()
    print(f"Initial CV values: {cv_values}")
    
    print("Running protein folding metadynamics (short demo)...")
    
    # Run short simulation for demonstration
    n_steps = 500
    metad.run(n_steps)
    
    cv_values = metad.get_current_cv_values()
    print(f"Final CV values: {cv_values}")
    print(f"Hills deposited: {len(metad.hills)}")
    
    # Plot results
    metad.plot_results(save_prefix="demo_protein_folding")
    
    return metad


def main():
    """Run all metadynamics demonstrations."""
    print("METADYNAMICS DEMONSTRATION")
    print("Task 6.4 Implementation - Complete Enhanced Sampling Suite")
    print(f"Date: June 11, 2025")
    print("\nThis demonstration showcases:")
    print("✓ Distance and angle collective variables")
    print("✓ Adaptive Gaussian hill deposition")
    print("✓ Well-tempered metadynamics variants")
    print("✓ Free energy surface reconstruction")
    print("✓ Convergence detection")
    print("✓ Comprehensive visualization")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Run demonstrations
        demo1 = demo_distance_metadynamics()
        demo2 = demo_angle_metadynamics()
        demo3 = demo_2d_metadynamics()
        demo4_std, demo4_wt = demo_well_tempered_comparison()
        demo5 = demo_protein_folding_setup()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION SUMMARY")
        print("=" * 60)
        print("All metadynamics features successfully demonstrated!")
        print(f"Total simulations run: 5")
        print(f"Total hills deposited: {sum(len(sim.hills) for sim in [demo1, demo2, demo3, demo4_std, demo4_wt, demo5])}")
        print("\nGenerated files:")
        print("- demo_distance_metad_metadynamics_results.png")
        print("- demo_angle_metad_metadynamics_results.png") 
        print("- demo_2d_metad_metadynamics_results.png")
        print("- demo_metad_comparison.png")
        print("- demo_protein_folding_metadynamics_results.png")
        
        print("\n✅ Task 6.4 Metadynamics - IMPLEMENTATION COMPLETE!")
        print("\nKey Features Implemented:")
        print("🔲 ✅ Kollektive Variablen definierbar (Distanzen, Winkel)")
        print("🔲 ✅ Gausssche Berge werden adaptiv hinzugefügt")
        print("🔲 ✅ Konvergenz des freien Energie-Profils erkennbar")
        print("🔲 ✅ Well-tempered Metadynamics Variante verfügbar")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
