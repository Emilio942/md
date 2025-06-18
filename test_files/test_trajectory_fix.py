#!/usr/bin/env python3
"""
Test script to reproduce the trajectory storage issue.
"""
import sys
import os
import numpy as np
from pathlib import Path

# Add proteinMD to path
sys.path.append(str(Path(__file__).parent))

from proteinMD.core.simulation import MolecularDynamicsSimulation

def test_trajectory_saving():
    """Test trajectory saving with different numbers of steps."""
    
    print("Testing trajectory saving with different step counts...")
    
    # Create a simple simulation with 5 particles
    sim = MolecularDynamicsSimulation(
        num_particles=5,
        box_dimensions=np.array([5.0, 5.0, 5.0]),
        temperature=300.0,
        time_step=0.002  # Same as protein simulations
    )
    
    # Set trajectory stride (default is 100)
    sim.trajectory_stride = 100
    
    # Add some particles
    positions = np.random.uniform(0, 5, (5, 3))
    masses = np.ones(5)
    charges = np.zeros(5)
    
    sim.add_particles(positions=positions, masses=masses, charges=charges)
    sim.initialize_velocities()
    
    # Test different step counts
    test_cases = [
        (50, "test_50_steps.npz"),
        (100, "test_100_steps.npz"), 
        (250, "test_250_steps.npz"),
        (500, "test_500_steps.npz"),
        (1000, "test_1000_steps.npz")
    ]
    
    for steps, filename in test_cases:
        print(f"\n--- Testing {steps} steps ---")
        
        # Reset simulation state
        sim.time = 0.0
        sim.step_count = 0
        sim.trajectory.clear()
        sim.energies = {'kinetic': [], 'potential': [], 'total': []}
        sim.temperatures = []
        
        # Run simulation
        def progress_callback(step_index, simulation):
            if step_index % 100 == 0 or step_index < 10:
                current_T = simulation.calculate_temperature()
                print(f"Step {step_index}: time={simulation.time:.4f}ps, T={current_T:.2f}K")
        
        final_state = sim.run(steps, callback=progress_callback)
        
        # Save trajectory
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            temp_filename = tmp.name
        sim.save_trajectory(temp_filename)
        
        # Check trajectory file
        try:
            data = np.load(temp_filename)
            n_frames = len(data['times'])
            expected_frames = (steps // sim.trajectory_stride)
            if steps % sim.trajectory_stride == 0:
                expected_frames += 0  # Only count multiples
            else:
                expected_frames = max(1, expected_frames)  # At least 1 frame if we went past stride
                
            print(f"Trajectory file {filename}:")
            print(f"  - {n_frames} frames saved")
            print(f"  - Expected ~{expected_frames} frames (steps {steps}, stride {sim.trajectory_stride})")
            
            os.unlink(temp_filename)  # Clean up
            print(f"  - Times: {data['times']}")
            print(f"  - Position shape: {data['positions'].shape}")
            
            # Print trajectory save points
            save_steps = [i for i in range(1, steps+1) if i % sim.trajectory_stride == 0]
            print(f"  - Expected save steps: {save_steps}")
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")

if __name__ == "__main__":
    test_trajectory_saving()
