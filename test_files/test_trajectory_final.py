#!/usr/bin/env python3
"""
Final comprehensive test for trajectory storage functionality.
Tests all requirements for task 1.1.
"""
import sys
import numpy as np
from pathlib import Path

# Add proteinMD to path
sys.path.append(str(Path(__file__).parent))

from proteinMD.core.simulation import MolecularDynamicsSimulation

def test_trajectory_storage_comprehensive():
    """
    Comprehensive test for trajectory storage that verifies:
    1. Trajectories are correctly saved as .npz files
    2. No errors occur when loading saved trajectories  
    3. Test with at least 100 simulation steps runs successfully
    """
    
    print("🧪 COMPREHENSIVE TRAJECTORY STORAGE TEST")
    print("=" * 50)
    
    # Test 1: Basic trajectory saving
    print("\n📝 Test 1: Basic trajectory saving with 100+ steps")
    
    sim = MolecularDynamicsSimulation(
        num_particles=0,  # Start with 0, then add particles
        box_dimensions=np.array([5.0, 5.0, 5.0]),
        temperature=300.0,
        time_step=0.002
    )
    
    # Set trajectory stride
    sim.trajectory_stride = 50  # Save every 50 steps
    
    # Add particles
    positions = np.random.uniform(0, 5, (10, 3))
    masses = np.ones(10) * 12.0  # Carbon mass
    charges = np.zeros(10)  # Neutral particles
    
    sim.add_particles(positions=positions, masses=masses, charges=charges)
    num_particles = sim.num_particles  # Get actual particle count
    sim.initialize_velocities()
    
    # Run 150 steps (should produce 3 trajectory frames)
    steps = 150
    expected_frames = steps // sim.trajectory_stride
    
    print(f"Running {steps} steps with stride {sim.trajectory_stride}")
    print(f"Expected frames: {expected_frames}")
    
    # Run simulation with minimal output
    final_state = sim.run(steps, callback=None)
    
    # Save trajectory
    trajectory_file = "test_final_trajectory.npz"
    sim.save_trajectory(trajectory_file)
    
    print(f"✓ Simulation completed")
    print(f"✓ Trajectory saved to {trajectory_file}")
    
    # Test 2: Loading and verifying trajectory
    print(f"\n📂 Test 2: Loading and verifying trajectory")
    
    try:
        # Load trajectory
        data = np.load(trajectory_file)
        
        # Verify structure
        assert 'times' in data, "Missing 'times' array"
        assert 'positions' in data, "Missing 'positions' array"
        
        n_frames = len(data['times'])
        
        # Verify correct number of frames
        assert n_frames == expected_frames, f"Expected {expected_frames} frames, got {n_frames}"
        
        # Verify time progression
        times = data['times']
        assert len(times) > 0, "No time data"
        assert times[0] > 0, "First time should be > 0"
        if len(times) > 1:
            time_diffs = np.diff(times)
            expected_dt = sim.trajectory_stride * sim.time_step
            assert np.allclose(time_diffs, expected_dt, rtol=1e-6), "Time intervals incorrect"
        
        # Verify position data
        positions = data['positions']
        assert positions.shape == (n_frames, num_particles, 3), f"Wrong position shape: {positions.shape}, expected ({n_frames}, {num_particles}, 3)"
        assert not np.any(np.isnan(positions)), "NaN values in positions"
        assert not np.any(np.isinf(positions)), "Infinite values in positions"
        
        print(f"✓ Trajectory loaded successfully")
        print(f"✓ {n_frames} frames verified")
        print(f"✓ Time range: {times[0]:.3f} to {times[-1]:.3f} ps")
        print(f"✓ Position data validated")
        
    except Exception as e:
        print(f"✗ Error loading/verifying trajectory: {e}")
        return False
    
    # Test 3: Multiple trajectory saves
    print(f"\n🔄 Test 3: Multiple trajectory operations")
    
    test_cases = [
        (100, "test_100_final.npz"),
        (200, "test_200_final.npz"),
        (300, "test_300_final.npz")
    ]
    
    for steps, filename in test_cases:
        # Reset simulation
        sim.time = 0.0
        sim.step_count = 0
        sim.trajectory.clear()
        sim.energies = {'kinetic': [], 'potential': [], 'total': []}
        sim.temperatures = []
        sim.initialize_velocities()
        
        # Run simulation
        final_state = sim.run(steps, callback=None)
        sim.save_trajectory(filename)
        
        # Quick verification
        data = np.load(filename)
        expected = steps // sim.trajectory_stride
        actual = len(data['times'])
        
        if actual == expected:
            print(f"✓ {filename}: {actual} frames (correct)")
        else:
            print(f"✗ {filename}: {actual} frames (expected {expected})")
            return False
    
    # Test 4: Large simulation test
    print(f"\n🚀 Test 4: Large simulation (500 steps)")
    
    # Reset for large test
    sim.time = 0.0
    sim.step_count = 0
    sim.trajectory.clear()
    sim.energies = {'kinetic': [], 'potential': [], 'total': []}
    sim.temperatures = []
    sim.initialize_velocities()
    
    large_steps = 500
    large_file = "test_large_final.npz"
    
    final_state = sim.run(large_steps, callback=None)
    sim.save_trajectory(large_file)
    
    # Verify large simulation
    data = np.load(large_file)
    expected_large = large_steps // sim.trajectory_stride
    actual_large = len(data['times'])
    
    if actual_large == expected_large:
        print(f"✓ Large simulation: {actual_large} frames saved correctly")
    else:
        print(f"✗ Large simulation: {actual_large} frames (expected {expected_large})")
        return False
    
    # Final summary
    print(f"\n🎉 ALL TESTS PASSED!")
    print(f"✓ Trajectories are correctly saved as .npz files")
    print(f"✓ No errors occur when loading saved trajectories")
    print(f"✓ Tests with 100+ simulation steps run successfully")
    print(f"✓ Multiple file operations work correctly")
    print(f"✓ Large simulations (500 steps) work correctly")
    
    return True

if __name__ == "__main__":
    success = test_trajectory_storage_comprehensive()
    sys.exit(0 if success else 1)
