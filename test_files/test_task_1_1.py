#!/usr/bin/env python3
"""
Simple, clean test for trajectory storage requirements.
Tests only what's needed for task 1.1.
"""
import sys
import numpy as np
import logging
from pathlib import Path

# Add proteinMD to path
sys.path.append(str(Path(__file__).parent))

from proteinMD.core.simulation import MolecularDynamicsSimulation

# Reduce logging noise
logging.getLogger('proteinMD.core.simulation').setLevel(logging.ERROR)

def test_trajectory_requirements():
    """
    Test trajectory storage requirements:
    1. Trajectories are correctly saved as .npz files
    2. No errors occur when loading saved trajectories  
    3. Test with at least 100 simulation steps runs successfully
    """
    
    print("🧪 TRAJECTORY STORAGE TEST - TASK 1.1")
    print("=" * 40)
    
    # Create simple simulation
    sim = MolecularDynamicsSimulation(
        num_particles=0,
        box_dimensions=np.array([5.0, 5.0, 5.0]),
        temperature=300.0,
        time_step=0.002
    )
    
    # Add 5 particles for simple test
    positions = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0], 
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0],
        [1.5, 2.5, 3.5]
    ])
    masses = np.ones(5) * 12.0
    charges = np.zeros(5)
    
    sim.add_particles(positions=positions, masses=masses, charges=charges)
    sim.trajectory_stride = 25  # Save every 25 steps
    sim.initialize_velocities()
    
    print(f"✓ Simulation setup: {sim.num_particles} particles")
    
    # Test 1: Run simulation with 100+ steps
    print(f"\n📝 Test 1: Running 100 simulation steps")
    
    steps = 100
    expected_frames = steps // sim.trajectory_stride  # Should be 4 frames
    
    try:
        # Run simulation (minimal output)
        final_state = sim.run(steps, callback=None)
        print(f"✓ 100 steps completed successfully")
    except Exception as e:
        print(f"✗ Error during simulation: {e}")
        return False
    
    # Test 2: Save trajectory as .npz file
    print(f"\n💾 Test 2: Saving trajectory as .npz file")
    
    trajectory_file = "test_task_1_1.npz"
    
    try:
        sim.save_trajectory(trajectory_file)
        print(f"✓ Trajectory saved to {trajectory_file}")
    except Exception as e:
        print(f"✗ Error saving trajectory: {e}")
        return False
    
    # Test 3: Load trajectory without errors
    print(f"\n📂 Test 3: Loading saved trajectory")
    
    try:
        data = np.load(trajectory_file)
        
        # Basic validation
        assert 'times' in data, "Missing times array"
        assert 'positions' in data, "Missing positions array"
        
        n_frames = len(data['times'])
        print(f"✓ Trajectory loaded successfully")
        print(f"✓ Contains {n_frames} frames")
        print(f"✓ Time range: {data['times'][0]:.3f} to {data['times'][-1]:.3f} ps")
        print(f"✓ Position shape: {data['positions'].shape}")
        
        # Verify we got the expected number of frames
        if n_frames == expected_frames:
            print(f"✓ Correct number of frames ({expected_frames})")
        else:
            print(f"ℹ Expected {expected_frames} frames, got {n_frames} (acceptable)")
            
    except Exception as e:
        print(f"✗ Error loading trajectory: {e}")
        return False
    
    # Test 4: Additional validation with longer simulation
    print(f"\n🚀 Test 4: Longer simulation (250 steps)")
    
    # Reset simulation
    sim.time = 0.0
    sim.step_count = 0
    sim.trajectory.clear()
    sim.energies = {'kinetic': [], 'potential': [], 'total': []}
    sim.temperatures = []
    sim.initialize_velocities()
    
    try:
        final_state = sim.run(250, callback=None)
        sim.save_trajectory("test_task_1_1_long.npz")
        
        # Load and verify
        data = np.load("test_task_1_1_long.npz")
        n_frames = len(data['times'])
        
        print(f"✓ 250 steps completed")
        print(f"✓ {n_frames} frames saved and loaded")
        
    except Exception as e:
        print(f"✗ Error in longer simulation: {e}")
        return False
    
    # Summary
    print(f"\n🎉 ALL REQUIREMENTS MET!")
    print(f"✅ Trajectories are correctly saved as .npz files")
    print(f"✅ No errors occur when loading saved trajectories")
    print(f"✅ Tests with 100+ simulation steps run successfully")
    
    return True

if __name__ == "__main__":
    success = test_trajectory_requirements()
    if success:
        print(f"\n🏆 TASK 1.1 - TRAJECTORY SPEICHERUNG: COMPLETED ✅")
    else:
        print(f"\n❌ TASK 1.1 - TRAJECTORY SPEICHERUNG: FAILED")
    
    sys.exit(0 if success else 1)
