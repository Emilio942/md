#!/usr/bin/env python3
"""
Demonstration of fixed trajectory storage with proper protein simulation.
"""
import sys
import numpy as np
from pathlib import Path

# Add proteinMD to path
sys.path.append(str(Path(__file__).parent))

from proteinMD.structure.pdb_parser import PDBParser
from proteinMD.core.simulation import MolecularDynamicsSimulation

def run_fixed_protein_simulation():
    """Run a protein simulation with proper step count to demonstrate fixed trajectory storage."""
    
    print("Running fixed protein simulation with multiple trajectory frames...")
    
    # Load 1ubq protein
    pdb_file = "structures/1ubq.pdb"
    if not Path(pdb_file).exists():
        print(f"PDB file {pdb_file} not found. Creating a simple test simulation instead.")
        run_simple_test()
        return
    
    try:
        # Parse PDB file
        parser = PDBParser()
        protein = parser.parse_file(pdb_file)
        print(f"Loaded protein: {protein.protein_id} ({len(protein)} atoms)")
        
        # Extract protein data
        positions, masses, charges, atom_ids, chain_ids = protein.get_atoms_array()
        
        # Set up simulation
        sim = MolecularDynamicsSimulation(
            num_particles=len(positions),
            box_dimensions=np.array([10.0, 10.0, 10.0]),
            temperature=300.0,
            time_step=0.002
        )
        
        # Set trajectory stride for more frequent saves during demo
        sim.trajectory_stride = 50  # Save every 50 steps instead of 100
        
        sim.add_particles(positions=positions, masses=masses, charges=charges)
        sim.initialize_velocities()
        
        print(f"Simulation setup complete. Trajectory stride: {sim.trajectory_stride}")
        
    except Exception as e:
        print(f"Error loading protein: {e}")
        print("Running simple test simulation instead...")
        run_simple_test()
        return
    
    # Run simulation for different step counts to show trajectory scaling
    test_cases = [
        (200, "1ubq_fixed_200_steps.npz"),   # 4 frames expected
        (500, "1ubq_fixed_500_steps.npz"),   # 10 frames expected  
        (1000, "1ubq_fixed_1000_steps.npz")  # 20 frames expected
    ]
    
    for steps, filename in test_cases:
        print(f"\n--- Running {steps} steps (should save {steps//sim.trajectory_stride} trajectory frames) ---")
        
        # Reset simulation state but keep particles
        sim.time = 0.0
        sim.step_count = 0
        sim.trajectory.clear()
        sim.energies = {'kinetic': [], 'potential': [], 'total': []}
        sim.temperatures = []
        
        # Reinitialize velocities
        sim.initialize_velocities()
        
        # Progress callback
        def progress_callback(simulation, state, step_index):
            if step_index % 100 == 0:
                print(f"Step {step_index}: time={state['time']:.4f}ps, T={state['temperature']:.1f}K, E_total={state['energy']['total']:.2f}")
        
        # Run simulation
        final_state = sim.run(steps, callback=progress_callback)
        
        # Save trajectory
        output_file = f"output/{filename}"
        sim.save_trajectory(output_file)
        
        # Verify trajectory
        try:
            data = np.load(output_file)
            n_frames = len(data['times'])
            expected_frames = steps // sim.trajectory_stride
            
            print(f"✓ Trajectory saved to {output_file}")
            print(f"✓ {n_frames} frames saved (expected {expected_frames})")
            print(f"✓ Times: {data['times']}")
            print(f"✓ Position shape: {data['positions'].shape}")
            
            # Verify we have multiple frames with different times
            if n_frames > 1:
                print(f"✓ Multiple frames confirmed - trajectory storage is working correctly!")
                time_diff = data['times'][1] - data['times'][0]
                expected_diff = sim.trajectory_stride * sim.time_step
                print(f"✓ Time interval between frames: {time_diff:.4f}ps (expected {expected_diff:.4f}ps)")
            else:
                print("⚠ Only 1 frame saved")
                
        except Exception as e:
            print(f"✗ Error verifying trajectory: {e}")

def run_simple_test():
    """Run a simple test simulation if protein loading fails."""
    print("Running simple multi-particle test...")
    
    # Create simulation with more particles to mimic protein
    sim = MolecularDynamicsSimulation(
        num_particles=100,  # Larger system
        box_dimensions=np.array([8.0, 8.0, 8.0]),
        temperature=300.0,
        time_step=0.002
    )
    
    sim.trajectory_stride = 25  # Save every 25 steps
    
    # Add random particles
    positions = np.random.uniform(0, 8, (100, 3))
    masses = np.ones(100) * 12.0  # Carbon-like masses
    charges = np.random.choice([0.0, 0.5, -0.5], 100)  # Some charges
    
    sim.add_particles(positions=positions, masses=masses, charges=charges)
    sim.initialize_velocities()
    
    print(f"Simple simulation setup complete. {sim.num_particles} particles, stride: {sim.trajectory_stride}")
    
    # Run 500 steps - should produce 20 trajectory frames
    steps = 500
    expected_frames = steps // sim.trajectory_stride
    
    print(f"\nRunning {steps} steps (expecting {expected_frames} trajectory frames)...")
    
    def progress_callback(simulation, state, step_index):
        if step_index % 100 == 0:
            print(f"Step {step_index}: T={state['temperature']:.1f}K")
    
    final_state = sim.run(steps, callback=progress_callback)
    
    # Save and verify
    output_file = "output/simple_test_fixed.npz"
    sim.save_trajectory(output_file)
    
    data = np.load(output_file)
    n_frames = len(data['times'])
    
    print(f"\n✓ Simple test completed!")
    print(f"✓ {n_frames} frames saved (expected {expected_frames})")
    print(f"✓ Times: {data['times'][:5]}...{data['times'][-5:] if len(data['times']) > 10 else data['times']}")
    print(f"✓ Position shape: {data['positions'].shape}")

if __name__ == "__main__":
    run_fixed_protein_simulation()
