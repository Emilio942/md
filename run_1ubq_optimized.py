#!/usr/bin/env python3
"""
Optimized 1ubq simulation with CUDA support and minimal output.
"""
import sys
import numpy as np
import time
from pathlib import Path

# Add proteinMD to path
sys.path.append(str(Path(__file__).parent))

try:
    import cupy as cp
    CUDA_AVAILABLE = True
    print("✓ CUDA available - using GPU acceleration")
except ImportError:
    CUDA_AVAILABLE = False
    print("! CUDA not available - using CPU")

from proteinMD.core.simulation import MolecularDynamicsSimulation

def run_optimized_1ubq_simulation():
    """Run optimized 1ubq simulation with minimal output."""
    
    print("Running optimized 1ubq simulation...")
    
    # Use existing basic simulation approach
    pdb_file = "data/proteins/1ubq.pdb"
    
    if not Path(pdb_file).exists():
        print(f"✗ PDB file {pdb_file} not found")
        return
    
    start_time = time.time()
    
    try:
        # Import PDB parser
        from proteinMD.structure.pdb_parser import PDBParser
        
        # Parse protein (silent)
        parser = PDBParser()
        protein = parser.parse_file(pdb_file)
        positions, masses, charges, atom_ids, chain_ids = protein.get_atoms_array()
        
        print(f"✓ Loaded {protein.protein_id}: {len(positions)} atoms")
        
        # Create optimized simulation
        sim = MolecularDynamicsSimulation(
            num_particles=len(positions),
            box_dimensions=np.array([8.0, 8.0, 8.0]),  # Smaller box
            temperature=300.0,
            time_step=0.002
        )
        
        # Set optimized parameters for speed
        sim.trajectory_stride = 100  # Save every 100 steps
        if hasattr(sim, 'energy_stride'):
            sim.energy_stride = 200      # Less frequent energy calc
        if hasattr(sim, 'verbose'):
            sim.verbose = False          # Minimal output
        
        # Enable CUDA if available
        if CUDA_AVAILABLE:
            try:
                from proteinMD.core.cuda_forces import compute_forces_cuda
                sim.use_cuda = True
                print("✓ CUDA acceleration enabled")
            except ImportError:
                print("! CUDA import failed - using CPU")
        
        # Add particles
        sim.add_particles(positions=positions, masses=masses, charges=charges)
        sim.initialize_velocities()
        
        print(f"✓ Simulation setup complete")
        
        # Run different step counts efficiently
        test_cases = [
            (1000, "1ubq_optimized_1000.npz")  # Focus on 1000 steps only
        ]
        
        for steps, filename in test_cases:
            step_start = time.time()
            
            print(f"\n--- Running {steps} steps ---")
            
            # Reset simulation state
            sim.time = 0.0
            sim.step_count = 0
            sim.trajectory.clear()
            sim.energies = {'kinetic': [], 'potential': [], 'total': []}
            sim.temperatures = []
            sim.initialize_velocities()
            
            # Run with minimal callbacks - only show progress every 250 steps
            def minimal_callback(simulation, state, step_index):
                if step_index % 250 == 0 or step_index == steps:
                    elapsed = time.time() - step_start
                    rate = step_index / elapsed if elapsed > 0 else 0
                    print(f"Step {step_index:4d}/{steps} - {rate:.0f} steps/s")
            
            # Run simulation
            final_state = sim.run(steps, callback=minimal_callback)
            
            # Save trajectory
            output_file = f"output/{filename}"
            sim.save_trajectory(output_file)
            
            # Quick verification
            data = np.load(output_file)
            n_frames = len(data['times'])
            expected_frames = steps // sim.trajectory_stride
            
            step_time = time.time() - step_start
            
            print(f"✓ Completed in {step_time:.1f}s")
            print(f"✓ {n_frames} trajectory frames (expected {expected_frames})")
            print(f"✓ Time range: {data['times'][0]:.3f} to {data['times'][-1]:.3f} ps")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    total_time = time.time() - start_time
    print(f"\n✓ Total simulation time: {total_time:.1f}s")
    print("✓ Trajectory files saved in output/ directory")

if __name__ == "__main__":
    run_optimized_1ubq_simulation()
