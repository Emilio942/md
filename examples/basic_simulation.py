#!/usr/bin/env python
# filepath: /home/emilio/Documents/ai/md/examples/basic_simulation.py
"""
Basic example of using proteinMD for molecular dynamics simulation.

This script demonstrates how to:
1. Load a protein structure from a PDB file
2. Set up a molecular dynamics simulation
3. Run the simulation
4. Save and analyze the results
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import proteinMD
sys.path.append(str(Path(__file__).parent.parent))

from proteinMD.structure.pdb_parser import PDBParser
from proteinMD.core.simulation import MolecularDynamicsSimulation

def run_simulation(pdb_file, output_dir, steps=1000):
    """
    Run a simple MD simulation on a protein.
    
    Parameters
    ----------
    pdb_file : str or Path
        Path to PDB file
    output_dir : str or Path
        Directory for output files
    steps : int, optional
        Number of simulation steps
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse PDB file
    print(f"Loading protein structure from {pdb_file}")
    parser = PDBParser()
    protein = parser.parse_file(pdb_file)
    
    print(f"Loaded protein: {protein.protein_id} ({len(protein)} atoms, {len(protein.chains)} chains)")
    
    # Extract protein data for simulation
    positions, masses, charges, atom_ids, chain_ids = protein.get_atoms_array()
    
    # Set up simulation
    print("Setting up simulation...")
    sim = MolecularDynamicsSimulation(
        num_particles=len(positions),
        box_dimensions=np.array([10.0, 10.0, 10.0]),  # nm
        temperature=300.0,  # K
        time_step=0.002,  # ps
        boundary_condition='periodic',
        integrator='velocity-verlet',
        thermostat='berendsen'
    )
    
    # Add particles to simulation
    sim.add_particles(positions=positions, masses=masses, charges=charges)
    
    # Add bonds from protein structure
    print("Adding bonds from protein structure...")
    for bond in protein.get_bonds():
        # Example bond parameters (these should be determined from force field)
        k_bond = 1000.0  # kJ/(mol·nm²)
        r_0 = 0.15  # nm
        sim.add_bonds([(bond[0], bond[1], k_bond, r_0)])
    
    # Generate angles and dihedrals from bond connectivity
    print("Generating angles and dihedrals from bond connectivity...")
    num_angles = sim.generate_angles_from_bonds()
    num_dihedrals = sim.generate_dihedrals_from_bonds()
    
    print(f"Added {num_angles} angles and {num_dihedrals} dihedrals")
    
    # Initialize velocities from Maxwell-Boltzmann distribution
    sim.initialize_velocities()
    
    # Define callback to report progress
    def progress_callback(simulation, state, step_index):
        if step_index % 100 == 0:
            print(f"Step {step_index}: T = {state['temperature']:.2f} K, "
                  f"E_kin = {state['kinetic']:.2f}, E_pot = {state['potential']:.2f}, "
                  f"E_tot = {state['total']:.2f}")
    
    # Run simulation
    print(f"Running simulation for {steps} steps...")
    final_state = sim.run(steps, callback=progress_callback)
    
    # Save trajectory and final state
    trajectory_file = output_dir / f"{protein.protein_id}_trajectory.npz"
    checkpoint_file = output_dir / f"{protein.protein_id}_checkpoint.gz"
    
    sim.save_trajectory(trajectory_file)
    sim.save_checkpoint(checkpoint_file)
    
    print(f"Simulation completed. Results saved to {output_dir}")
    
    # Plot energy over time
    plt.figure(figsize=(10, 6))
    
    # Get time series data
    time_points = np.arange(len(sim.energies['kinetic'])) * sim.time_step
    
    plt.plot(time_points, sim.energies['kinetic'], label='Kinetic')
    plt.plot(time_points, sim.energies['potential'], label='Potential')
    plt.plot(time_points, sim.energies['total'], label='Total')
    
    plt.xlabel('Time (ps)')
    plt.ylabel('Energy (kJ/mol)')
    plt.title(f'Energy vs. Time for {protein.protein_id}')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(output_dir / f"{protein.protein_id}_energy.png", dpi=300)
    plt.close()
    
    # Plot temperature over time
    plt.figure(figsize=(10, 4))
    plt.plot(time_points, sim.temperatures)
    plt.xlabel('Time (ps)')
    plt.ylabel('Temperature (K)')
    plt.title(f'Temperature vs. Time for {protein.protein_id}')
    plt.grid(True)
    
    # Save plot
    plt.savefig(output_dir / f"{protein.protein_id}_temperature.png", dpi=300)
    plt.close()
    
    return protein, sim

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a basic MD simulation with proteinMD")
    parser.add_argument('-p', '--pdb', required=True, help="Path to PDB file")
    parser.add_argument('-o', '--output', default='output', help="Output directory")
    parser.add_argument('-s', '--steps', type=int, default=1000, help="Number of simulation steps")
    
    args = parser.parse_args()
    
    run_simulation(args.pdb, args.output, args.steps)
