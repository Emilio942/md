#!/usr/bin/env python
# filepath: /home/emilio/Documents/ai/md/examples/membrane_simulation.py
"""
Example of a protein-membrane system simulation using proteinMD.

This script demonstrates how to:
1. Create a lipid bilayer membrane
2. Position a protein relative to the membrane
3. Set up a simulation with periodic boundary conditions
4. Run the simulation and analyze results
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import proteinMD
sys.path.append(str(Path(__file__).parent.parent))

from proteinMD.structure.pdb_parser import PDBParser
from proteinMD.structure.protein import Membrane
from proteinMD.core.simulation import MolecularDynamicsSimulation

def run_membrane_simulation(pdb_file, output_dir, steps=1000):
    """
    Run a simulation of a protein interacting with a membrane.
    
    Parameters
    ----------
    pdb_file : str or Path
        Path to protein PDB file
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
    
    # Create a membrane 
    membrane_size = 10.0  # nm
    membrane = Membrane(
        x_dim=membrane_size, 
        y_dim=membrane_size,
        lipid_type='POPC',
        thickness=4.0,
        area_per_lipid=0.65
    )
    
    # Position the protein above the membrane
    # First center the protein at the origin
    com = protein.get_center_of_mass()
    protein.translate(-com)
    
    # Move protein above membrane
    protein.translate(np.array([0, 0, 2.5]))  # 2.5 nm above membrane center
    
    # Extract protein data
    positions, masses, charges, atom_ids, chain_ids = protein.get_atoms_array()
    
    # Get lipid positions (simplified as point particles for demonstration)
    upper_leaflet, lower_leaflet = membrane.get_lipid_positions()
    
    # Simulate each lipid as a single particle for simplicity
    lipid_mass = 600.0  # Da
    lipid_charge = 0.0  # Neutral
    
    # Combine lipids as additional particles
    num_upper_lipids = len(upper_leaflet)
    num_lower_lipids = len(lower_leaflet)
    
    # Total particles: protein atoms + lipids
    num_particles = len(positions) + num_upper_lipids + num_lower_lipids
    
    # Create combined positions array
    all_positions = np.zeros((num_particles, 3))
    all_masses = np.zeros(num_particles)
    all_charges = np.zeros(num_particles)
    
    # Add protein atoms
    all_positions[:len(positions)] = positions
    all_masses[:len(positions)] = masses
    all_charges[:len(positions)] = charges
    
    # Add upper leaflet lipids
    start_idx = len(positions)
    end_idx = start_idx + num_upper_lipids
    all_positions[start_idx:end_idx] = upper_leaflet
    all_masses[start_idx:end_idx] = lipid_mass
    all_charges[start_idx:end_idx] = lipid_charge
    
    # Add lower leaflet lipids
    start_idx = end_idx
    end_idx = start_idx + num_lower_lipids
    all_positions[start_idx:end_idx] = lower_leaflet
    all_masses[start_idx:end_idx] = lipid_mass
    all_charges[start_idx:end_idx] = lipid_charge
    
    # Box size with padding
    box_size = max(membrane_size, np.max(np.abs(all_positions))) + 2.0  # 2 nm extra padding
    box_dimensions = np.array([box_size, box_size, box_size])
    
    # Set up simulation
    print("Setting up protein-membrane simulation...")
    sim = MolecularDynamicsSimulation(
        num_particles=num_particles,
        box_dimensions=box_dimensions,
        temperature=310.0,  # Body temperature (K)
        time_step=0.002,  # ps
        boundary_condition='periodic',
        integrator='velocity-verlet',
        thermostat='berendsen'
    )
    
    # Add particles to simulation
    sim.add_particles(positions=all_positions, masses=all_masses, charges=all_charges)
    
    # Add protein bonds
    for bond in protein.get_bonds():
        # Example bond parameters
        k_bond = 1000.0  # kJ/(mol·nm²)
        r_0 = 0.15  # nm
        sim.add_bonds([(bond[0], bond[1], k_bond, r_0)])
    
    # Add restraints to lipid positions 
    # This is a simplified approach - real membrane simulations would need more complex force fields
    k_position = 100.0  # kJ/(mol·nm²)
    for i in range(len(positions), num_particles):
        sim.add_position_restraint(i, k_position)
    
    # Initialize velocities
    sim.initialize_velocities()
    
    # Define callback for progress reporting
    def progress_callback(simulation, state, step_index):
        if step_index % 100 == 0:
            print(f"Step {step_index}: T = {state['temperature']:.2f} K, "
                  f"E_kin = {state['kinetic']:.2f}, E_pot = {state['potential']:.2f}, "
                  f"E_tot = {state['total']:.2f}")
    
    # Run simulation
    print(f"Running protein-membrane simulation for {steps} steps...")
    final_state = sim.run(steps, callback=progress_callback)
    
    # Save trajectory and final state
    trajectory_file = output_dir / f"{protein.protein_id}_membrane_trajectory.npz"
    checkpoint_file = output_dir / f"{protein.protein_id}_membrane_checkpoint.gz"
    
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
    plt.title(f'Energy vs. Time for {protein.protein_id} with Membrane')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(output_dir / f"{protein.protein_id}_membrane_energy.png", dpi=300)
    plt.close()
    
    # Plot temperature over time
    plt.figure(figsize=(10, 4))
    plt.plot(time_points, sim.temperatures)
    plt.xlabel('Time (ps)')
    plt.ylabel('Temperature (K)')
    plt.title(f'Temperature vs. Time for {protein.protein_id} with Membrane')
    plt.grid(True)
    
    # Save plot
    plt.savefig(output_dir / f"{protein.protein_id}_membrane_temperature.png", dpi=300)
    plt.close()
    
    # Visualize the protein-membrane system (simplified 2D projection)
    plt.figure(figsize=(8, 8))
    
    # Get final positions
    final_positions = sim.positions
    
    # Define markers and colors for different components
    protein_atoms = final_positions[:len(positions)]
    upper_lipids = final_positions[len(positions):len(positions) + num_upper_lipids]
    lower_lipids = final_positions[len(positions) + num_upper_lipids:]
    
    # Plot protein (top view - XY plane)
    plt.scatter(protein_atoms[:, 0], protein_atoms[:, 1], 
                c='blue', alpha=0.7, s=10, label='Protein')
    
    # Plot upper leaflet
    plt.scatter(upper_lipids[:, 0], upper_lipids[:, 1], 
                c='green', alpha=0.5, s=20, label='Upper Leaflet')
    
    # Plot lower leaflet
    plt.scatter(lower_lipids[:, 0], lower_lipids[:, 1], 
                c='red', alpha=0.5, s=20, label='Lower Leaflet')
    
    plt.axis('equal')
    plt.title(f'{protein.protein_id} with Membrane (Top View)')
    plt.xlabel('X (nm)')
    plt.ylabel('Y (nm)')
    plt.legend()
    plt.grid(True)
    
    # Save visualization
    plt.savefig(output_dir / f"{protein.protein_id}_membrane_topview.png", dpi=300)
    plt.close()
    
    # Side view (XZ plane)
    plt.figure(figsize=(10, 6))
    
    # Plot protein
    plt.scatter(protein_atoms[:, 0], protein_atoms[:, 2], 
                c='blue', alpha=0.7, s=10, label='Protein')
    
    # Plot upper leaflet
    plt.scatter(upper_lipids[:, 0], upper_lipids[:, 2], 
                c='green', alpha=0.5, s=20, label='Upper Leaflet')
    
    # Plot lower leaflet
    plt.scatter(lower_lipids[:, 0], lower_lipids[:, 2], 
                c='red', alpha=0.5, s=20, label='Lower Leaflet')
    
    plt.title(f'{protein.protein_id} with Membrane (Side View)')
    plt.xlabel('X (nm)')
    plt.ylabel('Z (nm)')
    plt.legend()
    plt.grid(True)
    
    # Save visualization
    plt.savefig(output_dir / f"{protein.protein_id}_membrane_sideview.png", dpi=300)
    plt.close()
    
    return protein, membrane, sim

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a protein-membrane simulation with proteinMD")
    parser.add_argument('-p', '--pdb', required=True, help="Path to protein PDB file")
    parser.add_argument('-o', '--output', default='output/membrane', help="Output directory")
    parser.add_argument('-s', '--steps', type=int, default=1000, help="Number of simulation steps")
    
    args = parser.parse_args()
    
    run_membrane_simulation(args.pdb, args.output, args.steps)
