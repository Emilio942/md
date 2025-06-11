#!/usr/bin/env python
"""
Advanced example demonstrating the new features in proteinMD.

This script shows how to use the enhanced features including:
1. Numerical stability improvements in force calculations
2. Barostat for pressure control
3. Visualization tools for trajectory analysis
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import proteinMD
sys.path.append(str(Path(__file__).parent.parent))

from proteinMD.core.simulation import MolecularDynamicsSimulation
from proteinMD.visualization.visualization import (
    MatplotlibTrajectoryVisualizer,
    visualize_trajectory,
    plot_energy,
    plot_temperature
)

def run_npt_simulation():
    """
    Run a simulation with constant temperature and pressure (NPT ensemble).
    
    This demonstrates the use of both thermostat and barostat.
    """
    # Create a simulation box with water-like particles
    box_dimensions = np.array([5.0, 5.0, 5.0])  # 5 nm cubic box
    
    # Initialize simulation with barostat
    sim = MolecularDynamicsSimulation(
        num_particles=0,
        box_dimensions=box_dimensions,
        temperature=300.0,  # 300 K
        thermostat='v-rescale',  # velocity rescaling thermostat
        barostat='berendsen',  # Berendsen barostat
        boundary_condition='periodic',
        time_step=0.002  # 2 fs time step
    )
    
    # Set target pressure (1 bar)
    sim.target_pressure = 1.0
    
    # Add water-like particles to the box
    num_water_molecules = 500
    positions = np.random.uniform(0, box_dimensions, (num_water_molecules, 3))
    
    # Typical masses for water (O and 2x H)
    masses = np.ones(num_water_molecules) * 18.0  # g/mol, approximate mass of water
    
    # Partial charges (roughly equivalent to water)
    charges = np.random.normal(0.0, 0.1, num_water_molecules)  # small random charges
    
    # Add particles to simulation
    sim.add_particles(positions, masses, charges)
    
    # Initialize velocities based on temperature
    sim.initialize_velocities(temperature=300.0)
    
    # Create output directory
    output_dir = Path("simulation_output")
    output_dir.mkdir(exist_ok=True)
    
    # Prepare for trajectory and data collection
    trajectory = []
    energies = {'kinetic': [], 'potential': [], 'total': []}
    temperatures = []
    pressures = []
    
    # Number of steps
    n_steps = 1000
    print_frequency = 100
    
    print(f"Starting NPT simulation for {n_steps} steps...")
    
    # Run simulation
    for step in range(n_steps):
        # Update (one step of simulation)
        sim.step()
        
        # Apply thermostat and barostat
        sim.apply_thermostat()
        pressure = sim.apply_barostat()
        
        # Store trajectory data
        if step % 10 == 0:  # Save every 10th frame
            trajectory.append(sim.positions.copy())
        
        # Store energy data
        kinetic_energy = sim.calculate_kinetic_energy()
        potential_energy = sim.calculate_potential_energy()
        total_energy = kinetic_energy + potential_energy
        
        energies['kinetic'].append(kinetic_energy)
        energies['potential'].append(potential_energy)
        energies['total'].append(total_energy)
        
        # Store temperature and pressure
        temperatures.append(sim.calculate_temperature(kinetic_energy))
        pressures.append(pressure)
        
        # Print progress
        if step % print_frequency == 0 or step == n_steps - 1:
            print(f"Step {step}: T = {temperatures[-1]:.1f} K, P = {pressures[-1]:.1f} bar, "
                  f"E = {total_energy:.1f} kJ/mol")
    
    print("Simulation complete!")
    
    # Save trajectory data
    trajectory_array = np.array(trajectory)
    np.savez(output_dir / "trajectory.npz", 
             trajectory=trajectory_array,
             box_dimensions=sim.box_dimensions)
    
    # Save other data
    np.savez(output_dir / "simulation_data.npz",
             energies_kinetic=np.array(energies['kinetic']),
             energies_potential=np.array(energies['potential']),
             energies_total=np.array(energies['total']),
             temperatures=np.array(temperatures),
             pressures=np.array(pressures))
    
    # Visualize the results
    visualize_results(output_dir)
    
    return sim, trajectory, energies, temperatures, pressures

def visualize_results(output_dir):
    """Visualize the simulation results using the new visualization tools."""
    # Load trajectory
    trajectory_file = output_dir / "trajectory.npz"
    
    # Load simulation data
    data_file = output_dir / "simulation_data.npz"
    data = np.load(data_file)
    
    # Create energy plot
    energy_data = {
        'kinetic': data['energies_kinetic'],
        'potential': data['energies_potential'],
        'total': data['energies_total']
    }
    
    # Plot energies
    fig_energy, ax_energy = plot_energy(energy_data, output_file=output_dir / "energy_plot.png")
    
    # Plot temperature with target temperature line
    fig_temp, ax_temp = plot_temperature(
        data['temperatures'],
        target_temperature=300.0,
        output_file=output_dir / "temperature_plot.png"
    )
    
    # Plot pressure
    fig_pressure, ax_pressure = plt.subplots(figsize=(12, 6))
    time_points = np.arange(len(data['pressures']))
    ax_pressure.plot(time_points, data['pressures'], 'b-', label='System Pressure')
    ax_pressure.axhline(y=1.0, color='r', linestyle='--', label='Target Pressure (1 bar)')
    ax_pressure.set_xlabel('Frame')
    ax_pressure.set_ylabel('Pressure (bar)')
    ax_pressure.set_title('System Pressure over Time')
    ax_pressure.legend()
    ax_pressure.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    fig_pressure.savefig(output_dir / "pressure_plot.png", dpi=300, bbox_inches='tight')
    
    # Create trajectory visualization
    visualizer = visualize_trajectory(
        str(trajectory_file),
        output_file=output_dir / "trajectory_animation.gif"
    )
    
    # Create a snapshot of the first and last frames
    fig_first, ax_first = visualizer.render_frame(frame_index=0)
    fig_first.savefig(output_dir / "first_frame.png", dpi=300, bbox_inches='tight')
    
    fig_last, ax_last = visualizer.render_frame(frame_index=-1)
    fig_last.savefig(output_dir / "last_frame.png", dpi=300, bbox_inches='tight')
    
    print(f"Visualization complete! Results saved to {output_dir}")

def test_numerical_stability():
    """Demonstrate the numerical stability improvements in force calculations."""
    print("Testing numerical stability of force calculations...")
    
    # Create a simple simulation
    sim = MolecularDynamicsSimulation(
        num_particles=0,
        box_dimensions=np.array([3.0, 3.0, 3.0])
    )
    
    # Add two particles very close to each other (which would cause problems without stability fixes)
    positions = np.array([
        [1.5, 1.5, 1.5],  # Center of box
        [1.5, 1.5, 1.501]  # Extremely close (0.001 nm apart)
    ])
    masses = np.array([12.0, 12.0])  # amu
    charges = np.array([1.0, -1.0])  # strong opposite charges
    
    sim.add_particles(positions, masses, charges)
    
    # Calculate forces (this would crash or give NaN/inf without our stability improvements)
    forces = sim._calculate_nonbonded_forces()
    
    # Check if forces are finite
    if np.all(np.isfinite(forces)):
        print("✓ Force calculation is numerically stable even with extremely close particles!")
    else:
        print("✗ Force calculation produced non-finite values!")
    
    # Print the force magnitude
    force_magnitude = np.linalg.norm(forces[0])
    print(f"Force magnitude: {force_magnitude:.2f} kJ/(mol·nm)")
    
    # Check if force is limited properly
    if force_magnitude <= 1000.0:  # Our force limiting threshold
        print("✓ Force magnitude is properly limited!")
    else:
        print("✗ Force exceeded the limiting threshold!")
        
    return sim, forces

if __name__ == "__main__":
    # Run numerical stability test
    test_numerical_stability()
    
    print("\n" + "="*50 + "\n")
    
    # Run NPT simulation with visualization
    sim, trajectory, energies, temperatures, pressures = run_npt_simulation()
