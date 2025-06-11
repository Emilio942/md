#!/usr/bin/env python3
"""
Memory-optimized molecular dynamics simulation.
Fixes identified memory leaks in long-running simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import gc
from collections import deque
from typing import Dict, Any, Optional, List
import weakref


class MemoryOptimizedMDSimulation:
    """
    Memory-optimized molecular dynamics simulation class.
    
    Fixes identified memory leaks:
    1. Bounded trajectory storage with circular buffers
    2. Limited energy/temperature history
    3. Proper matplotlib object cleanup
    4. Optimized field calculations
    5. Memory-efficient neighbor lists
    """
    
    # Physical constants
    BOLTZMANN = 1.38064852e-23  # J/K
    VACUUM_PERMITTIVITY = 8.854e-12  # F/m
    VACUUM_PERMEABILITY = 4 * np.pi * 1e-7  # T·m/A
    ELEMENTARY_CHARGE = 1.602e-19  # C
    
    def __init__(self, 
                 num_molecules: int = 5,
                 box_size: float = 10.0,  # nm
                 temperature: float = 300,  # K
                 time_step: float = 0.01,  # ps
                 charge_type: str = 'uniform',
                 boundary_condition: str = 'periodic',
                 initial_config: str = 'random',
                 max_trajectory_length: int = 1000,  # Memory limit for trajectories
                 max_energy_history: int = 10000):   # Memory limit for energy history
        
        self.num_molecules = num_molecules
        self.box_size = box_size
        self.temperature = temperature
        self.time_step = time_step
        self.charge_type = charge_type
        self.boundary_condition = boundary_condition
        
        # Memory-bounded storage
        self.max_trajectory_length = max_trajectory_length
        self.max_energy_history = max_energy_history
        
        # Use deques for bounded storage instead of lists
        self.trajectories = deque(maxlen=max_trajectory_length)
        self.energies = deque(maxlen=max_energy_history)
        self.temperatures = deque(maxlen=max_energy_history)
        
        # Initialize random generator
        np.random.seed(42)
        
        # Initialize positions and velocities
        self.initialize_configuration(initial_config)
        
        # Performance tracking
        self.last_fps = 0
        self.t_start = time.time()
        self.frame_count = 0
        
        # Optimization: Pre-allocate arrays to avoid repeated memory allocation
        self._forces_buffer = np.zeros((num_molecules, 3))
        self._temp_positions = np.zeros((num_molecules, 3))
        self._distances = np.zeros((num_molecules, num_molecules))
        
        # Neighbor list optimization
        self._neighbor_list = []
        self._neighbor_list_cutoff = 3.0  # cutoff for neighbor list
        self._neighbor_update_frequency = 10  # steps between neighbor list updates
        self._steps_since_neighbor_update = 0
        
        # Matplotlib object references for cleanup
        self._plot_objects = []
        
    def initialize_configuration(self, config_type: str):
        """Initialize positions and velocities with memory-efficient approach."""
        if config_type == 'random':
            self.positions = np.random.uniform(0, self.box_size, (self.num_molecules, 3))
        elif config_type == 'grid':
            # Arrange in a grid pattern
            grid_size = int(np.ceil(self.num_molecules ** (1/3)))
            spacing = self.box_size / grid_size
            
            positions = []
            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(grid_size):
                        if len(positions) < self.num_molecules:
                            positions.append([i*spacing + spacing/2, 
                                            j*spacing + spacing/2, 
                                            k*spacing + spacing/2])
            self.positions = np.array(positions[:self.num_molecules])
        else:  # cluster
            center = np.array([self.box_size/2, self.box_size/2, self.box_size/2])
            self.positions = center + np.random.normal(0, self.box_size/10, (self.num_molecules, 3))
            self.positions = np.clip(self.positions, 0, self.box_size)
        
        # Initialize charges
        if self.charge_type == 'uniform':
            self.charges = np.ones(self.num_molecules) * self.ELEMENTARY_CHARGE
        elif self.charge_type == 'alternating':
            self.charges = np.array([self.ELEMENTARY_CHARGE if i % 2 == 0 else -self.ELEMENTARY_CHARGE 
                                   for i in range(self.num_molecules)])
        else:  # random
            self.charges = np.random.choice([-1, 1], self.num_molecules) * self.ELEMENTARY_CHARGE
        
        # Initialize velocities
        self.initialize_velocities()
        
        # Store initial trajectory point
        self.trajectories.append(self.positions.copy())
        
    def initialize_velocities(self):
        """Initialize velocities with Maxwell-Boltzmann distribution."""
        # Generate random velocities (normal distribution)
        self.velocities = np.random.normal(0, 1, (self.num_molecules, 3))
        
        # Calculate kinetic energy and current temperature
        kinetic_energy = 0.5 * np.sum(self.velocities**2)
        current_temp = kinetic_energy / (1.5 * self.num_molecules * self.BOLTZMANN)
        
        # Scale velocities to reach desired temperature
        if current_temp > 0:
            scaling_factor = np.sqrt(self.temperature / current_temp)
            self.velocities *= scaling_factor
        
        # Remove center of mass motion
        self.velocities -= np.mean(self.velocities, axis=0)
        
    def update_neighbor_list(self):
        """Update neighbor list for efficient force calculations."""
        self._neighbor_list.clear()
        
        for i in range(self.num_molecules):
            for j in range(i+1, self.num_molecules):
                r_ij = self.positions[j] - self.positions[i]
                
                # Apply periodic boundary conditions
                if self.boundary_condition == 'periodic':
                    r_ij -= np.round(r_ij / self.box_size) * self.box_size
                
                distance = np.linalg.norm(r_ij)
                if distance < self._neighbor_list_cutoff:
                    self._neighbor_list.append((i, j, distance))
                    
        self._steps_since_neighbor_update = 0
        
    def calculate_electric_forces(self) -> np.ndarray:
        """Calculate electric forces with optimized algorithm."""
        # Reset forces buffer
        self._forces_buffer.fill(0.0)
        
        # Update neighbor list if needed
        if (self._steps_since_neighbor_update >= self._neighbor_update_frequency or
            not self._neighbor_list):
            self.update_neighbor_list()
        
        # Calculate forces using neighbor list
        for i, j, _ in self._neighbor_list:
            r_ij = self.positions[j] - self.positions[i]
            
            # Apply periodic boundary conditions
            if self.boundary_condition == 'periodic':
                r_ij -= np.round(r_ij / self.box_size) * self.box_size
            
            distance = np.linalg.norm(r_ij)
            if distance > 1e-10:  # Avoid division by zero
                # Coulomb force calculation
                force_magnitude = (self.charges[i] * self.charges[j] / 
                                 (4 * np.pi * self.VACUUM_PERMITTIVITY * distance**3))
                force_vector = force_magnitude * r_ij
                
                self._forces_buffer[i] += force_vector
                self._forces_buffer[j] -= force_vector
        
        self._steps_since_neighbor_update += 1
        return self._forces_buffer.copy()
        
    def calculate_lorentz_forces(self) -> np.ndarray:
        """Calculate Lorentz forces (magnetic forces)."""
        forces = np.zeros((self.num_molecules, 3))
        
        for i in range(self.num_molecules):
            for j in range(self.num_molecules):
                if i != j:
                    r_ij = self.positions[i] - self.positions[j]
                    
                    # Apply periodic boundary conditions
                    if self.boundary_condition == 'periodic':
                        r_ij -= np.round(r_ij / self.box_size) * self.box_size
                    
                    distance = np.linalg.norm(r_ij)
                    if distance > 1e-10:
                        # Simplified magnetic force (velocity-dependent)
                        magnetic_field = (self.VACUUM_PERMEABILITY * self.charges[j] / 
                                        (4 * np.pi * distance**2)) * np.cross(self.velocities[j], r_ij)
                        lorentz_force = self.charges[i] * np.cross(self.velocities[i], magnetic_field)
                        forces[i] += lorentz_force
        
        return forces
        
    def velocity_verlet_integration(self, forces: np.ndarray):
        """Perform velocity Verlet integration step."""
        # Update velocities by half step
        self.velocities += 0.5 * forces * self.time_step
        
        # Update positions
        self.positions += self.velocities * self.time_step
        
        # Apply boundary conditions
        if self.boundary_condition == 'periodic':
            self.positions = self.positions % self.box_size
        elif self.boundary_condition == 'reflective':
            # Reflect particles that hit boundaries
            for i in range(self.num_molecules):
                for dim in range(3):
                    if self.positions[i, dim] < 0:
                        self.positions[i, dim] = 0
                        self.velocities[i, dim] = -self.velocities[i, dim]
                    elif self.positions[i, dim] > self.box_size:
                        self.positions[i, dim] = self.box_size
                        self.velocities[i, dim] = -self.velocities[i, dim]
        
        # Calculate forces at new positions (for second half of velocity update)
        new_forces = self.calculate_electric_forces() + self.calculate_lorentz_forces()
        
        # Complete velocity update
        self.velocities += 0.5 * new_forces * self.time_step
        
    def apply_thermostat(self) -> float:
        """Apply Berendsen thermostat for temperature control."""
        # Calculate current kinetic energy and temperature
        kinetic_energy = 0.5 * np.sum(self.velocities**2)
        current_temp = kinetic_energy / (1.5 * self.num_molecules * self.BOLTZMANN)
        
        # Berendsen thermostat with coupling time
        tau = 10.0 * self.time_step  # Coupling time
        scaling_factor = np.sqrt(1 + self.time_step/tau * (self.temperature/current_temp - 1))
        
        # Avoid extreme scaling
        scaling_factor = np.clip(scaling_factor, 0.8, 1.2)
        
        self.velocities *= scaling_factor
        
        return current_temp
        
    def calculate_system_energy(self) -> tuple:
        """Calculate system energy efficiently."""
        # Kinetic energy
        kinetic_energy = 0.5 * np.sum(self.velocities**2)
        
        # Potential energy (only calculate for neighbor pairs)
        potential_energy = 0
        for i, j, distance in self._neighbor_list:
            if distance > 1e-10:
                potential_energy += (self.charges[i] * self.charges[j] / 
                                   (4 * np.pi * self.VACUUM_PERMITTIVITY * distance))
        
        total_energy = kinetic_energy + potential_energy
        
        # Store in bounded deques
        self.energies.append(total_energy)
        
        return total_energy, kinetic_energy, potential_energy
        
    def step_simulation(self) -> Dict[str, Any]:
        """Perform one simulation step with memory optimization."""
        # Calculate forces
        forces = self.calculate_electric_forces() + self.calculate_lorentz_forces()
        
        # Perform integration
        self.velocity_verlet_integration(forces)
        
        # Apply thermostat
        current_temp = self.apply_thermostat()
        self.temperatures.append(current_temp)
        
        # Store trajectory (bounded)
        self.trajectories.append(self.positions.copy())
        
        # Calculate energy
        total_energy, kinetic, potential = self.calculate_system_energy()
        
        # Performance tracking
        self.frame_count += 1
        elapsed = time.time() - self.t_start
        if elapsed > 1.0:
            self.last_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.t_start = time.time()
        
        return {
            'positions': self.positions.copy(),
            'velocities': self.velocities.copy(),
            'forces': forces.copy(),
            'temperature': current_temp,
            'energy': {'total': total_energy, 'kinetic': kinetic, 'potential': potential},
            'fps': self.last_fps
        }
        
    def calculate_field_grid(self, resolution: int = 20) -> Dict[str, np.ndarray]:
        """Calculate field grid for visualization with memory optimization."""
        # Use smaller resolution if needed to save memory
        resolution = min(resolution, 20)
        
        # Pre-allocate arrays
        x = np.linspace(0, self.box_size, resolution)
        y = np.linspace(0, self.box_size, resolution)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        E_magnitude = np.zeros_like(X)
        B_magnitude = np.zeros_like(X)
        
        # Calculate fields at grid points
        for i in range(resolution):
            for j in range(resolution):
                point = np.array([X[i, j], Y[i, j], self.box_size/2])
                
                E_total = np.zeros(3)
                B_total = np.zeros(3)
                
                # Sum contributions from all charges
                for k in range(self.num_molecules):
                    r = point - self.positions[k]
                    distance = np.linalg.norm(r)
                    
                    if distance > 1e-10:
                        # Electric field
                        E_field = (self.charges[k] / (4 * np.pi * self.VACUUM_PERMITTIVITY * distance**3)) * r
                        E_total += E_field
                        
                        # Magnetic field (simplified)
                        B_field = (self.VACUUM_PERMEABILITY * self.charges[k] / 
                                 (4 * np.pi * distance**2)) * np.cross(self.velocities[k], r)
                        B_total += B_field
                
                E_magnitude[i, j] = np.linalg.norm(E_total)
                B_magnitude[i, j] = np.linalg.norm(B_total)
        
        return {
            'X': X, 'Y': Y,
            'E_magnitude': E_magnitude,
            'B_magnitude': B_magnitude
        }
        
    def cleanup_plot_objects(self):
        """Clean up matplotlib objects to prevent memory leaks."""
        for obj_ref in self._plot_objects:
            obj = obj_ref()
            if obj is not None:
                try:
                    obj.remove()
                except:
                    pass
        self._plot_objects.clear()
        
    def run_simulation(self, steps: int) -> List[Dict[str, Any]]:
        """Run simulation for a specified number of steps."""
        results = []
        
        try:
            for step in range(steps):
                result = self.step_simulation()
                results.append(result)
                
                # Periodic garbage collection for long runs
                if step % 1000 == 0:
                    gc.collect()
                    
        except Exception as e:
            print(f"Simulation error at step {step}: {e}")
            
        return results
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        return {
            'trajectory_length': len(self.trajectories),
            'energy_history_length': len(self.energies),
            'temperature_history_length': len(self.temperatures),
            'neighbor_list_size': len(self._neighbor_list),
            'max_trajectory_length': self.max_trajectory_length,
            'max_energy_history': self.max_energy_history
        }
        
    def force_cleanup(self):
        """Force cleanup of all cached data and objects."""
        self.trajectories.clear()
        self.energies.clear() 
        self.temperatures.clear()
        self._neighbor_list.clear()
        self.cleanup_plot_objects()
        gc.collect()


# Example usage
if __name__ == "__main__":
    # Create memory-optimized simulation
    sim = MemoryOptimizedMDSimulation(
        num_molecules=10,
        box_size=10.0,
        temperature=300,
        max_trajectory_length=500,  # Limit trajectory memory
        max_energy_history=5000     # Limit energy history
    )
    
    print("Running memory-optimized simulation...")
    print(f"Initial memory stats: {sim.get_memory_stats()}")
    
    # Run a short test
    results = sim.run_simulation(100)
    
    print(f"Final memory stats: {sim.get_memory_stats()}")
    print(f"Simulation completed with {len(results)} steps")
