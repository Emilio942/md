"""
Core module for molecular dynamics simulations.

This module contains the fundamental classes and functions for MD simulations,
including system initialization, force calculations, and integration algorithms.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MDSystem:
    """
    Main class representing a molecular dynamics system.
    
    This class manages the entire simulation system, including particles,
    force field parameters, and simulation conditions.
    """
    
    def __init__(self, 
                 name: str = "MD_System",
                 temperature: float = 300.0,  # K
                 pressure: float = 1.0,       # atm
                 time_step: float = 0.002,    # ps
                 box_size: Optional[np.ndarray] = None):
        """
        Initialize a molecular dynamics system.
        
        Parameters
        ----------
        name : str
            Name of the simulation system
        temperature : float
            Target temperature in Kelvin
        pressure : float
            Target pressure in atmospheres
        time_step : float
            Integration time step in picoseconds
        box_size : np.ndarray, optional
            Simulation box dimensions [x, y, z] in nanometers
        """
        self.name = name
        self.temperature = temperature
        self.pressure = pressure
        self.time_step = time_step
        
        # Initialize box size with default if not provided
        if box_size is None:
            self.box_size = np.array([10.0, 10.0, 10.0])  # nm
        else:
            self.box_size = box_size
        
        # Initialize empty containers for particles and molecules
        self.particles = []
        self.molecules = []
        self.proteins = []
        
        # Simulation state
        self.current_step = 0
        self.total_steps = 0
        self.elapsed_time = 0.0  # ps
        
        # Force field and integrator (to be set later)
        self.force_field = None
        self.integrator = None
        
        logger.info(f"Initialized MD system: {self.name}")
        logger.info(f"Temperature: {self.temperature} K, Pressure: {self.pressure} atm")
        logger.info(f"Box size: {self.box_size} nm, Time step: {self.time_step} ps")
    
    def add_protein(self, protein):
        """Add a protein to the simulation system."""
        self.proteins.append(protein)
        self.particles.extend(protein.atoms)
        self.molecules.append(protein)
        logger.info(f"Added protein with {len(protein.atoms)} atoms to system")
    
    def add_molecules(self, molecules):
        """Add molecules to the simulation system."""
        if not isinstance(molecules, list):
            molecules = [molecules]
        
        for molecule in molecules:
            self.molecules.append(molecule)
            self.particles.extend(molecule.atoms)
        
        logger.info(f"Added {len(molecules)} molecules to system")
    
    def set_force_field(self, force_field):
        """Set the force field for the simulation."""
        self.force_field = force_field
        logger.info(f"Set force field: {force_field.name}")
    
    def set_integrator(self, integrator):
        """Set the integrator for the simulation."""
        self.integrator = integrator
        logger.info(f"Set integrator: {integrator.name}")
    
    def prepare_simulation(self):
        """Prepare the system for simulation."""
        if not self.particles:
            raise ValueError("No particles in the system. Add proteins or molecules first.")
        
        if self.force_field is None:
            raise ValueError("No force field set for the system.")
        
        if self.integrator is None:
            raise ValueError("No integrator set for the system.")
        
        # Initialize positions, velocities, forces
        self.positions = np.array([p.position for p in self.particles])
        self.velocities = np.array([p.velocity for p in self.particles])
        self.forces = np.zeros_like(self.positions)
        
        # Initialize force field parameters
        self.force_field.initialize(self)
        
        # Initialize integrator
        self.integrator.initialize(self)
        
        logger.info(f"System prepared for simulation with {len(self.particles)} particles")
    
    def run(self, steps: int, output_freq: int = 100, trajectory_file: str = None):
        """
        Run the molecular dynamics simulation.
        
        Parameters
        ----------
        steps : int
            Number of simulation steps to run
        output_freq : int
            Frequency of output generation
        trajectory_file : str, optional
            File to write trajectory data
        """
        if not hasattr(self, 'positions'):
            self.prepare_simulation()
        
        self.total_steps = steps
        
        logger.info(f"Starting simulation for {steps} steps")
        
        for step in range(steps):
            # Calculate forces
            self.forces = self.force_field.calculate_forces(self.positions)
            
            # Update positions and velocities
            self.positions, self.velocities = self.integrator.step(
                self.positions, self.velocities, self.forces)
            
            # Apply periodic boundary conditions
            self.positions = self.positions % self.box_size
            
            # Update time
            self.current_step += 1
            self.elapsed_time += self.time_step
            
            # Output if needed
            if step % output_freq == 0:
                logger.info(f"Step {step}/{steps} completed")
                
                # Save trajectory if requested
                if trajectory_file:
                    self._save_trajectory_frame(trajectory_file)
        
        logger.info(f"Simulation completed: {steps} steps, {self.elapsed_time} ps elapsed")
    
    def _save_trajectory_frame(self, trajectory_file):
        """Save the current frame to a trajectory file."""
        # Simplified implementation - in a real system, would use specialized formats
        with open(trajectory_file, 'a') as f:
            f.write(f"Frame {self.current_step}, Time: {self.elapsed_time} ps\n")
            for i, pos in enumerate(self.positions):
                f.write(f"Atom {i}: {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}\n")
            f.write("\n")


class Particle:
    """
    Class representing a single particle in the MD simulation.
    
    This could be an atom or a coarse-grained particle.
    """
    
    def __init__(self, 
                 particle_id: int,
                 name: str,
                 element: str,
                 mass: float,
                 charge: float,
                 position: np.ndarray,
                 velocity: Optional[np.ndarray] = None):
        """
        Initialize a particle.
        
        Parameters
        ----------
        particle_id : int
            Unique identifier for the particle
        name : str
            Name of the particle (e.g., atom name)
        element : str
            Chemical element
        mass : float
            Mass in atomic mass units (u)
        charge : float
            Charge in elementary charge units (e)
        position : np.ndarray
            Initial position [x, y, z] in nanometers
        velocity : np.ndarray, optional
            Initial velocity [vx, vy, vz] in nm/ps
        """
        self.id = particle_id
        self.name = name
        self.element = element
        self.mass = mass
        self.charge = charge
        self.position = np.array(position, dtype=float)
        
        # Initialize velocity if not provided
        if velocity is None:
            self.velocity = np.zeros(3, dtype=float)
        else:
            self.velocity = np.array(velocity, dtype=float)
        
        # Additional properties
        self.force = np.zeros(3, dtype=float)
        self.potential_energy = 0.0
        
    def __repr__(self):
        return f"Particle(id={self.id}, name='{self.name}', element='{self.element}')"


class Integrator:
    """Base class for integration algorithms."""
    
    def __init__(self, name: str, time_step: float):
        """
        Initialize an integrator.
        
        Parameters
        ----------
        name : str
            Name of the integrator
        time_step : float
            Integration time step in picoseconds
        """
        self.name = name
        self.time_step = time_step
    
    def initialize(self, system):
        """Initialize the integrator with a system reference."""
        self.system = system
    
    def step(self, positions, velocities, forces):
        """
        Perform one integration step.
        
        This method should be implemented by derived classes.
        """
        raise NotImplementedError("This method should be implemented by derived classes")


class VelocityVerlet(Integrator):
    """Velocity Verlet integration algorithm."""
    
    def __init__(self, time_step: float):
        """Initialize a Velocity Verlet integrator."""
        super().__init__("Velocity Verlet", time_step)
    
    def step(self, positions, velocities, forces):
        """
        Perform one Velocity Verlet integration step.
        
        Parameters
        ----------
        positions : np.ndarray
            Current positions of all particles
        velocities : np.ndarray
            Current velocities of all particles
        forces : np.ndarray
            Current forces on all particles
        
        Returns
        -------
        tuple
            Updated positions and velocities
        """
        # Get mass array from system
        masses = np.array([p.mass for p in self.system.particles])
        masses = masses.reshape(-1, 1)  # Reshape for broadcasting
        
        # Calculate accelerations
        accelerations = forces / masses
        
        # Update velocities by half step
        velocities_half = velocities + 0.5 * accelerations * self.time_step
        
        # Update positions
        new_positions = positions + velocities_half * self.time_step
        
        # Calculate new forces (typically done outside this method)
        new_forces = self.system.force_field.calculate_forces(new_positions)
        
        # Calculate new accelerations
        new_accelerations = new_forces / masses
        
        # Update velocities for the second half step
        new_velocities = velocities_half + 0.5 * new_accelerations * self.time_step
        
        return new_positions, new_velocities


class LangevinIntegrator(Integrator):
    """
    Langevin dynamics integrator for temperature control.
    
    This integrator adds friction and random forces to mimic
    the effect of a heat bath at a specific temperature.
    """
    
    def __init__(self, time_step: float, temperature: float, friction_coeff: float = 1.0):
        """
        Initialize a Langevin integrator.
        
        Parameters
        ----------
        time_step : float
            Integration time step in picoseconds
        temperature : float
            Target temperature in Kelvin
        friction_coeff : float
            Friction coefficient in ps^-1
        """
        super().__init__("Langevin Dynamics", time_step)
        self.temperature = temperature
        self.friction_coeff = friction_coeff
        self.kB = 8.31446e-3  # Boltzmann constant in kJ/(mol·K)
    
    def step(self, positions, velocities, forces):
        """
        Perform one Langevin integration step.
        
        This implements a simple first-order integration scheme for
        the Langevin equation of motion.
        """
        # Get mass array
        masses = np.array([p.mass for p in self.system.particles])
        masses = masses.reshape(-1, 1)  # Reshape for broadcasting
        
        # Calculate deterministic acceleration (force/mass)
        accelerations = forces / masses
        
        # Calculate friction term
        friction = -self.friction_coeff * velocities
        
        # Calculate random force term (Gaussian noise)
        sigma = np.sqrt(2.0 * self.friction_coeff * self.kB * self.temperature / masses / self.time_step)
        random_force = sigma * np.random.normal(0.0, 1.0, velocities.shape)
        
        # Update velocities
        new_velocities = velocities + (accelerations + friction + random_force) * self.time_step
        
        # Update positions
        new_positions = positions + new_velocities * self.time_step
        
        return new_positions, new_velocities
