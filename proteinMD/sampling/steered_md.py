"""
Steered Molecular Dynamics Implementation

Task 6.3: Steered Molecular Dynamics 📊
Status: IN PROGRESS

Kraftgeführte Simulationen für Protein-Unfolding

This module implements steered molecular dynamics (SMD) methods for applying
external forces to specific atoms to study force-induced conformational changes,
protein unfolding, and ligand unbinding pathways.

Features:
- Constant force mode (CF-SMD)
- Constant velocity mode (CV-SMD)
- Work calculation according to Jarzynski equality
- Force curve visualization and analysis
- Support for multiple pulling coordinates
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Callable
from pathlib import Path
import json
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SMDParameters:
    """Parameters for steered molecular dynamics simulation."""
    
    # Pulling coordinate definition
    atom_indices: List[int]  # Atoms involved in pulling coordinate
    coordinate_type: str = "distance"  # "distance", "angle", "dihedral", "com_distance"
    
    # SMD mode
    mode: str = "constant_velocity"  # "constant_velocity" or "constant_force"
    
    # Constant velocity parameters
    pulling_velocity: float = 0.01  # nm/ps (for CV-SMD)
    spring_constant: float = 1000.0  # kJ/(mol·nm²)
    
    # Constant force parameters
    applied_force: float = 100.0  # pN (for CF-SMD)
    
    # Simulation parameters
    n_steps: int = 10000
    output_frequency: int = 100
    
    # Reference coordinate (initial value, automatically determined if None)
    reference_coordinate: Optional[float] = None
    
    # Direction control
    pulling_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    
    # Output settings
    save_trajectory: bool = True
    save_force_curves: bool = True


class CoordinateCalculator:
    """Calculator for different types of pulling coordinates."""
    
    @staticmethod
    def distance(positions: np.ndarray, atom_indices: List[int]) -> float:
        """Calculate distance between two atoms."""
        if len(atom_indices) != 2:
            raise ValueError("Distance coordinate requires exactly 2 atoms")
        
        pos1, pos2 = positions[atom_indices[0]], positions[atom_indices[1]]
        return np.linalg.norm(pos2 - pos1)
    
    @staticmethod
    def com_distance(positions: np.ndarray, atom_indices: List[int], 
                    masses: Optional[np.ndarray] = None) -> float:
        """Calculate distance between centers of mass of two groups."""
        if len(atom_indices) < 2:
            raise ValueError("COM distance requires at least 2 atoms")
        
        # Split atoms into two groups (first half and second half)
        mid = len(atom_indices) // 2
        group1 = atom_indices[:mid]
        group2 = atom_indices[mid:]
        
        if masses is not None:
            # Weighted center of mass
            masses1 = masses[group1]
            masses2 = masses[group2]
            com1 = np.average(positions[group1], axis=0, weights=masses1)
            com2 = np.average(positions[group2], axis=0, weights=masses2)
        else:
            # Geometric center
            com1 = np.mean(positions[group1], axis=0)
            com2 = np.mean(positions[group2], axis=0)
        
        return np.linalg.norm(com2 - com1)
    
    @staticmethod
    def angle(positions: np.ndarray, atom_indices: List[int]) -> float:
        """Calculate angle between three atoms (in radians)."""
        if len(atom_indices) != 3:
            raise ValueError("Angle coordinate requires exactly 3 atoms")
        
        pos1, pos2, pos3 = positions[atom_indices]
        
        # Vectors from central atom to the other two
        v1 = pos1 - pos2
        v2 = pos3 - pos2
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        
        return np.arccos(cos_angle)
    
    @staticmethod
    def dihedral(positions: np.ndarray, atom_indices: List[int]) -> float:
        """Calculate dihedral angle between four atoms (in radians)."""
        if len(atom_indices) != 4:
            raise ValueError("Dihedral coordinate requires exactly 4 atoms")
        
        pos1, pos2, pos3, pos4 = positions[atom_indices]
        
        # Vectors along the bonds
        b1 = pos2 - pos1
        b2 = pos3 - pos2
        b3 = pos4 - pos3
        
        # Normal vectors to the planes
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        # Check for degenerate cases (collinear atoms)
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        
        if n1_norm < 1e-10 or n2_norm < 1e-10:
            # Return 0 for degenerate dihedral (atoms are collinear)
            return 0.0
        
        # Normalize normal vectors
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        
        # Calculate dihedral angle
        cos_dihedral = np.dot(n1, n2)
        cos_dihedral = np.clip(cos_dihedral, -1.0, 1.0)
        
        # Handle sign
        cross_product = np.cross(n1, n2)
        sign = np.sign(np.dot(cross_product, b2))
        
        return sign * np.arccos(cos_dihedral)


class SMDForceCalculator:
    """Calculate SMD forces for different modes."""
    
    def __init__(self, parameters: SMDParameters):
        self.params = parameters
        self.coordinate_calc = CoordinateCalculator()
        
        # Store work calculation
        self.accumulated_work = 0.0
        self.force_history = []
        self.coordinate_history = []
        self.work_history = []
        
    def calculate_coordinate(self, positions: np.ndarray, 
                           masses: Optional[np.ndarray] = None) -> float:
        """Calculate the current value of the pulling coordinate."""
        calc_method = getattr(self.coordinate_calc, self.params.coordinate_type)
        
        if self.params.coordinate_type == "com_distance":
            return calc_method(positions, self.params.atom_indices, masses)
        else:
            return calc_method(positions, self.params.atom_indices)
    
    def calculate_target_coordinate(self, step: int, initial_coordinate: float) -> float:
        """Calculate target coordinate value for current step."""
        if self.params.mode == "constant_velocity":
            time = step * 0.001  # Assuming 1 fs timestep
            return initial_coordinate + self.params.pulling_velocity * time
        else:
            # For constant force mode, target doesn't change
            return initial_coordinate
    
    def calculate_smd_force(self, positions: np.ndarray, step: int,
                           initial_coordinate: float,
                           masses: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, float]:
        """
        Calculate SMD forces and work.
        
        Returns:
            forces: Force array for all atoms
            current_coordinate: Current coordinate value
            work_step: Work done in this step
        """
        current_coordinate = self.calculate_coordinate(positions, masses)
        
        if self.params.mode == "constant_velocity":
            force_magnitude, work_step = self._constant_velocity_force(
                current_coordinate, step, initial_coordinate
            )
        else:  # constant_force
            force_magnitude, work_step = self._constant_force_force(
                current_coordinate, step, initial_coordinate
            )
        
        # Calculate force direction and distribute to atoms
        forces = self._distribute_force(positions, force_magnitude, masses)
        
        # Update histories
        self.force_history.append(force_magnitude)
        self.coordinate_history.append(current_coordinate)
        self.accumulated_work += work_step
        self.work_history.append(self.accumulated_work)
        
        return forces, current_coordinate, work_step
    
    def _constant_velocity_force(self, current_coord: float, step: int,
                                initial_coord: float) -> Tuple[float, float]:
        """Calculate force for constant velocity SMD."""
        target_coord = self.calculate_target_coordinate(step, initial_coord)
        displacement = target_coord - current_coord
        
        # Harmonic restraint force
        force_magnitude = self.params.spring_constant * displacement
        
        # Work calculation: W = k * (ξ - ξ₀) * v * dt
        dt = 0.001  # ps
        work_step = force_magnitude * self.params.pulling_velocity * dt
        
        return force_magnitude, work_step
    
    def _constant_force_force(self, current_coord: float, step: int,
                             initial_coord: float) -> Tuple[float, float]:
        """Calculate force for constant force SMD."""
        # Convert pN to kJ/(mol·nm)
        force_magnitude = self.params.applied_force * 0.06022
        
        # Work calculation: W = F * Δξ
        if len(self.coordinate_history) > 0:
            delta_coord = current_coord - self.coordinate_history[-1]
            work_step = force_magnitude * delta_coord
        else:
            work_step = 0.0
        
        return force_magnitude, work_step
    
    def _distribute_force(self, positions: np.ndarray, force_magnitude: float,
                         masses: Optional[np.ndarray] = None) -> np.ndarray:
        """Distribute SMD force to atoms based on coordinate type."""
        n_atoms = len(positions)
        forces = np.zeros((n_atoms, 3))
        
        if self.params.coordinate_type == "distance":
            forces = self._distribute_distance_force(positions, force_magnitude)
        elif self.params.coordinate_type == "com_distance":
            forces = self._distribute_com_distance_force(positions, force_magnitude, masses)
        elif self.params.coordinate_type == "angle":
            forces = self._distribute_angle_force(positions, force_magnitude)
        elif self.params.coordinate_type == "dihedral":
            forces = self._distribute_dihedral_force(positions, force_magnitude)
        
        return forces
    
    def _distribute_distance_force(self, positions: np.ndarray, 
                                  force_magnitude: float) -> np.ndarray:
        """Distribute force for distance coordinate."""
        n_atoms = len(positions)
        forces = np.zeros((n_atoms, 3))
        
        atom1, atom2 = self.params.atom_indices
        direction = positions[atom2] - positions[atom1]
        direction = direction / np.linalg.norm(direction)
        
        force_vector = force_magnitude * direction
        forces[atom1] -= force_vector
        forces[atom2] += force_vector
        
        return forces
    
    def _distribute_com_distance_force(self, positions: np.ndarray,
                                      force_magnitude: float,
                                      masses: Optional[np.ndarray] = None) -> np.ndarray:
        """Distribute force for center-of-mass distance coordinate."""
        n_atoms = len(positions)
        forces = np.zeros((n_atoms, 3))
        
        # Split atoms into two groups
        mid = len(self.params.atom_indices) // 2
        group1 = self.params.atom_indices[:mid]
        group2 = self.params.atom_indices[mid:]
        
        # Calculate COMs
        if masses is not None:
            masses1 = masses[group1]
            masses2 = masses[group2]
            com1 = np.average(positions[group1], axis=0, weights=masses1)
            com2 = np.average(positions[group2], axis=0, weights=masses2)
            total_mass1 = np.sum(masses1)
            total_mass2 = np.sum(masses2)
        else:
            com1 = np.mean(positions[group1], axis=0)
            com2 = np.mean(positions[group2], axis=0)
            masses1 = np.ones(len(group1))
            masses2 = np.ones(len(group2))
            total_mass1 = len(group1)
            total_mass2 = len(group2)
        
        # Force direction
        direction = com2 - com1
        direction = direction / np.linalg.norm(direction)
        
        # Distribute forces proportionally to masses
        for i, atom_idx in enumerate(group1):
            weight = masses1[i] / total_mass1
            forces[atom_idx] -= force_magnitude * direction * weight
        
        for i, atom_idx in enumerate(group2):
            weight = masses2[i] / total_mass2
            forces[atom_idx] += force_magnitude * direction * weight
        
        return forces
    
    def _distribute_angle_force(self, positions: np.ndarray,
                               force_magnitude: float) -> np.ndarray:
        """Distribute force for angle coordinate (simplified implementation)."""
        # This is a simplified implementation
        # A full implementation would require calculating angle derivatives
        n_atoms = len(positions)
        forces = np.zeros((n_atoms, 3))
        
        # Apply force to end atoms in direction perpendicular to bonds
        atom1, atom2, atom3 = self.params.atom_indices
        
        # Simplified: apply forces to move atoms apart/together
        v1 = positions[atom1] - positions[atom2]
        v3 = positions[atom3] - positions[atom2]
        
        v1_norm = v1 / np.linalg.norm(v1)
        v3_norm = v3 / np.linalg.norm(v3)
        
        forces[atom1] += force_magnitude * v1_norm * 0.5
        forces[atom3] += force_magnitude * v3_norm * 0.5
        forces[atom2] -= force_magnitude * (v1_norm + v3_norm) * 0.5
        
        return forces
    
    def _distribute_dihedral_force(self, positions: np.ndarray,
                                  force_magnitude: float) -> np.ndarray:
        """Distribute force for dihedral coordinate (simplified implementation)."""
        # This is a simplified implementation
        # A full implementation would require calculating dihedral derivatives
        n_atoms = len(positions)
        forces = np.zeros((n_atoms, 3))
        
        # Apply torque-like forces to change dihedral angle
        atoms = self.params.atom_indices
        
        # Simplified: apply forces perpendicular to central bond
        central_bond = positions[atoms[2]] - positions[atoms[1]]
        central_bond = central_bond / np.linalg.norm(central_bond)
        
        # Create perpendicular directions
        perp1 = np.cross(central_bond, [0, 0, 1])
        if np.linalg.norm(perp1) < 0.1:
            perp1 = np.cross(central_bond, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        
        forces[atoms[0]] += force_magnitude * perp1 * 0.5
        forces[atoms[3]] -= force_magnitude * perp1 * 0.5
        
        return forces


class SteeredMD:
    """Main class for steered molecular dynamics simulations."""
    
    def __init__(self, simulation_system, parameters: SMDParameters):
        """
        Initialize steered MD simulation.
        
        Args:
            simulation_system: The MD simulation system
            parameters: SMD parameters
        """
        self.system = simulation_system
        self.params = parameters
        self.force_calculator = SMDForceCalculator(parameters)
        
        # Results storage
        self.results = {
            'coordinates': [],
            'forces': [],
            'work': [],
            'time': [],
            'positions': [],
        }
        
        # Initial coordinate value
        self.initial_coordinate = None
        
        logger.info(f"Initialized SMD with mode: {parameters.mode}")
        logger.info(f"Pulling coordinate: {parameters.coordinate_type}")
        logger.info(f"Target atoms: {parameters.atom_indices}")
    
    def run_simulation(self, n_steps: Optional[int] = None) -> Dict:
        """
        Run steered molecular dynamics simulation.
        
        Args:
            n_steps: Number of steps to run (uses parameter default if None)
            
        Returns:
            Dictionary containing simulation results
        """
        if n_steps is None:
            n_steps = self.params.n_steps
        
        logger.info(f"Starting SMD simulation for {n_steps} steps")
        
        # Get initial positions and masses
        positions = self._get_positions()
        masses = self._get_masses()
        
        # Calculate initial coordinate
        if self.initial_coordinate is None:
            self.initial_coordinate = self.force_calculator.calculate_coordinate(
                positions, masses
            )
        
        logger.info(f"Initial coordinate value: {self.initial_coordinate:.4f}")
        
        # Main simulation loop
        for step in range(n_steps):
            # Update positions from simulation system
            positions = self._get_positions()
            
            # Calculate SMD forces
            smd_forces, current_coord, work_step = self.force_calculator.calculate_smd_force(
                positions, step, self.initial_coordinate, masses
            )
            
            # Apply forces to simulation system
            self._apply_forces(smd_forces)
            
            # Advance simulation one step
            self._step_simulation()
            
            # Store results
            if step % self.params.output_frequency == 0:
                self._store_results(step, current_coord, smd_forces, work_step, positions)
            
            # Progress logging
            progress_interval = max(1, n_steps // 10)
            if step % progress_interval == 0 and step > 0:
                logger.info(f"Step {step}/{n_steps}: coord={current_coord:.4f}, "
                          f"work={self.force_calculator.accumulated_work:.2f} kJ/mol")
        
        logger.info("SMD simulation completed")
        return self.get_results()
    
    def _get_positions(self) -> np.ndarray:
        """Get current positions from simulation system."""
        if hasattr(self.system, 'positions'):
            return self.system.positions.copy()
        else:
            # Mock positions for testing
            n_atoms = len(self.params.atom_indices) * 2
            return np.random.randn(n_atoms, 3)
    
    def _get_masses(self) -> Optional[np.ndarray]:
        """Get masses from simulation system."""
        if hasattr(self.system, 'masses'):
            return self.system.masses
        else:
            # Mock masses for testing
            n_atoms = len(self.params.atom_indices) * 2
            return np.ones(n_atoms) * 12.0
    
    def _apply_forces(self, forces: np.ndarray):
        """Apply SMD forces to simulation system."""
        if hasattr(self.system, 'add_external_forces'):
            self.system.add_external_forces(forces)
        elif hasattr(self.system, 'external_forces'):
            if hasattr(self.system.external_forces, 'shape'):
                # Ensure shapes match
                if self.system.external_forces.shape == forces.shape:
                    self.system.external_forces[:] = forces
                else:
                    # Resize if needed
                    self.system.external_forces = forces.copy()
            else:
                self.system.external_forces = forces.copy()
        else:
            # For testing, just store the forces
            self._last_applied_forces = forces
    
    def _step_simulation(self):
        """Advance simulation by one step."""
        if hasattr(self.system, 'step'):
            self.system.step()
        else:
            # Mock step for testing
            pass
    
    def _store_results(self, step: int, coordinate: float, forces: np.ndarray,
                      work_step: float, positions: np.ndarray):
        """Store simulation results."""
        self.results['time'].append(step * 0.001)  # Convert to ps
        self.results['coordinates'].append(coordinate)
        self.results['forces'].append(np.linalg.norm(forces))
        self.results['work'].append(self.force_calculator.accumulated_work)
        
        if self.params.save_trajectory:
            self.results['positions'].append(positions.copy())
    
    def get_results(self) -> Dict:
        """Get simulation results."""
        results = self.results.copy()
        results.update({
            'initial_coordinate': self.initial_coordinate,
            'final_coordinate': self.force_calculator.coordinate_history[-1] if self.force_calculator.coordinate_history else None,
            'total_work': self.force_calculator.accumulated_work,
            'parameters': self.params,
            'force_history': self.force_calculator.force_history.copy(),
            'coordinate_history': self.force_calculator.coordinate_history.copy(),
            'work_history': self.force_calculator.work_history.copy(),
        })
        return results
    
    def calculate_jarzynski_free_energy(self, temperature: float = 300.0) -> float:
        """
        Calculate free energy difference using Jarzynski equality.
        
        ΔG = -kT ln⟨exp(-W/kT)⟩
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Free energy difference in kJ/mol
        """
        if not self.force_calculator.work_history:
            raise ValueError("No work data available for Jarzynski calculation")
        
        kb = 0.008314  # kJ/(mol·K)
        beta = 1.0 / (kb * temperature)
        
        work_total = self.force_calculator.accumulated_work
        exp_term = np.exp(-beta * work_total)
        
        # For single trajectory, this is just the exponential
        # For multiple trajectories, would need ensemble average
        delta_g = -kb * temperature * np.log(exp_term)
        
        logger.info(f"Jarzynski free energy estimate: {delta_g:.2f} kJ/mol")
        return delta_g
    
    def plot_force_curves(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot force curves and work profiles.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if not self.results['time']:
            raise ValueError("No simulation data to plot")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        time_data = np.array(self.results['time'])
        
        # Create time arrays for force history (stored at every step)
        force_time = np.arange(len(self.force_calculator.force_history)) * 0.001  # Convert to ps
        
        # Coordinate vs time
        ax1.plot(time_data, self.results['coordinates'], 'b-', linewidth=2)
        ax1.set_xlabel('Time (ps)')
        ax1.set_ylabel(f'{self.params.coordinate_type.capitalize()} (nm)')
        ax1.set_title('Pulling Coordinate')
        ax1.grid(True, alpha=0.3)
        
        # Force vs time
        ax2.plot(force_time, self.force_calculator.force_history, 'r-', linewidth=2)
        ax2.set_xlabel('Time (ps)')
        ax2.set_ylabel('Force (kJ/(mol·nm))')
        ax2.set_title('Applied Force')
        ax2.grid(True, alpha=0.3)
        
        # Work vs time
        ax3.plot(time_data, self.results['work'], 'g-', linewidth=2)
        ax3.set_xlabel('Time (ps)')
        ax3.set_ylabel('Accumulated Work (kJ/mol)')
        ax3.set_title('Work Profile')
        ax3.grid(True, alpha=0.3)
        
        # Force vs coordinate (use coordinate history that matches force history)
        coord_history = self.force_calculator.coordinate_history
        ax4.plot(coord_history, self.force_calculator.force_history, 'purple', linewidth=2)
        ax4.set_xlabel(f'{self.params.coordinate_type.capitalize()} (nm)')
        ax4.set_ylabel('Force (kJ/(mol·nm))')
        ax4.set_title('Force vs Coordinate')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Force curves saved to {save_path}")
        
        return fig
    
    def save_results(self, output_dir: str):
        """Save simulation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save parameters
        params_dict = {
            'atom_indices': self.params.atom_indices,
            'coordinate_type': self.params.coordinate_type,
            'mode': self.params.mode,
            'pulling_velocity': self.params.pulling_velocity,
            'spring_constant': self.params.spring_constant,
            'applied_force': self.params.applied_force,
            'n_steps': self.params.n_steps,
            'output_frequency': self.params.output_frequency,
        }
        
        with open(output_path / 'smd_parameters.json', 'w') as f:
            json.dump(params_dict, f, indent=2)
        
        # Save time series data
        results_data = {
            'time': self.results['time'],
            'coordinates': self.results['coordinates'],
            'forces': self.force_calculator.force_history,
            'work': self.results['work'],
            'initial_coordinate': self.initial_coordinate,
            'total_work': self.force_calculator.accumulated_work,
        }
        
        np.savez(output_path / 'smd_results.npz', **results_data)
        
        # Save force curves plot
        if self.params.save_force_curves:
            self.plot_force_curves(str(output_path / 'force_curves.png'))
        
        logger.info(f"Results saved to {output_path}")


# Convenience functions for common SMD setups

def setup_protein_unfolding_smd(simulation_system, 
                                n_terminus_atoms: List[int],
                                c_terminus_atoms: List[int],
                                pulling_velocity: float = 0.005,
                                spring_constant: float = 1000.0) -> SteeredMD:
    """
    Set up SMD simulation for protein unfolding.
    
    Args:
        simulation_system: MD simulation system
        n_terminus_atoms: Atom indices for N-terminus group
        c_terminus_atoms: Atom indices for C-terminus group  
        pulling_velocity: Pulling velocity in nm/ps
        spring_constant: Spring constant in kJ/(mol·nm²)
        
    Returns:
        Configured SteeredMD object
    """
    atom_indices = n_terminus_atoms + c_terminus_atoms
    
    params = SMDParameters(
        atom_indices=atom_indices,
        coordinate_type="com_distance",
        mode="constant_velocity",
        pulling_velocity=pulling_velocity,
        spring_constant=spring_constant,
        n_steps=50000,  # Long simulation for unfolding
        output_frequency=100
    )
    
    return SteeredMD(simulation_system, params)


def setup_ligand_unbinding_smd(simulation_system,
                              ligand_atoms: List[int],
                              protein_atoms: List[int],
                              pulling_velocity: float = 0.01,
                              spring_constant: float = 500.0) -> SteeredMD:
    """
    Set up SMD simulation for ligand unbinding.
    
    Args:
        simulation_system: MD simulation system
        ligand_atoms: Atom indices for ligand
        protein_atoms: Atom indices for binding site
        pulling_velocity: Pulling velocity in nm/ps
        spring_constant: Spring constant in kJ/(mol·nm²)
        
    Returns:
        Configured SteeredMD object
    """
    atom_indices = ligand_atoms + protein_atoms
    
    params = SMDParameters(
        atom_indices=atom_indices,
        coordinate_type="com_distance",
        mode="constant_velocity",
        pulling_velocity=pulling_velocity,
        spring_constant=spring_constant,
        n_steps=20000,
        output_frequency=50
    )
    
    return SteeredMD(simulation_system, params)


def setup_bond_stretching_smd(simulation_system,
                             atom1: int,
                             atom2: int,
                             applied_force: float = 500.0) -> SteeredMD:
    """
    Set up SMD simulation for bond stretching with constant force.
    
    Args:
        simulation_system: MD simulation system
        atom1: First atom index
        atom2: Second atom index
        applied_force: Applied force in pN
        
    Returns:
        Configured SteeredMD object
    """
    params = SMDParameters(
        atom_indices=[atom1, atom2],
        coordinate_type="distance",
        mode="constant_force",
        applied_force=applied_force,
        n_steps=10000,
        output_frequency=10
    )
    
    return SteeredMD(simulation_system, params)


if __name__ == "__main__":
    # Example usage and testing
    print("Steered Molecular Dynamics Module")
    print("=================================")
    
    # Create mock simulation system for testing
    class MockSimulationSystem:
        def __init__(self):
            self.positions = np.random.randn(20, 3) * 2.0
            self.masses = np.ones(20) * 12.0
            self.external_forces = np.zeros((20, 3))
            
        def step(self):
            # Mock dynamics - just add some random motion
            self.positions += np.random.randn(*self.positions.shape) * 0.01
    
    # Test distance pulling
    mock_system = MockSimulationSystem()
    
    params = SMDParameters(
        atom_indices=[0, 10],
        coordinate_type="distance",
        mode="constant_velocity",
        pulling_velocity=0.01,
        spring_constant=1000.0,
        n_steps=100,
        output_frequency=10
    )
    
    smd = SteeredMD(mock_system, params)
    results = smd.run_simulation()
    
    print(f"Initial coordinate: {results['initial_coordinate']:.4f} nm")
    print(f"Final coordinate: {results['final_coordinate']:.4f} nm")
    print(f"Total work: {results['total_work']:.2f} kJ/mol")
    
    # Test Jarzynski calculation
    try:
        delta_g = smd.calculate_jarzynski_free_energy()
        print(f"Jarzynski ΔG estimate: {delta_g:.2f} kJ/mol")
    except Exception as e:
        print(f"Jarzynski calculation note: {e}")
    
    print("\nSMD module loaded successfully!")
