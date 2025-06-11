"""
Metadynamics implementation for enhanced sampling.

This module provides metadynamics simulation capabilities with collective variables,
adaptive Gaussian hills, free energy profile reconstruction, and well-tempered
metadynamics variants.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from scipy import interpolate
from scipy.optimize import minimize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetadynamicsParameters:
    """Parameters for metadynamics simulation."""
    height: float = 0.5  # kJ/mol - height of Gaussian hills
    width: float = 0.1   # width (sigma) of Gaussian hills
    deposition_interval: int = 500  # steps between hill depositions
    temperature: float = 300.0  # K
    bias_factor: float = 10.0  # γ for well-tempered metadynamics (γ=1 is standard)
    max_hills: int = 10000  # maximum number of hills
    convergence_threshold: float = 0.1  # kJ/mol - for convergence detection
    convergence_window: int = 100  # number of recent hills to check for convergence


class CollectiveVariable(ABC):
    """Abstract base class for collective variables."""
    
    def __init__(self, name: str):
        self.name = name
        self.history: List[float] = []
        
    @abstractmethod
    def calculate(self, positions: np.ndarray, box: Optional[np.ndarray] = None) -> float:
        """Calculate the collective variable value."""
        pass
        
    @abstractmethod
    def calculate_gradient(self, positions: np.ndarray, box: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate the gradient of the CV with respect to positions."""
        pass
        
    def reset_history(self):
        """Reset the history of CV values."""
        self.history = []


class DistanceCV(CollectiveVariable):
    """Distance collective variable between two atoms or groups."""
    
    def __init__(self, name: str, atom_indices_1: Union[int, List[int]], 
                 atom_indices_2: Union[int, List[int]]):
        super().__init__(name)
        self.atom_indices_1 = [atom_indices_1] if isinstance(atom_indices_1, int) else atom_indices_1
        self.atom_indices_2 = [atom_indices_2] if isinstance(atom_indices_2, int) else atom_indices_2
        
    def _get_center_of_mass(self, positions: np.ndarray, indices: List[int]) -> np.ndarray:
        """Calculate center of mass for a group of atoms (assuming equal masses)."""
        return np.mean(positions[indices], axis=0)
        
    def calculate(self, positions: np.ndarray, box: Optional[np.ndarray] = None) -> float:
        """Calculate distance between two groups."""
        com1 = self._get_center_of_mass(positions, self.atom_indices_1)
        com2 = self._get_center_of_mass(positions, self.atom_indices_2)
        
        diff = com2 - com1
        
        # Apply periodic boundary conditions if box is provided
        if box is not None and isinstance(box, np.ndarray):
            diff = diff - np.round(diff / box) * box
            
        distance = np.linalg.norm(diff)
        self.history.append(distance)
        return distance
        
    def calculate_gradient(self, positions: np.ndarray, box: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate gradient of distance CV."""
        com1 = self._get_center_of_mass(positions, self.atom_indices_1)
        com2 = self._get_center_of_mass(positions, self.atom_indices_2)
        
        diff = com2 - com1
        
        # Apply periodic boundary conditions if box is provided
        if box is not None and isinstance(box, np.ndarray):
            diff = diff - np.round(diff / box) * box
            
        distance = np.linalg.norm(diff)
        
        if distance < 1e-10:
            return np.zeros_like(positions)
            
        # Gradient for distance: d|r|/dr = r/|r|
        unit_vector = diff / distance
        
        gradient = np.zeros_like(positions)
        
        # Gradient for group 1 (negative)
        for i in self.atom_indices_1:
            gradient[i] = -unit_vector / len(self.atom_indices_1)
            
        # Gradient for group 2 (positive)
        for i in self.atom_indices_2:
            gradient[i] = unit_vector / len(self.atom_indices_2)
            
        return gradient


class AngleCV(CollectiveVariable):
    """Angle collective variable between three atoms or groups."""
    
    def __init__(self, name: str, atom_indices_1: Union[int, List[int]], 
                 atom_indices_2: Union[int, List[int]], 
                 atom_indices_3: Union[int, List[int]]):
        super().__init__(name)
        self.atom_indices_1 = [atom_indices_1] if isinstance(atom_indices_1, int) else atom_indices_1
        self.atom_indices_2 = [atom_indices_2] if isinstance(atom_indices_2, int) else atom_indices_2
        self.atom_indices_3 = [atom_indices_3] if isinstance(atom_indices_3, int) else atom_indices_3
        
    def _get_center_of_mass(self, positions: np.ndarray, indices: List[int]) -> np.ndarray:
        """Calculate center of mass for a group of atoms (assuming equal masses)."""
        return np.mean(positions[indices], axis=0)
        
    def calculate(self, positions: np.ndarray, box: Optional[np.ndarray] = None) -> float:
        """Calculate angle between three groups."""
        com1 = self._get_center_of_mass(positions, self.atom_indices_1)
        com2 = self._get_center_of_mass(positions, self.atom_indices_2)
        com3 = self._get_center_of_mass(positions, self.atom_indices_3)
        
        # Vectors from center atom to the other two
        vec1 = com1 - com2
        vec2 = com3 - com2
        
        # Apply periodic boundary conditions if box is provided
        if box is not None and isinstance(box, np.ndarray):
            vec1 = vec1 - np.round(vec1 / box) * box
            vec2 = vec2 - np.round(vec2 / box) * box
            
        # Calculate angle
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            angle = 0.0
        else:
            cos_angle = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
        self.history.append(angle)
        return angle
        
    def calculate_gradient(self, positions: np.ndarray, box: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate gradient of angle CV."""
        com1 = self._get_center_of_mass(positions, self.atom_indices_1)
        com2 = self._get_center_of_mass(positions, self.atom_indices_2)
        com3 = self._get_center_of_mass(positions, self.atom_indices_3)
        
        # Vectors from center atom to the other two
        vec1 = com1 - com2
        vec2 = com3 - com2
        
        # Apply periodic boundary conditions if box is provided
        if box is not None and isinstance(box, np.ndarray):
            vec1 = vec1 - np.round(vec1 / box) * box
            vec2 = vec2 - np.round(vec2 / box) * box
            
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        gradient = np.zeros_like(positions)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return gradient
            
        cos_angle = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
        sin_angle = np.sqrt(1.0 - cos_angle**2)
        
        if sin_angle < 1e-10:
            return gradient
            
        # Gradient calculation for angle
        # d(angle)/dr = -1/sin(angle) * [d(cos_angle)/dr]
        
        # Terms for gradient of cos_angle
        term1 = vec2 / (norm1 * norm2)
        term2 = vec1 / (norm1 * norm2)
        term3 = cos_angle * vec1 / norm1**2
        term4 = cos_angle * vec2 / norm2**2
        
        grad_cos_1 = term1 - term3  # gradient w.r.t. com1
        grad_cos_2 = -(term1 + term2) + term3 + term4  # gradient w.r.t. com2
        grad_cos_3 = term2 - term4  # gradient w.r.t. com3
        
        factor = -1.0 / sin_angle
        
        # Distribute gradients to atoms
        for i in self.atom_indices_1:
            gradient[i] = factor * grad_cos_1 / len(self.atom_indices_1)
        for i in self.atom_indices_2:
            gradient[i] = factor * grad_cos_2 / len(self.atom_indices_2)
        for i in self.atom_indices_3:
            gradient[i] = factor * grad_cos_3 / len(self.atom_indices_3)
            
        return gradient


@dataclass
class GaussianHill:
    """Represents a single Gaussian hill in metadynamics."""
    position: np.ndarray  # Position in CV space
    height: float  # Height of the hill
    width: float   # Width (sigma) of the hill
    deposition_time: int  # Time step when hill was deposited
    
    def evaluate(self, cv_values: np.ndarray) -> float:
        """Evaluate the Gaussian hill at given CV values."""
        diff = cv_values - self.position
        exponent = -0.5 * np.sum((diff / self.width)**2)
        return self.height * np.exp(exponent)
        
    def evaluate_gradient(self, cv_values: np.ndarray) -> np.ndarray:
        """Evaluate the gradient of the Gaussian hill."""
        diff = cv_values - self.position
        exponent = -0.5 * np.sum((diff / self.width)**2)
        gaussian_value = self.height * np.exp(exponent)
        return -gaussian_value * diff / (self.width**2)


class MetadynamicsSimulation:
    """Main metadynamics simulation class."""
    
    def __init__(self, 
                 collective_variables: List[CollectiveVariable],
                 parameters: MetadynamicsParameters,
                 system: Any):
        """
        Initialize metadynamics simulation.
        
        Args:
            collective_variables: List of collective variables to bias
            parameters: Metadynamics parameters
            system: MD system to simulate
        """
        self.cvs = collective_variables
        self.params = parameters
        self.system = system
        
        # Initialize hills and history
        self.hills: List[GaussianHill] = []
        self.cv_history: List[np.ndarray] = []
        self.bias_energy_history: List[float] = []
        self.step_count = 0
        self.last_deposition_step = 0
        
        # Well-tempered metadynamics
        self.is_well_tempered = parameters.bias_factor > 1.0
        self.kT = 8.314e-3 * parameters.temperature  # kJ/mol
        
        # Convergence detection
        self.convergence_detected = False
        self.convergence_step = None
        
        # Free energy surface
        self.fes_grid: Optional[np.ndarray] = None
        self.fes_values: Optional[np.ndarray] = None
        
        logger.info(f"Initialized metadynamics with {len(self.cvs)} CVs")
        if self.is_well_tempered:
            logger.info(f"Using well-tempered metadynamics with γ = {parameters.bias_factor}")
    
    def get_current_cv_values(self) -> np.ndarray:
        """Get current values of all collective variables."""
        positions = getattr(self.system, 'positions', np.random.randn(10, 3))
        box = getattr(self.system, 'box', None)
        
        cv_values = np.array([cv.calculate(positions, box) for cv in self.cvs])
        return cv_values
    
    def calculate_bias_potential(self, cv_values: Optional[np.ndarray] = None) -> float:
        """Calculate the bias potential at given CV values."""
        if cv_values is None:
            cv_values = self.get_current_cv_values()
        
        bias = 0.0
        for hill in self.hills:
            bias += hill.evaluate(cv_values)
        
        return bias
    
    def calculate_bias_forces(self, cv_values: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate bias forces on atoms due to metadynamics potential."""
        if cv_values is None:
            cv_values = self.get_current_cv_values()
        
        positions = getattr(self.system, 'positions', np.random.randn(10, 3))
        box = getattr(self.system, 'box', None)
        
        # Initialize forces array
        forces = np.zeros_like(positions)
        
        # Calculate gradient of bias potential with respect to CVs
        bias_gradient_cv = np.zeros(len(self.cvs))
        for hill in self.hills:
            bias_gradient_cv += hill.evaluate_gradient(cv_values)
        
        # Chain rule: F = -∂V_bias/∂r = -∂V_bias/∂CV * ∂CV/∂r
        for i, cv in enumerate(self.cvs):
            cv_gradient = cv.calculate_gradient(positions, box)
            forces -= bias_gradient_cv[i] * cv_gradient
        
        return forces
    
    def deposit_hill(self) -> None:
        """Deposit a new Gaussian hill at current CV position."""
        cv_values = self.get_current_cv_values()
        
        # Adjust hill height for well-tempered metadynamics
        if self.is_well_tempered:
            current_bias = self.calculate_bias_potential(cv_values)
            height = self.params.height * np.exp(-current_bias / (self.kT * (self.params.bias_factor - 1)))
        else:
            height = self.params.height
        
        # Create and add hill
        hill = GaussianHill(
            position=cv_values.copy(),
            height=height,
            width=self.params.width,
            deposition_time=self.step_count
        )
        
        self.hills.append(hill)
        self.last_deposition_step = self.step_count
        
        logger.debug(f"Deposited hill {len(self.hills)} at CV = {cv_values}, height = {height:.3f}")
        
        # Check for convergence
        if len(self.hills) >= self.params.convergence_window:
            self.check_convergence()
    
    def check_convergence(self) -> None:
        """Check if the free energy surface has converged."""
        if self.convergence_detected or len(self.hills) < self.params.convergence_window:
            return
        
        # Look at recent hill heights
        recent_hills = self.hills[-self.params.convergence_window:]
        recent_heights = [hill.height for hill in recent_hills]
        
        # Check if heights have become small and stable
        mean_height = np.mean(recent_heights)
        std_height = np.std(recent_heights)
        
        if mean_height < self.params.convergence_threshold and std_height < self.params.convergence_threshold:
            self.convergence_detected = True
            self.convergence_step = self.step_count
            logger.info(f"Metadynamics convergence detected at step {self.step_count}")
    
    def step(self) -> None:
        """Perform one metadynamics step."""
        self.step_count += 1
        
        # Record current CV values
        cv_values = self.get_current_cv_values()
        self.cv_history.append(cv_values.copy())
        
        # Record bias energy
        bias_energy = self.calculate_bias_potential(cv_values)
        self.bias_energy_history.append(bias_energy)
        
        # Apply bias forces to system
        if hasattr(self.system, 'external_forces'):
            bias_forces = self.calculate_bias_forces(cv_values)
            
            # Add bias forces to system
            if hasattr(self.system.external_forces, 'shape') and self.system.external_forces.shape == bias_forces.shape:
                self.system.external_forces += bias_forces
            else:
                # Initialize external forces if needed
                self.system.external_forces = bias_forces
        
        # Deposit hill if it's time
        steps_since_deposition = self.step_count - self.last_deposition_step
        if steps_since_deposition >= self.params.deposition_interval:
            if len(self.hills) < self.params.max_hills:
                self.deposit_hill()
            else:
                logger.warning(f"Maximum number of hills ({self.params.max_hills}) reached")
        
        # Perform MD step on the system
        if hasattr(self.system, 'step'):
            self.system.step()
    
    def run(self, n_steps: int) -> None:
        """Run metadynamics simulation for specified number of steps."""
        logger.info(f"Starting metadynamics simulation for {n_steps} steps")
        
        start_step = self.step_count
        for i in range(n_steps):
            self.step()
            
            # Progress reporting
            if (self.step_count - start_step) % max(1, n_steps // 10) == 0 and i > 0:
                progress = (i + 1) / n_steps * 100
                logger.info(f"Metadynamics progress: {progress:.1f}% ({len(self.hills)} hills deposited)")
        
        logger.info(f"Metadynamics simulation completed. Total hills: {len(self.hills)}")
        if self.convergence_detected:
            logger.info(f"Convergence achieved at step {self.convergence_step}")
    
    def calculate_free_energy_surface(self, 
                                     grid_points: Union[int, List[int]] = 50,
                                     cv_ranges: Optional[List[Tuple[float, float]]] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Calculate the free energy surface from deposited hills.
        
        Args:
            grid_points: Number of grid points for each CV (int for all, list for individual)
            cv_ranges: Ranges for each CV as [(min, max), ...]. If None, use data range.
        
        Returns:
            tuple: (grid_coordinates, free_energy_values)
        """
        if not self.cv_history:
            raise ValueError("No CV history available. Run simulation first.")
        
        # Convert to numpy array for easier handling
        cv_array = np.array(self.cv_history)
        n_cvs = len(self.cvs)
        
        # Set up grid points
        if isinstance(grid_points, int):
            grid_points = [grid_points] * n_cvs
        
        # Set up CV ranges
        if cv_ranges is None:
            cv_ranges = []
            for i in range(n_cvs):
                cv_min = np.min(cv_array[:, i])
                cv_max = np.max(cv_array[:, i])
                # Add some padding
                padding = (cv_max - cv_min) * 0.1
                cv_ranges.append((cv_min - padding, cv_max + padding))
        
        # Create grid
        grid_1d = []
        for i in range(n_cvs):
            grid_1d.append(np.linspace(cv_ranges[i][0], cv_ranges[i][1], grid_points[i]))
        
        # Create meshgrid
        grid_coords = np.meshgrid(*grid_1d, indexing='ij')
        grid_shape = grid_coords[0].shape
        
        # Flatten for easier calculation
        grid_flat = np.array([coord.flatten() for coord in grid_coords]).T
        
        # Calculate free energy at each grid point
        fes_flat = np.zeros(len(grid_flat))
        
        for i, point in enumerate(grid_flat):
            # Sum all Gaussian hills at this point
            bias = 0.0
            for hill in self.hills:
                bias += hill.evaluate(point)
            # Free energy is negative bias
            fes_flat[i] = -bias
        
        # Reshape back to grid
        fes_values = fes_flat.reshape(grid_shape)
        
        # Subtract minimum for relative free energy
        fes_values -= np.min(fes_values)
        
        # Store for later use
        self.fes_grid = grid_coords
        self.fes_values = fes_values
        
        logger.info(f"Calculated FES on {np.prod(grid_shape)} grid points")
        
        return grid_coords, fes_values
    
    def plot_results(self, save_prefix: Optional[str] = None) -> None:
        """Plot metadynamics results including CV evolution and free energy surface."""
        n_cvs = len(self.cvs)
        
        if n_cvs == 1:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        elif n_cvs == 2:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        else:
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        fig.suptitle('Metadynamics Simulation Results', fontsize=16)
        
        # Convert CV history to array
        if self.cv_history:
            cv_array = np.array(self.cv_history)
            time_axis = np.arange(len(self.cv_history)) * 0.001  # assuming 1 fs timestep
        
        # Plot 1: CV evolution over time
        ax1 = axes[0, 0] if n_cvs <= 2 else axes[0, 0]
        if self.cv_history:
            for i, cv in enumerate(self.cvs):
                ax1.plot(time_axis, cv_array[:, i], label=f'{cv.name}', alpha=0.7)
            ax1.set_xlabel('Time (ps)')
            ax1.set_ylabel('CV Value')
            ax1.set_title('Collective Variable Evolution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bias energy evolution
        ax2 = axes[0, 1] if n_cvs <= 2 else axes[0, 1]
        if self.bias_energy_history:
            ax2.plot(time_axis, self.bias_energy_history, 'b-', alpha=0.7)
            ax2.set_xlabel('Time (ps)')
            ax2.set_ylabel('Bias Energy (kJ/mol)')
            ax2.set_title('Bias Energy Evolution')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Hill heights over time
        ax3 = axes[1, 0] if n_cvs <= 2 else axes[0, 2]
        if self.hills:
            hill_times = [hill.deposition_time * 0.001 for hill in self.hills]
            hill_heights = [hill.height for hill in self.hills]
            ax3.plot(hill_times, hill_heights, 'ro-', markersize=3, alpha=0.7)
            ax3.set_xlabel('Time (ps)')
            ax3.set_ylabel('Hill Height (kJ/mol)')
            ax3.set_title('Hill Heights vs Time')
            ax3.grid(True, alpha=0.3)
            
            if self.convergence_detected:
                ax3.axvline(self.convergence_step * 0.001, color='green', 
                           linestyle='--', label='Convergence')
                ax3.legend()
        
        # Plot 4: Free energy surface
        if n_cvs == 1:
            ax4 = axes[1, 1]
            if self.fes_grid is not None and self.fes_values is not None:
                ax4.plot(self.fes_grid[0], self.fes_values, 'b-', linewidth=2)
                ax4.set_xlabel(f'{self.cvs[0].name}')
                ax4.set_ylabel('Free Energy (kJ/mol)')
                ax4.set_title('Free Energy Profile')
                ax4.grid(True, alpha=0.3)
            else:
                # Calculate FES if not done yet
                try:
                    grid_coords, fes_values = self.calculate_free_energy_surface()
                    ax4.plot(grid_coords[0], fes_values, 'b-', linewidth=2)
                    ax4.set_xlabel(f'{self.cvs[0].name}')
                    ax4.set_ylabel('Free Energy (kJ/mol)')
                    ax4.set_title('Free Energy Profile')
                    ax4.grid(True, alpha=0.3)
                except Exception as e:
                    ax4.text(0.5, 0.5, f'Error calculating FES:\n{str(e)}', 
                            transform=ax4.transAxes, ha='center', va='center')
        
        elif n_cvs == 2:
            ax4 = axes[1, 1]
            ax5 = axes[1, 2]
            
            # 2D free energy surface
            if self.fes_grid is not None and self.fes_values is not None:
                contour = ax4.contourf(self.fes_grid[0], self.fes_grid[1], self.fes_values, 
                                      levels=20, cmap='viridis')
                fig.colorbar(contour, ax=ax4, label='Free Energy (kJ/mol)')
                ax4.set_xlabel(f'{self.cvs[0].name}')
                ax4.set_ylabel(f'{self.cvs[1].name}')
                ax4.set_title('Free Energy Surface')
                
                # Overlay CV trajectory
                if self.cv_history:
                    ax4.plot(cv_array[:, 0], cv_array[:, 1], 'w-', alpha=0.6, linewidth=1)
                    ax4.scatter(cv_array[::100, 0], cv_array[::100, 1], c='white', s=10, alpha=0.8)
            else:
                try:
                    grid_coords, fes_values = self.calculate_free_energy_surface()
                    contour = ax4.contourf(grid_coords[0], grid_coords[1], fes_values, 
                                          levels=20, cmap='viridis')
                    fig.colorbar(contour, ax=ax4, label='Free Energy (kJ/mol)')
                    ax4.set_xlabel(f'{self.cvs[0].name}')
                    ax4.set_ylabel(f'{self.cvs[1].name}')
                    ax4.set_title('Free Energy Surface')
                    
                    if self.cv_history:
                        ax4.plot(cv_array[:, 0], cv_array[:, 1], 'w-', alpha=0.6, linewidth=1)
                        ax4.scatter(cv_array[::100, 0], cv_array[::100, 1], c='white', s=10, alpha=0.8)
                except Exception as e:
                    ax4.text(0.5, 0.5, f'Error calculating FES:\n{str(e)}', 
                            transform=ax4.transAxes, ha='center', va='center')
            
            # CV correlation
            if self.cv_history and len(cv_array) > 1:
                ax5.scatter(cv_array[:, 0], cv_array[:, 1], c=time_axis, 
                           cmap='plasma', alpha=0.6, s=10)
                ax5.set_xlabel(f'{self.cvs[0].name}')
                ax5.set_ylabel(f'{self.cvs[1].name}')
                ax5.set_title('CV Trajectory')
                cbar = fig.colorbar(ax5.collections[0], ax=ax5, label='Time (ps)')
        
        plt.tight_layout()
        
        if save_prefix:
            plt.savefig(f'{save_prefix}_metadynamics_results.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_prefix}_metadynamics_results.png")
        
        plt.show()
    
    def save_hills(self, filename: str) -> None:
        """Save deposited hills to file."""
        data = []
        for i, hill in enumerate(self.hills):
            row = [i, hill.deposition_time] + hill.position.tolist() + [hill.height, hill.width]
            data.append(row)
        
        header = ['hill_id', 'time'] + [f'cv_{i}' for i in range(len(self.cvs))] + ['height', 'width']
        
        np.savetxt(filename, data, header=' '.join(header), fmt='%.6f')
        logger.info(f"Saved {len(self.hills)} hills to {filename}")
    
    def load_hills(self, filename: str) -> None:
        """Load hills from file."""
        data = np.loadtxt(filename)
        
        self.hills = []
        for row in data:
            hill_id = int(row[0])
            deposition_time = int(row[1])
            position = row[2:2+len(self.cvs)]
            height = row[2+len(self.cvs)]
            width = row[3+len(self.cvs)]
            
            hill = GaussianHill(
                position=position,
                height=height,
                width=width,
                deposition_time=deposition_time
            )
            self.hills.append(hill)
        
        logger.info(f"Loaded {len(self.hills)} hills from {filename}")


# Convenience functions for common setups

def setup_distance_metadynamics(system: Any, 
                               atom_pairs: List[Tuple[int, int]],
                               height: float = 0.5,
                               width: float = 0.1,
                               bias_factor: float = 10.0) -> MetadynamicsSimulation:
    """
    Set up metadynamics simulation with distance collective variables.
    
    Args:
        system: MD system
        atom_pairs: List of (atom1_index, atom2_index) pairs
        height: Hill height in kJ/mol
        width: Hill width 
        bias_factor: Well-tempered bias factor (γ)
    
    Returns:
        MetadynamicsSimulation instance
    """
    cvs = []
    for i, (atom1, atom2) in enumerate(atom_pairs):
        cv = DistanceCV(f"distance_{i+1}", atom1, atom2)
        cvs.append(cv)
    
    params = MetadynamicsParameters(
        height=height,
        width=width,
        bias_factor=bias_factor
    )
    
    return MetadynamicsSimulation(cvs, params, system)


def setup_angle_metadynamics(system: Any,
                            atom_triplets: List[Tuple[int, int, int]],
                            height: float = 0.5,
                            width: float = 0.1,
                            bias_factor: float = 10.0) -> MetadynamicsSimulation:
    """
    Set up metadynamics simulation with angle collective variables.
    
    Args:
        system: MD system
        atom_triplets: List of (atom1, atom2, atom3) triplets for angles
        height: Hill height in kJ/mol
        width: Hill width in radians
        bias_factor: Well-tempered bias factor (γ)
    
    Returns:
        MetadynamicsSimulation instance
    """
    cvs = []
    for i, (atom1, atom2, atom3) in enumerate(atom_triplets):
        cv = AngleCV(f"angle_{i+1}", atom1, atom2, atom3)
        cvs.append(cv)
    
    params = MetadynamicsParameters(
        height=height,
        width=width,
        bias_factor=bias_factor
    )
    
    return MetadynamicsSimulation(cvs, params, system)


def setup_protein_folding_metadynamics(system: Any,
                                      backbone_atoms: Optional[List[int]] = None,
                                      height: float = 0.5,
                                      bias_factor: float = 15.0) -> MetadynamicsSimulation:
    """
    Set up metadynamics for protein folding with radius of gyration and end-to-end distance.
    
    Args:
        system: MD system with protein
        backbone_atoms: List of backbone atom indices, if None uses first 100 atoms
        height: Hill height in kJ/mol
        bias_factor: Well-tempered bias factor
    
    Returns:
        MetadynamicsSimulation instance
    """
    if backbone_atoms is None:
        # Use first 100 atoms as approximation
        backbone_atoms = list(range(min(100, getattr(system, 'n_atoms', 100))))
    
    cvs = []
    
    # End-to-end distance
    if len(backbone_atoms) >= 2:
        cv1 = DistanceCV("end_to_end", backbone_atoms[0], backbone_atoms[-1])
        cvs.append(cv1)
    
    # Radius of gyration approximation using multiple distances
    if len(backbone_atoms) >= 4:
        # Use distances between quartiles as approximation for Rg
        n = len(backbone_atoms)
        cv2 = DistanceCV("rg_proxy", backbone_atoms[:n//4], backbone_atoms[3*n//4:])
        cvs.append(cv2)
    
    params = MetadynamicsParameters(
        height=height,
        width=0.2,  # Larger width for protein folding
        bias_factor=bias_factor,
        deposition_interval=1000  # Less frequent for protein folding
    )
    
    return MetadynamicsSimulation(cvs, params, system)
