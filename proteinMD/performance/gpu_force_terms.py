"""
GPU-accelerated force field terms for molecular dynamics simulations.

This module provides GPU-accelerated versions of force field terms that
integrate seamlessly with the existing force field architecture, providing
automatic fallback to CPU implementations when GPU is not available.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from ..forcefield.forcefield import LennardJonesForceTerm, CoulombForceTerm, ForceTerm
from .gpu_acceleration import LennardJonesGPU, CoulombGPU, GPUBackend, is_gpu_available

logger = logging.getLogger(__name__)

class GPULennardJonesForceTerm(LennardJonesForceTerm):
    """
    GPU-accelerated Lennard-Jones force term.
    
    This class extends the standard Lennard-Jones force term with GPU acceleration
    while maintaining full compatibility with the existing force field interface.
    Automatically falls back to CPU implementation when GPU is not available.
    """
    
    def __init__(self, 
                 sigma=None, 
                 epsilon=None, 
                 cutoff: float = 1.0, 
                 name: str = "LennardJones_GPU",
                 switch_distance: Optional[float] = None,
                 gpu_backend: Optional[GPUBackend] = None,
                 device_id: int = 0,
                 force_cpu: bool = False):
        """
        Initialize GPU-accelerated Lennard-Jones force term.
        
        Parameters
        ----------
        sigma : array-like, optional
            Distance parameters in nm for each particle
        epsilon : array-like, optional
            Energy parameters in kJ/mol for each particle
        cutoff : float, optional
            Cutoff distance in nm
        name : str, optional
            Name of the force term
        switch_distance : float, optional
            Distance at which to start switching the potential
        gpu_backend : GPUBackend, optional
            Preferred GPU backend
        device_id : int, optional
            GPU device ID to use
        force_cpu : bool, optional
            Force CPU execution (for testing/debugging)
        """
        # Initialize parent class
        super().__init__(sigma, epsilon, cutoff, name, switch_distance)
        
        # Initialize GPU acceleration
        self.force_cpu = force_cpu
        self.gpu_calculator = None
        self.gpu_enabled = False
        
        if not force_cpu and is_gpu_available():
            try:
                self.gpu_calculator = LennardJonesGPU(gpu_backend, device_id, cutoff)
                self.gpu_enabled = self.gpu_calculator.is_gpu_enabled()
                if self.gpu_enabled:
                    logger.info(f"GPU acceleration enabled for {name} using {self.gpu_calculator.backend.value}")
                else:
                    logger.info(f"GPU acceleration not available for {name}, using CPU fallback")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU acceleration for {name}: {e}")
                self.gpu_enabled = False
        else:
            if force_cpu:
                logger.info(f"GPU acceleration disabled for {name} (forced CPU mode)")
            else:
                logger.info(f"GPU acceleration not available for {name}")
    
    def calculate(self, positions, box_vectors=None):
        """
        Calculate Lennard-Jones forces with automatic GPU/CPU selection.
        
        Parameters
        ----------
        positions : np.ndarray
            Particle positions with shape (n_particles, 3)
        box_vectors : np.ndarray, optional
            Periodic box vectors
            
        Returns
        -------
        tuple
            Tuple of (forces, potential_energy)
        """
        n_particles = positions.shape[0]
        
        # Use GPU acceleration for large systems if available
        if self.gpu_enabled and n_particles >= 100:  # GPU worthwhile for >=100 particles
            try:
                return self._calculate_gpu(positions, box_vectors)
            except Exception as e:
                logger.warning(f"GPU calculation failed, falling back to CPU: {e}")
                # Fall through to CPU calculation
        
        # Use CPU calculation
        return super().calculate(positions, box_vectors)
    
    def _calculate_gpu(self, positions, box_vectors=None):
        """Calculate forces using GPU acceleration."""
        n_particles = positions.shape[0]
        
        # Ensure we have parameters for all particles
        if len(self.particles) < n_particles:
            logger.warning(f"Not all particles have LJ parameters ({len(self.particles)} vs {n_particles})")
            return np.zeros((n_particles, 3)), 0.0
        
        # Extract sigma and epsilon arrays
        sigma = np.array([p[0] for p in self.particles], dtype=np.float32)
        epsilon = np.array([p[1] for p in self.particles], dtype=np.float32)
        
        # Handle periodic boundary conditions in positions if needed
        pos_for_gpu = positions.copy()
        if box_vectors is not None:
            # Apply minimum image convention before GPU calculation
            pos_for_gpu = self._apply_pbc(pos_for_gpu, box_vectors)
        
        # Calculate forces on GPU
        forces, potential_energy = self.gpu_calculator.calculate_forces_gpu(
            pos_for_gpu, sigma, epsilon
        )
        
        # Apply exclusions and scale factors on CPU (these are typically sparse)
        forces, potential_energy = self._apply_exclusions_and_scaling(
            positions, forces, potential_energy, box_vectors
        )
        
        return forces, potential_energy
    
    def _apply_pbc(self, positions, box_vectors):
        """Apply periodic boundary conditions to positions."""
        # Simple cubic box PBC
        if box_vectors is not None and np.allclose(box_vectors, np.diag(np.diag(box_vectors))):
            box_lengths = np.diag(box_vectors)
            positions = positions % box_lengths
        return positions
    
    def _apply_exclusions_and_scaling(self, positions, forces, potential_energy, box_vectors):
        """Apply exclusions and scale factors (CPU implementation)."""
        # This is typically a sparse operation, so CPU implementation is fine
        if not self.exclusions and not self.scale_factors:
            return forces, potential_energy
        
        # Calculate excluded/scaled interactions on CPU and subtract from GPU result
        n_particles = positions.shape[0]
        correction_forces = np.zeros((n_particles, 3))
        correction_energy = 0.0
        
        for i in range(n_particles):
            sigma_i, epsilon_i = self.particles[i]
            
            for j in range(i+1, n_particles):
                pair = (i, j)
                
                # Skip if not excluded or scaled
                if pair not in self.exclusions and pair not in self.scale_factors:
                    continue
                
                sigma_j, epsilon_j = self.particles[j]
                
                # Calculate vector and distance
                r_vec = positions[j] - positions[i]
                if box_vectors is not None:
                    # Apply minimum image convention
                    for k in range(3):
                        if box_vectors[k, k] > 0:
                            while r_vec[k] > 0.5 * box_vectors[k, k]:
                                r_vec[k] -= box_vectors[k, k]
                            while r_vec[k] < -0.5 * box_vectors[k, k]:
                                r_vec[k] += box_vectors[k, k]
                
                r = np.linalg.norm(r_vec)
                if r >= self.cutoff or r < 1e-10:
                    continue
                
                # Calculate full interaction
                sigma = 0.5 * (sigma_i + sigma_j)
                epsilon = np.sqrt(epsilon_i * epsilon_j)
                
                sigma_over_r = sigma / r
                sigma_over_r6 = sigma_over_r**6
                sigma_over_r12 = sigma_over_r6**2
                
                energy = 4.0 * epsilon * (sigma_over_r12 - sigma_over_r6)
                force_mag = 24.0 * epsilon / r * (2.0 * sigma_over_r12 - sigma_over_r6)
                force_vec = -force_mag * r_vec / r
                
                if pair in self.exclusions:
                    # Subtract full interaction (was included in GPU calculation)
                    correction_forces[i] -= force_vec
                    correction_forces[j] += force_vec
                    correction_energy -= energy
                elif pair in self.scale_factors:
                    # Subtract difference between full and scaled interaction
                    scale_factor = self.scale_factors[pair]
                    correction_factor = 1.0 - scale_factor
                    
                    correction_forces[i] -= correction_factor * force_vec
                    correction_forces[j] += correction_factor * force_vec
                    correction_energy -= correction_factor * energy
        
        return forces + correction_forces, potential_energy + correction_energy
    
    def benchmark_performance(self, n_particles: int = 1000, n_trials: int = 5) -> Dict[str, Any]:
        """
        Benchmark GPU vs CPU performance.
        
        Parameters
        ----------
        n_particles : int
            Number of particles for benchmark
        n_trials : int
            Number of benchmark trials
            
        Returns
        -------
        Dict[str, Any]
            Performance benchmark results
        """
        if not self.gpu_enabled:
            logger.warning("GPU not enabled for benchmarking")
            return {"error": "GPU not available"}
        
        # Create test system
        positions = np.random.uniform(0, 5, (n_particles, 3)).astype(np.float32)
        for i in range(n_particles):
            self.add_particle(0.3, 1.0)  # Add dummy parameters
        
        return self.gpu_calculator.benchmark_performance(positions, n_trials)

class GPUCoulombForceTerm(CoulombForceTerm):
    """
    GPU-accelerated Coulomb electrostatic force term.
    
    This class extends the standard Coulomb force term with GPU acceleration
    while maintaining full compatibility with the existing force field interface.
    Automatically falls back to CPU implementation when GPU is not available.
    """
    
    def __init__(self, 
                 charges=None, 
                 cutoff: float = 1.0, 
                 name: str = "Coulomb_GPU",
                 relative_permittivity: float = 1.0,
                 gpu_backend: Optional[GPUBackend] = None,
                 device_id: int = 0,
                 force_cpu: bool = False):
        """
        Initialize GPU-accelerated Coulomb force term.
        
        Parameters
        ----------
        charges : array-like, optional
            Particle charges in elementary charge units
        cutoff : float, optional
            Cutoff distance in nm
        name : str, optional
            Name of the force term
        relative_permittivity : float, optional
            Relative permittivity (dielectric constant)
        gpu_backend : GPUBackend, optional
            Preferred GPU backend
        device_id : int, optional
            GPU device ID to use
        force_cpu : bool, optional
            Force CPU execution (for testing/debugging)
        """
        # Initialize parent class
        super().__init__(charges, cutoff, name, relative_permittivity)
        
        # Initialize GPU acceleration
        self.force_cpu = force_cpu
        self.gpu_calculator = None
        self.gpu_enabled = False
        
        if not force_cpu and is_gpu_available():
            try:
                self.gpu_calculator = CoulombGPU(gpu_backend, device_id, cutoff)
                self.gpu_enabled = self.gpu_calculator.is_gpu_enabled()
                if self.gpu_enabled:
                    logger.info(f"GPU acceleration enabled for {name} using {self.gpu_calculator.backend.value}")
                else:
                    logger.info(f"GPU acceleration not available for {name}, using CPU fallback")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU acceleration for {name}: {e}")
                self.gpu_enabled = False
        else:
            if force_cpu:
                logger.info(f"GPU acceleration disabled for {name} (forced CPU mode)")
            else:
                logger.info(f"GPU acceleration not available for {name}")
    
    def calculate(self, positions, box_vectors=None):
        """
        Calculate Coulomb forces with automatic GPU/CPU selection.
        
        Parameters
        ----------
        positions : np.ndarray
            Particle positions with shape (n_particles, 3)
        box_vectors : np.ndarray, optional
            Periodic box vectors
            
        Returns
        -------
        tuple
            Tuple of (forces, potential_energy)
        """
        n_particles = positions.shape[0]
        
        # Use GPU acceleration for large systems if available
        if self.gpu_enabled and n_particles >= 100:  # GPU worthwhile for >=100 particles
            try:
                return self._calculate_gpu(positions, box_vectors)
            except Exception as e:
                logger.warning(f"GPU calculation failed, falling back to CPU: {e}")
                # Fall through to CPU calculation
        
        # Use CPU calculation
        return super().calculate(positions, box_vectors)
    
    def _calculate_gpu(self, positions, box_vectors=None):
        """Calculate forces using GPU acceleration."""
        n_particles = positions.shape[0]
        
        # Ensure we have charges for all particles
        if len(self.particles) < n_particles:
            logger.warning(f"Not all particles have charges ({len(self.particles)} vs {n_particles})")
            return np.zeros((n_particles, 3)), 0.0
        
        # Extract charges array
        charges = np.array(self.particles, dtype=np.float32)
        
        # Apply relative permittivity scaling
        scaled_charges = charges / np.sqrt(self.relative_permittivity)
        
        # Handle periodic boundary conditions in positions if needed
        pos_for_gpu = positions.copy()
        if box_vectors is not None:
            # Apply minimum image convention before GPU calculation
            pos_for_gpu = self._apply_pbc(pos_for_gpu, box_vectors)
        
        # Calculate forces on GPU
        forces, potential_energy = self.gpu_calculator.calculate_forces_gpu(
            pos_for_gpu, scaled_charges
        )
        
        # Apply exclusions and scale factors on CPU (these are typically sparse)
        forces, potential_energy = self._apply_exclusions_and_scaling(
            positions, forces, potential_energy, box_vectors
        )
        
        return forces, potential_energy
    
    def _apply_pbc(self, positions, box_vectors):
        """Apply periodic boundary conditions to positions."""
        # Simple cubic box PBC
        if box_vectors is not None and np.allclose(box_vectors, np.diag(np.diag(box_vectors))):
            box_lengths = np.diag(box_vectors)
            positions = positions % box_lengths
        return positions
    
    def _apply_exclusions_and_scaling(self, positions, forces, potential_energy, box_vectors):
        """Apply exclusions and scale factors (CPU implementation)."""
        # This is typically a sparse operation, so CPU implementation is fine
        if not self.exclusions and not self.scale_factors:
            return forces, potential_energy
        
        # Calculate excluded/scaled interactions on CPU and subtract from GPU result
        n_particles = positions.shape[0]
        correction_forces = np.zeros((n_particles, 3))
        correction_energy = 0.0
        
        coulomb_factor = self.COULOMB_CONSTANT / self.relative_permittivity
        
        for i in range(n_particles):
            q_i = self.particles[i]
            if abs(q_i) < 1e-10:
                continue
                
            for j in range(i+1, n_particles):
                pair = (i, j)
                
                # Skip if not excluded or scaled
                if pair not in self.exclusions and pair not in self.scale_factors:
                    continue
                
                q_j = self.particles[j]
                if abs(q_j) < 1e-10:
                    continue
                
                # Calculate vector and distance
                r_vec = positions[j] - positions[i]
                if box_vectors is not None:
                    # Apply minimum image convention
                    for k in range(3):
                        if box_vectors[k, k] > 0:
                            while r_vec[k] > 0.5 * box_vectors[k, k]:
                                r_vec[k] -= box_vectors[k, k]
                            while r_vec[k] < -0.5 * box_vectors[k, k]:
                                r_vec[k] += box_vectors[k, k]
                
                r = np.linalg.norm(r_vec)
                if r >= self.cutoff:
                    continue
                
                # Calculate full interaction
                qq = q_i * q_j
                energy = coulomb_factor * qq / r
                force_mag = -coulomb_factor * qq / r**2
                force_vec = force_mag * r_vec / r
                
                if pair in self.exclusions:
                    # Subtract full interaction (was included in GPU calculation)
                    correction_forces[i] -= force_vec
                    correction_forces[j] += force_vec
                    correction_energy -= energy
                elif pair in self.scale_factors:
                    # Subtract difference between full and scaled interaction
                    scale_factor = self.scale_factors[pair]
                    correction_factor = 1.0 - scale_factor
                    
                    correction_forces[i] -= correction_factor * force_vec
                    correction_forces[j] += correction_factor * force_vec
                    correction_energy -= correction_factor * energy
        
        return forces + correction_forces, potential_energy + correction_energy
    
    def benchmark_performance(self, n_particles: int = 1000, n_trials: int = 5) -> Dict[str, Any]:
        """
        Benchmark GPU vs CPU performance.
        
        Parameters
        ----------
        n_particles : int
            Number of particles for benchmark
        n_trials : int
            Number of benchmark trials
            
        Returns
        -------
        Dict[str, Any]
            Performance benchmark results
        """
        if not self.gpu_enabled:
            logger.warning("GPU not enabled for benchmarking")
            return {"error": "GPU not available"}
        
        # Create test system
        positions = np.random.uniform(0, 5, (n_particles, 3)).astype(np.float32)
        for i in range(n_particles):
            self.add_particle(np.random.uniform(-1, 1))  # Add dummy charges
        
        return self.gpu_calculator.benchmark_performance(positions, n_trials)

class GPUAmberForceField:
    """
    GPU-accelerated AMBER force field factory.
    
    This class provides methods to create GPU-accelerated versions of
    AMBER force field systems with automatic fallback to CPU.
    """
    
    @staticmethod
    def create_gpu_system(system, gpu_backend: Optional[GPUBackend] = None, device_id: int = 0):
        """
        Convert a regular force field system to use GPU acceleration.
        
        Parameters
        ----------
        system : ForceFieldSystem
            Original force field system
        gpu_backend : GPUBackend, optional
            Preferred GPU backend
        device_id : int, optional
            GPU device ID to use
            
        Returns
        -------
        ForceFieldSystem
            GPU-accelerated force field system
        """
        from ..forcefield.forcefield import ForceFieldSystem
        
        gpu_system = ForceFieldSystem(name=f"{system.name}_GPU")
        gpu_system.n_atoms = system.n_atoms
        
        for term in system.force_terms:
            if isinstance(term, LennardJonesForceTerm) and not isinstance(term, GPULennardJonesForceTerm):
                # Convert to GPU-accelerated LJ term
                gpu_term = GPULennardJonesForceTerm(
                    cutoff=term.cutoff,
                    name=f"{term.name}_GPU",
                    switch_distance=term.switch_distance,
                    gpu_backend=gpu_backend,
                    device_id=device_id
                )
                # Copy parameters
                gpu_term.particles = term.particles.copy()
                gpu_term.exclusions = term.exclusions.copy()
                gpu_term.scale_factors = term.scale_factors.copy()
                gpu_system.add_force_term(gpu_term)
                
            elif isinstance(term, CoulombForceTerm) and not isinstance(term, GPUCoulombForceTerm):
                # Convert to GPU-accelerated Coulomb term
                gpu_term = GPUCoulombForceTerm(
                    cutoff=term.cutoff,
                    name=f"{term.name}_GPU",
                    relative_permittivity=term.relative_permittivity,
                    gpu_backend=gpu_backend,
                    device_id=device_id
                )
                # Copy parameters
                gpu_term.particles = term.particles.copy()
                gpu_term.exclusions = term.exclusions.copy()
                gpu_term.scale_factors = term.scale_factors.copy()
                gpu_system.add_force_term(gpu_term)
                
            else:
                # Keep other terms as-is
                gpu_system.add_force_term(term)
        
        return gpu_system
    
    @staticmethod
    def benchmark_system(system, n_particles: int = 1000, n_trials: int = 5) -> Dict[str, Any]:
        """
        Benchmark a GPU-accelerated system.
        
        Parameters
        ----------
        system : ForceFieldSystem
            GPU-accelerated force field system
        n_particles : int
            Number of particles for benchmark
        n_trials : int
            Number of benchmark trials
            
        Returns
        -------
        Dict[str, Any]
            Performance benchmark results
        """
        results = {
            'n_particles': n_particles,
            'terms': {},
            'total_speedup': 1.0,
            'gpu_enabled_terms': 0,
            'total_terms': len(system.force_terms)
        }
        
        for term in system.force_terms:
            if hasattr(term, 'benchmark_performance'):
                term_results = term.benchmark_performance(n_particles, n_trials)
                results['terms'][term.name] = term_results
                
                if 'speedup' in term_results and term_results['speedup'] > 1.0:
                    results['gpu_enabled_terms'] += 1
        
        # Calculate overall speedup (simplified)
        speedups = [r.get('speedup', 1.0) for r in results['terms'].values() if 'speedup' in r]
        if speedups:
            results['total_speedup'] = np.mean(speedups)
        
        return results
