"""
CUDA acceleration for molecular dynamics force calculations.
"""
import numpy as np

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

def compute_forces_cuda(positions, charges, box_dimensions, cutoff=1.2):
    """
    Compute electrostatic forces using CUDA acceleration.
    
    Parameters
    ----------
    positions : np.ndarray
        Particle positions (N, 3)
    charges : np.ndarray
        Particle charges (N,)
    box_dimensions : np.ndarray
        Box size for periodic boundaries (3,)
    cutoff : float
        Cutoff distance for interactions
        
    Returns
    -------
    forces : np.ndarray
        Computed forces (N, 3)
    """
    if not CUDA_AVAILABLE:
        return compute_forces_cpu(positions, charges, box_dimensions, cutoff)
    
    # Transfer to GPU
    pos_gpu = cp.asarray(positions)
    charges_gpu = cp.asarray(charges)
    box_gpu = cp.asarray(box_dimensions)
    
    n_particles = len(positions)
    forces_gpu = cp.zeros_like(pos_gpu)
    
    # CUDA kernel for force calculation
    for i in range(n_particles):
        # Vectorized distance calculation
        r_vec = pos_gpu - pos_gpu[i]
        
        # Apply minimum image convention
        r_vec = r_vec - cp.round(r_vec / box_gpu) * box_gpu
        
        # Calculate distances
        r_mag = cp.linalg.norm(r_vec, axis=1)
        
        # Apply cutoff and avoid self-interaction
        mask = (r_mag < cutoff) & (r_mag > 0)
        
        # Coulomb force calculation
        # F = k * q1 * q2 * r_vec / r^3
        k_coulomb = 8.9875517873681764e9  # N⋅m²/C²
        
        if cp.any(mask):
            r_vec_masked = r_vec[mask]
            r_mag_masked = r_mag[mask]
            charges_masked = charges_gpu[mask]
            
            force_mag = k_coulomb * charges_gpu[i] * charges_masked / (r_mag_masked ** 3)
            force_contrib = force_mag[:, cp.newaxis] * r_vec_masked
            
            forces_gpu[i] = cp.sum(force_contrib, axis=0)
    
    # Transfer back to CPU
    return cp.asnumpy(forces_gpu)

def compute_forces_cpu(positions, charges, box_dimensions, cutoff=1.2):
    """
    CPU fallback for force calculation.
    """
    n_particles = len(positions)
    forces = np.zeros_like(positions)
    k_coulomb = 8.9875517873681764e9  # N⋅m²/C²
    
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            r_vec = positions[j] - positions[i]
            
            # Apply minimum image convention
            r_vec = r_vec - np.round(r_vec / box_dimensions) * box_dimensions
            
            r_mag = np.linalg.norm(r_vec)
            
            if 0 < r_mag < cutoff:
                # Coulomb force
                force_mag = k_coulomb * charges[i] * charges[j] / (r_mag ** 3)
                force = force_mag * r_vec
                
                forces[i] += force
                forces[j] -= force
    
    return forces
