import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
num_molecules = 5       # Number of molecules
box_size = 10.0         # Simulation box size (nm)
time_steps = 200        # Number of time steps
delta_t = 0.01          # Time step (ps)
charge = 1.0            # Charge unit (arbitrary for visualization)
epsilon_0 = 8.854e-12   # Vacuum permittivity (F/m)
mu_0 = 4 * np.pi * 1e-7 # Vacuum permeability (T·m/A)

# Initialize positions and velocities
np.random.seed(42)
positions = np.random.uniform(0, box_size, (num_molecules, 3))
velocities = np.random.uniform(-1.0, 1.0, (num_molecules, 3))

# Electric field calculation
def electric_field(r, charge):
    r_mag = np.linalg.norm(r)
    if r_mag == 0:
        return np.zeros(3)
    return (charge / (4 * np.pi * epsilon_0 * r_mag**3)) * r

# Vectorized magnetic field calculation
def calculate_magnetic_field(positions, velocities, X, Y):
    grid_res = X.shape[0]
    grid_points = np.stack([X, Y, np.zeros_like(X)], axis=2)
    
    # Calculate displacement vectors from molecules to grid points
    displacement = grid_points[:, :, np.newaxis, :] - positions
    r = np.linalg.norm(displacement, axis=3, keepdims=True)
    r[r == 0] = np.inf  # Avoid division by zero
    
    # Calculate magnetic field contributions
    velocities_reshaped = velocities.reshape(1, 1, num_molecules, 3)
    cross_prod = np.cross(velocities_reshaped, displacement, axis=3)
    B_contrib = (mu_0 / (4 * np.pi)) * (charge * cross_prod) / r**3
    
    # Sum contributions and calculate magnitude
    B_total = np.sum(B_contrib, axis=2)
    return np.linalg.norm(B_total, axis=2)

# Force calculation with periodic boundary conditions
def compute_forces(positions):
    forces = np.zeros_like(positions)
    for i in range(num_molecules):
        for j in range(i+1, num_molecules):
            r_ij = positions[j] - positions[i]
            r_ij -= np.rint(r_ij / box_size) * box_size  # Minimum image convention
            E_ij = electric_field(r_ij, charge)
            forces[i] += charge * E_ij
            forces[j] -= charge * E_ij
    return forces

# Update positions with velocity Verlet integration
def update_positions(positions, velocities, forces):
    velocities += forces * delta_t
    positions += velocities * delta_t
    return positions % box_size, velocities  # Periodic boundaries

# Initialize visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter([], [], [], c='b', s=100, label='Molecules')

# Magnetic field grid setup
grid_res = 15
X, Y = np.meshgrid(np.linspace(0, box_size, grid_res), 
                   np.linspace(0, box_size, grid_res))
contours = []

# Animation function
def animate(frame):
    global positions, velocities, contours
    forces = compute_forces(positions)
    positions, velocities = update_positions(positions, velocities, forces)
    
    # Update scatter plot
    scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    
    # Clear previous field contours
    for coll in contours:
        coll.remove()
    contours.clear()
    
    # Calculate and plot new magnetic field
    B_mag = calculate_magnetic_field(positions, velocities, X, Y)
    new_contour = ax.contour(X, Y, B_mag, levels=8, cmap='coolwarm')
    contours.extend(new_contour.collections)
    
    ax.set_title(f'Time Step: {frame}')
    return scat,

# Configure and run animation
ani = FuncAnimation(fig, animate, frames=time_steps, interval=50, blit=False)
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.set_zlim(0, box_size)
ax.set_xlabel('X (nm)')
ax.set_ylabel('Y (nm)')
ax.set_zlabel('Z (nm)')
ax.legend()
plt.show()
