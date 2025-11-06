import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameter der Simulation
num_molecules = 5  # Anzahl der Moleküle
box_size = 10.0  # Größe des Simulationswürfels (nm)
time_steps = 200  # Anzahl der Simulationsschritte
delta_t = 0.01  # Zeitschritt (ps)
charge = 1.0  # Elementarladung für Wasserstoff (willkürliche Einheit für Visualisierung)
epsilon_0 = 8.854e-12  # Permittivität des Vakuums (F/m)
mu_0 = 4 * np.pi * 1e-7  # Magnetische Feldkonstante (T*m/A)

# Erzeugen zufälliger Startpositionen und moderater Anfangsgeschwindigkeiten
positions = np.random.uniform(0, box_size, (num_molecules, 3))
velocities = np.random.uniform(-1.0, 1.0, (num_molecules, 3))

# Funktion zur Berechnung des elektrischen Feldes
def electric_field(r, charge):
    r_magnitude = np.linalg.norm(r)
    if r_magnitude == 0:
        return np.zeros(3)
    field = (charge / (4 * np.pi * epsilon_0 * r_magnitude**3)) * r
    return field

# Funktion zur Berechnung des Magnetfeldes
def magnetic_field(position, velocity):
    r = np.linalg.norm(position)
    if r == 0:
        return np.zeros(3)
    field = (mu_0 / (4 * np.pi)) * (charge * np.cross(velocity, position) / r**3)
    return field

# Funktion zur Berechnung der Kräfte für alle Atome
def compute_forces(positions):
    forces = np.zeros_like(positions)
    for i in range(num_molecules):
        for j in range(i + 1, num_molecules):
            if i != j:
                r_ij = positions[j] - positions[i]
                force_ij = electric_field(r_ij, charge)
                forces[i] += force_ij
                forces[j] -= force_ij
    return forces

# Funktion zur Aktualisierung der Positionen und Geschwindigkeiten
def update_positions(positions, velocities, forces):
    velocities += forces * delta_t
    positions += velocities * delta_t
    positions = positions % box_size  # Periodische Randbedingungen
    return positions, velocities

# Initialisieren der Animation
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', label='Atoms')

# Funktion zur Visualisierung der Magnetfeld-Kreise
def plot_magnetic_field(ax, positions, velocities):
    X, Y = np.meshgrid(np.linspace(0, box_size, 100), np.linspace(0, box_size, 100))
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            B_total = np.zeros(3)
            for k in range(num_molecules):
                B_total += magnetic_field(np.array([X[i, j], Y[i, j], 0]) - positions[k], velocities[k])
            Z[i, j] = np.linalg.norm(B_total)
    
    ax.contour(X, Y, Z, levels=10, cmap='coolwarm')

# Plotten der Anfangskonfiguration des Magnetfeldes
plot_magnetic_field(ax, positions, velocities)

def animate(step):
    global positions, velocities
    forces = compute_forces(positions)
    positions, velocities = update_positions(positions, velocities, forces)
    scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    ax.set_title(f'Schritt {step}')
    # Aktualisieren der Magnetfeld-Kreise
    ax.collections.clear()
    plot_magnetic_field(ax, positions, velocities)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b')
    
    return scat,

# Animieren der Simulation
ani = FuncAnimation(fig, animate, frames=time_steps, interval=50, blit=False)
ax.legend()
plt.show()
