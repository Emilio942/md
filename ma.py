import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameter der Simulation
num_molecules = 2  # Anzahl der Moleküle
box_size = 10.0  # Größe des Simulationswürfels (nm)
time_steps = 200  # Anzahl der Simulationsschritte
delta_t = 0.01  # Zeitschritt (ps)
charge = 1.0  # Elementarladung für Wasserstoff (willkürliche Einheit für Visualisierung)
epsilon_0 = 8.854e-12  # Permittivität des Vakuums (F/m)
mu_0 = 4 * np.pi * 1e-7  # Magnetische Feldkonstante (T*m/A)

# Manuelle Startpositionen und Geschwindigkeiten für bessere Kontrolle
positions = np.array([
    [2.0, 5.0, 5.0],  # Atom 1
    [8.0, 5.0, 5.0],  # Atom 2
])
velocities = np.array([
    [0.5, 0.0, 0.0],  # Geschwindigkeit von Atom 1
    [-0.5, 0.0, 0.0], # Geschwindigkeit von Atom 2
])

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

# Funktion zur Berechnung der Kräfte für die zwei Atome
def compute_forces(positions):
    forces = np.zeros_like(positions)
    r_12 = positions[1] - positions[0]
    force_12 = electric_field(r_12, charge)
    forces[0] = force_12
    forces[1] = -force_12
    return forces

# Funktion zur Aktualisierung der Positionen und Geschwindigkeiten
def update_positions(positions, velocities, forces):
    velocities += forces * delta_t
    positions += velocities * delta_t
    positions = positions % box_size  # Periodische Randbedingungen
    return positions, velocities

# Initialisieren der Animation
fig, ax = plt.subplots(figsize=(10, 8))
scat = ax.scatter(positions[:, 0], positions[:, 1], c='b', label='Atoms')

# Funktion zur Visualisierung der Felder
def plot_fields(ax, positions, velocities):
    X, Y = np.meshgrid(np.linspace(0, box_size, 100), np.linspace(0, box_size, 100))
    E_field_strength = np.zeros_like(X)
    B_field_strength = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            r1 = np.array([X[i, j], Y[i, j], 0]) - positions[0]
            r2 = np.array([X[i, j], Y[i, j], 0]) - positions[1]
            E1 = electric_field(r1, charge)
            E2 = electric_field(r2, charge)
            B1 = magnetic_field(r1, velocities[0])
            B2 = magnetic_field(r2, velocities[1])
            E_total = E1 + E2
            B_total = B1 + B2
            E_field_strength[i, j] = np.linalg.norm(E_total)
            B_field_strength[i, j] = np.linalg.norm(B_total)

    ax.contourf(X, Y, E_field_strength, levels=20, cmap='Blues', alpha=0.6, label='Electric Field')
    ax.contourf(X, Y, B_field_strength, levels=20, cmap='Greens', alpha=0.6, label='Magnetic Field')

# Plotten der Anfangskonfiguration der Felder
plot_fields(ax, positions, velocities)

def animate(step):
    global positions, velocities
    forces = compute_forces(positions)
    positions, velocities = update_positions(positions, velocities, forces)
    scat.set_offsets(positions[:, :2])
    ax.set_title(f'Schritt {step}')
    ax.collections.clear()
    plot_fields(ax, positions, velocities)
    ax.scatter(positions[:, 0], positions[:, 1], c='b')
    
    return scat,

# Animieren der Simulation
ani = FuncAnimation(fig, animate, frames=time_steps, interval=50, blit=False)
ax.legend()
plt.show()
