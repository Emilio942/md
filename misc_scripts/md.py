import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameter der Simulation
box_size = 10.0  # Größe des Simulationswürfels (nm)
time_steps = 200  # Anzahl der Simulationsschritte
delta_t = 0.01  # Zeitschritt (ps)
charge = 1.0  # Elementarladung für Wasserstoff (willkürliche Einheit für Visualisierung)
epsilon_0 = 8.854e-12  # Permittivität des Vakuums (F/m)

# Startpositionen der zwei Wasserstoffatome
positions = np.array([[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]])  # Positionen der zwei Atome
velocities = np.zeros((2, 3))  # Startgeschwindigkeiten (ruhend)

# Funktion zur Berechnung des elektrischen Feldes
def electric_field(r, charge):
    r_magnitude = np.linalg.norm(r)
    if r_magnitude == 0:
        return np.zeros(3)
    field = (charge / (4 * np.pi * epsilon_0 * r_magnitude**3)) * r
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
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', label='Atoms')

# Visualisierung des elektrischen Feldes
def plot_electric_field(ax, positions):
    X, Y, Z = np.meshgrid(np.linspace(0, box_size, 10),
                          np.linspace(0, box_size, 10),
                          np.linspace(0, box_size, 10))
    Ex, Ey, Ez = np.zeros(X.shape), np.zeros(Y.shape), np.zeros(Z.shape)
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Z.shape[2]):
                r1 = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]]) - positions[0]
                r2 = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]]) - positions[1]
                E1 = electric_field(r1, charge)
                E2 = electric_field(r2, charge)
                E_total = E1 + E2
                Ex[i, j, k], Ey[i, j, k], Ez[i, j, k] = E_total

    ax.quiver(X, Y, Z, Ex, Ey, Ez, length=0.5, color='r', alpha=0.3)

# Plotten der Anfangskonfiguration des elektrischen Feldes
plot_electric_field(ax, positions)

def animate(step):
    global positions, velocities
    forces = compute_forces(positions)
    positions, velocities = update_positions(positions, velocities, forces)
    scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    ax.set_title(f'Schritt {step}')
    # Aktualisieren des elektrischen Feldes
    ax.collections.clear()
    plot_electric_field(ax, positions)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b')
    return scat,

# Animieren der Simulation
ani = FuncAnimation(fig, animate, frames=time_steps, interval=50, blit=False)
plt.show()
