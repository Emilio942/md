# ProteinMD

A comprehensive molecular dynamics simulation system for protein behavior in cellular environments.

## Overview

ProteinMD provides a modular framework for simulating proteins in realistic cellular environments, with capabilities for data management, analysis, and visualization. It is designed to be flexible, efficient, and user-friendly, suitable for a wide range of applications in structural biology, drug discovery, and biophysics.

## Features

- **Protein Structure Handling**: Load and manipulate protein structures from PDB files
- **Molecular Dynamics Simulation**: Run efficient MD simulations with multiple force fields and integration algorithms
- **Environmental Factors**: Simulate proteins in various cellular environments (membranes, cytoplasm, etc.)
- **Data Analysis**: Tools for analyzing simulation results, including trajectory analysis and visualization
- **Extensible Design**: Modular architecture that allows for easy extension with new features
- **Numerical Stability**: Enhanced force calculations with safeguards against numerical issues
- **Pressure Control**: Barostat implementations for NPT ensemble simulations
- **Advanced Visualization**: Comprehensive visualization tools for trajectory and data analysis

## Project Structure

```
proteinMD/
│
├── __init__.py                 # Package initialization
├── analysis/                   # Analysis tools for MD simulations
│   └── __init__.py
├── core/                       # Core simulation functionality
│   ├── __init__.py
│   └── simulation.py           # Main simulation engine
├── database/                   # Database management
│   └── database.py
├── environment/                # Cellular environment modules
│   ├── __init__.py
│   └── environment.py
├── forcefield/                 # Force field implementations
│   ├── __init__.py
│   └── forcefield.py
├── structure/                  # Protein structure handling
│   ├── __init__.py
│   ├── pdb_parser.py           # PDB file parsing
│   └── protein.py              # Protein data structures
├── tests/                      # Test suite
├── utils/                      # Utility functions
└── visualization/              # Visualization tools
    ├── __init__.py
    └── visualization.py
```

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/proteinMD.git
cd proteinMD

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Basic Usage

Here's a simple example of how to use ProteinMD to run a simulation:

```python
from proteinMD.structure.pdb_parser import PDBParser
from proteinMD.core.simulation import MolecularDynamicsSimulation
import numpy as np

# Parse a PDB file
parser = PDBParser()
protein = parser.parse_file('path/to/protein.pdb')

# Extract protein data
positions, masses, charges, atom_ids, chain_ids = protein.get_atoms_array()

# Set up a simulation
sim = MolecularDynamicsSimulation(
    num_particles=len(positions),
    box_dimensions=np.array([10.0, 10.0, 10.0]),  # nm
    temperature=300.0,  # K
    time_step=0.002,  # ps
    boundary_condition='periodic',
    integrator='velocity-verlet',
    thermostat='berendsen'
)

# Add particles to simulation
sim.add_particles(positions=positions, masses=masses, charges=charges)

# Initialize velocities
sim.initialize_velocities()

# Run simulation
sim.run(steps=1000)

# Save trajectory
sim.save_trajectory('trajectory.npz')
```

For more examples, check out the `examples/` directory.

## Recent Enhancements

The package has been significantly improved with the following new features:

### Numerical Stability Improvements

Force calculations have been enhanced for better stability and to handle edge cases:

- Added minimum distance cutoffs to prevent division by zero
- Implemented safety checks for near-zero distances in bonded interactions
- Added force magnitude limiting to prevent extreme forces
- Improved angle and dihedral calculations with proper clamping
- Enhanced vector normalization with safety checks

### Barostat Functionality

Pressure control has been implemented with two algorithms:

- **Berendsen barostat**: Simple pressure scaling for equilibration
- **Parrinello-Rahman barostat**: More accurate NPT ensemble simulation
- Pressure calculation using the virial theorem

### Visualization Module

A comprehensive visualization module has been added:

- Trajectory visualization with 3D rendering and animation
- Energy, temperature, and pressure plotting tools
- Distance analysis between particles
- Support for various output formats (PNG, GIF, MP4)

## Usage Examples

### Basic Simulation

```python
from proteinMD.core.simulation import MolecularDynamicsSimulation
import numpy as np

# Create a simulation
sim = MolecularDynamicsSimulation(
    num_particles=0,
    box_dimensions=np.array([5.0, 5.0, 5.0]),  # 5 nm box
    temperature=300.0  # 300 K
)

# Add particles
positions = np.random.uniform(0, 5.0, (100, 3))
masses = np.ones(100) * 18.0  # water-like mass
charges = np.zeros(100)  # neutral

sim.add_particles(positions, masses, charges)

# Run simulation
for step in range(1000):
    sim.step()
```

### Using the Barostat

```python
# Create a simulation with pressure control
sim = MolecularDynamicsSimulation(
    num_particles=0,
    box_dimensions=np.array([5.0, 5.0, 5.0]),
    temperature=300.0,
    barostat='berendsen'  # Enable Berendsen barostat
)

# Set target pressure
sim.target_pressure = 1.0  # 1 bar

# In simulation loop
for step in range(1000):
    sim.step()
    sim.apply_thermostat()
    pressure = sim.apply_barostat()
```

### Visualizing Results

```python
from proteinMD.visualization.visualization import visualize_trajectory, plot_energy

# Load and visualize a trajectory
visualizer = visualize_trajectory("trajectory.npz", output_file="trajectory.gif")

# Plot energy data
energy_data = {
    'kinetic': kinetic_energies,
    'potential': potential_energies,
    'total': total_energies
}
plot_energy(energy_data, output_file="energy_plot.png")
```

For more detailed examples, see the `examples/advanced_features.py` file.

## Requirements

- Python 3.8+
- NumPy
- Matplotlib (for visualization)
- MDTraj (optional, for trajectory analysis)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- This project was inspired by various molecular dynamics packages like GROMACS, AMBER, and OpenMM
- Thanks to the computational biophysics community for their valuable research and open-source contributions
