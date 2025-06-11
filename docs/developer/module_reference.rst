Module Reference
================

This guide provides comprehensive documentation for ProteinMD's module structure, APIs, and implementation patterns to help developers understand and contribute to the codebase.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

ProteinMD is organized into a modular architecture that promotes code reusability, maintainability, and extensibility. This reference guide covers all major modules and their interactions.

Module Hierarchy
~~~~~~~~~~~~~~~

.. code-block:: text

   proteinmd/
   ├── core/                    # Core simulation engine
   │   ├── simulation.py        # Main simulation class
   │   ├── system.py            # System representation
   │   ├── integrators.py       # Time integration schemes
   │   └── state.py             # Simulation state management
   ├── forces/                  # Force calculation modules
   │   ├── bonded.py            # Bonded interactions
   │   ├── nonbonded.py         # Non-bonded interactions
   │   ├── electrostatic.py     # Electrostatic forces
   │   └── external.py          # External force fields
   ├── topology/                # Molecular topology
   │   ├── molecule.py          # Molecule representation
   │   ├── residue.py           # Residue handling
   │   ├── atom.py              # Atom properties
   │   └── bonds.py             # Bond connectivity
   ├── io/                      # Input/output handling
   │   ├── readers.py           # File format readers
   │   ├── writers.py           # File format writers
   │   └── trajectory.py        # Trajectory management
   ├── analysis/                # Analysis tools
   │   ├── structural.py        # Structural analysis
   │   ├── dynamics.py          # Dynamic analysis
   │   └── thermodynamics.py    # Thermodynamic properties
   ├── utils/                   # Utility functions
   │   ├── constants.py         # Physical constants
   │   ├── units.py             # Unit conversions
   │   └── math_utils.py        # Mathematical utilities
   └── cuda/                    # GPU acceleration
       ├── kernels/             # CUDA kernels
       ├── memory.py            # GPU memory management
       └── device.py            # Device management

Core Module (`proteinmd.core`)
-----------------------------

Simulation Engine
~~~~~~~~~~~~~~~~

**Main Simulation Class:**

.. code-block:: python

   class Simulation:
       """Main simulation engine for molecular dynamics.
       
       This class orchestrates all aspects of an MD simulation,
       including system setup, force calculations, integration,
       and data collection.
       
       Attributes:
           system (System): The molecular system being simulated
           integrator (Integrator): Time integration algorithm
           forces (list): List of force calculators
           reporters (list): Output data reporters
           state (SimulationState): Current simulation state
       """
       
       def __init__(self, system, integrator, platform='cpu'):
           """Initialize simulation.
           
           Args:
               system (System): Molecular system to simulate
               integrator (Integrator): Integration algorithm
               platform (str): Computational platform ('cpu', 'cuda')
           """
           self.system = system
           self.integrator = integrator
           self.platform = platform
           self.forces = []
           self.reporters = []
           self.state = SimulationState(system)
           
       def add_force(self, force):
           """Add force calculator to simulation.
           
           Args:
               force (Force): Force calculation object
           """
           self.forces.append(force)
           force.set_system(self.system)
           
       def add_reporter(self, reporter, interval=1):
           """Add data reporter.
           
           Args:
               reporter (Reporter): Data output reporter
               interval (int): Reporting interval in steps
           """
           self.reporters.append((reporter, interval))
           
       def minimize_energy(self, max_iterations=1000, tolerance=1e-6):
           """Minimize system energy.
           
           Args:
               max_iterations (int): Maximum optimization iterations
               tolerance (float): Convergence tolerance
               
           Returns:
               bool: True if converged, False otherwise
           """
           from .optimization import EnergyMinimizer
           
           minimizer = EnergyMinimizer(self.system, self.forces)
           converged = minimizer.minimize(
               self.state.positions,
               max_iterations=max_iterations,
               tolerance=tolerance
           )
           
           if converged:
               self.state.positions = minimizer.get_positions()
               self._update_forces()
               
           return converged
           
       def step(self):
           """Perform one simulation step."""
           # Calculate forces
           self._update_forces()
           
           # Integrate equations of motion
           self.integrator.step(self.state)
           
           # Apply constraints if any
           self._apply_constraints()
           
           # Update state
           self.state.step += 1
           self.state.time += self.integrator.timestep
           
           # Generate reports
           self._generate_reports()
           
       def run(self, steps):
           """Run simulation for specified number of steps.
           
           Args:
               steps (int): Number of simulation steps
           """
           for _ in range(steps):
               self.step()
               
       def _update_forces(self):
           """Calculate all forces acting on the system."""
           # Zero out forces
           self.state.forces.fill(0.0)
           
           # Calculate each force component
           for force in self.forces:
               force.calculate_forces(self.state)
               
       def _apply_constraints(self):
           """Apply geometric constraints."""
           for constraint in self.system.constraints:
               constraint.apply(self.state)
               
       def _generate_reports(self):
           """Generate output reports."""
           for reporter, interval in self.reporters:
               if self.state.step % interval == 0:
                   reporter.report(self.state)

**System Representation:**

.. code-block:: python

   class System:
       """Represents a molecular system for simulation.
       
       Contains all information about atoms, bonds, and
       simulation parameters needed for MD calculations.
       """
       
       def __init__(self):
           self.atoms = []
           self.bonds = []
           self.angles = []
           self.dihedrals = []
           self.constraints = []
           self.box_vectors = None
           self.periodic = False
           
       def add_atom(self, element, mass, charge, position):
           """Add atom to system.
           
           Args:
               element (str): Element symbol
               mass (float): Atomic mass in amu
               charge (float): Partial charge in elementary charge units
               position (array): Initial position in nm
               
           Returns:
               int: Atom index
           """
           from .topology import Atom
           
           atom = Atom(
               index=len(self.atoms),
               element=element,
               mass=mass,
               charge=charge,
               position=position
           )
           self.atoms.append(atom)
           return len(self.atoms) - 1
           
       def add_bond(self, atom1, atom2, length=None, force_constant=None):
           """Add bond between atoms.
           
           Args:
               atom1 (int): First atom index
               atom2 (int): Second atom index
               length (float): Equilibrium bond length in nm
               force_constant (float): Force constant in kJ/mol/nm²
           """
           from .topology import Bond
           
           bond = Bond(atom1, atom2, length, force_constant)
           self.bonds.append(bond)
           
       def set_periodic_box(self, box_vectors):
           """Set periodic boundary conditions.
           
           Args:
               box_vectors (array): Box vectors as 3x3 matrix in nm
           """
           self.box_vectors = np.array(box_vectors)
           self.periodic = True
           
       @property
       def n_atoms(self):
           """Number of atoms in system."""
           return len(self.atoms)
           
       def get_positions(self):
           """Get all atom positions.
           
           Returns:
               ndarray: Positions array (n_atoms, 3)
           """
           return np.array([atom.position for atom in self.atoms])
           
       def set_positions(self, positions):
           """Set all atom positions.
           
           Args:
               positions (array): New positions (n_atoms, 3)
           """
           for atom, pos in zip(self.atoms, positions):
               atom.position = pos

Forces Module (`proteinmd.forces`)
---------------------------------

Force Calculation Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Base Force Class:**

.. code-block:: python

   from abc import ABC, abstractmethod
   
   class Force(ABC):
       """Abstract base class for force calculations.
       
       All force calculators must inherit from this class
       and implement the calculate_forces method.
       """
       
       def __init__(self):
           self.system = None
           self.enabled = True
           
       def set_system(self, system):
           """Set the system for force calculations.
           
           Args:
               system (System): Molecular system
           """
           self.system = system
           
       @abstractmethod
       def calculate_forces(self, state):
           """Calculate forces and add to state.forces.
           
           Args:
               state (SimulationState): Current simulation state
           """
           pass
           
       @abstractmethod
       def get_energy(self, state):
           """Calculate potential energy.
           
           Args:
               state (SimulationState): Current simulation state
               
           Returns:
               float: Potential energy in kJ/mol
           """
           pass

**Lennard-Jones Forces:**

.. code-block:: python

   class LennardJonesForce(Force):
       """Lennard-Jones 12-6 potential for van der Waals interactions.
       
       V(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]
       
       Attributes:
           cutoff (float): Cutoff distance in nm
           switch_distance (float): Switching function start distance
           use_switching (bool): Whether to use switching function
       """
       
       def __init__(self, cutoff=1.0, switch_distance=None):
           super().__init__()
           self.cutoff = cutoff
           self.switch_distance = switch_distance
           self.use_switching = switch_distance is not None
           self.parameters = {}  # (atom_type1, atom_type2) -> (sigma, epsilon)
           
       def add_particle_type(self, atom_type, sigma, epsilon):
           """Add Lennard-Jones parameters for atom type.
           
           Args:
               atom_type (str): Atom type identifier
               sigma (float): LJ sigma parameter in nm
               epsilon (float): LJ epsilon parameter in kJ/mol
           """
           self.parameters[atom_type] = (sigma, epsilon)
           
       def calculate_forces(self, state):
           """Calculate Lennard-Jones forces."""
           positions = state.positions
           forces = state.forces
           n_atoms = len(positions)
           
           for i in range(n_atoms):
               for j in range(i + 1, n_atoms):
                   # Calculate distance
                   r_vec = positions[i] - positions[j]
                   
                   # Apply minimum image convention if periodic
                   if self.system.periodic:
                       r_vec = self._apply_pbc(r_vec)
                   
                   r = np.linalg.norm(r_vec)
                   
                   if r < self.cutoff and r > 0:
                       # Get LJ parameters
                       type_i = self.system.atoms[i].atom_type
                       type_j = self.system.atoms[j].atom_type
                       sigma, epsilon = self._get_mixed_parameters(type_i, type_j)
                       
                       # Calculate force
                       force_magnitude = self._calculate_lj_force(r, sigma, epsilon)
                       
                       if self.use_switching:
                           force_magnitude *= self._switching_function(r)
                       
                       force_vec = force_magnitude * r_vec / r
                       
                       forces[i] += force_vec
                       forces[j] -= force_vec
                       
       def get_energy(self, state):
           """Calculate Lennard-Jones potential energy."""
           positions = state.positions
           n_atoms = len(positions)
           energy = 0.0
           
           for i in range(n_atoms):
               for j in range(i + 1, n_atoms):
                   r_vec = positions[i] - positions[j]
                   
                   if self.system.periodic:
                       r_vec = self._apply_pbc(r_vec)
                   
                   r = np.linalg.norm(r_vec)
                   
                   if r < self.cutoff and r > 0:
                       type_i = self.system.atoms[i].atom_type
                       type_j = self.system.atoms[j].atom_type
                       sigma, epsilon = self._get_mixed_parameters(type_i, type_j)
                       
                       # LJ potential
                       energy += self._calculate_lj_energy(r, sigma, epsilon)
                       
           return energy
           
       def _calculate_lj_force(self, r, sigma, epsilon):
           """Calculate LJ force magnitude."""
           sigma_r = sigma / r
           sigma_r6 = sigma_r**6
           sigma_r12 = sigma_r6**2
           
           return 24 * epsilon / r * (2 * sigma_r12 - sigma_r6)
           
       def _calculate_lj_energy(self, r, sigma, epsilon):
           """Calculate LJ potential energy."""
           sigma_r = sigma / r
           sigma_r6 = sigma_r**6
           sigma_r12 = sigma_r6**2
           
           return 4 * epsilon * (sigma_r12 - sigma_r6)

**Bonded Forces:**

.. code-block:: python

   class BondForce(Force):
       """Harmonic bond stretching potential.
       
       V(r) = 0.5 * k * (r - r0)^2
       """
       
       def __init__(self):
           super().__init__()
           
       def calculate_forces(self, state):
           """Calculate bond forces."""
           positions = state.positions
           forces = state.forces
           
           for bond in self.system.bonds:
               i, j = bond.atom1, bond.atom2
               r_vec = positions[i] - positions[j]
               
               if self.system.periodic:
                   r_vec = self._apply_pbc(r_vec)
               
               r = np.linalg.norm(r_vec)
               
               if r > 0:
                   # Harmonic potential: F = -k * (r - r0) * r_hat
                   force_magnitude = -bond.force_constant * (r - bond.length)
                   force_vec = force_magnitude * r_vec / r
                   
                   forces[i] += force_vec
                   forces[j] -= force_vec
                   
       def get_energy(self, state):
           """Calculate bond potential energy."""
           positions = state.positions
           energy = 0.0
           
           for bond in self.system.bonds:
               i, j = bond.atom1, bond.atom2
               r_vec = positions[i] - positions[j]
               
               if self.system.periodic:
                   r_vec = self._apply_pbc(r_vec)
               
               r = np.linalg.norm(r_vec)
               dr = r - bond.length
               energy += 0.5 * bond.force_constant * dr**2
               
           return energy

Topology Module (`proteinmd.topology`)
-------------------------------------

Molecular Structure Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Atom Class:**

.. code-block:: python

   class Atom:
       """Represents an individual atom in the system.
       
       Attributes:
           index (int): Unique atom identifier
           element (str): Element symbol
           atom_type (str): Force field atom type
           mass (float): Atomic mass in amu
           charge (float): Partial charge in elementary charge units
           position (ndarray): 3D position coordinates in nm
           residue (Residue): Parent residue (if applicable)
       """
       
       def __init__(self, index, element, mass, charge, position, atom_type=None):
           self.index = index
           self.element = element
           self.atom_type = atom_type or element
           self.mass = mass
           self.charge = charge
           self.position = np.array(position)
           self.residue = None
           
       @property
       def atomic_number(self):
           """Get atomic number from element symbol."""
           from .constants import ATOMIC_NUMBERS
           return ATOMIC_NUMBERS.get(self.element, 0)
           
       def distance_to(self, other_atom):
           """Calculate distance to another atom.
           
           Args:
               other_atom (Atom): Target atom
               
           Returns:
               float: Distance in nm
           """
           return np.linalg.norm(self.position - other_atom.position)
           
       def __repr__(self):
           return f"Atom({self.index}, {self.element}, {self.position})"

**Molecule Class:**

.. code-block:: python

   class Molecule:
       """Represents a complete molecule (group of connected atoms).
       
       Provides methods for molecular manipulation, analysis,
       and property calculation.
       """
       
       def __init__(self, name=""):
           self.name = name
           self.atoms = []
           self.bonds = []
           self.residues = []
           
       def add_atom(self, atom):
           """Add atom to molecule.
           
           Args:
               atom (Atom): Atom to add
           """
           atom.index = len(self.atoms)
           self.atoms.append(atom)
           
       def add_bond(self, atom1, atom2, bond_order=1):
           """Add bond between atoms.
           
           Args:
               atom1 (int or Atom): First atom
               atom2 (int or Atom): Second atom
               bond_order (int): Bond order (1=single, 2=double, etc.)
           """
           if isinstance(atom1, Atom):
               atom1 = atom1.index
           if isinstance(atom2, Atom):
               atom2 = atom2.index
               
           bond = Bond(atom1, atom2, bond_order=bond_order)
           self.bonds.append(bond)
           
       def get_center_of_mass(self):
           """Calculate center of mass.
           
           Returns:
               ndarray: Center of mass coordinates in nm
           """
           total_mass = sum(atom.mass for atom in self.atoms)
           com = np.zeros(3)
           
           for atom in self.atoms:
               com += atom.mass * atom.position
               
           return com / total_mass
           
       def get_radius_of_gyration(self):
           """Calculate radius of gyration.
           
           Returns:
               float: Radius of gyration in nm
           """
           com = self.get_center_of_mass()
           total_mass = sum(atom.mass for atom in self.atoms)
           rg_squared = 0.0
           
           for atom in self.atoms:
               r_vec = atom.position - com
               rg_squared += atom.mass * np.dot(r_vec, r_vec)
               
           return np.sqrt(rg_squared / total_mass)
           
       def translate(self, translation_vector):
           """Translate all atoms by given vector.
           
           Args:
               translation_vector (array): Translation in nm
           """
           for atom in self.atoms:
               atom.position += translation_vector
               
       def rotate(self, rotation_matrix, center=None):
           """Rotate molecule around given center.
           
           Args:
               rotation_matrix (array): 3x3 rotation matrix
               center (array): Rotation center (default: center of mass)
           """
           if center is None:
               center = self.get_center_of_mass()
               
           for atom in self.atoms:
               # Translate to origin
               relative_pos = atom.position - center
               # Rotate
               rotated_pos = rotation_matrix @ relative_pos
               # Translate back
               atom.position = rotated_pos + center

I/O Module (`proteinmd.io`)
--------------------------

File Format Support
~~~~~~~~~~~~~~~~~~

**PDB Reader:**

.. code-block:: python

   class PDBReader:
       """Read Protein Data Bank (PDB) format files.
       
       Supports both standard PDB and mmCIF formats
       with full metadata preservation.
       """
       
       def __init__(self):
           self.atoms = []
           self.bonds = []
           self.metadata = {}
           
       def read(self, filename):
           """Read PDB file and return System object.
           
           Args:
               filename (str): Path to PDB file
               
           Returns:
               System: Molecular system object
           """
           system = System()
           
           with open(filename, 'r') as f:
               for line in f:
                   record_type = line[:6].strip()
                   
                   if record_type == 'ATOM' or record_type == 'HETATM':
                       atom = self._parse_atom_line(line)
                       system.add_atom(
                           element=atom['element'],
                           mass=atom['mass'],
                           charge=0.0,  # Will be set by force field
                           position=atom['position']
                       )
                       
                   elif record_type == 'CONECT':
                       bonds = self._parse_connect_line(line)
                       for atom1, atom2 in bonds:
                           system.add_bond(atom1, atom2)
                           
                   elif record_type == 'CRYST1':
                       box_vectors = self._parse_crystal_line(line)
                       system.set_periodic_box(box_vectors)
                       
           return system
           
       def _parse_atom_line(self, line):
           """Parse ATOM/HETATM record."""
           return {
               'serial': int(line[6:11]),
               'name': line[12:16].strip(),
               'alt_loc': line[16],
               'res_name': line[17:20].strip(),
               'chain_id': line[21],
               'res_seq': int(line[22:26]),
               'x': float(line[30:38]) / 10.0,  # Convert Å to nm
               'y': float(line[38:46]) / 10.0,
               'z': float(line[46:54]) / 10.0,
               'position': np.array([
                   float(line[30:38]) / 10.0,
                   float(line[38:46]) / 10.0,
                   float(line[46:54]) / 10.0
               ]),
               'occupancy': float(line[54:60]) if line[54:60].strip() else 1.0,
               'temp_factor': float(line[60:66]) if line[60:66].strip() else 0.0,
               'element': line[76:78].strip() if len(line) > 76 else line[12:14].strip()[0],
               'mass': self._get_atomic_mass(line[76:78].strip() if len(line) > 76 else line[12:14].strip()[0])
           }

**Trajectory Writer:**

.. code-block:: python

   class TrajectoryWriter:
       """Write simulation trajectories in various formats.
       
       Supports DCD, XTC, TRR, and custom HDF5 formats
       with compression and metadata support.
       """
       
       def __init__(self, filename, format='dcd'):
           self.filename = filename
           self.format = format.lower()
           self.frame_count = 0
           self._setup_writer()
           
       def _setup_writer(self):
           """Initialize format-specific writer."""
           if self.format == 'dcd':
               self._setup_dcd_writer()
           elif self.format == 'xtc':
               self._setup_xtc_writer()
           elif self.format == 'hdf5':
               self._setup_hdf5_writer()
           else:
               raise ValueError(f"Unsupported format: {self.format}")
               
       def write_frame(self, state):
           """Write single trajectory frame.
           
           Args:
               state (SimulationState): Current simulation state
           """
           if self.format == 'dcd':
               self._write_dcd_frame(state)
           elif self.format == 'xtc':
               self._write_xtc_frame(state)
           elif self.format == 'hdf5':
               self._write_hdf5_frame(state)
               
           self.frame_count += 1
           
       def close(self):
           """Close trajectory file."""
           if hasattr(self, '_file_handle'):
               self._file_handle.close()

Analysis Module (`proteinmd.analysis`)
-------------------------------------

Structural Analysis Tools
~~~~~~~~~~~~~~~~~~~~~~~~

**RMSD Calculator:**

.. code-block:: python

   class RMSDCalculator:
       """Calculate Root Mean Square Deviation between structures.
       
       Provides methods for optimal alignment and RMSD calculation
       with support for different atom selections.
       """
       
       def __init__(self, reference_positions, selection=None):
           """Initialize with reference structure.
           
           Args:
               reference_positions (array): Reference coordinates (n_atoms, 3)
               selection (array): Atom indices to include in calculation
           """
           self.reference = np.array(reference_positions)
           self.selection = selection or np.arange(len(reference_positions))
           
       def calculate(self, positions, align=True):
           """Calculate RMSD between reference and given positions.
           
           Args:
               positions (array): Target coordinates (n_atoms, 3)
               align (bool): Whether to perform optimal alignment
               
           Returns:
               float: RMSD value in nm
           """
           ref_sel = self.reference[self.selection]
           pos_sel = positions[self.selection]
           
           if align:
               # Perform optimal alignment using Kabsch algorithm
               pos_sel = self._align_structures(ref_sel, pos_sel)
               
           # Calculate RMSD
           diff = ref_sel - pos_sel
           rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
           
           return rmsd
           
       def _align_structures(self, reference, mobile):
           """Align mobile structure to reference using Kabsch algorithm.
           
           Args:
               reference (array): Reference coordinates
               mobile (array): Mobile coordinates
               
           Returns:
               array: Aligned mobile coordinates
           """
           # Center structures
           ref_center = np.mean(reference, axis=0)
           mob_center = np.mean(mobile, axis=0)
           
           ref_centered = reference - ref_center
           mob_centered = mobile - mob_center
           
           # Calculate optimal rotation matrix
           H = mob_centered.T @ ref_centered
           U, S, Vt = np.linalg.svd(H)
           R = Vt.T @ U.T
           
           # Ensure proper rotation (det(R) = 1)
           if np.linalg.det(R) < 0:
               Vt[-1, :] *= -1
               R = Vt.T @ U.T
               
           # Apply transformation
           aligned = (R @ mob_centered.T).T + ref_center
           
           return aligned

Utils Module (`proteinmd.utils`)
-------------------------------

Common Utilities
~~~~~~~~~~~~~~~

**Physical Constants:**

.. code-block:: python

   # Physical constants in SI units
   AVOGADRO = 6.02214076e23          # mol^-1
   BOLTZMANN = 1.380649e-23          # J/K
   ELEMENTARY_CHARGE = 1.602176634e-19  # C
   VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m
   
   # Conversion factors
   ANGSTROM_TO_NM = 0.1
   KCAL_TO_KJ = 4.184
   BAR_TO_PA = 1e5
   
   # Atomic masses (amu)
   ATOMIC_MASSES = {
       'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
       'P': 30.974, 'S': 32.065, 'Na': 22.990, 'Cl': 35.453,
       # ... more elements
   }

**Unit Conversions:**

.. code-block:: python

   class UnitConverter:
       """Handle unit conversions for physical quantities.
       
       Provides methods to convert between different unit systems
       commonly used in molecular dynamics simulations.
       """
       
       @staticmethod
       def energy(value, from_unit, to_unit):
           """Convert energy units.
           
           Args:
               value (float): Energy value
               from_unit (str): Source unit ('kJ/mol', 'kcal/mol', 'eV', 'hartree')
               to_unit (str): Target unit
               
           Returns:
               float: Converted value
           """
           # Convert to kJ/mol as intermediate
           to_kj_mol = {
               'kJ/mol': 1.0,
               'kcal/mol': 4.184,
               'eV': 96.485,
               'hartree': 2625.5
           }
           
           if from_unit not in to_kj_mol or to_unit not in to_kj_mol:
               raise ValueError(f"Unsupported unit conversion: {from_unit} -> {to_unit}")
               
           # Convert to kJ/mol, then to target unit
           kj_mol_value = value * to_kj_mol[from_unit]
           return kj_mol_value / to_kj_mol[to_unit]

CUDA Module (`proteinmd.cuda`)
-----------------------------

GPU Acceleration Framework
~~~~~~~~~~~~~~~~~~~~~~~~~

**Device Management:**

.. code-block:: python

   import cupy as cp
   
   class DeviceManager:
       """Manage CUDA devices and memory allocation.
       
       Provides unified interface for GPU resource management
       across different CUDA devices.
       """
       
       def __init__(self):
           self.devices = []
           self.current_device = None
           self._initialize_devices()
           
       def _initialize_devices(self):
           """Initialize available CUDA devices."""
           if not cp.cuda.is_available():
               raise RuntimeError("CUDA not available")
               
           n_devices = cp.cuda.runtime.getDeviceCount()
           
           for i in range(n_devices):
               device = cp.cuda.Device(i)
               device_info = {
                   'id': i,
                   'name': device.attributes['Name'],
                   'compute_capability': device.compute_capability,
                   'memory_gb': device.mem_info[1] / 1024**3,
                   'device_object': device
               }
               self.devices.append(device_info)
               
       def set_device(self, device_id):
           """Set active CUDA device.
           
           Args:
               device_id (int): Device ID to activate
           """
           if device_id >= len(self.devices):
               raise ValueError(f"Device {device_id} not available")
               
           self.current_device = device_id
           cp.cuda.Device(device_id).use()
           
       def get_memory_info(self, device_id=None):
           """Get memory information for device.
           
           Args:
               device_id (int): Device ID (default: current device)
               
           Returns:
               dict: Memory information
           """
           if device_id is None:
               device_id = self.current_device
               
           with cp.cuda.Device(device_id):
               free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
               
           return {
               'free_mb': free_bytes / 1024**2,
               'total_mb': total_bytes / 1024**2,
               'used_mb': (total_bytes - free_bytes) / 1024**2,
               'utilization': (total_bytes - free_bytes) / total_bytes
           }

Module Integration Patterns
--------------------------

Dependency Injection
~~~~~~~~~~~~~~~~~~

ProteinMD uses dependency injection patterns to maintain loose coupling between modules:

.. code-block:: python

   class SimulationBuilder:
       """Builder pattern for creating configured simulations.
       
       Provides fluent interface for simulation setup with
       proper dependency injection.
       """
       
       def __init__(self):
           self.system = None
           self.integrator = None
           self.forces = []
           self.reporters = []
           self.platform = 'cpu'
           
       def with_system(self, system):
           """Set molecular system.
           
           Args:
               system (System): Molecular system
               
           Returns:
               SimulationBuilder: Self for chaining
           """
           self.system = system
           return self
           
       def with_integrator(self, integrator):
           """Set time integrator.
           
           Args:
               integrator (Integrator): Integration algorithm
               
           Returns:
               SimulationBuilder: Self for chaining
           """
           self.integrator = integrator
           return self
           
       def add_force(self, force):
           """Add force calculator.
           
           Args:
               force (Force): Force calculation object
               
           Returns:
               SimulationBuilder: Self for chaining
           """
           self.forces.append(force)
           return self
           
       def on_platform(self, platform):
           """Set computational platform.
           
           Args:
               platform (str): Platform ('cpu', 'cuda')
               
           Returns:
               SimulationBuilder: Self for chaining
           """
           self.platform = platform
           return self
           
       def build(self):
           """Build configured simulation.
           
           Returns:
               Simulation: Configured simulation object
           """
           if not self.system:
               raise ValueError("System not specified")
           if not self.integrator:
               raise ValueError("Integrator not specified")
               
           simulation = Simulation(
               system=self.system,
               integrator=self.integrator,
               platform=self.platform
           )
           
           for force in self.forces:
               simulation.add_force(force)
               
           for reporter in self.reporters:
               simulation.add_reporter(reporter)
               
           return simulation

Plugin Architecture
~~~~~~~~~~~~~~~~~~

ProteinMD supports a plugin architecture for extensibility:

.. code-block:: python

   from abc import ABC, abstractmethod
   
   class Plugin(ABC):
       """Base class for ProteinMD plugins.
       
       Plugins can extend functionality without modifying
       core modules.
       """
       
       @abstractmethod
       def get_name(self):
           """Get plugin name."""
           pass
           
       @abstractmethod
       def get_version(self):
           """Get plugin version."""
           pass
           
       @abstractmethod
       def initialize(self, context):
           """Initialize plugin with context."""
           pass
           
   class PluginManager:
       """Manage plugin loading and lifecycle."""
       
       def __init__(self):
           self.plugins = {}
           
       def register_plugin(self, plugin):
           """Register plugin.
           
           Args:
               plugin (Plugin): Plugin instance
           """
           name = plugin.get_name()
           if name in self.plugins:
               raise ValueError(f"Plugin {name} already registered")
               
           self.plugins[name] = plugin
           
       def get_plugin(self, name):
           """Get registered plugin.
           
           Args:
               name (str): Plugin name
               
           Returns:
               Plugin: Plugin instance
           """
           return self.plugins.get(name)

Testing Integration
~~~~~~~~~~~~~~~~~~

Each module includes comprehensive test coverage:

.. code-block:: python

   import pytest
   import numpy as np
   from proteinmd.core import System, Simulation
   from proteinmd.forces import LennardJonesForce
   
   class TestLennardJonesForce:
       """Test suite for Lennard-Jones force calculations."""
       
       def setup_method(self):
           """Set up test fixtures."""
           self.system = System()
           
           # Create simple two-atom system
           self.system.add_atom('Ar', 39.948, 0.0, [0.0, 0.0, 0.0])
           self.system.add_atom('Ar', 39.948, 0.0, [0.5, 0.0, 0.0])
           
           self.lj_force = LennardJonesForce(cutoff=1.0)
           self.lj_force.add_particle_type('Ar', sigma=0.34, epsilon=0.996)
           self.lj_force.set_system(self.system)
           
       def test_force_calculation(self):
           """Test force calculation accuracy."""
           from proteinmd.core import SimulationState
           
           state = SimulationState(self.system)
           self.lj_force.calculate_forces(state)
           
           # Forces should be equal and opposite
           assert np.allclose(state.forces[0], -state.forces[1])
           
           # Force should be repulsive at short distance
           assert state.forces[0, 0] < 0  # Force on first atom in -x direction
           assert state.forces[1, 0] > 0  # Force on second atom in +x direction
           
       def test_energy_calculation(self):
           """Test energy calculation."""
           from proteinmd.core import SimulationState
           
           state = SimulationState(self.system)
           energy = self.lj_force.get_energy(state)
           
           # Energy should be positive (repulsive) at short distance
           assert energy > 0

This module reference provides a comprehensive overview of ProteinMD's architecture and serves as a guide for developers working with or extending the codebase.
