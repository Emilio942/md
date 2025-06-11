"""
Cellular environment module for molecular dynamics simulations.

This module provides classes and functions for creating and managing
the environment around proteins, including water models, ions,
membranes, and cellular crowding agents.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from ..core import Particle
from ..structure import Atom

# Set up logging
logger = logging.getLogger(__name__)

class Environment:
    """
    Base class for cellular environments.
    
    This class manages the particles representing the environment
    around the proteins in a simulation.
    """
    
    def __init__(self, name: str, box_size: np.ndarray):
        """
        Initialize an environment.
        
        Parameters
        ----------
        name : str
            Name of the environment
        box_size : np.ndarray
            Size of the simulation box [x, y, z] in nanometers
        """
        self.name = name
        self.box_size = box_size
        self.particles = []
    
    def add_to_system(self, system):
        """Add the environment particles to a simulation system."""
        system.add_molecules(self.particles)
        logger.info(f"Added {len(self.particles)} environment particles to the system")


class WaterModel:
    """
    Base class for water models.
    
    Water models represent water molecules with different levels of
    complexity and accuracy.
    """
    
    def __init__(self, name: str):
        """
        Initialize a water model.
        
        Parameters
        ----------
        name : str
            Name of the water model
        """
        self.name = name
    
    def create_water_molecule(self, position: np.ndarray, id_offset: int = 0) -> List[Atom]:
        """
        Create a water molecule at the given position.
        
        Parameters
        ----------
        position : np.ndarray
            Position of the oxygen atom [x, y, z] in nanometers
        id_offset : int
            Offset for atom IDs
        
        Returns
        -------
        List[Atom]
            List of atoms in the water molecule
        """
        raise NotImplementedError("This method should be implemented by derived classes")


class TIP3PWater(WaterModel):
    """
    TIP3P water model.
    
    A 3-site water model with charges on oxygen and hydrogen atoms.
    """
    
    def __init__(self):
        """Initialize a TIP3P water model."""
        super().__init__("TIP3P")
        
        # TIP3P parameters
        self.o_mass = 15.999  # u
        self.h_mass = 1.008   # u
        self.o_charge = -0.834  # e
        self.h_charge = 0.417   # e
        self.oh_bond = 0.09572  # nm
        self.hoh_angle = 104.52  # degrees
    
    def create_water_molecule(self, position: np.ndarray, id_offset: int = 0) -> List[Atom]:
        """Create a TIP3P water molecule."""
        atoms = []
        
        # Create oxygen atom
        o_atom = Atom(
            atom_id=id_offset,
            name="O",
            element="O",
            residue_name="HOH",
            residue_id=id_offset // 3,
            chain_id="W",
            mass=self.o_mass,
            charge=self.o_charge,
            position=position
        )
        atoms.append(o_atom)
        
        # Calculate hydrogen positions
        # Convert angle to radians
        theta = np.radians(self.hoh_angle / 2)
        
        # First hydrogen
        h1_pos = position + np.array([
            self.oh_bond * np.sin(theta),
            self.oh_bond * np.cos(theta),
            0.0
        ])
        
        h1_atom = Atom(
            atom_id=id_offset + 1,
            name="H1",
            element="H",
            residue_name="HOH",
            residue_id=id_offset // 3,
            chain_id="W",
            mass=self.h_mass,
            charge=self.h_charge,
            position=h1_pos
        )
        atoms.append(h1_atom)
        
        # Second hydrogen
        h2_pos = position + np.array([
            -self.oh_bond * np.sin(theta),
            self.oh_bond * np.cos(theta),
            0.0
        ])
        
        h2_atom = Atom(
            atom_id=id_offset + 2,
            name="H2",
            element="H",
            residue_name="HOH",
            residue_id=id_offset // 3,
            chain_id="W",
            mass=self.h_mass,
            charge=self.h_charge,
            position=h2_pos
        )
        atoms.append(h2_atom)
        
        return atoms


class TIP4PWater(WaterModel):
    """
    TIP4P water model.
    
    A 4-site water model with a virtual site (M) to improve electrostatics.
    """
    
    def __init__(self):
        """Initialize a TIP4P water model."""
        super().__init__("TIP4P")
        
        # TIP4P parameters
        self.o_mass = 15.999  # u
        self.h_mass = 1.008   # u
        self.o_charge = 0.0     # e (oxygen has no charge in TIP4P)
        self.h_charge = 0.52    # e
        self.m_charge = -1.04   # e (virtual site)
        self.oh_bond = 0.09572  # nm
        self.hoh_angle = 104.52  # degrees
        self.om_dist = 0.015    # nm (O-M distance)
    
    def create_water_molecule(self, position: np.ndarray, id_offset: int = 0) -> List[Atom]:
        """Create a TIP4P water molecule."""
        atoms = []
        
        # Create oxygen atom
        o_atom = Atom(
            atom_id=id_offset,
            name="O",
            element="O",
            residue_name="HOH",
            residue_id=id_offset // 4,
            chain_id="W",
            mass=self.o_mass,
            charge=self.o_charge,
            position=position
        )
        atoms.append(o_atom)
        
        # Calculate hydrogen positions
        # Convert angle to radians
        theta = np.radians(self.hoh_angle / 2)
        
        # First hydrogen
        h1_pos = position + np.array([
            self.oh_bond * np.sin(theta),
            self.oh_bond * np.cos(theta),
            0.0
        ])
        
        h1_atom = Atom(
            atom_id=id_offset + 1,
            name="H1",
            element="H",
            residue_name="HOH",
            residue_id=id_offset // 4,
            chain_id="W",
            mass=self.h_mass,
            charge=self.h_charge,
            position=h1_pos
        )
        atoms.append(h1_atom)
        
        # Second hydrogen
        h2_pos = position + np.array([
            -self.oh_bond * np.sin(theta),
            self.oh_bond * np.cos(theta),
            0.0
        ])
        
        h2_atom = Atom(
            atom_id=id_offset + 2,
            name="H2",
            element="H",
            residue_name="HOH",
            residue_id=id_offset // 4,
            chain_id="W",
            mass=self.h_mass,
            charge=self.h_charge,
            position=h2_pos
        )
        atoms.append(h2_atom)
        
        # Virtual site (M)
        m_pos = position + np.array([0.0, self.om_dist, 0.0])
        
        m_atom = Atom(
            atom_id=id_offset + 3,
            name="M",
            element="M",
            residue_name="HOH",
            residue_id=id_offset // 4,
            chain_id="W",
            mass=0.0,  # Virtual site has no mass
            charge=self.m_charge,
            position=m_pos
        )
        atoms.append(m_atom)
        
        return atoms


class WaterBox(Environment):
    """
    Class for creating a box of water molecules.
    
    This creates a box of water molecules with the specified density.
    """
    
    def __init__(self, 
                 box_size: np.ndarray,
                 water_model: WaterModel = None,
                 density: float = 33.3):  # molecules/nm^3 (~ 1 g/cm^3)
        """
        Initialize a water box.
        
        Parameters
        ----------
        box_size : np.ndarray
            Size of the simulation box [x, y, z] in nanometers
        water_model : WaterModel, optional
            Water model to use (default: TIP3P)
        density : float
            Density of water in molecules/nm^3
        """
        super().__init__("Water Box", box_size)
        
        if water_model is None:
            self.water_model = TIP3PWater()
        else:
            self.water_model = water_model
        
        self.density = density
        
        # Create water molecules
        self._create_water_molecules()
    
    def _create_water_molecules(self):
        """Create water molecules in the box."""
        # Calculate number of water molecules
        volume = np.prod(self.box_size)
        num_waters = int(volume * self.density)
        
        logger.info(f"Creating {num_waters} water molecules in a box of {volume:.1f} nm^3")
        
        # Create water molecules on a grid
        molecules_per_dim = int(np.ceil(num_waters**(1/3)))
        spacing = np.min(self.box_size) / molecules_per_dim
        
        count = 0
        id_offset = 0
        
        for i in range(molecules_per_dim):
            for j in range(molecules_per_dim):
                for k in range(molecules_per_dim):
                    if count >= num_waters:
                        break
                    
                    # Calculate position with a small random displacement
                    pos = np.array([i, j, k]) * spacing + np.random.uniform(-0.1, 0.1, 3)
                    
                    # Ensure position is within the box
                    pos = pos % self.box_size
                    
                    # Create water molecule
                    atoms = self.water_model.create_water_molecule(pos, id_offset)
                    self.particles.extend(atoms)
                    
                    # Update counters
                    count += 1
                    id_offset += len(atoms)
        
        logger.info(f"Created {count} water molecules ({len(self.particles)} atoms)")


class Ion:
    """
    Class for creating ions in a simulation.
    
    Ions are represented as charged particles.
    """
    
    def __init__(self, name: str, element: str, charge: float, mass: float):
        """
        Initialize an ion.
        
        Parameters
        ----------
        name : str
            Name of the ion (e.g., "Na+", "Cl-")
        element : str
            Chemical element
        charge : float
            Charge in elementary charge units (e)
        mass : float
            Mass in atomic mass units (u)
        """
        self.name = name
        self.element = element
        self.charge = charge
        self.mass = mass
    
    def create_ion(self, position: np.ndarray, id_offset: int = 0) -> Atom:
        """
        Create an ion at the given position.
        
        Parameters
        ----------
        position : np.ndarray
            Position [x, y, z] in nanometers
        id_offset : int
            Offset for atom ID
        
        Returns
        -------
        Atom
            The created ion as an Atom object
        """
        return Atom(
            atom_id=id_offset,
            name=self.name,
            element=self.element,
            residue_name=self.name,
            residue_id=id_offset,
            chain_id="I",
            mass=self.mass,
            charge=self.charge,
            position=position
        )


class IonicSolution(Environment):
    """
    Class for creating a solution with ions.
    
    This adds ions to a box, typically with water.
    """
    
    def __init__(self, 
                 box_size: np.ndarray,
                 ion_type: str = "NaCl",
                 concentration: float = 0.15):  # mol/L (physiological)
        """
        Initialize an ionic solution.
        
        Parameters
        ----------
        box_size : np.ndarray
            Size of the simulation box [x, y, z] in nanometers
        ion_type : str
            Type of ions to add (e.g., "NaCl", "KCl")
        concentration : float
            Ion concentration in mol/L
        """
        super().__init__(f"{ion_type} Solution", box_size)
        
        self.ion_type = ion_type
        self.concentration = concentration
        
        # Create ions
        self._create_ions()
    
    def _create_ions(self):
        """Create ions in the box."""
        # Define ion parameters
        ion_params = {
            "NaCl": {
                "cation": Ion("Na+", "Na", 1.0, 22.99),
                "anion": Ion("Cl-", "Cl", -1.0, 35.45)
            },
            "KCl": {
                "cation": Ion("K+", "K", 1.0, 39.10),
                "anion": Ion("Cl-", "Cl", -1.0, 35.45)
            },
            "CaCl2": {
                "cation": Ion("Ca2+", "Ca", 2.0, 40.08),
                "anion": Ion("Cl-", "Cl", -1.0, 35.45),
                "anion_count": 2  # 2 Cl- for each Ca2+
            }
        }
        
        if self.ion_type not in ion_params:
            raise ValueError(f"Unknown ion type: {self.ion_type}")
        
        # Get ion parameters
        ion_param = ion_params[self.ion_type]
        cation = ion_param["cation"]
        anion = ion_param["anion"]
        anion_count = ion_param.get("anion_count", 1)
        
        # Calculate number of ion pairs
        # Volume in nm^3, convert to L (1 nm^3 = 10^-24 L)
        volume_L = np.prod(self.box_size) * 1e-24
        
        # concentration in mol/L * volume in L * Avogadro's number
        num_pairs = int(self.concentration * volume_L * 6.022e23)
        
        logger.info(f"Adding {num_pairs} {self.ion_type} ion pairs to a {volume_L*1e3:.1f} nm^3 box")
        
        # Create ions at random positions
        id_offset = 0
        
        for i in range(num_pairs):
            # Create cation
            cation_pos = np.random.uniform(0, self.box_size)
            cation_atom = cation.create_ion(cation_pos, id_offset)
            self.particles.append(cation_atom)
            id_offset += 1
            
            # Create anions
            for j in range(anion_count):
                anion_pos = np.random.uniform(0, self.box_size)
                anion_atom = anion.create_ion(anion_pos, id_offset)
                self.particles.append(anion_atom)
                id_offset += 1
        
        logger.info(f"Added {len(self.particles)} ions")


class Membrane(Environment):
    """
    Class for creating a lipid membrane.
    
    This creates a simplified lipid bilayer for cell membrane simulations.
    """
    
    def __init__(self, 
                 box_size: np.ndarray,
                 lipid_type: str = "POPC",
                 thickness: float = 4.0,  # nm
                 area_per_lipid: float = 0.65):  # nm^2
        """
        Initialize a membrane.
        
        Parameters
        ----------
        box_size : np.ndarray
            Size of the simulation box [x, y, z] in nanometers
        lipid_type : str
            Type of lipid (e.g., "POPC", "DPPC")
        thickness : float
            Membrane thickness in nanometers
        area_per_lipid : float
            Area per lipid in nm^2
        """
        super().__init__(f"{lipid_type} Membrane", box_size)
        
        self.lipid_type = lipid_type
        self.thickness = thickness
        self.area_per_lipid = area_per_lipid
        
        # Create membrane
        self._create_membrane()
    
    def _create_membrane(self):
        """Create a lipid bilayer."""
        # Simplified implementation - in a real system, would use
        # actual lipid structures and proper placement
        
        # Calculate number of lipids per leaflet
        area = self.box_size[0] * self.box_size[1]
        num_lipids_per_leaflet = int(area / self.area_per_lipid)
        
        logger.info(f"Creating membrane with {num_lipids_per_leaflet * 2} lipids")
        
        # Create placeholder lipids as simple particles
        # In a real implementation, would use proper lipid structures
        id_offset = 0
        
        # Calculate z-positions of the two leaflets
        z_lower = (self.box_size[2] - self.thickness) / 2
        z_upper = (self.box_size[2] + self.thickness) / 2
        
        # Create lipids in lower leaflet
        for i in range(num_lipids_per_leaflet):
            # Calculate position
            x = np.random.uniform(0, self.box_size[0])
            y = np.random.uniform(0, self.box_size[1])
            
            # Create a simple particle to represent the lipid
            lipid = Atom(
                atom_id=id_offset,
                name="LIP",
                element="C",
                residue_name=self.lipid_type,
                residue_id=id_offset,
                chain_id="L",
                mass=600.0,  # Approximate mass of a lipid
                charge=0.0,
                position=np.array([x, y, z_lower])
            )
            self.particles.append(lipid)
            id_offset += 1
        
        # Create lipids in upper leaflet
        for i in range(num_lipids_per_leaflet):
            # Calculate position
            x = np.random.uniform(0, self.box_size[0])
            y = np.random.uniform(0, self.box_size[1])
            
            # Create a simple particle to represent the lipid
            lipid = Atom(
                atom_id=id_offset,
                name="LIP",
                element="C",
                residue_name=self.lipid_type,
                residue_id=id_offset,
                chain_id="L",
                mass=600.0,
                charge=0.0,
                position=np.array([x, y, z_upper])
            )
            self.particles.append(lipid)
            id_offset += 1
        
        logger.info(f"Created membrane with {len(self.particles)} lipids")


class CellularEnvironment:
    """
    Class for creating a complete cellular environment.
    
    This combines water, ions, membranes, and other components
    to create a realistic cellular environment.
    """
    
    def __init__(self, 
                 box_size: np.ndarray,
                 environment_type: str = "cytoplasm",
                 ion_concentration: float = 0.15,  # mol/L
                 crowding_level: float = 0.0):  # 0.0-0.3 volume fraction
        """
        Initialize a cellular environment.
        
        Parameters
        ----------
        box_size : np.ndarray
            Size of the simulation box [x, y, z] in nanometers
        environment_type : str
            Type of environment (e.g., "cytoplasm", "extracellular")
        ion_concentration : float
            Concentration of ions in mol/L
        crowding_level : float
            Level of molecular crowding (volume fraction)
        """
        self.box_size = box_size
        self.environment_type = environment_type
        self.ion_concentration = ion_concentration
        self.crowding_level = crowding_level
        
        # Components of the environment
        self.components = []
    
    def build(self):
        """Build the cellular environment."""
        if self.environment_type == "cytoplasm":
            self._build_cytoplasm()
        elif self.environment_type == "extracellular":
            self._build_extracellular()
        elif self.environment_type == "membrane":
            self._build_membrane_environment()
        else:
            raise ValueError(f"Unknown environment type: {self.environment_type}")
        
        logger.info(f"Built {self.environment_type} environment with {len(self.components)} components")
    
    def _build_cytoplasm(self):
        """Build a cytoplasmic environment."""
        # Add water
        water_box = WaterBox(self.box_size)
        self.components.append(water_box)
        
        # Add ions (K+ is more abundant in cytoplasm)
        ions = IonicSolution(self.box_size, "KCl", self.ion_concentration)
        self.components.append(ions)
        
        # Add molecular crowding if requested
        if self.crowding_level > 0:
            self._add_crowding_agents()
    
    def _build_extracellular(self):
        """Build an extracellular environment."""
        # Add water
        water_box = WaterBox(self.box_size)
        self.components.append(water_box)
        
        # Add ions (Na+ is more abundant in extracellular fluid)
        ions = IonicSolution(self.box_size, "NaCl", self.ion_concentration)
        self.components.append(ions)
        
        # Add a small amount of Ca2+
        ca_ions = IonicSolution(self.box_size, "CaCl2", 0.001)
        self.components.append(ca_ions)
    
    def _build_membrane_environment(self):
        """Build an environment with a membrane."""
        # Add a membrane
        membrane = Membrane(self.box_size)
        self.components.append(membrane)
        
        # Add water
        water_box = WaterBox(self.box_size)
        self.components.append(water_box)
        
        # Add ions
        ions = IonicSolution(self.box_size, "NaCl", self.ion_concentration)
        self.components.append(ions)
    
    def _add_crowding_agents(self):
        """Add molecular crowding agents to mimic cytoplasmic crowding."""
        # Simplified implementation - in a real system, would use
        # actual protein structures for crowding
        
        # Calculate number of crowding agents
        volume = np.prod(self.box_size)
        agent_volume = 8.0  # nm^3 (approximate volume of a small protein)
        num_agents = int(volume * self.crowding_level / agent_volume)
        
        logger.info(f"Adding {num_agents} crowding agents")
        
        # Create a simple environment with crowding agents
        class CrowdingEnvironment(Environment):
            def __init__(self, box_size, num_agents):
                super().__init__("Crowding Agents", box_size)
                
                # Create crowding agents
                for i in range(num_agents):
                    pos = np.random.uniform(0, box_size)
                    
                    agent = Atom(
                        atom_id=i,
                        name="CRW",
                        element="C",
                        residue_name="CRW",
                        residue_id=i,
                        chain_id="C",
                        mass=5000.0,  # Approximate mass of a small protein
                        charge=0.0,
                        position=pos
                    )
                    self.particles.append(agent)
        
        crowding = CrowdingEnvironment(self.box_size, num_agents)
        self.components.append(crowding)
    
    def add_to_system(self, system):
        """Add the cellular environment to a simulation system."""
        for component in self.components:
            component.add_to_system(system)
        
        logger.info(f"Added cellular environment to the system")
