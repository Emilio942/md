"""
Environment module for modeling cellular environments in molecular dynamics simulations.

This module provides classes and functions for setting up realistic
cellular environments, including membranes, cytoplasm, and organelles.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set, Any
import logging
from enum import Enum
import os
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class EnvironmentType(Enum):
    """Enumeration of supported cellular environment types."""
    WATER = "water"
    MEMBRANE = "membrane"
    CYTOPLASM = "cytoplasm"
    EXTRACELLULAR = "extracellular"
    NUCLEUS = "nucleus"
    ORGANELLE = "organelle"
    CUSTOM = "custom"

class CellEnvironment:
    """
    Base class for cellular environments in MD simulations.
    
    This class provides a framework for setting up and managing
    cellular environments with different components.
    """
    
    def __init__(self, 
                 name: str = "Cell",
                 box_dimensions: np.ndarray = None,
                 temperature: float = 310.0,  # Body temperature in K
                 ion_concentration: float = 0.15,  # Physiological (M)
                 pH: float = 7.4):  # Physiological pH
        """
        Initialize a cellular environment.
        
        Parameters
        ----------
        name : str, optional
            Name of the environment
        box_dimensions : np.ndarray, optional
            Dimensions of the simulation box in nm
        temperature : float, optional
            Temperature in Kelvin
        ion_concentration : float, optional
            Ion concentration in mol/L
        pH : float, optional
            pH value
        """
        self.name = name
        self.box_dimensions = box_dimensions if box_dimensions is not None else np.array([10.0, 10.0, 10.0])
        self.temperature = temperature
        self.ion_concentration = ion_concentration
        self.pH = pH
        
        # Store components
        self.components = {}  # Map name to component
        
        # Track all molecules, particles, etc.
        self.all_particles = []  # List of all particles
        self.all_molecules = []  # List of all molecules
        
        logger.info(f"Initialized {name} environment with dimensions {self.box_dimensions} nm")
    
    def add_component(self, component, name: Optional[str] = None):
        """
        Add a component to the environment.
        
        Parameters
        ----------
        component : EnvironmentComponent
            Component to add
        name : str, optional
            Name for the component. If None, uses component's name.
        """
        if name is None:
            name = component.name
            
        if name in self.components:
            logger.warning(f"Replacing existing component '{name}'")
            
        self.components[name] = component
        
        # Register component's particles and molecules
        if hasattr(component, 'get_particles'):
            self.all_particles.extend(component.get_particles())
        
        if hasattr(component, 'get_molecules'):
            self.all_molecules.extend(component.get_molecules())
            
        logger.info(f"Added {name} component to environment")
    
    def remove_component(self, name: str):
        """
        Remove a component from the environment.
        
        Parameters
        ----------
        name : str
            Name of the component to remove
        """
        if name in self.components:
            component = self.components.pop(name)
            
            # Remove component's particles and molecules
            if hasattr(component, 'get_particles'):
                for particle in component.get_particles():
                    if particle in self.all_particles:
                        self.all_particles.remove(particle)
            
            if hasattr(component, 'get_molecules'):
                for molecule in component.get_molecules():
                    if molecule in self.all_molecules:
                        self.all_molecules.remove(molecule)
                        
            logger.info(f"Removed {name} component from environment")
        else:
            logger.warning(f"Component '{name}' not found in environment")
    
    def get_component(self, name: str):
        """
        Get a component by name.
        
        Parameters
        ----------
        name : str
            Name of the component to get
            
        Returns
        -------
        component
            The requested component, or None if not found
        """
        return self.components.get(name)
    
    def get_all_particles(self):
        """
        Get all particles in the environment.
        
        Returns
        -------
        list
            List of all particles
        """
        return self.all_particles
    
    def get_all_molecules(self):
        """
        Get all molecules in the environment.
        
        Returns
        -------
        list
            List of all molecules
        """
        return self.all_molecules
    
    def get_particle_arrays(self):
        """
        Get arrays of particle properties for simulation.
        
        Returns
        -------
        tuple
            Tuple of (positions, masses, charges, types)
        """
        n_particles = len(self.all_particles)
        
        positions = np.zeros((n_particles, 3))
        masses = np.zeros(n_particles)
        charges = np.zeros(n_particles)
        types = []
        
        for i, particle in enumerate(self.all_particles):
            positions[i] = particle.position
            masses[i] = particle.mass
            charges[i] = particle.charge
            types.append(particle.type)
            
        return positions, masses, charges, types
    
    def get_topology(self):
        """
        Get the topology of the environment.
        
        Returns
        -------
        dict
            Dictionary containing topology information
        """
        # This would be implemented to return bonds, angles, dihedrals, etc.
        # based on the specific environment components
        raise NotImplementedError("Subclasses must implement this method")
    
    def __repr__(self):
        return f"{self.name}Environment(components={len(self.components)}, particles={len(self.all_particles)})"


class EnvironmentComponent:
    """
    Base class for components in a cellular environment.
    """
    
    def __init__(self, name: str, component_type: EnvironmentType):
        """
        Initialize an environment component.
        
        Parameters
        ----------
        name : str
            Name of the component
        component_type : EnvironmentType
            Type of component
        """
        self.name = name
        self.component_type = component_type
        
    def get_particles(self):
        """
        Get all particles in this component.
        
        Returns
        -------
        list
            List of particles
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_molecules(self):
        """
        Get all molecules in this component.
        
        Returns
        -------
        list
            List of molecules
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_bounding_box(self):
        """
        Get the bounding box of this component.
        
        Returns
        -------
        tuple
            Tuple of (min_coords, max_coords)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def __repr__(self):
        return f"{self.name}Component(type={self.component_type.value})"


class Membrane(EnvironmentComponent):
    """
    Represents a lipid bilayer membrane.
    """
    
    def __init__(self, 
                 name: str = "Membrane",
                 lipid_type: str = "POPC",
                 x_dim: float = 10.0,  # nm
                 y_dim: float = 10.0,  # nm
                 center_z: float = 0.0,  # nm
                 thickness: float = 4.0,  # nm
                 area_per_lipid: float = 0.65,  # nm²
                 asymmetric: bool = False,
                 upper_leaflet_lipids: Optional[Dict[str, float]] = None,
                 lower_leaflet_lipids: Optional[Dict[str, float]] = None):
        """
        Initialize a membrane.
        
        Parameters
        ----------
        name : str, optional
            Name of the membrane
        lipid_type : str, optional
            Type of lipid to use if not using a mixture
        x_dim : float, optional
            X dimension of membrane in nm
        y_dim : float, optional
            Y dimension of membrane in nm
        center_z : float, optional
            Z coordinate of membrane center
        thickness : float, optional
            Thickness of the bilayer in nm
        area_per_lipid : float, optional
            Area per lipid in nm²
        asymmetric : bool, optional
            Whether to use asymmetric leaflets
        upper_leaflet_lipids : dict, optional
            Dict mapping lipid types to fractions for upper leaflet
        lower_leaflet_lipids : dict, optional
            Dict mapping lipid types to fractions for lower leaflet
        """
        super().__init__(name, EnvironmentType.MEMBRANE)
        
        self.lipid_type = lipid_type
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.center_z = center_z
        self.thickness = thickness
        self.area_per_lipid = area_per_lipid
        self.asymmetric = asymmetric
        
        # Half thickness for convenience
        self.half_thickness = thickness / 2
        
        # Calculate number of lipids based on area
        area = x_dim * y_dim
        self.n_lipids_per_leaflet = int(area / area_per_lipid)
        
        # Set up lipid compositions
        if asymmetric:
            # Use provided compositions or default to single lipid type
            self.upper_leaflet_lipids = upper_leaflet_lipids or {lipid_type: 1.0}
            self.lower_leaflet_lipids = lower_leaflet_lipids or {lipid_type: 1.0}
        else:
            # Use same composition for both leaflets
            self.upper_leaflet_lipids = upper_leaflet_lipids or {lipid_type: 1.0}
            self.lower_leaflet_lipids = self.upper_leaflet_lipids
            
        # Make sure fractions sum to 1.0
        self._normalize_fractions(self.upper_leaflet_lipids)
        self._normalize_fractions(self.lower_leaflet_lipids)
        
        # Lists to store membrane components
        self.upper_leaflet = []
        self.lower_leaflet = []
        self.proteins = []
        self.small_molecules = []
        
        logger.info(f"Created {name} membrane with {self.n_lipids_per_leaflet*2} lipids")
    
    def _normalize_fractions(self, lipid_dict):
        """
        Normalize lipid fractions to sum to 1.0.
        
        Parameters
        ----------
        lipid_dict : dict
            Dictionary mapping lipid types to fractions
        """
        total = sum(lipid_dict.values())
        if total > 0:
            for lipid_type in lipid_dict:
                lipid_dict[lipid_type] /= total
    
    def generate_membrane(self):
        """
        Generate the membrane structure.
        
        This would typically generate a grid of lipids
        based on pre-equilibrated structures.
        """
        # In a real implementation, this would:
        # 1. Load lipid templates from a database
        # 2. Place lipids in a grid pattern with correct orientations
        # 3. Add water and ions around the membrane
        # 4. Perform a short equilibration
        
        # For now, we'll just create placeholder lipids
        self._create_lipid_grid()
        
        logger.info(f"Generated membrane structure with {len(self.upper_leaflet)} upper and {len(self.lower_leaflet)} lower lipids")
    
    def _create_lipid_grid(self):
        """
        Create a grid of lipids for the membrane.
        """
        # Clear existing lipids
        self.upper_leaflet = []
        self.lower_leaflet = []
        
        # Calculate grid dimensions
        nx = int(np.sqrt(self.n_lipids_per_leaflet))
        ny = int(self.n_lipids_per_leaflet / nx)
        
        # Adjust dimensions if needed
        while nx * ny < self.n_lipids_per_leaflet:
            nx += 1
            
        # Calculate spacing
        dx = self.x_dim / nx
        dy = self.y_dim / ny
        
        # Create upper leaflet
        for i in range(nx):
            for j in range(ny):
                x = i * dx + dx/2
                y = j * dy + dy/2
                z = self.center_z + self.half_thickness / 2
                
                # Select lipid type based on fractions
                lipid_type = self._select_lipid_type(self.upper_leaflet_lipids)
                
                # Create lipid (would be a more complex structure in reality)
                lipid = {
                    'type': lipid_type,
                    'position': np.array([x, y, z]),
                    'orientation': 'up'
                }
                
                self.upper_leaflet.append(lipid)
                
        # Create lower leaflet
        for i in range(nx):
            for j in range(ny):
                x = i * dx + dx/2
                y = j * dy + dy/2
                z = self.center_z - self.half_thickness / 2
                
                # Select lipid type based on fractions
                lipid_type = self._select_lipid_type(self.lower_leaflet_lipids)
                
                # Create lipid
                lipid = {
                    'type': lipid_type,
                    'position': np.array([x, y, z]),
                    'orientation': 'down'
                }
                
                self.lower_leaflet.append(lipid)
    
    def _select_lipid_type(self, lipid_fractions):
        """
        Select a lipid type based on the specified fractions.
        
        Parameters
        ----------
        lipid_fractions : dict
            Dictionary mapping lipid types to fractions
            
        Returns
        -------
        str
            Selected lipid type
        """
        rand = np.random.random()
        cumsum = 0.0
        
        for lipid_type, fraction in lipid_fractions.items():
            cumsum += fraction
            if rand < cumsum:
                return lipid_type
                
        # Default to first type if something goes wrong
        return next(iter(lipid_fractions.keys()))
    
    def add_protein(self, protein, position=None, orientation=None):
        """
        Add a protein to the membrane.
        
        Parameters
        ----------
        protein : Protein
            Protein to add
        position : np.ndarray, optional
            Position of the protein
        orientation : np.ndarray, optional
            Orientation of the protein
        """
        # In a real implementation, this would:
        # 1. Position the protein relative to the membrane
        # 2. Ensure proper orientation (transmembrane, peripheral, etc.)
        # 3. Remove overlapping lipids
        # 4. Relax the system
        
        if position is None:
            position = np.array([self.x_dim/2, self.y_dim/2, self.center_z])
            
        # Store protein with position
        self.proteins.append({
            'protein': protein,
            'position': position,
            'orientation': orientation
        })
        
        logger.info(f"Added protein {protein.name} to membrane at {position}")
    
    def get_particles(self):
        """
        Get all particles in the membrane.
        
        Returns
        -------
        list
            List of particles
        """
        # In a real implementation, this would return actual particles
        # from lipids and embedded proteins
        particles = []
        
        # For now, we return a placeholder
        return particles
    
    def get_molecules(self):
        """
        Get all molecules in the membrane.
        
        Returns
        -------
        list
            List of molecules
        """
        molecules = []
        molecules.extend(self.upper_leaflet)
        molecules.extend(self.lower_leaflet)
        molecules.extend([p['protein'] for p in self.proteins])
        molecules.extend(self.small_molecules)
        
        return molecules
    
    def get_bounding_box(self):
        """
        Get the bounding box of the membrane.
        
        Returns
        -------
        tuple
            Tuple of (min_coords, max_coords)
        """
        min_coords = np.array([0, 0, self.center_z - self.half_thickness])
        max_coords = np.array([self.x_dim, self.y_dim, self.center_z + self.half_thickness])
        
        return (min_coords, max_coords)
    
    def __repr__(self):
        n_proteins = len(self.proteins)
        protein_str = f", proteins={n_proteins}" if n_proteins > 0 else ""
        return f"Membrane(lipids={len(self.upper_leaflet)+len(self.lower_leaflet)}{protein_str})"


class Cytoplasm(EnvironmentComponent):
    """
    Represents the cytoplasm of a cell.
    """
    
    def __init__(self, 
                 name: str = "Cytoplasm",
                 volume: float = 1000.0,  # nm³
                 ion_concentration: float = 0.15,  # M
                 protein_concentration: float = 100.0,  # mg/mL
                 crowding_agents: Optional[Dict[str, float]] = None,
                 metabolites: Optional[Dict[str, float]] = None):
        """
        Initialize a cytoplasm environment.
        
        Parameters
        ----------
        name : str, optional
            Name of the cytoplasm
        volume : float, optional
            Volume in nm³
        ion_concentration : float, optional
            Concentration of ions in mol/L
        protein_concentration : float, optional
            Concentration of proteins in mg/mL
        crowding_agents : dict, optional
            Dict mapping crowding agent types to concentrations
        metabolites : dict, optional
            Dict mapping metabolite types to concentrations
        """
        super().__init__(name, EnvironmentType.CYTOPLASM)
        
        self.volume = volume
        self.ion_concentration = ion_concentration
        self.protein_concentration = protein_concentration
        
        # Set up components
        self.crowding_agents = crowding_agents or {}
        self.metabolites = metabolites or {}
        
        # Lists to store molecules
        self.proteins = []
        self.ions = []
        self.small_molecules = []
        self.waters = []
        
        # Calculate box dimensions from volume
        side_length = volume**(1/3)
        self.box_dimensions = np.array([side_length, side_length, side_length])
        
        logger.info(f"Created {name} cytoplasm with volume {volume:.1f} nm³")
    
    def generate_cytoplasm(self):
        """
        Generate the cytoplasm structure.
        
        This would typically generate a realistic distribution
        of proteins, ions, metabolites, and water.
        """
        # In a real implementation, this would:
        # 1. Add water molecules to fill the volume
        # 2. Add ions at the specified concentration
        # 3. Add proteins and other macromolecules
        # 4. Add metabolites and small molecules
        # 5. Equilibrate the system
        
        # For now, we'll just create placeholders
        self._add_water()
        self._add_ions()
        self._add_proteins()
        self._add_metabolites()
        
        logger.info(f"Generated cytoplasm with {len(self.waters)} waters, {len(self.ions)} ions, "
                   f"{len(self.proteins)} proteins, and {len(self.small_molecules)} small molecules")
    
    def _add_water(self, density: float = 33.3):  # 33.3 waters per nm³ at standard density
        """
        Add water molecules to the cytoplasm.
        
        Parameters
        ----------
        density : float, optional
            Density of water in molecules per nm³
        """
        # Clear existing waters
        self.waters = []
        
        # Calculate number of water molecules
        n_waters = int(self.volume * density)
        
        # In a real implementation, this would create actual water molecules
        # For now, just store the count
        self.waters = [None] * n_waters
    
    def _add_ions(self):
        """
        Add ions to the cytoplasm.
        """
        # Clear existing ions
        self.ions = []
        
        # Calculate number of ion pairs (Na+/Cl-) to add
        # 1 M = 0.6022 ions per nm³
        ion_density = self.ion_concentration * 0.6022  # ions/nm³
        n_ion_pairs = int(self.volume * ion_density)
        
        # In a real implementation, this would create actual ion objects
        # and place them in the volume
        
        # Placeholder for Na+ ions
        for i in range(n_ion_pairs):
            self.ions.append({
                'type': 'Na+',
                'position': np.random.rand(3) * self.box_dimensions,
                'charge': 1.0
            })
            
        # Placeholder for Cl- ions
        for i in range(n_ion_pairs):
            self.ions.append({
                'type': 'Cl-',
                'position': np.random.rand(3) * self.box_dimensions,
                'charge': -1.0
            })
    
    def _add_proteins(self):
        """
        Add proteins to the cytoplasm.
        """
        # Clear existing proteins
        self.proteins = []
        
        # In a real implementation, this would add a mixture of proteins
        # based on the protein concentration and a database of common
        # cytoplasmic proteins
        pass
    
    def _add_metabolites(self):
        """
        Add metabolites to the cytoplasm.
        """
        # Clear existing small molecules
        self.small_molecules = []
        
        # In a real implementation, this would add metabolites
        # based on the specified concentrations
        for metabolite, concentration in self.metabolites.items():
            # Calculate number of molecules
            # Assuming concentration in mM
            n_molecules = int(self.volume * concentration * 0.0006022)
            
            # Add molecules
            for i in range(n_molecules):
                self.small_molecules.append({
                    'type': metabolite,
                    'position': np.random.rand(3) * self.box_dimensions
                })
    
    def add_protein(self, protein, position=None):
        """
        Add a protein to the cytoplasm.
        
        Parameters
        ----------
        protein : Protein
            Protein to add
        position : np.ndarray, optional
            Position of the protein
        """
        if position is None:
            position = np.random.rand(3) * self.box_dimensions
            
        # Store protein with position
        self.proteins.append({
            'protein': protein,
            'position': position
        })
        
        logger.info(f"Added protein {protein.name} to cytoplasm at {position}")
    
    def get_particles(self):
        """
        Get all particles in the cytoplasm.
        
        Returns
        -------
        list
            List of particles
        """
        # In a real implementation, this would return actual particles
        particles = []
        
        # For now, we return a placeholder
        return particles
    
    def get_molecules(self):
        """
        Get all molecules in the cytoplasm.
        
        Returns
        -------
        list
            List of molecules
        """
        molecules = []
        molecules.extend([p['protein'] for p in self.proteins])
        molecules.extend(self.ions)
        molecules.extend(self.small_molecules)
        # Not including water molecules as there are too many
        
        return molecules
    
    def get_bounding_box(self):
        """
        Get the bounding box of the cytoplasm.
        
        Returns
        -------
        tuple
            Tuple of (min_coords, max_coords)
        """
        min_coords = np.zeros(3)
        max_coords = self.box_dimensions
        
        return (min_coords, max_coords)
    
    def __repr__(self):
        return f"Cytoplasm(volume={self.volume:.1f} nm³, proteins={len(self.proteins)}, ions={len(self.ions)})"


class ExtracellularEnvironment(EnvironmentComponent):
    """
    Represents the extracellular environment.
    """
    
    def __init__(self, 
                 name: str = "Extracellular",
                 volume: float = 1000.0,  # nm³
                 ion_concentration: float = 0.15,  # M
                 protein_concentration: float = 50.0,  # mg/mL
                 metabolites: Optional[Dict[str, float]] = None,
                 ecm_components: Optional[Dict[str, float]] = None):
        """
        Initialize an extracellular environment.
        
        Parameters
        ----------
        name : str, optional
            Name of the environment
        volume : float, optional
            Volume in nm³
        ion_concentration : float, optional
            Concentration of ions in mol/L
        protein_concentration : float, optional
            Concentration of proteins in mg/mL
        metabolites : dict, optional
            Dict mapping metabolite types to concentrations
        ecm_components : dict, optional
            Dict mapping ECM component types to concentrations
        """
        super().__init__(name, EnvironmentType.EXTRACELLULAR)
        
        self.volume = volume
        self.ion_concentration = ion_concentration
        self.protein_concentration = protein_concentration
        
        # Set up components
        self.metabolites = metabolites or {}
        self.ecm_components = ecm_components or {}
        
        # Lists to store molecules
        self.proteins = []
        self.ions = []
        self.small_molecules = []
        self.ecm = []
        self.waters = []
        
        # Calculate box dimensions from volume
        side_length = volume**(1/3)
        self.box_dimensions = np.array([side_length, side_length, side_length])
        
        logger.info(f"Created {name} extracellular environment with volume {volume:.1f} nm³")
    
    def generate_environment(self):
        """
        Generate the extracellular environment structure.
        """
        # Similar implementation to cytoplasm, but with different
        # compositions and the addition of ECM components
        # ...
        
        logger.info(f"Generated extracellular environment")
    
    def add_ecm_component(self, component_type, position=None, orientation=None):
        """
        Add an extracellular matrix component.
        
        Parameters
        ----------
        component_type : str
            Type of ECM component
        position : np.ndarray, optional
            Position of the component
        orientation : np.ndarray, optional
            Orientation of the component
        """
        if position is None:
            position = np.random.rand(3) * self.box_dimensions
            
        self.ecm.append({
            'type': component_type,
            'position': position,
            'orientation': orientation
        })
        
        logger.info(f"Added {component_type} ECM component at {position}")
    
    def get_particles(self):
        """
        Get all particles in the extracellular environment.
        
        Returns
        -------
        list
            List of particles
        """
        # Implementation similar to cytoplasm
        return []
    
    def get_molecules(self):
        """
        Get all molecules in the extracellular environment.
        
        Returns
        -------
        list
            List of molecules
        """
        molecules = []
        molecules.extend([p['protein'] for p in self.proteins])
        molecules.extend(self.ions)
        molecules.extend(self.small_molecules)
        molecules.extend(self.ecm)
        
        return molecules
    
    def get_bounding_box(self):
        """
        Get the bounding box of the extracellular environment.
        
        Returns
        -------
        tuple
            Tuple of (min_coords, max_coords)
        """
        min_coords = np.zeros(3)
        max_coords = self.box_dimensions
        
        return (min_coords, max_coords)
    
    def __repr__(self):
        return f"ExtracellularEnvironment(volume={self.volume:.1f} nm³, proteins={len(self.proteins)}, ECM={len(self.ecm)})"


class CompleteCell(CellEnvironment):
    """
    Represents a complete cell with all components.
    """
    
    def __init__(self, 
                 name: str = "Cell",
                 cell_radius: float = 10.0,  # µm
                 organelles: Optional[List[str]] = None,
                 membrane_composition: Optional[Dict[str, float]] = None,
                 cytoplasm_composition: Optional[Dict[str, float]] = None,
                 extracellular_composition: Optional[Dict[str, float]] = None):
        """
        Initialize a complete cell.
        
        Parameters
        ----------
        name : str, optional
            Name of the cell
        cell_radius : float, optional
            Radius of the cell in µm
        organelles : list, optional
            List of organelles to include
        membrane_composition : dict, optional
            Dict mapping lipid types to fractions
        cytoplasm_composition : dict, optional
            Dict mapping molecule types to concentrations
        extracellular_composition : dict, optional
            Dict mapping molecule types to concentrations
        """
        # Convert radius to nm
        radius_nm = cell_radius * 1000
        
        # Create box large enough to contain the cell
        box_size = 2.2 * radius_nm  # Leave some space around the cell
        box_dimensions = np.array([box_size, box_size, box_size])
        
        super().__init__(name, box_dimensions)
        
        self.cell_radius = cell_radius
        self.organelles = organelles or []
        self.membrane_composition = membrane_composition or {'POPC': 0.7, 'POPE': 0.15, 'CHOL': 0.15}
        self.cytoplasm_composition = cytoplasm_composition or {}
        self.extracellular_composition = extracellular_composition or {}
        
        # Cell components
        self.membrane = None
        self.cytoplasm = None
        self.extracellular = None
        self.nucleus = None
        self.other_organelles = {}
        
        logger.info(f"Created {name} cell with radius {cell_radius} µm")
    
    def build_cell(self):
        """
        Build the complete cell structure.
        """
        # Create plasma membrane
        self._build_membrane()
        
        # Create cytoplasm
        self._build_cytoplasm()
        
        # Create extracellular environment
        self._build_extracellular()
        
        # Create nucleus if requested
        if 'nucleus' in self.organelles:
            self._build_nucleus()
            
        # Create other organelles
        for organelle in self.organelles:
            if organelle != 'nucleus':
                self._build_organelle(organelle)
                
        logger.info(f"Built complete cell with {len(self.components)} components")
    
    def _build_membrane(self):
        """
        Build the plasma membrane.
        """
        # Calculate membrane dimensions for a spherical cell
        # For simplicity, we'll use a cubic approximation
        # In a real implementation, we would use a spherical membrane
        
        diameter = 2 * self.cell_radius * 1000  # nm
        
        membrane = Membrane(
            name="PlasmaMembrane",
            lipid_type="mixed",
            x_dim=diameter,
            y_dim=diameter,
            center_z=self.box_dimensions[2]/2,
            thickness=5.0,  # Typical bilayer thickness in nm
            upper_leaflet_lipids=self.membrane_composition,
            # Make inner leaflet different
            lower_leaflet_lipids={k: v*1.1 for k, v in self.membrane_composition.items()},
            asymmetric=True
        )
        
        # Generate the membrane structure
        membrane.generate_membrane()
        
        # Add to environment
        self.add_component(membrane, "membrane")
        self.membrane = membrane
        
        logger.info(f"Built plasma membrane with {len(membrane.get_molecules())} lipids")
    
    def _build_cytoplasm(self):
        """
        Build the cytoplasm.
        """
        # Calculate cytoplasm volume (spherical cell)
        volume = (4/3) * np.pi * (self.cell_radius * 1000)**3  # nm³
        
        # If we have a nucleus, reduce the volume
        if 'nucleus' in self.organelles:
            # Assume nucleus is about 10% of cell volume
            volume *= 0.9
            
        cytoplasm = Cytoplasm(
            name="Cytoplasm",
            volume=volume,
            ion_concentration=0.15,  # Physiological
            protein_concentration=100.0,  # mg/mL, typical for cells
            metabolites=self.cytoplasm_composition
        )
        
        # Generate the cytoplasm
        cytoplasm.generate_cytoplasm()
        
        # Add to environment
        self.add_component(cytoplasm, "cytoplasm")
        self.cytoplasm = cytoplasm
        
        logger.info(f"Built cytoplasm with volume {volume:.1e} nm³")
    
    def _build_extracellular(self):
        """
        Build the extracellular environment.
        """
        # Calculate extracellular volume
        # (total box volume minus cell volume)
        box_volume = np.prod(self.box_dimensions)
        cell_volume = (4/3) * np.pi * (self.cell_radius * 1000)**3  # nm³
        volume = box_volume - cell_volume
        
        extracellular = ExtracellularEnvironment(
            name="Extracellular",
            volume=volume,
            ion_concentration=0.15,  # Physiological
            protein_concentration=50.0,  # mg/mL, less than inside cell
            metabolites=self.extracellular_composition
        )
        
        # Generate the environment
        extracellular.generate_environment()
        
        # Add to environment
        self.add_component(extracellular, "extracellular")
        self.extracellular = extracellular
        
        logger.info(f"Built extracellular environment with volume {volume:.1e} nm³")
    
    def _build_nucleus(self):
        """
        Build the cell nucleus.
        """
        # Nucleus is typically about 10% of cell volume
        # and about half the diameter of the cell
        
        radius = self.cell_radius * 0.5 * 1000  # nm
        volume = (4/3) * np.pi * radius**3  # nm³
        
        # Create nucleus as a cytoplasm-like component
        # In a real implementation, this would be a specialized class
        nucleus = Cytoplasm(
            name="Nucleus",
            volume=volume,
            ion_concentration=0.15,
            protein_concentration=200.0,  # Higher in nucleus
        )
        
        # Generate the nucleus
        nucleus.generate_cytoplasm()
        
        # Add to environment
        self.add_component(nucleus, "nucleus")
        self.nucleus = nucleus
        
        logger.info(f"Built nucleus with radius {radius/1000:.1f} µm")
    
    def _build_organelle(self, organelle_type):
        """
        Build an organelle.
        
        Parameters
        ----------
        organelle_type : str
            Type of organelle to build
        """
        # In a real implementation, this would create different
        # specialized organelles based on the type
        
        # Placeholder for now
        logger.info(f"Building {organelle_type} (placeholder)")
    
    def add_protein(self, protein, location="membrane", position=None):
        """
        Add a protein to the cell.
        
        Parameters
        ----------
        protein : Protein
            Protein to add
        location : str, optional
            Where to add the protein (membrane, cytoplasm, etc.)
        position : np.ndarray, optional
            Specific position for the protein
        """
        if location == "membrane" and self.membrane is not None:
            self.membrane.add_protein(protein, position)
        elif location == "cytoplasm" and self.cytoplasm is not None:
            self.cytoplasm.add_protein(protein, position)
        elif location == "extracellular" and self.extracellular is not None:
            self.extracellular.add_protein(protein, position)
        elif location == "nucleus" and self.nucleus is not None:
            self.nucleus.add_protein(protein, position)
        else:
            logger.warning(f"Cannot add protein to {location}: component not found")
    
    def get_topology(self):
        """
        Get the topology of the environment.
        
        Returns
        -------
        dict
            Dictionary containing topology information
        """
        # In a real implementation, this would combine topology
        # information from all components
        
        return {
            'box_dimensions': self.box_dimensions,
            'components': list(self.components.keys())
        }
    
    def __repr__(self):
        return f"Cell(radius={self.cell_radius} µm, components={len(self.components)})"


# Factory functions

def create_membrane_environment(size: float = 10.0, lipid_type: str = "POPC"):
    """
    Create a membrane-only simulation environment.
    
    Parameters
    ----------
    size : float, optional
        Size of the membrane in nm
    lipid_type : str, optional
        Type of lipid to use
        
    Returns
    -------
    CellEnvironment
        Environment containing a membrane
    """
    # Create environment
    env = CellEnvironment(
        name="MembraneEnv",
        box_dimensions=np.array([size, size, size * 2])
    )
    
    # Create membrane
    membrane = Membrane(
        name="Membrane",
        lipid_type=lipid_type,
        x_dim=size,
        y_dim=size,
        center_z=size,
        thickness=4.0
    )
    
    # Generate membrane structure
    membrane.generate_membrane()
    
    # Add to environment
    env.add_component(membrane, "membrane")
    
    return env


def create_protein_water_environment(protein, box_size: float = 10.0, ion_concentration: float = 0.15):
    """
    Create a protein in water simulation environment.
    
    Parameters
    ----------
    protein : Protein
        Protein to simulate
    box_size : float, optional
        Size of the simulation box in nm
    ion_concentration : float, optional
        Ion concentration in mol/L
        
    Returns
    -------
    CellEnvironment
        Environment containing a protein in water
    """
    # Create environment
    env = CellEnvironment(
        name="ProteinWaterEnv",
        box_dimensions=np.array([box_size, box_size, box_size])
    )
    
    # Create cytoplasm-like environment with just water and ions
    cytoplasm = Cytoplasm(
        name="Water",
        volume=box_size**3,
        ion_concentration=ion_concentration,
        protein_concentration=0.0
    )
    
    # Add water and ions
    cytoplasm._add_water()
    cytoplasm._add_ions()
    
    # Add protein
    cytoplasm.add_protein(protein, np.array([box_size/2, box_size/2, box_size/2]))
    
    # Add to environment
    env.add_component(cytoplasm, "water")
    
    return env


def create_protein_membrane_environment(protein, membrane_size: float = 20.0, box_height: float = 15.0):
    """
    Create a protein-membrane simulation environment.
    
    Parameters
    ----------
    protein : Protein
        Protein to simulate (typically a membrane protein)
    membrane_size : float, optional
        Size of the membrane in nm
    box_height : float, optional
        Height of the simulation box in nm
        
    Returns
    -------
    CellEnvironment
        Environment containing a membrane with embedded protein
    """
    # Create environment
    env = CellEnvironment(
        name="ProteinMembraneEnv",
        box_dimensions=np.array([membrane_size, membrane_size, box_height])
    )
    
    # Create membrane
    membrane = Membrane(
        name="Membrane",
        lipid_type="POPC",
        x_dim=membrane_size,
        y_dim=membrane_size,
        center_z=box_height/2,
        thickness=4.0
    )
    
    # Generate membrane structure
    membrane.generate_membrane()
    
    # Add protein to membrane
    membrane.add_protein(protein)
    
    # Add to environment
    env.add_component(membrane, "membrane")
    
    # Add water on both sides of membrane
    water_volume = membrane_size * membrane_size * (box_height - 4.0)
    
    cytoplasm = Cytoplasm(
        name="Water",
        volume=water_volume,
        ion_concentration=0.15
    )
    
    # Add water and ions
    cytoplasm._add_water()
    cytoplasm._add_ions()
    
    # Add to environment
    env.add_component(cytoplasm, "water")
    
    return env
