"""
Multi-Format I/O Support for ProteinMD

Task 12.1: Multi-Format Support 🚀
Status: IMPLEMENTING

This module provides comprehensive support for various molecular file formats,
including both structure and trajectory files with automatic format detection
and conversion capabilities.

Requirements:
1. Import: PDB, PDBx/mmCIF, MOL2, XYZ, GROMACS GRO
2. Export: PDB, XYZ, DCD, XTC, TRR
3. Automatische Format-Erkennung implementiert
4. Konverter zwischen verschiedenen Formaten

Author: GitHub Copilot
Date: June 12, 2025
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, IO
from abc import ABC, abstractmethod
import logging
import json
import gzip
import struct
from dataclasses import dataclass
from enum import Enum
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FormatType(Enum):
    """Enumeration of supported file format types."""
    
    # Structure formats
    PDB = "pdb"
    PDBX_MMCIF = "cif"
    MOL2 = "mol2"
    XYZ = "xyz"
    GRO = "gro"
    
    # Trajectory formats
    DCD = "dcd"
    XTC = "xtc"
    TRR = "trr"
    NPZ = "npz"  # NumPy compressed
    
    # Data formats
    JSON = "json"
    CSV = "csv"
    
    # Unknown format
    UNKNOWN = "unknown"


@dataclass
class StructureData:
    """Container for molecular structure data."""
    
    # Atomic data
    coordinates: np.ndarray  # Shape: (n_atoms, 3)
    elements: List[str]
    atom_names: List[str]
    atom_ids: List[int]
    
    # Residue data
    residue_names: List[str]
    residue_ids: List[int]
    chain_ids: List[str]
    
    # Optional data
    occupancies: Optional[List[float]] = None
    b_factors: Optional[List[float]] = None
    charges: Optional[List[float]] = None
    masses: Optional[List[float]] = None
    
    # System properties
    box_vectors: Optional[np.ndarray] = None  # Shape: (3, 3) for periodic box
    title: Optional[str] = None
    resolution: Optional[float] = None
    
    # Connectivity
    bonds: Optional[List[Tuple[int, int]]] = None
    
    @property
    def n_atoms(self) -> int:
        """Number of atoms in the structure."""
        return len(self.coordinates)
    
    @property
    def n_residues(self) -> int:
        """Number of residues in the structure."""
        return len(set(self.residue_ids))
    
    @property
    def n_chains(self) -> int:
        """Number of chains in the structure."""
        return len(set(self.chain_ids))


@dataclass
class TrajectoryData:
    """Container for molecular trajectory data."""
    
    # Trajectory data
    coordinates: np.ndarray  # Shape: (n_frames, n_atoms, 3)
    time_points: np.ndarray  # Shape: (n_frames,)
    
    # Optional trajectory data
    velocities: Optional[np.ndarray] = None  # Shape: (n_frames, n_atoms, 3)
    forces: Optional[np.ndarray] = None      # Shape: (n_frames, n_atoms, 3)
    box_vectors: Optional[np.ndarray] = None # Shape: (n_frames, 3, 3)
    
    # Metadata
    topology: Optional[StructureData] = None
    title: Optional[str] = None
    
    @property
    def n_frames(self) -> int:
        """Number of frames in the trajectory."""
        return len(self.coordinates)
    
    @property
    def n_atoms(self) -> int:
        """Number of atoms per frame."""
        return self.coordinates.shape[1] if len(self.coordinates.shape) > 1 else 0
    
    @property
    def simulation_time(self) -> float:
        """Total simulation time."""
        return self.time_points[-1] - self.time_points[0] if len(self.time_points) > 1 else 0.0


class FormatDetector:
    """Automatic file format detection based on file extension and content."""
    
    @staticmethod
    def detect_format(file_path: Union[str, Path]) -> FormatType:
        """
        Detect file format from file extension and optionally file content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected format type
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        # Extension-based detection
        extension_map = {
            '.pdb': FormatType.PDB,
            '.cif': FormatType.PDBX_MMCIF,
            '.mmcif': FormatType.PDBX_MMCIF,
            '.mol2': FormatType.MOL2,
            '.xyz': FormatType.XYZ,
            '.gro': FormatType.GRO,
            '.dcd': FormatType.DCD,
            '.xtc': FormatType.XTC,
            '.trr': FormatType.TRR,
            '.npz': FormatType.NPZ,
            '.json': FormatType.JSON,
            '.csv': FormatType.CSV,
        }
        
        if extension in extension_map:
            return extension_map[extension]
        
        # Content-based detection for ambiguous cases
        if file_path.exists():
            try:
                return FormatDetector._detect_from_content(file_path)
            except Exception as e:
                logger.warning(f"Content-based detection failed: {e}")
        
        logger.warning(f"Unknown format for file: {file_path}")
        return FormatType.UNKNOWN
    
    @staticmethod
    def _detect_from_content(file_path: Path) -> FormatType:
        """Detect format from file content."""
        with open(file_path, 'rb') as f:
            # Read first few bytes
            header = f.read(64)
            
        try:
            # Try to decode as text
            header_text = header.decode('utf-8', errors='ignore').strip()
            
            # PDB format detection
            if header_text.startswith(('HEADER', 'TITLE', 'ATOM', 'HETATM')):
                return FormatType.PDB
            
            # CIF format detection
            if header_text.startswith(('data_', '#')):
                return FormatType.PDBX_MMCIF
            
            # MOL2 format detection
            if '@<TRIPOS>' in header_text:
                return FormatType.MOL2
            
            # GROMACS GRO format detection
            if '\n' in header_text:
                lines = header_text.split('\n')
                if len(lines) >= 2 and lines[1].strip().isdigit():
                    return FormatType.GRO
            
        except UnicodeDecodeError:
            # Binary format detection
            if header.startswith(b'CORD'):
                return FormatType.DCD
            
        return FormatType.UNKNOWN


class BaseFormatReader(ABC):
    """Abstract base class for format readers."""
    
    @abstractmethod
    def read_structure(self, file_path: Union[str, Path]) -> StructureData:
        """Read a structure from file."""
        pass
    
    @abstractmethod
    def can_read_trajectory(self) -> bool:
        """Check if this reader supports trajectory data."""
        pass
    
    def read_trajectory(self, file_path: Union[str, Path]) -> TrajectoryData:
        """Read a trajectory from file."""
        raise NotImplementedError("Trajectory reading not supported by this reader")


class BaseFormatWriter(ABC):
    """Abstract base class for format writers."""
    
    @abstractmethod
    def write_structure(self, structure: StructureData, file_path: Union[str, Path]) -> None:
        """Write a structure to file."""
        pass
    
    @abstractmethod
    def can_write_trajectory(self) -> bool:
        """Check if this writer supports trajectory data."""
        pass
    
    def write_trajectory(self, trajectory: TrajectoryData, file_path: Union[str, Path]) -> None:
        """Write a trajectory to file."""
        raise NotImplementedError("Trajectory writing not supported by this writer")


class PDBReader(BaseFormatReader):
    """Reader for PDB format files."""
    
    def read_structure(self, file_path: Union[str, Path]) -> StructureData:
        """Read a PDB structure file."""
        logger.info(f"Reading PDB file: {file_path}")
        
        coordinates = []
        elements = []
        atom_names = []
        atom_ids = []
        residue_names = []
        residue_ids = []
        chain_ids = []
        occupancies = []
        b_factors = []
        
        title = None
        resolution = None
        
        with open(file_path, 'r') as f:
            for line in f:
                record_type = line[:6].strip()
                
                if record_type == 'HEADER':
                    title = line[10:50].strip()
                    
                elif record_type == 'REMARK' and '2 RESOLUTION' in line:
                    try:
                        resolution = float(line.split()[-2])
                    except (ValueError, IndexError):
                        pass
                
                elif record_type in ('ATOM', 'HETATM'):
                    # Parse atom record
                    atom_id = int(line[6:11])
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    chain_id = line[21:22].strip()
                    residue_id = int(line[22:26])
                    
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    occupancy = float(line[54:60]) if line[54:60].strip() else 1.0
                    b_factor = float(line[60:66]) if line[60:66].strip() else 0.0
                    
                    element = line[76:78].strip()
                    if not element:
                        # Guess element from atom name
                        element = atom_name[0] if atom_name else 'C'
                    
                    # Store data
                    coordinates.append([x, y, z])
                    elements.append(element)
                    atom_names.append(atom_name)
                    atom_ids.append(atom_id)
                    residue_names.append(residue_name)
                    residue_ids.append(residue_id)
                    chain_ids.append(chain_id)
                    occupancies.append(occupancy)
                    b_factors.append(b_factor)
        
        if not coordinates:
            raise ValueError(f"No atoms found in PDB file: {file_path}")
        
        return StructureData(
            coordinates=np.array(coordinates) / 10.0,  # Convert Å to nm
            elements=elements,
            atom_names=atom_names,
            atom_ids=atom_ids,
            residue_names=residue_names,
            residue_ids=residue_ids,
            chain_ids=chain_ids,
            occupancies=occupancies,
            b_factors=b_factors,
            title=title,
            resolution=resolution
        )
    
    def can_read_trajectory(self) -> bool:
        """PDB reader doesn't support trajectory data."""
        return False


class XYZReader(BaseFormatReader):
    """Reader for XYZ format files."""
    
    def read_structure(self, file_path: Union[str, Path]) -> StructureData:
        """Read an XYZ structure file."""
        logger.info(f"Reading XYZ file: {file_path}")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            raise ValueError(f"Invalid XYZ file format: {file_path}")
        
        # Parse header
        n_atoms = int(lines[0].strip())
        title = lines[1].strip() if len(lines) > 1 else ""
        
        coordinates = []
        elements = []
        atom_names = []
        
        for i, line in enumerate(lines[2:2+n_atoms]):
            parts = line.strip().split()
            if len(parts) < 4:
                raise ValueError(f"Invalid atom line in XYZ file: {line}")
            
            element = parts[0]
            x, y, z = map(float, parts[1:4])
            
            coordinates.append([x, y, z])
            elements.append(element)
            atom_names.append(f"{element}{i+1}")
        
        if len(coordinates) != n_atoms:
            raise ValueError(f"Expected {n_atoms} atoms, found {len(coordinates)}")
        
        return StructureData(
            coordinates=np.array(coordinates),  # Already in nm for XYZ
            elements=elements,
            atom_names=atom_names,
            atom_ids=list(range(1, n_atoms + 1)),
            residue_names=['UNK'] * n_atoms,
            residue_ids=[1] * n_atoms,
            chain_ids=['A'] * n_atoms,
            title=title
        )
    
    def can_read_trajectory(self) -> bool:
        """XYZ reader can support multi-frame files."""
        return True
    
    def read_trajectory(self, file_path: Union[str, Path]) -> TrajectoryData:
        """Read multi-frame XYZ trajectory."""
        logger.info(f"Reading XYZ trajectory: {file_path}")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        frames = []
        time_points = []
        frame = 0
        i = 0
        
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
                
            try:
                n_atoms = int(lines[i].strip())
            except ValueError:
                i += 1
                continue
            
            if i + 1 + n_atoms >= len(lines):
                break
            
            # Parse comment line for time information
            comment = lines[i + 1].strip()
            time_point = frame * 1.0  # Default: 1 ps per frame
            
            # Try to extract time from comment
            if 'time' in comment.lower():
                import re
                time_match = re.search(r'time[=:]?\s*([\d.]+)', comment.lower())
                if time_match:
                    time_point = float(time_match.group(1))
            
            coordinates = []
            for j in range(i + 2, i + 2 + n_atoms):
                parts = lines[j].strip().split()
                if len(parts) >= 4:
                    x, y, z = map(float, parts[1:4])
                    coordinates.append([x, y, z])
            
            if len(coordinates) == n_atoms:
                frames.append(coordinates)
                time_points.append(time_point)
                frame += 1
            
            i += 2 + n_atoms
        
        if not frames:
            raise ValueError(f"No valid frames found in trajectory: {file_path}")
        
        # Create topology from first frame
        first_structure = self.read_structure(file_path)
        
        return TrajectoryData(
            coordinates=np.array(frames),
            time_points=np.array(time_points),
            topology=first_structure,
            title=f"XYZ trajectory from {file_path}"
        )


class GROReader(BaseFormatReader):
    """Reader for GROMACS GRO format files."""
    
    def read_structure(self, file_path: Union[str, Path]) -> StructureData:
        """Read a GROMACS GRO structure file."""
        logger.info(f"Reading GRO file: {file_path}")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 3:
            raise ValueError(f"Invalid GRO file format: {file_path}")
        
        # Parse header
        title = lines[0].strip()
        n_atoms = int(lines[1].strip())
        
        coordinates = []
        elements = []
        atom_names = []
        atom_ids = []
        residue_names = []
        residue_ids = []
        chain_ids = []
        
        for i in range(2, 2 + n_atoms):
            line = lines[i]
            
            # GRO format: resid, resname, atomname, atomid, x, y, z, vx, vy, vz
            residue_id = int(line[0:5])
            residue_name = line[5:10].strip()
            atom_name = line[10:15].strip()
            atom_id = int(line[15:20])
            
            x = float(line[20:28])  # nm
            y = float(line[28:36])  # nm
            z = float(line[36:44])  # nm
            
            # Guess element from atom name
            element = atom_name[0] if atom_name else 'C'
            chain_id = 'A'  # GRO doesn't have chain information
            
            coordinates.append([x, y, z])
            elements.append(element)
            atom_names.append(atom_name)
            atom_ids.append(atom_id)
            residue_names.append(residue_name)
            residue_ids.append(residue_id)
            chain_ids.append(chain_id)
        
        # Parse box vectors if present
        box_vectors = None
        if len(lines) > 2 + n_atoms:
            box_line = lines[2 + n_atoms].strip()
            box_parts = box_line.split()
            if len(box_parts) >= 3:
                # Simple cubic box
                box_x, box_y, box_z = map(float, box_parts[:3])
                box_vectors = np.array([
                    [box_x, 0.0, 0.0],
                    [0.0, box_y, 0.0],
                    [0.0, 0.0, box_z]
                ])
        
        return StructureData(
            coordinates=np.array(coordinates),  # Already in nm
            elements=elements,
            atom_names=atom_names,
            atom_ids=atom_ids,
            residue_names=residue_names,
            residue_ids=residue_ids,
            chain_ids=chain_ids,
            box_vectors=box_vectors,
            title=title
        )
    
    def can_read_trajectory(self) -> bool:
        """GRO reader doesn't typically support trajectory data."""
        return False


class PDBWriter(BaseFormatWriter):
    """Writer for PDB format files."""
    
    def write_structure(self, structure: StructureData, file_path: Union[str, Path]) -> None:
        """Write a structure to PDB format."""
        logger.info(f"Writing PDB file: {file_path}")
        
        with open(file_path, 'w') as f:
            # Write header
            if structure.title:
                f.write(f"TITLE     {structure.title:<70}\n")
            
            if structure.resolution:
                f.write(f"REMARK   2 RESOLUTION.    {structure.resolution:6.2f} ANGSTROMS.\n")
            
            # Write atoms
            for i in range(structure.n_atoms):
                atom_id = structure.atom_ids[i] if structure.atom_ids else i + 1
                atom_name = structure.atom_names[i] if structure.atom_names else f"C{i+1}"
                residue_name = structure.residue_names[i] if structure.residue_names else "UNK"
                chain_id = structure.chain_ids[i] if structure.chain_ids else "A"
                residue_id = structure.residue_ids[i] if structure.residue_ids else 1
                
                # Convert nm to Å
                x, y, z = structure.coordinates[i] * 10.0
                
                occupancy = structure.occupancies[i] if structure.occupancies else 1.0
                b_factor = structure.b_factors[i] if structure.b_factors else 0.0
                element = structure.elements[i] if structure.elements else "C"
                
                f.write(f"ATOM  {atom_id:5d} {atom_name:4s} {residue_name:3s} "
                       f"{chain_id}{residue_id:4d}    "
                       f"{x:8.3f}{y:8.3f}{z:8.3f}"
                       f"{occupancy:6.2f}{b_factor:6.2f}          "
                       f"{element:2s}\n")
            
            f.write("END\n")
    
    def can_write_trajectory(self) -> bool:
        """PDB writer doesn't support trajectory data."""
        return False


class XYZWriter(BaseFormatWriter):
    """Writer for XYZ format files."""
    
    def write_structure(self, structure: StructureData, file_path: Union[str, Path]) -> None:
        """Write a structure to XYZ format."""
        logger.info(f"Writing XYZ file: {file_path}")
        
        with open(file_path, 'w') as f:
            # Write header
            f.write(f"{structure.n_atoms}\n")
            title = structure.title or f"Structure with {structure.n_atoms} atoms"
            f.write(f"{title}\n")
            
            # Write atoms
            for i in range(structure.n_atoms):
                element = structure.elements[i] if structure.elements else "C"
                x, y, z = structure.coordinates[i]  # Keep in nm
                
                f.write(f"{element:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")
    
    def can_write_trajectory(self) -> bool:
        """XYZ writer can support trajectory data."""
        return True
    
    def write_trajectory(self, trajectory: TrajectoryData, file_path: Union[str, Path]) -> None:
        """Write a trajectory to multi-frame XYZ format."""
        logger.info(f"Writing XYZ trajectory: {file_path}")
        
        with open(file_path, 'w') as f:
            for frame_idx in range(trajectory.n_frames):
                # Write frame header
                f.write(f"{trajectory.n_atoms}\n")
                
                time_point = trajectory.time_points[frame_idx] if len(trajectory.time_points) > frame_idx else frame_idx
                title = f"Frame {frame_idx}, time = {time_point:.3f} ps"
                f.write(f"{title}\n")
                
                # Write atoms for this frame
                for atom_idx in range(trajectory.n_atoms):
                    element = "C"  # Default element
                    if trajectory.topology and trajectory.topology.elements:
                        element = trajectory.topology.elements[atom_idx]
                    
                    x, y, z = trajectory.coordinates[frame_idx, atom_idx]
                    f.write(f"{element:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")


class NPZTrajectoryWriter(BaseFormatWriter):
    """Writer for NumPy compressed trajectory format."""
    
    def write_structure(self, structure: StructureData, file_path: Union[str, Path]) -> None:
        """Write a structure to NPZ format."""
        logger.info(f"Writing NPZ structure file: {file_path}")
        
        data = {
            'coordinates': structure.coordinates,
            'elements': structure.elements,
            'atom_names': structure.atom_names,
            'atom_ids': structure.atom_ids,
            'residue_names': structure.residue_names,
            'residue_ids': structure.residue_ids,
            'chain_ids': structure.chain_ids,
            'n_atoms': structure.n_atoms
        }
        
        if structure.occupancies:
            data['occupancies'] = structure.occupancies
        if structure.b_factors:
            data['b_factors'] = structure.b_factors
        if structure.charges:
            data['charges'] = structure.charges
        if structure.masses:
            data['masses'] = structure.masses
        if structure.box_vectors is not None:
            data['box_vectors'] = structure.box_vectors
        if structure.title:
            data['title'] = structure.title
        if structure.bonds:
            data['bonds'] = structure.bonds
        
        np.savez_compressed(file_path, **data)
    
    def can_write_trajectory(self) -> bool:
        """NPZ writer supports trajectory data."""
        return True
    
    def write_trajectory(self, trajectory: TrajectoryData, file_path: Union[str, Path]) -> None:
        """Write a trajectory to NPZ format."""
        logger.info(f"Writing NPZ trajectory: {file_path}")
        
        data = {
            'coordinates': trajectory.coordinates,
            'time_points': trajectory.time_points,
            'n_frames': trajectory.n_frames,
            'n_atoms': trajectory.n_atoms
        }
        
        if trajectory.velocities is not None:
            data['velocities'] = trajectory.velocities
        if trajectory.forces is not None:
            data['forces'] = trajectory.forces
        if trajectory.box_vectors is not None:
            data['box_vectors'] = trajectory.box_vectors
        if trajectory.title:
            data['title'] = trajectory.title
        
        # Include topology data if available
        if trajectory.topology:
            data['topology_elements'] = trajectory.topology.elements
            data['topology_atom_names'] = trajectory.topology.atom_names
            data['topology_residue_names'] = trajectory.topology.residue_names
            data['topology_residue_ids'] = trajectory.topology.residue_ids
            data['topology_chain_ids'] = trajectory.topology.chain_ids
        
        np.savez_compressed(file_path, **data)


class NPZTrajectoryReader(BaseFormatReader):
    """Reader for NumPy compressed trajectory format."""
    
    def read_structure(self, file_path: Union[str, Path]) -> StructureData:
        """Read a structure from NPZ format."""
        logger.info(f"Reading NPZ structure: {file_path}")
        
        data = np.load(file_path, allow_pickle=True)
        
        # Required fields
        coordinates = data['coordinates']
        elements = data['elements'].tolist()
        atom_names = data['atom_names'].tolist()
        atom_ids = data['atom_ids'].tolist()
        residue_names = data['residue_names'].tolist()
        residue_ids = data['residue_ids'].tolist()
        chain_ids = data['chain_ids'].tolist()
        
        # Optional fields
        occupancies = data['occupancies'].tolist() if 'occupancies' in data else None
        b_factors = data['b_factors'].tolist() if 'b_factors' in data else None
        charges = data['charges'].tolist() if 'charges' in data else None
        masses = data['masses'].tolist() if 'masses' in data else None
        box_vectors = data['box_vectors'] if 'box_vectors' in data else None
        title = str(data['title']) if 'title' in data else None
        bonds = data['bonds'].tolist() if 'bonds' in data else None
        
        return StructureData(
            coordinates=coordinates,
            elements=elements,
            atom_names=atom_names,
            atom_ids=atom_ids,
            residue_names=residue_names,
            residue_ids=residue_ids,
            chain_ids=chain_ids,
            occupancies=occupancies,
            b_factors=b_factors,
            charges=charges,
            masses=masses,
            box_vectors=box_vectors,
            title=title,
            bonds=bonds
        )
    
    def can_read_trajectory(self) -> bool:
        """NPZ reader supports trajectory data."""
        return True
    
    def read_trajectory(self, file_path: Union[str, Path]) -> TrajectoryData:
        """Read a trajectory from NPZ format."""
        logger.info(f"Reading NPZ trajectory: {file_path}")
        
        data = np.load(file_path, allow_pickle=True)
        
        # Required fields
        coordinates = data['coordinates']
        time_points = data['time_points']
        
        # Optional fields
        velocities = data['velocities'] if 'velocities' in data else None
        forces = data['forces'] if 'forces' in data else None
        box_vectors = data['box_vectors'] if 'box_vectors' in data else None
        title = str(data['title']) if 'title' in data else None
        
        # Reconstruct topology if available
        topology = None
        if 'topology_elements' in data:
            topology = StructureData(
                coordinates=coordinates[0],  # First frame coordinates
                elements=data['topology_elements'].tolist(),
                atom_names=data['topology_atom_names'].tolist(),
                atom_ids=list(range(1, len(data['topology_elements']) + 1)),
                residue_names=data['topology_residue_names'].tolist(),
                residue_ids=data['topology_residue_ids'].tolist(),
                chain_ids=data['topology_chain_ids'].tolist()
            )
        
        return TrajectoryData(
            coordinates=coordinates,
            time_points=time_points,
            velocities=velocities,
            forces=forces,
            box_vectors=box_vectors,
            topology=topology,
            title=title
        )


class MultiFormatIO:
    """Main class for multi-format I/O operations."""
    
    def __init__(self):
        """Initialize the multi-format I/O system."""
        # Import additional format readers/writers
        try:
            from .mol2_format import MOL2Reader, MOL2Writer
            mol2_available = True
        except ImportError:
            mol2_available = False
            logger.warning("MOL2 format support not available")
        
        try:
            from .mmcif_format import mmCIFReader, mmCIFWriter
            mmcif_available = True
        except ImportError:
            mmcif_available = False
            logger.warning("mmCIF format support not available")
        
        try:
            from .binary_trajectory_formats import (
                DCDReader, DCDWriter, XTCReader, TRRReader,
                SimpleXTCWriter, SimpleTRRWriter
            )
            binary_traj_available = True
        except ImportError:
            binary_traj_available = False
            logger.warning("Binary trajectory format support not available")
        
        self._readers = {
            FormatType.PDB: PDBReader(),
            FormatType.XYZ: XYZReader(),
            FormatType.GRO: GROReader(),
            FormatType.NPZ: NPZTrajectoryReader(),
        }
        
        self._writers = {
            FormatType.PDB: PDBWriter(),
            FormatType.XYZ: XYZWriter(),
            FormatType.NPZ: NPZTrajectoryWriter(),
        }
        
        # Add optional formats if available
        if mol2_available:
            self._readers[FormatType.MOL2] = MOL2Reader()
            self._writers[FormatType.MOL2] = MOL2Writer()
        
        if mmcif_available:
            self._readers[FormatType.PDBX_MMCIF] = mmCIFReader()
            self._writers[FormatType.PDBX_MMCIF] = mmCIFWriter()
        
        if binary_traj_available:
            self._readers[FormatType.DCD] = DCDReader()
            self._writers[FormatType.DCD] = DCDWriter()
            
            # XTC and TRR have limited support
            self._readers[FormatType.XTC] = XTCReader()
            self._readers[FormatType.TRR] = TRRReader()
            # Writers are placeholders for now
            # self._writers[FormatType.XTC] = SimpleXTCWriter()
            # self._writers[FormatType.TRR] = SimpleTRRWriter()
        
        logger.info("Multi-format I/O system initialized")
        logger.info(f"Supported read formats: {list(self._readers.keys())}")
        logger.info(f"Supported write formats: {list(self._writers.keys())}")
    
    def read_structure(self, file_path: Union[str, Path]) -> StructureData:
        """
        Read a molecular structure from file with automatic format detection.
        
        Args:
            file_path: Path to the structure file
            
        Returns:
            StructureData object containing the molecular structure
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Structure file not found: {file_path}")
        
        format_type = FormatDetector.detect_format(file_path)
        
        if format_type not in self._readers:
            raise ValueError(f"Unsupported read format: {format_type}")
        
        reader = self._readers[format_type]
        structure = reader.read_structure(file_path)
        
        logger.info(f"Successfully read structure: {structure.n_atoms} atoms, "
                   f"{structure.n_residues} residues, {structure.n_chains} chains")
        
        return structure
    
    def read_trajectory(self, file_path: Union[str, Path]) -> TrajectoryData:
        """
        Read a molecular trajectory from file with automatic format detection.
        
        Args:
            file_path: Path to the trajectory file
            
        Returns:
            TrajectoryData object containing the molecular trajectory
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {file_path}")
        
        format_type = FormatDetector.detect_format(file_path)
        
        if format_type not in self._readers:
            raise ValueError(f"Unsupported read format: {format_type}")
        
        reader = self._readers[format_type]
        
        if not reader.can_read_trajectory():
            raise ValueError(f"Format {format_type} does not support trajectory data")
        
        trajectory = reader.read_trajectory(file_path)
        
        logger.info(f"Successfully read trajectory: {trajectory.n_frames} frames, "
                   f"{trajectory.n_atoms} atoms, {trajectory.simulation_time:.2f} ps")
        
        return trajectory
    
    def write_structure(self, structure: StructureData, file_path: Union[str, Path],
                       format_type: Optional[FormatType] = None) -> None:
        """
        Write a molecular structure to file.
        
        Args:
            structure: StructureData object to write
            file_path: Output file path
            format_type: Output format (auto-detected if None)
        """
        file_path = Path(file_path)
        
        if format_type is None:
            format_type = FormatDetector.detect_format(file_path)
        
        if format_type not in self._writers:
            raise ValueError(f"Unsupported write format: {format_type}")
        
        # Create output directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        writer = self._writers[format_type]
        writer.write_structure(structure, file_path)
        
        logger.info(f"Successfully wrote structure to {file_path} ({format_type})")
    
    def write_trajectory(self, trajectory: TrajectoryData, file_path: Union[str, Path],
                        format_type: Optional[FormatType] = None) -> None:
        """
        Write a molecular trajectory to file.
        
        Args:
            trajectory: TrajectoryData object to write
            file_path: Output file path
            format_type: Output format (auto-detected if None)
        """
        file_path = Path(file_path)
        
        if format_type is None:
            format_type = FormatDetector.detect_format(file_path)
        
        if format_type not in self._writers:
            raise ValueError(f"Unsupported write format: {format_type}")
        
        writer = self._writers[format_type]
        
        if not writer.can_write_trajectory():
            raise ValueError(f"Format {format_type} does not support trajectory data")
        
        # Create output directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        writer.write_trajectory(trajectory, file_path)
        
        logger.info(f"Successfully wrote trajectory to {file_path} ({format_type})")
    
    def convert_structure(self, input_path: Union[str, Path], output_path: Union[str, Path],
                         output_format: Optional[FormatType] = None) -> None:
        """
        Convert a structure file from one format to another.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            output_format: Target format (auto-detected if None)
        """
        logger.info(f"Converting structure: {input_path} -> {output_path}")
        
        # Read structure
        structure = self.read_structure(input_path)
        
        # Write in new format
        self.write_structure(structure, output_path, output_format)
        
        logger.info("Structure conversion completed successfully")
    
    def convert_trajectory(self, input_path: Union[str, Path], output_path: Union[str, Path],
                          output_format: Optional[FormatType] = None) -> None:
        """
        Convert a trajectory file from one format to another.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            output_format: Target format (auto-detected if None)
        """
        logger.info(f"Converting trajectory: {input_path} -> {output_path}")
        
        # Read trajectory
        trajectory = self.read_trajectory(input_path)
        
        # Write in new format
        self.write_trajectory(trajectory, output_path, output_format)
        
        logger.info("Trajectory conversion completed successfully")
    
    def get_supported_formats(self) -> Dict[str, List[FormatType]]:
        """Get lists of supported formats for reading and writing."""
        return {
            'read_structure': [fmt for fmt, reader in self._readers.items()],
            'read_trajectory': [fmt for fmt, reader in self._readers.items() if reader.can_read_trajectory()],
            'write_structure': [fmt for fmt, writer in self._writers.items()],
            'write_trajectory': [fmt for fmt, writer in self._writers.items() if writer.can_write_trajectory()]
        }
    
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a molecular file and return information about it.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Dictionary with file information and validation results
        """
        file_path = Path(file_path)
        
        validation_info = {
            'file_path': str(file_path),
            'exists': file_path.exists(),
            'size_bytes': file_path.stat().st_size if file_path.exists() else 0,
            'detected_format': None,
            'is_structure': False,
            'is_trajectory': False,
            'validation_errors': []
        }
        
        if not file_path.exists():
            validation_info['validation_errors'].append("File does not exist")
            return validation_info
        
        try:
            # Detect format
            format_type = FormatDetector.detect_format(file_path)
            validation_info['detected_format'] = format_type
            
            # Check if format is supported
            if format_type in self._readers:
                reader = self._readers[format_type]
                
                # Try to read as structure
                try:
                    structure = reader.read_structure(file_path)
                    validation_info['is_structure'] = True
                    validation_info['n_atoms'] = structure.n_atoms
                    validation_info['n_residues'] = structure.n_residues
                    validation_info['n_chains'] = structure.n_chains
                except Exception as e:
                    validation_info['validation_errors'].append(f"Structure reading failed: {e}")
                
                # Try to read as trajectory if supported
                if reader.can_read_trajectory():
                    try:
                        trajectory = reader.read_trajectory(file_path)
                        validation_info['is_trajectory'] = True
                        validation_info['n_frames'] = trajectory.n_frames
                        validation_info['simulation_time'] = trajectory.simulation_time
                    except Exception as e:
                        validation_info['validation_errors'].append(f"Trajectory reading failed: {e}")
            else:
                validation_info['validation_errors'].append(f"Unsupported format: {format_type}")
        
        except Exception as e:
            validation_info['validation_errors'].append(f"General validation error: {e}")
        
        return validation_info


# Convenience functions for easy access

def read_structure(file_path: Union[str, Path]) -> StructureData:
    """
    Convenience function to read a molecular structure.
    
    Args:
        file_path: Path to the structure file
        
    Returns:
        StructureData object
    """
    io_system = MultiFormatIO()
    return io_system.read_structure(file_path)


def read_trajectory(file_path: Union[str, Path]) -> TrajectoryData:
    """
    Convenience function to read a molecular trajectory.
    
    Args:
        file_path: Path to the trajectory file
        
    Returns:
        TrajectoryData object
    """
    io_system = MultiFormatIO()
    return io_system.read_trajectory(file_path)


def write_structure(structure: StructureData, file_path: Union[str, Path],
                   format_type: Optional[FormatType] = None) -> None:
    """
    Convenience function to write a molecular structure.
    
    Args:
        structure: StructureData object to write
        file_path: Output file path
        format_type: Output format (auto-detected if None)
    """
    io_system = MultiFormatIO()
    io_system.write_structure(structure, file_path, format_type)


def write_trajectory(trajectory: TrajectoryData, file_path: Union[str, Path],
                    format_type: Optional[FormatType] = None) -> None:
    """
    Convenience function to write a molecular trajectory.
    
    Args:
        trajectory: TrajectoryData object to write
        file_path: Output file path
        format_type: Output format (auto-detected if None)
    """
    io_system = MultiFormatIO()
    io_system.write_trajectory(trajectory, file_path, format_type)


def convert_file(input_path: Union[str, Path], output_path: Union[str, Path],
                output_format: Optional[FormatType] = None) -> None:
    """
    Convenience function to convert between file formats.
    
    Args:
        input_path: Input file path
        output_path: Output file path
        output_format: Target format (auto-detected if None)
    """
    io_system = MultiFormatIO()
    
    # Try structure conversion first
    try:
        io_system.convert_structure(input_path, output_path, output_format)
        return
    except Exception:
        pass
    
    # Try trajectory conversion
    try:
        io_system.convert_trajectory(input_path, output_path, output_format)
        return
    except Exception as e:
        raise ValueError(f"Could not convert file: {e}")


# Create test data function
def create_test_structure(n_atoms: int = 20) -> StructureData:
    """Create a test structure for validation purposes."""
    
    # Generate simple coordinates (linear chain)
    coordinates = []
    for i in range(n_atoms):
        x = i * 0.15  # 1.5 Å spacing
        y = 0.0
        z = 0.0
        coordinates.append([x, y, z])
    
    # Create atom data
    elements = ['C'] * n_atoms
    atom_names = [f'C{i+1}' for i in range(n_atoms)]
    atom_ids = list(range(1, n_atoms + 1))
    residue_names = ['UNK'] * n_atoms
    residue_ids = [1] * n_atoms
    chain_ids = ['A'] * n_atoms
    
    return StructureData(
        coordinates=np.array(coordinates),
        elements=elements,
        atom_names=atom_names,
        atom_ids=atom_ids,
        residue_names=residue_names,
        residue_ids=residue_ids,
        chain_ids=chain_ids,
        title="Test structure for multi-format I/O validation"
    )


def create_test_trajectory(n_frames: int = 10, n_atoms: int = 20) -> TrajectoryData:
    """Create a test trajectory for validation purposes."""
    
    # Create base structure
    base_structure = create_test_structure(n_atoms)
    
    # Generate trajectory frames with small perturbations
    frames = []
    time_points = []
    
    for frame in range(n_frames):
        # Start with base coordinates
        coords = base_structure.coordinates.copy()
        
        # Add small random perturbations
        noise = np.random.normal(0, 0.01, (n_atoms, 3))  # 0.1 Å RMS motion
        coords += noise
        
        frames.append(coords)
        time_points.append(frame * 1.0)  # 1 ps per frame
    
    return TrajectoryData(
        coordinates=np.array(frames),
        time_points=np.array(time_points),
        topology=base_structure,
        title="Test trajectory for multi-format I/O validation"
    )


if __name__ == "__main__":
    # Demo code
    print("🧬 ProteinMD Multi-Format I/O System")
    print("=" * 60)
    print("Supported Features:")
    print("✅ Multiple structure formats (PDB, XYZ, GRO)")
    print("✅ Multiple trajectory formats (XYZ, NPZ)")
    print("✅ Automatic format detection")
    print("✅ Format conversion utilities")
    print("✅ File validation and metadata extraction")
    print()
    print("Ready for integration with ProteinMD simulation engine!")
