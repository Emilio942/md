"""
Database module for molecular dynamics simulations.

This module provides classes and functions for storing and retrieving
simulation data, including trajectories, structures, and analysis results.
"""
import numpy as np
import sqlite3
import json
import pickle
import os
import time
import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import h5py

# Configure logging
logger = logging.getLogger(__name__)

class SimulationDatabase:
    """
    Base class for simulation databases.
    
    This class defines the interface for simulation database implementations.
    """
    
    def __init__(self, database_path: str):
        """
        Initialize a simulation database.
        
        Parameters
        ----------
        database_path : str
            Path to the database file or directory
        """
        self.database_path = database_path
    
    def initialize(self):
        """
        Initialize the database structure.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def store_simulation_metadata(self, simulation_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Store metadata for a simulation.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        metadata : dict
            Dictionary of metadata for the simulation
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def store_trajectory_frame(self, simulation_id: str, frame_index: int, 
                               positions: np.ndarray, velocities: Optional[np.ndarray] = None,
                               forces: Optional[np.ndarray] = None, 
                               energy: Optional[Dict[str, float]] = None,
                               time: Optional[float] = None,
                               box_vectors: Optional[np.ndarray] = None) -> bool:
        """
        Store a trajectory frame.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        frame_index : int
            Index of the frame in the trajectory
        positions : np.ndarray
            Particle positions with shape (n_particles, 3)
        velocities : np.ndarray, optional
            Particle velocities with shape (n_particles, 3)
        forces : np.ndarray, optional
            Particle forces with shape (n_particles, 3)
        energy : dict, optional
            Dictionary of energy components
        time : float, optional
            Simulation time in picoseconds
        box_vectors : np.ndarray, optional
            Periodic box vectors
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_simulation_metadata(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a simulation.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
            
        Returns
        -------
        dict or None
            Dictionary of metadata for the simulation, or None if not found
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_trajectory_frame(self, simulation_id: str, frame_index: int) -> Optional[Dict[str, Any]]:
        """
        Get a trajectory frame.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        frame_index : int
            Index of the frame to retrieve
            
        Returns
        -------
        dict or None
            Dictionary containing frame data, or None if not found
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_trajectory_frames(self, simulation_id: str, start_frame: int = 0, 
                              end_frame: Optional[int] = None, stride: int = 1) -> List[Dict[str, Any]]:
        """
        Get multiple trajectory frames.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        start_frame : int, optional
            Index of the first frame to retrieve
        end_frame : int, optional
            Index of the last frame to retrieve (exclusive)
        stride : int, optional
            Stride between frames
            
        Returns
        -------
        list
            List of dictionaries containing frame data
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def list_simulations(self) -> List[str]:
        """
        List all simulations in the database.
        
        Returns
        -------
        list
            List of simulation IDs
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def delete_simulation(self, simulation_id: str) -> bool:
        """
        Delete a simulation from the database.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def store_analysis_result(self, simulation_id: str, analysis_type: str, 
                             result: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store an analysis result.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        analysis_type : str
            Type of analysis
        result : any
            Analysis result (will be serialized)
        metadata : dict, optional
            Dictionary of metadata for the analysis
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_analysis_result(self, simulation_id: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """
        Get an analysis result.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        analysis_type : str
            Type of analysis
            
        Returns
        -------
        dict or None
            Dictionary containing the analysis result and metadata, or None if not found
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def list_analysis_results(self, simulation_id: str) -> List[str]:
        """
        List all analysis results for a simulation.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
            
        Returns
        -------
        list
            List of analysis types
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def close(self):
        """
        Close the database connection.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SQLiteDatabase(SimulationDatabase):
    """
    SQLite implementation of simulation database.
    
    This implementation uses SQLite for metadata and small data,
    and stores large arrays in binary files.
    """
    
    def __init__(self, database_path: str):
        """
        Initialize an SQLite simulation database.
        
        Parameters
        ----------
        database_path : str
            Path to the database file
        """
        super().__init__(database_path)
        self.conn = None
        self.data_dir = f"{os.path.splitext(database_path)[0]}_data"
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def initialize(self):
        """
        Initialize the database structure.
        """
        self.conn = sqlite3.connect(self.database_path)
        
        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Create tables
        with self.conn:
            # Table for simulations
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS simulations (
                id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                created_at TEXT,
                num_particles INTEGER,
                metadata TEXT
            )
            """)
            
            # Table for trajectory frames
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trajectory_frames (
                simulation_id TEXT,
                frame_index INTEGER,
                time REAL,
                data_path TEXT,
                energy TEXT,
                PRIMARY KEY (simulation_id, frame_index),
                FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE
            )
            """)
            
            # Table for analysis results
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                simulation_id TEXT,
                analysis_type TEXT,
                result_path TEXT,
                metadata TEXT,
                created_at TEXT,
                PRIMARY KEY (simulation_id, analysis_type),
                FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE
            )
            """)
        
        logger.info(f"Initialized SQLite database at {self.database_path}")
    
    def store_simulation_metadata(self, simulation_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Store metadata for a simulation.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        metadata : dict
            Dictionary of metadata for the simulation
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self.conn is None:
            self.initialize()
            
        try:
            with self.conn:
                # Check if simulation exists
                cursor = self.conn.execute("SELECT id FROM simulations WHERE id = ?", (simulation_id,))
                if cursor.fetchone() is None:
                    # Insert new simulation
                    self.conn.execute("""
                    INSERT INTO simulations (id, name, description, created_at, num_particles, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        simulation_id,
                        metadata.get('name', ''),
                        metadata.get('description', ''),
                        datetime.datetime.now().isoformat(),
                        metadata.get('num_particles', 0),
                        json.dumps(metadata)
                    ))
                else:
                    # Update existing simulation
                    self.conn.execute("""
                    UPDATE simulations SET name = ?, description = ?, num_particles = ?, metadata = ?
                    WHERE id = ?
                    """, (
                        metadata.get('name', ''),
                        metadata.get('description', ''),
                        metadata.get('num_particles', 0),
                        json.dumps(metadata),
                        simulation_id
                    ))
            
            return True
        except Exception as e:
            logger.error(f"Error storing simulation metadata: {e}")
            return False
    
    def store_trajectory_frame(self, simulation_id: str, frame_index: int, 
                              positions: np.ndarray, velocities: Optional[np.ndarray] = None,
                              forces: Optional[np.ndarray] = None, 
                              energy: Optional[Dict[str, float]] = None,
                              time: Optional[float] = None,
                              box_vectors: Optional[np.ndarray] = None) -> bool:
        """
        Store a trajectory frame.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        frame_index : int
            Index of the frame in the trajectory
        positions : np.ndarray
            Particle positions with shape (n_particles, 3)
        velocities : np.ndarray, optional
            Particle velocities with shape (n_particles, 3)
        forces : np.ndarray, optional
            Particle forces with shape (n_particles, 3)
        energy : dict, optional
            Dictionary of energy components
        time : float, optional
            Simulation time in picoseconds
        box_vectors : np.ndarray, optional
            Periodic box vectors
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self.conn is None:
            self.initialize()
            
        try:
            # Create frame data directory
            sim_dir = os.path.join(self.data_dir, simulation_id)
            os.makedirs(sim_dir, exist_ok=True)
            
            # Save frame data to a binary file
            data_path = os.path.join(sim_dir, f"frame_{frame_index}.npz")
            data_dict = {
                'positions': positions
            }
            
            if velocities is not None:
                data_dict['velocities'] = velocities
            
            if forces is not None:
                data_dict['forces'] = forces
            
            if box_vectors is not None:
                data_dict['box_vectors'] = box_vectors
            
            np.savez_compressed(data_path, **data_dict)
            
            # Store frame metadata in database
            with self.conn:
                self.conn.execute("""
                INSERT OR REPLACE INTO trajectory_frames (simulation_id, frame_index, time, data_path, energy)
                VALUES (?, ?, ?, ?, ?)
                """, (
                    simulation_id,
                    frame_index,
                    time if time is not None else frame_index,
                    data_path,
                    json.dumps(energy) if energy is not None else None
                ))
            
            return True
        except Exception as e:
            logger.error(f"Error storing trajectory frame: {e}")
            return False
    
    def get_simulation_metadata(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a simulation.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
            
        Returns
        -------
        dict or None
            Dictionary of metadata for the simulation, or None if not found
        """
        if self.conn is None:
            self.initialize()
            
        try:
            cursor = self.conn.execute("""
            SELECT metadata FROM simulations WHERE id = ?
            """, (simulation_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
                
            return json.loads(row[0])
        except Exception as e:
            logger.error(f"Error getting simulation metadata: {e}")
            return None
    
    def get_trajectory_frame(self, simulation_id: str, frame_index: int) -> Optional[Dict[str, Any]]:
        """
        Get a trajectory frame.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        frame_index : int
            Index of the frame to retrieve
            
        Returns
        -------
        dict or None
            Dictionary containing frame data, or None if not found
        """
        if self.conn is None:
            self.initialize()
            
        try:
            cursor = self.conn.execute("""
            SELECT time, data_path, energy FROM trajectory_frames
            WHERE simulation_id = ? AND frame_index = ?
            """, (simulation_id, frame_index))
            
            row = cursor.fetchone()
            if row is None:
                return None
                
            time, data_path, energy_json = row
            
            # Load data from file
            if not os.path.exists(data_path):
                logger.error(f"Frame data file not found: {data_path}")
                return None
                
            data = np.load(data_path)
            
            result = {
                'frame_index': frame_index,
                'time': time,
                'positions': data['positions']
            }
            
            # Add optional data if available
            for key in ['velocities', 'forces', 'box_vectors']:
                if key in data:
                    result[key] = data[key]
            
            # Add energy data if available
            if energy_json is not None:
                result['energy'] = json.loads(energy_json)
            
            return result
        except Exception as e:
            logger.error(f"Error getting trajectory frame: {e}")
            return None
    
    def get_trajectory_frames(self, simulation_id: str, start_frame: int = 0, 
                             end_frame: Optional[int] = None, stride: int = 1) -> List[Dict[str, Any]]:
        """
        Get multiple trajectory frames.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        start_frame : int, optional
            Index of the first frame to retrieve
        end_frame : int, optional
            Index of the last frame to retrieve (exclusive)
        stride : int, optional
            Stride between frames
            
        Returns
        -------
        list
            List of dictionaries containing frame data
        """
        if self.conn is None:
            self.initialize()
            
        try:
            # Get frame range
            if end_frame is None:
                cursor = self.conn.execute("""
                SELECT MAX(frame_index) FROM trajectory_frames WHERE simulation_id = ?
                """, (simulation_id,))
                
                row = cursor.fetchone()
                if row[0] is None:
                    return []
                    
                end_frame = row[0] + 1
            
            # Get frame data
            frames = []
            for i in range(start_frame, end_frame, stride):
                frame = self.get_trajectory_frame(simulation_id, i)
                if frame is not None:
                    frames.append(frame)
            
            return frames
        except Exception as e:
            logger.error(f"Error getting trajectory frames: {e}")
            return []
    
    def list_simulations(self) -> List[str]:
        """
        List all simulations in the database.
        
        Returns
        -------
        list
            List of simulation IDs
        """
        if self.conn is None:
            self.initialize()
            
        try:
            cursor = self.conn.execute("SELECT id FROM simulations")
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error listing simulations: {e}")
            return []
    
    def delete_simulation(self, simulation_id: str) -> bool:
        """
        Delete a simulation from the database.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self.conn is None:
            self.initialize()
            
        try:
            # Delete simulation data files
            sim_dir = os.path.join(self.data_dir, simulation_id)
            if os.path.exists(sim_dir):
                for file_name in os.listdir(sim_dir):
                    file_path = os.path.join(sim_dir, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(sim_dir)
            
            # Delete from database
            with self.conn:
                self.conn.execute("DELETE FROM simulations WHERE id = ?", (simulation_id,))
            
            return True
        except Exception as e:
            logger.error(f"Error deleting simulation: {e}")
            return False
    
    def store_analysis_result(self, simulation_id: str, analysis_type: str, 
                             result: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store an analysis result.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        analysis_type : str
            Type of analysis
        result : any
            Analysis result (will be serialized)
        metadata : dict, optional
            Dictionary of metadata for the analysis
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self.conn is None:
            self.initialize()
            
        try:
            # Create analysis data directory
            sim_dir = os.path.join(self.data_dir, simulation_id)
            os.makedirs(sim_dir, exist_ok=True)
            
            # Save analysis result to a file
            result_path = os.path.join(sim_dir, f"analysis_{analysis_type}.pkl")
            with open(result_path, 'wb') as f:
                pickle.dump(result, f)
            
            # Store metadata in database
            with self.conn:
                self.conn.execute("""
                INSERT OR REPLACE INTO analysis_results 
                (simulation_id, analysis_type, result_path, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                """, (
                    simulation_id,
                    analysis_type,
                    result_path,
                    json.dumps(metadata) if metadata is not None else None,
                    datetime.datetime.now().isoformat()
                ))
            
            return True
        except Exception as e:
            logger.error(f"Error storing analysis result: {e}")
            return False
    
    def get_analysis_result(self, simulation_id: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """
        Get an analysis result.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        analysis_type : str
            Type of analysis
            
        Returns
        -------
        dict or None
            Dictionary containing the analysis result and metadata, or None if not found
        """
        if self.conn is None:
            self.initialize()
            
        try:
            cursor = self.conn.execute("""
            SELECT result_path, metadata, created_at FROM analysis_results
            WHERE simulation_id = ? AND analysis_type = ?
            """, (simulation_id, analysis_type))
            
            row = cursor.fetchone()
            if row is None:
                return None
                
            result_path, metadata_json, created_at = row
            
            # Load result from file
            if not os.path.exists(result_path):
                logger.error(f"Analysis result file not found: {result_path}")
                return None
                
            with open(result_path, 'rb') as f:
                result = pickle.load(f)
            
            return {
                'result': result,
                'metadata': json.loads(metadata_json) if metadata_json is not None else {},
                'created_at': created_at
            }
        except Exception as e:
            logger.error(f"Error getting analysis result: {e}")
            return None
    
    def list_analysis_results(self, simulation_id: str) -> List[str]:
        """
        List all analysis results for a simulation.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
            
        Returns
        -------
        list
            List of analysis types
        """
        if self.conn is None:
            self.initialize()
            
        try:
            cursor = self.conn.execute("""
            SELECT analysis_type FROM analysis_results WHERE simulation_id = ?
            """, (simulation_id,))
            
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error listing analysis results: {e}")
            return []
    
    def close(self):
        """
        Close the database connection.
        """
        if self.conn is not None:
            self.conn.close()
            self.conn = None


class HDF5Database(SimulationDatabase):
    """
    HDF5 implementation of simulation database.
    
    This implementation stores all data in a single HDF5 file,
    which is efficient for storing large arrays and provides
    good compression.
    """
    
    def __init__(self, database_path: str):
        """
        Initialize an HDF5 simulation database.
        
        Parameters
        ----------
        database_path : str
            Path to the HDF5 file
        """
        super().__init__(database_path)
        self.file = None
    
    def initialize(self):
        """
        Initialize the database structure.
        """
        # Open file in append mode ('a'), create if it doesn't exist
        self.file = h5py.File(self.database_path, 'a')
        
        # Create top-level groups if they don't exist
        if 'simulations' not in self.file:
            self.file.create_group('simulations')
        
        logger.info(f"Initialized HDF5 database at {self.database_path}")
    
    def store_simulation_metadata(self, simulation_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Store metadata for a simulation.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        metadata : dict
            Dictionary of metadata for the simulation
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self.file is None:
            self.initialize()
            
        try:
            simulations = self.file['simulations']
            
            # Create simulation group if it doesn't exist
            if simulation_id not in simulations:
                sim_group = simulations.create_group(simulation_id)
                
                # Create trajectory group
                sim_group.create_group('trajectory')
                
                # Create analysis group
                sim_group.create_group('analysis')
            else:
                sim_group = simulations[simulation_id]
            
            # Store metadata as attributes
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool, np.number)):
                    sim_group.attrs[key] = value
            
            # Store full metadata as a serialized JSON string
            sim_group.attrs['_metadata_json'] = json.dumps(metadata)
            sim_group.attrs['_created_at'] = datetime.datetime.now().isoformat()
            
            self.file.flush()
            return True
        except Exception as e:
            logger.error(f"Error storing simulation metadata: {e}")
            return False
    
    def store_trajectory_frame(self, simulation_id: str, frame_index: int, 
                              positions: np.ndarray, velocities: Optional[np.ndarray] = None,
                              forces: Optional[np.ndarray] = None, 
                              energy: Optional[Dict[str, float]] = None,
                              time: Optional[float] = None,
                              box_vectors: Optional[np.ndarray] = None) -> bool:
        """
        Store a trajectory frame.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        frame_index : int
            Index of the frame in the trajectory
        positions : np.ndarray
            Particle positions with shape (n_particles, 3)
        velocities : np.ndarray, optional
            Particle velocities with shape (n_particles, 3)
        forces : np.ndarray, optional
            Particle forces with shape (n_particles, 3)
        energy : dict, optional
            Dictionary of energy components
        time : float, optional
            Simulation time in picoseconds
        box_vectors : np.ndarray, optional
            Periodic box vectors
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self.file is None:
            self.initialize()
            
        try:
            # Get simulation group
            sim_group = self.file['simulations'].get(simulation_id)
            if sim_group is None:
                logger.error(f"Simulation {simulation_id} not found")
                return False
            
            # Get trajectory group
            traj_group = sim_group['trajectory']
            
            # Create frame group
            frame_name = f"frame_{frame_index}"
            if frame_name in traj_group:
                # Replace existing frame
                del traj_group[frame_name]
            
            frame_group = traj_group.create_group(frame_name)
            
            # Store positions
            frame_group.create_dataset('positions', data=positions, compression='gzip')
            
            # Store optional data
            if velocities is not None:
                frame_group.create_dataset('velocities', data=velocities, compression='gzip')
            
            if forces is not None:
                frame_group.create_dataset('forces', data=forces, compression='gzip')
            
            if box_vectors is not None:
                frame_group.create_dataset('box_vectors', data=box_vectors)
            
            # Store metadata
            frame_group.attrs['frame_index'] = frame_index
            frame_group.attrs['time'] = time if time is not None else frame_index
            
            # Store energy data
            if energy is not None:
                for key, value in energy.items():
                    frame_group.attrs[f"energy_{key}"] = value
                frame_group.attrs['_energy_json'] = json.dumps(energy)
            
            self.file.flush()
            return True
        except Exception as e:
            logger.error(f"Error storing trajectory frame: {e}")
            return False
    
    def get_simulation_metadata(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a simulation.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
            
        Returns
        -------
        dict or None
            Dictionary of metadata for the simulation, or None if not found
        """
        if self.file is None:
            self.initialize()
            
        try:
            sim_group = self.file['simulations'].get(simulation_id)
            if sim_group is None:
                return None
            
            # Get full metadata from JSON string
            if '_metadata_json' in sim_group.attrs:
                return json.loads(sim_group.attrs['_metadata_json'])
            
            # Fall back to individual attributes
            metadata = {}
            for key, value in sim_group.attrs.items():
                if not key.startswith('_'):
                    metadata[key] = value
            
            return metadata
        except Exception as e:
            logger.error(f"Error getting simulation metadata: {e}")
            return None
    
    def get_trajectory_frame(self, simulation_id: str, frame_index: int) -> Optional[Dict[str, Any]]:
        """
        Get a trajectory frame.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        frame_index : int
            Index of the frame to retrieve
            
        Returns
        -------
        dict or None
            Dictionary containing frame data, or None if not found
        """
        if self.file is None:
            self.initialize()
            
        try:
            sim_group = self.file['simulations'].get(simulation_id)
            if sim_group is None:
                return None
            
            traj_group = sim_group['trajectory']
            frame_name = f"frame_{frame_index}"
            
            if frame_name not in traj_group:
                return None
            
            frame_group = traj_group[frame_name]
            
            # Build result dictionary
            result = {
                'frame_index': frame_index,
                'time': frame_group.attrs['time'],
                'positions': frame_group['positions'][()]
            }
            
            # Add optional data if available
            for key in ['velocities', 'forces', 'box_vectors']:
                if key in frame_group:
                    result[key] = frame_group[key][()]
            
            # Add energy data if available
            if '_energy_json' in frame_group.attrs:
                result['energy'] = json.loads(frame_group.attrs['_energy_json'])
            
            return result
        except Exception as e:
            logger.error(f"Error getting trajectory frame: {e}")
            return None
    
    def get_trajectory_frames(self, simulation_id: str, start_frame: int = 0, 
                             end_frame: Optional[int] = None, stride: int = 1) -> List[Dict[str, Any]]:
        """
        Get multiple trajectory frames.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        start_frame : int, optional
            Index of the first frame to retrieve
        end_frame : int, optional
            Index of the last frame to retrieve (exclusive)
        stride : int, optional
            Stride between frames
            
        Returns
        -------
        list
            List of dictionaries containing frame data
        """
        if self.file is None:
            self.initialize()
            
        try:
            sim_group = self.file['simulations'].get(simulation_id)
            if sim_group is None:
                return []
            
            traj_group = sim_group['trajectory']
            
            # Get frame indices
            frame_indices = []
            for name in traj_group:
                if name.startswith('frame_'):
                    try:
                        index = int(name.split('_')[1])
                        frame_indices.append(index)
                    except ValueError:
                        continue
            
            frame_indices.sort()
            
            # Apply range and stride
            if end_frame is None:
                end_frame = max(frame_indices) + 1 if frame_indices else 0
                
            selected_indices = [i for i in frame_indices if start_frame <= i < end_frame and (i - start_frame) % stride == 0]
            
            # Get frames
            frames = []
            for index in selected_indices:
                frame = self.get_trajectory_frame(simulation_id, index)
                if frame is not None:
                    frames.append(frame)
            
            return frames
        except Exception as e:
            logger.error(f"Error getting trajectory frames: {e}")
            return []
    
    def list_simulations(self) -> List[str]:
        """
        List all simulations in the database.
        
        Returns
        -------
        list
            List of simulation IDs
        """
        if self.file is None:
            self.initialize()
            
        try:
            return list(self.file['simulations'].keys())
        except Exception as e:
            logger.error(f"Error listing simulations: {e}")
            return []
    
    def delete_simulation(self, simulation_id: str) -> bool:
        """
        Delete a simulation from the database.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self.file is None:
            self.initialize()
            
        try:
            if simulation_id in self.file['simulations']:
                del self.file['simulations'][simulation_id]
                self.file.flush()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting simulation: {e}")
            return False
    
    def store_analysis_result(self, simulation_id: str, analysis_type: str, 
                             result: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store an analysis result.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        analysis_type : str
            Type of analysis
        result : any
            Analysis result (will be serialized)
        metadata : dict, optional
            Dictionary of metadata for the analysis
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self.file is None:
            self.initialize()
            
        try:
            # Get simulation group
            sim_group = self.file['simulations'].get(simulation_id)
            if sim_group is None:
                logger.error(f"Simulation {simulation_id} not found")
                return False
            
            # Get analysis group
            analysis_group = sim_group['analysis']
            
            # Create or replace analysis result group
            if analysis_type in analysis_group:
                del analysis_group[analysis_type]
            
            result_group = analysis_group.create_group(analysis_type)
            
            # Store result based on type
            if isinstance(result, np.ndarray):
                result_group.create_dataset('result', data=result, compression='gzip')
            elif isinstance(result, (dict, list, tuple, str, int, float, bool)):
                result_group.attrs['_result_json'] = json.dumps(result)
            else:
                # Serialize arbitrary objects using pickle
                pickled = pickle.dumps(result)
                result_group.create_dataset('_pickled_result', data=np.void(pickled))
            
            # Store metadata
            if metadata is not None:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool, np.number)):
                        result_group.attrs[key] = value
                
                result_group.attrs['_metadata_json'] = json.dumps(metadata)
            
            result_group.attrs['_created_at'] = datetime.datetime.now().isoformat()
            
            self.file.flush()
            return True
        except Exception as e:
            logger.error(f"Error storing analysis result: {e}")
            return False
    
    def get_analysis_result(self, simulation_id: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """
        Get an analysis result.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        analysis_type : str
            Type of analysis
            
        Returns
        -------
        dict or None
            Dictionary containing the analysis result and metadata, or None if not found
        """
        if self.file is None:
            self.initialize()
            
        try:
            sim_group = self.file['simulations'].get(simulation_id)
            if sim_group is None:
                return None
            
            analysis_group = sim_group['analysis']
            if analysis_type not in analysis_group:
                return None
            
            result_group = analysis_group[analysis_type]
            
            # Get result based on how it was stored
            if 'result' in result_group:
                result = result_group['result'][()]
            elif '_result_json' in result_group.attrs:
                result = json.loads(result_group.attrs['_result_json'])
            elif '_pickled_result' in result_group:
                pickled = result_group['_pickled_result'][()]
                result = pickle.loads(pickled.tobytes())
            else:
                result = None
            
            # Get metadata
            if '_metadata_json' in result_group.attrs:
                metadata = json.loads(result_group.attrs['_metadata_json'])
            else:
                metadata = {}
                for key, value in result_group.attrs.items():
                    if not key.startswith('_'):
                        metadata[key] = value
            
            # Get created timestamp
            created_at = result_group.attrs.get('_created_at', None)
            
            return {
                'result': result,
                'metadata': metadata,
                'created_at': created_at
            }
        except Exception as e:
            logger.error(f"Error getting analysis result: {e}")
            return None
    
    def list_analysis_results(self, simulation_id: str) -> List[str]:
        """
        List all analysis results for a simulation.
        
        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
            
        Returns
        -------
        list
            List of analysis types
        """
        if self.file is None:
            self.initialize()
            
        try:
            sim_group = self.file['simulations'].get(simulation_id)
            if sim_group is None:
                return []
            
            analysis_group = sim_group['analysis']
            return list(analysis_group.keys())
        except Exception as e:
            logger.error(f"Error listing analysis results: {e}")
            return []
    
    def close(self):
        """
        Close the database connection.
        """
        if self.file is not None:
            self.file.close()
            self.file = None


class DatabaseFactory:
    """
    Factory class for creating database instances.
    """
    
    @staticmethod
    def create_database(database_type: str, database_path: str) -> SimulationDatabase:
        """
        Create a database instance.
        
        Parameters
        ----------
        database_type : str
            Type of database ('sqlite' or 'hdf5')
        database_path : str
            Path to the database file
        
        Returns
        -------
        SimulationDatabase
            Database instance
        
        Raises
        ------
        ValueError
            If the database type is invalid
        """
        if database_type.lower() == 'sqlite':
            return SQLiteDatabase(database_path)
        elif database_type.lower() in ('hdf5', 'h5'):
            return HDF5Database(database_path)
        else:
            raise ValueError(f"Invalid database type: {database_type}")


# Add a function to the global scope for convenience
def open_database(database_path: str, database_type: Optional[str] = None) -> SimulationDatabase:
    """
    Open a simulation database.
    
    Parameters
    ----------
    database_path : str
        Path to the database file
    database_type : str, optional
        Type of database. If None, inferred from file extension.
    
    Returns
    -------
    SimulationDatabase
        Database instance
    """
    if database_type is None:
        ext = os.path.splitext(database_path)[1].lower()
        if ext in ('.sqlite', '.db', '.sqlite3'):
            database_type = 'sqlite'
        elif ext in ('.h5', '.hdf5'):
            database_type = 'hdf5'
        else:
            raise ValueError(f"Could not infer database type from extension: {ext}")
    
    db = DatabaseFactory.create_database(database_type, database_path)
    db.initialize()
    return db
