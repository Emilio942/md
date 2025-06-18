"""
Database Models and Schema Definition

This module defines the database schema for storing ProteinMD simulation metadata,
analysis results, and workflow information using SQLAlchemy ORM.
"""

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean, 
    JSON, ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid
import json
import uuid

Base = declarative_base()

class SimulationRecord(Base):
    """
    Main simulation record storing metadata about MD simulations.
    
    This table stores comprehensive metadata about each molecular dynamics
    simulation run, including input parameters, execution details, and results.
    """
    __tablename__ = 'simulations'
    
    # Primary key and identifiers
    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Temporal metadata
    created_at = Column(DateTime, nullable=False, default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    last_modified = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
    
    # Execution metadata
    status = Column(String(50), nullable=False, default='created')  # created, running, completed, failed, cancelled
    exit_code = Column(Integer)
    execution_time = Column(Float)  # seconds
    host_system = Column(String(255))
    working_directory = Column(Text)
    
    # Input parameters
    input_structure_file = Column(Text)
    input_structure_format = Column(String(10))  # pdb, mol2, etc.
    force_field = Column(String(100))
    water_model = Column(String(50))
    box_size = Column(JSON)  # [x, y, z] dimensions
    
    # Simulation parameters
    temperature = Column(Float)  # Kelvin
    pressure = Column(Float)    # bar
    timestep = Column(Float)    # ps
    total_steps = Column(Integer)
    total_time = Column(Float)  # ps
    output_frequency = Column(Integer)
    
    # Environment setup
    solvent_type = Column(String(50))  # explicit, implicit, vacuum
    periodic_boundary = Column(Boolean, default=True)
    constraints = Column(JSON)  # constraint parameters
    
    # Resource usage
    cpu_cores = Column(Integer)
    memory_usage_mb = Column(Float)
    gpu_devices = Column(JSON)  # list of GPU device IDs
    wall_clock_time = Column(Float)  # seconds
    
    # File paths and outputs
    output_directory = Column(Text)
    trajectory_file = Column(Text)
    energy_file = Column(Text)
    log_file = Column(Text)
    
    # Results summary
    final_energy = Column(Float)  # kJ/mol
    average_temperature = Column(Float)
    average_pressure = Column(Float)
    convergence_achieved = Column(Boolean)
    
    # User and project metadata
    user_id = Column(String(255))
    project_name = Column(String(255))
    tags = Column(JSON)  # list of tags
    notes = Column(Text)
    
    # Foreign keys
    protein_structure_id = Column(Integer, ForeignKey('protein_structures.id'))
    
    # Version and provenance
    proteinmd_version = Column(String(50))
    config_hash = Column(String(64))  # SHA256 of configuration
    reproducibility_data = Column(JSON)  # seeds, versions, etc.
    
    # Relationships
    analyses = relationship("AnalysisRecord", back_populates="simulation", cascade="all, delete-orphan")
    protein_structure = relationship("ProteinStructure", back_populates="simulations")
    trajectory_metadata = relationship("TrajectoryMetadata", back_populates="simulation", uselist=False)
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_simulation_status', 'status'),
        Index('idx_simulation_created', 'created_at'),
        Index('idx_simulation_user', 'user_id'),
        Index('idx_simulation_project', 'project_name'),
        Index('idx_simulation_force_field', 'force_field'),
        CheckConstraint('status IN ("created", "running", "completed", "failed", "cancelled")', name='status_check'),
        CheckConstraint('temperature > 0', name='temperature_positive'),
        CheckConstraint('total_steps > 0', name='steps_positive'),
    )
    
    @validates('status')
    def validate_status(self, key, status):
        allowed_statuses = ['created', 'running', 'completed', 'failed', 'cancelled']
        if status not in allowed_statuses:
            raise ValueError(f"Invalid status '{status}'. Must be one of: {allowed_statuses}")
        return status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for serialization."""
        return {
            'id': self.id,
            'simulation_id': self.simulation_id,
            'name': self.name,
            'description': self.description,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'execution_time': self.execution_time,
            'force_field': self.force_field,
            'temperature': self.temperature,
            'total_steps': self.total_steps,
            'final_energy': self.final_energy,
            'user_id': self.user_id,
            'project_name': self.project_name,
            'tags': self.tags,
        }

class AnalysisRecord(Base):
    """
    Analysis results and metadata for simulations.
    
    Stores information about various analyses performed on simulation data,
    including RMSD, Ramachandran plots, secondary structure, etc.
    """
    __tablename__ = 'analyses'
    
    # Primary key and relationships
    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    simulation_id = Column(Integer, ForeignKey('simulations.id'), nullable=False)
    
    # Analysis metadata
    analysis_type = Column(String(100), nullable=False)  # rmsd, ramachandran, rg, etc.
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Execution details
    created_at = Column(DateTime, nullable=False, default=func.now())
    completed_at = Column(DateTime)
    status = Column(String(50), nullable=False, default='created')
    execution_time = Column(Float)
    
    # Input parameters
    analysis_parameters = Column(JSON)
    time_range = Column(JSON)  # [start_time, end_time] in ps
    selection_criteria = Column(Text)  # atom selection string
    
    # Results
    output_files = Column(JSON)  # list of output file paths
    results_summary = Column(JSON)  # summary statistics
    raw_data_file = Column(Text)  # path to raw data file
    
    # Quality metrics
    convergence_achieved = Column(Boolean)
    statistical_significance = Column(Float)
    error_estimates = Column(JSON)
    
    # Visualization
    plot_files = Column(JSON)  # list of generated plot files
    visualization_config = Column(JSON)
    
    # Relationships
    simulation = relationship("SimulationRecord", back_populates="analyses")
    
    # Indexes
    __table_args__ = (
        Index('idx_analysis_simulation', 'simulation_id'),
        Index('idx_analysis_type', 'analysis_type'),
        Index('idx_analysis_created', 'created_at'),
        Index('idx_analysis_status', 'status'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis record to dictionary."""
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'simulation_id': self.simulation_id,
            'analysis_type': self.analysis_type,
            'name': self.name,
            'description': self.description,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'results_summary': self.results_summary,
        }

class WorkflowRecord(Base):
    """
    Workflow execution records and metadata.
    
    Stores information about automated workflow executions, including
    multi-step analysis pipelines and their execution status.
    """
    __tablename__ = 'workflows'
    
    # Primary key and identifiers
    id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Workflow definition
    workflow_file = Column(Text)  # path to workflow definition file
    workflow_version = Column(String(50))
    workflow_hash = Column(String(64))  # SHA256 of workflow definition
    
    # Execution metadata
    created_at = Column(DateTime, nullable=False, default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    status = Column(String(50), nullable=False, default='created')
    
    # Execution details
    total_steps = Column(Integer)
    completed_steps = Column(Integer, default=0)
    failed_steps = Column(Integer, default=0)
    output_directory = Column(Text)
    
    # Resource usage
    total_execution_time = Column(Float)  # seconds
    peak_memory_usage = Column(Float)  # MB
    cpu_hours = Column(Float)
    
    # Parameters and configuration
    global_parameters = Column(JSON)
    scheduler_config = Column(JSON)
    retry_config = Column(JSON)
    
    # Results
    final_report_file = Column(Text)
    output_artifacts = Column(JSON)  # list of output files/directories
    execution_log = Column(Text)  # path to execution log
    
    # User metadata
    user_id = Column(String(255))
    project_name = Column(String(255))
    tags = Column(JSON)
    
    # Indexes
    __table_args__ = (
        Index('idx_workflow_status', 'status'),
        Index('idx_workflow_created', 'created_at'),
        Index('idx_workflow_user', 'user_id'),
        Index('idx_workflow_project', 'project_name'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow record to dictionary."""
        return {
            'id': self.id,
            'workflow_id': self.workflow_id,
            'name': self.name,
            'description': self.description,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'total_steps': self.total_steps,
            'completed_steps': self.completed_steps,
            'user_id': self.user_id,
            'project_name': self.project_name,
        }

class ProteinStructure(Base):
    """
    Protein structure metadata and information.
    
    Stores information about protein structures used in simulations,
    including PDB information, sequence data, and structural features.
    """
    __tablename__ = 'protein_structures'
    
    # Primary key and identifiers
    id = Column(Integer, primary_key=True, autoincrement=True)
    structure_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # PDB information
    pdb_id = Column(String(10))  # official PDB ID if available
    structure_name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # File information
    file_path = Column(Text, nullable=False)
    file_format = Column(String(10), nullable=False)  # pdb, mol2, etc.
    file_size = Column(Integer)  # bytes
    file_hash = Column(String(64))  # SHA256 hash for integrity
    
    # Structure metadata
    created_at = Column(DateTime, nullable=False, default=func.now())
    resolution = Column(Float)  # Angstroms
    experimental_method = Column(String(100))  # X-ray, NMR, etc.
    organism = Column(String(255))
    
    # Sequence and composition
    sequence = Column(Text)  # amino acid sequence
    sequence_length = Column(Integer)
    molecular_weight = Column(Float)  # kDa
    
    # Structural features
    num_chains = Column(Integer)
    num_residues = Column(Integer)
    num_atoms = Column(Integer)
    secondary_structure_content = Column(JSON)  # helix, sheet, loop percentages
    
    # Classification
    protein_family = Column(String(255))
    enzyme_class = Column(String(50))  # EC number
    function_description = Column(Text)
    
    # Quality metrics
    structure_quality_score = Column(Float)
    validation_notes = Column(Text)
    
    # Relationships
    simulations = relationship("SimulationRecord", back_populates="protein_structure")
    
    # Indexes
    __table_args__ = (
        Index('idx_protein_pdb_id', 'pdb_id'),
        Index('idx_protein_name', 'structure_name'),
        Index('idx_protein_organism', 'organism'),
        Index('idx_protein_family', 'protein_family'),
        UniqueConstraint('file_path', 'file_hash', name='unique_structure_file'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert protein structure record to dictionary."""
        return {
            'id': self.id,
            'structure_id': self.structure_id,
            'pdb_id': self.pdb_id,
            'structure_name': self.structure_name,
            'description': self.description,
            'sequence_length': self.sequence_length,
            'molecular_weight': self.molecular_weight,
            'num_chains': self.num_chains,
            'num_residues': self.num_residues,
            'organism': self.organism,
            'protein_family': self.protein_family,
        }

class TrajectoryMetadata(Base):
    """
    Trajectory file metadata and statistics.
    
    Stores metadata about trajectory files including format, size,
    frame information, and computed statistics.
    """
    __tablename__ = 'trajectory_metadata'
    
    # Primary key and relationships
    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(Integer, ForeignKey('simulations.id'), nullable=False)
    
    # File information
    file_path = Column(Text, nullable=False)
    file_format = Column(String(20), nullable=False)  # dcd, xtc, trr, etc.
    file_size = Column(Integer)  # bytes
    file_hash = Column(String(64))  # SHA256 hash
    
    # Trajectory properties
    created_at = Column(DateTime, nullable=False, default=func.now())
    num_frames = Column(Integer)
    num_atoms = Column(Integer)
    frame_frequency = Column(Float)  # ps between frames
    total_time = Column(Float)  # ps
    
    # Data quality
    missing_frames = Column(JSON)  # list of missing frame indices
    corrupted_frames = Column(JSON)  # list of corrupted frame indices
    quality_score = Column(Float)
    
    # Statistics
    average_energy = Column(Float)  # kJ/mol
    energy_std = Column(Float)
    average_temperature = Column(Float)  # K
    temperature_std = Column(Float)
    
    # Compression and storage
    compression_ratio = Column(Float)
    compressed_size = Column(Integer)  # bytes
    storage_location = Column(String(255))  # local, cloud, archive
    
    # Relationships
    simulation = relationship("SimulationRecord", back_populates="trajectory_metadata")
    
    # Indexes
    __table_args__ = (
        Index('idx_trajectory_simulation', 'simulation_id'),
        Index('idx_trajectory_format', 'file_format'),
        Index('idx_trajectory_created', 'created_at'),
    )

class ForceFieldRecord(Base):
    """
    Force field definitions and parameters.
    
    Stores information about force field parameters used in simulations
    for reproducibility and validation purposes.
    """
    __tablename__ = 'force_fields'
    
    # Primary key and identifiers
    id = Column(Integer, primary_key=True, autoincrement=True)
    force_field_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # Force field metadata
    name = Column(String(100), nullable=False)
    version = Column(String(50))
    description = Column(Text)
    family = Column(String(50))  # AMBER, CHARMM, GROMOS, etc.
    
    # File information
    parameter_file = Column(Text)
    parameter_hash = Column(String(64))  # SHA256 of parameter file
    created_at = Column(DateTime, nullable=False, default=func.now())
    
    # Parameters summary
    num_atom_types = Column(Integer)
    num_bond_types = Column(Integer)
    num_angle_types = Column(Integer)
    num_dihedral_types = Column(Integer)
    
    # Validation
    validation_status = Column(String(50))  # validated, pending, failed
    validation_date = Column(DateTime)
    validation_notes = Column(Text)
    
    # References and citations
    reference_papers = Column(JSON)  # list of DOIs or citations
    original_authors = Column(Text)
    
    # Indexes
    __table_args__ = (
        Index('idx_forcefield_name', 'name'),
        Index('idx_forcefield_family', 'family'),
        Index('idx_forcefield_validation', 'validation_status'),
        UniqueConstraint('name', 'version', name='unique_forcefield_version'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert force field record to dictionary."""
        return {
            'id': self.id,
            'force_field_id': self.force_field_id,
            'name': self.name,
            'version': self.version,
            'family': self.family,
            'description': self.description,
            'validation_status': self.validation_status,
            'num_atom_types': self.num_atom_types,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
