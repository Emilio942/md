"""
ProteinMD Database Integration Package

Task 9.1: Database Integration 📊

This package provides comprehensive database integration capabilities for ProteinMD,
enabling structured storage, search, and management of simulation metadata with
support for both SQLite and PostgreSQL backends.

Key Features:
- SQLite/PostgreSQL backend support
- Simulation metadata storage and retrieval
- Advanced search and filtering capabilities
- Automated backup and restore strategies
- Database migration utilities
- CLI integration for database management
- RESTful API for external access

Components:
- models.py: Database schema and ORM models
- database_manager.py: Connection handling and database operations
- search_engine.py: Search and indexing functionality
- backup_manager.py: Backup and restore utilities
- migration_tools.py: Database migration and export/import
- database_cli.py: Command-line interface
- api_server.py: REST API for external access
"""

from .database_manager import DatabaseManager, DatabaseConfig
from .models import (
    SimulationRecord, AnalysisRecord, WorkflowRecord,
    ProteinStructure, TrajectoryMetadata, ForceFieldRecord
)
from .search_engine import SimulationSearchEngine, SearchQuery, SearchResult
from .backup_manager import BackupManager, BackupConfig
from .database_cli import DatabaseCLI

# Cloud storage integration (Task 9.2)
try:
    from .cloud_storage import (
        CloudStorageManager, CloudConfig, CachePolicy,
        CloudStorageProvider, AWSStorageProvider, GCSStorageProvider,
        EncryptionManager, LocalCache, CloudFile,
        create_cloud_storage_manager, auto_sync_large_files
    )
    HAS_CLOUD_STORAGE = True
except ImportError as e:
    HAS_CLOUD_STORAGE = False
    import logging
    logging.getLogger(__name__).warning(f"Cloud storage not available: {e}")

# Metadata management (Task 9.3)
from .metadata_manager import (
    MetadataManager, MetadataEntry, MetadataType, MetadataSchema,
    TagManager, Tag, ProvenanceTracker, ProvenanceEntry,
    MetadataQueryBuilder, DataLevel,
    create_simulation_metadata, track_analysis_provenance
)

__all__ = [
    'DatabaseManager',
    'DatabaseConfig',
    'SimulationRecord',
    'AnalysisRecord', 
    'WorkflowRecord',
    'ProteinStructure',
    'TrajectoryMetadata',
    'ForceFieldRecord',
    'SimulationSearchEngine',
    'SearchQuery',
    'SearchResult',
    'BackupManager',
    'BackupConfig',
    'DatabaseCLI',

    # Cloud storage (Task 9.2)
    'CloudStorageManager', 'CloudConfig', 'CachePolicy',
    'CloudStorageProvider', 'AWSStorageProvider', 'GCSStorageProvider', 
    'EncryptionManager', 'LocalCache', 'CloudFile',
    'create_cloud_storage_manager', 'auto_sync_large_files',
    
    # Metadata management (Task 9.3)
    'MetadataManager', 'MetadataEntry', 'MetadataType', 'MetadataSchema',
    'TagManager', 'Tag', 'ProvenanceTracker', 'ProvenanceEntry',
    'MetadataQueryBuilder', 'DataLevel',
    'create_simulation_metadata', 'track_analysis_provenance',
]

# Version information
__version__ = '1.2.0'  # Updated for Tasks 9.2 and 9.3
__author__ = 'ProteinMD Team'
__description__ = 'Comprehensive data management for molecular dynamics simulations'

# Module-level convenience functions
def create_database_manager(db_type: str = 'sqlite', **kwargs):
    """
    Create a database manager with default configuration.
    
    Parameters
    ----------
    db_type : str
        Database type ('sqlite' or 'hdf5')
    **kwargs
        Additional configuration parameters
        
    Returns
    -------
    DatabaseManager
        Configured database manager
    """
    return DatabaseManager(db_type=db_type, **kwargs)

def create_cloud_enabled_system(db_config: dict, cloud_config: dict):
    """
    Create a complete data management system with cloud integration.
    
    Parameters
    ----------
    db_config : dict
        Database configuration
    cloud_config : dict
        Cloud storage configuration
        
    Returns
    -------
    tuple
        (DatabaseManager, CloudStorageManager, MetadataManager)
    """
    if not HAS_CLOUD_STORAGE:
        raise ImportError("Cloud storage dependencies not available")
    
    # Create components
    db_manager = DatabaseManager(**db_config)
    
    cloud_config_obj = CloudConfig(**cloud_config)
    cloud_manager = CloudStorageManager(cloud_config_obj)
    
    metadata_manager = MetadataManager()
    
    return db_manager, cloud_manager, metadata_manager

# System information
def get_system_info():
    """Get information about available database system capabilities."""
    return {
        'version': __version__,
        'has_cloud_storage': HAS_CLOUD_STORAGE,
        'has_cli': HAS_CLI,
        'supported_backends': ['sqlite', 'hdf5'],
        'cloud_providers': ['aws', 'gcs'] if HAS_CLOUD_STORAGE else [],
        'metadata_features': [
            'automatic_extraction',
            'provenance_tracking', 
            'hierarchical_tags',
            'advanced_queries'
        ]
    }
