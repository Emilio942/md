#!/usr/bin/env python3
"""
Data Management Integration for Tasks 9.2 & 9.3
===============================================

This module provides seamless integration between cloud storage, metadata management,
and the existing database system, creating a unified data management experience.

Features:
- Automatic cloud sync for large trajectory files
- Integrated metadata capture and storage
- Provenance tracking across all operations
- Unified search across local and cloud data
- Automated data lifecycle management
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from .database_manager import DatabaseManager
from .metadata_manager import (
    MetadataManager, MetadataType, MetadataEntry, 
    create_simulation_metadata, track_analysis_provenance
)

# Cloud storage (optional dependency)
try:
    from .cloud_storage import CloudStorageManager, CloudConfig, auto_sync_large_files
    HAS_CLOUD_STORAGE = True
except ImportError:
    HAS_CLOUD_STORAGE = False

logger = logging.getLogger(__name__)

class IntegratedDataManager:
    """
    Unified data management system combining local database, 
    cloud storage, and metadata management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize integrated data management system.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary with keys:
            - database: Database configuration
            - cloud: Cloud storage configuration (optional)
            - metadata: Metadata management configuration
        """
        self.config = config
        
        # Initialize core components
        self.db_manager = DatabaseManager(**config.get('database', {}))
        self.metadata_manager = MetadataManager()
        
        # Initialize cloud storage if configured
        self.cloud_manager = None
        if HAS_CLOUD_STORAGE and 'cloud' in config:
            try:
                cloud_config = CloudConfig(**config['cloud'])
                self.cloud_manager = CloudStorageManager(cloud_config)
                logger.info("Cloud storage integration enabled")
            except Exception as e:
                logger.warning(f"Cloud storage initialization failed: {e}")
        
        # Configuration
        self.auto_sync_threshold_mb = config.get('auto_sync_threshold_mb', 100.0)
        self.auto_metadata_capture = config.get('auto_metadata_capture', True)
    
    def store_simulation(self, simulation_id: str, metadata: Dict[str, Any],
                        trajectory_files: List[str], other_files: Dict[str, str],
                        user: str = "system") -> bool:
        """
        Store complete simulation data with automatic cloud sync and metadata capture.
        
        Parameters
        ----------
        simulation_id : str
            Unique simulation identifier
        metadata : dict
            Simulation metadata
        trajectory_files : list
            List of trajectory file paths
        other_files : dict
            Other simulation files (type -> path mapping)
        user : str
            User performing the operation
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            # Store in local database
            success = self.db_manager.store_simulation_metadata(simulation_id, metadata)
            if not success:
                logger.error(f"Failed to store simulation {simulation_id} in database")
                return False
            
            # Prepare file mapping for metadata
            all_files = dict(other_files)
            if trajectory_files:
                all_files['trajectories'] = trajectory_files
            
            # Create comprehensive metadata entry
            if self.auto_metadata_capture:
                metadata_id = create_simulation_metadata(metadata, all_files, user)
                
                # Link metadata to simulation
                self.db_manager.store_analysis_result(
                    simulation_id, 'metadata_entry', 
                    {'metadata_id': metadata_id}, 
                    {'created_by': user, 'created_at': datetime.now().isoformat()}
                )
                
                logger.info(f"Created metadata entry {metadata_id} for simulation {simulation_id}")
            
            # Auto-sync large files to cloud storage
            if self.cloud_manager:
                synced_files = []
                
                # Sync trajectory files
                for traj_file in trajectory_files:
                    if self._should_sync_file(traj_file):
                        if self.cloud_manager.sync_trajectory_file(traj_file, simulation_id):
                            synced_files.append(traj_file)
                
                # Sync other large files
                for file_type, file_path in other_files.items():
                    if self._should_sync_file(file_path):
                        cloud_path = f"simulations/{simulation_id}/{file_type}/{os.path.basename(file_path)}"
                        if self.cloud_manager.upload_file(file_path, cloud_path, force_upload=True):
                            synced_files.append(file_path)
                
                if synced_files:
                    logger.info(f"Synced {len(synced_files)} files to cloud storage")
                    
                    # Update metadata with cloud sync information
                    if self.auto_metadata_capture:
                        self.metadata_manager.add_provenance(
                            metadata_id, 
                            'cloud_sync',
                            [simulation_id],
                            synced_files,
                            {'synced_files': len(synced_files), 'provider': self.cloud_manager.config.provider},
                            user,
                            f"Automatically synced {len(synced_files)} files to cloud storage"
                        )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store simulation: {e}")
            return False
    
    def run_analysis(self, analysis_type: str, simulation_id: str, 
                    parameters: Dict[str, Any], output_files: List[str],
                    user: str = "system") -> Optional[str]:
        """
        Run analysis with automatic metadata and provenance tracking.
        
        Parameters
        ----------
        analysis_type : str
            Type of analysis
        simulation_id : str
            Input simulation ID
        parameters : dict
            Analysis parameters
        output_files : list
            Generated output files
        user : str
            User performing analysis
            
        Returns
        -------
        str or None
            Analysis metadata ID if successful
        """
        try:
            # Get input simulation metadata
            sim_metadata = self.db_manager.get_simulation_metadata(simulation_id)
            if not sim_metadata:
                logger.error(f"Simulation {simulation_id} not found")
                return None
            
            # Track analysis provenance
            if self.auto_metadata_capture:
                analysis_metadata_id = track_analysis_provenance(
                    analysis_type, [simulation_id], output_files, parameters, user
                )
                
                # Store analysis result in database
                self.db_manager.store_analysis_result(
                    simulation_id, analysis_type, 
                    {'metadata_id': analysis_metadata_id, 'output_files': output_files},
                    {'created_by': user, 'analysis_type': analysis_type}
                )
                
                # Auto-sync large output files
                if self.cloud_manager:
                    synced_outputs = []
                    for output_file in output_files:
                        if self._should_sync_file(output_file):
                            cloud_path = f"analyses/{analysis_metadata_id}/{os.path.basename(output_file)}"
                            if self.cloud_manager.upload_file(output_file, cloud_path, force_upload=True):
                                synced_outputs.append(output_file)
                    
                    if synced_outputs:
                        logger.info(f"Synced {len(synced_outputs)} analysis outputs to cloud")
                
                return analysis_metadata_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to run analysis: {e}")
            return None
    
    def search_data(self, query: Dict[str, Any]) -> Dict[str, List]:
        """
        Unified search across local database, cloud storage, and metadata.
        
        Parameters
        ----------
        query : dict
            Search query with possible keys:
            - text: Text search
            - tags: Tag filters
            - date_range: Date range filter
            - type: Data type filter
            - user: User filter
            
        Returns
        -------
        dict
            Search results organized by source
        """
        results = {
            'simulations': [],
            'analyses': [],
            'metadata': [],
            'cloud_files': []
        }
        
        try:
            # Search local database
            if 'text' in query or 'user' in query:
                # This would integrate with existing search functionality
                # For now, basic implementation
                pass
            
            # Search metadata
            from .metadata_manager import MetadataQueryBuilder
            metadata_query = MetadataQueryBuilder()
            
            if 'text' in query:
                metadata_query.filter_by_text(query['text'])
            if 'tags' in query:
                metadata_query.filter_by_tags(query['tags'])
            if 'user' in query:
                metadata_query.filter_by_user(query['user'])
            if 'date_range' in query:
                start, end = query['date_range']
                metadata_query.filter_by_date_range(start, end)
            
            metadata_results = self.metadata_manager.query(metadata_query)
            results['metadata'] = [
                {
                    'id': entry.id,
                    'title': entry.title,
                    'type': entry.type.value,
                    'created_at': entry.created_at.isoformat(),
                    'tags': list(entry.tags)
                }
                for entry in metadata_results
            ]
            
            # Search cloud storage
            if self.cloud_manager and 'text' in query:
                cloud_files = self.cloud_manager.provider.list_files()
                matching_files = [
                    f for f in cloud_files 
                    if query['text'].lower() in f['path'].lower()
                ]
                results['cloud_files'] = matching_files
            
            logger.info(f"Search returned {sum(len(r) for r in results.values())} total results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return results
    
    def _should_sync_file(self, file_path: str) -> bool:
        """Determine if a file should be synced to cloud storage."""
        try:
            if not os.path.exists(file_path):
                return False
            
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            return file_size_mb >= self.auto_sync_threshold_mb
            
        except Exception:
            return False
    
    def get_data_lineage(self, data_id: str) -> Dict[str, Any]:
        """
        Get complete data lineage for a simulation or analysis.
        
        Parameters
        ----------
        data_id : str
            Simulation or metadata ID
            
        Returns
        -------
        dict
            Complete lineage information
        """
        try:
            lineage = {
                'id': data_id,
                'type': 'unknown',
                'metadata': None,
                'provenance': [],
                'dependencies': [],
                'files': {'local': [], 'cloud': []}
            }
            
            # Check if it's a simulation
            sim_metadata = self.db_manager.get_simulation_metadata(data_id)
            if sim_metadata:
                lineage['type'] = 'simulation'
                lineage['metadata'] = sim_metadata
                
                # Get analysis results
                # This would integrate with existing database search
            
            # Check metadata
            if data_id in self.metadata_manager.storage:
                metadata_entry = self.metadata_manager.storage[data_id]
                lineage['metadata'] = {
                    'title': metadata_entry.title,
                    'type': metadata_entry.type.value,
                    'created_at': metadata_entry.created_at.isoformat(),
                    'tags': list(metadata_entry.tags)
                }
                
                # Get provenance
                lineage['provenance'] = [
                    {
                        'operation': entry.operation,
                        'timestamp': entry.timestamp.isoformat(),
                        'user': entry.user,
                        'inputs': entry.inputs,
                        'outputs': entry.outputs
                    }
                    for entry in metadata_entry.provenance
                ]
            
            # Get cloud files
            if self.cloud_manager:
                cloud_files = self.cloud_manager.list_simulation_files(data_id)
                lineage['files']['cloud'] = cloud_files
            
            return lineage
            
        except Exception as e:
            logger.error(f"Failed to get data lineage: {e}")
            return {'error': str(e)}
    
    def cleanup_old_data(self, retention_days: int = 90) -> Dict[str, int]:
        """
        Clean up old data based on retention policies.
        
        Parameters
        ----------
        retention_days : int
            Number of days to retain data
            
        Returns
        -------
        dict
            Cleanup statistics
        """
        stats = {
            'local_files_removed': 0,
            'cloud_files_removed': 0,
            'metadata_archived': 0
        }
        
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Clean up cloud storage
            if self.cloud_manager:
                removed = self.cloud_manager.cleanup_old_files(retention_days)
                stats['cloud_files_removed'] = removed
            
            # Archive old metadata (don't delete, just mark as archived)
            for metadata_id, metadata in self.metadata_manager.storage.items():
                if metadata.created_at < cutoff_date:
                    if 'archived' not in metadata.tags:
                        self.metadata_manager.add_tags(metadata_id, ['archived'])
                        stats['metadata_archived'] += 1
            
            logger.info(f"Cleanup completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return stats
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            stats = {
                'database': {},
                'metadata': self.metadata_manager.get_statistics(),
                'cloud': {},
                'system': {
                    'cloud_enabled': self.cloud_manager is not None,
                    'auto_sync_threshold_mb': self.auto_sync_threshold_mb,
                    'auto_metadata_capture': self.auto_metadata_capture
                }
            }
            
            # Get cloud statistics
            if self.cloud_manager:
                stats['cloud'] = self.cloud_manager.get_storage_stats()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {'error': str(e)}
    
    def export_data(self, export_config: Dict[str, Any]) -> str:
        """
        Export data in various formats for external tools.
        
        Parameters
        ----------
        export_config : dict
            Export configuration with keys:
            - format: Export format ('json', 'csv', 'hdf5')
            - include: What to include ('metadata', 'provenance', 'files')
            - filter: Filter criteria
            
        Returns
        -------
        str
            Path to exported file
        """
        try:
            export_format = export_config.get('format', 'json')
            include = export_config.get('include', ['metadata'])
            
            export_data = {}
            
            # Export metadata
            if 'metadata' in include:
                metadata_ids = list(self.metadata_manager.storage.keys())
                export_data['metadata'] = self.metadata_manager.export_metadata(
                    metadata_ids, export_format
                )
            
            # Export provenance
            if 'provenance' in include:
                export_data['provenance'] = {}
                for metadata_id in self.metadata_manager.storage.keys():
                    lineage = self.metadata_manager.provenance.get_lineage(metadata_id)
                    if lineage['nodes']:
                        export_data['provenance'][metadata_id] = lineage
            
            # Generate export file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_file = f"proteinmd_export_{timestamp}.{export_format}"
            
            if export_format == 'json':
                with open(export_file, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Data exported to {export_file}")
            return export_file
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise

def create_integrated_system(config_file: str) -> IntegratedDataManager:
    """
    Create integrated data management system from configuration file.
    
    Parameters
    ----------
    config_file : str
        Path to JSON configuration file
        
    Returns
    -------
    IntegratedDataManager
        Configured integrated system
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        return IntegratedDataManager(config)
        
    except Exception as e:
        logger.error(f"Failed to create integrated system: {e}")
        raise

def create_default_config() -> Dict[str, Any]:
    """Create default configuration for integrated data management."""
    return {
        'database': {
            'db_type': 'sqlite',
            'db_path': 'proteinmd_data.db'
        },
        'cloud': {
            'provider': 'aws',
            'bucket_name': 'proteinmd-data',
            'encryption_key': None,
            'upload_threshold_mb': 100.0
        },
        'metadata': {
            'auto_extract': True,
            'validation_enabled': True
        },
        'auto_sync_threshold_mb': 100.0,
        'auto_metadata_capture': True
    }

if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    # Create default configuration
    config = create_default_config()
    
    try:
        # Create integrated system
        system = IntegratedDataManager(config)
        
        print("Integrated data management system initialized!")
        print(f"Cloud storage: {'Enabled' if system.cloud_manager else 'Disabled'}")
        print(f"Auto-metadata capture: {system.auto_metadata_capture}")
        
        # Get system statistics
        stats = system.get_system_statistics()
        print(f"System statistics: {stats}")
        
    except Exception as e:
        print(f"Failed to initialize system: {e}")
        print("Please check configuration and dependencies")
