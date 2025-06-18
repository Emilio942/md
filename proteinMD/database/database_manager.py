"""
Database Manager and Connection Handling

This module provides the main database management functionality, including
connection handling, session management, and high-level database operations
for both SQLite and PostgreSQL backends.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
from urllib.parse import urlparse
import json

import sqlalchemy as sa
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from .models import Base, SimulationRecord, AnalysisRecord, WorkflowRecord, ProteinStructure

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """
    Database configuration parameters.
    
    Supports both SQLite and PostgreSQL with comprehensive configuration options.
    """
    
    # Database type and connection
    database_type: str = 'sqlite'  # 'sqlite' or 'postgresql'
    database_path: Optional[str] = None  # for SQLite
    database_url: Optional[str] = None   # for PostgreSQL or custom URLs
    
    # PostgreSQL specific settings
    host: str = 'localhost'
    port: int = 5432
    username: Optional[str] = None
    password: Optional[str] = None
    database_name: str = 'proteinmd'
    
    # Connection pool settings
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # SQLite specific settings
    sqlite_timeout: int = 20
    sqlite_check_same_thread: bool = False
    
    # General settings
    echo_sql: bool = False
    autocommit: bool = False
    autoflush: bool = True
    
    # Schema and migration
    create_tables: bool = True
    upgrade_schema: bool = True
    
    # Backup settings
    backup_directory: Optional[str] = None
    auto_backup: bool = True
    backup_retention_days: int = 30
    
    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if self.database_type not in ['sqlite', 'postgresql']:
            raise ValueError(f"Unsupported database type: {self.database_type}")
        
        # Set default paths for SQLite
        if self.database_type == 'sqlite' and not self.database_path:
            home_dir = Path.home()
            config_dir = home_dir / '.proteinmd'
            config_dir.mkdir(exist_ok=True)
            self.database_path = str(config_dir / 'proteinmd.db')
        
        # Set default backup directory
        if not self.backup_directory:
            if self.database_type == 'sqlite' and self.database_path:
                self.backup_directory = str(Path(self.database_path).parent / 'backups')
            else:
                self.backup_directory = str(Path.home() / '.proteinmd' / 'backups')
    
    def get_database_url(self) -> str:
        """Generate database URL for SQLAlchemy."""
        if self.database_url:
            return self.database_url
        
        if self.database_type == 'sqlite':
            return f"sqlite:///{self.database_path}"
        elif self.database_type == 'postgresql':
            auth = ""
            if self.username:
                auth = self.username
                if self.password:
                    auth += f":{self.password}"
                auth += "@"
            return f"postgresql://{auth}{self.host}:{self.port}/{self.database_name}"
        
        raise ValueError(f"Cannot generate URL for database type: {self.database_type}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (excluding sensitive data)."""
        config_dict = {
            'database_type': self.database_type,
            'database_path': self.database_path,
            'host': self.host,
            'port': self.port,
            'username': self.username,
            'database_name': self.database_name,
            'pool_size': self.pool_size,
            'echo_sql': self.echo_sql,
            'backup_directory': self.backup_directory,
            'auto_backup': self.auto_backup,
        }
        return {k: v for k, v in config_dict.items() if v is not None}

class DatabaseManager:
    """
    Main database manager for ProteinMD simulation metadata.
    
    Provides high-level database operations including connection management,
    session handling, and common database operations for simulation metadata.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize database manager with configuration.
        
        Args:
            config: Database configuration parameters
        """
        self.config = config
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._metadata_cache: Dict[str, Any] = {}
        
        logger.info(f"Initializing DatabaseManager with {config.database_type} backend")
    
    def connect(self) -> None:
        """Establish database connection and initialize schema."""
        try:
            # Create engine with appropriate configuration
            engine_kwargs = {
                'echo': self.config.echo_sql,
            }
            
            # SQLite specific settings
            if self.config.database_type == 'sqlite':
                engine_kwargs.update({
                    'poolclass': StaticPool,
                    'connect_args': {
                        'timeout': self.config.sqlite_timeout,
                        'check_same_thread': self.config.sqlite_check_same_thread,
                    }
                })
                
                # Ensure directory exists for SQLite
                db_path = Path(self.config.database_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # PostgreSQL and other databases can use connection pooling
                engine_kwargs.update({
                    'pool_size': self.config.pool_size,
                    'max_overflow': self.config.max_overflow,
                    'pool_timeout': self.config.pool_timeout,
                    'pool_recycle': self.config.pool_recycle,
                })
            
            # Create engine
            database_url = self.config.get_database_url()
            self.engine = create_engine(database_url, **engine_kwargs)
            
            # Test connection
            with self.engine.connect() as conn:
                if self.config.database_type == 'sqlite':
                    conn.execute(text("SELECT 1"))
                else:
                    conn.execute(text("SELECT version()"))
                logger.info("Database connection established successfully")
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=self.config.autocommit,
                autoflush=self.config.autoflush,
                bind=self.engine
            )
            
            # Initialize schema
            if self.config.create_tables:
                self.create_tables()
            
            # Load metadata cache
            self._update_metadata_cache()
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close database connections and cleanup."""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Database connections closed")
        except Exception as e:
            logger.warning(f"Error during database disconnection: {e}")
    
    def create_tables(self) -> None:
        """Create database tables from models."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self) -> None:
        """Drop all database tables (use with caution!)."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions.
        
        Yields:
            Session: SQLAlchemy session object
        """
        if not self.SessionLocal:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        try:
            with self.get_session() as session:
                # Get table counts
                counts = {}
                for table_name in ['simulations', 'analyses', 'workflows', 'protein_structures']:
                    count = session.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
                    counts[table_name] = count
                
                # Database size (SQLite only)
                db_size = None
                if self.config.database_type == 'sqlite' and self.config.database_path:
                    db_path = Path(self.config.database_path)
                    if db_path.exists():
                        db_size = db_path.stat().st_size
                
                # Get database version
                if self.config.database_type == 'sqlite':
                    version = session.execute(text("SELECT sqlite_version()")).scalar()
                else:
                    version = session.execute(text("SELECT version()")).scalar()
                
                return {
                    'database_type': self.config.database_type,
                    'database_path': self.config.database_path,
                    'database_url': self.config.get_database_url() if self.config.database_type == 'postgresql' else None,
                    'version': version,
                    'size_bytes': db_size,
                    'table_counts': counts,
                    'connection_pool_size': self.config.pool_size,
                    'last_updated': self._metadata_cache.get('last_updated'),
                }
                
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            raise
    
    def _update_metadata_cache(self) -> None:
        """Update internal metadata cache."""
        try:
            from datetime import datetime
            self._metadata_cache['last_updated'] = datetime.now().isoformat()
            logger.debug("Metadata cache updated")
        except Exception as e:
            logger.warning(f"Failed to update metadata cache: {e}")
    
    # High-level operations for simulation records
    
    def store_simulation(self, simulation_data: Dict[str, Any]) -> int:
        """
        Store simulation metadata in database.
        
        Args:
            simulation_data: Dictionary containing simulation metadata
            
        Returns:
            int: Database ID of stored simulation
        """
        try:
            with self.get_session() as session:
                simulation = SimulationRecord(**simulation_data)
                session.add(simulation)
                session.flush()  # Get ID without committing
                simulation_id = simulation.id
                logger.info(f"Stored simulation record with ID: {simulation_id}")
                return simulation_id
                
        except IntegrityError as e:
            logger.error(f"Integrity error storing simulation: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to store simulation: {e}")
            raise
    
    def get_simulation(self, simulation_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve simulation record by ID.
        
        Args:
            simulation_id: Database ID of simulation
            
        Returns:
            Optional[Dict]: Simulation data or None if not found
        """
        try:
            with self.get_session() as session:
                simulation = session.query(SimulationRecord).filter_by(id=simulation_id).first()
                if simulation:
                    return simulation.to_dict()
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve simulation {simulation_id}: {e}")
            raise
    
    def update_simulation_status(self, simulation_id: int, status: str, **kwargs) -> bool:
        """
        Update simulation status and related fields.
        
        Args:
            simulation_id: Database ID of simulation
            status: New status value
            **kwargs: Additional fields to update
            
        Returns:
            bool: True if update successful, False if simulation not found
        """
        try:
            with self.get_session() as session:
                simulation = session.query(SimulationRecord).filter_by(id=simulation_id).first()
                if not simulation:
                    return False
                
                simulation.status = status
                for key, value in kwargs.items():
                    if hasattr(simulation, key):
                        setattr(simulation, key, value)
                
                logger.info(f"Updated simulation {simulation_id} status to {status}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update simulation {simulation_id}: {e}")
            raise
    
    def list_simulations(self, limit: int = 100, offset: int = 0, 
                        status: Optional[str] = None, 
                        user_id: Optional[str] = None,
                        project_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List simulations with optional filtering.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            status: Filter by status
            user_id: Filter by user ID
            project_name: Filter by project name
            
        Returns:
            List[Dict]: List of simulation records
        """
        try:
            with self.get_session() as session:
                query = session.query(SimulationRecord)
                
                # Apply filters
                if status:
                    query = query.filter_by(status=status)
                if user_id:
                    query = query.filter_by(user_id=user_id)
                if project_name:
                    query = query.filter_by(project_name=project_name)
                
                # Order by creation date (newest first)
                query = query.order_by(SimulationRecord.created_at.desc())
                
                # Apply pagination
                simulations = query.limit(limit).offset(offset).all()
                
                return [sim.to_dict() for sim in simulations]
                
        except Exception as e:
            logger.error(f"Failed to list simulations: {e}")
            raise
    
    def delete_simulation(self, simulation_id: int) -> bool:
        """
        Delete simulation and all related records.
        
        Args:
            simulation_id: Database ID of simulation
            
        Returns:
            bool: True if deletion successful, False if simulation not found
        """
        try:
            with self.get_session() as session:
                simulation = session.query(SimulationRecord).filter_by(id=simulation_id).first()
                if not simulation:
                    return False
                
                session.delete(simulation)
                logger.info(f"Deleted simulation record {simulation_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete simulation {simulation_id}: {e}")
            raise
    
    # Analysis record operations
    
    def store_analysis(self, analysis_data: Dict[str, Any]) -> int:
        """Store analysis metadata in database."""
        try:
            with self.get_session() as session:
                analysis = AnalysisRecord(**analysis_data)
                session.add(analysis)
                session.flush()
                analysis_id = analysis.id
                logger.info(f"Stored analysis record with ID: {analysis_id}")
                return analysis_id
                
        except Exception as e:
            logger.error(f"Failed to store analysis: {e}")
            raise
    
    def get_simulation_analyses(self, simulation_id: int) -> List[Dict[str, Any]]:
        """Get all analyses for a simulation."""
        try:
            with self.get_session() as session:
                analyses = session.query(AnalysisRecord).filter_by(simulation_id=simulation_id).all()
                return [analysis.to_dict() for analysis in analyses]
                
        except Exception as e:
            logger.error(f"Failed to get analyses for simulation {simulation_id}: {e}")
            raise
    
    # Utility methods
    
    def vacuum_database(self) -> None:
        """Optimize database (SQLite VACUUM or PostgreSQL VACUUM)."""
        try:
            with self.engine.connect() as conn:
                if self.config.database_type == 'sqlite':
                    conn.execute(text("VACUUM"))
                    logger.info("SQLite database vacuumed")
                else:
                    conn.execute(text("VACUUM ANALYZE"))
                    logger.info("PostgreSQL database vacuumed and analyzed")
                    
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
            raise
    
    def get_schema_version(self) -> str:
        """Get current database schema version."""
        # This would typically check a schema_version table
        # For now, return a default version
        return "1.0.0"
    
    def check_health(self) -> Dict[str, Any]:
        """Check database health and connectivity."""
        try:
            with self.get_session() as session:
                # Simple connectivity test
                result = session.execute(text("SELECT 1")).scalar()
                
                # Get basic stats
                info = self.get_database_info()
                
                return {
                    'status': 'healthy',
                    'connectivity': 'ok' if result == 1 else 'failed',
                    'database_info': info,
                    'schema_version': self.get_schema_version(),
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'connectivity': 'failed',
            }
