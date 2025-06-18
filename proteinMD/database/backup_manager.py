"""
Database Backup and Restore Manager

This module provides comprehensive backup and restore capabilities for the
ProteinMD simulation database, including automated backup strategies,
compression, and data migration utilities.
"""

import os
import shutil
import gzip
import tarfile
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
import subprocess
import hashlib

from .database_manager import DatabaseManager, DatabaseConfig

logger = logging.getLogger(__name__)

@dataclass
class BackupConfig:
    """
    Configuration for database backup operations.
    
    Defines backup strategies, retention policies, and storage options.
    """
    
    # Backup destination
    backup_directory: str
    
    # Backup strategy
    backup_type: str = 'full'  # 'full', 'incremental', 'differential'
    compression: bool = True
    compression_level: int = 6  # 1-9 for gzip
    
    # Retention policy
    retention_days: int = 30
    max_backups: int = 50
    
    # Scheduling
    auto_backup: bool = True
    backup_interval_hours: int = 24
    
    # Verification
    verify_backup: bool = True
    test_restore: bool = False
    
    # Metadata
    include_metadata: bool = True
    include_logs: bool = True
    include_config: bool = True
    
    # Cloud storage (future expansion)
    cloud_upload: bool = False
    cloud_provider: Optional[str] = None
    cloud_credentials: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Validate and create backup directory."""
        backup_path = Path(self.backup_directory)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        if not backup_path.is_dir():
            raise ValueError(f"Cannot create backup directory: {self.backup_directory}")

@dataclass
class BackupInfo:
    """Information about a database backup."""
    
    backup_id: str
    backup_type: str
    timestamp: datetime
    file_path: str
    file_size: int
    compressed: bool
    checksum: str
    database_type: str
    schema_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert backup info to dictionary."""
        return {
            'backup_id': self.backup_id,
            'backup_type': self.backup_type,
            'timestamp': self.timestamp.isoformat(),
            'file_path': self.file_path,
            'file_size': self.file_size,
            'compressed': self.compressed,
            'checksum': self.checksum,
            'database_type': self.database_type,
            'schema_version': self.schema_version,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackupInfo':
        """Create BackupInfo from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class BackupManager:
    """
    Database backup and restore manager for ProteinMD.
    
    Provides automated backup creation, verification, retention management,
    and restore capabilities for both SQLite and PostgreSQL databases.
    """
    
    def __init__(self, database_manager: DatabaseManager, config: BackupConfig):
        """
        Initialize backup manager.
        
        Args:
            database_manager: Database manager instance
            config: Backup configuration
        """
        self.db_manager = database_manager
        self.config = config
        self.backup_registry_file = Path(config.backup_directory) / 'backup_registry.json'
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Load existing backup registry
        self.backup_registry: List[BackupInfo] = self._load_backup_registry()
    
    def create_backup(self, backup_type: str = 'full', 
                     description: Optional[str] = None) -> BackupInfo:
        """
        Create database backup.
        
        Args:
            backup_type: Type of backup ('full', 'incremental', 'differential')
            description: Optional description for the backup
            
        Returns:
            BackupInfo: Information about created backup
        """
        try:
            timestamp = datetime.now()
            backup_id = f"backup_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            self.logger.info(f"Creating {backup_type} backup: {backup_id}")
            
            # Create backup based on database type
            if self.db_manager.config.database_type == 'sqlite':
                backup_info = self._create_sqlite_backup(backup_id, backup_type, timestamp)
            elif self.db_manager.config.database_type == 'postgresql':
                backup_info = self._create_postgresql_backup(backup_id, backup_type, timestamp)
            else:
                raise ValueError(f"Unsupported database type: {self.db_manager.config.database_type}")
            
            # Add description to metadata
            if description:
                backup_info.metadata['description'] = description
            
            # Verify backup if configured
            if self.config.verify_backup:
                if self._verify_backup(backup_info):
                    backup_info.metadata['verified'] = True
                    self.logger.info(f"Backup {backup_id} verified successfully")
                else:
                    backup_info.metadata['verified'] = False
                    self.logger.warning(f"Backup {backup_id} verification failed")
            
            # Register backup
            self.backup_registry.append(backup_info)
            self._save_backup_registry()
            
            # Clean up old backups
            self._cleanup_old_backups()
            
            self.logger.info(f"Backup {backup_id} created successfully")
            return backup_info
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            raise
    
    def _create_sqlite_backup(self, backup_id: str, backup_type: str, 
                             timestamp: datetime) -> BackupInfo:
        """Create SQLite database backup."""
        source_db = self.db_manager.config.database_path
        if not source_db or not os.path.exists(source_db):
            raise FileNotFoundError(f"SQLite database not found: {source_db}")
        
        # Determine backup file path
        backup_filename = f"{backup_id}.db"
        if self.config.compression:
            backup_filename += ".gz"
        
        backup_path = Path(self.config.backup_directory) / backup_filename
        
        # Create backup
        if backup_type == 'full':
            # Full backup: copy entire database
            if self.config.compression:
                with open(source_db, 'rb') as f_in:
                    with gzip.open(backup_path, 'wb', compresslevel=self.config.compression_level) as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(source_db, backup_path)
        else:
            # For SQLite, incremental/differential backups would require
            # custom implementation using WAL files or schema comparison
            raise NotImplementedError(f"SQLite {backup_type} backup not implemented")
        
        # Calculate file info
        file_size = backup_path.stat().st_size
        checksum = self._calculate_checksum(backup_path)
        
        # Get database schema version
        schema_version = self.db_manager.get_schema_version()
        
        return BackupInfo(
            backup_id=backup_id,
            backup_type=backup_type,
            timestamp=timestamp,
            file_path=str(backup_path),
            file_size=file_size,
            compressed=self.config.compression,
            checksum=checksum,
            database_type='sqlite',
            schema_version=schema_version,
            metadata={
                'original_db_path': source_db,
                'original_db_size': os.path.getsize(source_db),
            }
        )
    
    def _create_postgresql_backup(self, backup_id: str, backup_type: str, 
                                 timestamp: datetime) -> BackupInfo:
        """Create PostgreSQL database backup using pg_dump."""
        backup_filename = f"{backup_id}.sql"
        if self.config.compression:
            backup_filename += ".gz"
        
        backup_path = Path(self.config.backup_directory) / backup_filename
        
        # Build pg_dump command
        cmd = [
            'pg_dump',
            '-h', self.db_manager.config.host,
            '-p', str(self.db_manager.config.port),
            '-U', self.db_manager.config.username,
            '-d', self.db_manager.config.database_name,
            '--no-password',  # Use .pgpass or environment variables
        ]
        
        if backup_type == 'full':
            cmd.extend(['--create', '--clean'])
        elif backup_type == 'incremental':
            # PostgreSQL doesn't have built-in incremental backups
            # This would require custom implementation
            raise NotImplementedError("PostgreSQL incremental backup not implemented")
        
        try:
            # Execute pg_dump
            if self.config.compression:
                with gzip.open(backup_path, 'wt', compresslevel=self.config.compression_level) as f:
                    result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, 
                                          text=True, timeout=3600)
            else:
                with open(backup_path, 'w') as f:
                    result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, 
                                          text=True, timeout=3600)
            
            if result.returncode != 0:
                raise RuntimeError(f"pg_dump failed: {result.stderr}")
            
            # Calculate file info
            file_size = backup_path.stat().st_size
            checksum = self._calculate_checksum(backup_path)
            
            # Get schema version
            schema_version = self.db_manager.get_schema_version()
            
            return BackupInfo(
                backup_id=backup_id,
                backup_type=backup_type,
                timestamp=timestamp,
                file_path=str(backup_path),
                file_size=file_size,
                compressed=self.config.compression,
                checksum=checksum,
                database_type='postgresql',
                schema_version=schema_version,
                metadata={
                    'pg_dump_version': self._get_pg_dump_version(),
                    'database_name': self.db_manager.config.database_name,
                }
            )
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("pg_dump timed out")
        except Exception as e:
            if backup_path.exists():
                backup_path.unlink()  # Clean up partial backup
            raise
    
    def restore_backup(self, backup_id: str, 
                      target_database: Optional[str] = None,
                      overwrite: bool = False) -> bool:
        """
        Restore database from backup.
        
        Args:
            backup_id: ID of backup to restore
            target_database: Target database path/name (default: current database)
            overwrite: Whether to overwrite existing database
            
        Returns:
            bool: True if restore successful
        """
        try:
            # Find backup info
            backup_info = self.get_backup_info(backup_id)
            if not backup_info:
                raise ValueError(f"Backup not found: {backup_id}")
            
            backup_path = Path(backup_info.file_path)
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            self.logger.info(f"Restoring backup: {backup_id}")
            
            # Verify backup before restore
            if not self._verify_backup(backup_info):
                raise RuntimeError(f"Backup verification failed: {backup_id}")
            
            # Restore based on database type
            if backup_info.database_type == 'sqlite':
                success = self._restore_sqlite_backup(backup_info, target_database, overwrite)
            elif backup_info.database_type == 'postgresql':
                success = self._restore_postgresql_backup(backup_info, target_database, overwrite)
            else:
                raise ValueError(f"Unsupported database type: {backup_info.database_type}")
            
            if success:
                self.logger.info(f"Backup {backup_id} restored successfully")
            else:
                self.logger.error(f"Failed to restore backup {backup_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            raise
    
    def _restore_sqlite_backup(self, backup_info: BackupInfo, 
                              target_database: Optional[str], 
                              overwrite: bool) -> bool:
        """Restore SQLite backup."""
        backup_path = Path(backup_info.file_path)
        target_path = target_database or self.db_manager.config.database_path
        
        if not target_path:
            raise ValueError("No target database specified")
        
        target_path = Path(target_path)
        
        # Check if target exists
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"Target database exists: {target_path}")
        
        try:
            # Create backup of current database if it exists
            if target_path.exists():
                backup_current = target_path.with_suffix('.backup')
                shutil.copy2(target_path, backup_current)
            
            # Restore database
            if backup_info.compressed:
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(target_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(backup_path, target_path)
            
            # Verify restored database
            if self._test_database_connection(target_path):
                return True
            else:
                # Restore failed, revert if possible
                if 'backup_current' in locals():
                    shutil.move(backup_current, target_path)
                return False
                
        except Exception as e:
            self.logger.error(f"SQLite restore failed: {e}")
            # Attempt to revert
            if 'backup_current' in locals() and backup_current.exists():
                shutil.move(backup_current, target_path)
            return False
    
    def _restore_postgresql_backup(self, backup_info: BackupInfo, 
                                  target_database: Optional[str], 
                                  overwrite: bool) -> bool:
        """Restore PostgreSQL backup using psql."""
        backup_path = Path(backup_info.file_path)
        target_db = target_database or self.db_manager.config.database_name
        
        cmd = [
            'psql',
            '-h', self.db_manager.config.host,
            '-p', str(self.db_manager.config.port),
            '-U', self.db_manager.config.username,
            '-d', target_db,
            '--no-password',
        ]
        
        try:
            # Execute restore
            if backup_info.compressed:
                with gzip.open(backup_path, 'rt') as f:
                    result = subprocess.run(cmd, stdin=f, stderr=subprocess.PIPE, 
                                          text=True, timeout=3600)
            else:
                with open(backup_path, 'r') as f:
                    result = subprocess.run(cmd, stdin=f, stderr=subprocess.PIPE, 
                                          text=True, timeout=3600)
            
            if result.returncode == 0:
                return True
            else:
                self.logger.error(f"psql restore failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("psql restore timed out")
            return False
        except Exception as e:
            self.logger.error(f"PostgreSQL restore failed: {e}")
            return False
    
    def list_backups(self, limit: Optional[int] = None) -> List[BackupInfo]:
        """List available backups."""
        backups = sorted(self.backup_registry, key=lambda x: x.timestamp, reverse=True)
        if limit:
            backups = backups[:limit]
        return backups
    
    def get_backup_info(self, backup_id: str) -> Optional[BackupInfo]:
        """Get information about specific backup."""
        for backup in self.backup_registry:
            if backup.backup_id == backup_id:
                return backup
        return None
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete specific backup."""
        try:
            backup_info = self.get_backup_info(backup_id)
            if not backup_info:
                return False
            
            # Delete backup file
            backup_path = Path(backup_info.file_path)
            if backup_path.exists():
                backup_path.unlink()
            
            # Remove from registry
            self.backup_registry = [b for b in self.backup_registry if b.backup_id != backup_id]
            self._save_backup_registry()
            
            self.logger.info(f"Deleted backup: {backup_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    def _verify_backup(self, backup_info: BackupInfo) -> bool:
        """Verify backup integrity."""
        try:
            backup_path = Path(backup_info.file_path)
            if not backup_path.exists():
                return False
            
            # Verify checksum
            current_checksum = self._calculate_checksum(backup_path)
            if current_checksum != backup_info.checksum:
                self.logger.error(f"Checksum mismatch for backup {backup_info.backup_id}")
                return False
            
            # Additional verification based on database type
            if backup_info.database_type == 'sqlite':
                return self._verify_sqlite_backup(backup_path, backup_info.compressed)
            elif backup_info.database_type == 'postgresql':
                return self._verify_postgresql_backup(backup_path, backup_info.compressed)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Backup verification failed: {e}")
            return False
    
    def _verify_sqlite_backup(self, backup_path: Path, compressed: bool) -> bool:
        """Verify SQLite backup by checking database integrity."""
        try:
            if compressed:
                # Create temporary uncompressed file for verification
                temp_path = backup_path.with_suffix('.temp')
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(temp_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                verify_path = temp_path
            else:
                verify_path = backup_path
            
            # Check SQLite database integrity
            with sqlite3.connect(verify_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                
                integrity_ok = result and result[0] == 'ok'
            
            # Clean up temporary file
            if compressed and temp_path.exists():
                temp_path.unlink()
            
            return integrity_ok
            
        except Exception as e:
            self.logger.error(f"SQLite backup verification failed: {e}")
            return False
    
    def _verify_postgresql_backup(self, backup_path: Path, compressed: bool) -> bool:
        """Verify PostgreSQL backup by checking SQL syntax."""
        try:
            # Basic verification: check if file contains SQL commands
            if compressed:
                with gzip.open(backup_path, 'rt') as f:
                    content = f.read(1000)  # Read first 1KB
            else:
                with open(backup_path, 'r') as f:
                    content = f.read(1000)
            
            # Check for common SQL dump patterns
            sql_patterns = ['CREATE', 'INSERT', 'COPY', 'SET', '--']
            has_sql = any(pattern in content.upper() for pattern in sql_patterns)
            
            return has_sql
            
        except Exception as e:
            self.logger.error(f"PostgreSQL backup verification failed: {e}")
            return False
    
    def _test_database_connection(self, db_path: str) -> bool:
        """Test if database file is valid by connecting to it."""
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return True
        except Exception:
            return False
    
    def _cleanup_old_backups(self) -> None:
        """Clean up old backups based on retention policy."""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=self.config.retention_days)
            
            # Find old backups
            old_backups = [
                backup for backup in self.backup_registry
                if backup.timestamp < cutoff_time
            ]
            
            # Sort by timestamp and keep only the newest ones if we exceed max_backups
            all_backups = sorted(self.backup_registry, key=lambda x: x.timestamp, reverse=True)
            if len(all_backups) > self.config.max_backups:
                excess_backups = all_backups[self.config.max_backups:]
                old_backups.extend(excess_backups)
            
            # Remove duplicates
            old_backups = list(set(old_backups))
            
            # Delete old backups
            for backup in old_backups:
                self.delete_backup(backup.backup_id)
            
            if old_backups:
                self.logger.info(f"Cleaned up {len(old_backups)} old backups")
                
        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _get_pg_dump_version(self) -> str:
        """Get pg_dump version."""
        try:
            result = subprocess.run(['pg_dump', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unknown"
    
    def _load_backup_registry(self) -> List[BackupInfo]:
        """Load backup registry from file."""
        try:
            if self.backup_registry_file.exists():
                with open(self.backup_registry_file, 'r') as f:
                    data = json.load(f)
                    return [BackupInfo.from_dict(item) for item in data]
        except Exception as e:
            self.logger.warning(f"Failed to load backup registry: {e}")
        
        return []
    
    def _save_backup_registry(self) -> None:
        """Save backup registry to file."""
        try:
            data = [backup.to_dict() for backup in self.backup_registry]
            with open(self.backup_registry_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save backup registry: {e}")
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """Get backup statistics and metrics."""
        if not self.backup_registry:
            return {'total_backups': 0}
        
        total_size = sum(backup.file_size for backup in self.backup_registry)
        backup_types = {}
        for backup in self.backup_registry:
            backup_types[backup.backup_type] = backup_types.get(backup.backup_type, 0) + 1
        
        oldest_backup = min(self.backup_registry, key=lambda x: x.timestamp)
        newest_backup = max(self.backup_registry, key=lambda x: x.timestamp)
        
        return {
            'total_backups': len(self.backup_registry),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'backup_types': backup_types,
            'oldest_backup': oldest_backup.timestamp.isoformat(),
            'newest_backup': newest_backup.timestamp.isoformat(),
            'average_size_mb': round(total_size / len(self.backup_registry) / (1024 * 1024), 2),
        }
