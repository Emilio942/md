"""
Database Command Line Interface

This module provides CLI commands for database management operations
including initialization, backup, restore, search, and maintenance.
"""

import argparse
import json
import sys
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

from .database_manager import DatabaseManager, DatabaseConfig
from .search_engine import SimulationSearchEngine, SearchQuery, SearchOperator, SortOrder
from .backup_manager import BackupManager, BackupConfig

logger = logging.getLogger(__name__)

class DatabaseCLI:
    """
    Command-line interface for ProteinMD database operations.
    
    Provides comprehensive database management capabilities including
    initialization, backup/restore, search, and maintenance operations.
    """
    
    def __init__(self):
        """Initialize database CLI."""
        self.db_manager: Optional[DatabaseManager] = None
        self.search_engine: Optional[SimulationSearchEngine] = None
        self.backup_manager: Optional[BackupManager] = None
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for database commands."""
        parser = argparse.ArgumentParser(
            description="ProteinMD Database Management",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Initialize database
  proteinmd database init --type sqlite --path ~/.proteinmd/database.db

  # Create backup
  proteinmd database backup create --description "Daily backup"

  # Search simulations
  proteinmd database search --text "protein folding" --status completed

  # List recent simulations
  proteinmd database list --recent 7 --limit 20

  # Database statistics
  proteinmd database stats

  # Restore from backup
  proteinmd database restore backup_20250612_120000
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Database commands')
        
        # Initialize database command
        init_parser = subparsers.add_parser('init', help='Initialize database')
        init_parser.add_argument('--type', choices=['sqlite', 'postgresql'], 
                                default='sqlite', help='Database type')
        init_parser.add_argument('--path', help='Database file path (SQLite)')
        init_parser.add_argument('--host', default='localhost', help='Database host (PostgreSQL)')
        init_parser.add_argument('--port', type=int, default=5432, help='Database port (PostgreSQL)')
        init_parser.add_argument('--username', help='Database username (PostgreSQL)')
        init_parser.add_argument('--password', help='Database password (PostgreSQL)')
        init_parser.add_argument('--database', default='proteinmd', help='Database name (PostgreSQL)')
        init_parser.add_argument('--force', action='store_true', help='Force recreation of existing database')
        
        # Database info command
        info_parser = subparsers.add_parser('info', help='Show database information')
        info_parser.add_argument('--detailed', action='store_true', help='Show detailed information')
        
        # Statistics command
        stats_parser = subparsers.add_parser('stats', help='Show database statistics')
        stats_parser.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
        
        # List simulations command
        list_parser = subparsers.add_parser('list', help='List simulations')
        list_parser.add_argument('--limit', type=int, default=20, help='Maximum number of results')
        list_parser.add_argument('--offset', type=int, default=0, help='Results offset')
        list_parser.add_argument('--status', help='Filter by status')
        list_parser.add_argument('--user', help='Filter by user ID')
        list_parser.add_argument('--project', help='Filter by project name')
        list_parser.add_argument('--recent', type=int, help='Show simulations from last N days')
        list_parser.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
        
        # Search command
        search_parser = subparsers.add_parser('search', help='Search simulations')
        search_parser.add_argument('--text', help='Text search query')
        search_parser.add_argument('--status', help='Status filter')
        search_parser.add_argument('--user', help='User filter')
        search_parser.add_argument('--project', help='Project filter')
        search_parser.add_argument('--force-field', help='Force field filter')
        search_parser.add_argument('--min-temp', type=float, help='Minimum temperature')
        search_parser.add_argument('--max-temp', type=float, help='Maximum temperature')
        search_parser.add_argument('--min-energy', type=float, help='Minimum final energy')
        search_parser.add_argument('--max-energy', type=float, help='Maximum final energy')
        search_parser.add_argument('--converged', action='store_true', help='Only converged simulations')
        search_parser.add_argument('--failed', action='store_true', help='Only failed simulations')
        search_parser.add_argument('--limit', type=int, default=20, help='Maximum number of results')
        search_parser.add_argument('--sort', choices=['created', 'modified', 'name', 'energy'], 
                                  default='created', help='Sort field')
        search_parser.add_argument('--order', choices=['asc', 'desc'], default='desc', help='Sort order')
        search_parser.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
        
        # Show simulation command
        show_parser = subparsers.add_parser('show', help='Show simulation details')
        show_parser.add_argument('simulation_id', type=int, help='Simulation ID')
        show_parser.add_argument('--include-analyses', action='store_true', help='Include analysis results')
        show_parser.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
        
        # Delete simulation command
        delete_parser = subparsers.add_parser('delete', help='Delete simulation')
        delete_parser.add_argument('simulation_id', type=int, help='Simulation ID')
        delete_parser.add_argument('--force', action='store_true', help='Force deletion without confirmation')
        
        # Backup commands
        backup_parser = subparsers.add_parser('backup', help='Backup operations')
        backup_subparsers = backup_parser.add_subparsers(dest='backup_command', help='Backup commands')
        
        # Create backup
        backup_create = backup_subparsers.add_parser('create', help='Create backup')
        backup_create.add_argument('--type', choices=['full', 'incremental'], default='full', help='Backup type')
        backup_create.add_argument('--description', help='Backup description')
        backup_create.add_argument('--compress', action='store_true', help='Compress backup')
        
        # List backups
        backup_list = backup_subparsers.add_parser('list', help='List backups')
        backup_list.add_argument('--limit', type=int, default=10, help='Maximum number of backups to show')
        
        # Backup info
        backup_info = backup_subparsers.add_parser('info', help='Show backup information')
        backup_info.add_argument('backup_id', help='Backup ID')
        
        # Delete backup
        backup_delete = backup_subparsers.add_parser('delete', help='Delete backup')
        backup_delete.add_argument('backup_id', help='Backup ID')
        backup_delete.add_argument('--force', action='store_true', help='Force deletion without confirmation')
        
        # Restore command
        restore_parser = subparsers.add_parser('restore', help='Restore from backup')
        restore_parser.add_argument('backup_id', help='Backup ID')
        restore_parser.add_argument('--target', help='Target database path/name')
        restore_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing database')
        
        # Maintenance commands
        maintenance_parser = subparsers.add_parser('maintenance', help='Database maintenance')
        maintenance_subparsers = maintenance_parser.add_subparsers(dest='maintenance_command', 
                                                                   help='Maintenance commands')
        
        # Vacuum database
        vacuum_parser = maintenance_subparsers.add_parser('vacuum', help='Vacuum database')
        
        # Check health
        health_parser = maintenance_subparsers.add_parser('health', help='Check database health')
        
        # Cleanup old data
        cleanup_parser = maintenance_subparsers.add_parser('cleanup', help='Cleanup old data')
        cleanup_parser.add_argument('--days', type=int, default=90, help='Delete data older than N days')
        cleanup_parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted')
        
        return parser
    
    def run(self, args: List[str] = None) -> int:
        """Run database CLI command."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        try:
            if parsed_args.command == 'init':
                return self.cmd_init(parsed_args)
            elif parsed_args.command == 'info':
                return self.cmd_info(parsed_args)
            elif parsed_args.command == 'stats':
                return self.cmd_stats(parsed_args)
            elif parsed_args.command == 'list':
                return self.cmd_list(parsed_args)
            elif parsed_args.command == 'search':
                return self.cmd_search(parsed_args)
            elif parsed_args.command == 'show':
                return self.cmd_show(parsed_args)
            elif parsed_args.command == 'delete':
                return self.cmd_delete(parsed_args)
            elif parsed_args.command == 'backup':
                return self.cmd_backup(parsed_args)
            elif parsed_args.command == 'restore':
                return self.cmd_restore(parsed_args)
            elif parsed_args.command == 'maintenance':
                return self.cmd_maintenance(parsed_args)
            else:
                parser.print_help()
                return 1
                
        except Exception as e:
            logger.error(f"Command failed: {e}")
            return 1
    
    def _initialize_database(self, args) -> None:
        """Initialize database connection."""
        config = DatabaseConfig(
            database_type=args.type if hasattr(args, 'type') else 'sqlite',
            database_path=getattr(args, 'path', None),
            host=getattr(args, 'host', 'localhost'),
            port=getattr(args, 'port', 5432),
            username=getattr(args, 'username', None),
            password=getattr(args, 'password', None),
            database_name=getattr(args, 'database', 'proteinmd'),
        )
        
        self.db_manager = DatabaseManager(config)
        self.db_manager.connect()
        
        self.search_engine = SimulationSearchEngine(self.db_manager)
        
        # Initialize backup manager with default config
        backup_config = BackupConfig(
            backup_directory=str(Path.home() / '.proteinmd' / 'backups')
        )
        self.backup_manager = BackupManager(self.db_manager, backup_config)
    
    def cmd_init(self, args) -> int:
        """Initialize database command."""
        print("🔧 Initializing ProteinMD Database")
        print("=" * 40)
        
        try:
            # Check if database already exists
            if args.type == 'sqlite' and args.path:
                db_path = Path(args.path)
                if db_path.exists() and not args.force:
                    print(f"❌ Database already exists: {db_path}")
                    print("Use --force to recreate the database")
                    return 1
            
            # Initialize database
            self._initialize_database(args)
            
            # Get database info
            info = self.db_manager.get_database_info()
            
            print(f"✅ Database initialized successfully")
            print(f"   Type: {info['database_type']}")
            if info.get('database_path'):
                print(f"   Path: {info['database_path']}")
            else:
                print(f"   URL: {info.get('database_url', 'N/A')}")
            print(f"   Version: {info['version']}")
            
            return 0
            
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            return 1
    
    def cmd_info(self, args) -> int:
        """Show database information command."""
        try:
            self._initialize_database(args)
            info = self.db_manager.get_database_info()
            
            print("📊 ProteinMD Database Information")
            print("=" * 40)
            print(f"Database Type: {info['database_type']}")
            print(f"Version: {info['version']}")
            
            if info.get('database_path'):
                print(f"Path: {info['database_path']}")
                if info.get('size_bytes'):
                    size_mb = info['size_bytes'] / (1024 * 1024)
                    print(f"Size: {size_mb:.2f} MB")
            
            print("\nTable Counts:")
            for table, count in info['table_counts'].items():
                print(f"  {table}: {count}")
            
            if args.detailed:
                print(f"\nConnection Pool Size: {info['connection_pool_size']}")
                print(f"Last Updated: {info.get('last_updated', 'Unknown')}")
            
            return 0
            
        except Exception as e:
            print(f"❌ Failed to get database info: {e}")
            return 1
    
    def cmd_stats(self, args) -> int:
        """Show database statistics command."""
        try:
            self._initialize_database(args)
            
            with self.db_manager.get_session() as session:
                # Get basic statistics
                from sqlalchemy import func, text
                from .models import SimulationRecord
                
                total_sims = session.query(func.count(SimulationRecord.id)).scalar()
                
                # Status distribution
                status_stats = session.query(
                    SimulationRecord.status,
                    func.count(SimulationRecord.id)
                ).group_by(SimulationRecord.status).all()
                
                # Recent activity (last 30 days)
                cutoff_date = datetime.now() - timedelta(days=30)
                recent_count = session.query(func.count(SimulationRecord.id)).filter(
                    SimulationRecord.created_at >= cutoff_date
                ).scalar()
                
                if args.format == 'json':
                    stats = {
                        'total_simulations': total_sims,
                        'recent_simulations_30d': recent_count,
                        'status_distribution': dict(status_stats),
                    }
                    print(json.dumps(stats, indent=2))
                else:
                    print("📈 Database Statistics")
                    print("=" * 30)
                    print(f"Total Simulations: {total_sims}")
                    print(f"Recent (30 days): {recent_count}")
                    print("\nStatus Distribution:")
                    for status, count in status_stats:
                        print(f"  {status}: {count}")
            
            return 0
            
        except Exception as e:
            print(f"❌ Failed to get statistics: {e}")
            return 1
    
    def cmd_list(self, args) -> int:
        """List simulations command."""
        try:
            self._initialize_database(args)
            
            # Apply filters
            kwargs = {
                'limit': args.limit,
                'offset': args.offset,
            }
            
            if args.status:
                kwargs['status'] = args.status
            if args.user:
                kwargs['user_id'] = args.user
            if args.project:
                kwargs['project_name'] = args.project
            
            simulations = self.db_manager.list_simulations(**kwargs)
            
            if args.format == 'json':
                print(json.dumps(simulations, indent=2, default=str))
            else:
                print(f"📋 Simulations ({len(simulations)} results)")
                print("=" * 60)
                print(f"{'ID':<5} {'Name':<20} {'Status':<12} {'Created':<20}")
                print("-" * 60)
                
                for sim in simulations:
                    created = sim.get('created_at', '')[:19] if sim.get('created_at') else ''
                    print(f"{sim['id']:<5} {sim['name'][:19]:<20} {sim['status']:<12} {created:<20}")
            
            return 0
            
        except Exception as e:
            print(f"❌ Failed to list simulations: {e}")
            return 1
    
    def cmd_search(self, args) -> int:
        """Search simulations command."""
        try:
            self._initialize_database(args)
            
            # Build search query
            query = SearchQuery()
            
            if args.text:
                query.add_text_search(args.text)
            
            if args.status:
                query.status_filter = [args.status]
            
            if args.user:
                query.user_filter = [args.user]
            
            if args.project:
                query.project_filter = [args.project]
            
            if args.force_field:
                query.force_fields = [args.force_field]
            
            if args.min_temp or args.max_temp:
                min_temp = args.min_temp or 0
                max_temp = args.max_temp or 1000
                query.temperature_range = (min_temp, max_temp)
            
            if args.min_energy or args.max_energy:
                min_energy = args.min_energy or -1e10
                max_energy = args.max_energy or 1e10
                query.energy_range = (min_energy, max_energy)
            
            if args.converged:
                query.convergence_filter = True
            
            if args.failed:
                query.status_filter = ['failed']
            
            query.limit = args.limit
            
            # Set sorting
            sort_field_map = {
                'created': 'created_at',
                'modified': 'last_modified',
                'name': 'name',
                'energy': 'final_energy'
            }
            query.sort_field = sort_field_map.get(args.sort, 'created_at')
            query.sort_order = SortOrder.DESC if args.order == 'desc' else SortOrder.ASC
            
            # Execute search
            result = self.search_engine.search(query)
            
            if args.format == 'json':
                print(json.dumps({
                    'simulations': result.simulations,
                    'total_count': result.total_count,
                    'query_time_ms': result.query_time_ms,
                }, indent=2, default=str))
            else:
                print(f"🔍 Search Results ({result.total_count} total, {result.query_time_ms:.1f}ms)")
                print("=" * 70)
                
                if result.applied_filters:
                    print("Applied filters:", ", ".join(result.applied_filters))
                    print("-" * 70)
                
                print(f"{'ID':<5} {'Name':<25} {'Status':<12} {'Energy':<12} {'Created':<20}")
                print("-" * 70)
                
                for sim in result.simulations:
                    energy = f"{sim.get('final_energy', 0):.1f}" if sim.get('final_energy') else 'N/A'
                    created = sim.get('created_at', '')[:19] if sim.get('created_at') else ''
                    print(f"{sim['id']:<5} {sim['name'][:24]:<25} {sim['status']:<12} {energy:<12} {created:<20}")
            
            return 0
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return 1
    
    def cmd_store(self, args) -> int:
        """Store simulation record command."""
        try:
            self._initialize_database(args)
            
            simulation_dir = Path(args.simulation_dir)
            if not simulation_dir.exists():
                print(f"❌ Simulation directory not found: {simulation_dir}")
                return 1
            
            # Extract simulation metadata from directory
            from .models import SimulationRecord
            
            # Create simulation record with metadata from directory
            record = SimulationRecord(
                name=simulation_dir.name,
                description=f"Simulation from {simulation_dir}",
                status="completed",
                working_directory=str(simulation_dir),
                # Add more fields as needed based on available data
            )
            
            with self.db_manager.get_session() as session:
                session.add(record)
                session.commit()
                
            print(f"✅ Simulation record stored with ID: {record.id}")
            return 0
            
        except Exception as e:
            print(f"❌ Failed to store simulation: {e}")
            return 1
    
    def cmd_show(self, args) -> int:
        """Show simulation details command."""
        try:
            self._initialize_database(args)
            
            with self.db_manager.get_session() as session:
                from .models import SimulationRecord
                
                # Try to find by ID or simulation_id
                try:
                    sim_id = int(args.simulation_id)
                    simulation = session.query(SimulationRecord).filter_by(id=sim_id).first()
                except ValueError:
                    simulation = session.query(SimulationRecord).filter_by(simulation_id=args.simulation_id).first()
                
                if not simulation:
                    print(f"❌ Simulation not found: {args.simulation_id}")
                    return 1
                
                if getattr(args, 'format', 'table') == 'json':
                    import json
                    print(json.dumps(simulation.to_dict(), indent=2))
                else:
                    print(f"\n📋 Simulation Details")
                    print(f"==================")
                    print(f"ID: {simulation.id}")
                    print(f"Name: {simulation.name}")
                    print(f"Status: {simulation.status}")
                    print(f"Created: {simulation.created_at}")
                    if simulation.description:
                        print(f"Description: {simulation.description}")
                    if simulation.working_directory:
                        print(f"Directory: {simulation.working_directory}")
                
            return 0
            
        except Exception as e:
            print(f"❌ Failed to show simulation: {e}")
            return 1
    
    def cmd_delete(self, args) -> int:
        """Delete simulation record command."""
        try:
            self._initialize_database(args)
            
            # Confirmation prompt
            if not getattr(args, 'force', False):
                response = input(f"⚠️  Are you sure you want to delete simulation {args.simulation_id}? (y/N): ")
                if response.lower() != 'y':
                    print("❌ Deletion cancelled")
                    return 0
            
            with self.db_manager.get_session() as session:
                from .models import SimulationRecord
                
                # Try to find by ID or simulation_id
                try:
                    sim_id = int(args.simulation_id)
                    simulation = session.query(SimulationRecord).filter_by(id=sim_id).first()
                except ValueError:
                    simulation = session.query(SimulationRecord).filter_by(simulation_id=args.simulation_id).first()
                
                if not simulation:
                    print(f"❌ Simulation not found: {args.simulation_id}")
                    return 1
                
                session.delete(simulation)
                session.commit()
                
                print(f"✅ Simulation {args.simulation_id} deleted successfully")
            
            return 0
            
        except Exception as e:
            print(f"❌ Failed to delete simulation: {e}")
            return 1
    
    def cmd_backup(self, args) -> int:
        """Backup operations command."""
        try:
            self._initialize_database(args)
            
            if args.backup_command == 'create':
                print("💾 Creating database backup...")
                backup_info = self.backup_manager.create_backup(
                    backup_type=args.type,
                    description=args.description
                )
                print(f"✅ Backup created: {backup_info.backup_id}")
                print(f"   File: {backup_info.file_path}")
                print(f"   Size: {backup_info.file_size / (1024*1024):.2f} MB")
                
            elif args.backup_command == 'list':
                backups = self.backup_manager.list_backups(limit=args.limit)
                print(f"💾 Database Backups ({len(backups)} found)")
                print("=" * 60)
                print(f"{'Backup ID':<25} {'Type':<8} {'Size (MB)':<10} {'Created':<20}")
                print("-" * 60)
                
                for backup in backups:
                    size_mb = backup.file_size / (1024 * 1024)
                    created = backup.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"{backup.backup_id:<25} {backup.backup_type:<8} {size_mb:<10.2f} {created:<20}")
                
            elif args.backup_command == 'info':
                backup_info = self.backup_manager.get_backup_info(args.backup_id)
                if not backup_info:
                    print(f"❌ Backup not found: {args.backup_id}")
                    return 1
                
                print(f"💾 Backup Information: {args.backup_id}")
                print("=" * 40)
                print(f"Type: {backup_info.backup_type}")
                print(f"Created: {backup_info.timestamp}")
                print(f"File: {backup_info.file_path}")
                print(f"Size: {backup_info.file_size / (1024*1024):.2f} MB")
                print(f"Compressed: {backup_info.compressed}")
                print(f"Checksum: {backup_info.checksum[:16]}...")
                
                if backup_info.metadata:
                    print("\nMetadata:")
                    for key, value in backup_info.metadata.items():
                        print(f"  {key}: {value}")
                
            elif args.backup_command == 'delete':
                if not args.force:
                    response = input(f"Delete backup '{args.backup_id}'? [y/N]: ")
                    if response.lower() != 'y':
                        print("Deletion cancelled")
                        return 0
                
                success = self.backup_manager.delete_backup(args.backup_id)
                if success:
                    print(f"✅ Backup {args.backup_id} deleted")
                else:
                    print(f"❌ Failed to delete backup {args.backup_id}")
                    return 1
            
            return 0
            
        except Exception as e:
            print(f"❌ Backup operation failed: {e}")
            return 1
    
    def cmd_restore(self, args) -> int:
        """Restore from backup command."""
        try:
            self._initialize_database(args)
            
            print(f"🔄 Restoring from backup: {args.backup_id}")
            
            success = self.backup_manager.restore_backup(
                backup_id=args.backup_id,
                target_database=args.target,
                overwrite=args.overwrite
            )
            
            if success:
                print(f"✅ Database restored successfully from {args.backup_id}")
                return 0
            else:
                print(f"❌ Restore failed")
                return 1
                
        except Exception as e:
            print(f"❌ Restore failed: {e}")
            return 1
    
    def cmd_maintenance(self, args) -> int:
        """Database maintenance command."""
        try:
            self._initialize_database(args)
            
            if args.maintenance_command == 'vacuum':
                print("🧹 Vacuuming database...")
                self.db_manager.vacuum_database()
                print("✅ Database vacuum completed")
                
            elif args.maintenance_command == 'health':
                print("🔍 Checking database health...")
                health = self.db_manager.check_health()
                
                print(f"Status: {health['status']}")
                print(f"Connectivity: {health['connectivity']}")
                
                if health['status'] == 'healthy':
                    print("✅ Database is healthy")
                    return 0
                else:
                    print("❌ Database health issues detected")
                    if 'error' in health:
                        print(f"Error: {health['error']}")
                    return 1
                    
            elif args.maintenance_command == 'cleanup':
                print(f"🧹 Cleaning up data older than {args.days} days...")
                if args.dry_run:
                    print("(Dry run - no data will be deleted)")
                
                # Implementation would go here
                print("Cleanup not yet implemented")
            
            return 0
            
        except Exception as e:
            print(f"❌ Maintenance operation failed: {e}")
            return 1
    
    def cmd_export(self, args) -> int:
        """Export database command."""
        try:
            self._initialize_database(args)
            
            with self.db_manager.get_session() as session:
                from .models import SimulationRecord
                
                simulations = session.query(SimulationRecord).all()
                
                if getattr(args, 'format', 'json') == 'json':
                    import json
                    data = [sim.to_dict() for sim in simulations]
                    
                    with open(args.output_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    print(f"✅ Database exported to {args.output_file} (JSON)")
                else:
                    # CSV export
                    import csv
                    
                    if simulations:
                        with open(args.output_file, 'w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=simulations[0].to_dict().keys())
                            writer.writeheader()
                            for sim in simulations:
                                writer.writerow(sim.to_dict())
                    
                    print(f"✅ Database exported to {args.output_file} (CSV)")
                
                print(f"📊 Exported {len(simulations)} simulations")
            
            return 0
            
        except Exception as e:
            print(f"❌ Failed to export database: {e}")
            return 1
    
    def cmd_import(self, args) -> int:
        """Import database command."""
        try:
            self._initialize_database(args)
            
            input_file = Path(args.input_file)
            if not input_file.exists():
                print(f"❌ Input file not found: {input_file}")
                return 1
            
            with self.db_manager.get_session() as session:
                from .models import SimulationRecord
                
                if getattr(args, 'format', 'json') == 'json':
                    import json
                    
                    with open(input_file) as f:
                        data = json.load(f)
                    
                    count = 0
                    for item in data:
                        # Create simulation record from imported data
                        record = SimulationRecord()
                        for key, value in item.items():
                            if hasattr(record, key) and key not in ['id']:  # Skip primary key
                                setattr(record, key, value)
                        
                        session.add(record)
                        count += 1
                    
                    session.commit()
                    print(f"✅ Imported {count} simulations from {input_file} (JSON)")
                else:
                    # CSV import
                    import csv
                    
                    with open(input_file, newline='') as f:
                        reader = csv.DictReader(f)
                        count = 0
                        for row in reader:
                            record = SimulationRecord()
                            for key, value in row.items():
                                if hasattr(record, key) and key not in ['id']:
                                    setattr(record, key, value)
                            
                            session.add(record)
                            count += 1
                        
                        session.commit()
                    print(f"✅ Imported {count} simulations from {input_file} (CSV)")
            
            return 0
            
        except Exception as e:
            print(f"❌ Failed to import database: {e}")
            return 1
    
    def cmd_health(self, args) -> int:
        """Health check command."""
        try:
            self._initialize_database(args)
            
            # Perform health checks
            health_status = True
            
            # Check database connection
            try:
                with self.db_manager.get_session() as session:
                    from sqlalchemy import text
                    if self.db_manager.config.database_type == 'sqlite':
                        session.execute(text("SELECT 1"))
                    else:
                        session.execute(text("SELECT version()"))
                print("✅ Database connection: OK")
            except Exception as e:
                print(f"❌ Database connection: FAILED ({e})")
                health_status = False
            
            # Check table integrity
            try:
                with self.db_manager.get_session() as session:
                    from .models import SimulationRecord
                    count = session.query(SimulationRecord).count()
                print(f"✅ Table integrity: OK ({count} simulations)")
            except Exception as e:
                print(f"❌ Table integrity: FAILED ({e})")
                health_status = False
            
            # Check disk space (for SQLite)
            if self.db_manager.config.database_type == 'sqlite':
                try:
                    import shutil
                    db_path = Path(self.db_manager.config.database_path)
                    if db_path.exists():
                        file_size = db_path.stat().st_size / (1024 * 1024)  # MB
                        free_space = shutil.disk_usage(db_path.parent).free / (1024 * 1024 * 1024)  # GB
                        print(f"✅ Database size: {file_size:.2f} MB")
                        print(f"✅ Free disk space: {free_space:.2f} GB")
                    else:
                        print("⚠️  Database file does not exist yet")
                except Exception as e:
                    print(f"⚠️  Disk space check: {e}")
            
            if getattr(args, 'format', 'table') == 'json':
                import json
                result = {
                    'status': 'healthy' if health_status else 'unhealthy',
                    'database_type': self.db_manager.config.database_type,
                    'timestamp': datetime.now().isoformat()
                }
                print(json.dumps(result, indent=2))
            else:
                status_icon = "✅" if health_status else "❌"
                print(f"\n{status_icon} Overall Status: {'HEALTHY' if health_status else 'UNHEALTHY'}")
            
            return 0 if health_status else 1
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return 1
    
    def cmd_backup_simple(self, args) -> int:
        """Simple backup command for CLI integration."""
        try:
            self._initialize_database(args)
            
            print("💾 Creating database backup...")
            
            # Create backup config if output directory specified
            output_dir = getattr(args, 'output_dir', None)
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
                # Simple backup creation
                from .backup_manager import BackupManager, BackupConfig
                backup_config = BackupConfig(
                    backup_directory=output_dir,
                    max_backups=10,
                    compression=getattr(args, 'compress', False)
                )
                backup_manager = BackupManager(self.db_manager, backup_config)
            else:
                backup_manager = self.backup_manager
            
            backup_info = backup_manager.create_backup(description="CLI backup")
            
            print(f"✅ Backup created: {backup_info.backup_id}")
            print(f"   File: {backup_info.file_path}")
            print(f"   Size: {backup_info.file_size / (1024*1024):.2f} MB")
            
            return 0
            
        except Exception as e:
            print(f"❌ Backup creation failed: {e}")
            return 1
