#!/usr/bin/env python3
"""
Test script for database integration system

This script tests the comprehensive database integration system
including models, database manager, search engine, and backup manager.
"""

import os
import sys
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_database_models():
    """Test database models and schema."""
    
    print("🧪 Testing Database Models")
    print("="*50)
    
    try:
        from proteinMD.database.models import SimulationRecord, AnalysisRecord, ProteinStructure
        
        # Test model creation
        simulation = SimulationRecord(
            name="Test Simulation",
            description="Test simulation for database testing",
            status="created",
            temperature=300.0,
            total_steps=10000,
            force_field="AMBER_ff14SB"
        )
        
        print("✅ SimulationRecord model created successfully")
        
        # Test model validation
        try:
            simulation.status = "invalid_status"
            simulation.validate_status("status", "invalid_status")
            print("❌ Model validation should have failed")
            return False
        except ValueError:
            print("✅ Model validation works correctly")
        
        # Test to_dict conversion
        simulation.status = "created"
        sim_dict = simulation.to_dict()
        if isinstance(sim_dict, dict) and 'name' in sim_dict:
            print("✅ Model to_dict conversion works")
        else:
            print("❌ Model to_dict conversion failed")
            return False
        
        print("🎉 All database model tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_database_manager():
    """Test database manager functionality."""
    
    print("\n🧪 Testing Database Manager")
    print("="*50)
    
    try:
        from proteinMD.database.database_manager import DatabaseManager, DatabaseConfig
        
        # Create temporary database
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            # Create database config
            config = DatabaseConfig(
                database_type='sqlite',
                database_path=str(db_path),
                create_tables=True
            )
            
            # Initialize database manager
            db_manager = DatabaseManager(config)
            db_manager.connect()
            
            print("✅ Database manager connected successfully")
            
            # Test database info
            info = db_manager.get_database_info()
            if info and 'database_type' in info:
                print("✅ Database info retrieval works")
            else:
                print("❌ Database info retrieval failed")
                return False
            
            # Test simulation storage
            simulation_data = {
                'name': 'Test Simulation',
                'description': 'Test simulation for database manager',
                'status': 'created',
                'temperature': 300.0,
                'total_steps': 10000,
                'force_field': 'AMBER_ff14SB',
                'user_id': 'test_user',
                'project_name': 'test_project'
            }
            
            sim_id = db_manager.store_simulation(simulation_data)
            if sim_id and sim_id > 0:
                print(f"✅ Simulation stored with ID: {sim_id}")
            else:
                print("❌ Failed to store simulation")
                return False
            
            # Test simulation retrieval
            retrieved_sim = db_manager.get_simulation(sim_id)
            if retrieved_sim and retrieved_sim['name'] == 'Test Simulation':
                print("✅ Simulation retrieval works")
            else:
                print("❌ Simulation retrieval failed")
                return False
            
            # Test simulation update
            success = db_manager.update_simulation_status(sim_id, 'running', execution_time=120.5)
            if success:
                updated_sim = db_manager.get_simulation(sim_id)
                if updated_sim['status'] == 'running':
                    print("✅ Simulation update works")
                else:
                    print("❌ Simulation update failed - status not updated")
                    return False
            else:
                print("❌ Simulation update failed")
                return False
            
            # Test simulation listing
            simulations = db_manager.list_simulations(limit=10)
            if simulations and len(simulations) > 0:
                print(f"✅ Simulation listing works ({len(simulations)} found)")
            else:
                print("❌ Simulation listing failed")
                return False
            
            # Test health check
            health = db_manager.check_health()
            if health['status'] == 'healthy':
                print("✅ Database health check works")
            else:
                print("❌ Database health check failed")
                return False
            
            # Clean up
            db_manager.disconnect()
            
        print("🎉 All database manager tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database manager test error: {e}")
        return False

def test_search_engine():
    """Test search engine functionality."""
    
    print("\n🧪 Testing Search Engine")
    print("="*50)
    
    try:
        from proteinMD.database.database_manager import DatabaseManager, DatabaseConfig
        from proteinMD.database.search_engine import SimulationSearchEngine, SearchQuery, SearchOperator
        
        # Create temporary database with test data
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_search.db"
            
            config = DatabaseConfig(
                database_type='sqlite',
                database_path=str(db_path),
                create_tables=True
            )
            
            db_manager = DatabaseManager(config)
            db_manager.connect()
            
            # Create test simulations
            test_simulations = [
                {
                    'name': 'Protein Folding Study',
                    'description': 'Folding simulation of small protein',
                    'status': 'completed',
                    'temperature': 300.0,
                    'total_steps': 100000,
                    'force_field': 'AMBER_ff14SB',
                    'final_energy': -1500.0,
                    'user_id': 'researcher1',
                    'project_name': 'protein_folding'
                },
                {
                    'name': 'Membrane Protein Analysis',
                    'description': 'Analysis of membrane protein dynamics',
                    'status': 'running',
                    'temperature': 310.0,
                    'total_steps': 200000,
                    'force_field': 'CHARMM36',
                    'final_energy': -2000.0,
                    'user_id': 'researcher2',
                    'project_name': 'membrane_studies'
                },
                {
                    'name': 'Drug Binding Simulation',
                    'description': 'Drug-protein binding study',
                    'status': 'failed',
                    'temperature': 295.0,
                    'total_steps': 50000,
                    'force_field': 'AMBER_ff14SB',
                    'final_energy': -800.0,
                    'user_id': 'researcher1',
                    'project_name': 'drug_discovery'
                }
            ]
            
            sim_ids = []
            for sim_data in test_simulations:
                sim_id = db_manager.store_simulation(sim_data)
                sim_ids.append(sim_id)
            
            print(f"✅ Created {len(sim_ids)} test simulations")
            
            # Initialize search engine
            search_engine = SimulationSearchEngine(db_manager)
            
            # Test basic text search
            query = SearchQuery(text_search="protein")
            result = search_engine.search(query)
            
            if result.total_count >= 2:  # Should find at least 2 simulations with "protein"
                print(f"✅ Text search works ({result.total_count} results)")
            else:
                print(f"❌ Text search failed (expected >=2, got {result.total_count})")
                return False
            
            # Test status filter
            query = SearchQuery(status_filter=['completed'])
            result = search_engine.search(query)
            
            if result.total_count == 1:
                print("✅ Status filter works")
            else:
                print(f"❌ Status filter failed (expected 1, got {result.total_count})")
                return False
            
            # Test temperature range filter
            query = SearchQuery(temperature_range=(300.0, 310.0))
            result = search_engine.search(query)
            
            if result.total_count >= 2:
                print("✅ Temperature range filter works")
            else:
                print(f"❌ Temperature range filter failed")
                return False
            
            # Test quick search
            quick_results = search_engine.quick_search("folding")
            if len(quick_results) > 0:
                print("✅ Quick search works")
            else:
                print("❌ Quick search failed")
                return False
            
            # Test search by status
            status_results = search_engine.search_by_status('completed')
            if len(status_results) == 1:
                print("✅ Search by status works")
            else:
                print("❌ Search by status failed")
                return False
            
            db_manager.disconnect()
            
        print("🎉 All search engine tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Search engine test error: {e}")
        return False

def test_backup_manager():
    """Test backup manager functionality."""
    
    print("\n🧪 Testing Backup Manager")
    print("="*50)
    
    try:
        from proteinMD.database.database_manager import DatabaseManager, DatabaseConfig
        from proteinMD.database.backup_manager import BackupManager, BackupConfig
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_backup.db"
            backup_dir = Path(temp_dir) / "backups"
            backup_dir.mkdir()
            
            # Setup database
            config = DatabaseConfig(
                database_type='sqlite',
                database_path=str(db_path),
                create_tables=True
            )
            
            db_manager = DatabaseManager(config)
            db_manager.connect()
            
            # Add some test data
            sim_data = {
                'name': 'Backup Test Simulation',
                'status': 'completed',
                'temperature': 300.0,
                'total_steps': 1000,
                'force_field': 'AMBER_ff14SB'
            }
            db_manager.store_simulation(sim_data)
            
            # Setup backup manager
            backup_config = BackupConfig(
                backup_directory=str(backup_dir),
                compression=True,
                verify_backup=True
            )
            
            backup_manager = BackupManager(db_manager, backup_config)
            
            # Test backup creation
            backup_info = backup_manager.create_backup(
                backup_type='full',
                description='Test backup'
            )
            
            if backup_info and backup_info.backup_id:
                print(f"✅ Backup created: {backup_info.backup_id}")
            else:
                print("❌ Backup creation failed")
                return False
            
            # Test backup listing
            backups = backup_manager.list_backups()
            if len(backups) == 1:
                print("✅ Backup listing works")
            else:
                print(f"❌ Backup listing failed (expected 1, got {len(backups)})")
                return False
            
            # Test backup info retrieval
            retrieved_info = backup_manager.get_backup_info(backup_info.backup_id)
            if retrieved_info and retrieved_info.backup_id == backup_info.backup_id:
                print("✅ Backup info retrieval works")
            else:
                print("❌ Backup info retrieval failed")
                return False
            
            # Test backup statistics
            stats = backup_manager.get_backup_statistics()
            if stats['total_backups'] == 1:
                print("✅ Backup statistics work")
            else:
                print("❌ Backup statistics failed")
                return False
            
            # Test backup restoration (to different location)
            restore_db_path = Path(temp_dir) / "restored.db"
            success = backup_manager.restore_backup(
                backup_info.backup_id,
                target_database=str(restore_db_path),
                overwrite=True
            )
            
            if success and restore_db_path.exists():
                print("✅ Backup restoration works")
            else:
                print("❌ Backup restoration failed")
                return False
            
            db_manager.disconnect()
            
        print("🎉 All backup manager tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Backup manager test error: {e}")
        return False

def test_database_cli():
    """Test database CLI functionality."""
    
    print("\n🧪 Testing Database CLI")
    print("="*50)
    
    try:
        from proteinMD.database.database_cli import DatabaseCLI
        
        # Test CLI instantiation
        cli = DatabaseCLI()
        print("✅ Database CLI instantiated successfully")
        
        # Test parser creation
        parser = cli.create_parser()
        if parser:
            print("✅ CLI parser created successfully")
        else:
            print("❌ CLI parser creation failed")
            return False
        
        # Test help generation (should not raise exception)
        help_text = parser.format_help()
        if 'database' in help_text.lower() and 'init' in help_text:
            print("✅ CLI help generation works")
        else:
            print("❌ CLI help generation failed")
            return False
        
        print("🎉 All database CLI tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database CLI test error: {e}")
        return False

def test_integration():
    """Test full database integration."""
    
    print("\n🧪 Testing Database Integration")
    print("="*50)
    
    try:
        from proteinMD.database import (
            DatabaseManager, DatabaseConfig, SimulationSearchEngine,
            BackupManager, BackupConfig, DatabaseCLI
        )
        
        print("✅ All database modules imported successfully")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Full integration test
            db_path = Path(temp_dir) / "integration_test.db"
            
            # Initialize complete system
            config = DatabaseConfig(
                database_type='sqlite',
                database_path=str(db_path),
                create_tables=True
            )
            
            db_manager = DatabaseManager(config)
            db_manager.connect()
            
            search_engine = SimulationSearchEngine(db_manager)
            
            backup_config = BackupConfig(
                backup_directory=str(Path(temp_dir) / "backups")
            )
            backup_manager = BackupManager(db_manager, backup_config)
            
            cli = DatabaseCLI()
            
            print("✅ Complete database system initialized")
            
            # Test workflow: store -> search -> backup -> restore
            
            # 1. Store simulation
            sim_data = {
                'name': 'Integration Test Simulation',
                'description': 'Full integration test',
                'status': 'completed',
                'temperature': 300.0,
                'total_steps': 50000,
                'force_field': 'AMBER_ff14SB',
                'final_energy': -1200.0,
                'user_id': 'integration_test',
                'project_name': 'testing'
            }
            
            sim_id = db_manager.store_simulation(sim_data)
            print(f"✅ Simulation stored (ID: {sim_id})")
            
            # 2. Search for simulation
            from proteinMD.database.search_engine import SearchQuery
            query = SearchQuery(text_search="integration")
            result = search_engine.search(query)
            
            if result.total_count == 1:
                print("✅ Search integration works")
            else:
                print("❌ Search integration failed")
                return False
            
            # 3. Create backup
            backup_info = backup_manager.create_backup(
                backup_type='full',
                description='Integration test backup'
            )
            print(f"✅ Backup created: {backup_info.backup_id}")
            
            # 4. Test CLI command parsing
            parser = cli.create_parser()
            args = parser.parse_args(['info'])
            if args.command == 'info':
                print("✅ CLI command parsing works")
            else:
                print("❌ CLI command parsing failed")
                return False
            
            db_manager.disconnect()
            
        print("🎉 All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        return False

def main():
    """Run all database tests."""
    print("🚀 ProteinMD Database Integration Tests")
    print("="*60)
    
    tests = [
        test_database_models,
        test_database_manager,
        test_search_engine,
        test_backup_manager,
        test_database_cli,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print("="*60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All database tests successful!")
        print("📊 Database Integration (Task 9.1) is ready!")
        return 0
    else:
        print("❌ Some tests failed - database integration needs attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
