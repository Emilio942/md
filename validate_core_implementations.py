#!/usr/bin/env python3
"""
Core Implementation Validation for Tasks 9.2 & 9.3
==================================================

Test the core implementations that are working:
- Task 9.2: Cloud Storage Integration
- Task 9.3: Metadata Management
"""

import os
import sys
import json
import tempfile
import unittest
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/emilio/Documents/ai/md')

def test_cloud_storage():
    """Test cloud storage functionality."""
    print("\n=== Testing Cloud Storage Integration (Task 9.2) ===")
    
    try:
        from proteinMD.database.cloud_storage import (
            CloudStorageManager, CloudConfig, CachePolicy, 
            EncryptionManager, LocalCache, CloudFile
        )
        print("✅ All cloud storage imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test encryption functionality
    try:
        print("\n1. Testing Encryption Manager...")
        encryption = EncryptionManager("test_password")
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Test data for encryption")
            test_file = f.name
        
        # Test encryption/decryption
        encrypted_file = test_file + ".enc"
        decrypted_file = test_file + ".dec"
        
        success = encryption.encrypt_file(test_file, encrypted_file)
        if success:
            print("   ✅ File encryption successful")
        else:
            print("   ❌ File encryption failed")
            return False
            
        success = encryption.decrypt_file(encrypted_file, decrypted_file)
        if success:
            print("   ✅ File decryption successful")
        else:
            print("   ❌ File decryption failed")
            return False
            
        # Verify content
        with open(test_file, 'r') as f1, open(decrypted_file, 'r') as f2:
            if f1.read() == f2.read():
                print("   ✅ Encryption/decryption round-trip successful")
            else:
                print("   ❌ Content mismatch after decryption")
                return False
        
        # Cleanup
        for f in [test_file, encrypted_file, decrypted_file]:
            if os.path.exists(f):
                os.unlink(f)
                
    except Exception as e:
        print(f"   ❌ Encryption test failed: {e}")
        return False
    
    # Test cache functionality
    try:
        print("\n2. Testing Local Cache...")
        
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_policy = CachePolicy(
                max_size_gb=1.0,
                max_age_days=7,
                cleanup_threshold=0.8
            )
            
            cache = LocalCache(cache_dir, cache_policy)
            
            # Test cache operations
            test_data = b"Test cache data"
            cache_key = "test_file.dat"
            
            cache.store(cache_key, test_data)
            if cache.exists(cache_key):
                print("   ✅ Cache store operation successful")
            else:
                print("   ❌ Cache store failed")
                return False
            
            retrieved_data = cache.get(cache_key)
            if retrieved_data == test_data:
                print("   ✅ Cache retrieval successful")
            else:
                print("   ❌ Cache retrieval failed")
                return False
                
    except Exception as e:
        print(f"   ❌ Cache test failed: {e}")
        return False
    
    print("\n✅ Cloud Storage Integration (Task 9.2) validation PASSED")
    return True

def test_metadata_management():
    """Test metadata management functionality."""
    print("\n=== Testing Metadata Management (Task 9.3) ===")
    
    try:
        from proteinMD.database.metadata_manager import (
            MetadataManager, MetadataType, MetadataEntry, 
            TagManager, Tag, ProvenanceTracker,
            MetadataQueryBuilder, create_simulation_metadata
        )
        print("✅ All metadata management imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    try:
        print("\n1. Testing Metadata Manager...")
        
        # Initialize metadata manager
        metadata_manager = MetadataManager()
        
        # Test simulation metadata creation
        sim_metadata = create_simulation_metadata(
            parameters={
                'timestep': 0.002,
                'temperature': 300.0,
                'steps': 100000
            },
            forcefield_info={
                'name': 'CHARMM36',
                'version': '1.0'
            },
            system_info={
                'n_atoms': 1000,
                'box_size': [5.0, 5.0, 5.0]
            }
        )
        
        if sim_metadata:
            print("   ✅ Simulation metadata creation successful")
        else:
            print("   ❌ Simulation metadata creation failed")
            return False
        
        # Test metadata storage
        metadata_id = metadata_manager.store_metadata(sim_metadata)
        if metadata_id:
            print("   ✅ Metadata storage successful")
        else:
            print("   ❌ Metadata storage failed")
            return False
        
        # Test metadata retrieval
        retrieved = metadata_manager.get_metadata(metadata_id)
        if retrieved:
            print("   ✅ Metadata retrieval successful")
        else:
            print("   ❌ Metadata retrieval failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Metadata manager test failed: {e}")
        return False
    
    try:
        print("\n2. Testing Tag Manager...")
        
        tag_manager = TagManager()
        
        # Create hierarchical tags
        parent_tag = Tag(
            name="molecular_dynamics",
            description="Molecular dynamics simulations",
            category="simulation_type"
        )
        
        child_tag = Tag(
            name="protein_folding",
            description="Protein folding simulations",
            category="simulation_type",
            parent_id=parent_tag.id
        )
        
        # Store tags
        parent_id = tag_manager.create_tag(parent_tag)
        child_id = tag_manager.create_tag(child_tag)
        
        if parent_id and child_id:
            print("   ✅ Hierarchical tag creation successful")
        else:
            print("   ❌ Tag creation failed")
            return False
        
        # Test tag relationships
        children = tag_manager.get_child_tags(parent_id)
        if children and len(children) > 0:
            print("   ✅ Tag hierarchy retrieval successful")
        else:
            print("   ❌ Tag hierarchy retrieval failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Tag manager test failed: {e}")
        return False
    
    try:
        print("\n3. Testing Provenance Tracker...")
        
        provenance_tracker = ProvenanceTracker()
        
        # Create a simple provenance record
        entity_id = "test_simulation_001"
        entity_type = "simulation"
        
        # Track creation
        provenance_tracker.track_creation(
            entity_id=entity_id,
            entity_type=entity_type,
            metadata={'parameters': {'temperature': 300.0}},
            user="test_user"
        )
        
        # Track derivation
        analysis_id = "analysis_001"
        provenance_tracker.track_derivation(
            derived_entity_id=analysis_id,
            derived_entity_type="analysis",
            source_entities=[entity_id],
            operation="trajectory_analysis",
            metadata={'analysis_type': 'rmsd'},
            user="test_user"
        )
        
        # Get lineage
        lineage = provenance_tracker.get_lineage(analysis_id)
        if lineage and len(lineage) > 0:
            print("   ✅ Provenance tracking successful")
        else:
            print("   ❌ Provenance tracking failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Provenance tracker test failed: {e}")
        return False
    
    try:
        print("\n4. Testing Query Builder...")
        
        query_builder = MetadataQueryBuilder()
        
        # Build a complex query
        query_builder.filter_by_type(MetadataType.SIMULATION)
        query_builder.filter_by_tag("molecular_dynamics")
        query_builder.filter_by_date_range(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31)
        )
        query_builder.sort_by("created_at", ascending=False)
        
        # Get the built query
        query = query_builder.build()
        if query and 'filters' in query:
            print("   ✅ Query builder successful")
        else:
            print("   ❌ Query builder failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Query builder test failed: {e}")
        return False
    
    print("\n✅ Metadata Management (Task 9.3) validation PASSED")
    return True

def main():
    """Run all validation tests."""
    print("ProteinMD Task 9.2 & 9.3 Core Implementation Validation")
    print("=" * 60)
    
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Suppress info logs during testing
    
    # Run tests
    task_9_2_success = test_cloud_storage()
    task_9_3_success = test_metadata_management()
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Task 9.2 (Cloud Storage Integration): {'✅ PASSED' if task_9_2_success else '❌ FAILED'}")
    print(f"Task 9.3 (Metadata Management): {'✅ PASSED' if task_9_3_success else '❌ FAILED'}")
    
    overall_success = task_9_2_success and task_9_3_success
    print(f"\nOverall validation: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")
    
    if overall_success:
        print("\n🎉 Both Task 9.2 and Task 9.3 implementations are working correctly!")
        print("Ready to mark these tasks as completed and proceed with next priorities.")
    else:
        print("\n⚠️  Some tests failed. Please review the implementation before proceeding.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
