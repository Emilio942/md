#!/usr/bin/env python3
"""
Database CLI Integration Test

This script tests the full integration of the database system into the main ProteinMD CLI.
"""

import subprocess
import tempfile
import shutil
import json
import os
from pathlib import Path

def run_cli_command(cmd, capture=True):
    """Run a CLI command and return result."""
    full_cmd = f"python -m proteinMD.cli {cmd}"
    try:
        if capture:
            result = subprocess.run(
                full_cmd, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=30
            )
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(full_cmd, shell=True, timeout=30)
            return result.returncode, "", ""
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"

def test_database_cli_integration():
    """Test complete database CLI integration."""
    print("🚀 ProteinMD Database CLI Integration Tests")
    print("=" * 60)
    
    # Create temporary database for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_cli.db")
        
        tests_passed = 0
        tests_total = 0
        
        # Test 1: Database help
        print("\n🧪 Test 1: Database help command")
        tests_total += 1
        code, stdout, stderr = run_cli_command("database --help")
        if code == 0 and "Database operations" in stdout:
            print("✅ Database help works")
            tests_passed += 1
        else:
            print(f"❌ Database help failed: {code}")
        
        # Test 2: Database init
        print("\n🧪 Test 2: Database initialization")
        tests_total += 1
        code, stdout, stderr = run_cli_command(f"database init --type sqlite --path {db_path}")
        if code == 0:
            print("✅ Database initialization works")
            tests_passed += 1
        else:
            print(f"❌ Database init failed: {code}, {stderr}")
        
        # Test 3: Database health check
        print("\n🧪 Test 3: Database health check")
        tests_total += 1
        code, stdout, stderr = run_cli_command(f"database health --config {db_path}")
        if code == 0 and "HEALTHY" in stdout:
            print("✅ Database health check works")
            tests_passed += 1
        else:
            print(f"❌ Database health failed: {code}")
        
        # Test 4: Database stats
        print("\n🧪 Test 4: Database statistics")
        tests_total += 1
        code, stdout, stderr = run_cli_command(f"database stats --config {db_path}")
        if code == 0 and "Database Statistics" in stdout:
            print("✅ Database stats work")
            tests_passed += 1
        else:
            print(f"❌ Database stats failed: {code}")
        
        # Test 5: List empty database
        print("\n🧪 Test 5: List simulations (empty)")
        tests_total += 1
        code, stdout, stderr = run_cli_command(f"database list --config {db_path}")
        if code == 0 and "0 results" in stdout:
            print("✅ Database list (empty) works")
            tests_passed += 1
        else:
            print(f"❌ Database list failed: {code}")
        
        # Test 6: Database backup
        print("\n🧪 Test 6: Database backup")
        tests_total += 1
        backup_dir = os.path.join(tmpdir, "backups")
        code, stdout, stderr = run_cli_command(f"database backup --config {db_path} --output-dir {backup_dir}")
        if code == 0:
            print("✅ Database backup works")
            tests_passed += 1
        else:
            print(f"❌ Database backup failed: {code}")
        
        # Test 7: JSON output format
        print("\n🧪 Test 7: JSON output format")
        tests_total += 1
        # Use stderr redirect to get clean JSON output
        code, stdout, stderr = run_cli_command(f"database stats --config {db_path} --format json 2>/dev/null")
        if code == 0:
            try:
                # Parse the JSON output directly
                json_content = stdout.strip()
                data = json.loads(json_content)
                if 'total_simulations' in data:
                    print("✅ JSON output format works")
                    tests_passed += 1
                else:
                    print("❌ JSON output missing expected fields")
            except Exception as e:
                print(f"❌ JSON output format invalid: {e}")
                print(f"Raw output: {repr(stdout)}")
        else:
            print(f"❌ JSON format failed: {code}")
        
        # Test 8: Database export (empty)
        print("\n🧪 Test 8: Database export")
        tests_total += 1
        export_file = os.path.join(tmpdir, "export.json")
        code, stdout, stderr = run_cli_command(f"database export {export_file} --config {db_path}")
        if code == 0 and os.path.exists(export_file):
            print("✅ Database export works")
            tests_passed += 1
        else:
            print(f"❌ Database export failed: {code}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Database CLI Integration Test Results:")
    print(f"✅ Passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("🎉 All database CLI integration tests passed!")
        print("📊 Task 9.1 Database Integration is COMPLETE!")
        return 0
    else:
        print("❌ Some tests failed - database CLI integration needs attention")
        return 1

if __name__ == "__main__":
    exit(test_database_cli_integration())
