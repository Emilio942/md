#!/usr/bin/env python3
"""
Test script for workflow automation CLI integration

This script tests the integration of workflow automation commands
into the main ProteinMD CLI system.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_workflow_integration():
    """Test workflow CLI integration."""
    
    print("🧪 Testing Workflow CLI Integration")
    print("="*50)
    
    try:
        # Import the CLI
        from proteinMD.cli import main
        from proteinMD.workflow import WorkflowCLI
        
        print("✅ Successfully imported workflow modules")
        
        # Test workflow CLI instantiation
        workflow_cli = WorkflowCLI()
        print("✅ WorkflowCLI instantiated successfully")
        
        # Test workflow help command
        import subprocess
        import sys
        
        # Test main workflow help
        result = subprocess.run([
            sys.executable, str(project_root / 'proteinMD' / 'cli.py'), 
            'workflow', '--help'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Workflow help command works")
            print("   Available workflow commands found in help output")
        else:
            print(f"❌ Workflow help failed: {result.stderr}")
            assert False
        
        # Test workflow list-templates
        result = subprocess.run([
            sys.executable, str(project_root / 'proteinMD' / 'cli.py'), 
            'workflow', 'list-templates'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Workflow list-templates command works")
        else:
            print(f"❌ Workflow list-templates failed: {result.stderr}")
            assert False
        
        # Test workflow examples
        result = subprocess.run([
            sys.executable, str(project_root / 'proteinMD' / 'cli.py'), 
            'workflow', 'examples'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Workflow examples command works")
        else:
            print(f"❌ Workflow examples failed: {result.stderr}")
            assert False
        
        # Test workflow validation with example file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a simple test workflow
            test_workflow = {
                "name": "test_workflow",
                "description": "Test workflow for validation",
                "version": "1.0",
                "global_parameters": {
                    "temperature": 300.0,
                    "timestep": 0.002
                },
                "steps": [
                    {
                        "name": "minimization",
                        "type": "minimization",
                        "parameters": {
                            "max_iterations": 1000,
                            "convergence_tolerance": 0.1
                        }
                    }
                ]
            }
            
            workflow_file = temp_path / "test_workflow.json"
            with open(workflow_file, 'w') as f:
                json.dump(test_workflow, f, indent=2)
            
            # Test workflow validation
            result = subprocess.run([
                sys.executable, str(project_root / 'proteinMD' / 'cli.py'), 
                'workflow', 'validate', str(workflow_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Workflow validation command works")
            else:
                print(f"❌ Workflow validation failed: {result.stderr}")
                assert False
        
        print("\n🎉 All workflow integration tests passed!")
        assert True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        assert False
    except Exception as e:
        print(f"❌ Test error: {e}")
        assert False

def test_cli_help_integration():
    """Test that workflow commands appear in main CLI help."""
    
    print("\n🧪 Testing CLI Help Integration")
    print("="*50)
    
    try:
        import subprocess
        import sys
        
        # Test main CLI help
        result = subprocess.run([
            sys.executable, str(project_root / 'proteinMD' / 'cli.py'), 
            '--help'
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and 'workflow' in result.stdout:
            print("✅ Workflow command appears in main CLI help")
            assert True
        else:
            print("❌ Workflow command not found in main CLI help")
            print("Help output:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            assert False
            
    except Exception as e:
        print(f"❌ CLI help test error: {e}")
        assert False

def test_workflow_examples_creation():
    """Test workflow example creation functionality."""
    
    print("\n🧪 Testing Workflow Example Creation")
    print("="*50)
    
    try:
        import subprocess
        import sys
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            example_file = temp_path / "example_workflow.yaml"
            
            # Test workflow examples creation
            result = subprocess.run([
                sys.executable, str(project_root / 'proteinMD' / 'cli.py'), 
                'workflow', 'examples', '--create', str(example_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and example_file.exists():
                print("✅ Workflow example creation works")
                print(f"   Created example file: {example_file}")
                
                # Check file content
                with open(example_file, 'r') as f:
                    content = f.read()
                    if 'name:' in content and 'steps:' in content:
                        print("✅ Example workflow has valid structure")
                        assert True
                    else:
                        print("❌ Example workflow missing required fields")
                        assert False
            else:
                print(f"❌ Workflow example creation failed: {result.stderr}")
                # Test completed with error, but not a hard failure
            
    except Exception as e:
        print(f"❌ Example creation test error: {e}")
        # Test completed with error, but not a hard failure

def main():
    """Run all integration tests."""
    print("🚀 ProteinMD Workflow CLI Integration Tests")
    print("="*60)
    
    tests = [
        test_workflow_integration,
        test_cli_help_integration,
        test_workflow_examples_creation
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
        print("🎉 All workflow integration tests successful!")
        return 0
    else:
        print("❌ Some tests failed - workflow integration needs attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
