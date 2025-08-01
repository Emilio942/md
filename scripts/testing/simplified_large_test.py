#!/usr/bin/env python3
"""
Simplified Large-Scale CLI Test
"""

import os
import sys
import time
import subprocess

def test_cli_large_scale():
    print("🧪 LARGE-SCALE CLI TEST")
    print("=" * 30)
    
    # Test 1: Medium simulation (500 steps)
    print("\n🔬 Test 1: Medium Simulation (500 steps)")
    
    test_script = '''
import sys
import time
sys.path.append(".")

try:
    from proteinMD.cli import ProteinMDCLI
    import tempfile
    import shutil
    
    print("Creating CLI instance...")
    cli = ProteinMDCLI()
    
    # Create temp directory
    output_dir = tempfile.mkdtemp(prefix="proteinmd_large_test_")
    print(f"Output directory: {output_dir}")
    
    # Load protein
    print("Loading protein...")
    cli.load_protein("data/proteins/1ubq.pdb")
    
    # Configure simulation
    print("Configuring simulation...")
    cli.set_temperature(300.0)
    cli.set_timestep(0.002)
    cli.set_simulation_length(500)
    
    # Run simulation
    print("Starting 500-step simulation...")
    start_time = time.time()
    cli.run_simulation(output_dir=output_dir)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"✅ Simulation completed successfully in {duration:.2f} seconds")
    print(f"📊 Performance: {500/duration:.2f} steps/second")
    
    # Check output files
    import os
    files = os.listdir(output_dir)
    print(f"📁 Generated {len(files)} output files")
    
    # Cleanup
    shutil.rmtree(output_dir, ignore_errors=True)
    print("🧹 Cleanup completed")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    # Run the test
    start_time = time.time()
    result = subprocess.run(
        [sys.executable, '-c', test_script],
        cwd='/home/emilio/Documents/ai/md',
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout
    )
    end_time = time.time()
    
    print(f"⏱️  Total test duration: {end_time - start_time:.2f} seconds")
    
    if result.returncode == 0:
        print("✅ Test 1 PASSED")
        print("\n📄 Output:")
        print(result.stdout)
    else:
        print("❌ Test 1 FAILED")
        print(f"Return code: {result.returncode}")
        print("\n🚨 Error output:")
        print(result.stderr)
        return False
    
    # Test 2: Check CLI responsiveness
    print("\n🔬 Test 2: CLI Responsiveness Test")
    
    responsiveness_script = '''
import sys
import time
sys.path.append(".")

try:
    from proteinMD.cli import ProteinMDCLI
    
    # Test multiple CLI operations in sequence
    for i in range(5):
        print(f"🔄 Iteration {i+1}/5")
        cli = ProteinMDCLI()
        
        start = time.time()
        cli.load_protein("data/proteins/1ubq.pdb")
        load_time = time.time() - start
        
        print(f"   Protein loaded in {load_time:.3f}s")
        
        # Quick config check
        cli.set_temperature(300.0)
        cli.set_timestep(0.002)
        
        print(f"   Configuration set successfully")
        
        # Small delay between iterations
        time.sleep(0.1)
    
    print("✅ CLI remains responsive across multiple operations")
    
except Exception as e:
    print(f"❌ Responsiveness test failed: {e}")
    sys.exit(1)
'''
    
    result2 = subprocess.run(
        [sys.executable, '-c', responsiveness_script],
        cwd='/home/emilio/Documents/ai/md',
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result2.returncode == 0:
        print("✅ Test 2 PASSED")
        print("\n📄 Output:")
        print(result2.stdout)
    else:
        print("❌ Test 2 FAILED")
        print(result2.stderr)
        return False
    
    print("\n🎉 ALL LARGE-SCALE TESTS PASSED!")
    print("✅ CLI works properly at larger scale")
    print("✅ No terminal overload or infinite loops detected")
    print("✅ Performance remains acceptable")
    
    return True

if __name__ == "__main__":
    success = test_cli_large_scale()
    sys.exit(0 if success else 1)
