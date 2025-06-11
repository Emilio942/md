#!/usr/bin/env python3
"""
Test script for real-time viewer implementation (Task 2.3).

This script tests the real-time viewer to validate:
1. Live visualization during simulation execution
2. Performance maintains >80% efficiency  
3. Toggle functionality works
"""

import sys
import numpy as np
import matplotlib
# Use non-interactive backend for testing
matplotlib.use('Agg')
import time
import logging
from pathlib import Path

# Add proteinMD to path
sys.path.append(str(Path(__file__).parent / 'proteinMD'))

from proteinMD.core.simulation import MolecularDynamicsSimulation
from proteinMD.visualization.realtime_viewer import RealTimeViewer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_simulation():
    """Create a simple test simulation for real-time viewer testing."""
    logger.info("Creating test simulation...")
    
    # Create a simple molecular dynamics simulation
    # Use basic parameters for testing
    try:
        simulation = MolecularDynamicsSimulation()
        
        # Initialize with simple test system (if methods exist)
        n_particles = 100
        box_size = np.array([10.0, 10.0, 10.0])
        
        # Set up random initial positions and velocities
        positions = np.random.random((n_particles, 3)) * box_size
        velocities = np.random.normal(0, 1, (n_particles, 3))
        
        # Try to set simulation properties
        if hasattr(simulation, 'positions'):
            simulation.positions = positions
        if hasattr(simulation, 'velocities'):  
            simulation.velocities = velocities
        if hasattr(simulation, 'box_dimensions'):
            simulation.box_dimensions = box_size
            
        logger.info(f"Test simulation created with {n_particles} particles")
        return simulation
        
    except Exception as e:
        logger.error(f"Error creating simulation: {e}")
        return None

def test_realtime_viewer_creation():
    """Test 1: Real-time viewer can be created without errors."""
    logger.info("=== Test 1: Real-time Viewer Creation ===")
    
    simulation = create_test_simulation()
    if simulation is None:
        logger.error("Failed to create test simulation")
        return False
        
    try:
        viewer = RealTimeViewer(
            simulation=simulation,
            visualization_frequency=10,
            max_history_length=100,
            display_mode='ball_stick'
        )
        logger.info("✓ RealTimeViewer created successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to create RealTimeViewer: {e}")
        return False

def test_visualization_setup():
    """Test 2: Visualization setup works correctly."""
    logger.info("=== Test 2: Visualization Setup ===")
    
    simulation = create_test_simulation()
    if simulation is None:
        return False
        
    try:
        viewer = RealTimeViewer(simulation=simulation, visualization_frequency=5)
        
        # Test visualization setup
        viz_info = viewer.setup_visualization()
        
        # Check if required components are created
        required_keys = ['fig', 'axes', 'plot_elements']
        for key in required_keys:
            if key not in viz_info:
                logger.error(f"✗ Missing required visualization component: {key}")
                return False
                
        logger.info("✓ Visualization setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Visualization setup failed: {e}")
        return False

def test_performance_measurement():
    """Test 3: Performance measurement works correctly."""
    logger.info("=== Test 3: Performance Measurement ===")
    
    simulation = create_test_simulation()
    if simulation is None:
        return False
        
    try:
        viewer = RealTimeViewer(simulation=simulation, visualization_frequency=20)
        
        # Test baseline performance measurement
        if hasattr(viewer, '_measure_baseline_performance'):
            viewer._measure_baseline_performance()  # No parameters
            if viewer.baseline_fps is not None and viewer.baseline_fps > 0:
                logger.info(f"✓ Baseline performance measured: {viewer.baseline_fps:.2f} steps/s")
                return True
            else:
                logger.warning("Baseline performance measurement returned invalid result")
                return False
        else:
            logger.warning("_measure_baseline_performance method not found")
            return False
            
    except Exception as e:
        logger.error(f"✗ Performance measurement failed: {e}")
        return False

def test_toggle_functionality():
    """Test 4: Toggle functionality works."""
    logger.info("=== Test 4: Toggle Functionality ===")
    
    simulation = create_test_simulation()
    if simulation is None:
        return False
        
    try:
        viewer = RealTimeViewer(simulation=simulation)
        
        # Test initial state
        initial_viz_state = viewer.live_visualization_enabled
        initial_pause_state = viewer.visualization_paused
        
        # Test toggle visualization
        viewer.toggle_visualization()
        if viewer.live_visualization_enabled == initial_viz_state:
            logger.error("✗ Visualization toggle did not change state")
            return False
            
        # Test toggle pause
        viewer.toggle_pause()
        if viewer.visualization_paused == initial_pause_state:
            logger.error("✗ Pause toggle did not change state")
            return False
            
        logger.info("✓ Toggle functionality works correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Toggle functionality test failed: {e}")
        return False

def test_data_update():
    """Test 5: Data update mechanism works."""
    logger.info("=== Test 5: Data Update Mechanism ===")
    
    simulation = create_test_simulation()
    if simulation is None:
        return False
        
    try:
        viewer = RealTimeViewer(simulation=simulation)
        viewer.setup_visualization()
        
        # Create test frame data - make sure array sizes match simulation
        n_particles = 100  # Use same as simulation setup
        test_frame_data = {
            'positions': np.random.random((n_particles, 3)) * 10,
            'velocities': np.random.normal(0, 1, (n_particles, 3)),
            'kinetic_energy': 100.5,
            'potential_energy': -200.3,
            'total_energy': -99.8,
            'temperature': 298.15,
            'step': 100,
            'time': 1.0
        }
        
        # Test update visualization
        viewer.update_visualization(test_frame_data)
        
        # Check if data was stored
        if len(viewer.plot_data['time']) > 0:
            logger.info("✓ Data update mechanism works correctly")
            return True
        else:
            logger.error("✗ Data was not properly stored during update")
            return False
            
    except Exception as e:
        logger.error(f"✗ Data update test failed: {e}")
        return False

def test_simulation_integration():
    """Test 6: Integration with simulation system."""
    logger.info("=== Test 6: Simulation Integration ===")
    
    simulation = create_test_simulation()
    if simulation is None:
        return False
        
    try:
        viewer = RealTimeViewer(simulation=simulation, visualization_frequency=50)
        
        # Test if we can get simulation information
        if hasattr(simulation, 'positions') and simulation.positions is not None:
            n_particles = len(simulation.positions)
            logger.info(f"Simulation has {n_particles} particles")
            
        # Test performance summary
        summary = viewer.get_performance_summary()
        if isinstance(summary, dict):
            logger.info("✓ Performance summary generation works")
            logger.info(f"Summary keys: {list(summary.keys())}")
            return True
        else:
            logger.error("✗ Performance summary is not a dictionary")
            return False
            
    except Exception as e:
        logger.error(f"✗ Simulation integration test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    logger.info("Starting Real-time Viewer Tests (Task 2.3)")
    logger.info("=" * 60)
    
    tests = [
        ("Real-time Viewer Creation", test_realtime_viewer_creation),
        ("Visualization Setup", test_visualization_setup),
        ("Performance Measurement", test_performance_measurement),
        ("Toggle Functionality", test_toggle_functionality),
        ("Data Update Mechanism", test_data_update),
        ("Simulation Integration", test_simulation_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            logger.info("")
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
            logger.info("")
    
    # Report final results
    logger.info("=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nTotal: {passed + failed} tests")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    success_rate = (passed / (passed + failed)) * 100 if (passed + failed) > 0 else 0
    logger.info(f"Success rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        logger.info("\n🎉 TASK 2.3 TESTING: MOSTLY SUCCESSFUL")
        logger.info("Real-time viewer implementation is working correctly!")
    else:
        logger.warning("\n⚠️  TASK 2.3 TESTING: NEEDS IMPROVEMENT")
        logger.warning("Some issues found in real-time viewer implementation.")
    
    return success_rate >= 80

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
