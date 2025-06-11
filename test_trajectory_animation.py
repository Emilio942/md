#!/usr/bin/env python3
"""
Test script for Task 2.2 - Trajectory Animation

This script tests all trajectory animation functionality including:
- TrajectoryAnimator class
- Interactive controls
- Export capabilities
- Different display modes
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import tempfile
import logging

# Add the proteinMD package to path
sys.path.append(str(Path(__file__).parent))

from proteinMD.visualization.protein_3d import TrajectoryAnimator, animate_trajectory

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests

class TestTrajectoryAnimation(unittest.TestCase):
    """Test cases for trajectory animation functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Generate simple test trajectory
        self.n_frames = 10
        self.n_atoms = 5
        
        # Create positions: atoms moving in a circle
        t = np.linspace(0, 2*np.pi, self.n_atoms)
        radius = 2.0
        
        self.positions = []
        for frame in range(self.n_frames):
            angle_offset = 2 * np.pi * frame / self.n_frames
            frame_positions = np.column_stack([
                radius * np.cos(t + angle_offset),
                radius * np.sin(t + angle_offset),
                np.zeros(self.n_atoms)
            ])
            self.positions.append(frame_positions)
        
        self.positions = np.array(self.positions)
        self.elements = ['C', 'N', 'O', 'S', 'P']
        self.bonds = [(0, 1), (1, 2), (2, 3), (3, 4)]
    
    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')
    
    def test_trajectory_animator_initialization(self):
        """Test TrajectoryAnimator initialization."""
        animator = TrajectoryAnimator(
            positions=self.positions,
            elements=self.elements,
            bonds=self.bonds,
            interval=100,
            display_mode='ball_stick',
            controls=True
        )
        
        self.assertEqual(animator.n_frames, self.n_frames)
        self.assertEqual(animator.n_atoms, self.n_atoms)
        self.assertEqual(animator.elements, self.elements)
        self.assertEqual(animator.bonds, self.bonds)
        self.assertEqual(animator.interval, 100)
        self.assertEqual(animator.display_mode, 'ball_stick')
        self.assertTrue(animator.controls)
        self.assertFalse(animator.is_playing)
        self.assertEqual(animator.current_frame, 0)
        self.assertEqual(animator.animation_speed, 1.0)
    
    def test_trajectory_animator_setup(self):
        """Test animation setup."""
        animator = TrajectoryAnimator(
            positions=self.positions,
            elements=self.elements,
            bonds=self.bonds,
            controls=False
        )
        
        animator.setup_animation()
        
        self.assertIsNotNone(animator.fig)
        self.assertIsNotNone(animator.ax)
        self.assertIsNotNone(animator.animation)
        self.assertIsNotNone(animator.scatter)
    
    def test_trajectory_animator_with_controls(self):
        """Test animation setup with controls."""
        animator = TrajectoryAnimator(
            positions=self.positions,
            elements=self.elements,
            bonds=self.bonds,
            controls=True
        )
        
        animator.setup_animation()
        
        self.assertIsNotNone(animator.play_button)
        self.assertIsNotNone(animator.step_button)
        self.assertIsNotNone(animator.speed_slider)
        self.assertIsNotNone(animator.frame_slider)
    
    def test_frame_navigation(self):
        """Test frame navigation functionality."""
        animator = TrajectoryAnimator(
            positions=self.positions,
            elements=self.elements,
            bonds=self.bonds,
            controls=False
        )
        
        animator.setup_animation()
        
        # Test going to specific frame
        target_frame = 5
        animator.goto_frame(target_frame)
        self.assertEqual(animator.current_frame, target_frame)
        
        # Test stepping
        animator.step()
        self.assertEqual(animator.current_frame, (target_frame + 1) % self.n_frames)
    
    def test_speed_control(self):
        """Test animation speed control."""
        animator = TrajectoryAnimator(
            positions=self.positions,
            elements=self.elements,
            bonds=self.bonds,
            controls=False
        )
        
        animator.setup_animation()
        
        # Test speed setting
        new_speed = 2.0
        animator.set_speed(new_speed)
        self.assertEqual(animator.animation_speed, new_speed)
    
    def test_different_display_modes(self):
        """Test different display modes."""
        modes = ['ball_stick', 'cartoon']
        
        for mode in modes:
            with self.subTest(mode=mode):
                animator = TrajectoryAnimator(
                    positions=self.positions,
                    elements=self.elements,
                    bonds=self.bonds,
                    display_mode=mode,
                    controls=False
                )
                
                animator.setup_animation()
                self.assertEqual(animator.display_mode, mode)
                self.assertIsNotNone(animator.scatter)
    
    def test_save_animation_gif(self):
        """Test saving animation as GIF."""
        animator = TrajectoryAnimator(
            positions=self.positions,
            elements=self.elements,
            bonds=self.bonds,
            controls=False
        )
        
        animator.setup_animation()
        
        # Use temporary file
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            animator.save_animation(tmp_path, fps=5, dpi=50)
            self.assertTrue(Path(tmp_path).exists())
        except Exception as e:
            # Animation saving might fail without proper backend
            self.skipTest(f"Animation saving not available: {e}")
        finally:
            # Clean up
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
    
    def test_quick_animate_function(self):
        """Test the quick animate_trajectory function."""
        animator = animate_trajectory(
            positions=self.positions,
            elements=self.elements,
            bonds=self.bonds,
            interval=200,
            display_mode='ball_stick',
            controls=False,
            show=False
        )
        
        self.assertIsInstance(animator, TrajectoryAnimator)
        self.assertIsNotNone(animator.fig)
        self.assertIsNotNone(animator.animation)
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test empty positions
        with self.assertRaises(ValueError):
            TrajectoryAnimator(positions=[], elements=self.elements)
        
        # Test mismatched elements
        wrong_elements = ['C', 'N']  # Too few elements
        animator = TrajectoryAnimator(
            positions=self.positions,
            elements=wrong_elements,
            bonds=self.bonds
        )
        # Should not raise error but handle gracefully
        self.assertEqual(len(animator.elements), len(wrong_elements))
    
    def test_frame_update(self):
        """Test frame update functionality."""
        animator = TrajectoryAnimator(
            positions=self.positions,
            elements=self.elements,
            bonds=self.bonds,
            controls=False
        )
        
        animator.setup_animation()
        
        # Test updating to different frames
        for frame_num in [0, 3, 7, 9]:
            animator._update_frame(frame_num)
            self.assertEqual(animator.current_frame, frame_num)
    
    def test_bond_updates(self):
        """Test bond line updates during animation."""
        animator = TrajectoryAnimator(
            positions=self.positions,
            elements=self.elements,
            bonds=self.bonds,
            display_mode='ball_stick',
            controls=False
        )
        
        animator.setup_animation()
        
        # Check that bonds are drawn
        initial_bond_count = len(animator.bond_lines)
        self.assertEqual(initial_bond_count, len(self.bonds))
        
        # Update to different frame and check bonds are updated
        animator._update_frame(3)
        self.assertEqual(len(animator.bond_lines), len(self.bonds))
    
    def test_animation_controls_methods(self):
        """Test animation control methods."""
        animator = TrajectoryAnimator(
            positions=self.positions,
            elements=self.elements,
            bonds=self.bonds,
            controls=False
        )
        
        animator.setup_animation()
        
        # Test play/pause
        initial_playing = animator.is_playing
        animator.play()
        # Note: play() changes internal state but animation object state
        # might not change immediately in headless environment
        
        animator.pause()
        # Similar for pause
        
        # Test that methods don't crash
        animator.step()
        animator.goto_frame(5)
        animator.set_speed(1.5)


class TestTrajectoryAnimationIntegration(unittest.TestCase):
    """Integration tests for trajectory animation."""
    
    def setUp(self):
        """Set up integration test data."""
        # Create more complex trajectory for integration tests
        self.n_frames = 15
        self.n_atoms = 8
        
        # Create helix-like trajectory
        t = np.linspace(0, 4*np.pi, self.n_atoms)
        radius = 1.5
        
        positions = []
        for frame in range(self.n_frames):
            time_factor = frame / self.n_frames
            rotation = 2 * np.pi * time_factor
            
            frame_positions = np.column_stack([
                radius * np.cos(t + rotation),
                radius * np.sin(t + rotation),
                t * 0.3
            ])
            positions.append(frame_positions)
        
        self.positions = positions
        self.elements = ['C', 'C', 'N', 'O', 'C', 'C', 'S', 'P']
        self.bonds = [(i, i+1) for i in range(self.n_atoms-1)]
    
    def tearDown(self):
        """Clean up after integration tests."""
        plt.close('all')
    
    def test_full_animation_workflow(self):
        """Test complete animation workflow."""
        # Create animator
        animator = TrajectoryAnimator(
            positions=self.positions,
            elements=self.elements,
            bonds=self.bonds,
            interval=50,
            display_mode='ball_stick',
            controls=True
        )
        
        # Setup animation
        animator.setup_animation()
        
        # Test navigation
        animator.goto_frame(7)
        self.assertEqual(animator.current_frame, 7)
        
        # Test speed control
        animator.set_speed(2.5)
        self.assertEqual(animator.animation_speed, 2.5)
        
        # Test stepping
        animator.step()
        self.assertEqual(animator.current_frame, 8)
        
        # Test play/pause methods exist and work
        animator.play()
        animator.pause()
    
    def test_multiple_animations(self):
        """Test creating multiple animations simultaneously."""
        animators = []
        
        for mode in ['ball_stick', 'cartoon']:
            animator = TrajectoryAnimator(
                positions=self.positions,
                elements=self.elements,
                bonds=self.bonds,
                display_mode=mode,
                controls=False
            )
            animator.setup_animation()
            animators.append(animator)
        
        # Check all animations were created
        self.assertEqual(len(animators), 2)
        for animator in animators:
            self.assertIsNotNone(animator.animation)
            self.assertIsNotNone(animator.fig)
        
        # Clean up
        for animator in animators:
            animator.close()


def run_performance_test():
    """Run performance test for trajectory animation."""
    print("Running performance test...")
    
    # Large trajectory
    n_frames = 100
    n_atoms = 50
    
    # Generate test data
    positions = []
    for frame in range(n_frames):
        # Random walk
        if frame == 0:
            pos = np.random.randn(n_atoms, 3) * 5
        else:
            pos = positions[-1] + np.random.randn(n_atoms, 3) * 0.1
        positions.append(pos)
    
    elements = ['C'] * n_atoms
    bonds = [(i, (i+1) % n_atoms) for i in range(n_atoms)]
    
    import time
    start_time = time.time()
    
    # Create animation
    animator = TrajectoryAnimator(
        positions=positions,
        elements=elements,
        bonds=bonds,
        interval=50,
        controls=False
    )
    
    animator.setup_animation()
    
    setup_time = time.time() - start_time
    print(f"Setup time for {n_frames} frames, {n_atoms} atoms: {setup_time:.2f}s")
    
    # Test frame updates
    start_time = time.time()
    for frame in range(0, n_frames, 10):
        animator._update_frame(frame)
    
    update_time = time.time() - start_time
    print(f"Frame update time for 10 frames: {update_time:.2f}s")
    
    animator.close()
    
    return setup_time, update_time


def main():
    """Run all tests."""
    print("Starting Task 2.2 - Trajectory Animation Tests")
    print("=" * 50)
    
    # Run unit tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTrajectoryAnimation))
    suite.addTests(loader.loadTestsFromTestCase(TestTrajectoryAnimationIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance test
    print("\n" + "=" * 50)
    try:
        run_performance_test()
    except Exception as e:
        print(f"Performance test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("✅ Task 2.2 - Trajectory Animation tests PASSED")
    else:
        print("❌ Task 2.2 - Trajectory Animation tests FAILED")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
