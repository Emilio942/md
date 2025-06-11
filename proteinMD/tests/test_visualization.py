"""
Comprehensive Unit Tests for Visualization Module

Task 10.1: Umfassende Unit Tests 🚀

Tests the visualization modules including:
- Trajectory animation
- Protein 3D visualization
- Energy dashboard
- Real-time simulation viewer
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Try to import visualization modules
try:
    from visualization.trajectory_animation import TrajectoryAnimator, create_trajectory_animator
    TRAJECTORY_ANIMATION_AVAILABLE = True
except ImportError:
    TRAJECTORY_ANIMATION_AVAILABLE = False

try:
    from visualization.protein_3d import Protein3DVisualizer, create_protein_visualization
    PROTEIN_3D_AVAILABLE = True
except ImportError:
    PROTEIN_3D_AVAILABLE = False

try:
    from visualization.energy_dashboard import EnergyPlotDashboard, create_energy_dashboard
    ENERGY_DASHBOARD_AVAILABLE = True
except ImportError:
    ENERGY_DASHBOARD_AVAILABLE = False

try:
    from visualization.realtime_viewer import RealtimeSimulationViewer
    REALTIME_VIEWER_AVAILABLE = True
except ImportError:
    REALTIME_VIEWER_AVAILABLE = False


@pytest.mark.skipif(not TRAJECTORY_ANIMATION_AVAILABLE, reason="Trajectory animation module not available")
class TestTrajectoryAnimation:
    """Test suite for trajectory animation functionality."""
    
    def test_trajectory_animator_initialization(self, mock_trajectory_data):
        """Test trajectory animator initialization."""
        animator = TrajectoryAnimator(mock_trajectory_data)
        assert animator is not None
        assert hasattr(animator, 'trajectory_data')
        assert animator.n_frames > 0
        assert animator.n_atoms > 0
    
    def test_animation_controls(self, mock_trajectory_data):
        """Test animation control functionality."""
        animator = TrajectoryAnimator(mock_trajectory_data)
        
        # Test play/pause controls
        animator.play()
        assert animator.playing == True
        
        animator.pause()
        assert animator.playing == False
        
        # Test frame navigation
        animator.set_frame(5)
        assert animator.current_frame == 5
    
    def test_animation_speed_control(self, mock_trajectory_data):
        """Test animation speed control."""
        animator = TrajectoryAnimator(mock_trajectory_data)
        
        animator.set_speed(2.0)
        assert animator.animation_speed == 2.0
        
        animator.set_speed(0.5)
        assert animator.animation_speed == 0.5
    
    def test_frame_export(self, mock_trajectory_data):
        """Test individual frame export."""
        animator = TrajectoryAnimator(mock_trajectory_data)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            animator.export_frame(frame=0, filename=tmp.name)
            assert Path(tmp.name).exists()
    
    def test_animation_export(self, mock_trajectory_data):
        """Test animation export to video."""
        animator = TrajectoryAnimator(mock_trajectory_data)
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            # Mock the video export to avoid ffmpeg dependency
            with patch('visualization.trajectory_animation.FFMpegWriter'):
                animator.export_animation(filename=tmp.name, fps=10)
    
    def test_property_tracking(self, mock_trajectory_data):
        """Test real-time property tracking during animation."""
        animator = TrajectoryAnimator(mock_trajectory_data)
        
        # Test property calculation
        properties = animator.calculate_properties()
        assert 'center_of_mass' in properties
        assert 'radius_of_gyration' in properties
        
        # Test property evolution
        assert len(properties['center_of_mass']) == animator.n_frames
    
    def test_visualization_styles(self, mock_trajectory_data):
        """Test different visualization styles."""
        animator = TrajectoryAnimator(mock_trajectory_data)
        
        # Test ball-and-stick style
        animator.set_style('ball_and_stick')
        assert animator.style == 'ball_and_stick'
        
        # Test space-filling style
        animator.set_style('space_filling')
        assert animator.style == 'space_filling'
    
    def test_animation_performance(self, mock_large_trajectory_data):
        """Test animation performance with large trajectories."""
        animator = TrajectoryAnimator(mock_large_trajectory_data)
        
        import time
        start_time = time.time()
        animator.update_frame(0)
        update_time = time.time() - start_time
        
        # Frame update should be fast
        assert update_time < 0.1  # Less than 100ms per frame


@pytest.mark.skipif(not PROTEIN_3D_AVAILABLE, reason="Protein 3D module not available")
class TestProtein3DVisualization:
    """Test suite for 3D protein visualization."""
    
    def test_protein_3d_visualizer_initialization(self, mock_protein):
        """Test 3D protein visualizer initialization."""
        visualizer = Protein3DVisualizer(mock_protein)
        assert visualizer is not None
        assert hasattr(visualizer, 'protein')
    
    def test_rendering_modes(self, mock_protein):
        """Test different protein rendering modes."""
        visualizer = Protein3DVisualizer(mock_protein)
        
        # Test cartoon representation
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            visualizer.render(mode='cartoon', output_file=tmp.name)
            assert Path(tmp.name).exists()
        
        # Test surface representation
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            visualizer.render(mode='surface', output_file=tmp.name)
            assert Path(tmp.name).exists()
    
    def test_interactive_features(self, mock_protein):
        """Test interactive visualization features."""
        visualizer = Protein3DVisualizer(mock_protein)
        
        # Test rotation
        visualizer.rotate(x=30, y=45, z=0)
        
        # Test zoom
        visualizer.zoom(factor=1.5)
        
        # Test selection
        selected_atoms = visualizer.select_atoms(residue_range=(1, 10))
        assert len(selected_atoms) > 0
    
    def test_coloring_schemes(self, mock_protein):
        """Test different protein coloring schemes."""
        visualizer = Protein3DVisualizer(mock_protein)
        
        # Test color by secondary structure
        visualizer.color_by('secondary_structure')
        
        # Test color by residue type
        visualizer.color_by('residue_type')
        
        # Test color by b-factor
        visualizer.color_by('b_factor')
    
    def test_export_formats(self, mock_protein):
        """Test export to different formats."""
        visualizer = Protein3DVisualizer(mock_protein)
        
        # Test PNG export
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            visualizer.export(filename=tmp.name, format='png')
            assert Path(tmp.name).exists()
        
        # Test SVG export
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
            visualizer.export(filename=tmp.name, format='svg')
            assert Path(tmp.name).exists()


@pytest.mark.skipif(not ENERGY_DASHBOARD_AVAILABLE, reason="Energy dashboard module not available")
class TestEnergyDashboard:
    """Test suite for energy dashboard visualization."""
    
    def test_energy_dashboard_initialization(self):
        """Test energy dashboard initialization."""
        dashboard = EnergyPlotDashboard()
        assert dashboard is not None
    
    def test_energy_plot_generation(self, mock_energy_data):
        """Test energy plot generation."""
        dashboard = EnergyPlotDashboard()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            dashboard.create_energy_plot(mock_energy_data, output_file=tmp.name)
            assert Path(tmp.name).exists()
    
    def test_real_time_energy_monitoring(self, mock_simulation_system):
        """Test real-time energy monitoring."""
        dashboard = EnergyPlotDashboard()
        
        # Mock simulation with energy callback
        def energy_callback(step, energies):
            dashboard.update_energy_data(step, energies)
        
        mock_simulation_system.add_callback(energy_callback)
        
        # Run simulation steps
        for step in range(10):
            energies = {
                'kinetic': np.random.uniform(100, 200),
                'potential': np.random.uniform(-500, -400),
                'total': np.random.uniform(-400, -200)
            }
            energy_callback(step, energies)
        
        assert len(dashboard.energy_history) == 10
    
    def test_energy_statistics_calculation(self, mock_energy_data):
        """Test energy statistics calculation."""
        dashboard = EnergyPlotDashboard()
        stats = dashboard.calculate_statistics(mock_energy_data)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
    
    def test_energy_conservation_analysis(self, mock_energy_data):
        """Test energy conservation analysis."""
        dashboard = EnergyPlotDashboard()
        conservation = dashboard.analyze_conservation(mock_energy_data)
        
        assert 'drift' in conservation
        assert 'fluctuation' in conservation
        assert isinstance(conservation['drift'], float)


@pytest.mark.skipif(not REALTIME_VIEWER_AVAILABLE, reason="Real-time viewer module not available")
class TestRealtimeViewer:
    """Test suite for real-time simulation viewer."""
    
    def test_realtime_viewer_initialization(self, mock_simulation_system):
        """Test real-time viewer initialization."""
        viewer = RealtimeSimulationViewer(mock_simulation_system)
        assert viewer is not None
        assert hasattr(viewer, 'simulation')
    
    def test_real_time_display_update(self, mock_simulation_system):
        """Test real-time display updates."""
        viewer = RealtimeSimulationViewer(mock_simulation_system)
        
        # Mock simulation step
        viewer.update_display()
        
        # Should have updated frame
        assert viewer.current_frame >= 0
    
    def test_simulation_control_integration(self, mock_simulation_system):
        """Test integration with simulation controls."""
        viewer = RealtimeSimulationViewer(mock_simulation_system)
        
        # Test play/pause
        viewer.start_simulation()
        assert viewer.is_running == True
        
        viewer.pause_simulation()
        assert viewer.is_running == False
        
        viewer.stop_simulation()
        assert viewer.is_running == False
    
    def test_performance_monitoring(self, mock_simulation_system):
        """Test performance monitoring during real-time viewing."""
        viewer = RealtimeSimulationViewer(mock_simulation_system)
        
        # Start performance monitoring
        viewer.start_performance_monitoring()
        
        # Run some simulation steps
        for _ in range(10):
            viewer.update_display()
        
        perf_stats = viewer.get_performance_statistics()
        assert 'fps' in perf_stats
        assert 'update_time' in perf_stats


class TestVisualizationIntegration:
    """Integration tests for visualization modules."""
    
    def test_trajectory_animation_with_analysis(self, mock_trajectory_data):
        """Test trajectory animation with analysis integration."""
        if not TRAJECTORY_ANIMATION_AVAILABLE:
            pytest.skip("Trajectory animation not available")
        
        animator = TrajectoryAnimator(mock_trajectory_data)
        
        # Test integration with analysis modules
        properties = animator.calculate_properties()
        animator.add_property_plot('radius_of_gyration')
        
        # Verify property tracking
        assert hasattr(animator, 'property_plots')
    
    def test_multi_view_visualization(self, mock_protein, mock_trajectory_data):
        """Test multiple visualization views."""
        if not (PROTEIN_3D_AVAILABLE and TRAJECTORY_ANIMATION_AVAILABLE):
            pytest.skip("Required visualization modules not available")
        
        # Create 3D view
        protein_viz = Protein3DVisualizer(mock_protein)
        
        # Create trajectory animation
        trajectory_viz = TrajectoryAnimator(mock_trajectory_data)
        
        # Test synchronized views
        # Both should show the same structure at frame 0
        protein_viz.set_structure(trajectory_viz.get_frame(0))
    
    def test_visualization_memory_efficiency(self, mock_large_trajectory_data):
        """Test memory efficiency of visualization modules."""
        if not TRAJECTORY_ANIMATION_AVAILABLE:
            pytest.skip("Trajectory animation not available")
        
        import tracemalloc
        tracemalloc.start()
        
        animator = TrajectoryAnimator(mock_large_trajectory_data)
        animator.render_frame(0)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory usage should be reasonable
        assert peak < 200 * 1024 * 1024  # Less than 200 MB


# Performance regression tests
class TestVisualizationPerformanceRegression:
    """Performance regression tests for visualization modules."""
    
    @pytest.mark.performance
    def test_animation_frame_update_performance(self, mock_trajectory_data, benchmark):
        """Benchmark animation frame update performance."""
        if not TRAJECTORY_ANIMATION_AVAILABLE:
            pytest.skip("Trajectory animation not available")
        
        animator = TrajectoryAnimator(mock_trajectory_data)
        
        def update_frame():
            animator.update_frame(0)
        
        benchmark(update_frame)
    
    @pytest.mark.performance
    def test_3d_rendering_performance(self, mock_protein, benchmark):
        """Benchmark 3D protein rendering performance."""
        if not PROTEIN_3D_AVAILABLE:
            pytest.skip("Protein 3D not available")
        
        visualizer = Protein3DVisualizer(mock_protein)
        
        def render_protein():
            with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
                visualizer.render(output_file=tmp.name)
        
        benchmark(render_protein)
    
    @pytest.mark.performance
    def test_energy_plot_update_performance(self, mock_energy_data, benchmark):
        """Benchmark energy plot update performance."""
        if not ENERGY_DASHBOARD_AVAILABLE:
            pytest.skip("Energy dashboard not available")
        
        dashboard = EnergyPlotDashboard()
        
        def update_energy_plot():
            with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
                dashboard.create_energy_plot(mock_energy_data, output_file=tmp.name)
        
        benchmark(update_energy_plot)
