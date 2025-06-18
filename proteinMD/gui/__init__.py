"""
ProteinMD Graphical User Interface Package

Task 8.1: Graphical User Interface 🛠

This package provides a comprehensive graphical user interface for ProteinMD,
enabling users to perform molecular dynamics simulations without command-line usage.

Features:
- Intuitive drag & drop PDB file loading
- Interactive simulation parameter configuration  
- Real-time simulation control (Start/Stop/Pause)
- Live progress monitoring with visual feedback
- Integrated 3D protein visualization
- Template-based simulation workflows
- Results management and export
- Configuration save/load functionality

Components:
- MainWindow: Primary application interface
- ParameterPanels: Interactive parameter configuration
- SimulationControls: Start/Stop/Pause functionality
- ProgressMonitoring: Real-time progress tracking
- ResultsViewer: Analysis and visualization of results

Usage:
    from proteinMD.gui import ProteinMDGUI
    
    app = ProteinMDGUI()
    app.run()

Or run directly:
    python -m proteinMD.gui
"""

from .main_window import ProteinMDGUI

__all__ = ['ProteinMDGUI']

# Package version
__version__ = '1.0.0'
