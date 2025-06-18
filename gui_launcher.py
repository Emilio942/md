#!/usr/bin/env python3
"""
ProteinMD GUI Launcher

A simple launcher script for the ProteinMD Graphical User Interface.
This provides an easy entry point for users to start the GUI application.

Usage:
    python gui_launcher.py

Or make it executable:
    chmod +x gui_launcher.py
    ./gui_launcher.py
"""

import sys
import os
from pathlib import Path

# Add the proteinMD directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Launch the ProteinMD GUI application."""
    print("🚀 Starting ProteinMD GUI...")
    
    try:
        from proteinMD.gui.main_window import ProteinMDGUI
        
        # Create and run the application
        app = ProteinMDGUI()
        print("✅ GUI initialized successfully")
        app.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Please ensure ProteinMD is properly installed")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
