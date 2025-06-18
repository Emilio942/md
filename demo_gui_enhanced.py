#!/usr/bin/env python3
"""
Enhanced ProteinMD GUI Demo

This demo showcases the complete GUI functionality including:
- File loading with sample PDB
- Parameter configuration
- Template system
- Simulation control
- Progress monitoring
- Results viewing

Task 8.1: Graphical User Interface - Enhanced Demo
"""

import sys
import os
from pathlib import Path
import time
import threading

# Add proteinMD to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from proteinMD.gui.main_window import ProteinMDGUI, PROTEINMD_AVAILABLE
    import tkinter as tk
    from tkinter import messagebox
    GUI_AVAILABLE = True
except ImportError as e:
    print(f"GUI not available: {e}")
    GUI_AVAILABLE = False

def create_enhanced_demo():
    """Create an enhanced demo with pre-loaded configurations."""
    
    if not GUI_AVAILABLE:
        print("❌ GUI modules not available")
        return None
    
    print("🚀 Starting Enhanced ProteinMD GUI Demo...")
    print("✅ GUI modules loaded successfully")
    
    # Create GUI instance
    app = ProteinMDGUI()
    
    # Check if test protein exists
    test_pdb = Path(__file__).parent / "test_protein.pdb"
    if test_pdb.exists():
        print(f"✅ Test protein found: {test_pdb}")
        # Auto-load the test protein after GUI starts
        def auto_load_protein():
            time.sleep(1)  # Give GUI time to initialize
            try:
                app.load_pdb_file(str(test_pdb))
                app.log_message("Auto-loaded test protein for demo")
                
                # Set up demo output directory
                demo_output = Path(__file__).parent / "demo_output"
                demo_output.mkdir(exist_ok=True)
                app.output_directory = str(demo_output)
                app.output_dir_label.config(text="demo_output")
                app.log_message(f"Demo output directory: {demo_output}")
                
                # Show a helpful message
                demo_message = """
🎯 GUI Demo Ready!

Features to explore:
• File tab: Protein information displayed
• Parameters tab: Adjust simulation settings  
• Simulation tab: Run demo simulation
• Results tab: View outputs after simulation

Try the template system and save/load configurations!
"""
                messagebox.showinfo("Demo Ready", demo_message)
                
            except Exception as e:
                app.log_message(f"Demo setup error: {str(e)}")
        
        # Start auto-load in background
        threading.Timer(0.5, auto_load_protein).start()
    else:
        print(f"⚠️  Test protein not found at {test_pdb}")
        app.log_message("Demo mode: No test protein available")
    
    # Add demo-specific functionality
    enhance_demo_features(app)
    
    return app

def enhance_demo_features(app):
    """Add enhanced demo features to the GUI."""
    
    # Add demo templates
    def add_demo_templates():
        """Add some demo simulation templates."""
        try:
            # Create demo templates directory
            templates_dir = Path.home() / '.proteinmd' / 'templates'
            templates_dir.mkdir(parents=True, exist_ok=True)
            
            # Quick simulation template
            quick_template = {
                'name': 'Quick Demo',
                'description': 'Fast simulation for demonstration purposes',
                'parameters': {
                    'simulation': {
                        'temperature': 300.0,
                        'timestep': 0.002,
                        'n_steps': 1000
                    },
                    'forcefield': {
                        'type': 'amber_ff14sb'
                    },
                    'environment': {
                        'solvent': 'explicit',
                        'box_padding': 1.0
                    },
                    'analysis': {
                        'rmsd': True,
                        'ramachandran': True,
                        'radius_of_gyration': True
                    },
                    'visualization': {
                        'enabled': True,
                        'realtime': False
                    }
                }
            }
            
            # Production simulation template
            production_template = {
                'name': 'Production Run',
                'description': 'Standard production simulation parameters',
                'parameters': {
                    'simulation': {
                        'temperature': 310.0,
                        'timestep': 0.002,
                        'n_steps': 100000
                    },
                    'forcefield': {
                        'type': 'amber_ff14sb'
                    },
                    'environment': {
                        'solvent': 'explicit',
                        'box_padding': 1.5
                    },
                    'analysis': {
                        'rmsd': True,
                        'ramachandran': True,
                        'radius_of_gyration': True
                    },
                    'visualization': {
                        'enabled': True,
                        'realtime': True
                    }
                }
            }
            
            # Save templates
            import json
            with open(templates_dir / 'quick_demo.json', 'w') as f:
                json.dump(quick_template, f, indent=2)
            
            with open(templates_dir / 'production_run.json', 'w') as f:
                json.dump(production_template, f, indent=2)
            
            app.log_message("Demo templates created")
            
            # Refresh template list if possible
            if hasattr(app, 'refresh_template_list'):
                app.refresh_template_list()
            
        except Exception as e:
            app.log_message(f"Demo template setup error: {str(e)}")
    
    # Add demo templates after a short delay
    threading.Timer(1.0, add_demo_templates).start()
    
    # Override demo simulation to be more interactive
    original_demo = app.demo_simulation
    
    def enhanced_demo_simulation():
        """Enhanced demo simulation with more realistic progress."""
        def demo_worker():
            steps = [
                (5, "Initializing molecular system..."),
                (15, "Setting up force field parameters..."),
                (25, "Creating simulation environment..."),
                (35, "Energy minimization..."),
                (45, "Heating system to target temperature..."),
                (55, "Equilibration phase..."),
                (70, "Production simulation running..."),
                (85, "Analyzing trajectory..."),
                (95, "Generating reports..."),
                (100, "Simulation completed successfully!")
            ]
            
            for progress, message in steps:
                if not hasattr(app, '_demo_running') or not app._demo_running:
                    break
                app.update_progress(progress, message)
                time.sleep(0.8)  # More realistic timing
            
            app.on_simulation_finished(True, "Enhanced demo simulation completed")
        
        app._demo_running = True
        app.start_button.config(state=tk.DISABLED)
        app.pause_button.config(state=tk.NORMAL)
        app.stop_button.config(state=tk.NORMAL)
        
        demo_thread = threading.Thread(target=demo_worker)
        demo_thread.daemon = True
        demo_thread.start()
        
        app.log_message("Enhanced demo simulation started")
    
    # Replace demo simulation method
    app.demo_simulation = enhanced_demo_simulation

def main():
    """Main demo function."""
    print("=" * 60)
    print("🧬 ProteinMD Enhanced GUI Demo")
    print("=" * 60)
    
    if not GUI_AVAILABLE:
        print("❌ Cannot run demo - GUI not available")
        print("Please ensure tkinter and ProteinMD modules are installed")
        return 1
    
    # Create and run enhanced demo
    app = create_enhanced_demo()
    
    if app is None:
        print("❌ Failed to create GUI demo")
        return 1
    
    print("✅ GUI demo initialized")
    print("👆 Use the graphical interface to explore features")
    print("💡 Check the simulation log for helpful messages")
    
    try:
        # Start the GUI
        app.run()
        print("✅ GUI demo completed")
        return 0
        
    except KeyboardInterrupt:
        print("\\n⚠️  Demo interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
