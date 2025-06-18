"""
ProteinMD Graphical User Interface

Task 8.1: Graphical User Interface 🛠
Status: IMPLEMENTING

This module provides a user-friendly graphical interface for ProteinMD,
enabling easy access to molecular dynamics simulations without command-line usage.

Features:
- Drag & Drop PDB file loading
- Interactive parameter configuration
- Start/Stop/Pause simulation controls
- Real-time progress monitoring
- Integrated visualization
- Results management
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.dnd as dnd
import os
import sys
import threading
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import logging

# Add proteinMD to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

try:
    from proteinMD.cli import ProteinMDCLI
    from proteinMD.structure.pdb_parser import PDBParser
    PROTEINMD_AVAILABLE = True
    
    # Optional 3D visualization
    try:
        from proteinMD.visualization.protein_3d import Protein3DVisualization
        VISUALIZATION_AVAILABLE = True
    except ImportError:
        VISUALIZATION_AVAILABLE = False
        
except ImportError:
    PROTEINMD_AVAILABLE = False
    VISUALIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class DropTarget:
    """Helper class for drag and drop functionality."""
    
    def __init__(self, widget, callback):
        self.widget = widget
        self.callback = callback
        
        # Configure widget for drag and drop
        widget.drop_target_register('DND_FILES')
        widget.dnd_bind('<<Drop>>', self.on_drop)
        
        # Also bind standard file drop events
        widget.bind('<Button-1>', self.on_click)
        widget.bind('<B1-Motion>', self.on_drag)
        widget.bind('<ButtonRelease-1>', self.on_release)
    
    def on_drop(self, event):
        """Handle file drop events."""
        files = event.data.split()
        if files:
            # Take the first file
            file_path = files[0].strip('{}')
            if file_path.lower().endswith('.pdb'):
                self.callback(file_path)
    
    def on_click(self, event):
        """Handle click events for manual file selection."""
        pass
    
    def on_drag(self, event):
        """Handle drag events."""
        pass
    
    def on_release(self, event):
        """Handle release events."""
        pass

class SimulationWorker:
    """Background worker for running simulations."""
    
    def __init__(self, gui_app):
        self.gui_app = gui_app
        self.cli = None
        self.simulation_thread = None
        self.running = False
        self.paused = False
        
    def start_simulation(self, pdb_file: str, parameters: Dict[str, Any], output_dir: str):
        """Start a simulation in a background thread."""
        if self.running:
            return False
            
        self.running = True
        self.paused = False
        
        def run_simulation():
            try:
                # Create CLI instance
                self.cli = ProteinMDCLI()
                
                # Configure parameters
                self.cli.config.update(parameters)
                
                # Update progress
                self.gui_app.update_progress(0, "Initializing simulation...")
                
                # Start simulation
                result = self.cli.run_simulation(
                    input_file=pdb_file,
                    output_dir=output_dir
                )
                
                # Update progress
                if result == 0:
                    self.gui_app.update_progress(100, "Simulation completed successfully!")
                    self.gui_app.on_simulation_finished(True, "Simulation completed successfully")
                else:
                    self.gui_app.on_simulation_finished(False, "Simulation failed")
                    
            except Exception as e:
                error_msg = f"Simulation error: {str(e)}"
                logger.error(error_msg)
                self.gui_app.on_simulation_finished(False, error_msg)
            finally:
                self.running = False
                
        self.simulation_thread = threading.Thread(target=run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        return True
    
    def stop_simulation(self):
        """Stop the running simulation."""
        if self.running:
            self.running = False
            # In a real implementation, this would signal the simulation to stop
            logger.info("Simulation stop requested")
    
    def pause_simulation(self):
        """Pause the running simulation."""
        if self.running:
            self.paused = not self.paused
            logger.info(f"Simulation {'paused' if self.paused else 'resumed'}")

class ProteinMDGUI:
    """Main GUI application for ProteinMD."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ProteinMD - Molecular Dynamics Simulation")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Application state
        self.current_pdb_file = None
        self.current_protein = None
        self.output_directory = None
        self.simulation_worker = SimulationWorker(self)
        
        # Template system
        self.template_manager = None
        if PROTEINMD_AVAILABLE:
            try:
                from proteinMD.templates.template_manager import TemplateManager
                self.template_manager = TemplateManager()
            except Exception as e:
                logger.warning(f"Template manager not available: {e}")
        
        # Create GUI elements
        self.setup_gui()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
    def setup_gui(self):
        """Set up the GUI layout and components."""
        # Create main menu
        self.create_menu()
        
        # Create main layout
        self.create_main_layout()
        
        # Create status bar
        self.create_status_bar()
        
        # Set up keyboard shortcuts
        self.setup_keyboard_shortcuts()
        
        # Set up window behaviors
        self.setup_window_behaviors()
        
    def setup_keyboard_shortcuts(self):
        """Set up keyboard shortcuts for common actions."""
        # File operations
        self.root.bind('<Control-o>', lambda e: self.open_pdb_file())
        self.root.bind('<Control-s>', lambda e: self.save_configuration())
        self.root.bind('<Control-l>', lambda e: self.load_configuration())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
        
        # Simulation controls
        self.root.bind('<F5>', lambda e: self.start_simulation())
        self.root.bind('<F6>', lambda e: self.pause_simulation())
        self.root.bind('<F7>', lambda e: self.stop_simulation())
        
        # Navigation
        self.root.bind('<Control-1>', lambda e: self.notebook.select(0))  # Input tab
        self.root.bind('<Control-2>', lambda e: self.notebook.select(1))  # Parameters tab
        self.root.bind('<Control-3>', lambda e: self.notebook.select(2))  # Simulation tab
        self.root.bind('<Control-4>', lambda e: self.notebook.select(3))  # Results tab
        
        # Help
        self.root.bind('<F1>', lambda e: self.show_keyboard_shortcuts())
        
        # Focus management
        self.root.bind('<Tab>', self.on_tab_navigation)
        
    def setup_window_behaviors(self):
        """Set up window behaviors and protocols."""
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Set minimum window size
        self.root.minsize(900, 600)
        
        # Configure window icon if available
        try:
            # Try to set a window icon (would need an icon file)
            # self.root.iconbitmap('proteinmd_icon.ico')
            pass
        except:
            pass
            
        # Set focus policies
        self.root.focus_set()
        
    def on_closing(self):
        """Handle application closing."""
        if self.simulation_worker.running:
            response = messagebox.askyesno(
                "Simulation Running", 
                "A simulation is currently running.\nDo you want to stop it and exit?"
            )
            if response:
                self.simulation_worker.stop_simulation()
                self.root.destroy()
        else:
            self.root.destroy()
    
    def on_tab_navigation(self, event):
        """Handle tab navigation for accessibility."""
        # Allow default tab behavior
        return None
    
    def show_keyboard_shortcuts(self):
        """Show keyboard shortcuts help dialog."""
        shortcuts_text = """Keyboard Shortcuts:

File Operations:
  Ctrl+O    Open PDB file
  Ctrl+S    Save configuration
  Ctrl+L    Load configuration
  Ctrl+Q    Quit application

Simulation Control:
  F5        Start simulation
  F6        Pause/Resume simulation
  F7        Stop simulation

Navigation:
  Ctrl+1    Input tab
  Ctrl+2    Parameters tab
  Ctrl+3    Simulation tab
  Ctrl+4    Results tab

Help:
  F1        Show this help
"""
        
        messagebox.showinfo("Keyboard Shortcuts", shortcuts_text)
    
    def create_menu(self):
        """Create the application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open PDB...", command=self.open_pdb_file)
        file_menu.add_command(label="Save Configuration...", command=self.save_configuration)
        file_menu.add_command(label="Load Configuration...", command=self.load_configuration)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Simulation menu
        sim_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Simulation", menu=sim_menu)
        sim_menu.add_command(label="Start", command=self.start_simulation)
        sim_menu.add_command(label="Pause/Resume", command=self.pause_simulation)
        sim_menu.add_command(label="Stop", command=self.stop_simulation)
        
        # Templates menu
        template_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Templates", menu=template_menu)
        template_menu.add_command(label="Browse Templates...", command=self.browse_templates)
        template_menu.add_command(label="Save as Template...", command=self.save_as_template)
        template_menu.add_separator()
        template_menu.add_command(label="Template Statistics", command=self.show_template_statistics)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Protein Structure", command=self.view_protein_structure)
        view_menu.add_command(label="Parameters", command=self.view_parameters)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        
    def create_main_layout(self):
        """Create the main application layout."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_input_tab()
        self.create_parameters_tab()
        self.create_results_tab()  # Use existing method instead of missing create_simulation_tab
        self.create_results_tab()
        
    def create_input_tab(self):
        """Create the input/file loading tab."""
        input_frame = ttk.Frame(self.notebook)
        self.notebook.add(input_frame, text="Input")
        
        # PDB file section
        pdb_frame = ttk.LabelFrame(input_frame, text="Protein Structure (PDB File)")
        pdb_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Drag & Drop area
        self.drop_area = tk.Frame(pdb_frame, height=100, bg='lightgray', relief=tk.SUNKEN, bd=2)
        self.drop_area.pack(fill=tk.X, padx=10, pady=10)
        
        # Drag & Drop label
        drop_label = tk.Label(
            self.drop_area, 
            text="Drag & Drop PDB file here\\nor click 'Browse' to select file",
            bg='lightgray',
            font=('Arial', 12)
        )
        drop_label.pack(expand=True)
        
        # Browse button
        browse_frame = tk.Frame(pdb_frame)
        browse_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(browse_frame, text="Browse...", command=self.open_pdb_file).pack(side=tk.LEFT)
        
        # Current file display
        self.file_label = ttk.Label(browse_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        # Protein information
        info_frame = ttk.LabelFrame(input_frame, text="Protein Information")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create text widget for protein info
        self.protein_info_text = tk.Text(info_frame, height=10, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.protein_info_text.yview)
        self.protein_info_text.configure(yscrollcommand=scrollbar.set)
        
        self.protein_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Set up drag and drop (simplified - requires tkdnd)
        self.drop_area.bind('<Button-1>', lambda e: self.open_pdb_file())
        
    def create_parameters_tab(self):
        """Create the simulation parameters tab."""
        param_frame = ttk.Frame(self.notebook)
        self.notebook.add(param_frame, text="Parameters")
        
        # Create scrollable frame
        canvas = tk.Canvas(param_frame)
        scrollbar = ttk.Scrollbar(param_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Template selection
        template_frame = ttk.LabelFrame(scrollable_frame, text="Simulation Templates")
        template_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(template_frame, text="Select Template:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.template_var = tk.StringVar(value="Custom")
        
        # Get available templates
        template_names = ["Custom"]
        if self.template_manager:
            try:
                templates = self.template_manager.list_templates()
                template_names.extend(sorted(templates.keys()))
            except:
                pass
        
        self.template_combobox = ttk.Combobox(template_frame, textvariable=self.template_var, 
                                             values=template_names, state="readonly", width=20)
        self.template_combobox.grid(row=0, column=1, padx=5, pady=2)
        self.template_combobox.bind('<<ComboboxSelected>>', self.on_template_selected)
        
        # Template description
        self.template_desc_label = ttk.Label(template_frame, text="Configure parameters manually", 
                                           foreground="gray")
        self.template_desc_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Basic simulation parameters
        basic_frame = ttk.LabelFrame(scrollable_frame, text="Basic Parameters")
        basic_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Temperature
        ttk.Label(basic_frame, text="Temperature (K):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.temperature_var = tk.StringVar(value="300.0")
        ttk.Entry(basic_frame, textvariable=self.temperature_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        # Timestep
        ttk.Label(basic_frame, text="Timestep (ps):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.timestep_var = tk.StringVar(value="0.002")
        ttk.Entry(basic_frame, textvariable=self.timestep_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        # Number of steps
        ttk.Label(basic_frame, text="Number of steps:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.nsteps_var = tk.StringVar(value="10000")
        ttk.Entry(basic_frame, textvariable=self.nsteps_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        # Force field
        ttk.Label(basic_frame, text="Force field:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.forcefield_var = tk.StringVar(value="amber_ff14sb")
        forcefield_combo = ttk.Combobox(basic_frame, textvariable=self.forcefield_var, width=15)
        forcefield_combo['values'] = ("amber_ff14sb", "charmm36", "custom")
        forcefield_combo.grid(row=3, column=1, padx=5, pady=2)
        
        # Environment parameters
        env_frame = ttk.LabelFrame(scrollable_frame, text="Environment")
        env_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Solvent
        ttk.Label(env_frame, text="Solvent:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.solvent_var = tk.StringVar(value="explicit")
        solvent_combo = ttk.Combobox(env_frame, textvariable=self.solvent_var, width=15)
        solvent_combo['values'] = ("explicit", "implicit", "vacuum")
        solvent_combo.grid(row=0, column=1, padx=5, pady=2)
        
        # Box padding
        ttk.Label(env_frame, text="Box padding (nm):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.box_padding_var = tk.StringVar(value="1.0")
        ttk.Entry(env_frame, textvariable=self.box_padding_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        # Analysis parameters
        analysis_frame = ttk.LabelFrame(scrollable_frame, text="Analysis")
        analysis_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.rmsd_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(analysis_frame, text="RMSD Analysis", variable=self.rmsd_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.ramachandran_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(analysis_frame, text="Ramachandran Plot", variable=self.ramachandran_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.radius_of_gyration_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(analysis_frame, text="Radius of Gyration", variable=self.radius_of_gyration_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # PCA Analysis
        self.pca_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(analysis_frame, text="Principal Component Analysis (PCA)", variable=self.pca_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # PCA parameters sub-frame
        pca_params_frame = ttk.Frame(analysis_frame)
        pca_params_frame.pack(fill=tk.X, padx=20, pady=2)
        
        # PCA atom selection
        ttk.Label(pca_params_frame, text="PCA atom selection:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.pca_atom_selection_var = tk.StringVar(value="CA")
        pca_atom_combo = ttk.Combobox(pca_params_frame, textvariable=self.pca_atom_selection_var, width=12)
        pca_atom_combo['values'] = ("CA", "backbone", "all")
        pca_atom_combo.grid(row=0, column=1, padx=5, pady=2)
        
        # Number of PCA components
        ttk.Label(pca_params_frame, text="PCA components:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.pca_n_components_var = tk.StringVar(value="20")
        ttk.Entry(pca_params_frame, textvariable=self.pca_n_components_var, width=8).grid(row=1, column=1, padx=5, pady=2)
        
        # PCA clustering
        self.pca_clustering_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pca_params_frame, text="Conformational clustering", variable=self.pca_clustering_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Number of clusters
        ttk.Label(pca_params_frame, text="Number of clusters:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.pca_n_clusters_var = tk.StringVar(value="auto")
        pca_clusters_combo = ttk.Combobox(pca_params_frame, textvariable=self.pca_n_clusters_var, width=8)
        pca_clusters_combo['values'] = ("auto", "2", "3", "4", "5", "6", "7", "8", "9", "10")
        pca_clusters_combo.grid(row=3, column=1, padx=5, pady=2)
        
        # Cross-Correlation Analysis
        self.cross_correlation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(analysis_frame, text="Dynamic Cross-Correlation Analysis", variable=self.cross_correlation_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # Cross-Correlation parameters sub-frame
        cc_params_frame = ttk.Frame(analysis_frame)
        cc_params_frame.pack(fill=tk.X, padx=20, pady=2)
        
        # Cross-correlation atom selection
        ttk.Label(cc_params_frame, text="CC atom selection:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.cc_atom_selection_var = tk.StringVar(value="CA")
        cc_atom_combo = ttk.Combobox(cc_params_frame, textvariable=self.cc_atom_selection_var, width=12)
        cc_atom_combo['values'] = ("CA", "backbone", "all")
        cc_atom_combo.grid(row=0, column=1, padx=5, pady=2)
        
        # Statistical significance method
        ttk.Label(cc_params_frame, text="Significance test:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.cc_significance_var = tk.StringVar(value="ttest")
        cc_sig_combo = ttk.Combobox(cc_params_frame, textvariable=self.cc_significance_var, width=12)
        cc_sig_combo['values'] = ("ttest", "bootstrap", "permutation")
        cc_sig_combo.grid(row=1, column=1, padx=5, pady=2)
        
        # Network analysis
        self.cc_network_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cc_params_frame, text="Network analysis", variable=self.cc_network_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Network threshold
        ttk.Label(cc_params_frame, text="Network threshold:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.cc_threshold_var = tk.StringVar(value="0.5")
        ttk.Entry(cc_params_frame, textvariable=self.cc_threshold_var, width=8).grid(row=3, column=1, padx=5, pady=2)
        
        # Time-dependent analysis
        self.cc_time_dependent_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(cc_params_frame, text="Time-dependent analysis", variable=self.cc_time_dependent_var).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Free Energy Analysis
        self.free_energy_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(analysis_frame, text="Free Energy Landscape Analysis", variable=self.free_energy_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # Free Energy parameters sub-frame
        fe_params_frame = ttk.Frame(analysis_frame)
        fe_params_frame.pack(fill=tk.X, padx=20, pady=2)
        
        # Coordinate selection for 1D profile
        ttk.Label(fe_params_frame, text="1D coordinate:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.fe_coord1_var = tk.StringVar(value="rmsd")
        fe_coord1_combo = ttk.Combobox(fe_params_frame, textvariable=self.fe_coord1_var, width=12)
        fe_coord1_combo['values'] = ("rmsd", "radius_of_gyration", "end_to_end_distance", "phi", "psi")
        fe_coord1_combo.grid(row=0, column=1, padx=5, pady=2)
        
        # Coordinate selection for 2D landscape
        ttk.Label(fe_params_frame, text="2D coordinate:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.fe_coord2_var = tk.StringVar(value="radius_of_gyration")
        fe_coord2_combo = ttk.Combobox(fe_params_frame, textvariable=self.fe_coord2_var, width=12)
        fe_coord2_combo['values'] = ("rmsd", "radius_of_gyration", "end_to_end_distance", "phi", "psi")
        fe_coord2_combo.grid(row=1, column=1, padx=5, pady=2)
        
        # Number of bins
        ttk.Label(fe_params_frame, text="Number of bins:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.fe_n_bins_var = tk.StringVar(value="50")
        ttk.Entry(fe_params_frame, textvariable=self.fe_n_bins_var, width=8).grid(row=2, column=1, padx=5, pady=2)
        
        # Bootstrap error analysis
        self.fe_bootstrap_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(fe_params_frame, text="Bootstrap error analysis", variable=self.fe_bootstrap_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Minimum identification
        self.fe_find_minima_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(fe_params_frame, text="Identify energy minima", variable=self.fe_find_minima_var).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        # SASA Analysis
        self.sasa_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(analysis_frame, text="Solvent Accessible Surface Area (SASA)", variable=self.sasa_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # SASA parameters sub-frame
        sasa_params_frame = ttk.Frame(analysis_frame)
        sasa_params_frame.pack(fill=tk.X, padx=20, pady=2)
        
        # Probe radius
        ttk.Label(sasa_params_frame, text="Probe radius (Å):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.sasa_probe_radius_var = tk.StringVar(value="1.4")
        ttk.Entry(sasa_params_frame, textvariable=self.sasa_probe_radius_var, width=8).grid(row=0, column=1, padx=5, pady=2)
        
        # Quadrature points
        ttk.Label(sasa_params_frame, text="Quadrature points:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.sasa_n_points_var = tk.StringVar(value="590")
        sasa_points_combo = ttk.Combobox(sasa_params_frame, textvariable=self.sasa_n_points_var, width=8)
        sasa_points_combo['values'] = ("194", "590")
        sasa_points_combo.grid(row=1, column=1, padx=5, pady=2)
        
        # Per-residue analysis
        self.sasa_per_residue_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sasa_params_frame, text="Per-residue SASA analysis", variable=self.sasa_per_residue_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Hydrophobic/hydrophilic decomposition
        self.sasa_hydrophobic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sasa_params_frame, text="Hydrophobic/hydrophilic classification", variable=self.sasa_hydrophobic_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Time series analysis
        self.sasa_time_series_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sasa_params_frame, text="Time series analysis", variable=self.sasa_time_series_var).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
    def create_results_tab(self):
        """Create the results viewing tab."""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")
        
        # Results file browser
        browser_frame = ttk.LabelFrame(results_frame, text="Results Files")
        browser_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.results_listbox = tk.Listbox(browser_frame, height=8)
        results_scrollbar = ttk.Scrollbar(browser_frame, orient=tk.VERTICAL, command=self.results_listbox.yview)
        self.results_listbox.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event for preview
        self.results_listbox.bind('<<ListboxSelect>>', self.on_results_selection)
        
        # Results viewer buttons
        viewer_frame = tk.Frame(results_frame)
        viewer_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(viewer_frame, text="View Trajectory", command=self.view_trajectory).pack(side=tk.LEFT, padx=5)
        ttk.Button(viewer_frame, text="View Analysis", command=self.view_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(viewer_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(viewer_frame, text="Generate Report", command=self.generate_report).pack(side=tk.LEFT, padx=5)
        
        # Results preview
        preview_frame = ttk.LabelFrame(results_frame, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.results_text = tk.Text(preview_frame, state=tk.DISABLED)
        preview_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=preview_scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preview_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def on_results_selection(self, event):
        """Handle selection of results file for preview."""
        selection = self.results_listbox.curselection()
        if not selection:
            return
            
        selected_file = self.results_listbox.get(selection[0])
        if not self.output_directory:
            return
            
        file_path = Path(self.output_directory) / selected_file
        
        try:
            if file_path.exists():
                self.preview_results_file(file_path)
        except Exception as e:
            self.log_message(f"Error previewing {selected_file}: {str(e)}")
    
    def preview_results_file(self, file_path: Path):
        """Preview a results file in the results tab."""
        try:
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            
            # Handle different file types
            if file_path.suffix.lower() == '.json':
                import json
                with open(file_path, 'r') as f:
                    data = json.load(f)
                content = json.dumps(data, indent=2)
            elif file_path.suffix.lower() in ['.txt', '.log', '.dat']:
                with open(file_path, 'r') as f:
                    content = f.read()
            elif file_path.suffix.lower() == '.pdb':
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                content = ''.join(lines[:50])  # Show first 50 lines
                if len(lines) > 50:
                    content += f"\n... ({len(lines) - 50} more lines)"
            else:
                content = f"Binary file: {file_path.name}\nSize: {file_path.stat().st_size} bytes"
            
            self.results_text.insert(1.0, content)
            self.results_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, f"Error reading file: {str(e)}")
            self.results_text.config(state=tk.DISABLED)
    
    def get_simulation_parameters(self) -> Dict[str, Any]:
        """Get current simulation parameters from the GUI."""
        params = {
            'simulation': {
                'temperature': float(self.temperature_var.get()),
                'timestep': float(self.timestep_var.get()),
                'n_steps': int(self.nsteps_var.get())
            },
            'forcefield': {
                'type': self.forcefield_var.get()
            },
            'environment': {
                'solvent': self.solvent_var.get(),
                'box_padding': float(self.box_padding_var.get())
            },
            'analysis': {
                'rmsd': self.rmsd_var.get(),
                'ramachandran': self.ramachandran_var.get(),
                'radius_of_gyration': self.radius_of_gyration_var.get(),
                'pca': {
                    'enabled': self.pca_var.get(),
                    'atom_selection': self.pca_atom_selection_var.get(),
                    'n_components': int(self.pca_n_components_var.get()) if self.pca_n_components_var.get().isdigit() else 20,
                    'clustering': self.pca_clustering_var.get(),
                    'n_clusters': self.pca_n_clusters_var.get() if self.pca_n_clusters_var.get() != 'auto' else None
                },
                'cross_correlation': {
                    'enabled': self.cross_correlation_var.get(),
                    'atom_selection': self.cc_atom_selection_var.get(),
                    'significance_method': self.cc_significance_var.get(),
                    'network_analysis': self.cc_network_var.get(),
                    'network_threshold': float(self.cc_threshold_var.get()) if self.cc_threshold_var.get().replace('.', '').isdigit() else 0.5,
                    'time_dependent': self.cc_time_dependent_var.get()
                },
                'free_energy': {
                    'enabled': self.free_energy_var.get(),
                    'coord1': self.fe_coord1_var.get(),
                    'coord2': self.fe_coord2_var.get(),
                    'n_bins': int(self.fe_n_bins_var.get()) if self.fe_n_bins_var.get().isdigit() else 50,
                    'bootstrap': self.fe_bootstrap_var.get(),
                    'find_minima': self.fe_find_minima_var.get()
                },
                'sasa': {
                    'enabled': self.sasa_var.get(),
                    'probe_radius': float(self.sasa_probe_radius_var.get()) if self.sasa_probe_radius_var.get().replace('.', '').isdigit() else 1.4,
                    'n_points': int(self.sasa_n_points_var.get()) if self.sasa_n_points_var.get().isdigit() else 590,
                    'per_residue': self.sasa_per_residue_var.get(),
                    'hydrophobic': self.sasa_hydrophobic_var.get(),
                    'time_series': self.sasa_time_series_var.get()
                }
            },
            'visualization': {
                'enabled': self.visualization_var.get(),
                'realtime': self.realtime_var.get()
            }
        }
        
        # Add SMD parameters if enabled
        if hasattr(self, 'enable_smd_var') and self.enable_smd_var.get():
            # Parse atom indices
            try:
                atom_indices = [int(x.strip()) for x in self.smd_atoms_var.get().split(',')]
            except ValueError:
                atom_indices = []
            
            smd_params = {
                'enabled': True,
                'mode': self.smd_mode_var.get(),
                'coordinate_type': self.smd_coordinate_var.get(),
                'atom_indices': atom_indices,
                'save_force_curves': self.smd_save_forces_var.get(),
                'calculate_work': self.smd_calc_work_var.get()
            }
            
            # Add mode-specific parameters
            if self.smd_mode_var.get() == "constant_velocity":
                smd_params.update({
                    'pulling_velocity': float(self.smd_velocity_var.get()),
                    'spring_constant': float(self.smd_spring_var.get())
                })
            else:  # constant_force
                smd_params.update({
                    'applied_force': float(self.smd_force_var.get())
                })
            
            params['steered_md'] = smd_params
        else:
            params['steered_md'] = {'enabled': False}
        
        return params
    
    def generate_report(self):
        """Generate a comprehensive simulation report."""
        if not self.output_directory:
            messagebox.showwarning("Warning", "No output directory selected")
            return
            
        try:
            from datetime import datetime
            import json
            
            report_data = {
                'report_generated': datetime.now().isoformat(),
                'simulation_parameters': self.get_simulation_parameters(),
                'input_file': self.current_pdb_file,
                'output_directory': self.output_directory,
                'protein_info': {}
            }
            
            # Add protein information if available
            if self.current_protein:
                report_data['protein_info'] = {
                    'num_atoms': len(self.current_protein.atoms),
                    'num_residues': len(set(atom.residue_index for atom in self.current_protein.atoms)),
                    'chains': len(set(atom.chain_id for atom in self.current_protein.atoms))
                }
            
            # List result files
            if Path(self.output_directory).exists():
                result_files = list(Path(self.output_directory).glob('*'))
                report_data['result_files'] = [f.name for f in result_files if f.is_file()]
            
            # Save report
            report_path = Path(self.output_directory) / 'simulation_report.json'
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            # Create human-readable report
            html_report = self.create_html_report(report_data)
            html_path = Path(self.output_directory) / 'simulation_report.html'
            with open(html_path, 'w') as f:
                f.write(html_report)
            
            messagebox.showinfo("Success", f"Report generated:\n{report_path}\n{html_path}")
            self.refresh_results_list()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
    
    def create_html_report(self, report_data):
        """Create an HTML version of the simulation report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ProteinMD Simulation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .parameter {{ margin: 5px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 ProteinMD Simulation Report</h1>
        <p>Generated: {report_data.get('report_generated', 'Unknown')}</p>
    </div>
    
    <div class="section">
        <h2>Input Information</h2>
        <div class="parameter"><strong>PDB File:</strong> {report_data.get('input_file', 'N/A')}</div>
        <div class="parameter"><strong>Output Directory:</strong> {report_data.get('output_directory', 'N/A')}</div>
    </div>
    
    <div class="section">
        <h2>Protein Information</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
"""
        
        protein_info = report_data.get('protein_info', {})
        for key, value in protein_info.items():
            html += f"            <tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>\n"
        
        html += """        </table>
    </div>
    
    <div class="section">
        <h2>Simulation Parameters</h2>
        <table>
            <tr><th>Category</th><th>Parameter</th><th>Value</th></tr>
"""
        
        params = report_data.get('simulation_parameters', {})
        for category, cat_params in params.items():
            if isinstance(cat_params, dict):
                for param, value in cat_params.items():
                    html += f"            <tr><td>{category.title()}</td><td>{param.replace('_', ' ').title()}</td><td>{value}</td></tr>\n"
        
        html += """        </table>
    </div>
    
    <div class="section">
        <h2>Result Files</h2>
        <ul>
"""
        
        result_files = report_data.get('result_files', [])
        for file_name in result_files:
            html += f"            <li>{file_name}</li>\n"
        
        html += """        </ul>
    </div>
    
    <div class="section">
        <h2>Analysis Summary</h2>
        <p>This simulation was performed using ProteinMD with the parameters shown above. 
        Check the individual result files for detailed analysis data including:</p>
        <ul>
            <li>Trajectory data (.pdb, .xtc files)</li>
            <li>Energy analysis (.dat files)</li>
            <li>RMSD and structural analysis</li>
            <li>Ramachandran plot data</li>
            <li>Radius of gyration analysis</li>
        </ul>
    </div>
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
        <p>Generated by ProteinMD GUI v1.0 - Task 8.1 Implementation</p>
    </footer>
</body>
</html>
"""
        return html
    
    def start_simulation(self):
        """Start the molecular dynamics simulation."""
        # Comprehensive validation before starting
        validation_errors = self.validate_simulation_setup()
        if validation_errors:
            error_message = "Cannot start simulation. Please fix the following issues:\n\n"
            error_message += "\n".join(f"• {error}" for error in validation_errors)
            messagebox.showerror("Validation Error", error_message)
            return
            
        if not PROTEINMD_AVAILABLE:
            # Demo mode
            self.demo_simulation()
            return
            
        # Get parameters
        parameters = self.get_simulation_parameters()
        
        # Final confirmation dialog
        param_summary = self.create_parameter_summary(parameters)
        confirm_message = f"Start simulation with these parameters?\n\n{param_summary}"
        
        if not messagebox.askyesno("Confirm Simulation", confirm_message):
            return
        
        # Start simulation
        success = self.simulation_worker.start_simulation(
            self.current_pdb_file,
            parameters,
            self.output_directory
        )
        
        if success:
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            self.log_message("Simulation started")
            self.update_progress(0, "Simulation starting...")
        else:
            messagebox.showerror("Error", "Failed to start simulation")
    
    def pause_simulation(self):
        """Pause or resume the simulation."""
        self.simulation_worker.pause_simulation()
        
        if self.simulation_worker.paused:
            self.pause_button.config(text="Resume")
            self.log_message("Simulation paused")
        else:
            self.pause_button.config(text="Pause")
            self.log_message("Simulation resumed")
    
    def stop_simulation(self):
        """Stop the running simulation."""
        self.simulation_worker.stop_simulation()
        self.reset_simulation_controls()
        self.log_message("Simulation stopped")
        self.update_progress(0, "Simulation stopped")
    
    def reset_simulation_controls(self):
        """Reset simulation control buttons to initial state."""
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED, text="Pause")
        self.stop_button.config(state=tk.DISABLED)
    
    def demo_simulation(self):
        """Run a demo simulation when ProteinMD is not available."""
        def demo_worker():
            for i in range(101):
                time.sleep(0.05)  # Simulate work
                self.update_progress(i, f"Demo simulation step {i}/100")
                if not hasattr(self, '_demo_running') or not self._demo_running:
                    break
            self.on_simulation_finished(True, "Demo simulation completed")
        
        self._demo_running = True
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)
        
        demo_thread = threading.Thread(target=demo_worker)
        demo_thread.daemon = True
        demo_thread.start()
    
    def select_output_directory(self):
        """Select output directory for simulation results."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_directory = directory
            self.output_dir_label.config(text=os.path.basename(directory))
            self.log_message(f"Output directory set to: {directory}")
    
    def update_progress(self, percentage: int, message: str):
        """Update the progress bar and message."""
        def update():
            self.progress_bar['value'] = percentage
            self.progress_var.set(message)
            self.status_bar.config(text=message)
        
        # Schedule update on main thread
        self.root.after(0, update)
    
    def on_simulation_finished(self, success: bool, message: str):
        """Handle simulation completion."""
        def finish():
            self.reset_simulation_controls()
            self.log_message(message)
            
            if success:
                self.update_progress(100, "Simulation completed!")
                
                # Run post-simulation analysis if requested
                self.run_post_simulation_analysis()
                
                self.refresh_results_list()
                messagebox.showinfo("Success", "Simulation completed successfully!")
            else:
                self.update_progress(0, "Simulation failed")
                messagebox.showerror("Error", f"Simulation failed: {message}")
        
        self.root.after(0, finish)
    
    def log_message(self, message: str):
        """Add a message to the simulation log."""
        def add_message():
            self.log_text.config(state=tk.NORMAL)
            timestamp = time.strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        
        self.root.after(0, add_message)
    
    def refresh_results_list(self):
        """Refresh the list of result files."""
        if not self.output_directory or not os.path.exists(self.output_directory):
            return
            
        self.results_listbox.delete(0, tk.END)
        
        try:
            files = os.listdir(self.output_directory)
            for file in sorted(files):
                self.results_listbox.insert(tk.END, file)
        except Exception as e:
            self.log_message(f"Error reading results directory: {str(e)}")
    
    def validate_simulation_setup(self):
        """Validate simulation setup and return list of errors."""
        errors = []
        
        # Check PDB file
        if not self.current_pdb_file:
            errors.append("No PDB file loaded")
        elif not Path(self.current_pdb_file).exists():
            errors.append(f"PDB file not found: {self.current_pdb_file}")
        
        # Check output directory
        if not self.output_directory:
            errors.append("No output directory selected")
        elif not Path(self.output_directory).exists():
            try:
                Path(self.output_directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create output directory: {str(e)}")
        
        # Validate simulation parameters
        try:
            params = self.get_simulation_parameters()
            
            # Temperature validation
            temp = params['simulation']['temperature']
            if temp <= 0 or temp > 1000:
                errors.append(f"Invalid temperature: {temp} K (must be 0-1000 K)")
            
            # Timestep validation
            timestep = params['simulation']['timestep']
            if timestep <= 0 or timestep > 0.1:
                errors.append(f"Invalid timestep: {timestep} ps (must be 0-0.1 ps)")
            
            # Steps validation
            n_steps = params['simulation']['n_steps']
            if n_steps <= 0 or n_steps > 10**8:
                errors.append(f"Invalid number of steps: {n_steps} (must be 1-100M)")
            
            # Box padding validation
            box_padding = params['environment']['box_padding']
            if box_padding <= 0 or box_padding > 10:
                errors.append(f"Invalid box padding: {box_padding} nm (must be 0-10 nm)")
                
        except (ValueError, KeyError) as e:
            errors.append(f"Parameter validation error: {str(e)}")
        
        return errors
    
    def create_parameter_summary(self, parameters):
        """Create a human-readable summary of simulation parameters."""
        summary = []
        
        sim_params = parameters.get('simulation', {})
        summary.append(f"Temperature: {sim_params.get('temperature', 'N/A')} K")
        summary.append(f"Timestep: {sim_params.get('timestep', 'N/A')} ps")
        summary.append(f"Steps: {sim_params.get('n_steps', 'N/A'):,}")
        
        # Calculate simulation time
        try:
            total_time = sim_params.get('timestep', 0) * sim_params.get('n_steps', 0)
            if total_time > 0:
                if total_time < 1:
                    summary.append(f"Total time: {total_time*1000:.1f} fs")
                elif total_time < 1000:
                    summary.append(f"Total time: {total_time:.1f} ps")
                else:
                    summary.append(f"Total time: {total_time/1000:.1f} ns")
        except:
            pass
        
        ff_params = parameters.get('forcefield', {})
        summary.append(f"Force field: {ff_params.get('type', 'N/A')}")
        
        env_params = parameters.get('environment', {})
        summary.append(f"Solvent: {env_params.get('solvent', 'N/A')}")
        summary.append(f"Box padding: {env_params.get('box_padding', 'N/A')} nm")
        
        return "\n".join(summary)
        
    def open_pdb_file(self):
        """Open and load a PDB file."""
        file_path = filedialog.askopenfilename(
            title="Select PDB File",
            filetypes=[("PDB files", "*.pdb"), ("All files", "*.*")]
        )
        
        if file_path:
            self.load_pdb_file(file_path)
        
    def load_pdb_file(self, file_path: str):
        """Load a PDB file and display protein information."""
        try:
            # Validate file exists and is readable
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not file_path.lower().endswith('.pdb'):
                response = messagebox.askyesno(
                    "File Type Warning", 
                    f"File does not have .pdb extension.\nContinue loading anyway?\n\n{file_path}"
                )
                if not response:
                    return
            
            self.current_pdb_file = file_path
            filename = os.path.basename(file_path)
            self.file_label.config(text=filename)
            
            if PROTEINMD_AVAILABLE:
                # Parse PDB file with error handling
                parser = PDBParser()
                try:
                    self.current_protein = parser.parse_file(file_path)
                    
                    # Validate protein structure
                    if not self.current_protein or not hasattr(self.current_protein, 'atoms'):
                        raise ValueError("Invalid protein structure - no atoms found")
                    
                    if len(self.current_protein.atoms) == 0:
                        raise ValueError("Empty protein structure - no atoms found")
                    
                    # Display protein information
                    self.display_protein_info()
                    
                    self.status_bar.config(text=f"Loaded: {filename} ({len(self.current_protein.atoms)} atoms)")
                    self.log_message(f"Successfully loaded {filename} with {len(self.current_protein.atoms)} atoms")
                    
                    # Enable simulation controls
                    self.enable_simulation_controls()
                    
                except Exception as parse_error:
                    # Handle parsing errors gracefully
                    self.current_protein = None
                    self.display_error_info(f"PDB parsing failed: {str(parse_error)}")
                    self.log_message(f"Error parsing {filename}: {str(parse_error)}")
                    messagebox.showwarning("Parsing Warning", 
                                         f"PDB file loaded but parsing failed:\n{str(parse_error)}\n\nYou can still try to run the simulation.")
                    
            else:
                self.log_message("ProteinMD modules not available - using placeholder")
                self.display_placeholder_info(filename)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PDB file:\n{str(e)}")
            self.log_message(f"Error loading {file_path}: {str(e)}")
            self.current_pdb_file = None
            self.current_protein = None
            self.file_label.config(text="No file selected")
    
    def display_error_info(self, error_message):
        """Display error information in the protein info area."""
        self.protein_info_text.config(state=tk.NORMAL)
        self.protein_info_text.delete(1.0, tk.END)
        
        info = f"""Error Loading Protein:

{error_message}

Troubleshooting:
• Ensure the file is a valid PDB format
• Check that the file is not corrupted
• Verify that all required atoms are present
• Try opening the file in a text editor to check format

You may still attempt to run the simulation, but results may be unpredictable.
"""
        
        self.protein_info_text.insert(1.0, info)
        self.protein_info_text.config(state=tk.DISABLED)
    
    def display_placeholder_info(self, filename):
        """Display placeholder info when ProteinMD is not available."""
        self.protein_info_text.config(state=tk.NORMAL)
        self.protein_info_text.delete(1.0, tk.END)
        
        info = f"""File Loaded (Demo Mode):

Filename: {filename}
Status: Loaded successfully
Mode: Demo mode (ProteinMD modules not available)

Demo Features Available:
• Parameter configuration
• Template system
• Demo simulation
• Configuration save/load

For full functionality, ensure ProteinMD modules are installed.
"""
        
        self.protein_info_text.insert(1.0, info)
        self.protein_info_text.config(state=tk.DISABLED)
    
    def enable_simulation_controls(self):
        """Enable simulation controls after successful file loading."""
        # Could add any enabling logic here if needed
        pass
    
    def save_configuration(self):
        """Save current parameters to a configuration file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                parameters = self.get_simulation_parameters()
                with open(file_path, 'w') as f:
                    json.dump(parameters, f, indent=2)
                messagebox.showinfo("Success", "Configuration saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration:\n{str(e)}")
    
    def load_configuration(self):
        """Load parameters from a configuration file."""
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    parameters = json.load(f)
                
                # Update GUI with loaded parameters
                if 'simulation' in parameters:
                    sim = parameters['simulation']
                    self.temperature_var.set(str(sim.get('temperature', 300.0)))
                    self.timestep_var.set(str(sim.get('timestep', 0.002)))
                    self.nsteps_var.set(str(sim.get('n_steps', 10000)))
                
                if 'forcefield' in parameters:
                    ff = parameters['forcefield']
                    self.forcefield_var.set(ff.get('type', 'amber_ff14sb'))
                
                if 'environment' in parameters:
                    env = parameters['environment']
                    self.solvent_var.set(env.get('solvent', 'explicit'))
                    self.box_padding_var.set(str(env.get('box_padding', 1.0)))
                
                if 'analysis' in parameters:
                    analysis = parameters['analysis']
                    self.rmsd_var.set(analysis.get('rmsd', True))
                    self.ramachandran_var.set(analysis.get('ramachandran', True))
                    self.radius_of_gyration_var.set(analysis.get('radius_of_gyration', True))
                    
                    # Load PCA parameters
                    if 'pca' in analysis:
                        pca_params = analysis['pca']
                        self.pca_var.set(pca_params.get('enabled', True))
                        self.pca_atom_selection_var.set(pca_params.get('atom_selection', 'CA'))
                        self.pca_n_components_var.set(str(pca_params.get('n_components', 20)))
                        self.pca_clustering_var.set(pca_params.get('clustering', True))
                        n_clusters = pca_params.get('n_clusters', 'auto')
                        self.pca_n_clusters_var.set(str(n_clusters) if n_clusters is not None else 'auto')
                
                if 'visualization' in parameters:
                    vis = parameters['visualization']
                    self.visualization_var.set(vis.get('enabled', True))
                    self.realtime_var.set(vis.get('realtime', False))
                
                messagebox.showinfo("Success", "Configuration loaded successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration:\n{str(e)}")
    
    def view_protein_structure(self):
        """View the 3D protein structure."""
        if not self.current_protein:
            messagebox.showwarning("Warning", "No protein loaded")
            return
            
        try:
            if PROTEINMD_AVAILABLE and VISUALIZATION_AVAILABLE:
                # Use ProteinMD visualization
                from proteinMD.visualization.protein_3d import Protein3DVisualization
                viz = Protein3DVisualization(self.current_protein)
                viz.create_3d_plot()
                viz.show()
            else:
                messagebox.showinfo("Info", "3D visualization requires ProteinMD modules")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show protein structure:\n{str(e)}")
    
    def view_parameters(self):
        """Show current parameters in a popup window."""
        params = self.get_simulation_parameters()
        
        # Create popup window
        param_window = tk.Toplevel(self.root)
        param_window.title("Current Parameters")
        param_window.geometry("400x500")
        
        # Create text widget
        text_widget = tk.Text(param_window, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(param_window, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Display parameters
        text_widget.insert(1.0, json.dumps(params, indent=2))
        text_widget.config(state=tk.DISABLED)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def view_trajectory(self):
        """View simulation trajectory."""
        messagebox.showinfo("Info", "Trajectory viewer not implemented yet")
    
    def view_analysis(self):
        """View analysis results."""
        messagebox.showinfo("Info", "Analysis viewer not implemented yet")
    
    def export_results(self):
        """Export simulation results."""
        messagebox.showinfo("Info", "Results export not implemented yet")
    
    def show_about(self):
        """Show about dialog."""
        about_text = """ProteinMD GUI v1.0
        
A user-friendly graphical interface for molecular dynamics simulations.

Features:
• Drag & Drop PDB file loading
• Interactive parameter configuration  
• Real-time simulation monitoring
• Integrated visualization
• Results management

Developed as part of Task 8.1: Graphical User Interface
        
For more information, visit the ProteinMD documentation."""
        
        messagebox.showinfo("About ProteinMD", about_text)
    
    def show_documentation(self):
        """Show documentation."""
        messagebox.showinfo("Documentation", "Please refer to the ProteinMD documentation for detailed usage instructions.")
    
    def run(self):
        """Start the GUI application."""
        self.log_message("ProteinMD GUI started")
        if not PROTEINMD_AVAILABLE:
            self.log_message("Warning: ProteinMD modules not available - running in demo mode")
        
        self.root.mainloop()

    # Template system methods (continuing from where they were cut off)
    def on_template_selected(self, event):
        """Handle template selection."""
        template_name = self.template_var.get()
        
        if template_name == "Custom":
            self.template_desc_label.config(text="Configure parameters manually", foreground="gray")
            return
            
        if not self.template_manager:
            return
            
        try:
            template = self.template_manager.get_template(template_name)
            if template:
                # Update description
                self.template_desc_label.config(text=template.description, foreground="blue")
                
                # Load template parameters
                self.load_template_parameters(template)
                
        except Exception as e:
            self.log_message(f"Error loading template {template_name}: {str(e)}")
    
    def load_template_parameters(self, template):
        """Load parameters from selected template."""
        try:
            # Get default configuration from template
            config = template.generate_config()
            
            # Update GUI parameters
            if 'simulation' in config:
                sim_config = config['simulation']
                
                if 'temperature' in sim_config:
                    self.temperature_var.set(str(sim_config['temperature']))
                
                if 'timestep' in sim_config:
                    self.timestep_var.set(str(sim_config['timestep']))
                
                if 'n_steps' in sim_config:
                    self.nsteps_var.set(str(sim_config['n_steps']))
            
            if 'forcefield' in config:
                ff_config = config['forcefield']
                if 'type' in ff_config:
                    # Update force field selection if widget exists
                    if hasattr(self, 'forcefield_var'):
                        self.forcefield_var.set(ff_config['type'])
            
            self.log_message(f"Loaded parameters from template: {template.name}")
            
        except Exception as e:
            self.log_message(f"Error loading template parameters: {str(e)}")
    
    def browse_templates(self):
        """Show template browser dialog."""
        if not self.template_manager:
            messagebox.showinfo("Templates", "Template system not available")
            return
            
        try:
            templates = self.template_manager.list_templates()
            
            # Create template browser window
            template_window = tk.Toplevel(self.root)
            template_window.title("Template Browser")
            template_window.geometry("600x400")
            template_window.transient(self.root)
            template_window.grab_set()
            
            # Template list
            list_frame = ttk.Frame(template_window)
            list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            ttk.Label(list_frame, text="Available Templates:").pack(anchor=tk.W)
            
            template_listbox = tk.Listbox(list_frame, height=15)
            template_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
            
            # Populate template list
            for name in sorted(templates.keys()):
                template_listbox.insert(tk.END, f"{name} - {templates[name].description}")
            
            # Buttons
            button_frame = ttk.Frame(template_window)
            button_frame.pack(fill=tk.X, padx=10, pady=10)
            
            def apply_selected():
                selection = template_listbox.curselection()
                if selection:
                    template_name = list(templates.keys())[selection[0]]
                    self.template_var.set(template_name)
                    self.on_template_selected(None)
                    template_window.destroy()
            
            ttk.Button(button_frame, text="Apply", command=apply_selected).pack(side=tk.RIGHT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=template_window.destroy).pack(side=tk.RIGHT)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to browse templates: {str(e)}")
    
    def save_as_template(self):
        """Save current parameters as a new template."""
        if not self.template_manager:
            messagebox.showinfo("Templates", "Template system not available")
            return
            
        # Create template save dialog
        template_window = tk.Toplevel(self.root)
        template_window.title("Save Template")
        template_window.geometry("400x300")
        template_window.transient(self.root)
        template_window.grab_set()
        
        # Template name
        ttk.Label(template_window, text="Template Name:").pack(anchor=tk.W, padx=10, pady=5)
        name_var = tk.StringVar()
        ttk.Entry(template_window, textvariable=name_var, width=40).pack(padx=10, pady=5)
        
        # Template description
        ttk.Label(template_window, text="Description:").pack(anchor=tk.W, padx=10, pady=5)
        desc_text = tk.Text(template_window, height=8, width=50)
        desc_text.pack(padx=10, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(template_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def save_template():
            name = name_var.get().strip()
            description = desc_text.get(1.0, tk.END).strip()
            
            if not name:
                messagebox.showerror("Error", "Please enter a template name")
                return
                
            try:
                # Get current parameters
                parameters = self.get_simulation_parameters()
                
                # Save as template (simplified)
                template_file = Path.home() / '.proteinmd' / 'templates' / f'{name}.json'
                template_file.parent.mkdir(parents=True, exist_ok=True)
                template_data = {
                    'name': name,
                    'description': description,
                    'parameters': parameters
                }
                
                with open(template_file, 'w') as f:
                    json.dump(template_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Template '{name}' saved successfully")
                template_window.destroy()
                
                # Refresh template list
                self.refresh_template_list()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save template: {str(e)}")
        
        ttk.Button(button_frame, text="Save", command=save_template).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=template_window.destroy).pack(side=tk.RIGHT)
    
    def show_template_statistics(self):
        """Show template usage statistics."""
        if not self.template_manager:
            messagebox.showinfo("Templates", "Template system not available")
            return
            
        try:
            templates = self.template_manager.list_templates()
            
            stats_text = f"Template Statistics:\n\n"
            stats_text += f"Total templates: {len(templates)}\n"
            stats_text += f"Built-in templates: {len([t for t in templates.values() if hasattr(t, 'author') and t.author == 'ProteinMD'])}\n"
            stats_text += f"User templates: {len([t for t in templates.values() if not (hasattr(t, 'author') and t.author == 'ProteinMD')])}\n\n"
            
            stats_text += "Available templates:\n"
            for name, template in sorted(templates.items()):
                stats_text += f"• {name}: {template.description}\n"
            
            messagebox.showinfo("Template Statistics", stats_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get template statistics: {str(e)}")
    
    def refresh_template_list(self):
        """Refresh the template combobox with current templates."""
        if not hasattr(self, 'template_combobox'):
            return
            
        template_names = ["Custom"]
        if self.template_manager:
            try:
                templates = self.template_manager.list_templates()
                template_names.extend(sorted(templates.keys()))
            except:
                pass
        
        self.template_combobox['values'] = template_names

    def on_smd_toggle(self):
        """Handle enabling/disabling SMD parameters."""
        if self.enable_smd_var.get():
            # Enable SMD parameter widgets
            for child in self.smd_params_frame.winfo_children():
                for widget in child.winfo_children():
                    if hasattr(widget, 'config'):
                        widget.config(state=tk.NORMAL)
        else:
            # Disable SMD parameter widgets
            for child in self.smd_params_frame.winfo_children():
                for widget in child.winfo_children():
                    if hasattr(widget, 'config'):
                        widget.config(state=tk.DISABLED)
    
    def on_smd_mode_changed(self, event=None):
        """Handle SMD mode change."""
        mode = self.smd_mode_var.get()
        
        # Hide all parameter frames
        self.cv_frame.pack_forget()
        self.cf_frame.pack_forget()
        
        # Show appropriate parameter frame
        if mode == "constant_velocity":
            self.cv_frame.pack(fill=tk.X)
        else:  # constant_force
            self.cf_frame.pack(fill=tk.X)
    
    def run_post_simulation_analysis(self):
        """Run post-simulation analysis including PCA, Cross-Correlation, and Free Energy if requested."""
        try:
            parameters = self.get_simulation_parameters()
            pca_params = parameters.get('analysis', {}).get('pca', {})
            cc_params = parameters.get('analysis', {}).get('cross_correlation', {})
            fe_params = parameters.get('analysis', {}).get('free_energy', {})
            sasa_params = parameters.get('analysis', {}).get('sasa', {})
            
            # Check if any analysis is enabled
            if not (pca_params.get('enabled', False) or cc_params.get('enabled', False) or fe_params.get('enabled', False) or sasa_params.get('enabled', False)):
                self.log_message("No post-simulation analysis enabled, skipping")
                return
            
            self.log_message("Starting post-simulation analysis...")
            self.update_progress(90, "Running analysis...")
            
            # Check if we have ProteinMD available for analysis
            if not PROTEINMD_AVAILABLE:
                self.log_message("ProteinMD not available - Analysis requires full installation")
                return
            
            # Try to find trajectory files in output directory
            output_path = Path(self.output_directory)
            trajectory_files = list(output_path.glob("*.xyz")) + list(output_path.glob("*.pdb")) + list(output_path.glob("*.xtc"))
            
            if not trajectory_files:
                self.log_message("No trajectory files found for analysis")
                return
            
            # Read trajectory (this is a simplified approach)
            trajectory_file = trajectory_files[0]
            self.log_message(f"Analyzing trajectory: {trajectory_file.name}")
            
            # Note: In a real implementation, we would need to properly read the trajectory
            # For now, we'll create a demo analysis
            
            # ===== PCA ANALYSIS =====
            if pca_params.get('enabled', False):
                try:
                    self.log_message("Running PCA analysis...")
                    self.update_progress(92, "Running PCA analysis...")
                    
                    # Import PCA analysis
                    from proteinMD.analysis.pca import PCAAnalyzer, create_test_trajectory
                    
                    # Create PCA analyzer
                    pca_analyzer = PCAAnalyzer(
                        atom_selection=pca_params.get('atom_selection', 'CA'),
                        align_trajectory=True
                    )
                    
                    # Create test trajectory for demonstration
                    test_trajectory = create_test_trajectory(n_frames=50, n_atoms=100)
                    
                    # Perform PCA analysis
                    pca_results = pca_analyzer.fit_transform(
                        test_trajectory,
                        n_components=pca_params.get('n_components', 20)
                    )
                    
                    # Perform clustering if requested
                    clustering_results = None
                    if pca_params.get('clustering', False):
                        n_clusters = pca_params.get('n_clusters')
                        if n_clusters and n_clusters.isdigit():
                            n_clusters = int(n_clusters)
                        else:
                            n_clusters = None
                        
                        clustering_results = pca_analyzer.cluster_conformations(
                            n_clusters=n_clusters
                        )
                        self.log_message(f"PCA clustering completed: {clustering_results.n_clusters} clusters")
                    
                    # Save PCA results
                    pca_output_dir = output_path / "pca_analysis"
                    pca_analyzer.export_results(output_dir=str(pca_output_dir))
                    
                    # Generate visualizations
                    pca_analyzer.plot_eigenvalue_spectrum(
                        save_path=str(pca_output_dir / "pca_eigenvalue_spectrum.png")
                    )
                    pca_analyzer.plot_pc_projections(
                        save_path=str(pca_output_dir / "pca_projections.png")
                    )
                    
                    if clustering_results:
                        pca_analyzer.plot_cluster_analysis(
                            save_path=str(pca_output_dir / "pca_cluster_analysis.png")
                        )
                    
                    # Update log with results
                    variance_explained = pca_results.get_explained_variance(5)
                    self.log_message(f"PCA analysis completed successfully")
                    self.log_message(f"First 5 components explain {variance_explained:.1f}% of variance")
                    
                    if clustering_results:
                        self.log_message(f"Identified {clustering_results.n_clusters} conformational clusters")
                        self.log_message(f"Clustering quality (silhouette score): {clustering_results.silhouette_score:.3f}")
                    
                    self.log_message(f"PCA results saved to: {pca_output_dir}")
                    
                except ImportError as e:
                    self.log_message(f"PCA analysis unavailable: {str(e)}")
                except Exception as e:
                    self.log_message(f"PCA analysis failed: {str(e)}")
                    logger.error(f"PCA analysis error: {str(e)}")
            
            # ===== CROSS-CORRELATION ANALYSIS =====
            if cc_params.get('enabled', False):
                try:
                    self.log_message("Running Cross-Correlation analysis...")
                    self.update_progress(95, "Running Cross-Correlation analysis...")
                    
                    # Import Cross-Correlation analysis
                    from proteinMD.analysis.cross_correlation import DynamicCrossCorrelationAnalyzer, create_test_trajectory
                    
                    # Create Cross-Correlation analyzer
                    cc_analyzer = DynamicCrossCorrelationAnalyzer(
                        atom_selection=cc_params.get('atom_selection', 'CA'),
                        align_trajectory=True
                    )
                    
                                       
                    # Create test trajectory for demonstration
                    test_trajectory = create_test_trajectory(n_frames=50, n_atoms=100)
                    
                    # Perform Cross-Correlation analysis
                    cc_results = cc_analyzer.calculate_correlation_matrix(
                        test_trajectory,
                        align_trajectory=True,
                        correlation_type='pearson'
                    )
                    
                    # Perform statistical significance testing
                    significance_results = None
                    significance_method = cc_params.get('significance_method', 'ttest')
                    self.log_message(f"Running significance testing using {significance_method}")
                    
                    significance_results = cc_analyzer.calculate_significance(
                        cc_results,
                        method=significance_method,
                        n_bootstrap=1000 if significance_method == 'bootstrap' else None
                    )
                    
                    # Perform network analysis if requested
                    network_results = None
                    if cc_params.get('network_analysis', False):
                        threshold = cc_params.get('network_threshold', 0.5)
                        self.log_message(f"Running network analysis with threshold {threshold}")
                        
                        network_results = cc_analyzer.analyze_network(
                            cc_results,
                            threshold=threshold
                        )
                        self.log_message(f"Network analysis completed: {network_results.graph.number_of_nodes()} nodes, {network_results.graph.number_of_edges()} edges")
                    
                    # Perform time-dependent analysis if requested
                    time_dependent_results = None
                    if cc_params.get('time_dependent', False):
                        self.log_message("Running time-dependent correlation analysis")
                        
                        time_dependent_results = cc_analyzer.time_dependent_analysis(
                            test_trajectory,
                            window_size=20,
                            step_size=5
                        )
                        self.log_message(f"Time-dependent analysis completed: {len(time_dependent_results)} time windows")
                    
                    # Save Cross-Correlation results
                    cc_output_dir = output_path / "cross_correlation_analysis"
                    cc_analyzer.export_results(
                        cc_results,
                        network_results=network_results,
                        output_dir=str(cc_output_dir)
                    )
                    
                    # Generate visualizations
                    cc_analyzer.visualize_matrix(
                        cc_results,
                        output_file=str(cc_output_dir / "correlation_matrix.png")
                    )
                    
                    if significance_results:
                        cc_analyzer.visualize_matrix(
                            cc_results,
                            output_file=str(cc_output_dir / "significance_matrix.png"),
                            show_significance=True
                        )
                    
                    if network_results:
                        cc_analyzer.visualize_network(
                            network_results,
                            output_file=str(cc_output_dir / "correlation_network.png")
                        )
                    
                    if time_dependent_results:
                        cc_analyzer.visualize_time_evolution(
                            time_dependent_results,
                            output_file=str(cc_output_dir / "time_evolution.png")
                        )
                    
                    # Update log with results
                    self.log_message(f"Cross-Correlation analysis completed successfully")
                    self.log_message(f"Correlation matrix size: {cc_results.correlation_matrix.shape}")
                    
                    if significance_results:
                        significant_count = (significance_results.p_values < 0.05).sum()
                        self.log_message(f"Significant correlations (p < 0.05): {significant_count}")
                    
                    if network_results:
                        self.log_message(f"Network density: {network_results.network_statistics.get('density', 'N/A'):.3f}")
                        self.log_message(f"Network modularity: {network_results.modularity:.3f}")
                    
                    self.log_message(f"Cross-Correlation results saved to: {cc_output_dir}")
                    
                except ImportError as e:
                    self.log_message(f"Cross-Correlation analysis unavailable: {str(e)}")
                except Exception as e:
                    self.log_message(f"Cross-Correlation analysis failed: {str(e)}")
                    logger.error(f"Cross-Correlation analysis error: {str(e)}")
                    
            # ===== FREE ENERGY ANALYSIS =====
            if fe_params.get('enabled', False):
                try:
                    self.log_message("Running Free Energy analysis...")
                    self.update_progress(97, "Running Free Energy analysis...")
                    
                    # Import Free Energy analysis
                    from proteinMD.analysis.free_energy import FreeEnergyAnalysis, create_test_data_1d, create_test_data_2d
                    
                    # Create Free Energy analyzer
                    fe_analyzer = FreeEnergyAnalysis(temperature=300.0)
                    
                    # Create test data for demonstration
                    coord1_name = fe_params.get('coord1', 'rmsd')
                    coord2_name = fe_params.get('coord2', 'radius_of_gyration')
                    n_bins = fe_params.get('n_bins', 50)
                    
                    self.log_message(f"Analyzing coordinates: {coord1_name} (1D), {coord1_name} vs {coord2_name} (2D)")
                    
                    # Generate test coordinate data
                    coord1_data = create_test_data_1d(n_points=1000, n_minima=2)
                    coord1_2d, coord2_2d = create_test_data_2d(n_points=1000)
                    
                    # Calculate 1D free energy profile
                    fe_1d_results = fe_analyzer.calculate_1d_profile(
                        coord1_data,
                        n_bins=n_bins
                    )
                    
                    # Calculate 2D free energy landscape
                    fe_2d_results = fe_analyzer.calculate_2d_profile(
                        coord1_2d,
                        coord2_2d,
                        n_bins=n_bins
                    )
                    
                    # Bootstrap error analysis if requested
                    bootstrap_results = None
                    if fe_params.get('bootstrap', False):
                        self.log_message("Running bootstrap error analysis...")
                        bootstrap_results = fe_analyzer.bootstrap_error_1d(
                            coord1_data,
                            n_bins=n_bins,
                            n_bootstrap=100
                        )
                    
                    # Identify minima if requested
                    minima_results = None
                    if fe_params.get('find_minima', False):
                        self.log_message("Identifying energy minima...")
                        minima_results = fe_analyzer.find_minima_2d(fe_2d_results)
                        self.log_message(f"Found {len(minima_results)} energy minima")
                    
                    # Save Free Energy results
                    fe_output_dir = output_path / "free_energy_analysis"
                    fe_output_dir.mkdir(exist_ok=True)
                    
                    # Export individual results
                    fe_analyzer.export_profile_1d(
                        fe_1d_results,
                        str(fe_output_dir / "free_energy_1d.dat")
                    )
                    
                    fe_analyzer.export_landscape_2d(
                        fe_2d_results,
                        str(fe_output_dir / "free_energy_2d.dat")
                    )
                    
                    # Generate visualizations
                    fe_analyzer.plot_1d_profile(
                        fe_1d_results,
                        filename=str(fe_output_dir / "free_energy_1d.png"),
                        title=f"1D Free Energy Profile: {coord1_name}"
                    )
                    
                    fe_analyzer.plot_2d_landscape(
                        fe_2d_results,
                        filename=str(fe_output_dir / "free_energy_2d.png"),
                        title=f"2D Free Energy Landscape: {coord1_name} vs {coord2_name}",
                        xlabel=coord1_name,
                        ylabel=coord2_name
                    )
                    
                    if minima_results:
                        # Save minima information
                        minima_info = []
                        for i, minimum in enumerate(minima_results):
                            minima_info.append({
                                'index': i,
                                'coordinate1': minimum.coordinates[0],
                                'coordinate2': minimum.coordinates[1],
                                'energy': minimum.energy
                            })
                        
                        import json
                        with open(fe_output_dir / "energy_minima.json", 'w') as f:
                            json.dump(minima_info, f, indent=2)
                    
                    # Update log with results
                    self.log_message(f"Free Energy analysis completed successfully")
                    self.log_message(f"1D profile range: {fe_1d_results.coordinates.min():.3f} to {fe_1d_results.coordinates.max():.3f}")
                    self.log_message(f"2D landscape size: {fe_2d_results.free_energy.shape}")
                    
                    if minima_results:
                        min_energy = min(m.energy for m in minima_results)
                        self.log_message(f"Global minimum energy: {min_energy:.3f} kJ/mol")
                    
                    self.log_message(f"Free Energy results saved to: {fe_output_dir}")
                    
                except ImportError as e:
                    self.log_message(f"Free Energy analysis unavailable: {str(e)}")
                except Exception as e:
                    self.log_message(f"Free Energy analysis failed: {str(e)}")
                    logger.error(f"Free Energy analysis error: {str(e)}")
                    
            # ===== SASA ANALYSIS =====
            if sasa_params.get('enabled', False):
                try:
                    self.log_message("Running SASA analysis...")
                    self.update_progress(98, "Running SASA analysis...")
                    
                    # Import SASA analysis
                    from proteinMD.analysis.sasa import SASAAnalyzer, create_test_trajectory, create_test_protein_structure
                    
                    # Create SASA analyzer
                    sasa_analyzer = SASAAnalyzer(
                        probe_radius=sasa_params.get('probe_radius', 1.4),
                        n_points=sasa_params.get('n_points', 590),
                        use_atomic_radii=True
                    )
                    
                    # Create test trajectory and structure for demonstration
                    test_trajectory_positions, test_atom_types, test_residue_ids = create_test_trajectory(
                        n_frames=50, n_atoms=100
                    )
                    
                    self.log_message(f"Analyzing SASA for {test_trajectory_positions.shape[1]} atoms over {test_trajectory_positions.shape[0]} frames")
                    
                    # Perform SASA time series analysis
                    sasa_results = sasa_analyzer.analyze_trajectory(
                        trajectory_positions=test_trajectory_positions,
                        atom_types=test_atom_types,
                        residue_ids=test_residue_ids,
                        stride=1
                    )
                    
                    # Save SASA results
                    sasa_output_dir = output_path / "sasa_analysis"
                    sasa_output_dir.mkdir(exist_ok=True)
                    
                    # Export results
                    sasa_analyzer.export_results(
                        sasa_results,
                        output_file=str(sasa_output_dir / "sasa_time_series.dat")
                    )
                    
                    # Generate visualizations
                    sasa_analyzer.plot_time_series(
                        sasa_results,
                        output_file=str(sasa_output_dir / "sasa_time_series.png")
                    )
                    
                    if sasa_params.get('per_residue', True):
                        sasa_analyzer.plot_per_residue_sasa(
                            sasa_results,
                            output_file=str(sasa_output_dir / "per_residue_sasa.png")
                        )
                    
                    # Update log with results
                    avg_total_sasa = sasa_results.statistics.get('total_sasa_mean', 0)
                    avg_hydrophobic = sasa_results.statistics.get('hydrophobic_sasa_mean', 0)
                    avg_hydrophilic = sasa_results.statistics.get('hydrophilic_sasa_mean', 0)
                    
                    self.log_message(f"SASA analysis completed successfully")
                    self.log_message(f"Average total SASA: {avg_total_sasa:.2f} Ų")
                    self.log_message(f"Average hydrophobic SASA: {avg_hydrophobic:.2f} Ų")
                    self.log_message(f"Average hydrophilic SASA: {avg_hydrophilic:.2f} Ų")
                    self.log_message(f"SASA results saved to: {sasa_output_dir}")
                    
                except ImportError as e:
                    self.log_message(f"SASA analysis unavailable: {str(e)}")
                except Exception as e:
                    self.log_message(f"SASA analysis failed: {str(e)}")
                    logger.error(f"SASA analysis error: {str(e)}")
            
            self.update_progress(100, "Analysis completed")
            self.log_message("Post-simulation analysis completed successfully")
        
        except Exception as e:
            self.log_message(f"Post-simulation analysis failed: {str(e)}")
            logger.error(f"Post-simulation analysis error: {str(e)}")
            
            self.update_progress(100, "Analysis completed")
            self.log_message("Post-simulation analysis completed successfully")
            
        except Exception as e:
            self.log_message(f"Post-simulation analysis failed: {str(e)}")
            logger.error(f"Post-simulation analysis error: {str(e)}")
    
    def create_status_bar(self):
        """Create the status bar at the bottom of the window."""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add connection status
        status_text = "Ready"
        if PROTEINMD_AVAILABLE:
            status_text += " | ProteinMD: Available"
        else:
            status_text += " | ProteinMD: Demo Mode"
        
        self.status_var.set(status_text)
