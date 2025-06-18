#!/usr/bin/env python3
"""
ProteinMD Command Line Interface

Task 8.3: Command Line Interface 🚀
Status: IMPLEMENTING

Vollständige CLI für automatisierte Workflows

This module provides a comprehensive command-line interface for ProteinMD,
enabling automated workflows for molecular dynamics simulations, analysis,
and visualization.

Features:
- Complete simulation workflows
- Analysis pipeline automation  
- Batch processing capabilities
- Template-based configuration
- Progress monitoring and reporting
- Error handling and logging
- Return codes for script integration
"""

import argparse
import sys
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime
import traceback
import subprocess
import time
import numpy as np

# Add proteinMD to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import proteinMD modules
try:
    # Core simulation
    from proteinMD.core.simulation import MolecularDynamicsSimulation
    from proteinMD.structure.protein import Protein
    from proteinMD.structure.pdb_parser import PDBParser
    
    # Force fields
    from proteinMD.forcefield.amber_ff14sb import AmberFF14SB
    
    # Template system
    from proteinMD.templates.template_manager import TemplateManager
    from proteinMD.templates.builtin_templates import BUILTIN_TEMPLATES
    from proteinMD.templates.template_cli import TemplateCLI
    
    # Workflow automation
    from proteinMD.workflow import WorkflowCLI
    
    # Database integration
    from proteinMD.database import DatabaseCLI
    
    # Environment
    from proteinMD.environment.water import TIP3PWaterModel, WaterSolvationBox
    from proteinMD.environment.periodic_boundary import PeriodicBoundaryConditions, create_cubic_box
    from proteinMD.environment.implicit_solvent import ImplicitSolventModel
    
    # Analysis
    from proteinMD.analysis.rmsd import RMSDAnalyzer
    from proteinMD.analysis.ramachandran import RamachandranAnalyzer
    from proteinMD.analysis.radius_of_gyration import RadiusOfGyrationAnalyzer
    from proteinMD.analysis.secondary_structure import SecondaryStructureAnalyzer
    from proteinMD.analysis.hydrogen_bonds import HydrogenBondAnalyzer
    
    # Sampling
    from proteinMD.sampling.umbrella_sampling import UmbrellaSampling
    from proteinMD.sampling.replica_exchange import ReplicaExchangeMD
    from proteinMD.sampling.steered_md import SteeredMD, SMDParameters
    
    # Visualization
    from proteinMD.visualization.protein_3d import Protein3DVisualizer
    from proteinMD.visualization.trajectory_animation import TrajectoryAnimator
    from proteinMD.visualization.realtime_viewer import RealTimeViewer
    from proteinMD.visualization.energy_dashboard import EnergyPlotDashboard
    
    # Project management
    from proteinMD.project_management.project_manager import ProjectManager, ProjectNotFoundError, TaskNotFoundError
    from proteinMD.project_management.task_manager import TaskManager
    
    # Simulation setup
    from proteinMD.simulation_setup.system_builder import SystemBuilder
    
    # Trajectory analysis
    from proteinMD.analysis.trajectory_analysis import TrajectoryAnalyzer
    
    # Advanced analysis
    from proteinMD.analysis.advanced_analysis import AdvancedAnalysis
    
    # Visualization generator
    from proteinMD.analysis.visualization_generator import VisualizationGenerator
    
    # Force field manager
    from proteinMD.forcefield_management.forcefield_manager import ForceFieldManager
    
    # Database manager
    from proteinMD.database_integration.database_manager import DatabaseManager
    
    # Workflow manager
    from proteinMD.workflow_management.workflow_manager import WorkflowManager
    
    # Plugin manager
    from proteinMD.plugin_management.plugin_manager import PluginManager
    
    # IO handlers
    from proteinMD.io.multi_format_importer import MultiFormatImporter
    from proteinMD.io.large_file_handler import LargeFileHandler
    from proteinMD.io.remote_data_access import download_pdb, fetch_rcsb_metadata, download_remote_file, clear_cache as clear_remote_cache

    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some ProteinMD modules not available: {e}")
    IMPORTS_AVAILABLE = False
    
    # Define placeholder classes for missing modules
    class TaskManager:
        def __init__(self):
            pass
        def add_task(self, project_name, task_name, description, status):
            pass
        def update_task(self, project_name, task_name, description, status):
            pass
        def list_tasks(self, project_name):
            return []

# Configure logging
# Setup logging with error handling for file creation
handlers = [logging.StreamHandler()]
try:
    # Try to create log file, but don't fail if it can't
    handlers.append(logging.FileHandler('proteinmd_cli.log'))
except (PermissionError, FileNotFoundError, OSError):
    # Continue without file logging if there are issues
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger('proteinmd-cli')

# CLI Configuration and Templates
DEFAULT_CONFIG = {
    'simulation': {
        'timestep': 0.002,  # ps
        'temperature': 300.0,  # K
        'n_steps': 10000,
        'output_frequency': 100,
        'trajectory_output': 'trajectory.npz',
        'log_output': 'simulation.log'
    },
    'forcefield': {
        'type': 'amber_ff14sb',
        'water_model': 'tip3p',
        'cutoff': 1.2  # nm
    },
    'environment': {
        'solvent': 'explicit',  # 'explicit', 'implicit', 'vacuum'
        'box_padding': 1.0,  # nm
        'periodic_boundary': True
    },
    'analysis': {
        'rmsd': True,
        'ramachandran': True,
        'radius_of_gyration': True,
        'secondary_structure': True,
        'hydrogen_bonds': True,
        'output_dir': 'analysis_results'
    },
    'visualization': {
        'enabled': True,
        'realtime': False,
        'animation_output': 'animation.gif',
        'plots_output': 'plots'
    }
}

WORKFLOW_TEMPLATES = {
    'protein_folding': {
        'description': 'Standard protein folding simulation',
        'config': {
            'simulation': {
                'n_steps': 50000,
                'temperature': 300.0,
                'timestep': 0.002
            },
            'environment': {'solvent': 'explicit'},
            'analysis': {
                'rmsd': True,
                'radius_of_gyration': True,
                'secondary_structure': True
            }
        }
    },
    'equilibration': {
        'description': 'System equilibration workflow',
        'config': {
            'simulation': {
                'n_steps': 25000,
                'temperature': 300.0,
                'timestep': 0.001
            },
            'environment': {'solvent': 'explicit'},
            'analysis': {
                'rmsd': True,
                'hydrogen_bonds': True
            }
        }
    },
    'free_energy': {
        'description': 'Free energy calculation workflow',
        'config': {
            'simulation': {
                'n_steps': 100000,
                'temperature': 300.0,
                'timestep': 0.002
            },
            'sampling': {
                'method': 'umbrella_sampling',
                'windows': 20,
                'force_constant': 1000.0
            },
            'analysis': {
                'pmf_calculation': True
            }
        }
    },
    'steered_md': {
        'description': 'Steered molecular dynamics simulation',
        'config': {
            'simulation': {
                'n_steps': 50000,
                'temperature': 300.0,
                'timestep': 0.002
            },
            'sampling': {
                'method': 'steered_md',
                'pulling_velocity': 0.005,
                'spring_constant': 1000.0,
                'coordinate_type': 'distance'
            },
            'analysis': {
                'force_curves': True,
                'work_calculation': True
            }
        }
    }
}

class ProteinMDCLI:
    """Main CLI class for ProteinMD operations."""
    
    def __init__(self, workspace: Optional[Union[str, Path]] = None):
        """Initialize the CLI."""
        self.config = DEFAULT_CONFIG.copy()
        try:
            self.workspace = Path(workspace) if workspace else Path.cwd()
        except FileNotFoundError:
            # Fallback to a valid directory if current working directory doesn't exist
            import tempfile
            self.workspace = Path(tempfile.gettempdir())
        self.simulation = None
        self.protein = None
        self.results = {}
        
        # Initialize template manager
        self.template_manager = TemplateManager()
        # Initialize template CLI
        self.template_cli = TemplateCLI()
        
        # Initialize workflow CLI
        self.workflow_cli = WorkflowCLI()
        
        # Initialize database CLI
        self.database_cli = DatabaseCLI()
        
        # Ensure user template directory exists
        self._ensure_user_template_directory()
        
    def run_simulation(self, input_file: str, config_file: Optional[str] = None,
                      template: Optional[str] = None, output_dir: Optional[str] = None, 
                      skip_analysis: bool = False, skip_visualization: bool = False,
                      skip_report: bool = False) -> int:
        """
        Run a complete molecular dynamics simulation.
        
        Args:
            input_file: Input PDB file path
            config_file: Configuration file (JSON/YAML)
            template: Workflow template name
            output_dir: Output directory
            
        Returns:
            Exit code (0 = success, 1 = error)
        """
        try:
            logger.info(f"Starting ProteinMD simulation: {input_file}")
            
            # Load configuration
            if config_file:
                self._load_config(config_file)
            if template:
                self._apply_template(template)
                
            # Resolve absolute path for input file before changing directories
            input_file = Path(input_file).resolve()
                
            # Setup output directory
            if output_dir:
                self.workspace = Path(output_dir)
                self.workspace.mkdir(parents=True, exist_ok=True)
                os.chdir(self.workspace)
                
            # Load protein structure
            logger.info(f"Loading protein structure from {input_file}")
            # Check imports dynamically to allow test patching
            import proteinMD.cli as cli_module
            imports_available = getattr(cli_module, 'IMPORTS_AVAILABLE', True)  # Default to True for testing
            if not imports_available:
                logger.error("ProteinMD modules not available")
                return 1
                
            parser = PDBParser()
            self.protein = parser.parse_file(input_file)
            logger.info(f"Loaded protein with {len(self.protein.atoms)} atoms")
            
            # Setup force field
            logger.info("Setting up force field")
            force_field = self._setup_forcefield()
            
            # Setup environment
            logger.info("Setting up simulation environment")
            environment = self._setup_environment()
            
            # Create simulation system
            logger.info("Creating simulation system")
            
            # Get protein properties for simulation setup
            protein_positions = self.protein.get_positions()
            n_atoms = len(self.protein.atoms)
            
            # Calculate box dimensions from environment setup
            if hasattr(self, 'box_dimensions'):
                box_dims = self.box_dimensions
            else:
                # Estimate box dimensions from protein
                bbox_min = np.min(protein_positions, axis=0)
                bbox_max = np.max(protein_positions, axis=0)
                box_size = bbox_max - bbox_min + 2 * environment.get('box_padding', 1.0)
                box_dims = np.array([box_size.max(), box_size.max(), box_size.max()])
            
            self.simulation = MolecularDynamicsSimulation(
                num_particles=n_atoms,
                box_dimensions=box_dims,
                temperature=self.config['simulation']['temperature'],
                time_step=self.config['simulation']['timestep']
            )
            
            # Configure outputs
            trajectory_file = self.workspace / self.config['simulation']['trajectory_output']
            if hasattr(self.simulation, 'set_trajectory_output'):
                self.simulation.set_trajectory_output(
                    str(trajectory_file),
                    save_interval=self.config['simulation']['output_frequency']
                )
                
            # Setup visualization if enabled
            if self.config['visualization']['enabled'] and self.config['visualization']['realtime']:
                logger.info("Setting up real-time visualization")
                viewer = RealTimeViewer(self.simulation)
                viewer.setup_visualization()
                
            # Run simulation
            logger.info(f"Running simulation for {self.config['simulation']['n_steps']} steps")
            start_time = time.time()
            
            self.simulation.run(self.config['simulation']['n_steps'])
            
            elapsed_time = time.time() - start_time
            logger.info(f"Simulation completed in {elapsed_time:.2f} seconds")
            
            # Run analysis
            if not skip_analysis and any(self.config['analysis'].values()):
                logger.info("Running analysis pipeline")
                self._run_analysis()
                
            # Generate visualization
            if not skip_visualization and self.config['visualization']['enabled']:
                logger.info("Generating visualization")
                self._generate_visualization()
                
            # Generate summary report
            if not skip_report:
                self._generate_report()
            
            logger.info("ProteinMD simulation workflow completed successfully")
            return 0
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            logger.error(traceback.format_exc())
            return 1
            
    def run_analysis(self, trajectory_file: str, structure_file: str,
                    analysis_config: Optional[str] = None, output_dir: Optional[str] = None,
                    skip_analysis: bool = False, skip_visualization: bool = False, skip_report: bool = False) -> int:
        """
        Run analysis on existing trajectory data.
        
        Args:
            trajectory_file: Trajectory file path
            structure_file: Reference structure file
            analysis_config: Analysis configuration file
            output_dir: Output directory
            
        Returns:
            Exit code (0 = success, 1 = error)
        """
        try:
            logger.info(f"Starting ProteinMD analysis: {trajectory_file}")
            
            # Check imports dynamically to allow test patching
            import proteinMD.cli as cli_module
            imports_available = getattr(cli_module, 'IMPORTS_AVAILABLE', True)  # Default to True for testing
            if not imports_available:
                logger.error("ProteinMD modules not available")
                return 1
                
            # Load configuration
            if analysis_config:
                self._load_config(analysis_config)
                
            # Setup output directory
            if output_dir:
                self.workspace = Path(output_dir)
                self.workspace.mkdir(parents=True, exist_ok=True)
                os.chdir(self.workspace)
                
            # Load structure and trajectory
            parser = PDBParser()
            self.protein = parser.parse_file(structure_file)
            
            # Load trajectory data
            import numpy as np
            trajectory_data = np.load(trajectory_file)
            
            # Run analysis pipeline
            if not skip_analysis:
                self._run_analysis(trajectory_data)
            
            # Generate visualization
            if not skip_visualization and self.config['visualization']['enabled']:
                self._generate_visualization(trajectory_data)
                
            # Generate report
            if not skip_report:
                self._generate_report()
            
            logger.info("Analysis workflow completed successfully")
            return 0
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logger.error(traceback.format_exc())
            return 1
            
    def create_template(self, name: str, description: str, config_file: str) -> int:
        """
        Create a new workflow template.
        
        Args:
            name: Template name
            description: Template description
            config_file: Configuration file to use as template
            
        Returns:
            Exit code (0 = success, 1 = error)
        """
        try:
            # Load configuration
            config = self._load_config_file(config_file)
            
            # Create template
            template = {
                'description': description,
                'config': config
            }
            
            # Save template
            template_file = Path.home() / '.proteinmd' / 'templates' / f'{name}.json'
            template_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(template_file, 'w') as f:
                json.dump(template, f, indent=2)
                
            logger.info(f"Template '{name}' created successfully")
            return 0
            
        except Exception as e:
            logger.error(f"Template creation failed: {e}")
            return 1
            
    def list_templates(self) -> int:
        """List available workflow templates using the new template system."""
        try:
            print("\n📋 Available Workflow Templates:")
            print("="*50)
            
            # Get all templates from template manager
            all_templates = self.template_manager.list_templates()
            
            if all_templates:
                print("\n🔧 Available Templates:")
                for template_name in sorted(all_templates.keys()):
                    template = all_templates[template_name]
                    # Handle both template objects and dict-based templates
                    if hasattr(template, 'description'):
                        description = template.description
                        print(f"  • {template_name}: {description}")
                        
                        # Show parameters if any
                        if hasattr(template, 'parameters') and template.parameters:
                            print(f"    Parameters: {', '.join(template.parameters.keys())}")
                    else:
                        # Handle legacy dict format
                        description = template.get('description', 'No description')
                        print(f"  • {template_name}: {description}")
            else:
                print("\n⚠️  No templates found.")
                
            # Also show legacy templates for backward compatibility
            if WORKFLOW_TEMPLATES:
                print("\n🔄 Legacy Templates (for backward compatibility):")
                for name, template in WORKFLOW_TEMPLATES.items():
                    print(f"  • {name}: {template['description']}")
                            
            return 0
            
        except Exception as e:
            logger.error(f"Template listing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 1
            
    def batch_process(self, input_dir: str, pattern: str = "*.pdb",
                     config_file: Optional[str] = None, template: Optional[str] = None,
                     output_dir: Optional[str] = None, parallel: bool = False) -> int:
        """
        Process multiple PDB files in batch mode.
        
        Args:
            input_dir: Directory containing input files
            pattern: File pattern to match
            config_file: Configuration file
            template: Workflow template
            output_dir: Output directory
            parallel: Enable parallel processing
            
        Returns:
            Exit code (0 = success, 1 = error)
        """
        try:
            logger.info(f"Starting batch processing: {input_dir}")
            
            input_path = Path(input_dir)
            if not input_path.exists():
                logger.error(f"Input directory not found: {input_dir}")
                return 1
                
            # Find input files
            input_files = list(input_path.glob(pattern))
            if not input_files:
                logger.error(f"No files found matching pattern: {pattern}")
                return 1
                
            logger.info(f"Found {len(input_files)} files to process")
            
            # Setup output directory
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                output_path = Path.cwd() / 'batch_results'
                output_path.mkdir(parents=True, exist_ok=True)
                
            # Process files
            success_count = 0
            failed_files = []
            
            for i, input_file in enumerate(input_files, 1):
                logger.info(f"Processing file {i}/{len(input_files)}: {input_file.name}")
                
                # Create individual output directory
                file_output_dir = output_path / input_file.stem
                
                # Run simulation
                exit_code = self.run_simulation(
                    str(input_file),
                    config_file=config_file,
                    template=template,
                    output_dir=str(file_output_dir)
                )
                
                if exit_code == 0:
                    success_count += 1
                    logger.info(f"Successfully processed: {input_file.name}")
                else:
                    failed_files.append(input_file.name)
                    logger.error(f"Failed to process: {input_file.name}")
                    
            # Generate batch summary
            self._generate_batch_summary(output_path, input_files, success_count, failed_files)
            
            logger.info(f"Batch processing completed: {success_count}/{len(input_files)} successful")
            return 0 if len(failed_files) == 0 else 1
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            logger.error(traceback.format_exc())
            return 1
            
    def validate_setup(self) -> int:
        """Validate ProteinMD installation and dependencies."""
        try:
            print("\n🔍 ProteinMD Setup Validation")
            print("="*40)
            
            # Check module imports - import IMPORTS_AVAILABLE to get current value
            import proteinMD.cli as cli_module
            imports_available = getattr(cli_module, 'IMPORTS_AVAILABLE', False)
            
            print("\n📦 Module Availability:")
            if imports_available:
                print("  ✅ Core modules: Available")
                print("  ✅ Force fields: Available")
                print("  ✅ Environment: Available")
                print("  ✅ Analysis: Available")
                print("  ✅ Sampling: Available")
                print("  ✅ Visualization: Available")
            else:
                print("  ❌ Some modules missing - check installation")
                return 1
                
            # Check dependencies
            print("\n📚 Dependencies:")
            try:
                import numpy
                print(f"  ✅ NumPy: {numpy.__version__}")
            except ImportError:
                print("  ❌ NumPy: Missing")
                
            try:
                import matplotlib
                print(f"  ✅ Matplotlib: {matplotlib.__version__}")
            except ImportError:
                print("  ❌ Matplotlib: Missing")
                
            try:
                import scipy
                print(f"  ✅ SciPy: {scipy.__version__}")
            except ImportError:
                print("  ❌ SciPy: Missing")
                
            # Check file system
            print("\n📁 File System:")
            print(f"  📍 Current directory: {Path.cwd()}")
            print(f"  📍 Home directory: {Path.home()}")
            
            # Create config directory if needed
            config_dir = Path.home() / '.proteinmd'
            config_dir.mkdir(exist_ok=True)
            print(f"  📍 Config directory: {config_dir}")
            
            print("\n✅ ProteinMD setup validation completed")
            return 0
            
        except Exception as e:
            logger.error(f"Setup validation failed: {e}")
            return 1
            
    def _load_config(self, config_file: str):
        """Load configuration from file."""
        config = self._load_config_file(config_file)
        
        if config is None:
            logger.warning(f"Configuration file {config_file} is empty or invalid")
            return
        
        # Merge with default config
        def merge_dicts(default: dict, override: dict) -> dict:
            result = default.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result
            
        self.config = merge_dicts(self.config, config)
        
    def _load_config_file(self, config_file: str) -> dict:
        """Load configuration file (JSON or YAML)."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        with open(config_path) as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                return json.load(f)
                
    def _apply_template(self, template_name: str):
        """Apply workflow template using the new template system."""
        try:
            # Try to get template from the template manager
            template = self.template_manager.get_template(template_name)
            if template:
                # Generate configuration from template with default parameters
                template_config = template.generate_config()
                logger.info(f"Applied template: {template_name}")
            else:
                # Fallback to old system for backward compatibility
                if template_name in WORKFLOW_TEMPLATES:
                    template_config = WORKFLOW_TEMPLATES[template_name]['config']
                    logger.info(f"Applied legacy template: {template_name}")
                else:
                    # Check user templates (old format)
                    template_file = Path.home() / '.proteinmd' / 'templates' / f'{template_name}.json'
                    if template_file.exists():
                        with open(template_file) as f:
                            template = json.load(f)
                        template_config = template['config']
                        logger.info(f"Applied user template: {template_name}")
                    else:
                        raise ValueError(f"Template not found: {template_name}")
                
            # Merge template config
            def merge_dicts(default: dict, override: dict) -> dict:
                result = default.copy()
                for key, value in override.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = merge_dicts(result[key], value)
                    else:
                        result[key] = value
                return result
                
            self.config = merge_dicts(self.config, template_config)
            
        except Exception as e:
            logger.error(f"Failed to apply template {template_name}: {e}")
            raise
        
    def _setup_forcefield(self):
        """Setup force field based on configuration."""
        # Check if imports are available, using the same approach as validate_setup
        import proteinMD.cli as cli_module
        imports_available = getattr(cli_module, 'IMPORTS_AVAILABLE', True)  # Default to True for testing
        
        if not imports_available:
            raise ValueError("Force field modules not available")
            
        ff_type = self.config['forcefield']['type']
        
        if ff_type == 'amber_ff14sb':
            return AmberFF14SB()
        else:
            raise ValueError(f"Unsupported force field: {ff_type}")
            
    def _setup_environment(self):
        """Setup simulation environment."""
        env_config = self.config['environment']
        
        if env_config['solvent'] == 'explicit':
            # Setup explicit water
            water_model = TIP3PWaterModel()
            solvation_box = WaterSolvationBox(tip3p_model=water_model)
            
            # Get protein positions and estimate box size
            protein_positions = self.protein.get_positions()
            
            # Calculate box dimensions (protein bounding box + padding)
            bbox_min = np.min(protein_positions, axis=0)
            bbox_max = np.max(protein_positions, axis=0)
            box_size = bbox_max - bbox_min + 2 * env_config['box_padding']
            box_dimensions = np.array([box_size.max(), box_size.max(), box_size.max()])  # cubic box
            
            # Solvate protein
            water_data = solvation_box.solvate_protein(
                protein_positions, 
                box_dimensions, 
                padding=env_config['box_padding']
            )
            print(f"Added {water_data['n_molecules']} water molecules")
            
            if env_config['periodic_boundary']:
                # Create periodic box using the calculated box dimensions
                box_length = box_dimensions[0]  # Use cubic box
                periodic_box = create_cubic_box(box_length)
                pbc = PeriodicBoundaryConditions(box=periodic_box)
                print(f"Set up periodic boundary conditions with box length: {box_length:.2f} nm")
                # Store box dimensions for simulation setup
                self.box_dimensions = box_dimensions
                
        elif env_config['solvent'] == 'implicit':
            # Setup implicit solvent
            implicit_solvent = ImplicitSolventModel()
            # Store reference to the implicit solvent model separately (not in config)
            self.implicit_model = implicit_solvent
            # Set default box dimensions for implicit solvent
            box_dimensions = np.array([5.0, 5.0, 5.0])  # Default 5nm box
            
        # Store box dimensions if not already set
        if not hasattr(self, 'box_dimensions'):
            self.box_dimensions = box_dimensions
            
        # Return environment setup
        return env_config
        
    def _run_analysis(self, trajectory_data=None):
        """Run analysis pipeline."""
        analysis_config = self.config['analysis']
        output_dir = Path(analysis_config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get trajectory data
        if trajectory_data is None and hasattr(self.simulation, 'trajectory'):
            trajectory_data = self.simulation.trajectory
        elif trajectory_data is None:
            logger.warning("No trajectory data available for analysis")
            return
            
        # RMSD Analysis
        if analysis_config.get('rmsd', False):
            logger.info("Running RMSD analysis")
            rmsd_analyzer = RMSDAnalyzer(self.protein)
            if hasattr(rmsd_analyzer, 'analyze_trajectory'):
                rmsd_results = rmsd_analyzer.analyze_trajectory(trajectory_data)
                rmsd_analyzer.export_data(str(output_dir / 'rmsd_analysis.csv'))
            
        # Ramachandran Analysis
        if analysis_config.get('ramachandran', False):
            logger.info("Running Ramachandran analysis")
            rama_analyzer = RamachandranAnalyzer()
            if hasattr(rama_analyzer, 'analyze_trajectory'):
                rama_results = rama_analyzer.analyze_trajectory(trajectory_data)
                rama_analyzer.export_data(str(output_dir / 'ramachandran_data.csv'))
            
        # Radius of Gyration
        if analysis_config.get('radius_of_gyration', False):
            logger.info("Running radius of gyration analysis")
            rg_analyzer = RadiusOfGyrationAnalyzer(self.protein)
            if hasattr(rg_analyzer, 'analyze_trajectory'):
                rg_results = rg_analyzer.analyze_trajectory(trajectory_data)
                rg_analyzer.export_data(str(output_dir / 'radius_of_gyration.csv'))
            
        # Secondary Structure
        if analysis_config.get('secondary_structure', False):
            logger.info("Running secondary structure analysis")
            ss_analyzer = SecondaryStructureAnalyzer(self.protein)
            if hasattr(ss_analyzer, 'analyze_trajectory'):
                ss_results = ss_analyzer.analyze_trajectory(trajectory_data)
                ss_analyzer.export_timeline_data(str(output_dir / 'secondary_structure.csv'))
            
        # Hydrogen Bonds
        if analysis_config.get('hydrogen_bonds', False):
            logger.info("Running hydrogen bond analysis")
            hb_analyzer = HydrogenBondAnalyzer(self.protein)
            if hasattr(hb_analyzer, 'analyze_trajectory'):
                hb_results = hb_analyzer.analyze_trajectory(trajectory_data)
                hb_analyzer.export_statistics_csv(str(output_dir / 'hydrogen_bonds.csv'))
            
        self.results['analysis'] = {
            'output_dir': str(output_dir),
            'completed_analyses': [k for k, v in analysis_config.items() if v and k != 'output_dir']
        }
        
    def _generate_visualization(self, trajectory_data=None):
        """Generate visualization outputs."""
        viz_config = self.config['visualization']
        
        if not viz_config.get('enabled', True):
            return
            
        plots_dir = Path(viz_config['plots_output'])
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 3D Structure visualization
        logger.info("Generating 3D structure visualization")
        visualizer = Protein3DVisualizer()
        fig = visualizer.visualize_protein(self.protein)
        fig.savefig(plots_dir / 'protein_structure.png', dpi=300, bbox_inches='tight')
        
        # Trajectory animation
        if trajectory_data is not None:
            logger.info("Generating trajectory animation")
            animator = TrajectoryAnimator(trajectory_data)
            animation_file = plots_dir / viz_config['animation_output']
            animator.save_animation(str(animation_file))
            
        self.results['visualization'] = {
            'plots_dir': str(plots_dir),
            'files_generated': ['protein_structure.png']
        }
        
    def _generate_report(self):
        """Generate summary report."""
        report_file = self.workspace / 'simulation_report.json'
        
        # Create a serializable copy of config
        serializable_config = {}
        for key, value in self.config.items():
            if isinstance(value, dict):
                serializable_config[key] = {}
                for subkey, subvalue in value.items():
                    # Skip non-serializable objects
                    if hasattr(subvalue, '__dict__') and not isinstance(subvalue, (str, int, float, bool, list, dict)):
                        serializable_config[key][subkey] = str(subvalue)
                    else:
                        serializable_config[key][subkey] = subvalue
            elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict)):
                serializable_config[key] = str(value)
            else:
                serializable_config[key] = value
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': serializable_config,
            'protein_info': {
                'n_atoms': len(self.protein.atoms) if self.protein else 0,
                'n_residues': len(self.protein.residues) if self.protein else 0
            },
            'simulation_info': {
                'completed': True,
                'final_step': self.simulation.current_step if self.simulation else 0,
                'final_time': self.simulation.current_time if self.simulation else 0.0
            },
            'results': self.results
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Report saved to: {report_file}")
        
    def _generate_batch_summary(self, output_dir: Path, input_files: List[Path],
                              success_count: int, failed_files: List[str]):
        """Generate batch processing summary."""
        summary_file = output_dir / 'batch_summary.json'
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(input_files),
            'successful': success_count,
            'failed': len(failed_files),
            'success_rate': success_count / len(input_files) * 100,
            'input_files': [f.name for f in input_files],
            'failed_files': failed_files,
            'output_directory': str(output_dir)
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Also create human-readable summary
        summary_text = output_dir / 'batch_summary.txt'
        with open(summary_text, 'w') as f:
            f.write("ProteinMD Batch Processing Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Timestamp: {summary['timestamp']}\n")
            f.write(f"Total files processed: {summary['total_files']}\n")
            f.write(f"Successful: {summary['successful']}\n")
            f.write(f"Failed: {summary['failed']}\n")
            f.write(f"Success rate: {summary['success_rate']:.1f}%\n\n")
            
            if failed_files:
                f.write("Failed files:\n")
                for failed_file in failed_files:
                    f.write(f"  - {failed_file}\n")
                    
        logger.info(f"Batch summary saved to: {summary_file}")
    
    def _ensure_user_template_directory(self):
        """Ensure the user template directory exists."""
        user_templates_dir = Path.home() / '.proteinmd' / 'templates'
        user_templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a README if it doesn't exist
        readme_file = user_templates_dir / 'README.md'
        if not readme_file.exists():
            readme_content = """# ProteinMD User Templates

This directory contains your custom simulation templates.

## Template Format
Templates can be in JSON or YAML format and should follow the ProteinMD template schema.

## Example Template Structure
```json
{
    "name": "my_template",
    "description": "Custom simulation template",
    "parameters": {
        "temperature": {
            "type": "float",
            "default": 300.0,
            "range": [250.0, 400.0],
            "description": "Simulation temperature in Kelvin"
        }
    },
    "configuration": {
        "simulation": {
            "n_steps": 50000,
            "temperature": "{{temperature}}",
            "timestep": 0.002
        }
    }
}
```

For more information, see the ProteinMD documentation.
"""
            readme_file.write_text(readme_content)
        
        logger.debug(f"User template directory ensured at: {user_templates_dir}")

    # --- Remote Data Access Handlers ---
    def _handle_remote_download_pdb(self, args):
        """Handles the 'remote download-pdb' command."""
        try:
            logger.info(f"Downloading PDB ID: {args.pdb_id}, Overwrite: {args.overwrite}")
            file_path = download_pdb(args.pdb_id, overwrite=args.overwrite)
            print(f"PDB file for {args.pdb_id} downloaded to: {file_path}")
            return 0
        except FileExistsError as e:
            logger.error(f"Error downloading PDB {args.pdb_id}: {e}. Use --overwrite to replace.")
            print(f"Error: {e}. Use --overwrite to replace the existing file.", file=sys.stderr)
            return 1
        except Exception as e:
            logger.error(f"Failed to download PDB {args.pdb_id}: {e}")
            logger.error(traceback.format_exc())
            print(f"Error: {e}", file=sys.stderr)
            return 1

    def _handle_remote_fetch_metadata(self, args):
        """Handles the 'remote fetch-metadata' command."""
        try:
            logger.info(f"Fetching metadata for PDB ID: {args.pdb_id}")
            metadata = fetch_rcsb_metadata(args.pdb_id)
            print(json.dumps(metadata, indent=2))
            return 0
        except Exception as e:
            logger.error(f"Failed to fetch metadata for PDB {args.pdb_id}: {e}")
            logger.error(traceback.format_exc())
            print(f"Error: {e}", file=sys.stderr)
            return 1

    def _handle_remote_download_file(self, args):
        """Handles the 'remote download-file' command."""
        try:
            logger.info(f"Downloading remote file: {args.url}, Overwrite: {args.overwrite}")
            file_path = download_remote_file(args.url, overwrite=args.overwrite)
            print(f"File downloaded from {args.url} to: {file_path}")
            return 0
        except FileExistsError as e:
            logger.error(f"Error downloading file {args.url}: {e}. Use --overwrite to replace.")
            print(f"Error: {e}. Use --overwrite to replace the existing file.", file=sys.stderr)
            return 1
        except Exception as e:
            logger.error(f"Failed to download file {args.url}: {e}")
            logger.error(traceback.format_exc())
            print(f"Error: {e}", file=sys.stderr)
            return 1

    def _handle_remote_clear_cache(self, args):
        """Handles the 'remote clear-cache' command."""
        try:
            logger.info("Clearing remote data cache.")
            clear_remote_cache() # Uses the aliased import
            print("Remote data cache cleared.")
            return 0
        except Exception as e:
            logger.error(f"Failed to clear remote data cache: {e}")
            logger.error(traceback.format_exc())
            print(f"Error: {e}", file=sys.stderr)
            return 1
    # --- End Remote Data Access Handlers ---

    def _setup_argument_parser(self):
        parser = argparse.ArgumentParser(
            description="ProteinMD - Comprehensive Molecular Dynamics Simulation CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic simulation
  proteinmd simulate protein.pdb
  
  # Simulation with template
  proteinmd simulate protein.pdb --template protein_folding
  
  # Simulation with custom config
  proteinmd simulate protein.pdb --config my_config.json
  
  # Analysis only
  proteinmd analyze trajectory.npz protein.pdb
  
  # Batch processing
  proteinmd batch-process ./structures/ --template equilibration
  
  # Create sample config
  proteinmd sample-config
  
  # List templates
  proteinmd list-templates

  # Remote data access
  proteinmd remote download-pdb 1ehz
  proteinmd remote fetch-metadata 1ehz
  proteinmd remote download-file https://example.com/somefile.txt
  proteinmd remote clear-cache
"""
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

        # Simulate command
        simulate_parser = subparsers.add_parser('simulate', help='Run a molecular dynamics simulation')
        simulate_parser.add_argument('input_file', help='Input PDB file path')
        simulate_parser.add_argument('--config', help='Configuration file (JSON/YAML)')
        simulate_parser.add_argument('--template', help='Workflow template name')
        simulate_parser.add_argument('--output-dir', help='Output directory')
        
        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Run analysis on trajectory data')
        analyze_parser.add_argument('trajectory_file', help='Trajectory file path')
        analyze_parser.add_argument('structure_file', help='Reference structure file')
        analyze_parser.add_argument('--config', help='Analysis configuration file')
        analyze_parser.add_argument('--output-dir', help='Output directory')
        
        # Create template command
        create_template_parser = subparsers.add_parser('create-template', help='Create a new workflow template')
        create_template_parser.add_argument('name', help='Template name')
        create_template_parser.add_argument('description', help='Template description')
        create_template_parser.add_argument('config_file', help='Configuration file to use as template')
        
        # List templates command
        list_templates_parser = subparsers.add_parser('list-templates', help='List available workflow templates')
        
        # Batch process command
        batch_parser = subparsers.add_parser('batch-process', help='Process multiple PDB files in batch mode')
        batch_parser.add_argument('input_dir', help='Directory containing input files')
        batch_parser.add_argument('--pattern', default='*.pdb', help='File pattern to match (default: *.pdb)')
        batch_parser.add_argument('--config', help='Configuration file')
        batch_parser.add_argument('--template', help='Workflow template name')
        batch_parser.add_argument('--output-dir', help='Output directory')
        batch_parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
        
        # Validate setup command
        validate_parser = subparsers.add_parser('validate', help='Validate ProteinMD installation and dependencies')
        
        # Sample config command
        sample_config_parser = subparsers.add_parser('sample-config', help='Generate a sample configuration file')
        sample_config_parser.add_argument('--output', default='proteinmd_config.json', help='Output file path')

        # Template CLI commands (delegated)
        template_cli_parser = subparsers.add_parser('template', help='Manage workflow templates')
        self.template_cli.setup_subparsers(template_cli_parser)

        # Workflow CLI commands (delegated)
        workflow_cli_parser = subparsers.add_parser('workflow', help='Manage and run automated workflows')
        self.workflow_cli.setup_subparsers(workflow_cli_parser)
        
        # Database CLI commands (delegated)
        database_cli_parser = subparsers.add_parser('database', help='Manage database integration')
        self.database_cli.setup_subparsers(database_cli_parser)

        # Project management commands
        project_parser = subparsers.add_parser('project', help='Manage ProteinMD projects')
        project_subparsers = project_parser.add_subparsers(dest='project_command', help='Project commands', required=True)

        project_init_parser = project_subparsers.add_parser('init', help='Initialize a new project')
        project_init_parser.add_argument('name', help='Project name')
        project_init_parser.add_argument('--description', help='Project description', default="")

        project_list_parser = project_subparsers.add_parser('list', help='List all projects')
        
        project_info_parser = project_subparsers.add_parser('info', help='Display project information')
        project_info_parser.add_argument('name', help='Project name')

        project_task_parser = project_subparsers.add_parser('task', help='Manage project tasks')
        task_subparsers = project_task_parser.add_subparsers(dest='task_command', help='Task commands', required=True)
        
        task_add_parser = task_subparsers.add_parser('add', help='Add a new task to a project')
        task_add_parser.add_argument('project_name', help='Name of the project')
        task_add_parser.add_argument('task_name', help='Name of the task')
        task_add_parser.add_argument('--description', help='Task description', default="")
        task_add_parser.add_argument('--status', help='Task status', default="pending")

        task_update_parser = task_subparsers.add_parser('update', help='Update an existing task')
        task_update_parser.add_argument('project_name', help='Name of the project')
        task_update_parser.add_argument('task_name', help='Name of the task')
        task_update_parser.add_argument('--description', help='New task description')
        task_update_parser.add_argument('--status', help='New task status')

        task_list_parser = task_subparsers.add_parser('list', help='List tasks in a project')
        task_list_parser.add_argument('project_name', help='Name of the project')

        # Remote data access commands
        remote_parser = subparsers.add_parser('remote', help='Access remote data (PDB, RCSB, general files)')
        remote_subparsers = remote_parser.add_subparsers(dest='remote_command', help='Remote data commands', required=True)

        # proteinmd remote download-pdb <pdb_id> [--overwrite]
        remote_download_pdb_parser = remote_subparsers.add_parser('download-pdb', help='Download a PDB file from RCSB PDB.')
        remote_download_pdb_parser.add_argument('pdb_id', help='The PDB ID (e.g., 1EHZ).')
        remote_download_pdb_parser.add_argument('--overwrite', action='store_true', help='Overwrite the file if it already exists in the cache.')

        # proteinmd remote fetch-metadata <pdb_id>
        remote_fetch_metadata_parser = remote_subparsers.add_parser('fetch-metadata', help='Fetch metadata for a PDB ID from RCSB PDB.')
        remote_fetch_metadata_parser.add_argument('pdb_id', help='The PDB ID (e.g., 1EHZ).')

        # proteinmd remote download-file <url> [--overwrite]
        remote_download_file_parser = remote_subparsers.add_parser('download-file', help='Download a generic file from a URL (HTTP/FTP).')
        remote_download_file_parser.add_argument('url', help='The full URL of the file to download.')
        remote_download_file_parser.add_argument('--overwrite', action='store_true', help='Overwrite the file if it already exists in the cache.')
        
        # proteinmd remote clear-cache
        remote_clear_cache_parser = remote_subparsers.add_parser('clear-cache', help='Clear the local cache for remote data.')

        return parser

    def _handle_project_command(self, args):
        try:
            if args.project_command == 'init':
                # Initialize a new project
                logger.info(f"Initializing new project: {args.name}")
                project_manager = ProjectManager()
                project_manager.init_project(args.name, args.description)
                logger.info(f"Project '{args.name}' initialized successfully")
                print(f"Project '{args.name}' initialized successfully")
                return 0
            elif args.project_command == 'list':
                # List all projects
                logger.info("Listing all projects")
                project_manager = ProjectManager()
                projects = project_manager.list_projects()
                
                if projects:
                    print("\n📂 Available Projects:")
                    print("="*50)
                    for project in projects:
                        print(f"  • {project['name']}: {project.get('description', 'No description')}")
                else:
                    print("\n⚠️  No projects found.")
                    
                return 0
            elif args.project_command == 'info':
                # Display project information
                logger.info(f"Fetching info for project: {args.name}")
                project_manager = ProjectManager()
                project_info = project_manager.get_project_info(args.name)
                
                if project_info:
                    print("\n📊 Project Information:")
                    print("="*50)
                    print(f"  Name: {project_info['name']}")
                    print(f"  Description: {project_info.get('description', 'N/A')}")
                    print(f"  Created: {project_info['created_at']}")
                    print(f"  Modified: {project_info['modified_at']}")
                else:
                    print(f"⚠️  Project '{args.name}' not found.")
                    
                return 0
            elif args.project_command == 'task':
                # Delegate to task subcommands
                return self._handle_task_command(args)
            else:
                logger.error(f"Unknown project command: {args.project_command}")
                return 1
        except Exception as e:
            logger.error(f"Error handling project command: {e}")
            return 1

    def _handle_task_command(self, args):
        try:
            # Check if TaskManager is available
            if not IMPORTS_AVAILABLE:
                print("Error: Task management functionality not available. Missing project management modules.")
                return 1
                
            task_manager = TaskManager()
            
            if args.task_command == 'add':
                # Add a new task to a project
                logger.info(f"Adding new task '{args.task_name}' to project '{args.project_name}'")
                task_manager.add_task(args.project_name, args.task_name, args.description, args.status)
                logger.info(f"Task '{args.task_name}' added successfully")
                print(f"Task '{args.task_name}' added successfully")
                return 0
            elif args.task_command == 'update':
                # Update an existing task
                logger.info(f"Updating task '{args.task_name}' in project '{args.project_name}'")
                task_manager.update_task(args.project_name, args.task_name, args.description, args.status)
                logger.info(f"Task '{args.task_name}' updated successfully")
                print(f"Task '{args.task_name}' updated successfully")
                return 0
            elif args.task_command == 'list':
                # List tasks in a project
                logger.info(f"Listing tasks for project '{args.project_name}'")
                tasks = task_manager.list_tasks(args.project_name)
                
                if tasks:
                    print("\n📝 Project Tasks:")
                    print("="*50)
                    for task in tasks:
                        print(f"  • {task['name']} (Status: {task['status']})")
                else:
                    print(f"\n⚠️  No tasks found for project '{args.project_name}'.")
                    
                return 0
            else:
                logger.error(f"Unknown task command: {args.task_command}")
                return 1
        except Exception as e:
            logger.error(f"Error handling task command: {e}")
            return 1

def create_sample_config(output_file: str = "proteinmd_config.json") -> int:
    """Create a sample configuration file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"Sample configuration saved to: {output_file}")
        return 0
    except Exception as e:
        print(f"Error creating config file: {e}")
        return 1

def create_completion_script() -> int:
    """Create bash completion script."""
    try:
        completion_script = '''#!/bin/bash
# ProteinMD CLI bash completion

_proteinmd_complete() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Main commands
    if [[ ${COMP_CWORD} == 1 ]]; then
        opts="simulate analyze batch-process create-template list-templates"
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
    
    # Command-specific completions
    case "${prev}" in
        --template)
            opts="protein_folding equilibration free_energy steered_md"
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            return 0
            ;;
        --config|--analysis-config)
            COMPREPLY=( $(compgen -f -X '!*.@(json|yaml|yml)' -- ${cur}) )
            return 0
            ;;
        --input|--structure)
            COMPREPLY=( $(compgen -f -X '!*.pdb' -- ${cur}) )
            return 0
            ;;
        --trajectory)
            COMPREPLY=( $(compgen -f -X '!*.npz' -- ${cur}) )
            return 0
            ;;
        --input-dir|--output-dir)
            COMPREPLY=( $(compgen -d -- ${cur}) )
            return 0
            ;;
    esac
    
    # General file completion
    COMPREPLY=( $(compgen -f -- ${cur}) )
}

complete -F _proteinmd_complete proteinmd
'''
        
        completion_file = Path.home() / '.proteinmd' / 'proteinmd_completion.bash'
        completion_file.parent.mkdir(exist_ok=True)
        
        with open(completion_file, 'w') as f:
            f.write(completion_script)
            
        print(f"Bash completion script created: {completion_file}")
        print("To enable, add this to your ~/.bashrc:")
        print(f"source {completion_file}")
        return 0
        
    except Exception as e:
        print(f"Error creating completion script: {e}")
        return 1

def main():
    """Main CLI entry point."""
    try:
        parser = argparse.ArgumentParser(
            description="ProteinMD - Comprehensive Molecular Dynamics Simulation CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic simulation
  proteinmd simulate protein.pdb
  
  # Simulation with template
  proteinmd simulate protein.pdb --template protein_folding
  
  # Simulation with custom config
  proteinmd simulate protein.pdb --config my_config.json
  
  # Analysis only
  proteinmd analyze trajectory.npz protein.pdb
  
  # Batch processing
  proteinmd batch-process ./structures/ --template equilibration
  
  # Create sample config
  proteinmd sample-config
  
  # List templates
  proteinmd list-templates

  # Remote data access
  proteinmd remote download-pdb 1ehz
  proteinmd remote fetch-metadata 1ehz
  proteinmd remote download-file https://example.com/somefile.txt
  proteinmd remote clear-cache
"""
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands', required=False)
    
        # Simulate command
        sim_parser = subparsers.add_parser('simulate', help='Run a molecular dynamics simulation')
        sim_parser.add_argument('input', help='Input PDB file')
        sim_parser.add_argument('--config', help='Configuration file (JSON/YAML)')
        sim_parser.add_argument('--template', help='Workflow template name')
        sim_parser.add_argument('--output-dir', help='Output directory')
        sim_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Run analysis on trajectory')
        analyze_parser.add_argument('trajectory', help='Trajectory file (.npz)')
        analyze_parser.add_argument('structure', help='Reference structure file (.pdb)')
        analyze_parser.add_argument('--analysis-config', help='Analysis configuration file')
        analyze_parser.add_argument('--output-dir', help='Output directory')
        analyze_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
        # Batch process command
        batch_parser = subparsers.add_parser('batch-process', help='Process multiple files')
        batch_parser.add_argument('input_dir', help='Input directory')
        batch_parser.add_argument('--pattern', default='*.pdb', help='File pattern (default: *.pdb)')
        batch_parser.add_argument('--config', help='Configuration file')
        batch_parser.add_argument('--template', help='Workflow template')
        batch_parser.add_argument('--output-dir', help='Output directory')
        batch_parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
        batch_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
        # Create template command
        template_parser = subparsers.add_parser('create-template', help='Create workflow template')
        template_parser.add_argument('name', help='Template name')
        template_parser.add_argument('description', help='Template description')
        template_parser.add_argument('config_file', help='Configuration file to use as template')
    
        list_parser = subparsers.add_parser('list-templates', help='List available templates')
    
        # Advanced template commands
        show_template_parser = subparsers.add_parser('show-template', help='Show template details')
        show_template_parser.add_argument('name', help='Template name')
    
        validate_template_parser = subparsers.add_parser('validate-template', help='Validate template')
        validate_template_parser.add_argument('file', help='Template file to validate')
    
        export_template_parser = subparsers.add_parser('export-template', help='Export template')
        export_template_parser.add_argument('name', help='Template name')
        export_template_parser.add_argument('--format', choices=['json', 'yaml'], default='json', help='Export format')
        export_template_parser.add_argument('--output', help='Output file (default: stdout)')
    
        import_template_parser = subparsers.add_parser('import-template', help='Import template')
        import_template_parser.add_argument('file', help='Template file to import')
        import_template_parser.add_argument('--name', help='Template name (override)')
    
        generate_config_parser = subparsers.add_parser('generate-config', help='Generate configuration from template')
        generate_config_parser.add_argument('template', help='Template name')
        generate_config_parser.add_argument('--parameters', help='Parameter file (JSON/YAML)')
        generate_config_parser.add_argument('--output', help='Output file (default: stdout)')
    
        template_stats_parser = subparsers.add_parser('template-stats', help='Show template statistics')
    
        # Workflow automation commands
        workflow_parser = subparsers.add_parser('workflow', help='Workflow automation commands')
        workflow_subparsers = workflow_parser.add_subparsers(dest='workflow_command', help='Workflow operations')
    
        # Create workflow command
        workflow_create = workflow_subparsers.add_parser('create', help='Create new workflow')
        workflow_create.add_argument('output_file', help='Output workflow file')
        workflow_create.add_argument('--template', help='Workflow template to use')
        workflow_create.add_argument('--name', help='Workflow name')
        workflow_create.add_argument('--description', help='Workflow description')
        workflow_create.add_argument('--author', help='Workflow author')
    
        # Run workflow command
        workflow_run = workflow_subparsers.add_parser('run', help='Run workflow')
        workflow_run.add_argument('workflow_file', help='Workflow definition file')
        workflow_run.add_argument('--input', help='Input file (e.g., PDB file)')
        workflow_run.add_argument('--output-dir', help='Output directory')
        workflow_run.add_argument('--scheduler', help='Job scheduler to use', choices=['local', 'slurm', 'pbs', 'sge'])
        workflow_run.add_argument('--parameters', help='Parameter overrides (JSON)')
        workflow_run.add_argument('--dry-run', action='store_true', help='Validate without running')
        workflow_run.add_argument('--monitor', action='store_true', help='Monitor execution progress')
    
        # Status command
        workflow_status = workflow_subparsers.add_parser('status', help='Check workflow status')
        workflow_status.add_argument('output_dir', help='Workflow output directory')
        workflow_status.add_argument('--detailed', action='store_true', help='Show detailed status')
        workflow_status.add_argument('--refresh', type=int, default=0, help='Auto-refresh interval (seconds)')
    
        # Validate workflow command
        workflow_validate = workflow_subparsers.add_parser('validate', help='Validate workflow')
        workflow_validate.add_argument('workflow_file', help='Workflow definition file')
        workflow_validate.add_argument('--verbose', action='store_true', help='Verbose validation')
    
        # List workflow templates command
        workflow_list = workflow_subparsers.add_parser('list-templates', help='List workflow templates')
        workflow_list.add_argument('--detailed', action='store_true', help='Show detailed information')
    
        # Stop workflow command
        workflow_stop = workflow_subparsers.add_parser('stop', help='Stop running workflow')
        workflow_stop.add_argument('output_dir', help='Workflow output directory')
    
        # Generate workflow report command
        workflow_report = workflow_subparsers.add_parser('report', help='Generate workflow report')
        workflow_report.add_argument('output_dir', help='Workflow output directory')
        workflow_report.add_argument('--format', choices=['html', 'json'], default='html', help='Report format')
    
        # Workflow examples command
        workflow_examples = workflow_subparsers.add_parser('examples', help='Show workflow examples')
        workflow_examples.add_argument('--create', help='Create example workflow file')
    
        # Database commands
        database_parser = subparsers.add_parser('database', help='Database management commands')
        database_subparsers = database_parser.add_subparsers(dest='database_command', help='Database operations')
    
        # Initialize database command
        db_init = database_subparsers.add_parser('init', help='Initialize database')
        db_init.add_argument('--config', help='Database configuration file')
        db_init.add_argument('--type', choices=['sqlite', 'postgresql'], default='sqlite', help='Database type')
        db_init.add_argument('--path', help='Database path (SQLite) or connection string')
    
        # Store simulation command
        db_store = database_subparsers.add_parser('store', help='Store simulation record')
        db_store.add_argument('simulation_dir', help='Simulation output directory')
        db_store.add_argument('--config', help='Database configuration file')
    
        # List simulations command
        db_list = database_subparsers.add_parser('list', help='List stored simulations')
        db_list.add_argument('--config', help='Database configuration file')
        db_list.add_argument('--status', help='Filter by status')
        db_list.add_argument('--user', help='Filter by user')
        db_list.add_argument('--project', help='Filter by project')
        db_list.add_argument('--limit', type=int, default=20, help='Maximum results')
        db_list.add_argument('--offset', type=int, default=0, help='Results offset for pagination')
        db_list.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
    
        # Search simulations command
        db_search = database_subparsers.add_parser('search', help='Search simulations')
        db_search.add_argument('query', help='Search query')
        db_search.add_argument('--config', help='Database configuration file')
        db_search.add_argument('--limit', type=int, default=20, help='Maximum results')
        db_search.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
    
        # Show simulation command
        db_show = database_subparsers.add_parser('show', help='Show simulation details')
        db_show.add_argument('simulation_id', help='Simulation ID')
        db_show.add_argument('--config', help='Database configuration file')
        db_show.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
    
        # Delete simulation command
        db_delete = database_subparsers.add_parser('delete', help='Delete simulation record')
        db_delete.add_argument('simulation_id', help='Simulation ID')
        db_delete.add_argument('--config', help='Database configuration file')
        db_delete.add_argument('--force', action='store_true', help='Skip confirmation')
    
        # Export database command
        db_export = database_subparsers.add_parser('export', help='Export database')
        db_export.add_argument('output_file', help='Output file')
        db_export.add_argument('--config', help='Database configuration file')
        db_export.add_argument('--format', choices=['json', 'csv'], default='json', help='Export format')
    
        # Import database command
        db_import = database_subparsers.add_parser('import', help='Import database')
        db_import.add_argument('input_file', help='Input file')
        db_import.add_argument('--config', help='Database configuration file')
        db_import.add_argument('--format', choices=['json', 'csv'], default='json', help='Import format')
    
        # Backup database command
        db_backup = database_subparsers.add_parser('backup', help='Create database backup')
        db_backup.add_argument('--config', help='Database configuration file')
        db_backup.add_argument('--output-dir', help='Backup output directory')
        db_backup.add_argument('--compress', action='store_true', help='Compress backup')
    
        # Restore database command
        db_restore = database_subparsers.add_parser('restore', help='Restore database from backup')
        db_restore.add_argument('backup_file', help='Backup file to restore')
        db_restore.add_argument('--config', help='Database configuration file')
        db_restore.add_argument('--force', action='store_true', help='Overwrite existing data')
    
        # Database statistics command
        db_stats = database_subparsers.add_parser('stats', help='Show database statistics')
        db_stats.add_argument('--config', help='Database configuration file')
        db_stats.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
    
        # Health check command
        db_health = database_subparsers.add_parser('health', help='Check database health')
        db_health.add_argument('--config', help='Database configuration file')
        db_health.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
    
        # Utility commands
        validate_parser = subparsers.add_parser('validate-setup', help='Validate installation')
        sample_parser = subparsers.add_parser('sample-config', help='Create sample configuration')
        sample_parser.add_argument('--output', default='proteinmd_config.json', help='Output file')
    
        completion_parser = subparsers.add_parser('bash-completion', help='Create bash completion script')
    
        args = parser.parse_args()
    
        # Handle commands that don't need CLI instance
        if args.command == 'sample-config':
            return create_sample_config(args.output)
        elif args.command == 'bash-completion':
            return create_completion_script()
        elif args.command is None:
            parser.print_help()
            return 1
            
        # Set up logging level
        if hasattr(args, 'verbose') and args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            
        # Create CLI instance and dispatch commands
        cli = ProteinMDCLI()
        
        exit_code = 0
        if args.command == 'simulate':
            exit_code = cli.run_simulation(
                args.input,
                config_file=args.config,
                template=args.template,
                output_dir=args.output_dir
            )
        elif args.command == 'analyze':
            exit_code = cli.run_analysis(
                args.trajectory,
                args.structure,
                analysis_config=args.analysis_config,
                output_dir=args.output_dir
            )
        elif args.command == 'create-template':
            exit_code = cli.create_template(args.name, args.description, args.config_file)
        elif args.command == 'list-templates':
            exit_code = cli.list_templates()
        elif args.command == 'batch-process':
            exit_code = cli.batch_process(
                args.input_dir,
                pattern=args.pattern,
                config_file=args.config,
                template=args.template,
                output_dir=args.output_dir,
                parallel=args.parallel
            )
        elif args.command == 'validate-setup':
            exit_code = cli.validate_setup()
        elif args.command == 'sample-config':
            config_content = json.dumps(DEFAULT_CONFIG, indent=2)
            output_file = Path(args.output)
            output_file.write_text(config_content)
            print(f"Sample configuration written to {output_file}")
            exit_code = 0
        elif args.command == 'template': # Delegated to TemplateCLI
            exit_code = cli.template_cli.handle_command(args)
        elif args.command == 'workflow': # Delegated to WorkflowCLI
            exit_code = cli.workflow_cli.handle_command(args)
        elif args.command == 'database': # Delegated to DatabaseCLI
            exit_code = cli.database_cli.handle_command(args)
        elif args.command == 'project':
            exit_code = cli._handle_project_command(args)
        elif args.command == 'remote':
            if args.remote_command == 'download-pdb':
                exit_code = cli._handle_remote_download_pdb(args)
            elif args.remote_command == 'fetch-metadata':
                exit_code = cli._handle_remote_fetch_metadata(args)
            elif args.remote_command == 'download-file':
                exit_code = cli._handle_remote_download_file(args)
            elif args.remote_command == 'clear-cache':
                exit_code = cli._handle_remote_clear_cache(args)
            else:
                logger.error("No remote data subcommand specified.")
                # Attempt to print help for the 'remote' subcommand group
                # This requires a bit of a workaround if subparsers are not easily accessible
                # For simplicity, printing general help or a specific message
                parser.parse_args(['remote', '-h']) # This should trigger help for remote
                exit_code = 1
        else:
            parser.print_help()
            exit_code = 1
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

    return exit_code

if __name__ == "__main__":
    main()