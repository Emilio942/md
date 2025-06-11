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
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some ProteinMD modules not available: {e}")
    IMPORTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('proteinmd_cli.log')
    ]
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
        self.workspace = Path(workspace) if workspace else Path.cwd()
        self.simulation = None
        self.protein = None
        self.results = {}
        
    def run_simulation(self, input_file: str, config_file: Optional[str] = None,
                      template: Optional[str] = None, output_dir: Optional[str] = None) -> int:
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
            if not IMPORTS_AVAILABLE:
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
            if any(self.config['analysis'].values()):
                logger.info("Running analysis pipeline")
                self._run_analysis()
                
            # Generate visualization
            if self.config['visualization']['enabled']:
                logger.info("Generating visualization")
                self._generate_visualization()
                
            # Generate summary report
            self._generate_report()
            
            logger.info("ProteinMD simulation workflow completed successfully")
            return 0
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            logger.error(traceback.format_exc())
            return 1
            
    def run_analysis(self, trajectory_file: str, structure_file: str,
                    analysis_config: Optional[str] = None, output_dir: Optional[str] = None) -> int:
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
            
            if not IMPORTS_AVAILABLE:
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
            self._run_analysis(trajectory_data)
            
            # Generate visualization
            if self.config['visualization']['enabled']:
                self._generate_visualization(trajectory_data)
                
            # Generate report
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
        """List available workflow templates."""
        try:
            print("\n📋 Available Workflow Templates:")
            print("="*50)
            
            # Built-in templates
            print("\n🔧 Built-in Templates:")
            for name, template in WORKFLOW_TEMPLATES.items():
                print(f"  • {name}: {template['description']}")
                
            # User templates
            user_templates_dir = Path.home() / '.proteinmd' / 'templates'
            if user_templates_dir.exists():
                user_templates = list(user_templates_dir.glob('*.json'))
                if user_templates:
                    print("\n👤 User Templates:")
                    for template_file in user_templates:
                        try:
                            with open(template_file) as f:
                                template = json.load(f)
                            print(f"  • {template_file.stem}: {template.get('description', 'No description')}")
                        except Exception as e:
                            print(f"  • {template_file.stem}: (Error loading: {e})")
                            
            return 0
            
        except Exception as e:
            logger.error(f"Template listing failed: {e}")
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
            
            # Check module imports
            print("\n📦 Module Availability:")
            if IMPORTS_AVAILABLE:
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
        """Apply workflow template."""
        # Check built-in templates
        if template_name in WORKFLOW_TEMPLATES:
            template_config = WORKFLOW_TEMPLATES[template_name]['config']
            logger.info(f"Applied template: {template_name}")
        else:
            # Check user templates
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
        
    def _setup_forcefield(self):
        """Setup force field based on configuration."""
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
            implicit_solvent.apply_to_protein(self.protein)
            
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
            rama_analyzer = RamachandranAnalyzer(self.protein)
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
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config,
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
        opts="simulate analyze batch-process create-template list-templates validate-setup sample-config"
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
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Run molecular dynamics simulation')
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
    
    # Template commands
    template_parser = subparsers.add_parser('create-template', help='Create workflow template')
    template_parser.add_argument('name', help='Template name')
    template_parser.add_argument('description', help='Template description')
    template_parser.add_argument('config_file', help='Configuration file to use as template')
    
    list_parser = subparsers.add_parser('list-templates', help='List available templates')
    
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
    
    try:
        if args.command == 'simulate':
            return cli.run_simulation(
                args.input,
                config_file=args.config,
                template=args.template,
                output_dir=args.output_dir
            )
        elif args.command == 'analyze':
            return cli.run_analysis(
                args.trajectory,
                args.structure,
                analysis_config=args.analysis_config,
                output_dir=args.output_dir
            )
        elif args.command == 'batch-process':
            return cli.batch_process(
                args.input_dir,
                pattern=args.pattern,
                config_file=args.config,
                template=args.template,
                output_dir=args.output_dir,
                parallel=args.parallel
            )
        elif args.command == 'create-template':
            return cli.create_template(args.name, args.description, args.config_file)
        elif args.command == 'list-templates':
            return cli.list_templates()
        elif args.command == 'validate-setup':
            return cli.validate_setup()
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
