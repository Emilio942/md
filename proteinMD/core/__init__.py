"""
Core module for molecular dynamics simulations.

This module contains the fundamental classes and functions for MD simulations,
including system initialization, force calculations, and integration algorithms.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Error Handling & Logging System Integration
# ==========================================

# Import error handling components
try:
    from .exceptions import *
    from .logging_system import ProteinMDLogger, setup_logging, get_logger
    from .logging_config import LoggingConfig, ConfigurationManager, get_template_config
    from .error_integration import (
        ModuleIntegrator, integrate_proteinmd_package, 
        setup_comprehensive_logging, safe_operation, 
        with_error_recovery, exception_context, ValidationMixin
    )
    
    # Error handling system availability flag
    ERROR_HANDLING_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Error handling system not fully available: {e}")
    ERROR_HANDLING_AVAILABLE = False


class MDSystem:
    """
    Main class representing a molecular dynamics system.
    
    This class manages the entire simulation system, including particles,
    force field parameters, and simulation conditions.
    """
    
    def __init__(self, 
                 name: str = "MD_System",
                 temperature: float = 300.0,  # K
                 pressure: float = 1.0,       # atm
                 time_step: float = 0.002,    # ps
                 box_size: Optional[np.ndarray] = None):
        """
        Initialize a molecular dynamics system.
        
        Parameters
        ----------
        name : str
            Name of the simulation system
        temperature : float
            Target temperature in Kelvin
        pressure : float
            Target pressure in atmospheres
        time_step : float
            Integration time step in picoseconds
        box_size : np.ndarray, optional
            Simulation box dimensions [x, y, z] in nanometers
        """
        self.name = name
        self.temperature = temperature
        self.pressure = pressure
        self.time_step = time_step
        
        # Initialize box size with default if not provided
        if box_size is None:
            self.box_size = np.array([10.0, 10.0, 10.0])  # nm
        else:
            self.box_size = box_size
        
        # Initialize empty containers for particles and molecules
        self.particles = []
        self.molecules = []
        self.proteins = []
        
        # Simulation state
        self.current_step = 0
        self.total_steps = 0
        self.elapsed_time = 0.0  # ps
        
        # Force field and integrator (to be set later)
        self.force_field = None
        self.integrator = None
        
        logger.info(f"Initialized MD system: {self.name}")
        logger.info(f"Temperature: {self.temperature} K, Pressure: {self.pressure} atm")
        logger.info(f"Box size: {self.box_size} nm, Time step: {self.time_step} ps")
    
    def add_protein(self, protein):
        """Add a protein to the simulation system."""
        self.proteins.append(protein)
        self.particles.extend(protein.atoms)
        self.molecules.append(protein)
        logger.info(f"Added protein with {len(protein.atoms)} atoms to system")
    
    def add_molecules(self, molecules):
        """Add molecules to the simulation system."""
        if not isinstance(molecules, list):
            molecules = [molecules]
        
        for molecule in molecules:
            self.molecules.append(molecule)
            self.particles.extend(molecule.atoms)
        
        logger.info(f"Added {len(molecules)} molecules to system")
    
    def set_force_field(self, force_field):
        """Set the force field for the simulation."""
        self.force_field = force_field
        logger.info(f"Set force field: {force_field.name}")
    
    def set_integrator(self, integrator):
        """Set the integrator for the simulation."""
        self.integrator = integrator
        logger.info(f"Set integrator: {integrator.name}")
    
    def prepare_simulation(self):
        """Prepare the system for simulation."""
        if not self.particles:
            raise ValueError("No particles in the system. Add proteins or molecules first.")
        
        if self.force_field is None:
            raise ValueError("No force field set for the system.")
        
        if self.integrator is None:
            raise ValueError("No integrator set for the system.")
        
        # Initialize positions, velocities, forces
        self.positions = np.array([p.position for p in self.particles])
        self.velocities = np.array([p.velocity for p in self.particles])
        self.forces = np.zeros_like(self.positions)
        
        # Initialize force field parameters
        self.force_field.initialize(self)
        
        # Initialize integrator
        self.integrator.initialize(self)
        
        logger.info(f"System prepared for simulation with {len(self.particles)} particles")
    
    def run(self, steps: int, output_freq: int = 100, trajectory_file: str = None):
        """
        Run the molecular dynamics simulation.
        
        Parameters
        ----------
        steps : int
            Number of simulation steps to run
        output_freq : int
            Frequency of output generation
        trajectory_file : str, optional
            File to write trajectory data
        """
        if not hasattr(self, 'positions'):
            self.prepare_simulation()
        
        self.total_steps = steps
        
        logger.info(f"Starting simulation for {steps} steps")
        
        for step in range(steps):
            # Calculate forces
            self.forces = self.force_field.calculate_forces(self.positions)
            
            # Update positions and velocities
            self.positions, self.velocities = self.integrator.step(
                self.positions, self.velocities, self.forces)
            
            # Apply periodic boundary conditions
            self.positions = self.positions % self.box_size
            
            # Update time
            self.current_step += 1
            self.elapsed_time += self.time_step
            
            # Output if needed
            if step % output_freq == 0:
                logger.info(f"Step {step}/{steps} completed")
                
                # Save trajectory if requested
                if trajectory_file:
                    self._save_trajectory_frame(trajectory_file)
        
        logger.info(f"Simulation completed: {steps} steps, {self.elapsed_time} ps elapsed")
    
    def _save_trajectory_frame(self, trajectory_file):
        """Save the current frame to a trajectory file."""
        # Simplified implementation - in a real system, would use specialized formats
        with open(trajectory_file, 'a') as f:
            f.write(f"Frame {self.current_step}, Time: {self.elapsed_time} ps\n")
            for i, pos in enumerate(self.positions):
                f.write(f"Atom {i}: {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}\n")
            f.write("\n")


class Particle:
    """
    Class representing a single particle in the MD simulation.
    
    This could be an atom or a coarse-grained particle.
    """
    
    def __init__(self, 
                 particle_id: int,
                 name: str,
                 element: str,
                 mass: float,
                 charge: float,
                 position: np.ndarray,
                 velocity: Optional[np.ndarray] = None):
        """
        Initialize a particle.
        
        Parameters
        ----------
        particle_id : int
            Unique identifier for the particle
        name : str
            Name of the particle (e.g., atom name)
        element : str
            Chemical element
        mass : float
            Mass in atomic mass units (u)
        charge : float
            Charge in elementary charge units (e)
        position : np.ndarray
            Initial position [x, y, z] in nanometers
        velocity : np.ndarray, optional
            Initial velocity [vx, vy, vz] in nm/ps
        """
        self.id = particle_id
        self.name = name
        self.element = element
        self.mass = mass
        self.charge = charge
        self.position = np.array(position, dtype=float)
        
        # Initialize velocity if not provided
        if velocity is None:
            self.velocity = np.zeros(3, dtype=float)
        else:
            self.velocity = np.array(velocity, dtype=float)
        
        # Additional properties
        self.force = np.zeros(3, dtype=float)
        self.potential_energy = 0.0
        
    def __repr__(self):
        return f"Particle(id={self.id}, name='{self.name}', element='{self.element}')"


class Integrator:
    """Base class for integration algorithms."""
    
    def __init__(self, name: str, time_step: float):
        """
        Initialize an integrator.
        
        Parameters
        ----------
        name : str
            Name of the integrator
        time_step : float
            Integration time step in picoseconds
        """
        self.name = name
        self.time_step = time_step
    
    def initialize(self, system):
        """Initialize the integrator with a system reference."""
        self.system = system
    
    def step(self, positions, velocities, forces):
        """
        Perform one integration step.
        
        This method should be implemented by derived classes.
        """
        raise NotImplementedError("This method should be implemented by derived classes")


class VelocityVerlet(Integrator):
    """Velocity Verlet integration algorithm."""
    
    def __init__(self, time_step: float):
        """Initialize a Velocity Verlet integrator."""
        super().__init__("Velocity Verlet", time_step)
    
    def step(self, positions, velocities, forces):
        """
        Perform one Velocity Verlet integration step.
        
        Parameters
        ----------
        positions : np.ndarray
            Current positions of all particles
        velocities : np.ndarray
            Current velocities of all particles
        forces : np.ndarray
            Current forces on all particles
        
        Returns
        -------
        tuple
            Updated positions and velocities
        """
        # Get mass array from system
        masses = np.array([p.mass for p in self.system.particles])
        masses = masses.reshape(-1, 1)  # Reshape for broadcasting
        
        # Calculate accelerations
        accelerations = forces / masses
        
        # Update velocities by half step
        velocities_half = velocities + 0.5 * accelerations * self.time_step
        
        # Update positions
        new_positions = positions + velocities_half * self.time_step
        
        # Calculate new forces (typically done outside this method)
        new_forces = self.system.force_field.calculate_forces(new_positions)
        
        # Calculate new accelerations
        new_accelerations = new_forces / masses
        
        # Update velocities for the second half step
        new_velocities = velocities_half + 0.5 * new_accelerations * self.time_step
        
        return new_positions, new_velocities


class LangevinIntegrator(Integrator):
    """
    Langevin dynamics integrator for temperature control.
    
    This integrator adds friction and random forces to mimic
    the effect of a heat bath at a specific temperature.
    """
    
    def __init__(self, time_step: float, temperature: float, friction_coeff: float = 1.0):
        """
        Initialize a Langevin integrator.
        
        Parameters
        ----------
        time_step : float
            Integration time step in picoseconds
        temperature : float
            Target temperature in Kelvin
        friction_coeff : float
            Friction coefficient in ps^-1
        """
        super().__init__("Langevin Dynamics", time_step)
        self.temperature = temperature
        self.friction_coeff = friction_coeff
        self.kB = 8.31446e-3  # Boltzmann constant in kJ/(mol·K)
    
    def step(self, positions, velocities, forces):
        """
        Perform one Langevin integration step.
        
        This implements a simple first-order integration scheme for
        the Langevin equation of motion.
        """
        # Get mass array
        masses = np.array([p.mass for p in self.system.particles])
        masses = masses.reshape(-1, 1)  # Reshape for broadcasting
        
        # Calculate deterministic acceleration (force/mass)
        accelerations = forces / masses
        
        # Calculate friction term
        friction = -self.friction_coeff * velocities
        
        # Calculate random force term (Gaussian noise)
        sigma = np.sqrt(2.0 * self.friction_coeff * self.kB * self.temperature / masses / self.time_step)
        random_force = sigma * np.random.normal(0.0, 1.0, velocities.shape)
        
        # Update velocities
        new_velocities = velocities + (accelerations + friction + random_force) * self.time_step
        
        # Update positions
        new_positions = positions + new_velocities * self.time_step
        
        return new_positions, new_velocities


class ProteinMDErrorHandlingSystem:
    """Main class for managing the ProteinMD error handling and logging system."""
    
    def __init__(self):
        self.logger = None
        self.config_manager = None
        self.integrator = None
        self.is_initialized = False
        
    def initialize(self, 
                  config_file=None,
                  environment="development",
                  log_level="INFO",
                  enable_integration=True,
                  custom_config=None):
        """Initialize the complete error handling and logging system."""
        
        if not ERROR_HANDLING_AVAILABLE:
            logger.warning("Error handling system components not available")
            return False
        
        try:
            # Step 1: Setup configuration
            self._setup_configuration(config_file, environment, custom_config)
            
            # Step 2: Initialize logging system
            self._initialize_logging(log_level)
            
            # Step 3: Setup module integration
            if enable_integration:
                self._setup_integration()
            
            # Step 4: Register default fallback operations
            self._register_default_fallbacks()
            
            self.is_initialized = True
            if self.logger:
                self.logger.info("ProteinMD Error Handling & Logging System initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ProteinMD Error Handling System: {e}")
            return False
    
    def _setup_configuration(self, 
                           config_file, 
                           environment,
                           custom_config):
        """Setup configuration management."""
        
        self.config_manager = ConfigurationManager(config_file)
        
        if config_file and Path(config_file).exists():
            # Load from file
            config = self.config_manager.load_configuration()
        elif custom_config:
            # Use custom configuration
            config = LoggingConfig.from_dict(custom_config)
        else:
            # Use template configuration
            config = get_template_config(environment)
            config.apply_environment_overrides()
        
        self.config = config
    
    def _initialize_logging(self, log_level):
        """Initialize the logging system."""
        
        # Override log level if specified
        if log_level != "INFO":
            config_dict = self.config.to_dict()
            config_dict['log_level'] = log_level
            self.config = LoggingConfig.from_dict(config_dict)
        
        # Setup main logger
        self.logger = setup_logging(self.config.to_dict())
    
    def _setup_integration(self):
        """Setup module integration."""
        
        self.integrator = ModuleIntegrator(self.logger)
        
        # Define common fallback operations
        common_fallbacks = {
            'read_file': self._fallback_read_file,
            'write_file': self._fallback_write_file,
            'parse_structure': self._fallback_parse_structure,
            'calculate_energy': self._fallback_calculate_energy,
        }
        
        # Register fallbacks for core modules
        core_modules = ['io', 'structure', 'simulation', 'analysis', 'visualization']
        for module in core_modules:
            try:
                self.integrator._register_fallbacks(f"proteinMD.{module}", common_fallbacks)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Could not register fallbacks for {module}: {e}")
    
    def _register_default_fallbacks(self):
        """Register default fallback operations."""
        
        if not self.logger:
            return
        
        # File I/O fallbacks
        self.logger.graceful_degradation.register_fallback(
            "read_file", self._fallback_read_file
        )
        self.logger.graceful_degradation.register_fallback(
            "write_file", self._fallback_write_file
        )
        
        # Structure processing fallbacks
        self.logger.graceful_degradation.register_fallback(
            "parse_structure", self._fallback_parse_structure
        )
        
        # Simulation fallbacks
        self.logger.graceful_degradation.register_fallback(
            "calculate_energy", self._fallback_calculate_energy
        )
        
        self.logger.info("Default fallback operations registered")
    
    # Fallback functions
    def _fallback_read_file(self, filename, *args, **kwargs):
        """Fallback for file reading operations."""
        if self.logger:
            self.logger.warning(f"Using fallback for reading file: {filename}")
        return f"# Fallback content for {filename}\n# Original file could not be read\n"
    
    def _fallback_write_file(self, filename, content, *args, **kwargs):
        """Fallback for file writing operations."""
        if self.logger:
            self.logger.warning(f"Using fallback for writing file: {filename}")
        try:
            # Try to write to a backup location
            backup_path = Path(filename).with_suffix('.backup')
            with open(backup_path, 'w') as f:
                f.write(content)
            return True
        except Exception:
            return False
    
    def _fallback_parse_structure(self, structure_data, *args, **kwargs):
        """Fallback for structure parsing operations."""
        if self.logger:
            self.logger.warning("Using fallback for structure parsing")
        return {
            'atoms': [],
            'bonds': [],
            'residues': [],
            'chains': [],
            'fallback': True,
            'message': 'Structure parsing failed, using empty structure'
        }
    
    def _fallback_calculate_energy(self, *args, **kwargs):
        """Fallback for energy calculation operations."""
        if self.logger:
            self.logger.warning("Using fallback for energy calculation")
        return 0.0  # Return neutral energy
    
    def get_system_status(self):
        """Get comprehensive system status."""
        
        if not self.is_initialized:
            return {'initialized': False, 'error': 'System not initialized'}
        
        status = {
            'initialized': True,
            'configuration': self.config.to_dict() if hasattr(self, 'config') else {},
            'logging_statistics': self.logger.get_statistics() if self.logger else {},
            'integrated_modules': self.integrator.integrated_modules if self.integrator else {},
            'fallback_operations': len(self.logger.graceful_degradation.fallback_registry) if self.logger else 0
        }
        
        return status
    
    def shutdown(self):
        """Shutdown the error handling system gracefully."""
        
        if self.logger:
            self.logger.info("Shutting down ProteinMD Error Handling & Logging System")
            
            # Log final statistics
            stats = self.logger.get_statistics()
            self.logger.info("Final system statistics", stats)
        
        self.is_initialized = False


# Global system instance
_error_handling_system = None


def initialize_error_handling(config_file=None,
                            environment="development",
                            log_level="INFO",
                            enable_integration=True,
                            custom_config=None):
    """Initialize the global ProteinMD error handling system."""
    
    global _error_handling_system
    
    if not ERROR_HANDLING_AVAILABLE:
        logger.warning("Error handling system not available")
        return False
    
    _error_handling_system = ProteinMDErrorHandlingSystem()
    success = _error_handling_system.initialize(
        config_file=config_file,
        environment=environment,
        log_level=log_level,
        enable_integration=enable_integration,
        custom_config=custom_config
    )
    
    return success


def get_error_handling_system():
    """Get the global error handling system."""
    return _error_handling_system


def shutdown_error_handling():
    """Shutdown the global error handling system."""
    global _error_handling_system
    
    if _error_handling_system:
        _error_handling_system.shutdown()
        _error_handling_system = None


def get_system_logger():
    """Get the system logger."""
    if ERROR_HANDLING_AVAILABLE and _error_handling_system and _error_handling_system.is_initialized:
        return _error_handling_system.logger
    return None


def get_system_status():
    """Get system status."""
    if _error_handling_system:
        return _error_handling_system.get_system_status()
    return {'initialized': False, 'error': 'System not created', 'available': ERROR_HANDLING_AVAILABLE}


# Auto-initialization for development
def auto_initialize_error_handling():
    """Auto-initialize the error handling system with sensible defaults."""
    
    if not ERROR_HANDLING_AVAILABLE:
        return False
    
    # Check if already initialized
    if _error_handling_system and _error_handling_system.is_initialized:
        return True
    
    # Determine environment
    import os
    environment = os.getenv('PROTEINMD_ENVIRONMENT', 'development')
    log_level = os.getenv('PROTEINMD_LOG_LEVEL', 'INFO')
    
    # Look for config file
    config_file = None
    possible_configs = [
        'proteinmd_config.json',
        'config/logging.json',
        'config/proteinmd.json',
        os.path.expanduser('~/.proteinmd/config.json')
    ]
    
    for config_path in possible_configs:
        if Path(config_path).exists():
            config_file = config_path
            break
    
    return initialize_error_handling(
        config_file=config_file,
        environment=environment,
        log_level=log_level,
        enable_integration=True
    )


# Auto-initialize on import in development mode
if ERROR_HANDLING_AVAILABLE:
    import os
    if os.getenv('PROTEINMD_AUTO_INIT', 'true').lower() in ['true', '1', 'yes']:
        try:
            auto_initialize_error_handling()
            logger.info("Error handling system auto-initialized")
        except Exception as e:
            logger.warning(f"Auto-initialization of error handling failed: {e}")


# Export error handling components if available
if ERROR_HANDLING_AVAILABLE:
    __all__ = [
        # Original core components
        'MDSystem', 'Force', 'Particle', 'VelocityVerletIntegrator', 
        'LangevinIntegrator', 'LennardJonesForce',
        # Error handling components
        'ProteinMDError', 'SimulationError', 'StructureError', 'ForceFieldError',
        'ProteinMDIOError', 'AnalysisError', 'VisualizationError', 
        'PerformanceError', 'ConfigurationError', 'ProteinMDWarning',
        'ProteinMDLogger', 'LoggingConfig', 'initialize_error_handling',
        'get_error_handling_system', 'get_system_logger', 'get_system_status',
        'safe_operation', 'with_error_recovery', 'exception_context',
        'ValidationMixin', 'ERROR_HANDLING_AVAILABLE'
    ]
else:
    __all__ = [
        # Original core components only
        'MDSystem', 'Force', 'Particle', 'VelocityVerletIntegrator', 
        'LangevinIntegrator', 'LennardJonesForce', 'ERROR_HANDLING_AVAILABLE'
    ]

# Log initialization status
logger.info(f"ProteinMD core module initialized. Error handling available: {ERROR_HANDLING_AVAILABLE}")
