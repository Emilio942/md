"""
Integration of Error Handling & Logging with ProteinMD Core Modules

This module integrates the error handling and logging system with existing
ProteinMD modules to provide comprehensive error management throughout the package.

Author: GitHub Copilot
Date: June 13, 2025
"""

import sys
import traceback
import functools
from typing import Any, Callable, Optional, Dict
from pathlib import Path

# Import our error handling components
from .error_handling import (
    ProteinMDError, SimulationError, ForceCalculationError, IntegrationError,
    IOError, FileFormatError, SystemError, TopologyError, ValidationError,
    ErrorSeverity, logger, error_reporter, robust_operation, graceful_degradation
)
from .enhanced_logging import (
    LogManager, LogConfig, ComponentLogger, PerformanceLogger,
    performance_monitoring, performance_monitor, get_component_logger
)

# Component-specific loggers
simulation_logger = get_component_logger("simulation")
io_logger = get_component_logger("io")
analysis_logger = get_component_logger("analysis")
visualization_logger = get_component_logger("visualization")
forcefield_logger = get_component_logger("forcefield")


def setup_global_exception_handler():
    """Set up global exception handler for uncaught exceptions."""
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Global exception handler."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow keyboard interrupts to work normally
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        # Log the uncaught exception
        logger.critical(
            f"Uncaught exception: {exc_type.__name__}: {exc_value}",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        
        # Create error report
        try:
            report = error_reporter.create_error_report(
                exc_value,
                context={"exception_type": "uncaught", "location": "global"}
            )
            
            # Save error report
            timestamp = report.timestamp.replace(":", "-").replace(".", "-")
            report_file = f"error_report_{timestamp}.json"
            error_reporter.save_error_report(report, report_file)
            
        except Exception as e:
            # If error reporting fails, at least log it
            logger.error(f"Failed to create error report: {e}")
    
    sys.excepthook = handle_exception


class SimulationErrorHandler:
    """Error handler specifically for simulation operations."""
    
    def __init__(self):
        self.logger = simulation_logger
        
    def handle_integration_error(self, error: Exception, simulation_state: Dict[str, Any]):
        """Handle integration errors with specific recovery strategies."""
        context = {
            "step": simulation_state.get("step", "unknown"),
            "current_step": simulation_state.get("step", "unknown"),
            "timestep": simulation_state.get("timestep", "unknown"),
            "temperature": simulation_state.get("temperature", "unknown"),
            "total_energy": simulation_state.get("total_energy", "unknown")
        }
        
        if "nan" in str(error).lower() or "inf" in str(error).lower():
            raise IntegrationError(
                "Numerical instability detected during integration",
                context=context,
                suggestions=[
                    "Reduce the integration timestep",
                    "Check for overlapping atoms",
                    "Verify force field parameters",
                    "Increase energy minimization steps"
                ],
                severity=ErrorSeverity.CRITICAL
            )
        elif "constraint" in str(error).lower():
            raise IntegrationError(
                "Constraint violation during integration",
                context=context,
                suggestions=[
                    "Check constraint definitions",
                    "Reduce timestep for constrained systems",
                    "Verify system geometry"
                ],
                severity=ErrorSeverity.ERROR
            )
        else:
            raise IntegrationError(
                f"Integration failed: {error}",
                context=context,
                severity=ErrorSeverity.ERROR
            ) from error
            
    def handle_force_calculation_error(self, error: Exception, system_info: Dict[str, Any]):
        """Handle force calculation errors."""
        context = {
            "n_atoms": system_info.get("n_atoms", "unknown"),
            "forcefield": system_info.get("forcefield", "unknown"),
            "cutoff": system_info.get("cutoff", "unknown")
        }
        
        if "memory" in str(error).lower():
            raise ForceCalculationError(
                "Insufficient memory for force calculation",
                context=context,
                suggestions=[
                    "Reduce system size",
                    "Use larger cutoff distances",
                    "Enable memory optimization",
                    "Increase available RAM"
                ],
                severity=ErrorSeverity.CRITICAL
            )
        else:
            raise ForceCalculationError(
                f"Force calculation failed: {error}",
                context=context,
                severity=ErrorSeverity.ERROR
            ) from error


class IOErrorHandler:
    """Error handler for I/O operations."""
    
    def __init__(self):
        self.logger = io_logger
        
    def handle_file_read_error(self, error: Exception, filepath: str, format_type: str = None):
        """Handle file reading errors with format-specific suggestions."""
        context = {
            "filepath": filepath,
            "format": format_type,
            "file_exists": Path(filepath).exists(),
            "file_size": Path(filepath).stat().st_size if Path(filepath).exists() else 0
        }
        
        if not Path(filepath).exists():
            raise FileFormatError(
                f"File not found: {filepath}",
                context=context,
                suggestions=[
                    "Check file path spelling",
                    "Verify file exists in specified location",
                    "Check file permissions"
                ],
                severity=ErrorSeverity.ERROR
            )
        elif Path(filepath).stat().st_size == 0:
            raise FileFormatError(
                f"Empty file: {filepath}",
                context=context,
                suggestions=[
                    "Check if file was corrupted during transfer",
                    "Verify file generation completed successfully"
                ],
                severity=ErrorSeverity.ERROR
            )
        elif "permission" in str(error).lower():
            raise FileFormatError(
                f"Permission denied accessing file: {filepath}",
                context=context,
                suggestions=[
                    "Check file permissions",
                    "Run with appropriate user privileges",
                    "Verify directory access rights"
                ],
                severity=ErrorSeverity.ERROR
            )
        else:
            format_suggestions = []
            if format_type:
                if format_type.lower() == "pdb":
                    format_suggestions.extend([
                        "Verify PDB format compliance",
                        "Check for non-standard residue names",
                        "Validate ATOM/HETATM record formatting"
                    ])
                elif format_type.lower() in ["dcd", "xtc", "trr"]:
                    format_suggestions.extend([
                        "Check trajectory file integrity",
                        "Verify binary format compatibility",
                        "Try alternative trajectory reader"
                    ])
                    
            raise FileFormatError(
                f"Failed to read {format_type or 'file'}: {filepath}",
                context=context,
                suggestions=format_suggestions or [
                    "Check file format compatibility",
                    "Verify file is not corrupted",
                    "Try alternative file format"
                ],
                severity=ErrorSeverity.ERROR
            ) from error
            
    def handle_file_write_error(self, error: Exception, filepath: str):
        """Handle file writing errors."""
        context = {
            "filepath": filepath,
            "directory_exists": Path(filepath).parent.exists(),
            "directory_writable": os.access(Path(filepath).parent, os.W_OK) if Path(filepath).parent.exists() else False
        }
        
        if "space" in str(error).lower() or "disk" in str(error).lower():
            raise IOError(
                f"Insufficient disk space to write file: {filepath}",
                context=context,
                suggestions=[
                    "Free up disk space",
                    "Write to alternative location",
                    "Use compression to reduce file size"
                ],
                severity=ErrorSeverity.CRITICAL
            )
        elif not Path(filepath).parent.exists():
            raise IOError(
                f"Directory does not exist: {Path(filepath).parent}",
                context=context,
                suggestions=[
                    "Create parent directory",
                    "Check path spelling",
                    "Verify directory permissions"
                ],
                severity=ErrorSeverity.ERROR
            )
        else:
            raise IOError(
                f"Failed to write file: {filepath}",
                context=context,
                severity=ErrorSeverity.ERROR
            ) from error


class SystemErrorHandler:
    """Error handler for system setup and configuration."""
    
    def __init__(self):
        self.logger = get_component_logger("system")
        
    def handle_topology_error(self, error: Exception, system_info: Dict[str, Any]):
        """Handle topology-related errors."""
        context = {
            "n_atoms": system_info.get("n_atoms", "unknown"),
            "n_residues": system_info.get("n_residues", "unknown"),
            "chains": system_info.get("chains", "unknown"),
            "forcefield": system_info.get("forcefield", "unknown")
        }
        
        if "missing" in str(error).lower() and "parameter" in str(error).lower():
            raise TopologyError(
                "Missing force field parameters in topology",
                context=context,
                suggestions=[
                    "Check force field parameter coverage",
                    "Add missing parameters manually",
                    "Use alternative force field",
                    "Validate residue and atom names"
                ],
                severity=ErrorSeverity.ERROR
            )
        elif "bond" in str(error).lower():
            raise TopologyError(
                "Bond topology error",
                context=context,
                suggestions=[
                    "Check for missing bonds",
                    "Verify connectivity definitions",
                    "Check for unrealistic bond lengths"
                ],
                severity=ErrorSeverity.ERROR
            )
        else:
            raise TopologyError(
                f"Topology error: {error}",
                context=context,
                severity=ErrorSeverity.ERROR
            ) from error


# Create global error handlers
simulation_error_handler = SimulationErrorHandler()
io_error_handler = IOErrorHandler()
system_error_handler = SystemErrorHandler()


def proteinmd_exception_handler(component: str = "general"):
    """Decorator to add ProteinMD-specific exception handling to functions."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            comp_logger = get_component_logger(component)
            
            try:
                return func(*args, **kwargs)
            except ProteinMDError:
                # Re-raise ProteinMD errors as they're already properly formatted
                raise
            except Exception as e:
                # Convert generic exceptions to ProteinMD errors with context
                context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "component": component,
                    "args_count": len(args),
                    "kwargs": list(kwargs.keys())
                }
                
                comp_logger.error(f"Unexpected error in {func.__name__}", exc_info=True)
                
                # Create error report
                report = error_reporter.create_error_report(e, context=context)
                
                # Convert to appropriate ProteinMD error
                if component == "simulation":
                    raise SimulationError(
                        f"Simulation error in {func.__name__}: {e}",
                        context=context,
                        severity=ErrorSeverity.ERROR
                    ) from e
                elif component == "io":
                    raise IOError(
                        f"I/O error in {func.__name__}: {e}",
                        context=context,
                        severity=ErrorSeverity.ERROR
                    ) from e
                elif component == "analysis":
                    raise ValidationError(
                        f"Analysis error in {func.__name__}: {e}",
                        context=context,
                        severity=ErrorSeverity.ERROR
                    ) from e
                else:
                    raise ProteinMDError(
                        f"Error in {func.__name__}: {e}",
                        context=context,
                        severity=ErrorSeverity.ERROR
                    ) from e
                    
        return wrapper
    return decorator


def log_function_calls(component: str = "general", 
                      log_args: bool = False,
                      log_performance: bool = True):
    """Decorator to log function calls with optional performance monitoring."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            comp_logger = get_component_logger(component)
            
            # Log function entry
            entry_msg = f"Entering {func.__name__}"
            if log_args:
                entry_msg += f" with args={len(args)}, kwargs={list(kwargs.keys())}"
            comp_logger.debug(entry_msg)
            
            # Execute with performance monitoring if enabled
            if log_performance:
                with performance_monitoring(
                    operation=f"{component}.{func.__name__}",
                    context={"args_count": len(args), "kwargs_count": len(kwargs)}
                ):
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            # Log function exit
            comp_logger.debug(f"Exiting {func.__name__}")
            return result
            
        return wrapper
    return decorator


# Initialize global error handling
def initialize_error_handling(log_config: LogConfig = None):
    """Initialize the global error handling system."""
    
    # Set up logging
    if log_config:
        from .enhanced_logging import setup_logging
        setup_logging(log_config)
    
    # Set up global exception handler
    setup_global_exception_handler()
    
    # Log initialization
    logger.info("ProteinMD error handling and logging system initialized")


if __name__ == "__main__":
    print("Testing ProteinMD Error Handler Integration...")
    
    # Initialize error handling
    initialize_error_handling()
    
    # Test component-specific error handling
    @proteinmd_exception_handler("simulation")
    @log_function_calls("simulation", log_performance=True)
    def test_simulation_function():
        simulation_logger.info("Running test simulation")
        return "simulation_complete"
    
    @proteinmd_exception_handler("io")
    def test_io_function():
        io_logger.info("Testing I/O operation")
        # Simulate file read error
        raise FileNotFoundError("test.pdb")
    
    # Test successful operation
    result = test_simulation_function()
    print(f"Simulation result: {result}")
    
    # Test error handling
    try:
        test_io_function()
    except ProteinMDError as e:
        print(f"Caught ProteinMD error: {e.severity.value}")
        print(f"Suggestions: {e.suggestions}")
    
    print("Error handler integration test completed!")
