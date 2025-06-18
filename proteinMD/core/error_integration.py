"""
ProteinMD Error Handling & Logging Integration

This module provides integration utilities to apply comprehensive error handling
and logging across all existing proteinMD modules. It includes decorators,
context managers, and utilities for seamless integration.

Author: ProteinMD Development Team
Date: 2024
"""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union
from contextlib import contextmanager
import importlib
import pkgutil
import sys
from pathlib import Path

from .exceptions import (
    ProteinMDError, SimulationError, StructureError, ForceFieldError,
    ProteinMDIOError, AnalysisError, VisualizationError, PerformanceError,
    ConfigurationError, ProteinMDWarning
)
from .logging_system import ProteinMDLogger, get_logger, logged_function
from .logging_config import LoggingConfig, ConfigurationManager


class ModuleIntegrator:
    """Integrates error handling and logging into existing modules."""
    
    def __init__(self, logger: Optional[ProteinMDLogger] = None):
        self.logger = logger or get_logger()
        self.integrated_modules: Dict[str, bool] = {}
        self.fallback_registry: Dict[str, Dict[str, Callable]] = {}
    
    def integrate_module(self, module_name: str, 
                        module_obj: Any = None,
                        error_mapping: Optional[Dict[str, Type[ProteinMDError]]] = None,
                        fallback_functions: Optional[Dict[str, Callable]] = None) -> bool:
        """Integrate error handling and logging into a module."""
        try:
            if module_obj is None:
                module_obj = importlib.import_module(module_name)
            
            # Default error mapping based on module name
            if error_mapping is None:
                error_mapping = self._get_default_error_mapping(module_name)
            
            # Apply logging decorators to functions
            self._apply_logging_decorators(module_obj, module_name)
            
            # Register fallback functions
            if fallback_functions:
                self._register_fallbacks(module_name, fallback_functions)
            
            # Add error context to classes
            self._add_error_context(module_obj, error_mapping)
            
            self.integrated_modules[module_name] = True
            self.logger.info(f"Successfully integrated error handling for module: {module_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to integrate module {module_name}: {e}", exception=e)
            return False
    
    def _get_default_error_mapping(self, module_name: str) -> Dict[str, Type[ProteinMDError]]:
        """Get default error mapping based on module name."""
        mappings = {
            'simulation': SimulationError,
            'dynamics': SimulationError,
            'integrator': SimulationError,
            'structure': StructureError,
            'protein': StructureError,
            'molecule': StructureError,
            'forcefield': ForceFieldError,
            'force': ForceFieldError,
            'potential': ForceFieldError,
            'io': ProteinMDIOError,
            'parser': ProteinMDIOError,
            'writer': ProteinMDIOError,
            'analysis': AnalysisError,
            'trajectory': AnalysisError,
            'visualization': VisualizationError,
            'plot': VisualizationError,
            'render': VisualizationError,
            'config': ConfigurationError,
            'settings': ConfigurationError
        }
        
        default_error = ProteinMDError
        for keyword, error_class in mappings.items():
            if keyword in module_name.lower():
                default_error = error_class
                break
        
        return {'default': default_error}
    
    def _apply_logging_decorators(self, module_obj: Any, module_name: str):
        """Apply logging decorators to module functions."""
        module_logger = get_logger(module_name)
        
        for name in dir(module_obj):
            obj = getattr(module_obj, name)
            
            # Skip private attributes and non-callables
            if name.startswith('_') or not callable(obj):
                continue
            
            # Skip already decorated functions
            if hasattr(obj, '_proteinmd_logged'):
                continue
            
            # Apply logging decorator
            try:
                decorated_func = self._create_logged_function(obj, module_logger, module_name)
                decorated_func._proteinmd_logged = True
                setattr(module_obj, name, decorated_func)
                
            except Exception as e:
                self.logger.warning(f"Could not decorate function {name} in {module_name}: {e}")
    
    def _create_logged_function(self, func: Callable, logger: ProteinMDLogger, module_name: str) -> Callable:
        """Create a logged version of a function."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{module_name}.{func.__name__}"
            
            try:
                # Log function entry
                logger.debug(f"Entering function: {func_name}")
                
                # Execute with performance monitoring for significant functions
                if self._should_monitor_performance(func):
                    with logger.performance_context(func_name):
                        result = func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Log successful completion
                logger.debug(f"Function {func_name} completed successfully")
                return result
                
            except Exception as e:
                # Determine appropriate error type
                error_class = self._determine_error_class(e, module_name)
                
                # Convert to ProteinMD error if needed
                if not isinstance(e, ProteinMDError):
                    proteinmd_error = error_class(
                        message=str(e),
                        error_code=f"{module_name.upper()}_ERR",
                        details={'original_exception': type(e).__name__},
                        suggestions=[f"Check input parameters for {func_name}"]
                    )
                else:
                    proteinmd_error = e
                
                # Try graceful degradation
                recovery_op = f"{module_name}.{func.__name__}"
                recovery_successful = logger.log_exception(
                    proteinmd_error,
                    f"Error in {func_name}",
                    {'function': func_name, 'module': module_name},
                    attempt_recovery=True,
                    recovery_operation=recovery_op
                )
                
                if not recovery_successful:
                    raise proteinmd_error
        
        return wrapper
    
    def _should_monitor_performance(self, func: Callable) -> bool:
        """Determine if function should have performance monitoring."""
        # Monitor functions that likely involve significant computation
        monitor_keywords = [
            'simulate', 'calculate', 'compute', 'analyze', 'optimize',
            'integrate', 'solve', 'process', 'transform', 'render'
        ]
        
        func_name = func.__name__.lower()
        return any(keyword in func_name for keyword in monitor_keywords)
    
    def _determine_error_class(self, exception: Exception, module_name: str) -> Type[ProteinMDError]:
        """Determine appropriate ProteinMD error class for an exception."""
        error_mapping = self._get_default_error_mapping(module_name)
        return error_mapping.get('default', ProteinMDError)
    
    def _add_error_context(self, module_obj: Any, error_mapping: Dict[str, Type[ProteinMDError]]):
        """Add error context to module classes."""
        for name in dir(module_obj):
            obj = getattr(module_obj, name)
            
            # Check if it's a class
            if inspect.isclass(obj) and not name.startswith('_'):
                try:
                    self._enhance_class_with_error_handling(obj, error_mapping)
                except Exception as e:
                    self.logger.warning(f"Could not enhance class {name}: {e}")
    
    def _enhance_class_with_error_handling(self, cls: Type, error_mapping: Dict[str, Type[ProteinMDError]]):
        """Enhance a class with error handling capabilities."""
        # Add error context manager
        def error_context(self, operation: str):
            return ErrorContext(operation, self.__class__.__name__, self.logger if hasattr(self, 'logger') else get_logger())
        
        # Add to class if not already present
        if not hasattr(cls, 'error_context'):
            cls.error_context = error_context
    
    def _register_fallbacks(self, module_name: str, fallback_functions: Dict[str, Callable]):
        """Register fallback functions for a module."""
        self.fallback_registry[module_name] = fallback_functions
        
        for operation, fallback in fallback_functions.items():
            self.logger.graceful_degradation.register_fallback(
                f"{module_name}.{operation}", fallback
            )
    
    def integrate_all_modules(self, package_name: str = "proteinMD") -> Dict[str, bool]:
        """Integrate all modules in a package."""
        results = {}
        
        try:
            package = importlib.import_module(package_name)
            package_path = package.__path__
            
            for importer, modname, ispkg in pkgutil.walk_packages(package_path, package_name + "."):
                try:
                    module = importlib.import_module(modname)
                    success = self.integrate_module(modname, module)
                    results[modname] = success
                except Exception as e:
                    self.logger.error(f"Failed to integrate module {modname}: {e}")
                    results[modname] = False
        
        except Exception as e:
            self.logger.error(f"Failed to walk package {package_name}: {e}")
        
        return results


class ErrorContext:
    """Context manager for error handling in specific operations."""
    
    def __init__(self, operation: str, context: str, logger: ProteinMDLogger):
        self.operation = operation
        self.context = context
        self.logger = logger
        self.start_time = None
    
    def __enter__(self):
        self.start_time = self.logger.performance_monitor.start_timing(f"{self.context}.{self.operation}")
        self.logger.debug(f"Starting operation: {self.operation} in {self.context}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success
            execution_time, memory_delta = self.logger.performance_monitor.end_timing(f"{self.context}.{self.operation}")
            self.logger.debug(f"Completed operation: {self.operation} in {self.context}", 
                            {'execution_time': execution_time, 'memory_delta': memory_delta})
        else:
            # Error occurred
            self.logger.error(f"Error in operation: {self.operation} in {self.context}", 
                            exception=exc_val)
        
        return False  # Don't suppress exceptions


def safe_operation(operation_name: str, 
                  error_class: Type[ProteinMDError] = ProteinMDError,
                  fallback_result: Any = None,
                  suppress_errors: bool = False):
    """Decorator for safe operation execution with error handling."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            
            try:
                return func(*args, **kwargs)
            except ProteinMDError:
                # Already a ProteinMD error, re-raise
                raise
            except Exception as e:
                # Convert to ProteinMD error
                proteinmd_error = error_class(
                    message=f"Error in {operation_name}: {str(e)}",
                    error_code=f"{operation_name.upper()}_ERR",
                    details={'original_exception': type(e).__name__}
                )
                
                logger.error(f"Error in safe operation {operation_name}", exception=proteinmd_error)
                
                if suppress_errors:
                    return fallback_result
                else:
                    raise proteinmd_error
        
        return wrapper
    return decorator


def with_error_recovery(recovery_function: Callable, 
                       max_retries: int = 3,
                       retry_delay: float = 1.0):
    """Decorator for functions with automatic error recovery."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, retrying...")
                        
                        # Try recovery function
                        try:
                            recovery_function(*args, **kwargs)
                            import time
                            time.sleep(retry_delay)
                        except Exception as recovery_error:
                            logger.error(f"Recovery failed: {recovery_error}")
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                        raise
        
        return wrapper
    return decorator


@contextmanager
def exception_context(operation: str, 
                     error_class: Type[ProteinMDError] = ProteinMDError,
                     log_level: str = "ERROR"):
    """Context manager for exception handling with logging."""
    logger = get_logger()
    
    try:
        logger.debug(f"Starting operation: {operation}")
        yield
        logger.debug(f"Completed operation: {operation}")
    except ProteinMDError:
        # Already a ProteinMD error, just log and re-raise
        logger.error(f"ProteinMD error in {operation}")
        raise
    except Exception as e:
        # Convert to ProteinMD error
        proteinmd_error = error_class(
            message=f"Error in {operation}: {str(e)}",
            error_code=f"{operation.upper().replace(' ', '_')}_ERR",
            details={'original_exception': type(e).__name__}
        )
        
        logger.error(f"Error in {operation}", exception=proteinmd_error)
        raise proteinmd_error


class ValidationMixin:
    """Mixin class to add validation and error handling to existing classes."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__module__)
    
    def validate_parameter(self, name: str, value: Any, 
                          validator: Callable[[Any], bool],
                          error_message: str = None) -> Any:
        """Validate a parameter value."""
        try:
            if not validator(value):
                msg = error_message or f"Invalid value for parameter {name}: {value}"
                raise ValueError(msg)
            return value
        except Exception as e:
            error = ConfigurationError(
                message=f"Parameter validation failed for {name}: {str(e)}",
                error_code="VALIDATION_ERR",
                details={'parameter': name, 'value': str(value)}
            )
            self.logger.error(f"Parameter validation failed", exception=error)
            raise error
    
    def require_parameter(self, name: str, value: Any) -> Any:
        """Require a parameter to be non-None."""
        if value is None:
            error = ConfigurationError(
                message=f"Required parameter {name} is None",
                error_code="REQUIRED_PARAM_ERR",
                details={'parameter': name}
            )
            self.logger.error(f"Required parameter missing", exception=error)
            raise error
        return value


# Utility functions for integration
def integrate_proteinmd_package():
    """Integrate error handling and logging into the entire ProteinMD package."""
    integrator = ModuleIntegrator()
    results = integrator.integrate_all_modules("proteinMD")
    
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    logger = get_logger()
    logger.info(f"Integration complete: {success_count}/{total_count} modules integrated successfully")
    
    if total_count > 0:
        for module_name, success in results.items():
            if not success:
                logger.warning(f"Failed to integrate module: {module_name}")
    
    return results


def log_exception(exception: Exception, context: str = "", logger: Optional[ProteinMDLogger] = None) -> None:
    """
    Log an exception with proper context and formatting.
    
    Args:
        exception: The exception to log
        context: Additional context information
        logger: Logger instance to use (if None, gets default logger)
    """
    if logger is None:
        logger = get_logger(__name__)
    
    error_msg = f"Exception in {context}: {type(exception).__name__}: {exception}"
    # Use the logger's error method without exc_info to avoid conflicts
    logger.error(error_msg)


def setup_comprehensive_logging(config_file: Optional[str] = None) -> ProteinMDLogger:
    """Setup comprehensive logging and error handling for ProteinMD."""
    # Load configuration
    if config_file:
        manager = ConfigurationManager(config_file)
        config = manager.load_configuration()
    else:
        from .logging_config import create_default_config
        config = create_default_config()
    
    # Setup main logger
    from .logging_system import setup_logging
    logger = setup_logging(config.to_dict())
    
    # Integrate all modules
    integrate_proteinmd_package()
    
    logger.info("Comprehensive logging and error handling setup complete")
    return logger


# Example usage and testing
if __name__ == "__main__":
    # Setup comprehensive logging
    logger = setup_comprehensive_logging()
    
    # Test integration
    @safe_operation("test_operation", SimulationError)
    def test_function():
        raise ValueError("Test error")
    
    try:
        test_function()
    except Exception as e:
        logger.info(f"Caught expected error: {e}")
    
    # Test error context
    with exception_context("test_context", AnalysisError):
        logger.info("Inside error context")
    
    logger.info("Integration testing complete")
