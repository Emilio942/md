"""
Task 15.1: Error Handling & Logging System for ProteinMD

This module provides robust error management for production environments with:
- Comprehensive exception handling
- Structured logging with different log levels
- Automatic error reports with stack traces
- Graceful degradation for non-critical failures

Author: GitHub Copilot
Date: June 13, 2025
"""

import logging
import traceback
import sys
import os
import json
import datetime
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import functools


class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class ProteinMDError(Exception):
    """Base exception for all ProteinMD errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, 
                 suggestions: Optional[List[str]] = None, severity: ErrorSeverity = ErrorSeverity.ERROR):
        super().__init__(message)
        self.context = context or {}
        self.suggestions = suggestions or []
        self.severity = severity
        self.timestamp = datetime.datetime.now()
        
    def __str__(self):
        msg = super().__str__()
        
        if self.context:
            msg += f"\n\nContext: {json.dumps(self.context, indent=2)}"
            
        if self.suggestions:
            msg += f"\n\nSuggestions:"
            for i, suggestion in enumerate(self.suggestions, 1):
                msg += f"\n  {i}. {suggestion}"
                
        return msg


class SimulationError(ProteinMDError):
    """Errors related to simulation execution."""
    pass


class ForceCalculationError(SimulationError):
    """Errors in force calculations."""
    pass


class IntegrationError(SimulationError):
    """Errors in time integration."""
    pass


class IOError(ProteinMDError):
    """Errors in input/output operations."""
    pass


class FileFormatError(IOError):
    """Errors related to file format parsing."""
    pass


class SystemError(ProteinMDError):
    """Errors in system setup or configuration."""
    pass


class TopologyError(SystemError):
    """Errors in molecular topology."""
    pass


class ValidationError(ProteinMDError):
    """Errors in data validation."""
    pass


@dataclass
class ErrorReport:
    """Comprehensive error report structure."""
    timestamp: str
    error_type: str
    error_message: str
    stack_trace: str
    severity: str
    context: Dict[str, Any]
    suggestions: List[str]
    system_info: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_details: Optional[str] = None


class ProteinMDLogger:
    """Centralized logging system for ProteinMD."""
    
    def __init__(self, name: str = "proteinmd"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.handlers: Dict[str, logging.Handler] = {}
        self.context_stack: List[Dict[str, Any]] = []
        self._setup_default_logging()
        
    def _setup_default_logging(self):
        """Set up default logging configuration."""
        if not self.logger.handlers:
            self.setup_logging()
    
    def setup_logging(self, level: int = logging.INFO, 
                     log_file: Optional[str] = None,
                     console_level: Optional[int] = None,
                     file_level: Optional[int] = None):
        """Configure comprehensive logging system."""
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set main logger level
        self.logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level or level)
        
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        self.handlers['console'] = console_handler
        
        # File handler
        if log_file:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level or logging.DEBUG)
            
            detailed_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
            )
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
            self.handlers['file'] = file_handler
            
        # Prevent duplicate messages
        self.logger.propagate = False
        
    def add_context(self, **context):
        """Add context to the logging stack."""
        self.context_stack.append(context)
        
    def remove_context(self):
        """Remove the last context from the stack."""
        if self.context_stack:
            self.context_stack.pop()
            
    def _format_message_with_context(self, message: str) -> str:
        """Format message with current context."""
        if not self.context_stack:
            return message
            
        context = {}
        for ctx in self.context_stack:
            context.update(ctx)
            
        return f"{message} [Context: {json.dumps(context)}]"
        
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        formatted_message = self._format_message_with_context(message)
        self.logger.debug(formatted_message, **kwargs)
        
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        formatted_message = self._format_message_with_context(message)
        self.logger.info(formatted_message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        formatted_message = self._format_message_with_context(message)
        self.logger.warning(formatted_message, **kwargs)
        
    def error(self, message: str, exc_info: bool = True, **kwargs):
        """Log error message with context and exception info."""
        formatted_message = self._format_message_with_context(message)
        self.logger.error(formatted_message, exc_info=exc_info, **kwargs)
        
    def critical(self, message: str, exc_info: bool = True, **kwargs):
        """Log critical message with context and exception info."""
        formatted_message = self._format_message_with_context(message)
        self.logger.critical(formatted_message, exc_info=exc_info, **kwargs)


@contextmanager
def error_context(logger: ProteinMDLogger, **context):
    """Context manager for adding logging context."""
    logger.add_context(**context)
    try:
        yield
    finally:
        logger.remove_context()


class ErrorRecoveryManager:
    """Manages error recovery strategies for robust operations."""
    
    def __init__(self, logger: Optional[ProteinMDLogger] = None):
        self.logger = logger or ProteinMDLogger()
        self.recovery_strategies: Dict[type, Callable] = {}
        self.max_retries: int = 3
        self.retry_delay: float = 1.0
        
    def register_strategy(self, error_type: type, strategy: Callable):
        """Register recovery strategy for specific error type."""
        self.recovery_strategies[error_type] = strategy
        self.logger.info(f"Registered recovery strategy for {error_type.__name__}")
        
    def execute_with_recovery(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with automatic error recovery."""
        retry_count = 0
        last_exception = None
        
        while retry_count < self.max_retries:
            try:
                return operation(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                self.logger.warning(
                    f"Operation failed (attempt {retry_count + 1}/{self.max_retries})",
                    exc_info=True
                )
                
                # Try recovery if strategy exists
                error_type = type(e)
                if error_type in self.recovery_strategies:
                    try:
                        self.logger.info(f"Attempting recovery for {error_type.__name__}")
                        recovery_strategy = self.recovery_strategies[error_type]
                        recovery_strategy(e, *args, **kwargs)
                        retry_count += 1
                        continue
                    except Exception as recovery_error:
                        self.logger.error(f"Recovery failed: {recovery_error}", exc_info=True)
                
                # If no recovery or recovery failed
                if retry_count == self.max_retries - 1:
                    break
                    
                retry_count += 1
                
                # Add delay before retry
                import time
                time.sleep(self.retry_delay * retry_count)
        
        # All retries exhausted
        self.logger.critical(f"Operation failed after {self.max_retries} attempts")
        raise RuntimeError(f"Maximum retry attempts exceeded") from last_exception


class ErrorReporter:
    """Generates comprehensive error reports."""
    
    def __init__(self, logger: Optional[ProteinMDLogger] = None):
        self.logger = logger or ProteinMDLogger()
        
    def create_error_report(self, error: Exception, 
                          context: Optional[Dict[str, Any]] = None,
                          recovery_attempted: bool = False,
                          recovery_successful: bool = False,
                          recovery_details: Optional[str] = None) -> ErrorReport:
        """Create comprehensive error report."""
        
        # Extract error information
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # Determine severity
        if isinstance(error, ProteinMDError):
            severity = error.severity.value
            suggestions = error.suggestions
            error_context = {**(context or {}), **error.context}
        else:
            severity = ErrorSeverity.ERROR.value
            suggestions = []
            error_context = context or {}
            
        # Gather system information
        system_info = self._gather_system_info()
        
        return ErrorReport(
            timestamp=datetime.datetime.now().isoformat(),
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            severity=severity,
            context=error_context,
            suggestions=suggestions,
            system_info=system_info,
            recovery_attempted=recovery_attempted,
            recovery_successful=recovery_successful,
            recovery_details=recovery_details
        )
        
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather system information for error reports."""
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'hostname': platform.node(),
        }
        
        # Add package versions
        try:
            import numpy as np
            system_info['numpy_version'] = np.__version__
        except ImportError:
            pass
            
        try:
            import matplotlib
            system_info['matplotlib_version'] = matplotlib.__version__
        except ImportError:
            pass
            
        # Add memory information
        try:
            import psutil
            memory = psutil.virtual_memory()
            system_info['total_memory_gb'] = round(memory.total / (1024**3), 2)
            system_info['available_memory_gb'] = round(memory.available / (1024**3), 2)
            system_info['memory_percent'] = memory.percent
        except ImportError:
            pass
            
        return system_info
        
    def save_error_report(self, report: ErrorReport, filepath: str):
        """Save error report to file."""
        report_dict = asdict(report)
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
            
        self.logger.info(f"Error report saved to {filepath}")


def robust_operation(max_retries: int = 3, 
                    retry_delay: float = 1.0,
                    logger: Optional[ProteinMDLogger] = None,
                    recovery_manager: Optional[ErrorRecoveryManager] = None):
    """Decorator for making operations robust with error handling."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_logger = logger or ProteinMDLogger()
            
            if recovery_manager:
                return recovery_manager.execute_with_recovery(func, *args, **kwargs)
            else:
                # Simple retry logic
                retry_count = 0
                last_exception = None
                
                while retry_count < max_retries:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        retry_count += 1
                        
                        if retry_count < max_retries:
                            op_logger.warning(
                                f"Operation {func.__name__} failed (attempt {retry_count}/{max_retries})",
                                exc_info=True
                            )
                            import time
                            time.sleep(retry_delay * retry_count)
                        else:
                            op_logger.error(
                                f"Operation {func.__name__} failed after {max_retries} attempts",
                                exc_info=True
                            )
                            raise
                            
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def graceful_degradation(operation_name: str, 
                        fallback_value: Any = None,
                        logger: Optional[ProteinMDLogger] = None):
    """Context manager for graceful degradation of non-critical operations."""
    op_logger = logger or ProteinMDLogger()
    
    try:
        with error_context(op_logger, operation=operation_name):
            yield
    except Exception as e:
        op_logger.warning(
            f"Non-critical operation '{operation_name}' failed, continuing with fallback",
            exc_info=True
        )
        
        # Return fallback value if this is used in an assignment context
        return fallback_value


# Global logger instance
logger = ProteinMDLogger()

# Default recovery strategies
def numerical_instability_recovery(error, simulation, *args, **kwargs):
    """Recovery strategy for numerical instabilities."""
    logger.info("Attempting numerical instability recovery")
    
    # Reduce timestep
    if hasattr(simulation, 'integrator') and hasattr(simulation.integrator, 'timestep'):
        original_timestep = simulation.integrator.timestep
        simulation.integrator.timestep *= 0.5
        logger.info(f"Reduced timestep from {original_timestep} to {simulation.integrator.timestep}")
    
    # Restore from checkpoint if available
    if hasattr(simulation, 'restore_checkpoint'):
        simulation.restore_checkpoint()
        logger.info("Restored simulation from checkpoint")


def memory_error_recovery(error, *args, **kwargs):
    """Recovery strategy for memory errors."""
    logger.info("Attempting memory error recovery")
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Try to clear unnecessary data
    if hasattr(kwargs.get('simulation'), 'clear_cache'):
        kwargs['simulation'].clear_cache()
        logger.info("Cleared simulation cache")


# Create default recovery manager with common strategies
default_recovery_manager = ErrorRecoveryManager(logger)
default_recovery_manager.register_strategy(MemoryError, memory_error_recovery)

# Error reporter instance
error_reporter = ErrorReporter(logger)


def setup_production_logging(log_level: int = logging.INFO, 
                           log_file: str = "proteinmd.log"):
    """Set up production-ready logging configuration."""
    logger.setup_logging(
        level=log_level,
        log_file=log_file,
        console_level=logging.WARNING,  # Only warnings and errors to console
        file_level=logging.DEBUG       # Everything to file
    )
    logger.info("ProteinMD production logging initialized")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing ProteinMD Error Handling & Logging System...")
    
    # Test basic logging
    logger.info("Testing basic logging functionality")
    
    # Test context logging
    with error_context(logger, module="test", operation="example"):
        logger.info("Testing context logging")
        
    # Test error reporting
    try:
        raise SimulationError(
            "Test simulation error",
            context={"timestep": 0.002, "step": 1000},
            suggestions=["Reduce timestep", "Check system stability"]
        )
    except Exception as e:
        report = error_reporter.create_error_report(e)
        print(f"Created error report: {report.error_type}")
        
    # Test graceful degradation
    with graceful_degradation("non_critical_visualization"):
        # This would normally fail, but won't crash the program
        result = 1 / 0  # This will be caught and logged
        
    print("Error handling system test completed successfully!")
