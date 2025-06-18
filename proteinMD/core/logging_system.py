"""
ProteinMD Structured Logging System

This module provides a comprehensive logging framework for the ProteinMD project,
including structured logging, error reporting, performance monitoring, and 
graceful degradation mechanisms.

Features:
- Multi-level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Multiple output handlers (console, file, rotating files)
- Structured log formatting with JSON support
- Performance monitoring integration
- Automatic error reporting with stack traces
- Configuration management
- Integration with ProteinMD exception hierarchy
- Graceful degradation for non-critical errors

Author: ProteinMD Development Team
Date: 2024
"""

import logging
import logging.handlers
import json
import traceback
import sys
import os
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import functools

from .exceptions import (
    ProteinMDError, ProteinMDWarning, SimulationError, StructureError,
    ForceFieldError, ProteinMDIOError, AnalysisError,
    VisualizationError, PerformanceError, ConfigurationError
)


# Log levels mapping
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


@dataclass
class LogContext:
    """Context information for structured logging."""
    module: str
    function: str
    line_number: int
    thread_id: str
    timestamp: str
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class ErrorReport:
    """Structured error report for automatic error reporting."""
    error_id: str
    error_type: str
    error_code: str
    message: str
    timestamp: str
    module: str
    function: str
    line_number: int
    stack_trace: List[str]
    context: Dict[str, Any]
    recovery_attempted: bool
    recovery_successful: bool
    suggestions: List[str]
    user_data: Optional[Dict[str, Any]] = None
    
    def to_json(self) -> str:
        """Convert error report to JSON string."""
        return json.dumps(asdict(self), indent=2, default=str)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error report to dictionary."""
        return asdict(self)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON support."""
    
    def __init__(self, use_json: bool = False, include_context: bool = True):
        self.use_json = use_json
        self.include_context = include_context
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured text or JSON."""
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': threading.current_thread().name
        }
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add custom context if available
        if hasattr(record, 'context') and self.include_context:
            log_data['context'] = record.context
        
        # Add performance metrics if available
        if hasattr(record, 'execution_time'):
            log_data['execution_time'] = record.execution_time
        
        if hasattr(record, 'memory_usage'):
            log_data['memory_usage'] = record.memory_usage
        
        if self.use_json:
            return json.dumps(log_data, default=str)
        else:
            # Human-readable format
            base_msg = f"[{log_data['timestamp']}] {log_data['level']} - {log_data['logger']}: {log_data['message']}"
            if 'exception' in log_data:
                base_msg += f"\nException: {log_data['exception']['type']}: {log_data['exception']['message']}"
            if 'context' in log_data:
                base_msg += f"\nContext: {json.dumps(log_data['context'], default=str)}"
            return base_msg


class PerformanceMonitor:
    """Performance monitoring for logging system."""
    
    def __init__(self):
        self._start_times: Dict[str, float] = {}
        self._memory_snapshots: Dict[str, float] = {}
    
    def start_timing(self, operation_id: str):
        """Start timing an operation."""
        self._start_times[operation_id] = time.time()
        try:
            import psutil
            process = psutil.Process()
            self._memory_snapshots[operation_id] = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            self._memory_snapshots[operation_id] = 0.0
    
    def end_timing(self, operation_id: str) -> tuple[float, float]:
        """End timing an operation and return (execution_time, memory_delta)."""
        execution_time = time.time() - self._start_times.pop(operation_id, time.time())
        
        try:
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = current_memory - self._memory_snapshots.pop(operation_id, current_memory)
        except ImportError:
            memory_delta = 0.0
        
        return execution_time, memory_delta


class ErrorReporter:
    """Automatic error reporting system."""
    
    def __init__(self, logger_name: str = "proteinmd.errors"):
        self.logger = logging.getLogger(logger_name)
        self.error_count = 0
        self.error_history: List[ErrorReport] = []
        self.max_history = 1000
    
    def report_error(self, 
                    error: Exception, 
                    context: Optional[Dict[str, Any]] = None,
                    recovery_attempted: bool = False,
                    recovery_successful: bool = False) -> ErrorReport:
        """Generate and log an error report."""
        self.error_count += 1
        
        # Get stack trace information
        tb = traceback.extract_tb(error.__traceback__) if error.__traceback__ else []
        stack_trace = traceback.format_list(tb)
        
        # Get current frame info for context
        frame = sys._getframe(1)
        
        # Generate error report
        report = ErrorReport(
            error_id=f"ERR_{self.error_count:06d}_{int(time.time())}",
            error_type=type(error).__name__,
            error_code=getattr(error, 'error_code', 'UNKNOWN'),
            message=str(error),
            timestamp=datetime.now(timezone.utc).isoformat(),
            module=frame.f_code.co_filename,
            function=frame.f_code.co_name,
            line_number=frame.f_lineno,
            stack_trace=stack_trace,
            context=context or {},
            recovery_attempted=recovery_attempted,
            recovery_successful=recovery_successful,
            suggestions=getattr(error, 'suggestions', [])
        )
        
        # Add to history
        self.error_history.append(report)
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
        
        # Log the error
        self.logger.error(f"Error Report {report.error_id}: {report.message}", 
                         extra={'error_report': report.to_dict()})
        
        return report
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_history:
            return {'total_errors': 0}
        
        error_types = {}
        recovery_rate = 0
        
        for report in self.error_history:
            error_types[report.error_type] = error_types.get(report.error_type, 0) + 1
            if report.recovery_attempted and report.recovery_successful:
                recovery_rate += 1
        
        return {
            'total_errors': len(self.error_history),
            'error_types': error_types,
            'recovery_rate': recovery_rate / len(self.error_history) if self.error_history else 0,
            'latest_error': self.error_history[-1].to_dict() if self.error_history else None
        }


class GracefulDegradation:
    """Graceful degradation mechanisms for non-critical errors."""
    
    def __init__(self, logger_name: str = "proteinmd.degradation"):
        self.logger = logging.getLogger(logger_name)
        self.fallback_registry: Dict[str, Callable] = {}
        self.degradation_count = 0
    
    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """Register a fallback function for an operation."""
        self.fallback_registry[operation_name] = fallback_func
        self.logger.info(f"Registered fallback for operation: {operation_name}")
    
    def attempt_graceful_degradation(self, 
                                   operation_name: str, 
                                   error: Exception,
                                   *args, **kwargs) -> tuple[bool, Any]:
        """Attempt graceful degradation for a failed operation."""
        if operation_name not in self.fallback_registry:
            self.logger.warning(f"No fallback registered for operation: {operation_name}")
            return False, None
        
        try:
            self.degradation_count += 1
            self.logger.warning(f"Attempting graceful degradation for {operation_name} due to {type(error).__name__}: {error}")
            
            fallback_func = self.fallback_registry[operation_name]
            result = fallback_func(*args, **kwargs)
            
            self.logger.info(f"Graceful degradation successful for {operation_name}")
            return True, result
            
        except Exception as fallback_error:
            self.logger.error(f"Graceful degradation failed for {operation_name}: {fallback_error}")
            return False, None


class ProteinMDLogger:
    """Main logging system for ProteinMD."""
    
    def __init__(self, 
                 name: str = "proteinmd",
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 use_json: bool = False,
                 console_output: bool = True):
        
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(LOG_LEVELS[log_level.upper()])
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Setup formatters
        self.json_formatter = StructuredFormatter(use_json=True)
        self.text_formatter = StructuredFormatter(use_json=False)
        
        # Setup handlers
        if console_output:
            self._setup_console_handler(use_json)
        
        if log_file:
            self._setup_file_handler(log_file, max_file_size, backup_count, use_json)
        
        # Initialize subsystems
        self.performance_monitor = PerformanceMonitor()
        self.error_reporter = ErrorReporter(f"{name}.errors")
        self.graceful_degradation = GracefulDegradation(f"{name}.degradation")
        
        # Prevent propagation to root logger
        self.logger.propagate = False
        
        self.logger.info(f"ProteinMD logging system initialized for {name}")
    
    def _setup_console_handler(self, use_json: bool):
        """Setup console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.json_formatter if use_json else self.text_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self, log_file: str, max_size: int, backup_count: int, use_json: bool):
        """Setup file logging handler with rotation."""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_size, backupCount=backup_count
        )
        file_handler.setFormatter(self.json_formatter if use_json else self.text_formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, context, **kwargs)
    
    def info(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, context, **kwargs)
    
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, context, **kwargs)
    
    def error(self, message: str, context: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None, **kwargs):
        """Log error message."""
        extra = kwargs.copy()
        if exception:
            # Don't override exc_info - let Python's logging handle it
            # Generate error report
            report = self.error_reporter.report_error(exception, context)
            extra['error_report'] = report.to_dict()
        
        self._log(logging.ERROR, message, context, exception=exception, **extra)
    
    def critical(self, message: str, context: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None, **kwargs):
        """Log critical message."""
        extra = kwargs.copy()
        if exception:
            # Don't override exc_info - let Python's logging handle it
            # Generate error report
            report = self.error_reporter.report_error(exception, context)
            extra['error_report'] = report.to_dict()
        
        self._log(logging.CRITICAL, message, context, exception=exception, **extra)
    
    def _log(self, level: int, message: str, context: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None, **kwargs):
        """Internal logging method."""
        extra = kwargs.copy()
        if context:
            extra['context'] = context
        
        # Handle exception properly for Python's logging system
        if exception:
            self.logger.log(level, message, exc_info=(type(exception), exception, exception.__traceback__), extra=extra)
        else:
            self.logger.log(level, message, extra=extra)
    
    @contextmanager
    def performance_context(self, operation_name: str, log_level: str = "DEBUG"):
        """Context manager for performance monitoring."""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        self.performance_monitor.start_timing(operation_id)
        
        try:
            yield
        finally:
            execution_time, memory_delta = self.performance_monitor.end_timing(operation_id)
            
            context = {
                'operation': operation_name,
                'execution_time': execution_time,
                'memory_delta': memory_delta
            }
            
            level = LOG_LEVELS[log_level.upper()]
            self._log(level, f"Performance: {operation_name}", context, 
                     execution_time=execution_time, memory_usage=memory_delta)
    
    def log_exception(self, 
                     exception: Exception, 
                     message: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None,
                     attempt_recovery: bool = False,
                     recovery_operation: Optional[str] = None) -> bool:
        """Log exception with optional recovery attempt."""
        msg = message or f"Exception occurred: {exception}"
        
        recovery_successful = False
        if attempt_recovery and recovery_operation:
            recovery_successful, _ = self.graceful_degradation.attempt_graceful_degradation(
                recovery_operation, exception
            )
        
        # Generate error report
        report = self.error_reporter.report_error(
            exception, context, attempt_recovery, recovery_successful
        )
        
        # Log based on exception type (without triggering another error report)
        if isinstance(exception, ProteinMDError):
            if exception.severity == "critical":
                self._log(logging.CRITICAL, msg, context, exception=exception, error_report=report.to_dict())
            else:
                self._log(logging.ERROR, msg, context, exception=exception, error_report=report.to_dict())
        else:
            self._log(logging.ERROR, msg, context, exception=exception, error_report=report.to_dict())
        
        return recovery_successful
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging and error statistics."""
        return {
            'logger_name': self.name,
            'logger_level': self.logger.level,
            'handlers_count': len(self.logger.handlers),
            'error_statistics': self.error_reporter.get_error_statistics(),
            'degradation_count': self.graceful_degradation.degradation_count
        }


# Decorator for automatic error handling and logging
def logged_function(logger: Optional[ProteinMDLogger] = None, 
                   log_level: str = "DEBUG",
                   log_performance: bool = False,
                   handle_exceptions: bool = True,
                   recovery_operation: Optional[str] = None):
    """Decorator for automatic function logging and error handling."""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or get_logger()
            
            if log_performance:
                with func_logger.performance_context(func.__name__, log_level):
                    try:
                        result = func(*args, **kwargs)
                        func_logger._log(LOG_LEVELS[log_level.upper()], 
                                       f"Function {func.__name__} completed successfully")
                        return result
                    except Exception as e:
                        if handle_exceptions:
                            recovery_successful = func_logger.log_exception(
                                e, f"Exception in function {func.__name__}",
                                {'function': func.__name__, 'args': str(args)[:100]},
                                attempt_recovery=bool(recovery_operation),
                                recovery_operation=recovery_operation
                            )
                            if not recovery_successful:
                                raise
                        else:
                            raise
            else:
                try:
                    result = func(*args, **kwargs)
                    func_logger._log(LOG_LEVELS[log_level.upper()], 
                                   f"Function {func.__name__} completed successfully")
                    return result
                except Exception as e:
                    if handle_exceptions:
                        recovery_successful = func_logger.log_exception(
                            e, f"Exception in function {func.__name__}",
                            {'function': func.__name__, 'args': str(args)[:100]},
                            attempt_recovery=bool(recovery_operation),
                            recovery_operation=recovery_operation
                        )
                        if not recovery_successful:
                            raise
                    else:
                        raise
        
        return wrapper
    return decorator


# Global logger instance
_global_logger: Optional[ProteinMDLogger] = None


def setup_logging(config: Optional[Dict[str, Any]] = None) -> ProteinMDLogger:
    """Setup global ProteinMD logging system."""
    global _global_logger
    
    default_config = {
        'name': 'proteinmd',
        'log_level': 'INFO',
        'log_file': None,
        'max_file_size': 10 * 1024 * 1024,
        'backup_count': 5,
        'use_json': False,
        'console_output': True
    }
    
    if config:
        default_config.update(config)
    
    # Filter out parameters not accepted by ProteinMDLogger
    logger_params = {
        'name', 'log_level', 'log_file', 'max_file_size', 
        'backup_count', 'use_json', 'console_output'
    }
    
    filtered_config = {k: v for k, v in default_config.items() if k in logger_params}
    
    _global_logger = ProteinMDLogger(**filtered_config)
    return _global_logger


def get_logger(name: Optional[str] = None) -> ProteinMDLogger:
    """Get the global logger or create a new one."""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = setup_logging()
    
    if name and name != _global_logger.name:
        # Create a child logger
        return ProteinMDLogger(name=f"{_global_logger.name}.{name}")
    
    return _global_logger


def configure_logging_from_file(config_file: str) -> ProteinMDLogger:
    """Configure logging from a JSON configuration file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return setup_logging(config)
    except Exception as e:
        print(f"Failed to load logging configuration from {config_file}: {e}")
        return setup_logging()


# Example usage and testing functions
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging({
        'log_level': 'DEBUG',
        'log_file': 'proteinmd.log',
        'use_json': False,
        'console_output': True
    })
    
    # Test basic logging
    logger.info("ProteinMD logging system test started")
    logger.debug("Debug message with context", {'test': True, 'value': 42})
    logger.warning("Warning message")
    
    # Test performance monitoring
    with logger.performance_context("test_operation"):
        time.sleep(0.1)  # Simulate some work
    
    # Test error handling
    try:
        raise SimulationError("Test simulation error", "SIM_001")
    except Exception as e:
        logger.log_exception(e, "Test exception handling")
    
    # Test graceful degradation
    def fallback_function(*args, **kwargs):
        return "fallback_result"
    
    logger.graceful_degradation.register_fallback("test_operation", fallback_function)
    
    try:
        raise ValueError("Test error for graceful degradation")
    except Exception as e:
        success, result = logger.graceful_degradation.attempt_graceful_degradation(
            "test_operation", e
        )
        logger.info(f"Graceful degradation: {success}, result: {result}")
    
    # Print statistics
    stats = logger.get_statistics()
    logger.info("Logging statistics", stats)
    
    logger.info("ProteinMD logging system test completed")
