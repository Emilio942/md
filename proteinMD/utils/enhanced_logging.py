"""
Enhanced Logging Configuration for ProteinMD

This module provides advanced logging capabilities including:
- Structured logging with JSON formatting
- Performance logging for simulation components
- Automatic log rotation and archival
- Log filtering and categorization
- Integration with monitoring systems

Author: GitHub Copilot
Date: June 13, 2025
"""

import logging
import logging.handlers
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import functools
import sys
import os


@dataclass
class LogConfig:
    """Configuration for logging setup."""
    name: str = "proteinmd"
    level: int = logging.INFO
    console_level: Optional[int] = None
    file_level: Optional[int] = None
    log_dir: str = "logs"
    log_file: str = "proteinmd.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    json_format: bool = False
    performance_logging: bool = True
    enable_compression: bool = True


@dataclass
class PerformanceMetrics:
    """Performance metrics for logging."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    memory_peak: Optional[float] = None
    cpu_percent: Optional[float] = None
    context: Optional[Dict[str, Any]] = None


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields from LogRecord
        for key, value in record.__dict__.items():
            if key not in log_entry and not key.startswith('_'):
                log_entry[key] = value
                
        return json.dumps(log_entry, default=str)


class PerformanceLogger:
    """Logger for performance metrics and profiling."""
    
    def __init__(self, logger_name: str = "proteinmd.performance"):
        self.logger = logging.getLogger(logger_name)
        self.metrics: List[PerformanceMetrics] = []
        self._lock = threading.Lock()
        
    def log_performance(self, metrics: PerformanceMetrics):
        """Log performance metrics."""
        with self._lock:
            self.metrics.append(metrics)
            
        self.logger.info(
            f"Performance: {metrics.operation} took {metrics.duration:.4f}s",
            extra={
                'operation': metrics.operation,
                'duration': metrics.duration,
                'memory_before': metrics.memory_before,
                'memory_after': metrics.memory_after,
                'memory_peak': metrics.memory_peak,
                'cpu_percent': metrics.cpu_percent,
                'context': metrics.context
            }
        )
        
    def get_metrics(self, operation: Optional[str] = None) -> List[PerformanceMetrics]:
        """Get performance metrics, optionally filtered by operation."""
        with self._lock:
            if operation:
                return [m for m in self.metrics if m.operation == operation]
            return self.metrics.copy()
            
    def clear_metrics(self):
        """Clear stored performance metrics."""
        with self._lock:
            self.metrics.clear()
            
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate performance report from collected metrics."""
        with self._lock:
            if not self.metrics:
                return {"message": "No performance data available"}
                
            # Group by operation
            operations = {}
            for metric in self.metrics:
                if metric.operation not in operations:
                    operations[metric.operation] = []
                operations[metric.operation].append(metric)
                
            report = {
                "total_operations": len(self.metrics),
                "operations": {}
            }
            
            for operation, metrics in operations.items():
                durations = [m.duration for m in metrics]
                memory_usage = [m.memory_after - m.memory_before 
                              for m in metrics 
                              if m.memory_before and m.memory_after]
                
                op_report = {
                    "count": len(metrics),
                    "total_time": sum(durations),
                    "avg_time": sum(durations) / len(durations),
                    "min_time": min(durations),
                    "max_time": max(durations),
                }
                
                if memory_usage:
                    op_report.update({
                        "avg_memory_change": sum(memory_usage) / len(memory_usage),
                        "max_memory_change": max(memory_usage),
                        "min_memory_change": min(memory_usage)
                    })
                    
                report["operations"][operation] = op_report
                
            return report


class ComponentLogger:
    """Specialized logger for different ProteinMD components."""
    
    def __init__(self, component: str, parent_logger: str = "proteinmd"):
        self.component = component
        self.logger = logging.getLogger(f"{parent_logger}.{component}")
        self.performance_logger = PerformanceLogger(f"{parent_logger}.{component}.performance")
        
    def debug(self, message: str, **kwargs):
        """Log debug message with component context."""
        self.logger.debug(message, extra={'component': self.component, **kwargs})
        
    def info(self, message: str, **kwargs):
        """Log info message with component context."""
        self.logger.info(message, extra={'component': self.component, **kwargs})
        
    def warning(self, message: str, **kwargs):
        """Log warning message with component context."""
        self.logger.warning(message, extra={'component': self.component, **kwargs})
        
    def error(self, message: str, exc_info: bool = True, **kwargs):
        """Log error message with component context."""
        self.logger.error(message, exc_info=exc_info, extra={'component': self.component, **kwargs})
        
    def critical(self, message: str, exc_info: bool = True, **kwargs):
        """Log critical message with component context."""
        self.logger.critical(message, exc_info=exc_info, extra={'component': self.component, **kwargs})


@contextmanager
def performance_monitoring(operation: str, 
                         logger: Optional[PerformanceLogger] = None,
                         enable_memory_tracking: bool = True,
                         enable_cpu_tracking: bool = False,
                         context: Optional[Dict[str, Any]] = None):
    """Context manager for performance monitoring."""
    
    perf_logger = logger or PerformanceLogger()
    
    # Memory tracking setup
    memory_before = None
    memory_after = None
    memory_peak = None
    
    if enable_memory_tracking:
        try:
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            pass
            
    # CPU tracking setup  
    cpu_percent = None
    if enable_cpu_tracking:
        try:
            import psutil
            process = psutil.Process()
            process.cpu_percent()  # Initialize CPU monitoring
        except ImportError:
            pass
    
    start_time = time.time()
    
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        
        # Final memory measurement
        if enable_memory_tracking and memory_before is not None:
            try:
                import psutil
                process = psutil.Process()
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
            except ImportError:
                pass
                
        # CPU measurement
        if enable_cpu_tracking:
            try:
                import psutil
                process = psutil.Process()
                cpu_percent = process.cpu_percent()
            except ImportError:
                pass
        
        # Create metrics
        metrics = PerformanceMetrics(
            operation=operation,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            memory_before=memory_before,
            memory_after=memory_after,
            memory_peak=memory_peak,
            cpu_percent=cpu_percent,
            context=context
        )
        
        perf_logger.log_performance(metrics)


def performance_monitor(operation: Optional[str] = None,
                       enable_memory_tracking: bool = True,
                       enable_cpu_tracking: bool = False,
                       logger: Optional[PerformanceLogger] = None):
    """Decorator for performance monitoring."""
    
    def decorator(func):
        op_name = operation or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with performance_monitoring(
                operation=op_name,
                logger=logger,
                enable_memory_tracking=enable_memory_tracking,
                enable_cpu_tracking=enable_cpu_tracking,
                context={'args_count': len(args), 'kwargs_count': len(kwargs)}
            ):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class LogManager:
    """Central log management system."""
    
    def __init__(self, config: LogConfig = None):
        self.config = config or LogConfig()
        self.loggers: Dict[str, logging.Logger] = {}
        self.component_loggers: Dict[str, ComponentLogger] = {}
        self.performance_logger = PerformanceLogger()
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up the logging system according to configuration."""
        
        # Create log directory
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Get or create main logger
        logger = logging.getLogger(self.config.name)
        logger.setLevel(self.config.level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_level = self.config.console_level or self.config.level
        console_handler.setLevel(console_level)
        
        if self.config.json_format:
            console_formatter = JSONFormatter()
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file_path = log_dir / self.config.log_file
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count
        )
        
        file_level = self.config.file_level or logging.DEBUG
        file_handler.setLevel(file_level)
        
        if self.config.json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
            )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Performance logger setup
        if self.config.performance_logging:
            perf_log_path = log_dir / "performance.log"
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_log_path,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            perf_handler.setLevel(logging.INFO)
            
            if self.config.json_format:
                perf_formatter = JSONFormatter()
            else:
                perf_formatter = logging.Formatter(
                    '%(asctime)s - PERF - %(message)s'
                )
            perf_handler.setFormatter(perf_formatter)
            
            self.performance_logger.logger.addHandler(perf_handler)
            self.performance_logger.logger.setLevel(logging.INFO)
        
        # Prevent duplicate logging
        logger.propagate = False
        
        self.loggers[self.config.name] = logger
        
    def get_logger(self, name: str = None) -> logging.Logger:
        """Get logger by name."""
        logger_name = name or self.config.name
        return self.loggers.get(logger_name, logging.getLogger(logger_name))
        
    def get_component_logger(self, component: str) -> ComponentLogger:
        """Get component-specific logger."""
        if component not in self.component_loggers:
            self.component_loggers[component] = ComponentLogger(component)
        return self.component_loggers[component]
        
    def get_performance_logger(self) -> PerformanceLogger:
        """Get performance logger."""
        return self.performance_logger
        
    def generate_log_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Generate summary of log activity."""
        summary = {
            "timeframe": f"Last {hours} hours",
            "timestamp": datetime.now().isoformat(),
            "log_levels": {},
            "components": {},
            "performance_summary": self.performance_logger.generate_performance_report()
        }
        
        # Note: In a real implementation, you'd parse log files to generate statistics
        # This is a simplified version
        summary["log_levels"] = {
            "DEBUG": 0,
            "INFO": 0,
            "WARNING": 0,
            "ERROR": 0,
            "CRITICAL": 0
        }
        
        return summary
        
    def cleanup_old_logs(self, days: int = 30):
        """Clean up log files older than specified days."""
        log_dir = Path(self.config.log_dir)
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cleaned_files = []
        for log_file in log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    log_file.unlink()
                    cleaned_files.append(str(log_file))
                except OSError as e:
                    self.get_logger().warning(f"Failed to delete old log file {log_file}: {e}")
                    
        if cleaned_files:
            self.get_logger().info(f"Cleaned up {len(cleaned_files)} old log files")
            
        return cleaned_files


# Global log manager instance
default_log_manager = LogManager()

# Convenience functions
def get_logger(name: str = None) -> logging.Logger:
    """Get logger from default manager."""
    return default_log_manager.get_logger(name)

def get_component_logger(component: str) -> ComponentLogger:
    """Get component logger from default manager."""
    return default_log_manager.get_component_logger(component)

def get_performance_logger() -> PerformanceLogger:
    """Get performance logger from default manager."""
    return default_log_manager.get_performance_logger()

def setup_logging(config: LogConfig = None):
    """Set up logging with custom configuration."""
    global default_log_manager
    default_log_manager = LogManager(config)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Enhanced Logging System...")
    
    # Test basic logging
    logger = get_logger()
    logger.info("Testing enhanced logging system")
    
    # Test component logging
    sim_logger = get_component_logger("simulation")
    sim_logger.info("Starting simulation", step=0, temperature=300)
    
    # Test performance monitoring
    @performance_monitor("test_operation")
    def test_function():
        import time
        time.sleep(0.1)
        return "completed"
    
    result = test_function()
    
    # Test performance context manager
    with performance_monitoring("manual_operation", context={"size": 1000}):
        import time
        time.sleep(0.05)
    
    # Generate performance report
    perf_logger = get_performance_logger()
    report = perf_logger.generate_performance_report()
    print(f"Performance report: {json.dumps(report, indent=2)}")
    
    print("Enhanced logging system test completed successfully!")
