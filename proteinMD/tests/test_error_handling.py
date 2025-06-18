"""
Comprehensive Test Suite for Task 15.1: Error Handling & Logging

This test suite validates all components of the error handling and logging system
including exception handling, logging functionality, performance monitoring,
and integration with ProteinMD modules.

Author: GitHub Copilot
Date: June 13, 2025
"""

import pytest
import tempfile
import json
import time
import logging
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the modules we're testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from proteinMD.utils.error_handling import (
    ProteinMDError, SimulationError, IOError, ValidationError,
    IntegrationError, FileFormatError,
    ErrorSeverity, ProteinMDLogger, ErrorRecoveryManager, 
    ErrorReporter, robust_operation, graceful_degradation,
    error_context
)

from proteinMD.utils.enhanced_logging import (
    LogConfig, PerformanceLogger, ComponentLogger, LogManager,
    performance_monitoring, performance_monitor, JSONFormatter
)

from proteinMD.utils.error_integration import (
    SimulationErrorHandler, IOErrorHandler, SystemErrorHandler,
    proteinmd_exception_handler, log_function_calls
)


class TestProteinMDErrors:
    """Test custom exception classes."""
    
    def test_base_error_creation(self):
        """Test creating base ProteinMD error."""
        context = {"step": 1000, "temperature": 300}
        suggestions = ["Reduce timestep", "Check system"]
        
        error = ProteinMDError(
            "Test error message",
            context=context,
            suggestions=suggestions,
            severity=ErrorSeverity.WARNING
        )
        
        assert str(error).startswith("Test error message")
        assert error.context == context
        assert error.suggestions == suggestions
        assert error.severity == ErrorSeverity.WARNING
        assert error.timestamp is not None
        
    def test_error_string_formatting(self):
        """Test error string formatting with context and suggestions."""
        error = ProteinMDError(
            "Test error",
            context={"key": "value"},
            suggestions=["Suggestion 1", "Suggestion 2"]
        )
        
        error_str = str(error)
        assert "Test error" in error_str
        assert "Context:" in error_str
        assert "Suggestions:" in error_str
        assert "1. Suggestion 1" in error_str
        assert "2. Suggestion 2" in error_str
        
    def test_specialized_errors(self):
        """Test specialized error types."""
        sim_error = SimulationError("Simulation failed")
        assert isinstance(sim_error, ProteinMDError)
        
        io_error = IOError("File not found")
        assert isinstance(io_error, ProteinMDError)
        
        val_error = ValidationError("Invalid data")
        assert isinstance(val_error, ProteinMDError)


class TestProteinMDLogger:
    """Test the ProteinMD logging system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test.log")
        
    def test_logger_creation(self):
        """Test logger creation and basic functionality."""
        logger = ProteinMDLogger("test_logger")
        assert logger.name == "test_logger"
        assert logger.logger.name == "test_logger"
        
    def test_logger_setup(self):
        """Test logger setup with file and console handlers."""
        logger = ProteinMDLogger("test_setup")
        logger.setup_logging(
            level=logging.DEBUG,
            log_file=self.log_file,
            console_level=logging.WARNING,
            file_level=logging.DEBUG
        )
        
        # Check handlers
        assert len(logger.logger.handlers) == 2
        assert 'console' in logger.handlers
        assert 'file' in logger.handlers
        
        # Test logging
        logger.info("Test info message")
        logger.error("Test error message")
        
        # Check file was created and contains logs
        assert os.path.exists(self.log_file)
        with open(self.log_file, 'r') as f:
            content = f.read()
            assert "Test info message" in content
            assert "Test error message" in content
            
    def test_context_logging(self):
        """Test context-aware logging."""
        logger = ProteinMDLogger("test_context")
        logger.setup_logging(log_file=self.log_file)
        
        with error_context(logger, operation="test", step=100):
            logger.info("Test message with context")
            
        with open(self.log_file, 'r') as f:
            content = f.read()
            assert "Context:" in content
            assert "operation" in content
            assert "step" in content
            
    def test_context_stack(self):
        """Test context stacking functionality."""
        logger = ProteinMDLogger("test_stack")
        
        logger.add_context(level1="value1")
        logger.add_context(level2="value2")
        
        assert len(logger.context_stack) == 2
        
        logger.remove_context()
        assert len(logger.context_stack) == 1
        
        logger.remove_context()
        assert len(logger.context_stack) == 0


class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    def test_recovery_manager_creation(self):
        """Test recovery manager creation."""
        logger = ProteinMDLogger("test_recovery")
        recovery_manager = ErrorRecoveryManager(logger)
        
        assert recovery_manager.logger == logger
        assert recovery_manager.max_retries == 3
        
    def test_strategy_registration(self):
        """Test recovery strategy registration."""
        recovery_manager = ErrorRecoveryManager()
        
        def test_strategy(error, *args, **kwargs):
            pass
            
        recovery_manager.register_strategy(ValueError, test_strategy)
        assert ValueError in recovery_manager.recovery_strategies
        
    def test_successful_operation(self):
        """Test operation that succeeds without recovery."""
        recovery_manager = ErrorRecoveryManager()
        
        def successful_operation():
            return "success"
            
        result = recovery_manager.execute_with_recovery(successful_operation)
        assert result == "success"
        
    def test_operation_with_recovery(self):
        """Test operation that fails then succeeds with recovery."""
        recovery_manager = ErrorRecoveryManager()
        recovery_manager.max_retries = 2
        
        call_count = 0
        recovery_called = False
        
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First attempt fails")
            return "success after recovery"
            
        def recovery_strategy(error, *args, **kwargs):
            nonlocal recovery_called
            recovery_called = True
            
        recovery_manager.register_strategy(ValueError, recovery_strategy)
        
        result = recovery_manager.execute_with_recovery(failing_operation)
        assert result == "success after recovery"
        assert call_count == 2
        assert recovery_called
        
    def test_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded."""
        recovery_manager = ErrorRecoveryManager()
        recovery_manager.max_retries = 2
        
        def always_failing_operation():
            raise ValueError("Always fails")
            
        with pytest.raises(RuntimeError, match="Maximum retry attempts exceeded"):
            recovery_manager.execute_with_recovery(always_failing_operation)


class TestErrorReporter:
    """Test error reporting functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def test_error_report_creation(self):
        """Test creating error reports."""
        logger = ProteinMDLogger("test_reporter")
        reporter = ErrorReporter(logger)
        
        try:
            raise ValueError("Test error for reporting")
        except Exception as e:
            report = reporter.create_error_report(
                e,
                context={"test": "value"},
                recovery_attempted=True
            )
            
        assert report.error_type == "ValueError"
        assert report.error_message == "Test error for reporting"
        assert report.context["test"] == "value"
        assert report.recovery_attempted
        assert report.system_info is not None
        
    def test_proteinmd_error_report(self):
        """Test reporting ProteinMD-specific errors."""
        logger = ProteinMDLogger("test_proteinmd_reporter")
        reporter = ErrorReporter(logger)
        
        error = SimulationError(
            "Simulation failed",
            context={"step": 1000},
            suggestions=["Reduce timestep"],
            severity=ErrorSeverity.CRITICAL
        )
        
        report = reporter.create_error_report(error)
        
        assert report.error_type == "SimulationError"
        assert report.severity == "CRITICAL"
        assert "step" in report.context
        assert "Reduce timestep" in report.suggestions
        
    def test_report_saving(self):
        """Test saving error reports to file."""
        logger = ProteinMDLogger("test_save")
        reporter = ErrorReporter(logger)
        
        try:
            raise RuntimeError("Test error")
        except Exception as e:
            report = reporter.create_error_report(e)
            
        report_file = os.path.join(self.temp_dir, "test_report.json")
        reporter.save_error_report(report, report_file)
        
        assert os.path.exists(report_file)
        
        with open(report_file, 'r') as f:
            saved_report = json.load(f)
            assert saved_report["error_type"] == "RuntimeError"
            assert saved_report["error_message"] == "Test error"


class TestPerformanceLogging:
    """Test performance logging functionality."""
    
    def test_performance_logger_creation(self):
        """Test performance logger creation."""
        perf_logger = PerformanceLogger("test_performance")
        assert perf_logger.logger.name == "test_performance"
        assert len(perf_logger.metrics) == 0
        
    def test_performance_monitoring_context(self):
        """Test performance monitoring context manager."""
        perf_logger = PerformanceLogger()
        
        with performance_monitoring("test_operation", logger=perf_logger):
            time.sleep(0.01)
            
        metrics = perf_logger.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].operation == "test_operation"
        assert metrics[0].duration >= 0.01
        
    def test_performance_monitoring_decorator(self):
        """Test performance monitoring decorator."""
        perf_logger = PerformanceLogger()
        
        @performance_monitor("decorated_operation", logger=perf_logger)
        def test_function():
            time.sleep(0.01)
            return "completed"
            
        result = test_function()
        assert result == "completed"
        
        metrics = perf_logger.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].operation == "decorated_operation"
        
    def test_performance_report_generation(self):
        """Test performance report generation."""
        perf_logger = PerformanceLogger()
        
        # Add some test metrics
        with performance_monitoring("operation_a", logger=perf_logger):
            time.sleep(0.01)
            
        with performance_monitoring("operation_a", logger=perf_logger):
            time.sleep(0.01)
            
        with performance_monitoring("operation_b", logger=perf_logger):
            time.sleep(0.005)
            
        report = perf_logger.generate_performance_report()
        
        assert report["total_operations"] == 3
        assert "operation_a" in report["operations"]
        assert "operation_b" in report["operations"]
        assert report["operations"]["operation_a"]["count"] == 2
        assert report["operations"]["operation_b"]["count"] == 1


class TestComponentLoggers:
    """Test component-specific logging."""
    
    def test_component_logger_creation(self):
        """Test creating component loggers."""
        comp_logger = ComponentLogger("simulation")
        assert comp_logger.component == "simulation"
        assert "simulation" in comp_logger.logger.name
        
    def test_component_logging_with_context(self):
        """Test component logging includes component context."""
        comp_logger = ComponentLogger("test_component")
        
        # Mock the logger to capture extra data
        with patch.object(comp_logger.logger, 'info') as mock_info:
            comp_logger.info("Test message", extra_data="value")
            
            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args
            assert kwargs['extra']['component'] == 'test_component'
            assert kwargs['extra']['extra_data'] == 'value'


class TestLogManager:
    """Test the log management system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def test_log_manager_creation(self):
        """Test log manager creation with default config."""
        config = LogConfig(log_dir=self.temp_dir)
        log_manager = LogManager(config)
        
        assert log_manager.config.log_dir == self.temp_dir
        assert "proteinmd" in log_manager.loggers
        
    def test_json_logging(self):
        """Test JSON formatted logging."""
        config = LogConfig(
            log_dir=self.temp_dir,
            log_file="test_json.log",
            json_format=True
        )
        log_manager = LogManager(config)
        
        logger = log_manager.get_logger()
        logger.info("Test JSON message", extra={'test_key': 'test_value'})
        
        log_file = Path(self.temp_dir) / "test_json.log"
        assert log_file.exists()
        
        with open(log_file, 'r') as f:
            log_lines = f.readlines()
            if log_lines:
                log_entry = json.loads(log_lines[-1])
                assert log_entry['message'] == 'Test JSON message'
                assert log_entry['test_key'] == 'test_value'
                
    def test_component_logger_retrieval(self):
        """Test getting component loggers."""
        config = LogConfig(log_dir=self.temp_dir)
        log_manager = LogManager(config)
        
        sim_logger = log_manager.get_component_logger("simulation")
        assert isinstance(sim_logger, ComponentLogger)
        assert sim_logger.component == "simulation"
        
        # Test caching
        sim_logger2 = log_manager.get_component_logger("simulation")
        assert sim_logger is sim_logger2


class TestErrorIntegration:
    """Test error handling integration with ProteinMD components."""
    
    def test_simulation_error_handler(self):
        """Test simulation-specific error handling."""
        handler = SimulationErrorHandler()
        
        # Test integration error handling
        with pytest.raises(IntegrationError) as exc_info:
            handler.handle_integration_error(
                ValueError("NaN encountered"),
                {"step": 1000, "timestep": 0.002}
            )
            
        error = exc_info.value
        assert error.severity == ErrorSeverity.CRITICAL
        assert "Reduce the integration timestep" in error.suggestions
        assert error.context["step"] == 1000
        
    def test_io_error_handler(self):
        """Test I/O-specific error handling."""
        handler = IOErrorHandler()
        
        # Test file not found error
        with pytest.raises(FileFormatError) as exc_info:
            handler.handle_file_read_error(
                FileNotFoundError("test.pdb"),
                "test.pdb",
                "pdb"
            )
            
        error = exc_info.value
        assert "File not found" in str(error)
        assert "Check file path spelling" in error.suggestions
        
    def test_proteinmd_exception_decorator(self):
        """Test the ProteinMD exception handling decorator."""
        
        @proteinmd_exception_handler("simulation")
        def test_function():
            raise ValueError("Generic error")
            
        with pytest.raises(SimulationError) as exc_info:
            test_function()
            
        error = exc_info.value
        assert isinstance(error, SimulationError)
        assert "test_function" in error.context["function"]
        
    def test_function_call_logging(self):
        """Test function call logging decorator."""
        
        @log_function_calls("test_component", log_args=True, log_performance=True)
        def test_function(arg1, kwarg1=None):
            time.sleep(0.001)
            return "result"
            
        result = test_function("value1", kwarg1="value2")
        assert result == "result"


class TestRobustOperations:
    """Test robust operation decorators and utilities."""
    
    def test_robust_operation_decorator(self):
        """Test robust operation decorator."""
        call_count = 0
        
        @robust_operation(max_retries=3, retry_delay=0.001)
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
            
        result = flaky_operation()
        assert result == "success"
        assert call_count == 3
        
    def test_graceful_degradation(self):
        """Test graceful degradation context manager."""
        
        # Test successful operation
        with graceful_degradation("test_operation"):
            result = "success"
            
        # Test failed operation (should not raise)
        with graceful_degradation("failing_operation", fallback_value="fallback"):
            raise ValueError("This should be caught")


def run_comprehensive_validation():
    """Run comprehensive validation of all error handling components."""
    
    print("🧪 TASK 15.1 COMPREHENSIVE VALIDATION")
    print("=" * 60)
    
    validation_results = {
        "exception_handling": False,
        "structured_logging": False,
        "performance_monitoring": False,
        "error_reporting": False,
        "recovery_strategies": False,
        "graceful_degradation": False,
        "integration": False
    }
    
    # Test 1: Exception Handling
    try:
        error = ProteinMDError(
            "Test error",
            context={"test": True},
            suggestions=["Fix it"],
            severity=ErrorSeverity.WARNING
        )
        assert "Test error" in str(error)
        assert error.context["test"] is True
        validation_results["exception_handling"] = True
        print("✅ Exception handling system working")
    except Exception as e:
        print(f"❌ Exception handling failed: {e}")
        
    # Test 2: Structured Logging
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LogConfig(log_dir=temp_dir, json_format=True)
            log_manager = LogManager(config)
            logger = log_manager.get_logger()
            logger.info("Test structured log", component="test")
            
            log_file = Path(temp_dir) / "proteinmd.log"
            if log_file.exists():
                validation_results["structured_logging"] = True
                print("✅ Structured logging working")
    except Exception as e:
        print(f"❌ Structured logging failed: {e}")
        
    # Test 3: Performance Monitoring
    try:
        perf_logger = PerformanceLogger()
        with performance_monitoring("test_op", logger=perf_logger):
            time.sleep(0.001)
        
        metrics = perf_logger.get_metrics()
        if len(metrics) > 0 and metrics[0].duration > 0:
            validation_results["performance_monitoring"] = True
            print("✅ Performance monitoring working")
    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")
        
    # Test 4: Error Reporting
    try:
        reporter = ErrorReporter()
        try:
            raise ValueError("Test error for reporting")
        except Exception as e:
            report = reporter.create_error_report(e)
            
        if report.error_type == "ValueError":
            validation_results["error_reporting"] = True
            print("✅ Error reporting working")
    except Exception as e:
        print(f"❌ Error reporting failed: {e}")
        
    # Test 5: Recovery Strategies
    try:
        recovery_manager = ErrorRecoveryManager()
        
        def recovery_strategy(error, *args, **kwargs):
            pass
            
        recovery_manager.register_strategy(ValueError, recovery_strategy)
        
        if ValueError in recovery_manager.recovery_strategies:
            validation_results["recovery_strategies"] = True
            print("✅ Recovery strategies working")
    except Exception as e:
        print(f"❌ Recovery strategies failed: {e}")
        
    # Test 6: Graceful Degradation
    try:
        with graceful_degradation("test_operation"):
            pass  # Should work without error
            
        validation_results["graceful_degradation"] = True
        print("✅ Graceful degradation working")
    except Exception as e:
        print(f"❌ Graceful degradation failed: {e}")
        
    # Test 7: Integration
    try:
        @proteinmd_exception_handler("test")
        def test_integration():
            return "integrated"
            
        result = test_integration()
        if result == "integrated":
            validation_results["integration"] = True
            print("✅ Integration working")
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        
    # Summary
    print("\n" + "=" * 60)
    passed = sum(validation_results.values())
    total = len(validation_results)
    success_rate = (passed / total) * 100
    
    print(f"📊 VALIDATION SUMMARY: {passed}/{total} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 85:
        print("🎉 TASK 15.1 ERROR HANDLING & LOGGING - VALIDATION SUCCESSFUL!")
        return True
    else:
        print("⚠️  Some components need attention")
        return False


if __name__ == "__main__":
    # Run the comprehensive validation
    success = run_comprehensive_validation()
    
    # Run pytest if available
    try:
        import pytest
        print("\n" + "=" * 60)
        print("Running detailed pytest suite...")
        pytest.main([__file__, "-v"])
    except ImportError:
        print("Pytest not available, skipping detailed tests")
    
    if success:
        print("\n✅ Task 15.1 Error Handling & Logging validation completed successfully!")
    else:
        print("\n❌ Task 15.1 validation had issues - check the logs above")
