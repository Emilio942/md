"""
ProteinMD Exception Hierarchy
Comprehensive exception classes for robust error handling across all modules.

This module provides a structured exception hierarchy for the ProteinMD project,
enabling precise error handling and meaningful error messages.
"""

import traceback
from typing import Optional, Dict, Any, List
import json
from datetime import datetime


class ProteinMDError(Exception):
    """Base exception class for all ProteinMD-specific errors.
    
    Provides structured error information including:
    - Error code for programmatic handling
    - User-friendly messages
    - Technical details for debugging
    - Context information for troubleshooting
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "GENERAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        recoverable: bool = False,
        severity: str = "error",
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.suggestions = suggestions or []
        self.recoverable = recoverable
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now()
        self.stack_trace = traceback.format_exc()
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/reporting."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity,
            "context": self.context,
            "details": self.details,
            "suggestions": self.suggestions,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
            "stack_trace": self.stack_trace
        }
    
    def to_json(self) -> str:
        """Convert exception to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# SIMULATION ERRORS
# =============================================================================

class SimulationError(ProteinMDError):
    """Base class for simulation-related errors."""
    pass


class SimulationSetupError(SimulationError):
    """Errors during simulation setup and initialization."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="SIM_SETUP_ERROR",
            **kwargs
        )


class SimulationRuntimeError(SimulationError):
    """Errors during simulation execution."""
    
    def __init__(self, message: str, step: Optional[int] = None, **kwargs):
        details = kwargs.get('details', {})
        if step is not None:
            details['simulation_step'] = step
        
        super().__init__(
            message,
            error_code="SIM_RUNTIME_ERROR",
            details=details,
            **kwargs
        )


class ConvergenceError(SimulationError):
    """Simulation convergence issues."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="CONVERGENCE_ERROR",
            suggestions=[
                "Try reducing the time step",
                "Check force field parameters",
                "Verify initial structure quality",
                "Consider different integration algorithm"
            ],
            **kwargs
        )


# =============================================================================
# STRUCTURE ERRORS
# =============================================================================

class StructureError(ProteinMDError):
    """Base class for protein structure-related errors."""
    pass


class PDBParsingError(StructureError):
    """Errors during PDB file parsing."""
    
    def __init__(self, message: str, filename: Optional[str] = None, line_number: Optional[int] = None, **kwargs):
        details = kwargs.get('details', {})
        if filename:
            details['filename'] = filename
        if line_number:
            details['line_number'] = line_number
        
        super().__init__(
            message,
            error_code="PDB_PARSING_ERROR",
            details=details,
            suggestions=[
                "Check PDB file format compliance",
                "Verify file is not corrupted",
                "Try cleaning the PDB file",
                "Use alternative structure format"
            ],
            **kwargs
        )


class StructureValidationError(StructureError):
    """Structure validation and quality check errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="STRUCTURE_VALIDATION_ERROR",
            suggestions=[
                "Check for missing atoms or residues",
                "Verify coordinate ranges",
                "Consider structure minimization",
                "Review experimental resolution"
            ],
            **kwargs
        )


class TopologyError(StructureError):
    """Molecular topology-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="TOPOLOGY_ERROR",
            **kwargs
        )


# =============================================================================
# FORCE FIELD ERRORS
# =============================================================================

class ForceFieldError(ProteinMDError):
    """Base class for force field-related errors."""
    pass


class ParameterError(ForceFieldError):
    """Missing or invalid force field parameters."""
    
    def __init__(self, message: str, parameter_type: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if parameter_type:
            details['parameter_type'] = parameter_type
        
        super().__init__(
            message,
            error_code="PARAMETER_ERROR",
            details=details,
            suggestions=[
                "Check force field parameter files",
                "Verify atom types are supported",
                "Consider alternative force field",
                "Add custom parameters if needed"
            ],
            **kwargs
        )


class ForceFieldValidationError(ForceFieldError):
    """Force field validation and consistency errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="FF_VALIDATION_ERROR",
            **kwargs
        )


# =============================================================================
# I/O ERRORS
# =============================================================================

class ProteinMDIOError(ProteinMDError):
    """Base class for input/output-related errors."""
    pass


class FileFormatError(ProteinMDIOError):
    """Unsupported or invalid file format errors."""
    
    def __init__(self, message: str, format_type: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if format_type:
            details['format_type'] = format_type
        
        super().__init__(
            message,
            error_code="FILE_FORMAT_ERROR",
            details=details,
            suggestions=[
                "Check file extension matches content",
                "Verify file format specification",
                "Try format conversion tools",
                "Use supported file formats"
            ],
            **kwargs
        )


class TrajectoryError(ProteinMDIOError):
    """Trajectory file handling errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="TRAJECTORY_ERROR",
            **kwargs
        )


# =============================================================================
# ANALYSIS ERRORS
# =============================================================================

class AnalysisError(ProteinMDError):
    """Base class for analysis-related errors."""
    pass


class InsufficientDataError(AnalysisError):
    """Insufficient data for meaningful analysis."""
    
    def __init__(self, message: str, required_frames: Optional[int] = None, available_frames: Optional[int] = None, **kwargs):
        details = kwargs.get('details', {})
        if required_frames:
            details['required_frames'] = required_frames
        if available_frames:
            details['available_frames'] = available_frames
        
        super().__init__(
            message,
            error_code="INSUFFICIENT_DATA_ERROR",
            details=details,
            suggestions=[
                "Run longer simulation",
                "Reduce analysis time step",
                "Check trajectory completeness",
                "Adjust analysis parameters"
            ],
            recoverable=True,
            **kwargs
        )


class CalculationError(AnalysisError):
    """Numerical calculation errors in analysis."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="CALCULATION_ERROR",
            **kwargs
        )


# =============================================================================
# VISUALIZATION ERRORS
# =============================================================================

class VisualizationError(ProteinMDError):
    """Base class for visualization-related errors."""
    pass


class RenderingError(VisualizationError):
    """3D rendering and graphics errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="RENDERING_ERROR",
            suggestions=[
                "Check graphics drivers",
                "Verify OpenGL support",
                "Try different rendering backend",
                "Reduce visualization complexity"
            ],
            **kwargs
        )


class PlotGenerationError(VisualizationError):
    """Plot and figure generation errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="PLOT_GENERATION_ERROR",
            recoverable=True,
            **kwargs
        )


# =============================================================================
# PERFORMANCE ERRORS
# =============================================================================

class PerformanceError(ProteinMDError):
    """Base class for performance-related errors."""
    pass


class MemoryError(PerformanceError):
    """Memory allocation and management errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="MEMORY_ERROR",
            suggestions=[
                "Reduce system size",
                "Increase available memory",
                "Use memory-efficient algorithms",
                "Enable disk-based storage"
            ],
            **kwargs
        )


class TimeoutError(PerformanceError):
    """Operation timeout errors."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        details = kwargs.get('details', {})
        if timeout_seconds:
            details['timeout_seconds'] = timeout_seconds
        
        super().__init__(
            message,
            error_code="TIMEOUT_ERROR",
            details=details,
            suggestions=[
                "Increase timeout duration",
                "Optimize calculation parameters",
                "Use more powerful hardware",
                "Consider approximate methods"
            ],
            **kwargs
        )


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================

class ConfigurationError(ProteinMDError):
    """Configuration and setup errors."""
    
    def __init__(self, message: str, error_code: str = "CONFIGURATION_ERROR", **kwargs):
        super().__init__(
            message,
            error_code=error_code,
            suggestions=[
                "Check configuration file syntax",
                "Verify parameter values",
                "Use default configuration",
                "Consult documentation"
            ],
            **kwargs
        )


# =============================================================================
# WARNING CLASSES
# =============================================================================

class ProteinMDWarning(UserWarning):
    """Base warning class for ProteinMD."""
    
    def __init__(self, message: str, warning_code: str = "GENERAL_WARNING", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.warning_code = warning_code
        self.details = details or {}
        self.timestamp = datetime.now()
        super().__init__(message)


class SimulationWarning(ProteinMDWarning):
    """Warnings during simulation that don't stop execution."""
    pass


class ParameterWarning(ProteinMDWarning):
    """Warnings about potentially problematic parameters."""
    pass


class PerformanceWarning(ProteinMDWarning):
    """Warnings about performance issues."""
    pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_error_report(error: ProteinMDError) -> str:
    """Format a comprehensive error report."""
    report = f"""
=== ProteinMD Error Report ===
Time: {error.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Error Type: {error.__class__.__name__}
Error Code: {error.error_code}
Message: {error.message}

Recoverable: {'Yes' if error.recoverable else 'No'}
"""
    
    if error.details:
        report += f"\nDetails:\n"
        for key, value in error.details.items():
            report += f"  {key}: {value}\n"
    
    if error.suggestions:
        report += f"\nSuggestions:\n"
        for suggestion in error.suggestions:
            report += f"  • {suggestion}\n"
    
    if error.stack_trace and error.stack_trace != "NoneType: None\n":
        report += f"\nStack Trace:\n{error.stack_trace}"
    
    report += "\n" + "="*50
    return report


def get_error_hierarchy() -> Dict[str, List[str]]:
    """Get the complete error hierarchy for documentation."""
    hierarchy = {
        "ProteinMDError": [
            "SimulationError",
            "StructureError", 
            "ForceFieldError",
            "ProteinMDIOError",
            "AnalysisError",
            "VisualizationError",
            "PerformanceError",
            "ConfigurationError"
        ],
        "SimulationError": [
            "SimulationSetupError",
            "SimulationRuntimeError",
            "ConvergenceError"
        ],
        "StructureError": [
            "PDBParsingError",
            "StructureValidationError",
            "TopologyError"
        ],
        "ForceFieldError": [
            "ParameterError",
            "ForceFieldValidationError"
        ],
        "ProteinMDIOError": [
            "FileFormatError",
            "TrajectoryError"
        ],
        "AnalysisError": [
            "InsufficientDataError",
            "CalculationError"
        ],
        "VisualizationError": [
            "RenderingError",
            "PlotGenerationError"
        ],
        "PerformanceError": [
            "MemoryError",
            "TimeoutError"
        ]
    }
    return hierarchy
