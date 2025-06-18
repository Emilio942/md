"""
ProteinMD Workflow Automation Package

Task 8.4: Workflow Automation 📊

This package provides comprehensive workflow automation capabilities for ProteinMD,
enabling users to define, execute, and manage complex multi-step analysis pipelines
with dependency resolution, automatic report generation, and HPC integration.

Key Features:
- Configurable workflows via YAML/JSON
- Step dependency resolution and execution ordering
- Automatic report generation after simulation completion
- Integration with job scheduling systems (SLURM, PBS, SGE)
- Workflow monitoring and error handling
- Template-based workflow library

Components:
- workflow_engine.py: Core execution engine
- workflow_definition.py: Configuration schema and validation  
- dependency_resolver.py: Step dependency management
- report_generator.py: Automated report generation
- job_scheduler.py: HPC integration layer
- workflow_cli.py: Command-line interface
- schedulers/: Job scheduler implementations
"""

from .workflow_engine import WorkflowEngine
from .workflow_definition import (
    WorkflowDefinition, 
    WorkflowStep, 
    WorkflowValidationError,
    load_workflow_from_file
)
from .dependency_resolver import DependencyResolver
from .report_generator import WorkflowReportGenerator
from .job_scheduler import JobSchedulerManager
from .workflow_cli import WorkflowCLI

__all__ = [
    'WorkflowEngine',
    'WorkflowDefinition',
    'WorkflowStep', 
    'WorkflowValidationError',
    'load_workflow_from_file',
    'DependencyResolver',
    'WorkflowReportGenerator', 
    'JobSchedulerManager',
    'WorkflowCLI'
]

__version__ = '1.0.0'
