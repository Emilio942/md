"""
Workflow Definition and Configuration Schema

This module defines the structure and validation for ProteinMD workflows,
enabling users to create complex multi-step analysis pipelines with dependencies.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WorkflowValidationError(Exception):
    """Exception raised for workflow validation errors."""
    pass


@dataclass
class WorkflowStep:
    """
    Definition of a single workflow step.
    
    A workflow step represents a single operation in the analysis pipeline,
    such as running a simulation, performing analysis, or generating reports.
    """
    
    name: str
    """Unique step name within the workflow"""
    
    command: str
    """Command to execute (e.g., 'simulate', 'analyze', 'report')"""
    
    parameters: Dict[str, Any] = field(default_factory=dict)
    """Parameters for the command"""
    
    dependencies: List[str] = field(default_factory=list)
    """List of step names this step depends on"""
    
    inputs: Dict[str, str] = field(default_factory=dict)
    """Input files/data from other steps"""
    
    outputs: Dict[str, str] = field(default_factory=dict)
    """Output files/data produced by this step"""
    
    resources: Dict[str, Any] = field(default_factory=dict)
    """Resource requirements (CPU, memory, time)"""
    
    retry_count: int = 0
    """Number of retry attempts if step fails"""
    
    timeout: Optional[int] = None
    """Timeout in seconds (None = no timeout)"""
    
    condition: Optional[str] = None
    """Conditional execution (Python expression)"""
    
    environment: Dict[str, str] = field(default_factory=dict)
    """Environment variables for step execution"""
    
    working_directory: Optional[str] = None
    """Working directory for step execution"""
    
    def validate(self) -> None:
        """Validate step configuration."""
        if not self.name:
            raise WorkflowValidationError("Step name cannot be empty")
            
        if not self.command:
            raise WorkflowValidationError(f"Step '{self.name}': command cannot be empty")
            
        # Validate retry count
        if self.retry_count < 0:
            raise WorkflowValidationError(f"Step '{self.name}': retry_count must be >= 0")
            
        # Validate timeout
        if self.timeout is not None and self.timeout <= 0:
            raise WorkflowValidationError(f"Step '{self.name}': timeout must be > 0")
            
        # Validate step doesn't depend on itself
        if self.name in self.dependencies:
            raise WorkflowValidationError(f"Step '{self.name}' cannot depend on itself")


@dataclass
class WorkflowDefinition:
    """
    Complete workflow definition with metadata and steps.
    
    Defines a complete analysis workflow including all steps, their dependencies,
    global configuration, and execution metadata.
    """
    
    name: str
    """Workflow name"""
    
    description: str = ""
    """Workflow description"""
    
    version: str = "1.0.0"
    """Workflow version"""
    
    author: str = ""
    """Workflow author"""
    
    created: Optional[str] = None
    """Creation timestamp"""
    
    steps: List[WorkflowStep] = field(default_factory=list)
    """List of workflow steps"""
    
    global_parameters: Dict[str, Any] = field(default_factory=dict)
    """Global parameters available to all steps"""
    
    global_environment: Dict[str, str] = field(default_factory=dict)
    """Global environment variables"""
    
    default_resources: Dict[str, Any] = field(default_factory=dict)
    """Default resource requirements"""
    
    output_directory: str = "workflow_output"
    """Default output directory"""
    
    on_failure: str = "stop"
    """Failure handling: 'stop', 'continue', 'ignore'"""
    
    max_parallel_steps: int = 1
    """Maximum number of steps to run in parallel"""
    
    scheduler: Dict[str, Any] = field(default_factory=dict)
    """Job scheduler configuration"""
    
    notifications: Dict[str, Any] = field(default_factory=dict)
    """Notification settings"""
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.created is None:
            self.created = datetime.now().isoformat()
            
    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow."""
        # Check for duplicate step names
        if any(s.name == step.name for s in self.steps):
            raise WorkflowValidationError(f"Step name '{step.name}' already exists")
            
        step.validate()
        self.steps.append(step)
        
    def get_step(self, name: str) -> Optional[WorkflowStep]:
        """Get a step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None
        
    def get_step_names(self) -> List[str]:
        """Get list of all step names."""
        return [step.name for step in self.steps]
        
    def validate(self) -> None:
        """Validate complete workflow definition."""
        if not self.name:
            raise WorkflowValidationError("Workflow name cannot be empty")
            
        if not self.steps:
            raise WorkflowValidationError("Workflow must have at least one step")
            
        # Validate all steps
        step_names = set()
        for step in self.steps:
            step.validate()
            
            # Check for duplicate step names
            if step.name in step_names:
                raise WorkflowValidationError(f"Duplicate step name: {step.name}")
            step_names.add(step.name)
            
        # Validate dependencies exist
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in step_names:
                    raise WorkflowValidationError(
                        f"Step '{step.name}' depends on non-existent step '{dep}'"
                    )
                    
        # Check for circular dependencies
        self._check_circular_dependencies()
        
        # Validate failure handling
        if self.on_failure not in ['stop', 'continue', 'ignore']:
            raise WorkflowValidationError(
                f"Invalid on_failure value: {self.on_failure}. Must be 'stop', 'continue', or 'ignore'"
            )
            
        # Validate max_parallel_steps
        if self.max_parallel_steps < 1:
            raise WorkflowValidationError("max_parallel_steps must be >= 1")
            
    def _check_circular_dependencies(self) -> None:
        """Check for circular dependencies in the workflow."""
        # Build dependency graph
        graph = {step.name: set(step.dependencies) for step in self.steps}
        
        # DFS-based cycle detection
        white = set(graph.keys())  # Unvisited
        gray = set()               # Currently processing
        black = set()              # Completed
        
        def visit(node: str, path: List[str]) -> None:
            if node in gray:
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                raise WorkflowValidationError(
                    f"Circular dependency detected: {' -> '.join(cycle)}"
                )
                
            if node in black:
                return
                
            white.discard(node)
            gray.add(node)
            
            for neighbor in graph.get(node, []):
                visit(neighbor, path + [node])
                
            gray.discard(node)
            black.add(node)
            
        # Check all nodes
        while white:
            node = white.pop()
            white.add(node)  # Add back temporarily
            visit(node, [])
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary representation."""
        def convert_dataclass(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: convert_dataclass(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [convert_dataclass(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_dataclass(v) for k, v in obj.items()}
            else:
                return obj
                
        return convert_dataclass(self)
        
    def to_json(self, indent: int = 2) -> str:
        """Convert workflow to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
        
    def to_yaml(self) -> str:
        """Convert workflow to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
        
    def save(self, file_path: Union[str, Path]) -> None:
        """Save workflow to file (JSON or YAML based on extension)."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'w') as f:
                f.write(self.to_yaml())
        else:
            with open(file_path, 'w') as f:
                f.write(self.to_json())
                
        logger.info(f"Workflow saved to {file_path}")


def load_workflow_from_dict(data: Dict[str, Any]) -> WorkflowDefinition:
    """Load workflow from dictionary data."""
    # Extract steps data
    steps_data = data.pop('steps', [])
    
    # Create workflow
    workflow = WorkflowDefinition(**data)
    
    # Create and add steps
    for step_data in steps_data:
        step = WorkflowStep(**step_data)
        workflow.add_step(step)
        
    # Validate complete workflow
    workflow.validate()
    
    return workflow


def load_workflow_from_file(file_path: Union[str, Path]) -> WorkflowDefinition:
    """
    Load workflow from file (JSON or YAML).
    
    Args:
        file_path: Path to workflow file
        
    Returns:
        WorkflowDefinition instance
        
    Raises:
        FileNotFoundError: If file doesn't exist
        WorkflowValidationError: If workflow is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Workflow file not found: {file_path}")
        
    try:
        with open(file_path) as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
                
        workflow = load_workflow_from_dict(data)
        logger.info(f"Workflow loaded from {file_path}")
        return workflow
        
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise WorkflowValidationError(f"Failed to parse workflow file {file_path}: {e}")
    except Exception as e:
        raise WorkflowValidationError(f"Failed to load workflow from {file_path}: {e}")


def create_example_workflow() -> WorkflowDefinition:
    """Create an example workflow for demonstration purposes."""
    workflow = WorkflowDefinition(
        name="protein_analysis_pipeline",
        description="Complete protein analysis workflow with simulation and comprehensive analysis",
        version="1.0.0",
        author="ProteinMD",
        output_directory="protein_analysis_results",
        max_parallel_steps=2,
        global_parameters={
            "simulation_steps": 50000,
            "temperature": 300.0,
            "analysis_stride": 10
        },
        default_resources={
            "cpu_cores": 4,
            "memory_gb": 8,
            "time_limit": "02:00:00"
        }
    )
    
    # Step 1: Simulation
    simulation_step = WorkflowStep(
        name="simulate",
        command="proteinmd simulate",
        parameters={
            "template": "equilibration",
            "steps": "${global_parameters.simulation_steps}",
            "temperature": "${global_parameters.temperature}"
        },
        outputs={
            "trajectory": "simulation/trajectory.npz",
            "structure": "simulation/final_structure.pdb"
        },
        resources={
            "cpu_cores": 8,
            "time_limit": "04:00:00"
        }
    )
    
    # Step 2: RMSD Analysis
    rmsd_step = WorkflowStep(
        name="rmsd_analysis",
        command="proteinmd analyze",
        parameters={
            "analysis": ["rmsd"],
            "stride": "${global_parameters.analysis_stride}"
        },
        dependencies=["simulate"],
        inputs={
            "trajectory": "${simulate.outputs.trajectory}",
            "structure": "${simulate.outputs.structure}"
        },
        outputs={
            "rmsd_data": "analysis/rmsd_analysis.csv",
            "rmsd_plot": "analysis/rmsd_plot.png"
        }
    )
    
    # Step 3: Secondary Structure Analysis (parallel with RMSD)
    ss_step = WorkflowStep(
        name="secondary_structure",
        command="proteinmd analyze",
        parameters={
            "analysis": ["secondary_structure"],
            "stride": "${global_parameters.analysis_stride}"
        },
        dependencies=["simulate"],
        inputs={
            "trajectory": "${simulate.outputs.trajectory}",
            "structure": "${simulate.outputs.structure}"
        },
        outputs={
            "ss_data": "analysis/secondary_structure.csv",
            "ss_timeline": "analysis/ss_timeline.png"
        }
    )
    
    # Step 4: Generate Report (depends on all analyses)
    report_step = WorkflowStep(
        name="generate_report",
        command="proteinmd report",
        parameters={
            "format": "html",
            "include_plots": True
        },
        dependencies=["rmsd_analysis", "secondary_structure"],
        inputs={
            "rmsd_data": "${rmsd_analysis.outputs.rmsd_data}",
            "ss_data": "${secondary_structure.outputs.ss_data}"
        },
        outputs={
            "report": "report/analysis_report.html",
            "summary": "report/summary.json"
        }
    )
    
    # Add all steps
    workflow.add_step(simulation_step)
    workflow.add_step(rmsd_step)
    workflow.add_step(ss_step)
    workflow.add_step(report_step)
    
    return workflow


# Built-in workflow templates
BUILTIN_WORKFLOWS = {
    "protein_analysis_pipeline": create_example_workflow,
}


def get_builtin_workflow(name: str) -> WorkflowDefinition:
    """Get a built-in workflow by name."""
    if name not in BUILTIN_WORKFLOWS:
        raise ValueError(f"Built-in workflow '{name}' not found. Available: {list(BUILTIN_WORKFLOWS.keys())}")
        
    return BUILTIN_WORKFLOWS[name]()


def list_builtin_workflows() -> List[str]:
    """List available built-in workflows."""
    return list(BUILTIN_WORKFLOWS.keys())
