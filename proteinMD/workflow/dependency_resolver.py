"""
Dependency Resolver for Workflow Execution

This module implements dependency resolution and execution ordering for workflow steps,
ensuring proper execution sequence and enabling parallel execution where possible.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

from .workflow_definition import WorkflowDefinition, WorkflowStep

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    READY = "ready" 
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepExecution:
    """Execution information for a workflow step."""
    step: WorkflowStep
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    outputs: Dict[str, str] = None
    
    def __post_init__(self):
        if self.outputs is None:
            self.outputs = {}


class DependencyResolver:
    """
    Resolves dependencies and determines execution order for workflow steps.
    
    Provides topological sorting of steps based on dependencies and identifies
    steps that can be executed in parallel.
    """
    
    def __init__(self, workflow: WorkflowDefinition):
        """
        Initialize dependency resolver with workflow.
        
        Args:
            workflow: Workflow definition to resolve
        """
        self.workflow = workflow
        self.step_executions: Dict[str, StepExecution] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.reverse_dependencies: Dict[str, Set[str]] = {}
        
        # Initialize step executions
        for step in workflow.steps:
            self.step_executions[step.name] = StepExecution(step)
            
        # Build dependency graphs
        self._build_dependency_graph()
        
    def _build_dependency_graph(self) -> None:
        """Build dependency and reverse dependency graphs."""
        # Initialize graphs
        for step in self.workflow.steps:
            self.dependency_graph[step.name] = set(step.dependencies)
            self.reverse_dependencies[step.name] = set()
            
        # Build reverse dependencies
        for step_name, deps in self.dependency_graph.items():
            for dep in deps:
                self.reverse_dependencies[dep].add(step_name)
                
    def get_execution_order(self) -> List[List[str]]:
        """
        Get execution order as list of batches.
        
        Each batch contains steps that can be executed in parallel.
        
        Returns:
            List of batches, where each batch is a list of step names
            that can be executed in parallel
        """
        # Copy dependency graph for modification
        remaining_deps = {name: deps.copy() for name, deps in self.dependency_graph.items()}
        completed_steps = set()
        execution_order = []
        
        while remaining_deps:
            # Find steps with no remaining dependencies
            ready_steps = [
                step_name for step_name, deps in remaining_deps.items()
                if not deps
            ]
            
            if not ready_steps:
                # This should not happen if workflow is valid (no cycles)
                remaining_step_names = list(remaining_deps.keys())
                raise RuntimeError(
                    f"Circular dependency detected. Remaining steps: {remaining_step_names}"
                )
                
            # Add ready steps to execution order
            execution_order.append(ready_steps)
            
            # Mark steps as completed and remove them
            for step_name in ready_steps:
                completed_steps.add(step_name)
                del remaining_deps[step_name]
                
            # Update remaining dependencies
            for step_name in remaining_deps:
                remaining_deps[step_name] -= completed_steps
                
        return execution_order
        
    def get_ready_steps(self) -> List[str]:
        """
        Get steps that are ready to execute now.
        
        Returns:
            List of step names ready for execution
        """
        ready_steps = []
        
        for step_name, execution in self.step_executions.items():
            if execution.status != StepStatus.PENDING:
                continue
                
            # Check if all dependencies are completed
            dependencies_completed = True
            for dep_name in execution.step.dependencies:
                dep_execution = self.step_executions[dep_name]
                if dep_execution.status != StepStatus.COMPLETED:
                    dependencies_completed = False
                    break
                    
            if dependencies_completed:
                ready_steps.append(step_name)
                
        return ready_steps
        
    def can_execute_parallel(self, step_names: List[str]) -> bool:
        """
        Check if given steps can be executed in parallel.
        
        Args:
            step_names: List of step names to check
            
        Returns:
            True if steps can run in parallel, False otherwise
        """
        # Check for direct dependencies between steps
        for i, step1 in enumerate(step_names):
            for j, step2 in enumerate(step_names):
                if i != j:
                    # Check if step1 depends on step2 or vice versa
                    if (step2 in self.dependency_graph[step1] or 
                        step1 in self.dependency_graph[step2]):
                        return False
                        
        return True
        
    def update_step_status(self, step_name: str, status: StepStatus,
                          error_message: Optional[str] = None,
                          exit_code: Optional[int] = None,
                          outputs: Optional[Dict[str, str]] = None) -> None:
        """
        Update status of a step.
        
        Args:
            step_name: Name of the step
            status: New status
            error_message: Error message if failed
            exit_code: Exit code of execution
            outputs: Output files/data produced
        """
        if step_name not in self.step_executions:
            raise ValueError(f"Step '{step_name}' not found")
            
        execution = self.step_executions[step_name]
        execution.status = status
        
        if error_message:
            execution.error_message = error_message
            
        if exit_code is not None:
            execution.exit_code = exit_code
            
        if outputs:
            execution.outputs.update(outputs)
            
        logger.debug(f"Step '{step_name}' status updated to {status.value}")
        
    def increment_retry_count(self, step_name: str) -> int:
        """
        Increment retry count for a step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            New retry count
        """
        if step_name not in self.step_executions:
            raise ValueError(f"Step '{step_name}' not found")
            
        execution = self.step_executions[step_name]
        execution.retry_count += 1
        
        logger.debug(f"Step '{step_name}' retry count: {execution.retry_count}")
        return execution.retry_count
        
    def can_retry(self, step_name: str) -> bool:
        """
        Check if a step can be retried.
        
        Args:
            step_name: Name of the step
            
        Returns:
            True if step can be retried, False otherwise
        """
        if step_name not in self.step_executions:
            return False
            
        execution = self.step_executions[step_name]
        return execution.retry_count < execution.step.retry_count
        
    def get_failed_steps(self) -> List[str]:
        """Get list of failed steps."""
        return [
            name for name, execution in self.step_executions.items()
            if execution.status == StepStatus.FAILED
        ]
        
    def get_completed_steps(self) -> List[str]:
        """Get list of completed steps."""
        return [
            name for name, execution in self.step_executions.items()
            if execution.status == StepStatus.COMPLETED
        ]
        
    def get_pending_steps(self) -> List[str]:
        """Get list of pending steps."""
        return [
            name for name, execution in self.step_executions.items()
            if execution.status == StepStatus.PENDING
        ]
        
    def get_running_steps(self) -> List[str]:
        """Get list of currently running steps."""
        return [
            name for name, execution in self.step_executions.items()
            if execution.status == StepStatus.RUNNING
        ]
        
    def is_workflow_complete(self) -> bool:
        """Check if workflow execution is complete."""
        for execution in self.step_executions.values():
            if execution.status in [StepStatus.PENDING, StepStatus.READY, StepStatus.RUNNING]:
                return False
        return True
        
    def is_workflow_successful(self) -> bool:
        """Check if workflow completed successfully."""
        if not self.is_workflow_complete():
            return False
            
        # Check if any steps failed
        for execution in self.step_executions.values():
            if execution.status == StepStatus.FAILED:
                return False
                
        return True
        
    def get_workflow_status(self) -> Dict[str, int]:
        """Get summary of workflow status."""
        status_counts = {status.value: 0 for status in StepStatus}
        
        for execution in self.step_executions.items():
            status_counts[execution.status.value] += 1
            
        return status_counts
        
    def get_step_dependencies_completed(self, step_name: str) -> Tuple[List[str], List[str]]:
        """
        Get completed and pending dependencies for a step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Tuple of (completed_dependencies, pending_dependencies)
        """
        if step_name not in self.step_executions:
            raise ValueError(f"Step '{step_name}' not found")
            
        step = self.step_executions[step_name].step
        completed = []
        pending = []
        
        for dep_name in step.dependencies:
            dep_execution = self.step_executions[dep_name]
            if dep_execution.status == StepStatus.COMPLETED:
                completed.append(dep_name)
            else:
                pending.append(dep_name)
                
        return completed, pending
        
    def get_step_outputs(self, step_name: str) -> Dict[str, str]:
        """
        Get outputs from a completed step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Dictionary of output names to file paths
        """
        if step_name not in self.step_executions:
            raise ValueError(f"Step '{step_name}' not found")
            
        execution = self.step_executions[step_name]
        if execution.status != StepStatus.COMPLETED:
            logger.warning(f"Step '{step_name}' is not completed (status: {execution.status.value})")
            
        return execution.outputs.copy()
        
    def reset_step(self, step_name: str) -> None:
        """
        Reset a step to pending status.
        
        Args:
            step_name: Name of the step to reset
        """
        if step_name not in self.step_executions:
            raise ValueError(f"Step '{step_name}' not found")
            
        execution = self.step_executions[step_name]
        execution.status = StepStatus.PENDING
        execution.start_time = None
        execution.end_time = None
        execution.exit_code = None
        execution.error_message = None
        execution.retry_count = 0
        execution.outputs.clear()
        
        logger.debug(f"Step '{step_name}' reset to pending status")
        
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary."""
        status_counts = self.get_workflow_status()
        
        return {
            'workflow_name': self.workflow.name,
            'total_steps': len(self.workflow.steps),
            'status_counts': status_counts,
            'is_complete': self.is_workflow_complete(),
            'is_successful': self.is_workflow_successful(),
            'failed_steps': self.get_failed_steps(),
            'completed_steps': self.get_completed_steps(),
            'pending_steps': self.get_pending_steps(),
            'running_steps': self.get_running_steps()
        }
