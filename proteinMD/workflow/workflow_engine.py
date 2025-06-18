"""
Workflow Execution Engine

Core engine for executing ProteinMD workflows with dependency resolution,
parallel execution, error handling, and progress monitoring.
"""

import os
import sys
import time
import subprocess
import threading
import signal
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import logging
import json
import tempfile

from .workflow_definition import WorkflowDefinition, WorkflowStep
from .dependency_resolver import DependencyResolver, StepStatus, StepExecution
from .report_generator import WorkflowReportGenerator

logger = logging.getLogger(__name__)


class WorkflowExecutionError(Exception):
    """Exception raised during workflow execution."""
    pass


class ParameterResolver:
    """Resolves parameter references in workflow steps."""
    
    def __init__(self, global_parameters: Dict[str, Any], step_outputs: Dict[str, Dict[str, str]]):
        """
        Initialize parameter resolver.
        
        Args:
            global_parameters: Global workflow parameters
            step_outputs: Outputs from completed steps
        """
        self.global_parameters = global_parameters
        self.step_outputs = step_outputs
        
    def resolve_string(self, value: str) -> str:
        """
        Resolve parameter references in a string.
        
        Supports references like:
        - ${global_parameters.simulation_steps}
        - ${step_name.outputs.trajectory}
        """
        import re
        
        # Pattern to match ${...} references
        pattern = r'\$\{([^}]+)\}'
        
        def replace_reference(match):
            ref = match.group(1)
            parts = ref.split('.')
            
            try:
                if parts[0] == 'global_parameters':
                    # Reference to global parameters
                    value = self.global_parameters
                    for part in parts[1:]:
                        value = value[part]
                    return str(value)
                    
                elif len(parts) >= 3 and parts[1] == 'outputs':
                    # Reference to step output
                    step_name = parts[0]
                    output_name = parts[2]
                    
                    if step_name in self.step_outputs:
                        if output_name in self.step_outputs[step_name]:
                            return self.step_outputs[step_name][output_name]
                            
                    logger.warning(f"Step output reference not found: {ref}")
                    return match.group(0)  # Return original if not found
                    
                else:
                    logger.warning(f"Unknown parameter reference: {ref}")
                    return match.group(0)
                    
            except (KeyError, TypeError) as e:
                logger.warning(f"Failed to resolve parameter reference {ref}: {e}")
                return match.group(0)
                
        return re.sub(pattern, replace_reference, value)
        
    def resolve_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve all parameter references in a parameter dictionary."""
        resolved = {}
        
        for key, value in parameters.items():
            if isinstance(value, str):
                resolved[key] = self.resolve_string(value)
            elif isinstance(value, dict):
                resolved[key] = self.resolve_parameters(value)
            elif isinstance(value, list):
                resolved[key] = [
                    self.resolve_string(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                resolved[key] = value
                
        return resolved


class WorkflowEngine:
    """
    Core workflow execution engine.
    
    Manages workflow execution including dependency resolution, parallel execution,
    error handling, progress monitoring, and result collection.
    """
    
    def __init__(self, workflow: WorkflowDefinition, working_directory: Optional[str] = None):
        """
        Initialize workflow engine.
        
        Args:
            workflow: Workflow definition to execute
            working_directory: Working directory for execution
        """
        self.workflow = workflow
        self.dependency_resolver = DependencyResolver(workflow)
        self.working_directory = Path(working_directory) if working_directory else Path.cwd()
        self.output_directory = self.working_directory / workflow.output_directory
        
        # Execution state
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.is_running = False
        self.should_stop = False
        self.execution_log: List[Dict[str, Any]] = []
        
        # Thread pool for parallel execution
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.running_futures: Dict[str, Future] = {}
        
        # Progress callbacks
        self.progress_callback: Optional[Callable[[str, Dict], None]] = None
        self.step_callback: Optional[Callable[[str, StepStatus, Optional[str]], None]] = None
        
        # Setup directories
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_execution_logging()
        
    def _setup_execution_logging(self) -> None:
        """Setup execution-specific logging."""
        log_file = self.output_directory / 'workflow_execution.log'
        
        # Create file handler for this workflow execution
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.file_handler.setFormatter(formatter)
        logger.addHandler(self.file_handler)
        
    def set_progress_callback(self, callback: Callable[[str, Dict], None]) -> None:
        """Set callback for progress updates."""
        self.progress_callback = callback
        
    def set_step_callback(self, callback: Callable[[str, StepStatus, Optional[str]], None]) -> None:
        """Set callback for step status updates."""
        self.step_callback = callback
        
    def execute(self) -> bool:
        """
        Execute the workflow.
        
        Returns:
            True if workflow completed successfully, False otherwise
        """
        try:
            logger.info(f"Starting workflow execution: {self.workflow.name}")
            self.start_time = datetime.now()
            self.is_running = True
            self.should_stop = False
            
            # Validate workflow before execution
            self.workflow.validate()
            
            # Create thread pool for parallel execution
            max_workers = min(self.workflow.max_parallel_steps, len(self.workflow.steps))
            self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
            
            # Execute workflow
            success = self._execute_workflow()
            
            self.end_time = datetime.now()
            self.is_running = False
            
            # Generate final report
            self._generate_final_report(success)
            
            logger.info(f"Workflow execution completed. Success: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            self.end_time = datetime.now()
            self.is_running = False
            self._generate_final_report(False)
            return False
            
        finally:
            # Cleanup
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            if hasattr(self, 'file_handler'):
                logger.removeHandler(self.file_handler)
                
    def _execute_workflow(self) -> bool:
        """Execute the workflow steps."""
        # Get execution order
        execution_batches = self.dependency_resolver.get_execution_order()
        logger.info(f"Workflow execution plan: {len(execution_batches)} batches")
        
        for batch_idx, batch in enumerate(execution_batches):
            if self.should_stop:
                logger.info("Workflow execution stopped by user")
                return False
                
            logger.info(f"Executing batch {batch_idx + 1}/{len(execution_batches)}: {batch}")
            
            # Execute batch (steps in parallel)
            success = self._execute_batch(batch)
            
            if not success:
                if self.workflow.on_failure == 'stop':
                    logger.error("Workflow execution stopped due to step failure")
                    return False
                elif self.workflow.on_failure == 'continue':
                    logger.warning("Continuing workflow execution despite step failure")
                    continue
                elif self.workflow.on_failure == 'ignore':
                    logger.info("Ignoring step failure and continuing")
                    continue
                    
        # Check final status
        return self.dependency_resolver.is_workflow_successful()
        
    def _execute_batch(self, step_names: List[str]) -> bool:
        """Execute a batch of steps in parallel."""
        if len(step_names) == 1:
            # Single step - execute directly
            return self._execute_step(step_names[0])
            
        # Multiple steps - execute in parallel
        futures = {}
        for step_name in step_names:
            future = self.thread_pool.submit(self._execute_step, step_name)
            futures[step_name] = future
            self.running_futures[step_name] = future
            
        # Wait for all steps to complete
        batch_success = True
        for step_name, future in futures.items():
            try:
                step_success = future.result()
                if not step_success:
                    batch_success = False
            except Exception as e:
                logger.error(f"Step '{step_name}' execution failed with exception: {e}")
                batch_success = False
            finally:
                if step_name in self.running_futures:
                    del self.running_futures[step_name]
                    
        return batch_success
        
    def _execute_step(self, step_name: str) -> bool:
        """Execute a single workflow step."""
        try:
            step_execution = self.dependency_resolver.step_executions[step_name]
            step = step_execution.step
            
            logger.info(f"Starting step: {step_name}")
            
            # Update status
            self.dependency_resolver.update_step_status(step_name, StepStatus.RUNNING)
            step_execution.start_time = datetime.now().isoformat()
            
            if self.step_callback:
                self.step_callback(step_name, StepStatus.RUNNING, None)
                
            # Check condition if specified
            if step.condition and not self._evaluate_condition(step.condition):
                logger.info(f"Step '{step_name}' skipped due to condition: {step.condition}")
                self.dependency_resolver.update_step_status(step_name, StepStatus.SKIPPED)
                if self.step_callback:
                    self.step_callback(step_name, StepStatus.SKIPPED, f"Condition not met: {step.condition}")
                return True
                
            # Resolve parameters
            parameter_resolver = ParameterResolver(
                self.workflow.global_parameters,
                {name: exec.outputs for name, exec in self.dependency_resolver.step_executions.items()}
            )
            resolved_parameters = parameter_resolver.resolve_parameters(step.parameters)
            
            # Execute step
            success = False
            retry_count = 0
            max_retries = step.retry_count + 1  # +1 for initial attempt
            
            while retry_count < max_retries and not self.should_stop:
                try:
                    if retry_count > 0:
                        logger.info(f"Retrying step '{step_name}' (attempt {retry_count + 1}/{max_retries})")
                        
                    # Execute the step command
                    exit_code, outputs, error_message = self._execute_step_command(
                        step, resolved_parameters
                    )
                    
                    if exit_code == 0:
                        # Success
                        step_execution.end_time = datetime.now().isoformat()
                        self.dependency_resolver.update_step_status(
                            step_name, StepStatus.COMPLETED, 
                            exit_code=exit_code, outputs=outputs
                        )
                        
                        if self.step_callback:
                            self.step_callback(step_name, StepStatus.COMPLETED, None)
                            
                        logger.info(f"Step '{step_name}' completed successfully")
                        success = True
                        break
                    else:
                        # Failed
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.warning(f"Step '{step_name}' failed (exit code {exit_code}), retrying...")
                            time.sleep(2)  # Brief delay before retry
                        else:
                            # Final failure
                            step_execution.end_time = datetime.now().isoformat()
                            self.dependency_resolver.update_step_status(
                                step_name, StepStatus.FAILED,
                                error_message=error_message, exit_code=exit_code
                            )
                            
                            if self.step_callback:
                                self.step_callback(step_name, StepStatus.FAILED, error_message)
                                
                            logger.error(f"Step '{step_name}' failed after {max_retries} attempts")
                            
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)
                    
                    if retry_count < max_retries:
                        logger.warning(f"Step '{step_name}' failed with exception, retrying: {error_msg}")
                        time.sleep(2)
                    else:
                        step_execution.end_time = datetime.now().isoformat()
                        self.dependency_resolver.update_step_status(
                            step_name, StepStatus.FAILED, error_message=error_msg
                        )
                        
                        if self.step_callback:
                            self.step_callback(step_name, StepStatus.FAILED, error_msg)
                            
                        logger.error(f"Step '{step_name}' failed with exception: {error_msg}")
                        
            return success
            
        except Exception as e:
            logger.error(f"Unexpected error executing step '{step_name}': {e}")
            self.dependency_resolver.update_step_status(step_name, StepStatus.FAILED, error_message=str(e))
            if self.step_callback:
                self.step_callback(step_name, StepStatus.FAILED, str(e))
            return False
            
    def _execute_step_command(self, step: WorkflowStep, parameters: Dict[str, Any]) -> tuple[int, Dict[str, str], str]:
        """
        Execute the actual command for a step.
        
        Returns:
            Tuple of (exit_code, outputs, error_message)
        """
        # Prepare working directory
        if step.working_directory:
            work_dir = Path(step.working_directory)
        else:
            work_dir = self.output_directory / step.name
            
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare environment
        env = os.environ.copy()
        env.update(self.workflow.global_environment)
        env.update(step.environment)
        
        # Build command
        if step.command.startswith('proteinmd'):
            # ProteinMD command - execute via CLI
            cmd_parts = step.command.split()
            
            # Add parameters
            for key, value in parameters.items():
                if isinstance(value, bool):
                    if value:
                        cmd_parts.append(f'--{key}')
                elif isinstance(value, list):
                    for item in value:
                        cmd_parts.extend([f'--{key}', str(item)])
                else:
                    cmd_parts.extend([f'--{key}', str(value)])
                    
            # Add output directory
            cmd_parts.extend(['--output-dir', str(work_dir)])
            
        else:
            # Custom command
            cmd_parts = step.command.split()
            
        logger.debug(f"Executing command: {' '.join(cmd_parts)}")
        
        # Execute command
        try:
            process = subprocess.Popen(
                cmd_parts,
                cwd=work_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Handle timeout
            try:
                stdout, stderr = process.communicate(timeout=step.timeout)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                exit_code = -1
                stderr = f"Command timed out after {step.timeout} seconds\n" + stderr
                
        except Exception as e:
            return 1, {}, f"Failed to execute command: {e}"
            
        # Collect outputs
        outputs = {}
        for output_name, output_path in step.outputs.items():
            full_path = work_dir / output_path
            if full_path.exists():
                outputs[output_name] = str(full_path)
            else:
                logger.warning(f"Expected output '{output_name}' not found: {full_path}")
                
        # Log execution details
        with open(work_dir / 'execution.log', 'w') as f:
            f.write(f"Command: {' '.join(cmd_parts)}\n")
            f.write(f"Exit code: {exit_code}\n")
            f.write(f"Working directory: {work_dir}\n")
            f.write("\n--- STDOUT ---\n")
            f.write(stdout)
            f.write("\n--- STDERR ---\n")
            f.write(stderr)
            
        error_message = stderr if exit_code != 0 else ""
        return exit_code, outputs, error_message
        
    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a condition expression."""
        try:
            # Simple evaluation - could be extended with more sophisticated logic
            # For now, just support basic comparisons
            return eval(condition, {"__builtins__": {}}, {})
        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False
            
    def _generate_final_report(self, success: bool) -> None:
        """Generate final workflow execution report."""
        try:
            report_generator = WorkflowReportGenerator(
                self.workflow, 
                self.dependency_resolver,
                self.output_directory
            )
            
            report_path = report_generator.generate_report(
                execution_time=self.get_execution_time(),
                success=success
            )
            
            logger.info(f"Workflow report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate workflow report: {e}")
            
    def stop(self) -> None:
        """Stop workflow execution."""
        logger.info("Stopping workflow execution...")
        self.should_stop = True
        
        # Cancel running futures
        for step_name, future in self.running_futures.items():
            logger.info(f"Cancelling step: {step_name}")
            future.cancel()
            
    def get_execution_time(self) -> Optional[float]:
        """Get total execution time in seconds."""
        if self.start_time is None:
            return None
            
        end_time = self.end_time or datetime.now()
        return (end_time - self.start_time).total_seconds()
        
    def get_status(self) -> Dict[str, Any]:
        """Get current workflow execution status."""
        execution_summary = self.dependency_resolver.get_execution_summary()
        
        return {
            'workflow_name': self.workflow.name,
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'execution_time': self.get_execution_time(),
            'output_directory': str(self.output_directory),
            **execution_summary
        }
        
    def get_step_status(self, step_name: str) -> Dict[str, Any]:
        """Get detailed status of a specific step."""
        if step_name not in self.dependency_resolver.step_executions:
            raise ValueError(f"Step '{step_name}' not found")
            
        execution = self.dependency_resolver.step_executions[step_name]
        
        return {
            'name': step_name,
            'status': execution.status.value,
            'start_time': execution.start_time,
            'end_time': execution.end_time,
            'exit_code': execution.exit_code,
            'error_message': execution.error_message,
            'retry_count': execution.retry_count,
            'outputs': execution.outputs,
            'dependencies': execution.step.dependencies,
            'command': execution.step.command
        }
