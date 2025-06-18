"""
SLURM Scheduler Implementation

Extended SLURM integration with advanced features for workflow automation.
"""

import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from ..job_scheduler import JobScheduler, JobResource, JobStatus

logger = logging.getLogger(__name__)


class SLURMAdvancedScheduler(JobScheduler):
    """
    Advanced SLURM integration with extended features.
    
    Provides additional SLURM-specific functionality including:
    - Job arrays
    - Job dependencies
    - Resource monitoring
    - Accounting integration
    """
    
    def __init__(self, default_partition: Optional[str] = None):
        """
        Initialize SLURM scheduler.
        
        Args:
            default_partition: Default partition to use
        """
        super().__init__("slurm_advanced")
        self.default_partition = default_partition
        
    def is_available(self) -> bool:
        """Check if SLURM is available."""
        try:
            result = subprocess.run(
                ['sinfo', '--version'], 
                capture_output=True, 
                check=True,
                timeout=10
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
            
    def submit_job_array(self, command_template: str, job_name: str, 
                        array_size: int, resources: JobResource,
                        working_dir: Path, environment: Dict[str, str] = None) -> str:
        """
        Submit job array to SLURM.
        
        Args:
            command_template: Command template with ${SLURM_ARRAY_TASK_ID} placeholder
            job_name: Base job name
            array_size: Number of array tasks
            resources: Resource requirements
            working_dir: Working directory
            environment: Environment variables
            
        Returns:
            Job array ID
        """
        # Create SLURM script with array directive
        script_content = self._create_array_script(
            command_template, job_name, array_size, resources, working_dir, environment
        )
        
        # Write script to file
        script_file = working_dir / f'{job_name}_array_slurm.sh'
        with open(script_file, 'w') as f:
            f.write(script_content)
            
        script_file.chmod(0o755)
        
        # Submit job array
        try:
            result = subprocess.run(
                ['sbatch', str(script_file)],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract job ID from output
            job_id = result.stdout.strip().split()[-1]
            
            logger.info(f"Submitted SLURM job array {job_id}: {job_name} ({array_size} tasks)")
            return job_id
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to submit SLURM job array: {e.stderr}")
            
    def submit_job_with_dependency(self, command: str, job_name: str, 
                                  resources: JobResource, working_dir: Path,
                                  dependency_job_ids: List[str],
                                  dependency_type: str = "afterok",
                                  environment: Dict[str, str] = None) -> str:
        """
        Submit job with dependencies.
        
        Args:
            command: Command to execute
            job_name: Job name
            resources: Resource requirements
            working_dir: Working directory
            dependency_job_ids: List of job IDs to depend on
            dependency_type: Type of dependency (afterok, afterany, etc.)
            environment: Environment variables
            
        Returns:
            Job ID
        """
        # Create SLURM script
        script_content = self._create_slurm_script(
            command, job_name, resources, working_dir, environment
        )
        
        # Write script to file
        script_file = working_dir / f'{job_name}_dep_slurm.sh'
        with open(script_file, 'w') as f:
            f.write(script_content)
            
        script_file.chmod(0o755)
        
        # Build dependency string
        dependency_str = f"{dependency_type}:{':'.join(dependency_job_ids)}"
        
        # Submit job with dependency
        try:
            result = subprocess.run(
                ['sbatch', '--dependency', dependency_str, str(script_file)],
                capture_output=True,
                text=True,
                check=True
            )
            
            job_id = result.stdout.strip().split()[-1]
            
            logger.info(f"Submitted SLURM job {job_id} with dependency: {dependency_str}")
            return job_id
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to submit SLURM job with dependency: {e.stderr}")
            
    def get_job_efficiency(self, job_id: str) -> Dict[str, Any]:
        """
        Get job efficiency metrics from SLURM accounting.
        
        Args:
            job_id: Job ID
            
        Returns:
            Dictionary with efficiency metrics
        """
        try:
            result = subprocess.run(
                ['seff', job_id],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Parse seff output
                lines = result.stdout.split('\n')
                efficiency = {}
                
                for line in lines:
                    if 'CPU Efficiency:' in line:
                        efficiency['cpu_efficiency'] = line.split(':')[1].strip()
                    elif 'Memory Efficiency:' in line:
                        efficiency['memory_efficiency'] = line.split(':')[1].strip()
                    elif 'Job Wall-clock time:' in line:
                        efficiency['wall_time'] = line.split(':')[1].strip()
                    elif 'Memory Utilized:' in line:
                        efficiency['memory_used'] = line.split(':')[1].strip()
                        
                return efficiency
                
        except subprocess.CalledProcessError:
            pass
            
        return {}
        
    def get_queue_info(self) -> Dict[str, Any]:
        """Get SLURM queue information."""
        try:
            result = subprocess.run(
                ['sinfo', '--format=%P,%A,%N,%T', '--noheader'],
                capture_output=True,
                text=True,
                check=True
            )
            
            partitions = {}
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(',')
                    if len(parts) >= 4:
                        partition = parts[0]
                        available_nodes = parts[1]
                        total_nodes = parts[2]
                        state = parts[3]
                        
                        partitions[partition] = {
                            'available_nodes': available_nodes,
                            'total_nodes': total_nodes,
                            'state': state
                        }
                        
            return partitions
            
        except subprocess.CalledProcessError:
            return {}
            
    def _create_array_script(self, command_template: str, job_name: str, 
                            array_size: int, resources: JobResource,
                            working_dir: Path, environment: Dict[str, str] = None) -> str:
        """Create SLURM job array script."""
        script = "#!/bin/bash\n\n"
        
        # SLURM directives
        script += f"#SBATCH --job-name={job_name}\n"
        script += f"#SBATCH --array=1-{array_size}\n"
        script += f"#SBATCH --ntasks=1\n"
        script += f"#SBATCH --cpus-per-task={resources.cpu_cores}\n"
        script += f"#SBATCH --mem={resources.memory_gb}G\n"
        script += f"#SBATCH --time={resources.time_limit}\n"
        
        if resources.gpu_count > 0:
            script += f"#SBATCH --gres=gpu:{resources.gpu_count}\n"
            
        partition = resources.partition or self.default_partition
        if partition:
            script += f"#SBATCH --partition={partition}\n"
            
        if resources.exclusive:
            script += f"#SBATCH --exclusive\n"
            
        # Array-specific output files
        script += f"#SBATCH --output={working_dir}/{job_name}_%A_%a.out\n"
        script += f"#SBATCH --error={working_dir}/{job_name}_%A_%a.err\n"
        
        script += "\n"
        
        # Environment setup
        if environment:
            for key, value in environment.items():
                script += f"export {key}={value}\n"
            script += "\n"
            
        # Array-specific environment
        script += "export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}\n"
        script += f"cd {working_dir}\n\n"
        
        # Execute command with array task ID substitution
        command = command_template.replace('${SLURM_ARRAY_TASK_ID}', '${SLURM_ARRAY_TASK_ID}')
        script += f"{command}\n"
        
        return script
        
    def _create_slurm_script(self, command: str, job_name: str, resources: JobResource,
                            working_dir: Path, environment: Dict[str, str] = None) -> str:
        """Create standard SLURM batch script."""
        script = "#!/bin/bash\n\n"
        
        # SLURM directives
        script += f"#SBATCH --job-name={job_name}\n"
        script += f"#SBATCH --ntasks=1\n"
        script += f"#SBATCH --cpus-per-task={resources.cpu_cores}\n"
        script += f"#SBATCH --mem={resources.memory_gb}G\n"
        script += f"#SBATCH --time={resources.time_limit}\n"
        
        if resources.gpu_count > 0:
            script += f"#SBATCH --gres=gpu:{resources.gpu_count}\n"
            
        partition = resources.partition or self.default_partition
        if partition:
            script += f"#SBATCH --partition={partition}\n"
            
        if resources.exclusive:
            script += f"#SBATCH --exclusive\n"
            
        # Output files
        script += f"#SBATCH --output={working_dir}/{job_name}.out\n"
        script += f"#SBATCH --error={working_dir}/{job_name}.err\n"
        
        script += "\n"
        
        # Environment setup
        if environment:
            for key, value in environment.items():
                script += f"export {key}={value}\n"
            script += "\n"
            
        # Change to working directory
        script += f"cd {working_dir}\n\n"
        
        # Execute command
        script += f"{command}\n"
        
        return script
        
    def submit_job(self, command: str, job_name: str, resources: JobResource,
                   working_dir: Path, environment: Dict[str, str] = None) -> str:
        """Submit standard job to SLURM."""
        return self.submit_job_with_dependency(
            command, job_name, resources, working_dir, [], environment=environment
        )
        
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get SLURM job status with enhanced information."""
        try:
            # Try to get status from queue first
            result = subprocess.run(
                ['squeue', '--job', job_id, '--format=%T,%S,%E,%M', '--noheader'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Job is in queue
                status_parts = result.stdout.strip().split(',')
                slurm_status = status_parts[0]
                
                # Map SLURM status to our status
                status_map = {
                    'PENDING': 'pending',
                    'RUNNING': 'running',
                    'COMPLETED': 'completed',
                    'FAILED': 'failed',
                    'CANCELLED': 'cancelled',
                    'TIMEOUT': 'failed',
                    'NODE_FAIL': 'failed',
                    'PREEMPTED': 'failed'
                }
                
                status = status_map.get(slurm_status, 'unknown')
                
                return JobStatus(job_id=job_id, status=status)
                
            else:
                # Job not in queue, check accounting
                return self._get_completed_job_status(job_id)
                
        except subprocess.CalledProcessError:
            return JobStatus(job_id, 'unknown', error_message='Failed to query job status')
            
    def _get_completed_job_status(self, job_id: str) -> JobStatus:
        """Get status of completed job from SLURM accounting."""
        try:
            result = subprocess.run(
                ['sacct', '--job', job_id, '--format=State,ExitCode,Start,End', '--noheader', '--parsable2'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split('|')
                    state = parts[0]
                    exit_code_str = parts[1] if len(parts) > 1 else '0:0'
                    start_time = parts[2] if len(parts) > 2 else None
                    end_time = parts[3] if len(parts) > 3 else None
                    
                    # Parse exit code
                    exit_code = 0
                    if ':' in exit_code_str:
                        exit_code = int(exit_code_str.split(':')[0])
                        
                    # Map state to our status
                    if state == 'COMPLETED':
                        status = 'completed'
                    else:
                        status = 'failed'
                        
                    return JobStatus(
                        job_id=job_id, 
                        status=status, 
                        exit_code=exit_code,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
        except subprocess.CalledProcessError:
            pass
            
        return JobStatus(job_id, 'unknown', error_message='Job status unavailable')
        
    def cancel_job(self, job_id: str) -> bool:
        """Cancel SLURM job."""
        try:
            subprocess.run(['scancel', job_id], check=True)
            logger.info(f"Cancelled SLURM job {job_id}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cancel SLURM job {job_id}: {e}")
            return False
            
    def get_supported_features(self) -> List[str]:
        """Get list of supported SLURM features."""
        return [
            'basic_submission',
            'status_query', 
            'job_cancellation',
            'job_arrays',
            'job_dependencies',
            'efficiency_metrics',
            'queue_info',
            'accounting_data'
        ]
