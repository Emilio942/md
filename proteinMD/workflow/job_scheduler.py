"""
Job Scheduler Integration for Workflow Automation

Provides integration with various job scheduling systems including SLURM, PBS, SGE,
and local execution for high-performance computing environments.
"""

import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class JobResource:
    """Resource requirements for a job."""
    cpu_cores: int = 1
    memory_gb: int = 4
    time_limit: str = "01:00:00"  # HH:MM:SS format
    gpu_count: int = 0
    partition: Optional[str] = None
    queue: Optional[str] = None
    exclusive: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'time_limit': self.time_limit,
            'gpu_count': self.gpu_count,
            'partition': self.partition,
            'queue': self.queue,
            'exclusive': self.exclusive
        }


@dataclass
class JobStatus:
    """Job execution status."""
    job_id: str
    status: str  # pending, running, completed, failed, cancelled
    exit_code: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error_message: Optional[str] = None
    
    def is_active(self) -> bool:
        """Check if job is still active (pending or running)."""
        return self.status in ['pending', 'running']
        
    def is_complete(self) -> bool:
        """Check if job is complete (finished execution)."""
        return self.status in ['completed', 'failed', 'cancelled']
        
    def is_successful(self) -> bool:
        """Check if job completed successfully."""
        return self.status == 'completed' and (self.exit_code is None or self.exit_code == 0)


class JobScheduler(ABC):
    """Abstract base class for job schedulers."""
    
    def __init__(self, name: str):
        """
        Initialize job scheduler.
        
        Args:
            name: Scheduler name
        """
        self.name = name
        self.submitted_jobs: Dict[str, Dict[str, Any]] = {}
        
    @abstractmethod
    def is_available(self) -> bool:
        """Check if scheduler is available on the system."""
        pass
        
    @abstractmethod
    def submit_job(self, command: str, job_name: str, resources: JobResource,
                   working_dir: Path, environment: Dict[str, str] = None) -> str:
        """
        Submit a job to the scheduler.
        
        Args:
            command: Command to execute
            job_name: Name for the job
            resources: Resource requirements
            working_dir: Working directory for job execution
            environment: Environment variables
            
        Returns:
            Job ID
        """
        pass
        
    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get status of a submitted job."""
        pass
        
    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        pass
        
    def wait_for_job(self, job_id: str, poll_interval: int = 30) -> JobStatus:
        """
        Wait for job completion.
        
        Args:
            job_id: Job ID to wait for
            poll_interval: Polling interval in seconds
            
        Returns:
            Final job status
        """
        logger.info(f"Waiting for job {job_id} to complete...")
        
        while True:
            status = self.get_job_status(job_id)
            
            if status.is_complete():
                logger.info(f"Job {job_id} completed with status: {status.status}")
                return status
                
            logger.debug(f"Job {job_id} status: {status.status}")
            time.sleep(poll_interval)
            
    def get_supported_features(self) -> List[str]:
        """Get list of supported features."""
        return ['basic_submission', 'status_query', 'job_cancellation']


class LocalScheduler(JobScheduler):
    """Local execution scheduler (no actual scheduling system)."""
    
    def __init__(self):
        super().__init__("local")
        self.job_counter = 0
        
    def is_available(self) -> bool:
        """Local scheduler is always available."""
        return True
        
    def submit_job(self, command: str, job_name: str, resources: JobResource,
                   working_dir: Path, environment: Dict[str, str] = None) -> str:
        """Submit job for local execution."""
        self.job_counter += 1
        job_id = f"local_{self.job_counter}"
        
        # Store job information
        self.submitted_jobs[job_id] = {
            'command': command,
            'job_name': job_name,
            'resources': resources.to_dict(),
            'working_dir': str(working_dir),
            'environment': environment or {},
            'status': 'pending',
            'submit_time': time.time()
        }
        
        logger.info(f"Submitted local job {job_id}: {job_name}")
        
        # Execute immediately in background
        self._execute_local_job(job_id)
        
        return job_id
        
    def _execute_local_job(self, job_id: str) -> None:
        """Execute job locally."""
        job_info = self.submitted_jobs[job_id]
        
        try:
            # Update status to running
            job_info['status'] = 'running'
            job_info['start_time'] = time.time()
            
            # Prepare environment
            env = os.environ.copy()
            env.update(job_info['environment'])
            
            # Execute command
            process = subprocess.run(
                job_info['command'],
                shell=True,
                cwd=job_info['working_dir'],
                env=env,
                capture_output=True,
                text=True
            )
            
            # Update job status
            job_info['end_time'] = time.time()
            job_info['exit_code'] = process.returncode
            
            if process.returncode == 0:
                job_info['status'] = 'completed'
            else:
                job_info['status'] = 'failed'
                job_info['error_message'] = process.stderr
                
            # Save execution log
            log_file = Path(job_info['working_dir']) / f'{job_id}_execution.log'
            with open(log_file, 'w') as f:
                f.write(f"Job ID: {job_id}\n")
                f.write(f"Command: {job_info['command']}\n")
                f.write(f"Exit code: {process.returncode}\n")
                f.write(f"\n--- STDOUT ---\n{process.stdout}\n")
                f.write(f"\n--- STDERR ---\n{process.stderr}\n")
                
        except Exception as e:
            job_info['status'] = 'failed'
            job_info['error_message'] = str(e)
            job_info['end_time'] = time.time()
            
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get status of local job."""
        if job_id not in self.submitted_jobs:
            return JobStatus(job_id, 'unknown', error_message='Job not found')
            
        job_info = self.submitted_jobs[job_id]
        
        return JobStatus(
            job_id=job_id,
            status=job_info['status'],
            exit_code=job_info.get('exit_code'),
            start_time=job_info.get('start_time'),
            end_time=job_info.get('end_time'),
            error_message=job_info.get('error_message')
        )
        
    def cancel_job(self, job_id: str) -> bool:
        """Cancel local job (not really possible for local execution)."""
        if job_id in self.submitted_jobs:
            self.submitted_jobs[job_id]['status'] = 'cancelled'
            return True
        return False


class SLURMScheduler(JobScheduler):
    """SLURM workload manager integration."""
    
    def __init__(self):
        super().__init__("slurm")
        
    def is_available(self) -> bool:
        """Check if SLURM is available."""
        try:
            subprocess.run(['sinfo', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
            
    def submit_job(self, command: str, job_name: str, resources: JobResource,
                   working_dir: Path, environment: Dict[str, str] = None) -> str:
        """Submit job to SLURM."""
        # Create SLURM script
        script_content = self._create_slurm_script(
            command, job_name, resources, working_dir, environment
        )
        
        # Write script to file
        script_file = working_dir / f'{job_name}_slurm.sh'
        with open(script_file, 'w') as f:
            f.write(script_content)
            
        script_file.chmod(0o755)
        
        # Submit job
        try:
            result = subprocess.run(
                ['sbatch', str(script_file)],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract job ID from output
            # SLURM output: "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1]
            
            logger.info(f"Submitted SLURM job {job_id}: {job_name}")
            return job_id
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to submit SLURM job: {e.stderr}")
            
    def _create_slurm_script(self, command: str, job_name: str, resources: JobResource,
                            working_dir: Path, environment: Dict[str, str] = None) -> str:
        """Create SLURM batch script."""
        script = "#!/bin/bash\n\n"
        
        # SLURM directives
        script += f"#SBATCH --job-name={job_name}\n"
        script += f"#SBATCH --ntasks=1\n"
        script += f"#SBATCH --cpus-per-task={resources.cpu_cores}\n"
        script += f"#SBATCH --mem={resources.memory_gb}G\n"
        script += f"#SBATCH --time={resources.time_limit}\n"
        
        if resources.gpu_count > 0:
            script += f"#SBATCH --gres=gpu:{resources.gpu_count}\n"
            
        if resources.partition:
            script += f"#SBATCH --partition={resources.partition}\n"
            
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
        
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get SLURM job status."""
        try:
            result = subprocess.run(
                ['squeue', '--job', job_id, '--format=%T,%S,%E', '--noheader'],
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
                    'NODE_FAIL': 'failed'
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
                ['sacct', '--job', job_id, '--format=State,ExitCode', '--noheader', '--parsable2'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split('|')
                    state = parts[0]
                    exit_code_str = parts[1] if len(parts) > 1 else '0:0'
                    
                    # Parse exit code
                    exit_code = 0
                    if ':' in exit_code_str:
                        exit_code = int(exit_code_str.split(':')[0])
                        
                    # Map state to our status
                    if state == 'COMPLETED':
                        status = 'completed'
                    else:
                        status = 'failed'
                        
                    return JobStatus(job_id=job_id, status=status, exit_code=exit_code)
                    
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


class PBSScheduler(JobScheduler):
    """PBS/Torque scheduler integration."""
    
    def __init__(self):
        super().__init__("pbs")
        
    def is_available(self) -> bool:
        """Check if PBS is available."""
        try:
            subprocess.run(['qstat', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
            
    def submit_job(self, command: str, job_name: str, resources: JobResource,
                   working_dir: Path, environment: Dict[str, str] = None) -> str:
        """Submit job to PBS."""
        # Create PBS script
        script_content = self._create_pbs_script(
            command, job_name, resources, working_dir, environment
        )
        
        # Write script to file
        script_file = working_dir / f'{job_name}_pbs.sh'
        with open(script_file, 'w') as f:
            f.write(script_content)
            
        script_file.chmod(0o755)
        
        # Submit job
        try:
            result = subprocess.run(
                ['qsub', str(script_file)],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract job ID from output
            job_id = result.stdout.strip()
            
            logger.info(f"Submitted PBS job {job_id}: {job_name}")
            return job_id
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to submit PBS job: {e.stderr}")
            
    def _create_pbs_script(self, command: str, job_name: str, resources: JobResource,
                          working_dir: Path, environment: Dict[str, str] = None) -> str:
        """Create PBS batch script."""
        script = "#!/bin/bash\n\n"
        
        # PBS directives
        script += f"#PBS -N {job_name}\n"
        script += f"#PBS -l nodes=1:ppn={resources.cpu_cores}\n"
        script += f"#PBS -l mem={resources.memory_gb}gb\n"
        script += f"#PBS -l walltime={resources.time_limit}\n"
        
        if resources.queue:
            script += f"#PBS -q {resources.queue}\n"
            
        # Output files
        script += f"#PBS -o {working_dir}/{job_name}.out\n"
        script += f"#PBS -e {working_dir}/{job_name}.err\n"
        
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
        
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get PBS job status."""
        try:
            result = subprocess.run(
                ['qstat', '-f', job_id],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Parse qstat output
                lines = result.stdout.split('\n')
                job_state = None
                exit_status = None
                
                for line in lines:
                    if 'job_state =' in line:
                        job_state = line.split('=')[1].strip()
                    elif 'exit_status =' in line:
                        exit_status = int(line.split('=')[1].strip())
                        
                # Map PBS state to our status
                status_map = {
                    'Q': 'pending',
                    'R': 'running',
                    'C': 'completed',
                    'E': 'failed',
                    'H': 'pending'
                }
                
                status = status_map.get(job_state, 'unknown')
                
                return JobStatus(job_id=job_id, status=status, exit_code=exit_status)
                
        except subprocess.CalledProcessError:
            pass
            
        return JobStatus(job_id, 'unknown', error_message='Failed to query job status')
        
    def cancel_job(self, job_id: str) -> bool:
        """Cancel PBS job."""
        try:
            subprocess.run(['qdel', job_id], check=True)
            logger.info(f"Cancelled PBS job {job_id}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cancel PBS job {job_id}: {e}")
            return False


class JobSchedulerManager:
    """
    Manager for different job schedulers.
    
    Automatically detects available schedulers and provides unified interface.
    """
    
    def __init__(self):
        """Initialize scheduler manager."""
        self.schedulers: Dict[str, JobScheduler] = {}
        self.default_scheduler: Optional[JobScheduler] = None
        
        # Initialize available schedulers
        self._initialize_schedulers()
        
    def _initialize_schedulers(self) -> None:
        """Initialize and detect available schedulers."""
        # Always available local scheduler
        local = LocalScheduler()
        self.schedulers['local'] = local
        self.default_scheduler = local
        
        # Try to initialize HPC schedulers
        schedulers_to_try = [
            ('slurm', SLURMScheduler),
            ('pbs', PBSScheduler),
        ]
        
        for name, scheduler_class in schedulers_to_try:
            try:
                scheduler = scheduler_class()
                if scheduler.is_available():
                    self.schedulers[name] = scheduler
                    # Prefer HPC schedulers over local
                    if self.default_scheduler.name == 'local':
                        self.default_scheduler = scheduler
                    logger.info(f"Detected {name.upper()} scheduler")
                else:
                    logger.debug(f"{name.upper()} scheduler not available")
            except Exception as e:
                logger.debug(f"Failed to initialize {name} scheduler: {e}")
                
    def get_available_schedulers(self) -> List[str]:
        """Get list of available scheduler names."""
        return list(self.schedulers.keys())
        
    def get_scheduler(self, name: Optional[str] = None) -> JobScheduler:
        """
        Get scheduler by name.
        
        Args:
            name: Scheduler name (None for default)
            
        Returns:
            JobScheduler instance
        """
        if name is None:
            return self.default_scheduler
            
        if name not in self.schedulers:
            raise ValueError(f"Scheduler '{name}' not available. Available: {self.get_available_schedulers()}")
            
        return self.schedulers[name]
        
    def submit_workflow_step(self, command: str, step_name: str, 
                           resources: Optional[Dict[str, Any]] = None,
                           working_dir: Optional[Path] = None,
                           scheduler_name: Optional[str] = None,
                           environment: Dict[str, str] = None) -> tuple[str, str]:
        """
        Submit workflow step to scheduler.
        
        Args:
            command: Command to execute
            step_name: Step name
            resources: Resource requirements
            working_dir: Working directory
            scheduler_name: Scheduler to use
            environment: Environment variables
            
        Returns:
            Tuple of (job_id, scheduler_name)
        """
        scheduler = self.get_scheduler(scheduler_name)
        
        # Convert resources
        if resources:
            job_resources = JobResource(**resources)
        else:
            job_resources = JobResource()
            
        if working_dir is None:
            working_dir = Path.cwd()
            
        job_id = scheduler.submit_job(
            command=command,
            job_name=step_name,
            resources=job_resources,
            working_dir=working_dir,
            environment=environment
        )
        
        return job_id, scheduler.name
        
    def get_job_status(self, job_id: str, scheduler_name: str) -> JobStatus:
        """Get job status from specific scheduler."""
        scheduler = self.get_scheduler(scheduler_name)
        return scheduler.get_job_status(job_id)
        
    def cancel_job(self, job_id: str, scheduler_name: str) -> bool:
        """Cancel job on specific scheduler."""
        scheduler = self.get_scheduler(scheduler_name)
        return scheduler.cancel_job(job_id)
        
    def wait_for_jobs(self, jobs: List[tuple[str, str]], poll_interval: int = 30) -> Dict[str, JobStatus]:
        """
        Wait for multiple jobs to complete.
        
        Args:
            jobs: List of (job_id, scheduler_name) tuples
            poll_interval: Polling interval in seconds
            
        Returns:
            Dictionary mapping job_id to final JobStatus
        """
        job_statuses = {}
        pending_jobs = jobs.copy()
        
        while pending_jobs:
            completed_jobs = []
            
            for job_id, scheduler_name in pending_jobs:
                status = self.get_job_status(job_id, scheduler_name)
                
                if status.is_complete():
                    job_statuses[job_id] = status
                    completed_jobs.append((job_id, scheduler_name))
                    logger.info(f"Job {job_id} completed with status: {status.status}")
                    
            # Remove completed jobs from pending list
            for job in completed_jobs:
                pending_jobs.remove(job)
                
            if pending_jobs:
                logger.debug(f"Waiting for {len(pending_jobs)} jobs to complete...")
                time.sleep(poll_interval)
                
        return job_statuses
