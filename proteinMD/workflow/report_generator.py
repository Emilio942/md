"""
Workflow Report Generator

Generates comprehensive reports after workflow execution, including analysis results,
performance metrics, and visualization summaries.
"""

import json
import html
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .workflow_definition import WorkflowDefinition
from .dependency_resolver import DependencyResolver, StepStatus

logger = logging.getLogger(__name__)


class WorkflowReportGenerator:
    """
    Generates comprehensive reports for workflow execution.
    
    Creates HTML and JSON reports summarizing workflow execution,
    step results, performance metrics, and analysis outputs.
    """
    
    def __init__(self, workflow: WorkflowDefinition, dependency_resolver: DependencyResolver,
                 output_directory: Path):
        """
        Initialize report generator.
        
        Args:
            workflow: Workflow definition
            dependency_resolver: Dependency resolver with execution status
            output_directory: Output directory for reports
        """
        self.workflow = workflow
        self.dependency_resolver = dependency_resolver
        self.output_directory = Path(output_directory)
        self.report_directory = self.output_directory / 'reports'
        
        # Ensure report directory exists
        self.report_directory.mkdir(parents=True, exist_ok=True)
        
    def generate_report(self, execution_time: Optional[float] = None,
                       success: bool = True) -> Path:
        """
        Generate comprehensive workflow report.
        
        Args:
            execution_time: Total execution time in seconds
            success: Whether workflow completed successfully
            
        Returns:
            Path to generated HTML report
        """
        try:
            logger.info("Generating workflow execution report...")
            
            # Generate JSON report
            json_report_path = self._generate_json_report(execution_time, success)
            
            # Generate HTML report
            html_report_path = self._generate_html_report(execution_time, success)
            
            # Generate summary file
            self._generate_summary_file(execution_time, success)
            
            logger.info(f"Workflow report generated: {html_report_path}")
            return html_report_path
            
        except Exception as e:
            logger.error(f"Failed to generate workflow report: {e}")
            raise
            
    def _generate_json_report(self, execution_time: Optional[float], success: bool) -> Path:
        """Generate JSON report with detailed execution data."""
        report_data = {
            'workflow': {
                'name': self.workflow.name,
                'description': self.workflow.description,
                'version': self.workflow.version,
                'author': self.workflow.author,
                'created': self.workflow.created
            },
            'execution': {
                'success': success,
                'execution_time_seconds': execution_time,
                'timestamp': datetime.now().isoformat(),
                'output_directory': str(self.output_directory)
            },
            'steps': [],
            'summary': self.dependency_resolver.get_execution_summary()
        }
        
        # Add step details
        for step_name, execution in self.dependency_resolver.step_executions.items():
            step_data = {
                'name': step_name,
                'command': execution.step.command,
                'status': execution.status.value,
                'start_time': execution.start_time,
                'end_time': execution.end_time,
                'exit_code': execution.exit_code,
                'error_message': execution.error_message,
                'retry_count': execution.retry_count,
                'dependencies': execution.step.dependencies,
                'parameters': execution.step.parameters,
                'outputs': execution.outputs,
                'resources': execution.step.resources
            }
            report_data['steps'].append(step_data)
            
        # Save JSON report
        json_path = self.report_directory / 'workflow_execution_report.json'
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        return json_path
        
    def _generate_html_report(self, execution_time: Optional[float], success: bool) -> Path:
        """Generate HTML report with visual presentation."""
        # Generate HTML content
        html_content = self._create_html_report(execution_time, success)
        
        # Save HTML report
        html_path = self.report_directory / 'workflow_execution_report.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
            
        return html_path
        
    def _create_html_report(self, execution_time: Optional[float], success: bool) -> str:
        """Create HTML report content."""
        # Calculate statistics
        total_steps = len(self.workflow.steps)
        completed_steps = len(self.dependency_resolver.get_completed_steps())
        failed_steps = len(self.dependency_resolver.get_failed_steps())
        
        success_rate = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        
        status_color = "#28a745" if success else "#dc3545"
        status_text = "SUCCESS" if success else "FAILED"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ProteinMD Workflow Report - {html.escape(self.workflow.name)}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 5px;
            background-color: {status_color};
            color: white;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .section {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 25px;
        }}
        
        .section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .step-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        
        .step-table th,
        .step-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        .step-table th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        
        .status-completed {{ color: #28a745; font-weight: bold; }}
        .status-failed {{ color: #dc3545; font-weight: bold; }}
        .status-skipped {{ color: #6c757d; font-weight: bold; }}
        .status-pending {{ color: #ffc107; font-weight: bold; }}
        
        .dependency-graph {{
            text-align: center;
            margin: 20px 0;
        }}
        
        .workflow-metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .metadata-item {{
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
        
        .metadata-label {{
            font-weight: bold;
            color: #495057;
        }}
        
        .error-details {{
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }}
        
        .step-outputs {{
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-family: monospace;
            font-size: 0.9em;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.3s ease;
            width: {success_rate}%;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 ProteinMD Workflow Report</h1>
        <div class="subtitle">{html.escape(self.workflow.name)}</div>
        <div class="status-badge">{status_text}</div>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{total_steps}</div>
            <div class="stat-label">Total Steps</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{completed_steps}</div>
            <div class="stat-label">Completed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{failed_steps}</div>
            <div class="stat-label">Failed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{success_rate:.1f}%</div>
            <div class="stat-label">Success Rate</div>
        </div>
        {f'''<div class="stat-card">
            <div class="stat-value">{execution_time:.1f}s</div>
            <div class="stat-label">Execution Time</div>
        </div>''' if execution_time else ''}
    </div>
    
    <div class="progress-bar">
        <div class="progress-fill"></div>
    </div>
    
    <div class="section">
        <h2>📋 Workflow Information</h2>
        <div class="workflow-metadata">
            <div class="metadata-item">
                <div class="metadata-label">Name:</div>
                {html.escape(self.workflow.name)}
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Description:</div>
                {html.escape(self.workflow.description)}
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Version:</div>
                {html.escape(self.workflow.version)}
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Author:</div>
                {html.escape(self.workflow.author)}
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Created:</div>
                {html.escape(self.workflow.created or 'Unknown')}
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Generated:</div>
                {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>⚙️ Step Execution Details</h2>
        {self._generate_step_details_html()}
    </div>
    
    <div class="section">
        <h2>📊 Execution Summary</h2>
        {self._generate_execution_summary_html()}
    </div>
    
    {self._generate_outputs_section_html()}
    
    <div class="footer">
        Generated by ProteinMD Workflow Engine v1.0.0<br>
        Execution completed on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}
    </div>
</body>
</html>
"""
        return html
        
    def _generate_step_details_html(self) -> str:
        """Generate HTML for step execution details."""
        html_parts = ['<table class="step-table">']
        html_parts.append('''
            <thead>
                <tr>
                    <th>Step</th>
                    <th>Command</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Dependencies</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
        ''')
        
        for step_name, execution in self.dependency_resolver.step_executions.items():
            # Calculate duration
            duration = ""
            if execution.start_time and execution.end_time:
                try:
                    start = datetime.fromisoformat(execution.start_time)
                    end = datetime.fromisoformat(execution.end_time)
                    duration_seconds = (end - start).total_seconds()
                    duration = f"{duration_seconds:.1f}s"
                except:
                    duration = "Unknown"
                    
            # Status styling
            status_class = f"status-{execution.status.value.replace('_', '-')}"
            
            # Dependencies
            deps_str = ", ".join(execution.step.dependencies) if execution.step.dependencies else "None"
            
            # Command (truncated)
            command = execution.step.command
            if len(command) > 50:
                command = command[:47] + "..."
                
            html_parts.append(f'''
                <tr>
                    <td><strong>{html.escape(step_name)}</strong></td>
                    <td><code>{html.escape(command)}</code></td>
                    <td class="{status_class}">{execution.status.value.upper()}</td>
                    <td>{duration}</td>
                    <td>{html.escape(deps_str)}</td>
                    <td>
            ''')
            
            # Add error details if failed
            if execution.status == StepStatus.FAILED and execution.error_message:
                html_parts.append(f'''
                    <div class="error-details">
                        <strong>Error:</strong> {html.escape(execution.error_message[:200])}
                        {f"..." if len(execution.error_message) > 200 else ""}
                    </div>
                ''')
                
            # Add outputs if completed
            if execution.status == StepStatus.COMPLETED and execution.outputs:
                outputs_str = ", ".join([f"{k}: {v}" for k, v in execution.outputs.items()])
                html_parts.append(f'''
                    <div class="step-outputs">
                        <strong>Outputs:</strong> {html.escape(outputs_str)}
                    </div>
                ''')
                
            html_parts.append('</td></tr>')
            
        html_parts.append('</tbody></table>')
        return "".join(html_parts)
        
    def _generate_execution_summary_html(self) -> str:
        """Generate HTML for execution summary."""
        summary = self.dependency_resolver.get_execution_summary()
        
        return f'''
            <div class="workflow-metadata">
                <div class="metadata-item">
                    <div class="metadata-label">Total Steps:</div>
                    {summary['total_steps']}
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Completed:</div>
                    {len(summary['completed_steps'])}
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Failed:</div>
                    {len(summary['failed_steps'])}
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Pending:</div>
                    {len(summary['pending_steps'])}
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Overall Status:</div>
                    {"✅ Success" if summary['is_successful'] else "❌ Failed"}
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Complete:</div>
                    {"Yes" if summary['is_complete'] else "No"}
                </div>
            </div>
        '''
        
    def _generate_outputs_section_html(self) -> str:
        """Generate HTML section for workflow outputs."""
        # Collect all outputs
        all_outputs = {}
        for step_name, execution in self.dependency_resolver.step_executions.items():
            if execution.outputs:
                all_outputs[step_name] = execution.outputs
                
        if not all_outputs:
            return ""
            
        html_parts = ['''
            <div class="section">
                <h2>📁 Generated Outputs</h2>
                <div class="workflow-metadata">
        ''']
        
        for step_name, outputs in all_outputs.items():
            for output_name, output_path in outputs.items():
                html_parts.append(f'''
                    <div class="metadata-item">
                        <div class="metadata-label">{html.escape(step_name)}.{html.escape(output_name)}:</div>
                        <code>{html.escape(output_path)}</code>
                    </div>
                ''')
                
        html_parts.append('</div></div>')
        return "".join(html_parts)
        
    def _generate_summary_file(self, execution_time: Optional[float], success: bool) -> None:
        """Generate simple summary file."""
        summary = {
            'workflow_name': self.workflow.name,
            'success': success,
            'execution_time_seconds': execution_time,
            'total_steps': len(self.workflow.steps),
            'completed_steps': len(self.dependency_resolver.get_completed_steps()),
            'failed_steps': len(self.dependency_resolver.get_failed_steps()),
            'timestamp': datetime.now().isoformat(),
            'output_directory': str(self.output_directory)
        }
        
        summary_path = self.report_directory / 'execution_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
    def generate_step_report(self, step_name: str) -> Optional[Path]:
        """Generate detailed report for a specific step."""
        if step_name not in self.dependency_resolver.step_executions:
            logger.warning(f"Step '{step_name}' not found")
            return None
            
        execution = self.dependency_resolver.step_executions[step_name]
        
        # Create step-specific report
        step_report = {
            'step_name': step_name,
            'command': execution.step.command,
            'status': execution.status.value,
            'start_time': execution.start_time,
            'end_time': execution.end_time,
            'exit_code': execution.exit_code,
            'error_message': execution.error_message,
            'retry_count': execution.retry_count,
            'parameters': execution.step.parameters,
            'dependencies': execution.step.dependencies,
            'outputs': execution.outputs,
            'resources': execution.step.resources
        }
        
        # Save step report
        step_report_path = self.report_directory / f'step_{step_name}_report.json'
        with open(step_report_path, 'w') as f:
            json.dump(step_report, f, indent=2)
            
        return step_report_path
