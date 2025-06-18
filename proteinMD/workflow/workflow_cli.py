"""
Workflow CLI Integration

Command-line interface for workflow automation, integrating with the existing
ProteinMD CLI system to provide workflow management capabilities.
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from .workflow_definition import (
    WorkflowDefinition, 
    WorkflowStep,
    load_workflow_from_file,
    create_example_workflow,
    get_builtin_workflow,
    list_builtin_workflows
)
from .workflow_engine import WorkflowEngine
from .job_scheduler import JobSchedulerManager

logger = logging.getLogger(__name__)


class WorkflowCLI:
    """
    Command-line interface for workflow automation.
    
    Provides commands for creating, running, monitoring, and managing workflows.
    """
    
    def __init__(self):
        """Initialize workflow CLI."""
        self.scheduler_manager = JobSchedulerManager()
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for workflow commands."""
        parser = argparse.ArgumentParser(
            description="ProteinMD Workflow Automation",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Create a new workflow
  proteinmd workflow create my_workflow.yaml --template protein_analysis

  # Run a workflow
  proteinmd workflow run my_workflow.yaml --input protein.pdb

  # Monitor workflow execution
  proteinmd workflow status workflow_output/

  # List available templates
  proteinmd workflow list-templates

  # Validate workflow definition
  proteinmd workflow validate my_workflow.yaml
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Workflow commands')
        
        # Create workflow command
        create_parser = subparsers.add_parser('create', help='Create new workflow')
        create_parser.add_argument('output_file', help='Output workflow file')
        create_parser.add_argument('--template', help='Workflow template to use')
        create_parser.add_argument('--name', help='Workflow name')
        create_parser.add_argument('--description', help='Workflow description')
        create_parser.add_argument('--author', help='Workflow author')
        
        # Run workflow command
        run_parser = subparsers.add_parser('run', help='Run workflow')
        run_parser.add_argument('workflow_file', help='Workflow definition file')
        run_parser.add_argument('--input', help='Input file (e.g., PDB file)')
        run_parser.add_argument('--output-dir', help='Output directory')
        run_parser.add_argument('--scheduler', help='Job scheduler to use',
                               choices=self.scheduler_manager.get_available_schedulers())
        run_parser.add_argument('--parameters', help='Parameter overrides (JSON)')
        run_parser.add_argument('--dry-run', action='store_true', help='Validate without running')
        run_parser.add_argument('--monitor', action='store_true', help='Monitor execution progress')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Check workflow status')
        status_parser.add_argument('output_dir', help='Workflow output directory')
        status_parser.add_argument('--detailed', action='store_true', help='Show detailed status')
        status_parser.add_argument('--refresh', type=int, default=0, help='Auto-refresh interval (seconds)')
        
        # Validate command
        validate_parser = subparsers.add_parser('validate', help='Validate workflow')
        validate_parser.add_argument('workflow_file', help='Workflow definition file')
        validate_parser.add_argument('--verbose', action='store_true', help='Verbose validation')
        
        # List templates command
        list_parser = subparsers.add_parser('list-templates', help='List workflow templates')
        list_parser.add_argument('--detailed', action='store_true', help='Show detailed information')
        
        # Stop workflow command
        stop_parser = subparsers.add_parser('stop', help='Stop running workflow')
        stop_parser.add_argument('output_dir', help='Workflow output directory')
        
        # Generate report command
        report_parser = subparsers.add_parser('report', help='Generate workflow report')
        report_parser.add_argument('output_dir', help='Workflow output directory')
        report_parser.add_argument('--format', choices=['html', 'json'], default='html',
                                  help='Report format')
        
        # Examples command
        examples_parser = subparsers.add_parser('examples', help='Show workflow examples')
        examples_parser.add_argument('--create', help='Create example workflow file')
        
        return parser
        
    def run(self, args: List[str] = None) -> int:
        """
        Run workflow CLI.
        
        Args:
            args: Command line arguments (None for sys.argv)
            
        Returns:
            Exit code
        """
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if not parsed_args.command:
            parser.print_help()
            return 1
            
        try:
            # Dispatch to command handlers
            if parsed_args.command == 'create':
                return self.cmd_create(parsed_args)
            elif parsed_args.command == 'run':
                return self.cmd_run(parsed_args)
            elif parsed_args.command == 'status':
                return self.cmd_status(parsed_args)
            elif parsed_args.command == 'validate':
                return self.cmd_validate(parsed_args)
            elif parsed_args.command == 'list-templates':
                return self.cmd_list_templates(parsed_args)
            elif parsed_args.command == 'stop':
                return self.cmd_stop(parsed_args)
            elif parsed_args.command == 'report':
                return self.cmd_report(parsed_args)
            elif parsed_args.command == 'examples':
                return self.cmd_examples(parsed_args)
            else:
                parser.print_help()
                return 1
                
        except Exception as e:
            logger.error(f"Command failed: {e}")
            return 1
            
    def cmd_create(self, args) -> int:
        """Create new workflow command."""
        print(f"🔧 Creating workflow: {args.output_file}")
        
        # Determine workflow source
        if args.template:
            if args.template in list_builtin_workflows():
                workflow = get_builtin_workflow(args.template)
                print(f"✅ Using built-in template: {args.template}")
            else:
                print(f"❌ Template '{args.template}' not found")
                print(f"Available templates: {', '.join(list_builtin_workflows())}")
                return 1
        else:
            # Create basic workflow
            workflow = WorkflowDefinition(
                name=args.name or "custom_workflow",
                description=args.description or "Custom workflow created via CLI",
                author=args.author or "Unknown"
            )
            
            # Add example step
            example_step = WorkflowStep(
                name="example_step",
                command="echo 'Hello, Workflow!'",
                parameters={"message": "Example step"}
            )
            workflow.add_step(example_step)
            
        # Override metadata if provided
        if args.name:
            workflow.name = args.name
        if args.description:
            workflow.description = args.description
        if args.author:
            workflow.author = args.author
            
        # Save workflow
        try:
            workflow.save(args.output_file)
            print(f"✅ Workflow created: {args.output_file}")
            print(f"   Name: {workflow.name}")
            print(f"   Steps: {len(workflow.steps)}")
            return 0
        except Exception as e:
            print(f"❌ Failed to create workflow: {e}")
            return 1
            
    def cmd_run(self, args) -> int:
        """Run workflow command."""
        print(f"🚀 Running workflow: {args.workflow_file}")
        
        try:
            # Load workflow
            workflow = load_workflow_from_file(args.workflow_file)
            print(f"✅ Loaded workflow: {workflow.name}")
            print(f"   Description: {workflow.description}")
            print(f"   Steps: {len(workflow.steps)}")
            
            # Apply parameter overrides
            if args.parameters:
                try:
                    param_overrides = json.loads(args.parameters)
                    workflow.global_parameters.update(param_overrides)
                    print(f"✅ Applied parameter overrides")
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid parameters JSON: {e}")
                    return 1
                    
            # Add input file to global parameters if provided
            if args.input:
                workflow.global_parameters['input_file'] = args.input
                print(f"✅ Set input file: {args.input}")
                
            # Determine output directory
            if args.output_dir:
                output_dir = Path(args.output_dir)
            else:
                # Create timestamped output directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path.cwd() / f"workflow_{workflow.name}_{timestamp}"
                
            output_dir.mkdir(parents=True, exist_ok=True)
            workflow.output_directory = str(output_dir)
            
            print(f"📁 Output directory: {output_dir}")
            
            # Dry run validation
            if args.dry_run:
                print("🔍 Performing dry run validation...")
                workflow.validate()
                print("✅ Workflow validation passed")
                return 0
                
            # Configure scheduler
            if args.scheduler:
                workflow.scheduler = {'type': args.scheduler}
                print(f"⚙️  Using scheduler: {args.scheduler}")
            else:
                default_scheduler = self.scheduler_manager.default_scheduler.name
                print(f"⚙️  Using default scheduler: {default_scheduler}")
                
            # Create and run workflow engine
            engine = WorkflowEngine(workflow, str(output_dir))
            
            # Set up progress monitoring if requested
            if args.monitor:
                engine.set_progress_callback(self._progress_callback)
                engine.set_step_callback(self._step_callback)
                
            print("🎯 Starting workflow execution...")
            success = engine.execute()
            
            if success:
                print("🎉 Workflow completed successfully!")
                print(f"📊 Results available in: {output_dir}")
                return 0
            else:
                print("❌ Workflow execution failed")
                print(f"📋 Check logs in: {output_dir}")
                return 1
                
        except Exception as e:
            print(f"❌ Workflow execution failed: {e}")
            logger.error(f"Workflow execution error: {e}", exc_info=True)
            return 1
            
    def cmd_status(self, args) -> int:
        """Check workflow status command."""
        output_dir = Path(args.output_dir)
        
        if not output_dir.exists():
            print(f"❌ Output directory not found: {output_dir}")
            return 1
            
        print(f"📊 Workflow Status: {output_dir}")
        
        # Look for status files
        status_file = output_dir / 'reports' / 'execution_summary.json'
        
        if status_file.exists():
            try:
                with open(status_file) as f:
                    status = json.load(f)
                    
                print(f"   Name: {status.get('workflow_name', 'Unknown')}")
                print(f"   Success: {'✅' if status.get('success') else '❌'}")
                print(f"   Total Steps: {status.get('total_steps', 0)}")
                print(f"   Completed: {status.get('completed_steps', 0)}")
                print(f"   Failed: {status.get('failed_steps', 0)}")
                
                if status.get('execution_time_seconds'):
                    exec_time = status['execution_time_seconds']
                    print(f"   Execution Time: {exec_time:.1f}s")
                    
                if status.get('timestamp'):
                    print(f"   Completed: {status['timestamp']}")
                    
            except Exception as e:
                print(f"❌ Failed to read status: {e}")
                return 1
        else:
            print("⏳ No execution summary found - workflow may still be running")
            
            # Check for workflow execution log
            log_file = output_dir / 'workflow_execution.log'
            if log_file.exists():
                print(f"📋 Log file: {log_file}")
                
                # Show last few lines
                if args.detailed:
                    try:
                        with open(log_file) as f:
                            lines = f.readlines()
                            print("\n📄 Recent log entries:")
                            for line in lines[-10:]:
                                print(f"   {line.rstrip()}")
                    except Exception as e:
                        print(f"❌ Failed to read log: {e}")
                        
        return 0
        
    def cmd_validate(self, args) -> int:
        """Validate workflow command."""
        print(f"🔍 Validating workflow: {args.workflow_file}")
        
        try:
            workflow = load_workflow_from_file(args.workflow_file)
            
            if args.verbose:
                print(f"   Name: {workflow.name}")
                print(f"   Description: {workflow.description}")
                print(f"   Steps: {len(workflow.steps)}")
                print(f"   Max Parallel: {workflow.max_parallel_steps}")
                
                print("\n📋 Step Dependencies:")
                for step in workflow.steps:
                    deps = ', '.join(step.dependencies) if step.dependencies else 'None'
                    print(f"   {step.name}: {deps}")
                    
            workflow.validate()
            print("✅ Workflow validation passed")
            return 0
            
        except Exception as e:
            print(f"❌ Workflow validation failed: {e}")
            return 1
            
    def cmd_list_templates(self, args) -> int:
        """List workflow templates command."""
        print("📋 Available Workflow Templates:")
        
        # Built-in templates
        builtin_templates = list_builtin_workflows()
        
        if builtin_templates:
            print("\n🔧 Built-in Templates:")
            for template_name in builtin_templates:
                try:
                    workflow = get_builtin_workflow(template_name)
                    print(f"   • {template_name}: {workflow.description}")
                    
                    if args.detailed:
                        print(f"     Steps: {len(workflow.steps)}")
                        print(f"     Author: {workflow.author}")
                        print(f"     Version: {workflow.version}")
                        
                except Exception as e:
                    print(f"   • {template_name}: Error loading template - {e}")
        else:
            print("   No built-in templates found")
            
        return 0
        
    def cmd_stop(self, args) -> int:
        """Stop workflow command."""
        print(f"🛑 Stopping workflow in: {args.output_dir}")
        
        # This is a simplified implementation
        # In a real system, you'd need to track running engines
        print("⚠️  Workflow stop functionality requires active monitoring")
        print("    Use Ctrl+C to stop locally running workflows")
        print("    Use scheduler-specific commands (e.g., scancel) for HPC jobs")
        
        return 0
        
    def cmd_report(self, args) -> int:
        """Generate workflow report command."""
        output_dir = Path(args.output_dir)
        
        if not output_dir.exists():
            print(f"❌ Output directory not found: {output_dir}")
            return 1
            
        print(f"📊 Generating workflow report: {args.format}")
        
        # Look for existing reports
        report_dir = output_dir / 'reports'
        
        if args.format == 'html':
            report_file = report_dir / 'workflow_execution_report.html'
        else:
            report_file = report_dir / 'workflow_execution_report.json'
            
        if report_file.exists():
            print(f"✅ Report found: {report_file}")
            return 0
        else:
            print(f"❌ Report not found: {report_file}")
            print("   Reports are generated automatically during workflow execution")
            return 1
            
    def cmd_examples(self, args) -> int:
        """Show workflow examples command."""
        if args.create:
            print(f"📝 Creating example workflow: {args.create}")
            
            try:
                example_workflow = create_example_workflow()
                example_workflow.save(args.create)
                print(f"✅ Example workflow created: {args.create}")
                return 0
            except Exception as e:
                print(f"❌ Failed to create example: {e}")
                return 1
        else:
            print("📋 Workflow Examples:")
            print("""
1. Basic Protein Analysis Pipeline:
   proteinmd workflow create analysis.yaml --template protein_analysis_pipeline
   proteinmd workflow run analysis.yaml --input protein.pdb

2. Custom Workflow:
   proteinmd workflow create custom.yaml --name "My Analysis"
   # Edit custom.yaml to add your steps
   proteinmd workflow run custom.yaml

3. HPC Execution:
   proteinmd workflow run analysis.yaml --scheduler slurm --input protein.pdb

4. Monitor Execution:
   proteinmd workflow run analysis.yaml --monitor --input protein.pdb
   proteinmd workflow status workflow_output/
            """)
            
        return 0
        
    def _progress_callback(self, message: str, data: Dict[str, Any]) -> None:
        """Progress callback for monitoring."""
        print(f"📈 {message}")
        
    def _step_callback(self, step_name: str, status: str, error_message: Optional[str]) -> None:
        """Step status callback for monitoring."""
        status_icons = {
            'pending': '⏳',
            'running': '🔄',
            'completed': '✅',
            'failed': '❌',
            'skipped': '⏭️'
        }
        
        icon = status_icons.get(status, '❓')
        print(f"   {icon} {step_name}: {status.upper()}")
        
        if error_message:
            print(f"      Error: {error_message}")


def main():
    """Main entry point for workflow CLI."""
    cli = WorkflowCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())
