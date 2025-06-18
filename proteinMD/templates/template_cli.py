"""
Template CLI Extension for ProteinMD

This module extends the existing CLI with enhanced template management capabilities.
It provides comprehensive template operations beyond the basic template support
already available in the main CLI.
"""

import argparse
import json
import yaml
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from .template_manager import TemplateManager
from .base_template import TemplateValidationError


class TemplateCLI:
    """
    Extended CLI interface for template management.
    
    This class provides advanced template operations including
    search, validation, export/import, and template development tools.
    """
    
    def __init__(self):
        """Initialize template CLI."""
        self.manager = TemplateManager()
        
    def list_templates(self, args: argparse.Namespace) -> int:
        """
        List available templates with filtering options.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            # Apply filters
            templates = self.manager.search_templates(
                query=args.query,
                tags=args.tags,
                source=args.source
            )
            
            if not templates:
                print("No templates found matching criteria.")
                return 0
                
            print(f"\n📋 Found {len(templates)} Template(s)")
            print("=" * 60)
            
            # Group by source
            builtin_templates = {k: v for k, v in templates.items() if v['source'] == 'builtin'}
            user_templates = {k: v for k, v in templates.items() if v['source'] == 'user'}
            
            if builtin_templates:
                print("\n🔧 Built-in Templates:")
                for name, info in builtin_templates.items():
                    tags_str = ', '.join(info.get('tags', []))
                    print(f"  • {name}")
                    print(f"    Description: {info['description']}")
                    print(f"    Version: {info['version']}")
                    print(f"    Tags: {tags_str}")
                    print(f"    Parameters: {info['parameter_count']}")
                    print()
                    
            if user_templates:
                print("\n👤 User Templates:")
                for name, info in user_templates.items():
                    tags_str = ', '.join(info.get('tags', []))
                    print(f"  • {name}")
                    print(f"    Description: {info['description']}")
                    print(f"    Author: {info.get('author', 'Unknown')}")
                    print(f"    Version: {info['version']}")
                    print(f"    Tags: {tags_str}")
                    print(f"    Parameters: {info['parameter_count']}")
                    print()
                    
            # Show statistics if requested
            if args.show_stats:
                stats = self.manager.get_statistics()
                print("\n📊 Template Statistics:")
                print(f"  Total templates: {stats['total_templates']}")
                print(f"  Built-in: {stats['builtin_templates']}")
                print(f"  User: {stats['user_templates']}")
                
                if stats['templates_by_tag']:
                    print("\n  Popular tags:")
                    for tag, count in sorted(stats['templates_by_tag'].items(), 
                                           key=lambda x: x[1], reverse=True)[:5]:
                        print(f"    {tag}: {count}")
                        
            return 0
            
        except Exception as e:
            print(f"Error listing templates: {e}")
            return 1
            
    def show_template(self, args: argparse.Namespace) -> int:
        """
        Show detailed information about a specific template.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            template = self.manager.get_template(args.name)
            
            print(f"\n📄 Template: {template.name}")
            print("=" * 50)
            print(f"Description: {template.description}")
            print(f"Version: {template.version}")
            print(f"Author: {template.author}")
            print(f"Tags: {', '.join(template.tags)}")
            
            if template.dependencies:
                print(f"Dependencies: {', '.join(template.dependencies)}")
                
            print(f"\n⚙️ Parameters ({len(template.parameters)}):")
            if template.parameters:
                for name, param in template.parameters.items():
                    required_str = " (required)" if param.required else ""
                    units_str = f" [{param.units}]" if param.units else ""
                    print(f"  • {name}{required_str}{units_str}")
                    print(f"    {param.description}")
                    print(f"    Type: {param.parameter_type}")
                    print(f"    Default: {param.default_value}")
                    
                    if param.allowed_values:
                        print(f"    Allowed: {param.allowed_values}")
                    if param.min_value is not None:
                        print(f"    Min: {param.min_value}")
                    if param.max_value is not None:
                        print(f"    Max: {param.max_value}")
                    print()
            else:
                print("  No configurable parameters")
                
            # Show sample configuration if requested
            if args.show_config:
                print("\n🔧 Default Configuration:")
                config = template.get_default_config()
                if args.format == 'yaml':
                    print(yaml.dump(config, default_flow_style=False, indent=2))
                else:
                    print(json.dumps(config, indent=2))
                    
            return 0
            
        except KeyError:
            print(f"Template '{args.name}' not found")
            return 1
        except Exception as e:
            print(f"Error showing template: {e}")
            return 1
            
    def create_template(self, args: argparse.Namespace) -> int:
        """
        Create a new user template.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            # Load configuration from file
            config_path = Path(args.config_file)
            if not config_path.exists():
                print(f"Configuration file not found: {config_path}")
                return 1
                
            if config_path.suffix.lower() == '.json':
                with open(config_path) as f:
                    config = json.load(f)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
            else:
                print("Configuration file must be JSON or YAML format")
                return 1
                
            # Create template
            template = self.manager.create_template_from_config(
                name=args.name,
                description=args.description,
                config=config,
                author=args.author or "User",
                tags=args.tags or []
            )
            
            # Save template
            self.manager.save_user_template(template, overwrite=args.overwrite)
            
            print(f"✅ Template '{args.name}' created successfully")
            return 0
            
        except FileExistsError:
            print(f"Template '{args.name}' already exists. Use --overwrite to replace.")
            return 1
        except Exception as e:
            print(f"Error creating template: {e}")
            return 1
            
    def validate_template(self, args: argparse.Namespace) -> int:
        """
        Validate a template and optionally test parameters.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            template = self.manager.get_template(args.name)
            
            print(f"🔍 Validating template '{args.name}'...")
            
            # Basic template validation
            if not template.description:
                print("⚠️  Warning: Template has no description")
                
            if not template.parameters:
                print("ℹ️  Note: Template has no configurable parameters")
                
            # Parameter validation
            if args.parameters_file:
                param_path = Path(args.parameters_file)
                if param_path.suffix.lower() == '.json':
                    with open(param_path) as f:
                        params = json.load(f)
                elif param_path.suffix.lower() in ['.yaml', '.yml']:
                    with open(param_path) as f:
                        params = yaml.safe_load(f)
                else:
                    print("Parameters file must be JSON or YAML format")
                    return 1
                    
                template.validate_parameters(params)
                print("✅ Parameter validation passed")
                
                # Generate and validate configuration
                config = template.generate_config(**params)
                print("✅ Configuration generation successful")
                
            print("✅ Template validation completed successfully")
            return 0
            
        except KeyError:
            print(f"Template '{args.name}' not found")
            return 1
        except TemplateValidationError as e:
            print(f"❌ Validation failed: {e}")
            return 1
        except Exception as e:
            print(f"Error validating template: {e}")
            return 1
            
    def export_template(self, args: argparse.Namespace) -> int:
        """
        Export a template to a file.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            self.manager.export_template(args.name, args.output_file, args.format)
            print(f"✅ Template '{args.name}' exported to {args.output_file}")
            return 0
            
        except KeyError:
            print(f"Template '{args.name}' not found")
            return 1
        except Exception as e:
            print(f"Error exporting template: {e}")
            return 1
            
    def import_template(self, args: argparse.Namespace) -> int:
        """
        Import a template from a file.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            template = self.manager.import_template(args.template_file, 
                                                  overwrite=args.overwrite)
            print(f"✅ Template '{template.name}' imported successfully")
            return 0
            
        except FileExistsError:
            print("Template already exists. Use --overwrite to replace.")
            return 1
        except Exception as e:
            print(f"Error importing template: {e}")
            return 1
            
    def delete_template(self, args: argparse.Namespace) -> int:
        """
        Delete a user template.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            if not args.force:
                response = input(f"Delete template '{args.name}'? [y/N]: ")
                if response.lower() not in ['y', 'yes']:
                    print("Deletion cancelled")
                    return 0
                    
            self.manager.delete_user_template(args.name)
            print(f"✅ Template '{args.name}' deleted successfully")
            return 0
            
        except KeyError:
            print(f"User template '{args.name}' not found")
            return 1
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        except Exception as e:
            print(f"Error deleting template: {e}")
            return 1
            
    def generate_config(self, args: argparse.Namespace) -> int:
        """
        Generate configuration from template with parameters.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            template = self.manager.get_template(args.name)
            
            # Load parameters if provided
            params = {}
            if args.parameters_file:
                param_path = Path(args.parameters_file)
                if param_path.suffix.lower() == '.json':
                    with open(param_path) as f:
                        params = json.load(f)
                elif param_path.suffix.lower() in ['.yaml', '.yml']:
                    with open(param_path) as f:
                        params = yaml.safe_load(f)
                        
            # Generate configuration
            config = template.generate_config(**params)
            
            # Output configuration
            if args.output_file:
                output_path = Path(args.output_file)
                if args.format == 'yaml' or output_path.suffix.lower() in ['.yaml', '.yml']:
                    with open(output_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False, indent=2)
                else:
                    with open(output_path, 'w') as f:
                        json.dump(config, f, indent=2)
                print(f"✅ Configuration saved to {output_path}")
            else:
                if args.format == 'yaml':
                    print(yaml.dump(config, default_flow_style=False, indent=2))
                else:
                    print(json.dumps(config, indent=2))
                    
            return 0
            
        except KeyError:
            print(f"Template '{args.name}' not found")
            return 1
        except TemplateValidationError as e:
            print(f"Parameter validation failed: {e}")
            return 1
        except Exception as e:
            print(f"Error generating configuration: {e}")
            return 1
            
    def template_statistics(self, args: argparse.Namespace) -> int:
        """
        Show template library statistics.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            stats = self.manager.get_statistics()
            
            print("\n📊 Template Library Statistics")
            print("=" * 40)
            print(f"Total templates: {stats['total_templates']}")
            print(f"Built-in templates: {stats['builtin_templates']}")
            print(f"User templates: {stats['user_templates']}")
            
            if stats['templates_by_tag']:
                print(f"\n🏷️  Templates by Tag:")
                for tag, count in sorted(stats['templates_by_tag'].items(), 
                                       key=lambda x: x[1], reverse=True):
                    print(f"  {tag}: {count}")
                    
            if stats['templates_by_author']:
                print(f"\n👥 Templates by Author:")
                for author, count in sorted(stats['templates_by_author'].items(), 
                                          key=lambda x: x[1], reverse=True):
                    print(f"  {author}: {count}")
                    
            return 0
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return 1


def create_template_parser() -> argparse.ArgumentParser:
    """Create argument parser for template CLI."""
    parser = argparse.ArgumentParser(
        prog='proteinmd-templates',
        description='Advanced template management for ProteinMD'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Template commands')
    
    # List templates
    list_parser = subparsers.add_parser('list', help='List available templates')
    list_parser.add_argument('--query', help='Search query')
    list_parser.add_argument('--tags', nargs='+', help='Filter by tags')
    list_parser.add_argument('--source', choices=['builtin', 'user'], 
                           help='Filter by source')
    list_parser.add_argument('--show-stats', action='store_true',
                           help='Show template statistics')
    
    # Show template details
    show_parser = subparsers.add_parser('show', help='Show template details')
    show_parser.add_argument('name', help='Template name')
    show_parser.add_argument('--show-config', action='store_true',
                           help='Show default configuration')
    show_parser.add_argument('--format', choices=['json', 'yaml'], default='json',
                           help='Configuration format')
    
    # Create template
    create_parser = subparsers.add_parser('create', help='Create user template')
    create_parser.add_argument('name', help='Template name')
    create_parser.add_argument('description', help='Template description')
    create_parser.add_argument('config_file', help='Configuration file')
    create_parser.add_argument('--author', help='Template author')
    create_parser.add_argument('--tags', nargs='+', help='Template tags')
    create_parser.add_argument('--overwrite', action='store_true',
                             help='Overwrite existing template')
    
    # Validate template
    validate_parser = subparsers.add_parser('validate', help='Validate template')
    validate_parser.add_argument('name', help='Template name')
    validate_parser.add_argument('--parameters-file', help='Parameters file to test')
    
    # Export template
    export_parser = subparsers.add_parser('export', help='Export template')
    export_parser.add_argument('name', help='Template name')
    export_parser.add_argument('output_file', help='Output file')
    export_parser.add_argument('--format', choices=['json', 'yaml'], default='json',
                             help='Export format')
    
    # Import template
    import_parser = subparsers.add_parser('import', help='Import template')
    import_parser.add_argument('template_file', help='Template file to import')
    import_parser.add_argument('--overwrite', action='store_true',
                             help='Overwrite existing template')
    
    # Delete template
    delete_parser = subparsers.add_parser('delete', help='Delete user template')
    delete_parser.add_argument('name', help='Template name')
    delete_parser.add_argument('--force', action='store_true',
                             help='Skip confirmation')
    
    # Generate configuration
    generate_parser = subparsers.add_parser('generate-config', 
                                          help='Generate configuration from template')
    generate_parser.add_argument('name', help='Template name')
    generate_parser.add_argument('--parameters-file', help='Parameters file')
    generate_parser.add_argument('--output-file', help='Output configuration file')
    generate_parser.add_argument('--format', choices=['json', 'yaml'], default='json',
                               help='Output format')
    
    # Statistics
    stats_parser = subparsers.add_parser('stats', help='Show template statistics')
    
    return parser


def main():
    """Main entry point for template CLI."""
    parser = create_template_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    cli = TemplateCLI()
    
    # Dispatch to appropriate command handler
    command_handlers = {
        'list': cli.list_templates,
        'show': cli.show_template,
        'create': cli.create_template,
        'validate': cli.validate_template,
        'export': cli.export_template,
        'import': cli.import_template,
        'delete': cli.delete_template,
        'generate-config': cli.generate_config,
        'stats': cli.template_statistics
    }
    
    handler = command_handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
