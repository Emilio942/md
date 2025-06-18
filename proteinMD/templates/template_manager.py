"""
Template Manager for ProteinMD Simulation Templates

This module provides centralized management of simulation templates,
including built-in templates, user templates, and template operations.
"""

import json
import yaml
import shutil
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datetime import datetime

from .base_template import BaseTemplate, GenericTemplate, TemplateValidationError
from .builtin_templates import (
    ProteinFoldingTemplate,
    EquilibrationTemplate, 
    FreeEnergyTemplate,
    MembraneProteinTemplate,
    LigandBindingTemplate,
    EnhancedSamplingTemplate,
    DrugDiscoveryTemplate,
    StabilityAnalysisTemplate,
    ConformationalAnalysisTemplate
)


class TemplateManager:
    """
    Central manager for all simulation templates in ProteinMD.
    
    Handles built-in templates, user templates, template discovery,
    validation, and template library management.
    """
    
    def __init__(self, user_templates_dir: Optional[Union[str, Path]] = None):
        """
        Initialize template manager.
        
        Args:
            user_templates_dir: Directory for user templates (default: ~/.proteinmd/templates)
        """
        if user_templates_dir is None:
            self.user_templates_dir = Path.home() / '.proteinmd' / 'templates'
        else:
            self.user_templates_dir = Path(user_templates_dir)
            
        # Ensure user templates directory exists
        self.user_templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize built-in templates
        self.builtin_templates: Dict[str, BaseTemplate] = {
            'protein_folding': ProteinFoldingTemplate(),
            'equilibration': EquilibrationTemplate(),
            'free_energy': FreeEnergyTemplate(),
            'membrane_protein': MembraneProteinTemplate(),
            'ligand_binding': LigandBindingTemplate(),
            'enhanced_sampling': EnhancedSamplingTemplate(),
            'drug_discovery': DrugDiscoveryTemplate(),
            'stability_analysis': StabilityAnalysisTemplate(),
            'conformational_analysis': ConformationalAnalysisTemplate()
        }
        
        # Cache for user templates
        self._user_templates_cache: Optional[Dict[str, BaseTemplate]] = None
        
    def get_template(self, name: str) -> BaseTemplate:
        """
        Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template instance
            
        Raises:
            KeyError: If template not found
        """
        # Check built-in templates first
        if name in self.builtin_templates:
            return self.builtin_templates[name]
            
        # Check user templates
        user_templates = self.get_user_templates()
        if name in user_templates:
            return user_templates[name]
            
        raise KeyError(f"Template '{name}' not found")
        
    def list_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available templates with their information.
        
        Returns:
            Dictionary mapping template names to their summary information
        """
        templates = {}
        
        # Built-in templates
        for name, template in self.builtin_templates.items():
            summary = template.get_summary()
            summary['source'] = 'builtin'
            templates[name] = summary
            
        # User templates
        for name, template in self.get_user_templates().items():
            summary = template.get_summary()
            summary['source'] = 'user'
            templates[name] = summary
            
        return templates
        
    def get_user_templates(self) -> Dict[str, BaseTemplate]:
        """
        Load and cache user templates from the user templates directory.
        
        Returns:
            Dictionary of user template names to template instances
        """
        if self._user_templates_cache is None:
            self._user_templates_cache = {}
            
            # Scan user templates directory
            for template_file in self.user_templates_dir.glob('*.json'):
                try:
                    template = BaseTemplate.load_template(template_file)
                    self._user_templates_cache[template.name] = template
                except Exception as e:
                    print(f"Warning: Could not load user template {template_file}: {e}")
                    
            for template_file in self.user_templates_dir.glob('*.yaml'):
                try:
                    template = BaseTemplate.load_template(template_file)
                    self._user_templates_cache[template.name] = template
                except Exception as e:
                    print(f"Warning: Could not load user template {template_file}: {e}")
                    
            for template_file in self.user_templates_dir.glob('*.yml'):
                try:
                    template = BaseTemplate.load_template(template_file)
                    self._user_templates_cache[template.name] = template
                except Exception as e:
                    print(f"Warning: Could not load user template {template_file}: {e}")
                    
        return self._user_templates_cache
        
    def save_user_template(self, template: BaseTemplate, overwrite: bool = False):
        """
        Save a template to the user templates directory.
        
        Args:
            template: Template to save
            overwrite: Whether to overwrite existing templates
            
        Raises:
            FileExistsError: If template exists and overwrite=False
        """
        template_file = self.user_templates_dir / f"{template.name}.json"
        
        if template_file.exists() and not overwrite:
            raise FileExistsError(f"User template '{template.name}' already exists")
            
        # Set creation date if not set
        if not template.created_date:
            template.created_date = datetime.now().isoformat()
            
        template.save_template(template_file, format='json')
        
        # Clear cache to force reload
        self._user_templates_cache = None
        
    def delete_user_template(self, name: str):
        """
        Delete a user template.
        
        Args:
            name: Template name to delete
            
        Raises:
            KeyError: If template not found
            ValueError: If trying to delete built-in template
        """
        if name in self.builtin_templates:
            raise ValueError(f"Cannot delete built-in template '{name}'")
            
        # Look for template file
        template_files = [
            self.user_templates_dir / f"{name}.json",
            self.user_templates_dir / f"{name}.yaml", 
            self.user_templates_dir / f"{name}.yml"
        ]
        
        deleted = False
        for template_file in template_files:
            if template_file.exists():
                template_file.unlink()
                deleted = True
                
        if not deleted:
            raise KeyError(f"User template '{name}' not found")
            
        # Clear cache
        self._user_templates_cache = None
        
    def create_template_from_config(self, name: str, description: str, 
                                  config: Dict[str, Any], **metadata) -> BaseTemplate:
        """
        Create a new template from a configuration dictionary.
        
        Args:
            name: Template name
            description: Template description
            config: Base configuration
            **metadata: Additional metadata (author, tags, etc.)
            
        Returns:
            New template instance
        """
        template = GenericTemplate(name, description)
        template.base_config = config
        
        # Apply metadata
        template.author = metadata.get('author', 'User')
        template.tags = metadata.get('tags', [])
        template.dependencies = metadata.get('dependencies', [])
        template.created_date = datetime.now().isoformat()
        
        return template
        
    def export_template(self, name: str, filepath: Union[str, Path], 
                       format: str = 'json'):
        """
        Export a template to a file.
        
        Args:
            name: Template name
            filepath: Output file path
            format: Export format ('json' or 'yaml')
        """
        template = self.get_template(name)
        template.save_template(filepath, format)
        
    def import_template(self, filepath: Union[str, Path], 
                       overwrite: bool = False) -> BaseTemplate:
        """
        Import a template from a file.
        
        Args:
            filepath: Template file path
            overwrite: Whether to overwrite existing templates
            
        Returns:
            Imported template instance
        """
        template = BaseTemplate.load_template(filepath)
        self.save_user_template(template, overwrite=overwrite)
        return template
        
    def search_templates(self, query: str = None, tags: List[str] = None,
                        source: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Search templates by query string, tags, or source.
        
        Args:
            query: Search query (matches name and description)
            tags: Required tags
            source: Template source ('builtin' or 'user')
            
        Returns:
            Dictionary of matching templates
        """
        all_templates = self.list_templates()
        results = {}
        
        for name, info in all_templates.items():
            # Filter by source
            if source and info['source'] != source:
                continue
                
            # Filter by tags
            if tags:
                template_tags = set(info.get('tags', []))
                if not set(tags).issubset(template_tags):
                    continue
                    
            # Filter by query
            if query:
                query_lower = query.lower()
                if (query_lower not in name.lower() and 
                    query_lower not in info['description'].lower()):
                    continue
                    
            results[name] = info
            
        return results
        
    def validate_template(self, name: str, parameters: Dict[str, Any] = None) -> bool:
        """
        Validate a template and optionally its parameters.
        
        Args:
            name: Template name
            parameters: Parameter values to validate
            
        Returns:
            True if validation passes
            
        Raises:
            TemplateValidationError: If validation fails
        """
        template = self.get_template(name)
        
        if parameters:
            template.validate_parameters(parameters)
            
        return True
        
    def get_template_dependencies(self, name: str) -> List[str]:
        """
        Get dependencies for a template.
        
        Args:
            name: Template name
            
        Returns:
            List of dependency names
        """
        template = self.get_template(name)
        return template.dependencies
        
    def backup_user_templates(self, backup_path: Union[str, Path]):
        """
        Create a backup of all user templates.
        
        Args:
            backup_path: Backup directory path
        """
        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all template files
        for template_file in self.user_templates_dir.glob('*'):
            if template_file.is_file():
                shutil.copy2(template_file, backup_path / template_file.name)
                
    def restore_user_templates(self, backup_path: Union[str, Path], 
                              overwrite: bool = False):
        """
        Restore user templates from a backup.
        
        Args:
            backup_path: Backup directory path
            overwrite: Whether to overwrite existing templates
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup directory not found: {backup_path}")
            
        for template_file in backup_path.glob('*'):
            if template_file.is_file() and template_file.suffix in ['.json', '.yaml', '.yml']:
                target_file = self.user_templates_dir / template_file.name
                
                if target_file.exists() and not overwrite:
                    print(f"Skipping existing template: {template_file.name}")
                    continue
                    
                shutil.copy2(template_file, target_file)
                
        # Clear cache
        self._user_templates_cache = None
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the template library.
        
        Returns:
            Dictionary with template statistics
        """
        all_templates = self.list_templates()
        
        stats = {
            'total_templates': len(all_templates),
            'builtin_templates': len(self.builtin_templates),
            'user_templates': len(self.get_user_templates()),
            'templates_by_tag': {},
            'templates_by_author': {}
        }
        
        # Count by tags and authors
        for name, info in all_templates.items():
            for tag in info.get('tags', []):
                stats['templates_by_tag'][tag] = stats['templates_by_tag'].get(tag, 0) + 1
                
            author = info.get('author', 'Unknown')
            stats['templates_by_author'][author] = stats['templates_by_author'].get(author, 0) + 1
            
        return stats
