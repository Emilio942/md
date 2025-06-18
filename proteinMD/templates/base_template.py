"""
Base Template System for ProteinMD Simulation Templates

This module provides the foundation for creating and managing simulation templates
in ProteinMD. Templates define reusable configurations for common MD workflows.
"""

import json
import yaml
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod


class TemplateValidationError(Exception):
    """Exception raised when template validation fails."""
    pass


@dataclass
class TemplateParameter:
    """
    Defines a configurable parameter in a simulation template.
    
    Attributes:
        name: Parameter name
        description: Human-readable description
        default_value: Default value for the parameter
        parameter_type: Data type (int, float, str, bool, list)
        allowed_values: List of allowed values (optional)
        min_value: Minimum value for numeric parameters
        max_value: Maximum value for numeric parameters
        required: Whether parameter is required
        units: Physical units for the parameter
    """
    name: str
    description: str
    default_value: Any
    parameter_type: str
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    required: bool = False
    units: Optional[str] = None
    
    def validate(self, value: Any) -> bool:
        """Validate a parameter value against constraints."""
        if self.required and value is None:
            raise TemplateValidationError(f"Required parameter '{self.name}' is missing")
            
        if value is None:
            return True
            
        # Type validation
        if self.parameter_type == 'int' and not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise TemplateValidationError(f"Parameter '{self.name}' must be an integer")
                
        elif self.parameter_type == 'float' and not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise TemplateValidationError(f"Parameter '{self.name}' must be a number")
                
        elif self.parameter_type == 'str' and not isinstance(value, str):
            raise TemplateValidationError(f"Parameter '{self.name}' must be a string")
            
        elif self.parameter_type == 'bool' and not isinstance(value, bool):
            raise TemplateValidationError(f"Parameter '{self.name}' must be a boolean")
            
        elif self.parameter_type == 'list' and not isinstance(value, list):
            raise TemplateValidationError(f"Parameter '{self.name}' must be a list")
        
        # Range validation
        if self.parameter_type in ['int', 'float']:
            if self.min_value is not None and value < self.min_value:
                raise TemplateValidationError(
                    f"Parameter '{self.name}' must be >= {self.min_value} {self.units or ''}"
                )
            if self.max_value is not None and value > self.max_value:
                raise TemplateValidationError(
                    f"Parameter '{self.name}' must be <= {self.max_value} {self.units or ''}"
                )
        
        # Allowed values validation
        if self.allowed_values and value not in self.allowed_values:
            raise TemplateValidationError(
                f"Parameter '{self.name}' must be one of: {self.allowed_values}"
            )
            
        return True


class BaseTemplate(ABC):
    """
    Abstract base class for all simulation templates.
    
    Each template represents a specific type of MD simulation workflow
    with predefined parameters and analysis configurations.
    """
    
    def __init__(self, name: str, description: str, version: str = "1.0.0"):
        """
        Initialize a simulation template.
        
        Args:
            name: Template name (unique identifier)
            description: Human-readable description
            version: Template version
        """
        self.name = name
        self.description = description
        self.version = version
        self.parameters: Dict[str, TemplateParameter] = {}
        self.tags: List[str] = []
        self.author: str = "ProteinMD"
        self.created_date: str = ""
        self.dependencies: List[str] = []
        
        # Initialize template-specific parameters
        self._setup_parameters()
        
    @abstractmethod
    def _setup_parameters(self):
        """Setup template-specific parameters. Must be implemented by subclasses."""
        pass
        
    @abstractmethod
    def generate_config(self, **kwargs) -> Dict[str, Any]:
        """
        Generate simulation configuration from template.
        
        Args:
            **kwargs: Parameter values to override defaults
            
        Returns:
            Complete simulation configuration dictionary
        """
        pass
        
    def add_parameter(self, param: TemplateParameter):
        """Add a configurable parameter to the template."""
        self.parameters[param.name] = param
        
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate all parameters against their constraints.
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            True if all parameters are valid
            
        Raises:
            TemplateValidationError: If validation fails
        """
        for name, param in self.parameters.items():
            value = params.get(name, param.default_value)
            param.validate(value)
            
        return True
        
    def get_parameter_info(self, param_name: str) -> Optional[TemplateParameter]:
        """Get information about a specific parameter."""
        return self.parameters.get(param_name)
        
    def list_parameters(self) -> List[TemplateParameter]:
        """Get list of all template parameters."""
        return list(self.parameters.values())
        
    def get_default_config(self) -> Dict[str, Any]:
        """Get configuration with all default parameter values."""
        default_params = {name: param.default_value 
                         for name, param in self.parameters.items()}
        return self.generate_config(**default_params)
        
    def save_template(self, filepath: Union[str, Path], format: str = 'json'):
        """
        Save template definition to file.
        
        Args:
            filepath: Output file path
            format: File format ('json' or 'yaml')
        """
        template_data = {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'author': self.author,
            'created_date': self.created_date,
            'tags': self.tags,
            'dependencies': self.dependencies,
            'parameters': {
                name: {
                    'description': param.description,
                    'default_value': param.default_value,
                    'type': param.parameter_type,
                    'allowed_values': param.allowed_values,
                    'min_value': param.min_value,
                    'max_value': param.max_value,
                    'required': param.required,
                    'units': param.units
                }
                for name, param in self.parameters.items()
            },
            'config': self.get_default_config()
        }
        
        filepath = Path(filepath)
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(template_data, f, indent=2)
        elif format.lower() in ['yaml', 'yml']:
            with open(filepath, 'w') as f:
                yaml.dump(template_data, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    @classmethod
    def load_template(cls, filepath: Union[str, Path]) -> 'BaseTemplate':
        """
        Load template from file.
        
        Args:
            filepath: Template file path
            
        Returns:
            Template instance
        """
        filepath = Path(filepath)
        
        if filepath.suffix.lower() == '.json':
            with open(filepath) as f:
                data = json.load(f)
        elif filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
        # Create generic template instance
        template = GenericTemplate(
            name=data['name'],
            description=data['description'],
            version=data.get('version', '1.0.0')
        )
        
        # Load parameters
        for name, param_data in data.get('parameters', {}).items():
            param = TemplateParameter(
                name=name,
                description=param_data['description'],
                default_value=param_data['default_value'],
                parameter_type=param_data['type'],
                allowed_values=param_data.get('allowed_values'),
                min_value=param_data.get('min_value'),
                max_value=param_data.get('max_value'),
                required=param_data.get('required', False),
                units=param_data.get('units')
            )
            template.add_parameter(param)
            
        # Store the base configuration
        template.base_config = data.get('config', {})
        template.author = data.get('author', 'Unknown')
        template.created_date = data.get('created_date', '')
        template.tags = data.get('tags', [])
        template.dependencies = data.get('dependencies', [])
        
        return template
        
    def add_tag(self, tag: str):
        """Add a tag for template categorization."""
        if tag not in self.tags:
            self.tags.append(tag)
            
    def has_tag(self, tag: str) -> bool:
        """Check if template has a specific tag."""
        return tag in self.tags
        
    def get_summary(self) -> Dict[str, Any]:
        """Get template summary information."""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'author': self.author,
            'tags': self.tags,
            'parameter_count': len(self.parameters),
            'dependencies': self.dependencies
        }


class GenericTemplate(BaseTemplate):
    """
    Generic template class for user-defined templates.
    
    This class allows loading arbitrary templates from files
    without requiring a specific template subclass.
    """
    
    def __init__(self, name: str, description: str, version: str = "1.0.0"):
        self.base_config: Dict[str, Any] = {}
        super().__init__(name, description, version)
        
    def _setup_parameters(self):
        """Generic templates don't have predefined parameters."""
        pass
        
    def generate_config(self, **kwargs) -> Dict[str, Any]:
        """
        Generate configuration by merging base config with provided parameters.
        
        Args:
            **kwargs: Parameter overrides
            
        Returns:
            Merged configuration
        """
        # Start with base configuration
        config = self.base_config.copy()
        
        # Apply parameter overrides
        for name, value in kwargs.items():
            if name in self.parameters:
                self.parameters[name].validate(value)
                # Apply value to configuration (simplified approach)
                # In a real implementation, this would map parameters to config paths
                config.setdefault('parameters', {})[name] = value
                
        return config
