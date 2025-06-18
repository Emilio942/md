"""
ProteinMD Logging Configuration Management

This module provides configuration management for the ProteinMD logging system,
including configuration validation, environment-based settings, and dynamic
configuration updates.

Author: ProteinMD Development Team
Date: 2024
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
import logging

from .exceptions import ConfigurationError


@dataclass
class LoggingConfig:
    """Configuration class for ProteinMD logging system."""
    
    # Basic logging settings
    name: str = "proteinmd"
    log_level: str = "INFO"
    console_output: bool = True
    use_json: bool = False
    
    # File logging settings
    log_file: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Advanced settings
    enable_performance_monitoring: bool = True
    enable_error_reporting: bool = True
    enable_graceful_degradation: bool = True
    
    # Error reporting settings
    max_error_history: int = 1000
    error_report_format: str = "json"  # json, text
    
    # Performance monitoring settings
    performance_log_level: str = "DEBUG"
    monitor_memory_usage: bool = True
    
    # Module-specific settings
    module_log_levels: Dict[str, str] = field(default_factory=dict)
    
    # Fallback operations for graceful degradation
    fallback_operations: Dict[str, str] = field(default_factory=dict)
    
    # Custom formatters
    custom_format: Optional[str] = None
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Environment-specific overrides
    environment: str = "development"  # development, testing, production
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self):
        """Validate configuration settings."""
        # Validate log level
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ConfigurationError(
                f"Invalid log level: {self.log_level}. Must be one of {valid_levels}",
                error_code="LOG_001"
            )
        
        # Validate performance log level
        if self.performance_log_level.upper() not in valid_levels:
            raise ConfigurationError(
                f"Invalid performance log level: {self.performance_log_level}. Must be one of {valid_levels}",
                "LOG_002"
            )
        
        # Validate file size
        if self.max_file_size <= 0:
            raise ConfigurationError(
                f"Invalid max_file_size: {self.max_file_size}. Must be positive",
                "LOG_003"
            )
        
        # Validate backup count
        if self.backup_count < 0:
            raise ConfigurationError(
                f"Invalid backup_count: {self.backup_count}. Must be non-negative",
                "LOG_004"
            )
        
        # Validate error report format
        valid_formats = {"json", "text"}
        if self.error_report_format not in valid_formats:
            raise ConfigurationError(
                f"Invalid error_report_format: {self.error_report_format}. Must be one of {valid_formats}",
                "LOG_005"
            )
        
        # Validate environment
        valid_environments = {"development", "testing", "production"}
        if self.environment not in valid_environments:
            raise ConfigurationError(
                f"Invalid environment: {self.environment}. Must be one of {valid_environments}",
                "LOG_006"
            )
        
        # Validate module log levels
        for module, level in self.module_log_levels.items():
            if level.upper() not in valid_levels:
                raise ConfigurationError(
                    f"Invalid log level for module {module}: {level}. Must be one of {valid_levels}",
                    "LOG_007"
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def save_to_file(self, file_path: str, format: str = "json"):
        """Save configuration to file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        elif format.lower() == "yaml":
            with open(path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            raise ConfigurationError(
                f"Unsupported configuration format: {format}. Use 'json' or 'yaml'",
                "LOG_008"
            )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LoggingConfig':
        """Create configuration from dictionary."""
        # Filter out unknown fields and simulation config sections
        known_fields = set(cls.__dataclass_fields__.keys())
        # Exclude simulation-specific config sections
        simulation_sections = {'simulation', 'forcefield', 'environment', 'analysis', 'visualization'}
        filtered_dict = {
            k: v for k, v in config_dict.items() 
            if k in known_fields and k not in simulation_sections
        }
        return cls(**filtered_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'LoggingConfig':
        """Create configuration from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_file(cls, file_path: str) -> 'LoggingConfig':
        """Load configuration from file."""
        path = Path(file_path)
        
        if not path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {file_path}",
                "LOG_009"
            )
        
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() == '.json':
                    config_dict = json.load(f)
                elif path.suffix.lower() in ['.yml', '.yaml']:
                    config_dict = yaml.safe_load(f)
                else:
                    # Try to auto-detect format
                    content = f.read()
                    try:
                        config_dict = json.loads(content)
                    except json.JSONDecodeError:
                        try:
                            config_dict = yaml.safe_load(content)
                        except yaml.YAMLError:
                            raise ConfigurationError(
                                f"Unable to parse configuration file: {file_path}. Use JSON or YAML format",
                                "LOG_010"
                            )
            
            return cls.from_dict(config_dict)
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Error loading configuration from {file_path}: {e}",
                "LOG_011"
            )
    
    def apply_environment_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            'PROTEINMD_LOG_LEVEL': 'log_level',
            'PROTEINMD_LOG_FILE': 'log_file',
            'PROTEINMD_LOG_JSON': 'use_json',
            'PROTEINMD_LOG_CONSOLE': 'console_output',
            'PROTEINMD_LOG_MAX_SIZE': 'max_file_size',
            'PROTEINMD_LOG_BACKUP_COUNT': 'backup_count',
            'PROTEINMD_ENVIRONMENT': 'environment',
            'PROTEINMD_PERFORMANCE_MONITORING': 'enable_performance_monitoring',
            'PROTEINMD_ERROR_REPORTING': 'enable_error_reporting',
            'PROTEINMD_GRACEFUL_DEGRADATION': 'enable_graceful_degradation'
        }
        
        for env_var, config_attr in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if config_attr in ['use_json', 'console_output', 'enable_performance_monitoring', 
                                  'enable_error_reporting', 'enable_graceful_degradation', 'monitor_memory_usage']:
                    setattr(self, config_attr, env_value.lower() in ['true', '1', 'yes', 'on'])
                elif config_attr in ['max_file_size', 'backup_count', 'max_error_history']:
                    try:
                        setattr(self, config_attr, int(env_value))
                    except ValueError:
                        raise ConfigurationError(
                            f"Invalid integer value for {env_var}: {env_value}",
                            "LOG_012"
                        )
                else:
                    setattr(self, config_attr, env_value)
        
        # Revalidate after applying overrides
        self.validate()
    
    def get_environment_defaults(self) -> Dict[str, Any]:
        """Get default settings for the current environment."""
        defaults = {
            'development': {
                'log_level': 'DEBUG',
                'console_output': True,
                'use_json': False,
                'enable_performance_monitoring': True,
                'max_error_history': 100
            },
            'testing': {
                'log_level': 'WARNING',
                'console_output': False,
                'use_json': True,
                'enable_performance_monitoring': False,
                'max_error_history': 50
            },
            'production': {
                'log_level': 'INFO',
                'console_output': True,
                'use_json': True,
                'enable_performance_monitoring': True,
                'max_error_history': 1000
            }
        }
        
        return defaults.get(self.environment, {})
    
    def apply_environment_defaults(self):
        """Apply environment-specific defaults."""
        defaults = self.get_environment_defaults()
        
        for key, value in defaults.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.validate()


class ConfigurationManager:
    """Manager for ProteinMD logging configuration."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._config: Optional[LoggingConfig] = None
        self._watchers: list = []
    
    def load_configuration(self, 
                          config_file: Optional[str] = None,
                          apply_env_overrides: bool = True,
                          apply_env_defaults: bool = True) -> LoggingConfig:
        """Load logging configuration."""
        file_path = config_file or self.config_file
        
        if file_path and Path(file_path).exists():
            self._config = LoggingConfig.from_file(file_path)
        else:
            # Use default configuration
            self._config = LoggingConfig()
        
        # Apply environment-specific defaults
        if apply_env_defaults:
            self._config.apply_environment_defaults()
        
        # Apply environment variable overrides
        if apply_env_overrides:
            self._config.apply_environment_overrides()
        
        return self._config
    
    def get_configuration(self) -> LoggingConfig:
        """Get current configuration."""
        if self._config is None:
            return self.load_configuration()
        return self._config
    
    def update_configuration(self, updates: Dict[str, Any]) -> LoggingConfig:
        """Update configuration with new values."""
        if self._config is None:
            self.load_configuration()
        
        # Create a new configuration with updates
        current_dict = self._config.to_dict()
        current_dict.update(updates)
        
        self._config = LoggingConfig.from_dict(current_dict)
        
        # Notify watchers
        self._notify_watchers()
        
        return self._config
    
    def save_configuration(self, file_path: Optional[str] = None, format: str = "json"):
        """Save current configuration to file."""
        if self._config is None:
            raise ConfigurationError("No configuration loaded", error_code="LOG_013")
        
        save_path = file_path or self.config_file
        if save_path is None:
            raise ConfigurationError("No file path specified for saving configuration", error_code="LOG_014")
        
        self._config.save_to_file(save_path, format)
    
    def register_watcher(self, callback: callable):
        """Register a callback to be called when configuration changes."""
        self._watchers.append(callback)
    
    def _notify_watchers(self):
        """Notify all registered watchers of configuration changes."""
        for callback in self._watchers:
            try:
                callback(self._config)
            except Exception as e:
                # Log the error but don't let it stop other watchers
                logging.getLogger(__name__).error(f"Error in configuration watcher: {e}")
    
    def create_default_config_file(self, file_path: str, environment: str = "development"):
        """Create a default configuration file."""
        config = LoggingConfig(environment=environment)
        config.apply_environment_defaults()
        config.save_to_file(file_path)
        return config


# Utility functions for easy configuration management
def load_logging_config(config_file: Optional[str] = None) -> LoggingConfig:
    """Load logging configuration from file or environment."""
    manager = ConfigurationManager(config_file)
    return manager.load_configuration()


def create_default_config(environment: str = "development") -> LoggingConfig:
    """Create a default logging configuration."""
    config = LoggingConfig(environment=environment)
    config.apply_environment_defaults()
    return config


def validate_config_file(file_path: str) -> tuple[bool, Optional[str]]:
    """Validate a configuration file."""
    try:
        LoggingConfig.from_file(file_path)
        return True, None
    except Exception as e:
        return False, str(e)


# Configuration templates
CONFIG_TEMPLATES = {
    "development": {
        "name": "proteinmd",
        "log_level": "DEBUG",
        "console_output": True,
        "use_json": False,
        "log_file": "logs/proteinmd_dev.log",
        "max_file_size": 5242880,  # 5MB
        "backup_count": 3,
        "enable_performance_monitoring": True,
        "enable_error_reporting": True,
        "enable_graceful_degradation": True,
        "environment": "development"
    },
    "production": {
        "name": "proteinmd",
        "log_level": "INFO",
        "console_output": True,
        "use_json": True,
        "log_file": "/var/log/proteinmd/proteinmd.log",
        "max_file_size": 10485760,  # 10MB
        "backup_count": 10,
        "enable_performance_monitoring": True,
        "enable_error_reporting": True,
        "enable_graceful_degradation": True,
        "environment": "production"
    },
    "testing": {
        "name": "proteinmd_test",
        "log_level": "WARNING",
        "console_output": False,
        "use_json": True,
        "log_file": "tests/logs/test.log",
        "max_file_size": 1048576,  # 1MB
        "backup_count": 2,
        "enable_performance_monitoring": False,
        "enable_error_reporting": True,
        "enable_graceful_degradation": False,
        "environment": "testing"
    }
}


def get_template_config(template_name: str) -> LoggingConfig:
    """Get a configuration template."""
    if template_name not in CONFIG_TEMPLATES:
        raise ConfigurationError(
            f"Unknown configuration template: {template_name}. Available templates: {list(CONFIG_TEMPLATES.keys())}",
            "LOG_015"
        )
    
    return LoggingConfig.from_dict(CONFIG_TEMPLATES[template_name])


# Example usage
if __name__ == "__main__":
    # Create development configuration
    dev_config = get_template_config("development")
    print("Development configuration:")
    print(dev_config.to_json())
    
    # Save to file
    dev_config.save_to_file("config/logging_dev.json")
    
    # Load from file and apply environment overrides
    manager = ConfigurationManager("config/logging_dev.json")
    config = manager.load_configuration()
    
    print("\nLoaded configuration:")
    print(config.to_json())
