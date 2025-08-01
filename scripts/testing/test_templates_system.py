"""
Comprehensive Test Suite for ProteinMD Templates System

This test suite validates all aspects of the template system including:
- Template creation and management
- Parameter validation
- Configuration generation
- Import/export functionality
- CLI operations
"""

import pytest
import json
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import template system components
try:
    from proteinMD.templates import (
        TemplateManager,
        BaseTemplate,
        TemplateParameter,
        TemplateValidationError,
        ProteinFoldingTemplate,
        EquilibrationTemplate,
        FreeEnergyTemplate
    )
    from proteinMD.templates.template_cli import TemplateCLI
    TEMPLATES_AVAILABLE = True
except ImportError:
    TEMPLATES_AVAILABLE = False
    pytest.skip("Template system not available", allow_module_level=True)


class TestTemplateParameter:
    """Test TemplateParameter validation and functionality."""
    
    def test_parameter_creation(self):
        """Test basic parameter creation."""
        param = TemplateParameter(
            name="test_param",
            description="Test parameter",
            default_value=100.0,
            parameter_type="float",
            min_value=10.0,
            max_value=1000.0,
            units="nm"
        )
        
        assert param.name == "test_param"
        assert param.description == "Test parameter"
        assert param.default_value == 100.0
        assert param.parameter_type == "float"
        assert param.min_value == 10.0
        assert param.max_value == 1000.0
        assert param.units == "nm"
        assert not param.required
        
    def test_parameter_validation_success(self):
        """Test successful parameter validation."""
        param = TemplateParameter(
            name="temp",
            description="Temperature",
            default_value=300.0,
            parameter_type="float",
            min_value=250.0,
            max_value=400.0,
            units="K"
        )
        
        # Valid values
        assert param.validate(300.0) is True
        assert param.validate(250.0) is True
        assert param.validate(400.0) is True
        assert param.validate(325.5) is True
        
    def test_parameter_validation_range_errors(self):
        """Test parameter validation range errors."""
        param = TemplateParameter(
            name="temp",
            description="Temperature",
            default_value=300.0,
            parameter_type="float",
            min_value=250.0,
            max_value=400.0,
            units="K"
        )
        
        # Out of range values
        with pytest.raises(TemplateValidationError, match="must be >= 250.0"):
            param.validate(200.0)
            
        with pytest.raises(TemplateValidationError, match="must be <= 400.0"):
            param.validate(500.0)
            
    def test_parameter_validation_type_errors(self):
        """Test parameter validation type errors."""
        param = TemplateParameter(
            name="steps",
            description="Number of steps",
            default_value=10000,
            parameter_type="int"
        )
        
        # Valid integer
        assert param.validate(5000) is True
        
        # Invalid type
        with pytest.raises(TemplateValidationError, match="must be an integer"):
            param.validate("not_an_integer")
            
    def test_parameter_validation_allowed_values(self):
        """Test parameter validation with allowed values."""
        param = TemplateParameter(
            name="solvent",
            description="Solvent type",
            default_value="explicit",
            parameter_type="str",
            allowed_values=["explicit", "implicit", "vacuum"]
        )
        
        # Valid values
        assert param.validate("explicit") is True
        assert param.validate("implicit") is True
        assert param.validate("vacuum") is True
        
        # Invalid value
        with pytest.raises(TemplateValidationError, match="must be one of"):
            param.validate("invalid_solvent")
            
    def test_parameter_validation_required(self):
        """Test required parameter validation."""
        param = TemplateParameter(
            name="input_file",
            description="Input file",
            default_value=None,
            parameter_type="str",
            required=True
        )
        
        # Valid value
        assert param.validate("protein.pdb") is True
        
        # Missing required parameter
        with pytest.raises(TemplateValidationError, match="Required parameter"):
            param.validate(None)


class TestBaseTemplate:
    """Test BaseTemplate functionality."""
    
    def test_template_creation(self):
        """Test basic template creation."""
        class TestTemplate(BaseTemplate):
            def _setup_parameters(self):
                self.add_parameter(TemplateParameter(
                    name="test_param",
                    description="Test parameter",
                    default_value=100.0,
                    parameter_type="float"
                ))
                
            def generate_config(self, **kwargs):
                return {"test": kwargs.get("test_param", 100.0)}
                
        template = TestTemplate("test", "Test template")
        
        assert template.name == "test"
        assert template.description == "Test template"
        assert len(template.parameters) == 1
        assert "test_param" in template.parameters
        
    def test_parameter_management(self):
        """Test parameter addition and retrieval."""
        class TestTemplate(BaseTemplate):
            def _setup_parameters(self):
                pass
                
            def generate_config(self, **kwargs):
                return {}
                
        template = TestTemplate("test", "Test template")
        
        # Add parameter
        param = TemplateParameter(
            name="new_param",
            description="New parameter",
            default_value="default",
            parameter_type="str"
        )
        template.add_parameter(param)
        
        # Check parameter retrieval
        assert template.get_parameter_info("new_param") == param
        assert template.get_parameter_info("nonexistent") is None
        assert len(template.list_parameters()) == 1
        
    def test_parameter_validation(self):
        """Test template parameter validation."""
        class TestTemplate(BaseTemplate):
            def _setup_parameters(self):
                self.add_parameter(TemplateParameter(
                    name="temp",
                    description="Temperature",
                    default_value=300.0,
                    parameter_type="float",
                    min_value=250.0,
                    max_value=400.0
                ))
                
            def generate_config(self, **kwargs):
                return {}
                
        template = TestTemplate("test", "Test template")
        
        # Valid parameters
        assert template.validate_parameters({"temp": 300.0}) is True
        assert template.validate_parameters({}) is True  # Uses default
        
        # Invalid parameters
        with pytest.raises(TemplateValidationError):
            template.validate_parameters({"temp": 500.0})
            
    def test_template_tags(self):
        """Test template tagging functionality."""
        class TestTemplate(BaseTemplate):
            def _setup_parameters(self):
                pass
                
            def generate_config(self, **kwargs):
                return {}
                
        template = TestTemplate("test", "Test template")
        
        # Add tags
        template.add_tag("protein")
        template.add_tag("folding")
        template.add_tag("protein")  # Duplicate should be ignored
        
        assert len(template.tags) == 2
        assert template.has_tag("protein")
        assert template.has_tag("folding")
        assert not template.has_tag("membrane")
        
    def test_template_summary(self):
        """Test template summary generation."""
        class TestTemplate(BaseTemplate):
            def _setup_parameters(self):
                self.add_parameter(TemplateParameter(
                    name="param1",
                    description="Parameter 1",
                    default_value=1,
                    parameter_type="int"
                ))
                
            def generate_config(self, **kwargs):
                return {}
                
        template = TestTemplate("test", "Test template", "2.0.0")
        template.add_tag("test")
        template.author = "Test Author"
        
        summary = template.get_summary()
        
        assert summary["name"] == "test"
        assert summary["description"] == "Test template"
        assert summary["version"] == "2.0.0"
        assert summary["author"] == "Test Author"
        assert summary["tags"] == ["test"]
        assert summary["parameter_count"] == 1


class TestBuiltinTemplates:
    """Test built-in template functionality."""
    
    def test_protein_folding_template(self):
        """Test protein folding template."""
        template = ProteinFoldingTemplate()
        
        assert template.name == "protein_folding"
        assert "folding" in template.tags
        assert "protein" in template.tags
        assert len(template.parameters) > 0
        
        # Test parameter presence
        assert "simulation_time" in template.parameters
        assert "temperature" in template.parameters
        assert "timestep" in template.parameters
        
        # Test configuration generation
        config = template.generate_config()
        assert "simulation" in config
        assert "forcefield" in config
        assert "environment" in config
        assert "analysis" in config
        
        # Test with custom parameters
        config = template.generate_config(
            simulation_time=200.0,
            temperature=310.0
        )
        assert config["simulation"]["temperature"] == 310.0
        
    def test_equilibration_template(self):
        """Test equilibration template."""
        template = EquilibrationTemplate()
        
        assert template.name == "equilibration"
        assert "equilibration" in template.tags
        
        # Test configuration generation
        config = template.generate_config()
        assert config["simulation"]["timestep"] == 0.001  # Shorter timestep
        assert "restraints" in config
        assert "minimization" in config
        
    def test_free_energy_template(self):
        """Test free energy template."""
        template = FreeEnergyTemplate()
        
        assert template.name == "free_energy"
        assert "free_energy" in template.tags
        assert "umbrella_sampling" in template.tags
        
        # Test configuration generation
        config = template.generate_config(window_count=15)
        assert "sampling" in config
        assert config["sampling"]["method"] == "umbrella_sampling"
        assert config["sampling"]["windows"]["count"] == 15


class TestTemplateManager:
    """Test TemplateManager functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TemplateManager(user_templates_dir=self.temp_dir)
        
    def test_manager_initialization(self):
        """Test template manager initialization."""
        assert len(self.manager.builtin_templates) > 0
        assert "protein_folding" in self.manager.builtin_templates
        assert "equilibration" in self.manager.builtin_templates
        
    def test_get_builtin_template(self):
        """Test getting built-in templates."""
        template = self.manager.get_template("protein_folding")
        assert isinstance(template, ProteinFoldingTemplate)
        assert template.name == "protein_folding"
        
    def test_get_nonexistent_template(self):
        """Test getting nonexistent template."""
        with pytest.raises(KeyError):
            self.manager.get_template("nonexistent_template")
            
    def test_list_templates(self):
        """Test listing templates."""
        templates = self.manager.list_templates()
        assert len(templates) >= len(self.manager.builtin_templates)
        
        # Check that all built-in templates are listed
        for name in self.manager.builtin_templates:
            assert name in templates
            assert templates[name]["source"] == "builtin"
            
    def test_create_user_template(self):
        """Test creating user template."""
        config = {
            "simulation": {"timestep": 0.002},
            "forcefield": {"type": "amber_ff14sb"}
        }
        
        template = self.manager.create_template_from_config(
            name="test_user_template",
            description="Test user template",
            config=config,
            author="Test Author",
            tags=["test", "custom"]
        )
        
        assert template.name == "test_user_template"
        assert template.description == "Test user template"
        assert template.author == "Test Author"
        assert "test" in template.tags
        assert "custom" in template.tags
        
    def test_save_and_load_user_template(self):
        """Test saving and loading user templates."""
        # Create template
        config = {"simulation": {"timestep": 0.002}}
        template = self.manager.create_template_from_config(
            name="test_save_load",
            description="Test save/load",
            config=config
        )
        
        # Save template
        self.manager.save_user_template(template)
        
        # Clear cache and reload
        self.manager._user_templates_cache = None
        loaded_template = self.manager.get_template("test_save_load")
        
        assert loaded_template.name == "test_save_load"
        assert loaded_template.description == "Test save/load"
        
    def test_delete_user_template(self):
        """Test deleting user templates."""
        # Create and save template
        config = {"simulation": {"timestep": 0.002}}
        template = self.manager.create_template_from_config(
            name="test_delete",
            description="Test delete",
            config=config
        )
        self.manager.save_user_template(template)
        
        # Verify it exists
        assert "test_delete" in self.manager.list_templates()
        
        # Delete template
        self.manager.delete_user_template("test_delete")
        
        # Verify it's gone
        with pytest.raises(KeyError):
            self.manager.get_template("test_delete")
            
    def test_search_templates(self):
        """Test template searching."""
        # Search by query
        results = self.manager.search_templates(query="folding")
        assert "protein_folding" in results
        
        # Search by source
        results = self.manager.search_templates(source="builtin")
        assert all(info["source"] == "builtin" for info in results.values())
        
        # Search by tags
        results = self.manager.search_templates(tags=["protein"])
        assert len(results) > 0
        
    def test_template_validation(self):
        """Test template validation."""
        # Valid template
        assert self.manager.validate_template("protein_folding") is True
        
        # Valid template with parameters
        params = {"simulation_time": 150.0, "temperature": 310.0}
        assert self.manager.validate_template("protein_folding", params) is True
        
        # Invalid parameters
        params = {"temperature": 600.0}  # Too high
        with pytest.raises(TemplateValidationError):
            self.manager.validate_template("protein_folding", params)
            
    def test_export_import_template(self):
        """Test template export and import."""
        # Export built-in template
        export_file = Path(self.temp_dir) / "exported_template.json"
        self.manager.export_template("protein_folding", export_file)
        
        assert export_file.exists()
        
        # Import template with new name
        imported = self.manager.import_template(export_file)
        assert imported.name == "protein_folding"
        
    def test_get_statistics(self):
        """Test template statistics."""
        stats = self.manager.get_statistics()
        
        assert "total_templates" in stats
        assert "builtin_templates" in stats
        assert "user_templates" in stats
        assert "templates_by_tag" in stats
        assert "templates_by_author" in stats
        
        assert stats["builtin_templates"] > 0
        assert stats["total_templates"] >= stats["builtin_templates"]


class TestTemplateCLI:
    """Test Template CLI functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cli = TemplateCLI()
        self.cli.manager = TemplateManager(user_templates_dir=self.temp_dir)
        
    def test_list_templates_command(self):
        """Test list templates CLI command."""
        args = MagicMock()
        args.query = None
        args.tags = None
        args.source = None
        args.show_stats = False
        
        with patch('builtins.print') as mock_print:
            result = self.cli.list_templates(args)
            
        assert result == 0
        mock_print.assert_called()
        
    def test_show_template_command(self):
        """Test show template CLI command."""
        args = MagicMock()
        args.name = "protein_folding"
        args.show_config = False
        args.format = "json"
        
        with patch('builtins.print') as mock_print:
            result = self.cli.show_template(args)
            
        assert result == 0
        mock_print.assert_called()
        
    def test_create_template_command(self):
        """Test create template CLI command."""
        # Create config file
        config_file = Path(self.temp_dir) / "test_config.json"
        config = {"simulation": {"timestep": 0.002}}
        with open(config_file, 'w') as f:
            json.dump(config, f)
            
        args = MagicMock()
        args.name = "test_cli_template"
        args.description = "Test CLI template"
        args.config_file = str(config_file)
        args.author = "Test Author"
        args.tags = ["test"]
        args.overwrite = False
        
        with patch('builtins.print') as mock_print:
            result = self.cli.create_template(args)
            
        assert result == 0
        mock_print.assert_called()
        
        # Verify template was created
        template = self.cli.manager.get_template("test_cli_template")
        assert template.name == "test_cli_template"
        
    def test_validate_template_command(self):
        """Test validate template CLI command."""
        args = MagicMock()
        args.name = "protein_folding"
        args.parameters_file = None
        
        with patch('builtins.print') as mock_print:
            result = self.cli.validate_template(args)
            
        assert result == 0
        mock_print.assert_called()
        
    def test_generate_config_command(self):
        """Test generate config CLI command."""
        args = MagicMock()
        args.name = "protein_folding"
        args.parameters_file = None
        args.output_file = None
        args.format = "json"
        
        with patch('builtins.print') as mock_print:
            result = self.cli.generate_config(args)
            
        assert result == 0
        mock_print.assert_called()


class TestTemplateFileOperations:
    """Test template file I/O operations."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def test_save_load_json_template(self):
        """Test saving and loading JSON templates."""
        template = ProteinFoldingTemplate()
        template_file = Path(self.temp_dir) / "test_template.json"
        
        # Save template
        template.save_template(template_file, format='json')
        assert template_file.exists()
        
        # Load template
        loaded_template = BaseTemplate.load_template(template_file)
        assert loaded_template.name == template.name
        assert loaded_template.description == template.description
        
    def test_save_load_yaml_template(self):
        """Test saving and loading YAML templates."""
        template = EquilibrationTemplate()
        template_file = Path(self.temp_dir) / "test_template.yaml"
        
        # Save template
        template.save_template(template_file, format='yaml')
        assert template_file.exists()
        
        # Load template
        loaded_template = BaseTemplate.load_template(template_file)
        assert loaded_template.name == template.name
        assert loaded_template.description == template.description
        
    def test_invalid_file_format(self):
        """Test handling of invalid file formats."""
        template = ProteinFoldingTemplate()
        template_file = Path(self.temp_dir) / "test_template.txt"
        
        # Save with invalid format
        with pytest.raises(ValueError, match="Unsupported format"):
            template.save_template(template_file, format='txt')
            
        # Create invalid file
        with open(template_file, 'w') as f:
            f.write("invalid content")
            
        # Load invalid format
        with pytest.raises(ValueError, match="Unsupported file format"):
            BaseTemplate.load_template(template_file)


class TestTemplateIntegration:
    """Test template system integration with main ProteinMD."""
    
    def test_template_config_compatibility(self):
        """Test that template configurations are compatible with MD simulation."""
        # This test would require the actual simulation modules
        # For now, we test config structure
        
        template = ProteinFoldingTemplate()
        config = template.generate_config()
        
        # Check required configuration sections
        required_sections = ['simulation', 'forcefield', 'environment', 'analysis']
        for section in required_sections:
            assert section in config, f"Missing required section: {section}"
            
        # Check simulation parameters
        sim_config = config['simulation']
        assert 'timestep' in sim_config
        assert 'temperature' in sim_config
        assert 'n_steps' in sim_config
        
        # Check forcefield parameters
        ff_config = config['forcefield']
        assert 'type' in ff_config
        
    def test_template_parameter_ranges(self):
        """Test that template parameters have reasonable ranges."""
        template = ProteinFoldingTemplate()
        
        # Temperature parameter
        temp_param = template.get_parameter_info('temperature')
        assert temp_param.min_value >= 250.0  # Reasonable minimum
        assert temp_param.max_value <= 500.0  # Reasonable maximum
        
        # Timestep parameter
        timestep_param = template.get_parameter_info('timestep')
        assert timestep_param.min_value > 0.0
        assert timestep_param.max_value <= 0.01  # Reasonable maximum
        
    def test_all_builtin_templates_functional(self):
        """Test that all built-in templates can generate configurations."""
        manager = TemplateManager()
        
        for template_name in manager.builtin_templates:
            template = manager.get_template(template_name)
            
            # Generate default configuration
            config = template.get_default_config()
            assert isinstance(config, dict)
            assert len(config) > 0
            
            # Validate parameters
            template.validate_parameters({})  # Should work with defaults


def test_template_system_availability():
    """Test that the template system is properly available."""
    assert TEMPLATES_AVAILABLE, "Template system should be available for testing"
    
    # Test imports
    from proteinMD.templates import TemplateManager, BaseTemplate
    from proteinMD.templates.builtin_templates import ProteinFoldingTemplate
    
    # Test basic functionality
    manager = TemplateManager()
    assert len(manager.builtin_templates) > 0
    
    template = ProteinFoldingTemplate()
    config = template.get_default_config()
    assert isinstance(config, dict)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
