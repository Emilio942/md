"""
ProteinMD Simulation Templates Package

This package provides a comprehensive template system for common MD simulation workflows.
Templates include predefined parameter sets, analysis configurations, and documentation
for various types of molecular dynamics simulations.

Key Features:
- Built-in templates for common workflows
- JSON/YAML format support
- Template validation and parameter checking
- User template management
- Template inheritance and composition
- Documentation and usage examples

Available Templates:
- Protein Folding: Complete folding simulation with comprehensive analysis
- MD Equilibration: System equilibration with temperature and pressure control
- Free Energy: Free energy perturbation calculations
- Membrane Proteins: Membrane-embedded protein simulations
- Ligand Binding: Protein-ligand binding studies
- Enhanced Sampling: Advanced sampling methods (REMD, Metadynamics)
- Drug Discovery: Virtual screening and lead optimization
- Stability Analysis: Protein stability assessment
- Conformational Analysis: Conformational space exploration

Usage:
    from proteinMD.templates import TemplateManager
    
    manager = TemplateManager()
    template = manager.get_template('protein_folding')
    config = template.generate_config(protein_file='protein.pdb')
"""

from .template_manager import TemplateManager
from .base_template import BaseTemplate, TemplateParameter, TemplateValidationError
from .template_cli import TemplateCLI
from .builtin_templates import (
    ProteinFoldingTemplate,
    EquilibrationTemplate,
    FreeEnergyTemplate,
    MembraneProteinTemplate,
    LigandBindingTemplate,
    EnhancedSamplingTemplate,
    DrugDiscoveryTemplate,
    StabilityAnalysisTemplate,
    ConformationalAnalysisTemplate,
    BUILTIN_TEMPLATES
)

__all__ = [
    'TemplateManager',
    'BaseTemplate',
    'TemplateParameter', 
    'TemplateValidationError',
    'TemplateCLI',
    'ProteinFoldingTemplate',
    'EquilibrationTemplate',
    'FreeEnergyTemplate',
    'MembraneProteinTemplate',
    'LigandBindingTemplate',
    'EnhancedSamplingTemplate',
    'DrugDiscoveryTemplate',
    'StabilityAnalysisTemplate',
    'ConformationalAnalysisTemplate',
    'BUILTIN_TEMPLATES'
]

# Package version
__version__ = '1.0.0'
