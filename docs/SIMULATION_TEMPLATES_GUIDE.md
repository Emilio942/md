# ProteinMD Simulation Templates Guide

## Overview

The ProteinMD template system provides a powerful and flexible way to create, manage, and use predefined simulation configurations for common molecular dynamics workflows. Templates encapsulate best practices, optimized parameters, and analysis configurations for specific types of studies.

## Table of Contents

1. [Template Basics](#template-basics)
2. [Built-in Templates](#built-in-templates)
3. [Using Templates](#using-templates)
4. [Creating Custom Templates](#creating-custom-templates)
5. [Template Management](#template-management)
6. [Advanced Features](#advanced-features)
7. [Examples](#examples)
8. [Best Practices](#best-practices)

## Template Basics

### What are Templates?

Templates are predefined simulation configurations that include:
- **Simulation parameters**: Timestep, temperature, pressure, simulation length
- **Force field settings**: Force field type, water models, cutoffs
- **Environment setup**: Solvent type, box dimensions, periodic boundaries
- **Analysis configurations**: Which analyses to run and their parameters
- **Visualization settings**: Real-time visualization, output formats

### Template Structure

Each template consists of:
```json
{
  "name": "template_name",
  "description": "Human-readable description",
  "version": "1.0.0",
  "author": "Author name",
  "tags": ["tag1", "tag2"],
  "parameters": {
    "param1": {
      "description": "Parameter description",
      "default_value": "default",
      "type": "parameter_type",
      "units": "physical_units"
    }
  },
  "config": {
    "simulation": {...},
    "forcefield": {...},
    "environment": {...},
    "analysis": {...}
  }
}
```

## Built-in Templates

ProteinMD includes 9 comprehensive built-in templates:

### 1. Protein Folding (`protein_folding`)
**Purpose**: Complete protein folding simulation with comprehensive analysis
**Best for**: Studying protein folding dynamics, conformational changes
**Key features**:
- Extended simulation times (100 ns default)
- Comprehensive analysis (RMSD, Rg, secondary structure, H-bonds)
- Explicit water solvation
- Temperature control

**Parameters**:
- `simulation_time`: Total simulation time (1-1000 ns)
- `temperature`: Simulation temperature (250-450 K)
- `pressure`: Simulation pressure (0.1-10 bar)
- `timestep`: Integration timestep (0.0005-0.004 ps)
- `save_frequency`: Trajectory save frequency (100-10000 steps)
- `analysis_stride`: Analysis frame stride (1-100)

### 2. Equilibration (`equilibration`)
**Purpose**: System equilibration with temperature and pressure control
**Best for**: Preparing systems for production runs
**Key features**:
- Shorter timesteps for stability
- Protein restraints during equilibration
- Energy minimization
- Gradual temperature ramping

**Parameters**:
- `equilibration_time`: Total equilibration time (1-50 ns)
- `final_temperature`: Target temperature (250-400 K)
- `minimize_steps`: Energy minimization steps (1000-20000)
- `restraint_force`: Protein restraint force constant (100-5000 kJ/mol/nm²)

### 3. Free Energy (`free_energy`)
**Purpose**: Free energy calculations using umbrella sampling
**Best for**: PMF calculations, binding free energies
**Key features**:
- Umbrella sampling configuration
- WHAM analysis setup
- Multiple reaction coordinates
- Bootstrap error analysis

**Parameters**:
- `coordinate_type`: Reaction coordinate type (distance, angle, dihedral, rmsd)
- `window_count`: Number of umbrella windows (10-50)
- `force_constant`: Restraint force constant (500-5000 kJ/mol/nm²)
- `window_time`: Simulation time per window (2-20 ns)
- `coordinate_range`: Total coordinate range (0.5-10 nm)

### 4. Membrane Protein (`membrane_protein`)
**Purpose**: Membrane protein simulation with lipid bilayer
**Best for**: Membrane-embedded proteins, protein-lipid interactions
**Key features**:
- Lipid bilayer setup
- Semi-isotropic pressure coupling
- Membrane-specific analysis
- Physiological temperature

**Parameters**:
- `lipid_type`: Lipid type (POPC, POPE, DPPC, DOPC, mixed)
- `membrane_thickness`: Membrane thickness (3-6 nm)
- `simulation_time`: Total simulation time (10-200 ns)
- `semi_isotropic`: Use semi-isotropic pressure coupling (boolean)

### 5. Ligand Binding (`ligand_binding`)
**Purpose**: Protein-ligand binding simulation and analysis
**Best for**: Drug discovery, binding studies, residence times
**Key features**:
- Ligand-specific analysis
- Binding site monitoring
- Contact analysis
- Residence time calculations

**Parameters**:
- `binding_site_residues`: Binding site residue numbers (list)
- `simulation_time`: Total simulation time (20-500 ns)
- `restraint_ligand`: Apply ligand restraints (boolean)

### 6. Enhanced Sampling (`enhanced_sampling`)
**Purpose**: Enhanced sampling with REMD and metadynamics
**Best for**: Conformational sampling, rare events
**Key features**:
- Multiple sampling methods
- Replica exchange setup
- Metadynamics configuration
- Free energy landscapes

**Parameters**:
- `sampling_method`: Method (remd, metadynamics, umbrella_sampling)
- `replica_count`: Number of replicas for REMD (4-32)
- `temp_range`: Temperature range for REMD [min, max]

### 7. Drug Discovery (`drug_discovery`)
**Purpose**: Drug discovery workflow with virtual screening
**Best for**: Virtual screening, ADMET predictions
**Key features**:
- High-throughput screening
- Implicit solvent for speed
- ADMET property predictions
- Interaction analysis

**Parameters**:
- `screening_mode`: Mode (binding_affinity, selectivity, admet)
- `ligand_library_size`: Number of ligands (10-100000)

### 8. Stability Analysis (`stability_analysis`)
**Purpose**: Protein stability assessment at various conditions
**Best for**: Thermal stability, mutation effects
**Key features**:
- Temperature series
- Stability metrics
- Melting temperature calculation
- Unfolding pathway analysis

**Parameters**:
- `temperature_range`: Temperature series (list)
- `simulation_time_per_temp`: Time per temperature (5-100 ns)

### 9. Conformational Analysis (`conformational_analysis`)
**Purpose**: Systematic conformational space exploration
**Best for**: Conformational states, clustering, PCA
**Key features**:
- Extended sampling
- PCA analysis
- Clustering algorithms
- Representative structures

**Parameters**:
- `sampling_method`: Method (extended_md, multiple_runs, temperature_series)
- `total_sampling_time`: Total sampling time (50-1000 ns)
- `cluster_count`: Target number of clusters (5-50)

## Using Templates

### Command Line Usage

**Basic template usage**:
```bash
# List available templates
proteinmd list-templates

# Use a template
proteinmd simulate protein.pdb --template protein_folding

# Use template with custom parameters
proteinmd simulate protein.pdb \
  --template protein_folding \
  --config custom_params.json
```

**Extended template CLI**:
```bash
# Show template details
proteinmd-templates show protein_folding

# Generate configuration from template
proteinmd-templates generate-config protein_folding \
  --parameters-file params.yaml \
  --output-file config.json
```

### Python API Usage

```python
from proteinMD.templates import TemplateManager

# Initialize template manager
manager = TemplateManager()

# Get a template
template = manager.get_template('protein_folding')

# Generate configuration
config = template.generate_config(
    simulation_time=200.0,  # 200 ns
    temperature=310.0,      # 310 K
    analysis_stride=20      # Every 20th frame
)

# Use configuration in simulation
from proteinMD.simulation import MDSimulation
sim = MDSimulation.from_config(config)
```

## Creating Custom Templates

### Method 1: From Configuration File

Create a configuration file and convert it to a template:

```bash
# Create template from config
proteinmd-templates create my_template \
  "Custom workflow description" \
  my_config.json \
  --author "Your Name" \
  --tags protein folding custom
```

### Method 2: Programmatically

```python
from proteinMD.templates import TemplateManager
from proteinMD.templates.base_template import TemplateParameter

# Create custom template class
class MyCustomTemplate(BaseTemplate):
    def __init__(self):
        super().__init__(
            name="my_custom",
            description="My custom workflow",
            version="1.0.0"
        )
        
    def _setup_parameters(self):
        self.add_parameter(TemplateParameter(
            name="custom_param",
            description="Custom parameter",
            default_value=100.0,
            parameter_type="float",
            min_value=10.0,
            max_value=1000.0,
            units="custom_unit"
        ))
        
    def generate_config(self, **kwargs):
        # Generate configuration logic
        return {
            "simulation": {...},
            # ... rest of config
        }

# Register and save template
manager = TemplateManager()
template = MyCustomTemplate()
manager.save_user_template(template)
```

### Method 3: From Existing Template

```python
# Clone and modify existing template
base_template = manager.get_template('protein_folding')
custom_config = base_template.get_default_config()

# Modify configuration
custom_config['simulation']['temperature'] = 350.0
custom_config['environment']['solvent'] = 'implicit'

# Create new template
new_template = manager.create_template_from_config(
    name="high_temp_folding",
    description="High temperature protein folding",
    config=custom_config,
    tags=["protein", "folding", "high_temperature"]
)

manager.save_user_template(new_template)
```

## Template Management

### Listing and Searching

```bash
# List all templates
proteinmd-templates list

# Search by query
proteinmd-templates list --query "folding"

# Filter by tags
proteinmd-templates list --tags protein membrane

# Filter by source
proteinmd-templates list --source user

# Show statistics
proteinmd-templates list --show-stats
```

### Import/Export

```bash
# Export template
proteinmd-templates export protein_folding my_template.json

# Export as YAML
proteinmd-templates export protein_folding my_template.yaml --format yaml

# Import template
proteinmd-templates import downloaded_template.json

# Import with overwrite
proteinmd-templates import new_template.yaml --overwrite
```

### Template Validation

```bash
# Validate template
proteinmd-templates validate protein_folding

# Validate with parameters
proteinmd-templates validate protein_folding \
  --parameters-file test_params.json
```

### Template Statistics

```bash
# Show template library statistics
proteinmd-templates stats
```

## Advanced Features

### Template Inheritance

Templates can be composed and inherited:

```python
class AdvancedFoldingTemplate(ProteinFoldingTemplate):
    def __init__(self):
        super().__init__()
        self.name = "advanced_folding"
        self.description = "Advanced folding with custom analysis"
        
        # Add additional parameters
        self.add_parameter(TemplateParameter(
            name="custom_analysis",
            description="Enable custom analysis",
            default_value=True,
            parameter_type="bool"
        ))
        
    def generate_config(self, **kwargs):
        # Get base configuration
        config = super().generate_config(**kwargs)
        
        # Add custom modifications
        if kwargs.get("custom_analysis", True):
            config["analysis"]["custom_metrics"] = True
            
        return config
```

### Parameter Validation

Templates include comprehensive parameter validation:

```python
# Parameters are automatically validated
template.validate_parameters({
    "simulation_time": 150.0,    # Valid
    "temperature": 600.0,        # Invalid - exceeds max
    "timestep": -0.001          # Invalid - below min
})
```

### Configuration Generation

Templates can generate complex configurations:

```python
# Generate configuration with parameter overrides
config = template.generate_config(
    simulation_time=300.0,
    temperature=320.0,
    custom_analysis=True
)

# Configuration is ready for simulation
print(json.dumps(config, indent=2))
```

## Examples

### Example 1: High-Temperature Unfolding Study

```bash
# Create custom unfolding template
cat > unfolding_params.json << EOF
{
  "simulation_time": 100.0,
  "temperature": 400.0,
  "analysis_stride": 5
}
EOF

# Generate configuration
proteinmd-templates generate-config protein_folding \
  --parameters-file unfolding_params.json \
  --output-file unfolding_config.json

# Run simulation
proteinmd simulate protein.pdb --config unfolding_config.json
```

### Example 2: Drug Screening Workflow

```python
from proteinMD.templates import TemplateManager

manager = TemplateManager()
drug_template = manager.get_template('drug_discovery')

# Screen multiple ligands
ligand_files = ['ligand1.pdb', 'ligand2.pdb', 'ligand3.pdb']

for ligand in ligand_files:
    config = drug_template.generate_config(
        screening_mode='binding_affinity',
        ligand_library_size=1
    )
    
    # Run screening simulation
    # ... simulation code ...
```

### Example 3: Membrane Protein Study

```yaml
# membrane_params.yaml
lipid_type: "POPC"
membrane_thickness: 4.5
simulation_time: 200.0
semi_isotropic: true
```

```bash
# Generate membrane protein configuration
proteinmd-templates generate-config membrane_protein \
  --parameters-file membrane_params.yaml \
  --output-file membrane_config.yaml \
  --format yaml

# Run simulation
proteinmd simulate membrane_protein.pdb --config membrane_config.yaml
```

## Best Practices

### Template Design

1. **Clear naming**: Use descriptive, consistent names
2. **Comprehensive parameters**: Include all relevant configurable options
3. **Validation**: Add appropriate parameter constraints
4. **Documentation**: Provide detailed descriptions and units
5. **Versioning**: Use semantic versioning for template updates

### Parameter Selection

1. **Physical relevance**: Ensure parameters are physically meaningful
2. **Reasonable defaults**: Choose sensible default values
3. **Range validation**: Set appropriate min/max values
4. **Units**: Always specify physical units
5. **Dependencies**: Consider parameter interactions

### Configuration Generation

1. **Modularity**: Generate configurations in logical sections
2. **Consistency**: Ensure parameter values are consistent
3. **Completeness**: Include all required configuration sections
4. **Validation**: Validate generated configurations
5. **Documentation**: Include comments where helpful

### Template Management

1. **Organization**: Use tags for categorization
2. **Backup**: Regularly backup custom templates
3. **Sharing**: Use export/import for template sharing
4. **Version control**: Track template changes
5. **Testing**: Validate templates with test cases

### Performance Considerations

1. **Simulation length**: Balance accuracy with computational cost
2. **Output frequency**: Optimize trajectory saving frequency
3. **Analysis stride**: Choose appropriate analysis intervals
4. **Resource usage**: Consider memory and CPU requirements
5. **Parallel processing**: Leverage available computational resources

## Troubleshooting

### Common Issues

**Template not found**:
```bash
# Check template name
proteinmd-templates list | grep template_name
```

**Parameter validation failed**:
```bash
# Validate parameters
proteinmd-templates validate template_name --parameters-file params.json
```

**Configuration generation error**:
```bash
# Check template details
proteinmd-templates show template_name --show-config
```

**Import/export issues**:
```bash
# Validate file format
file template.json
# Check file permissions
ls -la template.json
```

### Getting Help

1. **Documentation**: Check template descriptions and parameter details
2. **Examples**: Use built-in templates as references
3. **Validation**: Use template validation tools
4. **Community**: Share templates and get feedback
5. **Support**: Contact ProteinMD support for complex issues

---

*For more information, see the ProteinMD User Manual and API Documentation.*
