# ProteinMD Command Line Interface Documentation

## Overview

The ProteinMD CLI provides a comprehensive command-line interface for molecular dynamics simulations, analysis, and visualization. This tool enables automated workflows, batch processing, and integration with computational pipelines.

## Installation

### Prerequisites
- Python 3.8+
- NumPy, SciPy, Matplotlib
- ProteinMD package

### Setup
1. Make the CLI executable:
   ```bash
   chmod +x proteinmd
   ```

2. Add to PATH (optional):
   ```bash
   sudo ln -s $(pwd)/proteinmd /usr/local/bin/proteinmd
   ```

3. Enable bash completion:
   ```bash
   proteinmd bash-completion
   source ~/.proteinmd/proteinmd_completion.bash
   ```

## Commands

### Basic Simulation
Run a molecular dynamics simulation:
```bash
proteinmd simulate protein.pdb
```

### Simulation with Template
Use predefined workflow templates:
```bash
proteinmd simulate protein.pdb --template protein_folding
```

### Simulation with Custom Configuration
Use custom configuration file:
```bash
proteinmd simulate protein.pdb --config my_config.json --output-dir results/
```

### Analysis Only
Analyze existing trajectory data:
```bash
proteinmd analyze trajectory.npz protein.pdb --output-dir analysis/
```

### Batch Processing
Process multiple PDB files:
```bash
proteinmd batch-process ./structures/ --template equilibration --output-dir batch_results/
```

### Template Management
List available templates:
```bash
proteinmd list-templates
```

Create custom template:
```bash
proteinmd create-template my_template "Custom workflow" config.json
```

### Utility Commands
Create sample configuration:
```bash
proteinmd sample-config --output my_config.json
```

Validate installation:
```bash
proteinmd validate-setup
```

## Configuration Files

### JSON Configuration Example
```json
{
  "simulation": {
    "timestep": 0.002,
    "temperature": 300.0,
    "n_steps": 50000,
    "output_frequency": 100,
    "trajectory_output": "trajectory.npz"
  },
  "forcefield": {
    "type": "amber_ff14sb",
    "water_model": "tip3p",
    "cutoff": 1.2
  },
  "environment": {
    "solvent": "explicit",
    "box_padding": 1.0,
    "periodic_boundary": true
  },
  "analysis": {
    "rmsd": true,
    "ramachandran": true,
    "radius_of_gyration": true,
    "secondary_structure": true,
    "hydrogen_bonds": true,
    "output_dir": "analysis_results"
  },
  "visualization": {
    "enabled": true,
    "realtime": false,
    "animation_output": "animation.gif",
    "plots_output": "plots"
  }
}
```

### YAML Configuration Example
```yaml
simulation:
  timestep: 0.002
  temperature: 300.0
  n_steps: 50000
  output_frequency: 100

forcefield:
  type: amber_ff14sb
  water_model: tip3p

environment:
  solvent: explicit
  box_padding: 1.0

analysis:
  rmsd: true
  ramachandran: true
  secondary_structure: true

visualization:
  enabled: true
  realtime: false
```

## Workflow Templates

### Built-in Templates

#### `protein_folding`
Standard protein folding simulation with comprehensive analysis:
- 50,000 steps at 300K
- Explicit water solvation
- RMSD, Rg, and secondary structure analysis

#### `equilibration`
System equilibration workflow:
- 25,000 steps with shorter timestep
- Explicit water
- RMSD and hydrogen bond analysis

#### `free_energy`
Free energy calculation using umbrella sampling:
- 100,000 steps
- 20 umbrella windows
- PMF calculation

#### `steered_md`
Steered molecular dynamics for protein unfolding:
- 50,000 steps
- Constant velocity pulling
- Force curve analysis

### Custom Templates
Create your own templates by:
1. Creating a configuration file
2. Using `proteinmd create-template`
3. Templates are stored in `~/.proteinmd/templates/`

## Batch Processing

Process multiple structures efficiently:

```bash
# Process all PDB files in a directory
proteinmd batch-process ./structures/ --template protein_folding

# Use custom pattern and configuration
proteinmd batch-process ./pdbs/ --pattern "*.pdb" --config custom.json

# Parallel processing (future feature)
proteinmd batch-process ./structures/ --parallel --template equilibration
```

### Batch Output Structure
```
batch_results/
├── batch_summary.json
├── batch_summary.txt
├── protein1/
│   ├── trajectory.npz
│   ├── analysis_results/
│   └── simulation_report.json
├── protein2/
│   └── ...
└── protein3/
    └── ...
```

## Analysis Pipeline

The CLI automatically runs analysis based on configuration:

### Available Analyses
- **RMSD**: Root Mean Square Deviation
- **Ramachandran**: Phi-Psi angle analysis
- **Radius of Gyration**: Protein compactness
- **Secondary Structure**: α-helix and β-sheet content
- **Hydrogen Bonds**: H-bond network analysis

### Analysis Configuration
```json
{
  "analysis": {
    "rmsd": true,
    "ramachandran": true,
    "radius_of_gyration": true,
    "secondary_structure": true,
    "hydrogen_bonds": true,
    "output_dir": "analysis_results"
  }
}
```

## Visualization

### Automatic Visualization
The CLI generates visualizations automatically:
- 3D protein structure
- Trajectory animations
- Analysis plots
- Energy dashboards

### Real-time Visualization
Enable real-time monitoring during simulation:
```json
{
  "visualization": {
    "enabled": true,
    "realtime": true,
    "animation_output": "animation.gif"
  }
}
```

## Error Handling and Logging

### Log Files
- Simulation logs: `simulation.log`
- CLI logs: `proteinmd_cli.log`

### Return Codes
- `0`: Success
- `1`: General error
- `130`: Interrupted by user (Ctrl+C)

### Verbose Output
Use `-v` or `--verbose` for detailed output:
```bash
proteinmd simulate protein.pdb --verbose
```

## Integration with Scripts

### Shell Scripts
```bash
#!/bin/bash
for pdb in *.pdb; do
    proteinmd simulate "$pdb" --template protein_folding
    if [ $? -eq 0 ]; then
        echo "Successfully processed $pdb"
    else
        echo "Failed to process $pdb"
    fi
done
```

### Python Scripts
```python
import subprocess
import sys

def run_proteinmd(pdb_file, template="protein_folding"):
    cmd = ["proteinmd", "simulate", pdb_file, "--template", template]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Success: {pdb_file}")
        return True
    else:
        print(f"Error processing {pdb_file}: {result.stderr}")
        return False

# Process multiple files
pdb_files = ["protein1.pdb", "protein2.pdb", "protein3.pdb"]
for pdb in pdb_files:
    run_proteinmd(pdb)
```

## Performance Considerations

### Resource Management
- Memory usage scales with system size
- Explicit water increases computational cost
- Real-time visualization adds overhead

### Optimization Tips
1. Use implicit solvent for faster calculations
2. Adjust output frequency to reduce I/O
3. Disable real-time visualization for production runs
4. Use batch processing for multiple structures

## Troubleshooting

### Common Issues

#### Missing Dependencies
```bash
proteinmd validate-setup
```
This command checks all dependencies and installation.

#### Configuration Errors
- Validate JSON/YAML syntax
- Check file paths are absolute or relative to working directory
- Ensure template names are correct

#### Memory Issues
- Reduce system size or simulation length
- Use implicit solvent instead of explicit
- Adjust output frequency

#### File Access Errors
- Check file permissions
- Ensure output directories are writable
- Verify input files exist and are readable

### Getting Help
```bash
proteinmd --help
proteinmd simulate --help
proteinmd analyze --help
```

## Advanced Usage

### Environment Variables
Set default configurations:
```bash
export PROTEINMD_CONFIG=/path/to/default/config.json
export PROTEINMD_TEMPLATES=/path/to/templates/
```

### Configuration Hierarchy
1. Command-line arguments (highest priority)
2. Configuration file specified with `--config`
3. Template configuration
4. Default configuration (lowest priority)

### Custom Force Fields
Add support for custom force fields in configuration:
```json
{
  "forcefield": {
    "type": "custom",
    "parameter_file": "/path/to/parameters.json"
  }
}
```

## Examples

### Complete Workflow Example
```bash
# 1. Create sample configuration
proteinmd sample-config --output my_config.json

# 2. Edit configuration as needed
# (modify my_config.json)

# 3. Run simulation
proteinmd simulate protein.pdb --config my_config.json --output-dir results/

# 4. Additional analysis
proteinmd analyze results/trajectory.npz protein.pdb --output-dir extended_analysis/
```

### High-Throughput Processing
```bash
# Process entire protein family
proteinmd batch-process ./protein_family/ \
    --template free_energy \
    --output-dir family_analysis/ \
    --verbose

# Check batch results
cat family_analysis/batch_summary.txt
```

### Custom Workflow Creation
```bash
# 1. Create custom configuration
cat > custom_workflow.json << EOF
{
  "simulation": {
    "n_steps": 75000,
    "temperature": 310.0,
    "timestep": 0.0015
  },
  "analysis": {
    "rmsd": true,
    "secondary_structure": true,
    "hydrogen_bonds": true
  }
}
EOF

# 2. Create template
proteinmd create-template high_temp "High temperature simulation" custom_workflow.json

# 3. Use template
proteinmd simulate protein.pdb --template high_temp
```

## API Reference

The CLI provides programmatic access through the `ProteinMDCLI` class for integration with custom applications. See the source code documentation for detailed API information.
