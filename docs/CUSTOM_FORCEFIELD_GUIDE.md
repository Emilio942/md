# Custom Force Field Import Documentation

## Overview

The Custom Force Field Import module allows users to define and import their own force field parameters in ProteinMD. This feature provides flexibility for research applications requiring specialized parameters or novel force field developments.

## Features

- **JSON and XML Format Support**: Import force field parameters from JSON or XML files
- **Comprehensive Validation**: Automatic validation of parameter values for physical reasonableness
- **Unit Conversion**: Automatic conversion between different unit systems
- **Error Handling**: Robust error handling for invalid parameters and file formats
- **Export Functionality**: Export custom force fields back to JSON format
- **Integration**: Seamless integration with existing ProteinMD force field infrastructure

## File Formats

### JSON Format

The JSON format provides a structured way to define force field parameters:

```json
{
  "metadata": {
    "name": "Custom Force Field Name",
    "version": "1.0.0",
    "description": "Description of the force field",
    "author": "Author Name",
    "date": "2025-06-11",
    "units": {
      "length": "nm",
      "energy": "kJ/mol",
      "mass": "amu",
      "angle": "degrees"
    }
  },
  "atom_types": [
    {
      "atom_type": "CT",
      "mass": 12.01,
      "sigma": 0.339967,
      "epsilon": 0.457730,
      "charge": -0.1,
      "description": "Aliphatic carbon",
      "source": "Custom parameters"
    }
  ],
  "bond_types": [
    {
      "atom_types": ["CT", "HC"],
      "k": 284512.0,
      "r0": 0.1090,
      "description": "Carbon-hydrogen bond",
      "source": "Custom parameters"
    }
  ],
  "angle_types": [
    {
      "atom_types": ["HC", "CT", "HC"],
      "k": 276.144,
      "theta0": 109.5,
      "description": "H-C-H angle",
      "source": "Custom parameters"
    }
  ],
  "dihedral_types": [
    {
      "atom_types": ["HC", "CT", "CT", "HC"],
      "k": 0.6508,
      "n": 3,
      "phase": 0.0,
      "description": "H-C-C-H dihedral",
      "source": "Custom parameters"
    }
  ]
}
```

### XML Format

The XML format provides an alternative structured approach:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<forcefield>
  <metadata>
    <name>Custom Force Field Name</name>
    <version>1.0.0</version>
    <units>
      <length>nm</length>
      <energy>kJ/mol</energy>
      <mass>amu</mass>
      <angle>degrees</angle>
    </units>
  </metadata>
  
  <atom_types>
    <atom_type>
      <atom_type>CT</atom_type>
      <mass>12.01</mass>
      <sigma>0.339967</sigma>
      <epsilon>0.457730</epsilon>
      <charge>-0.1</charge>
      <description>Aliphatic carbon</description>
    </atom_type>
  </atom_types>
  
  <bond_types>
    <bond_type>
      <atom_types>CT, HC</atom_types>
      <k>284512.0</k>
      <r0>0.1090</r0>
      <description>Carbon-hydrogen bond</description>
    </bond_type>
  </bond_types>
</forcefield>
```

## Usage Examples

### Basic Import

```python
from proteinMD.forcefield.custom_import import import_custom_forcefield

# Import from JSON file
custom_ff = import_custom_forcefield("my_forcefield.json")

# Import from XML file
custom_ff = import_custom_forcefield("my_forcefield.xml")
```

### Advanced Usage

```python
from proteinMD.forcefield.custom_import import CustomForceFieldImporter

# Create importer with custom validation settings
importer = CustomForceFieldImporter(validate_schema=True)

# Import force field
custom_ff = importer.import_from_json("my_forcefield.json")

# Access parameters
atom_params = custom_ff.get_atom_parameters("CT")
bond_params = custom_ff.get_bond_parameters("CT", "HC")

# Export to new file
custom_ff.export_to_json("exported_forcefield.json")
```

### Using in Simulations

```python
from proteinMD.core.simulation import MolecularDynamicsSimulation
from proteinMD.forcefield.custom_import import import_custom_forcefield

# Load custom force field
custom_ff = import_custom_forcefield("my_forcefield.json")

# Create simulation with custom force field
simulation = MolecularDynamicsSimulation(
    num_particles=1000,
    box_dimensions=[5.0, 5.0, 5.0],
    temperature=300.0
)

# Apply custom force field parameters
simulation.set_force_field(custom_ff)
```

## Parameter Definitions

### Atom Types
- **atom_type**: Unique identifier for the atom type (string)
- **mass**: Atomic mass in amu (float, > 0)
- **sigma**: Lennard-Jones sigma parameter in nm (float, > 0)
- **epsilon**: Lennard-Jones epsilon parameter in kJ/mol (float, ≥ 0)
- **charge**: Partial charge in elementary units (float, optional)
- **description**: Human-readable description (string, optional)
- **source**: Reference or source of parameters (string, optional)

### Bond Types
- **atom_types**: Array of two atom type identifiers (string array)
- **k**: Spring constant in kJ/mol/nm² (float, > 0)
- **r0**: Equilibrium bond length in nm (float, > 0)
- **description**: Human-readable description (string, optional)
- **source**: Reference or source of parameters (string, optional)

### Angle Types
- **atom_types**: Array of three atom type identifiers (string array)
- **k**: Spring constant in kJ/mol/rad² (float, > 0)
- **theta0**: Equilibrium angle in degrees (float, 0-180)
- **description**: Human-readable description (string, optional)
- **source**: Reference or source of parameters (string, optional)

### Dihedral Types
- **atom_types**: Array of four atom type identifiers (string array)
- **k**: Force constant in kJ/mol (float)
- **n**: Periodicity (integer, > 0)
- **phase**: Phase angle in degrees (float, -180 to 180)
- **description**: Human-readable description (string, optional)
- **source**: Reference or source of parameters (string, optional)

## Unit Systems

The following unit systems are supported with automatic conversion:

### Length
- **nm** (nanometers) - default
- **angstrom** (Ångströms)

### Energy
- **kJ/mol** (kilojoules per mole) - default
- **kcal/mol** (kilocalories per mole)

### Mass
- **amu** (atomic mass units) - default
- **g/mol** (grams per mole)

### Angle
- **degrees** - default
- **radians**

## Validation

The import process includes several validation steps:

### Schema Validation
- JSON structure validation against predefined schema (when jsonschema is available)
- Required field checking
- Data type validation

### Parameter Validation
- Physical reasonableness checks (e.g., positive masses, reasonable bond lengths)
- Value range validation (e.g., angles between 0-180 degrees)
- Consistency checks

### Error Handling
- Clear error messages for invalid parameters
- File format detection and appropriate error reporting
- Graceful handling of missing optional dependencies

## Best Practices

### Parameter Development
1. **Start with known parameters**: Base custom parameters on established force fields
2. **Document sources**: Always include description and source fields
3. **Validate thoroughly**: Test parameters with small systems before large simulations
4. **Use consistent units**: Stick to one unit system throughout your parameter file

### File Organization
1. **Use descriptive names**: Include force field type and version in filenames
2. **Version control**: Keep track of parameter versions and changes
3. **Backup parameters**: Maintain copies of working parameter sets

### Performance Considerations
1. **Minimize parameter sets**: Only include necessary atom types and interactions
2. **Optimize cutoffs**: Choose appropriate cutoff distances for your system
3. **Test convergence**: Verify energy and force convergence with your parameters

## Troubleshooting

### Common Issues

#### File Format Errors
- Ensure JSON syntax is valid (use a JSON validator)
- Check XML structure and closing tags
- Verify file encoding (UTF-8 recommended)

#### Parameter Validation Failures
- Check for negative or zero values where positive values are required
- Verify angle values are within valid ranges
- Ensure all required fields are present

#### Import Errors
- Verify file paths are correct
- Check file permissions
- Ensure ProteinMD modules are properly installed

### Error Messages

#### "JSON schema validation failed"
- Check JSON structure against the schema
- Verify all required fields are present
- Check data types (numbers vs strings)

#### "Parameter validation failed"
- Review parameter values for physical reasonableness
- Check value ranges (especially angles)
- Verify positive values where required

#### "File not found"
- Check file path spelling and location
- Verify file permissions
- Ensure file exists and is accessible

## Examples

See the `examples/` directory for complete example files:
- `simple_custom_ff.json` - Basic JSON example
- `custom_ff_example.xml` - Basic XML example

## API Reference

### Functions

#### `import_custom_forcefield(filename, validate_schema=True)`
Import a custom force field from JSON or XML file.

**Parameters:**
- `filename` (str or Path): Path to the force field file
- `validate_schema` (bool): Whether to validate JSON schema

**Returns:**
- `CustomForceField`: The imported custom force field

### Classes

#### `CustomForceFieldImporter`
Main importer class for custom force fields.

#### `CustomForceField`
Custom force field implementation extending base ForceField class.

**Methods:**
- `get_atom_parameters(atom_type)`: Get parameters for atom type
- `get_bond_parameters(type1, type2)`: Get bond parameters
- `get_angle_parameters(type1, type2, type3)`: Get angle parameters
- `get_dihedral_parameters(type1, type2, type3, type4)`: Get dihedral parameters
- `export_to_json(filename)`: Export force field to JSON file

## Integration with ProteinMD

Custom force fields integrate seamlessly with the ProteinMD workflow:

1. **CLI Integration**: Use custom force fields in CLI workflows
2. **Analysis Tools**: All analysis tools work with custom force fields
3. **Visualization**: Custom force field simulations can be visualized normally
4. **Performance**: Custom force fields use the same optimized calculation routines

This documentation provides comprehensive guidance for using the Custom Force Field Import feature in ProteinMD. For additional support, consult the main ProteinMD documentation or contact the development team.
