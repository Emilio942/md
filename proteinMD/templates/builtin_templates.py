"""
Built-in Simulation Templates for ProteinMD

This module contains predefined templates for common molecular dynamics
simulation workflows. Each template provides optimized parameters and
analysis configurations for specific types of studies.
"""

from datetime import datetime
from typing import Dict, Any

from .base_template import BaseTemplate, TemplateParameter


class ProteinFoldingTemplate(BaseTemplate):
    """
    Template for protein folding simulations.
    
    Optimized for studying protein folding dynamics with comprehensive
    analysis including RMSD, radius of gyration, secondary structure,
    and hydrogen bonding patterns.
    """
    
    def __init__(self):
        super().__init__(
            name="protein_folding",
            description="Complete protein folding simulation with comprehensive analysis",
            version="1.2.0"
        )
        self.add_tag("protein")
        self.add_tag("folding")
        self.add_tag("dynamics")
        self.created_date = "2024-12-19"
        
    def _setup_parameters(self):
        """Setup protein folding specific parameters."""
        self.add_parameter(TemplateParameter(
            name="simulation_time",
            description="Total simulation time",
            default_value=100.0,
            parameter_type="float",
            min_value=1.0,
            max_value=1000.0,
            units="ns"
        ))
        
        self.add_parameter(TemplateParameter(
            name="temperature",
            description="Simulation temperature",
            default_value=300.0,
            parameter_type="float",
            min_value=250.0,
            max_value=450.0,
            units="K"
        ))
        
        self.add_parameter(TemplateParameter(
            name="pressure", 
            description="Simulation pressure",
            default_value=1.0,
            parameter_type="float",
            min_value=0.1,
            max_value=10.0,
            units="bar"
        ))
        
        self.add_parameter(TemplateParameter(
            name="timestep",
            description="Integration timestep",
            default_value=0.002,
            parameter_type="float",
            min_value=0.0005,
            max_value=0.004,
            units="ps"
        ))
        
        self.add_parameter(TemplateParameter(
            name="save_frequency",
            description="Trajectory save frequency",
            default_value=1000,
            parameter_type="int",
            min_value=100,
            max_value=10000,
            units="steps"
        ))
        
        self.add_parameter(TemplateParameter(
            name="analysis_stride",
            description="Analysis frame stride",
            default_value=10,
            parameter_type="int",
            min_value=1,
            max_value=100
        ))
        
    def generate_config(self, **kwargs) -> Dict[str, Any]:
        """Generate protein folding simulation configuration."""
        # Get parameter values
        sim_time = kwargs.get("simulation_time", 100.0)
        temperature = kwargs.get("temperature", 300.0)
        pressure = kwargs.get("pressure", 1.0)
        timestep = kwargs.get("timestep", 0.002)
        save_freq = kwargs.get("save_frequency", 1000)
        analysis_stride = kwargs.get("analysis_stride", 10)
        
        # Calculate steps
        n_steps = int(sim_time * 1000 / timestep)  # Convert ns to steps
        
        return {
            "simulation": {
                "timestep": timestep,
                "temperature": temperature,
                "pressure": pressure,
                "n_steps": n_steps,
                "output_frequency": save_freq,
                "trajectory_output": "folding_trajectory.npz",
                "log_output": "folding_simulation.log"
            },
            "forcefield": {
                "type": "amber_ff14sb",
                "water_model": "tip3p",
                "cutoff": 1.2
            },
            "environment": {
                "solvent": "explicit",
                "box_padding": 1.2,
                "periodic_boundary": True,
                "ion_concentration": 0.15
            },
            "analysis": {
                "rmsd": True,
                "radius_of_gyration": True,
                "secondary_structure": True,
                "hydrogen_bonds": True,
                "ramachandran": True,
                "stride": analysis_stride,
                "output_dir": "folding_analysis"
            },
            "visualization": {
                "enabled": True,
                "realtime": False,
                "animation_output": "folding_animation.gif",
                "plots_output": "folding_plots"
            }
        }


class EquilibrationTemplate(BaseTemplate):
    """
    Template for system equilibration workflows.
    
    Designed for proper equilibration of protein-solvent systems
    with gradual temperature and pressure equilibration.
    """
    
    def __init__(self):
        super().__init__(
            name="equilibration",
            description="System equilibration with temperature and pressure control",
            version="1.1.0"
        )
        self.add_tag("equilibration")
        self.add_tag("preparation")
        self.created_date = "2024-12-19"
        
    def _setup_parameters(self):
        """Setup equilibration specific parameters."""
        self.add_parameter(TemplateParameter(
            name="equilibration_time",
            description="Total equilibration time",
            default_value=10.0,
            parameter_type="float",
            min_value=1.0,
            max_value=50.0,
            units="ns"
        ))
        
        self.add_parameter(TemplateParameter(
            name="final_temperature",
            description="Final equilibration temperature",
            default_value=300.0,
            parameter_type="float",
            min_value=250.0,
            max_value=400.0,
            units="K"
        ))
        
        self.add_parameter(TemplateParameter(
            name="minimize_steps",
            description="Energy minimization steps",
            default_value=5000,
            parameter_type="int",
            min_value=1000,
            max_value=20000
        ))
        
        self.add_parameter(TemplateParameter(
            name="restraint_force",
            description="Protein restraint force constant",
            default_value=1000.0,
            parameter_type="float",
            min_value=100.0,
            max_value=5000.0,
            units="kJ/mol/nm²"
        ))
        
    def generate_config(self, **kwargs) -> Dict[str, Any]:
        """Generate equilibration configuration."""
        eq_time = kwargs.get("equilibration_time", 10.0)
        temp = kwargs.get("final_temperature", 300.0)
        min_steps = kwargs.get("minimize_steps", 5000)
        restraint = kwargs.get("restraint_force", 1000.0)
        
        timestep = 0.001  # Shorter timestep for equilibration
        n_steps = int(eq_time * 1000 / timestep)
        
        return {
            "simulation": {
                "timestep": timestep,
                "temperature": temp,
                "pressure": 1.0,
                "n_steps": n_steps,
                "output_frequency": 500,
                "trajectory_output": "equilibration_trajectory.npz",
                "log_output": "equilibration.log"
            },
            "forcefield": {
                "type": "amber_ff14sb",
                "water_model": "tip3p",
                "cutoff": 1.0
            },
            "environment": {
                "solvent": "explicit",
                "box_padding": 1.0,
                "periodic_boundary": True,
                "ion_concentration": 0.15
            },
            "restraints": {
                "protein_backbone": {
                    "enabled": True,
                    "force_constant": restraint,
                    "reference": "initial"
                }
            },
            "minimization": {
                "enabled": True,
                "max_iterations": min_steps,
                "tolerance": 10.0
            },
            "analysis": {
                "rmsd": True,
                "potential_energy": True,
                "kinetic_energy": True,
                "temperature_monitor": True,
                "pressure_monitor": True,
                "stride": 5,
                "output_dir": "equilibration_analysis"
            },
            "visualization": {
                "enabled": True,
                "realtime": True,
                "energy_plots": True
            }
        }


class FreeEnergyTemplate(BaseTemplate):
    """
    Template for free energy calculations using umbrella sampling.
    
    Optimized for calculating potential of mean force (PMF) along
    reaction coordinates using umbrella sampling and WHAM analysis.
    """
    
    def __init__(self):
        super().__init__(
            name="free_energy",
            description="Free energy calculation using umbrella sampling",
            version="1.3.0"
        )
        self.add_tag("free_energy")
        self.add_tag("umbrella_sampling")
        self.add_tag("pmf")
        self.created_date = "2024-12-19"
        
    def _setup_parameters(self):
        """Setup free energy calculation parameters."""
        self.add_parameter(TemplateParameter(
            name="coordinate_type",
            description="Type of reaction coordinate",
            default_value="distance",
            parameter_type="str",
            allowed_values=["distance", "angle", "dihedral", "rmsd"]
        ))
        
        self.add_parameter(TemplateParameter(
            name="window_count",
            description="Number of umbrella windows",
            default_value=20,
            parameter_type="int",
            min_value=10,
            max_value=50
        ))
        
        self.add_parameter(TemplateParameter(
            name="force_constant",
            description="Umbrella restraint force constant",
            default_value=1000.0,
            parameter_type="float",
            min_value=500.0,
            max_value=5000.0,
            units="kJ/mol/nm²"
        ))
        
        self.add_parameter(TemplateParameter(
            name="window_time",
            description="Simulation time per window",
            default_value=5.0,
            parameter_type="float",
            min_value=2.0,
            max_value=20.0,
            units="ns"
        ))
        
        self.add_parameter(TemplateParameter(
            name="coordinate_range",
            description="Total coordinate range",
            default_value=2.0,
            parameter_type="float",
            min_value=0.5,
            max_value=10.0,
            units="nm"
        ))
        
    def generate_config(self, **kwargs) -> Dict[str, Any]:
        """Generate free energy calculation configuration."""
        coord_type = kwargs.get("coordinate_type", "distance")
        windows = kwargs.get("window_count", 20)
        force_const = kwargs.get("force_constant", 1000.0)
        window_time = kwargs.get("window_time", 5.0)
        coord_range = kwargs.get("coordinate_range", 2.0)
        
        timestep = 0.002
        steps_per_window = int(window_time * 1000 / timestep)
        
        return {
            "simulation": {
                "timestep": timestep,
                "temperature": 300.0,
                "pressure": 1.0,
                "n_steps": steps_per_window,
                "output_frequency": 1000,
                "trajectory_output": "umbrella_trajectory.npz",
                "log_output": "free_energy.log"
            },
            "forcefield": {
                "type": "amber_ff14sb",
                "water_model": "tip3p",
                "cutoff": 1.2
            },
            "environment": {
                "solvent": "explicit",
                "box_padding": 1.5,
                "periodic_boundary": True
            },
            "sampling": {
                "method": "umbrella_sampling",
                "coordinate": {
                    "type": coord_type,
                    "range": coord_range
                },
                "windows": {
                    "count": windows,
                    "force_constant": force_const,
                    "steps_per_window": steps_per_window
                }
            },
            "analysis": {
                "pmf_calculation": True,
                "wham_analysis": True,
                "bootstrap_iterations": 100,
                "output_dir": "free_energy_analysis"
            },
            "visualization": {
                "enabled": True,
                "pmf_plots": True,
                "convergence_plots": True
            }
        }


class MembraneProteinTemplate(BaseTemplate):
    """
    Template for membrane protein simulations.
    
    Specialized for simulating membrane-embedded proteins with
    lipid bilayers and appropriate boundary conditions.
    """
    
    def __init__(self):
        super().__init__(
            name="membrane_protein",
            description="Membrane protein simulation with lipid bilayer",
            version="1.0.0"
        )
        self.add_tag("membrane")
        self.add_tag("protein")
        self.add_tag("lipid")
        self.created_date = "2024-12-19"
        
    def _setup_parameters(self):
        """Setup membrane protein parameters."""
        self.add_parameter(TemplateParameter(
            name="lipid_type",
            description="Type of lipid for bilayer",
            default_value="POPC",
            parameter_type="str",
            allowed_values=["POPC", "POPE", "DPPC", "DOPC", "mixed"]
        ))
        
        self.add_parameter(TemplateParameter(
            name="membrane_thickness",
            description="Membrane thickness",
            default_value=4.0,
            parameter_type="float",
            min_value=3.0,
            max_value=6.0,
            units="nm"
        ))
        
        self.add_parameter(TemplateParameter(
            name="simulation_time",
            description="Total simulation time",
            default_value=50.0,
            parameter_type="float",
            min_value=10.0,
            max_value=200.0,
            units="ns"
        ))
        
        self.add_parameter(TemplateParameter(
            name="semi_isotropic",
            description="Use semi-isotropic pressure coupling",
            default_value=True,
            parameter_type="bool"
        ))
        
    def generate_config(self, **kwargs) -> Dict[str, Any]:
        """Generate membrane protein configuration."""
        lipid = kwargs.get("lipid_type", "POPC")
        thickness = kwargs.get("membrane_thickness", 4.0)
        sim_time = kwargs.get("simulation_time", 50.0)
        semi_iso = kwargs.get("semi_isotropic", True)
        
        timestep = 0.002
        n_steps = int(sim_time * 1000 / timestep)
        
        return {
            "simulation": {
                "timestep": timestep,
                "temperature": 310.0,  # Physiological temperature
                "pressure": 1.0,
                "n_steps": n_steps,
                "output_frequency": 2000,
                "trajectory_output": "membrane_trajectory.npz"
            },
            "forcefield": {
                "type": "amber_ff14sb",
                "lipid_forcefield": "lipid17",
                "water_model": "tip3p",
                "cutoff": 1.2
            },
            "environment": {
                "solvent": "explicit",
                "membrane": {
                    "lipid_type": lipid,
                    "thickness": thickness,
                    "area_per_lipid": 0.65
                },
                "box_type": "rectangular",
                "periodic_boundary": True
            },
            "pressure_control": {
                "semi_isotropic": semi_iso,
                "surface_tension": 0.0
            },
            "analysis": {
                "rmsd": True,
                "membrane_thickness": True,
                "lipid_order_parameters": True,
                "protein_tilt": True,
                "output_dir": "membrane_analysis"
            }
        }


class LigandBindingTemplate(BaseTemplate):
    """
    Template for protein-ligand binding studies.
    
    Optimized for studying ligand binding, residence times,
    and binding affinity calculations.
    """
    
    def __init__(self):
        super().__init__(
            name="ligand_binding",
            description="Protein-ligand binding simulation and analysis",
            version="1.1.0"
        )
        self.add_tag("ligand")
        self.add_tag("binding")
        self.add_tag("drug_discovery")
        self.created_date = "2024-12-19"
        
    def _setup_parameters(self):
        """Setup ligand binding parameters."""
        self.add_parameter(TemplateParameter(
            name="binding_site_residues",
            description="Residue numbers defining binding site",
            default_value=[],
            parameter_type="list"
        ))
        
        self.add_parameter(TemplateParameter(
            name="simulation_time",
            description="Total simulation time",
            default_value=100.0,
            parameter_type="float",
            min_value=20.0,
            max_value=500.0,
            units="ns"
        ))
        
        self.add_parameter(TemplateParameter(
            name="restraint_ligand",
            description="Apply restraints to ligand position",
            default_value=False,
            parameter_type="bool"
        ))
        
    def generate_config(self, **kwargs) -> Dict[str, Any]:
        """Generate ligand binding configuration."""
        binding_site = kwargs.get("binding_site_residues", [])
        sim_time = kwargs.get("simulation_time", 100.0)
        restraint = kwargs.get("restraint_ligand", False)
        
        timestep = 0.002
        n_steps = int(sim_time * 1000 / timestep)
        
        config = {
            "simulation": {
                "timestep": timestep,
                "temperature": 300.0,
                "pressure": 1.0,
                "n_steps": n_steps,
                "output_frequency": 1000,
                "trajectory_output": "binding_trajectory.npz"
            },
            "forcefield": {
                "type": "amber_ff14sb",
                "ligand_forcefield": "gaff2",
                "water_model": "tip3p",
                "cutoff": 1.2
            },
            "environment": {
                "solvent": "explicit",
                "box_padding": 1.5,
                "periodic_boundary": True,
                "ion_concentration": 0.15
            },
            "analysis": {
                "rmsd": True,
                "ligand_rmsd": True,
                "binding_distance": True,
                "contact_analysis": True,
                "binding_site_analysis": True,
                "residence_time": True,
                "output_dir": "binding_analysis"
            }
        }
        
        if binding_site:
            config["analysis"]["binding_site_residues"] = binding_site
            
        if restraint:
            config["restraints"] = {
                "ligand_position": {
                    "enabled": True,
                    "force_constant": 500.0
                }
            }
            
        return config


class EnhancedSamplingTemplate(BaseTemplate):
    """
    Template for enhanced sampling simulations.
    
    Includes replica exchange molecular dynamics (REMD) and
    metadynamics for exploring conformational space.
    """
    
    def __init__(self):
        super().__init__(
            name="enhanced_sampling",
            description="Enhanced sampling with REMD and metadynamics",
            version="1.2.0"
        )
        self.add_tag("enhanced_sampling")
        self.add_tag("remd")
        self.add_tag("metadynamics")
        self.created_date = "2024-12-19"
        
    def _setup_parameters(self):
        """Setup enhanced sampling parameters."""
        self.add_parameter(TemplateParameter(
            name="sampling_method",
            description="Enhanced sampling method",
            default_value="remd",
            parameter_type="str",
            allowed_values=["remd", "metadynamics", "umbrella_sampling"]
        ))
        
        self.add_parameter(TemplateParameter(
            name="replica_count",
            description="Number of replicas for REMD",
            default_value=8,
            parameter_type="int",
            min_value=4,
            max_value=32
        ))
        
        self.add_parameter(TemplateParameter(
            name="temp_range",
            description="Temperature range for REMD [min, max]",
            default_value=[300.0, 400.0],
            parameter_type="list"
        ))
        
    def generate_config(self, **kwargs) -> Dict[str, Any]:
        """Generate enhanced sampling configuration."""
        method = kwargs.get("sampling_method", "remd")
        replicas = kwargs.get("replica_count", 8)
        temp_range = kwargs.get("temp_range", [300.0, 400.0])
        
        config = {
            "simulation": {
                "timestep": 0.002,
                "n_steps": 100000,
                "output_frequency": 1000,
                "trajectory_output": f"{method}_trajectory.npz"
            },
            "forcefield": {
                "type": "amber_ff14sb",
                "water_model": "tip3p",
                "cutoff": 1.2
            },
            "environment": {
                "solvent": "explicit",
                "box_padding": 1.2,
                "periodic_boundary": True
            },
            "sampling": {
                "method": method
            },
            "analysis": {
                "rmsd": True,
                "clustering": True,
                "free_energy_landscapes": True,
                "output_dir": f"{method}_analysis"
            }
        }
        
        if method == "remd":
            config["sampling"]["replica_count"] = replicas
            config["sampling"]["temperature_range"] = temp_range
            config["sampling"]["exchange_frequency"] = 1000
            
        elif method == "metadynamics":
            config["sampling"]["collective_variables"] = ["rmsd", "radius_of_gyration"]
            config["sampling"]["bias_factor"] = 10.0
            config["sampling"]["gaussian_height"] = 1.0
            
        return config


class DrugDiscoveryTemplate(BaseTemplate):
    """
    Template for drug discovery simulations.
    
    Designed for virtual screening, lead optimization,
    and ADMET property predictions.
    """
    
    def __init__(self):
        super().__init__(
            name="drug_discovery",
            description="Drug discovery workflow with virtual screening",
            version="1.0.0"
        )
        self.add_tag("drug_discovery")
        self.add_tag("virtual_screening")
        self.add_tag("admet")
        self.created_date = "2024-12-19"
        
    def _setup_parameters(self):
        """Setup drug discovery parameters."""
        self.add_parameter(TemplateParameter(
            name="screening_mode",
            description="Virtual screening mode",
            default_value="binding_affinity",
            parameter_type="str",
            allowed_values=["binding_affinity", "selectivity", "admet"]
        ))
        
        self.add_parameter(TemplateParameter(
            name="ligand_library_size",
            description="Number of ligands to screen",
            default_value=1000,
            parameter_type="int",
            min_value=10,
            max_value=100000
        ))
        
    def generate_config(self, **kwargs) -> Dict[str, Any]:
        """Generate drug discovery configuration."""
        mode = kwargs.get("screening_mode", "binding_affinity")
        library_size = kwargs.get("ligand_library_size", 1000)
        
        return {
            "simulation": {
                "timestep": 0.002,
                "temperature": 300.0,
                "n_steps": 25000,  # Shorter for screening
                "output_frequency": 1000
            },
            "forcefield": {
                "type": "amber_ff14sb",
                "ligand_forcefield": "gaff2",
                "water_model": "tip3p"
            },
            "environment": {
                "solvent": "implicit",  # Faster for screening
                "gb_model": "obc2"
            },
            "screening": {
                "mode": mode,
                "library_size": library_size,
                "docking_scoring": True,
                "md_scoring": True
            },
            "analysis": {
                "binding_affinity": True,
                "interaction_analysis": True,
                "pharmacophore_mapping": True,
                "admet_prediction": True,
                "output_dir": "drug_discovery_analysis"
            }
        }


class StabilityAnalysisTemplate(BaseTemplate):
    """
    Template for protein stability analysis.
    
    Designed for assessing thermal stability, pH effects,
    and mutation impact on protein structure.
    """
    
    def __init__(self):
        super().__init__(
            name="stability_analysis",
            description="Protein stability assessment at various conditions",
            version="1.0.0"
        )
        self.add_tag("stability")
        self.add_tag("thermodynamics")
        self.add_tag("mutations")
        
    def _setup_parameters(self):
        """Setup stability analysis parameters."""
        self.add_parameter(TemplateParameter(
            name="temperature_range",
            description="Temperature range for stability testing",
            default_value=[300.0, 350.0, 400.0],
            parameter_type="list"
        ))
        
        self.add_parameter(TemplateParameter(
            name="simulation_time_per_temp",
            description="Simulation time per temperature",
            default_value=20.0,
            parameter_type="float",
            min_value=5.0,
            max_value=100.0,
            units="ns"
        ))
        
    def generate_config(self, **kwargs) -> Dict[str, Any]:
        """Generate stability analysis configuration."""
        temp_range = kwargs.get("temperature_range", [300.0, 350.0, 400.0])
        sim_time = kwargs.get("simulation_time_per_temp", 20.0)
        
        timestep = 0.002
        n_steps = int(sim_time * 1000 / timestep)
        
        return {
            "simulation": {
                "timestep": timestep,
                "temperature_series": temp_range,
                "n_steps": n_steps,
                "output_frequency": 1000,
                "trajectory_output": "stability_trajectory.npz"
            },
            "forcefield": {
                "type": "amber_ff14sb",
                "water_model": "tip3p",
                "cutoff": 1.2
            },
            "environment": {
                "solvent": "explicit",
                "box_padding": 1.0,
                "periodic_boundary": True
            },
            "analysis": {
                "rmsd": True,
                "radius_of_gyration": True,
                "secondary_structure": True,
                "hydrogen_bonds": True,
                "thermal_stability": True,
                "melting_temperature": True,
                "unfolding_pathway": True,
                "output_dir": "stability_analysis"
            },
            "visualization": {
                "enabled": True,
                "temperature_plots": True,
                "stability_curves": True
            }
        }


class ConformationalAnalysisTemplate(BaseTemplate):
    """
    Template for conformational space exploration.
    
    Designed for systematic exploration of protein conformational
    states using various sampling methods and clustering analysis.
    """
    
    def __init__(self):
        super().__init__(
            name="conformational_analysis",
            description="Systematic conformational space exploration",
            version="1.1.0"
        )
        self.add_tag("conformational")
        self.add_tag("clustering")
        self.add_tag("pca")
        
    def _setup_parameters(self):
        """Setup conformational analysis parameters."""
        self.add_parameter(TemplateParameter(
            name="sampling_method",
            description="Conformational sampling method",
            default_value="extended_md",
            parameter_type="str",
            allowed_values=["extended_md", "multiple_runs", "temperature_series"]
        ))
        
        self.add_parameter(TemplateParameter(
            name="total_sampling_time",
            description="Total sampling time",
            default_value=200.0,
            parameter_type="float",
            min_value=50.0,
            max_value=1000.0,
            units="ns"
        ))
        
        self.add_parameter(TemplateParameter(
            name="cluster_count",
            description="Target number of clusters",
            default_value=10,
            parameter_type="int",
            min_value=5,
            max_value=50
        ))
        
    def generate_config(self, **kwargs) -> Dict[str, Any]:
        """Generate conformational analysis configuration."""
        method = kwargs.get("sampling_method", "extended_md")
        total_time = kwargs.get("total_sampling_time", 200.0)
        clusters = kwargs.get("cluster_count", 10)
        
        timestep = 0.002
        
        if method == "extended_md":
            n_steps = int(total_time * 1000 / timestep)
            runs = 1
        elif method == "multiple_runs":
            runs = 5
            n_steps = int((total_time / runs) * 1000 / timestep)
        else:  # temperature_series
            runs = 3
            n_steps = int((total_time / runs) * 1000 / timestep)
        
        return {
            "simulation": {
                "timestep": timestep,
                "temperature": 300.0,
                "n_steps": n_steps,
                "multiple_runs": runs if method != "extended_md" else 1,
                "output_frequency": 2000,
                "trajectory_output": "conformational_trajectory.npz"
            },
            "forcefield": {
                "type": "amber_ff14sb",
                "water_model": "tip3p",
                "cutoff": 1.2
            },
            "environment": {
                "solvent": "explicit",
                "box_padding": 1.2,
                "periodic_boundary": True
            },
            "analysis": {
                "rmsd": True,
                "principal_component_analysis": True,
                "clustering": True,
                "cluster_count": clusters,
                "free_energy_landscapes": True,
                "conformational_transitions": True,
                "representative_structures": True,
                "output_dir": "conformational_analysis"
            },
            "visualization": {
                "enabled": True,
                "pca_plots": True,
                "cluster_visualization": True,
                "energy_landscapes": True
            }
        }


# Dictionary of all built-in templates for easy access
BUILTIN_TEMPLATES = {
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

__all__ = [
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
