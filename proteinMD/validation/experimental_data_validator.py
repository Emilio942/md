"""
Experimental Data Validation for ProteinMD Integration Tests

Task 10.2: Integration Tests 📊 - Experimental Data Validation Component

This module provides validation against experimental and reference data
for molecular dynamics simulations.
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Validation result status."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIP = "SKIP"


@dataclass
class ValidationMetric:
    """Container for validation metric results."""
    name: str
    value: float
    reference_range: Tuple[float, float]
    unit: str
    tolerance: float = 0.05
    result: ValidationResult = ValidationResult.SKIP
    
    def validate(self) -> ValidationResult:
        """Validate the metric against reference range."""
        min_val, max_val = self.reference_range
        
        if min_val <= self.value <= max_val:
            self.result = ValidationResult.PASS
        elif min_val * (1 - self.tolerance) <= self.value <= max_val * (1 + self.tolerance):
            self.result = ValidationResult.WARNING
        else:
            self.result = ValidationResult.FAIL
        
        return self.result


class ExperimentalDataValidator:
    """Main class for validating simulation results against experimental data."""
    
    def __init__(self):
        self.validation_results = {}
        self.reference_data = self._load_reference_data()
    
    def _load_reference_data(self) -> Dict:
        """Load reference experimental data."""
        return {
            # AMBER ff14SB validation data
            'amber_ff14sb': {
                'alanine_dipeptide': {
                    'total_energy': (-305.8, -295.8),  # kJ/mol ± 5%
                    'bond_energy': (240.0, 250.0),
                    'angle_energy': (130.0, 140.0),
                    'dihedral_energy': (10.0, 20.0),
                    'van_der_waals': (-15.0, -5.0),
                    'electrostatic': (-85.0, -75.0),
                    'rmsd_crystal': (0.05, 0.25),  # nm
                },
                'ubiquitin': {
                    'radius_of_gyration': (1.15, 1.25),  # nm
                    'rmsd_crystal': (0.10, 0.30),  # nm
                    'alpha_helix_content': (0.15, 0.35),  # fraction
                    'beta_sheet_content': (0.35, 0.55),  # fraction
                    'hydrogen_bonds': (45, 65),  # count
                    'solvent_accessible_surface': (4500, 5500),  # Ų
                }
            },
            
            # Water model validation
            'water_models': {
                'tip3p': {
                    'density_298K': (0.99, 1.01),  # g/cm³
                    'self_diffusion': (2.3e-5, 2.8e-5),  # cm²/s
                    'dielectric_constant': (75, 85),
                    'heat_capacity': (4.0, 4.3),  # J/g/K
                    'thermal_expansion': (2.5e-4, 3.5e-4),  # K⁻¹
                },
                'tip4p': {
                    'density_298K': (0.995, 1.005),  # g/cm³
                    'self_diffusion': (2.0e-5, 2.5e-5),  # cm²/s
                    'dielectric_constant': (78, 82),
                }
            },
            
            # Protein folding benchmarks
            'protein_folding': {
                'villin_headpiece': {
                    'folding_time': (1e-6, 1e-5),  # seconds
                    'native_contacts': (0.7, 0.9),  # fraction
                    'radius_of_gyration_folded': (0.8, 1.0),  # nm
                    'radius_of_gyration_unfolded': (1.5, 2.0),  # nm
                },
                'chignolin': {
                    'folding_time': (1e-7, 1e-6),  # seconds
                    'native_contacts': (0.8, 0.95),  # fraction
                    'beta_hairpin_content': (0.6, 0.9),  # fraction
                }
            },
            
            # Membrane systems
            'lipid_bilayers': {
                'popc': {
                    'area_per_lipid': (64, 70),  # Ų
                    'membrane_thickness': (3.6, 4.0),  # nm
                    'order_parameter_sn1': (0.15, 0.25),
                    'order_parameter_sn2': (0.10, 0.20),
                }
            },
            
            # Drug-target interactions
            'drug_binding': {
                'lysozyme_drug': {
                    'binding_affinity': (-8.0, -6.0),  # kcal/mol
                    'residence_time': (1e-3, 1e-1),  # seconds
                    'rmsd_bound': (0.1, 0.3),  # nm
                }
            }
        }
    
    def validate_amber_ff14sb_energies(self, energy_components: Dict[str, float], 
                                     system_type: str = 'alanine_dipeptide') -> Dict[str, ValidationMetric]:
        """Validate AMBER ff14SB energy components."""
        logger.info(f"Validating AMBER ff14SB energies for {system_type}")
        
        results = {}
        reference = self.reference_data['amber_ff14sb'][system_type]
        
        for component, value in energy_components.items():
            if component in reference:
                metric = ValidationMetric(
                    name=f"amber_ff14sb_{component}",
                    value=value,
                    reference_range=reference[component],
                    unit="kJ/mol" if "energy" in component else "nm"
                )
                metric.validate()
                results[component] = metric
                
                logger.info(f"{component}: {value:.2f} {metric.unit} - {metric.result.value}")
        
        self.validation_results[f'amber_ff14sb_{system_type}'] = results
        return results
    
    def validate_water_properties(self, properties: Dict[str, float], 
                                 model: str = 'tip3p') -> Dict[str, ValidationMetric]:
        """Validate water model properties."""
        logger.info(f"Validating water properties for {model}")
        
        results = {}
        reference = self.reference_data['water_models'][model]
        
        for prop, value in properties.items():
            if prop in reference:
                unit_map = {
                    'density_298K': 'g/cm³',
                    'self_diffusion': 'cm²/s',
                    'dielectric_constant': '',
                    'heat_capacity': 'J/g/K',
                    'thermal_expansion': 'K⁻¹'
                }
                
                metric = ValidationMetric(
                    name=f"water_{prop}",
                    value=value,
                    reference_range=reference[prop],
                    unit=unit_map.get(prop, '')
                )
                metric.validate()
                results[prop] = metric
                
                logger.info(f"{prop}: {value:.3e} {metric.unit} - {metric.result.value}")
        
        self.validation_results[f'water_{model}'] = results
        return results
    
    def validate_protein_structure(self, structure_metrics: Dict[str, float], 
                                 protein_type: str = 'ubiquitin') -> Dict[str, ValidationMetric]:
        """Validate protein structural properties."""
        logger.info(f"Validating protein structure for {protein_type}")
        
        results = {}
        reference = self.reference_data['amber_ff14sb'][protein_type]
        
        for metric_name, value in structure_metrics.items():
            if metric_name in reference:
                unit_map = {
                    'radius_of_gyration': 'nm',
                    'rmsd_crystal': 'nm',
                    'alpha_helix_content': 'fraction',
                    'beta_sheet_content': 'fraction',
                    'hydrogen_bonds': 'count',
                    'solvent_accessible_surface': 'Ų'
                }
                
                metric = ValidationMetric(
                    name=f"protein_{metric_name}",
                    value=value,
                    reference_range=reference[metric_name],
                    unit=unit_map.get(metric_name, '')
                )
                metric.validate()
                results[metric_name] = metric
                
                logger.info(f"{metric_name}: {value:.3f} {metric.unit} - {metric.result.value}")
        
        self.validation_results[f'protein_{protein_type}'] = results
        return results
    
    def validate_folding_dynamics(self, dynamics_metrics: Dict[str, float], 
                                protein_type: str = 'villin_headpiece') -> Dict[str, ValidationMetric]:
        """Validate protein folding dynamics."""
        logger.info(f"Validating folding dynamics for {protein_type}")
        
        results = {}
        reference = self.reference_data['protein_folding'][protein_type]
        
        for metric_name, value in dynamics_metrics.items():
            if metric_name in reference:
                unit_map = {
                    'folding_time': 's',
                    'native_contacts': 'fraction',
                    'radius_of_gyration_folded': 'nm',
                    'radius_of_gyration_unfolded': 'nm',
                    'beta_hairpin_content': 'fraction'
                }
                
                metric = ValidationMetric(
                    name=f"folding_{metric_name}",
                    value=value,
                    reference_range=reference[metric_name],
                    unit=unit_map.get(metric_name, '')
                )
                metric.validate()
                results[metric_name] = metric
                
                logger.info(f"{metric_name}: {value:.3e} {metric.unit} - {metric.result.value}")
        
        self.validation_results[f'folding_{protein_type}'] = results
        return results
    
    def validate_membrane_properties(self, membrane_metrics: Dict[str, float], 
                                   lipid_type: str = 'popc') -> Dict[str, ValidationMetric]:
        """Validate lipid bilayer properties."""
        logger.info(f"Validating membrane properties for {lipid_type}")
        
        results = {}
        reference = self.reference_data['lipid_bilayers'][lipid_type]
        
        for metric_name, value in membrane_metrics.items():
            if metric_name in reference:
                unit_map = {
                    'area_per_lipid': 'Ų',
                    'membrane_thickness': 'nm',
                    'order_parameter_sn1': '',
                    'order_parameter_sn2': ''
                }
                
                metric = ValidationMetric(
                    name=f"membrane_{metric_name}",
                    value=value,
                    reference_range=reference[metric_name],
                    unit=unit_map.get(metric_name, '')
                )
                metric.validate()
                results[metric_name] = metric
                
                logger.info(f"{metric_name}: {value:.3f} {metric.unit} - {metric.result.value}")
        
        self.validation_results[f'membrane_{lipid_type}'] = results
        return results
    
    def validate_drug_binding(self, binding_metrics: Dict[str, float], 
                            system_type: str = 'lysozyme_drug') -> Dict[str, ValidationMetric]:
        """Validate drug-target binding properties."""
        logger.info(f"Validating drug binding for {system_type}")
        
        results = {}
        reference = self.reference_data['drug_binding'][system_type]
        
        for metric_name, value in binding_metrics.items():
            if metric_name in reference:
                unit_map = {
                    'binding_affinity': 'kcal/mol',
                    'residence_time': 's',
                    'rmsd_bound': 'nm'
                }
                
                metric = ValidationMetric(
                    name=f"binding_{metric_name}",
                    value=value,
                    reference_range=reference[metric_name],
                    unit=unit_map.get(metric_name, '')
                )
                metric.validate()
                results[metric_name] = metric
                
                logger.info(f"{metric_name}: {value:.3f} {metric.unit} - {metric.result.value}")
        
        self.validation_results[f'binding_{system_type}'] = results
        return results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        logger.info("Generating validation report...")
        
        total_validations = 0
        passed_validations = 0
        warning_validations = 0
        failed_validations = 0
        
        detailed_results = {}
        
        for category, results in self.validation_results.items():
            category_results = {
                'metrics': {},
                'summary': {
                    'total': 0,
                    'passed': 0,
                    'warnings': 0,
                    'failed': 0
                }
            }
            
            for metric_name, metric in results.items():
                total_validations += 1
                category_results['summary']['total'] += 1
                
                metric_info = {
                    'value': metric.value,
                    'reference_range': metric.reference_range,
                    'unit': metric.unit,
                    'result': metric.result.value,
                    'in_range': metric.result == ValidationResult.PASS
                }
                
                if metric.result == ValidationResult.PASS:
                    passed_validations += 1
                    category_results['summary']['passed'] += 1
                elif metric.result == ValidationResult.WARNING:
                    warning_validations += 1
                    category_results['summary']['warnings'] += 1
                elif metric.result == ValidationResult.FAIL:
                    failed_validations += 1
                    category_results['summary']['failed'] += 1
                
                category_results['metrics'][metric_name] = metric_info
            
            detailed_results[category] = category_results
        
        # Overall summary
        overall_summary = {
            'total_validations': total_validations,
            'passed': passed_validations,
            'warnings': warning_validations,
            'failed': failed_validations,
            'success_rate': passed_validations / total_validations if total_validations > 0 else 0,
            'overall_status': 'PASS' if failed_validations == 0 else 'FAIL'
        }
        
        report = {
            'summary': overall_summary,
            'categories': detailed_results,
            'recommendations': self._generate_recommendations(detailed_results)
        }
        
        # Log summary
        logger.info(f"\nValidation Summary:")
        logger.info(f"Total validations: {total_validations}")
        logger.info(f"Passed: {passed_validations}")
        logger.info(f"Warnings: {warning_validations}")
        logger.info(f"Failed: {failed_validations}")
        logger.info(f"Success rate: {overall_summary['success_rate']:.1%}")
        
        return report
    
    def _generate_recommendations(self, detailed_results: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for category, results in detailed_results.items():
            failed_count = results['summary']['failed']
            warning_count = results['summary']['warnings']
            
            if failed_count > 0:
                recommendations.append(
                    f"Category '{category}': {failed_count} validations failed. "
                    f"Review force field parameters or simulation settings."
                )
            
            if warning_count > 0:
                recommendations.append(
                    f"Category '{category}': {warning_count} validations showed warnings. "
                    f"Consider extending simulation time or adjusting parameters."
                )
        
        if not recommendations:
            recommendations.append("All validations passed! The simulation shows good agreement with experimental data.")
        
        return recommendations
    
    def save_validation_report(self, output_file: Path):
        """Save validation report to file."""
        report = self.generate_validation_report()
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {output_file}")
    
    def compare_with_literature(self, literature_values: Dict[str, float], 
                              system_type: str) -> Dict[str, float]:
        """Compare simulation results with literature values."""
        logger.info(f"Comparing with literature values for {system_type}")
        
        deviations = {}
        
        for metric, sim_value in literature_values.items():
            # Find corresponding reference value
            for category in self.reference_data.values():
                if system_type in category and metric in category[system_type]:
                    ref_range = category[system_type][metric]
                    ref_center = (ref_range[0] + ref_range[1]) / 2
                    
                    deviation = abs(sim_value - ref_center) / ref_center
                    deviations[metric] = deviation
                    
                    logger.info(f"{metric}: simulation={sim_value:.3f}, "
                              f"reference={ref_center:.3f}, deviation={deviation:.1%}")
        
        return deviations


class BenchmarkComparison:
    """Class for comparing ProteinMD results with other MD software."""
    
    def __init__(self):
        self.benchmark_data = self._load_benchmark_data()
    
    def _load_benchmark_data(self) -> Dict:
        """Load benchmark data from established MD software."""
        return {
            'gromacs': {
                'alanine_dipeptide': {
                    'total_energy': -302.5,  # kJ/mol
                    'temperature': 300.0,    # K
                    'pressure': 1.0,         # bar
                    'density': 1.000,        # g/cm³
                    'rmsd': 0.12,           # nm
                    'radius_of_gyration': 0.65  # nm
                },
                'ubiquitin': {
                    'total_energy': -3250.0,
                    'temperature': 300.0,
                    'pressure': 1.0,
                    'rmsd': 0.18,
                    'radius_of_gyration': 1.20
                }
            },
            'amber': {
                'alanine_dipeptide': {
                    'total_energy': -301.8,
                    'temperature': 300.0,
                    'pressure': 1.0,
                    'rmsd': 0.13,
                    'radius_of_gyration': 0.66
                },
                'ubiquitin': {
                    'total_energy': -3245.0,
                    'temperature': 300.0,
                    'pressure': 1.0,
                    'rmsd': 0.19,
                    'radius_of_gyration': 1.21
                }
            },
            'namd': {
                'alanine_dipeptide': {
                    'total_energy': -303.1,
                    'temperature': 300.0,
                    'pressure': 1.0,
                    'rmsd': 0.11,
                    'radius_of_gyration': 0.64
                }
            }
        }
    
    def compare_with_software(self, proteinmd_results: Dict[str, float], 
                            system_type: str, 
                            reference_software: str = 'gromacs') -> Dict[str, Dict]:
        """Compare ProteinMD results with reference MD software."""
        logger.info(f"Comparing with {reference_software} for {system_type}")
        
        if reference_software not in self.benchmark_data:
            raise ValueError(f"Reference software {reference_software} not available")
        
        if system_type not in self.benchmark_data[reference_software]:
            raise ValueError(f"System type {system_type} not available for {reference_software}")
        
        reference = self.benchmark_data[reference_software][system_type]
        comparison = {}
        
        for metric, proteinmd_value in proteinmd_results.items():
            if metric in reference:
                ref_value = reference[metric]
                
                absolute_diff = proteinmd_value - ref_value
                relative_diff = absolute_diff / ref_value if ref_value != 0 else float('inf')
                
                comparison[metric] = {
                    'proteinmd': proteinmd_value,
                    'reference': ref_value,
                    'absolute_difference': absolute_diff,
                    'relative_difference': relative_diff,
                    'agreement': abs(relative_diff) < 0.05  # 5% threshold
                }
                
                logger.info(f"{metric}: ProteinMD={proteinmd_value:.3f}, "
                          f"{reference_software}={ref_value:.3f}, "
                          f"diff={relative_diff:.1%}")
        
        return comparison
    
    def statistical_significance_test(self, proteinmd_trajectory: np.ndarray, 
                                    reference_trajectory: np.ndarray) -> Dict[str, float]:
        """Test statistical significance of differences."""
        from scipy import stats
        
        # Student's t-test
        t_stat, t_pvalue = stats.ttest_ind(proteinmd_trajectory, reference_trajectory)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(proteinmd_trajectory, reference_trajectory)
        
        # Mann-Whitney U test
        u_stat, u_pvalue = stats.mannwhitneyu(proteinmd_trajectory, reference_trajectory)
        
        return {
            't_test_pvalue': t_pvalue,
            'ks_test_pvalue': ks_pvalue,
            'mannwhitney_pvalue': u_pvalue,
            'statistically_different': min(t_pvalue, ks_pvalue, u_pvalue) < 0.05
        }


def create_validation_test_data():
    """Create mock test data for validation testing."""
    # Mock AMBER ff14SB energy components for alanine dipeptide
    amber_energies = {
        'total_energy': -301.2,    # kJ/mol
        'bond_energy': 245.3,
        'angle_energy': 134.7,
        'dihedral_energy': 15.2,
        'van_der_waals': -10.8,
        'electrostatic': -80.4,
        'rmsd_crystal': 0.15
    }
    
    # Mock water properties for TIP3P
    water_properties = {
        'density_298K': 1.005,        # g/cm³
        'self_diffusion': 2.5e-5,     # cm²/s
        'dielectric_constant': 80.0
    }
    
    # Mock protein structure metrics for ubiquitin
    protein_structure = {
        'radius_of_gyration': 1.21,   # nm
        'rmsd_crystal': 0.18,         # nm
        'alpha_helix_content': 0.25,  # fraction
        'beta_sheet_content': 0.42,   # fraction
        'hydrogen_bonds': 52          # count
    }
    
    return amber_energies, water_properties, protein_structure


def main():
    """Main function to demonstrate validation framework."""
    logger.info("Testing Experimental Data Validation Framework")
    
    # Create validator
    validator = ExperimentalDataValidator()
    
    # Create test data
    amber_energies, water_properties, protein_structure = create_validation_test_data()
    
    # Run validations
    amber_results = validator.validate_amber_ff14sb_energies(amber_energies)
    water_results = validator.validate_water_properties(water_properties)
    structure_results = validator.validate_protein_structure(protein_structure)
    
    # Generate report
    report = validator.generate_validation_report()
    
    logger.info(f"Validation completed with {report['summary']['success_rate']:.1%} success rate")
    
    return report


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
