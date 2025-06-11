#!/usr/bin/env python3
"""
Scientific Validation Framework for ProteinMD

Task 10.4: Validation Studies 🚀 - Enhanced Scientific Validation Component

This module extends the existing experimental data validator with comprehensive
scientific validation capabilities for external peer review and publication.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Import existing validation infrastructure
try:
    from proteinMD.validation.experimental_data_validator import ExperimentalDataValidator, ValidationMetric
    from proteinMD.validation.literature_reproduction_validator import LiteratureReproductionValidator
    from proteinMD.utils.benchmark_comparison import BenchmarkAnalyzer, BenchmarkResult
except ImportError:
    # Mock imports for standalone testing
    class ValidationMetric:
        pass
    class ExperimentalDataValidator:
        pass
    class LiteratureReproductionValidator:
        pass
    class BenchmarkAnalyzer:
        pass
    class BenchmarkResult:
        pass

logger = logging.getLogger(__name__)

@dataclass
class PeerReviewCriteria:
    """Criteria for peer review evaluation."""
    accuracy_threshold: float = 0.05  # 5% accuracy requirement
    reproducibility_threshold: float = 0.03  # 3% reproducibility requirement
    performance_threshold: float = 0.5  # 50% of reference software performance
    statistical_significance: float = 0.05  # p-value threshold
    literature_agreement: float = 0.10  # 10% literature agreement threshold
    minimum_validation_studies: int = 5  # Minimum number of validation studies

@dataclass
class ValidationStudy:
    """Container for individual validation study results."""
    study_id: str
    study_type: str  # 'experimental', 'literature', 'benchmark', 'cross_validation'
    system_description: str
    validation_metrics: Dict[str, float]
    statistical_analysis: Dict[str, Any]
    quality_score: float
    passed: bool
    notes: str
    timestamp: str

@dataclass
class ScientificValidationReport:
    """Comprehensive scientific validation report."""
    proteinmd_version: str
    validation_date: str
    validation_studies: List[ValidationStudy]
    overall_statistics: Dict[str, Any]
    peer_review_assessment: Dict[str, Any]
    publication_readiness: str
    recommendations: List[str]
    external_reviewer_comments: List[str]

class ScientificValidationFramework:
    """Comprehensive scientific validation framework for ProteinMD."""
    
    def __init__(self, peer_review_criteria: Optional[PeerReviewCriteria] = None):
        self.criteria = peer_review_criteria or PeerReviewCriteria()
        self.validation_studies = []
        self.external_validators = {}
        
        # Initialize component validators
        try:
            self.experimental_validator = ExperimentalDataValidator()
            self.literature_validator = LiteratureReproductionValidator()
            self.benchmark_analyzer = BenchmarkAnalyzer()
        except Exception as e:
            logger.warning(f"Could not initialize validators: {e}")
            self.experimental_validator = None
            self.literature_validator = None
            self.benchmark_analyzer = None
    
    def add_experimental_validation_study(self, system_type: str, 
                                        experimental_data: Dict[str, float],
                                        simulation_results: Dict[str, float]) -> ValidationStudy:
        """Add experimental validation study to the framework."""
        study_id = f"exp_{system_type}_{len(self.validation_studies)}"
        
        # Calculate validation metrics
        validation_metrics = {}
        for metric, sim_value in simulation_results.items():
            if metric in experimental_data:
                exp_value = experimental_data[metric]
                if exp_value != 0:
                    relative_error = abs(sim_value - exp_value) / abs(exp_value)
                    validation_metrics[f'{metric}_relative_error'] = relative_error
                    validation_metrics[f'{metric}_absolute_error'] = abs(sim_value - exp_value)
                    validation_metrics[f'{metric}_agreement'] = 1.0 - min(relative_error, 1.0)
        
        # Statistical analysis
        statistical_analysis = self._perform_experimental_statistical_analysis(
            experimental_data, simulation_results
        )
        
        # Quality assessment
        relative_errors = [v for k, v in validation_metrics.items() if 'relative_error' in k]
        quality_score = 1.0 - np.mean(relative_errors) if relative_errors else 0.0
        passed = quality_score > (1.0 - self.criteria.accuracy_threshold)
        
        # Generate notes
        notes = self._generate_experimental_validation_notes(
            validation_metrics, statistical_analysis, passed
        )
        
        study = ValidationStudy(
            study_id=study_id,
            study_type='experimental',
            system_description=f"Experimental validation of {system_type}",
            validation_metrics=validation_metrics,
            statistical_analysis=statistical_analysis,
            quality_score=quality_score,
            passed=passed,
            notes=notes,
            timestamp=datetime.now().isoformat()
        )
        
        self.validation_studies.append(study)
        logger.info(f"Added experimental validation study: {study_id} (Passed: {passed})")
        
        return study
    
    def add_literature_reproduction_study(self, study_key: str,
                                        literature_results: Dict[str, float],
                                        proteinmd_results: Dict[str, float]) -> ValidationStudy:
        """Add literature reproduction study to the framework."""
        study_id = f"lit_{study_key}_{len(self.validation_studies)}"
        
        # Calculate reproduction metrics
        validation_metrics = {}
        for metric, proteinmd_value in proteinmd_results.items():
            if metric in literature_results:
                lit_value = literature_results[metric]
                if lit_value != 0:
                    relative_diff = abs(proteinmd_value - lit_value) / abs(lit_value)
                    validation_metrics[f'{metric}_reproduction_error'] = relative_diff
                    validation_metrics[f'{metric}_reproduction_quality'] = 1.0 - min(relative_diff, 1.0)
        
        # Statistical analysis
        statistical_analysis = self._perform_literature_statistical_analysis(
            literature_results, proteinmd_results
        )
        
        # Quality assessment
        reproduction_errors = [v for k, v in validation_metrics.items() if 'reproduction_error' in k]
        quality_score = 1.0 - np.mean(reproduction_errors) if reproduction_errors else 0.0
        passed = quality_score > (1.0 - self.criteria.literature_agreement)
        
        # Generate notes
        notes = self._generate_literature_validation_notes(
            validation_metrics, statistical_analysis, passed
        )
        
        study = ValidationStudy(
            study_id=study_id,
            study_type='literature',
            system_description=f"Literature reproduction of {study_key}",
            validation_metrics=validation_metrics,
            statistical_analysis=statistical_analysis,
            quality_score=quality_score,
            passed=passed,
            notes=notes,
            timestamp=datetime.now().isoformat()
        )
        
        self.validation_studies.append(study)
        logger.info(f"Added literature reproduction study: {study_id} (Passed: {passed})")
        
        return study
    
    def add_benchmark_validation_study(self, system_type: str,
                                     proteinmd_benchmark: BenchmarkResult,
                                     reference_benchmarks: List[BenchmarkResult]) -> ValidationStudy:
        """Add benchmark validation study to the framework."""
        study_id = f"bench_{system_type}_{len(self.validation_studies)}"
        
        # Calculate benchmark metrics
        validation_metrics = {}
        
        # Performance comparison
        ref_performances = [bench.performance for bench in reference_benchmarks]
        avg_ref_performance = np.mean(ref_performances)
        
        if avg_ref_performance > 0:
            performance_ratio = proteinmd_benchmark.performance / avg_ref_performance
            validation_metrics['performance_ratio'] = performance_ratio
            validation_metrics['performance_ranking'] = self._calculate_performance_ranking(
                proteinmd_benchmark.performance, ref_performances
            )
        
        # Energy accuracy comparison
        ref_energies = [bench.energy for bench in reference_benchmarks]
        avg_ref_energy = np.mean(ref_energies)
        
        if avg_ref_energy != 0:
            energy_error = abs(proteinmd_benchmark.energy - avg_ref_energy) / abs(avg_ref_energy)
            validation_metrics['energy_accuracy'] = 1.0 - min(energy_error, 1.0)
            validation_metrics['energy_relative_error'] = energy_error
        
        # Temperature and pressure stability
        ref_temperatures = [bench.temperature for bench in reference_benchmarks]
        ref_pressures = [bench.pressure for bench in reference_benchmarks]
        
        if ref_temperatures:
            temp_error = abs(proteinmd_benchmark.temperature - np.mean(ref_temperatures))
            validation_metrics['temperature_stability'] = max(0, 5.0 - temp_error) / 5.0
        
        if ref_pressures:
            pressure_error = abs(proteinmd_benchmark.pressure - np.mean(ref_pressures))
            validation_metrics['pressure_stability'] = max(0, 1.0 - pressure_error) / 1.0
        
        # Statistical analysis
        statistical_analysis = self._perform_benchmark_statistical_analysis(
            proteinmd_benchmark, reference_benchmarks
        )
        
        # Quality assessment
        performance_passed = validation_metrics.get('performance_ratio', 0) >= self.criteria.performance_threshold
        accuracy_passed = validation_metrics.get('energy_accuracy', 0) >= (1.0 - self.criteria.accuracy_threshold)
        
        quality_score = np.mean([
            validation_metrics.get('performance_ratio', 0),
            validation_metrics.get('energy_accuracy', 0),
            validation_metrics.get('temperature_stability', 0),
            validation_metrics.get('pressure_stability', 0)
        ])
        
        passed = performance_passed and accuracy_passed
        
        # Generate notes
        notes = self._generate_benchmark_validation_notes(
            validation_metrics, statistical_analysis, passed
        )
        
        study = ValidationStudy(
            study_id=study_id,
            study_type='benchmark',
            system_description=f"Benchmark validation of {system_type}",
            validation_metrics=validation_metrics,
            statistical_analysis=statistical_analysis,
            quality_score=quality_score,
            passed=passed,
            notes=notes,
            timestamp=datetime.now().isoformat()
        )
        
        self.validation_studies.append(study)
        logger.info(f"Added benchmark validation study: {study_id} (Passed: {passed})")
        
        return study
    
    def add_cross_validation_study(self, system_type: str,
                                 proteinmd_replicas: List[Dict[str, float]]) -> ValidationStudy:
        """Add cross-validation study to assess reproducibility."""
        study_id = f"cv_{system_type}_{len(self.validation_studies)}"
        
        if len(proteinmd_replicas) < 3:
            raise ValueError("Cross-validation requires at least 3 independent replicas")
        
        # Calculate reproducibility metrics
        validation_metrics = {}
        
        # Convert to arrays for analysis
        metrics_data = {}
        for replica in proteinmd_replicas:
            for metric, value in replica.items():
                if metric not in metrics_data:
                    metrics_data[metric] = []
                metrics_data[metric].append(value)
        
        # Calculate coefficient of variation for each metric
        for metric, values in metrics_data.items():
            values_array = np.array(values)
            mean_val = np.mean(values_array)
            std_val = np.std(values_array)
            
            if mean_val != 0:
                cv = std_val / abs(mean_val)
                validation_metrics[f'{metric}_coefficient_of_variation'] = cv
                validation_metrics[f'{metric}_reproducibility'] = max(0, 1.0 - cv / 0.1)  # 10% CV threshold
                validation_metrics[f'{metric}_mean'] = mean_val
                validation_metrics[f'{metric}_std'] = std_val
        
        # Overall reproducibility score
        cv_values = [v for k, v in validation_metrics.items() if 'coefficient_of_variation' in k]
        avg_cv = np.mean(cv_values) if cv_values else 1.0
        
        validation_metrics['overall_reproducibility'] = max(0, 1.0 - avg_cv / 0.1)
        validation_metrics['average_cv'] = avg_cv
        
        # Statistical analysis
        statistical_analysis = self._perform_cross_validation_statistical_analysis(proteinmd_replicas)
        
        # Quality assessment
        quality_score = validation_metrics['overall_reproducibility']
        passed = avg_cv <= self.criteria.reproducibility_threshold
        
        # Generate notes
        notes = self._generate_cross_validation_notes(
            validation_metrics, statistical_analysis, passed
        )
        
        study = ValidationStudy(
            study_id=study_id,
            study_type='cross_validation',
            system_description=f"Cross-validation reproducibility of {system_type}",
            validation_metrics=validation_metrics,
            statistical_analysis=statistical_analysis,
            quality_score=quality_score,
            passed=passed,
            notes=notes,
            timestamp=datetime.now().isoformat()
        )
        
        self.validation_studies.append(study)
        logger.info(f"Added cross-validation study: {study_id} (Passed: {passed})")
        
        return study
    
    def _perform_experimental_statistical_analysis(self, experimental_data: Dict[str, float],
                                                 simulation_results: Dict[str, float]) -> Dict[str, Any]:
        """Perform statistical analysis for experimental validation."""
        analysis = {}
        
        exp_values = []
        sim_values = []
        
        for metric in experimental_data.keys():
            if metric in simulation_results:
                exp_values.append(experimental_data[metric])
                sim_values.append(simulation_results[metric])
        
        if len(exp_values) >= 3:
            exp_array = np.array(exp_values)
            sim_array = np.array(sim_values)
            
            # Correlation analysis
            correlation, p_value = stats.pearsonr(exp_array, sim_array)
            analysis['correlation_coefficient'] = correlation
            analysis['correlation_p_value'] = p_value
            analysis['correlation_significant'] = p_value < self.criteria.statistical_significance
            
            # Paired t-test
            t_stat, t_p = stats.ttest_rel(sim_array, exp_array)
            analysis['paired_t_statistic'] = t_stat
            analysis['paired_t_p_value'] = t_p
            analysis['means_significantly_different'] = t_p < self.criteria.statistical_significance
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(exp_array) + np.var(sim_array)) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(sim_array) - np.mean(exp_array)) / pooled_std
                analysis['cohens_d'] = cohens_d
                analysis['effect_size'] = self._interpret_effect_size(abs(cohens_d))
        
        return analysis
    
    def _perform_literature_statistical_analysis(self, literature_results: Dict[str, float],
                                                proteinmd_results: Dict[str, float]) -> Dict[str, Any]:
        """Perform statistical analysis for literature reproduction."""
        analysis = {}
        
        lit_values = []
        proteinmd_values = []
        
        for metric in literature_results.keys():
            if metric in proteinmd_results:
                lit_values.append(literature_results[metric])
                proteinmd_values.append(proteinmd_results[metric])
        
        if len(lit_values) >= 3:
            lit_array = np.array(lit_values)
            proteinmd_array = np.array(proteinmd_values)
            
            # Correlation analysis
            correlation, p_value = stats.pearsonr(lit_array, proteinmd_array)
            analysis['correlation_coefficient'] = correlation
            analysis['correlation_p_value'] = p_value
            
            # Linear regression
            slope, intercept, r_value, reg_p, std_err = stats.linregress(lit_array, proteinmd_array)
            analysis['regression_slope'] = slope
            analysis['regression_intercept'] = intercept
            analysis['regression_r_squared'] = r_value**2
            analysis['regression_p_value'] = reg_p
            
            # Ideal slope should be 1.0 for perfect reproduction
            analysis['slope_deviation_from_ideal'] = abs(slope - 1.0)
            analysis['reproduction_fidelity'] = max(0, 1.0 - abs(slope - 1.0))
        
        return analysis
    
    def _perform_benchmark_statistical_analysis(self, proteinmd_benchmark: BenchmarkResult,
                                              reference_benchmarks: List[BenchmarkResult]) -> Dict[str, Any]:
        """Perform statistical analysis for benchmark validation."""
        analysis = {}
        
        # Performance distribution analysis
        ref_performances = [bench.performance for bench in reference_benchmarks]
        ref_energies = [bench.energy for bench in reference_benchmarks]
        
        if len(ref_performances) >= 3:
            # Performance percentile
            performance_percentile = stats.percentileofscore(ref_performances, proteinmd_benchmark.performance)
            analysis['performance_percentile'] = performance_percentile
            
            # Z-score for performance
            perf_mean = np.mean(ref_performances)
            perf_std = np.std(ref_performances)
            if perf_std > 0:
                performance_z_score = (proteinmd_benchmark.performance - perf_mean) / perf_std
                analysis['performance_z_score'] = performance_z_score
                analysis['performance_outlier'] = abs(performance_z_score) > 2.0
        
        if len(ref_energies) >= 3:
            # Energy accuracy analysis
            energy_percentile = stats.percentileofscore(ref_energies, proteinmd_benchmark.energy)
            analysis['energy_percentile'] = energy_percentile
            
            # Energy Z-score
            energy_mean = np.mean(ref_energies)
            energy_std = np.std(ref_energies)
            if energy_std > 0:
                energy_z_score = (proteinmd_benchmark.energy - energy_mean) / energy_std
                analysis['energy_z_score'] = energy_z_score
                analysis['energy_outlier'] = abs(energy_z_score) > 2.0
        
        return analysis
    
    def _perform_cross_validation_statistical_analysis(self, replicas: List[Dict[str, float]]) -> Dict[str, Any]:
        """Perform statistical analysis for cross-validation."""
        analysis = {}
        
        # Collect data for each metric across replicas
        metrics_data = {}
        for replica in replicas:
            for metric, value in replica.items():
                if metric not in metrics_data:
                    metrics_data[metric] = []
                metrics_data[metric].append(value)
        
        # Perform statistical tests for each metric
        for metric, values in metrics_data.items():
            if len(values) >= 3:
                values_array = np.array(values)
                
                # Normality test (Shapiro-Wilk)
                shapiro_stat, shapiro_p = stats.shapiro(values_array)
                analysis[f'{metric}_normality_p_value'] = shapiro_p
                analysis[f'{metric}_is_normal'] = shapiro_p > 0.05
                
                # Confidence interval for mean
                sem = stats.sem(values_array)
                confidence_interval = stats.t.interval(0.95, len(values_array)-1, 
                                                     loc=np.mean(values_array), scale=sem)
                analysis[f'{metric}_95_confidence_interval'] = confidence_interval
                
                # Outlier detection (IQR method)
                q1 = np.percentile(values_array, 25)
                q3 = np.percentile(values_array, 75)
                iqr = q3 - q1
                outlier_threshold_low = q1 - 1.5 * iqr
                outlier_threshold_high = q3 + 1.5 * iqr
                
                outliers = values_array[(values_array < outlier_threshold_low) | 
                                      (values_array > outlier_threshold_high)]
                analysis[f'{metric}_outlier_count'] = len(outliers)
                analysis[f'{metric}_outlier_fraction'] = len(outliers) / len(values_array)
        
        return analysis
    
    def _calculate_performance_ranking(self, proteinmd_performance: float,
                                     reference_performances: List[float]) -> float:
        """Calculate performance ranking (0-1 scale, 1 = best)."""
        all_performances = reference_performances + [proteinmd_performance]
        sorted_performances = sorted(all_performances, reverse=True)
        
        proteinmd_rank = sorted_performances.index(proteinmd_performance) + 1
        total_entries = len(sorted_performances)
        
        # Convert to 0-1 scale (1 = best, 0 = worst)
        ranking = 1.0 - (proteinmd_rank - 1) / (total_entries - 1)
        return ranking
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _generate_experimental_validation_notes(self, validation_metrics: Dict[str, float],
                                              statistical_analysis: Dict[str, Any],
                                              passed: bool) -> str:
        """Generate notes for experimental validation study."""
        notes = []
        
        if passed:
            notes.append("✅ PASSED: Experimental validation criteria met")
        else:
            notes.append("❌ FAILED: Experimental validation criteria not met")
        
        # Accuracy summary
        relative_errors = [v for k, v in validation_metrics.items() if 'relative_error' in k]
        if relative_errors:
            avg_error = np.mean(relative_errors) * 100
            notes.append(f"Average relative error: {avg_error:.1f}%")
        
        # Statistical significance
        if 'correlation_significant' in statistical_analysis:
            if statistical_analysis['correlation_significant']:
                corr = statistical_analysis['correlation_coefficient']
                notes.append(f"Significant correlation with experimental data (r={corr:.3f})")
            else:
                notes.append("No significant correlation with experimental data")
        
        # Effect size interpretation
        if 'effect_size' in statistical_analysis:
            effect = statistical_analysis['effect_size']
            notes.append(f"Effect size: {effect}")
        
        return "\n".join(notes)
    
    def _generate_literature_validation_notes(self, validation_metrics: Dict[str, float],
                                            statistical_analysis: Dict[str, Any],
                                            passed: bool) -> str:
        """Generate notes for literature reproduction study."""
        notes = []
        
        if passed:
            notes.append("✅ PASSED: Literature reproduction criteria met")
        else:
            notes.append("❌ FAILED: Literature reproduction criteria not met")
        
        # Reproduction quality
        reproduction_errors = [v for k, v in validation_metrics.items() if 'reproduction_error' in k]
        if reproduction_errors:
            avg_error = np.mean(reproduction_errors) * 100
            notes.append(f"Average reproduction error: {avg_error:.1f}%")
        
        # Regression analysis
        if 'regression_r_squared' in statistical_analysis:
            r_squared = statistical_analysis['regression_r_squared']
            notes.append(f"Regression R²: {r_squared:.3f}")
        
        if 'reproduction_fidelity' in statistical_analysis:
            fidelity = statistical_analysis['reproduction_fidelity']
            notes.append(f"Reproduction fidelity: {fidelity:.1%}")
        
        return "\n".join(notes)
    
    def _generate_benchmark_validation_notes(self, validation_metrics: Dict[str, float],
                                           statistical_analysis: Dict[str, Any],
                                           passed: bool) -> str:
        """Generate notes for benchmark validation study."""
        notes = []
        
        if passed:
            notes.append("✅ PASSED: Benchmark validation criteria met")
        else:
            notes.append("❌ FAILED: Benchmark validation criteria not met")
        
        # Performance comparison
        if 'performance_ratio' in validation_metrics:
            perf_ratio = validation_metrics['performance_ratio']
            notes.append(f"Performance ratio vs references: {perf_ratio:.2f}x")
        
        if 'performance_ranking' in validation_metrics:
            ranking = validation_metrics['performance_ranking']
            notes.append(f"Performance ranking: {ranking:.1%} percentile")
        
        # Energy accuracy
        if 'energy_accuracy' in validation_metrics:
            accuracy = validation_metrics['energy_accuracy']
            notes.append(f"Energy accuracy: {accuracy:.1%}")
        
        return "\n".join(notes)
    
    def _generate_cross_validation_notes(self, validation_metrics: Dict[str, float],
                                       statistical_analysis: Dict[str, Any],
                                       passed: bool) -> str:
        """Generate notes for cross-validation study."""
        notes = []
        
        if passed:
            notes.append("✅ PASSED: Cross-validation reproducibility criteria met")
        else:
            notes.append("❌ FAILED: Cross-validation reproducibility criteria not met")
        
        # Reproducibility metrics
        if 'overall_reproducibility' in validation_metrics:
            reproducibility = validation_metrics['overall_reproducibility']
            notes.append(f"Overall reproducibility: {reproducibility:.1%}")
        
        if 'average_cv' in validation_metrics:
            avg_cv = validation_metrics['average_cv']
            notes.append(f"Average coefficient of variation: {avg_cv:.1%}")
        
        # Outlier detection
        outlier_counts = [v for k, v in statistical_analysis.items() if 'outlier_count' in k]
        if outlier_counts:
            total_outliers = sum(outlier_counts)
            notes.append(f"Total outliers detected: {total_outliers}")
        
        return "\n".join(notes)
    
    def conduct_peer_review_assessment(self) -> Dict[str, Any]:
        """Conduct peer review assessment based on all validation studies."""
        if len(self.validation_studies) < self.criteria.minimum_validation_studies:
            return {
                'status': 'insufficient_studies',
                'message': f"Need at least {self.criteria.minimum_validation_studies} validation studies",
                'current_studies': len(self.validation_studies)
            }
        
        assessment = {
            'total_studies': len(self.validation_studies),
            'passed_studies': sum(1 for study in self.validation_studies if study.passed),
            'pass_rate': 0.0,
            'study_type_breakdown': {},
            'quality_distribution': {},
            'overall_recommendation': '',
            'scientific_rigor': '',
            'publication_readiness': '',
            'detailed_assessment': {}
        }
        
        # Calculate pass rate
        assessment['pass_rate'] = assessment['passed_studies'] / assessment['total_studies']
        
        # Study type breakdown
        for study in self.validation_studies:
            study_type = study.study_type
            if study_type not in assessment['study_type_breakdown']:
                assessment['study_type_breakdown'][study_type] = {'total': 0, 'passed': 0}
            
            assessment['study_type_breakdown'][study_type]['total'] += 1
            if study.passed:
                assessment['study_type_breakdown'][study_type]['passed'] += 1
        
        # Quality distribution
        quality_scores = [study.quality_score for study in self.validation_studies]
        assessment['quality_distribution'] = {
            'mean': np.mean(quality_scores),
            'median': np.median(quality_scores),
            'std': np.std(quality_scores),
            'min': np.min(quality_scores),
            'max': np.max(quality_scores)
        }
        
        # Overall assessment
        if assessment['pass_rate'] >= 0.9:
            assessment['overall_recommendation'] = 'highly_recommended'
            assessment['scientific_rigor'] = 'excellent'
            assessment['publication_readiness'] = 'ready_for_publication'
        elif assessment['pass_rate'] >= 0.8:
            assessment['overall_recommendation'] = 'recommended'
            assessment['scientific_rigor'] = 'good'
            assessment['publication_readiness'] = 'ready_with_minor_revisions'
        elif assessment['pass_rate'] >= 0.7:
            assessment['overall_recommendation'] = 'conditionally_recommended'
            assessment['scientific_rigor'] = 'acceptable'
            assessment['publication_readiness'] = 'requires_revisions'
        else:
            assessment['overall_recommendation'] = 'not_recommended'
            assessment['scientific_rigor'] = 'insufficient'
            assessment['publication_readiness'] = 'major_revisions_required'
        
        # Detailed assessment by category
        assessment['detailed_assessment'] = self._generate_detailed_peer_review_assessment()
        
        return assessment
    
    def _generate_detailed_peer_review_assessment(self) -> Dict[str, Any]:
        """Generate detailed peer review assessment by category."""
        detailed = {
            'accuracy_assessment': {},
            'reproducibility_assessment': {},
            'performance_assessment': {},
            'literature_validation_assessment': {},
            'recommendations': []
        }
        
        # Accuracy assessment
        experimental_studies = [s for s in self.validation_studies if s.study_type == 'experimental']
        if experimental_studies:
            accuracy_scores = []
            for study in experimental_studies:
                relative_errors = [v for k, v in study.validation_metrics.items() if 'relative_error' in k]
                if relative_errors:
                    accuracy_score = 1.0 - np.mean(relative_errors)
                    accuracy_scores.append(accuracy_score)
            
            if accuracy_scores:
                detailed['accuracy_assessment'] = {
                    'mean_accuracy': np.mean(accuracy_scores),
                    'studies_evaluated': len(accuracy_scores),
                    'meets_criteria': np.mean(accuracy_scores) >= (1.0 - self.criteria.accuracy_threshold)
                }
        
        # Reproducibility assessment
        cv_studies = [s for s in self.validation_studies if s.study_type == 'cross_validation']
        if cv_studies:
            reproducibility_scores = []
            for study in cv_studies:
                if 'overall_reproducibility' in study.validation_metrics:
                    reproducibility_scores.append(study.validation_metrics['overall_reproducibility'])
            
            if reproducibility_scores:
                detailed['reproducibility_assessment'] = {
                    'mean_reproducibility': np.mean(reproducibility_scores),
                    'studies_evaluated': len(reproducibility_scores),
                    'meets_criteria': np.mean(reproducibility_scores) >= (1.0 - self.criteria.reproducibility_threshold)
                }
        
        # Performance assessment
        benchmark_studies = [s for s in self.validation_studies if s.study_type == 'benchmark']
        if benchmark_studies:
            performance_ratios = []
            for study in benchmark_studies:
                if 'performance_ratio' in study.validation_metrics:
                    performance_ratios.append(study.validation_metrics['performance_ratio'])
            
            if performance_ratios:
                detailed['performance_assessment'] = {
                    'mean_performance_ratio': np.mean(performance_ratios),
                    'studies_evaluated': len(performance_ratios),
                    'meets_criteria': np.mean(performance_ratios) >= self.criteria.performance_threshold
                }
        
        # Literature validation assessment
        literature_studies = [s for s in self.validation_studies if s.study_type == 'literature']
        if literature_studies:
            reproduction_qualities = []
            for study in literature_studies:
                reproduction_errors = [v for k, v in study.validation_metrics.items() if 'reproduction_error' in k]
                if reproduction_errors:
                    quality = 1.0 - np.mean(reproduction_errors)
                    reproduction_qualities.append(quality)
            
            if reproduction_qualities:
                detailed['literature_validation_assessment'] = {
                    'mean_reproduction_quality': np.mean(reproduction_qualities),
                    'studies_evaluated': len(reproduction_qualities),
                    'meets_criteria': np.mean(reproduction_qualities) >= (1.0 - self.criteria.literature_agreement)
                }
        
        # Generate recommendations
        detailed['recommendations'] = self._generate_peer_review_recommendations(detailed)
        
        return detailed
    
    def _generate_peer_review_recommendations(self, detailed_assessment: Dict[str, Any]) -> List[str]:
        """Generate peer review recommendations based on detailed assessment."""
        recommendations = []
        
        # Accuracy recommendations
        if 'accuracy_assessment' in detailed_assessment:
            acc_assessment = detailed_assessment['accuracy_assessment']
            if not acc_assessment.get('meets_criteria', False):
                recommendations.append(
                    "Improve accuracy validation: Consider force field parameter optimization "
                    "or longer equilibration times to better match experimental data."
                )
        
        # Reproducibility recommendations
        if 'reproducibility_assessment' in detailed_assessment:
            repro_assessment = detailed_assessment['reproducibility_assessment']
            if not repro_assessment.get('meets_criteria', False):
                recommendations.append(
                    "Enhance reproducibility: Implement stricter numerical precision controls "
                    "and ensure proper random seed management across simulations."
                )
        
        # Performance recommendations
        if 'performance_assessment' in detailed_assessment:
            perf_assessment = detailed_assessment['performance_assessment']
            if not perf_assessment.get('meets_criteria', False):
                recommendations.append(
                    "Optimize performance: Consider algorithmic improvements, better compiler "
                    "optimizations, or parallel computing strategies to enhance computational efficiency."
                )
        
        # Literature validation recommendations
        if 'literature_validation_assessment' in detailed_assessment:
            lit_assessment = detailed_assessment['literature_validation_assessment']
            if not lit_assessment.get('meets_criteria', False):
                recommendations.append(
                    "Improve literature reproduction: Verify simulation protocols match published "
                    "conditions and consider implementing advanced sampling methods if needed."
                )
        
        # General recommendations
        if len(self.validation_studies) < 10:
            recommendations.append(
                "Expand validation coverage: Add more validation studies across different "
                "system types and conditions to strengthen scientific credibility."
            )
        
        # If all criteria are met
        if not recommendations:
            recommendations.append(
                "Excellent validation: ProteinMD demonstrates high scientific rigor and "
                "is ready for peer-reviewed publication."
            )
        
        return recommendations
    
    def generate_scientific_validation_report(self, output_file: Path,
                                            proteinmd_version: str = "1.0.0") -> ScientificValidationReport:
        """Generate comprehensive scientific validation report."""
        # Conduct peer review assessment
        peer_review_assessment = self.conduct_peer_review_assessment()
        
        # Calculate overall statistics
        overall_statistics = {
            'total_validation_studies': len(self.validation_studies),
            'pass_rate': peer_review_assessment.get('pass_rate', 0.0),
            'study_types': list(set(study.study_type for study in self.validation_studies)),
            'validation_period': {
                'start': min(study.timestamp for study in self.validation_studies) if self.validation_studies else None,
                'end': max(study.timestamp for study in self.validation_studies) if self.validation_studies else None
            },
            'quality_metrics': peer_review_assessment.get('quality_distribution', {})
        }
        
        # Generate recommendations
        recommendations = peer_review_assessment.get('detailed_assessment', {}).get('recommendations', [])
        
        # Create validation report
        report = ScientificValidationReport(
            proteinmd_version=proteinmd_version,
            validation_date=datetime.now().isoformat(),
            validation_studies=self.validation_studies.copy(),
            overall_statistics=overall_statistics,
            peer_review_assessment=peer_review_assessment,
            publication_readiness=peer_review_assessment.get('publication_readiness', 'unknown'),
            recommendations=recommendations,
            external_reviewer_comments=[]  # To be filled by external reviewers
        )
        
        # Save report to file
        with open(output_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        logger.info(f"Scientific validation report saved to {output_file}")
        return report
    
    def add_external_reviewer_comment(self, reviewer_name: str, comment: str,
                                    expertise_area: str, recommendation: str) -> None:
        """Add external reviewer comment to the validation framework."""
        reviewer_data = {
            'reviewer_name': reviewer_name,
            'expertise_area': expertise_area,
            'recommendation': recommendation,
            'comment': comment,
            'timestamp': datetime.now().isoformat()
        }
        
        self.external_validators[reviewer_name] = reviewer_data
        logger.info(f"Added external reviewer comment from {reviewer_name}")
    
    def export_validation_summary_table(self, output_file: Path) -> None:
        """Export validation summary as CSV table."""
        if not self.validation_studies:
            logger.warning("No validation studies to export")
            return
        
        # Prepare data for CSV
        rows = []
        for study in self.validation_studies:
            row = {
                'Study_ID': study.study_id,
                'Study_Type': study.study_type,
                'System_Description': study.system_description,
                'Quality_Score': study.quality_score,
                'Passed': study.passed,
                'Timestamp': study.timestamp
            }
            
            # Add key validation metrics
            for metric, value in study.validation_metrics.items():
                row[f'Metric_{metric}'] = value
            
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        logger.info(f"Validation summary table exported to {output_file}")

def create_mock_scientific_validation_data():
    """Create mock data for scientific validation testing."""
    
    # Mock experimental data
    experimental_systems = {
        'ubiquitin_folding': {
            'experimental_data': {
                'folding_time': 8.5e-6,  # seconds
                'radius_of_gyration': 1.20,  # nm
                'rmsd_crystal': 0.18,  # nm
                'native_contacts': 0.85,  # fraction
                'folding_temperature': 315.0  # K
            },
            'proteinmd_results': {
                'folding_time': 9.1e-6,  # 7% error
                'radius_of_gyration': 1.23,  # 2.5% error
                'rmsd_crystal': 0.19,  # 5.6% error
                'native_contacts': 0.82,  # 3.5% error
                'folding_temperature': 318.0  # 0.95% error
            }
        },
        'alanine_dipeptide': {
            'experimental_data': {
                'c7ax_population': 0.85,
                'phi_angle_avg': -63.0,
                'psi_angle_avg': 135.0,
                'transition_rate': 4.3e8  # s⁻¹
            },
            'proteinmd_results': {
                'c7ax_population': 0.83,  # 2.4% error
                'phi_angle_avg': -65.0,  # 3.2% error
                'psi_angle_avg': 138.0,  # 2.2% error
                'transition_rate': 4.1e8  # 4.7% error
            }
        }
    }
    
    # Mock literature reproduction data
    literature_systems = {
        'shaw_2010_villin': {
            'literature_results': {
                'folding_time': 4.5e-6,
                'native_contacts_folded': 0.85,
                'radius_of_gyration_folded': 0.85,
                'rmsd_from_nmr': 0.15
            },
            'proteinmd_results': {
                'folding_time': 4.8e-6,  # 6.7% error
                'native_contacts_folded': 0.82,  # 3.5% error
                'radius_of_gyration_folded': 0.87,  # 2.4% error
                'rmsd_from_nmr': 0.16  # 6.7% error
            }
        }
    }
    
    # Mock benchmark data
    benchmark_systems = {
        'protein_folding_20k': {
            'proteinmd_benchmark': BenchmarkResult(
                software='ProteinMD',
                system_size=20000,
                simulation_time=10.0,
                wall_time=129600,  # 36 hours
                performance=6.7,  # ns/day
                energy=-1248.5,
                temperature=300.1,
                pressure=1.0,
                platform='Linux',
                version='1.0.0'
            ),
            'reference_benchmarks': [
                BenchmarkResult(
                    software='GROMACS',
                    system_size=20000,
                    simulation_time=10.0,
                    wall_time=86400,  # 24 hours
                    performance=10.0,  # ns/day
                    energy=-1250.0,
                    temperature=300.0,
                    pressure=1.0,
                    platform='Linux',
                    version='2023.1'
                ),
                BenchmarkResult(
                    software='AMBER',
                    system_size=20000,
                    simulation_time=10.0,
                    wall_time=115200,  # 32 hours
                    performance=7.5,  # ns/day
                    energy=-1249.2,
                    temperature=300.0,
                    pressure=1.0,
                    platform='Linux',
                    version='22'
                )
            ]
        }
    }
    
    # Mock cross-validation data (multiple independent runs)
    cv_systems = {
        'membrane_simulation': [
            {
                'area_per_lipid': 67.2,  # Ų
                'membrane_thickness': 3.8,  # nm
                'order_parameter': 0.21,
                'diffusion_coefficient': 5.2e-8  # cm²/s
            },
            {
                'area_per_lipid': 66.8,  # Ų
                'membrane_thickness': 3.9,  # nm
                'order_parameter': 0.20,
                'diffusion_coefficient': 5.0e-8  # cm²/s
            },
            {
                'area_per_lipid': 67.5,  # Ų
                'membrane_thickness': 3.7,  # nm
                'order_parameter': 0.22,
                'diffusion_coefficient': 5.3e-8  # cm²/s
            },
            {
                'area_per_lipid': 66.9,  # Ų
                'membrane_thickness': 3.8,  # nm
                'order_parameter': 0.21,
                'diffusion_coefficient': 5.1e-8  # cm²/s
            }
        ]
    }
    
    return experimental_systems, literature_systems, benchmark_systems, cv_systems

def main():
    """Main function for testing scientific validation framework."""
    # Initialize framework
    framework = ScientificValidationFramework()
    output_dir = Path("scientific_validation_framework")
    output_dir.mkdir(exist_ok=True)
    
    # Get mock data
    experimental_systems, literature_systems, benchmark_systems, cv_systems = create_mock_scientific_validation_data()
    
    print("🔬 Scientific Validation Framework - Testing")
    print("=" * 60)
    
    # Add experimental validation studies
    for system_name, data in experimental_systems.items():
        study = framework.add_experimental_validation_study(
            system_name, 
            data['experimental_data'], 
            data['proteinmd_results']
        )
        print(f"Added experimental study: {study.study_id} ({'✅ PASS' if study.passed else '❌ FAIL'})")
    
    # Add literature reproduction studies
    for system_name, data in literature_systems.items():
        study = framework.add_literature_reproduction_study(
            system_name,
            data['literature_results'],
            data['proteinmd_results']
        )
        print(f"Added literature study: {study.study_id} ({'✅ PASS' if study.passed else '❌ FAIL'})")
    
    # Add benchmark validation studies
    for system_name, data in benchmark_systems.items():
        study = framework.add_benchmark_validation_study(
            system_name,
            data['proteinmd_benchmark'],
            data['reference_benchmarks']
        )
        print(f"Added benchmark study: {study.study_id} ({'✅ PASS' if study.passed else '❌ FAIL'})")
    
    # Add cross-validation studies
    for system_name, replicas in cv_systems.items():
        study = framework.add_cross_validation_study(system_name, replicas)
        print(f"Added cross-validation study: {study.study_id} ({'✅ PASS' if study.passed else '❌ FAIL'})")
    
    # Conduct peer review assessment
    print(f"\n📊 PEER REVIEW ASSESSMENT:")
    peer_review = framework.conduct_peer_review_assessment()
    print(f"   Total studies: {peer_review['total_studies']}")
    print(f"   Pass rate: {peer_review['pass_rate']:.1%}")
    print(f"   Overall recommendation: {peer_review['overall_recommendation']}")
    print(f"   Publication readiness: {peer_review['publication_readiness']}")
    
    # Add external reviewer comments (mock)
    framework.add_external_reviewer_comment(
        "Dr. Sarah Martinez",
        "The validation methodology is comprehensive and follows best practices. "
        "The literature reproduction studies demonstrate good fidelity to published results.",
        "Computational Biophysics",
        "Accept with minor revisions"
    )
    
    framework.add_external_reviewer_comment(
        "Prof. Michael Chen",
        "Performance benchmarks show competitive results against established software. "
        "Cross-validation studies confirm good reproducibility.",
        "Molecular Dynamics Algorithms",
        "Accept"
    )
    
    # Generate comprehensive report
    report_file = output_dir / "scientific_validation_report.json"
    report = framework.generate_scientific_validation_report(report_file)
    print(f"\n📋 Scientific validation report generated: {report_file}")
    
    # Export summary table
    summary_file = output_dir / "validation_summary.csv"
    framework.export_validation_summary_table(summary_file)
    print(f"📊 Validation summary table exported: {summary_file}")
    
    print(f"\n✅ Scientific validation framework testing complete!")
    print(f"📁 Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
