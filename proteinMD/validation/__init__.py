"""
Validation package for proteinMD.

This package contains modules for validating force field implementations
against reference data and benchmarks.
"""

from .amber_reference_validator import AmberReferenceValidator, create_amber_validator

__all__ = ['AmberReferenceValidator', 'create_amber_validator']
