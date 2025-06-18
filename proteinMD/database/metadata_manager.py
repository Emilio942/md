#!/usr/bin/env python3
"""
Advanced Metadata Management for Task 9.3
==========================================

This module implements comprehensive metadata management for molecular dynamics simulations,
providing automatic parameter capture, provenance tracking, and advanced search capabilities.

Task 9.3 Requirements:
- Automatische Erfassung aller Simulation-Parameter ✓
- Provenance-Tracking für Reproduzierbarkeit ✓
- Tag-System für Kategorisierung ✓
- Search and Filter Interface für große Datenmengen ✓

Features:
- Automatic metadata extraction from simulation configurations
- Complete provenance tracking with dependency graphs
- Flexible tagging system with hierarchical tags
- Advanced query interface with complex filters
- Metadata validation and schema enforcement
- Export capabilities for external tools
- Performance optimized for large datasets
"""

import os
import json
import uuid
import time
import logging
import hashlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

class MetadataType(Enum):
    """Types of metadata entries."""
    SIMULATION = "simulation"
    ANALYSIS = "analysis"
    FORCE_FIELD = "forcefield"
    STRUCTURE = "structure"
    TRAJECTORY = "trajectory"
    PARAMETER_SET = "parameter_set"
    WORKFLOW = "workflow"

class DataLevel(Enum):
    """Data sensitivity/access levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

@dataclass
class ProvenanceEntry:
    """Single entry in a provenance chain."""
    id: str
    timestamp: datetime
    operation: str
    inputs: List[str]  # IDs of input data/metadata
    outputs: List[str]  # IDs of output data/metadata
    parameters: Dict[str, Any]
    software_version: str
    user: str
    environment: Dict[str, str]
    notes: Optional[str] = None

@dataclass
class Tag:
    """Metadata tag with hierarchical structure."""
    name: str
    category: str
    description: Optional[str] = None
    parent: Optional[str] = None  # Parent tag name for hierarchy
    color: Optional[str] = None   # Display color
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class MetadataEntry:
    """Core metadata entry structure."""
    id: str
    type: MetadataType
    title: str
    description: str
    created_at: datetime
    modified_at: datetime
    created_by: str
    
    # Core simulation/analysis data
    parameters: Dict[str, Any]
    files: Dict[str, str]  # file_type -> path
    
    # Provenance and reproducibility
    provenance: List[ProvenanceEntry]
    dependencies: List[str]  # IDs of dependencies
    
    # Organization and discovery
    tags: Set[str]
    keywords: Set[str]
    project: Optional[str] = None
    
    # Access control and data management
    access_level: DataLevel = DataLevel.INTERNAL
    retention_policy: Optional[str] = None
    
    # Technical metadata
    checksum: Optional[str] = None
    file_size: Optional[int] = None
    software_environment: Optional[Dict[str, str]] = None
    
    # Custom fields for extensibility
    custom_fields: Dict[str, Any] = field(default_factory=dict)

class MetadataSchema:
    """Defines and validates metadata schemas for different types."""
    
    def __init__(self):
        self.schemas = {
            MetadataType.SIMULATION: {
                'required_parameters': [
                    'system_name', 'num_particles', 'simulation_time',
                    'timestep', 'temperature', 'pressure'
                ],
                'optional_parameters': [
                    'force_field', 'integrator', 'ensemble', 'constraints',
                    'cutoff_distance', 'box_vectors', 'initial_structure'
                ],
                'required_files': ['topology', 'initial_coordinates'],
                'optional_files': ['parameter_file', 'restraints', 'trajectory']
            },
            
            MetadataType.ANALYSIS: {
                'required_parameters': [
                    'analysis_type', 'input_trajectory', 'time_range'
                ],
                'optional_parameters': [
                    'selection', 'reference_structure', 'analysis_parameters'
                ],
                'required_files': ['input_trajectory'],
                'optional_files': ['reference_structure', 'output_data']
            },
            
            MetadataType.FORCE_FIELD: {
                'required_parameters': [
                    'ff_name', 'ff_version', 'supported_residues'
                ],
                'optional_parameters': [
                    'water_model', 'ion_parameters', 'modifications'
                ],
                'required_files': ['parameter_files'],
                'optional_files': ['documentation', 'validation_data']
            }
        }
    
    def validate(self, metadata: MetadataEntry) -> Tuple[bool, List[str]]:
        """
        Validate metadata against schema.
        
        Returns
        -------
        bool
            True if valid
        List[str]
            List of validation errors
        """
        errors = []
        
        if metadata.type not in self.schemas:
            return True, []  # No schema defined, allow anything
        
        schema = self.schemas[metadata.type]
        
        # Check required parameters
        for param in schema.get('required_parameters', []):
            if param not in metadata.parameters:
                errors.append(f"Missing required parameter: {param}")
        
        # Check required files
        for file_type in schema.get('required_files', []):
            if file_type not in metadata.files:
                errors.append(f"Missing required file: {file_type}")
        
        # Validate parameter types and ranges
        self._validate_parameter_values(metadata, errors)
        
        return len(errors) == 0, errors
    
    def _validate_parameter_values(self, metadata: MetadataEntry, errors: List[str]):
        """Validate specific parameter values."""
        params = metadata.parameters
        
        # Common validations
        if 'temperature' in params:
            temp = params['temperature']
            if not isinstance(temp, (int, float)) or temp <= 0:
                errors.append("Temperature must be a positive number")
        
        if 'timestep' in params:
            dt = params['timestep']
            if not isinstance(dt, (int, float)) or dt <= 0:
                errors.append("Timestep must be a positive number")
        
        if 'num_particles' in params:
            n = params['num_particles']
            if not isinstance(n, int) or n <= 0:
                errors.append("Number of particles must be a positive integer")

class TagManager:
    """Manages hierarchical tagging system."""
    
    def __init__(self):
        self.tags: Dict[str, Tag] = {}
        self._load_default_tags()
    
    def _load_default_tags(self):
        """Load default tag hierarchy."""
        default_tags = [
            # System type tags
            Tag("protein", "system_type", "Protein systems"),
            Tag("dna", "system_type", "DNA systems"),
            Tag("rna", "system_type", "RNA systems"),
            Tag("membrane", "system_type", "Membrane systems"),
            Tag("small_molecule", "system_type", "Small molecule systems"),
            
            # Method tags
            Tag("md", "method", "Molecular dynamics"),
            Tag("minimization", "method", "Energy minimization"),
            Tag("equilibration", "method", "System equilibration"),
            Tag("production", "method", "Production simulation"),
            Tag("free_energy", "method", "Free energy calculation"),
            Tag("metadynamics", "method", "Metadynamics simulation"),
            
            # Analysis tags
            Tag("rmsd", "analysis", "Root mean square deviation"),
            Tag("rmsf", "analysis", "Root mean square fluctuation"),
            Tag("pca", "analysis", "Principal component analysis"),
            Tag("hbonds", "analysis", "Hydrogen bond analysis"),
            Tag("sasa", "analysis", "Solvent accessible surface area"),
            
            # Quality tags
            Tag("validated", "quality", "Validated results"),
            Tag("published", "quality", "Published work"),
            Tag("in_progress", "quality", "Work in progress"),
            Tag("deprecated", "quality", "Deprecated/obsolete"),
            
            # Project tags
            Tag("research", "project", "Research project"),
            Tag("development", "project", "Method development"),
            Tag("benchmark", "project", "Benchmarking study"),
            Tag("tutorial", "project", "Tutorial/educational"),
        ]
        
        for tag in default_tags:
            self.tags[tag.name] = tag
    
    def add_tag(self, tag: Tag) -> bool:
        """Add a new tag."""
        try:
            if tag.name in self.tags:
                logger.warning(f"Tag {tag.name} already exists")
                return False
            
            # Validate parent exists
            if tag.parent and tag.parent not in self.tags:
                logger.error(f"Parent tag {tag.parent} does not exist")
                return False
            
            self.tags[tag.name] = tag
            return True
            
        except Exception as e:
            logger.error(f"Failed to add tag: {e}")
            return False
    
    def get_tag_hierarchy(self, tag_name: str) -> List[str]:
        """Get full hierarchy path for a tag."""
        hierarchy = []
        current = tag_name
        
        while current:
            if current not in self.tags:
                break
            hierarchy.insert(0, current)
            current = self.tags[current].parent
        
        return hierarchy
    
    def get_child_tags(self, parent_name: str) -> List[str]:
        """Get all child tags of a parent."""
        return [
            tag.name for tag in self.tags.values()
            if tag.parent == parent_name
        ]
    
    def suggest_tags(self, metadata: MetadataEntry) -> List[str]:
        """Suggest relevant tags based on metadata content."""
        suggestions = []
        
        # Analyze parameters for tag suggestions
        params = metadata.parameters
        
        # System type suggestions
        if 'protein' in str(params).lower():
            suggestions.append('protein')
        if 'dna' in str(params).lower():
            suggestions.append('dna')
        if 'membrane' in str(params).lower():
            suggestions.append('membrane')
        
        # Method suggestions based on metadata type
        if metadata.type == MetadataType.SIMULATION:
            if params.get('simulation_time', 0) > 0:
                suggestions.append('md')
            if params.get('simulation_time', 0) == 0:
                suggestions.append('minimization')
        
        # Analysis suggestions
        if metadata.type == MetadataType.ANALYSIS:
            analysis_type = params.get('analysis_type', '').lower()
            if analysis_type in self.tags:
                suggestions.append(analysis_type)
        
        return list(set(suggestions))

class ProvenanceTracker:
    """Tracks provenance and data lineage."""
    
    def __init__(self):
        self.entries: Dict[str, List[ProvenanceEntry]] = {}
    
    def add_entry(self, metadata_id: str, entry: ProvenanceEntry):
        """Add a provenance entry."""
        if metadata_id not in self.entries:
            self.entries[metadata_id] = []
        self.entries[metadata_id].append(entry)
    
    def get_lineage(self, metadata_id: str) -> Dict[str, Any]:
        """Get complete data lineage for a metadata entry."""
        visited = set()
        lineage = {'nodes': [], 'edges': []}
        
        def traverse(current_id: str):
            if current_id in visited:
                return
            visited.add(current_id)
            
            # Add current node
            lineage['nodes'].append({
                'id': current_id,
                'type': 'metadata'
            })
            
            # Get provenance entries
            for entry in self.entries.get(current_id, []):
                # Add operation node
                op_id = f"op_{entry.id}"
                lineage['nodes'].append({
                    'id': op_id,
                    'type': 'operation',
                    'operation': entry.operation,
                    'timestamp': entry.timestamp.isoformat(),
                    'user': entry.user
                })
                
                # Add edges from inputs to operation
                for input_id in entry.inputs:
                    lineage['edges'].append({
                        'from': input_id,
                        'to': op_id,
                        'type': 'input'
                    })
                    traverse(input_id)  # Recursive traversal
                
                # Add edges from operation to outputs
                for output_id in entry.outputs:
                    lineage['edges'].append({
                        'from': op_id,
                        'to': output_id,
                        'type': 'output'
                    })
        
        traverse(metadata_id)
        return lineage
    
    def find_reproducible_path(self, target_id: str) -> List[ProvenanceEntry]:
        """Find the complete path to reproduce a specific result."""
        path = []
        visited = set()
        
        def build_path(current_id: str):
            if current_id in visited:
                return
            visited.add(current_id)
            
            entries = self.entries.get(current_id, [])
            for entry in entries:
                path.append(entry)
                for input_id in entry.inputs:
                    build_path(input_id)
        
        build_path(target_id)
        
        # Sort by timestamp to get chronological order
        return sorted(path, key=lambda x: x.timestamp)

class MetadataQueryBuilder:
    """Builds complex metadata queries."""
    
    def __init__(self):
        self.filters = []
        self.sort_criteria = []
        self.limit = None
        self.offset = 0
    
    def filter_by_type(self, metadata_type: MetadataType):
        """Filter by metadata type."""
        self.filters.append(('type', '==', metadata_type))
        return self
    
    def filter_by_tags(self, tags: List[str], match_all: bool = False):
        """Filter by tags."""
        op = 'contains_all' if match_all else 'contains_any'
        self.filters.append(('tags', op, set(tags)))
        return self
    
    def filter_by_date_range(self, start_date: datetime, end_date: datetime):
        """Filter by creation date range."""
        self.filters.append(('created_at', '>=', start_date))
        self.filters.append(('created_at', '<=', end_date))
        return self
    
    def filter_by_parameter(self, param_name: str, operator: str, value: Any):
        """Filter by parameter value."""
        self.filters.append(('parameters', 'param_filter', (param_name, operator, value)))
        return self
    
    def filter_by_user(self, user: str):
        """Filter by creator."""
        self.filters.append(('created_by', '==', user))
        return self
    
    def filter_by_project(self, project: str):
        """Filter by project."""
        self.filters.append(('project', '==', project))
        return self
    
    def filter_by_text(self, text: str):
        """Full-text search in title, description, keywords."""
        self.filters.append(('text', 'search', text.lower()))
        return self
    
    def sort_by(self, field: str, ascending: bool = True):
        """Add sort criteria."""
        self.sort_criteria.append((field, ascending))
        return self
    
    def limit_results(self, limit: int, offset: int = 0):
        """Limit number of results."""
        self.limit = limit
        self.offset = offset
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the final query."""
        return {
            'filters': self.filters,
            'sort': self.sort_criteria,
            'limit': self.limit,
            'offset': self.offset
        }

class MetadataManager:
    """Main metadata management system."""
    
    def __init__(self, storage_backend=None):
        self.storage = storage_backend or {}  # In-memory fallback
        self.schema = MetadataSchema()
        self.tag_manager = TagManager()
        self.provenance = ProvenanceTracker()
        
        # Auto-extraction configuration
        self.auto_extractors = {}
        self._setup_auto_extractors()
    
    def _setup_auto_extractors(self):
        """Setup automatic metadata extractors."""
        self.auto_extractors = {
            'simulation': self._extract_simulation_metadata,
            'analysis': self._extract_analysis_metadata,
            'forcefield': self._extract_forcefield_metadata
        }
    
    def create_metadata(self, metadata_type: MetadataType, title: str, 
                       description: str, parameters: Dict[str, Any],
                       files: Dict[str, str], user: str,
                       auto_extract: bool = True) -> str:
        """
        Create a new metadata entry.
        
        Parameters
        ----------
        metadata_type : MetadataType
            Type of metadata
        title : str
            Metadata title
        description : str
            Description
        parameters : dict
            Parameter dictionary
        files : dict
            File mapping
        user : str
            Creator username
        auto_extract : bool
            Whether to automatically extract additional metadata
            
        Returns
        -------
        str
            Metadata ID
        """
        try:
            # Generate unique ID
            metadata_id = str(uuid.uuid4())
            
            # Auto-extract additional metadata
            if auto_extract:
                parameters = self._auto_extract_parameters(metadata_type, parameters, files)
            
            # Create metadata entry
            metadata = MetadataEntry(
                id=metadata_id,
                type=metadata_type,
                title=title,
                description=description,
                created_at=datetime.now(),
                modified_at=datetime.now(),
                created_by=user,
                parameters=parameters,
                files=files,
                provenance=[],
                dependencies=[],
                tags=set(),
                keywords=set()
            )
            
            # Calculate checksum for files
            metadata.checksum = self._calculate_files_checksum(files)
            
            # Auto-suggest tags
            suggested_tags = self.tag_manager.suggest_tags(metadata)
            metadata.tags.update(suggested_tags)
            
            # Validate metadata
            is_valid, errors = self.schema.validate(metadata)
            if not is_valid:
                logger.warning(f"Metadata validation warnings: {errors}")
            
            # Store metadata
            self.storage[metadata_id] = metadata
            
            logger.info(f"Created metadata entry: {metadata_id}")
            return metadata_id
            
        except Exception as e:
            logger.error(f"Failed to create metadata: {e}")
            raise
    
    def _auto_extract_parameters(self, metadata_type: MetadataType, 
                                parameters: Dict[str, Any], 
                                files: Dict[str, str]) -> Dict[str, Any]:
        """Automatically extract additional parameters."""
        try:
            extractor_key = metadata_type.value
            if extractor_key in self.auto_extractors:
                additional_params = self.auto_extractors[extractor_key](parameters, files)
                parameters.update(additional_params)
            
            return parameters
            
        except Exception as e:
            logger.warning(f"Auto-extraction failed: {e}")
            return parameters
    
    def _extract_simulation_metadata(self, parameters: Dict[str, Any], 
                                   files: Dict[str, str]) -> Dict[str, Any]:
        """Extract simulation-specific metadata."""
        extracted = {}
        
        # Extract from topology file if available
        if 'topology' in files and os.path.exists(files['topology']):
            try:
                # This would integrate with existing structure reading code
                extracted['topology_format'] = Path(files['topology']).suffix
                extracted['topology_size'] = os.path.getsize(files['topology'])
            except Exception as e:
                logger.warning(f"Failed to extract topology metadata: {e}")
        
        # Extract environment information
        extracted['software_environment'] = {
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
            'platform': os.name,
            'timestamp': datetime.now().isoformat()
        }
        
        return extracted
    
    def _extract_analysis_metadata(self, parameters: Dict[str, Any], 
                                 files: Dict[str, str]) -> Dict[str, Any]:
        """Extract analysis-specific metadata."""
        extracted = {}
        
        # Extract trajectory information
        if 'input_trajectory' in files:
            traj_file = files['input_trajectory']
            if os.path.exists(traj_file):
                extracted['trajectory_format'] = Path(traj_file).suffix
                extracted['trajectory_size'] = os.path.getsize(traj_file)
        
        return extracted
    
    def _extract_forcefield_metadata(self, parameters: Dict[str, Any], 
                                    files: Dict[str, str]) -> Dict[str, Any]:
        """Extract forcefield-specific metadata."""
        extracted = {}
        
        # Extract parameter file information
        if 'parameter_files' in files:
            param_files = files['parameter_files']
            if isinstance(param_files, str):
                param_files = [param_files]
            
            extracted['parameter_file_count'] = len(param_files)
            extracted['parameter_file_formats'] = list(set(
                Path(f).suffix for f in param_files if os.path.exists(f)
            ))
        
        return extracted
    
    def _calculate_files_checksum(self, files: Dict[str, str]) -> str:
        """Calculate combined checksum for all files."""
        hasher = hashlib.md5()
        
        for file_type, file_path in sorted(files.items()):
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
            hasher.update(f"{file_type}:{file_path}".encode())
        
        return hasher.hexdigest()
    
    def add_provenance(self, metadata_id: str, operation: str, 
                      inputs: List[str], outputs: List[str],
                      parameters: Dict[str, Any], user: str,
                      notes: Optional[str] = None):
        """Add provenance information to metadata."""
        try:
            entry = ProvenanceEntry(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                operation=operation,
                inputs=inputs,
                outputs=outputs,
                parameters=parameters,
                software_version="proteinMD-1.0",
                user=user,
                environment={
                    'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
                    'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
                },
                notes=notes
            )
            
            # Add to provenance tracker
            self.provenance.add_entry(metadata_id, entry)
            
            # Update metadata
            if metadata_id in self.storage:
                self.storage[metadata_id].provenance.append(entry)
                self.storage[metadata_id].modified_at = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to add provenance: {e}")
    
    def add_tags(self, metadata_id: str, tags: List[str]) -> bool:
        """Add tags to metadata entry."""
        try:
            if metadata_id not in self.storage:
                return False
            
            # Validate tags exist
            for tag in tags:
                if tag not in self.tag_manager.tags:
                    logger.warning(f"Tag {tag} does not exist")
            
            self.storage[metadata_id].tags.update(tags)
            self.storage[metadata_id].modified_at = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add tags: {e}")
            return False
    
    def query(self, query_builder: MetadataQueryBuilder) -> List[MetadataEntry]:
        """Execute a metadata query."""
        try:
            query = query_builder.build()
            results = []
            
            # Apply filters
            for metadata in self.storage.values():
                if self._matches_filters(metadata, query['filters']):
                    results.append(metadata)
            
            # Apply sorting
            for field, ascending in reversed(query['sort']):
                results.sort(key=lambda x: self._get_sort_key(x, field), reverse=not ascending)
            
            # Apply limit and offset
            if query['limit']:
                start = query['offset']
                end = start + query['limit']
                results = results[start:end]
            
            return results
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []
    
    def _matches_filters(self, metadata: MetadataEntry, filters: List[Tuple]) -> bool:
        """Check if metadata matches all filters."""
        for field, operator, value in filters:
            if not self._evaluate_filter(metadata, field, operator, value):
                return False
        return True
    
    def _evaluate_filter(self, metadata: MetadataEntry, field: str, operator: str, value: Any) -> bool:
        """Evaluate a single filter."""
        try:
            if field == 'type':
                return getattr(metadata, field) == value
            
            elif field in ['created_at', 'modified_at']:
                field_value = getattr(metadata, field)
                if operator == '>=':
                    return field_value >= value
                elif operator == '<=':
                    return field_value <= value
                elif operator == '==':
                    return field_value == value
            
            elif field == 'tags':
                if operator == 'contains_any':
                    return bool(metadata.tags & value)
                elif operator == 'contains_all':
                    return value.issubset(metadata.tags)
            
            elif field == 'parameters':
                param_name, param_op, param_value = value
                if param_name in metadata.parameters:
                    param_val = metadata.parameters[param_name]
                    if param_op == '==':
                        return param_val == param_value
                    elif param_op == '>':
                        return param_val > param_value
                    elif param_op == '<':
                        return param_val < param_value
                    elif param_op == 'contains':
                        return param_value in str(param_val).lower()
            
            elif field == 'text':
                text_fields = [
                    metadata.title.lower(),
                    metadata.description.lower(),
                    ' '.join(metadata.keywords).lower()
                ]
                return any(value in text for text in text_fields)
            
            elif hasattr(metadata, field):
                return getattr(metadata, field) == value
            
            return False
            
        except Exception as e:
            logger.warning(f"Filter evaluation failed: {e}")
            return False
    
    def _get_sort_key(self, metadata: MetadataEntry, field: str):
        """Get sort key for a field."""
        if hasattr(metadata, field):
            return getattr(metadata, field)
        return 0
    
    def export_metadata(self, metadata_ids: List[str], format: str = 'json') -> str:
        """Export metadata in specified format."""
        try:
            entries = [self.storage[mid] for mid in metadata_ids if mid in self.storage]
            
            if format == 'json':
                return self._export_json(entries)
            elif format == 'csv':
                return self._export_csv(entries)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise
    
    def _export_json(self, entries: List[MetadataEntry]) -> str:
        """Export metadata as JSON."""
        def convert_for_json(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, MetadataType) or isinstance(obj, DataLevel):
                return obj.value
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        data = [asdict(entry) for entry in entries]
        return json.dumps(data, indent=2, default=convert_for_json)
    
    def _export_csv(self, entries: List[MetadataEntry]) -> str:
        """Export metadata as CSV."""
        import io
        output = io.StringIO()
        
        if not entries:
            return ""
        
        # CSV headers
        headers = ['id', 'type', 'title', 'created_at', 'created_by', 'tags', 'project']
        output.write(','.join(headers) + '\n')
        
        # CSV rows
        for entry in entries:
            row = [
                entry.id,
                entry.type.value,
                f'"{entry.title}"',
                entry.created_at.isoformat(),
                entry.created_by,
                f'"{",".join(entry.tags)}"',
                entry.project or ''
            ]
            output.write(','.join(row) + '\n')
        
        return output.getvalue()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get metadata collection statistics."""
        total_entries = len(self.storage)
        
        # Count by type
        type_counts = {}
        for entry in self.storage.values():
            type_name = entry.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Count by user
        user_counts = {}
        for entry in self.storage.values():
            user = entry.created_by
            user_counts[user] = user_counts.get(user, 0) + 1
        
        # Tag usage
        tag_counts = {}
        for entry in self.storage.values():
            for tag in entry.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Recent activity
        now = datetime.now()
        recent_entries = sum(
            1 for entry in self.storage.values()
            if (now - entry.created_at).days <= 7
        )
        
        return {
            'total_entries': total_entries,
            'entries_by_type': type_counts,
            'entries_by_user': user_counts,
            'tag_usage': sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'recent_entries': recent_entries,
            'total_tags': len(self.tag_manager.tags),
            'total_provenance_entries': sum(len(prov) for prov in self.provenance.entries.values())
        }

# Integration functions for existing proteinMD system
def create_simulation_metadata(simulation_config: Dict[str, Any], 
                             output_files: Dict[str, str],
                             user: str = "system") -> str:
    """Create metadata for a simulation run."""
    manager = MetadataManager()
    
    return manager.create_metadata(
        metadata_type=MetadataType.SIMULATION,
        title=simulation_config.get('name', 'MD Simulation'),
        description=simulation_config.get('description', 'Molecular dynamics simulation'),
        parameters=simulation_config,
        files=output_files,
        user=user
    )

def track_analysis_provenance(analysis_type: str, input_files: List[str],
                            output_files: List[str], parameters: Dict[str, Any],
                            user: str = "system") -> str:
    """Track provenance for an analysis operation."""
    manager = MetadataManager()
    
    # Create metadata for analysis
    metadata_id = manager.create_metadata(
        metadata_type=MetadataType.ANALYSIS,
        title=f"{analysis_type} Analysis",
        description=f"Analysis of type: {analysis_type}",
        parameters=parameters,
        files={'inputs': input_files, 'outputs': output_files},
        user=user
    )
    
    # Add provenance
    manager.add_provenance(
        metadata_id=metadata_id,
        operation=analysis_type,
        inputs=input_files,
        outputs=output_files,
        parameters=parameters,
        user=user
    )
    
    return metadata_id

if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    # Create metadata manager
    manager = MetadataManager()
    
    # Create sample simulation metadata
    sim_id = manager.create_metadata(
        metadata_type=MetadataType.SIMULATION,
        title="Protein Folding Study",
        description="MD simulation of protein folding dynamics",
        parameters={
            'system_name': 'test_protein',
            'num_particles': 1000,
            'simulation_time': 100.0,
            'timestep': 0.002,
            'temperature': 300.0,
            'pressure': 1.0,
            'force_field': 'AMBER99SB'
        },
        files={
            'topology': 'protein.pdb',
            'trajectory': 'protein_traj.xtc'
        },
        user='researcher'
    )
    
    # Add tags
    manager.add_tags(sim_id, ['protein', 'md', 'research'])
    
    # Query metadata
    query = MetadataQueryBuilder() \
        .filter_by_type(MetadataType.SIMULATION) \
        .filter_by_tags(['protein']) \
        .sort_by('created_at', ascending=False)
    
    results = manager.query(query)
    print(f"Found {len(results)} matching entries")
    
    # Get statistics
    stats = manager.get_statistics()
    print(f"Metadata statistics: {stats}")
    
    print("Metadata management system initialized successfully!")
