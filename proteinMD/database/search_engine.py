"""
Search Engine for Simulation Database

This module provides advanced search and filtering capabilities for the
ProteinMD simulation database, enabling users to efficiently find and
retrieve simulation data based on various criteria.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import re
import json

from sqlalchemy import and_, or_, func, text
from sqlalchemy.orm import Session

from .models import SimulationRecord, AnalysisRecord, WorkflowRecord, ProteinStructure
from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class SearchOperator(Enum):
    """Search operators for query construction."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_EQUAL = "ge"
    LESS_THAN = "lt"
    LESS_EQUAL = "le"
    LIKE = "like"
    ILIKE = "ilike"  # case-insensitive like
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    CONTAINS = "contains"  # for JSON fields
    REGEX = "regex"

class SortOrder(Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"

@dataclass
class SearchFilter:
    """
    Individual search filter for query construction.
    
    Represents a single filtering condition that can be applied
    to simulation search queries.
    """
    field: str
    operator: SearchOperator
    value: Any
    case_sensitive: bool = True
    
    def validate(self) -> bool:
        """Validate filter parameters."""
        if not self.field:
            return False
        
        # Check value requirements for different operators
        if self.operator in [SearchOperator.IS_NULL, SearchOperator.IS_NOT_NULL]:
            return True  # No value needed
        
        if self.operator == SearchOperator.BETWEEN:
            return isinstance(self.value, (list, tuple)) and len(self.value) == 2
        
        if self.operator in [SearchOperator.IN, SearchOperator.NOT_IN]:
            return isinstance(self.value, (list, tuple))
        
        return self.value is not None

@dataclass
class SearchQuery:
    """
    Comprehensive search query for simulation database.
    
    Supports complex filtering, sorting, and pagination for
    efficient simulation data retrieval.
    """
    
    # Basic filtering
    filters: List[SearchFilter] = field(default_factory=list)
    text_search: Optional[str] = None  # Full-text search across text fields
    
    # Date range filtering
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    
    # Status and metadata filtering
    status_filter: Optional[List[str]] = None
    user_filter: Optional[List[str]] = None
    project_filter: Optional[List[str]] = None
    tags_filter: Optional[List[str]] = None  # Must contain all tags
    tags_any: Optional[List[str]] = None     # Must contain any tag
    
    # Simulation parameter ranges
    temperature_range: Optional[Tuple[float, float]] = None
    energy_range: Optional[Tuple[float, float]] = None
    time_range: Optional[Tuple[float, float]] = None  # simulation time in ps
    
    # Force field and environment
    force_fields: Optional[List[str]] = None
    water_models: Optional[List[str]] = None
    solvent_types: Optional[List[str]] = None
    
    # Result filtering
    convergence_filter: Optional[bool] = None
    has_analyses: Optional[bool] = None
    analysis_types: Optional[List[str]] = None
    
    # Sorting
    sort_field: str = "created_at"
    sort_order: SortOrder = SortOrder.DESC
    secondary_sort: Optional[Tuple[str, SortOrder]] = None
    
    # Pagination
    limit: int = 100
    offset: int = 0
    
    # Advanced options
    include_analyses: bool = False
    include_workflows: bool = False
    include_protein_info: bool = False
    
    def add_filter(self, field: str, operator: SearchOperator, value: Any) -> None:
        """Add a filter to the query."""
        filter_obj = SearchFilter(field=field, operator=operator, value=value)
        if filter_obj.validate():
            self.filters.append(filter_obj)
        else:
            raise ValueError(f"Invalid filter: {field} {operator} {value}")
    
    def add_text_search(self, text: str) -> None:
        """Add full-text search term."""
        self.text_search = text.strip() if text else None
    
    def add_date_range(self, start_date: Optional[datetime] = None, 
                      end_date: Optional[datetime] = None) -> None:
        """Add date range filter."""
        self.created_after = start_date
        self.created_before = end_date
    
    def set_pagination(self, page: int = 1, page_size: int = 100) -> None:
        """Set pagination parameters."""
        self.limit = max(1, page_size)
        self.offset = max(0, (page - 1) * self.limit)
    
    def validate(self) -> bool:
        """Validate the entire query."""
        # Validate all filters
        for filter_obj in self.filters:
            if not filter_obj.validate():
                return False
        
        # Validate ranges
        if self.temperature_range and len(self.temperature_range) != 2:
            return False
        if self.energy_range and len(self.energy_range) != 2:
            return False
        if self.time_range and len(self.time_range) != 2:
            return False
        
        # Validate pagination
        if self.limit < 1 or self.offset < 0:
            return False
        
        return True

@dataclass
class SearchResult:
    """
    Results from database search operation.
    
    Contains search results along with metadata about the search
    and pagination information.
    """
    
    # Results data
    simulations: List[Dict[str, Any]] = field(default_factory=list)
    total_count: int = 0
    
    # Search metadata
    query_time_ms: float = 0.0
    applied_filters: List[str] = field(default_factory=list)
    
    # Pagination info
    page: int = 1
    page_size: int = 100
    total_pages: int = 0
    has_next: bool = False
    has_previous: bool = False
    
    # Aggregation data (optional)
    aggregations: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate pagination metadata."""
        if self.page_size > 0:
            self.total_pages = (self.total_count + self.page_size - 1) // self.page_size
            self.has_next = self.page < self.total_pages
            self.has_previous = self.page > 1

class SimulationSearchEngine:
    """
    Advanced search engine for ProteinMD simulation database.
    
    Provides sophisticated search, filtering, and aggregation capabilities
    for efficiently querying large simulation datasets.
    """
    
    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize search engine with database manager.
        
        Args:
            database_manager: Database manager instance
        """
        self.db_manager = database_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def search(self, query: SearchQuery) -> SearchResult:
        """
        Execute search query and return results.
        
        Args:
            query: Search query parameters
            
        Returns:
            SearchResult: Search results with metadata
        """
        if not query.validate():
            raise ValueError("Invalid search query")
        
        start_time = datetime.now()
        
        try:
            with self.db_manager.get_session() as session:
                # Build base query
                base_query = self._build_base_query(session, query)
                
                # Get total count (before pagination)
                total_count = self._get_total_count(session, base_query)
                
                # Apply sorting and pagination
                final_query = self._apply_sorting_and_pagination(base_query, query)
                
                # Execute query
                simulations = final_query.all()
                
                # Convert to dictionaries and add related data if requested
                results = []
                for sim in simulations:
                    sim_data = sim.to_dict()
                    
                    if query.include_analyses:
                        sim_data['analyses'] = self._get_simulation_analyses(session, sim.id)
                    
                    if query.include_protein_info and sim.protein_structure:
                        sim_data['protein_structure'] = sim.protein_structure.to_dict()
                    
                    results.append(sim_data)
                
                # Calculate query time
                query_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Build result object
                result = SearchResult(
                    simulations=results,
                    total_count=total_count,
                    query_time_ms=query_time,
                    applied_filters=self._get_applied_filters(query),
                    page=query.offset // query.limit + 1,
                    page_size=query.limit
                )
                
                # Add aggregations if needed
                if len(results) > 0:
                    result.aggregations = self._calculate_aggregations(session, base_query)
                
                self.logger.info(f"Search completed: {total_count} results in {query_time:.2f}ms")
                return result
                
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise
    
    def _build_base_query(self, session: Session, query: SearchQuery):
        """Build base SQLAlchemy query from search parameters."""
        base_query = session.query(SimulationRecord)
        
        # Apply basic filters
        for filter_obj in query.filters:
            condition = self._build_filter_condition(filter_obj)
            if condition is not None:
                base_query = base_query.filter(condition)
        
        # Apply text search
        if query.text_search:
            text_conditions = self._build_text_search_conditions(query.text_search)
            base_query = base_query.filter(or_(*text_conditions))
        
        # Apply date range filters
        if query.created_after:
            base_query = base_query.filter(SimulationRecord.created_at >= query.created_after)
        if query.created_before:
            base_query = base_query.filter(SimulationRecord.created_at <= query.created_before)
        
        # Apply status filter
        if query.status_filter:
            base_query = base_query.filter(SimulationRecord.status.in_(query.status_filter))
        
        # Apply user filter
        if query.user_filter:
            base_query = base_query.filter(SimulationRecord.user_id.in_(query.user_filter))
        
        # Apply project filter
        if query.project_filter:
            base_query = base_query.filter(SimulationRecord.project_name.in_(query.project_filter))
        
        # Apply parameter range filters
        if query.temperature_range:
            min_temp, max_temp = query.temperature_range
            base_query = base_query.filter(
                and_(
                    SimulationRecord.temperature >= min_temp,
                    SimulationRecord.temperature <= max_temp
                )
            )
        
        if query.energy_range:
            min_energy, max_energy = query.energy_range
            base_query = base_query.filter(
                and_(
                    SimulationRecord.final_energy >= min_energy,
                    SimulationRecord.final_energy <= max_energy
                )
            )
        
        if query.time_range:
            min_time, max_time = query.time_range
            base_query = base_query.filter(
                and_(
                    SimulationRecord.total_time >= min_time,
                    SimulationRecord.total_time <= max_time
                )
            )
        
        # Apply force field filter
        if query.force_fields:
            base_query = base_query.filter(SimulationRecord.force_field.in_(query.force_fields))
        
        # Apply water model filter
        if query.water_models:
            base_query = base_query.filter(SimulationRecord.water_model.in_(query.water_models))
        
        # Apply solvent type filter
        if query.solvent_types:
            base_query = base_query.filter(SimulationRecord.solvent_type.in_(query.solvent_types))
        
        # Apply convergence filter
        if query.convergence_filter is not None:
            base_query = base_query.filter(SimulationRecord.convergence_achieved == query.convergence_filter)
        
        # Apply tags filters (requires JSON operations)
        if query.tags_filter:
            for tag in query.tags_filter:
                # This would need database-specific JSON operations
                # For SQLite: JSON_EXTRACT or json_each
                # For PostgreSQL: @> operator
                pass  # Implement based on database type
        
        return base_query
    
    def _build_filter_condition(self, filter_obj: SearchFilter):
        """Build SQLAlchemy condition from search filter."""
        try:
            # Get the field attribute
            field = getattr(SimulationRecord, filter_obj.field, None)
            if field is None:
                self.logger.warning(f"Unknown field: {filter_obj.field}")
                return None
            
            # Build condition based on operator
            if filter_obj.operator == SearchOperator.EQUALS:
                return field == filter_obj.value
            elif filter_obj.operator == SearchOperator.NOT_EQUALS:
                return field != filter_obj.value
            elif filter_obj.operator == SearchOperator.GREATER_THAN:
                return field > filter_obj.value
            elif filter_obj.operator == SearchOperator.GREATER_EQUAL:
                return field >= filter_obj.value
            elif filter_obj.operator == SearchOperator.LESS_THAN:
                return field < filter_obj.value
            elif filter_obj.operator == SearchOperator.LESS_EQUAL:
                return field <= filter_obj.value
            elif filter_obj.operator == SearchOperator.LIKE:
                return field.like(filter_obj.value)
            elif filter_obj.operator == SearchOperator.ILIKE:
                return field.ilike(filter_obj.value)
            elif filter_obj.operator == SearchOperator.IN:
                return field.in_(filter_obj.value)
            elif filter_obj.operator == SearchOperator.NOT_IN:
                return ~field.in_(filter_obj.value)
            elif filter_obj.operator == SearchOperator.BETWEEN:
                min_val, max_val = filter_obj.value
                return and_(field >= min_val, field <= max_val)
            elif filter_obj.operator == SearchOperator.IS_NULL:
                return field.is_(None)
            elif filter_obj.operator == SearchOperator.IS_NOT_NULL:
                return field.isnot(None)
            else:
                self.logger.warning(f"Unsupported operator: {filter_obj.operator}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to build filter condition: {e}")
            return None
    
    def _build_text_search_conditions(self, text: str) -> List:
        """Build text search conditions across multiple fields."""
        text_fields = [
            SimulationRecord.name,
            SimulationRecord.description,
            SimulationRecord.notes,
            SimulationRecord.project_name,
        ]
        
        conditions = []
        search_term = f"%{text}%"
        
        for field in text_fields:
            conditions.append(field.ilike(search_term))
        
        return conditions
    
    def _get_total_count(self, session: Session, query) -> int:
        """Get total count of results without pagination."""
        try:
            return query.count()
        except Exception as e:
            self.logger.error(f"Failed to get total count: {e}")
            return 0
    
    def _apply_sorting_and_pagination(self, query, search_query: SearchQuery):
        """Apply sorting and pagination to query."""
        # Apply primary sorting
        sort_field = getattr(SimulationRecord, search_query.sort_field, None)
        if sort_field:
            if search_query.sort_order == SortOrder.DESC:
                query = query.order_by(sort_field.desc())
            else:
                query = query.order_by(sort_field.asc())
        
        # Apply secondary sorting if specified
        if search_query.secondary_sort:
            secondary_field, secondary_order = search_query.secondary_sort
            field = getattr(SimulationRecord, secondary_field, None)
            if field:
                if secondary_order == SortOrder.DESC:
                    query = query.order_by(field.desc())
                else:
                    query = query.order_by(field.asc())
        
        # Apply pagination
        query = query.limit(search_query.limit).offset(search_query.offset)
        
        return query
    
    def _get_simulation_analyses(self, session: Session, simulation_id: int) -> List[Dict[str, Any]]:
        """Get analyses for a specific simulation."""
        try:
            analyses = session.query(AnalysisRecord).filter_by(simulation_id=simulation_id).all()
            return [analysis.to_dict() for analysis in analyses]
        except Exception as e:
            self.logger.error(f"Failed to get analyses for simulation {simulation_id}: {e}")
            return []
    
    def _get_applied_filters(self, query: SearchQuery) -> List[str]:
        """Get list of applied filter descriptions."""
        filters = []
        
        if query.filters:
            filters.extend([f"{f.field} {f.operator.value} {f.value}" for f in query.filters])
        
        if query.text_search:
            filters.append(f"text: {query.text_search}")
        
        if query.status_filter:
            filters.append(f"status: {query.status_filter}")
        
        if query.created_after or query.created_before:
            date_filter = "date_range"
            if query.created_after:
                date_filter += f" after {query.created_after.isoformat()}"
            if query.created_before:
                date_filter += f" before {query.created_before.isoformat()}"
            filters.append(date_filter)
        
        return filters
    
    def _calculate_aggregations(self, session: Session, query) -> Dict[str, Any]:
        """Calculate aggregation statistics for search results."""
        try:
            # Get basic aggregations
            aggregations = {}
            
            # Status distribution
            status_counts = session.query(
                SimulationRecord.status,
                func.count(SimulationRecord.id)
            ).group_by(SimulationRecord.status).all()
            
            aggregations['status_distribution'] = {status: count for status, count in status_counts}
            
            # Force field distribution
            ff_counts = session.query(
                SimulationRecord.force_field,
                func.count(SimulationRecord.id)
            ).group_by(SimulationRecord.force_field).all()
            
            aggregations['force_field_distribution'] = {ff: count for ff, count in ff_counts if ff}
            
            # Temperature statistics
            temp_stats = session.query(
                func.min(SimulationRecord.temperature),
                func.max(SimulationRecord.temperature),
                func.avg(SimulationRecord.temperature)
            ).first()
            
            if temp_stats and temp_stats[0] is not None:
                aggregations['temperature_stats'] = {
                    'min': temp_stats[0],
                    'max': temp_stats[1],
                    'avg': temp_stats[2]
                }
            
            return aggregations
            
        except Exception as e:
            self.logger.error(f"Failed to calculate aggregations: {e}")
            return {}
    
    def quick_search(self, text: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Quick text-based search for simulations.
        
        Args:
            text: Search text
            limit: Maximum number of results
            
        Returns:
            List[Dict]: List of matching simulations
        """
        query = SearchQuery(
            text_search=text,
            limit=limit,
            sort_field="created_at",
            sort_order=SortOrder.DESC
        )
        
        result = self.search(query)
        return result.simulations
    
    def search_by_status(self, status: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search simulations by status."""
        query = SearchQuery(
            status_filter=[status],
            limit=limit,
            sort_field="created_at",
            sort_order=SortOrder.DESC
        )
        
        result = self.search(query)
        return result.simulations
    
    def search_recent(self, days: int = 7, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for recent simulations."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = SearchQuery(
            created_after=cutoff_date,
            limit=limit,
            sort_field="created_at",
            sort_order=SortOrder.DESC
        )
        
        result = self.search(query)
        return result.simulations
