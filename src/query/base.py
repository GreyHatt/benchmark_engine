from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import time
import logging
from pathlib import Path
import json

from .metrics import MetricsCollector, QueryMetrics, QueryResultValidator

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Container for query execution results and metrics."""
    
    def __init__(self):
        # Basic execution info
        self.execution_time: float = 0.0
        self.rows_returned: int = 0
        self.query_plan: Optional[str] = None
        self.metrics: Dict[str, Any] = {}
        self.success: bool = False
        self.error: Optional[str] = None
        self.result_data: Optional[Any] = None
        
        # For hybrid execution
        self.engine_used: Optional[str] = None
        self.spark: Optional['QueryResult'] = None
        self.duckdb: Optional['QueryResult'] = None
        
        # For result validation
        self.validation_passed: Optional[bool] = None
        self.validation_errors: List[str] = field(default_factory=list)
        
        # System metrics
        self.system_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the query result to a dictionary."""
        result = {
            'execution_time': self.execution_time,
            'rows_returned': self.rows_returned,
            'success': self.success,
            'metrics': self.metrics,
            'validation_passed': self.validation_passed,
            'validation_errors': self.validation_errors,
            'system_metrics': self.system_metrics
        }
        
        if self.error:
            result['error'] = self.error
            
        if self.query_plan:
            result['query_plan'] = self.query_plan
            
        if self.result_data is not None:
            result['has_result_data'] = True
            
        # Handle nested results in hybrid mode
        if self.spark:
            result['spark'] = self.spark.to_dict()
        if self.duckdb:
            result['duckdb'] = self.duckdb.to_dict()
            
        return result
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert the query result to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def validate_results(self, other: 'QueryResult', **kwargs) -> bool:
        """
        Compare this result with another result for validation.
        
        Args:
            other: Another QueryResult to compare against
            **kwargs: Additional arguments to pass to the validator
            
        Returns:
            bool: True if results are equivalent within tolerance
        """
        if not (self.success and other.success):
            self.validation_errors.append("One or both queries failed execution")
            self.validation_passed = False
            return False
            
        # Compare row counts if available
        if hasattr(self, 'rows_returned') and hasattr(other, 'rows_returned'):
            if self.rows_returned != other.rows_returned:
                self.validation_errors.append(
                    f"Row count mismatch: {self.rows_returned} vs {other.rows_returned}"
                )
                self.validation_passed = False
                return False
                
        # If no data to compare, return True
        if not self.result_data or not other.result_data:
            self.validation_passed = True
            return True
            
        # Use the validator to compare results
        validator = QueryResultValidator()
        validation_result = validator.validate_results(
            self.result_data,
            other.result_data,
            **kwargs
        )
        
        self.validation_passed = validation_result['equal']
        if not self.validation_passed:
            self.validation_errors.extend(validation_result.get('errors', []))
            
        # Store validation metrics
        self.metrics['validation'] = {
            'passed': self.validation_passed,
            'errors': validation_result.get('errors', []),
            'stats': validation_result.get('stats', {})
        }
        
        return self.validation_passed

class BaseQueryExecutor(ABC):
    """Abstract base class for query executors."""
    
    def __init__(self, engine_name: str, **kwargs):
        """
        Initialize the query executor.
        
        Args:
            engine_name: Name of the query engine (e.g., 'spark', 'duckdb')
            **kwargs: Additional engine-specific configuration
        """
        self.engine_name = engine_name
        self.initialized = False
        self.metrics_collector = MetricsCollector()
        self.config = kwargs
        
    def __enter__(self):
        """Context manager entry point."""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.cleanup()
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the query engine and any required resources."""
        pass
        
    @abstractmethod    
    def cleanup(self) -> None:
        """Clean up resources used by the query engine."""
        pass
        
    @abstractmethod
    def execute_query(self, query: str, query_id: Optional[str] = None) -> QueryResult:
        """
        Execute a SQL query and return the result.
        
        Args:
            query: The SQL query to execute
            query_id: Optional identifier for the query (for logging)
            
        Returns:
            QueryResult containing the execution results and metrics
        """
        pass
        
    @abstractmethod
    def get_table_names(self) -> List[str]:
        """
        Get a list of table names in the database.
        
        Returns:
            List of table names
        """
        pass
    
    def _start_metrics_collection(self) -> None:
        """Start collecting system metrics."""
        self.metrics_collector.start()
        
    def _stop_metrics_collection(self) -> Dict[str, Any]:
        """
        Stop collecting metrics and return the collected data.
        
        Returns:
            Dictionary containing system metrics summary
        """
        metrics = self.metrics_collector.stop()
        return MetricsCollector.analyze_metrics(metrics)
    
    def _create_result(self) -> QueryResult:
        """Create a new QueryResult instance."""
        return QueryResult()
    
    def _log_execution_metrics(self, result: QueryResult, query_id: Optional[str] = None) -> None:
        """
        Log execution metrics for a query.
        
        Args:
            result: The QueryResult containing execution metrics
            query_id: Optional query identifier for logging
        """
        query_info = f"Query {query_id}" if query_id else "Query"
        logger.info(
            f"{query_info} executed in {result.execution_time:.4f} seconds. "
            f"Rows returned: {result.rows_returned}"
        )
        
        if result.system_metrics:
            logger.debug(
                f"System metrics - CPU: {result.system_metrics['cpu']['mean']:.1f}%, "
                f"Memory: {result.system_metrics['memory_used_mb']['max']:.1f}MB, "
                f"Disk read: {result.system_metrics['disk_read_mb']['total']:.2f}MB, "
                f"Disk write: {result.system_metrics['disk_write_mb']['total']:.2f}MB"
            )
            
        if hasattr(result, 'validation_passed') and result.validation_passed is not None:
            status = "PASSED" if result.validation_passed else "FAILED"
            logger.info(f"Result validation: {status}")
            if not result.validation_passed and hasattr(result, 'validation_errors'):
                for error in result.validation_errors[:5]:  # Log first 5 errors
                    logger.error(f"Validation error: {error}")
                if len(result.validation_errors) > 5:
                    logger.error(f"... and {len(result.validation_errors) - 5} more errors")
    
    def _execute_with_metrics(self, query_func: callable, *args, **kwargs) -> QueryResult:
        """
        Execute a query function with metrics collection.
        
        Args:
            query_func: The function to execute (should return a QueryResult)
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            QueryResult with execution metrics
        """
        result = self._create_result()
        
        try:
            # Start metrics collection
            self._start_metrics_collection()
            start_time = time.time()
            
            # Execute the query
            query_result = query_func(*args, **kwargs)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Stop metrics collection and get system metrics
            system_metrics = self._stop_metrics_collection()
            
            # Update result with metrics
            result.execution_time = execution_time
            result.system_metrics = system_metrics
            result.metrics.update({
                'execution_time_seconds': execution_time,
                'system_metrics': system_metrics
            })
            
            # Copy attributes from the query result
            for attr in ['rows_returned', 'query_plan', 'result_data', 'success', 'error']:
                if hasattr(query_result, attr):
                    setattr(result, attr, getattr(query_result, attr))
            
            # Log execution metrics
            self._log_execution_metrics(
                result, 
                query_id=kwargs.get('query_id')
            )
            
            return result
            
        except Exception as e:
            # Ensure metrics collection is stopped on error
            self._stop_metrics_collection()
            
            # Update result with error information
            result.success = False
            result.error = str(e)
            result.execution_time = time.time() - start_time if 'start_time' in locals() else 0
            
            logger.error(
                f"Error executing query: {str(e)}",
                exc_info=True
            )
            
            return result
