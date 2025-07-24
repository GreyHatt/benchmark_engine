from typing import Dict, List, Optional, Any, Tuple, Union
import time
import logging
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .base import BaseQueryExecutor, QueryResult
from .spark_executor import SparkQueryExecutor
from .duckdb_executor import DuckDBQueryExecutor
from .metrics import MetricsCollector, QueryMetrics, QueryResultValidator

logger = logging.getLogger(__name__)

class QueryExecutorFactory:
    """Factory for creating query executors."""
    
    @staticmethod
    def create_executor(
        engine: str,
        data_dir: Union[str, Path],
        **kwargs
    ) -> BaseQueryExecutor:
        """
        Create a query executor for the specified engine.
        
        Args:
            engine: Name of the query engine ('pyspark' or 'duckdb')
            data_dir: Path to the directory containing TPC-H data files
            **kwargs: Additional keyword arguments for the executor
            
        Returns:
            An instance of a query executor
            
        Raises:
            ValueError: If an unknown engine is specified
        """
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
            
        engine = engine.lower()
        
        if engine in ('spark', 'pyspark'):
            # Default PySpark configuration for local mode
            spark_config = {
                'app_name': 'TPC-H Benchmark',
                'spark.driver.memory': '4g',
                'spark.executor.memory': '4g',
                'spark.sql.shuffle.partitions': '4',
                'spark.default.parallelism': '4',
                'spark.sql.adaptive.enabled': 'true',
                'spark.sql.adaptive.coalescePartitions.enabled': 'true',
                'spark.sql.adaptive.advisoryPartitionSizeInBytes': '128MB',
                **{k: v for k, v in kwargs.items() if k.startswith('spark.')}
            }
            return SparkQueryExecutor(data_dir, **spark_config)
            
        elif engine == 'duckdb':
            return DuckDBQueryExecutor(data_dir, **kwargs)
            
        else:
            raise ValueError(f"Unknown query engine: {engine}. Supported engines: 'pyspark', 'duckdb'")
    
    @staticmethod
    def create_hybrid_executor(
        data_dir: Union[str, Path],
        spark_kwargs: Optional[dict] = None,
        duckdb_kwargs: Optional[dict] = None
    ) -> 'HybridQueryExecutor':
        """
        Create a hybrid query executor that can use both PySpark and DuckDB.
        
        Args:
            data_dir: Path to the directory containing TPC-H data files
            spark_kwargs: Additional keyword arguments for the PySpark executor
            duckdb_kwargs: Additional keyword arguments for the DuckDB executor
            
        Returns:
            An instance of HybridQueryExecutor
        """
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
            
        spark_kwargs = spark_kwargs or {}
        duckdb_kwargs = duckdb_kwargs or {}
        
        return HybridQueryExecutor(
            data_dir=data_dir,
            spark_kwargs=spark_kwargs,
            duckdb_kwargs=duckdb_kwargs
        )


class HybridQueryExecutor(BaseQueryExecutor):
    """
    A query executor that can route queries to different execution engines
    based on query characteristics and collect comparative metrics.
    """
    
    def __init__(self, data_dir: Path, **kwargs):
        """
        Initialize the hybrid query executor.
        
        Args:
            data_dir: Path to the directory containing TPC-H data files
            **kwargs: Additional configuration options
        """
        super().__init__(engine_name='hybrid', **kwargs)
        self.data_dir = data_dir
        self.spark_executor = None
        self.duckdb_executor = None
        self.analyzer = QueryAnalyzer()
        
        # Get executor-specific kwargs
        self.spark_kwargs = kwargs.get('spark_kwargs', {})
        self.duckdb_kwargs = kwargs.get('duckdb_kwargs', {})
    
    def initialize(self) -> None:
        """Initialize both executors."""
        if self.initialized:
            return
            
        logger.info("Initializing hybrid query executor...")
        
        # Initialize PySpark executor
        self.spark_executor = QueryExecutorFactory.create_executor(
            'pyspark',
            self.data_dir,
            **self.spark_kwargs
        )
        self.spark_executor.initialize()
        
        # Initialize DuckDB executor
        self.duckdb_executor = QueryExecutorFactory.create_executor(
            'duckdb',
            self.data_dir,
            **self.duckdb_kwargs
        )
        self.duckdb_executor.initialize()
        
        self.initialized = True
        logger.info("Hybrid query executor initialized successfully")
    
    def execute_query(self, query: str, query_id: Optional[str] = None) -> QueryResult:
        """
        Execute a query using the most appropriate executor based on query analysis.
        
        Args:
            query: SQL query to execute
            query_id: Optional query identifier for logging
            
        Returns:
            QueryResult with execution results and metrics
        """
        if not self.initialized:
            self.initialize()
            
        result = QueryResult()
        
        try:
            # Analyze query to determine best execution engine
            analysis = self.analyzer.analyze_query(query)
            
            # Simple routing logic - can be enhanced based on analysis
            if analysis.get('has_joins', False) or analysis.get('is_analytical', False):
                # Use PySpark for complex analytical queries
                result = self.spark_executor.execute_query(query, query_id)
                result.engine_used = 'pyspark'
            else:
                # Use DuckDB for simpler queries
                result = self.duckdb_executor.execute_query(query, query_id)
                result.engine_used = 'duckdb'
                
            # For hybrid mode, always run with both engines and compare results
            if self.config.get('hybrid_validation', True):
                logger.info("Running hybrid validation with both engines...")
                
                # Execute with both engines in parallel
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_to_engine = {
                        executor.submit(self.spark_executor.execute_query, query, f"{query_id}_spark"): 'spark',
                        executor.submit(self.duckdb_executor.execute_query, query, f"{query_id}_duckdb"): 'duckdb'
                    }
                    
                    for future in as_completed(future_to_engine):
                        engine = future_to_engine[future]
                        try:
                            engine_result = future.result()
                            setattr(result, engine, engine_result)
                        except Exception as e:
                            logger.error(f"Error executing query with {engine}: {str(e)}", exc_info=True)
                
                # Compare results if both executions were successful
                if hasattr(result, 'spark') and hasattr(result, 'duckdb'):
                    result.validation_passed = result.spark.validate_results(result.duckdb)
                    if not result.validation_passed:
                        logger.warning("Result validation failed between PySpark and DuckDB")
            
            return result
            
        except Exception as e:
            error_msg = f"Error in hybrid query execution: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result.success = False
            result.error = error_msg
            return result
    
    def cleanup(self) -> None:
        """Clean up resources used by both executors."""
        if self.spark_executor:
            self.spark_executor.cleanup()
        if self.duckdb_executor:
            self.duckdb_executor.cleanup()
        self.initialized = False
    
    def get_table_names(self) -> List[str]:
        """Get table names from the primary executor (PySpark)."""
        if not self.initialized:
            self.initialize()
        return self.spark_executor.get_table_names()


class QueryAnalyzer:
    """Analyzes SQL queries to determine their characteristics."""
    
    def __init__(self):
        # Regular expressions for query analysis
        self.join_pattern = re.compile(r'\b(?:INNER|LEFT|RIGHT|FULL|CROSS|OUTER|JOIN)\b', re.IGNORECASE)
        self.aggregate_pattern = re.compile(r'\b(?:COUNT|SUM|AVG|MIN|MAX|GROUP BY|HAVING|DISTINCT)\b', re.IGNORECASE)
        self.subquery_pattern = re.compile(r'\(\s*SELECT\s+', re.IGNORECASE)
        self.cte_pattern = re.compile(r'\bWITH\s+[a-zA-Z_][a-zA-Z0-9_]*\s+AS\s*\(', re.IGNORECASE)
        self.table_pattern = re.compile(r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)', re.IGNORECASE)
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a SQL query to determine its characteristics.
        
        Args:
            query: The SQL query to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        normalized_query = self._normalize_query(query)
        
        return {
            'has_joins': bool(self.join_pattern.search(normalized_query)),
            'has_aggregates': bool(self.aggregate_pattern.search(normalized_query)),
            'has_subqueries': bool(self.subquery_pattern.search(normalized_query)),
            'has_ctes': bool(self.cte_pattern.search(normalized_query)),
            'tables_accessed': self.table_pattern.findall(normalized_query),
            'is_analytical': bool(self.aggregate_pattern.search(normalized_query) or 
                                 self.join_pattern.search(normalized_query))
        }
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize a SQL query for analysis.
        
        Args:
            query: The SQL query to normalize
            
        Returns:
            Normalized query string
        """
        # Remove comments
        query = re.sub(r'--.*?$', '', query, flags=re.MULTILINE)
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
        
        # Convert to single line and normalize whitespace
        query = ' '.join(query.split())
        
        # Convert to uppercase for case-insensitive matching
        return query.upper()


def create_query_executor(
    executor_type: str, 
    data_dir: Union[str, Path], 
    **kwargs
) -> BaseQueryExecutor:
    """
    Create a query executor of the specified type.
    
    Args:
        executor_type: Type of executor to create ('pyspark', 'duckdb', or 'hybrid')
        data_dir: Path to the directory containing TPC-H data files
        **kwargs: Additional configuration options for the executor
        
    Returns:
        An instance of the specified query executor
        
    Raises:
        ValueError: If an invalid executor type is specified
    """
    return QueryExecutorFactory.create_executor(executor_type, data_dir, **kwargs)


def get_available_executors() -> List[str]:
    """
    Get a list of available query executor types.
    
    Returns:
        List of available executor type names
    """
    return ['pyspark', 'duckdb', 'hybrid']
