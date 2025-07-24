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
            engine: Name of the query engine ('spark' or 'duckdb')
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
        
        if engine == 'spark':
            return SparkQueryExecutor(data_dir, **kwargs)
        elif engine == 'duckdb':
            return DuckDBQueryExecutor(data_dir, **kwargs)
        elif engine == 'hybrid':
            return HybridQueryExecutor(data_dir, **kwargs)
        else:
            raise ValueError(f"Unknown query engine: {engine}. Supported engines: 'spark', 'duckdb', 'hybrid'")
    
    @staticmethod
    def create_hybrid_executor(
        data_dir: Union[str, Path],
        spark_kwargs: Optional[dict] = None,
        duckdb_kwargs: Optional[dict] = None
    ) -> 'HybridQueryExecutor':
        """
        Create a hybrid query executor that can use both Spark and DuckDB.
        
        Args:
            data_dir: Path to the directory containing TPC-H data files
            spark_kwargs: Additional keyword arguments for the Spark executor
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
        self.data_dir = Path(data_dir)
        self.spark_executor = None
        self.duckdb_executor = None
        self.query_analyzer = QueryAnalyzer()
        
    def initialize(self) -> None:
        """Initialize both Spark and DuckDB executors."""
        if not self.initialized:
            logger.info("Initializing hybrid query executor...")
            
            # Initialize Spark executor
            self.spark_executor = SparkQueryExecutor(self.data_dir)
            self.spark_executor.initialize()
            
            # Initialize DuckDB executor
            self.duckdb_executor = DuckDBQueryExecutor(self.data_dir)
            self.duckdb_executor.initialize()
            
            self.initialized = True
            logger.info("Hybrid query executor initialized successfully")
    
    def cleanup(self) -> None:
        """Clean up resources used by both executors."""
        if self.initialized:
            logger.info("Cleaning up hybrid query executor...")
            
            if self.spark_executor:
                self.spark_executor.cleanup()
            if self.duckdb_executor:
                self.duckdb_executor.cleanup()
                
            self.initialized = False
            logger.info("Hybrid query executor cleaned up")
    
    def execute_query(self, query: str, query_id: Optional[str] = None, **kwargs) -> QueryResult:
        """
        Execute a query using the optimal execution strategy.
        
        Args:
            query: The SQL query to execute
            query_id: Optional identifier for the query
            **kwargs: Additional execution parameters
            
        Returns:
            QueryResult with execution results and metrics
        """
        result = self._create_result()
        result.query = query
        result.query_id = query_id
        
        try:
            # Analyze the query to determine the best execution strategy
            analysis = self.query_analyzer.analyze_query(query)
            logger.debug(f"Query analysis: {json.dumps(analysis, indent=2, default=str)}")
            
            # Determine execution strategy
            strategy = self._determine_execution_strategy(analysis, **kwargs)
            logger.info(f"Selected execution strategy: {strategy}")
            
            # Execute according to the selected strategy
            if strategy == 'spark':
                return self._execute_spark(query, query_id, analysis, **kwargs)
            elif strategy == 'duckdb':
                return self._execute_duckdb(query, query_id, analysis, **kwargs)
            elif strategy == 'parallel':
                return self._execute_parallel(query, query_id, analysis, **kwargs)
            else:
                raise ValueError(f"Unknown execution strategy: {strategy}")
                
        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Error executing query {query_id}: {str(e)}", exc_info=True)
            return result
    
    def get_table_names(self) -> List[str]:
        """Get the list of available table names from both executors."""
        if not self.initialized:
            self.initialize()
            
        # Get tables from both executors and return the union
        spark_tables = set(self.spark_executor.get_table_names())
        duckdb_tables = set(self.duckdb_executor.get_table_names())
        return list(spark_tables.union(duckdb_tables))
    
    def _execute_spark(self, query: str, query_id: str, analysis: Dict, **kwargs) -> QueryResult:
        """Execute a query using Spark."""
        logger.info(f"Executing query {query_id} on Spark")
        result = self._execute_with_metrics(
            self.spark_executor.execute_query,
            query,
            query_id=query_id,
            **kwargs
        )
        result.engine_used = 'spark'
        return result
    
    def _execute_duckdb(self, query: str, query_id: str, analysis: Dict, **kwargs) -> QueryResult:
        """Execute a query using DuckDB."""
        logger.info(f"Executing query {query_id} on DuckDB")
        result = self._execute_with_metrics(
            self.duckdb_executor.execute_query,
            query,
            query_id=query_id,
            **kwargs
        )
        result.engine_used = 'duckdb'
        return result
    
    def _execute_parallel(self, query: str, query_id: str, analysis: Dict, **kwargs) -> QueryResult:
        """
        Execute a query on both engines in parallel and compare results.
        
        This runs the query on both Spark and DuckDB concurrently and validates
        that the results match within an acceptable tolerance.
        """
        result = self._create_result()
        result.query = query
        result.query_id = query_id
        result.engine_used = 'hybrid'
        
        # Execute on both engines in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            future_to_engine = {
                executor.submit(self._execute_spark, query, f"{query_id}_spark", analysis, **kwargs): 'spark',
                executor.submit(self._execute_duckdb, query, f"{query_id}_duckdb", analysis, **kwargs): 'duckdb'
            }
            
            # Process results as they complete
            for future in as_completed(future_to_engine):
                engine = future_to_engine[future]
                try:
                    engine_result = future.result()
                    if engine == 'spark':
                        result.spark = engine_result
                        result.metrics['spark'] = engine_result.metrics
                    else:
                        result.duckdb = engine_result
                        result.metrics['duckdb'] = engine_result.metrics
                        
                    # Update overall execution time
                    result.execution_time = max(
                        result.execution_time,
                        engine_result.execution_time
                    )
                    
                except Exception as e:
                    logger.error(f"Error executing query on {engine}: {str(e)}", exc_info=True)
                    setattr(result, engine, QueryResult())
                    getattr(result, engine).error = str(e)
        
        # Validate results if both executions succeeded
        if (result.spark and result.spark.success and 
            result.duckdb and result.duckdb.success):
            
            # Validate results with a small tolerance for floating point differences
            validation_result = result.spark.validate_results(
                result.duckdb,
                tolerance=1e-6,
                ignore_order=True,
                ignore_case=True
            )
            
            result.validation_passed = validation_result
            
            # Determine which engine was faster
            if result.spark.execution_time < result.duckdb.execution_time:
                faster_engine = 'spark'
                speedup = result.duckdb.execution_time / result.spark.execution_time
            else:
                faster_engine = 'duckdb'
                speedup = result.spark.execution_time / result.duckdb.execution_time
            
            # Add comparison metrics
            result.metrics['comparison'] = {
                'faster_engine': faster_engine,
                'speedup': speedup,
                'spark_time': result.spark.execution_time,
                'duckdb_time': result.duckdb.execution_time,
                'time_difference': abs(result.spark.execution_time - result.duckdb.execution_time),
                'validation_passed': validation_result
            }
            
            logger.info(
                f"Query {query_id} - Spark: {result.spark.execution_time:.3f}s, "
                f"DuckDB: {result.duckdb.execution_time:.3f}s, "
                f"Faster: {faster_engine} ({speedup:.2f}x)"
            )
            
            if not validation_result:
                logger.warning(
                    f"Result validation failed for query {query_id}. "
                    f"Spark returned {result.spark.rows_returned} rows, "
                    f"DuckDB returned {result.duckdb.rows_returned} rows."
                    f"Errors: {result.spark.validation_errors[:3]}..."
                )
        
        # Set overall success based on both executions
        result.success = (
            result.spark and result.spark.success and 
            result.duckdb and result.duckdb.success and
            (not hasattr(result, 'validation_passed') or result.validation_passed)
        )
        
        return result
    
    def _determine_execution_strategy(self, analysis: Dict, **kwargs) -> str:
        """
        Determine the best execution strategy based on query analysis.
        
        Args:
            analysis: Dictionary containing query analysis results
            
        Returns:
            One of: 'spark', 'duckdb', or 'parallel'
        """
        # Allow strategy override from kwargs
        if 'strategy' in kwargs:
            return kwargs['strategy']
        
        # Simple rule-based strategy selection
        if analysis['table_count'] > 5 or analysis['complexity_score'] > 8:
            return 'spark'  # Spark is better for complex queries across many tables
        elif analysis['has_aggregations'] and analysis['table_count'] <= 2:
            return 'duckdb'  # DuckDB is often faster for simple aggregations
        else:
            return 'parallel'  # Run both and compare


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
        # Basic query information
        query = query.strip()
        is_select = query.upper().startswith('SELECT')
        
        # Check for common patterns
        has_joins = bool(self.join_pattern.search(query))
        has_aggregations = bool(self.aggregate_pattern.search(query))
        has_subqueries = bool(self.subquery_pattern.search(query))
        has_ctes = bool(self.cte_pattern.search(query))
        
        # Extract table names
        table_matches = self.table_pattern.findall(query)
        tables = list({tbl.lower() for tbl in table_matches if tbl and not tbl.upper() in ['AS', 'ON', 'USING']})
        
        # Calculate complexity score (simple heuristic)
        complexity_score = 0
        if has_joins:
            complexity_score += 2
        if has_aggregations:
            complexity_score += 2
        if has_subqueries:
            complexity_score += 3
        if has_ctes:
            complexity_score += 1
        complexity_score = min(10, complexity_score + len(tables))  # Cap at 10
        
        return {
            'is_select': is_select,
            'has_joins': has_joins,
            'has_aggregations': has_aggregations,
            'has_subqueries': has_subqueries,
            'has_ctes': has_ctes,
            'tables': tables,
            'table_count': len(tables),
            'complexity_score': complexity_score,
            'query_length': len(query),
            'normalized_query': self._normalize_query(query)
        }
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize a SQL query for comparison purposes.
        
        Args:
            query: The SQL query to normalize
            
        Returns:
            Normalized query string
        """
        # Convert to lowercase and remove extra whitespace
        normalized = ' '.join(query.split()).lower()
        
        # Remove comments
        normalized = re.sub(r'--.*?\n|/\*.*?\*/', '', normalized, flags=re.DOTALL)
        
        # Standardize whitespace around operators and parentheses
        normalized = re.sub(r'\s*([=<>!+\-*/%])\s*', r' \1 ', normalized)
        normalized = re.sub(r'\s*\(\s*', ' (', normalized)
        normalized = re.sub(r'\s*\)\s*', ') ', normalized)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized


def create_query_executor(executor_type: str, data_dir: Path, **kwargs) -> BaseQueryExecutor:
    """
    Create a query executor of the specified type.
    
    Args:
        executor_type: Type of executor to create ('spark', 'duckdb', or 'hybrid')
        data_dir: Path to the directory containing TPC-H data files
        **kwargs: Additional configuration options for the executor
        
    Returns:
        An instance of the specified query executor
        
    Raises:
        ValueError: If an invalid executor type is specified
    """
    if executor_type == 'spark':
        return SparkQueryExecutor(data_dir, **kwargs)
    elif executor_type == 'duckdb':
        return DuckDBQueryExecutor(data_dir, **kwargs)
    elif executor_type == 'hybrid':
        return HybridQueryExecutor(data_dir, **kwargs)
    else:
        raise ValueError(f"Unknown executor type: {executor_type}")


def get_available_executors() -> List[str]:
    """
    Get a list of available query executor types.
    
    Returns:
        List of available executor type names
    """
    return ['spark', 'duckdb', 'hybrid']
