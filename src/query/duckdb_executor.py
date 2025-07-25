from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import duckdb
import logging
import time
import json
from .base import BaseQueryExecutor, QueryResult
from .tpch_queries import get_query, get_query_parameters
from .metrics import MetricsCollector
from src.data.loader import DuckDBDataLoader

logger = logging.getLogger(__name__)

class DuckDBQueryExecutor(BaseQueryExecutor):
    """Query executor for DuckDB."""
    
    def __init__(self, data_dir: Union[str, Path], **kwargs):
        """
        Initialize the DuckDB query executor.
        
        Args:
            data_dir: Path to the directory containing TPC-H data files
            **kwargs: Additional configuration options
        """
        super().__init__(engine_name='duckdb', **kwargs)
        self.data_dir = Path(data_dir)
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self.data_loader: Optional[DuckDBDataLoader] = None
        self.tables: Dict[str, str] = {}
        self.metrics_collector = MetricsCollector()
    
    def initialize(self) -> None:
        """Initialize the DuckDB connection and load TPC-H data."""
        if self.initialized:
            return
            
        logger.info("Initializing DuckDB...")
        
        # Initialize DuckDB connection
        self.conn = duckdb.connect(database=':memory:')
        
        # Set configuration options
        self.conn.execute("PRAGMA threads=4")
        self.conn.execute("PRAGMA enable_progress_bar=true")
        self.conn.execute("CREATE SCHEMA IF NOT EXISTS tpch;")
        self.conn.execute("SET search_path TO tpch;")
        
        # Initialize data loader
        self.data_loader = DuckDBDataLoader(self.conn, self.data_dir)
        
        # Load all tables
        logger.info("Loading TPC-H data into DuckDB...")
        self.tables = self.data_loader.load_data()
        
        self.initialized = True
        logger.info("DuckDB query executor initialized successfully")
    
    def execute_query(self, query_id: str, parameters: Optional[Dict[str, Any]] = None, 
                     validate: bool = False) -> QueryResult:
        """
        Execute a TPC-H query by ID with the given parameters.
        
        Args:
            query_id: The ID of the TPC-H query to execute (e.g., 'q1', 'q2')
            parameters: Dictionary of parameter values to substitute in the query
            validate: Whether to validate the query results (not implemented yet)
            
        Returns:
            QueryResult containing the execution results and metrics
        """
        if not self.initialized:
            self.initialize()
            
        result = self._create_result()
        
        try:
            # Get the query template and merge with default parameters
            query_template = get_query(query_id)
            default_params = get_query_parameters(query_id)
            
            # Merge default parameters with provided ones (provided ones take precedence)
            final_params = {**default_params, **(parameters or {})}
            
            # Log the query being executed
            logger.info(f"Executing TPC-H query {query_id} with parameters: {final_params}")

            self.metrics_collector.start()
            
            # Execute the query with parameters
            start_time = time.time()
            cursor = self.conn.execute(query_template)
            
            # Fetch all results
            rows = cursor.fetchall()
            execution_time = time.time() - start_time

            system_metrics = self.metrics_collector.stop()
            
            # Populate the result object
            result.execution_time = execution_time
            result.rows_returned = len(rows)
            result.result_data = rows
            result.success = True
            result.metrics = MetricsCollector.analyze_metrics(system_metrics)
            
            # Get query plan if available
            try:
                plan = self.conn.execute(f"EXPLAIN {query_template}").fetchall()
                result.query_plan = '\n'.join(str(row[1]) for row in plan)
            except Exception as e:
                logger.warning(f"Could not get query plan: {str(e)}")
                result.query_plan = "Query plan not available"
            
            logger.info(f"Query {query_id} executed in {execution_time:.4f} seconds, returned {len(rows)} rows")
            
        except Exception as e:
            error_msg = f"Error executing query {query_id}: {str(e)}"
            logger.error(error_msg)
            result.error = error_msg
            result.success = False
            
        return result
    
    def cleanup(self) -> None:
        """Clean up DuckDB resources."""
        if self.conn:
            logger.info("Closing DuckDB connection...")
            self.conn.close()
            self.conn = None
            self.initialized = False
            logger.info("DuckDB connection closed")
    
    def get_table_names(self) -> List[str]:
        """Get a list of available table names."""
        if not self.initialized:
            self.initialize()
        return list(self.tables.keys())
    
    def _time_execution(self, func):
        """Helper method to time a function's execution."""
        start_time = time.time()
        result = func()
        execution_time = time.time() - start_time
        return result, execution_time
    
    def _create_result(self) -> QueryResult:
        return QueryResult()
