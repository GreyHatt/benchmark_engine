from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import duckdb
import logging
import time
import json

from .base import BaseQueryExecutor, QueryResult
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
        
        # Initialize data loader
        self.data_loader = DuckDBDataLoader(self.conn, self.data_dir)
        
        # Load all tables
        logger.info("Loading TPC-H data into DuckDB...")
        self.tables = self.data_loader.load_data()
        
        self.initialized = True
        logger.info("DuckDB query executor initialized successfully")
    
    def execute_query(self, query: str, query_id: Optional[str] = None) -> QueryResult:
        """
        Execute a SQL query using DuckDB.
        
        Args:
            query: SQL query to execute
            query_id: Optional identifier for the query (e.g., 'q1', 'q2')
            
        Returns:
            QueryResult object containing execution results and metrics
        """
        if not self.initialized:
            self.initialize()
            
        result = QueryResult()
        
        try:
            # Execute query and time it
            logger.info(f"Executing query {query_id or ''} on DuckDB")
            
            # Get query execution plan
            plan = self.conn.sql(f"EXPLAIN {query}").fetchall()
            result.query_plan = "\n".join([str(row[0]) for row in plan])
            
            # Execute query and measure time
            def execute():
                return self.conn.sql(query).fetchall()
                
            query_result, execution_time = self._time_execution(execute)
            
            # Populate result
            result.execution_time = execution_time
            result.rows_returned = len(query_result)
            result.result_data = query_result
            result.success = True
            
            # Add DuckDB config info
            config = self.conn.sql("SELECT * FROM duckdb_settings()").fetchall()
            result.metrics['duckdb_config'] = {
                setting[0]: setting[1] for setting in config
            }
            
            logger.info(f"Query executed in {execution_time:.2f} seconds, returned {result.rows_returned} rows")
            
        except Exception as e:
            error_msg = f"Error executing query on DuckDB: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result.success = False
            result.error = error_msg
            
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
