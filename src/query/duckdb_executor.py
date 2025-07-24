from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import duckdb

from .base import BaseQueryExecutor, QueryResult
from ..data.loader import DuckDBDataLoader

logger = logging.getLogger(__name__)

class DuckDBQueryExecutor(BaseQueryExecutor):
    """Query executor for DuckDB."""
    
    def __init__(self, data_dir: Path, memory_db: bool = True, db_path: Optional[Path] = None):
        """
        Initialize the DuckDB query executor.
        
        Args:
            data_dir: Path to the directory containing TPC-H data files
            memory_db: If True, use an in-memory database
            db_path: If provided, use a persistent database at this path
        """
        super().__init__(data_dir)
        self.memory_db = memory_db
        self.db_path = db_path
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self.data_loader: Optional[DuckDBDataLoader] = None
    
    def initialize(self) -> None:
        """Initialize the DuckDB connection and load TPC-H data."""
        if self.initialized:
            return
            
        logger.info("Initializing DuckDB connection...")
        
        # Connect to DuckDB
        if self.memory_db:
            self.conn = duckdb.connect(database=':memory:')
        elif self.db_path:
            self.conn = duckdb.connect(database=str(self.db_path))
        else:
            self.conn = duckdb.connect(database=':memory:')
        
        # Set configuration for better performance
        self.conn.execute("PRAGMA threads=4")
        self.conn.execute("PRAGMA enable_progress_bar=true")
        
        # Initialize data loader
        self.data_loader = DuckDBDataLoader(self.conn, self.data_dir)
        
        # Load all tables
        logger.info("Loading TPC-H data into DuckDB...")
        self.data_loader.load_data()
        
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
        if not self.initialized or not self.conn:
            self.initialize()
            
        result = QueryResult()
        
        try:
            # Execute query and time it
            logger.info(f"Executing query {query_id or ''} on DuckDB")
            
            # Get query execution plan
            explain_result = self.conn.execute(f"EXPLAIN {query}")
            result.query_plan = "\n".join([row[0] for row in explain_result.fetchall()])
            
            # Execute query and measure time
            def execute():
                return self.conn.execute(query).fetchall()
                
            query_result, execution_time = self._time_execution(execute)
            
            # Get number of rows affected/returned
            rowcount = len(query_result) if query_result else 0
            
            # Populate result
            result.execution_time = execution_time
            result.rows_returned = rowcount
            result.result_data = query_result
            result.success = True
            
            # Get query profile information if available
            try:
                profile = self.conn.execute("SELECT * FROM duckdb_profiling_data()").fetchall()
                if profile:
                    result.metrics["profile"] = dict(profile[0])
            except Exception as e:
                logger.debug(f"Could not get query profile: {e}")
                
            logger.info(f"Query executed in {execution_time:.4f} seconds, returned {rowcount} rows")
            
        except Exception as e:
            logger.error(f"Error executing query on DuckDB: {str(e)}", exc_info=True)
            result.error = str(e)
            result.success = False
            
        return result
    
    def get_table_names(self) -> List[str]:
        """
        Get the list of available table names.
        
        Returns:
            List of table names
        """
        if not self.initialized or not self.conn:
            self.initialize()
            
        result = self.conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main'
        """).fetchall()
        
        return [row[0] for row in result]
    
    def close(self) -> None:
        """Close the DuckDB connection."""
        if self.conn:
            logger.info("Closing DuckDB connection...")
            self.conn.close()
            self.conn = None
            self.initialized = False
