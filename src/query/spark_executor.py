from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

from .base import BaseQueryExecutor, QueryResult
from ..data.loader import SparkDataLoader

logger = logging.getLogger(__name__)

class SparkQueryExecutor(BaseQueryExecutor):
    """Query executor for Apache Spark."""
    
    def __init__(self, data_dir: Path, app_name: str = "TPC-H Benchmark"):
        """
        Initialize the Spark query executor.
        
        Args:
            data_dir: Path to the directory containing TPC-H data files
            app_name: Name for the Spark application
        """
        super().__init__(data_dir)
        self.app_name = app_name
        self.spark: Optional[SparkSession] = None
        self.data_loader: Optional[SparkDataLoader] = None
        self.tables: Dict[str, DataFrame] = {}
    
    def initialize(self) -> None:
        """Initialize the Spark session and load TPC-H data."""
        if self.initialized:
            return
            
        logger.info("Initializing Spark session...")
        self.spark = SparkSession.builder \
            .appName(self.app_name) \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
            
        # Initialize data loader
        self.data_loader = SparkDataLoader(self.spark, self.data_dir)
        
        # Load all tables
        logger.info("Loading TPC-H data into Spark...")
        self.tables = self.data_loader.load_data()
        
        # Register tables as temp views
        for table_name, df in self.tables.items():
            df.createOrReplaceTempView(table_name)
            
        self.initialized = True
        logger.info("Spark query executor initialized successfully")
    
    def execute_query(self, query: str, query_id: Optional[str] = None) -> QueryResult:
        """
        Execute a SQL query using Spark.
        
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
            logger.info(f"Executing query {query_id or ''} on Spark")
            
            # Get query execution plan
            query_plan = self.spark.sql(f"EXPLAIN EXTENDED {query}")
            result.query_plan = "\n".join([row[0] for row in query_plan.collect()])
            
            # Execute query and measure time
            def execute():
                df = self.spark.sql(query)
                # Force execution and collect results
                return df.collect()
                
            query_result, execution_time = self._time_execution(execute)
            
            # Populate result
            result.execution_time = execution_time
            result.rows_returned = len(query_result)
            result.success = True
            
            # Add Spark UI URL if available
            ui_url = self.spark.sparkContext.uiWebUrl
            if ui_url:
                result.metrics["spark_ui"] = ui_url
                
            logger.info(f"Query executed in {execution_time:.2f} seconds, returned {result.rows_returned} rows")
            
        except Exception as e:
            logger.error(f"Error executing query on Spark: {str(e)}", exc_info=True)
            result.error = str(e)
            result.success = False
            
        return result
    
    def get_table_names(self) -> List[str]:
        """
        Get the list of available table names.
        
        Returns:
            List of table names
        """
        if not self.initialized:
            self.initialize()
        return list(self.tables.keys())
    
    def close(self) -> None:
        """Stop the Spark session."""
        if self.spark is not None:
            logger.info("Stopping Spark session...")
            self.spark.stop()
            self.spark = None
            self.initialized = False
