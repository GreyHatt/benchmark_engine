from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

from .base import BaseQueryExecutor, QueryResult
from src.data.loader import SparkDataLoader

logger = logging.getLogger(__name__)

class SparkQueryExecutor(BaseQueryExecutor):
    """Query executor for PySpark in local mode."""
    
    def __init__(self, data_dir: Path, app_name: str = "TPC-H Benchmark", **kwargs):
        """
        Initialize the PySpark query executor in local mode.
        
        Args:
            data_dir: Path to the directory containing TPC-H data files
            app_name: Name for the Spark application
            **kwargs: Additional configuration options
        """
        super().__init__(engine_name='pyspark', **kwargs)
        self.data_dir = data_dir
        self.app_name = app_name
        self.spark: Optional[SparkSession] = None
        self.data_loader: Optional[SparkDataLoader] = None
        self.tables: Dict[str, DataFrame] = {}
    
    def initialize(self) -> None:
        """Initialize the PySpark session in local mode and load TPC-H data."""
        if self.initialized:
            return
            
        logger.info("Initializing PySpark session in local mode...")
        self.spark = SparkSession.builder \
            .appName(self.app_name) \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.default.parallelism", "4") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB") \
            .getOrCreate()
            
        # Initialize data loader
        self.data_loader = SparkDataLoader(self.spark, self.data_dir)
        
        # Load all tables
        logger.info("Loading TPC-H data into PySpark...")
        self.tables = self.data_loader.load_data()
        
        # Register tables as temp views
        for table_name, df in self.tables.items():
            df.createOrReplaceTempView(table_name)
            
        self.initialized = True
        logger.info("PySpark query executor initialized successfully in local mode")
    
    def execute_query(self, query: str, query_id: Optional[str] = None) -> QueryResult:
        """
        Execute a SQL query using PySpark in local mode.
        
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
            logger.info(f"Executing query {query_id or ''} on PySpark (local mode)")
            
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
            result.result_data = query_result
            result.success = True
            
            # Add PySpark config info
            result.metrics.update({
                "spark_config": {
                    k: v for k, v in self.spark.sparkContext.getConf().getAll()
                    if not k.startswith("spark.kubernetes.")  # Filter out k8s settings
                }
            })
                
            logger.info(f"Query executed in {execution_time:.2f} seconds, returned {result.rows_returned} rows")
            
        except Exception as e:
            error_msg = f"Error executing query on PySpark: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result.success = False
            result.error = error_msg
            
        return result
    
    def cleanup(self) -> None:
        """Clean up PySpark resources."""
        if self.spark:
            logger.info("Stopping PySpark session...")
            self.spark.stop()
            self.spark = None
            self.initialized = False
            logger.info("PySpark session stopped")
    
    def get_table_names(self) -> List[str]:
        """Get a list of available table names."""
        if not self.initialized:
            self.initialize()
        return list(self.tables.keys())
