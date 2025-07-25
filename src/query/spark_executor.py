from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import time
import traceback

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

from .base import BaseQueryExecutor, QueryResult
from .tpch_queries import get_query, get_query_parameters
from .metrics import MetricsCollector
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
        self.metrics_collector = MetricsCollector()
    
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
    
    def execute_query(self, query_id: str, parameters: Optional[Dict[str, Any]] = None, 
                     validate: bool = False) -> QueryResult:
        """
        Execute a TPC-H query by ID with the given parameters using PySpark.
        
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
            
            # Register temp views for all tables
            for table_name, df in self.tables.items():
                df.createOrReplaceTempView(table_name)

            self.metrics_collector.start()
            
            # Execute the query with parameters
            start_time = time.time()
            
            # For Spark, we need to substitute parameters in the query string
            # since Spark SQL doesn't support parameterized queries like DuckDB
            query = query_template
            for param, value in final_params.items():
                query = query.replace(f':{param}', str(value))
            
            # Execute the query
            df = self.spark.sql(query)
            
            # Force execution and collect results
            rows = df.collect()
            execution_time = time.time() - start_time

            system_metrics = self.metrics_collector.stop()
            
            # Convert rows to list of dictionaries for serialization
            result_data = [row.asDict() for row in rows]
            
            # Populate the result object
            result.execution_time = execution_time
            result.rows_returned = len(rows)
            result.result_data = result_data
            result.success = True
            result.metrics = MetricsCollector.analyze_metrics(system_metrics)
            
            # Get query plan if available
            try:
                plan = self.spark.sql(f"EXPLAIN {query}")
                result.query_plan = '\n'.join([row[0] for row in plan.collect()])
            except Exception as e:
                logger.warning(f"Could not get query plan: {str(e)}")
                result.query_plan = "Query plan not available"
            
            logger.info(f"Query {query_id} executed in {execution_time:.4f} seconds, returned {len(rows)} rows")
            
        except Exception as e:
            error_msg = f"Error executing query {query_id}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            result.error = error_msg
            result.success = False
            
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
