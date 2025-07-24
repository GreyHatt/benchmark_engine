import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import duckdb
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
import pandas as pd

logger = logging.getLogger(__name__)

class DataLoader:
    """Base class for data loaders."""
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing TPC-H data files
        """
        self.data_dir = Path(data_dir)
        self.tables = [
            'part', 'supplier', 'partsupp', 'customer', 
            'orders', 'lineitem', 'nation', 'region'
        ]
    
    def load_data(self) -> Dict[str, Any]:
        """Load TPC-H data into the target system."""
        raise NotImplementedError


class SparkDataLoader(DataLoader):
    """Load TPC-H data into Spark."""
    
    def __init__(self, spark: SparkSession, data_dir: str = "./data"):
        """
        Initialize the Spark data loader.
        
        Args:
            spark: SparkSession instance
            data_dir: Directory containing TPC-H data files
        """
        super().__init__(data_dir)
        self.spark = spark
    
    def load_data(self) -> Dict[str, SparkDataFrame]:
        """
        Load TPC-H data into Spark DataFrames.
        
        Returns:
            Dictionary mapping table names to Spark DataFrames
        """
        logger.info("Loading TPC-H data into Spark...")
        
        schemas = self._get_table_schemas()
        tables = {}
        
        for table in self.tables:
            file_path = self.data_dir / f"{table}.tbl"
            if not file_path.exists():
                raise FileNotFoundError(f"TPC-H data file not found: {file_path}")
                
            df = self.spark.read \
                .option("delimiter", "|") \
                .option("header", "false") \
                .csv(str(file_path))
            
            schema = schemas[table]
            for i, (col_name, col_type) in enumerate(schema.items()):
                df = df.withColumnRenamed(f"_c{i}", col_name)
                df = df.withColumn(col_name, df[col_name].cast(col_type))
            
            tables[table] = df
            logger.info(f"Loaded table: {table} with {df.count():,} rows")
            
        return tables
    
    def _get_table_schemas(self) -> Dict[str, Dict[str, str]]:
        """Get the schema definitions for all TPC-H tables."""
        return {
            'part': {
                'p_partkey': 'bigint', 'p_name': 'string', 'p_mfgr': 'string',
                'p_brand': 'string', 'p_type': 'string', 'p_size': 'int',
                'p_container': 'string', 'p_retailprice': 'double', 'p_comment': 'string'
            },
            'supplier': {
                's_suppkey': 'bigint', 's_name': 'string', 's_address': 'string',
                's_nationkey': 'int', 's_phone': 'string', 's_acctbal': 'double',
                's_comment': 'string'
            },
            'partsupp': {
                'ps_partkey': 'bigint', 'ps_suppkey': 'bigint', 'ps_availqty': 'int',
                'ps_supplycost': 'double', 'ps_comment': 'string'
            },
            'customer': {
                'c_custkey': 'bigint', 'c_name': 'string', 'c_address': 'string',
                'c_nationkey': 'int', 'c_phone': 'string', 'c_acctbal': 'double',
                'c_mktsegment': 'string', 'c_comment': 'string'
            },
            'orders': {
                'o_orderkey': 'bigint', 'o_custkey': 'bigint', 'o_orderstatus': 'string',
                'o_totalprice': 'double', 'o_orderdate': 'date', 'o_orderpriority': 'string',
                'o_clerk': 'string', 'o_shippriority': 'int', 'o_comment': 'string'
            },
            'lineitem': {
                'l_orderkey': 'bigint', 'l_partkey': 'bigint', 'l_suppkey': 'bigint',
                'l_linenumber': 'int', 'l_quantity': 'double', 'l_extendedprice': 'double',
                'l_discount': 'double', 'l_tax': 'double', 'l_returnflag': 'string',
                'l_linestatus': 'string', 'l_shipdate': 'date', 'l_commitdate': 'date',
                'l_receiptdate': 'date', 'l_shipinstruct': 'string',
                'l_shipmode': 'string', 'l_comment': 'string'
            },
            'nation': {
                'n_nationkey': 'int', 'n_name': 'string',
                'n_regionkey': 'int', 'n_comment': 'string'
            },
            'region': {
                'r_regionkey': 'int', 'r_name': 'string', 'r_comment': 'string'
            }
        }


class DuckDBDataLoader(DataLoader):
    """Load TPC-H data into DuckDB."""
    
    def __init__(self, conn: Optional[duckdb.DuckDBPyConnection] = None, data_dir: str = "./data"):
        """
        Initialize the DuckDB data loader.
        
        Args:
            conn: DuckDB connection (will create one if not provided)
            data_dir: Directory containing TPC-H data files
        """
        super().__init__(data_dir)
        self.conn = conn or duckdb.connect(database=':memory:')
    
    def load_data(self) -> Dict[str, Any]:
        """
        Load TPC-H data into DuckDB.
        
        Returns:
            Dictionary mapping table names to DuckDB relations
        """
        logger.info("Loading TPC-H data into DuckDB...")
        
        self._create_tables()
        
        tables = {}
        for table in self.tables:
            file_path = self.data_dir / f"{table}.tbl"
            if not file_path.exists():
                raise FileNotFoundError(f"TPC-H data file not found: {file_path}")
                
            self.conn.execute(f"""
                COPY {table} FROM '{file_path}' 
                WITH (DELIMITER '|', FORMAT 'csv', HEADER false)
            """)
            
            tables[table] = self.conn.table(table)
            count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info(f"Loaded table: {table} with {count:,} rows")
            
        return tables
    
    def _create_tables(self) -> None:
        """Create all TPC-H tables in DuckDB with proper schema and constraints."""
        self.conn.execute("""
            CREATE SCHEMA IF NOT EXISTS tpch;
            SET search_path TO tpch;
            
            -- Region table
            CREATE TABLE IF NOT EXISTS region (
                r_regionkey INTEGER PRIMARY KEY,
                r_name     CHAR(25) NOT NULL,
                r_comment  VARCHAR(152)
            );
            
            -- Nation table
            CREATE TABLE IF NOT EXISTS nation (
                n_nationkey  INTEGER PRIMARY KEY,
                n_name      CHAR(25) NOT NULL,
                n_regionkey INTEGER NOT NULL,
                n_comment   VARCHAR(152),
                FOREIGN KEY (n_regionkey) REFERENCES region(r_regionkey)
            );
            
            -- Part table
            CREATE TABLE IF NOT EXISTS part (
                p_partkey     BIGINT PRIMARY KEY,
                p_name        VARCHAR(55) NOT NULL,
                p_mfgr        CHAR(25) NOT NULL,
                p_brand       CHAR(10) NOT NULL,
                p_type        VARCHAR(25) NOT NULL,
                p_size        INTEGER NOT NULL,
                p_container   CHAR(10) NOT NULL,
                p_retailprice DECIMAL(15,2) NOT NULL,
                p_comment     VARCHAR(23) NOT NULL
            );
            
            -- Supplier table
            CREATE TABLE IF NOT EXISTS supplier (
                s_suppkey     BIGINT PRIMARY KEY,
                s_name        CHAR(25) NOT NULL,
                s_address     VARCHAR(40) NOT NULL,
                s_nationkey   INTEGER NOT NULL,
                s_phone       CHAR(15) NOT NULL,
                s_acctbal     DECIMAL(15,2) NOT NULL,
                s_comment     VARCHAR(101) NOT NULL,
                FOREIGN KEY (s_nationkey) REFERENCES nation(n_nationkey)
            );
            
            -- Partsupp table
            CREATE TABLE IF NOT EXISTS partsupp (
                ps_partkey     BIGINT NOT NULL,
                ps_suppkey     BIGINT NOT NULL,
                ps_availqty    INTEGER NOT NULL,
                ps_supplycost  DECIMAL(15,2) NOT NULL,
                ps_comment     VARCHAR(199) NOT NULL,
                PRIMARY KEY (ps_partkey, ps_suppkey),
                FOREIGN KEY (ps_partkey) REFERENCES part(p_partkey),
                FOREIGN KEY (ps_suppkey) REFERENCES supplier(s_suppkey)
            );
            
            -- Customer table
            CREATE TABLE IF NOT EXISTS customer (
                c_custkey     BIGINT PRIMARY KEY,
                c_name        VARCHAR(25) NOT NULL,
                c_address     VARCHAR(40) NOT NULL,
                c_nationkey   INTEGER NOT NULL,
                c_phone       CHAR(15) NOT NULL,
                c_acctbal     DECIMAL(15,2) NOT NULL,
                c_mktsegment  CHAR(10) NOT NULL,
                c_comment     VARCHAR(117) NOT NULL,
                FOREIGN KEY (c_nationkey) REFERENCES nation(n_nationkey)
            );
            
            -- Orders table
            CREATE TABLE IF NOT EXISTS orders (
                o_orderkey       BIGINT PRIMARY KEY,
                o_custkey        BIGINT NOT NULL,
                o_orderstatus    CHAR(1) NOT NULL,
                o_totalprice     DECIMAL(15,2) NOT NULL,
                o_orderdate      DATE NOT NULL,
                o_orderpriority  CHAR(15) NOT NULL,
                o_clerk          CHAR(15) NOT NULL,
                o_shippriority   INTEGER NOT NULL,
                o_comment        VARCHAR(79) NOT NULL,
                FOREIGN KEY (o_custkey) REFERENCES customer(c_custkey)
            );
            
            -- Lineitem table
            CREATE TABLE IF NOT EXISTS lineitem (
                l_orderkey      BIGINT NOT NULL,
                l_partkey       BIGINT NOT NULL,
                l_suppkey       BIGINT NOT NULL,
                l_linenumber    INTEGER NOT NULL,
                l_quantity      DECIMAL(15,2) NOT NULL,
                l_extendedprice DECIMAL(15,2) NOT NULL,
                l_discount      DECIMAL(15,2) NOT NULL,
                l_tax           DECIMAL(15,2) NOT NULL,
                l_returnflag    CHAR(1) NOT NULL,
                l_linestatus    CHAR(1) NOT NULL,
                l_shipdate      DATE NOT NULL,
                l_commitdate    DATE NOT NULL,
                l_receiptdate   DATE NOT NULL,
                l_shipinstruct  CHAR(25) NOT NULL,
                l_shipmode      CHAR(10) NOT NULL,
                l_comment       VARCHAR(44) NOT NULL,
                PRIMARY KEY (l_orderkey, l_linenumber),
                FOREIGN KEY (l_orderkey) REFERENCES orders(o_orderkey),
                FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp(ps_partkey, ps_suppkey)
            );
            
            -- Create indexes for better query performance
            CREATE INDEX IF NOT EXISTS idx_lineitem_orderkey ON lineitem(l_orderkey);
            CREATE INDEX IF NOT EXISTS idx_lineitem_partkey ON lineitem(l_partkey);
            CREATE INDEX IF NOT EXISTS idx_lineitem_suppkey ON lineitem(l_suppkey);
            CREATE INDEX IF NOT EXISTS idx_orders_custkey ON orders(o_custkey);
            CREATE INDEX IF NOT EXISTS idx_customer_nationkey ON customer(c_nationkey);
            CREATE INDEX IF NOT EXISTS idx_supplier_nationkey ON supplier(s_nationkey);
            CREATE INDEX IF NOT EXISTS idx_partsupp_partkey ON partsupp(ps_partkey);
            CREATE INDEX IF NOT EXISTS idx_partsupp_suppkey ON partsupp(ps_suppkey);
        """)
        
    def close(self) -> None:
        """Close the DuckDB connection."""
        if self.conn:
            self.conn.close()


def load_data(engine: str, **kwargs) -> Dict[str, Any]:
    """
    Load TPC-H data using the specified engine.
    
    Args:
        engine: Engine to use ('spark' or 'duckdb')
        **kwargs: Additional arguments for the data loader
        
    Returns:
        Dictionary mapping table names to their respective data structures
    """
    if engine == 'spark':
        spark = kwargs.get('spark', SparkSession.builder.appName("TPC-H Benchmark").getOrCreate())
        return SparkDataLoader(spark, **kwargs).load_data()
    elif engine == 'duckdb':
        return DuckDBDataLoader(**kwargs).load_data()
    else:
        raise ValueError(f"Unsupported engine: {engine}")
