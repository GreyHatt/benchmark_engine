from fastapi import FastAPI, HTTPException, BackgroundTasks
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union
from src.query.factory import QueryExecutorFactory
from src.query.tpch_queries import get_query, get_all_queries
from src.data.generator import generate_tpch_data
import json
import logging
import traceback
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TPC-H Benchmark API",
    description="API for running TPC-H benchmarks",
    version="1.0.0"
)

class BenchmarkRequest(BaseModel):
    query_id: str
    engine: str
    parameters: Optional[Dict[str, Any]] = None
    validate_results: bool = False

class BenchmarkResult(BaseModel):
    query_id: str
    engine: str
    execution_time: float
    status: str
    metrics: Dict[str, Any] = {}
    result_validation: Optional[Dict[str, Any]] = None
    timestamp: str = datetime.utcnow().isoformat()
    spark_result: Optional[Dict[str, Any]] = None
    duckdb_result: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None
    comparison_metrics: Optional[Dict[str, Any]] = None

class DataGenerationRequest(BaseModel):
    scale_factor: float = 1.0
    force: bool = False

class DataGenerationResponse(BaseModel):
    status: str
    message: str
    output_dir: Optional[str] = None
    generated_files: Optional[List[str]] = None
    error: Optional[str] = None

class DataStatusResponse(BaseModel):
    data_exists: bool
    data_directory: str
    tables: Dict[str, bool]
    missing_tables: List[str]
    total_size_mb: float

DATA_DIR = Path("/app/data")

def check_tpch_data() -> DataStatusResponse:
    """Check if TPC-H data files exist and return their status."""
    data_dir = Path(DATA_DIR)
    required_tables = [
        'part', 'supplier', 'partsupp', 'customer',
        'orders', 'lineitem', 'nation', 'region'
    ]
    
    tables_status = {}
    missing_tables = []
    total_size = 0
    
    for table in required_tables:
        table_file = data_dir / f"{table}.tbl"
        exists = table_file.exists()
        tables_status[table] = exists
        if not exists:
            missing_tables.append(table)
        elif exists:
            total_size += table_file.stat().st_size
    
    return DataStatusResponse(
        data_exists=all(tables_status.values()),
        data_directory=str(data_dir.absolute()),
        tables=tables_status,
        missing_tables=missing_tables,
        total_size_mb=round(total_size / (1024 * 1024), 2)  # Convert to MB
    )

@app.post("/benchmark/run", response_model=BenchmarkResult)
async def run_benchmark(request: BenchmarkRequest):
    """Execute a benchmark with the specified query and engine."""
    try:
        # Get the data directory and check if data exists
        data_dir = Path(DATA_DIR)
        if not data_dir.exists() or not any(data_dir.glob("*.tbl")):
            raise HTTPException(
                status_code=400,
                detail="TPC-H data not found. Please generate the data first."
            )
            
        # Normalize query ID format (e.g., '1' -> 'q1')
        query_id = request.query_id
        if query_id.isdigit():
            query_id = f"q{query_id}"
        elif not query_id.startswith('q') and query_id[1:].isdigit():
            query_id = f"q{query_id[1:] if query_id[0].lower() == 'q' else query_id}"

        # Initialize the query executor
        executor = QueryExecutorFactory.create_executor(
            engine=request.engine,
            data_dir=str(data_dir.absolute())
        )

        # Execute the query
        with executor:
            result = executor.execute_query(
                query_id=query_id,
                parameters=request.parameters or {},
                validate=request.validate_results
            )

        # Convert the result to a dictionary
        result_dict = result.to_dict()
        
        # Create the response with all required fields
        response = {
            'query_id': query_id,
            'engine': request.engine,
            'execution_time': result_dict.get('execution_time', 0),
            'status': 'success' if result_dict.get('success') else 'error',
            'metrics': result_dict.get('metrics', {}),
            'result_validation': result_dict.get('validation_result'),
            'spark_result': result_dict if request.engine == 'spark' else None,
            'duckdb_result': result_dict if request.engine == 'duckdb' else None,
            'validation_result': result_dict.get('validation_result'),
            'comparison_metrics': result_dict.get('comparison_metrics'),
            'timestamp': datetime.utcnow().isoformat()
        }
        # If there was an error, include it in the response
        if not result_dict.get('success') and 'error' in result_dict:
            response['error'] = result_dict['error']
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error running benchmark: {str(e)}"
        )

@app.get("/benchmark/queries")
async def list_available_queries():
    """List all available TPC-H benchmark queries."""
    queries = get_all_queries()
    result = []
    for qid, query in queries.items():
        lines = query.split('\n')
        name = lines[0].strip('-- ').strip() if lines[0].startswith('--') else qid
        description = '\n'.join(
            line.strip('--').strip()
            for line in lines[1:]
            if line.strip().startswith('--')
        )
        result.append({
            "id": qid,
            "name": name,
            "description": description or "No description available"
        })
    return {"queries": result}

@app.get("/benchmark/engines")
async def list_engines():
    """List all available query engines."""
    return ["spark", "duckdb"]

@app.get("/benchmark/data/status", response_model=DataStatusResponse)
async def get_data_status():
    """
    Check if TPC-H data files exist.
    Returns detailed status about the data files.
    """
    try:
        return check_tpch_data()
    except Exception as e:
        logger.error(f"Error checking data status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error checking data status: {str(e)}"
        )

@app.post("/benchmark/data/generate", response_model=DataGenerationResponse)
async def generate_data(request: DataGenerationRequest):
    """
    Generate TPC-H benchmark data with the specified scale factor.
    
    Args:
        scale_factor: Scale factor for data generation (default: 1.0)
        force: If True, regenerate data even if it already exists
    """
    try:
        data_dir = Path(DATA_DIR)
        
        # Generate data
        logger.info(f"Generating TPC-H data with scale factor {request.scale_factor}")
        result = generate_tpch_data(
            scale_factor=request.scale_factor,
            data_dir=str(data_dir.absolute()),
            force=request.force
        )
        
        # Get list of generated files
        generated_files = [str(f) for f in data_dir.glob("*.tbl")]
        
        return DataGenerationResponse(
            status="success",
            message=f"Successfully generated TPC-H data (scale factor: {request.scale_factor})",
            output_dir=str(data_dir.absolute()),
            generated_files=generated_files
        )
        
    except Exception as e:
        error_msg = f"Error generating TPC-H data: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return DataGenerationResponse(
            status="error",
            message="Failed to generate TPC-H data",
            error=error_msg
        )