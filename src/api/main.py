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
    scale_factor: float = 1.0
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

DATA_DIR = Path("./.data")

@app.post("/benchmark/run", response_model=BenchmarkResult)
async def run_benchmark(request: BenchmarkRequest):
    """Execute a benchmark with the specified query and engine."""
    try:
        logger.info(f"Running benchmark with engine: {request.engine}")
        
        query = request.query_id
        if request.query_id.startswith("q") and request.query_id[1:].isdigit():
            query = get_query(request.query_id)
            if not query:
                raise HTTPException(
                    status_code=400,
                    detail=f"Query {request.query_id} not found"
                )
        
        factory = QueryExecutorFactory()
        
        if request.engine.lower() == 'hybrid':
            executor = factory.create_hybrid_executor(
                data_dir=DATA_DIR,
                spark_kwargs={"app_name": "TPC-H Benchmark (Spark)"},
                duckdb_kwargs={"memory_db": True}
            )
        else:
            executor = factory.create_executor(
                engine=request.engine.lower(),
                data_dir=DATA_DIR,
                app_name=f"TPC-H Benchmark ({request.engine})"
            )
        
        with executor:
            result = executor.execute_query(query, request.query_id)
            
            if request.engine.lower() == 'hybrid' and result.spark and result.duckdb:
                validation_result = {
                    "spark_time": result.spark.execution_time,
                    "duckdb_time": result.duckdb.execution_time,
                    "faster_engine": result.metrics.get('faster_engine', 'unknown'),
                    "speedup": result.metrics.get('speedup', 1.0)
                }
                
                if result.spark.success and result.duckdb.success:
                    is_valid = result.spark.validate_results(result.duckdb)
                    validation_result.update({
                        "validation_passed": is_valid,
                        "validation_errors": result.spark.validation_errors
                    })
                
                result.metrics["comparison"] = validation_result
            
            return BenchmarkResult(
                query_id=request.query_id,
                engine=request.engine,
                execution_time=result.execution_time,
                status="success" if result.success else "error",
                metrics=result.metrics,
                result_validation=validation_result if request.engine.lower() == 'hybrid' else None,
                spark_result=result.spark.to_dict() if hasattr(result, 'spark') else None,
                duckdb_result=result.duckdb.to_dict() if hasattr(result, 'duckdb') else None,
                validation_result=validation_result if request.engine.lower() == 'hybrid' else None,
                comparison_metrics=result.metrics.get('comparison') if request.engine.lower() == 'hybrid' else None
            )
            
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "traceback": str(traceback.format_exc())
            }
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
    return ["spark", "duckdb", "hybrid"]

@app.post("/data/generate", response_model=DataGenerationResponse)
async def generate_data(request: DataGenerationRequest):
    """
    Generate TPC-H benchmark data with the specified scale factor.
    
    Args:
        scale_factor: Scale factor for data generation (default: 1.0)
        force: If True, regenerate data even if it already exists
    """
    try:
        # Ensure data directory exists
        data_dir = Path("./data")
        data_dir.mkdir(parents=True, exist_ok=True)
        
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