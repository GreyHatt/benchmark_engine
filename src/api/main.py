from pathlib import Path
from datetime import datetime
from query.factory import QueryExecutorFactory
from query.tpch_queries import get_query, get_all_queries
import json

# Update the BenchmarkResult model
class BenchmarkResult(BaseModel):
    query: str
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

# Add this near the top of the file
DATA_DIR = Path("./data")  # Update this path as needed

# Update the run_benchmark endpoint
@app.post("/benchmark/run", response_model=BenchmarkResult)
async def run_benchmark(request: BenchmarkRequest):
    """Execute a benchmark with the specified query and engine."""
    try:
        logger.info(f"Running benchmark with engine: {request.engine}")
        
        # Get the query (support both direct SQL and TPC-H query IDs)
        query = request.query
        if request.query.startswith("q") and request.query[1:].isdigit():
            query = get_query(request.query)
            if not query:
                raise HTTPException(
                    status_code=400,
                    detail=f"Query {request.query} not found"
                )
        
        # Initialize the appropriate executor
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
        
        # Execute the query
        with executor:
            result = executor.execute_query(query, request.query)
            
            # For hybrid mode, add validation results
            if request.engine.lower() == 'hybrid' and result.spark and result.duckdb:
                validation_result = {
                    "spark_time": result.spark.execution_time,
                    "duckdb_time": result.duckdb.execution_time,
                    "faster_engine": result.metrics.get('faster_engine', 'unknown'),
                    "speedup": result.metrics.get('speedup', 1.0)
                }
                
                # Add validation details if both executions were successful
                if result.spark.success and result.duckdb.success:
                    is_valid = result.spark.validate_results(result.duckdb)
                    validation_result.update({
                        "validation_passed": is_valid,
                        "validation_errors": result.spark.validation_errors
                    })
                
                result.metrics["comparison"] = validation_result
            
            # Prepare the response
            return BenchmarkResult(
                query=request.query,
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

# Update the list_available_queries endpoint
@app.get("/benchmark/queries")
async def list_available_queries():
    """List all available TPC-H benchmark queries."""
    queries = get_all_queries()
    return {
        "queries": [
            {
                "id": qid,
                "name": q.split('\n')[0].strip('-- ').strip(),
                "description": '\n'.join(
                    line.strip('-- ') 
                    for line in q.split('\n')[1:] 
                    if line.strip().startswith('--')
                ).strip()
            }
            for qid, q in queries.items()
        ]
    }