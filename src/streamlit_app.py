import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
import json

# Set page config
st.set_page_config(
    page_title="Big Data Benchmarking Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000/api"

# Custom CSS for better styling
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    }
    .success {
        color: #28a745;
    }
    .error {
        color: #dc3545;
    }
    .warning {
        color: #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

def get_queries() -> List[Dict]:
    """Fetch available queries from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/queries")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching queries: {str(e)}")
        return []

def get_engines() -> List[str]:
    """Fetch available query engines from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/engines")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching engines: {str(e)}")
        return ["spark", "duckdb", "hybrid"]  # Fallback

def run_benchmark(query_id: str, engine: str, scale_factor: float = 1.0, parameters: Optional[Dict] = None) -> Optional[str]:
    """Run a benchmark and return the benchmark ID."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/benchmark/run",
            json={
                "query_id": query_id,
                "engine": engine,
                "scale_factor": scale_factor,
                "parameters": parameters or {},
                "validate_results": True,
                "execution_mode": "auto"
            }
        )
        response.raise_for_status()
        return response.json().get("benchmark_id")
    except requests.exceptions.RequestException as e:
        st.error(f"Error running benchmark: {str(e)}")
        return None

def get_benchmark_status(benchmark_id: str) -> Optional[Dict]:
    """Get the status of a benchmark."""
    try:
        response = requests.get(f"{API_BASE_URL}/benchmark/status/{benchmark_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting benchmark status: {str(e)}")
        return None

def display_metrics(metrics: Dict, title: str = "Metrics") -> None:
    """Display metrics in a card."""
    if not metrics:
        return
        
    with st.expander(title):
        cols = st.columns(3)
        for i, (key, value) in enumerate(metrics.items()):
            if isinstance(value, (int, float)):
                cols[i % 3].metric(key, f"{value:.4f}" if isinstance(value, float) else value)
            elif isinstance(value, dict):
                display_metrics(value, key)
            else:
                cols[i % 3].text(f"{key}: {value}")

def display_validation_results(validation: Dict) -> None:
    """Display validation results."""
    if not validation:
        return
        
    with st.expander("Validation Results", expanded=True):
        if validation.get("passed"):
            st.success("✅ Results validated successfully")
        else:
            st.error("❌ Validation failed")
            
        if "errors" in validation and validation["errors"]:
            with st.container():
                st.write("#### Errors")
                for error in validation["errors"][:5]:  # Show first 5 errors
                    st.error(f"- {error}")
                if len(validation["errors"]) > 5:
                    st.warning(f"... and {len(validation['errors']) - 5} more errors")
        
        if "stats" in validation and validation["stats"]:
            st.write("#### Statistics")
            display_metrics(validation["stats"])

def display_system_metrics(metrics: Dict) -> None:
    """Display system metrics in a card."""
    if not metrics:
        return
        
    with st.expander("System Metrics", expanded=False):
        # CPU Usage
        if "cpu" in metrics and "mean" in metrics["cpu"]:
            st.write("#### CPU Usage (%)")
            cpu_data = {
                "min": metrics["cpu"].get("min", 0),
                "mean": metrics["cpu"].get("mean", 0),
                "max": metrics["cpu"].get("max", 0)
            }
            st.metric("Average CPU Usage", f"{cpu_data['mean']:.1f}%")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cpu_data["mean"],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CPU Usage"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgreen"},
                           {'range': [50, 80], 'color': "yellow"},
                           {'range': [80, 100], 'color': "red"}],
                       'threshold': {
                           'line': {'color': "black", 'width': 4},
                           'thickness': 0.75,
                           'value': cpu_data["mean"]}}
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        # Memory Usage
        if "memory_used_mb" in metrics and "max" in metrics["memory_used_mb"]:
            st.write("#### Memory Usage (MB)")
            mem_data = {
                "min": metrics["memory_used_mb"].get("min", 0),
                "mean": metrics["memory_used_mb"].get("mean", 0),
                "max": metrics["memory_used_mb"].get("max", 0)
            }
            st.metric("Peak Memory Usage", f"{mem_data['max']:.1f} MB")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=mem_data["max"],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Peak Memory Usage (MB)"},
                gauge={'axis': {'range': [0, mem_data["max"] if mem_data["max"] > 0 else 100]},
                       'bar': {'color': "darkblue"},
                       'threshold': {
                           'line': {'color': "black", 'width': 4},
                           'thickness': 0.75,
                           'value': mem_data["max"]}}
            ))
            st.plotly_chart(fig, use_container_width=True)

def display_comparison_metrics(metrics: Dict) -> None:
    """Display comparison metrics between engines."""
    if not metrics:
        return
        
    with st.expander("Engine Comparison", expanded=True):
        if "faster_engine" in metrics and "speedup" in metrics:
            st.metric(
                f"Faster Engine: {metrics['faster_engine'].upper()}",
                f"{metrics['speedup']:.2f}x speedup"
            )
            
            # Create a bar chart comparing execution times
            if "spark_time" in metrics and "duckdb_time" in metrics:
                df = pd.DataFrame({
                    "Engine": ["Spark", "DuckDB"],
                    "Execution Time (s)": [metrics["spark_time"], metrics["duckdb_time"]]
                })
                
                fig = px.bar(
                    df, 
                    x="Engine", 
                    y="Execution Time (s)",
                    color="Engine",
                    title="Execution Time Comparison",
                    text_auto='.3s'
                )
                fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed comparison metrics
        if "time_difference" in metrics:
            st.metric("Time Difference", f"{metrics['time_difference']:.4f} seconds")

def main():
    """Main Streamlit application."""
    st.title("Big Data Benchmarking Engine")
    st.markdown("Compare query performance across different execution engines.")
    
    # Sidebar for benchmark configuration
    with st.sidebar:
        st.header("Benchmark Configuration")
        
        # Query selection
        queries = get_queries()
        query_options = {q["id"]: f"{q['id']}: {q.get('description', 'No description')}" for q in queries}
        selected_query = st.selectbox(
            "Select Query",
            options=list(query_options.keys()),
            format_func=lambda x: query_options[x]
        )
        
        # Display query details
        if selected_query:
            query_info = next((q for q in queries if q["id"] == selected_query), None)
            if query_info:
                st.write("**Description:**", query_info.get("description", "No description"))
                st.write("**Category:**", query_info.get("category", "N/A"))
                
                # Display parameters if any
                if "parameters" in query_info and query_info["parameters"]:
                    st.write("**Parameters:**")
                    for param, default in query_info["parameters"].items():
                        st.text_input(f"{param} (default: {default})", value=default)
        
        # Engine selection
        engines = get_engines()
        selected_engine = st.selectbox("Execution Engine", engines)
        
        # Scale factor
        scale_factor = st.slider(
            "Scale Factor",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="TPC-H scale factor for the dataset"
        )
        
        # Run benchmark button
        if st.button("Run Benchmark", type="primary"):
            with st.spinner("Running benchmark..."):
                benchmark_id = run_benchmark(
                    query_id=selected_query,
                    engine=selected_engine,
                    scale_factor=scale_factor
                )
                
                if benchmark_id:
                    # Store benchmark ID in session state
                    if "benchmark_history" not in st.session_state:
                        st.session_state.benchmark_history = []
                    st.session_state.benchmark_history.insert(0, benchmark_id)
                    st.session_state.current_benchmark = benchmark_id
                    st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display current benchmark results
        if "current_benchmark" in st.session_state:
            display_benchmark_results(st.session_state.current_benchmark)
        else:
            st.info("Configure and run a benchmark to see results.")
    
    with col2:
        # Display benchmark history
        if "benchmark_history" in st.session_state and st.session_state.benchmark_history:
            st.write("### Recent Benchmarks")
            for bid in st.session_state.benchmark_history[:5]:  # Show last 5 benchmarks
                if st.button(f"Benchmark: {bid[:8]}...", key=f"btn_{bid}"):
                    st.session_state.current_benchmark = bid
                    st.rerun()

def display_benchmark_results(benchmark_id: str) -> None:
    """Display detailed results for a benchmark."""
    result = get_benchmark_status(benchmark_id)
    if not result:
        st.error(f"Could not load results for benchmark {benchmark_id}")
        return
    
    # Display benchmark header
    status_emoji = "✅" if result.get("status") == "completed" else "⏳"
    st.header(f"{status_emoji} Benchmark Results")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Query", result.get("query_id", "N/A"))
    with col2:
        st.metric("Engine", result.get("engine", "N/A").upper())
    with col3:
        st.metric("Status", result.get("status", "unknown").capitalize())
    
    # Execution time
    if "execution_time" in result and result["execution_time"] is not None:
        st.metric("Execution Time", f"{result['execution_time']:.4f} seconds")
    
    # Display results based on engine type
    if result.get("engine") == "hybrid":
        # Hybrid mode results
        st.write("### Hybrid Execution Results")
        
        # Comparison metrics
        if "comparison_metrics" in result:
            display_comparison_metrics(result["comparison_metrics"])
        
        # Spark results
        if "spark_result" in result:
            with st.expander("Spark Execution Details", expanded=False):
                display_engine_results(result["spark_result"])
        
        # DuckDB results
        if "duckdb_result" in result:
            with st.expander("DuckDB Execution Details", expanded=False):
                display_engine_results(result["duckdb_result"])
        
        # Validation results
        if "validation_result" in result:
            display_validation_results(result["validation_result"])
    else:
        # Single engine results
        engine_result = result.get("spark_result") or result.get("duckdb_result")
        if engine_result:
            display_engine_results(engine_result)
        
        # Show validation results if available
        if "validation_result" in result:
            display_validation_results(result["validation_result"])
    
    # Show raw JSON for debugging
    with st.expander("Raw Results"):
        st.json(result)

def display_engine_results(engine_result: Dict) -> None:
    """Display results for a specific engine."""
    if not engine_result:
        return
    
    # Basic metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Execution Time", f"{engine_result.get('execution_time', 0):.4f} seconds")
    with col2:
        st.metric("Rows Returned", engine_result.get("rows_returned", 0))
    with col3:
        status = "success" if engine_result.get("success", False) else "error"
        st.metric("Status", status.capitalize())
    
    # Display error if any
    if "error" in engine_result and engine_result["error"]:
        st.error(f"Error: {engine_result['error']}")
    
    # Display system metrics
    if "system_metrics" in engine_result:
        display_system_metrics(engine_result["system_metrics"])
    
    # Display detailed metrics
    if "metrics" in engine_result:
        display_metrics(engine_result["metrics"])
    
    # Display query plan if available
    if "query_plan" in engine_result and engine_result["query_plan"]:
        with st.expander("Query Plan"):
            st.code(engine_result["query_plan"], language="sql")

if __name__ == "__main__":
    main()
