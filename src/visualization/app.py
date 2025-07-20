import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os
from typing import List, Dict, Any

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Set page config
st.set_page_config(
    page_title="Big Data Benchmarking Engine",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; }
    .sub-header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
    .metric-card { 
        background-color: #f8f9fa; 
        border-radius: 5px; 
        padding: 15px; 
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Big Data Benchmarking Engine</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuration")
    
    engine = st.radio(
        "Select Engine",
        ["spark", "duckdb", "hybrid"],
        index=0,
        help="Select the query execution engine"
    )
    
    scale_factor = st.slider(
        "Scale Factor",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="TPC-H scale factor for data generation"
    )
    
    try:
        response = requests.get(f"{API_URL}/benchmark/queries")
        queries = response.json().get("queries", [])
        query_options = {q["id"]: q["name"] for q in queries}
        selected_query = st.selectbox(
            "Select Query",
            options=list(query_options.keys()),
            format_func=lambda x: f"{x}: {query_options[x]}",
            help="Select a TPC-H query to benchmark"
        )
    except Exception as e:
        st.error(f"Failed to load queries: {str(e)}")
        st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Query Execution")
    
    if st.button("Run Benchmark", type="primary"):
        with st.spinner("Running benchmark..."):
            try:
                response = requests.post(
                    f"{API_URL}/benchmark/run",
                    json={
                        "query": selected_query,
                        "engine": engine,
                        "scale_factor": scale_factor
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                if "benchmark_results" not in st.session_state:
                    st.session_state.benchmark_results = []
                st.session_state.benchmark_results.append(result)
                
                st.success("Benchmark completed successfully!")
                
            except Exception as e:
                st.error(f"Error running benchmark: {str(e)}")
    
    if "benchmark_results" in st.session_state and st.session_state.benchmark_results:
        st.markdown("### Benchmark Results")
        
        df = pd.DataFrame(st.session_state.benchmark_results)
        
        for _, row in df.iterrows():
            with st.expander(f"{row['query']} - {row['engine']} - {row['execution_time']:.2f}s"):
                st.json(row.to_dict())

with col2:
    st.markdown("### Performance Metrics")
    
    if "benchmark_results" in st.session_state and st.session_state.benchmark_results:
        df = pd.DataFrame(st.session_state.benchmark_results)
        
        if not df.empty:
            fig = px.bar(
                df,
                x="query",
                y="execution_time",
                color="engine",
                title="Query Execution Time by Engine",
                labels={"execution_time": "Execution Time (s)", "query": "Query"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Total Benchmarks Run", len(df))
            fastest = df.loc[df['execution_time'].idxmin()]
            st.metric("Fastest Execution", f"{fastest['execution_time']:.2f}s ({fastest['engine']} - {fastest['query']})")

st.markdown("---")
st.markdown("""
### About
This application allows you to benchmark TPC-H queries across different execution engines (Spark, DuckDB, and Hybrid).

**Instructions:**
1. Select the execution engine and scale factor
2. Choose a TPC-H query to benchmark
3. Click "Run Benchmark" to execute the query
4. View and compare performance metrics
""")
